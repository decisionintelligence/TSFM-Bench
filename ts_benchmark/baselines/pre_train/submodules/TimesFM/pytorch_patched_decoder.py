
import dataclasses
import math
from typing import List, Tuple
import torch
from torch import nn
import torch.nn.functional as F


def _create_quantiles() -> list[float]:
  return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@dataclasses.dataclass
class TimesFMConfig:
  num_layers: int = 20
  num_heads: int = 16
  num_kv_heads: int = 16
  hidden_size: int = 1280
  intermediate_size: int = 1280
  head_dim: int = 80
  rms_norm_eps: float = 1e-6
  patch_len: int = 32
  horizon_len: int = 128
  quantiles: List[float] = dataclasses.field(default_factory=_create_quantiles)
  pad_val: float = 1123581321.0
  tolerance: float = 1e-6
  dtype: str = "bfloat32"
  use_positional_embedding: bool = True


def _masked_mean_std(
    inputs: torch.Tensor,
    padding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

  pad_sum = torch.sum(1 - padding, dim=2)

  def _get_patch_index(arr: torch.Tensor):
    indices = torch.argmax((arr >= 3).to(torch.int32), dim=1)
    row_sum = (arr >= 3).to(torch.int32).sum(dim=1)
    return torch.where(row_sum == 0, arr.shape[1] - 1, indices)

  patch_indices = _get_patch_index(pad_sum)
  bidxs = torch.arange(inputs.shape[0])

  arr = inputs[bidxs, patch_indices, :]
  pad = padding[bidxs, patch_indices, :]

  mask = 1 - pad
  
  num_valid_elements = torch.sum(mask, dim=1)
  num_valid_elements = torch.where(
      num_valid_elements == 0,
      torch.tensor(1,
                   dtype=num_valid_elements.dtype,
                   device=num_valid_elements.device),
      num_valid_elements,
  )

  masked_sum = torch.sum(arr * mask, dim=1)
  masked_squared_sum = torch.sum((arr * mask)**2, dim=1)

  masked_mean = masked_sum / num_valid_elements
  masked_var = masked_squared_sum / num_valid_elements - masked_mean**2
  masked_var = torch.where(
      masked_var < 0.0,
      torch.tensor(0.0, dtype=masked_var.dtype, device=masked_var.device),
      masked_var,
  )
  masked_std = torch.sqrt(masked_var)

  return masked_mean, masked_std


def _shift_padded_seq(mask: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:

  batch_size, num_seq, feature_dim = seq.shape

  new_mask: torch.BoolTensor = mask == 0

  indices = new_mask.to(torch.int32).argmax(dim=1)

  indices[~new_mask.any(dim=1)] = -1

  idx_range = (torch.arange(num_seq).to(
      seq.device).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1,
                                                    feature_dim))

  shifted_idx = (idx_range - indices[:, None, None]) % num_seq

  shifted_seq = seq.gather(1, shifted_idx)

  return shifted_seq


def get_large_negative_number(dtype: torch.dtype) -> torch.Tensor:

  if dtype.is_floating_point:
    dtype_max = torch.finfo(dtype).max
  else:
    dtype_max = torch.iinfo(dtype).max
  return torch.tensor(-0.7 * dtype_max, dtype=dtype)


def apply_mask_to_logits(logits: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:


  min_value = get_large_negative_number(logits.dtype)

  return torch.where((mask >= min_value * 0.5), logits, min_value)


def convert_paddings_to_mask(
    paddings: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:

  attention_mask = paddings.detach().clone()
  attention_mask = attention_mask[:, None, None, :]  
  attention_mask *= get_large_negative_number(dtype)
  return attention_mask


def causal_mask(input_t: torch.Tensor) -> torch.Tensor:

  assert input_t.dtype.is_floating_point, input_t.dtype
  large_negative_number = get_large_negative_number(input_t.dtype)
  t = input_t.shape[1]
  col_idx = torch.arange(t).unsqueeze(0).repeat(t, 1)
  row_idx = torch.arange(t).unsqueeze(1).repeat(1, t)
  mask = (row_idx < col_idx).to(input_t.dtype) * large_negative_number
  return (mask.unsqueeze(0).unsqueeze(0).to(input_t.device)
         ) 


def merge_masks(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:


  def expand_t(key_mask):
    query_mask = key_mask.transpose(-1, -2)  
    return torch.minimum(query_mask, key_mask)

  if a.shape[2] != b.shape[2]:
    if a.shape[2] == 1:
      a = expand_t(a)
    else:
      assert b.shape[2] == 1
      b = expand_t(b)

  assert a.shape[1:] == b.shape[1:], f"a.shape={a.shape}, b.shape={b.shape}."
  return torch.minimum(a, b)  


class ResidualBlock(nn.Module):

  def __init__(
      self,
      input_dims,
      hidden_dims,
      output_dims,
  ):
    super(ResidualBlock, self).__init__()
    self.input_dims = input_dims
    self.hidden_dims = hidden_dims
    self.output_dims = output_dims

    self.hidden_layer = nn.Sequential(
        nn.Linear(input_dims, hidden_dims),
        nn.SiLU(),
    )

    self.output_layer = nn.Linear(hidden_dims, output_dims)
    self.residual_layer = nn.Linear(input_dims, output_dims)

  def forward(self, x):
    hidden = self.hidden_layer(x)
    output = self.output_layer(hidden)
    residual = self.residual_layer(x)
    return output + residual


class RMSNorm(torch.nn.Module):


  def __init__(
      self,
      dim: int,
      eps: float = 1e-6,
      add_unit_offset: bool = False,
  ):
    super().__init__()
    self.eps = eps
    self.add_unit_offset = add_unit_offset
    self.weight = nn.Parameter(torch.zeros(dim))

  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x):
    output = self._norm(x.float())
    if self.add_unit_offset:
      output = output * (1 + self.weight.float())
    else:
      output = output * self.weight.float()
    return output.type_as(x)


class TransformerMLP(nn.Module):


  def __init__(
      self,
      hidden_size: int,
      intermediate_size: int,
  ):
    super().__init__()
    self.gate_proj = nn.Linear(hidden_size, intermediate_size)
    self.down_proj = nn.Linear(intermediate_size, hidden_size)
    self.layer_norm = nn.LayerNorm(normalized_shape=hidden_size, eps=1e-6)

  def forward(self, x, paddings=None):
    gate_inp = self.layer_norm(x)
    gate = self.gate_proj(gate_inp)
    gate = F.relu(gate)
    outputs = self.down_proj(gate)
    if paddings is not None:
      outputs = outputs * (1.0 - paddings[:, :, None])
    return outputs + x


class TimesFMAttention(nn.Module):


  def __init__(
      self,
      hidden_size: int,
      num_heads: int,
      num_kv_heads: int,
      head_dim: int,
  ):
    super().__init__()

    self.num_heads = num_heads
    self.num_kv_heads = num_kv_heads

    assert self.num_heads % self.num_kv_heads == 0
    self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    self.hidden_size = hidden_size
    self.head_dim = head_dim

    self.q_size = self.num_heads * self.head_dim
    self.kv_size = self.num_kv_heads * self.head_dim
    self.scaling = nn.Parameter(
        torch.empty((self.head_dim,), dtype=torch.float32),)

    self.qkv_proj = nn.Linear(
        self.hidden_size,
        (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
    )
    self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

  def _per_dim_scaling(self, query: torch.Tensor) -> torch.Tensor:

    r_softplus_0 = 1.442695041
    softplus_func = torch.nn.Softplus()
    scale = r_softplus_0 / math.sqrt(self.head_dim)
    scale = scale * softplus_func(self.scaling)
    return query * scale[None, None, None, :]

  def forward(
      self,
      hidden_states: torch.Tensor,
      mask: torch.Tensor,
      kv_write_indices: torch.Tensor | None = None,
      kv_cache: Tuple[torch.Tensor, torch.Tensor] | None = None,
  ) -> torch.Tensor:
    hidden_states_shape = hidden_states.shape
    assert len(hidden_states_shape) == 3

    batch_size, input_len, _ = hidden_states_shape

    qkv = self.qkv_proj(hidden_states)
    xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
    xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
    xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)
    xq = self._per_dim_scaling(xq)

    if kv_cache is not None and kv_write_indices is not None:
      k_cache, v_cache = kv_cache
      k_cache.index_copy_(1, kv_write_indices, xk)
      v_cache.index_copy_(1, kv_write_indices, xv)

      key = k_cache
      value = v_cache
    else:
      key = xk
      value = xv
    if self.num_kv_heads != self.num_heads:
      key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
      value = torch.repeat_interleave(value, self.num_queries_per_kv, dim=2)

    q = xq.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    scores = torch.matmul(q, k.transpose(2, 3))
    scores = scores + mask
    scores = F.softmax(scores.float(), dim=-1).type_as(q)

    output = torch.matmul(scores, v)

    output = output.transpose(1, 2).contiguous().view(batch_size, input_len, -1)
    output = self.o_proj(output)
    return scores, output


class TimesFMDecoderLayer(nn.Module):


  def __init__(
      self,
      hidden_size: int,
      intermediate_size: int,
      num_heads: int,
      num_kv_heads: int,
      head_dim: int,
      rms_norm_eps: float = 1e-6,
  ):
    super().__init__()
    self.self_attn = TimesFMAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    self.mlp = TransformerMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

  def forward(
      self,
      hidden_states: torch.Tensor,
      mask: torch.Tensor,
      paddings: torch.Tensor,
      kv_write_indices: torch.Tensor | None = None,
      kv_cache: Tuple[torch.Tensor, torch.Tensor] | None = None,
  ) -> torch.Tensor:

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    scores, hidden_states = self.self_attn(
        hidden_states=hidden_states,
        mask=mask,
        kv_write_indices=kv_write_indices,
        kv_cache=kv_cache,
    )
    hidden_states = residual + hidden_states


    hidden_states = self.mlp(hidden_states, paddings=paddings)

    return scores, hidden_states


class StackedDecoder(nn.Module):

  def __init__(
      self,
      hidden_size: int,
      intermediate_size: int,
      num_heads: int,
      num_kv_heads: int,
      head_dim: int,
      num_layers: int,
      rms_norm_eps: float = 1e-6,
  ):
    super().__init__()

    self.layers = nn.ModuleList()
    for _ in range(num_layers):
      self.layers.append(
          TimesFMDecoderLayer(
              hidden_size=hidden_size,
              intermediate_size=intermediate_size,
              num_heads=num_heads,
              num_kv_heads=num_kv_heads,
              head_dim=head_dim,
              rms_norm_eps=rms_norm_eps,
          ))

  def forward(
      self,
      hidden_states: torch.Tensor,
      paddings: torch.Tensor,
      kv_write_indices: torch.Tensor | None = None,
      kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] | None = None,
  ) -> torch.Tensor:
    padding_mask = convert_paddings_to_mask(paddings, hidden_states.dtype)
    atten_mask = causal_mask(hidden_states)
    mask = merge_masks(padding_mask, atten_mask)
    for i in range(len(self.layers)):
      layer = self.layers[i]
      kv_cache = kv_caches[i] if kv_caches is not None else None
      _, hidden_states = layer(
          hidden_states=hidden_states,
          mask=mask,
          paddings=paddings,
          kv_write_indices=kv_write_indices,
          kv_cache=kv_cache,
      )
    return hidden_states


class PositionalEmbedding(torch.nn.Module):


  def __init__(
      self,
      embedding_dims: int,
      min_timescale: int = 1,
      max_timescale: int = 10_000,
  ) -> None:
    super().__init__()
    self.min_timescale = min_timescale
    self.max_timescale = max_timescale
    self.embedding_dims = embedding_dims

  def forward(self, seq_length=None, position=None):

    if position is None:
      assert seq_length is not None

      position = torch.arange(seq_length, dtype=torch.float32).unsqueeze(0)
    else:
      assert position.ndim == 2, position.shape

    num_timescales = self.embedding_dims // 2
    log_timescale_increment = math.log(
        float(self.max_timescale) / float(self.min_timescale)) / max(
            num_timescales - 1, 1)
    inv_timescales = self.min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float32) *
        -log_timescale_increment)
    scaled_time = position.unsqueeze(2) * inv_timescales.unsqueeze(0).unsqueeze(
        0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)

    signal = F.pad(signal, (0, 0, 0, self.embedding_dims % 2))
    return signal


class PatchedTimeSeriesDecoder(nn.Module):


  def __init__(self, config: TimesFMConfig):
    super().__init__()
    self.config = config
    self.input_ff_layer = ResidualBlock(
        input_dims=2 * config.patch_len,
        output_dims=config.hidden_size,
        hidden_dims=config.intermediate_size,
    )
    self.freq_emb = nn.Embedding(num_embeddings=3,
                                 embedding_dim=config.hidden_size)
    self.horizon_ff_layer = ResidualBlock(
        input_dims=config.hidden_size,
        output_dims=config.horizon_len * (1 + len(config.quantiles)),
        hidden_dims=config.intermediate_size,
    )
    self.stacked_transformer = StackedDecoder(
        hidden_size=self.config.hidden_size,
        intermediate_size=self.config.intermediate_size,
        num_heads=self.config.num_heads,
        num_kv_heads=self.config.num_kv_heads,
        head_dim=self.config.head_dim,
        num_layers=self.config.num_layers,
        rms_norm_eps=self.config.rms_norm_eps,
    )
    if self.config.use_positional_embedding:
      self.position_emb = PositionalEmbedding(self.config.hidden_size)


  def _forward_transform(
      self, inputs: torch.Tensor, patched_pads: torch.Tensor
  ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

    mu, sigma = _masked_mean_std(inputs, patched_pads)
    sigma = torch.where(
        sigma < self.config.tolerance,
        torch.tensor(1.0, dtype=sigma.dtype, device=sigma.device),
        sigma,
    )

    outputs = (inputs - mu[:, None, None]) / sigma[:, None, None]
    outputs = torch.where(
        torch.abs(inputs - self.config.pad_val) < self.config.tolerance,
        torch.tensor(self.config.pad_val,
                     dtype=outputs.dtype,
                     device=outputs.device),
        outputs,
    )
    return outputs, (mu, sigma)

  def _reverse_transform(
      self, outputs: torch.Tensor, stats: tuple[torch.Tensor,
                                                torch.Tensor]) -> torch.Tensor:
    mu, sigma = stats
    return outputs * sigma[:, None, None, None] + mu[:, None, None, None]

  def _preprocess_input(
      self,
      input_ts: torch.Tensor,
      input_padding: torch.Tensor,
  ) -> tuple[
      torch.Tensor,
      torch.Tensor,
      tuple[torch.Tensor, torch.Tensor] | None,
      torch.Tensor,
  ]:

    bsize = input_ts.shape[0]
    patched_inputs = input_ts.view(bsize, -1, self.config.patch_len)
    patched_pads = input_padding.view(bsize, -1, self.config.patch_len)

    patched_inputs = torch.where(
        torch.abs(patched_pads - 1.0) < self.config.tolerance,
        torch.tensor(0.0,
                     dtype=patched_inputs.dtype,
                     device=patched_inputs.device),
        patched_inputs,
    )
    patched_pads = torch.where(
        torch.abs(patched_inputs - self.config.pad_val) < self.config.tolerance,
        torch.tensor(1.0, dtype=patched_pads.dtype, device=patched_pads.device),
        patched_pads,
    )
    patched_inputs, stats = self._forward_transform(patched_inputs,
                                                    patched_pads)

    patched_inputs = patched_inputs * (1.0 - patched_pads)
    concat_inputs = torch.cat([patched_inputs, patched_pads], dim=-1)
    model_input = self.input_ff_layer(concat_inputs)

    patched_padding = torch.min(patched_pads,
                                dim=-1)[0]
    if self.config.use_positional_embedding:
      pos_emb = self.position_emb(model_input.shape[1]).to(model_input.device)
      pos_emb = torch.concat([pos_emb] * model_input.shape[0], dim=0)
      pos_emb = _shift_padded_seq(patched_padding, pos_emb)
      model_input += pos_emb

    return model_input, patched_padding, stats, patched_inputs

  def _postprocess_output(
      self,
      model_output: torch.Tensor,
      num_outputs: int,
      stats: tuple[torch.Tensor, torch.Tensor],
  ) -> torch.Tensor:

    output_ts = self.horizon_ff_layer(model_output)

    b, n, _ = output_ts.shape
    output_ts = output_ts.view(b, n, self.config.horizon_len, num_outputs)

    return self._reverse_transform(output_ts, stats)

  def forward(
      self,
      input_ts: torch.Tensor,
      input_padding: torch.LongTensor,
      freq: torch.Tensor,
  ) -> torch.Tensor:
    num_outputs = len(self.config.quantiles) + 1
    model_input, patched_padding, stats, _ = self._preprocess_input(
        input_ts=input_ts,
        input_padding=input_padding,
    )
    f_emb = self.freq_emb(freq)  # B x 1 x D
    model_input += f_emb
    model_output = self.stacked_transformer(model_input, patched_padding)

    output_ts = self._postprocess_output(model_output, num_outputs, stats)
    return output_ts

  def decode(
      self,
      input_ts: torch.Tensor,
      paddings: torch.Tensor,
      freq: torch.LongTensor,
      horizon_len: int,
      output_patch_len: int | None = None,
      max_len: int = 512,
      return_forecast_on_context: bool = False,
  ) -> tuple[torch.Tensor, torch.Tensor]:

    final_out = input_ts
    context_len = final_out.shape[1]
    full_outputs = []
    if paddings.shape[1] != final_out.shape[1] + horizon_len:
      raise ValueError(
          "Length of paddings must match length of input + horizon_len:"
          f" {paddings.shape[1]} != {final_out.shape[1]} + {horizon_len}")
    if output_patch_len is None:
      output_patch_len = self.config.horizon_len
    num_decode_patches = (horizon_len + output_patch_len -
                          1) // output_patch_len
    for step_index in range(num_decode_patches):
      current_padding = paddings[:, 0:final_out.shape[1]]
      input_ts = final_out[:, -max_len:]
      input_padding = current_padding[:, -max_len:]
      fprop_outputs = self(input_ts, input_padding, freq)
      if return_forecast_on_context and step_index == 0:
        new_full_ts = fprop_outputs[:, :-1, :self.config.patch_len, :]
        new_full_ts = fprop_outputs.view(new_full_ts.size(0), -1,
                                         new_full_ts.size(3))

        full_outputs.append(new_full_ts)

      new_ts = fprop_outputs[:, -1, :output_patch_len, 0]
      new_full_ts = fprop_outputs[:, -1, :output_patch_len, :]
      full_outputs.append(new_full_ts)
      final_out = torch.concatenate([final_out, new_ts], axis=-1)

    if return_forecast_on_context:
      full_outputs = torch.concatenate(
          full_outputs,
          axis=1)[:, :(context_len - self.config.patch_len + horizon_len), :]
    else:
      full_outputs = torch.concatenate(full_outputs, axis=1)[:,
                                                             0:horizon_len, :]

    return (full_outputs[:, :, 0], full_outputs)
