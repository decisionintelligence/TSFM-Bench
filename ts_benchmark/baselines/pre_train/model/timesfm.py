import logging
from os import path
import time
from typing import Any, Literal, OrderedDict, Sequence

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

import ts_benchmark.baselines.pre_train.submodules.TimesFM.pytorch_patched_decoder as ppd


class TimesFmTorch(nn.Module):
  def __init__(
        self,
        config
        ):
    super().__init__()
    self.model_config = ppd.TimesFMConfig(
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        hidden_size=config.model_dims,
        intermediate_size=config.model_dims,
        patch_len=config.input_patch_len,
        horizon_len=config.output_patch_len,
        head_dim=config.model_dims // config.num_heads,
        quantiles=config.quantiles,
    )

    self.context_len = config.context_len
    self.horizon_len = config.horizon_len
    self.num_cores = 1
    self.per_core_batch_size = config.per_core_batch_size
    self.global_batch_size = config.per_core_batch_size
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.model = ppd.PatchedTimeSeriesDecoder(self.model_config)
    loaded_checkpoint = torch.load(config.checkpoint, weights_only=True)

    self.model.load_state_dict(loaded_checkpoint)

  def preprocess(self, inputs: Sequence[torch.Tensor], freq: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:

    input_ts, input_padding, inp_freq = [], [], []

    pmap_pad = ((len(inputs) - 1) // self.global_batch_size + 1) * self.global_batch_size - len(inputs) 

    for i, ts in enumerate(inputs):
        input_len = ts.shape[0]
        padding = torch.zeros(input_len + self.horizon_len, device=ts.device)  

        if input_len < self.context_len:
            num_front_pad = self.context_len - input_len
            ts = torch.cat([torch.zeros(num_front_pad, device=ts.device), ts], dim=0)
            padding = torch.cat([torch.ones(num_front_pad, device=ts.device), padding], dim=0)
        elif input_len > self.context_len:
            ts = ts[-self.context_len:]
            padding = padding[-(self.context_len + self.horizon_len):]

        input_ts.append(ts)
        input_padding.append(padding)
        inp_freq.append(freq[i])

    for _ in range(pmap_pad):
        input_ts.append(input_ts[-1])
        input_padding.append(input_padding[-1])
        inp_freq.append(inp_freq[-1])

    return (
        torch.stack(input_ts, dim=0, out=torch.empty(0, device=inputs[0].device)),  
        torch.stack(input_padding, dim=0, out=torch.empty(0, device=inputs[0].device)), 
        torch.tensor(inp_freq, device=inputs[0].device).reshape(-1, 1),
        pmap_pad,
    )

  def forecast(
      self,
      inputs,
      freq: Sequence[int] | None = None,
    #   window_size: int | None = None,
    #   forecast_context_len: int | None = None,
      return_forecast_on_context: bool = False,
    #   truncate_negative: bool = False,
  ):
  
    inputs = [ts[-self.context_len:] for ts in inputs]
    
    input_ts, input_padding, inp_freq, pmap_pad = self.preprocess(inputs, freq)

    mean_outputs = []
    full_outputs = []
    assert input_ts.shape[0] % self.global_batch_size == 0
    for i in range(input_ts.shape[0] // self.global_batch_size):
      
      input_ts_in = input_ts[i * self.global_batch_size:(i + 1) * self.global_batch_size]
      input_padding_in = input_padding[i * self.global_batch_size:(i + 1) * self.global_batch_size]
      inp_freq_in = inp_freq[i * self.global_batch_size:(i + 1) * self.global_batch_size].long()

      mean_output, full_output = self.model.decode(
          input_ts=input_ts_in,
          paddings=input_padding_in,
          freq=inp_freq_in,
          horizon_len=self.horizon_len,
          return_forecast_on_context=return_forecast_on_context,
      )

      mean_outputs.append(mean_output)
      full_outputs.append(full_output)

    mean_outputs = torch.cat(mean_outputs, dim=0)
    full_outputs = torch.cat(full_outputs, dim=0)
    
    if pmap_pad > 0:
      mean_outputs = mean_outputs[:-pmap_pad, ...]
      full_outputs = full_outputs[:-pmap_pad, ...]

    return mean_outputs, full_outputs

class TimesFM(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()

        class Configs:
           def __init__(self,
                        context_len=512,
                        horizon_len=128,
                        checkpoint='',
                        
                        # backend="gpu"
                        ):
              self.context_len: int = context_len
              self.horizon_len: int = horizon_len
              self.checkpoint: str = checkpoint
              self.input_patch_len: int = 32
              self.output_patch_len: int = 128
              self.num_layers: int = 20
              self.num_heads: int = 16
              self.model_dims: int = 1280
              self.per_core_batch_size: int = 32
              self.quantiles: Sequence[float] | None = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        
        model_config = Configs(
           context_len=config.seq_len,
           horizon_len=config.pred_len,
           checkpoint="ts_benchmark/baselines/pre_train/checkpoints/timesfm/torch_model.ckpt",
        )
        self.model = TimesFmTorch(model_config)
        
        freq_dict = {'h': 0, 't': 0, 'd': 0, 'b': 0, 'u': 0, 'w': 1, 'm': 1, 'q': 2, 'y': 2}
        freq = config.freq.lower()
        
        if freq in freq_dict:
            self.freq_int = freq_dict[freq]
        else:
            self.freq_int = 0

    def forward(self, inputs, dec_inp=None, x_mark_enc=None, x_mark_dec=None, device=None, num_samples=None):        
        B, _, K = inputs.shape
        inputs = rearrange(inputs, 'b l k -> (b k) l')

        frequency_input = [self.freq_int] * inputs.shape[0]
        
        _, out = self.model.forecast(inputs, freq=frequency_input)
        point_forecast = out[:, :, 5]
        point_forecast = rearrange(point_forecast, '(b k) l -> b l k', b=B,k=K)

        return point_forecast