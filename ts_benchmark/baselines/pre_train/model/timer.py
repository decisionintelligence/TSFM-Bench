import torch
from torch import nn
from einops import rearrange, repeat
import sys
sys.path.insert(0, 'ts_benchmark/baselines/pre_train/submodules/Timer')
from ts_benchmark.baselines.pre_train.submodules.Timer.timer_backbone import Model

class Timer(nn.Module):
# class Timer(nn.Module):
    def __init__(
        self,
        config,
        **kwargs
    ):
        super().__init__()
        self.context_length = config.seq_len
        self.prediction_length = config.pred_len
        self.target_dim = config.target_dim
        self.freq = config.freq
        self.dataset = config.dataset
        self.config = config
        config.input_token_len = 96
        config.output_token_len = 96
        config.d_model = 1024
        config.d_ff = 2048
        config.n_heads = 8
        config.e_layers = 8
        config.dropout = 0.1
        config.activation = 'relu'
        config.use_norm = True
        config.flash_attention = False
        config.covariate = False
        config.output_attention = False

        self.input_token_len = 96 # fixed by the pre-trained model
        self.output_patch_len = 96 # fixed by the pre-trained model
        self.label_len = config.seq_len - 96

        # self.model = torch.jit.load("ts_benchmark/baselines/pre_train/checkpoints/timer/Timer_67M_UTSD_4G.pt")
        # self.model = torch.load("ts_benchmark/baselines/pre_train/checkpoints/timer/Timer_forecast_1.0.ckpt")
        self.model = Model(config)
        self.model.load_state_dict(torch.load("ts_benchmark/baselines/pre_train/checkpoints/timer/checkpoint.pth"))

        if not config.use_p:
            for param in self.model.parameters():
                param.data.uniform_(-0.02, 0.02)
     
    def forward(self, inputs, dec_inp, x_mark_enc, x_mark_dec, device=None, num_samples=None):        
        
        B, _, K = inputs.shape

        inputs = rearrange(inputs, 'b l k -> (b k) l 1')
        x_mark_enc = repeat(x_mark_enc, 'b l f -> (b k) l f', k=K)
        x_mark_dec = repeat(x_mark_dec, 'b l f -> (b k) l f', k=K)

        dec_inp = torch.zeros_like(inputs[:, -self.prediction_length:, :]).float()
        dec_inp = torch.cat((inputs[:, -self.label_len:, ...], dec_inp), dim=1).float()

        if self.config.is_train == 0:
            inference_steps = self.prediction_length // self.output_patch_len
            dis = self.prediction_length - inference_steps * self.output_patch_len
            if dis != 0:
                inference_steps += 1

            pred_y = []

            for j in range(inference_steps):
                if len(pred_y) != 0:
                    inputs = torch.cat([inputs[:, self.output_patch_len:, :], pred_y[-1]], dim=1)
                    tmp = x_mark_dec[:, j - 1:j, :]
                    x_mark_enc = torch.cat([x_mark_enc[:, 1:, :], tmp], dim=1)

                outputs = self.model(inputs, x_mark_enc, x_mark_dec)
                pred_y.append(outputs[:, -self.output_patch_len:, :])

            pred_y = torch.cat(pred_y, dim=1)
            if dis != 0:
                pred_y = pred_y[:, :-dis, :]
            pred_y = rearrange(pred_y, '(b k) l 1 -> b l k', b=B, k=K)
            pred_y = pred_y[:, :self.prediction_length, :]
        else:
            outputs = self.model(inputs, x_mark_enc, x_mark_dec)
            pred_y = rearrange(outputs, '(b k) l 1 -> b l k', b=B, k=K)
            
        return pred_y
