# ---------------------------------------------------------------------------------
# Portions of this file are derived from Chronos
# - Source: https://github.com/amazon-science/chronos-forecasting
# - Paper: Chronos: Learning the Language of Time Series
# - License: Apache License 2.0
#
# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------
import sys
sys.path.insert(0, 'ts_benchmark/baselines/pre_train/submodules/chronos')
import torch
from ts_benchmark.baselines.pre_train.submodules.chronos import ChronosBoltPipeline
from einops import rearrange
from torch import nn

class Chronos(nn.Module):
    def __init__(
        self,
        config,
        model_size: str = 'base',
        **kwargs
    ):
        super().__init__(**kwargs)

        
        self.context_length = config.seq_len
        self.prediction_length = config.pred_len
        self.pred_len = self.prediction_length

        # Load Chronos
        self.pipeline = ChronosBoltPipeline.from_pretrained(
            "ts_benchmark/baselines/pre_train/checkpoints/chronos",
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )


    def forward(self, inputs, dec_inp, x_mark_enc, x_mark_dec, device=None, num_samples=None):  

        
        B, _, K = inputs.shape
        inputs = rearrange(inputs, 'b l k -> (b k) l').cpu()
        context = [inputs[i] for i in range(B*K)]
        inner_batch_size = 12 
        forecast_samples = []

        for i in range(0, len(context), inner_batch_size):
            batch_context = context[i:i + inner_batch_size]
            batch_forecast_samples = self.pipeline.predict(
                batch_context,
                prediction_length=self.pred_len,
                limit_prediction_length=False
            )
            forecast_samples.append(batch_forecast_samples)
        
        forecast_samples = torch.cat(forecast_samples, dim=0)
        prob_forecast = rearrange(forecast_samples, '(b k) s l -> b s l k', b=B, k=K)
        return prob_forecast[:, 5, :, :].to(device)