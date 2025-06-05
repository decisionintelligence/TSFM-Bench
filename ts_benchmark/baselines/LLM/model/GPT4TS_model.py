import torch
from torch import nn

import sys
sys.path.insert(0,"ts_benchmark/baselines/LLM/submodules/GPT4TS")

from ts_benchmark.baselines.LLM.submodules.GPT4TS import GPT4TS

class GPT4TSModel(nn.Module):
    def __init__(
        self,
        config,
        device
    ):
        super().__init__()
        # config.pred_len = config.horizon
        self.model = GPT4TS.GPT4TS(config, device)
       
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, device):        
        output = self.model(x_enc, device)
        return output
