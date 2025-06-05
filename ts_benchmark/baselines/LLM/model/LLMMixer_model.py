import json
import torch
from torch import nn

import sys
sys.path.insert(0,"ts_benchmark/baselines/LLM/submodules/LLMMixer")

from ts_benchmark.baselines.LLM.submodules.LLMMixer.models import LLMMixer

class LLMMixerModel(nn.Module):
    def __init__(
        self,
        config,
        device
    ):
        super().__init__()
        # config.pred_len = config.horizon

        with open(config.instruct_path, 'r') as f:
            instruct_list = json.load(f)
            config.description = instruct_list[config.dataset]
        
        self.model = LLMMixer.Model(config)

       
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, device):        
        output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return output
