import sys
sys.path.insert(0, "ts_benchmark/baselines/LLM/submodules/S2IP_LLM")

import torch
from torch import nn

from ts_benchmark.baselines.LLM.submodules.S2IP_LLM import S2IPLLM

class S2IPLLMModel(nn.Module):
    def __init__(
        self,
        config,
        device
    ):
        super().__init__()
        self.model = S2IPLLM.Model(config)
       
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, device):        
        output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return output[0]
