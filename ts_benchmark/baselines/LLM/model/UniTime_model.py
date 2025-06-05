import sys
sys.path.insert(0, "ts_benchmark/baselines/LLM/submodules/UniTime")

from einops import rearrange
import torch
from torch import nn
import json

from ts_benchmark.baselines.LLM.submodules.UniTime import unitime

class UniTimeModel(nn.Module):
    def __init__(
        self,
        config,
        device
    ):
        super().__init__()
        self.device = device
        self.mask_rate = config.mask_rate

        self.config = config
        self.model = unitime.UniTime(config)

        self.data_id = config.dataset + '_' + str(config.seq_len) + '_' + str(config.pred_len)
        

        with open(config.instruct_path, 'r') as f:
            instruct_list = json.load(f)
            self.instruct = instruct_list[config.dataset]

        self.info = [self.data_id, config.seq_len, config.stride, self.instruct]
       
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, device): 

        b, t, n = x_enc.shape
        mask = torch.rand((b, t, n)).to(self.device)
        mask[mask < self.mask_rate] = 0  # masked
        mask[mask >= self.mask_rate] = 1  # remained
        inp = x_enc.masked_fill(mask == 0, 0)
        output = self.model(self.info, inp, mask)
        return output
