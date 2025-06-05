import sys
sys.path.insert(0, "ts_benchmark/baselines/LLM/submodules/Time_LLM")

from einops import rearrange
import torch
from torch import nn

from ts_benchmark.baselines.LLM.submodules.Time_LLM.models import TimeLLM

class TimeLLMsModel(nn.Module):
    def __init__(
        self,
        config,
        device
    ):
        super().__init__()
        if config.prompt_domain:
            domains = ['ETT', 'Electricity', 'Traffic', "Solar", 'Weather', 'ILI', 'ZafNoo', 'Exchange', 'NN5', 'Wike2000', 'NASDAQ', 'PEMS08', 'Wind', 'AQShunyi', 'Covid-19', 'CzeLan', 'FRED-MD', 'NYSE']
            for d in domains:
                if d.lower() in config.dataset.lower():
                    domain = d
                    break
            file_path = 'dataset/prompt_bank/' + domain + '.txt'
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            config.content = content
        self.model = TimeLLM.Model(config)
       
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, device):  
        B, T, K = x_enc.shape  
        x_enc = rearrange(x_enc, 'b l k -> (b k) l 1') 
        output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec) 
        output = rearrange(output, '(b k) l 1 -> b l k', b=B, k=K)
        return output
