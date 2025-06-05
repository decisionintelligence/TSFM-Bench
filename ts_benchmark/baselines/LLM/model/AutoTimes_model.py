from einops import rearrange
import torch
from torch import nn
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

import sys
sys.path.insert(0,"ts_benchmark/baselines/LLM/submodules/AutoTimes")

from ts_benchmark.baselines.LLM.submodules.AutoTimes.models import AutoTimes_Gpt2

class PtModel(nn.Module):
    def __init__(self, device):
        super(PtModel, self).__init__()
        self.device = device
        print(self.device)
        
        self.gpt2_config = GPT2Config.from_pretrained('ts_benchmark/baselines/LLM/checkpoints/gpt2')
        self.llm_model = GPT2Model.from_pretrained(
                        'ts_benchmark/baselines/LLM/checkpoints/gpt2',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.gpt2_config,
                    )
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained(
                        'ts_benchmark/baselines/LLM/checkpoints/gpt2',
                        trust_remote_code=True,
                        local_files_only=True
                    )
        # self.llama_tokenizer = LlamaTokenizer.from_pretrained(configs.llm_ckp_dir)
        self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
        self.vocab_size = self.gpt_tokenizer.vocab_size
        self.hidden_dim_of_llama = 768
        
        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False
            
        
        self.llm_model.to(self.device)

    def tokenizer(self, x):
        output = self.gpt_tokenizer(x, padding=True, truncation=True, return_tensors="pt")['input_ids'].to(self.device)
        result = self.llm_model.get_input_embeddings()(output)
        return result   
    
    def forecast(self, x_mark_enc):        
        # x_mark_enc: [bs x T x hidden_dim_of_llama]
        x_mark_encs = []
        # for i in range(len(x_mark_enc)):

        x_enc = self.tokenizer(x_mark_enc)
        # x_mark_encs.append(x_enc)
        # x_mark_enc = torch.cat(x_mark_encs, 0)
        # x_mark_enc = torch.cat([self.tokenizer(x_mark_enc[i]) for i in range(len(x_mark_enc))], 0)
        text_outputs = self.llm_model(inputs_embeds=x_enc)[0]
        text_outputs = text_outputs[:, -1, :]
        return text_outputs
    
    def forward(self, x_mark_enc):
        return self.forecast(x_mark_enc)

class AutoTimesModel(nn.Module):
    def __init__(
        self,
        config,
        device
    ):
        super().__init__()
        self.pred_len = config.horizon
        self.token_len = config.token_len
        # self.get_pt_model = PtModel(device)
        self.model = AutoTimes_Gpt2.Model(config)

        if not config.use_p:
            for param in self.model.parameters():
                param.data.uniform_(-0.02, 0.02)
       
    def forward(self, x_enc, x_mark_enc, x_mark_dec):     
        # if is_test:
        #     output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec, device)
            
        # B, _, K = x_enc.shape
        # x_enc = rearrange(x_enc, 'b l k -> (b k) l 1')
        inference_steps = self.pred_len // self.token_len
        dis = self.pred_len - inference_steps * self.token_len
        if dis != 0:
            inference_steps += 1

        pred_y = []
        for j in range(inference_steps):
            if len(pred_y) != 0:
                x_enc = torch.cat([x_enc[:, self.token_len:, :], pred_y[-1]], dim=1)
                tmp = x_mark_dec[:, j-1:j, :]
                x_mark_enc = torch.cat([x_mark_enc[:, 1:, :], tmp], dim=1)
            
            # outputs = self.model(x_enc, x_mark_enc, None, x_mark_dec)
            outputs = []
            for feat_id in range(x_enc.shape[2]):
                input = x_enc[:, :, feat_id].unsqueeze(2)
                output = self.model(input, x_mark_enc, None, x_mark_dec)
                outputs.append(output)
            outputs = torch.cat(outputs, dim=2)

            pred_y.append(outputs[:, -self.token_len:, :])

        pred_y = torch.cat(pred_y, dim=1)

        if dis != 0:
            pred_y = pred_y[:, :-(self.token_len - dis), :]
        pred_y = pred_y[:, -self.pred_len:, :]

        return pred_y
