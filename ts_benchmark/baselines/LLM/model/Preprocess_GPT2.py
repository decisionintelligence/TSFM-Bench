import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.device = configs.gpu
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