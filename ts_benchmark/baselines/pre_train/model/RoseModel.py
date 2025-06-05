
from einops import rearrange
from torch import nn
import torch
from torch.optim.lr_scheduler import _LRScheduler

from ts_benchmark.baselines.pre_train.submodules.ROSE.ROSE_lowrank import ROSE as ROSE_finetune
from ts_benchmark.baselines.pre_train.submodules.ROSE.ROSE_predict import ROSE as ROSE_predict

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

class RoseModel(nn.Module):

    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__()
        # self.context_length = config.seq_len
        self.prediction_length = config.pred_len
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.n_embedding = config.n_embedding
        # get number of patches
        num_patch = (max(config.seq_len, config.patch_len)-config.patch_len) // config.stride + 1    
        print('number of patches:', num_patch)
        
        self.revin = RevIN(num_features=config.enc_in)
    
        if config.is_train:
            print('finetuning')
            self.model = ROSE_finetune(c_in=config.enc_in,
                        target_dim=config.pred_len,
                        patch_len=config.patch_len,
                        stride=config.stride,
                        n_embedding=config.n_embedding,
                        num_patch=num_patch,
                        n_layers=config.n_layers,
                        n_heads=config.n_heads,
                        d_model=config.d_model,
                        shared_embedding=True,
                        d_ff=config.d_ff,                        
                        dropout=config.dropout,
                        head_dropout=config.head_dropout,
                        # norm ='LayerNorm',
                        act='relu',
                        head_type=config.head_type,
                        res_attention=False
                        )    
            ckpt_path = "ts_benchmark/baselines/pre_train/checkpoints/rose/full-shot.pth"
        else:
            print('zero-shot')
            self.model = ROSE_predict(c_in=config.enc_in,
                target_dim=config.pred_len,
                patch_len=config.patch_len,
                stride=config.stride,
                n_embedding=config.n_embedding,
                num_slots=config.num_slots,
                num_patch=num_patch,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
                d_model=config.d_model,
                shared_embedding=True,
                d_ff=config.d_ff,                        
                dropout=config.dropout,
                head_dropout=config.head_dropout,
                norm='BatchNorm',
                act='relu',
                head_type=config.head_type,
                res_attention=False
                )    
            ckpt_path = "ts_benchmark/baselines/pre_train/checkpoints/rose/zero-shot.pth"
        self.model = self.transfer_weights(ckpt_path, self.model, exclude_head=False)

        
    def transfer_weights(self, ckpt_path, model, exclude_head=True):

        new_state_dict = torch.load(ckpt_path)
        new_state_dict=new_state_dict['model']

        matched_layers = 0
        m_layers = []
        unmatched_layers = []
        for name, param in model.state_dict().items():        
            if exclude_head and 'head' in name: continue
      
            if name in new_state_dict:            
                matched_layers += 1          
                input_param = new_state_dict[name]
                if input_param.shape == param.shape: 
                    param.copy_(input_param)
                    m_layers.append(name)
                else: unmatched_layers.append(name)
            else:
                unmatched_layers.append(name)
                pass 
        if matched_layers == 0:
            print(f'matched)layers:{m_layers}')
            print(f'check unmatched_layers: {unmatched_layers}') 
            raise Exception("No shared weight names were found between the models")

        return model

    def freeze(self):
        """ 
        freeze the model head
        require the model to have head attribute
        """
        if hasattr(self.model, 'head_720'): 
            # print('model head is available')
            for param in self.model.parameters(): param.requires_grad = False
            for param in self.model.head_720.parameters(): param.requires_grad = True
            # self.model.task_token_prompt.requires_grad = True
            self.model.u.requires_grad = True
            self.model.v.requires_grad = True
        if hasattr(self.model, 'head_336'): 
            # print('model head is available')
            for param in self.model.parameters(): param.requires_grad = False
            for param in self.model.head_336.parameters(): param.requires_grad = True
            # self.model.task_token_prompt.requires_grad = True
            self.model.u.requires_grad = True
            self.model.v.requires_grad = True
        if hasattr(self.model, 'head_192'): 
            # print('model head is available')
            for param in self.model.parameters(): param.requires_grad = False
            for param in self.model.head_192.parameters(): param.requires_grad = True
            # self.model.task_token_prompt.requires_grad = True
            self.model.u.requires_grad = True
            self.model.v.requires_grad = True
        if hasattr(self.model, 'head_96'): 
            # print('model head is available')
            for param in self.model.parameters(): param.requires_grad = False
            for param in self.model.head_96.parameters(): param.requires_grad = True
            # self.model.task_token_prompt.requires_grad = True
            self.model.u.requires_grad = True
            self.model.v.requires_grad = True
            print('model is frozen except the head')        
            
    def unfreeze(self):
        for param in self.model.parameters(): param.requires_grad = True
        for param in self.model.vq_embedding.parameters(): param.requires_grad = False

    def forward(self, inputs, finetune=True):        
        """
        inputs: [bs x seq_len x n_vars]
        """
        B, seq_len, K = inputs.shape
        
        revin_inputs = self.revin(inputs, 'norm')
        
        num_patch = (max(seq_len, self.patch_len)-self.patch_len) // self.stride + 1
        tgt_len = self.patch_len  + self.stride*(num_patch-1)
        s_begin = seq_len - tgt_len
            
        patch_inputs = revin_inputs[:, s_begin:, :]  # inputs: [bs x tgt_len x nvars]
        patch_inputs = patch_inputs.unfold(dimension=1, size=self.patch_len, step=self.stride)  # inputs: [bs x num_patch x n_vars x patch_len]

        output, xe, xq = 0, 0, 0
        if self.n_embedding!=0:
            
            if finetune:
                pred_96, pred_192, pred_336, pred_720, xe, xq = self.model(patch_inputs)
            else:
                pred_96, pred_192, pred_336, pred_720, res, xe, xq = self.model(patch_inputs)
                
            if self.prediction_length ==96:
                pred = pred_96
            elif self.prediction_length ==192:
                pred = pred_192
            elif self.prediction_length ==336:
                pred = pred_336
            elif self.prediction_length ==720:
                pred = pred_720
            # return output, xe, xq
        else:
            pred = self.model(self.xb)
            
        output = self.revin(pred, 'denorm')
        return output, xe, xq
            
