
__all__ = ['ROSE']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict

from zmq import device
from .layers.pos_encoding import *
from .layers.basics import *
from .layers.attention import *

import torch.fft as fft
import math
import numpy as np
from einops import reduce, rearrange, repeat



# Cell
class ROSE(nn.Module):
    """
    Output dimension: 
         [bs x target_dim x nvars] for prediction
         [bs x target_dim] for regression
         [bs x target_dim] for classification
         [bs x num_patch x n_vars x patch_len] for pretrain
    """
    def __init__(self, c_in:int, target_dim:int, patch_len:int, stride:int,n_embedding:int, num_patch:int, mask_mode:str = 'patch',mask_nums:int = 3,
                 n_layers:int=3, d_model=128, n_heads=16, shared_embedding=True, d_ff:int=256, num_slots:int=10,num_token=1,
                 norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", 
                 res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, head_dropout = 0, 
                 head_type = "prediction", individual = False, 
                 y_range:Optional[tuple]=None, verbose:bool=False, fft=False, **kwargs):

        super().__init__()

        assert head_type in ['pretrain', 'prediction', 'regression', 'classification'], 'head type should be either pretrain, prediction, or regression'

        # Backbone
        self.backbone = PatTSTEncoder(c_in, num_patch=num_patch+num_token, patch_len=patch_len, 
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads,fft=True, 
                                shared_embedding=shared_embedding, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, 
                                res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)
        self.backbone_2 = PatTSTEncoder(c_in, num_patch=num_patch+num_token, patch_len=patch_len, 
                                n_layers=21, d_model=d_model, n_heads=n_heads,fft=True, 
                                shared_embedding=shared_embedding, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, 
                                res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)
        self.encoder_predict = TowerEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=3, 
                                    store_attn=store_attn)
        self.encoder_reconstruct = TowerEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=3, 
                                    store_attn=store_attn)
        
        # Input encoding
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.token_embed = nn.Linear(num_patch*patch_len, d_model*num_token)

        # Vq 
        self.n_embedding=n_embedding
        if self.n_embedding!=0:
            self.vq_embedding = nn.Embedding(self.n_embedding, d_model*num_token)
            self.vq_embedding.weight.data.uniform_(-1.0 / d_model,
                                                1.0 / d_model)

        
        # Head
        self.n_vars = c_in
        self.head_type = head_type
        self.mask_mode = mask_mode
        self.mask_nums = mask_nums
        self.d_model  = d_model
        self.target_dim = target_dim
        self.num_token = num_token

        if head_type == "pretrain":
            self.head_reconstruct = PretrainHead(d_model, patch_len, head_dropout) # custom head passed as a partial func with all its kwargs
            self.head_96 = PredictionHead(individual, self.n_vars, d_model, num_patch, 96, head_dropout)
            self.head_192 = PredictionHead(individual, self.n_vars, d_model, num_patch, 192, head_dropout)
            self.head_336 = PredictionHead(individual, self.n_vars, d_model, num_patch, 336, head_dropout)
            self.head_720 = PredictionHead(individual, self.n_vars, d_model, num_patch, 720, head_dropout)
        elif head_type == "prediction":
            self.head_96 = PredictionHead(individual, self.n_vars, d_model, num_patch, 96, head_dropout)
            self.head_192 = PredictionHead(individual, self.n_vars, d_model, num_patch, 192, head_dropout)
            self.head_336 = PredictionHead(individual, self.n_vars, d_model, num_patch, 336, head_dropout)
            self.head_720 = PredictionHead(individual, self.n_vars, d_model, num_patch, 720, head_dropout)
            # self.head = SimpleHead(self.d_model, self.patch_target, head_dropout)
        elif head_type == "regression":
            self.head = RegressionHead(self.n_vars, d_model, target_dim, head_dropout, y_range)
        elif head_type == "classification":
            self.head = ClassificationHead(self.n_vars, d_model, target_dim, head_dropout)

        # self.initialize_weights()
    
    def initialize_weights(self):
        # initialization
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.task_token_predict, std=.02)
        torch.nn.init.normal_(self.task_token_reconstruct, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def encoder_copy(self):
        for n_r, n_p in zip(self.encoder_reconstruct.named_parameters(), self.encoder_predict.named_parameters()):
            if torch.equal(n_r[1].data, n_p[1].data):
                continue
            else:
                print(f'Parameters "{n_r[0]}" are different')
                print('false')
            n_p[1].data.copy_(n_r[1].data)  # initialize
            n_p[1].requires_grad = False  # not update by gradient
                         
    # @profile(precision=4,stream=open('forward.log','w+')) 
    def forward(self, z):                             
        """
        z: tensor [bs x num_patch x n_vars x patch_len]
        """   

        bs, num_patch, n_vars, patch_len = z.shape 

        #token
        z_token=self.token_embed(z.permute(0,2,1,3).reshape(-1, n_vars, num_patch*patch_len))
        z_token=z_token.unsqueeze(1)        #[bs x 1 x nvars x d_model]

        if self.n_embedding!=0:
            embedding = self.vq_embedding.weight.data
            B, N, C, D = z_token.shape
            K, _ = embedding.shape
            embedding_broadcast = embedding.reshape(1, K, 1, 1, D)
            z_token_broadcast = z_token.reshape(B, 1, N, C, D)
            distance = torch.sum((embedding_broadcast - z_token_broadcast) ** 2, 4)
            # make C to the second dim
            z_token_q = self.vq_embedding(torch.argmin(distance, 1))
            # stop gradient
            z_token_encoderinput = z_token + (z_token_q - z_token).detach()
            z_token_encoderinput = z_token_encoderinput.reshape(-1,n_vars,self.num_token,self.d_model).permute(0,2,1,3)
        else:
            z_token_encoderinput = z_token
            z_token_encoderinput = z_token_encoderinput.reshape(-1,n_vars,self.num_token,self.d_model).permute(0,2,1,3)

        # patch embedding
        z = self.patch_embed(z)

        # concat task token
        z = torch.cat((z_token_encoderinput, z), dim=1)
        
        # sharedn backbone
        res=self.backbone_2(z)
        z = self.backbone(z)
        
        # tower encoder
        # with torch.no_grad():
        #     self.encoder_copy()
        output = self.encoder_reconstruct(z[:,:,:,self.num_token:])
        # output_predict = self.encoder_predict(z[:,:,:,1:])
        # output = (output_predict + output_reconstruct)/2

        # head
        output_96 = self.head_96(output[:,:,:,:])
        output_192 = self.head_192(output[:,:,:,:])
        output_336 = self.head_336(output[:,:,:,:])
        output_720 = self.head_720(output[:,:,:,:])
        return output_96, output_192,output_336,output_720,res, z_token_q, z_token


        # if self.n_embedding!=0:
        #     return output, z_token_q, z_token
        # else:
        #     return output


class RegressionHead(nn.Module):
    def __init__(self, n_vars, d_model, output_dim, head_dropout, y_range=None):
        super().__init__()
        self.y_range = y_range
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, output_dim)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x output_dim]
        """
        x = x[:,:,:,-1]             # only consider the last item in the sequence, x: bs x nvars x d_model
        x = self.flatten(x)         # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x output_dim
        if self.y_range: y = SigmoidRange(*self.y_range)(y)        
        return y


class ClassificationHead(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, n_classes)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x n_classes]
        """
        x = x[:,:,:,-1]             # only consider the last item in the sequence, x: bs x nvars x d_model
        x = self.flatten(x)         # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x n_classes
        return y
    

class PredictionHead(nn.Module):
    def __init__(self, individual, n_vars, d_model, num_patch, forecast_len, head_dropout=0, flatten=False):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        self.flatten = flatten
        head_dim = d_model*num_patch
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(head_dim, forecast_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(head_dim, forecast_len)
            self.dropout = nn.Dropout(head_dropout)


    def forward(self, x):                     
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * num_patch]
                z = self.linears[i](z)                    # z: [bs x forecast_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)         # x: [bs x nvars x forecast_len]
        else:
            x = self.flatten(x)     # x: [bs x nvars x (d_model * num_patch)]    
            x = self.dropout(x)
            x = self.linear(x)      # x: [bs x nvars x forecast_len]
        return x.transpose(2,1)     # [bs x forecast_len x nvars]


class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        """
        x: tensor [bs x nvars x d_model x num_patch]
        output: tensor [bs x nvars x num_patch x patch_len]
        """

        x = x.transpose(2,3)                     # [bs x nvars x num_patch x d_model]
        x = self.linear( self.dropout(x) )      # [bs x nvars x num_patch x patch_len]
        x = x.permute(0,2,1,3)                  # [bs x num_patch x nvars x patch_len]
        return x

class SimpleHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(d_model, 64)
        self.linear_2 = nn.Linear(64, 32)
        self.linear_3 = nn.Linear(32, patch_len)
        self.patch_len = patch_len

    def forward(self, x):
        """
        x: tensor [bs x nvars x d_model x num_patch]
        output: tensor [bs x nvars x num_patch x patch_len]
        """
        bs , n_vars, d_model , num_patch = x.shape
        x = x.transpose(2,3)                     # [bs x nvars x num_patch x d_model]
        x = self.linear_1(x)      # [bs x nvars x num_patch x patch_len]
        x = self.linear_2( self.dropout(x) )
        x = self.linear_3( self.dropout(x) )
        x = x.reshape(bs, n_vars, num_patch*self.patch_len)                  # [bs x num_patch x nvars*patch_len]
        return x.transpose(1,2)


class PatTSTEncoder(nn.Module):
    def __init__(self, c_in, num_patch, patch_len, 
                 n_layers=3, d_model=128, n_heads=16, shared_embedding=True,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):

        super().__init__()
        self.n_vars = c_in
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding          

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, num_patch, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, 
                                    store_attn=store_attn)

        # 

    def forward(self, x) -> Tensor:          
        """
        x: tensor [bs x num_patch x nvars x patch_len]
        """
        bs, num_patch, n_vars, _ = x.shape

        x = x.transpose(1,2)                                                     # x: [bs x nvars x num_patch x d_model]        

        u = torch.reshape(x, (bs*n_vars, num_patch, self.d_model) )              # u: [bs * nvars x num_patch x d_model]
        u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x num_patch x d_model]

        # Encoder
        z = self.encoder(u)                                                      # z: [bs * nvars x num_patch x d_model]
        z = torch.reshape(z, (-1,n_vars, num_patch, self.d_model))               # z: [bs x nvars x num_patch x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x num_patch]

        return z
    
    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor):
        """
        src: tensor [bs x q_len x d_model]
        """
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores)
            return output
        else:
            for mod in self.layers: output = mod(output)
            return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, 
                activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None):
        """
        src: tensor [bs x q_len x d_model]
        """
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev)
        else:
            src2, attn = self.self_attn(src, src, src)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src


class ContrastiveHead(nn.Module):
    def __init__(self, num_patch, kernels, d_model):
        super().__init__()
        self.kernels = kernels
        self.tfd = nn.ModuleList(
                [nn.Conv1d(d_model, d_model, k, padding=k-1) for k in kernels]
            )

        self.sfd = nn.ModuleList(
                [BandedFourierLayer(d_model, d_model, b, 1, length=num_patch) for b in range(1)]
            )
        self.repr_dropout = nn.Dropout(p=0.1)
    
    def forward(self, x):
        # x: [bs x nvars x d_model x num_patch] 
        bs, n_vars, d_model, num_patch = x.shape
        x = x.reshape((bs*n_vars, d_model, num_patch))
        return self.trend(x), self.season(x)

    def trend(self, x):
        trend = []
        for idx, mod in enumerate(self.tfd):
            out = mod(x)  # B*C x d_model x q_len
            if self.kernels[idx] != 1:
                out = out[..., :-(self.kernels[idx] - 1)]
            trend.append(out.transpose(1, 2))  # B*C x patch_num x d_model
        trend = reduce(
            rearrange(trend, 'list b t d -> list b t d'),
            'list b t d -> b t d', 'mean'
        )

        return trend
        
    def season(self, x):

        x = x.transpose(1, 2)  # B x T x Co
        season = []
        for mod in self.sfd:
            out = mod(x)  # b t d
            season.append(out)
        season = season[0]
        season = self.repr_dropout(season)

        return season


class BandedFourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, band, num_bands, length=201):
        super().__init__()

        self.length = length
        self.total_freqs = (self.length // 2) + 1

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.band = band  # zero indexed
        self.num_bands = num_bands

        self.num_freqs = self.total_freqs // self.num_bands + (self.total_freqs % self.num_bands if self.band == self.num_bands - 1 else 0)

        self.start = self.band * (self.total_freqs // self.num_bands)
        self.end = self.start + self.num_freqs


        # case: from other frequencies
        self.weight = nn.Parameter(torch.empty((self.num_freqs, in_channels, out_channels), dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.empty((self.num_freqs, out_channels), dtype=torch.cfloat))
        self.reset_parameters()

    def forward(self, input):
        # input - b t d
        b, t, _ = input.shape
        input_fft = fft.rfft(input, dim=1)
        output_fft = torch.zeros(b, t // 2 + 1, self.out_channels, device=input.device, dtype=torch.cfloat)
        output_fft[:, self.start:self.end] = self._forward(input_fft)
        return fft.irfft(output_fft, n=input.size(1), dim=1)

    def _forward(self, input):
        output = torch.einsum('bti,tio->bto', input[:, self.start:self.end], self.weight)
        return output + self.bias

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

class TowerEncoder(nn.Module):
    '''
    input:  x [bs x nvars x d_model x num_patch]
    out:    x [bs x nvars x d_model x num_patch] 
    '''

    def __init__(self, d_model, n_heads, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        
        super().__init__()
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=activation, res_attention=res_attention, n_layers=n_layers, 
                                    store_attn=store_attn)
    
    def forward(self, x):

        bs, nvars, d_model, num_patch = x.shape
        x = x.permute(0,1,3,2)
        x = x.reshape(-1, num_patch, d_model)
        x = self.encoder(x)
        x = x.reshape(bs, nvars, num_patch, d_model)
        x = x.permute(0,1,3,2)

        return x



# Define the gating model 
class Gating(nn.Module): 
    def __init__(self, input_dim, 
                    num_experts, dropout_rate=0.1,hard=True): 
        super(Gating, self).__init__() 

        # Layers 
        self.layer1 = nn.Linear(input_dim, 128) 
        self.dropout1 = nn.Dropout(dropout_rate) 

        self.layer2 = nn.Linear(128, 256) 
        self.leaky_relu1 = nn.LeakyReLU() 
        self.dropout2 = nn.Dropout(dropout_rate) 

        self.layer3 = nn.Linear(256, 128) 
        self.leaky_relu2 = nn.LeakyReLU() 
        self.dropout3 = nn.Dropout(dropout_rate) 

        self.layer4 = nn.Linear(128, num_experts) 
        self.hard = hard

    def forward(self, x): 
        x = torch.relu(self.layer1(x)) 
        x = self.dropout1(x) 

        x = self.layer2(x) 
        x = self.leaky_relu1(x) 
        x = self.dropout2(x) 

        x = self.layer3(x) 
        x = self.leaky_relu2(x) 
        x = self.dropout3(x) 
        x = self.layer4(x) 

        return F.gumbel_softmax(x, tau=1,hard=self.hard)
 
class Choose(nn.Module): 

    def __init__(self, input_dim, num_experts, dropout_rate=0.1, hard=True): 
        super(Choose, self).__init__() 

        self.gate = Gating( input_dim=input_dim, num_experts= num_experts, dropout_rate=dropout_rate, hard=hard)

    def forward(self, x, expert1, expert2): 

        bs, n_vars, d_model, num_patch = x.shape
        prob = self.gate(x.permute(0,1,3,2).reshape(bs, n_vars, -1))  # prob: [bs x n_vars x 2]]
        # print(prob[:,:,0].sum()/prob[:,:,:].sum())
        expert1 = prob[:,:,0].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, d_model, num_patch)*expert1
        expert2 = prob[:,:,1].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, d_model, num_patch)*expert2

        return expert1+expert2

    
     
