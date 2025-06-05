import torch 
from torch import nn
from .attention import *
from .pos_encoding import *
from .basics import *

class DecoderLayer(nn.Module):
    def __init__(self, patch_len, d_model, n_heads, d_ff=None, dropout=0.1, out_num_patch = 10, norm="BatchNorm"):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiheadAttention(d_model, n_heads, res_attention=False, attn_dropout=dropout)
        self.cross_attention = MultiheadAttention(d_model, n_heads, attn_dropout=dropout)
        
        if 'batch' in norm.lower():
            self.norm1 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm2 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model),
                                nn.GELU(),
                                nn.Linear(d_model, d_model))

    def forward(self, x, cross):
        batch, n_vars, num_patch, d_model = x.shape
        x = x.reshape(batch*n_vars, num_patch, d_model)
        cross = cross.reshape(batch*n_vars, -1, d_model)

        # x_attn , _= self.self_attention(x) 
        x_attn =  x

        x_cross , _ = self.cross_attention(x_attn, cross, cross)
        x_cross = self.dropout(self.norm1(x_cross)) + x_attn

        x_ff = self.MLP1(x_cross)
        x_ff = self.norm2(x_ff) + x_ff

        x_ff = x_ff.reshape(batch, n_vars, num_patch, d_model)

        return x_ff

class Decoder(nn.Module):
    def __init__(self, d_layers, patch_len, d_model, n_heads, d_ff=None, dropout=0.1, out_num_patch = 10):
        super(Decoder, self).__init__()

        self.decoder_layers = nn.ModuleList()
        for i in range(d_layers):
            self.decoder_layers.append(DecoderLayer(patch_len, d_model, n_heads, d_ff, dropout, out_num_patch))

    def forward(self, x, cross):
        output = x
        for layer in self.decoder_layers:
            output = layer(x, cross)
        return output


 

        

