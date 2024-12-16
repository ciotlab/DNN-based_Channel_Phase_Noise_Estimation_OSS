import math
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.parameter import Parameter
from torch.nn.functional import linear, softmax, dropout
from einops import rearrange, repeat


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, kdim, vdim, dropout=0., bias=True, dtype=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim
        self.vdim = vdim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), dtype=dtype))
        self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), dtype=dtype))
        self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), dtype=dtype))
        if bias:
            self.in_proj_bias = Parameter(torch.empty(3*embed_dim, dtype=dtype))
        else:
            self.in_proj_bias = None
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, dtype=dtype)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.q_proj_weight)
        xavier_uniform_(self.k_proj_weight)
        xavier_uniform_(self.v_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
        xavier_uniform_(self.out_proj.weight)
        constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, query_pos, key_pos, attn_mask=None, need_weights=True, average_attn_weights=True):
        tgt_len, batch_size, _ = query.shape
        src_len, _, _ = key.shape

        if self.in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = self.in_proj_bias.chunk(3)

        query = query + query_pos[:, None, :]  # query_seq, batch, data
        key = key + key_pos[:, None, :]  # key_seq, batch, data

        q = linear(query, self.q_proj_weight, b_q)
        k = linear(key, self.k_proj_weight, b_k)
        v = linear(value, self.v_proj_weight, b_v)

        q = q.contiguous().view(tgt_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(k.shape[0], batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(v.shape[0], batch_size * self.num_heads, self.head_dim).transpose(0, 1)

        if attn_mask is not None:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask == 0, float("-inf"))
            attn_mask = new_attn_mask

        if not self.training:
            dropout_p = 0.0
        else:
            dropout_p = self.dropout

        q = q / math.sqrt(self.head_dim)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        if attn_mask is not None:
            if attn_mask.dim() == 2:  # (query_seq, key_seq)
                # attn_mask = repeat(attn_mask, 'qs ks -> qs ks b', b=batch_size)
                attn_mask = attn_mask[:, :, None].repeat(1, 1, batch_size)
            #attn_mask = repeat(attn_mask, 'qs ks b -> (b h) qs ks', h=self.num_heads)
            attn_mask = attn_mask.permute(2, 0, 1)
            attn_mask = attn_mask[:, None, :, :].repeat(1,  self.num_heads, 1, 1)
            attn_mask = attn_mask.flatten(start_dim=0, end_dim=1)
            attn_weights += attn_mask
        attn_weights = softmax(attn_weights, dim=-1)
        if dropout_p > 0.0:
            attn_weights = dropout(attn_weights, p=dropout_p)
        attn_output = torch.bmm(attn_weights, v)

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * batch_size, self.embed_dim)
        attn_output = linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        attn_output = attn_output.view(tgt_len, batch_size, self.embed_dim)

        if need_weights:
            attn_output_weights = attn_weights.view(batch_size, self.num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.sum(dim=1) / self.num_heads
            return attn_output, attn_output_weights
        else:
            return attn_output, None
