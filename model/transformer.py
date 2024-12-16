import numpy as np
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange, repeat
from model.positional_encoding import positional_encoding_sine
from model.multi_head_attention import MultiheadAttention
from torch.nn import Linear, LayerNorm, Dropout
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_, normal_


class ConditionNetwork(nn.Module):
    def __init__(self, length, in_channels, step_size, steps_per_token):
        super(ConditionNetwork, self).__init__()
        self._length = length
        self._in_channels = in_channels
        self._step_size = step_size
        self._steps_per_token = steps_per_token
        self._d_model = step_size * steps_per_token * in_channels
        if self._length % self._step_size != 0:
            raise Exception('Length is not divisible by step size')
        self._n_token = int(self._length / self._step_size) - steps_per_token + 1

    @property
    def n_token(self):
        return self._n_token

    @property
    def d_model(self):
        return self._d_model

    def forward(self, c):
        # n_batch, n_channel, n_data = c.shape
        # if n_channel != self._in_channels:
        #     raise Exception('Channel does not match')
        # if n_data != self._length:
        #     raise Exception('Data length does not match')
        # c = c.unfold(dimension=-1, size=self._step_size * self._steps_per_token, step=self._step_size)
        c = torch.reshape(c, (c.shape[0], c.shape[1], -1, self._step_size))
        # c = rearrange(c, 'b c t d -> t b (c d)')
        c = c.permute(2, 0, 1, 3)
        c = c.flatten(start_dim=2, end_dim=3)
        return c


class Transformer(nn.Module):
    def __init__(self, length, channels, num_layers, d_model, n_token, n_head, dim_feedforward, dropout,
                 activation="relu", cond_net=None):
        super(Transformer, self).__init__()
        self._length = length
        self._channels = channels
        self._num_layers = num_layers
        self._d_model = d_model
        self._n_token = n_token
        if self._length % self._n_token != 0:
            raise Exception('Length is not divisible by number of tokens')
        self._step_size = int((self._channels * self._length) / self._n_token)
        self._n_head = n_head
        self._dim_feedforward = dim_feedforward
        self._dropout = dropout
        self._activation = activation
        self._cond_net = cond_net
        self._cond_d_model = cond_net.d_model
        self._cond_n_token = cond_net.n_token

        # Transformer layers
        self._layers = nn.ModuleList()
        for _ in range(self._num_layers):
            layer = TransformerLayer(self._d_model, self._n_token, n_head, dim_feedforward, dropout,
                                     activation="relu", cond_d_model=self._cond_d_model,
                                     cond_n_token=self._cond_n_token)
            self._layers.append(layer)

        self._embedding = Parameter(torch.empty((self._n_token, self._d_model), requires_grad=True))
        self._linear = Linear(in_features=self._d_model, out_features=self._step_size, bias=True)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self._embedding)
        xavier_uniform_(self._linear.weight)
        constant_(self._linear.bias, 0.)

    def forward(self, cond):
        c = self._cond_net(cond)
        n_batch = c.shape[1]
        #x = repeat(self._embedding, 't d -> t b d', b=n_batch)
        x = self._embedding[:, None, :].repeat(1, n_batch, 1)
        for i, layer in enumerate(self._layers):
            x = layer(input=x, cond=c)
        #y = rearrange(self._linear(x), 't b (c d) -> b c (t d)', c=self._channels)
        y = self._linear(x)
        y = torch.reshape(y, shape=(y.shape[0], y.shape[1], self._channels, -1))
        y = y.permute(1, 2, 0, 3)
        y = y.flatten(start_dim=2, end_dim=3)
        return y


class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_token, n_head, dim_feedforward, dropout, activation="relu",
                 cond_d_model=None, cond_n_token=None):
        super(TransformerLayer, self).__init__()
        self._d_model = d_model
        self._n_token = n_token
        self._n_head = n_head
        self._dim_feedforward = dim_feedforward
        self._dropout = dropout
        self._activation = activation
        self._cond_d_model = cond_d_model
        self._cond_n_token = cond_n_token

        # Positional encoding and mask
        pe = positional_encoding_sine(n_pos=n_token, d_model=d_model, max_n_pos=n_token,
                                      normalize=True, scale=None)
        self.register_buffer('pe', pe)
        cond_pe = positional_encoding_sine(n_pos=cond_n_token, d_model=cond_d_model, max_n_pos=cond_n_token,
                                           normalize=True, scale=None)
        self.register_buffer('cond_pe', cond_pe)
        # Multihead attention modules
        self.mha = MultiheadAttention(embed_dim=d_model, num_heads=n_head, kdim=d_model, vdim=d_model,
                                      dropout=dropout, bias=True, dtype=None)
        self.cond_mha = MultiheadAttention(embed_dim=d_model, num_heads=n_head, kdim=cond_d_model,
                                           vdim=cond_d_model, dropout=dropout, bias=True,
                                           dtype=None)
        # Feedforward neural network
        self.ffnn_linear1 = Linear(in_features=d_model, out_features=dim_feedforward, bias=True)
        self.ffnn_dropout = Dropout(dropout)
        self.ffnn_linear2 = Linear(in_features=dim_feedforward, out_features=d_model, bias=True)
        self.activation = _get_activation_fn(activation)
        # Layer norm and dropout
        layer_norm_eps = 1e-5
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        # Reset parameters
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.ffnn_linear1.weight)
        xavier_uniform_(self.ffnn_linear2.weight)
        constant_(self.ffnn_linear1.bias, 0.)
        constant_(self.ffnn_linear2.bias, 0.)

    def forward(self, input, cond=None):
        # Multi-head attention
        x = input
        key = value = input
        x2, _ = self.mha(query=x, key=key, value=value, query_pos=self.pe, key_pos=self.pe,
                         attn_mask=None, need_weights=False, average_attn_weights=False)
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x2, _ = self.cond_mha(query=x, key=cond, value=cond, query_pos=self.pe, key_pos=self.cond_pe,
                              attn_mask=None, need_weights=False, average_attn_weights=False)
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        x2 = self.ffnn_linear2(self.ffnn_dropout(self.activation(self.ffnn_linear1(x))))
        x = x + self.dropout3(x2)
        x = self.norm3(x)
        return x


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu/glu, not {activation}.")


class Shrink(nn.Module):
    def __init__(self, shape, device):
        super(Shrink, self).__init__()
        self._shape = shape
        self._device = device
        self._alpha = nn.Parameter(torch.empty(shape, device=device, requires_grad=True))
        self._beta = nn.Parameter(torch.empty(shape, device=device, requires_grad=True))
        # Reset parameters
        self._reset_parameters()

    def _reset_parameters(self):
        normal_(self._alpha)
        normal_(self._beta)

    def forward(self, x):
        alpha = F.softplus(self._alpha)
        beta = F.softplus(self._beta)
        y = beta * (x - torch.div(F.tanh(alpha * x), alpha))
        return y
