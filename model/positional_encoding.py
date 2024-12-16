import math
import torch
from einops import rearrange, repeat


def positional_encoding_sine(n_pos, d_model, max_n_pos, normalize=False, scale=None):
    seq_embed = torch.arange(1, n_pos + 1)
    if normalize:
        eps = 1e-6
        if scale is None:
            scale = 2 * math.pi * max_n_pos
        seq_embed = seq_embed / (seq_embed[-1] + eps) * scale
    dim_t = torch.arange(d_model)
    dim_t = max_n_pos ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / d_model)

    pe = seq_embed[:, None] / dim_t
    pe = torch.stack((pe[:, 0::2].sin(), pe[:, 1::2].cos()), dim=2).flatten(1)
    return pe
