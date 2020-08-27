import torch
from torch import nn, einsum
from einops import rearrange
import torch.nn.functional as F

class KroneckerSelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads = 32):
        super().__init__()
        hidden_dim = heads * dim_heads

        self.heads = heads
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        h = x.shape[-2]

        x = torch.cat((x.mean(dim=-1), x.mean(dim=-2)), dim=-1)
        x = rearrange(x, 'b c n -> b n c')

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        
        dots = einsum('bhid,bhjd->bhij', q, k)
        attn = dots.softmax(dim=-1)
        out = einsum('bhij,bhjd->bhid', attn, v)
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = rearrange(out, 'b n c -> b c n')

        # outer sum
        out = out[..., :h][..., :, None] + out[..., h:][..., None, :]
        return out
