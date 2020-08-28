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
        _, _, h, _ = x.shape

        x = torch.cat((x.mean(dim=-1), x.mean(dim=-2)), dim=-1)
        x = rearrange(x, 'b c n -> b n c')

        q, k, v = rearrange(self.to_qkv(x), 'b n (kqv h d) -> kqv b h n d', h=self.heads, kqv=3)
        
        dots = einsum('bhid,bhjd->bhij', q, k)
        attn = dots.softmax(dim=-1)
        out = einsum('bhij,bhjd->bhid', attn, v)
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = rearrange(out, 'b n c -> b c n')

        # outer sum
        out = rearrange(out[..., :h], 'b c n -> b c n 1') + rearrange(out[..., h:], 'b c n -> b c 1 n')
        return out
