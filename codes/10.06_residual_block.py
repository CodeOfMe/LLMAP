import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x / rms)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        B, S, _ = x.shape
        Q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(attn)

class FeedForward(nn.Module):
    def __init__(self, hidden_dim, ff_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_dim):
        super().__init__()
        self.ln1 = RMSNorm(hidden_dim)
        self.attn = MultiHeadAttention(hidden_dim, num_heads)
        self.ln2 = RMSNorm(hidden_dim)
        self.ff = FeedForward(hidden_dim, ff_dim)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

if __name__ == "__main__":
    torch.manual_seed(42)
    hidden_dim = 32
    num_heads = 4
    ff_dim = 128
    block = TransformerBlock(hidden_dim, num_heads, ff_dim)
    x = torch.randn(1, 4, hidden_dim)
    out = block(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Residual connection output vs input difference norm: {(out - x).norm().item():.4f}")