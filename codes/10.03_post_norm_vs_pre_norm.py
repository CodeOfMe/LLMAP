import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

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
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class PostNormBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(hidden_dim, num_heads)
        self.ff = FeedForward(hidden_dim)
        self.ln1 = LayerNorm(hidden_dim)
        self.ln2 = LayerNorm(hidden_dim)
    
    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ff(x))
        return x

class PreNormBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(hidden_dim, num_heads)
        self.ff = FeedForward(hidden_dim)
        self.ln1 = RMSNorm(hidden_dim)
        self.ln2 = RMSNorm(hidden_dim)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

if __name__ == "__main__":
    torch.manual_seed(42)
    hidden_dim = 32
    num_heads = 4
    x = torch.randn(1, 4, hidden_dim)
    
    post_block = PostNormBlock(hidden_dim, num_heads)
    pre_block = PreNormBlock(hidden_dim, num_heads)
    
    post_out = post_block(x)
    pre_out = pre_block(x)
    
    print(f"Post-Norm output shape: {post_out.shape}")
    print(f"Pre-Norm output shape: {pre_out.shape}")
    print(f"Post-Norm output norm: {post_out.norm().item():.4f}")
    print(f"Pre-Norm output norm: {pre_out.norm().item():.4f}")