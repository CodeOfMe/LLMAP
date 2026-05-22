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

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_dim):
        super().__init__()
        self.ln1 = RMSNorm(hidden_dim)
        self.attn = MultiHeadAttention(hidden_dim, num_heads)
        self.ln2 = RMSNorm(hidden_dim)
        self.w1 = nn.Linear(hidden_dim, ff_dim)
        self.w2 = nn.Linear(ff_dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, ff_dim)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        gate = F.silu(self.w1(self.ln2(x)))
        value = self.w3(self.ln2(x))
        x = x + self.w2(gate * value)
        return x

if __name__ == "__main__":
    torch.manual_seed(42)
    hidden_dim = 32
    num_heads = 4
    ff_dim = 128
    block = TransformerBlock(hidden_dim, num_heads, ff_dim)
    x = torch.randn(1, 4, hidden_dim)
    out = block(x)
    total_params = sum(p.numel() for p in block.parameters())
    attn_params = sum(p.numel() for n, p in block.named_parameters() if 'attn' in n or 'ln1' in n)
    ffn_params = block.w1.weight.numel() + block.w1.bias.numel() + block.w2.weight.numel() + block.w2.bias.numel() + block.w3.weight.numel() + block.w3.bias.numel()
    rmsnorm_params = block.ln1.gamma.numel() + block.ln2.gamma.numel()
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Total parameters: {total_params}")
    print(f"Attention parameters: {attn_params}")
    print(f"FFN parameters: {ffn_params}")
    print(f"RMSNorm parameters: {rmsnorm_params}")