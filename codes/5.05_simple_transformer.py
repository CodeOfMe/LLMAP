import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
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
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, num_heads)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ff = FeedForward(hidden_dim, ff_dim)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, ff_dim, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim
    
    def forward(self, token_ids):
        seq_len = token_ids.shape[1]
        positions = torch.arange(seq_len, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.pos_embedding(positions)
        hidden = hidden * (self.hidden_dim ** 0.5)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.ln_f(hidden)
        logits = F.linear(hidden, self.token_embedding.weight)
        return logits

if __name__ == "__main__":
    torch.manual_seed(42)
    vocab_size = 1000
    hidden_dim = 64
    num_layers = 2
    num_heads = 4
    ff_dim = 256
    max_seq_len = 128
    
    model = SimpleTransformer(vocab_size, hidden_dim, num_layers, num_heads, ff_dim, max_seq_len)
    total_params = sum(p.numel() for p in model.parameters())
    
    token_ids_list = [1, 50, 300, 500, 999]
    token_ids = torch.tensor([token_ids_list])
    logits = model(token_ids)
    
    emb_params = model.token_embedding.weight.numel() + model.pos_embedding.weight.numel()
    attn_params = sum(p.numel() for name, p in model.named_parameters() if 'attn' in name)
    ff_params = sum(p.numel() for name, p in model.named_parameters() if 'ff' in name or '.net.' in name)
    ln_params = sum(p.numel() for name, p in model.named_parameters() if 'ln' in name)
    
    probs = F.softmax(logits[0, -1], dim=-1)
    top5_probs, top5_ids = probs.topk(5)
    
    print(f"Model parameters: {total_params:,}")
    print(f"Input token IDs: {token_ids_list}")
    print(f"Input shape: {token_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Embedding params: {emb_params:,}")
    print(f"Attention params: {attn_params:,}")
    print(f"FeedForward params: {ff_params:,}")
    print(f"LayerNorm params: {ln_params:,}")
    print(f"Top-5 predicted token IDs: {top5_ids.tolist()}")
    print(f"Top-5 probabilities: {[f'{p:.1f}' for p in top5_probs.tolist()]}")