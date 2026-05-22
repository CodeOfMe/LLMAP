import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardSwiGLU(nn.Module):
    def __init__(self, hidden_dim, ff_dim):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, ff_dim)
        self.w2 = nn.Linear(ff_dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, ff_dim)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

if __name__ == "__main__":
    torch.manual_seed(42)
    hidden_dim = 32
    ff_dim = 128
    ffn = FeedForwardSwiGLU(hidden_dim, ff_dim)
    x = torch.randn(1, 4, hidden_dim)
    out = ffn(x)
    w1_params = ffn.w1.weight.numel() + ffn.w1.bias.numel()
    w2_params = ffn.w2.weight.numel() + ffn.w2.bias.numel()
    w3_params = ffn.w3.weight.numel() + ffn.w3.bias.numel()
    total = w1_params + w2_params + w3_params
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Total parameters: {total}")
    print(f"w1 parameters: {w1_params}")
    print(f"w2 parameters: {w2_params}")
    print(f"w3 parameters: {w3_params}")