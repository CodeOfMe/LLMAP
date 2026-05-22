import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

def gelu(x):
    return 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))

if __name__ == "__main__":
    x = torch.randn(2, 10, 64)
    ln = LayerNorm(64)
    rn = RMSNorm(64)
    print(f"LayerNorm output shape: {ln(x).shape}")
    print(f"LayerNorm output mean: {ln(x).mean(dim=-1)[0, 0].item():.6f} (should be ~0)")
    print(f"RMSNorm output shape: {rn(x).shape}")

    x_act = torch.linspace(-3, 3, 7)
    print(f"\nInput: {x_act.tolist()}")
    print(f"ReLU: {torch.relu(x_act).tolist()}")
    print(f"GELU: {gelu(x_act).tolist()}")

    # Parameter comparison
    hidden_dim = 4096
    ff_dim = 14336
    standard_ffn_params = 2 * hidden_dim * ff_dim
    swiglu_ffn_params = 3 * hidden_dim * ff_dim
    print(f"\nStandard FFN params: {standard_ffn_params:,}")
    print(f"SwiGLU FFN params: {swiglu_ffn_params:,}")