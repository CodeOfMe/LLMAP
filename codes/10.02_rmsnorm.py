import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x / rms)

if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(2, 4, 8)
    rmsnorm = RMSNorm(8)
    out = rmsnorm(x)
    input_rms = torch.sqrt(torch.mean(x ** 2, dim=-1))
    output_rms = torch.sqrt(torch.mean(out ** 2, dim=-1))
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Input RMS: {input_rms}")
    print(f"Output RMS: {output_rms}")