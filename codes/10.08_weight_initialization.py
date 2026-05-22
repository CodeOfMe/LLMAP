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

def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, RMSNorm):
        nn.init.ones_(module.gamma)

if __name__ == "__main__":
    num_layers = 16
    layer = nn.Linear(64, 64)
    _init_weights(layer)
    normal_std = layer.weight.std().item()
    print(f"Standard layer weight std: {normal_std:.4f}")
    
    scaled_std = 0.02 / (2 * num_layers) ** 0.5
    nn.init.normal_(layer.weight, mean=0.0, std=scaled_std)
    scaled_weight_std = layer.weight.std().item()
    print(f"Scaled std: {scaled_weight_std:.6f}")
    
    last_layer_std = 0.02 / (2 * num_layers) ** 0.5
    print(f"Last layer weight std: {last_layer_std:.6f}")
    print(f"Scaling factor 1/sqrt(2N) = {1/(2*num_layers)**0.5:.4f}")