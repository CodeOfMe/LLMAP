import torch
import math

def gelu(x):
    return 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))

if __name__ == "__main__":
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    custom_gelu = gelu(x)
    pytorch_gelu = torch.nn.functional.gelu(x)
    print(f"Input: {x.tolist()}")
    print(f"GELU output: {[f'{v:.4f}' for v in custom_gelu.tolist()]}")
    print(f"PyTorch built-in GELU: {[f'{v:.4f}' for v in pytorch_gelu.tolist()]}")
    print(f"Max difference: {(custom_gelu - pytorch_gelu).abs().max().item():.3e}")