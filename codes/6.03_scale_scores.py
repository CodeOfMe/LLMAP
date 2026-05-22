import torch
import math

if __name__ == "__main__":
    torch.manual_seed(42)
    seq_len = 4
    d_k = 2
    
    Q = torch.randn(seq_len, d_k)
    K = torch.randn(seq_len, d_k)
    scores = Q @ K.T
    scaled_scores = scores / math.sqrt(d_k)
    
    print(f"scaled_scores:\n{scaled_scores}")