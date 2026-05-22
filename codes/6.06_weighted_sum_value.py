import torch
import torch.nn.functional as F
import math

if __name__ == "__main__":
    torch.manual_seed(42)
    seq_len = 4
    d_k = 2
    
    Q = torch.randn(seq_len, d_k)
    K = torch.randn(seq_len, d_k)
    V = torch.randn(seq_len, d_k)
    
    scores = Q @ K.T / math.sqrt(d_k)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scaled_scores = scores.masked_fill(mask, float('-inf'))
    attn_weights = F.softmax(scaled_scores, dim=-1)
    output = attn_weights @ V
    
    print(f"output shape: {output.shape}")
    print(f"output:\n{output}")