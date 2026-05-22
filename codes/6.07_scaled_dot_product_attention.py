import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, d_k):
    scores = Q @ K.T / math.sqrt(d_k)
    mask = torch.triu(torch.ones_like(scores), diagonal=1).bool()
    scores = scores.masked_fill(mask, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    output = attn_weights @ V
    return output

if __name__ == "__main__":
    torch.manual_seed(42)
    seq_len = 4
    d_k = 2
    
    Q = torch.randn(seq_len, d_k)
    K = torch.randn(seq_len, d_k)
    V = torch.randn(seq_len, d_k)
    
    custom_output = scaled_dot_product_attention(Q, K, V, d_k)
    pytorch_output = F.scaled_dot_product_attention(
        Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0), is_causal=True
    ).squeeze(0)
    
    print(f"Custom implementation result:\n{custom_output}")
    print(f"PyTorch F.scaled_dot_product_attention result:\n{pytorch_output}")
    print(f"Max difference: {(custom_output - pytorch_output).abs().max().item():.2e} (within floating point precision)")