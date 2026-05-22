import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, d_k):
    scores = Q @ K.T / math.sqrt(d_k)
    seq_len = scores.shape[0]
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores = scores.masked_fill(mask, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    output = attn_weights @ V
    return output, attn_weights

if __name__ == "__main__":
    d_k = 4
    Q = torch.randn(4, d_k)
    K = torch.randn(4, d_k)
    V = torch.randn(4, d_k)

    output, weights = scaled_dot_product_attention(Q, K, V, d_k)
    print(f"Q shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"V shape: {V.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Lower triangular (causal) check:")
    print(f"  Row 0 sums to: {weights[0].sum().item():.4f} (should be ~1.0)")
    print(f"  Upper triangle all zero: {(weights.triu(diagonal=1) == 0).all().item()}")