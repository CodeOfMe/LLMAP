import torch
import torch.nn.functional as F

def apply_causal_mask(attn_scores):
    seq_len = attn_scores.shape[-1]
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    attn_scores = attn_scores.masked_fill(mask, float('-inf'))
    return attn_scores

if __name__ == "__main__":
    torch.manual_seed(42)
    scores = torch.randn(4, 4)
    
    print(f"Original attention scores:\n{scores}")
    masked = apply_causal_mask(scores.clone())
    print(f"\nAfter applying causal mask:\n{masked}")
    weights = F.softmax(masked, dim=-1)
    print(f"\nAttention weights after softmax:\n{weights}")