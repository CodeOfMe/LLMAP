import torch
import math

def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def precompute_freqs_cis(dim, max_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, freqs)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin

def rope_pairs(x, cos_val, sin_val):
    xp = x.float().reshape(*x.shape[:-1], -1, 2)
    x1, x2 = xp[..., 0], xp[..., 1]
    o1 = x1 * cos_val - x2 * sin_val
    o2 = x1 * sin_val + x2 * cos_val
    return torch.stack([o1, o2], dim=-1).flatten(-2)

if __name__ == "__main__":
    cos, sin = precompute_freqs_cis(dim=8, max_len=16)
    print(f">>> cos, sin = precompute_freqs_cis(dim=8, max_len=16)")
    print(f">>> cos.shape, sin.shape\n({cos.shape}, {sin.shape})")
    print(f">>> cos[0]  # Cosine values at position 0\n{cos[0]}")
    print(f">>> sin[0]  # Sine values at position 0\n{sin[0]}")
    
    torch.manual_seed(42)
    q, k = torch.randn(1,1,1,8), torch.randn(1,1,1,8)
    d0 = (rope_pairs(q,cos[0],sin[0]) * rope_pairs(k,cos[2],sin[2])).sum()
    d1 = (rope_pairs(q,cos[2],sin[2]) * rope_pairs(k,cos[4],sin[4])).sum()
    d2 = (rope_pairs(q,cos[3],sin[3]) * rope_pairs(k,cos[5],sin[5])).sum()
    print(f"\n# Verify RoPE key property: inner product depends only on relative position")
    print(f">>> d0.item(), d1.item(), d2.item()")
    print(f"({d0.item():.6f}, {d1.item():.6f}, {d2.item():.6f})  # Relative distance is always 2, inner products are equal")