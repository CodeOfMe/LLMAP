import torch

def ntk_aware_scaling(dim, max_train_len, max_new_len, base=10000.0):
    scale = max_new_len / max_train_len
    new_base = base * (scale ** (dim / (dim - 2)))
    freqs = 1.0 / (new_base ** (torch.arange(0, dim, 2).float() / dim))
    return freqs

if __name__ == "__main__":
    dim = 64
    original_freqs = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
    ntk_freqs = ntk_aware_scaling(dim, 2048, 8192)
    print(f">>> dim = {dim}")
    print(f">>> original_freqs = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))")
    print(f">>> ntk_freqs = ntk_aware_scaling(dim, 2048, 8192)")
    print(f">>> original_freqs[:5]\n{original_freqs[:5]}")
    print(f">>> ntk_freqs[:5]\n{ntk_freqs[:5]}")
    scale = 8192 / 2048
    new_base = 10000.0 * (scale ** (dim / (dim - 2)))
    print(f"# New base = {new_base:.0f} (originally 10000), frequencies overall decrease, low-frequency dimensions change more")