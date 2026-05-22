import torch

if __name__ == "__main__":
    torch.manual_seed(42)
    seq_len = 4
    d_k = 2
    
    Q = torch.randn(seq_len, d_k)
    K = torch.randn(seq_len, d_k)
    scores = Q @ K.T
    
    print(f"scores shape: {scores.shape}")
    print(f"scores:\n{scores}")