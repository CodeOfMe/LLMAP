import torch

if __name__ == "__main__":
    seq_len = 4
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    print(f"mask:\n{mask}")