import torch

def alibi_bias(num_heads, seq_len, device):
    slopes = torch.tensor(
        [2 ** (-8 * i / num_heads) for i in range(num_heads)],
        device=device
    )
    positions = torch.arange(seq_len, device=device)
    distance = positions.unsqueeze(0) - positions.unsqueeze(1)
    distance = distance.abs().unsqueeze(0)
    alibi = slopes.unsqueeze(1).unsqueeze(1) * distance * (-1)
    return alibi

if __name__ == "__main__":
    bias = alibi_bias(num_heads=8, seq_len=6, device='cpu')
    print(f">>> bias = alibi_bias(num_heads=8, seq_len=6, device='cpu')")
    print(f">>> bias.shape\n{bias.shape}")
    print(f"# Head 0 has the steepest slope ({2**(-8*0/8):.4f}), sharpest decay at close range")
    print(f">>> bias[0]\n{bias[0]}")
    print(f"# Head 7 has the gentlest slope ({2**(-8*7/8):.4f}), slowest decay")
    print(f">>> bias[7]\n{bias[7]}")