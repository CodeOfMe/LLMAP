import torch
import math

def sinusoidal_position_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def alibi_bias(num_heads, seq_len, device='cpu'):
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
    pe = sinusoidal_position_encoding(100, 64)
    print(f"Sinusoidal PE shape: {pe.shape}")
    print(f"Position 0, first 4 dims: {pe[0, :4].tolist()}")
    print(f"Position 1, first 4 dims: {pe[1, :4].tolist()}")
    print(f"Norm of position 0: {pe[0].norm():.4f}")
    print(f"Norm of position 50: {pe[50].norm():.4f}")

    bias = alibi_bias(8, 16)
    print(f"\nALiBi bias shape: {bias.shape}")
    print(f"Head 0 slope: {2**(-8*0/8):.6f}")
    print(f"Head 7 slope: {2**(-8*7/8):.6f}")