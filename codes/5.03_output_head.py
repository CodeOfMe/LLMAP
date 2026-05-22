import torch
import torch.nn as nn
import torch.nn.functional as F

class OutputHead(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, hidden_states):
        logits = self.linear(hidden_states)
        probabilities = F.softmax(logits, dim=-1)
        return probabilities

if __name__ == "__main__":
    hidden_dim = 4096
    vocab_size = 128256
    head = OutputHead(hidden_dim, vocab_size)
    hidden_states = torch.randn(1, 5, hidden_dim)
    probs = head(hidden_states)
    print(f"Input shape: {hidden_states.shape}")
    print(f"Output probabilities shape: {probs.shape}")
    print(f"Probabilities sum: {probs[0, 0].sum().item():.6f}")
    print(f"Parameters: {head.linear.weight.numel() + head.linear.bias.numel():,}")