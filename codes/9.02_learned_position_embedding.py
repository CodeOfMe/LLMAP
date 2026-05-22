import torch
import torch.nn as nn

class LearnedPositionEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        return x + self.embedding(positions)

if __name__ == "__main__":
    torch.manual_seed(42)
    model = LearnedPositionEmbedding(max_len=8, d_model=4)
    x = torch.randn(2, 5, 4)
    output = model(x)
    print(f">>> model = LearnedPositionEmbedding(max_len=8, d_model=4)")
    print(f">>> x = torch.randn(2, 5, 4)")
    print(f">>> output = model(x)")
    print(f">>> output.shape\n{output.shape}")
    print(f">>> model.embedding.weight[:5]\n{model.embedding.weight[:5]}")