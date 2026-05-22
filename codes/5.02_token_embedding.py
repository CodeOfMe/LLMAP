import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
    
    def forward(self, token_ids):
        return self.embedding(token_ids)

if __name__ == "__main__":
    vocab_size = 128256
    hidden_dim = 4096
    emb = TokenEmbedding(vocab_size, hidden_dim)
    token_ids = torch.tensor([[1, 50, 300]])
    output = emb(token_ids)
    print(f"Input shape: {token_ids.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Embedding matrix shape: {emb.embedding.weight.shape}")
    print(f"Parameters: {emb.embedding.weight.numel():,}")