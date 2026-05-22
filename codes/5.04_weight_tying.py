import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
    
    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        return self.ff(attn_out)

class TransformerWithWeightTying(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([TransformerLayer(hidden_dim, 4) for _ in range(num_layers)])
    
    def forward(self, token_ids):
        hidden = self.token_embedding(token_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        logits = F.linear(hidden, self.token_embedding.weight)
        return logits

if __name__ == "__main__":
    vocab_size = 1000
    hidden_dim = 64
    num_layers = 2
    
    model_tied = TransformerWithWeightTying(vocab_size, hidden_dim, num_layers)
    tied_params = sum(p.numel() for p in model_tied.parameters())
    
    class TransformerWithoutWeightTying(nn.Module):
        def __init__(self, vocab_size, hidden_dim, num_layers):
            super().__init__()
            self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
            self.layers = nn.ModuleList([TransformerLayer(hidden_dim, 4) for _ in range(num_layers)])
            self.output_head = nn.Linear(hidden_dim, vocab_size)
        
        def forward(self, token_ids):
            hidden = self.token_embedding(token_ids)
            for layer in self.layers:
                hidden = layer(hidden)
            return self.output_head(hidden)
    
    model_untied = TransformerWithoutWeightTying(vocab_size, hidden_dim, num_layers)
    untied_params = sum(p.numel() for p in model_untied.parameters())
    
    print(f"Parameters WITH weight tying: {tied_params:,}")
    print(f"Parameters WITHOUT weight tying: {untied_params:,}")
    saved = untied_params - tied_params
    print(f"Parameters saved: {saved:,} ({saved/untied_params*100:.1f}%)")
    
    token_ids = torch.tensor([[1, 50, 300]])
    logits = model_tied(token_ids)
    print(f"Input shape: {token_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Output logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")