import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size=100, hidden_dim=32, num_heads=4, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, output_attentions=False):
        h = self.embedding(x)
        attn_weights_all = []
        for layer in self.layers:
            if output_attentions:
                h2, attn = layer.self_attn(h, h, h, need_weights=True, average_attn_weights=False)
                attn_weights_all.append(attn)
                h = layer.linear1(h)
                h = layer.linear2(h)
            else:
                h = layer(h)
        return h, attn_weights_all

def visualize_attention(token_labels, attn_weight, max_width=8):
    seq_len = len(token_labels)
    print("Attention weights (row=query position, column=attended position):")
    print(f"{'':>10}", end="")
    for label in token_labels:
        display = label[:max_width]
        print(f"{display:>10}", end="")
    print()
    for i, label in enumerate(token_labels):
        display = label[:max_width]
        print(f"{display:>10}", end="")
        for j in range(seq_len):
            print(f"{attn_weight[i,j].item():>10.3f}", end="")
        print()

if __name__ == "__main__":
    torch.manual_seed(42)
    model = MiniTransformer(vocab_size=100, hidden_dim=32, num_heads=4, num_layers=1)
    
    text = "Weekend trip to Hangzhou"
    token_labels = list(text)
    token_ids = torch.randint(0, 100, (1, len(token_labels)))
    
    with torch.no_grad():
        _, attns = model(token_ids, output_attentions=True)
    
    if len(attns) > 0:
        layer0_attn = attns[0].mean(dim=1)[0]
        print(f"Layer 0, averaged heads attention:")
        visualize_attention(token_labels, layer0_attn)
    else:
        print("No attention weights captured")