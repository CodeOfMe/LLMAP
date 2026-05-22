import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_autoregressive(model, prompt_ids, max_new_tokens=100):
    logits = model(prompt_ids)
    next_token = torch.argmax(logits[:, -1, :], dim=-1)
    generated = [next_token.item()]
    
    all_ids = torch.cat([prompt_ids, next_token.view(1, 1)], dim=1)
    
    for _ in range(max_new_tokens - 1):
        logits = model(all_ids)
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        generated.append(next_token.item())
        all_ids = torch.cat([all_ids, next_token.view(1, 1)], dim=1)
        
        if all_ids.shape[1] >= 32:
            break
    
    return generated

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=50, hidden_dim=32, num_heads=4, max_seq_len=32):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output_head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        B, S = x.shape
        positions = torch.arange(S, device=x.device)
        h = self.token_embedding(x) + self.pos_embedding(positions)
        h = self.transformer(h)
        return self.output_head(h)

if __name__ == "__main__":
    torch.manual_seed(42)
    model = TinyTransformer(vocab_size=50, hidden_dim=32, num_heads=4)
    prompt = torch.tensor([[5, 10, 15, 20]])
    
    print(f"Prompt token IDs: {prompt.tolist()[0]}")
    print(f"Generating (autoregressive - recompute all tokens each step)...")
    result = generate_autoregressive(model, prompt, max_new_tokens=8)
    print(f"Generated token IDs: {result}")
    print(f"Total tokens generated: {len(result)}")
    print()
    print("Note: This code is a simplified illustration. A real KV Cache implementation would")
    print("cache computed Keys and Values during the Decode phase, computing KV for only the new token at each step, avoiding redundant computation.")