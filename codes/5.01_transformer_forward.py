import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimpleTokenizer:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.char_to_id = {}
        self.id_to_char = {}
        idx = 0
        for i in range(2048):
            c = chr(i)
            if c.isprintable() and idx < vocab_size:
                self.char_to_id[c] = idx
                self.id_to_char[idx] = str(c)
                idx += 1
    
    def encode(self, text):
        return [self.char_to_id.get(c, 0) for c in text]
    
    def decode(self, token_id):
        return self.id_to_char.get(token_id, "<unk>")

class SimpleModel:
    def __init__(self, vocab_size=100, hidden_dim=64, num_layers=1, max_seq_len=32):
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
            for _ in range(num_layers)
        ])
        self.output_head = nn.Linear(hidden_dim, vocab_size)

def transformer_forward(tokens, model, tokenizer):
    token_ids = tokenizer.encode(tokens)
    token_ids_t = torch.tensor([token_ids])
    embeddings = model.token_embedding(token_ids_t)
    seq_len = token_ids_t.shape[1]
    positions = torch.arange(seq_len)
    embeddings = embeddings + model.position_embedding(positions)
    hidden = embeddings
    for layer in model.layers:
        hidden = layer(hidden)
    logits = model.output_head(hidden)
    next_token_logits = logits[0, -1]
    probabilities = F.softmax(next_token_logits, dim=-1)
    next_token = torch.argmax(probabilities).item()
    return tokenizer.decode(next_token)

if __name__ == "__main__":
    torch.manual_seed(42)
    vocab_size = 100
    hidden_dim = 64
    num_layers = 1
    
    tokenizer = SimpleTokenizer(vocab_size)
    model = SimpleModel(vocab_size, hidden_dim, num_layers)
    
    text = "Weekend trip to Hangzhou"
    token_ids = tokenizer.encode(text)
    token_ids_t = torch.tensor([token_ids])
    
    embeddings = model.token_embedding(token_ids_t)
    seq_len = token_ids_t.shape[1]
    positions = torch.arange(seq_len)
    embeddings = embeddings + model.position_embedding(positions)
    
    hidden = embeddings
    for layer in model.layers:
        hidden = layer(hidden)
    
    logits = model.output_head(hidden)
    
    next_token_logits = logits[0, -1]
    probabilities = F.softmax(next_token_logits, dim=-1)
    next_token = torch.argmax(probabilities).item()
    top5_probs, top5_ids = probabilities.topk(5)
    
    print(f"Input text: \"{text}\"")
    print(f"Tokenized: {[tokenizer.decode(t) for t in token_ids]}")
    print(f"Number of tokens: {len(token_ids)}")
    print(f"After token embedding: shape = {embeddings.shape}")
    print(f"After position embedding: shape = {embeddings.shape}")
    print(f"After 1 transformer layer: shape = {hidden.shape}")
    print(f"After output head: logits shape = {logits.shape}")
    print(f"Next token prediction: token_id = {next_token}")
    print(f"Top-5 tokens: {top5_ids.tolist()}")
    print(f"Top-5 tokens: {top5_ids.tolist()}")
    print(f"Top-5 probabilities: {[f'{p:.1f}' for p in top5_probs.tolist()]}")