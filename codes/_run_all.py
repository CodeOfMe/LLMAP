#!/usr/bin/env python3
"""
Smart runner for chapter code blocks. Adds necessary imports and test calls.
"""
import re
import os
import subprocess
import sys
import json

BASE_DIR = "/Users/fred/Documents/GitHub/Daily/ArticleWriting/Textbook/ConciseLargeLanguageModelTutorial"
CODES_DIR = os.path.join(BASE_DIR, "codes")
os.makedirs(CODES_DIR, exist_ok=True)

def run_code(label, code, timeout=30):
    """Run a Python code block and return stdout or None."""
    filepath = os.path.join(CODES_DIR, f"_test_{label}.py")
    with open(filepath, 'w') as f:
        f.write(code)
    try:
        result = subprocess.run(
            [sys.executable, filepath],
            capture_output=True, text=True, timeout=timeout
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        # Try with stderr too
        combined = (result.stdout or '') + (result.stderr or '')
        if result.returncode != 0 and combined.strip():
            return f"ERROR: {combined.strip()[:500]}"
        return None
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        return None

# === Chapter 4 ===
print("=== Chapter 4 ===")

# 4.1 remove_pii
code = '''import re

def remove_pii(text):
    text = re.sub(r\'\\b[\\w.-]+@[\\w.-]+\\.\\w+\\b\', \'[EMAIL]\', text)
    text = re.sub(r\'\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b\', \'[PHONE]\', text)
    text = re.sub(r\'\\b\\d{3}-\\d{2}-\\d{4}\\b\', \'[SSN]\', text)
    text = re.sub(r\'\\b\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\b\', \'[IP]\', text)
    return text

text = "Contact john@example.com or call 555-123-4567. SSN: 123-45-6789, IP: 192.168.1.1"
print(remove_pii(text))
'''
result = run_code("4.01_remove_pii", code)
print(f"4.01 remove_pii: {result}")

# 4.2 deduplicate
code = '''from datasketch import MinHash, MinHashLSH

def deduplicate(documents, num_perm=128, threshold=0.8):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    unique_docs = []
    
    for i, doc in enumerate(documents):
        mh = MinHash(num_perm=num_perm)
        for word in doc.split():
            mh.update(word.encode(\'utf8\'))
        
        if not lsh.query(mh):
            lsh.insert(f"doc_{i}", mh)
            unique_docs.append(doc)
    
    return unique_docs

docs = [
    "the cat sat on the mat",
    "the cat sat on the mat",  # duplicate
    "a dog ran in the park",
    "the cat sat on the mat",  # duplicate
    "a dog ran in the park",   # duplicate
    "birds fly in the sky",
]
result = deduplicate(docs)
print(f"Original: {len(docs)} docs")
print(f"After dedup: {len(result)} docs")
for doc in result:
    print(f"  - {doc}")
'''
result = run_code("4.02_deduplicate", code)
print(f"4.02 deduplicate: {result}")

# 4.3 quality_score
code = '''def quality_score(text):
    """Simple text quality scoring function"""
    score = 0
    
    # Appropriate length
    if 100 < len(text.split()) < 10000:
        score += 1
    
    # Vocabulary diversity
    words = text.lower().split()
    unique_ratio = len(set(words)) / max(len(words), 1)
    if 0.3 < unique_ratio < 0.8:
        score += 1
    
    # Sentence completeness
    sentences = text.split(\'.\')
    complete_ratio = sum(1 for s in sentences if len(s.split()) > 3) / max(len(sentences), 1)
    if complete_ratio > 0.5:
        score += 1
    
    # Low repetition rate
    bigrams = list(zip(words[:-1], words[1:]))
    repeat_ratio = len(set(bigrams)) / max(len(bigrams), 1)
    if repeat_ratio > 0.5:
        score += 1
    
    return score

text1 = "The quick brown fox jumps over the lazy dog. This sentence contains a variety of words with different lengths and meanings."
text2 = "the the the the the the the the the the the"
text3 = "Hi"

print(f"Good text score: {quality_score(text1)}")
print(f"Repetitive text score: {quality_score(text2)}")
print(f"Too short text score: {quality_score(text3)}")
'''
result = run_code("4.03_quality_score", code)
print(f"4.03 quality_score: {result}")

# === Chapter 5 ===
print("\n=== Chapter 5 ===")

# 5.1-5.5 transformer code (needs torch)
code = '''import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Token Embedding
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
    
    def forward(self, token_ids):
        return self.embedding(token_ids)

# Test
vocab_size, hidden_dim = 1000, 128
emb = TokenEmbedding(vocab_size, hidden_dim)
ids = torch.tensor([1, 50, 200])
out = emb(ids)
print(f"TokenEmbedding: input shape {ids.shape}, output shape {out.shape}")

# Output Head
class OutputHead(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, hidden_states):
        logits = self.linear(hidden_states)
        probabilities = F.softmax(logits, dim=-1)
        return probabilities

head = OutputHead(hidden_dim, vocab_size)
hidden = torch.randn(1, 10, hidden_dim)
probs = head(hidden)
print(f"OutputHead: input shape {hidden.shape}, output shape {probs.shape}")

# Weight Tying
class TransformerWithWeightTying(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True) for _ in range(num_layers)])
    
    def forward(self, token_ids):
        hidden = self.token_embedding(token_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        logits = F.linear(hidden, self.token_embedding.weight)
        return logits

model = TransformerWithWeightTying(vocab_size, hidden_dim, 2)
ids = torch.randint(0, vocab_size, (1, 10))
out = model(ids)
print(f"WeightTying: input shape {ids.shape}, output shape {out.shape}")

# Simple Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim)
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.ff(self.ln2(x))
        return x

block = TransformerBlock(hidden_dim, 4, hidden_dim * 4)
x = torch.randn(1, 10, hidden_dim)
out = block(x)
print(f"TransformerBlock: input shape {x.shape}, output shape {out.shape}")
'''
result = run_code("5.transformer", code, timeout=60)
print(f"5. transformer: {result}")

# === Chapter 6 ===
print("\n=== Chapter 6 ===")

code = '''import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, d_k):
    scores = Q @ K.T / math.sqrt(d_k)
    seq_len = scores.shape[0]
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores = scores.masked_fill(mask, float(\'-inf\'))
    attn_weights = F.softmax(scores, dim=-1)
    output = attn_weights @ V
    return output, attn_weights

# Test with small example
d_k = 4
Q = torch.randn(4, d_k)
K = torch.randn(4, d_k)
V = torch.randn(4, d_k)

output, weights = scaled_dot_product_attention(Q, K, V, d_k)
print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"Lower triangular (causal) check:")
print(f"  Row 0 sums to: {weights[0].sum().item():.4f} (should be ~1.0)")
print(f"  Upper triangle all zero: {(weights.triu(diagonal=1) == 0).all().item()}")
'''
result = run_code("6.attention", code)
print(f"6. attention: {result}")

# === Chapter 7 ===
print("\n=== Chapter 7 ===")

code = '''import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        B, S, _ = x.shape
        Q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill(mask, float(\'-inf\'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(attn_output)

# Test
mha = MultiHeadAttention(64, 4)
x = torch.randn(1, 10, 64)
out = mha(x)
print(f"MultiHeadAttention: input {x.shape} -> output {out.shape}")

# MQA
class MQA(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, self.head_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        B, S, _ = x.shape
        Q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, S, 1, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, 1, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(Q, K, V)
        attn = attn.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(attn)

mqa = MQA(64, 4)
out_mqa = mqa(x)
print(f"MQA: input {x.shape} -> output {out_mqa.shape}")

# GQA
class GQA(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        B, S, _ = x.shape
        Q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        K = K.repeat_interleave(self.num_groups, dim=1)
        V = V.repeat_interleave(self.num_groups, dim=1)
        attn = F.scaled_dot_product_attention(Q, K, V)
        attn = attn.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(attn)

gqa = GQA(64, 4, 2)
out_gqa = gqa(x)
print(f"GQA: input {x.shape} -> output {out_gqa.shape}")

# Parameter comparison
mha_params = sum(p.numel() for p in mha.parameters())
mqa_params = sum(p.numel() for p in mqa.parameters())
gqa_params = sum(p.numel() for p in gqa.parameters())
print(f"MHA params: {mha_params}, MQA params: {mqa_params}, GQA params: {gqa_params}")
'''
result = run_code("7.multihead", code, timeout=60)
print(f"7. multihead: {result}")

# === Chapter 8 ===
print("\n=== Chapter 8 ===")

code = '''def calculate_kv_cache_size(num_layers, num_kv_heads, head_dim, seq_length, batch_size=1, fp16=True):
    """Calculate KV Cache memory footprint"""
    bytes_per_element = 2 if fp16 else 4
    size = 2 * num_layers * num_kv_heads * head_dim * seq_length * bytes_per_element * batch_size
    return size

# Test with different configs
models = {
    "LLaMA 3 8B": {"layers": 32, "kv_heads": 8, "head_dim": 128},
    "LLaMA 2 70B": {"layers": 80, "kv_heads": 8, "head_dim": 128},
    "GPT-3 175B": {"layers": 96, "kv_heads": 96, "head_dim": 128},
}

for name, config in models.items():
    for seq_len in [2048, 4096, 8192, 32768]:
        size = calculate_kv_cache_size(
            config["layers"], config["kv_heads"], config["head_dim"], seq_len
        )
        print(f"{name} @ {seq_len}: {size / 1e9:.2f} GB")
'''
result = run_code("8.kv_cache_size", code)
print(f"8. kv_cache_size: {result}")

code = '''def estimate_serving_capacity(gpu_memory_gb, model_params_b, 
                              seq_length, num_kv_heads, head_dim, 
                              num_layers, fp16=True):
    """Estimate the number of concurrent requests a GPU can serve"""
    model_memory = model_params_b * (2 if fp16 else 4)
    
    kv_per_request = (2 * num_layers * num_kv_heads * head_dim * 
                       seq_length * (2 if fp16 else 4))
    kv_per_request_gb = kv_per_request / 1e9
    
    available = gpu_memory_gb - model_memory
    if available <= 0:
        return 0
    
    max_requests = int(available / kv_per_request_gb)
    return max_requests

# LLaMA 3 8B on A100 80GB
requests = estimate_serving_capacity(80, 8, 4096, 8, 128, 32)
print(f"LLaMA 3 8B on A100 80GB, 4096 context: max {requests} concurrent requests")

# With larger context
requests_8k = estimate_serving_capacity(80, 8, 8192, 8, 128, 32)
print(f"LLaMA 3 8B on A100 80GB, 8192 context: max {requests_8k} concurrent requests")
'''
result = run_code("8.serving_capacity", code)
print(f"8. serving_capacity: {result}")

# === Chapter 9 ===
print("\n=== Chapter 9 ===")

code = '''import torch
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

# Test
pe = sinusoidal_position_encoding(100, 64)
print(f"Sinusoidal PE shape: {pe.shape}")
print(f"Position 0, first 4 dims: {pe[0, :4].tolist()}")
print(f"Position 1, first 4 dims: {pe[1, :4].tolist()}")
print(f"Norm of position 0: {pe[0].norm():.4f}")
print(f"Norm of position 50: {pe[50].norm():.4f}")

# ALiBi
def alibi_bias(num_heads, seq_len, device=\'cpu\'):
    slopes = torch.tensor(
        [2 ** (-8 * i / num_heads) for i in range(num_heads)],
        device=device
    )
    positions = torch.arange(seq_len, device=device)
    distance = positions.unsqueeze(0) - positions.unsqueeze(1)
    distance = distance.abs().unsqueeze(0)
    alibi = slopes.unsqueeze(1).unsqueeze(1) * distance * (-1)
    return alibi

bias = alibi_bias(8, 16)
print(f"\\nALiBi bias shape: {bias.shape}")
print(f"Head 0 slope: {2**(-8*0/8):.6f}")
print(f"Head 7 slope: {2**(-8*7/8):.6f}")
'''
result = run_code("9.positional_encoding", code)
print(f"9. positional_encoding: {result}")

# === Chapter 10 ===
print("\n=== Chapter 10 ===")

code = '''import torch
import torch.nn as nn
import math

# LayerNorm vs RMSNorm
class LayerNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

class RMSNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x / rms)

# Test
x = torch.randn(2, 10, 64)
ln = LayerNorm(64)
rn = RMSNorm(64)
print(f"LayerNorm output shape: {ln(x).shape}")
print(f"LayerNorm output mean: {ln(x).mean(dim=-1)[0, 0].item():.6f} (should be ~0)")
print(f"RMSNorm output shape: {rn(x).shape}")

# SwiGLU
def swiglu(x, w1, w2, w3):
    gate = F.silu(w1(x))
    value = w3(x)
    return w2(gate * value)

# GELU
def gelu(x):
    return 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))

# Test activations
x_act = torch.linspace(-3, 3, 7)
print(f"\\nInput: {x_act.tolist()}")
print(f"ReLU(-2 to 2): {torch.relu(x_act).tolist()}")
print(f"GELU(-2 to 2): {gelu(x_act).tolist()}")

import torch.nn.functional as F
# Compare parameter counts
hidden_dim = 4096
ff_dim = 14336
# Standard FFN: 2 * hidden_dim * ff_dim
standard_ffn_params = 2 * hidden_dim * ff_dim
# SwiGLU FFN: 3 * hidden_dim * ff_dim
swiglu_ffn_params = 3 * hidden_dim * ff_dim
print(f"\\nStandard FFN params: {standard_ffn_params:,}")
print(f"SwiGLU FFN params: {swiglu_ffn_params:,}")
'''
result = run_code("10.norm_activation", code, timeout=30)
print(f"10. norm_activation: {result}")

print("\n=== Summary ===")
print("All runnable code blocks have been tested.")