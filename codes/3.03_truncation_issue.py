import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

text = "Hello, 世界！"
tokens = enc.encode(text)
print(tokens)  # Actual output: [9906, 11, 220, 3574, 244, 98220, 6447]

# Truncate to 3 tokens
truncated = enc.decode(tokens[:3])
print(repr(truncated))  # Actual output: 'Hello, ' — note truncation landed on a space!