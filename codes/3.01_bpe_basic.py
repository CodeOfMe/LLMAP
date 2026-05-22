import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

text = "unfriendly"
tokens = enc.encode(text)
print(tokens)  # Actual output: [359, 82630]
decoded = [enc.decode([t]) for t in tokens]
print(decoded)  # Actual output: ['un', 'friendly']