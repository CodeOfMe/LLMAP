import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

print(len(enc.encode("I love programming")))   # Actual output: 3
print(len(enc.encode("I love programming!")))  # Actual output: 4 — adding an exclamation mark adds 1 token
print(len(enc.encode("I love programming.")))  # Actual output: 4 — adding a period also adds 1 token
print(len(enc.encode("Iloveprogramming")))     # Actual output: 3 — removing spaces still 3 tokens