import tiktoken
from transformers import AutoTokenizer

qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
text = "你好世界"
print(qwen_tokenizer.encode(text))
print(qwen_tokenizer.decode(qwen_tokenizer.encode(text)))

gpt_tokenizer = tiktoken.get_encoding("cl100k_base")
print(gpt_tokenizer.encode(text))