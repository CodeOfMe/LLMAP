import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

texts = {
    "English": "The meeting starts at noon.",
    "Chinese": "会议中午开始。",
    "Japanese": "会議は正午から始まります。",
}

for lang, text in texts.items():
    tokens = enc.encode(text)
    chars = len(text)
    ratio = len(tokens) / chars
    print(f"{lang}: {chars} chars → {len(tokens)} tokens, ratio = {ratio:.2f}")

# Actual running results:
# English: 27 chars → 6 tokens, ratio = 0.22
# Chinese: 7 chars → 6 tokens, ratio = 0.86
# Japanese: 13 chars → 12 tokens, ratio = 0.92