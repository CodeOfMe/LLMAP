import re

def tokenize_with_number_fix(text, tokenizer, min_digits=3):
    """Keep longer numbers as single tokens"""
    tokens = []
    last_end = 0
    for match in re.finditer(r'\d+', text):
        if match.end() - match.start() >= min_digits:
            tokens.extend(tokenizer.encode(text[last_end:match.start()]))
            tokens.append(match.group())  # Keep number as a whole
            last_end = match.end()
    tokens.extend(tokenizer.encode(text[last_end:]))
    return tokens