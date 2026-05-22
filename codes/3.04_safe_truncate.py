import tiktoken

def safe_truncate(text, max_tokens, encoding_name="cl100k_base"):
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)
    while len(tokens) > max_tokens:
        # Remove tokens one by one until decoding produces valid Unicode
        tokens = tokens[:-1]
        decoded = enc.decode(tokens)
        if decoded == decoded.rstrip("\ufffd"):
            return decoded
    return enc.decode(tokens)