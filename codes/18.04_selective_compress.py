def selective_compress(messages, importance_threshold=0.7):
    """Keep or remove messages based on importance score"""
    compressed = []
    for msg in messages:
        if msg["role"] == "system":
            compressed.append(msg)  # System messages are always kept
            continue
        
        importance = score_importance(msg["content"])
        if importance >= importance_threshold:
            compressed.append(msg)
    
    return compressed

def score_importance(text):
    """Evaluate message importance"""
    prompt = f"""Evaluate the importance of the following text (0 to 1):
- 1: Contains key decisions or facts
- 0.5: Contains useful but non-critical information
- 0: Casual chat or irrelevant content

Text: {text}"""
    return float(call_llm(prompt))