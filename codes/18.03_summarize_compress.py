def summarize_compress(messages, max_recent=4):
    """Keep the most recent conversation turns, summarize earlier ones"""
    recent = messages[-max_recent:]
    old = messages[:-max_recent]
    
    if not old:
        return messages
    
    old_text = "\n".join(f"{m['role']}: {m['content']}" for m in old)
    summary = call_llm(f"Summarize the key information from this conversation in 100 words:\n{old_text}")
    
    return [
        {"role": "system", "content": f"Previous conversation summary: {summary}"},
        *recent
    ]