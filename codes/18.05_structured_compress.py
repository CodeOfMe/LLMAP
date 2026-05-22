def structured_compress(messages):
    """Compress free-text conversations into structured records"""
    records = []
    for msg in messages:
        if msg["role"] == "tool":
            # Compress tool results into a one-line summary
            records.append(f"[Tool {msg['tool_call_id']} result] {msg['content'][:100]}")
        elif msg["role"] == "assistant":
            # Compress assistant responses into key points
            key_points = extract_key_points(msg["content"])
            records.append(f"[Assistant] {key_points}")
    
    return "\n".join(records)