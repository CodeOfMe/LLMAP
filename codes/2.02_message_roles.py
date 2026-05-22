if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are a Python debugging assistant. Keep answers concise, under 100 words."},
        {"role": "user", "content": "Why does this code throw a divide-by-zero error?\nresult = total / count"},
        {"role": "assistant", "content": "Because count might be 0. Add a check:\nif count != 0:\n    result = total / count"},
        {"role": "user", "content": "How do I wrap this code into a function?"},
    ]
    print("Message list:")
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        print(f"  role={role}: {content[:50]}...")
    print(f"Total {len(messages)} messages")