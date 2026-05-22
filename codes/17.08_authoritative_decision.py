def authoritative_decision(responses, authority):
    # Authority agent synthesizes all opinions and makes a final decision
    prompt = f"Here are opinions from multiple experts:\n"
    for r in responses:
        prompt += f"\n[{r['agent']}]: {r['content']}\n"
    prompt += f"\nAs the {authority}, please make the final decision."
    return call_llm(prompt)