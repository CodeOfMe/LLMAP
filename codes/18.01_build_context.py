def build_context(query, tools, memory, history):
    context = []
    
    # Beginning: most important system instructions and current task
    context.append({"role": "system", "content": system_prompt})
    context.append({"role": "user", "content": query})
    
    # Middle: conversation history and retrieved memories
    summarized_history = summarize_if_long(history, max_tokens=1000)
    context.extend(summarized_history)
    
    relevant_memories = memory.search(query, top_k=3)
    for mem in relevant_memories:
        context.append({"role": "system", "content": f"[Memory] {mem}"})
    
    # End: tool descriptions and final instruction
    context.append({"role": "system", "content": format_tools(tools)})
    context.append({"role": "system", "content": "Please answer the user's question based on the above information."})
    
    return context