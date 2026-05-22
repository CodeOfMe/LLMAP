def evaluate_component(component_name, test_cases):
    """Component-level evaluation"""
    results = []
    for case in test_cases:
        if component_name == "prompt":
            output = call_llm(case["input"])
            score = case["evaluator"](output, case["expected"])
        elif component_name == "retrieval":
            docs = retrieve(case["query"])
            score = case["evaluator"](docs, case["expected_docs"])
        elif component_name == "tool_call":
            result = call_tool(case["tool"], case["args"])
            score = case["evaluator"](result, case["expected_result"])
        
        results.append({"case": case, "score": score})
    
    return {
        "component": component_name,
        "avg_score": sum(r["score"] for r in results) / len(results),
        "min_score": min(r["score"] for r in results),
        "results": results
    }