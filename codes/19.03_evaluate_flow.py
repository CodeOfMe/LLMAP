def evaluate_flow(flow_name, scenarios):
    """Flow-level evaluation"""
    results = []
    for scenario in scenarios:
        agent = create_agent(flow_name)
        output = agent.run(scenario["task"])
        
        results.append({
            "scenario": scenario["name"],
            "success": scenario["validator"](output),
            "steps": agent.step_count,
            "tokens": agent.total_tokens,
            "time": agent.elapsed_time
        })
    
    return {
        "flow": flow_name,
        "success_rate": sum(r["success"] for r in results) / len(results),
        "avg_steps": sum(r["steps"] for r in results) / len(results),
        "avg_tokens": sum(r["tokens"] for r in results) / len(results),
    }