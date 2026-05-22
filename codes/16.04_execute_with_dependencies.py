def execute_with_dependencies(plan, subagents):
    completed = {}
    remaining = plan["subtasks"][:]
    
    while remaining:
        # Find all subtasks whose dependencies are satisfied
        ready = [t for t in remaining 
                 if all(d in completed for d in t["depends_on"])]
        
        if not ready:
            raise Exception("Circular dependency! Cannot continue execution.")
        
        # Execute all ready subtasks in parallel
        with ThreadPoolExecutor() as executor:
            futures = {}
            for task in ready:
                agent = subagents[task["agent_type"]]
                futures[task["id"]] = executor.submit(
                    agent.run, task["description"]
                )
            
            for task_id, future in futures.items():
                completed[task_id] = future.result()
                remaining.remove(next(t for t in remaining if t["id"] == task_id))
    
    return completed