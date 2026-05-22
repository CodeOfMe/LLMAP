class AgentMetrics:
    def __init__(self):
        self.request_count = 0
        self.success_count = 0
        self.total_tokens = 0
        self.total_cost = 0
        self.latencies = []
        self.tool_usage = {}
        self.errors = []
    
    def record_request(self, success, tokens, cost, latency, tools_used=None, error=None):
        self.request_count += 1
        if success:
            self.success_count += 1
        self.total_tokens += tokens
        self.total_cost += cost
        self.latencies.append(latency)
        
        for tool in (tools_used or []):
            self.tool_usage[tool] = self.tool_usage.get(tool, 0) + 1
        
        if error:
            self.errors.append(error)
    
    def get_dashboard(self):
        latencies = sorted(self.latencies)
        return {
            "success_rate": self.success_count / max(self.request_count, 1),
            "avg_latency": sum(self.latencies) / max(len(self.latencies), 1),
            "p99_latency": latencies[int(len(latencies) * 0.99)] if latencies else 0,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "cost_per_request": self.total_cost / max(self.request_count, 1),
            "tool_usage": self.tool_usage,
            "error_rate": len(self.errors) / max(self.request_count, 1),
        }

if __name__ == "__main__":
    import json
    metrics = AgentMetrics()
    metrics.record_request(True, 500, 0.01, 1.2, tools_used=["search"])
    metrics.record_request(True, 800, 0.02, 2.5, tools_used=["search", "calculator"])
    metrics.record_request(False, 300, 0.005, 0.8, error="timeout")
    print(json.dumps(metrics.get_dashboard(), indent=2))