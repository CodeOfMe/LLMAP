class ModelLoadBalancer:
    def __init__(self):
        self.models = [
            {"name": "gpt-4o", "tier": "premium", "cost_per_1k": 0.005},
            {"name": "gpt-4o-mini", "tier": "standard", "cost_per_1k": 0.00015},
            {"name": "claude-3-haiku", "tier": "standard", "cost_per_1k": 0.00025},
        ]
    
    def route(self, request):
        complexity = self.estimate_complexity(request)
        
        if complexity == "simple":
            return self.models[1]
        elif complexity == "moderate":
            return self.models[2]
        else:
            return self.models[0]
    
    def estimate_complexity(self, request):
        prompt = request.get("prompt", "")
        if len(prompt) < 100:
            return "simple"
        if any(kw in prompt for kw in ["analyze", "reason", "explain principles"]):
            return "complex"
        return "moderate"

if __name__ == "__main__":
    lb = ModelLoadBalancer()
    print(lb.route({"prompt": "Hello"})["name"])
    print(lb.route({"prompt": "Please analyze the performance bottlenecks of this code and provide optimization suggestions"})["name"])
    print(lb.route({"prompt": "This is a moderately long descriptive text that exceeds 100 characters but does not contain keywords like analyze or reason"})["name"])