class TokenBudget:
    def __init__(self, daily_budget=100000, per_request_budget=5000):
        self.daily_budget = daily_budget
        self.per_request_budget = per_request_budget
        self.daily_used = 0
        self.request_used = {}
    
    def can_afford(self, request_id, estimated_tokens):
        if self.daily_used + estimated_tokens > self.daily_budget:
            return False
        if estimated_tokens > self.per_request_budget:
            return False
        return True
    
    def spend(self, request_id, tokens):
        self.daily_used += tokens
        self.request_used[request_id] = tokens

if __name__ == "__main__":
    budget = TokenBudget(daily_budget=10000, per_request_budget=3000)
    print(budget.can_afford("req1", 2000))
    print(budget.can_afford("req2", 5000))
    budget.spend("req1", 2000)
    print(budget.daily_used)
    print(budget.request_used)