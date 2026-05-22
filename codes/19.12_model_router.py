class ModelRouter:
    def __init__(self):
        self.models = {
            "simple": {"model": "gpt-4o-mini", "max_tokens": 500},
            "medium": {"model": "gpt-4o", "max_tokens": 2000},
            "complex": {"model": "deepseek-reasoner", "max_tokens": 10000},
        }
    
    def assess_complexity(self, prompt):
        if len(prompt) < 50 and "?" in prompt:
            return "simple"
        if any(kw in prompt for kw in ["reasoning", "prove", "why", "analyze"]):
            return "complex"
        return "medium"

if __name__ == "__main__":
    router = ModelRouter()
    print(router.assess_complexity("What time is it?"))
    print(router.assess_complexity("Please reason through the proof of this mathematical proposition"))
    print(router.assess_complexity("Please summarize the main points of this article about machine learning"))
    print(router.models[router.assess_complexity("What time is it?")]["model"])
    print(router.models[router.assess_complexity("Please reason through the proof of this mathematical proposition")]["model"])