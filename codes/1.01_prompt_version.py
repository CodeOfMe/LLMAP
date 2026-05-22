import time

class PromptVersion:
    def __init__(self):
        self.versions = {}
    
    def register(self, name, prompt, metrics=None):
        if name not in self.versions:
            self.versions[name] = []
        version_num = len(self.versions[name]) + 1
        self.versions[name].append({
            "version": version_num,
            "prompt": prompt,
            "metrics": metrics or {},
            "timestamp": time.time()
        })
        return version_num
    
    def get_latest(self, name):
        return self.versions[name][-1]["prompt"]
    
    def get(self, name, version):
        return self.versions[name][version - 1]["prompt"]
    
    def compare(self, name, v1, v2):
        m1 = self.versions[name][v1 - 1]["metrics"]
        m2 = self.versions[name][v2 - 1]["metrics"]
        return {"v1": m1, "v2": m2, "diff": {k: m2.get(k,0) - m1.get(k,0) for k in m1}}

if __name__ == "__main__":
    pv = PromptVersion()
    v1 = pv.register("code_review", "Review this code", metrics={"accuracy": 0.6, "latency": 1.2})
    print(f"Registered version: v{v1}")
    v2 = pv.register("code_review", "Review this Python code for quality, security, and performance", metrics={"accuracy": 0.75, "latency": 1.0})
    print(f"Registered version: v{v2}")
    v3 = pv.register("code_review", "You are a Python engineer with 10 years of experience, review code for quality, security, and performance", metrics={"accuracy": 0.92, "latency": 0.8})
    print(f"Registered version: v{v3}")
    print(f"Latest prompt: {pv.get_latest('code_review')}")
    print(f"Version 1 prompt: {pv.get('code_review', 1)}")
    print(f"Version comparison v1 vs v3: {pv.compare('code_review', 1, 3)}")