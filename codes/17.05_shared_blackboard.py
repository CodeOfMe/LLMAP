import time

class SharedBlackboard:
    def __init__(self):
        self.sections = {
            "requirements": [],
            "design": [],
            "code": [],
            "review": []
        }
    
    def write(self, section, agent_name, content):
        self.sections[section].append({
            "author": agent_name,
            "content": content,
            "timestamp": time.time()
        })
    
    def read(self, section=None):
        if section:
            return self.sections[section]
        return self.sections

if __name__ == "__main__":
    bb = SharedBlackboard()
    bb.write("requirements", "PM", "Users need login functionality")
    bb.write("design", "Designer", "Login page prototype design")
    bb.write("code", "Coder", "login() function implementation")
    bb.write("review", "Reviewer", "Security needs improvement")
    for section, items in bb.read().items():
        if items:
            print(f"{section}: {[i['author']+':'+i['content'] for i in items]}")