import re

class OutputGuardrail:
    def __init__(self):
        self.checks = [
            self.pii_check,
            self.toxicity_check,
        ]
    
    def check(self, output, context=None):
        results = {}
        for check_fn in self.checks:
            name = check_fn.__name__
            passed, message = check_fn(output, context)
            results[name] = {"passed": passed, "message": message}
            if not passed:
                break
        all_passed = all(r["passed"] for r in results.values())
        return {"approved": all_passed, "details": results}
    
    def pii_check(self, output, context=None):
        pii_patterns = {
            "phone": r'\b1[3-9]\d{9}\b',
            "email": r'\b[\w.-]+@[\w.-]+\.\w+\b',
            "id_card": r'\b\d{17}[\dXx]\b',
        }
        for pii_type, pattern in pii_patterns.items():
            if re.search(pattern, output):
                return False, f"Detected {pii_type} type of private information"
        return True, "No private information detected"
    
    def toxicity_check(self, output, context=None):
        toxic_keywords = ["violence", "hate", "discrimination"]
        for keyword in toxic_keywords:
            if keyword in output:
                return False, f"Detected harmful content: {keyword}"
        return True, "No harmful content detected"

if __name__ == "__main__":
    guardrail = OutputGuardrail()
    print(guardrail.check("Contact: test@example.com"))
    print(guardrail.check("This article contains violence"))
    print(guardrail.check("Python is an excellent programming language"))