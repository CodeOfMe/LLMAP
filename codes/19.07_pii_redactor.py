import re

class PIIRedactor:
    def __init__(self):
        self.patterns = {
            "phone": (r'\b1[3-9]\d{9}\b', '[PHONE]'),
            "email": (r'\b[\w.-]+@[\w.-]+\.\w+\b', '[EMAIL]'),
            "id_card": (r'(?<!\d)\d{17}[\dXx](?!\d)', '[ID_CARD]'),
            "bank_card": (r'\b\d{16,19}\b', '[BANK_CARD]'),
        }
    
    def redact(self, text):
        for pii_type, (pattern, replacement) in self.patterns.items():
            text = re.sub(pattern, replacement, text)
        return text
    
    def restore(self, redacted_text, original_text):
        pass

if __name__ == "__main__":
    redactor = PIIRedactor()
    print(redactor.redact("Contact email test@example.com"))
    print(redactor.redact("ID number 110101199001011234"))