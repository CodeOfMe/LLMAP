import re

trigger_patterns = [
    r"look\s*up.*(?:data|records|info)",
    r"(?:is\s*there|got)\s+(?:account|user|order)",
]

if __name__ == "__main__":
    test_cases = [
        ("look up order data", trigger_patterns[0]),
        ("how's the weather today", trigger_patterns[0]),
        ("look up order data", trigger_patterns[1]),
        ("how's the weather today", trigger_patterns[1]),
    ]
    for text, pattern in test_cases:
        match = re.search(pattern, text)
        result = "match" if match else "no match"
        print(f'Pattern: {pattern}')
        print(f'  "{text}" -> {result}')