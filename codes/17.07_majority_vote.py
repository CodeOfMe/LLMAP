from collections import Counter

def majority_vote(responses):
    votes = [r["answer"] for r in responses]
    result = Counter(votes).most_common(1)[0]
    return {"answer": result[0], "confidence": result[1] / len(votes)}

if __name__ == "__main__":
    responses = [
        {"agent": "A", "answer": "Python"},
        {"agent": "B", "answer": "Python"},
        {"agent": "C", "answer": "JavaScript"},
    ]
    print(majority_vote(responses))