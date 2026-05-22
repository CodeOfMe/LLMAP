import math

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a * norm_b > 0 else 0

if __name__ == "__main__":
    a = [1.0, 2.0, 3.0]
    b = [1.0, 2.0, 3.0]
    c = [3.0, 2.0, 1.0]
    print(f"Identical vectors: {cosine_similarity(a, b):.4f}")
    print(f"Different vectors: {cosine_similarity(a, c):.4f}")
    print(f"Orthogonal vectors: {cosine_similarity([1,0], [0,1]):.4f}")