def quality_score(text):
    """Simple text quality scoring function"""
    score = 0
    
    # Appropriate length (too short = insufficient info, too long = may be off-topic)
    if 100 < len(text.split()) < 10000:
        score += 1
    
    # Vocabulary diversity
    words = text.lower().split()
    unique_ratio = len(set(words)) / max(len(words), 1)
    if 0.3 < unique_ratio < 0.8:
        score += 1
    
    # Sentence completeness (ratio ending with period)
    sentences = text.split('.')
    complete_ratio = sum(1 for s in sentences if len(s.split()) > 3) / max(len(sentences), 1)
    if complete_ratio > 0.5:
        score += 1
    
    # Low repetition rate
    bigrams = list(zip(words[:-1], words[1:]))
    repeat_ratio = len(set(bigrams)) / max(len(bigrams), 1)
    if repeat_ratio > 0.5:
        score += 1
    
    return score

if __name__ == "__main__":
    text1 = "The quick brown fox jumps over the lazy dog. This sentence contains a variety of words with different lengths and meanings."
    text2 = "the the the the the the the the the the the"
    text3 = "Hi"
    
    print(f"Good text score: {quality_score(text1)}")
    print(f"Repetitive text score: {quality_score(text2)}")
    print(f"Too short text score: {quality_score(text3)}")