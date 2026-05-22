from datasketch import MinHash, MinHashLSH

def deduplicate(documents, num_perm=128, threshold=0.8):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    unique_docs = []
    
    for i, doc in enumerate(documents):
        mh = MinHash(num_perm=num_perm)
        for word in doc.split():
            mh.update(word.encode('utf8'))
        
        if not lsh.query(mh):
            lsh.insert(f"doc_{i}", mh)
            unique_docs.append(doc)
    
    return unique_docs

if __name__ == "__main__":
    docs = [
        "the cat sat on the mat",
        "the cat sat on the mat",
        "a dog ran in the park",
        "the cat sat on the mat",
        "a dog ran in the park",
        "birds fly in the sky",
    ]
    result = deduplicate(docs)
    print(f"Original: {len(docs)} docs")
    print(f"After dedup: {len(result)} docs")
    for doc in result:
        print(f"  - {doc}")