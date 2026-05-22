class KnowledgeGraphMemory:
    def __init__(self):
        self.triples = []

    def add(self, subject, predicate, obj):
        triple = (subject, predicate, obj)
        if triple not in self.triples:
            self.triples.append(triple)

    def query(self, subject=None, predicate=None, obj=None):
        results = []
        for s, p, o in self.triples:
            if (subject is None or s == subject) and \
               (predicate is None or p == predicate) and \
               (obj is None or o == obj):
                results.append((s, p, o))
        return results

    def get_related(self, entity, hops=2):
        visited = {entity}
        frontier = {entity}
        for _ in range(hops):
            next_frontier = set()
            for e in frontier:
                for s, p, o in self.triples:
                    if s == e and o not in visited:
                        next_frontier.add(o)
                        visited.add(o)
                    if o == e and s not in visited:
                        next_frontier.add(s)
                        visited.add(s)
            frontier = next_frontier
        return visited

if __name__ == "__main__":
    kg = KnowledgeGraphMemory()
    kg.add("Zhang San", "coworker", "Li Si")
    kg.add("Zhang San", "position", "Engineer")
    kg.add("Li Si", "coworker", "Wang Wu")
    kg.add("Li Si", "position", "Designer")

    print(kg.query(subject="Zhang San"))
    print(kg.get_related("Zhang San", hops=2))