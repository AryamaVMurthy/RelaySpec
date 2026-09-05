"""Bounded-memory lexical exclusion of a small evaluation candidate pool."""

from collections import defaultdict

from relayspec.scaling_data import content_hash, shingles


class CandidateOverlapIndex:
    """Return every matching candidate, including duplicates and short texts."""

    def __init__(self, texts, threshold=0.6):
        if not 0 < threshold <= 1:
            raise ValueError("overlap threshold must lie in (0, 1]")
        self.threshold = threshold
        self.exact = defaultdict(set)
        self.grams = []
        self.inverted = defaultdict(list)
        for index, text in enumerate(texts):
            self.exact[content_hash(text)].add(index)
            grams = shingles(text)
            self.grams.append(grams)
            if len(grams) >= 10:
                for gram in grams:
                    self.inverted[gram].append(index)

    def matches(self, text):
        result = {index: "exact" for index in self.exact.get(content_hash(text), ())}
        grams = shingles(text)
        if len(grams) < 10:
            return result
        counts = defaultdict(int)
        for gram in grams:
            for index in self.inverted.get(gram, ()):
                counts[index] += 1
        for index, intersection in counts.items():
            union = len(grams) + len(self.grams[index]) - intersection
            if intersection / union >= self.threshold:
                result.setdefault(index, "near")
        return result
