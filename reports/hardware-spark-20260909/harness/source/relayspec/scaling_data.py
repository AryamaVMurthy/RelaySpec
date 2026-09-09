"""Deterministic nested data and explicit lexical overlap exclusions."""

import hashlib
import re
from collections import defaultdict


def normalize_text(text):
    return " ".join(str(text).lower().split())


def content_hash(text):
    return hashlib.sha256(normalize_text(text).encode()).hexdigest()


def shingles(text, width=5):
    tokens = re.findall(r"\w+|[^\w\s]", normalize_text(text))
    return {tuple(tokens[i : i + width]) for i in range(len(tokens) - width + 1)}


class OverlapIndex:
    """Exact normalized text and complete inverted-index 5-shingle Jaccard search."""

    def __init__(self, threshold=0.6):
        if not 0 < threshold <= 1:
            raise ValueError("overlap threshold must lie in (0, 1]")
        self.threshold = threshold
        self.exact = set()
        self.sets = []
        self.inverted = defaultdict(list)

    def add(self, text):
        self.exact.add(content_hash(text))
        grams = shingles(text)
        if len(grams) < 10:
            return
        i = len(self.sets)
        self.sets.append(grams)
        for gram in grams:
            self.inverted[gram].append(i)

    def match(self, text):
        if content_hash(text) in self.exact:
            return "exact"
        grams = shingles(text)
        if len(grams) < 10:
            return None
        counts = defaultdict(int)
        for gram in grams:
            for i in self.inverted.get(gram, ()):
                counts[i] += 1
        for i, intersection in counts.items():
            union = len(grams) + len(self.sets[i]) - intersection
            if intersection / union >= self.threshold:
                return "near"
        return None


def stratified_order(rows, seed):
    """Weighted round robin over independently hash-shuffled source strata."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["source"]].append(row)
    for group in groups.values():
        group.sort(
            key=lambda r: hashlib.sha256(
                f"{seed}:{r['problem']}:{r['solution']}".encode()
            ).hexdigest()
        )
    total = len(rows)
    used = {k: 0 for k in groups}
    for index in range(total):
        active = [k for k in groups if used[k] < len(groups[k])]
        name = max(
            active, key=lambda k: ((index + 1) * len(groups[k]) / total - used[k], k)
        )
        yield groups[name][used[name]]
        used[name] += 1
