"""Keyword retrieval (BM25) over the chunks of a corpus.

Pure Python, no model download, good enough for one directive. To move to
embeddings later, write another class with the same ``search`` method and
pass it to the engine.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from .corpus import Chunk, Corpus

STOPWORDS = set("""
a an and are as at be been by can do does for from has have how i if in into is it its
may me my of on or shall should such than that the their them there these they this
those to under was what when where which who whom will with would you your
""".split())


def tokenise(text: str) -> list[str]:
    tokens = []
    for raw in re.findall(r"[^\W_]+", text.lower()):
        if raw in STOPWORDS or (len(raw) == 1 and not raw.isdigit()):
            continue
        tokens.append(_stem(raw))
    return tokens


def _stem(word: str) -> str:
    """Very light English suffix stripping so 'authorities' meets 'authority'."""
    if len(word) > 5 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("es") and word[-3] in "sxz":
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


@dataclass
class Hit:
    chunk: Chunk
    score: float


class BM25Retriever:
    def __init__(self, corpus: Corpus, k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus = corpus
        self.chunks = corpus.chunks()
        self.k1, self.b = k1, b
        self._tf: list[Counter] = []
        self._len: list[int] = []
        postings: dict[str, list[int]] = defaultdict(list)
        for i, chunk in enumerate(self.chunks):
            p = corpus.get(chunk.provision_id)
            # The heading is indexed with each chunk so a question that uses
            # the article's title finds every part of that article.
            heading = f"{p.label} {p.title} {p.title}" if p else ""
            counts = Counter(tokenise(f"{heading} {chunk.text}"))
            self._tf.append(counts)
            self._len.append(sum(counts.values()))
            for term in counts:
                postings[term].append(i)
        self._postings = postings
        self._avg = (sum(self._len) / len(self._len)) if self._len else 0.0

    def search(self, query: str, k: int = 8) -> list[Hit]:
        n = len(self.chunks)
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenise(query)):
            docs = self._postings.get(term)
            if not docs:
                continue
            idf = math.log(1 + (n - len(docs) + 0.5) / (len(docs) + 0.5))
            for i in docs:
                tf = self._tf[i][term]
                norm = tf + self.k1 * (1 - self.b + self.b * self._len[i] / self._avg)
                scores[i] += idf * tf * (self.k1 + 1) / norm
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [Hit(self.chunks[i], s) for i, s in ranked]

    def search_many(self, queries: list[str], k: int = 8) -> list[Hit]:
        """Merge several queries with reciprocal-rank fusion."""
        fused: dict[tuple, float] = defaultdict(float)
        by_key: dict[tuple, Chunk] = {}
        for query in queries:
            for rank, hit in enumerate(self.search(query, k=k * 2)):
                key = (hit.chunk.provision_id, hit.chunk.start, hit.chunk.text[:40])
                fused[key] += 1.0 / (60 + rank)
                by_key[key] = hit.chunk
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [Hit(by_key[key], score) for key, score in ranked]

    def chunks_of(self, provision_id: str, limit: int = 3) -> list[Chunk]:
        return [c for c in self.chunks if c.provision_id == provision_id][:limit]
