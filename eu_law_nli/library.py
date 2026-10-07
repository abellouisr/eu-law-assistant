"""Search across every document in the library.

The engine answers from the selected document only. Before it does, it asks
the library which documents match the question best, so that it can tell the
user when another one probably holds more relevant material and offer to
switch to it.

All documents share one keyword index (their reference-language text, latest
version), so scores are comparable between documents. A document's score is
the average of its three best-matching provisions, summed over the search
queries, which keeps long documents from winning on size alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .corpus import Corpus
from .registry import Document, list_documents, load_document
from .retriever import BM25Retriever

TOP_PROVISIONS = 3


@dataclass
class ActMatch:
    document: str
    short_name: str
    score: float
    provisions: list[str] = field(default_factory=list)  # e.g. ["Article 83", "Recital 150"]
    provision_ids: list[str] = field(default_factory=list)  # e.g. ["art_83", "rct_150"]


class Library:
    def __init__(self, documents: list[str] | None = None) -> None:
        self.documents: dict[str, Document] = {}
        merged = []
        self._labels: dict[str, str] = {}
        for doc_id in documents if documents is not None else list_documents():
            doc = load_document(doc_id)
            version = doc.version("latest")
            try:
                corpus = Corpus.load(doc.id, version.id, doc.reference_language)
            except FileNotFoundError:
                continue  # registered but not ingested yet
            self.documents[doc.id] = doc
            for p in corpus.provisions:
                key = f"{doc.id}/{p.id}"
                merged.append(replace(p, id=key))
                self._labels[key] = f"Recital {p.number}" if p.kind == "recital" else p.label
        self._retriever = (BM25Retriever(Corpus("library", "", "", provisions=merged))
                           if merged else None)

    def __len__(self) -> int:
        return len(self.documents)

    def rank(self, queries: list[str], k: int = 40) -> list[ActMatch]:
        """Documents ordered by how well they match the queries, best first."""
        if self._retriever is None:
            return []
        totals: dict[str, float] = {}
        best_provisions: dict[str, dict[str, float]] = {}
        for query in (q for q in queries if q.strip()):
            per_doc: dict[str, dict[str, float]] = {}
            for hit in self._retriever.search(query, k=k):
                doc_id = hit.chunk.provision_id.split("/", 1)[0]
                scores = per_doc.setdefault(doc_id, {})
                pid = hit.chunk.provision_id
                scores[pid] = max(scores.get(pid, 0.0), hit.score)
            for doc_id, scores in per_doc.items():
                top = sorted(scores.values(), reverse=True)[:TOP_PROVISIONS]
                totals[doc_id] = totals.get(doc_id, 0.0) + sum(top) / TOP_PROVISIONS
                merged = best_provisions.setdefault(doc_id, {})
                for pid, score in scores.items():
                    merged[pid] = max(merged.get(pid, 0.0), score)
        ranked = []
        for doc_id, score in sorted(totals.items(), key=lambda kv: kv[1], reverse=True):
            pids = sorted(best_provisions[doc_id], key=best_provisions[doc_id].get,
                          reverse=True)[:TOP_PROVISIONS]
            ranked.append(ActMatch(doc_id, self.documents[doc_id].short_name, round(score, 2),
                                   [self._labels[p] for p in pids],
                                   [p.split("/", 1)[1] for p in pids]))
        return ranked
