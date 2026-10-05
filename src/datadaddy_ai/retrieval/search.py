"""Doc-level search facade over a retriever, with metadata filters (used by the agent tool and the API)."""

from __future__ import annotations

from .corpus import Card, Chunk, chunk_corpus
from .index import BM25Index, DenseIndex, HashEmbedder, HybridIndex, to_docs


class DatasetSearch:
    def __init__(self, cards: list[Card], retriever=None, max_words: int = 150, overlap: int = 25):
        self.cards = {c.id: c for c in cards}
        self.chunks: list[Chunk] = chunk_corpus(cards, max_words, overlap)
        self.chunk_doc = [c.doc_id for c in self.chunks]
        self.retriever = retriever or BM25Index([c.text for c in self.chunks])

    @classmethod
    def hybrid(cls, cards, embedder=None, max_words=150, overlap=25):
        s = cls(cards, None, max_words, overlap)
        texts = [c.text for c in s.chunks]
        s.retriever = HybridIndex(BM25Index(texts), DenseIndex(embedder or HashEmbedder(), texts))
        return s

    @staticmethod
    def _match(card: Card, filters: dict) -> bool:
        def has(vals, want):
            return want is None or want.lower() in [v.lower() for v in vals]

        return (
            has(card.task_categories, filters.get("task"))
            and has(card.languages, filters.get("language"))
            and has(card.license, filters.get("license"))
            and has(card.size_categories, filters.get("size"))
        )

    def search(self, query: str, k: int = 5, filters: dict | None = None) -> list[dict]:
        filters = {a: b for a, b in (filters or {}).items() if b}
        ranked = self.retriever.search(query, 200 if filters else max(k * 8, 50))
        out = []
        for doc, score in to_docs(ranked, self.chunk_doc, 10**6):
            c = self.cards[doc]
            if filters and not self._match(c, filters):
                continue
            out.append(
                {
                    "id": doc,
                    "score": round(score, 5),
                    "tasks": c.task_categories,
                    "languages": c.languages,
                    "license": c.license,
                }
            )
            if len(out) >= k:
                break
        return out
