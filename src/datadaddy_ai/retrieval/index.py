"""BM25, dense (FAISS) and hybrid (RRF) chunk retrievers + optional cross-encoder reranker.

All retrievers expose `search(query, k) -> list[(chunk_index, score)]` over a fixed chunk list.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable

import numpy as np

_TOK = re.compile(r"[a-z0-9]+")


def tokenize(s: str) -> list[str]:
    return _TOK.findall(s.lower())


class BM25Index:
    def __init__(self, texts: list[str]):
        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query: str, k: int = 100) -> list[tuple[int, float]]:
        s = self.bm25.get_scores(tokenize(query))
        top = np.argpartition(-s, min(k, len(s) - 1))[:k]
        top = top[np.argsort(-s[top])]
        return [(int(i), float(s[i])) for i in top]


class HashEmbedder:
    """Deterministic offline stand-in for a dense model (signed hashing of word unigrams+bigrams). TESTS ONLY."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = tokenize(text)
        for t in toks + [a + "_" + b for a, b in zip(toks, toks[1:], strict=False)]:
            h = int(hashlib.md5(t.encode()).hexdigest(), 16)
            v[h % self.dim] += 1 if (h >> 64) % 2 else -1
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    encode_queries = encode_docs


class STEmbedder:
    """sentence-transformers wrapper. bge models want a query instruction prefix."""

    def __init__(self, name: str = "BAAI/bge-small-en-v1.5", max_seq_length: int = 256, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(name, device="cpu")
        self.model.max_seq_length = max_seq_length
        self.name, self.bs = name, batch_size
        self.qprefix = "Represent this sentence for searching relevant passages: " if "bge" in name else ""

    def encode_docs(self, texts):
        return self.model.encode(
            texts,
            batch_size=self.bs,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

    def encode_queries(self, texts):
        return self.encode_docs([self.qprefix + t for t in texts])


class DenseIndex:
    def __init__(self, embedder, texts: list[str] | None = None, embeddings: np.ndarray | None = None):
        import faiss

        self.embedder = embedder
        emb = embeddings if embeddings is not None else embedder.encode_docs(texts)
        self.embeddings = emb
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    def search(self, query: str, k: int = 100) -> list[tuple[int, float]]:
        q = self.embedder.encode_queries([query])
        d, i = self.index.search(q, min(k, self.index.ntotal))
        return [(int(a), float(b)) for a, b in zip(i[0], d[0], strict=True) if a >= 0]


def rrf(rankings: list[list[tuple[int, float]]], k_rrf: int = 60, top: int = 100) -> list[tuple[int, float]]:
    """Reciprocal rank fusion over ranked lists of (id, score)."""
    s: dict[int, float] = {}
    for r in rankings:
        for rank, (i, _) in enumerate(r):
            s[i] = s.get(i, 0.0) + 1.0 / (k_rrf + rank + 1)
    return sorted(s.items(), key=lambda x: -x[1])[:top]


class HybridIndex:
    def __init__(self, bm25: BM25Index, dense: DenseIndex, depth: int = 100):
        self.bm25, self.dense, self.depth = bm25, dense, depth

    def search(self, query: str, k: int = 100):
        return rrf([self.bm25.search(query, self.depth), self.dense.search(query, self.depth)], top=k)


class CrossEncoderReranker:
    def __init__(self, name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length: int = 384):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(name, device="cpu", max_length=max_length)

    def rerank(self, query: str, cand: list[tuple[int, float]], texts: list[str], top_n: int = 30):
        head = cand[:top_n]
        if not head:
            return cand
        sc = self.model.predict([(query, texts[i]) for i, _ in head], show_progress_bar=False, batch_size=32)
        re_ = sorted(zip((i for i, _ in head), sc, strict=True), key=lambda x: -x[1])
        return [(int(i), float(s)) for i, s in re_] + cand[top_n:]


class RerankedIndex:
    def __init__(self, base, reranker, texts: list[str], top_n: int = 30):
        self.base, self.reranker, self.texts, self.top_n = base, reranker, texts, top_n

    def search(self, query: str, k: int = 100):
        return self.reranker.rerank(
            query, self.base.search(query, max(k, self.top_n)), self.texts, self.top_n
        )[:k]


def to_docs(ranked: list[tuple[int, float]], chunk_doc: list[str], k: int) -> list[tuple[str, float]]:
    """Collapse chunk ranking to doc ranking (doc score = best chunk, first occurrence order)."""
    seen, out = set(), []
    for i, s in ranked:
        d = chunk_doc[i]
        if d not in seen:
            seen.add(d)
            out.append((d, s))
            if len(out) >= k:
                break
    return out


Embedder = Callable
