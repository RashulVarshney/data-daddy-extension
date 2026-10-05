from __future__ import annotations

import math

import numpy as np


def first_rank(ranked: list[str], relevant: set[str]) -> int | None:
    for r, d in enumerate(ranked, 1):
        if d in relevant:
            return r
    return None


def hit_at_k(ranked, relevant, k=5) -> float:
    return float(any(d in relevant for d in ranked[:k]))


def mrr_at_k(ranked, relevant, k=10) -> float:
    r = first_rank(ranked[:k], relevant)
    return 1.0 / r if r else 0.0


def ndcg_at_k(ranked, relevant, k=10) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in relevant)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0.0


def summarize(per_query: list[dict], latencies: list[float]) -> dict:
    return {
        "n": len(per_query),
        "hit@5": float(np.mean([q["hit@5"] for q in per_query])),
        "mrr@10": float(np.mean([q["mrr@10"] for q in per_query])),
        "ndcg@10": float(np.mean([q["ndcg@10"] for q in per_query])),
        "latency_ms_p50": float(np.percentile(latencies, 50) * 1000),
        "latency_ms_p95": float(np.percentile(latencies, 95) * 1000),
    }
