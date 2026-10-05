"""Retrieval evaluation: BM25 vs dense vs hybrid vs hybrid+rerank, two chunk settings, SILVER and MANUAL-REVIEW sets."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import numpy as np

from datadaddy_ai.common.seed import lib_versions, set_global_seed
from datadaddy_ai.retrieval.corpus import CHUNK_SETTINGS, chunk_corpus, load_cards
from datadaddy_ai.retrieval.index import (
    BM25Index,
    CrossEncoderReranker,
    DenseIndex,
    HybridIndex,
    RerankedIndex,
    STEmbedder,
    to_docs,
)
from datadaddy_ai.retrieval.metrics import first_rank, hit_at_k, mrr_at_k, ndcg_at_k, summarize
from eval.silver_queries import build_silver, save

OUT = pathlib.Path("results/retrieval")
EMB_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_manual(path="eval/queries_manual_review.jsonl"):
    return [json.loads(line) for line in open(path)]


def run_config(index, chunk_doc, queries, depth=100):
    per, lats, ranks = [], [], {}
    for q in queries:
        t = time.perf_counter()
        ranked = to_docs(index.search(q["query"], depth), chunk_doc, 10)
        lats.append(time.perf_counter() - t)
        docs = [d for d, _ in ranked]
        rel = set(q["relevant"])
        per.append(
            {
                "qid": q["qid"],
                "hit@5": hit_at_k(docs, rel, 5),
                "mrr@10": mrr_at_k(docs, rel, 10),
                "ndcg@10": ndcg_at_k(docs, rel, 10),
            }
        )
        ranks[q["qid"]] = {"first_rank_top10": first_rank(docs, rel), "top3": docs[:3]}
    return summarize(per, lats), per, ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n-silver", type=int, default=120)
    ap.add_argument("--limit-cards", type=int, default=None)
    ap.add_argument("--no-rerank", action="store_true")
    a = ap.parse_args()
    seed = set_global_seed(a.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    cards = load_cards(limit=a.limit_cards)
    silver = build_silver(cards, a.n_silver, seed)
    save(silver, "eval/queries_silver.jsonl")
    manual = load_manual()
    verified = all(q.get("verified") for q in manual)
    sets = {"silver": silver, "manual_review": manual}
    print("cards", len(cards), "silver", len(silver), "manual", len(manual), "manual_verified", verified)

    emb = STEmbedder(EMB_MODEL, 512)
    reranker = None if a.no_rerank else CrossEncoderReranker(RERANK_MODEL)
    results, perq_all, ranks_all, meta_chunks = {}, {}, {}, {}
    for cs_name, (mw, ov) in CHUNK_SETTINGS.items():
        chunks = chunk_corpus(cards, mw, ov)
        texts, chunk_doc = [c.text for c in chunks], [c.doc_id for c in chunks]
        meta_chunks[cs_name] = {
            "n_chunks": len(chunks),
            "max_words": mw,
            "overlap": ov,
            "mean_chunks_per_card": len(chunks) / len(cards),
        }
        cache = pathlib.Path(f"data/hf_cards/emb_{cs_name}_{len(cards)}.npy")
        t = time.time()
        if cache.exists():
            E = np.load(cache)
        else:
            E = emb.encode_docs(texts)
            np.save(cache, E)
        meta_chunks[cs_name]["embed_seconds"] = round(time.time() - t, 1)
        bm25 = BM25Index(texts)
        dense = DenseIndex(emb, embeddings=E)
        hybrid = HybridIndex(bm25, dense)
        cfgs = {"bm25": bm25, "dense": dense, "hybrid_rrf": hybrid}
        if reranker:
            cfgs["hybrid_rrf+rerank"] = RerankedIndex(hybrid, reranker, texts, top_n=30)
        for cfg, idx in cfgs.items():
            for sname, qs in sets.items():
                summ, per, ranks = run_config(idx, chunk_doc, qs)
                key = f"{cs_name}/{cfg}/{sname}"
                results[key] = summ
                perq_all[key], ranks_all[key] = per, ranks
                print(key, {k: round(v, 3) for k, v in summ.items()})

    # error analysis inputs
    qmeta = {q["qid"]: q for q in silver}
    ea = {}
    for key, per in perq_all.items():
        if key.endswith("/silver"):
            by = {}
            for p in per:
                q = qmeta[p["qid"]]
                for c in q["constraints"]:
                    by.setdefault(f"has_{c}", []).append(p["hit@5"])
                by.setdefault(f"n_constraints={q['n_constraints']}", []).append(p["hit@5"])
                by.setdefault("rel_set_size<=5" if len(q["relevant"]) <= 5 else "rel_set_size>5", []).append(
                    p["hit@5"]
                )
            ea[key] = {k: {"hit@5": round(float(np.mean(v)), 3), "n": len(v)} for k, v in sorted(by.items())}
        else:
            ea[key] = {
                "misses_top10": [
                    q["qid"] for q in manual if ranks_all[key][q["qid"]]["first_rank_top10"] is None
                ],
                "found_not_top5": [
                    q["qid"] for q in manual if (ranks_all[key][q["qid"]]["first_rank_top10"] or 0) > 5
                ],
            }
    json.dump(
        {
            "meta": {
                "command": f"python -m eval.run_retrieval --seed {seed}",
                "seed": seed,
                "versions": lib_versions(),
                "n_cards": len(cards),
                "dense_model": EMB_MODEL,
                "reranker_model": None if a.no_rerank else RERANK_MODEL,
                "chunking": meta_chunks,
                "manual_review_verified": verified,
                "labels": {
                    "silver": "SILVER: auto-generated from card metadata, not human-verified",
                    "manual_review": "VERIFIED" if verified else "UNVERIFIED (verified:false in file)",
                },
            },
            "results": results,
            "error_analysis": ea,
        },
        open(OUT / "metrics.json", "w"),
        indent=1,
    )
    json.dump({"per_query": perq_all, "ranks": ranks_all}, open(OUT / "per_query.json", "w"))
    lines = [
        "| chunking | retriever | set | Hit@5 | MRR@10 | nDCG@10 | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k, v in results.items():
        cs, cfg, s = k.split("/")
        tag = s if s == "silver" else ("manual (UNVERIFIED)" if not verified else "manual")
        lines.append(
            f"| {cs} | {cfg} | {tag} | {v['hit@5']:.3f} | {v['mrr@10']:.3f} | {v['ndcg@10']:.3f} | "
            f"{v['latency_ms_p50']:.1f} | {v['latency_ms_p95']:.1f} |"
        )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    os.sync()


if __name__ == "__main__":
    main()
