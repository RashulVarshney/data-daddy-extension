"""Adversarial + benign evaluation of the buyer agent, guardrails OFF vs ON.

LLM_PROVIDER=anthropic|openai_compat -> real-model run (results/agent/<provider>_<model>/).
LLM_PROVIDER unset or `mock`         -> harness validation with NaiveMockAgent (results/agent/mock_naive/); these
                                        numbers validate the harness and the code-enforced guardrails only.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import time
from collections import defaultdict

import numpy as np

from datadaddy_ai.agent.catalog import Catalog
from datadaddy_ai.agent.loop import run_agent
from datadaddy_ai.agent.naive_mock import NaiveMockAgent
from datadaddy_ai.agent.providers import get_provider
from datadaddy_ai.agent.tools import Guardrails, Session, catalog_to_search
from datadaddy_ai.classifier.model import ColumnClassifier
from datadaddy_ai.common.seed import lib_versions, set_global_seed
from eval.scoring import ScoringContext, attack_success, benign_success, load_jsonl


def pct(x):
    return float(np.percentile(x, 50)), float(np.percentile(x, 95))


def run_mode(name, g, provider, catalog, ctx, clf, attacks, benign, outdir, limit):
    search = catalog_to_search(catalog)
    rows_a, rows_b = [], []
    tf = open(outdir / f"transcripts_{name}.jsonl", "w")
    for kind, cases, rows in [("attack", attacks, rows_a), ("benign", benign, rows_b)]:
        for c in cases[:limit]:
            rec = run_agent(
                provider,
                catalog,
                g,
                Session(buyer_tier=c["buyer_tier"]),
                c["user_message"],
                c["id"],
                c.get("card_overrides") or None,
                classifier=clf,
                search=search,
                canaries=ctx.canaries,
            )
            d = rec.to_dict()
            sc = attack_success(c, d, ctx) if kind == "attack" else benign_success(c, d)
            row = {"case": c, "kind": kind, "score": sc, "record": d}
            tf.write(json.dumps(row) + "\n")
            rows.append(row)
            print(name, kind, c["id"], "success" if sc["success"] else "-", flush=True)
    tf.close()
    return rows_a, rows_b


def summarize(rows_a, rows_b):
    by = defaultdict(lambda: [0, 0, 0])
    for r in rows_a:
        cat = r["case"]["category"]
        by[cat][0] += r["score"]["success"]
        by[cat][1] += r["score"]["attempted"]
        by[cat][2] += 1
    cats = {
        k: {"attack_success": v[0], "attempted": v[1], "n": v[2], "rate": v[0] / v[2]} for k, v in by.items()
    }
    tot_s, tot_a, tot_n = (sum(v[i] for v in by.values()) for i in range(3))
    allr = rows_a + rows_b
    lat = [r["record"]["wall_s"] for r in allr]
    llm = [r["record"]["llm_latency_s"] for r in allr]
    ben_ok = sum(r["score"]["success"] for r in rows_b)
    return {
        "attack": {
            "total_success": tot_s,
            "total_attempted": tot_a,
            "n": tot_n,
            "rate": tot_s / max(tot_n, 1),
            "by_category": cats,
        },
        "benign": {
            "n": len(rows_b),
            "task_success": ben_ok,
            "task_success_rate": ben_ok / max(len(rows_b), 1),
            "blocked": sum(r["record"]["blocked"] for r in rows_b),
            "block_rate": sum(r["record"]["blocked"] for r in rows_b) / max(len(rows_b), 1),
            "citation_verified": sum(bool(r["record"]["citations"].get("verified")) for r in rows_b),
            "failed_checks": dict(
                sorted(
                    (
                        (k, sum(1 for r in rows_b if not r["score"]["checks"].get(k, True)))
                        for k in {k for r in rows_b for k in r["score"]["checks"]}
                    ),
                    key=lambda x: x[0],
                )
            ),
        },
        "latency_s": {
            "wall_p50": pct(lat)[0],
            "wall_p95": pct(lat)[1],
            "llm_p50": pct(llm)[0],
            "llm_p95": pct(llm)[1],
        },
        "tokens": {
            "input_total": sum(r["record"]["usage"]["input_tokens"] for r in allr),
            "output_total": sum(r["record"]["usage"]["output_tokens"] for r in allr),
            "mean_per_case": float(
                np.mean(
                    [
                        r["record"]["usage"]["input_tokens"] + r["record"]["usage"]["output_tokens"]
                        for r in allr
                    ]
                )
            ),
        },
        "errors": sum(bool(r["record"]["error"]) for r in allr),
        "max_steps_hit": sum(r["record"]["max_steps_hit"] for r in allr),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--limit", type=int, default=10**6, help="cases per list (debug)")
    ap.add_argument("--modes", default="off,on")
    a = ap.parse_args()
    seed = set_global_seed(a.seed)
    name = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if name in {"", "mock"}:
        provider, label, real = NaiveMockAgent(), "mock_naive", False
    else:
        provider = get_provider(name)  # raises loudly on missing credentials
        label, real = f"{provider.name}_{re.sub(r'[^A-Za-z0-9._-]+', '-', provider.model)}", True
    outdir = pathlib.Path("results/agent") / label
    outdir.mkdir(parents=True, exist_ok=True)
    catalog = Catalog.load()
    ctx = ScoringContext(catalog)
    clf = ColumnClassifier.load("models/classifier.joblib")
    attacks, benign = load_jsonl("eval/adversarial_cases.jsonl"), load_jsonl("eval/benign_cases.jsonl")
    res = {
        "meta": {
            "command": f"LLM_PROVIDER={name or 'unset'} python -m eval.run_agent --seed {seed}",
            "seed": seed,
            "provider": provider.name,
            "model": provider.model,
            "temperature": 0.0,
            "real_model": real,
            "n_attacks": len(attacks),
            "n_benign": len(benign),
            "versions": lib_versions(),
            "date": time.strftime("%Y-%m-%d"),
            "note": (
                "REAL MODEL RUN"
                if real
                else "MOCK RUN: NaiveMockAgent is a deliberately vulnerable scripted "
                "stand-in. Validates the harness and code-enforced guardrails only; says nothing about real LLMs. "
                "Prompt-level guardrail (a) cannot be evaluated with it."
            ),
        },
        "modes": {},
    }
    for mode in a.modes.split(","):
        g = Guardrails.all_off() if mode == "off" else Guardrails.all_on()
        ra, rb = run_mode(mode, g, provider, catalog, ctx, clf, attacks, benign, outdir, a.limit)
        res["modes"][mode] = {"guardrails": g.as_dict(), **summarize(ra, rb)}
    json.dump(res, open(outdir / "metrics.json", "w"), indent=1)
    lines = [
        f"Provider: {provider.name}/{provider.model} ({'REAL' if real else 'MOCK harness validation'})",
        "",
        "| category | n | OFF success | ON success |",
        "|---|---|---|---|",
    ]
    cats = res["modes"][a.modes.split(",")[0]]["attack"]["by_category"]
    for cat in cats:
        row = [
            f"{res['modes'][m]['attack']['by_category'][cat]['attack_success']}/{cats[cat]['n']}"
            for m in a.modes.split(",")
        ]
        lines.append(f"| {cat} | {cats[cat]['n']} | " + " | ".join(row) + " |")
    lines.append(
        "| **total** | "
        + str(res["modes"][a.modes.split(",")[0]]["attack"]["n"])
        + " | "
        + " | ".join(
            f"{res['modes'][m]['attack']['total_success']}/{res['modes'][m]['attack']['n']}"
            for m in a.modes.split(",")
        )
        + " |"
    )
    lines += [
        "",
        "| mode | benign success | benign blocked | p50 wall s | p95 wall s | tokens/case |",
        "|---|---|---|---|---|---|",
    ]
    for m in a.modes.split(","):
        r = res["modes"][m]
        lines.append(
            f"| {m} | {r['benign']['task_success']}/{r['benign']['n']} | {r['benign']['blocked']}/{r['benign']['n']} | "
            f"{r['latency_s']['wall_p50']:.2f} | {r['latency_s']['wall_p95']:.2f} | {r['tokens']['mean_per_case']:.0f} |"
        )
    (outdir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
