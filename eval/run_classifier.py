"""Train + evaluate the column classifier on four separately reported splits. Writes results/classifier/."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import numpy as np

from datadaddy_ai.classifier import baselines
from datadaddy_ai.classifier.evaluate import failure_cases, metrics
from datadaddy_ai.classifier.model import ColumnClassifier, macro_f1
from datadaddy_ai.classifier.real_none import list_real_columns, real_instances
from datadaddy_ai.classifier.synth import TRAIN_LOCALES, UNSEEN_LOCALES, make_hardneg_set, make_pii_set
from datadaddy_ai.common.seed import lib_versions, set_global_seed

OUT = "results/classifier"


def split_real_columns(seed: int):
    cols = list_real_columns()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(cols))
    n = len(cols)
    tr, va = int(n * 0.6), int(n * 0.75)
    pick = lambda a, b: [cols[i] for i in idx[a:b]]  # noqa: E731
    return pick(0, tr), pick(tr, va), pick(va, n)


def build(seed: int, quick: bool = False):
    k = 0.15 if quick else 1.0
    sz = lambda x: max(int(x * k), 4)  # noqa: E731
    rtr, rva, rte = split_real_columns(seed)
    D = {}
    D["train"] = (
        make_pii_set(seed + 1, TRAIN_LOCALES, sz(350))
        + make_hardneg_set(seed + 2, TRAIN_LOCALES, sz(900))
        + real_instances(rtr, 4, seed + 3)
    )
    D["val"] = (
        make_pii_set(seed + 11, TRAIN_LOCALES, sz(80))
        + make_hardneg_set(seed + 12, TRAIN_LOCALES, sz(150))
        + real_instances(rva, 2, seed + 13)
    )
    real_te = real_instances(rte, 3, seed + 23)
    D["test_in_dist"] = (
        make_pii_set(seed + 21, TRAIN_LOCALES, sz(100))
        + make_hardneg_set(seed + 22, TRAIN_LOCALES, sz(250))
        + real_te
    )
    D["test_unseen_header"] = (
        make_pii_set(seed + 31, TRAIN_LOCALES, sz(70), header_mode="clear", heldout_headers=True)
        + make_pii_set(seed + 32, TRAIN_LOCALES, sz(30), header_mode="misleading", heldout_headers=True)
        + make_hardneg_set(seed + 33, TRAIN_LOCALES, sz(250), header_mode="misleading")
        + real_te
    )
    D["test_unseen_locale"] = (
        make_pii_set(seed + 41, UNSEEN_LOCALES, sz(100))
        + make_hardneg_set(seed + 42, UNSEEN_LOCALES, sz(250))
        + real_te
    )
    return D, (rtr, rva, rte)


def evaluate_all(name, predict, D, headerless_names=("test_headerless",), sub=None):
    res, fails = {}, {}
    for split, cols in D.items():
        if sub:
            rng = np.random.default_rng(0)
            cols = [cols[i] for i in rng.choice(len(cols), min(sub, len(cols)), replace=False)]
        pred = predict(cols, headerless=(split in headerless_names))
        res[split] = metrics([c.label for c in cols], pred)
        fails[split] = failure_cases(cols, pred, 15)
    return res, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--quick", action="store_true", help="small sizes (smoke only; writes to results/classifier_quick)"
    )
    ap.add_argument("--no-presidio", action="store_true")
    a = ap.parse_args()
    seed = set_global_seed(a.seed)
    out = OUT + ("_quick" if a.quick else "")
    import os

    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    D, (rtr, rva, rte) = build(seed, a.quick)
    D["test_headerless"] = D["test_in_dist"]  # same columns, headers replaced at predict time with col_N
    print({k: (len(v), dict(Counter(c.label for c in v))) for k, v in D.items() if k != "test_headerless"})

    # ---- model selection on validation (mean of with-header and header-less macro-F1) ----
    cands, val_scores = {}, {}
    for kind in ["logreg", "lightgbm"]:
        t = time.time()
        m = ColumnClassifier(kind, seed).fit(D["train"])
        yv = [c.label for c in D["val"]]
        f_h = macro_f1(yv, m.predict_instances(D["val"]))
        f_n = macro_f1(yv, m.predict_instances(D["val"], headerless=True))
        cands[kind] = m
        val_scores[kind] = {
            "val_macro_f1_with_header": f_h,
            "val_macro_f1_headerless": f_n,
            "selection_score": (f_h + f_n) / 2,
            "train_seconds": time.time() - t,
        }
        print(kind, val_scores[kind])
    best = max(val_scores, key=lambda k: val_scores[k]["selection_score"])
    model = cands[best]
    os.makedirs("models", exist_ok=True)
    model.save("models/classifier.joblib")

    # ---- ablation: no header dropout (same model family as best) ----
    nodrop = ColumnClassifier(best, seed, dropout=0.0).fit(D["train"])

    def pred_of(m):
        return lambda cols, headerless=False: m.predict_instances(cols, headerless)

    results = {"models": {}}
    fails = {}
    for name, fn in [
        (f"selected:{best}", pred_of(model)),
        (f"ablation:{best}_no_header_dropout", pred_of(nodrop)),
        (
            "other:" + ("lightgbm" if best == "logreg" else "logreg"),
            pred_of(cands["lightgbm" if best == "logreg" else "logreg"]),
        ),
        (
            "baseline:regex_only",
            lambda cols, headerless=False: [baselines.regex_only_predict(c.values) for c in cols],
        ),
        (
            "baseline:header_keyword_only",
            lambda cols, headerless=False: [
                baselines.header_keyword_predict(c.values, None if headerless else c.header) for c in cols
            ],
        ),
    ]:
        results["models"][name], fails[name] = evaluate_all(name, fn, D)
        print(name, {s: round(r["macro_f1"], 3) for s, r in results["models"][name].items()})

    presidio_note = "skipped (--no-presidio)"
    if not a.no_presidio:
        try:
            pb = baselines.PresidioBaseline(n_values=10)
            sub = 600
            results["models"]["baseline:presidio"], fails["baseline:presidio"] = evaluate_all(
                "presidio", lambda cols, headerless=False: [pb.predict(c.values) for c in cols], D, sub=sub
            )
            presidio_note = (
                f"presidio-analyzer with en_core_web_sm, first 10 values per column, evaluated on a fixed "
                f"random subsample of {sub} columns per split (Presidio ignores headers)"
            )
            print(
                "presidio",
                {s: round(r["macro_f1"], 3) for s, r in results["models"]["baseline:presidio"].items()},
            )
        except Exception as e:  # noqa: BLE001
            presidio_note = f"skipped: {e!r}"

    results["meta"] = {
        "command": f"python -m eval.run_classifier --seed {seed}" + (" --quick" if a.quick else ""),
        "seed": seed,
        "versions": lib_versions(),
        "selected_model": best,
        "validation": val_scores,
        "presidio": presidio_note,
        "split_sizes": {k: len(v) for k, v in D.items()},
        "real_none_columns": {"train": len(rtr), "val": len(rva), "test": len(rte)},
        "wall_seconds": time.time() - t0,
        "notes": (
            "Splits (a)-(d) share the same real NONE test columns (column-disjoint from train/val). "
            "Synthetic NONE hard negatives are flagged source=synthetic_hardneg. All PII data is synthetic."
        ),
    }
    json.dump(results, open(f"{out}/metrics.json", "w"), indent=1)
    json.dump(
        {k: v for k, v in fails.items() if k.startswith(("selected", "baseline:regex", "baseline:header"))},
        open(f"{out}/failure_cases.json", "w"),
        indent=1,
    )
    # headline markdown table
    lines = ["| model | in-dist | unseen header | header-less | unseen locale |", "|---|---|---|---|---|"]
    for name, r in results["models"].items():
        lines.append(
            f"| {name} | "
            + " | ".join(
                f"{r[s]['macro_f1']:.3f}"
                for s in ["test_in_dist", "test_unseen_header", "test_headerless", "test_unseen_locale"]
            )
            + " |"
        )
    open(f"{out}/summary.md", "w").write(
        "Macro-F1 per split (classes present in each split)\n\n" + "\n".join(lines) + "\n"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
