"""Tiny offline smoke eval for CI: tiny classifier + scan, BM25/hash-hybrid retrieval, mock-agent harness, <1 min."""

from __future__ import annotations

import json
import sys

import pandas as pd

from datadaddy_ai.agent.catalog import Catalog
from datadaddy_ai.agent.loop import run_agent
from datadaddy_ai.agent.naive_mock import NaiveMockAgent
from datadaddy_ai.agent.tools import Guardrails, Session
from datadaddy_ai.classifier.model import ColumnClassifier
from datadaddy_ai.classifier.scan import scan_dataframe
from datadaddy_ai.classifier.synth import SyntheticGenerator, make_hardneg_set, make_pii_set
from datadaddy_ai.common.seed import SEED, set_global_seed
from datadaddy_ai.retrieval.corpus import Card
from datadaddy_ai.retrieval.index import HashEmbedder
from datadaddy_ai.retrieval.search import DatasetSearch
from eval.scoring import ScoringContext, attack_success, load_jsonl


def main() -> int:
    set_global_seed(SEED)
    clf = ColumnClassifier("logreg", SEED).fit(
        make_pii_set(1, ["en_US", "en_IN"], 20) + make_hardneg_set(2, ["en_US"], 120)
    )
    g = SyntheticGenerator(3)
    df = pd.DataFrame(
        {"a": g.gen_email(40, "en_US"), "b": g.gen_ip(40, "en_US"), "c": [str(i % 4) for i in range(40)]}
    )
    rep = scan_dataframe(df, clf, headerless=True)
    preds = [c["predicted_class"] for c in rep["columns"]]
    s = DatasetSearch.hybrid(
        [Card("o/spam", "# Spam\nSMS spam messages", ["text-classification"], ["en"], ["mit"], [])],
        HashEmbedder(),
    )
    cat = Catalog.load()
    ctx = ScoringContext(cat)
    case = next(c for c in load_jsonl("eval/adversarial_cases.jsonl") if c["category"] == "direct_injection")
    rec = run_agent(
        NaiveMockAgent(), cat, Guardrails.all_on(), Session(1), case["user_message"], classifier=clf
    ).to_dict()
    out = {
        "scan_preds": preds,
        "risk": rep["risk"]["level"],
        "search_top": s.search("spam sms", 1)[0]["id"],
        "agent_attack_success_guardrails_on": attack_success(case, rec, ctx)["success"],
    }
    print(json.dumps(out))
    ok = (
        preds[:2] == ["EMAIL", "IP_ADDRESS"]
        and out["search_top"] == "o/spam"
        and not out["agent_attack_success_guardrails_on"]
    )
    print("smoke: ok" if ok else "smoke: FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
