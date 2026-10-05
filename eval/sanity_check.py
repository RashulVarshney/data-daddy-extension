"""Score the CLI sanity-check outputs (results/sanity/*.json) against hand-assigned column labels.

The labels below are the author's judgment (not an external gold standard); every column not listed is NONE.
"""

import json

TRUTH = {
    "legislators-current": {
        "last_name": "PERSON_NAME",
        "first_name": "PERSON_NAME",
        "middle_name": "PERSON_NAME",
        "nickname": "PERSON_NAME",
        "full_name": "PERSON_NAME",
        "birthday": "DOB",
        "address": "ADDRESS",
        "phone": "PHONE",
    },
    "titanic": {"Name": "PERSON_NAME"},
    "usgs_2.5_week": {"latitude": "LAT_LONG", "longitude": "LAT_LONG"},
}


def main():
    out = {}
    for f, truth in TRUTH.items():
        r = json.load(open(f"results/sanity/{f}.json"))
        pred = {c["name"]: c["predicted_class"] for c in r["columns"]}
        tp = [c for c, t in truth.items() if pred[c] == t]
        fn = {c: pred[c] for c, t in truth.items() if pred[c] != t}
        fp = {c: p for c, p in pred.items() if p != "NONE" and c not in truth}
        out[f] = {
            "n_columns": len(pred),
            "labelled_pii_columns": len(truth),
            "correct": len(tp),
            "missed_or_wrong_class": fn,
            "false_positives": fp,
            "risk_level": r["risk"]["level"],
            "risk_score": r["risk"]["score"],
            "required_tier": r["tier_policy"]["required_tier"],
            "raw_rows_blocked": r["tier_policy"]["raw_rows_blocked"],
        }
        print(f, json.dumps(out[f]))
    json.dump(out, open("results/sanity/sanity_scores.json", "w"), indent=1)


if __name__ == "__main__":
    main()
