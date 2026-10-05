"""SILVER query generation from structured card metadata (task, language, license, size) with template paraphrases.

Relevant set = every card whose metadata satisfies all constraints in the query. Labels are derived mechanically from
metadata, not human-verified, and the index also contains a rendering of that metadata, so lexical retrievers are
favoured (see docs/RESULTS.md, known bias).
"""

from __future__ import annotations

import json

import numpy as np

from datadaddy_ai.retrieval.corpus import LANG_NAMES, SIZE_TEXT, Card, license_text

TASK_PHRASES = {
    "text-classification": ["text classification", "classifying text"],
    "question-answering": ["question answering", "QA"],
    "text-generation": ["text generation", "language modeling"],
    "translation": ["machine translation", "translation"],
    "summarization": ["summarization"],
    "token-classification": ["token classification", "sequence labeling"],
    "automatic-speech-recognition": ["speech recognition", "ASR"],
    "image-classification": ["image classification"],
    "text-to-image": ["text-to-image generation"],
    "sentence-similarity": ["sentence similarity", "semantic similarity"],
    "feature-extraction": ["embedding training", "feature extraction"],
    "fill-mask": ["masked language modeling"],
    "zero-shot-classification": ["zero-shot classification"],
    "object-detection": ["object detection"],
    "text-retrieval": ["text retrieval", "information retrieval"],
}
TEMPLATES = [
    "datasets for {task} {rest}",
    "I need a dataset for {task} {rest}",
    "looking for {task} data {rest}",
    "find me {task} corpora {rest}",
    "{task} dataset {rest}",
    "which datasets support {task} {rest}",
    "open data for {task} {rest}",
    "recommend {task} datasets {rest}",
]


def _task(t: str, rng) -> str:
    opts = TASK_PHRASES.get(t, [t.replace("-", " ")])
    return str(rng.choice(opts))


def build_silver(cards: list[Card], n: int, seed: int, max_rel: int = 40) -> list[dict]:
    rng = np.random.default_rng(seed)
    usable = [c for c in cards if c.task_categories and c.languages and c.license and c.size_categories]
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < n * 200:
        tries += 1
        c = usable[int(rng.integers(len(usable)))]
        langs = [x for x in c.languages if x in LANG_NAMES]
        if not langs:
            continue
        cons = {
            "task": str(rng.choice(c.task_categories)),
            "language": str(rng.choice(langs)),
            "license": str(rng.choice(c.license)),
            "size": str(rng.choice(c.size_categories)),
        }
        k = int(rng.choice([2, 2, 3, 3, 4]))
        keys = list(rng.choice(["task", "language", "license", "size"], size=k, replace=False))
        if "task" not in keys:
            keys[0] = "task"
        cons = {key: cons[key] for key in sorted(set(keys))}
        sig = tuple(sorted(cons.items()))
        if sig in seen:
            continue
        rel = [
            d.id
            for d in cards
            if all(
                (
                    v.lower()
                    in [
                        x.lower()
                        for x in {
                            "task": d.task_categories,
                            "language": d.languages,
                            "license": d.license,
                            "size": d.size_categories,
                        }[key]
                    ]
                )
                for key, v in cons.items()
            )
        ]
        if not 1 <= len(rel) <= max_rel:
            continue
        seen.add(sig)
        frags = []
        if "language" in cons:
            frags.append(
                str(rng.choice(["in {x}", "in the {x} language", "covering {x}", "for {x}"])).format(
                    x=LANG_NAMES[cons["language"]]
                )
            )
        if "license" in cons:
            frags.append(
                str(rng.choice(["under the {x} license", "with a {x} license", "licensed {x}"])).format(
                    x=license_text(cons["license"])
                )
            )
        if "size" in cons:
            frags.append(
                str(rng.choice(["with {x}", "containing {x}", "of size {x}"])).format(
                    x=SIZE_TEXT.get(cons["size"], cons["size"])
                )
            )
        rng.shuffle(frags)
        q = str(rng.choice(TEMPLATES)).format(task=_task(cons["task"], rng), rest=" ".join(frags)).strip()
        out.append(
            {
                "qid": f"silver-{len(out):03d}",
                "query": q,
                "constraints": cons,
                "relevant": rel,
                "n_constraints": len(cons),
            }
        )
    return out


def save(queries, path):
    with open(path, "w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
