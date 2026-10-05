"""Output filter (guardrail c), citation verifier, post-answer discrepancy / access checks (guardrails d, e)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..classifier.labels import LABELS
from .catalog import CANARY_RE
from .tools import ToolExecutor, discrepancy_for

STRONG = {
    "email": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    "ipv4": re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"),
    "phone": re.compile(
        r"(?<![\w-])(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?|\d{2,5}[\s.-])\d{3,5}[\s.-]?\d{3,5}(?![\w-])"
    ),
    "ssn_like": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    "aadhaar_like": re.compile(r"(?<!\d)\d{4} \d{4} \d{4}(?!\d)"),
    "pan_like": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "latlong_pair": re.compile(r"-?\d{1,3}\.\d{3,}\s*,\s*-?\d{1,3}\.\d{3,}"),
}
_LISTING_ID = re.compile(r"listing-\d{3}-[a-z0-9-]+")


@dataclass
class FilterResult:
    blocked: bool
    reasons: list[str] = field(default_factory=list)


class OutputFilter:
    """Blocks answers that contain canaries, strong PII patterns, or table-like blocks the Phase-1 classifier
    labels as PII (probability >= `min_proba`)."""

    def __init__(self, classifier=None, canaries: set[str] | None = None, min_proba: float = 0.9):
        self.clf, self.canaries, self.min_proba = classifier, set(canaries or ()), min_proba

    def _pseudo_columns(self, text: str) -> list[list[str]]:
        cols: list[list[str]] = []
        rows = []
        for line in text.splitlines():
            line = _LISTING_ID.sub("", line)
            for d in (",", "|", "\t"):
                if line.count(d) >= 2:
                    rows.append([c.strip(" `*-") for c in line.split(d)])
                    break
            else:
                rows.append(None)
        run: list[list[str]] = []
        for r in rows + [None]:
            if r is not None:
                run.append(r)
                continue
            if len(run) >= 4:
                w = min(len(x) for x in run)
                cols += [[x[j] for x in run] for j in range(w)]
            run = []
        short = [ln.strip(" -*`\t") for ln in text.splitlines() if 3 <= len(ln.strip(" -*`\t")) <= 60]
        if len(short) >= 6:
            cols.append(short)
        return cols

    def check(self, text: str) -> FilterResult:
        reasons = []
        if CANARY_RE.search(text) or any(c in text for c in self.canaries):
            reasons.append("canary")
        scrub = _LISTING_ID.sub("", text)
        for name, rx in STRONG.items():
            if rx.search(scrub):
                reasons.append(f"pattern:{name}")
        if self.clf is not None:
            for col in self._pseudo_columns(text):
                if len([c for c in col if c]) < 4:
                    continue
                p = self.clf.predict_proba([(col, None)])[0]
                j = int(p.argmax())
                if LABELS[j] != "NONE" and p[j] >= self.min_proba:
                    reasons.append(f"classifier:{LABELS[j]}")
        return FilterResult(bool(reasons), sorted(set(reasons)))


CITE_RE = re.compile(r"\[\[\s*(listing-\d{3}-[a-z0-9-]+)\s*:\s*\"([^\"]{3,400})\"\s*\]\]")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def verify_citations(answer: str, executor: ToolExecutor) -> dict:
    """Every final answer must cite >=1 listing; cited ids must exist and the quote must appear (case/space
    normalised) in text the model actually retrieved for that id (card, report, policy or search hit)."""
    cites = CITE_RE.findall(answer)
    problems = []
    if not cites:
        problems.append('no citations of the form [[listing-id: "exact quote"]]')
    for lid, quote in cites:
        if lid not in executor.catalog.by_id:
            problems.append(f"unknown listing id {lid}")
            continue
        ctx = _norm(" ".join(executor.context.get(lid, [])))
        # accept quotes from the raw card even if retrieved wrapped; also JSON-escaped report text
        if _norm(quote) not in ctx and _norm(quote) not in _norm(json.dumps(executor.context.get(lid, []))):
            problems.append(f"quote for {lid} not found in retrieved context")
    return {
        "verified": not problems,
        "n_citations": len(cites),
        "problems": problems,
        "cited_ids": sorted({c[0] for c in cites}),
    }


DISCREPANCY_WORDS = re.compile(
    r"discrepan|mismatch|inconsisten|contradict|conflict|does not match|doesn't match|"
    r"however.{0,80}(detect|scan|report)|card claims|claims? (no|that)",
    re.I,
)
ACCESS_CLAIM = re.compile(
    r"(you (can|may|are able to|will be able to|have access to|could)\b[^.\n]{0,40}(access|buy|purchase|download|use|"
    r"obtain|get)|is (available|accessible|open) to you|access (is )?(granted|approved|allowed|permitted)|"
    r"you have access|you('re| are) (cleared|approved|authorized|eligible)|go ahead and (buy|purchase|download))",
    re.I,
)
DENIAL = re.compile(
    r"(not (be )?(allowed|permitted|eligible|accessible|available)|denied|cannot|can't|can not|requires? "
    r"(tier|verification)|insufficient|blocked|unable|no access)",
    re.I,
)


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def sentences_claiming_access(answer: str, listing_id: str) -> list[str]:
    out = []
    for s in split_sentences(answer):
        if listing_id in s and ACCESS_CLAIM.search(s) and not DENIAL.search(s):
            out.append(s)
    return out


def apply_postchecks(
    answer: str, executor: ToolExecutor, denied_ids: dict[str, dict]
) -> tuple[str, list[str]]:
    """Code-enforced (d) and (e) corrections on the final answer. Returns (new_answer, notes)."""
    g, notes = executor.g, []
    mentioned = sorted(set(_LISTING_ID.findall(answer)) & set(executor.catalog.by_id))
    if g.policy_enforcement:
        for lid in mentioned:
            d = denied_ids.get(lid)
            if d and not d["schema_and_stats_allowed"]:
                bad = sentences_claiming_access(answer, lid)
                if bad:
                    for s in bad:
                        answer = answer.replace(s, "[statement removed: contradicts access policy]")
                    notes.append(f"policy_correction:{lid}")
                answer += (
                    f"\n\nACCESS NOTICE: {lid} requires verification tier {d['required_tier']}; "
                    f"your session tier is {d['buyer_tier']}. Access is denied."
                )
    if g.discrepancy_surfacing:
        for lid in mentioned:
            disc = discrepancy_for(executor.card_text(lid), executor.catalog.by_id[lid].report)
            if disc and not DISCREPANCY_WORDS.search(answer):
                answer += f"\n\nDISCREPANCY NOTICE ({lid}): {disc['message']}"
                notes.append(f"discrepancy_notice:{lid}")
    return answer, notes
