"""Baselines: regex-only, header-keyword-only, and (optional) Presidio."""

from __future__ import annotations

import re
from collections import Counter

from .features import RX, luhn_valid, sample_values

# Regex-only: first rule whose match rate >= 0.6 wins (ordered most specific first).
REGEX_RULES = [
    ("EMAIL", lambda v: bool(RX["email"].match(v))),
    ("IP_ADDRESS", lambda v: bool(RX["ipv4"].match(v) or RX["ipv6"].match(v))),
    (
        "GOVT_ID",
        lambda v: bool(RX["ssn_like"].match(v) or RX["pan_like"].match(v) or RX["aadhaar_like"].match(v)),
    ),
    ("LAT_LONG", lambda v: bool(RX["latlon_pair"].match(v))),
    ("DOB", lambda v: bool(RX["date_iso"].match(v) or RX["date_slash"].match(v) or RX["date_text"].match(v))),
    ("PHONE", lambda v: bool(RX["phone_like"].match(v)) and sum(c.isdigit() for c in v) >= 10),
    ("ADDRESS", lambda v: bool(RX["street_words"].search(v) and RX["addr_shape"].match(v))),
]


def regex_only_predict(values: list, header: str | None = None) -> str:
    vals, _, _ = sample_values(values)
    if not vals:
        return "NONE"
    for label, fn in REGEX_RULES:
        if sum(fn(v) for v in vals) / len(vals) >= 0.6:
            return label
    return "NONE"


HEADER_KEYWORDS = {
    "EMAIL": ["email", "e-mail", "e_mail", "mail"],
    "PHONE": ["phone", "mobile", "tel", "cell", "contact_no", "contact_number", "ph_no", "ph no"],
    "PERSON_NAME": ["name", "surname", "person"],
    "ADDRESS": ["address", "addr", "street", "residence"],
    "DOB": ["dob", "birth", "born"],
    "GOVT_ID": ["ssn", "aadhaar", "aadhar", "pan", "govt", "national_id", "tax_id", "social_sec", "passport"],
    "IP_ADDRESS": ["ip_", "ip ", "ipaddr", "ipv4", "ipv6", "_ip", "remote_addr"],
    "LAT_LONG": ["lat", "lon", "lng", "coord", "gps", "geo"],
}


def header_keyword_predict(values: list, header: str | None) -> str:
    h = (header or "").lower().replace("-", "_")
    if not h:
        return "NONE"
    h2 = f" {h} "
    for label in ["EMAIL", "IP_ADDRESS", "DOB", "GOVT_ID", "LAT_LONG", "PHONE", "ADDRESS", "PERSON_NAME"]:
        for kw in HEADER_KEYWORDS[label]:
            if kw in h or kw in h2:
                return label
    return "NONE"


class PresidioBaseline:
    """Column-level wrapper over presidio-analyzer with the small spaCy model (en_core_web_sm)."""

    ENT = {
        "EMAIL_ADDRESS": "EMAIL",
        "PHONE_NUMBER": "PHONE",
        "PERSON": "PERSON_NAME",
        "LOCATION": "ADDRESS",
        "DATE_TIME": "DOB",
        "US_SSN": "GOVT_ID",
        "IN_PAN": "GOVT_ID",
        "IN_AADHAAR": "GOVT_ID",
        "UK_NHS": "GOVT_ID",
        "IP_ADDRESS": "IP_ADDRESS",
        "US_DRIVER_LICENSE": "GOVT_ID",
        "US_PASSPORT": "GOVT_ID",
        "UK_NINO": "GOVT_ID",
    }

    def __init__(self, n_values: int = 25, min_rate: float = 0.4):
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        cfg = {"nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]}
        self.engine = AnalyzerEngine(
            nlp_engine=NlpEngineProvider(nlp_configuration=cfg).create_engine(), supported_languages=["en"]
        )
        self.n_values, self.min_rate = n_values, min_rate

    def predict(self, values: list, header: str | None = None) -> str:
        vals, _, _ = sample_values(values)
        vals = vals[: self.n_values]
        if not vals:
            return "NONE"
        c: Counter = Counter()
        for v in vals:
            labs = {
                self.ENT[r.entity_type]
                for r in self.engine.analyze(text=v, language="en")
                if r.entity_type in self.ENT and r.score >= 0.4
            }
            c.update(labs)
        if not c:
            return "NONE"
        lab, k = c.most_common(1)[0]
        return lab if k / len(vals) >= self.min_rate else "NONE"


_ = (re, luhn_valid)
