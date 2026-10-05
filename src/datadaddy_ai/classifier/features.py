"""Column featurisation: value char-n-gram TF-IDF (raw + shape), shape stats, regex rates, checksum checks, header n-grams."""

from __future__ import annotations

import re

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .labels import MISSING_TOKENS

SAMPLE_SIZE = 200

RX = {
    "email": re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$"),
    "ipv4": re.compile(r"^(\d{1,3}\.){3}\d{1,3}$"),
    "ipv6": re.compile(r"^([0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}$"),
    "phone_like": re.compile(r"^\+?[\d\s().-]{7,20}(x\d+)?$"),
    "date_iso": re.compile(r"^\d{4}-\d{2}-\d{2}"),
    "date_slash": re.compile(r"^\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}$"),
    "date_text": re.compile(
        r"^(\d{1,2}[ -])?[A-Za-z]{3,9}[ -]\d{1,2}?,? ?\d{0,4}|^\d{1,2}[ -][A-Za-z]{3}[ -]\d{4}$"
    ),
    "latlon_scalar": re.compile(r"^-?\d{1,3}\.\d{3,8}$"),
    "latlon_pair": re.compile(r"^\(?\s*-?\d{1,3}\.\d+\s*[, ]\s*-?\d{1,3}\.\d+\s*\)?$"),
    "ssn_like": re.compile(r"^\d{3}-?\d{2}-?\d{4}$"),
    "aadhaar_like": re.compile(r"^\d{4} ?\d{4} ?\d{4}$"),
    "pan_like": re.compile(r"^[A-Z]{5}\d{4}[A-Z]$"),
    "zip_like": re.compile(r"^\d{5}(-\d{4})?$|^\d{6}$"),
    "uuid": re.compile(r"^[0-9a-fA-F]{8}-([0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}$"),
    "int": re.compile(r"^-?\d+$"),
    "float": re.compile(r"^-?\d+\.\d+$"),
    "name_like": re.compile(r"^(?:[A-Z][a-zA-Z'.-]+)(?:[ ,]+[A-Z][a-zA-Z'.-]+){0,3}$"),
    "street_words": re.compile(
        r"\b(st|street|rd|road|ave|avenue|lane|ln|nagar|marg|drive|dr|blvd|chowk|"
        r"circle|apt|suite|house|h\.no|platz|stra(ss|ß)e|allee|way|court|close)\b",
        re.I,
    ),
    "addr_shape": re.compile(r"\d+.*[A-Za-z]{3,}.*|[A-Za-z]{3,}.*\d+"),
}
REGEX_NAMES = list(RX)


def luhn_valid(s: str) -> bool:
    d = re.sub(r"\D", "", s)
    if len(d) < 2:
        return False
    total = 0
    for i, ch in enumerate(reversed(d)):
        x = int(ch)
        if i % 2 == 1:
            x *= 2
            if x > 9:
                x -= 9
        total += x
    return total % 10 == 0


def is_missing(v) -> bool:
    return v is None or str(v).strip().lower() in MISSING_TOKENS


def normalise_header(h: str | None) -> str:
    if not h:
        return "__none__"
    h = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(h))  # camelCase -> camel Case
    return re.sub(r"[^a-z0-9]+", " ", h.lower()).strip() or "__none__"


def is_generic_header(h: str | None) -> bool:
    n = normalise_header(h)
    return (
        n == "__none__"
        or bool(re.fullmatch(r"(col|column|f|var|c|x|field|unnamed)\s?[a-z0-9]{1,3}( \d+)?", n))
        or bool(re.fullmatch(r"(col|column|field|unnamed|var) [a-z0-9]+", n))
    )


def shape(v: str) -> str:
    v = re.sub(r"[A-Z]", "A", v)
    v = re.sub(r"[a-z]", "a", v)
    return re.sub(r"\d", "9", v)


def sample_values(values: list, seed: int = 0, k: int = SAMPLE_SIZE) -> tuple[list[str], float, float]:
    """Return (non-missing sample as stripped strings, null_rate, whitespace_rate) over up to k rows."""
    vals = list(values)
    if len(vals) > k:
        rng = np.random.default_rng(seed)
        vals = [vals[i] for i in rng.choice(len(vals), k, replace=False)]
    n = max(len(vals), 1)
    keep = [str(v) for v in vals if not is_missing(v)]
    ws = sum(1 for v in keep if v != v.strip()) / max(len(keep), 1)
    return [v.strip() for v in keep], 1 - len(keep) / n, ws


def dense_features(vals: list[str], null_rate: float, ws_rate: float) -> np.ndarray:
    n = len(vals)
    if n == 0:
        return np.zeros(len(DENSE_NAMES))
    lens = np.array([len(v) for v in vals], dtype=float)
    dig = np.array([sum(c.isdigit() for c in v) / max(len(v), 1) for v in vals])
    alp = np.array([sum(c.isalpha() for c in v) / max(len(v), 1) for v in vals])
    upp = np.array([sum(c.isupper() for c in v) / max(len(v), 1) for v in vals])
    spc = np.array([v.count(" ") / max(len(v), 1) for v in vals])
    sym = 1 - dig - alp - spc
    ntok = np.array([len(v.split()) for v in vals], dtype=float)
    f = [
        null_rate,
        ws_rate,
        n / SAMPLE_SIZE,
        len(set(vals)) / n,
        dig.mean(),
        alp.mean(),
        upp.mean(),
        spc.mean(),
        sym.mean(),
        lens.mean(),
        lens.std(),
        lens.min(),
        lens.max(),
        lens.std() / (lens.mean() + 1e-9),
        ntok.mean(),
        ntok.std(),
        len({shape(v) for v in vals}) / n,
    ]
    for ch in "@.-(+/:,":
        f.append(np.mean([ch in v for v in vals]))
    # numeric stats
    nums = []
    for v in vals:
        try:
            x = float(v.replace(",", ""))
            if np.isfinite(x):
                nums.append(max(min(x, 1e12), -1e12))
        except ValueError:
            pass
    fr = len(nums) / n
    f.append(fr)
    if nums:
        a = np.array(nums)
        dec = np.mean([len(v.split(".")[1]) if "." in v and v.split(".")[1].isdigit() else 0 for v in vals])
        f += [
            np.log1p(abs(a).mean()),
            np.log1p(a.std()),
            np.mean(np.abs(a) <= 90),
            np.mean(np.abs(a) <= 180),
            np.mean(a < 0),
            dec,
            np.mean(a == np.round(a)),
        ]
    else:
        f += [0] * 7
    return np.array(f, dtype=float)


DENSE_NAMES = (
    [
        "null_rate",
        "ws_rate",
        "n_frac",
        "uniq",
        "dig",
        "alpha",
        "upper",
        "space",
        "sym",
        "len_mean",
        "len_std",
        "len_min",
        "len_max",
        "len_cv",
        "ntok_mean",
        "ntok_std",
        "shape_div",
    ]
    + [f"has_{c}" for c in "@.-(+/:,"]
    + [
        "numeric_frac",
        "num_logmean",
        "num_logstd",
        "num_le90",
        "num_le180",
        "num_neg",
        "num_decimals",
        "num_isint",
    ]
)


def regex_features(vals: list[str]) -> np.ndarray:
    n = max(len(vals), 1)
    f = []
    for name, rx in RX.items():
        hit = rx.search if name == "street_words" else rx.match
        f.append(sum(1 for v in vals if hit(v)) / n)
    aad = [v for v in vals if RX["aadhaar_like"].match(v)]
    f.append(np.mean([luhn_valid(v) for v in aad]) * len(aad) / n if aad else 0.0)  # checksum-pattern check
    f.append(np.mean([luhn_valid(v) for v in vals]) if vals else 0.0)
    f.append(np.mean([v.lower() == v for v in vals]) if vals else 0.0)
    return np.array(f, dtype=float)


REGEX_FEATURE_NAMES = REGEX_NAMES + ["aadhaar_luhn_valid", "luhn_any", "all_lower"]


def column_doc(vals: list[str]) -> tuple[str, str]:
    sub = vals[:60]
    return " │ ".join(v.lower() for v in sub), " │ ".join(shape(v) for v in sub)


class ColumnFeaturizer:
    """fit() on training columns; transform() maps (values, header) -> sparse feature row."""

    def __init__(
        self,
        max_val_feats: int = 4000,
        max_shape_feats: int = 2000,
        max_hdr_feats: int = 1500,
        use_header: bool = True,
    ):
        self.val_tfidf = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            max_features=max_val_feats,
            sublinear_tf=True,
            min_df=3,
            dtype=np.float32,
        )
        self.shape_tfidf = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            max_features=max_shape_feats,
            sublinear_tf=True,
            min_df=3,
            lowercase=False,
            dtype=np.float32,
        )
        self.hdr_tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=max_hdr_feats,
            sublinear_tf=True,
            min_df=2,
            dtype=np.float32,
        )
        self.scaler = StandardScaler()
        self.use_header = use_header

    def _prep(self, cols, seed=0):
        docs_v, docs_s, dense, hdrs = [], [], [], []
        for i, (values, header) in enumerate(cols):
            vals, nr, ws = sample_values(values, seed + i)
            a, b = column_doc(vals)
            docs_v.append(a)
            docs_s.append(b)
            dense.append(
                np.concatenate(
                    [dense_features(vals, nr, ws), regex_features(vals), [float(is_generic_header(header))]]
                )
            )
            hdrs.append(normalise_header(header))
        return docs_v, docs_s, np.vstack(dense), hdrs

    @property
    def dense_names(self):
        return DENSE_NAMES + REGEX_FEATURE_NAMES + ["header_is_generic"]

    def fit_transform(self, cols, seed=0):
        dv, ds, dn, hd = self._prep(cols, seed)
        parts = [
            self.val_tfidf.fit_transform(dv),
            self.shape_tfidf.fit_transform(ds),
            sp.csr_matrix(self.scaler.fit_transform(dn).astype(np.float32)),
        ]
        self.hdr_tfidf.fit(hd)
        if self.use_header:
            parts.append(self.hdr_tfidf.transform(hd))
        return sp.hstack(parts, format="csr")

    def transform(self, cols, seed=0):
        dv, ds, dn, hd = self._prep(cols, seed)
        parts = [
            self.val_tfidf.transform(dv),
            self.shape_tfidf.transform(ds),
            sp.csr_matrix(self.scaler.transform(dn).astype(np.float32)),
        ]
        if self.use_header:
            parts.append(self.hdr_tfidf.transform(hd))
        return sp.hstack(parts, format="csr")
