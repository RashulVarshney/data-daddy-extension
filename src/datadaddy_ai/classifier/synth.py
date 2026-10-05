"""Synthetic column generator (Faker). ALL values are synthetic.

Government-ID-like values are random strings in formats that are invalid by construction
(e.g. SSN area 000/666/9xx, Aadhaar-like starting 0/1, PAN-like with 4th letter in XYZQ, NINO-like with
never-issued prefixes). They are not real identifiers and are marked synthetic.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field

import numpy as np
from faker import Faker

from .labels import PII_LABELS

LOCALE_FAKER = {"en_US": "en_US", "en_IN": "en_IN", "en_GB": "en_GB", "de_DE": "de_DE"}
TRAIN_LOCALES = ["en_US", "en_IN"]
UNSEEN_LOCALES = ["en_GB", "de_DE"]

# Header name families. SEEN names are used in training; HELDOUT names only in the unseen-header split.
SEEN_HEADERS = {
    "EMAIL": [
        "email",
        "email_address",
        "emailaddress",
        "e_mail",
        "mail",
        "mail_id",
        "contact_email",
        "work_email",
        "user_email",
        "em",
        "eml",
        "emlid",
    ],
    "PHONE": [
        "phone",
        "phone_number",
        "telephone",
        "mobile",
        "mobile_no",
        "mobile_number",
        "cell",
        "contact_number",
        "contact_no",
        "tel",
        "ph",
        "ph_no",
        "ph_num",
    ],
    "PERSON_NAME": [
        "name",
        "full_name",
        "fullname",
        "customer_name",
        "person",
        "person_name",
        "patient_name",
        "client_name",
        "employee_name",
        "first_name",
        "last_name",
        "surname",
        "nm",
        "cust_nm",
    ],
    "ADDRESS": [
        "address",
        "street_address",
        "addr",
        "address_line1",
        "home_address",
        "mailing_address",
        "shipping_address",
        "billing_address",
        "delivery_address",
        "adr",
        "addr1",
        "st_addr",
    ],
    "DOB": [
        "dob",
        "date_of_birth",
        "birth_date",
        "birthdate",
        "birthday",
        "born",
        "d_o_b",
        "bdate",
        "dt_birth",
        "brth_dt",
    ],
    "GOVT_ID": [
        "ssn",
        "social_security",
        "ss_number",
        "aadhaar",
        "aadhar_no",
        "pan",
        "pan_number",
        "govt_id",
        "national_id",
        "id_number",
        "tax_id",
        "id_proof",
        "citizen_id",
        "kyc_id",
    ],
    "IP_ADDRESS": [
        "ip",
        "ip_address",
        "ipaddress",
        "client_ip",
        "remote_addr",
        "source_ip",
        "src_ip",
        "host_ip",
        "ipv4",
        "last_login_ip",
        "device_ip",
        "ip_addr",
    ],
    "LAT_LONG": [
        "latitude",
        "lat",
        "geo_lat",
        "longitude",
        "lng",
        "lon",
        "long",
        "coordinates",
        "geo",
        "gps",
        "location_coords",
        "position",
        "geolocation",
        "geo_point",
        "latlong",
    ],
}
HELDOUT_HEADERS = {
    "EMAIL": ["contact_mail", "mailbox", "inbox", "eaddr"],
    "PHONE": ["call_me", "contact_line", "whatsapp", "landline", "dial_no"],
    "PERSON_NAME": ["customer", "cardholder", "applicant", "beneficiary", "account_holder"],
    "ADDRESS": ["residence", "domicile", "where_live", "dwelling", "home_loc"],
    "DOB": ["born_on", "natal_date", "date_of_b", "yob_full"],
    "GOVT_ID": ["tax_ref", "citizen_no", "kyc_doc", "identity_ref"],
    "IP_ADDRESS": ["remote_host", "client_addr_v4", "origin_node", "peer"],
    "LAT_LONG": ["y_coord", "x_coord", "loc_pt", "where_at"],
}
MISLEADING_SEEN = ["notes", "ref", "remarks", "value", "data", "info", "details", "misc", "code", "key"]
MISLEADING_UNSEEN = ["comment", "attribute", "field", "entry", "token", "payload", "descr", "item"]
GENERIC_PATTERNS = ["col_{n}", "f{n}", "field_{c}", "var{n}", "c_{n}", "X{n}", "column{n}", "unnamed_{n}"]


def decorate(name: str, rng: np.random.Generator) -> str:
    """Case / delimiter / prefix variation applied to a base header name."""
    r = rng.random()
    if r < 0.55:
        return name
    if r < 0.65:
        return name.upper()
    if r < 0.75:
        return "".join(w.capitalize() if i else w for i, w in enumerate(name.split("_")))  # camelCase
    if r < 0.82:
        return name.replace("_", " ").title()
    if r < 0.88:
        return name.replace("_", "-")
    if r < 0.94:
        return rng.choice(["cust_", "tbl_", "user_", "src_", "x_"]) + name
    return name + rng.choice(["_1", "_2", "_val", "_raw"])


def generic_header(rng: np.random.Generator) -> str:
    pat = GENERIC_PATTERNS[int(rng.integers(len(GENERIC_PATTERNS)))]
    return pat.format(n=int(rng.integers(0, 60)), c=string.ascii_lowercase[int(rng.integers(0, 26))])


@dataclass
class ColumnInstance:
    values: list  # list[str | None]; may contain missing tokens
    header: str | None
    label: str
    locale: str = "en_US"
    source: str = "synthetic"  # synthetic | synthetic_hardneg | real:<dataset>
    header_style: str = "clear"  # clear | abbr | generic | misleading | none | original
    group: str = ""  # grouping key for leakage-safe splits (real columns)
    meta: dict = field(default_factory=dict)


def _luhn_check_digit(digits: str) -> str:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return str((10 - total % 10) % 10)


class SyntheticGenerator:
    """Deterministic given `seed`. One instance owns its RNG and per-locale Faker instances."""

    def __init__(self, seed: int):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._fakers: dict[str, Faker] = {}

    def fk(self, locale: str) -> Faker:
        if locale not in self._fakers:
            f = Faker(LOCALE_FAKER[locale])
            f.seed_instance(self.seed + sum(map(ord, locale)))
            self._fakers[locale] = f
        return self._fakers[locale]

    # ---------- per-class value generators (return list[str]) ----------
    def _digits(self, n: int) -> str:
        return "".join(str(int(x)) for x in self.rng.integers(0, 10, n))

    def _letters(self, n: int, pool: str = string.ascii_uppercase) -> str:
        return "".join(pool[int(i)] for i in self.rng.integers(0, len(pool), n))

    def gen_email(self, n, locale):
        f = self.fk(locale)
        style = int(self.rng.integers(0, 4))
        out = []
        for _ in range(n):
            if style == 0:
                v = f.email()
            elif style == 1:
                v = f.free_email()
            elif style == 2:
                v = f.company_email()
            else:
                v = f"{f.first_name()[:1]}{f.last_name()}{self.rng.integers(1, 999)}@{f.domain_name()}"
            v = re.sub(r"[^\x00-\x7f]", "x", v).replace(" ", "")
            if self.rng.random() < 0.1:
                v = v.upper()
            out.append(v)
        return out

    def gen_phone(self, n, locale):
        f = self.fk(locale)
        style = int(self.rng.integers(0, 4))
        out = []
        for _ in range(n):
            if locale == "en_IN":
                num = str(int(self.rng.integers(6, 10))) + self._digits(9)
                v = [num, f"+91 {num[:5]} {num[5:]}", f"+91-{num}", f"0{num[:5]} {num[5:]}"][style]
            elif locale == "en_US":
                a, b, c = (
                    str(int(self.rng.integers(2, 10))) + self._digits(2),
                    self._digits(3),
                    self._digits(4),
                )
                v = [f"({a}) {b}-{c}", f"{a}-{b}-{c}", f"+1 {a} {b} {c}", f"{a}{b}{c}"][style]
            else:
                v = f.phone_number()
                if style == 3:
                    v = re.sub(r"\D", "", v)
            out.append(v)
        return out

    def gen_person_name(self, n, locale):
        f = self.fk(locale)
        style = str(self.rng.choice(["full", "full", "first", "last", "last_first", "upper", "lower"]))
        out = []
        for _ in range(n):
            if style == "first":
                v = f.first_name()
            elif style == "last":
                v = f.last_name()
            elif style == "last_first":
                v = f"{f.last_name()}, {f.first_name()}"
            else:
                v = f.name()
                if style == "upper":
                    v = v.upper()
                elif style == "lower":
                    v = v.lower()
            out.append(v)
        return out

    def gen_address(self, n, locale):
        f = self.fk(locale)
        style = int(self.rng.integers(0, 3))
        out = []
        for _ in range(n):
            if style == 0:
                v = f.street_address()
            elif style == 1:
                v = f.address().replace("\n", ", ")
            else:
                v = f.address().replace("\n", " ")
            out.append(v)
        return out

    def gen_dob(self, n, locale):
        f = self.fk(locale)
        fmts = {
            "en_US": ["%m/%d/%Y", "%Y-%m-%d", "%B %d, %Y"],
            "en_IN": ["%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"],
            "en_GB": ["%d/%m/%Y", "%Y-%m-%d", "%d %b %Y"],
            "de_DE": ["%d.%m.%Y", "%Y-%m-%d"],
        }[locale]
        fmts = fmts + ["%d-%b-%Y"]
        fmt = fmts[int(self.rng.integers(len(fmts)))]
        mix = self.rng.random() < 0.2  # mixed formats within a column
        return [
            f.date_of_birth(minimum_age=18, maximum_age=90).strftime(
                fmts[int(self.rng.integers(len(fmts)))] if mix else fmt
            )
            for _ in range(n)
        ]

    def gen_govt_id(self, n, locale):
        out = []
        kind = (
            "aadhaar"
            if locale == "en_IN" and self.rng.random() < 0.5
            else "pan"
            if locale == "en_IN"
            else None
        )
        dash = self.rng.random() < 0.7
        for _ in range(n):
            if locale == "en_US":
                area = str(self.rng.choice(["000", "666", "9" + self._digits(2)]))
                a = f"{area}{'-' if dash else ''}{self._digits(2)}{'-' if dash else ''}{self._digits(4)}"
                out.append(a)
            elif locale == "en_IN" and kind == "aadhaar":
                body = str(self.rng.choice(["0", "1"])) + self._digits(10)
                d = body + _luhn_check_digit(body)
                out.append(f"{d[:4]} {d[4:8]} {d[8:]}" if dash else d)
            elif locale == "en_IN":
                out.append(
                    self._letters(3)
                    + str(self.rng.choice(list("XYZQ")))
                    + self._letters(1)
                    + self._digits(4)
                    + self._letters(1)
                )
            elif locale == "en_GB":
                pre = str(self.rng.choice(["BG", "GB", "NK", "KN", "TN", "NT", "ZZ"]))
                out.append(
                    f"{pre} {self._digits(2)} {self._digits(2)} {self._digits(2)} {self._letters(1, 'ABCD')}"
                )
            else:  # de_DE: 11 digits, leading 0 (never issued)
                d = "0" + self._digits(10)
                out.append(f"{d[:3]} {d[3:6]} {d[6:9]} {d[9:]}" if dash else d)
        return out

    def gen_ip(self, n, locale):
        f = self.fk(locale)
        v6 = self.rng.random() < 0.15
        out = []
        for _ in range(n):
            if v6:
                out.append(f.ipv6())
            elif self.rng.random() < 0.3:
                out.append(f.ipv4_private())
            else:
                out.append(f.ipv4())
        return out

    def gen_latlong(self, n, locale):
        box = {
            "en_US": (25, 49, -124, -67),
            "en_IN": (8, 35, 68, 97),
            "en_GB": (50, 59, -8, 2),
            "de_DE": (47, 55, 6, 15),
        }[locale]
        kind = str(self.rng.choice(["lat", "lon", "pair", "pair_paren", "pair_space"]))
        dec = int(self.rng.integers(3, 7))
        out = []
        for _ in range(n):
            lat = self.rng.uniform(box[0], box[1])
            lon = self.rng.uniform(box[2], box[3])
            if kind == "lat":
                v = f"{lat:.{dec}f}"
            elif kind == "lon":
                v = f"{lon:.{dec}f}"
            elif kind == "pair":
                v = f"{lat:.{dec}f},{lon:.{dec}f}"
            elif kind == "pair_paren":
                v = f"({lat:.{dec}f}, {lon:.{dec}f})"
            else:
                v = f"{lat:.{dec}f} {lon:.{dec}f}"
            out.append(v)
        return out

    GEN = {
        "EMAIL": "gen_email",
        "PHONE": "gen_phone",
        "PERSON_NAME": "gen_person_name",
        "ADDRESS": "gen_address",
        "DOB": "gen_dob",
        "GOVT_ID": "gen_govt_id",
        "IP_ADDRESS": "gen_ip",
        "LAT_LONG": "gen_latlong",
    }

    # ---------- synthetic hard negatives (label NONE; flagged source=synthetic_hardneg) ----------
    def gen_hard_negative(self, n, locale) -> tuple[str, list[str]]:
        f = self.fk(locale)
        kinds = [
            "zip_us",
            "pin_in",
            "age",
            "order_id",
            "rating",
            "product_code",
            "price",
            "measure4",
            "event_date",
            "year",
            "version",
            "hex_id",
            "numeric10",
            "numeric12",
            "percent",
            "company",
            "job",
            "city",
            "color",
            "country",
            "uuid",
            "boolean",
            "dotted_code",
            "small_int",
            "numeric9",
            "dashed_code",
            "ip_like_ver",
        ]
        k = kinds[int(self.rng.integers(len(kinds)))]
        r = self.rng
        if k == "zip_us":
            vals = [f.postcode() if locale == "en_US" else f"{r.integers(10000, 99999)}" for _ in range(n)]
        elif k == "pin_in":
            vals = [f"{r.integers(100000, 999999)}" for _ in range(n)]
        elif k == "age":
            vals = [str(int(np.clip(r.normal(40, 15), 1, 99))) for _ in range(n)]
        elif k == "order_id":
            vals = [f"ORD-{r.integers(0, 10**6):06d}" for _ in range(n)]
        elif k == "rating":
            vals = [str(int(r.integers(1, 6))) for _ in range(n)]
        elif k == "product_code":
            vals = [f"{self._letters(2)}{r.integers(100, 9999)}" for _ in range(n)]
        elif k == "price":
            vals = [f"{r.uniform(1, 999):.2f}" for _ in range(n)]
        elif k == "measure4":
            vals = [
                f"{r.uniform(-90, 90):.1f}" if r.random() < 0.5 else f"{r.uniform(0, 1):.4f}"
                for _ in range(n)
            ]
        elif k == "event_date":
            vals = [f.date_between(start_date="-3y", end_date="today").isoformat() for _ in range(n)]
        elif k == "year":
            vals = [str(int(r.integers(1990, 2025))) for _ in range(n)]
        elif k == "version":
            vals = [f"{r.integers(0, 5)}.{r.integers(0, 20)}.{r.integers(0, 50)}" for _ in range(n)]
        elif k == "hex_id":
            vals = [self._letters(8, "abcdef0123456789") for _ in range(n)]
        elif k == "numeric10":
            vals = [str(int(r.integers(2, 10))) + self._digits(9) for _ in range(n)]
        elif k == "numeric9":
            vals = [self._digits(9) for _ in range(n)]
        elif k == "dashed_code":
            vals = [f"{self._digits(3)}-{self._digits(2)}-{self._digits(4)}" for _ in range(n)]
        elif k == "ip_like_ver":
            vals = [
                f"{r.integers(1, 30)}.{r.integers(0, 99)}.{r.integers(0, 99)}.{r.integers(0, 999)}"
                for _ in range(n)
            ]
        elif k == "numeric12":
            vals = [self._digits(12) for _ in range(n)]
        elif k == "percent":
            vals = [f"{r.uniform(0, 100):.1f}%" for _ in range(n)]
        elif k == "company":
            vals = [f.company() for _ in range(n)]
        elif k == "job":
            vals = [f.job() for _ in range(n)]
        elif k == "city":
            vals = [f.city() for _ in range(n)]
        elif k == "color":
            vals = [f.color_name() for _ in range(n)]
        elif k == "country":
            vals = [f.country() for _ in range(n)]
        elif k == "uuid":
            vals = [f.uuid4() for _ in range(n)]
        elif k == "boolean":
            vals = [str(r.choice(["Y", "N", "yes", "no", "True", "False"])) for _ in range(n)]
        elif k == "dotted_code":
            vals = [".".join(str(int(x)) for x in r.integers(0, 300, 4)) + "x" for _ in range(n)]
        else:
            vals = [str(int(r.integers(0, 50))) for _ in range(n)]
        return k, vals

    NONE_HEADERS = {
        "zip_us": ["zip", "zipcode", "postal_code", "zip_code"],
        "pin_in": ["pincode", "pin", "postcode"],
        "age": ["age", "age_years", "customer_age"],
        "order_id": ["order_id", "order_no", "txn_id", "ref_no"],
        "rating": ["rating", "stars", "score"],
        "product_code": ["sku", "product_code", "item_code"],
        "price": ["price", "amount", "total", "cost"],
        "measure4": ["reading", "measurement", "value", "ratio"],
        "event_date": ["order_date", "created_at", "event_date", "signup_date", "updated_on"],
        "year": ["year", "model_year", "yr"],
        "version": ["app_version", "version", "release"],
        "hex_id": ["session_id", "hash", "token_id"],
        "numeric9": ["emp_no", "part_no", "employee_number"],
        "dashed_code": ["part_code", "policy_no", "batch_ref"],
        "ip_like_ver": ["release_tag", "build_no", "sw_ver"],
        "numeric10": ["tracking_no", "invoice_no", "account_ref"],
        "numeric12": ["consignment_no", "card_ref", "serial_no"],
        "percent": ["discount", "pct", "growth"],
        "company": ["company", "employer", "vendor"],
        "job": ["job_title", "occupation", "role"],
        "city": ["city", "town", "district"],
        "color": ["color", "colour", "shade"],
        "country": ["country", "nation", "region"],
        "uuid": ["uuid", "guid", "record_id"],
        "boolean": ["active", "subscribed", "flag"],
        "dotted_code": ["build", "firmware", "code"],
        "small_int": ["qty", "count", "num_items"],
    }

    # ---------- headers / noise ----------
    def pick_header(self, label: str, mode: str, heldout: bool = False) -> tuple[str | None, str]:
        """mode: mixed | clear | generic | none."""
        r = self.rng
        if mode == "mixed":
            p = r.random()
            mode = "clear" if p < 0.5 else "generic" if p < 0.7 else "misleading" if p < 0.85 else "none"
        if mode == "none":
            return None, "none"
        if mode == "generic":
            return generic_header(r), "generic"
        if mode == "misleading":
            pool = MISLEADING_UNSEEN if heldout else MISLEADING_SEEN
            if r.random() < 0.5 and not heldout:  # cross-class names (seen)
                other = [x for x in PII_LABELS if x != label]
                pool = SEEN_HEADERS[other[int(r.integers(len(other)))]]
            return decorate(str(r.choice(pool)), r), "misleading"
        pool = HELDOUT_HEADERS[label] if heldout else SEEN_HEADERS[label]
        return decorate(str(r.choice(pool)), r), "clear"

    def add_noise(self, vals: list[str], strength: float = 1.0) -> list:
        r = self.rng
        miss = float(r.choice([0, 0, 0.05, 0.15, 0.3])) * strength
        out = []
        junk = ["unknown", "xxx", "test", "123", "abc", "see notes", "refused"]
        ws = r.random() < 0.15
        for v in vals:
            u = r.random()
            if u < miss:
                out.append(str(r.choice(["", "", "N/A", "null", "-", "NaN"])) if r.random() < 0.8 else None)
                continue
            if r.random() < 0.015 * strength:
                v = str(r.choice(junk))
            if ws and r.random() < 0.3:
                v = "  " + v + " " * int(r.integers(0, 3))
            out.append(v)
        return out

    def pii_column(
        self,
        label: str,
        locale: str,
        header_mode: str = "mixed",
        heldout_headers: bool = False,
        n_rows: int = 200,
    ) -> ColumnInstance:
        n = int(n_rows)
        vals = getattr(self, self.GEN[label])(n, locale)
        header, style = self.pick_header(label, header_mode, heldout_headers)
        return ColumnInstance(self.add_noise(vals), header, label, locale, "synthetic", style)

    def hard_negative_column(
        self, locale: str, header_mode: str = "mixed", n_rows: int = 200
    ) -> ColumnInstance:
        kind, vals = self.gen_hard_negative(n_rows, locale)
        r = self.rng
        p = r.random()
        mode = (
            header_mode
            if header_mode != "mixed"
            else ("clear" if p < 0.5 else "generic" if p < 0.7 else "misleading" if p < 0.85 else "none")
        )
        if mode == "clear":
            header, style = decorate(str(r.choice(self.NONE_HEADERS[kind])), r), "clear"
        elif mode == "generic":
            header, style = generic_header(r), "generic"
        elif mode == "misleading":  # PII-sounding header on a non-PII column
            lab = PII_LABELS[int(r.integers(len(PII_LABELS)))]
            header, style = decorate(str(r.choice(SEEN_HEADERS[lab])), r), "misleading"
        else:
            header, style = None, "none"
        return ColumnInstance(
            self.add_noise(vals), header, "NONE", locale, "synthetic_hardneg", style, meta={"kind": kind}
        )


def make_pii_set(
    seed: int,
    locales: list[str],
    per_class_per_locale: int,
    header_mode="mixed",
    heldout_headers=False,
    n_rows=200,
) -> list[ColumnInstance]:
    g = SyntheticGenerator(seed)
    out = []
    for loc in locales:
        for lab in PII_LABELS:
            for _ in range(per_class_per_locale):
                out.append(g.pii_column(lab, loc, header_mode, heldout_headers, n_rows))
    return out


def make_hardneg_set(seed: int, locales: list[str], n: int, header_mode="mixed") -> list[ColumnInstance]:
    g = SyntheticGenerator(seed)
    return [g.hard_negative_column(locales[i % len(locales)], header_mode) for i in range(n)]
