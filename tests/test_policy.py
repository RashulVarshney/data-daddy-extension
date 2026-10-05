import pytest

from datadaddy_ai.policy.risk import (
    detect_quasi_identifiers,
    quasi_identifier_uniqueness,
    risk_level,
    risk_score,
)
from datadaddy_ai.policy.tiers import access_decision, tier_policy


def test_no_pii_is_low():
    r = risk_score(["NONE"] * 10)
    assert r["level"] == "LOW" and r["score"] == 0.0


def test_score_monotone_in_added_classes():
    a = risk_score(["NONE"] * 8 + ["IP_ADDRESS"])["score"]
    b = risk_score(["NONE"] * 7 + ["IP_ADDRESS", "EMAIL"])["score"]
    c = risk_score(["NONE"] * 6 + ["IP_ADDRESS", "EMAIL", "PHONE"])["score"]
    assert 0 < a < b < c <= 1


def test_govt_id_floor_high_and_critical_with_identifier():
    assert (
        risk_score(["GOVT_ID", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE"])[
            "level"
        ]
        == "HIGH"
    )
    assert risk_score(["GOVT_ID", "PERSON_NAME", "NONE"])["level"] == "CRITICAL"


def test_qi_uniqueness_raises_risk_only_with_two_columns():
    base = risk_score(["NONE"] * 4)["score"]
    assert risk_score(["NONE"] * 4, 1.0, 1)["score"] == base
    assert risk_score(["NONE"] * 4, 1.0, 3)["score"] > base


def test_uniqueness_function():
    assert quasi_identifier_uniqueness([("a", 1), ("a", 1), ("b", 2), ("c", 3)]) == 0.5
    assert quasi_identifier_uniqueness([]) == 0.0


def test_detect_quasi_identifiers():
    heads = ["zip", "gender", "salary", "x"]
    cols = [["12345"], ["M", "F"], ["10"], ["1990-01-01"]]
    assert detect_quasi_identifiers(heads, cols, ["NONE", "NONE", "NONE", "DOB"]) == [0, 1, 3]


@pytest.mark.parametrize("level,tier", [("LOW", 1), ("MEDIUM", 2), ("HIGH", 3), ("CRITICAL", 3)])
def test_tier_mapping(level, tier):
    assert tier_policy(level)["required_tier"] == tier


def test_critical_always_blocks_raw_rows_even_for_top_tier():
    d = access_decision("CRITICAL", 3, want_raw_rows=True)
    assert d["raw_rows_blocked"] and not d["raw_rows_allowed"] and d["schema_and_stats_allowed"]


def test_low_tier_buyer_denied_on_high_risk():
    d = access_decision("HIGH", 1)
    assert not d["schema_and_stats_allowed"] and not d["raw_rows_allowed"]


def test_unknown_level_rejected():
    with pytest.raises(ValueError):
        tier_policy("SEVERE")


def test_level_thresholds():
    assert [risk_level(x) for x in (0.0, 0.3, 0.6, 0.9)] == ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
