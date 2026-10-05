"""Risk level -> required buyer verification tier. Abstract tiers; no blockchain/wallet code."""

from __future__ import annotations

TIER_NAMES = {1: "wallet-level (basic account)", 2: "identity proof", 3: "document verification"}
REQUIRED_TIER = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 3}
RAW_ROWS_BLOCKED = {"LOW": False, "MEDIUM": False, "HIGH": False, "CRITICAL": True}


def tier_policy(risk_level: str) -> dict:
    if risk_level not in REQUIRED_TIER:
        raise ValueError(f"unknown risk level {risk_level!r}")
    t = REQUIRED_TIER[risk_level]
    return {
        "risk_level": risk_level,
        "required_tier": t,
        "tier_name": TIER_NAMES[t],
        "raw_rows_blocked": RAW_ROWS_BLOCKED[risk_level],
    }


def access_decision(risk_level: str, buyer_tier: int, want_raw_rows: bool = False) -> dict:
    """Pure access decision. `buyer_tier` must come from trusted session state, never from model/user text."""
    p = tier_policy(risk_level)
    tier_ok = isinstance(buyer_tier, int) and buyer_tier >= p["required_tier"]
    raw_ok = tier_ok and not p["raw_rows_blocked"]
    return {
        **p,
        "buyer_tier": buyer_tier,
        "schema_and_stats_allowed": tier_ok,
        "raw_rows_allowed": raw_ok,
        "raw_rows_requested": want_raw_rows,
        "reason": (
            "buyer tier sufficient"
            if tier_ok
            else f"requires tier {p['required_tier']}, buyer has tier {buyer_tier}"
        )
        + ("; raw rows blocked for this risk level" if p["raw_rows_blocked"] else ""),
    }
