"""Explicit native prerequisites that the generic isolated runner cannot grant."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

NATIVE_SELECTORS = (
    "tests/execution/test_workload_withdrawal.py",
    "tests/regional/test_cov95_notify008_postgres.py",
)
NATIVE_CASES = frozenset({"GF-REGIONAL-PREEMPT-033", "GF-REGIONAL-NOTIFY-008"})


def native_prerequisite_reason(case: Mapping[str, Any]) -> str | None:
    selectors = [
        str(case.get("pytest_nodeid") or ""),
        *[str(item) for item in case.get("command") or []],
    ]
    if case.get("id") not in NATIVE_CASES and not any(
        item.split("::", 1)[0] in NATIVE_SELECTORS for item in selectors
    ):
        return None
    return (
        "BLOCKED_PREREQUISITE: this complete contract includes native PostgreSQL "
        "assertions. Run it serially through the owned isolated PostgreSQL gate "
        "and retain its source-bound complete pytest receipt. The generic runner "
        "does not grant a database allocation or forward ambient Store URLs; "
        "missing native assertions or skipped variants cannot count as PASS."
    )
