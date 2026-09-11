"""PREEMPT-012 takes its predecessor from the formal order, not a fixed id.

The order names PREEMPT-009 (the last PREEMPT contract case that writes
evidence); the runner pinned PREEMPT-011, a pytest-only case that never
writes ``cases/<id>/<id>.json``, so live attempt 1 could not have resolved.
"""

from __future__ import annotations

from pathlib import Path

from scripts.e2e.regional import regional_case_contract as contract
from scripts.e2e.regional import run_preempt012_acceptance as preempt012


def test_the_runner_names_no_fixed_predecessor() -> None:
    source = Path(preempt012.__file__).read_text(encoding="utf-8")
    assert "GF-REGIONAL-PREEMPT-011" not in source, (
        "the predecessor is pinned in the runner"
    )
    assert (
        contract.formal_predecessor(preempt012.CASE_ID) == "GF-REGIONAL-PREEMPT-009"
    ), "the formal order moved; update this test and the case document together"
    assert "predecessor_id=predecessor_id" in source, (
        "the resolved id is not passed to preflight"
    )
