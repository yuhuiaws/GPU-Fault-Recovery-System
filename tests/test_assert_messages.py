from __future__ import annotations

import ast
from pathlib import Path

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[1]
MODULE = lazy_script_module(ROOT / "scripts/check-assert-messages.py")


def expression(value: str) -> ast.expr:
    return ast.parse(f"assert {value}").body[0].test


def test_bare_boolean_classifier_excludes_comparisons() -> None:
    assert MODULE.is_bare_boolean(expression("ready")), "Name must be classified"
    assert MODULE.is_bare_boolean(expression("not ready")), (
        "Negated Name must be classified"
    )
    assert MODULE.is_bare_boolean(expression("store.has_pending(node)")), (
        "Call must be classified"
    )
    assert MODULE.is_bare_boolean(expression("state.ready")), (
        "Attribute must be classified"
    )
    assert not MODULE.is_bare_boolean(expression("actual == expected")), (
        "Compare already has pytest value diagnostics"
    )


def test_assert_message_ratchet_rejects_growth_slack_and_target() -> None:
    baseline = {"initial_missing_messages": 100, "missing_messages": 60}

    assert MODULE.failures(baseline, 61) == [
        "bare boolean asserts without messages grew from 60 to 61",
        "bare boolean assert target exceeded: 61 > 60 (60% of initial 100)",
    ]
    assert MODULE.failures(baseline, 59) == [
        "assert message baseline has slack: "
        "current 59, recorded 60; run --write-baseline"
    ]
    assert MODULE.failures(baseline, 60) == []
