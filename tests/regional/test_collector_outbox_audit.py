"""The Collector outbox audit fails loudly, not through ``assert``."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripts.e2e.regional import audit_collector_outbox as audit


def test_the_audit_has_no_bare_asserts() -> None:
    tree = ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
    asserts = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    assert asserts == [], "a bare assert is stripped under python -O"


def test_expect_names_the_expectation_and_the_observed_value() -> None:
    with pytest.raises(audit.OutboxAuditFailure, match=r"order: observed \[1, 2\]"):
        audit.expect(False, "order", [1, 2])
    audit.expect(True, "never raised")


def test_the_audit_still_passes_end_to_end(capsys) -> None:
    audit.main()
    out = capsys.readouterr().out
    assert out.startswith("PASS"), "the audit verdict line leads the output"
    assert "'replay_order': [1, 2, 1]" in out
