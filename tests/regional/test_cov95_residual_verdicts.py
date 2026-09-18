"""Acceptance decisions from real local probe effects and malformed evidence."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import cmd017_verdicts as barrier
from scripts.e2e.regional import cmd018_verdicts as sibling
from scripts.e2e.regional.probes import cmd018_ledger_executor as ledger
from tests.regional import _cov95_residual_support as support

residual_isolation = support.residual_isolation


def test_two_real_ledger_effects_cannot_pass_executed_once_acceptance(
    monkeypatch, tmp_path
):
    path = tmp_path / "ledger.json"
    monkeypatch.setattr(ledger, "LEDGER", path)
    adapter = ledger.LedgerAdapter()
    adapter.execute(SimpleNamespace(idempotency_key="first", command_id="command-a"))
    adapter.execute(SimpleNamespace(idempotency_key="second", command_id="command-b"))
    document = json.loads(path.read_text())
    errors = sibling.executed_once_errors(
        document,
        {"command_id": "command-a", "status": "SUCCEEDED"},
        first_id="command-a",
    )
    assert any("2 attempts" in error for error in errors), errors
    assert any("2 idempotency keys" in error for error in errors), errors
    assert sibling.case_verdict({"execution": errors}) == "FAIL"
    assert sibling.case_verdict({"execution": []}) == "PASS"


@pytest.mark.parametrize("cancelled_id", ["second", "first", "unrelated", None])
def test_cancelled_command_must_be_the_newly_minted_sibling(cancelled_id):
    release = {
        "outcome": {"status": "WAITING", "details": {"remote_command_id": "second"}},
        "cancelled": True,
        "cancelled_command": {"command_id": cancelled_id, "status": "FAILED"},
    }
    errors = sibling.release_errors(release, first_id="first")
    assert bool(errors) is (cancelled_id != "second"), (
        "a failed command from another identity cannot prove this sibling was cancelled"
    )


@pytest.mark.parametrize(
    "defect", ["first-status", "held-status", "mutation", "missing-held-id"]
)
def test_sibling_dispatch_requires_a_real_distinct_hold_without_mutation(defect):
    document = {
        "first": {"status": "WAITING", "remote_command_id": "first"},
        "held": {
            "status": "WAITING",
            "details": {
                "reason": sibling.HOLD_REASON,
                "remote_command_id": "first",
                "held_command_id": "second",
                "mutation_submitted_by_control_plane": False,
            },
        },
        "open_sibling_holds_total": 1,
        "open_command_ids": ["first"],
        "registered_agents": [],
    }
    if defect == "first-status":
        document["first"]["status"] = "FAILED"
    elif defect == "held-status":
        document["held"]["status"] = "SUCCEEDED"
    elif defect == "mutation":
        document["held"]["details"]["mutation_submitted_by_control_plane"] = True
    else:
        document["held"]["details"]["held_command_id"] = ""
    assert sibling.dispatch_errors(document), f"invalid {defect} proof must be rejected"


def test_barrier_counters_and_owner_mismatches_are_explicit_failures():
    assert barrier.ready_errors(
        {"owner": "unrelated", "adapter_has_barrier_coordinator": False}
    ), "a foreign probe owner cannot prove a barrier hold"
    state = {
        "barrier_unavailable_holds_total": 2,
        "claimed_total": 2,
        "unexpected_failures": 1,
        "reported_failures": 0,
        "adapter_executed": False,
    }
    assert barrier.executor_state_errors(state) == [
        "executor recorded an unexpected failure"
    ]
    errors = barrier.breadcrumb_errors(
        {"counters": {"barrier_unavailable_holds_total": 3, "claimed_total": 3}}, state
    )
    assert len(errors) == 2
    assert sibling.executed_once_errors(
        {"physical_count": 1, "keys": ["first"]},
        {"command_id": "wrong", "status": "SUCCEEDED"},
        first_id="first",
    ) == ["the completed command is not the first sibling"]
