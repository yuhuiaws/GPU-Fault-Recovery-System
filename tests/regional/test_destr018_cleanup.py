"""DESTR-018 cleanup: who carries the validated restore and how the reset
incident is closed when that restore is refused (live 2026-09-27)."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from scripts.e2e.regional import destr018_verdicts as verdicts
from scripts.e2e.regional import run_destr018_lifetime_deadline as runner
from tests.regional.test_destr018_lifetime_deadline import (
    BASELINE_IDS,
    EMPTY_JOURNAL,
    T_CANCEL,
    WORKFLOW_ID,
    _command,
    _ledger_row,
    happy_commands,
    happy_ledger,
)


def _cleanup_run(snapshot: dict[str, Any], **fields: Any) -> Any:
    from types import SimpleNamespace

    regional = SimpleNamespace(node_snapshot=lambda node: dict(snapshot))
    base = dict(
        regional=regional,
        settings=SimpleNamespace(node="node-a"),
        support_incident_id="",
        incident_id="inc-reset",
        workflow_request_id="workflow-1",
        profile_version="v1",
    )
    base.update(fields)
    return SimpleNamespace(**base)


ISOLATED = {
    "name": "node-a",
    "ready": "True",
    "unschedulable": True,
    "taints": [
        {"key": "gpu-fault.io/quarantined", "value": "x", "effect": "NoSchedule"}
    ],
    "ownership_annotations": {
        "gpu-fault.io/incident-id": "inc-support-after-workflow-1"
    },
}
CLEAN = {
    "name": "node-a",
    "ready": "True",
    "unschedulable": False,
    "taints": [],
    "ownership_annotations": {},
}


def test_restore_finds_the_support_owner_without_the_verdict_phase(monkeypatch) -> None:
    # Live 2026-09-27: the verdict phase stopped before the escalation read, the
    # support id stayed empty and the cleanup offered only the reset incident,
    # which the restore refused; the node stayed quarantined.
    calls: list[str] = []

    class FakeWarm:
        def __init__(self, regional: Any, cluster: str) -> None:
            pass

        def wait_incident_idle(self, incident_id: str) -> dict[str, Any]:
            return {"incident_id": incident_id}

        def create_restore_workflow(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs["incident_id"])
            if kwargs["incident_id"] != "inc-support-after-workflow-1":
                raise RuntimeError("operator hold retained")
            return {"workflow_request_id": "workflow-restore"}

        def wait_workflow_id(self, workflow_id: str) -> dict[str, Any]:
            return {"status": "SUCCEEDED"}

    monkeypatch.setattr(runner, "WarmSpareLiveFixture", FakeWarm)
    result = runner.restore_isolated_node(_cleanup_run(ISOLATED))
    assert result["incident_id"] == "inc-support-after-workflow-1"
    assert result["restore"] == "SUCCEEDED"
    assert calls == ["inc-support-after-workflow-1"], (
        "the annotated support owner must be tried before the reset incident"
    )


def test_close_reset_incident_falls_back_to_the_isolation_evidence_close(
    monkeypatch,
) -> None:
    from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError

    class FakeWarm:
        closed: list[dict[str, Any]] = []

        def __init__(self, regional: Any, cluster: str) -> None:
            pass

        def incident_by_id(self, incident_id: str) -> dict[str, Any]:
            return {"incident_id": incident_id, "state": "QUARANTINED"}

        def wait_incident_idle(self, incident_id: str) -> dict[str, Any]:
            return {"incident_id": incident_id}

        def create_restore_workflow(self, **kwargs: Any) -> dict[str, Any]:
            raise RegionalFixtureError("cpu Pod probe failed: operator hold retained")

        def close_incident_with_evidence(self, incident_id: str, **kwargs: Any) -> dict:
            self.closed.append({"incident_id": incident_id, **kwargs})
            return {"closed": True, "refusal": None, "state": "RECOVERED"}

    monkeypatch.setattr(runner, "WarmSpareLiveFixture", FakeWarm)
    result = runner.close_reset_incident(_cleanup_run(CLEAN))
    assert result["closed"] is True and result["method"] == "isolation-evidence", result
    assert FakeWarm.closed[0]["incident_id"] == "inc-reset"
    assert FakeWarm.closed[0]["evidence"][0]["node_id"] == "node-a"

    FakeWarm.closed.clear()
    with pytest.raises(RegionalFixtureError):
        # Still isolated: the refusal stands, nothing is closed by hand.
        runner.close_reset_incident(_cleanup_run(ISOLATED))
    assert FakeWarm.closed == []


def test_a_straddling_verify_attempt_row_matches_its_batched_command() -> None:
    # Live DESTR-018 2026-09-27: the batched VERIFY_NO_GPU_CLIENTS is re-sent per
    # attempt as ``<key>/<node>/attempt-N``, so the straddling ledger row was
    # ``.../3/VERIFY_NO_GPU_CLIENTS/<node>/attempt-16/agent-37`` and the matcher,
    # which only knew ``<key>/<node>/agent-G``, reported 0 matches.
    key = f"{WORKFLOW_ID}/3/VERIFY_NO_GPU_CLIENTS"
    row = _ledger_row(
        f"{key}/node-a/attempt-16/agent-37",
        verdicts.WAITING_STEP,
        started=T_CANCEL - timedelta(seconds=3),
        completed=T_CANCEL + timedelta(seconds=4),
    )
    row["agent_generation"] = 37
    compound = _command(
        "QUIESCE_GPU_SERVICES",
        status="FAILED",
        status_source=verdicts.CANCELLED_BY_TIMEOUT,
        command_id="remote-compound",
    )
    compound["batched_steps"] = [
        {
            "idempotency_key": key,
            "step": {"operation": verdicts.WAITING_STEP, "node_ids": ["node-a"]},
        }
    ]
    assert verdicts.ledger_command_matches(row, compound), (
        "the per-attempt VERIFY row must match its batched compound command"
    )
    other_generation = dict(row, command_id=f"{key}/node-a/attempt-16/agent-36")
    assert not verdicts.ledger_command_matches(other_generation, compound), (
        "another agent generation must not match"
    )
    other_node = dict(row, command_id=f"{key}/node-b/attempt-16/agent-37")
    assert not verdicts.ledger_command_matches(other_node, compound), (
        "another node must not match"
    )
    errors = verdicts.straddling_row_errors(
        [*happy_ledger(), row],
        t_cancel=T_CANCEL,
        baseline_command_ids=BASELINE_IDS,
        kernel_journal=EMPTY_JOURNAL,
        commands=[*happy_commands(), compound],
    )
    assert errors == [], errors
