"""DESTR-019 on a graded Hopper NVSwitch node: the audited command is the reset.

The case restarts the Node Agent and then audits the next real command's trail.
Its injection is DESTR-010's solo XID 45, and on a node whose NVLink fabric the
control plane grades (batch 128) that is no longer one Fabric Manager restart
but the full GPU + NVSwitch reset chain. The audit then reads the chain's
RESET_ALL_GPUS_NVSWITCHES -- one ledger row at attempt 1, three journald lines
with the nine identifier fields -- and the health counters must move exactly as
the Agent's ledger did since the restart. A node the product does not grade
keeps the one-command contract.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import destr019_verdicts as verdicts
from scripts.e2e.regional import fabric_restart_shape as shapes
from scripts.e2e.regional import run_destr019_agent_restart_ledger as destr019
from tests.regional._cov95_destr_agent import (
    GENERATION,
    INCIDENT,
    NODE,
    WORKFLOW,
    AgentHarness,
    audit,
    health,
    journal,
)
from tests.regional._cov95_destr_warm import NOW

GPUS = [f"GPU-{index}" for index in range(8)]
GRADED = {
    "available": True,
    "graded_gpus": GPUS,
    "states": dict.fromkeys(GPUS, 3.0),
    "product": "NVIDIA H200",
}
RESET_KEY = f"{WORKFLOW}/4/RESET_ALL_GPUS_NVSWITCHES"
RESET_COMMAND_ID = f"{RESET_KEY}/{NODE}/agent-{GENERATION}"
OWNERS = {
    "FREEZE_EVIDENCE": "gpu-fault-control-plane",
    "MARK_UNSCHEDULABLE": "gpu-fault-kubernetes-adapter",
    "RESTORE_SCHEDULING": "gpu-fault-kubernetes-adapter",
    "VALIDATE_GPU": "gpu-fault-validation-adapter",
    "VALIDATE_FABRIC": "gpu-fault-validation-adapter",
}


def _chain_state() -> dict[str, Any]:
    steps = []
    for operation in shapes.FABRIC_RESET_STEPS:
        step: dict[str, Any] = {
            "operation": operation,
            "execution_owner": OWNERS.get(operation, "gpu-fault-node-agent"),
            "gpu_uuids": GPUS,
            "parameters": {},
        }
        if operation == "RESET_ALL_GPUS_NVSWITCHES":
            step["parameters"] = {"fabric_partition": "c/n/local-nvswitch", "xid": 45}
        steps.append(step)
    commands = [
        {
            "status": "SUCCEEDED",
            "idempotency_key": f"{WORKFLOW}/{index}/{operation}",
            "step": {"operation": operation},
            "batched_steps": [],
        }
        for index, operation in (
            (1, "MARK_UNSCHEDULABLE"),
            (2, "QUIESCE_GPU_SERVICES"),
            (3, "VERIFY_NO_GPU_CLIENTS"),
            (4, "RESET_ALL_GPUS_NVSWITCHES"),
            (5, "RESTORE_GPU_SERVICES"),
            (8, "RESTORE_SCHEDULING"),
        )
    ]
    return {
        "event": {"xid": 45},
        "decision": {"official_action": "RESTART_FM", "disposition": "EXECUTABLE"},
        "workflow": {
            "request_id": WORKFLOW,
            "status": "SUCCEEDED",
            "fencing_token": 2,
            "official_steps": steps,
            "completed_operations": list(shapes.FABRIC_RESET_STEPS),
        },
        "incident": {"incident_id": INCIDENT, "node_ids": [NODE]},
        "commands": commands,
    }


def _row(operation: str, state: str = "SUCCEEDED", attempt: int = 1) -> dict[str, Any]:
    return {
        "command_id": f"{operation}-id",
        "operation": operation,
        "state": state,
        "attempt": attempt,
    }


CHAIN_ROWS = [
    _row("QUIESCE_GPU_SERVICES"),
    _row("VERIFY_NO_GPU_CLIENTS", "FAILED"),
    _row("VERIFY_NO_GPU_CLIENTS"),
    _row("RESET_ALL_GPUS_NVSWITCHES"),
    _row("RESTORE_GPU_SERVICES"),
]


def test_a_graded_node_needs_the_chain_s_node_actions() -> None:
    errors = verdicts.preflight_errors(
        node={"ready": "True"},
        agent={
            "lifecycle_state": "ACTIVE",
            "generation": GENERATION,
            "allowed_operations": ["RESTART_FABRIC_MANAGER"],
        },
        profile={},
        workloads=[],
        queue={"depth": 0},
        remote_commands={},
        recent_event=None,
        host={
            "agent": {"ActiveState": "active"},
            "ledger": {"present": True, "user_version": verdicts.LEDGER_SCHEMA_VERSION},
        },
        tests_passed=True,
        shape=shapes.FABRIC_RESET,
    )

    assert sorted(errors) == sorted(
        f"target Node Agent does not allow {item}"
        for item in shapes.FABRIC_RESET_NODE_STEPS
    )


def test_the_chain_passes_and_the_audited_command_is_its_reset() -> None:
    state = _chain_state()

    assert (
        verdicts.workflow_errors(
            state, node=NODE, shape=shapes.FABRIC_RESET, expected_gpus=GPUS
        )
        == []
    )
    command = verdicts.audited_command(state, shapes.FABRIC_RESET)
    assert (
        verdicts.expected_command_id(command, node=NODE, generation=GENERATION)
        == RESET_COMMAND_ID
    )
    assert (
        verdicts.AUDITED_OPERATION[shapes.FABRIC_RESET] == "RESET_ALL_GPUS_NVSWITCHES"
    )


def test_the_two_step_restart_or_a_widened_incident_fails_on_a_graded_node() -> None:
    state = _chain_state()
    state["workflow"]["official_steps"] = state["workflow"]["official_steps"][:1]
    widened = _chain_state()
    widened["incident"]["node_ids"] = [NODE, "node-b"]

    assert verdicts.workflow_errors(
        state, node=NODE, shape=shapes.FABRIC_RESET, expected_gpus=GPUS
    ), "the two-step restart is no pass on a graded node"
    errors = verdicts.workflow_errors(
        widened, node=NODE, shape=shapes.FABRIC_RESET, expected_gpus=GPUS
    )
    assert any("not just node-a" in item for item in errors), errors


def test_the_counters_follow_the_ledger_on_a_graded_node() -> None:
    assert verdicts.counter_expectation(shapes.FABRIC_RESET, {"rows": CHAIN_ROWS}) == {
        "accepted": 5,
        "completed": 4,
        "failed": 1,
        "rejected": 0,
    }
    assert verdicts.counter_expectation(shapes.FABRIC_MANAGER_RESTART) == {
        "accepted": 1,
        "completed": 1,
        "failed": 0,
        "rejected": 0,
    }
    assert verdicts.chain_ledger_errors({"rows": CHAIN_ROWS}) == []


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([*CHAIN_ROWS, _row("RESTART_FABRIC_MANAGER")], "outside the chain"),
        ([*CHAIN_ROWS, _row("RESET_ALL_GPUS_NVSWITCHES", attempt=2)], "2 RESET_ALL"),
        (CHAIN_ROWS[:1] + CHAIN_ROWS[3:], "no client verification"),
    ],
    ids=["foreign-operation", "second-reset-attempt", "no-verification"],
)
def test_a_chain_that_is_not_one_window_fails(rows, message) -> None:
    errors = verdicts.chain_ledger_errors({"rows": rows})

    assert any(message in item for item in errors), errors


def _graded_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentHarness:
    h = AgentHarness(tmp_path, monkeypatch)
    h.grading = deepcopy(GRADED)
    h.agent["allowed_operations"] = list(shapes.FABRIC_RESET_NODE_STEPS)
    h.state = _chain_state()
    lines = journal()
    for line in lines["lines"]:
        line["fields"].update(
            command_id=RESET_COMMAND_ID,
            operation="RESET_ALL_GPUS_NVSWITCHES",
            gpu_count="8",
        )
    h.journal = lines
    reset_audit = audit()
    reset_audit["rows"][0].update(
        command_id=RESET_COMMAND_ID,
        operation="RESET_ALL_GPUS_NVSWITCHES",
        gpu_uuid_count=8,
    )
    h.audit = reset_audit
    h.chain_audit = {
        "rows": [
            {**row, "started_at": (NOW + timedelta(seconds=index)).isoformat()}
            for index, row in enumerate(CHAIN_ROWS)
        ]
    }
    h.health_after = health({"accepted": 5, "completed": 4, "failed": 1, "rejected": 0})
    return h


def test_the_runner_audits_the_reset_of_a_graded_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = _graded_harness(tmp_path, monkeypatch)
    preflight = h.plan(tmp_path)
    assert preflight["errors"] == [], preflight
    details = destr019.plan_details(h.settings, preflight)
    assert details["risk"] == "destructive", details["risk"]
    assert details["workflow_shape"] == shapes.FABRIC_RESET
    assert "full GPU + NVSwitch reset chain" in details["mutation"]

    code, report = h.execute(tmp_path)

    assert code == 0 and report["errors"] == [], report
    audits = [detail for name, detail in h.calls if name == "agent.ledger-audit"]
    assert len(audits) == 2, audits
    assert audits[0]["args"] == ("--command-id", RESET_COMMAND_ID), audits
    assert audits[1]["args"][0] == "--since", "the chain audit reads every attempt"


def test_counters_that_disagree_with_the_ledger_fail_the_graded_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = _graded_harness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.health_after = health({"accepted": 1, "completed": 1, "failed": 0, "rejected": 0})

    code, report = h.execute(tmp_path)

    assert code == 1, report
    assert "counters" in " ".join(report["errors"]), report["errors"]
