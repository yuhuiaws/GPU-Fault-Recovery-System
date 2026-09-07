"""Contract tests for GF-REGIONAL-DESTR-017.

The case proves the generation fence: a node rebooted from outside the control
plane while a RESET_GPU workflow waits on it comes back with a new boot id and a
new Node Agent generation, and every in-flight command of the retired generation
is refused. These tests pin what the runner will accept as that proof, against
synthetic control-plane, node and CloudTrail snapshots.

They pass without the shared catalog entry: the confirmation token is checked
against the contract derivation rather than against the catalog row.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.fleet import AgentRecord
from gpu_fault.models import WorkflowOperation
from scripts.e2e.regional import destr017_verdicts as verdicts
from scripts.e2e.regional import run_destr017_out_of_band_reboot_fence as destr017
from scripts.e2e.regional.regional_case_contract import RegionalCaseMetadata
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureError,
    RegionalLiveFixture,
)

NODE = "node-b"
OTHER = "node-c"
REQUEST = "workflow-destr017-test"
SUPPORT_REQUEST = f"workflow-support-after-{REQUEST}"
INCIDENT = "inc-destr017-test"
GENERATION = 7
WINDOW_START = "2026-09-06T10:02:00+00:00"
WINDOW_END = "2026-09-06T10:09:00+00:00"
FENCE_ERROR = (
    f"maintenance agent fence failed for {NODE}: agent generation changed "
    f"from {GENERATION} to {GENERATION + 1}"
)
BOOT_BEFORE = "11111111-1111-4111-8111-111111111111"
BOOT_AFTER = "22222222-2222-4222-8222-222222222222"


def _at(minute: int, second: int = 0) -> str:
    return f"2026-09-06T10:{minute:02d}:{second:02d}+00:00"


def _step(operation: str, *, nodes: list[str] | None = None) -> dict[str, Any]:
    return {
        "operation": operation,
        "execution_owner": "test",
        "node_ids": [NODE] if nodes is None else list(nodes),
        "gpu_uuids": [],
        "parameters": {},
    }


def _execution(
    index: int,
    operation: str,
    status: str,
    *,
    started_at: str,
    updated_at: str,
    error: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "step_index": index,
        "operation": operation,
        "status": status,
        "phase": "official",
        "error": error,
        "details": dict(details or {}),
        "started_at": started_at,
        "updated_at": updated_at,
    }


def quiesce_execution() -> dict[str, Any]:
    """The step that pins the generation and the window the fence uses."""

    return _execution(
        2,
        "QUIESCE_GPU_SERVICES",
        "SUCCEEDED",
        started_at=_at(1, 30),
        updated_at=WINDOW_START,
        details={
            "agent_generations": {NODE: GENERATION},
            "maintenance_window_started_at": WINDOW_START,
            "maintenance_window_expires_at": WINDOW_END,
        },
    )


def fenced_workflow(**overrides: Any) -> dict[str, Any]:
    """The terminal record the runner must see: FAILED at the waiting step, the
    generation fence in the record, no reset anywhere."""

    workflow = {
        "request_id": REQUEST,
        "incident_id": INCIDENT,
        "status": "FAILED",
        "official_action": "RESET_GPU",
        "error": f"step 3 VERIFY_NO_GPU_CLIENTS failed: {FENCE_ERROR}",
        "official_steps": [
            _step(operation) for operation in verdicts.OFFICIAL_OPERATIONS
        ],
        "completed_operations": [
            "FREEZE_EVIDENCE",
            "MARK_UNSCHEDULABLE",
            "QUIESCE_GPU_SERVICES",
        ],
        "step_executions": [
            _execution(
                0,
                "FREEZE_EVIDENCE",
                "SUCCEEDED",
                started_at=_at(0),
                updated_at=_at(0, 20),
            ),
            _execution(
                1,
                "MARK_UNSCHEDULABLE",
                "SUCCEEDED",
                started_at=_at(0, 25),
                updated_at=_at(1, 20),
            ),
            quiesce_execution(),
            _execution(
                3,
                "VERIFY_NO_GPU_CLIENTS",
                "FAILED",
                started_at=_at(2, 5),
                updated_at=_at(6, 40),
                error=FENCE_ERROR,
            ),
            _execution(
                5,
                "RESTORE_GPU_SERVICES",
                "FAILED",
                started_at=_at(6, 45),
                updated_at=_at(7, 10),
                error=FENCE_ERROR,
            ),
        ],
        "superseded_step_indexes": [],
        "blocked_kind": None,
        "blocked_reasons": [],
        "lifetime_deadline_at": _at(59),
    }
    workflow.update(overrides)
    return workflow


def fenced_incident(**overrides: Any) -> dict[str, Any]:
    incident = {
        "incident_id": INCIDENT,
        "state": "QUARANTINED",
        "node_ids": [NODE],
        "workflow_request_id": REQUEST,
    }
    incident.update(overrides)
    return incident


def support_workflow(**overrides: Any) -> dict[str, Any]:
    workflow = {
        "request_id": SUPPORT_REQUEST,
        "status": "SUCCEEDED",
        "official_action": "ESCALATE_OPERATOR",
        "official_steps": [
            _step(operation) for operation in verdicts.SUPPORT_OPERATIONS
        ],
        "predecessor_workflow_id": None,
        "lifetime_deadline_at": _at(59),
    }
    workflow.update(overrides)
    return workflow


def support_incident(**overrides: Any) -> dict[str, Any]:
    incident = {
        "incident_id": f"inc-support-after-{REQUEST}",
        "state": "ESCALATED",
        "node_ids": [NODE],
        "workflow_request_id": SUPPORT_REQUEST,
    }
    incident.update(overrides)
    return incident


def _row(
    command_id: str, operation: str, state: str, *, completed_at: str
) -> dict[str, Any]:
    return {
        "command_id": command_id,
        "operation": operation,
        "state": state,
        "attempt": 1,
        "completed_at": completed_at,
        "started_at": completed_at,
    }


def host_before(**overrides: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "captured_at": _at(0),
        "boot_id": BOOT_BEFORE,
        "kmsg_exists": True,
        "kmsg_writable": True,
        "gpu_inventory": [
            {"index": index, "pci_bdf": f"0000:{index:02d}:00.0"} for index in range(8)
        ],
        "compute_clients": [],
        "services": {
            "nvidia-fabricmanager.service": {"ActiveState": "active"},
            "kubelet.service": {"ActiveState": "active"},
        },
        "ledger": [
            _row(
                "old/4/VERIFY_NO_GPU_CLIENTS/commit",
                "VERIFY_NO_GPU_CLIENTS",
                "SUCCEEDED",
                completed_at="2026-09-05T08:00:00+00:00",
            )
        ],
        "gpu_fault_timers": [],
        "quiesce_states": [],
        "kernel_reset_journal": {"target_reset_count": 0},
    }
    snapshot.update(overrides)
    return snapshot


def host_after(**overrides: Any) -> dict[str, Any]:
    snapshot = host_before(
        captured_at=_at(20),
        boot_id=BOOT_AFTER,
        ledger=[
            *host_before()["ledger"],
            _row(
                f"{REQUEST}/2/QUIESCE_GPU_SERVICES/commit",
                "QUIESCE_GPU_SERVICES",
                "SUCCEEDED",
                completed_at=WINDOW_START,
            ),
            _row(
                f"{REQUEST}/3/VERIFY_NO_GPU_CLIENTS/commit",
                "VERIFY_NO_GPU_CLIENTS",
                "FAILED",
                completed_at=_at(6, 40),
            ),
        ],
    )
    snapshot.update(overrides)
    return snapshot


def reboot_status(**overrides: Any) -> dict[str, Any]:
    status = {
        "armed": True,
        "fired": True,
        "boot_id_before_reboot": BOOT_BEFORE,
        "boot_id_now": BOOT_AFTER,
        "observed_boot_ids": [BOOT_BEFORE, BOOT_AFTER],
        "boot_changes": 1,
        "reboot_cancelled_at": None,
    }
    status.update(overrides)
    return status


def agent_before(**overrides: Any) -> dict[str, Any]:
    """The store's agent record, dumped from the real ``AgentRecord`` model.

    Built from the model rather than by hand so a key the model does not have
    (``capability_mode``, ``supported_operations``, ``incarnation_id``) cannot
    make a verdict pass here that would read ``None`` on a live cluster.
    Overrides are applied to the dump, so a test may still hand the verdict an
    impossible value (``generation=None``) on purpose.
    """

    seen = datetime(2026, 9, 6, 9, 0, tzinfo=timezone.utc)
    record: dict[str, Any] = AgentRecord(
        cluster_id="cluster-a",
        node_id=NODE,
        endpoint="https://10.0.0.2:8443",
        agent_version="1.0.0",
        artifact_sha256="a" * 64,
        policy_version="p1",
        runtime_profile_version="p1",
        config_digest="c" * 64,
        allowed_operations=[
            WorkflowOperation(name) for name in verdicts.AGENT_OPERATIONS
        ],
        boot_id=BOOT_BEFORE,
        agent_incarnation_id="inc-a",
        first_seen_at=seen,
        last_seen_at=seen,
        generation=GENERATION,
    ).model_dump(mode="json")
    record.update(overrides)
    return record


def agent_after(**overrides: Any) -> dict[str, Any]:
    record = agent_before(
        generation=GENERATION + 1,
        agent_incarnation_id="inc-b",
        boot_id=BOOT_AFTER,
        retired_incarnation_ids=["inc-a"],
    )
    record.update(overrides)
    return record


def owned_profile(**overrides: Any) -> dict[str, Any]:
    """A runtime profile whose gpuReset is OWN by the Node Agent."""

    profile: dict[str, Any] = {
        "profile_version": "p1",
        "warnings": [],
        "capabilities": [
            {
                "capability": "gpuReset",
                "mode": "OWN",
                "owner": "gpu-fault-node-agent",
                "adapter": "node-action",
            }
        ],
    }
    profile.update(overrides)
    return profile


def fence_command(**overrides: Any) -> dict[str, Any]:
    """The remote command row the refused dispatch leaves behind: the step
    execution only points at it, the fence text is here."""

    command: dict[str, Any] = {
        "command_id": f"{REQUEST}/3/VERIFY_NO_GPU_CLIENTS/{NODE}/attempt-1",
        "status": "FAILED",
        "status_source": "executor",
        "step": {"operation": "VERIFY_NO_GPU_CLIENTS", "node_ids": [NODE]},
        "error": FENCE_ERROR,
        "result_details": {},
    }
    command.update(overrides)
    return command


# --------------------------------------------------------------------------- #
# Case identity
# --------------------------------------------------------------------------- #
def test_confirmation_token_matches_the_case_contract_derivation() -> None:
    metadata = RegionalCaseMetadata(
        case_id=destr017.CASE_ID,
        title="",
        category="regional-destructive-acceptance",
        level="staging",
        risk="destructive",
        automation="manual",
        procedure="docs/x.md#gf-regional-destr-017",
        predecessor=destr017.PREDECESSOR_CASE_ID,
    )
    assert destr017.CASE_ID == "GF-REGIONAL-DESTR-017"
    assert destr017.CONFIRMATION == metadata.confirmation == "DESTR017_EXECUTE"
    assert destr017.PREDECESSOR_CASE_ID == "GF-REGIONAL-DESTR-002"


def test_the_case_arms_its_holder_on_quiesce_not_on_verify() -> None:
    """On one node the verify step is the step the holder has to break, so the
    holder must already be open when the first verify attempt runs."""

    assert destr017.ARM_LEDGER_OPERATION == "QUIESCE_GPU_SERVICES"
    assert destr017.ARM_LEDGER_OPERATION != verdicts.FENCE_STEP_OPERATION


# --------------------------------------------------------------------------- #
# Fence variants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (FENCE_ERROR, verdicts.FENCE_AGENT_GENERATION),
        (
            f"maintenance agent fence failed for {NODE}: no active agent endpoint",
            verdicts.FENCE_TRANSPORT,
        ),
        (
            f"quiesce maintenance window expired at {WINDOW_END}",
            verdicts.FENCE_WINDOW_EXPIRED,
        ),
        ("GPU device clients are still active: 1", verdicts.FENCE_DEVICE_CLIENTS),
        ("step 3 stayed non-terminal for 600s", verdicts.FENCE_WAITING_CAP),
        ("", verdicts.FENCE_UNKNOWN),
        ("connection reset by peer", verdicts.FENCE_UNKNOWN),
    ],
)
def test_fence_variant_classifies_every_fail_closed_refusal(
    error: str, expected: str
) -> None:
    assert verdicts.fence_variant(error) == expected


def test_the_generation_fence_is_recognised_inside_the_maintenance_wrapper() -> None:
    """The barrier wraps the generation change in its own message; the more
    specific classification has to win, or the case would accept a transport
    failure as its proof."""

    assert verdicts.MAINTENANCE_FENCE_LITERAL in FENCE_ERROR
    assert verdicts.GENERATION_FENCE_LITERAL in FENCE_ERROR
    assert verdicts.fence_variant(FENCE_ERROR) == verdicts.FENCE_AGENT_GENERATION


def test_fence_evidence_records_the_variant_of_every_terminal_error() -> None:
    evidence = verdicts.fence_evidence(fenced_workflow())
    assert evidence["workflow_status"] == "FAILED"
    assert evidence["verify_statuses"] == ["FAILED"]
    assert evidence["verify_variants"] == [verdicts.FENCE_AGENT_GENERATION]
    assert evidence["compensation_statuses"] == ["FAILED"]
    assert evidence["compensation_variants"] == [verdicts.FENCE_AGENT_GENERATION]
    assert "RESET_GPU" not in evidence["executed_operations"]
    assert evidence["command_statuses"] == []


def test_fence_evidence_reads_the_refusal_off_the_remote_command() -> None:
    """A remote step's execution record carries pointers; the fence text is on
    the command row, and the evidence has to say which variant it was."""

    evidence = verdicts.fence_evidence(fenced_workflow(), [fence_command()])
    assert evidence["command_statuses"] == [
        {
            "operation": "VERIFY_NO_GPU_CLIENTS",
            "status": "FAILED",
            "variant": verdicts.FENCE_AGENT_GENERATION,
        }
    ]
    in_details = fence_command(error=None, result_details={"error": FENCE_ERROR})
    evidence = verdicts.fence_evidence(fenced_workflow(), [in_details])
    assert evidence["command_statuses"][0]["variant"] == (
        verdicts.FENCE_AGENT_GENERATION
    )


# --------------------------------------------------------------------------- #
# Workflow verdicts
# --------------------------------------------------------------------------- #
def test_the_fenced_workflow_contract_accepts_the_expected_terminal_record() -> None:
    assert (
        verdicts.workflow_errors(fenced_workflow(), fenced_incident(), node=NODE) == []
    )


def test_a_successful_verify_is_a_failure_of_the_case() -> None:
    workflow = fenced_workflow(
        step_executions=[
            *fenced_workflow()["step_executions"][:3],
            _execution(
                3,
                "VERIFY_NO_GPU_CLIENTS",
                "SUCCEEDED",
                started_at=_at(2, 5),
                updated_at=_at(2, 30),
            ),
            fenced_workflow()["step_executions"][4],
        ]
    )
    errors = verdicts.workflow_errors(workflow, fenced_incident(), node=NODE)
    assert any("device holder did not hold" in item for item in errors), errors


@pytest.mark.parametrize("operation", sorted(verdicts.FORBIDDEN_EXECUTIONS))
def test_no_node_mutation_may_be_executed_by_the_fenced_workflow(
    operation: str,
) -> None:
    executions = [
        *fenced_workflow()["step_executions"],
        _execution(4, operation, "FAILED", started_at=_at(7), updated_at=_at(7, 30)),
    ]
    errors = verdicts.workflow_errors(
        fenced_workflow(step_executions=executions), fenced_incident(), node=NODE
    )
    assert any("must never reach" in item for item in errors), errors


def test_a_completed_reset_fails_the_case_even_without_an_execution_row() -> None:
    workflow = fenced_workflow(
        completed_operations=["QUIESCE_GPU_SERVICES", "RESET_GPU"]
    )
    errors = verdicts.workflow_errors(workflow, fenced_incident(), node=NODE)
    assert any("completed GPU reset" in item for item in errors), errors


def test_a_workflow_without_the_generation_fence_does_not_prove_the_case() -> None:
    """A run whose waiting step died of the window or of the per-step cap, and
    whose compensation then succeeded, never fenced the new boot out."""

    workflow = fenced_workflow(
        error=f"step 3 VERIFY_NO_GPU_CLIENTS failed: {verdicts.WAITING_CAP_LITERAL} 600s",
        step_executions=[
            *fenced_workflow()["step_executions"][:3],
            _execution(
                3,
                "VERIFY_NO_GPU_CLIENTS",
                "FAILED",
                started_at=_at(2, 5),
                updated_at=_at(12, 5),
                error="step 3 stayed non-terminal for 600s",
            ),
            _execution(
                5,
                "RESTORE_GPU_SERVICES",
                "SUCCEEDED",
                started_at=_at(12, 10),
                updated_at=_at(12, 40),
            ),
        ],
    )
    errors = verdicts.workflow_errors(workflow, fenced_incident(), node=NODE)
    assert any(verdicts.GENERATION_FENCE_LITERAL in item for item in errors), errors


def _pointer_only_workflow() -> dict[str, Any]:
    """The fenced workflow as the store really records a remote step: the
    verify execution's error is a generic failure and its details point at the
    command; nothing in step_executions or the workflow error names the fence."""

    executions = fenced_workflow()["step_executions"]
    pointer = _execution(
        3,
        "VERIFY_NO_GPU_CLIENTS",
        "FAILED",
        started_at=_at(2, 5),
        updated_at=_at(6, 40),
        error="remote command failed",
        details={
            "remote_command_id": fence_command()["command_id"],
            "remote_status": "FAILED",
        },
    )
    return fenced_workflow(
        error="step 3 VERIFY_NO_GPU_CLIENTS failed: remote command failed",
        step_executions=[
            *executions[:3],
            pointer,
            _execution(
                5,
                "RESTORE_GPU_SERVICES",
                "FAILED",
                started_at=_at(6, 45),
                updated_at=_at(7),
                error="remote command failed",
            ),
        ],
    )


def test_the_generation_fence_on_the_remote_command_proves_the_case() -> None:
    """Remote steps keep only pointers in step.details; the refusal the case
    exists to prove is on the command row and must be read from there."""

    workflow = _pointer_only_workflow()
    without = verdicts.workflow_errors(workflow, fenced_incident(), node=NODE)
    assert any(verdicts.GENERATION_FENCE_LITERAL in item for item in without), without
    with_commands = verdicts.workflow_errors(
        workflow, fenced_incident(), node=NODE, commands=[fence_command()]
    )
    assert with_commands == [], with_commands
    other = verdicts.workflow_errors(
        workflow,
        fenced_incident(),
        node=NODE,
        commands=[fence_command(error=FENCE_ERROR.replace(NODE, OTHER))],
    )
    assert any("names another node" in item for item in other), other


def test_a_generation_fence_naming_another_node_is_refused() -> None:
    other = FENCE_ERROR.replace(NODE, OTHER)
    workflow = fenced_workflow(
        error=other,
        step_executions=[
            *fenced_workflow()["step_executions"][:3],
            _execution(
                3,
                "VERIFY_NO_GPU_CLIENTS",
                "FAILED",
                started_at=_at(2, 5),
                updated_at=_at(6, 40),
                error=other,
            ),
            _execution(
                5,
                "RESTORE_GPU_SERVICES",
                "FAILED",
                started_at=_at(6, 45),
                updated_at=_at(7, 10),
                error=other,
            ),
        ],
    )
    errors = verdicts.workflow_errors(workflow, fenced_incident(), node=NODE)
    assert any("names another node" in item for item in errors), errors


def test_a_superseded_or_blocked_workflow_is_not_the_documented_contract() -> None:
    """A retired-generation sweep taking the workflow over, or a BLOCKED record
    waiting for an operator, is a different outcome than failing closed."""

    swept = verdicts.workflow_errors(
        fenced_workflow(superseded_step_indexes=[3]), fenced_incident(), node=NODE
    )
    assert any("retired-generation sweep" in item for item in swept), swept
    blocked = verdicts.workflow_errors(
        fenced_workflow(status="BLOCKED", blocked_kind="awaiting_operator"),
        fenced_incident(),
        node=NODE,
    )
    assert any("BLOCKED rather than FAILED" in item for item in blocked), blocked


def test_an_unresolved_quiesce_must_be_compensated() -> None:
    workflow = fenced_workflow(step_executions=fenced_workflow()["step_executions"][:4])
    errors = verdicts.workflow_errors(workflow, fenced_incident(), node=NODE)
    assert any("compensation never ran" in item for item in errors), errors


def test_the_incident_must_end_quarantined() -> None:
    errors = verdicts.workflow_errors(
        fenced_workflow(), fenced_incident(state="RESOLVED"), node=NODE
    )
    assert any("not QUARANTINED" in item for item in errors), errors


def test_a_step_addressing_another_node_is_refused() -> None:
    steps = [_step(operation) for operation in verdicts.OFFICIAL_OPERATIONS]
    steps[4] = _step("RESET_GPU", nodes=[OTHER])
    errors = verdicts.workflow_errors(
        fenced_workflow(official_steps=steps), fenced_incident(), node=NODE
    )
    assert any("addresses another node" in item for item in errors), errors


# --------------------------------------------------------------------------- #
# Remote commands
# --------------------------------------------------------------------------- #
def test_no_remote_command_may_be_a_reset_or_a_succeeded_old_verify() -> None:
    assert (
        verdicts.command_errors(
            [
                {
                    "command_id": f"{REQUEST}/3/VERIFY_NO_GPU_CLIENTS/commit",
                    "status": "EXPIRED",
                    "step": {"operation": "VERIFY_NO_GPU_CLIENTS"},
                }
            ],
            node=NODE,
        )
        == []
    )
    errors = verdicts.command_errors(
        [
            {
                "command_id": f"{REQUEST}/4/RESET_GPU/commit",
                "status": "PENDING",
                "step": {"operation": "RESET_GPU"},
            },
            {
                "command_id": f"{REQUEST}/3/VERIFY_NO_GPU_CLIENTS/commit",
                "status": "SUCCEEDED",
                "step": {"operation": "VERIFY_NO_GPU_CLIENTS"},
            },
        ],
        node=NODE,
    )
    assert len(errors) == 2, errors
    assert any("GPU reset command was dispatched" in item for item in errors), errors
    assert any("recorded SUCCEEDED" in item for item in errors), errors


# --------------------------------------------------------------------------- #
# Escalation
# --------------------------------------------------------------------------- #
def _forbidden(**overrides: Any) -> dict[str, Any]:
    value = {"reboot": None, "replace": None, "drain": None}
    value.update(overrides)
    return value


def test_the_failed_compensation_opens_exactly_one_support_escalation() -> None:
    assert (
        verdicts.successor_errors(
            support_workflow(),
            support_incident(),
            node=NODE,
            predecessor_request_id=REQUEST,
            forbidden_escalations=_forbidden(),
            compensation_failed=True,
        )
        == []
    )


@pytest.mark.parametrize("rung", ["reboot", "replace", "drain"])
def test_an_out_of_band_reboot_may_never_escalate_to_hardware(rung: str) -> None:
    errors = verdicts.successor_errors(
        support_workflow(),
        support_incident(),
        node=NODE,
        predecessor_request_id=REQUEST,
        forbidden_escalations=_forbidden(**{rung: {"incident": {}}}),
        compensation_failed=True,
    )
    assert any("hardware escalation was opened" in item for item in errors), errors


def test_a_missing_support_escalation_leaves_nobody_accountable() -> None:
    errors = verdicts.successor_errors(
        {},
        {},
        node=NODE,
        predecessor_request_id=REQUEST,
        forbidden_escalations=_forbidden(),
        compensation_failed=True,
    )
    assert any("no support escalation" in item for item in errors), errors


def test_no_escalation_is_expected_when_no_containment_step_failed() -> None:
    assert (
        verdicts.successor_errors(
            {},
            {},
            node=NODE,
            predecessor_request_id=REQUEST,
            forbidden_escalations=_forbidden(),
            compensation_failed=False,
        )
        == []
    )
    errors = verdicts.successor_errors(
        support_workflow(),
        support_incident(),
        node=NODE,
        predecessor_request_id=REQUEST,
        forbidden_escalations=_forbidden(),
        compensation_failed=False,
    )
    assert any("no containment step failed" in item for item in errors), errors


def test_the_successor_must_be_this_workflows_support_escalation() -> None:
    errors = verdicts.successor_errors(
        support_workflow(request_id="workflow-support-after-someone-else"),
        support_incident(),
        node=NODE,
        predecessor_request_id=REQUEST,
        forbidden_escalations=_forbidden(),
        compensation_failed=True,
    )
    assert any("not this workflow's support escalation" in item for item in errors), (
        errors
    )


def test_the_support_escalation_must_be_the_operator_plan() -> None:
    errors = verdicts.successor_errors(
        support_workflow(
            official_action="RESTART_NODE", official_steps=[_step("RESTART_NODE")]
        ),
        support_incident(),
        node=NODE,
        predecessor_request_id=REQUEST,
        forbidden_escalations=_forbidden(),
        compensation_failed=True,
    )
    assert any("not the support plan" in item for item in errors), errors
    assert any("not ESCALATE_OPERATOR" in item for item in errors), errors


# --------------------------------------------------------------------------- #
# Agent generation and boot id
# --------------------------------------------------------------------------- #
def test_the_agent_must_advance_exactly_one_generation() -> None:
    assert verdicts.agent_errors(agent_before(), agent_after(), node=NODE) == []


def test_a_generation_that_did_not_move_means_no_reboot_happened() -> None:
    errors = verdicts.agent_errors(
        agent_before(), agent_after(generation=GENERATION), node=NODE
    )
    assert any("did not advance exactly once" in item for item in errors), errors


def test_two_generations_mean_the_node_rebooted_twice() -> None:
    errors = verdicts.agent_errors(
        agent_before(), agent_after(generation=GENERATION + 2), node=NODE
    )
    assert any("did not advance exactly once" in item for item in errors), errors


def test_the_pre_reboot_incarnation_must_be_the_retired_one() -> None:
    errors = verdicts.agent_errors(
        agent_before(), agent_after(retired_incarnation_ids=["inc-z"]), node=NODE
    )
    assert any("not the pre-reboot one" in item for item in errors), errors
    none_retired = verdicts.agent_errors(
        agent_before(), agent_after(retired_incarnation_ids=[]), node=NODE
    )
    assert any("exactly one incarnation" in item for item in none_retired), none_retired


def test_the_retired_incarnation_is_matched_on_the_models_field_name() -> None:
    """``AgentRecord`` retires ``agent_incarnation_id``; a verdict reading any
    other key would never compare the retired generation to anything."""

    assert "agent_incarnation_id" in agent_before()
    assert "incarnation_id" not in agent_before()
    unknown = verdicts.agent_errors(
        agent_before(agent_incarnation_id=None), agent_after(), node=NODE
    )
    assert any("no agent_incarnation_id" in item for item in unknown), unknown
    mismatched = verdicts.agent_errors(
        agent_before(agent_incarnation_id="inc-other"), agent_after(), node=NODE
    )
    assert any("not the pre-reboot one" in item for item in mismatched), mismatched


def test_an_agent_that_is_not_active_again_is_a_failure() -> None:
    errors = verdicts.agent_errors(
        agent_before(), agent_after(lifecycle_state="DEGRADED"), node=NODE
    )
    assert any("is not ACTIVE" in item for item in errors), errors


def test_agent_records_must_both_be_for_the_case_node() -> None:
    errors = verdicts.agent_errors(
        agent_before(), agent_after(node_id=OTHER), node=NODE
    )
    assert errors == [f"agent records are not both for {NODE}"]


def test_exactly_one_boot_change_is_witnessed_twice() -> None:
    assert (
        verdicts.boot_errors(
            node=NODE,
            baseline_boot_id=BOOT_BEFORE,
            final_boot_id=BOOT_AFTER,
            host_boot_ids=[BOOT_BEFORE, BOOT_BEFORE, BOOT_AFTER],
            reboot_status=reboot_status(),
        )
        == []
    )


def test_an_unchanged_boot_id_means_the_reboot_never_happened() -> None:
    errors = verdicts.boot_errors(
        node=NODE,
        baseline_boot_id=BOOT_BEFORE,
        final_boot_id=BOOT_BEFORE,
        host_boot_ids=[BOOT_BEFORE],
        reboot_status=reboot_status(fired=False, boot_id_now=BOOT_BEFORE),
    )
    assert any("boot id did not change" in item for item in errors), errors
    assert any("does not record a fired reboot" in item for item in errors), errors


def test_a_third_boot_observation_means_the_node_rebooted_twice() -> None:
    third = "33333333-3333-4333-8333-333333333333"
    errors = verdicts.boot_errors(
        node=NODE,
        baseline_boot_id=BOOT_BEFORE,
        final_boot_id=third,
        host_boot_ids=[BOOT_BEFORE, BOOT_AFTER, third],
        reboot_status=reboot_status(boot_id_now=third),
    )
    assert any("observed 3 boots" in item for item in errors), errors


def test_a_marker_from_another_boot_is_refused() -> None:
    errors = verdicts.boot_errors(
        node=NODE,
        baseline_boot_id=BOOT_BEFORE,
        final_boot_id=BOOT_AFTER,
        host_boot_ids=[BOOT_BEFORE, BOOT_AFTER],
        reboot_status=reboot_status(boot_id_before_reboot="99999999"),
    )
    assert any("another pre-reboot boot id" in item for item in errors), errors


# --------------------------------------------------------------------------- #
# Data plane
# --------------------------------------------------------------------------- #
def test_the_ledger_after_the_reboot_has_no_reset_and_no_new_success() -> None:
    assert verdicts.ledger_errors(host_before(), host_after(), node=NODE) == []


def test_the_agent_reconciled_the_quiesce_state_the_reboot_orphaned() -> None:
    assert verdicts.boot_reconcile_errors(host_after(), node=NODE) == []


def test_a_quiesce_state_file_surviving_the_reboot_fails_the_case() -> None:
    after = host_after()
    after["quiesce_states"] = [{"name": "quiesce-abc.json", "phase": "QUIESCED"}]
    errors = verdicts.boot_reconcile_errors(after, node=NODE)
    assert any("did not reconcile" in item for item in errors), errors


def test_a_reset_row_in_the_ledger_fails_the_case() -> None:
    after = host_after()
    after["ledger"] = [
        *after["ledger"],
        _row(
            f"{REQUEST}/4/RESET_GPU/commit",
            "RESET_GPU",
            "SUCCEEDED",
            completed_at=_at(8),
        ),
    ]
    errors = verdicts.ledger_errors(host_before(), after, node=NODE)
    assert any("recorded a GPU reset" in item for item in errors), errors


def test_a_succeeded_verify_row_means_the_holder_did_not_hold() -> None:
    after = host_after()
    after["ledger"][-1] = _row(
        f"{REQUEST}/3/VERIFY_NO_GPU_CLIENTS/commit",
        "VERIFY_NO_GPU_CLIENTS",
        "SUCCEEDED",
        completed_at=_at(6, 40),
    )
    errors = verdicts.ledger_errors(host_before(), after, node=NODE)
    assert any("did not hold" in item for item in errors), errors


def test_a_new_success_the_fence_should_have_refused_fails_the_case() -> None:
    after = host_after()
    after["ledger"] = [
        *after["ledger"],
        _row(
            f"{REQUEST}/5/RESTORE_GPU_SERVICES/commit",
            "RESTORE_GPU_SERVICES",
            "SUCCEEDED",
            completed_at=_at(9),
        ),
    ]
    errors = verdicts.ledger_errors(host_before(), after, node=NODE)
    assert any("fence should have refused" in item for item in errors), errors


def test_a_ledger_without_the_pinning_quiesce_row_fails_the_case() -> None:
    after = host_after()
    after["ledger"] = [after["ledger"][0], after["ledger"][2]]
    errors = verdicts.ledger_errors(host_before(), after, node=NODE)
    assert any("no successful QUIESCE_GPU_SERVICES" in item for item in errors), errors


def test_the_baseline_rows_of_earlier_drills_are_ignored() -> None:
    """The ledger replays by command id for seven days, so the rows an earlier
    case left behind must not be read as this case's evidence."""

    before = host_before()
    before["ledger"] = [
        *before["ledger"],
        _row("old/4/RESET_GPU/commit", "RESET_GPU", "SUCCEEDED", completed_at=_at(0)),
    ]
    after = host_after()
    after["ledger"] = [*before["ledger"], *after["ledger"][1:]]
    assert verdicts.ledger_errors(before, after, node=NODE) == []


def test_the_kernel_journal_is_the_last_word_on_the_reset() -> None:
    assert verdicts.reset_journal_errors(host_after(), node=NODE) == []
    errors = verdicts.reset_journal_errors(
        host_after(kernel_reset_journal={"target_reset_count": 1}), node=NODE
    )
    assert any("kernel journal shows 1 reset" in item for item in errors), errors
    missing = verdicts.reset_journal_errors(
        host_after(kernel_reset_journal={}), node=NODE
    )
    assert missing == [f"{NODE} kernel reset journal was not captured"]


def test_the_node_after_cleanup_carries_no_residue() -> None:
    assert (
        verdicts.host_final_errors(
            host_before(), host_after(), node=NODE, expected_gpu_count=8
        )
        == []
    )


def test_an_orphaned_quiesce_state_file_or_timer_fails_cleanup() -> None:
    errors = verdicts.host_final_errors(
        host_before(),
        host_after(
            quiesce_states=[{"name": "quiesce-abc.json"}],
            gpu_fault_timers=["gpu-fault-destr017-reboot-x.timer"],
        ),
        node=NODE,
        expected_gpu_count=8,
    )
    assert any("quiesce state file remains" in item for item in errors), errors
    assert any("timer inventory did not return" in item for item in errors), errors


def test_a_service_that_did_not_come_back_fails_cleanup() -> None:
    errors = verdicts.host_final_errors(
        host_before(),
        host_after(
            services={
                "nvidia-fabricmanager.service": {"ActiveState": "inactive"},
                "kubelet.service": {"ActiveState": "active"},
            }
        ),
        node=NODE,
        expected_gpu_count=8,
    )
    assert errors == [
        f"{NODE} service did not return active: nvidia-fabricmanager.service"
    ]


def test_a_missing_gpu_after_the_reboot_fails_cleanup() -> None:
    after = host_after()
    after["gpu_inventory"] = after["gpu_inventory"][:7]
    errors = verdicts.host_final_errors(
        host_before(), after, node=NODE, expected_gpu_count=8
    )
    assert any("GPU inventory is not 8" in item for item in errors), errors


def test_the_holder_must_have_matched_this_drills_quiesce_row() -> None:
    assert (
        verdicts.holder_errors(
            {
                "matched_row": {
                    "command_id": f"{REQUEST}/2/QUIESCE_GPU_SERVICES/commit"
                },
                "hold_started_at": WINDOW_START,
                "holder_error": None,
            },
            node=NODE,
        )
        == []
    )
    errors = verdicts.holder_errors(
        {"matched_row": None, "hold_started_at": None, "holder_error": "no row"},
        node=NODE,
    )
    assert len(errors) == 3, errors


def test_the_node_ends_ready_and_unfenced() -> None:
    assert (
        verdicts.schedulability_errors(
            {
                "ready": "True",
                "unschedulable": False,
                "taints": [],
                "ownership_annotations": {},
            },
            node=NODE,
        )
        == []
    )
    errors = verdicts.schedulability_errors(
        {
            "ready": "False",
            "unschedulable": True,
            "taints": [{"key": verdicts.QUARANTINE_TAINT}],
            "ownership_annotations": {"gpu-fault.io/incident-id": INCIDENT},
        },
        node=NODE,
    )
    assert len(errors) == 4, errors
    assert any(verdicts.QUARANTINE_TAINT in item for item in errors), errors


def test_an_os_reboot_is_not_a_provider_mutation() -> None:
    assert verdicts.cloudtrail_errors([]) == []
    errors = verdicts.cloudtrail_errors(
        [{"event_name": "BatchRebootClusterNodes"}, {"event_name": "UpdateCluster"}]
    )
    assert len(errors) == 1, errors
    assert "BatchRebootClusterNodes" in errors[0]
    assert "UpdateCluster" in errors[0]


# --------------------------------------------------------------------------- #
# Retired-generation reconcile
# --------------------------------------------------------------------------- #
def test_the_reconcile_is_recorded_as_a_plan_that_was_never_applied() -> None:
    plan = {
        "mode": verdicts.RETIRED_GENERATION_PLAN_MODE,
        "applied": False,
        "items": [{"request_id": "workflow-somebody-else"}],
        "plan_sha256": "0" * 64,
    }
    assert verdicts.reconcile_plan_errors(plan, request_ids={REQUEST}) == []
    applied = verdicts.reconcile_plan_errors(
        {**plan, "applied": True}, request_ids={REQUEST}
    )
    assert any("must stay a plan" in item for item in applied), applied
    wrong_mode = verdicts.reconcile_plan_errors(
        {**plan, "mode": "apply"}, request_ids={REQUEST}
    )
    assert any("not run in plan mode" in item for item in wrong_mode), wrong_mode


def test_a_plan_that_wants_to_revoke_this_case_means_the_fence_wedged_it() -> None:
    plan = {
        "mode": verdicts.RETIRED_GENERATION_PLAN_MODE,
        "applied": False,
        "items": [{"request_id": REQUEST}, {"request_id": SUPPORT_REQUEST}],
        "plan_sha256": "0" * 64,
    }
    errors = verdicts.reconcile_plan_errors(
        plan, request_ids={REQUEST, SUPPORT_REQUEST}
    )
    assert any("wedged record" in item for item in errors), errors
    assert REQUEST in errors[0] and SUPPORT_REQUEST in errors[0]


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
def _preflight(**overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "node": NODE,
        "node_snapshot": {
            "ready": "True",
            "unschedulable": False,
            "taints": [],
            "ownership_annotations": {},
        },
        "agent": agent_before(),
        "profile": owned_profile(),
        "business_workloads": [],
        "queue": {"depth": 0},
        # ``Store.remote_command_stats()`` reports ``open_by_cluster``.
        "remote_commands": {"open_by_cluster": {}},
        "recent_events": [],
        "host_snapshot": host_before(),
        "reboot_status": {"armed": False},
        "expected_gpu_count": 8,
    }
    arguments.update(overrides)
    return arguments


def test_an_idle_healthy_node_passes_the_preflight() -> None:
    assert verdicts.preflight_errors(**_preflight()) == []


def test_the_preflight_refuses_a_node_that_is_not_idle() -> None:
    errors = verdicts.preflight_errors(
        **_preflight(
            node_snapshot={
                "ready": "False",
                "unschedulable": True,
                "taints": [{"key": verdicts.QUARANTINE_TAINT}],
                "ownership_annotations": {"gpu-fault.io/incident-id": INCIDENT},
            },
            business_workloads=[{"name": "training/pytorchjob/job"}],
        )
    )
    assert len(errors) == 5, errors


def test_the_preflight_needs_a_generation_to_fence_on() -> None:
    errors = verdicts.preflight_errors(
        **_preflight(agent=agent_before(generation=None))
    )
    assert any("no generation to fence on" in item for item in errors), errors


def test_the_preflight_needs_every_maintenance_operation_advertised() -> None:
    errors = verdicts.preflight_errors(
        **_preflight(
            agent=agent_before(
                allowed_operations=["QUIESCE_GPU_SERVICES", "VERIFY_NO_GPU_CLIENTS"]
            )
        )
    )
    assert any("does not allow" in item for item in errors), errors
    assert any("RESET_GPU" in item for item in errors), errors


def test_the_preflight_reads_own_from_the_profile_not_the_agent_record() -> None:
    """OWN is the runtime profile's gpuReset mode; ``AgentRecord`` has no
    capability mode, so a verdict reading one off the agent would refuse every
    real node."""

    assert verdicts.preflight_errors(**_preflight()) == []
    missing = verdicts.preflight_errors(
        **_preflight(profile=owned_profile(capabilities=[]))
    )
    assert any("no gpuReset capability" in item for item in missing), missing
    delegated = verdicts.preflight_errors(
        **_preflight(
            profile=owned_profile(
                capabilities=[
                    {
                        "capability": "gpuReset",
                        "mode": "DELEGATE",
                        "owner": "gpu-fault-node-agent",
                    }
                ]
            )
        )
    )
    assert any("not OWN by the Node Agent" in item for item in delegated), delegated


def test_the_preflight_reads_open_commands_the_way_the_store_reports_them() -> None:
    busy = verdicts.preflight_errors(
        **_preflight(remote_commands={"open_by_cluster": {"cluster-a": 2}})
    )
    assert any("remote commands are not idle" in item for item in busy), busy
    # The shape the store never emits must not read as idle-by-accident either:
    # an empty ``open_by_cluster`` is the only idle reading.
    assert verdicts.preflight_errors(**_preflight(remote_commands={})) == []


def test_the_preflight_refuses_a_node_that_cannot_take_a_real_xid() -> None:
    """A node whose /dev/kmsg is not writable would make the injection
    synthetic, which is never allowed to be recorded as a hardware fault."""

    errors = verdicts.preflight_errors(
        **_preflight(host_snapshot=host_before(kmsg_writable=False))
    )
    assert any("would be fake" in item for item in errors), errors


def test_the_preflight_refuses_a_node_that_is_already_quiesced_or_busy() -> None:
    errors = verdicts.preflight_errors(
        **_preflight(
            host_snapshot=host_before(
                quiesce_states=[{"name": "quiesce-abc.json"}],
                compute_clients=[{"pid": 1234, "device": "/dev/nvidia0"}],
            )
        )
    )
    assert any("quiesce state file" in item for item in errors), errors
    assert any("already has GPU compute clients" in item for item in errors), errors


def test_the_preflight_refuses_an_armed_reboot_timer_from_an_earlier_run() -> None:
    errors = verdicts.preflight_errors(
        **_preflight(
            reboot_status={
                "armed": True,
                "reboot_cancelled_at": None,
                "reboot_unit": "gpu-fault-destr017-reboot-abc.timer",
            }
        )
    )
    assert any("already has an armed reboot timer" in item for item in errors), errors
    cancelled = verdicts.preflight_errors(
        **_preflight(
            reboot_status={
                "armed": True,
                "reboot_cancelled_at": _at(0),
                "reboot_unit": "u",
            }
        )
    )
    assert cancelled == []


def test_a_reboot_timer_that_already_fired_is_not_armed() -> None:
    """After the reboot the marker still says armed and uncancelled -- the
    timer is spent, the boot id moved. A later --plan in the same run_dir must
    not be refused for it."""

    fired = verdicts.preflight_errors(
        **_preflight(
            reboot_status={
                "armed": True,
                "reboot_cancelled_at": None,
                "fired": True,
                "boot_id_before_reboot": BOOT_BEFORE,
                "boot_id_now": BOOT_AFTER,
                "reboot_unit": "gpu-fault-destr017-reboot-abc.timer",
            }
        )
    )
    assert fired == []


def test_the_preflight_refuses_a_busy_control_plane_or_a_recent_event() -> None:
    errors = verdicts.preflight_errors(
        **_preflight(
            queue={"depth": 3},
            remote_commands={"open_by_cluster": {"cluster-a": 1}},
            recent_events=[{"event_id": "evt-1"}],
        )
    )
    assert len(errors) == 3, errors


# --------------------------------------------------------------------------- #
# Timeline and arithmetic
# --------------------------------------------------------------------------- #
def test_step_transitions_only_report_changes() -> None:
    executions = fenced_workflow()["step_executions"]
    state, changes = verdicts.step_transitions({}, executions)
    assert len(changes) == len(executions)
    assert state["3/VERIFY_NO_GPU_CLIENTS#0"] == "FAILED"
    assert changes[3]["fence_variant"] == verdicts.FENCE_AGENT_GENERATION
    state, again = verdicts.step_transitions(state, executions)
    assert again == []


def test_step_transitions_report_a_waiting_step_that_later_fails() -> None:
    waiting = _execution(
        3, "VERIFY_NO_GPU_CLIENTS", "WAITING", started_at=_at(2, 5), updated_at=_at(3)
    )
    state, _ = verdicts.step_transitions({}, [waiting])
    assert state["3/VERIFY_NO_GPU_CLIENTS#0"] == "WAITING"
    state, changes = verdicts.step_transitions(
        state,
        [
            _execution(
                3,
                "VERIFY_NO_GPU_CLIENTS",
                "FAILED",
                started_at=_at(2, 5),
                updated_at=_at(6, 40),
                error=FENCE_ERROR,
            )
        ],
    )
    assert [item["status"] for item in changes] == ["FAILED"]
    assert changes[0]["fence_variant"] == verdicts.FENCE_AGENT_GENERATION


def test_step_transitions_do_not_flip_between_a_kept_waiting_row_and_its_end() -> None:
    """The store keeps the WAITING row when the fence adds the FAILED one. A
    key of index/operation alone would see the pair flip on every poll and the
    timeline would grow for ever."""

    waiting = _execution(
        3, "VERIFY_NO_GPU_CLIENTS", "WAITING", started_at=_at(2, 5), updated_at=_at(3)
    )
    failed = _execution(
        3,
        "VERIFY_NO_GPU_CLIENTS",
        "FAILED",
        started_at=_at(2, 5),
        updated_at=_at(6, 40),
        error=FENCE_ERROR,
    )
    state, first = verdicts.step_transitions({}, [waiting, failed])
    assert [(item["occurrence"], item["status"]) for item in first] == [
        (0, "WAITING"),
        (1, "FAILED"),
    ]
    state, second = verdicts.step_transitions(state, [waiting, failed])
    assert second == []
    state, third = verdicts.step_transitions(state, [waiting, failed])
    assert third == []


def test_the_case_has_to_fit_one_node_workflow_lifetime() -> None:
    """The support successor shares the fenced workflow's lifetime deadline, so
    the whole chain is bounded by the predecessor's lifetime."""

    estimated = verdicts.estimated_duration_seconds()
    assert estimated == 1560
    assert (
        verdicts.lifetime_errors(estimated_seconds=estimated, lifetime_seconds=3600)
        == []
    )
    tight = verdicts.lifetime_errors(estimated_seconds=estimated, lifetime_seconds=900)
    assert any("does not fit" in item for item in tight), tight
    unknown = verdicts.lifetime_errors(
        estimated_seconds=estimated, lifetime_seconds=None
    )
    assert any("lifetime is unknown" in item for item in unknown), unknown


def test_the_reboot_must_land_inside_the_pinned_window_and_the_waiting_cap() -> None:
    assert (
        verdicts.reboot_window_errors(
            delay_seconds=30,
            window_remaining_seconds=300.0,
            step_waiting_limit_seconds=600,
        )
        == []
    )
    late = verdicts.reboot_window_errors(
        delay_seconds=300,
        window_remaining_seconds=120.0,
        step_waiting_limit_seconds=600,
    )
    assert any("after the maintenance window" in item for item in late), late
    capped = verdicts.reboot_window_errors(
        delay_seconds=600,
        window_remaining_seconds=900.0,
        step_waiting_limit_seconds=600,
    )
    assert any("per-step waiting cap" in item for item in capped), capped
    unknown = verdicts.reboot_window_errors(
        delay_seconds=30, window_remaining_seconds=None, step_waiting_limit_seconds=600
    )
    assert any("window is unknown" in item for item in unknown), unknown


def test_the_waiting_cap_is_measured_from_when_the_step_parked() -> None:
    """A step that has already waited 550s of a 600s cap has 50s left; a 60s
    reboot placed against the whole cap would land after the cap fired."""

    late = verdicts.reboot_window_errors(
        delay_seconds=60,
        window_remaining_seconds=900.0,
        step_waiting_limit_seconds=600,
        step_waiting_elapsed_seconds=550.0,
    )
    assert any("per-step waiting cap" in item for item in late), late
    assert any("50s remain" in item for item in late), late
    early = verdicts.reboot_window_errors(
        delay_seconds=60,
        window_remaining_seconds=900.0,
        step_waiting_limit_seconds=600,
        step_waiting_elapsed_seconds=120.0,
    )
    assert early == []


def test_waiting_elapsed_is_read_off_the_waiting_execution() -> None:
    now = datetime.fromisoformat(_at(5))
    workflow = fenced_workflow(
        step_executions=[
            _execution(
                3,
                "VERIFY_NO_GPU_CLIENTS",
                "WAITING",
                started_at=_at(2),
                updated_at=_at(4),
            )
        ]
    )
    assert (
        destr017.waiting_elapsed_seconds(
            workflow, operation="VERIFY_NO_GPU_CLIENTS", now=now
        )
        == 180.0
    )
    assert (
        destr017.waiting_elapsed_seconds(
            fenced_workflow(), operation="VERIFY_NO_GPU_CLIENTS", now=now
        )
        == 0.0
    )


# --------------------------------------------------------------------------- #
# Runner helpers
# --------------------------------------------------------------------------- #
def test_the_maintenance_pin_is_read_from_the_succeeded_quiesce_step() -> None:
    pin = destr017.maintenance_pin(fenced_workflow(), node=NODE)
    assert pin == {
        "pinned_generation": GENERATION,
        "window_started_at": WINDOW_START,
        "window_expires_at": WINDOW_END,
    }


def test_no_pin_is_read_from_a_failed_quiesce_or_another_node() -> None:
    failed = quiesce_execution()
    failed["status"] = "FAILED"
    assert (
        destr017.maintenance_pin(fenced_workflow(step_executions=[failed]), node=NODE)
        == {}
    )
    assert destr017.maintenance_pin(fenced_workflow(), node=OTHER) == {}
    assert destr017.maintenance_pin({}, node=NODE) == {}


def test_the_window_remaining_is_measured_against_the_pinned_expiry() -> None:
    pin = destr017.maintenance_pin(fenced_workflow(), node=NODE)
    now = datetime(2026, 9, 6, 10, 5, tzinfo=timezone.utc)
    assert destr017.window_remaining_seconds(pin, now=now) == 240.0
    assert destr017.window_remaining_seconds({}, now=now) is None


def test_the_expected_fence_text_names_the_pinned_generation_and_the_node() -> None:
    """Recorded as evidence, not asserted against: which refusal fires depends
    on how the reboot raced the window."""

    text = destr017.expected_fence_text(
        destr017.maintenance_pin(fenced_workflow(), node=NODE), node=NODE
    )
    assert text == FENCE_ERROR
    assert verdicts.fence_variant(text) == verdicts.FENCE_AGENT_GENERATION
    assert destr017.expected_fence_text({}, node=NODE) == ""


def test_plan_identity_digest_is_stable_across_key_order() -> None:
    preflight = {
        "release_id": "rel",
        "node": {"uid": "u1", "boot_id": BOOT_BEFORE},
        "store": {"agent": agent_before(), "profile": {"profile_version": "p1"}},
        "runtime_identity": {"x": 1},
    }
    identity = destr017.plan_identity(preflight, node=NODE)
    shuffled = dict(reversed(list(identity.items())))
    assert destr017.identity_digest(identity) == destr017.identity_digest(shuffled)
    assert identity["agent_generation"] == GENERATION
    # Read from ``AgentRecord.agent_incarnation_id``, the field that exists.
    assert identity["agent_incarnation_id"] == "inc-a"
    assert identity["node_boot_id"] == BOOT_BEFORE
    assert identity["node"] == NODE


def test_the_plan_identity_pins_the_generation_the_fence_will_compare() -> None:
    """A run whose plan was built against another Agent generation is a
    different case: the node rebooted between plan and execute."""

    base = {
        "release_id": "rel",
        "node": {"uid": "u1", "boot_id": BOOT_BEFORE},
        "store": {"agent": agent_before(), "profile": {"profile_version": "p1"}},
        "runtime_identity": {"x": 1},
    }
    drifted = json.loads(json.dumps(base))
    drifted["store"]["agent"]["generation"] = GENERATION + 1
    first = destr017.identity_digest(destr017.plan_identity(base, node=NODE))
    second = destr017.identity_digest(destr017.plan_identity(drifted, node=NODE))
    assert first != second
    reincarnated = json.loads(json.dumps(base))
    reincarnated["store"]["agent"]["agent_incarnation_id"] = "inc-b"
    third = destr017.identity_digest(destr017.plan_identity(reincarnated, node=NODE))
    assert first != third


def test_derived_identity_is_deterministic_per_run_and_attempt(tmp_path: Path) -> None:
    first = destr017.derived_identity(tmp_path, 1)
    assert first == destr017.derived_identity(tmp_path, 1)
    assert first != destr017.derived_identity(tmp_path, 2)
    assert first.startswith("destr017-"), first
    assert first.endswith("-a1"), first


def test_configure_carries_the_attempt_the_preflight_derives_its_probes_from(
    tmp_path: Path,
) -> None:
    """The preflight used to derive the probe identity from attempt 1 whatever
    ``--attempt`` said, so a second attempt read the first one's on-node state."""

    cpu = tmp_path / "cpu.kubeconfig"
    gpu = tmp_path / "gpu.kubeconfig"
    cpu.write_text("apiVersion: v1\n", encoding="utf-8")
    gpu.write_text("apiVersion: v1\n", encoding="utf-8")
    arguments = destr017.parser().parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--attempt",
            "3",
            "--node",
            NODE,
            "--host-probe-image",
            "img",
            "--cpu-kubeconfig",
            str(cpu),
            "--gpu-kubeconfig",
            str(gpu),
            "--gpu-context",
            "gpu",
            "--cluster-id",
            "cluster-a",
            "--region",
            "us-west-2",
        ]
    )
    settings = destr017.configure(arguments)
    assert settings.attempt == 3
    assert destr017.derived_identity(tmp_path, settings.attempt).endswith("-a3"), (
        "the derived identity carries the attempt suffix"
    )


class _FlakyProbe:
    """A probe whose Pod died: the first exec fails, a re-created one answers."""

    def __init__(self, *, failures: int) -> None:
        self.failures = failures
        self.created = 0
        self.calls: list[tuple[str, ...]] = []

    def create(self) -> None:
        self.created += 1

    def execute(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
        self.calls.append(arguments)
        if self.failures:
            self.failures -= 1
            raise RuntimeError("pod not found")
        return {"ok": True, "arguments": list(arguments)}


def test_cleanup_recreates_a_dead_probe_once_and_only_when_the_exec_fails() -> None:
    healthy = _FlakyProbe(failures=0)
    assert destr017.execute_or_recreate(healthy, "disarm-holder")["ok"] is True
    assert healthy.created == 0
    dead = _FlakyProbe(failures=1)
    assert (
        destr017.execute_or_recreate(dead, "cancel-reboot", "--run-id", "r")["ok"]
        is True
    )
    assert dead.created == 1
    assert dead.calls == [("cancel-reboot", "--run-id", "r")] * 2
    gone = _FlakyProbe(failures=2)
    with pytest.raises(RuntimeError):
        destr017.execute_or_recreate(gone, "holder-status")
    assert gone.created == 1


def test_evidence_components_carry_digests_only() -> None:
    details = {
        "preflight_identity": {"release_id": "rel"},
        "workflow": fenced_workflow(),
        "incident": fenced_incident(),
        "hosts": {"host_after": host_after()},
    }
    components = destr017.evidence_components(details)
    assert set(components) == {"preflight_identity", "workflow", "incident", "hosts"}
    for value in components.values():
        assert len(value) == 64, value
        assert int(value, 16) >= 0
    expected = hashlib.sha256(
        json.dumps({"release_id": "rel"}, sort_keys=True, default=str).encode()
    ).hexdigest()
    assert components["preflight_identity"] == expected
    assert len(destr017.case_digest(components)) == 64


def test_parser_accepts_the_documented_arguments() -> None:
    parser = destr017.parser()
    arguments = parser.parse_args(["--run-dir", "/tmp/run"])
    assert arguments.execute is False and arguments.plan is False
    assert arguments.reboot_delay_seconds == destr017.DEFAULT_REBOOT_DELAY_SECONDS
    assert arguments.max_hold_seconds == destr017.DEFAULT_MAX_HOLD_SECONDS
    arguments = parser.parse_args(
        [
            "--run-dir",
            "/tmp/run",
            "--execute",
            "--confirm",
            "DESTR017_EXECUTE",
            "--maintenance-window-end",
            "2026-09-06T12:00:00+00:00",
            "--node",
            NODE,
            "--pci-bdf",
            "0000:0a:00.0",
            "--host-probe-image",
            "img",
            "--reboot-delay-seconds",
            "45",
            "--max-hold-seconds",
            "900",
        ]
    )
    assert arguments.node == NODE
    assert arguments.pci_bdf == "0000:0a:00.0"
    assert arguments.reboot_delay_seconds == 45
    help_text = parser.format_help()
    for option in ("--plan", "--execute", "--confirm", "--maintenance-window-end"):
        assert option in help_text, option


def test_configure_bounds_the_reboot_delay_and_the_device_hold() -> None:
    parser = destr017.parser()
    for flag, value in (
        ("--reboot-delay-seconds", "10"),
        ("--reboot-delay-seconds", "900"),
        ("--max-hold-seconds", "30"),
        ("--max-hold-seconds", "7200"),
    ):
        arguments = parser.parse_args(
            [
                "--run-dir",
                "/tmp/run",
                "--node",
                NODE,
                "--host-probe-image",
                "img",
                flag,
                value,
            ]
        )
        with pytest.raises(RegionalFixtureError):
            destr017.configure(arguments)


def test_execute_reuses_the_plans_focused_tests_only_for_the_same_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--plan records the focused pytest with a source digest; --execute reuses
    it instead of paying for the same run twice, and only for this exact tree."""

    from scripts.e2e.regional import live_driver_guard

    def refuse(*arguments: Any, **keywords: Any) -> Any:
        raise AssertionError("pytest must not run when the plan's result is reusable")

    monkeypatch.setattr(RegionalLiveFixture, "run", staticmethod(refuse))
    recorded = {"passed": True, "returncode": 0, "command": ["pytest"]}
    details: dict[str, Any] = {}
    live_driver_guard.record_focused_tests(details, recorded)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"details": details}), encoding="utf-8")
    reused = destr017.focused_tests(tmp_path, reuse=True)
    assert reused == {**recorded, "focused_tests_reused": True}
    # A --plan never reuses, and a result taken against another tree is rerun.
    with pytest.raises(AssertionError, match="must not run"):
        destr017.focused_tests(tmp_path, reuse=False)
    details["focused_tests_source_digest"] = "0" * 64
    plan_path.write_text(json.dumps({"details": details}), encoding="utf-8")
    with pytest.raises(AssertionError, match="must not run"):
        destr017.focused_tests(tmp_path, reuse=True)
