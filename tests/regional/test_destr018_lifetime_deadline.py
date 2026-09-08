"""GF-REGIONAL-DESTR-018 verdicts: the workflow lifetime as a hard deadline.

Every fixture here is a synthetic control-plane, Node Agent ledger or kernel
journal snapshot. Nothing in this file touches a cluster, and the case metadata
is built inline so the contract holds before the catalog entry lands.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import control_plane_env_window as env_window
from scripts.e2e.regional import destr018_verdicts as verdicts
from scripts.e2e.regional import run_destr018_lifetime_deadline as destr018
from scripts.e2e.regional.regional_case_contract import RegionalCaseMetadata
from scripts.e2e.regional.regional_live_fixture import (
    RUNTIME_IDENTITY_DEPLOYMENTS,
    RegionalLiveSettings,
)

ROOT = Path(__file__).resolve().parents[2]
NODE = "node-a"
T0 = datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc)
T_CANCEL = T0 + timedelta(seconds=200)
WORKFLOW_ID = "workflow-active"
SUPPORT_WORKFLOW_ID = f"workflow-support-after-{WORKFLOW_ID}"
SUPPORT_INCIDENT_ID = f"inc-support-after-{WORKFLOW_ID}"
BASELINE_COMMAND = "workflow-old/4/VERIFY_NO_GPU_CLIENTS/commit"
BASELINE_IDS = {BASELINE_COMMAND}


def _text(errors: list[str]) -> str:
    return "\n".join(errors)


# --------------------------------------------------------------------------- #
# Case contract (holds without a catalog entry)
# --------------------------------------------------------------------------- #
def test_the_confirmation_token_names_this_case_and_its_predecessor() -> None:
    metadata = RegionalCaseMetadata(
        case_id=destr018.CASE_ID,
        title="",
        category="regional-destructive-acceptance",
        level="staging",
        risk="live-service-action",
        automation="manual",
        procedure="docs/x.md#gf-regional-destr-018",
        predecessor=destr018.PREDECESSOR_CASE_ID,
    )
    prefix = metadata.confirmation.removesuffix("EXECUTE")
    assert prefix == "DESTR018_"
    assert destr018.CONFIRMATION == "DESTR018_LIFETIME_DEADLINE"
    assert destr018.CONFIRMATION.startswith(prefix) is True
    assert destr018.PREDECESSOR_CASE_ID == "GF-REGIONAL-DESTR-010"
    assert verdicts.CASE_ID == "GF-REGIONAL-DESTR-018"
    assert metadata.risk == "live-service-action"


def test_the_case_declares_the_six_variables_the_env_window_may_manage() -> None:
    assert env_window.ALLOWED_VARIABLES == (
        "GPU_FAULT_NODE_WORKFLOW_MAX_LIFETIME_SECONDS",
        "GPU_FAULT_WORKFLOW_EXECUTION_TIMEOUT_SECONDS",
        "GPU_FAULT_WORKFLOW_STEP_TIMEOUT_SECONDS",
        "GPU_FAULT_HYPERPOD_MANAGED_RECOVERY_TIMEOUT_SECONDS",
        "GPU_FAULT_WORKFLOW_STEP_WARNING_SECONDS",
        "GPU_FAULT_WORKFLOW_LEASE_DURATION_SECONDS",
    )
    assert env_window.DEPLOYMENT == "gpu-fault-control-worker"


def test_the_runner_opens_the_window_with_a_consistent_full_set(
    tmp_path: Path,
) -> None:
    """The runner emits every allow-listed variable in one internally
    consistent set, and the helper accepts it. Compressing the lifetime alone
    is what CrashLoopBackOff'd the live control plane; the full set is bootable.
    """

    assignments = _settings(tmp_path).assignments()

    assert set(assignments) == set(env_window.ALLOWED_VARIABLES)
    assert env_window.assignment_errors(assignments) == []


# --------------------------------------------------------------------------- #
# Timing arithmetic
# --------------------------------------------------------------------------- #
def test_the_shipped_window_leaves_the_attempt_margin_it_promises() -> None:
    attempts = verdicts.worst_case_verify_attempts(
        lifetime_seconds=verdicts.LIFETIME_SECONDS,
        cadence_seconds=verdicts.CADENCE_FLOOR_SECONDS,
    )
    assert attempts == 18
    assert attempts + verdicts.ATTEMPT_MARGIN <= verdicts.VERIFY_MAX_ATTEMPTS
    # The window compresses the per-step waiting cap down to the lifetime: the
    # control plane refuses a step cap *above* the lifetime, and a cap *below*
    # it would let the step fire first, so equal is the only value that works.
    assert verdicts.STEP_WAITING_CAP_SECONDS == verdicts.LIFETIME_SECONDS
    assert (
        verdicts.lifetime_margin_errors(
            lifetime_seconds=verdicts.LIFETIME_SECONDS,
            execution_timeout_seconds=verdicts.EXECUTION_TIMEOUT_SECONDS,
            cadence_seconds=float(verdicts.CADENCE_FLOOR_SECONDS),
        )
        == []
    )


def test_a_shorter_containment_means_more_attempts_not_fewer() -> None:
    pessimistic = verdicts.worst_case_verify_attempts(
        lifetime_seconds=300, cadence_seconds=5.0, containment_allowance_seconds=10
    )
    optimistic = verdicts.worst_case_verify_attempts(
        lifetime_seconds=300, cadence_seconds=5.0, containment_allowance_seconds=200
    )
    assert pessimistic == 58
    assert optimistic == 20
    assert pessimistic > optimistic


def test_an_unknown_cadence_refuses_the_drill_rather_than_guessing() -> None:
    errors = verdicts.lifetime_margin_errors(
        lifetime_seconds=180, execution_timeout_seconds=180, cadence_seconds=None
    )
    assert "cadence is unknown" in _text(errors)
    assert len(errors) == 1


def test_a_cadence_below_the_deployed_floor_refuses_the_drill() -> None:
    errors = verdicts.lifetime_margin_errors(
        lifetime_seconds=180, execution_timeout_seconds=180, cadence_seconds=1.0
    )
    assert "below the deployed floor" in _text(errors)


def test_a_lifetime_long_enough_to_burn_the_attempt_budget_is_refused() -> None:
    errors = verdicts.lifetime_margin_errors(
        lifetime_seconds=420, execution_timeout_seconds=420, cadence_seconds=5.0
    )
    assert "GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS" in _text(errors)
    assert "attempt budget" in _text(errors)


def test_a_step_cap_equal_to_the_lifetime_carries_no_step_cap_error() -> None:
    """The window sets the step cap to the lifetime, so equal is the drill's own
    config and must not be flagged; only a cap *below* the lifetime is refused."""

    assert (
        verdicts.lifetime_margin_errors(
            lifetime_seconds=180,
            execution_timeout_seconds=180,
            cadence_seconds=5.0,
            step_waiting_cap_seconds=180,
        )
        == []
    )


def test_a_step_cap_under_the_lifetime_lets_the_step_fire_first_and_is_refused() -> None:
    errors = verdicts.lifetime_margin_errors(
        lifetime_seconds=180,
        execution_timeout_seconds=180,
        cadence_seconds=5.0,
        step_waiting_cap_seconds=120,
    )
    assert "per-step waiting cap 120s" in _text(errors)


def test_an_execution_timeout_below_the_lifetime_is_refused() -> None:
    errors = verdicts.lifetime_margin_errors(
        lifetime_seconds=180, execution_timeout_seconds=120, cadence_seconds=5.0
    )
    assert "would fire first" in _text(errors)
    assert "workflow_lifetime_exceeded" in _text(errors)


def test_a_window_outside_the_helpers_bounds_is_refused() -> None:
    errors = verdicts.lifetime_margin_errors(
        lifetime_seconds=30, execution_timeout_seconds=30, cadence_seconds=5.0
    )
    assert "outside the env window's bounds" in _text(errors)


def test_timing_evidence_records_the_arithmetic_a_reviewer_must_redo() -> None:
    evidence = verdicts.timing_evidence(
        lifetime_seconds=180, execution_timeout_seconds=180, cadence_seconds=6.0
    )
    assert evidence["worst_case_verify_attempts"] == 15
    assert evidence["cadence_seconds"] == 6.0
    assert evidence["attempt_margin"] == verdicts.ATTEMPT_MARGIN
    assert (
        verdicts.timing_evidence(
            lifetime_seconds=180, execution_timeout_seconds=180, cadence_seconds=None
        )["worst_case_verify_attempts"]
        is None
    )


# --------------------------------------------------------------------------- #
# Observed cadence
# --------------------------------------------------------------------------- #
def _ledger_row(
    command_id: str,
    operation: str,
    *,
    started: datetime | None,
    completed: datetime | None,
    state: str = "FAILED",
    attempt: int = 1,
) -> dict[str, Any]:
    return {
        "command_id": command_id,
        "operation": operation,
        "state": state,
        "attempt": attempt,
        "started_at": started.isoformat() if started else None,
        "completed_at": completed.isoformat() if completed else None,
    }


def _verify_rows(count: int, *, cadence: int = 5) -> list[dict[str, Any]]:
    return [
        _ledger_row(
            f"{WORKFLOW_ID}/3/VERIFY_NO_GPU_CLIENTS/commit-{index}",
            verdicts.WAITING_STEP,
            started=T0 + timedelta(seconds=90 + index * cadence),
            completed=T0 + timedelta(seconds=91 + index * cadence),
        )
        for index in range(count)
    ]


def test_the_cadence_is_measured_from_consecutive_verify_attempts() -> None:
    sample = verdicts.cadence_sample(
        [
            _ledger_row(
                BASELINE_COMMAND,
                verdicts.WAITING_STEP,
                started=T0 - timedelta(hours=1),
                completed=T0 - timedelta(hours=1),
            ),
            *_verify_rows(4),
        ],
        baseline_command_ids=BASELINE_IDS,
    )
    assert sample["row_count"] == 4
    assert sample["gap_count"] == 3
    assert sample["min_gap_seconds"] == 5.0
    assert verdicts.cadence_errors(sample) == []


def test_a_sample_with_too_few_gaps_cannot_judge_the_margin() -> None:
    sample = verdicts.cadence_sample(_verify_rows(2))
    assert sample["gap_count"] == 1
    assert "fewer than" in _text(verdicts.cadence_errors(sample))


def test_verify_rows_without_started_at_are_not_a_cadence() -> None:
    rows = [
        _ledger_row(
            f"{WORKFLOW_ID}/3/VERIFY_NO_GPU_CLIENTS/commit-{index}",
            verdicts.WAITING_STEP,
            started=None,
            completed=T0 + timedelta(seconds=90 + index * 5),
        )
        for index in range(5)
    ]
    sample = verdicts.cadence_sample(rows)
    assert sample["row_count"] == 0
    assert "0 gaps" in _text(verdicts.cadence_errors(sample))


# --------------------------------------------------------------------------- #
# Control plane: workflow, commands, escalation
# --------------------------------------------------------------------------- #
def happy_workflow() -> dict[str, Any]:
    return {
        "request_id": WORKFLOW_ID,
        "status": "FAILED",
        "lifetime_deadline_at": (T0 + timedelta(seconds=180)).isoformat(),
        "official_steps": [
            {"operation": operation} for operation in destr018.EXPECTED_STEPS
        ],
        "step_executions": [
            {"step_index": 0, "operation": "FREEZE_EVIDENCE", "status": "SUCCEEDED"},
            {"step_index": 1, "operation": "MARK_UNSCHEDULABLE", "status": "SUCCEEDED"},
            {
                "step_index": 2,
                "operation": "QUIESCE_GPU_SERVICES",
                "status": "SUCCEEDED",
            },
            {
                "step_index": 3,
                "operation": verdicts.WAITING_STEP,
                "status": "FAILED",
                "details": {
                    "workflow_lifetime_exceeded": True,
                    "workflow_execution_deadline": (
                        T0 + timedelta(seconds=180)
                    ).isoformat(),
                    "workflow_deadline_overdue_seconds": 20,
                    "workflow_deadline_remote_command_cancellation": {"cancelled": 1},
                },
            },
            {
                "step_index": 5,
                "operation": verdicts.COMPENSATION_STEP,
                "status": "SUCCEEDED",
            },
        ],
    }


def happy_incident() -> dict[str, Any]:
    return {
        "incident_id": "inc-1",
        "state": "ESCALATED",
        "node_ids": [NODE],
        "workflow_request_id": WORKFLOW_ID,
    }


def _workflow_errors(
    workflow: dict[str, Any] | None = None, incident: dict[str, Any] | None = None
) -> list[str]:
    return verdicts.workflow_errors(
        workflow if workflow is not None else happy_workflow(),
        incident if incident is not None else happy_incident(),
        node=NODE,
    )


def test_the_lifetime_contract_passes_on_the_intended_terminal_state() -> None:
    assert _workflow_errors() == []


def test_a_workflow_that_failed_for_another_reason_fails_the_case() -> None:
    workflow = happy_workflow()
    for execution in workflow["step_executions"]:
        execution.pop("details", None)
    assert "workflow_lifetime_exceeded" in _text(_workflow_errors(workflow))


def test_a_lifetime_failure_on_the_wrong_step_fails_the_case() -> None:
    workflow = happy_workflow()
    workflow["step_executions"][3]["operation"] = verdicts.RESET_STEP
    assert "did not land on VERIFY_NO_GPU_CLIENTS" in _text(_workflow_errors(workflow))


def test_a_lifetime_failure_missing_its_deadline_details_fails_the_case() -> None:
    workflow = happy_workflow()
    del workflow["step_executions"][3]["details"][
        "workflow_deadline_remote_command_cancellation"
    ]
    errors = _workflow_errors(workflow)
    assert "workflow_deadline_remote_command_cancellation" in _text(errors)


def test_a_workflow_without_a_lifetime_deadline_never_saw_the_env_window() -> None:
    workflow = happy_workflow()
    workflow["lifetime_deadline_at"] = None
    assert "no lifetime_deadline_at" in _text(_workflow_errors(workflow))


def test_a_reset_step_that_succeeded_after_the_deadline_fails_the_case() -> None:
    workflow = happy_workflow()
    workflow["step_executions"].append(
        {"step_index": 4, "operation": verdicts.RESET_STEP, "status": "SUCCEEDED"}
    )
    errors = _workflow_errors(workflow)
    assert "turned a late node result into success" in _text(errors)


def test_the_compensation_must_run_exactly_once() -> None:
    workflow = happy_workflow()
    workflow["step_executions"] = [
        item
        for item in workflow["step_executions"]
        if item["operation"] != verdicts.COMPENSATION_STEP
    ]
    assert "exactly one successful RESTORE_GPU_SERVICES step" in _text(
        _workflow_errors(workflow)
    )


@pytest.mark.parametrize(
    "operation", ["RESTART_NODE", "REPLACE_NODE", "VALIDATE_GPU", "RESTORE_SCHEDULING"]
)
def test_the_deadline_may_not_climb_the_remediation_ladder(operation: str) -> None:
    workflow = happy_workflow()
    workflow["step_executions"].append(
        {"step_index": 6, "operation": operation, "status": "SUCCEEDED"}
    )
    assert "operations the deadline forbids" in _text(_workflow_errors(workflow))


def test_an_incident_that_recovered_or_widened_fails_the_case() -> None:
    incident = happy_incident()
    incident["state"] = "RECOVERED"
    assert "not ESCALATED or QUARANTINED" in _text(_workflow_errors(None, incident))

    widened = happy_incident()
    widened["node_ids"] = [NODE, "node-b"]
    assert "not just node-a" in _text(_workflow_errors(None, widened))


def _command(
    operation: str,
    *,
    status: str,
    status_source: str | None = None,
    updated_at: datetime = T_CANCEL,
    cancellation_requested_at: datetime | None = None,
    result_details: dict[str, Any] | None = None,
    command_id: str = "",
) -> dict[str, Any]:
    return {
        "command_id": command_id or f"{WORKFLOW_ID}/x/{operation}/commit",
        "step": {"operation": operation},
        "status": status,
        "status_source": status_source,
        "updated_at": updated_at.isoformat(),
        "cancellation_requested_at": (
            cancellation_requested_at.isoformat() if cancellation_requested_at else None
        ),
        "result_details": result_details or {},
    }


def happy_commands() -> list[dict[str, Any]]:
    return [
        _command(
            "QUIESCE_GPU_SERVICES",
            status="SUCCEEDED",
            status_source="node-agent",
            updated_at=T0 + timedelta(seconds=30),
        ),
        _command(
            verdicts.WAITING_STEP,
            status="FAILED",
            status_source=verdicts.CANCELLED_BY_TIMEOUT,
            updated_at=T_CANCEL,
        ),
        _command(
            verdicts.COMPENSATION_STEP,
            status="SUCCEEDED",
            status_source="node-agent",
            updated_at=T_CANCEL + timedelta(seconds=30),
        ),
    ]


def test_the_cancellation_moment_is_the_earliest_cancellation_stamp() -> None:
    assert verdicts.cancellation_moment(happy_commands()) == T_CANCEL
    leased = happy_commands()
    leased[1] = _command(
        verdicts.RESET_STEP,
        status="FAILED",
        status_source=verdicts.COMPLETED_AFTER_CANCELLATION,
        updated_at=T_CANCEL + timedelta(seconds=40),
        cancellation_requested_at=T_CANCEL - timedelta(seconds=1),
        result_details={"post_cancellation_status": "FAILED"},
    )
    assert verdicts.cancellation_moment(leased) == T_CANCEL - timedelta(seconds=1)
    assert verdicts.cancellation_moment([happy_commands()[0]]) is None


def test_the_command_contract_passes_on_a_waiting_command_timed_out() -> None:
    assert verdicts.remote_command_errors(happy_commands(), t_cancel=T_CANCEL) == []


def test_a_cancelled_command_left_in_a_non_failed_state_fails_the_case() -> None:
    commands = happy_commands()
    commands[1]["status"] = "SUCCEEDED"
    errors = verdicts.remote_command_errors(commands, t_cancel=T_CANCEL)
    assert "is not FAILED" in _text(errors)


def test_a_late_completion_must_report_the_state_the_node_reached() -> None:
    commands = happy_commands()
    commands[1] = _command(
        verdicts.RESET_STEP,
        status="FAILED",
        status_source=verdicts.COMPLETED_AFTER_CANCELLATION,
        cancellation_requested_at=T_CANCEL,
        updated_at=T_CANCEL + timedelta(seconds=20),
    )
    errors = verdicts.remote_command_errors(commands, t_cancel=T_CANCEL)
    assert "post_cancellation_status" in _text(errors)


def test_a_command_that_succeeded_after_the_deadline_must_be_the_restore() -> None:
    commands = happy_commands()
    commands.append(
        _command(
            verdicts.RESET_STEP,
            status="SUCCEEDED",
            status_source="node-agent",
            updated_at=T_CANCEL + timedelta(seconds=10),
        )
    )
    errors = verdicts.remote_command_errors(commands, t_cancel=T_CANCEL)
    assert "only RESTORE_GPU_SERVICES is deadline-exempt" in _text(errors)


def test_a_deadline_that_cancelled_nothing_fails_the_case() -> None:
    commands = [
        _command(
            "QUIESCE_GPU_SERVICES",
            status="SUCCEEDED",
            status_source="node-agent",
            updated_at=T0,
        ),
        _command(
            verdicts.COMPENSATION_STEP,
            status="SUCCEEDED",
            status_source="node-agent",
            updated_at=T_CANCEL + timedelta(seconds=5),
        ),
    ]
    errors = verdicts.remote_command_errors(commands, t_cancel=T_CANCEL)
    assert "no remote command was issued for VERIFY_NO_GPU_CLIENTS" in _text(errors)
    assert "the deadline cancelled nothing" in _text(errors)


def happy_escalation() -> dict[str, Any]:
    return {
        "incident": {
            "incident_id": SUPPORT_INCIDENT_ID,
            "effective_action": verdicts.SUPPORT_ESCALATION_ACTION,
            "reasons": [f"{verdicts.SUPPORT_ESCALATION_STAGE}: deadline expired"],
            "node_ids": [NODE],
        },
        "workflow": {
            "request_id": SUPPORT_WORKFLOW_ID,
            "status": "SUCCEEDED",
            "official_steps": [
                {"operation": operation}
                for operation in verdicts.SUPPORT_ESCALATION_OPERATIONS
            ],
            "step_executions": [
                {"step_index": index, "operation": operation, "status": "SUCCEEDED"}
                for index, operation in enumerate(
                    verdicts.SUPPORT_ESCALATION_OPERATIONS
                )
            ],
        },
        "second_order_incident": None,
        "second_order_workflow": None,
        "executable_workflows": [],
    }


def test_the_escalation_contract_passes_on_one_reachable_human_handoff() -> None:
    assert verdicts.escalation_errors(happy_escalation(), node=NODE) == []


def test_a_lifetime_failure_that_escalated_nothing_fails_the_case() -> None:
    errors = verdicts.escalation_errors({"incident": None, "workflow": None}, node=NODE)
    assert "raised no support escalation" in _text(errors)


def test_a_support_workflow_that_inherited_an_expired_lifetime_fails_the_case() -> None:
    """The live defect this case is written to catch.

    ``HardwareEscalationService.emit`` copies ``lifetime_deadline_at`` into the
    support workflow and ``claim_deadlines`` never re-stamps it, so on a real
    deployment the handoff fails its own FREEZE_EVIDENCE and escalates again.
    """

    escalation = happy_escalation()
    escalation["workflow"]["status"] = "FAILED"
    escalation["workflow"]["step_executions"] = [
        {
            "step_index": 0,
            "operation": "FREEZE_EVIDENCE",
            "status": "FAILED",
            "details": {"workflow_lifetime_exceeded": True},
        }
    ]
    escalation["second_order_workflow"] = {
        "request_id": f"workflow-support-after-{SUPPORT_WORKFLOW_ID}",
        "status": "PENDING",
    }
    errors = verdicts.escalation_errors(escalation, node=NODE)
    assert "failed on the lifetime it inherited" in _text(errors)
    assert "escalated its own escalation" in _text(errors)
    assert "the node was never quarantined" in _text(errors)


def test_a_support_escalation_with_the_wrong_action_or_stage_fails() -> None:
    action = happy_escalation()
    action["incident"]["effective_action"] = "ESCALATE_SUPPORT"
    assert "action is not ESCALATE_OPERATOR" in _text(
        verdicts.escalation_errors(action, node=NODE)
    )

    stage = happy_escalation()
    stage["incident"]["reasons"] = ["reset_failed: adapter error"]
    assert "does not name the lifetime_exceeded stage" in _text(
        verdicts.escalation_errors(stage, node=NODE)
    )


def test_a_support_workflow_with_other_steps_fails_the_case() -> None:
    escalation = happy_escalation()
    escalation["workflow"]["official_steps"].append({"operation": "RESTART_NODE"})
    assert "support workflow steps are" in _text(
        verdicts.escalation_errors(escalation, node=NODE)
    )


def test_the_node_must_end_unschedulable_and_tainted() -> None:
    assert (
        verdicts.quarantine_errors(
            {
                "unschedulable": True,
                "taints": [{"key": verdicts.QUARANTINE_TAINT, "value": "abc"}],
            }
        )
        == []
    )
    errors = verdicts.quarantine_errors({"unschedulable": False, "taints": []})
    assert "does not carry the gpu-fault.io/quarantined taint" in _text(errors)
    assert "still schedulable" in _text(errors)


# --------------------------------------------------------------------------- #
# Record-only absorb of the second XID
# --------------------------------------------------------------------------- #
def happy_absorb() -> dict[str, Any]:
    return {
        "event": {"event_id": "evt-2", "xid": verdicts.ABSORB_XID},
        "incident": happy_incident(),
        "workflow": happy_workflow(),
    }


def _absorb_errors(snapshot: dict[str, Any]) -> list[str]:
    return verdicts.absorb_errors(
        snapshot,
        incident_id="inc-1",
        workflow_request_id=WORKFLOW_ID,
        official_step_count=len(destr018.EXPECTED_STEPS),
    )


def test_the_second_xid_lands_on_the_existing_incident_without_new_steps() -> None:
    assert _absorb_errors(happy_absorb()) == []


def test_the_second_xid_may_also_land_on_the_support_incident() -> None:
    snapshot = happy_absorb()
    snapshot["incident"] = {
        "incident_id": SUPPORT_INCIDENT_ID,
        "state": "ACTION_PENDING",
        "node_ids": [NODE],
    }
    snapshot["workflow"] = {"request_id": SUPPORT_WORKFLOW_ID, "status": "PENDING"}
    assert _absorb_errors(snapshot) == []


def test_a_second_xid_that_opened_a_new_incident_fails_the_case() -> None:
    snapshot = happy_absorb()
    snapshot["incident"] = {"incident_id": "inc-2", "node_ids": [NODE]}
    assert "opened incident inc-2" in _text(_absorb_errors(snapshot))


def test_an_absorb_that_added_steps_or_revived_the_workflow_fails() -> None:
    revived = happy_absorb()
    revived["workflow"]["status"] = "RUNNING"
    assert "revived the failed workflow" in _text(_absorb_errors(revived))

    grown = happy_absorb()
    grown["workflow"]["official_steps"].append({"operation": verdicts.RESET_STEP})
    assert "added steps to the failed workflow" in _text(_absorb_errors(grown))


def test_a_second_xid_that_never_arrived_fails_the_case() -> None:
    assert "never reached the control plane" in _text(_absorb_errors({}))


def test_no_executable_workflow_may_outlive_the_known_pair() -> None:
    known = {WORKFLOW_ID, SUPPORT_WORKFLOW_ID}
    assert (
        verdicts.new_executable_workflow_errors(
            [
                {"request_id": WORKFLOW_ID, "status": "FAILED"},
                {"request_id": SUPPORT_WORKFLOW_ID, "status": "PENDING"},
            ],
            known_request_ids=known,
        )
        == []
    )
    errors = verdicts.new_executable_workflow_errors(
        [{"request_id": "workflow-third", "status": "PENDING"}], known_request_ids=known
    )
    assert "left executable workflows behind" in _text(errors)


# --------------------------------------------------------------------------- #
# Data plane: the three refined verdicts
# --------------------------------------------------------------------------- #
def happy_ledger() -> list[dict[str, Any]]:
    return [
        _ledger_row(
            BASELINE_COMMAND,
            verdicts.WAITING_STEP,
            started=T0 - timedelta(hours=1),
            completed=T0 - timedelta(hours=1),
            state="SUCCEEDED",
        ),
        _ledger_row(
            f"{WORKFLOW_ID}/2/QUIESCE_GPU_SERVICES/commit",
            "QUIESCE_GPU_SERVICES",
            started=T0 + timedelta(seconds=30),
            completed=T0 + timedelta(seconds=40),
            state="SUCCEEDED",
        ),
        *_verify_rows(4),
        _ledger_row(
            f"{WORKFLOW_ID}/5/RESTORE_GPU_SERVICES/commit",
            verdicts.COMPENSATION_STEP,
            started=T_CANCEL + timedelta(seconds=10),
            completed=T_CANCEL + timedelta(seconds=25),
            state="SUCCEEDED",
        ),
    ]


EMPTY_JOURNAL = {"entry_count": 0, "target_reset_count": 0, "messages": []}


def _data_plane_errors(
    rows: list[dict[str, Any]],
    *,
    journal: dict[str, Any] | None = None,
    commands: list[dict[str, Any]] | None = None,
) -> list[str]:
    return verdicts.data_plane_errors(
        rows,
        t_cancel=T_CANCEL,
        baseline_command_ids=BASELINE_IDS,
        kernel_journal=journal if journal is not None else EMPTY_JOURNAL,
        commands=commands if commands is not None else happy_commands(),
    )


def test_the_data_plane_contract_passes_when_the_node_stopped_on_time() -> None:
    assert _data_plane_errors(happy_ledger()) == []


def test_a_reset_started_after_the_cancellation_is_the_strict_failure() -> None:
    rows = [
        *happy_ledger(),
        _ledger_row(
            f"{WORKFLOW_ID}/4/RESET_GPU/commit",
            verdicts.RESET_STEP,
            started=T_CANCEL + timedelta(seconds=5),
            completed=T_CANCEL + timedelta(seconds=45),
        ),
    ]
    errors = verdicts.late_row_errors(
        rows, t_cancel=T_CANCEL, baseline_command_ids=BASELINE_IDS
    )
    assert "the node started RESET_GPU at" in _text(errors)
    assert "after the cancellation" in _text(errors)


def test_a_mutating_row_without_started_at_is_undecidable_not_a_pass() -> None:
    rows = [
        *happy_ledger(),
        _ledger_row(
            f"{WORKFLOW_ID}/4/RESET_GPU/commit",
            verdicts.RESET_STEP,
            started=None,
            completed=T_CANCEL + timedelta(seconds=45),
        ),
    ]
    errors = verdicts.late_row_errors(
        rows, t_cancel=T_CANCEL, baseline_command_ids=BASELINE_IDS
    )
    assert "has no started_at" in _text(errors)
    assert "cannot prove the node stopped" in _text(errors)


def test_a_baseline_row_from_an_earlier_case_is_never_judged() -> None:
    rows = [
        _ledger_row(
            BASELINE_COMMAND,
            verdicts.RESET_STEP,
            started=T_CANCEL + timedelta(seconds=5),
            completed=T_CANCEL + timedelta(seconds=45),
        )
    ]
    assert (
        verdicts.late_row_errors(
            rows, t_cancel=T_CANCEL, baseline_command_ids={BASELINE_COMMAND}
        )
        == []
    )


def _straddling_reset(state: str = "FAILED") -> dict[str, Any]:
    return _ledger_row(
        f"{WORKFLOW_ID}/4/RESET_GPU/commit",
        verdicts.RESET_STEP,
        started=T_CANCEL - timedelta(seconds=5),
        completed=T_CANCEL + timedelta(seconds=40),
        state=state,
    )


def _leased_reset_command(
    *,
    status: str = "FAILED",
    status_source: str = verdicts.COMPLETED_AFTER_CANCELLATION,
    post_status: str | None = "FAILED",
) -> dict[str, Any]:
    return _command(
        verdicts.RESET_STEP,
        status=status,
        status_source=status_source,
        updated_at=T_CANCEL + timedelta(seconds=45),
        cancellation_requested_at=T_CANCEL,
        result_details=(
            {"post_cancellation_status": post_status} if post_status else {}
        ),
        command_id=f"{WORKFLOW_ID}/4/RESET_GPU/commit",
    )


def test_one_straddling_row_is_tolerated_when_everything_agrees() -> None:
    rows = [*happy_ledger(), _straddling_reset()]
    commands = [*happy_commands(), _leased_reset_command()]
    assert _data_plane_errors(rows, commands=commands) == []


def test_two_straddling_rows_fail_the_case() -> None:
    second = _ledger_row(
        f"{WORKFLOW_ID}/3/VERIFY_NO_GPU_CLIENTS/commit-late",
        verdicts.WAITING_STEP,
        started=T_CANCEL - timedelta(seconds=2),
        completed=T_CANCEL + timedelta(seconds=10),
    )
    rows = [*happy_ledger(), _straddling_reset(), second]
    errors = verdicts.straddling_row_errors(
        rows,
        t_cancel=T_CANCEL,
        baseline_command_ids=BASELINE_IDS,
        kernel_journal=EMPTY_JOURNAL,
        commands=happy_commands(),
    )
    assert "more than one ledger row straddles" in _text(errors)


def test_a_straddling_reset_must_agree_with_the_kernel_journal() -> None:
    rows = [*happy_ledger(), _straddling_reset(state="SUCCEEDED")]
    commands = [*happy_commands(), _leased_reset_command(post_status="SUCCEEDED")]
    errors = verdicts.straddling_row_errors(
        rows,
        t_cancel=T_CANCEL,
        baseline_command_ids=BASELINE_IDS,
        kernel_journal=EMPTY_JOURNAL,
        commands=commands,
        expect_forced_failure=False,
    )
    assert "claims SUCCEEDED but the kernel journal shows 0 resets" in _text(errors)

    failed_rows = [*happy_ledger(), _straddling_reset()]
    errors = verdicts.straddling_row_errors(
        failed_rows,
        t_cancel=T_CANCEL,
        baseline_command_ids=BASELINE_IDS,
        kernel_journal={"target_reset_count": 1, "entry_count": 1, "messages": []},
        commands=[*happy_commands(), _leased_reset_command()],
    )
    assert "claims FAILED but the kernel journal shows 1 resets" in _text(errors)


def test_the_holder_drill_expects_every_straddling_attempt_to_have_failed() -> None:
    rows = [*happy_ledger(), _straddling_reset(state="SUCCEEDED")]
    commands = [*happy_commands(), _leased_reset_command(post_status="SUCCEEDED")]
    errors = _data_plane_errors(
        rows,
        journal={"target_reset_count": 1, "entry_count": 1, "messages": []},
        commands=commands,
    )
    assert "the device holder was meant to make every attempt fail" in _text(errors)


def test_the_control_plane_may_not_turn_a_late_result_into_success() -> None:
    rows = [*happy_ledger(), _straddling_reset()]
    commands = [
        *happy_commands(),
        _leased_reset_command(status="SUCCEEDED", status_source="node-agent"),
    ]
    errors = verdicts.straddling_row_errors(
        rows,
        t_cancel=T_CANCEL,
        baseline_command_ids=BASELINE_IDS,
        kernel_journal=EMPTY_JOURNAL,
        commands=commands,
    )
    assert "a cancelled command stays FAILED" in _text(errors)
    assert "not one of ['workflow-timeout', 'completed-after-cancellation']" in _text(
        errors
    )


def test_a_late_completion_must_report_the_straddling_rows_own_state() -> None:
    rows = [*happy_ledger(), _straddling_reset()]
    commands = [*happy_commands(), _leased_reset_command(post_status="SUCCEEDED")]
    errors = verdicts.straddling_row_errors(
        rows,
        t_cancel=T_CANCEL,
        baseline_command_ids=BASELINE_IDS,
        kernel_journal=EMPTY_JOURNAL,
        commands=commands,
    )
    assert "post_cancellation_status is SUCCEEDED, not" in _text(errors)


def test_a_straddling_row_with_no_matching_command_fails_the_case() -> None:
    rows = [*happy_ledger(), _straddling_reset()]
    errors = verdicts.straddling_row_errors(
        rows,
        t_cancel=T_CANCEL,
        baseline_command_ids=BASELINE_IDS,
        kernel_journal=EMPTY_JOURNAL,
        commands=[happy_commands()[0]],
    )
    assert "no remote command matches the straddling ledger row" in _text(errors)


def test_the_one_compensation_after_the_cancellation_must_have_succeeded() -> None:
    rows = [
        row for row in happy_ledger() if row["operation"] != verdicts.COMPENSATION_STEP
    ]
    errors = verdicts.compensation_row_errors(
        rows, t_cancel=T_CANCEL, baseline_command_ids=BASELINE_IDS
    )
    assert "expected exactly one RESTORE_GPU_SERVICES ledger row" in _text(errors)

    failed = happy_ledger()
    failed[-1]["state"] = "FAILED"
    errors = verdicts.compensation_row_errors(
        failed, t_cancel=T_CANCEL, baseline_command_ids=BASELINE_IDS
    )
    assert "the one deadline-exempt compensation failed" in _text(errors)


def test_a_compensation_row_without_started_at_cannot_be_placed() -> None:
    rows = happy_ledger()
    rows[-1]["started_at"] = None
    errors = verdicts.compensation_row_errors(
        rows, t_cancel=T_CANCEL, baseline_command_ids=BASELINE_IDS
    )
    assert "has no started_at" in _text(errors)


# --------------------------------------------------------------------------- #
# Optional metric evidence
# --------------------------------------------------------------------------- #
def test_counters_are_summed_across_every_replica_that_reports_them() -> None:
    first = f"{verdicts.LIFETIME_METRIC} 2.0\n"
    second = f"{verdicts.LIFETIME_METRIC} 3.0\n"
    assert verdicts.counter_total([first, second], verdicts.LIFETIME_METRIC) == 5.0
    assert verdicts.counter_total(["# nothing\n"], verdicts.LIFETIME_METRIC) is None
    assert verdicts.counter_value(first, verdicts.LIFETIME_METRIC) == 2.0


def test_metric_evidence_reports_the_delta_of_both_counters() -> None:
    before = [f"{verdicts.LIFETIME_METRIC} 1.0\n{verdicts.RECORD_ONLY_METRIC} 4.0\n"]
    after = [f"{verdicts.LIFETIME_METRIC} 2.0\n{verdicts.RECORD_ONLY_METRIC} 5.0\n"]
    evidence = verdicts.metric_evidence(before, after)
    assert evidence["lifetime_exceeded_total"]["delta"] == 1.0
    assert evidence["merge_record_only_total"]["delta"] == 1.0
    assert verdicts.metric_errors(evidence) == []


def test_an_unreadable_counter_is_recorded_rather_than_failed() -> None:
    evidence = verdicts.metric_evidence(["# no counters\n"], ["# no counters\n"])
    assert evidence["lifetime_exceeded_total"]["delta"] is None
    assert verdicts.metric_errors(evidence) == []


def test_a_readable_counter_that_did_not_move_fails_the_case() -> None:
    before = [f"{verdicts.LIFETIME_METRIC} 7.0\n{verdicts.RECORD_ONLY_METRIC} 1.0\n"]
    after = [f"{verdicts.LIFETIME_METRIC} 7.0\n{verdicts.RECORD_ONLY_METRIC} 2.0\n"]
    errors = verdicts.metric_errors(verdicts.metric_evidence(before, after))
    assert "did not move across the drill" in _text(errors)
    assert len(errors) == 1


# --------------------------------------------------------------------------- #
# Runner: preflight, identity, parser
# --------------------------------------------------------------------------- #
def _settings(tmp_path: Path) -> destr018.Settings:
    cpu = tmp_path / "cpu.kubeconfig"
    gpu = tmp_path / "gpu.kubeconfig"
    cpu.write_text("apiVersion: v1\n", encoding="utf-8")
    gpu.write_text("apiVersion: v1\n", encoding="utf-8")
    return destr018.Settings(
        regional=RegionalLiveSettings(
            cpu_kubeconfig=cpu,
            gpu_kubeconfig=gpu,
            gpu_context="gpu-context",
            namespace="gpu-fault-system",
            cluster_id="cluster-a",
            region="us-west-2",
        ),
        node=NODE,
        host_probe_image="registry.example/probe@sha256:" + "a" * 64,
        predecessor_path=tmp_path / "predecessor.json",
        lifetime_seconds=verdicts.LIFETIME_SECONDS,
        execution_timeout_seconds=verdicts.EXECUTION_TIMEOUT_SECONDS,
        step_timeout_seconds=verdicts.STEP_TIMEOUT_SECONDS,
        managed_recovery_seconds=verdicts.MANAGED_RECOVERY_SECONDS,
        step_warning_seconds=verdicts.STEP_WARNING_SECONDS,
        lease_duration_seconds=verdicts.LEASE_DURATION_SECONDS,
        hold_seconds=1200,
    )


def _survey(**values: str | None) -> dict[str, Any]:
    replica_values = {name: None for name in env_window.SURVEYED_VARIABLES}
    replica_values.update(values)
    return {
        "deployment": {
            "variables": {
                name: {"present": False, "value": None}
                for name in env_window.ALLOWED_VARIABLES
            }
        },
        "replicas": [
            {"pod": "worker-1", "values": dict(replica_values)},
            {"pod": "worker-2", "values": dict(replica_values)},
        ],
    }


def _state() -> dict[str, Any]:
    return {
        "release_id": "rel-1",
        "agent": {
            "lifecycle_state": "ACTIVE",
            "generation": 4,
            "allowed_operations": sorted(destr018.REQUIRED_AGENT_OPERATIONS),
        },
        "profile": {
            "profile_version": "v9",
            "warnings": [],
            "capabilities": [
                {
                    "capability": "gpuReset",
                    "mode": "OWN",
                    "owner": "gpu-fault-node-agent",
                    "adapter": "node-action",
                }
            ],
        },
        "queue": {"depth": 0},
        "remote_commands": {"open_by_cluster": {}},
        "event": None,
    }


def _node() -> dict[str, Any]:
    return {
        "name": NODE,
        "uid": "uid-1",
        "ready": "True",
        "unschedulable": False,
        "taints": [],
        "gpu_allocatable": "8",
        "ownership_annotations": {},
    }


def _preflight_errors(
    tmp_path: Path,
    *,
    state: dict[str, Any] | None = None,
    node: dict[str, Any] | None = None,
    workloads: list[dict[str, str]] | None = None,
    tests: dict[str, Any] | None = None,
    survey: dict[str, Any] | None = None,
) -> list[str]:
    return destr018.preflight_errors(
        _settings(tmp_path),
        state if state is not None else _state(),
        node if node is not None else _node(),
        workloads if workloads is not None else [],
        tests if tests is not None else {"passed": True},
        survey if survey is not None else _survey(),
    )


def test_the_preflight_passes_on_an_idle_node_and_a_closed_window(
    tmp_path: Path,
) -> None:
    assert _preflight_errors(tmp_path) == []


def test_the_preflight_refuses_a_window_someone_else_left_open(tmp_path: Path) -> None:
    survey = _survey()
    survey["deployment"]["variables"][env_window.LIFETIME_VARIABLE] = {
        "present": True,
        "value": "240",
    }
    errors = _preflight_errors(tmp_path, survey=survey)
    assert "an env window is open that this run did not record" in _text(errors)


def test_the_preflight_refuses_replicas_that_disagree_on_the_bounds(
    tmp_path: Path,
) -> None:
    survey = _survey()
    survey["replicas"][1]["values"][destr018.POLL_INTERVAL_VARIABLE] = "30"
    errors = _preflight_errors(tmp_path, survey=survey)
    assert "cadence is unknown" in _text(errors)


def test_the_preflight_reads_an_unset_variable_as_the_shipped_default() -> None:
    assert destr018.observed_cadence(_survey()) == 5.0
    assert destr018.observed_step_cap(_survey()) == 600
    lowered = _survey(**{destr018.STEP_TIMEOUT_VARIABLE: "120"})
    assert destr018.observed_step_cap(lowered) == 120
    assert destr018.observed_cadence({"replicas": []}) is None


def test_the_preflight_refuses_a_run_configured_to_leave_the_step_cap_low(
    tmp_path: Path,
) -> None:
    """The margin is judged against the in-window cap the run *will set*, not the
    pre-window survey value (the window compresses the step timeout to the
    lifetime). A run misconfigured to leave the step timeout at 120s while the
    lifetime is 180s would let the step's own bound end the wait first, so the
    preflight refuses it."""

    settings = replace(_settings(tmp_path), step_timeout_seconds=120)
    errors = destr018.preflight_errors(
        settings,
        _state(),
        _node(),
        [],
        {"passed": True},
        _survey(),
    )
    assert "per-step waiting cap 120s" in _text(errors)


def test_the_preflight_refuses_a_busy_or_isolated_or_faulted_node(
    tmp_path: Path,
) -> None:
    node = _node()
    node["unschedulable"] = True
    node["taints"] = [{"key": verdicts.QUARANTINE_TAINT}]
    errors = _preflight_errors(tmp_path, node=node)
    assert "already unschedulable" in _text(errors)
    assert "pre-existing taints" in _text(errors)

    assert "non-system running Pods" in _text(
        _preflight_errors(tmp_path, workloads=[{"name": "trainer"}])
    )

    state = _state()
    state["event"] = {"event_id": "evt-0", "xid": 46}
    assert "recent XID event" in _text(_preflight_errors(tmp_path, state=state))

    delegated = _state()
    delegated["profile"]["capabilities"][0]["mode"] = "DELEGATE"
    assert "gpuReset is not OWN" in _text(_preflight_errors(tmp_path, state=delegated))

    partial = _state()
    partial["agent"]["allowed_operations"] = ["QUIESCE_GPU_SERVICES"]
    assert "allowlist is incomplete" in _text(
        _preflight_errors(tmp_path, state=partial)
    )

    assert "focused regression tests failed" in _text(
        _preflight_errors(tmp_path, tests={"passed": False})
    )


def test_the_preflight_gates_on_the_fault_tier_not_routine_telemetry(
    tmp_path: Path,
) -> None:
    """The env window rolls the control-worker, so the gate must be clear of
    roll-unsafe fault-tier work. A routine gpu-inventory backlog (the stale
    fencing-token livelock) leaves total depth non-zero while the fault tier is
    empty, and must not flap this preflight."""

    fault_busy = _state()
    fault_busy["queue"] = {"depth": 2, "fault_backlog_depth": 2}
    assert "processor fault-tier backlog is not empty (2)" in _text(
        _preflight_errors(tmp_path, state=fault_busy)
    )

    routine_only = _state()
    routine_only["queue"] = {"depth": 1, "fault_backlog_depth": 0}
    assert _preflight_errors(tmp_path, state=routine_only) == []


def test_the_preflight_falls_back_to_total_depth_without_a_fault_reading(
    tmp_path: Path,
) -> None:
    """A snapshot from before the fault-tier reading existed still gates on the
    total depth, so the check never silently becomes a no-op."""

    legacy = _state()
    legacy["queue"] = {"depth": 1}
    assert "processor queue is not empty" in _text(
        _preflight_errors(tmp_path, state=legacy)
    )


def _rolled_out(generation: int, template: str) -> dict[str, Any]:
    return {
        "generation": generation,
        "observed_generation": generation,
        "desired_replicas": 2,
        "updated_replicas": 2,
        "ready_replicas": 2,
        "available_replicas": 2,
        "template_sha256": template,
        "images": ["registry/example@sha256:" + "b" * 64],
    }


def _identity(*, generation: int, template: str = "sha-1") -> dict[str, Any]:
    """A runtime identity the shared safety check also accepts.

    ``runtime_identity_errors`` demands every deployment of both planes, fully
    rolled out, so the fixture names them from the shared constant rather than
    a hand-picked subset.
    """

    return {
        "release_state": {"release_id": "rel-1", "phase": "committed"},
        "deployments": {
            plane: {
                name: (
                    _rolled_out(generation, template)
                    if name == env_window.DEPLOYMENT
                    else _rolled_out(3, f"sha-{name}")
                )
                for name in names
            }
            for plane, names in RUNTIME_IDENTITY_DEPLOYMENTS.items()
        },
    }


def test_a_closed_window_gives_back_the_deployment_it_borrowed() -> None:
    before = _identity(generation=7)
    after = _identity(generation=9)
    assert destr018.identity_errors(before, after, worker_generation_delta=2) == []


def test_an_env_restored_to_a_different_template_fails_the_case() -> None:
    before = _identity(generation=7)
    after = _identity(generation=9, template="sha-2")
    errors = destr018.identity_errors(before, after, worker_generation_delta=2)
    assert "differs from the pre-window baseline" in _text(errors)


def test_the_open_window_may_change_the_worker_template() -> None:
    """While the window is open, setting the managed env vars is the point, so
    the worker's ``template_sha256`` is expected to differ and is not drift."""

    before = _identity(generation=7)
    after = _identity(generation=8, template="sha-in-window")
    assert (
        destr018.identity_errors(
            before,
            after,
            worker_generation_delta=1,
            allow_worker_template_change=True,
        )
        == []
    )


def test_the_open_window_still_catches_non_template_worker_drift() -> None:
    """The template escape hatch is narrow: an off-by-one generation, a settling
    rollout, a sibling deployment's drift, or a release-state change must still
    fail even while the worker template is allowed to move."""

    before = _identity(generation=7)
    wrong_delta = _identity(generation=10, template="sha-in-window")
    assert "generation is 10, not the 8" in _text(
        destr018.identity_errors(
            before,
            wrong_delta,
            worker_generation_delta=1,
            allow_worker_template_change=True,
        )
    )

    sibling = _identity(generation=8, template="sha-in-window")
    sibling["deployments"]["cpu"]["gpu-fault-api-ha"]["template_sha256"] = "sha-other"
    assert "gpu-fault-api-ha identity drifted" in _text(
        destr018.identity_errors(
            before,
            sibling,
            worker_generation_delta=1,
            allow_worker_template_change=True,
        )
    )

    released = _identity(generation=8, template="sha-in-window")
    released["release_state"]["release_id"] = "rel-2"
    assert "release state identity drifted" in _text(
        destr018.identity_errors(
            before,
            released,
            worker_generation_delta=1,
            allow_worker_template_change=True,
        )
    )


def test_an_unexplained_worker_rollout_fails_the_case() -> None:
    before = _identity(generation=7)
    after = _identity(generation=12)
    errors = destr018.identity_errors(before, after, worker_generation_delta=2)
    assert "generation is 12, not the 9" in _text(errors)


def test_a_settling_rollout_is_not_a_restored_identity() -> None:
    before = _identity(generation=7)
    after = _identity(generation=9)
    after["deployments"]["cpu"][env_window.DEPLOYMENT]["observed_generation"] = 8
    errors = destr018.identity_errors(before, after, worker_generation_delta=2)
    assert "rollout is not settled" in _text(errors)


def test_a_deployment_the_window_never_touched_must_be_identical() -> None:
    before = _identity(generation=7)
    after = _identity(generation=9)
    after["deployments"]["cpu"]["gpu-fault-api-ha"]["template_sha256"] = "sha-other"
    errors = destr018.identity_errors(before, after, worker_generation_delta=2)
    assert "gpu-fault-api-ha identity drifted" in _text(errors)

    released = _identity(generation=9)
    released["release_state"]["release_id"] = "rel-2"
    errors = destr018.identity_errors(before, released, worker_generation_delta=2)
    assert "release state identity drifted" in _text(errors)


# --------------------------------------------------------------------------- #
# Plan and command line
# --------------------------------------------------------------------------- #
def test_the_plan_names_the_risk_the_window_and_the_stop_conditions(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    preflight = {
        "release_id": "rel-1",
        "node": _node(),
        "store": _state(),
        "predecessor": {"valid": True, "case_id": destr018.PREDECESSOR_CASE_ID},
        "timing": verdicts.timing_evidence(
            lifetime_seconds=180, execution_timeout_seconds=180, cadence_seconds=5.0
        ),
    }
    details = destr018.plan_details(settings, preflight)
    assert details["risk"] == "live-service-action"
    assert details["target_node"] == NODE
    assert env_window.DEPLOYMENT in details["mutation"]
    assert details["preflight_identity"]["node_uid"] == "uid-1"
    assert details["rollback"]["quarantine_taint_is_never_deleted_by_hand"] is True
    assert (
        details["rollback"][
            "runtime_identity_is_compared_against_the_pre_window_baseline"
        ]
        is True
    )
    conditions = "\n".join(details["stop_conditions"])
    assert "attempt margin" in conditions
    assert "env window is already open" in conditions
    assert "holder is not visible" in conditions
    assert json.dumps(details, sort_keys=True, default=str).count(NODE) >= 1


def test_a_plan_that_drifted_from_its_preflight_is_refused(tmp_path: Path) -> None:
    case_dir = tmp_path / "cases" / destr018.CASE_ID
    case_dir.mkdir(parents=True)
    preflight = {
        "release_id": "rel-1",
        "node": _node(),
        "store": _state(),
        "predecessor": {"valid": True},
        "timing": {},
    }
    (case_dir / "plan.json").write_text(
        json.dumps({"details": {"preflight_identity": {"release_id": "rel-0"}}}),
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="plan drifted"):
        destr018.verify_plan_identity(case_dir, preflight)


def test_the_runner_is_plan_by_default_and_needs_an_exact_confirmation() -> None:
    parser = destr018.parser()
    plan = parser.parse_args(["--run-dir", "/tmp/run"])
    assert plan.execute is False
    assert plan.lifetime_seconds == verdicts.LIFETIME_SECONDS
    assert plan.execution_timeout_seconds == verdicts.EXECUTION_TIMEOUT_SECONDS
    assert plan.step_timeout_seconds == verdicts.STEP_TIMEOUT_SECONDS
    assert plan.managed_recovery_seconds == verdicts.MANAGED_RECOVERY_SECONDS
    assert plan.step_warning_seconds == verdicts.STEP_WARNING_SECONDS
    assert plan.lease_duration_seconds == verdicts.LEASE_DURATION_SECONDS
    execute = parser.parse_args(
        [
            "--run-dir",
            "/tmp/run",
            "--execute",
            "--confirm",
            destr018.CONFIRMATION,
            "--maintenance-window-end",
            "2026-09-06T12:00:00+00:00",
            "--node",
            NODE,
        ]
    )
    assert execute.execute is True
    assert execute.confirm == destr018.CONFIRMATION
    assert execute.node == NODE
    with pytest.raises(SystemExit):
        parser.parse_args(["--run-dir", "/tmp/run", "--plan", "--execute"])


def test_the_help_text_offers_the_four_documented_live_flags() -> None:
    help_text = destr018.parser().format_help()
    for flag in ("--plan", "--execute", "--confirm", "--maintenance-window-end"):
        assert flag in help_text


def test_the_runner_and_the_env_window_helper_are_executable_with_a_shebang() -> None:
    for path in (
        ROOT / "scripts/e2e/regional/run_destr018_lifetime_deadline.py",
        ROOT / "scripts/e2e/regional/control_plane_env_window.py",
        ROOT / "scripts/e2e/regional/probes/destr018_node_probe.py",
    ):
        mode = path.stat().st_mode & 0o777
        assert mode == 0o775, f"{path.name} is {oct(mode)}, not 0o775"
        first = path.read_text(encoding="utf-8").splitlines()[0]
        assert first == "#!/usr/bin/env python3"


def test_the_expected_step_sequence_is_the_reset_contract() -> None:
    assert destr018.EXPECTED_STEPS.index(verdicts.WAITING_STEP) == 3
    assert destr018.EXPECTED_STEPS.index(verdicts.RESET_STEP) == 4
    assert destr018.EXPECTED_STEPS.index(verdicts.COMPENSATION_STEP) == 5
    assert isinstance(destr018.parser(), argparse.ArgumentParser) is True
