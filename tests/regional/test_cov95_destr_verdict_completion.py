from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from scripts.e2e.regional import destr014_verdicts as v14
from scripts.e2e.regional import destr015_verdicts as v15
from scripts.e2e.regional import destr016_verdicts as v16
from scripts.e2e.regional import destr017_verdicts as v17
from scripts.e2e.regional import destr018_verdicts as v18
from scripts.e2e.regional import destr022_verdicts as v22
from tests.regional import test_destr014_branch_exhaustion as d14
from tests.regional import test_destr015_parallel_branch_join as d15
from tests.regional import test_destr016_preempting_reboot as d16
from tests.regional import test_destr017_out_of_band_reboot_fence as d17
from tests.regional import test_destr018_lifetime_deadline as d18


@pytest.mark.parametrize(
    ("node", "operation", "expected"),
    [
        (d14.FAULT, "RESTART_NODE", "in-place RESTART_NODE did not SUCCEED"),
        (d14.SIBLING, "RESTART_NODE", "has no FAILED RESTART_NODE"),
        (d14.SIBLING, "REPLACE_NODE", "branch has no REPLACE_NODE step"),
    ],
)
def test_exhaustion_requires_both_branches_to_reach_their_exact_terminal_steps(
    node: str, operation: str, expected: str
) -> None:
    workflow = d14.happy_workflow()
    step = next(
        row
        for row in workflow["official_steps"]
        if row["node_ids"] == [node] and row["operation"] == operation
    )
    index = workflow["official_steps"].index(step)
    if operation == "REPLACE_NODE":
        step["operation"] = "FREEZE_EVIDENCE"
    else:
        for row in workflow["step_executions"]:
            if row["step_index"] == index:
                row["status"] = "CANCELLED"
    errors = v14.workflow_errors(
        workflow,
        d14.happy_incident(),
        fault_node=d14.FAULT,
        sibling_node=d14.SIBLING,
        failure_reason=f"node branch escalation exhausted: branch:{d14.SIBLING}",
    )
    assert any(expected in error for error in errors), errors


def test_exhaustion_support_handoff_requires_a_support_step() -> None:
    errors = v14.follow_up_errors(
        {
            "incident": {"node_ids": [d14.SIBLING]},
            "workflow": {"official_steps": [{"operation": "FREEZE_EVIDENCE"}]},
        },
        fault_node=d14.FAULT,
        sibling_node=d14.SIBLING,
    )
    assert errors == ["follow-up workflow has no ESCALATE_SUPPORT step"], errors


@pytest.mark.parametrize("defect", ["stop", "join", "timestamps", "serial"])
def test_parallel_join_requires_shared_steps_and_separate_physical_overlap(
    defect: str,
) -> None:
    workflow = d15.happy_workflow()
    if defect in {"stop", "join"}:
        operation = "STOP_WORKLOADS" if defect == "stop" else "RESTART_WORKLOAD"
        workflow["official_steps"] = [
            row for row in workflow["official_steps"] if row["operation"] != operation
        ]
        expected = f"does not contain exactly one {operation}"
    else:
        for row in workflow["step_executions"]:
            index = row["step_index"]
            if workflow["official_steps"][index]["node_ids"] != [d15.NODE_B]:
                continue
            for key in ("started_at", "updated_at"):
                row[key] = (
                    None
                    if defect == "timestamps"
                    else (
                        datetime.fromisoformat(row[key]) + timedelta(hours=1)
                    ).isoformat()
                )
        expected = (
            "incomplete or unordered timestamps"
            if defect == "timestamps"
            else "actual reset process intervals"
        )
    errors = v15.workflow_errors(
        workflow,
        d15.happy_incident(),
        nodes=d15.NODES,
        expected_gpu_count=16,
        restart_budget=d15.happy_budget(),
    )
    if defect == "serial":
        from scripts.e2e.regional.destr015_physical_evidence import (
            physical_overlap_errors,
        )
        from tests.regional.test_acceptance_physical_interval_alignment import (
            BASE,
            SECOND,
            interval_fixture,
        )

        captures, scopes, physical_workflow, hosts = interval_fixture()
        captures["node-b"]["end"]["actions"][0].update(
            started_ns=BASE + 22 * SECOND, ended_ns=BASE + 29 * SECOND
        )
        errors.extend(
            physical_overlap_errors(
                captures, scopes=scopes, workflow=physical_workflow, hosts=hosts
            )
        )
    assert any(expected in error for error in errors), errors


@pytest.mark.parametrize("defect", ["policy", "unlinked", "foreign-incident"])
def test_parallel_injections_need_one_reset_workflow_with_consistent_incident(
    defect: str,
) -> None:
    states = [
        {
            "event": {"xid": 46, "evidence_ref": f"kmsg://{node}/unit"},
            "decision": {"official_action": "RESET_GPU"},
            "workflow": {"request_id": "shared"},
            "incident": {"workflow_request_id": "shared"},
        }
        for node in d15.NODES
    ]
    assert v15.injection_errors(*states, nodes=d15.NODES) == [], states
    if defect == "policy":
        states[0]["decision"]["official_action"] = "RESTART_NODE"
        expected = "did not resolve to RESET_GPU"
    elif defect == "unlinked":
        states[0]["workflow"] = states[0]["incident"] = {}
        expected = "event has no workflow"
    else:
        states[0]["incident"]["workflow_request_id"] = "another"
        expected = "incident points at another workflow"
    errors = v15.injection_errors(*states, nodes=d15.NODES)
    assert any(expected in error for error in errors), errors


@pytest.mark.parametrize("defect", ["missing-id", "wrong-action"])
def test_preemption_cannot_handoff_to_an_unknown_or_wrong_action_incident(
    defect: str,
) -> None:
    predecessor = d16.superseded_reset_workflow()
    incident = d16.escalated_incident()
    if defect == "missing-id":
        predecessor["request_id"] = ""
        expected = "workflow is unknown"
    else:
        incident["official_action"] = "RESET_GPU"
        expected = "incident action is not"
    errors = v16.escalation_errors(
        predecessor,
        d16.successor_workflow(),
        incident,
        decision={"official_action": "RESTART_BM"},
    )
    assert any(expected in error for error in errors), errors


@pytest.mark.parametrize("timestamp", [None, "unparseable", "2026-09-06T10:00:00"])
def test_preemption_restore_timestamps_keep_the_declared_naive_utc_policy(
    timestamp: str | None,
) -> None:
    workflow = d16.terminal_state()["workflow"]
    reboot = next(
        row
        for row in workflow["step_executions"]
        if row["operation"] == "RESTART_NODE" and row["status"] == "SUCCEEDED"
    )
    restore = next(
        row
        for row in workflow["step_executions"]
        if row["operation"] == "RESTORE_GPU_SERVICES"
    )
    reboot["updated_at"] = timestamp
    restore["started_at"] = "2026-09-06T10:00:01"
    errors = v16.restore_after_reboot_errors(workflow)
    if timestamp is None or timestamp == "unparseable":
        assert errors == ["the reboot/restore timestamps are incomplete"], errors
    else:
        assert errors == [], errors


@pytest.mark.parametrize(
    "defect", ["missing-boot", "missing-barrier", "inactive-baseline-service"]
)
def test_preemption_host_proof_distinguishes_missing_actions_from_optional_services(
    defect: str,
) -> None:
    baseline, after = d16.host_baseline(), d16.host_after()
    expected = ""
    if defect == "missing-boot":
        baseline["boot_id"] = ""
        expected = "host boot id snapshot is missing"
    elif defect == "missing-barrier":
        after["ledger"] = [
            row for row in after["ledger"] if row["operation"] != v16.BARRIER_OPERATION
        ]
        expected = "has no VERIFY_NO_GPU_CLIENTS row"
    else:
        baseline["services"]["optional.service"] = {"ActiveState": "inactive"}
    errors = v16.host_errors(baseline, after, expected_gpu_count=2)
    if expected:
        assert any(expected in error for error in errors), errors
    else:
        assert errors == [], errors


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("baseline_boot_id", "", "not captured on both sides"),
        ("host_boot_ids", ["different", "after"], "does not match"),
        (
            "reboot_status",
            {"fired": False, "boot_id_before_reboot": "before"},
            "does not record a fired reboot",
        ),
    ],
)
def test_fence_reboot_proof_requires_two_matching_independent_witnesses(
    field: str, value: Any, expected: str
) -> None:
    frame: dict[str, Any] = {
        "node": d17.NODE,
        "baseline_boot_id": "before",
        "final_boot_id": "after",
        "host_boot_ids": ["before", "after"],
        "reboot_status": {"fired": True, "boot_id_before_reboot": "before"},
    }
    assert v17.boot_errors(**frame) == [], frame
    frame[field] = value
    errors = v17.boot_errors(**frame)
    assert any(expected in error for error in errors), errors


@pytest.mark.parametrize("defect", ["node", "status"])
def test_fence_failed_compensation_must_handoff_to_reachable_same_node_support(
    defect: str,
) -> None:
    workflow, incident = d17.support_workflow(), d17.support_incident()
    if defect == "node":
        incident["node_ids"] = ["foreign"]
        expected = "does not name"
    else:
        workflow["status"] = "FAILED"
        expected = "did not run"
    errors = v17.successor_errors(
        workflow,
        incident,
        node=d17.NODE,
        predecessor_request_id=d17.fenced_workflow()["request_id"],
        forbidden_escalations={},
        compensation_failed=True,
    )
    assert any(expected in error for error in errors), errors


def test_fence_generation_must_be_an_integer_before_comparing_incarnations() -> None:
    before, after = d17.agent_before(), d17.agent_after()
    after["generation"] = str(after["generation"])
    errors = v17.agent_errors(before, after, node=d17.NODE)
    assert len(errors) == 1 and "not an integer" in errors[0], errors


@pytest.mark.parametrize("defect", ["workflow-status", "step-status"])
def test_lifetime_failure_requires_failed_workflow_and_failed_deadline_step(
    defect: str,
) -> None:
    workflow = d18.happy_workflow()
    if defect == "workflow-status":
        workflow["status"] = "RUNNING"
        expected = "workflow status is not FAILED"
    else:
        workflow["step_executions"][3]["status"] = "WAITING"
        expected = "lifetime-exceeded step execution is not FAILED"
    errors = v18.workflow_errors(workflow, d18.happy_incident(), node=d18.NODE)
    assert any(expected in error for error in errors), errors


@pytest.mark.parametrize(
    ("taints", "status", "settled"),
    [
        ([], "RUNNING", False),
        (["unreadable"], None, False),
        ([], "FAILED", True),
        ([{"key": v18.QUARANTINE_TAINT}], "RUNNING", True),
    ],
)
def test_lifetime_quarantine_waits_for_taint_or_terminal_support(
    taints: list[Any], status: str | None, settled: bool
) -> None:
    node = {"taints": taints}
    workflow = None if status is None else {"status": status}
    assert v18.quarantine_settled(node, workflow) is settled, (node, workflow)


@pytest.mark.parametrize("defect", ["xid", "incident", "workflow"])
def test_lifetime_absorption_cannot_hide_a_new_recovery(defect: str) -> None:
    snapshot = d18.happy_absorb()
    if defect == "xid":
        snapshot["event"]["xid"] = 46
        expected = "second event is not XID 79"
    elif defect == "incident":
        snapshot["incident"] = {}
        expected = "no incident record"
    else:
        snapshot["workflow"]["request_id"] = "new-executable-workflow"
        expected = "second XID opened workflow"
    errors = v18.absorb_errors(
        snapshot,
        incident_id="inc-1",
        workflow_request_id=d18.WORKFLOW_ID,
        official_step_count=len(d18.happy_workflow()["official_steps"]),
    )
    assert any(expected in error for error in errors), errors


@pytest.mark.parametrize(
    ("created_at", "node", "status", "known", "expected"),
    [
        ("2026-09-12T11:59:59+00:00", "target", "RUNNING", False, False),
        ("2026-09-12T12:00:00+00:00", "target", "PENDING", False, True),
        ("unparseable", "target", "RUNNING", False, True),
        (None, "target", "RUNNING", False, True),
        ("2026-09-12T12:00:00+00:00", "foreign", "RUNNING", False, False),
        ("2026-09-12T12:00:00+00:00", "target", "FAILED", False, False),
        ("2026-09-12T12:00:00+00:00", "target", "RUNNING", True, False),
    ],
)
def test_lifetime_residual_scope_keeps_unknown_time_and_excludes_proven_unrelated_work(
    created_at: str | None, node: str, status: str, known: bool, expected: bool
) -> None:
    workflow = {
        "request_id": "candidate",
        "created_at": created_at,
        "node_ids": [node],
        "status": status,
    }
    errors = v18.new_executable_workflow_errors(
        [workflow],
        known_request_ids={"candidate"} if known else set(),
        started_after=datetime(2026, 9, 12, 12, tzinfo=timezone.utc),
        node="target",
    )
    assert bool(errors) is expected, (workflow, errors)
    if expected:
        assert errors == [
            "the drill left executable workflows behind: ['candidate']"
        ], errors


@pytest.mark.parametrize("state", ["FAILED", "RUNNING"])
def test_straddling_verify_must_be_terminal_without_becoming_control_plane_success(
    state: str,
) -> None:
    row = {
        "command_id": "verify-owned",
        "operation": v18.WAITING_STEP,
        "state": state,
        "started_at": (d18.T_CANCEL - timedelta(seconds=1)).isoformat(),
        "completed_at": (d18.T_CANCEL + timedelta(seconds=1)).isoformat(),
    }
    command = {
        "command_id": row["command_id"],
        "step": {"operation": row["operation"]},
        "status": "FAILED",
        "status_source": "workflow-timeout",
    }
    assert command["status_source"] in v18.CANCELLATION_STATUS_SOURCES, command
    errors = v18.straddling_row_errors(
        [row],
        t_cancel=d18.T_CANCEL,
        baseline_command_ids=set(),
        kernel_journal=d18.EMPTY_JOURNAL,
        commands=[command],
    )
    if state == "FAILED":
        assert errors == [], errors
    else:
        assert len(errors) == 1 and "never reached a terminal state" in errors[0], (
            errors
        )


def test_straddling_receipt_needs_identity_and_complete_timestamps() -> None:
    row = {"operation": v18.WAITING_STEP, "started_at": None, "completed_at": None}
    assert not v18.ledger_command_matches(
        row, {"step": {"operation": v18.WAITING_STEP}}
    ), row
    assert (
        v18.straddling_rows([row], t_cancel=d18.T_CANCEL, baseline_command_ids=set())
        == []
    ), row


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("bad", None),
        ("2026-09-12T12:00:00", None),
        ("2026-09-12T14:00:00+02:00", datetime(2026, 9, 12, 12, tzinfo=timezone.utc)),
    ],
)
def test_reservation_reclaim_timestamp_parser_never_guesses_missing_or_malformed_time(
    value: str | None, expected: datetime | None
) -> None:
    assert v22.parse_time(value) == expected, value
