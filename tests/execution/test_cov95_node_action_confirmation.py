"""``gpu_fault.execution.node_action_confirmation``: evidence edges.

``tests/admin/test_node_action_confirmation.py`` drives the operator
confirmation end to end. These tests pin the readers underneath it on the
inputs that run turned up nothing for: Kubernetes and fleet evidence with
missing or malformed fields, remote-command payloads that carry no usable
batch, step records that match no planned step, every place a pre-reboot boot
id may be recorded, and the guards of the write itself.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.execution.node_action_confirmation import (
    AgentEvidence,
    KubernetesNodeEvidence,
    NodeActionConfirmationVerdict,
    apply_node_action_confirmation,
    confirm_node_action,
    confirmed_execution,
    node_action_executions,
    node_validation,
    previous_boot_id,
    remote_command_evidence,
    unresolved_node_actions_on,
)
from gpu_fault.execution.node_action_uncertainty import OPERATOR_CONFIRMED_KEY
from gpu_fault.models import (
    BlockedKind,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from tests._builders import (
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

NOW = datetime(2026, 9, 19, 8, 0, tzinfo=timezone.utc)
STARTED = NOW - timedelta(hours=2)
NODE = "hyperpod-i-0246"
ACTOR = "arn:aws:sts::123456789012:assumed-role/Admin/ops"
RESET = WorkflowOperation.RESET_GPU
RESTART = WorkflowOperation.RESTART_NODE
VALIDATE = WorkflowOperation.VALIDATE_GPU
RESTORE = WorkflowOperation.RESTORE_SCHEDULING
UNKNOWN = {"outcome_unknown": True, "manual_confirmation_required": True}


def _reset_workflow(**overrides: Any):
    values: dict[str, Any] = {
        "status": WorkflowStatus.BLOCKED,
        "blocked_kind": BlockedKind.NEEDS_OPERATOR,
        "fencing_token": 4,
        "official_steps": [workflow_step(RESET, node_ids=[NODE], gpu_uuids=["GPU-1"])],
        "step_executions": [
            workflow_step_execution(
                0,
                RESET,
                WorkflowStepStatus.FAILED,
                phase="official",
                adapter_operation_id="remote/cmd-reset",
                details={**UNKNOWN, "remote_command_id": "cmd-reset"},
                started_at=STARTED,
            )
        ],
    }
    values.update(overrides)
    return workflow_request("wf-reset", "inc-xid-79", **values)


def _incident(incident_id: str = "inc-xid-79", **overrides: Any):
    values: dict[str, Any] = {
        "node_ids": [NODE],
        "fencing_token": 4,
        "workflow_request_id": "wf-reset",
    }
    values.update(overrides)
    return fault_incident(incident_id, "event-79", **values)


def _node() -> KubernetesNodeEvidence:
    return KubernetesNodeEvidence.from_mapping(
        {
            "node_id": NODE,
            "uid": "uid-0246",
            "boot_id": "boot-new",
            "ready": True,
            "ready_since": (NOW - timedelta(minutes=50)).isoformat(),
        }
    )


def _agent(**overrides: Any) -> AgentEvidence:
    values: dict[str, Any] = {
        "node_id": NODE,
        "boot_id": "boot-new",
        "agent_incarnation_id": "inc-new",
        "lifecycle_state": "ACTIVE",
        "generation": 9,
        "last_seen_at": (NOW - timedelta(seconds=20)).isoformat(),
        "lease_expires_at": (NOW + timedelta(seconds=40)).isoformat(),
    }
    values.update(overrides)
    return AgentEvidence.from_mapping(values)


def _confirm(workflow, incident=None, *, commands=(), **extra: Any):
    return confirm_node_action(
        workflow=workflow,
        incident=incident or _incident(),
        node_id=NODE,
        node=_node(),
        agent=_agent(),
        remote_commands=list(commands),
        actor=ACTOR,
        reference="CHG-1",
        now=NOW,
        **extra,
    )


def test_kubernetes_evidence_reads_only_the_ready_condition() -> None:
    since = (NOW - timedelta(minutes=5)).isoformat()
    evidence = KubernetesNodeEvidence.from_node(
        NODE,
        {
            "metadata": {"uid": "uid-0246"},
            "spec": {"unschedulable": True},
            "status": {
                "nodeInfo": {"bootID": "boot-new"},
                "conditions": [
                    "Ready",
                    {"type": "MemoryPressure", "status": "True"},
                    {"type": "Ready", "status": "True", "lastTransitionTime": since},
                ],
            },
        },
    )
    assert evidence.as_dict() == {
        "node_id": NODE,
        "exists": True,
        "uid": "uid-0246",
        "boot_id": "boot-new",
        "ready": True,
        "ready_since": since,
        "unschedulable": True,
    }
    assert KubernetesNodeEvidence.from_node(NODE, None).exists is False, (
        "a node the API no longer returns does not exist"
    )


def test_agent_evidence_without_a_record_is_absent() -> None:
    absent = AgentEvidence.from_record(NODE, None)
    assert (absent.node_id, absent.present) == (NODE, False)
    assert absent.as_dict()["last_seen_at"] is None
    assert AgentEvidence.from_mapping({"node_id": NODE, "present": False}) == absent
    with pytest.raises(ValueError, match="agent evidence requires a node id"):
        AgentEvidence.from_mapping(None)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (NOW, NOW),
        (NOW.isoformat(), NOW),
        ("", None),
        (None, None),
        (1_758_268_800, None),
        ("last tuesday", None),
        ("2026-09-19T08:00:00", None),
    ],
    ids=["datetime", "iso", "empty", "none", "epoch", "unparsable", "naive"],
)
def test_agent_timestamps_are_aware_datetimes_or_nothing(
    value: Any, expected: datetime | None
) -> None:
    evidence = AgentEvidence.from_mapping(
        {"node_id": NODE, "last_seen_at": value, "lease_expires_at": value}
    )
    assert (evidence.last_seen_at, evidence.lease_expires_at) == (expected, expected)


def test_remote_command_evidence_keeps_only_well_formed_batched_entries() -> None:
    command = {
        "command_id": "cmd-batch",
        "step_index": 1,
        "status": "FAILED",
        "batched_steps": [
            {"step_index": 1},
            {"step_index": "2"},
            "3",
            {"step_index": 4},
        ],
        "result_details": {
            "node_results": {NODE: {"status": "FAILED"}},
            "workflow": {"request_id": "wf-reset"},
            "batched_results": {
                "1": {"status": "FAILED", "details": {"node_results": {}, "big": 1}},
                "2": "not an entry",
                "4": {"status": "SUCCEEDED", "details": ["not", "details"]},
            },
        },
    }
    evidence = remote_command_evidence(command)
    assert evidence["batched_step_indexes"] == [1, 4]
    assert evidence["result_details"] == {
        "node_results": {NODE: {"status": "FAILED"}},
        "batched_results": {
            "1": {"status": "FAILED", "details": {"node_results": {}}},
            "4": {"status": "SUCCEEDED", "details": {}},
        },
    }, "bodies are dropped; an entry that is not a mapping is skipped"
    flat = remote_command_evidence(
        {
            "command_id": "cmd-flat",
            "batched_step_indexes": [3],
            "result_details": {"batched_results": ["1"]},
        }
    )
    assert flat["batched_step_indexes"] == [3]
    assert flat["result_details"] == {}, "a batch that is not a mapping carries nothing"


def test_a_step_record_is_paired_with_its_planned_step_by_phase_and_index() -> None:
    workflow = _reset_workflow(
        official_steps=[workflow_step(RESTART, node_ids=["other-node"])],
        safety_steps=[workflow_step(RESET, node_ids=[NODE])],
        step_executions=[
            workflow_step_execution(
                0, RESET, WorkflowStepStatus.FAILED, phase="safety", details=UNKNOWN
            ),
            workflow_step_execution(
                5, RESTART, WorkflowStepStatus.FAILED, phase=None, details=UNKNOWN
            ),
        ],
    )
    pairs = node_action_executions(workflow)
    assert [(item.phase, step) for item, step in pairs] == [
        ("safety", workflow.safety_steps[0]),
        (None, None),
    ], "a safety record reads its safety step; an index no list has reads nothing"
    on_node, elsewhere = unresolved_node_actions_on(workflow, NODE)
    assert on_node == [f"step 0 RESET_GPU on {NODE}"]
    assert elsewhere == ["step 5 RESTART_NODE on unknown nodes"]


def _restart_execution(details: dict[str, Any]):
    return workflow_step_execution(
        2,
        RESTART,
        WorkflowStepStatus.FAILED,
        phase="official",
        details={**UNKNOWN, **details},
        started_at=STARTED,
    )


def test_a_pre_reboot_boot_id_is_read_from_the_command_that_carried_the_step() -> None:
    execution = _restart_execution({"agent_baselines": {NODE: "boot-old"}})
    commands = [
        {
            "command_id": "cmd-other-step",
            "step_index": 7,
            "result_details": {"agent_baselines": {NODE: {"boot_id": "boot-wrong"}}},
        },
        {
            "command_id": "cmd-restart",
            "step_index": 2,
            "result_details": {
                "agent_baselines": {
                    NODE: {"boot_id": "boot-old", "agent_incarnation_id": "inc-old"}
                }
            },
        },
    ]
    assert previous_boot_id(execution, _incident(), NODE, _agent(), commands) == (
        "boot-old",
        "remote_command.cmd-restart.agent_baselines",
        "inc-old",
    ), "a malformed step baseline and another step's command are not evidence"


def test_a_pre_reboot_boot_id_is_read_from_a_batched_stop_authorization() -> None:
    execution = _restart_execution({"stop_reboot_authorization_v1": {"nodes": [NODE]}})
    authorization = {"nodes": {NODE: {"boot_id": "boot-auth"}}}
    commands = [
        {
            "command_id": "cmd-other-step",
            "step_index": 9,
            "result_details": {
                "stop_reboot_authorization_v1": {
                    "nodes": {NODE: {"boot_id": "boot-no"}}
                }
            },
        },
        {"command_id": "cmd-empty", "step_index": 2, "result_details": None},
        {
            "command_id": "cmd-batch",
            "step_index": 1,
            "batched_step_indexes": [2],
            "result_details": {
                "batched_results": {
                    "2": {"details": {"stop_reboot_authorization_v1": authorization}}
                }
            },
        },
    ]
    assert previous_boot_id(execution, _incident(), NODE, _agent(), commands) == (
        "boot-auth",
        "remote_command.cmd-batch.batched_results.2.stop_reboot_authorization_v1",
        None,
    )


def test_a_workflow_of_another_incident_is_refused() -> None:
    verdict = _confirm(_reset_workflow(), _incident(incident_id="inc-other"))
    assert verdict.confirmations == ()
    assert (
        "workflow wf-reset belongs to incident inc-xid-79, not inc-other"
        in verdict.refusals
    ), verdict.refusals


def test_the_terminal_answer_is_read_only_from_commands_that_carried_the_step() -> None:
    commands = [
        {
            "command_id": "cmd-other-step",
            "step_index": 3,
            "status": "SUCCEEDED",
            "result_details": {"node_results": {NODE: {"status": "SUCCEEDED"}}},
        },
        {
            "command_id": "cmd-batch",
            "step_index": 1,
            "batched_step_indexes": [0],
            "status": "FAILED",
            "result_details": {
                "batched_results": {
                    "0": {"details": {"node_results": {NODE: {"status": "FAILED"}}}}
                }
            },
        },
    ]
    verdict = _confirm(_reset_workflow(), commands=commands)
    assert verdict.refusals == (), verdict.refusals
    (confirmation,) = verdict.confirmations
    assert confirmation["terminal_node_result"] == {"status": "FAILED"}
    assert confirmation["terminal_node_result_source"] == (
        "remote_command.cmd-batch.batched_results.0.node_results"
    )
    assert confirmation["settled_by"] == "terminal_node_result"


def _validated_restore(request_id: str, validated_at: datetime):
    return workflow_request(
        request_id,
        f"inc-{request_id}",
        status=WorkflowStatus.SUCCEEDED,
        official_steps=[
            workflow_step(VALIDATE, node_ids=[NODE], gpu_uuids=["GPU-1", "GPU-2"]),
            workflow_step(RESTORE, node_ids=[NODE]),
        ],
        completed_operations=[VALIDATE, RESTORE],
        step_executions=[
            workflow_step_execution(
                0, VALIDATE, WorkflowStepStatus.SUCCEEDED, updated_at=validated_at
            )
        ],
    )


def test_the_newest_validated_restore_after_the_action_answers_it() -> None:
    workflow = _reset_workflow()
    execution = workflow.step_executions[0]
    newest = _validated_restore("wf-newest", NOW - timedelta(minutes=5))
    older = _validated_restore("wf-older", NOW - timedelta(minutes=30))
    before = _validated_restore("wf-before", STARTED - timedelta(minutes=1))
    result = node_validation(
        [newest, older, before],
        workflow,
        _incident(),
        NODE,
        execution,
        workflow.official_steps[0],
    )
    assert result is not None, "a later validated restore of the node answers"
    assert result["request_id"] == "wf-newest", result
    assert result["cross_incident"] is True
    assert result["validated_gpu_uuids"] == ["GPU-1", "GPU-2"]
    assert result["validated_at"] == (NOW - timedelta(minutes=5)).isoformat()


def test_a_confirmed_pending_action_reads_operator_confirmed() -> None:
    execution = workflow_step_execution(
        0,
        RESET,
        WorkflowStepStatus.FAILED,
        details={**UNKNOWN, "node_action_state": "PENDING", "reason": "lost"},
    )
    confirmed = confirmed_execution(execution, {"actor": ACTOR}, now=NOW)
    assert confirmed.details["node_action_state"] == "OPERATOR_CONFIRMED"
    assert confirmed.details["outcome_unknown"] is False
    assert confirmed.details["reason"] == "lost", "other details are kept"
    assert confirmed.details[OPERATOR_CONFIRMED_KEY] == {"actor": ACTOR}
    assert confirmed.updated_at == NOW


def test_a_refused_verdict_is_never_written() -> None:
    verdict = NodeActionConfirmationVerdict(
        node_id=NODE, refusals=("node is not Ready", "lease expired")
    )
    with pytest.raises(
        ValueError,
        match="node action confirmation refused: node is not Ready; lease expired",
    ):
        apply_node_action_confirmation(_reset_workflow(), verdict, now=NOW)


def test_a_confirmation_for_a_step_no_longer_on_the_record_is_not_written() -> None:
    workflow = _reset_workflow()
    gone = workflow_step_execution(
        4, RESET, WorkflowStepStatus.FAILED, phase="official", details=UNKNOWN
    )
    verdict = NodeActionConfirmationVerdict(
        node_id=NODE,
        confirmations=({"actor": ACTOR, "reference": "CHG-1"},),
        executions=(gone,),
    )
    with pytest.raises(
        ValueError, match="step 4 RESET_GPU is no longer on the workflow"
    ):
        apply_node_action_confirmation(workflow, verdict, now=NOW)
