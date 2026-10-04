"""Operator confirmation of a node action whose outcome the product never saw.

GF-REGIONAL-DESTR-014 (unknown-reboot) parks its workflow BLOCKED /
NEEDS_OPERATOR with ``outcome_unknown`` on the RESTART_NODE step: the node
really rebooted, but the Node Agent was down while the executor waited, so
nothing could confirm it. ``has_unresolved_node_action`` then fences every
lever (validated restore, incident close, reconcile) and nothing let the
operator record what the fleet and Kubernetes already showed. These tests pin
the confirmation the product now accepts: the evidence it requires, the
refusals when the evidence does not prove the outcome, and that a confirmed
record reads as resolved everywhere -- and stays resolved when remote
receipts are refreshed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.execution.node_action_confirmation import (
    AgentEvidence,
    KubernetesNodeEvidence,
    apply_node_action_confirmation,
    confirm_node_action,
    remote_command_evidence,
)
from gpu_fault.execution.node_action_uncertainty import (
    OPERATOR_CONFIRMED_KEY,
    has_unresolved_node_action,
    node_actions_operator_confirmed,
    operator_confirmation,
    refresh_remote_action_state,
)
from gpu_fault.models import (
    BlockedKind,
    WorkflowEventCode,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.regional import RemoteActionCommand
from gpu_fault.remote_command_models import RemoteCommandStatus
from tests._builders import (
    build_store,
    copy_model,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

NOW = datetime(2026, 9, 19, 8, 0, tzinfo=timezone.utc)
STARTED = NOW - timedelta(hours=2)
NODE = "hyperpod-i-0246"
SIBLING = "hyperpod-i-096b"
OLD_BOOT = "boot-old-1111"
NEW_BOOT = "boot-new-2222"
ACTOR = "arn:aws:sts::123456789012:assumed-role/Admin/ops"
RESTART = WorkflowOperation.RESTART_NODE
RESET = WorkflowOperation.RESET_GPU


def _blocked_workflow(**overrides):
    values = {
        "status": WorkflowStatus.BLOCKED,
        "blocked_kind": BlockedKind.NEEDS_OPERATOR,
        "fencing_token": 4,
        "execution_epoch": 3,
        "official_steps": [
            workflow_step(WorkflowOperation.STOP_WORKLOADS, node_ids=[NODE, SIBLING]),
            workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE, node_ids=[NODE]),
            workflow_step(RESTART, node_ids=[NODE]),
            workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE, node_ids=[SIBLING]),
            workflow_step(RESTART, node_ids=[SIBLING]),
            workflow_step(WorkflowOperation.RESTORE_SCHEDULING, node_ids=[SIBLING]),
        ],
        "completed_step_indexes": [0, 1, 3, 4, 5],
        "completed_operations": [
            WorkflowOperation.STOP_WORKLOADS,
            WorkflowOperation.MARK_UNSCHEDULABLE,
            RESTART,
            WorkflowOperation.RESTORE_SCHEDULING,
        ],
        "step_executions": [
            workflow_step_execution(
                2,
                RESTART,
                WorkflowStepStatus.FAILED,
                phase="official",
                adapter_operation_id="remote/cmd-restart-0246",
                details={
                    "outcome_unknown": True,
                    "manual_confirmation_required": True,
                    "remote_command_id": "cmd-restart-0246",
                    "agent_baselines": {
                        NODE: {"boot_id": OLD_BOOT, "agent_incarnation_id": "inc-old"}
                    },
                },
                started_at=STARTED,
                updated_at=STARTED + timedelta(minutes=10),
            ),
            workflow_step_execution(
                4,
                RESTART,
                WorkflowStepStatus.SUCCEEDED,
                phase="official",
                adapter_operation_id="remote/cmd-restart-096b",
                started_at=STARTED,
                updated_at=STARTED + timedelta(minutes=5),
            ),
        ],
        "execution_deadline": NOW - timedelta(hours=1),
        "lifetime_deadline_at": NOW - timedelta(minutes=30),
    }
    values.update(overrides)
    return workflow_request("wf-2a1858b1", "inc-xid-79", **values)


def _incident(**overrides):
    values = {
        "node_ids": [NODE, SIBLING],
        "fencing_token": 4,
        "workflow_request_id": "wf-2a1858b1",
    }
    values.update(overrides)
    return fault_incident("inc-xid-79", "event-79", **values)


def _node(**overrides) -> KubernetesNodeEvidence:
    values = {
        "node_id": NODE,
        "exists": True,
        "uid": "uid-0246",
        "boot_id": NEW_BOOT,
        "ready": True,
        "ready_since": (NOW - timedelta(minutes=50)).isoformat(),
        "unschedulable": False,
    }
    values.update(overrides)
    return KubernetesNodeEvidence.from_mapping(values)


def _agent(**overrides) -> AgentEvidence:
    values = {
        "node_id": NODE,
        "present": True,
        "boot_id": NEW_BOOT,
        "agent_incarnation_id": "inc-new",
        "lifecycle_state": "ACTIVE",
        "generation": 9,
        "last_seen_at": (NOW - timedelta(seconds=20)).isoformat(),
        "lease_expires_at": (NOW + timedelta(seconds=40)).isoformat(),
        "retired_incarnation_ids": ["inc-old"],
    }
    values.update(overrides)
    return AgentEvidence.from_mapping(values)


def _confirm(workflow=None, incident=None, *, node=None, agent=None, commands=()):
    return confirm_node_action(
        workflow=workflow or _blocked_workflow(),
        incident=incident or _incident(),
        node_id=NODE,
        node=node or _node(),
        agent=agent or _agent(),
        remote_commands=list(commands),
        actor=ACTOR,
        reference="CHG-2026-0919-01",
        now=NOW,
    )


def test_a_rebooted_node_with_an_active_agent_confirms_the_restart_outcome() -> None:
    verdict = _confirm()

    assert verdict.refusals == (), verdict.refusals
    assert len(verdict.confirmations) == 1
    confirmation = verdict.confirmations[0]
    assert confirmation["node_id"] == NODE
    assert confirmation["operation"] == "RESTART_NODE"
    assert confirmation["step_index"] == 2
    assert confirmation["previous_boot_id"] == OLD_BOOT
    assert confirmation["previous_boot_id_source"] == "step.agent_baselines"
    assert confirmation["observed_boot_id"] == NEW_BOOT
    assert confirmation["agent_generation"] == 9
    assert confirmation["remote_command_id"] == "cmd-restart-0246"
    assert confirmation["actor"] == ACTOR
    assert confirmation["reference"] == "CHG-2026-0919-01"
    assert confirmation["confirmed_at"] == NOW.isoformat()
    assert confirmation["superseded_flags"] == {
        "outcome_unknown": True,
        "manual_confirmation_required": True,
    }


def test_the_confirmation_resolves_the_step_and_is_audited_on_the_workflow() -> None:
    workflow = _blocked_workflow()
    assert has_unresolved_node_action(workflow), "the fixture must start unresolved"
    verdict = _confirm(workflow)

    updated = apply_node_action_confirmation(workflow, verdict, now=NOW)

    assert not has_unresolved_node_action(updated), "confirmation resolves the step"
    assert node_actions_operator_confirmed(updated), "the record reads as confirmed"
    execution = next(item for item in updated.step_executions if item.step_index == 2)
    assert execution.status is WorkflowStepStatus.FAILED, (
        "the step still failed at its cap; only its outcome is now known"
    )
    assert execution.details["outcome_unknown"] is False
    assert execution.details["manual_confirmation_required"] is False
    assert operator_confirmation(execution.details) == verdict.confirmations[0]
    assert updated.status is WorkflowStatus.BLOCKED, (
        "confirmation reconciles the uncertainty; closing the record is the "
        "reconcile's or the incident close's job"
    )
    event = updated.events[-1]
    assert event.kind is WorkflowEventKind.OPERATOR_RECONCILED
    assert event.code == WorkflowEventCode.NODE_ACTION_CONFIRMED.value
    assert event.actor == ACTOR
    assert event.step_index == 2 and event.operation is RESTART
    assert event.details["reference"] == "CHG-2026-0919-01"
    assert event.details["node_id"] == NODE
    assert event.details["observed_boot_id"] == NEW_BOOT
    assert event.details["previous_boot_id"] == OLD_BOOT
    assert len(updated.step_executions) == len(workflow.step_executions), (
        "the record is replaced in place, not appended"
    )


def test_an_already_confirmed_node_is_reported_as_such_not_confirmed_twice() -> None:
    workflow = _blocked_workflow()
    confirmed = apply_node_action_confirmation(workflow, _confirm(workflow), now=NOW)

    verdict = _confirm(confirmed)

    assert verdict.refusals == ()
    assert verdict.confirmations == ()
    assert len(verdict.already_confirmed) == 1
    assert verdict.already_confirmed[0]["reference"] == "CHG-2026-0919-01"


def test_a_node_the_workflow_never_left_unresolved_is_refused() -> None:
    verdict = confirm_node_action(
        workflow=_blocked_workflow(),
        incident=_incident(),
        node_id=SIBLING,
        node=_node(node_id=SIBLING),
        agent=_agent(node_id=SIBLING),
        remote_commands=[],
        actor=ACTOR,
        reference="CHG-1",
        now=NOW,
    )

    assert verdict.confirmations == ()
    assert any(
        "no unresolved node action on node hyperpod-i-096b" in item
        and "step 2 RESTART_NODE on hyperpod-i-0246" in item
        for item in verdict.refusals
    ), verdict.refusals


def test_a_node_outside_the_incident_is_refused() -> None:
    verdict = confirm_node_action(
        workflow=_blocked_workflow(),
        incident=_incident(),
        node_id="hyperpod-i-else",
        node=_node(node_id="hyperpod-i-else"),
        agent=_agent(node_id="hyperpod-i-else"),
        remote_commands=[],
        actor=ACTOR,
        reference="CHG-1",
        now=NOW,
    )

    assert any("is not named by incident inc-xid-79" in r for r in verdict.refusals), (
        verdict.refusals
    )


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"status": WorkflowStatus.FAILED}, "is FAILED, not BLOCKED"),
        ({"execution_owner_id": "executor-1"}, "still has an execution owner"),
        (
            {"execution_lease_expires_at": NOW + timedelta(seconds=30)},
            "execution lease has not expired",
        ),
    ],
)
def test_a_record_an_executor_may_still_drive_is_refused(changes, expected) -> None:
    verdict = _confirm(_blocked_workflow(**changes))

    assert verdict.confirmations == ()
    assert any(expected in item for item in verdict.refusals), verdict.refusals


def test_a_waiting_sibling_step_refuses_the_confirmation() -> None:
    workflow = _blocked_workflow()
    waiting = workflow_step_execution(
        4, RESTART, WorkflowStepStatus.WAITING, phase="official"
    )
    workflow = copy_model(
        workflow, step_executions=[*workflow.step_executions[:1], waiting]
    )

    verdict = _confirm(workflow)

    assert any("WAITING" in item for item in verdict.refusals), verdict.refusals


def test_an_open_remote_command_of_the_workflow_refuses_the_confirmation() -> None:
    command = {
        "command_id": "cmd-late",
        "step_index": 2,
        "status": "LEASED",
        "result_details": {},
    }

    verdict = _confirm(commands=[command])

    assert any(
        "open remote command cmd-late (LEASED)" in item for item in verdict.refusals
    ), verdict.refusals


def test_the_deadline_of_a_parked_record_is_not_a_refusal() -> None:
    """The BLOCKED record's deadlines passed long ago; that is what parked it."""

    workflow = _blocked_workflow(
        execution_deadline=NOW - timedelta(days=3),
        lifetime_deadline_at=NOW - timedelta(days=3),
    )

    assert _confirm(workflow).refusals == ()


@pytest.mark.parametrize(
    ("agent_changes", "expected"),
    [
        ({"present": False}, "has no Node Agent record"),
        ({"lifecycle_state": "DRAINING"}, "Node Agent lifecycle state is DRAINING"),
        (
            {"lease_expires_at": (NOW - timedelta(seconds=1)).isoformat()},
            "Node Agent lease expired",
        ),
        (
            {"last_seen_at": (STARTED - timedelta(minutes=1)).isoformat()},
            "not been seen since RESTART_NODE started",
        ),
        ({"boot_id": None}, "Node Agent record carries no boot id"),
        ({"boot_id": OLD_BOOT}, "still runs boot boot-old-1111"),
        (
            {"boot_id": "boot-other-3333"},
            "kubelet reports boot boot-new-2222 but the Node Agent reports "
            "boot-other-3333",
        ),
        ({"agent_incarnation_id": "inc-old"}, "same Node Agent incarnation"),
    ],
)
def test_agent_evidence_that_does_not_prove_the_reboot_is_refused(
    agent_changes, expected
) -> None:
    verdict = _confirm(agent=_agent(**agent_changes))

    assert verdict.confirmations == ()
    assert any(expected in item for item in verdict.refusals), verdict.refusals


@pytest.mark.parametrize(
    ("node_changes", "expected"),
    [
        ({"exists": False}, "is not in the cluster"),
        ({"ready": False}, "is not Ready"),
        ({"uid": None}, "node identity is incomplete"),
        ({"boot_id": None}, "kubelet reports no boot id"),
        ({"boot_id": OLD_BOOT}, "still runs boot boot-old-1111"),
    ],
)
def test_kubernetes_evidence_that_does_not_prove_the_reboot_is_refused(
    node_changes, expected
) -> None:
    verdict = _confirm(node=_node(**node_changes))

    assert verdict.confirmations == ()
    assert any(expected in item for item in verdict.refusals), verdict.refusals


def _without_baseline():
    workflow = _blocked_workflow()
    execution = workflow.step_executions[0]
    details = {
        key: value
        for key, value in execution.details.items()
        if key != "agent_baselines"
    }
    return copy_model(
        workflow,
        step_executions=[
            copy_model(execution, details=details),
            *workflow.step_executions[1:],
        ],
    )


def test_without_any_recorded_pre_reboot_boot_id_the_confirmation_is_refused() -> None:
    verdict = _confirm(_without_baseline(), _incident(source_boot_id=None))

    assert verdict.confirmations == ()
    assert any(
        "no pre-reboot boot id is recorded for node hyperpod-i-0246" in item
        for item in verdict.refusals
    ), verdict.refusals


def test_the_remote_command_baseline_serves_when_the_step_lost_it() -> None:
    command = {
        "command_id": "cmd-restart-0246",
        "step_index": 0,
        "batched_step_indexes": [1, 2],
        "status": "FAILED",
        "result_details": {
            "batched_results": {
                "2": {
                    "status": "FAILED",
                    "details": {"agent_baselines": {NODE: {"boot_id": OLD_BOOT}}},
                }
            }
        },
    }

    verdict = _confirm(_without_baseline(), commands=[command])

    assert verdict.refusals == ()
    assert verdict.confirmations[0]["previous_boot_id"] == OLD_BOOT
    assert (
        verdict.confirmations[0]["previous_boot_id_source"]
        == "remote_command.cmd-restart-0246.batched_results.2.agent_baselines"
    )


def test_the_stop_reboot_authorization_serves_as_a_baseline() -> None:
    workflow = _without_baseline()
    execution = workflow.step_executions[0]
    workflow = copy_model(
        workflow,
        step_executions=[
            copy_model(
                execution,
                details={
                    **execution.details,
                    "stop_reboot_authorization_v1": {
                        "nodes": {NODE: {"uid": "uid-0246", "boot_id": OLD_BOOT}}
                    },
                },
            ),
            *workflow.step_executions[1:],
        ],
    )

    verdict = _confirm(workflow)

    assert verdict.refusals == ()
    assert (
        verdict.confirmations[0]["previous_boot_id_source"]
        == "step.stop_reboot_authorization_v1"
    )


def test_a_single_node_incidents_source_boot_id_serves_as_a_baseline() -> None:
    workflow = copy_model(
        _without_baseline(),
        official_steps=[workflow_step(RESTART, node_ids=[NODE])] * 6,
    )
    incident = _incident(node_ids=[NODE], source_boot_id=OLD_BOOT)

    verdict = _confirm(workflow, incident)

    assert verdict.refusals == ()
    assert (
        verdict.confirmations[0]["previous_boot_id_source"] == "incident.source_boot_id"
    )


def test_a_multi_node_incidents_source_boot_id_needs_the_agent_to_have_retired_it() -> (
    None
):
    incident = _incident(source_boot_id=OLD_BOOT)

    refused = _confirm(_without_baseline(), incident)
    assert refused.confirmations == (), refused.refusals

    accepted = _confirm(
        _without_baseline(),
        incident,
        agent=_agent(retired_incarnation_ids=["inc-old", OLD_BOOT]),
    )
    assert accepted.refusals == ()
    assert accepted.confirmations[0]["previous_boot_id_source"] == (
        "incident.source_boot_id"
    )


def test_a_reset_gpu_needs_a_terminal_node_agent_result_on_the_command() -> None:
    workflow = _blocked_workflow(
        official_steps=[workflow_step(RESET, node_ids=[NODE], gpu_uuids=["GPU-1"])],
        step_executions=[
            workflow_step_execution(
                0,
                RESET,
                WorkflowStepStatus.FAILED,
                phase="official",
                adapter_operation_id="remote/cmd-reset",
                details={
                    "outcome_unknown": True,
                    "manual_confirmation_required": True,
                    "remote_command_id": "cmd-reset",
                },
                started_at=STARTED,
            )
        ],
        completed_step_indexes=[],
        completed_operations=[],
    )

    refused = _confirm(workflow, commands=[])
    assert any(
        "no terminal Node Agent result for RESET_GPU on node hyperpod-i-0246" in item
        for item in refused.refusals
    ), refused.refusals

    interrupted = {
        "command_id": "cmd-reset",
        "step_index": 0,
        "status": "FAILED",
        "result_details": {"node_results": {NODE: {"status": "INTERRUPTED"}}},
    }
    assert _confirm(workflow, commands=[interrupted]).confirmations == ()

    terminal = {
        "command_id": "cmd-reset",
        "step_index": 0,
        "status": "FAILED",
        "result_details": {
            "node_results": {
                NODE: {"status": "SUCCEEDED", "command_id": "cmd-reset", "gpu": "GPU-1"}
            }
        },
    }
    verdict = _confirm(workflow, commands=[terminal])
    assert verdict.refusals == ()
    assert verdict.confirmations[0]["terminal_node_result"] == {
        "status": "SUCCEEDED",
        "command_id": "cmd-reset",
        "gpu": "GPU-1",
    }
    assert verdict.confirmations[0]["previous_boot_id"] is None


def _refused_reset_workflow():
    """HA-004 attempt 3: the agent refused the RESET_GPU before touching the
    GPU (STOP_OWNERSHIP_UNVERIFIABLE), the executor read the refusal as
    outcome-unknown, RESTORE_GPU_SERVICES was withheld, the record parked."""

    return _blocked_workflow(
        fencing_token=1,
        official_steps=[
            workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE, node_ids=[NODE]),
            workflow_step(WorkflowOperation.QUIESCE_GPU_SERVICES, node_ids=[NODE]),
            workflow_step(RESET, node_ids=[NODE], gpu_uuids=["GPU-7"]),
            workflow_step(WorkflowOperation.RESTORE_GPU_SERVICES, node_ids=[NODE]),
        ],
        completed_step_indexes=[0, 1],
        completed_operations=[
            WorkflowOperation.MARK_UNSCHEDULABLE,
            WorkflowOperation.QUIESCE_GPU_SERVICES,
        ],
        step_executions=[
            workflow_step_execution(
                2,
                RESET,
                WorkflowStepStatus.FAILED,
                phase="official",
                adapter_operation_id="remote/cmd-reset-7",
                details={
                    "outcome_unknown": True,
                    "manual_confirmation_required": True,
                    "reason": "STOP_OWNERSHIP_UNVERIFIABLE",
                    "remote_command_id": "cmd-reset-7",
                },
                started_at=STARTED,
            ),
            workflow_step_execution(
                3,
                WorkflowOperation.RESTORE_GPU_SERVICES,
                WorkflowStepStatus.FAILED,
                phase="official",
                details={
                    "outcome_unknown": True,
                    "manual_confirmation_required": True,
                    "restore_gpu_services_withheld": True,
                    "reason": "NODE_ACTION_OUTCOME_UNRESOLVED",
                },
                started_at=STARTED,
            ),
        ],
    )


def _validating_successor(**overrides):
    values = {
        "status": WorkflowStatus.SUCCEEDED,
        "fencing_token": 1,
        "official_action": "RESTORE_SCHEDULING",
        "official_steps": [
            workflow_step(
                WorkflowOperation.VALIDATE_GPU, node_ids=[NODE], gpu_uuids=["GPU-7"]
            ),
            workflow_step(WorkflowOperation.VALIDATE_HOST, node_ids=[NODE]),
            workflow_step(WorkflowOperation.VALIDATE_FABRIC, node_ids=[NODE]),
            workflow_step(WorkflowOperation.RESTORE_SCHEDULING, node_ids=[NODE]),
        ],
        "completed_step_indexes": [0, 1, 2, 3],
        "completed_operations": [
            WorkflowOperation.VALIDATE_GPU,
            WorkflowOperation.VALIDATE_HOST,
            WorkflowOperation.VALIDATE_FABRIC,
            WorkflowOperation.RESTORE_SCHEDULING,
        ],
        "updated_at": NOW - timedelta(minutes=20),
    }
    values.update(overrides)
    return workflow_request("workflow-validated-restore-e076", "inc-xid-79", **values)


def test_a_refused_reset_is_confirmed_by_the_agents_terminal_answer_on_the_command() -> (
    None
):
    command = {
        "command_id": "cmd-reset-7",
        "step_index": 2,
        "status": "FAILED",
        "result_details": {
            "node_results": {
                NODE: {
                    "status": "FAILED",
                    "error": "ResetProgressError: STOP_OWNERSHIP_UNVERIFIABLE",
                }
            }
        },
    }

    verdict = _confirm(
        _refused_reset_workflow(), _incident(fencing_token=1), commands=[command]
    )

    assert verdict.refusals == (), verdict.refusals
    [confirmation] = verdict.confirmations
    assert confirmation["operation"] == "RESET_GPU"
    assert confirmation["terminal_node_result"]["error"].endswith(
        "STOP_OWNERSHIP_UNVERIFIABLE"
    ), "the agent's own refusal is the recorded terminal answer"
    assert confirmation["terminal_node_result_source"] == (
        "remote_command.cmd-reset-7.node_results"
    )
    assert confirmation["validated_by_workflow"] is None
    confirmed = apply_node_action_confirmation(
        _refused_reset_workflow(), verdict, now=NOW
    )
    assert not has_unresolved_node_action(confirmed), (
        "the withheld RESTORE_GPU_SERVICES never counted; the reset now answers"
    )


def test_a_refused_reset_is_confirmed_by_the_later_validated_restore_of_its_node() -> (
    None
):
    workflow = _refused_reset_workflow()
    successor = _validating_successor()
    incident = _incident(
        fencing_token=1, state="RECOVERED", workflow_request_id=successor.request_id
    )

    verdict = confirm_node_action(
        workflow=workflow,
        incident=incident,
        node_id=NODE,
        node=_node(),
        agent=_agent(),
        remote_commands=[],
        actor=ACTOR,
        reference="CHG-HA-004",
        now=NOW,
        successor=successor,
    )

    assert verdict.refusals == (), verdict.refusals
    [confirmation] = verdict.confirmations
    assert confirmation["terminal_node_result"] is None
    assert confirmation["validated_by_workflow"] == {
        "request_id": successor.request_id,
        "status": "SUCCEEDED",
        "completed_operations": [
            "VALIDATE_GPU",
            "VALIDATE_HOST",
            "VALIDATE_FABRIC",
            "RESTORE_SCHEDULING",
        ],
        "validated_gpu_uuids": ["GPU-7"],
        "updated_at": successor.updated_at.isoformat(),
    }


@pytest.mark.parametrize(
    ("successor_changes", "incident_changes"),
    [
        ({"status": WorkflowStatus.FAILED}, {}),
        ({"fencing_token": 2}, {}),
        ({"completed_operations": [WorkflowOperation.RESTORE_SCHEDULING]}, {}),
        (
            {
                "official_steps": [
                    workflow_step(WorkflowOperation.VALIDATE_GPU, node_ids=[SIBLING]),
                    workflow_step(
                        WorkflowOperation.RESTORE_SCHEDULING, node_ids=[SIBLING]
                    ),
                ]
            },
            {},
        ),
        ({}, {"state": "QUARANTINED"}),
        ({}, {"workflow_request_id": "workflow-elsewhere"}),
    ],
)
def test_a_successor_that_did_not_validate_the_node_is_no_evidence(
    successor_changes, incident_changes
) -> None:
    successor = _validating_successor(**successor_changes)
    incident_values = {
        "fencing_token": 1,
        "state": "RECOVERED",
        "workflow_request_id": successor.request_id,
    }
    incident_values.update(incident_changes)
    incident = _incident(**incident_values)

    verdict = confirm_node_action(
        workflow=_refused_reset_workflow(),
        incident=incident,
        node_id=NODE,
        node=_node(),
        agent=_agent(),
        remote_commands=[],
        actor=ACTOR,
        reference="CHG-HA-004",
        now=NOW,
        successor=successor,
    )

    assert verdict.confirmations == ()
    assert any("no later validated restore" in item for item in verdict.refusals), (
        verdict.refusals
    )


def test_remote_command_evidence_keeps_only_what_the_verdict_reads() -> None:
    dumped = {
        "command_id": "cmd-1",
        "step_index": 2,
        "status": "FAILED",
        "status_source": "executor-execution-timeout-outcome-unknown",
        "fencing_token": 4,
        "lease_token": "secret",
        "workflow": {"request_id": "wf"},
        "incident": {"incident_id": "inc"},
        "batched_steps": [{"step_index": 3, "step": {}}],
        "result_details": {
            "agent_baselines": {NODE: {"boot_id": OLD_BOOT}},
            "node_results": {NODE: {"status": "SUCCEEDED"}},
            "huge_payload": "x" * 10000,
            "batched_results": {
                "3": {
                    "status": "SUCCEEDED",
                    "details": {
                        "node_results": {SIBLING: {"status": "SUCCEEDED"}},
                        "big": 1,
                    },
                }
            },
        },
    }

    compact = remote_command_evidence(dumped)

    assert compact == {
        "command_id": "cmd-1",
        "step_index": 2,
        "batched_step_indexes": [3],
        "status": "FAILED",
        "status_source": "executor-execution-timeout-outcome-unknown",
        "fencing_token": 4,
        "result_details": {
            "agent_baselines": {NODE: {"boot_id": OLD_BOOT}},
            "node_results": {NODE: {"status": "SUCCEEDED"}},
            "batched_results": {
                "3": {
                    "status": "SUCCEEDED",
                    "details": {"node_results": {SIBLING: {"status": "SUCCEEDED"}}},
                }
            },
        },
    }


# ------------------------------------------------------ reader-side contract


def test_operator_confirmation_requires_the_attributing_fields() -> None:
    assert operator_confirmation({}) is None
    assert operator_confirmation({OPERATOR_CONFIRMED_KEY: "yes"}) is None
    assert (
        operator_confirmation(
            {OPERATOR_CONFIRMED_KEY: {"actor": ACTOR, "reference": "CHG"}}
        )
        is None
    ), "a confirmation without its timestamp, node and operation is not one"
    complete = {
        "actor": ACTOR,
        "reference": "CHG-1",
        "confirmed_at": NOW.isoformat(),
        "node_id": NODE,
        "operation": "RESTART_NODE",
    }
    assert operator_confirmation({OPERATOR_CONFIRMED_KEY: complete}) == complete


def test_node_actions_operator_confirmed_needs_every_action_resolved() -> None:
    workflow = _blocked_workflow()
    assert not node_actions_operator_confirmed(workflow), "nothing confirmed yet"

    confirmed = apply_node_action_confirmation(workflow, _confirm(workflow), now=NOW)
    assert node_actions_operator_confirmed(confirmed), "the only action is confirmed"

    # A second unresolved action (a different command) reopens the question.
    reopened = copy_model(
        confirmed,
        step_executions=[
            *confirmed.step_executions,
            workflow_step_execution(
                4,
                RESTART,
                WorkflowStepStatus.FAILED,
                phase="official",
                adapter_operation_id="remote/cmd-restart-096b-2",
                details={"outcome_unknown": True},
            ),
        ],
    )
    assert not node_actions_operator_confirmed(reopened), "one action is unresolved"
    assert has_unresolved_node_action(reopened), "the new record is unresolved"

    # Not BLOCKED: the predicate is about parked records only.
    assert not node_actions_operator_confirmed(
        copy_model(confirmed, status=WorkflowStatus.FAILED)
    ), "a FAILED record is not a parked one"


def _remote_command(store, workflow, incident, *, command_id: str, step_index: int):
    command = RemoteActionCommand(
        command_id=command_id,
        cluster_id=incident.cluster_id,
        workflow_request_id=workflow.request_id,
        incident_id=incident.incident_id,
        step_index=step_index,
        fencing_token=workflow.fencing_token,
        idempotency_key=f"{workflow.request_id}/{step_index}/RESTART_NODE",
        step=workflow.official_steps[step_index],
        workflow=workflow,
        incident=incident,
        status=RemoteCommandStatus.FAILED,
        status_source="executor-execution-timeout-outcome-unknown",
        result_details={
            "outcome_unknown": True,
            "manual_confirmation_required": True,
            "agent_baselines": {NODE: {"boot_id": OLD_BOOT}},
        },
        created_at=STARTED,
        updated_at=STARTED + timedelta(minutes=10),
    )
    return store.ensure_remote_command(command)


def test_refreshing_remote_receipts_does_not_undo_an_operator_confirmation() -> None:
    store = build_store()
    incident = _incident()
    workflow = _blocked_workflow()
    store.save_incident(incident)
    store.save_workflow(workflow)
    _remote_command(
        store, workflow, incident, command_id="cmd-restart-0246", step_index=2
    )
    assert has_unresolved_node_action(refresh_remote_action_state(store, workflow)), (
        "the fixture's remote receipt must read as unresolved before confirmation"
    )
    confirmed = apply_node_action_confirmation(workflow, _confirm(workflow), now=NOW)
    store.save_workflow(confirmed, expected=workflow)

    refreshed = refresh_remote_action_state(store, confirmed)

    assert not has_unresolved_node_action(refreshed), "the receipt cannot unconfirm"
    execution = next(item for item in refreshed.step_executions if item.step_index == 2)
    assert operator_confirmation(execution.details) is not None
    assert execution.details["outcome_unknown"] is False


def test_a_different_unresolved_command_on_the_confirmed_step_is_flagged_again() -> (
    None
):
    store = build_store()
    incident = _incident()
    workflow = _blocked_workflow()
    store.save_incident(incident)
    store.save_workflow(workflow)
    confirmed = apply_node_action_confirmation(workflow, _confirm(workflow), now=NOW)
    store.save_workflow(confirmed, expected=workflow)
    _remote_command(
        store, workflow, incident, command_id="cmd-restart-0246-b", step_index=2
    )

    refreshed = refresh_remote_action_state(store, confirmed)

    assert has_unresolved_node_action(refreshed), (
        "an operator confirmed one command's outcome, not a later command's"
    )
    assert len(refreshed.step_executions) == len(confirmed.step_executions) + 1
