"""Regional time bounds do not turn an unresolved mutation into a safe failure."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.execution.node_action_uncertainty import has_unresolved_node_action
from gpu_fault.models import (
    BlockedKind,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
    workflow_is_open,
)
from gpu_fault.regional import (
    RegionalClusterRegistration,
    RegionalRemoteWorkflowAdapter,
    RemoteActionCommand,
    RemoteCommandResult,
    RemoteCommandStatus,
)
from gpu_fault.regional_compatibility import CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION
from gpu_fault.remote_command_models import BATCHED_RESULTS_KEY
from gpu_fault.remote_step_batching import RemoteStepBatchingPolicy
from tests._builders import active_workflow_executor, workflow_step_execution
from tests.execution._cov95_runtime_workflows import FlowHarness

OP = WorkflowOperation
UNKNOWN_RESET = {
    "node_action_state": "PENDING",
    "node_action_command_id": "agent/reset",
    "node_action_accepted_nodes": ["node-a"],
}
PENDING_REBOOT = {
    "action": "REBOOT",
    "provider_mutation_submitted": True,
    "requires_external_confirmation": True,
    "provider_operation_id": "provider/reboot",
}


def remote_flow(operation: OP = OP.RESET_GPU, *, batched: bool = False) -> FlowHarness:
    operations = (
        [OP.QUIESCE_GPU_SERVICES, operation, OP.RESTORE_GPU_SERVICES]
        if operation is OP.RESET_GPU
        else [OP.QUARANTINE, operation, OP.VALIDATE_GPU, OP.RESTORE_SCHEDULING]
    )
    flow = FlowHarness(operations)
    flow.store.save_regional_cluster(
        RegionalClusterRegistration(
            cluster_id="cluster-a",
            region="us-east-1",
            hyperpod_cluster_name="cluster-a",
            eks_cluster_arn="arn:aws:eks:us-east-1:000000000000:cluster/test",
            token_sha256="0" * 64,
            allowed_namespaces=["training"],
            agent_endpoint_allowed_cidrs=["192.0.2.0/24"],
        )
    )
    flow.amend(
        completed_step_indexes=[0],
        completed_operations=[operations[0]],
        step_executions=[workflow_step_execution(0, operations[0], phase="official")],
    )
    flow.executor = active_workflow_executor(
        flow.store,
        [
            RegionalRemoteWorkflowAdapter(
                flow.store,
                owners={"owner-a"},
                step_batching=RemoteStepBatchingPolicy(
                    batched, CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION
                ),
            )
        ],
        operations,
    )
    flow.execute()
    return flow


def claim(flow: FlowHarness) -> RemoteActionCommand:
    return flow.store.claim_remote_commands(
        "cluster-a",
        "cluster-executor",
        limit=1,
        lease_seconds=180,
        executor_protocol_version=CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION,
    )[0]


def report(
    flow: FlowHarness,
    command: RemoteActionCommand,
    details: dict[str, Any],
    *,
    status: RemoteCommandStatus = RemoteCommandStatus.WAITING,
) -> RemoteActionCommand:
    if command.batched_steps:
        indexes = (
            command.covered_step_indexes
            if status is RemoteCommandStatus.SUCCEEDED
            else (command.step_index,)
        )
        details = {
            BATCHED_RESULTS_KEY: {
                str(index): {
                    "status": status.value,
                    "details": details if index == command.step_index else {},
                    "error": "confirmed rejection"
                    if status is RemoteCommandStatus.FAILED
                    else None,
                }
                for index in indexes
            }
        }
    return flow.store.complete_remote_command(
        "cluster-a",
        command.command_id,
        RemoteCommandResult(
            lease_token=str(command.lease_token),
            status=status,
            details=details,
            error="confirmed rejection"
            if status is RemoteCommandStatus.FAILED
            else None,
        ),
    )


def expire(flow: FlowHarness, kind: str) -> None:
    now = datetime.now(timezone.utc)
    current = flow.store.get_workflow(flow.workflow.request_id)
    if kind == "execution":
        flow.amend(execution_deadline=now - timedelta(seconds=1))
    elif kind == "lifetime":
        flow.amend(lifetime_deadline_at=now - timedelta(seconds=1))
    else:
        limit = max(
            flow.executor.config.step_waiting_limit(step.operation)
            for step in current.official_steps
        )
        started = now - timedelta(seconds=limit + 60)
        flow.amend(
            execution_deadline=now + timedelta(seconds=600),
            events=[
                item.model_copy(update={"at": started})
                if item.kind is WorkflowEventKind.CLAIM
                else item
                for item in current.events
            ],
            step_executions=[
                item.model_copy(update={"started_at": started})
                for item in current.step_executions
            ],
        )


def assert_no_restoration(flow: FlowHarness) -> None:
    assert not any(
        command.step.operation is OP.RESTORE_GPU_SERVICES
        for command in flow.store.list_remote_commands()
    ), "an unresolved action must not authorize a new restoration command"


@pytest.mark.parametrize(
    "operation,batched",
    [(OP.RESET_GPU, False), (OP.RESET_GPU, True), (OP.RESTART_NODE, False)],
)
@pytest.mark.parametrize("state", ["leased", "waiting"])
@pytest.mark.parametrize("bound", ["execution", "lifetime", "step"])
def test_remote_mutation_stays_occupied_at_every_time_bound(
    operation: OP, batched: bool, state: str, bound: str
) -> None:
    flow = remote_flow(operation, batched=batched)
    command = claim(flow)
    if state == "waiting":
        report(
            flow,
            command,
            dict(UNKNOWN_RESET if operation is OP.RESET_GPU else PENDING_REBOOT),
        )
    flow.execute()
    pending = flow.store.get_workflow(flow.workflow.request_id)
    assert has_unresolved_node_action(pending), (
        "regional projection must retain the physical action's unresolved state"
    )

    expire(flow, bound)
    result = flow.execute()
    saved = flow.store.get_workflow(flow.workflow.request_id)

    assert result.status is WorkflowStatus.BLOCKED
    assert saved.blocked_kind is BlockedKind.NEEDS_OPERATOR
    assert workflow_is_open(saved.status, saved.blocked_kind), (
        "a deadline or cancellation request is not a physical completion receipt"
    )
    assert_no_restoration(flow)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(
    "details",
    [
        {"fleet_preflight_blocked": True},
        {"multi_node_barrier_unavailable": True},
        {"node_action_not_started": True},
        {"node_action_state": "TRANSPORT_RETRY", "node_action_command_id": "unsent"},
    ],
)
def test_known_preflight_hold_remains_safely_cancellable(
    batched: bool, details: dict[str, Any]
) -> None:
    flow = remote_flow(batched=batched)
    command = claim(flow)
    flow.execute()
    report(flow, command, details)
    expire(flow, "execution")

    result = flow.execute()
    saved = flow.store.get_workflow(flow.workflow.request_id)

    assert result.status is not WorkflowStatus.BLOCKED, (
        "a newer no-start receipt must resolve an earlier leased snapshot"
    )
    assert not has_unresolved_node_action(saved), (
        "the fresh no-start receipt resolves the lease"
    )
    assert any(
        item.step.operation is OP.RESTORE_GPU_SERVICES
        for item in flow.store.list_remote_commands()
    ), "confirmed preflight refusal still permits compensation for completed quiesce"


@pytest.mark.parametrize("batched", [False, True])
def test_never_claimed_command_can_be_cancelled_without_an_operator_hold(
    batched: bool,
) -> None:
    flow = remote_flow(batched=batched)
    expire(flow, "execution")

    flow.execute()
    saved = flow.store.get_workflow(flow.workflow.request_id)

    assert saved.status is not WorkflowStatus.BLOCKED
    assert not has_unresolved_node_action(saved), (
        "an unclaimed command has no physical action"
    )
    assert any(
        item.step.operation is OP.RESTORE_GPU_SERVICES
        for item in flow.store.list_remote_commands()
    ), "cancelling a genuinely unclaimed command must preserve ordinary compensation"


def test_claim_between_deadline_read_and_cancellation_is_not_a_no_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flow = remote_flow()
    original = flow.store.cancel_remote_commands_for_workflow

    def cancel(request_id: str, *, reason: str) -> dict[str, int]:
        claim(flow)
        return original(request_id, reason=reason)

    monkeypatch.setattr(flow.store, "cancel_remote_commands_for_workflow", cancel)
    expire(flow, "execution")

    result = flow.execute()

    assert result.status is WorkflowStatus.BLOCKED, (
        "the cancellation result and post-cancel lease must supersede the old PENDING read"
    )
    assert_no_restoration(flow)


@pytest.mark.parametrize(
    "details",
    [
        {**UNKNOWN_RESET, "fleet_preflight_blocked": True},
        {**UNKNOWN_RESET, "node_action_not_started": True},
        {"outcome_unknown": True, "fleet_preflight_blocked": True},
        {**PENDING_REBOOT, "node_action_not_started": True},
    ],
)
def test_no_start_hint_cannot_override_accepted_or_unknown_action(
    details: dict[str, Any],
) -> None:
    flow = remote_flow()
    report(flow, claim(flow), details)
    expire(flow, "execution")

    assert flow.execute().status is WorkflowStatus.BLOCKED
    assert_no_restoration(flow)


def test_terminal_funnel_rechecks_remote_state_after_watchdog_updates() -> None:
    flow = remote_flow()
    claim(flow)
    current = flow.store.get_workflow(flow.workflow.request_id)

    result = flow.executor.terminalize_claimed(
        current,
        flow.incident,
        WorkflowStatus.FAILED,
        current.execution_epoch,
        reason="watchdog deadline",
        actor="test-watchdog",
        updates={
            "step_executions": [
                current.step_executions[0],
                workflow_step_execution(
                    1,
                    OP.RESET_GPU,
                    WorkflowStepStatus.FAILED,
                    phase="official",
                    details={"workflow_deadline_overdue_seconds": 1},
                ),
            ]
        },
    )

    assert result.status is WorkflowStatus.BLOCKED
    assert flow.store.get_workflow(current.request_id).blocked_kind is (
        BlockedKind.NEEDS_OPERATOR
    )
    assert_no_restoration(flow)


@pytest.mark.parametrize("batched", [False, True])
def test_same_command_success_resolves_its_pending_physical_state(
    batched: bool,
) -> None:
    flow = remote_flow(batched=batched)
    command = claim(flow)
    flow.execute()
    assert has_unresolved_node_action(
        flow.store.get_workflow(flow.workflow.request_id)
    ), "the initial leased command must be treated as unresolved"

    report(flow, command, {}, status=RemoteCommandStatus.SUCCEEDED)
    flow.execute()
    saved = flow.store.get_workflow(flow.workflow.request_id)

    assert not has_unresolved_node_action(saved), (
        "a matching completion resolves the wait"
    )
    assert OP.RESET_GPU in saved.completed_operations
    assert [
        item.command_id
        for item in flow.store.list_remote_commands()
        if item.step.operation is OP.RESET_GPU
    ] == [command.command_id], "polling must not mint a replacement reset"
    if not batched:
        restoration = claim(flow)
        assert restoration.step.operation is OP.RESTORE_GPU_SERVICES
        report(flow, restoration, {}, status=RemoteCommandStatus.SUCCEEDED)
        flow.execute()
    assert (
        flow.store.get_workflow(flow.workflow.request_id).status
        is WorkflowStatus.SUCCEEDED
    )


@pytest.mark.parametrize("batched", [False, True])
def test_confirmed_ordinary_failure_still_compensates(batched: bool) -> None:
    flow = remote_flow(batched=batched)
    command = claim(flow)
    flow.execute()
    report(flow, command, {}, status=RemoteCommandStatus.FAILED)

    flow.execute()

    assert not has_unresolved_node_action(
        flow.store.get_workflow(flow.workflow.request_id)
    ), "a conclusive failure resolves the earlier leased snapshot"
    assert any(
        item.step.operation is OP.RESTORE_GPU_SERVICES
        for item in flow.store.list_remote_commands()
    ), "a confirmed ordinary failure must retain its compensation path"


@pytest.mark.parametrize(
    "post_status", [RemoteCommandStatus.WAITING, RemoteCommandStatus.FAILED]
)
@pytest.mark.parametrize("started", [False, True])
def test_post_cancellation_receipt_distinguishes_no_start_from_unknown(
    post_status: RemoteCommandStatus, started: bool
) -> None:
    flow = remote_flow()
    command = claim(flow)
    flow.execute()
    flow.store.cancel_remote_commands_for_workflow(
        flow.workflow.request_id, reason="cancel"
    )
    report(
        flow,
        command,
        dict(UNKNOWN_RESET) if started else {"node_action_not_started": True},
        status=post_status,
    )
    expire(flow, "execution")

    result = flow.execute()

    if started:
        assert result.status is WorkflowStatus.BLOCKED
        assert_no_restoration(flow)
    else:
        assert result.status is not WorkflowStatus.BLOCKED
        assert not has_unresolved_node_action(
            flow.store.get_workflow(flow.workflow.request_id)
        ), "the post-cancellation no-start receipt resolves the wait"


@pytest.mark.parametrize("raw_status", [None, [], {}, 1, "UNKNOWN"])
def test_malformed_batched_progress_cannot_authorize_compensation(
    raw_status: Any,
) -> None:
    flow = remote_flow(batched=True)
    command = claim(flow)
    flow.store.complete_remote_command(
        "cluster-a",
        command.command_id,
        RemoteCommandResult(
            lease_token=str(command.lease_token),
            status=RemoteCommandStatus.WAITING,
            details={BATCHED_RESULTS_KEY: {"1": {"status": raw_status, "details": {}}}},
        ),
    )
    expire(flow, "execution")

    assert flow.execute().status is WorkflowStatus.BLOCKED
    assert_no_restoration(flow)


def test_unavailable_receipt_read_does_not_release_or_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flow = remote_flow()
    claim(flow)
    commands = flow.store.list_remote_commands
    expire(flow, "execution")

    def unavailable(**_kwargs):
        raise RuntimeError("remote receipt read unavailable")

    monkeypatch.setattr(flow.store, "list_remote_commands", unavailable)
    with pytest.raises(RuntimeError, match="receipt read unavailable"):
        flow.execute()

    saved = flow.store.get_workflow(flow.workflow.request_id)
    assert workflow_is_open(saved.status, saved.blocked_kind), (
        "a receipt read failure must retain node occupancy"
    )
    assert not any(
        item.step.operation is OP.RESTORE_GPU_SERVICES for item in commands()
    ), "receipt unavailability must not authorize restoration"
