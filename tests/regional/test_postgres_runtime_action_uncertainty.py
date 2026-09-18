"""Native receipt/cancellation round trips on an explicitly granted local database."""

from __future__ import annotations

import os
from contextlib import closing
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.execution.node_action_uncertainty import has_unresolved_node_action
from gpu_fault.models import (
    BlockedKind,
    IncidentState,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStatus,
    workflow_is_open,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.families.conflicts import NodeConflictService
from gpu_fault.regional import (
    RegionalClusterRegistration,
    RegionalRemoteWorkflowAdapter,
    RemoteCommandResult,
    RemoteCommandStatus,
)
from gpu_fault.regional_compatibility import CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION
from gpu_fault.remote_command_models import BATCHED_RESULTS_KEY
from gpu_fault.remote_step_batching import RemoteStepBatchingPolicy
from tests._builders import (
    active_workflow_executor,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

OP = WorkflowOperation
pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("GPU_FAULT_TEST_POSTGRES_URL"),
        reason="requires an exclusive owned PostgreSQL grant",
    ),
    pytest.mark.allows_cluster_binaries("docker"),
]


@pytest.fixture(params=["legacy", "dual", "dedicated"])
def store(request):
    import psycopg

    from gpu_fault.state_table_migrate import backfill_state_table, set_state_table_mode
    from tests.regional._cov95_notify008_postgres import isolated_database

    with isolated_database() as factory:
        if request.param != "legacy":
            with psycopg.connect(factory.url, autocommit=True) as connection:
                for kind in ("workflow", "remote_command"):
                    set_state_table_mode(
                        connection, kind, "dual", expected_mode="legacy"
                    )
                    if request.param == "dedicated":
                        backfill_state_table(connection, kind)
                        set_state_table_mode(
                            connection,
                            kind,
                            "dedicated",
                            expected_mode="dual",
                            confirm_dedicated=True,
                        )
        with closing(factory()) as instance:
            yield instance


def native_runtime(store, *, operation=OP.RESET_GPU, batched=False):
    operations = (
        [OP.QUIESCE_GPU_SERVICES, operation, OP.RESTORE_GPU_SERVICES]
        if operation is OP.RESET_GPU
        else [OP.QUARANTINE, operation, OP.VALIDATE_GPU, OP.RESTORE_SCHEDULING]
    )
    incident = fault_incident(
        "native-incident",
        "native-event",
        fencing_token=3,
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="native-workflow",
    )
    workflow = workflow_request(
        "native-workflow",
        incident.incident_id,
        official_steps=[workflow_step(value) for value in operations],
        completed_step_indexes=[0],
        completed_operations=[operations[0]],
        step_executions=[workflow_step_execution(0, operations[0], phase="official")],
    )
    store.save_incident_and_workflow(incident, workflow)
    store.save_regional_cluster(
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
    executor = active_workflow_executor(
        store,
        [
            RegionalRemoteWorkflowAdapter(
                store,
                owners={"owner-a"},
                step_batching=RemoteStepBatchingPolicy(
                    batched, CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION
                ),
            )
        ],
        [step.operation for step in workflow.official_steps],
    )
    request = WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token)
    executor.execute(workflow.request_id, request)
    return workflow, executor, request


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(
    "state", ["unclaimed", "preflight", "waiting", "leased", "claim-race"]
)
def test_native_deadline_preserves_physical_uncertainty(
    store, monkeypatch, state, batched
):
    workflow, executor, request = native_runtime(store, batched=batched)

    def claim():
        return store.claim_remote_commands(
            "cluster-a",
            "native-executor",
            limit=1,
            lease_seconds=180,
            executor_protocol_version=CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION,
        )[0]

    if state in {"preflight", "waiting", "leased"}:
        command = claim()
        if state != "leased":
            details = (
                {"fleet_preflight_blocked": True}
                if state == "preflight"
                else {
                    "node_action_state": "PENDING",
                    "node_action_command_id": "native-agent/reset",
                    "node_action_accepted_nodes": ["node-a"],
                }
            )
            if batched:
                details = {
                    BATCHED_RESULTS_KEY: {
                        str(command.step_index): {
                            "status": "WAITING",
                            "details": details,
                        }
                    }
                }
            store.complete_remote_command(
                "cluster-a",
                command.command_id,
                RemoteCommandResult(
                    lease_token=command.lease_token,
                    status=RemoteCommandStatus.WAITING,
                    details=details,
                ),
            )
    elif state == "claim-race":
        cancel = store.cancel_remote_commands_for_workflow

        def racing_cancel(request_id, *, reason):
            claim()
            return cancel(request_id, reason=reason)

        monkeypatch.setattr(store, "cancel_remote_commands_for_workflow", racing_cancel)
    current = store.get_workflow(workflow.request_id)
    store.save_workflow(
        current.model_copy(
            update={
                "execution_deadline": datetime.now(timezone.utc) - timedelta(seconds=1)
            }
        ),
        expected=current,
    )

    result = executor.execute(workflow.request_id, request)
    saved = store.get_workflow(workflow.request_id)
    restorations = [
        command
        for command in store.list_remote_commands()
        if command.step.operation is OP.RESTORE_GPU_SERVICES
    ]
    if state in {"waiting", "leased", "claim-race"}:
        assert result.status is WorkflowStatus.BLOCKED
        assert saved.blocked_kind is BlockedKind.NEEDS_OPERATOR
        assert workflow_is_open(saved.status, saved.blocked_kind), (
            "an unresolved native action must continue to occupy its node"
        )
        assert not restorations, "native cancellation is not physical completion"
    else:
        assert result.status is not WorkflowStatus.BLOCKED
        assert restorations, (
            "native no-start cancellation still permits owed compensation"
        )


@pytest.mark.parametrize("operation", [OP.RESET_GPU, OP.RESTART_NODE])
@pytest.mark.parametrize("entrypoint", ["execute", "dispatcher"])
def test_claim_then_cancel_then_failed_receipt_read_keeps_native_occupancy(
    store, monkeypatch, operation, entrypoint
):
    import psycopg

    workflow, executor, request = native_runtime(store, operation=operation)
    current = store.get_workflow(workflow.request_id)
    assert current.step_executions[-1].details["remote_status"] == "PENDING"
    assert not has_unresolved_node_action(current), (
        "the CPU must start from PENDING without any uncertainty flag"
    )
    commands = store.list_remote_commands
    command_id = commands()[0].command_id
    cancel = store.cancel_remote_commands_for_workflow
    cancelled = False
    failed_reads = 0

    def racing_cancel(request_id, *, reason):
        nonlocal cancelled
        (claimed,) = store.claim_remote_commands(
            "cluster-a",
            "native-racing-executor",
            limit=1,
            lease_seconds=180,
            executor_protocol_version=CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION,
        )
        assert claimed.command_id == command_id
        result = cancel(request_id, reason=reason)
        assert result == {"cancelled": 0, "cancellation_requested": 1}
        cancelled = True
        return result

    def unreadable_receipts(**kwargs):
        nonlocal failed_reads
        if cancelled:
            failed_reads += 1
            raise psycopg.OperationalError("receipt connection lost after cancellation")
        return commands(**kwargs)

    monkeypatch.setattr(store, "cancel_remote_commands_for_workflow", racing_cancel)
    monkeypatch.setattr(store, "list_remote_commands", unreadable_receipts)
    store.save_workflow(
        current.model_copy(
            update={
                "execution_deadline": datetime.now(timezone.utc) - timedelta(seconds=1)
            }
        ),
        expected=current,
    )
    if entrypoint == "execute":
        with pytest.raises(psycopg.OperationalError, match="receipt connection lost"):
            executor.execute(workflow.request_id, request)
    else:
        dispatcher = WorkflowDispatcher(
            store,
            executor,
            WorkflowDispatcherConfig(
                enabled=True, batch_size=1, max_workers=1, dispatch_lease_seconds=0
            ),
        )
        try:
            result = dispatcher.run_once()
            assert result.waiting == 1
            assert result.failed == result.completed == result.internal_errors == 0
        finally:
            dispatcher.stop()

    saved = store.get_workflow(workflow.request_id)
    command = store.get_remote_command(command_id)
    assert cancelled and failed_reads > 0, (
        "the fault must occur after the claim/cancel race"
    )
    assert command.status is RemoteCommandStatus.LEASED
    assert command.cancellation_requested_at is not None
    assert saved.status is WorkflowStatus.RUNNING
    assert saved.execution_owner_id == executor.config.executor_id
    assert workflow_is_open(saved.status, saved.blocked_kind), (
        "metadata I/O failure must not release the reboot/reset node"
    )
    incumbent = NodeConflictService(
        store, RecoveryArbiter()
    ).active_node_exclusive_workflow("cluster-a", {"node-a"})
    assert incumbent is not None and incumbent.request_id == saved.request_id, (
        "the native node-conflict index must still expose this workflow as the owner"
    )
    assert not any(
        item.step.operation is OP.RESTORE_GPU_SERVICES for item in commands()
    ), "a failed receipt read after cancellation must not dispatch restoration"
