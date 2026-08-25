from __future__ import annotations

from gpu_fault.hyperpod_spares import SpareHealthPending
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    execute_workflow,
    workflow_step_execution,
)
from tests.execution._support import _remote_waiting_state

from ._support import (
    NOW,
    AgentHeartbeat,
    AgentLifecycleState,
    FakeAdapter,
    FakeHyperPodLifecycle,
    FakeSpareCoordinator,
    FleetRegistry,
    HyperPodLifecycleStepAdapter,
    HyperPodNode,
    RecordingNodeActionAdapter,
    RemoteActionCommand,
    RemoteCommandStatus,
    SignedAgentHeartbeat,
    SpareAllocation,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepContext,
    WorkflowStepOutcome,
    WorkflowStepStatus,
    _preempting_successor,
    sign_agent_heartbeat,
    workflow_state,
)


def test_remote_waiting_step_is_not_preempted() -> None:
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.RESTART_NODE])
    command = RemoteActionCommand(
        command_id="remote-leased",
        cluster_id=incident.cluster_id,
        workflow_request_id=workflow.request_id,
        incident_id=incident.incident_id,
        step_index=0,
        fencing_token=workflow.fencing_token,
        idempotency_key="workflow-active/0/RESTART_NODE",
        step=workflow.official_steps[0],
        workflow=workflow,
        incident=incident,
    )
    store.ensure_remote_command(command)
    claimed = store.claim_remote_commands(
        incident.cluster_id, "cluster-executor-a", limit=1, lease_seconds=60
    )
    assert claimed[0].status is RemoteCommandStatus.LEASED
    workflow = copy_model(
        workflow,
        status=WorkflowStatus.RUNNING,
        step_executions=[
            workflow_step_execution(
                0,
                WorkflowOperation.RESTART_NODE,
                WorkflowStepStatus.WAITING,
                adapter_operation_id="remote/remote-leased",
                details={
                    "remote_status": "LEASED",
                    "remote_command_id": "remote-leased",
                },
            )
        ],
    )
    store.save_workflow(workflow)
    _preempting_successor(store, incident, workflow)
    adapter = FakeAdapter(
        {
            WorkflowOperation.RESTART_NODE: (
                WorkflowStepOutcome.waiting(operation_id="remote/remote-leased")
            )
        }
    )
    active = active_workflow_executor(
        store, [adapter], {WorkflowOperation.RESTART_NODE}
    )

    result = execute_workflow(active, workflow.request_id)

    assert result.status is WorkflowStatus.RUNNING
    assert store.get_workflow(workflow.request_id).preempted_by_workflow_id is None
    assert adapter.calls == ["workflow-active/0/RESTART_NODE"]


def test_unclaimed_remote_command_is_cancelled_before_preemption() -> None:
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.RESTART_NODE])
    command = RemoteActionCommand(
        command_id="remote-pending",
        cluster_id=incident.cluster_id,
        workflow_request_id=workflow.request_id,
        incident_id=incident.incident_id,
        step_index=0,
        fencing_token=workflow.fencing_token,
        idempotency_key="workflow-active/0/RESTART_NODE",
        step=workflow.official_steps[0],
        workflow=workflow,
        incident=incident,
    )
    store.ensure_remote_command(command)
    workflow = copy_model(
        workflow,
        status=WorkflowStatus.RUNNING,
        step_executions=[
            workflow_step_execution(
                0,
                WorkflowOperation.RESTART_NODE,
                WorkflowStepStatus.WAITING,
                adapter_operation_id="remote/remote-pending",
                details={
                    "remote_status": "PENDING",
                    "remote_command_id": "remote-pending",
                },
            )
        ],
    )
    store.save_workflow(workflow)
    _preempting_successor(store, incident, workflow)
    adapter = FakeAdapter(
        {WorkflowOperation.RESTART_NODE: (WorkflowStepOutcome.waiting())}
    )
    active = active_workflow_executor(
        store, [adapter], {WorkflowOperation.RESTART_NODE}
    )

    result = execute_workflow(active, workflow.request_id)

    assert result.status is WorkflowStatus.SUPERSEDED
    cancelled = store.get_remote_command("remote-pending")
    assert cancelled.status is RemoteCommandStatus.FAILED
    assert cancelled.status_source == "workflow-preempted"
    assert cancelled.lease_owner is None
    assert cancelled.lease_token is None
    assert cancelled.lease_expires_at is None
    assert (
        store.claim_remote_commands(
            incident.cluster_id,
            "executor-after-preemption",
            limit=1,
            lease_seconds=60,
            execution_owners={workflow.official_steps[0].execution_owner},
        )
        == []
    )
    assert adapter.calls == []


def test_safe_remote_waiting_is_cancelled_before_preemption() -> None:
    store = build_store()
    incident, workflow, command_id = _remote_waiting_state(
        store, WorkflowOperation.VERIFY_NO_GPU_CLIENTS
    )
    _preempting_successor(store, incident, workflow)
    adapter = FakeAdapter(
        {WorkflowOperation.VERIFY_NO_GPU_CLIENTS: (WorkflowStepOutcome.waiting())}
    )
    active = active_workflow_executor(
        store, [adapter], {WorkflowOperation.VERIFY_NO_GPU_CLIENTS}
    )

    result = execute_workflow(active, workflow.request_id)

    assert result.status is WorkflowStatus.SUPERSEDED
    assert store.get_remote_command(command_id).status is RemoteCommandStatus.FAILED
    assert adapter.calls == []


def test_restart_workload_remote_waiting_is_not_preempted() -> None:
    store = build_store()
    incident, workflow, command_id = _remote_waiting_state(
        store, WorkflowOperation.RESTART_WORKLOAD
    )
    _preempting_successor(store, incident, workflow)
    adapter = FakeAdapter(
        {
            WorkflowOperation.RESTART_WORKLOAD: (
                WorkflowStepOutcome.waiting(operation_id=f"remote/{command_id}")
            )
        }
    )
    active = active_workflow_executor(
        store, [adapter], {WorkflowOperation.RESTART_WORKLOAD}
    )

    result = execute_workflow(active, workflow.request_id)

    assert result.status is WorkflowStatus.RUNNING
    assert store.get_remote_command(command_id).status is RemoteCommandStatus.WAITING
    assert adapter.calls == ["workflow-active/0/RESTART_WORKLOAD"]


def test_hyperpod_replace_is_blocked_when_spares_are_insufficient():
    store = build_store()
    _, workflow = workflow_state(store, [WorkflowOperation.REPLACE_NODE])
    lifecycle = FakeHyperPodLifecycle()
    spares = FakeSpareCoordinator(
        SpareAllocation(
            applicable=True,
            sufficient=False,
            required=1,
            reason="required=1, healthy=0",
        )
    )
    adapter = HyperPodLifecycleStepAdapter(lifecycle, spare_coordinator=spares)
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        parameters={"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"},
    )
    store.save_workflow(copy_model(workflow, official_steps=[step]))
    active = active_workflow_executor(
        store, [adapter], {WorkflowOperation.REPLACE_NODE}
    )

    result = execute_workflow(
        active,
        workflow.request_id,
        isolation_verified_nodes=["node-a"],
        confirm_cluster_name="hp-cluster",
    )

    assert result.status is WorkflowStatus.FAILED
    assert "required=1, healthy=0" in result.error
    assert lifecycle.calls == 0
    assert spares.calls[0]["fault_node_ids"] == ["node-a"]


def test_warm_spare_requires_enabled_coordinator_before_submission():
    store = build_store()
    _, workflow = workflow_state(store, [WorkflowOperation.REPLACE_NODE])
    lifecycle = FakeHyperPodLifecycle()
    adapter = HyperPodLifecycleStepAdapter(lifecycle)
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        parameters={"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"},
    )
    store.save_workflow(copy_model(workflow, official_steps=[step]))
    active = active_workflow_executor(
        store, [adapter], {WorkflowOperation.REPLACE_NODE}
    )

    result = execute_workflow(
        active,
        workflow.request_id,
        isolation_verified_nodes=["node-a"],
        confirm_cluster_name="hp-cluster",
    )

    assert result.status is WorkflowStatus.FAILED
    assert result.error == (
        "healthy warm-spare replacement is required but the "
        "spare coordinator is disabled"
    )
    assert lifecycle.calls == 0


def test_pending_spare_health_check_is_retried():
    class PendingSpareCoordinator:
        def __init__(self):
            self.calls = []
            self.releases = []

        def allocate(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise SpareHealthPending("node agent GPU client check is pending")
            return SpareAllocation(
                applicable=True,
                sufficient=True,
                required=1,
                selected_node_ids=("spare-a",),
            )

        def release(self, node_ids, incident_id):
            self.releases.append((node_ids, incident_id))

    store = build_store()
    _, workflow = workflow_state(store, [WorkflowOperation.REPLACE_NODE])
    lifecycle = FakeHyperPodLifecycle()
    spares = PendingSpareCoordinator()
    node_actions = RecordingNodeActionAdapter()
    adapter = HyperPodLifecycleStepAdapter(
        lifecycle, spare_coordinator=spares, node_action_adapter=node_actions
    )
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        parameters={"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"},
    )
    store.save_workflow(copy_model(workflow, official_steps=[step]))
    active = active_workflow_executor(
        store, [adapter], {WorkflowOperation.REPLACE_NODE}
    )

    request = WorkflowExecutionRequest(
        expected_fencing_token=3,
        isolation_verified_nodes=["node-a"],
        confirm_cluster_name="hp-cluster",
    )
    waiting = active.execute(workflow.request_id, request)

    execution = store.get_workflow(workflow.request_id).step_executions[0]
    assert waiting.status is WorkflowStatus.RUNNING
    assert execution.status is WorkflowStepStatus.WAITING
    assert execution.adapter_operation_id is None
    assert execution.details["spare_health_pending"] is True
    completed = active.execute(workflow.request_id, request)
    assert completed.status is WorkflowStatus.SUCCEEDED
    assert lifecycle.calls == 0
    assert len(spares.calls) == 2


def test_remote_pending_spare_health_check_is_retried():
    class PendingThenReady:
        def __init__(self):
            self.calls = []
            self.releases = []

        def allocate(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise SpareHealthPending("node action is pending")
            return SpareAllocation(
                applicable=True,
                sufficient=True,
                required=1,
                selected_node_ids=("spare-a",),
            )

        def release(self, node_ids, incident_id):
            self.releases.append((node_ids, incident_id))

    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.REPLACE_NODE])
    spares = PendingThenReady()
    adapter = HyperPodLifecycleStepAdapter(
        FakeHyperPodLifecycle(),
        spare_coordinator=spares,
        node_action_adapter=RecordingNodeActionAdapter(),
    )
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        parameters={"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"},
    )
    workflow = copy_model(workflow, official_steps=[step])
    request = WorkflowExecutionRequest(
        expected_fencing_token=3,
        isolation_verified_nodes=["node-a"],
        confirm_cluster_name="hp-cluster",
    )

    waiting = adapter.execute(
        WorkflowStepContext(
            workflow, incident, step, 0, request, "workflow-active/0/REPLACE_NODE"
        )
    )
    remote_workflow = copy_model(
        workflow,
        step_executions=[
            workflow_step_execution(
                0,
                WorkflowOperation.REPLACE_NODE,
                WorkflowStepStatus.WAITING,
                adapter_operation_id="remote/command-a",
                details=waiting.details,
            )
        ],
    )
    completed = adapter.execute(
        WorkflowStepContext(
            remote_workflow,
            incident,
            step,
            0,
            request,
            "workflow-active/0/REPLACE_NODE",
        )
    )

    assert waiting.status is WorkflowStepStatus.WAITING
    assert waiting.details["spare_failover_pending"] is True
    assert completed.status is WorkflowStepStatus.SUCCEEDED
    assert len(spares.calls) == 2
    assert spares.releases == []


def test_remote_pending_spare_snapshot_resumes_without_reallocation():
    class WaitingSnapshot(RecordingNodeActionAdapter):
        def execute(self, context):
            self.contexts.append(context)
            if len(self.contexts) == 1:
                return WorkflowStepOutcome.waiting(operation_id=context.idempotency_key)
            return WorkflowStepOutcome.succeeded(
                operation_id=context.idempotency_key,
                details={"snapshot_triggered": True},
            )

    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.REPLACE_NODE])
    spares = FakeSpareCoordinator(
        SpareAllocation(
            applicable=True, sufficient=True, required=1, selected_node_ids=("spare-a",)
        )
    )
    node_actions = WaitingSnapshot()
    adapter = HyperPodLifecycleStepAdapter(
        FakeHyperPodLifecycle(),
        spare_coordinator=spares,
        node_action_adapter=node_actions,
    )
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        parameters={"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"},
    )
    workflow = copy_model(workflow, official_steps=[step])
    request = WorkflowExecutionRequest(
        expected_fencing_token=3,
        isolation_verified_nodes=["node-a"],
        confirm_cluster_name="hp-cluster",
    )
    context = WorkflowStepContext(
        workflow, incident, step, 0, request, "workflow-active/0/REPLACE_NODE"
    )

    waiting = adapter.execute(context)
    remote_workflow = copy_model(
        workflow,
        step_executions=[
            workflow_step_execution(
                0,
                WorkflowOperation.REPLACE_NODE,
                WorkflowStepStatus.WAITING,
                adapter_operation_id="remote/command-b",
                details=waiting.details,
            )
        ],
    )
    completed = adapter.execute(
        WorkflowStepContext(
            remote_workflow, incident, step, 0, request, context.idempotency_key
        )
    )

    assert waiting.status is WorkflowStepStatus.WAITING
    assert waiting.details["activated_spare_nodes"] == ["spare-a"]
    assert completed.status is WorkflowStepStatus.SUCCEEDED
    assert len(spares.calls) == 1
    assert spares.releases == []
    assert len(node_actions.contexts) == 2


def test_warm_spare_rejects_automatic_node_recovery_before_allocation():
    from types import SimpleNamespace

    store = build_store()
    _, workflow = workflow_state(store, [WorkflowOperation.REPLACE_NODE])
    lifecycle = FakeHyperPodLifecycle()
    spares = FakeSpareCoordinator(
        SpareAllocation(
            applicable=True, sufficient=True, required=1, selected_node_ids=("spare-a",)
        )
    )
    adapter = HyperPodLifecycleStepAdapter(lifecycle, spare_coordinator=spares)
    adapter.dispatcher.preflight = lambda *_args, **_kwargs: (
        SimpleNamespace(
            safe_to_submit=False,
            gate_failures=["HyperPod automatic node recovery is enabled"],
            node_recovery="Automatic",
        )
    )
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        parameters={"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"},
    )
    store.save_workflow(copy_model(workflow, official_steps=[step]))
    active = active_workflow_executor(
        store, [adapter], {WorkflowOperation.REPLACE_NODE}
    )

    result = execute_workflow(
        active,
        workflow.request_id,
        isolation_verified_nodes=["node-a"],
        confirm_cluster_name="gpu-fault-control-plane-test",
    )

    assert result.status is WorkflowStatus.FAILED
    assert ("healthy warm-spare replacement requires HyperPod NodeRecovery=None") in (
        result.error or ""
    )
    assert spares.calls == []
    assert lifecycle.calls == 0


def test_hyperpod_replace_records_activated_spare_nodes():
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.REPLACE_NODE])
    lifecycle = FakeHyperPodLifecycle()
    spares = FakeSpareCoordinator(
        SpareAllocation(
            applicable=True, sufficient=True, required=1, selected_node_ids=("spare-a",)
        )
    )
    node_actions = RecordingNodeActionAdapter({"hyperpod-i-spare": "http://spare:9099"})
    adapter = HyperPodLifecycleStepAdapter(
        lifecycle,
        spare_coordinator=spares,
        node_action_adapter=node_actions,
        notification_sink=store,
    )
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        parameters={"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"},
    )
    store.save_workflow(copy_model(workflow, official_steps=[step]))
    active = active_workflow_executor(
        store, [adapter], {WorkflowOperation.REPLACE_NODE}
    )

    result = execute_workflow(
        active,
        workflow.request_id,
        isolation_verified_nodes=["node-a"],
        confirm_cluster_name="hp-cluster",
    )

    execution = store.get_workflow(workflow.request_id).step_executions[0]
    assert result.status is WorkflowStatus.SUCCEEDED
    assert execution.details["activated_spare_nodes"] == ["spare-a"]
    assert execution.details["action"] == "SPARE_FAILOVER"
    assert execution.details["node_rebindings"] == {"node-a": "spare-a"}
    assert execution.details["provider_mutation_submitted"] is False
    assert execution.details["notification_id"]
    notifications = store.list_notifications()
    assert len(notifications) == 1
    notification = notifications[0]
    assert notification.notification_id == (execution.details["notification_id"])
    assert "warm-spare替换成功" in notification.subject
    assert "node-a -> spare-a" in notification.body_text
    assert incident.incident_id in notification.body_text
    assert workflow.request_id in notification.body_text
    assert execution.adapter_operation_id in notification.body_text
    assert "healthy-running-warm-spare" in notification.body_text
    assert "Provider replacement API submitted：false" in (notification.body_text)
    assert "GPU/Fabric validation" in notification.body_text
    assert "Workflow 最终状态为准" in notification.body_text
    assert lifecycle.calls == 0
    assert lifecycle.preflight_kwargs == {
        "isolation_verified_nodes": ["node-a"],
        "require_execution_enabled": False,
    }
    assert spares.calls[0]["local_only"] is True
    assert [item.step.operation for item in node_actions.contexts] == [
        WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT
    ]
    assert node_actions.contexts[0].step.node_ids == ["spare-a"]
    assert execution.details["active_health_snapshot"] == {"snapshot_triggered": True}
    retry_id = adapter._notify_warm_spare_replaced(
        WorkflowStepContext(
            workflow=store.get_workflow(workflow.request_id),
            incident=store.get_incident(incident.incident_id),
            step=step,
            step_index=0,
            request=WorkflowExecutionRequest(expected_fencing_token=3),
            idempotency_key=execution.adapter_operation_id,
        ),
        execution.adapter_operation_id,
        execution.details,
    )
    assert retry_id == notification.notification_id
    assert len(store.list_notifications()) == 1


def test_control_plane_persists_remote_warm_spare_success_email() -> None:
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.REPLACE_NODE])
    adapter = FakeAdapter(
        {
            WorkflowOperation.REPLACE_NODE: (
                WorkflowStepOutcome.succeeded(
                    operation_id="remote/replace-a",
                    details={
                        "action": "SPARE_FAILOVER",
                        "activated_spare_nodes": ["node-spare"],
                        "node_rebindings": {"node-a": "node-spare"},
                        "confirmation_source": ("healthy-running-warm-spare"),
                        "provider_mutation_submitted": False,
                    },
                )
            )
        }
    )
    sent = []
    executor = active_workflow_executor(
        store,
        [adapter],
        {WorkflowOperation.REPLACE_NODE},
        notification_sender=sent.append,
    )

    result = execute_workflow(executor, workflow.request_id)

    assert result.status is WorkflowStatus.SUCCEEDED
    notifications = store.list_notifications()
    assert len(notifications) == 1
    assert sent == [notifications[0].notification_id]
    assert "node-a -> node-spare" in notifications[0].body_text
    execution = store.get_workflow(workflow.request_id).step_executions[0]
    assert execution.details["notification_id"] == (notifications[0].notification_id)


def test_hyperpod_spare_checker_uses_node_agent_for_each_phase():
    store = build_store()
    _, workflow = workflow_state(store, [WorkflowOperation.REPLACE_NODE])
    lifecycle = FakeHyperPodLifecycle()
    spares = FakeSpareCoordinator(
        SpareAllocation(
            applicable=True, sufficient=True, required=1, selected_node_ids=("spare-a",)
        )
    )

    node_actions = RecordingNodeActionAdapter()
    adapter = HyperPodLifecycleStepAdapter(
        lifecycle, spare_coordinator=spares, node_action_adapter=node_actions
    )
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        parameters={"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"},
    )
    store.save_workflow(copy_model(workflow, official_steps=[step]))
    active = active_workflow_executor(
        store, [adapter], {WorkflowOperation.REPLACE_NODE}
    )

    execute_workflow(
        active,
        workflow.request_id,
        isolation_verified_nodes=["node-a"],
        confirm_cluster_name="hp-cluster",
    )
    checker = spares.calls[0]["gpu_client_checker"]
    provider_node = HyperPodNode(
        node_logical_id="logical-spare", instance_id="i-spare", status="Running"
    )

    assert checker(provider_node, "hyperpod-i-spare", "candidate") == []
    assert checker(provider_node, "hyperpod-i-spare", "activation") == []
    assert [item.step.operation for item in node_actions.contexts] == [
        WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT,
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
    ]
    assert all(
        item.step.parameters.get("spare_health_check") is True
        for item in node_actions.contexts
        if item.step.operation is WorkflowOperation.VERIFY_NO_GPU_CLIENTS
    )
    snapshot = node_actions.contexts[0]
    assert snapshot.step.node_ids == ["spare-a"]
    assert snapshot.step.gpu_uuids == []
    assert snapshot.step.parameters == {}
    assert all(
        item.step.node_ids == ["hyperpod-i-spare"]
        and item.step.gpu_uuids == []
        and item.step.parameters
        == {"compute_clients_only": True, "spare_health_check": True}
        for item in node_actions.contexts[1:]
    )
    assert (
        node_actions.contexts[1].idempotency_key
        != node_actions.contexts[2].idempotency_key
    )


def test_hyperpod_replace_maps_two_fault_nodes_to_two_warm_spares():
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.REPLACE_NODE])
    store.save_incident(copy_model(incident, node_ids=["node-a", "node-b"]))
    lifecycle = FakeHyperPodLifecycle()
    spares = FakeSpareCoordinator(
        SpareAllocation(
            applicable=True,
            sufficient=True,
            required=2,
            selected_node_ids=("spare-a", "spare-b"),
        )
    )
    node_actions = RecordingNodeActionAdapter()
    adapter = HyperPodLifecycleStepAdapter(
        lifecycle, spare_coordinator=spares, node_action_adapter=node_actions
    )
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        node_ids=["node-a", "node-b"],
    )
    store.save_workflow(copy_model(workflow, official_steps=[step]))
    active = active_workflow_executor(
        store, [adapter], {WorkflowOperation.REPLACE_NODE}
    )

    result = execute_workflow(
        active,
        workflow.request_id,
        isolation_verified_nodes=["node-a", "node-b"],
        confirm_cluster_name="hp-cluster",
    )

    execution = store.get_workflow(workflow.request_id).step_executions[0]
    assert result.status is WorkflowStatus.SUCCEEDED
    assert spares.calls[0]["fault_node_ids"] == ["node-a", "node-b"]
    assert execution.details["node_rebindings"] == {
        "node-a": "spare-a",
        "node-b": "spare-b",
    }
    assert execution.details["provider_mutation_submitted"] is False
    assert lifecycle.calls == 0
    assert node_actions.contexts[0].step.node_ids == ["spare-a", "spare-b"]


def test_hyperpod_automatic_recovery_cannot_bypass_spare_only_mode():
    class AutomaticLifecycle(FakeHyperPodLifecycle):
        def preflight(self, *_args, **_kwargs):
            from types import SimpleNamespace

            return SimpleNamespace(
                safe_to_submit=True, gate_failures=[], node_recovery="Automatic"
            )

    store = build_store()
    _, workflow = workflow_state(store, [WorkflowOperation.REPLACE_NODE])
    lifecycle = AutomaticLifecycle()
    spares = FakeSpareCoordinator(
        SpareAllocation(
            applicable=True,
            sufficient=False,
            required=1,
            reason="spare gate must not run",
        )
    )
    adapter = HyperPodLifecycleStepAdapter(lifecycle, spare_coordinator=spares)
    step = copy_model(workflow.official_steps[0], execution_owner=adapter.owner)
    store.save_workflow(copy_model(workflow, official_steps=[step]))
    active = active_workflow_executor(
        store, [adapter], {WorkflowOperation.REPLACE_NODE}
    )

    result = execute_workflow(
        active,
        workflow.request_id,
        isolation_verified_nodes=["node-a"],
        confirm_cluster_name="hp-cluster",
    )

    assert result.status is WorkflowStatus.FAILED
    assert "provider node replacement API fallback is disabled" in (result.error)
    assert spares.calls == []
    assert lifecycle.calls == 0


def test_hyperpod_step_waits_for_external_confirmation() -> None:
    store = build_store()
    secret = "agent-secret-" + "x" * 32
    fleet = FleetRegistry(store, secret, now=lambda: NOW)
    heartbeat = AgentHeartbeat(
        cluster_id="cluster-a",
        node_id="node-a",
        endpoint="http://node-a:9099",
        agent_protocol_version=3,
        agent_version="0.10.0",
        artifact_sha256="a" * 64,
        policy_version="610",
        runtime_profile_version="active-v1",
        config_digest="c" * 64,
        allowed_operations=[
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            WorkflowOperation.RESET_GPU,
        ],
        boot_id="boot-a",
        node_instance_id="instance-a",
        agent_incarnation_id="incarnation-a",
        observed_at=NOW,
    )
    fleet.register(
        SignedAgentHeartbeat(
            heartbeat=heartbeat, signature=sign_agent_heartbeat(heartbeat, secret)
        )
    )
    _, workflow = workflow_state(store, [WorkflowOperation.RESTART_NODE])
    lifecycle = FakeHyperPodLifecycle()
    adapter = HyperPodLifecycleStepAdapter(lifecycle, registry=fleet)
    step = copy_model(workflow.official_steps[0], execution_owner=adapter.owner)
    workflow = copy_model(workflow, official_steps=[step])
    store.save_workflow(workflow)
    active = active_workflow_executor(
        store, [adapter], {WorkflowOperation.RESTART_NODE}
    )
    request = WorkflowExecutionRequest(
        expected_fencing_token=3,
        isolation_verified_nodes=["node-a"],
        confirm_cluster_name="hp-cluster",
    )

    submitted = active.execute(workflow.request_id, request)
    observed = active.execute(workflow.request_id, request)
    completed = active.execute(
        workflow.request_id,
        copy_model(request, confirmed_adapter_operation_ids=["hyperpod-operation-1"]),
    )

    assert submitted.status is WorkflowStatus.RUNNING
    assert observed.status is WorkflowStatus.RUNNING
    assert lifecycle.calls == 1
    assert completed.status is WorkflowStatus.SUCCEEDED
    record = store.get_agent("cluster-a", "node-a")
    assert record.lifecycle_state is AgentLifecycleState.REVOKED
    assert record.transition_id == ("workflow-active/0/RESTART_NODE/hyperpod-lifecycle")
