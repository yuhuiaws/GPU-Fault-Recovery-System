from __future__ import annotations

import copy

from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    execute_workflow,
    node_action_result,
    workflow_step,
    workflow_step_execution,
)

from ._support import (
    AdvisoryNotificationService,
    FakeAdapter,
    FakeBatchApi,
    FakeCoreApi,
    IncidentState,
    KubernetesWorkflowAdapter,
    NodeActionWorkflowAdapter,
    NotificationResult,
    NotificationStatus,
    RecordingOwnershipProvider,
    SimpleNamespace,
    UnusedApi,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepContext,
    WorkflowStepOutcome,
    WorkflowStepStatus,
    _storeless_isolation_context,
    executor,
    pytest,
    quarantine_taint_value,
    workflow_state,
)


def test_successful_quarantine_keeps_incident_quarantined() -> None:
    store = build_store()
    operation = WorkflowOperation.QUARANTINE
    incident, workflow = workflow_state(store, [operation])
    adapter = FakeAdapter({operation: WorkflowStepOutcome.succeeded()})
    active = executor(store, adapter, [operation])

    result = execute_workflow(active, workflow.request_id)

    assert result.status is WorkflowStatus.SUCCEEDED
    assert store.get_incident(incident.incident_id).state is (IncidentState.QUARANTINED)


def test_reboot_workflow_restarts_workload_only_after_node_recovers() -> None:
    store = build_store()
    operations = [
        WorkflowOperation.STOP_WORKLOADS,
        WorkflowOperation.RESTART_NODE,
        WorkflowOperation.VALIDATE_GPU,
        WorkflowOperation.VALIDATE_HOST,
        WorkflowOperation.VALIDATE_FABRIC,
        WorkflowOperation.RESTORE_SCHEDULING,
        WorkflowOperation.RESTART_WORKLOAD,
    ]
    _, workflow = workflow_state(store, operations)
    adapter = FakeAdapter(
        {
            operation: (
                WorkflowStepOutcome.waiting(operation_id="provider-reboot-1")
                if operation is WorkflowOperation.RESTART_NODE
                else WorkflowStepOutcome.succeeded(operation_id=operation.value)
            )
            for operation in operations
        }
    )
    active = executor(store, adapter, operations)

    waiting = execute_workflow(active, workflow.request_id)

    assert waiting.status is WorkflowStatus.RUNNING
    assert waiting.waiting_step_index == 1
    assert adapter.calls == [
        "workflow-active/0/STOP_WORKLOADS",
        "workflow-active/1/RESTART_NODE",
    ]

    completed = execute_workflow(
        active,
        workflow.request_id,
        confirmed_adapter_operation_ids=["provider-reboot-1"],
    )

    assert completed.status is WorkflowStatus.SUCCEEDED
    assert adapter.calls == [
        "workflow-active/0/STOP_WORKLOADS",
        "workflow-active/1/RESTART_NODE",
        "workflow-active/1/RESTART_NODE",
        "workflow-active/2/VALIDATE_GPU",
        "workflow-active/3/VALIDATE_HOST",
        "workflow-active/4/VALIDATE_FABRIC",
        "workflow-active/5/RESTORE_SCHEDULING",
        "workflow-active/6/RESTART_WORKLOAD",
    ]


def test_fabric_manager_restart_sends_one_idempotent_email() -> None:
    store = build_store()
    incident, workflow = workflow_state(
        store, [WorkflowOperation.RESTART_FABRIC_MANAGER]
    )
    step = copy_model(
        workflow.official_steps[0],
        execution_owner="gpu-fault-node-agent",
        node_ids=["node-a"],
        workload_ids=["training/pytorchjob/training-a"],
    )
    incident = copy_model(
        incident,
        event_id="kernel-log-xid45",
        event_type="XID",
        policy_source="NVIDIA_CATALOG",
        official_action="RESTART_FM",
        reasons=["solo XID 45"],
    )
    workflow = copy_model(workflow, official_steps=[step])
    store.save_incident(incident)
    store.save_workflow(workflow)

    class RecordingNotifier:
        def __init__(self) -> None:
            self.deliveries = []

        def send(self, notification):
            self.deliveries.append(notification)
            return NotificationResult(
                notification_id=notification.notification_id,
                status=NotificationStatus.SENT,
                provider_message_id="ses-fm-1",
            )

    notifier = RecordingNotifier()
    service = AdvisoryNotificationService(store, notifier)

    def sender(_, envelope):
        return node_action_result(
            envelope.command.command_id,
            envelope.command.operation,
            details={
                "service": "nvidia-fabricmanager",
                "active": True,
                "previous_main_pid": "101",
                "current_main_pid": "202",
            },
        )

    adapter = NodeActionWorkflowAdapter(
        {"node-a": "http://node-a:9099"},
        "s" * 32,
        sender=sender,
        store=store,
        alert_sender=service.send,
    )
    context = WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=3),
        idempotency_key=(f"{workflow.request_id}/0/RESTART_FABRIC_MANAGER"),
    )

    first = adapter.execute(context)
    second = adapter.execute(context)

    assert first.status is WorkflowStepStatus.SUCCEEDED
    assert second.status is WorkflowStepStatus.SUCCEEDED
    assert len(store.list_notifications()) == 1
    assert len(notifier.deliveries) == 1
    assert first.details["notification_id"] == (second.details["notification_id"])
    assert "XID 45" in notifier.deliveries[0].body_text


def test_failure_after_isolation_keeps_incident_quarantined() -> None:
    store = build_store()
    operations = [WorkflowOperation.MARK_UNSCHEDULABLE, WorkflowOperation.RESET_GPU]
    _, workflow = workflow_state(store, operations)
    adapter = FakeAdapter(
        {
            WorkflowOperation.MARK_UNSCHEDULABLE: (WorkflowStepOutcome.succeeded()),
            WorkflowOperation.RESET_GPU: (
                WorkflowStepOutcome.failed("agent unavailable")
            ),
        }
    )

    result = execute_workflow(executor(store, adapter, operations), workflow.request_id)

    assert result.status is WorkflowStatus.FAILED
    assert "agent unavailable" in result.error
    assert store.get_incident("incident-active").state is IncidentState.QUARANTINED


def test_kubernetes_adapter_restarts_efa_device_plugin() -> None:
    class EfaCore:
        def __init__(self) -> None:
            self.node = {
                "metadata": {"resourceVersion": "1", "annotations": {}},
                "status": {"allocatable": {"vpc.amazonaws.com/efa": "0"}},
            }
            self.pods = [
                {
                    "metadata": {"name": "efa-plugin-old", "uid": "pod-old"},
                    "status": {"conditions": [{"type": "Ready", "status": "True"}]},
                }
            ]
            self.deleted = []

        def read_node(self, _node_id):
            return self.node

        def patch_node(self, _node_id, body):
            values = body["metadata"]["annotations"]
            for key, value in values.items():
                if value is None:
                    self.node["metadata"]["annotations"].pop(key, None)
                else:
                    self.node["metadata"]["annotations"][key] = value

        def list_namespaced_pod(self, *_args, **_kwargs):
            return SimpleNamespace(items=self.pods)

        def delete_namespaced_pod(self, name, namespace, **_kwargs):
            self.deleted.append((namespace, name))

    store = build_store()
    incident, workflow = workflow_state(
        store, [WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN]
    )
    step = copy_model(
        workflow.official_steps[0],
        execution_owner="gpu-fault-kubernetes-adapter",
        parameters={"expected_count": 16},
    )
    context = WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=3),
        idempotency_key="workflow-active/0/RESTART_EFA_DEVICE_PLUGIN",
    )
    core = EfaCore()
    adapter = KubernetesWorkflowAdapter(
        core_api=core, batch_api=UnusedApi(), custom_api=UnusedApi()
    )

    first = adapter.execute(context)

    assert first.status is WorkflowStepStatus.WAITING
    assert core.deleted == [("kube-system", "efa-plugin-old")]
    core.node["status"]["allocatable"]["vpc.amazonaws.com/efa"] = "16"
    core.pods = [
        {
            "metadata": {"name": "efa-plugin-new", "uid": "pod-new"},
            "status": {"conditions": [{"type": "Ready", "status": "True"}]},
        }
    ]

    second = adapter.execute(context)

    assert second.status is WorkflowStepStatus.SUCCEEDED
    assert second.details["node_results"]["node-a"]["replacement_pod_uids"] == [
        "pod-new"
    ]
    assert not core.node["metadata"]["annotations"]


def test_kubernetes_adapter_waits_when_device_plugin_pod_is_absent() -> None:
    class Core:
        def __init__(self):
            self.node = {
                "metadata": {"resourceVersion": "1", "annotations": {}},
                "status": {"allocatable": {"vpc.amazonaws.com/efa": "0"}},
            }
            self.pods = []

        def read_node(self, _node_id):
            return self.node

        def patch_node(self, _node_id, body):
            for key, value in body["metadata"]["annotations"].items():
                if value is None:
                    self.node["metadata"]["annotations"].pop(key, None)
                else:
                    self.node["metadata"]["annotations"][key] = value

        def list_namespaced_pod(self, *_args, **_kwargs):
            return SimpleNamespace(items=self.pods)

    store = build_store()
    incident, workflow = workflow_state(
        store, [WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN]
    )
    core = Core()
    adapter = KubernetesWorkflowAdapter(
        core_api=core, batch_api=UnusedApi(), custom_api=UnusedApi()
    )
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        parameters={"expected_count": 16},
    )
    context = WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=3),
        idempotency_key="workflow-active/0/RESTART_EFA_DEVICE_PLUGIN",
    )

    first = adapter.execute(context)

    assert first.status is WorkflowStepStatus.WAITING
    assert first.details["node_results"]["node-a"]["waiting_for_daemonset_pod"] is True
    core.node["status"]["allocatable"]["vpc.amazonaws.com/efa"] = "16"
    core.pods = [
        {
            "metadata": {"name": "efa-plugin-new", "uid": "pod-new"},
            "status": {"conditions": [{"type": "Ready", "status": "True"}]},
        }
    ]

    second = adapter.execute(context)

    assert second.status is WorkflowStepStatus.SUCCEEDED
    assert not core.node["metadata"]["annotations"]


def test_kubernetes_adapter_preserves_provider_taint() -> None:
    core = FakeCoreApi()
    adapter = KubernetesWorkflowAdapter(
        core_api=core, batch_api=UnusedApi(), custom_api=UnusedApi()
    )
    store = build_store()
    incident, workflow = workflow_state(
        store,
        [WorkflowOperation.MARK_UNSCHEDULABLE, WorkflowOperation.RESTORE_SCHEDULING],
    )
    steps = [
        copy_model(step, execution_owner=adapter.owner)
        for step in workflow.official_steps
    ]
    workflow = copy_model(workflow, official_steps=steps)
    store.save_workflow(workflow)
    active = active_workflow_executor(
        store,
        [adapter],
        {WorkflowOperation.MARK_UNSCHEDULABLE, WorkflowOperation.RESTORE_SCHEDULING},
    )

    result = execute_workflow(active, workflow.request_id)

    assert result.status is WorkflowStatus.SUCCEEDED
    assert core.node["spec"]["unschedulable"] is False
    assert core.node["spec"]["taints"] == [
        {
            "key": "sagemaker.amazonaws.com/node-health-status",
            "value": "Unschedulable",
            "effect": "NoSchedule",
        }
    ]
    assert incident.incident_id == "incident-active"


def test_restore_removes_stale_quarantine_taint_after_ownership_check() -> None:
    core = FakeCoreApi()
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.RESTORE_SCHEDULING])
    adapter = KubernetesWorkflowAdapter(
        core_api=core, batch_api=UnusedApi(), custom_api=UnusedApi()
    )
    workflow = copy_model(
        workflow,
        official_steps=[
            copy_model(workflow.official_steps[0], execution_owner=adapter.owner)
        ],
    )
    store.save_workflow(workflow)
    core.node["metadata"]["annotations"] = {
        "gpu-fault.io/incident-id": incident.incident_id,
        "gpu-fault.io/fencing-token": str(workflow.fencing_token),
        "gpu-fault.io/previous-unschedulable": "false",
    }
    core.node["spec"]["unschedulable"] = True
    core.node["spec"]["taints"].append(
        {
            "key": "gpu-fault.io/quarantined",
            "value": "incident-no-longer-in-store",
            "effect": "NoSchedule",
        }
    )

    result = execute_workflow(
        active_workflow_executor(
            store, [adapter], {WorkflowOperation.RESTORE_SCHEDULING}
        ),
        workflow.request_id,
    )

    assert result.status is WorkflowStatus.SUCCEEDED
    assert core.node["spec"]["unschedulable"] is False
    assert core.node["spec"]["taints"] == [
        {
            "key": "sagemaker.amazonaws.com/node-health-status",
            "value": "Unschedulable",
            "effect": "NoSchedule",
        }
    ]
    assert core.node["metadata"]["annotations"] == {}


def test_restore_of_a_node_nobody_isolated_is_an_idempotent_success() -> None:
    """A workflow that fail-closed at compile time never ran
    MARK_UNSCHEDULABLE, so the validated restore that closes its incident meets
    a node with no gpu-fault isolation at all. Refusing it left the incident
    ESCALATED forever (2026-09-06, REMEDIATE_EFA_DRIVER without a profile
    owner); there is nothing to undo, so the step succeeds and says so."""

    core = FakeCoreApi()
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.RESTORE_SCHEDULING])
    adapter = KubernetesWorkflowAdapter(
        core_api=core, batch_api=UnusedApi(), custom_api=UnusedApi()
    )
    workflow = copy_model(
        workflow,
        official_steps=[
            copy_model(workflow.official_steps[0], execution_owner=adapter.owner)
        ],
    )
    store.save_workflow(workflow)
    core.node["metadata"]["annotations"] = {}
    core.node["spec"]["unschedulable"] = False
    core.node["spec"]["taints"] = [
        item
        for item in core.node["spec"]["taints"]
        if item.get("key") != "gpu-fault.io/quarantined"
    ]
    before = copy.deepcopy(core.node)

    result = execute_workflow(
        active_workflow_executor(
            store, [adapter], {WorkflowOperation.RESTORE_SCHEDULING}
        ),
        workflow.request_id,
    )

    assert result.status is WorkflowStatus.SUCCEEDED
    # The node was not touched: no taint, annotation or cordon changed.
    assert core.node["spec"] == before["spec"]
    assert core.node["metadata"]["annotations"] == {}
    restore = next(
        item
        for item in store.get_workflow(workflow.request_id).step_executions
        if item.operation is WorkflowOperation.RESTORE_SCHEDULING
    )
    assert restore.details["already_restored_nodes"] == restore.details["restored_nodes"]


def test_restore_still_refuses_a_node_another_incident_isolated() -> None:
    core = FakeCoreApi()
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.RESTORE_SCHEDULING])
    adapter = KubernetesWorkflowAdapter(
        core_api=core, batch_api=UnusedApi(), custom_api=UnusedApi()
    )
    workflow = copy_model(
        workflow,
        official_steps=[
            copy_model(workflow.official_steps[0], execution_owner=adapter.owner)
        ],
    )
    store.save_workflow(workflow)
    core.node["metadata"]["annotations"] = {
        "gpu-fault.io/incident-id": "inc-someone-else",
        "gpu-fault.io/fencing-token": "999",
    }
    core.node["spec"]["unschedulable"] = True

    result = execute_workflow(
        active_workflow_executor(
            store, [adapter], {WorkflowOperation.RESTORE_SCHEDULING}
        ),
        workflow.request_id,
    )

    assert result.status is WorkflowStatus.FAILED
    assert core.node["spec"]["unschedulable"] is True
    assert core.node["metadata"]["annotations"]["gpu-fault.io/incident-id"] == (
        "inc-someone-else"
    )


def test_kubernetes_adapter_preserves_initial_schedulability_across_isolation_steps() -> (
    None
):
    core = FakeCoreApi()
    adapter = KubernetesWorkflowAdapter(
        core_api=core, batch_api=UnusedApi(), custom_api=UnusedApi()
    )
    store = build_store()
    _, workflow = workflow_state(
        store,
        [
            WorkflowOperation.MARK_UNSCHEDULABLE,
            WorkflowOperation.QUARANTINE,
            WorkflowOperation.RESTORE_SCHEDULING,
        ],
    )
    workflow = copy_model(
        workflow,
        official_steps=[
            copy_model(step, execution_owner=adapter.owner)
            for step in workflow.official_steps
        ],
    )
    store.save_workflow(workflow)
    active = active_workflow_executor(
        store,
        [adapter],
        {
            WorkflowOperation.MARK_UNSCHEDULABLE,
            WorkflowOperation.QUARANTINE,
            WorkflowOperation.RESTORE_SCHEDULING,
        },
    )

    result = execute_workflow(active, workflow.request_id)

    assert result.status is WorkflowStatus.SUCCEEDED
    assert core.node["spec"]["unschedulable"] is False
    assert (
        "gpu-fault.io/previous-unschedulable"
        not in core.node["metadata"]["annotations"]
    )


@pytest.mark.parametrize("intermediate_unschedulable", [True, False])
def test_kubernetes_adapter_takes_over_terminal_isolation(
    intermediate_unschedulable: bool,
) -> None:
    core = FakeCoreApi()
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.MARK_UNSCHEDULABLE])
    old_incident = copy_model(
        incident,
        incident_id="incident-terminal",
        workflow_request_id="workflow-terminal",
        state=IncidentState.RECOVERED,
    )
    old_workflow = copy_model(
        workflow,
        request_id="workflow-terminal",
        incident_id="incident-terminal",
        status=WorkflowStatus.SUCCEEDED,
    )
    store.save_incident(old_incident)
    store.save_workflow(old_workflow)
    core.node["metadata"]["annotations"] = {
        "gpu-fault.io/incident-id": "incident-terminal",
        "gpu-fault.io/fencing-token": "1",
        "gpu-fault.io/previous-unschedulable": "false",
    }
    core.node["spec"]["unschedulable"] = intermediate_unschedulable
    core.node["spec"]["taints"].append(
        {
            "key": "gpu-fault.io/quarantined",
            "value": quarantine_taint_value("incident-terminal"),
            "effect": "NoSchedule",
        }
    )
    adapter = KubernetesWorkflowAdapter(
        core_api=core, batch_api=UnusedApi(), custom_api=UnusedApi(), store=store
    )
    workflow = copy_model(
        workflow,
        official_steps=[
            copy_model(workflow.official_steps[0], execution_owner=adapter.owner)
        ],
    )
    store.save_workflow(workflow)
    result = execute_workflow(
        active_workflow_executor(
            store, [adapter], {WorkflowOperation.MARK_UNSCHEDULABLE}
        ),
        workflow.request_id,
    )

    assert result.status is WorkflowStatus.SUCCEEDED
    annotations = core.node["metadata"]["annotations"]
    assert annotations["gpu-fault.io/incident-id"] == (incident.incident_id)
    assert annotations["gpu-fault.io/previous-unschedulable"] == "false"
    quarantine = [
        item
        for item in core.node["spec"]["taints"]
        if item["key"] == "gpu-fault.io/quarantined"
    ]
    assert quarantine == [
        {
            "key": "gpu-fault.io/quarantined",
            "value": quarantine_taint_value(incident.incident_id),
            "effect": "NoSchedule",
        }
    ]


def test_terminal_isolation_takeover_restores_original_schedulability() -> None:
    core = FakeCoreApi()
    store = build_store()
    incident, workflow = workflow_state(
        store,
        [WorkflowOperation.MARK_UNSCHEDULABLE, WorkflowOperation.RESTORE_SCHEDULING],
    )
    old_incident = copy_model(
        incident,
        incident_id="incident-terminal",
        workflow_request_id="workflow-terminal",
        state=IncidentState.RECOVERED,
    )
    old_workflow = copy_model(
        workflow,
        request_id="workflow-terminal",
        incident_id=old_incident.incident_id,
        status=WorkflowStatus.SUCCEEDED,
    )
    store.save_incident(old_incident)
    store.save_workflow(old_workflow)
    core.node["metadata"]["annotations"] = {
        "gpu-fault.io/incident-id": old_incident.incident_id,
        "gpu-fault.io/fencing-token": "1",
        "gpu-fault.io/previous-unschedulable": "false",
    }
    core.node["spec"]["unschedulable"] = True
    core.node["spec"]["taints"].append(
        {
            "key": "gpu-fault.io/quarantined",
            "value": quarantine_taint_value(old_incident.incident_id),
            "effect": "NoSchedule",
        }
    )
    adapter = KubernetesWorkflowAdapter(
        core_api=core, batch_api=UnusedApi(), custom_api=UnusedApi(), store=store
    )
    workflow = copy_model(
        workflow,
        official_steps=[
            copy_model(step, execution_owner=adapter.owner)
            for step in workflow.official_steps
        ],
    )
    store.save_workflow(workflow)

    result = execute_workflow(
        active_workflow_executor(
            store,
            [adapter],
            {
                WorkflowOperation.MARK_UNSCHEDULABLE,
                WorkflowOperation.RESTORE_SCHEDULING,
            },
        ),
        workflow.request_id,
    )

    assert result.status is WorkflowStatus.SUCCEEDED
    assert core.node["spec"]["unschedulable"] is False
    assert core.node["metadata"]["annotations"] == {}
    assert core.node["spec"]["taints"] == [
        {
            "key": "sagemaker.amazonaws.com/node-health-status",
            "value": "Unschedulable",
            "effect": "NoSchedule",
        }
    ]


def test_kubernetes_adapter_rejects_active_isolation_takeover() -> None:
    core = FakeCoreApi()
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.MARK_UNSCHEDULABLE])
    old_incident = copy_model(
        incident, incident_id="incident-running", workflow_request_id="workflow-running"
    )
    old_workflow = copy_model(
        workflow,
        request_id="workflow-running",
        incident_id="incident-running",
        status=WorkflowStatus.RUNNING,
    )
    store.save_incident(old_incident)
    store.save_workflow(old_workflow)
    core.node["metadata"]["annotations"] = {
        "gpu-fault.io/incident-id": "incident-running",
        "gpu-fault.io/fencing-token": "1",
    }
    core.node["spec"]["unschedulable"] = True
    adapter = KubernetesWorkflowAdapter(
        core_api=core, batch_api=UnusedApi(), custom_api=UnusedApi(), store=store
    )
    workflow = copy_model(
        workflow,
        official_steps=[
            copy_model(workflow.official_steps[0], execution_owner=adapter.owner)
        ],
    )
    store.save_workflow(workflow)
    result = execute_workflow(
        active_workflow_executor(
            store, [adapter], {WorkflowOperation.MARK_UNSCHEDULABLE}
        ),
        workflow.request_id,
    )

    assert result.status is WorkflowStatus.FAILED
    assert "already isolated" in (result.error or "")


def test_storeless_adapter_takes_over_terminal_node_isolation() -> None:
    # The regional executor runs with REMOTE_STATE=true and therefore
    # store=None. Before the ownership provider existed the takeover
    # branch was unreachable there, so a node still annotated by a
    # workflow that died before RESTORE_SCHEDULING was refused forever
    # and no later incident could ever isolate it again.
    core = FakeCoreApi()
    core.node["spec"]["unschedulable"] = True
    provider = RecordingOwnershipProvider({"incident-dead": True})
    adapter, context = _storeless_isolation_context(core, provider)

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED
    assert provider.queried == ["incident-dead"]
    annotations = core.node["metadata"]["annotations"]
    assert annotations["gpu-fault.io/incident-id"] == "incident-active"
    assert annotations["gpu-fault.io/fencing-token"] == "3"
    assert annotations["gpu-fault.io/previous-unschedulable"] == "false"
    quarantine = [
        item
        for item in core.node["spec"]["taints"]
        if item["key"] == "gpu-fault.io/quarantined"
    ]
    assert quarantine == [
        {
            "key": "gpu-fault.io/quarantined",
            "value": quarantine_taint_value("incident-active"),
            "effect": "NoSchedule",
        }
    ]


def test_terminal_quarantine_hold_rejects_automatic_reset_takeover() -> None:
    store = build_store()
    old_incident, old_workflow = workflow_state(store, [WorkflowOperation.QUARANTINE])
    old_incident = copy_model(old_incident, state=IncidentState.QUARANTINED)
    old_workflow = copy_model(
        old_workflow,
        status=WorkflowStatus.SUCCEEDED,
        completed_step_indexes=[0],
        completed_operations=[WorkflowOperation.QUARANTINE],
    )
    store.save_incident(old_incident)
    store.save_workflow(old_workflow)
    core = FakeCoreApi()
    core.node["metadata"]["annotations"] = {
        "gpu-fault.io/incident-id": old_incident.incident_id,
        "gpu-fault.io/fencing-token": "3",
        "gpu-fault.io/previous-unschedulable": "false",
    }
    core.node["spec"]["taints"].append(
        {
            "key": "gpu-fault.io/quarantined",
            "value": quarantine_taint_value(old_incident.incident_id),
            "effect": "NoSchedule",
        }
    )
    adapter = KubernetesWorkflowAdapter(
        core_api=core, batch_api=UnusedApi(), custom_api=UnusedApi(), store=store
    )
    reset_incident = copy_model(
        old_incident,
        incident_id="incident-reset-after-quarantine",
        event_id="event-reset-after-quarantine",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="workflow-reset-after-quarantine",
    )
    reset_step = workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE, adapter.owner)
    reset_workflow = copy_model(
        old_workflow,
        request_id="workflow-reset-after-quarantine",
        incident_id=reset_incident.incident_id,
        status=WorkflowStatus.PENDING,
        official_steps=[reset_step],
        completed_step_indexes=[],
        completed_operations=[],
        step_executions=[],
    )
    reset_context = WorkflowStepContext(
        workflow=reset_workflow,
        incident=reset_incident,
        step=reset_step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=3),
        idempotency_key="reset-after-quarantine/mark",
    )

    outcome = adapter.execute(reset_context)
    assert outcome.status is WorkflowStepStatus.FAILED
    assert "already isolated by another incident" in (outcome.error or "")
    assert outcome.details["safety_rejection"] is True
    assert (
        core.node["metadata"]["annotations"]["gpu-fault.io/incident-id"]
        == old_incident.incident_id
    )
    assert any(
        item.get("key") == "gpu-fault.io/quarantined"
        and item.get("value") == quarantine_taint_value(old_incident.incident_id)
        for item in core.node["spec"]["taints"]
    )

    quarantine_incident = copy_model(
        reset_incident,
        incident_id="incident-new-quarantine",
        event_id="event-new-quarantine",
        workflow_request_id="workflow-new-quarantine",
    )
    quarantine_workflow = copy_model(
        reset_workflow,
        request_id="workflow-new-quarantine",
        incident_id=quarantine_incident.incident_id,
        official_steps=[
            reset_step,
            workflow_step(WorkflowOperation.QUARANTINE, adapter.owner),
        ],
    )
    quarantine_context = WorkflowStepContext(
        workflow=quarantine_workflow,
        incident=quarantine_incident,
        step=reset_step,
        step_index=0,
        request=reset_context.request,
        idempotency_key="new-quarantine/mark",
    )

    accepted = adapter.execute(quarantine_context)

    assert accepted.status is WorkflowStepStatus.SUCCEEDED
    assert (
        core.node["metadata"]["annotations"]["gpu-fault.io/incident-id"]
        == quarantine_incident.incident_id
    )


def test_same_incident_stale_workload_generation_is_rejected() -> None:
    core = FakeCoreApi()
    adapter, context = _storeless_isolation_context(core, None)

    with pytest.raises(ValueError, match="newer workflow generation"):
        adapter._operation_already_applied(
            {
                "gpu-fault.io/incident-id": "incident-active",
                "gpu-fault.io/fencing-token": "4",
                "gpu-fault.io/execution-epoch": "1",
                "gpu-fault.io/workflow-step-index": "0",
            },
            context,
        )


def test_kubernetes_adapter_retries_node_isolation_conflict() -> None:
    class ConflictCore(FakeCoreApi):
        def __init__(self) -> None:
            super().__init__()
            self.patch_calls = 0

        def patch_node(self, node_id, body):
            self.patch_calls += 1
            if self.patch_calls == 1:
                self.node["metadata"]["resourceVersion"] = "2"
                error = RuntimeError("node was modified")
                error.status = 409
                raise error
            return super().patch_node(node_id, body)

    core = ConflictCore()
    adapter = KubernetesWorkflowAdapter(
        core_api=core, batch_api=UnusedApi(), custom_api=UnusedApi()
    )
    store = build_store()
    _, workflow = workflow_state(store, [WorkflowOperation.MARK_UNSCHEDULABLE])
    workflow = copy_model(
        workflow,
        official_steps=[
            copy_model(workflow.official_steps[0], execution_owner=adapter.owner)
        ],
    )
    store.save_workflow(workflow)
    result = execute_workflow(
        active_workflow_executor(
            store, [adapter], {WorkflowOperation.MARK_UNSCHEDULABLE}
        ),
        workflow.request_id,
    )

    assert result.status is WorkflowStatus.SUCCEEDED
    assert core.patch_calls == 2
    assert core.node["spec"]["unschedulable"] is True


def test_kubernetes_adapter_treats_absent_node_as_isolated() -> None:
    class MissingCore(FakeCoreApi):
        def read_node(self, _node_id):
            error = KeyError("node is gone")
            error.status = 404
            raise error

    adapter = KubernetesWorkflowAdapter(
        core_api=MissingCore(), batch_api=UnusedApi(), custom_api=UnusedApi()
    )
    store = build_store()
    _, workflow = workflow_state(store, [WorkflowOperation.MARK_UNSCHEDULABLE])
    workflow = copy_model(
        workflow,
        official_steps=[
            copy_model(workflow.official_steps[0], execution_owner=adapter.owner)
        ],
    )
    store.save_workflow(workflow)
    result = execute_workflow(
        active_workflow_executor(
            store, [adapter], {WorkflowOperation.MARK_UNSCHEDULABLE}
        ),
        workflow.request_id,
    )

    assert result.status is WorkflowStatus.SUCCEEDED
    execution = store.get_workflow(workflow.request_id).step_executions[0]
    assert execution.details["already_absent_nodes"] == ["node-a"]


def test_kubernetes_adapter_hashes_long_incident_id_for_taint() -> None:
    core = FakeCoreApi()
    adapter = KubernetesWorkflowAdapter(
        core_api=core, batch_api=UnusedApi(), custom_api=UnusedApi()
    )
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.MARK_UNSCHEDULABLE])
    long_incident_id = "inc-kernel-log-kmsg-boot-id-5408-xid-11-" + "x" * 40
    incident = copy_model(incident, incident_id=long_incident_id)
    workflow = copy_model(
        workflow,
        incident_id=long_incident_id,
        official_steps=[
            copy_model(workflow.official_steps[0], execution_owner=adapter.owner)
        ],
    )
    store.save_incident(incident)
    store.save_workflow(workflow)
    active = active_workflow_executor(
        store, [adapter], {WorkflowOperation.MARK_UNSCHEDULABLE}
    )

    result = execute_workflow(active, workflow.request_id)

    assert result.status is WorkflowStatus.SUCCEEDED
    quarantine = [
        item
        for item in core.node["spec"]["taints"]
        if item["key"] == "gpu-fault.io/quarantined"
    ]
    assert len(quarantine) == 1
    assert quarantine[0]["value"].startswith("incident-")
    assert len(quarantine[0]["value"]) <= 63
    assert (
        core.node["metadata"]["annotations"]["gpu-fault.io/incident-id"]
        == long_incident_id
    )


def test_kubernetes_stop_waits_for_job_to_be_inactive() -> None:
    store = build_store()
    operations = [WorkflowOperation.STOP_WORKLOADS, WorkflowOperation.RESTART_WORKLOAD]
    _, workflow = workflow_state(store, operations)
    batch = FakeBatchApi()
    adapter = KubernetesWorkflowAdapter(
        core_api=UnusedApi(), batch_api=batch, custom_api=UnusedApi(), store=store
    )
    steps = [
        copy_model(
            step,
            execution_owner=adapter.owner,
            workload_ids=["gpu-fault-system/job/training-job"],
            parameters={
                "cluster_id": "cluster-a",
                "job_id": "training-job",
                "source_attempt_id": "attempt-a",
                "source_gpu_count": 1,
                "restart_budget": 1,
            }
            if step.operation is WorkflowOperation.RESTART_WORKLOAD
            else {},
        )
        for step in workflow.official_steps
    ]
    workflow = copy_model(workflow, official_steps=steps)
    store.save_workflow(workflow)
    active = active_workflow_executor(store, [adapter], operations)
    request = WorkflowExecutionRequest(expected_fencing_token=3)

    waiting = active.execute(workflow.request_id, request)
    batch.active = 0
    batch.terminating = 1
    still_waiting = active.execute(workflow.request_id, request)
    batch.terminating = 0
    completed = active.execute(workflow.request_id, request)

    assert waiting.status is WorkflowStatus.RUNNING
    assert waiting.waiting_step_index == 0
    assert still_waiting.status is WorkflowStatus.RUNNING
    assert completed.status is WorkflowStatus.SUCCEEDED
    assert batch.suspend_patches == [True]
    assert len(batch.created) == 1


def test_kubernetes_stop_deletes_active_suspended_pytorch_pods() -> None:
    class Core:
        def __init__(self) -> None:
            self.pods = [
                {
                    "metadata": {
                        "name": f"worker-{index}",
                        "namespace": "gpu-fault-system",
                    }
                }
                for index in range(3)
            ]
            self.patched = []
            self.deleted = []

        def list_namespaced_pod(self, *_args, **_kwargs):
            return SimpleNamespace(items=list(self.pods))

        def patch_namespaced_pod(self, name, namespace, body):
            self.patched.append((namespace, name, body))

        def delete_namespaced_pod(self, name, namespace, grace_period_seconds):
            self.deleted.append((namespace, name, grace_period_seconds))
            self.pods = [pod for pod in self.pods if pod["metadata"]["name"] != name]

    class Custom:
        def __init__(self) -> None:
            self.workload = {
                "metadata": {
                    "resourceVersion": "1",
                    "labels": {
                        "gpu-fault.io/job-id": "training-job",
                        "gpu-fault.io/attempt-id": "attempt-a",
                    },
                    "annotations": {},
                },
                "spec": {"runPolicy": {"suspend": False}},
                "status": {
                    "conditions": [{"type": "Suspended", "status": "True"}],
                    "replicaStatuses": {
                        "Master": {"active": 0},
                        "Worker": {"active": 2},
                    },
                },
            }

        def get_namespaced_custom_object(self, *_args):
            return self.workload

        def patch_namespaced_custom_object(
            self, _group, _version, _namespace, _plural, _name, body
        ):
            self.workload["metadata"]["annotations"].update(
                body["metadata"]["annotations"]
            )
            self.workload["spec"] = body["spec"]

    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.STOP_WORKLOADS])
    core = Core()
    custom = Custom()
    adapter = KubernetesWorkflowAdapter(
        core_api=core, batch_api=UnusedApi(), custom_api=custom, store=store
    )
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        workload_ids=["gpu-fault-system/pytorchjob/training-job"],
        parameters={"termination_initiator_incident_id": incident.incident_id},
    )
    workflow = copy_model(workflow, official_steps=[step])
    context = WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
        idempotency_key="workflow-stop/0/STOP_WORKLOADS",
    )

    waiting = adapter.execute(context)

    assert waiting.status is WorkflowStepStatus.WAITING
    assert len(core.patched) == 3
    assert len(core.deleted) == 3
    assert waiting.details["deleted_pods"] == [
        "gpu-fault-system/worker-0",
        "gpu-fault-system/worker-1",
        "gpu-fault-system/worker-2",
    ]

    # The PyTorch operator may leave replicaStatuses.active stale after
    # the Pods have already disappeared. Pod absence is authoritative.
    waiting_workflow = copy_model(
        workflow,
        step_executions=[
            workflow_step_execution(
                0,
                WorkflowOperation.STOP_WORKLOADS,
                WorkflowStepStatus.WAITING,
                adapter_operation_id=waiting.adapter_operation_id,
                details=waiting.details,
            )
        ],
    )
    completed = adapter.execute(
        context.__class__(
            workflow=waiting_workflow,
            incident=incident,
            step=step,
            step_index=0,
            request=context.request,
            idempotency_key=context.idempotency_key,
        )
    )

    assert completed.status is WorkflowStepStatus.SUCCEEDED
    assert completed.details["attempt_pods_absent"] is True
