from __future__ import annotations

from gpu_fault.hyperpod import HyperPodSubmissionRecord
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    execute_workflow,
    node_action_result,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

from ._support import (
    FakeAdapter,
    HyperPodAction,
    IncidentState,
    ManagedRecoveryObserverAdapter,
    NodeActionStatus,
    NodeActionWorkflowAdapter,
    StubFleetRegistry,
    WorkflowDispatcher,
    WorkflowDispatcherConfig,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepContext,
    WorkflowStepOutcome,
    WorkflowStepStatus,
    _preempting_successor,
    datetime,
    executor,
    pytest,
    threading,
    time,
    timedelta,
    timezone,
    workflow_state,
)


def test_failed_reset_runs_restore_compensation_before_terminal() -> None:
    store = build_store()
    operations = [
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESTORE_GPU_SERVICES,
    ]
    incident, workflow = workflow_state(store, operations)
    adapter = FakeAdapter(
        {
            WorkflowOperation.QUIESCE_GPU_SERVICES: (WorkflowStepOutcome.succeeded()),
            WorkflowOperation.RESET_GPU: (WorkflowStepOutcome.failed("reset rejected")),
            WorkflowOperation.RESTORE_GPU_SERVICES: (WorkflowStepOutcome.succeeded()),
        }
    )
    active = active_workflow_executor(store, [adapter], operations)

    result = execute_workflow(active, workflow.request_id)

    saved = store.get_workflow(workflow.request_id)
    assert result.status is WorkflowStatus.FAILED
    assert adapter.calls == [
        "workflow-active/0/QUIESCE_GPU_SERVICES",
        "workflow-active/1/RESET_GPU",
        "workflow-active/2/RESTORE_GPU_SERVICES",
    ]
    assert WorkflowOperation.RESTORE_GPU_SERVICES in (saved.completed_operations)
    assert saved.pending_failure_step_index is None
    assert saved.pending_failure_error is None
    assert result.error == "reset rejected"


def test_reset_to_reboot_hands_off_quiesce_and_skips_low_reset() -> None:
    store = build_store()
    operations = [
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESTORE_GPU_SERVICES,
    ]
    incident, workflow = workflow_state(store, operations)
    workflow = copy_model(
        workflow,
        status=WorkflowStatus.RUNNING,
        completed_step_indexes=[0],
        completed_operations=[WorkflowOperation.QUIESCE_GPU_SERVICES],
        step_executions=[
            workflow_step_execution(
                0,
                WorkflowOperation.QUIESCE_GPU_SERVICES,
                details={
                    "agent_generations": {"node-a": 2},
                    "maintenance_window_expires_at": (
                        datetime.now(timezone.utc) + timedelta(minutes=5)
                    ).isoformat(),
                },
            )
        ],
    )
    store.save_workflow(workflow)
    successor = _preempting_successor(
        store, incident, workflow, WorkflowOperation.RESTART_NODE
    )
    adapter = FakeAdapter(
        {
            WorkflowOperation.RESET_GPU: (WorkflowStepOutcome.succeeded()),
            WorkflowOperation.RESTART_NODE: (
                WorkflowStepOutcome.waiting(operation_id="reboot-submitted")
            ),
            WorkflowOperation.RESTORE_GPU_SERVICES: (WorkflowStepOutcome.succeeded()),
        }
    )
    active = active_workflow_executor(
        store, [adapter], [*operations, WorkflowOperation.RESTART_NODE]
    )

    result = execute_workflow(active, workflow.request_id)

    assert result.status is WorkflowStatus.SUPERSEDED
    assert adapter.calls == []
    assert [
        step.operation
        for step in store.get_workflow(successor.request_id).official_steps
    ] == [WorkflowOperation.RESTART_NODE]

    handoff = execute_workflow(active, successor.request_id)
    updated_successor = store.get_workflow(successor.request_id)
    assert handoff.status is WorkflowStatus.RUNNING
    assert [step.operation for step in updated_successor.official_steps] == [
        WorkflowOperation.RESTART_NODE,
        WorkflowOperation.RESTORE_GPU_SERVICES,
    ]
    assert updated_successor.official_steps[1].parameters[
        "preemption_quiesce_handoff_after_reboot"
    ]
    assert updated_successor.official_steps[1].depends_on_step_indexes == [0]
    assert updated_successor.quiesce_handoff_from_workflow_id == workflow.request_id


def test_submitted_reset_restores_before_reboot_preemption() -> None:
    store = build_store()
    operations = [
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESTORE_GPU_SERVICES,
        WorkflowOperation.VALIDATE_GPU,
    ]
    incident, workflow = workflow_state(store, operations)
    workflow = copy_model(
        workflow,
        status=WorkflowStatus.RUNNING,
        completed_step_indexes=[0, 1],
        completed_operations=operations[:2],
        step_executions=[
            workflow_step_execution(index, operations[index]) for index in (0, 1)
        ]
        + [
            workflow_step_execution(
                2,
                WorkflowOperation.RESTORE_GPU_SERVICES,
                WorkflowStepStatus.WAITING,
                adapter_operation_id="remote/restore-leased",
                details={
                    "remote_status": "LEASED",
                    "remote_command_id": "restore-leased",
                },
            )
        ],
    )
    store.save_workflow(workflow)
    successor = _preempting_successor(
        store, incident, workflow, WorkflowOperation.RESTART_NODE
    )
    adapter = FakeAdapter(
        {
            WorkflowOperation.RESTORE_GPU_SERVICES: (WorkflowStepOutcome.succeeded()),
            WorkflowOperation.VALIDATE_GPU: (WorkflowStepOutcome.succeeded()),
            WorkflowOperation.RESTART_NODE: (WorkflowStepOutcome.succeeded()),
        }
    )
    active = active_workflow_executor(
        store, [adapter], {*operations, WorkflowOperation.RESTART_NODE}
    )

    predecessor_result = execute_workflow(active, workflow.request_id)
    successor_result = execute_workflow(active, successor.request_id)

    predecessor = store.get_workflow(workflow.request_id)
    assert predecessor_result.status is WorkflowStatus.SUPERSEDED
    assert predecessor.completed_step_indexes == [0, 1, 2]
    assert WorkflowOperation.VALIDATE_GPU not in (predecessor.completed_operations)
    assert (
        sum(
            execution.operation is WorkflowOperation.RESET_GPU
            for execution in predecessor.step_executions
        )
        == 1
    )
    assert successor_result.status is WorkflowStatus.SUCCEEDED
    assert adapter.calls == [
        "workflow-active/2/RESTORE_GPU_SERVICES",
        "workflow-successor/0/RESTART_NODE",
        "workflow-successor/1/RESTORE_GPU_SERVICES",
    ]
    assert adapter.calls.count("workflow-successor/0/RESTART_NODE") == 1


def test_submitted_reboot_finishes_before_replace_successor() -> None:
    store = build_store()
    incident, workflow = workflow_state(
        store, [WorkflowOperation.RESTART_NODE, WorkflowOperation.VALIDATE_GPU]
    )
    store.save_hyperpod_submission(
        HyperPodSubmissionRecord(
            cluster_name="cluster-a",
            idempotency_key="reboot-submitted",
            action=HyperPodAction.REBOOT,
            requested_node_identifiers=["node-a"],
            state="SUBMITTED",
        )
    )
    workflow = copy_model(
        workflow,
        status=WorkflowStatus.RUNNING,
        step_executions=[
            workflow_step_execution(
                0,
                WorkflowOperation.RESTART_NODE,
                WorkflowStepStatus.WAITING,
                adapter_operation_id="remote/reboot-leased",
                details={
                    "remote_status": "LEASED",
                    "remote_command_id": "reboot-leased",
                },
            )
        ],
    )
    store.save_workflow(workflow)
    successor = _preempting_successor(
        store, incident, workflow, WorkflowOperation.REPLACE_NODE
    )
    adapter = FakeAdapter(
        {
            WorkflowOperation.RESTART_NODE: (
                WorkflowStepOutcome.waiting(operation_id="remote/reboot-leased")
            ),
            WorkflowOperation.VALIDATE_GPU: (WorkflowStepOutcome.succeeded()),
            WorkflowOperation.REPLACE_NODE: (
                WorkflowStepOutcome.succeeded(operation_id="warm-spare/simulated")
            ),
        }
    )
    active = active_workflow_executor(
        store,
        [adapter],
        {
            WorkflowOperation.RESTART_NODE,
            WorkflowOperation.VALIDATE_GPU,
            WorkflowOperation.REPLACE_NODE,
        },
    )

    still_running = execute_workflow(active, workflow.request_id)
    adapter.outcomes[WorkflowOperation.RESTART_NODE] = WorkflowStepOutcome.succeeded(
        operation_id="remote/reboot-leased"
    )
    superseded = execute_workflow(active, workflow.request_id)
    replacement = execute_workflow(active, successor.request_id)

    submission = store.get_hyperpod_submission("cluster-a", "reboot-submitted")
    assert still_running.status is WorkflowStatus.RUNNING
    assert superseded.status is WorkflowStatus.SUPERSEDED
    assert WorkflowOperation.VALIDATE_GPU not in (
        store.get_workflow(workflow.request_id).completed_operations
    )
    assert submission.state == "SUBMITTED"
    assert replacement.status is WorkflowStatus.SUCCEEDED
    assert adapter.calls == [
        "workflow-active/0/RESTART_NODE",
        "workflow-active/0/RESTART_NODE",
        "workflow-successor/0/REPLACE_NODE",
    ]


def test_reset_all_successor_inherits_quiesce_but_not_verify() -> None:
    store = build_store()
    operations = [
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESTORE_GPU_SERVICES,
    ]
    incident, workflow = workflow_state(store, operations)
    workflow = copy_model(
        workflow,
        status=WorkflowStatus.RUNNING,
        completed_step_indexes=[0],
        completed_operations=[WorkflowOperation.QUIESCE_GPU_SERVICES],
        step_executions=[
            workflow_step_execution(
                0,
                WorkflowOperation.QUIESCE_GPU_SERVICES,
                details={
                    "agent_generations": {"node-a": 2},
                    "maintenance_window_expires_at": (
                        datetime.now(timezone.utc) + timedelta(minutes=5)
                    ).isoformat(),
                },
            )
        ],
    )
    store.save_workflow(workflow)
    successor = workflow_request(
        "workflow-successor",
        incident.incident_id,
        fencing_token=workflow.fencing_token,
        predecessor_workflow_id=workflow.request_id,
        preempt_predecessor=True,
        preemption_reason="rank 30 -> 40",
        runtime_profile_version="active-v1",
        official_action="RESET_ALL_GPUS_AND_NVSWITCHES",
        official_steps=[
            workflow_step(operation, gpu_uuids=["GPU-a", "GPU-b"])
            for operation in [
                WorkflowOperation.QUIESCE_GPU_SERVICES,
                WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
                WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
                WorkflowOperation.RESTORE_GPU_SERVICES,
            ]
        ],
    )
    store.save_workflow(successor)
    store.save_incident(copy_model(incident, workflow_request_id=successor.request_id))
    adapter = FakeAdapter(
        {
            WorkflowOperation.QUIESCE_GPU_SERVICES: (WorkflowStepOutcome.succeeded()),
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS: (WorkflowStepOutcome.succeeded()),
            WorkflowOperation.RESET_GPU: (WorkflowStepOutcome.succeeded()),
            WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES: (
                WorkflowStepOutcome.succeeded()
            ),
            WorkflowOperation.RESTORE_GPU_SERVICES: (WorkflowStepOutcome.succeeded()),
        }
    )
    active = active_workflow_executor(
        store,
        [adapter],
        {
            *operations,
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        },
    )

    result = execute_workflow(active, workflow.request_id)

    assert result.status is WorkflowStatus.SUPERSEDED
    assert store.get_workflow(successor.request_id).completed_step_indexes == []
    completed = execute_workflow(active, successor.request_id)
    assert completed.status is WorkflowStatus.SUCCEEDED
    updated = store.get_workflow(successor.request_id)
    assert updated.completed_step_indexes == [0, 1, 2, 3]
    assert updated.inherited_step_indexes == [0]
    assert updated.completed_operations == [
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        WorkflowOperation.RESTORE_GPU_SERVICES,
    ]
    assert updated.step_executions[0].details["preemption_quiesce_handoff"]
    assert 1 not in updated.inherited_step_indexes


@pytest.mark.parametrize(
    ("expected_status", "step_outcome", "preemption_enabled"),
    [
        (WorkflowStatus.SUCCEEDED, WorkflowStepOutcome.succeeded(), False),
        (WorkflowStatus.FAILED, WorkflowStepOutcome.failed("freeze rejected"), False),
        (WorkflowStatus.SUPERSEDED, WorkflowStepOutcome.succeeded(), True),
    ],
    ids=["succeeded", "failed", "superseded"],
)
def test_predecessor_terminal_save_preserves_successor_incident_pointer(
    expected_status: WorkflowStatus,
    step_outcome: WorkflowStepOutcome,
    preemption_enabled: bool,
) -> None:
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.FREEZE_EVIDENCE])
    successor = _preempting_successor(store, incident, workflow)
    adapter = FakeAdapter({WorkflowOperation.FREEZE_EVIDENCE: step_outcome})
    active = active_workflow_executor(
        store,
        [adapter],
        {WorkflowOperation.FREEZE_EVIDENCE},
        workflow_preemption_enabled=preemption_enabled,
    )

    result = execute_workflow(active, workflow.request_id)

    assert result.status is expected_status
    assert store.get_workflow(workflow.request_id).status is expected_status
    current_incident = store.get_incident(incident.incident_id)
    assert current_incident.workflow_request_id == successor.request_id
    assert current_incident.fencing_token == successor.fencing_token
    assert current_incident.state is IncidentState.ACTION_PENDING


def test_queued_successor_does_not_fence_running_predecessor() -> None:
    store = build_store()
    operation = WorkflowOperation.FREEZE_EVIDENCE
    incident, predecessor = workflow_state(store, [operation])
    now = datetime.now(timezone.utc)
    predecessor = copy_model(
        predecessor,
        status=WorkflowStatus.RUNNING,
        execution_owner_id="expired-executor",
        execution_epoch=1,
        execution_lease_expires_at=now - timedelta(seconds=1),
    )
    successor = copy_model(
        predecessor,
        request_id="workflow-successor",
        predecessor_workflow_id=predecessor.request_id,
        status=WorkflowStatus.PENDING,
        fencing_token=predecessor.fencing_token + 1,
        execution_owner_id=None,
        execution_epoch=0,
        execution_lease_expires_at=None,
        created_at=now,
        updated_at=now,
    )
    incident = copy_model(
        incident,
        workflow_request_id=successor.request_id,
        fencing_token=successor.fencing_token,
    )
    store.save_incident(incident)
    store.save_workflow(predecessor)
    store.save_workflow(successor)
    adapter = FakeAdapter({operation: WorkflowStepOutcome.succeeded()})
    dispatcher = WorkflowDispatcher(
        store,
        executor(store, adapter, [operation]),
        WorkflowDispatcherConfig(enabled=True),
    )

    first = dispatcher.run_once()
    second = dispatcher.run_once()

    assert first.completed == 1
    assert second.completed == 1
    assert store.get_workflow(predecessor.request_id).status is (
        WorkflowStatus.SUCCEEDED
    )
    assert store.get_workflow(successor.request_id).status is (WorkflowStatus.SUCCEEDED)


def test_managed_recovery_observer_never_submits_mutation() -> None:
    store = build_store()
    _, workflow = workflow_state(store, [WorkflowOperation.RESTART_NODE])
    owner = "hyperpod-managed-node-recovery"
    step = copy_model(workflow.official_steps[0], execution_owner=owner)
    workflow = copy_model(workflow, official_steps=[step])
    store.save_workflow(workflow)
    adapter = ManagedRecoveryObserverAdapter({owner})
    active = active_workflow_executor(
        store, [adapter], {WorkflowOperation.RESTART_NODE}
    )

    waiting = execute_workflow(active, workflow.request_id)
    operation_id = (
        store.get_workflow(workflow.request_id).step_executions[0].adapter_operation_id
    )
    completed = execute_workflow(
        active, workflow.request_id, confirmed_adapter_operation_ids=[operation_id]
    )

    assert waiting.status is WorkflowStatus.RUNNING
    assert operation_id.startswith("delegated/"), (
        'expected operation_id.startswith("delegated/") to be true'
    )
    assert completed.status is WorkflowStatus.SUCCEEDED


def test_managed_recovery_rejects_warm_spare_replacement() -> None:
    store = build_store()
    _, workflow = workflow_state(store, [WorkflowOperation.REPLACE_NODE])
    owner = "hyperpod-managed-node-recovery"
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=owner,
        parameters={"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"},
    )
    store.save_workflow(copy_model(workflow, official_steps=[step]))
    adapter = ManagedRecoveryObserverAdapter({owner})
    active = active_workflow_executor(
        store, [adapter], {WorkflowOperation.REPLACE_NODE}
    )

    result = execute_workflow(active, workflow.request_id)

    assert result.status is WorkflowStatus.FAILED
    assert result.error == (
        "healthy warm-spare replacement cannot be delegated "
        "to managed/provider node recovery"
    )


@pytest.mark.parametrize(
    ("operation", "expected_status", "expected_sends"),
    [
        (WorkflowOperation.RESTORE_GPU_SERVICES, WorkflowStepStatus.SUCCEEDED, 1),
        (WorkflowOperation.RESET_GPU, WorkflowStepStatus.FAILED, 0),
    ],
)
def test_expired_quiesce_allows_restore_but_not_hardware_action(
    operation: WorkflowOperation,
    expected_status: WorkflowStepStatus,
    expected_sends: int,
) -> None:
    store = build_store()
    incident, workflow = workflow_state(
        store, [WorkflowOperation.QUIESCE_GPU_SERVICES, operation]
    )
    steps = [
        copy_model(
            step,
            execution_owner="gpu-fault-node-agent",
            node_ids=["node-a"],
            gpu_uuids=["GPU-a"],
        )
        for step in workflow.official_steps
    ]
    expired_at = datetime(2026, 8, 11, 3, 10, tzinfo=timezone.utc)
    workflow = copy_model(
        workflow,
        official_steps=steps,
        step_executions=[
            workflow_step_execution(
                0,
                WorkflowOperation.QUIESCE_GPU_SERVICES,
                details={
                    "agent_generations": {"node-a": 7},
                    "maintenance_window_expires_at": expired_at.isoformat(),
                },
            )
        ],
    )
    registry = StubFleetRegistry({"node-a": "http://node-a:9099"}, generation=7)
    registry.now = lambda: expired_at + timedelta(minutes=30)
    sent = []

    def sender(_, envelope):
        sent.append(envelope.command.operation)
        return node_action_result(
            envelope.command.command_id,
            envelope.command.operation,
            details={"already_restored": True},
        )

    outcome = NodeActionWorkflowAdapter(
        {}, "s" * 32, sender=sender, registry=registry
    ).execute(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=steps[1],
            step_index=1,
            request=WorkflowExecutionRequest(expected_fencing_token=3),
            idempotency_key=f"workflow/expired/{operation.value}",
        )
    )

    assert outcome.status is expected_status
    assert len(sent) == expected_sends
    if operation is WorkflowOperation.RESET_GPU:
        assert "maintenance window expired" in (outcome.error or "")


def test_post_reboot_handoff_restore_uses_fresh_agent_generation() -> None:
    store = build_store()
    incident, workflow = workflow_state(
        store,
        [
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            WorkflowOperation.RESTART_NODE,
            WorkflowOperation.RESTORE_GPU_SERVICES,
        ],
    )
    steps = [
        copy_model(step, execution_owner="gpu-fault-node-agent", node_ids=["node-a"])
        for step in workflow.official_steps
    ]
    steps[2] = copy_model(
        steps[2], parameters={"preemption_quiesce_handoff_after_reboot": True}
    )
    workflow = copy_model(
        workflow,
        official_steps=steps,
        completed_step_indexes=[0, 1],
        completed_operations=[
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            WorkflowOperation.RESTART_NODE,
        ],
        step_executions=[
            workflow_step_execution(
                0,
                WorkflowOperation.QUIESCE_GPU_SERVICES,
                details={
                    "agent_generations": {"node-a": 7},
                    "maintenance_window_expires_at": (
                        datetime.now(timezone.utc) - timedelta(minutes=1)
                    ).isoformat(),
                },
            ),
            workflow_step_execution(1, WorkflowOperation.RESTART_NODE),
        ],
    )
    registry = StubFleetRegistry({"node-a": "http://node-a:9099"}, generation=8)
    captured = []

    def sender(_, envelope):
        captured.append(envelope.command.agent_generation)
        return node_action_result(
            envelope.command.command_id,
            envelope.command.operation,
            details={"already_restored": True},
        )

    outcome = NodeActionWorkflowAdapter(
        {}, "s" * 32, sender=sender, registry=registry
    ).execute(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=steps[2],
            step_index=2,
            request=WorkflowExecutionRequest(expected_fencing_token=3),
            idempotency_key="workflow/reboot-handoff/restore",
        )
    )

    assert outcome.status is WorkflowStepStatus.SUCCEEDED
    assert captured == [8]


def test_knows_node_prefers_registry_over_stale_endpoint_map() -> None:
    """Alias selection must not consult a stale map when a registry exists.

    _spare_gpu_client_reasons picks which provider alias to address by
    asking the adapter which names it knows. Reading the static map
    there would pick a decommissioned alias, or none at all once the map
    went stale.
    """
    registry = StubFleetRegistry({"node-live": "http://live:9099"})
    adapter = NodeActionWorkflowAdapter(
        {"node-stale": "http://stale:9099"}, "s" * 32, registry=registry
    )

    assert adapter.knows_node("cluster-a", "node-live") is True
    # Unknown to the registry, and the stale map must not rescue it.
    assert adapter.knows_node("cluster-a", "node-stale") is False

    without_registry = NodeActionWorkflowAdapter(
        {"node-stale": "http://stale:9099"}, "s" * 32
    )
    assert without_registry.knows_node("cluster-a", "node-stale") is True


def test_registry_addressing_fences_on_agent_generation() -> None:
    """Registry addressing keeps generation fencing intact."""
    registry = StubFleetRegistry({"node-a": "http://live-a:9099"}, generation=7)
    adapter = NodeActionWorkflowAdapter({}, "s" * 32, registry=registry)

    assert adapter._endpoint("cluster-a", "node-a", 7) == "http://live-a:9099"
    with pytest.raises(ValueError, match="generation changed"):
        adapter._endpoint("cluster-a", "node-a", 6)

    # Quiesced agents stop heartbeating, so maintenance addressing must
    # bypass readiness but still fence on generation.
    assert (
        adapter._endpoint("cluster-a", "node-a", 7, maintenance=True)
        == "http://live-a:9099"
    )
    assert registry.maintenance_calls == [("cluster-a", "node-a", 7)]


def test_hung_bundle_fans_out_to_all_attempt_nodes_after_failure() -> None:
    store = build_store()
    incident, workflow = workflow_state(
        store, [WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE]
    )
    captured = []

    def sender(endpoint, envelope):
        captured.append((endpoint, envelope.command))
        if envelope.command.node_id == "node-a":
            return node_action_result(
                envelope.command.command_id,
                envelope.command.operation,
                NodeActionStatus.FAILED,
                error="strace unavailable",
            )
        return node_action_result(
            envelope.command.command_id,
            envelope.command.operation,
            details={"evidence_ref": "s3://evidence/node-b/bundle.tgz"},
        )

    step = copy_model(
        workflow.official_steps[0],
        execution_owner="gpu-fault-node-agent",
        node_ids=["node-a", "node-b"],
        parameters={
            "diagnostic_reason": "EFA_TRAFFIC_HUNG_SUSPECTED",
            "capture_process_state": True,
            "gpu_uuids_by_node": {"node-a": ["GPU-a"], "node-b": ["GPU-b"]},
        },
    )
    outcome = NodeActionWorkflowAdapter(
        {"node-a": "http://node-a:9099", "node-b": "http://node-b:9099"},
        "s" * 32,
        sender=sender,
    ).execute(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=step,
            step_index=0,
            request=WorkflowExecutionRequest(expected_fencing_token=3),
            idempotency_key="workflow/hung/all-nodes",
        )
    )

    assert outcome.status is WorkflowStepStatus.FAILED
    # Read-only bundle collection dispatches concurrently, so assert on
    # the set of commands rather than their arrival order.
    gpu_uuids_by_node = {command.node_id: command.gpu_uuids for _, command in captured}
    assert gpu_uuids_by_node == {"node-a": ["GPU-a"], "node-b": ["GPU-b"]}
    assert outcome.details["failed_nodes"] == ["node-a"]
    assert "node-b" in outcome.details["node_results"]


def test_hung_triage_tolerates_unreachable_node() -> None:
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.COLLECT_HUNG_TRIAGE])

    def sender(endpoint, envelope):
        if envelope.command.node_id == "node-b":
            raise OSError("node unreachable")
        return node_action_result(
            envelope.command.command_id,
            envelope.command.operation,
            details={"ranks": [{"rank": 0, "pid": 100}]},
        )

    step = copy_model(
        workflow.official_steps[0],
        execution_owner="gpu-fault-node-agent",
        node_ids=["node-a", "node-b"],
        parameters={"triage_timeout_seconds": 10},
    )
    outcome = NodeActionWorkflowAdapter(
        {"node-a": "http://node-a:9099", "node-b": "http://node-b:9099"},
        "s" * 32,
        sender=sender,
    ).execute(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=step,
            step_index=0,
            request=WorkflowExecutionRequest(expected_fencing_token=3),
            idempotency_key="workflow/hung/triage",
        )
    )

    assert outcome.status is WorkflowStepStatus.SUCCEEDED
    assert outcome.details["triage_completed_nodes"] == ["node-a"]
    assert outcome.details["undetermined_nodes"] == ["node-b"]
    assert outcome.details["node_results"]["node-a"]["ranks"][0]["rank"] == 0


def test_hung_triage_dispatches_nodes_concurrently() -> None:
    # Serial dispatch cost the agent-side triage timeout per node, so a
    # wide attempt could not be sampled inside one step lease and every
    # rank stayed stopped for the whole sweep. Eight nodes that each
    # block for 0.2 s must overlap.
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.COLLECT_HUNG_TRIAGE])
    node_ids = [f"node-{index}" for index in range(8)]
    concurrent = 0
    peak = 0
    guard = threading.Lock()

    def sender(endpoint, envelope):
        nonlocal concurrent, peak
        with guard:
            concurrent += 1
            peak = max(peak, concurrent)
        time.sleep(0.2)
        with guard:
            concurrent -= 1
        return node_action_result(
            envelope.command.command_id,
            envelope.command.operation,
            details={"ranks": [{"rank": 0, "pid": 100}]},
        )

    step = copy_model(
        workflow.official_steps[0],
        execution_owner="gpu-fault-node-agent",
        node_ids=node_ids,
        parameters={
            "triage_timeout_seconds": 10,
            "not_sampled_nodes": ["node-8", "node-9"],
        },
    )
    started = time.monotonic()
    outcome = NodeActionWorkflowAdapter(
        {node_id: f"http://{node_id}:9099" for node_id in node_ids},
        "s" * 32,
        sender=sender,
    ).execute(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=step,
            step_index=0,
            request=WorkflowExecutionRequest(expected_fencing_token=3),
            idempotency_key="workflow/hung/parallel",
        )
    )
    elapsed = time.monotonic() - started

    assert outcome.status is WorkflowStepStatus.SUCCEEDED
    assert outcome.details["triage_completed_nodes"] == sorted(node_ids)
    assert peak > 1
    assert elapsed < 0.2 * len(node_ids)
    # Nodes left out of the sample are neither completed nor
    # undetermined; the classifier must not invent ranks for them.
    assert outcome.details["not_sampled_nodes"] == ["node-8", "node-9"]
    assert outcome.details["undetermined_nodes"] == []


def test_serial_dispatch_is_kept_for_mutating_operations() -> None:
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.MARK_UNSCHEDULABLE])
    node_ids = ["node-a", "node-b", "node-c"]
    order = []

    def sender(endpoint, envelope):
        order.append(envelope.command.node_id)
        return node_action_result(
            envelope.command.command_id, envelope.command.operation, details={}
        )

    step = copy_model(
        workflow.official_steps[0],
        execution_owner="gpu-fault-node-agent",
        node_ids=node_ids,
    )
    NodeActionWorkflowAdapter(
        {node_id: f"http://{node_id}:9099" for node_id in node_ids},
        "s" * 32,
        sender=sender,
    ).execute(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=step,
            step_index=0,
            request=WorkflowExecutionRequest(expected_fencing_token=3),
            idempotency_key="workflow/serial/cordon",
        )
    )

    assert order == node_ids
