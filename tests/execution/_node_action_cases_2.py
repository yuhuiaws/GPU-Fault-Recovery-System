from __future__ import annotations

from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    execute_workflow,
    node_action_result,
    workflow_step_execution,
)

from ._support import (
    NODE_ACTION_KEY_VERSION_DERIVED,
    NOW,
    AgentHeartbeat,
    FakeCoreApi,
    FakeHyperPodLifecycle,
    FleetRegistry,
    HTTPError,
    HyperPodLifecycleStepAdapter,
    IncidentState,
    KubernetesWorkflowAdapter,
    NodeActionPending,
    NodeActionStatus,
    NodeActionWorkflowAdapter,
    RecordingNodeActionAdapter,
    SignedAgentHeartbeat,
    SimpleNamespace,
    StubFleetRegistry,
    UnusedApi,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepContext,
    WorkflowStepOutcome,
    WorkflowStepStatus,
    derive_node_action_secret,
    io,
    json,
    pytest,
    sign_agent_heartbeat,
    timedelta,
    workflow_state,
)


@pytest.mark.parametrize(
    "snapshot_status",
    [
        WorkflowStepStatus.WAITING,
        WorkflowStepStatus.SUCCEEDED,
        WorkflowStepStatus.FAILED,
    ],
)
def test_hyperpod_reboot_auto_confirms_new_ready_agent_incarnation(
    snapshot_status: WorkflowStepStatus,
) -> None:
    class ObservableLifecycle(FakeHyperPodLifecycle):
        def resolve_nodes(self, identifiers):
            return [
                SimpleNamespace(
                    node_logical_id="logical-a",
                    instance_id="instance-a",
                    status="Running",
                )
                for _ in identifiers
            ]

    class SnapshotAdapter(RecordingNodeActionAdapter):
        def __init__(self, status: WorkflowStepStatus) -> None:
            super().__init__()
            self.status = status

        def execute(self, context):
            self.contexts.append(context)
            if self.status is WorkflowStepStatus.WAITING:
                return WorkflowStepOutcome.waiting(operation_id=context.idempotency_key)
            if self.status is WorkflowStepStatus.FAILED:
                return WorkflowStepOutcome.failed("snapshot unavailable")
            return WorkflowStepOutcome.succeeded(
                operation_id=context.idempotency_key,
                details={"snapshot_triggered": True},
            )

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
    lifecycle = ObservableLifecycle()
    sent: list[str] = []
    node_actions = SnapshotAdapter(snapshot_status)
    adapter = HyperPodLifecycleStepAdapter(
        lifecycle,
        registry=fleet,
        node_action_adapter=node_actions,
        store=store,
        alert_sender=sent.append,
        post_reboot_stabilization_seconds=0,
    )
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
    restarted = copy_model(
        heartbeat,
        boot_id="boot-b",
        agent_incarnation_id="incarnation-b",
        observed_at=NOW + timedelta(seconds=1),
    )
    fleet.register(
        SignedAgentHeartbeat(
            heartbeat=restarted, signature=sign_agent_heartbeat(restarted, secret)
        )
    )
    confirmation = active.execute(workflow.request_id, request)
    if snapshot_status is WorkflowStepStatus.WAITING:
        assert confirmation.status is WorkflowStatus.RUNNING
        node_actions.status = WorkflowStepStatus.SUCCEEDED
        completed = active.execute(workflow.request_id, request)
    else:
        completed = confirmation

    assert submitted.status is WorkflowStatus.RUNNING
    assert completed.status is WorkflowStatus.SUCCEEDED
    execution = store.get_workflow(workflow.request_id).step_executions[0]
    assert execution.details["externally_confirmed"] is True
    assert execution.details["confirmation_source"] == (
        "hyperpod-running-and-new-agent-incarnation"
    )
    assert all(
        context.step.operation is WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT
        for context in node_actions.contexts
    )
    assert len(node_actions.contexts) == (
        2 if snapshot_status is WorkflowStepStatus.WAITING else 1
    )
    if snapshot_status in {WorkflowStepStatus.WAITING, WorkflowStepStatus.SUCCEEDED}:
        assert execution.details["post_reboot_health_snapshot"] == {
            "snapshot_triggered": True
        }
        assert "post_reboot_health_snapshot_error" not in execution.details
    else:
        assert (
            execution.details["post_reboot_health_snapshot_error"]
            == "snapshot unavailable"
        )
        assert "post_reboot_health_snapshot" not in execution.details
    assert [item.step.operation for item in node_actions.contexts] == [
        WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT
    ] * len(node_actions.contexts)
    notifications = store.list_notifications()
    assert len(notifications) == 1
    assert sent == [notifications[0].notification_id]
    assert execution.details["notification_id"] == (notifications[0].notification_id)
    assert "系统动作：RESTART_NODE" in notifications[0].body_text


def test_hyperpod_replace_auto_confirms_and_rebinds_ready_new_instance() -> None:
    class ReplaceLifecycle(FakeHyperPodLifecycle):
        def __init__(self) -> None:
            super().__init__()
            self.instance_id = "i-old"
            self.node_name = "node-a"

        def resolve_nodes(self, identifiers):
            return [
                SimpleNamespace(
                    node_logical_id="logical-a",
                    instance_id=self.instance_id,
                    status="Running",
                    kubernetes_labels={"kubernetes.io/hostname": self.node_name},
                    aliases={"logical-a", self.instance_id, self.node_name},
                )
                for _ in identifiers
            ]

    class IsolationAdapter:
        def __init__(self) -> None:
            self.nodes = []

        def _isolate(self, context):
            self.nodes.extend(context.step.node_ids)
            return WorkflowStepOutcome.succeeded()

    store = build_store()
    secret = "agent-secret-" + "x" * 32
    fleet = FleetRegistry(store, secret, now=lambda: NOW)
    old_heartbeat = AgentHeartbeat(
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
        boot_id="boot-old",
        node_instance_id="i-old",
        agent_incarnation_id="incarnation-old",
        observed_at=NOW,
    )
    fleet.register(
        SignedAgentHeartbeat(
            heartbeat=old_heartbeat,
            signature=sign_agent_heartbeat(old_heartbeat, secret),
        )
    )
    incident, workflow = workflow_state(store, [WorkflowOperation.REPLACE_NODE])
    lifecycle = ReplaceLifecycle()
    isolation = IsolationAdapter()
    adapter = HyperPodLifecycleStepAdapter(
        lifecycle,
        registry=fleet,
        kubernetes_adapter=isolation,
        post_reboot_stabilization_seconds=0,
    )
    step = copy_model(workflow.official_steps[0], execution_owner=adapter.owner)
    workflow = copy_model(workflow, official_steps=[step])
    store.save_workflow(workflow)
    active = active_workflow_executor(
        store, [adapter], {WorkflowOperation.REPLACE_NODE}
    )
    request = WorkflowExecutionRequest(
        expected_fencing_token=3,
        isolation_verified_nodes=["node-a"],
        confirm_cluster_name="hp-cluster",
    )

    submitted = active.execute(workflow.request_id, request)
    lifecycle.instance_id = "i-new"
    lifecycle.node_name = "node-new"
    no_agent = active.execute(workflow.request_id, request)
    new_heartbeat = copy_model(
        old_heartbeat,
        node_id="node-new",
        endpoint="http://node-new:9099",
        boot_id="boot-new",
        node_instance_id="i-new",
        agent_incarnation_id="incarnation-new",
        observed_at=NOW + timedelta(seconds=1),
    )
    fleet.register(
        SignedAgentHeartbeat(
            heartbeat=new_heartbeat,
            signature=sign_agent_heartbeat(new_heartbeat, secret),
        )
    )
    completed = active.execute(workflow.request_id, request)

    assert submitted.status is WorkflowStatus.RUNNING
    assert no_agent.status is WorkflowStatus.RUNNING
    assert completed.status is WorkflowStatus.SUCCEEDED
    execution = store.get_workflow(workflow.request_id).step_executions[0]
    assert execution.details["confirmation_source"] == (
        "hyperpod-logical-node-new-instance-and-ready-agent"
    )
    assert execution.details["node_rebindings"] == {
        "node-a": "node-new",
        "i-old": "node-new",
    }
    assert isolation.nodes == ["node-new"]
    assert incident.node_ids == ["node-a"]
    assert store.get_incident(incident.incident_id).node_ids == ["node-new"]


def test_hyperpod_reboot_waits_for_post_reboot_stabilization() -> None:
    class ObservableLifecycle(FakeHyperPodLifecycle):
        def resolve_nodes(self, identifiers):
            return [
                SimpleNamespace(
                    node_logical_id="logical-a",
                    instance_id="instance-a",
                    status="Running",
                )
                for _ in identifiers
            ]

    clock = [NOW]
    store = build_store()
    secret = "agent-secret-" + "x" * 32
    fleet = FleetRegistry(store, secret, now=lambda: clock[0])
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
    adapter = HyperPodLifecycleStepAdapter(
        ObservableLifecycle(), registry=fleet, post_reboot_stabilization_seconds=60
    )
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
    restarted = copy_model(
        heartbeat,
        boot_id="boot-b",
        agent_incarnation_id="incarnation-b",
        observed_at=NOW + timedelta(seconds=1),
    )
    fleet.register(
        SignedAgentHeartbeat(
            heartbeat=restarted, signature=sign_agent_heartbeat(restarted, secret)
        )
    )
    stabilizing = active.execute(workflow.request_id, request)
    clock[0] = NOW + timedelta(seconds=59)
    still_stabilizing = active.execute(workflow.request_id, request)
    clock[0] = NOW + timedelta(seconds=60)
    completed = active.execute(workflow.request_id, request)

    assert submitted.status is WorkflowStatus.RUNNING
    assert stabilizing.status is WorkflowStatus.RUNNING
    assert still_stabilizing.status is WorkflowStatus.RUNNING
    assert completed.status is WorkflowStatus.SUCCEEDED
    execution = store.get_workflow(workflow.request_id).step_executions[0]
    assert (
        execution.details["submission_idempotency_key"]
        == "workflow-active/RESTART_NODE/0"
    )
    assert execution.details["post_reboot_stabilization_seconds"] == 60


def test_hyperpod_preflight_failure_does_not_revoke_agent() -> None:
    class RejectingLifecycle(FakeHyperPodLifecycle):
        def preflight(self, *_args, **_kwargs):
            from types import SimpleNamespace

            return SimpleNamespace(
                safe_to_submit=False, gate_failures=["automatic recovery owns mutation"]
            )

    store = build_store()
    _, workflow = workflow_state(store, [WorkflowOperation.RESTART_NODE])
    lifecycle = RejectingLifecycle()
    adapter = HyperPodLifecycleStepAdapter(lifecycle, registry=object())
    step = copy_model(workflow.official_steps[0], execution_owner=adapter.owner)
    store.save_workflow(copy_model(workflow, official_steps=[step]))
    active = active_workflow_executor(
        store, [adapter], {WorkflowOperation.RESTART_NODE}
    )

    result = execute_workflow(
        active,
        workflow.request_id,
        isolation_verified_nodes=["node-a"],
        confirm_cluster_name="hp-cluster",
    )

    assert result.status is WorkflowStatus.FAILED
    assert "automatic recovery owns mutation" in result.error
    assert lifecycle.calls == 0


def test_hyperpod_preflight_credential_error_is_a_configuration_failure() -> None:
    # A pod whose ServiceAccount has no eks.amazonaws.com/role-arn
    # annotation fails inside preflight's DescribeCluster with
    # NoCredentialsError, which is not in the (HyperPodAdapterError,
    # KeyError, ValueError) tuple. It used to escape the adapter
    # entirely: reported as executor_internal_error, and -- worse -- it
    # skipped every guard after preflight, including the refusal to fall
    # back to the provider replacement API.
    class NoCredentialsLifecycle(FakeHyperPodLifecycle):
        def preflight(self, *_args, **_kwargs):
            raise type(
                "NoCredentialsError",
                (Exception,),
                {"__module__": "botocore.exceptions"},
            )("Unable to locate credentials")

    store = build_store()
    _, workflow = workflow_state(store, [WorkflowOperation.RESTART_NODE])
    lifecycle = NoCredentialsLifecycle()
    adapter = HyperPodLifecycleStepAdapter(
        lifecycle,
        # A real registry: the configuration failure is returned after
        # the agent-baseline lookup, so a bare object() would fail this
        # test for the wrong reason.
        registry=FleetRegistry(store, "agent-secret-" + "x" * 32, now=lambda: NOW),
    )
    step = copy_model(workflow.official_steps[0], execution_owner=adapter.owner)
    store.save_workflow(copy_model(workflow, official_steps=[step]))
    active = active_workflow_executor(
        store, [adapter], {WorkflowOperation.RESTART_NODE}
    )

    result = execute_workflow(
        active,
        workflow.request_id,
        isolation_verified_nodes=["node-a"],
        confirm_cluster_name="hp-cluster",
    )

    assert result.status is WorkflowStatus.FAILED
    assert "HyperPod preflight cannot run" in result.error
    assert "eks.amazonaws.com/role-arn" in result.error
    # No submission, and no agent revoked: a node must not be fenced
    # because the executor cannot talk to AWS.
    assert lifecycle.calls == 0
    execution = store.get_workflow(workflow.request_id).step_executions[0]
    assert execution.details["configuration_error"] is True
    assert execution.details["exception_type"] == "NoCredentialsError"
    assert "rollout restart" in execution.details["remediation"]


def test_hyperpod_preflight_defect_still_propagates() -> None:
    # The classifier must not swallow real bugs: anything that is not AWS
    # misconfiguration keeps reaching the executor's internal-error path.
    class BrokenLifecycle(FakeHyperPodLifecycle):
        def preflight(self, *_args, **_kwargs):
            raise AttributeError("dispatcher has no core")

    store = build_store()
    _, workflow = workflow_state(store, [WorkflowOperation.RESTART_NODE])
    adapter = HyperPodLifecycleStepAdapter(BrokenLifecycle(), registry=object())
    step = copy_model(workflow.official_steps[0], execution_owner=adapter.owner)
    store.save_workflow(copy_model(workflow, official_steps=[step]))

    with pytest.raises(AttributeError):
        adapter.execute(
            WorkflowStepContext(
                workflow=store.get_workflow(workflow.request_id),
                incident=store.get_incident(workflow.incident_id),
                step=step,
                step_index=0,
                request=WorkflowExecutionRequest(
                    expected_fencing_token=3,
                    isolation_verified_nodes=["node-a"],
                    confirm_cluster_name="hp-cluster",
                ),
                idempotency_key="idem-a",
            )
        )


def test_node_action_adapter_derives_key_for_version_two_agent() -> None:
    master = "node-action-master-" + "m" * 32
    registry = StubFleetRegistry(
        {"node-a": "http://node-a:9099"},
        node_action_key_version=NODE_ACTION_KEY_VERSION_DERIVED,
    )
    adapter = NodeActionWorkflowAdapter({}, master, registry=registry)

    assert adapter._secret_for_node("cluster-a", "node-a") == derive_node_action_secret(
        master, "cluster-a", "node-a"
    )
    assert adapter._secret_for_node("cluster-a", "node-a") != master


def test_node_action_adapter_refuses_shared_key_when_derived_required() -> None:
    registry = StubFleetRegistry(
        {"node-a": "http://node-a:9099"},
        node_action_key_version=1,
        required_node_action_key_version=(NODE_ACTION_KEY_VERSION_DERIVED),
    )
    adapter = NodeActionWorkflowAdapter(
        {}, "node-action-master-" + "m" * 32, registry=registry
    )

    with pytest.raises(ValueError, match="key version.*not accepted"):
        adapter._secret_for_node("cluster-a", "node-a")


@pytest.mark.parametrize(
    "error",
    [TimeoutError("submit timed out"), NodeActionPending("workflow/step/node-a")],
)
def test_node_action_pending_or_transport_timeout_returns_waiting(
    error: Exception,
) -> None:
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.RESET_GPU])
    registry = StubFleetRegistry({"node-a": "http://node-a:9099"})

    def sender(_endpoint, _envelope):
        raise error

    adapter = NodeActionWorkflowAdapter({}, "s" * 32, sender=sender, registry=registry)
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        node_ids=["node-a"],
        gpu_uuids=["GPU-a"],
    )
    outcome = adapter.execute(
        WorkflowStepContext(
            workflow=copy_model(workflow, official_steps=[step]),
            incident=incident,
            step=step,
            step_index=0,
            request=WorkflowExecutionRequest(
                expected_fencing_token=workflow.fencing_token
            ),
            idempotency_key="workflow/step",
        )
    )

    assert outcome.status is WorkflowStepStatus.WAITING
    assert outcome.details["node_action_state"] in {"PENDING", "TRANSPORT_RETRY"}


def test_node_action_http_rejection_preserves_structured_code() -> None:
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.RESET_GPU])
    registry = StubFleetRegistry({"node-a": "http://node-a:9099"})
    detail = {
        "detail": {
            "code": "COMMAND_EXPIRED",
            "message": "node action command has expired",
            "retryable": True,
            "requires_new_command": True,
        }
    }

    def sender(_endpoint, _envelope):
        raise HTTPError(
            "http://node-a:9099/v1/node-actions/submit",
            410,
            "Gone",
            {},
            io.BytesIO(json.dumps(detail).encode()),
        )

    adapter = NodeActionWorkflowAdapter({}, "s" * 32, sender=sender, registry=registry)
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        node_ids=["node-a"],
        gpu_uuids=["GPU-a"],
    )
    outcome = adapter.execute(
        WorkflowStepContext(
            workflow=copy_model(workflow, official_steps=[step]),
            incident=incident,
            step=step,
            step_index=0,
            request=WorkflowExecutionRequest(
                expected_fencing_token=workflow.fencing_token
            ),
            idempotency_key="workflow/expired",
        )
    )

    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.details == {
        "node_action_error_code": "COMMAND_EXPIRED",
        "node_action_retryable": True,
        "node_action_requires_new_command": True,
        "http_status": 410,
    }


def test_retryable_node_action_failure_returns_waiting() -> None:
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.RESET_GPU])
    registry = StubFleetRegistry({"node-a": "http://node-a:9099"})

    def sender(_endpoint, envelope):
        return node_action_result(
            envelope.command.command_id,
            envelope.command.operation,
            NodeActionStatus.FAILED,
            error="OSError: temporary transport error",
            retryable=True,
        )

    adapter = NodeActionWorkflowAdapter({}, "s" * 32, sender=sender, registry=registry)
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        node_ids=["node-a"],
        gpu_uuids=["GPU-a"],
    )
    outcome = adapter.execute(
        WorkflowStepContext(
            workflow=copy_model(workflow, official_steps=[step]),
            incident=incident,
            step=step,
            step_index=0,
            request=WorkflowExecutionRequest(
                expected_fencing_token=workflow.fencing_token
            ),
            idempotency_key="workflow/retryable",
        )
    )

    assert outcome.status is WorkflowStepStatus.WAITING
    assert outcome.details["retryable_node_action"] is True


def test_node_action_adapter_addresses_via_registry_not_static_map() -> None:
    """A stale endpoint map must not decide where actions are sent.

    Regression: the regional executor built the adapter without a
    registry, so the hand-maintained GPU_FAULT_NODE_AGENT_ENDPOINTS map
    was the only address source. Once nodes were replaced it pointed at
    destroyed instances and every node action failed closed with no
    prior signal.
    """
    store = build_store()
    _, workflow = workflow_state(store, [WorkflowOperation.RESET_GPU])
    captured = []

    def sender(endpoint, envelope):
        captured.append(endpoint)
        return node_action_result(
            envelope.command.command_id,
            envelope.command.operation,
            details={"reset_gpu_uuids": ["GPU-a"]},
        )

    registry = StubFleetRegistry({"node-a": "http://live-a:9099"})
    adapter = NodeActionWorkflowAdapter(
        # Deliberately stale: a decommissioned node at a recycled IP.
        {"node-a": "http://decommissioned:9099"},
        "s" * 32,
        sender=sender,
        registry=registry,
    )
    step = copy_model(
        workflow.official_steps[0], execution_owner=adapter.owner, gpu_uuids=["GPU-a"]
    )
    workflow = copy_model(workflow, official_steps=[step])
    store.save_workflow(workflow)
    active = active_workflow_executor(store, [adapter], {WorkflowOperation.RESET_GPU})

    result = execute_workflow(active, workflow.request_id)

    assert result.status is WorkflowStatus.SUCCEEDED
    assert captured == ["http://live-a:9099"]
    assert registry.endpoint_calls == [("cluster-a", "node-a")]


def test_node_action_adapter_without_registry_uses_static_map() -> None:
    """The static map stays the fallback for non-regional deployments."""
    captured = []

    def sender(endpoint, envelope):
        captured.append(endpoint)
        return node_action_result(
            envelope.command.command_id, envelope.command.operation
        )

    adapter = NodeActionWorkflowAdapter(
        {"node-a": "http://node-a:9099"}, "s" * 32, sender=sender
    )

    assert adapter._endpoint("cluster-a", "node-a", None) == "http://node-a:9099"
    assert adapter._endpoint("cluster-a", "node-b", None) is None


def test_node_action_adapter_uses_explicit_endpoint_and_signature() -> None:
    store = build_store()
    _, workflow = workflow_state(store, [WorkflowOperation.RESET_GPU])
    captured = []

    def sender(endpoint, envelope):
        captured.append((endpoint, envelope))
        return node_action_result(
            envelope.command.command_id,
            envelope.command.operation,
            details={"reset_gpu_uuids": ["GPU-a"]},
        )

    adapter = NodeActionWorkflowAdapter(
        {"node-a": "http://node-a:9099"}, "s" * 32, sender=sender
    )
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        gpu_uuids=["GPU-a"],
        parameters={"target_driver_branch": 575},
    )
    workflow = copy_model(workflow, official_steps=[step])
    store.save_workflow(workflow)
    active = active_workflow_executor(store, [adapter], {WorkflowOperation.RESET_GPU})

    result = execute_workflow(active, workflow.request_id)

    assert result.status is WorkflowStatus.SUCCEEDED
    assert captured[0][0] == "http://node-a:9099"
    assert captured[0][1].signature
    assert captured[0][1].command.fencing_token == 3
    assert captured[0][1].command.parameters == {"target_driver_branch": 575}


@pytest.mark.parametrize(
    ("diagnostic_outcome", "expected_status"),
    [
        ("PASS", WorkflowStepStatus.SUCCEEDED),
        ("WARN", WorkflowStepStatus.SUCCEEDED),
        ("FAIL", WorkflowStepStatus.FAILED),
        ("INCONCLUSIVE", WorkflowStepStatus.FAILED),
    ],
)
def test_node_action_adapter_interprets_dcgm_diagnostic_outcome(
    diagnostic_outcome: str, expected_status: WorkflowStepStatus
) -> None:
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.RUN_DCGM_DIAGNOSTIC])
    step = copy_model(
        workflow.official_steps[0],
        execution_owner="gpu-fault-node-agent",
        node_ids=["node-a"],
        gpu_uuids=["GPU-a"],
    )

    def sender(_, envelope):
        return node_action_result(
            envelope.command.command_id,
            envelope.command.operation,
            details={
                "diagnostic_outcome": diagnostic_outcome,
                "evidence_ref": "file:///diagnostics/dcgm.json",
                "sha256": "a" * 64,
                "recommended_actions": [
                    {
                        "action_code": "PCIE_AER_INSPECTION",
                        "priority": "IMMEDIATE",
                        "instruction": "Inspect PCIe and AER evidence.",
                        "trigger_tests": ["PCIe"],
                    }
                ],
            },
        )

    outcome = NodeActionWorkflowAdapter(
        {"node-a": "http://node-a:9099"}, "s" * 32, sender=sender, store=store
    ).execute(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=step,
            step_index=0,
            request=WorkflowExecutionRequest(expected_fencing_token=3),
            idempotency_key="workflow/dcgm-quick-diagnostic",
        )
    )

    assert outcome.status is expected_status
    notification = store.list_notifications()[0]
    assert "DCGM 快速诊断结果通知" in notification.body_text
    assert "PCIE_AER_INSPECTION" in notification.body_text
    assert "file:///diagnostics/dcgm.json" in (notification.evidence_refs)
    assert outcome.details["notification_id"] == (notification.notification_id)
    if expected_status is WorkflowStepStatus.FAILED:
        assert outcome.details["failed_nodes"] == ["node-a"]
        assert "PCIE_AER_INSPECTION" in (outcome.details["node_failures"]["node-a"])
        assert outcome.details["control_plane_action"] == ("DRAIN_AND_QUARANTINE")
    else:
        assert outcome.details["control_plane_action"] == ("COOLDOWN_AND_VALIDATE")


def test_completed_spare_failover_keeps_old_node_safety_hold() -> None:
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.REPLACE_NODE])
    incident = copy_model(incident, state=IncidentState.RECOVERED)
    workflow = copy_model(
        workflow,
        status=WorkflowStatus.SUCCEEDED,
        completed_step_indexes=[0],
        completed_operations=[WorkflowOperation.REPLACE_NODE],
        step_executions=[
            workflow_step_execution(
                0,
                WorkflowOperation.REPLACE_NODE,
                details={
                    "action": "SPARE_FAILOVER",
                    "node_rebindings": {"node-a": "node-spare"},
                },
            )
        ],
    )
    store.save_incident(incident)
    store.save_workflow(workflow)
    adapter = KubernetesWorkflowAdapter(
        core_api=FakeCoreApi(),
        batch_api=UnusedApi(),
        custom_api=UnusedApi(),
        store=store,
    )

    assert adapter._incident_ownership(incident.incident_id) == (True, True)


def test_node_action_adapter_uses_prederived_node_secret() -> None:
    adapter = NodeActionWorkflowAdapter(
        {"node-a": "http://node-a:9099"},
        "",
        node_secrets={"node-a": "n" * 64},
        node_action_key_version=NODE_ACTION_KEY_VERSION_DERIVED,
    )

    assert adapter._secret_for_node("cluster-a", "node-a") == ("n" * 64)
    with pytest.raises(
        ValueError, match="node action secret is unavailable for node-b"
    ):
        adapter._secret_for_node("cluster-a", "node-b")


def test_node_action_adapter_loads_node_secret_directory(monkeypatch, tmp_path) -> None:
    (tmp_path / "node-a").write_text("a" * 64, encoding="utf-8")
    monkeypatch.setenv(
        "GPU_FAULT_NODE_AGENT_ENDPOINTS", '{"node-a":"http://node-a:9099"}'
    )
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_KEYS_DIR", str(tmp_path))
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_KEY_VERSION", "2")
    monkeypatch.delenv("GPU_FAULT_NODE_ACTION_SECRET", raising=False)

    adapter = NodeActionWorkflowAdapter.from_environment()

    assert adapter._secret_for_node("cluster-a", "node-a") == ("a" * 64)
