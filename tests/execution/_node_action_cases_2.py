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
    isolated_kubernetes_adapter,
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
        kubernetes_adapter=isolated_kubernetes_adapter(),
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
            self.core = isolated_kubernetes_adapter().core

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
        ObservableLifecycle(),
        registry=fleet,
        post_reboot_stabilization_seconds=60,
        kubernetes_adapter=isolated_kubernetes_adapter(),
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
    adapter = HyperPodLifecycleStepAdapter(
        lifecycle, registry=object(), kubernetes_adapter=isolated_kubernetes_adapter()
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
        kubernetes_adapter=isolated_kubernetes_adapter(),
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
    adapter = HyperPodLifecycleStepAdapter(
        BrokenLifecycle(),
        registry=object(),
        kubernetes_adapter=isolated_kubernetes_adapter(),
    )
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

    # An expired command is not a failed GPU: the next dispatch signs a fresh
    # envelope, so the step holds and says a new command is required.
    assert outcome.status is WorkflowStepStatus.WAITING
    assert outcome.details["node_action_state"] == "NEW_COMMAND_REQUIRED"
    assert outcome.details["node_action_error_code"] == "COMMAND_EXPIRED"
    assert outcome.details["node_action_retryable"] is True
    assert outcome.details["node_action_requires_new_command"] is True
    assert outcome.details["http_status"] == 410
    assert outcome.details["node_action_command_id"] == (
        "workflow/expired/node-a/agent-7"
    )


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


@pytest.mark.parametrize("spare_health_check", [False, True])
def test_busy_spare_client_check_fails_instead_of_waiting(spare_health_check) -> None:
    """Quiesce-waiting belongs to the faulted node, not to a spare candidate.

    Regression: the warm-spare eligibility check dispatched
    VERIFY_NO_GPU_CLIENTS to the spare's agent, and the agent's definitive
    "somebody else's GPU work is running here" answer was mapped to
    WAITING. The eligibility check then re-cast that as SpareHealthPending,
    so REPLACE_NODE waited out its whole execution deadline, no
    hyperpod-spare-insufficient alert was raised, and the retry counter
    never advanced because the synthetic check borrows the parent step's
    index -- making verify_max_attempts unreachable and replaying one
    finished command record forever.
    """

    store = build_store()
    incident, workflow = workflow_state(
        store, [WorkflowOperation.VERIFY_NO_GPU_CLIENTS]
    )
    registry = StubFleetRegistry({"node-a": "http://node-a:9099"})

    def sender(_endpoint, envelope):
        return node_action_result(
            envelope.command.command_id,
            envelope.command.operation,
            NodeActionStatus.FAILED,
            error=("RuntimeError: GPU compute clients are still active: GPU-abc:4242"),
        )

    adapter = NodeActionWorkflowAdapter({}, "s" * 32, sender=sender, registry=registry)
    parameters = {"compute_clients_only": True}
    if spare_health_check:
        parameters["spare_health_check"] = True
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        node_ids=["node-a"],
        gpu_uuids=["GPU-a"],
        parameters=parameters,
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
            idempotency_key="workflow/busy-clients",
        )
    )

    assert outcome.details["gpu_client_quiesce_attempt"] == 1
    if spare_health_check:
        assert outcome.status is WorkflowStepStatus.FAILED
        assert "clients are still active" in (outcome.error or "")
    else:
        assert outcome.status is WorkflowStepStatus.WAITING
        assert "clients are still active" in outcome.details["reason"]


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


def _two_node_outcome(
    operation: WorkflowOperation, answers: dict[str, dict]
) -> WorkflowStepOutcome:
    """One non-parallel step over node-a then node-b with canned answers.

    ``answers`` maps a node id to the keyword arguments of the
    ``NodeActionResult`` its agent returns, so a case only has to say how the
    second node answered.
    """

    store = build_store()
    incident, workflow = workflow_state(store, [operation])
    registry = StubFleetRegistry(
        {"node-a": "http://node-a:9099", "node-b": "http://node-b:9099"}
    )

    def sender(_endpoint, envelope):
        answer = answers[envelope.command.node_id]
        return node_action_result(
            envelope.command.command_id, envelope.command.operation, **answer
        )

    adapter = NodeActionWorkflowAdapter({}, "s" * 32, sender=sender, registry=registry)
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        node_ids=["node-a", "node-b"],
        gpu_uuids=["GPU-a"],
    )
    return adapter.execute(
        WorkflowStepContext(
            workflow=copy_model(workflow, official_steps=[step]),
            incident=incident,
            step=step,
            step_index=0,
            request=WorkflowExecutionRequest(
                expected_fencing_token=workflow.fencing_token
            ),
            idempotency_key="workflow/two-nodes",
        )
    )


def test_a_failed_second_node_still_reports_the_first_nodes_result() -> None:
    """A terminal FAILED must still name the nodes the step already changed.

    The step folds one node at a time, so node-a's action really ran on the
    cluster before node-b refused. Reporting only ``node agent node-b: ...``
    hides that from the operator reading ``result_details.node_results`` and
    from any later step that has to undo what node-a already did.
    """

    outcome = _two_node_outcome(
        WorkflowOperation.VALIDATE_GPU,
        {
            "node-a": {
                "status": NodeActionStatus.SUCCEEDED,
                "details": {"validated": True},
            },
            "node-b": {
                "status": NodeActionStatus.FAILED,
                "error": "GPU 0 is still faulted",
            },
        },
    )

    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    assert outcome.error == "node agent node-b: GPU 0 is still faulted", outcome.error
    assert outcome.details["node_results"]["node-a"] == {"validated": True}, (
        "the failed step dropped the node that already succeeded"
    )
    assert outcome.details["completed_nodes"] == ["node-a"], outcome.details


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (
            {
                "status": NodeActionStatus.FAILED,
                "error": "OSError: temporary transport error",
                "retryable": True,
            },
            WorkflowStepStatus.WAITING,
        ),
        (
            {
                "status": NodeActionStatus.INTERRUPTED,
                "error": "the agent restarted mid-action",
            },
            WorkflowStepStatus.FAILED,
        ),
    ],
    ids=["retryable", "interrupted"],
)
def test_an_early_held_or_interrupted_node_keeps_the_earlier_node_results(
    answer: dict, expected: WorkflowStepStatus
) -> None:
    """A hold and a manual-confirmation exit report the same partial state.

    The retryable hold is re-claimed and replays node-a from the agent ledger,
    but the INTERRUPTED exit is terminal, and an operator asked to confirm by
    hand needs to know node-a was already acted on.
    """

    outcome = _two_node_outcome(
        WorkflowOperation.VALIDATE_GPU,
        {
            "node-a": {
                "status": NodeActionStatus.SUCCEEDED,
                "details": {"validated": True},
            },
            "node-b": answer,
        },
    )

    assert outcome.status is expected, outcome
    assert outcome.details["node_results"]["node-a"] == {"validated": True}, (
        "the early exit dropped the node that already succeeded"
    )
    assert outcome.details["completed_nodes"] == ["node-a"], outcome.details


def test_a_quiesce_wait_on_the_second_node_keeps_the_first_nodes_result() -> None:
    """VERIFY_NO_GPU_CLIENTS waits per attempt and must not forget node-a.

    The attempt counter rides on the step details across the wait, and the
    per-node results have to ride along with it: the operator reads which
    nodes are already clear while the step keeps waiting on the rest.
    """

    outcome = _two_node_outcome(
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        {
            "node-a": {
                "status": NodeActionStatus.SUCCEEDED,
                "details": {"gpu_clients": []},
            },
            "node-b": {
                "status": NodeActionStatus.FAILED,
                "error": "GPU clients are still active: 2",
            },
        },
    )

    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    assert outcome.details["waiting_node"] == "node-b", outcome.details
    assert outcome.details["node_results"]["node-a"] == {"gpu_clients": []}, (
        "the quiesce wait dropped the node that already verified clean"
    )
    assert outcome.details["completed_nodes"] == ["node-a"], outcome.details
