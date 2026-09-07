from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from gpu_fault.adapters import ManagedRecoveryObserverAdapter
from gpu_fault.execution import WorkflowStepContext, WorkflowStepOutcome
from gpu_fault.fleet import (
    AgentLifecycleState,
    AgentRecord,
    FleetCompatibilityPolicy,
    FleetRegistry,
)
from gpu_fault.hyperpod import HyperPodNode
from gpu_fault.managed_recovery import (
    HyperPodIdentityRegistry,
    HyperPodManagedRecoveryObserver,
    HyperPodNodeIdentity,
)
from gpu_fault.models import (
    IncidentState,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.store import SqliteStore
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    execute_workflow,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

NOW = datetime.now(timezone.utc)
OWNER = "hyperpod-managed-node-recovery"


class MutableLifecycle:
    def __init__(self, node: HyperPodNode) -> None:
        self.config = SimpleNamespace(cluster_name="hp-cluster")
        self.node = node
        self.mutations = []

    def list_nodes(self, *, enrich: bool = False):
        assert enrich
        return [self.node]

    def batch_replace_cluster_nodes(self, **kwargs):
        self.mutations.append(kwargs)

    def batch_reboot_cluster_nodes(self, **kwargs):
        self.mutations.append(kwargs)


class IsolationAdapter:
    def __init__(self) -> None:
        self.nodes = []

    def _isolate(self, context):
        self.nodes.extend(context.step.node_ids)
        return WorkflowStepOutcome.succeeded()


def node(instance: str, kubernetes_name: str) -> HyperPodNode:
    return HyperPodNode(
        node_logical_id="worker-group-1",
        instance_id=instance,
        instance_group_name="worker-group",
        instance_type="p5.48xlarge",
        status="Running",
        kubernetes_labels={"kubernetes.io/hostname": kubernetes_name},
    )


def agent(
    node_id: str, instance_id: str, incarnation: str, *, generation: int = 1
) -> AgentRecord:
    return AgentRecord(
        cluster_id="hp-cluster",
        node_id=node_id,
        endpoint=f"http://{node_id}:9099",
        agent_protocol_version=3,
        agent_version="1.0.0",
        artifact_sha256="a" * 64,
        policy_version="catalog",
        runtime_profile_version="profile",
        config_digest="b" * 64,
        allowed_operations=[
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            WorkflowOperation.RESET_GPU,
        ],
        boot_id=incarnation,
        node_instance_id=instance_id,
        agent_incarnation_id=incarnation,
        first_seen_at=NOW,
        last_seen_at=NOW,
        lease_expires_at=NOW + timedelta(minutes=5),
        generation=generation,
    )


def state(operation: WorkflowOperation):
    incident = fault_incident(
        "incident-managed",
        "event-managed",
        cluster_id="hp-cluster",
        node_ids=["k8s-old"],
        state=IncidentState.ACTION_PENDING,
        fencing_token=3,
        created_at=NOW,
        updated_at=NOW,
    )
    step = workflow_step(operation, OWNER, node_ids=["k8s-old"])
    workflow = workflow_request(
        "workflow-managed",
        incident.incident_id,
        WorkflowStatus.RUNNING,
        official_steps=[step],
        created_at=NOW,
        updated_at=NOW,
    )
    request = WorkflowExecutionRequest(expected_fencing_token=3)
    return incident, workflow, step, request


def context_with_previous(
    incident, workflow, step, request, outcome: WorkflowStepOutcome
) -> WorkflowStepContext:
    execution = workflow_step_execution(
        0,
        step.operation,
        outcome.status,
        adapter_operation_id=outcome.adapter_operation_id,
        details=outcome.details or {},
    )
    workflow = copy_model(workflow, step_executions=[execution])
    return WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=0,
        request=request,
        idempotency_key="workflow-managed/0/managed",
    )


def observer_fixture(operation=WorkflowOperation.REPLACE_NODE):
    store = build_store()
    lifecycle = MutableLifecycle(node("i-old", "k8s-old"))
    identities = HyperPodIdentityRegistry(lifecycle, store)
    registry = FleetRegistry(
        store, "s" * 32, FleetCompatibilityPolicy(), now=lambda: NOW
    )
    isolation = IsolationAdapter()
    observer = HyperPodManagedRecoveryObserver(
        identities, store, registry=registry, kubernetes_adapter=isolation
    )
    incident, workflow, step, request = state(operation)
    return (
        store,
        lifecycle,
        identities,
        observer,
        isolation,
        incident,
        workflow,
        step,
        request,
    )


def test_identity_registry_retains_replaced_instance_alias() -> None:
    store = build_store()
    lifecycle = MutableLifecycle(node("i-old", "k8s-old"))
    identities = HyperPodIdentityRegistry(lifecycle, store)

    first = identities.refresh()[0]
    lifecycle.node = node("i-new", "k8s-new")
    second = identities.refresh()[0]

    assert first.generation == 1
    assert second.generation == 2
    assert "i-old" in second.retired_aliases
    assert "worker-group-1" not in second.retired_aliases
    assert identities.resolve("i-old") == second

    lifecycle.node = node("i-old", "k8s-old")
    stale = identities.refresh()[0]
    assert stale.instance_id == "i-new"
    assert stale.generation == 2


def test_managed_replace_retires_old_agent_and_rebinds() -> None:
    (store, lifecycle, _, observer, isolation, incident, workflow, step, request) = (
        observer_fixture()
    )
    store.save_agent(agent("k8s-old", "i-old", "boot-old"))
    first_context = WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=0,
        request=request,
        idempotency_key="workflow-managed/0/replace",
    )

    waiting = observer.observe(first_context)
    lifecycle.node = node("i-new", "k8s-new")
    second_context = context_with_previous(incident, workflow, step, request, waiting)
    replacing = observer.observe(second_context)

    assert replacing.status is WorkflowStepStatus.WAITING
    assert replacing.details["managed_recovery_state"] == "AWS_REPLACING"
    assert (
        store.get_agent("hp-cluster", "k8s-old").lifecycle_state
        is AgentLifecycleState.REVOKED
    )

    store.save_agent(agent("k8s-new", "i-new", "boot-new"))
    final_context = context_with_previous(incident, workflow, step, request, replacing)
    completed = observer.observe(final_context)

    assert completed.status is WorkflowStepStatus.SUCCEEDED
    assert completed.details["node_rebindings"]["k8s-old"] == "k8s-new"
    assert completed.details["node_rebindings"]["i-old"] == "k8s-new"
    assert isolation.nodes == ["k8s-new"]
    assert lifecycle.mutations == []


def test_managed_replace_detects_change_observed_before_workflow() -> None:
    (store, lifecycle, identities, observer, _, incident, workflow, step, request) = (
        observer_fixture()
    )
    store.save_agent(agent("k8s-old", "i-old", "boot-old"))
    identities.refresh()
    lifecycle.node = node("i-new", "k8s-new")
    identities.refresh()

    first = observer.observe(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=step,
            step_index=0,
            request=request,
            idempotency_key="workflow-managed/0/already-replaced",
        )
    )
    second = observer.observe(
        context_with_previous(incident, workflow, step, request, first)
    )

    assert first.details["managed_targets"][0]["replacement_already_observed"]
    assert second.status is WorkflowStepStatus.WAITING
    assert (
        store.get_agent("hp-cluster", "k8s-old").lifecycle_state
        is AgentLifecycleState.REVOKED
    )


def test_managed_reboot_uses_new_incarnation_without_revoking_it() -> None:
    (store, _, _, observer, _, incident, workflow, step, request) = observer_fixture(
        WorkflowOperation.RESTART_NODE
    )
    store.save_agent(agent("k8s-old", "i-old", "boot-old"))
    first = observer.observe(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=step,
            step_index=0,
            request=request,
            idempotency_key="workflow-managed/0/reboot",
        )
    )
    store.save_agent(agent("k8s-old", "i-old", "boot-new", generation=2))

    completed = observer.observe(
        context_with_previous(incident, workflow, step, request, first)
    )

    assert completed.status is WorkflowStepStatus.SUCCEEDED
    assert (
        store.get_agent("hp-cluster", "k8s-old").lifecycle_state
        is AgentLifecycleState.ACTIVE
    )


def test_managed_recovery_timeout_creates_one_notification() -> None:
    (store, _, identities, _, _, incident, workflow, step, request) = observer_fixture()
    observer = HyperPodManagedRecoveryObserver(
        identities, store, registry=None, timeout=timedelta(0)
    )
    first = observer.observe(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=step,
            step_index=0,
            request=request,
            idempotency_key="workflow-managed/0/timeout",
        )
    )

    failed = observer.observe(
        context_with_previous(incident, workflow, step, request, first)
    )

    assert failed.status is WorkflowStepStatus.FAILED
    assert "timed out" in failed.error
    assert len(store.list_notifications()) == 1


def test_a_delegated_wait_gives_up_before_the_workflow_does() -> None:
    """The escalation has to be sent by whoever notices, and this is the noticer.

    The provider window and the workflow budget are configured independently, so
    the workflow can be the shorter of the two -- and at the shipped defaults it
    is, because the two are the same length and the workflow has already spent
    time on the steps before this one. Left unclamped, the workflow deadline
    reaches the step first, on the generic failure path, and the operator never
    gets the support-case notification that is the only actionable output of a
    recovery the provider did not complete.
    """

    (store, _, identities, _, _, incident, workflow, step, request) = observer_fixture()
    observer = HyperPodManagedRecoveryObserver(
        identities, store, registry=None, timeout=timedelta(minutes=30)
    )
    first = observer.observe(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=step,
            step_index=0,
            request=request,
            idempotency_key="workflow-managed/0/clamped",
        )
    )
    waiting = context_with_previous(incident, workflow, step, request, first)
    # Inside the margin, so the observation is out of time while the workflow
    # itself still has some -- which is the whole point of the margin: the
    # deadline is checked before a step is dispatched, so an observation that
    # expired exactly with the workflow would never run.
    deadline = datetime.now(timezone.utc) + timedelta(seconds=30)

    failed = observer.observe(
        WorkflowStepContext(
            workflow=copy_model(waiting.workflow, execution_deadline=deadline),
            incident=incident,
            step=step,
            step_index=0,
            request=request,
            idempotency_key="workflow-managed/0/clamped",
        )
    )

    assert deadline > datetime.now(timezone.utc), (
        "the workflow's own deadline must not have passed, or this proves nothing"
    )
    assert failed.status is WorkflowStepStatus.FAILED
    assert "timed out" in failed.error
    assert len(store.list_notifications()) == 1


def test_sqlite_persists_hyperpod_identity(tmp_path) -> None:
    path = tmp_path / "managed.db"
    first = SqliteStore(str(path))
    identity = HyperPodNodeIdentity(
        cluster_name="hp-cluster",
        node_logical_id="worker-group-1",
        instance_id="i-new",
        kubernetes_node_name="k8s-new",
        status="Running",
        aliases=["worker-group-1", "i-new", "k8s-new"],
        retired_aliases=["i-old", "k8s-old"],
        observed_at=NOW,
    )
    first.save_hyperpod_node_identity(identity)
    first.close()

    second = SqliteStore(str(path))
    loaded = second.get_hyperpod_node_identity("hp-cluster", "worker-group-1")
    second.close()

    assert loaded == identity


class RebindingAdapter:
    def __init__(self) -> None:
        self.calls = []

    def supports(self, step):
        return True

    def execute(self, context):
        self.calls.append((context.step.operation, list(context.step.node_ids)))
        if context.step.operation is WorkflowOperation.REPLACE_NODE:
            return WorkflowStepOutcome.succeeded(
                details={"node_rebindings": {"k8s-old": "k8s-new"}}
            )
        return WorkflowStepOutcome.succeeded()


def test_executor_rebinds_future_validation_and_restore_steps() -> None:
    store = build_store()
    incident, workflow, _, _ = state(WorkflowOperation.REPLACE_NODE)
    operations = [
        WorkflowOperation.REPLACE_NODE,
        WorkflowOperation.VALIDATE_GPU,
        WorkflowOperation.VALIDATE_HOST,
        WorkflowOperation.VALIDATE_FABRIC,
        WorkflowOperation.RESTORE_SCHEDULING,
    ]
    workflow = copy_model(
        workflow,
        status=WorkflowStatus.PENDING,
        official_steps=[
            workflow_step(operation, "owner", node_ids=["k8s-old"])
            for operation in operations
        ],
    )
    store.save_incident(incident)
    store.save_workflow(workflow)
    adapter = RebindingAdapter()
    executor = active_workflow_executor(
        store, [adapter], operations, executor_id="executor"
    )

    result = execute_workflow(executor, workflow.request_id)

    assert result.status is WorkflowStatus.SUCCEEDED
    assert adapter.calls[0][1] == ["k8s-old"]
    assert all(nodes == ["k8s-new"] for _, nodes in adapter.calls[1:])
    assert store.get_incident(incident.incident_id).node_ids == ["k8s-new"]
    assert all(
        step.node_ids == ["k8s-new"]
        for step in store.get_workflow(workflow.request_id).official_steps[1:]
    )


def test_managed_adapter_uses_observer_without_provider_mutation() -> None:
    (store, lifecycle, _, observer, _, incident, workflow, step, request) = (
        observer_fixture()
    )
    store.save_agent(agent("k8s-old", "i-old", "boot-old"))
    adapter = ManagedRecoveryObserverAdapter({OWNER}, observer)

    outcome = adapter.execute(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=step,
            step_index=0,
            request=request,
            idempotency_key="workflow-managed/0/adapter",
        )
    )

    assert outcome.status is WorkflowStepStatus.WAITING
    assert lifecycle.mutations == []
