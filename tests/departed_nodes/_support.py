"""A control plane with one departed node per provider, on any store.

The live shape of 2026-10-14: a spot reclaim left the node's fleet Agent
``ACTIVE`` with an expired lease, an ESCALATED incident whose workflow went
BLOCKED / NEEDS_OPERATOR without changing the node, and -- the mid-release
variant of 2026-10-01 -- a PENDING record never dispatched. Times are the real
clock moved backwards (``ago``), never a fixed date, so retention windows and
guards are judged exactly as in production.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from gpu_fault.departed_node_cleanup import (
    DepartedNodeCleanup,
    DepartedNodeCleanupConfig,
)
from gpu_fault.departed_node_provider import ProviderLookupError
from gpu_fault.fleet import AgentLifecycleState, AgentRecord, FleetRegistry
from gpu_fault.kubernetes_node_inventory import KubernetesNodeInventoryReport
from gpu_fault.models import (
    BlockedKind,
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.orchestration.incident_closure import IncidentClosureService
from gpu_fault.regional import RegionalClusterLifecycle, RegionalClusterRegistration
from tests._builders import fault_incident, workflow_request, workflow_step

HP_CLUSTER = "gpu-hyperpod"
HP_NAME = "gpu-hyperpod-hp"
EKS_CLUSTER = "gpu-eks"
REGION = "us-west-2"
HP_INSTANCE = "i-0a000001"
HP_NODE = f"hyperpod-{HP_INSTANCE}"
HP_PRESENT = "hyperpod-i-0a000002"
EKS_INSTANCE = "i-0e000001"
EKS_NODE = "ip-10-0-38-180.us-west-2.compute.internal"
EKS_PRESENT = "ip-10-0-41-164.us-west-2.compute.internal"
EKS_PRESENT_INSTANCE = "i-0e000002"
PLUGIN = WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN


def now() -> datetime:
    return datetime.now(timezone.utc)


def ago(**delta: float) -> datetime:
    return now() - timedelta(**delta)


def registration(
    cluster_id: str,
    *,
    provider: str = "hyperpod",
    lifecycle: RegionalClusterLifecycle = RegionalClusterLifecycle.ACTIVE,
) -> RegionalClusterRegistration:
    return RegionalClusterRegistration(
        cluster_id=cluster_id,
        region=REGION,
        hyperpod_cluster_name=HP_NAME if provider == "hyperpod" else "",
        eks_cluster_arn=f"arn:aws:eks:{REGION}:123456789012:cluster/{cluster_id}",
        provider=provider,
        token_sha256="a" * 64,
        lifecycle_state=lifecycle,
        agent_endpoint_allowed_cidrs=["10.0.0.0/16"],
    )


def agent(
    cluster_id: str,
    node_id: str,
    *,
    instance_id: str | None,
    last_seen: datetime,
    lease_seconds: float = 120,
    lifecycle: AgentLifecycleState = AgentLifecycleState.ACTIVE,
) -> AgentRecord:
    return AgentRecord(
        cluster_id=cluster_id,
        node_id=node_id,
        endpoint="https://10.0.38.180:9099",
        agent_version="2026.10.14",
        artifact_sha256="a" * 64,
        policy_version="610",
        runtime_profile_version="regional-v1",
        config_digest="b" * 64,
        allowed_operations=[],
        node_instance_id=instance_id,
        agent_incarnation_id=f"incarnation-{node_id}",
        first_seen_at=last_seen - timedelta(days=1),
        last_seen_at=last_seen,
        lease_expires_at=last_seen + timedelta(seconds=lease_seconds),
        generation=4,
        lifecycle_state=lifecycle,
    )


def inventory_report(
    cluster_id: str,
    nodes: list[str],
    *,
    observed_at: datetime,
    spares: list[str] | None = None,
    executor_id: str = "executor-a",
) -> KubernetesNodeInventoryReport:
    return KubernetesNodeInventoryReport(
        cluster_id=cluster_id,
        executor_id=executor_id,
        observed_at=observed_at,
        resource_version="12345",
        node_names=nodes,
        spare_node_names=spares or [],
    )


@dataclass
class FakeProvider:
    """``ProviderMembership`` with a scripted answer per provider."""

    hyperpod: dict[str, set[str]] = field(default_factory=dict)
    ec2: dict[str, str] = field(default_factory=dict)
    failure: str | None = None
    calls: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)

    def hyperpod_instance_ids(self, cluster_name: str, region: str) -> frozenset[str]:
        self.calls.append(("hyperpod", (cluster_name, region)))
        if self.failure:
            raise ProviderLookupError(self.failure)
        return frozenset(self.hyperpod.get(cluster_name, set()))

    def ec2_instance_states(self, region: str, instance_ids: Any) -> dict[str, str]:
        ids = tuple(sorted(instance_ids))
        self.calls.append(("ec2", ids))
        if self.failure:
            raise ProviderLookupError(self.failure)
        return {item: self.ec2[item] for item in ids if item in self.ec2}


@dataclass
class FakeNotifications:
    sent: list[str] = field(default_factory=list)

    def send(self, notification_id: str) -> None:
        self.sent.append(notification_id)


@dataclass
class World:
    store: Any
    registry: FleetRegistry
    closure: IncidentClosureService
    provider: FakeProvider
    notifications: FakeNotifications
    cleanup: DepartedNodeCleanup

    def report(
        self,
        cluster_id: str,
        nodes: list[str],
        *,
        at: datetime,
        spares: list[str] | None = None,
    ) -> None:
        self.store.record_kubernetes_node_inventory(
            inventory_report(cluster_id, nodes, observed_at=at, spares=spares),
            received_at=at,
        )

    def reports_since(
        self,
        cluster_id: str,
        nodes: list[str],
        *,
        minutes: int,
        spares: list[str] | None = None,
    ) -> None:
        """One report a minute for ``minutes`` minutes, the last one now."""

        for minute in range(minutes, -1, -1):
            self.report(cluster_id, nodes, at=ago(minutes=minute), spares=spares)

    def notification_subjects(self) -> list[str]:
        return sorted(
            self.store.get_notification(item).subject
            for item in self.notifications.sent
        )


def world(store: Any, **config: Any) -> World:
    registry = FleetRegistry(store, "s" * 32)
    closure = IncidentClosureService(store)
    provider = FakeProvider()
    notifications = FakeNotifications()
    cleanup = DepartedNodeCleanup(
        store,
        registry=registry,
        incident_closure=closure,
        provider=provider,
        notifications=notifications,
        config=DepartedNodeCleanupConfig(**config),
        waiting_ttl=timedelta(hours=2),
    )
    return World(store, registry, closure, provider, notifications, cleanup)


def blocked_never_changed(
    store: Any, cluster_id: str, node_id: str, *, suffix: str, **values: Any
) -> tuple[str, str]:
    """The ESCALATED incident whose BLOCKED / NEEDS_OPERATOR workflow never
    changed the node (``efa_kubernetes_allocatable_mismatch``, live)."""

    request_id = f"workflow-{suffix}"
    incident_id = f"incident-{suffix}"
    created = ago(hours=2)
    store.save_workflow(
        workflow_request(
            request_id,
            incident_id,
            status=WorkflowStatus.BLOCKED,
            blocked_kind=BlockedKind.NEEDS_OPERATOR,
            fencing_token=3,
            execution_epoch=1,
            official_steps=[workflow_step(PLUGIN, node_ids=[node_id])],
            created_at=created,
            updated_at=created,
            **values,
        )
    )
    store.save_incident(
        fault_incident(
            incident_id,
            f"event-{suffix}",
            event_type="efa_kubernetes_allocatable_mismatch",
            cluster_id=cluster_id,
            node_ids=[node_id],
            state=IncidentState.ESCALATED,
            workflow_request_id=request_id,
            fencing_token=3,
        )
    )
    return request_id, incident_id


def pending_never_dispatched(
    store: Any,
    cluster_id: str,
    node_id: str,
    *,
    suffix: str,
    age: timedelta = timedelta(hours=1),
    node_ids: list[str] | None = None,
) -> tuple[str, str]:
    """The PENDING record created for the node's last telemetry, never dispatched."""

    request_id = f"workflow-{suffix}"
    incident_id = f"incident-{suffix}"
    created = now() - age
    nodes = node_ids or [node_id]
    store.save_workflow(
        workflow_request(
            request_id,
            incident_id,
            status=WorkflowStatus.PENDING,
            fencing_token=2,
            execution_epoch=0,
            official_steps=[workflow_step(PLUGIN, node_ids=nodes)],
            created_at=created,
            updated_at=created,
        )
    )
    store.save_incident(
        fault_incident(
            incident_id,
            f"event-{suffix}",
            cluster_id=cluster_id,
            node_ids=nodes,
            state=IncidentState.ACTION_PENDING,
            workflow_request_id=request_id,
            fencing_token=2,
        )
    )
    return request_id, incident_id


def hyperpod_departure(store: Any, **config: Any) -> World:
    """HyperPod reclaimed ``HP_NODE``; ``HP_PRESENT`` is still there."""

    result = world(store, **config)
    store.save_regional_cluster(registration(HP_CLUSTER))
    store.save_agent(
        agent(HP_CLUSTER, HP_NODE, instance_id="node-uid-1", last_seen=ago(minutes=20))
    )
    store.save_agent(
        agent(
            HP_CLUSTER, HP_PRESENT, instance_id="node-uid-2", last_seen=ago(seconds=5)
        )
    )
    result.provider.hyperpod[HP_NAME] = {"i-0a000002"}
    blocked_never_changed(store, HP_CLUSTER, HP_NODE, suffix="hp-efa")
    pending_never_dispatched(store, HP_CLUSTER, HP_NODE, suffix="hp-xid")
    result.reports_since(HP_CLUSTER, [HP_PRESENT], minutes=12)
    return result


def eks_departure(store: Any, **config: Any) -> World:
    """EC2 terminated ``EKS_NODE``'s instance; ``EKS_PRESENT`` is running."""

    result = world(store, **config)
    store.save_regional_cluster(registration(EKS_CLUSTER, provider="eks-managed"))
    store.save_agent(
        agent(
            EKS_CLUSTER, EKS_NODE, instance_id=EKS_INSTANCE, last_seen=ago(minutes=20)
        )
    )
    store.save_agent(
        agent(
            EKS_CLUSTER,
            EKS_PRESENT,
            instance_id=EKS_PRESENT_INSTANCE,
            last_seen=ago(seconds=5),
        )
    )
    result.provider.ec2[EKS_INSTANCE] = "terminated"
    result.provider.ec2[EKS_PRESENT_INSTANCE] = "running"
    blocked_never_changed(store, EKS_CLUSTER, EKS_NODE, suffix="eks-efa")
    result.reports_since(EKS_CLUSTER, [EKS_PRESENT], minutes=12)
    return result
