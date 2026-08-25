from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Callable, Protocol, runtime_checkable

from gpu_fault.fleet import FleetRegistry
from gpu_fault.hyperpod import (
    HyperPodAdapterError,
    HyperPodLifecycleAdapter,
    HyperPodNode,
)
from gpu_fault.models import (
    AdvisoryNotification,
    NodeMarker,
    RecoveryAction,
    WorkflowStatus,
)


LOGGER = logging.getLogger(__name__)

SPARE_RESERVATION_ANNOTATION = "gpu-fault.io/spare-reservation"
SPARE_POOL_STATE_ANNOTATION = "gpu-fault.io/spare-pool-state"
HYPERPOD_NODE_HEALTH_LABEL = "sagemaker.amazonaws.com/node-health-status"
HYPERPOD_SCHEDULABLE = "Schedulable"
GPU_RESOURCE_NAME = "nvidia.com/gpu"

#: Recommended actions that mean "this node must not take on work".
#: A marker whose action is outside this set is observational (for
#: example ``RUN_DIAGNOSTICS`` on a TCP retransmission blip) and must
#: not disqualify an otherwise healthy warm spare -- warm-spare
#: failover is the only supported node replacement path, so treating
#: advisory noise as disqualifying makes real replacements fail.
SPARE_BLOCKING_ACTIONS = frozenset(
    {
        RecoveryAction.QUARANTINE,
        RecoveryAction.REPLACE_NODE,
        RecoveryAction.REBOOT_NODE,
        RecoveryAction.RESET_GPU,
        RecoveryAction.DRAIN,
        RecoveryAction.MARK_UNSCHEDULABLE,
        RecoveryAction.ESCALATE_OPERATOR,
    }
)

GpuClientChecker = Callable[[HyperPodNode, str, str], list[str]]


class SpareHealthPending(RuntimeError):
    """A remote spare-health check has not completed yet."""


def marker_disqualifies_spare(marker: NodeMarker, now: datetime) -> bool:
    """Whether ``marker`` means a node is unfit to be a warm spare.

    Only markers that actually ask for the node to stop taking work
    disqualify it. Advisory markers (``MONITOR_ONLY`` dispositions such
    as ``RUN_DIAGNOSTICS``) are recorded continuously on healthy nodes
    and never produce a workflow, so the ``SUCCEEDED``-workflow escape
    below can never clear them -- counting them would leave every node
    permanently ineligible.
    """
    if not marker.active or not marker.trusted:
        return False
    if marker.expires_at is not None and marker.expires_at <= now:
        return False
    if marker.recommended_action not in SPARE_BLOCKING_ACTIONS:
        return False
    return True


def describe_blocking_marker(marker: NodeMarker) -> str:
    """Operator-readable identity of a disqualifying marker.

    The bare "a marker exists" reason left no way to decide whether a
    spare could be released by hand; the caller has the only copy of
    this record, so it has to be named in the failure reason.
    """
    action = (
        marker.recommended_action.value
        if marker.recommended_action is not None
        else "unknown"
    )
    detail = f"{marker.marker_id} ({marker.severity.value}/{action}"
    if marker.raw_reason:
        detail += f": {marker.raw_reason}"
    return detail + ")"


@runtime_checkable
class NotificationSink(Protocol):
    """Persists advisory notifications so an operator can be paged.

    The data-plane executor reaches this coordinator through a fleet
    registry proxy that owns no local state, so the capability is
    optional and must be probed rather than assumed.
    """

    def save_notification_if_absent(
        self, notification: AdvisoryNotification
    ) -> AdvisoryNotification: ...


class SparePoolState(StrEnum):
    AVAILABLE = "AVAILABLE"
    REMEDIATING = "REMEDIATING"
    ALLOCATED = "ALLOCATED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class SpareAllocation:
    applicable: bool
    sufficient: bool
    required: int
    selected_node_ids: tuple[str, ...] = ()
    reason: str | None = None
    notification_id: str | None = None
    rejected_candidates: tuple[tuple[str, tuple[str, ...]], ...] = ()


class HyperPodSpareCoordinator:
    """Atomically activates healthy, topology-compatible warm spares."""

    def __init__(
        self,
        lifecycle: HyperPodLifecycleAdapter,
        store,
        core_api: Any,
        *,
        registry: FleetRegistry | None = None,
        remote_health_provider: Any | None = None,
        spare_label: str = "gpu-fault.io/spare",
        spare_label_value: str = "true",
        alert_sender: Callable[[str], object] | None = None,
    ) -> None:
        self.lifecycle = lifecycle
        self.store = store
        self.notification_sink: NotificationSink | None = (
            store if isinstance(store, NotificationSink) else None
        )
        self.core = core_api
        self.registry = registry
        self.remote_health_provider = remote_health_provider
        self.spare_label = spare_label
        self.spare_label_value = spare_label_value
        self.alert_sender = alert_sender

    def allocate(
        self,
        *,
        cluster_id: str,
        incident_id: str,
        fault_node_ids: list[str],
        gpu_client_checker: GpuClientChecker | None = None,
        local_only: bool = False,
    ) -> SpareAllocation:
        nodes = (
            self._kubernetes_nodes()
            if local_only
            else self.lifecycle.list_nodes(enrich=True)
        )
        declared_spare_names = self._declared_spare_names(nodes)
        spares = [
            node
            for node in nodes
            if (
                node.kubernetes_labels.get(self.spare_label) == self.spare_label_value
                or self._kubernetes_node_name(node) in declared_spare_names
            )
        ]
        if not spares:
            return SpareAllocation(
                applicable=False,
                sufficient=True,
                required=len(set(fault_node_ids)),
            )

        targets = self._resolve_targets(list(dict.fromkeys(fault_node_ids)), nodes)
        requirements: dict[tuple[str | None, str | None], int] = {}
        for target in targets:
            key = (
                target.instance_group_name,
                target.instance_type,
            )
            requirements[key] = requirements.get(key, 0) + 1

        candidates: dict[
            tuple[str | None, str | None], list[tuple[HyperPodNode, str]]
        ] = {}
        target_logical_ids = {target.node_logical_id for target in targets}
        rejections: dict[str, list[str]] = {}
        for spare in spares:
            if spare.node_logical_id in target_logical_ids:
                continue
            node_name = self._kubernetes_node_name(spare)
            if node_name is None:
                rejections[spare.node_logical_id] = [
                    "spare has no resolvable Kubernetes node name"
                ]
                continue
            unhealthy = self.health_reasons(
                cluster_id,
                spare,
                node_name,
                incident_id=incident_id,
                gpu_client_checker=gpu_client_checker,
            )
            if unhealthy:
                rejections[node_name] = unhealthy
                continue
            key = (
                spare.instance_group_name,
                spare.instance_type,
            )
            candidates.setdefault(key, []).append((spare, node_name))

        selected: list[tuple[HyperPodNode, str]] = []
        shortages = []
        for key, required in sorted(
            requirements.items(), key=lambda item: str(item[0])
        ):
            available = sorted(
                candidates.get(key, []),
                key=lambda item: item[0].node_logical_id,
            )
            if len(available) < required:
                shortages.append(
                    f"{key[0] or 'unknown-group'}/"
                    f"{key[1] or 'unknown-type'}: "
                    f"required={required}, healthy={len(available)}"
                )
            selected.extend(available[:required])
        if shortages:
            reason = "insufficient healthy HyperPod spares: " + "; ".join(shortages)
            if rejections:
                reason += " | rejected candidates: " + "; ".join(
                    f"{name} [{', '.join(items)}]"
                    for name, items in sorted(rejections.items())
                )
            notification = self._alert(
                cluster_id,
                incident_id,
                targets,
                len(targets),
                len(selected),
                reason,
            )
            return SpareAllocation(
                applicable=True,
                sufficient=False,
                required=len(targets),
                reason=reason,
                notification_id=(
                    notification.notification_id if notification is not None else None
                ),
                rejected_candidates=tuple(
                    (name, tuple(items)) for name, items in sorted(rejections.items())
                ),
            )

        activated: list[str] = []
        try:
            for spare, node_name in selected:
                self._reserve_and_activate(
                    spare,
                    node_name,
                    incident_id,
                    gpu_client_checker=gpu_client_checker,
                )
                activated.append(node_name)
        except SpareHealthPending:
            self._rollback(activated, incident_id)
            raise
        except Exception as exc:
            self._rollback(activated, incident_id)
            reason = (
                "failed to atomically reserve HyperPod spares: "
                f"{type(exc).__name__}: {exc}"
            )
            notification = self._alert(
                cluster_id,
                incident_id,
                targets,
                len(targets),
                len(activated),
                reason,
            )
            return SpareAllocation(
                applicable=True,
                sufficient=False,
                required=len(targets),
                reason=reason,
                notification_id=(
                    notification.notification_id if notification is not None else None
                ),
            )
        return SpareAllocation(
            applicable=True,
            sufficient=True,
            required=len(targets),
            selected_node_ids=tuple(activated),
        )

    def _kubernetes_nodes(self) -> list[HyperPodNode]:
        response = self.core.list_node()
        items = (
            response.get("items", [])
            if isinstance(response, dict)
            else getattr(response, "items", []) or []
        )
        return [self._kubernetes_provider_view(item) for item in items]

    @classmethod
    def _kubernetes_provider_view(cls, node: Any) -> HyperPodNode:
        metadata = cls._metadata(node)
        labels = dict(cls._labels(node))
        if isinstance(metadata, dict):
            node_name = str(metadata.get("name") or "")
        else:
            node_name = str(getattr(metadata, "name", "") or "")
        if not node_name:
            raise ValueError("Kubernetes node is missing metadata.name")
        spec = (
            node.get("spec", {})
            if isinstance(node, dict)
            else getattr(node, "spec", None)
        )
        provider_id = (
            spec.get("providerID")
            if isinstance(spec, dict)
            else getattr(spec, "provider_id", None)
        )
        instance_id = (
            node_name.removeprefix("hyperpod-")
            if node_name.startswith("hyperpod-i-")
            else None
        )
        if provider_id:
            provider_tail = str(provider_id).rsplit("/", 1)[-1]
            if "-i-" in provider_tail:
                instance_id = "i-" + provider_tail.rsplit("-i-", 1)[-1]
        status = "Running" if cls._node_ready(node) else "Unknown"
        labels.setdefault("kubernetes.io/hostname", node_name)
        return HyperPodNode(
            node_logical_id=node_name,
            instance_id=instance_id,
            instance_group_name=labels.get(
                "sagemaker.amazonaws.com/instance-group-name"
            ),
            instance_type=(
                labels.get("node.kubernetes.io/instance-type")
                or labels.get("beta.kubernetes.io/instance-type")
            ),
            status=status,
            availability_zone=labels.get("topology.kubernetes.io/zone"),
            kubernetes_labels=labels,
        )

    def _resolve_targets(
        self,
        fault_node_ids: list[str],
        nodes: list[HyperPodNode],
    ) -> list[HyperPodNode]:
        list_identities = getattr(self.store, "list_hyperpod_node_identities", None)
        identities = (
            list_identities(self.lifecycle.config.cluster_name)
            if list_identities is not None
            else []
        )
        targets = []
        for node_id in fault_node_ids:
            try:
                targets.extend(self.lifecycle.resolve_nodes([node_id], nodes=nodes))
                continue
            except (HyperPodAdapterError, KeyError, ValueError):
                pass
            identity = next(
                (
                    item
                    for item in identities
                    if node_id
                    in {
                        item.node_logical_id,
                        *item.aliases,
                        *item.retired_aliases,
                    }
                ),
                None,
            )
            if identity is None:
                raise ValueError(
                    "cannot resolve HyperPod fault node identity: " + node_id
                )
            targets.extend(
                self.lifecycle.resolve_nodes([identity.node_logical_id], nodes=nodes)
            )
        return targets

    def _declared_spare_names(self, provider_nodes: list[HyperPodNode]) -> set[str]:
        names = set()
        for provider_node in provider_nodes:
            name = self._kubernetes_node_name(provider_node)
            if not name:
                continue
            try:
                item = self.core.read_node(name)
            except Exception as exc:
                if isinstance(exc, KeyError) or getattr(exc, "status", None) == 404:
                    continue
                raise
            metadata = (
                item.get("metadata", {}) if isinstance(item, dict) else item.metadata
            )
            labels = (
                metadata.get("labels", {})
                if isinstance(metadata, dict)
                else metadata.labels
            )
            if (labels or {}).get(self.spare_label) == self.spare_label_value:
                names.add(name)
        return names

    def release(self, node_ids: list[str], incident_id: str) -> None:
        self._rollback(node_ids, incident_id)

    def _healthy(
        self,
        cluster_id: str,
        node: HyperPodNode,
        node_name: str,
        incident_id: str,
        *,
        gpu_client_checker: GpuClientChecker | None = None,
    ) -> bool:
        return not self.health_reasons(
            cluster_id,
            node,
            node_name,
            incident_id=incident_id,
            gpu_client_checker=gpu_client_checker,
        )

    def health_reasons(
        self,
        cluster_id: str,
        node: HyperPodNode,
        node_name: str,
        *,
        incident_id: str | None = None,
        observed_after: datetime | None = None,
        check_pool_state: bool = True,
        gpu_client_checker: GpuClientChecker | None = None,
    ) -> list[str]:
        reasons = []
        if node.status != "Running":
            reasons.append(f"HyperPod status is {node.status}")
        kubernetes_node = self.core.read_node(node_name)
        if not self._node_ready(kubernetes_node):
            reasons.append("Kubernetes node is not Ready")
        hyperpod_health = self._labels(kubernetes_node).get(HYPERPOD_NODE_HEALTH_LABEL)
        if hyperpod_health != HYPERPOD_SCHEDULABLE:
            reasons.append(f"HyperPod node health is {hyperpod_health or 'unknown'}")
        reservation = self._annotation(kubernetes_node)
        pool_state = self._pool_state(kubernetes_node)
        if reservation and reservation != incident_id:
            reasons.append(f"reserved by incident {reservation}")
        allowed_pool_states = {
            None,
            SparePoolState.AVAILABLE,
        }
        if reservation == incident_id:
            allowed_pool_states.add(SparePoolState.ALLOCATED)
        if check_pool_state and pool_state not in allowed_pool_states:
            reasons.append(f"spare pool state is {pool_state.value}")
        if not reservation and not self._unschedulable(kubernetes_node):
            reasons.append("unreserved spare is schedulable")
        reasons.extend(self._active_gpu_pod_reasons(node_name))
        if gpu_client_checker is not None:
            reasons.extend(gpu_client_checker(node, node_name, "candidate"))
        if self.remote_health_provider is not None:
            reasons.extend(
                self.remote_health_provider.spare_health_reasons(
                    cluster_id=cluster_id,
                    node_aliases=sorted(node.aliases),
                    incident_id=incident_id,
                    observed_after=observed_after,
                )
            )
            return reasons
        if self.registry is None:
            reasons.append("agent registry is unavailable")
            return reasons
        matching_agents = [
            agent
            for agent in self.store.list_agents(cluster_id)
            if agent.node_id in node.aliases
        ]
        if len(matching_agents) != 1:
            reasons.append(f"expected one matching agent, found {len(matching_agents)}")
            return reasons
        agent = matching_agents[0]
        if not self.registry.readiness(cluster_id, [agent.node_id]).ready:
            reasons.append("node agent is not fleet-ready")
        active_markers = [
            marker
            for marker in self.store.list_markers()
            if (
                self._marker_blocks_spare(marker)
                and set(marker.scope.node_ids).intersection(node.aliases)
                and (observed_after is None or marker.observed_at > observed_after)
            )
        ]
        if active_markers:
            reasons.append(
                "active trusted node fault marker exists: "
                + ", ".join(
                    describe_blocking_marker(marker) for marker in active_markers
                )
            )
        findings = [
            finding
            for alias in node.aliases
            for finding in self.store.list_gpu_findings(
                cluster_id, alias, active_only=True
            )
            if (observed_after is None or finding.observed_at > observed_after)
        ]
        if findings:
            reasons.append("active GPU health finding exists")
        return reasons

    def _marker_blocks_spare(self, marker) -> bool:
        if not marker_disqualifies_spare(marker, datetime.now(timezone.utc)):
            return False
        if marker.incident_id:
            try:
                incident = self.store.get_incident(marker.incident_id)
                workflow = self.store.get_workflow(incident.workflow_request_id)
                if workflow.status is WorkflowStatus.SUCCEEDED:
                    return False
            except (KeyError, TypeError):
                pass
        return True

    def _reserve_and_activate(
        self,
        provider_node: HyperPodNode,
        node_name: str,
        incident_id: str,
        *,
        gpu_client_checker: GpuClientChecker | None = None,
    ) -> None:
        node = self.core.read_node(node_name)
        reservation = self._annotation(node)
        if reservation == incident_id:
            return
        if reservation:
            raise ValueError(f"node {node_name} is reserved by {reservation}")
        occupancy_reasons = self._active_gpu_pod_reasons(node_name)
        if gpu_client_checker is not None:
            occupancy_reasons.extend(
                gpu_client_checker(provider_node, node_name, "activation")
            )
        if occupancy_reasons:
            raise ValueError(
                f"spare {node_name} became occupied: " + "; ".join(occupancy_reasons)
            )
        self.core.patch_node(
            node_name,
            {
                "metadata": {
                    "resourceVersion": self._resource_version(node),
                    "annotations": {
                        SPARE_RESERVATION_ANNOTATION: incident_id,
                        SPARE_POOL_STATE_ANNOTATION: (SparePoolState.ALLOCATED.value),
                    },
                },
                "spec": {"unschedulable": False},
            },
        )

    def _active_gpu_pod_reasons(self, node_name: str) -> list[str]:
        try:
            response = self.core.list_pod_for_all_namespaces(
                field_selector=f"spec.nodeName={node_name}"
            )
        except Exception as exc:
            return [f"cannot verify active GPU pods: {type(exc).__name__}: {exc}"]
        pods = (
            response.get("items", [])
            if isinstance(response, dict)
            else getattr(response, "items", []) or []
        )
        occupied = []
        for pod in pods:
            if self._pod_phase(pod) in {"Succeeded", "Failed"}:
                continue
            if self._pod_requests_gpu(pod):
                occupied.append(self._pod_name(pod))
        if not occupied:
            return []
        return ["active GPU resource pods exist: " + ", ".join(sorted(occupied))]

    @classmethod
    def _pod_requests_gpu(cls, pod: Any) -> bool:
        spec = (
            pod.get("spec", {}) if isinstance(pod, dict) else getattr(pod, "spec", None)
        )
        if isinstance(spec, dict):
            containers = [
                *(spec.get("initContainers") or []),
                *(spec.get("containers") or []),
                *(spec.get("ephemeralContainers") or []),
            ]
        else:
            containers = [
                *(getattr(spec, "init_containers", None) or []),
                *(getattr(spec, "containers", None) or []),
                *(getattr(spec, "ephemeral_containers", None) or []),
            ]
        return any(cls._container_requests_gpu(item) for item in containers)

    @staticmethod
    def _container_requests_gpu(container: Any) -> bool:
        resources = (
            container.get("resources", {})
            if isinstance(container, dict)
            else getattr(container, "resources", None)
        )
        for field in ("requests", "limits"):
            values = (
                resources.get(field, {})
                if isinstance(resources, dict)
                else getattr(resources, field, {}) or {}
            )
            value = values.get(GPU_RESOURCE_NAME, 0)
            try:
                if int(str(value)) > 0:
                    return True
            except (TypeError, ValueError):
                return True
        return False

    @staticmethod
    def _pod_phase(pod: Any) -> str:
        status = (
            pod.get("status", {})
            if isinstance(pod, dict)
            else getattr(pod, "status", None)
        )
        return str(
            status.get("phase", "")
            if isinstance(status, dict)
            else getattr(status, "phase", "")
        )

    @classmethod
    def _pod_name(cls, pod: Any) -> str:
        metadata = cls._metadata(pod)
        if isinstance(metadata, dict):
            namespace = metadata.get("namespace") or "default"
            name = metadata.get("name") or "unknown"
        else:
            namespace = getattr(metadata, "namespace", None) or "default"
            name = getattr(metadata, "name", None) or "unknown"
        return f"{namespace}/{name}"

    def _rollback(self, node_ids: list[str], incident_id: str) -> None:
        for node_name in reversed(node_ids):
            node = self.core.read_node(node_name)
            if self._annotation(node) != incident_id:
                continue
            self.core.patch_node(
                node_name,
                {
                    "metadata": {
                        "resourceVersion": self._resource_version(node),
                        "annotations": {
                            SPARE_RESERVATION_ANNOTATION: None,
                            SPARE_POOL_STATE_ANNOTATION: (
                                SparePoolState.AVAILABLE.value
                            ),
                        },
                    },
                    "spec": {"unschedulable": True},
                },
            )

    def _alert(
        self,
        cluster_id: str,
        incident_id: str,
        targets: list[HyperPodNode],
        required: int,
        healthy: int,
        reason: str,
    ) -> AdvisoryNotification | None:
        target_ids = ", ".join(target.node_logical_id for target in targets)
        body = "\n".join(
            [
                "HyperPod warm-spare capacity alert",
                "",
                f"Cluster: {cluster_id}",
                f"Incident: {incident_id}",
                f"Fault nodes: {target_ids}",
                f"Required compatible spare nodes: {required}",
                f"Healthy compatible spare nodes: {healthy}",
                f"Reason: {reason}",
                "",
                "The replacement workflow remains blocked and the "
                "fault nodes remain isolated. Add healthy compatible "
                "capacity or release an existing spare reservation.",
            ]
        )
        if self.notification_sink is None:
            LOGGER.warning(
                "spare capacity alert is not persistable for incident %s: %s",
                incident_id,
                reason,
            )
            return None
        sink = self.notification_sink
        notification = sink.save_notification_if_absent(
            AdvisoryNotification(
                deduplication_key=(f"{incident_id}/hyperpod-spare-insufficient"),
                cluster_name=cluster_id,
                incident_id=incident_id,
                subject=(
                    f"[GPU action required] {cluster_id}: "
                    "insufficient HyperPod spare nodes"
                ),
                body_text=body,
                support_case_draft=body,
            )
        )
        if self.alert_sender is not None:
            self.alert_sender(notification.notification_id)
        return notification

    def _kubernetes_node_name(self, node: HyperPodNode) -> str | None:
        return node.kubernetes_labels.get("kubernetes.io/hostname") or (
            f"hyperpod-{node.instance_id}" if node.instance_id else None
        )

    @staticmethod
    def _node_ready(node: Any) -> bool:
        conditions = (
            node.get("status", {}).get("conditions", [])
            if isinstance(node, dict)
            else getattr(getattr(node, "status", None), "conditions", [])
        )
        return any(
            (
                condition.get("type") == "Ready"
                and str(condition.get("status")).lower() == "true"
            )
            if isinstance(condition, dict)
            else (
                getattr(condition, "type", None) == "Ready"
                and str(getattr(condition, "status", "")).lower() == "true"
            )
            for condition in conditions
        )

    @staticmethod
    def _metadata(node: Any) -> Any:
        return (
            node.get("metadata", {})
            if isinstance(node, dict)
            else getattr(node, "metadata", None)
        )

    @classmethod
    def _labels(cls, node: Any) -> dict[str, str]:
        metadata = cls._metadata(node)
        return (
            metadata.get("labels", {})
            if isinstance(metadata, dict)
            else getattr(metadata, "labels", {}) or {}
        )

    @classmethod
    def _annotation(cls, node: Any) -> str | None:
        metadata = cls._metadata(node)
        annotations = (
            metadata.get("annotations", {})
            if isinstance(metadata, dict)
            else getattr(metadata, "annotations", {}) or {}
        )
        return annotations.get(SPARE_RESERVATION_ANNOTATION)

    @classmethod
    def _pool_state(cls, node: Any) -> SparePoolState | None:
        metadata = cls._metadata(node)
        annotations = (
            metadata.get("annotations", {})
            if isinstance(metadata, dict)
            else getattr(metadata, "annotations", {}) or {}
        )
        value = annotations.get(SPARE_POOL_STATE_ANNOTATION)
        return SparePoolState(value) if value else None

    @classmethod
    def _resource_version(cls, node: Any) -> str:
        metadata = cls._metadata(node)
        return (
            str(metadata.get("resourceVersion", ""))
            if isinstance(metadata, dict)
            else str(getattr(metadata, "resource_version", ""))
        )

    @staticmethod
    def _unschedulable(node: Any) -> bool:
        spec = (
            node.get("spec", {})
            if isinstance(node, dict)
            else getattr(node, "spec", None)
        )
        return bool(
            spec.get("unschedulable", False)
            if isinstance(spec, dict)
            else getattr(spec, "unschedulable", False)
        )
