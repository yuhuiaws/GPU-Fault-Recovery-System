from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Callable

from gpu_fault.adapters.kubernetes.primitives import patch_node_with_retry
from gpu_fault.host_health import (
    NodeHealthCategory,
    NodeHealthFinding,
)
from gpu_fault.hyperpod_spares import (
    SPARE_POOL_STATE_ANNOTATION,
    SPARE_RESERVED_AT_ANNOTATION,
    HyperPodSpareCoordinator,
    SparePoolState,
)
from gpu_fault.markers import retire_markers_for_incident
from gpu_fault.models import (
    AdvisoryNotification,
    RecoveryAction,
    Severity,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
    WorkloadState,
)
from gpu_fault.store import NotFoundError


LOGGER = logging.getLogger(__name__)
HEALTH_ANNOTATION = "gpu-fault.io/spare-health"
FAILURES_ANNOTATION = "gpu-fault.io/spare-health-failures"
INCIDENT_ANNOTATION = "gpu-fault.io/spare-health-incident"
UNAVAILABLE_AT_ANNOTATION = "gpu-fault.io/spare-health-unavailable-at"
LAST_ALERT_AT_ANNOTATION = "gpu-fault.io/spare-health-last-alert-at"


class SpareHealthState(StrEnum):
    HEALTHY = "HEALTHY"
    SUSPECT = "SUSPECT"
    REBOOT_PENDING = "REBOOT_PENDING"
    RECHECKING = "RECHECKING"
    UNAVAILABLE = "UNAVAILABLE"


class SpareHealthReasonClass(StrEnum):
    CONFIGURATION = "CONFIGURATION"
    OBSERVATION = "OBSERVATION"
    HARDWARE = "HARDWARE"


class SpareReservationReclaimer:
    """Decides whether a warm-spare reservation still has a living owner."""

    def __init__(
        self,
        coordinator: HyperPodSpareCoordinator,
        store: Any,
        *,
        now: Callable[[], datetime],
        ttl_seconds: float,
    ) -> None:
        self.coordinator = coordinator
        self.store = store
        self.now = now
        self.ttl_seconds = ttl_seconds

    def reason(
        self,
        node_name: str,
        kubernetes_node: Any,
        reservation: str,
    ) -> str | None:
        """Why a reservation no longer has an owner, or ``None`` to keep it.

        A spare a successful failover consumed is a training node now and is
        never touched, nor is any node still running GPU pods. With a store,
        the owning workflow decides: terminal means orphaned. Without one
        (the regional executor runs storeless), or when the incident is
        unknown, only the reservation timestamp can decide, so a reservation
        written before the timestamp existed is kept -- a documented limit.
        """
        if self.coordinator._active_gpu_pod_reasons(node_name):
            return None
        annotations = HyperPodSpareHealthController._annotations(kubernetes_node)
        reserved_at = HyperPodSpareHealthController._timestamp(
            annotations.get(SPARE_RESERVED_AT_ANNOTATION),
            node_name=node_name,
            annotation=SPARE_RESERVED_AT_ANNOTATION,
        )
        expired = (
            reserved_at is not None
            and (self.now() - reserved_at).total_seconds() >= self.ttl_seconds
        )
        ttl_reason = f"reservation by {reservation} exceeded {self.ttl_seconds:g}s TTL"
        if self.store is None:
            return ttl_reason if expired else None
        try:
            incident = self.store.get_incident(reservation)
        except NotFoundError:
            return ttl_reason if expired else None
        workflow = None
        if incident.workflow_request_id:
            try:
                workflow = self.store.get_workflow(incident.workflow_request_id)
            except NotFoundError:
                workflow = None
        if workflow is not None:
            consumed = any(
                execution.operation is WorkflowOperation.REPLACE_NODE
                and execution.status is WorkflowStepStatus.SUCCEEDED
                and node_name in (execution.details.get("activated_spare_nodes") or [])
                for execution in workflow.step_executions
            )
            if consumed:
                return None
            if workflow.status in {
                WorkflowStatus.PENDING,
                WorkflowStatus.SAFETY_PENDING,
                WorkflowStatus.RUNNING,
            }:
                return ttl_reason if expired else None
            return (
                f"workflow {workflow.request_id} of {reservation} is "
                f"{workflow.status.value}"
            )
        return f"incident {reservation} has no workflow"


class HyperPodSpareHealthController:
    """Continuously remediates declared, unallocated HyperPod spares."""

    def __init__(
        self,
        coordinator: HyperPodSpareCoordinator,
        orchestrator,
        store,
        *,
        failure_threshold: int = 2,
        alert_sender: Callable[[str], object] | None = None,
        unavailable_recheck_seconds: float = 3600,
        unavailable_alert_seconds: float = 86400,
        now: Callable[[], datetime] | None = None,
        reservation_ttl_seconds: float = 86400,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be positive")
        self.coordinator = coordinator
        self.orchestrator = orchestrator
        self.store = store
        self.failure_threshold = failure_threshold
        self.alert_sender = alert_sender
        if unavailable_recheck_seconds < 1:
            raise ValueError("unavailable_recheck_seconds must be positive")
        if unavailable_alert_seconds < 1:
            raise ValueError("unavailable_alert_seconds must be positive")
        self.unavailable_recheck_seconds = unavailable_recheck_seconds
        self.unavailable_alert_seconds = unavailable_alert_seconds
        if reservation_ttl_seconds < 1:
            raise ValueError("reservation_ttl_seconds must be positive")
        self.reservation_ttl_seconds = reservation_ttl_seconds
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.reclaimer = SpareReservationReclaimer(
            coordinator,
            store,
            now=self.now,
            ttl_seconds=reservation_ttl_seconds,
        )
        self._active_reservations = 0
        self._reclaimed_reservations = 0
        self._reservations_observed_at: datetime | None = None

    def metrics_snapshot(self) -> dict[str, Any]:
        """Gauge-able reservation state from the last ``scan``."""
        return {
            "spare_reservations_active": self._active_reservations,
            "spare_reservations_reclaimed_total": self._reclaimed_reservations,
            "spare_reservations_observed_at": (
                self._reservations_observed_at.isoformat()
                if self._reservations_observed_at is not None
                else None
            ),
        }

    def scan(self) -> list[dict[str, Any]]:
        results = []
        active_reservations = 0
        lifecycle = self.coordinator.lifecycle
        for node in lifecycle.list_nodes(enrich=True):
            if (
                node.kubernetes_labels.get(self.coordinator.spare_label)
                != self.coordinator.spare_label_value
            ):
                continue
            node_name = self.coordinator._kubernetes_node_name(node)
            if node_name is None:
                continue
            try:
                kubernetes_node = self.coordinator.core.read_node(node_name)
                reservation = self.coordinator._annotation(kubernetes_node)
                if reservation:
                    reason = self.reclaimer.reason(
                        node_name, kubernetes_node, reservation
                    )
                    if reason is None:
                        active_reservations += 1
                        continue
                    self.coordinator.release([node_name], reservation)
                    self._reclaimed_reservations += 1
                    results.append(
                        {
                            "node_id": node_name,
                            "state": SpareHealthState.SUSPECT.value,
                            "reasons": ["reclaimed stale spare reservation: " + reason],
                            "incident_id": reservation,
                            "notification_id": None,
                        }
                    )
                    continue
                results.append(
                    self._reconcile(
                        node,
                        node_name,
                        kubernetes_node=kubernetes_node,
                    )
                )
            except Exception:
                LOGGER.exception(
                    "spare health reconciliation failed: node=%s",
                    node_name,
                )
                results.append(
                    self._result(
                        node_name,
                        SpareHealthState.SUSPECT,
                        ["spare health reconciliation failed"],
                    )
                )
        self._active_reservations = active_reservations
        self._reservations_observed_at = self.now()
        return results

    def _reconcile(
        self,
        node,
        node_name: str,
        *,
        kubernetes_node: Any | None = None,
    ) -> dict[str, Any]:
        if kubernetes_node is None:
            kubernetes_node = self.coordinator.core.read_node(node_name)
        annotations = self._annotations(kubernetes_node)
        state = self._health_state(
            annotations.get(HEALTH_ANNOTATION),
            node_name=node_name,
        )
        incident_id = annotations.get(INCIDENT_ANNOTATION)
        observed_after = None
        workflow = None
        if incident_id:
            try:
                incident = self.store.get_incident(incident_id)
                if incident.workflow_request_id:
                    workflow = self.store.get_workflow(incident.workflow_request_id)
                    if workflow.status is WorkflowStatus.SUCCEEDED:
                        observed_after = workflow.updated_at
            except NotFoundError:
                LOGGER.warning(
                    "clearing stale spare-health incident annotation: "
                    "node=%s incident=%s",
                    node_name,
                    incident_id,
                )
                incident_id = None
                self._patch(
                    node_name,
                    state=SpareHealthState.SUSPECT,
                    failures=0,
                    incident_id=None,
                )
                kubernetes_node = self.coordinator.core.read_node(node_name)
                annotations = self._annotations(kubernetes_node)
                state = SpareHealthState.SUSPECT

        reasons = self.coordinator.health_reasons(
            self.coordinator.lifecycle.config.cluster_name,
            node,
            node_name,
            observed_after=observed_after,
            check_pool_state=False,
        )
        if state is SpareHealthState.UNAVAILABLE:
            return self._reconcile_unavailable(
                node_name,
                annotations,
                reasons,
                incident_id=incident_id,
            )
        if state is SpareHealthState.REBOOT_PENDING:
            if workflow is None or workflow.status in {
                WorkflowStatus.FAILED,
                WorkflowStatus.BLOCKED,
            }:
                return self._mark_unavailable(
                    node_name,
                    incident_id or "unknown",
                    reasons or ["spare reboot workflow failed"],
                )
            if workflow.status is not WorkflowStatus.SUCCEEDED:
                return self._result(node_name, state, reasons)
            self._patch(
                node_name,
                state=SpareHealthState.RECHECKING,
                failures=0,
                incident_id=incident_id,
            )
            return self._result(
                node_name,
                SpareHealthState.RECHECKING,
                reasons,
                incident_id,
            )
        if state is SpareHealthState.RECHECKING:
            if not reasons:
                self._patch(
                    node_name,
                    state=SpareHealthState.HEALTHY,
                    failures=0,
                    incident_id=None,
                    unavailable_at=None,
                    last_alert_at=None,
                )
                self._retire_incident_markers(node_name, incident_id)
                return self._result(node_name, SpareHealthState.HEALTHY, [])
            failures = self._failure_count(annotations, node_name=node_name) + 1
            if failures < self.failure_threshold:
                self._patch(
                    node_name,
                    state=SpareHealthState.RECHECKING,
                    failures=failures,
                    incident_id=incident_id,
                )
                return self._result(
                    node_name,
                    SpareHealthState.RECHECKING,
                    reasons,
                    incident_id,
                )
            if not self._hardware_reasons(reasons):
                return self._defer_non_hardware(
                    node_name,
                    reasons,
                    failures=failures,
                    incident_id=incident_id,
                )
            return self._mark_unavailable(node_name, incident_id or "unknown", reasons)
        if not reasons:
            if state is not SpareHealthState.HEALTHY:
                self._patch(
                    node_name,
                    state=SpareHealthState.HEALTHY,
                    failures=0,
                    incident_id=None,
                    unavailable_at=None,
                    last_alert_at=None,
                )
            return self._result(node_name, SpareHealthState.HEALTHY, [])

        failures = self._failure_count(annotations, node_name=node_name) + 1
        if failures < self.failure_threshold:
            self._patch(
                node_name,
                state=SpareHealthState.SUSPECT,
                failures=failures,
                incident_id=None,
            )
            return self._result(node_name, SpareHealthState.SUSPECT, reasons)
        if not self._hardware_reasons(reasons):
            return self._defer_non_hardware(
                node_name,
                reasons,
                failures=failures,
                incident_id=None,
            )

        existing_incident = self._active_remediation(node)
        if existing_incident is not None:
            self._patch(
                node_name,
                state=SpareHealthState.REBOOT_PENDING,
                failures=failures,
                incident_id=existing_incident.incident_id,
            )
            return self._result(
                node_name,
                SpareHealthState.REBOOT_PENDING,
                reasons,
                existing_incident.incident_id,
            )

        agent = self._matching_agent(node)
        incarnation = (
            getattr(agent, "agent_incarnation_id", None)
            or getattr(agent, "boot_id", None)
            or node.instance_id
            or "unknown"
        )
        event_id = f"spare-health-{node.node_logical_id}-{incarnation}"
        finding = NodeHealthFinding(
            finding_id=event_id,
            event_id=event_id,
            cluster_id=self.coordinator.lifecycle.config.cluster_name,
            node_id=node_name,
            observed_at=self.now(),
            category=self._hardware_category(reasons),
            severity=Severity.CRITICAL,
            reason="; ".join(reasons),
            recommended_action=RecoveryAction.REBOOT_NODE,
            runtime_profile_version=(
                getattr(agent, "runtime_profile_version", None)
                if agent is not None
                else None
            ),
            workload_state=WorkloadState.IDLE,
        )
        incident, _ = self.orchestrator.ingest_node_health(finding)
        self._patch(
            node_name,
            state=SpareHealthState.REBOOT_PENDING,
            failures=failures,
            incident_id=incident.incident_id,
        )
        return self._result(
            node_name,
            SpareHealthState.REBOOT_PENDING,
            reasons,
            incident.incident_id,
        )

    def _retire_incident_markers(self, node_name: str, incident_id: str | None) -> None:
        """The spare is healthy after its own remediation succeeded: the
        markers that asked for that remediation no longer describe the node.
        Best effort -- the health verdict was already recorded on the node."""
        if not incident_id:
            return
        try:
            retire_markers_for_incident(
                self.store,
                incident_id,
                reason=f"spare {node_name} rechecked healthy after remediation",
                retired_by="spare-health",
            )
        except Exception:  # noqa: BLE001 - never undo a HEALTHY verdict for this
            LOGGER.exception(
                "cannot retire markers of incident %s for spare %s",
                incident_id,
                node_name,
            )

    @staticmethod
    def _reason_class(
        reason: str,
    ) -> SpareHealthReasonClass:
        normalized = reason.strip().lower()
        if normalized.startswith(
            (
                "unreserved spare is schedulable",
                "reserved by incident ",
                "spare pool state is ",
                "active gpu resource pods exist",
                "active gpu clients",
            )
        ):
            return SpareHealthReasonClass.CONFIGURATION
        if normalized.startswith(
            (
                "kubernetes node is not ready",
                "hyperpod status is ",
                "active trusted node fault marker exists",
                "active gpu health finding exists",
            )
        ):
            return SpareHealthReasonClass.HARDWARE
        if normalized.startswith("hyperpod node health is "):
            return (
                SpareHealthReasonClass.OBSERVATION
                if normalized.endswith("unknown")
                else SpareHealthReasonClass.HARDWARE
            )
        return SpareHealthReasonClass.OBSERVATION

    @classmethod
    def _hardware_reasons(cls, reasons: list[str]) -> list[str]:
        return [
            reason
            for reason in reasons
            if cls._reason_class(reason) is SpareHealthReasonClass.HARDWARE
        ]

    @classmethod
    def _hardware_category(cls, reasons: list[str]) -> NodeHealthCategory:
        return (
            NodeHealthCategory.GPU
            if any("gpu" in reason.lower() for reason in reasons)
            else NodeHealthCategory.SYSTEM_LOG
        )

    def _defer_non_hardware(
        self,
        node_name: str,
        reasons: list[str],
        *,
        failures: int,
        incident_id: str | None,
    ) -> dict[str, Any]:
        self._patch(
            node_name,
            state=SpareHealthState.SUSPECT,
            failures=failures,
            incident_id=incident_id,
            unavailable_at=None,
            last_alert_at=None,
        )
        classes = sorted({self._reason_class(reason).value for reason in reasons})
        advisory_id = incident_id or f"spare-health-advisory-{node_name}"
        notification = self.store.save_notification_if_absent(
            AdvisoryNotification(
                deduplication_key=(f"{advisory_id}/spare-health/" + "-".join(classes)),
                cluster_name=(self.coordinator.lifecycle.config.cluster_name),
                incident_id=advisory_id,
                subject=(
                    "[GPU configuration warning] HyperPod spare node "
                    f"{node_name} needs operator attention"
                ),
                body_text="\n".join(
                    [
                        f"Spare node: {node_name}",
                        "No physical recovery action was submitted.",
                        "Reason classes: " + ", ".join(classes),
                        "Reasons: " + "; ".join(reasons),
                    ]
                ),
                support_case_draft=(
                    "Correct the spare configuration or restore "
                    f"observability for {node_name}."
                ),
            )
        )
        if self.alert_sender is not None:
            self.alert_sender(notification.notification_id)
        return self._result(
            node_name,
            SpareHealthState.SUSPECT,
            reasons,
            incident_id,
            notification.notification_id,
        )

    def _matching_agent(self, node):
        agents = [
            agent
            for agent in self.store.list_agents(
                self.coordinator.lifecycle.config.cluster_name
            )
            if agent.node_id in node.aliases
        ]
        return agents[0] if len(agents) == 1 else None

    def _active_remediation(self, node):
        supported = {
            RecoveryAction.RESET_GPU,
            RecoveryAction.REBOOT_NODE,
            RecoveryAction.REPLACE_NODE,
        }
        candidates = self.store.list_active_markers_for_nodes(
            set(node.aliases),
            supported,
            # Tenant scope (H-14): this cluster's markers only, so a
            # same-named node in another tenant cannot drive remediation here.
            self.coordinator.lifecycle.config.cluster_name,
        )
        for marker in candidates:
            try:
                incident = self.store.get_incident(marker.incident_id)
            except NotFoundError:
                continue
            if not incident.workflow_request_id:
                continue
            try:
                workflow = self.store.get_workflow(incident.workflow_request_id)
            except NotFoundError:
                continue
            if workflow.status in {
                WorkflowStatus.PENDING,
                WorkflowStatus.SAFETY_PENDING,
                WorkflowStatus.RUNNING,
            }:
                return incident
        return None

    def _mark_unavailable(
        self,
        node_name: str,
        incident_id: str,
        reasons: list[str],
    ) -> dict[str, Any]:
        observed_at = self.now()
        self._patch(
            node_name,
            state=SpareHealthState.UNAVAILABLE,
            failures=self.failure_threshold,
            incident_id=incident_id,
            unschedulable=True,
            unavailable_at=observed_at,
            last_alert_at=observed_at,
        )
        notification = self._unavailable_notification(
            node_name,
            incident_id,
            reasons,
            observed_at=observed_at,
        )
        return self._result(
            node_name,
            SpareHealthState.UNAVAILABLE,
            reasons,
            incident_id,
            notification.notification_id,
        )

    def _unavailable_notification(
        self,
        node_name: str,
        incident_id: str,
        reasons: list[str],
        *,
        observed_at: datetime,
    ) -> AdvisoryNotification:
        alert_bucket = int(observed_at.timestamp() // self.unavailable_alert_seconds)
        notification = self.store.save_notification_if_absent(
            AdvisoryNotification(
                deduplication_key=(
                    f"{incident_id}/spare-node-unavailable/{alert_bucket}"
                ),
                cluster_name=(self.coordinator.lifecycle.config.cluster_name),
                incident_id=incident_id,
                subject=(
                    "[GPU action required] HyperPod spare node "
                    f"{node_name} is unavailable"
                ),
                body_text="\n".join(
                    [
                        f"Spare node: {node_name}",
                        f"Incident: {incident_id}",
                        "The node remained unhealthy after reboot.",
                        "Reasons: " + "; ".join(reasons),
                    ]
                ),
                support_case_draft=(f"Investigate or replace spare node {node_name}."),
            )
        )
        if self.alert_sender is not None:
            self.alert_sender(notification.notification_id)
        return notification

    def _reconcile_unavailable(
        self,
        node_name: str,
        annotations: dict[str, str],
        reasons: list[str],
        *,
        incident_id: str | None,
    ) -> dict[str, Any]:
        observed_at = self.now()
        unavailable_at = self._timestamp(
            annotations.get(UNAVAILABLE_AT_ANNOTATION),
            node_name=node_name,
            annotation=UNAVAILABLE_AT_ANNOTATION,
        )
        if unavailable_at is None:
            unavailable_at = observed_at
            self._patch(
                node_name,
                state=SpareHealthState.UNAVAILABLE,
                failures=self.failure_threshold,
                incident_id=incident_id,
                unschedulable=True,
                unavailable_at=unavailable_at,
            )
        last_alert_at = self._timestamp(
            annotations.get(LAST_ALERT_AT_ANNOTATION),
            node_name=node_name,
            annotation=LAST_ALERT_AT_ANNOTATION,
        )
        notification_id = None
        if incident_id and (
            last_alert_at is None
            or (observed_at - last_alert_at).total_seconds()
            >= self.unavailable_alert_seconds
        ):
            notification = self._unavailable_notification(
                node_name,
                incident_id,
                reasons or ["spare remains unavailable"],
                observed_at=observed_at,
            )
            notification_id = notification.notification_id
            self._patch(
                node_name,
                state=SpareHealthState.UNAVAILABLE,
                failures=self.failure_threshold,
                incident_id=incident_id,
                unschedulable=True,
                unavailable_at=unavailable_at,
                last_alert_at=observed_at,
            )
            last_alert_at = observed_at
        if (
            observed_at - unavailable_at
        ).total_seconds() < self.unavailable_recheck_seconds:
            return self._result(
                node_name,
                SpareHealthState.UNAVAILABLE,
                reasons,
                incident_id,
                notification_id,
            )
        if reasons:
            if not self._hardware_reasons(reasons):
                return self._defer_non_hardware(
                    node_name,
                    reasons,
                    failures=self.failure_threshold,
                    incident_id=incident_id,
                )
            self._patch(
                node_name,
                state=SpareHealthState.UNAVAILABLE,
                failures=self.failure_threshold,
                incident_id=incident_id,
                unschedulable=True,
                unavailable_at=observed_at,
                last_alert_at=last_alert_at,
            )
            return self._result(
                node_name,
                SpareHealthState.UNAVAILABLE,
                reasons,
                incident_id,
                notification_id,
            )
        self._patch(
            node_name,
            state=SpareHealthState.RECHECKING,
            failures=0,
            incident_id=incident_id,
            unschedulable=True,
            unavailable_at=None,
            last_alert_at=None,
        )
        return self._result(
            node_name,
            SpareHealthState.RECHECKING,
            [],
            incident_id,
            notification_id,
        )

    def _patch(
        self,
        node_name: str,
        *,
        state: SpareHealthState,
        failures: int,
        incident_id: str | None,
        unschedulable: bool | None = None,
        unavailable_at: datetime | None | object = ...,
        last_alert_at: datetime | None | object = ...,
    ) -> None:
        def body(node: Any) -> dict[str, Any]:
            patch: dict[str, Any] = {
                "metadata": {
                    "resourceVersion": (self.coordinator._resource_version(node)),
                    "annotations": {
                        HEALTH_ANNOTATION: state.value,
                        FAILURES_ANNOTATION: str(failures),
                        INCIDENT_ANNOTATION: incident_id,
                        SPARE_POOL_STATE_ANNOTATION: (self._pool_state(state).value),
                        **(
                            {}
                            if unavailable_at is ...
                            else {
                                UNAVAILABLE_AT_ANNOTATION: (
                                    unavailable_at.isoformat()
                                    if isinstance(unavailable_at, datetime)
                                    else None
                                )
                            }
                        ),
                        **(
                            {}
                            if last_alert_at is ...
                            else {
                                LAST_ALERT_AT_ANNOTATION: (
                                    last_alert_at.isoformat()
                                    if isinstance(last_alert_at, datetime)
                                    else None
                                )
                            }
                        ),
                    },
                }
            }
            if unschedulable is not None:
                patch["spec"] = {"unschedulable": unschedulable}
            return patch

        patch_node_with_retry(self.coordinator.core, node_name, body)

    @staticmethod
    def _health_state(value: str | None, *, node_name: str) -> SpareHealthState:
        if not value:
            return SpareHealthState.HEALTHY
        try:
            return SpareHealthState(value)
        except ValueError:
            LOGGER.warning(
                "invalid spare health annotation; treating as SUSPECT: "
                "node=%s value=%r",
                node_name,
                value,
            )
            return SpareHealthState.SUSPECT

    @staticmethod
    def _failure_count(annotations: dict[str, str], *, node_name: str) -> int:
        raw = annotations.get(FAILURES_ANNOTATION, "0")
        try:
            value = int(raw)
        except (TypeError, ValueError):
            LOGGER.warning(
                "invalid spare failure count; resetting to zero: node=%s value=%r",
                node_name,
                raw,
            )
            return 0
        if value < 0:
            LOGGER.warning(
                "negative spare failure count; resetting to zero: node=%s value=%r",
                node_name,
                raw,
            )
            return 0
        return value

    @staticmethod
    def _timestamp(
        value: str | None,
        *,
        node_name: str,
        annotation: str,
    ) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            LOGGER.warning(
                "invalid spare timestamp annotation: node=%s annotation=%s value=%r",
                node_name,
                annotation,
                value,
            )
            return None
        return (
            parsed.replace(tzinfo=timezone.utc)
            if parsed.tzinfo is None
            else parsed.astimezone(timezone.utc)
        )

    @staticmethod
    def _pool_state(
        health_state: SpareHealthState,
    ) -> SparePoolState:
        if health_state is SpareHealthState.HEALTHY:
            return SparePoolState.AVAILABLE
        if health_state is SpareHealthState.UNAVAILABLE:
            return SparePoolState.UNAVAILABLE
        return SparePoolState.REMEDIATING

    @staticmethod
    def _annotations(node: Any) -> dict[str, str]:
        metadata = (
            node.get("metadata", {})
            if isinstance(node, dict)
            else getattr(node, "metadata", None)
        )
        return (
            metadata.get("annotations", {})
            if isinstance(metadata, dict)
            else getattr(metadata, "annotations", {}) or {}
        )

    @staticmethod
    def _result(
        node_id,
        state,
        reasons,
        incident_id=None,
        notification_id=None,
    ):
        return {
            "node_id": node_id,
            "state": state.value,
            "reasons": reasons,
            "incident_id": incident_id,
            "notification_id": notification_id,
        }
