from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from gpu_fault.adapters.common import (
    ANNOTATION_EFA_PLUGIN_RESTART_OPERATION,
    ANNOTATION_EFA_PLUGIN_RESTART_POD_UID,
    ANNOTATION_EFA_PLUGIN_RESTART_STARTED_AT,
    ANNOTATION_FENCING,
    ANNOTATION_GPU_PLUGIN_RESTART_OPERATION,
    ANNOTATION_GPU_PLUGIN_RESTART_POD_UID,
    ANNOTATION_GPU_PLUGIN_RESTART_STARTED_AT,
    ANNOTATION_INCIDENT,
    ANNOTATION_MECHANICAL_INSPECTION_COMPLETE,
    ANNOTATION_PREVIOUS_UNSCHEDULABLE,
    QUARANTINE_TAINT,
    NodeIsolationRejected,
    quarantine_taint_value,
)
from gpu_fault.adapters.kubernetes.primitives import (
    NodePatchConflict,
    node_scheduling_snapshot,
    patch_node_with_retry,
)
from gpu_fault.execution import (
    WorkflowStepContext,
    WorkflowStepOutcome,
)
from gpu_fault.models import (
    IncidentState,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
)
from gpu_fault.store import NotFoundError

LOGGER = logging.getLogger(__name__)

ANNOTATION_EFA_PLUGIN_RESTART_INCIDENT = "gpu-fault.io/efa-plugin-restart-incident"

ANNOTATION_GPU_PLUGIN_RESTART_INCIDENT = "gpu-fault.io/gpu-plugin-restart-incident"


@dataclass(frozen=True)
class _PluginRestartKeys:
    operation: str
    incident: str
    pod_uid: str
    started: str


_EFA_PLUGIN_RESTART_KEYS = _PluginRestartKeys(
    operation=ANNOTATION_EFA_PLUGIN_RESTART_OPERATION,
    incident=ANNOTATION_EFA_PLUGIN_RESTART_INCIDENT,
    pod_uid=ANNOTATION_EFA_PLUGIN_RESTART_POD_UID,
    started=ANNOTATION_EFA_PLUGIN_RESTART_STARTED_AT,
)

_GPU_PLUGIN_RESTART_KEYS = _PluginRestartKeys(
    operation=ANNOTATION_GPU_PLUGIN_RESTART_OPERATION,
    incident=ANNOTATION_GPU_PLUGIN_RESTART_INCIDENT,
    pod_uid=ANNOTATION_GPU_PLUGIN_RESTART_POD_UID,
    started=ANNOTATION_GPU_PLUGIN_RESTART_STARTED_AT,
)


class KubernetesNodeOperationsMixin:
    # Attributes supplied by the composed concrete implementation.
    store: Any

    _annotations: Callable[..., Any]
    _efa_plugin_pods: Callable[..., Any]
    _node_allocatable: Callable[..., Any]
    _pod_name: Callable[..., Any]
    _pod_ready: Callable[..., Any]
    _pod_uid: Callable[..., Any]
    _resource_version: Callable[..., Any]
    _taints: Callable[..., Any]
    _unschedulable: Callable[..., Any]
    alert_sender: Callable[..., Any]
    core: Any
    mechanical_email_builder: Any
    notification_sink: Any
    ownership_provider: Any

    def _restart_device_plugin(
        self, context: WorkflowStepContext
    ) -> WorkflowStepOutcome:
        parameters = context.step.parameters
        efa = context.step.operation is WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN
        namespace = str(parameters.get("plugin_namespace", "kube-system"))
        selector = str(
            parameters.get(
                "plugin_label_selector",
                (
                    "name=hyperpod-dependencies-aws-efa-k8s-device-plugin"
                    if efa
                    else "app.kubernetes.io/name=nvidia-device-plugin"
                ),
            )
        )
        resource_name = str(
            parameters.get(
                "resource_name",
                ("vpc.amazonaws.com/efa" if efa else "nvidia.com/gpu"),
            )
        )
        keys = _EFA_PLUGIN_RESTART_KEYS if efa else _GPU_PLUGIN_RESTART_KEYS
        # The expected device count is what "restored" means. Defaulting it
        # to 1 declared an 8-GPU node healthy the moment one device came
        # back; without a known count there is nothing to judge against.
        raw_expected = parameters.get("expected_count")
        try:
            expected = int(raw_expected) if raw_expected is not None else 0
        except (TypeError, ValueError):
            expected = 0
        if expected < 1:
            return WorkflowStepOutcome.failed(
                "device plugin restart requires a positive expected_count for "
                f"{resource_name}; refusing to judge node health without it",
                details={
                    "safety_rejection": True,
                    "resource_name": resource_name,
                    "expected_count": raw_expected,
                },
            )
        timeout_seconds = int(parameters.get("restart_timeout_seconds", 180))
        node_results: dict[str, dict[str, Any]] = {}
        waiting = False
        now = datetime.now(timezone.utc)
        for node_id in context.step.node_ids:
            try:
                result = self._restart_device_plugin_node(
                    context,
                    node_id,
                    keys=keys,
                    namespace=namespace,
                    selector=selector,
                    resource_name=resource_name,
                    expected=expected,
                    timeout_seconds=timeout_seconds,
                    now=now,
                )
            except NodePatchConflict:
                # The node is unchanged; the next pass re-reads and retries.
                waiting = True
                node_results[node_id] = {"patch_conflict_retry": True}
                continue
            if isinstance(result, WorkflowStepOutcome):
                return result
            node_result, node_waiting = result
            waiting = waiting or node_waiting
            node_results[node_id] = node_result
        if waiting:
            return WorkflowStepOutcome.waiting(
                operation_id=context.idempotency_key,
                details={"node_results": node_results},
            )
        return WorkflowStepOutcome.succeeded(
            operation_id=context.idempotency_key,
            details={"node_results": node_results},
        )

    def _restart_device_plugin_node(
        self,
        context: WorkflowStepContext,
        node_id: str,
        *,
        keys: _PluginRestartKeys,
        namespace: str,
        selector: str,
        resource_name: str,
        expected: int,
        timeout_seconds: int,
        now: datetime,
    ) -> WorkflowStepOutcome | tuple[dict[str, Any], bool]:
        node = self.core.read_node(node_id)
        allocatable = self._node_allocatable(node, resource_name)
        annotations = self._annotations(node)
        operation = annotations.get(keys.operation)
        old_uid = annotations.get(keys.pod_uid, "")
        started = self._plugin_restart_started(annotations.get(keys.started))
        took_over: str | None = None
        if operation and operation != context.idempotency_key:
            owner_incident = annotations.get(keys.incident)
            if (
                owner_incident != context.incident.incident_id
                and not self._plugin_restart_is_stale(
                    started, now=now, timeout_seconds=timeout_seconds
                )
            ):
                return WorkflowStepOutcome.failed(
                    f"node {node_id} has another device plugin restart "
                    f"in progress: {operation}"
                )
            # The owner is this incident's own earlier workflow, or a foreign
            # restart that outlived its own timeout: either way it will never
            # clear its annotation, and leaving it there poisons every later
            # restart on this node.
            self._clear_plugin_restart(node_id, keys)
            took_over = owner_incident or operation
            operation = None
            old_uid = ""
            started = None
        pods = self._efa_plugin_pods(
            node_id,
            namespace=namespace,
            label_selector=selector,
        )
        ready_new_pods = [
            pod
            for pod in pods
            if self._pod_ready(pod) and self._pod_uid(pod) != old_uid
        ]
        if allocatable >= expected and (operation is None or ready_new_pods):
            if operation == context.idempotency_key:
                self._clear_plugin_restart(node_id, keys)
            result: dict[str, Any] = {
                "allocatable": allocatable,
                "expected": expected,
                "already_healthy": operation is None,
                "replacement_pod_uids": [self._pod_uid(pod) for pod in ready_new_pods],
            }
            if took_over is not None:
                result["took_over_incident"] = took_over
            return result, False
        if operation == context.idempotency_key:
            if started is None:
                started = now
            if (now - started).total_seconds() >= timeout_seconds:
                self._clear_plugin_restart(node_id, keys)
                return WorkflowStepOutcome.failed(
                    f"device plugin did not restore {resource_name} on {node_id}",
                    details={
                        "node_id": node_id,
                        "allocatable": allocatable,
                        "expected": expected,
                        "plugin_pods": [self._pod_name(pod) for pod in pods],
                    },
                )
            return {
                "allocatable": allocatable,
                "expected": expected,
                "waiting_for_replacement_pod": True,
            }, True
        if not pods:
            self._mark_plugin_restart(node_id, keys, context, pod_uid="", now=now)
            result = {
                "allocatable": allocatable,
                "expected": expected,
                "waiting_for_daemonset_pod": True,
            }
            if took_over is not None:
                result["took_over_incident"] = took_over
            return result, True
        if len(pods) != 1:
            return WorkflowStepOutcome.failed(
                f"expected at most one device plugin Pod on {node_id}, found {len(pods)}"
            )
        pod = pods[0]
        pod_uid = self._pod_uid(pod)
        pod_name = self._pod_name(pod)
        self._mark_plugin_restart(node_id, keys, context, pod_uid=pod_uid, now=now)
        try:
            self.core.delete_namespaced_pod(
                pod_name,
                namespace,
                grace_period_seconds=0,
            )
        except Exception as exc:
            if getattr(exc, "status", None) != 404:
                # The annotation was written for a delete that never
                # happened; left in place it would report "another restart
                # in progress" to every later workflow.
                try:
                    self._clear_plugin_restart(node_id, keys)
                except Exception:  # noqa: BLE001 - the delete error is the story
                    LOGGER.exception(
                        "cannot clear device plugin restart annotation on %s",
                        node_id,
                    )
                raise
        result = {
            "deleted_pod": f"{namespace}/{pod_name}",
            "deleted_pod_uid": pod_uid,
            "allocatable": allocatable,
            "expected": expected,
        }
        if took_over is not None:
            result["took_over_incident"] = took_over
        return result, True

    @staticmethod
    def _plugin_restart_started(raw: str | None) -> datetime | None:
        if not raw:
            return None
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _plugin_restart_is_stale(
        started: datetime | None,
        *,
        now: datetime,
        timeout_seconds: int,
    ) -> bool:
        """A foreign restart annotation may be taken over once it is stale.

        Age is the only evidence available: an annotation with no parsable
        start time cannot be proven stale, so it stays refused (fail closed).
        """
        if started is None:
            return False
        return (now - started).total_seconds() >= timeout_seconds

    def _mark_plugin_restart(
        self,
        node_id: str,
        keys: _PluginRestartKeys,
        context: WorkflowStepContext,
        *,
        pod_uid: str,
        now: datetime,
    ) -> None:
        patch_node_with_retry(
            self.core,
            node_id,
            lambda node: {
                "metadata": {
                    "resourceVersion": self._resource_version(node),
                    "annotations": {
                        keys.operation: context.idempotency_key,
                        keys.incident: context.incident.incident_id,
                        keys.pod_uid: pod_uid,
                        keys.started: now.isoformat(),
                    },
                }
            },
        )

    def _clear_plugin_restart(self, node_id: str, keys: _PluginRestartKeys) -> None:
        patch_node_with_retry(
            self.core,
            node_id,
            lambda node: {
                "metadata": {
                    "resourceVersion": self._resource_version(node),
                    "annotations": {
                        keys.operation: None,
                        keys.incident: None,
                        keys.pod_uid: None,
                        keys.started: None,
                    },
                }
            },
        )

    def _check_mechanicals(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        expected = f"{context.incident.incident_id}:{context.workflow.fencing_token}"
        pending = []
        confirmations = {}
        for node_id in context.step.node_ids:
            node = self.core.read_node(node_id)
            actual = self._annotations(node).get(
                ANNOTATION_MECHANICAL_INSPECTION_COMPLETE
            )
            if actual != expected:
                pending.append(node_id)
            else:
                confirmations[node_id] = actual
        if pending:
            notification_id = None
            if self.notification_sink is not None:
                notification = self.notification_sink.save_notification_if_absent(
                    self.mechanical_email_builder.build(
                        cluster_id=context.incident.cluster_id,
                        incident_id=context.incident.incident_id,
                        workflow_id=context.workflow.request_id,
                        node_ids=pending,
                        link_id=context.step.parameters.get("nvlink_link_id"),
                        pci_bdf=context.step.parameters.get("pci_bdf"),
                        occurrence_counts=context.step.parameters.get(
                            "nvlink_occurrence_counts", {}
                        ),
                        annotation=(ANNOTATION_MECHANICAL_INSPECTION_COMPLETE),
                        annotation_value=expected,
                        xid=int(context.step.parameters.get("xid", 74)),
                    )
                )
                notification_id = notification.notification_id
                if self.alert_sender is not None:
                    self.alert_sender(notification_id)
            return WorkflowStepOutcome.waiting(
                details={
                    "pending_nodes": pending,
                    "required_annotation": (ANNOTATION_MECHANICAL_INSPECTION_COMPLETE),
                    "required_annotation_value": expected,
                    "acknowledgement_commands": [
                        "kubectl annotate node "
                        f"{node_id} "
                        f"{ANNOTATION_MECHANICAL_INSPECTION_COMPLETE}"
                        f"='{expected}' --overwrite"
                        for node_id in pending
                    ],
                    "required_evidence": (
                        "physical seating and applicable NVLink "
                        "connections inspected; reseating completed "
                        "when required"
                    ),
                    "notification_id": notification_id,
                }
            )
        return WorkflowStepOutcome.succeeded(
            operation_id=context.idempotency_key,
            details={
                "confirmed_nodes": sorted(confirmations),
                "confirmation_values": confirmations,
            },
        )

    def _observed_snapshot(self, node_id: str) -> dict[str, Any]:
        """Re-read the node so the recorded 'after' state is observed."""
        try:
            return node_scheduling_snapshot(self.core.read_node(node_id))
        except Exception as exc:  # noqa: BLE001 - the patch already succeeded
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _isolate(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        baselines: dict[str, dict[str, Any]] = {}
        conflicts: list[str] = []
        for node_id in context.step.node_ids:
            # Snapshot inside the callback: it sees the node as read before
            # the patch, whatever object the client hands back afterwards.
            before: dict[str, Any] = {}

            def isolation_body(node: Any) -> dict[str, Any]:
                before.clear()
                before.update(node_scheduling_snapshot(node))
                return self._node_isolation_patch(node, context)

            try:
                patch_node_with_retry(self.core, node_id, isolation_body)
            except NodeIsolationRejected as exc:
                return WorkflowStepOutcome.failed(
                    str(exc),
                    details={
                        "safety_rejection": True,
                        "node_id": node_id,
                    },
                )
            except NodePatchConflict:
                conflicts.append(node_id)
                continue
            except Exception as exc:
                if getattr(exc, "status", None) == 404:
                    # A node this workflow must isolate is not there to be
                    # isolated. Calling that "already isolated" was memory
                    # standing in for observation; nothing downstream can
                    # verify a node that cannot be read.
                    return WorkflowStepOutcome.failed(
                        f"node {node_id} is absent; isolation cannot be observed",
                        details={
                            "safety_rejection": True,
                            "node_id": node_id,
                            "absent": True,
                        },
                    )
                raise
            baselines[node_id] = {
                "before": dict(before),
                "after": self._observed_snapshot(node_id),
            }
        isolated = [
            node_id for node_id in context.step.node_ids if node_id not in conflicts
        ]
        if conflicts:
            return WorkflowStepOutcome.waiting(
                details={
                    "patch_conflict_retry": conflicts,
                    "isolated_nodes": isolated,
                    "node_baselines": baselines,
                },
            )
        return WorkflowStepOutcome.succeeded(
            operation_id=context.idempotency_key,
            details={
                "isolated_nodes": isolated,
                "node_baselines": baselines,
            },
        )

    def _gpu_fault_isolated(self, node: Any) -> bool:
        annotations = self._annotations(node)
        return (
            ANNOTATION_INCIDENT in annotations
            or ANNOTATION_FENCING in annotations
            or any(item.get("key") == QUARANTINE_TAINT for item in self._taints(node))
        )

    def _node_restore_patch(
        self, node: Any, context: WorkflowStepContext
    ) -> dict[str, Any] | None:
        if not self._gpu_fault_isolated(node):
            # Nothing on the node says gpu-fault isolated it. A workflow
            # that fail-closed at compile time never reached
            # MARK_UNSCHEDULABLE, so the validated restore that closes its
            # incident has nothing to undo here; refusing left such
            # incidents ESCALATED forever (2026-09-06, REMEDIATE_EFA_DRIVER
            # with no profile owner). Restoring a node nobody isolated is
            # a no-op, not a theft of another incident's isolation, which
            # the ownership check below still refuses.
            return None
        annotations = self._annotations(node)
        if annotations.get(
            ANNOTATION_INCIDENT
        ) != context.incident.incident_id or annotations.get(ANNOTATION_FENCING) != str(
            context.workflow.fencing_token
        ):
            raise NodeIsolationRejected(
                "isolation ownership does not match incident/fencing token"
            )
        taints = [
            item for item in self._taints(node) if item.get("key") != QUARANTINE_TAINT
        ]
        was_unschedulable = (
            annotations.get(
                ANNOTATION_PREVIOUS_UNSCHEDULABLE,
                "false",
            ).lower()
            == "true"
        )
        return {
            "metadata": {
                "resourceVersion": self._resource_version(node),
                "annotations": {
                    ANNOTATION_INCIDENT: None,
                    ANNOTATION_FENCING: None,
                    ANNOTATION_PREVIOUS_UNSCHEDULABLE: None,
                },
            },
            "spec": {
                "unschedulable": was_unschedulable,
                "taints": taints,
            },
        }

    def _restore(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        already_restored: list[str] = []
        absent: list[str] = []
        conflicts: list[str] = []
        baselines: dict[str, dict[str, Any]] = {}
        for node_id in context.step.node_ids:
            before: dict[str, Any] = {}
            isolated_before: list[bool] = []

            def restore_body(node: Any) -> dict[str, Any] | None:
                before.clear()
                before.update(node_scheduling_snapshot(node))
                isolated_before[:] = [self._gpu_fault_isolated(node)]
                return self._node_restore_patch(node, context)

            try:
                patch_node_with_retry(self.core, node_id, restore_body)
            except NodeIsolationRejected as exc:
                return WorkflowStepOutcome.failed(
                    f"node {node_id} {exc}",
                    details={"safety_rejection": True, "node_id": node_id},
                )
            except NodePatchConflict:
                conflicts.append(node_id)
                continue
            except Exception as exc:
                if getattr(exc, "status", None) == 404:
                    # There is no node left to schedule onto; restoring it
                    # is a no-op, exactly like a node nobody isolated.
                    absent.append(node_id)
                    continue
                raise
            if not any(isolated_before):
                already_restored.append(node_id)
                baselines[node_id] = {"before": dict(before), "after": dict(before)}
                continue
            baselines[node_id] = {
                "before": dict(before),
                "after": self._observed_snapshot(node_id),
            }
        restored = [
            node_id
            for node_id in context.step.node_ids
            if node_id not in absent and node_id not in conflicts
        ]
        if conflicts:
            # RESTORE_SCHEDULING is idempotent: the next pass re-reads the
            # node and re-derives the patch, so a lost race is a retry, not
            # a failure that leaves the node cordoned for good.
            return WorkflowStepOutcome.waiting(
                details={
                    "patch_conflict_retry": conflicts,
                    "restored_nodes": restored,
                    "already_restored_nodes": already_restored,
                    "absent_nodes": absent,
                    "node_baselines": baselines,
                },
            )
        return WorkflowStepOutcome.succeeded(
            operation_id=context.idempotency_key,
            details={
                "restored_nodes": restored,
                "already_restored_nodes": already_restored,
                "absent_nodes": absent,
                "node_baselines": baselines,
            },
        )

    def _incident_workflow_is_terminal(self, incident_id: str) -> bool:
        """Whether another incident's workflow has finished.

        Fails closed on every error path: an unreachable control plane or
        an unknown incident must not license taking a node away from a
        workflow that could still be running.
        """
        if self.store is not None:
            try:
                incident = self.store.get_incident(incident_id)
                workflow = self.store.get_workflow(incident.workflow_request_id)
            except (NotFoundError, KeyError, TypeError):
                return False
            return workflow.status in {
                WorkflowStatus.BLOCKED,
                WorkflowStatus.SUCCEEDED,
                WorkflowStatus.FAILED,
                WorkflowStatus.SUPERSEDED,
            }
        if self.ownership_provider is None:
            return False
        try:
            return bool(
                self.ownership_provider.incident_workflow_is_terminal(incident_id)
            )
        except Exception:
            return False

    def _incident_ownership(self, incident_id: str) -> tuple[bool, bool]:
        if self.store is not None:
            try:
                incident = self.store.get_incident(incident_id)
                workflow = self.store.get_workflow(incident.workflow_request_id)
            except (NotFoundError, KeyError, TypeError):
                return False, False
            terminal = workflow.status in {
                WorkflowStatus.BLOCKED,
                WorkflowStatus.SUCCEEDED,
                WorkflowStatus.FAILED,
                WorkflowStatus.SUPERSEDED,
            }
            quarantine_hold = (
                incident.state is IncidentState.QUARANTINED
                or (
                    WorkflowOperation.QUARANTINE in workflow.completed_operations
                    and WorkflowOperation.RESTORE_SCHEDULING
                    not in workflow.completed_operations
                )
                or any(
                    execution.operation is WorkflowOperation.REPLACE_NODE
                    and execution.details.get("action") == "SPARE_FAILOVER"
                    for execution in workflow.step_executions
                )
            )
            return terminal, quarantine_hold
        if self.ownership_provider is None:
            return False, False
        try:
            if hasattr(
                self.ownership_provider,
                "incident_ownership",
            ):
                report = self.ownership_provider.incident_ownership(incident_id)
                return (
                    bool(report.known and report.terminal),
                    bool(report.quarantine_hold),
                )
            return (
                bool(
                    self.ownership_provider.incident_workflow_is_terminal(incident_id)
                ),
                False,
            )
        except Exception:
            return False, False

    @staticmethod
    def _workflow_preserves_quarantine(
        workflow: WorkflowRequest,
    ) -> bool:
        operations = {step.operation for step in workflow.official_steps}
        return (
            WorkflowOperation.QUARANTINE in operations
            and WorkflowOperation.RESTORE_SCHEDULING not in operations
        )

    def _can_take_over_node_isolation(
        self,
        node: Any,
        incident_id: str,
        context: WorkflowStepContext,
    ) -> bool:
        terminal, quarantine_hold = self._incident_ownership(incident_id)
        return terminal and (
            not quarantine_hold or self._workflow_preserves_quarantine(context.workflow)
        )

    def _node_isolation_patch(
        self, node: Any, context: WorkflowStepContext
    ) -> dict[str, Any]:
        annotations = self._annotations(node)
        existing_incident = annotations.get(ANNOTATION_INCIDENT)
        existing_fencing = annotations.get(ANNOTATION_FENCING)
        newer_same_incident_generation = False
        existing_fencing_token = None
        if existing_fencing is not None:
            try:
                existing_fencing_token = int(existing_fencing)
            except ValueError as exc:
                raise ValueError("node has invalid gpu-fault fencing token") from exc
        if (
            existing_incident == context.incident.incident_id
            and existing_fencing_token is not None
        ):
            if context.workflow.fencing_token < existing_fencing_token:
                raise NodeIsolationRejected(
                    "node is controlled by a newer workflow generation"
                )
            newer_same_incident_generation = (
                context.workflow.fencing_token > existing_fencing_token
            )
        takeover = bool(
            existing_incident
            and (
                existing_incident != context.incident.incident_id
                or existing_fencing != str(context.workflow.fencing_token)
            )
            and (
                newer_same_incident_generation
                or (
                    existing_incident != context.incident.incident_id
                    and self._can_take_over_node_isolation(
                        node, existing_incident, context
                    )
                )
                or (
                    existing_incident == context.incident.incident_id
                    and existing_fencing_token is None
                    and self._can_take_over_node_isolation(
                        node, existing_incident, context
                    )
                )
            )
        )
        if (
            existing_incident
            and not takeover
            and (
                existing_incident != context.incident.incident_id
                or existing_fencing != str(context.workflow.fencing_token)
            )
        ):
            raise NodeIsolationRejected(
                "node is already isolated by another incident/token"
            )
        previous_unschedulable = annotations.get(ANNOTATION_PREVIOUS_UNSCHEDULABLE)
        # This annotation is the scheduling baseline from before the first
        # isolation. A successor incident takes over an already cordoned node,
        # so resampling here would turn that intermediate state into the new
        # baseline and make validation-first restore leave the node cordoned.
        if previous_unschedulable is None:
            previous_unschedulable = str(self._unschedulable(node)).lower()
        taints = self._taints(node)
        if takeover:
            taints = [item for item in taints if item.get("key") != QUARANTINE_TAINT]
        owned_taint_values = {
            context.incident.incident_id,
            quarantine_taint_value(context.incident.incident_id),
        }
        if not any(
            item.get("key") == QUARANTINE_TAINT
            and item.get("value") in owned_taint_values
            for item in taints
        ):
            taints.append(
                {
                    "key": QUARANTINE_TAINT,
                    "value": quarantine_taint_value(context.incident.incident_id),
                    "effect": "NoSchedule",
                }
            )
        return {
            "metadata": {
                "resourceVersion": self._resource_version(node),
                "annotations": {
                    ANNOTATION_INCIDENT: context.incident.incident_id,
                    ANNOTATION_FENCING: str(context.workflow.fencing_token),
                    ANNOTATION_PREVIOUS_UNSCHEDULABLE: (previous_unschedulable),
                },
            },
            "spec": {
                "unschedulable": True,
                "taints": taints,
            },
        }
