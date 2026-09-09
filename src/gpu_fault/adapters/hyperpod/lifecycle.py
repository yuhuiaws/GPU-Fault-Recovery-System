from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gpu_fault.adapters.common import (
    ANNOTATION_FENCING,
    ANNOTATION_INCIDENT,
    QUARANTINE_TAINT,
    quarantine_taint_value,
)
from gpu_fault.adapters.hyperpod.confirmation import HyperPodConfirmationMixin
from gpu_fault.adapters.hyperpod.notifications import HyperPodNotificationMixin
from gpu_fault.adapters.kubernetes.adapter import KubernetesWorkflowAdapter
from gpu_fault.adapters.kubernetes.primitives import (
    NodePatchConflict,
    patch_node_with_retry,
)
from gpu_fault.adapters.node_action.adapter import NodeActionWorkflowAdapter
from gpu_fault.aws_errors import aws_configuration_error
from gpu_fault.execution import (
    WorkflowStepContext,
    WorkflowStepOutcome,
)
from gpu_fault.fleet import (
    FleetRegistry,
)
from gpu_fault.hyperpod import (
    HyperPodAdapterError,
    HyperPodLifecycleAdapter,
    HyperPodWorkflowDispatcher,
)
from gpu_fault.hyperpod_spares import (
    HyperPodSpareCoordinator,
    SpareAllocation,
    SpareHealthPending,
)
from gpu_fault.models import (
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.notifications import (
    RestartGuardEmailBuilder,
    WarmSpareReplacementEmailBuilder,
)
from gpu_fault.operation_registry import (
    OperationAdapter,
    operations_for_adapter,
)


@dataclass(frozen=True)
class ObservedIsolation:
    """What the scheduler currently shows for the nodes a step will mutate."""

    verified: list[str]
    nodes: dict[str, dict[str, Any]]


class HyperPodLifecycleStepAdapter(
    HyperPodConfirmationMixin,
    HyperPodNotificationMixin,
):
    OPERATIONS = operations_for_adapter(OperationAdapter.HYPERPOD)

    def __init__(
        self,
        adapter: HyperPodLifecycleAdapter,
        *,
        owner: str = "gpu-fault-hyperpod-adapter",
        registry: FleetRegistry | None = None,
        spare_coordinator: HyperPodSpareCoordinator | None = None,
        node_action_adapter: NodeActionWorkflowAdapter | None = None,
        kubernetes_adapter: KubernetesWorkflowAdapter | None = None,
        store: Any | None = None,
        notification_sink: Any | None = None,
        alert_sender=None,
        post_reboot_stabilization_seconds: int = 60,
    ) -> None:
        if post_reboot_stabilization_seconds < 0:
            raise ValueError("post_reboot_stabilization_seconds cannot be negative")
        self.owner = owner
        self.registry = registry
        self.spare_coordinator = spare_coordinator
        self.node_action_adapter = node_action_adapter
        self.kubernetes_adapter = kubernetes_adapter
        self.store = store
        self.notification_sink = notification_sink or store
        self.alert_sender = alert_sender
        self.post_reboot_stabilization_seconds = post_reboot_stabilization_seconds
        self.restart_email_builder = RestartGuardEmailBuilder()
        self.warm_spare_email_builder = WarmSpareReplacementEmailBuilder()
        self.dispatcher = HyperPodWorkflowDispatcher(adapter, execution_owner=owner)

    def supports(self, step: WorkflowStepSpec) -> bool:
        return step.execution_owner == self.owner and step.operation in self.OPERATIONS

    def release_spare_reservations(
        self, workflow: WorkflowRequest, incident_id: str
    ) -> list[str]:
        """Release warm spares a REPLACE_NODE step reserved but never consumed.

        A step that went WAITING with ``activated_spare_nodes`` and was then
        ended by the watchdog, the lifetime bound or a preemption never ran
        the ``except`` path that released them. The executor's terminal
        path should call this for every non-SUCCEEDED end; spares named by
        a SUCCEEDED failover are training nodes now and are kept.
        """
        if self.spare_coordinator is None:
            return []
        consumed: set[str] = set()
        pending: list[str] = []
        for execution in workflow.step_executions:
            if execution.operation is not WorkflowOperation.REPLACE_NODE:
                continue
            nodes = [
                str(item)
                for item in execution.details.get("activated_spare_nodes") or []
            ]
            if execution.status is WorkflowStepStatus.SUCCEEDED:
                consumed.update(nodes)
            else:
                pending.extend(nodes)
        release = [node for node in dict.fromkeys(pending) if node not in consumed]
        if release:
            self.spare_coordinator.release(release, incident_id)
        return release

    def _kubernetes_node_candidates(self, node_id: str) -> list[str]:
        candidates = [node_id]
        try:
            nodes = self.dispatcher.adapter.resolve_nodes([node_id])
        except (AttributeError, HyperPodAdapterError, KeyError, ValueError):
            return candidates
        for node in nodes:
            labels = getattr(node, "kubernetes_labels", {}) or {}
            for candidate in (
                labels.get("kubernetes.io/hostname"),
                f"hyperpod-{node.instance_id}" if node.instance_id else None,
            ):
                if candidate and candidate not in candidates:
                    candidates.append(str(candidate))
        return candidates

    @staticmethod
    def _isolation_problems(node: Any, context: WorkflowStepContext) -> list[str]:
        incident_id = context.incident.incident_id
        problems = []
        if not KubernetesWorkflowAdapter._unschedulable(node):
            problems.append("spec.unschedulable is not true")
        owned = {incident_id, quarantine_taint_value(incident_id)}
        if not any(
            item.get("key") == QUARANTINE_TAINT and item.get("value") in owned
            for item in KubernetesWorkflowAdapter._taints(node)
        ):
            problems.append("quarantine taint for this incident is missing")
        annotations = KubernetesWorkflowAdapter._annotations(node)
        if annotations.get(ANNOTATION_INCIDENT) != incident_id:
            problems.append(
                "incident annotation is "
                f"{annotations.get(ANNOTATION_INCIDENT)!r}, not this incident"
            )
        return problems

    def _reassert_isolation(
        self, context: WorkflowStepContext, details: dict[str, Any]
    ) -> dict[str, Any]:
        """Re-cordon a rebooting node the provider's bootstrap uncordoned.

        HyperPod's node bootstrap patches the Node after a managed reboot and
        clears ``spec.unschedulable`` (audit: ``hyperpod-service-linked-role``,
        ``bootstrap/v0.0.0``, DESTR-014 2026-09-09); the ownership annotations
        survive. Until RESTORE_SCHEDULING runs the node must stay cordoned, and
        the REPLACE_NODE rung an unconfirmed reboot escalates into observes
        isolation and refuses an uncordoned node. Runs on every poll of an
        unconfirmed RESTART_NODE, touches only a node this incident still owns,
        and records what it re-cordoned; a node absent mid-reboot or a patch
        conflict waits for the next poll.
        """

        observed = details.get("observed_isolation")
        if (
            self.kubernetes_adapter is None
            or context.step.operation is not WorkflowOperation.RESTART_NODE
            or not isinstance(observed, dict)
        ):
            return {}
        core = self.kubernetes_adapter.core
        incident_id = context.incident.incident_id
        reasserted = [str(item) for item in details.get("isolation_reasserted") or []]
        for node_id, record in observed.items():
            if not isinstance(record, dict) or not record.get("unschedulable"):
                continue
            kubernetes_node = str(record.get("kubernetes_node") or node_id)
            patched: list[bool] = []

            def body(node: Any) -> dict[str, Any] | None:
                annotations = KubernetesWorkflowAdapter._annotations(node)
                if annotations.get(ANNOTATION_INCIDENT) != incident_id:
                    return None
                if KubernetesWorkflowAdapter._unschedulable(node):
                    return None
                patched.append(True)
                return {
                    "metadata": {"annotations": {}},
                    "spec": {"unschedulable": True},
                }

            try:
                patch_node_with_retry(core, kubernetes_node, body)
            except NodePatchConflict:
                continue
            except Exception as exc:
                if isinstance(exc, KeyError) or getattr(exc, "status", None) == 404:
                    continue
                raise
            if patched and node_id not in reasserted:
                reasserted.append(node_id)
        return {"isolation_reasserted": reasserted} if reasserted else {}

    def _observe_isolation(
        self, context: WorkflowStepContext
    ) -> ObservedIsolation | WorkflowStepOutcome:
        """Read each target node and require it to be isolated right now.

        ``isolation_verified_nodes`` on the request and ``MARK_UNSCHEDULABLE``
        in ``completed_operations`` are memories of an earlier step; a node
        can be uncordoned, re-registered or renamed between that step and
        the provider mutation. Unresolvable identity fails closed.
        """
        if self.kubernetes_adapter is None:
            return WorkflowStepOutcome.failed(
                "HyperPod mutation requires the Kubernetes adapter to observe "
                "node isolation",
                details={"safety_rejection": True},
            )
        core = self.kubernetes_adapter.core
        verified: set[str] = set()
        nodes: dict[str, dict[str, Any]] = {}
        for node_id in context.step.node_ids:
            candidates = self._kubernetes_node_candidates(node_id)
            node = None
            kubernetes_node = None
            for candidate in candidates:
                try:
                    node = core.read_node(candidate)
                except Exception as exc:
                    if isinstance(exc, KeyError) or (
                        getattr(exc, "status", None) == 404
                    ):
                        continue
                    raise
                kubernetes_node = candidate
                break
            if node is None or kubernetes_node is None:
                return WorkflowStepOutcome.failed(
                    f"cannot resolve Kubernetes node for {node_id}; isolation "
                    "cannot be observed",
                    details={
                        "safety_rejection": True,
                        "node_id": node_id,
                        "candidates": candidates,
                    },
                )
            problems = self._isolation_problems(node, context)
            if problems:
                return WorkflowStepOutcome.failed(
                    f"node {node_id} is not isolated: " + "; ".join(problems),
                    details={
                        "safety_rejection": True,
                        "node_id": node_id,
                        "kubernetes_node": kubernetes_node,
                        "isolation_problems": problems,
                    },
                )
            annotations = KubernetesWorkflowAdapter._annotations(node)
            nodes[node_id] = {
                "kubernetes_node": kubernetes_node,
                "unschedulable": True,
                "incident": annotations.get(ANNOTATION_INCIDENT),
                "fencing_token": annotations.get(ANNOTATION_FENCING),
                "resource_version": KubernetesWorkflowAdapter._resource_version(node),
            }
            verified.update({node_id, kubernetes_node})
        return ObservedIsolation(sorted(verified), nodes)

    def _allocate_spare_or_wait(
        self,
        context: WorkflowStepContext,
        *,
        local_only: bool,
    ) -> SpareAllocation | WorkflowStepOutcome:
        if self.spare_coordinator is None:
            raise RuntimeError("spare coordinator is unavailable")
        try:
            return self.spare_coordinator.allocate(
                cluster_id=context.incident.cluster_id,
                incident_id=context.incident.incident_id,
                fault_node_ids=context.step.node_ids,
                local_only=local_only,
                gpu_client_checker=(
                    self._spare_gpu_client_reasons(context)
                    if self.node_action_adapter is not None
                    else None
                ),
            )
        except SpareHealthPending as exc:
            return WorkflowStepOutcome.waiting(
                details={
                    "spare_health_pending": True,
                    "reason": str(exc),
                },
            )

    def _run_spare_failover(
        self,
        context: WorkflowStepContext,
        *,
        local_only: bool,
        state: dict[str, Any] | None = None,
    ) -> WorkflowStepOutcome:
        coordinator = self.spare_coordinator
        if coordinator is None:
            raise RuntimeError("spare coordinator is unavailable")
        pending = dict(state or {})
        activated = list(pending.get("activated_spare_nodes") or [])
        if not activated:
            allocation = self._allocate_spare_or_wait(context, local_only=local_only)
            if isinstance(allocation, WorkflowStepOutcome):
                return WorkflowStepOutcome.waiting(
                    details={
                        **pending,
                        **(allocation.details or {}),
                        "spare_failover_pending": True,
                    },
                )
            if not allocation.sufficient:
                return WorkflowStepOutcome.failed(
                    allocation.reason or "insufficient healthy HyperPod spares"
                )
            if not allocation.applicable or not allocation.selected_node_ids:
                return WorkflowStepOutcome.failed(
                    "warm-spare replacement is required; provider node "
                    "replacement API fallback is disabled"
                )
            activated = list(allocation.selected_node_ids)
        provider_baselines = dict(pending.get("provider_baselines") or {})
        if not provider_baselines and not local_only:
            provider_baselines = self._provider_baselines(context)
        try:
            revoked_agents = (
                list(pending.get("revoked_agents") or [])
                if "revoked_agents" in pending
                else self._revoke_agents(context)
            )
            retry_details = {
                **pending,
                "spare_failover_pending": True,
                "activated_spare_nodes": activated,
                "revoked_agents": revoked_agents,
                "provider_baselines": provider_baselines,
            }
            details = self._spare_failover_details(
                context,
                activated,
                revoked_agents=revoked_agents,
                provider_baselines=provider_baselines,
            )
            notification_id = self._notify_warm_spare_replaced(
                context,
                context.idempotency_key,
                details,
            )
            if notification_id:
                details["notification_id"] = notification_id
            return WorkflowStepOutcome.succeeded(
                operation_id=context.idempotency_key,
                details=details,
            )
        except SpareHealthPending as exc:
            return WorkflowStepOutcome.waiting(
                details={
                    **retry_details,
                    "spare_health_pending": True,
                    "reason": str(exc),
                },
            )
        except Exception:
            coordinator.release(activated, context.incident.incident_id)
            raise

    def execute(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        previous = next(
            (
                item
                for item in context.workflow.step_executions
                if item.step_index == context.step_index
                and item.operation is context.step.operation
                and item.status is WorkflowStepStatus.WAITING
            ),
            None,
        )
        if previous and (
            previous.details.get("spare_failover_pending")
            or previous.details.get("spare_health_pending")
        ):
            return self._run_spare_failover(
                context,
                local_only=(
                    context.step.parameters.get("replacement_strategy")
                    == "HEALTHY_WARM_SPARE_ONLY"
                ),
                state=previous.details,
            )
        if previous and previous.adapter_operation_id:
            base_details = {
                **previous.details,
                **self._reassert_isolation(context, previous.details),
            }
            if (
                previous.adapter_operation_id
                in context.request.confirmed_adapter_operation_ids
            ):
                details = {
                    **base_details,
                    "externally_confirmed": True,
                    "confirmation_source": "explicit-operation-id",
                }
                notification_id = self._notify_node_restarted(
                    context,
                    previous.adapter_operation_id,
                    details,
                )
                if notification_id:
                    details["notification_id"] = notification_id
                return WorkflowStepOutcome.succeeded(
                    operation_id=previous.adapter_operation_id,
                    details=details,
                )
            automatic_confirmation = self._automatic_confirmation(context, base_details)
            if automatic_confirmation is not None:
                details = {
                    **base_details,
                    **automatic_confirmation,
                }
                notification_id = self._notify_node_restarted(
                    context,
                    previous.adapter_operation_id,
                    details,
                )
                if notification_id:
                    details["notification_id"] = notification_id
                return WorkflowStepOutcome.succeeded(
                    operation_id=previous.adapter_operation_id,
                    details=details,
                )
            return WorkflowStepOutcome.waiting(
                operation_id=previous.adapter_operation_id,
                details=base_details,
            )
        if not context.request.confirm_cluster_name:
            return WorkflowStepOutcome.failed(
                "confirm_cluster_name is required for HyperPod mutation"
            )
        # What the scheduler shows now, not what the workflow remembers: an
        # unschedulable node carrying this incident's quarantine taint and
        # annotation. Anything less refuses the provider mutation.
        observed = self._observe_isolation(context)
        if isinstance(observed, WorkflowStepOutcome):
            return observed
        isolation = list(observed.verified)
        warm_spare_only = (
            context.step.operation is WorkflowOperation.REPLACE_NODE
            and context.step.parameters.get("replacement_strategy")
            == "HEALTHY_WARM_SPARE_ONLY"
        )
        preflight = None
        preflight_error = None
        preflight_configuration_error = None
        try:
            preflight = self.dispatcher.preflight(
                context.workflow,
                context.step_index,
                isolation_verified_nodes=isolation,
                # Warm-spare replacement never submits a provider
                # replacement, but it still must read NodeRecovery so an
                # Automatic provider controller cannot race the spare
                # coordinator.
                require_execution_enabled=not warm_spare_only,
            )
        except (HyperPodAdapterError, KeyError, ValueError) as exc:
            preflight_error = exc
        except Exception as exc:
            # preflight calls DescribeCluster, so a pod with no IRSA
            # annotation fails here with NoCredentialsError -- outside
            # the tuple above, so it used to escape to the executor's
            # catch-all and be reported as executor_internal_error, i.e.
            # a deployment gap dressed up as an adapter bug. Worse, the
            # escape skipped every guard below, including the one that
            # refuses to fall back to the provider replacement API, so
            # the run proved nothing about those guards either.
            reason = aws_configuration_error(exc)
            if reason is None:
                raise
            preflight_configuration_error = reason
            preflight_error = exc
        if (
            warm_spare_only
            and preflight is not None
            and getattr(preflight, "node_recovery", None) == "Automatic"
        ):
            return WorkflowStepOutcome.failed(
                "healthy warm-spare replacement requires HyperPod "
                "NodeRecovery=None; provider replacement is disabled"
            )
        if preflight is not None and not preflight.safe_to_submit:
            return WorkflowStepOutcome.failed(
                "HyperPod preflight failed: " + "; ".join(preflight.gate_failures)
            )
        if warm_spare_only and self.spare_coordinator is None:
            return WorkflowStepOutcome.failed(
                "healthy warm-spare replacement is required but the "
                "spare coordinator is disabled"
            )
        if (
            context.step.operation is WorkflowOperation.REPLACE_NODE
            and self.spare_coordinator is not None
            and (
                warm_spare_only
                or getattr(preflight, "node_recovery", None) != "Automatic"
            )
        ):
            return self._run_spare_failover(context, local_only=warm_spare_only)
        agent_baselines = self._agent_baselines(context)
        provider_baselines = (
            self._provider_baselines(context)
            if (
                context.step.operation is WorkflowOperation.REPLACE_NODE
                and not warm_spare_only
            )
            else {}
        )
        if (
            context.step.operation is WorkflowOperation.REPLACE_NODE
            and self.spare_coordinator is not None
        ):
            return WorkflowStepOutcome.failed(
                "warm-spare replacement is required; provider node "
                "replacement API fallback is disabled"
            )
        if preflight_configuration_error is not None:
            return WorkflowStepOutcome.failed(
                "HyperPod preflight cannot run: " + preflight_configuration_error,
                details={
                    "configuration_error": True,
                    "exception_type": type(preflight_error).__name__,
                    "remediation": (
                        "kubectl -n gpu-fault-system annotate sa "
                        "gpu-fault-cluster-executor --overwrite "
                        "eks.amazonaws.com/role-arn=<role>, then "
                        "rollout restart the executor: the projected "
                        "token volume is only injected at pod creation"
                    ),
                },
            )
        if preflight_error is not None:
            return WorkflowStepOutcome.failed(
                "HyperPod preflight failed: " + str(preflight_error)
            )
        try:
            revoked_agents = self._revoke_agents(context)
        except (KeyError, ValueError) as exc:
            return WorkflowStepOutcome.failed(
                f"failed to revoke node agent before HyperPod mutation: {exc}"
            )
        # The dispatcher owns the provider submission key; it is distinct
        # from the remote command idempotency key carried by this context.
        result = self.dispatcher.submit(
            context.workflow,
            context.step_index,
            isolation_verified_nodes=isolation,
            confirm_cluster_name=(context.request.confirm_cluster_name),
            expected_fencing_token=(context.request.expected_fencing_token),
        )
        if result.failures:
            return WorkflowStepOutcome.failed(
                "HyperPod rejected nodes: "
                + ", ".join(
                    item.node_logical_id or item.node_id or "unknown"
                    for item in result.failures
                )
            )
        return WorkflowStepOutcome.waiting(
            operation_id=result.operation_id,
            details={
                "action": result.action.value,
                "submission_idempotency_key": result.idempotency_key,
                "submitted_nodes": (result.successful_node_logical_ids),
                "revoked_agents": revoked_agents,
                "agent_baselines": agent_baselines,
                "provider_baselines": provider_baselines,
                "activated_spare_nodes": [],
                "requires_external_confirmation": True,
                "observed_isolation": observed.nodes,
            },
        )
