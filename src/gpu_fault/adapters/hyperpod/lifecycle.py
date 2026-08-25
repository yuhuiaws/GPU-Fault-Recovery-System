from __future__ import annotations

from typing import Any

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
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.operation_registry import (
    OperationAdapter,
    operations_for_adapter,
)
from gpu_fault.notifications import (
    RestartGuardEmailBuilder,
    WarmSpareReplacementEmailBuilder,
)


from gpu_fault.adapters.kubernetes.adapter import KubernetesWorkflowAdapter
from gpu_fault.adapters.node_action.adapter import NodeActionWorkflowAdapter
from gpu_fault.adapters.hyperpod.confirmation import HyperPodConfirmationMixin
from gpu_fault.adapters.hyperpod.notifications import HyperPodNotificationMixin


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
            if (
                previous.adapter_operation_id
                in context.request.confirmed_adapter_operation_ids
            ):
                details = {
                    **previous.details,
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
            automatic_confirmation = self._automatic_confirmation(
                context, previous.details
            )
            if automatic_confirmation is not None:
                details = {
                    **previous.details,
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
                details=previous.details,
            )
        if not context.request.confirm_cluster_name:
            return WorkflowStepOutcome.failed(
                "confirm_cluster_name is required for HyperPod mutation"
            )
        isolation = sorted(
            set(context.request.isolation_verified_nodes).union(
                context.step.node_ids
                if WorkflowOperation.MARK_UNSCHEDULABLE
                in context.workflow.completed_operations
                else []
            )
        )
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
        try:
            result = self.dispatcher.submit(
                context.workflow,
                context.step_index,
                isolation_verified_nodes=isolation,
                confirm_cluster_name=(context.request.confirm_cluster_name),
                expected_fencing_token=(context.request.expected_fencing_token),
            )
        except Exception:
            raise
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
                "submitted_nodes": (result.successful_node_logical_ids),
                "revoked_agents": revoked_agents,
                "agent_baselines": agent_baselines,
                "provider_baselines": provider_baselines,
                "activated_spare_nodes": [],
                "requires_external_confirmation": True,
            },
        )
