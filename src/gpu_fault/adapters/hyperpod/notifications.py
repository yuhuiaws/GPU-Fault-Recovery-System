from __future__ import annotations

import re
from typing import Any, Callable

from gpu_fault.execution import (
    WorkflowStepContext,
)
from gpu_fault.fleet import (
    AgentLifecycleState,
    AgentTransitionRequest,
)
from gpu_fault.models import (
    WorkflowOperation,
)


class HyperPodNotificationMixin:
    # Attributes supplied by the composed concrete implementation.
    alert_sender: Callable[..., Any]
    notification_sink: Any
    registry: Any
    restart_email_builder: Any
    warm_spare_email_builder: Any

    @staticmethod
    def _source_boot_id(
        context: WorkflowStepContext,
    ) -> str | None:
        if context.incident.source_boot_id:
            return context.incident.source_boot_id
        match = re.search(
            r"kernel-log-kmsg-"
            r"([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-"
            r"[0-9a-f]{12})-\d+-xid-\d+$",
            context.incident.event_id,
        )
        return match.group(1) if match is not None else None

    def _notify_node_restarted(
        self,
        context: WorkflowStepContext,
        operation_id: str,
        details: dict[str, Any],
    ) -> str | None:
        if (
            context.step.operation is WorkflowOperation.REPLACE_NODE
            and details.get("action") == "SPARE_FAILOVER"
        ):
            return self._notify_warm_spare_replaced(context, operation_id, details)
        if (
            self.notification_sink is None
            or context.step.operation is not WorkflowOperation.RESTART_NODE
        ):
            return None
        baselines = details.get("agent_baselines") or {}
        source_boot_ids = {
            item.get("boot_id") for item in baselines.values() if item.get("boot_id")
        }
        source_boot_id = (
            next(iter(source_boot_ids))
            if len(source_boot_ids) == 1
            else self._source_boot_id(context)
        )
        notification = self.restart_email_builder.build_node_restarted(
            cluster_id=context.incident.cluster_id,
            incident_id=context.incident.incident_id,
            workflow_id=context.workflow.request_id,
            event_id=context.incident.event_id,
            event_type=context.incident.event_type,
            event_source=context.incident.event_source,
            xid=self._incident_xid(context),
            policy_source=(context.incident.policy_source or "UNKNOWN"),
            official_action=context.incident.official_action,
            effective_action=(
                context.incident.effective_action.value
                if context.incident.effective_action
                else None
            ),
            reasons=context.incident.reasons,
            operation_id=operation_id,
            node_ids=context.step.node_ids,
            source_boot_id=source_boot_id,
            agent_baselines=baselines,
            agent_observations=details.get("agent_observations", []),
            provider_observations=details.get("provider_observations", []),
            confirmation_source=str(
                details.get(
                    "confirmation_source",
                    "explicit-operation-id",
                )
            ),
        )
        notification = self.notification_sink.save_notification_if_absent(notification)
        if self.alert_sender is not None:
            self.alert_sender(notification.notification_id)
        return notification.notification_id

    def _notify_warm_spare_replaced(
        self,
        context: WorkflowStepContext,
        operation_id: str,
        details: dict[str, Any],
    ) -> str | None:
        if self.notification_sink is None:
            return None
        spare_nodes = list(details.get("activated_spare_nodes") or [])
        rebindings = {
            str(key): str(value)
            for key, value in (details.get("node_rebindings") or {}).items()
        }
        notification = self.warm_spare_email_builder.build(
            cluster_id=context.incident.cluster_id,
            incident_id=context.incident.incident_id,
            workflow_id=context.workflow.request_id,
            event_id=context.incident.event_id,
            policy_source=context.incident.policy_source,
            official_action=context.incident.official_action,
            effective_action=(
                context.incident.effective_action.value
                if context.incident.effective_action
                else None
            ),
            reasons=context.incident.reasons,
            operation_id=operation_id,
            fault_node_ids=context.step.node_ids,
            spare_node_ids=spare_nodes,
            node_rebindings=rebindings,
            confirmation_source=str(
                details.get(
                    "confirmation_source",
                    "healthy-running-warm-spare",
                )
            ),
            provider_mutation_submitted=bool(
                details.get("provider_mutation_submitted", False)
            ),
        )
        notification = self.notification_sink.save_notification_if_absent(notification)
        if self.alert_sender is not None:
            self.alert_sender(notification.notification_id)
        return notification.notification_id

    @staticmethod
    def _incident_xid(
        context: WorkflowStepContext,
    ) -> int | None:
        match = re.search(r"-xid-(\d+)$", context.incident.event_id)
        return int(match.group(1)) if match is not None else None

    def _revoke_agents(self, context: WorkflowStepContext) -> list[str]:
        if self.registry is None:
            return []
        transition_id = f"{context.idempotency_key}/hyperpod-lifecycle"
        revoked = []
        for node_id in context.step.node_ids:
            try:
                record = self.registry.store.get_agent(
                    context.incident.cluster_id, node_id
                )
            except KeyError:
                continue
            if record.lifecycle_state is AgentLifecycleState.REVOKED:
                revoked.append(node_id)
                continue
            request = AgentTransitionRequest(
                expected_generation=record.generation,
                transition_id=transition_id,
                reason=(f"HyperPod {context.step.operation.value} submission"),
            )
            self.registry.drain_agent(
                context.incident.cluster_id,
                node_id,
                request,
            )
            self.registry.revoke_agent(
                context.incident.cluster_id,
                node_id,
                request,
            )
            revoked.append(node_id)
        return revoked
