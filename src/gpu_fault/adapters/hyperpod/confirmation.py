from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable

from gpu_fault.execution import (
    WorkflowStepContext,
    WorkflowStepOutcome,
)
from gpu_fault.hyperpod import (
    HyperPodAdapterError,
    HyperPodNode,
)
from gpu_fault.hyperpod_spares import SpareHealthPending
from gpu_fault.models import (
    WorkflowOperation,
    WorkflowStepStatus,
)


class HyperPodConfirmationMixin:
    # Attributes supplied by the composed concrete implementation.
    _source_boot_id: Callable[..., Any]
    dispatcher: Any
    kubernetes_adapter: Any
    node_action_adapter: Any
    post_reboot_stabilization_seconds: Any
    registry: Any
    spare_coordinator: Any

    def _agent_baselines(
        self, context: WorkflowStepContext
    ) -> dict[str, dict[str, str]]:
        if self.registry is None:
            return {}
        baselines = {}
        for node_id in context.step.node_ids:
            try:
                record = self.registry.store.get_agent(
                    context.incident.cluster_id, node_id
                )
            except KeyError:
                continue
            baselines[node_id] = {
                "boot_id": record.boot_id,
                "agent_incarnation_id": (record.agent_incarnation_id),
            }
        return baselines

    def _provider_baselines(
        self, context: WorkflowStepContext
    ) -> dict[str, dict[str, str | None]]:
        try:
            nodes = self.dispatcher.adapter.resolve_nodes(context.step.node_ids)
        except (
            AttributeError,
            HyperPodAdapterError,
            KeyError,
            ValueError,
        ):
            return {}
        baselines = {}
        for requested_id, node in zip(context.step.node_ids, nodes, strict=True):
            labels = getattr(node, "kubernetes_labels", {}) or {}
            baselines[requested_id] = {
                "node_logical_id": node.node_logical_id,
                "instance_id": node.instance_id,
                "kubernetes_node_name": (
                    labels.get("kubernetes.io/hostname")
                    or (f"hyperpod-{node.instance_id}" if node.instance_id else None)
                ),
            }
        return baselines

    def _automatic_confirmation(
        self,
        context: WorkflowStepContext,
        details: dict[str, Any],
    ) -> dict[str, Any] | None:
        if self.registry is None:
            return None
        if context.step.operation is WorkflowOperation.REPLACE_NODE:
            failover = self._waiting_spare_failover(context, details)
            if failover is not None:
                return failover
            return self._replacement_confirmation(context, details)
        if context.step.operation is not WorkflowOperation.RESTART_NODE:
            return None
        baselines = details.get("agent_baselines") or {}
        observations = []
        for node_id in context.step.node_ids:
            try:
                record = self.registry.store.get_agent(
                    context.incident.cluster_id, node_id
                )
            except KeyError:
                return None
            readiness = self.registry.readiness(context.incident.cluster_id, [node_id])
            if not readiness.ready:
                return None
            baseline = baselines.get(node_id)
            if baseline:
                if record.boot_id == baseline.get(
                    "boot_id"
                ) or record.agent_incarnation_id == baseline.get(
                    "agent_incarnation_id"
                ):
                    return None
            else:
                source_boot_id = self._source_boot_id(context)
                if (
                    source_boot_id is None
                    or record.boot_id == source_boot_id
                    or source_boot_id not in record.retired_incarnation_ids
                ):
                    return None
            observations.append(
                {
                    "node_id": node_id,
                    "boot_id": record.boot_id,
                    "agent_incarnation_id": (record.agent_incarnation_id),
                }
            )
        try:
            provider_nodes = self.dispatcher.adapter.resolve_nodes(
                context.step.node_ids
            )
        except (
            AttributeError,
            HyperPodAdapterError,
            KeyError,
            ValueError,
        ):
            return None
        if len(provider_nodes) != len(context.step.node_ids) or any(
            node.status != "Running" for node in provider_nodes
        ):
            return None
        if self.post_reboot_stabilization_seconds:
            now = self.registry.now()
            started_value = details.get("post_reboot_stabilization_started_at")
            if started_value is None:
                details["post_reboot_stabilization_started_at"] = now.isoformat()
                details["post_reboot_stabilization_seconds"] = (
                    self.post_reboot_stabilization_seconds
                )
                return None
            started_at = datetime.fromisoformat(str(started_value))
            if now - started_at < timedelta(
                seconds=self.post_reboot_stabilization_seconds
            ):
                return None
        provider_observations = []
        for index, node in enumerate(provider_nodes):
            aliases = getattr(node, "aliases", set())
            node_id = next(
                (
                    candidate
                    for candidate in context.step.node_ids
                    if candidate in aliases
                ),
                context.step.node_ids[index],
            )
            provider_observations.append(
                {
                    "node_id": node_id,
                    "node_logical_id": node.node_logical_id,
                    "instance_id": node.instance_id,
                    "status": node.status,
                }
            )
        snapshot_details: dict[str, Any] = {}
        if self.node_action_adapter is not None:
            snapshot = self._trigger_health_snapshot(
                context,
                context.step.node_ids,
                suffix="post-reboot-health-snapshot",
            )
            if snapshot.status is WorkflowStepStatus.WAITING:
                return None
            if snapshot.status is WorkflowStepStatus.SUCCEEDED:
                snapshot_details["post_reboot_health_snapshot"] = snapshot.details
            else:
                snapshot_details["post_reboot_health_snapshot_error"] = (
                    snapshot.error or snapshot.status.value
                )
        return {
            "externally_confirmed": True,
            "confirmation_source": ("hyperpod-running-and-new-agent-incarnation"),
            "agent_observations": observations,
            "provider_observations": provider_observations,
            **snapshot_details,
        }

    def _waiting_spare_failover(
        self,
        context: WorkflowStepContext,
        details: dict[str, Any],
    ) -> dict[str, Any] | None:
        activated = list(details.get("activated_spare_nodes") or [])
        if not activated and self.spare_coordinator is not None:
            allocation = self.spare_coordinator.allocate(
                cluster_id=context.incident.cluster_id,
                incident_id=context.incident.incident_id,
                fault_node_ids=context.step.node_ids,
                local_only=True,
                gpu_client_checker=(
                    self._spare_gpu_client_reasons(context)
                    if self.node_action_adapter is not None
                    else None
                ),
            )
            if (
                allocation.applicable
                and allocation.sufficient
                and allocation.selected_node_ids
            ):
                activated = list(allocation.selected_node_ids)
        if not activated:
            return None
        result = self._spare_failover_details(
            context,
            activated,
            revoked_agents=list(details.get("revoked_agents") or []),
            provider_baselines=dict(details.get("provider_baselines") or {}),
        )
        result["provider_mutation_submitted"] = False
        return result

    def _spare_gpu_client_reasons(self, context: WorkflowStepContext):
        def check(
            provider_node: HyperPodNode,
            node_name: str,
            phase: str,
        ) -> list[str]:
            aliases = {
                node_name,
                provider_node.node_logical_id,
                *provider_node.aliases,
            }
            target = next(
                (
                    value
                    for value in aliases
                    if self.node_action_adapter.knows_node(
                        context.incident.cluster_id, value
                    )
                ),
                node_name,
            )
            check_context = WorkflowStepContext(
                workflow=context.workflow,
                incident=context.incident,
                step=context.step.model_copy(
                    update={
                        "operation": (WorkflowOperation.VERIFY_NO_GPU_CLIENTS),
                        "node_ids": [target],
                        "gpu_uuids": [],
                        "parameters": {
                            "compute_clients_only": True,
                            "spare_health_check": True,
                        },
                        "execution_owner": (self.node_action_adapter.owner),
                    }
                ),
                step_index=context.step_index,
                request=context.request,
                idempotency_key=(
                    f"{context.idempotency_key}/spare-client-check/{node_name}/{phase}"
                ),
            )
            outcome = self.node_action_adapter.execute(check_context)
            details = outcome.details or {}
            if outcome.status is WorkflowStepStatus.SUCCEEDED:
                return []
            if outcome.status is WorkflowStepStatus.WAITING:
                if "gpu_client_quiesce_attempt" in details:
                    # The agent answered, and the answer is that the spare
                    # is busy. Only a missing answer is pending; a definitive
                    # one must reject the candidate so the shortage surfaces.
                    return [
                        "node agent GPU client check rejected the spare: "
                        + str(details.get("reason") or outcome.status.value)
                    ]
                raise SpareHealthPending(
                    "node agent GPU client check is pending: "
                    + str(
                        details.get("reason")
                        or outcome.adapter_operation_id
                        or outcome.status.value
                    )
                )
            return [
                "node agent GPU client check failed: "
                + (
                    outcome.error
                    or str(details.get("reason") or "")
                    or outcome.status.value
                )
            ]

        return check

    def _trigger_health_snapshot(
        self,
        context: WorkflowStepContext,
        node_ids: list[str],
        *,
        suffix: str,
    ) -> WorkflowStepOutcome:
        if self.node_action_adapter is None:
            return WorkflowStepOutcome.succeeded(details={"snapshot_triggered": False})
        snapshot_context = WorkflowStepContext(
            workflow=context.workflow,
            incident=context.incident,
            step=context.step.model_copy(
                update={
                    "operation": (WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT),
                    "node_ids": node_ids,
                    "gpu_uuids": [],
                    "parameters": {},
                    "execution_owner": (self.node_action_adapter.owner),
                }
            ),
            step_index=context.step_index,
            request=context.request,
            idempotency_key=(context.idempotency_key + f"/{suffix}"),
        )
        return self.node_action_adapter.execute(snapshot_context)

    def _spare_failover_details(
        self,
        context: WorkflowStepContext,
        spare_nodes: list[str],
        *,
        revoked_agents: list[str],
        provider_baselines: dict[str, Any],
    ) -> dict[str, Any]:
        if len(spare_nodes) != len(context.step.node_ids):
            raise ValueError("warm-spare allocation does not match fault node count")
        if self.kubernetes_adapter is not None:
            replacement_context = WorkflowStepContext(
                workflow=context.workflow,
                incident=context.incident,
                step=context.step.model_copy(update={"node_ids": spare_nodes}),
                step_index=context.step_index,
                request=context.request,
                idempotency_key=(context.idempotency_key + "/spare-isolation"),
            )
            outcome = self.kubernetes_adapter._isolate(replacement_context)
            if outcome.status is not WorkflowStepStatus.SUCCEEDED:
                raise ValueError(outcome.error or "warm-spare isolation failed")
        rebindings = {}
        for old_node, spare_node in zip(
            context.step.node_ids, spare_nodes, strict=True
        ):
            rebindings[old_node] = spare_node
            baseline = provider_baselines.get(old_node) or {}
            for alias in {
                baseline.get("instance_id"),
                baseline.get("kubernetes_node_name"),
            }:
                if alias:
                    rebindings[str(alias)] = spare_node
        if self.node_action_adapter is None:
            raise ValueError(
                "warm-spare activation requires a node agent health snapshot adapter"
            )
        snapshot_context = WorkflowStepContext(
            workflow=context.workflow,
            incident=context.incident,
            step=context.step.model_copy(
                update={
                    "operation": (WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT),
                    "node_ids": spare_nodes,
                    "gpu_uuids": [],
                    "parameters": {},
                    "execution_owner": (self.node_action_adapter.owner),
                }
            ),
            step_index=context.step_index,
            request=context.request,
            idempotency_key=(context.idempotency_key + "/active-health-snapshot"),
        )
        snapshot = self.node_action_adapter.execute(snapshot_context)
        if snapshot.status is WorkflowStepStatus.WAITING:
            raise SpareHealthPending(
                "warm-spare active health snapshot is pending: "
                + str(
                    (snapshot.details or {}).get("reason")
                    or snapshot.adapter_operation_id
                    or snapshot.status.value
                )
            )
        if snapshot.status is not WorkflowStepStatus.SUCCEEDED:
            raise ValueError(
                snapshot.error or "warm-spare active health snapshot failed"
            )
        return {
            "action": "SPARE_FAILOVER",
            "activated_spare_nodes": spare_nodes,
            "revoked_agents": revoked_agents,
            "provider_baselines": provider_baselines,
            "node_rebindings": rebindings,
            "active_health_snapshot": snapshot.details,
            "externally_confirmed": True,
            "confirmation_source": "healthy-running-warm-spare",
            "provider_mutation_submitted": False,
            "requires_external_confirmation": False,
        }

    def _replacement_confirmation(
        self,
        context: WorkflowStepContext,
        details: dict[str, Any],
    ) -> dict[str, Any] | None:
        baselines = details.get("provider_baselines") or {}
        if set(baselines) != set(context.step.node_ids):
            return None
        logical_ids = [
            str(baselines[node_id]["node_logical_id"])
            for node_id in context.step.node_ids
        ]
        try:
            provider_nodes = self.dispatcher.adapter.resolve_nodes(logical_ids)
        except (
            AttributeError,
            HyperPodAdapterError,
            KeyError,
            ValueError,
        ):
            return None
        if len(provider_nodes) != len(logical_ids) or any(
            node.status != "Running" for node in provider_nodes
        ):
            return None

        observations = []
        rebindings = {}
        for requested_id, node in zip(
            context.step.node_ids,
            provider_nodes,
            strict=True,
        ):
            baseline = baselines[requested_id]
            if not node.instance_id or node.instance_id == baseline.get("instance_id"):
                return None
            kubernetes_node_name = (
                node.kubernetes_labels.get("kubernetes.io/hostname")
                or f"hyperpod-{node.instance_id}"
            )
            agents = [
                agent
                for agent in self.registry.store.list_agents(
                    context.incident.cluster_id
                )
                if (
                    agent.node_instance_id == node.instance_id
                    or agent.node_id in node.aliases
                )
            ]
            ready_agents = [
                agent
                for agent in agents
                if self.registry.readiness(
                    context.incident.cluster_id,
                    [agent.node_id],
                ).ready
            ]
            if len(ready_agents) != 1:
                return None
            agent = ready_agents[0]
            if self.kubernetes_adapter is not None:
                replacement_context = WorkflowStepContext(
                    workflow=context.workflow,
                    incident=context.incident,
                    step=context.step.model_copy(
                        update={"node_ids": [kubernetes_node_name]}
                    ),
                    step_index=context.step_index,
                    request=context.request,
                    idempotency_key=(
                        context.idempotency_key + "/replacement-isolation"
                    ),
                )
                isolation = self.kubernetes_adapter._isolate(replacement_context)
                if isolation.status is not WorkflowStepStatus.SUCCEEDED:
                    return None
            for old_identifier in {
                requested_id,
                baseline.get("instance_id"),
                baseline.get("kubernetes_node_name"),
            }:
                if old_identifier:
                    rebindings[str(old_identifier)] = kubernetes_node_name
            observations.append(
                {
                    "node_id": kubernetes_node_name,
                    "node_logical_id": node.node_logical_id,
                    "instance_id": node.instance_id,
                    "status": node.status,
                    "boot_id": agent.boot_id,
                    "agent_incarnation_id": (agent.agent_incarnation_id),
                }
            )

        if self.post_reboot_stabilization_seconds:
            now = self.registry.now()
            started_value = details.get("post_replacement_stabilization_started_at")
            if started_value is None:
                details["post_replacement_stabilization_started_at"] = now.isoformat()
                details["post_replacement_stabilization_seconds"] = (
                    self.post_reboot_stabilization_seconds
                )
                return None
            if now - datetime.fromisoformat(str(started_value)) < timedelta(
                seconds=self.post_reboot_stabilization_seconds
            ):
                return None
        return {
            "externally_confirmed": True,
            "confirmation_source": (
                "hyperpod-logical-node-new-instance-and-ready-agent"
            ),
            "agent_observations": observations,
            "provider_observations": observations,
            "node_rebindings": rebindings,
        }
