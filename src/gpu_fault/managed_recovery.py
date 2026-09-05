from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable

from pydantic import Field

from gpu_fault.execution import (
    WorkflowStepContext,
    WorkflowStepOutcome,
)
from gpu_fault.fleet import (
    AgentLifecycleState,
    AgentTransitionRequest,
    FleetRegistry,
)
from gpu_fault.hyperpod import HyperPodLifecycleAdapter, HyperPodNode
from gpu_fault.models import (
    AdvisoryNotification,
    StrictModel,
    WorkflowOperation,
    WorkflowStepStatus,
)

WORKFLOW_DEADLINE_MARGIN_SECONDS = 60


class HyperPodNodeIdentity(StrictModel):
    cluster_name: str
    node_logical_id: str
    instance_id: str | None = None
    kubernetes_node_name: str | None = None
    status: str
    generation: int = Field(default=1, ge=1)
    aliases: list[str] = Field(default_factory=list)
    retired_aliases: list[str] = Field(default_factory=list)
    observed_at: datetime


class HyperPodIdentityRegistry:
    """Persists stable logical-node identity across replacement."""

    def __init__(self, lifecycle: HyperPodLifecycleAdapter, store) -> None:
        self.lifecycle = lifecycle
        self.store = store

    def refresh(self) -> list[HyperPodNodeIdentity]:
        now = datetime.now(timezone.utc)
        identities = []
        for node in self.lifecycle.list_nodes(enrich=True):
            try:
                previous = self.store.get_hyperpod_node_identity(
                    self.lifecycle.config.cluster_name,
                    node.node_logical_id,
                )
            except KeyError:
                previous = None
            if (
                previous is not None
                and node.instance_id is not None
                and node.instance_id in previous.retired_aliases
            ):
                identities.append(previous)
                continue
            aliases = sorted(node.aliases)
            changed = previous is not None and (
                previous.instance_id != node.instance_id
                or previous.kubernetes_node_name != self._kubernetes_name(node)
            )
            retired = list(previous.retired_aliases if previous else [])
            if changed and previous is not None:
                retired = list(
                    dict.fromkeys(
                        [
                            *retired,
                            *[
                                alias
                                for alias in previous.aliases
                                if alias
                                not in {
                                    node.node_logical_id,
                                    *aliases,
                                }
                            ],
                        ]
                    )
                )[-64:]
            identity = HyperPodNodeIdentity(
                cluster_name=self.lifecycle.config.cluster_name,
                node_logical_id=node.node_logical_id,
                instance_id=node.instance_id,
                kubernetes_node_name=self._kubernetes_name(node),
                status=node.status,
                generation=(
                    1
                    if previous is None
                    else previous.generation + (1 if changed else 0)
                ),
                aliases=aliases,
                retired_aliases=retired,
                observed_at=now,
            )
            identities.append(self.store.save_hyperpod_node_identity(identity))
        return identities

    def resolve(self, identifier: str) -> HyperPodNodeIdentity | None:
        for identity in self.store.list_hyperpod_node_identities(
            self.lifecycle.config.cluster_name
        ):
            if identifier in {
                identity.node_logical_id,
                *identity.aliases,
                *identity.retired_aliases,
            }:
                return identity
        return None

    @staticmethod
    def _kubernetes_name(node: HyperPodNode) -> str | None:
        return node.kubernetes_labels.get("kubernetes.io/hostname") or (
            f"hyperpod-{node.instance_id}" if node.instance_id else None
        )


class HyperPodManagedRecoveryObserver:
    """Confirms AWS-managed reboot/replace from provider and agent state."""

    def __init__(
        self,
        identities: HyperPodIdentityRegistry,
        store,
        *,
        registry: FleetRegistry | None,
        kubernetes_adapter=None,
        timeout: timedelta = timedelta(minutes=30),
        alert_sender: Callable[[str], object] | None = None,
    ) -> None:
        self.identities = identities
        self.store = store
        self.registry = registry
        self.kubernetes_adapter = kubernetes_adapter
        self.timeout = timeout
        self.alert_sender = alert_sender

    def _deadline(
        self,
        context: WorkflowStepContext,
        started_at: datetime,
    ) -> datetime:
        """When this observation gives up, never later than the workflow does.

        The provider window and the workflow budget are set independently, so the
        workflow can be the shorter of the two -- and then the workflow deadline
        would terminalize the step first, on the generic failure path, without
        the escalation this observer sends. Clamping means the observer is the
        one that reports giving up, whichever bound is actually binding.

        The margin is what makes the clamp effective rather than decorative: the
        workflow deadline is checked before a step is dispatched, so an observer
        that gave up exactly at that deadline would never be reached. One minute
        is twelve dispatches at the default five-second poll interval, so the
        observation gets its say even if most of them are lost.
        """

        deadline = started_at + self.timeout
        workflow_deadline: datetime | None = context.workflow.execution_deadline
        if workflow_deadline is None:
            return deadline
        return min(
            deadline,
            workflow_deadline - timedelta(seconds=WORKFLOW_DEADLINE_MARGIN_SECONDS),
        )

    def observe(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        identities = self.identities.refresh()
        by_logical = {item.node_logical_id: item for item in identities}
        previous = next(
            (
                item
                for item in context.workflow.step_executions
                if item.step_index == context.step_index
                and item.status is WorkflowStepStatus.WAITING
            ),
            None,
        )
        if previous is None:
            targets = []
            for node_id in context.step.node_ids:
                identity = self.identities.resolve(node_id)
                if identity is None:
                    return WorkflowStepOutcome.failed(
                        "cannot resolve managed HyperPod node identity: " + node_id
                    )
                replacement_already_observed = (
                    context.step.operation is WorkflowOperation.REPLACE_NODE
                    and node_id in identity.retired_aliases
                )
                retired_instance = next(
                    (
                        alias
                        for alias in reversed(identity.retired_aliases)
                        if alias.startswith("i-")
                    ),
                    None,
                )
                baseline_instance = (
                    node_id
                    if node_id.startswith("i-")
                    else (
                        retired_instance
                        if replacement_already_observed
                        else identity.instance_id
                    )
                )
                targets.append(
                    {
                        "old_node_id": node_id,
                        "node_logical_id": identity.node_logical_id,
                        "baseline_instance_id": baseline_instance,
                        "baseline_generation": (
                            max(1, identity.generation - 1)
                            if replacement_already_observed
                            else identity.generation
                        ),
                        "baseline_agent_incarnation": (
                            self._agent_incarnation(identity)
                        ),
                        "baseline_agent_node_id": (
                            self._retired_agent_node_id(identity, node_id)
                            if replacement_already_observed
                            else self._agent_node_id(identity)
                        ),
                        "replacement_already_observed": (replacement_already_observed),
                    }
                )
            return WorkflowStepOutcome.waiting(
                operation_id=self._operation_id(context),
                details={
                    "managed_recovery_started_at": (
                        datetime.now(timezone.utc).isoformat()
                    ),
                    "managed_targets": targets,
                    "managed_recovery_state": ("AWS_RECOVERY_PENDING"),
                    "mutation_submitted_by_control_plane": False,
                    "requires_external_confirmation": False,
                },
            )

        started_at = datetime.fromisoformat(
            previous.details["managed_recovery_started_at"]
        )
        deadline = self._deadline(context, started_at)
        if datetime.now(timezone.utc) > deadline:
            notification = self._alert_timeout(context)
            return WorkflowStepOutcome.failed(
                "HyperPod managed recovery timed out; notification="
                + notification.notification_id
            )

        rebindings = {}
        observations = []
        for target in previous.details.get("managed_targets", []):
            identity = by_logical.get(target["node_logical_id"])
            if identity is None or identity.status != "Running":
                return self._waiting(previous, "AWS_REPLACING")
            if context.step.operation is WorkflowOperation.REPLACE_NODE:
                changed = (
                    identity.instance_id != target["baseline_instance_id"]
                    or identity.generation > target["baseline_generation"]
                )
            else:
                current_incarnation = self._agent_incarnation(identity)
                changed = bool(
                    current_incarnation
                    and current_incarnation != target["baseline_agent_incarnation"]
                )
            if not changed or not identity.kubernetes_node_name:
                return self._waiting(previous, "AWS_RECOVERY_PENDING")
            if context.step.operation is WorkflowOperation.REPLACE_NODE:
                self._retire_old_agent(
                    context,
                    target.get("baseline_agent_node_id"),
                )
            new_agent = self._ready_agent(identity)
            if new_agent is None:
                return self._waiting(previous, "AWS_REPLACING")
            try:
                isolation = self._isolate_replacement(
                    context, identity.kubernetes_node_name
                )
            except Exception as exc:
                return WorkflowStepOutcome.failed(
                    "replacement node isolation failed: " + str(exc)
                )
            if isolation.status is WorkflowStepStatus.WAITING:
                return self._waiting(
                    previous,
                    "REPLACEMENT_ISOLATION_PENDING",
                    isolation.details,
                )
            if isolation.status is WorkflowStepStatus.FAILED:
                return WorkflowStepOutcome.failed(
                    isolation.error or "replacement node isolation failed"
                )
            for old_identifier in {
                target["old_node_id"],
                target.get("baseline_instance_id"),
                target.get("baseline_agent_node_id"),
            }:
                if old_identifier:
                    rebindings[old_identifier] = identity.kubernetes_node_name
            observations.append(
                {
                    "node_logical_id": identity.node_logical_id,
                    "instance_id": identity.instance_id,
                    "kubernetes_node_name": (identity.kubernetes_node_name),
                    "identity_generation": identity.generation,
                    "agent_node_id": new_agent.node_id,
                    "agent_incarnation_id": (new_agent.agent_incarnation_id),
                }
            )
        return WorkflowStepOutcome.succeeded(
            operation_id=previous.adapter_operation_id,
            details={
                **previous.details,
                "externally_confirmed": True,
                "managed_recovery_state": "NODE_REBOUND",
                "node_rebindings": rebindings,
                "managed_observations": observations,
            },
        )

    def _ready_agent(self, identity: HyperPodNodeIdentity):
        if self.registry is None:
            return None
        matches = [
            agent
            for agent in self.store.list_agents(identity.cluster_name)
            if (
                identity.instance_id is not None
                and agent.node_instance_id == identity.instance_id
            )
            or agent.node_id in identity.aliases
        ]
        ready = [
            agent
            for agent in matches
            if self.registry.readiness(identity.cluster_name, [agent.node_id]).ready
        ]
        return ready[0] if len(ready) == 1 else None

    def _agent_incarnation(self, identity: HyperPodNodeIdentity) -> str | None:
        matches = self._matching_agents(identity)
        return matches[0].agent_incarnation_id if len(matches) == 1 else None

    def _agent_node_id(self, identity: HyperPodNodeIdentity) -> str | None:
        matches = self._matching_agents(identity)
        return matches[0].node_id if len(matches) == 1 else None

    def _matching_agents(self, identity: HyperPodNodeIdentity) -> list:
        return [
            agent
            for agent in self.store.list_agents(identity.cluster_name)
            if (
                identity.instance_id is not None
                and agent.node_instance_id == identity.instance_id
            )
            or agent.node_id in identity.aliases
        ]

    def _retired_agent_node_id(
        self,
        identity: HyperPodNodeIdentity,
        requested_node_id: str,
    ) -> str | None:
        candidates = [
            agent
            for agent in self.store.list_agents(identity.cluster_name)
            if agent.node_instance_id != identity.instance_id
            and (
                agent.node_id == requested_node_id
                or agent.node_id in identity.retired_aliases
                or agent.node_instance_id in identity.retired_aliases
            )
        ]
        return candidates[0].node_id if len(candidates) == 1 else None

    def _retire_old_agent(
        self,
        context: WorkflowStepContext,
        old_node_id: str | None,
    ) -> None:
        if self.registry is None or old_node_id is None:
            return
        try:
            record = self.store.get_agent(context.incident.cluster_id, old_node_id)
        except KeyError:
            return
        if record.lifecycle_state is AgentLifecycleState.REVOKED:
            return
        transition = AgentTransitionRequest(
            expected_generation=record.generation,
            transition_id=(f"{context.idempotency_key}/managed-recovery"),
            reason="HyperPod managed node recovery completed",
        )
        if record.lifecycle_state is AgentLifecycleState.ACTIVE:
            record = self.registry.drain_agent(
                context.incident.cluster_id,
                old_node_id,
                transition,
            )
            transition = transition.model_copy(
                update={"expected_generation": record.generation}
            )
        self.registry.revoke_agent(context.incident.cluster_id, old_node_id, transition)

    @staticmethod
    def _waiting(
        previous, state: str, details: dict | None = None
    ) -> WorkflowStepOutcome:
        return WorkflowStepOutcome.waiting(
            operation_id=previous.adapter_operation_id,
            details={
                **previous.details,
                **(details or {}),
                "managed_recovery_state": state,
            },
        )

    def _isolate_replacement(
        self, context: WorkflowStepContext, node_id: str
    ) -> WorkflowStepOutcome:
        if self.kubernetes_adapter is None:
            return WorkflowStepOutcome.succeeded()
        replacement_context = WorkflowStepContext(
            workflow=context.workflow,
            incident=context.incident,
            step=context.step.model_copy(
                update={
                    "operation": (WorkflowOperation.MARK_UNSCHEDULABLE),
                    "execution_owner": ("gpu-fault-kubernetes-adapter"),
                    "node_ids": [node_id],
                }
            ),
            step_index=context.step_index,
            request=context.request,
            idempotency_key=(context.idempotency_key + "/replacement-isolation"),
        )
        isolate = getattr(self.kubernetes_adapter, "_isolate", None)
        return (
            isolate(replacement_context)
            if isolate is not None
            else self.kubernetes_adapter.execute(replacement_context)
        )

    def _alert_timeout(self, context: WorkflowStepContext) -> AdvisoryNotification:
        notification = self.store.save_notification_if_absent(
            AdvisoryNotification(
                deduplication_key=(
                    f"{context.incident.incident_id}/managed-recovery-timeout"
                ),
                cluster_name=context.incident.cluster_id,
                incident_id=context.incident.incident_id,
                subject=("[GPU action required] HyperPod managed recovery timed out"),
                body_text=(
                    "HyperPod did not complete managed node recovery "
                    f"within {self.timeout}."
                ),
                support_case_draft=(
                    "Open an AWS Support case for the HyperPod managed "
                    "node recovery timeout."
                ),
            )
        )
        if self.alert_sender:
            self.alert_sender(notification.notification_id)
        return notification

    @staticmethod
    def _operation_id(context: WorkflowStepContext) -> str:
        return (
            f"delegated/{context.workflow.request_id}/"
            f"{context.step_index}/{context.step.operation.value}"
        )


class RegionalHyperPodManagedRecoveryObserver:
    """Routes managed recovery observation by incident cluster."""

    def __init__(
        self,
        observers: dict[str, HyperPodManagedRecoveryObserver],
    ) -> None:
        self.observers = observers

    def observe(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        observer = self.observers.get(context.incident.cluster_id)
        if observer is None:
            return WorkflowStepOutcome.failed(
                "no managed recovery observer registered for cluster "
                + context.incident.cluster_id
            )
        return observer.observe(context)
