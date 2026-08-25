from __future__ import annotations

from typing import Any, Callable

from datetime import datetime

from gpu_fault.execution import (
    WorkflowStepContext,
    WorkflowStepOutcome,
)
from gpu_fault.fleet import (
    BarrierParticipantState,
    BarrierState,
)
from gpu_fault.models import (
    WorkflowOperation,
    WorkflowStepStatus,
)
from gpu_fault.node_agent import (
    NodeActionStatus,
)
from gpu_fault.operation_registry import (
    MAINTENANCE_GENERATION_OPERATIONS,
)
from gpu_fault.store import NotFoundError


class NodeActionBarrierMixin:
    # Attributes supplied by the composed concrete implementation.
    verify_max_attempts: Any

    _send_action: Callable[..., Any]
    barriers: Any
    registry: Any

    @staticmethod
    def _verify_attempt(
        context: WorkflowStepContext,
    ) -> int:
        previous = next(
            (
                item
                for item in reversed(context.workflow.step_executions)
                if item.step_index == context.step_index
                and "gpu_client_quiesce_attempt" in item.details
            ),
            None,
        )
        if previous is None:
            return 1
        raw_attempt = previous.details.get("gpu_client_quiesce_attempt", 0)
        try:
            return int(raw_attempt) + 1
        except (TypeError, ValueError):
            return 1

    def _readiness(
        self, context: WorkflowStepContext
    ) -> dict[str, int] | WorkflowStepOutcome:
        if self.registry is None:
            return {}
        pinned = self._maintenance_generations(context)
        if isinstance(pinned, WorkflowStepOutcome):
            return pinned
        if pinned is not None:
            return pinned
        report = self.registry.readiness(
            context.incident.cluster_id,
            context.step.node_ids,
        )
        if not report.ready:
            reasons = []
            for node in report.nodes:
                reasons.extend(f"{node.node_id}: {reason}" for reason in node.reasons)
            return WorkflowStepOutcome.failed(
                "fleet consistency gate failed: " + "; ".join(reasons)
            )
        return {node.node_id: node.generation or 0 for node in report.nodes}

    def _maintenance_generations(
        self, context: WorkflowStepContext
    ) -> dict[str, int] | WorkflowStepOutcome | None:
        if context.step.operation not in MAINTENANCE_GENERATION_OPERATIONS:
            return None
        if context.step.parameters.get("spare_health_check"):
            # A healthy warm-spare candidate was never quiesced by this
            # workflow. It must use the live fleet endpoint rather than
            # inherit maintenance generations from the faulted node's
            # earlier QUIESCE_GPU_SERVICES execution.
            return None
        if (
            context.step.operation is WorkflowOperation.RESTORE_GPU_SERVICES
            and context.step.parameters.get("preemption_quiesce_handoff_after_reboot")
        ):
            # Reboot changes the agent generation. Address the fresh
            # agent and use RESTORE only to clear the predecessor's
            # durable quiesce state/timer.
            return None
        quiesce = next(
            (
                item
                for item in reversed(context.workflow.step_executions)
                if item.step_index < context.step_index
                and item.operation is WorkflowOperation.QUIESCE_GPU_SERVICES
                and item.status is WorkflowStepStatus.SUCCEEDED
            ),
            None,
        )
        if quiesce is None:
            return None
        raw_generations = quiesce.details.get("agent_generations")
        raw_expires_at = quiesce.details.get("maintenance_window_expires_at")
        if not isinstance(raw_generations, dict) or not isinstance(raw_expires_at, str):
            return WorkflowStepOutcome.failed(
                "quiesce maintenance evidence is incomplete"
            )
        try:
            expires_at = datetime.fromisoformat(raw_expires_at)
        except ValueError:
            return WorkflowStepOutcome.failed("quiesce maintenance expiry is invalid")
        if expires_at.tzinfo is None:
            return WorkflowStepOutcome.failed(
                "quiesce maintenance expiry must include a timezone"
            )
        now = self.registry.now()
        # Restore is idempotent and must still run after the fail-safe
        # timer: the agent reports already_restored if the timer won.
        # Hardware actions must never reuse expired quiesce evidence.
        if (
            now >= expires_at
            and context.step.operation is not WorkflowOperation.RESTORE_GPU_SERVICES
        ):
            return WorkflowStepOutcome.failed(
                f"quiesce maintenance window expired at {expires_at.isoformat()}"
            )
        generations: dict[str, int] = {}
        for node_id in context.step.node_ids:
            generation = raw_generations.get(node_id)
            if not isinstance(generation, int) or generation < 1:
                return WorkflowStepOutcome.failed(
                    f"quiesce maintenance evidence is missing "
                    f"agent generation for {node_id}"
                )
            try:
                self.registry.maintenance_endpoint(
                    context.incident.cluster_id,
                    node_id,
                    generation,
                )
            except (NotFoundError, ValueError) as exc:
                return WorkflowStepOutcome.failed(
                    f"maintenance agent fence failed for {node_id}: {exc}"
                )
            generations[node_id] = generation
        return generations

    def _execute_multi_node_reset(
        self,
        context: WorkflowStepContext,
        generations: dict[str, int],
    ) -> WorkflowStepOutcome:
        if self.registry is None or self.barriers is None:
            return WorkflowStepOutcome.failed(
                "multi-node GPU reset requires fleet registry and barrier coordinator"
            )
        raw_mapping = context.step.parameters.get("gpu_uuids_by_node")
        if not isinstance(raw_mapping, dict):
            return WorkflowStepOutcome.failed(
                "multi-node GPU reset requires parameters.gpu_uuids_by_node"
            )
        gpu_mapping = {}
        for node_id in context.step.node_ids:
            values = raw_mapping.get(node_id)
            if (
                not isinstance(values, list)
                or not values
                or not all(isinstance(value, str) for value in values)
            ):
                return WorkflowStepOutcome.failed(
                    f"missing explicit GPU UUIDs for {node_id}"
                )
            gpu_mapping[node_id] = values

        barrier_id = context.idempotency_key
        barrier = self.barriers.create(
            barrier_id=barrier_id,
            cluster_id=context.incident.cluster_id,
            workflow_request_id=context.workflow.request_id,
            incident_id=context.incident.incident_id,
            fencing_token=context.workflow.fencing_token,
            operation=context.step.operation,
            generations=generations,
        )
        if barrier.state is BarrierState.PREPARING:
            for participant in barrier.participants:
                if barrier.state is BarrierState.ABORTED:
                    break
                if participant.state is not BarrierParticipantState.PENDING:
                    continue
                result = self._send_action(
                    context,
                    participant.node_id,
                    WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
                    gpu_mapping[participant.node_id],
                    command_suffix=(f"barrier/prepare/{participant.node_id}"),
                    agent_generation=(participant.agent_generation),
                )
                if isinstance(result, WorkflowStepOutcome):
                    barrier = self.barriers.record_prepare(
                        barrier_id,
                        participant.node_id,
                        error=result.error or "prepare request failed",
                    )
                elif result.status is NodeActionStatus.FAILED:
                    barrier = self.barriers.record_prepare(
                        barrier_id,
                        participant.node_id,
                        error=result.error or "prepare failed",
                    )
                else:
                    barrier = self.barriers.record_prepare(
                        barrier_id,
                        participant.node_id,
                        details=result.details,
                    )
            if barrier.state is BarrierState.ABORTED:
                return WorkflowStepOutcome.failed(self._barrier_error(barrier))
            return WorkflowStepOutcome.waiting(
                operation_id=context.idempotency_key,
                details={
                    "barrier_id": barrier_id,
                    "barrier_state": barrier.state.value,
                    "prepared_nodes": [
                        item.node_id
                        for item in barrier.participants
                        if item.state is BarrierParticipantState.PREPARED
                    ],
                },
            )

        if barrier.state is BarrierState.PREPARED:
            barrier = self.barriers.begin_commit(barrier_id, generations)
        if barrier.state is BarrierState.ABORTED:
            return WorkflowStepOutcome.failed(self._barrier_error(barrier))
        if barrier.state is BarrierState.COMMITTED:
            return WorkflowStepOutcome.succeeded(
                operation_id=context.idempotency_key,
                details={
                    "barrier_id": barrier_id,
                    "barrier_state": barrier.state.value,
                },
            )
        if barrier.state not in {
            BarrierState.COMMITTING,
            BarrierState.FAILED,
        }:
            return WorkflowStepOutcome.failed(
                f"barrier {barrier_id} is in unexpected state {barrier.state.value}"
            )

        commit_attempt = self._barrier_commit_attempt(context)
        for participant in barrier.participants:
            if participant.state is BarrierParticipantState.COMMITTED:
                continue
            result = self._send_action(
                context,
                participant.node_id,
                context.step.operation,
                gpu_mapping[participant.node_id],
                command_suffix=(
                    f"barrier/commit/{participant.node_id}/attempt-{commit_attempt}"
                ),
                agent_generation=participant.agent_generation,
            )
            if isinstance(result, WorkflowStepOutcome):
                barrier = self.barriers.record_commit(
                    barrier_id,
                    participant.node_id,
                    error=result.error or "commit request failed",
                )
            elif result.status is NodeActionStatus.FAILED:
                barrier = self.barriers.record_commit(
                    barrier_id,
                    participant.node_id,
                    error=result.error or "commit failed",
                )
            else:
                barrier = self.barriers.record_commit(
                    barrier_id,
                    participant.node_id,
                    details=result.details,
                )
        failed = [
            item
            for item in barrier.participants
            if item.state is BarrierParticipantState.FAILED
        ]
        if (
            failed
            and commit_attempt < self.verify_max_attempts
            and all(
                item.error and "clients are still active" in item.error
                for item in failed
            )
        ):
            return WorkflowStepOutcome.waiting(
                operation_id=context.idempotency_key,
                details={
                    "barrier_id": barrier_id,
                    "barrier_state": barrier.state.value,
                    "gpu_reset_commit_attempt": commit_attempt,
                    "waiting_nodes": [item.node_id for item in failed],
                    "reasons": {item.node_id: item.error for item in failed},
                },
            )
        if barrier.state is not BarrierState.COMMITTED:
            return WorkflowStepOutcome.failed(self._barrier_error(barrier))
        return WorkflowStepOutcome.succeeded(
            operation_id=context.idempotency_key,
            details={
                "barrier_id": barrier_id,
                "barrier_state": barrier.state.value,
                "node_results": {
                    item.node_id: item.commit_details for item in barrier.participants
                },
            },
        )

    @staticmethod
    def _barrier_commit_attempt(
        context: WorkflowStepContext,
    ) -> int:
        previous = next(
            (
                item
                for item in reversed(context.workflow.step_executions)
                if item.step_index == context.step_index
                and "gpu_reset_commit_attempt" in item.details
            ),
            None,
        )
        if previous is None:
            return 1
        raw_attempt = previous.details.get("gpu_reset_commit_attempt", 0)
        try:
            return int(raw_attempt) + 1
        except (TypeError, ValueError):
            return 1

    @staticmethod
    def _barrier_error(barrier) -> str:
        errors = [
            f"{item.node_id}: {item.error}"
            for item in barrier.participants
            if item.error
        ]
        return f"barrier {barrier.barrier_id} {barrier.state.value}: " + "; ".join(
            errors
        )
