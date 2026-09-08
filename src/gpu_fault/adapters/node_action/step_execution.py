from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any

from gpu_fault.adapters.common import (
    NODE_ACTION_ACCEPTED_NODES_KEY,
    dcgm_result_is_configuration_only,
    node_action_accepted_nodes,
)
from gpu_fault.execution import WorkflowStepContext, WorkflowStepOutcome
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from gpu_fault.node_agent.protocol import NodeActionResult, NodeActionStatus
from gpu_fault.operation_registry import MULTI_NODE_BARRIER_OPERATIONS


@dataclass
class NodeActionBatchState:
    readiness: dict[str, int]
    verify_attempt: int
    node_results: dict[str, dict] = field(default_factory=dict)
    triage_failures: dict[str, str] = field(default_factory=dict)
    triage_pending: dict[str, str] = field(default_factory=dict)
    diagnostic_failures: dict[str, str] = field(default_factory=dict)
    diagnostic_waiting: dict[str, str] = field(default_factory=dict)


class NodeActionExecutionService:
    def __init__(self, adapter) -> None:
        self.adapter = adapter

    def execute(
        self,
        context: WorkflowStepContext,
    ) -> WorkflowStepOutcome:
        readiness = self._preflight(context)
        if isinstance(readiness, WorkflowStepOutcome):
            return readiness
        if (
            context.step.operation in MULTI_NODE_BARRIER_OPERATIONS
            and len(context.step.node_ids) > 1
        ):
            return self._multi_node(context, readiness)
        state = NodeActionBatchState(
            readiness=readiness,
            verify_attempt=self.adapter._verify_attempt(context),
        )
        outcome = self._dispatch(context, state)
        if outcome is not None:
            return outcome
        return self._finalize(context, state)

    def _preflight(
        self,
        context: WorkflowStepContext,
    ) -> dict[str, int] | WorkflowStepOutcome:
        if not context.step.node_ids:
            return WorkflowStepOutcome.failed(
                "node action requires an explicit node target"
            )
        if len(set(context.step.node_ids)) != len(context.step.node_ids):
            return WorkflowStepOutcome.failed("node action targets contain duplicates")
        readiness = self.adapter._readiness(context)
        if not isinstance(readiness, WorkflowStepOutcome):
            return readiness
        if context.step.operation is WorkflowOperation.COLLECT_HUNG_TRIAGE:
            return {}
        if (
            context.step.operation in MULTI_NODE_BARRIER_OPERATIONS
            and len(context.step.node_ids) > 1
            and self.adapter.barriers is not None
        ):
            try:
                self.adapter.barriers.abort(
                    context.idempotency_key,
                    readiness.error or "fleet consistency gate failed",
                )
            except KeyError:
                pass
        return readiness

    def _multi_node(
        self,
        context: WorkflowStepContext,
        readiness: dict[str, int],
    ) -> WorkflowStepOutcome:
        outcome = self.adapter._execute_multi_node_reset(
            context,
            readiness,
        )
        if not (
            context.step.operation is WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES
            and outcome.status is WorkflowStepStatus.SUCCEEDED
        ):
            return outcome
        details = dict(outcome.details)
        self.adapter._add_fabric_reset_notification(
            context,
            details.get("node_results", {}),
            details,
        )
        return WorkflowStepOutcome.succeeded(
            operation_id=outcome.adapter_operation_id,
            details=details,
        )

    def _dispatch(
        self,
        context: WorkflowStepContext,
        state: NodeActionBatchState,
    ) -> WorkflowStepOutcome | None:
        gpu_map = self._gpu_map(context)
        if isinstance(gpu_map, WorkflowStepOutcome):
            return gpu_map

        def send(node_id: str):
            return self.adapter._send_action(
                context,
                node_id,
                context.step.operation,
                gpu_map[node_id],
                command_suffix=(
                    f"{node_id}/attempt-{state.verify_attempt}"
                    if context.step.operation is WorkflowOperation.VERIFY_NO_GPU_CLIENTS
                    else node_id
                ),
                agent_generation=state.readiness.get(node_id),
            )

        dispatched = self._parallel_dispatch(context, send)
        for node_id in context.step.node_ids:
            result = dispatched.get(node_id)
            if result is None:
                result = send(node_id)
            outcome = self._fold_result(
                context,
                state,
                node_id,
                result,
            )
            if outcome is not None:
                return outcome
        return None

    @staticmethod
    def _gpu_map(
        context: WorkflowStepContext,
    ) -> dict[str, list[str]] | WorkflowStepOutcome:
        result = {}
        raw_mapping = context.step.parameters.get("gpu_uuids_by_node")
        for node_id in context.step.node_ids:
            gpu_uuids = context.step.gpu_uuids
            if raw_mapping is not None:
                values = (
                    raw_mapping.get(node_id) if isinstance(raw_mapping, dict) else None
                )
                if (
                    not isinstance(values, list)
                    or not values
                    or not all(isinstance(value, str) for value in values)
                ):
                    return WorkflowStepOutcome.failed(
                        f"missing explicit GPU UUIDs for {node_id}"
                    )
                gpu_uuids = values
            result[node_id] = gpu_uuids
        return result

    def _parallel_dispatch(self, context, send) -> dict:
        if not (
            context.step.operation
            in {
                WorkflowOperation.COLLECT_HUNG_TRIAGE,
                WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
            }
            and len(context.step.node_ids) > 1
        ):
            return {}
        workers = min(
            len(context.step.node_ids),
            self.adapter.max_parallel_node_actions,
        )
        # Worker threads start with an empty context, so the regional
        # executor's lease guard (a context variable) would not reach the
        # per-node sends. Each task runs in a copy of this thread's context.
        contexts = [contextvars.copy_context() for _ in context.step.node_ids]
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="gpu-fault-node-action",
        ) as executor:
            return dict(
                zip(
                    context.step.node_ids,
                    executor.map(
                        lambda item: item[0].run(send, item[1]),
                        zip(contexts, context.step.node_ids),
                    ),
                )
            )

    def _fold_result(
        self,
        context: WorkflowStepContext,
        state: NodeActionBatchState,
        node_id: str,
        result: NodeActionResult | WorkflowStepOutcome,
    ) -> WorkflowStepOutcome | None:
        if isinstance(result, WorkflowStepOutcome):
            return self._fold_step_outcome(
                context,
                state,
                node_id,
                result,
            )
        if result.status is NodeActionStatus.INTERRUPTED:
            return WorkflowStepOutcome.failed(
                f"node agent {node_id}: {result.error}",
                details={
                    **self._partial_progress(state),
                    "node_action_interrupted": True,
                    "manual_confirmation_required": True,
                    "operation": context.step.operation.value,
                },
            )
        if result.status is not NodeActionStatus.FAILED:
            state.node_results[node_id] = result.details
            return None
        if context.step.operation is WorkflowOperation.COLLECT_HUNG_TRIAGE:
            state.triage_failures[node_id] = result.error or "hung triage failed"
            return None
        if result.retryable:
            return WorkflowStepOutcome.waiting(
                operation_id=context.idempotency_key,
                details={
                    **self._partial_progress(state),
                    "waiting_node": node_id,
                    "reason": result.error or "retryable node action failure",
                    "retryable_node_action": True,
                },
            )
        if context.step.operation is WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE:
            state.diagnostic_failures[node_id] = (
                result.error or "diagnostic bundle failed"
            )
            return None
        if (
            context.step.operation is WorkflowOperation.VERIFY_NO_GPU_CLIENTS
            and result.error
            and "clients are still active" in result.error
            and state.verify_attempt < self.adapter.verify_max_attempts
            # Waiting for clients to exit is the faulted node's quiesce
            # contract. A warm-spare candidate is never quiesced by this
            # workflow, so somebody else's live GPU work on it is a
            # definitive "not eligible", not a transient state to outwait.
            and not context.step.parameters.get("spare_health_check")
        ):
            return WorkflowStepOutcome.waiting(
                operation_id=context.idempotency_key,
                details={
                    **self._partial_progress(state),
                    "gpu_client_quiesce_attempt": state.verify_attempt,
                    "waiting_node": node_id,
                    "reason": result.error,
                },
            )
        failure_details: dict[str, Any] = {
            "gpu_client_quiesce_attempt": state.verify_attempt
        }
        if result.details.get("node_action_retry_exhausted"):
            # The transport turned a retryable failure terminal after the
            # re-submit bound; the attempt count and last error it recorded
            # are the operator's only view of what the agent tried.
            failure_details.update(result.details)
        # Last, so this step's own accounting of what it finished cannot be
        # overwritten by a same-named key the agent happened to report.
        failure_details.update(self._partial_progress(state))
        return WorkflowStepOutcome.failed(
            f"node agent {node_id}: {result.error or 'action failed'}",
            details=failure_details,
        )

    @staticmethod
    def _partial_progress(state: NodeActionBatchState) -> dict[str, Any]:
        """What this step already did, for an outcome that exits mid-batch.

        A non-parallel multi-node step folds one node at a time, so every node
        before the one that failed, was interrupted or has to be waited on was
        really acted on: its GPUs were reset, its services quiesced, its
        diagnostic run. Operators (and the follow-up steps that have to undo
        that work) read ``result_details.node_results``, and only ``_finalize``
        used to fill it -- so a terminal FAILED from the second node reported
        the failure alone and never named the first, as if nothing had
        happened on the cluster.

        A copy, not the live dict: the outcome is the record of this instant,
        and folding continues in the WAITING/diagnostic cases.
        """

        return {
            "node_results": dict(state.node_results),
            "completed_nodes": sorted(state.node_results),
        }

    @staticmethod
    def _fold_step_outcome(
        context: WorkflowStepContext,
        state: NodeActionBatchState,
        node_id: str,
        result: WorkflowStepOutcome,
    ) -> WorkflowStepOutcome | None:
        reason = (
            result.error
            or str(result.details.get("reason") or "")
            or result.status.value
        )
        if context.step.operation is WorkflowOperation.COLLECT_HUNG_TRIAGE:
            if (
                result.status is WorkflowStepStatus.WAITING
                and result.details.get("node_action_state") == "PENDING"
            ):
                state.triage_pending[node_id] = reason
            else:
                state.triage_failures[node_id] = reason
            return None
        if context.step.operation is WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE:
            target = (
                state.diagnostic_waiting
                if result.status is WorkflowStepStatus.WAITING
                else state.diagnostic_failures
            )
            target[node_id] = reason
            return None
        if len(context.step.node_ids) > 1 and state.node_results:
            # The transport (not the agent) answered for this node -- a
            # rejection, a hold, a lost lease. The step still exits mid-batch,
            # so it owes the same accounting as the outcomes built above, or a
            # multi-node failure would have two shapes depending on which
            # layer said no. A single-node step keeps the lean shape.
            return replace(
                result,
                details={
                    **(result.details or {}),
                    **NodeActionExecutionService._partial_progress(state),
                    **NodeActionExecutionService._accepted_nodes(state, result),
                },
            )
        return result

    @staticmethod
    def _accepted_nodes(
        state: NodeActionBatchState,
        result: WorkflowStepOutcome,
    ) -> dict[str, Any]:
        """Which of this step's nodes are past the point of being fenced.

        The regional executor skips the destructive fleet preflight only when
        every node of the step is in this list, so it has to name each node that
        cannot be called back: the one this outcome is about, when its agent
        accepted the send (the transport stamps that), plus every node already
        folded into ``node_results`` -- those were sent, accepted and finished,
        and no fence can undo them.

        Nodes the batch has not reached are deliberately absent, which keeps the
        fence closed for them. The key is omitted entirely when nothing was
        accepted rather than written empty: absent means "not accepted" to every
        reader, including one holding a record from before this key existed.
        """

        accepted = node_action_accepted_nodes(result.details) | set(state.node_results)
        if not accepted:
            return {}
        return {NODE_ACTION_ACCEPTED_NODES_KEY: sorted(accepted)}

    def _finalize(
        self,
        context: WorkflowStepContext,
        state: NodeActionBatchState,
    ) -> WorkflowStepOutcome:
        details: dict[str, Any] = {"node_results": state.node_results}
        if context.step.operation is WorkflowOperation.COLLECT_HUNG_TRIAGE:
            return self._triage_outcome(context, state, details)
        if context.step.operation is WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE:
            outcome = self._diagnostic_outcome(context, state, details)
            if outcome is not None:
                return outcome
        if context.step.operation is WorkflowOperation.RUN_DCGM_DIAGNOSTIC:
            outcome = self._dcgm_outcome(context, state.node_results, details)
            if outcome is not None:
                return outcome
        self._decorate_success(context, state, details)
        return WorkflowStepOutcome.succeeded(
            operation_id=context.idempotency_key,
            details=details,
        )

    def _triage_outcome(self, context, state, details):
        previous = next(
            (
                item
                for item in reversed(context.workflow.step_executions)
                if item.step_index == context.step_index
                and item.operation is context.step.operation
            ),
            None,
        )
        raw_started_at = (
            previous.details.get("triage_started_at") if previous is not None else None
        )
        try:
            started_at = (
                datetime.fromisoformat(raw_started_at)
                if isinstance(raw_started_at, str)
                else datetime.now(timezone.utc)
            )
        except ValueError:
            started_at = datetime.now(timezone.utc)
        timeout = min(
            10,
            max(
                1,
                int(context.step.parameters.get("triage_timeout_seconds", 10)),
            ),
        )
        elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
        if state.triage_pending and elapsed < timeout:
            return WorkflowStepOutcome.waiting(
                operation_id=context.idempotency_key,
                details={
                    **details,
                    "triage_started_at": started_at.isoformat(),
                    "pending_nodes": sorted(state.triage_pending),
                    "node_pending": state.triage_pending,
                },
            )
        state.triage_failures.update(state.triage_pending)
        details.update(
            {
                "triage_started_at": started_at.isoformat(),
                "undetermined_nodes": sorted(state.triage_failures),
                "node_failures": state.triage_failures,
                "triage_completed_nodes": sorted(state.node_results),
            }
        )
        not_sampled = context.step.parameters.get("not_sampled_nodes")
        if isinstance(not_sampled, list) and not_sampled:
            details["not_sampled_nodes"] = sorted(
                {str(node_id) for node_id in not_sampled if node_id}
            )
        return WorkflowStepOutcome.succeeded(
            operation_id=context.idempotency_key,
            details=details,
        )

    @staticmethod
    def _diagnostic_outcome(context, state, details):
        details.update(
            {
                "failed_nodes": sorted(state.diagnostic_failures),
                "node_failures": state.diagnostic_failures,
                "waiting_nodes": sorted(state.diagnostic_waiting),
                "node_waiting": state.diagnostic_waiting,
            }
        )
        if state.diagnostic_waiting:
            return WorkflowStepOutcome.waiting(
                operation_id=context.idempotency_key,
                details=details,
            )
        if state.diagnostic_failures:
            return WorkflowStepOutcome.failed(
                "diagnostic bundle collection failed on nodes: "
                + ", ".join(sorted(state.diagnostic_failures)),
                details=details,
            )
        return None

    def _dcgm_outcome(self, context, node_results, details):
        configuration_only = sorted(
            node_id
            for node_id, value in node_results.items()
            if dcgm_result_is_configuration_only(value)
        )
        details.update(
            {
                "recommended_actions_by_node": {
                    node_id: value.get("recommended_actions", [])
                    for node_id, value in node_results.items()
                },
                "evidence_refs_by_node": {
                    node_id: value.get("evidence_ref")
                    for node_id, value in node_results.items()
                    if value.get("evidence_ref")
                },
                "configuration_only_nodes": configuration_only,
            }
        )
        self._send_dcgm_notification(context, node_results, details)
        failed = sorted(
            node_id
            for node_id, value in node_results.items()
            if value.get("diagnostic_outcome") in {"FAIL", "INCONCLUSIVE"}
            and not dcgm_result_is_configuration_only(value)
        )
        if not failed:
            details["control_plane_action"] = "COOLDOWN_AND_VALIDATE"
            return None
        node_failures = {
            node_id: [
                "dcgm_diagnostic_"
                + str(
                    node_results[node_id].get("diagnostic_outcome", "INCONCLUSIVE")
                ).lower(),
                *[
                    str(action["action_code"])
                    for action in node_results[node_id].get("recommended_actions", [])
                    if action.get("action_code")
                ],
            ]
            for node_id in failed
        }
        return WorkflowStepOutcome.failed(
            "DCGM diagnostic failed or was inconclusive on " + ", ".join(failed),
            details={
                **details,
                "failed_nodes": failed,
                "node_failures": node_failures,
                "validation": WorkflowOperation.RUN_DCGM_DIAGNOSTIC.value,
                "control_plane_action": "DRAIN_AND_QUARANTINE",
            },
        )

    def _send_dcgm_notification(self, context, node_results, details):
        if self.adapter.store is None:
            return
        destructive = any(
            value.get("diagnostic_outcome") in {"FAIL", "INCONCLUSIVE"}
            and not dcgm_result_is_configuration_only(value)
            for value in node_results.values()
        )
        notification = self.adapter.dcgm_email_builder.build(
            cluster_id=context.incident.cluster_id,
            incident_id=context.incident.incident_id,
            workflow_id=context.workflow.request_id,
            event_id=context.incident.event_id,
            operation_id=context.idempotency_key,
            workload_ids=context.step.workload_ids,
            node_results=node_results,
            control_plane_action=(
                "DRAIN_AND_QUARANTINE" if destructive else "COOLDOWN_AND_VALIDATE"
            ),
        )
        notification = self.adapter.store.save_notification_if_absent(notification)
        details["notification_id"] = notification.notification_id
        if self.adapter.alert_sender is not None:
            self.adapter.alert_sender(notification.notification_id)

    def _decorate_success(self, context, state, details) -> None:
        operation = context.step.operation
        if operation is WorkflowOperation.QUIESCE_GPU_SERVICES and state.readiness:
            started = (
                self.adapter.registry.now()
                if self.adapter.registry is not None
                else datetime.now(timezone.utc)
            )
            returned = [
                value.get("failsafe_seconds")
                for value in state.node_results.values()
                if isinstance(value, dict)
                and isinstance(value.get("failsafe_seconds"), int)
            ]
            window = self.adapter.maintenance_window
            if returned:
                window = min(
                    window,
                    timedelta(seconds=min(returned)),
                )
            details.update(
                {
                    "agent_generations": state.readiness,
                    "maintenance_window_started_at": started.isoformat(),
                    "maintenance_window_expires_at": (started + window).isoformat(),
                }
            )
        if (
            operation is WorkflowOperation.RESTART_FABRIC_MANAGER
            and self.adapter.store is not None
        ):
            notification = (
                self.adapter.restart_email_builder.build_fabric_manager_restarted(
                    cluster_id=context.incident.cluster_id,
                    incident_id=context.incident.incident_id,
                    workflow_id=context.workflow.request_id,
                    event_id=context.incident.event_id,
                    event_type=context.incident.event_type,
                    policy_source=context.incident.policy_source,
                    official_action=context.incident.official_action,
                    reasons=context.incident.reasons,
                    operation_id=context.idempotency_key,
                    node_results=state.node_results,
                    workload_ids=context.step.workload_ids,
                )
            )
            notification = self.adapter.store.save_notification_if_absent(notification)
            details["notification_id"] = notification.notification_id
            if self.adapter.alert_sender is not None:
                self.adapter.alert_sender(notification.notification_id)
        if operation is WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES:
            self.adapter._add_fabric_reset_notification(
                context,
                state.node_results,
                details,
            )
        if operation is WorkflowOperation.RESTORE_GPU_SERVICES:
            notification_id = self.adapter._retry_fabric_reset_notification(context)
            if notification_id is not None:
                details["fabric_reset_notification_id"] = notification_id
