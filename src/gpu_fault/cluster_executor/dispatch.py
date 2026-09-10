"""One claimed command to one verdict.

``CommandDispatch`` is the layer between the lease lifecycle and the adapters:
it validates the command against this executor's cluster and namespaces, runs
the fleet preflight for destructive steps, picks the single adapter that
supports the step, refuses multi-node barrier steps this storeless topology
cannot coordinate, and turns whatever the adapter raised into a result the
control plane can act on. It never raises for a command's own failure; only
a command with no lease token is a defect it lets escape.

Like the lifecycle, it reads the executor's configuration and counters live
through ``self.executor``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gpu_fault.adapters.common import node_action_accepted_nodes
from gpu_fault.aws_errors import aws_configuration_error
from gpu_fault.cluster_executor.batching import execute_batched_command
from gpu_fault.cluster_executor.lease import adapter_facing_details
from gpu_fault.cluster_executor.regional_client import ClusterExecutorError
from gpu_fault.execution import WorkflowStepContext, WorkflowStepOutcome
from gpu_fault.execution.fleet_preflight import (
    command_requires_fleet_preflight,
    fleet_preflight_reason,
)
from gpu_fault.execution.transient_errors import retryable_adapter_error
from gpu_fault.models import (
    WorkflowOperation,
    WorkflowStepExecution,
    WorkflowStepStatus,
    execution_phase,
)
from gpu_fault.operation_registry import MULTI_NODE_BARRIER_OPERATIONS
from gpu_fault.regional import (
    RemoteActionCommand,
    RemoteCommandResult,
    RemoteCommandStatus,
)
from gpu_fault.transport_errors import retryable_transport_result

if TYPE_CHECKING:
    from gpu_fault.cluster_executor.executor import ClusterActionExecutor

# Deliberately the pre-split module's name and not ``__name__``: the log format
# carries ``%(name)s`` and operators filter on ``gpu_fault.cluster_executor``, so
# every layer of the package logs under the one name it always had.
LOGGER = logging.getLogger("gpu_fault.cluster_executor")


class CommandDispatch:
    """Validate, preflight, run and classify one claimed command."""

    def __init__(self, executor: ClusterActionExecutor) -> None:
        self.executor = executor

    def execute(self, command: RemoteActionCommand) -> RemoteCommandResult:
        """One claimed command to its verdict; never raises for a command's
        own failure.

        Validation, the fleet preflight, adapter selection and the barrier
        check come first; then the single matching adapter runs the step, and
        whatever it raised is classified into a result by
        ``_classify_failure`` -- retryable (WAITING), rejected, misconfigured
        or an executor defect (FAILED). A compound command (``batched_steps``)
        leaves here after the barrier check: ``execute_batched_command`` runs
        its covered steps one at a time through these same pieces.
        """

        lease_token = command.lease_token
        if not lease_token:
            raise ClusterExecutorError("claimed command has no lease token")
        try:
            self._validate(command)
            if not command.batched_steps:
                # A compound command's head is preflighted inside the batched
                # run, against the workflow as it stands when its turn comes,
                # like every step it covers; doing it here too would cost the
                # control-plane round trip the compound command exists to save.
                hold = self._fleet_preflight_hold(command, lease_token)
                if hold is not None:
                    return hold
            matches = [
                adapter
                for adapter in self.executor.adapters
                if adapter.supports(command.step)
            ]
            if len(matches) != 1:
                raise ClusterExecutorError(
                    "remote command requires exactly one local adapter; "
                    f"found {len(matches)}"
                )
            barrier_hold = self._barrier_hold(command, matches[0], lease_token)
            if barrier_hold is not None:
                return barrier_hold
            if command.batched_steps:
                return execute_batched_command(
                    self.executor, command, matches[0], lease_token
                )
            workflow = command.workflow
            if command.result_details:
                previous = WorkflowStepExecution(
                    step_index=command.step_index,
                    operation=command.step.operation,
                    status=WorkflowStepStatus.WAITING,
                    phase=execution_phase(workflow),
                    adapter_operation_id=(f"remote/{command.command_id}"),
                    details=adapter_facing_details(command.result_details),
                )
                workflow = workflow.model_copy(
                    update={
                        "step_executions": [
                            *workflow.step_executions,
                            previous,
                        ]
                    }
                )
            outcome = matches[0].execute(
                WorkflowStepContext(
                    workflow=workflow,
                    incident=command.incident,
                    step=command.step,
                    step_index=command.step_index,
                    request=self.execution_request(command),
                    idempotency_key=command.idempotency_key,
                )
            )
            return self.outcome_result(
                outcome, lease_token, operation=command.step.operation
            )
        except Exception as exc:
            return self._classify_failure(exc, command, lease_token)

    @staticmethod
    def outcome_result(
        outcome: WorkflowStepOutcome,
        lease_token: str,
        *,
        operation: WorkflowOperation,
    ) -> RemoteCommandResult:
        """The adapter's own verdict as a result; no merge, no classification.

        ``operation`` is the step's own so a compound command's later steps
        are described as themselves, not as the head.
        """

        status = {
            WorkflowStepStatus.WAITING: (RemoteCommandStatus.WAITING),
            WorkflowStepStatus.SUCCEEDED: (RemoteCommandStatus.SUCCEEDED),
            WorkflowStepStatus.FAILED: (RemoteCommandStatus.FAILED),
        }[outcome.status]
        details = dict(outcome.details or {})
        error = outcome.error
        if status is RemoteCommandStatus.FAILED and not error:
            # A FAILED outcome without a message is still the adapter's
            # verdict, not an executor defect. Left as None it failed the
            # result model's validation inside the caller's try block and was
            # caught as an executor-internal-error -- a stack trace, an
            # unexpected-failure count and an alert for a refusal the adapter
            # merely forgot to describe.
            error = (
                f"{operation.value} adapter reported FAILED without an error message"
            )
            details["error_message_missing"] = True
        return RemoteCommandResult(
            lease_token=lease_token,
            status=status,
            details=details,
            error=error,
        )

    def _fleet_preflight_hold(
        self,
        command: RemoteActionCommand,
        lease_token: str,
    ) -> RemoteCommandResult | None:
        """WAITING while a fleet rollout fences this destructive step, else None.

        Only for operations that require the preflight, and never once an agent
        has accepted the step's mutation (``_node_action_already_started``):
        the work is then in the agent's ledger and cannot be called back.
        """

        if (
            self.executor.fleet_registry is not None
            and command_requires_fleet_preflight(command.step.operation)
            and not self._node_action_already_started(command)
        ):
            steps = (
                command.workflow.safety_steps
                if command.workflow.executes_safety_steps
                else command.workflow.official_steps
            )
            preflight_error = fleet_preflight_reason(
                self.executor.fleet_registry,
                command.workflow,
                command.incident,
                steps,
            )
            if preflight_error is not None:
                self.executor.increment("fleet_fence_holds_total")
                LOGGER.warning(
                    "remote command held before destructive action: "
                    "command=%s workflow=%s operation=%s reason=%s",
                    command.command_id,
                    command.workflow_request_id,
                    command.step.operation.value,
                    preflight_error,
                )
                return RemoteCommandResult(
                    lease_token=lease_token,
                    status=RemoteCommandStatus.WAITING,
                    details=self._hold_details(
                        command,
                        {
                            "fleet_preflight_blocked": True,
                            "reason": preflight_error,
                        },
                    ),
                )
        return None

    def _classify_failure(
        self,
        exc: Exception,
        command: RemoteActionCommand,
        lease_token: str,
    ) -> RemoteCommandResult:
        """The result for an exception that escaped the adapter or the checks.

        Order matters and mirrors the two ``except`` clauses this replaced: a
        ``ClusterExecutorError`` is first a transient control-plane failure,
        then a transport/adapter retry, then a deliberate rejection; anything
        else is a retry, an AWS configuration gap, or an executor defect.

        ``command`` is the step being judged: for a step inside a compound
        command the caller passes its per-step view (own ``step``,
        ``step_index``, ``result_details``), so the log lines, the hold merge
        and the accepted-node check all describe that step and not the head.
        """

        if isinstance(exc, ClusterExecutorError):
            retryable = self._retryable_control_plane_result(exc, command, lease_token)
            if retryable is not None:
                return retryable
            # A control-plane request that never got an answer now arrives here
            # as a ClusterExecutorError with no status code instead of a raw
            # URLError, so the transport classification has to be consulted on
            # this branch too. Without it, wrapping the transport error would
            # have turned every proxy timeout inside an adapter from a WAITING
            # hold into a FAILED step -- a healthy GPU declared unrecoverable
            # because a read timed out.
            retryable = self._retryable_result(exc, command, lease_token)
            if retryable is not None:
                return retryable
            # Rejections the executor raises on purpose: cluster
            # mismatch, stale fencing token, no or ambiguous adapter.
            # These are legitimate FAILED results, not executor bugs.
            LOGGER.warning(
                "regional cluster executor rejected command: "
                "command=%s cluster=%s operation=%s owner=%s nodes=%s: %s",
                command.command_id,
                command.cluster_id,
                command.step.operation.value,
                command.step.execution_owner,
                ",".join(command.step.node_ids),
                exc,
            )
            return RemoteCommandResult(
                lease_token=lease_token,
                status=RemoteCommandStatus.FAILED,
                status_source="executor-rejected",
                error=f"{type(exc).__name__}: {exc}",
            )
        retryable = self._retryable_result(exc, command, lease_token)
        if retryable is not None:
            return retryable
        configuration_reason = aws_configuration_error(exc)
        if configuration_reason is not None:
            # A missing IRSA annotation, an unassumable role or a
            # denied API call is a deployment gap, not a defect: no
            # stack trace, no internal-error count (which alerting
            # watches), and a reason that names the knob. Counting
            # these as executor bugs is what once made "the
            # ServiceAccount has no role-arn" look like an adapter
            # crash for the operator reading the step details.
            LOGGER.error(
                "regional cluster executor is misconfigured for "
                "AWS: command=%s cluster=%s operation=%s nodes=%s: "
                "%s (%s)",
                command.command_id,
                command.cluster_id,
                command.step.operation.value,
                ",".join(command.step.node_ids),
                configuration_reason,
                type(exc).__name__,
            )
            return RemoteCommandResult(
                lease_token=lease_token,
                status=RemoteCommandStatus.FAILED,
                status_source="executor-configuration-error",
                error=configuration_reason,
                details={
                    "configuration_error": True,
                    "executor_id": self.executor.executor_id,
                    "exception_type": type(exc).__name__,
                },
            )
        # Anything else is an executor-side defect (a missing
        # attribute, a bad adapter wiring, an unhandled provider
        # error). Reporting it as a bare FAILED string hides the
        # difference between "the action was refused" and "the
        # executor is broken", so record a stack trace, tag the
        # result, and count it for alerting.
        self.executor.increment("unexpected_failures")
        LOGGER.exception(
            "regional cluster executor raised while executing: "
            "command=%s cluster=%s operation=%s owner=%s nodes=%s",
            command.command_id,
            command.cluster_id,
            command.step.operation.value,
            command.step.execution_owner,
            ",".join(command.step.node_ids),
            # Explicit: the batched run classifies a step's exception after
            # its ``except`` block has closed, where there is no "current"
            # exception for LOGGER.exception to pick up.
            exc_info=exc,
        )
        return RemoteCommandResult(
            lease_token=lease_token,
            status=RemoteCommandStatus.FAILED,
            status_source="executor-internal-error",
            error=f"{type(exc).__name__}: {exc}",
            details={
                "executor_internal_error": True,
                "executor_id": self.executor.executor_id,
                "exception_type": type(exc).__name__,
            },
        )

    def _retryable_result(
        self,
        exc: BaseException,
        command: RemoteActionCommand,
        lease_token: str,
    ) -> RemoteCommandResult | None:
        """WAITING for an exception that says nothing about the step, else None.

        Transport failures keep their established shape. An adapter error
        ARCH-B1 classifies as retryable (a Kubernetes 409/429/5xx, a urllib3
        timeout raised inside the adapter) gets the same treatment from the
        regional topology: the command stays WAITING and is re-claimed,
        bounded by the control plane's per-step waiting cap exactly like a
        transport retry (ARCH-E2E-1 finding 1).
        """

        retryable = retryable_transport_result(
            exc,
            lease_token=lease_token,
            executor_id=self.executor.executor_id,
        )
        if retryable is not None:
            # Unlike an adapter or control-plane retryable, a transport failure
            # names this executor's own connectivity (gaierror, refused
            # connection, timeout reaching the target); the claim loop counts
            # whole cycles of these and backs off (``_idle_delay``).
            self.executor.increment("retryable_transport_errors_total")
            LOGGER.warning(
                "regional cluster executor transport failed; "
                "command remains retryable: command=%s cluster=%s "
                "operation=%s nodes=%s: %s",
                command.command_id,
                command.cluster_id,
                command.step.operation.value,
                ",".join(command.step.node_ids),
                retryable.details["reason"],
            )
            return retryable.model_copy(
                update={"details": self._hold_details(command, retryable.details)}
            )
        if not retryable_adapter_error(exc):
            return None
        self.executor.increment("retryable_adapter_errors_total")
        LOGGER.warning(
            "regional cluster executor adapter raised a retryable "
            "error; command remains retryable: command=%s cluster=%s "
            "operation=%s nodes=%s: %s: %s",
            command.command_id,
            command.cluster_id,
            command.step.operation.value,
            ",".join(command.step.node_ids),
            type(exc).__name__,
            exc,
        )
        return RemoteCommandResult(
            lease_token=lease_token,
            status=RemoteCommandStatus.WAITING,
            status_source="executor-retryable-adapter-error",
            details=self._hold_details(
                command,
                {
                    "retryable_adapter_error": True,
                    "reason": "RETRYABLE_ADAPTER_ERROR",
                    "executor_id": self.executor.executor_id,
                    "exception_type": type(exc).__name__,
                    "adapter_error": str(exc)[:200],
                },
            ),
        )

    @staticmethod
    def _node_action_already_started(command: RemoteActionCommand) -> bool:
        """Whether an agent has *accepted* this command's mutation.

        Only then is re-running the destructive preflight pointless: the work
        lives in the agent's ledger, cannot be called back, and a rollout that
        starts meanwhile would flip the poll to WAITING and strand the only path
        to the outcome. It costs two control-plane round trips per poll.

        The pointer alone proves nothing. ``node_action_command_id`` is also
        stamped when the submit never left this process (a refused connection --
        the normal state of a faulty node), when the agent answered 5xx/408/429,
        and when it demanded a new envelope (COMMAND_EXPIRED,
        STALE_AGENT_GENERATION). In all of those the mutation has not begun, so
        the gate has to see the acceptance the transport records only after it
        parsed a PENDING submission out of the agent.

        Acceptance is counted per node, never per step. A multi-node step is
        folded one node at a time, so its ``result_details`` describe the node it
        is waiting on and the nodes it already finished -- the ones behind them
        have not been sent anything at all. A step-level marker therefore let a
        never-contacted node's reset run alongside a fleet rollout, which is the
        one overlap this fence exists to prevent. So the skip requires the
        accepted set to cover every node of the step
        (``adapters/common.py::node_action_accepted_nodes``), and a record
        without that list -- written before the key existed -- counts as not
        accepted.

        The cost is bounded and deliberate: while a rollout is in progress, a
        node whose action *was* accepted can be made to wait behind the fence
        together with its siblings. Its services are not left quiesced forever
        either way, because the agent's own 420s fail-safe restores them.
        """

        details = command.result_details or {}
        if not details.get("node_action_command_id"):
            return False
        node_ids = set(command.step.node_ids)
        return bool(node_ids) and node_ids <= node_action_accepted_nodes(details)

    @staticmethod
    def _hold_details(
        command: RemoteActionCommand,
        details: dict[str, Any],
    ) -> dict[str, Any]:
        """This hold's own keys over whatever the previous cycle recorded.

        ``complete_remote_command`` *replaces* ``result_details`` rather than
        merging it, so a WAITING result the executor manufactures itself
        becomes the entire record of the step. Everything an adapter needs to
        resume its own asynchronous operation lives only there:
        ``agent_baselines`` and ``spare_failover_pending`` for the HyperPod
        lifecycle steps, ``gpu_client_quiesce_attempt`` for the 60-attempt
        VERIFY_NO_GPU_CLIENTS bound. Posting only the hold's own keys made the
        next cycle read a WAITING execution with no continuation state, so a
        reboot that had already landed could never be auto-confirmed and the
        quiesce bound reset on every transient failure.

        Two rules keep this from widening anything. The hold's own keys win, so
        ``reason``/``executor_id`` always describe *this* cycle. And only
        executor-manufactured WAITING results merge: a terminal SUCCEEDED or
        FAILED must never carry mid-flight state, and an adapter's own outcome
        is its own -- it already sees the previous details as a replayed step
        execution and carries forward only what it means to keep.
        """

        return {**(command.result_details or {}), **details}

    def _barrier_hold(
        self,
        command: RemoteActionCommand,
        adapter: Any,
        lease_token: str,
    ) -> RemoteCommandResult | None:
        """Hold a multi-node barrier step this topology cannot coordinate.

        ``BarrierCoordinator`` persists barrier state through the control-plane
        store, which the regional executor does not have (REMOTE_STATE=true)
        and which the control plane exposes read-only over the API. The
        node-action adapter is therefore built with ``barriers=None`` here,
        and ``_execute_multi_node_reset`` would answer with a bare FAILED that
        reads like the reset itself failed. Refuse at the claim boundary
        instead, with a reason and a counter. Planners now emit one reset
        branch per node, so this hold is a defensive guard, not a product path.
        """

        if not (
            command.step.operation in MULTI_NODE_BARRIER_OPERATIONS
            and len(command.step.node_ids) > 1
        ):
            return None
        if getattr(adapter, "barriers", None) is not None:
            return None
        self.executor.increment("barrier_unavailable_holds_total")
        reason = (
            f"{command.step.operation.value} across {len(command.step.node_ids)} "
            "nodes needs a multi-node barrier coordinator, and this regional "
            "executor has none (barrier state lives in the control-plane "
            "store); split the step per node or run it from a topology with "
            "a barrier coordinator"
        )
        LOGGER.warning(
            "remote command held: multi-node barrier unavailable: command=%s "
            "cluster=%s operation=%s nodes=%s",
            command.command_id,
            command.cluster_id,
            command.step.operation.value,
            ",".join(command.step.node_ids),
        )
        return RemoteCommandResult(
            lease_token=lease_token,
            status=RemoteCommandStatus.WAITING,
            status_source="executor-barrier-unavailable",
            details=self._hold_details(
                command,
                {
                    "multi_node_barrier_unavailable": True,
                    "operation": command.step.operation.value,
                    "node_ids": list(command.step.node_ids),
                    "reason": reason,
                    "executor_id": self.executor.executor_id,
                },
            ),
        )

    def _retryable_control_plane_result(
        self,
        exc: ClusterExecutorError,
        command: RemoteActionCommand,
        lease_token: str,
    ) -> RemoteCommandResult | None:
        if exc.status_code not in {408, 425, 429} and (
            exc.status_code is None or exc.status_code < 500
        ):
            return None
        LOGGER.warning(
            "regional cluster executor control-plane request failed "
            "transiently; command remains retryable: command=%s "
            "cluster=%s operation=%s nodes=%s status=%s",
            command.command_id,
            command.cluster_id,
            command.step.operation.value,
            ",".join(command.step.node_ids),
            exc.status_code,
        )
        return RemoteCommandResult(
            lease_token=lease_token,
            status=RemoteCommandStatus.WAITING,
            status_source="executor-retryable-control-plane",
            details=self._hold_details(
                command,
                {
                    "retryable_control_plane_error": True,
                    "status_code": exc.status_code,
                    "reason": str(exc),
                    "executor_id": self.executor.executor_id,
                    "exception_type": type(exc).__name__,
                },
            ),
        )

    def _validate(self, command: RemoteActionCommand) -> None:
        if command.cluster_id != self.executor.client.cluster_id:
            raise ClusterExecutorError(
                "command cluster does not match executor cluster"
            )
        if (
            command.fencing_token != command.workflow.fencing_token
            or command.incident.fencing_token != command.fencing_token
        ):
            raise ClusterExecutorError(
                "command fencing token does not match workflow/incident"
            )
        for workload_id in command.step.workload_ids:
            namespace = workload_id.split("/", 1)[0]
            if not self.executor.allowed_namespaces:
                raise ClusterExecutorError(
                    "executor has no allowed workload namespaces"
                )
            if namespace not in self.executor.allowed_namespaces:
                raise ClusterExecutorError(
                    f"workload namespace is not allowed: {namespace}"
                )

    def execution_request(self, command: RemoteActionCommand):
        from gpu_fault.models import (
            WorkflowExecutionRequest,
        )

        # confirm_cluster_name is the HyperPod adapter's second, human-
        # intent confirmation: it must come from the executor's own
        # configuration, not from the command being executed. Copying
        # command.cluster_id into it made the command confirm itself, so
        # the check could never fail and added no safety at all.
        return WorkflowExecutionRequest(
            expected_fencing_token=command.fencing_token,
            confirm_cluster_name=self.executor.confirm_cluster_name,
            restart_authorization=command.restart_authorization,
        )
