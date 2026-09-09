"""One claimed command's lease, from claim to posted verdict.

``CommandLeaseWatch`` is the executor's local view of one lease: renewed by the
renewer thread, read by the executing thread through the node-action lease
guard. ``CommandLifecycle`` is the layer that runs one command under that
lease -- the renewer thread, the execution cap, the abandoned-worker hold and
the bounded result report -- and hands the claim loop one status per command.
It holds no per-command state of its own: the stop event, the watch, the
renewer and the abandoned slot are created inside ``run`` for each command,
and the executor's configuration and counters are read live through
``self.executor`` so a caller that swaps ``executor.client`` after
construction is honoured.
"""

from __future__ import annotations

import logging
import random
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any, Callable

from gpu_fault.adapters.node_action.lease_guard import active_lease_guard
from gpu_fault.cluster_executor.regional_client import (
    _TRANSPORT_ERRORS,
    ClusterExecutorError,
)
from gpu_fault.models import WorkflowOperation
from gpu_fault.operation_registry import (
    DESTRUCTIVE_OPERATIONS,
    OperationAdapter,
    operations_for_adapter,
)
from gpu_fault.regional import (
    RemoteActionCommand,
    RemoteCommandResult,
    RemoteCommandStatus,
)

if TYPE_CHECKING:
    from gpu_fault.cluster_executor.executor import ClusterActionExecutor

# Deliberately the pre-split module's name and not ``__name__``: the log format
# carries ``%(name)s`` and operators filter on ``gpu_fault.cluster_executor``, so
# every layer of the package logs under the one name it always had.
LOGGER = logging.getLogger("gpu_fault.cluster_executor")


# on the cluster, so a dropped connection on the result post costs a whole
# lease of latency and then a second execution after the command is re-claimed.
# Three attempts is enough for the baseline "Remote end closed connection"
# (~1/min) and for a control-plane rollout's 503 window, and short enough that
# the lease (>= 10 s, 120 s in production) is still ours while retrying.
# Three attempts means two sleeps, so there are two delays and not three: the
# worst case with jitter is 0.75 + 1.5 = 2.25s of extra latency on the command's
# own worker thread.
_RESULT_REPORT_ATTEMPTS = 3
_RESULT_REPORT_BACKOFF_SECONDS = (0.5, 1.0)
_RETRYABLE_REPORT_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
# How long one claimed command may execute before this executor gives up on it.
# Nothing else bounds ``_execute``: the lease renewer keeps a wedged command
# LEASED for as long as the process lives, and ``claim`` excludes a leased
# command, so no sibling replica can ever take it over. Half an hour is longer
# than the slowest legitimate action (UPDATE_SOFTWARE_FIRMWARE, REPLACE_NODE)
# and far shorter than "forever".
DEFAULT_MAX_EXECUTION_SECONDS = 1800.0
# Operations whose timeout leaves real state unknown: a node action may still
# be running in the agent's ledger, and every destructive operation may have
# half-applied. Their timeout demands manual confirmation instead of closing
# the step silently.
_TIMEOUT_UNKNOWN_STATE_OPERATIONS = (
    operations_for_adapter(OperationAdapter.NODE_ACTION)
    | DESTRUCTIVE_OPERATIONS
    | {
        # Neither destructive nor node actions, but each leaves state outside
        # this process that a plain failure would misrepresent: a support case
        # may already be open with the vendor, an evidence freeze may already
        # have copied part of its bundle, and a checkpoint may still be running
        # inside the job.
        WorkflowOperation.ESCALATE_SUPPORT,
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.CHECKPOINT_WORKLOADS,
    }
)
# Two status sources: the plain one means "this executor gave up waiting for a
# step that changes nothing", the unknown one means "something outside this
# process may still be mutating and a human has to confirm what happened".
EXECUTION_TIMEOUT_STATUS_SOURCE = "executor-execution-timeout"
EXECUTION_TIMEOUT_UNKNOWN_STATUS_SOURCE = "executor-execution-timeout-outcome-unknown"
# Why an abandoned worker's later sends are refused once its verdict landed.
ABANDONED_WORKER_HOLD_REASON = "abandoned after the execution cap"
# How often the guard thread checks whether an abandoned worker came back.
_ABANDONED_WORKER_POLL_SECONDS = 0.25


def _retryable_report_failure(exc: BaseException) -> bool:
    """Whether re-posting a result could plausibly land.

    Retry only failures that carry no verdict about the result: no HTTP status
    at all (the request never got an answer), or one of the statuses that means
    "not now". A 403, 409 or 422 is the control plane's decision -- re-sending
    it would just be refused again.
    """

    if isinstance(exc, ClusterExecutorError):
        return (
            exc.status_code is None or exc.status_code in _RETRYABLE_REPORT_STATUS_CODES
        )
    return isinstance(exc, _TRANSPORT_ERRORS)


def _report_backoff_seconds(attempt: int) -> float:
    """Jittered delay before report attempt ``attempt + 1``.

    Jitter matters here because both replicas can be reporting into the same
    control-plane rollout window; an un-jittered 0.5/1 retries in lockstep.
    """

    base = _RESULT_REPORT_BACKOFF_SECONDS[
        min(attempt, len(_RESULT_REPORT_BACKOFF_SECONDS)) - 1
    ]
    return base + random.random() * (base / 2)


class CommandLeaseWatch:
    """This executor's local view of one claimed command's lease.

    Updated by the renewal thread, read by the executing thread (through the
    node-action lease guard), by ``_execute_under_lease`` before it posts and by
    ``_report_result`` between report attempts. ``hold_reason()`` is non-None
    once the executor can no longer vouch for the command: renewal failed
    ``failure_limit`` times in a row, the lease window passed without a renewal
    landing, the control plane asked for cancellation, or this executor gave the
    command up (``abandon``). The first two mean another executor may already
    own the command; all four mean nothing new should start.
    """

    def __init__(
        self,
        *,
        lease_seconds: int,
        failure_limit: int,
        clock: Callable[[], float],
    ) -> None:
        self.lease_seconds = lease_seconds
        self.failure_limit = failure_limit
        self.clock = clock
        self._lock = Lock()
        self.expires_at = clock() + lease_seconds
        self.consecutive_failures = 0
        self.lost_reason: str | None = None
        self.cancellation_reason: str | None = None

    def renewed(self, response: Any) -> str | None:
        """Record a successful renewal; return the cancellation reason if any."""

        with self._lock:
            self.consecutive_failures = 0
            self.expires_at = self.clock() + self.lease_seconds
            requested_at = getattr(response, "cancellation_requested_at", None)
            if requested_at is not None and self.cancellation_reason is None:
                self.cancellation_reason = (
                    getattr(response, "cancellation_reason", None)
                    or "cancellation requested by the control plane"
                )
                return self.cancellation_reason
        return None

    def renewal_failed(self, error: BaseException) -> bool:
        """Count one failed renewal; True when the lease is now treated as lost."""

        with self._lock:
            self.consecutive_failures += 1
            if (
                self.lost_reason is None
                and self.consecutive_failures >= self.failure_limit
            ):
                self.lost_reason = (
                    f"lease renewal failed {self.consecutive_failures} time(s) "
                    f"in a row: {type(error).__name__}: {error}"
                )
                return True
        return False

    def abandon(self, reason: str) -> None:
        with self._lock:
            if self.cancellation_reason is None:
                self.cancellation_reason = reason

    def lost(self) -> bool:
        return self.lost_reason is not None or self.clock() >= self.expires_at

    def hold_reason(self) -> str | None:
        with self._lock:
            if self.lost_reason is not None:
                return self.lost_reason
            if self.clock() >= self.expires_at:
                return (
                    f"lease expired locally after {self.lease_seconds}s "
                    "without a successful renewal"
                )
            return self.cancellation_reason


# ARCH-A4b: the regional executor has no store, so a stale warm-spare
# reservation can only be judged by its timestamp. One day mirrors the


class CommandLifecycle:
    """Run one claimed command under its lease and report its verdict once.

    Everything between ``run_once`` handing over a claimed command and one
    ``RemoteCommandStatus`` coming back: the renewer thread, the execution cap
    (``_execute_within_deadline``), the verdict for a command this executor gave
    up on, the lease hold over a thread that may still be mutating, and the
    bounded, lease-aware result report. The command itself is executed by the
    executor's dispatch layer through ``executor._execute``, on the worker
    thread this object starts.
    """

    def __init__(self, executor: ClusterActionExecutor) -> None:
        self.executor = executor

    def run(self, command: RemoteActionCommand) -> RemoteCommandStatus:
        stop = Event()
        watch = CommandLeaseWatch(
            lease_seconds=self.executor.lease_seconds,
            failure_limit=self.executor.lease_renewal_failure_limit,
            clock=self.executor.clock,
        )
        renewer = Thread(
            target=self.renew_lease,
            args=(command, stop, watch),
            name=f"lease-{command.command_id[:24]}",
            daemon=True,
        )
        renewer.start()
        # Set by _execute_within_deadline when it gives up on a thread whose
        # outcome is unknown: the finally has to decide between releasing the
        # lease and holding it over a mutation that may still be running.
        abandoned: dict[str, Any] = {}
        reported = False
        try:
            result = self._execute_within_deadline(command, watch, stop, abandoned)
            if watch.lost():
                # Another executor may hold this command by now. The agent
                # ledger keeps whatever ran; the next lease holder polls it
                # by command_id. Posting here would race that holder's result
                # under a lease this process no longer owns.
                if watch.lost_reason is None:
                    # Expired on the local clock without the renewer ever
                    # declaring it lost; count it here, once.
                    self.executor.increment("lease_lost_total")
                self.executor.increment("results_withheld_total")
                LOGGER.warning(
                    "regional cluster executor withheld a result under a lost "
                    "lease: command=%s cluster=%s operation=%s status=%s "
                    "reason=%s",
                    command.command_id,
                    command.cluster_id,
                    command.step.operation.value,
                    result.status.value,
                    watch.hold_reason(),
                )
                return RemoteCommandStatus.WAITING
            reported = self._report_result(command, result, watch)
            if not reported:
                # The action ran but its verdict never landed. WAITING keeps
                # this cycle off the fast path (the command is still open on
                # the control plane) instead of claiming that it advanced.
                return RemoteCommandStatus.WAITING
            return result.status
        finally:
            worker = abandoned.get("worker")
            if worker is not None and not reported and not watch.lost():
                # Unknown outcome, no verdict on the control plane, and a thread
                # that may still be mutating: keep the heartbeat going instead
                # of inviting a sibling replica in.
                self._hold_lease_for_abandoned_worker(command, stop, renewer, worker)
            else:
                if worker is not None:
                    # Verdict recorded or lease gone: no next node for the thread.
                    watch.abandon(ABANDONED_WORKER_HOLD_REASON)
                stop.set()
                renewer.join(timeout=2)

    def _execute_within_deadline(
        self,
        command: RemoteActionCommand,
        watch: CommandLeaseWatch,
        stop: Event,
        abandoned: dict[str, Any],
    ) -> RemoteCommandResult:
        """Run one command on its own thread, bounded by the execution cap.

        Nothing here can cancel the work: an adapter blocked in a boto3 call or
        a socket read with no timeout stays blocked, and Python offers no way to
        kill the thread. What the cap buys is that *this* executor stops
        claiming to own the command -- it stops renewing the lease, reports an
        unknown outcome, and lets the control plane decide -- instead of holding
        it LEASED forever, where no sibling replica may take it over.

        The thread is a daemon and the outcome is settled under a lock, so a
        late result cannot be posted under a lease this executor stopped
        renewing, and shutdown does not wait on the abandoned thread.
        """

        settled = Lock()
        outcome: dict[str, Any] = {}

        def execute() -> None:
            # contextvars do not cross thread boundaries, so the adapter's
            # lease guard has to be set on the thread that does the sending.
            guard_token = active_lease_guard.set(watch.hold_reason)
            try:
                result: RemoteCommandResult | None = None
                error: BaseException | None = None
                try:
                    result = self.executor._execute(command)
                except BaseException as exc:  # noqa: BLE001 - re-raised below
                    error = exc
                with settled:
                    if outcome.get("abandoned"):
                        self.executor.increment("stuck_executions", -1)
                        LOGGER.warning(
                            "abandoned regional command finally returned after "
                            "the execution cap; the result is discarded: "
                            "command=%s cluster=%s operation=%s status=%s",
                            command.command_id,
                            command.cluster_id,
                            command.step.operation.value,
                            None if result is None else result.status.value,
                            exc_info=error,
                        )
                        return
                    if error is not None:
                        outcome["error"] = error
                    else:
                        outcome["result"] = result
            finally:
                active_lease_guard.reset(guard_token)

        worker = Thread(
            target=execute,
            name=f"cmd-{command.command_id[:24]}",
            daemon=True,
        )
        worker.start()
        worker.join(self.executor.max_execution_seconds)
        with settled:
            if "error" in outcome:
                raise outcome["error"]
            if "result" in outcome:
                settled_result: RemoteCommandResult = outcome["result"]
                return settled_result
            outcome["abandoned"] = True
            self.executor.increment("execution_timeouts_total")
            self.executor.increment("stuck_executions")
        if self._timeout_outcome_is_unknown(command):
            # The thread may still be mutating a node. Hand it to the caller so
            # the lease can be held until the verdict lands or the thread comes
            # back: an expired lease over a live mutation is how two replicas
            # end up resetting the same GPU.
            abandoned["worker"] = worker
        else:
            # Nothing outside this process is in flight, so stop the heartbeat
            # before the verdict is posted: telling the control plane the lease
            # is alive while reporting a step this executor gave up on would be
            # two contradictory claims in the same cycle.
            stop.set()
        LOGGER.error(
            "regional command exceeded the execution cap and was abandoned; "
            "the thread cannot be killed and stays stuck: command=%s "
            "cluster=%s operation=%s nodes=%s cap=%.1fs",
            command.command_id,
            command.cluster_id,
            command.step.operation.value,
            ",".join(command.step.node_ids),
            self.executor.max_execution_seconds,
        )
        return self._execution_timeout_result(command)

    def _execution_timeout_result(
        self, command: RemoteActionCommand
    ) -> RemoteCommandResult:
        """The verdict for a command this executor gave up waiting for.

        FAILED, because the step did not succeed and the workflow must not stay
        WAITING on a thread nobody will ever hear from again. But never a plain
        failure for anything that mutates: the node action may still be running
        on the node, so the result marks the outcome unknown and demands manual
        confirmation (the INTERRUPTED shape); the hardware escalation reads
        those ``details`` and hands the step to an operator, no rung climbed.
        """

        operation = command.step.operation
        unknown = self._timeout_outcome_is_unknown(command)
        details: dict[str, Any] = {
            "execution_timeout": True,
            "execution_timeout_seconds": (self.executor.max_execution_seconds),
            "operation": operation.value,
        }
        pointer = (command.result_details or {}).get("node_action_command_id")
        if pointer is not None:
            # The ledger row is how an operator learns what the agent did.
            details["node_action_command_id"] = pointer
        if unknown:
            # Only here, like ``status_source``: a read-only timeout did not
            # happen, and this is the key that means "go look at the node".
            # ``node_failures`` is the per-node cause the escalation quotes.
            details["outcome_unknown"] = True
            details["manual_confirmation_required"] = True
            details["node_failures"] = {
                node_id: ["execution timed out; outcome unknown"]
                for node_id in command.step.node_ids
            }
        return RemoteCommandResult(
            lease_token=str(command.lease_token),
            status=RemoteCommandStatus.FAILED,
            status_source=(
                EXECUTION_TIMEOUT_UNKNOWN_STATUS_SOURCE
                if unknown
                else EXECUTION_TIMEOUT_STATUS_SOURCE
            ),
            details=details,
            error=(
                f"executor abandoned {operation.value} after "
                f"{self.executor.max_execution_seconds:.0f}s"
                + (
                    "; the outcome is unknown and the operation may still be "
                    "running on the node"
                    if unknown
                    else "; the operation mutates no node state, so nothing "
                    "is left running"
                )
            ),
        )

    @staticmethod
    def _timeout_outcome_is_unknown(command: RemoteActionCommand) -> bool:
        """Whether abandoning this command leaves state nobody can see.

        A node action lives in the agent's own ledger, a destructive operation
        may have half-applied, and a support escalation may already have opened
        a case. For those the timeout is not a verdict about the step, only
        about this executor's patience.
        """

        return command.step.operation in _TIMEOUT_UNKNOWN_STATE_OPERATIONS

    def _hold_lease_for_abandoned_worker(
        self,
        command: RemoteActionCommand,
        stop: Event,
        renewer: Thread,
        worker: Thread,
    ) -> None:
        """Keep an abandoned command LEASED while its thread may still mutate.

        Reached only when all three are true: the outcome is unknown, the
        timeout verdict never reached the control plane (so the command is still
        open), and the thread has not come back. Letting the lease lapse there
        would let a sibling replica claim the same destructive command while
        this process is, as far as anyone can tell, still driving it -- two
        replicas resetting one GPU is worse than one command parked for a while.

        Bounded on purpose: renewal ends when the thread returns, when SIGTERM
        arrives (the process is going away and its threads with it, so the agent
        ledger becomes the only record either way) or after one more execution
        window, whichever comes first. A guard thread rather than an inline wait,
        because the poll loop must keep turning.
        """

        self.executor.increment("abandoned_lease_holds_total")
        LOGGER.error(
            "regional cluster executor is holding a lease for an abandoned "
            "command whose verdict never landed: command=%s cluster=%s "
            "operation=%s nodes=%s hold_window=%.1fs",
            command.command_id,
            command.cluster_id,
            command.step.operation.value,
            ",".join(command.step.node_ids),
            self.executor.max_execution_seconds,
        )
        deadline = self.executor.clock() + self.executor.max_execution_seconds

        def hold() -> None:
            try:
                while worker.is_alive() and not self.executor.stop_requested:
                    if self.executor.clock() >= deadline:
                        LOGGER.error(
                            "abandoned regional command is still running after "
                            "the lease hold window; releasing the lease so the "
                            "control plane can decide: command=%s operation=%s",
                            command.command_id,
                            command.step.operation.value,
                        )
                        break
                    worker.join(_ABANDONED_WORKER_POLL_SECONDS)
            finally:
                stop.set()
                renewer.join(timeout=2)

        Thread(
            target=hold,
            name=f"hold-{command.command_id[:24]}",
            daemon=True,
        ).start()

    def _report_result(
        self,
        command: RemoteActionCommand,
        result: RemoteCommandResult,
        watch: CommandLeaseWatch,
    ) -> bool:
        """Post one result, retrying a transport failure under a live lease.

        The action has already happened on the cluster, so this post is the
        only place its verdict exists. Dropping it on the first ``URLError``
        left the command LEASED until expiry and then re-executed by whichever
        replica re-claimed it -- a second RESET_GPU or STOP_WORKLOADS for one
        workflow step. Retrying is safe because ``complete_remote_command`` is
        idempotent: a command that is already terminal is returned unchanged.

        The retry is bounded three ways, because a report that keeps failing
        must not become a spin: at most ``_RESULT_REPORT_ATTEMPTS`` attempts,
        only for failures with no verdict in them (no HTTP status, or one of
        the transient statuses), and only while the lease is still ours -- past
        that, another executor may already own the command and this result
        would race its result, which is exactly what the withhold rule above
        exists to prevent. A cancellation is not that: the store accepts a cancelled
        command's result, so the retry goes on under the lease this executor holds.
        """

        for attempt in range(1, _RESULT_REPORT_ATTEMPTS + 1):
            try:
                self.executor.client.complete(command, result)
                self.executor.metrics.result_posted(result.status)
                return True
            except Exception as exc:
                hold_reason = watch.hold_reason()
                last_attempt = attempt >= _RESULT_REPORT_ATTEMPTS
                if not _retryable_report_failure(exc) or last_attempt or watch.lost():
                    # A rejected result (stale lease, stale fencing token,
                    # already terminal) and an unreachable control plane both
                    # end here, and neither may discard the results of the
                    # remaining commands in this batch. A cancellation is not
                    # a reason to stop: the store accepts a cancelled command's
                    # result, and the lease is still ours -- only a lost lease
                    # means another replica may own it.
                    LOGGER.exception(
                        "regional cluster executor could not report result: "
                        "command=%s cluster=%s operation=%s status=%s "
                        "attempts=%d lease_hold=%s",
                        command.command_id,
                        command.cluster_id,
                        command.step.operation.value,
                        result.status.value,
                        attempt,
                        hold_reason,
                    )
                    self.executor.increment("reported_failures")
                    return False
                delay = _report_backoff_seconds(attempt)
                LOGGER.warning(
                    "regional cluster executor could not report result; "
                    "retrying in %.2fs: command=%s cluster=%s operation=%s "
                    "status=%s attempt=%d: %s: %s",
                    delay,
                    command.command_id,
                    command.cluster_id,
                    command.step.operation.value,
                    result.status.value,
                    attempt,
                    type(exc).__name__,
                    exc,
                )
                self.executor.increment("transport_retries_total")
                self.executor.sleep(delay)
                # The backoff is where a lease actually runs out: the local
                # deadline can be crossed and the renewer can hit its failure
                # limit while this thread sleeps. Re-check before re-posting,
                # or the retry sends a result under a lease another replica may
                # already own -- the very race the withhold rule prevents. This
                # is a withheld result, not a reporting failure: nothing was
                # refused, and the next lease holder will redo the step.
                hold_reason = watch.hold_reason()
                if watch.lost():
                    self.executor.increment("results_withheld_total")
                    LOGGER.warning(
                        "regional cluster executor withheld an unreported "
                        "result: the lease lapsed during the report backoff: "
                        "command=%s cluster=%s operation=%s status=%s "
                        "attempts=%d reason=%s",
                        command.command_id,
                        command.cluster_id,
                        command.step.operation.value,
                        result.status.value,
                        attempt,
                        hold_reason,
                    )
                    return False
        # Unreachable: the final attempt always takes the terminal branch
        # above, which is where ``reported_failures`` is counted. Fail loudly
        # rather than returning an uncounted False if that ever changes -- the
        # ``_execute_and_report`` boundary turns this into WAITING.
        raise AssertionError(
            "the result report loop must terminate inside its final attempt"
        )

    def renew_lease(
        self,
        command: RemoteActionCommand,
        stop: Event,
        watch: CommandLeaseWatch | None = None,
    ) -> None:
        # ``lease_seconds`` is validated to 10..7200 at construction, so a third
        # of it is never below 3.3s; the only clamp that can bind is the 30s
        # ceiling that keeps a long lease from going unrenewed for minutes.
        interval = min(30.0, self.executor.lease_seconds / 3)
        while not stop.wait(interval):
            if self.executor.stop_requested:
                # Renewing now would park the command for a full lease window
                # after this process is gone. Let it lapse instead.
                LOGGER.warning(
                    "regional command lease left to lapse during shutdown: "
                    "command=%s cluster=%s reason=%s",
                    command.command_id,
                    command.cluster_id,
                    self.executor.stop_reason,
                )
                return
            try:
                renewed = self.executor.client.renew(
                    command,
                    self.executor.executor_id,
                    self.executor.lease_seconds,
                )
            except Exception as exc:
                self.executor.increment("lease_renewal_failures")
                LOGGER.exception(
                    "regional command lease renewal failed: command=%s cluster=%s",
                    command.command_id,
                    command.cluster_id,
                )
                if watch is not None and watch.renewal_failed(exc):
                    self.executor.increment("lease_lost_total")
                    LOGGER.error(
                        "regional command lease treated as lost; no further "
                        "node actions will start and the result will not be "
                        "posted: command=%s cluster=%s reason=%s",
                        command.command_id,
                        command.cluster_id,
                        watch.lost_reason,
                    )
                    return
                continue
            if watch is None:
                continue
            cancellation = watch.renewed(renewed)
            if cancellation is not None:
                self.executor.increment("cancellations_observed_total")
                LOGGER.warning(
                    "regional command cancellation requested during execution; "
                    "no further node actions will start: command=%s cluster=%s "
                    "reason=%s",
                    command.command_id,
                    command.cluster_id,
                    cancellation,
                )
