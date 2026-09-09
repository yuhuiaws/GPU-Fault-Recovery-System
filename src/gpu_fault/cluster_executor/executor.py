"""The claim loop: what the regional executor process does all day.

``ClusterActionExecutor`` claims commands from the control plane, runs each
one on its own thread through ``CommandLifecycle`` (lease renewal, execution
cap, result report) and ``CommandDispatch`` (validation, preflight, adapter,
failure classification), keeps the counters every layer increments, writes the
readiness and liveness breadcrumbs the container probes read, and turns
SIGTERM into "stop claiming, let the leases lapse". The
``SpareReservationSweep`` (``gpu_fault.spare_reservation_sweep``) is the one
piece of housekeeping the loop runs between claims.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as wait_for_futures
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable

from gpu_fault.cluster_executor.dispatch import CommandDispatch
from gpu_fault.cluster_executor.lease import (
    DEFAULT_MAX_EXECUTION_SECONDS,
    CommandLifecycle,
    CommandOutcome,
)
from gpu_fault.cluster_executor.metrics import ClusterExecutorMetrics
from gpu_fault.cluster_executor.regional_client import (
    ClusterExecutorError,
    RegionalExecutorClient,
)
from gpu_fault.regional import (
    RemoteActionCommand,
    RemoteCommandResult,
    RemoteCommandStatus,
)
from gpu_fault.spare_reservation_sweep import SpareReservationSweep

# Deliberately the pre-split module's name and not ``__name__``: the log format
# carries ``%(name)s`` and operators filter on ``gpu_fault.cluster_executor``, so
# every layer of the package logs under the one name it always had.
LOGGER = logging.getLogger("gpu_fault.cluster_executor")


class ClusterExecutorClaimError(ClusterExecutorError):
    """The claim round trip itself failed, so no command was executed.

    ``run()`` used to log every ``run_once`` exception as "claim failed",
    including a result that could not be posted and a claim response the
    executor could not use -- which pointed the operator at the wrong side of
    the wire. Everything else that can still leave ``run_once`` is a defect
    after the claim, and says so.
    """


# The poll loop's liveness breadcrumb, refreshed every cycle *and* while a
# batch is still running. The readiness breadcrumb cannot answer liveness: it
# is only written after a successful claim, so a control-plane outage makes it
# stale for a loop that is perfectly alive.
DEFAULT_LIVENESS_STATE_PATH = "/tmp/executor-loop-alive"
DEFAULT_LIVENESS_INTERVAL_SECONDS = 15.0
# The age at which the container's liveness probe calls the loop wedged. Wide
# enough that a slow cycle, a long claim or a paused scheduler cannot trip it.
LIVENESS_STALE_AFTER_SECONDS = 300


def transport_degraded_idle_delay(
    *,
    cycles: int,
    backoff_after: int,
    poll_seconds: float,
    cap: float,
) -> float:
    """Idle wait after ``cycles`` consecutive network-degraded run_once cycles.

    Below ``backoff_after`` the executor keeps polling normally. At and past it
    the wait doubles each further degraded cycle (capped at ``cap``), so an
    executor that can reach the control plane but not the target stops
    re-claiming the commands it cannot progress and a replica whose network
    works claims them instead.
    """

    over = cycles - backoff_after
    if over < 0:
        return poll_seconds
    return min(cap, poll_seconds * (1 << min(over + 1, 8)))


def _validate_settings(
    *,
    poll_seconds: float,
    claim_wait_seconds: float,
    lease_seconds: int,
    batch_size: int,
    max_concurrent_commands: int,
    claim_backoff_max_seconds: float,
    lease_renewal_failure_limit: int,
    transport_degraded_backoff_after: int,
    max_execution_seconds: float,
    liveness_interval_seconds: float,
) -> None:
    """Refuse an executor configuration at construction, before the first claim.

    The bounds mirror the control plane's request models where one exists
    (``claim_wait_seconds`` is the protocol field's 0..30) and the operational
    envelope otherwise.
    """

    if poll_seconds <= 0:
        raise ClusterExecutorError("cluster executor poll seconds must be positive")
    if not 0 <= claim_wait_seconds <= 30:
        raise ClusterExecutorError(
            "cluster executor claim wait seconds must be between 0 and 30"
        )
    if not 10 <= lease_seconds <= 7200:
        raise ClusterExecutorError(
            "cluster executor lease seconds must be between 10 and 7200"
        )
    if not 1 <= batch_size <= 25:
        raise ClusterExecutorError(
            "cluster executor batch size must be between 1 and 25"
        )
    if not 1 <= max_concurrent_commands <= 25:
        raise ClusterExecutorError(
            "cluster executor concurrency must be between 1 and 25"
        )
    if claim_backoff_max_seconds < poll_seconds:
        raise ClusterExecutorError(
            "claim backoff max must not be less than poll seconds"
        )
    if not 1 <= lease_renewal_failure_limit <= 100:
        raise ClusterExecutorError(
            "cluster executor lease renewal failure limit must be between 1 and 100"
        )
    if not 1 <= transport_degraded_backoff_after <= 100:
        raise ClusterExecutorError(
            "cluster executor transport degraded backoff must be between 1 and 100"
        )
    if not 0 < max_execution_seconds <= 86400:
        raise ClusterExecutorError(
            "cluster executor max execution seconds must be between 0 and 86400"
        )
    if liveness_interval_seconds <= 0:
        raise ClusterExecutorError(
            "cluster executor liveness interval seconds must be positive"
        )


class ClusterActionExecutor:
    def __init__(
        self,
        client: RegionalExecutorClient,
        adapters: list,
        *,
        executor_id: str,
        allowed_namespaces: set[str],
        poll_seconds: float = 2,
        claim_wait_seconds: float = 0,
        lease_seconds: int = 120,
        batch_size: int = 5,
        max_concurrent_commands: int = 5,
        confirm_cluster_name: str | None = None,
        claim_backoff_max_seconds: float = 60,
        claim_state_path: str | None = None,
        liveness_state_path: str | None = None,
        lease_renewal_failure_limit: int = 3,
        transport_degraded_backoff_after: int = 3,
        max_execution_seconds: float = DEFAULT_MAX_EXECUTION_SECONDS,
        liveness_interval_seconds: float = (DEFAULT_LIVENESS_INTERVAL_SECONDS),
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        spare_reservation_sweep: SpareReservationSweep | None = None,
    ) -> None:
        _validate_settings(
            poll_seconds=poll_seconds,
            claim_wait_seconds=claim_wait_seconds,
            lease_seconds=lease_seconds,
            batch_size=batch_size,
            max_concurrent_commands=max_concurrent_commands,
            claim_backoff_max_seconds=claim_backoff_max_seconds,
            lease_renewal_failure_limit=lease_renewal_failure_limit,
            transport_degraded_backoff_after=transport_degraded_backoff_after,
            max_execution_seconds=max_execution_seconds,
            liveness_interval_seconds=liveness_interval_seconds,
        )
        self.lease_renewal_failure_limit = lease_renewal_failure_limit
        self.transport_degraded_backoff_after = transport_degraded_backoff_after
        self.max_execution_seconds = max_execution_seconds
        self.liveness_interval_seconds = liveness_interval_seconds
        self.clock = clock
        # Only the report backoff sleeps inside a command's own thread, so it
        # is injectable: a test must not really wait half a second to prove the
        # retry happened.
        self.sleep = sleep
        self.spare_reservation_sweep = spare_reservation_sweep
        self.client = client
        self.adapters = adapters
        self.executor_id = executor_id
        self.allowed_namespaces = allowed_namespaces
        # Independent of anything the command carries: read from this
        # executor's own environment so a command for another HyperPod
        # cluster fails the adapter's confirmation gate.
        self.confirm_cluster_name = confirm_cluster_name
        self.poll_seconds = poll_seconds
        # Long-poll: how long the control plane may hold an empty claim. While
        # positive, run() goes straight back into claim() after an empty
        # answer -- the server already waited -- and poll_seconds only paces a
        # held (WAITING) command and the transport backoff. 0 is pure polling.
        self.claim_wait_seconds = claim_wait_seconds
        self.lease_seconds = lease_seconds
        self.batch_size = batch_size
        self.max_concurrent_commands = max_concurrent_commands
        self.claim_backoff_max_seconds = claim_backoff_max_seconds
        self.execution_owners = sorted(
            {
                owner
                for adapter in adapters
                if (owner := getattr(adapter, "owner", None))
            }
        )
        if len(self.execution_owners) != len(adapters):
            raise ClusterExecutorError(
                "every local adapter must declare a unique owner"
            )
        # Liveness and observability counters. A regional executor can be
        # Ready and claiming nothing at all, so operators need a signal
        # that is tied to actual work rather than to process startup.
        #
        # Every one of them is written from more than one thread -- the claim
        # loop, one worker per claimed command, one lease renewer per command
        # -- and ``x += 1`` is a read and a write with a bytecode boundary in
        # between, so concurrent increments were silently lost and the
        # breadcrumb under-reported exactly when the executor was busiest.
        # ``increment`` is the only way they change.
        self._counter_lock = Lock()
        self.claimed_total = 0
        self.reported_failures = 0
        self.unexpected_failures = 0
        self.lease_renewal_failures = 0
        # Commands whose lease this executor stopped trusting (renewals
        # exhausted or the window passed), results it therefore did not
        # post, cancellations seen in a renew response, and multi-node
        # barrier steps held because no coordinator is wired.
        self.lease_lost_total = 0
        self.results_withheld_total = 0
        self.cancellations_observed_total = 0
        self.barrier_unavailable_holds_total = 0
        # Adapter exceptions that say nothing about the step (Kubernetes
        # 409/429/5xx, urllib3 timeouts) reported WAITING instead of FAILED
        # (ARCH-B1 reached from the regional topology).
        self.retryable_adapter_errors_total = 0
        # Actions that failed on this executor's own connectivity (gaierror,
        # refused connection, timeout reaching the target): WAITING as retryable
        # transport. Unlike adapter/control-plane retryables, the executor's
        # network is the problem, not the target's -- see _idle_delay.
        self.retryable_transport_errors_total = 0
        # Compound commands run, the steps they carried, and progress posts
        # the control plane did not accept (性能 C). The terminal result
        # carries every step's verdict, so a failed progress post costs the
        # control plane latency, never a step.
        self.batched_commands_total = 0
        self.batched_steps_total = 0
        self.batched_progress_failures_total = 0
        # Commands abandoned at ``max_execution_seconds`` (a counter), and the
        # threads still stuck behind them (a gauge: Python cannot kill a
        # thread, so an operator has to see them accumulate).
        self.execution_timeouts_total = 0
        self.stuck_executions = 0
        # Leases kept alive for a command this executor abandoned and could not
        # report: the local thread may still be mutating, so the lease is held
        # rather than handed to a sibling replica.
        self.abandoned_lease_holds_total = 0
        # Destructive steps held behind the fleet rollout fence, and result
        # posts retried after a transport failure (F9: both were decided in
        # the dispatch and lease layers with no count anywhere).
        self.fleet_fence_holds_total = 0
        self.transport_retries_total = 0
        # The Prometheus view of the counters above (F9). ``increment`` mirrors
        # every change, so the scrape and the breadcrumb read the same numbers.
        self.metrics = ClusterExecutorMetrics()
        # SIGTERM asks the loop to stop claiming and asks every in-flight
        # command to stop renewing, so the lease lapses on the control plane's
        # own schedule instead of being parked for a full window.
        self._stop_requested = False
        self._stop_reason: str | None = None
        self.last_successful_claim_at: datetime | None = None
        # Whether the last claim cycle moved any command off WAITING. run()
        # takes the idle path when it did not, so a held command polls at
        # poll_seconds instead of as fast as the control plane will answer.
        self.last_cycle_advanced = True
        # Whole cycles in a row that advanced nothing and failed at least one
        # command on this executor's own connectivity. A streak, not a total:
        # written only by the claim loop (never through ``increment``), read
        # by ``_idle_delay`` and the breadcrumb.
        self.consecutive_transport_degraded_cycles = 0
        # The readiness probe runs in a separate process (kubectl exec),
        # so the claim timestamp has to leave this one. Written to the
        # container filesystem, not the store: the default regional
        # executor has no store of its own.
        self.claim_state_path = claim_state_path or os.getenv(
            "GPU_FAULT_CLUSTER_EXECUTOR_CLAIM_STATE_PATH",
            "/tmp/executor-claim-state.json",
        )
        # A separate file from the claim breadcrumb, never a field inside it:
        # the liveness probe must be able to tell "the loop is turning" from
        # "the last claim succeeded", which are different questions with
        # different answers during a control-plane outage.
        #
        # Deliberately a constant and not derived from the claim-state path or
        # from an environment variable of its own: the container's probe reads
        # one hardcoded path, so anything that could move this file at runtime
        # would leave the probe reading a file nobody writes and kill a healthy
        # executor every failure budget. Only an in-process caller (a test) may
        # redirect it.
        self.liveness_state_path = liveness_state_path or DEFAULT_LIVENESS_STATE_PATH
        self.fleet_registry = next(
            (
                registry
                for adapter in adapters
                if (registry := getattr(adapter, "registry", None)) is not None
            ),
            None,
        )
        # The two layers below the claim loop. Both read this executor's
        # configuration and counters live, so nothing here is copied.
        self.lifecycle = CommandLifecycle(self)
        self.dispatch = CommandDispatch(self)

    @property
    def stop_requested(self) -> bool:
        """Whether a signal asked this executor to wind down."""

        return self._stop_requested

    @property
    def stop_reason(self) -> str | None:
        return self._stop_reason

    def request_stop(self, reason: str) -> None:
        """Stop claiming, and stop renewing what is already in flight.

        Renewal is the half that matters for recovery time. Without it a
        rollout that kills the process mid-command leaves the command LEASED
        for a full lease window (120s) before any sibling may re-claim it,
        which is added directly to the incident's recovery time. Nothing is
        cancelled here: the work already done is still reported, and the agent
        ledger still holds whatever the node was doing.
        """

        self._stop_requested = True
        self._stop_reason = reason
        LOGGER.warning(
            "regional cluster executor asked to stop: executor=%s reason=%s",
            self.executor_id,
            reason,
        )

    def _handle_stop_signal(self, signum: int, _frame: Any) -> None:
        self.request_stop(f"{signal.Signals(signum).name} received")

    def install_signal_handlers(self) -> None:
        """Route SIGTERM into ``request_stop``.

        Only SIGTERM: SIGINT must stay a KeyboardInterrupt so an interactive
        run still dies on the first Ctrl-C.
        """

        try:
            signal.signal(signal.SIGTERM, self._handle_stop_signal)
        except ValueError:
            # Only the main thread may install handlers. A library caller that
            # runs the executor on a worker thread owns its own shutdown.
            LOGGER.warning(
                "could not install the SIGTERM handler outside the main thread"
            )

    def _record_liveness(self) -> None:
        """Refresh the poll loop's own breadcrumb.

        Written by the loop and never by a worker: the file's age has to mean
        "the claim loop is turning", so that a wedged claim restarts the Pod
        while a legitimately long command does not.
        """

        try:
            temporary = f"{self.liveness_state_path}.tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "executor_id": self.executor_id,
                        "updated_at": (datetime.now(timezone.utc).isoformat()),
                        "stuck_executions": self.stuck_executions,
                    },
                    handle,
                )
            os.replace(temporary, self.liveness_state_path)
            self.metrics.loop_iteration(alive=True)
        except OSError:
            self.metrics.loop_iteration(alive=False)
            # A breadcrumb that cannot be written goes stale, and stale is the
            # correct direction to fail in: the probe restarts the Pod.
            LOGGER.warning(
                "could not write executor liveness state to %s",
                self.liveness_state_path,
                exc_info=True,
            )

    def increment(self, counter: str, amount: int = 1) -> None:
        """Add to one shared counter under the counter lock."""

        with self._counter_lock:
            value = getattr(self, counter) + amount
            setattr(self, counter, value)
            # Mirrored under the same lock: a gauge set from a released
            # snapshot could apply two racing changes in the wrong order.
            self.metrics.counter_changed(counter, value, amount)

    def metrics_snapshot(self) -> dict[str, Any]:
        """Every executor counter, for the claim breadcrumb and operators.

        The readiness probe reads the claim-state breadcrumb out of process,
        so the counters ride along in that file (``kubectl exec ... cat``)
        next to their ``/metrics`` export: every key here that ``increment``
        writes is also a ``gpu_fault_cluster_executor_*`` series (the metrics
        test pins the two sets against each other), and the two views read
        the same numbers because ``increment`` mirrors each change under this
        lock. Read under the counter lock so the breadcrumb cannot catch a
        half-applied increment.
        """

        with self._counter_lock:
            return self._counters()

    def _counters(self) -> dict[str, Any]:
        return {
            "claimed_total": self.claimed_total,
            "reported_failures": self.reported_failures,
            "unexpected_failures": self.unexpected_failures,
            "lease_renewal_failures": self.lease_renewal_failures,
            "lease_lost_total": self.lease_lost_total,
            "results_withheld_total": self.results_withheld_total,
            "cancellations_observed_total": (self.cancellations_observed_total),
            "barrier_unavailable_holds_total": (self.barrier_unavailable_holds_total),
            "retryable_adapter_errors_total": self.retryable_adapter_errors_total,
            "retryable_transport_errors_total": (self.retryable_transport_errors_total),
            "consecutive_transport_degraded_cycles": (
                self.consecutive_transport_degraded_cycles
            ),
            "batched_commands_total": self.batched_commands_total,
            "batched_steps_total": self.batched_steps_total,
            "batched_progress_failures_total": self.batched_progress_failures_total,
            "execution_timeouts_total": self.execution_timeouts_total,
            "stuck_executions": self.stuck_executions,
            "abandoned_lease_holds_total": (self.abandoned_lease_holds_total),
            "fleet_fence_holds_total": self.fleet_fence_holds_total,
            "transport_retries_total": self.transport_retries_total,
            "spare_reservations_reclaimed_total": (
                self.spare_reservation_sweep.reclaimed_total
                if self.spare_reservation_sweep is not None
                else 0
            ),
            "last_successful_claim_at": (
                self.last_successful_claim_at.isoformat()
                if self.last_successful_claim_at is not None
                else None
            ),
        }

    def _record_successful_claim(self, claimed_at: datetime) -> None:
        try:
            path = self.claim_state_path
            temporary = f"{path}.tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "executor_id": self.executor_id,
                        "execution_owners": self.execution_owners,
                        "last_successful_claim_at": (claimed_at.isoformat()),
                        "counters": self.metrics_snapshot(),
                    },
                    handle,
                )
            os.replace(temporary, path)
        except OSError:
            # Never fail a claim cycle over the readiness breadcrumb.
            # The probe treats a missing or stale file as not-ready,
            # which is the correct direction to fail in.
            LOGGER.warning(
                "could not write executor claim state to %s",
                self.claim_state_path,
                exc_info=True,
            )

    def run_once(self) -> int:
        # Before the claim, not after: a claim that hangs must let the
        # breadcrumb go stale, and a claim that fails must not (the loop is
        # alive and merely cannot reach the control plane).
        self._record_liveness()
        try:
            commands = self.client.claim(
                self.executor_id,
                execution_owners=self.execution_owners,
                max_commands=self.batch_size,
                lease_seconds=self.lease_seconds,
                wait_seconds=self.claim_wait_seconds,
            )
        except Exception as exc:
            # Tag the phase for run()'s log. Everything after this point either
            # handles its own failures or is an executor defect, and both used
            # to be reported to the operator as "claim failed".
            raise ClusterExecutorClaimError(
                f"{type(exc).__name__}: {exc}",
                status_code=getattr(exc, "status_code", None),
            ) from exc
        # A successful claim round-trip proves the token, the TLS trust
        # chain and the control-plane route all work, even when the
        # queue is empty. That is the only useful executor liveness
        # signal; the readiness marker file only proves pip install ran.
        self.last_successful_claim_at = datetime.now(timezone.utc)
        self.increment("claimed_total", len(commands))
        self.metrics.claimed()
        self._record_successful_claim(self.last_successful_claim_at)
        self.last_cycle_advanced = True
        if commands:
            pool = ThreadPoolExecutor(
                max_workers=min(
                    self.max_concurrent_commands,
                    len(commands),
                ),
                thread_name_prefix="gpu-fault-command",
            )
            try:
                futures = [
                    pool.submit(self._execute_and_report, command)
                    for command in commands
                ]
                pending = set(futures)
                self.metrics.in_flight(len(pending))
                while pending:
                    # Every worker is bounded by ``max_execution_seconds``, so
                    # this loop always ends; the tick exists so that a
                    # twenty-minute REPLACE_NODE keeps refreshing the liveness
                    # breadcrumb instead of looking like a wedged loop.
                    _finished, pending = wait_for_futures(
                        pending, timeout=self.liveness_interval_seconds
                    )
                    self.metrics.in_flight(len(pending))
                    self._record_liveness()
                outcomes = [future.result() for future in futures]
            finally:
                # Never wait: a worker that abandoned a stuck command has
                # returned, but the thread it abandoned may still be inside a
                # call that cannot be interrupted.
                pool.shutdown(wait=False)
            # A command that reports WAITING is re-claimable at once, so a
            # batch that only waited puts run() straight back into claim()
            # with nothing changed: on 2026-09-04 a single held
            # STOP_WORKLOADS drove 25 claim/execute/complete round trips a
            # second across two replicas, and 759 identical log lines a
            # minute. Waiting is not progress, so it takes the idle path.
            self.last_cycle_advanced = any(
                outcome.status is not RemoteCommandStatus.WAITING
                for outcome in outcomes
            )
            # A whole cycle that advanced nothing and failed at least one
            # command on this executor's own connectivity is a degraded
            # cycle: run() backs off once enough of them stack up, so the
            # command this executor keeps releasing lands on a replica whose
            # network works. Any progress, or a WAITING that was the target's
            # fault (an adapter 5xx, a control-plane 503), clears the streak.
            if not self.last_cycle_advanced and any(
                outcome.transport_degraded for outcome in outcomes
            ):
                self.consecutive_transport_degraded_cycles += 1
            else:
                self.consecutive_transport_degraded_cycles = 0
        else:
            # An empty claim round trip proves the control-plane route works
            # and leaves nothing blocked, so it is not a degraded cycle.
            self.consecutive_transport_degraded_cycles = 0
        return len(commands)

    def _execute_and_report(self, command: RemoteActionCommand) -> CommandOutcome:
        """One command, start to reported: never raises into ``run_once``.

        Anything that leaves this method comes back out of ``future.result()``
        in ``run_once``, which loses every *sibling* command's report and sends
        ``run()`` into claim backoff while this command sits LEASED until it
        expires and is executed again. ``_execute`` answers its own failures
        with a result, so whatever arrives here -- a claimed command with no
        lease token, a thread that could not be started, a defect in the
        withhold or report path -- is an executor-side defect, counted as one
        and held WAITING for the next lease holder.
        """

        try:
            return self.lifecycle.run(command)
        except Exception:
            self.increment("unexpected_failures")
            LOGGER.exception(
                "regional cluster executor raised outside command execution: "
                "command=%s cluster=%s",
                command.command_id,
                command.cluster_id,
            )
            return CommandOutcome(RemoteCommandStatus.WAITING, False)

    def sweep_spare_reservations(self) -> None:
        """Run the periodic spare sweep when due; never raises (ARCH-A4b)."""

        sweep = self.spare_reservation_sweep
        if sweep is None or not sweep.due():
            return
        try:
            sweep.run()
        except Exception:  # noqa: BLE001 - housekeeping must not stop claims
            LOGGER.warning("spare reservation sweep failed", exc_info=True)

    def run(self) -> None:
        consecutive_failures = 0
        while not self._stop_requested:
            try:
                count = self.run_once()
                consecutive_failures = 0
                self.sweep_spare_reservations()
            except Exception as exc:
                consecutive_failures += 1
                delay = min(
                    self.claim_backoff_max_seconds,
                    self.poll_seconds * (2 ** min(consecutive_failures - 1, 8)),
                )
                # A result that could not be posted no longer reaches here at
                # all, but a defect after the claim still can, and calling it
                # "claim failed" sent the operator to the wrong side of the
                # wire (F11).
                phase = (
                    "claim failed"
                    if isinstance(exc, ClusterExecutorClaimError)
                    else "cycle failed after a successful claim"
                )
                if (
                    consecutive_failures == 1
                    or consecutive_failures & (consecutive_failures - 1) == 0
                ):
                    LOGGER.exception(
                        "regional cluster executor %s; "
                        "retrying in %.1fs (consecutive=%d)",
                        phase,
                        delay,
                        consecutive_failures,
                    )
                else:
                    LOGGER.warning(
                        "regional cluster executor %s, still failing: "
                        "%s: %s; retrying in %.1fs "
                        "(consecutive=%d)",
                        phase,
                        type(exc).__name__,
                        exc,
                        delay,
                        consecutive_failures,
                    )
                time.sleep(delay)
                continue
            if count == 0 and self.claim_wait_seconds > 0:
                # The control plane held this claim for up to
                # claim_wait_seconds already; sleeping again would only add
                # latency to the next command.
                continue
            if count == 0 or not self.last_cycle_advanced:
                time.sleep(self._idle_delay())
        self.metrics.loop_iteration(alive=False)
        LOGGER.warning(
            "regional cluster executor stopped claiming: executor=%s reason=%s",
            self.executor_id,
            self._stop_reason,
        )

    def _idle_delay(self) -> float:
        """How long run() waits before the next claim when nothing advanced."""

        return transport_degraded_idle_delay(
            cycles=self.consecutive_transport_degraded_cycles,
            backoff_after=self.transport_degraded_backoff_after,
            poll_seconds=self.poll_seconds,
            cap=self.claim_backoff_max_seconds,
        )

    def _execute(self, command: RemoteActionCommand) -> RemoteCommandResult:
        """The lifecycle's worker thread enters the dispatch layer here.

        One method on the executor rather than a direct call into ``dispatch``
        so a caller holding only the executor -- the lifecycle, a test that
        replaces the whole execution -- has a single place to intercept.
        """

        return self.dispatch.execute(command)
