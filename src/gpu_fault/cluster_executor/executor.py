"""The claim loop: what the regional executor process does all day.

``ClusterActionExecutor`` claims commands from the control plane, runs each
one on its own thread through ``CommandLifecycle`` (lease renewal, execution
cap, result report) and ``CommandDispatch`` (validation, preflight, adapter,
failure classification), keeps the counters every layer increments, writes the
readiness and liveness breadcrumbs the container probes read, and turns
SIGTERM into "stop claiming, let the leases lapse". ``ClaimLoop`` is ``run()``:
every claimed command finishes and reports on its own worker, and the slot it
frees is claimed again at once. The ``SpareReservationSweep``
(``gpu_fault.spare_reservation_sweep``) is the one piece of housekeeping the
loop runs between claims.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_for_futures
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable

from gpu_fault.cluster_executor.dispatch import CommandDispatch
from gpu_fault.cluster_executor.lease import (
    DEFAULT_MAX_EXECUTION_SECONDS,
    CommandLifecycle,
    CommandLeaseWatch,
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
from gpu_fault.cluster_executor.node_inventory import NodeInventoryReporter
from gpu_fault.cluster_executor.node_key_sync import NodeActionKeySync
from gpu_fault.spare_reservation_sweep import SpareReservationSweep

# Deliberately the pre-split module's name and not ``__name__``: the log format
# carries ``%(name)s`` and operators filter on ``gpu_fault.cluster_executor``, so
# every layer of the package logs under the one name it always had.
LOGGER = logging.getLogger("gpu_fault.cluster_executor")
#: An idle executor logs that it is polling at most this often.
HEARTBEAT_LOG_SECONDS = 30.0


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


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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
        node_key_sync: NodeActionKeySync | None = None,
        node_inventory: NodeInventoryReporter | None = None,
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
        # The wall clock the claim and renewal timestamps are read from, which
        # the out-of-process readiness probe ages against its own. An attribute
        # rather than a parameter: a test moves it to age the breadcrumb.
        self.now: Callable[[], datetime] = _utc_now
        self._last_heartbeat_log_at: float | None = None
        # Only the report backoff sleeps inside a command's own thread, so it
        # is injectable: a test must not really wait half a second to prove the
        # retry happened.
        self.sleep = sleep
        self.spare_reservation_sweep = spare_reservation_sweep
        self.node_key_sync = node_key_sync
        self.node_inventory = node_inventory
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
        # The last successful lease renewal of a command in flight, assigned by
        # the renewer threads (``lease_renewed``). It is the same authenticated
        # round trip as a claim -- token, trust chain, route -- so while every
        # slot holds a long command and nothing is claimed, it is what keeps
        # the readiness breadcrumb fresh (``refresh_claim_state``).
        self.last_successful_renewal_at: datetime | None = None
        self._renewal_in_breadcrumb: datetime | None = None
        # Whether the last claim cycle -- one claim's commands, judged once
        # the last of them reported -- moved any command off WAITING. run()
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
        self._record_claim_state(claimed_at)

    def advertise_execution_owners(self) -> None:
        """Write the readiness breadcrumb before the first claim.

        The readiness probe reads ``execution_owners`` from this file, and
        the control plane calls an executor that advertises none not ready.
        Owners are configuration, not the result of a claim: a cluster still
        PENDING in the registry (a join before activation) has every claim
        refused with 423, so a breadcrumb written only by a successful claim
        kept the executor Deployment from ever becoming Ready (live,
        2026-09-12). ``last_successful_claim_at`` stays null until a claim.
        """

        self._record_claim_state(None)

    def lease_renewed(self) -> None:
        """A lease renewal of a command in flight landed (admission or renewer).

        Called from the renewer threads; the loop thread copies the timestamp
        into the readiness breadcrumb on its own cadence.
        """

        renewed_at = self.now()
        self.last_successful_renewal_at = renewed_at
        self.metrics.lease_renewed(renewed_at)

    def refresh_claim_state(self) -> None:
        """Rewrite the readiness breadcrumb when a renewal landed since the last.

        The claim loop calls this while it waits: with every slot holding a
        long command nothing is claimed, and a breadcrumb only claims rewrote
        aged into "last successful claim was 301s ago" -- not ready -- for an
        executor whose leases were all renewing. A loop that neither claims nor
        rewrites this file still ages out, which is the stuck loop the check is
        for.
        """

        if self.last_successful_renewal_at != self._renewal_in_breadcrumb:
            self._record_claim_state(self.last_successful_claim_at)

    def _record_claim_state(self, claimed_at: datetime | None) -> None:
        renewed_at = self.last_successful_renewal_at
        self._renewal_in_breadcrumb = renewed_at
        try:
            path = self.claim_state_path
            temporary = f"{path}.tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "executor_id": self.executor_id,
                        "execution_owners": self.execution_owners,
                        "last_successful_claim_at": (
                            claimed_at.isoformat() if claimed_at is not None else None
                        ),
                        "last_successful_renewal_at": (
                            renewed_at.isoformat() if renewed_at is not None else None
                        ),
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
        """One whole claim cycle: claim, run every command, return the count.

        Claims up to ``batch_size`` and returns only once every claimed
        command has reported -- the single-cycle primitive tests and the
        acceptance probes drive. ``run()`` no longer loops over it: waiting for
        the whole batch left every freed slot idle and every new claim queued
        behind the batch's longest command (``ClaimLoop``).
        """

        claimed = self._claim(self.batch_size)
        self.last_cycle_advanced = True
        if claimed:
            self._conclude_cycle(self._run_cycle(claimed))
        else:
            self._clear_degraded_streak()
        return len(claimed)

    def _claim(
        self, max_commands: int
    ) -> list[tuple[RemoteActionCommand, CommandLeaseWatch]]:
        """One claim round trip; each claimed command with its lease watch."""

        # Before the claim, not after: a claim that hangs must let the
        # breadcrumb go stale, and a claim that fails must not (the loop is
        # alive and merely cannot reach the control plane).
        self._record_liveness()
        try:
            commands = self.client.claim(
                self.executor_id,
                execution_owners=self.execution_owners,
                max_commands=max_commands,
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
        claimed = [
            (command, self.lifecycle.watch_claim(command)) for command in commands
        ]
        # A successful claim round-trip proves the token, the TLS trust
        # chain and the control-plane route all work, even when the
        # queue is empty. That is the only useful executor liveness
        # signal; the readiness marker file only proves pip install ran.
        self.last_successful_claim_at = self.now()
        self.increment("claimed_total", len(commands))
        self.metrics.claimed(self.last_successful_claim_at)
        self._record_successful_claim(self.last_successful_claim_at)
        self._log_claims(commands)
        return claimed

    def _run_cycle(
        self, claimed: list[tuple[RemoteActionCommand, CommandLeaseWatch]]
    ) -> list[CommandOutcome]:
        """``run_once``'s batch: every claimed command run to its report."""

        pool = ThreadPoolExecutor(
            max_workers=min(self.max_concurrent_commands, len(claimed)),
            thread_name_prefix="gpu-fault-command",
        )
        try:
            futures = [
                pool.submit(self._execute_and_report, command, watch)
                for command, watch in claimed
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
            self._log_outcomes([command for command, _ in claimed], outcomes)
        finally:
            # Never wait: a worker that abandoned a stuck command has
            # returned, but the thread it abandoned may still be inside a
            # call that cannot be interrupted.
            pool.shutdown(wait=False)
        return outcomes

    def _conclude_cycle(self, outcomes: list[CommandOutcome]) -> None:
        """Judge one claim's commands once the last of them has reported."""

        # A command that reports WAITING is re-claimable at once, so a
        # batch that only waited puts run() straight back into claim()
        # with nothing changed: on 2026-09-04 a single held
        # STOP_WORKLOADS drove 25 claim/execute/complete round trips a
        # second across two replicas, and 759 identical log lines a
        # minute. Waiting is not progress, so it takes the idle path.
        self.last_cycle_advanced = any(
            outcome.status is not RemoteCommandStatus.WAITING for outcome in outcomes
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
        # Assigned, not incremented, so ``increment`` cannot mirror it: the
        # gauge is set where the streak changes, here and below.
        self.metrics.transport_degraded_cycles(
            self.consecutive_transport_degraded_cycles
        )

    def _clear_degraded_streak(self) -> None:
        """An empty claim with nothing in flight is not a degraded cycle.

        The round trip proves the control-plane route works and leaves
        nothing blocked.
        """

        self.consecutive_transport_degraded_cycles = 0
        self.metrics.transport_degraded_cycles(0)

    def _log_claims(self, commands: list[RemoteActionCommand]) -> None:
        """What this executor took on, at INFO -- the first thing an operator
        reads, and what the acceptance runner's log window expects to find.

        One line per claimed command (id, cluster, step, owner, nodes,
        workflow, generation; never the lease token), plus a heartbeat at most
        every ``HEARTBEAT_LOG_SECONDS`` so an idle executor still shows it is
        polling. Live (ISO-001): the executors logged only the metrics
        listener's start line and executed everything in silence; the runner
        refused the empty window as missing evidence.
        """

        for command in commands:
            step = getattr(command, "step", None)
            LOGGER.info(
                "regional cluster executor claimed command=%s cluster=%s "
                "operation=%s owner=%s nodes=%s workflow=%s step=%s generation=%s",
                getattr(command, "command_id", "?"),
                getattr(command, "cluster_id", "?"),
                getattr(getattr(step, "operation", None), "value", "?"),
                getattr(step, "execution_owner", "?"),
                ",".join(getattr(step, "node_ids", None) or ()),
                getattr(getattr(command, "workflow", None), "request_id", "?"),
                getattr(command, "step_index", "?"),
                getattr(command, "fencing_token", "?"),
            )
        now = self.clock()
        if (
            self._last_heartbeat_log_at is None
            or now - self._last_heartbeat_log_at >= HEARTBEAT_LOG_SECONDS
        ):
            self._last_heartbeat_log_at = now
            LOGGER.info(
                "regional cluster executor heartbeat: executor=%s claimed_now=%d "
                "claimed_total=%d transport_degraded_cycles=%d",
                self.executor_id,
                len(commands),
                self.claimed_total,
                self.consecutive_transport_degraded_cycles,
            )

    @staticmethod
    def _log_outcomes(
        commands: list[RemoteActionCommand], outcomes: list[CommandOutcome]
    ) -> None:
        for command, outcome in zip(commands, outcomes, strict=False):
            step = getattr(command, "step", None)
            LOGGER.info(
                "regional cluster executor finished command=%s operation=%s "
                "nodes=%s status=%s",
                getattr(command, "command_id", "?"),
                getattr(getattr(step, "operation", None), "value", "?"),
                ",".join(getattr(step, "node_ids", None) or ()),
                getattr(getattr(outcome, "status", None), "value", "?"),
            )

    def _execute_and_report(
        self, command: RemoteActionCommand, watch: CommandLeaseWatch | None = None
    ) -> CommandOutcome:
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
            return self.lifecycle.run(command, watch)
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
        # The sweep owns its total and may raise after releasing some of the
        # reservations it set out to, so the series follows the total's delta,
        # not ``run``'s return value.
        before = sweep.reclaimed_total
        try:
            sweep.run()
        except Exception:  # noqa: BLE001 - housekeeping must not stop claims
            LOGGER.warning("spare reservation sweep failed", exc_info=True)
        finally:
            self.metrics.spare_reservations_reclaimed(sweep.reclaimed_total - before)

    def sync_node_action_keys(self) -> None:
        """Key the nodes the Secret lacks when due; never raises into the loop.

        A failed sync is one warning and the next cycle's retry: the claim loop
        must keep turning while the control plane refuses or is unreachable.
        The log carries node names and counts only, never key material.
        """

        sync = self.node_key_sync
        if sync is None or not sync.due():
            return
        try:
            sync.run()
        except Exception as exc:  # noqa: BLE001 - housekeeping must not stop claims
            LOGGER.warning(
                "node action key sync failed (consecutive failures %d): %s: %s",
                sync.failures_total,
                type(exc).__name__,
                exc,
            )

    def report_node_inventory(self) -> None:
        """Post the cluster's Kubernetes Node names when due; never raises.

        The departed-node cleanup's Kubernetes evidence (``node_inventory``).
        """

        if self.node_inventory is not None:
            self.node_inventory.report()

    def run(self) -> None:
        """Claim and execute until a stop is requested, then drain.

        ``ClaimLoop`` does the work: each claimed command reports on its own
        and its slot is claimed again at once. Returns once every command in
        flight at the stop has reported.
        """

        self.advertise_execution_owners()
        ClaimLoop(self).run()
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


@dataclass
class _ClaimCycle:
    """One claim's commands: each reports on its own, judged together."""

    remaining: int
    outcomes: list[CommandOutcome] = field(default_factory=list)


class ClaimLoop:
    """``ClusterActionExecutor.run``: claim into free worker slots, drain on stop.

    ``run()`` used to be ``run_once`` in a loop: claim a batch, wait until every
    command of it had reported, claim again. A reboot of fifteen to forty-five
    minutes therefore held the batch's short commands' follow-up steps and every
    new claim behind it (the quantizer batch 87 removed from the perf harness's
    simulated executor). Here each claimed command runs and reports on its own
    pool worker -- ``_execute_and_report``, so the lease watch, renewer,
    execution cap, fencing checks and bounded idempotent report are the ones it
    always had -- and the slot it frees can be claimed again at once.

    At most ``min(batch_size, max_concurrent_commands)`` commands are in flight,
    what one batch cycle could run at once, and a claim asks for the free slots
    only: every claimed command starts on a free worker, so no claimed lease
    waits in a local queue. Pacing keeps the batch cycle's rules, with one
    claim's commands still the cycle that is judged, once its last command has
    reported (``_conclude_cycle``):

    * a command that finished with a verdict (SUCCEEDED or FAILED) frees its
      slot for an immediate claim -- its workflow's next step is queued behind
      that verdict;
    * a cycle that only waited takes the idle path: the next claim waits
      ``_idle_delay()`` from that conclusion (``poll_seconds``, or the
      network-degraded back-off), and a verdict ends the wait early;
    * with nothing new since the last claim, a long-polling executor claims
      again at once while its last cycle advanced -- the control plane holds
      the claim -- and a polling one waits ``_idle_delay()``;
    * an empty long-poll answer that came back in under half of
      ``claim_wait_seconds`` was not held (the control plane's per-cluster
      waiter cap, no wakeup hub): the next claim starts no sooner than
      ``_idle_delay()`` after it, like a poll, with commands in flight or not;
    * while the network-degraded back-off is active nothing is claimed with
      commands in flight, as in the batch cycle: a replica that cannot reach
      its targets must not compete with its sibling for the work it releases.

    A claim held open while a command runs can take that command again when
    it next becomes claimable (a WAITING hand-back, at the end of the hold or
    on the next PENDING wakeup) before the loop has judged that WAITING; the
    idle path applies from the following claim. Accepted: bounded by the
    claims themselves, never a spin.

    Housekeeping (the spare sweep, the node-key sync) is due-gated and runs on
    every turn of the loop, and while the loop waits it rewrites the readiness
    breadcrumb whenever a lease renewal landed: neither waits for a claim, so
    five long reboots holding every slot keep both on schedule and the Pod
    Ready.

    After SIGTERM nothing new is claimed and ``run`` returns once every
    command in flight has reported; their renewers already stopped, so a
    command the kubelet kills first lapses on the control plane's schedule.
    Owned by the loop thread; the workers only run ``_execute_and_report``.
    """

    def __init__(self, executor: ClusterActionExecutor) -> None:
        self.executor = executor
        self.capacity = min(executor.batch_size, executor.max_concurrent_commands)
        self._pool = ThreadPoolExecutor(
            max_workers=self.capacity, thread_name_prefix="gpu-fault-command"
        )
        self._running: dict[
            Future[CommandOutcome], tuple[int, RemoteActionCommand, _ClaimCycle]
        ] = {}
        self._started = 0
        self._failures = 0
        # Claim at the next turn: the first claim, a claim after the failure
        # back-off, and a slot freed by a verdict.
        self._claim_now = True
        # The deadline of the idle wait in progress, if any.
        self._gate_until: float | None = None

    @property
    def free(self) -> int:
        return self.capacity - len(self._running)

    def run(self) -> None:
        try:
            self._claim_until_stopped()
            self._drain()
        finally:
            # Never wait: a worker that abandoned a stuck command has returned,
            # but the thread it abandoned may still be inside a call that
            # cannot be interrupted.
            self._pool.shutdown(wait=False)

    def _claim_until_stopped(self) -> None:
        executor = self.executor
        while not executor.stop_requested:
            self._settle(0)
            wait = self._next_claim_wait()
            if wait is None or (wait > 0 and self._running):
                # The tick keeps a forty-minute reboot from reading as a wedged
                # loop, and bounds how long a stop request waits.
                tick = executor.liveness_interval_seconds
                self._settle(tick if wait is None else min(wait, tick))
                executor._record_liveness()
                # Renewals, not claims, are the progress while slots are full.
                executor.refresh_claim_state()
            elif wait > 0:
                # Nothing in flight: the batch cycle's plain idle sleep.
                time.sleep(wait)
                self._claim_now = True
            else:
                self._claim()
            if not executor.stop_requested:
                # Due-gated, whether or not this turn claimed: five reboots
                # holding every slot must not stall the spare sweep, the
                # node-key sync or the node inventory for as long as they run.
                executor.sweep_spare_reservations()
                executor.sync_node_action_keys()
                executor.report_node_inventory()

    def _next_claim_wait(self) -> float | None:
        """Seconds until the next claim; None waits for a command to finish."""

        executor = self.executor
        if self.free <= 0:
            return None
        if (
            self._running
            and executor.consecutive_transport_degraded_cycles
            >= executor.transport_degraded_backoff_after
        ):
            return None
        if self._claim_now:
            return 0.0
        if self._gate_until is None:
            if executor.claim_wait_seconds > 0 and executor.last_cycle_advanced:
                # The control plane holds this claim for up to
                # claim_wait_seconds; sleeping first would only add latency.
                return 0.0
            delay = executor._idle_delay()
            self._gate_until = executor.clock() + delay
            return delay
        return max(0.0, self._gate_until - executor.clock())

    def _claim(self) -> None:
        """Claim into the free slots; a failed cycle backs off first."""

        executor = self.executor
        started = executor.clock()
        try:
            claimed = executor._claim(self.free)
            self._start(claimed)
        except Exception as exc:
            self._failures += 1
            time.sleep(self._failure_delay(exc))
            self._claim_now = True
            return
        self._failures = 0
        self._claim_now = False
        self._gate_until = None
        if claimed:
            return
        executor.last_cycle_advanced = True
        if not self._running:
            executor._clear_degraded_streak()
        if (
            executor.claim_wait_seconds > 0
            and executor.clock() - started < executor.claim_wait_seconds / 2
        ):
            # Not held: re-claiming at once would spin on a control plane that
            # answers every long-poll empty and at once. A held claim that a
            # wakeup ended early already waited, so only the rest is slept.
            self._gate_until = started + executor._idle_delay()

    def _start(
        self, claimed: list[tuple[RemoteActionCommand, CommandLeaseWatch]]
    ) -> None:
        if not claimed:
            return
        cycle = _ClaimCycle(remaining=len(claimed))
        for command, watch in claimed:
            future = self._pool.submit(
                self.executor._execute_and_report, command, watch
            )
            self._running[future] = (self._started, command, cycle)
            self._started += 1
        self.executor.metrics.in_flight(len(self._running))

    def _settle(self, timeout: float) -> None:
        """Collect what finished, waiting up to ``timeout`` for the first."""

        if not self._running:
            return
        done, _pending = wait_for_futures(
            tuple(self._running), timeout=timeout, return_when=FIRST_COMPLETED
        )
        if not done:
            return
        executor = self.executor
        commands: list[RemoteActionCommand] = []
        outcomes: list[CommandOutcome] = []
        concluded: list[_ClaimCycle] = []
        # In start order, so the log reads in claim order within one wakeup and
        # cycles that end together are judged in the order they were claimed.
        for future in sorted(done, key=lambda item: self._running[item][0]):
            _order, command, cycle = self._running.pop(future)
            outcome = future.result()
            commands.append(command)
            outcomes.append(outcome)
            cycle.outcomes.append(outcome)
            cycle.remaining -= 1
            if not cycle.remaining:
                concluded.append(cycle)
        executor.metrics.in_flight(len(self._running))
        executor._log_outcomes(commands, outcomes)
        for cycle in concluded:
            executor._conclude_cycle(cycle.outcomes)
            if not executor.last_cycle_advanced:
                # The idle wait runs from this cycle's conclusion.
                self._gate_until = None
        if any(item.status is not RemoteCommandStatus.WAITING for item in outcomes):
            self._claim_now = True

    def _drain(self) -> None:
        """After a stop, wait until every command in flight has reported."""

        executor = self.executor
        if self._running:
            LOGGER.info(
                "regional cluster executor draining before it stops: "
                "executor=%s in_flight=%d",
                executor.executor_id,
                len(self._running),
            )
        while self._running:
            # Every worker is bounded by ``max_execution_seconds``.
            self._settle(executor.liveness_interval_seconds)
            executor._record_liveness()

    def _failure_delay(self, exc: Exception) -> float:
        """Log one failed claim cycle; the back-off before the next claim."""

        executor = self.executor
        failures = self._failures
        delay = min(
            executor.claim_backoff_max_seconds,
            executor.poll_seconds * (1 << min(failures - 1, 8)),
        )
        # A result that could not be posted no longer reaches here at all, but
        # a defect after the claim still can, and calling it "claim failed"
        # sent the operator to the wrong side of the wire (F11).
        phase = (
            "claim failed"
            if isinstance(exc, ClusterExecutorClaimError)
            else "cycle failed after a successful claim"
        )
        if (
            failures == 1
            and isinstance(exc, ClusterExecutorClaimError)
            and exc.status_code is None
        ):
            # No verdict at all (the connection closed or timed out before a
            # status): the client defines this as safe to retry, and one of
            # them is what a control-plane Pod leaving during its rolling
            # restart can cost -- not an executor error. Two in a row are
            # reported as before.
            LOGGER.warning(
                "regional cluster executor claim got no verdict from "
                "the control plane: %s; retrying in %.1fs (consecutive=1)",
                exc,
                delay,
            )
        elif failures == 1 or failures & (failures - 1) == 0:
            LOGGER.exception(
                "regional cluster executor %s; retrying in %.1fs (consecutive=%d)",
                phase,
                delay,
                failures,
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
                failures,
            )
        return delay
