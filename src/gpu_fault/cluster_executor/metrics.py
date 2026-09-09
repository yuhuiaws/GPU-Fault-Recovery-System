"""The executor Pod's Prometheus series (data-plane review F9).

The executor kept its counters and published them nowhere a collector could
read: they rode along in the readiness breadcrumb behind ``kubectl exec ...
cat``. F7 gave the Pod a scrape annotation, a port named ``metrics`` and a
keep rule for ``gpu_fault_cluster_executor_.+`` in the per-cluster ADOT
collector; this module is the family behind that port.

The series are the executor's own counters, not a second set: ``increment``
is the only writer of an executor counter and mirrors every change here, so
the scrape and the breadcrumb always read the same numbers. Results, the loop
breadcrumb and the batch in flight are the three things the counters never
said, and they are recorded at the one place each is decided.
"""

from __future__ import annotations

import logging
import os
import time

from gpu_fault.dataplane_metrics import MetricFamily
from gpu_fault.regional import RemoteCommandStatus

# Deliberately the pre-split module's name and not ``__name__``: the log format
# carries ``%(name)s`` and operators filter on ``gpu_fault.cluster_executor``, so
# every layer of the package logs under the one name it always had.
LOGGER = logging.getLogger("gpu_fault.cluster_executor")

CLUSTER_EXECUTOR_METRICS_PREFIX = "gpu_fault_cluster_executor_"

# Executor counter attribute -> exported series. A counter that is not listed
# stays breadcrumb-only; a series listed here that the executor does not have
# is a construction-time AttributeError in the first test that increments it.
COUNTER_SERIES: dict[str, str] = {
    "claimed_total": "claims_total",
    "execution_timeouts_total": "execution_timeouts_total",
    "abandoned_lease_holds_total": "abandoned_lease_holds_total",
    "fleet_fence_holds_total": "fleet_fence_holds_total",
    "lease_lost_total": "lease_lost_total",
    "transport_retries_total": "transport_retries_total",
}
GAUGE_SERIES: dict[str, str] = {"stuck_executions": "stuck_executions"}
# One counter per verdict: the renderer has no labels, and a labelled
# ``results_posted_total{status=...}`` is what an alert would have to split
# anyway. PENDING and LEASED are queue states a result may never carry.
RESULT_SERIES: dict[RemoteCommandStatus, str] = {
    RemoteCommandStatus.SUCCEEDED: "results_succeeded_total",
    RemoteCommandStatus.FAILED: "results_failed_total",
    RemoteCommandStatus.WAITING: "results_waiting_total",
}


def cluster_executor_metrics() -> MetricFamily:
    return MetricFamily(
        CLUSTER_EXECUTOR_METRICS_PREFIX,
        counters=(
            (
                "claims_total",
                "Commands claimed from the control plane; a claim round trip "
                "that returned nothing adds 0 (see last_claim_timestamp).",
            ),
            (
                "results_succeeded_total",
                "Results posted to the control plane with status SUCCEEDED.",
            ),
            (
                "results_failed_total",
                "Results posted with status FAILED, execution timeouts included.",
            ),
            (
                "results_waiting_total",
                "Results posted with status WAITING: fence holds, barrier holds, "
                "retryable adapter errors and node actions still running.",
            ),
            (
                "execution_timeouts_total",
                "Commands abandoned at max_execution_seconds; the thread behind "
                "each one cannot be killed (see stuck_executions).",
            ),
            (
                "abandoned_lease_holds_total",
                "Leases kept alive over an abandoned worker whose verdict had not "
                "reached the control plane, so no sibling replica takes the node.",
            ),
            (
                "fleet_fence_holds_total",
                "Destructive steps held WAITING because a fleet rollout fenced "
                "the cluster.",
            ),
            (
                "lease_lost_total",
                "Commands whose lease this executor stopped trusting (renewals "
                "exhausted or the window passed); their results were withheld.",
            ),
            (
                "transport_retries_total",
                "Result posts retried after a transport failure with no verdict "
                "in it (connection dropped, timeout); one per backoff sleep.",
            ),
        ),
        gauges=(
            (
                "in_flight_commands",
                "Commands of the current batch still executing on a worker.",
            ),
            (
                "claim_loop_alive",
                "1 while the claim loop last wrote its liveness breadcrumb "
                "successfully, 0 after a failed write or once the loop stopped.",
            ),
            (
                "stuck_executions",
                "Threads still inside a command this executor abandoned at the "
                "execution cap; Python cannot kill them, so they accumulate.",
            ),
        ),
        timestamps=(
            (
                "last_claim_timestamp",
                "Unix seconds of the last successful claim round trip, commands "
                "or not; 0 until the first one.",
            ),
            (
                "last_loop_iteration_timestamp",
                "Unix seconds of the last liveness breadcrumb the claim loop "
                "wrote; ageing past 300 s is the loop having stopped turning.",
            ),
        ),
    )


class ClusterExecutorMetrics:
    """The family plus the four moves the executor makes on it."""

    def __init__(self) -> None:
        self.family = cluster_executor_metrics()

    def counter_changed(self, counter: str, value: int, amount: int) -> None:
        """Mirror one ``executor.increment`` into the exported series."""

        series = COUNTER_SERIES.get(counter)
        if series is not None and amount > 0:
            self.family.inc(series, amount)
        gauge = GAUGE_SERIES.get(counter)
        if gauge is not None:
            self.family.set(gauge, value)

    def result_posted(self, status: RemoteCommandStatus) -> None:
        series = RESULT_SERIES.get(status)
        if series is not None:
            self.family.inc(series)

    def claimed(self) -> None:
        self.family.mark("last_claim_timestamp")

    def loop_iteration(self, *, alive: bool) -> None:
        """The claim loop wrote (or failed to write) its liveness breadcrumb."""

        self.family.set("claim_loop_alive", 1 if alive else 0)
        if alive:
            self.family.mark("last_loop_iteration_timestamp")

    def in_flight(self, count: int) -> None:
        self.family.set("in_flight_commands", count)


def loop_breadcrumb_is_fresh(path: str, max_age_seconds: float) -> bool:
    """The exec probe's question: was the loop breadcrumb touched recently?

    Same file, same age, same answer for a missing file (unhealthy), so
    ``/healthz`` and the container's liveness probe can never disagree.
    """

    try:
        return time.time() - os.path.getmtime(path) < max_age_seconds
    except OSError:
        return False
