from __future__ import annotations

import json
import os
import ssl
import statistics
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from urllib import error, request  # noqa: F401 - error: what callers classify

try:
    from . import benchmark_transport_retry as transport
except ImportError:  # /scripts in the executor Pod, shipped beside this file
    import benchmark_transport_retry as transport

# Every owner a planned step can carry, both providers' lifecycle adapters
# included: the simulator stands in for whichever executor the synthetic
# cluster's profile names (a profile from the other provider's cluster still
# fails the seed's provider check; this list is not what hides that).
OWNERS = [
    "gpu-fault-kubernetes-adapter",
    "gpu-fault-node-agent",
    "gpu-fault-hyperpod-adapter",
    "gpu-fault-eks-adapter",
    "gpu-fault-validation-adapter",
]
DELAYS = {
    "MARK_UNSCHEDULABLE": 0.5,
    "CHECKPOINT_WORKLOADS": 3.0,
    "STOP_WORKLOADS": 2.0,
    "COLLECT_DIAGNOSTIC_BUNDLE": 4.0,
    "QUIESCE_GPU_SERVICES": 2.0,
    "VERIFY_NO_GPU_CLIENTS": 1.0,
    "RESET_GPU": 8.0,
    "RESTORE_GPU_SERVICES": 2.0,
    "RESTART_NODE": 120.0,
    "VALIDATE_GPU": 3.0,
    "VALIDATE_HOST": 2.0,
    "VALIDATE_FABRIC": 3.0,
    "RESTART_WORKLOAD": 2.0,
    "RESTORE_SCHEDULING": 0.5,
}
CLAIM_ERROR_SAMPLE_LIMIT = transport.SAMPLE_LIMIT
# How the claim loop paces itself: an empty claim waits as long as it always
# did, a failed claim waits the product executor's retry pause
# (``benchmark_transport_retry``), and a loop whose every worker slot is busy
# waits for the first command to finish (bounded so the wall-clock limit is
# still checked).
IDLE_POLL_SECONDS = 0.05
BUSY_WAIT_SECONDS = 1.0


def claim_payload(executor_id: str, max_commands: int) -> dict:
    return {
        "executor_id": executor_id,
        "executor_protocol_version": int(os.environ["EXECUTOR_PROTOCOL_VERSION"]),
        "executor_artifact_sha256": os.environ["EXECUTOR_ARTIFACT_SHA256"],
        "executor_compatibility_digest": os.environ["EXECUTOR_COMPATIBILITY_DIGEST"],
        "execution_owners": OWNERS,
        "max_commands": max_commands,
        "lease_seconds": int(os.getenv("ACTION_LEASE_SECONDS", "120")),
    }


def percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[int((len(ordered) - 1) * ratio)]


request_error_category = transport.category


class Client:
    def __init__(
        self,
        *,
        base_url: str,
        cluster_id: str,
        token: str,
        ca_file: str,
        timeout: float = 30,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cluster_id = cluster_id
        self.token = token
        self.context = ssl.create_default_context(cafile=ca_file)
        self.timeout = timeout

    def post(self, path: str, payload: dict) -> dict:
        value = request.Request(
            self.base_url + path,
            data=json.dumps(payload, separators=(",", ":")).encode(),
            headers={
                "Authorization": f"Bearer {self.token}",
                "X-GPU-Fault-Cluster-ID": self.cluster_id,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(
            value,
            context=self.context,
            timeout=self.timeout,
        ) as response:
            return json.loads(response.read() or b"{}")


@dataclass(frozen=True)
class Outcome:
    """One command's end: ``failure`` is the category of a lost result (still
    undelivered after its attempts, or refused); ``retried`` lists the result
    posts that failed transiently and were repeated."""

    command_id: str
    operation: str
    elapsed: float
    failure: str | None
    retried: list[str] = field(default_factory=list)


@dataclass
class SimulationState:
    client: Client
    executor_id: str
    operation_delays: dict[str, float]
    delay_scale: float
    lease_seconds: int
    renewal_interval: float
    inject_renew_failure_pending: bool
    # The simulated action duration; a test substitutes a virtual clock.
    sleep: Callable[[float], None] = time.sleep
    # The pause between result attempts (``transport.deliver``).
    retry_sleep: Callable[[float], None] = time.sleep
    counters_lock: Lock = field(default_factory=Lock)
    renewals: int = 0
    renewal_errors: int = 0
    injected_renewal_failures: int = 0
    long_commands: int = 0
    active_commands: int = 0
    max_active_commands: int = 0

    def execute(self, command: dict) -> Outcome:
        command_id = command["command_id"]
        operation = command["step"]["operation"]
        node_count = len(command["step"].get("node_ids") or [])
        delay = (
            self.operation_delays.get(operation, 0.5) + 0.01 * max(1, node_count)
        ) * self.delay_scale
        with self.counters_lock:
            self.active_commands += 1
            self.max_active_commands = max(
                self.max_active_commands,
                self.active_commands,
            )
            if delay >= self.lease_seconds:
                self.long_commands += 1
        stop_renewal = Event()

        def renew() -> None:
            while not stop_renewal.wait(self.renewal_interval):
                with self.counters_lock:
                    if self.inject_renew_failure_pending:
                        self.inject_renew_failure_pending = False
                        self.injected_renewal_failures += 1
                        self.renewal_errors += 1
                        continue
                try:
                    self.client.post(
                        f"/v1/regional/executors/{command_id}/renew",
                        {
                            "executor_id": self.executor_id,
                            "lease_token": command["lease_token"],
                            "lease_seconds": self.lease_seconds,
                        },
                    )
                except Exception:
                    with self.counters_lock:
                        self.renewal_errors += 1
                else:
                    with self.counters_lock:
                        self.renewals += 1

        renewal_thread = Thread(
            target=renew,
            name=f"renew-{command_id}",
            daemon=True,
        )
        renewal_thread.start()
        begin = time.monotonic()
        result = {
            "lease_token": command["lease_token"],
            "status": "SUCCEEDED",
            "status_source": "action-capacity-simulator",
            "details": {
                "simulated": True,
                "operation": operation,
                "node_count": node_count,
                "executor_id": self.executor_id,
                "simulated_delay_seconds": delay,
            },
        }
        try:
            self.sleep(delay)
            # The same body on every attempt: the lease token makes it idempotent.
            _answer, retried = transport.deliver(
                lambda: self.client.post(
                    f"/v1/regional/executors/{command_id}/result", result
                ),
                sleep=self.retry_sleep,
            )
        except transport.Undelivered as lost:
            return Outcome(
                command_id,
                operation,
                time.monotonic() - begin,
                lost.category,
                lost.retried,
            )
        finally:
            stop_renewal.set()
            renewal_thread.join(timeout=max(1.0, self.renewal_interval + 1))
            with self.counters_lock:
                self.active_commands -= 1
        return Outcome(command_id, operation, time.monotonic() - begin, None, retried)


def operation_delays() -> dict[str, float]:
    return {
        **DELAYS,
        **{
            str(name): float(value)
            for name, value in json.loads(
                os.getenv("ACTION_OPERATION_DELAYS_JSON", "{}")
            ).items()
        },
    }


@dataclass
class ClaimTally:
    """What one executor's claim loop observed, for its summary document.

    ``claim_errors`` and ``result_errors`` are the fatal ones -- a refused or
    persistently failing claim, a result still undelivered after its
    attempts; the transient failures that were retried are counted apart
    (``claims.retries``, ``result_retries``) and never fail the Pod.
    """

    completed_ids: set[str] = field(default_factory=set)
    duplicate_claims: int = 0
    claims: transport.ClaimRetries = field(default_factory=transport.ClaimRetries)
    claim_errors: int = 0
    claim_error_counts: dict[str, int] = field(default_factory=dict)
    claim_error_samples: list[dict[str, str | float]] = field(default_factory=list)
    result_errors: int = 0
    result_error_counts: dict[str, int] = field(default_factory=dict)
    result_retries: int = 0
    result_retry_counts: dict[str, int] = field(default_factory=dict)
    latencies: list[float] = field(default_factory=list)
    by_operation: dict[str, list[float]] = field(default_factory=dict)
    wall_seconds: float = 0.0

    def claim_failed(self, exc: BaseException, *, elapsed: float) -> None:
        self.claim_errors += 1
        category = request_error_category(exc)
        transport.count(self.claim_error_counts, category)
        if len(self.claim_error_samples) < CLAIM_ERROR_SAMPLE_LIMIT:
            self.claim_error_samples.append(
                {"category": category, "elapsed_seconds": round(elapsed, 6)}
            )

    def finished(self, outcome: Outcome) -> None:
        self.latencies.append(outcome.elapsed)
        self.by_operation.setdefault(outcome.operation, []).append(outcome.elapsed)
        for category in outcome.retried:
            self.result_retries += 1
            transport.count(self.result_retry_counts, category)
        if outcome.failure is None:
            self.completed_ids.add(outcome.command_id)
        else:
            self.result_errors += 1
            transport.count(self.result_error_counts, outcome.failure)


def run_claim_loop(
    state: SimulationState,
    *,
    workers: int,
    expected: int,
    max_seconds: float,
    idle_exit_seconds: float,
) -> ClaimTally:
    """Claim into free worker slots; every command reports on its own clock.

    Each claimed command runs on its own worker and posts its result when its
    own simulated duration elapses, and the slot it frees is claimed again at
    once. The loop used to wait for its whole claimed batch, so one 120 s
    RESTART_NODE parked the cluster's other workflows behind it and the
    integrated workflow percentiles moved in harness-made steps (batch 80).
    In flight never exceeds ``workers``: a claim asks for the free slots only
    (and the old cap of 25). A claim or a result that fails without a verdict
    is repeated on the product executor's policy (``benchmark_transport_retry``):
    one dropped connection costs a pause, not the Pod.
    """

    tally = ClaimTally()
    in_flight: dict[Future[Outcome], str] = {}
    started = time.monotonic()
    last_activity = started
    next_claim_at = started

    def settle(timeout: float) -> None:
        """Record every finished command, waiting up to ``timeout`` for one."""

        nonlocal last_activity
        if not in_flight:
            time.sleep(timeout)
            return
        done, _ = wait(list(in_flight), timeout=timeout, return_when=FIRST_COMPLETED)
        for future in done:
            del in_flight[future]
            tally.finished(future.result())
            last_activity = time.monotonic()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        while (
            expected <= 0 or len(tally.completed_ids) < expected
        ) and time.monotonic() - started < max_seconds:
            free = workers - len(in_flight)
            pause = next_claim_at - time.monotonic()
            if free <= 0 or pause > 0:
                # A full pool waits for a slot, a failed claim for its retry
                # pause; finished commands are recorded meanwhile either way.
                settle(
                    BUSY_WAIT_SECONDS if free <= 0 else min(BUSY_WAIT_SECONDS, pause)
                )
                continue
            try:
                claim = state.client.post(
                    "/v1/regional/executors/claim",
                    claim_payload(state.executor_id, min(25, free)),
                )
                commands = claim.get("commands") or []
            except Exception as exc:  # noqa: BLE001 - classified right here
                elapsed = time.monotonic() - started
                retry_in = tally.claims.failed(exc, elapsed=elapsed)
                if retry_in is None:
                    # Refused, or failing for a whole lease: claiming stops,
                    # the commands in flight still report, and the Pod fails.
                    tally.claim_failed(exc, elapsed=elapsed)
                    break
                next_claim_at = time.monotonic() + retry_in
                continue
            tally.claims.succeeded()
            if not commands:
                if (
                    expected <= 0
                    and tally.completed_ids
                    and not in_flight
                    and idle_exit_seconds > 0
                    and time.monotonic() - last_activity >= idle_exit_seconds
                ):
                    break
                settle(IDLE_POLL_SECONDS)
                continue
            last_activity = time.monotonic()
            for command in commands:
                command_id = command["command_id"]
                if (
                    command_id in tally.completed_ids
                    or command_id in in_flight.values()
                ):
                    tally.duplicate_claims += 1
                in_flight[pool.submit(state.execute, command)] = command_id
            settle(0)
        # A command claimed before the wall-clock limit still reports, exactly
        # as the last batch of the old loop did.
        while in_flight:
            settle(BUSY_WAIT_SECONDS)
    tally.wall_seconds = time.monotonic() - started
    return tally


def fatal_conditions(
    output: dict,
    *,
    min_concurrent_commands: int,
    inject_renew_failure: bool,
) -> list[str]:
    """Why the executor Pod fails: a lost result, a refused or persistently
    failing claim, a protocol violation or an unmet run condition. A retried
    claim or result is not among them; the summary counts it."""

    expected = int(output["expected_commands"])
    completed = int(output["completed_commands"])
    fatal = []
    if expected > 0 and completed != expected:
        fatal.append(f"completed {completed} of {expected} commands")
    if expected <= 0 and not completed:
        fatal.append("completed no command")
    if output["duplicate_claims"]:
        fatal.append("duplicate claims")
    if output["claim_errors"]:
        fatal.append("a claim was refused or kept failing for a whole lease")
    if output["result_errors"]:
        fatal.append("results lost: undelivered after their attempts or refused")
    if output["max_concurrent_commands"] < min_concurrent_commands:
        fatal.append("concurrency stayed below ACTION_MIN_CONCURRENT_COMMANDS")
    if output["long_commands"] and not output["renewals"]:
        fatal.append("commands longer than their lease were never renewed")
    if inject_renew_failure and (
        output["injected_renewal_failures"] != 1 or output["renewal_errors"] < 1
    ):
        fatal.append("the injected renewal failure was not observed")
    return fatal


def main() -> None:
    registrations = json.loads(Path(os.environ["CLUSTERS_FILE"]).read_text())
    offset = int(os.environ["CLUSTER_OFFSET"])
    registration = registrations[offset]
    cluster_id = registration["cluster_id"]
    expected = int(os.environ["EXPECTED_COMMANDS"])
    workers = int(os.getenv("ACTION_EXECUTOR_WORKERS", "8"))
    min_concurrent_commands = int(
        os.getenv(
            "ACTION_MIN_CONCURRENT_COMMANDS",
            str(min(workers, expected or 1)),
        )
    )
    if min_concurrent_commands < 1 or min_concurrent_commands > workers:
        raise ValueError(
            "ACTION_MIN_CONCURRENT_COMMANDS must be within 1..ACTION_EXECUTOR_WORKERS"
        )
    delay_scale = float(os.getenv("ACTION_DELAY_SCALE", "1"))
    lease_seconds = int(os.getenv("ACTION_LEASE_SECONDS", "120"))
    renewal_interval = float(
        os.getenv(
            "ACTION_RENEWAL_INTERVAL_SECONDS",
            str(max(1.0, min(30.0, lease_seconds / 3))),
        )
    )
    inject_renew_failure = (
        os.getenv("ACTION_INJECT_RENEW_FAILURE_ONCE", "false").lower() == "true"
    )
    max_seconds = float(os.getenv("ACTION_MAX_SECONDS", "900"))
    idle_exit_seconds = float(os.getenv("ACTION_IDLE_EXIT_SECONDS", "0"))
    executor_id = f"action-executor-{offset:03d}"
    client = Client(
        base_url=os.environ["GPU_FAULT_CONTROL_PLANE_URL"],
        cluster_id=cluster_id,
        token=registration["token"],
        ca_file=os.environ["SSL_CERT_FILE"],
    )
    state = SimulationState(
        client=client,
        executor_id=executor_id,
        operation_delays=operation_delays(),
        delay_scale=delay_scale,
        lease_seconds=lease_seconds,
        renewal_interval=renewal_interval,
        inject_renew_failure_pending=inject_renew_failure,
    )
    tally = run_claim_loop(
        state,
        workers=workers,
        expected=expected,
        max_seconds=max_seconds,
        idle_exit_seconds=idle_exit_seconds,
    )
    completed_ids = tally.completed_ids
    latencies = tally.latencies
    output = {
        "cluster_id": cluster_id,
        "executor_id": executor_id,
        "expected_commands": expected,
        "idle_exit_seconds": idle_exit_seconds,
        "completion_mode": "per-command",
        "completed_commands": len(completed_ids),
        "duplicate_claims": tally.duplicate_claims,
        "claim_errors": tally.claim_errors,
        "claim_error_counts": dict(sorted(tally.claim_error_counts.items())),
        "claim_error_samples": tally.claim_error_samples,
        "claim_retries": tally.claims.retries,
        "claim_retry_counts": dict(sorted(tally.claims.retry_counts.items())),
        "claim_retry_samples": tally.claims.retry_samples,
        "result_errors": tally.result_errors,
        "result_error_counts": dict(sorted(tally.result_error_counts.items())),
        "result_retries": tally.result_retries,
        "result_retry_counts": dict(sorted(tally.result_retry_counts.items())),
        "lease_seconds": lease_seconds,
        "renewal_interval_seconds": renewal_interval,
        "renewals": state.renewals,
        "renewal_errors": state.renewal_errors,
        "injected_renewal_failures": state.injected_renewal_failures,
        "long_commands": state.long_commands,
        "max_concurrent_commands": state.max_active_commands,
        "min_concurrent_commands": min_concurrent_commands,
        "wall_seconds": tally.wall_seconds,
        "command_p50_ms": statistics.median(latencies) * 1000 if latencies else 0.0,
        "command_p95_ms": percentile(latencies, 0.95) * 1000,
        "command_p99_ms": percentile(latencies, 0.99) * 1000,
        "operations": {
            operation: {
                "count": len(values),
                "p95_ms": percentile(values, 0.95) * 1000,
            }
            for operation, values in sorted(tally.by_operation.items())
        },
    }
    output["fatal"] = fatal_conditions(
        output,
        min_concurrent_commands=min_concurrent_commands,
        inject_renew_failure=inject_renew_failure,
    )
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    if output["fatal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
