from __future__ import annotations

import json
import os
import ssl
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from urllib import error, request


OWNERS = [
    "gpu-fault-kubernetes-adapter",
    "gpu-fault-node-agent",
    "gpu-fault-hyperpod-adapter",
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
CLAIM_ERROR_SAMPLE_LIMIT = 20


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


def request_error_category(exc: BaseException) -> str:
    if isinstance(exc, error.HTTPError):
        return f"{type(exc).__name__}:{exc.code}"
    if isinstance(exc, error.URLError):
        return f"{type(exc).__name__}:{type(exc.reason).__name__}"
    return type(exc).__name__


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


@dataclass
class SimulationState:
    client: Client
    executor_id: str
    operation_delays: dict[str, float]
    delay_scale: float
    lease_seconds: int
    renewal_interval: float
    inject_renew_failure_pending: bool
    counters_lock: Lock = field(default_factory=Lock)
    renewals: int = 0
    renewal_errors: int = 0
    injected_renewal_failures: int = 0
    long_commands: int = 0
    active_commands: int = 0
    max_active_commands: int = 0

    def execute(self, command: dict) -> tuple[str, str, float, str | None]:
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
        try:
            time.sleep(delay)
            self.client.post(
                f"/v1/regional/executors/{command_id}/result",
                {
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
                },
            )
        except Exception as exc:
            return (
                command_id,
                operation,
                time.monotonic() - begin,
                type(exc).__name__,
            )
        finally:
            stop_renewal.set()
            renewal_thread.join(timeout=max(1.0, self.renewal_interval + 1))
            with self.counters_lock:
                self.active_commands -= 1
        return command_id, operation, time.monotonic() - begin, None


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
    completed_ids: set[str] = set()
    duplicate_claims = 0
    claim_errors = 0
    claim_error_counts: dict[str, int] = {}
    claim_error_samples: list[dict[str, str | float]] = []
    result_errors = 0
    state = SimulationState(
        client=client,
        executor_id=executor_id,
        operation_delays=operation_delays(),
        delay_scale=delay_scale,
        lease_seconds=lease_seconds,
        renewal_interval=renewal_interval,
        inject_renew_failure_pending=inject_renew_failure,
    )
    latencies: list[float] = []
    by_operation: dict[str, list[float]] = {}
    started = time.monotonic()
    last_activity = started

    while (
        expected <= 0 or len(completed_ids) < expected
    ) and time.monotonic() - started < max_seconds:
        try:
            claim = client.post(
                "/v1/regional/executors/claim",
                claim_payload(executor_id, min(25, workers)),
            )
            commands = claim.get("commands") or []
        except (error.HTTPError, error.URLError, OSError) as exc:
            claim_errors += 1
            category = request_error_category(exc)
            claim_error_counts[category] = claim_error_counts.get(category, 0) + 1
            if len(claim_error_samples) < CLAIM_ERROR_SAMPLE_LIMIT:
                claim_error_samples.append(
                    {
                        "category": category,
                        "elapsed_seconds": round(time.monotonic() - started, 6),
                    }
                )
            time.sleep(0.1)
            continue
        if not commands:
            if (
                expected <= 0
                and completed_ids
                and idle_exit_seconds > 0
                and time.monotonic() - last_activity >= idle_exit_seconds
            ):
                break
            time.sleep(0.05)
            continue
        last_activity = time.monotonic()
        for command in commands:
            if command["command_id"] in completed_ids:
                duplicate_claims += 1
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(state.execute, item) for item in commands]
            for future in as_completed(futures):
                command_id, operation, elapsed, failure = future.result()
                latencies.append(elapsed)
                by_operation.setdefault(operation, []).append(elapsed)
                if failure is None:
                    completed_ids.add(command_id)
                else:
                    result_errors += 1

    wall = time.monotonic() - started
    output = {
        "cluster_id": cluster_id,
        "executor_id": executor_id,
        "expected_commands": expected,
        "idle_exit_seconds": idle_exit_seconds,
        "completed_commands": len(completed_ids),
        "duplicate_claims": duplicate_claims,
        "claim_errors": claim_errors,
        "claim_error_counts": dict(sorted(claim_error_counts.items())),
        "claim_error_samples": claim_error_samples,
        "result_errors": result_errors,
        "lease_seconds": lease_seconds,
        "renewal_interval_seconds": renewal_interval,
        "renewals": state.renewals,
        "renewal_errors": state.renewal_errors,
        "injected_renewal_failures": state.injected_renewal_failures,
        "long_commands": state.long_commands,
        "max_concurrent_commands": state.max_active_commands,
        "min_concurrent_commands": min_concurrent_commands,
        "wall_seconds": wall,
        "command_p50_ms": statistics.median(latencies) * 1000 if latencies else 0.0,
        "command_p95_ms": percentile(latencies, 0.95) * 1000,
        "command_p99_ms": percentile(latencies, 0.99) * 1000,
        "operations": {
            operation: {
                "count": len(values),
                "p95_ms": percentile(values, 0.95) * 1000,
            }
            for operation, values in sorted(by_operation.items())
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    if (
        (expected > 0 and len(completed_ids) != expected)
        or (expected <= 0 and not completed_ids)
        or duplicate_claims
        or result_errors
        or state.max_active_commands < min_concurrent_commands
        or (state.long_commands and state.renewals == 0)
        or (
            inject_renew_failure
            and (state.injected_renewal_failures != 1 or state.renewal_errors < 1)
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
