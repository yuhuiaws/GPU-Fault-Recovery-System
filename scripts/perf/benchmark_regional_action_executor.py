from __future__ import annotations

import json
import os
import ssl
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Event, Lock, Thread
from urllib import error, request


OWNERS = [
    "gpu-fault-kubernetes-adapter",
    "gpu-fault-node-agent",
    "gpu-fault-hyperpod-adapter",
]
DELAYS = {
    "MARK_UNSCHEDULABLE": 0.03,
    "CHECKPOINT_WORKLOADS": 0.08,
    "STOP_WORKLOADS": 0.08,
    "COLLECT_DIAGNOSTIC_BUNDLE": 0.15,
    "QUIESCE_GPU_SERVICES": 0.12,
    "VERIFY_NO_GPU_CLIENTS": 0.08,
    "RESET_GPU": 0.15,
    "RESTORE_GPU_SERVICES": 0.10,
    "RESTART_NODE": 0.20,
    "RESTORE_SCHEDULING": 0.04,
}


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


def main() -> None:
    registrations = json.loads(Path(os.environ["CLUSTERS_FILE"]).read_text())
    offset = int(os.environ["CLUSTER_OFFSET"])
    registration = registrations[offset]
    cluster_id = registration["cluster_id"]
    expected = int(os.environ["EXPECTED_COMMANDS"])
    workers = int(os.getenv("ACTION_EXECUTOR_WORKERS", "8"))
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
    result_errors = 0
    renewals = 0
    renewal_errors = 0
    injected_renewal_failures = 0
    long_commands = 0
    active_commands = 0
    max_active_commands = 0
    inject_renew_failure_pending = inject_renew_failure
    counters_lock = Lock()
    latencies: list[float] = []
    by_operation: dict[str, list[float]] = {}
    started = time.monotonic()

    def execute(command: dict) -> tuple[str, str, float, str | None]:
        nonlocal active_commands
        nonlocal inject_renew_failure_pending
        nonlocal injected_renewal_failures
        nonlocal long_commands
        nonlocal max_active_commands
        nonlocal renewal_errors
        nonlocal renewals
        command_id = command["command_id"]
        operation = command["step"]["operation"]
        node_count = len(command["step"].get("node_ids") or [])
        delay = (DELAYS.get(operation, 0.05) + 0.01 * max(1, node_count)) * delay_scale
        with counters_lock:
            active_commands += 1
            max_active_commands = max(max_active_commands, active_commands)
            if delay >= lease_seconds:
                long_commands += 1
        stop_renewal = Event()

        def renew() -> None:
            nonlocal inject_renew_failure_pending
            nonlocal injected_renewal_failures
            nonlocal renewal_errors
            nonlocal renewals
            while not stop_renewal.wait(renewal_interval):
                with counters_lock:
                    if inject_renew_failure_pending:
                        inject_renew_failure_pending = False
                        injected_renewal_failures += 1
                        renewal_errors += 1
                        continue
                try:
                    client.post(
                        f"/v1/regional/executors/{command_id}/renew",
                        {
                            "executor_id": executor_id,
                            "lease_token": command["lease_token"],
                            "lease_seconds": lease_seconds,
                        },
                    )
                except Exception:
                    with counters_lock:
                        renewal_errors += 1
                else:
                    with counters_lock:
                        renewals += 1

        renewal_thread = Thread(
            target=renew,
            name=f"renew-{command_id}",
            daemon=True,
        )
        renewal_thread.start()
        begin = time.monotonic()
        try:
            time.sleep(delay)
            client.post(
                f"/v1/regional/executors/{command_id}/result",
                {
                    "lease_token": command["lease_token"],
                    "status": "SUCCEEDED",
                    "status_source": "action-capacity-simulator",
                    "details": {
                        "simulated": True,
                        "operation": operation,
                        "node_count": node_count,
                        "executor_id": executor_id,
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
            renewal_thread.join(timeout=max(1.0, renewal_interval + 1))
            with counters_lock:
                active_commands -= 1
        return command_id, operation, time.monotonic() - begin, None

    while len(completed_ids) < expected and time.monotonic() - started < max_seconds:
        try:
            claim = client.post(
                "/v1/regional/executors/claim",
                claim_payload(executor_id, min(25, workers)),
            )
            commands = claim.get("commands") or []
        except (error.HTTPError, error.URLError, OSError):
            claim_errors += 1
            time.sleep(0.1)
            continue
        if not commands:
            time.sleep(0.05)
            continue
        for command in commands:
            if command["command_id"] in completed_ids:
                duplicate_claims += 1
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(execute, item) for item in commands]
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
        "completed_commands": len(completed_ids),
        "duplicate_claims": duplicate_claims,
        "claim_errors": claim_errors,
        "result_errors": result_errors,
        "lease_seconds": lease_seconds,
        "renewal_interval_seconds": renewal_interval,
        "renewals": renewals,
        "renewal_errors": renewal_errors,
        "injected_renewal_failures": injected_renewal_failures,
        "long_commands": long_commands,
        "max_concurrent_commands": max_active_commands,
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
        len(completed_ids) != expected
        or duplicate_claims
        or result_errors
        or max_active_commands < min(workers, expected)
        or (long_commands and renewals == 0)
        or (
            inject_renew_failure
            and (injected_renewal_failures != 1 or renewal_errors < 1)
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
