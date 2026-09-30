"""CAP002 retries through the production Executor loop, without action adapters."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from gpu_fault.cluster_executor import ClusterActionExecutor, ClusterExecutorError
from scripts.e2e.regional.capacity_wire import CapacityWireClient, ExecutorPins
from scripts.e2e.regional.probes.cap004_commands import CLUSTER_ID


class RetryingClaimClient(CapacityWireClient):
    def __init__(
        self,
        url: str,
        token: str,
        release_hold: Callable[[], None],
        *,
        executor_pins: ExecutorPins | None = None,
    ) -> None:
        super().__init__(url, token, cluster_id=CLUSTER_ID, executor_pins=executor_pins)
        self.release_hold = release_hold
        self.stop: Callable[[str], None] = lambda _reason: None
        self.attempts: list[dict[str, Any]] = []

    def claim(self, *args: Any, **kwargs: Any):
        if len(self.attempts) >= 5:
            self.stop("CAP002 retry attempt limit")
            raise ClusterExecutorError("CAP002 retry attempt limit")
        entry: dict[str, Any] = {"started": time.monotonic()}
        try:
            commands = super().claim(*args, **kwargs)
        except ClusterExecutorError as exc:
            entry["status"] = exc.status_code
            if exc.status_code == 503 and not self.attempts:
                self.release_hold()
            elif exc.status_code != 503:
                self.stop("CAP002 unexpected claim response")
            raise
        else:
            entry["status"] = 200
            entry["commands"] = len(commands)
            self.stop("CAP002 claim recovered")
            if commands:
                raise ClusterExecutorError("CAP002 empty database returned commands")
            return commands
        finally:
            entry["finished"] = time.monotonic()
            self.attempts.append(entry)


def run_claim_retry_proof(
    url: str,
    token: str,
    state_dir: Path,
    release_hold: Callable[[], None],
    *,
    poll_seconds: float = 2,
    executor_pins: ExecutorPins | None = None,
) -> dict[str, Any]:
    client = RetryingClaimClient(url, token, release_hold, executor_pins=executor_pins)
    executor = ClusterActionExecutor(
        client,
        [],
        executor_id="cap002-real-executor",
        allowed_namespaces=set(),
        batch_size=1,
        max_concurrent_commands=1,
        poll_seconds=poll_seconds,
        claim_backoff_max_seconds=max(poll_seconds, 8),
        claim_state_path=str(state_dir / "retry-claim-state.json"),
        liveness_state_path=str(state_dir / "retry-loop-state.json"),
    )
    client.stop = executor.request_stop
    try:
        executor.run()
    finally:
        executor.request_stop("CAP002 retry proof finished")
    attempts = client.attempts
    delays = [
        later["started"] - earlier["finished"]
        for earlier, later in zip(attempts, attempts[1:])
    ]
    passed = (
        len(attempts) >= 2
        and all(item["status"] == 503 for item in attempts[:-1])
        and attempts[-1]["status"] == 200
        and attempts[-1].get("commands") == 0
        and all(
            delay >= min(8, poll_seconds * 2**index)
            for index, delay in enumerate(delays)
        )
        and executor.claimed_total == 0
        and executor.unexpected_failures == 0
    )
    return {
        "passed": passed,
        "execution_path": "production-ClusterActionExecutor.run/claim-backoff",
        "attempts": attempts,
        "retry_delays_seconds": delays,
        "poll_seconds": poll_seconds,
        "adapter_count": len(executor.adapters),
        "commands_executed": executor.claimed_total,
    }
