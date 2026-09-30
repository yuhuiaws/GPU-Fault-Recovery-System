"""Bounded CAP004 proof through the production Executor and HTTP client.

Only the adapter's action is synthetic: one durable SQLite ledger insertion.
No environment-driven adapter factory, Node Agent or provider client is used.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request

from gpu_fault.adapters.node_action.lease_guard import (
    active_lease_guard,
    lease_hold_reason,
)
from gpu_fault.cluster_executor.executor import ClusterActionExecutor
from gpu_fault.cluster_executor.regional_client import (
    ClusterExecutorError,
)
from gpu_fault.execution.models import WorkflowStepContext, WorkflowStepOutcome
from gpu_fault.models import WorkflowStepSpec
from gpu_fault.regional import (
    RemoteActionCommand,
    RemoteCommandResult,
    RemoteCommandStatus,
)
from scripts.e2e.regional.capacity_wire import (
    PAYLOAD_BUDGET_BYTES,
    CapacityWireClient,
    CapacityWireError as Cap004Error,
    CapacityThreadsRunning as Cap004ThreadsRunning,
    ExecutorPins,
)
from scripts.e2e.regional.capacity_queued_lease import run_queued_lease_proof
from scripts.e2e.regional.probes.cap004_commands import (
    CLUSTER_ID,
    COMMAND_COUNT,
    OWNER,
    commands_for_run,
    require_owned,
)

EXECUTOR_LEASE_SECONDS = 10
EXECUTOR_BATCH_SIZE = 5
MAX_CONCURRENCY = 5


class Cap004WireClient(CapacityWireClient):
    """Keep the production wire protocol, scoped to the isolated API."""

    def __init__(
        self, url: str, token: str, *, executor_pins: ExecutorPins | None = None
    ) -> None:
        super().__init__(url, token, cluster_id=CLUSTER_ID, executor_pins=executor_pins)


def measure_bulk_api(
    url: str,
    token: str,
    run_id: str,
    *,
    executor_pins: ExecutorPins | None = None,
) -> dict[str, Any]:
    """Measure 25-command API capacity without executing a queued batch of 25."""
    client = Cap004WireClient(url, token, executor_pins=executor_pins)
    started = time.monotonic()
    claimed = client.claim(
        f"{run_id}-api-measurement",
        execution_owners=[OWNER],
        max_commands=COMMAND_COUNT,
        lease_seconds=60,
    )
    latency_ms = (time.monotonic() - started) * 1000
    expected = {item.command_id: item for item in commands_for_run(run_id)}
    if len(claimed) != COMMAND_COUNT or {item.command_id for item in claimed} != set(
        expected
    ):
        raise Cap004Error("CAP004 bulk API claim did not return exactly 25 commands")
    for item in claimed:
        require_owned(item, expected[item.command_id])
    competitor = Cap004WireClient(url, token, executor_pins=executor_pins)
    if competitor.claim(
        f"{run_id}-competitor",
        execution_owners=[OWNER],
        max_commands=COMMAND_COUNT,
        lease_seconds=60,
    ):
        raise Cap004Error("CAP004 bulk API duplicated an active lease")
    for item in claimed:
        saved = client.complete(
            item,
            RemoteCommandResult(
                lease_token=str(item.lease_token),
                status=RemoteCommandStatus.WAITING,
                status_source="cap004-api-measurement",
                details={"nonphysical": True, "api_measurement_only": True},
            ),
        )
        if (
            saved.command_id != item.command_id
            or saved.status is not RemoteCommandStatus.WAITING
            or any((saved.lease_token, saved.lease_owner, saved.lease_expires_at))
        ):
            raise Cap004Error("CAP004 bulk API handback did not release the command")
    return {
        "scope": "API payload and lease exclusion only; no adapter execution",
        "max_commands": COMMAND_COUNT,
        "claim_lease_seconds": 60,
        "commands_claimed": len(claimed),
        "claim_response_bytes": client.claim_response_bytes,
        "claim_payload_budget_bytes": PAYLOAD_BUDGET_BYTES,
        "claim_latency_ms": latency_ms,
        "waiting_handbacks": len(claimed),
        "competitor_samples": 1,
        "competitor_duplicate_claims": 0,
    }


class Cap004Client(Cap004WireClient):
    """Observe actual five-command execution batches and inject bounded failures."""

    def __init__(
        self,
        url: str,
        token: str,
        run_id: str,
        *,
        executor_pins: ExecutorPins | None = None,
    ) -> None:
        super().__init__(url, token, executor_pins=executor_pins)
        self.expected = {item.command_id: item for item in commands_for_run(run_id)}
        self.failure_command_id = list(self.expected)[-1]
        self.lock = threading.Lock()
        self.first_claimed = threading.Event()
        self.stop_executor: Callable[[str], None] = lambda _reason: None
        self.claim_attempts: list[float] = []
        self.claimed: Counter[str] = Counter()
        self.claim_latency_ms = 0.0
        self.claim_retry_injected = False
        self.renewal_rejection_injected = False
        self.injected_renewal_status: int | None = None
        self.progressions: dict[str, dict[str, Any]] = {}
        self.original_lease_seconds: dict[str, float] = {}
        self.completed: Counter[str] = Counter()
        self.completed_leases: dict[str, str] = {}
        self.rejected_renewals: Counter[tuple[str, str]] = Counter()
        self.problems: list[str] = []

    def _send(self, request: Request, *, timeout_seconds: float | None = None) -> bytes:
        if request.full_url.endswith("/claim"):
            self.claim_attempts.append(time.monotonic())
            if not self.claim_retry_injected:
                self.claim_retry_injected = True
                raise ClusterExecutorError("CAP004 controlled claim transport failure")
        try:
            return super()._send(request, timeout_seconds=timeout_seconds)
        except Cap004Error:
            self.problems.append("CAP004 claim exceeded the payload budget")
            self.stop_executor("CAP004 payload budget")
            raise

    def claim(
        self,
        executor_id: str,
        *,
        execution_owners: list[str] | None = None,
        max_commands: int,
        lease_seconds: int,
        wait_seconds: float = 0,
    ) -> list[RemoteActionCommand]:
        started = time.monotonic()
        commands = super().claim(
            executor_id,
            execution_owners=execution_owners,
            max_commands=max_commands,
            lease_seconds=lease_seconds,
            wait_seconds=wait_seconds,
        )
        first = not self.first_claimed.is_set()
        ids = [item.command_id for item in commands]
        if (
            len(ids) != len(set(ids))
            or len(ids) > EXECUTOR_BATCH_SIZE
            or (first and len(ids) != EXECUTOR_BATCH_SIZE)
            or any(
                identity not in self.expected
                or (
                    self.claimed[identity] > 0
                    and (
                        identity != self.failure_command_id
                        or self.claimed[identity] != 1
                    )
                )
                for identity in ids
            )
        ):
            self.problems.append("claim did not contain the exact expected command set")
            self.stop_executor("CAP004 unexpected claim")
            raise Cap004Error(self.problems[-1])
        for item in commands:
            require_owned(item, self.expected[item.command_id])
            if (
                item.status is not RemoteCommandStatus.LEASED
                or not item.lease_token
                or item.lease_owner != executor_id
                or item.lease_expires_at is None
            ):
                raise Cap004Error("CAP004 claim has no valid lease")
            self.claimed[item.command_id] += 1
            if item.command_id not in self.progressions:
                original_lease = (
                    item.lease_expires_at - item.updated_at
                ).total_seconds()
                if original_lease <= 0:
                    raise Cap004Error("CAP004 initial lease duration is invalid")
                self.original_lease_seconds[item.command_id] = original_lease
                self.progressions[item.command_id] = {
                    "command_id": item.command_id,
                    "initial": item.lease_expires_at.isoformat(),
                    "latest": item.lease_expires_at.isoformat(),
                    "renewal_count": 0,
                }
        if first:
            self.claim_latency_ms = (time.monotonic() - started) * 1000
            self.first_claimed.set()
        return commands

    def renew(
        self, command: RemoteActionCommand, executor_id: str, lease_seconds: int
    ) -> RemoteActionCommand:
        with self.lock:
            inject = (
                command.command_id == self.failure_command_id
                and self.progressions[command.command_id]["renewal_count"] > 0
                and not self.renewal_rejection_injected
            )
            if inject:
                self.renewal_rejection_injected = True
        try:
            renewed = super().renew(
                command,
                executor_id + "-rejected" if inject else executor_id,
                lease_seconds,
            )
        except ClusterExecutorError as exc:
            with self.lock:
                if inject:
                    self.injected_renewal_status = exc.status_code
                elif exc.status_code == 409 and executor_id == command.lease_owner:
                    digest = hashlib.sha256(
                        str(command.lease_token).encode()
                    ).hexdigest()
                    self.rejected_renewals[(command.command_id, digest)] += 1
                else:
                    self.problems.append("unexpected renewal failure")
            raise
        with self.lock:
            row = self.progressions[command.command_id]
            if (
                inject
                or renewed.command_id != command.command_id
                or renewed.lease_token != command.lease_token
                or renewed.lease_owner != executor_id
                or renewed.lease_expires_at is None
                or renewed.lease_expires_at <= datetime.fromisoformat(row["latest"])
            ):
                self.problems.append("renewal identity or lease progression is invalid")
                raise Cap004Error(self.problems[-1])
            row["latest"] = renewed.lease_expires_at.isoformat()
            row["renewal_count"] += 1
        return renewed

    def renewed_once(self, command_id: str) -> bool:
        with self.lock:
            return bool(self.progressions[command_id]["renewal_count"])

    def renewal_evidence(self) -> tuple[dict[str, int], int]:
        # A committed result's ACK may arrive after an in-flight renewal's 409.
        with self.lock:
            terminal: Counter[str] = Counter()
            unconfirmed = 0
            for (command_id, digest), count in self.rejected_renewals.items():
                if self.completed_leases.get(command_id) == digest:
                    terminal[command_id] += count
                else:
                    unconfirmed += count
            return dict(terminal), unconfirmed

    def complete(
        self, command: RemoteActionCommand, result: RemoteCommandResult
    ) -> RemoteActionCommand:
        with self.lock:
            if (
                command.command_id == self.failure_command_id
                and self.claimed[command.command_id] == 1
            ):
                self.problems.append("result sent despite the injected lost lease")
        saved = super().complete(command, result)
        with self.lock:
            if (
                saved.command_id != command.command_id
                or saved.status is not RemoteCommandStatus.SUCCEEDED
                or saved.lease_token is not None
                or saved.lease_owner is not None
                or saved.lease_expires_at is not None
            ):
                self.problems.append("result did not close the owned command")
            else:
                self.completed_leases[command.command_id] = hashlib.sha256(
                    str(command.lease_token).encode()
                ).hexdigest()
            self.completed[command.command_id] += 1
            finished = set(self.completed) == set(self.expected)
        if finished:
            self.stop_executor("CAP004 exact command set completed")
        return saved


class NonphysicalLedgerAdapter:
    owner = OWNER

    def __init__(
        self,
        client: Cap004Client,
        path: Path,
        stop: threading.Event,
        *,
        minimum_hold_seconds: float = 12,
        action_timeout_seconds: float = 75,
    ) -> None:
        self.client = client
        self.stop = stop
        self.minimum_hold_seconds = minimum_hold_seconds
        self.action_timeout_seconds = action_timeout_seconds
        self.expected = {
            item.idempotency_key: item for item in client.expected.values()
        }
        path.touch(mode=0o600, exist_ok=False)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute(
            "CREATE TABLE ledger (key TEXT PRIMARY KEY, mutation_count INTEGER "
            "NOT NULL CHECK(mutation_count = 1), calls INTEGER NOT NULL)"
        )
        self.db.commit()
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.durations: list[float] = []
        self.timings: list[dict[str, Any]] = []
        self.hold_observed = False

    def supports(self, step: WorkflowStepSpec) -> bool:
        return any(step == item.step for item in self.expected.values())

    def execute(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        expected = self.expected.get(context.idempotency_key)
        if (
            expected is None
            or context.step != expected.step
            or context.workflow.request_id != expected.workflow_request_id
            or context.incident.incident_id != expected.incident_id
            or active_lease_guard.get() is None
            or lease_hold_reason() is not None
        ):
            raise Cap004Error("CAP004 ledger requires an owned command and live guard")
        started = time.monotonic()
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        cached = False
        succeeded = False
        try:
            with self.lock, self.db:
                cached = (
                    self.db.execute(
                        "SELECT 1 FROM ledger WHERE key=?", (context.idempotency_key,)
                    ).fetchone()
                    is not None
                )
                self.db.execute(
                    "INSERT INTO ledger VALUES (?,1,1) ON CONFLICT(key) "
                    "DO UPDATE SET calls=calls+1",
                    (context.idempotency_key,),
                )
            if not cached:
                self.wait_for_lease_evidence(expected.command_id, started)
            succeeded = True
            return WorkflowStepOutcome.succeeded(
                details={"nonphysical": True, "ledger_count": 1, "cached": cached}
            )
        finally:
            with self.lock:
                self.active -= 1
                elapsed = time.monotonic() - started
                self.timings.append(
                    {
                        "command_id": expected.command_id,
                        "elapsed_seconds": elapsed,
                        "original_lease_seconds": self.client.original_lease_seconds[
                            expected.command_id
                        ],
                        "cached": cached,
                        "adapter_succeeded": succeeded,
                        "lease_guard_held": lease_hold_reason() is not None,
                    }
                )
                if not cached:
                    self.durations.append(elapsed)

    def wait_for_lease_evidence(self, command_id: str, started: float) -> None:
        while not self.stop.wait(0.01):
            elapsed = time.monotonic() - started
            hold = lease_hold_reason()
            if command_id == self.client.failure_command_id:
                if hold is not None:
                    self.hold_observed = True
                    return
            elif hold is not None:
                raise Cap004Error("CAP004 healthy command lost its lease")
            elif (
                elapsed >= self.minimum_hold_seconds
                and elapsed >= self.client.original_lease_seconds[command_id] * 1.2
                and self.client.renewed_once(command_id)
            ):
                return
            if elapsed >= self.action_timeout_seconds:
                raise Cap004Error("CAP004 lease evidence deadline exceeded")
        raise Cap004Error("CAP004 action stopped")

    def snapshot(self) -> list[dict[str, Any]]:
        with self.lock:
            return [
                {"idempotency_key": key, "mutation_count": count, "calls": calls}
                for key, count, calls in self.db.execute(
                    "SELECT key,mutation_count,calls FROM ledger ORDER BY key"
                )
            ]

    def close(self) -> None:
        self.db.close()


def run_executor_proof(
    url: str,
    token: str,
    run_id: str,
    state_dir: Path,
    *,
    timeout_seconds: float = 600,
    poll_seconds: float = 1,
    minimum_hold_seconds: float = 12,
    action_timeout_seconds: float = 75,
    queue_timeout_seconds: float = 75,
    executor_pins: ExecutorPins | None = None,
) -> dict[str, Any]:
    bulk_api = measure_bulk_api(url, token, run_id, executor_pins=executor_pins)
    queued_lease = run_queued_lease_proof(
        url,
        token,
        run_id,
        state_dir,
        timeout_seconds=queue_timeout_seconds,
        executor_pins=executor_pins,
    )
    client = Cap004Client(url, token, run_id, executor_pins=executor_pins)
    stop = threading.Event()
    adapter = NonphysicalLedgerAdapter(
        client,
        state_dir / "nonphysical-ledger.sqlite",
        stop,
        minimum_hold_seconds=minimum_hold_seconds,
        action_timeout_seconds=action_timeout_seconds,
    )
    executor = ClusterActionExecutor(
        client,
        [adapter],
        executor_id=f"{run_id}-executor",
        allowed_namespaces=set(),
        batch_size=EXECUTOR_BATCH_SIZE,
        max_concurrent_commands=MAX_CONCURRENCY,
        lease_seconds=EXECUTOR_LEASE_SECONDS,
        poll_seconds=poll_seconds,
        lease_renewal_failure_limit=1,
        max_execution_seconds=action_timeout_seconds + 5,
        claim_state_path=str(state_dir / "claim-state.json"),
        liveness_state_path=str(state_dir / "loop-alive.json"),
    )
    client.stop_executor = executor.request_stop

    def deadline() -> None:
        if not stop.wait(timeout_seconds):
            client.problems.append("CAP004 overall deadline exceeded")
            stop.set()
            executor.request_stop("CAP004 deadline")

    before = set(threading.enumerate())
    watchers = [
        threading.Thread(target=deadline, name=f"{run_id}-deadline"),
    ]
    started = time.monotonic()
    try:
        for thread in watchers:
            thread.start()
        executor.run()
    finally:
        stop.set()
        executor.request_stop("CAP004 shutdown")
        # The real lifecycle owns its workers. Join only new Executor threads,
        # never threads belonging to another test or unrelated application.
        owned = set(watchers) | {
            thread
            for thread in threading.enumerate()
            if thread not in before
            and thread.name.startswith(
                (
                    "gpu-fault-command",
                    "cmd-command-",
                    "lease-command-",
                    "hold-command-",
                )
            )
        }
        join_deadline = time.monotonic() + action_timeout_seconds + 10
        join_error: BaseException | None = None
        for thread in owned:
            if thread.ident is not None:
                try:
                    thread.join(max(0, join_deadline - time.monotonic()))
                except BaseException as exc:
                    join_error = exc
        if join_error is not None or any(thread.is_alive() for thread in owned):
            raise Cap004ThreadsRunning(
                "CAP004 executor shutdown could not be verified"
            ) from join_error
        ledger = adapter.snapshot()
        adapter.close()
    final_claim = Cap004WireClient(url, token, executor_pins=executor_pins).claim(
        f"{run_id}-final",
        execution_owners=[OWNER],
        max_commands=COMMAND_COUNT,
        lease_seconds=10,
    )
    counters = executor.metrics_snapshot()
    problems = list(client.problems)
    expected_calls = Counter({key: 1 for key in client.expected})
    expected_calls[client.failure_command_id] = 2
    expected_ledger_calls = {
        item.idempotency_key: expected_calls[item.command_id]
        for item in client.expected.values()
    }
    successful_long_actions = [
        row
        for row in adapter.timings
        if row["cached"] is False
        and row["adapter_succeeded"] is True
        and row["lease_guard_held"] is False
        and row["elapsed_seconds"] > row["original_lease_seconds"]
        and client.completed[row["command_id"]] == 1
        and client.claimed[row["command_id"]] == 1
    ]
    terminal_rejections, unconfirmed_rejections = client.renewal_evidence()
    expected_renewal_failures = 1 + sum(terminal_rejections.values())
    checks = {
        "primary_and_recovery_claims": client.claimed == expected_calls,
        "concurrency": adapter.max_active == MAX_CONCURRENCY and adapter.active == 0,
        "successful_actions_outlive_initial_lease": (
            len(successful_long_actions) >= 5
            and len({row["command_id"] for row in successful_long_actions}) >= 5
        ),
        "claim_retry": len(client.claim_attempts) >= 2
        and (client.claim_attempts[1] - client.claim_attempts[0] >= poll_seconds),
        "lease_loss": (
            client.injected_renewal_status == 409
            and adapter.hold_observed
            and counters["lease_renewal_failures"] == expected_renewal_failures
            and counters["lease_lost_total"] == expected_renewal_failures
            and counters["results_withheld_total"] == 1
        ),
        "terminal_renewal_rejections": all(
            count == 1 and client.completed[command_id] == 1
            for command_id, count in terminal_rejections.items()
        )
        and unconfirmed_rejections == 0,
        "ledger_exactly_once": (
            {row["idempotency_key"]: row["calls"] for row in ledger}
            == expected_ledger_calls
            and all(row["mutation_count"] == 1 for row in ledger)
        ),
        "reports": client.completed == Counter({key: 1 for key in client.expected}),
        "terminal_claim": not final_claim,
        "executor_health": all(
            counters[name] == 0
            for name in (
                "reported_failures",
                "unexpected_failures",
                "execution_timeouts_total",
                "stuck_executions",
                "abandoned_lease_holds_total",
            )
        ),
    }
    problems.extend(name for name, passed in checks.items() if not passed)
    return {
        "execution_path": "production-ClusterActionExecutor/nonphysical-ledger",
        "bulk_api": bulk_api,
        "queued_lease": queued_lease,
        "commands_claimed": len(client.expected),
        "executor_batch_size": executor.batch_size,
        "executor_claim_latency_ms": client.claim_latency_ms,
        "max_concurrent_commands": adapter.max_active,
        "lease_seconds": EXECUTOR_LEASE_SECONDS,
        "long_commands": len(successful_long_actions),
        "long_action_requirement": "at least 5 successful uncached actions: elapsed > original lease",
        "action_timings": adapter.timings,
        "command_durations_seconds": adapter.durations,
        "batch_wall_seconds": time.monotonic() - started,
        "lease_progressions": list(client.progressions.values()),
        "injected_renewal_failure_status": client.injected_renewal_status,
        "terminal_renewal_rejections": terminal_rejections,
        "unconfirmed_renewal_rejections": unconfirmed_rejections,
        "executor_counters": counters,
        "claim_retry_backoff_verified": checks["claim_retry"],
        "ledger": ledger,
        "ledger_exactly_once": checks["ledger_exactly_once"],
        "final_open_claims": len(final_claim),
        "background_threads_stopped": True,
        "http_connections_closed": True,
        "adapter_types": [type(item).__name__ for item in executor.adapters],
        "limits": [
            "25-command claims measure the API only and are handed back WAITING.",
            "The separate 2/1 phase proves expiry and competing reclamation before "
            "a queued adapter starts; it does not execute 25 queued long actions.",
            "Competitor exclusion is observed during the separate bulk API measurement.",
        ],
        "problems": problems,
    }
