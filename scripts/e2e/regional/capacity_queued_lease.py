"""CAP004's nondefault 2/1 queue proof; no adapter makes an external mutation."""

from __future__ import annotations

import hashlib
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gpu_fault.cluster_executor import ClusterActionExecutor, ClusterExecutorError
from gpu_fault.execution.models import WorkflowStepContext, WorkflowStepOutcome
from gpu_fault.models import WorkflowStepSpec
from gpu_fault.regional import (
    RemoteActionCommand,
    RemoteCommandResult,
    RemoteCommandStatus,
)
from scripts.e2e.regional.capacity_wire import (
    CapacityWireError as Cap004Error,
    CapacityThreadsRunning as Cap004ThreadsRunning,
    CapacityWireClient,
)
from scripts.e2e.regional.probes.cap004_commands import (
    CLUSTER_ID,
    COMMAND_COUNT,
    OWNER,
    commands_for_run,
    require_owned,
)


class QueueClient(CapacityWireClient):
    """Observe only the reserved pair and acknowledged same-lease handbacks."""

    def __init__(
        self,
        url: str,
        token: str,
        *,
        executor_id: str,
        selected: list[RemoteActionCommand],
    ) -> None:
        super().__init__(url, token, cluster_id=CLUSTER_ID)
        self.executor_id = executor_id
        self.expected = {item.command_id: item for item in selected}
        self.claimed: list[RemoteActionCommand] = []
        self.lock = threading.Lock()
        self.completed: Counter[str] = Counter()
        self.completed_leases: dict[str, str] = {}
        self.rejected_renewals: Counter[tuple[str, str]] = Counter()
        self.problems: list[str] = []

    def claim(
        self,
        executor_id: str,
        *,
        execution_owners: list[str] | None = None,
        max_commands: int,
        lease_seconds: int,
        wait_seconds: float = 0,
    ) -> list[RemoteActionCommand]:
        commands: list[RemoteActionCommand] = super().claim(
            executor_id,
            execution_owners=execution_owners,
            max_commands=max_commands,
            lease_seconds=lease_seconds,
            wait_seconds=wait_seconds,
        )
        if (
            executor_id != self.executor_id
            or len(commands) != 2
            or {item.command_id for item in commands} != set(self.expected)
        ):
            raise Cap004Error("CAP004 2/1 claim did not match the reserved pair")
        for item in commands:
            require_owned(item, self.expected[item.command_id])
            if (
                item.status is not RemoteCommandStatus.LEASED
                or not item.lease_token
                or item.lease_owner != executor_id
                or item.lease_expires_at is None
            ):
                raise Cap004Error("CAP004 queued claim has no valid lease")
        commands.sort(key=lambda item: item.command_id)
        self.claimed.extend(commands)
        return commands

    def renew(
        self, command: RemoteActionCommand, executor_id: str, lease_seconds: int
    ) -> RemoteActionCommand:
        try:
            return super().renew(command, executor_id, lease_seconds)
        except ClusterExecutorError as exc:
            with self.lock:
                if (
                    exc.status_code == 409
                    and executor_id == self.executor_id == command.lease_owner
                    and command.lease_token
                ):
                    digest = hashlib.sha256(command.lease_token.encode()).hexdigest()
                    self.rejected_renewals[(command.command_id, digest)] += 1
                else:
                    self.problems.append("unexpected queued renewal failure")
            raise

    def complete(
        self, command: RemoteActionCommand, result: RemoteCommandResult
    ) -> RemoteActionCommand:
        original = self.claimed[0] if self.claimed else None
        if (
            original is None
            or command.command_id != original.command_id
            or command.status is not RemoteCommandStatus.LEASED
            or command.lease_owner != self.executor_id
            or not original.lease_token
            or command.lease_token != original.lease_token
            or result.lease_token != original.lease_token
            or result.status is not RemoteCommandStatus.WAITING
            or result.error is not None
            or result.details != {"nonphysical": True, "queue_admission_only": True}
        ):
            raise Cap004Error("CAP004 queued handback lacks the original authority")
        require_owned(command, original)
        saved = super().complete(command, result)
        require_owned(saved, original)
        if (
            saved.status is not RemoteCommandStatus.WAITING
            or saved.lease_token is not None
            or saved.lease_owner is not None
            or saved.lease_expires_at is not None
            or saved.last_lease_owner != self.executor_id
            or saved.result_details != result.details
            or saved.error is not None
            or saved.status_source != result.status_source
        ):
            raise Cap004Error("CAP004 queued handback did not acknowledge WAITING")
        with self.lock:
            self.completed[command.command_id] += 1
            self.completed_leases[command.command_id] = hashlib.sha256(
                result.lease_token.encode()
            ).hexdigest()
        return saved

    def renewal_evidence(self) -> tuple[dict[str, int], int]:
        # WAITING releases the lease; its ACK may follow an in-flight renewal's 409.
        with self.lock:
            handbacks: Counter[str] = Counter()
            unconfirmed = 0
            for (command_id, digest), count in self.rejected_renewals.items():
                if self.completed_leases.get(command_id) == digest:
                    handbacks[command_id] += count
                else:
                    unconfirmed += count
            return dict(handbacks), unconfirmed


def run_queued_lease_proof(
    url: str,
    token: str,
    run_id: str,
    state_dir: Path,
    *,
    timeout_seconds: float = 75,
) -> dict[str, Any]:
    wire = CapacityWireClient(url, token, cluster_id=CLUSTER_ID)
    expected = {item.command_id: item for item in commands_for_run(run_id)}
    held = wire.claim(
        f"{run_id}-queue-reserve",
        execution_owners=[OWNER],
        max_commands=COMMAND_COUNT,
        lease_seconds=60,
    )
    if len(held) != COMMAND_COUNT or {item.command_id for item in held} != set(
        expected
    ):
        raise Cap004Error("CAP004 queue proof needs the exact owned command inventory")
    for item in held:
        require_owned(item, expected[item.command_id])
    selected = sorted(held, key=lambda item: item.command_id)[:2]

    def hand_back(command: RemoteActionCommand) -> RemoteActionCommand:
        return wire.complete(
            command,
            RemoteCommandResult(
                lease_token=str(command.lease_token),
                status=RemoteCommandStatus.WAITING,
                status_source="cap004-queued-lease-proof",
                details={"nonphysical": True, "queue_admission_only": True},
            ),
        )

    for item in selected:
        hand_back(item)
    selected_ids = {item.command_id for item in selected}
    rest = [item for item in held if item.command_id not in selected_ids]
    competitor_claims = []
    executed = []
    first_started = threading.Event()
    competitor_done = threading.Event()
    stop = threading.Event()
    errors: list[str] = []
    stale_renewal_status = None
    owner = f"{run_id}-queued-executor"
    client = QueueClient(url, token, executor_id=owner, selected=selected)
    claimed = client.claimed

    class WaitingAdapter:
        owner = OWNER

        def supports(self, step: WorkflowStepSpec) -> bool:
            return any(item.step == step for item in selected)

        def execute(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
            executed.append(context.idempotency_key)
            first_started.set()
            if not competitor_done.wait(timeout_seconds):
                raise Cap004Error(
                    "CAP004 competitor did not finish before the deadline"
                )
            return WorkflowStepOutcome.waiting(
                details={"nonphysical": True, "queue_admission_only": True}
            )

    def compete() -> None:
        nonlocal stale_renewal_status
        try:
            if not first_started.wait(timeout_seconds):
                raise Cap004Error("CAP004 queued adapter never started")
            if stop.is_set():
                return
            expiry = claimed[1].lease_expires_at
            if expiry is None:
                raise Cap004Error("CAP004 queued command has no original expiry")
            delay = (expiry - datetime.now(timezone.utc)).total_seconds() + 0.05
            if stop.wait(max(0, delay)):
                return
            commands = wire.claim(
                f"{run_id}-queue-competitor",
                execution_owners=[OWNER],
                max_commands=COMMAND_COUNT,
                lease_seconds=60,
            )
            competitor_claims.extend(commands)
            if (
                len(commands) != 1
                or commands[0].command_id != claimed[1].command_id
                or commands[0].lease_token == claimed[1].lease_token
            ):
                raise Cap004Error(
                    "CAP004 competitor did not acquire only the queued lease"
                )
            try:
                wire.renew(claimed[1], owner, 10)
            except ClusterExecutorError as exc:
                stale_renewal_status = exc.status_code
            else:
                raise Cap004Error("CAP004 stale queued lease was renewed")
        except Exception as exc:
            errors.append(f"queue competitor: {type(exc).__name__}")
        finally:
            competitor_done.set()

    executor = ClusterActionExecutor(
        client,
        [WaitingAdapter()],
        executor_id=owner,
        allowed_namespaces=set(),
        lease_seconds=10,
        batch_size=2,
        max_concurrent_commands=1,
        max_execution_seconds=timeout_seconds + 5,
        claim_state_path=str(state_dir / "queued-claim-state.json"),
        liveness_state_path=str(state_dir / "queued-loop-state.json"),
    )
    competitor = threading.Thread(target=compete, name=f"{run_id}-queued-competitor")
    started = time.monotonic()
    before = set(threading.enumerate())
    competitor.start()
    try:
        executor.run_once()
    finally:
        stop.set()
        first_started.set()
        executor.request_stop("CAP004 queued proof complete")
        competitor_done.set()
        owned = {competitor} | {
            thread
            for thread in threading.enumerate()
            if thread not in before
            and thread.name.startswith(
                ("gpu-fault-command", "cmd-command-", "lease-command-", "hold-command-")
            )
        }
        deadline = time.monotonic() + timeout_seconds + 5
        join_error: BaseException | None = None
        for thread in owned:
            if thread.ident is not None:
                try:
                    thread.join(timeout=max(0, deadline - time.monotonic()))
                except BaseException as exc:
                    join_error = exc
        if join_error is not None or any(thread.is_alive() for thread in owned):
            raise Cap004ThreadsRunning(
                "CAP004 queued Executor threads did not stop"
            ) from join_error
    counters = executor.metrics_snapshot()
    handback_rejections, unconfirmed_rejections = client.renewal_evidence()
    expected_renewal_failures = sum(handback_rejections.values())
    passed = (
        not errors
        and not client.problems
        and len(claimed) == 2
        and len(competitor_claims) == 1
        and executed == [claimed[0].idempotency_key]
        and stale_renewal_status == 409
        and client.completed == Counter({claimed[0].command_id: 1})
        and all(
            command_id == claimed[0].command_id and count == 1
            for command_id, count in handback_rejections.items()
        )
        and unconfirmed_rejections == 0
        and counters["lease_renewal_failures"] == expected_renewal_failures
        and counters["lease_lost_total"] == 1 + expected_renewal_failures
        and counters["results_withheld_total"] == 1
        and counters["reported_failures"] == 0
        and counters["unexpected_failures"] == 0
        and counters["execution_timeouts_total"] == 0
        and counters["stuck_executions"] == 0
    )
    # Only acknowledged holders hand back their leases. Expired/stolen tokens
    # never enter cleanup; the main proof reclaims all 25 with new authority.
    for item in [*rest, *competitor_claims]:
        hand_back(item)
    if not passed:
        raise Cap004Error("CAP004 queued lease authority proof failed")
    queued_expiry = claimed[1].lease_expires_at
    if queued_expiry is None:
        raise Cap004Error("CAP004 original queued lease expiry is missing")
    return {
        "passed": True,
        "batch_size": 2,
        "max_concurrent_commands": 1,
        "original_queued_expiry": queued_expiry.isoformat(),
        "stale_renewal_status": stale_renewal_status,
        "adapter_calls": len(executed),
        "queued_adapter_calls": 0,
        "competitor_claims": 1,
        "competitor_token_changed": True,
        "elapsed_seconds": time.monotonic() - started,
        "executor_counters": counters,
        "handback_renewal_rejections": handback_rejections,
        "unconfirmed_renewal_rejections": unconfirmed_rejections,
        "acknowledged_waiting_handbacks": [
            {
                "command_id": command_id,
                "lease_sha256": digest,
                "count": client.completed[command_id],
            }
            for command_id, digest in client.completed_leases.items()
        ],
        "renewal_rejections": [
            {
                "command_id": command_id,
                "lease_sha256": digest,
                "count": count,
                "acknowledged_waiting": client.completed_leases.get(command_id)
                == digest,
            }
            for (command_id, digest), count in client.rejected_renewals.items()
        ],
        "scope": "nonphysical queue admission; no hardware action or ledger mutation",
    }
