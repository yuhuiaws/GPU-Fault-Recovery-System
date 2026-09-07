#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
from threading import Lock, Thread
import time

from gpu_fault.cluster_executor import (
    ClusterActionExecutor,
    RegionalExecutorClient,
    RegionalFleetRegistry,
)
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.models import AdvisoryNotification


STATE = Path("/state")
READY = STATE / "ready.json"
EXECUTOR_STATE = STATE / "executor-state.json"
WINNER = STATE / "winner.json"
PHYSICAL = STATE / "physical.json"
OWNER = "gpu-fault-ha006-test"


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


class SharedLedgerAdapter:
    """Two-round adapter: WAITING on the first claim, the shared action on the second.

    HA-006 must exercise both takeover branches. The first time this process
    sees an idempotency key it reports WAITING without touching anything, so the
    command goes back to the queue and is re-claimed by whichever replica polls
    next -- that is the WAITING branch. The second time it runs the shared
    action: it tries to win the notification-dedup ledger and, if it wins,
    counts one physical action and holds the lease while sleeping -- the LEASED
    branch when the runner kills it there. A loser replays the winner's result
    from the ledger and counts nothing.

    ``physical_actions`` is written to ``/state/physical.json`` and into the
    executor-state record, so the runner can sum the counters of both Pods
    instead of trusting a constant in the result details.
    """

    owner = OWNER

    def __init__(
        self,
        registry: RegionalFleetRegistry,
        *,
        run_id: str,
        sleep_seconds: float,
        wait_first_round: bool = False,
    ) -> None:
        self.registry = registry
        self.run_id = run_id
        self.sleep_seconds = sleep_seconds
        self.wait_first_round = wait_first_round
        self.physical_actions = 0
        self.rounds_by_key: dict[str, int] = {}
        self._lock = Lock()

    def supports(self, step) -> bool:
        return step.execution_owner == self.owner

    def _next_round(self, key: str) -> int:
        with self._lock:
            self.rounds_by_key[key] = self.rounds_by_key.get(key, 0) + 1
            return self.rounds_by_key[key]

    def _record_physical(self) -> None:
        with self._lock:
            self.physical_actions += 1
            # Next to the winner marker, so both live in the same state
            # directory (and a test that relocates WINNER relocates this too).
            atomic_json(
                WINNER.with_name(PHYSICAL.name),
                {
                    "pod": socket.gethostname(),
                    "physical_actions": self.physical_actions,
                    "observed_at": time.time(),
                },
            )

    def execute(self, context) -> WorkflowStepOutcome:
        round_number = self._next_round(context.idempotency_key)
        if self.wait_first_round and round_number == 1:
            return WorkflowStepOutcome.waiting(
                operation_id=f"ha006/{context.idempotency_key}",
                details={
                    "simulated": True,
                    "round": round_number,
                    "pod": socket.gethostname(),
                    "waiting_reason": "first round reports WAITING by design",
                },
            )
        candidate = AdvisoryNotification(
            notification_id=f"notification-{self.run_id}-{socket.gethostname()}",
            deduplication_key=f"{self.run_id}/shared-action-ledger",
            cluster_name=context.incident.cluster_id,
            incident_id=context.incident.incident_id,
            subject="HA-006 shared action ledger drill",
            body_text="Synthetic non-destructive executor takeover marker.",
            support_case_draft="Acceptance drill only; no provider action.",
            drill_id=self.run_id,
            category="HA_ACCEPTANCE",
        )
        saved = self.registry.save_notification_if_absent(candidate)
        won = saved.notification_id == candidate.notification_id
        if won:
            self._record_physical()
            atomic_json(
                WINNER,
                {
                    "notification_id": saved.notification_id,
                    "pod": socket.gethostname(),
                    "observed_at": time.time(),
                },
            )
            time.sleep(self.sleep_seconds)
        return WorkflowStepOutcome.succeeded(
            operation_id=f"ha006/{context.idempotency_key}",
            details={
                "simulated": True,
                "round": round_number,
                "pod": socket.gethostname(),
                "cached": not won,
                "physical_count": self.physical_actions,
                "shared_notification_id": saved.notification_id,
            },
        )


def record_state(executor: ClusterActionExecutor, adapter: SharedLedgerAdapter) -> None:
    while True:
        atomic_json(
            EXECUTOR_STATE,
            {
                "pod": socket.gethostname(),
                "claimed_total": executor.claimed_total,
                "reported_failures": executor.reported_failures,
                "unexpected_failures": executor.unexpected_failures,
                "lease_renewal_failures": executor.lease_renewal_failures,
                "physical_actions": adapter.physical_actions,
                "rounds_by_key": dict(adapter.rounds_by_key),
                "last_successful_claim_at": (
                    executor.last_successful_claim_at.isoformat()
                    if executor.last_successful_claim_at is not None
                    else None
                ),
            },
        )
        time.sleep(0.5)


def main() -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    registrations = json.loads(Path("/tokens/clusters.json").read_text())
    registration = registrations[0]
    client = RegionalExecutorClient(
        os.environ["CONTROL_PLANE_URL"],
        registration["cluster_id"],
        registration["token"],
        timeout_seconds=30,
        ca_file="/tls/ca.crt",
        executor_artifact_sha256=os.environ["EXECUTOR_ARTIFACT_SHA256"],
        executor_compatibility_digest=os.environ["EXECUTOR_COMPATIBILITY_DIGEST"],
    )
    registry = RegionalFleetRegistry(client)
    run_id = os.environ["RUN_ID"]
    lease_seconds = int(os.environ.get("LEASE_SECONDS", "30"))
    poll_seconds = int(os.environ.get("POLL_SECONDS", "1"))
    claim_backoff_max_seconds = int(os.environ.get("CLAIM_BACKOFF_MAX_SECONDS", "4"))
    adapter = SharedLedgerAdapter(
        registry,
        run_id=run_id,
        sleep_seconds=float(os.environ.get("WINNER_SLEEP_SECONDS", "60")),
        wait_first_round=os.environ.get("WAIT_FIRST_ROUND", "true") == "true",
    )
    executor = ClusterActionExecutor(
        client,
        [adapter],
        executor_id=socket.gethostname(),
        allowed_namespaces={"default"},
        poll_seconds=poll_seconds,
        lease_seconds=lease_seconds,
        batch_size=1,
        max_concurrent_commands=1,
        claim_backoff_max_seconds=claim_backoff_max_seconds,
        claim_state_path="/state/claim-state.json",
    )
    atomic_json(
        READY,
        {
            "cluster_id": registration["cluster_id"],
            "executor_id": executor.executor_id,
            "lease_seconds": lease_seconds,
            "poll_seconds": poll_seconds,
            "claim_backoff_max_seconds": claim_backoff_max_seconds,
            "wait_first_round": adapter.wait_first_round,
            "run_id": run_id,
        },
    )
    Thread(target=record_state, args=(executor, adapter), daemon=True).start()
    executor.run()


if __name__ == "__main__":
    main()
