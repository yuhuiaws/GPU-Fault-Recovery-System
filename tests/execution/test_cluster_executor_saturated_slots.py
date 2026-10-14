"""Every slot holding a long command is a working executor, not a stalled one.

``ClaimLoop`` claims only into free slots, so five reboots holding all five
claim nothing for as long as they run. Two things used to ride on a claim:
the housekeeping (the spare sweep, the node-key sync -- which keys the nodes a
provider replaced) and the readiness breadcrumb, whose claim time the control
plane ages against ``GPU_FAULT_EXECUTOR_READINESS_MAX_CLAIM_AGE_SECONDS`` (300)
into "last successful claim was 301s ago": not ready. The loop now runs the
due housekeeping on every turn, rewrites the breadcrumb whenever a lease
renewal landed, and the probe ages the newest claim *or* renewal -- the same
authenticated round trip -- while a loop that does neither still ages out.

The executor's wall clock starts six minutes in the past and is moved a minute
at a time while the renewals land, so the probe, which ages the breadcrumb
against the real clock, sees a claim more than 300 s old.
"""

from __future__ import annotations

import io
import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.cluster_executor import bootstrap, regional_client
from gpu_fault.models import WorkflowOperation
from tests.execution.test_cluster_executor_claim_slots import (
    FastRenewalTick,
    GatedAdapter,
    LeasingControlPlane,
    eventually,
    running,
    slot_executor,
)
from tests.execution.test_cluster_executor_lease_and_report import (
    CLUSTER,
    EXECUTOR,
    remote_command,
)
from tests.execution.test_cluster_executor_metrics import executor_samples

# The control plane's default readiness limit on the claim age
# (``app/routes/regional.py``: age > max_claim_age_seconds is not ready).
MAX_CLAIM_AGE_SECONDS = 300


class WallClock:
    """The executor's wall clock, moved by the test."""

    def __init__(self, start: datetime) -> None:
        self._lock = threading.Lock()
        self._now = start

    def __call__(self) -> datetime:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now += timedelta(seconds=seconds)


class Ticker:
    """The monotonic clock the housekeeping jobs read their schedule from."""

    def __init__(self) -> None:
        self.seconds = 0.0

    def __call__(self) -> float:
        return self.seconds


class ScheduledJob:
    """The spare sweep or the node-key sync: due every ``interval``, at once first."""

    def __init__(self, clock: Ticker, interval: float) -> None:
        self.clock = clock
        self.interval = interval
        self.next_due = 0.0
        self.runs: list[float] = []
        self.reclaimed_total = 0
        self.failures_total = 0

    def due(self) -> bool:
        return self.clock() >= self.next_due

    def run(self) -> list[str]:
        self.runs.append(self.clock())
        self.next_due = self.clock() + self.interval
        return []


def readiness(
    monkeypatch: pytest.MonkeyPatch, breadcrumb: Path
) -> tuple[int, float | None]:
    """The real readiness probe on ``breadcrumb``: its verdict and the age sent.

    The control plane is the readiness route's claim-age rule; registration,
    pins and the backlog are not what these tests are about.
    """

    monkeypatch.setenv("GPU_FAULT_CLUSTER_EXECUTOR_CLAIM_STATE_PATH", str(breadcrumb))
    monkeypatch.setenv("GPU_FAULT_CLUSTER_ID", CLUSTER)
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_URL", "https://control.invalid")
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_TOKEN", "synthetic-cluster-a-token")
    sent: list[float | None] = []

    def control_plane(request: Any, **_kwargs: Any) -> io.BytesIO:
        age = json.loads(request.data)["last_successful_claim_age_seconds"]
        sent.append(age)
        stale = age is not None and age > MAX_CLAIM_AGE_SECONDS
        reasons = [f"last successful claim was {age:.0f}s ago"] if stale else []
        return io.BytesIO(json.dumps({"ready": not stale, "reasons": reasons}).encode())

    monkeypatch.setattr(regional_client, "urlopen", control_plane)
    verdict = bootstrap.readiness_probe()
    (age,) = sent
    return verdict, age


def read(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def test_five_long_restarts_holding_every_slot_keep_the_pod_ready_and_on_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("gpu_fault.cluster_executor.lease.Event", FastRenewalTick)
    restarts = [
        remote_command(f"restart-{index}", operation=WorkflowOperation.RESTART_NODE)
        for index in range(5)
    ]
    client = LeasingControlPlane([*restarts, remote_command("next-step")])
    adapter = GatedAdapter(*(command.command_id for command in restarts))
    ticker = Ticker()
    sweep = ScheduledJob(ticker, 300)
    key_sync = ScheduledJob(ticker, 60)
    executor = slot_executor(
        client,
        adapter,
        tmp_path,
        batch_size=5,
        max_concurrent_commands=5,
        lease_seconds=120,
        spare_reservation_sweep=sweep,
        node_key_sync=key_sync,
    )
    wall = WallClock(datetime.now(timezone.utc) - timedelta(minutes=6))
    executor.now = wall
    breadcrumb = tmp_path / "claim-state.json"

    with running(executor, adapter):
        eventually(
            lambda: len(adapter.started_ids()) == 5, "the five restarts never ran"
        )
        claimed_at = read(breadcrumb)["last_successful_claim_at"]
        for minute in range(1, 7):
            wall.advance(60)
            ticker.seconds += 60
            eventually(
                lambda: read(breadcrumb).get("last_successful_renewal_at")
                == wall().isoformat(),
                f"no renewal reached the readiness breadcrumb in minute {minute}",
            )
            eventually(
                lambda: len(key_sync.runs) == 1 + minute,
                f"the node-key sync missed minute {minute}: {key_sync.runs}",
            )
        state = read(breadcrumb)
        verdict, age = readiness(monkeypatch, breadcrumb)
        samples = executor_samples(executor)
        claims_while_busy = list(client.claims)
        adapter.release("restart-2")
        eventually(
            lambda: "next-step" in client.reported_ids(),
            "the slot the restart freed was not claimed again",
        )

    assert state["last_successful_claim_at"] == claimed_at, (
        "nothing may be claimed while every slot is busy"
    )
    assert [claim.command_ids for claim in claims_while_busy] == [
        tuple(command.command_id for command in restarts)
    ], claims_while_busy
    claim_age = (
        datetime.now(timezone.utc) - datetime.fromisoformat(claimed_at)
    ).total_seconds()
    assert claim_age > MAX_CLAIM_AGE_SECONDS, claim_age
    assert verdict == 0, f"the Pod went not ready with every lease renewing: {age}"
    assert age is not None and age < 60, (
        f"the probe must age the newest renewal, not the {claim_age:.0f}s claim"
    )
    assert key_sync.runs == [0.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0], (
        key_sync.runs
    )
    assert sweep.runs == [0.0, 300.0], sweep.runs
    assert (
        samples["last_lease_renewal_timestamp"] - samples["last_claim_timestamp"] >= 360
    ), samples
    freed = client.claims[1]
    assert (freed.max_commands, freed.command_ids) == (1, ("next-step",)), freed


@pytest.mark.parametrize(
    ("renewed_seconds_ago", "ready"),
    [(10, True), (MAX_CLAIM_AGE_SECONDS + 1, False), (None, False)],
)
def test_a_loop_that_neither_claims_nor_renews_still_goes_not_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    renewed_seconds_ago: int | None,
    ready: bool,
) -> None:
    """A renewal stands in for a claim only while it is itself fresh."""

    now = datetime.now(timezone.utc)
    breadcrumb = tmp_path / "claim-state.json"
    breadcrumb.write_text(
        json.dumps(
            {
                "executor_id": EXECUTOR,
                "execution_owners": ["owner-a"],
                "last_successful_claim_at": (now - timedelta(seconds=400)).isoformat(),
                "last_successful_renewal_at": (
                    None
                    if renewed_seconds_ago is None
                    else (now - timedelta(seconds=renewed_seconds_ago)).isoformat()
                ),
            }
        ),
        encoding="utf-8",
    )

    verdict, age = readiness(monkeypatch, breadcrumb)

    assert verdict == (0 if ready else 1), (renewed_seconds_ago, age)
    expected = 400 if renewed_seconds_ago is None else renewed_seconds_ago
    assert age is not None and expected <= age < expected + 30, age
