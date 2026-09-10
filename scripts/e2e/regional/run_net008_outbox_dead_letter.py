#!/usr/bin/env python3
"""GF-REGIONAL-NET-008: outbox dead-letter semantics and a kept kernel stream.

NET-001 proved the kernel collector buffers through a blackout and replays by
idempotency key. ARCH-G2/G3/G4 changed what happens around that: a 401/403 is
a token-rotation window and stays *replayable* instead of dying; a replay
4xx that is a verdict on the record dead-letters it where ``gpu-fault-collector
outbox`` can list, count and requeue it; and a delivery failure no longer
makes the collector reopen ``/dev/kmsg`` at the live end and lose the lines in
between. This case proves the three on one idle node.

The wrong token is generated on the node and never leaves it; the dead letter
is a record naming a channel the control plane does not serve, seeded only
while the unit is stopped; the blackout is NET-001's tagged iptables reject
with a rollback timer. Every kmsg line is a labelled user-space monitor-only
XID 63. Plan-only by default.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import net008_verdicts as verdicts  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.collector_window_fixture import (  # noqa: E402
    open_window_or_rollback,
    CollectorWindowFixture,
    WindowSettings,
    run_case_main,
    run_window_case,
    utc_now,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
)

CASE_ID = verdicts.CASE_ID
CONFIRMATION = verdicts.CONFIRMATION


def stop_conditions() -> list[str]:
    return [
        "predecessor evidence is not PASS",
        "the node has a business workload, ownership or a non-ACTIVE Agent",
        "the kernel outbox is not empty at baseline",
        "a 403-refused record is dead-lettered or the collector crash-loops",
        "the rollback timer is not active before the network reject",
        "the kernel collector changes PID, invocation or /dev/kmsg fd",
        "any firewall rule, window, seeded record, Pod or ConfigMap remains",
    ]


def plan_details(settings: WindowSettings, preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk": "live-kernel-log-injection",
        "target_node": settings.node,
        "predecessor": preflight["predecessor"],
        "unit": verdicts.UNIT,
        "mutation": (
            "A: restart the kernel collector under a wrong cluster token and write "
            f"{verdicts.TRANSIENT_EVENTS} monitor-only XID 63 lines; B: seed one "
            "retired-channel outbox record while the unit is stopped and requeue it "
            "with the shipped CLI; C: reject TCP/443 to the control plane for "
            f"{verdicts.BLOCK_SECONDS}s with a {verdicts.BLOCK_TTL_SECONDS}s rollback "
            "timer and write one more XID 63 line"
        ),
        "stop_conditions": stop_conditions(),
        "rollback": {
            "window_deadman_seconds": verdicts.WINDOW_RESTORE_SECONDS,
            "firewall_rollback_seconds": verdicts.BLOCK_TTL_SECONDS,
            "runner_finally_closes_window_unblocks_and_purges_seed": True,
            "runner_finally_deletes_probe_resources": True,
        },
    }


def _bdf(snapshot: dict[str, Any]) -> str:
    inventory = snapshot.get("gpu_inventory") or []
    if not inventory:
        raise RegionalFixtureError("host probe read no GPU inventory")
    return str(inventory[0]["pci_bus_id"])


def _kernel(snapshot: dict[str, Any]) -> dict[str, Any]:
    return dict((snapshot.get("outboxes") or {}).get(verdicts.COLLECTOR) or {})


@dataclass
class _Run:
    """Everything the phases share, and the flags ``_cleanup`` reads."""

    settings: WindowSettings
    fixture: CollectorWindowFixture
    case_dir: Path
    run_id: str
    marker_base: str
    baseline: dict[str, Any]
    bdf: str
    resolved: dict[str, Any]
    started: datetime
    tag: str
    unit_baseline: dict[str, Any]
    stages: dict[str, list[str]] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    transient_markers: list[str] = field(default_factory=list)
    opened: dict[str, Any] = field(default_factory=dict)
    window_open: bool = False
    blocked: bool = False
    seeded: bool = False

    def ip_arguments(self) -> list[str]:
        ips = list(self.resolved.get("endpoint_ipv4") or [])
        return [value for ip in ips for value in ("--ip", ip)]

    def write_kmsg(self, marker: str) -> None:
        self.fixture.execute(
            "write-kmsg", "--kind", "xid63", "--marker", marker, "--pci-bdf", self.bdf
        )

    def kernel_activity(self) -> dict[str, Any]:
        return self.fixture.node_activity(self.started, evidence_kind="NVIDIA_KERNEL")

    def stop_unit(self) -> dict[str, Any]:
        return self.fixture.execute(
            "stop-unit",
            "--unit",
            verdicts.UNIT,
            "--run-id",
            self.run_id,
            "--restore-seconds",
            "300",
        )

    def start_unit(self) -> None:
        self.fixture.execute(
            "start-unit", "--unit", verdicts.UNIT, "--run-id", self.run_id
        )


def _kept_records(kernel: dict[str, Any]) -> list[dict[str, Any]]:
    """Records other than the seeded retired-channel dead letter."""

    return [
        item
        for item in kernel.get("records") or []
        if item.get("path") != verdicts.RETIRED_CHANNEL_PATH
    ]


def _prepare_run(
    settings: WindowSettings,
    fixture: CollectorWindowFixture,
    case_dir: Path,
    attempt: int,
) -> _Run:
    run_id = f"n008-a{attempt}-{int(time.time())}"
    marker_base = f"net008-{int(time.time())}-a{attempt}"
    baseline = fixture.snapshot(marker_base)
    write_json_atomic(case_dir / "host-baseline.json", baseline)
    if int((_kernel(baseline).get("stats") or {}).get("depth") or 0):
        raise RegionalFixtureError("kernel outbox is not empty at baseline")
    if not settings.endpoint_host:
        raise RegionalFixtureError(
            "--endpoint-host (or GPU_FAULT_NET_ENDPOINT_HOST) is required"
        )
    resolved = fixture.execute("resolve", "--endpoint-host", settings.endpoint_host)
    run = _Run(
        settings=settings,
        fixture=fixture,
        case_dir=case_dir,
        run_id=run_id,
        marker_base=marker_base,
        baseline=baseline,
        bdf=_bdf(baseline),
        resolved=resolved,
        started=utc_now(),
        tag=f"gpu-fault-net008-{int(time.time()) % 100000}",
        unit_baseline=baseline["services"][verdicts.UNIT],
    )
    run.evidence = {"run_id": run_id, "host_baseline": baseline}
    run.transient_markers = [
        f"{marker_base}-t{index}" for index in range(1, verdicts.TRANSIENT_EVENTS + 1)
    ]
    return run


def _phase_transient_403(run: _Run) -> None:
    """Phase A: a wrong token makes every record replayable, never dead."""

    run.opened = open_window_or_rollback(
        run.fixture,
        run.run_id,
        "--unit",
        verdicts.UNIT,
        "--env",
        "GPU_FAULT_CONTROL_PLANE_TOKEN=@invalid",
        "--restore-seconds",
        str(verdicts.WINDOW_RESTORE_SECONDS),
    )
    run.window_open = True
    write_json_atomic(run.case_dir / "window-open.json", run.opened)
    run.stages["window"] = verdicts.window_errors(run.opened)
    # open-window restarts the unit, and the restarted collector reopens
    # /dev/kmsg at the live tail: a line written before its reader is up is
    # never seen (attempt 1, 2026-09-10: the first transient marker vanished
    # while the second, ten seconds later, was buffered). Let it settle first.
    time.sleep(verdicts.COLLECTOR_SETTLE_SECONDS)
    for marker in run.transient_markers:
        run.write_kmsg(marker)
        time.sleep(10)

    def refused() -> dict[str, Any] | None:
        snapshot = run.fixture.snapshot(run.marker_base)
        kernel = _kernel(snapshot)
        if verdicts.transient_outbox_errors(
            kernel.get("records") or [],
            kernel.get("stats") or {},
            expected=len(run.transient_markers),
        ):
            return None
        return snapshot

    refused_snapshot = run.fixture.wait_until(
        refused,
        timeout_seconds=300,
        poll_seconds=10,
        case_dir=run.case_dir,
        name="transient",
    ) or run.fixture.snapshot(run.marker_base)
    kernel = _kernel(refused_snapshot)
    run.stages["transient_outbox"] = verdicts.transient_outbox_errors(
        kernel.get("records") or [],
        kernel.get("stats") or {},
        expected=len(run.transient_markers),
    )
    run.stages["service_under_403"] = verdicts.service_errors(
        run.opened.get("after") or {}, refused_snapshot["services"][verdicts.UNIT]
    )
    run.evidence["transient_outbox"] = kernel


def _phase_seed_dead_letter(run: _Run) -> None:
    """Phase B seed: one retired-channel record, written while the unit is
    stopped by close-window's cycle; then a live post wakes the replay."""

    stopped = run.stop_unit()
    seed = run.fixture.execute(
        "seed-outbox-record",
        "--collector",
        verdicts.COLLECTOR,
        "--marker",
        run.marker_base,
    )
    run.seeded = True
    run.evidence["seed"] = seed
    closed = run.fixture.execute("close-window", "--run-id", run.run_id, timeout=300)
    run.window_open = False
    run.start_unit()
    write_json_atomic(
        run.case_dir / "window-close.json", {"stopped": stopped, "closed": closed}
    )
    # A live post wakes the replay (NET-001 step 7).
    run.write_kmsg(f"{run.marker_base}-wake")


def _phase_replay_404_dead_letter(run: _Run) -> None:
    """Phase B verdicts: the 403 records deliver, the retired-channel record
    dead-letters, and the shipped CLI lists and requeues it."""

    def replayed() -> dict[str, Any] | None:
        snapshot = run.fixture.snapshot(run.marker_base)
        kernel = _kernel(snapshot)
        activity = run.kernel_activity()
        records = kernel.get("records") or []
        if any(
            item.get("marker_present")
            and item.get("path") != verdicts.RETIRED_CHANNEL_PATH
            for item in records
        ):
            return None
        if verdicts.dead_letter_errors(
            records, kernel.get("stats") or {}, marker=run.marker_base
        ):
            return None
        if verdicts.delivered_errors(
            activity.get("evidence") or [],
            _kept_records(kernel),
            {**(kernel.get("stats") or {}), "replayable": 0},
            markers=run.transient_markers,
        ):
            return None
        return {"snapshot": snapshot, "activity": activity}

    converged = run.fixture.wait_until(
        replayed,
        timeout_seconds=verdicts.REPLAY_TIMEOUT_SECONDS,
        poll_seconds=15,
        case_dir=run.case_dir,
        name="replay",
    )
    if converged is None:
        converged = {
            "snapshot": run.fixture.snapshot(run.marker_base),
            "activity": run.kernel_activity(),
        }
    kernel = _kernel(converged["snapshot"])
    records = kernel.get("records") or []
    run.stages["delivered"] = verdicts.delivered_errors(
        converged["activity"].get("evidence") or [],
        _kept_records(kernel),
        {**(kernel.get("stats") or {}), "replayable": 0},
        markers=run.transient_markers,
    )
    run.stages["dead_letter"] = verdicts.dead_letter_errors(
        records, kernel.get("stats") or {}, marker=run.marker_base
    )
    listing = run.fixture.execute(
        "outbox",
        "--collector",
        verdicts.COLLECTOR,
        "--action",
        "list",
        "--marker",
        run.marker_base,
    )
    run.stages["listing"] = verdicts.listing_errors(listing.get("lines") or [])
    requeued = run.fixture.execute(
        "outbox", "--collector", verdicts.COLLECTOR, "--action", "requeue-dead"
    )
    run.stages["requeue"] = verdicts.requeue_errors(requeued)
    run.evidence["dead_letter"] = {
        "records": records,
        "stats": kernel.get("stats"),
        "listing": listing.get("lines"),
        "requeued": requeued,
    }
    run.evidence["delivered_record_ids"] = [
        item.get("record_id") for item in converged["activity"].get("evidence") or []
    ]


def _phase_blackout_stream(run: _Run) -> None:
    """Phase C: a blackout buffers one line and never reopens /dev/kmsg."""

    ip_arguments = run.ip_arguments()
    if not ip_arguments:
        raise RegionalFixtureError("the control-plane endpoint resolved to no IPv4")
    before_stream = run.fixture.snapshot(run.marker_base)["kmsg_stream"]
    block = run.fixture.execute(
        "block",
        "--tag",
        run.tag,
        "--ttl-seconds",
        str(verdicts.BLOCK_TTL_SECONDS),
        *ip_arguments,
    )
    run.blocked = True
    blackout_marker = f"{run.marker_base}-blackout"
    time.sleep(5)
    run.write_kmsg(blackout_marker)
    time.sleep(verdicts.BLOCK_SECONDS)
    during_stream = run.fixture.snapshot(run.marker_base)["kmsg_stream"]
    unblock = run.fixture.execute("unblock", "--tag", run.tag, *ip_arguments)
    run.blocked = False
    run.stages["blackout"] = verdicts.blackout_errors(block, unblock)
    run.write_kmsg(f"{run.marker_base}-wake2")

    def blackout_delivered() -> dict[str, Any] | None:
        activity = run.kernel_activity()
        hits = [
            item
            for item in activity.get("evidence") or []
            if blackout_marker in str(item.get("payload"))
        ]
        return activity if len(hits) == 1 else None

    after_activity = (
        run.fixture.wait_until(
            blackout_delivered,
            timeout_seconds=verdicts.REPLAY_TIMEOUT_SECONDS,
            poll_seconds=15,
            case_dir=run.case_dir,
            name="blackout-replay",
        )
        or run.kernel_activity()
    )
    after_snapshot = run.fixture.snapshot(run.marker_base)
    run.stages["stream_identity"] = [
        *verdicts.stream_identity_errors(before_stream, during_stream),
        *verdicts.stream_identity_errors(during_stream, after_snapshot["kmsg_stream"]),
    ]
    run.stages["blackout_delivered"] = verdicts.delivered_errors(
        after_activity.get("evidence") or [],
        _kept_records(_kernel(after_snapshot)),
        {"replayable": 0},
        markers=[blackout_marker],
    )
    run.stages["service_final"] = verdicts.service_errors(
        run.unit_baseline, after_snapshot["services"][verdicts.UNIT]
    )
    run.evidence["stream"] = {
        "before": before_stream,
        "during": during_stream,
        "after": after_snapshot["kmsg_stream"],
    }


def _cleanup(run: _Run) -> None:
    """Undo whatever is still armed, in the order the phases armed it:
    firewall reject, env window, seeded record (unit stopped around the
    purge with a fail-safe start)."""

    if run.blocked:
        run.fixture.execute("unblock", "--tag", run.tag, *run.ip_arguments())
    if run.window_open:
        run.fixture.execute("close-window", "--run-id", run.run_id, timeout=300)
    if run.seeded:
        run.stop_unit()
        purged = run.fixture.execute(
            "purge-outbox-record",
            "--collector",
            verdicts.COLLECTOR,
            "--marker",
            run.marker_base,
        )
        run.start_unit()
        run.stages["purge"] = verdicts.purge_errors(purged)
        run.evidence["purge"] = purged


def _result(run: _Run) -> dict[str, Any]:
    return {
        "verdict": verdicts.case_verdict(run.stages),
        "errors": [
            f"{stage}: {error}"
            for stage, items in run.stages.items()
            for error in items
        ],
        "stages": run.stages,
        **run.evidence,
        "limitations": [
            "The 403 comes from a wrong token generated on the node, not from a real "
            "rotation; AUTH-016 covers the rotation itself.",
            "The dead letter is a record naming a retired channel path; a handler 4xx "
            "after the 202 is booked by the processor and is COLLECT-018's subject.",
            "The seed and purge stop the kernel collector twice for a few seconds "
            "each, with a fail-safe start armed; lines written in those seconds are "
            "read from the ring buffer when it starts.",
        ],
    }


def execute(
    settings: WindowSettings,
    fixture: CollectorWindowFixture,
    case_dir: Path,
    attempt: int,
    deadline: datetime,
) -> dict[str, Any]:
    del deadline
    run = _prepare_run(settings, fixture, case_dir, attempt)
    try:
        _phase_transient_403(run)
        _phase_seed_dead_letter(run)
        _phase_replay_404_dead_letter(run)
        _phase_blackout_stream(run)
    finally:
        _cleanup(run)
    return _result(run)


def main() -> int:
    return run_window_case(
        case_id=CASE_ID,
        confirmation=CONFIRMATION,
        parser_description=(
            "Run the guarded NET-008 acceptance: 403 stays replayable, a replay 4xx "
            "dead-letters and can be requeued, and a blackout never reopens /dev/kmsg."
        ),
        plan_details=plan_details,
        execute=execute,
    )


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
