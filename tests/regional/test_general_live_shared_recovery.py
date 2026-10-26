"""GF-GENERAL-LIVE-005: the delivery hold and the recovery reads, driven end to end.

Two live verdicts of 2026-10-26 on plain EKS (release R13g) were wrong about a
right product. ``healthy_other_mount`` failed ``bounded_delivery`` after a
20-second detection: one host batch between the pause and the finding -- the
edge, the only batch the collector ships for a held breach before its health
summary. ``mount_disappears`` failed ``recovery`` because its ``since`` was
stamped after the remount while the recovered edge landed between the mount
syscall and the probe's answer. The tests here drive the variants of
``general_live_shared`` over a scripted edge-only product
(``edge_only_documents``) with the fakes of ``test_general_live_shared``, and
pin the reads of ``general_live_shared_recovery`` directly.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_shared as shared
from scripts.e2e.regional import general_live_shared_recovery as recovery
from scripts.e2e.regional.general_live_common import (
    STORE_POLL_SECONDS,
    sustained_detection_budget,
)
from scripts.e2e.regional.general_live_store import signal_key
from tests.regional._general_live_fakes import store_poll_clock, store_script
from tests.regional.test_general_live_shared import (
    NODE,
    SPECS,
    T0,
    batch,
    context,
    diagnostic,
    fake_server,  # noqa: F401 - the autouse fake export server, active here too
    sample,
    servers_by_suffix,
)


# --------------------------------------------------------------------------- #
# the hang held across the collector's next delivery (live 2026-10-26 14:18Z)
# --------------------------------------------------------------------------- #
def at_seconds(seconds: float) -> str:
    return (T0 + timedelta(seconds=seconds)).isoformat()


def batch_at(observed_at: str, *items: dict[str, Any]) -> dict[str, Any]:
    return {
        **batch(0, *items),
        "observed_at": observed_at,
        "batch_id": f"host-{observed_at}",
    }


def edge_only_documents(
    ctx: Any,
    mount: str,
    *,
    node: str = NODE,
    other: str | None = None,
    source: str = "10.0.9.1:/a",
    metric: str = recovery.UNAVAILABLE,
    mode: str | None = None,
) -> dict[str, dict[str, Any]]:
    """The product as the host collector really delivers a held condition.

    The baseline before the injection (120 min back), ONE edge batch at T+41 s,
    the health summary at T+43 s (the finding rides along, so it is evidence),
    the recovered edge at T+45 s -- the only reading after the undo -- and the
    health signal active until it clears at T+9 min. ``detected`` is the store
    at the finding, ``held`` once the summary landed, ``recovered`` after the
    undo. ``T`` is ``T0``; the stamps the tests take are T+40 s (before the
    undo) and T+50 s (after it).
    """

    identity = {"source": source, "fstype": "nfs4"}
    fault_labels = {**identity, **({"failure_mode": mode} if mode else {})}
    companion = sample(recovery.COMPANION_METRIC, None, 40.0)
    other_ok = (
        [sample(recovery.UNAVAILABLE, other, 0.0), sample(recovery.USED, other, 2.0)]
        if other
        else []
    )
    before = batch(
        -120,
        sample(recovery.UNAVAILABLE, mount, 0.0, identity),
        sample(recovery.USED, mount, 2.0, identity),
        *other_ok,
    )
    fault = (
        sample(recovery.USED, mount, 99.2, identity)
        if metric == recovery.USED
        else sample(recovery.UNAVAILABLE, mount, 1.0, fault_labels)
    )
    edge = batch_at(at_seconds(41), fault, companion, *other_ok)
    summary = batch_at(at_seconds(43), fault, companion, *other_ok)
    recovered = batch_at(
        at_seconds(45),
        sample(recovery.UNAVAILABLE, mount, 0.0, identity),
        sample(recovery.USED, mount, 1.0, identity),
    )
    chain = diagnostic(mount, metric, node)
    key = signal_key(
        ctx.settings.regional.cluster_id, node, recovery.UNAVAILABLE, mount
    )

    def document(*batches: dict[str, Any], cleared: bool) -> dict[str, Any]:
        signal = {
            "signal_key": key,
            "active": not cleared,
            "observed_at": at_seconds(9 * 60) if cleared else at_seconds(41),
        }
        return {
            "incidents": [chain["incident"]],
            "workflows": [chain["workflow"]],
            "batches": list(batches),
            "markers": [],
            "commands": [],
            "health_signals": {key: signal},
            "notifications": [],
        }

    return {
        "detected": document(before, edge, cleared=False),
        "held": document(before, edge, summary, cleared=False),
        "recovered": document(before, edge, summary, recovered, cleared=True),
    }


@pytest.mark.parametrize("variant", ["stat_timeout", "healthy_other_mount"])
def test_the_hang_is_held_until_the_collector_delivers_again(
    tmp_path: Path, variant: str
) -> None:
    """Live 2026-10-26 14:15-14:21Z (``healthy_other_mount``, plain EKS, R13g):
    paused 14:18:31Z, finding 14:18:51Z, resumed 14:19:03Z -- one host batch,
    the edge, between the pause and the finding, and ``bounded_delivery`` failed
    a product that detected within 20 s. The collector ships a held breach once
    and then on its health summary, so the variant keeps the export paused until
    the second batch is in the store, resumes after it, and grades the window
    the resume closes."""

    ctx = context(variant, tmp_path, activity=None)
    mount = shared.mount_path(ctx, shared.PRIMARY)
    other = (
        shared.mount_path(ctx, shared.OTHER)
        if variant == "healthy_other_mount"
        else None
    )
    documents = edge_only_documents(ctx, mount, other=other)
    reads: list[tuple[str, bool]] = []

    def product(_script: str, *_arguments: str, **_: Any) -> dict[str, Any]:
        server = servers_by_suffix()["a"]
        if not server.paused:
            stage = "recovered" if "resume" in server.calls else "detected"
        else:
            under_hang = sum(1 for _stage, paused in reads if paused)
            stage = "detected" if under_hang < 2 else "held"
        reads.append((stage, server.paused))
        return documents[stage]

    ctx.regional.cpu_python_responder = product
    result = SPECS[variant].execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["stages"]["bounded_delivery"] == [], result["stages"]
    timeline = json.loads((ctx.case_dir / "hang-delivery-timeline.json").read_text())
    assert [entry["accepted"] for entry in timeline["entries"]] == [False, True], (
        "the hold polled until the second batch after the pause was accepted"
    )
    assert [stage for stage, paused in reads if paused] == [
        "detected",
        "detected",
        "held",
    ], "the detection read, one hold poll with the edge alone, the summary"
    server = servers_by_suffix()["a"]
    assert server.calls == ["create", "wait_ready", "pause", "resume"], server.calls
    assert reads.index(("held", True)) < reads.index(("recovered", False)), (
        "the resume came after the read that saw the second batch"
    )


def test_a_collector_the_hang_stalled_fails_the_delivery_stage(tmp_path: Path) -> None:
    """No second batch ever lands: the hold runs its budget out -- one summary
    period, the interval and the margin -- the export is resumed anyway, and
    ``bounded_delivery`` names the single batch and the window."""

    ctx = store_poll_clock(context("stat_timeout", tmp_path, activity=None))
    mount = shared.mount_path(ctx, shared.PRIMARY)
    documents = edge_only_documents(ctx, mount)

    def product(_script: str, *_arguments: str, **_: Any) -> dict[str, Any]:
        server = servers_by_suffix()["a"]
        return documents["recovered" if "resume" in server.calls else "detected"]

    ctx.regional.cpu_python_responder = product
    result = SPECS["stat_timeout"].execute(ctx)
    assert result["verdict"] == "FAIL", result
    delivery = result["stages"]["bounded_delivery"]
    assert (
        len(delivery) == 1 and "only 1 host batch(es) were accepted" in delivery[0]
    ), delivery
    assert "next scheduled delivery never arrived" in delivery[0], delivery
    assert result["stages"]["recovery"] == [], "resumed anyway, and it recovered"
    assert servers_by_suffix()["a"].paused is False, "resumed after the budget ran out"
    timeline = json.loads((ctx.case_dir / "hang-delivery-timeline.json").read_text())
    assert all(entry["accepted"] is False for entry in timeline["entries"]), timeline
    budget = sustained_detection_budget(ctx, 0, summaries=1)
    assert budget == 300 + 15 + 180, "one summary, the interval, the margin"
    assert len(timeline["entries"]) == -(-budget // STORE_POLL_SECONDS), (
        "one store poll every 20 s until the budget ran out"
    )


def test_delivery_errors_count_the_window_between_the_pause_and_the_resume() -> None:
    since = T0 + timedelta(seconds=40)
    until = T0 + timedelta(seconds=44)
    companion = sample(recovery.COMPANION_METRIC, None, 1.0)
    edge, summary, recovered = (
        batch_at(at_seconds(s), companion) for s in (41, 43, 45)
    )
    assert (
        recovery.delivery_errors({"batches": [edge, summary, recovered]}, since, until)
        == []
    ), "the edge and the summary inside the window are the claim"
    errors = recovery.delivery_errors({"batches": [edge, recovered]}, since, until)
    assert len(errors) == 1 and "only 1 host batch(es)" in errors[0], errors
    assert until.isoformat() in errors[0] and since.isoformat() in errors[0], errors
    assert recovery.delivery_errors({"batches": [edge, recovered]}, since) == [], (
        "without a bound the read's own moment closes the window"
    )


# --------------------------------------------------------------------------- #
# the recovery's since precedes the undo (live 2026-10-26 14:33Z)
# --------------------------------------------------------------------------- #
UNDONE = {
    "stat_timeout": lambda ctx: "resume" in servers_by_suffix()["a"].calls,
    "healthy_other_mount": lambda ctx: "resume" in servers_by_suffix()["a"].calls,
    "io_only_failure": lambda ctx: "mode-1777" in servers_by_suffix()["a"].calls,
    "mount_disappears": lambda ctx: any(
        call[0] == "nfs-remount" for call in ctx.probes[f"shared@{NODE}"].calls
    ),
    "capacity_full": lambda ctx: any(
        call[0] == "nfs-unfill" for call in ctx.probes[f"shared@{NODE}"].calls
    ),
}


@pytest.mark.parametrize("variant", sorted(UNDONE))
def test_the_recovery_since_is_stamped_before_the_undo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    """Live 2026-10-26 14:27-14:47Z (``mount_disappears``, plain EKS, R13g):
    umount receipt 14:33:25Z, unavailable=1 batch 14:33:28Z, the recovered edge
    14:33:43Z, remount receipt 14:33:45Z -- the variant stamped the recovery's
    ``since`` after the remount, so the only 0 reading the product ever wrote
    predated it; the health signal cleared nine minutes later with ``fresh``
    still empty (34 polls, FAIL). Replayed: the undo lands at T+45 s between a
    stamp taken before it (T+40 s) and one taken after it (T+50 s); every
    variant stamps before, so the recovered edge is fresh."""

    ctx = context(variant, tmp_path, activity=None)
    mount = shared.mount_path(ctx, shared.PRIMARY)
    other = (
        shared.mount_path(ctx, shared.OTHER)
        if variant == "healthy_other_mount"
        else None
    )
    metric = recovery.USED if variant == "capacity_full" else recovery.UNAVAILABLE
    mode = {
        "mount_disappears": shared.MODE_MOUNT_MISSING,
        "io_only_failure": "IO_ERROR",
    }.get(variant)
    documents = edge_only_documents(ctx, mount, other=other, metric=metric, mode=mode)
    undone = UNDONE[variant]

    def clock() -> datetime:
        return T0 + timedelta(seconds=50 if undone(ctx) else 40)

    monkeypatch.setattr(shared, "utc_now", clock)
    ctx.regional.cpu_python_responder = lambda _script, *_a, **_k: documents[
        "recovered" if undone(ctx) else "held"
    ]
    result = SPECS[variant].execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["stages"]["recovery"] == [], result["stages"]
    assert [item["observed_at"] for item in result["recovery"]["samples"]] == [
        at_seconds(45)
    ], "the recovered edge at T+45 s is fresh against a stamp from before the undo"


def test_await_clear_accepts_the_latest_record_with_a_cleared_signal(
    tmp_path: Path,
) -> None:
    """The live chronology as the product delivered it, read with the old
    stamp: unavailable=1 at T, the recovered edge 0 at T+15 s, ``since`` T+17 s,
    the health signal cleared at T+9 min by the summary that also moved the
    latest-metric record. No evidence 0 reading follows ``since`` (edge-only
    delivery), so the read accepts the latest record observed at or after
    ``since`` together with the cleared signal; without that record the signal
    alone proves nothing fresh and the budget runs out."""

    ctx = context("mount_disappears", tmp_path, activity=None)
    mount = shared.mount_path(ctx, shared.PRIMARY)
    key = signal_key(
        ctx.settings.regional.cluster_id, NODE, recovery.UNAVAILABLE, mount
    )
    since = T0 + timedelta(seconds=17)
    cleared_at = at_seconds(9 * 60)
    document = {
        "since": since.isoformat(),
        "batches": [
            batch_at(
                at_seconds(0),
                sample(
                    recovery.UNAVAILABLE, mount, 1.0, {"failure_mode": "MOUNT_MISSING"}
                ),
            ),
            batch_at(
                at_seconds(15),
                sample(recovery.UNAVAILABLE, mount, 0.0),
                sample(recovery.USED, mount, 0.0),
            ),
        ],
        "health_signals": {
            key: {"signal_key": key, "active": False, "observed_at": cleared_at}
        },
        "metric_latest": [
            {
                "name": recovery.UNAVAILABLE,
                "device": mount,
                "value": 0.0,
                "observed_at": cleared_at,
                "labels": {},
            }
        ],
        "incidents": [],
        "workflows": [],
        "markers": [],
        "commands": [],
        "notifications": [],
    }
    ctx.regional.cpu_python_responder = store_script({"activity": document})
    recovered = recovery.await_clear(ctx, NODE, mount, since)
    assert recovered["errors"] == [], recovered
    assert [(item["record"], item["observed_at"]) for item in recovered["samples"]] == [
        ("metric_latest", cleared_at)
    ], recovered["samples"]
    assert recovered["health_signal"]["active"] is False, recovered["health_signal"]
    document["metric_latest"] = []
    store_poll_clock(ctx)
    denied = recovery.await_clear(ctx, NODE, mount, since)
    assert denied["errors"] and "no 0 reading observed since" in denied["errors"][0], (
        denied
    )
