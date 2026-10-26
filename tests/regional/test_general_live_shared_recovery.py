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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.collectors.scheduling import stable_phase_seconds
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


# --------------------------------------------------------------------------- #
# the second delivery is a record the store keeps (live 2026-10-26 19:46Z)
# --------------------------------------------------------------------------- #
#: The HyperPod chronology (``healthy_other_mount``, release 37db33f254a1) in
#: seconds after the pause at 19:46:30Z: the ``threshold:`` edge batch
#: 19:46:43, the summary tick 19:51:17 (the node's 19:51:15 phase), then no
#: batch until the ``recovered`` edge 19:55:32 after the 19:55:24 SIGCONT.
EDGE_AT, SUMMARY_AT, RECOVERED_AT = 13, 287, 542
HUNG = {"source": "10.0.9.1:/a", "fstype": "nfs4", "failure_mode": "PROBE_TIMEOUT"}
HEALTHY = {"source": "10.0.9.1:/a", "fstype": "nfs4"}
OTHER_HEALTHY = {"source": "10.0.9.2:/b", "fstype": "nfs4"}
#: The read's own ``since`` in the replayed documents: live it is the pause
#: stamp, before the finding it causes; here before the fakes' incident stamp
#: (``_general_live_fakes.incident``, 2026-10-24T00:03Z, which
#: ``minted_in_read`` holds against it) and before every T0-based record.
READ_SINCE = "2026-10-24T00:00:00+00:00"


def zulu(seconds: float) -> str:
    """A collector-status stamp as the store prints it (``...Z``, not ``+00:00``)."""

    return (T0 + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def host_status(seconds: float, *, node: str = NODE) -> dict[str, Any]:
    """The node's HOST_TELEMETRY collector status after the batch at ``seconds``."""

    stamp = zulu(seconds)
    return {
        "batch_id": f"host-{node}-{seconds}",
        "cluster_id": "gpu-eks-cluster",
        "collector": recovery.HOST_COLLECTOR,
        "errors": [],
        "ingested_at": stamp,
        "last_error_at": None,
        "last_success_at": stamp,
        "node_id": node,
        "observed_at": stamp,
        "sample_count": 105,
    }


def latest_row(
    name: str,
    device: str | None,
    value: float,
    seconds: float,
    labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "device": device,
        "value": value,
        "observed_at": at_seconds(seconds),
        "labels": dict(labels or {}),
    }


def hpb_documents(
    ctx: Any, mount: str, *, other: str | None = None, proof: str = "both"
) -> dict[str, dict[str, Any]]:
    """The product's records as HyperPod kept them on 2026-10-26 19:46-19:56Z.

    ``detected``: the edge batch (the hung mount's ``PROBE_TIMEOUT`` 1 with
    every other sample of the tick -- the companion, the healthy neighbour),
    the HOST_TELEMETRY status and the latest rows all at the edge's instant.
    ``held``: the 19:51 summary -- no new batch; by ``proof`` the status
    advances (``collector_status``), the latest rows move (``metric_latest``)
    or both do (``both``, as live), the hung mount's row still at 1.
    ``recovered``: the ``recovered`` edge after the resume, every record at it,
    the health signal cleared. Every document names its ``since`` (the probe
    does), without which the latest rows would count for nothing.
    """

    other_ok = (
        [
            sample(recovery.UNAVAILABLE, other, 0.0, OTHER_HEALTHY),
            sample(recovery.USED, other, 0.0, OTHER_HEALTHY),
        ]
        if other
        else []
    )
    before = batch(
        -120,
        sample(recovery.UNAVAILABLE, mount, 0.0, HEALTHY),
        sample(recovery.USED, mount, 2.0, HEALTHY),
        *other_ok,
    )
    edge = {
        **batch_at(
            at_seconds(EDGE_AT),
            sample(recovery.UNAVAILABLE, mount, 1.0, HUNG),
            sample(recovery.COMPANION_METRIC, None, 2.6),
            *other_ok,
        ),
        "edge_filter_reasons": [f"threshold:{recovery.UNAVAILABLE}"],
    }
    recovered = {
        **batch_at(
            at_seconds(RECOVERED_AT),
            sample(recovery.UNAVAILABLE, mount, 0.0, HEALTHY),
            sample(recovery.USED, mount, 0.0, HEALTHY),
            *other_ok,
        ),
        "edge_filter_reasons": ["recovered"],
    }
    chain = diagnostic(mount, recovery.UNAVAILABLE, NODE)
    # The probe's document names its cluster and ``minted_in_read`` holds every
    # incident against it; the live incident carries the cluster id.
    chain["incident"]["cluster_id"] = ctx.settings.regional.cluster_id
    key = signal_key(
        ctx.settings.regional.cluster_id, NODE, recovery.UNAVAILABLE, mount
    )

    def rows(seconds: float, value: float) -> list[dict[str, Any]]:
        result = [
            latest_row(
                recovery.UNAVAILABLE, mount, value, seconds, HUNG if value else HEALTHY
            ),
            latest_row(recovery.COMPANION_METRIC, None, 2.6, seconds),
        ]
        if other:
            result.append(latest_row(recovery.UNAVAILABLE, other, 0.0, seconds))
            result.append(latest_row(recovery.USED, other, 0.0, seconds))
        return result

    def document(
        *batches: dict[str, Any],
        status_at: float,
        latest_at: float,
        value: float,
        cleared: bool,
    ) -> dict[str, Any]:
        signal = {
            "signal_key": key,
            "active": not cleared,
            "observed_at": at_seconds(RECOVERED_AT if cleared else EDGE_AT),
        }
        return {
            "since": READ_SINCE,
            "cluster_id": ctx.settings.regional.cluster_id,
            "node_id": NODE,
            "batches": list(batches),
            "incidents": [chain["incident"]],
            "workflows": [chain["workflow"]],
            "markers": [],
            "commands": [],
            "health_signals": {key: signal},
            "metric_latest": rows(latest_at, value),
            "collector_statuses": [host_status(status_at)],
            "notifications": [],
        }

    status_at = SUMMARY_AT if proof in {"both", "collector_status"} else EDGE_AT
    latest_at = SUMMARY_AT if proof in {"both", "metric_latest"} else EDGE_AT
    return {
        "detected": document(
            before, edge, status_at=EDGE_AT, latest_at=EDGE_AT, value=1.0, cleared=False
        ),
        "held": document(
            before,
            edge,
            status_at=status_at,
            latest_at=latest_at,
            value=1.0,
            cleared=False,
        ),
        "recovered": document(
            before,
            edge,
            recovered,
            status_at=RECOVERED_AT,
            latest_at=RECOVERED_AT,
            value=0.0,
            cleared=True,
        ),
    }


#: What witnesses the summary on the accepted read, by ``proof`` -- and what is
#: left witnessing the edge: the status and the latest rows keep only their
#: newest instant, so the record that moved no longer names the edge.
SUMMARY_RECORDS = {
    "both": [recovery.STATUS_RECORD, recovery.LATEST_RECORD],
    "collector_status": [recovery.STATUS_RECORD],
    "metric_latest": [recovery.LATEST_RECORD],
}
EDGE_RECORDS = {
    "both": [recovery.EVIDENCE_RECORD],
    "collector_status": [recovery.EVIDENCE_RECORD, recovery.LATEST_RECORD],
    "metric_latest": [recovery.STATUS_RECORD, recovery.EVIDENCE_RECORD],
}


@pytest.mark.parametrize("proof", sorted(SUMMARY_RECORDS))
@pytest.mark.parametrize("variant", ["stat_timeout", "healthy_other_mount"])
def test_the_hang_is_held_until_a_latest_only_record_shows_the_summary(
    tmp_path: Path, variant: str, proof: str
) -> None:
    """Live 2026-10-26 19:46-19:57Z (``healthy_other_mount``, HyperPod, release
    37db33f254a1): export paused 19:46:30Z, the ``threshold:`` edge batch
    19:46:43Z, the node's 19:51:15 summary served by the 19:51:17Z tick, SIGCONT
    19:55:24Z, the ``recovered`` edge 19:55:32Z -- and ``bounded_delivery``
    failed after 495 s because the summary that re-reports a claimed breach is
    never raw evidence, so ``batches`` kept the edge alone. The summary did move
    the HOST_TELEMETRY collector status and the latest-metric rows; the hold now
    reads those, one poll with the edge alone, the second accepted, the resume
    after it, whichever of the two records moved."""

    ctx = context(variant, tmp_path, activity=None)
    mount = shared.mount_path(ctx, shared.PRIMARY)
    other = (
        shared.mount_path(ctx, shared.OTHER)
        if variant == "healthy_other_mount"
        else None
    )
    documents = hpb_documents(ctx, mount, other=other, proof=proof)
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
        "the hold polled until a record of the summary after the pause was in the store"
    )
    seen = json.loads((ctx.case_dir / "hang-delivery-deliveries.json").read_text())
    assert [(item["observed_at"], item["records"]) for item in seen["deliveries"]] == [
        (at_seconds(EDGE_AT), EDGE_RECORDS[proof]),
        (at_seconds(SUMMARY_AT), SUMMARY_RECORDS[proof]),
    ], "the accepted read: the record that moved witnesses the summary, not the edge"
    assert seen["wanted"] == recovery.DELIVERY_BATCHES, seen
    assert seen["summary_schedule"]["channel"] == recovery.SUMMARY_CHANNEL, seen
    assert [stage for stage, paused in reads if paused] == [
        "detected",
        "detected",
        "held",
    ], "the detection read, one hold poll with the edge alone, the summary"
    server = servers_by_suffix()["a"]
    assert server.calls == ["create", "wait_ready", "pause", "resume"], server.calls
    assert reads.index(("held", True)) < reads.index(("recovered", False)), (
        "the resume came after the read that saw the summary's record"
    )


def test_records_pinned_at_the_edge_are_not_a_second_delivery(tmp_path: Path) -> None:
    """A collector the hang truly stalled: the status and the latest rows exist
    but stay at the edge's instant for the whole budget -- one batch, three
    records -- so the hold runs out (25 polls of 20 s over 495 s), the export is
    resumed anyway and ``bounded_delivery`` fails with the S33 sentence, now
    naming the records read and the last batch they witness; the evidence file
    keeps the single delivery and the node's summary grid."""

    ctx = store_poll_clock(context("stat_timeout", tmp_path, activity=None))
    mount = shared.mount_path(ctx, shared.PRIMARY)
    documents = hpb_documents(ctx, mount)

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
    assert (
        f"none moved past the batch at {at_seconds(EDGE_AT)} "
        "(collector_status, evidence, metric_latest)" in delivery[0]
    ), delivery
    assert result["stages"]["recovery"] == [], "resumed anyway, and it recovered"
    assert servers_by_suffix()["a"].paused is False, "resumed after the budget ran out"
    timeline = json.loads((ctx.case_dir / "hang-delivery-timeline.json").read_text())
    assert all(entry["accepted"] is False for entry in timeline["entries"]), timeline
    budget = sustained_detection_budget(ctx, 0, summaries=1)
    assert budget == 300 + 15 + 180, "one summary, the interval, the margin"
    assert len(timeline["entries"]) == -(-budget // STORE_POLL_SECONDS), (
        "one store poll every 20 s until the budget ran out"
    )
    seen = json.loads((ctx.case_dir / "hang-delivery-deliveries.json").read_text())
    assert [(item["observed_at"], item["records"]) for item in seen["deliveries"]] == [
        (
            at_seconds(EDGE_AT),
            [recovery.STATUS_RECORD, recovery.EVIDENCE_RECORD, recovery.LATEST_RECORD],
        )
    ], seen
    schedule = seen["summary_schedule"]
    assert schedule["phase_seconds"] == stable_phase_seconds(
        ctx.settings.regional.cluster_id, NODE, recovery.SUMMARY_CHANNEL, 300
    ), schedule
    assert (
        schedule["interval_seconds"] == 300
        and schedule["tick_seconds"] == 15
        and schedule["jitter_seconds"] == 15.0
    ), schedule
    since = datetime.fromisoformat(seen["since"])
    assert schedule["due"] and all(
        0 < (datetime.fromisoformat(due) - since).total_seconds() <= budget
        for due in schedule["due"]
    ), "the due phases listed are the ones inside the hold's budget"


def test_deliveries_fold_the_three_records_by_the_batch_instant() -> None:
    """One accepted batch, three records, two spellings: the status prints
    ``19:46:12.811331Z``, the latest rows ``19:46:12.811331+00:00`` (live
    baseline read, HyperPod 2026-10-26) -- folded by the parsed moment. Only the
    node's own ``host-`` status on HOST_TELEMETRY counts, only the hung mount's
    and the companion's latest rows, only stamps inside ``[since, until]``, and
    latest rows only in a document that names its ``since``."""

    since = T0 + timedelta(seconds=1)
    mount = "/mnt/gfrs-live/e031ac3e79e7/shared"
    edge = batch_at(
        at_seconds(EDGE_AT),
        sample(recovery.UNAVAILABLE, mount, 1.0, HUNG),
        sample(recovery.COMPANION_METRIC, None, 2.6),
    )
    document = {
        "since": at_seconds(0),
        "batches": [edge],
        "collector_statuses": [
            host_status(EDGE_AT),
            host_status(SUMMARY_AT),
            {**host_status(SUMMARY_AT + 2), "batch_id": f"k8s-efa-{NODE}-1"},
            {**host_status(SUMMARY_AT + 4), "collector": "GPU_METRICS"},
        ],
        "metric_latest": [
            latest_row(recovery.UNAVAILABLE, mount, 1.0, SUMMARY_AT, HUNG),
            latest_row(recovery.COMPANION_METRIC, None, 2.6, SUMMARY_AT),
            latest_row(recovery.UNAVAILABLE, "/mnt/other", 0.0, SUMMARY_AT + 6),
            latest_row(recovery.USED, mount, 0.0, SUMMARY_AT + 8),
        ],
    }
    seen = recovery.deliveries(document, since, mount=mount)
    assert [
        (item["observed_at"], item["batch_id"], item["records"], item["metrics"])
        for item in seen
    ] == [
        (
            at_seconds(EDGE_AT),
            edge["batch_id"],
            [recovery.STATUS_RECORD, recovery.EVIDENCE_RECORD],
            [recovery.COMPANION_METRIC, recovery.UNAVAILABLE],
        ),
        (
            at_seconds(SUMMARY_AT),
            f"host-{NODE}-{SUMMARY_AT}",
            [recovery.STATUS_RECORD, recovery.LATEST_RECORD],
            [recovery.COMPANION_METRIC, recovery.UNAVAILABLE],
        ),
    ], seen
    assert (
        seen[0]["edge_filter_reasons"] == [] and seen[1]["edge_filter_reasons"] == []
    ), "the reasons come from the evidence batch alone"
    assert recovery.delivery_errors(document, since, mount=mount) == [], (
        "the edge and the summary's status/latest rows are the two deliveries"
    )
    resumed = T0 + timedelta(seconds=SUMMARY_AT - 1)
    bounded = recovery.deliveries(document, since, resumed, mount=mount)
    assert [item["observed_at"] for item in bounded] == [at_seconds(EDGE_AT)], (
        "a status observed after the resume proves nothing about the hang"
    )
    errors = recovery.delivery_errors(document, since, resumed, mount=mount)
    assert len(errors) == 1 and "only 1 host batch(es) were accepted" in errors[0], (
        errors
    )
    assert resumed.isoformat() in errors[0] and at_seconds(EDGE_AT) in errors[0], errors
    nameless = {key: value for key, value in document.items() if key != "since"}
    assert [
        item["records"] for item in recovery.deliveries(nameless, since, mount=mount)
    ] == [
        [recovery.STATUS_RECORD, recovery.EVIDENCE_RECORD],
        [recovery.STATUS_RECORD],
    ], "a document that names no since yields no latest rows (latest_samples)"
    assert [item["records"] for item in recovery.deliveries(document, since)] == [
        [recovery.STATUS_RECORD, recovery.EVIDENCE_RECORD],
        [recovery.STATUS_RECORD, recovery.LATEST_RECORD],
    ], "without a mount the companion's latest row still witnesses the summary"


def test_delivery_errors_want_the_companion_in_some_record() -> None:
    """A status advance alone names no metric: the companion must be carried by
    an evidence sample or a moved latest row somewhere in the window."""

    since = T0 + timedelta(seconds=1)
    mount = "/mnt/gfrs-live/e031ac3e79e7/shared"
    document: dict[str, Any] = {
        "since": at_seconds(0),
        "batches": [
            batch_at(at_seconds(EDGE_AT), sample(recovery.UNAVAILABLE, mount, 1.0))
        ],
        "collector_statuses": [host_status(SUMMARY_AT)],
        "metric_latest": [],
    }
    errors = recovery.delivery_errors(document, since, mount=mount)
    assert len(errors) == 1 and recovery.COMPANION_METRIC in errors[0], errors
    assert "no moved latest-metric row" in errors[0], errors
    document["metric_latest"] = [
        latest_row(recovery.COMPANION_METRIC, None, 2.6, SUMMARY_AT)
    ]
    assert recovery.delivery_errors(document, since, mount=mount) == [], (
        "the companion's latest row moved with the summary"
    )
    assert recovery.delivery_errors({"since": at_seconds(0)}, since, mount=mount) == [
        "only 0 host batch(es) were accepted after the injection (between the pause "
        f"at {since.isoformat()} and the read): the collector's next scheduled "
        "delivery never arrived while the mount hung -- raw-evidence batches, the "
        "HOST_TELEMETRY collector status and the latest-metric rows read: none "
        "observed in the window",
        f"no host batch after the injection carried {recovery.COMPANION_METRIC} (no "
        "evidence sample, no moved latest-metric row): the hung mount stalled the "
        "collector's other metrics",
    ], "an empty store names both shortfalls"


def test_other_mount_errors_accept_the_latest_record_of_the_healthy_mount() -> None:
    """The summary under the hang carries the healthy neighbour at 0 but is no
    raw evidence; its latest record moved to a stamp after the pause is the
    reading, a record from before the pause or a 1 is not."""

    since = T0 + timedelta(seconds=1)
    other = "/mnt/gfrs-live/e031ac3e79e7/other"
    document: dict[str, Any] = {
        "since": at_seconds(0),
        "batches": [],
        "metric_latest": [latest_row(recovery.UNAVAILABLE, other, 0.0, SUMMARY_AT)],
    }
    assert shared.other_mount_errors(document, other, since) == [], (
        "the healthy mount's latest 0 record after the pause is the reading"
    )
    document["metric_latest"][0]["value"] = 1.0
    errors = shared.other_mount_errors(document, other, since)
    assert len(errors) == 1 and "no moved latest-metric record" in errors[0], errors
    stale = {
        "since": at_seconds(0),
        "batches": [],
        "metric_latest": [latest_row(recovery.UNAVAILABLE, other, 0.0, -5)],
    }
    assert shared.other_mount_errors(stale, other, since), (
        "a record from before the pause is no reading under the hang"
    )


def test_summary_schedule_names_the_node_phase_grid() -> None:
    """The product's own stable phase of the node's ``host-telemetry`` channel,
    the summary interval apart, jitter +-15 s, served by the next 15 s tick. Live
    2026-10-26 the HyperPod lane's node had phase 75 s -- summaries 19:41:15 /
    19:46:15 / 19:51:15 / 19:56:15, one due inside the 495 s budget after the
    19:46:28 pause, the 19:51:17Z summary the status recorded; the fakes' node
    hashes to 173 s, so the same pause leaves two phases inside the budget."""

    ctx = SimpleNamespace(
        host_summary=300,
        host_interval=15,
        settings=SimpleNamespace(
            regional=SimpleNamespace(cluster_id="gpu-eks-cluster")
        ),
    )
    pause = datetime(2026, 10, 26, 19, 46, 28, 229473, tzinfo=timezone.utc)
    schedule = recovery.summary_schedule(ctx, NODE, pause, 495)
    assert schedule == {
        "channel": "host-telemetry",
        "interval_seconds": 300,
        "phase_seconds": stable_phase_seconds(
            "gpu-eks-cluster", NODE, "host-telemetry", 300
        ),
        "jitter_seconds": 15.0,
        "tick_seconds": 15,
        "due": ["2026-10-26T19:47:53+00:00", "2026-10-26T19:52:53+00:00"],
    }, "the phases inside the budget sit on the node's grid, interval apart"
    assert schedule["phase_seconds"] == 173, "the fakes' node identity hashes to 173 s"
    longer = recovery.summary_schedule(ctx, NODE, pause, 900)
    assert longer["due"] == [
        "2026-10-26T19:47:53+00:00",
        "2026-10-26T19:52:53+00:00",
        "2026-10-26T19:57:53+00:00",
    ], "every phase inside a longer budget, one summary interval apart"
