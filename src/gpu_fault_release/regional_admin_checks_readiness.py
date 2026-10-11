"""Collector readiness for the ``control_api`` verification check.

The control plane's ``/v1/collector-readiness/{cluster_id}`` calls a node's
collector ready when its last successful post is younger than the kind's
silence threshold (``gpu_fault.collector_kinds.collector_silent_thresholds``:
180 s for the inventory, 420 s for metrics and host telemetry, 900 s for the
log, kernel and fabric kinds) and its systemd unit is running. The health
summaries post about every 300 s, so one post lost while the ``api-ha`` Pods
are being replaced -- the NLB still hands connections to a target that is going
away, and a collector's outbox replays only on its next successful live post --
shows up as an age of up to ~600 s at the first poll after the roll: above the
420 s threshold, so one node flips its whole cluster to not ready.

The verify used to judge that from the one snapshot it took seconds after the
roll and failed closed on it, rolling back releases whose deploy had completed
cleanly while the same endpoint reported every node ready a few minutes later.
The judgement here still fails closed, but first waits out one missed posting
cycle: a cluster that is not ready is re-asked -- only that cluster, only its
collector readiness -- every ``POST_ROLL_POLL_SECONDS`` until it is ready or
``POST_ROLL_GRACE_SECONDS`` has elapsed, and the grace never outlives the
release's own remaining time budget. A ``status`` read describes the fleet as
it is and gets no grace.

Whatever the verdict, the judged snapshot is kept -- per node and collector
kind: ``ready``, ``age_seconds`` and the threshold the control plane applied,
with how long the check waited and how many polls it made -- so a FAIL names
the stale node and collector instead of only the cluster.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from gpu_fault.admin.deadlines import DeploymentDeadlineExceeded, remaining_timeout
from gpu_fault_release.regional_release_config import CheckFailure, ReleaseError
from gpu_fault_release.regional_release_probes import probe_source
from gpu_fault_release.regional_release_runtime_identity import (
    CONTROL_PLANE_PYTHON,
    exec_cpu_ingress,
)

# One missed posting cycle of the slowest unconditional cadence (the ~300 s
# health summaries) plus the time a collector needs to post again once its
# outbox replays: the budget ``join-cluster`` gives a new cluster's first
# reports (``gpu_fault.admin.cluster_readiness.wait_collector_readiness``).
POST_ROLL_GRACE_SECONDS = 600.0
POST_ROLL_POLL_SECONDS = 10.0
# How many stale collectors the one-line summary names before counting the rest.
STALE_SUMMARY_LIMIT = 6

Fetch = Callable[[Any, list[str]], dict[str, Any]]


def grace_seconds(verify: bool) -> float:
    """The post-roll grace a report mode earns: a verify waits, ``status`` reads."""

    return POST_ROLL_GRACE_SECONDS if verify else 0.0


def fetch_collector_readiness(release: Any, cluster_ids: list[str]) -> dict[str, Any]:
    """Re-ask the control plane about ``cluster_ids`` only, in one exec.

    The same exec path as the full control API report, but the probe reads just
    ``/v1/collector-readiness/{cluster_id}`` for the clusters named and reports
    the silence thresholds the control plane applies.
    """

    result = json.loads(
        exec_cpu_ingress(
            release,
            arguments=(
                "env",
                "ADMIN_CLUSTER_IDS_JSON="
                + json.dumps(cluster_ids, separators=(",", ":")),
                CONTROL_PLANE_PYTHON,
                "-c",
                probe_source("collector_readiness_inspect"),
            ),
            failure="the collector readiness report",
            sensitive=True,
            interactive=False,
        )
    )
    if not isinstance(result, dict) or not isinstance(result.get("clusters"), dict):
        raise ReleaseError("collector readiness report returned non-object evidence")
    return result


def judge_collector_readiness(
    release: Any,
    report: dict[str, Any],
    *,
    grace_seconds: float,
    poll_seconds: float = POST_ROLL_POLL_SECONDS,
    fetch: Fetch = fetch_collector_readiness,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Judge every cluster's collector readiness in ``report``; fail closed.

    Returns the readiness evidence once every cluster is ready. Raises
    ``CheckFailure`` carrying the same evidence when a cluster is still not
    ready after the grace -- ``grace_seconds`` capped at the release's remaining
    time budget -- has run out; a grace of zero judges the snapshot in hand.
    """

    clusters = report.get("clusters") or {}
    snapshots: dict[str, Any] = {
        cluster_id: (cluster or {}).get("collector_readiness")
        for cluster_id, cluster in clusters.items()
    }
    thresholds = _thresholds(report)
    budget = _capped_grace(grace_seconds)
    started = monotonic()
    remaining = budget
    polls = 0
    probe_errors: list[str] = []
    pending = _not_ready(snapshots)
    while pending and remaining > 0:
        pause = min(poll_seconds, remaining)
        sleep(pause)
        remaining -= pause
        polls += 1
        try:
            fresh = fetch(release, pending)
        except TimeoutError as exc:
            # The release itself is out of time: the verdict is the last answer.
            probe_errors.append(str(exc))
            break
        except (ReleaseError, ValueError) as exc:
            # A probe that failed -- the ingress Pod it landed on was itself
            # being replaced, say -- is not an answer; the next poll asks again.
            probe_errors.append(str(exc))
        else:
            for cluster_id in pending:
                if cluster_id in fresh["clusters"]:
                    snapshots[cluster_id] = fresh["clusters"][cluster_id]
            thresholds = _thresholds(fresh) or thresholds
            pending = _not_ready(snapshots)
        # The probe's own duration counts against the grace as well.
        remaining = min(remaining, budget - (monotonic() - started))
    evidence = {
        "ready": not pending,
        "grace_seconds": budget,
        "requested_grace_seconds": grace_seconds,
        "poll_seconds": poll_seconds,
        "polls": polls,
        "waited_seconds": round(max(0.0, monotonic() - started), 3),
        "probe_errors": probe_errors,
        "stale": [
            line
            for cluster_id in pending
            for line in _stale(cluster_id, snapshots[cluster_id], thresholds)
        ],
        "clusters": {
            cluster_id: _annotate(snapshot, thresholds)
            for cluster_id, snapshot in snapshots.items()
        },
    }
    if pending:
        raise CheckFailure(_failure_message(pending, evidence), evidence)
    return evidence


def _capped_grace(grace: float) -> float:
    """``grace`` or what is left of the release's time budget, whichever is less."""

    if grace <= 0:
        return 0.0
    try:
        return remaining_timeout(grace)
    except DeploymentDeadlineExceeded:
        return 0.0


def _not_ready(snapshots: dict[str, Any]) -> list[str]:
    return [
        cluster_id
        for cluster_id, snapshot in snapshots.items()
        if not (isinstance(snapshot, dict) and snapshot.get("ready") is True)
    ]


def _thresholds(document: dict[str, Any]) -> dict[str, float]:
    """The silence thresholds a probe reported, by collector kind, in seconds."""

    raw = document.get("collector_thresholds")
    if not isinstance(raw, dict):
        return {}
    thresholds: dict[str, float] = {}
    for kind, value in raw.items():
        try:
            thresholds[str(kind)] = float(value)
        except (TypeError, ValueError):
            continue
    return thresholds


def _annotate(snapshot: Any, thresholds: dict[str, float]) -> Any:
    """The snapshot with the threshold the control plane applied beside each age."""

    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("nodes"), list):
        return snapshot
    nodes = []
    for node in snapshot["nodes"]:
        collectors = node.get("collectors") if isinstance(node, dict) else None
        if not isinstance(collectors, dict):
            nodes.append(node)
            continue
        nodes.append(
            {
                **node,
                "collectors": {
                    kind: (
                        {**collector, "threshold_seconds": thresholds.get(kind)}
                        if isinstance(collector, dict)
                        else collector
                    )
                    for kind, collector in collectors.items()
                },
            }
        )
    return {**snapshot, "nodes": nodes}


def _stale(cluster_id: str, snapshot: Any, thresholds: dict[str, float]) -> list[str]:
    """One line per collector that holds ``cluster_id`` back, or why none can be named."""

    if not isinstance(snapshot, dict):
        return [f"{cluster_id}: no collector readiness answer"]
    nodes = snapshot.get("nodes")
    if not isinstance(nodes, list):
        return [f"{cluster_id}: no node evidence in the readiness answer"]
    if not nodes:
        return [f"{cluster_id}: no ACTIVE Agent has reported"]
    lines = []
    for node in nodes:
        if not isinstance(node, dict) or node.get("ready") is True:
            continue
        collectors = node.get("collectors")
        for kind, collector in (
            collectors.items() if isinstance(collectors, dict) else ()
        ):
            if isinstance(collector, dict) and collector.get("ready") is not True:
                reason = _reason(collector, thresholds.get(kind))
                lines.append(f"{cluster_id}/{node.get('node_id')}/{kind}: {reason}")
    return lines or [f"{cluster_id}: not ready"]


def _reason(collector: dict[str, Any], threshold: float | None) -> str:
    unit_state = collector.get("unit_state")
    if unit_state not in (None, "active", "unknown"):
        return f"unit {collector.get('unit')} is {unit_state}"
    age = collector.get("age_seconds")
    if threshold is None:
        limit = ", threshold unknown"
        return (
            "never reported" + limit if age is None else f"age {float(age):.0f}s{limit}"
        )
    if age is None:
        return f"never reported (threshold {threshold:g}s)"
    return f"age {float(age):.0f}s exceeds threshold {threshold:g}s"


def _failure_message(pending: list[str], evidence: dict[str, Any]) -> str:
    stale = evidence["stale"]
    shown = "; ".join(stale[:STALE_SUMMARY_LIMIT])
    if len(stale) > STALE_SUMMARY_LIMIT:
        shown += f"; +{len(stale) - STALE_SUMMARY_LIMIT} more"
    waited = ""
    if evidence["polls"]:
        waited = (
            f" after {evidence['waited_seconds']:.0f}s of post-roll grace "
            f"({evidence['polls']} polls)"
        )
    return f"{', '.join(pending)} collector readiness failed{waited}: {shown}"
