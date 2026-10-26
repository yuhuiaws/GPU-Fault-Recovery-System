"""GF-GENERAL-LIVE-005: what the product delivers after an injection and its undo.

The reads ``general_live_shared.py`` grades a hang's delivery and every
variant's recovery with, in a sibling module because the variants module sits
at the 1500-line file cap. They encode how the host collector ships a held
condition (``collectors/host/collector.py`` ``collect_once``): a breach is
delivered once, as its crossing's edge (``delivery_reasons = reasons -
_active_edge_reasons``), then only on the health summary
(``GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS``, +-5 % jitter), on another edge, or
as the ``recovered`` edge when the reason clears -- ticks under a steady hang
add no batch. The ingest side (``app/ingest/telemetry.py``
``_persist_host_telemetry``) keeps a batch as raw evidence when it carries a
finding, a collection error or a reason other than ``health-summary`` alone:
a summary during a hang (the finding rides along) is evidence, a summary after
the recovery is not and moves only the latest-metric records
(``LATEST_METRICS_REQUIRED``), the product's only per-mount witness in a quiet
window.

Two live verdicts on plain EKS (release R13g, 2026-10-26) were wrong about a
right product:

* ``healthy_other_mount`` 14:15-14:21Z: export paused 14:18:31Z, finding
  accepted 14:18:51Z, resumed 14:19:03Z. The delivery oracle counted the host
  batches between the pause and the finding and wanted two; a product that
  detects within 20 s leaves exactly one, the edge. ``hold_hang_until_delivered``
  keeps the export paused until a second batch after the pause is in the store
  -- the collector's next scheduled delivery -- and the variant resumes after it.
* ``mount_disappears`` 14:27-14:47Z: umount receipt 14:33:25Z, unavailable=1
  batch 14:33:28Z, remount, the recovered edge (unavailable=0 and a used %)
  at 14:33:43Z, remount receipt 14:33:45Z. The variant stamped the recovery's
  ``since`` after the remount, so the only 0 evidence the product ever writes
  predated it; nine minutes later the health signal cleared (observed
  14:42:54Z) with ``fresh`` still empty. Every caller now stamps ``since``
  before its undo, and ``await_clear`` also accepts the latest-metric 0 reading
  observed at or after ``since`` together with the cleared signal.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

from scripts.e2e.regional.general_live_common import (
    STORE_POLL_SECONDS,
    sustained_detection_budget,
)
from scripts.e2e.regional.general_live_fixture import VariantContext
from scripts.e2e.regional.general_live_store import node_activity, signal_key
from scripts.e2e.regional.general_live_verdicts import (
    latest_samples,
    parse_time,
    samples,
)

UNAVAILABLE = "shared_filesystem_unavailable"
USED = "shared_filesystem_used_percent"
#: A host metric every tick carries: proof the collector kept delivering while
#: one mount hung.
COMPANION_METRIC = "memory_used_percent"
ACTIVITY_METRICS = frozenset({UNAVAILABLE, USED, COMPANION_METRIC})
#: The capacity rule's threshold (``shared_filesystem_used_percent`` WARNING).
FULL_PERCENT = 98
#: Host batches the store must hold between the pause and the resume: the
#: breach's edge and the collector's next scheduled delivery.
DELIVERY_BATCHES = 2


def read_activity(
    ctx: VariantContext,
    node: str,
    since: datetime,
    *,
    mount: str | None = None,
    metrics: Iterable[str] = ACTIVITY_METRICS,
) -> dict[str, Any]:
    """The product's records of ``node`` since ``since`` (``mount``'s signal too)."""

    keys = (
        [signal_key(ctx.settings.regional.cluster_id, node, UNAVAILABLE, mount)]
        if mount
        else []
    )
    return node_activity(
        ctx.regional,
        ctx.settings.regional.cluster_id,
        node,
        since,
        metrics=metrics,
        signal_keys=keys,
    )


def baseline_readings(
    activity: dict[str, Any], metric: str, mount: str
) -> list[dict[str, Any]]:
    """The product's accepted readings of ``metric`` for ``mount`` inside the read.

    Evidence batch samples first, then the latest-wins record when it moved
    inside the window (``latest_samples``). A mount that appears *healthy* is
    no edge for the host collector (``_edge_reasons``: a threshold crossing, a
    collection error, the collector's own first tick, a cleared reason), so its
    first readings travel with the next health summary -- a batch
    ``app/ingest/telemetry.py`` by design never writes as raw evidence (no
    finding, no collection error). What that batch does move is the
    ``metric_latest`` record of every ``LATEST_METRICS_REQUIRED`` sample, both
    shared metrics among them: in a quiet window it is the product's only
    per-mount witness. Live 2026-10-26, both lanes, every 005 variant: the
    export was mounted and read healthy, the summaries moved those records, and
    an evidence-only read called the export "not discovered".
    """

    evidence = [
        {**item, "record": "evidence"}
        for item in samples(activity, metric, device=mount)
    ]
    latest = [
        {**item, "record": "metric_latest"}
        for item in latest_samples(activity, metric, device=mount)
    ]
    return evidence + latest


def _observed_within(
    items: Iterable[dict[str, Any]], since: datetime, until: datetime | None = None
) -> list[dict[str, Any]]:
    """The items whose ``observed_at`` falls in ``[since, until]``."""

    return [
        item
        for item in items
        if (moment := parse_time(item.get("observed_at"))) is not None
        and moment >= since
        and (until is None or moment <= until)
    ]


def delivery_errors(
    activity: dict[str, Any], since: datetime, until: datetime | None = None
) -> list[str]:
    """The collector kept delivering bounded batches while a mount hung.

    Counts the host batches the product accepted with ``observed_at`` from
    ``since`` (the pause) to ``until`` (the resume; open when None) and wants
    ``DELIVERY_BATCHES`` of them -- the breach's edge and the collector's next
    scheduled delivery -- one of them carrying ``COMPANION_METRIC``. The window
    is the caller's to hold open: see ``hold_hang_until_delivered``.
    """

    batches = _observed_within(activity.get("batches") or [], since, until)
    with_companion = [
        item
        for item in batches
        if any(
            sample.get("name") == COMPANION_METRIC
            for sample in item.get("samples") or []
        )
    ]
    errors = []
    if len(batches) < DELIVERY_BATCHES:
        window = f"between the pause at {since.isoformat()} and " + (
            f"the resume at {until.isoformat()}" if until is not None else "the read"
        )
        errors.append(
            f"only {len(batches)} host batch(es) were accepted after the injection "
            f"({window}): the collector's next scheduled delivery never arrived "
            "while the mount hung"
        )
    if not with_companion:
        errors.append(
            f"no host batch after the injection carried {COMPANION_METRIC}: the hung "
            "mount stalled the collector's other metrics"
        )
    return errors


def hold_hang_until_delivered(
    ctx: VariantContext, node: str, mount: str, since: datetime, *, name: str
) -> dict[str, Any]:
    """Keep the export paused until the store holds a second host batch after the pause.

    The host collector ships a held breach once -- its crossing is the edge --
    and then only on its health summary, so the window between the pause and
    the finding holds exactly one host batch whenever the product is quick
    (live 2026-10-26, plain EKS, ``healthy_other_mount``: paused 14:18:31Z,
    finding 14:18:51Z, resumed 14:19:03Z, one batch), and grading
    ``delivery_errors`` on it failed a product that detected within 20 s. The
    claim "the collector kept delivering while a mount hung" needs the hang
    held across the collector's next scheduled delivery: this polls the store,
    export still paused, until ``DELIVERY_BATCHES`` host batches observed at or
    after ``since`` are accepted -- the second is the summary, kept as evidence
    because the finding rides along -- for at most one summary period plus the
    interval and the margin (``sustained_detection_budget(ctx, 0, summaries=1)``,
    495 s on a 300 s / 15 s cadence). A collector the hung mount stalled never
    posts it, the budget runs out, and the caller's ``delivery_errors`` reports
    the single batch. Returns the last read; the caller resumes the export after
    it and grades the read, so no post-resume batch is counted as delivery
    under the hang. The polls are saved as ``<name>-timeline.json``.
    """

    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = read_activity(ctx, node, since, mount=mount)
        batches = _observed_within(last.get("batches") or [], since)
        return last if len(batches) >= DELIVERY_BATCHES else None

    value = ctx.wait_until(
        accept,
        timeout_seconds=sustained_detection_budget(ctx, 0, summaries=1),
        poll_seconds=STORE_POLL_SECONDS,
        name=name,
    )
    return value if value is not None else last


def await_clear(
    ctx: VariantContext, node: str, mount: str, since: datetime
) -> dict[str, Any]:
    """Fresh product evidence that ``mount`` reads available again.

    ``since`` is the caller's stamp from *before* its undo (remount, SIGCONT,
    chmod back): the collector ships the recovery as one ``recovered`` edge
    batch, the first tick that reads 0 after the undo, and then only summaries
    that move the latest-metric record. A tick landing between the undo's
    syscall and a stamp taken after the probe answered puts the only 0 evidence
    the product will ever write before ``since`` (live 2026-10-26 14:33Z,
    ``mount_disappears`` on plain EKS: recovered edge 14:33:43Z, remount
    receipt 14:33:45Z, no further sample in twelve minutes). Accepted: two 0
    readings of ``mount`` observed at or after ``since`` in evidence batches, or
    one such reading -- an evidence batch's or the latest-metric record's, which
    every accepted batch moves, summaries included -- together with the health
    signal cleared (``active`` false) at or after ``since``.
    """

    key = signal_key(ctx.settings.regional.cluster_id, node, UNAVAILABLE, mount)

    def accept() -> dict[str, Any] | None:
        current = read_activity(ctx, node, since, mount=mount)
        fresh = [
            item
            for item in _observed_within(
                baseline_readings(current, UNAVAILABLE, mount), since
            )
            if item.get("value") == 0
        ]
        evidence = [item for item in fresh if item.get("record") == "evidence"]
        signal = (current.get("health_signals") or {}).get(key) or {}
        observed = parse_time(signal.get("observed_at"))
        cleared = signal.get("active") is False and (
            observed is not None and observed >= since
        )
        if len(evidence) >= 2 or (fresh and cleared):
            return {"samples": fresh, "health_signal": signal}
        return None

    value = ctx.wait_until(
        accept,
        timeout_seconds=ctx.host_summary * 2 + ctx.host_interval * 6,
        poll_seconds=ctx.host_interval,
        name=f"recovery-{node}",
    )
    if value is None:
        return {
            "errors": [
                f"{UNAVAILABLE} on {mount} ({node}) did not recover: no 0 reading "
                f"observed since {since.isoformat()} (evidence batch or latest record) "
                "with a cleared health signal, nor two fresh evidence readings"
            ]
        }
    return {"errors": [], **value}


def await_used_recovery(
    ctx: VariantContext, node: str, mount: str, since: datetime
) -> dict[str, Any]:
    """``USED`` of ``mount`` read below ``FULL_PERCENT`` again after the unfill.

    ``since`` is stamped before the unfill (the drop is one ``recovered`` edge,
    then summaries moving the latest record -- see ``await_clear``); an
    evidence reading or the latest record observed at or after it counts.
    """

    def accept() -> dict[str, Any] | None:
        current = read_activity(ctx, node, since, mount=mount)
        fresh = [
            item
            for item in _observed_within(baseline_readings(current, USED, mount), since)
            if isinstance(item.get("value"), (int, float))
            and item["value"] < FULL_PERCENT
        ]
        return {"samples": fresh} if fresh else None

    value = ctx.wait_until(
        accept,
        timeout_seconds=ctx.host_summary * 2 + ctx.host_interval * 6,
        poll_seconds=ctx.host_interval,
        name="used-recovery",
    )
    if value is None:
        return {
            "errors": [
                f"{USED} on {mount} never read below {FULL_PERCENT} % after the unfill "
                f"(no evidence reading or latest record since {since.isoformat()})"
            ]
        }
    return {"errors": [], **value}
