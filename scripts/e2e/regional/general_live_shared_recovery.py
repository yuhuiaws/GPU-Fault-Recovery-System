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

The hold itself then failed a right product on HyperPod the same day
(``healthy_other_mount`` 19:46-19:55Z, release 37db33f254a1; plain EKS
``stat_timeout`` 15:30Z alike): it counted raw-evidence batches, and the
summary that re-reports an already claimed ``shared_filesystem_unavailable=1``
is never one -- ``evaluate_metrics`` mints a finding only on a claim
transition, so the batch carries no finding, no collection error and the
reason ``health-summary`` alone. The 19:51 summary (the node's phase, see
``summary_schedule``) moved only the HOST_TELEMETRY collector status and the
latest-metric rows; ``batches`` stayed at the 19:46:43 edge until the
``recovered`` edge after the resume. ``deliveries`` folds the three records
the store keeps of an accepted batch by the batch's instant, so a status
advance or a latest-row move under the hang is the second delivery; the two
latest-only records must be read while the export is still paused.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Any

from gpu_fault.collectors.cadence import DEFAULT_JITTER_RATIO
from gpu_fault.collectors.scheduling import next_stable_phase, stable_phase_seconds
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
#: The host collector's status channel and its batch ids: ``_record_collector_status``
#: upserts the HOST_TELEMETRY row for every accepted ``host-<node>-<micros>``
#: batch of the node's own collector, evidence or not; the control plane's
#: allocatable reader posts ``k8s-efa-`` batches on the same channel
#: (``general_live_verdicts.collector_delivery``).
HOST_COLLECTOR = "HOST_TELEMETRY"
HOST_BATCH_PREFIX = "host-"
#: The channel the host collector schedules its health summaries on
#: (``collectors/host/collector.py`` ``collect_once``: ``jittered_next_phase(...,
#: channel="host-telemetry", interval_seconds=health_summary_seconds)``).
SUMMARY_CHANNEL = "host-telemetry"
#: The three records the store keeps of one accepted host batch.
EVIDENCE_RECORD = "evidence"
STATUS_RECORD = "collector_status"
LATEST_RECORD = "metric_latest"


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


def deliveries(
    activity: dict[str, Any],
    since: datetime,
    until: datetime | None = None,
    *,
    mount: str | None = None,
) -> list[dict[str, Any]]:
    """The host batches accepted with ``observed_at`` in ``[since, until]``, from every record the store keeps.

    One entry per batch, folded by the batch's instant -- all three records
    carry ``batch.observed_at``, the tick's start, and the fold compares the
    parsed moments, not their spellings (the status prints ``...Z``, the latest
    rows ``...+00:00``) -- with the records that witness it:

    * ``evidence``: a raw host batch (``list_raw_evidence``), which
      ``app/ingest/telemetry.py`` ``_persist_host_telemetry`` writes only for a
      batch carrying a finding, a collection error or a reason other than
      ``health-summary``; ``evaluate_metrics`` mints a finding only on a claim
      transition, so the summary that re-reports a claimed breach never is one;
    * ``collector_status``: the node's HOST_TELEMETRY status
      (``_record_collector_status``: ``observed_at`` and ``batch_id`` of the
      newest accepted ``host-`` batch, summaries included);
    * ``metric_latest``: the latest-metric rows of ``COMPANION_METRIC`` and, when
      ``mount`` is given, of its ``UNAVAILABLE`` reading -- ``LATEST_METRICS_REQUIRED``
      rows every accepted batch moves to its ``observed_at`` (the hung mount's
      ``PROBE_TIMEOUT`` 1 gets a new stamp per summary).

    The last two keep only the newest value, so they witness a delivery while
    it is the newest: read them under the hang; after the resume they prove
    nothing about it. ``metrics`` lists the metric names a batch is known to
    carry (evidence samples, moved latest rows); a status alone names none.
    Live 2026-10-26 19:46Z (HyperPod ``healthy_other_mount``): the edge at
    19:46:43 was the only evidence batch until the resume; the 19:51:17 summary
    is the status advance and the latest rows at 19:51:17.
    """

    found: dict[datetime, dict[str, Any]] = {}

    def entry(observed_at: Any) -> dict[str, Any]:
        moment = parse_time(observed_at)
        assert moment is not None  # _observed_within kept only parseable stamps
        return found.setdefault(
            moment,
            {
                "observed_at": moment.isoformat(),
                "batch_id": None,
                "records": set(),
                "metrics": set(),
                "edge_filter_reasons": [],
            },
        )

    for item in _observed_within(activity.get("batches") or [], since, until):
        row = entry(item.get("observed_at"))
        row["records"].add(EVIDENCE_RECORD)
        row["batch_id"] = row["batch_id"] or item.get("batch_id")
        row["edge_filter_reasons"] = list(item.get("edge_filter_reasons") or [])
        row["metrics"].update(
            str(sample["name"])
            for sample in item.get("samples") or []
            if isinstance(sample, dict) and sample.get("name")
        )
    statuses = [
        item
        for item in activity.get("collector_statuses") or []
        if item.get("collector") == HOST_COLLECTOR
        and str(item.get("batch_id") or "").startswith(HOST_BATCH_PREFIX)
    ]
    for item in _observed_within(statuses, since, until):
        row = entry(item.get("observed_at"))
        row["records"].add(STATUS_RECORD)
        row["batch_id"] = row["batch_id"] or item.get("batch_id")
    wanted = [(COMPANION_METRIC, None)] + ([(UNAVAILABLE, mount)] if mount else [])
    for metric, device in wanted:
        latest = latest_samples(activity, metric, device=device)
        for item in _observed_within(latest, since, until):
            row = entry(item.get("observed_at"))
            row["records"].add(LATEST_RECORD)
            row["metrics"].add(metric)
    return [
        {**row, "records": sorted(row["records"]), "metrics": sorted(row["metrics"])}
        for _moment, row in sorted(found.items())
    ]


def delivery_errors(
    activity: dict[str, Any],
    since: datetime,
    until: datetime | None = None,
    *,
    mount: str | None = None,
) -> list[str]:
    """The collector kept delivering bounded batches while a mount hung.

    Counts the host batches the product accepted with ``observed_at`` from
    ``since`` (the pause) to ``until`` (the resume; open when None) --
    ``deliveries``: a raw-evidence batch, a HOST_TELEMETRY status advance or a
    latest-row move of ``mount``'s ``UNAVAILABLE`` / ``COMPANION_METRIC`` each
    witness one -- and wants ``DELIVERY_BATCHES`` of them, the breach's edge and
    the collector's next scheduled delivery, one of them known to carry
    ``COMPANION_METRIC``. The window is the caller's to hold open: see
    ``hold_hang_until_delivered``.
    """

    seen = deliveries(activity, since, until, mount=mount)
    errors = []
    if len(seen) < DELIVERY_BATCHES:
        window = f"between the pause at {since.isoformat()} and " + (
            f"the resume at {until.isoformat()}" if until is not None else "the read"
        )
        witness = (
            f"none moved past the batch at {seen[-1]['observed_at']} "
            f"({', '.join(seen[-1]['records'])})"
            if seen
            else "none observed in the window"
        )
        errors.append(
            f"only {len(seen)} host batch(es) were accepted after the injection "
            f"({window}): the collector's next scheduled delivery never arrived "
            "while the mount hung -- raw-evidence batches, the HOST_TELEMETRY "
            f"collector status and the latest-metric rows read: {witness}"
        )
    if not any(COMPANION_METRIC in item["metrics"] for item in seen):
        errors.append(
            f"no host batch after the injection carried {COMPANION_METRIC} (no "
            "evidence sample, no moved latest-metric row): the hung mount stalled "
            "the collector's other metrics"
        )
    return errors


def summary_schedule(
    ctx: VariantContext, node: str, since: datetime, budget_seconds: float
) -> dict[str, Any]:
    """Where ``node``'s health summaries fall during a hold that starts at ``since``.

    The host collector serves a summary on the first tick at or after the
    restart-stable phase of ``SUMMARY_CHANNEL`` on the summary interval
    (``stable_phase_seconds``, a hash of cluster, node and channel the harness
    can compute), moved by the cadence jitter of at most ``ratio x interval / 2``
    (``jittered_next_phase``; ``DEFAULT_JITTER_RATIO`` unless the node sets
    ``GPU_FAULT_COLLECTOR_CADENCE_JITTER_RATIO``, which the survey does not
    list). The next summary is therefore up to ``interval + jitter + tick``
    after any moment -- 330 s on a 300 s / 15 s cadence -- never "last delivery
    + interval". Live 2026-10-26, the HyperPod lane's node: phase 75 s,
    summaries 19:41:15 / 19:46:15 / 19:51:15 / 19:56:15, the one after the
    19:46:28 pause served by the 19:51:17 tick. ``due`` lists the phases inside
    the hold's budget, for the evidence.
    """

    interval = ctx.host_summary
    cluster_id = ctx.settings.regional.cluster_id
    due = []
    moment = next_stable_phase(
        since,
        cluster_id=cluster_id,
        node_id=node,
        channel=SUMMARY_CHANNEL,
        interval_seconds=interval,
    )
    while (moment - since).total_seconds() <= budget_seconds:
        due.append(moment.isoformat())
        moment += timedelta(seconds=interval)
    return {
        "channel": SUMMARY_CHANNEL,
        "interval_seconds": interval,
        "phase_seconds": stable_phase_seconds(
            cluster_id, node, SUMMARY_CHANNEL, interval
        ),
        "jitter_seconds": DEFAULT_JITTER_RATIO * interval / 2,
        "tick_seconds": ctx.host_interval,
        "due": due,
    }


def hold_hang_until_delivered(
    ctx: VariantContext, node: str, mount: str, since: datetime, *, name: str
) -> dict[str, Any]:
    """Keep the export paused until the store shows a second host batch after the pause.

    The host collector ships a held breach once -- its crossing is the edge --
    and then only on its health summary, so the window between the pause and
    the finding holds exactly one host batch whenever the product is quick
    (live 2026-10-26, plain EKS, ``healthy_other_mount``: paused 14:18:31Z,
    finding 14:18:51Z, resumed 14:19:03Z, one batch), and grading
    ``delivery_errors`` on it failed a product that detected within 20 s. The
    claim "the collector kept delivering while a mount hung" needs the hang
    held across the collector's next scheduled delivery: this polls the store,
    export still paused, until ``deliveries`` counts ``DELIVERY_BATCHES`` host
    batches observed at or after ``since``. The second is the summary, and the
    summary is no raw evidence (the finding does not ride along: the breach is
    already claimed), so it is read where the store keeps it -- the
    HOST_TELEMETRY collector status moved to the summary's ``observed_at`` /
    ``batch_id``, or the latest-metric row of the hung mount's ``UNAVAILABLE``
    (or of ``COMPANION_METRIC``) moved there -- both latest-only, hence read
    here, under the hang (live 2026-10-26 19:46Z, HyperPod: the 19:51:17
    summary moved both while ``batches`` kept the 19:46:43 edge alone). The
    budget is one summary period plus the interval and the margin
    (``sustained_detection_budget(ctx, 0, summaries=1)`` on the survey's
    ``GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS`` / ``GPU_FAULT_HOST_INTERVAL_SECONDS``,
    495 s on 300 s / 15 s): the summaries fall on the node's stable phase grid
    (``summary_schedule``), so the next one is up to interval + jitter + tick
    (330 s) after the pause, not interval after the edge; a collector the hung
    mount stalled moves no record, the budget runs out, and the caller's
    ``delivery_errors`` reports the single batch. Returns the last read; the
    caller resumes the export after it and grades that read, so no post-resume
    record is counted as delivery under the hang. The polls are saved as
    ``<name>-timeline.json``, the deliveries seen and the node's summary grid as
    ``<name>-deliveries.json``.
    """

    last: dict[str, Any] = {}
    budget = sustained_detection_budget(ctx, 0, summaries=1)
    schedule = summary_schedule(ctx, node, since, budget)

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = read_activity(ctx, node, since, mount=mount)
        seen = deliveries(last, since, mount=mount)
        ctx.evidence.write(
            f"{name}-deliveries",
            {
                "since": since.isoformat(),
                "wanted": DELIVERY_BATCHES,
                "deliveries": seen,
                "summary_schedule": schedule,
            },
        )
        return last if len(seen) >= DELIVERY_BATCHES else None

    value = ctx.wait_until(
        accept,
        timeout_seconds=budget,
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
