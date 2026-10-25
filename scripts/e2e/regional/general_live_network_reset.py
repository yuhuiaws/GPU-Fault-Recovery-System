"""GF-GENERAL-LIVE-009 ``counter_reset``: a recreated drop interface is a reset
tick, never a recovery and never a new fault.

Since batch 173 the netdev reader (``collectors/host/network.py``
``_monotonic_delta``/``_reset_problem``) treats a counter below its own
baseline as unknown for that tick: a ``_network`` collection error naming the
interface, the counter and ``delta unknown``, no ``network_drops_delta`` for
the interface in that batch, the baseline re-armed from the new value. The
collector's delivery contract makes that observable without luck: a collection
error is an edge (``collectors/host/collector.py`` ``_edge_reasons``), so the
reset tick is delivered and stored as raw evidence; the tick after a fresh
burst on the recreated pair is a ``threshold:network_drops_delta`` edge, stored
with the delta counted from the new baseline. The old clamp -- the fallen
counter shipped as a 0 delta, read as a recovery -- is observable the same
way: a reading after the recreate with no reset error ever named.

Baselines are keyed by interface *name* and pruned when a tick does not list
the name, so the pair must never be absent while a tick runs. The recreate is
therefore one probe command (``general_live_net_probe.py`` ``recreate-veth``:
``ip link del`` then ``ip link add`` in the same process, the gap measured in
milliseconds) instead of two probe round trips (10-15 s each live, most of a
collector interval). A tick inside the gap would ship CRITICAL
``network_link_down`` for both required ends and quarantine the node; that is
the product's own witness of a straddled recreate, graded INCONCLUSIVE after
the controlled readmission, never FAIL.

The stimulus is the runner's own: the netdev oracle must see the pair's drop
counters rise before the recreate and again on the recreated pair; the
probe's output must show the interface index change. The product's side is
read with cadence-derived budgets, never a single read after a fixed hold.

Live 2026-10-25 (HyperPod lane, release d5d7610460e0, harness 26f860e4): the
previous version recreated the pair in two round trips, read the store once
60 s after the recreate and found neither the reset error nor a reading --
INCONCLUSIVE, with no record of what the collector or the counters did. The
EKS PASS of 2026-10-24 (release bf1bb0037847) came from the earlier oracle,
which graded no hardware step only and noted the clamp as an observation: it
did not test the reset rule, so it did not depend on timing either.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from scripts.e2e.regional.general_live_common import (
    Oracle,
    activity,
    await_oracle_crossing,
    close_window,
    restore_incidents,
)
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    utc_now,
)
from scripts.e2e.regional.general_live_verdicts import (
    ChainExpectation,
    chain_errors,
    finding_incidents,
    hardware_errors,
    isolation_operations,
    node_isolated,
    oracle_values,
    parse_time,
    samples,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError

RESET_METRIC = "network_drops_delta"
LINK_METRIC = "network_link_down"
#: The burst before the recreate: long enough for its edge to be shipped,
#: shorter than the drops rule's 60-s sustain so it mints no finding.
BURST_SECONDS = 45
#: The burst on the recreated pair: its first positive delta from the new
#: baseline proves the re-arm; it ends before a second active sample could
#: exist, so it mints no finding either.
RAMP_SECONDS = 30
RATE = 10
#: How long the generator may take to move the oracle's drop counters.
RAMP_BUDGET_SECONDS = 90
#: Node and runner clocks may differ by a few seconds; used only when the
#: probe did not report the recreate's own node-clock time.
CLOCK_TOLERANCE_SECONDS = 5
HOST_COLLECTOR = "HOST_TELEMETRY"
HOST_BATCH_PREFIX = "host-"


def _network() -> Any:
    """The 008/009 module (imported lazily: it registers this variant)."""

    from scripts.e2e.regional import general_live_network

    return general_live_network


def _moment(value: Any) -> datetime | None:
    try:
        moment = parse_time(value)
    except ValueError:
        return None
    if moment is not None and moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def edge_budget(ctx: VariantContext) -> int:
    """How long an edge the collector must deliver at once may take to be on record.

    A threshold crossing, a recovered reason and a collection error are all
    delivered on the tick that sees them; the budget covers slow ticks (the
    tick's contributors are minutes on a sick host), the ingest and the
    outbox's retry ladder: six collector intervals plus a minute (150 s on a
    15-s site).
    """

    return ctx.host_interval * 6 + 60


def oracle_seconds(ctx: VariantContext) -> int:
    """The netdev oracle outlives every wait of the variant."""

    budget = edge_budget(ctx)
    return min(
        3600, RAMP_BUDGET_SECONDS * 2 + budget * 5 + BURST_SECONDS + RAMP_SECONDS
    )


def dropped_frames(a: str, b: str) -> Callable[[dict[str, Any]], Any]:
    """One oracle record's drop count: ``a``'s tx_dropped plus ``b``'s rx_dropped."""

    def read(record: dict[str, Any]) -> Any:
        netdev = record.get("netdev") or {}
        counters = [
            (netdev.get(a) or {}).get("tx_dropped"),
            (netdev.get(b) or {}).get("rx_dropped"),
        ]
        values = [value for value in counters if isinstance(value, (int, float))]
        return sum(values) if values else None

    return read


def frames_since(a: str, b: str, since_epoch: float) -> Callable[[dict[str, Any]], Any]:
    """``dropped_frames`` for the records taken at or after ``since_epoch``."""

    frames = dropped_frames(a, b)

    def read(record: dict[str, Any]) -> Any:
        moment = record.get("t")
        if not isinstance(moment, (int, float)) or moment < since_epoch:
            return None
        return frames(record)

    return read


def reset_evidence(
    observed: dict[str, Any], device: str, since: Any
) -> dict[str, list[Any]]:
    """What the product recorded about ``device`` after a counter reset.

    ``errors``: the ``_network`` collection errors naming the interface, a
    counter ``reset`` and ``delta unknown`` (``collectors/host/network.py``
    ``_reset_problem``) in batches observed since ``since``; ``reset_tick``:
    ``network_drops_delta`` samples of the interface shipped in those very
    batches (the rule is no delta for that tick); ``readings``: every drops
    sample of the interface since ``since``.
    """

    errors: list[str] = []
    reset_tick: list[dict[str, Any]] = []
    readings: list[dict[str, Any]] = []
    for batch in observed.get("batches") or []:
        moment = _moment(batch.get("observed_at"))
        if moment is None or moment < since:
            continue
        drops = [
            {**sample, "observed_at": batch.get("observed_at")}
            for sample in batch.get("samples") or []
            if sample.get("name") == RESET_METRIC and sample.get("device") == device
        ]
        readings.extend(drops)
        named = [
            str(error)
            for error in batch.get("collection_errors") or []
            if "_network" in str(error)
            and f"{device}:" in str(error)
            and "reset" in str(error)
            and "delta unknown" in str(error)
        ]
        if named:
            errors.extend(named)
            reset_tick.extend(drops)
    return {"errors": errors, "reset_tick": reset_tick, "readings": readings}


def drops_samples(
    document: dict[str, Any],
    devices: Sequence[str],
    *,
    after: datetime,
    positive: bool,
) -> list[dict[str, Any]]:
    """Accepted ``network_drops_delta`` samples of ``devices`` observed after
    ``after``: the positive ones (an edge the burst shipped) or the zeros (the
    recovered edge after it)."""

    result = []
    for device in devices:
        for sample in samples(document, RESET_METRIC, device=device):
            moment = _moment(sample.get("observed_at"))
            value = sample.get("value")
            if moment is None or moment <= after:
                continue
            if not isinstance(value, (int, float)):
                continue
            if (value > 0) if positive else (value == 0):
                result.append(sample)
    return sorted(result, key=lambda item: str(item["observed_at"]))


def link_down_incidents(
    document: dict[str, Any], *devices: str
) -> list[dict[str, Any]]:
    """``network_link_down`` incidents on the pair: a tick saw it missing."""

    return [
        item
        for device in devices
        for item in finding_incidents(document, LINK_METRIC, device=device)
    ]


def host_delivery_after(
    document: dict[str, Any], after: datetime
) -> dict[str, Any] | None:
    """The node's host collector status when its newest accepted batch postdates ``after``."""

    for item in document.get("collector_statuses") or []:
        if item.get("collector") != HOST_COLLECTOR:
            continue
        if not str(item.get("batch_id") or "").startswith(HOST_BATCH_PREFIX):
            continue
        moment = _moment(item.get("last_success_at") or item.get("observed_at"))
        if moment is not None and moment > after:
            return dict(item)
    return None


def oracle_summary(
    records: Sequence[dict[str, Any]], a: str, b: str, anchor_epoch: float
) -> dict[str, Any]:
    """The runner's view of both bursts and of the recreate itself."""

    frames = dropped_frames(a, b)
    before = [
        item
        for item in records
        if isinstance(item.get("t"), (int, float)) and item["t"] < anchor_epoch
    ]
    after = [
        item
        for item in records
        if isinstance(item.get("t"), (int, float)) and item["t"] >= anchor_epoch
    ]
    pre = oracle_values(before, frames)
    post = oracle_values(after, frames)
    indexes_before = [
        (item.get("netdev") or {}).get(a, {}).get("ifindex") for item in before
    ]
    indexes_after = [
        (item.get("netdev") or {}).get(a, {}).get("ifindex") for item in after
    ]
    return {
        "samples_before": len(before),
        "samples_after": len(after),
        "pre_reset_drops": (max(pre) - min(pre)) if pre else None,
        "post_reset_drops": max(post) if post else None,
        "ifindex_before": sorted({str(i) for i in indexes_before if i is not None}),
        "ifindex_after": sorted({str(i) for i in indexes_after if i is not None}),
        "absent_samples": sum(
            1
            for item in after
            if ((item.get("netdev") or {}).get(a) or {}).get("present") is False
        ),
    }


def rearm_errors(
    a: str,
    b: str,
    *,
    rearmed: Sequence[dict[str, Any]],
    zeros_after: Sequence[dict[str, Any]],
    readings: Sequence[dict[str, Any]],
    post_reset_drops: float | None,
    budget: int,
) -> list[str]:
    """The baseline re-armed from the new value: a positive delta from the
    recreated pair, never negative, never more than the oracle counted there,
    then a zero once the second burst ended."""

    errors = []
    if not rearmed:
        errors.append(
            f"no positive {RESET_METRIC} from the recreated {a}/{b} reached the "
            f"product within {budget} s of the second burst, although the oracle "
            f"counted {post_reset_drops} dropped frames on the new pair"
        )
    negative = [
        item["value"]
        for item in readings
        if isinstance(item.get("value"), (int, float)) and item["value"] < 0
    ]
    if negative:
        errors.append(f"negative {RESET_METRIC} shipped after the recreate: {negative}")
    if post_reset_drops is not None:
        too_big = [
            item["value"]
            for item in rearmed
            if isinstance(item.get("value"), (int, float))
            and item["value"] > post_reset_drops
        ]
        if too_big:
            errors.append(
                f"{RESET_METRIC} {too_big} after the recreate exceeds the "
                f"{post_reset_drops} frames the oracle counted on the new pair: not "
                "a delta from the new baseline"
            )
    if rearmed and not zeros_after:
        errors.append(
            f"no zero {RESET_METRIC} followed the second burst within {budget} s"
        )
    return errors


def _burst(ctx: VariantContext, seconds: int, name: str) -> None:
    started = ctx.net(
        "start-drops", "--slot", "a", "--rate", str(RATE), "--seconds", str(seconds)
    )
    ctx.cleanup.push("net-drops", lambda: ctx.net("stop-drops"))
    ctx.evidence.write(name, {**started, "seconds": seconds, "rate": RATE})


def _stop_burst(ctx: VariantContext) -> None:
    ctx.net("stop-drops")
    ctx.cleanup.discard("net-drops")


def _await_samples(
    ctx: VariantContext,
    started: datetime,
    devices: Sequence[str],
    *,
    after: datetime,
    positive: bool,
    budget: int,
    name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    last: dict[str, Any] = {}
    found: list[dict[str, Any]] = []

    def accept() -> dict[str, Any] | None:
        nonlocal last, found
        last = activity(ctx, started, metrics={RESET_METRIC, LINK_METRIC})
        found = drops_samples(last, devices, after=after, positive=positive)
        return last if found else None

    value = ctx.wait_until(
        accept, timeout_seconds=budget, poll_seconds=ctx.host_interval, name=name
    )
    return (value if value is not None else last), found


def _await_reset(
    ctx: VariantContext,
    started: datetime,
    a: str,
    b: str,
    *,
    anchor: datetime,
    budget: int,
) -> dict[str, Any]:
    """Until the product recorded the reset tick (or a tick that saw the pair missing)."""

    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = activity(ctx, started, metrics={RESET_METRIC, LINK_METRIC})
        evidence = reset_evidence(last, a, anchor)
        if (
            evidence["errors"]
            or evidence["readings"]
            or link_down_incidents(last, a, b)
        ):
            return last
        return None

    value = ctx.wait_until(
        accept,
        timeout_seconds=budget,
        poll_seconds=ctx.host_interval,
        name="reset-tick",
    )
    return value if value is not None else last


def _straddled(
    ctx: VariantContext,
    incidents: Sequence[dict[str, Any]],
    *,
    a: str,
    b: str,
    since: datetime,
    gap_ms: Any,
) -> VariantInconclusive:
    """A tick saw the pair missing: readmit the node, then say so.

    The pair is back and up, so the collector's next readings clear the
    link-down edge; those must be on record before the window closes
    (``_await_pair_up``), or the validated restore judges the stale "missing".
    """

    net = _network()
    pair_errors = net._await_pair_up(ctx, since, a, b)
    close_window(ctx)
    ctx.cleanup.discard("collector-window")
    restored = restore_incidents(
        ctx, incidents, reason="GF-GENERAL-LIVE-009 counter_reset straddled recreate"
    )
    return VariantInconclusive(
        f"the recreate straddled a collector tick (pair absent for {gap_ms} ms): the "
        f"product shipped {LINK_METRIC} for the deleted pair and isolated the node "
        f"({[item['incident_id'] for item in incidents]}); readmitted through its "
        f"validated restore ({len(restored)} workflow(s), pair-up errors "
        f"{pair_errors}) -- not a verdict on the reset rule"
    )


def _recreate(ctx: VariantContext) -> dict[str, Any]:
    recreated = ctx.net(
        "recreate-veth",
        "--slot",
        "a",
        "--peer-slot",
        "b",
        "--deadman-seconds",
        str(ctx.spec.max_minutes * 60 + 600),
    )
    ctx.evidence.write("recreate", recreated)
    old_index = ((recreated.get("before") or {}).get("a") or {}).get("ifindex")
    new_index = (recreated.get("link_a") or {}).get("ifindex")
    if old_index is None or new_index is None or str(old_index) == str(new_index):
        raise RegionalFixtureError(
            f"recreate-veth did not replace the interface: ifindex {old_index} -> "
            f"{new_index}"
        )
    if (recreated.get("link_a") or {}).get("operstate") != "up":
        raise RegionalFixtureError(f"the recreated pair is not up: {recreated}")
    return recreated


def _burst_phase(
    ctx: VariantContext,
    started: datetime,
    a: str,
    b: str,
    oracle: Oracle,
    *,
    budget: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], datetime]:
    """The first burst: the oracle sees the pair's counters rise, the product
    ships the burst's edge and then its recovered zero. Returns the positive
    samples, the zeros and the moment of the first zero (the signal is clear
    from there, so the recreate cannot fall inside an active window)."""

    _burst(ctx, BURST_SECONDS, "reset-burst")
    await_oracle_crossing(
        ctx,
        oracle,
        dropped_frames(a, b),
        lambda values: len(values) >= 2 and values[-1] > values[0],
        timeout_seconds=RAMP_BUDGET_SECONDS,
        name="reset-burst-ramp",
        condition=f"the generator did not move the drop counters of {a}/{b}",
    )
    shipped, positives = _await_samples(
        ctx,
        started,
        (a, b),
        after=started,
        positive=True,
        budget=budget,
        name="reset-edge",
    )
    _stop_burst(ctx)
    if not positives:
        delivery = host_delivery_after(shipped, started)
        raise VariantInconclusive(
            f"the product shipped no positive {RESET_METRIC} for {a}/{b} within "
            f"{budget} s of the burst (host collector last success "
            f"{(delivery or {}).get('last_success_at') or 'none since the start'}); "
            "the baseline the recreate must fall below is unknown"
        )
    # The burst's edge is its first positive sample; a later poll may already
    # list the second burst's.
    edge_at = min(_moment(item["observed_at"]) or started for item in positives)
    _, zeros = _await_samples(
        ctx,
        started,
        (a, b),
        after=edge_at,
        positive=False,
        budget=budget,
        name="reset-quiet",
    )
    if not zeros:
        raise VariantInconclusive(
            f"the burst's recovered edge (a zero {RESET_METRIC} for {a}/{b}) did not "
            f"arrive within {budget} s of its last positive reading at "
            f"{edge_at.isoformat()}; the signal state before the recreate is unknown"
        )
    zero_at = min(_moment(item["observed_at"]) or edge_at for item in zeros)
    return positives, zeros, zero_at


def _anchor(
    ctx: VariantContext,
    recreated: dict[str, Any],
    *,
    reset_at: datetime,
    zero_at: datetime,
) -> tuple[datetime, float]:
    """Where the product's post-recreate batches start, and where the oracle's do.

    The reset tick postdates the recreate on the node's clock; a tick that
    started just before the add and read the pair just after it is stamped up
    to one interval earlier. Nothing between the burst's zero and the recreate
    is stored, so the anchor loses nothing by reaching back. The oracle's own
    samples split at the recreate itself: a pre-recreate sample inside the
    reach-back still shows the old counters.
    """

    node_clock = _moment(recreated.get("recreated_at"))
    if node_clock is not None:
        anchor = max(zero_at, node_clock - timedelta(seconds=ctx.host_interval))
    else:
        anchor = max(zero_at, reset_at - timedelta(seconds=CLOCK_TOLERANCE_SECONDS))
    return anchor, (node_clock or reset_at).timestamp()


def _ramp_phase(
    ctx: VariantContext,
    started: datetime,
    a: str,
    b: str,
    oracle: Oracle,
    *,
    anchor: datetime,
    recreate_epoch: float,
    budget: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The second burst on the recreated pair: the oracle sees the new
    counters rise, the product ships a positive delta from the new baseline
    and, once the burst ended, a zero."""

    _burst(ctx, RAMP_SECONDS, "reset-ramp")
    await_oracle_crossing(
        ctx,
        oracle,
        frames_since(a, b, recreate_epoch),
        lambda values: len(values) >= 2 and values[-1] > values[0],
        timeout_seconds=RAMP_BUDGET_SECONDS,
        name="reset-ramp-ramp",
        condition=(
            f"the generator did not move the drop counters of the recreated {a}/{b}"
        ),
    )
    _, rearmed = _await_samples(
        ctx,
        started,
        (a, b),
        after=anchor,
        positive=True,
        budget=budget,
        name="reset-rearmed",
    )
    _stop_burst(ctx)
    zeros_after: list[dict[str, Any]] = []
    if rearmed:
        last_rearmed = max(_moment(item["observed_at"]) or anchor for item in rearmed)
        _, zeros_after = _await_samples(
            ctx,
            started,
            (a, b),
            after=last_rearmed,
            positive=False,
            budget=budget,
            name="reset-rearmed-quiet",
        )
    return rearmed, zeros_after


def _reset_unknown(
    a: str, evidence: dict[str, list[Any]], delivery: dict[str, Any] | None
) -> list[str]:
    """The ``reset_unknown`` stage: the reset tick named, or what came instead."""

    if evidence["errors"]:
        return []
    if evidence["readings"]:
        return [
            f"no _network collection error names a {a} counter reset with 'delta "
            f"unknown' while {len(evidence['readings'])} {RESET_METRIC} reading(s) "
            f"were shipped after the recreate: the old clamp"
        ]
    return [
        f"the product accepted host batches after the recreate (last success "
        f"{(delivery or {}).get('last_success_at')}) but recorded neither a reset "
        f"error nor a {RESET_METRIC} reading for {a}"
    ]


def counter_reset(ctx: VariantContext) -> dict[str, Any]:
    """The drop interface is deleted and recreated between ticks: its counters
    fall below the collector's baseline (see the module docstring)."""

    net = _network()
    created = net._veth(ctx)
    a, b = str(created["a"]), str(created["b"])
    net._declare_required(ctx, a, b)
    ctx.hold(ctx.host_interval * 3, name="healthy-samples")
    started = utc_now()
    budget = edge_budget(ctx)
    oracle = Oracle(ctx, "net")
    oracle.start(oracle_seconds(ctx), interval=5, netdev=[a, b])
    positives, zeros, zero_at = _burst_phase(ctx, started, a, b, oracle, budget=budget)
    reset_at = utc_now()
    recreated = _recreate(ctx)
    anchor, recreate_epoch = _anchor(ctx, recreated, reset_at=reset_at, zero_at=zero_at)
    waited = _await_reset(ctx, started, a, b, anchor=anchor, budget=budget)
    straddled = link_down_incidents(waited, a, b)
    if straddled:
        raise _straddled(
            ctx, straddled, a=a, b=b, since=reset_at, gap_ms=recreated.get("gap_ms")
        )
    rearmed, zeros_after = _ramp_phase(
        ctx,
        started,
        a,
        b,
        oracle,
        anchor=anchor,
        recreate_epoch=recreate_epoch,
        budget=budget,
    )
    observed = activity(ctx, started, metrics={RESET_METRIC, LINK_METRIC})
    evidence = reset_evidence(observed, a, anchor)
    delivery = host_delivery_after(observed, anchor)
    oracle.read()
    summary = oracle_summary(oracle.records, a, b, recreate_epoch)
    ctx.evidence.write(
        "reset-evidence",
        {
            "reset_at": reset_at.isoformat(),
            "anchor": anchor.isoformat(),
            "recreated_at": recreated.get("recreated_at"),
            "gap_ms": recreated.get("gap_ms"),
            "host_delivery_after": delivery,
            "oracle": summary,
            "burst_positive": positives[:4],
            "burst_zero": zeros[:4],
            "rearmed": rearmed[:4],
            "rearmed_zero": zeros_after[:4],
            **evidence,
        },
    )
    straddled = link_down_incidents(observed, a, b)
    if straddled:
        raise _straddled(
            ctx, straddled, a=a, b=b, since=reset_at, gap_ms=recreated.get("gap_ms")
        )
    close_window(ctx)
    ctx.cleanup.discard("collector-window")
    if not evidence["errors"] and not evidence["readings"] and delivery is None:
        before = host_delivery_after(observed, started) or {}
        raise VariantInconclusive(
            "the host collector delivered nothing after the recreate within the "
            f"waits (last success before it: {before.get('last_success_at') or 'none'}); "
            f"a silent channel cannot show how the recreated {a} was read"
        )
    stages = {
        "reset_unknown": _reset_unknown(a, evidence, delivery),
        "no_reading_in_reset_tick": [
            f"{RESET_METRIC}={item.get('value')} shipped for {a} in the reset tick "
            f"(observed {item.get('observed_at')})"
            for item in evidence["reset_tick"]
        ],
        "rearmed": rearm_errors(
            a,
            b,
            rearmed=rearmed,
            zeros_after=zeros_after,
            readings=evidence["readings"],
            post_reset_drops=summary["post_reset_drops"],
            budget=budget,
        ),
        "no_finding": [
            error
            for device in (a, b)
            for error in chain_errors(
                observed,
                ChainExpectation(metric=RESET_METRIC, device=device, present=False),
            )
        ],
        "hardware": hardware_errors(observed),
        "no_isolation": [
            f"isolation planned: {item}" for item in isolation_operations(observed)
        ]
        + (
            [f"{ctx.settings.node} is isolated after a counter reset"]
            if node_isolated(ctx.regional.node_snapshot(ctx.settings.node))
            else []
        ),
    }
    return net._result(
        stages,
        notes=[
            "a counter below its baseline is unknown for that tick (collection error, "
            "no delta), not a decrement, not a recovery and not a new fault",
            "the reset tick and the re-armed delta are edges the collector delivers "
            "at once; the recreate is one probe process so no tick sees the pair missing",
        ],
        extra={
            "recreated": recreated,
            "reset_at": reset_at.isoformat(),
            "anchor": anchor.isoformat(),
            "oracle": summary,
            "host_delivery_after": delivery,
        },
    )


__all__ = [
    "BURST_SECONDS",
    "CLOCK_TOLERANCE_SECONDS",
    "RAMP_SECONDS",
    "counter_reset",
    "dropped_frames",
    "drops_samples",
    "edge_budget",
    "frames_since",
    "host_delivery_after",
    "link_down_incidents",
    "oracle_seconds",
    "oracle_summary",
    "rearm_errors",
    "reset_evidence",
]
