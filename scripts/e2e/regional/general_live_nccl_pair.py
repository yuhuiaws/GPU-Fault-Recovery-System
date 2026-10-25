"""GF-GENERAL-LIVE-011 ``broken_pair``: the EFA traffic pair broken inside the host collector.

Builds on ``general_live_nccl`` (the managed two-node job, the probe's pair
oracle, the SITE_EFA_TRAFFIC records) and on the shared EFA source fixture
(``general_live_efa_fixture``): one port's ``hw_counters/tx_bytes`` is bound to
text the product's ``read_counter`` rejects, so ``rdma_traffic.port_traffic``
reports an incomplete pair, ``_rdma`` ships ``efa_traffic_observation_valid=0``
and no rate, and ``efa_traffic_state`` resets the zero streak instead of
counting the stalled job's real zero. A category module of its own (listed in
``CATEGORY_MODULES`` after ``general_live_nccl``, which it imports) so the 011
module stays under the size limit without an import cycle.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from scripts.e2e.regional.general_live_common import activity, run_survey
from scripts.e2e.regional.general_live_efa_fixture import (
    EfaSourceFixture,
    collection_errors,
    pair_target,
)
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
)
from scripts.e2e.regional.general_live_registry import LIVE_REPLAY
from scripts.e2e.regional.general_live_source_replay import HOST_SOURCE_REPLAY
from scripts.e2e.regional.general_live_nccl import (
    CLOCK_TOLERANCE_SECONDS,
    CREATOR,
    HUNG_SUFFIX,
    SETTLED_STATES,
    STALL_SETTLE_SECONDS,
    WARNING_SUFFIX,
    ZERO_WARNING_SECONDS,
    ManagedNcclJob,
    _efa_window,
    _job,
    _prove_baseline,
    _records,
    _release_and_resume,
    _signal_incidents,
    _thresholds,
    drop_errors,
    stall_event_errors,
    variant_result,
    variant_spec,
)
from scripts.e2e.regional.general_live_verdicts import parse_time, samples


def _node_signals(
    records: dict[str, Any], node: str, suffix: str
) -> list[dict[str, Any]]:
    """The SITE_EFA_TRAFFIC incidents ``node``'s own batches raised (``suffix``).

    The event id names the reporting node (``host-<node>-<us>-efa-traffic-...``);
    a single-node incident naming only ``node`` counts too. The peer's own
    ZERO_WARNING on its real zero is not this node's reading.
    """

    found = []
    for item in _signal_incidents(records, suffix):
        match = CREATOR.match(str(item.get("event_id", "")))
        creator = match.group("node") if match else None
        if creator == node or (creator is None and item.get("node_ids") == [node]):
            found.append(item)
    return found


def _node_warnings(records: dict[str, Any], node: str) -> list[dict[str, Any]]:
    return _node_signals(records, node, WARNING_SUFFIX)


def _await_records(
    ctx: VariantContext,
    job: ManagedNcclJob,
    since: datetime,
    accept: Any,
    *,
    timeout_seconds: float,
    name: str,
) -> dict[str, Any]:
    last: dict[str, Any] = {}

    def poll() -> dict[str, Any] | None:
        nonlocal last
        last = _records(ctx, job, since)
        return last if accept(last) else None

    value = ctx.wait_until(
        poll, timeout_seconds=timeout_seconds, poll_seconds=20, name=name
    )
    records = value if value is not None else last
    ctx.evidence.write(f"{name}-records", records)
    return records


def broken_pair(ctx: VariantContext) -> dict[str, Any]:
    """An unparsable TX half of the EFA pair inside the host collector, around a real stall.

    The pair is broken (one port's ``hw_counters/tx_bytes`` bound to text the
    product's ``read_counter`` rejects) before the job is stalled, and held past
    the warning threshold plus a health summary: the product must read every
    sample as ``efa_traffic_observation_valid=0`` with no rate and start no zero
    streak (no ZERO_WARNING, no HUNG_SUSPECTED), although the job's traffic really
    is zero. Once the window closes and the pair reads again, the still-stalled
    job's zero must be graded: a ZERO_WARNING no earlier than the warning
    threshold after the restore proves the streak began on real readings only.
    """

    run_survey(ctx)
    job = _job(ctx, "bp")
    _prove_baseline(ctx, name="bp-active")
    fixture = EfaSourceFixture(ctx)
    target = pair_target(fixture.survey())
    ctx.evidence.write("bp-target", target.describe())
    fixture.bind("efa-tx-bytes", target.counter_dest, "broken")
    fixture.open()
    broken_at = ctx.clock()
    stall_at = ctx.clock()
    ctx.cleanup.push("release", job.release)
    ctx.evidence.write("bp-stall", {"at": stall_at.isoformat(), **job.stall()})
    ctx.hold(STALL_SETTLE_SECONDS, name="bp-settle")
    # Past the warning threshold and one health summary (a reader that took the
    # broken pair for a zero would have warned), short of the hung threshold on
    # the peer, whose real zero is graded meanwhile.
    hold = ZERO_WARNING_SECONDS + ctx.host_summary + ctx.host_interval * 2
    _minimum, zero = _thresholds(ctx)
    window = _efa_window(ctx, seconds=hold, name="bp-broken")
    errors = drop_errors(
        window, zero_bps=zero, minimum_seconds=hold - ctx.host_interval * 2
    )
    liveness, events = job.alive(), job.trainer_events()
    ctx.evidence.write("bp-job", {"liveness": liveness, "trainer": events})
    if not liveness.get("alive"):
        errors.append("the job did not stay alive through the stall")
    errors.extend(stall_event_errors(events, job.stall_mode))
    if errors:
        raise VariantInconclusive(
            "the stall did not hold a live job at zero EFA traffic while the pair "
            "was broken: " + "; ".join(errors)
        )
    records = _records(ctx, job, stall_at)
    ctx.evidence.write("bp-broken-records", records)
    observed = activity(
        ctx,
        broken_at,
        metrics={"efa_traffic_observation_valid", "efa_traffic_bytes_per_second"},
    )
    invalid = [
        item
        for item in samples(observed, "efa_traffic_observation_valid")
        if item.get("value") == 0
    ]
    node = ctx.settings.node
    raised = [
        str(item.get("event_id"))
        for suffix in (WARNING_SUFFIX, HUNG_SUFFIX)
        for item in _node_signals(records, node, suffix)
    ]
    peer_warnings = [
        str(item.get("event_id"))
        for item in _signal_incidents(records, WARNING_SUFFIX)
        if item not in _node_warnings(records, node)
    ]
    stages: dict[str, list[str]] = {
        "no_streak_while_broken": [
            f"{node} raised a zero-traffic suspicion on a broken counter pair: {raised}"
        ]
        if raised
        else [],
        "observation_invalid": []
        if invalid and collection_errors(observed, "incomplete traffic counter pair")
        else [
            "the broken pair was not recorded as efa_traffic_observation_valid=0 with "
            "an 'incomplete traffic counter pair' collection error"
        ],
        "no_rate_from_broken_pair": [
            f"efa_traffic_bytes_per_second {item.get('value')} shipped from a broken pair"
            for item in samples(observed, "efa_traffic_bytes_per_second")
        ],
    }
    fixture.close()
    closed_at = fixture.closed_at or ctx.clock()
    stages["pair_restored"] = fixture.restored_errors()
    warned = _await_records(
        ctx,
        job,
        stall_at,
        lambda current: bool(_node_warnings(current, node)),
        timeout_seconds=ZERO_WARNING_SECONDS
        + ctx.host_summary
        + ctx.host_interval * 6
        + 60,
        name="bp-warning",
    )
    warnings = _node_warnings(warned, node)
    earliest = closed_at + timedelta(
        seconds=ZERO_WARNING_SECONDS - CLOCK_TOLERANCE_SECONDS
    )
    if not warnings:
        stages["warning_after_restore"] = [
            "no ZERO_WARNING for the stalled job after the pair was restored: the real "
            "zero was not graded"
        ]
    else:
        created = parse_time(warnings[0].get("created_at"))
        stages["warning_after_restore"] = (
            []
            if created is not None and created >= earliest
            else [
                f"ZERO_WARNING created {created} is under {ZERO_WARNING_SECONDS} s after "
                f"the restore at {closed_at.isoformat()}: the streak began on the broken pair"
            ]
        )
    stages["resumed"] = _release_and_resume(ctx, job, name="bp")
    # This node's warnings, and any hung incident the peer's real zero grew into
    # meanwhile (it names this node too and would block the next variant).
    ids = {str(item.get("incident_id")) for item in warnings} | {
        str(item.get("incident_id"))
        for item in _signal_incidents(warned, HUNG_SUFFIX)
        if node in (item.get("node_ids") or [])
    }
    if ids:
        final = _await_records(
            ctx,
            job,
            stall_at,
            lambda current: all(
                item.get("state") in SETTLED_STATES
                for item in current.get("incidents") or []
                if str(item.get("incident_id")) in ids
            ),
            timeout_seconds=ctx.host_summary + 300,
            name="bp-settled",
        )
        open_states = [
            f"{item.get('incident_id')} is {item.get('state')}"
            for item in final.get("incidents") or []
            if str(item.get("incident_id")) in ids
            and item.get("state") not in SETTLED_STATES
        ]
        stages["settled"] = open_states
    return variant_result(
        stages,
        notes=[
            f"{target.counter_dest} read 'broken' inside the host collector from "
            f"{broken_at.isoformat()} to {closed_at.isoformat()}; the probe's own pair "
            "oracle read the real zero throughout",
            "an invalid observation is never a zero sample: the streak starts on real readings",
            f"peer warnings while broken (its pair is real): {peer_warnings}",
        ],
        extra={
            "attempt": job.identity(),
            "target": target.describe(),
            "proof_scope": "live-replay",
            "broken_at": broken_at.isoformat(),
            "closed_at": closed_at.isoformat(),
        },
    )


VARIANTS = [
    variant_spec(
        "broken_pair",
        "two-node job proven active; one EFA port's hw_counters/tx_bytes bound to an"
        " unparsable value inside the host collector (HOST_SOURCE_REPLAY), the job"
        " stalled for the warning threshold plus a health summary, then the pair restored",
        "while broken: efa_traffic_observation_valid=0, no rate sample, no ZERO_WARNING or"
        " HUNG_SUSPECTED from this node (an invalid observation starts no zero streak);"
        " once restored, the still-stalled job's ZERO_WARNING comes no earlier than 60 s"
        " after the restore; traffic resumes on release",
        broken_pair,
        proof=LIVE_REPLAY,
        approvals=(HOST_SOURCE_REPLAY,),
        max_minutes=50,
    ),
]

__all__ = ["VARIANTS", "broken_pair"]
