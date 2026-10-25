"""GF-GENERAL-LIVE-009 ``tcp_only``: TCP retransmissions are not NIC errors.

The native negative of 009: a real TCP flow loses 5 % of its segments to
``tc netem loss`` and retransmits them, and the product -- which reads the host
network namespace's ``/proc/net/snmp`` (``tcp_retransmits_delta``, collected
but deliberately without a rule) and the NIC ``statistics`` counters
(``network_errors_delta``/``network_drops_delta``) -- must not take that for a
NIC fault: no finding, no fabric hardware path, no isolation.

The fixture (``general_live_net_probe.py`` ``tcp-*``): a ``gfrs<digest>ns``
test netns with a gfrs veth pair to the host netns, built behind the probe's
real-interface guards; netem on the *host-side* veth's egress only; a bounded
sender in the host netns and a receiver in the test netns (transient units with
``RuntimeMaxSec``), so the retransmissions land in the host counter the
collector reads; a deadman that deletes the qdisc, netns and veth. Only the
host-side end is declared required: the peer lives in the test netns, where
the host collector cannot see it, and declaring it would read as a missing
required interface (``network_link_down`` -> QUARANTINE).

``tc`` and ``sch_netem``: used as the node has them. When ``tc`` is missing the
variant stays NOT_RUN(CAPABILITY_ABSENT) unless ``TEMP_TEST_PACKAGE_INSTALL``
is given; then ``general_live_deps_probe.py`` installs exactly ``iproute-tc``
(or the running kernel's ``kernel-modules-extra`` for a module the kernel
cannot load) for this run, refuses -- NOT_RUN(FIXTURE_NOT_READY) -- any
transaction that would pull anything else, loads ``sch_netem`` only when it is
not loaded (``KERNEL_MODULE_LOAD``), and its cleanup and deadman undo exactly
what this run did.

The effectiveness oracle (``tcp_oracle``): over the flow window the host
``RetransSegs`` rose by at least ``MIN_RETRANSMITS``, netem dropped packets
and the receiver got data, while the test veth's error/drop counters and every
real interface's error counters stayed flat; otherwise INCONCLUSIVE.

The negative is decided on records the product produces for a *quiet* window.
``tcp_retransmits_delta`` has no rule, so a batch carrying it is never an edge
(``collectors/host/collector.py`` ``_edge_reasons``): inside the flow the host
collector delivers only its health summaries, and ``app/ingest/telemetry.py``
keeps a summary with no finding and no collection error as no raw evidence.
What every accepted batch does leave behind is the ``HOST_TELEMETRY`` collector
status row (``_record_collector_status``: ``host-<node>-<observed_at us>``,
upserted per accepted batch) and the latest-wins records of the required test
veth's ``network_errors_delta``/``network_drops_delta``
(``host_health.evaluate_metrics`` over ``LATEST_METRICS_REQUIRED``; a zero
ships for them). A *witness* is such a record observed inside the flow window
and confirmed against the runner's own RetransSegs samples: the collector
interval that ended at its ``observed_at`` counted retransmissions, so the
batch it stands for carried a positive ``tcp_retransmits_delta`` (``_tcp``
emits it unconditionally once it has a baseline) and the control plane graded
it. PASS needs a witness and none of: a NIC/TCP/RDMA finding, a stored batch
delivered on a NIC or fabric ``threshold:`` edge, a non-zero veth counter in
the product's own readings, a hardware step, an isolation. Any of those is
FAIL. INCONCLUSIVE only when the stimulus failed or the product accepted no
batch inside the window (a silent channel). Live 2026-10-25, both lanes
(releases 13daa08b2227 and d5d7610460e0): 22952 and 23608 retransmissions,
16 polls, not one stored batch -- the previous oracle waited for raw evidence
the product never writes for this metric.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from scripts.e2e.regional.general_live_common import (
    Oracle,
    activity,
    close_window,
    open_window,
    run_survey,
)
from scripts.e2e.regional.general_live_deps import (
    DEPS_TIMEOUT,
    PACKAGE_APPROVAL,
    checked_cleanup,
    dependency_summary,
    deps_deadman_seconds,
    ensure_dependencies as ensure_test_dependencies,
    not_run,
)
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    require_module,
    require_tools,
    utc_now,
)
from scripts.e2e.regional.general_live_verdicts import (
    ChainExpectation,
    chain_errors,
    finding_incidents,
    hardware_errors,
    isolation_operations,
    node_isolated,
    parse_time,
    samples,
    verdict,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError

LOSS_PERCENT = 5
RATE_KBPS = 8000
MIN_RETRANSMITS = 20
MIN_FLOW_SECONDS = 60
TAIL_SECONDS = 120
NIC_METRICS = ("network_errors_delta", "network_drops_delta", "network_link_down")
TCP_METRIC = "tcp_retransmits_delta"
#: The fabric error-counter rules (``host_health.py`` METRIC_RULES, the
#: VALIDATE_FABRIC family) a TCP-only loss must not mint either.
RDMA_METRICS = (
    "rdma_errors_delta",
    "rdma_link_down",
    "efa_rnr_errors_delta",
    "efa_retry_errors_delta",
    "efa_cq_errors_delta",
)
#: The test veth's latest-wins records: ``LATEST_METRICS_REQUIRED`` counters a
#: zero ships for, upserted from every accepted batch, summary or not.
VETH_LATEST_METRICS = ("network_errors_delta", "network_drops_delta")
#: Edge reasons that would mean the collector took the loss for a NIC or
#: fabric fault; a stored batch delivered on one fails the variant before any
#: finding (the drops rule has a 60-s sustain, so the edge comes first).
FORBIDDEN_EDGES = frozenset(
    f"threshold:{metric}" for metric in (*NIC_METRICS, *RDMA_METRICS)
)
HOST_COLLECTOR = "HOST_TELEMETRY"
#: The node's host collector posts ``host-`` batches; the control plane's
#: allocatable reader posts ``k8s-efa-`` batches on the same channel.
HOST_BATCH_PREFIX = "host-"
#: Slack around a witness batch's ``observed_at`` when reading the oracle's
#: 5-s RetransSegs samples over the collector interval that ended there.
ORACLE_SLACK_SECONDS = 5
#: Counters that must not move: every counter of the test veth (netem drops in
#: the qdisc, never in the NIC), the error counters of every real interface.
TEST_COUNTERS = ("rx_errors", "tx_errors", "rx_dropped", "tx_dropped")
REAL_COUNTERS = ("rx_errors", "tx_errors")
#: The host-side end the probe creates (``gfrs`` + run digest + slot ``a``).
TEST_INTERFACE = re.compile(r"^gfrs[0-9a-f]{6}a$")


def tcp_timing(ctx: VariantContext) -> dict[str, int]:
    """The flow, window and deadman budgets from the collector's own cadence.

    The product accepts a host batch at least once per health summary (+-5 %
    jitter), so the flow runs long enough for one accepted batch -- the
    collector status row and the veth's latest records move with it -- to be
    observed inside the flow. The window's restore deadman fires before the
    fixture's deadman deletes the required test interface. The package
    deadman is armed before the install (at most ``DEPS_TIMEOUT``) and the
    fixture's after it, so its offset covers that too: it always fires last,
    once nothing uses the module any more.
    """

    interval, summary = ctx.host_interval, ctx.host_summary
    flow_wait = min(900, int(summary * 1.5) + interval * 4)
    flow_seconds = MIN_FLOW_SECONDS + flow_wait + 120
    restore = min(1800, interval * 3 + flow_seconds + TAIL_SECONDS + 600)
    fixture_deadman = min(5400, max(ctx.spec.max_minutes * 60 + 600, restore + 600))
    return {
        "flow_wait": flow_wait,
        "flow_seconds": flow_seconds,
        "oracle_seconds": flow_seconds + 60,
        "restore_seconds": restore,
        "fixture_deadman": fixture_deadman,
        "deps_deadman": deps_deadman_seconds(fixture_deadman),
    }


def ensure_dependencies(
    ctx: VariantContext, survey: dict[str, Any], timing: dict[str, int]
) -> dict[str, Any]:
    """``tc`` and a loaded ``sch_netem``: as found, or for this run only.

    The shared temporary-package step (``general_live_deps``): without
    ``TEMP_TEST_PACKAGE_INSTALL`` a missing ``tc`` stays
    NOT_RUN(CAPABILITY_ABSENT), a module that is not loaded needs
    ``KERNEL_MODULE_LOAD``, and the probe's cleanup is on the LIFO stack before
    the install so the package goes after the fixture.
    """

    return ensure_test_dependencies(
        ctx,
        survey,
        tools=("tc",),
        modules=("sch_netem",),
        deadman_seconds=timing["deps_deadman"],
        evidence="tcp-dependencies",
    )


def create_fixture(ctx: VariantContext, timing: dict[str, int]) -> dict[str, Any]:
    ctx.cleanup.push("net-tcp", lambda: checked_cleanup(ctx, "net", "--kind", "tcp"))
    created = ctx.net(
        "tcp-setup",
        "--deadman-seconds",
        str(timing["fixture_deadman"]),
        "--loss-percent",
        str(LOSS_PERCENT),
    )
    ctx.evidence.write("tcp-fixture", created)
    if created.get("refused"):
        raise not_run(created["refused"], "the TCP fixture refused")
    if TEST_INTERFACE.fullmatch(str(created.get("interface"))) is None:
        raise RegionalFixtureError(f"not a gfrs test interface: {created}")
    qdisc = created.get("qdisc") or {}
    if qdisc.get("kind") != "netem" or not qdisc.get("root"):
        raise RegionalFixtureError(f"no root netem qdisc on the test veth: {qdisc}")
    return created


def _delta(first: Any, last: Any) -> float | None:
    if not isinstance(first, (int, float)) or not isinstance(last, (int, float)):
        return None
    return float(last) - float(first)


def _moment(value: Any) -> datetime | None:
    try:
        moment = parse_time(value)
    except ValueError:
        return None
    if moment is not None and moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def _inside(value: Any, since: datetime, until: datetime | None) -> datetime | None:
    """``value`` as a moment when it lies in ``[since, until]`` (``until`` open)."""

    moment = _moment(value)
    if moment is None or moment < since or (until is not None and moment > until):
        return None
    return moment


def tcp_oracle(
    records: Sequence[dict[str, Any]],
    *,
    interface: str,
    real_interfaces: Sequence[str],
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    """The runner's own reading of the flow window; ``errors`` -> INCONCLUSIVE.

    ``records`` are the node oracle's samples (host ``RetransSegs`` and the
    sysfs counters of the test veth and of every real interface); the window
    is the node-clock span between the ``before``/``after`` point readings,
    which also carry the netem qdisc drops and the receiver's byte count.
    """

    start, end = before.get("t"), after.get("t")
    window = [
        record
        for record in records
        if isinstance(record.get("t"), (int, float))
        and isinstance(start, (int, float))
        and isinstance(end, (int, float))
        and start <= record["t"] <= end
    ]
    errors: list[str] = []
    if len(window) < 2:
        errors.append(f"the oracle took {len(window)} sample(s) in the flow window")
    retrans = [
        record["tcp_retransmits"]
        for record in window
        if isinstance(record.get("tcp_retransmits"), (int, float))
    ]
    retrans_delta = _delta(retrans[0], retrans[-1]) if len(retrans) >= 2 else None
    if any(later < earlier for earlier, later in zip(retrans, retrans[1:])):
        errors.append("host RetransSegs went backwards inside the window")
    elif retrans_delta is None or retrans_delta < MIN_RETRANSMITS:
        errors.append(
            f"host RetransSegs rose by {retrans_delta} (< {MIN_RETRANSMITS}); "
            "the netem loss was ineffective"
        )

    def counter_deltas(name: str, counters: Sequence[str]) -> dict[str, Any]:
        readings = [(record.get("netdev") or {}).get(name) or {} for record in window]
        readable = [item for item in readings if item.get("present") is not False]
        if not readings or len(readable) != len(readings):
            return {"unread": True}
        return {
            counter: _delta(readable[0].get(counter), readable[-1].get(counter))
            for counter in counters
        }

    test = counter_deltas(interface, TEST_COUNTERS)
    real = {name: counter_deltas(name, REAL_COUNTERS) for name in real_interfaces}
    if test.get("unread") or any(test.get(c) is None for c in TEST_COUNTERS):
        errors.append(f"the test veth {interface} was not read throughout the window")
    else:
        moved = {c: test[c] for c in TEST_COUNTERS if test[c] != 0}
        if moved:
            errors.append(
                f"the test veth {interface} counted {moved}; netem must not move "
                "NIC counters"
            )
    for name, deltas in sorted(real.items()):
        if deltas.get("unread") or any(deltas.get(c) is None for c in REAL_COUNTERS):
            errors.append(f"real interface {name} was not read throughout the window")
        elif any(deltas[c] != 0 for c in REAL_COUNTERS):
            errors.append(
                f"real interface {name} error counters moved {deltas}; the window "
                "is not TCP-only"
            )
    netem_drops = _delta(
        (before.get("qdisc") or {}).get("drops"),
        (after.get("qdisc") or {}).get("drops"),
    )
    if not netem_drops or netem_drops <= 0:
        errors.append(f"netem dropped {netem_drops} packets on {interface}")
    received = _delta(
        (before.get("receiver") or {}).get("received_bytes", 0),
        (after.get("receiver") or {}).get("received_bytes"),
    )
    if not received or received <= 0:
        errors.append("no TCP data reached the receiver in the test netns")
    return {
        "samples": len(window),
        "retrans_delta": retrans_delta,
        "netem_drops": netem_drops,
        "received_bytes": received,
        "test_interface": {interface: test},
        "real_interfaces": real,
        "errors": errors,
    }


def oracle_retransmits(
    records: Sequence[dict[str, Any]], moment: datetime, interval: int
) -> float | None:
    """How far host RetransSegs rose over the collector interval ending at ``moment``.

    ``moment`` is a batch's ``observed_at`` (the node's clock, as the oracle's
    ``t``); the samples taken from ``interval`` plus the slack before it to the
    slack after it bracket the delta interval of the sample that batch carried.
    ``None`` when the oracle has fewer than two samples there.
    """

    end = moment.timestamp() + ORACLE_SLACK_SECONDS
    start = moment.timestamp() - interval - ORACLE_SLACK_SECONDS
    values = [
        float(record["tcp_retransmits"])
        for record in records
        if isinstance(record.get("t"), (int, float))
        and start <= record["t"] <= end
        and isinstance(record.get("tcp_retransmits"), (int, float))
    ]
    if len(values) < 2:
        return None
    return values[-1] - values[0]


def tcp_findings(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Every NIC, TCP or fabric finding the product minted in ``document``."""

    return [
        item
        for metric in (*NIC_METRICS, TCP_METRIC, *RDMA_METRICS)
        for item in finding_incidents(document, metric)
    ]


def product_retransmits(
    document: dict[str, Any], since: datetime, until: datetime | None = None
) -> list[Any]:
    """Accepted ``tcp_retransmits_delta`` > 0 samples from inside the flow.

    Raw evidence only exists for a batch delivered on some edge or with a
    collection error; when one happens to fall inside the flow it is the
    strongest witness there is. ``since`` is one collector interval after the
    flow started, so the sample's whole delta interval lies inside the flow;
    ``until`` is when it was stopped, so a later batch carrying only
    background retransmissions never counts.
    """

    result = []
    for sample in samples(document, TCP_METRIC):
        value = sample.get("value")
        if _inside(sample.get("observed_at"), since, until) is None:
            continue
        if isinstance(value, (int, float)) and value > 0:
            result.append(sample)
    return result


def host_delivery_inside(
    document: dict[str, Any], since: datetime, until: datetime | None
) -> dict[str, Any] | None:
    """The node's host collector status when its last accepted batch was
    observed inside ``[since, until]``; the row is latest-wins, so it is read
    while the flow runs and kept from there."""

    for item in document.get("collector_statuses") or []:
        if item.get("collector") != HOST_COLLECTOR:
            continue
        if not str(item.get("batch_id") or "").startswith(HOST_BATCH_PREFIX):
            continue
        moment = item.get("last_success_at") or item.get("observed_at")
        if _inside(moment, since, until) is None:
            continue
        return dict(item)
    return None


def veth_latest_inside(
    document: dict[str, Any], interface: str, since: datetime, until: datetime | None
) -> list[dict[str, Any]]:
    """The test veth's latest counter records observed inside ``[since, until]``."""

    rows = []
    for item in document.get("metric_latest") or []:
        if item.get("name") not in VETH_LATEST_METRICS:
            continue
        if item.get("device") != interface:
            continue
        if _inside(item.get("observed_at"), since, until) is None:
            continue
        rows.append({**item, "source": "metric_latest"})
    return sorted(rows, key=lambda item: (str(item["observed_at"]), str(item["name"])))


def witness_candidates(
    document: dict[str, Any], *, interface: str, since: datetime, until: datetime | None
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """The product's records of a host batch it accepted inside the window.

    Keyed so the polls of a flow accumulate every distinct record: the status
    row and the latest records are overwritten by the next accepted batch, a
    raw-evidence sample stays.
    """

    found: dict[tuple[str, str, str], dict[str, Any]] = {}
    delivery = host_delivery_inside(document, since, until)
    if delivery is not None:
        moment = str(delivery.get("last_success_at") or delivery.get("observed_at"))
        found["collector_status", moment, ""] = {
            "source": "collector_status",
            "observed_at": moment,
            "batch_id": delivery.get("batch_id"),
            "sample_count": delivery.get("sample_count"),
        }
    for row in veth_latest_inside(document, interface, since, until):
        key = ("metric_latest", str(row["observed_at"]), str(row["name"]))
        found[key] = {
            "source": "metric_latest",
            "observed_at": row["observed_at"],
            "name": row["name"],
            "device": interface,
            "value": row.get("value"),
        }
    for sample in product_retransmits(document, since, until):
        key = ("evidence", str(sample["observed_at"]), str(sample.get("batch_id")))
        found[key] = {
            "source": "evidence",
            "observed_at": sample["observed_at"],
            "batch_id": sample.get("batch_id"),
            "value": sample.get("value"),
        }
    return found


def confirm_witnesses(
    candidates: Iterable[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    interval: int,
) -> dict[str, list[dict[str, Any]]]:
    """Split the candidates by whether the oracle counted retransmissions in
    the collector interval ending at each one (a raw-evidence sample is its
    own confirmation: it carries the positive delta)."""

    confirmed: list[dict[str, Any]] = []
    unconfirmed: list[dict[str, Any]] = []
    for candidate in candidates:
        moment = _moment(candidate.get("observed_at"))
        rise = None if moment is None else oracle_retransmits(records, moment, interval)
        witness = {**candidate, "oracle_retransmits": rise}
        if candidate.get("source") == "evidence" or (rise is not None and rise > 0):
            confirmed.append(witness)
        else:
            unconfirmed.append(witness)

    def order(item: dict[str, Any]) -> tuple[str, str]:
        return str(item["observed_at"]), str(item["source"])

    return {
        "confirmed": sorted(confirmed, key=order),
        "unconfirmed": sorted(unconfirmed, key=order),
    }


def forbidden_edges(document: dict[str, Any], since: datetime) -> list[str]:
    """Stored batches since ``since`` that the collector delivered on a NIC or
    fabric threshold edge (or on anything naming the TCP metric)."""

    errors = []
    for batch in document.get("batches") or []:
        if _inside(batch.get("observed_at"), since, None) is None:
            continue
        reasons = [str(item) for item in batch.get("edge_filter_reasons") or []]
        found = [
            item for item in reasons if item in FORBIDDEN_EDGES or TCP_METRIC in item
        ]
        if found:
            errors.append(
                f"batch {batch.get('batch_id')} ({batch.get('observed_at')}) was "
                f"delivered on {found}: the collector took the TCP loss for a "
                "NIC/fabric fault"
            )
    return errors


def watch(
    ctx: VariantContext,
    started: datetime,
    *,
    seconds: float,
    name: str,
    interface: str,
    since: datetime,
    until_witness: bool,
) -> tuple[dict[str, Any], dict[tuple[str, str, str], dict[str, Any]]]:
    """Poll the product's records; stop early on a finding (or the first witness).

    Returns the last document and every witness candidate seen across the
    polls: the status row and the latest records are latest-wins, so a
    summary accepted inside the flow is only on record while it is the newest.
    """

    metrics = {*NIC_METRICS, TCP_METRIC, *RDMA_METRICS}
    last: dict[str, Any] = {}
    seen: dict[tuple[str, str, str], dict[str, Any]] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = activity(ctx, started, metrics=metrics)
        if tcp_findings(last):
            return last
        seen.update(
            witness_candidates(last, interface=interface, since=since, until=None)
        )
        if until_witness and seen:
            return last
        return None

    value = ctx.wait_until(
        accept,
        timeout_seconds=seconds,
        poll_seconds=ctx.host_interval * 2,
        name=name,
    )
    return (value if value is not None else last), seen


def product_stages(
    ctx: VariantContext,
    observed: dict[str, Any],
    interface: str,
    *,
    since: datetime,
    veth_latest: Sequence[dict[str, Any]] = (),
) -> dict[str, list[str]]:
    nic = [
        error
        for metric in NIC_METRICS
        for error in chain_errors(
            observed, ChainExpectation(metric=metric, present=False)
        )
    ]
    rdma = [
        error
        for metric in RDMA_METRICS
        for error in chain_errors(
            observed, ChainExpectation(metric=metric, present=False)
        )
    ]
    readings = [
        f"the product read {metric}={sample['value']} on {interface}; the oracle read 0"
        for metric in VETH_LATEST_METRICS
        for sample in samples(observed, metric, device=interface)
        if isinstance(sample.get("value"), (int, float)) and sample["value"] != 0
    ]
    readings += [
        f"the product's latest {row.get('name')} on {interface} read {row['value']} "
        f"at {row.get('observed_at')}; the oracle read 0"
        for row in veth_latest
        if isinstance(row.get("value"), (int, float)) and row["value"] != 0
    ]
    isolation = [
        f"isolation planned: {item}" for item in isolation_operations(observed)
    ]
    if node_isolated(ctx.regional.node_snapshot(ctx.settings.node)):
        isolation.append(f"{ctx.settings.node} is isolated after a TCP-only window")
    return {
        "no_nic_finding": nic,
        "no_tcp_finding": chain_errors(
            observed, ChainExpectation(metric=TCP_METRIC, present=False)
        ),
        "no_rdma_finding": rdma,
        "no_fabric_edge": forbidden_edges(observed, since),
        "test_interface_zero": readings,
        "hardware": hardware_errors(observed),
        "no_isolation": isolation,
    }


def window_view(
    observed: dict[str, Any],
    *,
    interface: str,
    since: datetime,
    stopped: datetime,
    witnesses: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """What the product recorded for the window, kept for reading the verdict later."""

    return {
        "since": since.isoformat(),
        "stopped": stopped.isoformat(),
        "batches": [
            {
                key: batch.get(key)
                for key in (
                    "observed_at",
                    "batch_id",
                    "edge_filter_reasons",
                    "collection_errors",
                    "sample_count",
                    "samples",
                )
            }
            for batch in observed.get("batches") or []
        ],
        "metric_latest": [
            item
            for item in observed.get("metric_latest") or []
            if item.get("device") == interface or item.get("name") == TCP_METRIC
        ],
        "host_collector": [
            item
            for item in observed.get("collector_statuses") or []
            if item.get("collector") == HOST_COLLECTOR
        ],
        "witnesses": witnesses,
        "incidents": [
            {
                key: item.get(key)
                for key in (
                    "incident_id",
                    "event_id",
                    "event_type",
                    "state",
                    "effective_action",
                    "created_at",
                )
            }
            for item in observed.get("incidents") or []
        ],
    }


def _silent_channel(
    observed: dict[str, Any],
    *,
    interface: str,
    since: datetime,
    stopped: datetime,
    unconfirmed: Sequence[dict[str, Any]],
) -> str:
    delivery = [
        item.get("last_success_at")
        for item in observed.get("collector_statuses") or []
        if item.get("collector") == HOST_COLLECTOR
        and str(item.get("batch_id") or "").startswith(HOST_BATCH_PREFIX)
    ]
    latest = [
        item.get("observed_at")
        for item in observed.get("metric_latest") or []
        if item.get("device") == interface and item.get("name") in VETH_LATEST_METRICS
    ]
    return (
        f"the product accepted no host batch observed inside the flow window "
        f"({since.isoformat()} .. {stopped.isoformat()}): host collector last "
        f"success {delivery[0] if delivery else 'none'}, latest {interface} counter "
        f"record observed {max(latest) if latest else 'never'}, "
        f"{len(unconfirmed)} candidate(s) the oracle could not confirm; a silent "
        f"channel cannot show how {TCP_METRIC} was graded, so 'no finding' "
        "proves nothing"
    )


def tcp_only(ctx: VariantContext) -> dict[str, Any]:
    survey = run_survey(ctx)
    require_tools(survey, "ip", "systemd-run", "nsenter")
    require_module(survey, "veth", ctx.settings.approvals)
    timing = tcp_timing(ctx)
    dependencies = ensure_dependencies(ctx, survey, timing)
    fixture = create_fixture(ctx, timing)
    interface = str(fixture["interface"])
    real = [str(name) for name in fixture.get("real_interfaces") or []]
    open_window(
        ctx,
        f"GPU_FAULT_REQUIRED_INTERFACES={interface}",
        restore_seconds=timing["restore_seconds"],
    )
    ctx.hold(ctx.host_interval * 3, name="healthy-samples")
    oracle = Oracle(ctx, "tcp")
    oracle.start(
        timing["oracle_seconds"], interval=5, netdev=[interface, *real], snmp=True
    )
    before = ctx.net("tcp-reading")
    started = utc_now()
    # One collector interval in: the first batch whose whole delta interval
    # lies inside the flow.
    since = started + timedelta(seconds=ctx.host_interval)
    ctx.cleanup.push("tcp-flow", lambda: ctx.net("tcp-stop"))
    flow = ctx.net(
        "tcp-start",
        "--seconds",
        str(timing["flow_seconds"]),
        "--rate-kbps",
        str(RATE_KBPS),
    )
    ctx.evidence.write("tcp-flow", flow)
    ctx.hold(MIN_FLOW_SECONDS, name="tcp-flow")
    _, candidates = watch(
        ctx,
        started,
        seconds=timing["flow_wait"],
        name="tcp-product",
        interface=interface,
        since=since,
        until_witness=True,
    )
    after = ctx.net("tcp-reading")
    stopped = utc_now()
    ctx.evidence.write("tcp-flow-stop", ctx.net("tcp-stop"))
    ctx.cleanup.discard("tcp-flow")
    oracle.read()
    reading = tcp_oracle(
        oracle.records,
        interface=interface,
        real_interfaces=real,
        before=before,
        after=after,
    )
    ctx.evidence.write("tcp-oracle", {**reading, "before": before, "after": after})
    if reading["errors"]:
        raise VariantInconclusive("; ".join(reading["errors"]))
    observed, _ = watch(
        ctx,
        started,
        seconds=TAIL_SECONDS,
        name="tcp-negative",
        interface=interface,
        since=since,
        until_witness=False,
    )
    candidates.update(
        witness_candidates(observed, interface=interface, since=since, until=stopped)
    )
    witnesses = confirm_witnesses(
        candidates.values(), oracle.records, ctx.host_interval
    )
    veth_latest = [
        item for item in candidates.values() if item["source"] == "metric_latest"
    ]
    stages = product_stages(
        ctx, observed, interface, since=since, veth_latest=veth_latest
    )
    ctx.evidence.write(
        "tcp-window",
        window_view(
            observed,
            interface=interface,
            since=since,
            stopped=stopped,
            witnesses=witnesses,
        ),
    )
    close_window(ctx)
    ctx.cleanup.discard("collector-window")
    if not any(stages.values()) and not witnesses["confirmed"]:
        raise VariantInconclusive(
            _silent_channel(
                observed,
                interface=interface,
                since=since,
                stopped=stopped,
                unconfirmed=witnesses["unconfirmed"],
            )
        )
    errors = [f"{stage}: {error}" for stage, items in stages.items() for error in items]
    return {
        "verdict": verdict(stages),
        "stages": stages,
        "errors": errors,
        "notes": [
            "tcp_retransmits_delta is collected without a rule: retransmissions "
            "are congestion control, not a NIC fault",
            "a summary-only batch leaves no raw evidence; the witness is the "
            "collector status row / the veth's latest records moving inside the "
            "flow, confirmed by the oracle's RetransSegs rise in that interval",
        ],
        "device": interface,
        "oracle": {key: value for key, value in reading.items() if key != "errors"},
        "witnesses": witnesses,
        "product_retransmits": [
            item for item in witnesses["confirmed"] if item["source"] == "evidence"
        ][:5],
        "dependencies": dependency_summary(dependencies),
        "proof_scope": "native",
    }


__all__ = [
    "DEPS_TIMEOUT",
    "FORBIDDEN_EDGES",
    "PACKAGE_APPROVAL",
    "RDMA_METRICS",
    "VETH_LATEST_METRICS",
    "checked_cleanup",
    "confirm_witnesses",
    "create_fixture",
    "ensure_dependencies",
    "forbidden_edges",
    "host_delivery_inside",
    "not_run",
    "oracle_retransmits",
    "product_retransmits",
    "product_stages",
    "tcp_findings",
    "tcp_only",
    "tcp_oracle",
    "tcp_timing",
    "veth_latest_inside",
    "witness_candidates",
]
