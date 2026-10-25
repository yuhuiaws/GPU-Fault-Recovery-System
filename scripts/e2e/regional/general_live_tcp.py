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
real interface's error counters stayed flat; otherwise INCONCLUSIVE. A clean
product record is only a proof once the product accepted a batch carrying a
positive ``tcp_retransmits_delta`` from inside the flow; otherwise the
negative would be vacuous and the verdict is INCONCLUSIVE too.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

from scripts.e2e.regional.collector_action_guard import require_action_time
from scripts.e2e.regional.general_live_common import (
    Oracle,
    activity,
    close_window,
    open_window,
    run_survey,
)
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    VariantNotRun,
    require_module,
    require_tools,
    utc_now,
)
from scripts.e2e.regional.general_live_registry import NOT_RUN_REASONS
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

PACKAGE_APPROVAL = "TEMP_TEST_PACKAGE_INSTALL"
MODULE_APPROVAL = "KERNEL_MODULE_LOAD"
LOSS_PERCENT = 5
RATE_KBPS = 8000
MIN_RETRANSMITS = 20
MIN_FLOW_SECONDS = 60
TAIL_SECONDS = 120
DEPS_TIMEOUT = 900
DEPS_CLEANUP_TIMEOUT = 600
NIC_METRICS = ("network_errors_delta", "network_drops_delta", "network_link_down")
TCP_METRIC = "tcp_retransmits_delta"
#: Counters that must not move: every counter of the test veth (netem drops in
#: the qdisc, never in the NIC), the error counters of every real interface.
TEST_COUNTERS = ("rx_errors", "tx_errors", "rx_dropped", "tx_dropped")
REAL_COUNTERS = ("rx_errors", "tx_errors")
#: The host-side end the probe creates (``gfrs`` + run digest + slot ``a``).
TEST_INTERFACE = re.compile(r"^gfrs[0-9a-f]{6}a$")


def tcp_timing(ctx: VariantContext) -> dict[str, int]:
    """The flow, window and deadman budgets from the collector's own cadence.

    The product ships a host batch at least once per health summary (+-10 %
    jitter), so the flow runs long enough for one accepted batch to carry its
    retransmissions. The window's restore deadman fires before the fixture's
    deadman deletes the required test interface. The package deadman is armed
    before the install (at most ``DEPS_TIMEOUT``) and the fixture's after it,
    so its offset covers that too: it always fires last, once nothing uses the
    module any more.
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
        "deps_deadman": min(5400, fixture_deadman + DEPS_TIMEOUT + 300),
    }


def checked_cleanup(ctx: VariantContext, probe: str, *arguments: str) -> Any:
    """Run a probe cleanup and fail the cleanup stack unless it proved clean."""

    timeout = DEPS_CLEANUP_TIMEOUT if probe == "deps" else 180
    value = getattr(ctx, probe)("cleanup", *arguments, timeout=timeout)
    if value.get("clean") is not True:
        raise RegionalFixtureError(
            f"the {probe} cleanup left residue: {value.get('residue')}"
        )
    return value


def not_run(refused: dict[str, Any], default: str) -> VariantNotRun:
    reason = str(refused.get("reason") or "")
    return VariantNotRun(
        reason if reason in NOT_RUN_REASONS else "FIXTURE_NOT_READY",
        str(refused.get("detail") or default),
    )


def ensure_dependencies(
    ctx: VariantContext, survey: dict[str, Any], timing: dict[str, int]
) -> dict[str, Any]:
    """``tc`` and a loaded ``sch_netem``: as found, or for this run only."""

    tools = survey.get("tools") or {}
    module = (survey.get("modules") or {}).get("sch_netem") or {}
    approvals = ctx.settings.approvals
    may_install = PACKAGE_APPROVAL in approvals
    need_tool = not tools.get("tc")
    need_module = not module.get("loaded")
    if need_tool and not may_install:
        raise VariantNotRun(
            "CAPABILITY_ABSENT",
            f"the node lacks tc ({PACKAGE_APPROVAL} installs iproute-tc for one run)",
        )
    if need_module and not module.get("available") and not may_install:
        raise VariantNotRun(
            "CAPABILITY_ABSENT",
            "kernel module sch_netem is not available "
            f"({PACKAGE_APPROVAL} installs the running kernel's module package)",
        )
    if need_module and MODULE_APPROVAL not in approvals:
        raise VariantNotRun(
            "NOT_AUTHORIZED",
            "kernel module sch_netem is not loaded; needs KERNEL_MODULE_LOAD",
        )
    if not (need_tool or need_module):
        return {"changed": False, "installed": [], "modules_loaded_by_run": []}
    arguments = ["ensure", "--tool", "tc", "--module", "sch_netem"]
    arguments += ["--deadman-seconds", str(timing["deps_deadman"])]
    if may_install:
        arguments.append("--allow-install")
    if MODULE_APPROVAL in approvals:
        arguments.append("--allow-module-load")
    require_action_time(DEPS_TIMEOUT)
    # Registered before the change: a half-finished install is undone too.
    ctx.cleanup.push("deps", lambda: checked_cleanup(ctx, "deps"))
    result = ctx.deps(*arguments, timeout=DEPS_TIMEOUT)
    ctx.evidence.write("tcp-dependencies", result)
    if result.get("refused"):
        raise not_run(result["refused"], "the dependency step refused")
    netem = (result.get("modules") or {}).get("sch_netem") or {}
    if not (result.get("tools") or {}).get("tc") or not (
        netem.get("loaded") or netem.get("builtin")
    ):
        raise RegionalFixtureError(f"tc/sch_netem still unusable: {result}")
    return {"changed": True, **result}


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


def tcp_findings(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Every NIC or TCP finding the product minted in ``document``."""

    return [
        item
        for metric in (*NIC_METRICS, TCP_METRIC)
        for item in finding_incidents(document, metric)
    ]


def product_retransmits(
    document: dict[str, Any], since: datetime, until: datetime | None = None
) -> list[Any]:
    """Accepted ``tcp_retransmits_delta`` > 0 samples from inside the flow.

    ``since`` is one collector interval after the flow started, so the
    sample's whole delta interval lies inside the flow; ``until`` is when it
    was stopped, so a later batch carrying only background retransmissions
    never counts.
    """

    result = []
    for sample in samples(document, TCP_METRIC):
        observed = parse_time(sample.get("observed_at"))
        value = sample.get("value")
        if observed is None or observed < since or not isinstance(value, (int, float)):
            continue
        if value > 0 and (until is None or observed <= until):
            result.append(sample)
    return result


def watch(
    ctx: VariantContext,
    started: datetime,
    *,
    seconds: float,
    name: str,
    until_retransmits: bool,
) -> dict[str, Any]:
    """Poll the product's records; stop early on a finding (or the batch)."""

    since = started + timedelta(seconds=ctx.host_interval)
    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = activity(ctx, started, metrics={*NIC_METRICS, TCP_METRIC})
        if tcp_findings(last):
            return last
        if until_retransmits and product_retransmits(last, since):
            return last
        return None

    value = ctx.wait_until(
        accept,
        timeout_seconds=seconds,
        poll_seconds=ctx.host_interval * 2,
        name=name,
    )
    return value if value is not None else last


def product_stages(
    ctx: VariantContext, observed: dict[str, Any], interface: str
) -> dict[str, list[str]]:
    nic = [
        error
        for metric in NIC_METRICS
        for error in chain_errors(
            observed, ChainExpectation(metric=metric, present=False)
        )
    ]
    readings = [
        f"the product read {metric}={sample['value']} on {interface}; the oracle read 0"
        for metric in ("network_errors_delta", "network_drops_delta")
        for sample in samples(observed, metric, device=interface)
        if isinstance(sample.get("value"), (int, float)) and sample["value"] != 0
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
        "test_interface_zero": readings,
        "hardware": hardware_errors(observed),
        "no_isolation": isolation,
    }


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
    watch(
        ctx,
        started,
        seconds=timing["flow_wait"],
        name="tcp-product",
        until_retransmits=True,
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
    observed = watch(
        ctx, started, seconds=TAIL_SECONDS, name="tcp-negative", until_retransmits=False
    )
    stages = product_stages(ctx, observed, interface)
    since = started + timedelta(seconds=ctx.host_interval)
    seen = product_retransmits(observed, since, stopped)
    close_window(ctx)
    ctx.cleanup.discard("collector-window")
    if not any(stages.values()) and not seen:
        raise VariantInconclusive(
            "the product accepted no host batch with tcp_retransmits_delta > 0 "
            "from inside the flow, so 'no finding' proves nothing"
        )
    errors = [f"{stage}: {error}" for stage, items in stages.items() for error in items]
    return {
        "verdict": verdict(stages),
        "stages": stages,
        "errors": errors,
        "notes": [
            "tcp_retransmits_delta is collected without a rule: retransmissions "
            "are congestion control, not a NIC fault"
        ],
        "device": interface,
        "oracle": {key: value for key, value in reading.items() if key != "errors"},
        "product_retransmits": seen[:5],
        "dependencies": {
            key: dependencies.get(key)
            for key in ("changed", "installed", "modules_loaded_by_run", "transaction")
        },
        "proof_scope": "native",
    }


__all__ = [
    "PACKAGE_APPROVAL",
    "checked_cleanup",
    "create_fixture",
    "ensure_dependencies",
    "product_retransmits",
    "product_stages",
    "tcp_findings",
    "tcp_only",
    "tcp_oracle",
    "tcp_timing",
]
