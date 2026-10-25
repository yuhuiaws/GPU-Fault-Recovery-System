"""GF-GENERAL-LIVE-008 NIC link down and 009 NIC errors/drops.

Both use *test* interfaces of ``general_live_net_probe.py`` -- host-local
``veth`` pairs and a ``dummy`` -- declared required to the host collector by a
``GPU_FAULT_REQUIRED_INTERFACES`` collector-window drop-in. The management ENI,
the VPC-CNI ENI, the EFA netdevs and pod veths are never touched: the net probe
refuses any interface with an address or a sysfs ``device`` link and records a
before/after digest of every real interface.

008 ``network_link_down`` is CRITICAL QUARANTINE; the node is returned through
the product's validated restore. 009 ``network_errors_delta`` is WARNING
(immediate) and ``network_drops_delta`` WARNING (60 s sustain), both
VALIDATE_FABRIC. The oracle reads the sysfs counters itself; a drop generator
that never moves the counters, or a ``tc netem`` flow that moves NIC counters,
is INCONCLUSIVE rather than a native pass. 009 ``tcp_only`` (its netns + netem
fixture, the temporary ``tc`` dependency and its oracle) is
``general_live_tcp.py``; 009 ``counter_reset`` (the one-process recreate of the
drop pair and its reset-tick oracle) is ``general_live_network_reset.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scripts.e2e.regional.general_live_common import (
    Oracle,
    activity,
    await_metric_recovery,
    await_oracle_crossing,
    cadence_record,
    close_window,
    open_window,
    restore_incidents,
    run_survey,
    sustained_detection_budget,
    sustained_negative_window,
    wait_for_incident,
    wait_settled,
)
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    VariantNotRun,
    require_module,
    require_tools,
    utc_now,
)
from scripts.e2e.regional.general_live_registry import (
    CAPABILITY,
    LIVE_REPLAY,
    NATIVE,
    OBSERVATION,
    VariantSpec,
)
from scripts.e2e.regional.general_live_source_replay import host_source_replay
from scripts.e2e.regional.general_live_tcp import PACKAGE_APPROVAL, tcp_only
from scripts.e2e.regional.general_live_network_reset import counter_reset
from scripts.e2e.regional.probes.collector_window_probe import MAX_RESTORE_SECONDS
from scripts.e2e.regional.general_live_verdicts import (
    DIAGNOSTIC_ONLY_STATES,
    ChainExpectation,
    chain_errors,
    chain_judgeable,
    finding_incidents,
    hardware_errors,
    node_isolated,
    oracle_values,
    samples,
    verdict,
)

CASE_008 = "GF-GENERAL-LIVE-008"
CASE_009 = "GF-GENERAL-LIVE-009"
QUARANTINE_BUDGET = 420
#: ``host_health.py`` NETWORK_DROPS_SUSTAIN_SECONDS: the drops rule's minimum
#: active time. ``network_drops_delta`` is a per-tick delta, so the generator
#: must keep dropping on every collector tick for the whole budget.
DROPS_SUSTAIN = 60
#: How long the generator may take to move the drop counters on the oracle:
#: a few 5-s oracle ticks (it drops from its first frame).
DROPS_RAMP_BUDGET = 90
SHORT_DROPS = 30


def drops_budget(ctx: VariantContext) -> int:
    """The drops detection budget from the oracle's crossing, on the product's cadence.

    ``threshold:network_drops_delta`` is shipped once, as the crossing's edge,
    then only on health summaries; the finding needs a later active sample at
    least 60 s after the first and VALIDATE_FABRIC one more summary: 60 s +
    2 x GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS + GPU_FAULT_HOST_INTERVAL_SECONDS
    + the chain margin (855 s on a 300-s/15-s site; the fixed 450-s budget
    before it, counted from the generator's start, could not reach the
    validation's summary).
    """

    return sustained_detection_budget(ctx, DROPS_SUSTAIN)


def drops_quiet_window(ctx: VariantContext) -> int:
    """The negative watch after a burst shorter than the sustain: the sustain
    plus one health summary plus a collector interval and the margin (555 s on
    a 300-s/15-s site)."""

    return sustained_negative_window(ctx, DROPS_SUSTAIN)


def link_quiet_window(ctx: VariantContext) -> int:
    """The negative watch for an interface the collector must not even sample:
    one health summary (the batch that would carry the sample) plus a
    collector interval and the margin (495 s on a 300-s/15-s site)."""

    return sustained_negative_window(ctx, 0)


def window_restore_seconds(ctx: VariantContext) -> int:
    """How long the required-interface drop-in may stay open before its deadman.

    The window must outlive the whole variant -- the drops budget is timed
    from the oracle's crossing and the recovery waits two more summaries --
    never less than the link variants' quarantine budget plus its old margin,
    and never more than the window probe's own cap.
    """

    wanted = max(QUARANTINE_BUDGET + 600, ctx.spec.max_minutes * 60 + 600)
    return min(MAX_RESTORE_SECONDS, wanted)


def _veth(ctx: VariantContext, *, a: str = "a", b: str = "b") -> dict[str, Any]:
    survey = run_survey(ctx)
    require_tools(survey, "ip", "systemd-run")
    require_module(survey, "veth", ctx.settings.approvals)
    created = ctx.net(
        "create-veth",
        "--slot",
        a,
        "--peer-slot",
        b,
        "--deadman-seconds",
        str(ctx.spec.max_minutes * 60 + 600),
    )
    ctx.cleanup.push("net-interfaces", lambda: ctx.net("cleanup", "--kind", "all"))
    ctx.evidence.write(f"veth-{a}{b}", created)
    return created


def _declare_required(ctx: VariantContext, *names: str) -> None:
    pairs = [f"GPU_FAULT_REQUIRED_INTERFACES={name}" for name in names]
    open_window(ctx, *pairs, restore_seconds=window_restore_seconds(ctx))


def _link_incidents(
    activity_document: dict[str, Any], *devices: str
) -> list[dict[str, Any]]:
    """The ``network_link_down`` incidents of the variant's test links.

    A veth's peer goes down with it: ``ip link set a down`` leaves the peer
    ``lowerlayerdown`` (NO-CARRIER) and ``ip link del a`` removes both ends.
    Both ends are declared required, so the peer reports ``network_link_down=1``
    in the same tick and the product folds that second critical finding into
    the isolation the first one opened (one incident, re-labelled
    NODE_HEALTH_GROUP; ``general_live_verdicts.NODE_HEALTH_EVENT_TYPES``). A
    peer finding the product only took in a later tick, after that isolation
    had ended, mints an incident of its own -- one this variant caused too, so
    it is handed back through the validated restore as well.
    """

    found: dict[str, dict[str, Any]] = {}
    for device in devices:
        for item in finding_incidents(
            activity_document, "network_link_down", device=device
        ):
            found.setdefault(str(item["incident_id"]), item)
    return list(found.values())


def _settled_isolation(
    ctx: VariantContext,
    started: Any,
    detection: dict[str, Any],
    devices: tuple[str, ...],
    *,
    name: str,
) -> dict[str, Any]:
    """The read the isolation chain is graded on.

    The detection poll returns the first read that lists the incident, and
    the product's isolation workflow (FREEZE_EVIDENCE, MARK_UNSCHEDULABLE,
    QUARANTINE) ends seconds after it, so that read can still show the
    incident ACTION_PENDING behind a RUNNING workflow; the incident is
    QUARANTINED once the workflow is terminal. Without an incident there is
    nothing to settle and the detection read carries the failure.
    """

    incidents = _link_incidents(detection, *devices)
    if not incidents:
        return detection
    return wait_settled(
        ctx,
        started,
        [item["incident_id"] for item in incidents],
        timeout_seconds=QUARANTINE_BUDGET,
        metrics={"network_link_down", "network_link_up"},
        name=name,
    )


def _isolation_chain(settled: dict[str, Any], device: str) -> list[str]:
    return chain_errors(
        settled,
        ChainExpectation(
            metric="network_link_down",
            device=device,
            operations=("MARK_UNSCHEDULABLE", "QUARANTINE"),
            forbidden=_hardware() | {"STOP_WORKLOADS"},
            incident_states=frozenset({"QUARANTINED"}),
        ),
    )


def required_link_down(ctx: VariantContext) -> dict[str, Any]:
    created = _veth(ctx)
    a, b = created["a"], created["b"]
    _declare_required(ctx, a, b)
    ctx.hold(ctx.host_interval * 3, name="healthy-samples")
    started = utc_now()
    ctx.net("set-link", "--slot", "a", "--state", "down")
    # network_link_down is an immediate rule (no sustain window) and the down
    # tick opens a ``threshold:network_link_down`` edge the host collector
    # delivers at once, so the incident follows within one collector interval
    # (live 2026-10-25: minted from the first down tick). The edge is per
    # metric, not per device: with another network_link_down already open on
    # the node the tick waits for the next health summary (300 s, jitter 0.1),
    # which QUARANTINE_BUDGET still covers.
    detection = wait_for_incident(
        ctx,
        started,
        "network_link_down",
        device=a,
        timeout_seconds=QUARANTINE_BUDGET,
        metrics={"network_link_down", "network_link_up"},
        name="link-quarantine",
    )
    reading = ctx.net("reading", "--slot", "a", "--slot", "b")
    ctx.evidence.write("link-reading", reading)
    links = reading.get("interfaces") or {}
    link = links.get(a) or {}
    if link.get("operstate") == "up":
        raise VariantInconclusive(f"the test link {a} did not go down: {link}")
    settled = _settled_isolation(ctx, started, detection, (a, b), name="link-settled")
    stages = {"chain": _isolation_chain(settled, a)}
    # Recovery order: bring the link up and let the product read it, close the
    # window, then validated restore; deleting the interface before that would
    # be a fresh interface_missing event.
    stages["link_restored"] = _link_up(ctx, a, b)
    close_window(ctx)
    ctx.cleanup.discard("collector-window")
    restore_incidents(
        ctx,
        _link_incidents(settled, a, b),
        reason="GF-GENERAL-LIVE-008 required_link_down restore",
    )
    return _result(stages, extra={"peer_link": links.get(b)})


def interface_missing(ctx: VariantContext) -> dict[str, Any]:
    created = _veth(ctx)
    a, b = created["a"], created["b"]
    _declare_required(ctx, a, b)
    ctx.hold(ctx.host_interval * 3, name="healthy-samples")
    started = utc_now()
    ctx.net("delete-interface", "--slot", "a")
    detection = wait_for_incident(
        ctx,
        started,
        "network_link_down",
        device=a,
        timeout_seconds=QUARANTINE_BUDGET,
        metrics={"network_link_down"},
        name="missing-quarantine",
    )
    settled = _settled_isolation(
        ctx, started, detection, (a, b), name="missing-settled"
    )
    stages = {
        "chain": _isolation_chain(settled, a),
        "link_restored": _restore_pair(ctx, a, b),
    }
    close_window(ctx)
    ctx.cleanup.discard("collector-window")
    restore_incidents(
        ctx,
        _link_incidents(settled, a, b),
        reason="GF-GENERAL-LIVE-008 interface_missing restore",
    )
    return _result(stages)


def _restore_pair(ctx: VariantContext, a: str, b: str) -> list[str]:
    """Put the deleted pair back while it is still required (``ip link del``
    removed both ends) and wait until the product has read both up again."""

    since = utc_now()
    recreated = ctx.net(
        "create-veth",
        "--slot",
        "a",
        "--peer-slot",
        "b",
        "--deadman-seconds",
        str(ctx.spec.max_minutes * 60 + 600),
    )
    ctx.evidence.write("veth-ab-restored", recreated)
    return _await_pair_up(ctx, since, a, b)


def _link_up(ctx: VariantContext, a: str, b: str) -> list[str]:
    """Bring the downed end back up and wait until the product read both up."""

    since = utc_now()
    ctx.net("set-link", "--slot", "a", "--state", "up")
    return _await_pair_up(ctx, since, a, b)


def _await_pair_up(ctx: VariantContext, since: Any, a: str, b: str) -> list[str]:
    """Fresh product evidence that both ends of the pair read up again.

    The validated restore judges the node on the product's newest link
    readings, and closing the collector window restarts the collector without
    the pair: a change the collector had not read by then never reaches the
    product. Live 2026-10-25 (ip-10-0-43-5, interface_missing) the window was
    closed with the pair still deleted, the newest readings of both ends
    stayed "missing" (``network_link_down=1``, ``network_link_up=0``) and the
    restore ended FAILED; ``required_link_down`` and ``recovery`` closed it a
    few seconds after ``set-link up`` and only passed because a collector tick
    happened to fall in between. The spec's recovery puts the original
    interface configuration back before the controlled readmission. The ends
    coming back clear the ``threshold:network_link_down`` edge, which the
    collector delivers at once; ``await_metric_recovery`` allows two health
    summaries.
    """

    errors: list[str] = []
    for slot, device in (("a", a), ("b", b)):
        recovered = await_metric_recovery(
            ctx,
            "network_link_down",
            1.0,
            device=device,
            since=since,
            name=f"link-restored-{slot}",
        )
        errors.extend(recovered["errors"])
    return errors


def unknown_state(ctx: VariantContext) -> dict[str, Any]:
    survey = run_survey(ctx)
    require_tools(survey, "ip")
    require_module(survey, "dummy", ctx.settings.approvals)
    created = ctx.net(
        "create-dummy",
        "--slot",
        "d",
        "--deadman-seconds",
        str(ctx.spec.max_minutes * 60 + 600),
    )
    ctx.cleanup.push("net-interfaces", lambda: ctx.net("cleanup", "--kind", "all"))
    name = created["name"]
    link = created.get("link") or {}
    if link.get("operstate") != "unknown":
        raise VariantInconclusive(
            f"the dummy {name} did not read operstate=unknown: {link}"
        )
    _declare_required(ctx, name)
    started = utc_now()
    observed = _await_collection_error(ctx, started)
    # A HostObservationError -> collection error -> UNKNOWN; no quarantine.
    stages = {
        "no_quarantine": chain_errors(
            observed,
            ChainExpectation(metric="network_link_down", device=name, present=False),
        ),
        "collection_error": []
        if _has_collection_error(observed)
        else ["the unknown operstate did not raise a host collection error"],
    }
    close_window(ctx)
    ctx.cleanup.discard("collector-window")
    return _result(
        stages,
        notes=["X4: a collection error also stalls this node's other validations"],
    )


def optional_interface(ctx: VariantContext) -> dict[str, Any]:
    created = _veth(ctx, a="c", b="b")
    c = created["a"]
    # c is not declared required and has no device link, so the collector never
    # samples it; bringing it down must mint no finding.
    started = utc_now()
    ctx.net("set-link", "--slot", "c", "--state", "down")
    observed = _watch_no_finding(
        ctx, started, "network_link_down", link_quiet_window(ctx), device=c
    )
    stages = {
        "no_finding": chain_errors(
            observed,
            ChainExpectation(metric="network_link_down", device=c, present=False),
        ),
        "hardware": hardware_errors(observed),
    }
    return _result(stages)


#: Interface roles of the net probe's ``inventory`` (008 ``empty_required_list``).
ROLES = ("management", "pod_network", "virtual", "dedicated")


def interface_roles(inventory: Mapping[str, Any]) -> dict[str, list[str]]:
    """Which real netdevs are a dedicated data NIC the required list could name.

    From the probe's read-only ``inventory`` of every netdev but ``lo`` and the
    run's own test interfaces: ``management`` carries the main table's default
    route; ``pod_network`` is a device-backed netdev that is the default route
    of a policy routing table and not of the main table -- the pattern the pod
    network sets up (the VPC CNI keeps one table per secondary ENI), so it is
    the CNI's, not an admin's; ``virtual`` has no sysfs ``device`` link
    (host-side pod veths, bridges, dummies, tunnels); ``dedicated`` is every
    other device-backed netdev -- an RDMA-capable netdev, a NIC of another
    driver, an admin-configured secondary interface routed in the main table --
    the candidates ``GPU_FAULT_REQUIRED_INTERFACES`` exists for. Generic: no
    site names, only sysfs and routing facts.
    """

    roles: dict[str, list[str]] = {role: [] for role in ROLES}
    for name, info in sorted((inventory.get("interfaces") or {}).items()):
        info = info if isinstance(info, Mapping) else {}
        if not info.get("has_device"):
            roles["virtual"].append(name)
        elif info.get("default_route"):
            roles["management"].append(name)
        elif info.get("policy_tables") and not info.get("rdma"):
            roles["pod_network"].append(name)
        else:
            roles["dedicated"].append(name)
    return roles


def empty_required_list(ctx: VariantContext) -> dict[str, Any]:
    """Observation: the live required-interface list is empty.

    A real NIC link-down can then never isolate the node -- the spec's site
    FAIL -- but only where the node has a dedicated data NIC to declare. A node
    whose real netdevs are the default-route interface and the pod network's
    secondary ENIs (everything else a container/virtual interface) has nothing
    to declare: an empty list is the correct configuration there, and the
    variant is NOT_RUN(CAPABILITY_ABSENT) rather than a FAIL. The decision is
    read from the node (``interface_roles`` over the net probe's ``inventory``),
    never from a site name.
    """

    survey = run_survey(ctx)
    env = survey.get("collector_env") or {}
    configured = str(env.get("GPU_FAULT_REQUIRED_INTERFACES") or "").strip()
    require_tools(survey, "ip")
    inventory = ctx.net("inventory")
    roles = interface_roles(inventory)
    ctx.evidence.write(
        "interface-roles",
        {
            "configured_required_interfaces": configured,
            "roles": roles,
            "inventory": inventory,
        },
    )
    if not configured and not roles["dedicated"]:
        raise VariantNotRun(
            "CAPABILITY_ABSENT",
            "no dedicated data NIC on this node (default route "
            f"{roles['management']}, pod network {roles['pod_network']}, "
            f"{len(roles['virtual'])} virtual/container interfaces); an empty "
            "GPU_FAULT_REQUIRED_INTERFACES is the correct configuration",
        )
    stages = {
        "required_list": [
            "GPU_FAULT_REQUIRED_INTERFACES is empty, so no real NIC link-down can "
            f"isolate this node (the installer default is empty) while it has the "
            f"dedicated data netdev(s) {roles['dedicated']} to declare"
        ]
        if not configured
        else []
    }
    return _result(
        stages,
        notes=[
            "remedy: an admin/installer decision to declare the dedicated data "
            f"netdev(s) {roles['dedicated']} (and the primary/EFA netdevs where "
            "the site has them)"
        ],
        extra={"configured_required_interfaces": configured, "interface_roles": roles},
    )


def link_recovery(ctx: VariantContext) -> dict[str, Any]:
    created = _veth(ctx)
    a, b = created["a"], created["b"]
    _declare_required(ctx, a, b)
    ctx.hold(ctx.host_interval * 3, name="healthy-samples")
    started = utc_now()
    ctx.net("set-link", "--slot", "a", "--state", "down")
    detection = wait_for_incident(
        ctx,
        started,
        "network_link_down",
        device=a,
        timeout_seconds=QUARANTINE_BUDGET,
        metrics={"network_link_down"},
        name="recovery-detection",
    )
    incidents = _link_incidents(detection, a, b)
    link_restored = _link_up(ctx, a, b)
    close_window(ctx)
    ctx.cleanup.discard("collector-window")
    restored = restore_incidents(
        ctx, incidents, reason="GF-GENERAL-LIVE-008 recovery restore"
    )
    operations = [op for item in restored for op in item.get("operations") or []]
    stages = {
        "link_restored": link_restored,
        "restore": []
        if "VALIDATE_FABRIC" in operations and "RESTORE_SCHEDULING" in operations
        else [
            "the validated restore did not run VALIDATE_FABRIC and RESTORE_SCHEDULING"
        ],
    }
    if not incidents:
        stages["restore"].append("no link-down finding to recover from")
    return _result(stages, extra={"restore_operations": operations})


# --------------------------------------------------------------------------- #
# 009 errors/drops
# --------------------------------------------------------------------------- #
def _drops_oracle(ctx: VariantContext, a: str, b: str, seconds: int) -> Oracle:
    oracle = Oracle(ctx, "net")
    oracle.start(seconds + 30, interval=5, netdev=[a, b])
    return oracle


def _dropped_frames(a: str, b: str) -> Any:
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


def _await_drops_crossing(ctx: VariantContext, oracle: Oracle, a: str, b: str) -> None:
    """Detection is timed from the oracle's first rise of the drop counters."""

    await_oracle_crossing(
        ctx,
        oracle,
        _dropped_frames(a, b),
        lambda values: len(values) >= 2 and values[-1] > values[0],
        timeout_seconds=DROPS_RAMP_BUDGET,
        name="drops-ramp",
        condition=f"the generator did not move the drop counters of {a}/{b}",
    )


def _start_drops(
    ctx: VariantContext, slot: str, seconds: int, *, budget_seconds: int
) -> None:
    """Start the drop generator; ``drops-start.json`` records the budgets."""

    started = ctx.net(
        "start-drops", "--slot", slot, "--rate", "10", "--seconds", str(seconds)
    )
    ctx.cleanup.push("net-drops", lambda: ctx.net("stop-drops"))
    ctx.evidence.write(
        "drops-start",
        {
            **started,
            **cadence_record(
                ctx,
                sustain_seconds=DROPS_SUSTAIN,
                detection_budget_seconds=budget_seconds,
                ramp_budget_seconds=DROPS_RAMP_BUDGET,
            ),
        },
    )


def _assert_drops_moved(oracle: Oracle, a: str, b: str) -> None:
    oracle.read()
    tx = oracle_values(
        [record.get("netdev", {}).get(a, {}) for record in oracle.records],
        lambda record: record.get("tx_dropped"),
    )
    rx = oracle_values(
        [record.get("netdev", {}).get(b, {}) for record in oracle.records],
        lambda record: record.get("rx_dropped"),
    )
    if not (len(tx) >= 2 and tx[-1] > tx[0]) and not (len(rx) >= 2 and rx[-1] > rx[0]):
        raise VariantInconclusive(
            f"the drop counters did not rise: tx={tx[-4:]}, rx={rx[-4:]}"
        )


def sustained_drops(ctx: VariantContext) -> dict[str, Any]:
    survey = run_survey(ctx)
    require_module(survey, "veth", ctx.settings.approvals)
    created = _veth(ctx)
    a, b = created["a"], created["b"]
    _declare_required(ctx, a, b)
    ctx.hold(ctx.host_interval * 3, name="healthy-samples")
    started = utc_now()
    budget = drops_budget(ctx)
    # The generator outlives the ramp, the detection budget and the chain's
    # grading: VALIDATE_FABRIC must still find the interface dropping.
    hold = DROPS_RAMP_BUDGET + budget + 120
    oracle = _drops_oracle(ctx, a, b, hold)
    _start_drops(ctx, "a", hold, budget_seconds=budget)
    _await_drops_crossing(ctx, oracle, a, b)
    detection = _wait_drops_finding(ctx, started, (a, b), validation="VALIDATE_FABRIC")
    _assert_drops_moved(oracle, a, b)
    device = _drops_device(detection, (a, b))
    incidents = finding_incidents(detection, "network_drops_delta", device=device)
    stages = {
        "chain": chain_errors(
            detection,
            ChainExpectation(
                metric="network_drops_delta",
                device=device,
                operations=("FREEZE_EVIDENCE", "VALIDATE_FABRIC"),
                validation="VALIDATE_FABRIC",
                validation_must_fail=True,
                incident_states=DIAGNOSTIC_ONLY_STATES,
            ),
        ),
        "hardware": hardware_errors(detection),
    }
    ctx.net("stop-drops")
    ctx.cleanup.discard("net-drops")
    recovered = _await_zero_drops(ctx, device)
    stages["recovery"] = recovered["errors"]
    wait_settled(
        ctx,
        started,
        [item["incident_id"] for item in incidents],
        timeout_seconds=budget,
        metrics={"network_drops_delta"},
        name="drops-settled",
    )
    return _result(stages, extra={"device": device, "recovery": recovered})


def short_drops(ctx: VariantContext) -> dict[str, Any]:
    created = _veth(ctx)
    a, b = created["a"], created["b"]
    _declare_required(ctx, a, b)
    started = utc_now()
    oracle = _drops_oracle(ctx, a, b, SHORT_DROPS + 10)
    ctx.net("start-drops", "--slot", "a", "--rate", "10", "--seconds", str(SHORT_DROPS))
    ctx.cleanup.push("net-drops", lambda: ctx.net("stop-drops"))
    ctx.hold(SHORT_DROPS + 10, name="short-drops")
    ctx.net("stop-drops")
    ctx.cleanup.discard("net-drops")
    _assert_drops_moved(oracle, a, b)
    # The burst's end is a recovered edge; the watch outlasts the sustain plus
    # the next health summary, the only later batch that could carry a delta.
    window = drops_quiet_window(ctx)
    observed = _watch_no_finding(ctx, started, "network_drops_delta", window)
    stages = {
        "no_finding": chain_errors(
            observed, ChainExpectation(metric="network_drops_delta", present=False)
        )
    }
    close_window(ctx)
    ctx.cleanup.discard("collector-window")
    return _result(
        stages,
        extra={
            "negative_window_seconds": window,
            "host_summary_seconds": ctx.host_summary,
            "host_interval_seconds": ctx.host_interval,
        },
    )


def wrong_interface(ctx: VariantContext) -> dict[str, Any]:
    first = _veth(ctx, a="a", b="b")
    second = ctx.net(
        "create-veth",
        "--slot",
        "c",
        "--peer-slot",
        "d",
        "--deadman-seconds",
        str(ctx.spec.max_minutes * 60 + 600),
    )
    a, b, c, d = first["a"], first["b"], second["a"], second["b"]
    _declare_required(ctx, a, b, c, d)
    ctx.hold(ctx.host_interval * 3, name="healthy-samples")
    started = utc_now()
    budget = drops_budget(ctx)
    # No oracle here: the generator drops from its first frame, so the edge is
    # within one collector tick of the start; the budget still covers the two
    # summaries the finding needs after it.
    _start_drops(ctx, "c", DROPS_RAMP_BUDGET + budget + 60, budget_seconds=budget)
    detection = _wait_drops_finding(ctx, started, (c, d))
    device = _drops_device(detection, (c, d))
    on_a = finding_incidents(detection, "network_drops_delta", device=a)
    stages = {
        "wrong_interface": [
            f"a drops finding named interface {a}, not the injected pair"
        ]
        if on_a
        else [],
        "right_interface": []
        if device in (c, d)
        else [f"no drops finding named the injected pair ({c}/{d})"],
        "hardware": hardware_errors(detection),
    }
    ctx.net("stop-drops")
    ctx.cleanup.discard("net-drops")
    close_window(ctx)
    ctx.cleanup.discard("collector-window")
    return _result(
        stages,
        notes=["the generic VALIDATE_FABRIC is node-wide"],
        extra={"device": device},
    )


def drops_recovery(ctx: VariantContext) -> dict[str, Any]:
    created = _veth(ctx)
    a, b = created["a"], created["b"]
    _declare_required(ctx, a, b)
    ctx.hold(ctx.host_interval * 3, name="healthy-samples")
    started = utc_now()
    budget = drops_budget(ctx)
    hold = DROPS_RAMP_BUDGET + budget + 60
    oracle = _drops_oracle(ctx, a, b, hold)
    _start_drops(ctx, "a", hold, budget_seconds=budget)
    _await_drops_crossing(ctx, oracle, a, b)
    detection = _wait_drops_finding(ctx, started, (a, b))
    device = _drops_device(detection, (a, b))
    incidents = finding_incidents(detection, "network_drops_delta", device=device)
    ctx.net("stop-drops")
    ctx.cleanup.discard("net-drops")
    recovered = _await_zero_drops(ctx, device)
    wait_settled(
        ctx,
        started,
        [item["incident_id"] for item in incidents],
        timeout_seconds=budget,
        metrics={"network_drops_delta"},
        name="drops-recovery-settled",
    )
    stages = {"recovery": recovered["errors"], "hardware": hardware_errors(detection)}
    if not incidents:
        stages["recovery"].append("no drops finding to recover from")
    close_window(ctx)
    ctx.cleanup.discard("collector-window")
    return _result(stages, extra={"device": device, "recovery": recovered})


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _hardware() -> frozenset[str]:
    from scripts.e2e.regional.general_live_registry import HARDWARE_OPERATIONS

    return HARDWARE_OPERATIONS


def _wait_drops_finding(
    ctx: VariantContext,
    since: Any,
    interfaces: tuple[str, ...],
    *,
    validation: str | None = None,
) -> dict[str, Any]:
    """The drops incident on one of ``interfaces``; with ``validation``, its
    chain judgeable (the step terminal), not the first snapshot showing it."""

    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = activity(ctx, since, metrics={"network_drops_delta"})
        for interface in interfaces:
            if validation is not None:
                ready = chain_judgeable(
                    last, "network_drops_delta", device=interface, validation=validation
                )
            else:
                ready = bool(
                    finding_incidents(last, "network_drops_delta", device=interface)
                )
            if ready:
                return last
        return None

    value = ctx.wait_until(
        accept,
        timeout_seconds=drops_budget(ctx),
        poll_seconds=ctx.host_interval,
        name="drops-detection",
    )
    return value if value is not None else last


def _drops_device(
    activity_document: dict[str, Any], interfaces: tuple[str, ...]
) -> str | None:
    for interface in interfaces:
        if finding_incidents(
            activity_document, "network_drops_delta", device=interface
        ):
            return interface
    return None


def _await_zero_drops(ctx: VariantContext, device: str | None) -> dict[str, Any]:
    started = utc_now()

    def accept() -> dict[str, Any] | None:
        current = activity(ctx, started, metrics={"network_drops_delta"})
        from scripts.e2e.regional.general_live_verdicts import samples

        zero = [
            sample
            for sample in samples(current, "network_drops_delta", device=device)
            if isinstance(sample.get("value"), (int, float)) and sample["value"] == 0
        ]
        return {"samples": zero} if zero else None

    value = ctx.wait_until(
        accept,
        timeout_seconds=ctx.host_summary + ctx.host_interval * 6,
        poll_seconds=ctx.host_interval,
        name="drops-recovery",
    )
    if value is None:
        return {"errors": [f"no fresh zero network_drops_delta on {device}"]}
    return {"errors": [], "fresh_samples": value["samples"]}


def _await_collection_error(ctx: VariantContext, since: Any) -> dict[str, Any]:
    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = activity(ctx, since, metrics={"network_link_down"})
        return last if _has_collection_error(last) else None

    value = ctx.wait_until(
        accept,
        timeout_seconds=QUARANTINE_BUDGET,
        poll_seconds=ctx.host_interval,
        name="collection-error",
    )
    return value if value is not None else last


def _has_collection_error(activity_document: dict[str, Any]) -> bool:
    for status in activity_document.get("collector_statuses") or []:
        if str(status.get("collector")) == "HOST_TELEMETRY" and (
            status.get("last_error_at") or status.get("errors")
        ):
            return True
    for batch in activity_document.get("batches") or []:
        if batch.get("collection_errors"):
            return True
    return False


def _watch_no_finding(
    ctx: VariantContext,
    since: Any,
    metric: str,
    window_seconds: int,
    *,
    device: str | None = None,
) -> dict[str, Any]:
    last = activity(ctx, since, metrics={metric})

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = activity(ctx, since, metrics={metric})
        return last if finding_incidents(last, metric, device=device) else None

    hit = ctx.wait_until(
        accept,
        timeout_seconds=window_seconds,
        poll_seconds=ctx.host_interval * 2,
        name="negative-window",
    )
    return hit if hit is not None else last


def _result(
    stages: dict[str, list[str]],
    *,
    notes: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "verdict": verdict(stages),
        "stages": stages,
        "errors": [
            f"{stage}: {error}" for stage, items in stages.items() for error in items
        ],
        "notes": notes or [],
        **(extra or {}),
    }


def _bind_statistics(
    ctx: VariantContext, interface: str, name: str, content: bytes
) -> Any:
    """Bind a reviewed statistics counter of one gfrs test interface (live-replay)."""

    replay = host_source_replay(ctx)
    replay.create()
    replay.install(name, content)
    dest = f"/sys/class/net/{interface}/statistics/rx_errors"
    opened = replay.open_bind([(name, dest)], restore_seconds=ctx.spec.max_minutes * 60)
    ctx.evidence.write("statistics-bind", opened)
    return replay


def errors_delta(ctx: VariantContext) -> dict[str, Any]:
    """Live-replay: a reviewed rx_errors counter stepped up -> WARNING + fabric diagnostic."""

    created = _veth(ctx)
    a = created["a"]
    _declare_required(ctx, a)
    ctx.hold(ctx.host_interval * 3, name="healthy-samples")
    replay = _bind_statistics(ctx, a, "rx-errors", b"0\n")
    # One poll at zero establishes the error-delta baseline, then step it up.
    ctx.hold(ctx.host_interval * 2 + 10, name="errors-baseline")
    started = utc_now()
    replay.set_value("rx-errors", 5000)
    detection = wait_for_incident(
        ctx,
        started,
        "network_errors_delta",
        device=a,
        timeout_seconds=QUARANTINE_BUDGET,
        metrics={"network_errors_delta"},
        name="errors-delta",
    )
    status = replay.status()
    ctx.evidence.append_oracle(
        [{"rx_errors_fixture": (status.get("fixtures") or {}).get("rx-errors")}],
        source="rx-errors",
    )
    incidents = finding_incidents(detection, "network_errors_delta", device=a)
    stages = {
        "chain": chain_errors(
            detection,
            ChainExpectation(
                metric="network_errors_delta",
                device=a,
                operations=("FREEZE_EVIDENCE", "VALIDATE_FABRIC"),
            ),
        ),
        "hardware": hardware_errors(detection),
    }
    if not incidents:
        stages["chain"].append("no network_errors_delta finding reached an incident")
    return _result(
        stages,
        notes=[
            "the bound rx_errors counter stepped up; the delta is a WARNING fabric diagnostic"
        ],
        extra={"device": a, "proof_scope": "live-replay"},
    )


def statistics_failed(ctx: VariantContext) -> dict[str, Any]:
    """Live-replay: an unreadable rx_errors counter keeps the link sample, not 'down'."""

    created = _veth(ctx)
    a = created["a"]
    _declare_required(ctx, a)
    ctx.hold(ctx.host_interval * 3, name="healthy-samples")
    started = utc_now()
    _bind_statistics(ctx, a, "rx-errors", b"not-a-number\n")
    ctx.hold(ctx.host_interval * 4 + 20, name="stats-failed")
    observed = activity(
        ctx,
        started,
        metrics={"network_link_up", "network_errors_delta", "network_drops_delta"},
        kinds=("HOST_TELEMETRY",),
    )
    link = samples(observed, "network_link_up", device=a)
    stages = {
        "link_kept": []
        if link
        else ["the link sample was dropped when the statistics read failed"],
        "no_false_fault": [
            f"a statistics read failure produced a {metric} finding on {a}"
            for metric in ("network_errors_delta", "network_drops_delta")
            if finding_incidents(observed, metric, device=a)
        ],
        "not_isolated": ["node isolated on a statistics read failure"]
        if node_isolated(ctx.regional.node_snapshot(ctx.settings.node))
        else [],
        "hardware": hardware_errors(observed),
    }
    return _result(
        stages,
        notes=["an unreadable counter is UNKNOWN; the link state is still reported"],
        extra={"device": a, "proof_scope": "live-replay"},
    )


def _spec(
    case: str,
    variant: str,
    injection: str,
    expected: str,
    execute,
    *,
    risk="M",
    lanes=("eks", "hp"),
    max_minutes=18,
    outcome="PASS",
    gap="",
    approvals=(),
    optional_approvals=(),
    proof=NATIVE,
    flags="S",
    cleanup="restore the link/config; delete the test interfaces after closing the window",
) -> VariantSpec:
    return VariantSpec(
        case_id=case,
        variant=variant,
        proof_mode=proof,
        lanes=lanes,
        risk=risk,
        flags=flags,
        injection=injection,
        expected=expected,
        cleanup=cleanup,
        max_minutes=max_minutes,
        execute=execute,
        expected_outcome=outcome,
        product_gap=gap,
        approvals=tuple(approvals),
        optional_approvals=tuple(optional_approvals),
    )


VARIANTS = [
    _spec(
        CASE_008,
        "required_link_down",
        "declare a veth pair required, 3 healthy samples, then ip link set a down",
        "network_link_down=1 -> MARK_UNSCHEDULABLE, QUARANTINE; no STOP; validated restore",
        required_link_down,
        approvals=("KERNEL_MODULE_LOAD",),
    ),
    _spec(
        CASE_008,
        "interface_missing",
        "ip link del a while it is required",
        "link_down=1 for the absent interface -> QUARANTINE; validated restore",
        interface_missing,
        approvals=("KERNEL_MODULE_LOAD",),
    ),
    _spec(
        CASE_008,
        "unknown_state",
        "a required dummy reads operstate=unknown",
        "HostObservationError -> collection error -> UNKNOWN; no QUARANTINE (X4 stalls validations)",
        unknown_state,
        risk="L",
        approvals=("KERNEL_MODULE_LOAD",),
    ),
    _spec(
        CASE_008,
        "statistics_failed",
        "bind a non-numeric rx_errors over a required gfrs test interface",
        "the link sample is kept and reported UNKNOWN; no false errors/drops finding, no isolation",
        statistics_failed,
        risk="L",
        max_minutes=14,
        proof=LIVE_REPLAY,
        approvals=("KERNEL_MODULE_LOAD",),
    ),
    _spec(
        CASE_008,
        "optional_interface",
        "a non-required veth is set down",
        "no finding (not even sampled)",
        optional_interface,
        risk="L",
        approvals=("KERNEL_MODULE_LOAD",),
    ),
    _spec(
        CASE_008,
        "empty_required_list",
        "read GPU_FAULT_REQUIRED_INTERFACES on the live node and the roles of its real "
        "netdevs (net probe inventory: device links, drivers, RDMA, the main default "
        "route, pod-network policy tables)",
        "the list is empty while the node has a dedicated data NIC to declare, so a real "
        "NIC link-down can never isolate (spec FAIL, site config); a node whose real "
        "netdevs are only the default-route interface and the pod network's secondary "
        "ENIs has nothing to declare: NOT_RUN(CAPABILITY_ABSENT), an empty list is the "
        "correct configuration there",
        empty_required_list,
        risk="L",
        proof=OBSERVATION,
        flags="P",
        outcome="FAIL",
        gap="the installer default GPU_FAULT_REQUIRED_INTERFACES is empty (install-gpu-fault-collector.sh:56)",
    ),
    VariantSpec(
        case_id=CASE_008,
        variant="bond_vf",
        proof_mode=CAPABILITY,
        lanes=("eks",),
        risk="L",
        flags="P",
        injection="not executed",
        expected="NOT_RUN(CAPABILITY_ABSENT): no bond/VF semantics and no SR-IOV VF netdev on this site",
        cleanup="none",
        max_minutes=1,
        deferred="CAPABILITY_ABSENT: the product has no bond/VF semantics and the site has no SR-IOV VF netdev",
    ),
    _spec(
        CASE_008,
        "recovery",
        "bring the required link back, then validated restore",
        "link_down=0 -> VALIDATE_FABRIC -> RESTORE_SCHEDULING; real ENIs untouched",
        link_recovery,
        approvals=("KERNEL_MODULE_LOAD",),
    ),
    _spec(
        CASE_009,
        "errors_delta",
        "bind a reviewed rx_errors counter on a required gfrs test interface and step it up",
        "immediate WARNING network_errors_delta -> VALIDATE_FABRIC on that interface; no hardware recovery",
        errors_delta,
        risk="L",
        max_minutes=16,
        proof=LIVE_REPLAY,
        approvals=("KERNEL_MODULE_LOAD",),
    ),
    _spec(
        CASE_009,
        "sustained_drops",
        "oversized AF_PACKET frames behind a 576-MTU peer through the ramp plus the "
        "cadence budget timed from the oracle's crossing (~18 min on a 300-s cadence)",
        "drops sustained 60 s -> WARNING -> VALIDATE_FABRIC over two host health "
        "summaries; no hardware recovery",
        sustained_drops,
        risk="L",
        max_minutes=30,
        approvals=("KERNEL_MODULE_LOAD",),
    ),
    _spec(
        CASE_009,
        "short_drops",
        "30 s drop burst",
        "no finding across the sustain plus one health summary after the burst "
        "(555 s on a 300-s cadence)",
        short_drops,
        risk="L",
        max_minutes=16,
        approvals=("KERNEL_MODULE_LOAD",),
    ),
    _spec(
        CASE_009,
        "tcp_only",
        "a bounded TCP flow from the host netns to a gfrs test netns over a gfrs "
        "veth pair, tc netem loss 5 % on the host-side veth egress; tc/sch_netem "
        "installed or loaded for this run only when missing",
        "host RetransSegs rises while the test veth and every real interface "
        "count no errors; the product accepts host batches inside the flow (its "
        "collector status / the veth's latest counter records move, confirmed by "
        "the oracle's RetransSegs rise in that interval) and mints nothing: no "
        "NIC/TCP/RDMA finding, no NIC or fabric threshold edge stored, no fabric "
        "hardware path, no isolation",
        tcp_only,
        risk="L",
        max_minutes=30,
        approvals=("KERNEL_MODULE_LOAD",),
        optional_approvals=(PACKAGE_APPROVAL,),
        cleanup="stop the flow, close the window, delete the qdisc/netns/veth; "
        "unload sch_netem and dnf-remove the packages only if this run "
        "loaded/installed them",
    ),
    _spec(
        CASE_009,
        "counter_reset",
        "a short drop burst, then the pair deleted and re-added in one probe "
        "process (recreate-veth, a millisecond gap: no tick sees the required pair "
        "missing), then a second burst on the recreated pair",
        "the reset tick is a _network collection error naming the interface, the "
        "counter and 'delta unknown' with no network_drops_delta reading in that "
        "batch (never a clamped 0 or a pseudo-rate), delivered as an edge; the "
        "second burst ships a positive delta from the new baseline; no finding, no "
        "hardware recovery, no isolation",
        counter_reset,
        risk="L",
        max_minutes=25,
        approvals=("KERNEL_MODULE_LOAD",),
    ),
    _spec(
        CASE_009,
        "wrong_interface",
        "two required pairs; drops only on B",
        "finding device = B, never A; the generic VALIDATE_FABRIC is node-wide",
        wrong_interface,
        risk="L",
        max_minutes=25,
        approvals=("KERNEL_MODULE_LOAD",),
    ),
    _spec(
        CASE_009,
        "recovery",
        "stop the generator",
        "new zero deltas and a communication check; no NIC reset, no DB cleanup to zero",
        drops_recovery,
        risk="L",
        max_minutes=25,
        approvals=("KERNEL_MODULE_LOAD",),
    ),
]
