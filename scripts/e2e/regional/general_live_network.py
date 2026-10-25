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
``general_live_tcp.py``.
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional.general_live_common import (
    Oracle,
    activity,
    close_window,
    open_window,
    restore_incidents,
    run_survey,
    wait_for_incident,
    wait_settled,
)
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
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
from scripts.e2e.regional.general_live_verdicts import (
    DIAGNOSTIC_ONLY_STATES,
    ChainExpectation,
    chain_errors,
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
DROPS_SUSTAIN = 60
DROPS_BUDGET = DROPS_SUSTAIN + 300 + 90
SHORT_DROPS = 30
NEGATIVE_WINDOW = DROPS_SUSTAIN + 420


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
    open_window(ctx, *pairs, restore_seconds=QUARANTINE_BUDGET + 600)


def required_link_down(ctx: VariantContext) -> dict[str, Any]:
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
        metrics={"network_link_down", "network_link_up"},
        name="link-quarantine",
    )
    reading = ctx.net("reading", "--slot", "a")
    ctx.evidence.write("link-reading", reading)
    link = (reading.get("interfaces") or {}).get(a) or {}
    if link.get("operstate") == "up":
        raise VariantInconclusive(f"the test link {a} did not go down: {link}")
    incidents = finding_incidents(detection, "network_link_down", device=a)
    stages = {
        "chain": chain_errors(
            detection,
            ChainExpectation(
                metric="network_link_down",
                device=a,
                operations=("MARK_UNSCHEDULABLE", "QUARANTINE"),
                forbidden=_hardware() | {"STOP_WORKLOADS"},
                incident_states=frozenset({"QUARANTINED"}),
            ),
        ),
    }
    wait_settled(
        ctx,
        started,
        [item["incident_id"] for item in incidents],
        timeout_seconds=QUARANTINE_BUDGET,
        metrics={"network_link_down"},
        name="link-settled",
    )
    # Recovery order: bring the link up, close the window (unit sees all links
    # up), then validated restore; deleting the interface before that would be
    # a fresh interface_missing event.
    ctx.net("set-link", "--slot", "a", "--state", "up")
    close_window(ctx)
    ctx.cleanup.discard("collector-window")
    restore_incidents(
        ctx, incidents, reason="GF-GENERAL-LIVE-008 required_link_down restore"
    )
    return _result(stages)


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
    incidents = finding_incidents(detection, "network_link_down", device=a)
    stages = {
        "chain": chain_errors(
            detection,
            ChainExpectation(
                metric="network_link_down",
                device=a,
                operations=("MARK_UNSCHEDULABLE", "QUARANTINE"),
                forbidden=_hardware() | {"STOP_WORKLOADS"},
                incident_states=frozenset({"QUARANTINED"}),
            ),
        ),
    }
    close_window(ctx)
    ctx.cleanup.discard("collector-window")
    restore_incidents(
        ctx, incidents, reason="GF-GENERAL-LIVE-008 interface_missing restore"
    )
    return _result(stages)


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
        ctx, started, "network_link_down", NEGATIVE_WINDOW, device=c
    )
    stages = {
        "no_finding": chain_errors(
            observed,
            ChainExpectation(metric="network_link_down", device=c, present=False),
        ),
        "hardware": hardware_errors(observed),
    }
    return _result(stages)


def empty_required_list(ctx: VariantContext) -> dict[str, Any]:
    """Observation: the live required-interface list is empty, so a real NIC
    link-down can never isolate the node (spec FAIL, site configuration)."""

    survey = run_survey(ctx)
    env = survey.get("collector_env") or {}
    configured = str(env.get("GPU_FAULT_REQUIRED_INTERFACES") or "").strip()
    stages = {
        "required_list": [
            "GPU_FAULT_REQUIRED_INTERFACES is empty, so no real NIC link-down can "
            "isolate this node (the installer default is empty)"
        ]
        if not configured
        else []
    }
    return _result(
        stages,
        notes=[
            "remedy: an admin/installer decision to declare the primary/EFA netdevs"
        ],
        extra={"configured_required_interfaces": configured},
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
    incidents = finding_incidents(detection, "network_link_down", device=a)
    ctx.net("set-link", "--slot", "a", "--state", "up")
    close_window(ctx)
    ctx.cleanup.discard("collector-window")
    restored = restore_incidents(
        ctx, incidents, reason="GF-GENERAL-LIVE-008 recovery restore"
    )
    operations = [op for item in restored for op in item.get("operations") or []]
    stages = {
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
    hold = DROPS_BUDGET + 120
    oracle = _drops_oracle(ctx, a, b, hold)
    ctx.net("start-drops", "--slot", "a", "--rate", "10", "--seconds", str(hold))
    ctx.cleanup.push("net-drops", lambda: ctx.net("stop-drops"))
    detection = _wait_drops_finding(ctx, started, (a, b))
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
        timeout_seconds=DROPS_BUDGET,
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
    observed = _watch_no_finding(ctx, started, "network_drops_delta", NEGATIVE_WINDOW)
    stages = {
        "no_finding": chain_errors(
            observed, ChainExpectation(metric="network_drops_delta", present=False)
        )
    }
    close_window(ctx)
    ctx.cleanup.discard("collector-window")
    return _result(stages)


def counter_reset(ctx: VariantContext) -> dict[str, Any]:
    created = _veth(ctx)
    a, b = created["a"], created["b"]
    _declare_required(ctx, a, b)
    started = utc_now()
    ctx.net("start-drops", "--slot", "a", "--rate", "10", "--seconds", "60")
    ctx.cleanup.push("net-drops", lambda: ctx.net("stop-drops"))
    ctx.hold(40, name="pre-reset")
    ctx.net("stop-drops")
    ctx.cleanup.discard("net-drops")
    recreated = ctx.net("delete-interface", "--slot", "a")
    ctx.net(
        "create-veth",
        "--slot",
        "a",
        "--peer-slot",
        "b",
        "--deadman-seconds",
        str(ctx.spec.max_minutes * 60 + 600),
    )
    ctx.hold(ctx.host_interval * 3, name="post-reset")
    observed = activity(ctx, started, metrics={"network_drops_delta"})
    stages = {"hardware": hardware_errors(observed)}
    close_window(ctx)
    ctx.cleanup.discard("collector-window")
    return _result(
        stages,
        notes=[
            "the reset clamps to 0 (X5); during active drops it reads as recovered -- recorded as an observation"
        ],
        extra={"recreated": recreated},
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
    ctx.net(
        "start-drops",
        "--slot",
        "c",
        "--rate",
        "10",
        "--seconds",
        str(DROPS_BUDGET + 60),
    )
    ctx.cleanup.push("net-drops", lambda: ctx.net("stop-drops"))
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
    _drops_oracle(ctx, a, b, DROPS_BUDGET + 60)
    ctx.net(
        "start-drops",
        "--slot",
        "a",
        "--rate",
        "10",
        "--seconds",
        str(DROPS_BUDGET + 60),
    )
    ctx.cleanup.push("net-drops", lambda: ctx.net("stop-drops"))
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
        timeout_seconds=DROPS_BUDGET,
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
    ctx: VariantContext, since: Any, interfaces: tuple[str, ...]
) -> dict[str, Any]:
    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = activity(ctx, since, metrics={"network_drops_delta"})
        for interface in interfaces:
            if finding_incidents(last, "network_drops_delta", device=interface):
                return last
        return None

    value = ctx.wait_until(
        accept,
        timeout_seconds=DROPS_BUDGET,
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
        "read GPU_FAULT_REQUIRED_INTERFACES on the live node",
        "the list is empty, so a real NIC link-down can never isolate (spec FAIL, site config)",
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
        "oversized AF_PACKET frames behind a 576-MTU peer for >= 7 min",
        "drops sustained 60 s -> WARNING -> VALIDATE_FABRIC; no hardware recovery",
        sustained_drops,
        risk="L",
        max_minutes=20,
        approvals=("KERNEL_MODULE_LOAD",),
    ),
    _spec(
        CASE_009,
        "short_drops",
        "30 s drop burst",
        "no finding",
        short_drops,
        risk="L",
        max_minutes=14,
        approvals=("KERNEL_MODULE_LOAD",),
    ),
    _spec(
        CASE_009,
        "tcp_only",
        "a bounded TCP flow from the host netns to a gfrs test netns over a gfrs "
        "veth pair, tc netem loss 5 % on the host-side veth egress; tc/sch_netem "
        "installed or loaded for this run only when missing",
        "host RetransSegs rises while the test veth and every real interface "
        "count no errors; the product collects tcp_retransmits_delta but mints no "
        "finding: no NIC finding, no fabric hardware path, no isolation",
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
        "atomic delete+add of the drop interface between ticks",
        "clamped to 0, no pseudo-rate, no hardware recovery",
        counter_reset,
        risk="L",
        max_minutes=10,
        approvals=("KERNEL_MODULE_LOAD",),
    ),
    _spec(
        CASE_009,
        "wrong_interface",
        "two required pairs; drops only on B",
        "finding device = B, never A; the generic VALIDATE_FABRIC is node-wide",
        wrong_interface,
        risk="L",
        max_minutes=20,
        approvals=("KERNEL_MODULE_LOAD",),
    ),
    _spec(
        CASE_009,
        "recovery",
        "stop the generator",
        "new zero deltas and a communication check; no NIC reset, no DB cleanup to zero",
        drops_recovery,
        risk="L",
        max_minutes=20,
        approvals=("KERNEL_MODULE_LOAD",),
    ),
]
