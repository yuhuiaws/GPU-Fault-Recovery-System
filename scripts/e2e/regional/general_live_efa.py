"""GF-GENERAL-LIVE-010 EFA/RDMA error counters, replayed at the host collector's readers.

The host collector grades EFA/RDMA errors from two raw readers
(``collectors/host/network.py``): the sysfs port files of every EFA RDMA device
(``/sys/class/infiniband/<dev>/ports/<n>/{counters,hw_counters}/<c>`` summed
into ``rdma_errors_delta`` per port, ``state``/``phys_state`` into
``rdma_link_down``) and ``ethtool -S`` on the netdevs an EFA function owns
(counter-name tokens ``rnr``/``retry``/``cq_err`` into ``efa_rnr_errors_delta``,
``efa_retry_errors_delta``, ``efa_cq_errors_delta``; a failed command is UNKNOWN,
a counter the netdev lacks is ``efa_ethtool_counters_unsupported``). This site
has no vendor lab and never disconnects a real EFA link or writes a real sysfs
counter, so every variant is ``live-replay`` on the shared host source replay
(``general_live_source_replay.py``, ``HOST_SOURCE_REPLAY``): exactly one reviewed
reading is replaced inside ``gpu-fault-host-collector.service`` -- a counter or
state file bound read-only into its mount namespace, or the reviewed ``ethtool``
pass-through wrapper first on its PATH -- and parsing, grading, incidents,
workflows and VALIDATE_FABRIC stay real.

Every variant starts from the node's real inventory (``inventory`` probe
command): the real driver identity, the real counter names of the chosen port,
the owning netdev and what the real ``ethtool -S`` prints are recorded as
evidence, and the fixture is built on them -- a bind targets a counter file the
port really has, an ``ethtool`` rule prefers a real counter carrying the token.
Verdicts read only the product's records (``general_live_verdicts`` matchers on
the product's metric names, pinned against ``host_health.py`` and
``collectors/host/network.py`` in the tests); the oracle is the fixture's own
value, the open-window verification and the wrapper's audit journal. Cleanup
closes the window and ``read-source`` proves the unit reads the real sysfs value
again. A reusable :class:`EfaSourceFixture` carries the same steps for other
categories (011 ``broken_pair`` breaks the traffic pair with it).

``counter_reset`` is an expected FAIL: ``_rdma`` clamps a counter that fell
below its baseline to a 0 delta and ships ``rdma_errors_delta=0`` as a reading,
which clears the active signal and re-mints a continuing fault as a new
incident, while ``_ethtool_counters`` marks the same case UNKNOWN. ``non_efa``
stays CAPABILITY_ABSENT: the site has no IB-only RDMA device.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any

from scripts.e2e.regional.general_live_common import (
    STORE_POLL_SECONDS,
    activity,
    await_metric_recovery,
    run_survey,
    wait_for_incident,
    wait_settled,
)
from scripts.e2e.regional.general_live_efa_fixture import (
    CASE,
    DETECTION_BUDGET,
    ERRORS_METRIC,
    ETHTOOL_GROUPS,
    FAILED_STDERR,
    INVENTORY_METRIC,
    LINK_DOWN_TEXT,
    LINK_METRIC,
    PROOF_SCOPE,
    RAMP_STEP,
    REPLAY_COUNTERS,
    STEP,
    UNSUPPORTED_METRIC,
    VALIDATION_BUDGET,
    EfaSourceFixture,
    EfaTarget,
    collection_errors,
    counter_target,
    metric_values,
    negative_readings,
    netdev_target,
    real_counter,
    served_errors,
    state_target,
    unsupported_markers,
    validation_errors,
    zero_readings,
)
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantNotRun,
    utc_now,
)
from scripts.e2e.regional.general_live_registry import (
    CAPABILITY,
    LIVE_REPLAY,
    VariantSpec,
    deferred_variant,
)
from scripts.e2e.regional.general_live_source_replay import (
    HOST_SOURCE_REPLAY,
    rule,
)
from scripts.e2e.regional.general_live_verdicts import (
    ChainExpectation,
    DIAGNOSTIC_ONLY_STATES,
    chain_errors,
    final_execution,
    finding_incidents,
    hardware_errors,
    isolation_operations,
    node_isolated,
    samples,
    settled,
    verdict,
    workflows_of,
)


def _wait(
    ctx: VariantContext,
    since: datetime,
    accept: Callable[[dict[str, Any]], bool],
    *,
    metrics: Iterable[str],
    timeout_seconds: float,
    name: str,
) -> dict[str, Any]:
    last: dict[str, Any] = {}
    wanted = set(metrics)

    def poll() -> dict[str, Any] | None:
        nonlocal last
        last = activity(ctx, since, metrics=wanted)
        return last if accept(last) else None

    value = ctx.wait_until(
        poll,
        timeout_seconds=timeout_seconds,
        poll_seconds=STORE_POLL_SECONDS,
        name=name,
    )
    return value if value is not None else last


def _await_validation(
    ctx: VariantContext,
    since: datetime,
    incident: Mapping[str, Any],
    *,
    metrics: Iterable[str],
    name: str,
) -> dict[str, Any]:
    """The records once the incident's VALIDATE_FABRIC has a terminal execution."""

    incident_id = str(incident.get("incident_id"))

    def accept(document: dict[str, Any]) -> bool:
        workflows = workflows_of(document, incident_id)
        if not workflows:
            return False
        execution = final_execution(workflows[0], "VALIDATE_FABRIC")
        return (
            execution is not None and execution.get("status") in {"SUCCEEDED", "FAILED"}
        ) or settled(document, [incident_id])

    return _wait(
        ctx,
        since,
        accept,
        metrics=metrics,
        timeout_seconds=VALIDATION_BUDGET,
        name=name,
    )


def _settle(
    ctx: VariantContext,
    since: datetime,
    incidents: Sequence[Mapping[str, Any]],
    *,
    metrics: Iterable[str],
    name: str,
) -> list[str]:
    """Every incident this variant minted reached a terminal workflow state."""

    ids = [str(item["incident_id"]) for item in incidents]
    if not ids:
        return []
    final = wait_settled(
        ctx,
        since,
        ids,
        timeout_seconds=ctx.host_summary + 300,
        metrics=metrics,
        name=name,
    )
    return [] if settled(final, ids) else [f"incidents did not settle: {ids}"]


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
        "proof_scope": PROOF_SCOPE,
        **(extra or {}),
    }


def _not_isolated(ctx: VariantContext, document: Mapping[str, Any]) -> list[str]:
    errors = [
        f"isolation planned: {item}" for item in isolation_operations(dict(document))
    ]
    if node_isolated(ctx.regional.node_snapshot(ctx.settings.node)):
        errors.append("node isolated on an EFA counter reading")
    return errors


def _baseline_hold(ctx: VariantContext, name: str) -> None:
    """One poll establishes the collector's delta baseline after the restart."""

    ctx.hold(ctx.host_interval * 2 + 10, name=name)


# --------------------------------------------------------------------------- #
# sysfs variants
# --------------------------------------------------------------------------- #
def rdma_error(ctx: VariantContext) -> dict[str, Any]:
    """A bound port counter stepped up -> rdma_errors_delta WARNING -> VALIDATE_FABRIC."""

    run_survey(ctx)
    fixture = EfaSourceFixture(ctx)
    target = counter_target(fixture.survey())
    ctx.evidence.write("efa-target", target.describe())
    fixture.bind("efa-counter", target.counter_dest, str(target.real_value))
    fixture.open()
    _baseline_hold(ctx, "counter-baseline")
    started = utc_now()
    stepped = int(str(target.real_value)) + STEP
    fixture.step("efa-counter", stepped)
    metrics = {ERRORS_METRIC, target.counter_metric}
    detection = wait_for_incident(
        ctx,
        started,
        ERRORS_METRIC,
        device=target.port_device,
        timeout_seconds=DETECTION_BUDGET,
        metrics=metrics,
        name="rdma-error",
    )
    incidents = finding_incidents(detection, ERRORS_METRIC, device=target.port_device)
    settled_doc = (
        _await_validation(ctx, started, incidents[0], metrics=metrics, name="fabric")
        if incidents
        else detection
    )
    grew = metric_values(settled_doc, target.counter_metric, target.port_device)
    stages = {
        "chain": chain_errors(
            settled_doc,
            ChainExpectation(
                metric=ERRORS_METRIC,
                device=target.port_device,
                operations=("FREEZE_EVIDENCE", "VALIDATE_FABRIC"),
                incident_states=DIAGNOSTIC_ONLY_STATES,
            ),
        ),
        "counter_grew": []
        if any(value >= STEP for value in grew)
        else [
            f"no {target.counter_metric} >= {STEP} on {target.port_device}: the "
            "product's reading did not grow with the fixture"
        ],
        "fabric_validation": validation_errors(settled_doc, incidents[0], ERRORS_METRIC)
        if incidents
        else [],
        "fresh_observation": []
        if metric_values(settled_doc, ERRORS_METRIC, target.port_device)
        else [f"no {ERRORS_METRIC} sample for {target.port_device} after the step"],
        "hardware": hardware_errors(settled_doc),
        "not_isolated": _not_isolated(ctx, settled_doc),
        "settled": _settle(
            ctx, started, incidents, metrics=metrics, name="rdma-settled"
        ),
    }
    fixture.close()
    stages["real_restored"] = fixture.restored_errors()
    return _result(
        stages,
        notes=[
            f"bound {target.counter_dest} ({target.real_value} -> {stepped}); the delta "
            "is a WARNING fabric diagnostic, never a hardware action"
        ],
        extra={"target": target.describe(), "fixture_value": stepped},
    )


def link_inactive(ctx: VariantContext) -> dict[str, Any]:
    """A bound port state != ACTIVE -> rdma_link_down=1 -> fabric diagnostic, no reboot."""

    run_survey(ctx)
    fixture = EfaSourceFixture(ctx)
    target = state_target(fixture.survey())
    ctx.evidence.write("efa-target", target.describe())
    fixture.bind("efa-state", target.dest("state"), LINK_DOWN_TEXT)
    started = utc_now()
    fixture.open()
    metrics = {LINK_METRIC, INVENTORY_METRIC}
    detection = wait_for_incident(
        ctx,
        started,
        LINK_METRIC,
        device=target.port_device,
        timeout_seconds=DETECTION_BUDGET,
        metrics=metrics,
        name="link-inactive",
    )
    incidents = finding_incidents(detection, LINK_METRIC, device=target.port_device)
    held = (
        _await_validation(ctx, started, incidents[0], metrics=metrics, name="fabric")
        if incidents
        else detection
    )
    inventory_samples = [
        item for item in samples(held, INVENTORY_METRIC) if item.get("value") == 1
    ]
    stages = {
        "chain": chain_errors(
            held,
            ChainExpectation(
                metric=LINK_METRIC,
                device=target.port_device,
                operations=("FREEZE_EVIDENCE", "VALIDATE_FABRIC"),
                validation="VALIDATE_FABRIC",
                validation_must_fail=True,
                incident_states=DIAGNOSTIC_ONLY_STATES,
            ),
        ),
        "link_sample": []
        if any(
            value == 1 for value in metric_values(held, LINK_METRIC, target.port_device)
        )
        else [f"no {LINK_METRIC}=1 sample for {target.port_device}"],
        "inventory_grading": [
            f"efa_inventory_mismatch graded {item.get('labels', {}).get('failure_mode')} "
            "instead of LINK_INACTIVE"
            for item in inventory_samples
            if (item.get("labels") or {}).get("failure_mode") != "LINK_INACTIVE"
        ],
        "hardware": hardware_errors(held),
        "not_isolated": _not_isolated(ctx, held),
    }
    fixture.close()
    closed_at = utc_now()
    stages["real_restored"] = fixture.restored_errors()
    recovered = await_metric_recovery(
        ctx,
        LINK_METRIC,
        1.0,
        device=target.port_device,
        name="link-recovery",
        since=closed_at,
    )
    stages["recovery"] = recovered["errors"]
    stages["settled"] = _settle(
        ctx,
        started,
        [*incidents, *finding_incidents(held, INVENTORY_METRIC)],
        metrics=metrics,
        name="link-settled",
    )
    return _result(
        stages,
        notes=[
            f"bound {target.dest('state')} = {LINK_DOWN_TEXT!r} (phys_state real); "
            "VALIDATE_FABRIC must fail on the held port and nothing may reboot or cordon",
            f"inventory co-finding samples: {len(inventory_samples)}",
        ],
        extra={"target": target.describe(), "recovery": recovered},
    )


# --------------------------------------------------------------------------- #
# ethtool variants
# --------------------------------------------------------------------------- #
def _ethtool_target(
    fixture: EfaSourceFixture, token: str
) -> tuple[EfaTarget, str, str]:
    """``(target, counter name, baseline value)`` for one ethtool group."""

    target = netdev_target(fixture.inventory)
    counter = real_counter(token, target.ethtool_names)
    return target, counter or REPLAY_COUNTERS[token], "0"


def _prepare_ethtool(
    fixture: EfaSourceFixture, target: EfaTarget, rules: Sequence[Mapping[str, Any]]
) -> None:
    if target.netdev_source == "fixture":
        fixture.attach_netdev(target)
    fixture.install_ethtool(rules)


def _stats_rule(interface: str, **fields: Any) -> dict[str, Any]:
    return rule("stats", interface=interface, **fields)


def _ethtool_delta(
    ctx: VariantContext, token: str, *, congestion: bool
) -> dict[str, Any]:
    """A counter of ``token``'s group stepped up -> its delta WARNING -> diagnostic only."""

    run_survey(ctx)
    fixture = EfaSourceFixture(ctx)
    fixture.survey()
    target, counter, baseline = _ethtool_target(fixture, token)
    metric = ETHTOOL_GROUPS[token]
    interface = str(target.netdev)
    ctx.evidence.write(
        "efa-target", {**target.describe(), "counter": counter, "metric": metric}
    )
    _prepare_ethtool(
        fixture,
        target,
        [_stats_rule(interface, action="set_counter", counter=counter, value=baseline)],
    )
    fixture.open()
    _baseline_hold(ctx, "ethtool-baseline")
    started = utc_now()
    fixture.replan_ethtool(
        [_stats_rule(interface, action="set_counter", counter=counter, value=str(STEP))]
    )
    metrics = {metric, UNSUPPORTED_METRIC}
    detection = wait_for_incident(
        ctx,
        started,
        metric,
        device=interface,
        timeout_seconds=DETECTION_BUDGET,
        metrics=metrics,
        name=f"{token}-delta",
    )
    incidents = finding_incidents(detection, metric, device=interface)
    held = (
        _await_validation(ctx, started, incidents[0], metrics=metrics, name="fabric")
        if incidents
        else detection
    )
    journal = fixture.audit()
    stages = {
        "chain": chain_errors(
            held,
            ChainExpectation(
                metric=metric,
                device=interface,
                operations=("FREEZE_EVIDENCE", "VALIDATE_FABRIC"),
                incident_states=DIAGNOSTIC_ONLY_STATES,
            ),
        ),
        "delta_reported": []
        if any(value >= STEP for value in metric_values(held, metric, interface))
        else [f"no {metric} >= {STEP} on {interface} in the product's records"],
        "fabric_validation": validation_errors(held, incidents[0], metric)
        if incidents
        else [],
        "served": served_errors(journal, interface),
        "hardware": hardware_errors(held),
        "not_isolated": _not_isolated(ctx, held),
        "settled": _settle(
            ctx, started, incidents, metrics=metrics, name=f"{token}-settled"
        ),
    }
    fixture.close()
    notes = [
        f"ethtool -S {interface}: {counter} 0 -> {STEP} through the reviewed wrapper "
        f"(netdev {target.netdev_source}); real counters listed: {len(target.ethtool_names)}"
    ]
    if congestion:
        notes.append(
            "RNR/retry may be congestion or resource pressure: a diagnostic only, "
            "no bad-card inference, no hardware action"
        )
    return _result(
        stages,
        notes=notes,
        extra={"target": target.describe(), "counter": counter, "metric": metric},
    )


def rnr(ctx: VariantContext) -> dict[str, Any]:
    return _ethtool_delta(ctx, "rnr", congestion=True)


def retry(ctx: VariantContext) -> dict[str, Any]:
    return _ethtool_delta(ctx, "retry", congestion=True)


def cq_error(ctx: VariantContext) -> dict[str, Any]:
    return _ethtool_delta(ctx, "cq_err", congestion=False)


def ethtool_failed(ctx: VariantContext) -> dict[str, Any]:
    """``ethtool -S`` exits 1 -> a collection error (UNKNOWN), no zero deltas, no finding."""

    run_survey(ctx)
    fixture = EfaSourceFixture(ctx)
    fixture.survey()
    target = netdev_target(fixture.inventory)
    interface = str(target.netdev)
    ctx.evidence.write("efa-target", target.describe())
    _prepare_ethtool(
        fixture,
        target,
        [_stats_rule(interface, action="exit", exit_code=1, stderr=FAILED_STDERR)],
    )
    started = utc_now()
    fixture.open()
    metrics = {*ETHTOOL_GROUPS.values(), UNSUPPORTED_METRIC}
    observed = _wait(
        ctx,
        started,
        lambda document: bool(collection_errors(document, "_efa_network", interface)),
        metrics=metrics,
        timeout_seconds=ctx.host_summary + ctx.host_interval * 4,
        name="ethtool-failed",
    )
    journal = fixture.audit()
    zeros = [
        f"{metric} shipped 0 for {interface} although ethtool failed"
        for metric in ETHTOOL_GROUPS.values()
        if metric_values(observed, metric, interface)
    ]
    stages = {
        "unknown_recorded": []
        if collection_errors(observed, "_efa_network", interface)
        else [
            "no collection error names _efa_network and the interface: the failure was silent"
        ],
        "no_zero_reading": zeros,
        "not_no_statistics": [
            "an I/O error was graded NO_STATISTICS (permanent gap) instead of UNKNOWN"
        ]
        if any(
            (item.get("labels") or {}).get("reason") == "NO_STATISTICS"
            for item in samples(observed, UNSUPPORTED_METRIC, device=interface)
        )
        else [],
        "no_finding": [
            f"a failed ethtool minted a {metric} incident {item['incident_id']}"
            for metric in ETHTOOL_GROUPS.values()
            for item in finding_incidents(observed, metric, device=interface)
        ],
        "served": served_errors(journal, interface, returncode=1),
        "hardware": hardware_errors(observed),
        "not_isolated": _not_isolated(ctx, observed),
    }
    fixture.close()
    return _result(
        stages,
        notes=[
            f"ethtool -S {interface} exit 1 ({FAILED_STDERR!r}) served by the wrapper; "
            "UNKNOWN is a collection error, never zero congestion counters"
        ],
        extra={"target": target.describe()},
    )


def missing_counter(ctx: VariantContext) -> dict[str, Any]:
    """A counter the netdev stops exposing -> MISSING_COUNTER marker, never a zero delta."""

    run_survey(ctx)
    fixture = EfaSourceFixture(ctx)
    fixture.survey()
    target = netdev_target(fixture.inventory)
    interface = str(target.netdev)
    token = next(
        (item for item in ETHTOOL_GROUPS if real_counter(item, target.ethtool_names)),
        "rnr",
    )
    metric = ETHTOOL_GROUPS[token]
    real = real_counter(token, target.ethtool_names)
    counter = real or REPLAY_COUNTERS[token]
    ctx.evidence.write(
        "efa-target",
        {**target.describe(), "counter": counter, "metric": metric, "real": bool(real)},
    )
    _prepare_ethtool(
        fixture,
        target,
        [_stats_rule(interface, action="set_counter", counter=counter, value="0")],
    )
    opened_at = utc_now()
    fixture.open()
    metrics = {metric, UNSUPPORTED_METRIC}
    present = _wait(
        ctx,
        opened_at,
        lambda document: bool(document.get("batches")),
        metrics=metrics,
        timeout_seconds=ctx.host_interval * 4 + 30,
        name="counter-present",
    )
    started = utc_now()
    fixture.replan_ethtool(
        [
            _stats_rule(interface, action="delete_counter", counter=counter)
            if real
            else _stats_rule(interface, action="passthrough")
        ]
    )
    observed = _wait(
        ctx,
        started,
        lambda document: bool(
            unsupported_markers(document, interface, metric, "MISSING_COUNTER")
        ),
        metrics=metrics,
        timeout_seconds=ctx.host_summary + ctx.host_interval * 4 + 30,
        name="counter-missing",
    )
    journal = fixture.audit()
    stages = {
        "was_present": []
        if present.get("batches")
        and not unsupported_markers(present, interface, metric, "MISSING_COUNTER")
        else [
            f"{metric} was already MISSING_COUNTER (or no batch) while the counter was served"
        ],
        "not_supported_marker": []
        if unsupported_markers(observed, interface, metric, "MISSING_COUNTER")
        else [
            f"no {UNSUPPORTED_METRIC} MISSING_COUNTER marker names {metric} on {interface}"
        ],
        "no_zero_reading": [
            f"{metric} shipped {value} for {interface} after the counter disappeared"
            for value in metric_values(observed, metric, interface)
        ],
        "no_finding": [
            f"a missing counter minted a {metric} incident {item['incident_id']}"
            for item in finding_incidents(observed, metric, device=interface)
        ],
        "served": served_errors(journal, interface),
        "hardware": hardware_errors(observed),
        "not_isolated": _not_isolated(ctx, observed),
    }
    fixture.close()
    return _result(
        stages,
        notes=[
            f"{counter} ({'real' if real else 'test-origin'}) served then withdrawn on "
            f"ethtool -S {interface}; NOT_SUPPORTED is named, never zeroed"
        ],
        extra={"target": target.describe(), "counter": counter, "metric": metric},
    )


# --------------------------------------------------------------------------- #
# counter reset (both readers)
# --------------------------------------------------------------------------- #
def _ramp(
    ctx: VariantContext,
    fixture: EfaSourceFixture,
    *,
    sysfs: str,
    ethtool: tuple[str, str] | None,
    values: dict[str, int],
    since: datetime,
    metrics: Iterable[str],
    accept: Callable[[dict[str, Any]], bool],
    timeout_seconds: float,
    name: str,
) -> dict[str, Any]:
    """Step both readers every half interval while polling the store.

    Each collector tick must see at least one step, or a 0 delta would clear
    the signal for a reason that is not the reset under test.
    """

    start = ctx.clock()
    last: dict[str, Any] = {}
    last_poll: datetime | None = None
    step_seconds = max(3, ctx.host_interval // 2)
    wanted = set(metrics)
    while True:
        values["sysfs"] += RAMP_STEP
        fixture.step(sysfs, values["sysfs"])
        if ethtool is not None:
            values["ethtool"] += RAMP_STEP
            interface, counter = ethtool
            fixture.replan_ethtool(
                [
                    _stats_rule(
                        interface,
                        action="set_counter",
                        counter=counter,
                        value=str(values["ethtool"]),
                    )
                ]
            )
        now = ctx.clock()
        if last_poll is None or (now - last_poll).total_seconds() >= STORE_POLL_SECONDS:
            last = activity(ctx, since, metrics=wanted)
            last_poll = now
            ctx.evidence.write(
                f"{name}-ramp", {"values": dict(values), "at": now.isoformat()}
            )
            if accept(last):
                return last
        if (now - start).total_seconds() >= timeout_seconds:
            return last
        ctx.sleep(step_seconds)


def counter_reset(ctx: VariantContext) -> dict[str, Any]:
    """A ramping counter reset below its baseline: not a decrement, not a new fault.

    Both readers ramp until their deltas mint one incident each, then fall to 1
    and ramp on. The ethtool reader marks the reset UNKNOWN; the sysfs reader
    clamps it to a 0 reading (the product gap this variant is expected to FAIL on).
    """

    run_survey(ctx)
    fixture = EfaSourceFixture(ctx)
    inventory = fixture.survey()
    target = counter_target(inventory)
    ethtool: tuple[str, str] | None = None
    netdev: EfaTarget | None = None
    try:
        netdev = netdev_target(inventory)
    except VariantNotRun as skipped:
        ctx.evidence.write("ethtool-skipped", {"reason": str(skipped)})
    if netdev is not None:
        interface = str(netdev.netdev)
        counter = real_counter("rnr", netdev.ethtool_names) or REPLAY_COUNTERS["rnr"]
        ethtool = (interface, counter)
        _prepare_ethtool(
            fixture,
            netdev,
            [_stats_rule(interface, action="set_counter", counter=counter, value="0")],
        )
    ctx.evidence.write(
        "efa-target",
        {
            **target.describe(),
            "ethtool": {"interface": ethtool[0], "counter": ethtool[1]}
            if ethtool
            else None,
        },
    )
    fixture.bind("efa-counter", target.counter_dest, str(target.real_value))
    fixture.open()
    _baseline_hold(ctx, "reset-baseline")
    values = {"sysfs": int(str(target.real_value)), "ethtool": 0}
    metric = ETHTOOL_GROUPS["rnr"]
    metrics = {ERRORS_METRIC, target.counter_metric, metric, UNSUPPORTED_METRIC}
    started = utc_now()

    def minted(document: dict[str, Any]) -> bool:
        if not finding_incidents(document, ERRORS_METRIC, device=target.port_device):
            return False
        return ethtool is None or bool(
            finding_incidents(document, metric, device=ethtool[0])
        )

    before = _ramp(
        ctx,
        fixture,
        sysfs="efa-counter",
        ethtool=ethtool,
        values=values,
        since=started,
        metrics=metrics,
        accept=minted,
        timeout_seconds=DETECTION_BUDGET,
        name="reset-before",
    )
    reset_at = utc_now()
    values["sysfs"], values["ethtool"] = 1, 1
    fixture.step("efa-counter", 1)
    if ethtool is not None:
        fixture.replan_ethtool(
            [
                _stats_rule(
                    ethtool[0], action="set_counter", counter=ethtool[1], value="1"
                )
            ]
        )
    _ramp(
        ctx,
        fixture,
        sysfs="efa-counter",
        ethtool=ethtool,
        values=values,
        since=started,
        metrics=metrics,
        accept=lambda _document: False,
        timeout_seconds=ctx.host_interval * 4 + 20,
        name="reset-after",
    )
    ctx.hold(ctx.host_interval + 5, name="reset-settle")
    after = activity(ctx, started, metrics=metrics)
    sysfs_incidents = finding_incidents(after, ERRORS_METRIC, device=target.port_device)
    stages: dict[str, list[str]] = {
        "fault_seen": []
        if finding_incidents(before, ERRORS_METRIC, device=target.port_device)
        else [
            f"no {ERRORS_METRIC} finding on {target.port_device} while the counter ramped"
        ],
        "no_decrement": negative_readings(
            after, (ERRORS_METRIC, target.counter_metric), target.port_device
        ),
        "sysfs_single_incident": []
        if len(sysfs_incidents) <= 1
        else [
            f"the reset re-minted the fault: {len(sysfs_incidents)} {ERRORS_METRIC} "
            f"incidents on {target.port_device}"
        ],
        "sysfs_reset_unknown": (
            []
            if collection_errors(after, target.port_device, "reset")
            else [
                f"no collection error names the {target.port_device} counter reset: "
                "the sysfs reader did not mark the reset UNKNOWN"
            ]
        )
        + [
            f"{ERRORS_METRIC}=0 shipped for {target.port_device} at the reset "
            f"(observed {item['observed_at']}): a clamped reset read as a clean port"
            for item in zero_readings(
                after, ERRORS_METRIC, target.port_device, after=reset_at
            )
        ],
        "hardware": hardware_errors(after),
        "not_isolated": _not_isolated(ctx, after),
    }
    if ethtool is not None:
        interface = ethtool[0]
        ethtool_incidents = finding_incidents(after, metric, device=interface)
        stages["ethtool_fault_seen"] = (
            []
            if finding_incidents(before, metric, device=interface)
            else [f"no {metric} finding on {interface}"]
        )
        stages["ethtool_single_incident"] = (
            []
            if len(ethtool_incidents) <= 1
            else [
                f"the reset re-minted the fault: {len(ethtool_incidents)} {metric} incidents"
            ]
        )
        stages["ethtool_reset_unknown"] = (
            []
            if collection_errors(after, "_efa_network", interface, "reset")
            else [f"no collection error names the {interface} ethtool counter reset"]
        ) + [
            f"{metric}=0 shipped for {interface} at the reset (observed {item['observed_at']})"
            for item in zero_readings(after, metric, interface, after=reset_at)
        ]
        stages["no_decrement"] += negative_readings(after, (metric,), interface)
    minted_incidents = list(sysfs_incidents)
    if ethtool is not None:
        minted_incidents += finding_incidents(after, metric, device=ethtool[0])
    stages["settled"] = _settle(
        ctx, started, minted_incidents, metrics=metrics, name="reset-settled"
    )
    fixture.close()
    stages["real_restored"] = fixture.restored_errors()
    return _result(
        stages,
        notes=[
            f"{target.counter_dest} ramped by {RAMP_STEP} per half interval, reset to 1 at "
            f"{reset_at.isoformat()}, ramped on; a reset is not a decrement, not a new "
            "fault and not a recovery",
            "ethtool half: "
            + (f"{ethtool[1]} on {ethtool[0]}" if ethtool else "skipped"),
        ],
        extra={
            "target": target.describe(),
            "reset_at": reset_at.isoformat(),
            "values": values,
        },
    )


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def _spec(
    variant: str,
    injection: str,
    expected: str,
    execute: Callable[[VariantContext], dict[str, Any]],
    *,
    risk: str = "L",
    max_minutes: int = 20,
    outcome: str = "PASS",
    gap: str = "",
    cleanup: str = (
        "close the source window (drop-in removed, unit restarted, deadman "
        "disarmed) and read the real sysfs value back inside the unit"
    ),
) -> VariantSpec:
    return VariantSpec(
        case_id=CASE,
        variant=variant,
        proof_mode=LIVE_REPLAY,
        lanes=("eks", "hp"),
        risk=risk,
        flags="S",
        injection=injection,
        expected=expected,
        cleanup=cleanup,
        max_minutes=max_minutes,
        approvals=(HOST_SOURCE_REPLAY,),
        expected_outcome=outcome,
        product_gap=gap,
        execute=execute,
    )


RESET_GAP = (
    "src/gpu_fault/collectors/host/network.py HostNetworkMixin._rdma: a port counter "
    "that falls below its baseline (a reset) is clamped by _delta to a 0 delta and "
    "shipped as rdma_errors_delta=0 -- a reading that clears the active RDMA error "
    "signal, that VALIDATE_FABRIC accepts as a clean port, and after which the next "
    "real increase re-mints the same fault as a new incident; _ethtool_counters marks "
    "the same case UNKNOWN (a collection error, no delta) and the sysfs reader must too"
)

VARIANTS = [
    _spec(
        "rdma_error",
        "bind one real EFA port counter file (hw_counters/rx_drops by preference) into the "
        f"host collector unit at its real value, then step it by {STEP}",
        f"{ERRORS_METRIC} and rdma_<counter>_delta grow by the step on the port -> WARNING "
        "RUN_DIAGNOSTICS -> FREEZE_EVIDENCE + VALIDATE_FABRIC executed on the port; the port "
        "keeps being observed; no hardware action, no isolation",
        rdma_error,
    ),
    _spec(
        "link_inactive",
        f"bind the port's state file to {LINK_DOWN_TEXT!r} inside the unit (phys_state real)",
        f"{LINK_METRIC}=1 on the port -> WARNING -> VALIDATE_FABRIC fails on the held port; "
        "any efa_inventory_mismatch co-finding is LINK_INACTIVE (diagnose only); no reboot "
        "or cordon; the link reads up again after the window closes",
        link_inactive,
        risk="M",
        max_minutes=35,
    ),
    _spec(
        "rnr",
        "the reviewed ethtool wrapper serves a counter carrying the rnr token (a real one when "
        f"the netdev prints one, else gfrs_rnr_err) at 0, then at {STEP}",
        "efa_rnr_errors_delta -> WARNING RUN_DIAGNOSTICS -> FREEZE_EVIDENCE + VALIDATE_FABRIC; "
        "diagnosed as congestion/resource pressure, no bad-card inference, no hardware action",
        rnr,
    ),
    _spec(
        "retry",
        "the ethtool wrapper serves a retry-token counter at 0, then at a step",
        "efa_retry_errors_delta -> WARNING -> diagnostic only, no hardware escalation",
        retry,
    ),
    _spec(
        "cq_error",
        "the ethtool wrapper serves a cq_err-token counter at 0, then at a step",
        "efa_cq_errors_delta -> WARNING -> FREEZE_EVIDENCE + VALIDATE_FABRIC; no hardware action",
        cq_error,
    ),
    _spec(
        "missing_counter",
        "the ethtool wrapper serves a group counter, then withdraws it (delete_counter of a "
        "real one, or pass-through once the test-origin line is gone)",
        f"the group is {UNSUPPORTED_METRIC} MISSING_COUNTER (NOT_SUPPORTED) and never a zero "
        "delta; no finding",
        missing_counter,
        max_minutes=25,
    ),
    _spec(
        "ethtool_failed",
        f"the ethtool wrapper answers `-S <efa netdev>` with exit 1 ({FAILED_STDERR!r})",
        "a collection error (UNKNOWN) names _efa_network and the netdev; no efa_*_delta zero, "
        "no NO_STATISTICS misgrading, no finding; the wrapper journal shows the failure served",
        ethtool_failed,
    ),
    _spec(
        "counter_reset",
        "ramp the bound port counter and the ethtool group counter until each mints its "
        "WARNING, reset both to 1 and ramp on",
        "no negative delta, one incident per reader (a reset is not a new fault), the reset "
        "tick UNKNOWN (a collection error, no 0 reading that clears the signal); no hardware",
        counter_reset,
        risk="M",
        max_minutes=30,
        outcome="FAIL",
        gap=RESET_GAP,
    ),
    deferred_variant(
        CASE,
        "non_efa",
        proof_mode=CAPABILITY,
        reason=(
            "CAPABILITY_ABSENT: this site has only EFA RDMA devices; there is no "
            "IB-only device whose counters could be mistaken for EFA readings"
        ),
        lanes=("eks", "hp"),
        expected="an IB-only counter is not treated as a valid EFA reading",
    ),
]

__all__ = [
    "CASE",
    "RESET_GAP",
    "VARIANTS",
    "cq_error",
    "counter_reset",
    "ethtool_failed",
    "link_inactive",
    "missing_counter",
    "rdma_error",
    "retry",
    "rnr",
]
