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

``counter_reset`` judges the one reset rule the three readers share since
batch 173 (``HostNetworkMixin._monotonic_delta`` / ``_reset_problem``): a
counter below its own baseline is missing evidence for that tick -- a
collection error naming the reader, the port or netdev and the counter
(``<counter> reset (<before> -> <value>); delta unknown``), no delta for the
counter nor for the aggregate it feeds, the baseline re-armed from the new
value -- never the clamped ``rdma_errors_delta=0`` that cleared the signal and
let the continuing increase mint the same fault again. ``non_efa`` stays
CAPABILITY_ABSENT: the site has no IB-only RDMA device.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timedelta
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
    CLOCK_TOLERANCE_SECONDS,
    DETECTION_BUDGET,
    ERRORS_METRIC,
    ETHTOOL_GROUPS,
    FAILED_STDERR,
    INVENTORY_METRIC,
    LINK_DOWN_TEXT,
    LINK_METRIC,
    MISSING_METRIC,
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
    device_readings,
    episodes,
    metric_values,
    negative_readings,
    netdev_target,
    readings_summary,
    real_counter,
    reset_tick_readings,
    served_errors,
    state_target,
    unsupported_markers,
    validation_errors,
    zero_readings,
)
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
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
    confirmation_ticks,
    exposure_fallback_seconds,
    exposure_reading_gaps,
    observed_exposure,
    rule,
)
from scripts.e2e.regional.general_live_verdicts import (
    ChainExpectation,
    DIAGNOSTIC_ONLY_STATES,
    chain_errors,
    executions,
    final_execution,
    finding_incidents,
    hardware_errors,
    isolation_operations,
    node_isolated,
    parse_time,
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


def _late_readings(
    document: Mapping[str, Any], metric: str, device: str | None, after: datetime
) -> list[dict[str, Any]]:
    """``metric`` readings >= 1 the product recorded after ``after`` (plus tolerance)."""

    cutoff = after + timedelta(seconds=CLOCK_TOLERANCE_SECONDS)
    return [
        item
        for item in samples(dict(document), metric, device=device)
        if isinstance(item.get("value"), (int, float))
        and item["value"] >= 1
        and (moment := parse_time(item.get("observed_at"))) is not None
        and moment > cutoff
    ]


def _gaps(
    exposure: Mapping[str, Any],
    document: Mapping[str, Any],
    metric: str,
    ticks: int,
    *,
    device: str | None = None,
) -> str:
    """The INCONCLUSIVE's naming of the reading a bounded exposure missed: the
    tick that was no collector read (the restarted unit's READY=1 stamp, live
    eks 2026-10-26), the batch without the reading, the reading after the close
    (``general_live_source_replay.exposure_reading_gaps``)."""

    lines = exposure_reading_gaps(
        exposure,
        document.get("batches") or [],
        metric=metric,
        required=ticks,
        tolerance_seconds=CLOCK_TOLERANCE_SECONDS,
        device=device,
    )
    return "".join(f"; {line}" for line in lines)


def link_inactive(ctx: VariantContext) -> dict[str, Any]:
    """A bound port state != ACTIVE for a bounded exposure -> ``rdma_link_down=1``
    -> the fabric diagnostic fails on the held port; the LINK_INACTIVE
    co-finding diagnoses the restored port; nothing reboots.

    :func:`observed_exposure` closes the window right after the collector's
    confirming tick as the probe observes it on the node: the second tick --
    which the ``rdma_link_down`` chain's VALIDATE_FABRIC judges (FAILED on the
    held port) and which confirms the ``efa_inventory_mismatch`` LINK_INACTIVE
    co-finding -- and before a third; the deadman is only the fallback (a fixed
    1.7-interval bound measured from the restart lost to the collector's
    startup phase live, 2026-10-26). The co-finding's diagnostic judges fresh
    readings (batch 175) and
    would escalate a port still inactive to REBOOT_NODE by design, so it only
    ever sees the restored port: SUCCEEDED, RECOVERED, no RESTART_NODE, under
    HOST_SOURCE_REPLAY alone. A verdict the product reached on the restored
    port, or an inactive reading after the restore, is INCONCLUSIVE (the bound
    decided nothing about the product), never a verdict on it.
    """

    run_survey(ctx)
    fixture = EfaSourceFixture(ctx)
    target = state_target(fixture.survey())
    ctx.evidence.write("efa-target", target.describe())
    fixture.bind("efa-state", target.dest("state"), LINK_DOWN_TEXT)
    ticks = confirmation_ticks(ctx)
    started = utc_now()
    opened = fixture.open(restore_seconds=exposure_fallback_seconds(ctx, ticks))
    # The probe closes the window right after the collector's confirming tick,
    # observed on the node; the deadman is only the fallback.
    exposure = observed_exposure(
        ctx, fixture.replay, ticks=ticks, name="link-inactive", opened=opened
    )
    fixture.closed_at = ctx.clock()
    restore_at = parse_time(exposure["closed_at"]) or utc_now()
    closed_by = str(exposure["closed_by"])
    opened_at = parse_time(opened.get("opened_at")) or started
    exposure_seconds = round((restore_at - opened_at).total_seconds(), 1)
    ctx.evidence.write("source-window-close", exposure)
    metrics = {LINK_METRIC, INVENTORY_METRIC, MISSING_METRIC}
    detection = wait_for_incident(
        ctx,
        started,
        LINK_METRIC,
        device=target.port_device,
        timeout_seconds=ctx.host_interval * 3 + 120,
        metrics=metrics,
        name="link-inactive",
    )
    incidents = finding_incidents(detection, LINK_METRIC, device=target.port_device)
    held = (
        _await_validation(ctx, started, incidents[0], metrics=metrics, name="fabric")
        if incidents
        else detection
    )
    ctx.evidence.append_oracle(
        [
            {
                "held_port": target.port_device,
                "ticks_required": ticks,
                "ticks_observed": exposure["ticks_observed"],
                "exposure_seconds": exposure_seconds,
                "restore_at": restore_at.isoformat(),
                "closed_by": closed_by,
            }
        ],
        source="efa-state",
    )
    if not incidents and not metric_values(held, LINK_METRIC, target.port_device):
        raise VariantInconclusive(
            f"the bounded exposure ({exposure_seconds} s, closed by {closed_by}) ended "
            f"before the product read "
            f"{LINK_METRIC} on {target.port_device}: no sample, nothing to judge"
            + _gaps(exposure, held, LINK_METRIC, ticks, device=target.port_device)
        )
    judged = (
        final_execution(
            workflows_of(held, str(incidents[0]["incident_id"]))[0], "VALIDATE_FABRIC"
        )
        if incidents and workflows_of(held, str(incidents[0]["incident_id"]))
        else None
    )
    judged_at = parse_time((judged or {}).get("updated_at"))
    if (
        judged is not None
        and judged.get("status") == "SUCCEEDED"
        and judged_at is not None
        and judged_at > restore_at - timedelta(seconds=CLOCK_TOLERANCE_SECONDS)
    ):
        raise VariantInconclusive(
            "the fabric diagnostic judged the port after the bounded exposure ended "
            f"({judged_at.isoformat()} > restore {restore_at.isoformat()}): its "
            "SUCCEEDED is about the restored port, not the held one"
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
    stages["real_restored"] = fixture.restored_errors()
    recovered = await_metric_recovery(
        ctx,
        LINK_METRIC,
        1.0,
        device=target.port_device,
        name="link-recovery",
        since=restore_at,
    )
    stages["recovery"] = recovered["errors"]
    inventory_incidents = finding_incidents(held, INVENTORY_METRIC)
    ids = [str(item["incident_id"]) for item in (*incidents, *inventory_incidents)]
    final = wait_settled(
        ctx,
        started,
        ids,
        timeout_seconds=ctx.host_summary + 300,
        metrics=metrics,
        name="link-settled",
    )
    late = _late_readings(final, MISSING_METRIC, None, restore_at)
    if late:
        raise VariantInconclusive(
            f"the exposure bound did not hold: {len(late)} {MISSING_METRIC} reading(s) "
            f"after the restore at {restore_at.isoformat()} (first "
            f"{late[0]['observed_at']}); the designed escalation on fresh inactive "
            "evidence applies and is not a verdict on the product"
        )
    stages["settled"] = (
        [] if settled(final, ids) else [f"incidents did not settle: {ids}"]
    )
    stages["co_finding_diagnosed"] = [
        f"the LINK_INACTIVE co-finding incident {item.get('incident_id')} is "
        f"{item.get('state')}, not RECOVERED, after the port was restored"
        for item in final.get("incidents") or []
        if str(item.get("incident_id"))
        in {str(inc["incident_id"]) for inc in inventory_incidents}
        and item.get("state") != "RECOVERED"
    ] + [
        f"VALIDATE_FABRIC of the co-finding waited on missing_inventory_action_completion "
        f"({workflow.get('request_id')})"
        for inc in inventory_incidents
        for workflow in workflows_of(final, str(inc["incident_id"]))
        for item in executions(workflow, "VALIDATE_FABRIC")
        if "missing_inventory_action_completion" in str(item.get("details") or "")
    ]
    stages["no_reboot_after_restore"] = hardware_errors(final)
    return _result(
        stages,
        notes=[
            f"bound {target.dest('state')} = {LINK_DOWN_TEXT!r} (phys_state real) for a "
            f"bounded exposure of {exposure_seconds} s (closed by {closed_by} after "
            f"{exposure['ticks_observed']} observed collector tick(s)); VALIDATE_FABRIC "
            "must fail on the held port and nothing may reboot or cordon",
            f"inventory co-finding samples: {len(inventory_samples)}; the co-finding's "
            "diagnostic judges the restored port",
        ],
        extra={
            "target": target.describe(),
            "recovery": recovered,
            "exposure_seconds": exposure_seconds,
            "ticks_required": ticks,
            "ticks_observed": exposure["ticks_observed"],
            "ticks": exposure["ticks"],
            "restore_at": restore_at.isoformat(),
            "closed_by": closed_by,
        },
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
    """A counter the netdev stops exposing -> MISSING_COUNTER marker, never a zero delta.

    The marker is read from the product's batches, so the product must
    *deliver* the tick on which it changed: its edge filter keys the marker by
    its content (``collectors/host/network.py`` ``not_supported_edge_reason``)
    since the live 2026-10-25 run, where the marker named the withdrawn counter
    on every tick and no batch carrying it ever left the node. What each batch
    carried for the netdev is written as ``counter-present-readings`` and
    ``counter-missing-readings`` so a failed run shows what was read.
    """

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
    ctx.evidence.write(
        "counter-present-readings", device_readings(present, interface, [metric])
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
    missing_readings = device_readings(observed, interface, [metric])
    ctx.evidence.write("counter-missing-readings", missing_readings)
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
            f"no {UNSUPPORTED_METRIC} MISSING_COUNTER marker names {metric} on "
            f"{interface} since the withdrawal ({readings_summary(missing_readings)})"
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
    poll: bool = True,
) -> dict[str, Any]:
    """Step both readers every half interval, polling the store when asked.

    A collector tick that sees no step reads a 0 delta: a reading, which ends
    the signal's episode for a reason that is not the reset under test (live
    2026-10-25: every step cost two probe round trips plus a store read, ~45 s
    against 15 s ticks, so most ticks shipped a 0 and each later step minted
    its own incident). The step period is therefore measured from the step,
    the store is read only when the caller can accept something, and every
    step is written with its time so the verdict can be read against the
    product's per-batch readings (``reset-readings``).
    """

    start = ctx.clock()
    last: dict[str, Any] = {}
    last_poll: datetime | None = None
    step_seconds = max(3, ctx.host_interval // 2)
    wanted = set(metrics)
    steps: list[dict[str, Any]] = []
    while True:
        stepped_at = ctx.clock()
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
        steps.append({"at": stepped_at.isoformat(), **values})
        now = ctx.clock()
        if poll and (
            last_poll is None or (now - last_poll).total_seconds() >= STORE_POLL_SECONDS
        ):
            last = activity(ctx, since, metrics=wanted)
            last_poll = now
        ctx.evidence.write(
            f"{name}-ramp",
            {"values": dict(values), "at": now.isoformat(), "steps": steps},
        )
        if poll and accept(last):
            return last
        if (now - start).total_seconds() >= timeout_seconds:
            return last
        ctx.sleep(max(0.0, step_seconds - (ctx.clock() - stepped_at).total_seconds()))


def counter_reset(ctx: VariantContext) -> dict[str, Any]:
    """A ramping counter reset below its baseline: not a decrement, not a new fault.

    Both readers ramp until their deltas mint one incident each, then fall to 1
    and ramp on. The reset tick must read UNKNOWN on both: a ``_rdma`` /
    ``_efa_network`` collection error naming the port or netdev, the counter
    and ``delta unknown``, and no delta of the counter or its aggregate in
    that batch -- the clamped 0 reading of the old ``_rdma`` is the FAIL. A 0
    reading on any *other* tick is a reading by the product's contract
    (``_ZERO_IS_A_READING``): a tick the ramp did not reach ends the episode
    and the next step opens a new one, so the incidents are judged against
    the episodes the product's own readings show, never against 1 (live
    2026-10-25: the first run flagged seven such readings and four incidents
    per reader as "the reset re-minted the fault" while the reset tick itself
    was UNKNOWN on both readers).
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
        poll=False,
    )
    ctx.hold(ctx.host_interval + 5, name="reset-settle")
    after = activity(ctx, started, metrics=metrics)
    readings = {
        "sysfs": device_readings(
            after, target.port_device, (ERRORS_METRIC, target.counter_metric)
        ),
        "ethtool": device_readings(after, ethtool[0], (metric,)) if ethtool else [],
    }
    ctx.evidence.write("reset-readings", readings)
    sysfs_incidents = finding_incidents(after, ERRORS_METRIC, device=target.port_device)
    sysfs_needles = (
        "_rdma",
        target.port_device,
        str(target.counter),
        "reset",
        "delta unknown",
    )
    gaps = {
        "sysfs": len(
            zero_readings(after, ERRORS_METRIC, target.port_device, after=started)
        )
    }
    stages: dict[str, list[str]] = {
        "fault_seen": []
        if finding_incidents(before, ERRORS_METRIC, device=target.port_device)
        else [
            f"no {ERRORS_METRIC} finding on {target.port_device} while the counter ramped"
        ],
        "no_decrement": negative_readings(
            after, (ERRORS_METRIC, target.counter_metric), target.port_device
        ),
        "sysfs_not_reminted": _reminted(
            sysfs_incidents, after, ERRORS_METRIC, target.port_device
        ),
        "sysfs_reset_unknown": (
            []
            if collection_errors(after, *sysfs_needles)
            else [
                f"no _rdma collection error names the {target.port_device} "
                f"{target.counter} reset with 'delta unknown': the sysfs reader did "
                "not mark the reset UNKNOWN"
            ]
        )
        + reset_tick_readings(
            after,
            sysfs_needles,
            (ERRORS_METRIC, target.counter_metric),
            target.port_device,
        ),
        "hardware": hardware_errors(after),
        "not_isolated": _not_isolated(ctx, after),
    }
    if ethtool is not None:
        gaps["ethtool"] = len(zero_readings(after, metric, ethtool[0], after=started))
        stages.update(_ethtool_reset_stages(before, after, ethtool, metric))
        stages["no_decrement"] += negative_readings(after, (metric,), ethtool[0])
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
            "ticks the ramp did not reach shipped a 0 reading (an episode's end by the "
            f"product's contract, not the reset): {gaps}; incidents are judged against "
            "the episodes the readings show (reset-readings)",
        ],
        extra={
            "target": target.describe(),
            "reset_at": reset_at.isoformat(),
            "values": values,
            "zero_readings_between_steps": gaps,
        },
    )


def _ethtool_reset_stages(
    before: dict[str, Any],
    after: dict[str, Any],
    ethtool: tuple[str, str],
    metric: str,
) -> dict[str, list[str]]:
    """The ethtool reader's half of the ``counter_reset`` verdict."""

    interface, counter = ethtool
    needles = ("_efa_network", interface, counter, "reset", "delta unknown")
    return {
        "ethtool_fault_seen": []
        if finding_incidents(before, metric, device=interface)
        else [f"no {metric} finding on {interface}"],
        "ethtool_not_reminted": _reminted(
            finding_incidents(after, metric, device=interface), after, metric, interface
        ),
        "ethtool_reset_unknown": (
            []
            if collection_errors(after, *needles)
            else [
                f"no _efa_network collection error names the {interface} "
                f"{counter} reset with 'delta unknown'"
            ]
        )
        + reset_tick_readings(after, needles, (metric,), interface),
    }


def _reminted(
    incidents: Sequence[Mapping[str, Any]],
    document: Mapping[str, Any],
    metric: str,
    device: str,
) -> list[str]:
    """More incidents than the readings' episodes: a fault minted without a reading."""

    count = episodes(document, metric, device)
    if len(incidents) <= count:
        return []
    return [
        f"the reset re-minted the fault: {len(incidents)} {metric} incidents on "
        f"{device} for {count} episode(s) in the product's own readings"
    ]


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
        f"bind the port's state file to {LINK_DOWN_TEXT!r} inside the unit (phys_state "
        "real) for a bounded exposure: the window's deadman closes it 1.7 collector "
        "intervals after it opened (two ticks, never a third)",
        f"{LINK_METRIC}=1 on the port -> WARNING -> VALIDATE_FABRIC fails on the held port "
        "(judged before the restore); the efa_inventory_mismatch co-finding is "
        "LINK_INACTIVE and its diagnostic SUCCEEDS on the restored port (RECOVERED, never "
        "missing_inventory_action_completion); no RESTART_NODE, no cordon; the link reads "
        "up again after the window closes",
        link_inactive,
        risk="M",
        max_minutes=35,
        cleanup=(
            "the window closes itself at the exposure bound (deadman); the runner's close "
            "confirms it and reads the real state back inside the unit"
        ),
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
        "no negative delta, one incident per reader (a reset is not a new fault); the reset "
        "tick is UNKNOWN on both readers: a _rdma / _efa_network collection error naming "
        "the port or netdev, the counter and 'delta unknown', with no rdma_errors_delta, "
        "rdma_<counter>_delta or efa_*_errors_delta reading in that batch and never a "
        "clamped 0; the ramp is measured from the new baseline afterwards; no hardware",
        counter_reset,
        risk="M",
        max_minutes=30,
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
