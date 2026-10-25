"""GF-GENERAL-LIVE-015: BMC sensor alerts through the product's own ``ipmitool`` reader.

The host collector (``collectors/host/system_metrics.py`` ``_bmc``) runs
``ipmitool sensor`` every tick when the tool is on its PATH, grades the fourth
``|`` column of every row (``cr``/``critical``/``nr``/``non-recoverable`` count,
rows with fewer than four columns are skipped) into ``bmc_critical_sensor``
(``>= 1`` is CRITICAL QUARANTINE, ``host_health.py``), and reports a non-zero
exit or a table without a single valid row as UNKNOWN (a ``_bmc`` collection
error, no sample). Without ``ipmitool`` it reports nothing.

EC2 instances expose no BMC and the AMIs carry no ``ipmitool``, so the grading
chain is reachable only as ``live-replay``: the shared host source replay
(``general_live_source_replay.py``, approval ``HOST_SOURCE_REPLAY``) puts the
reviewed wrapper first on the host collector unit's PATH with
``allow_missing_real`` -- the wrapper serves the plan's reviewed sensor table or
exit status for ``ipmitool sensor`` and answers "command not found" to anything
else, never inventing a BMC. Parsing, grading, the database and the QUARANTINE
/ validated-restore chain stay real; the wrapper's journal is the oracle that
the product asked. ``unsupported_bmc`` is the read-only capability record of
this BMC-less site; ``healthy`` would need a real BMC. A critical table is
replaced by a healthy one (the product reads the recovery) before the window
closes and the validated restore runs. ``proof_scope`` is ``live-replay``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from scripts.e2e.regional.general_live_common import (
    activity,
    await_metric_recovery,
    restore_incidents,
    run_survey,
    wait_for_incident,
    wait_settled,
)
from scripts.e2e.regional.general_live_fixture import (
    VariantBlocked,
    VariantContext,
    VariantInconclusive,
    utc_now,
)
from scripts.e2e.regional.general_live_registry import (
    CAPABILITY,
    HARDWARE_OPERATIONS,
    LIVE_REPLAY,
    OBSERVATION,
    VariantSpec,
    deferred_variant,
)
from scripts.e2e.regional.general_live_source_replay import (
    HOST_SOURCE_REPLAY,
    HostSourceReplay,
    command_plan,
    host_source_replay,
    rule,
)
from scripts.e2e.regional.general_live_verdicts import (
    ChainExpectation,
    chain_errors,
    finding_incidents,
    hardware_errors,
    node_isolated,
    samples,
    verdict,
)

CASE = "GF-GENERAL-LIVE-015"
TOOL = "ipmitool"
METRIC = "bmc_critical_sensor"
#: The product's collection-error prefix for the BMC contributor.
BMC_CONTRIBUTOR = "_bmc"
BMC_QUERY_FAILED = "BMC sensor query failed"
QUARANTINE_BUDGET = 420
MARKER_TICKS = 8
SETTLE_TICKS = 2
AUDIT_TAIL = 400
ISOLATION_FORBIDDEN = HARDWARE_OPERATIONS | {"STOP_WORKLOADS"}
#: What ``ipmitool`` prints on a host without a BMC device.
IPMI_NO_DEVICE_STDERR = (
    "Could not open device at /dev/ipmi0 or /dev/ipmi/0 or /dev/ipmidev/0: "
    "No such file or directory\n"
)
IPMI_FAILED_EXIT = 1

#: Sensor rows in ``ipmitool sensor`` shape:
#: ``name | value | units | status | lnr | lcr | lnc | unc | ucr | unr``.
_HEALTHY_ROWS = (
    (
        "Inlet Temp",
        "28.000",
        "degrees C",
        "ok",
        "na",
        "na",
        "na",
        "40.000",
        "45.000",
        "50.000",
    ),
    (
        "CPU1 Temp",
        "52.000",
        "degrees C",
        "ok",
        "na",
        "na",
        "na",
        "90.000",
        "95.000",
        "100.000",
    ),
    ("Fan1", "4200.000", "RPM", "ok", "na", "300.000", "500.000", "na", "na", "na"),
    ("PSU1 Status", "0x1", "discrete", "0x0180", "na", "na", "na", "na", "na", "na"),
)
#: Two critical sensors in one table (a hot CPU and the fan that failed with it).
CRITICAL_SENSORS = ("CPU1 Temp", "Fan1")
_CRITICAL_ROWS = (
    _HEALTHY_ROWS[0],
    (
        "CPU1 Temp",
        "96.000",
        "degrees C",
        "cr",
        "na",
        "na",
        "na",
        "90.000",
        "95.000",
        "100.000",
    ),
    ("Fan1", "0.000", "RPM", "cr", "na", "300.000", "500.000", "na", "na", "na"),
    _HEALTHY_ROWS[3],
)
NONRECOVERABLE_SENSOR = "CPU1 Temp"
_NONRECOVERABLE_ROWS = (
    _HEALTHY_ROWS[0],
    (
        "CPU1 Temp",
        "104.000",
        "degrees C",
        "nr",
        "na",
        "na",
        "na",
        "90.000",
        "95.000",
        "100.000",
    ),
    _HEALTHY_ROWS[2],
    _HEALTHY_ROWS[3],
)
#: Status words outside the status column and a three-column row: not alerts.
_WRONG_COLUMN_ROWS = (
    *_HEALTHY_ROWS,
    ("CR Probe", "12.000", "cr", "ok", "na", "na", "na", "na", "na", "na"),
    ("Weird Probe", "cr", "Volts", "ok", "na", "na", "na", "na", "na", "na"),
    ("nr", "0x0", "discrete", "0x0080", "na", "na", "na", "na", "na", "na"),
    ("Short Row", "1.000", "RPM"),
)
#: One sensor with no reading among healthy ones.
UNAVAILABLE_SENSOR = "PSU2 Temp"
_UNAVAILABLE_ROWS = (
    *_HEALTHY_ROWS,
    ("PSU2 Temp", "na", "degrees C", "na", "na", "na", "na", "na", "na", "na"),
)


def sensor_table(rows: Iterable[Sequence[str]]) -> str:
    """Rows joined the way ``ipmitool sensor`` prints them."""

    return "".join(" | ".join(row) + "\n" for row in rows)


HEALTHY_TABLE = sensor_table(_HEALTHY_ROWS)
CRITICAL_TABLE = sensor_table(_CRITICAL_ROWS)
NONRECOVERABLE_TABLE = sensor_table(_NONRECOVERABLE_ROWS)
WRONG_COLUMN_TABLE = sensor_table(_WRONG_COLUMN_ROWS)
UNAVAILABLE_TABLE = sensor_table(_UNAVAILABLE_ROWS)


def table_rules(table: str) -> list[dict[str, Any]]:
    return [rule("sensor", "replace_output", stdout=table)]


def command_failure_rules() -> list[dict[str, Any]]:
    return [
        rule("sensor", "exit", exit_code=IPMI_FAILED_EXIT, stderr=IPMI_NO_DEVICE_STDERR)
    ]


# --------------------------------------------------------------------------- #
# the window, the oracle, the product's records
# --------------------------------------------------------------------------- #
def open_bmc_window(
    ctx: VariantContext, rules: Sequence[Mapping[str, Any]]
) -> tuple[HostSourceReplay, dict[str, Any]]:
    """Wrap ``ipmitool`` with ``rules`` inside the host collector unit.

    ``allow_missing_real``: EC2 has no BMC and usually no ``ipmitool``; the shim
    then serves only the plan's whole-output rules. Reusable by any category
    that needs a BMC reading to reach the product.
    """

    replay = host_source_replay(ctx)
    replay.create()
    restore_seconds = ctx.spec.max_minutes * 60
    plan = command_plan(TOOL, rules, not_after_seconds=restore_seconds)
    ctx.evidence.write("bmc-plan", plan)
    replay.install_wrapper(TOOL, plan, allow_missing_real=True)
    opened = replay.open_command([TOOL], restore_seconds=restore_seconds)
    ctx.evidence.write("source-open", opened)
    return replay, opened


def served_calls(audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        entry
        for entry in audit.get("entries") or []
        if isinstance(entry, dict)
        and entry.get("query") == "sensor"
        and entry.get("applied_rules")
    ]


def require_served(
    ctx: VariantContext, replay: HostSourceReplay
) -> list[dict[str, Any]]:
    """The oracle: the product ran ``ipmitool sensor`` and was served the rule."""

    audit = replay.audit(TOOL, tail=AUDIT_TAIL)
    ctx.evidence.write(f"{TOOL}-audit", audit)
    hits = served_calls(audit)
    ctx.evidence.append_oracle(hits, source=f"{TOOL}-audit")
    if not hits:
        seen = [
            entry.get("argv")
            for entry in audit.get("entries") or []
            if isinstance(entry, dict)
        ]
        raise VariantInconclusive(
            "the host collector never had the ipmitool sensor rule served; "
            f"journal: {seen[-8:]}"
        )
    return hits


def bmc_errors(observed: Mapping[str, Any]) -> list[str]:
    return [
        str(error)
        for batch in observed.get("batches") or []
        for error in batch.get("collection_errors") or []
        if str(error).startswith(BMC_CONTRIBUTOR)
    ]


def await_bmc(
    ctx: VariantContext,
    since: Any,
    accept: Callable[[dict[str, Any]], bool],
    *,
    name: str,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    last: dict[str, Any] = {}

    def poll() -> dict[str, Any] | None:
        nonlocal last
        last = activity(ctx, since, metrics={METRIC})
        return last if accept(last) else None

    value = ctx.wait_until(
        poll,
        timeout_seconds=(
            timeout_seconds
            if timeout_seconds is not None
            else ctx.host_interval * MARKER_TICKS + 60
        ),
        poll_seconds=ctx.host_interval,
        name=name,
    )
    return value if value is not None else last


def settled_read(
    ctx: VariantContext, since: Any, accept: Callable[[dict[str, Any]], bool], name: str
) -> dict[str, Any]:
    await_bmc(ctx, since, accept, name=name)
    ctx.hold(ctx.host_interval * SETTLE_TICKS + 30, name=f"{name}-settle")
    return activity(ctx, since, metrics={METRIC})


def close_bmc_window(ctx: VariantContext, replay: HostSourceReplay) -> list[str]:
    """Close the window; afterwards the product must read no replayed table.

    There is no real BMC to answer again: the proof is a clean close (shim and
    drop-in gone, deadman disarmed) and that the batches after it carry no
    critical count and no replayed ``bmc_critical_sensor`` reading at all when
    the node has no ``ipmitool`` of its own.
    """

    closed_at = utc_now()
    closed = replay.close()
    ctx.evidence.write("source-close", closed)
    errors: list[str] = []
    if not closed.get("clean", True):
        errors.append(f"the source window did not close clean: {closed.get('residue')}")
    observed = await_bmc(
        ctx,
        closed_at,
        lambda current: bool(current.get("batches")),
        name="bmc-after-close",
        timeout_seconds=ctx.host_interval * MARKER_TICKS + 60,
    )
    if not observed.get("batches"):
        errors.append("no host batch was accepted after the window closed")
    critical = [
        sample
        for sample in samples(observed, METRIC)
        if isinstance(sample.get("value"), (int, float)) and sample["value"] >= 1
    ]
    if critical:
        errors.append(
            f"a critical BMC reading survived the window: {critical[-1].get('value')}"
        )
    tool_present = bool((ctx.survey.get("tools") or {}).get(TOOL))
    if not tool_present and samples(observed, METRIC):
        errors.append(
            "bmc_critical_sensor samples continue after the window closed on a node "
            "without ipmitool: the shim is still on the unit's PATH"
        )
    return errors


def not_isolated(ctx: VariantContext, why: str) -> list[str]:
    node = ctx.regional.node_snapshot(ctx.settings.node)
    return [f"node isolated on {why}"] if node_isolated(node) else []


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


def sensor_identity(observed: Mapping[str, Any], sensors: Iterable[str]) -> list[str]:
    """Each alerting sensor's name reaches the product's records (spec 015).

    Looked for in the ``bmc_critical_sensor`` sample labels, the minted
    incidents, their markers and notifications -- wherever a fix lands it.
    """

    text = json.dumps(
        {
            "labels": [
                sample.get("labels") for sample in samples(dict(observed), METRIC)
            ],
            "incidents": finding_incidents(dict(observed), METRIC),
            "markers": observed.get("markers") or [],
            "notifications": observed.get("notifications") or [],
        },
        default=str,
    )
    return [
        f"the critical sensor {name!r} is named nowhere in the product's records"
        for name in sensors
        if name not in text
    ]


# --------------------------------------------------------------------------- #
# variants
# --------------------------------------------------------------------------- #
def _isolation_variant(
    ctx: VariantContext,
    *,
    table: str,
    sensors: tuple[str, ...],
    status: str,
    reason: str,
) -> tuple[dict[str, list[str]], dict[str, Any], dict[str, Any]]:
    """A replayed critical/non-recoverable table -> QUARANTINE -> healthy table -> restore.

    Returns ``(stages, settled read, evidence extra)`` for the variant's result.
    """

    run_survey(ctx)
    replay, _opened = open_bmc_window(ctx, table_rules(table))
    started = utc_now()
    detection = wait_for_incident(
        ctx,
        started,
        METRIC,
        timeout_seconds=QUARANTINE_BUDGET,
        metrics={METRIC},
        name=f"bmc-{status}-quarantine",
    )
    require_served(ctx, replay)
    incidents = finding_incidents(detection, METRIC)
    settled = (
        wait_settled(
            ctx,
            started,
            [item["incident_id"] for item in incidents],
            timeout_seconds=QUARANTINE_BUDGET,
            metrics={METRIC},
            name=f"bmc-{status}-settled",
        )
        if incidents
        else detection
    )
    counts = [sample.get("value") for sample in samples(settled, METRIC)]
    stages = {
        "chain": chain_errors(
            settled,
            ChainExpectation(
                metric=METRIC,
                operations=("MARK_UNSCHEDULABLE", "QUARANTINE"),
                forbidden=ISOLATION_FORBIDDEN,
                incident_states=frozenset({"QUARANTINED"}),
            ),
        ),
        "count": []
        if any(value == len(sensors) for value in counts)
        else [
            f"bmc_critical_sensor never read {len(sensors)} for the {len(sensors)} "
            f"{status} rows: {counts[-4:]}"
        ],
        "no_error": [
            f"a parseable table was graded as a collection error: {error[:120]}"
            for error in bmc_errors(settled)
        ],
    }
    # Recovery: the same wrapper now serves a healthy table, the product reads
    # zero critical sensors (its own recovery record), then the window closes
    # and the product's validated restore hands the node back.
    recovered_at = utc_now()
    replay.replan(
        TOOL,
        command_plan(
            TOOL,
            table_rules(HEALTHY_TABLE),
            not_after_seconds=ctx.spec.max_minutes * 60,
        ),
    )
    recovered = await_metric_recovery(
        ctx, METRIC, 1.0, since=recovered_at, name=f"bmc-{status}-recovered"
    )
    stages["recovered"] = recovered["errors"]
    stages["window_closed"] = close_bmc_window(ctx, replay)
    restore_incidents(ctx, incidents, reason=reason)
    extra = {
        "sensors": list(sensors),
        "counts": counts,
        "recovery": recovered,
        **replay.evidence(tool=TOOL),
    }
    return stages, settled, extra


def critical(ctx: VariantContext) -> dict[str, Any]:
    """Live-replay: two ``cr`` rows -> one QUARANTINE incident naming both sensors."""

    stages, settled, extra = _isolation_variant(
        ctx,
        table=CRITICAL_TABLE,
        sensors=CRITICAL_SENSORS,
        status="cr",
        reason="GF-GENERAL-LIVE-015 critical restore",
    )
    stages["sensor_identity"] = sensor_identity(settled, CRITICAL_SENSORS)
    return _result(
        stages,
        notes=[
            "two cr rows in the 4th column -> bmc_critical_sensor=2 -> MARK_UNSCHEDULABLE "
            "+ QUARANTINE (one incident); the spec also wants the alert to carry the "
            "sensor names and ranges (sensor_identity stage)"
        ],
        extra=extra,
    )


def nonrecoverable(ctx: VariantContext) -> dict[str, Any]:
    """Live-replay: one ``nr`` row -> QUARANTINE -> healthy table -> restore."""

    stages, _settled, extra = _isolation_variant(
        ctx,
        table=NONRECOVERABLE_TABLE,
        sensors=(NONRECOVERABLE_SENSOR,),
        status="nr",
        reason="GF-GENERAL-LIVE-015 nonrecoverable restore",
    )
    return _result(
        stages,
        notes=[
            "an nr status in the 4th column -> bmc_critical_sensor=1 -> MARK_UNSCHEDULABLE "
            "+ QUARANTINE; healthy table read as the recovery; validated restore"
        ],
        extra=extra,
    )


def _not_an_alert(
    ctx: VariantContext, *, table: str, name: str, note: str, why: str
) -> dict[str, Any]:
    """A reviewed table the product must grade as zero critical, no error, no finding."""

    run_survey(ctx)
    replay, _opened = open_bmc_window(ctx, table_rules(table))
    started = utc_now()
    observed = settled_read(
        ctx,
        started,
        lambda current: bool(samples(current, METRIC) or bmc_errors(current)),
        name,
    )
    require_served(ctx, replay)
    readings = samples(observed, METRIC)
    stages = {
        "sample_present": []
        if readings
        else ["no bmc_critical_sensor sample: the sensor table was not graded"],
        "zero_critical": [
            f"bmc_critical_sensor={sample.get('value')} read from a table with no "
            "cr/nr status"
            for sample in readings
            if sample.get("value") != 0
        ],
        "no_error": [
            f"a parseable table was graded as a collection error: {error[:120]}"
            for error in bmc_errors(observed)
        ],
        "no_finding": chain_errors(
            observed, ChainExpectation(metric=METRIC, present=False)
        ),
        "hardware": hardware_errors(observed),
        "not_isolated": not_isolated(ctx, why),
        "window_closed": close_bmc_window(ctx, replay),
    }
    return _result(
        stages,
        notes=[note],
        extra={"readings": readings[-4:], **replay.evidence(tool=TOOL)},
    )


def wrong_column(ctx: VariantContext) -> dict[str, Any]:
    """Live-replay: ``cr``/``nr`` outside the status column and a 3-column row are not alerts."""

    return _not_an_alert(
        ctx,
        table=WRONG_COLUMN_TABLE,
        name="bmc-wrong-column",
        note=(
            "cr in the units and value columns, a sensor named nr and a 3-column row: "
            "bmc_critical_sensor=0, no error, no finding"
        ),
        why="a status word outside the 4th sensor column",
    )


def unavailable_sensor(ctx: VariantContext) -> dict[str, Any]:
    """Live-replay: an ``na`` sensor among healthy ones is neither critical nor an error."""

    result = _not_an_alert(
        ctx,
        table=UNAVAILABLE_TABLE,
        name="bmc-unavailable",
        note=(
            f"{UNAVAILABLE_SENSOR} reads na/na: not counted critical, not an error, "
            "no finding, no isolation; the product keeps no per-sensor "
            "unavailability marker (a table of only na rows would read 0; not exercised)"
        ),
        why="an unavailable BMC sensor",
    )
    result["unavailable_sensor"] = UNAVAILABLE_SENSOR
    return result


def command_failure(ctx: VariantContext) -> dict[str, Any]:
    """Live-replay: ``ipmitool sensor`` exits 1 -> UNKNOWN (error, no sample), no finding."""

    run_survey(ctx)
    replay, _opened = open_bmc_window(ctx, command_failure_rules())
    started = utc_now()
    observed = settled_read(
        ctx,
        started,
        lambda current: bool(bmc_errors(current) or samples(current, METRIC)),
        "bmc-command-failure",
    )
    require_served(ctx, replay)
    errors = bmc_errors(observed)
    stages = {
        "unknown_error": []
        if any(BMC_QUERY_FAILED in error for error in errors)
        else [
            f"no _bmc collection error naming {BMC_QUERY_FAILED!r}; errors seen: "
            f"{[error[:120] for error in errors][-3:]}"
        ],
        "no_sample": [
            f"a failed ipmitool query was graded as bmc_critical_sensor="
            f"{sample.get('value')}"
            for sample in samples(observed, METRIC)
        ],
        "no_finding": chain_errors(
            observed, ChainExpectation(metric=METRIC, present=False)
        ),
        "hardware": hardware_errors(observed),
        "not_isolated": not_isolated(ctx, "a failed BMC query"),
        "window_closed": close_bmc_window(ctx, replay),
    }
    return _result(
        stages,
        notes=[
            "ipmitool sensor exit 1 ('Could not open device') is a _bmc collection "
            "error (UNKNOWN), never a healthy zero and never an isolation"
        ],
        extra={"collection_errors": errors, **replay.evidence(tool=TOOL)},
    )


def unsupported_bmc(ctx: VariantContext) -> dict[str, Any]:
    """Read-only: record that this node has no BMC and the product stays silent on it.

    Like 002 ``no_swap``: nothing is injected. The survey says whether
    ``ipmitool`` is installed; the product's records over a window say whether
    the BMC contributor produced a reading (a BMC answered -- then this variant
    does not apply), a ``_bmc`` collection error every tick (the tool is there,
    no BMC behind it: a permanent capability gap graded as UNKNOWN, which holds
    every host validation, unlike SMART's ``smart_status_unsupported``), or
    nothing (no tool: the gap is recorded here, not by the product).
    """

    survey = run_survey(ctx)
    tool_present = bool((survey.get("tools") or {}).get(TOOL))
    started_at = utc_now()
    ctx.hold(ctx.host_interval * 4 + 30, name="bmc-capability-hold")
    observed = activity(ctx, started_at, metrics={METRIC})
    readings = samples(observed, METRIC)
    errors = bmc_errors(observed)
    if readings and not errors:
        raise VariantBlocked(
            "CAPABILITY_ABSENT",
            f"the node answers ipmitool sensor (bmc_critical_sensor={readings[-1].get('value')}); "
            "this variant records a BMC-less node (use healthy instead)",
        )
    stages = {
        "no_bmc_finding": chain_errors(
            observed, ChainExpectation(metric=METRIC, present=False)
        ),
        "no_permanent_error": []
        if not errors
        else [
            "ipmitool is installed but no BMC answers; the product grades that "
            "permanent gap as a _bmc collection error every tick (holds host "
            f"validation): {errors[-1][:120]}"
        ],
        "hardware": hardware_errors(observed),
    }
    capability = {
        "ipmitool_installed": tool_present,
        "bmc_samples": len(readings),
        "bmc_collection_errors": len(errors),
        "batches": len(observed.get("batches") or []),
    }
    ctx.evidence.write("bmc-capability", capability)
    return _result(
        stages,
        notes=[
            "capability record: EC2 instance without a BMC"
            + (" and without ipmitool" if not tool_present else " (ipmitool installed)")
            + "; the product emits no bmc_critical_sensor reading, so the BMC "
            "variants needing a real BMC stay NOT_RUN(CAPABILITY_ABSENT)"
        ],
        extra={"capability": capability},
    )


def _spec(
    variant: str,
    injection: str,
    expected: str,
    execute: Callable[[VariantContext], dict[str, Any]],
    *,
    risk: str = "L",
    max_minutes: int = 16,
    cleanup: str = "close the ipmitool window; confirm clean and no replayed reading after it",
    expected_outcome: str = "PASS",
    product_gap: str = "",
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
        expected_outcome=expected_outcome,
        product_gap=product_gap,
        execute=execute,
    )


SENSOR_IDENTITY_GAP = (
    "collectors/host/system_metrics.py HostSystemMetricsMixin._bmc reports "
    "bmc_critical_sensor as a bare count (unit 'sensors', no labels): the names, "
    "readings and thresholds of the cr/nr rows never reach the finding, incident, "
    "marker or notification, so a multi-sensor alert cannot name its sensors "
    "(spec 015: 多传感器告警需保留名称及范围并去重)"
)

VARIANTS = [
    _spec(
        "critical",
        "wrapper serves an ipmitool sensor table with two cr rows (CPU1 Temp, Fan1)",
        "bmc_critical_sensor=2 -> MARK_UNSCHEDULABLE + QUARANTINE, one incident; the "
        "alert names both sensors (expected FAIL: names are dropped); healthy table -> "
        "recovered; validated restore",
        critical,
        risk="M",
        max_minutes=32,
        cleanup="healthy table read, window closed, validated restore; node schedulable",
        expected_outcome="FAIL",
        product_gap=SENSOR_IDENTITY_GAP,
    ),
    _spec(
        "nonrecoverable",
        "wrapper serves an ipmitool sensor table with one nr row",
        "bmc_critical_sensor=1 -> MARK_UNSCHEDULABLE + QUARANTINE (no STOP, no hardware); "
        "healthy table -> recovered; validated restore",
        nonrecoverable,
        risk="M",
        max_minutes=32,
        cleanup="healthy table read, window closed, validated restore; node schedulable",
    ),
    deferred_variant(
        CASE,
        "healthy",
        proof_mode=CAPABILITY,
        reason=(
            "CAPABILITY_ABSENT: EC2 instances expose no BMC; a real healthy sensor table "
            "cannot be read here, and a replayed one proves only the grading already "
            "covered by wrong_column / unavailable_sensor"
        ),
        lanes=("eks", "hp"),
        expected="healthy sensors mint no finding",
    ),
    _spec(
        "wrong_column",
        "wrapper serves a table with cr/nr in the name, value and units columns and a 3-column row",
        "bmc_critical_sensor=0; no collection error, no finding, no isolation",
        wrong_column,
    ),
    _spec(
        "unavailable_sensor",
        "wrapper serves a healthy table plus one sensor reading na with status na",
        "bmc_critical_sensor=0; not critical, not an error, no finding, no isolation",
        unavailable_sensor,
    ),
    _spec(
        "command_failure",
        "wrapper makes ipmitool sensor exit 1 ('Could not open device at /dev/ipmi0')",
        "_bmc collection error (UNKNOWN); no bmc_critical_sensor sample, no finding, no isolation",
        command_failure,
    ),
    VariantSpec(
        case_id=CASE,
        variant="unsupported_bmc",
        proof_mode=OBSERVATION,
        lanes=("eks", "hp"),
        risk="L",
        flags="S",
        injection="none (read-only): survey ipmitool and watch the BMC contributor",
        expected="no BMC reading and no finding; the capability gap is recorded (not a PASS "
        "of any BMC grading); a permanent _bmc error on a BMC-less node is a FAIL",
        cleanup="none",
        max_minutes=8,
        execute=unsupported_bmc,
    ),
]

__all__ = [
    "CASE",
    "CRITICAL_SENSORS",
    "CRITICAL_TABLE",
    "HEALTHY_TABLE",
    "METRIC",
    "NONRECOVERABLE_TABLE",
    "SENSOR_IDENTITY_GAP",
    "UNAVAILABLE_TABLE",
    "VARIANTS",
    "WRONG_COLUMN_TABLE",
    "bmc_errors",
    "close_bmc_window",
    "command_failure_rules",
    "open_bmc_window",
    "require_served",
    "sensor_identity",
    "sensor_table",
    "served_calls",
    "table_rules",
]
