"""GF-GENERAL-LIVE-015: BMC sensor alerts through the product's own ``ipmitool`` reader.

The host collector (``collectors/host/bmc.py`` ``HostBmcMixin._bmc``) runs
``ipmitool sensor`` every tick when the tool is on its PATH and grades the
fourth ``|`` column of every row (rows with fewer than four columns are
skipped): ``bmc_critical_sensor`` is the number of distinct sensors reading
``cr``/``critical``/``nr``/``non-recoverable`` (``>= 1`` is CRITICAL QUARANTINE,
``host_health.py``), and a count ``>= 1`` carries the alerting sensors on its
labels -- ``sensors`` (names) and ``sensor_alerts`` (name, reading, units,
status, the threshold columns ipmitool printed) -- which the grader writes
into the finding's reason, so the incident reasons, the marker and the
``quarantined-node-notice-zh-v1`` notice the completed QUARANTINE plan sends
name them. A non-zero exit, a hung
tool, a table without a valid row and a table whose every status is ``na`` are
UNKNOWN (a ``_bmc`` collection error, no sample). BMC the node cannot provide
is NOT_SUPPORTED: ``bmc_status_unsupported`` ``TOOL_MISSING`` without
``ipmitool``, ``NO_DEVICE`` when ipmitool reports no IPMI device node and none
exists -- no collection error, nothing graded.

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

The capability marker is read from two product records (``marker_witness``):
the accepted host batches, which carry it on the tick it appears, changes or
clears (``host_capability_markers``: a ``not-supported:...`` edge, delivered
once), and the node's latest ``bmc_status_unsupported`` record, which every
accepted batch -- the routine health summary included -- moves and which is
the only witness of a marker that is the node's steady state (no edge, and a
summary without findings or collection errors is not evidence). Live 2026-10-26
on plain EKS: ``unsupported_bmc`` read zero batches in a whole summary period,
and ``command_failure``'s one delivered NO_DEVICE tick was stamped before the
read's ``since`` (the collector stamps a tick when it starts and runs ``_bmc``
last), so a read admits the latest record from one tick before ``since``.
"""

from __future__ import annotations

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
    latest_samples,
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
#: What ``ipmitool`` prints on a host without an IPMI device node (``open.c``,
#: ``strerror(ENOENT)``): the product reads NOT_SUPPORTED (``NO_DEVICE``).
IPMI_NO_DEVICE_STDERR = (
    "Could not open device at /dev/ipmi0 or /dev/ipmi/0 or /dev/ipmidev/0: "
    "No such file or directory\n"
)
#: A BMC that is there but does not answer (``ipmi_sdr.c``): UNKNOWN.
IPMI_TRANSIENT_STDERR = "Get Device ID command failed\nUnable to open SDR for reading\n"
IPMI_FAILED_EXIT = 1
#: The product's capability marker for BMC it cannot read (TOOL_MISSING / NO_DEVICE).
UNSUPPORTED = "bmc_status_unsupported"
#: Every BMC sample the activity reads keep.
METRICS = frozenset({METRIC, UNSUPPORTED})
#: How many collector intervals before a read's ``since`` the product's latest
#: marker record may be stamped and still be the window's. The host collector
#: stamps a batch when its tick starts and runs ``_bmc`` last, so the tick
#: whose ``ipmitool`` call answered a replan made moments before ``since`` is
#: stamped before it (live 2026-10-26 015 ``command_failure`` on eks: that one
#: tick carried the NO_DEVICE marker, the read missed it, every later tick was
#: the unchanged marker). A read of a steady state passes no lookback.
MARKER_LOOKBACK_TICKS = 1

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
    """``ipmitool sensor`` fails the way a BMC that does not answer makes it fail."""

    return [
        rule("sensor", "exit", exit_code=IPMI_FAILED_EXIT, stderr=IPMI_TRANSIENT_STDERR)
    ]


def no_device_rules() -> list[dict[str, Any]]:
    """``ipmitool sensor`` fails the way it does on a host without an IPMI device."""

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
        last = activity(ctx, since, metrics=METRICS)
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
    return activity(ctx, since, metrics=METRICS)


def marker_reason(record: Mapping[str, Any] | None) -> str:
    """The ``reason`` label of a marker sample or record; ``""`` without one."""

    return (
        "" if record is None else str((record.get("labels") or {}).get("reason") or "")
    )


def unsupported_markers(
    observed: Mapping[str, Any], reason: str
) -> list[dict[str, Any]]:
    """The ``bmc_status_unsupported`` samples whose ``reason`` label is ``reason``."""

    return [
        sample
        for sample in samples(dict(observed), UNSUPPORTED)
        if marker_reason(sample) == reason
    ]


def latest_marker(
    observed: Mapping[str, Any], *, lookback_seconds: float = 0.0
) -> dict[str, Any] | None:
    """The product's latest ``bmc_status_unsupported`` record inside the read.

    ``host_health.evaluate_metrics`` keeps the marker of every accepted batch,
    the health summary included (``LATEST_METRICS_REQUIRED``), so this is the
    witness of a marker no edge announces, and of the one delivered tick a
    read's ``since`` fell just after (``lookback_seconds``, see
    ``MARKER_LOOKBACK_TICKS``). Whatever its reason -- the caller names a wrong
    one; ``None`` when the record is older than the read or absent.
    """

    records = latest_samples(
        dict(observed), UNSUPPORTED, lookback_seconds=lookback_seconds
    )
    return records[-1] if records else None


def describe_latest(record: Mapping[str, Any] | None) -> str:
    if record is None:
        return f"no latest {UNSUPPORTED} record inside the read"
    return (
        f"latest {UNSUPPORTED} record reason={marker_reason(record)!r} "
        f"observed_at={record.get('observed_at')}"
    )


def marker_witness(
    observed: Mapping[str, Any], reason: str, *, lookback_seconds: float = 0.0
) -> dict[str, Any]:
    """Where the product recorded ``bmc_status_unsupported`` with ``reason``.

    ``source`` is ``evidence`` when an accepted batch inside the read carries
    the marker, ``metric_latest`` when only the node's latest record does (a
    steady marker, or the one delivered tick stamped just before ``since``),
    ``None`` when neither does; ``latest`` is the latest record inside the read
    whatever its reason, ``evidence`` the last two batch samples.
    """

    evidence = unsupported_markers(observed, reason)
    latest = latest_marker(observed, lookback_seconds=lookback_seconds)
    if evidence:
        source: str | None = "evidence"
    elif latest is not None and marker_reason(latest) == reason:
        source = "metric_latest"
    else:
        source = None
    return {
        "reason": reason,
        "source": source,
        "evidence": evidence[-2:],
        "latest": latest,
    }


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
    """Each alerting sensor is named where the product keeps it (spec 015).

    Three places, each checked: the ``sensors`` label of the critical
    ``bmc_critical_sensor`` sample (the collector's record), the reasons of
    the incident the finding minted (the text the quarantined-node notice of
    the completed QUARANTINE plan lists), and the ``raw_reason`` of that
    incident's marker.
    A bare count names nothing anywhere, and that was the registered gap.
    """

    activity = dict(observed)
    labelled = [
        str((sample.get("labels") or {}).get("sensors") or "")
        for sample in samples(activity, METRIC)
        if isinstance(sample.get("value"), (int, float)) and sample["value"] >= 1
    ]
    incidents = finding_incidents(activity, METRIC)
    reasons = [
        str(reason)
        for incident in incidents
        for reason in incident.get("reasons") or []
    ]
    incident_ids = {str(incident.get("incident_id")) for incident in incidents}
    markers = [
        str(marker.get("raw_reason") or "")
        for marker in activity.get("markers") or []
        if isinstance(marker, dict) and str(marker.get("incident_id")) in incident_ids
    ]
    places = (
        ("the sensors label of the critical bmc_critical_sensor sample", labelled),
        ("the reasons of its incident", reasons),
        ("the raw_reason of its incident's marker", markers),
    )
    return [
        f"the critical sensor {name!r} is not named in {place}: {texts[-2:]}"
        for name in sensors
        for place, texts in places
        if not any(name in text for text in texts)
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
        metrics=METRICS,
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
            metrics=METRICS,
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
        "sensor_identity": sensor_identity(settled, sensors),
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

    stages, _settled, extra = _isolation_variant(
        ctx,
        table=CRITICAL_TABLE,
        sensors=CRITICAL_SENSORS,
        status="cr",
        reason="GF-GENERAL-LIVE-015 critical restore",
    )
    return _result(
        stages,
        notes=[
            "two cr rows in the 4th column -> bmc_critical_sensor=2 -> MARK_UNSCHEDULABLE "
            "+ QUARANTINE (one incident); the sample labels, the incident reasons and "
            "the marker name both sensors with reading, status and thresholds "
            "(sensor_identity stage)"
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
            "+ QUARANTINE, the sensor named; healthy table read as the recovery; "
            "validated restore"
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
            "no finding, no isolation; a table of only na rows is UNKNOWN in the "
            "product (a _bmc collection error, no sample; software-tested)"
        ),
        why="an unavailable BMC sensor",
    )
    result["unavailable_sensor"] = UNAVAILABLE_SENSOR
    return result


def _no_bmc_batches(observed: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The batches carrying the NO_DEVICE capability marker."""

    wanted = {
        sample.get("batch_id") for sample in unsupported_markers(observed, "NO_DEVICE")
    }
    return [
        batch
        for batch in observed.get("batches") or []
        if isinstance(batch, dict) and batch.get("batch_id") in wanted
    ]


def command_failure(ctx: VariantContext) -> dict[str, Any]:
    """Live-replay: a failing ``ipmitool sensor`` is UNKNOWN; a missing IPMI device is NOT_SUPPORTED.

    Two answers from the same wrapper, in sequence. First the exit of a BMC
    that does not answer (``Get Device ID command failed``): a ``_bmc``
    collection error, no sample, no finding, no isolation. Then the exit of a
    host without an IPMI device node (``Could not open device at /dev/ipmi0
    ...: No such file or directory``, which is what an installed ipmitool
    prints on EC2): ``bmc_status_unsupported`` NO_DEVICE and the collection
    error stops -- a permanent capability gap is recorded, not graded as
    missing evidence every tick, and still never a healthy zero.
    """

    run_survey(ctx)
    replay, _opened = open_bmc_window(ctx, command_failure_rules())
    started = utc_now()
    observed = settled_read(
        ctx,
        started,
        lambda current: bool(
            bmc_errors(current)
            or samples(current, METRIC)
            or samples(current, UNSUPPORTED)
        ),
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
        "not_unsupported": [
            f"a BMC that does not answer was recorded as {UNSUPPORTED} "
            f"{sample.get('labels')}"
            for sample in samples(observed, UNSUPPORTED)
        ],
        "no_finding": chain_errors(
            observed, ChainExpectation(metric=METRIC, present=False)
        ),
        "hardware": hardware_errors(observed),
        "not_isolated": not_isolated(ctx, "a failed BMC query"),
    }
    replay.replan(
        TOOL,
        command_plan(
            TOOL, no_device_rules(), not_after_seconds=ctx.spec.max_minutes * 60
        ),
    )
    # Read from after the plan landed: a tick served the previous rule in the
    # seconds the install took would otherwise count as a lingering error.
    # The marker itself may also sit on the product's latest record alone:
    # the tick that answers the new plan is stamped when it started, which
    # can be before the plan landed (``MARKER_LOOKBACK_TICKS``), and every
    # tick after it is the unchanged marker, no edge.
    no_device_at = utc_now()
    lookback = ctx.host_interval * MARKER_LOOKBACK_TICKS
    unsupported = settled_read(
        ctx,
        no_device_at,
        lambda current: bool(
            samples(current, UNSUPPORTED)
            or samples(current, METRIC)
            or bmc_errors(current)
            or latest_marker(current, lookback_seconds=lookback)
        ),
        "bmc-no-device",
    )
    witness = marker_witness(unsupported, "NO_DEVICE", lookback_seconds=lookback)
    marked = _no_bmc_batches(unsupported)
    batches = [
        batch for batch in unsupported.get("batches") or [] if isinstance(batch, dict)
    ]
    last = batches[-1] if batches else {}
    stages["no_device_unsupported"] = (
        []
        if witness["source"]
        else [
            f"no {UNSUPPORTED} reason=NO_DEVICE after ipmitool answered 'Could not "
            f"open device': none of the {len(batches)} accepted batches in the read "
            f"carries it and the product has {describe_latest(witness['latest'])}; "
            f"errors: {[e[:120] for e in bmc_errors(unsupported)][-3:]}"
        ]
    )
    stages["no_device_no_error"] = [
        f"a missing IPMI device was graded as a collection error: {str(error)[:120]}"
        for batch in marked
        for error in batch.get("collection_errors") or []
        if str(error).startswith(BMC_CONTRIBUTOR)
    ] + (
        [
            "the last host batch after the no-device replan still carries a _bmc "
            f"collection error: {[str(e)[:120] for e in last.get('collection_errors') or []][-2:]}"
        ]
        if any(
            str(e).startswith(BMC_CONTRIBUTOR)
            for e in last.get("collection_errors") or []
        )
        else []
    )
    stages["no_device_no_sample"] = [
        f"a host without an IPMI device was graded as bmc_critical_sensor="
        f"{sample.get('value')}"
        for sample in samples(unsupported, METRIC)
    ]
    stages["no_device_no_finding"] = chain_errors(
        unsupported, ChainExpectation(metric=METRIC, present=False)
    )
    stages["window_closed"] = close_bmc_window(ctx, replay)
    return _result(
        stages,
        notes=[
            "ipmitool sensor exit 1 ('Get Device ID command failed') is a _bmc "
            "collection error (UNKNOWN), never a healthy zero and never an isolation; "
            "exit 1 ('Could not open device at /dev/ipmi0 ...: No such file or "
            "directory') on this BMC-less host is bmc_status_unsupported NO_DEVICE "
            "with no collection error",
            f"the NO_DEVICE marker was read from {witness['source'] or 'nowhere'}: "
            + (
                f"{len(marked)} accepted batch(es) carry it"
                if witness["source"] == "evidence"
                else describe_latest(witness["latest"])
            ),
        ],
        extra={
            "collection_errors": errors,
            "no_device_markers": witness["evidence"],
            "no_device_marker": witness,
            **replay.evidence(tool=TOOL),
        },
    )


def unsupported_bmc(ctx: VariantContext) -> dict[str, Any]:
    """Read-only: record that this node has no BMC and the product says so.

    Like 002 ``no_swap``: nothing is injected. The survey says whether
    ``ipmitool`` is installed; the product's records over a window say how the
    BMC contributor reads the node: ``bmc_status_unsupported`` TOOL_MISSING
    without the tool, NO_DEVICE with the tool and no IPMI device behind it --
    the capability gap recorded by the product, never a ``_bmc`` collection
    error every tick (which held every host validation). A BMC that answers
    (a ``bmc_critical_sensor`` reading, no error) means this variant does not
    apply.
    """

    survey = run_survey(ctx)
    tool_present = bool((survey.get("tools") or {}).get(TOOL))
    expected_reason = "NO_DEVICE" if tool_present else "TOOL_MISSING"
    started_at = utc_now()
    # No edge ever fires here: a steady capability marker travels with the
    # routine health summary, which is not evidence, so the marker is read
    # from the product's latest record, which that summary moves; the wait
    # spans one summary period. Live 2026-10-26 (eksb): zero batches in the
    # whole period, the marker in no record the read could see.
    observed = await_bmc(
        ctx,
        started_at,
        lambda current: bool(
            samples(current, UNSUPPORTED)
            or samples(current, METRIC)
            or bmc_errors(current)
            or latest_marker(current)
        ),
        name="bmc-capability",
        timeout_seconds=ctx.host_summary + ctx.host_interval * MARKER_TICKS + 60,
    )
    readings = samples(observed, METRIC)
    errors = bmc_errors(observed)
    markers = samples(observed, UNSUPPORTED)
    witness = marker_witness(observed, expected_reason)
    reasons = sorted(
        {marker_reason(sample) for sample in [*markers, witness["latest"]]} - {""}
    )
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
            "ipmitool "
            + ("is installed but no BMC answers" if tool_present else "is absent")
            + "; the product grades that permanent gap as a _bmc collection error "
            f"(holds host validation): {errors[-1][:120]}"
        ],
        "capability_marker": []
        if witness["source"]
        else [
            f"no {UNSUPPORTED} sample with reason={expected_reason} (ipmitool "
            + ("installed" if tool_present else "absent")
            + f"): none of the {len(observed.get('batches') or [])} accepted batches "
            f"carries it and the product has {describe_latest(witness['latest'])}; "
            f"reasons seen: {reasons}"
        ],
        "hardware": hardware_errors(observed),
    }
    capability = {
        "ipmitool_installed": tool_present,
        "bmc_samples": len(readings),
        "bmc_collection_errors": len(errors),
        "unsupported_reasons": reasons,
        "batches": len(observed.get("batches") or []),
        "marker_source": witness["source"],
        "latest_marker": witness["latest"],
    }
    ctx.evidence.write("bmc-capability", capability)
    return _result(
        stages,
        notes=[
            "capability record: EC2 instance without a BMC"
            + (" and without ipmitool" if not tool_present else " (ipmitool installed)")
            + f"; the product records bmc_status_unsupported {expected_reason} and "
            "emits no bmc_critical_sensor reading, so the BMC variants needing a real "
            "BMC stay NOT_RUN(CAPABILITY_ABSENT)",
            f"the marker was read from {witness['source'] or 'nowhere'}: "
            + (
                f"{len(markers)} accepted batch sample(s)"
                if witness["source"] == "evidence"
                else describe_latest(witness["latest"])
            ),
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
        execute=execute,
    )


VARIANTS = [
    _spec(
        "critical",
        "wrapper serves an ipmitool sensor table with two cr rows (CPU1 Temp, Fan1)",
        "bmc_critical_sensor=2 -> MARK_UNSCHEDULABLE + QUARANTINE, one incident whose "
        "sample labels, reasons and marker name both sensors with reading, status and "
        "thresholds; healthy table -> recovered; validated restore",
        critical,
        risk="M",
        max_minutes=32,
        cleanup="healthy table read, window closed, validated restore; node schedulable",
    ),
    _spec(
        "nonrecoverable",
        "wrapper serves an ipmitool sensor table with one nr row",
        "bmc_critical_sensor=1 -> MARK_UNSCHEDULABLE + QUARANTINE (no STOP, no hardware), "
        "the sensor named; healthy table -> recovered; validated restore",
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
        "wrapper makes ipmitool sensor exit 1 ('Get Device ID command failed'), then exit 1 "
        "('Could not open device at /dev/ipmi0 ...: No such file or directory')",
        "first a _bmc collection error (UNKNOWN), no bmc_critical_sensor sample, no finding, "
        "no isolation; then bmc_status_unsupported NO_DEVICE (read from the accepted "
        "batches or the product's latest record), the error stops, still no sample and "
        "no finding",
        command_failure,
        max_minutes=20,
    ),
    VariantSpec(
        case_id=CASE,
        variant="unsupported_bmc",
        proof_mode=OBSERVATION,
        lanes=("eks", "hp"),
        risk="L",
        flags="S",
        injection="none (read-only): survey ipmitool and watch the BMC contributor",
        expected="bmc_status_unsupported TOOL_MISSING (no ipmitool) or NO_DEVICE (ipmitool, "
        "no BMC) on the accepted batches or the product's latest record, no BMC reading, "
        "no _bmc collection error and no finding: the capability gap is recorded by the "
        "product (not a PASS of any BMC grading); a permanent _bmc error on a BMC-less "
        "node is a FAIL",
        cleanup="none",
        max_minutes=12,
        execute=unsupported_bmc,
    ),
]

__all__ = [
    "CASE",
    "CRITICAL_SENSORS",
    "CRITICAL_TABLE",
    "HEALTHY_TABLE",
    "IPMI_NO_DEVICE_STDERR",
    "IPMI_TRANSIENT_STDERR",
    "MARKER_LOOKBACK_TICKS",
    "METRIC",
    "METRICS",
    "NONRECOVERABLE_TABLE",
    "UNAVAILABLE_TABLE",
    "UNSUPPORTED",
    "VARIANTS",
    "WRONG_COLUMN_TABLE",
    "bmc_errors",
    "close_bmc_window",
    "command_failure_rules",
    "describe_latest",
    "latest_marker",
    "marker_reason",
    "marker_witness",
    "no_device_rules",
    "open_bmc_window",
    "require_served",
    "sensor_identity",
    "sensor_table",
    "served_calls",
    "table_rules",
    "unsupported_markers",
]
