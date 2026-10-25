"""GF-GENERAL-LIVE-007: SMART health through the product's own ``smartctl`` reader.

The host collector (``collectors/host/system_metrics.py`` ``_smart``) runs
``smartctl --scan-open`` every tick and ``smartctl -H -j <device>`` per drive
(verdicts cached 300 s) and reports one of three things per drive: a *verdict*
(``smart_health_failed`` 0/1; 1 is CRITICAL QUARANTINE in ``host_health.py``),
*UNKNOWN* (``smart_query_failed`` with ``failure_mode`` SCAN_FAILED /
OPEN_FAILED / QUERY_FAILED plus a ``_smart`` collection error, which grades
nothing and holds host validation) or *NOT_SUPPORTED* (``smart_status_unsupported``
with ``reason`` TOOL_MISSING / NO_DEVICES / NO_SMART_STATUS: no error, no
finding, never a passed drive).

The failing variants are ``live-replay`` on the shared host source replay
(``general_live_source_replay.py``, approval ``HOST_SOURCE_REPLAY``): the
reviewed pass-through wrapper goes first on the host collector unit's PATH and
rewrites only the plan's allow-listed field of the one query a variant is about
-- ``smart_status.passed`` (health_failed), the ``smart_status`` object
(unsupported_device), the ``-H -j`` output and exit status (query_failed,
malformed_json), the ``--scan-open`` exit status (scan_failed) -- or hides the
real ``smartctl`` from the unit alone (tool_missing). The drive is the product's
own: the device of its latest ``smart_health_failed`` reading. Parsing, grading,
the database and the QUARANTINE / validated-restore chain stay real; the
wrapper's journal (``replay.audit``) is the oracle that the product asked, and a
variant whose rule was never served is INCONCLUSIVE, not PASS. Every variant
closes its window and proves the real tool answers again before it ends;
``healthy`` is the read-only baseline. ``proof_scope`` is ``live-replay``.

A node without ``smartctl`` (the AL2023 EKS GPU AMI) keeps the six replay
variants NOT_RUN(CAPABILITY_ABSENT) unless ``TEMP_TEST_PACKAGE_INSTALL`` is
given; then the shared temporary-package step (``general_live_deps``) installs
exactly ``smartmontools`` and its reviewed dependency ``smartmontools-selinux``
(the same build's SELinux policy module, which AL2023's
``selinux-policy-targeted`` makes dnf pull) for this run before anything else
and removes both as the last LIFO cleanup, ``smartmontools`` first, after the
window closed. No collector restart is needed
for the install: ``_smart`` runs ``shutil.which("smartctl")`` on every tick and
the unit's ``ProtectSystem=strict`` is a read-only view of the live ``/usr``,
so the next tick scans and grades the drives; the runner then waits up to one
health summary plus a few ticks for the product's first ``smart_health_failed``
reading (``smart_devices(fresh=True)``) -- that device is the variant's drive.
The window itself restarts the unit on open and close as before, and after the
removal the product reports ``smart_status_unsupported`` TOOL_MISSING again.
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
from scripts.e2e.regional.general_live_deps import (
    PACKAGE_APPROVAL,
    dependency_summary,
    deps_deadman_seconds,
    ensure_dependencies,
)
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    VariantNotRun,
    utc_now,
)
from scripts.e2e.regional.general_live_registry import (
    HARDWARE_OPERATIONS,
    LIVE_REPLAY,
    OBSERVATION,
    VariantSpec,
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
from tools import host_command_replay as wrapper

CASE = "GF-GENERAL-LIVE-007"
TOOL = "smartctl"
HEALTH = "smart_health_failed"
QUERY_FAILED = "smart_query_failed"
UNSUPPORTED = "smart_status_unsupported"
SMART_METRICS = frozenset({HEALTH, QUERY_FAILED, UNSUPPORTED})
#: The product's collection-error prefix for the SMART contributor.
SMART_CONTRIBUTOR = "_smart"
QUARANTINE_BUDGET = 420
#: Ticks allowed for the first replayed reading: the unit restarts when the
#: window opens (startup spread + jitter) and its first tick is delivered as
#: ``baseline``, so the marker is normally in the first or second batch.
MARKER_TICKS = 8
SETTLE_TICKS = 2
#: Ticks past one health summary allowed for the product's first verdict after
#: ``smartctl`` was installed for this run: the collector finds the tool on its
#: next tick (no restart), a healthy verdict travels with the next summary.
FIRST_VERDICT_TICKS = 6
#: Journal lines read back for the oracle (the probe caps at 400): one scan per
#: tick plus one -H per drive per cache period, so this covers ~1.5 h.
AUDIT_TAIL = 400
ISOLATION_FORBIDDEN = HARDWARE_OPERATIONS | {"STOP_WORKLOADS"}
#: What a real ``smartctl -H -j`` prints when it cannot open the drive: a
#: verdict-less JSON document, ``exit_status`` 2 and the process exit 2.
FAILED_QUERY_EXIT = 2
#: A truncated ``-H -j`` document (exit 0): the parse, not the drive, failed.
MALFORMED_JSON = '{"json_format_version": [1, 0], "smartctl": {"exit_status": 0, "mess'
SCAN_FAILED_EXIT = 1
SCAN_FAILED_STDERR = "smartctl: unable to scan devices: Operation not permitted\n"


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def failed_query_json(device: str) -> str:
    """The JSON a real ``smartctl -H -j`` prints for a drive it cannot open."""

    return (
        json.dumps(
            {
                "json_format_version": [1, 0],
                "smartctl": {
                    "version": [7, 4],
                    "platform_info": "x86_64-linux",
                    "argv": [TOOL, "-H", "-j", device],
                    "exit_status": FAILED_QUERY_EXIT,
                    "messages": [
                        {
                            "string": (
                                f"Smartctl open device: {device} failed: No such device"
                            ),
                            "severity": "error",
                        }
                    ],
                },
            },
            indent=2,
        )
        + "\n"
    )


def health_failed_rules(device: str) -> list[dict[str, Any]]:
    return [
        rule(
            "health",
            "set_json",
            device=device,
            field="smart_status.passed",
            value=False,
        )
    ]


def query_failed_rules(device: str) -> list[dict[str, Any]]:
    return [
        rule(
            "health",
            "replace_output",
            device=device,
            stdout=failed_query_json(device),
            exit_code=FAILED_QUERY_EXIT,
        )
    ]


def malformed_json_rules(device: str) -> list[dict[str, Any]]:
    return [rule("health", "replace_output", device=device, stdout=MALFORMED_JSON)]


def scan_failed_rules() -> list[dict[str, Any]]:
    return [rule("scan", "exit", exit_code=SCAN_FAILED_EXIT, stderr=SCAN_FAILED_STDERR)]


def unsupported_device_rules(device: str) -> list[dict[str, Any]]:
    return [rule("health", "delete_json", device=device, field="smart_status")]


# --------------------------------------------------------------------------- #
# the tool, the product's drives, the window, the oracle
# --------------------------------------------------------------------------- #
def ensure_smartctl(ctx: VariantContext, survey: dict[str, Any]) -> dict[str, Any]:
    """``smartctl`` as the node has it, or ``smartmontools`` for this run only.

    The shared temporary-package step: NOT_RUN(CAPABILITY_ABSENT) naming the
    token without ``TEMP_TEST_PACKAGE_INSTALL``; with it the probe installs
    exactly ``smartmontools`` with its reviewed dependency
    ``smartmontools-selinux`` (a transaction pulling anything beyond that
    closure is refused, naming it) and their removal, ``smartmontools`` first,
    is the last LIFO cleanup, after the window closed. The package deadman is
    timed after the window's restore deadman.
    """

    return ensure_dependencies(
        ctx,
        survey,
        tools=(TOOL,),
        deadman_seconds=deps_deadman_seconds(ctx.spec.max_minutes * 60),
        evidence="smart-dependencies",
    )


def _verdict_devices(current: Mapping[str, Any]) -> tuple[list[str], list[Any]]:
    latest = [
        item
        for item in current.get("metric_latest") or []
        if item.get("name") in SMART_METRICS
    ]
    devices = sorted(
        {
            str(item["device"])
            for item in latest
            if item.get("name") == HEALTH
            and isinstance(item.get("device"), str)
            and wrapper.DEVICE_NODE.fullmatch(item["device"]) is not None
        }
    )
    return devices, latest


def smart_devices(ctx: VariantContext, *, fresh: bool = False) -> list[str]:
    """The drives the product grades now: its latest ``smart_health_failed`` devices.

    Read-only, from the product's own records, so the device a variant names in
    its plan is one the product's ``--scan-open`` lists and ``-H -j`` answers for
    on this node. No verdict for any drive means SMART is not a capability here
    (``smart_status_unsupported`` TOOL_MISSING / NO_DEVICES / NO_SMART_STATUS).

    ``fresh``: ``smartctl`` was installed for this run moments ago, so the
    product's first verdict is still to come -- the host collector finds the
    tool on its next tick (``shutil.which`` per tick, no restart) and a healthy
    verdict travels with the next health summary -- so wait up to one summary
    plus ``FIRST_VERDICT_TICKS`` ticks for it before deciding.
    """

    devices, latest = _verdict_devices(activity(ctx, utc_now(), metrics=SMART_METRICS))
    if not devices and fresh:

        def poll() -> list[str] | None:
            nonlocal devices, latest
            devices, latest = _verdict_devices(
                activity(ctx, utc_now(), metrics=SMART_METRICS)
            )
            return devices or None

        ctx.wait_until(
            poll,
            timeout_seconds=ctx.host_summary
            + ctx.host_interval * FIRST_VERDICT_TICKS
            + 60,
            poll_seconds=ctx.host_interval,
            name="smart-first-verdict",
        )
    ctx.evidence.write(
        "smart-baseline",
        {"devices": devices, "metric_latest": latest, "fresh_install": fresh},
    )
    if not devices:
        raise VariantNotRun(
            "CAPABILITY_ABSENT",
            "the product reports no drive with a SMART verdict on this node "
            + ("after smartctl was installed for this run " if fresh else "")
            + f"(latest SMART markers: {[item.get('labels') for item in latest]})",
        )
    return devices


def open_smart_window(
    ctx: VariantContext,
    rules: Sequence[Mapping[str, Any]] = (),
    *,
    hide_tool: bool = False,
) -> tuple[HostSourceReplay, dict[str, Any]]:
    """Wrap ``smartctl`` with ``rules`` (or hide it) inside the host collector unit.

    Reusable by any category that reads SMART through the product: the window is
    on the variant's LIFO cleanup stack and bounded by its ``max_minutes``.
    """

    replay = host_source_replay(ctx)
    replay.create()
    restore_seconds = ctx.spec.max_minutes * 60
    if hide_tool:
        opened = replay.open_window(
            inaccessible=[f"tool:{TOOL}"], restore_seconds=restore_seconds
        )
    else:
        plan = command_plan(TOOL, rules, not_after_seconds=restore_seconds)
        ctx.evidence.write("smart-plan", plan)
        replay.install_wrapper(TOOL, plan)
        opened = replay.open_command([TOOL], restore_seconds=restore_seconds)
    ctx.evidence.write("source-open", opened)
    return replay, opened


def served_calls(
    audit: Mapping[str, Any], query: str, device: str | None = None
) -> list[dict[str, Any]]:
    """Journal entries of ``query`` (for ``device``) on which a rule was applied."""

    return [
        entry
        for entry in audit.get("entries") or []
        if isinstance(entry, dict)
        and entry.get("query") == query
        and entry.get("applied_rules")
        and (device is None or device in (entry.get("argv") or []))
    ]


def require_served(
    ctx: VariantContext,
    replay: HostSourceReplay,
    query: str,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """The oracle: the product asked the wrapper and was served the rule.

    Read before the window closes (closing removes the journal). A variant the
    product never asked for, or whose rule never applied, proves nothing about
    the processing chain and is INCONCLUSIVE rather than PASS or FAIL.
    """

    audit = replay.audit(TOOL, tail=AUDIT_TAIL)
    ctx.evidence.write(f"{TOOL}-audit", audit)
    hits = served_calls(audit, query, device)
    ctx.evidence.append_oracle(hits, source=f"{TOOL}-audit")
    if not hits:
        asked = [
            {"query": entry.get("query"), "argv": entry.get("argv")}
            for entry in audit.get("entries") or []
            if isinstance(entry, dict)
        ][-12:]
        raise VariantInconclusive(
            f"the host collector never had the {TOOL} {query} rule served"
            + (f" for {device}" if device else "")
            + f"; catalogued calls seen: {asked}"
        )
    return hits


# --------------------------------------------------------------------------- #
# reading the product
# --------------------------------------------------------------------------- #
def smart_errors(observed: Mapping[str, Any]) -> list[str]:
    """The ``_smart`` collection errors of the accepted batches."""

    return [
        str(error)
        for batch in observed.get("batches") or []
        for error in batch.get("collection_errors") or []
        if str(error).startswith(SMART_CONTRIBUTOR)
    ]


def labelled(
    observed: Mapping[str, Any],
    metric: str,
    label: str,
    value: str,
    *,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Samples of ``metric`` whose ``labels[label] == value`` (on ``device``)."""

    return [
        sample
        for sample in samples(dict(observed), metric, device=device)
        if (sample.get("labels") or {}).get(label) == value
    ]


def await_smart(
    ctx: VariantContext,
    since: Any,
    accept: Callable[[dict[str, Any]], bool],
    *,
    name: str,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Poll the product's SMART records until ``accept`` holds (or the budget ends)."""

    last: dict[str, Any] = {}

    def poll() -> dict[str, Any] | None:
        nonlocal last
        last = activity(ctx, since, metrics=SMART_METRICS)
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
    """Wait for the marker, let the grading settle, then read once more."""

    await_smart(ctx, since, accept, name=name)
    ctx.hold(ctx.host_interval * SETTLE_TICKS + 30, name=f"{name}-settle")
    return activity(ctx, since, metrics=SMART_METRICS)


def real_tool_again(
    ctx: VariantContext, replay: HostSourceReplay, *, device: str | None = None
) -> list[str]:
    """Close the window and prove the real ``smartctl`` answers the product again.

    The close restarts the unit, so the first tick after it re-queries every
    drive through the real binary: a fresh ``smart_health_failed`` reading (for
    ``device`` when given) with no ``_smart`` collection error.
    """

    closed_at = utc_now()
    closed = replay.close()
    ctx.evidence.write("source-close", closed)
    errors: list[str] = []
    if not closed.get("clean", True):
        errors.append(f"the source window did not close clean: {closed.get('residue')}")

    def accept(current: dict[str, Any]) -> bool:
        return bool(samples(current, HEALTH, device=device))

    observed = await_smart(
        ctx,
        closed_at,
        accept,
        name="smart-real-again",
        timeout_seconds=ctx.host_summary + ctx.host_interval * 6,
    )
    fresh = samples(observed, HEALTH, device=device)
    if not fresh:
        errors.append(
            "no fresh smart_health_failed reading after the window closed"
            + (f" for {device}" if device else "")
        )
    elif any(sample.get("value") != 0 for sample in fresh):
        errors.append(
            "the real smartctl reports a failed drive after the window closed: "
            f"{[(s.get('device'), s.get('value')) for s in fresh][-4:]}"
        )
    if smart_errors(observed):
        errors.append(
            "the host collector still reports a _smart collection error after "
            f"the window closed: {smart_errors(observed)[-1][:160]}"
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


def _unknown_stages(
    observed: dict[str, Any],
    *,
    device: str | None,
    failure_mode: str,
    error_text: str,
) -> dict[str, list[str]]:
    """UNKNOWN is a marker plus an error, never a verdict and never a finding."""

    marker = labelled(
        observed, QUERY_FAILED, "failure_mode", failure_mode, device=device
    )
    errors = smart_errors(observed)
    verdicts = samples(observed, HEALTH, device=device)
    return {
        "unknown_marker": []
        if marker
        else [
            f"no smart_query_failed sample with failure_mode={failure_mode}"
            + (f" for {device}" if device else "")
        ],
        "collection_error": []
        if any(error_text in error for error in errors)
        else [
            f"no _smart collection error naming {error_text!r}; errors seen: "
            f"{[error[:120] for error in errors][-3:]}"
        ],
        "no_verdict": [
            f"a verdict smart_health_failed={sample.get('value')} was emitted for "
            f"{sample.get('device')} while its SMART reading was unknown"
            for sample in verdicts
        ],
        "no_finding": chain_errors(
            observed, ChainExpectation(metric=HEALTH, device=device, present=False)
        ),
        "hardware": hardware_errors(observed),
    }


def _other_drives_kept(
    observed: dict[str, Any], devices: Iterable[str], affected: str
) -> list[str]:
    """The other drives' verdicts survive one drive's UNKNOWN (``exc.samples``)."""

    others = [item for item in devices if item != affected]
    if not others:
        return []
    missing = [item for item in others if not samples(observed, HEALTH, device=item)]
    return (
        [f"the verdicts of the other drives were dropped: {missing}"] if missing else []
    )


# --------------------------------------------------------------------------- #
# variants
# --------------------------------------------------------------------------- #
def smart_healthy(ctx: VariantContext) -> dict[str, Any]:
    """The node's real ``smartctl`` grades the drives healthy -> no finding.

    No injection and no replay: the host collector already runs ``smartctl``
    every cycle, so this reads the product's own records across a window and
    checks that ``smart_health_failed`` minted no storage finding and that any
    latest ``smart_health_failed`` value is 0. It is the negative baseline the
    failing SMART variants are judged against.
    """

    run_survey(ctx)
    started_at = utc_now()
    ctx.hold(ctx.host_summary + ctx.host_interval * 3, name="smart-healthy-hold")
    observed = activity(ctx, started_at, metrics={HEALTH})
    smart_samples = samples(observed, HEALTH)
    latest = [
        item
        for item in observed.get("metric_latest") or []
        if item.get("name") == HEALTH
    ]
    stages = {
        "no_finding": chain_errors(
            observed, ChainExpectation(metric=HEALTH, present=False)
        ),
        "hardware": hardware_errors(observed),
        "healthy_values": []
        if all(
            not isinstance(item.get("value"), (int, float)) or item["value"] == 0
            for item in [*smart_samples, *latest]
        )
        else ["a smart_health_failed reading was non-zero on a healthy node"],
    }
    return _result(
        stages,
        notes=["read-only baseline for the failing SMART variants"],
        extra={"smart_latest": latest},
    )


def health_failed(ctx: VariantContext) -> dict[str, Any]:
    """Live-replay: ``smart_status.passed=false`` on one drive -> QUARANTINE -> restore."""

    survey = run_survey(ctx)
    dependencies = ensure_smartctl(ctx, survey)
    device = smart_devices(ctx, fresh=dependencies["changed"])[0]
    replay, _opened = open_smart_window(ctx, health_failed_rules(device))
    started = utc_now()
    # smart_health_failed is an immediate CRITICAL rule and the tick that reads
    # it opens a ``threshold:smart_health_failed`` edge delivered at once.
    detection = wait_for_incident(
        ctx,
        started,
        HEALTH,
        device=device,
        timeout_seconds=QUARANTINE_BUDGET,
        metrics=SMART_METRICS,
        name="smart-quarantine",
    )
    require_served(ctx, replay, "health", device)
    incidents = finding_incidents(detection, HEALTH, device=device)
    settled = (
        wait_settled(
            ctx,
            started,
            [item["incident_id"] for item in incidents],
            timeout_seconds=QUARANTINE_BUDGET,
            metrics=SMART_METRICS,
            name="smart-settled",
        )
        if incidents
        else detection
    )
    stages = {
        "chain": chain_errors(
            settled,
            ChainExpectation(
                metric=HEALTH,
                device=device,
                operations=("MARK_UNSCHEDULABLE", "QUARANTINE"),
                forbidden=ISOLATION_FORBIDDEN,
                incident_states=frozenset({"QUARANTINED"}),
            ),
        ),
        "verdict_not_unknown": [
            f"the failed verdict was also reported unknown ({sample.get('labels')})"
            for sample in samples(settled, QUERY_FAILED, device=device)
        ],
    }
    # Recovery order: the real tool first (close the window and let the product
    # read the drive healthy again), then the product's validated restore. The
    # recovery is judged from the close: the first tick after the restart is
    # the fresh healthy reading and clears the drive's health signal.
    closed_at = utc_now()
    stages["real_tool_again"] = real_tool_again(ctx, replay, device=device)
    recovered = await_metric_recovery(
        ctx, HEALTH, 1.0, device=device, since=closed_at, name="smart-recovered"
    )
    stages["recovered"] = recovered["errors"]
    restore_incidents(
        ctx,
        finding_incidents(settled, HEALTH, device=device),
        reason="GF-GENERAL-LIVE-007 health_failed restore",
    )
    return _result(
        stages,
        notes=[
            f"smart_status.passed=false replayed for {device} through the real "
            "smartctl JSON parser; MARK_UNSCHEDULABLE + QUARANTINE, then the "
            "validated restore"
        ],
        extra={
            "device": device,
            "recovery": recovered,
            "dependencies": dependency_summary(dependencies),
            **replay.evidence(tool=TOOL),
        },
    )


def _unknown_variant(
    ctx: VariantContext,
    *,
    rules: Callable[[str], list[dict[str, Any]]],
    failure_mode: str,
    error_text: str,
    name: str,
    note: str,
) -> dict[str, Any]:
    """One drive's ``-H -j`` answer replaced -> UNKNOWN for that drive only."""

    survey = run_survey(ctx)
    dependencies = ensure_smartctl(ctx, survey)
    devices = smart_devices(ctx, fresh=dependencies["changed"])
    device = devices[0]
    replay, _opened = open_smart_window(ctx, rules(device))
    started = utc_now()
    observed = settled_read(
        ctx,
        started,
        lambda current: bool(samples(current, QUERY_FAILED, device=device)),
        name,
    )
    require_served(ctx, replay, "health", device)
    stages = _unknown_stages(
        observed, device=device, failure_mode=failure_mode, error_text=error_text
    )
    stages["other_drives_kept"] = _other_drives_kept(observed, devices, device)
    stages["not_isolated"] = not_isolated(ctx, f"an unknown SMART reading of {device}")
    stages["real_tool_again"] = real_tool_again(ctx, replay, device=device)
    return _result(
        stages,
        notes=[note],
        extra={
            "device": device,
            "collection_errors": smart_errors(observed),
            "dependencies": dependency_summary(dependencies),
            **replay.evidence(tool=TOOL),
        },
    )


def query_failed(ctx: VariantContext) -> dict[str, Any]:
    """Live-replay: a verdict-less ``-H -j`` with exit 2 is UNKNOWN, never a zero."""

    return _unknown_variant(
        ctx,
        rules=query_failed_rules,
        failure_mode="QUERY_FAILED",
        error_text="no verdict, exit status",
        name="smart-query-failed",
        note=(
            "a failed drive query (smartctl exit 2, no smart_status) is "
            "smart_query_failed + a _smart collection error, not a verdict"
        ),
    )


def malformed_json(ctx: VariantContext) -> dict[str, Any]:
    """Live-replay: corrupt ``-H -j`` JSON is UNKNOWN, not a parsed healthy result."""

    return _unknown_variant(
        ctx,
        rules=malformed_json_rules,
        failure_mode="QUERY_FAILED",
        error_text="output is not JSON",
        name="smart-malformed",
        note=(
            "truncated smartctl JSON is smart_query_failed + a _smart collection "
            "error, not a verdict"
        ),
    )


def scan_failed(ctx: VariantContext) -> dict[str, Any]:
    """Live-replay: a non-zero ``--scan-open`` leaves the drive set UNKNOWN, not empty."""

    survey = run_survey(ctx)
    dependencies = ensure_smartctl(ctx, survey)
    smart_devices(ctx, fresh=dependencies["changed"])
    replay, _opened = open_smart_window(ctx, scan_failed_rules())
    started = utc_now()
    observed = settled_read(
        ctx,
        started,
        lambda current: bool(
            labelled(current, QUERY_FAILED, "failure_mode", "SCAN_FAILED")
        ),
        "smart-scan-failed",
    )
    require_served(ctx, replay, "scan")
    stages = _unknown_stages(
        observed,
        device=None,
        failure_mode="SCAN_FAILED",
        error_text="--scan-open failed",
    )
    # The historical gap the spec names: a failed scan parsed as "no drives".
    stages["not_no_drives"] = [
        f"a failed scan was reported as {UNSUPPORTED} reason={sample.get('labels')}"
        for sample in labelled(observed, UNSUPPORTED, "reason", "NO_DEVICES")
    ]
    stages["not_isolated"] = not_isolated(ctx, "an unknown SMART drive set")
    stages["real_tool_again"] = real_tool_again(ctx, replay)
    return _result(
        stages,
        notes=[
            "smartctl --scan-open exiting non-zero is smart_query_failed "
            "(SCAN_FAILED) + a _smart collection error: the drive set is unknown, "
            "not empty, and no cached verdict is reported"
        ],
        extra={
            "collection_errors": smart_errors(observed),
            "dependencies": dependency_summary(dependencies),
            **replay.evidence(tool=TOOL),
        },
    )


def tool_missing(ctx: VariantContext) -> dict[str, Any]:
    """Live-replay: no ``smartctl`` inside the unit -> NOT_SUPPORTED, no error, no finding."""

    survey = run_survey(ctx)
    dependencies = ensure_smartctl(ctx, survey)
    smart_devices(ctx, fresh=dependencies["changed"])
    replay, opened = open_smart_window(ctx, hide_tool=True)
    started = utc_now()
    observed = settled_read(
        ctx,
        started,
        lambda current: bool(
            labelled(current, UNSUPPORTED, "reason", "TOOL_MISSING")
            or samples(current, HEALTH)
        ),
        "smart-tool-missing",
    )
    marker = labelled(observed, UNSUPPORTED, "reason", "TOOL_MISSING")
    verdicts = samples(observed, HEALTH)
    if not marker and verdicts:
        raise VariantInconclusive(
            "the host collector still found smartctl inside the unit "
            f"(verification: {opened.get('verification')})"
        )
    stages = {
        "unsupported_marker": []
        if marker
        else ["no smart_status_unsupported sample with reason=TOOL_MISSING"],
        "no_error": [
            f"a missing tool was graded as a collection error: {error[:120]}"
            for error in smart_errors(observed)
        ],
        "no_verdict": [
            f"{sample.get('name')}={sample.get('value')} on {sample.get('device')} "
            "was emitted without a smartctl"
            for sample in [*verdicts, *samples(observed, QUERY_FAILED)]
        ],
        "no_finding": chain_errors(
            observed, ChainExpectation(metric=HEALTH, present=False)
        ),
        "hardware": hardware_errors(observed),
        "not_isolated": not_isolated(ctx, "a missing smartctl"),
        "real_tool_again": real_tool_again(ctx, replay),
    }
    return _result(
        stages,
        notes=[
            "smartctl hidden from the host collector unit only (InaccessiblePaths): "
            "smart_status_unsupported TOOL_MISSING, no collection error, no finding"
        ],
        extra={
            "verification": opened.get("verification"),
            "dependencies": dependency_summary(dependencies),
            **replay.evidence(tool=TOOL),
        },
    )


def unsupported_device(ctx: VariantContext) -> dict[str, Any]:
    """Live-replay: a drive without ``smart_status`` is NOT_SUPPORTED, not failed, not healthy."""

    survey = run_survey(ctx)
    dependencies = ensure_smartctl(ctx, survey)
    devices = smart_devices(ctx, fresh=dependencies["changed"])
    device = devices[0]
    replay, _opened = open_smart_window(ctx, unsupported_device_rules(device))
    started = utc_now()
    observed = settled_read(
        ctx,
        started,
        lambda current: bool(
            labelled(current, UNSUPPORTED, "reason", "NO_SMART_STATUS", device=device)
        ),
        "smart-unsupported-device",
    )
    require_served(ctx, replay, "health", device)
    stages = {
        "unsupported_marker": []
        if labelled(observed, UNSUPPORTED, "reason", "NO_SMART_STATUS", device=device)
        else [f"no smart_status_unsupported NO_SMART_STATUS sample for {device}"],
        "not_failed": [
            f"a verdict-less drive was reported as a failed query: {sample.get('labels')}"
            for sample in samples(observed, QUERY_FAILED, device=device)
        ],
        "not_healthy": [
            f"a verdict-less drive was reported as a passed scan "
            f"(smart_health_failed={sample.get('value')})"
            for sample in samples(observed, HEALTH, device=device)
        ],
        "no_error": [
            f"a capability gap was graded as a collection error: {error[:120]}"
            for error in smart_errors(observed)
        ],
        "other_drives_kept": _other_drives_kept(observed, devices, device),
        "no_finding": chain_errors(
            observed, ChainExpectation(metric=HEALTH, present=False)
        ),
        "hardware": hardware_errors(observed),
        "not_isolated": not_isolated(ctx, f"a SMART-less drive {device}"),
        "real_tool_again": real_tool_again(ctx, replay, device=device),
    }
    return _result(
        stages,
        notes=[
            f"smart_status removed from {device}'s real -H -j answer: "
            "smart_status_unsupported NO_SMART_STATUS, no error, no finding, "
            "never a passed drive"
        ],
        extra={
            "device": device,
            "dependencies": dependency_summary(dependencies),
            **replay.evidence(tool=TOOL),
        },
    )


def _spec(
    variant: str,
    injection: str,
    expected: str,
    execute: Callable[[VariantContext], dict[str, Any]],
    *,
    risk: str = "L",
    max_minutes: int = 16,
    cleanup: str = "close the smartctl window (the real tool answers again); confirm clean",
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
        # Only unlocks the temporary smartmontools install on a node without
        # smartctl; the variant runs without it wherever the tool exists.
        optional_approvals=(PACKAGE_APPROVAL,),
        execute=execute,
    )


VARIANTS = [
    _spec(
        "health_failed",
        "wrapper rewrites smart_status.passed=false in the real smartctl -H -j of one drive",
        "smart_health_failed=1 -> MARK_UNSCHEDULABLE + QUARANTINE (no STOP, no hardware); "
        "real tool again -> recovered; validated restore",
        health_failed,
        risk="M",
        max_minutes=32,
        cleanup="close the window, fresh healthy verdict, validated restore; node schedulable",
    ),
    _spec(
        "query_failed",
        "wrapper answers one drive's -H -j with a verdict-less JSON and exit 2",
        "smart_query_failed QUERY_FAILED + _smart collection error; no verdict for the drive, "
        "other drives kept; no finding, no isolation",
        query_failed,
    ),
    _spec(
        "scan_failed",
        "wrapper makes smartctl --scan-open exit 1",
        "smart_query_failed SCAN_FAILED + _smart collection error; no NO_DEVICES marker, "
        "no verdicts, no finding (the spec's scan exit-code gap)",
        scan_failed,
    ),
    _spec(
        "malformed_json",
        "wrapper answers one drive's -H -j with truncated JSON (exit 0)",
        "smart_query_failed QUERY_FAILED + 'output is not JSON' error; no verdict, no finding",
        malformed_json,
    ),
    _spec(
        "tool_missing",
        "smartctl made inaccessible inside the host collector unit only",
        "smart_status_unsupported TOOL_MISSING; no collection error, no SMART samples, no finding",
        tool_missing,
    ),
    _spec(
        "unsupported_device",
        "wrapper deletes smart_status from one drive's real -H -j answer",
        "smart_status_unsupported NO_SMART_STATUS for the drive; not failed, never a passed "
        "scan, other drives kept, no finding",
        unsupported_device,
    ),
    VariantSpec(
        case_id=CASE,
        variant="healthy",
        proof_mode=OBSERVATION,
        lanes=("eks", "hp"),
        risk="L",
        flags="S",
        injection="none (read-only): observe the node's real SMART grading",
        expected="a healthy smartctl mints no storage finding and no smart_health_failed > 0",
        cleanup="none",
        max_minutes=10,
        execute=smart_healthy,
    ),
]

__all__ = [
    "AUDIT_TAIL",
    "CASE",
    "FIRST_VERDICT_TICKS",
    "SMART_METRICS",
    "VARIANTS",
    "ensure_smartctl",
    "failed_query_json",
    "health_failed_rules",
    "malformed_json_rules",
    "open_smart_window",
    "query_failed_rules",
    "real_tool_again",
    "require_served",
    "scan_failed_rules",
    "served_calls",
    "smart_devices",
    "smart_errors",
    "unsupported_device_rules",
]
