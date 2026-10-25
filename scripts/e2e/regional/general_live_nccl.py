"""GF-GENERAL-LIVE-011 EFA traffic stall / NCCL hung.

HyperPod lane only: a hung collective needs at least two nodes with EFA/RDMA
between them, which the single plain-EKS node cannot provide (intra-node NCCL
uses NVLink, not EFA). A disposable, owned two-node managed NCCL test job
(``general_live_nccl_job``: a 1.5 MiB all-reduce every 2 s, one rank entering an
extra collective on the runner's pause file while the others hold before it,
bounded by an in-Pod watchdog) is brought up healthy, then stalled so its
collective genuinely stops while the job stays alive and its EFA RX/TX pair
falls to zero. The product's ``SITE_EFA_TRAFFIC`` rule reads that -- never the
pause -- as ``ZERO_WARNING`` (60 s) then ``HUNG_SUSPECTED`` (600 s), merges the
stall wave into one grouped incident, and compiles the read-only diagnostic DAG
``FREEZE_EVIDENCE -> COLLECT_HUNG_SUMMARY x2 -> COLLECT_HUNG_TRIAGE x2 ->
COLLECT_DIAGNOSTIC_BUNDLE -> VALIDATE_FABRIC -> ESCALATE_SUPPORT`` that ends
ESCALATED. The verdict reads that workflow's plan and step executions, the
per-node ``ZERO_WARNING`` incidents and the detection times, requires that
nothing cordoned, stopped, reset or rebooted on suspicion, and that the job's
traffic resumes once the runner releases the stall.

The effectiveness oracle is the runner's own raw EFA counter pair, read from
``/sys/class/infiniband`` by the node probe's ``efa-traffic`` command on both
attempt nodes every 30 s: before the stall every interval must run at or above
the product's active threshold (``GPU_FAULT_EFA_TRAFFIC_MIN_ACTIVE_BPS``,
1 MiB/s) for at least three intervals, and through the stall every interval
must stay at or below its zero threshold (``GPU_FAULT_EFA_TRAFFIC_ZERO_BPS``,
1 B/s) for at least the 600 s hung threshold, with both Pods alive and the
trainer reporting the stall. Without either the variant is INCONCLUSIVE, never
FAIL: a job that never went active, or a pause that kept communicating, is not
a triggered traffic rule. ``repeat_after_recovery`` stalls the job twice; in
between its pair must stay active past the product's rearm window (a stall
episode closes only after three samples over 120 s, which the host collector
sends every 30 s for 180 s once traffic is back) or it is INCONCLUSIVE, and the
second stall must open its own incident and workflow, chained to the first's
(``hung_episode.py``). The deep triage/forensics variants (per-rank FR
temporal completion, the second-dump-missing and whole-group cases, py-spy
failure classification, role and thread/CPU comparison) need a reviewed test
flight-recorder dump responder and a py-spy fault tool that are not wired this
phase, and the 128-node/1024-process scale variants need a cluster this site
does not have: each stays NOT_RUN with its reason, in the denominator.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

from gpu_fault.hung_summary_codec import decode_summary
from scripts.e2e.regional.general_live_common import run_survey
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    run_identity,
)
from scripts.e2e.regional.general_live_nccl_job import MIN_ACTIVE_BPS, ManagedNcclJob
from scripts.e2e.regional.general_live_registry import (
    HARDWARE_OPERATIONS,
    LIVE_REPLAY,
    NATIVE,
    VariantSpec,
    deferred_variant,
)
from scripts.e2e.regional.general_live_verdicts import operations, parse_time, verdict

CASE = "GF-GENERAL-LIVE-011"
ZERO_WARNING_SECONDS = 60
ZERO_HUNG_SECONDS = 600
STARTUP_GRACE = 120
HUNG_BUDGET = ZERO_HUNG_SECONDS + 600 + 180
#: The product's zero threshold (``MIN_ACTIVE_BPS`` is the active one); the
#: node's collector environment overrides either.
ZERO_BPS = 1
#: The spec keeps at least three normal samples before an injection.
BASELINE_INTERVALS = 3
#: The step in flight finishes and every rank reaches the stall well inside it.
STALL_SETTLE_SECONDS = 30
RESUME_SECONDS = 60
#: Skew allowed between the runner's clock and the control plane's.
CLOCK_TOLERANCE_SECONDS = 5
#: How long the runner keeps a job stalled at most: the settle, the drop window
#: at the default 300 s health summary, the hung wait. The in-Pod watchdog
#: (``general_live_nccl_job.STALL_MAX_SECONDS``) must outlast it.
RUNNER_STALL_SECONDS = STALL_SETTLE_SECONDS + ZERO_HUNG_SECONDS + 300 + HUNG_BUDGET
#: A stall episode closes, and a later stall can be a new one, only after
#: sustained recovery (``efa_stall_episode.EfaRecoveryPolicy``): three delivered
#: samples at or above its recovery threshold over 120 s, at most 60 s apart.
#: The host collector delivers one every min(30, gap / 2) s for max(120 + 2
#: spacings, 3 spacings) = 180 s once the pair is back over its noise floor
#: (``collectors/host/efa_recovery_sampling.py``); later ones come 300 s apart.
RECOVERY_SECONDS = 120
RECOVERY_MIN_SAMPLES = 3
RECOVERY_MAX_GAP_SECONDS = 60
#: The recovery threshold without a trusted pre-stall baseline; with one it is a
#: quarter of that baseline, under the active threshold for this job (< 4x it).
RECOVERY_FALLBACK_BPS = 1048576
#: The step a trainer line is about (``STALL step=``, ``RESUMED step=``).
STEP = re.compile(r"\bstep=(\d+)")
#: Incident states the hung chain does not leave on its own.
SETTLED_STATES = frozenset({"ESCALATED", "RECOVERED", "QUARANTINED"})
HUNG_SUFFIX = "-efa-traffic-hung_suspected"
WARNING_SUFFIX = "-efa-traffic-zero_warning"
#: ``<batch id>-efa-traffic-<signal>`` where the host batch id is
#: ``host-<node>-<microseconds>`` (``collectors/host/collector.py``).
CREATOR = re.compile(r"^host-(?P<node>.+)-\d+-efa-traffic-[a-z_]+$")
#: The reason a peer node's hung finding leaves on the merged incident
#: (``orchestration/families/grouped_health.py``).
CONCURRENT = ": concurrent hung evidence: "
#: The read-only diagnostic DAG a hung suspicion must compile to, in order.
DIAGNOSTIC_DAG = (
    "FREEZE_EVIDENCE",
    "COLLECT_HUNG_SUMMARY",
    "COLLECT_HUNG_SUMMARY",
    "COLLECT_HUNG_TRIAGE",
    "COLLECT_HUNG_TRIAGE",
    "COLLECT_DIAGNOSTIC_BUNDLE",
    "VALIDATE_FABRIC",
    "ESCALATE_SUPPORT",
)
#: A hung chain is diagnostic-only: no hardware action, no node isolation, no
#: workload stop or restart may ever appear in it.
FORBIDDEN = HARDWARE_OPERATIONS | {
    "MARK_UNSCHEDULABLE",
    "QUARANTINE",
    "STOP_WORKLOADS",
    "RESTART_WORKLOAD",
}
#: What triage may report for a rank whose job configured no flight recorder.
FR_OFF_STATUSES = frozenset({"not_configured", "BUDGET_EXHAUSTED", None})

#: Read-only: this attempt's SITE_EFA_TRAFFIC incidents on either attempt node
#: (the grouped hung incident and the per-node warnings), their workflows
#: (official steps, step executions with their details). Never writes.
HUNG_PROBE = r"""
import json
import sys
from datetime import datetime

from gpu_fault.app import ApplicationContext
from gpu_fault.models import IncidentState
from gpu_fault.store import NotFoundError

cluster_id, node_list, job_id, attempt_id, since_text = sys.argv[1:6]
node_ids = {item for item in node_list.split(",") if item}
since = datetime.fromisoformat(since_text.replace("Z", "+00:00"))
store = ApplicationContext.from_environment().store

incidents = []
workflows = {}
for incident in store.list_incidents_by_state(
    cluster_id, list(IncidentState), node_ids=node_ids
):
    if incident.policy_source != "SITE_EFA_TRAFFIC":
        continue
    if incident.attempt_id != attempt_id:
        continue
    if incident.created_at < since and incident.updated_at < since:
        continue
    incidents.append({
        "incident_id": incident.incident_id,
        "event_id": incident.event_id,
        "event_type": incident.event_type,
        "state": incident.state.value,
        "attempt_id": incident.attempt_id,
        "job_id": incident.job_id,
        "node_ids": list(incident.node_ids),
        "reasons": list(incident.reasons or []),
        "workflow_request_id": incident.workflow_request_id,
        "created_at": incident.created_at.isoformat(),
        "updated_at": incident.updated_at.isoformat(),
    })
    if incident.workflow_request_id and incident.workflow_request_id not in workflows:
        try:
            workflows[incident.workflow_request_id] = store.get_workflow(
                incident.workflow_request_id
            ).model_dump(mode="json")
        except NotFoundError:
            pass

print(json.dumps({
    "incidents": incidents,
    "workflows": workflows,
}, sort_keys=True, default=str))
"""


# --------------------------------------------------------------------------- #
# EFA counter oracle (pure)
# --------------------------------------------------------------------------- #
def _usable(reading: Mapping[str, Any]) -> tuple[float, int] | None:
    """``(observed_at, pair bytes)`` of a complete probe reading, else None."""

    observed = reading.get("observed_at")
    total = reading.get("pair_bytes_total")
    ports = reading.get("port_count")
    if (
        reading.get("error")
        or reading.get("complete_pairs") is not True
        or type(ports) is not int
        or ports < 1
        or not isinstance(observed, (int, float))
        or not math.isfinite(observed)
        or type(total) is not int
        or total < 0
    ):
        return None
    return float(observed), total


def efa_rates(readings: Sequence[Mapping[str, Any]]) -> list[float | None]:
    """RX+TX bytes per second between consecutive readings of one node.

    ``None`` marks an interval that proves nothing: an incomplete counter pair
    or a probe error at either end, no elapsed time, or a counter that went
    backwards (a reset).
    """

    rates: list[float | None] = []
    for before, after in zip(readings, readings[1:]):
        first, second = _usable(before), _usable(after)
        if first is None or second is None:
            rates.append(None)
            continue
        elapsed = second[0] - first[0]
        moved = second[1] - first[1]
        rates.append(moved / elapsed if elapsed > 0 and moved >= 0 else None)
    return rates


def _unusable(node: str, rates: list[float | None]) -> list[str]:
    missing = sum(rate is None for rate in rates)
    if not missing:
        return []
    return [
        f"{node}: {missing} of {len(rates)} EFA counter interval(s) unusable "
        "(incomplete pair, probe error or counter reset)"
    ]


def baseline_errors(
    window: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    minimum_bps: float,
    active: bool = True,
    minimum_intervals: int = BASELINE_INTERVALS,
) -> list[str]:
    """Why ``window`` does not prove the pre-stall baseline on every node.

    ``active``: every interval at or above the product's active threshold (the
    attempt sets ``had_active``). ``active=False`` (always-low): every interval
    below it, so the attempt never could.
    """

    errors: list[str] = []
    for node, readings in window.items():
        rates = efa_rates(readings)
        usable = [rate for rate in rates if rate is not None]
        errors.extend(_unusable(node, rates))
        if len(usable) < minimum_intervals:
            errors.append(
                f"{node}: {len(usable)} usable EFA interval(s), need {minimum_intervals}"
            )
        if not usable:
            continue
        if active and min(usable) < minimum_bps:
            errors.append(
                f"{node}: the EFA pair fell to {min(usable):.0f} B/s, under the "
                f"product's {minimum_bps:.0f} B/s active threshold"
            )
        if not active and max(usable) >= minimum_bps:
            errors.append(
                f"{node}: the EFA pair reached {max(usable):.0f} B/s, at or over the "
                f"product's {minimum_bps:.0f} B/s active threshold"
            )
    return errors


def drop_errors(
    window: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    zero_bps: float,
    minimum_seconds: float = ZERO_HUNG_SECONDS,
) -> list[str]:
    """Why ``window`` does not prove the pair stayed at zero on every node.

    Every interval at or below the product's zero threshold, over at least the
    hung threshold: the product needs that long a zero run to suspect a hang.
    """

    errors: list[str] = []
    for node, readings in window.items():
        rates = efa_rates(readings)
        usable = [rate for rate in rates if rate is not None]
        errors.extend(_unusable(node, rates))
        times = [item[0] for item in map(_usable, readings) if item is not None]
        span = times[-1] - times[0] if len(times) > 1 else 0.0
        if span < minimum_seconds:
            errors.append(
                f"{node}: the stall window covers {span:.0f} s, under the "
                f"{minimum_seconds:.0f} s hung threshold"
            )
        if usable and max(usable) > zero_bps:
            errors.append(
                f"{node}: the EFA pair kept moving at up to {max(usable):.1f} B/s "
                f"during the stall (the product's zero is at most {zero_bps:g} B/s)"
            )
    return errors


def resume_errors(
    window: Mapping[str, Sequence[Mapping[str, Any]]], *, floor_bps: float
) -> list[str]:
    """Why the job's traffic did not come back on every node after the release."""

    errors: list[str] = []
    for node, readings in window.items():
        rates = efa_rates(readings)
        last = rates[-1] if rates else None
        if last is None:
            errors.append(f"{node}: no usable EFA interval after the release")
        elif last < floor_bps:
            errors.append(
                f"{node}: the EFA pair moved {last:.0f} B/s after the release, "
                f"under {floor_bps:.0f} B/s; the job did not resume"
            )
    return errors


def stall_event_errors(events: Mapping[str, Sequence[str]], mode: str) -> list[str]:
    """Why the trainer's own account does not show the stall it was asked for.

    One rank entered the extra collective; every Pod has held ranks in the
    requested mode; no side activity failed and nothing resumed yet.
    """

    lines = [line for items in events.values() for line in items]
    errors: list[str] = []
    if not any("entered=extra_all_reduce" in line for line in lines):
        errors.append("no rank reports entering the extra collective")
    for pod, items in sorted(events.items()):
        if not any(f"held=before_extra_all_reduce mode={mode}" in i for i in items):
            errors.append(f"{pod}: no rank reports holding before it in mode {mode}")
    failed = [line for line in lines if "STALL-ACTIVITY-ERROR" in line]
    if failed:
        errors.append(f"the {mode} activity failed during the stall: {failed[0]}")
    resumed = [line for line in lines if "RESUMED step=" in line]
    if resumed:
        errors.append(f"the stall ended before the product's window: {resumed[0]}")
    return errors


def stall_step(events: Mapping[str, Sequence[str]]) -> int | None:
    """The step the trainer last reports stalling at."""

    steps = [
        int(match.group(1))
        for lines in events.values()
        for line in lines
        if "STALL step=" in line and (match := STEP.search(line)) is not None
    ]
    return max(steps, default=None)


def events_after(
    events: Mapping[str, Sequence[str]], step: int
) -> dict[str, list[str]]:
    """Each Pod's trainer lines of a later stall: about a step after ``step``.

    The log still holds the earlier stall and its ``RESUMED`` lines; a line
    that names no step (an activity error) is kept.
    """

    def later(line: str) -> bool:
        match = STEP.search(line)
        return match is None or int(match.group(1)) > step

    return {
        pod: [line for line in lines if later(line)] for pod, lines in events.items()
    }


# --------------------------------------------------------------------------- #
# Product records (pure)
# --------------------------------------------------------------------------- #
def _signal_incidents(records: dict[str, Any], suffix: str) -> list[dict[str, Any]]:
    return sorted(
        (
            item
            for item in records.get("incidents") or []
            if str(item.get("event_id", "")).endswith(suffix)
        ),
        key=lambda item: str(item.get("created_at") or ""),
    )


def hung_incident(records: dict[str, Any]) -> dict[str, Any] | None:
    hung = _signal_incidents(records, HUNG_SUFFIX)
    return hung[0] if hung else None


def episode(
    records: dict[str, Any], started: datetime, until: datetime | None = None
) -> dict[str, Any]:
    """``records`` with only the incidents one stall raised.

    Created from ``started`` and, with ``until``, before the next stall (both
    less the clock tolerance); the workflows stay as read.
    """

    slack = timedelta(seconds=CLOCK_TOLERANCE_SECONDS)

    def raised(item: Mapping[str, Any]) -> bool:
        created = parse_time(item.get("created_at"))
        return (
            created is not None
            and created >= started - slack
            and (until is None or created < until - slack)
        )

    incidents = [item for item in records.get("incidents") or [] if raised(item)]
    return {**records, "incidents": incidents}


def _hung_workflow(records: dict[str, Any]) -> dict[str, Any]:
    incident = hung_incident(records) or {}
    return (records.get("workflows") or {}).get(
        str(incident.get("workflow_request_id"))
    ) or {}


def dag_errors(records: dict[str, Any]) -> list[str]:
    if hung_incident(records) is None:
        return ["no EFA hung_suspected finding reached an incident"]
    ops = operations(_hung_workflow(records))
    errors: list[str] = []
    if ops != list(DIAGNOSTIC_DAG):
        errors.append(
            f"hung workflow steps {ops} != the diagnostic DAG {list(DIAGNOSTIC_DAG)}"
        )
    planned = sorted(set(ops) & FORBIDDEN)
    if planned:
        errors.append(f"hung workflow plans forbidden {planned}")
    return errors


def execution_errors(records: dict[str, Any]) -> list[str]:
    """Every planned step really ran and succeeded, not just the plan."""

    if hung_incident(records) is None:
        return []
    workflow = _hung_workflow(records)
    by_index: dict[int, list[dict[str, Any]]] = {}
    for item in workflow.get("step_executions") or []:
        if type(item.get("step_index")) is int:
            by_index.setdefault(item["step_index"], []).append(item)
    errors = []
    for index, step in enumerate(workflow.get("official_steps") or []):
        operation = step.get("operation")
        runs = sorted(by_index.get(index, []), key=lambda e: str(e.get("updated_at")))
        if not runs:
            errors.append(f"step {index} {operation} never executed")
        elif runs[-1].get("status") != "SUCCEEDED":
            errors.append(
                f"step {index} {operation} ended {runs[-1].get('status')}: "
                f"{runs[-1].get('error')}"
            )
    return errors


def escalated_errors(records: dict[str, Any]) -> list[str]:
    incident = hung_incident(records)
    if incident is None:
        return []
    if incident.get("state") != "ESCALATED":
        # QUARANTINED would mean the product isolated a node on suspicion.
        return [f"hung incident is {incident.get('state')}, expected ESCALATED"]
    return []


def grouped_errors(records: dict[str, Any], *, expect_single: bool) -> list[str]:
    hung = _signal_incidents(records, HUNG_SUFFIX)
    if expect_single and len(hung) > 1:
        return [f"{len(hung)} hung incidents for one stall wave (expected one merged)"]
    return []


def reported_nodes(incident: Mapping[str, Any]) -> set[str]:
    """The nodes whose own hung finding reached ``incident``.

    The first one created it (the host batch in its event id); a peer's finding
    in the same wave only appends a concurrent-evidence reason. ``node_ids`` is
    every attempt node from the start, so it cannot say who reported.
    """

    nodes = set()
    match = CREATOR.match(str(incident.get("event_id", "")))
    if match is not None:
        nodes.add(match.group("node"))
    for reason in incident.get("reasons") or []:
        head, separator, _ = str(reason).partition(CONCURRENT)
        if separator:
            nodes.add(head)
    return nodes


def both_nodes_errors(records: dict[str, Any], nodes: Sequence[str]) -> list[str]:
    incident = hung_incident(records)
    if incident is None:
        return ["no merged hung incident to cover the attempt nodes"]
    errors = []
    unnamed = sorted(set(nodes) - set(incident.get("node_ids") or []))
    if unnamed:
        errors.append(f"the merged hung incident does not name {unnamed}")
    silent = sorted(set(nodes) - reported_nodes(incident))
    if silent:
        errors.append(f"no hung evidence from {silent} reached the merged incident")
    return errors


def timing_errors(
    records: dict[str, Any], started: datetime, nodes: Sequence[str]
) -> list[str]:
    """ZERO_WARNING on every node after 60 s, then HUNG_SUSPECTED after 600 s."""

    incident = hung_incident(records)
    if incident is None:
        return []
    hung_at = parse_time(incident.get("created_at"))
    if hung_at is None:
        return ["the hung incident has no creation time"]
    errors = []
    waited = (hung_at - started).total_seconds()
    if waited < ZERO_HUNG_SECONDS - CLOCK_TOLERANCE_SECONDS:
        errors.append(
            f"HUNG_SUSPECTED {waited:.0f} s after the stall, before the "
            f"{ZERO_HUNG_SECONDS} s hung threshold"
        )
    warnings = _signal_incidents(records, WARNING_SUFFIX)
    for node in nodes:
        times = [
            moment
            for item in warnings
            if item.get("node_ids") == [node]
            and (moment := parse_time(item.get("created_at"))) is not None
        ]
        if not times:
            errors.append(f"{node}: no ZERO_WARNING incident for the stall")
            continue
        offset = (min(times) - started).total_seconds()
        if offset < ZERO_WARNING_SECONDS - CLOCK_TOLERANCE_SECONDS:
            errors.append(
                f"{node}: ZERO_WARNING {offset:.0f} s after the stall, before the "
                f"{ZERO_WARNING_SECONDS} s warning threshold"
            )
        if min(times) > hung_at:
            errors.append(f"{node}: ZERO_WARNING only after HUNG_SUSPECTED")
    return errors


def episode_stages(
    records: dict[str, Any], started: datetime, nodes: Sequence[str]
) -> dict[str, list[str]]:
    """The checks one stall wave's records must pass."""

    return {
        "chain": dag_errors(records),
        "executed": execution_errors(records),
        "escalated": escalated_errors(records),
        "timing": timing_errors(records, started, nodes),
        "merged": grouped_errors(records, expect_single=True),
    }


def repeat_errors(
    first: dict[str, Any], second: dict[str, Any], before: dict[str, Any]
) -> list[str]:
    """The second stall opened its own hung incident and workflow.

    After a rearm the product opens a new incident whose workflow names the
    first one as its predecessor (``hung_episode.new_hung_episode``,
    ``families/grouped_health.py``); otherwise the finding is only concurrent
    evidence on the first incident. ``first``/``second``: each stall's records;
    ``before``: the first stall's, read just before the second.
    """

    one = hung_incident(first)
    if one is None:
        return ["no first hung incident to repeat after"]
    errors = []
    earlier = hung_incident(before)
    if earlier is not None:
        known = set(earlier.get("reasons") or [])
        absorbed = [
            reason
            for reason in one.get("reasons") or []
            if CONCURRENT in str(reason) and reason not in known
        ]
        if absorbed:
            errors.append(
                f"the first hung incident {one.get('incident_id')} absorbed the "
                f"second stall's evidence: {absorbed}"
            )
    two = hung_incident(second)
    if two is None:
        return [*errors, "the second stall raised no hung incident of its own"]
    workflow, origin = two.get("workflow_request_id"), one.get("workflow_request_id")
    if not workflow or workflow == origin:
        return [
            *errors,
            f"the second hung incident has no workflow of its own: {workflow}",
        ]
    follows = ((second.get("workflows") or {}).get(workflow) or {}).get(
        "predecessor_workflow_id"
    )
    if follows != origin:
        errors.append(
            f"the second hung workflow {workflow} follows {follows}, not the first {origin}"
        )
    return errors


def no_suspicion_errors(records: dict[str, Any]) -> list[str]:
    raised = [
        str(item.get("event_id"))
        for suffix in (WARNING_SUFFIX, HUNG_SUFFIX)
        for item in _signal_incidents(records, suffix)
    ]
    if raised:
        return [f"a never-active job raised a zero-traffic suspicion: {raised}"]
    return []


def _triage_ranks(
    records: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]] | None:
    """Every rank row COLLECT_HUNG_TRIAGE returned, and the results unread.

    The executor stores each node's triage result as the summary codec's
    zlib/JSON envelope (``hung_summary_codec.encode_summary``); a document
    without ``encoding`` (an older record) is read as it is. A result that does
    not decode is named, never read as a node that sampled nothing.
    """

    executions = [
        item
        for item in _hung_workflow(records).get("step_executions") or []
        if item.get("operation") == "COLLECT_HUNG_TRIAGE"
    ]
    if not executions:
        return None
    rows: list[dict[str, Any]] = []
    unreadable: list[str] = []
    for execution in executions:
        results = (execution.get("details") or {}).get("node_results") or {}
        for node, value in sorted(results.items()):
            try:
                document = decode_summary(value)
            except ValueError as exc:
                unreadable.append(
                    f"COLLECT_HUNG_TRIAGE result of {node} is unreadable: {exc}"
                )
                continue
            ranks = document.get("ranks")
            rows.extend(item for item in ranks or [] if isinstance(item, dict))
    return rows, unreadable


def fr_errors(records: dict[str, Any], *, expect_configured: bool) -> list[str]:
    """What COLLECT_HUNG_TRIAGE read from each rank's flight recorder.

    Configured: no rank reads ``not_configured`` and at least one PID got two
    real dumps (``flight_recorder_series`` PAIRED). Not configured: nothing but
    ``not_configured`` (or an exhausted budget), and no dump series at all.
    """

    if hung_incident(records) is None:
        return []
    triage = _triage_ranks(records)
    if triage is None:
        return ["no COLLECT_HUNG_TRIAGE execution to read FR status from"]
    rows, errors = triage
    if not rows:
        return [*errors, "COLLECT_HUNG_TRIAGE sampled no rank process"]
    statuses = [(row.get("flight_recorder") or {}).get("status") for row in rows]
    series = [(row.get("flight_recorder_series") or {}).get("status") for row in rows]
    if expect_configured:
        if "not_configured" in statuses:
            errors.append(
                f"FR was configured but {statuses.count('not_configured')} "
                "rank(s) read not_configured"
            )
        if "PAIRED" not in series:
            errors.append(
                "no sampled rank got two real FR dumps (statuses "
                f"{sorted({str(item) for item in statuses})}, series "
                f"{sorted({str(item) for item in series})})"
            )
        return errors
    unexpected = sorted({str(item) for item in statuses if item not in FR_OFF_STATUSES})
    if unexpected or any(series):
        errors.append(
            f"FR was not configured yet triage reports {unexpected or series}"
        )
    return errors


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _job(
    ctx: VariantContext,
    suffix: str,
    *,
    flight_recorder: bool = False,
    keep_active: bool = True,
    stall_mode: str = "quiet",
) -> ManagedNcclJob:
    name = run_identity(CASE, suffix, ctx.attempt, ctx.case_dir).replace(
        "gl011", "gl11"
    )
    job = ManagedNcclJob(
        ctx,
        name=name,
        flight_recorder=flight_recorder,
        keep_active=keep_active,
        stall_mode=stall_mode,
    )
    ctx.cleanup.push(f"job-{name}", job.delete)
    job.start()
    ctx.evidence.write(f"{suffix}-job", job.identity())
    return job


def _nodes(ctx: VariantContext) -> list[str]:
    return [ctx.settings.node, ctx.settings.peer_node]


def _thresholds(ctx: VariantContext) -> tuple[float, float]:
    """The product's active and zero thresholds as this node's collector runs."""

    return (
        float(ctx.setting("GPU_FAULT_EFA_TRAFFIC_MIN_ACTIVE_BPS", MIN_ACTIVE_BPS)),
        float(ctx.setting("GPU_FAULT_EFA_TRAFFIC_ZERO_BPS", ZERO_BPS)),
    )


def rearm_seconds(ctx: VariantContext) -> int:
    """How long the job must communicate normally before a second stall.

    The collector's recovery delivery window, inside which the product closes
    the first episode or not at all, plus three collector intervals of margin.
    """

    sustained = ctx.setting("GPU_FAULT_EFA_TRAFFIC_RECOVERY_SECONDS", RECOVERY_SECONDS)
    samples = ctx.setting(
        "GPU_FAULT_EFA_TRAFFIC_RECOVERY_MIN_SAMPLES", RECOVERY_MIN_SAMPLES
    )
    gap = ctx.setting(
        "GPU_FAULT_EFA_TRAFFIC_RECOVERY_MAX_GAP_SECONDS", RECOVERY_MAX_GAP_SECONDS
    )
    spacing = min(30.0, gap / 2)
    window = max(sustained + 2 * spacing, samples * spacing)
    return math.ceil(window) + 3 * ctx.host_interval


def _efa_window(
    ctx: VariantContext, *, seconds: float, name: str
) -> dict[str, list[dict[str, Any]]]:
    """Both nodes' probe counter readings, now and every 30 s for ``seconds``."""

    window: dict[str, list[dict[str, Any]]] = {node: [] for node in _nodes(ctx)}

    def read() -> dict[str, Any]:
        for node, readings in window.items():
            reading = ctx.probe("node", node=node).execute(
                "efa-traffic", "--run-id", ctx.run_id
            )
            readings.append(dict(reading))
        return {
            node: items[-1].get("pair_bytes_total") for node, items in window.items()
        }

    ctx.hold(seconds, name=name, check=read)
    read()
    rates = {node: efa_rates(items) for node, items in window.items()}
    ctx.evidence.write(f"{name}-efa", {"readings": window, "rates_bps": rates})
    ctx.evidence.append_oracle(
        [
            {
                "window": name,
                "node": node,
                "observed_at": item.get("observed_at"),
                "pair_bytes_total": item.get("pair_bytes_total"),
                "port_count": item.get("port_count"),
                "complete_pairs": item.get("complete_pairs"),
            }
            for node, items in window.items()
            for item in items
        ],
        source="efa-traffic",
    )
    return window


def _prove_baseline(ctx: VariantContext, *, name: str, active: bool = True) -> None:
    """The pre-stall baseline, over the product's startup grace, on both nodes."""

    minimum, _zero = _thresholds(ctx)
    window = _efa_window(ctx, seconds=STARTUP_GRACE + ctx.host_interval * 3, name=name)
    errors = baseline_errors(window, minimum_bps=minimum, active=active)
    ctx.evidence.write(f"{name}-baseline", {"minimum_bps": minimum, "errors": errors})
    if errors:
        premise = (
            "no active EFA baseline before the stall"
            if active
            else "the always-low job reached the active threshold"
        )
        raise VariantInconclusive(f"{premise}: " + "; ".join(errors))


def _prove_stalled(
    ctx: VariantContext,
    job: ManagedNcclJob,
    *,
    name: str,
    after_step: int | None = None,
) -> dict[str, list[str]]:
    """Zero traffic for the hung threshold, a live job, the trainer's account.

    ``after_step``: an earlier stall's step; only later lines tell of this one.
    """

    _minimum, zero = _thresholds(ctx)
    ctx.hold(STALL_SETTLE_SECONDS, name=f"{name}-settle")
    window = _efa_window(ctx, seconds=ZERO_HUNG_SECONDS + ctx.host_summary, name=name)
    errors = drop_errors(window, zero_bps=zero)
    liveness = job.alive()
    events = job.trainer_events()
    ctx.evidence.write(f"{name}-job", {"liveness": liveness, "trainer": events})
    if not liveness.get("alive"):
        errors.append("the job did not stay alive through the stall")
    current = events if after_step is None else events_after(events, after_step)
    errors.extend(stall_event_errors(current, job.stall_mode))
    if errors:
        raise VariantInconclusive(
            "the stall did not hold a live job at zero EFA traffic: "
            + "; ".join(errors)
        )
    return events


def _stall(
    ctx: VariantContext,
    job: ManagedNcclJob,
    *,
    name: str,
    after_step: int | None = None,
) -> tuple[datetime, dict[str, list[str]]]:
    """Stall the job and prove it: ``(when, the trainer's account)``."""

    started = ctx.clock()
    ctx.cleanup.push("release", job.release)
    ctx.evidence.write(f"{name}-stall", {"at": started.isoformat(), **job.stall()})
    events = _prove_stalled(ctx, job, name=f"{name}-stalled", after_step=after_step)
    return started, events


def _records(ctx: VariantContext, job: ManagedNcclJob, since: datetime) -> Any:
    return ctx.regional.cpu_python(
        HUNG_PROBE,
        ctx.settings.regional.cluster_id,
        ",".join(_nodes(ctx)),
        job.name,
        job.attempt_id,
        since.isoformat(),
    )


def _hung(
    ctx: VariantContext,
    job: ManagedNcclJob,
    since: datetime,
    *,
    require_nodes: Sequence[str] = (),
    created_since: datetime | None = None,
    name: str = "hung",
) -> dict[str, Any]:
    """The records once the hung incident settled (and named its reporters).

    ``created_since``: wait for a later stall's incident; the records still
    cover everything since ``since``.
    """

    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = _records(ctx, job, since)
        view = last if created_since is None else episode(last, created_since)
        incident = hung_incident(view)
        if incident is None or incident.get("state") not in SETTLED_STATES:
            return None
        if not set(require_nodes) <= reported_nodes(incident):
            return None
        return last

    value = ctx.wait_until(
        accept, timeout_seconds=HUNG_BUDGET, poll_seconds=20, name=name
    )
    records = value if value is not None else last
    ctx.evidence.write(f"{name}-records", records)
    return records


def _release_and_resume(
    ctx: VariantContext, job: ManagedNcclJob, *, name: str
) -> list[str]:
    """Release the stall and prove the job's own traffic came back."""

    ctx.evidence.write(f"{name}-release", job.release())
    ctx.cleanup.discard("release")
    minimum, zero = _thresholds(ctx)
    window = _efa_window(ctx, seconds=RESUME_SECONDS, name=f"{name}-resume")
    return resume_errors(window, floor_bps=minimum if job.keep_active else 2 * zero)


def _prove_recovered(ctx: VariantContext, job: ManagedNcclJob, *, name: str) -> None:
    """Release the stall; the job's own pair stays active past the rearm window."""

    errors = _release_and_resume(ctx, job, name=name)
    if not errors:
        minimum, _zero = _thresholds(ctx)
        floor = max(
            minimum,
            ctx.setting(
                "GPU_FAULT_EFA_TRAFFIC_RECOVERY_FALLBACK_BPS", RECOVERY_FALLBACK_BPS
            ),
        )
        seconds = rearm_seconds(ctx)
        window = _efa_window(ctx, seconds=seconds, name=f"{name}-recovered")
        errors = baseline_errors(window, minimum_bps=floor)
        ctx.evidence.write(
            f"{name}-recovered-baseline",
            {"minimum_bps": floor, "seconds": seconds, "errors": errors},
        )
    if errors:
        raise VariantInconclusive(
            "the job did not communicate normally through the product's rearm "
            "window: " + "; ".join(errors)
        )


def stalled_chain(
    ctx: VariantContext,
    suffix: str,
    *,
    stall_mode: str = "quiet",
    flight_recorder: bool = False,
    require_both: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[str]]]:
    run_survey(ctx)
    job = _job(ctx, suffix, flight_recorder=flight_recorder, stall_mode=stall_mode)
    _prove_baseline(ctx, name=f"{suffix}-active")
    started, _events = _stall(ctx, job, name=suffix)
    nodes = _nodes(ctx)
    records = _hung(ctx, job, started, require_nodes=nodes if require_both else ())
    stages = {
        **episode_stages(records, started, nodes),
        "resumed": _release_and_resume(ctx, job, name=suffix),
    }
    return job.identity(), records, stages


# --------------------------------------------------------------------------- #
# variants
# --------------------------------------------------------------------------- #
def active_to_zero(ctx: VariantContext) -> dict[str, Any]:
    identity, _, stages = stalled_chain(ctx, "a2z")
    return _result(stages, extra={"attempt": identity})


def always_low(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    # A job that never reaches the active threshold, then goes silent for the
    # whole hung window: the EFA rule needs a prior had_active sample, so the
    # zero run alone must not warn or suspect a hang.
    job = _job(ctx, "low", keep_active=False)
    _prove_baseline(ctx, name="low-quiet", active=False)
    started, _events = _stall(ctx, job, name="low")
    records = _records(ctx, job, started)
    ctx.evidence.write("hung-records", records)
    stages = {
        "no_suspicion": no_suspicion_errors(records),
        "resumed": _release_and_resume(ctx, job, name="low"),
    }
    return _result(
        stages,
        notes=["always-low is a suspicion signal, never a confirmed hang"],
        extra={"attempt": job.identity()},
    )


def write_progress(ctx: VariantContext) -> dict[str, Any]:
    identity, _, stages = stalled_chain(ctx, "wp", stall_mode="write")
    return _result(
        stages,
        notes=[
            "held ranks rewrite 4 MiB/s of checkpoint I/O during the stall; host"
            " writes do not delay the RDMA idle clock (rank liveness is CONTEXT_ONLY)"
        ],
        extra={"attempt": identity},
    )


def phase_progress(ctx: VariantContext) -> dict[str, Any]:
    identity, _, stages = stalled_chain(ctx, "pp", stall_mode="phase")
    return _result(
        stages,
        notes=[
            "held ranks keep advancing the step and switching the phase of their"
            " rank-progress/v1 snapshots; declared progress does not delay the"
            " RDMA idle clock"
        ],
        extra={"attempt": identity},
    )


def fr_enabled(ctx: VariantContext) -> dict[str, Any]:
    identity, records, stages = stalled_chain(ctx, "fron", flight_recorder=True)
    stages["fr_dumps"] = fr_errors(records, expect_configured=True)
    return _result(
        stages,
        notes=["FR configured: triage must take two real dumps per PID"],
        extra={"attempt": identity},
    )


def fr_disabled(ctx: VariantContext) -> dict[str, Any]:
    identity, records, stages = stalled_chain(ctx, "froff")
    stages["fr_dumps"] = fr_errors(records, expect_configured=False)
    return _result(
        stages,
        notes=[
            "FR off: triage records not_configured and never fabricates a dump; still ESCALATED"
        ],
        extra={"attempt": identity},
    )


def simultaneous_nodes(ctx: VariantContext) -> dict[str, Any]:
    identity, records, stages = stalled_chain(ctx, "sim", require_both=True)
    stages["both_nodes"] = both_nodes_errors(records, _nodes(ctx))
    return _result(
        stages,
        notes=["two nodes in one stall wave merge into one workflow, not two"],
        extra={"attempt": identity},
    )


def repeat_after_recovery(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    job = _job(ctx, "rar")
    nodes = _nodes(ctx)
    _prove_baseline(ctx, name="rar-active")
    first_at, events = _stall(ctx, job, name="rar")
    first = episode(_hung(ctx, job, first_at), first_at)
    stages = episode_stages(first, first_at, nodes)
    if any(stages.values()):
        # Without a first episode the product handled there is nothing to repeat.
        stages["resumed"] = _release_and_resume(ctx, job, name="rar")
        stages["repeat"] = ["not attempted: the first stall's hung chain failed"]
        return _result(stages, extra={"attempt": job.identity()})
    _prove_recovered(ctx, job, name="rar")
    before = episode(_records(ctx, job, first_at), first_at)
    ctx.evidence.write("hung-before-repeat", before)
    second_at, _ = _stall(ctx, job, name="rar2", after_step=stall_step(events))
    records = _hung(ctx, job, first_at, created_since=second_at, name="hung2")
    first, second = episode(records, first_at, second_at), episode(records, second_at)
    stages = {
        **episode_stages(first, first_at, nodes),
        "repeat": repeat_errors(first, second, before),
        **{
            f"repeat_{key}": value
            for key, value in episode_stages(second, second_at, nodes).items()
        },
        "resumed": _release_and_resume(ctx, job, name="rar2"),
    }
    return _result(
        stages,
        notes=["a stall after a proven recovery is a new episode, not the old one"],
        extra={"attempt": job.identity(), "rearm_seconds": rearm_seconds(ctx)},
    )


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


def _spec(
    variant: str,
    injection: str,
    expected: str,
    execute,
    *,
    risk: str = "M",
    max_minutes: int = 65,
) -> VariantSpec:
    return VariantSpec(
        case_id=CASE,
        variant=variant,
        proof_mode=NATIVE,
        lanes=("hp",),
        risk=risk,
        flags="S",
        injection=injection,
        expected=expected,
        cleanup=(
            "release the stall and prove the EFA pair resumes; delete the test Job; "
            "confirm no node was isolated/stopped/reset"
        ),
        max_minutes=max_minutes,
        execute=execute,
    )


def _deferred(
    variant: str, reason: str, expected: str, *, proof: str = LIVE_REPLAY
) -> VariantSpec:
    return deferred_variant(
        CASE, variant, proof_mode=proof, reason=reason, lanes=("hp",), expected=expected
    )


_FR_FIXTURE = (
    "FIXTURE_NOT_READY: needs a reviewed test flight-recorder dump responder that "
    "updates/withholds the second dump on demand, not wired into this runner"
)
_SCALE = "SCALE: needs a 128-node/1024-process task; this site has one 4-node HyperPod cluster"
_ROLE_FIXTURE = (
    "FIXTURE_NOT_READY: needs a mixed-parallelism test job that declares PP/EP roles "
    "and thread roles through the rank-progress diagnostic_context, not wired this phase"
)


VARIANTS = [
    _spec(
        "active_to_zero",
        "a two-node NCCL job at 1-4 MiB/s of EFA traffic per node, then one rank in"
        " an extra collective so the EFA pair falls to zero",
        "ZERO_WARNING (60 s) then HUNG_SUSPECTED (600 s) -> the read-only diagnostic"
        " DAG, every step executed -> ESCALATED; no STOP/reset/cordon; traffic"
        " resumes on release",
        active_to_zero,
    ),
    _spec(
        "always_low",
        "a two-node job that only all-reduces its heartbeat (never 1 MiB/s), then"
        " goes silent for the hung window",
        "no ZERO_WARNING or hung suspicion from low/zero traffic alone (the rule"
        " needs prior had_active)",
        always_low,
        risk="L",
        max_minutes=40,
    ),
    _spec(
        "write_progress",
        "stall the collective while the held ranks keep writing checkpoint I/O",
        "host writes do not delay the RDMA idle clock; HUNG_SUSPECTED still fires -> ESCALATED",
        write_progress,
    ),
    _spec(
        "phase_progress",
        "stall the collective while the held ranks keep advancing a declared"
        " step/phase (rank-progress/v1)",
        "declared progress does not delay the RDMA idle clock; HUNG_SUSPECTED still fires -> ESCALATED",
        phase_progress,
    ),
    _spec(
        "fr_enabled",
        "the stalled job runs the flight recorder (trace buffer + debug pipe)",
        "triage takes two real FR dumps per PID; chain -> ESCALATED",
        fr_enabled,
    ),
    _spec(
        "fr_disabled",
        "the stalled job has no FR pipe and a zero trace buffer",
        "triage records not_configured and fabricates no dump; chain -> ESCALATED with process evidence",
        fr_disabled,
    ),
    _spec(
        "simultaneous_nodes",
        "both nodes of the two-node job lose EFA in one stall wave (needs --node-b)",
        "one merged grouped incident/workflow with both nodes' hung evidence, not two",
        simultaneous_nodes,
    ),
    _spec(
        "repeat_after_recovery",
        "stall the job, release it once the hung chain settled, keep its own EFA"
        " traffic active past the product's rearm window, then stall it again",
        "each stall: ZERO_WARNING then HUNG_SUSPECTED -> its own incident and"
        " read-only DAG, every step executed -> ESCALATED; the second workflow"
        " follows the first; nothing suppressed, merged or duplicated",
        repeat_after_recovery,
        max_minutes=120,
    ),
    _deferred(
        "broken_pair",
        "FIXTURE_NOT_READY: a broken RX/TX counter pair needs a sysfs hw_counters read fixture; it must read as an invalid observation, never zero",
        "an incomplete counter pair is observation-invalid and does not start a zero streak",
    ),
    _deferred(
        "attempt_change",
        "FIXTURE_NOT_READY: a controllable attempt change on a stalled two-node NCCL job (A warned, then a fresh attempt B) is not wired this phase",
        "a new attempt starts at WARMUP and never inherits the previous attempt's zero streak",
        proof=NATIVE,
    ),
    _deferred(
        "unreachable_node",
        "NOT_AUTHORIZED: isolating one node's Agent command channel needs a net block (WATCHER_NETWORK_ISOLATION-class) not arranged this phase",
        "the unreachable node is recorded as missing coverage, never a fabricated zero; still ESCALATED",
        proof=NATIVE,
    ),
    _deferred(
        "large_job",
        _SCALE,
        "a 1024-process task: coverage denominators and command counts recorded, not extrapolated",
        proof=NATIVE,
    ),
    _deferred(
        "few_candidates",
        _FR_FIXTURE,
        "1-3 trusted candidates get a targeted bundle plus at most one control rank",
    ),
    _deferred(
        "ambiguous_candidates",
        _FR_FIXTURE,
        "too many/weak candidates keep the conservative bundle scope",
    ),
    _deferred(
        "different_process_groups",
        _FR_FIXTURE,
        "collective progress is never compared across different process groups",
    ),
    _deferred(
        "no_step_no_rank",
        _SCALE,
        "128-node task with no step reporter, no FR and no RANK: full process evidence with rank unknown",
        proof=NATIVE,
    ),
    _deferred(
        "active_window_resume",
        _SCALE,
        "unfinished commands hold the concurrency window; resume does not re-sample completed nodes",
        proof=NATIVE,
    ),
    _deferred(
        "fr_temporal_completion",
        _FR_FIXTURE,
        "two new dumps per PID/PG compare enqueued/started/completed watermarks at >= 2 s apart",
    ),
    _deferred(
        "fr_second_missing",
        _FR_FIXTURE,
        "a withheld second dump is a forensics gap, never a real stall",
    ),
    _deferred(
        "fr_whole_group",
        _FR_FIXTURE,
        "a whole process group pending is reported without inventing a single culprit rank",
    ),
    _deferred(
        "pyspy_nonzero",
        "FIXTURE_NOT_READY: needs a test py-spy tool returning non-zero exit / error text / bad JSON on the node PATH, not wired this phase",
        "a py-spy failure is classified as a failed stack sample, no signature, other evidence kept",
    ),
    _deferred(
        "role_mismatch",
        _ROLE_FIXTURE,
        "different PP stages / EP roles are never attributed to one another",
    ),
    _deferred(
        "tid_namespace",
        _ROLE_FIXTURE,
        "container native TID, host NSpid and thread starttime are reconciled across samples",
    ),
    _deferred(
        "wchan_zero",
        _ROLE_FIXTURE,
        "a 0/empty wait channel is not a wait location; CPU is normalized by SC_CLK_TCK and elapsed seconds",
    ),
    _deferred(
        "cpu_interval",
        _ROLE_FIXTURE,
        "CPU cores use delta ticks over actual elapsed, a shared GPU is not claimed as the PID's utilization",
    ),
    _deferred(
        "fr_plus_process",
        _FR_FIXTURE,
        "FR completion progress does not silence the independent process I/O-wait evidence",
    ),
]
