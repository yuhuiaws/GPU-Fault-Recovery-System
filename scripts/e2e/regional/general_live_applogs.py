"""GF-GENERAL-LIVE-017 application error logs.

Two reviewed sources feed the node log collector's real rules
(``log_rules`` / ``log_recovery_context``):

* **journal replay** (``cgroup_oom``, ``nccl_error``, ``distributed_error``):
  the kernel-log replay helper in journal mode appends reviewed records to the
  real ``journalctl`` output. An out-of-memory line is a WARNING
  ``RUN_DIAGNOSTICS`` from any origin; NCCL and distributed-training errors are
  ``NO_ACTION`` evidence owned by the completion watcher -- never a hardware
  marker or an extra diagnostic workflow. The OOM line is the kernel's own
  ``Memory cgroup out of memory: Killed process ...`` (``CGROUP_OOM_MESSAGE``),
  and every record travels under a per-run synthetic cursor: the cursor is the
  entry id the control plane hashes into the incident's event id, so a second
  run on the same node and boot must not inherit the first run's incident
  (``general_live_kernel_source_probe``).
* **CRI Pod logs** (``live_pid``, ``exited_pid``, ``old_pod``,
  ``cri_partial_rotation``): ``general_live_applog_fixture`` writes reviewed
  CRI records under ``/var/log/pods`` in the directory kubelet made for a
  managed attempt the runner created (``general_live_applog_job``), so the Pod
  UID, container name, attempt and host PID the product attributes by are
  genuine. The verdict reads the attempt observation, the delivered NODE_LOGS
  entries (``pod_uid``/``container_name``, the CRI event time), the incident's
  attempt and the diagnostic bundle step's attribution parameters
  (``attribution``, ``process_state``, ``target_pids_by_node``): a live process
  is targeted, an exited one is never stack-sampled, an old Pod never reaches
  the new attempt, and a ``P``/``F`` record stitches across polls, a rename
  rotation and a collector restart.

``missing_flight_recorder`` is native: a managed two-node NCCL job without a
flight recorder (``ManagedNcclJob(flight_recorder=False)``) stalls at zero EFA
traffic while every process stays alive, and the product's independent
RDMA-idle trigger (``ATTEMPT_SCOPED_RDMA_IDLE``) must raise the hung chain with
``not_configured`` flight-recorder evidence -- no log line is involved.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any, Callable

from scripts.e2e.regional import general_live_nccl as nccl
from scripts.e2e.regional.general_live_applog_fixture import (
    CUDA_OOM_LINE,
    PARTIAL_HEAD,
    PARTIAL_TAIL,
    CriPodLogFixture,
    cri_pod_logs,
    reviewed_line,
)
from scripts.e2e.regional.general_live_applog_job import ManagedApplogJob
from scripts.e2e.regional.general_live_common import activity, restore_incidents
from scripts.e2e.regional.general_live_fixture import (
    GeneralLiveSettings,
    VariantContext,
    VariantInconclusive,
    run_identity,
    utc_now,
)
from scripts.e2e.regional.general_live_kernel_replay import (
    kernel_replay,
    reviewed_record,
)
from scripts.e2e.regional.general_live_registry import (
    LIVE_REPLAY,
    NATIVE,
    VariantSpec,
)
from scripts.e2e.regional.general_live_store import applog_records
from scripts.e2e.regional.general_live_verdicts import (
    final_execution,
    hardware_errors,
    node_isolated,
    operations,
    parse_time,
    settled,
    step_parameters,
    verdict,
    workflows_of,
)

CASE = "GF-GENERAL-LIVE-017"
ENABLE = "ENABLE_NODE_LOG_COLLECTOR"
DETECTION_BUDGET = 420
DIAGNOSTIC_WORKFLOW_OPS = frozenset(
    {"RUN_DIAGNOSTICS", "COLLECT_DIAGNOSTIC_BUNDLE", "RUN_DCGM_DIAGNOSTIC"}
)
#: How long the collector (10 s poll) plus ingestion may take to deliver a line.
ENTRY_BUDGET = 180
#: How long the completion watcher (5 s poll) may take to record a Pod change.
ATTEMPT_BUDGET = 300
#: How long the product's RESTART_WORKLOAD of a failed first attempt may take.
RESTART_BUDGET = 900
#: How long a diagnostic chain may take to settle before its parameters are read.
SETTLE_BUDGET = 900
#: ``log_recovery_context._resolve_log_attempt``: an attempt observation older
#: than this (relative to the batch) cannot be attributed to.
ATTRIBUTION_FRESHNESS_SECONDS = 120
#: ``old_pod``: a newly discovered file is read from this far before its end,
#: so the line written before kubelet's GC can remove the recreated directory
#: is read on the collector's first visit (one 10 s poll, no second visit).
OLD_POD_INITIAL_TAIL = 65536
OLD_POD_ATTEMPTS = 3
STEPS_WITH_DIAGNOSTIC_PARAMETERS = ("COLLECT_DIAGNOSTIC_BUNDLE", "COLLECT_HUNG_TRIAGE")


# --------------------------------------------------------------------------- #
# journal replay variants (phase 2)
# --------------------------------------------------------------------------- #
def _log_incidents(activity_document: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        (
            item
            for item in activity_document.get("incidents") or []
            if item.get("event_type") == "NODE_HEALTH"
            and str(item.get("event_id") or "").startswith("log-")
        ),
        key=lambda item: str(item.get("created_at")),
    )


def _drive(ctx: VariantContext, records: list[dict[str, Any]]) -> Any:
    replay = kernel_replay(ctx, mode="journal")
    replay.create()
    replay.install(records)
    opened = replay.open(approved=True, restore_seconds=ctx.spec.max_minutes * 60)
    ctx.evidence.write("replay-open", opened)
    started = utc_now()
    replay.arm()
    ctx.hold(90, name="inject")
    oracle = replay.read_back()
    ctx.evidence.append_oracle([{"injected": oracle.get("injected")}], source="applog")
    if not (oracle.get("injected") or {}).get("emission_counts"):
        raise VariantInconclusive(
            "the journal wrapper reported no application records ingested"
        )
    return started


def _wait_incident(ctx: VariantContext, since: Any, name: str) -> dict[str, Any]:
    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = activity(ctx, since, kinds=("NODE_LOGS",))
        return last if _log_incidents(last) else None

    value = ctx.wait_until(
        accept, timeout_seconds=DETECTION_BUDGET, poll_seconds=15, name=name
    )
    return value if value is not None else last


#: The kernel's memcg OOM-kill line, Linux v6.12 ``mm/oom_kill.c``
#: ``oom_kill_process()`` -> ``__oom_kill_process()``: ``pr_err("%s: Killed
#: process %d (%s) total-vm:%lukB, anon-rss:%lukB, file-rss:%lukB,
#: shmem-rss:%lukB, UID:%u pgtables:%lukB oom_score_adj:%hd\n", message, ...)``
#: with ``message`` = "Memory cgroup out of memory" for a cgroup limit
#: (``out_of_memory()``: ``!is_memcg_oom(oc) ? "Out of memory" : "Memory
#: cgroup out of memory"``). No ``pr_fmt`` prefix; the journal MESSAGE never
#: carries the ``kernel:`` syslog identifier that ``dmesg -T``/syslog show.
CGROUP_OOM_MESSAGE = (
    "Memory cgroup out of memory: Killed process 44021 (python3) "
    "total-vm:31485720kB, anon-rss:30961664kB, file-rss:12288kB, shmem-rss:0kB, "
    "UID:1000 pgtables:61200kB oom_score_adj:994"
)


def cgroup_oom(ctx: VariantContext) -> dict[str, Any]:
    """A reviewed OOM-kill line is a diagnostic, never isolation or a STOP."""

    started = _drive(
        ctx, [reviewed_record(CGROUP_OOM_MESSAGE, transport="kernel", priority=3)]
    )
    observed = _wait_incident(ctx, started, "oom-detect")
    incidents = _log_incidents(observed)
    stages = {
        "diagnostic": []
        if any(item.get("effective_action") == "RUN_DIAGNOSTICS" for item in incidents)
        else ["an OOM line did not produce a RUN_DIAGNOSTICS finding"],
        "no_isolation": ["node isolated on an OOM line"]
        if node_isolated(ctx.regional.node_snapshot(ctx.settings.node))
        else [],
        "no_stop": _no_stop(observed, incidents),
        "hardware": hardware_errors(observed),
    }
    return _result(
        stages,
        notes=["out-of-memory is a WARNING diagnostic; no STOP without attribution"],
    )


def _evidence_only_variant(
    ctx: VariantContext, message: str, note: str
) -> dict[str, Any]:
    started = _drive(
        ctx, [reviewed_record(message, transport="stdout", unit="train-app.service")]
    )
    observed = _wait_incident(ctx, started, "evidence-detect")
    incidents = _log_incidents(observed)
    stages = {
        "no_hardware_marker": hardware_errors(observed),
        "no_quarantine": [
            f"an application error quarantined the node: {item['incident_id']}"
            for item in incidents
            if item.get("effective_action") == "QUARANTINE"
        ],
        "no_diagnostic_workflow": _no_diagnostic_workflow(observed, incidents),
        "not_isolated": ["node isolated by an application error"]
        if node_isolated(ctx.regional.node_snapshot(ctx.settings.node))
        else [],
    }
    return _result(stages, notes=[note])


def nccl_error(ctx: VariantContext) -> dict[str, Any]:
    return _evidence_only_variant(
        ctx,
        "test-app: NCCL WARN Bootstrap : unhandled error, aborting",
        "a plain NCCL error is evidence-only (completion watcher), not a hardware fault",
    )


def distributed_error(ctx: VariantContext) -> dict[str, Any]:
    return _evidence_only_variant(
        ctx,
        "test-app: torch.distributed.DistBackendError: collective operation timeout",
        "a distributed-training error is evidence-only, not a hardware fault",
    )


def _no_stop(
    activity_document: dict[str, Any], incidents: list[dict[str, Any]]
) -> list[str]:
    errors = []
    for incident in incidents:
        for workflow in workflows_of(activity_document, str(incident["incident_id"])):
            if "STOP_WORKLOADS" in operations(workflow):
                errors.append(
                    f"workflow {workflow.get('request_id')} stopped a workload with no attribution"
                )
    return errors


def _no_diagnostic_workflow(
    activity_document: dict[str, Any], incidents: list[dict[str, Any]]
) -> list[str]:
    errors = []
    for incident in incidents:
        for workflow in workflows_of(activity_document, str(incident["incident_id"])):
            found = sorted(set(operations(workflow)) & DIAGNOSTIC_WORKFLOW_OPS)
            if found:
                errors.append(
                    f"an evidence-only application error ran a diagnostic workflow {found}"
                )
    return errors


# --------------------------------------------------------------------------- #
# CRI Pod-log attribution: product-record readers (pure)
# --------------------------------------------------------------------------- #
def safe_component(value: str) -> str:
    """``HostHealthEvaluator._safe``: the event-id component sanitizer."""

    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value)[:160]


def log_event_id(cluster_id: str, node_id: str, entry_id: str) -> str:
    """``log_recovery_context.log_event_id`` recomputed from a delivered entry."""

    digest = hashlib.sha256(
        json.dumps(
            [cluster_id, node_id, entry_id], ensure_ascii=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()[:32]
    return f"log-{safe_component(cluster_id)}-{safe_component(node_id)}-{digest}"


def entries_of(records: dict[str, Any], paths: set[str]) -> list[dict[str, Any]]:
    """The delivered NODE_LOGS entries read from any of the fixture ``paths``."""

    return [
        item
        for item in records.get("entries") or []
        if str((item.get("fields") or {}).get("path")) in paths
    ]


def incident_of(
    records: dict[str, Any], entry: dict[str, Any]
) -> dict[str, Any] | None:
    """The incident the product minted for ``entry`` (by its derived event id)."""

    wanted = log_event_id(
        str(records.get("cluster_id")),
        str(records.get("node_id")),
        str(entry.get("entry_id")),
    )
    for incident in records.get("incidents") or []:
        if incident.get("event_id") == wanted:
            return incident
    return None


def workflow_of(
    records: dict[str, Any], incident: dict[str, Any] | None
) -> dict[str, Any] | None:
    if not incident or not incident.get("workflow_request_id"):
        return None
    value = (records.get("workflows") or {}).get(str(incident["workflow_request_id"]))
    return dict(value) if isinstance(value, dict) else None


def attempt_of(records: dict[str, Any], attempt_id: str) -> dict[str, Any] | None:
    for item in records.get("attempts") or []:
        if item.get("attempt_id") == attempt_id:
            return item
    return None


def container_of(
    attempt: dict[str, Any] | None, pod_uid: str, container: str
) -> dict[str, Any] | None:
    for item in (attempt or {}).get("containers") or []:
        if item.get("pod_uid") == pod_uid and item.get("container_name") == container:
            return item
    return None


def diagnostic_parameters(workflow: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The finding's attribution parameters, as copied onto the bundle steps."""

    if not workflow:
        return []
    return [
        parameters
        for operation in STEPS_WITH_DIAGNOSTIC_PARAMETERS
        for parameters in step_parameters(workflow, operation)
    ]


def capture_steps(workflow: dict[str, Any] | None) -> list[str]:
    """Steps that would sample a process: capture requested or PIDs targeted."""

    if not workflow:
        return []
    return [
        str(step.get("operation"))
        for step in workflow.get("official_steps") or []
        if (step.get("parameters") or {}).get("capture_process_state")
        or (step.get("parameters") or {}).get("target_pids_by_node")
    ]


def entry_identity_errors(
    entry: dict[str, Any] | None,
    *,
    pod_uid: str,
    container: str,
    stamp: str | None,
    message: str | None = None,
) -> list[str]:
    """The delivered entry names the genuine Pod/container and the CRI time."""

    if entry is None:
        return ["no fixture line was delivered"]
    errors = []
    fields = entry.get("fields") or {}
    if entry.get("source") != "training-log":
        errors.append(f"entry source {entry.get('source')!r} is not training-log")
    if fields.get("pod_uid") != pod_uid:
        errors.append(f"entry pod_uid {fields.get('pod_uid')!r} != {pod_uid}")
    if fields.get("container_name") != container:
        errors.append(
            f"entry container {fields.get('container_name')!r} != {container}"
        )
    if fields.get("timestamp_source") != "cri":
        errors.append(f"timestamp source {fields.get('timestamp_source')!r} is not cri")
    event_at = parse_time(entry.get("observed_at"))
    stamped = parse_time(stamp) if stamp else None
    if event_at is None or stamped is None:
        errors.append("the entry or the fixture stamp has no readable time")
    elif abs((event_at - stamped).total_seconds()) > 1.0:
        errors.append(
            f"event time {entry.get('observed_at')} is not the CRI stamp {stamp}"
        )
    if message is not None and entry.get("message") != message:
        errors.append(f"entry message {entry.get('message')!r} != {message!r}")
    return errors


def attribution_errors(
    incident: dict[str, Any] | None,
    workflow: dict[str, Any] | None,
    *,
    job_id: str,
    attempt_id: str,
    process_state: str,
    node: str,
    pid: int | None,
) -> list[str]:
    """The incident names the attempt; the bundle step carries the attribution."""

    if incident is None:
        return ["the delivered line produced no incident"]
    errors = []
    if incident.get("job_id") != job_id or incident.get("attempt_id") != attempt_id:
        errors.append(
            f"incident attributed to {incident.get('job_id')}/{incident.get('attempt_id')}"
            f", expected {job_id}/{attempt_id}"
        )
    parameters = diagnostic_parameters(workflow)
    if process_state == "LIVE":
        if not parameters:
            errors.append(
                "no diagnostic bundle step carries the attribution parameters"
            )
        for item in parameters:
            if item.get("attribution") != "OBSERVED_PROCESS":
                errors.append(
                    f"attribution {item.get('attribution')!r} != OBSERVED_PROCESS"
                )
            if item.get("process_state") != "LIVE":
                errors.append(f"process_state {item.get('process_state')!r} != LIVE")
            if item.get("log_source") != "training-log":
                errors.append(f"log_source {item.get('log_source')!r} != training-log")
            targets = (item.get("target_pids_by_node") or {}).get(node)
            if pid is not None and targets != [pid]:
                errors.append(f"target pids {targets!r} != [{pid}] on {node}")
            if item.get("capture_process_state") is not True:
                errors.append("capture_process_state is not set for a live process")
    return errors


def diagnostic_chain_errors(
    incident: dict[str, Any] | None,
    workflow: dict[str, Any] | None,
    *,
    expect_bundle: bool,
) -> list[str]:
    """RUN_DIAGNOSTICS; the bundle ran for a live process and never for an exited one."""

    if incident is None:
        return ["no incident"]
    errors = []
    if incident.get("effective_action") != "RUN_DIAGNOSTICS":
        errors.append(
            f"effective action {incident.get('effective_action')!r} != RUN_DIAGNOSTICS"
        )
    if workflow is None:
        return [*errors, "the incident has no workflow"]
    ops = operations(workflow)
    if expect_bundle:
        if "COLLECT_DIAGNOSTIC_BUNDLE" not in ops:
            errors.append(f"no COLLECT_DIAGNOSTIC_BUNDLE for a live process: {ops}")
        execution = final_execution(workflow, "COLLECT_DIAGNOSTIC_BUNDLE")
        if execution is None or execution.get("status") != "SUCCEEDED":
            errors.append(
                "the diagnostic bundle did not succeed: "
                f"{(execution or {}).get('status')} {(execution or {}).get('error')}"
            )
        else:
            summary = (execution.get("details") or {}).get("manifest_summary") or {}
            if summary.get("capture_process_state") is not True:
                errors.append("the bundle did not capture the live process state")
    else:
        sampled = capture_steps(workflow)
        if sampled:
            errors.append(
                f"an exited process was still scheduled for sampling: {sampled}"
            )
        if "COLLECT_DIAGNOSTIC_BUNDLE" in ops:
            errors.append(
                "a diagnostic bundle (stack capture) was planned for an exited PID"
            )
    return errors


def stop_errors(workflow: dict[str, Any] | None) -> list[str]:
    if workflow and "STOP_WORKLOADS" in operations(workflow):
        return [f"workflow {workflow.get('request_id')} stopped the attempt"]
    return []


def no_diagnostic_errors(workflow: dict[str, Any] | None) -> list[str]:
    found = sorted(set(operations(workflow or {})) & DIAGNOSTIC_WORKFLOW_OPS)
    return [f"an evidence-only line ran a diagnostic workflow {found}"] if found else []


def new_attempt_errors(
    incident: dict[str, Any] | None,
    workflow: dict[str, Any] | None,
    *,
    new_attempt_id: str,
    new_pid: int | None,
    node: str,
) -> list[str]:
    """An old Pod's line never reaches the new attempt or its live PID."""

    if incident is None:
        return ["the delivered line produced no incident"]
    errors = []
    if incident.get("attempt_id") == new_attempt_id:
        errors.append(
            f"the old Pod's line was attributed to the new attempt {new_attempt_id}"
        )
    for item in diagnostic_parameters(workflow):
        targets = (item.get("target_pids_by_node") or {}).get(node) or []
        if new_pid is not None and new_pid in targets:
            errors.append(f"the new attempt's PID {new_pid} was targeted for sampling")
        if item.get("process_state") == "LIVE":
            errors.append("an old Pod's line was graded as a LIVE process")
    return errors


def stitch_errors(
    entries: list[dict[str, Any]],
    *,
    message: str,
    rotated_path: str,
    head_stamp: str | None,
) -> list[str]:
    """One record with the whole message, at the P fragment's time, on the rotated inode."""

    if not entries:
        return ["no record was delivered from the fixture file"]
    errors = []
    whole = [item for item in entries if item.get("message") == message]
    if len(whole) != 1:
        errors.append(
            f"{len(whole)} records carry the stitched message, expected exactly one"
        )
    fragments = [
        item.get("message") for item in entries if item.get("message") != message
    ]
    if fragments:
        errors.append(f"fragments were delivered on their own: {fragments}")
    if whole:
        entry = whole[0]
        if str((entry.get("fields") or {}).get("path")) != rotated_path:
            errors.append(
                f"the stitched record was read from {(entry.get('fields') or {}).get('path')}"
                f", not the rotated file {rotated_path}"
            )
        event_at = parse_time(entry.get("observed_at"))
        stamped = parse_time(head_stamp) if head_stamp else None
        if event_at is None or stamped is None:
            errors.append("the stitched record or the P stamp has no readable time")
        elif abs((event_at - stamped).total_seconds()) > 1.0:
            errors.append(
                f"the stitched record's time {entry.get('observed_at')} is not the "
                f"P fragment's stamp {head_stamp}"
            )
    return errors


def independent_trigger_errors(records: dict[str, Any]) -> list[str]:
    """The hung chain came from the RDMA-idle rule, with process capture."""

    incident = nccl.hung_incident(records)
    if incident is None:
        return ["no EFA hung_suspected incident"]
    workflow = (records.get("workflows") or {}).get(
        str(incident.get("workflow_request_id"))
    )
    parameters = [
        item
        for item in diagnostic_parameters(
            workflow if isinstance(workflow, dict) else None
        )
        if item
    ]
    if not parameters:
        return ["the hung workflow carries no diagnostic parameters"]
    errors = []
    for item in parameters:
        if item.get("diagnostic_reason") != "EFA_TRAFFIC_HUNG_SUSPECTED":
            errors.append(f"diagnostic reason {item.get('diagnostic_reason')!r}")
        if item.get("detection_basis") != "ATTEMPT_SCOPED_RDMA_IDLE":
            errors.append(f"detection basis {item.get('detection_basis')!r}")
        if item.get("capture_process_state") is not True:
            errors.append("the hung chain did not request process state capture")
    return sorted(set(errors))


# --------------------------------------------------------------------------- #
# CRI Pod-log attribution: drivers
# --------------------------------------------------------------------------- #
def _records(
    ctx: VariantContext, since: datetime, job: ManagedApplogJob, needle: str
) -> dict[str, Any]:
    return applog_records(
        ctx.regional,
        ctx.settings.regional.cluster_id,
        ctx.settings.node,
        since,
        job_id=job.name,
        path_needle=needle,
    )


def _job(
    ctx: VariantContext, suffix: str, *, mode: str, restart_budget: int = 0
) -> ManagedApplogJob:
    name = run_identity(CASE, suffix, ctx.attempt, ctx.case_dir).replace(
        "gl017", "gl17"
    )
    job = ManagedApplogJob(ctx, name=name, mode=mode, restart_budget=restart_budget)
    ctx.cleanup.push(f"job-{name}", job.delete)
    job.start()
    ctx.evidence.write(f"{suffix}-job", job.identity())
    return job


def _wait_records(
    ctx: VariantContext,
    since: datetime,
    job: ManagedApplogJob,
    needle: str,
    *,
    accept: Callable[[dict[str, Any]], bool],
    timeout_seconds: float,
    name: str,
) -> dict[str, Any]:
    last: dict[str, Any] = {}

    def check() -> dict[str, Any] | None:
        nonlocal last
        last = _records(ctx, since, job, needle)
        return last if accept(last) else None

    value = ctx.wait_until(
        check, timeout_seconds=timeout_seconds, poll_seconds=15, name=name
    )
    records = value if value is not None else last
    ctx.evidence.write(f"{name}-records", records)
    return records


def _wait_container(
    ctx: VariantContext,
    job: ManagedApplogJob,
    *,
    attempt_id: str,
    pod_uid: str,
    accept: Callable[[dict[str, Any]], bool],
    name: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The attempt observation whose container for ``pod_uid`` satisfies ``accept``."""

    def ready(records: dict[str, Any]) -> bool:
        container = container_of(
            attempt_of(records, attempt_id), pod_uid, job.container
        )
        return container is not None and accept(container)

    records = _wait_records(
        ctx,
        ctx.clock(),
        job,
        "never-a-path",
        accept=ready,
        timeout_seconds=ATTEMPT_BUDGET,
        name=name,
    )
    attempt = attempt_of(records, attempt_id)
    container = container_of(attempt, pod_uid, job.container)
    if container is None or not accept(container):
        return records, None
    return records, container


def _open_logs(
    ctx: VariantContext, job: ManagedApplogJob, *, initial_tail_bytes: int = 0
) -> CriPodLogFixture:
    logs = cri_pod_logs(ctx)
    logs.create()
    opened = logs.open(
        container=job.container,
        approved=True,
        restore_seconds=ctx.spec.max_minutes * 60,
        initial_tail_bytes=initial_tail_bytes,
    )
    ctx.evidence.write("cri-window-open", opened)
    return logs


def _declare_pid(
    ctx: VariantContext,
    job: ManagedApplogJob,
    logs: CriPodLogFixture,
    pod: dict[str, Any],
) -> int:
    """Read the container's init PID from its cgroup and declare it on the Pod."""

    container_id = (pod.get("container") or {}).get("container_id")
    if not container_id:
        raise VariantInconclusive(f"Pod {pod.get('name')} has no container id yet")
    inspected = logs.inspect(str(pod["uid"]), container_id=str(container_id))
    ctx.evidence.write(f"inspect-{pod['uid']}", inspected)
    pid = int((inspected.get("process") or {}).get("init_pid") or 0)
    if pid < 1:
        raise VariantInconclusive("the container's init PID could not be read")
    job.annotate_host_pid(str(pod["name"]), pid)
    return pid


def _wait_entry(
    ctx: VariantContext,
    since: datetime,
    job: ManagedApplogJob,
    needle: str,
    paths: set[str],
    *,
    name: str,
    count: int = 1,
) -> dict[str, Any]:
    records = _wait_records(
        ctx,
        since,
        job,
        needle,
        accept=lambda value: len(entries_of(value, paths)) >= count,
        timeout_seconds=ENTRY_BUDGET,
        name=name,
    )
    if len(entries_of(records, paths)) < count:
        raise VariantInconclusive(
            "the node log collector delivered no record from the fixture file"
        )
    return records


def _wait_graded(
    ctx: VariantContext,
    since: datetime,
    job: ManagedApplogJob,
    needle: str,
    entry: dict[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    """The incident for ``entry`` and, once it exists, its settled workflow."""

    def graded(records: dict[str, Any]) -> bool:
        return incident_of(records, entry) is not None

    records = _wait_records(
        ctx,
        since,
        job,
        needle,
        accept=graded,
        timeout_seconds=DETECTION_BUDGET,
        name=name,
    )
    incident = incident_of(records, entry)
    if incident is None:
        return records
    ctx.cleanup.push(
        f"restore-{incident['incident_id']}",
        lambda: _restore(ctx, incident),
    )

    def done(value: dict[str, Any]) -> bool:
        current = incident_of(value, entry)
        return current is not None and settled(
            {"workflows": list((value.get("workflows") or {}).values())},
            [str(current["incident_id"])],
        )

    return _wait_records(
        ctx,
        since,
        job,
        needle,
        accept=done,
        timeout_seconds=SETTLE_BUDGET,
        name=f"{name}-settled",
    )


def _restore(ctx: VariantContext, incident: dict[str, Any]) -> list[dict[str, Any]]:
    """Close the diagnostic incident through the product's validated restore."""

    return restore_incidents(
        ctx, [incident], reason=f"{ctx.spec.key} reviewed application log restore"
    )


def _first_pod(job: ManagedApplogJob) -> dict[str, Any]:
    pods = job.pods()
    if not pods:
        raise VariantInconclusive("the managed job has no Pod")
    return job.pod_detail(str(pods[0]["name"]))


def _file_path(receipt: dict[str, Any]) -> str:
    path = str((receipt.get("file") or {}).get("path") or "")
    if not path:
        raise VariantInconclusive("the fixture file was not created")
    return path


def _needle(logs: CriPodLogFixture) -> str:
    from scripts.e2e.regional.probes.general_live_applog_probe import fixture_file_name

    return fixture_file_name(logs.run_id)


def _common_stages(
    ctx: VariantContext, records: dict[str, Any], workflow: dict[str, Any] | None
) -> dict[str, list[str]]:
    view = {"workflows": list((records.get("workflows") or {}).values())}
    return {
        "no_stop": stop_errors(workflow),
        "hardware": hardware_errors(view),
        "not_isolated": ["node isolated by an application line"]
        if node_isolated(ctx.regional.node_snapshot(ctx.settings.node))
        else [],
    }


def live_pid(ctx: VariantContext) -> dict[str, Any]:
    """A reviewed OOM line from a live container targets its genuine PID."""

    job = _job(ctx, "lp", mode="trainer")
    pod = _first_pod(job)
    logs = _open_logs(ctx, job)
    pid = _declare_pid(ctx, job, logs, pod)
    _, container = _wait_container(
        ctx,
        job,
        attempt_id=job.attempt_id,
        pod_uid=str(pod["uid"]),
        accept=lambda item: not item.get("terminated") and item.get("host_pid") == pid,
        name="live-observed",
    )
    if container is None:
        raise VariantInconclusive(
            "the completion watcher did not record the live container with its host PID"
        )
    created = logs.create_log(
        str(pod["uid"]),
        container=job.container,
        namespace=str(pod["namespace"]),
        pod_name=str(pod["name"]),
    )
    path = _file_path(created)
    logs.wait_tracked(ctx, path, name="live-tracked")
    since = ctx.clock()
    written = logs.write(
        str(pod["uid"]), container=job.container, records=[reviewed_line(CUDA_OOM_LINE)]
    )
    ctx.evidence.write("live-write", written)
    needle = _needle(logs)
    records = _wait_entry(ctx, since, job, needle, {path}, name="live-entry")
    entry = entries_of(records, {path})[0]
    records = _wait_graded(ctx, since, job, needle, entry, name="live-graded")
    incident = incident_of(records, entry)
    workflow = workflow_of(records, incident)
    stamp = (written.get("stamps") or [None])[0]
    stages = {
        "entry_identity": entry_identity_errors(
            entry, pod_uid=str(pod["uid"]), container=job.container, stamp=stamp
        ),
        "attribution": attribution_errors(
            incident,
            workflow,
            job_id=job.name,
            attempt_id=job.attempt_id,
            process_state="LIVE",
            node=ctx.settings.node,
            pid=pid,
        ),
        "diagnostic": diagnostic_chain_errors(incident, workflow, expect_bundle=True),
        **_common_stages(ctx, records, workflow),
        "job_alive": []
        if job.alive().get("alive")
        else ["the live job did not survive"],
    }
    return _result(
        stages,
        notes=[
            "a live container's line is OBSERVED_PROCESS/LIVE with its host PID targeted"
        ],
        extra={"attempt": job.identity(), "pod_uid": pod["uid"], "host_pid": pid},
    )


def exited_pid(ctx: VariantContext) -> dict[str, Any]:
    """A reviewed OOM line from an exited container is EXITED: no stack capture."""

    job = _job(ctx, "ep", mode="hold")
    pod = _first_pod(job)
    logs = _open_logs(ctx, job)
    pid = _declare_pid(ctx, job, logs, pod)
    _, live = _wait_container(
        ctx,
        job,
        attempt_id=job.attempt_id,
        pod_uid=str(pod["uid"]),
        accept=lambda item: not item.get("terminated") and item.get("host_pid") == pid,
        name="exited-live-observed",
    )
    if live is None:
        raise VariantInconclusive(
            "the completion watcher did not record the container with its host PID"
        )
    created = logs.create_log(
        str(pod["uid"]),
        container=job.container,
        namespace=str(pod["namespace"]),
        pod_name=str(pod["name"]),
    )
    path = _file_path(created)
    logs.wait_tracked(ctx, path, name="exited-tracked")
    ctx.evidence.write("exit-request", {"output": job.request_exit(str(pod["name"]))})
    terminal, container = _wait_container(
        ctx,
        job,
        attempt_id=job.attempt_id,
        pod_uid=str(pod["uid"]),
        accept=lambda item: bool(item.get("terminated")),
        name="exited-observed",
    )
    if container is None:
        raise VariantInconclusive("the completion watcher did not record the exit")
    observed_at = parse_time(
        (attempt_of(terminal, job.attempt_id) or {}).get("observed_at")
    )
    since = ctx.clock()
    written = logs.write(
        str(pod["uid"]), container=job.container, records=[reviewed_line(CUDA_OOM_LINE)]
    )
    ctx.evidence.write("exited-write", written)
    needle = _needle(logs)
    records = _wait_entry(ctx, since, job, needle, {path}, name="exited-entry")
    entry = entries_of(records, {path})[0]
    records = _wait_graded(ctx, since, job, needle, entry, name="exited-graded")
    incident = incident_of(records, entry)
    workflow = workflow_of(records, incident)
    stamp = (written.get("stamps") or [None])[0]
    collected = parse_time(entry.get("collected_at"))
    aged = (
        observed_at is not None
        and collected is not None
        and (collected - observed_at).total_seconds() > ATTRIBUTION_FRESHNESS_SECONDS
    )
    if incident is not None and incident.get("attempt_id") is None and aged:
        raise VariantInconclusive(
            "the terminal attempt observation aged past the product's "
            f"{ATTRIBUTION_FRESHNESS_SECONDS} s attribution window before the line was collected"
        )
    stages = {
        "entry_identity": entry_identity_errors(
            entry, pod_uid=str(pod["uid"]), container=job.container, stamp=stamp
        ),
        "attribution": attribution_errors(
            incident,
            workflow,
            job_id=job.name,
            attempt_id=job.attempt_id,
            process_state="EXITED",
            node=ctx.settings.node,
            pid=pid,
        ),
        "no_stack_capture": diagnostic_chain_errors(
            incident, workflow, expect_bundle=False
        ),
        **_common_stages(ctx, records, workflow),
    }
    return _result(
        stages,
        notes=[
            "the container exited 0 with its host PID declared; the line is EXITED and "
            "no bundle/stack capture is planned for the dead PID"
        ],
        extra={
            "attempt": job.identity(),
            "pod_uid": pod["uid"],
            "host_pid": pid,
            "exit_code": container.get("exit_code"),
        },
    )


def old_pod(ctx: VariantContext) -> dict[str, Any]:
    """A line under the previous Pod's UID never reaches the restarted attempt."""

    job = _job(ctx, "op", mode="fail_once", restart_budget=1)
    old = _first_pod(job)
    ctx.evidence.write("old-pod", old)
    replaced = job.wait_replaced(
        {str(old["uid"])}, timeout_seconds=RESTART_BUDGET, name="old-restart"
    )
    new_pods = [item for item in replaced if str(item["uid"]) != str(old["uid"])]
    if not new_pods or new_pods[0].get("phase") != "Running":
        raise VariantInconclusive(
            "the product did not restart the failed first attempt"
        )
    new = job.pod_detail(str(new_pods[0]["name"]))
    new_attempt = str(new.get("attempt_id") or "")
    if not new_attempt or new_attempt == job.attempt_id:
        raise VariantInconclusive(f"the restarted Pod carries attempt {new_attempt!r}")
    logs = _open_logs(ctx, job, initial_tail_bytes=OLD_POD_INITIAL_TAIL)
    new_pid = _declare_pid(ctx, job, logs, new)
    _, container = _wait_container(
        ctx,
        job,
        attempt_id=new_attempt,
        pod_uid=str(new["uid"]),
        accept=lambda item: not item.get("terminated")
        and item.get("host_pid") == new_pid,
        name="new-observed",
    )
    if container is None:
        raise VariantInconclusive("the new attempt was not observed running")
    needle = _needle(logs)
    since = ctx.clock()
    entry: dict[str, Any] | None = None
    path = ""
    written: dict[str, Any] = {}
    for round_index in range(OLD_POD_ATTEMPTS):
        # kubelet's GC removes a directory of a Pod it no longer knows within
        # a minute; the collector reads the whole file on its first visit, so
        # one poll is enough and a lost round is simply repeated.
        created = logs.create_log(
            str(old["uid"]),
            container=job.container,
            namespace=str(old["namespace"]),
            pod_name=str(old["name"]),
            allow_create_dir=True,
        )
        path = _file_path(created)
        written = logs.write(
            str(old["uid"]),
            container=job.container,
            records=[reviewed_line(CUDA_OOM_LINE)],
        )
        ctx.evidence.write(f"old-write-{round_index}", {"created": created, **written})
        try:
            records = _wait_entry(
                ctx, since, job, needle, {path}, name=f"old-entry-{round_index}"
            )
        except VariantInconclusive:
            ctx.evidence.write(f"old-lost-{round_index}", logs.status())
            continue
        entry = entries_of(records, {path})[0]
        break
    if entry is None:
        raise VariantInconclusive(
            "the collector never read the old Pod's fixture file (removed by kubelet first?)"
        )
    records = _wait_graded(ctx, since, job, needle, entry, name="old-graded")
    incident = incident_of(records, entry)
    workflow = workflow_of(records, incident)
    stages = {
        "entry_identity": entry_identity_errors(
            entry,
            pod_uid=str(old["uid"]),
            container=job.container,
            stamp=(written.get("stamps") or [None])[0],
        ),
        "not_new_attempt": new_attempt_errors(
            incident,
            workflow,
            new_attempt_id=new_attempt,
            new_pid=new_pid,
            node=ctx.settings.node,
        ),
        **_common_stages(ctx, records, workflow),
        "new_attempt_alive": []
        if job.alive().get("alive")
        else ["the restarted attempt did not survive the old Pod's line"],
    }
    attributed = (incident or {}).get("attempt_id")
    return _result(
        stages,
        notes=[
            "an old Pod UID resolves to the prior attempt or stays UNRESOLVED, never the "
            f"new attempt (attributed: {attributed or 'UNRESOLVED'})"
        ],
        extra={
            "attempt": job.identity(),
            "old_pod_uid": old["uid"],
            "new_attempt_id": new_attempt,
            "new_pod_uid": new["uid"],
            "attributed_attempt": attributed,
        },
    )


def cri_partial_rotation(ctx: VariantContext) -> dict[str, Any]:
    """A CRI P fragment then its F stitches across polls, a rename and a restart."""

    job = _job(ctx, "cp", mode="trainer")
    pod = _first_pod(job)
    logs = _open_logs(ctx, job)
    created = logs.create_log(
        str(pod["uid"]),
        container=job.container,
        namespace=str(pod["namespace"]),
        pod_name=str(pod["name"]),
    )
    path = _file_path(created)
    logs.wait_tracked(ctx, path, name="partial-tracked")
    since = ctx.clock()
    head = logs.write(
        str(pod["uid"]),
        container=job.container,
        records=[reviewed_line(PARTIAL_HEAD, flag="P")],
    )
    ctx.evidence.write("partial-head", head)

    def held(status: dict[str, Any]) -> dict[str, Any] | None:
        fragment = logs.pending_fragment(status, path)
        return status if fragment else None

    pending = ctx.wait_until(
        lambda: held(logs.status()),
        timeout_seconds=ENTRY_BUDGET,
        poll_seconds=5,
        name="partial-pending",
    )
    ctx.evidence.write("partial-pending", pending or {})
    rotated = logs.rotate(str(pod["uid"]), container=job.container)
    ctx.evidence.write("partial-rotate", rotated)
    restarted = logs.restart_collector()
    ctx.evidence.write("partial-restart", restarted)
    tail = logs.write(
        str(pod["uid"]),
        container=job.container,
        records=[reviewed_line(PARTIAL_TAIL, flag="F")],
        target="rotated",
    )
    ctx.evidence.write("partial-tail", tail)
    rotated_path = str((rotated.get("rotated") or {}).get("path") or path + ".1")
    paths = {path, rotated_path}
    needle = _needle(logs)
    records = _wait_entry(ctx, since, job, needle, paths, name="partial-entry")
    message = PARTIAL_HEAD + PARTIAL_TAIL
    delivered = entries_of(records, paths)
    whole = [item for item in delivered if item.get("message") == message]
    incident = workflow = None
    if whole:
        records = _wait_graded(ctx, since, job, needle, whole[0], name="partial-graded")
        delivered = entries_of(records, paths)
        incident = incident_of(records, whole[0])
        workflow = workflow_of(records, incident)
    stages = {
        "stitched": stitch_errors(
            delivered,
            message=message,
            rotated_path=rotated_path,
            head_stamp=(head.get("stamps") or [None])[0],
        ),
        "held_across_polls": []
        if pending is not None
        else ["the collector never recorded the P fragment as pending"],
        "entry_identity": entry_identity_errors(
            whole[0] if whole else None,
            pod_uid=str(pod["uid"]),
            container=job.container,
            stamp=(head.get("stamps") or [None])[0],
            message=message,
        ),
        "single_incident": []
        if incident is not None and incident.get("effective_action") == "NO_ACTION"
        else ["the stitched line did not produce one NO_ACTION evidence incident"],
        "attributed": []
        if incident is not None and incident.get("attempt_id") == job.attempt_id
        else [
            f"the stitched line was attributed to {(incident or {}).get('attempt_id')}"
        ],
        "evidence_only": no_diagnostic_errors(workflow),
        **_common_stages(ctx, records, workflow),
    }
    return _result(
        stages,
        notes=[
            "the P fragment was held across polls, a rename rotation and a collector "
            "restart (persisted state), then stitched with its F; the rejected-batch "
            "rollback of that state is pinned by the collector's unit tests, not induced live"
        ],
        extra={
            "attempt": job.identity(),
            "pod_uid": pod["uid"],
            "paths": sorted(paths),
        },
    )


def missing_flight_recorder(ctx: VariantContext) -> dict[str, Any]:
    """A live job without a flight recorder: the RDMA-idle trigger still fires."""

    since = ctx.clock()
    identity, records, stages = nccl.stalled_chain(ctx, "mfr", flight_recorder=False)
    stages["fr_not_configured"] = nccl.fr_errors(records, expect_configured=False)
    stages["independent_trigger"] = independent_trigger_errors(records)
    logs_view = activity(ctx, since, kinds=("NODE_LOGS",))
    log_incidents = [
        item["incident_id"]
        for item in logs_view.get("incidents") or []
        if item.get("policy_source") == "SITE_APPLICATION_LOG"
        and item.get("attempt_id") == identity.get("attempt_id")
    ]
    return _result(
        stages,
        notes=[
            "no flight recorder: the hung chain is raised by the attempt-scoped RDMA idle "
            "rule on live processes; triage records not_configured and fabricates no dump"
        ],
        extra={"attempt": identity, "application_log_incidents": log_incidents},
        proof_scope="native",
    )


# --------------------------------------------------------------------------- #
# results and specs
# --------------------------------------------------------------------------- #
def _result(
    stages: dict[str, list[str]],
    *,
    notes: list[str] | None = None,
    extra: dict[str, Any] | None = None,
    proof_scope: str = "live-replay",
) -> dict[str, Any]:
    return {
        "verdict": verdict(stages),
        "stages": stages,
        "errors": [
            f"{stage}: {error}" for stage, items in stages.items() for error in items
        ],
        "notes": notes or [],
        "proof_scope": proof_scope,
        **(extra or {}),
    }


def _needs(
    *, site: bool = False, peer: bool = False
) -> Callable[[GeneralLiveSettings, Any], list[str]]:
    # ``spec.preflight`` hooks take ``(settings, regional)`` since the 030 candidate
    # preflight (S15, 2026-10-26); live 017 live_pid/exited_pid crashed with
    # "check() takes 1 positional argument but 2 were given" when this one did not.
    def check(settings: GeneralLiveSettings, _regional: Any = None) -> list[str]:
        errors = []
        if site and settings.site_file is None:
            errors.append("a managed test job needs --site-file")
        if peer and not settings.peer_node:
            errors.append("a two-node NCCL job needs --node-b")
        return errors

    return check


def _spec(variant: str, injection: str, expected: str, execute: Any) -> VariantSpec:
    return VariantSpec(
        case_id=CASE,
        variant=variant,
        proof_mode=LIVE_REPLAY,
        lanes=("eks", "hp"),
        risk="L",
        flags="S",
        injection=injection,
        expected=expected,
        cleanup="close the journalctl wrapper, disable the node log collector, confirm clean",
        max_minutes=16,
        approvals=(ENABLE,),
        execute=execute,
    )


def _cri_spec(
    variant: str, injection: str, expected: str, execute: Any, *, max_minutes: int = 30
) -> VariantSpec:
    return VariantSpec(
        case_id=CASE,
        variant=variant,
        proof_mode=LIVE_REPLAY,
        lanes=("eks", "hp"),
        risk="M",
        flags="S",
        injection=injection,
        expected=expected,
        cleanup=(
            "remove the fixture CRI file(s), close the log-collector window (drop-in, "
            "runtime enable, deadman), restore the diagnostic incident, delete the "
            "managed job; confirm clean"
        ),
        max_minutes=max_minutes,
        approvals=(ENABLE,),
        execute=execute,
        preflight=_needs(site=True),
    )


VARIANTS = [
    _spec(
        "cgroup_oom",
        "reviewed kernel-transport 'Memory cgroup out of memory: Killed process' line "
        "(mm/oom_kill.c) via the node log collector journal source",
        "WARNING RUN_DIAGNOSTICS diagnostic; no isolation, no STOP without attribution, no hardware op",
        cgroup_oom,
    ),
    _spec(
        "nccl_error",
        "reviewed application 'NCCL WARN ... unhandled error' line (journal source)",
        "NO_ACTION evidence (completion watcher); no diagnostic workflow, no hardware marker",
        nccl_error,
    ),
    _spec(
        "distributed_error",
        "reviewed application 'torch.distributed ... timeout' line (journal source)",
        "NO_ACTION evidence; no diagnostic workflow, no hardware marker",
        distributed_error,
    ),
    _cri_spec(
        "live_pid",
        "a reviewed CUDA out-of-memory line in the CRI Pod log of a running managed "
        "trainer whose host PID the Pod declares",
        "the entry carries the genuine Pod UID/container and CRI time; the incident names "
        "the attempt; OBSERVED_PROCESS/LIVE with target_pids_by_node = the declared PID; "
        "the diagnostic bundle captures it; no STOP, no isolation; the job keeps running",
        live_pid,
    ),
    _cri_spec(
        "exited_pid",
        "the same line in the CRI Pod log of a managed container that exited 0 with its "
        "host PID declared",
        "attributed to the attempt as EXITED; no COLLECT_DIAGNOSTIC_BUNDLE, no "
        "target_pids_by_node, no stack capture of the dead PID; no STOP",
        exited_pid,
    ),
    _cri_spec(
        "old_pod",
        "a managed job whose first attempt fails and is restarted by the product; the "
        "line is written under the previous Pod's UID while the new attempt runs",
        "the line resolves to the prior attempt or UNRESOLVED, never to the new attempt; "
        "the new attempt's PID is never targeted; the new attempt keeps running",
        old_pod,
        max_minutes=45,
    ),
    _cri_spec(
        "cri_partial_rotation",
        "a CRI P fragment, then a rename rotation and a collector restart, then the F "
        "fragment on the rotated inode",
        "one stitched record at the P fragment's time from the rotated file; no fragment "
        "delivered alone; one NO_ACTION evidence incident attributed to the attempt",
        cri_partial_rotation,
    ),
    VariantSpec(
        case_id=CASE,
        variant="missing_flight_recorder",
        proof_mode=NATIVE,
        lanes=("eks", "hp"),
        risk="M",
        flags="S",
        injection=(
            "a managed two-node NCCL job with no flight recorder (trace buffer 0) stalls "
            "at zero EFA traffic with every process alive"
        ),
        expected=(
            "ZERO_WARNING then HUNG_SUSPECTED from the attempt-scoped RDMA idle rule; "
            "the read-only diagnostic DAG runs with process capture; triage records "
            "not_configured; ESCALATED; no STOP/isolation; traffic resumes on release"
        ),
        cleanup=(
            "release the stall and prove the EFA pair resumes; delete the test Job; "
            "confirm no node was isolated/stopped/reset"
        ),
        max_minutes=65,
        execute=missing_flight_recorder,
        preflight=_needs(site=True, peer=True),
    ),
]
