"""GF-GENERAL-LIVE-017 application error logs.

Two reviewed sources feed the node log collector's real rules
(``log_rules`` / ``log_recovery_context``):

* **journal replay** (``cgroup_oom``, ``nccl_error``, ``distributed_error``):
  the kernel-log replay helper in journal mode appends reviewed records to the
  real ``journalctl`` output. An out-of-memory line is a WARNING
  ``RUN_DIAGNOSTICS`` from any origin; NCCL and distributed-training errors are
  ``NO_ACTION`` evidence owned by the completion watcher -- never a hardware
  marker or an extra diagnostic workflow, and never an incident either
  (``app/ingest/node_health.py`` keeps a ``SITE_APPLICATION_LOG`` NO_ACTION
  finding as the NODE_LOGS record only, so the journal variants' incident wait
  runs out by design and their stages check that nothing adverse happened).
  The OOM line is the kernel's own
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
  attempt, the diagnostic bundle step's attribution parameters
  (``attribution``, ``process_state``, ``target_pids_by_node``) and the Node
  Agent's per-node bundle receipt (``details.node_results[<node>]``): a live
  process is targeted and its bundle ran the process capture, an exited one is
  never stack-sampled, an old Pod never reaches the new attempt, and a
  ``P``/``F`` record stitches across polls, a rename rotation and a collector
  restart. Before a window is opened the probe's read-only ``preflight`` must
  show the node has what the collector binds by (``/var/log/pods``,
  kubelet-shaped Pod directories, the installed unit) or the variant is
  ``NOT_RUN(CAPABILITY_ABSENT)`` naming the gap. ``old_pod`` seeds the old
  Pod's directory, file and line in ONE probe call right after kubelet's
  log-directory GC tick (``seed-log``); a round the GC still takes
  (``HostProbeError.reason`` in ``GC_LOST_REASONS``) is repeated, any other
  probe refusal is the failure its reason token names.

``missing_flight_recorder`` is native: a managed two-node NCCL job without a
flight recorder (``ManagedNcclJob(flight_recorder=False)``) stalls at zero EFA
traffic while every process stays alive, and the product's independent
RDMA-idle trigger (``ATTEMPT_SCOPED_RDMA_IDLE``) must raise the hung chain with
``not_configured`` flight-recorder evidence -- no log line is involved. The
chain's process capture is read where the product puts it: requested on the
COLLECT_HUNG_TRIAGE steps and shown by the rank rows the agents returned; the
bundle step's ``capture_process_state`` is the triage decision's targetability
(``hung_bundle.bind_hung_bundle``), conservative without a flight recorder.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from scripts.e2e.regional import general_live_nccl as nccl
from scripts.e2e.regional.general_live_applog_fixture import (
    CUDA_OOM_LINE,
    GC_LOST_REASONS,
    PARTIAL_HEAD,
    PARTIAL_TAIL,
    CriPodLogFixture,
    capability_gaps,
    cri_pod_logs,
    reviewed_line,
)
from scripts.e2e.regional.general_live_applog_hung import independent_trigger_errors
from scripts.e2e.regional.general_live_applog_job import ManagedApplogJob
from scripts.e2e.regional.general_live_applog_verdicts import (
    DIAGNOSTIC_WORKFLOW_OPS,
    attributable_errors,
    attribution_errors,
    attempt_of,
    bundle_node_result,
    container_of,
    diagnostic_chain_errors,
    entries_of,
    entry_identity_errors,
    evidence_only_errors,
    incident_of,
    new_attempt_errors,
    stitch_errors,
    stop_errors,
    workflow_of,
)
from scripts.e2e.regional.general_live_common import activity, restore_incidents
from scripts.e2e.regional.general_live_fixture import (
    GeneralLiveSettings,
    VariantContext,
    VariantInconclusive,
    VariantNotRun,
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
    verdict,
    workflows_of,
)
from scripts.e2e.regional.host_probe_fixture import HostProbeError

CASE = "GF-GENERAL-LIVE-017"
ENABLE = "ENABLE_NODE_LOG_COLLECTOR"
DETECTION_BUDGET = 420
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
#: ``old_pod``: how long one ``seed-log`` call may watch the old Pod's directory
#: for kubelet's container GC tick before seeding -- kubelet's
#: ``ContainerGCPeriod`` is one minute, so one period plus a margin. Seeded
#: right after the tick, the file lives the whole next period; seeded between
#: two probe calls it lived 6 s live (HyperPod, 2026-10-26).
OLD_POD_GC_WAIT_SECONDS = 75


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


def _cri_logs(ctx: VariantContext) -> CriPodLogFixture:
    """The CRI Pod-log fixture, after the node proved it has what it binds by.

    The probe's ``preflight`` is read-only; a gap it shows (no ``/var/log/pods``,
    no kubelet-shaped Pod directory under it, the collector unit not installed)
    is ``NOT_RUN(CAPABILITY_ABSENT)`` naming it, before the managed job exists
    and before anything on the node is touched.
    """

    logs = cri_pod_logs(ctx)
    logs.create()
    preflight = logs.preflight()
    ctx.evidence.write("cri-preflight", preflight)
    gaps = capability_gaps(preflight)
    if gaps:
        raise VariantNotRun("CAPABILITY_ABSENT", "; ".join(gaps))
    return logs


def _open(
    ctx: VariantContext,
    logs: CriPodLogFixture,
    job: ManagedApplogJob,
    *,
    initial_tail_bytes: int = 0,
) -> dict[str, Any]:
    opened = logs.open(
        container=job.container,
        approved=True,
        restore_seconds=ctx.spec.max_minutes * 60,
        initial_tail_bytes=initial_tail_bytes,
    )
    ctx.evidence.write("cri-window-open", opened)
    return opened


def _declare_pid(
    ctx: VariantContext,
    job: ManagedApplogJob,
    logs: CriPodLogFixture,
    pod: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    """Read the container's init PID from its cgroup and declare it on the Pod.

    Returns the PID and the probe's read-only ``inspect`` receipt (the Pod's
    real log directories under the collector's root, its process rows).
    """

    container_id = (pod.get("container") or {}).get("container_id")
    if not container_id:
        raise VariantInconclusive(f"Pod {pod.get('name')} has no container id yet")
    inspected = logs.inspect(str(pod["uid"]), container_id=str(container_id))
    ctx.evidence.write(f"inspect-{pod['uid']}", inspected)
    pid = int((inspected.get("process") or {}).get("init_pid") or 0)
    if pid < 1:
        raise VariantInconclusive("the container's init PID could not be read")
    job.annotate_host_pid(str(pod["name"]), pid)
    return pid, inspected


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

    logs = _cri_logs(ctx)
    job = _job(ctx, "lp", mode="trainer")
    pod = _first_pod(job)
    _open(ctx, logs, job)
    pid, _ = _declare_pid(ctx, job, logs, pod)
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
        "diagnostic": diagnostic_chain_errors(
            incident, workflow, expect_bundle=True, node=ctx.settings.node
        ),
        **_common_stages(ctx, records, workflow),
        "job_alive": []
        if job.alive().get("alive")
        else ["the live job did not survive"],
    }
    # The agent's receipt travels with the result so a reviewer reads what the
    # bundle holds (archive, digest, capture counts) without the Store.
    bundle = bundle_node_result(
        final_execution(workflow, "COLLECT_DIAGNOSTIC_BUNDLE") if workflow else None,
        ctx.settings.node,
    )
    summary = (bundle or {}).get("manifest_summary") or {}
    failed = summary.get("failed_capture_count")
    return _result(
        stages,
        notes=[
            "a live container's line is OBSERVED_PROCESS/LIVE with its host PID targeted",
            *(
                [
                    f"{failed} of {summary.get('capture_count')} bundle captures on "
                    f"{ctx.settings.node} failed; the archive names them"
                ]
                if isinstance(failed, int) and failed > 0
                else []
            ),
        ],
        extra={
            "attempt": job.identity(),
            "pod_uid": pod["uid"],
            "host_pid": pid,
            "diagnostic_bundle": bundle,
        },
    )


def exited_pid(ctx: VariantContext) -> dict[str, Any]:
    """A reviewed OOM line from an exited container is EXITED: no stack capture."""

    logs = _cri_logs(ctx)
    job = _job(ctx, "ep", mode="hold")
    pod = _first_pod(job)
    _open(ctx, logs, job)
    pid, _ = _declare_pid(ctx, job, logs, pod)
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

    logs = _cri_logs(ctx)
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
    # Read-only first: the new Pod's real log directory proves kubelet keeps
    # this node's Pod logs where the collector binds them, which is the layout
    # the old Pod's directory is recreated in below. Then the window.
    new_pid, inspected = _declare_pid(ctx, job, logs, new)
    if not inspected.get("directories"):
        raise VariantInconclusive(
            f"kubelet made no log directory for the new Pod {new.get('name')} under "
            "the collector's root; the old Pod's directory cannot be placed by its "
            "convention"
        )
    _open(ctx, logs, job, initial_tail_bytes=OLD_POD_INITIAL_TAIL)
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
    rounds = 0
    for round_index in range(OLD_POD_ATTEMPTS):
        # kubelet's GC removes a directory of a Pod it no longer knows at every
        # one-minute tick. One probe call seeds directory, file and line right
        # after the tick it watched for, so the file is there for the
        # collector's next poll, which reads the whole file on its first visit
        # (OLD_POD_INITIAL_TAIL). A round the GC still takes -- the probe says
        # so with its reason token -- is simply repeated; every other refusal is
        # the failure its token names (live 2026-10-26 the two-call version lost
        # the file between create-log and write and read "output withheld").
        rounds = round_index + 1
        try:
            written = logs.seed_log(
                str(old["uid"]),
                container=job.container,
                namespace=str(old["namespace"]),
                pod_name=str(old["name"]),
                records=[reviewed_line(CUDA_OOM_LINE)],
                allow_create_dir=True,
                await_gc_seconds=OLD_POD_GC_WAIT_SECONDS,
            )
        except HostProbeError as failure:
            if failure.reason not in GC_LOST_REASONS:
                raise
            ctx.evidence.write(
                f"old-lost-{round_index}",
                {"stage": "seed", "reason": failure.reason, "error": str(failure)},
            )
            continue
        path = _file_path(written)
        ctx.evidence.write(f"old-seed-{round_index}", written)
        try:
            records = _wait_entry(
                ctx, since, job, needle, {path}, name=f"old-entry-{round_index}"
            )
        except VariantInconclusive:
            ctx.evidence.write(
                f"old-lost-{round_index}", {"stage": "read", "status": logs.status()}
            )
            continue
        entry = entries_of(records, {path})[0]
        break
    if entry is None:
        raise VariantInconclusive(
            "the collector never read the old Pod's fixture file in "
            f"{OLD_POD_ATTEMPTS} rounds (kubelet's GC took each one first?)"
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
    gc = written.get("gc") or {}
    return _result(
        stages,
        notes=[
            "an old Pod UID resolves to the prior attempt or stays UNRESOLVED, never the "
            f"new attempt (attributed: {attributed or 'UNRESOLVED'})",
            "the old Pod's directory, file and line were seeded in one probe call "
            + (
                f"{gc.get('after_seconds')} s after kubelet's log-directory GC tick"
                if gc.get("observed")
                else "without seeing kubelet's log-directory GC tick in "
                f"{OLD_POD_GC_WAIT_SECONDS} s"
            )
            + f" (round {rounds} of {OLD_POD_ATTEMPTS})",
        ],
        extra={
            "attempt": job.identity(),
            "old_pod_uid": old["uid"],
            "new_attempt_id": new_attempt,
            "new_pod_uid": new["uid"],
            "attributed_attempt": attributed,
            "seed_rounds": rounds,
            "gc": gc,
        },
    )


def cri_partial_rotation(ctx: VariantContext) -> dict[str, Any]:
    """A CRI P fragment then its F stitches across polls, a rename and a restart."""

    logs = _cri_logs(ctx)
    job = _job(ctx, "cp", mode="trainer")
    pod = _first_pod(job)
    _open(ctx, logs, job)
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
    entry = whole[0] if whole else None
    # The stitched line is a NO_ACTION application-log rule: the product keeps
    # it as the NODE_LOGS record and mints no incident (evidence_only_errors),
    # so there is nothing to wait for once the record is in the store, and the
    # attribution is read off the record and the attempt observation.
    workflow = workflow_of(records, incident_of(records, entry)) if entry else None
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
            entry,
            pod_uid=str(pod["uid"]),
            container=job.container,
            stamp=(head.get("stamps") or [None])[0],
            message=message,
        ),
        "attributable": attributable_errors(
            records,
            entry,
            attempt_id=job.attempt_id,
            pod_uid=str(pod["uid"]),
            container=job.container,
            node=ctx.settings.node,
        ),
        "evidence_only": evidence_only_errors(records, entry),
        **_common_stages(ctx, records, workflow),
    }
    return _result(
        stages,
        notes=[
            "the P fragment was held across polls, a rename rotation and a collector "
            "restart (persisted state), then stitched with its F; the rejected-batch "
            "rollback of that state is pinned by the collector's unit tests, not induced live",
            "a NO_ACTION application-log line mints no incident, marker or workflow "
            "(node_health.ingest): the delivered record is the evidence and names the "
            "attempt's live Pod UID/container",
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
        "line is written under the previous Pod's UID while the new attempt runs "
        "(directory, file and line seeded in one probe call right after kubelet's "
        "log-directory GC tick; a round the GC still takes is repeated)",
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
        "delivered alone; the record names the attempt's live Pod UID/container; no "
        "incident, marker or workflow for a NO_ACTION application-log line (kept as the "
        "NODE_LOGS record only)",
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
