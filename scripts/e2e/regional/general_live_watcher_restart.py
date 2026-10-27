"""GF-GENERAL-LIVE-029 ``worker_restart``: the completion watcher is replaced
inside the restart transaction of a managed job, timed from the source failure.

The transaction is the product's. The completion watcher observes the failed
critical container and publishes the failure (a passive-containment incident:
FREEZE_EVIDENCE, STOP_WORKLOADS) and, once the attempt's Pods are gone, its
terminal; the control plane decides a budgeted restart (PLAN_CREATED) and opens
the recovery workflow; the executor's claim reserves the RESTART_WORKLOAD step
on the job's durable ledger (``restart_budget_preflight.reserve_restart_
budgets``, ``<workflow>/<index>/RESTART_WORKLOAD``) and the data plane restarts
the job as a copy named ``<job>-r-<8 hex>``. What the variant proves is that
replacing the watcher anywhere inside that transaction changes none of its
counts: the replacement restores its attempt state from the ``-outbox-active``
ConfigMap, replays the write-ahead outbox and re-observes every attempt from
the Pods, and neither path re-decides the terminal, re-reserves or releases the
restart, or re-sends what was sent. Landed before the watcher published the
failure, the replacement publishes it (from the outbox or its own observation)
and must still yield exactly one decision, one reservation and one restart;
landed after, the replacement's re-observation must not re-publish, re-decide
or re-reserve. Both are the product's claims.

Why the delete is timed from the source failure and not from an "open claim":
the reservation is open for one control-loop round trip only. The claim mints
the data-plane command in the tick that reserved (``regional.py:945-1020``),
the cluster executor is woken by the command's NOTIFY, creates the restart copy
and returns SUCCEEDED in that same call -- ``adapters/kubernetes/workload_
operations._set_workloads`` (156-235) reads no Pod of the copy; it waits
neither for scheduling nor for Running -- and the command's terminal status
wakes the dispatcher (``execution/dispatcher_wakeups.WAKING_REMOTE_COMMAND_
STATUSES``), whose executor records the step SUCCEEDED. Live that is one to a
few seconds (HyperPod, 2026-10-27: opened and closed between two records reads
10 s apart), shorter than one records read, so no poll could gate on it and no
start barrier on the restart copy's Pod could widen it. The transaction as a
whole, from the failure to the settled recovery, is tens of seconds, and its
start is visible without the store, through the GPU-cluster API at a
one-second cadence (``wait_source_failure``): the source Pod's terminated
critical container or Failed phase, the Pod's disappearance after it was seen
Running, or the PyTorchJob's own Failed (or Succeeded) condition -- whichever
is observed first. The disappearance counts because the product removes the
evidence within a second of the publish: the containment's STOP_WORKLOADS
deletes the attempt's Pods with no grace (``workload_operations._delete_
terminating_pods``), so live (HyperPod, 2026-10-27 12:40) the source went from
Running straight to absent between two one-second reads and no terminated
state was ever listed. The job's Failed condition (the training operator's
``PyTorchJobFailed``) outlives the Pod and also covers a Pod that fails before
it was ever seen Running. The Pods carry the ``gpu-fault.io/job-id`` and
``gpu-fault.io/attempt-id`` labels the product's submit path stamps on every
template (``training_submit_cli._inject_template``) and the PyTorchJob the
same on its metadata, so one labelled ``get pod,pytorchjob`` reads both per
tick. The runner deletes the one completion-watcher Pod at that moment
(``DELETE_OFFSET_SECONDS`` after the first observation), fenced on its UID and
resourceVersion with no grace, so the Deployment (``replicas: 1``, Recreate)
replaces it -- inside the transaction by construction. The job runs in the
``exit_once`` mode with its restart copy bounded to ``COPY_RUN_SECONDS``: the
copy must still be there for the replacement to observe (the ``exit`` mode's
copy died 30 s after its start and the product's second containment deleted
its Pod -- a race against the replacement's cold start that ``attempt_once``
would lose), and it exits 0, so its end is a terminal reading the replacement
publishes after it is Ready -- the write ``watcher_restored`` needs -- with no
second failure, escalation or notification on the job.

Where the delete landed is dated afterwards from the product's own records
(``transaction_timeline``: the containment incident's ``created_at`` is the
publish, the recovery incident's ``created_at`` the decision, the recovery
workflow's CLAIM event the reservation, the terminal RESTART_WORKLOAD record's
``updated_at`` the restart's end and the workflow's terminal ``updated_at`` the
settle) and the delete request's bracket is classified (``classify_delete``):
BEFORE_PUBLISH, BETWEEN_PUBLISH_AND_RESTART, AFTER_RESTART_BEFORE_SETTLE or
AFTER_SETTLE. Only the last is INCONCLUSIVE -- the replacement acted on a
finished transaction -- as are a fenced delete that did not happen and a
replacement that never became Ready (both ``RegionalFixtureError``); everything
else is graded by the records oracle: the ledger (count 1 == one reservation,
belonging to the recovery workflow), one successful RESTART_WORKLOAD, one
successful STOP_WORKLOADS, the PLAN_CREATED decision, no notification twice,
exactly one restart attempt observed besides the source, and a write from the
replacement (its coverage heartbeat once the cluster is idle, or its
observation of the restart attempt).

The settle wait after the replacement is bounded by the product's own caps
(``settle_bound``): the replacement may need its whole startupProbe window
before it publishes, the containment's STOP_WORKLOADS and the recovery's
RESTART_WORKLOAD may each wait up to the step cap, each workflow's claim may
wait one fallback dispatcher poll, the replacement's first pass one watcher
poll, plus a margin for record propagation and the read itself.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from gpu_fault.execution.config import (
    ProductionExecutorConfig,
    WorkflowDispatcherConfig,
)
from gpu_fault.models import WorkflowOperation
from scripts.e2e.regional.collector_action_guard import require_action_time
from scripts.e2e.regional.general_live_common import STORE_POLL_SECONDS
from scripts.e2e.regional.general_live_fixture import VariantContext
from scripts.e2e.regional.general_live_network_isolation import (
    WATCHER_APP,
    single_action_errors,
    watcher_pod,
)
from scripts.e2e.regional.general_live_verdicts import TERMINAL_WORKFLOWS, executions
from scripts.e2e.regional.regional_commands import RegionalFixtureError

#: The Deployment's startupProbe gives a cold watcher ten minutes.
REPLACEMENT_TIMEOUT_SECONDS = 600
REPLACEMENT_POLL_SECONDS = 10
#: The pause between the first API observation of the source failure and the
#: delete request. Zero: the watcher's own poll (``WATCHER_POLL_SECONDS``)
#: already decides on which side of the publish the delete lands, and every
#: second added only moves the landing towards the end of a transaction that
#: is tens of seconds long on a warm cluster.
DELETE_OFFSET_SECONDS = 0.0
#: Source Pod reads through the GPU-cluster API: every second once the Pod
#: runs (the ``exit`` mode ends it 30 s later), every five while it is still
#: starting (prewarm, pull, scheduling). No store read before the delete.
SOURCE_POLL_SECONDS = 1.0
SOURCE_START_POLL_SECONDS = 5.0
#: The Pod start plus the 30 s run; the attempt budget the other managed-job
#: variants give the product is the ceiling.
SOURCE_FAILURE_TIMEOUT_SECONDS = 900
#: How long the ``exit_once`` restart copy runs before it exits 0 (the
#: ``GFRS_COPY_RUN_SECONDS`` knob of ``general_live_attempts``): long enough
#: for the replacement watcher's cold start to observe it (the startupProbe
#: allows 600 s; a warm start takes well under a minute), short enough that
#: its terminal reading lands inside the 300 s watcher-restored wait that
#: starts when the replacement is Ready.
COPY_RUN_SECONDS = 240
#: What ends the source attempt as the API shows it, in the order one tick
#: reports them when several hold at once.
POD_TERMINATED = "pod-terminated"
JOB_ENDED = "job-ended"
POD_ABSENT = "pod-absent"
SOURCE_END_SIGNALS = (POD_TERMINATED, JOB_ENDED, POD_ABSENT)
#: ``KubernetesCompletionController.poll_interval_seconds`` default
#: (``completion_controller.py:146``; the deployment's
#: ``GPU_FAULT_WATCHER_POLL_SECONDS`` at :573 defaults to the same).
WATCHER_POLL_SECONDS = 5.0
#: Record propagation (the probe reads the store the executor just wrote) and
#: the settle read itself.
SETTLE_MARGIN_SECONDS = 120.0
#: The step statuses that end a RESTART_WORKLOAD execution.
TERMINAL_STEPS = frozenset({"SUCCEEDED", "FAILED"})
JOB_ID_LABEL = "gpu-fault.io/job-id"
ATTEMPT_ID_LABEL = "gpu-fault.io/attempt-id"
#: Where the delete landed, by the product's own records.
BEFORE_PUBLISH = "BEFORE_PUBLISH"
BETWEEN_PUBLISH_AND_RESTART = "BETWEEN_PUBLISH_AND_RESTART"
AFTER_RESTART_BEFORE_SETTLE = "AFTER_RESTART_BEFORE_SETTLE"
AFTER_SETTLE = "AFTER_SETTLE"
UNDATED = "UNDATED"
LANDINGS = (
    BEFORE_PUBLISH,
    BETWEEN_PUBLISH_AND_RESTART,
    AFTER_RESTART_BEFORE_SETTLE,
    AFTER_SETTLE,
    UNDATED,
)


def settle_bound() -> dict[str, float]:
    """How long after the fenced delete the recovery may take to settle.

    Every term is the product's shipped default, read from the product rather
    than copied: ``ProductionExecutorConfig.step_waiting_timeout_seconds``
    (``execution/config.py:293``) is the cap of one WAITING step and applies
    to STOP_WORKLOADS and RESTART_WORKLOAD alike (no override for either in
    ``_adapter_owned_step_overrides``, :175-182; ``step_waiting_limit``
    :329-335); ``WorkflowDispatcherConfig.poll_interval_seconds``
    (:522) is the fallback scan when a wakeup is lost, paid once per claim
    (containment, recovery). The replacement's startupProbe window is the
    harness's own ``REPLACEMENT_TIMEOUT_SECONDS``; the watcher poll and the
    margin are named above. Measured from the delete, so a replacement that
    took long leaves the settle correspondingly less.
    """

    executor = ProductionExecutorConfig(
        enabled=False, executor_id="general-live-029", allowed_operations=frozenset()
    )
    dispatcher = WorkflowDispatcherConfig(enabled=False)
    terms = {
        "replacement_timeout_seconds": float(REPLACEMENT_TIMEOUT_SECONDS),
        "stop_workloads_cap_seconds": float(
            executor.step_waiting_limit(WorkflowOperation.STOP_WORKLOADS)
        ),
        "restart_workload_cap_seconds": float(
            executor.step_waiting_limit(WorkflowOperation.RESTART_WORKLOAD)
        ),
        "dispatcher_polls_seconds": 2 * float(dispatcher.poll_interval_seconds),
        "watcher_poll_seconds": WATCHER_POLL_SECONDS,
        "margin_seconds": SETTLE_MARGIN_SECONDS,
    }
    return {**terms, "total_seconds": sum(terms.values())}


# --------------------------------------------------------------------------- #
# the source failure, read through the GPU-cluster API
# --------------------------------------------------------------------------- #
def _pod_record(item: Mapping[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") or {}
    status = item.get("status") or {}
    containers = []
    for container in status.get("containerStatuses") or []:
        terminated = (container.get("state") or {}).get("terminated") or {}
        containers.append(
            {
                "name": container.get("name"),
                "terminated": bool(terminated),
                "exit_code": terminated.get("exitCode"),
                "reason": terminated.get("reason"),
                "finished_at": terminated.get("finishedAt"),
            }
        )
    phase = status.get("phase")
    return {
        "name": metadata.get("name"),
        "uid": metadata.get("uid"),
        "node": (item.get("spec") or {}).get("nodeName"),
        "phase": phase,
        "containers": containers,
        "ended": phase in {"Failed", "Succeeded"}
        or any(entry["terminated"] for entry in containers),
    }


def _job_record(item: Mapping[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") or {}
    status = item.get("status") or {}
    conditions = [
        {
            "type": condition.get("type"),
            "status": condition.get("status"),
            "reason": condition.get("reason"),
            "message": condition.get("message"),
            "last_transition_time": condition.get("lastTransitionTime"),
        }
        for condition in status.get("conditions") or []
        if isinstance(condition, dict)
    ]
    held = [
        entry for entry in conditions if str(entry.get("status", "")).lower() == "true"
    ]
    return {
        "name": metadata.get("name"),
        "uid": metadata.get("uid"),
        "conditions": conditions,
        "replica_statuses": status.get("replicaStatuses") or {},
        "ended": any(entry["type"] in {"Failed", "Succeeded"} for entry in held),
    }


def source_objects(regional: Any, job_id: str, attempt_id: str) -> dict[str, Any]:
    """The source attempt's Pods and its PyTorchJob as the API shows them, in
    one labelled read (no store read): per Pod the phase and each container's
    termination, per job its conditions. The restart copy carries another
    attempt id on both and is left out."""

    value = json.loads(
        regional.kubectl(
            "gpu",
            "get",
            "pod,pytorchjob",
            "-l",
            f"{JOB_ID_LABEL}={job_id}",
            "-o",
            "json",
        )
    )
    pods: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for item in value.get("items") or []:
        labels = (item.get("metadata") or {}).get("labels") or {}
        if labels.get(ATTEMPT_ID_LABEL) != attempt_id:
            continue
        kind = str(item.get("kind") or "")
        if kind == "Pod":
            pods.append(_pod_record(item))
        elif kind == "PyTorchJob":
            jobs.append(_job_record(item))
    return {
        "pods": sorted(pods, key=lambda entry: str(entry["name"])),
        "job": jobs[0] if jobs else None,
    }


def source_pods(regional: Any, job_id: str, attempt_id: str) -> list[dict[str, Any]]:
    """The Pod half of ``source_objects``."""

    return source_objects(regional, job_id, attempt_id)["pods"]


def _source_state(pods: Sequence[Mapping[str, Any]]) -> str:
    if not pods:
        return "absent"
    return ",".join(
        f"{pod['phase'] or 'Unknown'}"
        + (
            "(exit "
            + "/".join(
                str(item["exit_code"])
                for item in pod["containers"]
                if item["terminated"]
            )
            + ")"
            if any(item["terminated"] for item in pod["containers"])
            else ""
        )
        for pod in pods
    )


def _job_state(job: Mapping[str, Any] | None) -> str:
    if job is None:
        return "absent"
    held = [
        f"{item['type']}({item['reason']})" if item.get("reason") else str(item["type"])
        for item in job.get("conditions") or []
        if str(item.get("status", "")).lower() == "true"
    ]
    return ",".join(held) if held else "no condition"


def wait_source_failure(
    ctx: VariantContext,
    *,
    job_id: str,
    attempt_id: str,
    timeout_seconds: float = SOURCE_FAILURE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """The first API observation of the source attempt's end.

    Reads the job's Pods and its PyTorchJob together (``source_objects``)
    every ``SOURCE_POLL_SECONDS`` once a Pod is listed (every
    ``SOURCE_START_POLL_SECONDS`` before), never the store, and returns at the
    first tick that shows any of ``SOURCE_END_SIGNALS``: a Pod whose critical
    container has terminated or whose phase is Failed/Succeeded
    (``pod-terminated``), the job holding a Failed or Succeeded condition
    (``job-ended``), or no Pod listed after one was seen Running
    (``pod-absent``: the containment's STOP deleted it before a read could
    list the terminated state). Every change of either reading is written to
    ``source-failure-timeline`` with the runner's clock and which side changed;
    the document names the trigger, every signal the tick showed, the kubelet's
    ``finishedAt`` and the job's conditions, and the last Running reading. A
    source that shows none of them within the bound is a fixture failure that
    names the last Pod state and the last job condition.
    """

    start = ctx.clock()
    timeline: list[dict[str, Any]] = []
    last: tuple[str, str] | None = None
    seen_running = False
    last_running_at: datetime | None = None
    polls = 0
    while True:
        objects = source_objects(ctx.regional, job_id, attempt_id)
        now = ctx.clock()
        polls += 1
        pods, job = objects["pods"], objects["job"]
        pod_state, job_state = _source_state(pods), _job_state(job)
        if (pod_state, job_state) != last:
            previous = last or (None, None)
            timeline.append(
                {
                    "observed_at": now.isoformat(),
                    "pod": pod_state,
                    "job": job_state,
                    "changed": [
                        name
                        for name, value, before in (
                            ("pod", pod_state, previous[0]),
                            ("job", job_state, previous[1]),
                        )
                        if value != before
                    ],
                }
            )
            ctx.evidence.write("source-failure-timeline", {"entries": timeline})
            last = (pod_state, job_state)
        if any(pod["phase"] == "Running" for pod in pods):
            seen_running = True
            last_running_at = now
        signals = []
        if any(pod["ended"] for pod in pods):
            signals.append(POD_TERMINATED)
        if job is not None and job["ended"]:
            signals.append(JOB_ENDED)
        if not pods and seen_running:
            signals.append(POD_ABSENT)
        if signals:
            return {
                "observed_at": now.isoformat(),
                "waited_seconds": round((now - start).total_seconds(), 3),
                "polls": polls,
                "trigger": signals[0],
                "signals": signals,
                "pod_state": pod_state,
                "job_state": job_state,
                "pods": pods,
                "job": job,
                "last_running_at": _iso(last_running_at),
            }
        if (now - start).total_seconds() >= timeout_seconds:
            raise RegionalFixtureError(
                f"the source attempt {attempt_id} did not end within "
                f"{timeout_seconds:.0f} s (last Pod state {pod_state}; last job "
                f"condition {job_state})"
            )
        ctx.sleep(
            SOURCE_POLL_SECONDS if pods or seen_running else SOURCE_START_POLL_SECONDS
        )


# --------------------------------------------------------------------------- #
# the fenced delete and the replacement
# --------------------------------------------------------------------------- #
def delete_watcher_pod(regional: Any, pod: Mapping[str, Any]) -> None:
    """Delete ``pod`` fenced on its UID and resourceVersion, no grace."""

    if not pod.get("uid") or not pod.get("resource_version"):
        raise RegionalFixtureError("the watcher Pod's UID/resourceVersion is unknown")
    namespace = regional.settings.namespace
    require_action_time(120)
    regional.kubectl(
        "gpu",
        "delete",
        "--raw",
        f"/api/v1/namespaces/{namespace}/pods/{pod['name']}",
        "-f",
        "-",
        input_text=json.dumps(
            {
                "apiVersion": "v1",
                "kind": "DeleteOptions",
                "preconditions": {
                    "uid": pod["uid"],
                    "resourceVersion": pod["resource_version"],
                },
                "propagationPolicy": "Foreground",
                "gracePeriodSeconds": 0,
            }
        ),
    )


def wait_replacement(
    ctx: VariantContext, old_uid: str, *, timeout_seconds: float
) -> dict[str, Any]:
    """The Ready replacement watcher Pod (a new UID) once the old one is gone."""

    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        try:
            pod = watcher_pod(ctx.regional)
        except Exception as exc:  # noqa: BLE001 - the singleton is mid-replacement
            last = {"error": f"{type(exc).__name__}: {exc}"}
            return None
        last = pod
        return pod if pod["uid"] != old_uid else None

    value = ctx.wait_until(
        accept,
        timeout_seconds=timeout_seconds,
        poll_seconds=REPLACEMENT_POLL_SECONDS,
        name="watcher-replacement",
    )
    if value is None:
        raise RegionalFixtureError(
            f"no Ready replacement {WATCHER_APP} Pod within {timeout_seconds:.0f} s: {last}"
        )
    return value


def restart_watcher(
    ctx: VariantContext, *, source_failure: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Replace the watcher: the record of the deleted Pod and its successor."""

    pod = watcher_pod(ctx.regional)
    deleted_at = ctx.clock()
    delete_watcher_pod(ctx.regional, pod)
    # The Pod went between these two readings; ``classify_delete`` judges the
    # whole bracket against the product-dated transaction.
    returned_at = ctx.clock()
    replacement = wait_replacement(
        ctx, str(pod["uid"]), timeout_seconds=REPLACEMENT_TIMEOUT_SECONDS
    )
    document = {
        "deleted": pod,
        "deleted_at": deleted_at.isoformat(),
        "deletion_returned_at": returned_at.isoformat(),
        "replacement": replacement,
        "ready_at": ctx.clock().isoformat(),
        **(
            {
                "source_failure": dict(source_failure),
                "delete_offset_seconds": DELETE_OFFSET_SECONDS,
            }
            if source_failure is not None
            else {}
        ),
    }
    ctx.evidence.write("watcher-restart", document)
    return document


def replace_watcher_at_source_failure(
    ctx: VariantContext, *, job_id: str, attempt_id: str
) -> dict[str, Any]:
    """Watch the source attempt end, then replace the watcher at once."""

    failure = wait_source_failure(ctx, job_id=job_id, attempt_id=attempt_id)
    if DELETE_OFFSET_SECONDS > 0:
        ctx.sleep(DELETE_OFFSET_SECONDS)
    return restart_watcher(ctx, source_failure=failure)


def recovery_settled(records: Mapping[str, Any], attempt_id: str) -> bool:
    entry = (records.get("attempts") or {}).get(attempt_id) or {}
    recovery = entry.get("recovery") or []
    return bool(recovery) and all(
        str((item.get("workflow") or {}).get("status")) in TERMINAL_WORKFLOWS
        for item in recovery
    )


def wait_recovery_settled(
    ctx: VariantContext,
    read: Callable[[], dict[str, Any]],
    attempt_id: str,
    *,
    deleted_at: datetime,
) -> dict[str, Any]:
    """The records once every recovery workflow is terminal, within what is
    left of ``settle_bound`` since the delete; the last read otherwise (the
    graders then say what is still open)."""

    elapsed = (ctx.clock() - deleted_at).total_seconds()
    budget = max(float(STORE_POLL_SECONDS), settle_bound()["total_seconds"] - elapsed)
    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = read()
        return last if recovery_settled(last, attempt_id) else None

    value = ctx.wait_until(
        accept,
        timeout_seconds=budget,
        poll_seconds=STORE_POLL_SECONDS,
        name="recovery-settled",
    )
    return value if value is not None else last


# --------------------------------------------------------------------------- #
# dating the transaction and the delete (pure)
# --------------------------------------------------------------------------- #
def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _seconds(later: datetime | None, earlier: datetime | None) -> float | None:
    if later is None or earlier is None:
        return None
    return round((later - earlier).total_seconds(), 3)


def transaction_timeline(records: Mapping[str, Any], attempt_id: str) -> dict[str, Any]:
    """The restart transaction as the product's own records date it.

    ``source_ended_at`` is the terminal event's ``ended_at`` (the attempt's end
    as the watcher saw it); ``published_at`` the containment incident's
    ``created_at`` (the watcher's first write about the failure, accepted);
    ``decided_at`` the earliest recovery incident's ``created_at`` (the
    terminal decision opened it); ``reserved_at`` the recovery workflow's first
    CLAIM event (the claim preparation reserves the restart in that tick);
    ``dispatched_at`` the first RESTART_WORKLOAD STEP_ATTEMPT event (the step
    WAITING on the data-plane command); ``restart_closed_at`` the terminal
    RESTART_WORKLOAD record's ``updated_at`` (one record per step identity,
    replaced on every attempt: ``step_bounds.record_attempt``); ``settled_at``
    the latest terminal recovery workflow's ``updated_at``. A field the records
    cannot date is None.
    """

    entry = (records.get("attempts") or {}).get(attempt_id) or {}
    published = _parse_time((entry.get("containment_incident") or {}).get("created_at"))
    source_ended = _parse_time((entry.get("event") or {}).get("ended_at"))
    decisions: list[datetime] = []
    claims: list[datetime] = []
    dispatches: list[datetime] = []
    closes: list[datetime] = []
    settles: list[datetime] = []
    for recovery in entry.get("recovery") or []:
        decided = _parse_time((recovery.get("incident") or {}).get("created_at"))
        if decided is not None:
            decisions.append(decided)
        workflow = dict(recovery.get("workflow") or {})
        for event in workflow.get("events") or []:
            at = _parse_time(event.get("at"))
            if at is None:
                continue
            if event.get("kind") == "CLAIM":
                claims.append(at)
            elif (
                event.get("kind") == "STEP_ATTEMPT"
                and event.get("operation") == "RESTART_WORKLOAD"
            ):
                dispatches.append(at)
        for item in executions(workflow, "RESTART_WORKLOAD"):
            if item.get("status") in TERMINAL_STEPS:
                at = _parse_time(item.get("updated_at"))
                if at is not None:
                    closes.append(at)
        if str(workflow.get("status")) in TERMINAL_WORKFLOWS:
            at = _parse_time(workflow.get("updated_at"))
            if at is not None:
                settles.append(at)
    reserved = min(claims, default=None)
    closed = max(closes, default=None)
    settled = max(settles, default=None)
    budget = records.get("restart_budget")
    return {
        "source_ended_at": _iso(source_ended),
        "published_at": _iso(published),
        "decided_at": _iso(min(decisions, default=None)),
        "reserved_at": _iso(reserved),
        "dispatched_at": _iso(min(dispatches, default=None)),
        "restart_closed_at": _iso(closed),
        "settled_at": _iso(settled),
        "claim_open_seconds": _seconds(closed, reserved),
        "transaction_seconds": _seconds(settled, published),
        "ledger_updated_at": (
            budget.get("updated_at") if isinstance(budget, dict) else None
        ),
    }


def classify_delete(
    timeline: Mapping[str, Any], document: Mapping[str, Any]
) -> dict[str, Any]:
    """Where the fenced delete landed in the product-dated transaction.

    The delete's API round trip is bracketed by ``deleted_at`` (the runner's
    clock before the request) and ``deletion_returned_at`` (after it); the Pod
    went at some instant in between. A landing is claimed only when the whole
    bracket is before the marker that bounds it (``after < marker``), so a
    bracket straddling the settle counts as AFTER_SETTLE -- the replacement may
    have acted on a finished transaction -- and never as inside. Markers the
    records do not date make the landing UNDATED; the graders then speak.
    """

    before = _parse_time(document.get("deleted_at"))
    after = _parse_time(document.get("deletion_returned_at")) or before
    published = _parse_time(timeline.get("published_at"))
    closed = _parse_time(timeline.get("restart_closed_at"))
    settled = _parse_time(timeline.get("settled_at"))
    markers = ", ".join(
        f"{key} {timeline.get(key) or 'undated'}"
        for key in (
            "published_at",
            "decided_at",
            "reserved_at",
            "restart_closed_at",
            "settled_at",
        )
    )
    if before is None or after is None:
        landed = UNDATED
        detail = f"the restart document does not date the delete; {markers}"
    elif published is None or settled is None:
        landed = UNDATED
        detail = (
            f"the delete request ran {before.isoformat()} -> {after.isoformat()}; "
            f"the product's records do not date the transaction's "
            f"{'publish' if published is None else 'settle'}; {markers}"
        )
    else:
        if after < published:
            landed = BEFORE_PUBLISH
        elif closed is not None and after < closed:
            landed = BETWEEN_PUBLISH_AND_RESTART
        elif after < settled:
            landed = AFTER_RESTART_BEFORE_SETTLE
        else:
            landed = AFTER_SETTLE
        detail = (
            f"the delete request ran {before.isoformat()} -> {after.isoformat()}, "
            f"{_seconds(before, published)} s after the publish and "
            f"{_seconds(settled, after)} s before the settle; {markers}"
        )
    return {
        "landed": landed,
        "deleted_at": _iso(before),
        "deletion_returned_at": _iso(after),
        "after_publish_seconds": _seconds(before, published),
        "before_settle_seconds": _seconds(settled, after),
        "detail": detail,
    }


def landing_inconclusive_reason(landing: Mapping[str, Any]) -> str | None:
    """Why the landing leaves the variant INCONCLUSIVE, or None when it is
    inside the transaction (or undated, which the graders judge)."""

    if landing.get("landed") != AFTER_SETTLE:
        return None
    return (
        "the watcher was replaced after the restart transaction had settled: "
        f"{landing.get('detail')}"
    )


# --------------------------------------------------------------------------- #
# graders (pure)
# --------------------------------------------------------------------------- #
def replacement_errors(document: Mapping[str, Any]) -> list[str]:
    deleted = document.get("deleted") or {}
    replacement = document.get("replacement") or {}
    errors = []
    if not replacement.get("uid") or replacement.get("uid") == deleted.get("uid"):
        errors.append("the watcher Pod was not replaced by a new UID")
    if not replacement.get("ready"):
        errors.append("the replacement watcher Pod is not Ready")
    return errors


def worker_restart_errors(
    entry: Mapping[str, Any], records: Mapping[str, Any]
) -> list[str]:
    """The restart happened exactly once and the ledger kept exactly its claim."""

    errors = single_action_errors(entry, records)
    decision = entry.get("decision") or {}
    if decision and decision.get("status") != "PLAN_CREATED":
        errors.append(
            f"the terminal decision was {decision.get('status')}, expected a budgeted restart"
        )
    recovery = entry.get("recovery") or []
    if len(recovery) != 1:
        errors.append(f"expected one recovery workflow, found {len(recovery)}")
    workflows = [dict(item.get("workflow") or {}) for item in recovery]
    restarts = sum(
        1
        for workflow in workflows
        for item in executions(workflow, "RESTART_WORKLOAD")
        if item.get("status") == "SUCCEEDED"
    )
    if restarts != 1:
        errors.append(f"RESTART_WORKLOAD succeeded {restarts} times, expected once")
    for workflow in workflows:
        if workflow.get("status") != "SUCCEEDED":
            errors.append(
                f"recovery workflow {workflow.get('request_id')} ended {workflow.get('status')}"
            )
    budget = records.get("restart_budget")
    if not isinstance(budget, dict):
        errors.append("no restart-budget ledger row for the job")
        return errors
    count = int(budget.get("restart_count", 0))
    reservations = [str(item) for item in budget.get("reservation_ids") or []]
    if count == 0 or not reservations:
        errors.append(
            "the restart reservation was lost (ledger shows no consumed restart)"
        )
    elif count > 1 or len(reservations) > 1:
        errors.append(
            f"the ledger shows {count} restarts / {len(reservations)} reservations: "
            "a double decrement"
        )
    request_ids = {str(workflow.get("request_id")) for workflow in workflows}
    if reservations and not any(
        reservation.startswith(f"{request_id}/")
        for request_id in request_ids
        for reservation in reservations
    ):
        errors.append(
            f"the ledger's reservation {reservations} does not belong to the recovery "
            f"workflow {sorted(request_ids)}"
        )
    return errors


def attempt_once_errors(
    observations: Sequence[Mapping[str, Any]], *, job_id: str, source_attempt_id: str
) -> list[str]:
    """The watcher observed exactly one restart attempt of the job.

    Counted by attempt id over whichever watcher instance published: a source
    observation the replacement published after the delete (BEFORE_PUBLISH)
    is the same attempt as one the deleted watcher published.
    """

    attempts = sorted(
        {
            str(item.get("attempt_id"))
            for item in observations
            if item.get("job_id") == job_id and item.get("attempt_id")
        }
    )
    if source_attempt_id not in attempts:
        return [f"the watcher never published an observation of {source_attempt_id}"]
    restarted = [item for item in attempts if item != source_attempt_id]
    if not restarted:
        return ["no restart attempt of the job was observed by the watcher"]
    if len(restarted) > 1:
        return [f"{len(restarted)} restart attempts were observed: {restarted}"]
    return []


__all__ = [
    "AFTER_RESTART_BEFORE_SETTLE",
    "AFTER_SETTLE",
    "ATTEMPT_ID_LABEL",
    "BEFORE_PUBLISH",
    "BETWEEN_PUBLISH_AND_RESTART",
    "COPY_RUN_SECONDS",
    "DELETE_OFFSET_SECONDS",
    "JOB_ENDED",
    "JOB_ID_LABEL",
    "LANDINGS",
    "POD_ABSENT",
    "POD_TERMINATED",
    "REPLACEMENT_TIMEOUT_SECONDS",
    "SETTLE_MARGIN_SECONDS",
    "SOURCE_END_SIGNALS",
    "SOURCE_FAILURE_TIMEOUT_SECONDS",
    "SOURCE_POLL_SECONDS",
    "SOURCE_START_POLL_SECONDS",
    "TERMINAL_STEPS",
    "UNDATED",
    "WATCHER_POLL_SECONDS",
    "attempt_once_errors",
    "classify_delete",
    "delete_watcher_pod",
    "landing_inconclusive_reason",
    "recovery_settled",
    "replace_watcher_at_source_failure",
    "replacement_errors",
    "restart_watcher",
    "settle_bound",
    "source_objects",
    "source_pods",
    "transaction_timeline",
    "wait_recovery_settled",
    "wait_replacement",
    "wait_source_failure",
    "worker_restart_errors",
]
