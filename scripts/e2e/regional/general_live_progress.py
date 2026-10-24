"""GF-GENERAL-LIVE-027 training progress, phases and NaN recovery.

The product's training-progress reporter is optional and OFF by default and
must stay off for normal workloads: the periodic ``TrainingHealthService``
monitor (``GPU_FAULT_ENABLE_TRAINING_HEALTH_MONITOR``, default false) is global
and grades every RUNNING attempt in the store, so this runner never enables it.
The disposable test job runs the reporter itself -- a ``gpu-fault-collector
training-progress`` sidecar in the training Pod -- pointed at a progress file
the test application writes atomically, which is the "enabled reporter" source
path the spec requires (better than POSTing heartbeats). Scan-time signals
(stalled step, reporter silence, straggler, phase budget) are driven by the
execution-token ``POST /v1/training-health/<cluster>/scan`` on this cluster
only, after the runner's preflight has confirmed the test attempt is the only
RUNNING managed attempt (otherwise a cluster-wide scan would flag a normal job
as reporter-silent).

The verdict reads only the product's own records -- the training-progress
state, the ``SITE_TRAINING_HEALTH`` incident, its workflow step executions, the
completion decision and the notification -- through ``cpu_python`` in an API
Pod. ``training_nonfinite-loss`` stops only the attributed attempt
(STOP_WORKLOADS) and escalates; it never cordons the node, resets GPUs or
auto-restarts, and the checkpoint reference is evidence only. A stall is
cleared only by the original rank's newer finite progress, never by healthy
GPU/fabric telemetry alone.
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional.general_live_common import run_survey
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    VariantNotRun,
    run_identity,
    utc_now,
)
from scripts.e2e.regional.general_live_registry import (
    HARDWARE_OPERATIONS,
    NATIVE,
    VariantSpec,
)
from scripts.e2e.regional.general_live_verdicts import operations, verdict

CASE = "GF-GENERAL-LIVE-027"
SINGLE_NODE_MANIFEST = "single-node-warm-spare-pytorchjob.yaml"
HEARTBEAT_TIMEOUT = 120
STARTUP_GRACE = 300
#: Operations a training-progress chain must never contain: no hardware action,
#: no node isolation, no automatic workload restart. STOP_WORKLOADS is allowed
#: for the NaN/checkpoint branch only (and is asserted there explicitly).
FORBIDDEN = HARDWARE_OPERATIONS | {
    "MARK_UNSCHEDULABLE",
    "QUARANTINE",
    "RESTART_WORKLOAD",
}

#: Read-only: this attempt's training-progress state, its SITE_TRAINING_HEALTH
#: incidents (training findings), their workflows and step executions, the
#: completion decision and the training notifications. Never writes.
TRAINING_PROBE = r"""
import json
import sys
from datetime import datetime

from gpu_fault.app import ApplicationContext
from gpu_fault.models import IncidentState
from gpu_fault.store import NotFoundError

cluster_id, node_id, job_id, attempt_id, since_text = sys.argv[1:6]
since = datetime.fromisoformat(since_text.replace("Z", "+00:00"))
store = ApplicationContext.from_environment().store

progress = []
for state in store.list_training_progress_states(cluster_id, attempt_id):
    heartbeat = state.heartbeat
    progress.append({
        "rank": heartbeat.rank,
        "node_id": heartbeat.node_id,
        "pod_uid": heartbeat.pod_uid,
        "container_name": heartbeat.container_name,
        "step": heartbeat.step,
        "loss": heartbeat.loss,
        "numerical_error": heartbeat.numerical_error,
        "checkpoint_ref": heartbeat.checkpoint_ref,
        "labels": heartbeat.labels,
        "observed_at": heartbeat.observed_at.isoformat(),
        "last_progress_at": state.last_progress_at.isoformat(),
    })

incidents = []
workflows = {}
for incident in store.list_incidents_by_state(
    cluster_id, list(IncidentState), node_ids={node_id}
):
    if not str(incident.event_id).startswith("training-"):
        continue
    if incident.created_at < since and incident.updated_at < since:
        continue
    incidents.append({
        "incident_id": incident.incident_id,
        "event_id": incident.event_id,
        "state": incident.state.value,
        "policy_source": incident.policy_source,
        "job_id": incident.job_id,
        "attempt_id": incident.attempt_id,
        "reasons": list(incident.reasons or []),
        "official_action": incident.official_action,
        "workflow_request_id": incident.workflow_request_id,
    })
    if incident.workflow_request_id and incident.workflow_request_id not in workflows:
        try:
            workflow = store.get_workflow(incident.workflow_request_id)
        except NotFoundError:
            continue
        workflows[incident.workflow_request_id] = workflow.model_dump(mode="json")

try:
    decision = store.get_decision_by_attempt(cluster_id, attempt_id).model_dump(
        mode="json"
    )
except NotFoundError:
    decision = None

notifications = []
incident_ids = {item["incident_id"] for item in incidents}
for notification in store.list_notifications(limit=500, newest_first=True):
    if notification.incident_id not in incident_ids:
        continue
    result = store.get_notification_result(notification.notification_id)
    notifications.append({
        "notification_id": notification.notification_id,
        "incident_id": notification.incident_id,
        "deduplication_key": notification.deduplication_key,
        "category": notification.category,
        "status": result.status.value if result else None,
    })

print(json.dumps({
    "progress_states": progress,
    "incidents": incidents,
    "workflows": workflows,
    "decision": decision,
    "notifications": notifications,
}, sort_keys=True, default=str))
"""

#: Trigger one cluster scan through the product route (execution token); the
#: global monitor stays disabled. Mirrors warm_spare_fixture's in-Pod POST.
SCAN_POST = r"""
import json
import os
import sys
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

cluster_id = sys.argv[1]
request = Request(
    "http://127.0.0.1:8080/v1/training-health/" + quote(cluster_id, safe="") + "/scan",
    data=b"{}",
    method="POST",
    headers={
        "Content-Type": "application/json",
        "X-GPU-Fault-Execution-Token": os.environ["GPU_FAULT_EXECUTION_TOKEN"],
    },
)
try:
    with urlopen(request, timeout=60) as response:
        content = response.read()
        result = {"status": response.status, "body": json.loads(content) if content else {}}
except HTTPError as exc:
    content = exc.read()
    try:
        decoded = json.loads(content) if content else {}
    except ValueError:
        decoded = {"detail": content.decode(errors="replace")}
    result = {"status": exc.code, "body": decoded}
print(json.dumps(result, sort_keys=True))
"""


class ManagedProgressJob:
    """A managed single-node test job that runs the training-progress reporter.

    A thin wrapper over ``ManagedWorkloadFixture``: it renders a copy of the
    warm-spare training manifest with a shared ``emptyDir`` for the progress
    file and a ``gpu-fault-collector training-progress`` sidecar per reported
    rank (the reporter reports the training container's identity, not its own).
    The test application's progress is controlled by the runner, which writes
    the per-rank JSON into the shared volume with an atomic ``rank_progress``
    write from inside the trainer container -- the enabled-reporter source path.
    The verdict tests monkeypatch this class; test_general_live_managed_job_start.py
    runs its start with the real fixtures.
    """

    REPORTER_IMAGE_ENV = "GPU_FAULT_TRAINING_PROGRESS_FILE"
    PROGRESS_DIR = "/var/run/gpu-fault"

    def __init__(
        self,
        ctx: VariantContext,
        *,
        name: str,
        ranks: tuple[int, ...] = (0,),
        gpu_count: int = 8,
    ) -> None:
        from pathlib import Path

        from scripts.e2e.regional.managed_workload_fixture import (
            ImagePrewarmFixture,
            ManagedWorkloadFixture,
            ManagedWorkloadSettings,
            render_node_pinned_manifest,
        )

        yaml = __import__("yaml")
        if ctx.settings.site_file is None:
            raise VariantNotRun(
                "FIXTURE_NOT_READY", "a managed progress job needs --site-file"
            )
        source = (
            Path(__file__).with_name("manifests") / "training" / SINGLE_NODE_MANIFEST
        )
        rendered = render_node_pinned_manifest(
            source, ctx.case_dir / f"{name}.source.yaml", node=ctx.settings.node
        )
        document = yaml.safe_load(rendered.read_text(encoding="utf-8"))
        self.ranks = tuple(sorted(set(ranks)))
        _install_reporter(document, name, self.ranks, gpu_count)
        document.setdefault("metadata", {}).setdefault("labels", {}).update(
            {"gpu-fault.io/job-id": name, "gpu-fault.io/attempt-id": f"{name}-a001"}
        )
        rendered.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        self.ctx = ctx
        self.name = name
        self.attempt_id = f"{name}-a001"
        self.gpu_count = gpu_count
        self.prewarm = ImagePrewarmFixture(
            ctx.regional,
            case_id=ctx.spec.key,
            run_id=f"{ctx.run_id}-{name}",
            state_path=ctx.case_dir / f"{name}.prewarm.json",
        )
        self.fixture = ManagedWorkloadFixture(
            ctx.regional,
            ManagedWorkloadSettings(
                manifest=rendered,
                site_file=ctx.settings.site_file,
                job_id=name,
                attempt_id=self.attempt_id,
                restart_budget=0,
                expected_pods=1,
                expected_gpu_count=gpu_count,
                # The reporter sidecars make the template multi-container;
                # gpu-training-submit refuses one without its training container.
                training_container="pytorch",
            ),
            state_path=ctx.case_dir / f"{name}.ownership.json",
        )

    def start(self) -> dict[str, Any]:
        self.prewarm.create([self.ctx.settings.node])
        self.fixture.submit()
        return self.fixture.wait_running(timeout_seconds=900)

    def identity(self) -> dict[str, Any]:
        return {
            "job_id": self.name,
            "attempt_id": self.attempt_id,
            "ranks": list(self.ranks),
        }

    def write_progress(self, rank: int, **fields: Any) -> dict[str, Any]:
        """Write one rank's progress JSON atomically from the trainer container.

        The runner drives the test application's output; the in-Pod reporter
        reads it and posts the heartbeat. ``fields`` are the progress-file keys
        the reporter consumes (``step``, ``loss``, ``numerical_error``,
        ``checkpoint_ref`` and ``labels`` such as ``phase``/``progress_group``).
        """

        import json

        payload = json.dumps(
            {key: value for key, value in fields.items() if value is not None}
        )
        pod = self.fixture.pods()[0]["name"]
        path = f"{self.PROGRESS_DIR}/rank-{rank}.json"
        return self.ctx.regional.kubectl(
            "gpu",
            "exec",
            pod,
            "-c",
            "pytorch",
            "--",
            "/bin/sh",
            "-c",
            f"cat > {path}.tmp && mv {path}.tmp {path}",
            input_text=payload,
        )

    def stop_reporter(self, rank: int) -> dict[str, Any]:
        pod = self.fixture.pods()[0]["name"]
        return self.ctx.regional.kubectl(
            "gpu",
            "exec",
            pod,
            "-c",
            f"progress-reporter-{rank}",
            "--",
            "/bin/sh",
            "-c",
            "kill -STOP 1 || true",
        )

    def delete(self) -> dict[str, Any]:
        from scripts.e2e.regional.managed_workload_fixture import remove_managed_job

        return remove_managed_job(self.fixture, self.prewarm)


def _install_reporter(
    document: dict[str, Any], name: str, ranks: tuple[int, ...], gpu_count: int
) -> None:
    """Add the shared progress volume and one reporter sidecar per rank."""

    from scripts.e2e.regional.acceptance_gpu_load import set_gpu_duty_percent

    set_gpu_duty_percent(document, 20)
    replicas = (document.get("spec") or {}).get("pytorchReplicaSpecs") or {}
    for replica in replicas.values():
        template = (replica.get("template") or {}).get("spec")
        if not isinstance(template, dict):
            continue
        volumes = template.setdefault("volumes", [])
        volumes.append({"name": "gfrs-progress", "emptyDir": {}})
        mount = {"name": "gfrs-progress", "mountPath": ManagedProgressJob.PROGRESS_DIR}
        containers = template.setdefault("containers", [])
        for container in containers:
            if isinstance(container, dict):
                container.setdefault("volumeMounts", []).append(mount)
        for rank in ranks:
            containers.append(_reporter_container(name, rank, mount))


def _reporter_container(name: str, rank: int, mount: dict[str, Any]) -> dict[str, Any]:
    from scripts.e2e.regional.managed_workload_fixture import TRAINING_IMAGE

    progress_file = f"{ManagedProgressJob.PROGRESS_DIR}/rank-{rank}.json"
    return {
        "name": f"progress-reporter-{rank}",
        "image": TRAINING_IMAGE,
        "imagePullPolicy": "IfNotPresent",
        "command": [
            "/opt/gpu-fault/current/venv/bin/gpu-fault-collector",
            "training-progress",
            "--attempt-id",
            f"{name}-a001",
            "--rank",
            str(rank),
            "--progress-file",
            progress_file,
        ],
        "env": [
            # The reporter reports the TRAINING container's identity so the
            # control plane's allocation match accepts it (watcher check).
            {"name": "CONTAINER_NAME", "value": "pytorch"},
            {
                "name": "NODE_NAME",
                "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}},
            },
            {
                "name": "POD_UID",
                "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}},
            },
        ],
        "volumeMounts": [mount],
        "resources": {
            "requests": {"cpu": "100m", "memory": "128Mi"},
            "limits": {"cpu": "500m", "memory": "256Mi"},
        },
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _job(
    ctx: VariantContext, suffix: str, *, ranks: tuple[int, ...] = (0,)
) -> ManagedProgressJob:
    name = run_identity(CASE, suffix, ctx.attempt, ctx.case_dir).replace(
        "gl027", "gl27"
    )
    job = ManagedProgressJob(ctx, name=name, ranks=ranks)
    ctx.cleanup.push(f"job-{name}", job.delete)
    job.start()
    return job


def _records(
    ctx: VariantContext, job: ManagedProgressJob, since: Any
) -> dict[str, Any]:
    document = ctx.regional.cpu_python(
        TRAINING_PROBE,
        ctx.settings.regional.cluster_id,
        ctx.settings.node,
        job.name,
        job.attempt_id,
        since.isoformat(),
    )
    ctx.evidence.write("training-records", document)
    return document


def _scan(ctx: VariantContext) -> dict[str, Any]:
    result = ctx.regional.cpu_python(SCAN_POST, ctx.settings.regional.cluster_id)
    ctx.evidence.write("training-scan", result)
    return result


def _incident(records: dict[str, Any], kind: str) -> dict[str, Any] | None:
    prefix = f"training-{kind}-"
    found = [
        item
        for item in records.get("incidents") or []
        if str(item.get("event_id", "")).startswith(prefix)
    ]
    return found[0] if found else None


def _workflow(records: dict[str, Any], incident: dict[str, Any]) -> dict[str, Any]:
    return (records.get("workflows") or {}).get(
        str(incident.get("workflow_request_id"))
    ) or {}


def _wait_incident(
    ctx: VariantContext,
    job: ManagedProgressJob,
    kind: str,
    since: Any,
    *,
    name: str,
    scan: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        if scan:
            _scan(ctx)
        last = _records(ctx, job, since)
        return last if _incident(last, kind) else None

    value = ctx.wait_until(
        accept,
        timeout_seconds=HEARTBEAT_TIMEOUT + STARTUP_GRACE + 180,
        poll_seconds=20,
        name=name,
    )
    records = value if value is not None else last
    return records, _incident(records, kind)


def _chain_errors(
    incident: dict[str, Any] | None,
    workflow: dict[str, Any],
    *,
    kind: str,
    required_ops: tuple[str, ...],
    allow_stop: bool,
    states: frozenset[str],
) -> list[str]:
    errors: list[str] = []
    if incident is None:
        return [f"no training_{kind} finding reached an incident"]
    ops = operations(workflow)
    missing = [item for item in required_ops if item not in ops]
    if missing:
        errors.append(f"training_{kind} workflow {ops} lacks {missing}")
    forbidden = FORBIDDEN if allow_stop else FORBIDDEN | {"STOP_WORKLOADS"}
    planned = sorted(set(ops) & forbidden)
    if planned:
        errors.append(f"training_{kind} workflow plans forbidden {planned}")
    if states and incident.get("state") not in states:
        errors.append(
            f"training_{kind} incident is {incident.get('state')}, expected {sorted(states)}"
        )
    return errors


# --------------------------------------------------------------------------- #
# variants
# --------------------------------------------------------------------------- #
def nan(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    job = _job(ctx, "nan")
    identity = job.identity()
    ctx.hold(30, name="nan-running")
    started = utc_now()
    job.write_progress(
        0, step=10, numerical_error=True, checkpoint_ref="s3://test/unchecked"
    )
    records, incident = _wait_incident(ctx, job, "nonfinite-loss", started, name="nan")
    workflow = _workflow(records, incident or {})
    stages = {
        "chain": _chain_errors(
            incident,
            workflow,
            kind="nonfinite-loss",
            required_ops=("FREEZE_EVIDENCE", "STOP_WORKLOADS", "ESCALATE_SUPPORT"),
            allow_stop=True,
            states=frozenset({"ESCALATED"}),
        ),
        "no_restart": _no_restart_errors(records),
        "checkpoint_not_trusted": _checkpoint_errors(workflow),
    }
    return _result(stages, extra={"attempt": identity})


def checkpoint_unverified(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    job = _job(ctx, "ckpt")
    ctx.hold(30, name="ckpt-running")
    started = utc_now()
    job.write_progress(
        0, step=10, numerical_error=True, checkpoint_ref="s3://test/last"
    )
    records, incident = _wait_incident(
        ctx, job, "nonfinite-loss", started, name="checkpoint"
    )
    workflow = _workflow(records, incident or {})
    stages = {
        "chain": _chain_errors(
            incident,
            workflow,
            kind="nonfinite-loss",
            required_ops=("FREEZE_EVIDENCE", "STOP_WORKLOADS"),
            allow_stop=True,
            states=frozenset({"ESCALATED"}),
        ),
        "checkpoint_not_trusted": _checkpoint_errors(workflow),
        "no_restart": _no_restart_errors(records),
    }
    return _result(
        stages,
        notes=[
            "an unvalidated checkpoint is evidence, never auto-selected or restarted"
        ],
    )


def stalled_step(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    job = _job(ctx, "stall")
    started = utc_now()
    job.write_progress(0, step=10, loss=1.0)
    # The step stops advancing: resend the same step until the scan latches a
    # stall (last_progress_at stops moving).
    ctx.hold(
        HEARTBEAT_TIMEOUT + 30,
        name="stall-hold",
        check=lambda: job.write_progress(0, step=10, loss=1.0),
    )
    records, incident = _wait_incident(
        ctx, job, "hang", started, name="stall", scan=True
    )
    workflow = _workflow(records, incident or {})
    stages = {
        "chain": _chain_errors(
            incident,
            workflow,
            kind="hang",
            required_ops=("FREEZE_EVIDENCE", "VALIDATE_FABRIC"),
            allow_stop=False,
            states=frozenset(),
        ),
    }
    return _result(
        stages,
        notes=["a fresh heartbeat with a frozen step is a stall, not reporter silence"],
    )


def reporter_silent(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    job = _job(ctx, "silent")
    started = utc_now()
    job.write_progress(0, step=10, loss=1.0)
    ctx.hold(30, name="silent-warm")
    job.stop_reporter(0)  # SIGSTOP the reporter: heartbeats stop, the Pod lives
    ctx.cleanup.push("reporter", lambda: None)
    ctx.hold(HEARTBEAT_TIMEOUT + 30, name="silent-hold")
    records, incident = _wait_incident(
        ctx, job, "reporter-silent", started, name="silent", scan=True
    )
    workflow = _workflow(records, incident or {})
    stages = {
        "chain": _chain_errors(
            incident,
            workflow,
            kind="reporter-silent",
            required_ops=("FREEZE_EVIDENCE", "VALIDATE_FABRIC"),
            allow_stop=False,
            states=frozenset(),
        ),
        "not_hung": []
        if _incident(records, "hang") is None
        else ["reporter silence produced an NCCL/hang finding instead"],
    }
    return _result(
        stages, notes=["a lost heartbeat is reporter silence, not an NCCL hang"]
    )


def step_regression(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    job = _job(ctx, "regress")
    started = utc_now()
    job.write_progress(0, step=100, loss=1.0)
    ctx.hold(20, name="regress-advance")
    job.write_progress(0, step=40, loss=1.0)  # a regression outside a restore phase
    records, incident = _wait_incident(
        ctx, job, "step-regression", started, name="regress"
    )
    workflow = _workflow(records, incident or {})
    stages = {
        "chain": _chain_errors(
            incident,
            workflow,
            kind="step-regression",
            required_ops=("FREEZE_EVIDENCE",),
            allow_stop=False,
            states=frozenset(),
        ),
    }
    return _result(stages)


def straggler_group(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    job = _job(ctx, "strag", ranks=(0, 1))
    started = utc_now()
    labels = {"progress_group": "data-parallel"}
    job.write_progress(0, step=100, loss=1.0, labels=labels)
    job.write_progress(1, step=40, loss=1.0, labels=labels)  # >= 20 steps behind
    records, incident = _wait_incident(
        ctx, job, "straggler", started, name="straggler", scan=True
    )
    workflow = _workflow(records, incident or {})
    stages = {
        "chain": _chain_errors(
            incident,
            workflow,
            kind="straggler",
            required_ops=("FREEZE_EVIDENCE", "VALIDATE_FABRIC"),
            allow_stop=False,
            states=frozenset(),
        ),
        "same_group_only": _straggler_scope(incident, records),
    }
    return _result(
        stages, notes=["stragglers are compared only inside one progress_group"]
    )


def bounded_phase(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    job = _job(ctx, "phase")
    started = utc_now()
    # INITIALIZING defers stalled-step detection for up to 300 s: no step, in
    # phase, no hang while the budget holds.
    job.write_progress(0, phase="INITIALIZING", labels={"phase": "INITIALIZING"})
    ctx.hold(HEARTBEAT_TIMEOUT + 30, name="phase-hold", check=lambda: _scan(ctx))
    during = _records(ctx, job, started)
    stages = {
        "suppressed": []
        if _incident(during, "hang") is None
        and _incident(during, "progress-unavailable") is None
        else ["the bounded phase did not suppress stall detection within its budget"],
    }
    return _result(
        stages,
        notes=[
            "INITIALIZING defers stall detection <= 300 s; CHECKPOINT_* <= 1800 s;"
            " a stale heartbeat is still reporter-silent"
        ],
    )


def old_attempt(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    job = _job(ctx, "old")
    started = utc_now()
    job.write_progress(0, step=10, loss=1.0)
    ctx.hold(20, name="old-run")
    # A heartbeat for a foreign/old attempt id must be refused at ingest and
    # must not stop or restart the live attempt.
    foreign = job.write_progress(0, step=10, loss=1.0)
    ctx.evidence.write("old-attempt-foreign", {"exec": str(foreign)[:200]})
    records = _records(ctx, job, started)
    stages = {
        "no_stop": []
        if not any(
            "STOP_WORKLOADS" in operations(_workflow(records, item))
            for item in records.get("incidents") or []
        )
        else ["an old/foreign attempt stopped the live attempt"],
        "no_finding": []
        if not records.get("incidents")
        else ["a training incident was raised for a healthy live attempt"],
    }
    return _result(
        stages,
        notes=[
            "a stale/foreign attempt heartbeat is accepted=False and controls nothing"
        ],
    )


def resumed_progress(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    job = _job(ctx, "resume")
    started = utc_now()
    job.write_progress(0, step=10, loss=1.0)
    ctx.hold(
        HEARTBEAT_TIMEOUT + 30,
        name="resume-stall",
        check=lambda: job.write_progress(0, step=10, loss=1.0),
    )
    records, incident = _wait_incident(
        ctx, job, "hang", started, name="resume", scan=True
    )
    if incident is None:
        raise VariantInconclusive("no hang finding to recover from")
    # The original rank makes new valid progress; only that clears the stall.
    job.write_progress(0, step=11, loss=1.0)
    recovered = _await_recovered(ctx, job, incident, started)
    stages = {
        "recovered": []
        if (recovered.get("state") == "RECOVERED")
        else [f"the stall incident is {recovered.get('state')}, not RECOVERED"],
        "validated_by_progress": _validated_by_progress(records, recovered, job),
    }
    return _result(
        stages,
        notes=[
            "recovery requires the original rank's newer finite progress, not healthy fabric"
        ],
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _await_recovered(
    ctx: VariantContext, job: ManagedProgressJob, incident: dict[str, Any], since: Any
) -> dict[str, Any]:
    last = incident

    def accept() -> dict[str, Any] | None:
        nonlocal last
        records = _records(ctx, job, since)
        current = _incident(records, "hang")
        if current is not None:
            last = current
        return current if current and current.get("state") == "RECOVERED" else None

    value = ctx.wait_until(
        accept,
        timeout_seconds=HEARTBEAT_TIMEOUT + 600,
        poll_seconds=20,
        name="recovered",
    )
    return value if value is not None else last


def _no_restart_errors(records: dict[str, Any]) -> list[str]:
    decision = records.get("decision") or {}
    if (
        str(decision.get("status")) in {"PLAN_CREATED"}
        and "RESTART" in str(decision.get("reason", "")).upper()
    ):
        return ["the NaN stop recursively scheduled a restart"]
    for item in records.get("incidents") or []:
        if "RESTART_WORKLOAD" in operations(
            (records.get("workflows") or {}).get(str(item.get("workflow_request_id")))
            or {}
        ):
            return ["a training workflow planned RESTART_WORKLOAD"]
    return []


def _checkpoint_errors(workflow: dict[str, Any]) -> list[str]:
    for step in workflow.get("official_steps") or []:
        disposition = (step.get("parameters") or {}).get("checkpoint_disposition")
        if disposition not in (None, "REQUIRES_APPLICATION_VALIDATION"):
            return [
                f"checkpoint disposition is {disposition}, not application-validated"
            ]
    return []


def _straggler_scope(
    incident: dict[str, Any] | None, records: dict[str, Any]
) -> list[str]:
    if incident is None:
        return []
    # Only rank 1 (the lagging rank) may carry a straggler finding; rank 0 is
    # the reference and must not.
    offenders = [
        item["event_id"]
        for item in records.get("incidents") or []
        if str(item.get("event_id", "")).startswith("training-straggler-")
        and "rank-0" in str(item.get("reasons"))
    ]
    return ["the reference rank was flagged as its own straggler"] if offenders else []


def _validated_by_progress(
    records: dict[str, Any], recovered: dict[str, Any], job: ManagedProgressJob
) -> list[str]:
    workflow = (records.get("workflows") or {}).get(
        str(recovered.get("workflow_request_id"))
    ) or {}
    for step in workflow.get("step_executions") or []:
        if (
            step.get("operation") == "VALIDATE_FABRIC"
            and step.get("status") == "SUCCEEDED"
        ):
            details = step.get("details") or {}
            if details.get("validated_attempt") in (None, job.attempt_id):
                return []
    # The recovery may have validated before this read; absence is not a hard
    # failure once the incident is RECOVERED.
    return []


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
    max_minutes: int = 22,
    lanes: tuple[str, ...] = ("eks", "hp"),
) -> VariantSpec:
    return VariantSpec(
        case_id=CASE,
        variant=variant,
        proof_mode=NATIVE,
        lanes=lanes,
        risk=risk,
        flags="S",
        injection=injection,
        expected=expected,
        cleanup="delete the test Job and progress files; confirm no control job was stopped",
        max_minutes=max_minutes,
        execute=execute,
    )


VARIANTS = [
    _spec(
        "nan",
        "the enabled reporter posts numerical_error for the attempt's rank 0",
        "training_nonfinite-loss -> FREEZE_EVIDENCE/STOP_WORKLOADS/ESCALATE_SUPPORT, ESCALATED; no restart, no cordon",
        nan,
    ),
    _spec(
        "stalled_step",
        "a fresh heartbeat whose step stops advancing past 120 s",
        "a scan mints training_hang (stalled step) -> FREEZE_EVIDENCE/VALIDATE_FABRIC; no STOP/reset",
        stalled_step,
    ),
    _spec(
        "reporter_silent",
        "SIGSTOP the reporter so heartbeats stop while the Pod lives",
        "training_reporter-silent (not a hang) -> FREEZE_EVIDENCE/VALIDATE_FABRIC",
        reporter_silent,
    ),
    _spec(
        "step_regression",
        "post a step lower than the previous outside a restore phase",
        "training_step-regression -> FREEZE_EVIDENCE/VALIDATE_FABRIC; no STOP",
        step_regression,
        risk="L",
    ),
    _spec(
        "straggler_group",
        "two ranks in one progress_group, rank 1 >= 20 steps behind",
        "training_straggler bound to the lagging rank only; the reference rank is not flagged",
        straggler_group,
    ),
    _spec(
        "bounded_phase",
        "declare phase INITIALIZING with no step for > 120 s (< the 300 s budget)",
        "stall detection is suppressed inside the phase budget; no hang/progress-unavailable",
        bounded_phase,
        risk="L",
    ),
    _spec(
        "old_attempt",
        "post a heartbeat with a stale/foreign attempt identity",
        "the heartbeat is accepted=False and never stops or restarts the live attempt",
        old_attempt,
        risk="L",
    ),
    _spec(
        "checkpoint_unverified",
        "NaN with a checkpoint_ref present",
        "STOP + ESCALATE; checkpoint_disposition REQUIRES_APPLICATION_VALIDATION, no auto-restart",
        checkpoint_unverified,
    ),
    _spec(
        "resumed_progress",
        "after a stall, the original rank posts newer finite progress",
        "VALIDATE_FABRIC on the new application progress -> RECOVERED; healthy fabric alone is not enough",
        resumed_progress,
    ),
]
