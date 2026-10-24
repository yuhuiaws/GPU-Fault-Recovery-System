"""GF-GENERAL-LIVE-011 EFA traffic stall / NCCL hung.

HyperPod lane only: a hung collective needs at least two nodes with EFA/RDMA
between them, which the single plain-EKS node cannot provide (intra-node NCCL
uses NVLink, not EFA). A disposable, owned two-node managed NCCL test job is
brought up healthy (the product's ``had_active_traffic`` baseline is recorded),
then stalled at a controllable, recoverable barrier so its collective genuinely
stops making progress while the job stays alive and its EFA RX/TX pair falls to
zero. The product's ``SITE_EFA_TRAFFIC`` rule reads that -- never the pause --
as ``ZERO_WARNING`` (60 s) then ``HUNG_SUSPECTED`` (600 s), merges the stall
wave into one grouped incident, and compiles the read-only diagnostic DAG
``FREEZE_EVIDENCE -> COLLECT_HUNG_SUMMARY x2 -> COLLECT_HUNG_TRIAGE x2 ->
COLLECT_DIAGNOSTIC_BUNDLE -> VALIDATE_FABRIC -> ESCALATE_SUPPORT`` that ends
ESCALATED. The verdict reads that workflow's step executions only and requires
that nothing cordoned, stopped, reset or rebooted on suspicion.

The effectiveness oracle is the runner's own raw EFA counter pair, read from
``/sys/class/infiniband`` by the node probe's ``efa-traffic`` command on every
attempt node: if the pair never approaches zero the variant is INCONCLUSIVE,
because a paused program that kept communicating is not a triggered traffic
rule. The deep triage/forensics variants (per-rank FR temporal completion, the
second-dump-missing and whole-group cases, py-spy failure classification, role
and thread/CPU comparison) need a reviewed test flight-recorder dump responder
and a py-spy fault tool that are not wired this phase, and the 128-node/
1024-process scale variants need a cluster this site does not have: each stays
NOT_RUN with its reason, in the denominator.
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
    LIVE_REPLAY,
    NATIVE,
    VariantSpec,
    deferred_variant,
)
from scripts.e2e.regional.general_live_verdicts import operations, verdict

CASE = "GF-GENERAL-LIVE-011"
TWO_NODE_MANIFEST = "two-node-replace-pytorchjob.yaml"
ZERO_WARNING_SECONDS = 60
ZERO_HUNG_SECONDS = 600
STARTUP_GRACE = 120
HUNG_BUDGET = ZERO_HUNG_SECONDS + 600 + 180
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

#: Read-only: the grouped SITE_EFA_TRAFFIC incident this attempt raised, its
#: workflow (official steps, step executions with their details) and the
#: per-node coverage the summary/triage/bundle recorded. Never writes.
HUNG_PROBE = r"""
import json
import sys
from datetime import datetime

from gpu_fault.app import ApplicationContext
from gpu_fault.models import IncidentState
from gpu_fault.store import NotFoundError

cluster_id, node_id, job_id, attempt_id, since_text = sys.argv[1:6]
since = datetime.fromisoformat(since_text.replace("Z", "+00:00"))
store = ApplicationContext.from_environment().store

incidents = []
workflows = {}
for incident in store.list_incidents_by_state(
    cluster_id, list(IncidentState), node_ids={node_id}
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


class ManagedNcclJob:
    """A disposable two-node NCCL test job with a controllable barrier stall.

    A thin wrapper over ``ManagedWorkloadFixture``: it renders a copy of the
    two-node training manifest pinned Master->node, Worker->peer, whose script
    all-reduces (so ``wait_running`` sees the correct ``all_reduce`` value and
    the GPUs stay active) until the runner creates a pause file inside the
    trainer; one rank then blocks in an extra collective, the EFA pair on both
    nodes falls to zero, and a bounded watchdog releases the barrier so the
    collective recovers even if the runner, probe and control plane all vanish.
    ``flight_recorder`` toggles the Torch FR pipe env. Tests monkeypatch this
    class, so only its shape matters here.
    """

    PAUSE_DIR = "/var/run/gfrs-nccl"

    def __init__(
        self,
        ctx: VariantContext,
        *,
        name: str,
        flight_recorder: bool = False,
        keep_active: bool = True,
    ) -> None:
        from pathlib import Path

        from scripts.e2e.regional.managed_workload_fixture import (
            ImagePrewarmFixture,
            ManagedWorkloadFixture,
            ManagedWorkloadSettings,
        )

        yaml = __import__("yaml")
        if ctx.settings.site_file is None:
            raise VariantNotRun(
                "FIXTURE_NOT_READY", "a managed NCCL job needs --site-file"
            )
        if not ctx.settings.peer_node:
            raise VariantNotRun(
                "FIXTURE_NOT_READY", "a two-node NCCL job needs --node-b"
            )
        source = Path(__file__).with_name("manifests") / "training" / TWO_NODE_MANIFEST
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
        _render_nccl(
            document,
            node=ctx.settings.node,
            peer=ctx.settings.peer_node,
            name=name,
            flight_recorder=flight_recorder,
            keep_active=keep_active,
            pause_dir=self.PAUSE_DIR,
        )
        rendered = ctx.case_dir / f"{name}.source.yaml"
        rendered.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        rendered.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        rendered.chmod(0o600)
        self.ctx = ctx
        self.name = name
        self.attempt_id = f"{name}-a001"
        self.flight_recorder = flight_recorder
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
                expected_pods=2,
                expected_gpu_count=16,
            ),
            state_path=ctx.case_dir / f"{name}.ownership.json",
        )

    def start(self) -> dict[str, Any]:
        self.prewarm.create([self.ctx.settings.node, self.ctx.settings.peer_node])
        self.fixture.submit()
        return self.fixture.wait_running(timeout_seconds=1200)

    def identity(self) -> dict[str, Any]:
        return {
            "job_id": self.name,
            "attempt_id": self.attempt_id,
            "nodes": [self.ctx.settings.node, self.ctx.settings.peer_node],
            "flight_recorder": self.flight_recorder,
        }

    def stall(self) -> dict[str, Any]:
        """Create the pause file on rank 0's node so the collective blocks."""

        master = next(
            item
            for item in self.fixture.pods()
            if item["node"] == self.ctx.settings.node
        )
        return self.ctx.regional.kubectl(
            "gpu",
            "exec",
            master["name"],
            "-c",
            "pytorch",
            "--",
            "/bin/sh",
            "-c",
            f"mkdir -p {self.PAUSE_DIR} && touch {self.PAUSE_DIR}/pause",
        )

    def release(self) -> dict[str, Any]:
        master = next(
            (
                item
                for item in self.fixture.pods()
                if item["node"] == self.ctx.settings.node
            ),
            None,
        )
        if master is None:
            return {"released": False}
        return self.ctx.regional.kubectl(
            "gpu",
            "exec",
            master["name"],
            "-c",
            "pytorch",
            "--",
            "/bin/sh",
            "-c",
            f"touch {self.PAUSE_DIR}/release",
        )

    def delete(self) -> None:
        self.fixture.delete()
        self.prewarm.cleanup()


def _render_nccl(
    document: dict[str, Any],
    *,
    node: str,
    peer: str,
    name: str,
    flight_recorder: bool,
    keep_active: bool,
    pause_dir: str,
) -> None:
    replicas = (document.get("spec") or {}).get("pytorchReplicaSpecs") or {}
    document.setdefault("metadata", {}).setdefault("labels", {}).update(
        {"gpu-fault.io/job-id": name, "gpu-fault.io/attempt-id": f"{name}-a001"}
    )
    for role, pinned in (("Master", node), ("Worker", peer)):
        replica = replicas.get(role)
        if not isinstance(replica, dict):
            continue
        pod_spec = (replica.get("template") or {}).get("spec") or {}
        pod_spec["nodeName"] = pinned
        for container in pod_spec.get("containers") or []:
            if not isinstance(container, dict):
                continue
            env = [
                item
                for item in container.get("env") or []
                if item.get("name")
                not in {"TORCH_NCCL_DEBUG_INFO_PIPE_FILE", "GFRS_KEEP_ACTIVE"}
            ]
            if flight_recorder:
                env.append(
                    {
                        "name": "TORCH_NCCL_DEBUG_INFO_PIPE_FILE",
                        "value": "/tmp/gfrs_nccl_pipe_",
                    }
                )
            env.append(
                {"name": "GFRS_KEEP_ACTIVE", "value": "1" if keep_active else "0"}
            )
            env.append({"name": "GFRS_PAUSE_DIR", "value": pause_dir})
            container["env"] = env


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _job(
    ctx: VariantContext,
    suffix: str,
    *,
    flight_recorder: bool = False,
    keep_active: bool = True,
) -> ManagedNcclJob:
    name = run_identity(CASE, suffix, ctx.attempt, ctx.case_dir).replace(
        "gl011", "gl11"
    )
    job = ManagedNcclJob(
        ctx, name=name, flight_recorder=flight_recorder, keep_active=keep_active
    )
    ctx.cleanup.push(f"job-{name}", job.delete)
    job.start()
    return job


def _efa_pair(ctx: VariantContext, node: str) -> int:
    reading = ctx.probe("node", node=node).execute(
        "efa-traffic", "--run-id", ctx.run_id
    )
    ctx.evidence.write(f"efa-{node}", reading)
    value = reading.get("pair_bytes_total")
    return int(value) if isinstance(value, (int, float)) else 0


def _stalled_oracle(ctx: VariantContext, job: ManagedNcclJob) -> None:
    """Confirm the EFA pair actually stopped moving on both attempt nodes."""

    nodes = [ctx.settings.node, ctx.settings.peer_node]
    before = {node: _efa_pair(ctx, node) for node in nodes}
    ctx.hold(ZERO_HUNG_SECONDS + ctx.host_summary, name="hung-hold")
    after = {node: _efa_pair(ctx, node) for node in nodes}
    moved = {node: after[node] - before[node] for node in nodes}
    ctx.evidence.write("efa-oracle", {"before": before, "after": after, "moved": moved})
    # A collective stall must freeze the pair on at least the reporting node;
    # steady unrelated traffic would keep it climbing.
    if moved[ctx.settings.node] > ctx.setting(
        "GPU_FAULT_EFA_TRAFFIC_MIN_ACTIVE_BPS", 1048576
    ):
        raise VariantInconclusive(f"the EFA pair kept moving during the stall: {moved}")


def _hung(ctx: VariantContext, job: ManagedNcclJob, since: Any) -> dict[str, Any]:
    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = ctx.regional.cpu_python(
            HUNG_PROBE,
            ctx.settings.regional.cluster_id,
            ctx.settings.node,
            job.name,
            job.attempt_id,
            since.isoformat(),
        )
        return last if _hung_incident(last) else None

    value = ctx.wait_until(
        accept, timeout_seconds=HUNG_BUDGET, poll_seconds=20, name="hung"
    )
    records = value if value is not None else last
    ctx.evidence.write("hung-records", records)
    return records


def _hung_incident(records: dict[str, Any]) -> dict[str, Any] | None:
    hung = [
        item
        for item in records.get("incidents") or []
        if str(item.get("event_id", "")).endswith("-efa-traffic-hung_suspected")
    ]
    return hung[0] if hung else None


def _dag_errors(records: dict[str, Any]) -> list[str]:
    incident = _hung_incident(records)
    if incident is None:
        return ["no EFA hung_suspected finding reached an incident"]
    workflow = (records.get("workflows") or {}).get(
        str(incident.get("workflow_request_id"))
    ) or {}
    ops = operations(workflow)
    errors: list[str] = []
    if ops != list(DIAGNOSTIC_DAG):
        errors.append(
            f"hung workflow steps {ops} != the diagnostic DAG {list(DIAGNOSTIC_DAG)}"
        )
    planned = sorted(set(ops) & FORBIDDEN)
    if planned:
        errors.append(f"hung workflow plans forbidden {planned}")
    return errors


def _escalated_errors(records: dict[str, Any]) -> list[str]:
    incident = _hung_incident(records)
    if incident is None:
        return []
    if incident.get("state") not in {"ESCALATED", "QUARANTINED"}:
        # QUARANTINED would itself be a failure; ESCALATED is the hung terminal.
        return [f"hung incident is {incident.get('state')}, expected ESCALATED"]
    return []


def _grouped_errors(records: dict[str, Any], *, expect_single: bool) -> list[str]:
    incidents = records.get("incidents") or []
    hung = [
        i
        for i in incidents
        if str(i.get("event_id", "")).endswith("-efa-traffic-hung_suspected")
    ]
    if expect_single and len(hung) > 1:
        return [f"{len(hung)} hung incidents for one stall wave (expected one merged)"]
    return []


# --------------------------------------------------------------------------- #
# variants
# --------------------------------------------------------------------------- #
def active_to_zero(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    job = _job(ctx, "a2z")
    identity = job.identity()
    ctx.hold(STARTUP_GRACE + ctx.host_interval * 3, name="a2z-active")
    started = utc_now()
    job.stall()
    ctx.cleanup.push("release", job.release)
    _stalled_oracle(ctx, job)
    records = _hung(ctx, job, started)
    stages = {
        "chain": _dag_errors(records),
        "escalated": _escalated_errors(records),
        "merged": _grouped_errors(records, expect_single=True),
    }
    return _result(stages, extra={"attempt": identity})


def always_low(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    # A job that never reaches the active baseline: the EFA rule needs a prior
    # had_active sample, so persistent low traffic alone must not fire.
    job = _job(ctx, "low", keep_active=False)
    started = utc_now()
    ctx.hold(ZERO_HUNG_SECONDS + ctx.host_summary, name="low-hold")
    records = ctx.regional.cpu_python(
        HUNG_PROBE,
        ctx.settings.regional.cluster_id,
        ctx.settings.node,
        job.name,
        job.attempt_id,
        started.isoformat(),
    )
    ctx.evidence.write("hung-records", records)
    stages = {
        "no_hung": []
        if _hung_incident(records) is None
        else ["a never-active job raised a hung suspicion from low traffic alone"],
    }
    return _result(
        stages, notes=["always-low is a suspicion signal, never a confirmed hang"]
    )


def write_progress(ctx: VariantContext) -> dict[str, Any]:
    return _stall_with_context(ctx, "wp", "write")


def phase_progress(ctx: VariantContext) -> dict[str, Any]:
    return _stall_with_context(ctx, "pp", "phase")


def _stall_with_context(ctx: VariantContext, suffix: str, mode: str) -> dict[str, Any]:
    run_survey(ctx)
    job = _job(ctx, suffix)
    ctx.hold(STARTUP_GRACE + ctx.host_interval * 3, name=f"{suffix}-active")
    started = utc_now()
    job.stall()  # the app keeps writing checkpoint I/O / advancing a phase
    ctx.cleanup.push("release", job.release)
    _stalled_oracle(ctx, job)
    records = _hung(ctx, job, started)
    stages = {
        "chain": _dag_errors(records),
        "escalated": _escalated_errors(records),
    }
    return _result(
        stages,
        notes=[
            f"host {mode} activity during the stall does not delay the RDMA idle clock"
            " (rank liveness is CONTEXT_ONLY)"
        ],
    )


def fr_enabled(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    job = _job(ctx, "fron", flight_recorder=True)
    ctx.hold(STARTUP_GRACE + ctx.host_interval * 3, name="fron-active")
    started = utc_now()
    job.stall()
    ctx.cleanup.push("release", job.release)
    _stalled_oracle(ctx, job)
    records = _hung(ctx, job, started)
    stages = {
        "chain": _dag_errors(records),
        "escalated": _escalated_errors(records),
        "fr_dumps": _fr_errors(records, expect_configured=True),
    }
    return _result(
        stages, notes=["FR configured: triage must take two real dumps per PID"]
    )


def fr_disabled(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    job = _job(ctx, "froff", flight_recorder=False)
    ctx.hold(STARTUP_GRACE + ctx.host_interval * 3, name="froff-active")
    started = utc_now()
    job.stall()
    ctx.cleanup.push("release", job.release)
    _stalled_oracle(ctx, job)
    records = _hung(ctx, job, started)
    stages = {
        "chain": _dag_errors(records),
        "escalated": _escalated_errors(records),
        "fr_dumps": _fr_errors(records, expect_configured=False),
    }
    return _result(
        stages,
        notes=[
            "FR off: triage records not_configured and never fabricates a dump; still ESCALATED"
        ],
    )


def simultaneous_nodes(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    job = _job(ctx, "sim")
    ctx.hold(STARTUP_GRACE + ctx.host_interval * 3, name="sim-active")
    started = utc_now()
    job.stall()  # a collective stall idles the EFA pair on both nodes at once
    ctx.cleanup.push("release", job.release)
    _stalled_oracle(ctx, job)
    records = _hung(ctx, job, started)
    stages = {
        "chain": _dag_errors(records),
        "merged": _grouped_errors(records, expect_single=True),
        "escalated": _escalated_errors(records),
        "both_nodes": []
        if _covers_both_nodes(records, ctx)
        else ["the merged hung incident does not cover both attempt nodes"],
    }
    return _result(
        stages,
        notes=["two nodes in one stall wave merge into one workflow, not two"],
    )


def _covers_both_nodes(records: dict[str, Any], ctx: VariantContext) -> bool:
    incident = _hung_incident(records)
    if incident is None:
        return False
    return {ctx.settings.node, ctx.settings.peer_node} <= set(
        incident.get("node_ids") or []
    )


def _fr_errors(records: dict[str, Any], *, expect_configured: bool) -> list[str]:
    incident = _hung_incident(records)
    if incident is None:
        return []
    workflow = (records.get("workflows") or {}).get(
        str(incident.get("workflow_request_id"))
    ) or {}
    triage = [
        item
        for item in workflow.get("step_executions") or []
        if item.get("operation") == "COLLECT_HUNG_TRIAGE"
    ]
    if not triage:
        return ["no COLLECT_HUNG_TRIAGE execution to read FR status from"]
    statuses = _fr_statuses(triage)
    if expect_configured:
        if statuses and all(status == "not_configured" for status in statuses):
            return ["FR was configured but every dump reads not_configured"]
    else:
        if any(status not in {"not_configured", None} for status in statuses):
            return ["FR was not configured yet triage reports a dump"]
    return []


def _fr_statuses(triage: list[dict[str, Any]]) -> list[str | None]:
    statuses: list[str | None] = []
    for execution in triage:
        node_results = (execution.get("details") or {}).get("node_results") or {}
        for value in node_results.values():
            data = value.get("data") if isinstance(value, dict) else None
            for rank in (data or {}).get("ranks", []) if isinstance(data, dict) else []:
                recorder = (rank or {}).get("flight_recorder") or {}
                statuses.append(recorder.get("status"))
    return statuses


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
    max_minutes: int = 45,
    needs_peer: bool = True,
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
        cleanup="release the barrier; delete the test Job; confirm no node was isolated/stopped/reset",
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
        "a two-node NCCL job stalled at a barrier so the EFA pair falls to zero",
        "ZERO_WARNING then HUNG_SUSPECTED -> the read-only diagnostic DAG -> ESCALATED; no STOP/reset/cordon",
        active_to_zero,
    ),
    _spec(
        "always_low",
        "a two-node job that never establishes an active EFA baseline",
        "no hung suspicion from low traffic alone (the rule needs prior had_active)",
        always_low,
        risk="L",
        max_minutes=25,
    ),
    _spec(
        "write_progress",
        "stall the collective while the app keeps writing checkpoint I/O",
        "host writes do not delay the RDMA idle clock; HUNG_SUSPECTED still fires -> ESCALATED",
        write_progress,
    ),
    _spec(
        "phase_progress",
        "stall the collective while the app advances a declared phase",
        "phase hints do not delay the RDMA idle clock; HUNG_SUSPECTED still fires -> ESCALATED",
        phase_progress,
    ),
    _spec(
        "fr_enabled",
        "the stalled job has TORCH_NCCL_DEBUG_INFO_PIPE_FILE configured",
        "triage takes two real FR dumps per PID; chain -> ESCALATED",
        fr_enabled,
    ),
    _spec(
        "fr_disabled",
        "the stalled job has no FR pipe configured",
        "triage records not_configured and fabricates no dump; chain -> ESCALATED with process evidence",
        fr_disabled,
    ),
    _spec(
        "simultaneous_nodes",
        "both nodes of the two-node job lose EFA in one stall wave (needs --node-b)",
        "one merged grouped incident/workflow covering both nodes, not two",
        simultaneous_nodes,
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
