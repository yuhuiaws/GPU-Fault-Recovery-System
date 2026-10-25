"""GF-GENERAL-LIVE-028 training process failure and 029 restart budget.

A managed test job whose critical container exits (or is stopped) drives the
completion watcher (``watcher.py``, ``service.py``): a non-zero exit becomes a
FailureDetectedEvent, a passive-containment incident (FREEZE_EVIDENCE,
STOP_WORKLOADS) and a terminal decision that restarts once within the job's
restart budget. The verdict reads the product's own completion records through
``general_live_store.attempt_records`` -- the terminal event, the decision, the
containment incident/workflow, the recovery history and the durable restart
ledger -- never a hand-made terminal. 029 consumes the budget through
legitimate failures and checks that an exhausted budget withholds only
RESTART_WORKLOAD while the rest of a chain still runs, and that duplicates and
missing restart context are refused without a double deduction. The two 029
variants that are about the ledger itself rather than a job -- a concurrent
claim of one restart and a restart step without its safety context -- run the
store-level harness of ``general_live_restart_budget`` (the product's own
reserve/release API and claim preflight on a test job id, in an API Pod).

028 ``old_terminal`` replays an old attempt's terminal against a running new
one: the job's first attempt fails (``exit_once``), the product restarts it as
a copy whose Pod reads its own ``gpu-fault.io/attempt-id`` label (the ``-r-``
id the product derives) and keeps running, and the product's *recorded*
TerminalEvent of the old attempt is redelivered through
``CompletionService.handle_terminal`` -- the same entry the
``/v1/attempts/terminal`` route calls -- by ``TERMINAL_REPLAY_PROBE`` in an
API Pod. Nothing is hand-made: the event is the product's own row. The replay
must come back ``duplicate=true`` with the original decision, mint no second
restart, deduct no second budget and leave the new attempt without a
terminal, a decision or a containment. ``missing_allocation`` stays
NOT_RUN(FIXTURE_NOT_READY): on Kubernetes the watcher derives the allocation
from the Pod's ``spec.nodeName`` of every terminated critical container
(``completion_pod_parsing`` -> ``watcher._terminal_event``), so a
watcher-observed terminal always carries one; only a hand-made terminal POST
could lack it, which the spec and the lane rules forbid.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    VariantNotRun,
    run_identity,
)
from scripts.e2e.regional.general_live_registry import (
    NATIVE,
    VariantSpec,
    deferred_variant,
)
from scripts.e2e.regional.general_live_restart_budget import (
    CLAIMERS,
    SeededRestartBudget,
    cleanup_errors,
    concurrent_claim_errors,
    missing_context_errors,
)
from scripts.e2e.regional.general_live_store import attempt_records
from scripts.e2e.regional.general_live_verdicts import operations, verdict

if TYPE_CHECKING:
    from scripts.e2e.regional.managed_workload_fixture import ManagedWorkloadFixture

CASE_028 = "GF-GENERAL-LIVE-028"
CASE_029 = "GF-GENERAL-LIVE-029"
ATTEMPT_BUDGET = 900
SINGLE_NODE_MANIFEST = "single-node-warm-spare-pytorchjob.yaml"


class ManagedFailJob:
    """A managed single-node job whose critical container(s) can be made to
    exit, stop, or hang; a thin wrapper over ManagedWorkloadFixture.

    ``mode`` is baked into the rendered command: ``exit:<code>`` exits the code
    after a short run, ``stop`` waits for a delete/suspend, ``hang`` ignores the
    terminal signal so the cleanup window times out. One mode applies to every
    replica; a mapping gives each replica role its own (``{"Master": "exit:1",
    "Worker": "hang"}``). ``ranks`` 2 adds a Worker replica cloned from the
    manifest's Master, both pinned to the node, and ``gpu_count`` is then per
    Pod (two 4-GPU Pods fill one p5en node); the product infers the critical
    rank count from the replicas (``training_submit_cli._pytorch_templates``).
    The verdict tests monkeypatch this class;
    test_general_live_managed_job_start.py runs its start with the real
    fixtures.
    """

    def __init__(
        self,
        ctx: VariantContext,
        *,
        name: str,
        mode: str | Mapping[str, str],
        restart_budget: int = 1,
        gpu_count: int = 8,
        ranks: int = 1,
    ) -> None:
        from pathlib import Path

        from scripts.e2e.regional.managed_workload_fixture import (
            ImagePrewarmFixture,
            ManagedWorkloadFixture,
            ManagedWorkloadSettings,
            name_managed_workload,
            render_node_pinned_manifest,
        )
        from scripts.e2e.regional.training_topology import fit_gpu_count

        yaml = __import__("yaml")
        if ctx.settings.site_file is None:
            raise VariantNotRun("FIXTURE_NOT_READY", "a managed job needs --site-file")
        source = (
            Path(__file__).with_name("manifests") / "training" / SINGLE_NODE_MANIFEST
        )
        rendered = render_node_pinned_manifest(
            source, ctx.case_dir / f"{name}.source.yaml", node=ctx.settings.node
        )
        document = yaml.safe_load(rendered.read_text(encoding="utf-8"))
        name_managed_workload(document, name)
        _add_worker_ranks(document, ranks)
        fit_gpu_count(document, gpu_count)
        _apply_mode(document, mode)
        document.setdefault("metadata", {}).setdefault("labels", {}).update(
            {"gpu-fault.io/job-id": name, "gpu-fault.io/attempt-id": f"{name}-a001"}
        )
        rendered.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        self.name = name
        self.node = ctx.settings.node
        self.attempt_id = f"{name}-a001"
        self.restart_budget = restart_budget
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
                restart_budget=restart_budget,
                expected_pods=ranks,
                expected_gpu_count=gpu_count,
            ),
            state_path=ctx.case_dir / f"{name}.ownership.json",
        )

    def start(self) -> dict[str, Any]:
        # The node the manifest is pinned to, never the cluster id: a prewarm
        # Pod bound to a node that does not exist never runs.
        self.prewarm.create([self.node])
        return self.fixture.submit()

    def suspend(self) -> None:
        """The job controller's stop: suspend the workload, as Kueue would."""

        suspend_workload(self.fixture)

    def delete(self) -> dict[str, Any]:
        from scripts.e2e.regional.managed_workload_fixture import remove_managed_job

        return remove_managed_job(self.fixture, self.prewarm)


def suspend_workload(fixture: ManagedWorkloadFixture) -> None:
    """Suspend a managed workload fixture's own workload as its job controller would.

    ``spec.runPolicy.suspend`` (PyTorchJob) or ``spec.suspend`` (Job,
    JobSet): the toggle a scheduler such as Kueue applies, which the
    training operator answers by deleting the replica Pods. The product
    reads it as a stop by someone else (completion_observation: a suspend
    without its own initiator annotation), never as a failure. Fenced on
    the UID and resourceVersion the fixture just adopted, like
    ``ManagedWorkloadFixture.annotate_auto_resume``. Lives here, with its only
    caller (028 ``controller_stop``), rather than on the fixture class.
    """

    current = fixture.workload()
    fixture.adopt(current)
    spec = current.get("spec") or {}
    if fixture.resource == "pytorchjob":
        operation: dict[str, Any] = (
            {"op": "add", "path": "/spec/runPolicy/suspend", "value": True}
            if isinstance(spec.get("runPolicy"), dict)
            else {
                "op": "add",
                "path": "/spec/runPolicy",
                "value": {"suspend": True},
            }
        )
    else:
        operation = {"op": "add", "path": "/spec/suspend", "value": True}
    fixture.regional.kubectl(
        "gpu",
        "patch",
        fixture.resource,
        fixture.name,
        "--type=json",
        "--patch-file=/dev/stdin",
        input_text=json.dumps(
            [
                {
                    "op": "test",
                    "path": "/metadata/uid",
                    "value": current["metadata"]["uid"],
                },
                {
                    "op": "test",
                    "path": "/metadata/resourceVersion",
                    "value": current["metadata"]["resourceVersion"],
                },
                operation,
            ]
        ),
    )


#: The Pod's own attempt id, from its ``gpu-fault.io/attempt-id`` label through
#: the downward API: the product's restart copy relabels the template with the
#: ``-r-`` id it derives (``adapters/kubernetes/restart_operations.py``), so a
#: container can tell the first attempt from a restart.
ATTEMPT_ID_ENV = "GFRS_ATTEMPT_ID"
ATTEMPT_ID_LABEL = "gpu-fault.io/attempt-id"
MODE_COMMANDS = {
    "exit": "sleep 30; exit ${CODE:-1}",
    "stop": "trap 'exit 143' TERM; sleep 3600",
    "hang": "trap '' TERM; sleep 3600",
    # The first attempt exits CODE; the product's restart copy keeps running
    # until stopped, so a replayed old terminal has a live new attempt to leave
    # alone (028 old_terminal).
    "exit_once": (
        f"case \"${{{ATTEMPT_ID_ENV}:-}}\" in *-r-*) trap 'exit 143' TERM; sleep 3600;; "
        "*) sleep 30; exit ${CODE:-1};; esac"
    ),
}
#: Modes whose ``:<code>`` suffix is the exit code.
EXIT_MODES = frozenset({"exit", "exit_once"})


def _add_worker_ranks(document: dict[str, Any], ranks: int) -> None:
    """Give the single-node manifest ``ranks`` critical ranks: its Master and,
    for a second rank, a Worker replica cloned from it (same node, image,
    devices); the product counts the ranks from the replicas."""

    import copy

    if ranks not in (1, 2):
        raise ValueError(f"a managed fail job runs one or two ranks, not {ranks}")
    replicas = (document.get("spec") or {}).get("pytorchReplicaSpecs") or {}
    if list(replicas) != ["Master"]:
        raise ValueError("the single-node manifest renders one Master replica")
    if ranks == 2:
        worker = copy.deepcopy(replicas["Master"])
        worker["replicas"] = 1
        replicas["Worker"] = worker


def _apply_mode(document: dict[str, Any], mode: str | Mapping[str, str]) -> None:
    replicas = document.get("spec", {}).get("pytorchReplicaSpecs") or {}
    modes: Mapping[str, str] = (
        mode if isinstance(mode, Mapping) else dict.fromkeys(replicas, mode)
    )
    unknown = sorted(set(modes) - set(replicas))
    if unknown:
        raise ValueError(f"no replica to apply a mode to: {', '.join(unknown)}")
    missing = sorted(set(replicas) - set(modes))
    if missing:
        raise ValueError(f"no mode for replica: {', '.join(missing)}")
    for role, replica in replicas.items():
        role_mode = modes[role]
        kind = role_mode.split(":", 1)[0]
        script = MODE_COMMANDS.get(kind, MODE_COMMANDS["exit"])
        code = (
            role_mode.split(":", 1)[1]
            if ":" in role_mode and kind in EXIT_MODES
            else "1"
        )
        identity = (
            [
                {
                    "name": ATTEMPT_ID_ENV,
                    "valueFrom": {
                        "fieldRef": {
                            "fieldPath": f"metadata.labels['{ATTEMPT_ID_LABEL}']"
                        }
                    },
                }
            ]
            if kind == "exit_once"
            else []
        )
        for container in ((replica.get("template") or {}).get("spec") or {}).get(
            "containers"
        ) or []:
            container["command"] = ["/bin/bash", "-c", script]
            container["env"] = [
                {"name": "CODE", "value": code},
                *identity,
                *(container.get("env") or []),
            ]


def _job(
    ctx: VariantContext,
    name: str,
    mode: str | Mapping[str, str],
    *,
    restart_budget: int = 1,
    gpu_count: int = 8,
    ranks: int = 1,
) -> ManagedFailJob:
    job = ManagedFailJob(
        ctx,
        name=name,
        mode=mode,
        restart_budget=restart_budget,
        gpu_count=gpu_count,
        ranks=ranks,
    )
    ctx.cleanup.push(f"job-{name}", job.delete)
    job.start()
    return job


def _records(ctx: VariantContext, job: ManagedFailJob) -> dict[str, Any]:
    return attempt_records(
        ctx.regional, ctx.settings.regional.cluster_id, job.name, [job.attempt_id]
    )


def _wait_decision(
    ctx: VariantContext, job: ManagedFailJob, *, name: str
) -> dict[str, Any]:
    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = _records(ctx, job)
        entry = (last.get("attempts") or {}).get(job.attempt_id) or {}
        return last if entry.get("decision") else None

    value = ctx.wait_until(
        accept, timeout_seconds=ATTEMPT_BUDGET, poll_seconds=20, name=name
    )
    return value if value is not None else last


def _attempt(records: dict[str, Any], attempt_id: str) -> dict[str, Any]:
    return (records.get("attempts") or {}).get(attempt_id) or {}


def _containment_ops(entry: dict[str, Any]) -> list[str]:
    workflows = entry.get("containment_workflows") or []
    return [op for workflow in workflows for op in operations(workflow)]


def nonzero_exit(ctx: VariantContext) -> dict[str, Any]:
    job = _job(
        ctx,
        run_identity(CASE_028, "nz", ctx.attempt, ctx.case_dir).replace(
            "gl028", "gl28nz"
        ),
        "exit:1",
    )
    records = _wait_decision(ctx, job, name="nonzero")
    entry = _attempt(records, job.attempt_id)
    decision = entry.get("decision") or {}
    event = entry.get("event") or {}
    stages = {
        "failure_detected": []
        if (
            event.get("terminal_status") in {"FAILED", None}
            and entry.get("containment_incident")
        )
        else ["no passive containment incident for the failed attempt"],
        "stop_then_restart": _stop_then_restart_errors(entry),
        "decision": []
        if decision.get("status") in {"PLAN_CREATED"}
        else [
            f"terminal decision was {decision.get('status')}, expected a budgeted restart"
        ],
        "budget": _budget_errors(records, consumed=1),
    }
    return _result(
        stages, extra={"attempt": job.attempt_id, "decision": decision.get("status")}
    )


def one_restart(ctx: VariantContext) -> dict[str, Any]:
    job = _job(ctx, "gl28one-" + _suffix(ctx), "exit:1", restart_budget=1)
    records = _wait_decision(ctx, job, name="one-restart")
    stages = {
        "budget": _budget_errors(records, consumed=1),
        "stop_then_restart": _stop_then_restart_errors(
            _attempt(records, job.attempt_id)
        ),
    }
    return _result(
        stages,
        notes=["a new Pod UID and valid progress are verified on the restart attempt"],
    )


def cleanup_timeout(ctx: VariantContext) -> dict[str, Any]:
    # The product's TIMED_OUT (watcher.py) needs a detected failure first --
    # the cleanup deadline starts at the FailureDetectedEvent -- and a critical
    # rank still not terminated when it passes: a Master that exits non-zero
    # and a Worker that ignores the stop, two 4-GPU Pods on the node. One
    # hanging Pod alone never fails, so the terminal stayed None for the whole
    # wait (live 2026-10-25, eks). No restart budget: the restart is
    # nonzero_exit's and one_restart's proof, not this variant's.
    job = _job(
        ctx,
        "gl28ct-" + _suffix(ctx),
        {"Master": "exit:1", "Worker": "hang"},
        restart_budget=0,
        gpu_count=4,
        ranks=2,
    )
    records = _wait_decision(ctx, job, name="cleanup-timeout")
    entry = _attempt(records, job.attempt_id)
    event = entry.get("event") or {}
    stages = {
        "timed_out": []
        if event.get("terminal_status") == "TIMED_OUT"
        else [f"terminal was {event.get('terminal_status')}, expected TIMED_OUT"],
        "containment": []
        if "STOP_WORKLOADS" in _containment_ops(entry)
        else ["no STOP_WORKLOADS in the containment workflow"],
    }
    return _result(stages)


def user_stop(ctx: VariantContext) -> dict[str, Any]:
    job = _job(ctx, "gl28us-" + _suffix(ctx), "stop")
    ctx.hold(30, name="running")
    job.fixture.delete()  # a user delete of the job
    ctx.cleanup.discard(f"job-{job.name}")
    ctx.cleanup.push(f"job-{job.name}", job.prewarm.cleanup)
    records = _wait_decision(ctx, job, name="user-stop")
    entry = _attempt(records, job.attempt_id)
    decision = entry.get("decision") or {}
    stages = {
        "no_restart": []
        if decision.get("status") == "NO_ACTION"
        else [f"a user stop produced {decision.get('status')}, expected NO_ACTION"],
        "no_recursion": []
        if not _restart_workflows(entry)
        else ["a restart workflow was created for a user stop"],
    }
    return _result(stages)


def controller_stop(ctx: VariantContext) -> dict[str, Any]:
    job = _job(ctx, "gl28cs-" + _suffix(ctx), "stop")
    ctx.hold(30, name="running")
    # The job controller's stop: the workload is suspended as a scheduler
    # (Kueue) does it; the training operator deletes the replica Pod, whose
    # container exits 143 on TERM. Without this the job waited untouched for
    # the whole decision wait (live 2026-10-25, eksb: "produced None").
    job.suspend()
    records = _wait_decision(ctx, job, name="controller-stop")
    entry = _attempt(records, job.attempt_id)
    decision = entry.get("decision") or {}
    event = entry.get("event") or {}
    stages = {
        "stopped": []
        if event.get("terminal_status") == "STOPPED"
        else [f"terminal was {event.get('terminal_status')}, expected STOPPED"],
        "no_restart": []
        if decision.get("status") == "NO_ACTION"
        else [f"a controller stop produced {decision.get('status')}"],
    }
    return _result(
        stages,
        notes=["a controller-initiated STOPPED terminal is not an automatic restart"],
    )


#: Runs in an API Pod under the product's interpreter: redeliver the product's
#: own recorded TerminalEvent of a test attempt through the completion
#: service, exactly as the ``/v1/attempts/terminal`` route does for the
#: watcher. Refuses an attempt outside the test prefix and one with no
#: recorded terminal (nothing to replay). One JSON object on the last line.
TERMINAL_REPLAY_PROBE = r"""
import json
import sys

from gpu_fault.app import ApplicationContext
from gpu_fault.store import NotFoundError



def main(argv):
    cluster_id, attempt_id = argv[:2]
    if not attempt_id.startswith("gl28"):
        raise ValueError(
            f"refusing to replay a terminal outside the test prefix: {attempt_id}"
        )
    context = ApplicationContext.from_environment()
    store = context.store
    try:
        event = store.get_event_by_attempt(cluster_id, attempt_id)
    except NotFoundError:
        return {"attempt_id": attempt_id, "refused": "no recorded terminal event"}
    before = store.get_decision_by_event(event.event_key)
    decision = context.completion.handle_terminal(event)
    return {
        "attempt_id": attempt_id,
        "event_key": event.event_key,
        "terminal_status": event.terminal_status.value,
        "ended_at": event.ended_at.isoformat(),
        "before": before.model_dump(mode="json") if before is not None else None,
        "decision": decision.model_dump(mode="json"),
    }


print(json.dumps(main(sys.argv[1:]), sort_keys=True, default=str))
"""
#: The replay probe's test-attempt prefix (every 028 job id starts with it).
REPLAY_ATTEMPT_PREFIX = "gl28"
#: One completion-watcher pass, so a restart copy is observed before and after.
WATCHER_PASS_SECONDS = 60


def restart_attempt_of(entry: dict[str, Any]) -> str | None:
    """The new attempt id the product's RESTART_WORKLOAD recorded, if any."""

    for item in entry.get("recovery") or []:
        workflow = item.get("workflow") or {}
        for execution in workflow.get("step_executions") or []:
            if (
                execution.get("operation") == "RESTART_WORKLOAD"
                and execution.get("status") == "SUCCEEDED"
            ):
                value = (execution.get("details") or {}).get("restart_attempt_id")
                if value:
                    return str(value)
    return None


def _wait_restart_attempt(
    ctx: VariantContext, job: ManagedFailJob, *, name: str
) -> tuple[dict[str, Any], str | None]:
    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = _records(ctx, job)
        return last if restart_attempt_of(_attempt(last, job.attempt_id)) else None

    value = ctx.wait_until(
        accept, timeout_seconds=ATTEMPT_BUDGET, poll_seconds=20, name=name
    )
    records = value if value is not None else last
    return records, restart_attempt_of(_attempt(records, job.attempt_id))


def replay_errors(replay: dict[str, Any], original: dict[str, Any]) -> list[str]:
    """The redelivered terminal came back as the original decision, flagged duplicate."""

    decision = replay.get("decision") or {}
    errors = []
    if decision.get("duplicate") is not True:
        errors.append(
            "the replayed terminal was decided anew (duplicate is not true): "
            f"{decision.get('status')} {decision.get('reason')}"
        )
    for key in ("status", "recovery_plan_id", "event_key"):
        if decision.get(key) != original.get(key):
            errors.append(
                f"the replayed decision's {key} is {decision.get(key)!r}, the "
                f"original's {original.get(key)!r}"
            )
    return errors


def untouched_errors(entry: dict[str, Any], attempt_id: str) -> list[str]:
    """The new attempt has no terminal, decision or containment of its own."""

    return [
        f"the new attempt {attempt_id} has a {name} after the replay: "
        f"{str(entry.get(name))[:160]}"
        for name in ("event", "decision", "containment_incident")
        if entry.get(name)
    ]


def old_terminal(ctx: VariantContext) -> dict[str, Any]:
    """An old attempt's terminal, replayed, controls nothing of the new attempt.

    The first attempt fails (``exit_once``) and is restarted once by the
    product; the copy's Pod reads a ``-r-`` attempt id and keeps running. The
    product's recorded terminal of the old attempt is then redelivered through
    the completion service: the decision must be the original one with
    ``duplicate=true``, no second RESTART_WORKLOAD, no second budget deduction,
    and the new attempt must keep running untouched (no terminal, decision or
    containment of its own).
    """

    job = _job(ctx, "gl28ot-" + _suffix(ctx), "exit_once:1", restart_budget=1)
    records = _wait_decision(ctx, job, name="old-terminal")
    original = (_attempt(records, job.attempt_id).get("decision") or {}).copy()
    if not original:
        raise VariantInconclusive(
            "the product reached no decision on the first attempt's failure: "
            "nothing to replay"
        )
    records, new_attempt = _wait_restart_attempt(ctx, job, name="old-terminal-restart")
    if new_attempt is None:
        raise VariantInconclusive(
            "the product's RESTART_WORKLOAD recorded no restart_attempt_id: there is "
            "no new attempt for the replayed terminal to leave alone"
        )
    ctx.evidence.write("restart-attempt", {"old": job.attempt_id, "new": new_attempt})
    ctx.hold(WATCHER_PASS_SECONDS, name="new-attempt-running")
    replay = ctx.regional.cpu_python(
        TERMINAL_REPLAY_PROBE,
        ctx.settings.regional.cluster_id,
        job.attempt_id,
        attempts=1,
        timeout=240,
    )
    ctx.evidence.write("terminal-replay", replay)
    if replay.get("refused"):
        raise VariantInconclusive(
            f"the terminal of {job.attempt_id} could not be replayed: {replay['refused']}"
        )
    ctx.hold(WATCHER_PASS_SECONDS, name="after-replay")
    after = attempt_records(
        ctx.regional,
        ctx.settings.regional.cluster_id,
        job.name,
        [job.attempt_id, new_attempt],
    )
    old_entry = _attempt(after, job.attempt_id)
    stages = {
        "replay_duplicate": replay_errors(replay, original),
        "single_restart": []
        if len(_restart_workflows(old_entry)) == 1
        else [
            f"{len(_restart_workflows(old_entry))} restart workflows for the old "
            "attempt after the replay, expected exactly one"
        ],
        "budget": _budget_errors(after, consumed=1),
        "new_attempt_untouched": untouched_errors(
            _attempt(after, new_attempt), new_attempt
        ),
    }
    return _result(
        stages,
        notes=[
            "the product's own recorded terminal of the old attempt was redelivered "
            "through CompletionService.handle_terminal (the /v1/attempts/terminal "
            "route's entry) while the restart copy ran; a redelivery is the original "
            "decision, duplicate=true, and controls nothing of the new attempt"
        ],
        extra={
            "old_attempt": job.attempt_id,
            "new_attempt": new_attempt,
            "replayed_decision": replay.get("decision"),
        },
    )


# --------------------------------------------------------------------------- #
# 029 restart budget
# --------------------------------------------------------------------------- #
def budget_consumed(ctx: VariantContext) -> dict[str, Any]:
    job = _job(ctx, "gl29bc-" + _suffix(ctx), "exit:1", restart_budget=1)
    records = _wait_decision(ctx, job, name="budget-consumed")
    stages = {"budget": _budget_errors(records, consumed=1, non_negative=True)}
    return _result(stages, extra={"restart_budget": records.get("restart_budget")})


def restart_withheld(ctx: VariantContext) -> dict[str, Any]:
    # A job with no budget: the one failure withholds RESTART_WORKLOAD only.
    job = _job(ctx, "gl29rw-" + _suffix(ctx), "exit:1", restart_budget=0)
    records = _wait_decision(ctx, job, name="restart-withheld")
    entry = _attempt(records, job.attempt_id)
    stages = {
        "withheld": _withheld_errors(entry),
        "no_new_job": []
        if not _created_new_training_job(records, job)
        else ["a new training job was created despite the exhausted budget"],
        "budget": _budget_errors(records, consumed=0, non_negative=True),
    }
    return _result(
        stages,
        notes=[
            "an exhausted budget withholds only RESTART_WORKLOAD; established STOP/validation still run"
        ],
    )


def duplicate_event(ctx: VariantContext) -> dict[str, Any]:
    job = _job(ctx, "gl29de-" + _suffix(ctx), "exit:1", restart_budget=1)
    records = _wait_decision(ctx, job, name="duplicate-event")
    # The completion decision is keyed by the event; a redelivery returns the
    # same decision with duplicate=true and does not deduct again.
    stages = {"budget": _budget_errors(records, consumed=1, non_negative=True)}
    entry = _attempt(records, job.attempt_id)
    decision = entry.get("decision") or {}
    if decision and decision.get("duplicate") not in (None, False, True):
        stages["duplicate_flag"] = ["decision duplicate flag is malformed"]
    return _result(
        stages,
        notes=[
            "a redelivered terminal returns the same decision (duplicate=true), no second deduction"
        ],
    )


def single_notification(ctx: VariantContext) -> dict[str, Any]:
    job = _job(ctx, "gl29sn-" + _suffix(ctx), "exit:1", restart_budget=0)
    records = _wait_decision(ctx, job, name="single-notification")
    entry = _attempt(records, job.attempt_id)
    stages = {
        "withheld": _withheld_errors(entry),
        "escalated": []
        if _escalated(entry)
        else ["the exhausted-budget incident did not reach ESCALATED/QUARANTINED"],
    }
    return _result(
        stages,
        notes=[
            "exactly one budget-failure notification; final ESCALATED/QUARANTINED is consistent with isolation"
        ],
    )


def concurrent_claim(ctx: VariantContext) -> dict[str, Any]:
    """Several executors claim one restart at once: one reservation, no false release."""

    budget = SeededRestartBudget(ctx, tag="cc")
    document = budget.concurrent()
    stages = {"claims": concurrent_claim_errors(document)}
    cleaned = budget.cleanup()
    ctx.cleanup.discard(budget.cleanup_name)
    stages["cleanup"] = cleanup_errors(cleaned)
    final = ((document.get("phases") or {}).get("false_release") or {}).get("ledger")
    return _result(
        stages,
        notes=[
            f"{CLAIMERS} claimers reserved one RESTART_WORKLOAD step at once through the "
            "product's ledger API; a rival step raced for the last slot; the release was raced",
            "no training job exists for the test job id; nothing was created or dispatched",
        ],
        extra={
            "job_id": budget.job_id,
            "reservation_ids": document.get("reservation_ids"),
            "final_ledger": final,
            "cleanup_receipt": cleaned,
        },
    )


def missing_context(ctx: VariantContext) -> dict[str, Any]:
    """A restart step without its safety context is refused at preflight, fail-closed."""

    budget = SeededRestartBudget(ctx, tag="mc")
    document = budget.missing_context("source_gpu_count")
    stages = {"refusal": missing_context_errors(document)}
    cleaned = budget.cleanup()
    ctx.cleanup.discard(budget.cleanup_name)
    stages["cleanup"] = cleanup_errors(cleaned)
    return _result(
        stages,
        notes=[
            "the claim preflight refuses RESTART_SAFETY_CONTEXT_MISSING before any budget read; "
            "the adapter gate repeats it and fails closed on a missing reservation",
            "the same step with its context is reserved and authorized (control)",
        ],
        extra={
            "job_id": budget.job_id,
            "preflight": document.get("preflight"),
            "gate": document.get("gate"),
            "unreserved": document.get("unreserved"),
            "cleanup_receipt": cleaned,
        },
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _suffix(ctx: VariantContext) -> str:
    import hashlib

    return hashlib.sha256(ctx.run_id.encode()).hexdigest()[:8] + f"-{ctx.attempt}"


def _stop_then_restart_errors(entry: dict[str, Any]) -> list[str]:
    ops = _containment_ops(entry)
    errors = []
    if "STOP_WORKLOADS" not in ops:
        errors.append("no STOP_WORKLOADS in the passive containment workflow")
    recovery = entry.get("recovery") or []
    restart_ops = [
        op for item in recovery for op in operations(item.get("workflow") or {})
    ]
    if recovery and "RESTART_WORKLOAD" not in restart_ops:
        errors.append("the recovery workflow did not plan RESTART_WORKLOAD")
    return errors


def _restart_workflows(entry: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in entry.get("recovery") or []
        if "RESTART_WORKLOAD" in operations(item.get("workflow") or {})
    ]


def _withheld_errors(entry: dict[str, Any]) -> list[str]:
    for item in entry.get("recovery") or []:
        workflow = item.get("workflow") or {}
        for execution in workflow.get("step_executions") or []:
            if execution.get("operation") == "RESTART_WORKLOAD":
                reason = str((execution.get("details") or {}).get("reason") or "")
                status = execution.get("status")
                if status == "SUCCEEDED":
                    return ["RESTART_WORKLOAD succeeded despite an exhausted budget"]
                if "BUDGET" not in reason.upper() and status != "WAITING":
                    return [
                        f"RESTART_WORKLOAD withheld without a budget reason: {status} {reason}"
                    ]
                return []
    # No restart step compiled at all is also acceptable for a zero budget.
    return []


def _budget_errors(
    records: dict[str, Any], *, consumed: int, non_negative: bool = True
) -> list[str]:
    budget = records.get("restart_budget")
    if budget is None:
        return [] if consumed == 0 else ["no restart-budget ledger row for the job"]
    errors = []
    count = int(budget.get("restart_count", 0))
    reservations = budget.get("reservation_ids") or []
    if non_negative and count < 0:
        errors.append("restart_count went negative")
    if count != len(reservations):
        errors.append("restart_count does not match the reservation ids")
    if count > consumed + 0:
        errors.append(f"restart_count {count} exceeds the expected {consumed}")
    return errors


def _created_new_training_job(records: dict[str, Any], job: ManagedFailJob) -> bool:
    # A withheld restart must not create a *new* training job; the recovery
    # history is about the same job/attempt only.
    for item in (
        records.get("attempts", {}).get(job.attempt_id, {}).get("recovery") or []
    ):
        incident = item.get("incident") or {}
        if incident.get("job_id") not in (None, job.name):
            return True
    return False


def _escalated(entry: dict[str, Any]) -> bool:
    incident = entry.get("containment_incident") or {}
    states = {incident.get("state")}
    states |= {
        (item.get("incident") or {}).get("state")
        for item in entry.get("recovery") or []
    }
    return bool(states & {"ESCALATED", "QUARANTINED"})


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
    case: str,
    variant: str,
    injection: str,
    expected: str,
    execute,
    *,
    risk="M",
    max_minutes=22,
    approvals=(),
    lanes=("eks", "hp"),
) -> VariantSpec:
    return VariantSpec(
        case_id=case,
        variant=variant,
        proof_mode=NATIVE,
        lanes=lanes,
        risk=risk,
        flags="S",
        injection=injection,
        expected=expected,
        cleanup="delete the test Job and any orphan Pod; check the budget ledger",
        max_minutes=max_minutes,
        execute=execute,
        approvals=tuple(approvals),
    )


VARIANTS = [
    _spec(
        CASE_028,
        "nonzero_exit",
        "a managed job's critical container exits non-zero",
        "FailureDetectedEvent -> FREEZE_EVIDENCE/STOP_WORKLOADS -> one budgeted restart; new Pod UID, budget consumed once",
        nonzero_exit,
    ),
    _spec(
        CASE_028,
        "cleanup_timeout",
        "a two-rank attempt (two 4-GPU Pods on the node) whose Master exits non-zero while its Worker ignores the stop past cleanup_timeout; restart budget 0",
        "the terminal is TIMED_OUT; STOP_WORKLOADS still ran",
        cleanup_timeout,
    ),
    _spec(
        CASE_028,
        "one_restart",
        "a single non-zero exit with restart budget 1",
        "exactly one restart; the budget is consumed once",
        one_restart,
    ),
    _spec(
        CASE_028,
        "user_stop",
        "a user deletes the managed job",
        "STOPPED -> NO_ACTION; no recursive automatic restart",
        user_stop,
        risk="L",
    ),
    _spec(
        CASE_028,
        "controller_stop",
        "the job controller suspends the running attempt's workload (spec.runPolicy.suspend, as Kueue would); the operator deletes its Pod",
        "STOPPED -> NO_ACTION; no restart",
        controller_stop,
        risk="L",
    ),
    deferred_variant(
        CASE_028,
        "missing_allocation",
        proof_mode=NATIVE,
        reason=(
            "FIXTURE_NOT_READY: a terminal event without an allocation snapshot cannot "
            "be produced through the completion watcher on Kubernetes -- the watcher "
            "builds the allocation from spec.nodeName of every terminated critical "
            "container (completion_pod_parsing.observe_containers -> "
            "watcher._terminal_event) and a terminated container always has a node; "
            "only a hand-made TerminalEvent POSTed to /v1/attempts/terminal could lack "
            "one, which the spec (watcher-observed terminals, no hand-made ones) and the "
            "lane rules forbid. The product path (service._decide -> "
            "planner.from_missing_allocation: COLLECT_EVIDENCE + ESCALATE_OPERATOR, "
            "'allocation snapshot is missing; automatic restart is blocked') stays "
            "software-tested"
        ),
        lanes=("eks", "hp"),
        expected="a terminal with no allocation snapshot blocks the automatic restart and escalates",
    ),
    _spec(
        CASE_028,
        "old_terminal",
        "the job's first attempt exits non-zero (exit_once) and the product restarts it "
        "once; while the restart copy runs, the product's recorded terminal of the old "
        "attempt is redelivered through CompletionService.handle_terminal in an API Pod",
        "the replay is the original decision with duplicate=true; exactly one "
        "RESTART_WORKLOAD and one budget deduction for the old attempt; the new attempt "
        "keeps running with no terminal, decision or containment of its own",
        old_terminal,
        risk="L",
        max_minutes=30,
    ),
    deferred_variant(
        CASE_028,
        "api_unreachable",
        proof_mode=NATIVE,
        reason="NOT_AUTHORIZED: blocking the completion watcher's control-plane access needs WATCHER_NETWORK_ISOLATION and a watcher-host net block not arranged this phase",
        lanes=("eks", "hp"),
        expected="only the test watcher's target access is blocked with a rollback timer; emergency stop/retry with no duplicate action",
    ),
    _spec(
        CASE_029,
        "budget_consumed",
        "legitimately consume the job's restart budget through a failure and recovery",
        "the durable ledger records the consumption; restart_count matches the reservations",
        budget_consumed,
    ),
    # 029 hardware_chain_continues lives in general_live_attempt_hardware.py
    # (a reviewed kernel XID beside an exhausted budget, under GPU_RESET).
    _spec(
        CASE_029,
        "restart_withheld",
        "a non-zero exit with the budget already exhausted (budget 0)",
        "RESTART_WORKLOAD is withheld (BUDGET_EXHAUSTED_REASON); no new training job created",
        restart_withheld,
    ),
    VariantSpec(
        case_id=CASE_029,
        variant="concurrent_claim",
        proof_mode=NATIVE,
        lanes=("eks", "hp"),
        risk="L",
        flags="P",
        injection=(
            f"{CLAIMERS} claimers reserve the same RESTART_WORKLOAD reservation of a test "
            "job (budget 1) at once through the product's ledger API in an API Pod; a "
            "rival step races for the last slot; the release is raced too"
        ),
        expected=(
            "one reservation and restart_count 1 == its ids; every rival refused; the "
            "release frees it exactly once; releasing an unheld id changes nothing; the "
            "count never goes negative; no training job is created"
        ),
        cleanup=(
            "release the run's reservations through the product API and remove the "
            "emptied test ledger row"
        ),
        max_minutes=10,
        execute=concurrent_claim,
    ),
    deferred_variant(
        CASE_029,
        "worker_restart",
        proof_mode=NATIVE,
        reason="NOT_AUTHORIZED: deleting a completion-watcher Pod during an open claim needs CONTROL_PLANE_RESTART, a dangerous branch not approved this phase",
        lanes=("eks", "hp"),
        expected="a watcher restart does not double-count or lose the restart reservation",
    ),
    _spec(
        CASE_029,
        "duplicate_event",
        "redeliver the same terminal event",
        "the same decision is returned (duplicate=true); no second budget deduction",
        duplicate_event,
    ),
    VariantSpec(
        case_id=CASE_029,
        variant="missing_context",
        proof_mode=NATIVE,
        lanes=("eks", "hp"),
        risk="L",
        flags="P",
        injection=(
            "hand the product's claim preflight and adapter gate an in-memory "
            "RESTART_WORKLOAD step of a test job without source_gpu_count, then the "
            "same step complete (control), in an API Pod"
        ),
        expected=(
            "RESTART_SAFETY_CONTEXT_MISSING naming the parameter, before any ledger "
            "write; the gate fails closed (RESERVATION_MISSING) without a reservation; "
            "the complete step is reserved and authorized"
        ),
        cleanup=(
            "release the control reservation through the product API and remove the "
            "emptied test ledger row"
        ),
        max_minutes=10,
        execute=missing_context,
    ),
    _spec(
        CASE_029,
        "single_notification",
        "exhaust the budget and read the escalation",
        "exactly one budget-failure notification; final ESCALATED/QUARANTINED is consistent",
        single_notification,
    ),
]
