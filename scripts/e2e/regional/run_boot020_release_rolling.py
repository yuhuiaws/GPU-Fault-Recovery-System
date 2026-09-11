from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Protocol

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_fault_release import regional_admin_commands  # noqa: E402
from gpu_fault_release import regional_deployment_inventory  # noqa: E402
from gpu_fault_release import regional_release_config  # noqa: E402
from gpu_fault_release import regional_release_diff  # noqa: E402
from gpu_fault_release import rollout as rollout_regional_release  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    EvidenceRecorder,
    utc_now,
)

CASE_ID = "GF-REGIONAL-BOOT-020"
CONFIRMATION = "RUN_BOOT020_RELEASE_ROLLING"
EXPECTED_KINDS = {
    "noop": "NOOP",
    "control_plane": "CONTROL_PLANE_ONLY",
    "executor": "DATA_PLANE_COMPATIBLE",
    "agent": "DATA_PLANE_COMPATIBLE",
    "full": "FULL",
}
RTO_LIMITS = {
    "control_plane": {"safe": 1200.0, "full": 1500.0},
    "executor": {"safe": 900.0, "full": 1200.0, "resume": 1200.0},
    "agent": {"safe": 1800.0, "full": 2100.0},
    "full": {"safe": 2700.0, "full": 3000.0},
}


def configure_gpu_kubeconfig(value: Path | None) -> Path:
    candidate = value or (
        Path(os.environ["KUBECONFIG"]) if os.getenv("KUBECONFIG") else None
    )
    if candidate is None:
        raise SystemExit(
            "--execute requires --gpu-kubeconfig or an existing KUBECONFIG"
        )
    resolved = candidate.expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"GPU kubeconfig does not exist: {resolved}")
    os.environ["KUBECONFIG"] = str(resolved)
    return resolved


class InjectedAcceptanceFailure(RuntimeError):
    pass


class AcceptanceCheckError(RuntimeError):
    """A live observation contradicted the BOOT-020 contract.

    Raised explicitly rather than through ``assert``: ``python -O`` strips
    ``assert`` statements, and a runner whose checks vanish under an
    optimisation flag would report a constant PASS.
    """


def _require(condition: bool, message: str, context: Any = None) -> None:
    if not condition:
        detail = f"{message}: {context!r}" if context is not None else message
        raise AcceptanceCheckError(detail)


EXECUTOR_DEPLOYMENT = regional_deployment_inventory.GPU_EXECUTOR_DEPLOYMENT
REQUIRED_EXECUTOR_PIN = "required-regional-executor-artifact-sha256"
COMPATIBLE_EXECUTOR_PINS = "compatible-regional-executor-artifact-sha256s"


class ReleaseRollingBackend(Protocol):
    def classify(self, scenario: str) -> dict[str, Any]: ...

    def snapshot(self, scenario: str, *, live: bool = True) -> dict[str, Any]: ...

    def deploy(
        self,
        scenario: str,
        *,
        diff: dict[str, Any],
        fault_phase: str | None = None,
        resume: bool = False,
        auto_rollback: bool | None = None,
    ) -> dict[str, Any]: ...


def _assert_kind(scenario: str, diff: dict[str, Any]) -> None:
    _require(
        diff["kind"] == EXPECTED_KINDS[scenario], "classification", (scenario, diff)
    )


def _assert_noop(before: dict[str, Any], after: dict[str, Any]) -> None:
    _require(
        before["live"] == after["live"], "NOOP changed live state", (before, after)
    )
    _require(
        before["cpu_generations"] == after["cpu_generations"],
        "NOOP rolled a CPU Deployment",
        (before, after),
    )
    _require(
        before["gpu_generations"] == after["gpu_generations"],
        "NOOP rolled a GPU Deployment",
        (before, after),
    )


def _assert_control_plane_only(
    before: dict[str, Any],
    after: dict[str, Any],
) -> None:
    _require(
        before["live"]["clusters"] == after["live"]["clusters"],
        "CONTROL_PLANE_ONLY changed a GPU cluster",
        (before, after),
    )
    _require(
        before["gpu_generations"] == after["gpu_generations"],
        "CONTROL_PLANE_ONLY rolled a GPU Deployment",
        (before, after),
    )
    _require(
        before["cpu_generations"] != after["cpu_generations"],
        "CONTROL_PLANE_ONLY rolled no CPU Deployment",
        (before, after),
    )


def _generation_changes(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> dict[str, set[str]]:
    """Per cluster, the GPU Deployments whose generation moved."""

    changes: dict[str, set[str]] = {}
    for cluster_id in set(before) | set(after):
        old = before.get(cluster_id) or {}
        new = after.get(cluster_id) or {}
        moved = {name for name in set(old) | set(new) if old.get(name) != new.get(name)}
        if moved:
            changes[cluster_id] = moved
    return changes


def _assert_data_plane_changed(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    scenario: str,
) -> None:
    before_clusters = before["live"]["clusters"]
    after_clusters = after["live"]["clusters"]
    _require(
        set(before_clusters) == set(after_clusters),
        "DATA_PLANE_COMPATIBLE changed the cluster set",
        (before, after),
    )
    changed = [
        cluster_id
        for cluster_id in before_clusters
        if before_clusters[cluster_id] != after_clusters[cluster_id]
    ]
    _require(bool(changed), "DATA_PLANE_COMPATIBLE did not change any GPU data plane")
    generations = _generation_changes(
        before["gpu_generations"], after["gpu_generations"]
    )
    if scenario == "executor":
        # An executor-only release rolls the executor Deployment and nothing
        # else on the data plane; a collector or watcher generation moving
        # means the release touched more than its classification claims.
        _require(
            bool(generations),
            "executor-only release rolled no GPU Deployment",
            (before, after),
        )
        for cluster_id, moved in generations.items():
            _require(
                moved == {EXECUTOR_DEPLOYMENT},
                f"executor-only release rolled non-executor Deployments in {cluster_id}",
                sorted(moved),
            )


def _assert_rollback_rto(scenario: str, result: dict[str, Any]) -> None:
    timing = result.get("rollback_timing")
    _require(isinstance(timing, dict), "rollback timing missing", (scenario, result))
    timing = dict(timing or {})
    safe = float(timing["t_safe_seconds"])
    full = float(timing["t_full_seconds"])
    _require(0 <= safe <= full, "rollback timing order", (scenario, timing))
    _require(
        safe <= RTO_LIMITS[scenario]["safe"], "T_safe over limit", (scenario, timing)
    )
    _require(
        full <= RTO_LIMITS[scenario]["full"], "T_full over limit", (scenario, timing)
    )


def _assert_rollback_scope(scenario: str, result: dict[str, Any]) -> None:
    plan = result.get("rollback_plan")
    _require(isinstance(plan, dict), "rollback plan missing", (scenario, result))
    plan = dict(plan or {})
    clusters = plan.get("clusters") or {}
    # The rollback plan names data-plane components in lower case
    # ("collector", "executor", "reconciler", "watcher", "agent"); live
    # 2026-09-07 the executor stage failed on an upper-case comparison after a
    # correct rollback.
    components = {
        str(component).lower() for values in clusters.values() for component in values
    }
    global_components = {
        str(component).lower() for component in plan.get("global_components") or []
    }
    # An empty plan means the rollback had nothing to restore, which for an
    # injected failure after a real change is not a rollback at all.
    _require(
        bool(components) or bool(global_components),
        "rollback plan names no components",
        plan,
    )
    if scenario == "control_plane":
        _require(plan.get("restores_data_plane") is False, "CPU rollback scope", plan)
        _require(bool(global_components), "CPU rollback has no global components", plan)
    elif scenario == "executor":
        _require("executor" in components, "executor rollback scope", plan)
        _require("agent" not in components, "executor rollback touched agents", plan)
    elif scenario == "agent":
        _require("agent" in components, "agent rollback scope", plan)
        _require(
            plan.get("needs_controller") is True, "agent rollback controller", plan
        )
    elif scenario == "full":
        _require(bool(components), "FULL rollback restored no data plane", plan)


def _assert_resume_rto(scenario: str, result: dict[str, Any]) -> None:
    duration = float(result["operation_duration_seconds"])
    _require(
        duration <= RTO_LIMITS[scenario]["resume"],
        "resume over limit",
        (scenario, result),
    )


def _assert_rollback_snapshot(
    before: dict[str, Any],
    rolled_back: dict[str, Any],
) -> None:
    _require(
        rolled_back["live"] == before["live"],
        "rollback did not restore live state",
        (before, rolled_back),
    )


def _pins(result: dict[str, Any]) -> dict[str, Any]:
    pins = result.get("pins")
    _require(isinstance(pins, dict), "deploy result carries no pins", result)
    pins = dict(pins or {})
    for key in ("required", "compatible", "candidate", "previous_required"):
        _require(key in pins, f"pins lack {key}", pins)
    return pins


def _assert_executor_pins(result: dict[str, Any], *, phase: str) -> None:
    """The two-phase executor pin window at each executor-stage checkpoint.

    Staging (``cpu-staged``) keeps the previous artifact required and adds the
    candidate to the compatible set, so old and new executors both pass
    readiness while the data plane rolls; finalize promotes the candidate to
    required and empties the window; a rollback restores the previous required
    pin and drops the candidate. Reading the state's phase alone does not show
    which of these the metadata ConfigMap actually holds.
    """

    pins = _pins(result)
    required = str(pins["required"] or "")
    compatible = {str(item) for item in pins["compatible"] or []}
    candidate = str(pins["candidate"] or "")
    previous = str(pins["previous_required"] or "")
    _require(len(candidate) == 64, "candidate executor pin missing", pins)
    if phase == "rolled-back":
        _require(bool(previous) and required == previous, "rollback restored pin", pins)
        _require(candidate not in compatible, "rollback kept the candidate", pins)
    elif phase == "staged":
        _require(bool(previous) and required == previous, "staged required pin", pins)
        _require(candidate in compatible, "staged window lacks candidate", pins)
    elif phase == "finalized":
        _require(required == candidate, "finalized required pin", pins)
        _require(not compatible, "finalized window not empty", pins)
    else:
        raise ValueError(f"unknown pin phase: {phase}")


STAGES = ("noop", "control_plane", "executor", "agent", "full")
# The stage key whose presence in the evidence proves a stage ran to its end.
STAGE_TERMINAL_KEY = {
    "noop": "noop_after",
    "control_plane": "control_plane_after",
    "executor": "executor_after",
    "agent": "agent_after",
    "full": "full_after",
}


def _passed_marker(stage: str) -> str:
    """The record written only once every assertion in ``stage`` has passed.

    ``STAGE_TERMINAL_KEY`` proves a stage recorded its last observation, but the
    terminal assertions run *after* that record, so a stage can hold its
    terminal key and still have failed. This marker is written strictly last, so
    its presence is the only proof a stage may be replayed on a resume.
    """

    return f"{stage}_passed"


def _stage_owns(key: str, stage: str) -> bool:
    return key == stage or key.startswith(f"{stage}_")


def resume_target(document: dict[str, Any]) -> tuple[int, str | None]:
    """The first stage the evidence does not record as fully passed.

    Returns its index and name, or ``(len(STAGES), None)`` when every stage
    already carries its end-of-stage marker and there is nothing to resume.
    """

    stages = document.get("stages", {})
    for index, stage in enumerate(STAGES):
        if _passed_marker(stage) not in stages:
            return index, stage
    return len(STAGES), None


def _assert_next_noop(
    backend: ReleaseRollingBackend,
    recorder: EvidenceRecorder,
    scenario: str,
) -> None:
    """After a scenario is applied, the same config must classify NOOP.

    Recorded as a stage so a rerun against the same evidence does not re-read
    a live state that has since moved on to the next scenario.
    """

    following = recorder.stage(
        f"{scenario}_next_classification",
        lambda: backend.classify(scenario),
    )
    _require(following["kind"] == "NOOP", "next classification", following)


class _SnapshotChain:
    """Hand a stage's fresh ``*_after`` snapshot to the next stage's ``*_before``.

    A live snapshot costs 20-30 s of kubectl reads. Between stage N's
    ``*_after`` and stage N+1's ``*_before`` this driver only classifies (a
    read-only ``classify_release``) and writes recorder markers, so the two
    would observe the same live state and the second read is pure cost.

    The reuse is allowed only when the ``*_after`` was observed by this very
    process: a record replayed from the evidence file describes an older
    observation, and a ``--resume`` or ``--start-stage`` run has a convergence
    deploy or an unknown history between the file's ``*_after`` and now. Those
    paths, and ``noop_before`` (no predecessor), always observe afresh. A
    reused record carries ``reused_from`` so the evidence says what it is, and
    its ``next_deploy`` -- the one config-dependent field, informational only --
    is the new stage's own classification, which is what a fresh
    ``next_deploy`` returns for a committed ``complete`` release.
    """

    def __init__(self) -> None:
        self._carry: tuple[str, dict[str, Any]] | None = None

    def after(
        self,
        backend: ReleaseRollingBackend,
        recorder: EvidenceRecorder,
        stage: str,
    ) -> dict[str, Any]:
        key = f"{stage}_after"
        fresh = False

        def observe() -> dict[str, Any]:
            nonlocal fresh
            fresh = True
            return backend.snapshot(stage)

        value = recorder.stage(key, observe)
        self._carry = (key, value) if fresh else None
        return value

    def before(
        self,
        backend: ReleaseRollingBackend,
        recorder: EvidenceRecorder,
        stage: str,
        classification: dict[str, Any],
    ) -> dict[str, Any]:
        carry, self._carry = self._carry, None
        index = STAGES.index(stage)
        predecessor = f"{STAGES[index - 1]}_after" if index else None

        def observe() -> dict[str, Any]:
            if carry is not None and carry[0] == predecessor:
                return {
                    **copy.deepcopy(carry[1]),
                    "next_deploy": copy.deepcopy(classification),
                    "reused_from": carry[0],
                }
            return backend.snapshot(stage)

        return recorder.stage(f"{stage}_before", observe)


def _stage_noop(
    backend: ReleaseRollingBackend,
    recorder: EvidenceRecorder,
    chain: _SnapshotChain,
) -> None:
    noop_diff = recorder.stage("noop_classification", lambda: backend.classify("noop"))
    _assert_kind("noop", noop_diff)
    # The first observation of the run: nothing precedes it, so it is never
    # reused from an earlier snapshot.
    noop_before = recorder.stage("noop_before", lambda: backend.snapshot("noop"))
    noop_apply = recorder.stage(
        "noop_apply",
        lambda: backend.deploy("noop", diff=noop_diff),
    )
    _require(noop_apply["phase"] == "complete", "noop phase", noop_apply)
    noop_after = chain.after(backend, recorder, "noop")
    _assert_noop(noop_before, noop_after)


def _stage_control_plane(
    backend: ReleaseRollingBackend,
    recorder: EvidenceRecorder,
    chain: _SnapshotChain,
) -> None:
    control_diff = recorder.stage(
        "control_plane_classification",
        lambda: backend.classify("control_plane"),
    )
    _assert_kind("control_plane", control_diff)
    control_before = chain.before(backend, recorder, "control_plane", control_diff)
    control_failed = recorder.stage(
        "control_plane_injected_failure_and_rollback",
        lambda: backend.deploy(
            "control_plane",
            diff=control_diff,
            fault_phase="cpu-finalized",
            auto_rollback=True,
        ),
    )
    _require(
        control_failed["phase"] == "rolled-back", "CPU rollback phase", control_failed
    )
    _assert_rollback_scope("control_plane", control_failed)
    _assert_rollback_rto("control_plane", control_failed)
    control_rolled_back = recorder.stage(
        "control_plane_rollback_snapshot",
        lambda: backend.snapshot("control_plane"),
    )
    _assert_rollback_snapshot(control_before, control_rolled_back)
    control_apply = recorder.stage(
        "control_plane_apply",
        lambda: backend.deploy("control_plane", diff=control_diff),
    )
    _require(control_apply["phase"] == "complete", "CPU apply phase", control_apply)
    control_after = chain.after(backend, recorder, "control_plane")
    _assert_control_plane_only(control_before, control_after)
    _assert_next_noop(backend, recorder, "control_plane")


def _stage_executor(
    backend: ReleaseRollingBackend,
    recorder: EvidenceRecorder,
    chain: _SnapshotChain,
) -> None:
    executor_diff = recorder.stage(
        "executor_classification",
        lambda: backend.classify("executor"),
    )
    _assert_kind("executor", executor_diff)
    executor_before = chain.before(backend, recorder, "executor", executor_diff)
    executor_rolled_back = recorder.stage(
        "executor_injected_failure_and_rollback",
        lambda: backend.deploy(
            "executor",
            diff=executor_diff,
            fault_phase="data-converged",
            auto_rollback=True,
        ),
    )
    _require(
        executor_rolled_back["phase"] == "rolled-back",
        "executor rollback phase",
        executor_rolled_back,
    )
    _assert_rollback_scope("executor", executor_rolled_back)
    _assert_rollback_rto("executor", executor_rolled_back)
    _assert_executor_pins(executor_rolled_back, phase="rolled-back")
    executor_rollback_snapshot = recorder.stage(
        "executor_rollback_snapshot",
        lambda: backend.snapshot("executor"),
    )
    _assert_rollback_snapshot(executor_before, executor_rollback_snapshot)
    executor_failed = recorder.stage(
        "executor_interrupted_failure",
        lambda: backend.deploy(
            "executor",
            diff=executor_diff,
            fault_phase="cpu-staged",
            auto_rollback=False,
        ),
    )
    _require(executor_failed["phase"] == "failed", "interrupted phase", executor_failed)
    _require(
        executor_failed["injected_failure"] == "cpu-staged",
        "interrupted at the wrong phase",
        executor_failed,
    )
    _assert_executor_pins(executor_failed, phase="staged")
    # The interruption leaves the CPU Deployments on the candidate image and
    # the GPU plane on the previous one; the engine's previous-release capture
    # refuses that mixed fleet by design, and this snapshot only needs the CPU
    # generations to prove the resume does not roll them a second time.
    executor_interrupted = recorder.stage(
        "executor_interrupted_snapshot",
        lambda: backend.snapshot("executor", live=False),
    )
    executor_resumed = recorder.stage(
        "executor_resumed",
        lambda: backend.deploy(
            "executor",
            diff=executor_diff,
            resume=True,
            auto_rollback=False,
        ),
    )
    _require(executor_resumed["phase"] == "complete", "resume phase", executor_resumed)
    _assert_resume_rto("executor", executor_resumed)
    _assert_executor_pins(executor_resumed, phase="finalized")
    executor_after = chain.after(backend, recorder, "executor")
    # The interrupted attempt already staged the CPU side; the resume finishes
    # the data plane and then finalizes the pins, which is the one further CPU
    # rollout every release makes (``cpu-finalized``). What it must not do is
    # repeat the staged rollout: each CPU Deployment advances by at most one
    # generation between the interruption and the end of the stage (live
    # 2026-09-11: 505 -> 506, 563 -> 564, 455 -> 456).
    repeated = {
        name: (executor_interrupted["cpu_generations"].get(name), generation)
        for name, generation in executor_after["cpu_generations"].items()
        if generation - int(executor_interrupted["cpu_generations"].get(name) or 0) > 1
    }
    _require(
        not repeated,
        "resume repeated the staged CPU rollout (more than the finalize roll)",
        repeated,
    )
    _assert_data_plane_changed(executor_before, executor_after, scenario="executor")
    _assert_next_noop(backend, recorder, "executor")


def _stage_agent(
    backend: ReleaseRollingBackend,
    recorder: EvidenceRecorder,
    chain: _SnapshotChain,
) -> None:
    agent_diff = recorder.stage(
        "agent_classification",
        lambda: backend.classify("agent"),
    )
    _assert_kind("agent", agent_diff)
    agent_before = chain.before(backend, recorder, "agent", agent_diff)
    agent_failed = recorder.stage(
        "agent_injected_failure_and_rollback",
        lambda: backend.deploy(
            "agent",
            diff=agent_diff,
            fault_phase="data-converged",
            auto_rollback=True,
        ),
    )
    _require(
        agent_failed["phase"] == "rolled-back", "agent rollback phase", agent_failed
    )
    _assert_rollback_scope("agent", agent_failed)
    _assert_rollback_rto("agent", agent_failed)
    agent_rolled_back = recorder.stage(
        "agent_rollback_snapshot",
        lambda: backend.snapshot("agent"),
    )
    _assert_rollback_snapshot(agent_before, agent_rolled_back)
    agent_apply = recorder.stage(
        "agent_apply_after_rollback",
        lambda: backend.deploy("agent", diff=agent_diff),
    )
    _require(agent_apply["phase"] == "complete", "agent apply phase", agent_apply)
    agent_after = chain.after(backend, recorder, "agent")
    _assert_data_plane_changed(agent_before, agent_after, scenario="agent")
    _assert_next_noop(backend, recorder, "agent")


def _stage_full(
    backend: ReleaseRollingBackend,
    recorder: EvidenceRecorder,
    chain: _SnapshotChain,
) -> None:
    full_diff = recorder.stage(
        "full_classification",
        lambda: backend.classify("full"),
    )
    _assert_kind("full", full_diff)
    full_before = chain.before(backend, recorder, "full", full_diff)
    full_failed = recorder.stage(
        "full_injected_failure_and_rollback",
        lambda: backend.deploy(
            "full",
            diff=full_diff,
            fault_phase="data-converged",
            auto_rollback=True,
        ),
    )
    _require(full_failed["phase"] == "rolled-back", "FULL rollback phase", full_failed)
    _require(
        full_failed["injected_failure"] == "data-converged",
        "FULL interrupted at the wrong phase",
        full_failed,
    )
    _assert_rollback_scope("full", full_failed)
    _assert_rollback_rto("full", full_failed)
    full_rolled_back = recorder.stage(
        "full_rollback_snapshot",
        lambda: backend.snapshot("full"),
    )
    _assert_rollback_snapshot(full_before, full_rolled_back)
    full_apply = recorder.stage(
        "full_apply_after_rollback",
        lambda: backend.deploy("full", diff=full_diff),
    )
    _require(full_apply["phase"] == "complete", "FULL apply phase", full_apply)
    full_after = chain.after(backend, recorder, "full")
    _require(
        full_after["live"] != full_before["live"],
        "FULL changed nothing",
        (full_before, full_after),
    )
    _assert_next_noop(backend, recorder, "full")


STAGE_RUNNERS: dict[
    str,
    Callable[[ReleaseRollingBackend, EvidenceRecorder, _SnapshotChain], None],
] = {
    "noop": _stage_noop,
    "control_plane": _stage_control_plane,
    "executor": _stage_executor,
    "agent": _stage_agent,
    "full": _stage_full,
}


def run_release_rolling(
    backend: ReleaseRollingBackend,
    recorder: EvidenceRecorder,
    *,
    start_stage: str = "noop",
) -> dict[str, Any]:
    """Run the six-stage contract, optionally resuming at a later stage.

    Every observation goes through the recorder, so a rerun over the same
    evidence file replays recorded stages instead of touching the site again.
    ``start_stage`` skips the stages before it entirely; it is only accepted
    when the evidence already holds each skipped stage's terminal record, and
    the resume is itself recorded so the report shows where the run restarted.
    A driver defect found mid-run then costs one stage, not the ~2 h whole.

    Snapshots are chained: a stage's freshly observed ``*_after`` becomes the
    next stage's ``*_before`` (see ``_SnapshotChain``). The chain starts empty
    here, so whatever stage a run begins at observes its own ``*_before``.
    """

    if start_stage not in STAGES:
        raise ValueError(f"unknown BOOT-020 stage: {start_stage}")
    start = STAGES.index(start_stage)
    chain = _SnapshotChain()
    try:
        if start:
            missing = [
                stage
                for stage in STAGES[:start]
                if STAGE_TERMINAL_KEY[stage] not in recorder.document["stages"]
            ]
            if missing:
                raise RuntimeError(
                    "cannot resume BOOT-020 at "
                    f"{start_stage}: earlier stages have no recorded result: "
                    + ", ".join(missing)
                )
            recorder.stage(
                f"resumed_at_{start_stage}",
                lambda: {
                    "resumed_from_stage": start_stage,
                    "skipped_stages": list(STAGES[:start]),
                    "reason": (
                        "stages before this one carry their recorded results "
                        "from the earlier attempt in this evidence file"
                    ),
                },
            )
        for stage in STAGES[start:]:
            STAGE_RUNNERS[stage](backend, recorder, chain)
            recorder.stage(
                _passed_marker(stage),
                lambda stage=stage: {"stage": stage, "passed_at": utc_now()},
            )
        return recorder.complete()
    except BaseException as exc:
        recorder.fail(exc)
        raise


def _discard_incomplete(recorder: EvidenceRecorder, index: int) -> list[str]:
    """Drop the resume stage's partial records and every later stage's records.

    The resume stage failed a driver assertion, so its recorded sub-stages were
    taken against a live state the convergence is about to move; replaying them
    would compare stale observations. Earlier, passed stages are left intact so
    they still replay. Resume markers are dropped too so the resume re-records
    where it restarted.
    """

    targets = STAGES[index:]
    doomed = [
        key
        for key in recorder.document["stages"]
        if key.startswith("resumed_at_")
        or any(_stage_owns(key, stage) for stage in targets)
    ]
    return recorder.drop_stages(doomed)


def _converge_to_precondition(
    backend: ReleaseRollingBackend,
    recorder: EvidenceRecorder,
    index: int,
) -> dict[str, Any]:
    """Bring the live release back to the resume stage's precondition.

    The resume stage begins by classifying its config against the state the
    previous stage left, and asserting the classification. A failed stage's
    auto-rollback left the live state at that precondition already, but pinned
    at ``rolled-back``; classifying and deploying the previous stage's config is
    a NOOP that clears the phase to ``complete`` without touching artifacts. In
    the rarer case where the stage failed after applying its own config, the
    same deploy really rolls the artifacts back to the precondition. Either way
    the resume then runs exactly as a first pass would. This folds the
    hand-written convergence step into the driver.
    """

    scenario = STAGES[index - 1]
    diff = backend.classify(scenario)
    applied = backend.deploy(scenario, diff=diff)
    if applied.get("phase") != "complete":
        raise RuntimeError(
            f"BOOT-020 could not converge the live state to the {scenario} "
            f"precondition before resuming: {applied}"
        )
    history = list(recorder.document.get("convergence", []))
    history.append(
        {
            "converged_to": scenario,
            "classification": diff,
            "phase": applied.get("phase"),
            "at": utc_now(),
        }
    )
    recorder.note("convergence", history)
    return applied


def resume_release_rolling(
    backend: ReleaseRollingBackend,
    recorder: EvidenceRecorder,
) -> dict[str, Any]:
    """Continue a BOOT-020 run from the first stage its evidence lacks a pass.

    Unlike ``--start-stage``, the resume point is read from the evidence, the
    failed stage's partial records are discarded so it runs again from its
    classification, and the live release state is converged back to that stage's
    precondition first. A driver defect then costs the failed stage, not the
    whole ~2 h contract, and no verdict changes: every assertion runs as it
    would on a first pass.
    """

    index, stage = resume_target(recorder.document)
    if stage is None:
        return recorder.complete()
    _discard_incomplete(recorder, index)
    if index:
        _converge_to_precondition(backend, recorder, index)
    return run_release_rolling(backend, recorder, start_stage=stage)


def _read_snapshot(release: Any):
    """The engine's read snapshot when the release offers one, else a no-op.

    Inside it every read-only ``kubectl get`` is served once from one
    observation, and a nested ``read_snapshot`` -- ``_capture_previous`` opens
    its own -- joins the outer one instead of discarding its warm cache.
    """

    factory = getattr(release, "_read_snapshot", None)
    return factory() if callable(factory) else nullcontext()


def deployment_generations(
    release: Any,
    args: list[str],
    names: tuple[str, ...],
) -> dict[str, int]:
    """``metadata.generation`` of each named Deployment from ONE list read.

    ``args`` is the kube-context prefix (``release._cpu()`` or
    ``release._gpu(target)``); the argv is built exactly as the engine's
    ``prime_deployment_snapshot`` builds it, so inside a read snapshot the list
    ``_capture_previous`` already primed answers this without another kubectl
    call. A Deployment absent from the list reads as generation 0 -- the value
    the former per-name read fell back to when kubectl returned nothing (a
    per-name read of a missing object would have raised instead; the compared
    dicts keep the same key set either way, so no verdict moves).
    """

    listing = release._get_json(
        args + ["-n", release.config.namespace, "get", "deployment"]
    )
    found: dict[str, int] = {}
    for item in listing.get("items") or []:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata", {}) or {}
        name = metadata.get("name")
        if name in names:
            found[str(name)] = int(metadata.get("generation", 0))
    return {name: found.get(name, 0) for name in names}


class LiveReleaseRollingBackend:
    def __init__(self, configs: dict[str, Path]) -> None:
        self.configs = configs
        self.rollout = rollout_regional_release
        self.config_module = regional_release_config
        self.diff_module = regional_release_diff
        self.commands = regional_admin_commands
        self.inventory = regional_deployment_inventory

    def _release(
        self,
        scenario: str,
        *,
        auto_rollback: bool | None = None,
    ) -> Any:
        config = self.config_module.ReleaseConfig.load(self.configs[scenario])
        if auto_rollback is not None:
            config = replace(config, auto_rollback=auto_rollback)
        return self.rollout.RegionalRelease(config, self.rollout.Runner())

    def classify(self, scenario: str) -> dict[str, Any]:
        release = self._release(scenario)
        state = release._load_state()
        return dict(self.diff_module.classify_release(release, state).as_dict())

    def _next_deploy(self, release: Any, state: dict[str, Any]) -> Any:
        """What ``build_release_summary(release)["next_deploy"]`` would hold.

        ``build_release_summary`` first builds the whole release status
        (metadata ConfigMap, wheel and template reads per cluster) that this
        snapshot never records, then calls ``next_deploy(release, state)`` on a
        state it re-loads; calling ``next_deploy`` on the state already loaded
        yields the identical value. The summary swallowed a failure into
        ``next_deploy_error`` and the snapshot recorded ``None`` for the field,
        so that is kept -- the field is informational, never asserted.
        """

        try:
            return self.commands.next_deploy(release, state)
        except Exception:
            return None

    def snapshot(self, scenario: str, *, live: bool = True) -> dict[str, Any]:
        """Observe the live release once.

        ``live=False`` skips the engine's previous-release capture and the
        next-deploy classification: mid-transaction -- after an injected
        interruption at ``cpu-staged`` -- the CPU and GPU planes run different
        runtime images and ``_capture_previous`` refuses the mixed fleet
        (``require_consistent_images``); the caller then wants only the phase
        and the Deployment generations.

        Everything runs inside one read snapshot: the state ConfigMap is read
        once and shared with ``_capture_previous`` (its own nested snapshot
        joins this one), ``next_deploy`` reuses the reads the capture made,
        and the generations come from the ``get deployment`` lists the capture
        already primed -- one per kube context -- instead of one ``kubectl``
        per Deployment. ``live`` is the engine's full previous-release capture,
        untouched; only how it and the generations are read changed.
        """

        release = self._release(scenario)
        with _read_snapshot(release):
            state = release._load_state()
            captured = release._capture_previous() if live else None
            next_deploy = self._next_deploy(release, state) if live else None
            cpu_generations = deployment_generations(
                release,
                release._cpu(),
                self.inventory.CPU_RUNTIME_DEPLOYMENTS,
            )
            gpu_generations = {
                target.cluster_id: deployment_generations(
                    release,
                    release._gpu(target),
                    self.inventory.DEPLOYMENTS,
                )
                for target in release.config.clusters
            }
        return {
            "phase": state.get("phase"),
            "release_id": state.get("release_id"),
            "live": captured,
            "cpu_generations": cpu_generations,
            "gpu_generations": gpu_generations,
            "next_deploy": next_deploy,
        }

    def _pin_window(self, release: Any, state: dict[str, Any]) -> dict[str, Any]:
        """The executor pin window as the metadata ConfigMap holds it now.

        ``previous_required`` comes from the release state's ``previous``
        snapshot (what the transaction captured before mutating anything) and
        ``candidate`` from the config being rolled out, so a stage can say
        which of staged / finalized / rolled-back the live window is in.
        """

        metadata = release._config_map_data("gpu-fault-release-metadata")
        previous = state.get("previous") or {}
        previous_metadata = previous.get("metadata") or {}
        compatible = [
            item.strip()
            for item in str(metadata.get(COMPATIBLE_EXECUTOR_PINS) or "").split(",")
            if item.strip()
        ]
        return {
            "required": metadata.get(REQUIRED_EXECUTOR_PIN),
            "compatible": compatible,
            "candidate": release.executor_wheel_sha,
            "previous_required": previous_metadata.get(REQUIRED_EXECUTOR_PIN),
        }

    def _diff(self, value: dict[str, Any]) -> Any:
        return self.diff_module.ReleaseDiff(
            kind=self.diff_module.ReleaseChangeKind(value["kind"]),
            changed=frozenset(value.get("changed") or []),
        )

    def deploy(
        self,
        scenario: str,
        *,
        diff: dict[str, Any],
        fault_phase: str | None = None,
        resume: bool = False,
        auto_rollback: bool | None = None,
    ) -> dict[str, Any]:
        release = self._release(scenario, auto_rollback=auto_rollback)
        active_diff = self._diff(diff)
        original_save = release._save_state
        injected = False
        started = time.monotonic()

        def save_with_injection(phase: str, **updates: Any) -> None:
            nonlocal injected
            original_save(phase, **updates)
            if fault_phase == phase and not injected:
                injected = True
                raise InjectedAcceptanceFailure(
                    f"BOOT-020 injected failure after {phase}"
                )

        if fault_phase is not None:
            release._save_state = save_with_injection
        try:
            if active_diff.kind.value == "NOOP":
                release.noop(active_diff)
            else:
                release.upgrade(resume=resume, diff=active_diff)
        except InjectedAcceptanceFailure:
            state = release._load_state()
            return {
                "phase": state.get("phase"),
                "release_id": state.get("release_id"),
                "injected_failure": fault_phase,
                "release_diff": diff,
                "rollback_plan": state.get("rollback_plan"),
                "rollback_timing": state.get("rollback_timing"),
                "pins": self._pin_window(release, state),
                "operation_duration_seconds": time.monotonic() - started,
            }
        state = release._load_state()
        return {
            "phase": state.get("phase"),
            "release_id": state.get("release_id"),
            "injected_failure": None,
            "release_diff": diff,
            "rollback_plan": state.get("rollback_plan"),
            "rollback_timing": state.get("rollback_timing"),
            "pins": self._pin_window(release, state),
            "operation_duration_seconds": time.monotonic() - started,
        }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--noop-config", required=True, type=Path)
    value.add_argument("--control-plane-config", required=True, type=Path)
    value.add_argument("--data-plane-config", required=True, type=Path)
    value.add_argument("--agent-config", required=True, type=Path)
    value.add_argument("--full-config", required=True, type=Path)
    value.add_argument("--run-dir", required=True, type=Path)
    value.add_argument("--gpu-kubeconfig", type=Path)
    value.add_argument("--execute", action="store_true")
    value.add_argument("--confirm")
    value.add_argument(
        "--start-stage",
        choices=STAGES,
        default="noop",
        help=(
            "resume at this stage; the run directory must already hold the "
            "earlier stages' recorded results"
        ),
    )
    value.add_argument(
        "--resume",
        action="store_true",
        help=(
            "resume automatically: read the evidence, converge the live state "
            "back to the first unfinished stage's precondition, and restart "
            "only that stage. Chooses the stage for you; do not pass "
            "--start-stage with it"
        ),
    )
    return value


def main() -> int:
    arguments = parser().parse_args()
    if arguments.resume and arguments.start_stage != "noop":
        raise SystemExit(
            "--resume chooses the stage from the evidence; do not also pass "
            "--start-stage"
        )
    configs = {
        "noop": arguments.noop_config.resolve(),
        "control_plane": arguments.control_plane_config.resolve(),
        "executor": arguments.data_plane_config.resolve(),
        "agent": arguments.agent_config.resolve(),
        "full": arguments.full_config.resolve(),
    }
    if len(set(configs.values())) != 5:
        raise SystemExit(
            "the five release scenarios require five distinct config files"
        )
    for path in configs.values():
        if not path.is_file():
            raise SystemExit(f"release config does not exist: {path}")
    plan = {
        "case_id": CASE_ID,
        "run_dir": str(arguments.run_dir),
        "gpu_kubeconfig": (
            str(arguments.gpu_kubeconfig.resolve())
            if arguments.gpu_kubeconfig is not None
            else os.getenv("KUBECONFIG")
        ),
        "configs": {name: str(path) for name, path in configs.items()},
        "start_stage": "auto (--resume)" if arguments.resume else arguments.start_stage,
        "resume": arguments.resume,
        "stages": [
            "verify NOOP makes no live artifact change",
            "assert CPU-only rollback T_safe and T_full",
            "assert Executor-only rollback plus interrupted resume RTO",
            "assert Agent-only rollback T_safe and T_full",
            "assert FULL rollback T_safe and T_full",
            "apply FULL successfully and verify the next classification is NOOP",
        ],
    }
    if not arguments.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if arguments.confirm != CONFIRMATION:
        raise SystemExit(f"--execute requires --confirm {CONFIRMATION}")
    gpu_kubeconfig = configure_gpu_kubeconfig(arguments.gpu_kubeconfig)
    arguments.run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    inputs = {
        "configs": {name: str(path) for name, path in configs.items()},
        "gpu_kubeconfig": str(gpu_kubeconfig),
    }
    recorder = EvidenceRecorder(
        arguments.run_dir / f"{CASE_ID}.json",
        case_id=CASE_ID,
        inputs=inputs,
    )
    backend = LiveReleaseRollingBackend(configs)
    if arguments.resume:
        result = resume_release_rolling(backend, recorder)
    else:
        result = run_release_rolling(
            backend,
            recorder,
            start_stage=arguments.start_stage,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
