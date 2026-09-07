from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[3]
REGIONAL_DEPLOY = ROOT / "deploy/control-plane/regional"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REGIONAL_DEPLOY) not in sys.path:
    sys.path.insert(0, str(REGIONAL_DEPLOY))

from scripts.e2e.regional.acceptance_runner_common import EvidenceRecorder  # noqa: E402

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


class ReleaseRollingBackend(Protocol):
    def classify(self, scenario: str) -> dict[str, Any]: ...

    def snapshot(self, scenario: str) -> dict[str, Any]: ...

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
    assert diff["kind"] == EXPECTED_KINDS[scenario], (scenario, diff)


def _assert_noop(before: dict[str, Any], after: dict[str, Any]) -> None:
    assert before["live"] == after["live"], (before, after)
    assert before["cpu_generations"] == after["cpu_generations"], (before, after)
    assert before["gpu_generations"] == after["gpu_generations"], (before, after)


def _assert_control_plane_only(
    before: dict[str, Any],
    after: dict[str, Any],
) -> None:
    assert before["live"]["clusters"] == after["live"]["clusters"], (before, after)
    assert before["gpu_generations"] == after["gpu_generations"], (before, after)
    assert before["cpu_generations"] != after["cpu_generations"], (before, after)


def _assert_data_plane_changed(
    before: dict[str, Any],
    after: dict[str, Any],
) -> None:
    before_clusters = before["live"]["clusters"]
    after_clusters = after["live"]["clusters"]
    assert set(before_clusters) == set(after_clusters), (before, after)
    changed = []
    for cluster_id in before_clusters:
        if before_clusters[cluster_id] != after_clusters[cluster_id]:
            changed.append(cluster_id)
    assert changed, "DATA_PLANE_COMPATIBLE did not change any GPU data plane"


def _assert_rollback_rto(scenario: str, result: dict[str, Any]) -> None:
    timing = result.get("rollback_timing")
    assert isinstance(timing, dict), (scenario, result)
    safe = float(timing["t_safe_seconds"])
    full = float(timing["t_full_seconds"])
    assert 0 <= safe <= full, (scenario, timing)
    assert safe <= RTO_LIMITS[scenario]["safe"], (scenario, timing)
    assert full <= RTO_LIMITS[scenario]["full"], (scenario, timing)


def _assert_rollback_scope(scenario: str, result: dict[str, Any]) -> None:
    plan = result.get("rollback_plan")
    assert isinstance(plan, dict), (scenario, result)
    clusters = plan.get("clusters") or {}
    # The rollback plan names data-plane components in lower case
    # ("collector", "executor", "reconciler", "watcher", "agent"); live
    # 2026-09-07 the executor stage failed on an upper-case comparison after a
    # correct rollback.
    components = {
        str(component).lower() for values in clusters.values() for component in values
    }
    if scenario == "control_plane":
        assert plan.get("restores_data_plane") is False, plan
    elif scenario == "executor":
        assert "executor" in components, plan
        assert "agent" not in components, plan
    elif scenario == "agent":
        assert "agent" in components, plan
        assert plan.get("needs_controller") is True, plan


def _assert_resume_rto(scenario: str, result: dict[str, Any]) -> None:
    duration = float(result["operation_duration_seconds"])
    assert duration <= RTO_LIMITS[scenario]["resume"], (scenario, result)


def _assert_rollback_snapshot(
    before: dict[str, Any],
    rolled_back: dict[str, Any],
) -> None:
    assert rolled_back["live"] == before["live"], (before, rolled_back)


STAGES = ("noop", "control_plane", "executor", "agent", "full")
# The stage key whose presence in the evidence proves a stage ran to its end.
STAGE_TERMINAL_KEY = {
    "noop": "noop_after",
    "control_plane": "control_plane_after",
    "executor": "executor_after",
    "agent": "agent_after",
    "full": "full_after",
}


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
    assert following["kind"] == "NOOP", following


def _stage_noop(backend: ReleaseRollingBackend, recorder: EvidenceRecorder) -> None:
    noop_diff = recorder.stage("noop_classification", lambda: backend.classify("noop"))
    _assert_kind("noop", noop_diff)
    noop_before = recorder.stage("noop_before", lambda: backend.snapshot("noop"))
    noop_apply = recorder.stage(
        "noop_apply",
        lambda: backend.deploy("noop", diff=noop_diff),
    )
    assert noop_apply["phase"] == "complete", noop_apply
    noop_after = recorder.stage("noop_after", lambda: backend.snapshot("noop"))
    _assert_noop(noop_before, noop_after)


def _stage_control_plane(
    backend: ReleaseRollingBackend,
    recorder: EvidenceRecorder,
) -> None:
    control_diff = recorder.stage(
        "control_plane_classification",
        lambda: backend.classify("control_plane"),
    )
    _assert_kind("control_plane", control_diff)
    control_before = recorder.stage(
        "control_plane_before",
        lambda: backend.snapshot("control_plane"),
    )
    control_failed = recorder.stage(
        "control_plane_injected_failure_and_rollback",
        lambda: backend.deploy(
            "control_plane",
            diff=control_diff,
            fault_phase="cpu-finalized",
            auto_rollback=True,
        ),
    )
    assert control_failed["phase"] == "rolled-back", control_failed
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
    assert control_apply["phase"] == "complete", control_apply
    control_after = recorder.stage(
        "control_plane_after",
        lambda: backend.snapshot("control_plane"),
    )
    _assert_control_plane_only(control_before, control_after)
    _assert_next_noop(backend, recorder, "control_plane")


def _stage_executor(
    backend: ReleaseRollingBackend,
    recorder: EvidenceRecorder,
) -> None:
    executor_diff = recorder.stage(
        "executor_classification",
        lambda: backend.classify("executor"),
    )
    _assert_kind("executor", executor_diff)
    executor_before = recorder.stage(
        "executor_before",
        lambda: backend.snapshot("executor"),
    )
    executor_rolled_back = recorder.stage(
        "executor_injected_failure_and_rollback",
        lambda: backend.deploy(
            "executor",
            diff=executor_diff,
            fault_phase="data-converged",
            auto_rollback=True,
        ),
    )
    assert executor_rolled_back["phase"] == "rolled-back", executor_rolled_back
    _assert_rollback_scope("executor", executor_rolled_back)
    _assert_rollback_rto("executor", executor_rolled_back)
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
    assert executor_failed["phase"] == "failed", executor_failed
    assert executor_failed["injected_failure"] == "cpu-staged", executor_failed
    executor_resumed = recorder.stage(
        "executor_resumed",
        lambda: backend.deploy(
            "executor",
            diff=executor_diff,
            resume=True,
            auto_rollback=False,
        ),
    )
    assert executor_resumed["phase"] == "complete", executor_resumed
    _assert_resume_rto("executor", executor_resumed)
    executor_after = recorder.stage(
        "executor_after",
        lambda: backend.snapshot("executor"),
    )
    _assert_data_plane_changed(executor_before, executor_after)
    _assert_next_noop(backend, recorder, "executor")


def _stage_agent(backend: ReleaseRollingBackend, recorder: EvidenceRecorder) -> None:
    agent_diff = recorder.stage(
        "agent_classification",
        lambda: backend.classify("agent"),
    )
    _assert_kind("agent", agent_diff)
    agent_before = recorder.stage(
        "agent_before",
        lambda: backend.snapshot("agent"),
    )
    agent_failed = recorder.stage(
        "agent_injected_failure_and_rollback",
        lambda: backend.deploy(
            "agent",
            diff=agent_diff,
            fault_phase="data-converged",
            auto_rollback=True,
        ),
    )
    assert agent_failed["phase"] == "rolled-back", agent_failed
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
    assert agent_apply["phase"] == "complete", agent_apply
    agent_after = recorder.stage(
        "agent_after",
        lambda: backend.snapshot("agent"),
    )
    _assert_data_plane_changed(agent_before, agent_after)
    _assert_next_noop(backend, recorder, "agent")


def _stage_full(backend: ReleaseRollingBackend, recorder: EvidenceRecorder) -> None:
    full_diff = recorder.stage(
        "full_classification",
        lambda: backend.classify("full"),
    )
    _assert_kind("full", full_diff)
    full_before = recorder.stage(
        "full_before",
        lambda: backend.snapshot("full"),
    )
    full_failed = recorder.stage(
        "full_injected_failure_and_rollback",
        lambda: backend.deploy(
            "full",
            diff=full_diff,
            fault_phase="data-converged",
            auto_rollback=True,
        ),
    )
    assert full_failed["phase"] == "rolled-back", full_failed
    assert full_failed["injected_failure"] == "data-converged", full_failed
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
    assert full_apply["phase"] == "complete", full_apply
    full_after = recorder.stage(
        "full_after",
        lambda: backend.snapshot("full"),
    )
    assert full_after["live"] != full_before["live"], (full_before, full_after)
    _assert_next_noop(backend, recorder, "full")


STAGE_RUNNERS = {
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
    """

    if start_stage not in STAGES:
        raise ValueError(f"unknown BOOT-020 stage: {start_stage}")
    start = STAGES.index(start_stage)
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
            STAGE_RUNNERS[stage](backend, recorder)
        return recorder.complete()
    except BaseException as exc:
        recorder.fail(exc)
        raise


class LiveReleaseRollingBackend:
    def __init__(self, configs: dict[str, Path]) -> None:
        self.configs = configs
        self.rollout = importlib.import_module("rollout_regional_release")
        self.config_module = importlib.import_module("regional_release_config")
        self.diff_module = importlib.import_module("regional_release_diff")
        self.commands = importlib.import_module("regional_admin_commands")
        self.inventory = importlib.import_module("regional_deployment_inventory")

    def _release(
        self,
        scenario: str,
        *,
        auto_rollback: bool | None = None,
    ):
        config = self.config_module.ReleaseConfig.load(self.configs[scenario])
        if auto_rollback is not None:
            config = replace(config, auto_rollback=auto_rollback)
        return self.rollout.RegionalRelease(config, self.rollout.Runner())

    def classify(self, scenario: str) -> dict[str, Any]:
        release = self._release(scenario)
        state = release._load_state()
        return self.diff_module.classify_release(release, state).as_dict()

    def snapshot(self, scenario: str) -> dict[str, Any]:
        release = self._release(scenario)
        state = release._load_state()
        live = release._capture_previous()
        summary = self.commands.build_release_summary(release)
        cpu_generations = {
            name: int(
                release._get_json(
                    release._cpu(
                        "-n",
                        release.config.namespace,
                        "get",
                        "deployment",
                        name,
                    )
                )
                .get("metadata", {})
                .get("generation", 0)
            )
            for name in self.inventory.CPU_RUNTIME_DEPLOYMENTS
        }
        gpu_generations = {}
        for target in release.config.clusters:
            gpu_generations[target.cluster_id] = {
                name: int(
                    release._get_json(
                        release._gpu(
                            target,
                            "-n",
                            release.config.namespace,
                            "get",
                            "deployment",
                            name,
                        )
                    )
                    .get("metadata", {})
                    .get("generation", 0)
                )
                for name in self.inventory.DEPLOYMENTS
            }
        return {
            "phase": state.get("phase"),
            "release_id": state.get("release_id"),
            "live": live,
            "cpu_generations": cpu_generations,
            "gpu_generations": gpu_generations,
            "next_deploy": summary.get("next_deploy"),
        }

    def _diff(self, value: dict[str, Any]):
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
    return value


def main() -> int:
    arguments = parser().parse_args()
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
        "start_stage": arguments.start_stage,
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
    result = run_release_rolling(
        LiveReleaseRollingBackend(configs),
        recorder,
        start_stage=arguments.start_stage,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
