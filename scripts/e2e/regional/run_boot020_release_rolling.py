from __future__ import annotations

import argparse
from dataclasses import replace
import importlib
import json
from pathlib import Path
import sys
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
    "data_plane": "DATA_PLANE_COMPATIBLE",
    "full": "FULL",
}


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


def run_release_rolling(
    backend: ReleaseRollingBackend,
    recorder: EvidenceRecorder,
) -> dict[str, Any]:
    try:
        noop_diff = recorder.stage(
            "noop_classification", lambda: backend.classify("noop")
        )
        _assert_kind("noop", noop_diff)
        noop_before = recorder.stage("noop_before", lambda: backend.snapshot("noop"))
        noop_apply = recorder.stage(
            "noop_apply",
            lambda: backend.deploy("noop", diff=noop_diff),
        )
        assert noop_apply["phase"] == "complete", noop_apply
        noop_after = recorder.stage("noop_after", lambda: backend.snapshot("noop"))
        _assert_noop(noop_before, noop_after)

        control_diff = recorder.stage(
            "control_plane_classification",
            lambda: backend.classify("control_plane"),
        )
        _assert_kind("control_plane", control_diff)
        control_before = recorder.stage(
            "control_plane_before",
            lambda: backend.snapshot("control_plane"),
        )
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
        control_next = backend.classify("control_plane")
        assert control_next["kind"] == "NOOP", control_next

        data_diff = recorder.stage(
            "data_plane_classification",
            lambda: backend.classify("data_plane"),
        )
        _assert_kind("data_plane", data_diff)
        data_before = recorder.stage(
            "data_plane_before",
            lambda: backend.snapshot("data_plane"),
        )
        data_failed = recorder.stage(
            "data_plane_injected_failure",
            lambda: backend.deploy(
                "data_plane",
                diff=data_diff,
                fault_phase="cpu-staged",
                auto_rollback=False,
            ),
        )
        assert data_failed["phase"] == "failed", data_failed
        assert data_failed["injected_failure"] == "cpu-staged", data_failed
        data_resumed = recorder.stage(
            "data_plane_resumed",
            lambda: backend.deploy(
                "data_plane",
                diff=data_diff,
                resume=True,
                auto_rollback=False,
            ),
        )
        assert data_resumed["phase"] == "complete", data_resumed
        data_after = recorder.stage(
            "data_plane_after",
            lambda: backend.snapshot("data_plane"),
        )
        _assert_data_plane_changed(data_before, data_after)
        data_next = backend.classify("data_plane")
        assert data_next["kind"] == "NOOP", data_next

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
        full_rolled_back = recorder.stage(
            "full_rollback_snapshot",
            lambda: backend.snapshot("full"),
        )
        assert full_rolled_back["live"] == full_before["live"], (
            full_before,
            full_rolled_back,
        )
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
        full_next = backend.classify("full")
        assert full_next["kind"] == "NOOP", full_next
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
            }
        state = release._load_state()
        return {
            "phase": state.get("phase"),
            "release_id": state.get("release_id"),
            "injected_failure": None,
            "release_diff": diff,
        }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--noop-config", required=True, type=Path)
    value.add_argument("--control-plane-config", required=True, type=Path)
    value.add_argument("--data-plane-config", required=True, type=Path)
    value.add_argument("--full-config", required=True, type=Path)
    value.add_argument("--run-dir", required=True, type=Path)
    value.add_argument("--execute", action="store_true")
    value.add_argument("--confirm")
    return value


def main() -> int:
    arguments = parser().parse_args()
    configs = {
        "noop": arguments.noop_config.resolve(),
        "control_plane": arguments.control_plane_config.resolve(),
        "data_plane": arguments.data_plane_config.resolve(),
        "full": arguments.full_config.resolve(),
    }
    if len(set(configs.values())) != 4:
        raise SystemExit(
            "the four release scenarios require four distinct config files"
        )
    for path in configs.values():
        if not path.is_file():
            raise SystemExit(f"release config does not exist: {path}")
    plan = {
        "case_id": CASE_ID,
        "run_dir": str(arguments.run_dir),
        "configs": {name: str(path) for name, path in configs.items()},
        "stages": [
            "verify NOOP makes no live artifact change",
            "verify CONTROL_PLANE_ONLY preserves every GPU data plane",
            "inject DATA_PLANE_COMPATIBLE failure after cpu-staged and resume",
            "inject FULL failure after data-converged and verify automatic rollback",
            "apply FULL successfully and verify the next classification is NOOP",
        ],
    }
    if not arguments.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if arguments.confirm != CONFIRMATION:
        raise SystemExit(f"--execute requires --confirm {CONFIRMATION}")
    arguments.run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    inputs = {"configs": {name: str(path) for name, path in configs.items()}}
    recorder = EvidenceRecorder(
        arguments.run_dir / f"{CASE_ID}.json",
        case_id=CASE_ID,
        inputs=inputs,
    )
    result = run_release_rolling(
        LiveReleaseRollingBackend(configs),
        recorder,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
