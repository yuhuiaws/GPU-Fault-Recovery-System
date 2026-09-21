from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from scripts.e2e.regional import run_boot020_release_rolling as rolling


@dataclass(frozen=True)
class EngineConfig:
    namespace: str = "unit-namespace"
    auto_rollback: bool = True
    clusters: tuple = (SimpleNamespace(cluster_id="cluster-a", context="unit-context"),)


class ReleaseEngine:
    """Recording release-engine boundary, never a process or resource client."""

    def __init__(self, model, config, runner):
        self.model = model
        self.config = config
        self.runner = runner
        self.executor_wheel_sha = "b" * 64
        self.state = {
            "phase": "complete",
            "release_id": "unit-release",
            "previous": {"metadata": {rolling.REQUIRED_EXECUTOR_PIN: "a" * 64}},
        }
        self.metadata = {
            rolling.REQUIRED_EXECUTOR_PIN: "a" * 64,
            rolling.COMPATIBLE_EXECUTOR_PINS: "",
        }
        self.calls = []
        self.saved = []
        self.reads = []

    def _load_state(self):
        self.calls.append(("load-state",))
        return copy.deepcopy(self.state)

    def _save_state(self, phase, **updates):
        self.saved.append((phase, copy.deepcopy(updates)))
        self.state.update(phase=phase, **updates)
        if phase == "cpu-staged":
            self.metadata[rolling.COMPATIBLE_EXECUTOR_PINS] = "b" * 64
        elif phase == "complete":
            self.metadata[rolling.REQUIRED_EXECUTOR_PIN] = "b" * 64
            self.metadata[rolling.COMPATIBLE_EXECUTOR_PINS] = ""
        elif phase == "rolled-back":
            self.metadata[rolling.REQUIRED_EXECUTOR_PIN] = "a" * 64
            self.metadata[rolling.COMPATIBLE_EXECUTOR_PINS] = ""

    def _config_map_data(self, name):
        self.reads.append(("configmap", name))
        return copy.deepcopy(self.metadata)

    def _capture_previous(self):
        self.calls.append(("capture",))
        return {"clusters": {"cluster-a": {"wheel": "observed-wheel"}}}

    def _cpu(self):
        return ["unit-cpu"]

    def _gpu(self, target):
        return ["unit-gpu", target.cluster_id]

    def _get_json(self, args):
        self.reads.append(("deployments", tuple(args)))
        if self.model.listing_override is not None:
            return copy.deepcopy(self.model.listing_override)
        names = (
            rolling.regional_deployment_inventory.CPU_RUNTIME_DEPLOYMENTS
            if args[0] == "unit-cpu"
            else (
                *rolling.regional_deployment_inventory.DEPLOYMENTS,
                rolling.regional_deployment_inventory.GPU_RECONCILER_DEPLOYMENT,
            )
        )
        return {
            "items": [
                None,
                {"metadata": {"name": "unrelated", "generation": 99}},
                *(
                    {
                        "metadata": {"name": name, "generation": index + 1},
                        # The live listing always carries replicas and a Pod
                        # template; the snapshot reads both for the rollout
                        # identity next to the generation (2026-09-20).
                        "spec": {
                            "replicas": 1,
                            "template": {"metadata": {"labels": {"app": name}}},
                        },
                    }
                    for index, name in enumerate(names)
                ),
            ]
        }

    def noop(self, diff):
        self.calls.append(("noop", diff))
        if self.model.error is not None:
            raise self.model.error
        self._save_state("complete")

    def upgrade(self, *, resume, diff):
        self.calls.append(("upgrade", resume, diff))
        try:
            for phase in ("cpu-staged", "data-converged", "cpu-finalized", "complete"):
                self._save_state(phase)
                if self.model.error is not None:
                    raise self.model.error
        except rolling.InjectedAcceptanceFailure:
            if self.config.auto_rollback:
                self._save_state(
                    "rolled-back",
                    rollback_plan={"clusters": {"cluster-a": ["executor"]}},
                    rollback_timing={"t_safe_seconds": 2, "t_full_seconds": 3},
                )
            else:
                self._save_state("failed")
            raise


class BackendModel:
    def __init__(self):
        self.config = EngineConfig()
        self.loaded = []
        self.runners = []
        self.engines = []
        self.classifications = []
        self.error = None
        self.listing_override = None
        self.seconds = 0

    def backend(self):
        backend = rolling.LiveReleaseRollingBackend(
            {name: Path(f"/tmp/unit-{name}.json") for name in rolling.STAGES}
        )
        backend.config_module = SimpleNamespace(
            ReleaseConfig=SimpleNamespace(load=self.load)
        )
        backend.rollout = SimpleNamespace(
            RegionalRelease=self.release, Runner=self.runner
        )
        backend.diff_module = SimpleNamespace(
            ReleaseDiff=rolling.regional_release_diff.ReleaseDiff,
            ReleaseChangeKind=rolling.regional_release_diff.ReleaseChangeKind,
            classify_release=self.classify,
            build_execution_plan=rolling.regional_release_diff.build_execution_plan,
            control_plane_role_targets=rolling.regional_release_diff.control_plane_role_targets,
        )
        backend.commands = SimpleNamespace(
            next_deploy=lambda release, state: {"kind": "NOOP", "changed": []}
        )
        return backend

    def load(self, path):
        self.loaded.append(path)
        return self.config

    def runner(self):
        runner = object()
        self.runners.append(runner)
        return runner

    def release(self, config, runner):
        engine = ReleaseEngine(self, config, runner)
        self.engines.append(engine)
        return engine

    def classify(self, release, state):
        self.classifications.append((release, state))
        return rolling.regional_release_diff.ReleaseDiff(
            kind=rolling.regional_release_diff.ReleaseChangeKind.FULL,
            changed=frozenset({"runtime_profile", "executor"}),
        )

    def monotonic(self):
        self.seconds += 1
        return float(self.seconds)


class RollingEntry:
    def __init__(self, root, monkeypatch):
        self.root = root
        self.calls = []
        self.predecessor = {"valid": True, "verdict": "PASS"}
        self.configs = {}
        self.argv = [
            "boot020",
            "--run-dir",
            str(root / "run"),
            "--admin-state-dir",
            str(root),
            "--admin-reference",
            "CHG-ENTRY",
        ]
        monkeypatch.setattr(rolling, "validate_admin_target", lambda *_a: None)
        monkeypatch.setattr(
            rolling, "public_config_roundtrip", lambda *_a, **_kw: {"restored": True}
        )
        for flag, stage in (
            ("--noop-config", "noop"),
            ("--control-plane-config", "control_plane"),
            ("--data-plane-config", "executor"),
            ("--agent-config", "agent"),
            ("--full-config", "full"),
        ):
            path = root / f"{stage}.json"
            path.write_text("{}", encoding="utf-8")
            if stage == "control_plane":
                from gpu_fault.admin.config import default_admin_config

                path.write_text(
                    json.dumps(
                        {"admin_config": {"config": default_admin_config().as_dict()}}
                    )
                )
            self.configs[stage] = path
            self.argv.extend([flag, str(path)])
        self.kubeconfig = root / "unit-kubeconfig"
        self.kubeconfig.write_text("unit fake transport only", encoding="utf-8")
        self.argv.extend(
            [
                "--gpu-kubeconfig",
                str(self.kubeconfig),
                "--execute",
                "--confirm",
                rolling.CONFIRMATION,
            ]
        )
        self.path = root / "run" / "cases" / rolling.CASE_ID / f"{rolling.CASE_ID}.json"
        monkeypatch.setattr(rolling, "install_site_profile", lambda: None)
        monkeypatch.setattr(rolling, "authorize_execution", self.authorize)
        monkeypatch.setattr(
            rolling, "predecessor_path", lambda *args: ("unit-before", root / "before")
        )
        monkeypatch.setattr(
            rolling, "predecessor_evidence", lambda *args: dict(self.predecessor)
        )
        monkeypatch.setenv("KUBECONFIG", str(self.kubeconfig))

    def authorize(self, arguments, **kwargs):
        self.calls.append((arguments, kwargs))
