from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from gpu_fault_release import regional_release_config as config
from gpu_fault_release import regional_release_diff as diff
from scripts.e2e.regional import run_boot023_release_history as runner
from tests.regional._cov95_common_live import LiveModel
from tests.regional._cov95_release_engine import EngineRelease

DIGEST = "a" * 64


def history_entry(*, phase="complete", timestamp="2026-09-12T01:00:00Z"):
    return {
        "timestamp": timestamp,
        "release_id": "unit-release",
        "phase": phase,
        "release_lifecycle": "released",
        "state_sha256": DIGEST,
        "plan_sha256": None,
        "operator": "fixture-reviewer",
        "command": "gpu-fault-admin deploy --state-dir fixture",
    }


def delivery_manifest():
    delivery = {
        "schema_version": 1,
        "runtime_prebuilt": True,
        "components": {
            name: {"sha256": DIGEST}
            for name in (
                "collector",
                "cpu",
                "dcgm",
                "endpoint",
                "executor",
                "node",
                "observability",
                "schema",
                "watcher",
            )
        },
        "images": {
            name: {"reference": f"registry/{name}@sha256:{DIGEST}"}
            for name in ("runtime", "node_installer", "dcgm_exporter", "adot")
        },
        "node_template_inputs": {"sha256": "b" * 64},
    }
    delivery["sha256"] = config.canonical_sha256(delivery)
    return {
        "deployable": True,
        "delivery": delivery,
        "components": {"node_bundle": {"template_sha256": "b" * 64}},
    }


class HistoryRegion(LiveModel):
    def __init__(self, root, model):
        super().__init__(root)
        self.model = model

    def ready_pods(self, _plane, app):
        return [{"name": app + "-pod"}]

    def kubectl(self, plane, *args, **kwargs):
        if args[:3] == ("get", "configmap", runner.verdicts.HISTORY_CONFIG_MAP):
            if self.model.read_failure:
                raise runner.RegionalFixtureError("fixture history read failed")
            if self.model.absent:
                return ""
            return json.dumps(
                {
                    "data": {
                        runner.verdicts.HISTORY_KEY: "\n".join(
                            json.dumps(row) for row in self.model.history
                        )
                    }
                }
            )
        if args[:3] == ("get", "configmap", "-l"):
            return json.dumps(
                {
                    "items": [
                        {
                            "metadata": {
                                "name": "previous-0",
                                "creationTimestamp": "2026-09-11T01:00:00Z",
                                "annotations": {
                                    runner.verdicts.PREVIOUS_SNAPSHOT_DIGEST_ANNOTATION: DIGEST
                                },
                            }
                        }
                    ]
                }
            )
        if args[:2] == ("get", "pod"):
            return json.dumps(
                {
                    "spec": {
                        "containers": [
                            {
                                "name": "sidecar",
                                "ports": [{"name": "metrics", "containerPort": 9000}],
                            },
                            {
                                "name": "api",
                                "ports": (
                                    []
                                    if self.model.missing_port
                                    else [{"name": "http", "containerPort": 8080}]
                                ),
                            },
                        ]
                    }
                }
            )
        if args[0] == "exec":
            self.model.probe_calls.append((args, kwargs))
            return json.dumps(
                {
                    "secret_config_sha256": DIGEST,
                    "durable_config_sha256": DIGEST,
                    "head_generation": 7,
                    "head_content_sha256": DIGEST,
                    "healthz": {
                        "status": 200,
                        "payload": {
                            "regional_registry": {
                                "ready": True,
                                "secret_drift": False,
                                "secret_config_sha256": DIGEST,
                            }
                        },
                    },
                    "livez": {"status": 200},
                }
            )
        return super().kubectl(plane, *args, **kwargs)


class HistoryModel:
    def __init__(self, root, monkeypatch):
        self.history = [history_entry(phase="cpu-staged")]
        self.actions = []
        self.probe_calls = []
        self.focused_calls = []
        self.absent = False
        self.read_failure = False
        self.missing_port = False
        self.write_mirror = True
        self.test_code = 0
        self.kind = diff.ReleaseChangeKind.NOOP
        self.regional = HistoryRegion(root, self)
        engine_root = root / "engine"
        engine_root.mkdir()
        self.engine = EngineRelease(engine_root)
        self.engine.state = {
            "release_id": "unit-release",
            "phase": "complete",
            "release_lifecycle": "released",
        }
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(delivery_manifest()))
        noop_config = root / "noop.json"
        noop_config.write_text(json.dumps({"release": {"manifest": manifest.name}}))
        self.settings = runner.Settings(
            self.regional.settings, noop_config, root / "predecessor.json"
        )
        target = self.settings.regional
        bound_config = SimpleNamespace(
            cpu_kubeconfig=str(target.cpu_kubeconfig),
            namespace=target.namespace,
            aws_region=target.region,
            clusters=[
                SimpleNamespace(
                    cluster_id=target.cluster_id, context=target.gpu_context
                )
            ],
        )
        self.modules = runner.deploy_modules()
        self.modules["regional_release_config"] = SimpleNamespace(
            ReleaseError=config.ReleaseError,
            parse_delivery_identity=config.parse_delivery_identity,
            ReleaseConfig=SimpleNamespace(load=lambda _path: bound_config),
        )
        self.modules["regional_release_diff"] = SimpleNamespace(
            ReleaseChangeKind=diff.ReleaseChangeKind,
            ReleaseDiff=diff.ReleaseDiff,
            classify_release=lambda _release, _state: diff.ReleaseDiff(
                self.kind, frozenset()
            ),
        )
        self.modules["rollout_regional_release"] = SimpleNamespace(
            Runner=lambda **kwargs: SimpleNamespace(**kwargs),
            RegionalRelease=lambda _config, _transport: self.engine,
        )
        model = self

        class FixtureFactory:
            def __new__(cls, _settings):
                return model.regional

            @staticmethod
            def run(command, **kwargs):
                model.focused_calls.append((command, kwargs))
                return subprocess.CompletedProcess(
                    command, model.test_code, "fixture test output", ""
                )

        monkeypatch.setattr(runner, "RegionalLiveFixture", FixtureFactory)
        monkeypatch.setattr(runner, "deploy_modules", lambda: self.modules)
        monkeypatch.setattr(
            runner, "predecessor_evidence", lambda *_args: {"valid": True}
        )
        monkeypatch.setattr(self.engine, "noop", self.noop)
        monkeypatch.setenv("KUBECONFIG", str(target.gpu_kubeconfig))

    def noop(self, release_diff, *, allow_prerequisite_repair):
        assert allow_prerequisite_repair is False, (
            "history acceptance must never repair prerequisites"
        )
        assert release_diff.kind is diff.ReleaseChangeKind.NOOP, (
            "only a NOOP may reach release execution"
        )
        self.actions.append("noop")
        self.history.append(history_entry())
        if self.write_mirror:
            path = (
                Path(os.environ[runner.HISTORY_DIR_ENV])
                / runner.verdicts.HISTORY_MIRROR_FILE
            )
            path.write_text("\n".join(json.dumps(row) for row in self.history))

    def plan(self, run_dir):
        case_dir = run_dir / "cases" / runner.CASE_ID
        case_dir.mkdir(parents=True, exist_ok=True)
        preflight = runner.read_only_preflight(self.settings, case_dir)
        (case_dir / "plan.json").write_text(
            json.dumps({"details": runner.plan_details(self.settings, preflight)})
        )
        return copy.deepcopy(preflight)
