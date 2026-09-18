from __future__ import annotations

import base64
import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts.e2e.regional import boot_acceptance_common as common
from scripts.e2e.regional import boot_acceptance_lifecycle as lifecycle
from tests.regional.test_boot_acceptance_lifecycle import (
    AGENTS,
    CPU,
    EXEC,
    MANIFEST,
    METADATA,
)
from tests.regional.test_live_command_boundaries import ready_pod


def report() -> dict[str, Any]:
    return {
        "healthy": True,
        "checks": [
            {
                "name": "runtime_component_identity",
                "status": "PASS",
                "details": {
                    "control_plane": {"deployments": {"api": {"cpu-pod": CPU}}},
                    "executor": {
                        "clusters": {"cluster-a": {"executor": {"gpu-pod": EXEC}}}
                    },
                },
            }
        ],
    }


class BootRegion:
    def __init__(self, model: BootSite, settings: Any) -> None:
        self.model = model
        self.settings = settings

    def evidence_identity(self) -> dict[str, str]:
        return {"release_id": "unit-release", "cluster_id": "cluster-a"}

    def cpu_python(self, _script: str, *_args: str) -> dict[str, Any]:
        return {"agents": copy.deepcopy(self.model.agents)}

    def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
        self.model.reads.append((plane, args, kwargs))
        if args[0] == "exec":
            return self.model.probe_output
        if args[:2] == ("get", "secret"):
            return json.dumps(
                {
                    "metadata": {"creationTimestamp": "2026-09-12T00:00:00Z"},
                    "data": {
                        "clusters.json": base64.b64encode(
                            json.dumps(
                                [{"cluster_id": value} for value in self.model.registry]
                            ).encode()
                        ).decode()
                    },
                }
            )
        if args[:2] == ("get", "configmap"):
            return json.dumps(
                {"data": copy.deepcopy(self.model.metadata)}
                if args[2] == lifecycle.RELEASE_METADATA_CONFIGMAP
                else {"data": {"state.json": json.dumps({"phase": self.model.phase})}}
            )
        if args[:2] == ("get", "deployment"):
            items = self.model.deployments[plane]
            if len(args) > 2 and not args[2].startswith("-"):
                return json.dumps(
                    {
                        "metadata": {"name": args[2]},
                        "spec": {"replicas": self.model.replicas},
                    }
                )
            return json.dumps({"items": items})
        if args[:2] == ("get", "pod"):
            return json.dumps({"items": self.model.pods})
        raise AssertionError(f"unconfigured BOOT resource read: {plane} {args[:3]}")


class BootSite:
    def __init__(self, root: Path, monkeypatch: Any) -> None:
        self.root = root
        kubeconfig = root / "unused-kubeconfig"
        kubeconfig.touch()
        self.manifest = {
            "release_id": "unit-release",
            "bundle_sha256": "b" * 64,
            **copy.deepcopy(MANIFEST),
        }
        self.manifest_path = root / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.config = {
            "namespace": "test-namespace",
            "aws_region": "test-region",
            "cpu_kubeconfig": str(kubeconfig),
            "gpu_kubeconfig": str(kubeconfig),
            "clusters": [{"cluster_id": "cluster-a", "context": "unit-context"}],
            "release": {"manifest": str(self.manifest_path)},
        }
        self.site = SimpleNamespace(release_config=self.config, environment={})
        self.metadata = copy.deepcopy(METADATA)
        self.agents = copy.deepcopy(AGENTS)
        self.registry = ["cluster-a"]
        self.phase = "complete"
        self.replicas = 2
        self.pods = [ready_pod(), ready_pod()]
        self.pods[1]["metadata"] = {"name": "other-pod", "uid": "other-uid"}
        self.probe_output = '{"ok": true}'
        self.reads: list[Any] = []
        self.commands: list[Any] = []
        self.admin_calls: list[Any] = []
        self.failure = ""
        self.deploys = 0
        self.deployments = {
            "cpu": [
                self.deployment(name)
                for name in lifecycle.inventory.CPU_RUNTIME_DEPLOYMENTS
            ],
            "gpu": [
                self.deployment(name)
                for _manifest, name in lifecycle.inventory.GPU_ROLLOUT_DEPLOYMENTS
            ],
        }
        monkeypatch.setattr(common, "load_site", lambda *_args, **_kwargs: self.site)
        monkeypatch.setattr(lifecycle, "load_site", lambda *_args, **_kwargs: self.site)
        monkeypatch.setattr(
            common, "RegionalLiveFixture", lambda settings: BootRegion(self, settings)
        )
        monkeypatch.setattr(lifecycle, "admin_command", self.admin)
        monkeypatch.setattr(lifecycle, "run", self.run)

    @staticmethod
    def deployment(name: str) -> dict[str, Any]:
        return {
            "metadata": {
                "name": name,
                "generation": 1,
                "creationTimestamp": "2026-09-12T00:00:01Z",
            }
        }

    def admin(self, *args: str, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.admin_calls.append((args, kwargs))
        if args[0] == "deploy":
            self.deploys += 1
            state = Path(args[args.index("--state-dir") + 1])
            if self.failure != "missing-site":
                (state / "site.yaml").write_text("unit site")
                (state / "site.yaml").chmod(0o600)
            if self.failure == "generation" and self.deploys == 2:
                self.deployments["cpu"][0]["metadata"]["generation"] = 2
        return subprocess.CompletedProcess(
            args,
            1 if self.failure == args[0] else 0,
            json.dumps(report()) if args[0] == "status" else "unit admin log",
            "",
        )

    def run(self, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.commands.append((args, kwargs))
        return subprocess.CompletedProcess(
            args, 1 if self.failure == "gate" else 0, "unit gate log", ""
        )
