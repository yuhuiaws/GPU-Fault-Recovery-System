from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
MODULE = lazy_script_module(
    "sync_installed_resource_registry",
    ROOT / "deploy" / "control-plane" / "tools" / "sync_installed_resource_registry.py",
)
COLLECT_MODULE = lazy_script_module(
    "collect_installed_resource_registry",
    ROOT
    / "deploy"
    / "control-plane"
    / "tools"
    / "collect_installed_resource_registry.py",
)


class FakeKubectl:
    def __init__(
        self, *, present: set[tuple[str, str]], existing: dict | None = None
    ) -> None:
        self.present = present
        self.existing = existing
        self.applied: dict | None = None

    def run(
        self, arguments: list[str], *, input_text: str | None = None, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        if arguments[0:4] == ["-n", "gpu-fault-system", "get", "configmap"]:
            if self.existing is None:
                return subprocess.CompletedProcess(
                    arguments, 1, "", "Error from server (NotFound)"
                )
            value = {"data": {"inventory.json": json.dumps(self.existing)}}
            return subprocess.CompletedProcess(arguments, 0, json.dumps(value), "")
        if arguments[0] == "apply":
            self.applied = json.loads(input_text or "{}")
            return subprocess.CompletedProcess(arguments, 0, "", "")
        offset = 2 if arguments[0] == "-n" else 0
        kind = arguments[offset + 1]
        name = arguments[offset + 2]
        exists = (kind, name) in self.present
        return subprocess.CompletedProcess(
            arguments,
            0 if exists else 1,
            f"{kind}/{name}\n" if exists else "",
            "" if exists else "Error from server (NotFound)",
        )


class DiscoveryKubectl:
    def run(
        self, arguments: list[str], *, input_text: str | None = None, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        del input_text, check
        if "-A" in arguments:
            items = [
                {
                    "kind": "Deployment",
                    "metadata": {
                        "namespace": "gf-regional-old",
                        "name": "gpu-fault-legacy-executor",
                    },
                },
                {
                    "kind": "Deployment",
                    "metadata": {
                        "namespace": "training",
                        "name": "gpu-fault-customer-workload",
                    },
                },
            ]
        elif "clusterrole" in arguments[1]:
            items = [
                {"kind": "ClusterRole", "metadata": {"name": "gpu-fault-legacy-role"}}
            ]
        else:
            items = [{"kind": "Namespace", "metadata": {"name": "gf-regional-old"}}]
        return subprocess.CompletedProcess(
            arguments, 0, json.dumps({"items": items}), ""
        )


def write_inventory(tmp_path: Path) -> Path:
    path = tmp_path / "inventory.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cpu": {
                    "resources": [
                        {
                            "kind": "deployment",
                            "name": "gpu-fault-api",
                            "scope": "namespaced",
                            "phase": "ingress",
                            "order": 10,
                            "clean": "delete",
                        },
                        {
                            "kind": "daemonset",
                            "name": "gpu-fault-absent",
                            "scope": "namespaced",
                            "phase": "auxiliary",
                            "order": 20,
                            "clean": "delete",
                        },
                    ]
                },
                "gpu": {"resources": []},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_first_sync_writes_only_resources_that_exist(tmp_path: Path) -> None:
    kubectl = FakeKubectl(present={("deployment", "gpu-fault-api")})

    document = MODULE.synchronize(
        kubectl,
        plane="cpu",
        namespace="gpu-fault-system",
        inventory_path=write_inventory(tmp_path),
        release_id="release-a",
        apply=True,
    )

    assert [(item["kind"], item["name"]) for item in document["resources"]] == [
        ("deployment", "gpu-fault-api")
    ]
    assert kubectl.applied is not None
    stored = json.loads(kubectl.applied["data"]["inventory.json"])
    assert stored["release_id"] == "release-a"
    assert stored["resources"] == document["resources"]


def test_sync_retains_live_legacy_resource_until_deleted(tmp_path: Path) -> None:
    legacy = {
        "kind": "deployment",
        "name": "gpu-fault-legacy",
        "scope": "namespaced",
        "phase": "producer",
        "order": 90,
        "clean": "delete",
    }
    kubectl = FakeKubectl(
        present={("deployment", "gpu-fault-api"), ("deployment", "gpu-fault-legacy")},
        existing={"schema_version": 1, "plane": "cpu", "resources": [legacy]},
    )

    document = MODULE.synchronize(
        kubectl,
        plane="cpu",
        namespace="gpu-fault-system",
        inventory_path=write_inventory(tmp_path),
        release_id="release-b",
        apply=False,
    )

    assert {(item["kind"], item["name"]) for item in document["resources"]} == {
        ("deployment", "gpu-fault-api"),
        ("deployment", "gpu-fault-legacy"),
    }


def test_deployment_entrypoints_refresh_runtime_registry() -> None:
    control = (
        ROOT
        / "deploy"
        / "control-plane"
        / "tools"
        / "apply-control-plane-role-split.sh"
    ).read_text(encoding="utf-8")
    control_verify = (
        ROOT
        / "deploy"
        / "control-plane"
        / "tools"
        / "verify-control-plane-role-split.sh"
    ).read_text(encoding="utf-8")
    gpu = (ROOT / "deploy/node/deploy-node-installer-reconciler.sh").read_text(
        encoding="utf-8"
    )
    bundle = (ROOT / "deploy/node/build-node-installer-bundle.sh").read_text(
        encoding="utf-8"
    )

    assert "verify-control-plane-role-split.sh" in control
    for text in (control_verify, gpu, bundle):
        assert "sync_installed_resource_registry.py" in text

    hma = (ROOT / "deploy/hyperpod/deploy-cloudwatch-hma.sh").read_text(
        encoding="utf-8"
    )
    assert "gpu-fault-hma-cloudwatch-consumer" in hma
    assert "sync_installed_resource_registry.py" in hma


def test_cleanup_discovery_fails_closed_on_legacy_resources() -> None:
    found = COLLECT_MODULE.discover_unregistered(
        DiscoveryKubectl(),
        plane="gpu",
        context="gpu-context",
        namespace="gpu-fault-system",
        registered=set(),
    )

    assert {(item["kind"], item["name"]) for item in found} == {
        ("deployment", "gpu-fault-legacy-executor"),
        ("clusterrole", "gpu-fault-legacy-role"),
        ("namespace", "gf-regional-old"),
    }
