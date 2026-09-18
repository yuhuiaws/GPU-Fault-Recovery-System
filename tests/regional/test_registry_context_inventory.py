from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tests.regional.test_installed_resource_registry import COLLECT_MODULE, MODULE


def inputs(tmp_path: Path, contexts: list[str]) -> tuple[Path, Path]:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "cpu_kubeconfig": "fixture-cpu",
                "clusters": [{"context": context} for context in contexts],
            }
        )
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "cpu": {"resources": [], "database_pod_preference": []},
                "gpu": {"resources": []},
            }
        )
    )
    return config, inventory


def test_collector_keeps_each_cluster_own_resource_set(tmp_path, monkeypatch) -> None:
    config, inventory = inputs(tmp_path, ["gpu-a", "gpu-b"])

    class Kubectl:
        def __init__(self, *, kubeconfig, context, reuse_exec_credential):
            self.context = context or kubeconfig

        def run(self, arguments, *, check, timeout_seconds):
            assert arguments == [
                "get",
                "roles,rolebindings",
                "--all-namespaces",
                "-o",
                "json",
                "--request-timeout=15s",
            ], "unexpected read outside the dynamic RBAC inventory"
            assert check is False and timeout_seconds == 20, (
                "RBAC inspection must keep the bounded raw-result contract"
            )
            return subprocess.CompletedProcess(arguments, 0, '{"items":[]}', "")

        def close(self) -> None:
            pass

    def synchronize(kubectl, *, plane, namespace, **_kwargs):
        return {
            "plane": plane,
            "namespace": namespace,
            "resources": (
                [
                    {
                        "kind": "deployment",
                        "scope": "namespaced",
                        "namespace": namespace,
                        "name": "gpu-fault-" + kubectl.context,
                    }
                ]
                if plane == "gpu"
                else []
            ),
        }

    monkeypatch.setattr(COLLECT_MODULE, "Kubectl", Kubectl)
    monkeypatch.setattr(COLLECT_MODULE, "synchronize", synchronize)
    monkeypatch.setattr(COLLECT_MODULE, "discover_unregistered", lambda *_a, **_k: [])
    result = COLLECT_MODULE.collect(config, inventory_path=inventory, apply=False)
    per_context = result["gpu"]["by_context"]
    assert set(per_context) == {"gpu-a", "gpu-b"}
    for context in per_context:
        assert [item["name"] for item in per_context[context]["resources"]] == [
            "gpu-fault-" + context
        ]
    assert len(result["gpu"]["resources"]) == 2


def test_duplicated_contexts_are_refused_before_any_registry_write(
    tmp_path, monkeypatch
) -> None:
    config, inventory = inputs(tmp_path, ["gpu-a", "gpu-a"])
    monkeypatch.setattr(
        COLLECT_MODULE, "Kubectl", lambda **_kw: pytest.fail("reached a cluster")
    )
    with pytest.raises(MODULE.RegistryError, match="contexts"):
        COLLECT_MODULE.collect(config, inventory_path=inventory, apply=True)
