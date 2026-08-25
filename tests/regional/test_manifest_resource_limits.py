from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "deploy"


def _manifests() -> list[Path]:
    paths = []
    for path in sorted(DEPLOY.rglob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        if "!GetAtt" in text or "AWSTemplateFormatVersion" in text:
            continue
        paths.append(path)
    return paths


def _pod_specs(node: Any, found: list[dict[str, Any]]) -> None:
    if isinstance(node, dict):
        if isinstance(node.get("containers"), list):
            found.append(node)
        for value in node.values():
            _pod_specs(value, found)
    elif isinstance(node, list):
        for value in node:
            _pod_specs(value, found)


def _all_pod_specs() -> list[tuple[Path, dict[str, Any]]]:
    specs = []
    for path in _manifests():
        documents = yaml.safe_load_all(path.read_text(encoding="utf-8"))
        for document in documents:
            found: list[dict[str, Any]] = []
            _pod_specs(document, found)
            specs.extend((path, spec) for spec in found)
    return specs


def test_every_in_cluster_container_bounds_cpu_and_memory() -> None:
    specs = _all_pod_specs()

    assert specs
    for path, spec in specs:
        containers = [*spec.get("initContainers", []), *spec.get("containers", [])]
        for container in containers:
            where = f"{path.relative_to(ROOT)}:{container.get('name')}"
            resources = container.get("resources")
            assert resources, where
            for section in ("requests", "limits"):
                values = resources.get(section)
                assert values, f"{where} {section}"
                assert "cpu" in values, f"{where} {section} cpu"
                assert "memory" in values, f"{where} {section} memory"


def test_in_cluster_collectors_mount_writable_state_directory() -> None:
    checked: set[str] = set()
    for path, spec in _all_pod_specs():
        volumes = {volume["name"] for volume in spec.get("volumes", [])}
        for container in spec.get("containers", []):
            entrypoint = json.dumps(container.get("command", "")) + json.dumps(
                container.get("args", "")
            )
            if "gpu-fault-collector" not in entrypoint:
                continue
            where = f"{path.relative_to(ROOT)}:{container['name']}"
            mounts = [
                mount
                for mount in container.get("volumeMounts", [])
                if mount.get("mountPath") == "/var/lib/gpu-fault"
            ]
            assert len(mounts) == 1, where
            mount = mounts[0]
            assert not mount.get("readOnly"), where
            assert mount["name"] in volumes, where
            checked.add(where)

    assert checked == {
        "deploy/dataplane/gpu-metrics-collector.yaml:collector",
        ("deploy/dataplane/kubernetes-node-resource-collector.yaml:collector"),
        ("deploy/dataplane/optional/hma-cloudwatch-consumer.yaml:consumer"),
        "deploy/dataplane/optional/hma-watcher.yaml:watcher",
    }
