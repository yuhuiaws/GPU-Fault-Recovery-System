from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME_PATTERN = re.compile(r"\bGPU_FAULT_[A-Z0-9_]+\b")
ADMIN_SPEC = ROOT / "scripts" / "admin-env-reference.yaml"
ADMIN_REFERENCE = ROOT / "docs" / "管理员环境变量参考.md"
REGIONAL_GENERATED = ROOT / "deploy" / "control-plane" / "regional" / "generated"
DEPLOY_ENV_SOURCES = (
    ROOT / "deploy" / "node" / "install-gpu-fault-collector.sh",
    ROOT / "deploy" / "node" / "verify-gpu-fault-collector.sh",
    *(ROOT / "deploy" / "systemd").glob("*"),
)


def test_environment_reference_matches_source() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/generate-env-reference.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_deployment_environment_is_known_to_python_processes() -> None:
    inventory = json.loads(
        (ROOT / "src" / "gpu_fault" / "data" / "env-inventory.json").read_text(
            encoding="utf-8"
        )
    )
    known = set(inventory["variables"])
    deployed = {
        name
        for path in DEPLOY_ENV_SOURCES
        if path.is_file()
        for name in NAME_PATTERN.findall(path.read_text(encoding="utf-8"))
    }
    assert deployed <= known, (
        "deployment code uses GPU_FAULT_* variables absent from "
        f"the runtime inventory: {sorted(deployed - known)}"
    )


def test_environment_inventory_contains_only_concrete_names() -> None:
    inventory = json.loads(
        (ROOT / "src" / "gpu_fault" / "data" / "env-inventory.json").read_text(
            encoding="utf-8"
        )
    )
    variables = inventory["variables"]

    assert all(NAME_PATTERN.fullmatch(name) for name in variables), (
        "runtime inventory contains a malformed GPU_FAULT_* name"
    )
    assert not [name for name in variables if name.endswith("_")]
    assert not set(variables).intersection(inventory["dynamic_prefixes"]), (
        "dynamic environment prefixes must not be listed as concrete variables"
    )


def test_regional_generated_environment_is_in_runtime_inventory() -> None:
    import yaml

    inventory = json.loads(
        (ROOT / "src" / "gpu_fault" / "data" / "env-inventory.json").read_text(
            encoding="utf-8"
        )
    )
    known = set(inventory["variables"])
    deployed: set[str] = set()
    for path in REGIONAL_GENERATED.glob("*.yaml"):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            continue
        deployed.update(
            name
            for name in (document.get("data") or {})
            if name.startswith("GPU_FAULT_")
        )
        pod_spec = None
        kind = document.get("kind")
        if kind in {"Deployment", "DaemonSet", "Job"}:
            pod_spec = document["spec"]["template"]["spec"]
        elif kind == "CronJob":
            pod_spec = document["spec"]["jobTemplate"]["spec"]["template"]["spec"]
        if pod_spec is None:
            continue
        for container in [
            *(pod_spec.get("initContainers") or []),
            *(pod_spec.get("containers") or []),
        ]:
            deployed.update(
                item["name"]
                for item in container.get("env", [])
                if item.get("name", "").startswith("GPU_FAULT_")
            )

    assert deployed <= known, (
        "regional generated manifests use GPU_FAULT_* variables absent "
        f"from the runtime inventory: {sorted(deployed - known)}"
    )


def test_admin_environment_reference_is_curated_from_inventory() -> None:
    import yaml

    inventory = json.loads(
        (ROOT / "src" / "gpu_fault" / "data" / "env-inventory.json").read_text(
            encoding="utf-8"
        )
    )
    spec = yaml.safe_load(ADMIN_SPEC.read_text(encoding="utf-8"))
    selected = [
        item["name"]
        for category in spec["categories"]
        for item in category["variables"]
    ]
    reference = ADMIN_REFERENCE.read_text(encoding="utf-8")

    assert len(selected) == len(set(selected))
    assert set(selected) <= set(inventory["variables"])
    assert f"精选 **{len(selected)}** 个" in reference
    assert "区域清单值" in reference
    assert "应用默认/要求" in reference
    for name in selected:
        assert f"`{name}`" in reference
