from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import yaml

from gpu_fault.config_cli import validate, validate_operation_allowlists

ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "deploy/control-plane/regional/generated"


def test_generated_control_plane_configuration_is_valid() -> None:
    report = validate(sorted(GENERATED.glob("*.yaml")))

    assert report["valid"], report["errors"]
    assert report["deployments"] == 3
    assert report["config_maps"] == 18


def test_config_cli_rejects_literal_secret(tmp_path: Path) -> None:
    deployment = yaml.safe_load(
        (GENERATED / "gpu-fault-api-ha-ingress.yaml").read_text()
    )
    deployment = copy.deepcopy(deployment)
    deployment["spec"]["template"]["spec"]["containers"][0]["env"].append(
        {"name": "GPU_FAULT_EXECUTION_TOKEN", "value": "not-secret"}
    )
    path = tmp_path / "deployment.yaml"
    path.write_text(yaml.safe_dump(deployment))

    report = validate([*sorted(GENERATED.glob("*-config-*.yaml")), path])

    assert not report["valid"]
    assert any("must use valueFrom" in item for item in report["errors"])


def test_config_cli_command_validates_default_manifests() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "gpu_fault.config_cli", "validate", str(GENERATED)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_operation_allowlists_must_match_node_action_capability() -> None:
    errors = validate_operation_allowlists(
        {
            "control": {"GPU_FAULT_ALLOWED_OPERATIONS": ("RESET_GPU,REMEDIATE_DRIVER")},
            "node": {"GPU_FAULT_NODE_ALLOWED_OPERATIONS": "RESET_GPU"},
        }
    )
    assert any("REMEDIATE_DRIVER" in error for error in errors)
