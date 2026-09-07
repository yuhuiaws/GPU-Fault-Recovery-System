from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import yaml

from gpu_fault.config_cli import validate, validate_operation_allowlists, validate_role

ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "deploy/control-plane/regional/generated"


def _spool_worker_deployment(replicas: int) -> dict:
    return {
        "metadata": {"name": "gpu-fault-telemetry-spool-worker"},
        "spec": {
            "replicas": replicas,
            "template": {"spec": {"containers": [{"args": ["gpu-fault-spool"]}]}},
        },
    }


def test_spool_worker_lint_reads_the_switch_the_way_the_worker_does() -> None:
    """The worker accepts ``1``; the manifest lint used to demand ``true``."""

    values: dict[str, str | None] = {
        "GPU_FAULT_SERVICE_ROLE": "spool-worker",
        "GPU_FAULT_TELEMETRY_SPOOL": "1",
    }

    assert validate_role(_spool_worker_deployment(1), values) == []

    values["GPU_FAULT_TELEMETRY_SPOOL"] = "off"
    errors = validate_role(_spool_worker_deployment(1), values)
    assert len(errors) == 1
    assert "GPU_FAULT_TELEMETRY_SPOOL" in errors[0]


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


def _ingress_with_env(tmp_path: Path, name: str, value: str) -> list[Path]:
    deployment = copy.deepcopy(
        yaml.safe_load((GENERATED / "gpu-fault-api-ha-ingress.yaml").read_text())
    )
    deployment["spec"]["template"]["spec"]["containers"][0]["env"].append(
        {"name": name, "value": value}
    )
    path = tmp_path / "deployment.yaml"
    path.write_text(yaml.safe_dump(deployment))
    return [*sorted(GENERATED.glob("*-config-*.yaml")), path]


def test_manifest_env_names_must_exist_in_the_runtime_inventory(tmp_path: Path) -> None:
    """A Pod refuses to start on an unknown GPU_FAULT_* name; fail at render time."""

    report = validate(_ingress_with_env(tmp_path, "GPU_FAULT_NO_SUCH_SWITCH", "true"))

    assert not report["valid"]
    assert any(
        "GPU_FAULT_NO_SUCH_SWITCH" in item and "runtime inventory" in item
        for item in report["errors"]
    ), report["errors"]


def test_manifest_boolean_values_must_be_tokens_and_never_blank(tmp_path: Path) -> None:
    """Blank means "unset" at runtime, so a manifest that ships a blank boolean
    is saying nothing while looking like it says something."""

    blank = validate(_ingress_with_env(tmp_path, "GPU_FAULT_ALLOW_EMAIL", ""))
    assert not blank["valid"]
    assert any(
        "GPU_FAULT_ALLOW_EMAIL" in item and "blank" in item for item in blank["errors"]
    ), blank["errors"]

    bogus = validate(_ingress_with_env(tmp_path, "GPU_FAULT_ALLOW_EMAIL", "maybe"))
    assert not bogus["valid"]
    assert any("GPU_FAULT_ALLOW_EMAIL" in item for item in bogus["errors"]), bogus[
        "errors"
    ]
