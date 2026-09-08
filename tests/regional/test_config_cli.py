from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from gpu_fault.config_cli import (
    CURRENT_CONTRACT_VALIDATORS,
    validate,
    validate_operation_allowlists,
    validate_role,
)
from gpu_fault.container_env_snapshot import pod_container_env

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


# --- container env snapshot mode (verbatim rollback) -------------------------
#
# On rollback the renderer copies the previous release's container env into the
# rendered Deployments, so judging the render by the current release's
# validators would fail a correct rollback. With --container-env-snapshot the
# lint compares the render against the snapshot and skips those validators.

ROLE_MANIFESTS = {
    "gpu-fault-api-ha": "gpu-fault-api-ha-ingress.yaml",
    "gpu-fault-control-worker": "gpu-fault-control-worker.yaml",
    "gpu-fault-telemetry-spool-worker": "gpu-fault-telemetry-spool-worker.yaml",
}


def _role_deployments() -> dict[str, dict]:
    return {
        name: copy.deepcopy(yaml.safe_load((GENERATED / file_name).read_text()))
        for name, file_name in ROLE_MANIFESTS.items()
    }


def _container(deployment: dict) -> dict:
    (container,) = deployment["spec"]["template"]["spec"]["containers"]
    return container


def _snapshot_of(deployments: dict[str, dict]) -> dict:
    return {
        name: pod_container_env(deployment) for name, deployment in deployments.items()
    }


def _write_render(tmp_path: Path, deployments: dict[str, dict]) -> list[Path]:
    paths = []
    for name, deployment in deployments.items():
        path = tmp_path / ROLE_MANIFESTS[name]
        path.write_text(yaml.safe_dump(deployment))
        paths.append(path)
    return paths


def _write_snapshot(tmp_path: Path, snapshot: object) -> str:
    path = tmp_path / "previous-container-env.json"
    path.write_text(json.dumps(snapshot))
    return str(path)


def _run_cli(arguments: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "gpu_fault.config_cli", "validate", *arguments],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )


def test_snapshot_mode_accepts_the_previous_env_the_current_registry_rejects(
    tmp_path: Path,
) -> None:
    """The defect: a name the current code does not know is expected on rollback."""

    deployments = _role_deployments()
    _container(deployments["gpu-fault-api-ha"])["env"].append(
        {"name": "GPU_FAULT_ONLY_THE_OLD_IMAGE_KNOWS", "value": "1"}
    )
    paths = _write_render(tmp_path, deployments)
    snapshot_file = _write_snapshot(tmp_path, _snapshot_of(deployments))

    current = validate(paths)
    assert not current["valid"], "default mode must still hold the current contract"
    assert any(
        "GPU_FAULT_ONLY_THE_OLD_IMAGE_KNOWS" in item and "runtime inventory" in item
        for item in current["errors"]
    ), current["errors"]
    assert "skipped" not in current

    rollback = validate(paths, container_env_snapshot=snapshot_file)
    assert rollback["valid"], rollback["errors"]
    assert rollback["deployments"] == 3
    assert rollback["skipped"] == list(CURRENT_CONTRACT_VALIDATORS)
    assert rollback["container_env_snapshot"] == snapshot_file


def test_snapshot_mode_ignores_env_order_but_not_envfrom_order(tmp_path: Path) -> None:
    deployments = _role_deployments()
    snapshot_file = _write_snapshot(tmp_path, _snapshot_of(deployments))
    container = _container(deployments["gpu-fault-control-worker"])
    container["env"].reverse()

    report = validate(
        _write_render(tmp_path, deployments), container_env_snapshot=snapshot_file
    )
    assert report["valid"], report["errors"]

    container["envFrom"].reverse()
    report = validate(
        _write_render(tmp_path, deployments), container_env_snapshot=snapshot_file
    )
    assert not report["valid"], "envFrom precedence is order-dependent"
    assert report["errors"][0].startswith(
        "gpu-fault-control-worker/control-worker envFrom[0] differs from the snapshot"
    ), report["errors"]


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        pytest.param(
            lambda container: container["env"].__setitem__(
                0, {**container["env"][0], "value": "changed"}
            ),
            "gpu-fault-api-ha/api env {name} differs from the snapshot",
            id="changed-value",
        ),
        pytest.param(
            lambda container: container["env"].pop(0),
            "gpu-fault-api-ha/api env {name} is in the snapshot but not rendered",
            id="dropped-entry",
        ),
        pytest.param(
            lambda container: container["env"].append(
                {"name": "GPU_FAULT_AAA_NEW_IN_THIS_RELEASE", "value": "1"}
            ),
            "gpu-fault-api-ha/api env GPU_FAULT_AAA_NEW_IN_THIS_RELEASE is "
            "rendered but not in the snapshot",
            id="added-entry",
        ),
    ],
)
def test_snapshot_mode_names_the_first_differing_env_entry(
    tmp_path: Path, mutate, expected: str
) -> None:
    deployments = _role_deployments()
    snapshot_file = _write_snapshot(tmp_path, _snapshot_of(deployments))
    container = _container(deployments["gpu-fault-api-ha"])
    # Sort so index 0 is the first entry in comparison order and "first
    # difference" is unambiguous; the sort itself is not a difference.
    container["env"].sort(key=lambda item: item["name"])
    first_name = container["env"][0]["name"]
    mutate(container)

    report = validate(
        _write_render(tmp_path, deployments), container_env_snapshot=snapshot_file
    )

    assert not report["valid"]
    assert report["errors"] == [expected.format(name=first_name)]


def test_snapshot_mode_requires_the_same_deployments_on_both_sides(
    tmp_path: Path,
) -> None:
    deployments = _role_deployments()
    snapshot = _snapshot_of(deployments)
    snapshot.pop("gpu-fault-control-worker")
    report = validate(
        _write_render(tmp_path, deployments),
        container_env_snapshot=_write_snapshot(tmp_path, snapshot),
    )
    assert report["errors"] == [
        "rendered Deployment gpu-fault-control-worker is not in the snapshot"
    ]

    snapshot = _snapshot_of(deployments)
    deployments.pop("gpu-fault-telemetry-spool-worker")
    report = validate(
        _write_render(tmp_path, deployments),
        container_env_snapshot=_write_snapshot(tmp_path, snapshot),
    )
    assert report["errors"] == [
        "role Deployment gpu-fault-telemetry-spool-worker is not rendered",
        "snapshot names Deployment gpu-fault-telemetry-spool-worker, "
        "which is not rendered",
    ]

    deployments = _role_deployments()
    snapshot = _snapshot_of(deployments)
    snapshot["gpu-fault-api-ha"]["sidecar"] = snapshot["gpu-fault-api-ha"].pop("api")
    report = validate(
        _write_render(tmp_path, deployments),
        container_env_snapshot=_write_snapshot(tmp_path, snapshot),
    )
    assert report["errors"] == [
        "gpu-fault-api-ha has no rendered container named sidecar",
        "gpu-fault-api-ha rendered container api is not in the snapshot",
    ]


@pytest.mark.parametrize(
    ("snapshot", "detail"),
    [
        pytest.param({}, "expected a non-empty Deployment mapping", id="empty"),
        pytest.param(
            {"gpu-fault-api-ha": {"api": {"env": []}}},
            "gpu-fault-api-ha/api must carry exactly env and envFrom lists",
            id="missing-envFrom",
        ),
        pytest.param(
            {
                "gpu-fault-api-ha": {
                    "api": {
                        "env": [{"name": "GPU_FAULT_API_TOKEN", "value": "x"}],
                        "envFrom": [],
                    }
                }
            },
            "gpu-fault-api-ha/api sensitive env GPU_FAULT_API_TOKEN carries a "
            "literal value",
            id="sensitive-literal",
        ),
    ],
)
def test_snapshot_mode_fails_closed_on_a_malformed_snapshot(
    tmp_path: Path, snapshot: object, detail: str
) -> None:
    """Same shape rules as the renderer's loader: one module, no second copy."""

    paths = _write_render(tmp_path, _role_deployments())
    snapshot_file = _write_snapshot(tmp_path, snapshot)

    with pytest.raises(ValueError, match=detail):
        validate(paths, container_env_snapshot=snapshot_file)

    result = _run_cli(
        [*(str(path) for path in paths), "--container-env-snapshot", snapshot_file]
    )
    assert result.returncode == 1
    assert f"configuration invalid: {detail}" in result.stderr


def test_snapshot_mode_fails_closed_on_an_unreadable_snapshot(tmp_path: Path) -> None:
    paths = _write_render(tmp_path, _role_deployments())

    result = _run_cli(
        [
            *(str(path) for path in paths),
            "--container-env-snapshot",
            str(tmp_path / "missing.json"),
        ]
    )

    assert result.returncode == 1
    assert "configuration invalid: cannot read" in result.stderr


def test_cli_snapshot_flag_says_which_validators_it_skipped(tmp_path: Path) -> None:
    deployments = _role_deployments()
    paths = _write_render(tmp_path, deployments)
    snapshot_file = _write_snapshot(tmp_path, _snapshot_of(deployments))

    result = _run_cli(
        [*(str(path) for path in paths), "--container-env-snapshot", snapshot_file]
    )

    assert result.returncode == 0, result.stderr
    skipped, verdict = result.stdout.strip().splitlines()
    assert skipped.startswith(
        "configuration validators skipped: " + ", ".join(CURRENT_CONTRACT_VALIDATORS)
    ), skipped
    assert "snapshot mode" in skipped
    assert verdict == "configuration valid: 3 deployment(s), 0 ConfigMap(s)"

    plain = _run_cli([*(str(path) for path in paths), "--json"])
    assert "skipped" not in json.loads(plain.stdout), "default mode is unchanged"
