"""Actual CPU renderer and offline shell apply preserve SES routing by role."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault_release import regional_release_rendering as rendering
from gpu_fault_release import regional_release_state as state
from gpu_fault_release import rollout
from tests.deploy import test_cpu_template_apply as cpu_apply
from tests.regional.test_release_ses_configuration import (
    SES_ENV,
    release_config,
    rollback_environment,
)

ROOT = Path(__file__).resolve().parents[2]
RENDERER = ROOT / "deploy/control-plane/tools/render_control_plane_role_split.py"
BASE = ROOT / "deploy/control-plane/base/control-plane-deployment.yaml"
CONTAINERS = {
    "gpu-fault-api-ha": "api",
    "gpu-fault-control-worker": "control-worker",
    "gpu-fault-telemetry-spool-worker": "telemetry-spool-worker",
}
CAPTURE_FILE_APPLIES = """
if "apply" in args and "-f" in args:
    path = args[args.index("-f") + 1]
    if path != "-":
        with open(path) as source:
            documents = [item for item in yaml.safe_load_all(source) if item]
        with open(os.environ["APPLIES"], "a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            for document in documents:
                handle.write(json.dumps(document) + "\\n")
"""


def render_roles(
    environment: dict[str, str], *, out_dir: Path | None = None
) -> list[dict[str, Any]]:
    arguments = [sys.executable, str(RENDERER)]
    if out_dir is not None:
        out_dir.mkdir()
        arguments.extend(("--out-dir", str(out_dir)))
    completed = subprocess.run(
        arguments,
        input=BASE.read_text(encoding="utf-8"),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    if out_dir is None:
        return list(json.loads(completed.stdout)["items"])
    return [
        yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted(out_dir.glob("gpu-fault-*.yaml"))
    ]


def effective_sets(
    documents: list[dict[str, Any]],
    *,
    existing_config_maps: dict[str, dict[str, str]] | None = None,
) -> dict[str, str]:
    config_maps = {
        **(existing_config_maps or {}),
        **{
            document["metadata"]["name"]: document["data"]
            for document in documents
            if document["kind"] == "ConfigMap" and "data" in document
        },
    }
    result: dict[str, str] = {}
    for document in documents:
        if document["kind"] != "Deployment":
            continue
        name = document["metadata"]["name"]
        containers = document["spec"]["template"]["spec"]["containers"]
        assert [item["name"] for item in containers] == [CONTAINERS[name]]
        values: dict[str, str] = {}
        container = containers[0]
        for source in container.get("envFrom", []):
            if reference := source.get("configMapRef"):
                values.update(config_maps.get(reference["name"], {}))
        for entry in container.get("env", []):
            if "value" in entry:
                values[entry["name"]] = entry["value"]
            elif reference := entry.get("valueFrom", {}).get("configMapKeyRef"):
                value = config_maps.get(reference["name"], {}).get(reference["key"])
                if value is not None:
                    values[entry["name"]] = value
        result[name] = values[SES_ENV]
    return result


def apply_with_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    *,
    targets: str = "spool,worker,ingress",
) -> tuple[list[dict[str, Any]], list[list[str]]]:
    original_run = subprocess.run
    monkeypatch.setattr(
        cpu_apply, "FAKE_API", cpu_apply.FAKE_API + CAPTURE_FILE_APPLIES
    )

    def run(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert arguments == ["bash", str(cpu_apply.SCRIPT)]
        kwargs["env"] = {**kwargs["env"], **environment}
        return original_run(arguments, **kwargs)

    monkeypatch.setattr(cpu_apply.subprocess, "run", run)
    return cpu_apply.apply_roles(tmp_path, targets=targets, force=False)


@pytest.mark.parametrize("value", [None, "alerts-set"])
def test_plan_and_actual_cpu_apply_agree_for_all_real_container_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    config = release_config(tmp_path, value)
    release = rollout.RegionalRelease(config, rollout.Runner(dry_run=True))
    environment = rendering.build_cpu_apply_environment(release, finalize=False)
    documents = render_roles(environment)
    planned = [
        yaml.safe_load(
            rendering.render_cpu_manifest_text(
                release, document["metadata"]["name"], yaml.safe_dump(document)
            )
        )
        for document in documents
    ]
    applied, _commands = apply_with_environment(
        tmp_path, monkeypatch, {SES_ENV: environment[SES_ENV]}
    )

    expected = {name: value or "" for name in CONTAINERS}
    assert effective_sets(planned) == expected
    assert effective_sets(applied) == expected


def test_role_scoped_apply_does_not_rewrite_sibling_notification_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    applied, commands = apply_with_environment(
        tmp_path, monkeypatch, {SES_ENV: "worker-set"}, targets="worker"
    )
    assert effective_sets(applied) == {"gpu-fault-control-worker": "worker-set"}
    notification_maps = [
        item["metadata"]["name"]
        for item in applied
        if item["kind"] == "ConfigMap" and SES_ENV in item.get("data", {})
    ]
    assert notification_maps == ["gpu-fault-control-worker-config-notification"]
    assert not any("restart" in command or "set" in command for command in commands), (
        "role-scoped apply must not issue extra restart or set commands"
    )


@pytest.mark.parametrize("direct_env", [False, True])
def test_captured_rollback_environments_override_candidate_ses_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, direct_env: bool
) -> None:
    release = rollout.RegionalRelease(
        release_config(tmp_path, "candidate-set"), rollout.Runner(dry_run=True)
    )
    forward_environment = rendering.build_cpu_apply_environment(release, finalize=False)
    old_documents = render_roles(forward_environment)
    previous_maps = {
        document["metadata"]["name"]: copy.deepcopy(document["data"])
        for document in old_documents
        if document["kind"] == "ConfigMap"
    }
    deployments = {
        document["metadata"]["name"]: document
        for document in old_documents
        if document["kind"] == "Deployment"
    }
    expected = {}
    for name, deployment in deployments.items():
        value = f"previous-{CONTAINERS[name]}"
        previous_maps[f"{name}-config-notification"][SES_ENV] = value
        expected[name] = value
        if direct_env:
            value = f"direct-{CONTAINERS[name]}"
            deployment["spec"]["template"]["spec"]["containers"][0]["env"].append(
                {"name": SES_ENV, "value": value}
            )
            expected[name] = value
    snapshot = state.cpu_role_container_env(release, deployments)
    snapshot_file = tmp_path / "previous-env.json"
    snapshot_file.write_text(json.dumps(snapshot), encoding="utf-8")
    snapshot_file.chmod(0o600)
    environment = rollback_environment(
        release.config,
        preserve_role_config_maps=True,
        previous_container_env_file=str(snapshot_file),
    )
    generated = tmp_path / "generated"
    rendered = render_roles(environment, out_dir=generated)
    for document in rendered:
        if document["kind"] == "Deployment":
            name = document["metadata"]["name"]
            container = document["spec"]["template"]["spec"]["containers"][0]
            assert container["env"] == snapshot[name][CONTAINERS[name]]["env"]
            assert container["envFrom"] == snapshot[name][CONTAINERS[name]]["envFrom"]
    applied, _commands = apply_with_environment(
        tmp_path,
        monkeypatch,
        {
            SES_ENV: environment[SES_ENV],
            "GPU_FAULT_PRESERVE_ROLE_CONFIG_MAPS": "true",
            "GPU_FAULT_ROLE_SPLIT_CONTAINER_ENV_FILE": str(snapshot_file),
            "GPU_FAULT_ROLE_SPLIT_GENERATED_DIR": str(generated),
        },
    )

    assert not any(
        item["kind"] == "ConfigMap" and "-config-" in item["metadata"]["name"]
        for item in applied
    ), "rollback must preserve captured role ConfigMaps, not apply candidate maps"
    assert effective_sets(applied, existing_config_maps=previous_maps) == expected


def test_cpu_apply_rejects_bad_configuration_set_before_any_command(
    tmp_path: Path,
) -> None:
    kubectl = tmp_path / "kubectl"
    kubectl.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    kubectl.chmod(0o755)
    completed = subprocess.run(
        ["bash", str(cpu_apply.SCRIPT)],
        env={
            "HOME": str(tmp_path),
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "KUBECONFIG": os.devnull,
            "GPU_FAULT_AWS_REGION": "us-east-1",
            "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION": "profile-test",
            "GPU_FAULT_NOTIFICATION_CONFIG_SHA256": "a" * 64,
            SES_ENV: "invalid/set",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert completed.returncode == 2
    assert "GPU_FAULT_SES_CONFIGURATION_SET must be" in completed.stderr
    assert "invalid/set" not in completed.stderr
