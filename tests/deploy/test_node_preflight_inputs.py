from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

SCRIPT = (
    Path(__file__).resolve().parents[2] / "deploy/node/run-hyperpod-installer-job.sh"
)


def inputs() -> dict:
    return {
        "schema_version": 1,
        "cluster_id": "gpu-test",
        "context": "gpu-context",
        "namespace": "gpu-fault-system",
        "node_action_keys_secret": "gpu-fault-node-action-keys",
        "connection_secret": "gpu-fault-regional-connection",
        "references_validated": True,
        "nodes": {
            "node-a": {
                "metadata": {
                    "uid": "node-uid",
                    "labels": {"node.kubernetes.io/instance-type": "ml.p5.48xlarge"},
                },
                "status": {
                    "addresses": [{"type": "InternalIP", "address": "10.0.0.1"}]
                },
            }
        },
    }


def render(
    tmp_path: Path,
    document: dict,
    *,
    mode: tuple[str, ...] = ("--render-only",),
    permissions: int = 0o600,
) -> subprocess.CompletedProcess[str]:
    snapshot = tmp_path / "nodes.json"
    snapshot.write_text(json.dumps(document), encoding="utf-8")
    snapshot.chmod(permissions)
    binary = tmp_path / "kubectl"
    binary.write_text(
        '#!/bin/sh\nprintf "unexpected kubectl invocation\\n" >&2\nexit 99\n',
        encoding="utf-8",
    )
    binary.chmod(0o755)
    return subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--node",
            "node-a",
            "--node-inputs",
            str(snapshot),
            *mode,
        ],
        env={
            "PATH": str(tmp_path)
            + os.pathsep
            + os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(tmp_path),
            "GPU_FAULT_CLUSTER_ID": "gpu-test",
            "GPU_FAULT_KUBECTL_CONTEXT": "gpu-context",
            "GPU_FAULT_CONNECTION_MODE": "regional",
            "GPU_FAULT_INSTALLER_ARTIFACT_SHA256": "a" * 64,
            "GPU_FAULT_INSTALLER_CONFIG_DIGEST": "b" * 64,
            "GPU_FAULT_VERSION": "0.10.0",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


@pytest.mark.parametrize("preflight", [False, True])
def test_private_snapshot_renders_without_any_remote_reads(
    tmp_path: Path, preflight: bool
) -> None:
    mode = ("--render-only", "--preflight-only") if preflight else ("--render-only",)
    result = render(tmp_path, inputs(), mode=mode)

    assert result.returncode == 0, result.stderr
    assert "unexpected kubectl" not in result.stderr
    manifest = yaml.safe_load(result.stdout)
    assert manifest["kind"] == "Job"
    pod = manifest["spec"]["template"]["spec"]
    assert pod["nodeName"] == "node-a"
    root = next(
        mount
        for mount in pod["containers"][0]["volumeMounts"]
        if mount["name"] == "host-root"
    )
    assert root["readOnly"] is preflight


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cluster_id", "other"),
        ("context", "other"),
        ("namespace", "other"),
        ("node_action_keys_secret", "other"),
        ("connection_secret", "other"),
        ("references_validated", False),
        ("nodes", {}),
    ],
)
def test_cross_scope_or_incomplete_snapshot_is_rejected(
    tmp_path: Path, field: str, value: object
) -> None:
    result = render(tmp_path, {**inputs(), field: value})

    assert result.returncode != 0, "unbound node inputs were accepted"
    assert "unexpected kubectl" not in result.stderr


def test_snapshot_cannot_be_used_to_skip_live_checks_for_installation(
    tmp_path: Path,
) -> None:
    result = render(tmp_path, inputs(), mode=())

    assert result.returncode == 2
    assert "cannot authorize an installation" in result.stderr
    assert "unexpected kubectl" not in result.stderr


def test_group_readable_snapshot_is_rejected(tmp_path: Path) -> None:
    result = render(tmp_path, inputs(), permissions=0o644)

    assert result.returncode == 2
    assert "private, owned" in result.stderr
