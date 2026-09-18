from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy/control-plane/tools/apply-control-plane-role-split.sh"
DEPLOYMENTS = (
    "gpu-fault-api-ha",
    "gpu-fault-control-worker",
    "gpu-fault-telemetry-spool-worker",
)
RESTARTED_AT = "2026-01-01T00:00:00Z"
FAKE_API = """#!/usr/bin/env python3
import fcntl
import json
import os
import sys
import yaml

args = sys.argv[1:]
with open(os.environ["CALLS"], "a") as handle:
    fcntl.flock(handle, fcntl.LOCK_EX)
    handle.write(json.dumps(args) + "\\n")
state = json.loads(os.environ["LIVE_STATE"])
if "get" in args:
    kind = args[args.index("get") + 1]
    if kind == "deployments":
        print(json.dumps(state["deployments"]))
    elif kind == "deployment":
        print("{}")
    elif kind == "configmap":
        if "gpu-fault-release-metadata" in args:
            print(json.dumps(state["pins"]))
        else:
            print("false")
    elif kind not in ("pod", "pods"):
        raise SystemExit("unexpected get")
elif "create" in args:
    name = args[args.index("configmap") + 1]
    data = dict(arg[len("--from-literal="):].split("=", 1)
                for arg in args if arg.startswith("--from-literal="))
    print(json.dumps({"apiVersion":"v1", "kind":"ConfigMap",
                     "metadata":{"name":name}, "data":data}))
elif "apply" in args:
    documents = [item for item in yaml.safe_load_all(sys.stdin) if item]
    with open(os.environ["APPLIES"], "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        for document in documents:
            handle.write(json.dumps(document) + "\\n")
elif "patch" in args:
    patch = json.loads(args[args.index("-p") + 1])
    if "spec" in patch:
        raise SystemExit("a second Pod template mutation was attempted")
elif "rollout" in args and "status" in args:
    pass
else:
    raise SystemExit("unexpected mutation: " + " ".join(args))
"""


def apply_roles(
    tmp_path: Path, *, targets: str, force: bool, namespace: str = "gpu-fault-system"
) -> tuple[list[dict], list[list[str]]]:
    binary = tmp_path / "kubectl"
    binary.write_text(FAKE_API, encoding="utf-8")
    binary.chmod(0o755)
    python = tmp_path / "python3"
    python.write_text(
        "#!/bin/bash\nset -eu\n"
        'case "$*" in\n'
        "  *gpu_fault.config_cli*|*verify_control_plane_role_split.py*) exit 0 ;;\n"
        '  *) exec "$REAL_PYTHON" "$@" ;;\n'
        "esac\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    applies = tmp_path / "applies.jsonl"
    state = {
        "pins": {
            "data": {
                "required-agent-artifact-sha256": "a" * 64,
                "required-agent-config-digest": "b" * 64,
                "required-agent-protocol-version": "3",
                "required-regional-executor-protocol-version": "1",
            }
        },
        "deployments": {
            "items": [
                {
                    "metadata": {"name": name},
                    "spec": {
                        "template": {
                            "metadata": {
                                "annotations": {
                                    "kubectl.kubernetes.io/restartedAt": RESTARTED_AT
                                }
                            },
                            "spec": {"containers": [{"name": "api", "env": []}]},
                        }
                    },
                }
                for name in DEPLOYMENTS
            ]
        },
    }
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        env={
            "PATH": str(tmp_path)
            + os.pathsep
            + os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(tmp_path),
            "PYTHONPATH": str(ROOT / "src"),
            "REAL_PYTHON": sys.executable,
            "LIVE_STATE": json.dumps(state),
            "CALLS": str(calls),
            "APPLIES": str(applies),
            "KUBECONFIG": str(tmp_path / "cpu"),
            "GPU_FAULT_AWS_REGION": "us-east-1",
            "GPU_FAULT_NAMESPACE": namespace,
            "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION": "profile-test",
            "GPU_FAULT_NOTIFICATION_CONFIG_SHA256": "c" * 64,
            "GPU_FAULT_WHEEL_SHA256": "a" * 64,
            "GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST": "b" * 64,
            "GPU_FAULT_REQUIRED_AGENT_PROTOCOL_VERSION": "3",
            "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION": "1",
            "GPU_FAULT_CONTROL_PLANE_ROLE_TARGETS": targets,
            "GPU_FAULT_FORCE_ROLE_RESTART": str(force).lower(),
            "GPU_FAULT_RUNTIME_IMAGE": "registry.invalid/runtime@sha256:" + "d" * 64,
            "GPU_FAULT_FAILURE_DOMAIN_MAP_SHA256": "e" * 64,
            "GPU_FAULT_SYNC_INSTALLED_RESOURCE_REGISTRY": "false",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    return (
        [json.loads(line) for line in applies.read_text(encoding="utf-8").splitlines()],
        [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines()],
    )


@pytest.mark.parametrize("force", [False, True])
def test_each_cpu_role_receives_one_complete_template_update(
    tmp_path: Path, force: bool
) -> None:
    documents, calls = apply_roles(
        tmp_path, targets="spool,worker,ingress", force=force
    )
    deployments = [item for item in documents if item["kind"] == "Deployment"]

    assert sorted(item["metadata"]["name"] for item in deployments) == sorted(
        DEPLOYMENTS
    )
    for deployment in deployments:
        annotations = deployment["spec"]["template"]["metadata"]["annotations"]
        assert annotations["gpu-fault.io/control-plane-wheel-sha256"] == "a" * 64
        assert annotations["gpu-fault.io/runtime-image"].endswith("d" * 64), (
            "final template lacks the candidate image digest"
        )
        assert len(annotations["gpu-fault.io/pin-config-sha256"]) == 64
        assert ("gpu-fault.io/failure-domain-map-sha256" in annotations) is (
            deployment["metadata"]["name"] == "gpu-fault-control-worker"
        )
        assert (
            annotations["kubectl.kubernetes.io/restartedAt"] != RESTARTED_AT
        ) is force
    assert not any(
        "restart" in call or "patch" in call or "set" in call for call in calls
    ), "normal apply issued another mutation after the final template"


def test_role_scoped_apply_only_changes_the_selected_template(tmp_path: Path) -> None:
    documents, calls = apply_roles(tmp_path, targets="worker", force=False)
    deployments = [item for item in documents if item["kind"] == "Deployment"]

    assert [item["metadata"]["name"] for item in deployments] == [
        "gpu-fault-control-worker"
    ]
    annotations = deployments[0]["spec"]["template"]["metadata"]["annotations"]
    assert annotations["gpu-fault.io/failure-domain-map-sha256"] == "e" * 64
    patches = [call for call in calls if "patch" in call]
    assert len(patches) == 2
    assert all(
        "spec" not in json.loads(call[call.index("-p") + 1]) for call in patches
    ), "unselected CPU roles received a Pod template patch"


def test_final_cpu_templates_and_commands_keep_the_requested_namespace(
    tmp_path: Path,
) -> None:
    documents, calls = apply_roles(
        tmp_path, targets="spool,worker,ingress", force=False, namespace="isolated-cpu"
    )
    deployments = [item for item in documents if item["kind"] == "Deployment"]
    assert len(deployments) == 3
    assert {item["metadata"]["namespace"] for item in deployments} == {"isolated-cpu"}
    for call in calls:
        if "-n" in call:
            assert call[call.index("-n") + 1] == "isolated-cpu"
