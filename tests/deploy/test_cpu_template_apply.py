from __future__ import annotations

import hashlib
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
PIN_ANNOTATION = "gpu-fault.io/pin-config-sha256"
RESTARTED_AT = "2026-01-01T00:00:00Z"
FAKE_API = """#!/usr/bin/env python3
import fcntl
import hashlib
import json
import os
import sys
import yaml

args = sys.argv[1:]
with open(os.environ["CALLS"], "a") as handle:
    fcntl.flock(handle, fcntl.LOCK_EX)
    handle.write(json.dumps(args) + "\\n")
state = json.loads(os.environ["LIVE_STATE"])
# The release-metadata ConfigMap the script "created"; the Pods below serve it.
pins_file = os.path.join(os.environ["HOME"], "release-metadata.json")
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
    elif kind == "pods":
        deployment = next(arg for arg in args if arg.startswith("app="))[4:]
        print(json.dumps({"items": [{
            "metadata": {"name": deployment + "-0", "labels": {"app": deployment}},
            "spec": {"containers": [{"name": "api",
                                     "ports": [{"name": "http", "containerPort": 8080}]}]},
            "status": {"phase": "Running"},
        }]}))
    elif kind != "pod":
        raise SystemExit("unexpected get")
elif "exec" in args:
    # A Pod polls the ConfigMap itself and reports the digest it serves.
    with open(pins_file) as handle:
        data = json.load(handle)
    digest = hashlib.sha256(json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()
    print(json.dumps({"status": "ok", "fleet_pins": {"content_sha256": digest}}))
elif "create" in args:
    name = args[args.index("configmap") + 1]
    data = dict(arg[len("--from-literal="):].split("=", 1)
                for arg in args if arg.startswith("--from-literal="))
    with open(pins_file, "w") as handle:
        json.dump(data, handle)
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


def pin_content_sha256(data: dict[str, str]) -> str:
    """The shared fleet-pin digest: what every control-plane Pod reports."""

    return hashlib.sha256(
        json.dumps(
            data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


# The shared contract's vector: gpu_fault.fleet_pins.pin_content_sha256 and the
# script's PIN_METADATA_SHA256 must both produce it. ``jq -cS | sha256sum``
# would hash jq's trailing newline as well and yield c2f96399... instead.
PIN_VECTOR = {"b": "x,y", "a": "1", "z": ""}
PIN_VECTOR_SHA256 = "585414de19bbac53560be605120197fd16f19f6ccc14d77b275817cf39280b65"


def test_script_pin_digest_matches_the_shared_vector_without_jqs_newline() -> None:
    line = next(
        line
        for line in SCRIPT.read_text(encoding="utf-8").splitlines()
        if line.startswith("PIN_METADATA_SHA256=")
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            "\n".join(
                [
                    "set -euo pipefail",
                    'RELEASE_METADATA_JSON="$1"',
                    line,
                    'printf "%s" "${PIN_METADATA_SHA256}"',
                ]
            ),
            "digest",
            json.dumps({"data": PIN_VECTOR}),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout == PIN_VECTOR_SHA256
    assert pin_content_sha256(PIN_VECTOR) == PIN_VECTOR_SHA256


def apply_roles(
    tmp_path: Path,
    *,
    targets: str,
    force: bool,
    namespace: str = "gpu-fault-system",
    environment: dict[str, str] | None = None,
) -> tuple[list[dict], list[list[str]], list[list[str]]]:
    binary = tmp_path / "kubectl"
    binary.write_text(FAKE_API, encoding="utf-8")
    binary.chmod(0o755)
    python = tmp_path / "python3"
    python.write_text(
        "#!/bin/bash\nset -eu\n"
        'printf \'%s\\n\' "$*" >>"$PYTHON_CALLS"\n'
        'case "$*" in\n'
        "  *gpu_fault.config_cli*|*verify_control_plane_role_split.py*) exit 0 ;;\n"
        '  *) exec "$REAL_PYTHON" "$@" ;;\n'
        "esac\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    applies = tmp_path / "applies.jsonl"
    python_calls = tmp_path / "python-calls.txt"
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
            "PYTHON_CALLS": str(python_calls),
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
            **(environment or {}),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    return (
        [json.loads(line) for line in applies.read_text(encoding="utf-8").splitlines()],
        [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines()],
        [
            line.split()
            for line in python_calls.read_text(encoding="utf-8").splitlines()
        ],
    )


def applied_pin_digest(documents: list[dict]) -> str:
    """Digest of the release-metadata ConfigMap the script applied."""

    metadata = [
        item
        for item in documents
        if item["kind"] == "ConfigMap"
        and item["metadata"]["name"] == "gpu-fault-release-metadata"
    ]
    assert len(metadata) == 1, "the release-metadata ConfigMap was not applied once"
    return pin_content_sha256(metadata[0]["data"])


def pin_wait_call(python_calls: list[list[str]]) -> list[str]:
    waits = [
        call
        for call in python_calls
        if any(argument.endswith("/wait_control_plane_pins.py") for argument in call)
    ]
    assert len(waits) == 1, (
        "the pin convergence wait must run exactly once per apply: "
        f"{[' '.join(call) for call in python_calls]}"
    )
    return waits[0]


@pytest.mark.parametrize("force", [False, True])
def test_each_cpu_role_receives_one_complete_template_update(
    tmp_path: Path, force: bool
) -> None:
    documents, calls, python_calls = apply_roles(
        tmp_path, targets="spool,worker,ingress", force=force
    )
    deployments = [item for item in documents if item["kind"] == "Deployment"]
    expected_pins = applied_pin_digest(documents)

    assert sorted(item["metadata"]["name"] for item in deployments) == sorted(
        DEPLOYMENTS
    )
    for deployment in deployments:
        annotations = deployment["spec"]["template"]["metadata"]["annotations"]
        assert annotations["gpu-fault.io/control-plane-wheel-sha256"] == "a" * 64
        assert annotations["gpu-fault.io/runtime-image"].endswith("d" * 64), (
            "final template lacks the candidate image digest"
        )
        # A pin change must not roll the tier: the digest is recorded on the
        # Deployment and proven on /healthz, never mixed into the Pod template.
        assert PIN_ANNOTATION not in annotations, (
            "the pin digest on the Pod template rolls every replica per release"
        )
        assert deployment["metadata"]["annotations"][PIN_ANNOTATION] == expected_pins
        assert ("gpu-fault.io/failure-domain-map-sha256" in annotations) is (
            deployment["metadata"]["name"] == "gpu-fault-control-worker"
        )
        assert (
            annotations["kubectl.kubernetes.io/restartedAt"] != RESTARTED_AT
        ) is force
    assert not any(
        "restart" in call or "patch" in call or "set" in call for call in calls
    ), "normal apply issued another mutation after the final template"

    wait = pin_wait_call(python_calls)
    assert wait[wait.index("--expected-sha256") + 1] == expected_pins
    assert wait[wait.index("--namespace") + 1] == "gpu-fault-system"
    assert wait[wait.index("--kubeconfig") + 1] == str(tmp_path / "cpu")
    assert sorted(wait[wait.index("--deployments") + 1].split(",")) == sorted(
        DEPLOYMENTS
    )
    verify = next(
        index
        for index, call in enumerate(python_calls)
        if any(arg.endswith("/verify_control_plane_role_split.py") for arg in call)
    )
    assert python_calls.index(wait) < verify, (
        "the pin convergence wait must run before the role-split verifier"
    )
    probes = [call for call in calls if "exec" in call]
    assert sorted(call[call.index("exec") + 1] for call in probes) == sorted(
        f"{name}-0" for name in DEPLOYMENTS
    ), "every selected role's Pod must be asked for its /healthz pins"
    assert all("/healthz" in " ".join(call) for call in probes), probes
    last_apply = max(index for index, call in enumerate(calls) if "apply" in call)
    assert last_apply < min(calls.index(call) for call in probes), (
        "the convergence wait ran before the final templates were applied"
    )


def test_role_scoped_apply_only_changes_the_selected_template(tmp_path: Path) -> None:
    documents, calls, python_calls = apply_roles(
        tmp_path, targets="worker", force=False
    )
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
    wait = pin_wait_call(python_calls)
    assert wait[wait.index("--deployments") + 1] == "gpu-fault-control-worker", (
        "the convergence wait must cover only the selected roles"
    )
    assert [call[call.index("exec") + 1] for call in calls if "exec" in call] == [
        "gpu-fault-control-worker-0"
    ]


def test_verbatim_rollback_skips_the_pin_convergence_wait(tmp_path: Path) -> None:
    """A rollback restores the previous release's container env and forces a
    restart of every tier, so the restored Pods start from the ConfigMap's env
    snapshot; their wheel may predate ``fleet_pins`` on ``/healthz``. The
    previous release's own contract applies, as it does for the verifier."""

    snapshot = tmp_path / "container-env.json"
    snapshot.write_text("{}", encoding="utf-8")
    documents, calls, python_calls = apply_roles(
        tmp_path,
        targets="spool,worker,ingress",
        force=True,
        environment={"GPU_FAULT_ROLE_SPLIT_CONTAINER_ENV_FILE": str(snapshot)},
    )

    assert len([item for item in documents if item["kind"] == "Deployment"]) == 3
    assert not any(
        any(argument.endswith("/wait_control_plane_pins.py") for argument in call)
        for call in python_calls
    ), "a verbatim rollback must not wait for /healthz pins the old wheel lacks"
    assert not any("exec" in call for call in calls), (
        "a verbatim rollback probed a Pod for its pins"
    )


def test_final_cpu_templates_and_commands_keep_the_requested_namespace(
    tmp_path: Path,
) -> None:
    documents, calls, python_calls = apply_roles(
        tmp_path, targets="spool,worker,ingress", force=False, namespace="isolated-cpu"
    )
    deployments = [item for item in documents if item["kind"] == "Deployment"]
    assert len(deployments) == 3
    assert {item["metadata"]["namespace"] for item in deployments} == {"isolated-cpu"}
    for call in calls:
        if "-n" in call:
            assert call[call.index("-n") + 1] == "isolated-cpu"
    wait = pin_wait_call(python_calls)
    assert wait[wait.index("--namespace") + 1] == "isolated-cpu"
