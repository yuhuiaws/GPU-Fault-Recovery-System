"""Tests for deploy/dataplane/tools/verify_dataplane_executor.py.

Every data-plane gap found on real hardware so far had the same shape:
the Pod was Running, the logs were clean, and the executor still could
not act. These drive the verifier against a stubbed kubectl so each of
those shapes is pinned as a failure rather than a green deploy.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy/dataplane/tools/verify_dataplane_executor.py"
CONTROL_PLANE_KUBECONFIG = "/tmp/verify-dataplane-control-plane.kubeconfig"

ROLE_ARN = "arn:aws:iam::111122223333:role/gpu-fault-executor"
WHEEL = "gpu-fault-control-plane-wheel-0100-abcdef123456"


def encoded(value: str) -> str:
    return base64.b64encode(value.encode()).decode()


STUB = '''#!/usr/bin/env python3
"""A kubectl that answers from a JSON world file."""
import json
import os
import sys

world = json.load(open(os.environ["GPU_FAULT_TEST_WORLD"]))
args = [a for a in sys.argv[1:]]
control_plane = os.environ.get("KUBECONFIG") == %(kubeconfig)r
scope = "control_plane" if control_plane else "data_plane"

# Drop the flags the verifier passes so positional parsing is simple.
positional = []
skip_next = False
for arg in args:
    if skip_next:
        skip_next = False
        continue
    if arg in {"-n", "--context"}:
        skip_next = True
        continue
    if arg == "-o":
        skip_next = True
        continue
    positional.append(arg)

if positional[0] == "exec":
    sys.exit(0 if world.get("token_present", True) else 1)

kind = positional[1]
if kind == "pods":
    print(json.dumps({"items": world[scope].get("pods", [])}))
    sys.exit(0)
name = positional[2]
item = world[scope].get(kind, {}).get(name)
if item is None:
    sys.stderr.write("NotFound\\n")
    sys.exit(1)
print(json.dumps(item))
''' % {"kubeconfig": CONTROL_PLANE_KUBECONFIG}


def _container(*, flags: dict | None = None) -> dict:
    env = [
        {"name": "GPU_FAULT_ENABLE_NODE_ACTION_ADAPTER", "value": "true"},
        {
            "name": "GPU_FAULT_NODE_ACTION_KEYS_DIR",
            "value": "/etc/gpu-fault/node-action-keys",
        },
        {"name": "GPU_FAULT_ENABLE_HYPERPOD_ADAPTER", "value": "true"},
        {"name": "GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER", "value": "true"},
        {"name": "GPU_FAULT_CLUSTER_EXECUTOR_REMOTE_STATE", "value": "true"},
    ]
    for entry in env:
        if flags and entry["name"] in flags:
            entry["value"] = flags[entry["name"]]
    return {"name": "executor", "env": env}


def _pod(*, role_arn: str | None = ROLE_ARN, wheel: str = WHEEL) -> dict:
    executor = _container()
    executor["env"] = [
        *executor["env"],
        *(
            [
                {"name": "AWS_ROLE_ARN", "value": role_arn},
                {
                    "name": "AWS_WEB_IDENTITY_TOKEN_FILE",
                    "value": (
                        "/var/run/secrets/eks.amazonaws.com/serviceaccount/token"
                    ),
                },
            ]
            if role_arn
            else []
        ),
    ]
    return {
        "metadata": {"name": "gpu-fault-cluster-executor-abc"},
        "spec": {
            "serviceAccountName": "gpu-fault-cluster-executor",
            "containers": [executor],
            "volumes": [
                {"name": "artifact", "configMap": {"name": wheel}},
                {
                    "name": "node-action-keys",
                    "secret": {"secretName": "gpu-fault-node-action-keys"},
                },
            ],
        },
    }


def _world(**overrides) -> dict:
    world = {
        "data_plane": {
            "deployment": {
                "gpu-fault-cluster-executor": {
                    "metadata": {"name": "gpu-fault-cluster-executor"},
                    "spec": {
                        "replicas": 2,
                        "template": {
                            "spec": {
                                "serviceAccountName": ("gpu-fault-cluster-executor"),
                                "containers": [_container()],
                                "volumes": [
                                    {"name": "artifact", "configMap": {"name": WHEEL}},
                                    {
                                        "name": "node-action-keys",
                                        "secret": {
                                            "secretName": ("gpu-fault-node-action-keys")
                                        },
                                    },
                                ],
                            }
                        },
                    },
                    "status": {"readyReplicas": 2},
                }
            },
            "serviceaccount": {
                "gpu-fault-cluster-executor": {
                    "metadata": {
                        "name": "gpu-fault-cluster-executor",
                        "annotations": {"eks.amazonaws.com/role-arn": ROLE_ARN},
                    }
                }
            },
            "secret": {
                "gpu-fault-regional-connection": {
                    "data": {
                        "control-plane-url": encoded("https://control.example"),
                        "cluster-token": encoded("x" * 32),
                        "cluster-id": encoded("cluster-a"),
                        "allowed-namespaces": encoded("gpu-fault-system,training"),
                        "ca.crt": encoded("certificate"),
                        "hyperpod-cluster-name": encoded("hp-a"),
                        "hyperpod-confirm-cluster-name": encoded("hp-a"),
                    }
                },
                "gpu-fault-node-action-keys": {"data": {"hyperpod-i-123": "eA=="}},
            },
            "configmap": {WHEEL: {"metadata": {"name": WHEEL}}},
            "pods": [_pod()],
        },
        "control_plane": {
            "deployment": {
                "gpu-fault-api-ha": {
                    "spec": {
                        "replicas": 3,
                        "template": {
                            "spec": {
                                "volumes": [
                                    {"name": "artifact", "configMap": {"name": WHEEL}}
                                ]
                            }
                        },
                    }
                }
            },
            "secret": {
                "gpu-fault-regional-clusters": {
                    "data": {
                        "clusters.json": encoded(
                            json.dumps(
                                [
                                    {
                                        "cluster_id": "cluster-a",
                                        "allowed_namespaces": [
                                            "gpu-fault-system",
                                            "training",
                                        ],
                                    }
                                ]
                            )
                        )
                    }
                }
            },
        },
    }
    world.update(overrides)
    return world


def _run(tmp_path, world: dict) -> subprocess.CompletedProcess:
    stub = tmp_path / "kubectl"
    stub.write_text(STUB, encoding="utf-8")
    stub.chmod(0o755)
    world_file = tmp_path / "world.json"
    world_file.write_text(json.dumps(world), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        text=True,
        capture_output=True,
        check=False,
        env={
            "PATH": f"{tmp_path}:/usr/bin:/bin",
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "GPU_FAULT_TEST_WORLD": str(world_file),
            "GPU_FAULT_CONTROL_PLANE_KUBECONFIG": (CONTROL_PLANE_KUBECONFIG),
        },
    )


def test_healthy_data_plane_passes(tmp_path) -> None:
    result = _run(tmp_path, _world())

    assert result.returncode == 0, result.stdout + result.stderr
    assert "executor check passed" in result.stdout


def test_missing_irsa_annotation_fails(tmp_path) -> None:
    world = _world()
    del world["data_plane"]["serviceaccount"]["gpu-fault-cluster-executor"]["metadata"][
        "annotations"
    ]

    result = _run(tmp_path, world)

    assert result.returncode == 1
    assert "eks.amazonaws.com/role-arn annotation" in result.stdout


def test_unsubstituted_placeholder_fails(tmp_path) -> None:
    # A sed that missed the placeholder is worse than no annotation: the
    # apply overwrites a correct ARN with the literal placeholder.
    world = _world()
    world["data_plane"]["serviceaccount"]["gpu-fault-cluster-executor"]["metadata"][
        "annotations"
    ]["eks.amazonaws.com/role-arn"] = "REPLACE_WITH_EXECUTOR_IRSA_ROLE_ARN"

    result = _run(tmp_path, world)

    assert result.returncode == 1
    assert "manifest placeholder" in result.stdout


def test_pod_predating_the_annotation_fails(tmp_path) -> None:
    # The projected token volume is injected at pod creation, so a pod
    # older than the annotation keeps running with no credentials while
    # the ServiceAccount looks correct.
    world = _world()
    world["data_plane"]["pods"] = [_pod(role_arn=None)]

    result = _run(tmp_path, world)

    assert result.returncode == 1
    assert "rollout restart" in result.stdout


def test_pod_assuming_a_stale_role_fails(tmp_path) -> None:
    world = _world()
    world["data_plane"]["pods"] = [_pod(role_arn="arn:aws:iam::111122223333:role/old")]

    result = _run(tmp_path, world)

    assert result.returncode == 1
    assert "restart the deployment" in result.stdout


def test_absent_projected_token_fails(tmp_path) -> None:
    result = _run(tmp_path, _world(token_present=False))

    assert result.returncode == 1
    assert "no projected identity token" in result.stdout


def test_node_action_adapter_disabled_fails(tmp_path) -> None:
    world = _world()
    world["data_plane"]["deployment"]["gpu-fault-cluster-executor"]["spec"]["template"][
        "spec"
    ]["containers"] = [
        _container(flags={"GPU_FAULT_ENABLE_NODE_ACTION_ADAPTER": "false"})
    ]

    result = _run(tmp_path, world)

    assert result.returncode == 1
    # The reason matters: false looks like the safe default and is not.
    assert "isolate the whole node" in result.stdout


def test_spare_failover_without_remote_state_fails(tmp_path) -> None:
    world = _world()
    world["data_plane"]["deployment"]["gpu-fault-cluster-executor"]["spec"]["template"][
        "spec"
    ]["containers"] = [
        _container(flags={"GPU_FAULT_CLUSTER_EXECUTOR_REMOTE_STATE": "false"})
    ]

    result = _run(tmp_path, world)

    assert result.returncode == 1
    assert "REMOTE_STATE=true" in result.stdout


def test_missing_hyperpod_confirmation_key_fails(tmp_path) -> None:
    # The key is optional: true in the manifest, so a missing key is a
    # startup crash loop rather than an apply-time error.
    world = _world()
    del world["data_plane"]["secret"]["gpu-fault-regional-connection"]["data"][
        "hyperpod-confirm-cluster-name"
    ]

    result = _run(tmp_path, world)

    assert result.returncode == 1
    assert "hyperpod-confirm-cluster-name" in result.stdout


def test_allowed_namespace_drift_fails(tmp_path) -> None:
    world = _world()
    world["data_plane"]["secret"]["gpu-fault-regional-connection"]["data"][
        "allowed-namespaces"
    ] = encoded("gpu-fault-system")

    result = _run(tmp_path, world)

    assert result.returncode == 1
    assert "allowed namespace drift" in result.stdout


def test_wheel_behind_the_control_plane_fails(tmp_path) -> None:
    world = _world()
    world["control_plane"]["deployment"]["gpu-fault-api-ha"]["spec"]["template"][
        "spec"
    ]["volumes"] = [
        {
            "name": "artifact",
            "configMap": {"name": "gpu-fault-control-plane-wheel-0100-999999999999"},
        }
    ]

    result = _run(tmp_path, world)

    assert result.returncode == 1
    assert "the data plane is running different code" in result.stdout


def test_scaled_to_zero_fails(tmp_path) -> None:
    world = _world()
    world["data_plane"]["deployment"]["gpu-fault-cluster-executor"]["spec"][
        "replicas"
    ] = 0
    world["data_plane"]["pods"] = []

    result = _run(tmp_path, world)

    assert result.returncode == 1
    assert "scaled to zero" in result.stdout


def test_no_wheel_reference_available_fails(tmp_path) -> None:
    # A silent skip is how the data plane fell a release behind without
    # anyone noticing, so the missing baseline is itself a failure.
    stub = tmp_path / "kubectl"
    stub.write_text(STUB, encoding="utf-8")
    stub.chmod(0o755)
    world_file = tmp_path / "world.json"
    world_file.write_text(json.dumps(_world()), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        text=True,
        capture_output=True,
        check=False,
        env={
            "PATH": f"{tmp_path}:/usr/bin:/bin",
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "GPU_FAULT_TEST_WORLD": str(world_file),
        },
    )

    assert result.returncode == 1
    assert "cannot verify the wheel ConfigMap" in result.stdout
