from __future__ import annotations

import copy
import json

import pytest

from scripts.e2e.regional import audit_net004_dependency_boundary as runner


def network_facts(boundary=None):
    return {
        "load_balancer": {
            "scheme": "internal",
            "type": "network",
            "state": "active",
            "listener_protocol": "TLS",
            "listener_port": 443,
            "inbound_443_cidrs": [],
            "idle_timeout_seconds": 350,
        },
        "eks": {"endpoint_public_access": False, "endpoint_private_access": True},
        "nat_public_ips": [],
        "probes": [
            {
                "control_resolved_ips": ["10.0.0.1"],
                "nlb_resolved_ips": ["10.0.0.1"],
                "resolved_private": [True],
                "tls_version": "TLSv1.3",
                "tls_handshake_seconds": 0.1,
                "egress_ip": None,
                "executor_timeout_seconds": 10,
            }
        ],
        "credential_boundary": boundary
        or {
            "suspect_env_names": [],
            "suspect_secret_names": [],
            "suspect_mount_paths": [],
            "runtime": {"kubeconfig_env_names": [], "kubeconfig_files_present": []},
            "gpu_eks_unauthenticated_request": {
                "reachable": False,
                "accepted_boundary": False,
            },
        },
    }


@pytest.fixture
def inventory(monkeypatch):
    monkeypatch.setattr(runner, "GPU_EKS_NAME", "gpu-a")
    roles = (
        runner.CONTROL_APP,
        "gpu-fault-control-worker",
        "gpu-fault-telemetry-spool-worker",
    )
    replicas = {name: 2 if index < 2 else 1 for index, name in enumerate(roles)}
    calls, leak, drift = [], set(), set()
    pods = {}
    for name, count in replicas.items():
        for index in range(count):
            pod = f"{name}-{index}"
            pods[pod] = {
                "metadata": {"name": pod, "uid": pod + "-uid"},
                "spec": {
                    "containers": [{"name": "api", "env": [], "volumeMounts": []}],
                    "serviceAccountName": name,
                },
                "status": {
                    "phase": "Running",
                    "conditions": [{"type": "Ready", "status": "True"}],
                    "containerStatuses": [
                        {"name": "api", "ready": True, "containerID": pod}
                    ],
                },
            }

    def cpu(*args, stdin=None, **kwargs):
        calls.append(args)
        if args[:2] == ("get", "deployment"):
            name = args[2]
            return json.dumps(
                {
                    "metadata": {"uid": name + "-uid", "generation": 1},
                    "spec": {"replicas": replicas[name]},
                }
            )
        if args[:2] == ("get", "pods"):
            app = args[3].removeprefix("app=")
            return json.dumps(
                {
                    "items": [
                        value
                        for name, value in pods.items()
                        if name.rsplit("-", 1)[0] == app
                    ]
                }
            )
        if args[:2] == ("get", "pod"):
            result = copy.deepcopy(pods[args[2]])
            if args[2] in drift:
                result["metadata"]["uid"] += "-replacement"
            return json.dumps(result)
        assert args[0] == "exec"
        pod = args[2]
        if "kubeconfig_env_names" in stdin:
            return json.dumps(
                {
                    "kubeconfig_env_names": ["GPU_KUBECONFIG"] if pod in leak else [],
                    "kubeconfig_files_present": [],
                }
            )
        return json.dumps(
            {"reachable": False, "status": None, "accepted_boundary": False}
        )

    monkeypatch.setattr(runner, "cpu", cpu)
    return replicas, pods, calls, leak, drift


def test_all_cpu_roles_and_every_replica_are_checked(inventory) -> None:
    replicas, _, calls, _, _ = inventory
    result = runner.cpu_credential_boundary(
        {"endpoint": "https://unit.invalid", "certificate_authority_data": ""}
    )
    assert result["complete_inventory"] is True
    assert len(result["replicas"]) == sum(replicas.values())
    assert len([call for call in calls if call[0] == "exec"]) == 2 * sum(
        replicas.values()
    )
    facts = network_facts()
    facts["credential_boundary"] = result
    assert all(runner.evaluate_checks(**facts).values()), (
        "complete clean CPU and GPU evidence should satisfy the boundary"
    )


@pytest.mark.parametrize(
    "role",
    [
        "gpu-fault-api-ha",
        "gpu-fault-control-worker",
        "gpu-fault-telemetry-spool-worker",
    ],
)
def test_one_replica_leak_cannot_hide_behind_a_clean_ingress(inventory, role) -> None:
    _, _, _, leak, _ = inventory
    leak.add(role + "-0")
    result = runner.cpu_credential_boundary(
        {"endpoint": "https://unit.invalid", "certificate_authority_data": ""}
    )
    facts = network_facts()
    facts["credential_boundary"] = result
    assert runner.evaluate_checks(**facts)["cpu_has_no_gpu_kubeconfig"] is False


def test_missing_replica_and_uid_drift_fail_closed(inventory) -> None:
    _, pods, _, _, drift = inventory
    target = "gpu-fault-control-worker-1"
    drift.add(target)
    with pytest.raises(runner.CaseError, match="changed"):
        runner.cpu_credential_boundary(
            {"endpoint": "https://unit.invalid", "certificate_authority_data": ""}
        )
    drift.clear()
    pods.pop(target)
    with pytest.raises(runner.CaseError, match="missing"):
        runner.cpu_credential_boundary(
            {"endpoint": "https://unit.invalid", "certificate_authority_data": ""}
        )


def test_disabled_spool_is_declared_but_does_not_require_a_nonexistent_pod(
    inventory,
) -> None:
    replicas, pods, _, _, _ = inventory
    replicas["gpu-fault-telemetry-spool-worker"] = 0
    pods.pop("gpu-fault-telemetry-spool-worker-0")
    result = runner.cpu_credential_boundary(
        {"endpoint": "https://unit.invalid", "certificate_authority_data": ""}
    )
    assert len(result["deployments"]) == 3 and len(result["replicas"]) == 4


def test_old_one_pod_evidence_is_insufficient_even_when_clean() -> None:
    checks = runner.evaluate_checks(**network_facts())
    assert checks["cpu_has_no_gpu_kubeconfig"] is False
    assert checks["gpu_eks_reverse_boundary_is_enforced"] is False
