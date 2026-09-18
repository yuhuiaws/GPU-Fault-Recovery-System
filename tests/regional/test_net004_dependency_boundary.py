"""NET-004 audit verdicts over recorded facts."""

from __future__ import annotations

from typing import Any

import pytest

from scripts.e2e.regional import audit_net004_dependency_boundary as net004
from tests.regional import test_acceptance_alignment_net004 as boundary_fixtures

cpu_inventory = boundary_fixtures.inventory


def _facts(
    *probe_timeouts: float, boundary: dict[str, Any] | None = None
) -> dict[str, Any]:
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
                "executor_timeout_seconds": timeout,
            }
            for timeout in probe_timeouts
        ],
        "credential_boundary": boundary
        if boundary is not None
        else {
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


@pytest.mark.parametrize("spool_enabled", [True, False])
def test_executor_timeout_is_judged_against_the_idle_timeout_not_a_constant(
    cpu_inventory: Any, spool_enabled: bool
) -> None:
    replicas, pods, _calls, _leak, _drift = cpu_inventory
    if not spool_enabled:
        replicas["gpu-fault-telemetry-spool-worker"] = 0
        pods.pop("gpu-fault-telemetry-spool-worker-0")
    boundary = net004.cpu_credential_boundary(
        {"endpoint": "https://unit.invalid", "certificate_authority_data": ""}
    )
    assert {
        row["deployment"]: row["replicas"] for row in boundary["deployments"]
    } == replicas, "the boundary must declare every CPU role, including disabled spool"
    assert {row["pod_uid"] for row in boundary["replicas"]} == {
        pod["metadata"]["uid"] for pod in pods.values()
    }, "every existing CPU replica must have its own identity-bound proof"
    assert len(boundary["replicas"]) == sum(replicas.values()), (
        "the declared replica count must match the inspected inventory"
    )
    checks = net004.evaluate_checks(**_facts(12.0, boundary=boundary))
    assert checks["executor_timeout_is_below_nlb_idle_timeout"] is True, (
        "a retuned client timeout below the NLB idle timeout is still correct"
    )
    assert all(checks.values()), checks


def test_one_pod_boundary_cannot_satisfy_the_all_role_contract() -> None:
    checks = net004.evaluate_checks(**_facts(12.0))
    assert checks["cpu_has_no_gpu_kubeconfig"] is False, (
        "legacy evidence cannot prove all CPU replicas are credential-free"
    )
    assert checks["gpu_eks_reverse_boundary_is_enforced"] is False, (
        "legacy evidence cannot prove every replica's reverse boundary"
    )


def test_executor_timeout_at_or_over_the_idle_timeout_fails() -> None:
    assert (
        net004.evaluate_checks(**_facts(350.0))[
            "executor_timeout_is_below_nlb_idle_timeout"
        ]
        is False
    )
    assert (
        net004.evaluate_checks(**_facts(15.0, 400.0))[
            "executor_timeout_is_below_nlb_idle_timeout"
        ]
        is False
    )


def test_no_probes_is_not_a_pass() -> None:
    assert (
        net004.evaluate_checks(**_facts())["executor_timeout_is_below_nlb_idle_timeout"]
        is False
    )


def test_the_constant_is_gone() -> None:
    assert not hasattr(net004, "EXECUTOR_TIMEOUT_SECONDS"), (
        "the executor timeout must be read from the deployed config, not hard-coded"
    )
