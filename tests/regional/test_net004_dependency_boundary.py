"""NET-004 audit verdicts over recorded facts."""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional import audit_net004_dependency_boundary as net004


def _facts(*probe_timeouts: float) -> dict[str, Any]:
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
        "credential_boundary": {
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


def test_executor_timeout_is_judged_against_the_idle_timeout_not_a_constant() -> None:
    checks = net004.evaluate_checks(**_facts(12.0))
    assert checks["executor_timeout_is_below_nlb_idle_timeout"] is True, (
        "a retuned client timeout below the NLB idle timeout is still correct"
    )
    assert all(checks.values()), checks


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
