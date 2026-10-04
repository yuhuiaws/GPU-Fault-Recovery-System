"""NET-004 audit verdicts over recorded facts."""

from __future__ import annotations

from pathlib import Path
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


# -- mixed-provider site: the NLB allow-list spans every managed cluster's VPC ------


def _cluster(name: str, vpc: str) -> dict[str, Any]:
    return {
        "cluster": {
            "name": name,
            "status": "ACTIVE",
            "endpoint": f"https://{name}.eks",
            "certificateAuthority": {"data": "Q0E="},
            "resourcesVpcConfig": {
                "vpcId": vpc,
                "endpointPublicAccess": False,
                "endpointPrivateAccess": True,
                "publicAccessCidrs": [],
            },
        }
    }


def _fake_aws(
    monkeypatch: pytest.MonkeyPatch, nats: dict[str, list[str]]
) -> list[tuple]:
    calls: list[tuple] = []
    clusters = {
        "hp-eks": _cluster("hp-eks", "vpc-hp"),
        "gpu-eks": _cluster("gpu-eks", "vpc-eks"),
    }

    def aws(service: str, *args: str) -> dict:
        calls.append((service, *args))
        if (service, args[0]) == ("eks", "describe-cluster"):
            return clusters[args[2]]
        if (service, args[0]) == ("ec2", "describe-nat-gateways"):
            vpc = args[2].split("=")[-1]
            return {
                "NatGateways": [
                    {"NatGatewayAddresses": [{"PublicIp": ip}]} for ip in nats[vpc]
                ]
            }
        raise AssertionError(f"unexpected aws call {(service, *args)}")

    monkeypatch.setattr(net004, "aws", aws)
    return calls


def test_nat_egress_set_spans_every_managed_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live 2026-10-04: the NLB allowed the plain-EKS cluster's NAT EIP as well,
    and the audit, reading only the HyperPod VPC, called the allow-list wrong."""

    calls = _fake_aws(
        monkeypatch,
        {"vpc-hp": ["34.209.174.128", "34.218.94.223"], "vpc-eks": ["44.233.19.212"]},
    )
    monkeypatch.setattr(net004, "GPU_EKS_NAME", "hp-eks")
    monkeypatch.setattr(net004, "GPU_EKS_NAMES", ("hp-eks", "gpu-eks"))

    eks, nat_public_ips = net004.eks_and_nat_facts()

    assert nat_public_ips == ["34.209.174.128", "34.218.94.223", "44.233.19.212"]
    assert eks["name"] == "hp-eks" and eks["vpc_id"] == "vpc-hp"
    assert [item["cluster"] for item in eks["nat_sources"]] == ["hp-eks", "gpu-eks"]
    assert eks["nat_sources"][1]["nat_public_ips"] == ["44.233.19.212"]
    assert sum(1 for call in calls if call[1] == "describe-nat-gateways") == 2

    facts = _facts(12.0)
    facts["load_balancer"]["scheme"] = "internet-facing"
    facts["load_balancer"]["inbound_443_cidrs"] = [
        "34.209.174.128/32",
        "34.218.94.223/32",
        "44.233.19.212/32",
    ]
    facts["nat_public_ips"] = nat_public_ips
    facts["probes"][0].update(
        {"resolved_private": [False], "egress_ip": "34.209.174.128"}
    )
    assert (
        net004.evaluate_checks(**facts)["nlb_exposure_matches_deployment_mode"] is True
    )


def test_single_cluster_nat_egress_set_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_aws(monkeypatch, {"vpc-hp": ["34.209.174.128"]})
    monkeypatch.setattr(net004, "GPU_EKS_NAME", "hp-eks")
    monkeypatch.setattr(net004, "GPU_EKS_NAMES", ())

    eks, nat_public_ips = net004.eks_and_nat_facts()

    assert nat_public_ips == ["34.209.174.128"]
    assert [call[:2] for call in calls] == [
        ("eks", "describe-cluster"),
        ("ec2", "describe-nat-gateways"),
    ], "no site: the selected cluster's VPC alone, as before"


def test_managed_gpu_eks_names_come_from_the_site_and_must_include_the_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import gpu_fault.admin.site as site_module

    site_path = tmp_path / "site.yaml"
    site_path.write_text("site", encoding="ascii")
    monkeypatch.setattr(
        site_module,
        "load_site",
        lambda path: SimpleNamespace(
            release_config={
                "clusters": [
                    {"eks_cluster_arn": "arn:aws:eks:us-west-2:1:cluster/hp-eks"},
                    {"eks_cluster_arn": "arn:aws:eks:us-west-2:1:cluster/gpu-eks"},
                ]
            }
        ),
    )
    assert net004.managed_gpu_eks_names(site_path, selected="hp-eks") == (
        "hp-eks",
        "gpu-eks",
    )
    assert net004.managed_gpu_eks_names(None, selected="hp-eks") == ("hp-eks",)
    with pytest.raises(net004.CaseError, match="not a managed cluster"):
        net004.managed_gpu_eks_names(site_path, selected="other")
