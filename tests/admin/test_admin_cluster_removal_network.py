from __future__ import annotations

import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml

from gpu_fault.admin import cluster_removal as removal
from gpu_fault.admin import cluster_removal_network as network
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_removal import (
    RemoveClusterRequest,
    _parallel_cluster_networks,
)
from gpu_fault.admin.cluster_removal_resources import target_resource_plan
from gpu_fault.admin.execution import current_deadline, deadline_scope
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from gpu_fault.admin.site import load_site
from tests.admin.test_admin_cluster_removal import _snapshot
from tests.admin.test_admin_site import site_file

EKS_ARN = "arn:aws:eks:us-east-1:123456789012:cluster/gpu-a"


def configured_site(tmp_path: Path):
    path = site_file(tmp_path)
    value = yaml.safe_load(path.read_text())
    value["spec"]["dns"] = {"hostedZoneId": "Z123", "hostname": "api.example.internal"}
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    return load_site(path)


def result(value, *, code: int = 0, error: str = ""):
    return subprocess.CompletedProcess([], code, json.dumps(value), error)


@pytest.mark.parametrize("shared_with", ["cpu", "gpu"])
def test_shared_vpc_preserves_nat_and_private_dns(
    tmp_path, monkeypatch, shared_with
) -> None:
    site = configured_site(tmp_path)
    monkeypatch.setattr(
        network, "run_command", lambda *_a, **_k: pytest.fail("shared network mutated")
    )

    value = network.detach_network(
        RemoveClusterRequest(site, "gpu-a", "REMOVE_GPU_CLUSTER"),
        target_network={"vpc_id": "vpc-shared", "nat_eips": ["192.0.2.10"]},
        remaining_networks=(
            [{"vpc_id": "vpc-shared", "nat_eips": ["192.0.2.10"]}]
            if shared_with == "gpu"
            else []
        ),
        cpu_vpc_id="vpc-shared" if shared_with == "cpu" else "vpc-cpu",
    )

    assert value["revoked_nat_eips"] == []
    assert value["detached_vpc_id"] is None
    selected, _ = target_resource_plan(site, "gpu-a", _snapshot())
    assert not any(
        item.resource_type == "route53_vpc_association" for item in selected
    ), "shared VPC DNS associations must be excluded from the removal plan"


def test_already_absent_permission_does_not_skip_other_eips(
    tmp_path, monkeypatch
) -> None:
    site = load_site(site_file(tmp_path))
    revoked = []

    def command(arguments, **_kwargs):
        if "revoke-security-group-ingress" in arguments:
            permission = json.loads(arguments[arguments.index("--ip-permissions") + 1])
            revoked.append(permission[0]["IpRanges"][0]["CidrIp"])
            return (
                result(
                    {},
                    code=254,
                    error=(
                        "An error occurred (InvalidPermission.NotFound) when calling the "
                        "RevokeSecurityGroupIngress operation: permission absent"
                    ),
                )
                if len(revoked) == 1
                else result({})
            )
        assert "describe-security-groups" in arguments
        return result({"SecurityGroups": [{"GroupId": "sg-123", "IpPermissions": []}]})

    monkeypatch.setattr(network, "run_command", command)
    value = network.detach_network(
        RemoveClusterRequest(site, "gpu-a", "REMOVE_GPU_CLUSTER"),
        target_network={"vpc_id": "vpc-gpu", "nat_eips": ["192.0.2.10", "192.0.2.11"]},
        remaining_networks=[],
        cpu_vpc_id="vpc-cpu",
    )

    assert revoked == ["192.0.2.10/32", "192.0.2.11/32"]
    assert value["revoked_nat_eips"] == ["192.0.2.10", "192.0.2.11"]


def test_a_successful_revoke_must_be_verified(tmp_path, monkeypatch) -> None:
    site = load_site(site_file(tmp_path))

    def command(arguments, **_kwargs):
        return (
            result(
                {
                    "SecurityGroups": [
                        {
                            "GroupId": "sg-123",
                            "IpPermissions": [
                                {
                                    "IpProtocol": "tcp",
                                    "FromPort": 443,
                                    "ToPort": 443,
                                    "IpRanges": [{"CidrIp": "192.0.2.10/32"}],
                                }
                            ],
                        }
                    ]
                }
            )
            if "describe-security-groups" in arguments
            else result({})
        )

    monkeypatch.setattr(network, "run_command", command)
    with pytest.raises(BootstrapError, match="NAT ingress remains"):
        network.detach_network(
            RemoveClusterRequest(site, "gpu-a", "REMOVE_GPU_CLUSTER"),
            target_network={"vpc_id": "vpc-gpu", "nat_eips": ["192.0.2.10"]},
            remaining_networks=[],
            cpu_vpc_id="vpc-cpu",
        )


def test_default_network_cleanup_does_not_authorize_a_dns_detach(
    tmp_path, monkeypatch
) -> None:
    site = configured_site(tmp_path)
    monkeypatch.setattr(
        network,
        "run_command",
        lambda *_args, **_kwargs: pytest.fail("DNS detach was not authorized"),
    )
    value = network.detach_network(
        RemoveClusterRequest(site, "gpu-a", "REMOVE_GPU_CLUSTER"),
        target_network={"vpc_id": "vpc-gpu-a", "nat_eips": []},
        remaining_networks=[],
        cpu_vpc_id="vpc-cpu",
    )
    assert value["detached_vpc_id"] is None
    assert value["route53_change_id"] is None


def test_absent_route53_response_still_requires_readback(tmp_path, monkeypatch) -> None:
    site = configured_site(tmp_path)
    queried = []
    monkeypatch.setattr(
        network,
        "disassociate_vpc_from_hosted_zone",
        lambda **_k: {"changed": False, "change_id": None, "change_status": None},
    )
    monkeypatch.setattr(
        network, "wait_vpc_association_absent", lambda **kwargs: queried.append(kwargs)
    )

    value = network.detach_network(
        RemoveClusterRequest(site, "gpu-a", "REMOVE_GPU_CLUSTER"),
        target_network={"vpc_id": "vpc-gpu-a", "nat_eips": []},
        remaining_networks=[],
        cpu_vpc_id="vpc-cpu",
        detach_dns=True,
    )

    assert queried == [
        {"hosted_zone_id": "Z123", "region": "us-east-1", "vpc_id": "vpc-gpu-a"}
    ]
    assert value["detached_vpc_id"] == "vpc-gpu-a"
    selected, _ = target_resource_plan(
        site, "gpu-a", _snapshot(), detached_vpc_id="vpc-gpu-a"
    )
    assert any(item.resource_type == "route53_vpc_association" for item in selected), (
        "the verified detached VPC association was omitted from the removal plan"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"HostedZone": {"Id": "Z123"}},
        {"HostedZone": {"Id": "Zother"}, "VPCs": []},
        {"HostedZone": {"Id": "Z123"}, "VPCs": [{}]},
    ],
)
def test_incomplete_route53_inventory_is_not_absence(monkeypatch, payload) -> None:
    monkeypatch.setattr(network, "run_command", lambda *_a, **_k: result(payload))
    with pytest.raises(BootstrapError, match="hosted-zone identity"):
        network.wait_vpc_association_absent(
            hosted_zone_id="Z123", region="us-east-1", vpc_id="vpc-gpu-a"
        )


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"Id": "/change/other", "Status": "INSYNC"},
        {"Id": "/change/expected", "Status": ""},
        {"Id": "/change/expected", "Status": "UNKNOWN"},
    ],
)
def test_route53_waiter_rejects_missing_or_foreign_change_identity(
    monkeypatch, change
) -> None:
    monkeypatch.setattr(
        network, "run_command", lambda *_a, **_k: result({"ChangeInfo": change})
    )
    with pytest.raises(BootstrapError, match="change identity"):
        network.wait_route53_change_insync("/change/expected")


def test_late_route53_success_does_not_extend_the_deadline(monkeypatch) -> None:
    clock = [100.0]
    monkeypatch.setattr(network.time, "monotonic", lambda: clock[0])

    def query(*_args, **_kwargs):
        deadline = current_deadline()
        assert deadline is not None
        assert deadline.remaining() == 1
        clock[0] += 2
        return result({"ChangeInfo": {"Id": "/change/expected", "Status": "INSYNC"}})

    monkeypatch.setattr(network, "run_command", query)
    with deadline_scope("parent", 1):
        with pytest.raises(BootstrapError, match="deadline"):
            network.wait_route53_change_insync("/change/expected", timeout_seconds=300)


def test_supervision_loss_cannot_be_reported_as_idempotent_absence(monkeypatch) -> None:
    def lost(*_args, **_kwargs):
        raise ProcessSupervisionLost("synthetic supervision loss")

    monkeypatch.setattr(network, "run_command", lost)
    with pytest.raises(ProcessSupervisionLost):
        network.disassociate_vpc_from_hosted_zone(
            hosted_zone_id="Z123", region="us-east-1", vpc_id="vpc-gpu-a"
        )


@pytest.mark.parametrize(
    "observed",
    [
        "arn:aws:eks:us-east-1:111122223333:cluster/gpu-a",
        "arn:aws:eks:us-east-1:123456789012:cluster/other",
    ],
)
def test_network_discovery_checks_the_described_full_eks_arn(observed) -> None:
    calls = []

    class Runner:
        def aws_json(self, region, *arguments):
            calls.append(arguments)
            return {
                "cluster": {
                    "arn": observed,
                    "status": "ACTIVE",
                    "createdAt": "2026-01-01",
                    "resourcesVpcConfig": {"vpcId": "vpc-gpu"},
                }
            }

    with pytest.raises(BootstrapError, match="conflicting EKS identity"):
        network.cluster_network(Runner(), region="us-east-1", eks_arn=EKS_ARN)
    assert len(calls) == 1


@pytest.mark.parametrize("error_type", [BootstrapError, KeyboardInterrupt])
def test_failed_network_discovery_cancels_queued_and_drains_started_workers(
    monkeypatch, error_type
) -> None:
    calls = []
    finished = []
    release = threading.Event()
    lock = threading.Lock()

    def discover(_runner, *, region, eks_arn):
        with lock:
            calls.append(eks_arn)
        if eks_arn == "target":
            raise error_type("synthetic discovery failure")
        assert release.wait(timeout=2), (
            "executor shutdown did not release the started discovery worker"
        )
        with lock:
            finished.append(eks_arn)
        return {"vpc_id": eks_arn, "nat_eips": []}

    class DrainingExecutor(ThreadPoolExecutor):
        def __exit__(self, *exc):
            release.set()
            return super().__exit__(*exc)

    monkeypatch.setattr(removal, "_cluster_network", discover)
    monkeypatch.setattr(removal, "ThreadPoolExecutor", DrainingExecutor)
    try:
        with pytest.raises(error_type):
            _parallel_cluster_networks(
                object(),
                region="us-east-1",
                target={"eks_cluster_arn": "target"},
                remaining=[
                    {"eks_cluster_arn": f"remaining-{index}"} for index in range(20)
                ],
                cpu_eks_arn="cpu",
            )
    finally:
        release.set()

    assert len(calls) < 22
    assert len(finished) == len(calls) - 1
