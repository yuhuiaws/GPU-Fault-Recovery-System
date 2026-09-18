from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from gpu_fault.admin import cluster_removal_network as network
from gpu_fault.admin import deadlines
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import load_site
from tests.admin.test_admin_site import site_file


class NetworkTransport:
    def __init__(self):
        self.calls = []
        self.clock = 100.0
        self.advance = 0
        self.result = None
        self.responses = {
            "get-hosted-zone": {
                "HostedZone": {"Id": "/hostedzone/ZEXAMPLE"},
                "VPCs": [],
            },
            "get-change": {"ChangeInfo": {"Id": "/change/example", "Status": "INSYNC"}},
            "disassociate-vpc-from-hosted-zone": {
                "ChangeInfo": {"Id": "/change/example", "Status": "PENDING"}
            },
            "describe-security-groups": {
                "SecurityGroups": [{"GroupId": "sg-123", "IpPermissions": []}]
            },
        }

    def command(self, arguments, **_options):
        self.calls.append(arguments)
        self.clock += self.advance
        if self.result is not None:
            return self.result
        value = self.responses[arguments[2]]
        if callable(value):
            value = value()
        return subprocess.CompletedProcess(arguments, 0, json.dumps(value), "")

    def sleep(self, seconds):
        assert 0 < seconds <= 5
        self.clock += seconds


@pytest.fixture
def transport(monkeypatch):
    value = NetworkTransport()
    monkeypatch.setattr(network, "run_command", value.command)
    clock = SimpleNamespace(monotonic=lambda: value.clock, sleep=value.sleep)
    monkeypatch.setattr(network, "time", clock)
    monkeypatch.setattr(deadlines, "time", clock)
    return value


@pytest.mark.parametrize(
    "kind,arguments",
    [
        (
            network.wait_vpc_association_absent,
            {
                "hosted_zone_id": "ZEXAMPLE",
                "region": "us-east-1",
                "vpc_id": "vpc-example",
            },
        ),
        (network.wait_route53_change_insync, {"change_id": "/change/example"}),
    ],
)
@pytest.mark.parametrize("late", [False, True])
def test_route53_waiters_check_deadline_after_read(transport, kind, arguments, late):
    if late:
        transport.advance = 2
        with pytest.raises(BootstrapError, match="deadline"):
            kind(**arguments, timeout_seconds=1)
    else:
        kind(**arguments, timeout_seconds=1)
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "document",
    [
        {},
        {"HostedZone": []},
        {"HostedZone": {"Id": "ZOTHER"}, "VPCs": []},
        {"HostedZone": {"Id": "ZEXAMPLE"}, "VPCs": {}},
        {"HostedZone": {"Id": "ZEXAMPLE"}, "VPCs": [None]},
        {"HostedZone": {"Id": "ZEXAMPLE"}, "VPCs": [{}]},
    ],
)
def test_association_wait_requires_complete_zone_identity(transport, document):
    transport.responses["get-hosted-zone"] = document
    with pytest.raises(BootstrapError, match="hosted-zone identity"):
        network.wait_vpc_association_absent(
            hosted_zone_id="ZEXAMPLE", region="us-east-1", vpc_id="vpc-example"
        )
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "document",
    [
        {},
        {"ChangeInfo": []},
        {"ChangeInfo": {"Id": "/change/other", "Status": "INSYNC"}},
        {"ChangeInfo": {"Id": "/change/example", "Status": "UNKNOWN"}},
    ],
)
def test_change_wait_requires_exact_change_and_known_status(transport, document):
    transport.responses["get-change"] = document
    with pytest.raises(BootstrapError, match="conflicting change identity"):
        network.wait_route53_change_insync("/change/example")
    assert len(transport.calls) == 1


def test_pending_changes_poll_within_the_same_budget(transport):
    sequence = iter(
        [
            {"ChangeInfo": {"Id": "/change/example", "Status": "PENDING"}},
            {"ChangeInfo": {"Id": "/change/example", "Status": "INSYNC"}},
        ]
    )
    transport.responses["get-change"] = lambda: next(sequence)
    result = network.disassociate_vpc_from_hosted_zone(
        hosted_zone_id="ZEXAMPLE", region="us-east-1", vpc_id="vpc-example"
    )
    assert result == {
        "changed": True,
        "change_id": "/change/example",
        "change_status": "INSYNC",
    }
    assert transport.clock == 105
    assert [call[2] for call in transport.calls] == [
        "disassociate-vpc-from-hosted-zone",
        "get-change",
        "get-change",
    ]


@pytest.mark.parametrize(
    "status,output,error",
    [
        (1, "", "AccessDenied"),
        (0, "not-json", ""),
        (0, "[]", ""),
        (0, '{"ChangeInfo":{}}', ""),
        (0, '{"ChangeInfo":{"Id":"example","Status":"UNKNOWN"}}', ""),
    ],
)
def test_disassociate_does_not_treat_unknown_result_as_completion(
    transport, status, output, error
):
    transport.result = subprocess.CompletedProcess(["fake"], status, output, error)
    with pytest.raises(BootstrapError, match="cannot detach|invalid"):
        network.disassociate_vpc_from_hosted_zone(
            hosted_zone_id="ZEXAMPLE", region="us-east-1", vpc_id="vpc-example"
        )
    assert len(transport.calls) == 1


def test_disassociate_accepts_only_explicit_absence(transport):
    transport.result = subprocess.CompletedProcess(
        ["fake"], 254, "", "An error occurred (VPCAssociationNotFound)"
    )
    assert network.disassociate_vpc_from_hosted_zone(
        hosted_zone_id="ZEXAMPLE", region="us-east-1", vpc_id="vpc-example"
    ) == {"changed": False, "change_id": None, "change_status": None}
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "value",
    [
        [],
        None,
        {},
        {"SecurityGroups": []},
        {"SecurityGroups": [{}, {}]},
        {"SecurityGroups": [{"GroupId": "other", "IpPermissions": []}]},
    ],
)
def test_ingress_probe_rejects_missing_or_wrong_group(transport, tmp_path, value):
    site = load_site(site_file(tmp_path))
    transport.responses["describe-security-groups"] = value
    with pytest.raises(BootstrapError, match="object|group identity"):
        network.verify_revoked_nat_eips(site, ["192.0.2.20"])


@pytest.mark.parametrize(
    "permission",
    [
        None,
        {},
        {"IpRanges": [], "IpProtocol": "tcp"},
        {"IpRanges": [], "IpProtocol": "6", "FromPort": True, "ToPort": 443},
        {"IpRanges": [{}], "IpProtocol": "-1"},
        {"IpRanges": [{"CidrIp": "invalid"}], "IpProtocol": "-1"},
    ],
)
def test_ingress_probe_rejects_unverifiable_permission_shapes(
    transport, tmp_path, permission
):
    site = load_site(site_file(tmp_path))
    transport.responses["describe-security-groups"]["SecurityGroups"][0][
        "IpPermissions"
    ] = [permission]
    with pytest.raises(BootstrapError, match="invalid"):
        network.verify_revoked_nat_eips(site, ["192.0.2.20"])


@pytest.mark.parametrize(
    "protocol,start,end,source,blocked",
    [
        ("udp", 443, 443, "192.0.2.20/32", False),
        ("tcp", 80, 80, "192.0.2.20/32", False),
        ("6", 400, 500, "192.0.2.0/24", True),
        ("-1", None, None, "192.0.2.20/32", True),
        ("tcp", 443, 443, "198.51.100.0/24", False),
    ],
)
def test_ingress_probe_checks_effective_https_access(
    transport, tmp_path, protocol, start, end, source, blocked
):
    site = load_site(site_file(tmp_path))
    permission = {
        "IpProtocol": protocol,
        "FromPort": start,
        "ToPort": end,
        "IpRanges": [{"CidrIp": source}],
    }
    transport.responses["describe-security-groups"]["SecurityGroups"][0][
        "IpPermissions"
    ] = [permission]
    if blocked:
        with pytest.raises(BootstrapError, match="NAT ingress remains"):
            network.verify_revoked_nat_eips(site, ["192.0.2.20"])
    else:
        network.verify_revoked_nat_eips(site, ["192.0.2.20"])


@pytest.mark.parametrize(
    "changes",
    [
        {"arn": "arn:aws:eks:us-east-1:123456789012:cluster/other"},
        {"status": "DELETING"},
        {"createdAt": None},
        {"resourcesVpcConfig": {"vpcId": ""}},
    ],
)
def test_network_discovery_rejects_eks_drift(transport, changes):
    arn = "arn:aws:eks:us-east-1:123456789012:cluster/example"
    cluster = {
        "arn": arn,
        "status": "ACTIVE",
        "createdAt": "2026-01-01",
        "resourcesVpcConfig": {"vpcId": "vpc-example"},
    }
    cluster.update(changes)
    runner = SimpleNamespace(aws_json=lambda *_args: {"cluster": cluster})
    with pytest.raises(BootstrapError, match="identity"):
        network.cluster_network(runner, region="us-east-1", eks_arn=arn)
    assert transport.calls == []


@pytest.mark.parametrize(
    "arn",
    [
        "arn:aws:sns:us-east-1:123456789012:example",
        "arn:aws:eks:us-west-2:123456789012:cluster/example",
    ],
)
def test_network_discovery_requires_in_scope_eks_before_aws(arn):
    runner = SimpleNamespace(
        aws_json=lambda *_args: pytest.fail("invalid ARN reached AWS")
    )
    with pytest.raises(BootstrapError, match="EKS ARN"):
        network.cluster_network(runner, region="us-east-1", eks_arn=arn)


def test_network_discovery_requires_nat_inventory():
    arn = "arn:aws:eks:us-east-1:123456789012:cluster/example"
    calls = []

    def aws(_region, _service, operation, *_args):
        calls.append(operation)
        return (
            {
                "cluster": {
                    "arn": arn,
                    "status": "ACTIVE",
                    "createdAt": "2026-01-01",
                    "resourcesVpcConfig": {"vpcId": "vpc-example"},
                }
            }
            if operation == "describe-cluster"
            else {"NatGateways": None}
        )

    with pytest.raises(BootstrapError, match="NAT gateway inventory"):
        network.cluster_network(
            SimpleNamespace(aws_json=aws), region="us-east-1", eks_arn=arn
        )
    assert calls == ["describe-cluster", "describe-nat-gateways"]
