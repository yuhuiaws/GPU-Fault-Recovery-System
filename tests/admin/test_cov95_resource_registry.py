from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from collections import deque
from pathlib import Path

import pytest

from gpu_fault.admin import resource_registry as registry
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.deadlines import deadline_scope
from gpu_fault.admin.resource_registry_dns import vpc_association_entry
from gpu_fault.admin.site import load_site
from gpu_fault.installation_resources import InstallationResourceDeletePolicy as Policy
from gpu_fault.installation_resources import InstallationResourceOwnership as Ownership
from tests.admin._aws_cleanup_support import ACCOUNT, REGION, TAGS, Aws
from tests.admin.test_admin_site import site_file
from tests.admin.test_registry_transport import snapshot


class RegistryTransport:
    def __init__(self):
        self.calls = []
        self.pod = subprocess.CompletedProcess(["fake"], 0, "cpu-fixture", "")
        self.results = deque()
        arn = f"arn:aws:elasticloadbalancing:{REGION}:{ACCOUNT}:loadbalancer/net/example/id"
        self.aws = Aws(
            {
                ("elbv2", "describe-load-balancers"): {
                    "LoadBalancers": [
                        {"LoadBalancerArn": arn, "DNSName": "example.elb.amazonaws.com"}
                    ]
                },
                ("elbv2", "describe-listeners"): {
                    "Listeners": [{"ListenerArn": arn + "/listener"}, {}]
                },
                ("elbv2", "describe-target-groups"): {
                    "TargetGroups": [{"TargetGroupArn": arn + "/target"}, {}]
                },
                ("eks", "describe-cluster"): {
                    "cluster": {"resourcesVpcConfig": {"vpcId": "vpc-example"}}
                },
                ("ec2", "describe-internet-gateways"): {"InternetGateways": []},
            }
        )

    def __call__(self, arguments, **options):
        self.calls.append((arguments, options))
        if arguments[0] == "aws":
            return self.aws(arguments, **options)
        assert arguments[0] == "kubectl", "unexpected registry transport"
        if "get" in arguments and "pod" in arguments:
            return self.pod
        assert "exec" in arguments and "cpu-fixture" in arguments
        if self.results:
            return self.results.popleft()
        script = arguments[-1]
        if script == registry.FETCH_SCRIPT:
            value = snapshot()
        else:
            value = registry.InstallationResourceSnapshot.model_validate_json(
                options["input_text"]
            )
        response = (
            str(len(value.resources))
            if script == registry.DIRECT_SYNC_SCRIPT
            else json.dumps([item.model_dump(mode="json") for item in value.resources])
        )
        return subprocess.CompletedProcess(arguments, 0, response, "")


@pytest.fixture
def context(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    site = load_site(site_file(tmp_path))
    transport = RegistryTransport()
    monkeypatch.setattr(registry, "run_command", transport)
    return site, transport


@pytest.mark.parametrize(
    "operation",
    [
        registry.fetch_installation_resource_registry,
        registry.sync_installation_resource_registry,
    ],
)
def test_registry_lifecycle_reads_cpu_pod_and_writes_bound_snapshot(context, operation):
    site, transport = context
    result = operation(site)
    path = (
        result
        if isinstance(result, Path)
        else site.source.parent / "installation-resources.json"
    )
    loaded = registry.load_installation_resource_snapshot(path)
    loaded.require_source_binding()
    assert loaded.site_id == site.release_config["site_name"]
    assert path.stat().st_mode & 0o077 == 0
    assert path.with_suffix(".json.sha256").stat().st_mode & 0o077 == 0
    assert not path.with_suffix(".json.tmp").exists(), (
        "snapshot publication left a temporary file"
    )
    assert any("cpu-fixture" in arguments for arguments, _options in transport.calls), (
        "registry synchronization never reached the CPU Pod"
    )


@pytest.mark.parametrize("status,output", [(1, ""), (0, ""), (0, " \n")])
def test_registry_requires_running_cpu_pod_before_exec(context, status, output):
    site, transport = context
    transport.pod = subprocess.CompletedProcess(["fake"], status, output, "example")
    with pytest.raises(BootstrapError, match="no Running CPU"):
        registry.fetch_installation_resource_registry(site)
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "kind",
    [
        "missing-file",
        "missing-digest",
        "empty-digest",
        "invalid-model",
        "source-mismatch",
    ],
)
def test_snapshot_corruption_fails_closed_without_transport(context, kind):
    site, transport = context
    path = registry.write_installation_resource_snapshot(site, snapshot())
    digest = path.with_suffix(".json.sha256")
    if kind == "missing-file":
        path.unlink()
    elif kind == "missing-digest":
        digest.unlink()
    elif kind == "empty-digest":
        digest.write_text("")
    else:
        document = json.loads(path.read_text())
        if kind == "invalid-model":
            document["resources"] = "invalid"
        else:
            document["source_sha256"] = "0" * 64
        path.write_text(json.dumps(document))
        digest.write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(BootstrapError, match="incomplete|empty|invalid"):
        registry.load_installation_resource_snapshot(path)
    assert transport.calls == []


@pytest.mark.parametrize(
    "document",
    [
        [],
        {"site_id": "test-site"},
        {"site_id": "test-site", "resources": []},
        {"site_id": "test-site", "resources": {"initial_deploy_target": []}},
    ],
)
def test_bootstrap_documents_require_valid_inventory_and_target(context, document):
    site, transport = context
    (site.source.parent / "bootstrap-state.json").write_text(json.dumps(document))
    with pytest.raises(BootstrapError, match="object|inventory|CPU identity"):
        registry.find_bootstrap_state(site)
    assert transport.calls == []


def test_bootstrap_home_search_ignores_foreign_sites_and_rejects_ambiguity(context):
    site, transport = context
    home = Path.home() / ".gpu-fault/bootstrap"
    foreign = home / "foreign/bootstrap-state.json"
    foreign.parent.mkdir(parents=True)
    foreign.write_text(json.dumps({"site_id": "foreign", "resources": {}}))
    assert registry.find_bootstrap_state(site) is None
    state = {
        "site_id": "test-site",
        "resources": {
            "initial_deploy_target": {
                "cpu": {
                    "eks_arn": site.release_config["cpu_eks_arn"],
                    "hyperpod_name": "control",
                }
            }
        },
    }
    matching = home / "matching/bootstrap-state.json"
    matching.parent.mkdir()
    matching.write_text(json.dumps(state))
    assert registry.find_bootstrap_state(site) == state
    (site.source.parent / "bootstrap-state.json").write_text(json.dumps(state))
    with pytest.raises(BootstrapError, match="multiple bootstrap states"):
        registry.find_bootstrap_state(site)
    assert transport.calls == []


@pytest.mark.parametrize(
    "document", [[], {}, {"LoadBalancers": []}, {"LoadBalancers": [{}, {}]}]
)
def test_runtime_discovery_requires_one_nlb(context, document):
    site, transport = context
    transport.aws.responses[("elbv2", "describe-load-balancers")] = document
    with pytest.raises(BootstrapError, match="no object|exactly one"):
        registry.discover_runtime_resources(site)
    assert len(transport.calls) == 1


@pytest.mark.parametrize("status,output", [(1, "{}"), (0, "not-json")])
def test_runtime_discovery_rejects_tool_and_json_errors(context, status, output):
    site, transport = context
    transport.aws.responses[("elbv2", "describe-load-balancers")] = (
        subprocess.CompletedProcess(["fake"], status, output, "example-denial")
    )
    with pytest.raises(BootstrapError, match="failed to discover|invalid JSON"):
        registry.discover_runtime_resources(site)
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "output,expected",
    [
        ("invalid-json", "invalid JSON"),
        ("{}", "no resource list"),
        ('[{"resource_id":"example"}]', "invalid resources"),
        ("[]", "empty"),
    ],
)
def test_registry_fetch_rejects_invalid_or_empty_confirmation(
    context, output, expected
):
    site, transport = context
    transport.results.append(subprocess.CompletedProcess(["fake"], 0, output, ""))
    with pytest.raises(BootstrapError, match=expected):
        registry.fetch_installation_resource_registry(site)
    assert not (site.source.parent / "installation-resources.json").exists(), (
        "invalid registry response was persisted as a snapshot"
    )


@pytest.mark.parametrize(
    "status,output",
    [
        (44, "wrong-marker"),
        (1, "GPU_FAULT_LEGACY_REGISTRY_API"),
        (75, registry.UNAVAILABLE_REGISTRY_MARKER),
    ],
)
def test_registry_fetch_does_not_treat_arbitrary_failure_as_legacy(
    context, status, output
):
    site, transport = context
    transport.results.append(
        subprocess.CompletedProcess(["fake"], status, output, "example-error")
    )
    with pytest.raises(BootstrapError, match="fetch failed"):
        registry.fetch_installation_resource_registry(site)
    assert len(transport.calls) == 2


@pytest.mark.parametrize(
    "operation",
    [
        registry.sync_installation_resource_snapshot,
        registry.sync_installation_resource_snapshot_direct,
    ],
)
def test_sync_requires_exact_confirmation(context, operation):
    site, transport = context
    transport.results.append(
        subprocess.CompletedProcess(
            ["fake"],
            0,
            "[]" if operation is registry.sync_installation_resource_snapshot else "0",
            "",
        )
    )
    with pytest.raises(BootstrapError, match="proof"):
        operation(site, snapshot())
    assert len(transport.calls) == 2


def test_direct_sync_failure_is_not_success(context):
    site, transport = context
    transport.results.append(
        subprocess.CompletedProcess(["fake"], 1, "", "example-error")
    )
    with pytest.raises(BootstrapError, match="direct Aurora"):
        registry.sync_installation_resource_snapshot_direct(site, snapshot())
    assert len(transport.calls) == 2


def test_retryable_api_outage_reaches_direct_sync_only_after_bounded_attempts(
    context, monkeypatch
):
    site, transport = context
    transport.results.extend(
        [
            subprocess.CompletedProcess(
                ["fake"], 75, registry.UNAVAILABLE_REGISTRY_MARKER, ""
            )
        ]
        * 4
    )
    sleeps = []
    monkeypatch.setattr(registry.time, "sleep", sleeps.append)
    with deadline_scope("registry fixture", 20):
        registry.sync_installation_resource_snapshot(site, snapshot())
    assert sleeps == [2, 4, 8]
    scripts = [
        arguments[-1] for arguments, _options in transport.calls if "exec" in arguments
    ]
    assert scripts == [registry.SYNC_SCRIPT] * 4 + [registry.DIRECT_SYNC_SCRIPT]


def legacy_network():
    return {
        "site_id": "test-site",
        "resources": {
            "nlb_network": {
                "public_subnets": ["subnet-example"],
                "security_group": "sg-example",
            }
        },
    }


def configure_legacy_network(transport, *, owned=True):
    tags = TAGS if owned else []
    transport.aws.responses.update(
        {
            ("ec2", "describe-subnets"): {
                "Subnets": [
                    {
                        "SubnetId": "subnet-example",
                        "AvailabilityZone": "us-east-1a",
                        "VpcId": "vpc-example",
                        "Tags": tags,
                    }
                ]
            },
            ("ec2", "describe-route-tables"): {
                "RouteTables": [
                    {
                        "RouteTableId": "rtb-example",
                        "Associations": [
                            {"SubnetId": "subnet-other"},
                            {
                                "SubnetId": "subnet-example",
                                "RouteTableAssociationId": "rtbassoc-example",
                            },
                        ],
                    }
                ]
            },
            ("ec2", "describe-security-groups"): {
                "SecurityGroups": [{"GroupId": "sg-example", "Tags": tags}]
            },
            ("ec2", "describe-internet-gateways"): {
                "InternetGateways": [{"InternetGatewayId": "igw-example", "Tags": tags}]
            },
        }
    )


@pytest.mark.parametrize("owned", [False, True])
def test_legacy_enrichment_reconstructs_ownership_without_mutating_input(
    context, owned
):
    site, transport = context
    configure_legacy_network(transport, owned=owned)
    original = legacy_network()
    unchanged = copy.deepcopy(original)
    enriched = registry.enrich_legacy_bootstrap(site, original)
    assert original == unchanged
    network = enriched["resources"]["nlb_network"]
    expected = "CREATED" if owned else "EXTERNAL"
    assert network["subnet_resources"][0]["ownership"] == expected
    assert network["security_group_ownership"] == expected
    assert network["internet_gateway"]["ownership"] == expected
    if owned:
        assert network["subnet_resources"][0]["route_table_id"] == "rtb-example"
        assert (
            network["subnet_resources"][0]["route_table_association_id"]
            == "rtbassoc-example"
        )
    else:
        assert not any(
            call[2] == "describe-route-tables" for call in transport.aws.calls
        ), "external subnet was treated as owned route-table infrastructure"
    assert transport.aws.mutations == []


@pytest.mark.parametrize(
    "stage,count",
    [
        ("table", 0),
        ("table", 2),
        ("association", 0),
        ("association", 2),
        ("security-group", 0),
        ("security-group", 2),
    ],
)
def test_legacy_reconstruction_refuses_ambiguous_network_identity(
    context, stage, count
):
    site, transport = context
    configure_legacy_network(transport)
    if stage == "table":
        transport.aws.responses[("ec2", "describe-route-tables")]["RouteTables"] = [
            {}
        ] * count
    elif stage == "association":
        transport.aws.responses[("ec2", "describe-route-tables")]["RouteTables"][0][
            "Associations"
        ] = [{"SubnetId": "subnet-example"}] * count
    else:
        transport.aws.responses[("ec2", "describe-security-groups")][
            "SecurityGroups"
        ] = [{}] * count
    with pytest.raises(BootstrapError, match="cannot reconstruct"):
        registry.enrich_legacy_bootstrap(site, legacy_network())
    assert transport.aws.mutations == []


@pytest.mark.parametrize(
    "vpc,gateways", [(None, 0), ("vpc-example", 0), ("vpc-example", 2)]
)
def test_legacy_gateway_is_not_invented_from_missing_or_ambiguous_discovery(
    context, vpc, gateways
):
    site, transport = context
    transport.aws.responses[("eks", "describe-cluster")] = {
        "cluster": {"resourcesVpcConfig": {"vpcId": vpc}}
    }
    transport.aws.responses[("ec2", "describe-internet-gateways")] = {
        "InternetGateways": [{}] * gateways
    }
    result = registry.enrich_legacy_bootstrap(site, {"resources": {}})
    assert "internet_gateway" not in result["resources"]["nlb_network"]
    assert transport.aws.mutations == []


def test_legacy_snapshot_requires_matching_bootstrap_and_unambiguous_ownership(context):
    site, transport = context
    with pytest.raises(BootstrapError, match="ownership cannot be proven"):
        registry.build_legacy_installation_snapshot(site)
    path = site.source.parent / "bootstrap-state.json"
    path.write_text(json.dumps({"site_id": "test-site", "resources": {}}))
    with pytest.raises(BootstrapError, match="ownership is ambiguous"):
        registry.build_legacy_installation_snapshot(site)
    state = {
        "site_id": "test-site",
        "resources": {
            "nlb_network": {
                "security_group": "sg-123",
                "security_group_ownership": "CREATED",
            },
            "pki": {
                "certificate_arn": site.release_config["nlb"]["certificate_arn"],
                "certificate_ownership": "CREATED",
            },
            "monitoring_resources": {
                "sns_topic_arn": site.release_config["health"]["sns_topic_arn"],
                "sns_topic_ownership": "CREATED",
            },
            "aurora": {
                "cluster_id": "gpu-fault-aurora",
                "cluster_ownership": "CREATED",
            },
        },
    }
    path.write_text(json.dumps(state))
    result = registry.build_legacy_installation_snapshot(site)
    result.require_source_binding()
    assert result.resources, "legacy reconstruction returned an empty registry"
    assert transport.aws.mutations == []


@pytest.mark.parametrize("zone_owned", [False, True])
def test_snapshot_preserves_external_zones_but_records_owned_association_detach(
    context, zone_owned
):
    site, transport = context
    site.release_config["dns"] = {"hosted_zone_id": "ZEXAMPLE"}
    state = {
        "site_id": site.metadata_name,
        "resources": {
            "nlb_network": {"vpc_id": "vpc-example"},
            "pki": {
                "hosted_zone_id": "ZEXAMPLE",
                "zone_ownership": "CREATED" if zone_owned else "EXTERNAL",
                "vpc_associations": [
                    vpc_association_entry(
                        vpc_region=REGION,
                        vpc_id="vpc-gpu",
                        cluster_ids=["gpu-a"],
                        ownership="CREATED",
                    ),
                    vpc_association_entry(
                        vpc_region=REGION,
                        vpc_id="vpc-example",
                        cluster_ids=[],
                        ownership="CREATED" if zone_owned else "EXTERNAL",
                        native=True,
                    ),
                ],
            },
        },
    }
    result = registry.build_installation_snapshot(site, state)
    by_key = {item.resource_key: item for item in result.resources}
    key = f"aws/route53/vpc-association/{REGION}/vpc-gpu"
    assert [
        item.resource_key
        for item in result.resources
        if item.resource_type == "route53_vpc_association"
    ] == [key]
    assert by_key[key].resource_id == f"ZEXAMPLE:{REGION}:vpc-gpu"
    assert by_key[key].attributes == {
        "hosted_zone_id": "ZEXAMPLE",
        "vpc_region": REGION,
        "vpc_id": "vpc-gpu",
    }
    assert by_key[key].ownership is Ownership.CREATED
    assert by_key[key].delete_policy is Policy.DETACH
    assert by_key[key].dependencies == ["aws/route53/zone"]
    assert by_key["aws/route53/zone"].delete_policy is (
        Policy.DELETE if zone_owned else Policy.PRESERVE
    )
    assert transport.calls == []


@pytest.mark.parametrize("provider", [False, True])
@pytest.mark.parametrize(
    "lbc", [{"external": True}, {"role_arn": f"arn:aws:iam::{ACCOUNT}:role/lbc"}]
)
def test_snapshot_role_inventory_does_not_invent_oidc_or_lbc_policy(
    context, provider, lbc
):
    site, transport = context
    role = {
        "role_arn": f"arn:aws:iam::{ACCOUNT}:role/executor",
        "cluster_name": "gpu-a",
    }
    if provider:
        role["oidc_provider_arn"] = (
            f"arn:aws:iam::{ACCOUNT}:oidc-provider/example.invalid"
        )
        role["oidc_provider_ownership"] = "CREATED"
    state = {
        "resources": {"executor_role:gpu-a": role, "load_balancer_controller": lbc}
    }
    by_key = {
        item.resource_key: item
        for item in registry.build_installation_snapshot(site, state).resources
    }
    assert ("aws/iam/executor/gpu-a/oidc-provider" in by_key) == provider
    if provider:
        assert (
            by_key["aws/iam/executor/gpu-a/oidc-provider"].delete_policy
            is Policy.DETACH
        )
    assert "aws/iam/lbc/policy" not in by_key
    assert ("kubernetes/lbc/helm-release" in by_key) == ("external" not in lbc)
    assert by_key["aws/iam/executor/gpu-a/role"].ownership is Ownership.CREATED
    assert transport.calls == []


def test_snapshot_empty_optional_infrastructure_has_no_phantom_records(context):
    site, transport = context
    site.release_config["health"] = {}
    site.release_config["nlb"].update(
        public_subnets=",subnet-example,", security_group=None, certificate_arn=None
    )
    state = {
        "resources": {
            "nlb_network": {
                "subnet_resources": [
                    {
                        "subnet_id": "subnet-example",
                        "ownership": "EXTERNAL",
                        "route_table_id": "rtb-example",
                    }
                ]
            }
        }
    }
    by_key = {
        item.resource_key: item
        for item in registry.build_installation_snapshot(site, state).resources
    }
    assert "aws/aurora/cluster" not in by_key
    assert "aws/acm/certificate" not in by_key
    assert "aws/nlb/security-group" not in by_key
    assert "aws/network/public-route-association/1" not in by_key
    fallback = registry.build_installation_snapshot(site, None)
    subnets = [
        item for item in fallback.resources if item.resource_type == "ec2_subnet"
    ]
    assert [item.resource_id for item in subnets] == ["subnet-example"]
    assert transport.calls == []
