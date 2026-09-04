"""The public network bootstrap puts on the control plane's front door.

Everything here decides what can reach the regional TLS NLB: which subnets are
public, which internet gateway they route through, and which source addresses the
NLB security group admits. The allow-list is the whole access control -- the NLB
terminates TLS for the executor API, so an ingress rule wider than the GPU
clusters' NAT addresses exposes it to the internet, and a missing one silently
cuts a GPU cluster off from the control plane. These tests also pin the ownership
labels, because uninstall deletes exactly what bootstrap claims to have created.
"""

from __future__ import annotations

import ipaddress
import json
import subprocess
from itertools import takewhile
from typing import Any, Mapping, Sequence

import pytest

from gpu_fault.admin import bootstrap as admin_bootstrap
from gpu_fault.admin.bootstrap import (
    _ensure_internet_gateway,
    _ensure_nlb_network,
    _ensure_public_subnets,
    _ensure_security_group,
    _gpu_nat_eips,
    _unused_subnet_cidrs,
)
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    ClusterIdentity,
    CommandRunner,
)
from tests.admin._bootstrap_support import _cluster

SITE_ID = "site-a"
REGION = "us-east-1"
VPC = "vpc-control"
IGW = "igw-existing"
GPU_A_EIP = "203.0.113.10"
GPU_B_EIP = "203.0.113.11"


def _tags(values: Mapping[str, str]) -> list[dict[str, str]]:
    return [{"Key": key, "Value": item} for key, item in values.items()]


def _owned() -> list[dict[str, str]]:
    return _tags({"gpu-fault:site-id": SITE_ID, "gpu-fault:purpose": "public-nlb"})


def _subnet(
    subnet_id: str,
    availability_zone: str,
    cidr: str,
    *,
    tags: Sequence[dict[str, str]] = (),
) -> dict[str, Any]:
    return {
        "SubnetId": subnet_id,
        "AvailabilityZone": availability_zone,
        "CidrBlock": cidr,
        "VpcId": VPC,
        "Tags": list(tags),
    }


def _route_table(
    table_id: str,
    *,
    subnet_id: str | None = None,
    main: bool = False,
    tags: Sequence[dict[str, str]] = (),
    gateway: str | None = IGW,
) -> dict[str, Any]:
    association: dict[str, Any] = {"RouteTableAssociationId": f"rtbassoc-{table_id}"}
    if main:
        association["Main"] = True
    if subnet_id is not None:
        association["SubnetId"] = subnet_id
    routes = [{"DestinationCidrBlock": "10.0.0.0/16", "GatewayId": "local"}]
    if gateway is not None:
        routes.append({"DestinationCidrBlock": "0.0.0.0/0", "GatewayId": gateway})
    return {
        "RouteTableId": table_id,
        "VpcId": VPC,
        "Associations": [association],
        "Routes": routes,
        "Tags": list(tags),
    }


PRIVATE = [
    _subnet("subnet-private-a", f"{REGION}a", "10.0.1.0/24"),
    _subnet("subnet-private-b", f"{REGION}b", "10.0.2.0/24"),
]
PUBLIC = [
    _subnet("subnet-public-a", f"{REGION}a", "10.0.9.0/28", tags=_owned()),
    _subnet("subnet-public-b", f"{REGION}b", "10.0.9.16/28", tags=_owned()),
]
PUBLIC_TABLES = [
    _route_table("rtb-public-a", subnet_id="subnet-public-a", tags=_owned()),
    _route_table("rtb-public-b", subnet_id="subnet-public-b", tags=_owned()),
]


def _filters(argv: Sequence[str]) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    for item in argv:
        if item.startswith("Name=") and ",Values=" in item:
            name, raw = item[len("Name=") :].split(",Values=", 1)
            parsed[name] = raw.split(",")
    return parsed


def _following(argv: Sequence[str], flag: str) -> list[str]:
    index = list(argv).index(flag)
    return list(takewhile(lambda item: not item.startswith("--"), argv[index + 1 :]))


class Aws:
    """The EC2 API as this VPC currently looks.

    Read calls answer from the state the test declares; mutating calls hand back a
    freshly minted identifier and are recorded, so a test can assert both what was
    created and -- more often -- that nothing was.
    """

    def __init__(
        self,
        *,
        subnets: Sequence[dict[str, Any]] = PRIVATE,
        route_tables: Sequence[dict[str, Any]] = (),
        gateways: Sequence[dict[str, Any]] = (),
        security_groups: Sequence[dict[str, Any]] = (),
        nat_eips: Mapping[str, Sequence[str]] | None = None,
        vpc_cidrs: Sequence[str] = ("10.0.0.0/16",),
        failures: Sequence[tuple[str, int, str]] = (),
    ) -> None:
        self.subnets = list(subnets)
        self.route_tables = list(route_tables)
        self.gateways = list(gateways)
        self.security_groups = list(security_groups)
        self.nat_eips = dict(
            {"vpc-gpu-a": [GPU_A_EIP]} if nat_eips is None else nat_eips
        )
        self.vpc_cidrs = list(vpc_cidrs)
        self.failures = tuple(failures)
        self.calls: list[list[str]] = []
        self.created = 0

    def __call__(
        self, arguments: Sequence[Any], **_keywords: Any
    ) -> subprocess.CompletedProcess:
        argv = [str(item) for item in arguments]
        self.calls.append(argv)
        line = " ".join(argv)
        for fragment, returncode, stderr in self.failures:
            if fragment in line:
                return subprocess.CompletedProcess(argv, returncode, "", stderr)
        handler = getattr(self, "_" + argv[2].replace("-", "_"), None)
        payload: Any = "" if handler is None else handler(argv)
        if not isinstance(payload, str):
            payload = json.dumps(payload)
        return subprocess.CompletedProcess(argv, 0, payload, "")

    def matching(self, fragment: str) -> list[list[str]]:
        return [argv for argv in self.calls if fragment in " ".join(argv)]

    def mutations(self) -> list[str]:
        verbs = ("create-", "associate-", "modify-", "attach-", "authorize-", "delete-")
        return [argv[2] for argv in self.calls if argv[2].startswith(verbs)]

    # -- reads ---------------------------------------------------------------

    def _describe_subnets(self, argv: Sequence[str]) -> dict[str, Any]:
        if "--subnet-ids" in argv:
            wanted = set(_following(argv, "--subnet-ids"))
            return {
                "Subnets": [item for item in self.subnets if item["SubnetId"] in wanted]
            }
        return {"Subnets": list(self.subnets)}

    def _describe_route_tables(self, argv: Sequence[str]) -> dict[str, Any]:
        filters = _filters(argv)
        if "association.subnet-id" in filters:
            subnet_id = filters["association.subnet-id"][0]
            return {
                "RouteTables": [
                    table
                    for table in self.route_tables
                    if any(
                        item.get("SubnetId") == subnet_id
                        for item in table["Associations"]
                    )
                ]
            }
        return {
            "RouteTables": [
                table
                for table in self.route_tables
                if any(item.get("Main") for item in table["Associations"])
            ]
        }

    def _describe_vpcs(self, _argv: Sequence[str]) -> dict[str, Any]:
        return {
            "Vpcs": [
                {
                    "CidrBlock": self.vpc_cidrs[0],
                    "CidrBlockAssociationSet": [
                        {"CidrBlock": cidr, "CidrBlockState": {"State": "associated"}}
                        for cidr in self.vpc_cidrs
                    ],
                }
            ]
        }

    def _describe_internet_gateways(self, _argv: Sequence[str]) -> dict[str, Any]:
        return {"InternetGateways": list(self.gateways)}

    def _describe_security_groups(self, _argv: Sequence[str]) -> dict[str, Any]:
        return {"SecurityGroups": list(self.security_groups)}

    def _describe_nat_gateways(self, argv: Sequence[str]) -> dict[str, Any]:
        vpc_id = _filters(argv)["vpc-id"][0]
        return {
            "NatGateways": [
                {"NatGatewayAddresses": [{"PublicIp": eip}]}
                for eip in self.nat_eips.get(vpc_id, [])
            ]
        }

    # -- mutations -----------------------------------------------------------

    def _create_internet_gateway(self, _argv: Sequence[str]) -> dict[str, Any]:
        return {"InternetGateway": {"InternetGatewayId": "igw-created"}}

    def _create_subnet(self, _argv: Sequence[str]) -> dict[str, Any]:
        self.created += 1
        return {"Subnet": {"SubnetId": f"subnet-created-{self.created}"}}

    def _create_route_table(self, _argv: Sequence[str]) -> str:
        return f"rtb-created-{self.created}"

    def _associate_route_table(self, _argv: Sequence[str]) -> str:
        return f"rtbassoc-created-{self.created}"

    def _create_security_group(self, _argv: Sequence[str]) -> str:
        return "sg-created"


def _gpu(name: str, vpc_id: str) -> ClusterIdentity:
    account = "123456789012"
    return ClusterIdentity(
        input_arn=f"arn:aws:eks:{REGION}:{account}:cluster/{name}",
        role="gpu",
        region=REGION,
        account_id=account,
        hyperpod_arn=f"arn:aws:sagemaker:{REGION}:{account}:cluster/hp-{name}",
        hyperpod_name=f"hp-{name}",
        eks_arn=f"arn:aws:eks:{REGION}:{account}:cluster/{name}",
        eks_name=name,
        vpc_id=vpc_id,
        subnet_ids=(f"subnet-{name}",),
        node_recovery="None",
        context=name,
    )


def _network(
    aws: Aws,
    monkeypatch: pytest.MonkeyPatch,
    *,
    dry_run: bool = False,
    gpu_clusters: Sequence[ClusterIdentity] = (),
) -> dict[str, Any]:
    monkeypatch.setattr(admin_bootstrap.subprocess, "run", aws)
    return _ensure_nlb_network(
        CommandRunner(dry_run=dry_run),
        cpu=_cluster(),
        gpu_clusters=list(gpu_clusters) or [_gpu("gpu-a", "vpc-gpu-a")],
        site_id=SITE_ID,
    )


def _prepared(**overrides: Any) -> Aws:
    """A VPC that a previous bootstrap already prepared."""

    defaults: dict[str, Any] = {
        "subnets": PRIVATE + PUBLIC,
        "route_tables": PUBLIC_TABLES,
        "gateways": [{"InternetGatewayId": IGW, "Tags": _owned()}],
        "security_groups": [{"GroupId": "sg-existing", "Tags": _owned()}],
    }
    defaults.update(overrides)
    return Aws(**defaults)


def test_an_already_prepared_vpc_is_reused_without_touching_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second bootstrap of the same site must be a no-op on the network.

    Re-creating subnets or the security group would move the NLB to new subnets and
    drop every established executor connection; creating a second internet gateway
    is not even possible. Only the ingress authorization repeats, and that is
    idempotent on the AWS side.
    """

    aws = _prepared()

    result = _network(aws, monkeypatch)

    assert result["public_subnets"] == ["subnet-public-a", "subnet-public-b"]
    assert result["security_group"] == "sg-existing"
    assert result["security_group_ownership"] == "CREATED"
    assert result["internet_gateway"]["internet_gateway_id"] == IGW
    assert result["vpc_id"] == VPC
    assert aws.mutations() == ["authorize-security-group-ingress"], (
        "an existing network was rebuilt instead of reused"
    )


def test_the_ingress_allow_list_is_exactly_the_gpu_nat_addresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The NLB admits the GPU clusters' egress addresses and nothing else.

    This security group is the only thing between the public subnets and the TLS
    listener that fronts the executor API. Each rule is a single ``/32`` on 443; a
    wider CIDR here is a directly reachable control plane.
    """

    aws = _prepared(
        nat_eips={"vpc-gpu-a": [GPU_A_EIP], "vpc-gpu-b": [GPU_B_EIP, GPU_A_EIP]}
    )

    result = _network(
        aws,
        monkeypatch,
        gpu_clusters=[_gpu("gpu-a", "vpc-gpu-a"), _gpu("gpu-b", "vpc-gpu-b")],
    )

    assert result["gpu_nat_eips"] == [GPU_A_EIP, GPU_B_EIP], (
        "an address shared by two GPU VPCs was authorized twice"
    )
    authorized = aws.matching("authorize-security-group-ingress")
    assert [argv[argv.index("--cidr") + 1] for argv in authorized] == [
        f"{GPU_A_EIP}/32",
        f"{GPU_B_EIP}/32",
    ]
    for argv in authorized:
        assert argv[argv.index("--port") + 1] == "443"
        assert argv[argv.index("--protocol") + 1] == "tcp"
        assert argv[argv.index("--group-id") + 1] == "sg-existing"
    assert not [argv for argv in authorized if "0.0.0.0/0" in argv], (
        "the NLB security group was opened to the internet"
    )


def test_a_gpu_cluster_without_a_nat_address_stops_the_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A GPU cluster with no NAT egress can never reach the control plane.

    Continuing would produce a site that looks healthy while that cluster's
    collectors never report a fault, which is indistinguishable from a cluster with
    no faults.
    """

    aws = _prepared(nat_eips={})

    with pytest.raises(BootstrapError, match="no available NAT EIP"):
        _network(aws, monkeypatch)


def test_a_duplicate_ingress_rule_is_not_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rule already being there is the desired end state.

    Bootstrap is re-run after every partial failure, so the second run always finds
    its own rules; treating that as an error would make the site unrecoverable.
    """

    aws = _prepared(
        failures=[
            (
                "authorize-security-group-ingress",
                254,
                "An error occurred (InvalidPermission.Duplicate) ...",
            )
        ]
    )

    result = _network(aws, monkeypatch)

    assert result["gpu_nat_eips"] == [GPU_A_EIP]


def test_an_ingress_rule_that_could_not_be_added_stops_the_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reporting success without the rule hides an unreachable control plane.

    The failure surfaces at bootstrap time, where it can be fixed, rather than as
    connection timeouts from a GPU cluster later.
    """

    aws = _prepared(
        failures=[
            (
                "authorize-security-group-ingress",
                254,
                "An error occurred (AccessDenied)",
            )
        ]
    )

    with pytest.raises(BootstrapError, match="AccessDenied"):
        _network(aws, monkeypatch)


def test_missing_public_subnets_are_created_in_two_zones_behind_the_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh VPC gets two public subnets, each routed to the internet gateway.

    Two zones is what makes the NLB survive a zone failure, and the default route
    has to point at the discovered gateway: a subnet with no ``0.0.0.0/0`` route is
    not public, so the NLB would come up unreachable.
    """

    aws = Aws()

    result = _network(aws, monkeypatch)

    created = aws.matching("create-subnet")
    zones = [argv[argv.index("--availability-zone") + 1] for argv in created]
    assert zones == [f"{REGION}a", f"{REGION}b"]
    assert result["public_subnets"] == ["subnet-created-1", "subnet-created-2"]
    assert result["internet_gateway"]["internet_gateway_id"] == "igw-created"
    assert result["internet_gateway"]["ownership"] == "CREATED"
    assert aws.matching("attach-internet-gateway"), (
        "a new internet gateway was created but never attached to the VPC"
    )
    for argv in aws.matching("create-route "):
        assert argv[argv.index("--gateway-id") + 1] == "igw-created"
        assert argv[argv.index("--destination-cidr-block") + 1] == "0.0.0.0/0"
    assert len(aws.matching("modify-subnet-attribute")) == 2
    assert len(aws.matching("associate-route-table")) == 2
    resources = result["subnet_resources"]
    assert [item["ownership"] for item in resources] == ["CREATED", "CREATED"]
    assert [item["route_table_id"] for item in resources] == [
        "rtb-created-1",
        "rtb-created-2",
    ]


def test_created_subnets_never_overlap_an_existing_subnet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CIDR that overlaps an existing subnet cannot be allocated at all.

    The EKS private subnets in this VPC carry the customer's own workloads; picking
    an overlapping range fails the API call at best and, for the second of a pair,
    would collide with the subnet just created.
    """

    aws = Aws()

    _network(aws, monkeypatch)

    chosen = [
        ipaddress.ip_network(argv[argv.index("--cidr-block") + 1])
        for argv in aws.matching("create-subnet")
    ]
    existing = [ipaddress.ip_network(item["CidrBlock"]) for item in PRIVATE]
    for candidate in chosen:
        assert not any(
            candidate.overlaps(item)
            for item in existing + [other for other in chosen if other != candidate]
        ), f"{candidate} overlaps a subnet that already exists"


def test_a_vpc_with_no_free_address_space_stops_the_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """There is nowhere to put the public subnets, and that has to be said plainly.

    The operator has to widen the VPC or hand over existing public subnets; a
    generic API error from ``create-subnet`` would not say which.
    """

    full = _subnet("subnet-full", f"{REGION}a", "10.0.0.0/16")
    aws = Aws(subnets=PRIVATE + [full], vpc_cidrs=["10.0.0.0/16"])

    with pytest.raises(BootstrapError, match="no free /28 CIDR blocks"):
        _network(aws, monkeypatch)


def test_a_second_vpc_cidr_is_used_when_the_first_is_full() -> None:
    """Secondary CIDR blocks are part of the VPC's address space.

    Refusing to look at them would fail bootstrap on a VPC that has room, and the
    operator's only fix would be to resize the primary block.
    """

    selected = _unused_subnet_cidrs(
        ["10.0.0.0/16", "10.1.0.0/24"], ["10.0.0.0/16"], count=2
    )

    assert [str(ipaddress.ip_network(item).network_address) for item in selected] == [
        "10.1.0.0",
        "10.1.0.16",
    ]


def test_a_control_plane_cluster_in_one_zone_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single-zone control plane cannot host a zone-redundant NLB.

    Building it anyway would produce a control plane that a single zone outage takes
    down, which is exactly the failure this system is supposed to ride out.
    """

    single = [_subnet("subnet-private-a", f"{REGION}a", "10.0.1.0/24")]
    aws = Aws(subnets=single)
    monkeypatch.setattr(admin_bootstrap.subprocess, "run", aws)

    with pytest.raises(BootstrapError, match="at least two availability zones"):
        _ensure_public_subnets(CommandRunner(), cluster=_cluster(), site_id=SITE_ID)


def test_a_tagged_subnet_that_is_not_public_is_not_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ownership tags do not make a subnet public; a default route does.

    A subnet whose internet gateway route was removed cannot carry the NLB, so it
    has to be passed over rather than counted -- otherwise the NLB is placed in a
    subnet with no path to the internet.
    """

    private_route = _route_table(
        "rtb-public-a", subnet_id="subnet-public-a", tags=_owned(), gateway=None
    )
    aws = _prepared(
        subnets=PRIVATE + PUBLIC[:1],
        route_tables=[private_route],
        gateways=[{"InternetGatewayId": IGW, "Tags": _owned()}],
    )

    result = _network(aws, monkeypatch)

    assert result["public_subnets"] == ["subnet-created-1", "subnet-created-2"]
    assert len(aws.matching("create-subnet")) == 2


def test_an_owned_public_subnet_with_no_route_table_of_its_own_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A subnet on the VPC's main route table cannot be managed as ours.

    Its route table is shared with the rest of the VPC, so uninstall could not
    delete it and bootstrap must not record it as a created resource.
    """

    main = _route_table("rtb-main", main=True, tags=_owned())
    aws = _prepared(subnets=PRIVATE + PUBLIC[:1], route_tables=[main])

    with pytest.raises(BootstrapError, match="has no route table"):
        _network(aws, monkeypatch)


def test_an_untagged_route_table_under_our_subnet_is_adopted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An untagged route table is a half-finished earlier run, not a foreign one.

    Tagging it is what lets uninstall delete it; leaving it untagged would strand a
    route table in the customer's VPC after uninstall reports a clean run.
    """

    untagged = _route_table("rtb-public-a", subnet_id="subnet-public-a")
    aws = _prepared(route_tables=[untagged, PUBLIC_TABLES[1]])

    result = _network(aws, monkeypatch)

    tagging = aws.matching("create-tags")
    assert tagging, "the adopted route table was never tagged for this site"
    assert "rtb-public-a" in tagging[0]
    assert f"Key=gpu-fault:site-id,Value={SITE_ID}" in tagging[0]
    assert result["public_subnets"] == ["subnet-public-a", "subnet-public-b"]


def test_a_route_table_owned_by_another_site_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two sites in one VPC must not share a route table.

    Adopting it would make this site's uninstall delete the other site's route to
    the internet.
    """

    foreign = _route_table(
        "rtb-public-a",
        subnet_id="subnet-public-a",
        tags=_tags({"gpu-fault:site-id": "site-b"}),
    )
    aws = _prepared(route_tables=[foreign, PUBLIC_TABLES[1]])

    with pytest.raises(BootstrapError, match="belongs to site 'site-b'"):
        _network(aws, monkeypatch)


def test_a_security_group_owned_by_another_site_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The NLB security group is per site and never shared.

    Sharing it would let this site's GPU addresses reach the other site's control
    plane, and would have this site's uninstall revoke the other site's ingress.
    """

    aws = _prepared(
        security_groups=[
            {"GroupId": "sg-existing", "Tags": _tags({"gpu-fault:site-id": "site-b"})}
        ]
    )

    with pytest.raises(BootstrapError, match="security group .* belongs to site"):
        _network(aws, monkeypatch)


def test_an_untagged_security_group_with_our_name_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An untagged group of the same name is not safe to adopt.

    Its rules were written by something else; adding the GPU addresses to it would
    extend whatever it already fronts.
    """

    aws = _prepared(security_groups=[{"GroupId": "sg-existing", "Tags": []}])

    with pytest.raises(BootstrapError, match="refusing to share it"):
        _network(aws, monkeypatch)


def test_a_new_security_group_is_tagged_for_this_site(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tag is what makes the group findable and deletable later.

    Without it the next bootstrap refuses to adopt the group it created itself, and
    uninstall leaves it behind.
    """

    aws = _prepared(security_groups=[])

    result = _network(aws, monkeypatch)

    assert result["security_group"] == "sg-created"
    assert result["security_group_ownership"] == "CREATED"
    tagged = [argv for argv in aws.matching("create-tags") if "sg-created" in argv]
    assert tagged, "the new security group was never tagged for this site"
    assert f"Key=gpu-fault:site-id,Value={SITE_ID}" in tagged[0]
    create = aws.matching("create-security-group")[0]
    assert create[create.index("--vpc-id") + 1] == VPC


def test_an_internet_gateway_created_by_someone_else_is_recorded_external(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A VPC almost always already has a gateway, and it is not ours to delete.

    ``EXTERNAL`` is what keeps uninstall from detaching the gateway the customer's
    own workloads egress through.
    """

    aws = _prepared(gateways=[{"InternetGatewayId": "igw-customer", "Tags": []}])

    result = _network(aws, monkeypatch)

    assert result["internet_gateway"] == {
        "internet_gateway_id": "igw-customer",
        "ownership": "EXTERNAL",
        "vpc_id": VPC,
    }
    assert "create-internet-gateway" not in aws.mutations()


def test_a_dry_run_reports_the_plan_without_creating_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--dry-run`` is what an operator reads before authorizing a bootstrap.

    It has to answer the same questions -- how many subnets, which gateway, which
    ingress addresses -- while leaving the account untouched, including the ingress
    rules.
    """

    aws = Aws()

    result = _network(aws, monkeypatch, dry_run=True)

    assert aws.mutations() == [], "a dry run mutated the account"
    assert result["public_subnets"] == [
        "subnet-dryrun-public-1",
        "subnet-dryrun-public-2",
    ]
    assert result["internet_gateway"]["internet_gateway_id"] == f"igw-dryrun-{SITE_ID}"
    assert result["security_group"].startswith("sg-dryrun-"), (
        "the dry run reported a real security group id"
    )
    assert result["gpu_nat_eips"] == [GPU_A_EIP]


def test_the_nlb_name_is_a_stable_short_name_for_the_site(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The NLB and its security group share one name derived from the site id.

    AWS caps a load balancer name at 32 characters, and the name is how a re-run
    finds the group it created, so it cannot be random or truncated differently
    each time.
    """

    aws = _prepared()

    first = _network(aws, monkeypatch)
    second = _network(_prepared(), monkeypatch)

    assert first["name"] == second["name"]
    assert first["name"] == f"gpu-fault-{SITE_ID}"
    assert len(first["name"]) <= 32
    lookup = aws.matching("describe-security-groups")[0]
    assert f"Name=group-name,Values={first['name']}" in lookup


def test_a_gateway_lookup_failure_is_not_read_as_an_absent_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only an empty answer means there is no gateway.

    An access-denied read that fell through to the create path would try to attach a
    second internet gateway to a VPC that already has one.
    """

    aws = Aws(
        failures=[
            (
                "describe-internet-gateways",
                254,
                "An error occurred (UnauthorizedOperation)",
            )
        ]
    )
    monkeypatch.setattr(admin_bootstrap.subprocess, "run", aws)

    with pytest.raises(BootstrapError, match="UnauthorizedOperation"):
        _ensure_internet_gateway(
            CommandRunner(), region=REGION, vpc_id=VPC, site_id=SITE_ID
        )

    assert "create-internet-gateway" not in aws.mutations()


def test_nat_addresses_come_only_from_available_gateways(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deleted NAT gateway keeps its recorded address but not its function.

    Authorizing an address that no longer routes anywhere adds a rule for whoever
    holds that EIP next.
    """

    aws = Aws()
    monkeypatch.setattr(admin_bootstrap.subprocess, "run", aws)

    _gpu_nat_eips(CommandRunner(), _gpu("gpu-a", "vpc-gpu-a"))

    query = aws.matching("describe-nat-gateways")[0]
    assert "Name=state,Values=available" in query


def test_an_existing_group_is_matched_inside_its_own_vpc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Security group names are only unique per VPC.

    Looking one up without the VPC filter can find a same-named group in another
    VPC, and the returned id would then be authorized instead of the real one.
    """

    aws = _prepared()
    monkeypatch.setattr(admin_bootstrap.subprocess, "run", aws)

    result = _ensure_security_group(
        CommandRunner(),
        cluster=_cluster(),
        group_name="gpu-fault-site-a",
        description="GPU fault regional TLS NLB",
        site_id=SITE_ID,
    )

    assert result["vpc_id"] == VPC
    lookup = aws.matching("describe-security-groups")[0]
    assert f"Name=vpc-id,Values={VPC}" in lookup
