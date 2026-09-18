"""Bounded network discovery and ownership-aware removal."""

from __future__ import annotations

import json
import subprocess
import time
from ipaddress import ip_address, ip_network
from typing import Any, Protocol

from gpu_fault.admin.aws_commands import CommandResult, matches_not_found
from gpu_fault.admin.bootstrap_common import Arn, BootstrapError, CommandRunner
from gpu_fault.admin.execution import deadline_scope, run_command
from gpu_fault.admin.process_supervisor import ensure_supervision_safe
from gpu_fault.admin.resource_registry_dns import network_vpc_identity
from gpu_fault.admin.site import RenderedSite


class NetworkRemovalRequest(Protocol):
    @property
    def site(self) -> RenderedSite: ...


def json_command(arguments: list[str], *, description: str) -> dict[str, Any]:
    result = run_command(arguments)
    if result.returncode:
        raise BootstrapError(f"{description} (exit {result.returncode})")
    try:
        value = json.loads(result.stdout)
    except (ValueError, TypeError) as exc:
        raise BootstrapError(f"{description}: invalid JSON response") from exc
    if not isinstance(value, dict):
        raise BootstrapError(f"{description}: response must be an object")
    return value


def cluster_network(
    runner: CommandRunner,
    *,
    region: str,
    eks_arn: str,
) -> dict[str, Any]:
    arn = Arn.parse(eks_arn)
    if arn.service != "eks" or arn.region != region:
        raise BootstrapError("network discovery requires an EKS ARN in the site Region")
    cluster = runner.aws_json(
        region, "eks", "describe-cluster", "--name", arn.resource_name
    )["cluster"]
    if (
        cluster.get("arn") != eks_arn
        or cluster.get("status") != "ACTIVE"
        or not cluster.get("createdAt")
    ):
        raise BootstrapError("network discovery returned a conflicting EKS identity")
    vpc_id = str(cluster["resourcesVpcConfig"]["vpcId"])
    if not vpc_id:
        raise BootstrapError("network discovery returned an empty VPC identity")
    gateways = runner.aws_json(
        region,
        "ec2",
        "describe-nat-gateways",
        "--filter",
        f"Name=vpc-id,Values={vpc_id}",
        "Name=state,Values=available",
    ).get("NatGateways")
    if not isinstance(gateways, list):
        raise BootstrapError("network discovery returned no NAT gateway inventory")
    eips = sorted(
        {
            str(address["PublicIp"])
            for gateway in gateways
            for address in gateway.get("NatGatewayAddresses", [])
            if address.get("PublicIp")
        }
    )
    return {
        "vpc_id": vpc_id,
        "nat_eips": eips,
        "eks_arn": eks_arn,
        "eks_created_at": str(cluster["createdAt"]),
        "eks_endpoint": str(cluster.get("endpoint") or ""),
    }


def idempotent_aws(
    arguments: list[str],
    *,
    not_found: tuple[str, ...],
    description: str,
) -> bool:
    result = run_command(arguments)
    if result.returncode == 0:
        return True
    if matches_not_found(
        CommandResult(result.stdout or "", result.stderr or "", result.returncode),
        not_found,
    ):
        return False
    raise BootstrapError(f"{description} (exit {result.returncode})")


def wait_vpc_association_absent(
    *,
    hosted_zone_id: str,
    region: str,
    vpc_id: str,
    timeout_seconds: float = 300,
) -> None:
    try:
        with deadline_scope("Route53 VPC disassociation", timeout_seconds) as budget:
            while True:
                ensure_supervision_safe()
                document = json_command(
                    [
                        "aws",
                        "route53",
                        "get-hosted-zone",
                        "--id",
                        hosted_zone_id,
                        "--output",
                        "json",
                    ],
                    description="cannot verify private hosted-zone associations",
                )
                zone = document.get("HostedZone") or {}
                vpcs = document.get("VPCs")
                if (
                    not isinstance(zone, dict)
                    or str(zone.get("Id") or "").removeprefix("/hostedzone/")
                    != hosted_zone_id.removeprefix("/hostedzone/")
                    or not isinstance(vpcs, list)
                    or any(
                        not isinstance(item, dict)
                        or not item.get("VPCRegion")
                        or not item.get("VPCId")
                        for item in vpcs
                    )
                ):
                    raise BootstrapError(
                        "Route53 returned an incomplete hosted-zone identity"
                    )
                associations = {(item["VPCRegion"], item["VPCId"]) for item in vpcs}
                budget.remaining()
                if (region, vpc_id) not in associations:
                    return
                time.sleep(min(5, budget.remaining()))
    except (TimeoutError, subprocess.TimeoutExpired):
        raise BootstrapError(
            "Route53 VPC association removal exceeded its deadline"
        ) from None


def wait_route53_change_insync(
    change_id: str,
    *,
    timeout_seconds: float = 300,
) -> None:
    try:
        with deadline_scope("Route53 change INSYNC", timeout_seconds) as budget:
            while True:
                ensure_supervision_safe()
                document = json_command(
                    [
                        "aws",
                        "route53",
                        "get-change",
                        "--id",
                        change_id,
                        "--output",
                        "json",
                    ],
                    description="cannot read Route53 VPC association change",
                )
                change = document.get("ChangeInfo")
                if (
                    not isinstance(change, dict)
                    or change.get("Id") != change_id
                    or change.get("Status") not in {"PENDING", "INSYNC"}
                ):
                    raise BootstrapError(
                        "Route53 returned a conflicting change identity"
                    )
                budget.remaining()
                if change["Status"] == "INSYNC":
                    return
                time.sleep(min(5, budget.remaining()))
    except (TimeoutError, subprocess.TimeoutExpired):
        raise BootstrapError(
            "Route53 change did not reach INSYNC before its deadline"
        ) from None


def disassociate_vpc_from_hosted_zone(
    *,
    hosted_zone_id: str,
    region: str,
    vpc_id: str,
) -> dict[str, Any]:
    result = run_command(
        [
            "aws",
            "route53",
            "disassociate-vpc-from-hosted-zone",
            "--hosted-zone-id",
            hosted_zone_id,
            "--vpc",
            f"VPCRegion={region},VPCId={vpc_id}",
            "--output",
            "json",
        ]
    )
    if result.returncode:
        if matches_not_found(
            CommandResult(result.stdout or "", result.stderr or "", result.returncode),
            ("VPCAssociationNotFound",),
        ):
            return {"changed": False, "change_id": None, "change_status": None}
        raise BootstrapError(
            f"cannot detach GPU VPC from private hosted zone (exit {result.returncode})"
        )
    try:
        document = json.loads(result.stdout)
    except (TypeError, ValueError) as exc:
        raise BootstrapError(
            "Route53 VPC disassociation returned invalid JSON"
        ) from exc
    change = document.get("ChangeInfo") if isinstance(document, dict) else None
    if not isinstance(change, dict):
        raise BootstrapError("Route53 VPC disassociation returned invalid ChangeInfo")
    change_id = str(change.get("Id") or "")
    status = str(change.get("Status") or "")
    if not change_id or status not in {"PENDING", "INSYNC"}:
        raise BootstrapError("Route53 VPC disassociation returned invalid ChangeInfo")
    if status != "INSYNC":
        wait_route53_change_insync(change_id)
    return {"changed": True, "change_id": change_id, "change_status": "INSYNC"}


def verify_revoked_nat_eips(site: RenderedSite, eips: list[str]) -> None:
    config = site.release_config
    group_id = str(config["nlb"]["security_group"])
    document = json_command(
        [
            "aws",
            "ec2",
            "describe-security-groups",
            "--region",
            str(config["aws_region"]),
            "--group-ids",
            group_id,
            "--output",
            "json",
        ],
        description="cannot verify target NLB ingress removal",
    )
    groups = document.get("SecurityGroups")
    if (
        not isinstance(groups, list)
        or len(groups) != 1
        or not isinstance(groups[0], dict)
        or groups[0].get("GroupId") != group_id
        or not isinstance(groups[0].get("IpPermissions"), list)
    ):
        raise BootstrapError(
            "NLB ingress verification returned an incomplete group identity"
        )
    addresses = {ip_address(eip) for eip in eips}
    for permission in groups[0]["IpPermissions"]:
        if not isinstance(permission, dict) or not isinstance(
            permission.get("IpRanges"), list
        ):
            raise BootstrapError(
                "NLB ingress verification returned invalid permissions"
            )
        protocol = permission.get("IpProtocol")
        if protocol != "-1":
            if protocol not in {"tcp", "6"}:
                continue
            start, end = permission.get("FromPort"), permission.get("ToPort")
            if type(start) is not int or type(end) is not int:
                raise BootstrapError("NLB ingress verification returned invalid ports")
            if not start <= 443 <= end:
                continue
        for source in permission["IpRanges"]:
            try:
                cidr = ip_network(source["CidrIp"])
            except (KeyError, TypeError, ValueError):
                raise BootstrapError(
                    "NLB ingress verification returned invalid CIDRs"
                ) from None
            if any(address in cidr for address in addresses):
                raise BootstrapError(
                    "target NAT ingress remains on the NLB security group"
                )


def detach_network(
    request: NetworkRemovalRequest,
    *,
    target_network: dict[str, Any],
    remaining_networks: list[dict[str, Any]],
    cpu_vpc_id: str,
    detach_dns: bool = False,
) -> dict[str, Any]:
    config = request.site.release_config
    region = str(config["aws_region"])
    target_vpc = network_vpc_identity(target_network, region=region)
    if target_vpc[0] != region:
        raise BootstrapError("target network is outside the site Region")
    remaining_vpcs = {
        network_vpc_identity(item, region=region) for item in remaining_networks
    }
    remaining_eips = {
        eip for item in remaining_networks for eip in item.get("nat_eips", [])
    }
    shared_vpc = target_vpc in remaining_vpcs | {(region, cpu_vpc_id)}
    revoked = (
        [] if shared_vpc else sorted(set(target_network["nat_eips"]) - remaining_eips)
    )
    # Revoke independently: one already absent permission must not suppress the rest.
    for eip in revoked:
        permissions = [
            {
                "IpProtocol": "tcp",
                "FromPort": 443,
                "ToPort": 443,
                "IpRanges": [{"CidrIp": f"{eip}/32"}],
            }
        ]
        idempotent_aws(
            [
                "aws",
                "ec2",
                "revoke-security-group-ingress",
                "--region",
                region,
                "--group-id",
                str(config["nlb"]["security_group"]),
                "--ip-permissions",
                json.dumps(permissions, separators=(",", ":")),
            ],
            not_found=("InvalidPermission.NotFound",),
            description="cannot revoke target NLB ingress permissions",
        )
    if revoked:
        verify_revoked_nat_eips(request.site, revoked)
    detached_vpc = None
    route53_change_id = None
    route53_change_status = None
    if detach_dns and not shared_vpc and config.get("dns", {}).get("hosted_zone_id"):
        change = disassociate_vpc_from_hosted_zone(
            hosted_zone_id=str(config["dns"]["hosted_zone_id"]),
            region=region,
            vpc_id=str(target_network["vpc_id"]),
        )
        wait_vpc_association_absent(
            hosted_zone_id=str(config["dns"]["hosted_zone_id"]),
            region=region,
            vpc_id=str(target_network["vpc_id"]),
        )
        detached_vpc = target_network["vpc_id"]
        route53_change_id = change["change_id"]
        route53_change_status = change["change_status"]
    return {
        "revoked_nat_eips": revoked,
        "detached_vpc_id": detached_vpc,
        "route53_change_id": route53_change_id,
        "route53_change_status": route53_change_status,
    }
