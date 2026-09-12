"""Network prerequisites of a join and their undo, moved out of ``cluster_join``.

The NLB security-group ingress for the joined cluster's NAT EIPs and the private
hosted zone's VPC association: created by ``_ensure_network``, undone by
``_rollback`` through :func:`rollback_network`.
"""

from __future__ import annotations

import subprocess
import time
from typing import Any

from gpu_fault.admin.bootstrap_common import BootstrapError, CommandRunner
from gpu_fault.admin.cluster_join_rollback import rollback_command
from gpu_fault.admin.cluster_removal import _wait_vpc_association_absent
from gpu_fault.admin.site import RenderedSite


def ensure_nlb_ingress(
    *,
    region: str,
    security_group: str,
    eips: list[str],
) -> list[str]:
    created = []
    for eip in eips:
        result = subprocess.run(
            [
                "aws",
                "ec2",
                "authorize-security-group-ingress",
                "--region",
                region,
                "--group-id",
                security_group,
                "--protocol",
                "tcp",
                "--port",
                "443",
                "--cidr",
                f"{eip}/32",
            ],
            text=True,
            capture_output=True,
        )
        if result.returncode == 0:
            created.append(eip)
        elif "InvalidPermission.Duplicate" not in result.stderr:
            raise BootstrapError(
                f"cannot authorize NLB ingress for {eip}: {result.stderr.strip()}"
            )
    return created


def wait_zone_association(
    runner: CommandRunner,
    *,
    region: str,
    hosted_zone_id: str,
    vpc_id: str,
) -> None:
    for _ in range(60):
        document = runner.aws_json(
            region,
            "route53",
            "get-hosted-zone",
            "--id",
            hosted_zone_id,
        )
        if any(
            item.get("VPCRegion") == region and item.get("VPCId") == vpc_id
            for item in document.get("VPCs", [])
        ):
            return
        time.sleep(5)
    raise BootstrapError("Route53 VPC association did not become visible")


def rollback_network(network: dict[str, Any], site: RenderedSite) -> None:
    region = str(site.release_config["aws_region"])
    for eip in network.get("created_ingress_eips", []):
        rollback_command(
            [
                "aws",
                "ec2",
                "revoke-security-group-ingress",
                "--region",
                region,
                "--group-id",
                str(site.release_config["nlb"]["security_group"]),
                "--protocol",
                "tcp",
                "--port",
                "443",
                "--cidr",
                f"{eip}/32",
            ],
            not_found=("InvalidPermission.NotFound",),
        )
    if network.get("association_created"):
        rollback_command(
            [
                "aws",
                "route53",
                "disassociate-vpc-from-hosted-zone",
                "--hosted-zone-id",
                str(network["hosted_zone_id"]),
                "--vpc",
                f"VPCRegion={region},VPCId={network['vpc_id']}",
            ],
            not_found=("VPCAssociationNotFound",),
        )
        _wait_vpc_association_absent(
            hosted_zone_id=str(network["hosted_zone_id"]),
            region=region,
            vpc_id=str(network["vpc_id"]),
        )
