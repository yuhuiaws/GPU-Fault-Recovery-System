"""Supervised network preparation and journal-aware compensation for joins."""

from __future__ import annotations

from typing import Any

from gpu_fault.admin.aws_commands import CommandResult, matches_not_found, wait_until
from gpu_fault.admin.bootstrap_common import BootstrapError, CommandRunner
from gpu_fault.admin.diagnostics import diagnostic_text
from gpu_fault.admin.site import RenderedSite


def ensure_nlb_ingress(
    *, region: str, security_group: str, eips: list[str]
) -> list[str]:
    from gpu_fault.admin import cluster_join as join

    created = []
    for eip in eips:
        result = join.run_command(
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
        )
        if result.returncode == 0:
            created.append(eip)
        elif not matches_not_found(
            CommandResult(result.stdout or "", result.stderr or "", result.returncode),
            ("InvalidPermission.Duplicate",),
        ):
            raise BootstrapError(
                f"cannot authorize NLB ingress for {eip}: "
                + diagnostic_text(result.stderr.strip())
            )
    return created


def wait_zone_association(
    runner: CommandRunner, *, region: str, hosted_zone_id: str, vpc_id: str
) -> None:
    def associated() -> bool:
        document = runner.aws_json(
            region, "route53", "get-hosted-zone", "--id", hosted_zone_id
        )
        return any(
            item.get("VPCRegion") == region and item.get("VPCId") == vpc_id
            for item in document.get("VPCs", [])
        )

    wait_until(
        associated,
        description="Route53 VPC association",
        timeout_seconds=300,
        interval_seconds=5,
    )


def rollback_network(network: dict[str, Any], site: RenderedSite) -> None:
    from gpu_fault.admin.cluster_join_rollback import _network

    _network(network, site)
