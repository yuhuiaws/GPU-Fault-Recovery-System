"""Reading the CPU VPC's network: subnets and the route tables they use.

Bootstrap classifies subnets twice -- the NLB task wants the public ones, the
Aurora task the private ones -- and both decisions come down to "does this
subnet's route table send ``0.0.0.0/0`` to an internet gateway?". EC2 answers
that per subnet only with two calls (the explicit association, then the VPC main
table as the fallback), so a VPC with N subnets cost 2N ``describe-route-tables``
per task. A VPC's route tables fit in one page: ``RouteTableIndex`` loads them
once per VPC and answers every question from memory.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, cast

from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    ClusterIdentity,
    CommandRunner,
)


def describe_subnets(
    runner: CommandRunner,
    *,
    region: str,
    subnet_ids: Sequence[str],
) -> list[dict[str, Any]]:
    if not subnet_ids:
        return []
    return cast(
        list[dict[str, Any]],
        runner.aws_json(
            region,
            "ec2",
            "describe-subnets",
            "--subnet-ids",
            *subnet_ids,
        ).get("Subnets", []),
    )


class RouteTableIndex:
    """Every route table of one VPC, from one ``describe-route-tables`` call."""

    def __init__(self, tables: Sequence[Mapping[str, Any]]) -> None:
        self.tables = [dict(table) for table in tables]
        self._explicit: dict[str, list[dict[str, Any]]] = {}
        self._main: list[dict[str, Any]] = []
        for table in self.tables:
            for association in table.get("Associations", []):
                subnet_id = association.get("SubnetId")
                if subnet_id:
                    self._explicit.setdefault(str(subnet_id), []).append(table)
                if association.get("Main"):
                    self._main.append(table)

    @classmethod
    def load(
        cls, runner: CommandRunner, *, region: str, vpc_id: str
    ) -> RouteTableIndex:
        return cls(
            runner.aws_json(
                region,
                "ec2",
                "describe-route-tables",
                "--filters",
                f"Name=vpc-id,Values={vpc_id}",
            ).get("RouteTables", [])
        )

    def explicit_for(self, subnet_id: str) -> dict[str, Any] | None:
        """The table explicitly associated with a subnet, if any -- the VPC main
        table does not count: an owned public subnet must have its own."""

        tables = self._explicit.get(subnet_id)
        return tables[0] if tables else None

    def for_subnet(self, subnet_id: str) -> list[dict[str, Any]]:
        """The tables a subnet routes through: its own, else the VPC main table."""

        return self._explicit.get(subnet_id) or list(self._main)

    def is_public(self, subnet: Mapping[str, Any]) -> bool:
        return any(
            route.get("DestinationCidrBlock") == "0.0.0.0/0"
            and str(route.get("GatewayId", "")).startswith("igw-")
            for table in self.for_subnet(str(subnet["SubnetId"]))
            for route in table.get("Routes", [])
        )


def private_subnets(
    runner: CommandRunner,
    cluster: ClusterIdentity,
) -> list[dict[str, Any]]:
    """Two of the control-plane cluster's private subnets, one per AZ (Aurora)."""

    all_subnets = describe_subnets(
        runner,
        region=cluster.region,
        subnet_ids=cluster.subnet_ids,
    )
    routes = RouteTableIndex.load(runner, region=cluster.region, vpc_id=cluster.vpc_id)
    selected: dict[str, dict[str, Any]] = {}
    for item in all_subnets:
        if not routes.is_public(item):
            selected.setdefault(item["AvailabilityZone"], item)
    if len(selected) < 2:
        raise BootstrapError("CPU EKS requires private subnets in at least two AZs")
    return list(selected.values())[:2]
