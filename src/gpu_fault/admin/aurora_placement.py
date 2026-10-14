"""Where the Aurora writer runs, measured against where the control plane runs.

Bootstrap creates the writer in the CPU control plane's zone at promotion tier
0 and the reader in another private-subnet zone at tier 1
(``bootstrap_aurora.ensure_serverless_instances`` with ``writer_first_zones``)
and checks the placement once more in ``aurora_ready``. A later failover -- an
RDS host event, a maintenance reboot, an acceptance case -- leaves the writer
wherever RDS put it, and nothing moves it back: live 2026-10-13 HA-010's
injected failover left the writer in us-west-2b under a control plane entirely
in us-west-2a, every database round trip paid ~1.2 ms more and single-cluster
burst admission p99 tripled until an operator failed back by hand.

``status`` reports that as a finding, never a failing check, with the remedy;
``aurora-failback`` (``gpu_fault.admin.aurora_failback``) is the supported way
back. Both read the placement here: the zone labels of the nodes the Running
gpu-fault-api-ha and gpu-fault-control-worker Pods are scheduled on (the
processes that talk to the writer) and every cluster member's zone, status and
promotion tier. One definition serves both, bootstrap's own: the writer belongs
in the first zone of ``writer_first_zones`` -- the zone with the most of those
Pods that holds an Aurora member -- and the member to fail back to is the
available one there, lowest promotion tier first. ``status`` also flags
promotion tiers that do not prefer that member (tier 0 for it, never tier 0 for
another): tiers are set only when an instance is created, nothing re-tiers one.
Reads only, each bounded by ``READ_TIMEOUT_SECONDS``.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from gpu_fault.admin.bootstrap_aurora import NODE_ZONE_LABELS, writer_first_zones
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    CommandRunner,
    ReadOnlyProbeRunner,
)
from gpu_fault.admin.site import RenderedSite

VERB = "aurora-failback"
CONFIRMATION = "AURORA_FAILBACK"
# The Deployments whose processes hold the store connections; the telemetry
# spool tier is scaled to zero wherever spool admission is off.
CONTROL_PLANE_DEPLOYMENTS = ("gpu-fault-api-ha", "gpu-fault-control-worker")
POD_SELECTOR = f"app in ({','.join(CONTROL_PLANE_DEPLOYMENTS)})"
READ_TIMEOUT_SECONDS = 30.0
# Above every RDS promotion tier (0..15): a member without one sorts last.
_NO_TIER = 16
# What a fail-soft read survives: a failed or timed-out command (BootstrapError,
# an expired deadline), output that is not the JSON it should be, a document
# without the expected fields. A blocked test binary is an AssertionError and
# still surfaces.
READ_ERRORS = (RuntimeError, OSError, ValueError, LookupError, TypeError)


@dataclass(frozen=True)
class AuroraMember:
    instance_id: str
    availability_zone: str
    status: str
    promotion_tier: int | None
    writer: bool

    def evidence(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "availability_zone": self.availability_zone,
            "status": self.status,
            "promotion_tier": self.promotion_tier,
            "writer": self.writer,
        }


@dataclass(frozen=True)
class WriterPlacement:
    cluster_id: str
    cluster_status: str
    members: tuple[AuroraMember, ...]
    # Running control-plane Pod -> the zone label of its node.
    control_plane_pods: Mapping[str, str] = field(default_factory=dict)

    @property
    def control_plane_zones(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.control_plane_pods.values())))

    @property
    def pods_by_zone(self) -> dict[str, int]:
        return dict(sorted(Counter(self.control_plane_pods.values()).items()))

    @property
    def preferred_zone(self) -> str | None:
        """Bootstrap's writer zone: the first of ``writer_first_zones`` over the
        members' zones, weighted by Pod; ``None`` when no member is in a zone
        that runs a control-plane Pod."""

        member_zones = sorted(
            {member.availability_zone for member in self.members} - {""}
        )
        pod_zones = list(self.control_plane_pods.values())
        ordered = writer_first_zones(member_zones, pod_zones)
        return ordered[0] if ordered and ordered[0] in set(pod_zones) else None

    @property
    def writer(self) -> AuroraMember | None:
        writers = [member for member in self.members if member.writer]
        return writers[0] if len(writers) == 1 else None

    @property
    def placed(self) -> bool:
        """The writer runs in the preferred zone."""

        writer = self.writer
        zone = self.preferred_zone
        return (
            writer is not None and zone is not None and writer.availability_zone == zone
        )

    def preferred(self) -> AuroraMember | None:
        """The member to fail back to: available, in the preferred zone,
        lowest promotion tier first (bootstrap's writer is tier 0)."""

        candidates = [
            member
            for member in self.members
            if member.status == "available"
            and member.availability_zone == self.preferred_zone
        ]
        return min(
            candidates,
            key=lambda member: (
                _NO_TIER if member.promotion_tier is None else member.promotion_tier,
                member.instance_id,
            ),
            default=None,
        )

    def tier_problems(self) -> tuple[AuroraMember | None, list[str]]:
        """The member tiers should prefer -- the writer when it is placed, else
        the failback target -- and why they do not: it is not tier 0, or
        another member is. After a failover the writer is the other member, so
        the writer alone would flag every correctly tiered site."""

        reference = self.writer if self.placed else self.preferred()
        if reference is None:
            return None, []
        problems: list[str] = []
        if reference.promotion_tier != 0:
            problems.append(f"{_member_label(reference)} is {_tier(reference)}")
        problems.extend(
            f"{_member_label(member)} is promotion tier 0"
            for member in self.members
            if member is not reference and member.promotion_tier == 0
        )
        return reference, problems

    def evidence(self) -> dict[str, Any]:
        writer = self.writer
        preferred = self.preferred()
        return {
            "cluster_id": self.cluster_id,
            "cluster_status": self.cluster_status,
            "writer": writer.instance_id if writer is not None else None,
            "writer_zone": writer.availability_zone if writer is not None else None,
            "preferred_zone": self.preferred_zone,
            "control_plane_zones": list(self.control_plane_zones),
            "control_plane_pods_by_zone": self.pods_by_zone,
            "control_plane_pods": dict(sorted(self.control_plane_pods.items())),
            "placed": self.placed,
            "preferred": preferred.instance_id if preferred is not None else None,
            "members": [member.evidence() for member in self.members],
        }


def _member_label(member: AuroraMember) -> str:
    return f"{member.instance_id} ({'writer' if member.writer else 'reader'})"


def _tier(member: AuroraMember) -> str:
    if member.promotion_tier is None:
        return "of unknown promotion tier"
    return f"promotion tier {member.promotion_tier}"


def _json_object(raw: str, what: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise BootstrapError(f"{what} is not a JSON object")
    return value


def control_plane_pod_zones(
    runner: CommandRunner, *, kubeconfig: str, namespace: str
) -> dict[str, str]:
    """Every Running api-ha or control-worker Pod and its node's zone."""

    kubectl = ["kubectl", "--kubeconfig", kubeconfig]
    pods = _json_object(
        runner.run(
            [
                *kubectl,
                "-n",
                namespace,
                "get",
                "pod",
                "-l",
                POD_SELECTOR,
                "--field-selector=status.phase=Running",
                "-o",
                "json",
            ],
            timeout_seconds=READ_TIMEOUT_SECONDS,
        ),
        "control-plane Pod list",
    )
    placed_on = {
        str((item.get("metadata") or {}).get("name") or ""): str(
            (item.get("spec") or {}).get("nodeName") or ""
        )
        for item in pods.get("items") or []
    }
    placed_on = {pod: node for pod, node in placed_on.items() if pod and node}
    nodes = sorted(set(placed_on.values()))
    if not nodes:
        raise BootstrapError(
            "no Running " + " or ".join(CONTROL_PLANE_DEPLOYMENTS) + " Pod is on a node"
        )
    document = _json_object(
        runner.run(
            [*kubectl, "get", "node", *nodes, "-o", "json"],
            timeout_seconds=READ_TIMEOUT_SECONDS,
        ),
        "control-plane node list",
    )
    # One name answers with the Node itself, several with a List.
    items = document["items"] if "items" in document else [document]
    zones: dict[str, str] = {}
    for item in items:
        metadata = item.get("metadata") or {}
        labels = metadata.get("labels") or {}
        zone = next((labels[key] for key in NODE_ZONE_LABELS if labels.get(key)), "")
        if zone:
            zones[str(metadata.get("name") or "")] = str(zone)
    missing = [node for node in nodes if node not in zones]
    if missing:
        raise BootstrapError(
            f"control-plane node(s) {', '.join(missing)} carry no zone label"
        )
    return {pod: zones[node] for pod, node in sorted(placed_on.items())}


def aurora_members(
    runner: CommandRunner, *, aws_region: str, cluster_id: str
) -> tuple[str, tuple[AuroraMember, ...]]:
    """The cluster's status and every member's zone, status, tier and role."""

    def aws(*arguments: str) -> dict[str, Any]:
        return _json_object(
            runner.run(
                [
                    "aws",
                    "rds",
                    *arguments,
                    "--region",
                    aws_region,
                    "--output",
                    "json",
                ],
                timeout_seconds=READ_TIMEOUT_SECONDS,
            ),
            f"aws rds {arguments[0]}",
        )

    clusters = aws("describe-db-clusters", "--db-cluster-identifier", cluster_id).get(
        "DBClusters"
    )
    if (
        not isinstance(clusters, list)
        or len(clusters) != 1
        or clusters[0].get("DBClusterIdentifier") != cluster_id
    ):
        raise BootstrapError(f"cannot describe Aurora cluster {cluster_id}")
    instances = {
        str(item.get("DBInstanceIdentifier") or ""): item
        for item in aws(
            "describe-db-instances",
            "--filters",
            f"Name=db-cluster-id,Values={cluster_id}",
        ).get("DBInstances")
        or []
    }
    members: list[AuroraMember] = []
    for item in clusters[0].get("DBClusterMembers") or []:
        instance_id = str(item.get("DBInstanceIdentifier") or "")
        instance = instances.get(instance_id) or {}
        tier = item.get("PromotionTier", instance.get("PromotionTier"))
        members.append(
            AuroraMember(
                instance_id=instance_id,
                availability_zone=str(instance.get("AvailabilityZone") or ""),
                status=str(instance.get("DBInstanceStatus") or ""),
                promotion_tier=(
                    tier
                    if isinstance(tier, int) and not isinstance(tier, bool)
                    else None
                ),
                writer=item.get("IsClusterWriter") is True,
            )
        )
    return str(clusters[0].get("Status") or ""), tuple(
        sorted(members, key=lambda member: member.instance_id)
    )


def read_writer_placement(site: RenderedSite, runner: CommandRunner) -> WriterPlacement:
    config = site.release_config
    cluster_id = str(config["health"]["aurora_cluster_id"])
    pods = control_plane_pod_zones(
        runner,
        kubeconfig=str(config["cpu_kubeconfig"]),
        namespace=str(config["namespace"]),
    )
    status, members = aurora_members(
        runner, aws_region=str(config["aws_region"]), cluster_id=cluster_id
    )
    return WriterPlacement(cluster_id, status, members, pods)


def placement_problem(placement: WriterPlacement) -> str | None:
    """Why the placement cannot be judged, or ``None`` when it can."""

    writer = placement.writer
    writers = sum(1 for member in placement.members if member.writer)
    if writer is None:
        return (
            f"Aurora cluster {placement.cluster_id} reports {writers} writers "
            f"(status {placement.cluster_status or 'unknown'})"
        )
    if not writer.availability_zone:
        return f"Aurora writer {writer.instance_id} reports no availability zone"
    return None


def failback_commands(state_dir: Path) -> str:
    command = f"gpu-fault-admin {VERB} --state-dir {state_dir} --reference <change>"
    return f"{command} (prints the plan), then the same with --confirm {CONFIRMATION}"


def placement_finding(placement: WriterPlacement, *, state_dir: Path) -> str | None:
    """The ``status`` placement line, or ``None`` when the writer is placed."""

    problem = placement_problem(placement)
    if problem is not None:
        return f"Aurora writer placement could not be read: {problem}"
    if placement.placed:
        return None
    writer = placement.writer
    assert writer is not None  # placement_problem rules a missing writer out
    deployments = ", ".join(CONTROL_PLANE_DEPLOYMENTS)
    zone = placement.preferred_zone
    if zone is None:
        return (
            f"Aurora writer {writer.instance_id} runs in {writer.availability_zone} "
            f"and no Aurora member is in a zone running the CPU control plane "
            f"({deployments}: {', '.join(placement.control_plane_zones)}): every "
            "database round trip crosses zones and there is no member to fail back to"
        )
    pods = placement.pods_by_zone
    line = (
        f"Aurora writer {writer.instance_id} runs in {writer.availability_zone}, not "
        f"in {zone} where most CPU control-plane Pods run ({deployments}: "
        f"{pods[zone]} of {sum(pods.values())}): their database round trips cross zones"
    )
    preferred = placement.preferred()
    if preferred is None:
        return (
            f"{line}, and no available Aurora member is in {zone} to fail back "
            f"to; once one is, run {failback_commands(state_dir)}"
        )
    return (
        f"{line}; fail it back to {preferred.instance_id} ({zone}, {_tier(preferred)}) "
        f"with {failback_commands(state_dir)}"
    )


def tier_finding(placement: WriterPlacement) -> str | None:
    """The ``status`` promotion-tier line, or ``None`` when tiers prefer the
    member that should write."""

    reference, problems = placement.tier_problems()
    if reference is None or not problems:
        return None
    return (
        f"Aurora promotion tiers do not prefer {reference.instance_id} in "
        f"{reference.availability_zone}: {'; '.join(problems)}; tiers are set only "
        "when an instance is created (bootstrap: tier 0 for the writer in the "
        "control plane's zone, tier 1 for the reader) and nothing changes them "
        "afterwards"
    )


def placement_findings(
    site: RenderedSite,
    *,
    state_dir: Path | None = None,
    runner: CommandRunner | None = None,
) -> list[str]:
    """``status``'s placement and tier lines; an unreadable placement is itself
    a line.

    Read-only by construction: the runner refuses any mutating command.
    """

    try:
        placement = read_writer_placement(
            site, ReadOnlyProbeRunner(runner or CommandRunner())
        )
    except READ_ERRORS as exc:
        return [f"Aurora writer placement could not be read: {exc}"]
    lines = [
        placement_finding(placement, state_dir=state_dir or site.source.parent),
        None if placement_problem(placement) else tier_finding(placement),
    ]
    return [line for line in lines if line is not None]
