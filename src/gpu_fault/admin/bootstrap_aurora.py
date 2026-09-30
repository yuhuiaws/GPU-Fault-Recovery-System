from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence, cast

from gpu_fault.admin.aurora_capacity import (
    CAPACITY_SETTLE_STABLE_POLLS,
    reconcile_aurora_capacity,
)
from gpu_fault.admin.bootstrap_common import (
    SITE_TAG_KEY,
    BootstrapError,
    BootstrapMutationRequired,
    CommandRunner,
    assert_site_tag,
    describe_or_absent,
)
from gpu_fault.admin.bootstrap_platform_probes import kubectl_projection
from gpu_fault.admin.config import (
    AdminConfigError,
    AuroraCapacityConfig,
    load_desired_admin_config,
)
from gpu_fault.admin.execution import command_timeout, deadline_scope

CAPACITY_SETTLE_POLL_SECONDS = 10.0
CAPACITY_SETTLE_TIMEOUT_SECONDS = 1800.0
# Both RDS availability waiters poll every 30s for up to 60 attempts. Allow
# five more minutes for CLI/API overhead; run_command still caps this at the
# caller's remaining task/deployment deadline.
RDS_AVAILABILITY_TIMEOUT_SECONDS = 2100.0
PRIMARY_MEMBERSHIP_ATTEMPTS = 6
PRIMARY_MEMBERSHIP_POLL_SECONDS = 5.0
# The Kubernetes Secret the control plane reads its DSN from.
AURORA_SECRET_NAME = "gpu-fault-aurora"
# RDS reports the managed master secret's ARN before Secrets Manager can serve
# its value (the secret is created with the cluster; its AWSCURRENT version
# lands a little later). Once the instances are available the value is nearly
# always there, so this bound is for the lag, not for a broken account: eighteen
# reads ten seconds apart, three minutes, then a hard failure.
MASTER_SECRET_READ_ATTEMPTS = 18
MASTER_SECRET_READ_POLL_SECONDS = 10.0

# What the control plane needs the database to record about itself (store
# review 2026-09-07, item K). The store retries deadlocks (40P01) silently and
# the default Aurora parameter group logs neither lock waits nor slow
# statements, so a locking defect is invisible in production; pg_stat_statements
# is the only per-statement view of where the ACU budget goes. Three of these
# are dynamic and take effect as soon as the group is attached; the preload
# library is static and waits for a writer reboot the operator schedules
# (`管理员日常运维.md` §4.6) -- until then the cluster reads ``pending-reboot``,
# which is expected, not a defect.
DIAGNOSTIC_PARAMETERS: tuple[tuple[str, str, str], ...] = (
    ("log_lock_waits", "1", "immediate"),
    ("log_min_duration_statement", "1000", "immediate"),
    ("pg_stat_statements.track", "all", "immediate"),
    ("shared_preload_libraries", "pg_stat_statements", "pending-reboot"),
)
POSTGRESQL_LOG_EXPORT = "postgresql"


def bootstrap_aurora_capacity(state_dir: Path) -> AuroraCapacityConfig:
    return load_desired_admin_config(
        state_dir,
        migrate_legacy=True,
    ).aurora


def ensure_rds_site_tag(
    runner: CommandRunner,
    *,
    region: str,
    resource_arn: str,
    tags: object,
    site_id: str,
    description: str,
) -> None:
    tagged = assert_site_tag(
        tags,
        site_id=site_id,
        description=description,
        allow_missing=True,
    )
    if tagged:
        return
    runner.run(
        [
            "aws",
            "rds",
            "add-tags-to-resource",
            "--region",
            region,
            "--resource-name",
            resource_arn,
            "--tags",
            f"Key={SITE_TAG_KEY},Value={site_id}",
        ],
        mutate=True,
        capture=False,
    )


def ensure_subnet_group(
    runner: CommandRunner,
    *,
    aws_region: str,
    name: str,
    subnet_ids: Sequence[str],
    site_id: str,
) -> None:
    """Create the DB subnet group over the private subnets, or adopt the site's."""

    subnet_groups = describe_or_absent(
        runner,
        aws_region,
        "rds",
        "describe-db-subnet-groups",
        "--db-subnet-group-name",
        name,
        not_found=("DBSubnetGroupNotFoundFault",),
    )
    if subnet_groups is None:
        runner.run(
            [
                "aws",
                "rds",
                "create-db-subnet-group",
                "--region",
                aws_region,
                "--db-subnet-group-name",
                name,
                "--db-subnet-group-description",
                "GPU fault regional Aurora",
                "--subnet-ids",
                *subnet_ids,
                "--tags",
                f"Key={SITE_TAG_KEY},Value={site_id}",
            ],
            mutate=True,
            capture=False,
        )
        return
    details = subnet_groups["DBSubnetGroups"][0]
    ensure_rds_site_tag(
        runner,
        region=aws_region,
        resource_arn=str(details["DBSubnetGroupArn"]),
        tags=runner.aws_json(
            aws_region,
            "rds",
            "list-tags-for-resource",
            "--resource-name",
            str(details["DBSubnetGroupArn"]),
        ).get("TagList"),
        site_id=site_id,
        description=f"RDS subnet group {name}",
    )
    # The modify is a mutation even when nothing changes (and RDS answers it
    # slowly); a group that already holds these subnets is left alone.
    current_subnet_ids = sorted(
        str(item.get("SubnetIdentifier") or "") for item in details.get("Subnets") or []
    )
    if current_subnet_ids != sorted(subnet_ids):
        runner.run(
            [
                "aws",
                "rds",
                "modify-db-subnet-group",
                "--region",
                aws_region,
                "--db-subnet-group-name",
                name,
                "--subnet-ids",
                *subnet_ids,
            ],
            mutate=True,
            capture=False,
        )


def reconcile_existing_capacity(
    runner: CommandRunner,
    *,
    aws_region: str,
    cluster_id: str,
    cluster: dict[str, Any],
    capacity: AuroraCapacityConfig,
    instance_ids: Sequence[str] = (),
) -> None:
    """Bring an existing cluster to the administrator's window.

    The modify itself and the settle rule live in ``aurora_capacity`` -- the one
    writer shared with ``config apply`` -- and go through the runner so a
    ``--dry-run`` holds the mutation back. The caller has just described the
    cluster; a window that already matches costs no further call. Both member
    instances must be available: the reconciler proves the change on them, so
    when ``instance_ids`` are given the ones still creating are waited for
    first -- only on a drifted window, which is the one case that still needs
    the instances inside the foundation task.
    """

    scaling = cluster.get("ServerlessV2ScalingConfiguration") or {}
    live_capacity = (
        float(scaling.get("MinCapacity") or 0),
        float(scaling.get("MaxCapacity") or 0),
    )
    if live_capacity == (capacity.min_acu, capacity.max_acu):
        return
    if instance_ids:
        pending = pending_serverless_instances(
            runner,
            aws_region=aws_region,
            cluster_id=cluster_id,
            instance_ids=instance_ids,
        )
        await_serverless_instances(runner, aws_region=aws_region, instance_ids=pending)
    try:
        reconcile_aurora_capacity(
            aws_region=aws_region,
            cluster_id=cluster_id,
            desired=capacity,
            timeout_seconds=int(CAPACITY_SETTLE_TIMEOUT_SECONDS),
            poll_seconds=CAPACITY_SETTLE_POLL_SECONDS,
            aws_json=runner.aws_json,
        )
    except AdminConfigError as exc:
        raise BootstrapError(f"Aurora cluster {cluster_id}: {exc}") from exc


def wait_for_capacity_to_settle(
    runner: CommandRunner,
    *,
    aws_region: str,
    cluster_id: str,
    poll_seconds: float = CAPACITY_SETTLE_POLL_SECONDS,
    timeout_seconds: float = CAPACITY_SETTLE_TIMEOUT_SECONDS,
) -> None:
    """Block until the cluster and every member instance read ``available``.

    Used after the diagnostics modify (parameter group, log export). The
    capacity change has its own, stricter settle in ``aurora_capacity`` -- it
    also proves the window and the observed ACU -- but the rule is the same:
    ``modify-db-cluster --apply-immediately`` answers before RDS moves the
    cluster, so neither ``rds wait db-cluster-available`` nor a configuration
    comparison can tell "settled" from "not started yet". Require several
    consecutive quiet polls instead, so a late flip is still caught.
    """

    deadline = time.monotonic() + timeout_seconds
    stable = 0
    while True:
        quiet = _capacity_is_quiet(
            runner,
            aws_region=aws_region,
            cluster_id=cluster_id,
        )
        stable = stable + 1 if quiet else 0
        if stable >= CAPACITY_SETTLE_STABLE_POLLS:
            return
        if time.monotonic() >= deadline:
            raise BootstrapError(
                f"Aurora cluster {cluster_id} did not settle within "
                f"{timeout_seconds:g}s of the capacity change"
            )
        time.sleep(poll_seconds)


def _capacity_is_quiet(
    runner: CommandRunner,
    *,
    aws_region: str,
    cluster_id: str,
) -> bool:
    cluster = runner.aws_json(
        aws_region,
        "rds",
        "describe-db-clusters",
        "--db-cluster-identifier",
        cluster_id,
    )["DBClusters"][0]
    if str(cluster.get("Status") or "") != "available":
        return False
    instances = (
        runner.aws_json(
            aws_region,
            "rds",
            "describe-db-instances",
            "--filters",
            f"Name=db-cluster-id,Values={cluster_id}",
        ).get("DBInstances")
        or []
    )
    return all(
        str(instance.get("DBInstanceStatus") or "") == "available"
        for instance in instances
    )


def _await_replica_source(
    runner: CommandRunner,
    *,
    aws_region: str,
    cluster_id: str,
    instance_id: str,
    initial_writer_id: str,
) -> None:
    """RDS requires an available cluster and primary before adding a replica."""

    def cluster_state() -> dict[str, Any]:
        clusters = runner.aws_json(
            aws_region,
            "rds",
            "describe-db-clusters",
            "--db-cluster-identifier",
            cluster_id,
        ).get("DBClusters")
        if (
            not isinstance(clusters, list)
            or len(clusters) != 1
            or clusters[0].get("DBClusterIdentifier") != cluster_id
            or not isinstance(clusters[0].get("DBClusterMembers"), list)
        ):
            raise BootstrapError("cannot determine Aurora replica source cluster")
        return dict(clusters[0])

    def initial_primary_pending(value: dict[str, Any]) -> bool:
        members = value["DBClusterMembers"]
        return value.get("Status") in {"creating", "available"} and (
            not members
            or len(members) == 1
            and members[0].get("DBInstanceIdentifier") == initial_writer_id
            and members[0].get("IsClusterWriter") is False
        )

    cluster = cluster_state()
    if initial_primary_pending(cluster):
        if not cluster["DBClusterMembers"] and instance_id == initial_writer_id:
            return
        # RDS can list the first instance before designating it as the writer.
        # An available instance without a primary is not creation progress.
        if (
            cluster["DBClusterMembers"]
            and serverless_instance_statuses(
                runner, aws_region=aws_region, cluster_id=cluster_id
            ).get(initial_writer_id)
            != "creating"
        ):
            raise BootstrapError("cannot determine Aurora replica source primary")
        await_serverless_instances(
            runner, aws_region=aws_region, instance_ids=[initial_writer_id]
        )
        for attempt in range(PRIMARY_MEMBERSHIP_ATTEMPTS):
            cluster = cluster_state()
            if not initial_primary_pending(cluster):
                break
            if attempt + 1 == PRIMARY_MEMBERSHIP_ATTEMPTS:
                raise BootstrapError("cannot determine Aurora replica source primary")
            time.sleep(
                command_timeout(
                    ["aws", "rds", "describe-db-clusters"],
                    PRIMARY_MEMBERSHIP_POLL_SECONDS,
                )
            )
    writers = [
        member.get("DBInstanceIdentifier")
        for member in cluster["DBClusterMembers"]
        if member.get("IsClusterWriter") is True
    ]
    if len(writers) != 1 or not isinstance(writers[0], str) or not writers[0]:
        raise BootstrapError("cannot determine Aurora replica source primary")
    writer_id = writers[0]
    pending = pending_serverless_instances(
        runner, aws_region=aws_region, cluster_id=cluster_id, instance_ids=[writer_id]
    )
    await_serverless_instances(runner, aws_region=aws_region, instance_ids=pending)
    if cluster.get("Status") != "available":
        _await_rds_available(
            runner,
            aws_region=aws_region,
            kind="cluster",
            identifier=cluster_id,
        )
    cluster = cluster_state()
    current_writers = [
        member.get("DBInstanceIdentifier")
        for member in cluster["DBClusterMembers"]
        if member.get("IsClusterWriter") is True
    ]
    if (
        cluster.get("Status") != "available"
        or current_writers != [writer_id]
        or pending_serverless_instances(
            runner,
            aws_region=aws_region,
            cluster_id=cluster_id,
            instance_ids=[writer_id],
        )
    ):
        raise BootstrapError("Aurora replica source changed or is not available")


def ensure_serverless_instances(
    runner: CommandRunner,
    *,
    aws_region: str,
    cluster_id: str,
    availability_zones: Sequence[str],
    safe_name: Callable[..., str],
    wait: bool = True,
    site_id: str | None = None,
) -> list[str]:
    """Create the writer and the reader that are missing; returns both ids.

    ``wait=False`` defers final readiness to ``aurora_ready``, but never skips
    the available-primary/cluster barrier before a replica create. The whole
    Aurora chain runs alongside tasks that need no database.

    With ``site_id`` every instance carries the site tag: new instances are
    created with it and existing untagged ones receive it. The uninstall's
    ownership proof reads that tag on each instance, not only on the cluster
    (live 2026-09-20: two untagged instances stopped an uninstall at Aurora).
    """

    instance_ids = [
        safe_name(f"{cluster_id}-{suffix}", maximum=63)
        for suffix in ("writer", "reader")
    ]
    placements = list(zip(instance_ids, availability_zones, strict=True))
    for instance_id, availability_zone in placements:
        existing = describe_or_absent(
            runner,
            aws_region,
            "rds",
            "describe-db-instances",
            "--db-instance-identifier",
            instance_id,
            not_found=("DBInstanceNotFound",),
        )
        if existing is not None:
            if site_id is not None:
                _ensure_instance_site_tag(
                    runner,
                    aws_region=aws_region,
                    document=existing,
                    instance_id=instance_id,
                    site_id=site_id,
                )
            continue
        _await_replica_source(
            runner,
            aws_region=aws_region,
            cluster_id=cluster_id,
            instance_id=instance_id,
            initial_writer_id=instance_ids[0],
        )
        runner.run(
            [
                "aws",
                "rds",
                "create-db-instance",
                "--region",
                aws_region,
                "--db-instance-identifier",
                instance_id,
                "--db-cluster-identifier",
                cluster_id,
                "--engine",
                "aurora-postgresql",
                "--db-instance-class",
                "db.serverless",
                "--availability-zone",
                availability_zone,
                # The writer is tier 0; the reader tier 1, so an RDS-initiated
                # failover prefers the instance that was placed as the writer
                # (the control-plane zone, see ``writer_first_zones``).
                "--promotion-tier",
                "0" if instance_id == instance_ids[0] else "1",
                *(
                    ["--tags", f"Key={SITE_TAG_KEY},Value={site_id}"]
                    if site_id is not None
                    else []
                ),
            ],
            mutate=True,
            capture=False,
        )
    if wait:
        await_serverless_instances(
            runner, aws_region=aws_region, instance_ids=instance_ids
        )
    return instance_ids


NODE_ZONE_LABELS = (
    "topology.kubernetes.io/zone",
    "failure-domain.beta.kubernetes.io/zone",
)
WRITER_ZONE_FAILOVER_TIMEOUT_SECONDS = 600.0


def control_plane_node_zones(node_document: Mapping[str, Any]) -> list[str]:
    """One zone per CPU node from a ``kubectl get nodes -o json`` document."""

    zones: list[str] = []
    for item in node_document.get("items") or ():
        labels = (item.get("metadata") or {}).get("labels") or {}
        for key in NODE_ZONE_LABELS:
            zone = labels.get(key)
            if isinstance(zone, str) and zone:
                zones.append(zone)
                break
    return zones


def cpu_nodes_document(
    runner: CommandRunner, *, kubeconfig: Path, hyperpod_name: str
) -> dict[str, Any]:
    """The CPU HyperPod nodes as kubectl lists them (addresses and zone labels)."""

    return dict(
        json.loads(
            runner.run(
                [
                    "kubectl",
                    "--kubeconfig",
                    str(kubeconfig),
                    "get",
                    "nodes",
                    "-l",
                    f"sagemaker.amazonaws.com/cluster-name={hyperpod_name}",
                    "-o",
                    "json",
                ]
            )
        )
    )


def cpu_node_security_groups(
    runner: CommandRunner, *, aws_region: str, document: Mapping[str, Any]
) -> list[str]:
    """Security Groups of the CPU nodes' ENIs (the Aurora ingress sources)."""

    ips = sorted(
        {
            address["address"]
            for item in document.get("items", [])
            for address in item.get("status", {}).get("addresses", [])
            if address.get("type") == "InternalIP"
        }
    )
    if not ips:
        raise BootstrapError("CPU HyperPod has no node InternalIP")
    interfaces = cast(
        list[dict[str, Any]],
        runner.aws_json(
            aws_region,
            "ec2",
            "describe-network-interfaces",
            "--filters",
            "Name=addresses.private-ip-address,Values=" + ",".join(ips),
        ).get("NetworkInterfaces", []),
    )
    groups = sorted(
        {
            group["GroupId"]
            for interface in interfaces
            for group in interface.get("Groups", [])
        }
    )
    if not groups:
        raise BootstrapError("cannot discover CPU node Security Groups")
    return groups


def reconcile_writer_zone(
    runner: CommandRunner,
    *,
    aws_region: str,
    cluster_id: str,
    instance_ids: Sequence[str],
    control_plane_zones: Sequence[str],
    kubeconfig: Path,
    hyperpod_name: str,
) -> dict[str, Any]:
    """``ensure_writer_zone_affinity`` for ``aurora_ready``: a checkpoint written
    before the zones were recorded reads them from the CPU nodes now."""

    zones = [str(item) for item in control_plane_zones]
    if not zones:
        zones = control_plane_node_zones(
            cpu_nodes_document(
                runner, kubeconfig=kubeconfig, hyperpod_name=hyperpod_name
            )
        )
    return ensure_writer_zone_affinity(
        runner,
        aws_region=aws_region,
        cluster_id=cluster_id,
        instance_ids=instance_ids,
        control_plane_zones=zones,
    )


def writer_first_zones(
    availability_zones: Sequence[str], control_plane_zones: Sequence[str]
) -> list[str]:
    """Order the Aurora zones so the writer lands where the control plane runs.

    The initial writer is the first instance created, so the zone at index 0
    decides where every database round-trip of the API Pods terminates. A
    writer in another zone than the CPU nodes costs a cross-zone hop per query;
    the 2026-09-27 fresh bootstrap put the writer in us-west-2b under a
    control plane entirely in us-west-2a and the §8.4 50-cluster load then
    returned 3x the reserved 503s (2.1-2.6 % against 0.7 %) until the writer
    was failed over. The most common CPU node zone that also has a private
    subnet goes first; without a match the order is left alone.
    """

    ordered = list(availability_zones)
    counts: dict[str, int] = {}
    for zone in control_plane_zones:
        counts[zone] = counts.get(zone, 0) + 1
    for zone, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if zone in ordered:
            ordered.remove(zone)
            return [zone, *ordered]
    return ordered


def ensure_writer_zone_affinity(
    runner: CommandRunner,
    *,
    aws_region: str,
    cluster_id: str,
    instance_ids: Sequence[str],
    control_plane_zones: Sequence[str],
    timeout_seconds: float = WRITER_ZONE_FAILOVER_TIMEOUT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Fail the cluster over when its writer runs outside the control-plane zone.

    Creation order places the writer (``writer_first_zones``), but an RDS
    failover, a resumed bootstrap or a site created before this rule can leave
    the writer elsewhere. When another available member sits in a control-plane
    zone, ``failover-db-cluster`` targets it and the call waits for the cluster
    to report that member as the writer. With no such member nothing is done
    and the report says why.
    """

    wanted = set(control_plane_zones)
    if not wanted or not instance_ids:
        return {"action": "none", "reason": "control-plane zones unknown"}
    statuses = _instance_placements(
        runner, aws_region=aws_region, cluster_id=cluster_id
    )
    writer_id = _cluster_writer(runner, aws_region=aws_region, cluster_id=cluster_id)
    writer_zone = (statuses.get(writer_id) or {}).get("availability_zone")
    report = {
        "writer": writer_id,
        "writer_zone": writer_zone,
        "control_plane_zones": sorted(wanted),
    }
    if writer_zone in wanted:
        return {"action": "none", **report}
    candidates = [
        instance_id
        for instance_id in instance_ids
        if instance_id != writer_id
        and (statuses.get(instance_id) or {}).get("availability_zone") in wanted
        and (statuses.get(instance_id) or {}).get("status") == "available"
    ]
    if not candidates:
        return {
            "action": "none",
            "reason": "no available member in a control-plane zone",
            **report,
        }
    target = candidates[0]
    runner.run(
        [
            "aws",
            "rds",
            "failover-db-cluster",
            "--region",
            aws_region,
            "--db-cluster-identifier",
            cluster_id,
            "--target-db-instance-identifier",
            target,
        ],
        mutate=True,
        capture=False,
    )
    deadline = time.monotonic() + timeout_seconds
    while True:
        cluster = _cluster_document(
            runner, aws_region=aws_region, cluster_id=cluster_id
        )
        writers = [
            member.get("DBInstanceIdentifier")
            for member in cluster.get("DBClusterMembers") or []
            if member.get("IsClusterWriter")
        ]
        if cluster.get("Status") == "available" and writers == [target]:
            return {"action": "failover", "target": target, **report}
        if time.monotonic() >= deadline:
            raise BootstrapError(
                f"Aurora writer failover to {target} did not settle within "
                f"{timeout_seconds:.0f}s (status={cluster.get('Status')}, writers={writers})"
            )
        sleep(15.0)


def _instance_placements(
    runner: CommandRunner, *, aws_region: str, cluster_id: str
) -> dict[str, dict[str, str]]:
    """``DBInstanceIdentifier -> {status, availability_zone}`` from one filtered describe."""

    listed = (
        runner.aws_json(
            aws_region,
            "rds",
            "describe-db-instances",
            "--filters",
            f"Name=db-cluster-id,Values={cluster_id}",
        ).get("DBInstances")
        or []
    )
    return {
        str(instance.get("DBInstanceIdentifier") or ""): {
            "status": str(instance.get("DBInstanceStatus") or ""),
            "availability_zone": str(instance.get("AvailabilityZone") or ""),
        }
        for instance in listed
    }


def _cluster_document(
    runner: CommandRunner, *, aws_region: str, cluster_id: str
) -> dict[str, Any]:
    clusters = runner.aws_json(
        aws_region,
        "rds",
        "describe-db-clusters",
        "--db-cluster-identifier",
        cluster_id,
    ).get("DBClusters")
    if (
        not isinstance(clusters, list)
        or len(clusters) != 1
        or clusters[0].get("DBClusterIdentifier") != cluster_id
    ):
        raise BootstrapError(f"cannot describe Aurora cluster {cluster_id}")
    return dict(clusters[0])


def _cluster_writer(runner: CommandRunner, *, aws_region: str, cluster_id: str) -> str:
    cluster = _cluster_document(runner, aws_region=aws_region, cluster_id=cluster_id)
    writers = [
        str(member.get("DBInstanceIdentifier") or "")
        for member in cluster.get("DBClusterMembers") or []
        if member.get("IsClusterWriter")
    ]
    if len(writers) != 1 or not writers[0]:
        raise BootstrapError(f"Aurora cluster {cluster_id} has no single writer")
    return writers[0]


def _ensure_instance_site_tag(
    runner: CommandRunner,
    *,
    aws_region: str,
    document: Mapping[str, Any],
    instance_id: str,
    site_id: str,
) -> None:
    instances = document.get("DBInstances")
    instance = next(
        (
            item
            for item in (instances if isinstance(instances, list) else [])
            if isinstance(item, dict)
            and item.get("DBInstanceIdentifier") == instance_id
        ),
        None,
    )
    arn = str((instance or {}).get("DBInstanceArn") or "")
    if instance is None or not arn:
        raise BootstrapError(f"Aurora instance {instance_id} describe lacks its ARN")
    ensure_rds_site_tag(
        runner,
        region=aws_region,
        resource_arn=arn,
        tags=instance.get("TagList"),
        site_id=site_id,
        description=f"Aurora instance {instance_id}",
    )


def _await_rds_available(
    runner: CommandRunner,
    *,
    aws_region: str,
    kind: Literal["instance", "cluster"],
    identifier: str,
) -> None:
    # Load only local models. All transport stays on the budgeted runner, never
    # an SDK client, and no API slot is held between individual describe calls.
    from botocore import xform_name
    from botocore.loaders import Loader
    from botocore.waiter import WaiterModel

    resource = f"DB{kind.capitalize()}"
    waiter_name = f"{resource}Available"
    try:
        with deadline_scope(f"RDS {waiter_name}", RDS_AVAILABILITY_TIMEOUT_SECONDS):
            config = WaiterModel(
                Loader().load_service_model("rds", "waiters-2")
            ).get_waiter(waiter_name)
            acceptors = config.acceptors
            arguments = [
                "aws",
                "rds",
                xform_name(config.operation, "-"),
                "--region",
                aws_region,
                f"--db-{kind}-identifier",
                identifier,
                "--output",
                "json",
            ]
            for attempt in range(1, config.max_attempts + 1):
                raw = runner.run(
                    arguments,
                    # Probes must reject a long readiness wait even though
                    # each request is an ordinary read.
                    mutate=True,
                    timeout_seconds=command_timeout(arguments, None),
                )
                command_timeout(arguments, None)
                try:
                    response = json.loads(raw)
                except ValueError:
                    raise BootstrapError(
                        f"RDS {waiter_name} returned invalid JSON"
                    ) from None
                if not isinstance(response, dict) or "Error" in response:
                    raise BootstrapError(
                        f"RDS {waiter_name} returned an invalid or error response"
                    )
                entries = response.get(f"{resource}s")
                if (
                    not isinstance(entries, list)
                    or len(entries) > 1
                    or any(
                        not isinstance(entry, dict)
                        or entry.get(f"{resource}Identifier") != identifier
                        for entry in entries
                    )
                ):
                    raise BootstrapError(f"RDS {waiter_name} response identity differs")
                for acceptor in acceptors:
                    if acceptor.matcher_func(response):
                        if acceptor.state == "success":
                            return
                        if acceptor.state != "retry":
                            raise BootstrapError(
                                f"RDS {waiter_name} encountered a terminal failure "
                                f"state: {acceptor.explanation}"
                            )
                        break
                if attempt < config.max_attempts:
                    time.sleep(command_timeout(arguments, float(config.delay)))
            raise BootstrapError(
                f"RDS {waiter_name} exceeded {config.max_attempts} attempts"
            )
    except TimeoutError:
        raise BootstrapError(f"RDS {waiter_name} exceeded its time budget") from None


def await_serverless_instances(
    runner: CommandRunner,
    *,
    aws_region: str,
    instance_ids: Sequence[str],
) -> None:
    """Block until every instance reads ``available``, waiting for all at once.

    These waits only observe existing instances; they do not make dependent
    creates concurrent. The read-only probe runner rejects long waits so a
    probe answers in seconds instead of blocking on RDS.
    """

    def wait_for(instance_id: str) -> None:
        _await_rds_available(
            runner,
            aws_region=aws_region,
            kind="instance",
            identifier=instance_id,
        )

    if not instance_ids:
        return
    with ThreadPoolExecutor(max_workers=min(2, len(instance_ids))) as pool:
        # Copy separately so concurrent waiters retain the task deadline.
        futures = [
            pool.submit(copy_context().run, wait_for, instance_id)
            for instance_id in instance_ids
        ]
        failures = [
            failure
            for failure in (future.exception() for future in futures)
            if failure is not None
        ]
    if failures:
        # Lost supervision or interruption must not become a recoverable failure.
        raise next(
            (failure for failure in failures if not isinstance(failure, Exception)),
            failures[0],
        )


def serverless_instance_statuses(
    runner: CommandRunner, *, aws_region: str, cluster_id: str
) -> dict[str, str]:
    """``DBInstanceIdentifier -> DBInstanceStatus`` for the cluster's members,
    from one filtered describe. An instance that is absent from the mapping
    has no create behind it yet (or one still propagating)."""

    listed = (
        runner.aws_json(
            aws_region,
            "rds",
            "describe-db-instances",
            "--filters",
            f"Name=db-cluster-id,Values={cluster_id}",
        ).get("DBInstances")
        or []
    )
    return {
        str(instance.get("DBInstanceIdentifier") or ""): str(
            instance.get("DBInstanceStatus") or ""
        )
        for instance in listed
    }


def pending_serverless_instances(
    runner: CommandRunner,
    *,
    aws_region: str,
    cluster_id: str,
    instance_ids: Sequence[str],
) -> list[str]:
    """The instances that do not yet read ``available``, from one describe.

    An instance the listing does not know is pending too: its create may still
    be propagating, and the waiter that follows fails closed on one that never
    appears rather than letting this read stand in for it.
    """

    statuses = serverless_instance_statuses(
        runner, aws_region=aws_region, cluster_id=cluster_id
    )
    return [
        instance_id
        for instance_id in instance_ids
        if statuses.get(instance_id) != "available"
    ]


@dataclass(frozen=True)
class AuroraReadiness:
    """What the control plane needs to reach the database: the writer endpoint,
    the managed master secret and its credentials. The credentials are kept out
    of the repr so a failure message or a log line never carries them."""

    endpoint: str
    secret_arn: str
    kms_key_arn: str
    username: str = field(repr=False)
    password: str = field(repr=False)


_RETRYABLE_SECRET_READ = ("ResourceNotFoundException", "AWSCURRENT")


def read_master_secret(
    runner: CommandRunner,
    *,
    aws_region: str,
    cluster_id: str,
    attempts: int = MASTER_SECRET_READ_ATTEMPTS,
    poll_seconds: float = MASTER_SECRET_READ_POLL_SECONDS,
) -> AuroraReadiness:
    """Read the managed master secret, retrying only while it is not there yet.

    Bounded, and never fail-open: after ``attempts`` reads the deploy fails
    with the last error rather than continuing without a credential. A read
    that fails for any other reason -- a denied permission, a broken CLI --
    reads the same on every try and surfaces at once with its own message.
    """

    last_error: str = "the cluster reports no managed master secret yet"
    for attempt in range(1, attempts + 1):
        database = runner.aws_json(
            aws_region,
            "rds",
            "describe-db-clusters",
            "--db-cluster-identifier",
            cluster_id,
        )["DBClusters"][0]
        master = database.get("MasterUserSecret") or {}
        secret_arn = str(master.get("SecretArn") or "")
        if secret_arn:
            try:
                value = json.loads(
                    runner.aws_text(
                        aws_region,
                        "secretsmanager",
                        "get-secret-value",
                        "--secret-id",
                        secret_arn,
                        "--query",
                        "SecretString",
                        sensitive=True,
                    )
                )
            except BootstrapError as exc:
                message = str(exc)
                if not any(code in message for code in _RETRYABLE_SECRET_READ):
                    raise
                last_error = message
            except ValueError:
                last_error = "the secret value is not JSON yet"
            else:
                username = str(value.get("username") or "")
                password = str(value.get("password") or "")
                if username and password:
                    return AuroraReadiness(
                        endpoint=str(database["Endpoint"]),
                        secret_arn=secret_arn,
                        kms_key_arn=str(master.get("KmsKeyId") or ""),
                        username=username,
                        password=password,
                    )
                last_error = "the secret value carries no username/password yet"
        if attempt < attempts:
            time.sleep(poll_seconds)
    raise BootstrapError(
        f"Aurora cluster {cluster_id}: master user secret is not readable after "
        f"{attempts} attempts ({last_error})"
    )


def await_aurora_ready(
    runner: CommandRunner,
    *,
    aws_region: str,
    cluster_id: str,
    instance_ids: Sequence[str],
) -> AuroraReadiness:
    """Wait for the instances the foundation created, then read the credentials.

    One describe decides which instances still need polling, so a rerun
    against an available cluster issues no wait at all; the writer and reader
    still transitioning on a resumed deploy are waited for together.
    """

    pending = pending_serverless_instances(
        runner,
        aws_region=aws_region,
        cluster_id=cluster_id,
        instance_ids=instance_ids,
    )
    await_serverless_instances(
        runner,
        aws_region=aws_region,
        instance_ids=pending,
    )
    return read_master_secret(runner, aws_region=aws_region, cluster_id=cluster_id)


def assert_aurora_ready(
    runner: CommandRunner,
    *,
    aws_region: str,
    cluster_id: str,
    instance_ids: Sequence[str],
    kubeconfig: Path,
    namespace: str,
) -> None:
    """The read-only probe behind ``aurora_ready``: both instances available and
    the control-plane Secret present, else ``BootstrapMutationRequired``.

    Two reads, no wait: a probe answers in seconds. Nothing here reads Secret
    material -- the jsonpath projects only the object's name.
    """

    pending = pending_serverless_instances(
        runner,
        aws_region=aws_region,
        cluster_id=cluster_id,
        instance_ids=instance_ids,
    )
    if pending:
        raise BootstrapMutationRequired(
            f"Aurora instance(s) not available: {', '.join(pending)}"
        )
    present = kubectl_projection(
        runner,
        kubeconfig=kubeconfig,
        namespace=namespace,
        arguments=[
            "get",
            "secret",
            AURORA_SECRET_NAME,
            "-o",
            "jsonpath={.metadata.name}",
        ],
    )
    if present != AURORA_SECRET_NAME:
        raise BootstrapMutationRequired(f"{AURORA_SECRET_NAME} Secret")


def cluster_parameter_group_name(
    cluster_id: str, *, safe_name: Callable[..., str]
) -> str:
    # A group per cluster: the default group cannot be modified, and sharing
    # one between sites would let one site's change reach another's database.
    return safe_name(f"{cluster_id}-pg", maximum=255)


def ensure_cluster_parameter_group(
    runner: CommandRunner,
    *,
    aws_region: str,
    cluster_id: str,
    engine_version: str,
    safe_name: Callable[..., str],
    site_id: str | None = None,
) -> str:
    """Create the cluster parameter group when missing and hold its parameters
    at the diagnostic values; returns the group name.

    Read-compare-write per parameter so a routine deploy against a settled
    group issues no modification at all (``modify-db-cluster-parameter-group``
    is a mutation even when nothing changes), and so an operator's deliberate
    change to any other parameter in the group is left alone.
    """

    group = cluster_parameter_group_name(cluster_id, safe_name=safe_name)
    existing = describe_or_absent(
        runner,
        aws_region,
        "rds",
        "describe-db-cluster-parameter-groups",
        "--db-cluster-parameter-group-name",
        group,
        not_found=("DBParameterGroupNotFound",),
    )
    exists = existing is not None
    if existing is not None and site_id is not None:
        # The uninstall proves ownership of the group by its site tag; a group
        # created before the tag existed receives it on the next deploy.
        groups = existing.get("DBClusterParameterGroups")
        arn = str(
            (groups[0] if isinstance(groups, list) and groups else {}).get(
                "DBClusterParameterGroupArn"
            )
            or ""
        )
        if not arn:
            raise BootstrapError(f"parameter group {group} describe lacks its ARN")
        ensure_rds_site_tag(
            runner,
            region=aws_region,
            resource_arn=arn,
            tags=runner.aws_json(
                aws_region, "rds", "list-tags-for-resource", "--resource-name", arn
            ).get("TagList"),
            site_id=site_id,
            description=f"Aurora parameter group {group}",
        )
    if not exists:
        versions = (
            runner.aws_json(
                aws_region,
                "rds",
                "describe-db-engine-versions",
                "--engine",
                "aurora-postgresql",
                "--engine-version",
                engine_version,
            ).get("DBEngineVersions")
            or []
        )
        family = str(
            (versions[0] if versions else {}).get("DBParameterGroupFamily") or ""
        )
        if not family:
            raise BootstrapError(
                f"cannot resolve the parameter group family for aurora-postgresql "
                f"{engine_version}"
            )
        runner.run(
            [
                "aws",
                "rds",
                "create-db-cluster-parameter-group",
                "--region",
                aws_region,
                "--db-cluster-parameter-group-name",
                group,
                "--db-parameter-group-family",
                family,
                "--description",
                f"gpu-fault control plane diagnostics for {cluster_id}",
                *(
                    ["--tags", f"Key={SITE_TAG_KEY},Value={site_id}"]
                    if site_id is not None
                    else []
                ),
            ],
            mutate=True,
            capture=False,
        )
        current: dict[str, str] = {}
    else:
        names = ", ".join(
            f"'{name}'" for name, _value, _method in DIAGNOSTIC_PARAMETERS
        )
        listed: Any = runner.aws_json(
            aws_region,
            "rds",
            "describe-db-cluster-parameters",
            "--db-cluster-parameter-group-name",
            group,
            "--query",
            f"Parameters[?contains([{names}], ParameterName)]",
        )
        # `--query Parameters[...]` projects the list itself; only an unfiltered
        # response still wraps it in {"Parameters": [...]}. The wrapped shape
        # was the only one the first live run ever saw, because that run
        # created the group and never listed it (live 2026-09-08).
        items = listed if isinstance(listed, list) else (listed.get("Parameters") or [])
        current = {
            str(item.get("ParameterName")): str(item.get("ParameterValue") or "")
            for item in items
        }
    drifted = [
        f"ParameterName={name},ParameterValue={value},ApplyMethod={method}"
        for name, value, method in DIAGNOSTIC_PARAMETERS
        if current.get(name) != value
    ]
    if drifted:
        runner.run(
            [
                "aws",
                "rds",
                "modify-db-cluster-parameter-group",
                "--region",
                aws_region,
                "--db-cluster-parameter-group-name",
                group,
                "--parameters",
                *drifted,
            ],
            mutate=True,
            capture=False,
        )
    return group


def reconcile_cluster_diagnostics(
    runner: CommandRunner,
    *,
    aws_region: str,
    cluster_id: str,
    cluster: dict[str, Any],
    parameter_group: str,
) -> bool:
    """Attach the parameter group and the PostgreSQL log export to an existing
    cluster when either is missing; returns whether anything was changed.

    Both are online changes: no instance restarts, no connection drops. The
    static ``shared_preload_libraries`` becomes ``pending-reboot`` and stays so
    until the operator's reboot window.
    """

    arguments: list[str] = []
    if str(cluster.get("DBClusterParameterGroup") or "") != parameter_group:
        arguments.extend(["--db-cluster-parameter-group-name", parameter_group])
    exports = [str(item) for item in cluster.get("EnabledCloudwatchLogsExports") or []]
    if POSTGRESQL_LOG_EXPORT not in exports:
        arguments.extend(
            [
                "--cloudwatch-logs-export-configuration",
                f'{{"EnableLogTypes":["{POSTGRESQL_LOG_EXPORT}"]}}',
            ]
        )
    if not arguments:
        return False
    runner.run(
        [
            "aws",
            "rds",
            "modify-db-cluster",
            "--region",
            aws_region,
            "--db-cluster-identifier",
            cluster_id,
            *arguments,
            "--apply-immediately",
        ],
        mutate=True,
        capture=False,
    )
    # The modify answers before the cluster flips to ``modifying``; the same
    # settle wait the capacity change needs (see above).
    wait_for_capacity_to_settle(runner, aws_region=aws_region, cluster_id=cluster_id)
    return True
