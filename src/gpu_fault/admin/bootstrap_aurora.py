from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

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

CAPACITY_SETTLE_POLL_SECONDS = 10.0
CAPACITY_SETTLE_TIMEOUT_SECONDS = 1800.0
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


def ensure_serverless_instances(
    runner: CommandRunner,
    *,
    aws_region: str,
    cluster_id: str,
    availability_zones: list[str],
    safe_name: Callable[..., str],
    wait: bool = True,
) -> list[str]:
    """Create the writer and the reader that are missing; returns both ids.

    The foundation ``aurora`` task passes ``wait=False``: the five-to-ten-minute
    instance wait then belongs to ``aurora_ready``, which runs alongside the
    platform tasks that need no database. ``wait=True`` keeps the old contract
    of returning only once both instances are available.
    """

    instance_ids = [
        safe_name(f"{cluster_id}-{suffix}", maximum=63)
        for suffix in ("writer", "reader")
    ]
    for instance_id, availability_zone in zip(
        instance_ids,
        availability_zones,
        strict=True,
    ):
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
            continue
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
                "--promotion-tier",
                "0",
            ],
            mutate=True,
            capture=False,
        )
    if wait:
        await_serverless_instances(
            runner, aws_region=aws_region, instance_ids=instance_ids
        )
    return instance_ids


def await_serverless_instances(
    runner: CommandRunner,
    *,
    aws_region: str,
    instance_ids: Sequence[str],
) -> None:
    """Block until every instance reads ``available``, waiting for all at once.

    Two ``rds wait`` calls back to back cost the sum of both creations; issued
    together they cost the slower one. The wait is a mutation to the read-only
    probe runner on purpose: a probe must answer in seconds, not block on RDS.
    """

    def wait_for(instance_id: str) -> None:
        runner.run(
            [
                "aws",
                "rds",
                "wait",
                "db-instance-available",
                "--region",
                aws_region,
                "--db-instance-identifier",
                instance_id,
            ],
            mutate=True,
            capture=False,
        )

    if not instance_ids:
        return
    with ThreadPoolExecutor(max_workers=min(2, len(instance_ids))) as pool:
        futures = [pool.submit(wait_for, instance_id) for instance_id in instance_ids]
        failures = [
            failure
            for failure in (future.exception() for future in futures)
            if failure is not None
        ]
    if failures:
        raise failures[0]


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
    statuses = {
        str(instance.get("DBInstanceIdentifier") or ""): str(
            instance.get("DBInstanceStatus") or ""
        )
        for instance in listed
    }
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

    One describe decides which instances still need an ``rds wait``, so a rerun
    against an available cluster issues no wait at all; the writer and reader
    still creating on a first deploy are waited for together.
    """

    pending = pending_serverless_instances(
        runner,
        aws_region=aws_region,
        cluster_id=cluster_id,
        instance_ids=instance_ids,
    )
    await_serverless_instances(runner, aws_region=aws_region, instance_ids=pending)
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
) -> str:
    """Create the cluster parameter group when missing and hold its parameters
    at the diagnostic values; returns the group name.

    Read-compare-write per parameter so a routine deploy against a settled
    group issues no modification at all (``modify-db-cluster-parameter-group``
    is a mutation even when nothing changes), and so an operator's deliberate
    change to any other parameter in the group is left alone.
    """

    group = cluster_parameter_group_name(cluster_id, safe_name=safe_name)
    exists = (
        describe_or_absent(
            runner,
            aws_region,
            "rds",
            "describe-db-cluster-parameter-groups",
            "--db-cluster-parameter-group-name",
            group,
            not_found=("DBParameterGroupNotFound",),
        )
        is not None
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
