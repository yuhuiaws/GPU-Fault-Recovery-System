"""Installation-bound uninstall retirement and retained database handoff."""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import BootstrapError, CommandRunner
from gpu_fault.admin.uninstall_types import UNINSTALL_PHASES, UninstallRequest
from gpu_fault import installation_lifecycle as lifecycle_records
from gpu_fault.installation_lifecycle import (
    INSTALLATION_FILE as INSTALLATION_FILE,
    RETIREMENT_FILE as RETIREMENT_FILE,
    InstallationLifecycleError,
    content_sha256 as content_sha256,
    site_identity as site_identity,
)

if TYPE_CHECKING:
    from gpu_fault.admin.site import RenderedSite


def read_record(path: Path) -> dict[str, Any]:
    try:
        return lifecycle_records.read_record(path)
    except InstallationLifecycleError as exc:
        raise BootstrapError(str(exc)) from exc


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_record(path: Path, value: dict[str, Any]) -> None:
    write_json_atomic(path, value)
    _sync_directory(path.parent)


def _require_installation(value: dict[str, Any]) -> str:
    try:
        return lifecycle_records.require_installation(value)
    except InstallationLifecycleError as exc:
        raise BootstrapError(str(exc)) from exc


def _require_retirement_complete(state_dir: Path) -> None:
    try:
        lifecycle_records.require_retirement_complete(state_dir)
    except InstallationLifecycleError as exc:
        raise BootstrapError(str(exc)) from exc


def installation_for_uninstall(site: RenderedSite) -> str:
    state_dir = site.source.parent
    _require_retirement_complete(state_dir)
    path = state_dir / INSTALLATION_FILE
    if not path.exists():
        write_record(
            path,
            {
                "schema_version": 1,
                "installation_id": uuid.uuid4().hex,
                "site_identity": site_identity(site.release_config),
                "registry_site_id": site.metadata_name,
            },
        )
    value = read_record(path)
    identity = value.get("site_identity")
    if identity is not None and identity != site_identity(site.release_config):
        raise BootstrapError("uninstall installation belongs to another site")
    return _require_installation(value)


def uninstall_state(request: UninstallRequest, path: Path) -> dict[str, Any]:
    from gpu_fault.admin.bootstrap_common import safe_name

    if not path.exists() and path.parent.exists() and any(path.parent.iterdir()):
        raise BootstrapError("uninstall transaction files lack their original journal")
    expected = {
        "schema_version": 2,
        "installation_id": installation_for_uninstall(request.site),
        "site_id": request.site.release_config["site_name"],
        "registry_site_id": request.site.registry_site_id,
        "site_identity": site_identity(request.site.release_config),
        "site_sha256": request.site.source_sha256,
        "cpu_disposition": request.cpu_disposition,
        "final_snapshot_policy": request.final_snapshot_policy,
        "reset_database": request.reset_database,
    }
    if path.exists():
        value = read_record(path)
        if value.get("phase") not in UNINSTALL_PHASES:
            raise BootstrapError("invalid uninstall state")
        if "supervision_lost" in value:
            raise BootstrapError(
                "uninstall supervision was lost; reconcile before retry"
            )
        for key, item in expected.items():
            if value.get(key) != item or type(value.get(key)) is not type(item):
                raise BootstrapError(f"uninstall state conflicts on {key}")
        return value
    attempt_id = uuid.uuid4().hex
    value = {
        **expected,
        "attempt_id": attempt_id,
        "final_snapshot_identifier": (
            safe_name(f"gpu-fault-{expected['site_id']}-uninstall", maximum=45)
            + "-"
            + attempt_id[:16]
        ),
        "phase": "STARTED",
        "updated_at": datetime.now(UTC).isoformat(),
    }
    write_record(path, value)
    return value


def bind_retained_database(
    request: UninstallRequest,
    runner: CommandRunner,
    state_path: Path,
    state: dict[str, Any],
) -> None:
    if request.cpu_disposition != "keep" or request.reset_database:
        return
    identity = retained_database_identity(request.site, runner)
    previous = state.get("retained_database")
    if previous is None:
        if state["phase"] not in {"STARTED", "REGISTRY_EXPORTED"}:
            raise BootstrapError("retained database lacks its pre-cleanup incarnation")
        state["retained_database"] = identity
        write_record(state_path, state)
    elif previous != identity:
        raise BootstrapError("retained Aurora incarnation changed during uninstall")


# Aurora reports these while the cluster stays online, writable and unchanged
# in identity: ``backing-up`` during the first automated backup right after
# creation and during every daily backup window, ``maintenance`` during the
# maintenance window. Treating them as "binding differs" made the 2026-09-30
# first deploy fail its preflight fifteen minutes after creating the cluster;
# rerunning the same command once the backup ended passed. Anything else
# (``creating``, ``modifying``, ``failing-over``, ``deleting``, ``stopped`` ...)
# still fails closed.
AURORA_ONLINE_STATUSES = frozenset({"available", "backing-up", "maintenance"})


def aurora_binding(
    cluster: object,
    *,
    cpu_eks_arn: str,
    aws_region: str,
    cluster_id: str,
    master_secret_arn: str | None = None,
) -> dict[str, Any]:
    if not isinstance(cluster, dict):
        raise BootstrapError("retained Aurora identity is unavailable")
    cpu = cpu_eks_arn.split(":")
    expected_arn = f"arn:{cpu[1]}:rds:{aws_region}:{cpu[4]}:cluster:{cluster_id}"
    secret = cluster.get("MasterUserSecret")
    secret_arn = secret.get("SecretArn") if isinstance(secret, dict) else None
    required = ("DbClusterResourceId", "Endpoint", "DatabaseName", "MasterUsername")
    status = cluster.get("Status")
    if status not in AURORA_ONLINE_STATUSES:
        # Distinct from a binding mismatch: the identity may be exactly right
        # while the cluster is still creating, failing over or being deleted.
        raise BootstrapError(
            f"retained Aurora cluster is not online: status={status!r}"
        )
    if (
        cluster.get("DBClusterIdentifier") != cluster_id
        or cluster.get("DBClusterArn") != expected_arn
        or cluster.get("Engine") != "aurora-postgresql"
        or any(
            not isinstance(cluster.get(key), str) or not cluster[key]
            for key in required
        )
        or not isinstance(secret_arn, str)
        or not secret_arn.startswith(
            f"arn:{cpu[1]}:secretsmanager:{aws_region}:{cpu[4]}:secret:"
        )
        or master_secret_arn is not None
        and secret_arn != master_secret_arn
        or type(cluster.get("Port")) is not int
        or cluster["Port"] != 5432
    ):
        raise BootstrapError(
            "retained Aurora incarnation binding differs or is incomplete"
        )
    return {
        "cluster_arn": expected_arn,
        "cluster_resource_id": cluster["DbClusterResourceId"],
        "master_secret_arn": secret_arn,
        "endpoint": cluster["Endpoint"],
        "database": cluster["DatabaseName"],
        "username": cluster["MasterUsername"],
        "port": cluster["Port"],
    }


def retained_database_identity(
    site: RenderedSite, runner: CommandRunner
) -> dict[str, Any]:
    config = site.release_config
    cluster_id = str(config["health"]["aurora_cluster_id"])
    values = runner.aws_json(
        str(config["aws_region"]),
        "rds",
        "describe-db-clusters",
        "--db-cluster-identifier",
        cluster_id,
    ).get("DBClusters")
    if not isinstance(values, list) or len(values) != 1:
        raise BootstrapError("retained Aurora identity is unavailable")
    return aurora_binding(
        values[0],
        cpu_eks_arn=str(config["cpu_eks_arn"]),
        aws_region=str(config["aws_region"]),
        cluster_id=cluster_id,
    )


def completed_retained_uninstall(
    directory: Path, *, state: dict[str, Any] | None = None
) -> dict[str, Any]:
    from gpu_fault.installation_resources import InstallationResourceSnapshot

    state = read_record(directory / "state.json") if state is None else state
    try:
        finished = datetime.fromisoformat(state["updated_at"])
        cleanup = read_record(directory / "kubernetes-cleanup.json")
        final = InstallationResourceSnapshot.model_validate(
            read_record(directory / "installation-resources-final.json")
        )
        final.require_source_binding()
        namespace_uid = cleanup["namespace_snapshots"]["cpu"]["uid"]
        cluster = next(
            item for item in final.resources if item.resource_type == "aurora_cluster"
        )
        valid = (
            type(state["schema_version"]) is int
            and state["schema_version"] == 2
            and re.fullmatch(r"[a-f0-9]{32}", state["installation_id"]) is not None
            and state["phase"] == "COMPLETED"
            and "supervision_lost" not in state
            and state["cpu_disposition"] == "keep"
            and state["reset_database"] is False
            and state["delete_policy_residuals"] == 0
            and finished.tzinfo is not None
            and finished <= datetime.now(UTC)
            and final.site_id == state.get("registry_site_id", state["site_id"])
            and final.digest() == state["final_registry_sha256"]
            and cluster.status.value == "PRESERVED"
            and state["effective_policies"][cluster.resource_key] == "PRESERVE"
            and cluster.resource_id
            == state["retained_database"]["cluster_arn"].rsplit(":", 1)[-1]
            and cleanup["schema_version"] == 2
            and cleanup["phase"] == "CLEANUP_COMPLETED"
            and cleanup["status"] == "COMPLETED"
            and cleanup["scope"] == "all"
            and cleanup["mode"] == "reset"
            and cleanup["node_mode"] == "uninstall"
            and cleanup["content_sha256"] == state["cleanup_sha256"]
            and content_sha256(
                {
                    key: value
                    for key, value in cleanup.items()
                    if key != "content_sha256"
                }
            )
            == state["cleanup_sha256"]
            and isinstance(namespace_uid, str)
            and bool(namespace_uid)
        )
    except (KeyError, TypeError, ValueError, StopIteration):
        valid = False
    if not valid:
        raise BootstrapError("retained database requires a bound completed uninstall")
    return {**state, "previous_namespace_uid": namespace_uid}


def retained_handoff(
    path: Path, *, installation_id: str, identity: Mapping[str, Any], site_name: str
) -> dict[str, Any]:
    _require_retirement_complete(path.parent)
    value = read_record(path)
    handoff = value.get("retained_uninstall")
    retirement = read_record(path.parent / RETIREMENT_FILE)
    if (
        _require_installation(value) != installation_id
        or not isinstance(handoff, dict)
        or not re.fullmatch(
            r"retired-[0-9TZ]+-[a-f0-9]{32}", str(handoff.get("archive"))
        )
        or retirement.get("schema_version") != 1
        or retirement.get("phase") != "COMPLETED"
        or (retirement.get("next_installation") or {}).get("installation_id")
        != installation_id
        or (retirement.get("next_installation") or {}).get("retained_uninstall")
        != handoff
        or value.get("registry_site_id")
        != (retirement.get("next_installation") or {}).get("registry_site_id")
        or retirement.get("archive") != handoff.get("archive")
    ):
        raise BootstrapError(
            "retained database handoff belongs to another installation"
        )
    directory = path.parent / handoff["archive"] / "uninstall"
    raw_state = read_record(directory / "state.json")
    state = completed_retained_uninstall(directory, state=raw_state)
    expected_site = {
        "site_name": site_name,
        "aws_region": str(identity["cpu_eks_arn"]).split(":")[3],
        "cpu_eks_arn": identity["cpu_eks_arn"],
        "namespace": identity["namespace"],
    }
    if (
        content_sha256(raw_state) != handoff.get("uninstall_sha256")
        or state["installation_id"] != handoff.get("previous_installation_id")
        or state["installation_id"] == installation_id
        or value.get("site_identity") != expected_site
        or state.get("site_identity") != expected_site
        or state["previous_namespace_uid"] == identity["namespace_uid"]
        or any(
            identity.get(key) != item
            for key, item in state["retained_database"].items()
        )
        or set(state["retained_database"])
        != {
            "cluster_arn",
            "cluster_resource_id",
            "master_secret_arn",
            "endpoint",
            "database",
            "username",
            "port",
        }
    ):
        raise BootstrapError("retained database or new namespace identity differs")
    adoption = {
        "namespace_uid": identity["namespace_uid"],
        "identity_sha256": content_sha256(dict(identity)),
    }
    if "retained_adoption" in value:
        if value["retained_adoption"] != adoption:
            raise BootstrapError(
                "retained handoff was already bound to another namespace"
            )
    else:
        # Persist the new namespace before the first proof Job. Losing that
        # namespace cannot turn this handoff into an authorization for another.
        value["retained_adoption"] = adoption
        write_record(path, value)
    return {
        "installation_id": installation_id,
        "uninstall_sha256": handoff["uninstall_sha256"],
        "previous_installation_id": state["installation_id"],
        "previous_namespace_uid": state["previous_namespace_uid"],
    }


def _path_digest(path: Path) -> str:
    if path.is_symlink():
        raise BootstrapError("cannot retire a symlinked installation record")
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if not path.is_dir():
        raise BootstrapError("installation retirement record is missing")
    entries = {}
    for child in sorted(path.rglob("*")):
        if child.is_symlink() or not (child.is_dir() or child.is_file()):
            raise BootstrapError("cannot retire an unsafe installation record")
        entries[str(child.relative_to(path))] = (
            "directory"
            if child.is_dir()
            else hashlib.sha256(child.read_bytes()).hexdigest()
        )
    return content_sha256(entries)


def _move_record(source: Path, target: Path, digest: str) -> None:
    if source.exists():
        if target.exists() or _path_digest(source) != digest:
            raise BootstrapError("installation records changed during retirement")
        source.replace(target)
    elif not target.exists() or _path_digest(target) != digest:
        raise BootstrapError("retired installation record is missing or changed")
    _sync_directory(target.parent)
    _sync_directory(source.parent)


def retire_completed_site(state_dir: Path, names: tuple[str, ...]) -> Path | None:
    """Journal every move; archive the whole uninstall before publishing a new ID."""
    journal_path = state_dir / RETIREMENT_FILE
    journal = read_record(journal_path) if journal_path.exists() else None
    if journal is None or journal.get("phase") == "COMPLETED":
        record_path = state_dir / "uninstall" / "state.json"
        if not record_path.is_file():
            return None
        state = read_record(record_path)
        if state.get("phase") != "COMPLETED":
            return None
        archive_name = f"retired-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex}"
        next_installation: dict[str, Any] = {
            "schema_version": 1,
            "installation_id": uuid.uuid4().hex,
            "site_identity": state.get("site_identity"),
        }
        if isinstance(state.get("site_id"), str):
            next_installation["registry_site_id"] = (
                lifecycle_records.generation_registry_site_id(
                    state["site_id"], next_installation["installation_id"]
                )
            )
        if state.get("cpu_disposition") == "keep" and not state.get("reset_database"):
            completed_retained_uninstall(record_path.parent, state=state)
            previous = read_record(state_dir / INSTALLATION_FILE)
            if (
                _require_installation(previous) != state["installation_id"]
                or previous.get("site_identity") != state["site_identity"]
            ):
                raise BootstrapError(
                    "completed uninstall belongs to another installation"
                )
            next_installation["retained_uninstall"] = {
                "archive": archive_name,
                "uninstall_sha256": content_sha256(state),
                "previous_installation_id": state["installation_id"],
            }
        journal = {
            "schema_version": 1,
            "phase": "PREPARED",
            "archive": archive_name,
            "next_installation": next_installation,
            "records": {
                name: _path_digest(state_dir / name)
                for name in (*names, INSTALLATION_FILE, "uninstall")
                if (state_dir / name).exists()
            },
        }
        write_record(journal_path, journal)
    if (
        journal.get("schema_version") != 1
        or journal.get("phase") not in {"PREPARED", "RECORDS_RETIRED"}
        or not re.fullmatch(
            r"retired-[0-9TZ]+-[a-f0-9]{32}", str(journal.get("archive"))
        )
        or not isinstance(journal.get("records"), dict)
        or not set(journal["records"]).issubset(
            {*names, INSTALLATION_FILE, "uninstall"}
        )
        or "uninstall" not in journal["records"]
    ):
        raise BootstrapError("installation retirement journal is invalid")
    archive = state_dir / str(journal["archive"])
    if archive.is_symlink():
        raise BootstrapError("installation archive is unsafe")
    archive.mkdir(mode=0o700, exist_ok=True)
    _sync_directory(state_dir)
    if journal["phase"] == "PREPARED":
        for name, digest in journal["records"].items():
            _move_record(state_dir / name, archive / name, digest)
        journal["phase"] = "RECORDS_RETIRED"
        write_record(journal_path, journal)
    next_installation = journal["next_installation"]
    _require_installation(next_installation)
    current = state_dir / INSTALLATION_FILE
    if current.exists() and read_record(current) != next_installation:
        raise BootstrapError("new installation identity changed during retirement")
    write_record(current, next_installation)
    journal["phase"] = "COMPLETED"
    write_record(archive / RETIREMENT_FILE, journal)
    write_record(journal_path, journal)
    return archive
