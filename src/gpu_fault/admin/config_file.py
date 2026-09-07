from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Mapping

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin.config import (
    ADMIN_CONFIG_API_VERSION,
    ADMIN_CONFIG_KIND,
    AdminConfig,
    AdminConfigError,
    admin_config_desired_path,
    admin_config_lock,
    default_admin_config,
    load_desired_admin_config,
    persist_desired_admin_config,
)
from gpu_fault.admin.config_patch import apply_admin_config_patch

ADMIN_CONFIG_EDITABLE = Path("admin-config.yaml")


def _mapping(
    value: object,
    path: str,
    *,
    allowed: set[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AdminConfigError(f"{path} must be a mapping")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise AdminConfigError(f"{path} contains unknown fields: {', '.join(unknown)}")
    return value


def load_admin_config_file(
    path: Path,
    *,
    base: AdminConfig | None = None,
    require_private: bool = True,
) -> AdminConfig:
    source = path.expanduser().resolve()
    try:
        if require_private and source.stat().st_mode & 0o077:
            raise AdminConfigError(
                f"admin config must not grant group/other permissions: {source}"
            )
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
    except AdminConfigError:
        raise
    except (OSError, yaml.YAMLError) as exc:
        raise AdminConfigError(f"cannot read admin config {source}: {exc}") from exc
    data = _mapping(
        document,
        "admin config file",
        allowed={"apiVersion", "kind", "spec"},
    )
    if data.get("apiVersion") != ADMIN_CONFIG_API_VERSION:
        raise AdminConfigError(
            f"admin config apiVersion must be {ADMIN_CONFIG_API_VERSION}"
        )
    if data.get("kind") != ADMIN_CONFIG_KIND:
        raise AdminConfigError(f"admin config kind must be {ADMIN_CONFIG_KIND}")
    spec = _mapping(
        data.get("spec") or {},
        "admin config spec",
        allowed={
            "capacity",
            "aurora",
            "processor",
            "workflow",
            "notificationDelivery",
            "evidence",
        },
    )
    return apply_admin_config_patch(
        base or default_admin_config(),
        spec,
    )


def admin_config_file_path(state_dir: Path) -> Path:
    return state_dir.expanduser().resolve() / ADMIN_CONFIG_EDITABLE


def admin_config_file_document(config: AdminConfig) -> dict[str, object]:
    capacity = config.capacity
    remediation = capacity.remediation
    spool = capacity.telemetry_spool
    aurora = config.aurora
    processor = config.processor
    workflow = config.workflow
    notification = config.notification_delivery
    evidence = config.evidence
    return {
        "apiVersion": ADMIN_CONFIG_API_VERSION,
        "kind": ADMIN_CONFIG_KIND,
        "spec": {
            "capacity": {
                "controlWorkerReplicas": capacity.control_worker_replicas,
                "largestClusterNodeCount": capacity.largest_cluster_node_count,
                "managedNodeCount": capacity.managed_node_count,
                "telemetrySpool": {
                    "enabled": spool.enabled,
                    "replicas": spool.replicas,
                },
                "remediation": {
                    "maxActiveRegion": remediation.max_active_region,
                    "maxActivePerCluster": remediation.max_active_per_cluster,
                    "maxActivePerResourceClass": (
                        remediation.max_active_per_resource_class
                    ),
                },
            },
            "aurora": {
                "minAcu": aurora.min_acu,
                "maxAcu": aurora.max_acu,
            },
            "processor": {
                "maxQueueDepth": processor.max_queue_depth,
                "maxClusterQueueDepth": processor.max_cluster_queue_depth,
                "retryAfterSeconds": processor.retry_after_seconds,
                "retryBackoffSeconds": processor.retry_backoff_seconds,
                "retryBackoffMaxSeconds": (processor.retry_backoff_max_seconds),
                "completedRetentionSeconds": (processor.completed_retention_seconds),
            },
            "workflow": {
                "pollIntervalSeconds": workflow.poll_interval_seconds,
                "dispatcherWorkers": workflow.dispatcher_workers,
            },
            "notificationDelivery": {
                "batchSize": notification.batch_size,
                "maxAttempts": notification.max_attempts,
            },
            "evidence": {
                "retentionHours": evidence.retention_hours,
                "maxRecordsPerNode": evidence.max_records_per_node,
            },
        },
    }


def write_admin_config_file(
    path: Path,
    config: AdminConfig,
    *,
    overwrite: bool,
) -> Path:
    target = path.expanduser().resolve()
    if target.exists() and not overwrite:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.parent.chmod(0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                admin_config_file_document(config),
                handle,
                sort_keys=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def initialize_desired_admin_config(
    state_dir: Path,
    *,
    config_file: Path | None = None,
    permit_change: bool = False,
) -> AdminConfig:
    with admin_config_lock(state_dir):
        permit_change = (
            permit_change
            and not (state_dir.expanduser().resolve() / "site.yaml").is_file()
        )
        desired_path = admin_config_desired_path(state_dir)
        editable = admin_config_file_path(state_dir)
        current = load_desired_admin_config(
            state_dir,
            migrate_legacy=True,
        )
        source_file = config_file
        if source_file is None and permit_change and editable.is_file():
            source_file = editable
        desired = (
            load_admin_config_file(source_file, base=current)
            if source_file is not None
            else current
        )
        if desired != current and not permit_change:
            raise AdminConfigError(
                "existing site admin config differs from --config; use "
                "gpu-fault-admin config"
            )
        if not desired_path.is_file() or desired != current:
            persist_desired_admin_config(
                state_dir,
                config=desired,
                source=(
                    f"file:{source_file.expanduser().resolve()}"
                    if source_file is not None
                    else "release-defaults"
                ),
            )
        write_admin_config_file(
            editable,
            desired,
            overwrite=(source_file is not None and source_file != editable),
        )
        return desired
