from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from gpu_fault.execution import WorkflowStepOutcome


@dataclass
class _WorkloadMutation:
    workloads: list[tuple[str, str, str, str, Any]]
    parsed: list[tuple[str, str, str, str]]
    absent_workload_ids: list[str] = field(default_factory=list)
    source_workload_uids: dict[str, str] = field(default_factory=dict)
    source_resource_versions: dict[str, str] = field(default_factory=dict)
    terminating_pods: list[tuple[str, str]] = field(default_factory=list)
    log_evidence: list[dict[str, Any]] = field(default_factory=list)
    log_errors: list[dict[str, str]] = field(default_factory=list)
    target_gpu_count: int | None = None
    restart_count: int | None = None
    restart_attempt_id: str | None = None
    retry_workload_ids: list[str] = field(default_factory=list)
    created_retry_ids: list[str] = field(default_factory=list)


def workload_lifecycle_identity(
    workload: Any,
    *,
    metadata_reader: Callable[[Any], Any],
    resource_version_reader: Callable[[Any], Any],
) -> tuple[str, str, bool]:
    metadata = metadata_reader(workload)
    if isinstance(metadata, dict):
        uid = str(metadata.get("uid") or "")
        deleting = bool(
            metadata.get("deletionTimestamp") or metadata.get("deletion_timestamp")
        )
    else:
        uid = str(getattr(metadata, "uid", "") or "")
        deleting = bool(getattr(metadata, "deletion_timestamp", None))
    return uid, str(resource_version_reader(workload) or ""), deleting


def restart_source_failure(
    reason: str,
    message: str,
    workload_ids: list[str],
    **details: Any,
) -> WorkflowStepOutcome:
    return WorkflowStepOutcome.failed(
        message,
        details={
            "reason": reason,
            "source_workload_ids": sorted(workload_ids),
            **details,
        },
    )


def refresh_restart_workloads(
    state: _WorkloadMutation,
    *,
    read_workload: Callable[[str, str, str], Any],
    metadata_reader: Callable[[Any], Any],
    resource_version_reader: Callable[[Any], Any],
) -> WorkflowStepOutcome | None:
    refreshed = []
    missing = []
    deleting = []
    drifted: list[dict[str, str]] = []
    resource_versions: dict[str, str] = {}
    for namespace, kind, name, workload_id, _workload in state.workloads:
        try:
            latest = read_workload(namespace, kind, name)
        except Exception as exc:
            if getattr(exc, "status", None) in {404, 410}:
                missing.append(workload_id)
                continue
            raise
        current_uid, resource_version, is_deleting = workload_lifecycle_identity(
            latest,
            metadata_reader=metadata_reader,
            resource_version_reader=resource_version_reader,
        )
        expected_uid = state.source_workload_uids.get(workload_id, "")
        if is_deleting:
            deleting.append(workload_id)
            continue
        if expected_uid and current_uid != expected_uid:
            drifted.append(
                {
                    "workload_id": workload_id,
                    "expected_uid": expected_uid,
                    "current_uid": current_uid,
                }
            )
            continue
        if current_uid:
            state.source_workload_uids[workload_id] = current_uid
        resource_versions[workload_id] = resource_version
        refreshed.append((namespace, kind, name, workload_id, latest))
    if missing:
        return restart_source_failure(
            "RESTART_SOURCE_WORKLOAD_NOT_FOUND",
            "restart source workload is missing: " + ", ".join(sorted(missing)),
            missing,
        )
    if deleting:
        return restart_source_failure(
            "RESTART_SOURCE_WORKLOAD_DELETING",
            "restart source workload is being deleted: " + ", ".join(sorted(deleting)),
            deleting,
        )
    if drifted:
        return restart_source_failure(
            "RESTART_SOURCE_WORKLOAD_IDENTITY_DRIFT",
            "restart source workload identity changed before mutation",
            [item["workload_id"] for item in drifted],
            identity_drift=drifted,
        )
    state.workloads = refreshed
    state.source_resource_versions = resource_versions
    return None
