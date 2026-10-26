"""Attribute logs using observed containers, never a guessed PID from free text."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from gpu_fault.mce_grading import MceGrade, grade_mce_line
from gpu_fault.models import RecoveryAction, Severity, WorkloadState
from gpu_fault.nccl_events import event_kind
from gpu_fault.store.shared.errors import NotFoundError


def log_rule_decision(
    store: Any, batch: Any, entry: Any, rule: Any
) -> tuple[Severity, RecoveryAction] | None:
    if rule.action is RecoveryAction.QUARANTINE:
        if entry.source == "journal" and entry.fields.get("_TRANSPORT") != "kernel":
            return None
        if not current_kernel_evidence(store, batch, entry):
            return None
    if rule.category.value == "MCE":
        # The kernel prints a corrected record the same way as an uncorrected
        # one; ``mce_grading`` reads the IA32_MCi_STATUS bits and the kernel's
        # own words (quoted there). Corrected -> diagnostics, never isolation.
        # A detail line (TSC/PROCESSOR/RIP, the GHES section body, ...) has
        # no grade of its own: its header line already produced the finding.
        grade = grade_mce_line(entry.message)
        if grade is MceGrade.CORRECTED:
            return Severity.WARNING, RecoveryAction.RUN_DIAGNOSTICS
        if grade is MceGrade.DETAIL:
            return None
    return rule.severity, rule.action


def current_kernel_evidence(store: Any, batch: Any, entry: Any) -> bool:
    age = (batch.collected_at - entry.observed_at).total_seconds()
    if not -30 <= age <= 900:
        return False
    try:
        agent = store.get_agent(batch.cluster_id, batch.node_id)
    except NotFoundError:
        agent = None
    expected = getattr(agent, "boot_id", None) or entry.fields.get("_COLLECTOR_BOOT_ID")
    actual = entry.fields.get("_BOOT_ID")
    # Legacy trusted API producers have no boot metadata; once a generation is
    # known, absent or conflicting evidence must not isolate the current boot.
    return bool(
        not expected
        or (
            bool(actual)
            and actual.replace("-", "").lower() == expected.replace("-", "").lower()
        )
    )


def _matches_process(entry: Any, container: Any) -> bool:
    fields = entry.fields
    if fields.get("pod_uid"):
        return bool(
            fields["pod_uid"] == container.pod_uid
            and fields.get("container_name") == container.container_name
        )
    if fields.get("CONTAINER_ID_FULL"):
        return bool(
            fields["CONTAINER_ID_FULL"]
            == (container.container_id or "").split("://")[-1]
        )
    if fields.get("_PID", "").isdigit():
        return bool(int(fields["_PID"]) == container.host_pid)
    return False


def _resolve_log_attempt(store: Any, batch: Any, entry: Any) -> tuple[Any, Any]:
    candidates = []
    for attempt in store.list_attempt_observations(batch.cluster_id):
        if not -30 <= (batch.collected_at - attempt.observed_at).total_seconds() <= 120:
            continue
        if attempt.started_at and entry.observed_at < attempt.started_at:
            continue
        if not -120 <= (entry.observed_at - attempt.observed_at).total_seconds() <= 120:
            continue
        containers = [
            item for item in attempt.containers if item.node_id == batch.node_id
        ]
        matched = [item for item in containers if _matches_process(entry, item)]
        explicit_identity = any(
            entry.fields.get(key) for key in ("pod_uid", "CONTAINER_ID_FULL", "_PID")
        )
        if explicit_identity:
            if len(matched) != 1:
                continue
        elif (
            entry.source == "training-log"
            or attempt.workload_phase.value not in {"PENDING", "RUNNING"}
            or not set(batch.affected_workload_ids).intersection(attempt.workload_ids)
            or not any(not item.terminated for item in containers)
        ):
            continue
        candidates.append((attempt, matched[0] if matched else None))
    return candidates[0] if len(candidates) == 1 else (None, None)


def log_event_id(batch: Any, entry: Any, safe: Callable[[str], str]) -> str:
    """Bind file/cursor identity to tenant and node, not a collision-prone prefix."""
    digest = hashlib.sha256(
        json.dumps(
            [batch.cluster_id, batch.node_id, entry.entry_id],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:32]
    return f"log-{safe(batch.cluster_id)}-{safe(batch.node_id)}-{digest}"


def log_context(
    store: Any, batch: Any, entry: Any, action: RecoveryAction
) -> dict[str, Any]:
    attempt, process = _resolve_log_attempt(store, batch, entry)
    parameters: dict[str, Any] = {
        "stop_on_isolation": action is RecoveryAction.QUARANTINE
    }
    if action is RecoveryAction.NO_ACTION:
        parameters.update(
            recovery_owner="COMPLETION_WATCHER",
            capture_process_state=False,
            evidence_only=True,
        )
        kind = event_kind(entry.message)
        if kind:
            parameters["nccl_event_kind"] = kind
    result: dict[str, Any] = {
        "workload_state": WorkloadState.UNKNOWN,
        "affected_workload_ids": [],
        "diagnostic_parameters": parameters,
        "policy_source": (
            "SITE_APPLICATION_LOG"
            if action in {RecoveryAction.RUN_DIAGNOSTICS, RecoveryAction.NO_ACTION}
            else "SITE_KERNEL_LOG"
        ),
    }
    if attempt is None:
        parameters["attribution"] = "UNRESOLVED"
        return result
    if attempt.started_at is not None and entry.observed_at < attempt.started_at:
        parameters["attribution"] = "PREVIOUS_ATTEMPT"
        return result
    containers = [item for item in attempt.containers if item.node_id == batch.node_id]
    parameters.update(
        attribution="OBSERVED_PROCESS" if process else "UNIQUE_MANAGED_ATTEMPT",
        process_state=(
            "EXITED"
            if process is not None and process.terminated
            else "LIVE"
            if process is not None
            else "UNKNOWN"
        ),
        log_source=entry.source,
    )
    if process is not None and not process.terminated and process.host_pid is not None:
        if action is RecoveryAction.NO_ACTION:
            parameters["observed_host_pid"] = process.host_pid
        else:
            parameters["target_pids_by_node"] = {batch.node_id: [process.host_pid]}
    if action is RecoveryAction.RUN_DIAGNOSTICS:
        parameters["capture_process_state"] = any(
            not item.terminated for item in containers
        )
        parameters["diagnostic_reason"] = "APPLICATION_ERROR"
    result.update(
        job_id=attempt.job_id,
        attempt_id=attempt.attempt_id,
        affected_workload_ids=attempt.workload_ids,
        workload_state=(
            WorkloadState.ACTIVE
            if any(not item.terminated for item in containers)
            else WorkloadState.IDLE
        ),
    )
    return result
