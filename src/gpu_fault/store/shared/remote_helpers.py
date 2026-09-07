from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from gpu_fault.remote_command_models import RemoteCommandStatus

if TYPE_CHECKING:
    from gpu_fault.models import WorkflowRequest
    from gpu_fault.regional import RemoteActionCommand

UNCLAIMED_DEADLINE_STATUS_SOURCE = "unclaimed-deadline-exceeded"

# Which step list a remote command was compiled from. Safety and official
# steps share indexes (F-C2), so the open-command invariant of item D5 is keyed
# by (workflow, step_index, step space); the command carries the workflow it
# was minted from, whose ``safety_only`` flag says which list that was.
COMMAND_STEP_SPACE_OFFICIAL = "official"
COMMAND_STEP_SPACE_SAFETY = "safety"
OPEN_REMOTE_COMMAND_STATUSES = frozenset(
    {
        RemoteCommandStatus.PENDING,
        RemoteCommandStatus.LEASED,
        RemoteCommandStatus.WAITING,
    }
)


def workflow_step_space(workflow: WorkflowRequest) -> str:
    """The step space a command minted from ``workflow`` belongs to."""

    return (
        COMMAND_STEP_SPACE_SAFETY
        if workflow.executes_safety_steps
        else COMMAND_STEP_SPACE_OFFICIAL
    )


def remote_command_step_space(command: RemoteActionCommand) -> str:
    return workflow_step_space(command.workflow)


LEGACY_EXECUTOR_SAFETY_REJECTION_ERRORS = frozenset(
    {
        "ValueError: node is already isolated by another incident/token",
        "ValueError: node is controlled by a newer workflow generation",
    }
)


def remote_command_identity(command) -> tuple:
    return (
        command.cluster_id,
        command.workflow_request_id,
        command.step_index,
        command.fencing_token,
        command.idempotency_key,
        command.step.operation,
        command.step.execution_owner,
        tuple(command.step.node_ids),
        tuple(command.step.gpu_uuids),
        tuple(command.step.workload_ids),
    )


def unclaimed_expiry_update(command, now: datetime):
    age = max(0.0, (now - command.created_at).total_seconds())
    return command.model_copy(
        update={
            "status": RemoteCommandStatus.FAILED,
            "status_source": UNCLAIMED_DEADLINE_STATUS_SOURCE,
            "error": (
                "no cluster executor claimed this command within "
                f"{age:.0f}s; execution_owner="
                f"{command.step.execution_owner} is not advertised by "
                "any executor in this cluster"
            ),
            "result_details": {
                **command.result_details,
                "unclaimed_age_seconds": round(age, 3),
                "unclaimed_execution_owner": (command.step.execution_owner),
            },
            "lease_owner": None,
            "lease_token": None,
            "lease_expires_at": None,
            "updated_at": now,
        }
    )


def remote_command_stats(commands, *, now: datetime | None = None) -> dict:
    observed_at = now or datetime.now(timezone.utc)
    pending_states = {
        RemoteCommandStatus.PENDING,
        RemoteCommandStatus.WAITING,
        RemoteCommandStatus.LEASED,
    }
    by_status: dict[str, int] = {status.value: 0 for status in RemoteCommandStatus}
    by_cluster_status: dict[str, dict[str, int]] = {}
    unclaimed_age_by_cluster: dict[str, float] = {}
    open_by_cluster: dict[str, int] = {}
    internal_errors = 0
    internal_error_last_seen = 0.0
    unclaimed_expired = 0
    for command in commands:
        by_status[command.status.value] = by_status.get(command.status.value, 0) + 1
        cluster_counts = by_cluster_status.setdefault(command.cluster_id, {})
        cluster_counts[command.status.value] = (
            cluster_counts.get(command.status.value, 0) + 1
        )
        if (
            command.status_source == "executor-internal-error"
            and command.error not in LEGACY_EXECUTOR_SAFETY_REJECTION_ERRORS
        ):
            internal_errors += 1
            error_at = getattr(command, "updated_at", None) or command.created_at
            internal_error_last_seen = max(
                internal_error_last_seen,
                error_at.timestamp(),
            )
        if command.status_source == UNCLAIMED_DEADLINE_STATUS_SOURCE:
            unclaimed_expired += 1
        if command.status not in pending_states:
            continue
        open_by_cluster[command.cluster_id] = (
            open_by_cluster.get(command.cluster_id, 0) + 1
        )
        if (
            command.status is RemoteCommandStatus.PENDING
            and command.lease_owner is None
        ):
            age = max(
                0.0,
                (observed_at - command.created_at).total_seconds(),
            )
            unclaimed_age_by_cluster[command.cluster_id] = max(
                unclaimed_age_by_cluster.get(command.cluster_id, 0.0),
                age,
            )
    return {
        "total": len(commands),
        "by_status": by_status,
        # The (cluster, status) cells behind ``by_status`` (F-D12): the
        # per-cluster gauge reads them, the fleet total stays ``by_status``.
        "by_cluster_status": by_cluster_status,
        "open_by_cluster": open_by_cluster,
        "oldest_unclaimed_age_seconds_by_cluster": (unclaimed_age_by_cluster),
        "oldest_unclaimed_age_seconds": max(
            unclaimed_age_by_cluster.values(),
            default=0.0,
        ),
        "executor_internal_error_total": internal_errors,
        "executor_internal_error_last_seen_timestamp_seconds": (
            internal_error_last_seen
        ),
        "unclaimed_expired_total": unclaimed_expired,
    }
