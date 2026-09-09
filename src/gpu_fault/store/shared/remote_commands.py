"""Remote-command row transitions shared by the key/value stores: create if
absent, renew a lease, complete. Each reads one row and writes it back inside
one ``_state_transaction`` keyed by the command.

The generation fence (control-plane review 2026-09-08, D-9, direction B): a
command minted under ``fencing_token`` N whose workflow has since moved to N+1
-- the same ``request_id`` replaced in place -- is dead, but the data plane may
already be executing it. The claim path never hands it out again; here the
renewal tells the executor to stop (``cancellation_requested_at`` on the row
and in the response, lease still extended so it can report), the completion
settles it FAILED with ``status_source="stale-fence"`` instead of refusing the
result, and ``stale_fence_update`` is the one shape both writers and the
``expire_stale_fenced_remote_commands`` sweeps use.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable

from gpu_fault.remote_command_models import (
    BATCHED_RESULTS_KEY,
    RemoteCommandStatus,
    lease_deadline,
)
from gpu_fault.store.shared.errors import (
    NotFoundError,
    WorkflowLeaseError,
)
from gpu_fault.store.shared.primitives import (
    GetOptionalRecord,
    PutRecord,
    StateTransaction,
)
from gpu_fault.store.shared.remote_helpers import (
    OPEN_REMOTE_COMMAND_STATUSES,
    remote_command_step_space,
)
from gpu_fault.store.shared.remote_helpers import (
    remote_command_identity as _remote_command_identity,
)

if TYPE_CHECKING:
    from gpu_fault.regional import RemoteActionCommand

STALE_FENCE_STATUS_SOURCE = "stale-fence"


def stale_fence(command: Any, workflow: Any) -> bool:
    """Whether ``command`` was minted under a generation ``workflow`` has left."""

    return workflow is not None and workflow.fencing_token != command.fencing_token


def stale_fence_reason(command: Any, workflow: Any) -> str:
    return (
        f"workflow {command.workflow_request_id} moved to generation "
        f"{workflow.fencing_token}; this command belongs to generation "
        f"{command.fencing_token}"
    )


def stale_fence_update(
    command: Any,
    workflow: Any,
    now: datetime,
    *,
    result: Any | None = None,
    swept: bool = False,
) -> Any:
    """The FAILED row a stale-fenced command settles into.

    ``result`` is what the executor reported (kept under ``post_stale_fence_*``
    so the node-side outcome is not lost); ``swept`` marks a row closed by the
    sweep because its executor never reported.
    """

    details = dict(command.result_details)
    if result is not None:
        details = {
            **result.details,
            "post_stale_fence_status": result.status.value,
            "post_stale_fence_error": result.error,
        }
    if swept:
        details["stale_fence_swept"] = True
    return command.model_copy(
        update={
            "status": RemoteCommandStatus.FAILED,
            "error": command.cancellation_reason
            or stale_fence_reason(command, workflow),
            "status_source": STALE_FENCE_STATUS_SOURCE,
            "result_details": details,
            "last_lease_owner": command.lease_owner or command.last_lease_owner,
            "lease_owner": None,
            "lease_token": None,
            "lease_expires_at": None,
            "updated_at": now,
        }
    )


def merged_batched_results(
    result_details: dict[str, Any], batched_results: dict[str, Any]
) -> dict[str, Any]:
    """``result_details`` with ``batched_results`` merged over its existing
    per-step entries; the memory store and the shared writer use one shape."""

    existing = result_details.get(BATCHED_RESULTS_KEY)
    return {
        **result_details,
        BATCHED_RESULTS_KEY: {
            **(existing if isinstance(existing, dict) else {}),
            **batched_results,
        },
    }


def covering_compound_command(
    commands: Iterable[RemoteActionCommand],
    step_index: int,
    command_step_space: str,
) -> RemoteActionCommand | None:
    """Pick the compound command covering ``step_index`` from candidates of one
    workflow generation: an open one first, else the newest terminal one."""

    matches = sorted(
        (
            item
            for item in commands
            if item.batched_steps
            and step_index in item.covered_step_indexes
            and remote_command_step_space(item) == command_step_space
        ),
        key=lambda item: (
            item.status in OPEN_REMOTE_COMMAND_STATUSES,
            item.created_at,
            item.command_id,
        ),
    )
    return matches[-1] if matches else None


class SharedRemoteCommandMixin:
    # Attributes supplied by the composed concrete implementation.
    _get_optional: GetOptionalRecord
    _put: PutRecord
    _state_transaction: StateTransaction

    def ensure_remote_command(self, command):
        key = command.command_id
        with self._state_transaction(f"remote_command/{key}"):
            existing = self._get_optional("remote_command", key)
            if existing is not None:
                if _remote_command_identity(existing) != _remote_command_identity(
                    command
                ):
                    raise ValueError("remote command identity conflict")
                return existing
            self._put("remote_command", key, command)
            return command

    def renew_remote_command_lease(
        self,
        cluster_id: str,
        command_id: str,
        executor_id: str,
        lease_token: str,
        *,
        lease_seconds: int,
    ):
        now = datetime.now(timezone.utc)
        with self._state_transaction(f"remote_command/{command_id}"):
            command = self._get_optional("remote_command", command_id)
            if command is None or command.cluster_id != cluster_id:
                raise NotFoundError(f"{cluster_id}/{command_id}")
            if (
                command.status is not RemoteCommandStatus.LEASED
                or command.lease_owner != executor_id
                or command.lease_token != lease_token
                or command.lease_expires_at is None
                or command.lease_expires_at <= now
            ):
                raise WorkflowLeaseError("remote command lease is stale")
            update: dict[str, Any] = {
                "lease_expires_at": lease_deadline(lease_seconds),
                "updated_at": now,
            }
            workflow = self._get_optional("workflow", command.workflow_request_id)
            if (
                stale_fence(command, workflow)
                and command.cancellation_requested_at is None
            ):
                # The executor reads this off the renewal and starts nothing
                # further; the lease is still extended so it can report.
                update["cancellation_requested_at"] = now
                update["cancellation_reason"] = stale_fence_reason(command, workflow)
            command = command.model_copy(update=update)
            self._put("remote_command", command_id, command)
            return command

    def record_remote_command_progress(
        self,
        cluster_id: str,
        command_id: str,
        executor_id: str,
        lease_token: str,
        *,
        batched_results: dict[str, Any],
    ) -> RemoteActionCommand:
        now = datetime.now(timezone.utc)
        with self._state_transaction(f"remote_command/{command_id}"):
            command: RemoteActionCommand | None = self._get_optional(
                "remote_command", command_id
            )
            if command is None or command.cluster_id != cluster_id:
                raise NotFoundError(f"{cluster_id}/{command_id}")
            if (
                command.status is not RemoteCommandStatus.LEASED
                or command.lease_owner != executor_id
                or command.lease_token != lease_token
                or command.lease_expires_at is None
                or command.lease_expires_at <= now
            ):
                raise WorkflowLeaseError("remote command lease is stale")
            command = command.model_copy(
                update={
                    "result_details": merged_batched_results(
                        command.result_details, batched_results
                    ),
                    "updated_at": now,
                }
            )
            self._put("remote_command", command_id, command)
            return command

    def complete_remote_command(
        self,
        cluster_id: str,
        command_id: str,
        result,
    ):
        now = datetime.now(timezone.utc)
        with self._state_transaction(f"remote_command/{command_id}"):
            command = self._get_optional("remote_command", command_id)
            if command is None or command.cluster_id != cluster_id:
                raise NotFoundError(f"{cluster_id}/{command_id}")
            if command.status in {
                RemoteCommandStatus.SUCCEEDED,
                RemoteCommandStatus.FAILED,
            }:
                return command
            workflow = self._get_optional("workflow", command.workflow_request_id)
            if stale_fence(command, workflow):
                # Refusing the result (a 409) left the row LEASED for ever;
                # the generation that owns the node has moved on, so whatever
                # the executor did is recorded and the command ends FAILED.
                command = stale_fence_update(command, workflow, now, result=result)
                self._put("remote_command", command_id, command)
                return command
            if (
                command.status is not RemoteCommandStatus.LEASED
                or command.lease_token != result.lease_token
                or command.lease_expires_at is None
                or (
                    command.lease_expires_at <= now
                    and command.cancellation_requested_at is None
                )
            ):
                raise WorkflowLeaseError(
                    "remote command lease is missing, stale, or changed"
                )
            cancelled = command.cancellation_requested_at is not None
            command = command.model_copy(
                update={
                    "status": (
                        RemoteCommandStatus.FAILED if cancelled else result.status
                    ),
                    "result_details": (
                        {
                            **result.details,
                            "post_cancellation_status": (result.status.value),
                            "post_cancellation_error": result.error,
                        }
                        if cancelled
                        else result.details
                    ),
                    "error": (
                        command.cancellation_reason if cancelled else result.error
                    ),
                    "status_source": (
                        "completed-after-cancellation"
                        if cancelled
                        else result.status_source
                    ),
                    "last_lease_owner": command.lease_owner,
                    "lease_owner": None,
                    "lease_token": None,
                    "lease_expires_at": None,
                    "updated_at": now,
                }
            )
            self._put("remote_command", command_id, command)
            return command
