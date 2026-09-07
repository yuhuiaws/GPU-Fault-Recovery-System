"""Remote-command row transitions shared by the key/value stores: create if
absent, renew a lease, complete. Each reads one row and writes it back inside
one ``_state_transaction`` keyed by the command."""

from __future__ import annotations

from datetime import datetime, timezone

from gpu_fault.remote_command_models import (
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
    remote_command_identity as _remote_command_identity,
)


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
            command = command.model_copy(
                update={
                    "lease_expires_at": lease_deadline(lease_seconds),
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
            workflow = self._get_optional("workflow", command.workflow_request_id)
            if workflow is not None and workflow.fencing_token != command.fencing_token:
                raise WorkflowLeaseError("remote command fencing token is stale")
            if command.status in {
                RemoteCommandStatus.SUCCEEDED,
                RemoteCommandStatus.FAILED,
            }:
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
