from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable

from gpu_fault.remote_command_models import (
    RemoteCommandStatus,
    lease_deadline,
)
from gpu_fault.store.shared.cleanup_log import log_cleanup
from gpu_fault.store.shared.errors import (
    NotFoundError,
    WorkflowLeaseError,
)
from gpu_fault.store.shared.remote_commands import (
    stale_fence,
    stale_fence_reason,
    stale_fence_update,
)
from gpu_fault.store.shared.remote_helpers import (
    OPEN_REMOTE_COMMAND_STATUSES,
)
from gpu_fault.store.shared.remote_helpers import (
    remote_command_identity as _remote_command_identity,
)
from gpu_fault.store.shared.remote_helpers import (
    remote_command_stats as _remote_command_stats,
)
from gpu_fault.store.shared.remote_helpers import (
    remote_command_step_space as _remote_command_step_space,
)
from gpu_fault.store.shared.remote_helpers import (
    unclaimed_expiry_update as _unclaimed_expiry_update,
)

if TYPE_CHECKING:
    from gpu_fault.regional import RemoteActionCommand


class MemoryRemoteCommandMixin:
    # Attributes supplied by the composed concrete implementation.
    _remote_commands: Any

    _lock: Any
    _workflows: Any

    def ensure_remote_command(self, command):
        with self._lock:
            existing = self._remote_commands.get(command.command_id)
            if existing is not None:
                if _remote_command_identity(existing) != _remote_command_identity(
                    command
                ):
                    raise ValueError("remote command identity conflict")
                return existing
            self._remote_commands[command.command_id] = command
            return command

    def get_remote_command(self, command_id: str) -> RemoteActionCommand:
        with self._lock:
            command = self._remote_commands.get(command_id)
            if command is None:
                raise NotFoundError(command_id)
            return command  # type: ignore[no-any-return]

    def list_remote_commands(
        self,
        *,
        workflow_request_ids: Iterable[str] | None = None,
    ) -> list[RemoteActionCommand]:
        wanted = None if workflow_request_ids is None else set(workflow_request_ids)
        with self._lock:
            return sorted(
                (
                    item
                    for item in self._remote_commands.values()
                    if wanted is None or item.workflow_request_id in wanted
                ),
                key=lambda item: (item.created_at, item.command_id),
            )

    def find_open_remote_command(
        self,
        workflow_request_id: str,
        step_index: int,
        command_step_space: str,
        *,
        exclude_command_id: str | None = None,
    ) -> RemoteActionCommand | None:
        with self._lock:
            matches = sorted(
                (
                    item
                    for item in self._remote_commands.values()
                    if item.workflow_request_id == workflow_request_id
                    and item.step_index == step_index
                    and item.status in OPEN_REMOTE_COMMAND_STATUSES
                    and item.command_id != exclude_command_id
                    and _remote_command_step_space(item) == command_step_space
                ),
                key=lambda item: (item.created_at, item.command_id),
            )
        return matches[0] if matches else None

    def _open_remote_command_candidates(
        self,
        cluster_id: str,
        execution_owners: set[str] | None,
    ):
        """Narrow the claim to the open backlog of one cluster.

        The claim runs once per poll interval per executor, so its cost
        must track the open backlog, not the number of commands ever
        written. Terminal commands can never be claimed again and other
        clusters' commands are rejected by the filter in
        ``claim_remote_commands`` regardless, so excluding both here is
        purely a narrowing step -- that filter stays authoritative.
        """

        open_statuses = {
            RemoteCommandStatus.PENDING,
            RemoteCommandStatus.WAITING,
            RemoteCommandStatus.LEASED,
        }
        with self._lock:
            return [
                item
                for item in self._remote_commands.values()
                if item.cluster_id == cluster_id
                and item.status in open_statuses
                and item.cancellation_requested_at is None
            ]

    def claim_remote_commands(
        self,
        cluster_id: str,
        executor_id: str,
        *,
        limit: int,
        lease_seconds: int,
        execution_owners: set[str] | None = None,
    ):
        now = datetime.now(timezone.utc)
        claimed = []
        with self._lock:
            candidates = sorted(
                self._open_remote_command_candidates(cluster_id, execution_owners),
                key=lambda item: (
                    item.created_at,
                    item.command_id,
                ),
            )
            for command in candidates:
                workflow = self._workflows.get(command.workflow_request_id)
                if (
                    workflow is not None
                    and workflow.fencing_token != command.fencing_token
                ):
                    continue
                expired = (
                    command.status is RemoteCommandStatus.LEASED
                    and command.lease_expires_at is not None
                    and command.lease_expires_at <= now
                )
                if (
                    command.cluster_id != cluster_id
                    or (
                        execution_owners is not None
                        and command.step.execution_owner not in execution_owners
                    )
                    or (
                        command.status
                        not in {
                            RemoteCommandStatus.PENDING,
                            RemoteCommandStatus.WAITING,
                        }
                        and not expired
                    )
                ):
                    continue
                command = command.model_copy(
                    update={
                        "status": RemoteCommandStatus.LEASED,
                        "lease_owner": executor_id,
                        "lease_token": secrets.token_urlsafe(32),
                        "lease_expires_at": lease_deadline(lease_seconds),
                        "updated_at": now,
                    }
                )
                self._remote_commands[command.command_id] = command
                claimed.append(command)
                if len(claimed) >= limit:
                    break
        return claimed

    def remote_command_stats(self, *, now: datetime | None = None) -> dict[str, Any]:
        with self._lock:
            commands = list(self._remote_commands.values())
        return _remote_command_stats(commands, now=now)

    def expire_unclaimed_remote_commands(
        self,
        *,
        older_than: datetime,
        limit: int,
    ) -> int:
        """Dead-letter PENDING commands nobody ever claimed.

        See UNCLAIMED_DEADLINE_STATUS_SOURCE for why this lands on
        FAILED. Only PENDING is expired: LEASED has a lease that already
        expires back to claimable, and WAITING is an executor actively
        reporting progress.
        """

        now = datetime.now(timezone.utc)
        expired = 0
        with self._lock:
            stale = [
                item
                for item in sorted(
                    self._remote_commands.values(),
                    key=lambda item: (
                        item.created_at,
                        item.command_id,
                    ),
                )
                if item.status is RemoteCommandStatus.PENDING
                and item.created_at <= older_than
            ][:limit]
            for command in stale:
                self._remote_commands[command.command_id] = _unclaimed_expiry_update(
                    command, now
                )
                expired += 1
        return expired

    def expire_stale_fenced_remote_commands(
        self,
        *,
        lease_expired_before: datetime,
        limit: int,
    ) -> int:
        """Fail LEASED commands whose lease lapsed under a stale generation.

        The claim never re-leases such a row and its executor was told to stop
        on renewal; one that never reported (executor gone, lease lapsed for
        longer than ``lease_expired_before`` allows) is closed here so it does
        not stay LEASED for ever (control-plane review 2026-09-08, D-9).
        """

        now = datetime.now(timezone.utc)
        expired = 0
        with self._lock:
            stale = [
                item
                for item in sorted(
                    self._remote_commands.values(),
                    key=lambda item: (item.created_at, item.command_id),
                )
                if item.status is RemoteCommandStatus.LEASED
                and item.lease_expires_at is not None
                and item.lease_expires_at <= lease_expired_before
                and stale_fence(item, self._workflows.get(item.workflow_request_id))
            ][:limit]
            for command in stale:
                self._remote_commands[command.command_id] = stale_fence_update(
                    command,
                    self._workflows.get(command.workflow_request_id),
                    now,
                    swept=True,
                )
                expired += 1
        return expired

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
        with self._lock:
            command = self._remote_commands.get(command_id)
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
            workflow = self._workflows.get(command.workflow_request_id)
            if (
                stale_fence(command, workflow)
                and command.cancellation_requested_at is None
            ):
                # See ``SharedRemoteCommandMixin.renew_remote_command_lease``.
                update["cancellation_requested_at"] = now
                update["cancellation_reason"] = stale_fence_reason(command, workflow)
            command = command.model_copy(update=update)
            self._remote_commands[command_id] = command
            return command

    def cleanup_terminal_remote_commands(
        self,
        *,
        older_than: datetime,
        limit: int,
    ) -> int:
        """Drop SUCCEEDED/FAILED commands past the retention window.

        A remote command embeds the whole workflow request and incident,
        so terminal rows are the largest per-fault objects the regional
        store writes. Nothing ever read them after the step finished --
        the workflow itself is the audit record -- so without this they
        grow without bound and slow every claim.
        """

        terminal = {
            RemoteCommandStatus.SUCCEEDED,
            RemoteCommandStatus.FAILED,
        }
        with self._lock:
            command_ids = [
                item.command_id
                for item in sorted(
                    self._remote_commands.values(),
                    key=lambda item: (
                        item.updated_at,
                        item.command_id,
                    ),
                )
                if item.status in terminal and item.updated_at <= older_than
            ][:limit]
            for command_id in command_ids:
                del self._remote_commands[command_id]
            return log_cleanup("remote_command", command_ids)

    def complete_remote_command(
        self,
        cluster_id: str,
        command_id: str,
        result,
    ):
        now = datetime.now(timezone.utc)
        with self._lock:
            command = self._remote_commands.get(command_id)
            if command is None or command.cluster_id != cluster_id:
                raise NotFoundError(f"{cluster_id}/{command_id}")
            if command.status in {
                RemoteCommandStatus.SUCCEEDED,
                RemoteCommandStatus.FAILED,
            }:
                return command
            workflow = self._workflows.get(command.workflow_request_id)
            if stale_fence(command, workflow):
                # See ``SharedRemoteCommandMixin.complete_remote_command``.
                command = stale_fence_update(command, workflow, now, result=result)
                self._remote_commands[command_id] = command
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
            self._remote_commands[command_id] = command
            return command

    def cancel_remote_commands_for_workflow(
        self, workflow_request_id: str, *, reason: str
    ) -> dict[str, int]:
        now = datetime.now(timezone.utc)
        result = {
            "cancelled": 0,
            "cancellation_requested": 0,
        }
        with self._lock:
            for command_id, command in list(self._remote_commands.items()):
                if command.workflow_request_id != workflow_request_id:
                    continue
                if command.status in {
                    RemoteCommandStatus.PENDING,
                    RemoteCommandStatus.WAITING,
                }:
                    self._remote_commands[command_id] = command.model_copy(
                        update={
                            "status": RemoteCommandStatus.FAILED,
                            "error": reason,
                            "status_source": "workflow-timeout",
                            "lease_owner": None,
                            "lease_token": None,
                            "lease_expires_at": None,
                            "updated_at": now,
                        }
                    )
                    result["cancelled"] += 1
                elif (
                    command.status is RemoteCommandStatus.LEASED
                    and command.cancellation_requested_at is None
                ):
                    self._remote_commands[command_id] = command.model_copy(
                        update={
                            "cancellation_requested_at": now,
                            "cancellation_reason": reason,
                            "updated_at": now,
                        }
                    )
                    result["cancellation_requested"] += 1
        return result

    def cancel_remote_command(self, command_id: str, *, reason: str) -> bool:
        with self._lock:
            command = self._remote_commands.get(command_id)
            if command is None or command.status not in {
                RemoteCommandStatus.PENDING,
                RemoteCommandStatus.WAITING,
            }:
                return False
            now = datetime.now(timezone.utc)
            self._remote_commands[command_id] = command.model_copy(
                update={
                    "status": RemoteCommandStatus.FAILED,
                    "error": reason,
                    "status_source": "workflow-preempted",
                    "lease_owner": None,
                    "lease_token": None,
                    "lease_expires_at": None,
                    "updated_at": now,
                }
            )
            return True
