from __future__ import annotations

import secrets
import sqlite3
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Iterable

from gpu_fault.remote_command_models import (
    RemoteCommandStatus,
    lease_deadline,
)
from gpu_fault.store.shared.errors import (
    NotFoundError,
    WorkflowLeaseError,
)
from gpu_fault.store.shared.remote_helpers import (
    remote_command_identity as _remote_command_identity,
)
from gpu_fault.store.shared.remote_helpers import (
    remote_command_stats as _remote_command_stats,
)
from gpu_fault.store.shared.remote_helpers import (
    unclaimed_expiry_update as _unclaimed_expiry_update,
)

if TYPE_CHECKING:
    from gpu_fault.regional import RemoteActionCommand


class SqliteRemoteCommandMixin:
    """Command-row writers; every per-command key is ``remote_command/<id>``.

    On SQLite ``_state_transaction`` is one ``BEGIN IMMEDIATE`` under a
    process lock, so the key text does not matter here. It matters because
    ``PostgresStore`` inherits ``ensure_remote_command``,
    ``complete_remote_command`` and ``renew_remote_command_lease`` from this
    class and turns the key into an advisory lock; a writer on a different
    key was a lost update there (store review 2026-09-07, item A). The
    bulk sweeps (``unclaimed-expiry``, ``cleanup``) rely on the
    whole-connection transaction and are overridden on Postgres, which
    ``tests/store/test_remote_command_lock_key_convention.py`` checks.
    """

    # Attributes supplied by the composed concrete implementation.
    _db: sqlite3.Connection
    _models: dict[str, Any]
    _delete: Callable[..., Any]
    _get_optional: Callable[..., Any]
    _list: Callable[..., Any]
    _put: Callable[..., Any]
    _state_transaction: Callable[..., Any]

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

    def get_remote_command(self, command_id: str) -> RemoteActionCommand:
        command = self._get_optional("remote_command", command_id)
        if command is None:
            raise NotFoundError(command_id)
        return command  # type: ignore[no-any-return]

    def list_remote_commands(
        self,
        *,
        workflow_request_ids: Iterable[str] | None = None,
    ) -> list[RemoteActionCommand]:
        # Narrow in SQL (F-J5): the reconcile transactions call this while
        # holding the workflow's lock, so the read must scale with the
        # requested workflows, not with every command ever written.
        query = "SELECT payload FROM objects WHERE kind='remote_command'"
        parameters: list[Any] = []
        if workflow_request_ids is not None:
            wanted = sorted(set(workflow_request_ids))
            if not wanted:
                return []
            placeholders = ", ".join("?" for _ in wanted)
            query += (
                " AND json_extract(payload, '$.workflow_request_id')"
                f" IN ({placeholders})"
            )
            parameters.extend(wanted)
        rows = self._db.execute(query, parameters).fetchall()
        model = self._models["remote_command"]
        return sorted(
            (model.model_validate_json(row[0]) for row in rows),
            key=lambda item: (item.created_at, item.command_id),
        )

    def _open_remote_command_candidates(
        self,
        cluster_id: str,
        execution_owners: set[str] | None,
    ):
        """Narrow the claim to the open backlog of one cluster.

        The claim runs once every poll interval per executor, so it must
        not grow with the number of commands ever written. Terminal
        commands can never be claimed again, and another cluster's
        commands are rejected by the filter below anyway, so both are
        excluded before any row is decoded. Subclasses with a real query
        planner narrow this further in SQL; the filter in
        ``claim_remote_commands`` stays authoritative either way.
        """

        open_statuses = {
            RemoteCommandStatus.PENDING,
            RemoteCommandStatus.WAITING,
            RemoteCommandStatus.LEASED,
        }
        return [
            item
            for item in self._list("remote_command")
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
        candidates = sorted(
            self._open_remote_command_candidates(cluster_id, execution_owners),
            key=lambda item: (
                item.created_at,
                item.command_id,
            ),
        )
        for candidate in candidates:
            if len(claimed) >= limit:
                break
            with self._state_transaction(f"remote_command/{candidate.command_id}"):
                command = self._get_optional("remote_command", candidate.command_id)
                if command is None:
                    continue
                workflow = self._get_optional("workflow", command.workflow_request_id)
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
                self._put(
                    "remote_command",
                    command.command_id,
                    command,
                )
                claimed.append(command)
        return claimed

    def remote_command_stats(self, *, now: datetime | None = None) -> dict[str, Any]:
        return _remote_command_stats(self._list("remote_command"), now=now)

    def expire_unclaimed_remote_commands(
        self,
        *,
        older_than: datetime,
        limit: int,
    ) -> int:
        now = datetime.now(timezone.utc)
        expired = 0
        with self._state_transaction("remote_command/unclaimed-expiry"):
            stale = [
                item
                for item in sorted(
                    self._list("remote_command"),
                    key=lambda item: (
                        item.created_at,
                        item.command_id,
                    ),
                )
                if item.status is RemoteCommandStatus.PENDING
                and item.created_at <= older_than
            ][:limit]
            for command in stale:
                self._put(
                    "remote_command",
                    command.command_id,
                    _unclaimed_expiry_update(command, now),
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

    def cleanup_terminal_remote_commands(
        self,
        *,
        older_than: datetime,
        limit: int,
    ) -> int:
        terminal = {
            RemoteCommandStatus.SUCCEEDED,
            RemoteCommandStatus.FAILED,
        }
        with self._state_transaction("remote_command/cleanup"):
            command_ids = [
                item.command_id
                for item in sorted(
                    self._list("remote_command"),
                    key=lambda item: (
                        item.updated_at,
                        item.command_id,
                    ),
                )
                if item.status in terminal and item.updated_at <= older_than
            ][:limit]
            for command_id in command_ids:
                self._delete("remote_command", command_id)
            return len(command_ids)

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

    def cancel_remote_commands_for_workflow(
        self, workflow_request_id: str, *, reason: str
    ) -> dict[str, int]:
        result = {
            "cancelled": 0,
            "cancellation_requested": 0,
        }
        for candidate in self._list("remote_command"):
            if (
                candidate.workflow_request_id != workflow_request_id
                or candidate.status
                in {
                    RemoteCommandStatus.SUCCEEDED,
                    RemoteCommandStatus.FAILED,
                }
            ):
                continue
            with self._state_transaction(f"remote_command/{candidate.command_id}"):
                command = self._get_optional("remote_command", candidate.command_id)
                if (
                    command is None
                    or command.workflow_request_id != workflow_request_id
                ):
                    continue
                now = datetime.now(timezone.utc)
                if command.status in {
                    RemoteCommandStatus.PENDING,
                    RemoteCommandStatus.WAITING,
                }:
                    command = command.model_copy(
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
                    command = command.model_copy(
                        update={
                            "cancellation_requested_at": now,
                            "cancellation_reason": reason,
                            "updated_at": now,
                        }
                    )
                    result["cancellation_requested"] += 1
                else:
                    continue
                self._put(
                    "remote_command",
                    command.command_id,
                    command,
                )
        return result

    def cancel_remote_command(self, command_id: str, *, reason: str) -> bool:
        with self._state_transaction(f"remote_command/{command_id}"):
            command = self._get_optional("remote_command", command_id)
            if command is None or command.status not in {
                RemoteCommandStatus.PENDING,
                RemoteCommandStatus.WAITING,
            }:
                return False
            now = datetime.now(timezone.utc)
            self._put(
                "remote_command",
                command_id,
                command.model_copy(
                    update={
                        "status": RemoteCommandStatus.FAILED,
                        "error": reason,
                        "status_source": "workflow-preempted",
                        "lease_owner": None,
                        "lease_token": None,
                        "lease_expires_at": None,
                        "updated_at": now,
                    }
                ),
            )
            return True
