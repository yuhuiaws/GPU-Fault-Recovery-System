"""``COLLECTOR_OUTBOX_MAINTENANCE``: the collector outbox command, run remotely.

``gpu-fault-collector outbox {stats,list,requeue-dead}`` reads or requeues the
dead letters a collector buffered in ``/var/lib/gpu-fault/outbox/<c>.ndjson``
(ARCH-G2). Until this operation existed an administrator had to log on to the
GPU node to run it; ``gpu-fault-admin collector-outbox`` now carries the same
command here as a NODE_ACTION step, and this handler runs the outbox file layer
in-process -- never a subprocess -- with the CLI's safety properties:

* metadata only on the wire: ``stats`` and ``list`` never carry a payload
  byte, and ``requeue-dead`` returns counts;
* the lock is always taken strictly (there is no ``--force``: the remote path
  cannot see whether the collector is stopped), and a lock still held after
  the bounded poll is a *structured* failure naming the recorded holder, not a
  retry -- ``OutboxLockUnavailable`` is an ``OSError`` the executor would
  otherwise resubmit;
* records whose payload was truncated to a digest stay dead and are counted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gpu_fault.collectors.outbox_file import (
    OUTBOX_LOCK_FORCE_ADVICE,
    OutboxFile,
    OutboxLockUnavailable,
)
from gpu_fault.collectors.outbox_maintenance import (
    ACTION_LIST,
    ACTION_REQUEUE_DEAD,
    ACTION_STATS,
    OutboxMaintenanceRequest,
    describe_record,
    requeue_error_prefix,
)
from gpu_fault.node_agent.protocol import NodeActionCommand

#: How the refusal tells the operator where ``--force`` still lives.
NODE_LOCAL_FALLBACK = (
    "the remote path never rewrites without the lock; if the collector is "
    "stopped and left its lock behind, run the node-local "
    "`gpu-fault-collector outbox --collector <c> requeue-dead --yes --force`"
)


class CollectorOutboxRefused(RuntimeError):
    """A maintenance request the node would not or could not carry out.

    Non-retryable on purpose (a plain ``RuntimeError`` to the executor): a held
    lock, a bad parameter or an unreadable outbox does not get better by
    resubmitting the same command. ``action_details`` travels to the step's
    ``node_results`` so the CLI can print the holder line or the file error.
    """

    def __init__(self, message: str, action_details: dict[str, Any]) -> None:
        super().__init__(message)
        self.action_details = action_details


class CollectorOutboxOperationsMixin:
    # Attribute supplied by the composed concrete implementation.
    collector_outbox_directory: Path

    def _collector_outbox_maintenance(
        self, command: NodeActionCommand
    ) -> dict[str, Any]:
        try:
            request = OutboxMaintenanceRequest.from_parameters(command.parameters)
        except ValueError as exc:
            raise CollectorOutboxRefused(
                f"invalid collector outbox request: {exc}", {"invalid_request": True}
            ) from exc
        path = Path(self.collector_outbox_directory) / request.outbox_name
        # The role is what a refused collector or colleague reads out of
        # ``.lock`` while this agent holds it: the remote command, by action.
        outbox = OutboxFile(path, role=f"node-agent:{request.action}")
        details: dict[str, Any] = {
            "collector": request.collector,
            "action": request.action,
            "outbox_path": str(path),
            "outbox_exists": path.exists(),
            "path_filter": request.path,
        }
        try:
            if request.action == ACTION_STATS:
                details.update(self._outbox_stats(outbox))
            elif request.action == ACTION_LIST:
                details["records"] = self._outbox_list(outbox, request.path)
            elif request.action == ACTION_REQUEUE_DEAD:
                details.update(self._outbox_requeue(outbox, command, request))
        except OutboxLockUnavailable as exc:
            # The file layer's refusal ends in the CLI's ``--force`` advice;
            # there is no such flag here, so the fallback names where it lives.
            refusal = str(exc).replace(OUTBOX_LOCK_FORCE_ADVICE, "")
            raise CollectorOutboxRefused(
                f"collector outbox lock is held: {refusal}; {NODE_LOCAL_FALLBACK}",
                {
                    **details,
                    "lock_unavailable": True,
                    "lock_path": str(outbox.lock_path),
                    "lock_holder": _holder_from_refusal(str(exc)),
                },
            ) from exc
        except OSError as exc:
            # A file the agent cannot read or rewrite is not transient either.
            raise CollectorOutboxRefused(
                f"cannot read or rewrite the collector outbox {path}: {exc}",
                {**details, "outbox_error": f"{type(exc).__name__}: {exc}"},
            ) from exc
        return details

    @staticmethod
    def _outbox_stats(outbox: OutboxFile) -> dict[str, Any]:
        records = outbox.read()
        stats = OutboxFile.summarize(records)
        # ``unlocked_writes_total`` is the collector's own in-process counter;
        # read from another process it is always 0 and would mislead.
        stats.pop("unlocked_writes_total", None)
        stats.pop("evictions_total", None)
        stats["payload_truncated"] = sum(
            1 for record in records if record.get("payload_truncated")
        )
        return {"stats": stats, "lock_holder": outbox.lock_holder()}

    @staticmethod
    def _outbox_list(
        outbox: OutboxFile, path_filter: str | None
    ) -> list[dict[str, Any]]:
        return [
            describe_record(index, record)
            for index, record in enumerate(outbox.read())
            if path_filter is None or record.get("path") == path_filter
        ]

    @staticmethod
    def _outbox_requeue(
        outbox: OutboxFile,
        command: NodeActionCommand,
        request: OutboxMaintenanceRequest,
    ) -> dict[str, Any]:
        operator = command.parameters.get("operator")
        reference = command.parameters.get("reference")
        report = outbox.requeue_dead_report(
            path_filter=request.path,
            require_lock=True,
            error_prefix=requeue_error_prefix(
                operator if isinstance(operator, str) else None,
                reference if isinstance(reference, str) else None,
            ),
        )
        return dict(report)


def _holder_from_refusal(message: str) -> str:
    """The holder clause of an ``OutboxLockUnavailable`` message, for a field.

    The refusal reads ``... after 4.5s, recorded holder pid 12 (collector),
    alive, since ...: retry, stop the collector, or pass --force ...``; the
    field keeps the part between the wait and the CLI advice.
    """

    _head, separator, tail = message.partition("s, ")
    if not separator:
        return message
    return tail.split(": retry", 1)[0]
