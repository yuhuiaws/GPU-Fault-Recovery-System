"""The contract of a remote collector-outbox maintenance request (ARCH-G2).

``gpu-fault-collector outbox {stats,list,requeue-dead}`` is the node-local
entry to a collector's dead letters. ``COLLECTOR_OUTBOX_MAINTENANCE`` is the
same command carried to the node as a NODE_ACTION step, so an administrator on
the deploy host does not have to log on to the GPU node. Three parties read
one request shape -- the admin verb that validates it, the workflow builder
that puts it on the step, the node agent handler that executes it -- and this
module is the one place that shape is spelled. It knows nothing about HTTP,
workflows or the outbox file itself.

There is deliberately no ``force`` field: the remote path cannot see whether
the collector is stopped, so it always takes the outbox lock strictly and
reports the recorded holder on refusal. A stopped collector's leftover lock is
the node-local CLI's ``--force`` case, and stays there.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

#: The node collector units that write an outbox: every ``gpu-fault-collector
#: <command>`` the node installer runs (``deploy/systemd/*.service``; the
#: metrics unit runs ``dcgm`` or ``nvidia-smi``). The CPU-side collectors
#: (SQS/Kubernetes HMA, node resources) post from inside the cluster and have
#: no node-local outbox an operator would ever requeue.
OUTBOX_COLLECTORS: tuple[str, ...] = (
    "kernel",
    "dcgm",
    "nvidia-smi",
    "host",
    "logs",
    "fabric-manager",
)
DEFAULT_OUTBOX_DIRECTORY = "/var/lib/gpu-fault/outbox"

ACTION_STATS = "stats"
ACTION_LIST = "list"
ACTION_REQUEUE_DEAD = "requeue-dead"
OUTBOX_ACTIONS: tuple[str, ...] = (ACTION_STATS, ACTION_LIST, ACTION_REQUEUE_DEAD)

#: ``record["error"]`` is quoted at most this far, as the CLI's ``list`` does.
ERROR_PREFIX_LIMIT = 120
#: The outbox record's own error limit; the requeue prefix must fit inside it.
RECORD_ERROR_LIMIT = 500


@dataclass(frozen=True)
class OutboxMaintenanceRequest:
    """One validated ``outbox`` command: which collector, what, confirmed?"""

    collector: str
    action: str
    confirm: bool = False
    path: str | None = None

    def __post_init__(self) -> None:
        if self.collector not in OUTBOX_COLLECTORS:
            raise ValueError(
                f"unknown collector {self.collector!r}; the collectors with an "
                "outbox are " + ", ".join(OUTBOX_COLLECTORS)
            )
        if self.action not in OUTBOX_ACTIONS:
            raise ValueError(
                f"unknown outbox action {self.action!r}; choose one of "
                + ", ".join(OUTBOX_ACTIONS)
            )
        if self.path is not None and (
            not isinstance(self.path, str) or not self.path.startswith("/")
        ):
            raise ValueError("path filter must be a control-plane path such as /v1/...")
        if self.action == ACTION_REQUEUE_DEAD and self.confirm is not True:
            raise ValueError(
                f"{ACTION_REQUEUE_DEAD} replays records the control plane already "
                "rejected; it requires confirm=true (--yes)"
            )

    @property
    def outbox_name(self) -> str:
        return f"{self.collector}.ndjson"

    def as_parameters(self) -> dict[str, Any]:
        """The step parameters the node agent parses back with ``from_parameters``."""

        values: dict[str, Any] = {
            "collector": self.collector,
            "action": self.action,
            "confirm": bool(self.confirm),
        }
        if self.path is not None:
            values["path"] = self.path
        return values

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Any]) -> OutboxMaintenanceRequest:
        """Parse a step's parameters; ``ValueError`` names the first bad field."""

        collector = parameters.get("collector")
        if not isinstance(collector, str):
            raise ValueError("collector parameter is required")
        action = parameters.get("action")
        if not isinstance(action, str):
            raise ValueError("action parameter is required")
        confirm = parameters.get("confirm", False)
        if not isinstance(confirm, bool):
            raise ValueError("confirm parameter must be a boolean")
        path = parameters.get("path")
        if path is not None and not isinstance(path, str):
            raise ValueError("path filter must be a string")
        return cls(collector=collector, action=action, confirm=confirm, path=path)


def requeue_error_prefix(operator: str | None, reference: str | None) -> str:
    """What a requeued record's ``error`` starts with: who asked, and why.

    The CLI writes ``requeued by operator: <previous error>``; the remote path
    adds the identity and reference the audit trail already carries so the
    record on the node names them too.
    """

    if not operator:
        return "requeued by operator: "
    who = operator if not reference else f"{operator} ({reference})"
    return f"requeued by operator: {who}: "


def describe_record(index: int, record: Mapping[str, Any]) -> dict[str, Any]:
    """The metadata of one buffered record; never its payload (ARCH-G2).

    ``request_id`` is the event's idempotency key when the payload still carries
    one, or the key a truncated payload's digest kept; ``None`` when neither
    identifies the event.
    """

    payload = record.get("payload")
    request_id: str | None = None
    if isinstance(payload, dict):
        if record.get("payload_truncated"):
            key = payload.get("payload_event_key")
            request_id = key if isinstance(key, str) else None
        else:
            # Late import: ``sinks`` is the HTTP layer and this module is not.
            from gpu_fault.collectors.sinks import event_idempotency_key

            request_id = event_idempotency_key(payload)
    error = str(record.get("error") or "")[:ERROR_PREFIX_LIMIT].replace("\n", " ")
    replayable = bool(record.get("replayable"))
    return {
        "index": index,
        "request_id": request_id,
        "path": record.get("path"),
        "status": "replayable" if replayable else "dead",
        "replayable": replayable,
        "payload_truncated": bool(record.get("payload_truncated")),
        "error": error,
        "failed_at": record.get("failed_at"),
    }
