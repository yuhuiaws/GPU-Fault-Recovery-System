from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from gpu_fault.admin.atomic_json import write_json_atomic as _write_document

if __package__:
    from .acceptance_scope import scoped_case_evidence
else:
    from acceptance_scope import scoped_case_evidence


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Write one case evidence document, scoped to what the run may record.

    The scoping is the part specific to acceptance evidence -- a selective run must
    not claim the formal sequence -- and it has to happen before the bytes are
    written, not on read, because the document on disk is the artifact an auditor
    reads. The writing itself is the same all-or-nothing rename the admin tooling
    uses for its state.
    """

    _write_document(path, scoped_case_evidence(value))


class EvidenceRecorder:
    def __init__(
        self,
        path: Path,
        *,
        case_id: str,
        inputs: dict[str, Any],
    ) -> None:
        self.path = path
        if path.exists():
            document = json.loads(path.read_text(encoding="utf-8"))
            if document.get("case_id") != case_id:
                raise RuntimeError("evidence file belongs to another case")
            if document.get("inputs") != inputs:
                raise RuntimeError("evidence inputs differ from the existing run")
            self.document = document
        else:
            self.document = {
                "schema_version": 1,
                "case_id": case_id,
                "status": "RUNNING",
                "started_at": utc_now(),
                "updated_at": utc_now(),
                "inputs": inputs,
                "stages": {},
            }
            write_json_atomic(self.path, self.document)

    def stage(
        self,
        name: str,
        operation: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        existing = self.document["stages"].get(name)
        if existing is not None:
            return dict(existing)
        result = operation()
        self.document["stages"][name] = result
        self.document["updated_at"] = utc_now()
        write_json_atomic(self.path, self.document)
        return result

    def complete(self) -> dict[str, Any]:
        self.document["status"] = "COMPLETED"
        self.document["completed_at"] = utc_now()
        self.document["updated_at"] = utc_now()
        write_json_atomic(self.path, self.document)
        return self.document

    def fail(self, exc: BaseException) -> None:
        self.document["status"] = "FAILED"
        self.document["updated_at"] = utc_now()
        self.document["error"] = f"{type(exc).__name__}: {exc}"
        write_json_atomic(self.path, self.document)


def processor_queue_backlog(queue: dict[str, Any] | None) -> int:
    """The processor work a destructive preflight must wait out.

    The store snapshot reports two depths. ``fault_backlog_depth`` counts the
    reserved tier only -- control-plane actions and device events, the work a
    case's injected event would queue behind and a control-worker roll could
    disrupt. Total ``depth`` also counts routine telemetry (gpu-inventory,
    evidence), which is idempotent across a roll and can livelock one lane on a
    stale fencing token for ~120 s, so a gate on total depth flaps every
    preflight on a healthy cluster (DESTR-018 attempt 5, DESTR-019 attempt 2,
    2026-09-08). Gate on the fault tier when the snapshot carries it; an older
    snapshot without the reading falls back to total depth.
    """

    values = queue or {}
    if "fault_backlog_depth" in values:
        return int(values.get("fault_backlog_depth") or 0)
    return int(values.get("depth") or 0)


# What ``kubectl exec`` prints when its target stopped being a running replica
# between a ``ready_pods`` listing and the exec: a deleted Pod (NotFound), a
# Pod whose process already exited 0 on SIGTERM (phase Succeeded), or the
# kubelet's other wordings for the same moment. Every env window rolls its
# Deployment, so a survey taken during the roll meets these routinely
# (control-worker 2026-09-08 attempts 7/8, cluster-executor 2026-09-08
# DESTR-014 attempt 3).
VANISHED_REPLICA_MARKERS = (
    "not found",
    "notfound",
    "completed pod",
    "is not running",
    "not running",
    "terminating",
)


def replica_vanished(error: BaseException) -> bool:
    """Whether an exec failed because its Pod is no longer a running replica,
    as opposed to the read itself failing."""

    text = str(error).lower()
    return any(marker in text for marker in VANISHED_REPLICA_MARKERS)
