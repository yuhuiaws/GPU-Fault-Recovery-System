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
