from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tools import codex_acceptance as backend


def manual_case(case_id: str = "case-a") -> dict[str, Any]:
    return {
        "id": case_id,
        "title": "Local evidence review",
        "risk": "non-destructive",
        "problem": "Verify the supplied observation.",
        "injection": "Inspect only the supplied observation.",
        "expected": ["The observation has the requested identity."],
        "automation": "manual",
        "procedure": "Review the existing local evidence.",
    }


def regional_case(case_id: str = "case-a") -> dict[str, Any]:
    return {
        "id": case_id,
        "title": "Local dependency review",
        "phase": "BOOT",
        "risk": "non-destructive",
        "summary": "Inspect a local prerequisite.",
    }


def analysis(case_id: str = "case-a") -> dict[str, Any]:
    evidence = [
        {"source": "unit-fixture", "observation": "Synthetic local observation."}
    ]
    return {
        "case_id": case_id,
        "status": "PASS",
        "summary": "Local observation reviewed.",
        "failure_details": [],
        "reproduction": [],
        "evidence": evidence,
        "test_process": [
            {
                "step_id": "inspect",
                "kind": "evidence-check",
                "description": "Inspect the provided data.",
                "depends_on": [],
                "expected": "The data matches the requested identity.",
                "executor_ref": "provided-observations",
                "status": "PASS",
                "evidence": evidence,
            }
        ],
        "affected_dependents": [],
        "blockers": [],
        "human_actions": [],
    }


def dependency(case_id: str = "case-a") -> dict[str, Any]:
    return {
        "case_id": case_id,
        "depends_on": [],
        "locks": [{"resource": "unit-evidence", "mode": "shared"}],
        "executor": "codex-read-only",
        "confidence": 0.75,
        "rationale": "This is an untrusted local proposal.",
    }


def review(case_id: str = "case-a") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "review": {
            "case_id": case_id,
            "status": "PASS",
            "summary": "Local review complete.",
            "findings": ["The observation matches."],
            "missing_evidence": [],
            "contradictions": [],
            "human_actions": [],
        },
    }


class AcceptanceTransport:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.raw: bytes | None = None
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.schemas: list[dict[str, Any]] = []
        self.returncode = 0
        self.error: Exception | None = None
        self.write_output = True

    def run(
        self, arguments: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        assert arguments[0] == "fake-codex-no-execution"
        self.calls.append((arguments, kwargs))
        if self.error is not None:
            raise self.error
        schema = Path(arguments[arguments.index("--output-schema") + 1])
        self.schemas.append(json.loads(schema.read_text()))
        output = Path(arguments[arguments.index("--output-last-message") + 1])
        if self.write_output:
            output.write_bytes(
                self.raw
                if self.raw is not None
                else json.dumps(self.payload).encode("utf-8")
            )
        return subprocess.CompletedProcess(arguments, self.returncode, "", "")


def backend_with_output(
    monkeypatch: Any, root: Path, payload: Any, **options: Any
) -> tuple[backend.CodexAcceptanceBackend, AcceptanceTransport]:
    transport = AcceptanceTransport(payload)
    monkeypatch.setattr(
        backend,
        "subprocess",
        SimpleNamespace(
            run=transport.run,
            DEVNULL=subprocess.DEVNULL,
            TimeoutExpired=subprocess.TimeoutExpired,
        ),
    )
    instance = backend.CodexAcceptanceBackend(
        root,
        codex_binary="fake-codex-no-execution",
        environment={"HOME": str(root), "PATH": "/usr/bin:/bin"},
        **options,
    )
    return instance, transport
