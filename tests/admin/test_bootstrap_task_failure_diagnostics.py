from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Mapping

import pytest

from gpu_fault.admin import diagnostics
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    BootstrapState,
    run_parallel,
)


def test_failure_is_saved_before_an_independent_task_finishes(tmp_path: Path) -> None:
    saved = threading.Event()
    error = BootstrapError("replica source primary is not ready")

    class ObservedState(BootstrapState):
        def finish_task(self, name: str, value: Any, report: Mapping[str, Any]) -> None:
            super().finish_task(name, value, report)
            if name == "aurora":
                saved.set()

    state = ObservedState(tmp_path / "bootstrap.json", site_id="example")

    def fail() -> None:
        raise error

    def independent() -> dict[str, str]:
        assert saved.wait(5), "failed task was not persisted before the graph ended"
        report = json.loads(state.path.read_text())["task_reports"]["aurora"]
        assert report["status"] == "failed"
        assert report["error"] == {
            "type": "BootstrapError",
            "message": "replica source primary is not ready",
        }
        return {"result": "completed"}

    with pytest.raises(BootstrapError) as caught:
        run_parallel({"aurora": fail, "independent": independent}, state=state)

    assert caught.value is error
    assert state.value["completed_tasks"] == ["independent"]


def test_all_failed_tasks_retain_bounded_redacted_causes(tmp_path: Path) -> None:
    state = BootstrapState(tmp_path / "bootstrap.json", site_id="example")
    hidden = ("fixture-password-123", "fixture-token-456")
    errors = {
        "aurora": BootstrapError("password=" + hidden[0]),
        "grafana": ValueError(json.dumps({"detail": "x" * 3000, "token": hidden[1]})),
    }

    def fail(name: str) -> None:
        raise errors[name]

    with pytest.raises((BootstrapError, ValueError)) as caught:
        run_parallel(
            {name: lambda name=name: fail(name) for name in errors}, state=state
        )

    assert caught.value in errors.values()
    contents = state.path.read_text()
    assert all(value not in contents for value in hidden), (
        "task error diagnostics must not retain fixture credentials"
    )
    reports = json.loads(contents)["task_reports"]
    for name, error in errors.items():
        detail = reports[name]["error"]
        assert detail["type"] == type(error).__name__
        assert len(detail["message"]) < 2100
        assert "<redacted>" in detail["message"]
    assert state.value["completed_tasks"] == []


def test_task_failure_type_is_streamed_without_exception_contents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamed: list[str] = []

    def write(text: str, *, final: bool = False) -> bool:
        streamed.append(text)
        return True

    monkeypatch.setattr(diagnostics, "write_diagnostic", write)
    reader = diagnostics.DriverDiagnostics()
    reader.feed("bootstrap task=aurora failure_type=BootstrapError\n")
    assert streamed == ["bootstrap task=aurora failure_type=BootstrapError\n"]
