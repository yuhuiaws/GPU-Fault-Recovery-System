"""Execute the generated credential Job program against the shipped runtime API."""

from __future__ import annotations

import pytest

from gpu_fault import aurora_credential_refresh as REFRESH
from gpu_fault_release.regional_release_diff import (
    build_execution_plan,
    diff_from_changed,
)
from gpu_fault_release.regional_release_prerequisite_repair import (
    prepare_prerequisite_repair,
)
from gpu_fault_release.regional_release_runtime_identity import CONTROL_PLANE_PYTHON
from tests.regional._prerequisite_repair_support import repair_release


@pytest.fixture
def credential_program(monkeypatch: pytest.MonkeyPatch) -> str:
    release = repair_release(monkeypatch)
    diff = diff_from_changed({"aurora_refresh_drift", "schema_manifests"})
    prepare_prerequisite_repair(release, diff=diff, plan=build_execution_plan(diff))
    jobs = release.runner.created_jobs
    assert len(jobs) == 1, "the repair must produce exactly one credential proof Job"
    container = jobs[0]["spec"]["template"]["spec"]["containers"][0]
    command = container["command"]
    assert command[:3] == [CONTROL_PLANE_PYTHON, "-I", "-c"], (
        "the Job must use the isolated control-plane interpreter"
    )
    environment = {item["name"]: item.get("value") for item in container["env"]}
    assert environment[REFRESH.RESTART_SWITCH_ENV] == "false", (
        "the prerequisite Job must never authorize consumer restarts"
    )
    assert not container.get("args"), "extra arguments could enable consumer restarts"
    return str(command[3])


def test_generated_program_imports_current_runtime_api_before_main(
    monkeypatch: pytest.MonkeyPatch, credential_program: str
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(REFRESH, "main", lambda: calls.append("main"))

    exec(compile(credential_program, "<credential-proof-job>", "exec"), {})

    assert calls == ["main"], "the generated program failed to reach the real API"


def test_generated_program_rejects_runtime_without_restart_control(
    monkeypatch: pytest.MonkeyPatch, credential_program: str
) -> None:
    calls: list[str] = []

    def legacy_refresh() -> None:
        calls.append("legacy-refresh")

    monkeypatch.setattr(REFRESH, "refresh_once", legacy_refresh)
    monkeypatch.setattr(REFRESH, "main", lambda: calls.append("main"))

    with pytest.raises(
        RuntimeError, match="refresher cannot guarantee no consumer restart"
    ):
        exec(compile(credential_program, "<credential-proof-job>", "exec"), {})

    assert calls == [], "an incompatible refresher reached execution"
