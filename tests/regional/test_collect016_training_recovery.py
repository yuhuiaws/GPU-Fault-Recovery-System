"""COLLECT-016: resources are owned from the moment they exist, and the A/B
segments assert what is unique to them rather than only DESTR-009's contract."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_collect016_training_recovery as collect016


def _state_a(job_id: str = "c016-a-1") -> dict[str, Any]:
    return {
        "incident": {"workload_identity_source": "SOLE_ACTIVE_ATTEMPT_ON_NODE"},
        "workflow": {
            "status": "SUCCEEDED",
            "blocked_reasons": [],
            "official_steps": [
                {"operation": "STOP_WORKLOADS", "parameters": {"job_id": job_id}},
                {"operation": "RESTART_WORKLOAD", "parameters": {"job_id": job_id}},
            ],
        },
    }


def _state_b() -> dict[str, Any]:
    return {
        "workflow": {
            "status": "FAILED",
            "step_executions": [
                {
                    "operation": "RESTART_WORKLOAD",
                    "status": "FAILED",
                    "details": {"reason": "RESTART_BUDGET_EXHAUSTED"},
                }
            ],
        },
        "commands": [],
        "restart_budget": {"restart_count": 1, "budget": 1},
    }


def test_restart_budget_sections_pass_on_the_documented_readings() -> None:
    assert (
        collect016.restart_budget_section_errors(
            _state_a(), _state_b(), job_id="c016-a-1"
        )
        == []
    )


@pytest.mark.parametrize(
    ("mutate_a", "mutate_b", "fragment"),
    [
        (
            lambda a: a["incident"].__setitem__("workload_identity_source", None),
            None,
            "workload_identity_source",
        ),
        (
            lambda a: a["workflow"].__setitem__("blocked_reasons", ["no workload"]),
            None,
            "was blocked",
        ),
        (
            lambda a: a["workflow"]["official_steps"][1]["parameters"].__setitem__(
                "job_id", "other-job"
            ),
            None,
            "compiled for job",
        ),
        (
            None,
            lambda b: b["workflow"].__setitem__("status", "SUCCEEDED"),
            "not budget",
        ),
        (None, lambda b: b.__setitem__("commands", [{"command_id": "x"}]), "remote"),
        (
            None,
            lambda b: b["workflow"]["step_executions"][0]["details"].__setitem__(
                "reason", "OTHER"
            ),
            "RESTART_BUDGET_EXHAUSTED",
        ),
        (
            None,
            lambda b: b["restart_budget"].__setitem__("restart_count", 2),
            "restart_count",
        ),
    ],
)
def test_each_unique_reading_is_checked(mutate_a, mutate_b, fragment: str) -> None:
    state_a, state_b = _state_a(), _state_b()
    if mutate_a is not None:
        mutate_a(state_a)
    if mutate_b is not None:
        mutate_b(state_b)
    errors = collect016.restart_budget_section_errors(
        state_a, state_b, job_id="c016-a-1"
    )
    assert any(fragment in item for item in errors), (fragment, errors)


def _settings(tmp_path: Path) -> Any:
    class Regional:
        gpu_kubeconfig = "kc"
        gpu_context = "ctx"
        namespace = "ns"

    return collect016.Settings(
        regional=Regional(),  # type: ignore[arg-type]
        site_file=tmp_path / "site.yaml",
        host_probe_image="img",
        predecessor_path=tmp_path / "pred.json",
    )


def test_workload_is_owned_before_its_first_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wait_running timeout used to leak the 24-GPU job: it was only appended
    to the cleanup list after the helper returned."""

    class Workload:
        def submit(self) -> None:
            pass

        def wait_running(self, timeout_seconds: int) -> dict[str, Any]:
            raise TimeoutError("pods never ran")

    workload = Workload()
    monkeypatch.setattr(
        collect016.base, "render_named_training_manifest", lambda p, **_: p
    )
    monkeypatch.setattr(collect016, "managed_fixture", lambda *_, **__: workload)
    workloads: list[Any] = []
    fixtures: list[Any] = []

    with pytest.raises(TimeoutError):
        collect016.run_restart_budget_sections(
            _settings(tmp_path),
            object(),  # type: ignore[arg-type]
            tmp_path,
            "s1",
            workloads=workloads,
            fixtures=fixtures,
        )
    assert workloads == [workload], "the job is registered before wait_running"


def test_probe_is_owned_before_it_is_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Workload:
        def submit(self) -> None:
            pass

        def wait_running(self, timeout_seconds: int) -> dict[str, Any]:
            return {"pods": [{"uid": "u1", "node": "node-a"}]}

    class Collector:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def create(self) -> None:
            raise RuntimeError("image pull failed")

    monkeypatch.setattr(
        collect016.base, "render_named_training_manifest", lambda p, **_: p
    )
    monkeypatch.setattr(collect016, "managed_fixture", lambda *_, **__: Workload())
    monkeypatch.setattr(
        collect016.workload_case, "wait_observation", lambda *_, **__: None
    )
    monkeypatch.setattr(collect016, "CollectorAcceptanceFixture", Collector)
    workloads: list[Any] = []
    fixtures: list[Any] = []

    with pytest.raises(RuntimeError, match="image pull failed"):
        collect016.run_restart_budget_sections(
            _settings(tmp_path),
            object(),  # type: ignore[arg-type]
            tmp_path,
            "s1",
            workloads=workloads,
            fixtures=fixtures,
        )
    assert len(workloads) == 1
    assert len(fixtures) == 1 and isinstance(fixtures[0], Collector), (
        "the probe is registered before create() can fail"
    )


def test_reset_section_owns_workload_and_probe_before_the_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Workload:
        def submit(self) -> None:
            pass

        def wait_running(self, timeout_seconds: int) -> dict[str, Any]:
            return {"pods": [{"uid": "u1", "node": "node-a"}]}

    class Host:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def create(self) -> None:
            raise RuntimeError("no such node")

    monkeypatch.setattr(
        collect016.base, "render_named_training_manifest", lambda p, **_: p
    )
    monkeypatch.setattr(collect016, "managed_fixture", lambda *_, **__: Workload())
    monkeypatch.setattr(
        collect016.workload_case, "wait_observation", lambda *_, **__: None
    )
    monkeypatch.setattr(collect016, "HostProbeFixture", Host)
    monkeypatch.setattr(
        collect016, "HostProbeSettings", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    workloads: list[Any] = []
    fixtures: list[Any] = []

    with pytest.raises(RuntimeError, match="no such node"):
        collect016.run_reset_section(
            _settings(tmp_path),
            object(),  # type: ignore[arg-type]
            tmp_path,
            "s1",
            workloads=workloads,
            fixtures=fixtures,
        )
    assert len(workloads) == 1 and len(fixtures) == 1
