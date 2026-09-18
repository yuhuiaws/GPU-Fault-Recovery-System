from __future__ import annotations

import pytest

from scripts.e2e.regional import run_boot020_release_rolling as rolling
from tests.regional._cov95_boot_extra_rolling import BackendModel
from tests.regional._cov95_boot_extra_safety import (
    boot_extra_isolation as boot_extra_isolation,
)


def test_backend_classifies_the_loaded_state_using_the_existing_diff_model():
    model = BackendModel()
    backend = model.backend()
    result = backend.classify("executor")
    assert result["kind"] == "FULL" and result["changed"] == [
        "executor",
        "runtime_profile",
    ]
    assert "runtime-profile" in result["execution_plan"]["nodes"]
    assert result["cpu_roles"] == ["spool", "worker", "ingress"]
    engine = model.engines[0]
    assert model.loaded == [backend.configs["executor"]]
    assert model.classifications == [(engine, engine.state)]
    assert engine.calls == [("load-state",)]
    assert engine.saved == []


@pytest.mark.parametrize(
    ("kind", "resume", "auto_rollback"),
    [
        ("NOOP", False, None),
        ("FULL", False, True),
        ("DATA_PLANE_COMPATIBLE", True, False),
    ],
)
def test_backend_deploy_preserves_config_diff_resume_and_observed_pin_contract(
    monkeypatch, kind, resume, auto_rollback
):
    model = BackendModel()
    backend = model.backend()
    monkeypatch.setattr(rolling, "time", model)
    diff = {"kind": kind, "changed": ["executor"]}
    result = backend.deploy(
        "executor", diff=diff, resume=resume, auto_rollback=auto_rollback
    )
    engine = model.engines[0]
    assert result["phase"] == "complete"
    assert result["injected_failure"] is None
    assert result["release_id"] == "unit-release"
    assert result["release_diff"] == diff
    assert result["operation_duration_seconds"] == 1
    assert result["pins"] == {
        "required": "b" * 64,
        "compatible": [],
        "candidate": "b" * 64,
        "previous_required": "a" * 64,
    }
    assert engine.runner is model.runners[0]
    assert engine.config.auto_rollback is (
        True if auto_rollback is None else auto_rollback
    )
    assert model.config.auto_rollback is True, "override mutated the loaded config"
    call = engine.calls[0]
    assert call[0] == ("noop" if kind == "NOOP" else "upgrade")
    assert call[-1].kind.value == kind
    assert call[-1].changed == frozenset({"executor"})
    if kind != "NOOP":
        assert call[1] is resume


@pytest.mark.parametrize("phase", ["cpu-staged", "data-converged", "cpu-finalized"])
@pytest.mark.parametrize("rollback", [False, True])
def test_backend_injects_after_persisting_the_boundary_and_reports_engine_compensation(
    monkeypatch, phase, rollback
):
    model = BackendModel()
    backend = model.backend()
    monkeypatch.setattr(rolling, "time", model)
    result = backend.deploy(
        "executor",
        diff={"kind": "DATA_PLANE_COMPATIBLE", "changed": ["executor"]},
        fault_phase=phase,
        auto_rollback=rollback,
    )
    engine = model.engines[0]
    phases = [name for name, _updates in engine.saved]
    expected = "rolled-back" if rollback else "failed"
    assert phases[-2:] == [phase, expected], "injection ran before its durable boundary"
    assert phases.count(phase) == 1
    assert "complete" not in phases
    assert result["phase"] == expected
    assert result["injected_failure"] == phase
    assert result["operation_duration_seconds"] == 1
    assert result["pins"]["required"] == "a" * 64
    assert result["pins"]["compatible"] == ([] if rollback else ["b" * 64])
    if rollback:
        assert result["rollback_plan"] == {"clusters": {"cluster-a": ["executor"]}}
        assert result["rollback_timing"] == {"t_safe_seconds": 2, "t_full_seconds": 3}
    else:
        assert result["rollback_plan"] is None
        assert result["rollback_timing"] is None
    assert engine.calls[-1] == ("load-state",)


@pytest.mark.parametrize("kind", ["NOOP", "FULL"])
@pytest.mark.parametrize("aborted", [False, True])
def test_backend_propagates_noninjection_failures_without_fabricating_rollback(
    monkeypatch, kind, aborted
):
    model = BackendModel()
    model.error = (
        KeyboardInterrupt("unit abort") if aborted else ValueError("unit failure")
    )
    backend = model.backend()
    monkeypatch.setattr(rolling, "time", model)
    with pytest.raises(type(model.error)) as caught:
        backend.deploy("full", diff={"kind": kind})
    assert caught.value is model.error
    assert all(name != "rolled-back" for name, _updates in model.engines[0].saved), (
        "the wrapper invented compensation for a noninjection failure"
    )
    assert not any(call[0] == "load-state" for call in model.engines[0].calls), (
        "the wrapper converted an unknown failure into a successful observed result"
    )


def test_unreached_fault_boundary_is_not_reported_as_an_injected_failure(monkeypatch):
    model = BackendModel()
    backend = model.backend()
    monkeypatch.setattr(rolling, "time", model)
    result = backend.deploy(
        "full", diff={"kind": "FULL"}, fault_phase="unreached-phase"
    )
    assert result["phase"] == "complete"
    assert result["injected_failure"] is None
    assert [name for name, _updates in model.engines[0].saved] == [
        "cpu-staged",
        "data-converged",
        "cpu-finalized",
        "complete",
    ]


@pytest.mark.parametrize("listing", [{}, {"items": None}, {"items": []}])
def test_snapshot_rejects_unreadable_or_incomplete_deployment_inventory(listing):
    model = BackendModel()
    model.listing_override = listing
    backend = model.backend()
    with pytest.raises(rolling.AcceptanceCheckError, match="inventory"):
        backend.snapshot("noop")
    engine = model.engines[0]
    assert engine.calls == [("load-state",), ("capture",)]
    assert engine.saved == []


def test_snapshot_ignores_unrelated_rows_but_preserves_the_full_required_generation_set():
    model = BackendModel()
    result = model.backend().snapshot("noop")
    assert result["phase"] == "complete"
    assert set(result["cpu_generations"]) == set(
        rolling.regional_deployment_inventory.CPU_RUNTIME_DEPLOYMENTS
    )
    assert set(result["gpu_generations"]["cluster-a"]) == set(
        (
            *rolling.regional_deployment_inventory.DEPLOYMENTS,
            rolling.regional_deployment_inventory.GPU_RECONCILER_DEPLOYMENT,
        )
    )
    assert "unrelated" not in result["cpu_generations"]
    assert result["next_deploy"] == {"kind": "NOOP", "changed": []}
    assert len(model.engines[0].reads) == 2, "the snapshot repeated a context listing"
