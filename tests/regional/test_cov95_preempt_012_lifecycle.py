from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import run_preempt012_acceptance as runner
from tests.regional._cov95_preempt_012 import PreemptionModel


def test_complete_fake_cycle_checks_boundaries_and_tears_down(
    tmp_path, monkeypatch
) -> None:
    model = PreemptionModel(tmp_path, monkeypatch)
    code, result = model.run()
    assert code == 0 and result["verdict"] == "PASS", (
        "complete local evidence must pass"
    )
    assert len(result["checks"]) == 16, "retain all lifecycle and cleanup checks"
    assert all(result["checks"].values()), "each published check must be satisfied"
    assert model.predecessors == [
        runner.predecessor_path(tmp_path, runner.CASE_ID, None)
    ], "execution must forward the formal predecessor and its evidence path"
    assert model.calls == [
        "create",
        "snapshot",
        "arm",
        "read",
        "audit",
        "read",
        "snapshot",
        "audit-cleanup",
        "cleanup",
        "remove",
        "node",
    ], "the audit must overlap the cycle and cleanup must follow in order"
    assert result["control_cleanup"] == {"residual_objects": [], "residual_links": 0}, (
        "cleanup needs explicit zero residual counts"
    )


@pytest.mark.parametrize(
    ("failure", "error_key", "audit_started"),
    [
        ("baseline", "error", False),
        ("timer", "error", False),
        ("quiesce", "error", False),
        ("audit", "error", True),
        ("cycle", "error", True),
        ("audit-cleanup", "error", True),
        ("audit-residual", "error", True),
        ("host-cleanup", "host_cleanup_error", True),
        ("probe-cleanup", "probe_cleanup_error", True),
        ("node", "node_snapshot_error", True),
    ],
)
def test_partial_cycle_never_reports_success_and_still_cleans(
    failure, error_key, audit_started, tmp_path, monkeypatch
) -> None:
    model = PreemptionModel(tmp_path, monkeypatch)
    model.failure = failure
    code, result = model.run()
    assert code == 1 and result["verdict"] == "FAIL", f"{failure} must invalidate PASS"
    assert error_key in result, f"{failure} needs an auditable diagnostic"
    assert model.calls[-3:] == ["cleanup", "remove", "node"], (
        "cleanup errors must not prevent the independent cleanup/read attempts"
    )
    assert ("audit-cleanup" in model.calls) is audit_started, (
        "the control audit cleanup must run exactly when records may exist"
    )


def test_remaining_probe_resource_invalidates_otherwise_passing_checks(
    tmp_path, monkeypatch
) -> None:
    model = PreemptionModel(tmp_path, monkeypatch)
    model.failure = "residual"
    code, result = model.run()
    assert all(result["checks"].values()), "this fixture isolates the residual gate"
    assert result["probe_residuals"] == {"pod": True}, "retain the positive residual"
    assert code == 1 and result["verdict"] == "FAIL", "residual resources forbid PASS"


@pytest.mark.parametrize("gate", ["preflight", "deadline"])
def test_execute_gates_run_before_host_creation(gate, tmp_path, monkeypatch) -> None:
    model = PreemptionModel(tmp_path, monkeypatch)
    if gate == "preflight":
        model.preflight_errors = ["unsafe fixture"]
    else:
        model.minutes = 1
    with pytest.raises(runner.PreemptAcceptanceError, match="preflight|10 minutes"):
        model.run()
    assert model.calls == [], "failed authorization/preflight must not touch the host"


@pytest.mark.parametrize(
    ("path", "value", "check"),
    [
        (("clean", "status"), "RUNNING", "clean_boundary_superseded"),
        (
            ("clean", "new_adapter_calls"),
            ["QUIESCE_GPU_SERVICES"],
            "clean_boundary_superseded",
        ),
        (("clean", "preempted_by"), "other", "clean_successor_reused_containment"),
        (
            ("clean", "successor_adapter_calls"),
            [],
            "clean_successor_reused_containment",
        ),
        (
            ("clean", "successor_inherited_step_indexes"),
            [],
            "clean_successor_reused_containment",
        ),
        (
            ("clean", "successor_completed_operations"),
            [],
            "clean_successor_reused_containment",
        ),
        (
            ("clean", "successor_inherited_from"),
            [],
            "clean_successor_reused_containment",
        ),
        (("dirty", "status"), "RUNNING", "dirty_boundary_superseded"),
        (("dirty", "handoff_after_claim"), "other", "dirty_handoff_recorded"),
        (
            ("physical_operations_called",),
            ["RESET_GPU"],
            "no_physical_operation_called",
        ),
        (("remote_commands_for_audit_workflows",), 1, "no_remote_command_created"),
        (("residual_objects",), ["leftover"], "audit_objects_removed"),
    ],
)
def test_verdict_requires_each_independent_control_boundary(
    path, value, check, tmp_path, monkeypatch
) -> None:
    model = PreemptionModel(tmp_path, monkeypatch)
    target = model.control
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = copy.deepcopy(value)
    code, result = model.run()
    assert result["checks"][check] is False, f"the verdict must reject {path}"
    assert code == 1, "a false control-boundary check must fail the whole case"


@pytest.mark.parametrize("recovers", [False, True])
def test_cycle_poll_read_error_is_unknown_and_wait_is_bounded(
    recovers, monkeypatch
) -> None:
    state = SimpleNamespace(now=0.0, calls=0)

    def read(*_args):
        state.calls += 1
        if state.calls == 1:
            raise RuntimeError("transport unavailable")
        return {"status": "QUIESCED" if recovers else "PENDING"}

    monkeypatch.setattr(
        runner,
        "time",
        SimpleNamespace(
            monotonic=lambda: state.now,
            sleep=lambda seconds: setattr(state, "now", state.now + seconds),
        ),
    )
    result = runner.wait_for_cycle(
        SimpleNamespace(execute=read),
        "run-a",
        until=frozenset({"QUIESCED"}),
        timeout=4,
        poll_seconds=2,
    )
    assert state.calls == 2, "read retries must fit the supplied deadline"
    assert result == {"status": "QUIESCED" if recovers else "PENDING"}, (
        "a failed read cannot be converted to a completed cycle"
    )
