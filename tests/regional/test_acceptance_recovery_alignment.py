from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.e2e.regional import run_destr001_gpu_reset as runner
from scripts.e2e.regional.reset_dependency_chain import reset_chain_errors
from tests.regional._cov95_destr_idle import IdleHarness
from tests.regional._destructive_acceptance_builders import _reset_state


@pytest.mark.parametrize("failure", ["unknown", "read-error", "injection-ack-lost"])
def test_reset_runner_never_clears_quiesce_when_physical_outcome_is_unknown(
    tmp_path, monkeypatch, failure
):
    harness = IdleHarness(runner, tmp_path, monkeypatch)
    harness.plan(tmp_path)
    if failure == "unknown":
        harness.workflow["workflow"].update(
            status="BLOCKED",
            blocked_kind="NEEDS_OPERATOR",
            step_executions=[
                {
                    "operation": "RESET_GPU",
                    "status": "FAILED",
                    "details": {
                        "outcome_unknown": True,
                        "manual_confirmation_required": True,
                    },
                }
            ],
        )
        harness.after["quiesce_states"] = [{"reset_issued": "owned-reset"}]
    else:
        action = "workflow.wait" if failure == "read-error" else "host.write-xid46"
        harness.failures[action] = RuntimeError("receipt unavailable")
    code, report = harness.execute(tmp_path)
    assert code == 1 and report["verdict"] == "FAIL"
    assert report["quiesce_recovery"]["operator_review_required"] is True
    assert report["quiesce_recovery"]["existing_failsafe_preserved"] is True
    calls = [name for name, _ in harness.calls]
    assert "host.restore-quiesce" not in calls
    assert "host.stop-reset-sampler" in calls
    assert "host.cleanup" in calls


def test_reset_runner_observes_product_restore_without_reissuing_it(
    tmp_path, monkeypatch
):
    harness = IdleHarness(runner, tmp_path, monkeypatch)
    harness.plan(tmp_path)
    code, report = harness.execute(tmp_path)
    assert code == 0
    assert report["quiesce_recovery"]["product_restoration_observed"] is True
    assert report["quiesce_recovery"]["runner_restore_attempted"] is False
    assert "host.restore-quiesce" not in [name for name, _ in harness.calls]


@pytest.mark.parametrize("pending", [False, True])
def test_success_label_cannot_hide_an_unresolved_physical_receipt(pending):
    state = _reset_state()
    if pending:
        state["commands"].append({"status": "LEASED", "result_details": {}})
    else:
        state["workflow"]["step_executions"][0]["details"] = {
            "outcome_unknown": True,
            "manual_confirmation_required": True,
        }
    report = runner.quiesce_recovery_report(
        state, {"quiesce_states": []}, injection_attempted=True
    )
    assert report["operator_review_required"] is True
    assert report["product_restoration_observed"] is False
    assert report["runner_restore_attempted"] is False


def reset_dag():
    state = _reset_state()
    workflow = state["workflow"]
    workflow["dag_enabled"] = True
    for index, step in enumerate(workflow["official_steps"]):
        step.update(
            node_ids=["node-a"],
            gpu_uuids=["GPU-a"],
            execution_owner="owner-a",
            depends_on_step_indexes=[index - 1] if index else [],
        )
    workflow["official_steps"].append(
        {
            "operation": "VALIDATE_GPU",
            "node_ids": ["node-a"],
            "gpu_uuids": ["GPU-a"],
            "execution_owner": "owner-a",
            "depends_on_step_indexes": [0],
        }
    )
    workflow["completed_operations"].append("VALIDATE_GPU")
    workflow["completed_step_indexes"] = list(range(len(workflow["official_steps"])))
    workflow["step_executions"] = [
        {
            "step_index": index,
            "operation": step["operation"],
            "status": "SUCCEEDED",
            "phase": "official",
        }
        for index, step in enumerate(workflow["official_steps"])
    ]
    return workflow


def test_reset_chain_accepts_duplicate_readonly_validation_outside_required_chain():
    workflow = reset_dag()
    assert reset_chain_errors(workflow, runner.EXPECTED_STEPS) == []


def test_reset_chain_refuses_an_oversized_report_before_expanding_paths():
    workflow = reset_dag()
    workflow["official_steps"] = [{"operation": "VALIDATE_GPU"}] * 257
    assert reset_chain_errors(workflow, runner.EXPECTED_STEPS) == [
        "reset contract exceeds the bounded workflow step inventory"
    ]


@pytest.mark.parametrize(
    "fault",
    [
        "barrier",
        "readonly-owner",
        "scope",
        "mutation",
        "unknown-op",
        "missing",
        "cycle",
        "bad-index",
        "partial",
        "failed",
        "completion",
    ],
)
def test_reset_chain_keeps_barriers_scope_and_completion_strict(fault):
    workflow = deepcopy(reset_dag())
    steps = workflow["official_steps"]
    if fault == "barrier":
        steps[4]["depends_on_step_indexes"] = [2]
    elif fault == "readonly-owner":
        steps[-1]["execution_owner"] = None
    elif fault == "scope":
        steps[-1]["node_ids"] = ["foreign-node"]
    elif fault == "mutation":
        steps[-1]["operation"] = "QUARANTINE"
    elif fault == "unknown-op":
        steps[-1]["operation"] = "UNKNOWN"
    elif fault == "missing":
        steps[4]["operation"] = "VALIDATE_HOST"
    elif fault == "cycle":
        steps[0]["depends_on_step_indexes"] = [4]
    elif fault == "bad-index":
        steps[0]["depends_on_step_indexes"] = [True]
    elif fault == "partial":
        workflow["completed_step_indexes"].pop()
    elif fault == "failed":
        workflow["step_executions"][-1]["status"] = "FAILED"
    else:
        workflow["completed_operations"].pop()
    assert reset_chain_errors(workflow, runner.EXPECTED_STEPS), fault
