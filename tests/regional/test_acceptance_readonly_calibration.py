from __future__ import annotations

import pytest

from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.models import WorkflowOperation as Op
from scripts.e2e.regional.probes.late_ownership_executor_probe import (
    readonly_calibration_completed,
)
from tests.regional import test_late_ownership_executor_probe as probe_support

owned_probe = probe_support.owned_probe


@pytest.mark.parametrize(
    "defect", ["auth", "timeout", "unknown", "foreign", "pid", "missing"]
)
def test_readonly_calibration_never_accepts_an_unrelated_failure(defect):
    details = {"gpu_client_quiesce_attempt": 1}
    error = "node agent node-a: GPU compute clients are still active: GPU-a:123"
    if defect == "auth":
        error = "HTTP 403 Forbidden"
    elif defect == "timeout":
        return_value = WorkflowStepOutcome.waiting(details=details)
        assert not readonly_calibration_completed(return_value, node="node-a"), (
            "pending execution cannot calibrate an independent witness"
        )
        return
    elif defect == "unknown":
        details["outcome_unknown"] = True
    elif defect == "foreign":
        error = error.replace("node-a", "node-b")
    elif defect == "pid":
        error = error.replace(":123", ":unknown")
    else:
        details.clear()
    assert not readonly_calibration_completed(
        WorkflowStepOutcome.failed(error, details=details), node="node-a"
    ), "only the precise authenticated client-query refusal can calibrate"


def test_active_cuda_refusal_calibrates_the_probe_without_satisfying_stop_barrier(
    owned_probe, monkeypatch
):
    owned = owned_probe("unchanged-owner")
    original = owned.node.execute
    calibrations = []

    def execute(context):
        if "/calibration/" not in context.idempotency_key:
            return original(context)
        assert owned.state.pods, "the workload is deliberately active before STOP"
        assert context.step.operation is Op.VERIFY_NO_GPU_CLIENTS
        calibrations.append(context.step.node_ids[0])
        return WorkflowStepOutcome.failed(
            f"node agent {context.step.node_ids[0]}: "
            "GPU compute clients are still active: GPU-a:123",
            details={"gpu_client_quiesce_attempt": 1},
        )

    monkeypatch.setattr(owned.node, "execute", execute)
    owned.probe.run()
    assert calibrations == ["node-a", "node-b"]
    calibrated = next(
        item["payload"] for item in owned.messages if item["kind"] == "calibrated"
    )
    assert all(
        item["status"] == "FAILED" and item["workflow_barrier_satisfied"] is False
        for item in calibrated["queries"]
    ), "a calibration refusal must not become a successful no-client barrier"
    assert any(item["kind"] == "stop" for item in owned.messages), (
        "authenticated readonly calibration must allow the real STOP rendezvous"
    )
