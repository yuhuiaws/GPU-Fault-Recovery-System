from __future__ import annotations

import pytest

from gpu_fault.adapters.kubernetes.stop_ownership import (
    STOP_RECEIPT_KEY,
    node_submission_ownership_guard,
    stop_ownership_scope,
)
from gpu_fault.models import WorkflowStepStatus
from gpu_fault.orchestration.escalation import unknown_outcome_failure
from tests.regional._late_ownership_runtime import stopped_runtime


def test_real_stop_adapter_captures_both_participants_and_preserves_normal_admission():
    state, _adapter, validator, context = stopped_runtime()
    receipt = context.workflow.step_executions[0].details[STOP_RECEIPT_KEY]
    assert receipt["contained"] is True
    assert {pod["node_id"] for pod in receipt["pods"]} == {"node-a", "node-b"}
    assert {node["uid"] for node in receipt["nodes"]} == {"node-a-uid", "node-b-uid"}
    assert not state.pods, "real STOP must remove both original participant Pods"
    with stop_ownership_scope(validator):
        assert node_submission_ownership_guard(context) is None


@pytest.mark.parametrize(
    "drift", ["workload-uid", "owner", "late-sibling", "node-uid", "boot", "unreadable"]
)
def test_drift_after_real_stop_refuses_all_new_hardware_without_escalation(drift):
    state, _adapter, validator, context = stopped_runtime()
    if drift == "workload-uid":
        state.job["metadata"]["uid"] = "replaced"
    elif drift == "owner":
        state.job["metadata"]["ownerReferences"] = [
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "name": "new-owner",
                "uid": "new-owner-uid",
                "controller": False,
            }
        ]
    elif drift == "late-sibling":
        state.pods.append(state.pod("node-b", "late-pod"))
    elif drift == "node-uid":
        state.nodes["node-b"]["metadata"]["uid"] = "node-b-replaced"
    elif drift == "boot":
        state.nodes["node-b"]["status"]["nodeInfo"]["bootID"] = "new-boot"
    else:
        state.read_error = RuntimeError("private diagnostic must not be emitted")
    with stop_ownership_scope(validator):
        outcome = node_submission_ownership_guard(context)
    assert outcome is not None
    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.details["manual_confirmation_required"] is True
    assert outcome.details["safety_rejection"] is True
    assert unknown_outcome_failure(outcome.details), (
        "safety rejection must retain unknown-outcome protection"
    )
    assert outcome.details["reason"] == (
        "STOP_PARTICIPANTS_CHANGED"
        if drift == "late-sibling"
        else "STOP_OWNERSHIP_UNVERIFIABLE"
        if drift == "unreadable"
        else "STOP_OWNERSHIP_DRIFT"
    )
    assert "private diagnostic" not in str(outcome)
