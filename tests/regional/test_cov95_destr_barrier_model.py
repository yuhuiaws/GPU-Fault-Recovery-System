from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from gpu_fault.models import WorkflowStatus
from scripts.e2e.regional import destr_barrier_authorization as authorization
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._cov95_destr_warm import NOW, Clock
from tests.regional.test_destr_barrier_authorization import barrier_state


def authorize(state: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(authorization, "datetime", Clock())
    return authorization.barrier_authorization(
        state,
        run_id="review",
        node="node-a",
        boot_id="boot-a",
        device="/dev/nvidia0",
        drill_id="review-r",
        maintenance_window_end=NOW + timedelta(minutes=10),
    )


def test_running_workflow_with_waiting_step_can_produce_bound_controller_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = barrier_state()
    state["workflow"]["status"] = WorkflowStatus.RUNNING
    proof = authorize(state, monkeypatch)
    assert proof["waiting"] is True and proof["workflow_request_id"] == "workflow-a", (
        proof
    )
    assert proof["command_ids"] == {
        "QUIESCE_GPU_SERVICES": "quiesce_gpu_services/node-a/agent-4",
        "VERIFY_NO_GPU_CLIENTS": "verify_no_gpu_clients/node-a/agent-4",
    }, proof
    assert proof["fencing_token"] == 3 and proof["agent_generation"] == 4, proof


@pytest.mark.parametrize(
    "status",
    [
        WorkflowStatus.PENDING,
        WorkflowStatus.SAFETY_PENDING,
        WorkflowStatus.BLOCKED,
        WorkflowStatus.FAILED,
        WorkflowStatus.SUCCEEDED,
        WorkflowStatus.SUPERSEDED,
        "WAITING",
        "unknown",
        None,
    ],
)
def test_timer_authorization_refuses_non_running_or_non_model_workflow_status(
    status: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = barrier_state()
    state["workflow"]["status"] = status
    with pytest.raises(RegionalFixtureError, match="exact drill/workflow binding"):
        authorize(state, monkeypatch)


@pytest.mark.parametrize("expires", [None, "malformed", 3])
def test_controller_proof_rejects_unparseable_pinned_window(
    expires: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = barrier_state()
    state["workflow"]["status"] = WorkflowStatus.RUNNING
    state["workflow"]["step_executions"][0]["details"][
        "maintenance_window_expires_at"
    ] = expires
    with pytest.raises(RegionalFixtureError, match="pinned maintenance window"):
        authorize(state, monkeypatch)
