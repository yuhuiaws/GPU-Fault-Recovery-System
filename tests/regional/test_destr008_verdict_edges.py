from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_destr008_warm_spare_shortage as case
from scripts.e2e.regional.warm_spare_fixture import (
    QUARANTINE_TAINT,
    SPARE_POOL_STATE_ANNOTATION,
    SPARE_RESERVATION_ANNOTATION,
)
from tests.regional._cov95_destr_warm import failed_shortage, settings_for


def state() -> dict[str, Any]:
    value = failed_shortage("active-gpu-pod", "event")
    value["fault_node"] = {"unschedulable": True, "taints": [{"key": QUARANTINE_TAINT}]}
    value["spare_node"] = {"annotations": {}}
    return value


@pytest.mark.parametrize(
    "defect",
    [
        "workflow-status",
        "stop-missing",
        "stop-failed",
        "replace-missing",
        "replace-success",
        "wrong-reason",
        "missing-step",
        "unprotected-step",
        "missing-alert",
        "missing-marker",
        "schedulable",
        "unquarantined",
        "reserved",
        "allocated",
    ],
)
def test_each_shortage_invariant_is_required_independently(
    defect: str, tmp_path: Path
) -> None:
    value = state()
    workflow = value["workflow"]
    executions = workflow["step_executions"]
    if defect == "workflow-status":
        workflow["status"] = "SUCCEEDED"
    elif defect == "stop-missing":
        executions.pop(0)
    elif defect == "stop-failed":
        executions[0]["status"] = "FAILED"
    elif defect == "replace-missing":
        executions.pop(1)
    elif defect == "replace-success":
        executions[1]["status"] = "SUCCEEDED"
    elif defect == "wrong-reason":
        executions[1]["error"] = "unrelated failure"
    elif defect == "missing-step":
        workflow["official_steps"] = []
    elif defect == "unprotected-step":
        workflow["official_steps"][0]["parameters"].pop("activation_forbidden")
    elif defect == "missing-alert":
        value["notifications"] = []
    elif defect == "missing-marker":
        value["markers"] = []
    elif defect == "schedulable":
        value["fault_node"]["unschedulable"] = False
    elif defect == "unquarantined":
        value["fault_node"]["taints"] = []
    elif defect == "reserved":
        value["spare_node"]["annotations"][SPARE_RESERVATION_ANNOTATION] = value[
            "incident"
        ]["incident_id"]
    else:
        value["spare_node"]["annotations"][SPARE_POOL_STATE_ANNOTATION] = "ALLOCATED"
    errors = case.scenario_errors(
        value, settings_for(case, tmp_path), "active-gpu-pod", event_id="event"
    )
    assert errors, ("the independently broken invariant must fail", defect, value)


@pytest.mark.parametrize("key", ["activation_inhibited", "cached_activation_rejected"])
@pytest.mark.parametrize("flag", [True, None, 1, "false"])
def test_health_reason_cannot_hide_an_activation_guard_outcome(
    key: str, flag: Any, tmp_path: Path
) -> None:
    value = state()
    value["workflow"]["step_executions"][1]["details"] = {key: flag}
    errors = case.scenario_errors(
        value, settings_for(case, tmp_path), "active-gpu-pod", event_id="event"
    )
    assert "activation inhibition fired instead of the intended shortage gate" in errors


def test_error_prefix_is_an_inhibition_failure_even_without_outcome_details(
    tmp_path: Path,
) -> None:
    value = state()
    execution = value["workflow"]["step_executions"][1]
    execution["error"] += "; ACTIVATION_FORBIDDEN: controlled refusal"
    errors = case.scenario_errors(
        value, settings_for(case, tmp_path), "active-gpu-pod", event_id="event"
    )
    assert "activation inhibition fired instead of the intended shortage gate" in errors


def test_guard_present_but_not_fired_keeps_the_real_health_failure_oracle(
    tmp_path: Path,
) -> None:
    value = state()
    value["workflow"]["step_executions"][1]["details"] = {
        "activation_inhibited": False,
        "cached_activation_rejected": False,
    }
    assert (
        case.scenario_errors(
            value, settings_for(case, tmp_path), "active-gpu-pod", event_id="event"
        )
        == []
    ), "configured inhibition is not itself an attempted activation"
