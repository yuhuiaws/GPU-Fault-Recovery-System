"""The XID 45 opt-in rides the control-worker env window (batch 131).

DESTR-010 and DESTR-019 set ``GPU_FAULT_XID45_SOLO_NVSWITCH_RESTART_FM=true``
on the control-worker for the duration of the case, through the same window
DESTR-018 uses for its timing set: baseline first, convergence on every ready
replica, exact restore. The switch accepts ``true`` alone, the DESTR-018 set is
untouched, and the case's runtime identity allows exactly the window's two
control-worker rolls.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import control_plane_env_window as env_window
from scripts.e2e.regional import fabric_restart_shape as shapes
from scripts.e2e.regional import xid45_opt_in
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._cov95_destr_branches import ready_runtime

SWITCH = "GPU_FAULT_XID45_SOLO_NVSWITCH_RESTART_FM"
GRADED = {
    "available": True,
    "graded_gpus": ["GPU-0"],
    "states": {"GPU-0": 3.0},
    "product": "NVIDIA B200",
    "observe_only_default": True,
}


def test_the_switch_is_a_policy_switch_beside_the_unchanged_timing_set() -> None:
    assert xid45_opt_in.SWITCH == SWITCH == env_window.XID45_SOLO_RESTART_FM_VARIABLE
    assert env_window.POLICY_SWITCHES == {SWITCH: "true"}
    assert SWITCH not in env_window.ALLOWED_VARIABLES, "DESTR-018's set is unchanged"
    assert env_window.WINDOW_VARIABLES == (*env_window.ALLOWED_VARIABLES, SWITCH)
    assert SWITCH in env_window.SURVEYED_VARIABLES
    assert env_window.parse_assignments([f"{SWITCH}=true"]) == {SWITCH: "true"}
    assert env_window.complete_assignments({SWITCH: "true"}) == {SWITCH: "true"}, (
        "no timing variable is derived from the switch"
    )
    assert env_window.assignment_errors({SWITCH: "true"}) == []


@pytest.mark.parametrize("value", ["false", "1", "TRUE", ""])
def test_the_switch_accepts_true_alone(value: str) -> None:
    with pytest.raises(RegionalFixtureError, match="may only be set to 'true'"):
        env_window.parse_assignments([f"{SWITCH}={value}"])
    assert env_window.assignment_errors({SWITCH: value}), value


def test_the_switch_and_the_timing_set_are_judged_separately() -> None:
    assignments = env_window.parse_assignments(
        [
            f"{SWITCH}=true",
            "GPU_FAULT_NODE_WORKFLOW_MAX_LIFETIME_SECONDS=300",
            "GPU_FAULT_WORKFLOW_EXECUTION_TIMEOUT_SECONDS=300",
        ]
    )

    completed = env_window.complete_assignments(assignments)
    assert completed[SWITCH] == "true"
    assert set(completed) == {*env_window.ALLOWED_VARIABLES, SWITCH}
    assert env_window.assignment_errors(completed) == []


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([None, None], False),
        (["true", "true"], True),
        (["1", "yes"], True),
        (["true", None], None),
        ([], None),
    ],
    ids=["default", "opted-in", "true-tokens", "half-rolled", "no-replica"],
)
def test_the_control_worker_opts_in_only_when_every_replica_agrees(
    values: list[str | None], expected: bool | None
) -> None:
    survey = {
        "replicas": [
            {"pod": f"worker-{index}", "values": {SWITCH: value}}
            for index, value in enumerate(values)
        ]
    }

    assert xid45_opt_in.opted_in(survey) is expected


@pytest.mark.parametrize(
    ("grading", "switch", "window"),
    [
        (GRADED, {"opted_in": False}, True),
        (GRADED, {"opted_in": True}, False),
        ({**GRADED, "observe_only_default": False}, {"opted_in": False}, False),
        ({**GRADED, "graded_gpus": []}, {"opted_in": False}, False),
    ],
    ids=["default-observe-only", "opted-in", "pre-batch-131", "not-graded"],
)
def test_a_case_needs_the_window_only_under_the_observe_only_default(
    grading: dict[str, Any], switch: dict[str, Any], window: bool
) -> None:
    assert xid45_opt_in.needs_window(grading, switch) is window
    assert shapes.node_action_shape(grading) == (
        shapes.FABRIC_RESET if grading["graded_gpus"] else shapes.FABRIC_MANAGER_RESTART
    )
    assert xid45_opt_in.switch_errors(grading, switch) == []
    half_rolled = xid45_opt_in.switch_errors(grading, {"opted_in": None})
    assert bool(half_rolled) is shapes.observe_only_default(grading), (
        "a half-rolled switch matters only where the default is observe-only"
    )


def test_open_and_close_go_through_the_control_plane_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, Any]] = []

    def opened(settings: Any, regional: Any, report: Any, assignments: Any) -> dict:
        calls.append(("open", settings.baseline, assignments))
        return {"state": "OPEN", "assignments": assignments, "pre_window_survey": {}}

    def closed(settings: Any, regional: Any, report: Any) -> dict:
        calls.append(("close", settings.baseline, None))
        return {"state": "CLOSED", "close_survey": {}}

    monkeypatch.setattr(env_window, "survey", lambda _regional: {"replicas": []})
    monkeypatch.setattr(env_window, "open_window", opened)
    monkeypatch.setattr(env_window, "close_window", closed)

    assert xid45_opt_in.close_window(object(), tmp_path) is None, "nothing opened"
    report = xid45_opt_in.open_window(object(), tmp_path)
    (tmp_path / xid45_opt_in.BASELINE_NAME).write_text("{}", encoding="utf-8")
    closed_report = xid45_opt_in.close_window(object(), tmp_path)

    baseline = tmp_path / xid45_opt_in.BASELINE_NAME
    assert calls == [("open", baseline, {SWITCH: "true"}), ("close", baseline, None)]
    assert "pre_window_survey" not in report, "the evidence drops the surveys"
    assert closed_report == {"state": "CLOSED"}
    assert xid45_opt_in.window_errors(report, closed_report) == []
    assert xid45_opt_in.window_errors(report, None), "a window left open fails"
    assert (tmp_path / "xid45-opt-in-open.json").is_file(), "the open is evidence"
    assert (tmp_path / "xid45-opt-in-close.json").is_file(), "so is the close"


def test_the_runtime_identity_allows_the_window_s_two_rolls_only() -> None:
    before = ready_runtime()
    after = deepcopy(before)
    worker = after["deployments"]["cpu"]["gpu-fault-control-worker"]
    worker.update(generation=3, observed_generation=3)

    assert xid45_opt_in.identity_errors(before, after) == []

    one_roll = deepcopy(before)
    one_roll["deployments"]["cpu"]["gpu-fault-control-worker"].update(
        generation=2, observed_generation=2
    )
    assert xid45_opt_in.identity_errors(before, one_roll), "the close never rolled"
    other = deepcopy(after)
    other["deployments"]["cpu"]["gpu-fault-api-ha"]["generation"] = 2
    assert xid45_opt_in.identity_errors(before, other), "only the worker may move"
