"""GF-ENV-LIVE-010 through run_env_live over the scripted world (no cluster).

The target node runs the 004 chain on one GPU while the world's control node
holds the independent control task; the scope judge grades the product's
records on the target and the control side. Supporting logic coverage only.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import env_live_expectations as expectations
from scripts.e2e.regional import env_live_fixture as fixture
from scripts.e2e.regional import env_live_scope as scope
from scripts.e2e.regional import env_live_watch as watch
from scripts.e2e.regional import run_env_live as runner
from tests.regional._env_live_world import (
    GPU,
    H200_LIMITS,
    NODE,
    World,
    WorldController,
    WorldControlTask,
    WorldRegionalSettings,
    h200_preflight,
    install_world,
    review_chain,
)

CASE = "GF-ENV-LIVE-010"
CONTROL = "ip-10-0-32-162"
SECOND_GPU = "GPU-bbbb"


def _devices() -> list[dict[str, Any]]:
    return [
        {
            "gpu_uuid": uuid,
            "gpu_index": index,
            "product": "NVIDIA H200",
            "pci_bdf": bdf,
            "limits": dict(H200_LIMITS),
        }
        for uuid, index, bdf in ((GPU, "0", "59:00"), (SECOND_GPU, "1", "5a:00"))
    ]


def _preflight(**overrides: Any) -> dict[str, Any]:
    devices = _devices()
    value = h200_preflight(
        devices=devices,
        target_gpu=devices[0],
        control_node={"name": CONTROL, "errors": []},
    )
    value.update(overrides)
    return value


def _settings(tmp_path: Path, **overrides: Any) -> fixture.EnvLiveSettings:
    site = tmp_path / "site.yaml"
    site.write_text("clusters: []\n")
    values: dict[str, Any] = {
        "regional": WorldRegionalSettings(),
        "case_id": CASE,
        "confirmation": runner.confirmation_for(CASE),
        "node": NODE,
        "host_probe_image": "public.ecr.aws/x@sha256:" + "0" * 64,
        "gpu_uuid": GPU,
        "predecessor_path": None,
        "predecessor_id": None,
        "site_file": site,
        "isolate_confirm": runner.isolate_token(CASE),
        "stop_confirm": runner.stop_token(CASE),
        "control_node": CONTROL,
    }
    values.update(overrides)
    return fixture.EnvLiveSettings(**values)


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> World:
    value = World(start=datetime.now(timezone.utc))
    install_world(monkeypatch, value, preflight=_preflight())
    return value


def _window() -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=1)


def _read_case(run_dir: Path) -> dict[str, Any]:
    import json

    return json.loads((run_dir / "cases" / CASE / f"{CASE}.json").read_text())


# --------------------------------------------------------------------------- #
def test_scope_expectation_is_the_review_with_the_control_node() -> None:
    preflight = _preflight()
    plan = runner.replay_plans_for_case(CASE, preflight)["plans"]["core_tlimit/severe"]
    expectation = expectations.expectation_for(
        case_id=CASE,
        plan_name="core_tlimit/severe",
        plan=plan,
        preflight=preflight,
        cadence=watch.CollectorCadence(),
        node=NODE,
    )
    assert expectation.kind == "review" and expectation.control_node == CONTROL
    assert any("control task" in note for note in expectation.notes), expectation.notes


def test_plan_details_name_the_control_node_or_why_not_run(tmp_path: Path) -> None:
    details = runner.plan_details(_settings(tmp_path), _preflight())
    assert details["scope"]["control_node"] == CONTROL
    assert details["scope"]["not_run_reason"] is None
    assert details["verdict_if_not_run"] is None
    assert details["branch_confirmations"] == {
        "isolate": "ENVLIVE010_ISOLATE",
        "stop": "ENVLIVE010_STOP",
    }
    alone = runner.plan_details(_settings(tmp_path, control_node=""), _preflight())
    assert alone["verdict_if_not_run"] == "NOT_RUN"
    assert "no control node" in alone["not_run_reason"]


def test_execute_010_not_run_when_the_target_has_one_gpu(
    tmp_path: Path, world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    single = h200_preflight(control_node={"name": CONTROL, "errors": []})
    monkeypatch.setattr(runner, "read_only_preflight", lambda s, d: single)
    code = runner.execute_case(_settings(tmp_path), tmp_path, 1, _window())
    result = _read_case(tmp_path)
    assert code == 0 and result["verdict"] == "NOT_RUN"
    assert "fewer than two GPUs" in result["not_run_reason"]
    assert WorldControlTask.instances == [] and WorldController.instances == []


def test_execute_010_without_branch_tokens_refuses_before_the_task(
    tmp_path: Path, world: World
) -> None:
    from scripts.e2e.regional.regional_commands import RegionalFixtureError

    settings = _settings(tmp_path, isolate_confirm="", stop_confirm="")
    with pytest.raises(RegionalFixtureError, match="--confirm-isolate"):
        runner.execute_case(settings, tmp_path, 1, _window())
    assert WorldControlTask.instances == []


def test_execute_010_starts_the_control_task_first_and_removes_it_last(
    tmp_path: Path, world: World
) -> None:
    review_chain(world)
    code = runner.execute_case(_settings(tmp_path), tmp_path, 1, _window())
    result = _read_case(tmp_path)
    assert code == 0 and result["verdict"] == "PASS", (
        result["stages"],
        result["inconclusive"],
    )
    task = WorldControlTask.instances[0]
    assert task.node == CONTROL and task.calls[0] == "start"
    controller = WorldController.instances[0]
    assert controller.calls[0] == "create", "the task is running before the probe"
    assert task.calls[-1] == "cleanup"
    receipts = result["receipts"]
    assert receipts["control_task_before"]["pods"][0]["uid"] == "pod-ctl-1"
    assert receipts["control_task_cleanup"]["workload"] == "removed"
    assert receipts["control_node_before"]["name"] == CONTROL
    observation = receipts["observation"]
    assert observation["control_node"] == CONTROL
    assert observation["control_node_touched"] is False
    assert observation["control_task"]["pods_last"] == ["pod-ctl-1"]
    assert observation["review_workflows"] == ["wf-review"]
    assert observation["control_gpu_findings"] == 0
    assert result["control_node"] == CONTROL


def test_execute_010_fails_when_the_control_node_is_touched(
    tmp_path: Path, world: World
) -> None:
    review_chain(world)

    def cordon_control(w: World) -> None:
        w.control_node["unschedulable"] = True
        w.control_node["taints"] = [
            {"key": "gpu-fault.io/quarantined", "effect": "NoSchedule"}
        ]

    world.at(90, cordon_control)
    runner.execute_case(_settings(tmp_path), tmp_path, 1, _window())
    result = _read_case(tmp_path)
    assert result["verdict"] == "FAIL"
    assert any("control node" in item for item in result["stages"]["control"]), result[
        "stages"
    ]


def test_execute_010_fails_when_the_control_task_changes(
    tmp_path: Path, world: World
) -> None:
    review_chain(world)

    def replace_pod(w: World) -> None:
        w.control_pods[0] = {**w.control_pods[0], "uid": "pod-ctl-2"}

    world.at(120, replace_pod)
    runner.execute_case(_settings(tmp_path), tmp_path, 1, _window())
    result = _read_case(tmp_path)
    assert result["verdict"] == "FAIL"
    assert any(
        "control task changed" in item for item in result["stages"]["control"]
    ), result["stages"]


def test_execute_010_fails_when_a_control_gpu_is_found_or_replayed(
    tmp_path: Path, world: World
) -> None:
    review_chain(world)
    world.finding(
        70,
        "f-other",
        name="gpu_tlimit_c",
        value=-3.0,
        severity="WARNING",
        count=1,
        gpu=SECOND_GPU,
    )
    world.accept(75, {"gpu_tlimit_c": -3.0, "gpu_temperature_c": 30.0}, gpu=SECOND_GPU)
    runner.execute_case(_settings(tmp_path), tmp_path, 1, _window())
    result = _read_case(tmp_path)
    assert result["verdict"] == "FAIL"
    text = "\n".join(result["stages"]["scope"])
    assert "control GPU" in text and SECOND_GPU in text


def test_execute_010_fails_on_a_second_isolating_workflow_or_duplicate_notice(
    tmp_path: Path, world: World
) -> None:
    review_chain(world)
    # A second review workflow for the same confirmed finding's incident.
    world.workflow(
        80,
        "wf-review-2",
        "f-crit",
        operations=("FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"),
        status="RUNNING",
    )
    runner.execute_case(_settings(tmp_path), tmp_path, 1, _window())
    result = _read_case(tmp_path)
    assert result["verdict"] == "FAIL"
    assert any(
        "isolating workflows" in item for item in result["stages"]["concurrency"]
    ), result["stages"]


def test_execute_010_fails_when_the_control_node_receives_an_incident(
    tmp_path: Path, world: World
) -> None:
    review_chain(world)

    def control_incident(w: World) -> None:
        w.control_incidents["inc-ctl"] = {
            "incident_id": "inc-ctl",
            "event_id": "gpu-x",
            "state": "ACTION_PENDING",
            "created_at": w.now.isoformat(),
        }

    world.at(100, control_incident)
    runner.execute_case(_settings(tmp_path), tmp_path, 1, _window())
    result = _read_case(tmp_path)
    assert result["verdict"] == "FAIL"
    assert any("incidents/inc-ctl" in item for item in result["stages"]["control"]), (
        result["stages"]
    )


def test_control_task_names_are_dns_labels_bound_to_the_run() -> None:
    name = scope.control_task_name(CASE, "gf-env-live-010-a1-1")
    assert name.startswith("envctl-") and len(name) == len("envctl-") + 10
    assert name == scope.control_task_name(CASE, "gf-env-live-010-a1-1")
    assert name != scope.control_task_name(CASE, "gf-env-live-010-a2-1")
    assert scope.task_pods({"pods": [{"uid": "a"}, {"name": "x"}]}) == {
        "a": {"uid": "a"}
    }
