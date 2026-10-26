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
import yaml

from scripts.e2e.regional import acceptance_gpu_load as load
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
    # Not a low-utilization incident: no duty diagnosis is attached.
    assert not any("diagnosis" in item for item in result["stages"]["control"]), result[
        "stages"
    ]
    assert result["receipts"]["observation"]["control_utilization_incidents"] == []


def test_execute_010_diagnoses_the_control_task_graded_for_low_utilization(
    tmp_path: Path, world: World
) -> None:
    """Live 2026-10-26 (plain EKS, core_tlimit/severe): the product's
    LOW_GPU/LOW_CPU_UTILIZATION rules graded the harness's idle control task.
    Still FAIL, but the control-stage error names the cause (the task's duty)."""

    review_chain(world)
    incident = (
        f"inc-host-{CONTROL}-1793023102533329-host_gpu_utilization_percent-"
        "GPU-80e1ad6c-ca4f-c21b-07ac-80d802782c62-low_gpu_utilization"
    )
    cpu_incident = (
        f"inc-host-{CONTROL}-1793023102533329-training_rank_cpu_percent-"
        "GPU-49f1a9d9-a439-0480-2ea7-c5fd8d375c92-low_cpu_utilization"
    )

    def graded(w: World) -> None:
        for incident_id in (incident, cpu_incident):
            w.control_incidents[incident_id] = {
                "incident_id": incident_id,
                "event_id": incident_id.removeprefix("inc-"),
                "state": "ACTION_PENDING",
                "created_at": w.now.isoformat(),
            }
        w.control_workflows["workflow-6510fd499456d7013aa2902b"] = {
            "request_id": "workflow-6510fd499456d7013aa2902b",
            "state": "RUNNING",
            "created_at": w.now.isoformat(),
        }

    world.at(100, graded)
    runner.execute_case(_settings(tmp_path), tmp_path, 1, _window())
    result = _read_case(tmp_path)
    assert result["verdict"] == "FAIL"
    [control] = result["stages"]["control"]
    assert f"incidents/{incident}" in control and "received" in control
    assert "diagnosis: 2 of them are LOW_GPU_UTILIZATION / LOW_CPU_UTILIZATION" in (
        control
    )
    assert "control task's own utilization" in control
    assert "check the control task's GPU duty" in control
    assert f"{load.GPU_DUTY_ENV} in the rendered manifest" in control
    assert f"expected {scope.CONTROL_TASK_GPU_DUTY_PERCENT}" in control
    observation = result["receipts"]["observation"]
    # Sorted keys: the GPU-utilization incident sorts before the CPU one.
    assert observation["control_utilization_incidents"] == [
        f"incidents/{incident}",
        f"incidents/{cpu_incident}",
    ]
    assert (
        "workflows/workflow-6510fd499456d7013aa2902b"
        in (observation["control_node_activity"])
    )


def test_the_control_task_is_rendered_gpu_active_at_the_acceptance_duty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rendered control task (the manifest the live run submits) carries the
    acceptance default duty, never 0: an idle enrolled job is a low-utilization
    fault to the product by design (live 2026-10-26, the control stage FAILed on
    the harness's own duty-0 task about nine minutes after it started)."""

    from scripts.e2e.regional import fixture_ownership
    from scripts.e2e.regional.regional_live_fixture import (
        RegionalLiveFixture,
        RegionalLiveSettings,
    )

    monkeypatch.setattr(fixture_ownership, "host_identity", lambda: "a" * 64)
    cpu = tmp_path / "cpu.kubeconfig"
    gpu = tmp_path / "gpu.kubeconfig"
    cpu.write_text("apiVersion: v1\n", encoding="utf-8")
    gpu.write_text("apiVersion: v1\n", encoding="utf-8")
    regional = RegionalLiveFixture(
        RegionalLiveSettings(
            cpu_kubeconfig=cpu,
            gpu_kubeconfig=gpu,
            gpu_context="gpu-context",
            namespace="gpu-fault-system",
            cluster_id="gpu-eks-cluster",
            region="us-west-2",
        )
    )
    site = tmp_path / "site.yaml"
    site.write_text("clusters: []\n", encoding="utf-8")
    case_dir = tmp_path / "cases" / CASE

    task = scope.ControlTask(
        regional,
        site_file=site,
        case_dir=case_dir,
        case_id=CASE,
        run_id="gf-env-live-010-a7-1793022554",
        node=CONTROL,
    )

    rendered = task.fixture.settings.manifest
    assert rendered == case_dir / f"{task.name}.source.yaml"
    document = yaml.safe_load(rendered.read_text(encoding="utf-8"))
    assert document["kind"] == "PyTorchJob"
    assert document["metadata"]["labels"]["gpu-fault.io/job-id"] == task.name
    assert document["metadata"]["labels"]["gpu-fault.io/attempt-id"] == task.attempt_id
    duties: dict[str, list[str]] = {}
    for replica in document["spec"]["pytorchReplicaSpecs"].values():
        pod = replica["template"]["spec"]
        assert pod["nodeName"] == CONTROL, "pinned to the control node"
        for container in pod["containers"]:
            duties[container["name"]] = [
                str(item["value"])
                for item in container.get("env") or []
                if item.get("name") == load.GPU_DUTY_ENV
            ]
    assert duties == {"pytorch": [str(load.DEFAULT_GPU_DUTY_PERCENT)]}, duties
    assert duties["pytorch"] != ["0"], "the control task must not run idle"
    assert scope.CONTROL_TASK_GPU_DUTY_PERCENT == load.DEFAULT_GPU_DUTY_PERCENT
    assert 0 < scope.CONTROL_TASK_GPU_DUTY_PERCENT <= load.MAX_GPU_DUTY_PERCENT
    assert task.fixture.settings.expected_gpu_count == scope.CONTROL_TASK_GPU_COUNT
    assert task.fixture.settings.job_id == task.name


def test_control_task_names_are_dns_labels_bound_to_the_run() -> None:
    name = scope.control_task_name(CASE, "gf-env-live-010-a1-1")
    assert name.startswith("envctl-") and len(name) == len("envctl-") + 10
    assert name == scope.control_task_name(CASE, "gf-env-live-010-a1-1")
    assert name != scope.control_task_name(CASE, "gf-env-live-010-a2-1")
    assert scope.task_pods({"pods": [{"uid": "a"}, {"name": "x"}]}) == {
        "a": {"uid": "a"}
    }
