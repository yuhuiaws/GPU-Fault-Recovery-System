"""A quiesce that a reboot interrupted is reconciled when the agent starts.

The fail-safe restore is a transient systemd timer, so an out-of-band reboot
takes the timer with it while the ``quiesce-*.json`` state file survives on
disk. Before this, nothing on the node ever looked at that file again: the same
incident could not quiesce a second time ("belongs to another workflow"), a
RESET_GPU for it would have passed ``assert_quiesced`` on a boot that never
quiesced, and the fleet kept reporting residue. DESTR-017 found it live.
"""

from __future__ import annotations

import json
from pathlib import Path

from ._support import (
    GpuServiceQuiesceManager,
    ServiceRunner,
    WorkflowOperation,
    command,
    envelope,
    quiesce_executor,
)

SERVICES = ("nvidia-fabricmanager", "nvidia-dcgm", "kubelet")


def _manager(tmp_path: Path, runner: ServiceRunner, boot_id: str):
    return GpuServiceQuiesceManager(
        state_dir=str(tmp_path / "quiesce"),
        services=SERVICES,
        processes=("nv-hostengine",),
        failsafe_seconds=30,
        retry_seconds=10,
        settle_seconds=0,
        restore_settle_seconds=0,
        restore_command="/opt/gpu-fault/restore",
        runner=runner,
        boot_id_reader=lambda: boot_id,
    )


def _state_files(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / "quiesce").glob("quiesce-*.json"))


def _quiesce(tmp_path: Path, runner: ServiceRunner, boot_id: str) -> dict:
    manager = _manager(tmp_path, runner, boot_id)
    return manager.quiesce(incident_id="incident-a", workflow_request_id="workflow-a")


def test_a_quiesce_records_the_boot_it_happened_on(tmp_path: Path) -> None:
    runner = ServiceRunner(active=set(SERVICES))

    _quiesce(tmp_path, runner, "boot-1")

    (state_path,) = _state_files(tmp_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["boot_id"] == "boot-1"
    assert state["phase"] == "QUIESCED"


def test_a_quiesce_from_an_earlier_boot_is_restored_and_forgotten(
    tmp_path: Path,
) -> None:
    runner = ServiceRunner(active=set(SERVICES))
    quiesced = _quiesce(tmp_path, runner, "boot-1")
    # The reboot brought the enabled services back but took the timer with it.
    runner.active = set(SERVICES)
    runner.commands.clear()

    report = _manager(tmp_path, runner, "boot-2").reconcile_after_boot()

    assert report["boot_id"] == "boot-2"
    assert [item["incident_id"] for item in report["restored"]] == ["incident-a"]
    assert report["restored"][0]["recorded_boot_id"] == "boot-1"
    assert report["kept"] == [] and report["failed"] == []
    assert _state_files(tmp_path) == [], "the stale state file must be removed"
    started = [
        item[2] for item in runner.commands if item[:2] == ["systemctl", "start"]
    ]
    assert started == list(SERVICES), started
    assert ["systemctl", "stop", quiesced["timer_unit"]] in runner.commands


def test_a_quiesce_of_the_current_boot_with_a_live_timer_is_left_alone(
    tmp_path: Path,
) -> None:
    runner = ServiceRunner(active=set(SERVICES))
    quiesced = _quiesce(tmp_path, runner, "boot-1")
    runner.active.add(quiesced["timer_unit"])
    runner.commands.clear()

    report = _manager(tmp_path, runner, "boot-1").reconcile_after_boot()

    assert [item["incident_id"] for item in report["kept"]] == ["incident-a"]
    assert report["restored"] == [] and report["failed"] == []
    assert len(_state_files(tmp_path)) == 1
    assert not any(item[:2] == ["systemctl", "start"] for item in runner.commands), (
        "an in-flight quiesce must not be undone by an agent restart"
    )


def test_a_legacy_state_without_a_boot_id_is_judged_by_its_timer(
    tmp_path: Path,
) -> None:
    runner = ServiceRunner(active=set(SERVICES))
    manager = _manager(tmp_path, runner, "boot-1")
    state_dir = tmp_path / "quiesce"
    state_dir.mkdir()
    timer_unit = "gpu-fault-quiesce-legacy"
    legacy = {
        "schema_version": 1,
        "incident_id": "incident-legacy",
        "workflow_request_id": "workflow-legacy",
        "phase": "QUIESCED",
        "active_services": ["kubelet"],
        "configured_services": list(SERVICES),
        "container_targets": [],
        "target_device_paths": [],
        "workload_cgroup_paths": [],
        "timer_unit": timer_unit,
    }
    path = state_dir / "quiesce-legacy.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    runner.active.add(timer_unit + ".timer")
    kept = manager.reconcile_after_boot()
    assert [item["incident_id"] for item in kept["kept"]] == ["incident-legacy"]
    assert path.exists(), "a legacy file whose timer still runs is in flight"

    runner.active.discard(timer_unit + ".timer")
    restored = manager.reconcile_after_boot()
    assert [item["incident_id"] for item in restored["restored"]] == ["incident-legacy"]
    assert not path.exists(), "a legacy file without a timer is stale"


def test_a_failed_restore_is_reported_kept_for_retry_and_does_not_stop_others(
    tmp_path: Path,
) -> None:
    runner = ServiceRunner(active=set(SERVICES))
    manager = _manager(tmp_path, runner, "boot-1")
    manager.quiesce(incident_id="incident-a", workflow_request_id="workflow-a")
    manager.quiesce(incident_id="incident-b", workflow_request_id="workflow-b")
    runner.active = set(SERVICES)
    runner.fail_start = "kubelet"
    by_incident = {
        json.loads(path.read_text(encoding="utf-8"))["incident_id"]: path
        for path in _state_files(tmp_path)
    }
    broken_path = by_incident["incident-a"]
    healthy_path = by_incident["incident-b"]
    healthy = json.loads(healthy_path.read_text(encoding="utf-8"))
    healthy["active_services"] = ["nvidia-dcgm"]
    healthy_path.write_text(json.dumps(healthy), encoding="utf-8")

    report = _manager(tmp_path, runner, "boot-2").reconcile_after_boot()

    assert [item["incident_id"] for item in report["failed"]] == ["incident-a"]
    assert "start failed" in report["failed"][0]["error"]
    assert [item["incident_id"] for item in report["restored"]] == ["incident-b"]
    assert broken_path.exists(), "a failed restore keeps its state for the next try"
    # The failed start leaves the file mid-restore; the next agent start sees
    # the old boot id again and retries.
    assert json.loads(broken_path.read_text(encoding="utf-8"))["phase"] in {
        "RESTORING",
        "RESTORE_FAILED",
    }
    assert not healthy_path.exists(), "the healthy file was restored and removed"


def test_the_agent_reconciles_stale_quiesce_state_when_it_starts(
    tmp_path: Path,
) -> None:
    from fastapi.testclient import TestClient

    from gpu_fault.node_agent.app import create_node_agent_app

    runner = ServiceRunner(active=set(SERVICES))
    stale = _manager(tmp_path, runner, "boot-1")
    stale.quiesce(incident_id="incident-a", workflow_request_id="workflow-a")
    runner.active = set(SERVICES)
    runner.commands.clear()
    agent = quiesce_executor(tmp_path, runner)
    agent.quiesce_manager = _manager(tmp_path, runner, "boot-2")

    with TestClient(create_node_agent_app(agent, heartbeat_reporter=None)):
        pass

    assert _state_files(tmp_path) == [], "startup must restore what a reboot cut"
    # The same incident can now be quiesced again on this boot.
    quiesced = agent.execute(envelope(command(WorkflowOperation.QUIESCE_GPU_SERVICES)))
    assert quiesced.status.value == "SUCCEEDED", quiesced.error
    assert quiesced.details.get("already_quiesced") is not True
