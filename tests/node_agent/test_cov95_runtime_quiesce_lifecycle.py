from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import pytest

from gpu_fault.node_agent import quiesce as quiesce_module
from tests.node_agent._support import ServiceRunner
from tests.node_agent.test_cov95_runtime_quiesce import manager
from tests.regional._cov95_runtime_support import Clock
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


def test_hma_container_must_stop_after_kill_before_quiesce_and_restore_settle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    container_id = "c" * 64
    clock = Clock(step=0.1)
    monkeypatch.setattr(quiesce_module, "time", clock)

    class ContainerRunner(ServiceRunner):
        running = True

        def __call__(self, argv: list[str], **kwargs: Any) -> CompletedProcess:
            if argv[0] != "ctr":
                result = super().__call__(argv, **kwargs)
                if argv[:3] == ["systemctl", "start", "kubelet"]:
                    self.running = True
                return result
            self.commands.append(argv)
            output = ""
            if argv[3:5] == ["containers", "list"]:
                output = container_id + "\n"
            elif argv[3:5] == ["tasks", "list"]:
                output = "TASK PID STATUS\nincomplete-row\n"
                if self.running:
                    output += f"{container_id} 100 RUNNING\n"
            elif argv[3:5] == ["tasks", "kill"] and "SIGKILL" in argv:
                self.running = False
            return CompletedProcess(argv, 0, output, "")

    runner = ContainerRunner(active={"kubelet"})
    sleeps = []

    def sleep(seconds: float) -> None:
        sleeps.append((seconds, runner.running, set(runner.active)))
        clock.sleep(seconds)

    service = manager(
        tmp_path,
        runner,
        containers=("aws-hyperpod/health-monitoring-agent",),
        container_stop_timeout_seconds=5,
        settle_seconds=1,
        restore_settle_seconds=2,
        sleeper=sleep,
    )
    receipt = service.quiesce(incident_id="unit", workflow_request_id="workflow")
    kills = [
        argv
        for argv in runner.commands
        if argv[:5] == ["ctr", "-n", "k8s.io", "tasks", "kill"]
    ]
    assert [argv[6] for argv in kills] == ["SIGTERM", "SIGKILL"]
    assert all(argv[-1] == container_id for argv in kills), kills
    assert receipt["stopped_containers"] == [
        {
            "selector": "aws-hyperpod/health-monitoring-agent",
            "container_id": container_id,
        }
    ]
    assert (1, False, set()) in sleeps
    restored = service.restore(incident_id="unit")
    assert restored["restored"] is True
    assert restored["restore_settle_seconds"] == 2
    assert sleeps[-1] == (2, True, {"kubelet"})
    assert not Path(receipt["state_path"]).exists(), "restored state must be removed"
    before = list(sleeps)
    assert service.restore(incident_id="unit")["already_restored"] is True
    assert sleeps == before


@pytest.mark.parametrize("cgroup", ["owned", "foreign", "root", "invalid", "unicode"])
def test_device_holder_cleanup_requires_cgroup_ownership_even_when_process_name_is_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cgroup: str
) -> None:
    clock = Clock(step=0.1)
    monkeypatch.setattr(quiesce_module, "time", clock)
    proc = tmp_path / "proc" / "100"
    (proc / "fd").mkdir(parents=True)
    descriptor = proc / "fd" / "3"
    descriptor.symlink_to("/dev/nvidia0")
    (proc / "comm").write_bytes(b"\xff")
    (proc / "cgroup").write_bytes(
        {
            "owned": b"invalid\n0::/\n0::\n0::/training/owned/worker/\n",
            "foreign": b"0::/training/other\n",
            "root": b"0::/\n",
            "invalid": b"invalid\n",
            "unicode": b"\xff",
        }[cgroup]
    )
    runner = ServiceRunner(active={"kubelet"})

    def sleep(seconds: float) -> None:
        descriptor.unlink(missing_ok=True)
        clock.sleep(seconds)

    service = manager(tmp_path, runner, sleeper=sleep, device_sweep_processes=())
    receipt = service.quiesce(
        incident_id="unit",
        workflow_request_id="workflow",
        target_device_paths={"/dev/nvidia0"},
        workload_cgroup_paths={"/training/owned", "/", ""},
    )
    kills = [argv for argv in runner.commands if argv[0] == "kill"]
    assert kills == ([["kill", "--signal", "TERM", "100"]] if cgroup == "owned" else [])
    assert len(receipt["swept_device_holders"]) == int(cgroup == "owned")
    assert len(receipt["unswept_device_holders"]) == int(cgroup != "owned")
    report = (
        receipt["swept_device_holders"]
        if cgroup == "owned"
        else receipt["unswept_device_holders"]
    )[0]
    assert report["process_name"] == "unknown"
    assert report["devices"] == "/dev/nvidia0"
    service.restore(incident_id="unit")
    assert not Path(receipt["state_path"]).exists(), (
        "scoped cleanup must finish restore"
    )


def test_successful_start_without_active_service_keeps_restore_state_and_timer(
    tmp_path: Path,
) -> None:
    class InactiveRunner(ServiceRunner):
        def __call__(self, argv: list[str], **kwargs: Any) -> CompletedProcess:
            if argv[:2] == ["systemctl", "start"]:
                self.commands.append(argv)
                return CompletedProcess(argv, 0, "", "")
            return super().__call__(argv, **kwargs)

    runner = InactiveRunner(active={"kubelet"})
    service = manager(tmp_path, runner)
    receipt = service.quiesce(incident_id="unit", workflow_request_id="workflow")
    runner.commands.clear()
    with pytest.raises(RuntimeError, match="restored services are not active"):
        service.restore(incident_id="unit")
    state = json.loads(Path(receipt["state_path"]).read_text())
    assert state["phase"] == "RESTORE_FAILED"
    assert ["systemctl", "stop", receipt["timer_unit"]] not in runner.commands
    assert state["reset_issued"] is None


def test_first_boot_without_owned_quiesce_state_performs_no_restoration(
    tmp_path: Path,
) -> None:
    runner = ServiceRunner(active={"kubelet"})
    service = manager(tmp_path, runner)
    assert service.reconcile_after_boot() == {
        "boot_id": "boot-new",
        "restored": [],
        "kept": [],
        "failed": [],
    }
    assert runner.commands == []
