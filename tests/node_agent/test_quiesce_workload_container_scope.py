"""QUIESCE sweeps a managed job's own survivors, found by its runtime container ids.

The device sweep of ``QUIESCE_GPU_SERVICES`` may TERM/KILL a holder of the
target GPU that runs in the affected workload's cgroup
(``workload_cgroup_paths_by_node``). The control plane knew a container's
cgroup only from a ``gpu-fault.io/cgroup-path`` Pod annotation that nothing
sets, so on every managed Pod the scope was empty: a process of the stopped job
that still held the GPU was left alive ("outside_workload_cgroup_and_process_
whitelist") and VERIFY_NO_GPU_CLIENTS refused the reset on it -- fail closed,
the reset never ran, but a recovery the design allows did not happen either.

The plan now carries the attempt's runtime container ids per node
(``workload_container_ids_by_node``, the kubelet's ``containerID``); the Agent
finds each container's cgroup in its own ``/proc/<pid>/cgroup`` with the same
resolver the hung chain uses and adds it to the sweep scope. Stricter than the
read-only chain: a value it cannot read refuses the quiesce before anything
stops. A container with no process here adds nothing, and a holder left
outside the scope still meets VERIFY, which refuses the reset on it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.node_agent import NodeActionStatus
from gpu_fault.node_agent.quiesce import GpuServiceQuiesceManager
from tests.node_agent._support import (
    ServiceRunner,
    command,
    envelope,
    node_action_executor,
)

TRAINER = hashlib.sha256(b"trainer-container").hexdigest()
PEER = hashlib.sha256(b"second-trainer-container").hexdigest()
OTHER = hashlib.sha256(b"another-job-container").hexdigest()
POD = "kubepods-burstable-pod835a5f72_cf56_4102_912f_3dbb2ddfa995.slice"
POD_DIR = f"/kubepods.slice/kubepods-burstable.slice/{POD}"
TARGET = "/dev/nvidia0"
SIBLING = "/dev/nvidia1"


def scope(identifier: str, runtime: str = "cri-containerd") -> str:
    """The cgroup containerd's systemd driver gives a container (cgroup v2)."""

    return f"{POD_DIR}/{runtime}-{identifier}.scope"


def holder(proc: Path, pid: int, cgroup: str, device: str = TARGET) -> None:
    """A process in ``cgroup`` with ``device`` open."""

    directory = proc / str(pid)
    (directory / "fd").mkdir(parents=True)
    (directory / "comm").write_text("python\n")
    (directory / "cgroup").write_text(f"0::{cgroup}\n")
    (directory / "fd" / "7").symlink_to(device)


class ProcessRunner(ServiceRunner):
    """Services as ``ServiceRunner`` does; ``kill`` ends the fake process."""

    def __init__(self, proc: Path) -> None:
        super().__init__(active={"kubelet"})
        self.proc = proc

    def __call__(self, value, *, check=False, **kwargs):
        if value[:2] == ["kill", "--signal"]:
            shutil.rmtree(self.proc / value[-1], ignore_errors=True)
        return super().__call__(value, check=check, **kwargs)

    def signalled(self) -> list[str]:
        return [item[-1] for item in self.commands if item[:2] == ["kill", "--signal"]]


def agent(tmp_path: Path, runner: ProcessRunner):
    proc = runner.proc
    manager = GpuServiceQuiesceManager(
        state_dir=str(tmp_path / "quiesce"),
        services=("kubelet",),
        processes=(),
        containers=(),
        device_sweep_processes=(),
        failsafe_seconds=30,
        retry_seconds=10,
        settle_seconds=0,
        restore_settle_seconds=0,
        device_sweep_timeout_seconds=5,
        proc_root=str(proc),
        restore_command="/opt/gpu-fault/restore",
        runner=runner,
        sleeper=lambda _: None,
    )
    return node_action_executor(
        tmp_path,
        "actions.db",
        allowed_operations={Op.QUIESCE_GPU_SERVICES, Op.VERIFY_NO_GPU_CLIENTS},
        reset_enabled=True,
        service_quiesce_enabled=True,
        quiesce_manager=manager,
        runner=runner,
        proc_root=str(proc),
        gpu_device_path_finder=lambda: {"GPU-a": TARGET, "GPU-b": SIBLING},
        sleep=lambda _: None,
    )


def quiesce_then_verify(
    tmp_path: Path, runner: ProcessRunner, parameters: dict[str, Any]
) -> tuple[Any, Any]:
    node = agent(tmp_path, runner)
    quiesced = node.execute(
        envelope(command(Op.QUIESCE_GPU_SERVICES, parameters=parameters))
    )
    verified = node.execute(
        envelope(command(Op.VERIFY_NO_GPU_CLIENTS, command_id="workflow/verify/node-a"))
    )
    return quiesced, verified


def ids(*identifiers: str) -> dict[str, Any]:
    """The QUIESCE parameter the control plane plans for a managed job."""

    return {"workload_container_ids_by_node": {"node-a": list(identifiers)}}


def test_today_a_managed_survivor_is_left_alive_and_verify_refuses_the_reset(
    tmp_path: Path,
) -> None:
    """The plan of a managed Pod before this fix: no cgroup path, no id.

    Fail closed, not a safety defect: the quiesce reports ``quiesced`` (it
    never claims to have stopped the job -- STOP_WORKLOADS does that), the
    survivor is listed as unswept with its container, and VERIFY_NO_GPU_
    CLIENTS -- like the reset's own preflight after it -- refuses on it.
    """

    runner = ProcessRunner(tmp_path / "proc")
    holder(runner.proc, 4100, scope(TRAINER))

    quiesced, verified = quiesce_then_verify(tmp_path, runner, {})

    assert quiesced.status is NodeActionStatus.SUCCEEDED, quiesced
    assert runner.signalled() == [], "nothing outside a known scope is signalled"
    (left,) = quiesced.details["unswept_device_holders"]
    assert left["pid"] == "4100", left
    assert left["reason"] == "outside_workload_cgroup_and_process_whitelist", left
    assert left["container_id"] == TRAINER, left
    assert quiesced.details["workload_scope"] == {
        "basis": "UNAVAILABLE",
        "reason": "NO_CGROUP_PATH_OR_CONTAINER_ID",
    }, quiesced.details
    assert verified.status is NodeActionStatus.FAILED, verified
    assert "GPU-a:4100:python" in verified.error, verified.error


def test_the_attempts_container_ids_put_its_survivor_in_the_sweep(
    tmp_path: Path,
) -> None:
    runner = ProcessRunner(tmp_path / "proc")
    holder(runner.proc, 4100, scope(TRAINER))
    # The same container on the node's other GPU: never the sweep's business.
    holder(runner.proc, 4101, scope(TRAINER), device=SIBLING)

    quiesced, verified = quiesce_then_verify(
        tmp_path, runner, ids(f"containerd://{TRAINER}")
    )

    assert quiesced.status is NodeActionStatus.SUCCEEDED, quiesced
    assert runner.signalled() == ["4100"], runner.commands
    (swept,) = quiesced.details["swept_device_holders"]
    assert swept["pid"] == "4100" and swept["signal"] == "TERM", swept
    assert quiesced.details["unswept_device_holders"] == [], quiesced.details
    assert quiesced.details["workload_scope"] == {
        "basis": "RUNTIME_CONTAINER_ID",
        "container_count": 1,
        "resolved_container_count": 1,
        "unresolved_container_count": 0,
        "unresolved_container_ids": [],
        "cgroup_path_count": 1,
        "complete": True,
    }, quiesced.details
    state = json.loads(Path(quiesced.details["state_path"]).read_text())
    assert state["workload_cgroup_paths"] == [scope(TRAINER)], state
    assert verified.status is NodeActionStatus.SUCCEEDED, verified


def test_one_container_resolved_of_two_sweeps_only_that_ones_survivor(
    tmp_path: Path,
) -> None:
    runner = ProcessRunner(tmp_path / "proc")
    holder(runner.proc, 4100, scope(TRAINER))
    # PEER was stopped cleanly by STOP_WORKLOADS: no process of it is left.

    quiesced, verified = quiesce_then_verify(tmp_path, runner, ids(TRAINER, PEER))

    assert quiesced.status is NodeActionStatus.SUCCEEDED, quiesced
    assert runner.signalled() == ["4100"], runner.commands
    evidence = quiesced.details["workload_scope"]
    assert evidence["resolved_container_count"] == 1, evidence
    assert evidence["unresolved_container_ids"] == [PEER], evidence
    assert evidence["reason"] == "CONTAINER_CGROUP_NOT_FOUND", evidence
    assert evidence["complete"] is False, evidence
    assert verified.status is NodeActionStatus.SUCCEEDED, verified


def test_no_container_left_on_the_node_sweeps_nothing(tmp_path: Path) -> None:
    """The normal case after STOP_WORKLOADS: the job's containers are gone."""

    runner = ProcessRunner(tmp_path / "proc")
    (runner.proc / "1").mkdir(parents=True)
    (runner.proc / "1" / "cgroup").write_text("0::/init.scope\n")

    quiesced, verified = quiesce_then_verify(tmp_path, runner, ids(TRAINER, PEER))

    assert quiesced.status is NodeActionStatus.SUCCEEDED, quiesced
    assert runner.signalled() == [], runner.commands
    evidence = quiesced.details["workload_scope"]
    assert evidence["resolved_container_count"] == 0, evidence
    assert evidence["reason"] == "CONTAINER_CGROUP_NOT_FOUND", evidence
    assert verified.status is NodeActionStatus.SUCCEEDED, verified


def test_a_scan_cut_short_leaves_the_survivor_to_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = ProcessRunner(tmp_path / "proc")
    holder(runner.proc, 4100, scope(TRAINER))
    monkeypatch.setattr(
        "gpu_fault.node_agent.operations.workload_scope.MAX_SCAN_PROCESSES", 0
    )

    quiesced, verified = quiesce_then_verify(tmp_path, runner, ids(TRAINER))

    assert quiesced.status is NodeActionStatus.SUCCEEDED, quiesced
    assert runner.signalled() == [], "a container the scan did not see is not swept"
    assert quiesced.details["workload_scope"]["reason"] == "PROCESS_SCAN_INCOMPLETE"
    (left,) = quiesced.details["unswept_device_holders"]
    assert left["pid"] == "4100", left
    assert verified.status is NodeActionStatus.FAILED, verified
    assert "GPU-a:4100:python" in verified.error, verified.error


def test_the_ids_never_reach_another_jobs_container_or_the_runtime_monitor(
    tmp_path: Path,
) -> None:
    runner = ProcessRunner(tmp_path / "proc")
    holder(runner.proc, 4100, scope(TRAINER))
    holder(runner.proc, 4200, scope(OTHER))
    # CRI-O's monitor is named after the container but runs beside it.
    holder(runner.proc, 4300, f"{POD_DIR}/crio-conmon-{TRAINER}.scope")
    holder(runner.proc, 4400, "/system.slice/customer-monitor.service")

    quiesced, verified = quiesce_then_verify(tmp_path, runner, ids(TRAINER))

    assert quiesced.status is NodeActionStatus.SUCCEEDED, quiesced
    assert runner.signalled() == ["4100"], runner.commands
    left = {
        item["pid"]: item["reason"]
        for item in quiesced.details["unswept_device_holders"]
    }
    assert left == {
        "4200": "outside_workload_cgroup_and_process_whitelist",
        "4300": "outside_workload_cgroup_and_process_whitelist",
        "4400": "outside_workload_cgroup_and_process_whitelist",
    }, quiesced.details
    assert verified.status is NodeActionStatus.FAILED, verified
    for pid in ("4200", "4300", "4400"):
        assert f"GPU-a:{pid}:python" in verified.error, verified.error


def test_an_explicit_cgroup_path_and_a_runtime_id_are_one_scope(tmp_path: Path) -> None:
    runner = ProcessRunner(tmp_path / "proc")
    holder(runner.proc, 4100, scope(TRAINER))
    holder(runner.proc, 4500, "/training/annotated-job")

    quiesced, verified = quiesce_then_verify(
        tmp_path,
        runner,
        {
            "workload_cgroup_paths_by_node": {"node-a": ["/training/annotated-job"]},
            **ids(TRAINER),
        },
    )

    assert sorted(runner.signalled()) == ["4100", "4500"], runner.commands
    assert quiesced.details["workload_scope"]["explicit_cgroup_path_count"] == 1
    assert verified.status is NodeActionStatus.SUCCEEDED, verified


@pytest.mark.parametrize(
    "value",
    [
        pytest.param([TRAINER], id="not-a-mapping"),
        pytest.param({"node-a": TRAINER}, id="not-a-list"),
        pytest.param({"node-a": ["abc123"]}, id="short-id"),
        pytest.param({"node-a": [TRAINER.upper()]}, id="mixed-case"),
        pytest.param({"node-a": [None]}, id="null"),
        pytest.param({"node-a": [TRAINER] * 65}, id="over-the-bound"),
    ],
)
def test_a_container_id_this_agent_cannot_read_refuses_before_anything_stops(
    tmp_path: Path, value: Any
) -> None:
    runner = ProcessRunner(tmp_path / "proc")
    holder(runner.proc, 4100, scope(TRAINER))
    node = agent(tmp_path, runner)

    refused = node.execute(
        envelope(
            command(
                Op.QUIESCE_GPU_SERVICES,
                parameters={"workload_container_ids_by_node": value},
            )
        )
    )

    assert refused.status is NodeActionStatus.FAILED, refused
    assert "workload_container_ids_by_node" in refused.error, refused.error
    assert not any(
        item[0] == "systemd-run" or item[:2] == ["systemctl", "stop"]
        for item in runner.commands
    ), "nothing is armed or stopped for a plan this Agent cannot read"
    assert runner.signalled() == [], runner.commands


def test_another_nodes_ids_are_not_this_nodes_scope(tmp_path: Path) -> None:
    runner = ProcessRunner(tmp_path / "proc")
    holder(runner.proc, 4100, scope(TRAINER))

    quiesced, _ = quiesce_then_verify(
        tmp_path,
        runner,
        {"workload_container_ids_by_node": {"node-b": [TRAINER], "node-a": []}},
    )

    assert quiesced.status is NodeActionStatus.SUCCEEDED, quiesced
    assert runner.signalled() == [], runner.commands
    assert quiesced.details["workload_scope"]["basis"] == "UNAVAILABLE"
