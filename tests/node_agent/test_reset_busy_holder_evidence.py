"""A reset that stays "In use by another client" names who held the GPU.

Live 2026-10-05 (gpu-eks-cluster, XID 46): quiesce stopped every configured
service and container, the sampled preflight saw no ``/dev/nvidiaN`` holder,
and all three resets failed busy -- with no holder anywhere in the evidence.
The last busy refusal now carries ``gpu_client_holders``: every process with
any NVIDIA device node open (``nvidiactl``, ``uvm``, a replaced node), its
Kubernetes container, the compute apps and the kernel modules in use.
"""

from __future__ import annotations

from pathlib import Path

from gpu_fault.node_agent.operations import holders
from gpu_fault.node_agent.protocol import NodeActionStatus
from tests.node_agent._support import (
    FakeRunner,
    WorkflowOperation,
    command,
    envelope,
    no_device_clients,
    node_action_executor,
)

CONTAINER_ID = "54bb525cc1d10b8406b9d81e39dbcc53bbeeaf649abd48d5ce9f055337809132"
POD_UID = "e747090b_fef5_4687_a913_2bcab51934c6"
TOKEN_LIKE = "AKIA-NOT-A-REAL-TOKEN-0123456789"


def _proc(tmp_path: Path, *, holder: bool = True) -> Path:
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "modules").write_text(
        "nvidia_peermem 16384 1 -\n"
        "nvidia_uvm 1507328 0 -\n"
        "nvidia 8933376 44 nvidia_uvm,nvidia_peermem\n"
        "efa 69632 0 -\n",
        encoding="ascii",
    )
    if holder:
        process = proc / "4242"
        (process / "fd").mkdir(parents=True)
        (process / "fd" / "3").symlink_to("/dev/nvidiactl")
        (process / "fd" / "4").symlink_to("/dev/nvidia0 (deleted)")
        (process / "fd" / "5").symlink_to("/var/log/exporter.log")
        (process / "comm").write_text("dcgm-exporter\n", encoding="ascii")
        (process / "cmdline").write_text(f"dcgm-exporter\0--token={TOKEN_LIKE}\0")
        (process / "cgroup").write_text(
            "0::/kubepods.slice/kubepods-besteffort.slice/"
            f"kubepods-besteffort-pod{POD_UID}.slice/"
            f"cri-containerd-{CONTAINER_ID}.scope\n",
            encoding="ascii",
        )
    # A non-holder with an unreadable fd directory is skipped, not an error.
    other = proc / "77"
    other.mkdir()
    (other / "comm").write_text("sleep\n", encoding="ascii")
    return proc


def _agent(tmp_path: Path, runner: FakeRunner, proc: Path):
    return node_action_executor(
        tmp_path,
        "busy.db",
        allowed_operations={WorkflowOperation.RESET_GPU},
        reset_enabled=True,
        runner=runner,
        proc_root=str(proc),
        device_client_finder=no_device_clients,
        gpu_device_path_finder=lambda: {"GPU-a": "/dev/nvidia0"},
        sleep=lambda _: None,
    )


def test_a_busy_reset_carries_every_nvidia_device_holder(tmp_path: Path) -> None:
    runner = FakeRunner(reset_error="GPU 00000000:59:00.0: In use by another client")
    agent = _agent(tmp_path, runner, _proc(tmp_path))

    result = agent.execute(envelope(command(WorkflowOperation.RESET_GPU)))

    assert result.status is NodeActionStatus.FAILED
    assert result.retryable is False
    assert len([item for item in runner.commands if "--gpu-reset" in item]) == 3, (
        "the busy retry budget is unchanged"
    )
    assert "In use by another client" in (result.error or "")
    assert (
        "device holders: 4242:dcgm-exporter:/dev/nvidia0 (deleted),/dev/nvidiactl "
        f"(container {CONTAINER_ID[:12]})"
    ) in (result.error or ""), result.error
    assert result.details["reset_failed"] == ["GPU-a"]
    assert result.details["gpu_reset_busy_attempts"] == 3
    evidence = result.details["gpu_client_holders"]
    assert evidence["target_gpu_uuids"] == ["GPU-a"]
    assert evidence["target_device_paths"] == ["/dev/nvidia0"]
    assert evidence["compute_clients"] == [] and evidence["compute_client_count"] == 0
    (holder,) = evidence["device_holders"]
    assert holder == {
        "pid": "4242",
        "process_name": "dcgm-exporter",
        "devices": "/dev/nvidia0 (deleted),/dev/nvidiactl",
        "deleted_device_node": True,
        "cgroup_paths": [
            "/kubepods.slice/kubepods-besteffort.slice/"
            f"kubepods-besteffort-pod{POD_UID}.slice/cri-containerd-{CONTAINER_ID}.scope"
        ],
        "kubernetes": True,
        "container_id": CONTAINER_ID,
        "pod_uid": POD_UID.replace("_", "-"),
    }
    assert evidence["device_holder_count"] == 1
    assert evidence["kernel_modules"] == [
        {"module": "nvidia_peermem", "refcount": 1, "used_by": []},
        {"module": "nvidia_uvm", "refcount": 0, "used_by": []},
        {
            "module": "nvidia",
            "refcount": 44,
            "used_by": ["nvidia_uvm", "nvidia_peermem"],
        },
    ]
    assert TOKEN_LIKE not in repr(result.details), "a command line is never read"
    assert agent.ledger.get(result.command_id) == result, "the replay is attributed too"


def test_a_busy_reset_with_no_visible_holder_says_so_and_names_the_modules(
    tmp_path: Path,
) -> None:
    runner = FakeRunner(reset_error="GPU 00000000:59:00.0: In use by another client")
    agent = _agent(tmp_path, runner, _proc(tmp_path, holder=False))

    result = agent.execute(envelope(command(WorkflowOperation.RESET_GPU)))

    assert result.status is NodeActionStatus.FAILED
    assert (
        "no process holds an NVIDIA device node in /proc; compute apps: 0; "
        "kernel modules in use: nvidia_peermem(refcount=1), nvidia(refcount=44)"
    ) in (result.error or ""), result.error
    evidence = result.details["gpu_client_holders"]
    assert evidence["device_holders"] == [] and evidence["device_holder_count"] == 0


def test_a_transient_busy_reset_that_recovers_carries_no_holder_evidence(
    tmp_path: Path,
) -> None:
    class BusyOnceRunner(FakeRunner):
        def __init__(self) -> None:
            super().__init__()
            self.busy = True

        def __call__(self, command, **kwargs):
            if "--gpu-reset" in command and self.busy:
                self.busy = False
                self.reset_error = "GPU 00000000:59:00.0: In use by another client"
                try:
                    return super().__call__(command, **kwargs)
                finally:
                    self.reset_error = None
            return super().__call__(command, **kwargs)

    runner = BusyOnceRunner()
    agent = _agent(tmp_path, runner, _proc(tmp_path))

    result = agent.execute(envelope(command(WorkflowOperation.RESET_GPU)))

    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    assert "gpu_client_holders" not in (result.details or {})
    assert result.details["reset_busy_refusals"] == 1


def test_holder_evidence_never_raises_and_records_each_probe_failure(
    tmp_path: Path,
) -> None:
    def failing_clients() -> list[dict[str, str]]:
        raise RuntimeError("nvidia-smi exited with status 9")

    def failing_paths() -> dict[str, str]:
        raise OSError("device map unavailable")

    evidence = holders.gpu_client_holder_evidence(
        proc_root=tmp_path / "missing-proc",
        gpu_uuids=["GPU-a", "GPU-a"],
        compute_clients=failing_clients,
        device_paths=failing_paths,
    )

    assert evidence["target_gpu_uuids"] == ["GPU-a"]
    assert evidence["compute_clients_error"] == "RuntimeError"
    assert evidence["target_device_paths_error"] == "OSError"
    assert evidence["device_holders"] == [] and evidence["kernel_modules"] == []
    assert holders.gpu_client_holder_summary(evidence) == (
        "no process holds an NVIDIA device node in /proc; compute apps: 0; "
        "kernel modules in use: none"
    )


def test_kubernetes_attribution_reads_only_what_the_cgroup_names() -> None:
    assert holders.kubernetes_attribution(["/system.slice/kubelet.service"]) == {
        "kubernetes": False
    }
    assert holders.kubernetes_attribution(
        ["/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-podx.slice"]
    ) == {"kubernetes": True}, "a pod slice that does not parse yields no ids"
    summary = holders.gpu_client_holder_summary(
        {
            "device_holders": [
                {"pid": str(index), "process_name": "p", "devices": "/dev/nvidia0"}
                for index in range(10)
            ]
        }
    )
    assert summary.endswith("(+2 more)"), summary
