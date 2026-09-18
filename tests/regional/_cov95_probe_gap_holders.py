"""Owned local children and a closed sweep transport for holder probe tests."""

from __future__ import annotations

import signal
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
PROBE = lazy_script_module(ROOT / "scripts/e2e/regional/node_holder_scope_probe.py")
WORKLOAD_PATH = "/fixture/workload-owned"
MANAGER_OPTIONS = {
    "state_dir": "/tmp/gpu-fault-holder-scope-probe",
    "services": ("kubelet",),
    "failsafe_seconds": 30,
    "retry_seconds": 10,
    "settle_seconds": 0,
    "restore_settle_seconds": 0,
    "device_sweep_timeout_seconds": 1,
    "proc_root": "/proc",
    "restore_command": "/bin/true",
}


class HolderHarness:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        self.target = str(root / "target-device")
        self.other = str(root / "other-device")
        Path(self.target).touch()
        Path(self.other).touch()
        self.processes: list[subprocess.Popen[str]] = []
        self.launches: list[list[str]] = []
        self.stopped: list[int] = []
        self.sweeps: list[dict[str, set[str]]] = []
        self.manager_options: list[dict[str, Any]] = []
        self.proc_files: dict[str, Path] = {}
        self.proc_reads: list[str] = []
        self.outcome = "healthy"
        self.spawn_failure_at: int | None = None
        self.wait_before_return = False
        self.deadman_seconds: float | None = None
        self.deadmen: list[threading.Timer] = []
        self.deadman_fired: list[int] = []
        self.launcher = subprocess.Popen
        self.original_settled = PROBE.settled
        self.original_stop = PROBE.stop
        monkeypatch.setenv("TARGET_GPU_DEVICE", self.target)
        monkeypatch.setenv("OTHER_GPU_DEVICE", self.other)
        monkeypatch.setattr(PROBE, "Path", self.path)
        monkeypatch.setattr(PROBE, "GpuServiceQuiesceManager", self.manager)
        monkeypatch.setattr(PROBE, "settled", self.settled)
        monkeypatch.setattr(PROBE, "stop", self.stop)
        monkeypatch.setattr(
            PROBE,
            "subprocess",
            SimpleNamespace(
                Popen=self.popen,
                PIPE=subprocess.PIPE,
                TimeoutExpired=subprocess.TimeoutExpired,
            ),
        )

    def popen(self, command: list[str], **kwargs: Any) -> subprocess.Popen[str]:
        assert command[:3] == [sys.executable, "-c", PROBE.HELPER]
        assert len(command) == 5
        assert command[3] in {"nvidia-persiste", "unrelated-gpu"}
        assert command[4] in {self.target, self.other}
        assert kwargs == {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
        }
        self.launches.append(command)
        if len(self.launches) == self.spawn_failure_at:
            raise OSError("fixture holder startup refused")
        process = self.launcher(command, **kwargs)
        self.processes.append(process)
        if self.deadman_seconds is not None:
            watchdog = threading.Timer(
                self.deadman_seconds, self.abort_startup, args=(process,)
            )
            self.deadmen.append(watchdog)
            watchdog.start()
        cgroup_file = self.root / f"cgroup-{process.pid}"
        cgroup_file.write_text(f"0::{WORKLOAD_PATH}/\n", encoding="utf-8")
        self.proc_files[f"/proc/{process.pid}/cgroup"] = cgroup_file
        if self.wait_before_return:
            process.wait(timeout=5)
        return process

    def abort_startup(self, process: subprocess.Popen[str]) -> None:
        assert process in self.processes
        if process.poll() is None:
            self.deadman_fired.append(process.pid)
            process.kill()
            process.wait(timeout=5)

    def path(self, value: str) -> Path:
        self.proc_reads.append(value)
        assert value in self.proc_files, "only fixture cgroups may be read"
        return self.proc_files[value]

    def manager(self, **kwargs: Any) -> SimpleNamespace:
        self.manager_options.append(kwargs)
        assert kwargs == MANAGER_OPTIONS
        return SimpleNamespace(_sweep_device_holders=self.sweep)

    def terminate(self, process: subprocess.Popen[str]) -> None:
        assert process in self.processes, "never signal a foreign process"
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=5)

    def sweep(
        self, *, target_device_paths: set[str], workload_cgroup_paths: set[str]
    ) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
        assert target_device_paths == {self.target}
        self.sweeps.append(
            {
                "target_device_paths": set(target_device_paths),
                "workload_cgroup_paths": set(workload_cgroup_paths),
            }
        )
        whitelist, same, other = self.processes
        if len(self.sweeps) == 1:
            assert workload_cgroup_paths == set()
            if self.outcome == "first-io-error":
                raise OSError("fixture first sweep failed")
            if self.outcome != "whitelist-survives":
                self.terminate(whitelist)
            if self.outcome == "same-stopped-first":
                self.terminate(same)
            if self.outcome == "other-stopped-first":
                self.terminate(other)
            if self.outcome == "missing-cgroup":
                self.proc_files[f"/proc/{same.pid}/cgroup"].unlink()
            return (
                [] if self.outcome == "missing-whitelist" else [{"pid": whitelist.pid}],
                [] if self.outcome == "missing-skip" else [{"pid": same.pid}],
            )
        assert len(self.sweeps) == 2
        assert workload_cgroup_paths == {WORKLOAD_PATH}
        if self.outcome == "second-io-error":
            raise OSError("fixture second sweep failed")
        if self.outcome != "same-survives-second":
            self.terminate(same)
        if self.outcome == "other-stopped-second":
            self.terminate(other)
        return (
            [] if self.outcome == "missing-second-sweep" else [{"pid": same.pid}],
            [],
        )

    def settled(self, process: subprocess.Popen[str], timeout: float = 5.0) -> bool:
        assert process in self.processes
        return bool(self.original_settled(process, timeout=min(timeout, 0.2)))

    def stop(self, process: subprocess.Popen[str]) -> None:
        assert process in self.processes
        self.stopped.append(process.pid)
        self.original_stop(process)

    def close(self) -> None:
        for watchdog in self.deadmen:
            watchdog.cancel()
            watchdog.join(timeout=6)
            assert not watchdog.is_alive(), "the fixture watchdog must finish cleanup"
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
            assert process.poll() is not None
            assert process.stdout is not None and process.stdout.closed
            assert process.stderr is not None and process.stderr.closed
        assert not self.deadman_fired, "probe startup exceeded its fixture deadline"


@pytest.fixture
def holders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[HolderHarness]:
    harness = HolderHarness(tmp_path, monkeypatch)
    try:
        yield harness
    finally:
        harness.close()
