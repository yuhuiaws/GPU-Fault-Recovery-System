from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from gpu_fault.node_agent import GpuServiceQuiesceManager


HELPER = r"""
import ctypes
import os
import signal
import sys
import time

name, device = sys.argv[1:3]
libc = ctypes.CDLL(None)
if libc.prctl(15, name.encode(), 0, 0, 0) != 0:
    raise OSError(ctypes.get_errno(), "prctl(PR_SET_NAME) failed")
fd = os.open(device, os.O_RDONLY)
print(f"ready:{os.getpid()}:{device}", flush=True)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
try:
    while True:
        time.sleep(60)
finally:
    os.close(fd)
"""


def start_holder(name: str, device: str) -> subprocess.Popen[str]:
    process = subprocess.Popen(
        [sys.executable, "-c", HELPER, name, device],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ready = process.stdout.readline().strip() if process.stdout else ""
    if not ready.startswith("ready:"):
        error = process.stderr.read() if process.stderr else ""
        raise RuntimeError(f"holder {name} failed to start: {ready} {error}")
    return process


def cgroup_paths(pid: int) -> set[str]:
    rows = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8").splitlines()
    return {
        parts[2].rstrip("/")
        for row in rows
        if len(parts := row.split(":", 2)) == 3
        and parts[2].strip()
        and parts[2].strip() != "/"
    }


def alive(process: subprocess.Popen[str]) -> bool:
    return process.poll() is None


def stop(process: subprocess.Popen[str]) -> None:
    if not alive(process):
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> None:
    target = os.environ.get("TARGET_GPU_DEVICE", "/dev/nvidia0")
    other = os.environ.get("OTHER_GPU_DEVICE", "/dev/nvidia1")
    manager = GpuServiceQuiesceManager(
        state_dir="/tmp/gpu-fault-holder-scope-probe",
        services=("kubelet",),
        failsafe_seconds=30,
        retry_seconds=10,
        settle_seconds=0,
        restore_settle_seconds=0,
        device_sweep_timeout_seconds=1,
        proc_root="/proc",
        restore_command="/bin/true",
    )
    whitelist = start_holder("nvidia-persiste", target)
    same_gpu = start_holder("unrelated-gpu", target)
    other_gpu = start_holder("unrelated-gpu", other)
    try:
        swept, skipped = manager._sweep_device_holders(
            target_device_paths={target},
            workload_cgroup_paths=set(),
        )
        time.sleep(0.2)
        first = {
            "swept_pids": sorted(item["pid"] for item in swept),
            "skipped_pids": sorted(item["pid"] for item in skipped),
            "whitelist_alive": alive(whitelist),
            "same_gpu_alive": alive(same_gpu),
            "other_gpu_alive": alive(other_gpu),
        }
        if str(whitelist.pid) not in first["swept_pids"]:
            raise AssertionError(first)
        if str(same_gpu.pid) not in first["skipped_pids"]:
            raise AssertionError(first)
        if (
            first["whitelist_alive"]
            or not first["same_gpu_alive"]
            or not first["other_gpu_alive"]
        ):
            raise AssertionError(first)

        workload_paths = cgroup_paths(same_gpu.pid)
        swept, skipped = manager._sweep_device_holders(
            target_device_paths={target},
            workload_cgroup_paths=workload_paths,
        )
        time.sleep(0.2)
        second = {
            "swept_pids": sorted(item["pid"] for item in swept),
            "skipped_pids": sorted(item["pid"] for item in skipped),
            "same_gpu_alive": alive(same_gpu),
            "other_gpu_alive": alive(other_gpu),
            "workload_cgroup_paths": sorted(workload_paths),
        }
        if str(same_gpu.pid) not in second["swept_pids"]:
            raise AssertionError(second)
        if second["same_gpu_alive"] or not second["other_gpu_alive"]:
            raise AssertionError(second)
        print(json.dumps({"first": first, "second": second}, sort_keys=True))
    finally:
        for process in (whitelist, same_gpu, other_gpu):
            stop(process)


if __name__ == "__main__":
    main()
