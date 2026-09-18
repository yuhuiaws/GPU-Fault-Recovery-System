from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from gpu_fault.node_agent import GpuServiceQuiesceManager

STARTUP_TIMEOUT_SECONDS = 5.0
STARTUP_MAX_BYTES = 4096


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


def close_streams(process: subprocess.Popen[str]) -> None:
    with ExitStack() as cleanup:
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                cleanup.callback(stream.close)


def wait_ready(process: subprocess.Popen[str], name: str, device: str) -> None:
    expected = f"ready:{process.pid}:{device}\n".encode("utf-8")
    output = bytearray()
    errors = bytearray()
    stdout_closed = False
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    prefix = f"holder {name} failed to start"
    with selectors.DefaultSelector() as selector:
        for stream in (process.stdout, process.stderr):
            if stream is None:
                raise RuntimeError(f"{prefix}: startup pipe is missing")
            selector.register(stream, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"{prefix}: readiness timed out")
            events = selector.select(remaining)
            if not events:
                raise RuntimeError(f"{prefix}: readiness timed out")
            for key, _ in events:
                # This is the only reader; a ready pipe permits a bounded raw read
                # even when the child has not finished a line.
                chunk = os.read(
                    key.fd, STARTUP_MAX_BYTES + 1 - len(output) - len(errors)
                )
                if not chunk:
                    selector.unregister(key.fileobj)
                    if key.fileobj is process.stdout:
                        stdout_closed = True
                    continue
                buffer = output if key.fileobj is process.stdout else errors
                buffer.extend(chunk)
                if len(output) + len(errors) > STARTUP_MAX_BYTES:
                    raise RuntimeError(f"{prefix}: startup output limit exceeded")
            if b"\n" in output:
                if output != expected:
                    raise RuntimeError(f"{prefix}: readiness identity mismatch")
                if process.poll() is not None:
                    raise RuntimeError(f"{prefix}: holder already exited")
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"{prefix}: readiness timed out")
                return
            if stdout_closed:
                detail = errors.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"{prefix}: stdout ended before readiness: {detail}")


def start_holder(name: str, device: str) -> subprocess.Popen[str]:
    process = subprocess.Popen(
        [sys.executable, "-c", HELPER, name, device],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    with ExitStack() as cleanup:
        cleanup.callback(close_streams, process)
        cleanup.callback(process.wait, timeout=5)
        cleanup.callback(process.kill)
        wait_ready(process, name, device)
        cleanup.pop_all()
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


def settled(process: subprocess.Popen[str], timeout: float = 5.0) -> bool:
    """Whether ``process`` exited within ``timeout`` seconds.

    A fixed ``sleep`` before ``alive()`` raced the SIGTERM the sweep just sent:
    a holder still tearing down 200ms later read as "survived" and failed the
    probe, while a holder that took 200ms to *start* dying read as swept. The
    process's own exit is the fact being tested, so wait for it, bounded.
    """

    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    return True


def stop(process: subprocess.Popen[str]) -> None:
    try:
        if not alive(process):
            return
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    finally:
        close_streams(process)


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
    with ExitStack() as cleanup:
        whitelist = start_holder("nvidia-persiste", target)
        cleanup.callback(stop, whitelist)
        same_gpu = start_holder("unrelated-gpu", target)
        cleanup.callback(stop, same_gpu)
        other_gpu = start_holder("unrelated-gpu", other)
        cleanup.callback(stop, other_gpu)
        swept, skipped = manager._sweep_device_holders(
            target_device_paths={target},
            workload_cgroup_paths=set(),
        )
        # The swept holder must actually exit; the untouched ones must still be
        # running once it has.
        whitelist_exited = settled(whitelist)
        first_swept = sorted(str(item["pid"]) for item in swept)
        first_skipped = sorted(str(item["pid"]) for item in skipped)
        first: dict[str, Any] = {
            "swept_pids": first_swept,
            "skipped_pids": first_skipped,
            "whitelist_alive": not whitelist_exited,
            "same_gpu_alive": alive(same_gpu),
            "other_gpu_alive": alive(other_gpu),
        }
        if str(whitelist.pid) not in first_swept:
            raise AssertionError(first)
        if str(same_gpu.pid) not in first_skipped:
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
        same_gpu_exited = settled(same_gpu)
        second_swept = sorted(str(item["pid"]) for item in swept)
        second: dict[str, Any] = {
            "swept_pids": second_swept,
            "skipped_pids": sorted(str(item["pid"]) for item in skipped),
            "same_gpu_alive": not same_gpu_exited,
            "other_gpu_alive": alive(other_gpu),
            "workload_cgroup_paths": sorted(workload_paths),
        }
        if str(same_gpu.pid) not in second_swept:
            raise AssertionError(second)
        if second["same_gpu_alive"] or not second["other_gpu_alive"]:
            raise AssertionError(second)
    print(json.dumps({"first": first, "second": second}, sort_keys=True))


if __name__ == "__main__":
    main()
