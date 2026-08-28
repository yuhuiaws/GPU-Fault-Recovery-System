from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import Event, Lock, Thread
import time

from gpu_fault.lifecycle import ShutdownCoordinator


CASE_ID = "GF-REGIONAL-HA-007"
LIFESPAN_BUDGET_SECONDS = 130
KUBERNETES_GRACE_SECONDS = 240


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    path.chmod(0o600)


def _child(duration: float, started_file: Path, result_file: Path) -> int:
    shutdown_requested = Event()
    lock = Lock()
    completions = 0

    def handle_signal(_signum, _frame) -> None:
        shutdown_requested.set()

    def process_request() -> None:
        nonlocal completions
        time.sleep(duration)
        with lock:
            completions += 1

    signal.signal(signal.SIGTERM, handle_signal)
    worker = Thread(target=process_request, name="ha007-processor-request")
    worker.start()
    started_file.write_text("started\n")
    started_file.chmod(0o600)
    shutdown_requested.wait()
    shutdown_started = time.monotonic()
    coordinator = ShutdownCoordinator(LIFESPAN_BUDGET_SECONDS)
    coordinator.join(worker, worker.name)
    coordinator.raise_if_failed()
    elapsed = time.monotonic() - shutdown_started
    with lock:
        completed = completions
    _write_json(
        result_file,
        {
            "completed": completed,
            "duration_seconds": duration,
            "shutdown_seconds": round(elapsed, 3),
            "shutdown_failures": list(coordinator.failures),
        },
    )
    return int(completed != 1)


def _wait_for_file(path: Path, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            raise RuntimeError(
                f"HA-007 child exited before readiness: {process.returncode}"
            )
        time.sleep(0.02)
    raise RuntimeError("HA-007 child did not become ready")


def run_probe(run_dir: Path, durations: list[float]) -> dict:
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    results = []
    for index, duration in enumerate(durations):
        started_file = run_dir / f"started-{index}.txt"
        result_file = run_dir / f"result-{index}.json"
        log_file = run_dir / f"process-{index}.log"
        with log_file.open("w") as output:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--child",
                    "--duration",
                    str(duration),
                    "--started-file",
                    str(started_file),
                    "--result-file",
                    str(result_file),
                ],
                stdout=output,
                stderr=subprocess.STDOUT,
                env=os.environ,
            )
            _wait_for_file(started_file, process)
            signal_started = time.monotonic()
            process.send_signal(signal.SIGTERM)
            returncode = process.wait(timeout=KUBERNETES_GRACE_SECONDS)
        elapsed = time.monotonic() - signal_started
        result = json.loads(result_file.read_text())
        result.update(
            {
                "elapsed_seconds": round(elapsed, 3),
                "index": index,
                "returncode": returncode,
            }
        )
        if (
            returncode != 0
            or result["completed"] != 1
            or result["shutdown_failures"]
            or elapsed > KUBERNETES_GRACE_SECONDS
        ):
            raise RuntimeError(f"HA-007 shutdown probe failed: {result}")
        results.append(result)
    report = {
        "case_id": CASE_ID,
        "status": "PASS",
        "kubernetes_grace_seconds": KUBERNETES_GRACE_SECONDS,
        "lifespan_budget_seconds": LIFESPAN_BUDGET_SECONDS,
        "runs": results,
    }
    _write_json(run_dir / f"{CASE_ID}.json", report)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--run-dir", type=Path)
    value.add_argument(
        "--durations",
        type=float,
        nargs="+",
        default=[30, 70, 110, 30],
    )
    value.add_argument("--child", action="store_true")
    value.add_argument("--duration", type=float)
    value.add_argument("--started-file", type=Path)
    value.add_argument("--result-file", type=Path)
    return value


def main() -> int:
    arguments = parser().parse_args()
    if arguments.child:
        if (
            arguments.duration is None
            or arguments.started_file is None
            or arguments.result_file is None
        ):
            raise SystemExit("--child requires duration and state files")
        return _child(
            arguments.duration,
            arguments.started_file,
            arguments.result_file,
        )
    if arguments.run_dir is None:
        raise SystemExit("--run-dir is required")
    print(json.dumps(run_probe(arguments.run_dir, arguments.durations), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
