from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.lifecycle import ShutdownCoordinator

if __package__:
    from .acceptance_runner_common import write_json_atomic
else:
    from acceptance_runner_common import write_json_atomic


ROOT = Path(__file__).resolve().parents[3]
CASE_ID = "GF-REGIONAL-HA-007"
WORKER_THREAD_NAME = "ha007-processor-request"
GENERATED = ROOT / "deploy/control-plane/regional/generated"
CONTROL_WORKER_DEPLOYMENT = GENERATED / "gpu-fault-control-worker.yaml"
CONTROL_WORKER_CONFIG = GENERATED / "gpu-fault-control-worker-config-core.yaml"
LIFESPAN_VARIABLE = "GPU_FAULT_LIFESPAN_SHUTDOWN_MAX_SECONDS"
# 30/70/110 sit inside the control-worker lifespan budget; the last one is
# longer than the budget so the coordinator's give-up path is exercised too, not
# only asserted from the YAML.
DEFAULT_DURATIONS = [30.0, 70.0, 110.0, 140.0]


def control_worker_budgets(root: Path = ROOT) -> dict[str, Any]:
    """The lifespan budget and Pod grace the generated control-worker manifest declares.

    Read from the rendered manifests rather than kept as constants here: the
    case exists to show the two numbers leave real margin, so a renderer change
    that moves either must move this case with it.
    """

    generated = root / "deploy/control-plane/regional/generated"
    deployment_path = generated / CONTROL_WORKER_DEPLOYMENT.name
    config_path = generated / CONTROL_WORKER_CONFIG.name
    grace: int | None = None
    for document in yaml.safe_load_all(deployment_path.read_text(encoding="utf-8")):
        if not isinstance(document, dict) or document.get("kind") != "Deployment":
            continue
        grace = (
            document.get("spec", {})
            .get("template", {})
            .get("spec", {})
            .get("terminationGracePeriodSeconds")
        )
    budget: str | None = None
    for document in yaml.safe_load_all(config_path.read_text(encoding="utf-8")):
        if not isinstance(document, dict) or document.get("kind") != "ConfigMap":
            continue
        budget = (document.get("data") or {}).get(LIFESPAN_VARIABLE)
    if grace is None:
        raise RuntimeError(
            f"{deployment_path} declares no terminationGracePeriodSeconds"
        )
    if budget is None:
        raise RuntimeError(f"{config_path} declares no {LIFESPAN_VARIABLE}")
    return {
        "lifespan_budget_seconds": float(budget),
        "kubernetes_grace_seconds": float(grace),
        "sources": {
            "lifespan_budget_seconds": str(config_path.relative_to(root)),
            "kubernetes_grace_seconds": str(deployment_path.relative_to(root)),
        },
    }


def expected_outcome(duration: float, lifespan_budget_seconds: float) -> str:
    """``completed`` inside the budget, ``deadline-exceeded`` beyond it.

    ``ShutdownCoordinator.join`` waits at most ``max_seconds`` for the request
    thread and records the thread as a failure if it is still alive; that is
    the documented behaviour for a request longer than the budget.
    """

    return "completed" if duration <= lifespan_budget_seconds else "deadline-exceeded"


def _child(
    duration: float,
    started_file: Path,
    result_file: Path,
    *,
    lifespan_budget_seconds: float,
    signal_timeout_seconds: float,
) -> int:
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
    # Daemon, as the real processor threads are: once the coordinator has
    # given up on it the process exits without waiting for it.
    worker = Thread(target=process_request, name=WORKER_THREAD_NAME, daemon=True)
    worker.start()
    started_file.write_text("started\n")
    started_file.chmod(0o600)
    # A bounded wait: if the parent died before sending SIGTERM this process
    # must not stay behind as an orphan holding the request.
    if not shutdown_requested.wait(timeout=signal_timeout_seconds):
        write_json_atomic(
            result_file,
            {
                "completed": 0,
                "duration_seconds": duration,
                "shutdown_seconds": None,
                "shutdown_failures": [],
                "signal_timed_out": True,
            },
        )
        return 3
    shutdown_started = time.monotonic()
    coordinator = ShutdownCoordinator(lifespan_budget_seconds)
    coordinator.join(worker, worker.name)
    elapsed = time.monotonic() - shutdown_started
    with lock:
        completed = completions
    write_json_atomic(
        result_file,
        {
            "completed": completed,
            "duration_seconds": duration,
            "shutdown_seconds": round(elapsed, 3),
            "shutdown_failures": list(coordinator.failures),
            "deadline_exceeded": bool(coordinator.failures),
            "signal_timed_out": False,
        },
    )
    # The lifespan raises on a missed deadline and uvicorn exits non-zero; the
    # probe mirrors that exit status after the evidence is on disk.
    try:
        coordinator.raise_if_failed()
    except RuntimeError:
        return 1
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


def _child_environment() -> dict[str, str]:
    env = dict(os.environ)
    source = str(ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = source if not existing else source + os.pathsep + existing
    return env


def evaluate_run(
    result: dict[str, Any],
    *,
    lifespan_budget_seconds: float,
    kubernetes_grace_seconds: float,
) -> list[str]:
    """Why one child run did not behave as the coordinator documents."""

    errors = []
    outcome = expected_outcome(
        float(result["duration_seconds"]), lifespan_budget_seconds
    )
    if result.get("signal_timed_out"):
        errors.append("child never received SIGTERM")
        return errors
    if float(result["elapsed_seconds"]) > kubernetes_grace_seconds:
        errors.append("Pod exit exceeded terminationGracePeriodSeconds")
    if outcome == "completed":
        if result["returncode"] != 0:
            errors.append("in-budget request exited non-zero")
        if result["completed"] != 1:
            errors.append("in-budget request did not complete before exit")
        if result["shutdown_failures"]:
            errors.append("in-budget request was reported as a shutdown failure")
    else:
        if result["shutdown_failures"] != [WORKER_THREAD_NAME]:
            errors.append(
                "over-budget request was not reported as the coordinator's failure"
            )
        if result["completed"] != 0:
            errors.append("over-budget request completed inside the budget")
        if result["returncode"] == 0:
            errors.append("over-budget request exited zero; lifespan did not raise")
        if float(result["shutdown_seconds"] or 0) < lifespan_budget_seconds:
            errors.append("coordinator gave up before the lifespan budget elapsed")
    return errors


def run_probe(
    run_dir: Path,
    durations: list[float],
    *,
    budgets: dict[str, Any] | None = None,
) -> dict:
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    budgets = budgets or control_worker_budgets()
    lifespan_budget = float(budgets["lifespan_budget_seconds"])
    grace = float(budgets["kubernetes_grace_seconds"])
    results = []
    errors: list[str] = []
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
                    "--lifespan-budget-seconds",
                    str(lifespan_budget),
                    "--signal-timeout-seconds",
                    str(grace),
                ],
                stdout=output,
                stderr=subprocess.STDOUT,
                env=_child_environment(),
            )
            _wait_for_file(started_file, process)
            signal_started = time.monotonic()
            process.send_signal(signal.SIGTERM)
            try:
                returncode = process.wait(timeout=grace + 5)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait(timeout=10)
        elapsed = time.monotonic() - signal_started
        if not result_file.is_file():
            errors.append(
                f"run {index}: child wrote no result (returncode={returncode})"
            )
            continue
        result = json.loads(result_file.read_text())
        result.update(
            {
                "elapsed_seconds": round(elapsed, 3),
                "index": index,
                "returncode": returncode,
                "expected_outcome": expected_outcome(duration, lifespan_budget),
            }
        )
        run_errors = evaluate_run(
            result,
            lifespan_budget_seconds=lifespan_budget,
            kubernetes_grace_seconds=grace,
        )
        result["errors"] = run_errors
        errors.extend(f"run {index}: {item}" for item in run_errors)
        results.append(result)
    report = {
        "case_id": CASE_ID,
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "kubernetes_grace_seconds": grace,
        "lifespan_budget_seconds": lifespan_budget,
        "budget_sources": budgets.get("sources"),
        "runs": results,
    }
    write_json_atomic(run_dir / f"{CASE_ID}.json", report)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--run-dir", type=Path)
    value.add_argument(
        "--durations",
        type=float,
        nargs="+",
        default=list(DEFAULT_DURATIONS),
    )
    value.add_argument("--child", action="store_true")
    value.add_argument("--duration", type=float)
    value.add_argument("--started-file", type=Path)
    value.add_argument("--result-file", type=Path)
    value.add_argument("--lifespan-budget-seconds", type=float)
    value.add_argument("--signal-timeout-seconds", type=float, default=300.0)
    return value


def main() -> int:
    arguments = parser().parse_args()
    if arguments.child:
        if (
            arguments.duration is None
            or arguments.started_file is None
            or arguments.result_file is None
            or arguments.lifespan_budget_seconds is None
        ):
            raise SystemExit("--child requires duration, budget and state files")
        return _child(
            arguments.duration,
            arguments.started_file,
            arguments.result_file,
            lifespan_budget_seconds=arguments.lifespan_budget_seconds,
            signal_timeout_seconds=arguments.signal_timeout_seconds,
        )
    if arguments.run_dir is None:
        raise SystemExit("--run-dir is required")
    report = run_probe(arguments.run_dir, arguments.durations)
    print(json.dumps(report, indent=2))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
