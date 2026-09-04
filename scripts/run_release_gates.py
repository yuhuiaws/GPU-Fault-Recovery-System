from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]


class ReleaseGateError(RuntimeError):
    pass


def gate_parallelism() -> int:
    configured = os.getenv("GPU_FAULT_RELEASE_GATE_PARALLELISM", "").strip()
    if configured:
        try:
            value = int(configured)
        except ValueError as exc:
            raise ReleaseGateError(
                "GPU_FAULT_RELEASE_GATE_PARALLELISM must be an integer"
            ) from exc
        if not 1 <= value <= 3:
            raise ReleaseGateError(
                "GPU_FAULT_RELEASE_GATE_PARALLELISM must be within 1..3"
            )
        return value
    return min(3, max(1, (os.cpu_count() or 1) // 4))


def static_parallelism_for_release() -> int:
    return min(4, max(1, (os.cpu_count() or 1) // 4))


def _gate_environment(name: str, cache_root: Path) -> dict[str, str]:
    environment = {
        **os.environ,
        "PYTHONPYCACHEPREFIX": str(cache_root / name / "pycache"),
        "PYTEST_ADDOPTS": (
            (
                os.environ.get("PYTEST_ADDOPTS", "").strip() + " "
                if os.environ.get("PYTEST_ADDOPTS", "").strip()
                else ""
            )
            + f"-o cache_dir={cache_root / name / 'pytest-cache'}"
        ),
    }
    if (
        name == "static"
        and not environment.get("GPU_FAULT_STATIC_GATE_PARALLELISM", "").strip()
    ):
        environment["GPU_FAULT_STATIC_GATE_PARALLELISM"] = str(
            static_parallelism_for_release()
        )
    return environment


def _stream_gate(
    name: str,
    command: Sequence[str],
    *,
    environment: dict[str, str],
    output_lock: threading.Lock,
) -> int:
    with output_lock:
        print(f"release-gates: starting {name}", file=sys.stderr, flush=True)
    process = subprocess.Popen(
        list(command),
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    assert process.stdout is not None
    for line in process.stdout:
        with output_lock:
            print(f"[{name}] {line}", end="", flush=True)
    returncode = process.wait()
    with output_lock:
        print(
            f"release-gates: {name} exited with status {returncode}",
            file=sys.stderr,
            flush=True,
        )
    return returncode


def _run_parallel(
    commands: dict[str, list[str]],
    *,
    cache_root: Path,
    max_workers: int,
) -> dict[str, int]:
    output_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _stream_gate,
                name,
                command,
                environment=_gate_environment(name, cache_root),
                output_lock=output_lock,
            ): name
            for name, command in commands.items()
        }
        failures: dict[str, int] = {}
        for future in as_completed(futures):
            status = future.result()
            if status:
                failures[futures[future]] = status
    return failures


def _raise_failures(failures: dict[str, int], *, description: str) -> None:
    if failures:
        raise ReleaseGateError(
            f"{description}: "
            + ", ".join(f"{name}={status}" for name, status in sorted(failures.items()))
        )


def run_release_gates(python: str) -> None:
    if not os.getenv("GPU_FAULT_TEST_POSTGRES_URL", "").strip():
        raise ReleaseGateError("GPU_FAULT_TEST_POSTGRES_URL is required")
    with tempfile.TemporaryDirectory(prefix="gpu-fault-release-gates-") as directory:
        cache_root = Path(directory)
        commands = {
            "postgres": ["make", "test-postgres-stress", f"PYTHON={python}"],
            "pytest": ["make", "test-parallel-release", f"PYTHON={python}"],
            "static": ["make", "check-static", f"PYTHON={python}"],
        }
        failures = _run_parallel(
            commands,
            cache_root=cache_root,
            max_workers=gate_parallelism(),
        )
        _raise_failures(failures, description="parallel release gate failed")
        subprocess.run(
            ["make", "artifact-check", f"PYTHON={python}"],
            cwd=ROOT,
            env={
                **os.environ,
                "PYTHONPYCACHEPREFIX": str(cache_root / "artifact"),
            },
            check=True,
        )
        subprocess.run(
            ["make", "python-cache-clean", f"PYTHON={python}"],
            cwd=ROOT,
            check=True,
        )


def run_check_gates(python: str) -> None:
    with tempfile.TemporaryDirectory(prefix="gpu-fault-check-gates-") as directory:
        cache_root = Path(directory)
        subprocess.run(
            ["make", "check-static", f"PYTHON={python}"],
            cwd=ROOT,
            env=_gate_environment("static", cache_root),
            check=True,
        )
        failures = _run_parallel(
            {
                "artifact": ["make", "artifact-check", f"PYTHON={python}"],
                "pytest": ["make", "test-parallel-release", f"PYTHON={python}"],
            },
            cache_root=cache_root,
            max_workers=2,
        )
        _raise_failures(failures, description="parallel check tail failed")
        subprocess.run(
            ["make", "python-cache-clean", f"PYTHON={python}"],
            cwd=ROOT,
            check=True,
        )


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--mode", choices=("check", "release"), default="release")
    options = parser.parse_args(arguments)
    try:
        if options.mode == "check":
            run_check_gates(options.python)
        else:
            run_release_gates(options.python)
    except (OSError, ReleaseGateError, subprocess.SubprocessError) as exc:
        print(f"release-gates: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
