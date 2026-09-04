from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]


class StaticGateError(RuntimeError):
    pass


@dataclass(frozen=True)
class GateGroup:
    commands: tuple[tuple[str, ...], ...]
    environment: Mapping[str, str] | None = None


def gate_parallelism() -> int:
    configured = os.getenv("GPU_FAULT_STATIC_GATE_PARALLELISM", "").strip()
    if configured:
        try:
            value = int(configured)
        except ValueError as exc:
            raise StaticGateError(
                "GPU_FAULT_STATIC_GATE_PARALLELISM must be an integer"
            ) from exc
        if not 1 <= value <= 10:
            raise StaticGateError(
                "GPU_FAULT_STATIC_GATE_PARALLELISM must be within 1..10"
            )
        return value
    return min(10, max(1, (os.cpu_count() or 1) // 2))


def gate_groups(python: str) -> dict[str, GateGroup]:
    quality_roots = ("src", "tests", "deploy", "scripts", "tools")
    pythonpath = str(ROOT / "src")
    return {
        "ruff": GateGroup(
            commands=(
                (
                    python,
                    "-m",
                    "ruff",
                    "format",
                    "--check",
                    *quality_roots,
                ),
                (python, "-m", "ruff", "check", *quality_roots),
                (python, "-m", "ruff", "check", "--select", "I", "tests"),
            )
        ),
        "mypy": GateGroup(commands=((python, "scripts/check-mypy-baseline.py"),)),
        "compile": GateGroup(
            commands=((python, "-m", "compileall", "-q", *quality_roots),)
        ),
        "architecture": GateGroup(
            commands=((python, "scripts/check-python-architecture.py"),)
        ),
        "contracts": GateGroup(
            commands=(
                (python, "scripts/check-mixin-contracts.py"),
                (python, "scripts/check-test-private-coupling.py"),
                (python, "scripts/check-test-source-assertions.py"),
                (python, "scripts/check-assert-messages.py"),
            )
        ),
        "safety": GateGroup(
            commands=(
                (python, "scripts/check-public-release.py"),
                (python, "scripts/check-artifacts.py"),
            )
        ),
        "deployment": GateGroup(
            commands=(
                (python, "tools/generate_nvidia_xid_policy.py", "--check"),
                (
                    "deploy/control-plane/tools/update-deployment-contracts.sh",
                    "--check",
                ),
                (
                    python,
                    "-m",
                    "gpu_fault.config_cli",
                    "validate",
                    "deploy/control-plane/regional/generated",
                ),
                (python, "scripts/check-deploy-layout.py"),
            ),
            environment={
                "PYTHONPATH": pythonpath,
                "PYTHON": python,
            },
        ),
        "docs": GateGroup(
            commands=(("make", "docs-static-check", f"PYTHON={python}"),)
        ),
        "yaml": GateGroup(
            commands=(
                (
                    python,
                    "-m",
                    "yamllint",
                    "--strict",
                    "-c",
                    ".yamllint",
                    "deploy",
                    "examples",
                    "testcases",
                    "config",
                    "scripts/e2e",
                    "scripts/perf",
                ),
            )
        ),
        "shell": GateGroup(commands=(("make", "shell-check", f"PYTHON={python}"),)),
    }


def _run_group(
    name: str,
    group: GateGroup,
    *,
    cache_root: Path,
    output_lock: threading.Lock,
) -> int:
    environment = {
        **os.environ,
        **dict(group.environment or {}),
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
    with output_lock:
        print(f"static-gates: starting {name}", file=sys.stderr, flush=True)
    for command in group.commands:
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
                print(f"[static:{name}] {line}", end="", flush=True)
        status = process.wait()
        if status:
            with output_lock:
                print(
                    f"static-gates: {name} exited with status {status}",
                    file=sys.stderr,
                    flush=True,
                )
            return status
    with output_lock:
        print(f"static-gates: {name} passed", file=sys.stderr, flush=True)
    return 0


def run_static_gates(python: str) -> None:
    groups = gate_groups(python)
    output_lock = threading.Lock()
    with tempfile.TemporaryDirectory(prefix="gpu-fault-static-gates-") as directory:
        cache_root = Path(directory)
        with ThreadPoolExecutor(max_workers=gate_parallelism()) as executor:
            futures = {
                executor.submit(
                    _run_group,
                    name,
                    group,
                    cache_root=cache_root,
                    output_lock=output_lock,
                ): name
                for name, group in groups.items()
            }
            failures: dict[str, int] = {}
            for future in as_completed(futures):
                status = future.result()
                if status:
                    failures[futures[future]] = status
    if failures:
        raise StaticGateError(
            "parallel static gate failed: "
            + ", ".join(f"{name}={status}" for name, status in sorted(failures.items()))
        )


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    options = parser.parse_args(arguments)
    try:
        run_static_gates(options.python)
    except (OSError, StaticGateError, subprocess.SubprocessError) as exc:
        print(f"static-gates: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
