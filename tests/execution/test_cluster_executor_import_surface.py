"""The ``gpu_fault.cluster_executor`` package keeps the module's public surface.

``cluster_executor.py`` was split along its layers (wire client, lease
lifecycle, command dispatch, claim loop, process bootstrap). Everything that
imported a name from ``gpu_fault.cluster_executor`` before -- the console
scripts in ``pyproject.toml``, the regional e2e drivers, the tests -- must
keep working unchanged, each layer must import on its own without pulling the
package into a cycle, and the log lines must keep the logger name operators
filter on.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tomllib
from importlib import import_module
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "gpu_fault.cluster_executor"
LAYERS = ("regional_client", "lease", "dispatch", "executor", "bootstrap")
# The names the module published before the split, as importers spell them.
PUBLIC_NAMES = (
    "ABANDONED_WORKER_HOLD_REASON",
    "DEFAULT_LIVENESS_INTERVAL_SECONDS",
    "DEFAULT_LIVENESS_STATE_PATH",
    "DEFAULT_MAX_EXECUTION_SECONDS",
    "EXECUTION_TIMEOUT_STATUS_SOURCE",
    "EXECUTION_TIMEOUT_UNKNOWN_STATUS_SOURCE",
    "LIVENESS_STALE_AFTER_SECONDS",
    "SPARE_RESERVATION_SWEEP_INTERVAL_SECONDS",
    "SPARE_RESERVATION_TTL_SECONDS",
    "ClusterActionExecutor",
    "ClusterExecutorClaimError",
    "ClusterExecutorError",
    "CommandLeaseWatch",
    "RegionalExecutorClient",
    "RegionalFleetRegistry",
    "RegionalHyperPodSubmissionStore",
    "RegionalIncidentOwnershipProvider",
    "SpareReservationSweep",
    "executor_from_environment",
    "main",
    "readiness_probe",
)


@pytest.mark.parametrize("name", PUBLIC_NAMES)
def test_every_pre_split_name_is_importable_from_the_package(name: str) -> None:
    package = import_module(PACKAGE)

    assert hasattr(package, name), f"{PACKAGE} lost {name}"
    assert name in package.__all__, f"{PACKAGE}.__all__ does not list {name}"


@pytest.mark.parametrize("layer", LAYERS)
def test_each_layer_imports_alone_in_a_fresh_interpreter(layer: str) -> None:
    """No layer may need the package facade (or a sibling) imported first."""

    result = subprocess.run(
        [sys.executable, "-c", f"import {PACKAGE}.{layer}"],
        cwd=ROOT,
        # This checkout's package, not whichever one the interpreter has
        # installed: the venv may point at a sibling worktree.
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("layer", LAYERS)
def test_each_layer_logs_under_the_pre_split_logger_name(layer: str) -> None:
    """``%(name)s`` is in the log format, so the split must not rename the
    logger operators filter on."""

    module = import_module(f"{PACKAGE}.{layer}")

    assert isinstance(module.LOGGER, logging.Logger), layer
    assert module.LOGGER.name == PACKAGE, (layer, module.LOGGER.name)


def test_console_scripts_still_resolve_to_the_package() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]
    expected = {
        "gpu-fault-cluster-executor": f"{PACKAGE}:main",
        "gpu-fault-cluster-executor-readiness": f"{PACKAGE}:readiness_probe",
    }

    for script, target in expected.items():
        assert scripts[script] == target, (script, scripts[script])
        module_name, _, attribute = target.partition(":")
        assert callable(getattr(import_module(module_name), attribute)), target


def test_the_layers_are_the_only_modules_in_the_package() -> None:
    package_dir = ROOT / "src" / "gpu_fault" / "cluster_executor"
    modules = sorted(
        path.stem for path in package_dir.glob("*.py") if path.name != "__init__.py"
    )

    assert modules == sorted(LAYERS), modules
