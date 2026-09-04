from __future__ import annotations

from collections import Counter
from pathlib import Path

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/check-mypy-baseline.py"
MODULE = lazy_script_module(MODULE_PATH)


def test_mypy_baseline_rejects_growth_and_slack() -> None:
    baseline = {"src/a.py:attr-defined": 2}

    assert MODULE.failures(baseline, Counter({"src/a.py:attr-defined": 3})) == [
        "src/a.py:attr-defined grew from 2 to 3"
    ]
    assert MODULE.failures(baseline, Counter({"src/a.py:attr-defined": 1})) == [
        "src/a.py:attr-defined baseline has slack: "
        "current 1, recorded 2; run --write-baseline"
    ]


def test_mypy_targets_cover_the_package_and_the_release_orchestrator() -> None:
    """`deploy` carries the rollback path, so it is typed like `src`."""

    assert MODULE.TARGETS == ("src", "deploy")


def test_store_consumers_depend_on_protocol() -> None:
    for relative in (
        "src/gpu_fault/execution/dispatcher.py",
        "src/gpu_fault/execution/executor.py",
        "src/gpu_fault/service.py",
        "src/gpu_fault/notification_service.py",
        "src/gpu_fault/fleet.py",
        "src/gpu_fault/adapters/simulated.py",
        "src/gpu_fault/orchestration/coordinator.py",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "ControlPlaneStore" in text
