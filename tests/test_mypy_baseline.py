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


def test_directory_ceilings_give_the_ratchet_a_direction() -> None:
    """S19: the per-file ratchet stops growth; the ceilings say where to shrink.

    A directory ceiling fails only when the directory's total exceeds it. Going
    under is progress, not slack -- the ceiling ratchets down on the next
    ``--write-baseline`` and never back up.
    """

    current = Counter(
        {
            "src/gpu_fault/store/postgres/a.py:type-arg": 3,
            "src/gpu_fault/store/sqlite/b.py:no-any-return": 2,
            "src/gpu_fault/storefront.py:type-arg": 9,
            "src/gpu_fault/app/c.py:no-untyped-def": 4,
        }
    )
    targets = {
        "src/gpu_fault/store": {"ceiling": 5, "priority": 1, "why": "dict shapes"},
        "src/gpu_fault/app": {"ceiling": 3, "priority": 2, "why": "route handlers"},
    }

    assert MODULE.directory_totals(targets, current) == {
        "src/gpu_fault/store": 5,
        "src/gpu_fault/app": 4,
    }
    assert MODULE.convergence_failures(targets, current) == [
        "src/gpu_fault/app has 4 strict mypy errors, above its convergence "
        "ceiling of 3; fix errors in that directory, do not raise the ceiling"
    ]
    tightened = MODULE.tightened_ceilings(
        targets, Counter({"src/gpu_fault/store/postgres/a.py:type-arg": 2})
    )
    assert tightened["src/gpu_fault/store"]["ceiling"] == 2
    assert tightened["src/gpu_fault/app"]["ceiling"] == 0
    assert tightened["src/gpu_fault/store"]["why"] == "dict shapes"


def test_convergence_targets_file_matches_the_tree() -> None:
    """The committed ceilings must be real: nothing above them, nothing stale."""

    targets = MODULE.load_convergence_targets()
    assert targets, "at least one directory must have a convergence ceiling"
    for directory, target in targets.items():
        assert (ROOT / directory).is_dir(), directory
        assert isinstance(target["ceiling"], int) and target["ceiling"] >= 0
        assert isinstance(target["priority"], int) and target["priority"] >= 1
        assert target["why"].strip(), f"{directory} needs a why"
    baseline = Counter(MODULE.load_baseline())
    assert MODULE.convergence_failures(targets, baseline) == []
    # Ceilings are ratcheted to the baseline whenever it is rewritten, so a
    # ceiling above the recorded total is stale.
    totals = MODULE.directory_totals(targets, baseline)
    assert {name: target["ceiling"] for name, target in targets.items()} == totals


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
