"""join-cluster / remove-cluster pass their target to every scoped engine call.

b250 scoped the final verify. The join's candidate ``preflight`` and its
``sync-state`` (``RELEASE_STATE_UPDATED``, before activation: a refusal there
rolled the join back), and the removal's ``sync-state``, still judged every
cluster. The deploy host now names the verb's target there too (the engine's
``join-cluster`` mode scopes itself to its ``--cluster-id``), and reads the
preflight's report back like the verify's, so the command log keeps it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from gpu_fault.admin import cluster_join as join
from gpu_fault.admin import cluster_removal as removal
from gpu_fault.admin import membership_verify as SCOPED
from tests.admin._cov95_join_support import JoinScenario
from tests.admin._cov95_removal_support import RemovalTransport

SCOPED_MODES = {"preflight", "sync-state"}


def recorded(
    monkeypatch: Any, driver: Any, modules: tuple[Any, ...] = (join, removal)
) -> list[tuple[str, list[str], Any]]:
    """Record every scoped mode's flags past ``--config <file>``, and whether
    its stdout is read; the scenario's own driver still answers."""

    seen: list[tuple[str, list[str], Any]] = []

    def recording(arguments: list[str], **options: Any) -> Any:
        if arguments[1] in SCOPED_MODES:
            seen.append((arguments[1], list(arguments[4:]), options.get("stdout")))
        return driver(arguments, **options)

    for module in modules:
        monkeypatch.setattr(module, "run_driver", recording)
    return seen


def test_a_join_scopes_its_candidate_preflight_and_its_sync_state(
    tmp_path: Path, monkeypatch: Any
) -> None:
    scenario = JoinScenario(tmp_path, monkeypatch)
    seen = recorded(monkeypatch, scenario.driver)

    result = scenario.join()

    assert result["phase"] == "COMPLETED", result
    assert seen == [
        ("preflight", ["--joining-cluster-id", "hp-gpu-b"], subprocess.PIPE),
        ("sync-state", ["--joining-cluster-id", "hp-gpu-b"], None),
    ], "both are scoped to the joining cluster; the preflight's report is read"


def test_a_batch_join_scopes_its_preflight_to_the_batch(
    tmp_path: Path, monkeypatch: Any
) -> None:
    scenario = JoinScenario(tmp_path, monkeypatch)
    seen = recorded(monkeypatch, scenario.driver)

    result = scenario.batch()

    assert result["joined"], result
    assert seen[0] == (
        "preflight",
        ["--joining-cluster-id", "hp-gpu-b", "--joining-cluster-id", "hp-gpu-c"],
        subprocess.PIPE,
    ), seen
    assert sorted(flags[1] for mode, flags, _ in seen if mode == "sync-state") == [
        "hp-gpu-b",
        "hp-gpu-c",
    ], "each commit's sync-state names its own joining cluster"


def test_a_removal_scopes_its_sync_state_to_the_removed_cluster(
    tmp_path: Path, monkeypatch: Any
) -> None:
    transport = RemovalTransport(tmp_path, monkeypatch)
    seen = recorded(monkeypatch, transport.driver, (removal,))

    result = transport.remove()

    assert result["phase"] == "COMPLETED", result
    assert seen == [("sync-state", ["--removed-cluster-id", "gpu-a"], None)]


def test_the_scoped_command_lines() -> None:
    rollout, config = Path("rollout.sh"), Path("cfg.json")

    assert SCOPED.rollout_arguments(rollout, "preflight", config, ["b", "a"]) == [
        "rollout.sh",
        "preflight",
        "--config",
        "cfg.json",
        "--joining-cluster-id",
        "a",
        "--joining-cluster-id",
        "b",
    ]
    assert SCOPED.sync_arguments(SCOPED.JOIN_VERB, "hp-b") == [
        "--joining-cluster-id",
        "hp-b",
    ]
    assert SCOPED.sync_arguments(SCOPED.REMOVE_VERB, "gpu-a") == [
        "--removed-cluster-id",
        "gpu-a",
    ]
    assert SCOPED.sync_arguments(SCOPED.JOIN_VERB, None) == [], (
        "a join's rollback re-syncs the restored site unscoped"
    )
    assert SCOPED.driver_capture("preflight") == {"stdout": subprocess.PIPE}
    assert SCOPED.driver_capture("join-cluster") == {}
    completed = subprocess.CompletedProcess([], 0, '{"mode": "preflight"}', "")
    assert SCOPED.captured_report("preflight", completed) == {"mode": "preflight"}
    assert SCOPED.captured_report("activate-cluster", completed) is None
