"""COLLECT-002 power run ids are single-use on the node; the runner says so first.

Live HyperPod COLLECT-002: attempt 1 throttled, its restore failed with exit 78
(the node rebooted under the operation), and two later executions of the same
attempt were refused by the probe at ``throttle_gpu`` ("power run ID has
already been used; cleanup only") with the output withheld. The runner's own
journal now refuses the spent attempt by name before any probe runs, keeps a
pre-mutation retry legitimate, and the cleanup tool closes the old operation.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import collect002_power_cleanup as cleanup
from scripts.e2e.regional import collect002_power_journal as journal
from scripts.e2e.regional.regional_commands import RegionalFixtureError


def test_a_fresh_attempt_claims_the_deterministic_run_id(tmp_path: Path) -> None:
    run_id = journal.claim_power_run(tmp_path, 1)
    assert run_id == journal.power_run_id(tmp_path, 1)
    assert run_id.startswith("collect002-") and run_id.endswith("-a1"), run_id
    saved = json.loads(journal.power_journal_path(tmp_path, 1).read_text())
    assert saved == {"case_id": journal.CASE_ID, "attempt": 1, "run_id": run_id}


def test_a_retry_before_any_mutation_keeps_the_same_run_id(tmp_path: Path) -> None:
    first = journal.claim_power_run(tmp_path, 1)
    assert journal.claim_power_run(tmp_path, 1) == first, (
        "a throttle refused before mutation spent nothing on the node"
    )
    assert journal.claim_power_run(tmp_path, 2) != first


def test_a_spent_attempt_is_refused_by_name_before_the_probe_runs(
    tmp_path: Path,
) -> None:
    run_id = journal.claim_power_run(tmp_path, 1)
    journal.record_power_outcome(tmp_path, 1, mutation_started=True)
    with pytest.raises(RegionalFixtureError) as caught:
        journal.claim_power_run(tmp_path, 1)
    message = str(caught.value)
    assert (
        f"attempt 1 of this run-dir already drove power operation {run_id}" in message
    )
    assert "rerun with a new --attempt" in message
    assert journal.CLEANUP_TOOL not in message, "a clean restore needs no cleanup tool"
    assert json.loads(journal.power_journal_path(tmp_path, 1).read_text())[
        "run_id"
    ] == (run_id), "the refusal rewrites nothing"


def test_a_failed_restore_points_at_the_cleanup_tool(tmp_path: Path) -> None:
    journal.claim_power_run(tmp_path, 1)
    journal.record_power_outcome(tmp_path, 1, mutation_started=True)
    journal.record_power_outcome(
        tmp_path,
        1,
        cleanup_error="HostProbeError: host probe failed (exit 78); output withheld",
    )
    saved = json.loads(journal.power_journal_path(tmp_path, 1).read_text())
    assert saved["mutation_started"] is True and "exit 78" in saved["cleanup_error"]
    with pytest.raises(RegionalFixtureError) as caught:
        journal.claim_power_run(tmp_path, 1)
    assert "its restore failed (HostProbeError" in str(caught.value)
    assert f"after {journal.CLEANUP_TOOL} has closed the old operation" in str(
        caught.value
    )


def test_a_cleanup_error_alone_marks_the_attempt_spent(tmp_path: Path) -> None:
    """A throttle that failed after the probe mutated leaves the runner unsure;
    the restore it then attempted failed, and that is enough to refuse."""

    journal.claim_power_run(tmp_path, 1)
    journal.record_power_outcome(tmp_path, 1, cleanup_error="RuntimeError: boom")
    with pytest.raises(RegionalFixtureError, match="already drove power operation"):
        journal.claim_power_run(tmp_path, 1)


def test_a_record_the_probe_wrote_before_failing_marks_the_attempt_spent(
    tmp_path: Path,
) -> None:
    """The probe writes its record before it mutates; a throttle that failed
    after that point left the id used on the node. The restore's receipt then
    carries the record's baseline, and the runner records the id as spent."""

    journal.claim_power_run(tmp_path, 1)
    journal.record_power_outcome(tmp_path, 1, power_record_seen=False)
    assert journal.claim_power_run(tmp_path, 1), "no record on the node: retry allowed"
    journal.record_power_outcome(tmp_path, 1, power_record_seen=True)
    with pytest.raises(RegionalFixtureError, match="rerun with a new --attempt"):
        journal.claim_power_run(tmp_path, 1)


def _fixture(calls: list[tuple[str, ...]], *, boot_id: Any = "boot-2") -> Any:
    snapshot = {"boot_id": boot_id, "gpu_power": [{"index": 0, "power_limit_w": 700}]}

    def execute(*arguments: str, timeout: int = 180) -> dict[str, Any]:
        calls.append(arguments)
        return {"run_id": arguments[2], "cleanup_verified": True, "rebooted": True}

    return SimpleNamespace(snapshot=lambda: snapshot, execute=execute)


def test_cleanup_after_reboot_hands_the_probe_the_current_boot_id() -> None:
    calls: list[tuple[str, ...]] = []
    result = cleanup.power_cleanup(
        _fixture(calls), "collect002-x-a1", after_reboot=True
    )
    assert calls == [
        (
            "restore-gpu-power-limit",
            "--run-id",
            "collect002-x-a1",
            "--after-reboot",
            "boot-2",
        )
    ]
    assert result["cleanup_verified"] is True and result["boot_id"] == "boot-2"
    assert result["after_reboot"] is True and result["case_id"] == journal.CASE_ID


def test_plain_cleanup_never_claims_a_reboot() -> None:
    calls: list[tuple[str, ...]] = []
    result = cleanup.power_cleanup(
        _fixture(calls), "collect002-x-a1", after_reboot=False
    )
    assert calls == [("restore-gpu-power-limit", "--run-id", "collect002-x-a1")]
    assert result["after_reboot"] is False


def test_cleanup_after_reboot_requires_a_boot_id_in_the_snapshot() -> None:
    with pytest.raises(RegionalFixtureError, match="no boot id"):
        cleanup.power_cleanup(
            _fixture([], boot_id=None), "collect002-x-a1", after_reboot=True
        )


def test_cleanup_resolves_the_run_id_from_the_attempt_journal(tmp_path: Path) -> None:
    run_id = journal.claim_power_run(tmp_path, 3)
    assert cleanup.resolve_run_id(tmp_path, 3, "") == run_id
    assert cleanup.resolve_run_id(tmp_path, 3, " explicit-a3 ") == "explicit-a3"
    with pytest.raises(RegionalFixtureError, match="no power operation is journaled"):
        cleanup.resolve_run_id(tmp_path, 4, "")


def test_cleanup_plan_mode_prints_the_intent_without_touching_the_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_id = journal.claim_power_run(tmp_path / "cases" / journal.CASE_ID, 1)
    monkeypatch.setattr(cleanup, "install_site_profile", lambda *a, **k: None)
    monkeypatch.setattr(cleanup, "install_abort_signals", lambda *a, **k: None)
    monkeypatch.setattr(cleanup, "settings_from_arguments", lambda a: SimpleNamespace())
    monkeypatch.setattr(
        cleanup,
        "CollectorAcceptanceFixture",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("plan mode created a probe")
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "collect002_power_cleanup.py",
            "--run-dir",
            str(tmp_path),
            "--attempt",
            "1",
            "--node",
            "node-a",
            "--host-probe-image",
            "registry.example/probe:1",
            "--after-reboot",
            "--plan",
        ],
    )
    assert cleanup.main() == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["power_run_id"] == run_id and plan["after_reboot"] is True
    assert plan["confirmation"] == cleanup.CONFIRMATION
    assert "receipt are never removed" in plan["hard_stop"]
