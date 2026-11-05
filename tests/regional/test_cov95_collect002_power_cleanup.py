"""COLLECT-002 power cleanup in execute mode: confirmation, window and the record."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import collect002_power_cleanup as cleanup
from scripts.e2e.regional import collect002_power_journal as journal
from scripts.e2e.regional.regional_commands import RegionalFixtureError


class ProbeFixture:
    """The host-probe fixture the cleanup creates on the node."""

    def __init__(
        self,
        regional: Any,
        *,
        create_error: Exception | None = None,
        cleanup_error: Exception | None = None,
        verified: bool = True,
        **identity: Any,
    ) -> None:
        self.regional = regional
        self.identity = identity
        self.create_error = create_error
        self.cleanup_error = cleanup_error
        self.verified = verified
        self.calls: list[tuple[str, ...]] = []

    def create(self) -> None:
        self.calls.append(("create",))
        if self.create_error is not None:
            raise self.create_error

    def snapshot(self) -> dict[str, Any]:
        self.calls.append(("snapshot",))
        return {"boot_id": "boot-7", "gpu_power": [{"index": 0, "limit_w": 700}]}

    def execute(self, *arguments: str, timeout: int) -> dict[str, Any]:
        self.calls.append(arguments)
        return {"cleanup_verified": self.verified, "run_id": arguments[2]}

    def cleanup(self) -> list[str]:
        self.calls.append(("cleanup",))
        if self.cleanup_error is not None:
            raise self.cleanup_error
        return []


def execute_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    confirm: str = cleanup.CONFIRMATION,
    window_end: str | None = None,
    **fixture_options: Any,
) -> tuple[list[ProbeFixture], Path, str]:
    case_dir = tmp_path / "cases" / journal.CASE_ID
    run_id = journal.claim_power_run(case_dir, 2)
    fixtures: list[ProbeFixture] = []

    def make_fixture(regional: Any, **identity: Any) -> ProbeFixture:
        fixture = ProbeFixture(regional, **identity, **fixture_options)
        fixtures.append(fixture)
        return fixture

    monkeypatch.delenv("GPU_FAULT_ACCEPTANCE_WINDOW_END", raising=False)
    monkeypatch.setattr(cleanup, "install_site_profile", lambda *a, **k: None)
    monkeypatch.setattr(cleanup, "install_abort_signals", lambda *a, **k: None)
    monkeypatch.setattr(
        cleanup, "settings_from_arguments", lambda arguments: SimpleNamespace()
    )
    monkeypatch.setattr(cleanup, "RegionalLiveFixture", lambda settings: settings)
    monkeypatch.setattr(cleanup, "CollectorAcceptanceFixture", make_fixture)
    end = (
        (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        if window_end is None
        else window_end
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "collect002_power_cleanup.py",
            "--run-dir",
            str(tmp_path),
            "--attempt",
            "2",
            "--node",
            "node-a",
            "--host-probe-image",
            "registry.example/probe:1",
            "--execute",
            "--confirm",
            confirm,
            "--maintenance-window-end",
            end,
        ],
    )
    return fixtures, case_dir, run_id


@pytest.mark.parametrize(
    ("options", "fragment"),
    [
        ({"confirm": "COLLECT002"}, "confirmation must be exactly"),
        ({"window_end": "  "}, "--maintenance-window-end or GPU_FAULT_ACCEPTANCE"),
        ({"window_end": "2026-11-05T12:00:00"}, "must include a timezone"),
    ],
)
def test_execute_refuses_before_creating_the_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    options: dict[str, Any],
    fragment: str,
) -> None:
    fixtures, case_dir, _ = execute_run(tmp_path, monkeypatch, **options)

    with pytest.raises(RegionalFixtureError, match=fragment):
        cleanup.main()

    assert fixtures == []
    assert not (case_dir / "power-cleanup-a2.json").exists(), "nothing was recorded"


def test_execute_restores_the_journaled_run_and_records_the_verified_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixtures, case_dir, run_id = execute_run(tmp_path, monkeypatch)

    assert cleanup.main() == 0

    [fixture] = fixtures
    assert fixture.identity == {
        "node": "node-a",
        "image": "registry.example/probe:1",
        "case_id": journal.CASE_ID,
        "run_id": "collect002-power-cleanup-a2",
        "case_dir": case_dir,
    }
    assert fixture.calls == [
        ("create",),
        ("snapshot",),
        ("restore-gpu-power-limit", "--run-id", run_id),
        ("cleanup",),
    ]
    record = json.loads((case_dir / "power-cleanup-a2.json").read_text())
    assert record["cleanup_verified"] is True
    assert record["probe_residuals"] == []
    assert (record["boot_id"], record["power_run_id"]) == ("boot-7", run_id)
    assert json.loads(capsys.readouterr().out)["cleanup_verified"] is True


def test_execute_records_a_failed_probe_and_its_cleanup_error_and_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixtures, case_dir, _ = execute_run(
        tmp_path,
        monkeypatch,
        create_error=RegionalFixtureError("probe Pod never started"),
        cleanup_error=RuntimeError("probe Pod still terminating"),
    )

    assert cleanup.main() == 1

    record = json.loads((case_dir / "power-cleanup-a2.json").read_text())
    assert record["error"] == "RegionalFixtureError: probe Pod never started"
    assert record["probe_cleanup_error"] == "RuntimeError: probe Pod still terminating"
    assert record["cleanup_verified"] is False
    assert fixtures[0].calls == [("create",), ("cleanup",)]
    assert json.loads(capsys.readouterr().out)["cleanup_verified"] is False
