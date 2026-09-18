from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.probes import destr017_node_probe as reboot
from scripts.e2e.regional.probes import destructive_node_probe as injection


@pytest.mark.parametrize("deadline", ["2000-01-01T00:00:00Z", "invalid", "2099-01-01"])
def test_delayed_xid_refuses_before_opening_kmsg(
    deadline: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(
        injection, "os", SimpleNamespace(open=lambda path, *_args: opened.append(path))
    )
    arguments = argparse.Namespace(maintenance_window_end=deadline)

    with pytest.raises(injection.ProbeError, match="maintenance window"):
        injection.write_xid79(arguments)
    assert opened == []


def reboot_state(deadline: str) -> dict[str, Any]:
    return {
        "run_id": "review",
        "maintenance_window_end": deadline,
        "boot_id_before_reboot": "boot-a",
        "reboot_cancelled_at": None,
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"maintenance_window_end": "2000-01-01T00:00:00Z"},
        {"maintenance_window_end": ""},
        {"maintenance_window_end": "2099-01-01"},
        {"reboot_cancelled_at": "2026-09-12T00:00:00Z"},
        {"fire_requested_at": "2026-09-12T00:00:00Z"},
        {"boot_id_before_reboot": "another-boot"},
        {"run_id": "another-run"},
    ],
)
def test_timer_fire_rechecks_deadline_boot_and_single_use(
    overrides: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    reboot.write_state(path, {**reboot_state("2099-01-01T00:00:00Z"), **overrides})
    commands: list[list[str]] = []
    monkeypatch.setattr(reboot, "state_path", lambda _run: path)
    monkeypatch.setattr(reboot, "boot_id", lambda: "boot-a")
    monkeypatch.setattr(reboot, "run", lambda argv: commands.append(argv))

    with pytest.raises(reboot.ProbeError):
        reboot.fire_reboot(argparse.Namespace(run_id="review"))
    assert commands == []


def test_fire_intent_is_persisted_before_the_only_reboot_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    reboot.write_state(path, reboot_state("2099-01-01T00:00:00Z"))
    commands: list[list[str]] = []
    monkeypatch.setattr(reboot, "state_path", lambda _run: path)
    monkeypatch.setattr(reboot, "boot_id", lambda: "boot-a")

    def run(argv: list[str]) -> None:
        assert reboot.read_state(path).get("fire_requested_at"), (
            "durable intent must precede the reboot request"
        )
        commands.append(argv)

    monkeypatch.setattr(reboot, "run", run)
    checked: list[str] = []
    monkeypatch.setattr(
        reboot, "check_barrier", lambda _state: checked.append("barrier")
    )
    arguments = argparse.Namespace(run_id="review")

    reboot.fire_reboot(arguments)
    with pytest.raises(reboot.ProbeError, match="consumed"):
        reboot.fire_reboot(arguments)
    assert commands == [["/bin/systemctl", "reboot"]]
    assert checked == ["barrier"]


def test_timer_cannot_be_placed_beyond_the_approved_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    end = datetime.now(timezone.utc) + timedelta(seconds=10)
    reboot.write_state(path, reboot_state(end.isoformat()))
    commands: list[Any] = []
    monkeypatch.setattr(reboot, "run", lambda *args, **kwargs: commands.append(args))

    with pytest.raises(reboot.ProbeError, match="outlive"):
        reboot.place_reboot_timer("review", 30, path)
    assert commands == []
