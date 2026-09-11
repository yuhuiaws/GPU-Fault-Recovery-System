"""The collector node probe's env override/restore and the reads the runners poll.

COLLECT-004 restores ``collector.env`` through this probe. ``restore-collector-env``
used to answer ``restored: True`` unconditionally -- with the backup gone the
override stayed on the node and the runner recorded a clean restore. The deadman
timer was ``Persistent=true`` on a monotonic ``OnActiveSec`` timer, which systemd
ignores, so a reboot never re-armed the restore either; the ``OnBootSec=1`` that
replaced it fired at once on a node long past boot. The boot-time restore is now
an enabled oneshot service.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_collector_destructive as runner
from scripts.e2e.regional.host_probe_fixture import HostProbeError
from scripts.e2e.regional.probes import collector_node_probe as probe

ENV_TEXT = "GPU_FAULT_EXPECTED_GPU_COUNT=8\nGPU_FAULT_HOST_INTERVAL_SECONDS=15\n"


@pytest.fixture
def node(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    env = tmp_path / "collector.env"
    env.write_text(ENV_TEXT, encoding="utf-8")
    units = tmp_path / "systemd"
    units.mkdir()
    commands: list[list[str]] = []
    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(probe, "COLLECTOR_ENV", env)
    monkeypatch.setattr(probe, "ACCEPTANCE_STATE", tmp_path / "acceptance")
    monkeypatch.setattr(probe, "SYSTEMD_UNIT_DIR", units)
    monkeypatch.setattr(probe, "run", lambda command, **_: commands.append(command))
    monkeypatch.setattr(probe, "emit", emitted.append)
    return {"env": env, "units": units, "commands": commands, "emitted": emitted}


def _override(run_id: str = "c004-1") -> None:
    probe.override_expected_gpu_count(
        Namespace(run_id=run_id, value=9, restore_seconds=600)
    )


def test_override_arms_a_deadman_timer_and_a_boot_time_restore(
    node: dict[str, Any],
) -> None:
    """The timer restores after the deadline; the enabled oneshot restores at
    the next boot. ``OnBootSec=1`` on a timer enabled long after boot is
    already elapsed and fired the restore a second after the override
    (attempt 1), so the timer carries only the monotonic deadline."""

    _override()

    backup, unit = probe.collector_restore_paths("c004-1")
    timer = (node["units"] / f"{unit}.timer").read_text(encoding="utf-8")
    service = (node["units"] / f"{unit}.service").read_text(encoding="utf-8")
    assert "OnActiveSec=600s" in timer
    assert "OnBootSec" not in timer and "Persistent" not in timer, (
        "an already-elapsed boot offset fires the restore immediately"
    )
    assert "WantedBy=multi-user.target" in service, (
        "the restore does not run at the next boot"
    )
    assert f"Before={probe.HOST_COLLECTOR_UNIT}" in service, (
        "at boot the collector would read the override before the restore lands"
    )
    assert (
        f"ExecStart=/bin/systemctl restart {probe.HOST_COLLECTOR_UNIT}" not in service
    ), "an unconditional restart from a Before= unit cancels the collector's boot start"
    assert f"systemctl is-active --quiet {probe.HOST_COLLECTOR_UNIT} && " in service, (
        "the deadman path must still restart a running collector"
    )
    assert ["systemctl", "enable", f"{unit}.service"] in node["commands"], (
        "the boot-time restore was not enabled"
    )
    assert "GPU_FAULT_EXPECTED_GPU_COUNT=9" in node["env"].read_text(encoding="utf-8")
    assert backup.read_text(encoding="utf-8") == ENV_TEXT
    record = json.loads(
        probe.collector_override_record(backup).read_text(encoding="utf-8")
    )
    assert record["baseline"] == 8 and record["override"] == 9
    assert ["systemctl", "enable", "--now", f"{unit}.timer"] in node["commands"]


def test_restore_puts_the_original_bytes_back_and_says_so(node: dict[str, Any]) -> None:
    _override()
    probe.restore_collector_env(Namespace(run_id="c004-1"))

    result = node["emitted"][-1]
    assert result["restored"] is True, result
    assert result["backup_present"] is True
    assert node["env"].read_text(encoding="utf-8") == ENV_TEXT
    backup, unit = probe.collector_restore_paths("c004-1")
    assert not backup.exists() and not probe.collector_override_record(backup).exists()
    assert ["systemctl", "disable", f"{unit}.service"] in node["commands"], (
        "the boot-time restore stayed enabled after the restore"
    )
    assert not (node["units"] / f"{unit}.timer").exists(), (
        "the restore timer unit is removed once it fired"
    )
    assert ["systemctl", "restart", probe.HOST_COLLECTOR_UNIT] in node["commands"]


def test_restore_without_a_backup_does_not_claim_the_override_is_gone(
    node: dict[str, Any],
) -> None:
    _override()
    backup, _unit = probe.collector_restore_paths("c004-1")
    backup.unlink()  # the one thing the old probe did not check

    probe.restore_collector_env(Namespace(run_id="c004-1"))

    result = node["emitted"][-1]
    assert result["restored"] is False, result
    assert "still carries the override" in result["reason"]
    assert result["collector_env"]["GPU_FAULT_EXPECTED_GPU_COUNT"] == "9"
    # The record stays so a later attempt can still judge the file.
    assert probe.collector_override_record(backup).exists(), (
        "the override record survives a restore that found no backup"
    )


def test_restore_for_an_unknown_run_is_not_a_restore(node: dict[str, Any]) -> None:
    probe.restore_collector_env(Namespace(run_id="never-armed"))

    result = node["emitted"][-1]
    assert result["restored"] is False
    assert "no backup and no override record" in result["reason"]


def test_fabric_manager_cursor_reads_the_persisted_offsets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "fabric-manager-state.json"
    monkeypatch.setattr(probe, "FM_STATE_CANDIDATES", (tmp_path / "missing", state))
    assert probe.fabric_manager_cursor() is None

    state.write_text(
        json.dumps(
            {
                "journal_cursor": "s=abc",
                "files": {
                    "/var/log/fabricmanager.log": {
                        "device": 1,
                        "inode": 77,
                        "offset": 4096,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    cursor = probe.fabric_manager_cursor()
    assert cursor is not None
    assert cursor["files"]["/var/log/fabricmanager.log"]["offset"] == 4096
    assert cursor["journal_cursor"] == "s=abc"

    state.write_text("{not json", encoding="utf-8")
    broken = probe.fabric_manager_cursor()
    assert broken is not None and "error" in broken


def test_probe_exposes_the_light_reads_the_runners_poll() -> None:
    parser = probe.parser()
    handlers = {
        command: parser.parse_args([command]).handler
        for command in ("fm-cursor", "efa-inventory", "snapshot")
    }
    assert handlers == {
        "fm-cursor": probe.fm_cursor,
        "efa-inventory": probe.efa_inventory_command,
        "snapshot": probe.snapshot,
    }


class _RebootingCollector:
    """A probe whose node is down: every exec and every recreate fails."""

    def __init__(self, recreate_recovers: bool) -> None:
        self.recreate_recovers = recreate_recovers
        self.calls: list[str] = []

    def execute(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
        self.calls.append("execute")
        if self.recreate_recovers and "recreate" in self.calls:
            return {"restored": True, "run_id": arguments[2]}
        raise HostProbeError("kubectl failed (1): exec: pod is Failed")

    def recreate(self) -> None:
        self.calls.append("recreate")
        if not self.recreate_recovers:
            raise HostProbeError("kubectl failed (1): wait Ready timed out")


def test_pre_reboot_restore_defers_instead_of_failing_the_case_when_the_node_is_down():
    """Attempts 3 and 4 died here: the executor rebooted the node seconds after
    planning RESTART_NODE, the restore's exec lost its Pod, and the recreate
    waited 180 s for a Pod on a node that was down. The runner now records a
    deferred restore and retries once the node is back."""

    collector = _RebootingCollector(recreate_recovers=False)
    outcome = runner.restore_collector_env(collector, "c004-5")
    assert outcome["restored"] is False and outcome["deferred"] is True
    assert "node reboots" in outcome["reason"]
    assert collector.calls == ["execute", "recreate"]

    recovered = _RebootingCollector(recreate_recovers=True)
    assert runner.restore_collector_env(recovered, "c004-5") == {
        "restored": True,
        "run_id": "c004-5",
    }
    assert recovered.calls == ["execute", "recreate", "execute"]
