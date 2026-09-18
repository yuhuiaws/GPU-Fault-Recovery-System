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

import hashlib
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_collector_destructive as runner
from scripts.e2e.regional.host_probe_fixture import HostProbeTransportError
from scripts.e2e.regional.probes import collector_node_probe as probe
from tests.regional.test_collector_env_safety import ENV_TEXT, OWNER_NONCE, EnvHost
from tests.regional.test_collector_env_safety import env_host as env_host_fixture

env_host = env_host_fixture


@pytest.fixture
def node(env_host: EnvHost) -> dict[str, Any]:
    env_host.run_id = "c004-1"
    return {
        "env": probe.COLLECTOR_ENV,
        "units": probe.SYSTEMD_UNIT_DIR,
        "commands": env_host.calls,
        "emitted": env_host.emitted,
        "boot_id": probe.BOOT_ID_FILE,
        "host": env_host,
    }


def _arguments(run_id: str = "c004-1", **changes: Any) -> Namespace:
    return Namespace(
        **{
            "run_id": run_id,
            "owner_nonce": OWNER_NONCE,
            "expected_env_sha256": hashlib.sha256(ENV_TEXT.encode()).hexdigest(),
            "expected_boot_id": "boot-fixture-a",
            "cluster_id": "cluster-fixture",
            "node_id": "node-fixture",
            "value": 9,
            "restore_seconds": 600,
            **changes,
        }
    )


def _override(run_id: str = "c004-1") -> None:
    probe.override_expected_gpu_count(_arguments(run_id))


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
    assert "OnBootSec=1600.000s" in timer
    assert "OnActiveSec" not in timer, "re-enabling cannot renew the fixed deadline"
    assert "WantedBy=multi-user.target" in service, (
        "the restore does not run at the next boot"
    )
    assert f"Before={probe.HOST_COLLECTOR_UNIT}" in service, (
        "at boot the collector would read the override before the restore lands"
    )
    assert (
        f"ExecStart=/bin/systemctl restart {probe.HOST_COLLECTOR_UNIT}" not in service
    ), "an unconditional restart from a Before= unit cancels the collector's boot start"
    assert "--automatic" in service, (
        "the oneshot must execute the boot/deadline-bound recovery callback"
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
    probe.restore_collector_env(_arguments())

    result = node["emitted"][-1]
    assert result["restored"] is True, result
    assert result["cleanup_verified"] is True
    assert node["env"].read_text(encoding="utf-8") == ENV_TEXT
    backup, unit = probe.collector_restore_paths("c004-1")
    assert not backup.exists()
    assert (
        json.loads(probe.collector_override_record(backup).read_text())["state"]
        == "CLEANED"
    )
    assert ["systemctl", "disable", "--now", f"{unit}.service"] in node["commands"], (
        "the boot-time restore must be stopped as well as disabled"
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

    with pytest.raises(probe.ProbeError, match="backup is missing"):
        probe.restore_collector_env(_arguments())
    assert probe.parse_env()["GPU_FAULT_EXPECTED_GPU_COUNT"] == "9"
    # The record stays so a later attempt can still judge the file.
    assert probe.collector_override_record(backup).exists(), (
        "the override record survives a restore that found no backup"
    )


def test_restore_for_an_unknown_run_is_not_a_restore(node: dict[str, Any]) -> None:
    probe.restore_collector_env(_arguments("never-armed"))

    result = node["emitted"][-1]
    assert result["restored"] is False
    assert result["state"] == "NOT_STARTED" and result["no_mutation"]


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


def test_fabric_manager_cursor_candidates_include_the_deployed_state_path() -> None:
    assert (
        Path("/var/lib/gpu-fault/fabric-manager-collector-state.json")
        in probe.FM_STATE_CANDIDATES
    )


@pytest.mark.parametrize("malformed", [False, True])
def test_fabric_manager_cursor_prefers_deployed_state_over_legacy_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, malformed: bool
) -> None:
    candidates = tuple(tmp_path / path.name for path in probe.FM_STATE_CANDIDATES)
    deployed = tmp_path / "fabric-manager-collector-state.json"
    for path in candidates:
        path.write_text(json.dumps({"files": {"legacy": {"offset": 1}}}))
    deployed.write_text(
        "{invalid"
        if malformed
        else json.dumps({"files": {"deployed": {"offset": 4096}}})
    )
    monkeypatch.setattr(probe, "FM_STATE_CANDIDATES", candidates)

    cursor = probe.fabric_manager_cursor()

    assert cursor is not None and cursor["path"] == str(deployed), cursor
    if malformed:
        assert "error" in cursor, cursor
    else:
        assert cursor["files"] == {"deployed": {"offset": 4096}}, cursor


def test_override_keeps_private_permissions_and_arms_before_replacement(
    node: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[str] = []

    def run(command: list[str], **kwargs: Any) -> None:
        if command[:3] == ["systemctl", "enable", "--now"]:
            assert node["env"].read_text() == ENV_TEXT
            observed.append("armed")
        if command[:2] == ["systemctl", "restart"]:
            assert observed == ["armed"]
            assert "GPU_FAULT_EXPECTED_GPU_COUNT=9" in node["env"].read_text()

    node["host"].before_call = run
    _override()
    assert observed == ["armed"]
    assert node["env"].stat().st_mode & 0o777 == 0o600


def test_early_automatic_callback_defers_on_the_original_boot(
    node: dict[str, Any],
) -> None:
    _override()
    calls = len(node["commands"])
    probe.restore_collector_env(_arguments(automatic=True))
    assert node["emitted"][-1]["deferred"] is True
    assert all(item[1] == "show" for item in node["commands"][calls:])
    assert "GPU_FAULT_EXPECTED_GPU_COUNT=9" in node["env"].read_text()
    assert probe.collector_restore_paths("c004-1")[0].exists(), (
        "same-boot timer deferral discarded the original environment backup"
    )


@pytest.mark.parametrize("trigger", ["new_boot", "expired"])
def test_automatic_restore_runs_on_new_boot_or_elapsed_ttl(
    node: dict[str, Any], monkeypatch: pytest.MonkeyPatch, trigger: str
) -> None:
    _override()
    if trigger == "new_boot":
        node["boot_id"].write_text("boot-b")
    else:
        node["host"].now += 601
    probe.restore_collector_env(_arguments(automatic=True))
    assert node["emitted"][-1]["restored"] is True
    assert node["env"].read_text() == ENV_TEXT
    assert node["env"].stat().st_mode & 0o777 == 0o600


def test_failed_timer_arm_never_exposes_the_override(
    node: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> None:
        commands.append(command)
        assert node["env"].read_text() == ENV_TEXT
        if command[:3] == ["systemctl", "enable", "--now"]:
            raise probe.ProbeError("cannot arm")

    node["host"].before_call = run
    with pytest.raises(probe.ProbeError, match="cannot arm"):
        _override()
    assert not any(item[:2] == ["systemctl", "restart"] for item in commands), (
        f"collector restarted after timer arming failed: {commands!r}"
    )
    assert probe.collector_restore_paths("c004-1")[0].exists()
    assert node["host"].record()["mutation_started"] is False
    node["host"].before_call = None
    probe.restore_collector_env(_arguments())
    assert not probe.collector_restore_paths("c004-1")[0].exists()


def test_failed_service_restore_keeps_timer_and_backup_for_retry(
    node: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _override()
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> None:
        commands.append(command)
        if command[:2] == ["systemctl", "restart"]:
            raise probe.ProbeError("restart refused")

    node["host"].before_call = run
    with pytest.raises(probe.ProbeError, match="restart refused"):
        probe.restore_collector_env(_arguments())
    backup, unit = probe.collector_restore_paths("c004-1")
    assert backup.exists() and probe.collector_override_record(backup).exists()
    assert (node["units"] / f"{unit}.timer").exists(), (
        f"failed service restoration removed watchdog timer {unit}"
    )
    assert not any(item[:2] == ["systemctl", "disable"] for item in commands), (
        f"failed restoration disabled its recovery watchdog: {commands!r}"
    )
    node["host"].before_call = None
    probe.restore_collector_env(_arguments())
    assert node["emitted"][-1]["restored"] is True
    assert not backup.exists(), f"successful restore retry left backup {backup}"


def test_corrupt_backup_cannot_replace_the_live_env(node: dict[str, Any]) -> None:
    _override()
    backup, _ = probe.collector_restore_paths("c004-1")
    backup.write_bytes(b"not the baseline")
    current = node["env"].read_bytes()
    with pytest.raises(probe.ProbeError, match="backup digest"):
        probe.restore_collector_env(_arguments())
    assert node["env"].read_bytes() == current
    assert backup.exists(), (
        "digest failure discarded the backup needed for investigation"
    )


def test_reused_run_cannot_overwrite_the_original_backup(node: dict[str, Any]) -> None:
    _override()
    backup, _ = probe.collector_restore_paths("c004-1")
    with pytest.raises(probe.ProbeError, match="owns this node"):
        probe.override_expected_gpu_count(_arguments(value=10))
    assert backup.read_text() == ENV_TEXT


class _RebootingCollector:
    """A probe whose node is down: every exec and every recreate fails."""

    def __init__(self, recreate_recovers: bool) -> None:
        self.recreate_recovers = recreate_recovers
        self.calls: list[str] = []

    def execute(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
        self.calls.append("execute")
        if self.recreate_recovers and "recreate" in self.calls:
            return {"restored": True, "run_id": arguments[2]}
        raise HostProbeTransportError("kubectl failed (1): exec: pod is Failed")

    def recreate(self) -> None:
        self.calls.append("recreate")
        if not self.recreate_recovers:
            raise HostProbeTransportError("kubectl failed (1): wait Ready timed out")


def test_pre_reboot_restore_defers_instead_of_failing_the_case_when_the_node_is_down():
    """Attempts 3 and 4 died here: the executor rebooted the node seconds after
    planning RESTART_NODE, the restore's exec lost its Pod, and the recreate
    waited 180 s for a Pod on a node that was down. The runner now records a
    deferred restore and retries once the node is back."""

    collector = _RebootingCollector(recreate_recovers=False)
    outcome = runner.restore_collector_env(
        collector, "c004-5", owner_nonce="a" * 32, reboot_transition=lambda: True
    )
    assert outcome["restored"] is False and outcome["deferred"] is True
    assert "authorized reboot wait" in outcome["reason"]
    assert collector.calls == ["execute"]

    recovered = _RebootingCollector(recreate_recovers=True)
    waiting = runner.restore_collector_env(
        recovered, "c004-5", owner_nonce="a" * 32, reboot_transition=lambda: True
    )
    assert waiting["deferred"] is True and waiting["restored"] is False
    assert recovered.calls == ["execute"]
    recovered.recreate()
    assert runner.restore_collector_env(recovered, "c004-5", owner_nonce="a" * 32) == {
        "restored": True,
        "run_id": "c004-5",
    }
    assert recovered.calls == ["execute", "recreate", "execute"]


@pytest.mark.parametrize("active", [True, False])
def test_automatic_restore_does_not_wait_on_the_collectors_boot_job(
    active: bool, node: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _override()
    node["boot_id"].write_text("boot-b")
    if not active:
        node["host"].active.discard(probe.HOST_COLLECTOR_UNIT)
    node["commands"].clear()
    probe.restore_collector_env(_arguments(automatic=True))
    commands = node["commands"]

    assert node["env"].read_text() == ENV_TEXT
    assert node["emitted"][-1]["restored"] is True
    assert node["emitted"][-1]["cleanup_deferred"] is True
    assert ["systemctl", "restart", probe.HOST_COLLECTOR_UNIT] not in commands
    assert (
        ["systemctl", "--no-block", "restart", probe.HOST_COLLECTOR_UNIT] in commands
    ) is active
    backup, unit = probe.collector_restore_paths("c004-1")
    assert backup.exists() and (node["units"] / f"{unit}.service").exists(), (
        "boot restore must retain recovery state until health is independently verified"
    )


def test_automatic_restore_refuses_unknown_service_state(
    node: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _override()
    node["boot_id"].write_text("boot-b")
    node["host"].overrides[probe.HOST_COLLECTOR_UNIT] = {"LoadState": "not-found"}
    with pytest.raises(probe.ProbeError, match="host collector"):
        probe.restore_collector_env(_arguments(automatic=True))
    assert probe.collector_restore_paths("c004-1")[0].exists(), (
        "unknown service state must preserve the original recovery backup"
    )
