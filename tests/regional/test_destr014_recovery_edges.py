from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.probes import destr014_recovery_probe as probe
from tests.regional.test_destr014_recovery_probe import HostHarness


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> HostHarness:
    return HostHarness(tmp_path, monkeypatch)


@pytest.mark.parametrize("failure", ["metadata", "serialize", "sync", "replace"])
def test_failed_atomic_write_preserves_previous_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    tmp_path.chmod(0o700)
    path = tmp_path / "state.json"
    probe.atomic_state(path, {"phase": "before"})

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("owned test failure")

    with monkeypatch.context() as patch:
        if failure == "metadata":
            patch.setattr(
                probe.os,
                "fstat",
                lambda _fd: SimpleNamespace(
                    st_mode=stat.S_IFIFO | 0o600, st_uid=os.geteuid(), st_nlink=1
                ),
            )
            expected: type[Exception] = probe.RecoveryError
        else:
            module, name = {
                "serialize": (probe.json, "dump"),
                "sync": (probe.os, "fsync"),
                "replace": (probe.os, "replace"),
            }[failure]
            patch.setattr(module, name, fail)
            expected = OSError
        with pytest.raises(expected):
            probe.atomic_state(path, {"phase": "after"})
    assert probe.read_record(path) == {"phase": "before"}
    assert not list(tmp_path.glob(".state-*")), (
        "failed writes must remove temporary state files"
    )


def test_host_lock_serializes_independent_recovery_and_rejects_bad_metadata(
    host: HostHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    with host.recovery.locked():
        with pytest.raises(BlockingIOError):
            probe.Recovery(host.scope).cleanup()
    with monkeypatch.context() as patch:
        patch.setattr(
            probe.os,
            "fstat",
            lambda _fd: SimpleNamespace(
                st_mode=stat.S_IFREG | 0o600, st_uid=os.geteuid(), st_nlink=2
            ),
        )
        with pytest.raises(probe.RecoveryError, match="lock"):
            probe.Recovery(host.scope).cleanup()
    assert host.enable.exists() and host.calls == []


def test_missing_journal_with_owned_artifacts_cannot_be_replaced_by_a_tombstone(
    host: HostHarness,
) -> None:
    host.arm()
    host.recovery.disable()
    host.recovery.state_path.unlink()
    with pytest.raises(probe.RecoveryError, match="residual files"):
        probe.Recovery(host.scope).cleanup()
    assert (probe.SYSTEMD / host.recovery.timer).exists(), (
        "missing journal must not remove the recovery timer"
    )
    assert not host.recovery.state_path.exists(), (
        "missing ownership proof must not create a tombstone"
    )


def test_disable_rechecks_clock_after_slow_verification(host: HostHarness) -> None:
    host.arm()
    original = host.systemctl
    calls = 0

    def slow(*args: str) -> dict[str, str]:
        nonlocal calls
        result = original(*args)
        calls += 1
        if calls == 3:
            host.now += probe.ACK_SECONDS + 1
        return result

    host.monkeypatch.setattr(probe, "systemctl", slow)
    with pytest.raises(probe.RecoveryError, match="expired during"):
        host.recovery.disable()
    assert host.enable.exists() and host.read()["phase"] == "ARMED"


def test_recreated_enable_link_prevents_disable(host: HostHarness) -> None:
    host.arm()
    host.enable.unlink()
    host.symlink(str(host.fragment), host.enable)
    replacement = probe.file_identity(host.enable)
    with pytest.raises(probe.RecoveryError, match="owner changed"):
        host.recovery.disable()
    assert probe.file_identity(host.enable) == replacement
    assert host.read()["disable_started"] is False


def test_failed_disable_confirmation_keeps_restore_intent(host: HostHarness) -> None:
    host.arm()
    host.overrides[probe.AGENT] = {"UnitFileState": "enabled"}
    with pytest.raises(probe.RecoveryError, match="not verified"):
        host.recovery.disable()
    assert host.read()["phase"] == "DISABLING"
    assert not host.enable.exists(), (
        "lost disable confirmation must retain the disabled state"
    )
    host.overrides.clear()
    assert probe.Recovery(host.scope).cleanup()["phase"] == "CLOSED"


def test_direct_restore_after_expiry_cannot_issue_start(host: HostHarness) -> None:
    host.arm()
    host.recovery.disable()
    host.reboot()
    host.now = host.scope["expires_at"]
    with pytest.raises(probe.RecoveryError, match="expired"):
        host.recovery.restore()
    assert host.read()["phase"] == "EXPIRED"
    assert not host.enable.exists(), "expiry must not re-enable the Agent"
    assert not [
        call for call in host.calls if call[0] == "start" and call[-1] == probe.AGENT
    ]


def test_retirement_refuses_recreated_timer_even_without_disable(
    host: HostHarness,
) -> None:
    host.arm()
    timer = probe.SYSTEMD / host.recovery.timer
    timer.unlink()
    timer.write_text("new independent owner")
    timer.chmod(0o600)
    before = probe.file_identity(timer)
    with pytest.raises(probe.RecoveryError, match="replaced owner"):
        host.recovery.cleanup()
    assert probe.file_identity(timer) == before
    assert (probe.SYSTEMD / host.recovery.service).exists(), (
        "timer ownership drift must preserve the service unit"
    )


def test_removed_service_file_does_not_authorize_stopping_an_unknown_invocation(
    host: HostHarness,
) -> None:
    host.arm()
    host.recovery.restore()
    host.tick()
    host.running[host.recovery.service] = True
    host.overrides[host.recovery.service] = {"InvocationID": "foreign"}
    with pytest.raises(probe.RecoveryError, match="owner is unproven"):
        host.recovery.cleanup()
    assert ("stop", host.recovery.service) not in host.calls
    host.overrides.clear()
    assert host.recovery.cleanup()["phase"] == "CLOSED"


def test_nonterminal_retirement_is_refused(host: HostHarness) -> None:
    host.arm()
    with host.recovery.locked():
        with pytest.raises(probe.RecoveryError, match="not terminal"):
            host.recovery.retire_locked(independent=False)
    assert (probe.SYSTEMD / host.recovery.timer).exists(), (
        "nonterminal recovery must retain its timer"
    )


def test_pin_mismatch_and_empty_environment_lines_are_checked(
    host: HostHarness,
) -> None:
    values = {env: host.scope[key] for key, env in probe.PIN_KEYS.items()}
    values["NODE_NAME"] = "other-node"
    probe.AGENT_ENV.write_text(
        "\n\n# comment\n" + "\n".join(f"{key}={value}" for key, value in values.items())
    )
    with pytest.raises(probe.RecoveryError, match="pins changed"):
        host.recovery.prepare()
    assert host.enable.exists(), "pin mismatch must leave Agent boot activation enabled"


def test_relative_installer_enable_link_is_not_silently_normalized(
    host: HostHarness,
) -> None:
    host.links[host.enable] = "../" + probe.AGENT
    original = Path.resolve

    def resolve(path: Path, *args: Any, **kwargs: Any) -> Path:
        if path == host.enable:
            return host.fragment
        return original(path, *args, **kwargs)

    host.monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(probe.RecoveryError, match="installer link"):
        host.recovery.prepare()
    assert host.enable.exists(), "unrecognized installer links must remain untouched"


def test_main_rejects_a_journal_key_that_no_longer_matches_binding(
    host: HostHarness, capsys: pytest.CaptureFixture[str]
) -> None:
    host.recovery.prepare()
    host.update(binding={**host.scope, "owner": "another"})
    assert probe.main(["tick", "--key", host.recovery.key]) == 1
    assert json.loads(capsys.readouterr().out) == {
        "error": "RecoveryError",
        "recovery_required": True,
    }


def test_disabled_phase_requires_durable_disable_intent(host: HostHarness) -> None:
    host.arm()
    host.recovery.disable()
    host.update(disable_started=False)
    with pytest.raises(probe.RecoveryError, match="journal state"):
        host.recovery.cleanup()
    assert not host.enable.exists(), "invalid disable intent must not restore the Agent"


def test_empty_journal_cannot_mark_an_unproven_agent_recovered(
    host: HostHarness,
) -> None:
    host.arm()
    host.enable.unlink()
    with pytest.raises(probe.RecoveryError, match="restoration proof"):
        host.recovery.cleanup()
    assert host.read()["phase"] != "CLOSED"


def test_slow_intent_fsync_cannot_disable_after_the_ack_expires(
    host: HostHarness,
) -> None:
    host.arm()
    original = probe.atomic_state

    def save(path: Path, value: dict[str, Any]) -> None:
        original(path, value)
        if value["phase"] == "DISABLING":
            host.now += probe.ACK_SECONDS + 1

    host.monkeypatch.setattr(probe, "atomic_state", save)
    with pytest.raises(probe.RecoveryError, match="saving intent"):
        host.recovery.disable()
    assert host.enable.exists(), "expired arm ACK must not disable the Agent"
    assert probe.Recovery(host.scope).cleanup()["phase"] == "CLOSED"


def test_slow_start_intent_fsync_cannot_enqueue_after_expiry(host: HostHarness) -> None:
    host.arm()
    host.recovery.disable()
    host.reboot()
    original = probe.atomic_state

    def save(path: Path, value: dict[str, Any]) -> None:
        original(path, value)
        if value["phase"] == "RESTORING" and value["start_requested"]:
            host.now = host.scope["expires_at"]

    host.monkeypatch.setattr(probe, "atomic_state", save)
    with pytest.raises(probe.RecoveryError, match="saving intent"):
        host.recovery.restore()
    assert host.read()["phase"] == "EXPIRED"
    assert not [
        call for call in host.calls if call[0] == "start" and call[-1] == probe.AGENT
    ]


def test_timer_stop_must_be_proved_before_removing_units(host: HostHarness) -> None:
    host.arm()
    host.stop_stuck = True
    with pytest.raises(probe.RecoveryError, match="timer has not stopped"):
        host.recovery.cleanup()
    assert (probe.SYSTEMD / host.recovery.timer).exists(), (
        "unproved timer stop must retain the timer unit"
    )
    assert (probe.SYSTEMD / host.recovery.service).exists(), (
        "unproved timer stop must retain the service unit"
    )
    assert host.read()["phase"] == "RESTORED"


def test_service_stop_must_be_proved_separately_from_timer_stop(
    host: HostHarness,
) -> None:
    host.arm()
    host.running[host.recovery.service] = True
    original = host.systemctl

    def stuck(*args: str) -> dict[str, str]:
        if args == ("stop", host.recovery.service):
            return {}
        return original(*args)

    host.monkeypatch.setattr(probe, "systemctl", stuck)
    with pytest.raises(probe.RecoveryError, match="process has not stopped"):
        host.recovery.cleanup()
    assert not host.running[host.recovery.timer]
    assert (probe.SYSTEMD / host.recovery.service).exists(), (
        "unproved process exit must retain the service unit"
    )


def test_foreign_loaded_timer_is_not_stopped(host: HostHarness) -> None:
    host.arm()
    host.overrides[host.recovery.timer] = {"FragmentPath": "/run/foreign.timer"}
    with pytest.raises(probe.RecoveryError, match="timer owner"):
        host.recovery.cleanup()
    assert ("stop", host.recovery.timer) not in host.calls


def test_missing_hardware_identity_prevents_arming(host: HostHarness) -> None:
    original = Path.read_bytes

    def read(path: Path) -> bytes:
        if path == Path("/sys/class/dmi/id/product_uuid"):
            return b""
        return original(path)

    host.monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(probe.RecoveryError, match="incarnation evidence"):
        host.recovery.prepare()
    assert host.enable.exists(), "missing hardware identity must not disable the Agent"


def test_missing_resource_receipts_cannot_produce_a_forensic_tombstone(
    host: HostHarness,
) -> None:
    host.arm()
    host.update(resources=[])
    before = list(host.calls)
    with pytest.raises(probe.RecoveryError, match="publication receipt"):
        probe.Recovery(host.scope).cleanup()
    assert host.calls == before
    assert (probe.SYSTEMD / host.recovery.timer).exists(), (
        "missing publication receipts must preserve the timer"
    )
    assert host.read()["phase"] == "ARMED"


def test_absent_journal_cannot_adopt_a_preexisting_persistent_unit(
    host: HostHarness,
) -> None:
    target = probe.SYSTEMD / host.recovery.timer
    target.write_text("unrelated owner")
    target.chmod(0o600)
    before = probe.file_identity(target)
    with pytest.raises(probe.RecoveryError, match="publication receipt"):
        host.recovery.cleanup()
    assert probe.file_identity(target) == before
    assert not host.recovery.state_path.exists(), (
        "foreign units must not acquire a fabricated journal"
    )


def test_disabled_agent_requires_complete_baseline_evidence(host: HostHarness) -> None:
    host.arm()
    host.recovery.disable()
    host.update(baseline=None)
    with pytest.raises(probe.RecoveryError, match="complete recovery journal"):
        host.recovery.cleanup()
    assert not host.enable.exists(), "missing baseline must not enable the Agent"
    assert (probe.SYSTEMD / host.recovery.timer).exists(), (
        "missing baseline must preserve the recovery timer"
    )


def test_closed_status_is_forensic_not_an_unfinished_action(host: HostHarness) -> None:
    host.arm()
    host.recovery.cleanup()
    status = probe.Recovery(host.scope).status()
    assert status["phase"] == "CLOSED"
    assert status["record_kind"] == "FORENSIC_TOMBSTONE"
    assert all(
        not Path(item["target"]).exists() for item in host.read()["resources"]
    ), "CLOSED recovery must leave no actionable resource targets"
    assert host.recovery.state_path.exists(), (
        "CLOSED recovery must retain its forensic journal"
    )


def test_independent_failed_start_remains_unfinished_until_bounded_expiry(
    host: HostHarness,
) -> None:
    host.arm()
    host.recovery.disable()
    host.reboot()
    host.start_stuck = True
    host.now = host.scope["restore_at"]
    with pytest.raises(probe.RecoveryError, match="not complete"):
        host.tick()
    assert host.read()["phase"] == "RESTORING"
    assert (probe.SYSTEMD / host.recovery.timer).exists(), (
        "pending recovery must retain its timer until expiry"
    )
    host.now = host.scope["expires_at"]
    status = host.tick()
    assert status["phase"] == "EXPIRED"
    assert status["record_kind"] == "UNFINISHED_RECOVERY"
    assert not (probe.SYSTEMD / host.recovery.timer).exists(), (
        "expired recovery must remove its timer"
    )
    assert not host.agent_active, (
        "expired failed recovery must not report an active Agent"
    )
    assert (
        len(
            [
                call
                for call in host.calls
                if call[0] == "start" and call[-1] == probe.AGENT
            ]
        )
        == 1
    )


def test_expiry_preserves_job_bounds_until_pending_agent_job_is_gone(
    host: HostHarness,
) -> None:
    host.arm()
    host.recovery.disable()
    host.reboot()
    host.agent_job = "9 /pending/job"
    host.now = host.scope["expires_at"]
    with pytest.raises(probe.RecoveryError, match="job is pending"):
        host.tick()
    assert host.read()["phase"] == "EXPIRED"
    assert host.recovery.dropin.exists(), (
        "pending Agent jobs must retain their timeout bounds"
    )
    assert not host.running[host.recovery.timer]
    host.agent_job = ""
    assert host.tick()["record_kind"] == "UNFINISHED_RECOVERY"
    assert not host.recovery.dropin.exists(), (
        "settled Agent jobs must release the owned timeout drop-in"
    )
    assert not (probe.SYSTEMD / host.recovery.timer).exists(), (
        "expired recovery must leave no actionable timer"
    )


def test_boot_identity_reader_uses_only_the_boot_id_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded: list[Path] = []

    def read(path: Path, **kwargs: Any) -> str:
        recorded.append(path)
        return "test-boot\n"

    monkeypatch.setattr(Path, "read_text", read)
    assert probe.boot_id() == "test-boot"
    assert recorded == [Path("/proc/sys/kernel/random/boot_id")]
