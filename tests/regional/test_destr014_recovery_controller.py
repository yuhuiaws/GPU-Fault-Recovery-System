from __future__ import annotations

import copy
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import destr014_recovery as controller
from scripts.e2e.regional.probes import destr014_recovery_probe as probe
from tests.regional.test_destr014_recovery_probe import HostHarness, binding


class ProbeTransport:
    def __init__(self, host: HostHarness) -> None:
        self.host = host
        self.commands: list[str] = []
        self.lost_ack = ""
        self.arm_ack = True
        self.creates = 0
        self.cleanup_count = 0
        self.residuals: Any = {}
        self.reports: dict[str, Any] = {}

    def create(self) -> None:
        self.creates += 1

    def execute(
        self, command: str, flag: str, raw: str, *, timeout: int
    ) -> dict[str, Any]:
        assert flag == "--binding" and timeout == 120
        self.commands.append(command)
        value = json.loads(raw)
        instance = probe.Recovery(value)
        result = getattr(instance, command)()
        if command == "prepare" and self.arm_ack:
            self.host.tick()
        if self.lost_ack == command:
            self.lost_ack = ""
            raise TimeoutError("transport ACK lost")
        return copy.deepcopy(self.reports.get(command, result))

    def cleanup(self) -> dict[str, bool]:
        self.cleanup_count += 1
        return self.residuals


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    host = HostHarness(tmp_path, monkeypatch)
    journal = controller.RunJournal(
        tmp_path / "controller/journal.json", {"run": "one"}
    )
    journal.acquire()
    transport = ProbeTransport(host)
    window = controller.AgentRecoveryWindow(journal, transport)
    yield SimpleNamespace(host=host, journal=journal, probe=transport, window=window)
    journal.close()


def test_controller_roundtrip_requires_independent_arm_and_durable_cleanup(
    setup: Any,
) -> None:
    assert setup.window.arm(setup.host.scope)["phase"] == "ARMED"
    assert setup.window.disable()["phase"] == "DISABLED"
    setup.host.reboot()
    assert setup.window.restore()["restore_reason"] == "controller"
    assert setup.window.cleanup()["phase"] == "CLOSED"
    assert setup.probe.commands == [
        "prepare",
        "status",
        "disable",
        "restore",
        "cleanup",
    ]
    assert setup.probe.creates == 5 and setup.probe.cleanup_count == 1
    saved = json.loads(setup.journal.path.read_text())
    assert saved["host_binding"] == setup.host.scope
    assert saved["host_cleanup"]["phase"] == "CLOSED"


@pytest.mark.parametrize("operation", ["prepare", "disable", "restore", "cleanup"])
def test_lost_ack_resume_is_cleanup_only_and_never_replays_mutation(
    setup: Any, operation: str
) -> None:
    if operation != "prepare":
        setup.window.arm(setup.host.scope)
    if operation in {"restore", "cleanup"}:
        setup.window.disable()
        setup.host.reboot()
    setup.probe.lost_ack = operation
    with pytest.raises(TimeoutError):
        if operation == "prepare":
            setup.window.arm(setup.host.scope)
        else:
            getattr(setup.window, operation)()
    saved = json.loads(setup.journal.path.read_text())
    assert saved["host_request"] == operation
    setup.journal.close()
    journal = controller.RunJournal(setup.journal.path, setup.journal.scope)
    journal.acquire()
    try:
        assert journal.resumed, "lost-ACK state must reopen as cleanup-only recovery"
        fresh = controller.AgentRecoveryWindow(journal, setup.probe)
        with pytest.raises(controller.RegionalFixtureError, match="cleanup only"):
            fresh.arm(setup.host.scope)
        assert fresh.cleanup()["phase"] == "CLOSED"
        assert setup.host.enable.exists() and setup.host.agent_active
    finally:
        journal.close()
    assert setup.probe.commands.count("prepare") == 1
    assert setup.probe.commands.count("disable") <= 1


def test_missing_arm_ack_times_out_without_disabling(setup: Any) -> None:
    setup.probe.arm_ack = False
    with pytest.raises(controller.RegionalFixtureError, match="ACK was not observed"):
        setup.window.arm(setup.host.scope)
    assert setup.host.enable.exists() and "disable" not in setup.probe.commands
    assert setup.window.cleanup()["phase"] == "CLOSED"


@pytest.mark.parametrize(
    "defect",
    [
        "boot",
        "time",
        "future",
        "missing-time",
        "invocation",
        "binding",
        "phase",
        "types",
    ],
)
def test_invalid_arm_proof_blocks_disable(setup: Any, defect: str) -> None:
    original = setup.probe.execute

    def execute(command: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(command, *args, **kwargs)
        if command != "status":
            return result
        if defect == "boot":
            result["ack"]["boot_id"] = "another"
        elif defect == "time":
            result["ack"]["at"] -= probe.ACK_SECONDS + 1
        elif defect == "future":
            result["ack"]["at"] += 1
        elif defect == "missing-time":
            result["ack"].pop("at")
        elif defect == "invocation":
            result["ack"]["invocation_id"] = ""
        elif defect == "binding":
            result["binding_sha256"] = "e" * 64
        elif defect == "phase":
            result["phase"] = "RESTORED"
        else:
            result["disable_started"] = "false"
        return result

    setup.probe.execute = execute
    with pytest.raises(controller.RegionalFixtureError):
        setup.window.arm(setup.host.scope)
    assert "disable" not in setup.probe.commands
    assert setup.window.cleanup()["phase"] == "CLOSED"


def test_controller_rejects_a_disable_ack_without_disable_intent(setup: Any) -> None:
    setup.window.arm(setup.host.scope)
    setup.probe.reports["disable"] = {
        "binding_sha256": probe.binding_key(setup.host.scope),
        "phase": "DISABLED",
        "disable_started": False,
        "start_requested": False,
        "record_kind": "UNFINISHED_RECOVERY",
    }
    with pytest.raises(controller.RegionalFixtureError, match="not acknowledged"):
        setup.window.disable()
    assert setup.window.cleanup()["phase"] == "CLOSED"


def test_automatic_recovery_is_not_evidence_of_scenario_success(setup: Any) -> None:
    setup.window.arm(setup.host.scope)
    setup.window.disable()
    setup.host.reboot()
    setup.host.now = setup.host.scope["restore_at"]
    setup.host.tick()
    with pytest.raises(controller.RegionalFixtureError, match="automatic safeguard"):
        setup.window.restore()
    assert setup.window.cleanup()["phase"] == "CLOSED"


def test_controller_loss_after_disable_ack_preserves_recovery_time_across_reboot(
    setup: Any,
) -> None:
    setup.window.arm(setup.host.scope)
    setup.window.disable()
    saved = json.loads(setup.journal.path.read_text())
    assert saved["host_ack"]["phase"] == "DISABLED"
    timestamp = saved["host_binding"]["restore_at"]
    setup.journal.close()
    calls = list(setup.probe.commands)
    setup.host.reboot()
    setup.host.now = timestamp - 1
    assert setup.host.tick()["phase"] == "DISABLED"
    setup.host.now = timestamp
    restored = setup.host.tick()
    assert restored["phase"] == "RESTORED"
    assert restored["record_kind"] == "UNFINISHED_RECOVERY"
    assert setup.probe.commands == calls, "recovery cannot depend on controller polling"
    assert setup.host.read()["binding"]["restore_at"] == timestamp
    journal = controller.RunJournal(setup.journal.path, setup.journal.scope)
    journal.acquire()
    try:
        fresh = controller.AgentRecoveryWindow(journal, setup.probe)
        closed = fresh.cleanup()
        assert closed["record_kind"] == "FORENSIC_TOMBSTONE"
        assert not (probe.SYSTEMD / setup.host.recovery.timer).exists(), (
            "resumed cleanup must leave no persistent timer"
        )
        assert not (probe.SYSTEMD / setup.host.recovery.service).exists(), (
            "resumed cleanup must leave no recovery service unit"
        )
        assert not setup.host.recovery.dropin.exists(), (
            "resumed cleanup must release the owned timeout drop-in"
        )
    finally:
        journal.close()


def test_unfinished_record_is_not_accepted_as_cleanup_proof(setup: Any) -> None:
    setup.window.arm(setup.host.scope)
    setup.probe.reports["cleanup"] = {
        "binding_sha256": probe.binding_key(setup.host.scope),
        "phase": "CLOSED",
        "record_kind": "UNFINISHED_RECOVERY",
        "disable_started": False,
        "start_requested": False,
    }
    with pytest.raises(controller.RegionalFixtureError, match="unbound or incomplete"):
        setup.window.cleanup()
    assert setup.probe.cleanup_count == 0
    assert "host_cleanup" not in setup.journal.data
    setup.probe.reports.clear()
    assert setup.window.cleanup()["record_kind"] == "FORENSIC_TOMBSTONE"


@pytest.mark.parametrize("residuals", [{"pod": True}, None, ["unknown"]])
def test_cleanup_cannot_complete_with_residuals_or_unknown_shape(
    setup: Any, residuals: Any
) -> None:
    setup.window.arm(setup.host.scope)
    setup.probe.residuals = residuals
    with pytest.raises(controller.RegionalFixtureError, match="incomplete"):
        setup.window.cleanup()
    assert "host_cleanup" not in setup.journal.data
    setup.probe.residuals = {"pod": False}
    assert setup.window.cleanup()["phase"] == "CLOSED"


def test_cleanup_without_host_binding_has_no_remote_effect(setup: Any) -> None:
    assert setup.window.cleanup() == {"phase": "NOT_CREATED"}
    with pytest.raises(controller.RegionalFixtureError, match="no durable"):
        setup.window.disable()
    assert setup.probe.commands == [] and setup.probe.creates == 0


def test_make_binding_requires_same_agent_incarnation_and_runtime_pins() -> None:
    value = binding()
    scope = {
        key: value[key]
        for key in (
            "run_id",
            "release_id",
            "plan_sha256",
            "cluster_id",
            "node",
            "node_uid",
            "boot_id",
        )
    }
    agent = {
        "node_instance_id": scope["node_uid"],
        "artifact_sha256": value["artifact_sha256"],
        "installer_bundle_sha256": value["bundle_sha256"],
        "runtime_profile_version": value["profile_version"],
    }
    result = controller.make_binding(
        scope=scope,
        owner=value["owner"],
        agent=agent,
        restore_at=value["restore_at"],
        expires_at=value["expires_at"],
    )
    assert result == value
    with pytest.raises(controller.RegionalFixtureError, match="Node UID"):
        controller.make_binding(
            scope=scope,
            owner=value["owner"],
            agent={**agent, "node_instance_id": "other"},
            restore_at=value["restore_at"],
            expires_at=value["expires_at"],
        )
    with pytest.raises(probe.RecoveryError):
        controller.make_binding(
            scope=scope,
            owner=value["owner"],
            agent={},
            restore_at=value["restore_at"],
            expires_at=value["expires_at"],
        )


def test_journal_checkpoint_is_private_and_blocks_concurrent_owners(
    tmp_path: Path,
) -> None:
    path = tmp_path / "journal.json"
    first = controller.RunJournal(path, {"run": "one"})
    first.acquire()
    try:
        first.checkpoint(agent_disabled=True, marker="test-marker")
        assert path.stat().st_mode & 0o077 == 0
        with pytest.raises(BlockingIOError):
            controller.RunJournal(path, first.scope).acquire()
    finally:
        first.close()
    first.close()
    fresh = controller.RunJournal(path, first.scope)
    fresh.acquire()
    try:
        assert fresh.resumed, "a released journal must reopen with its original intent"
        assert fresh.data["run"] == {"agent_disabled": True, "marker": "test-marker"}
    finally:
        fresh.close()


@pytest.mark.parametrize(
    "defect",
    [
        "scope",
        "schema",
        "phase",
        "run",
        "supervision",
        "mode",
        "hardlink",
        "not-object",
    ],
)
def test_journal_corruption_or_identity_drift_refuses_resumption(
    tmp_path: Path, defect: str
) -> None:
    path = tmp_path / "journal.json"
    first = controller.RunJournal(path, {"run": "one"})
    first.acquire()
    first.close()
    data = first.data
    if defect == "scope":
        data["scope"] = {"run": "other"}
    elif defect == "schema":
        data["schema_version"] = 0
    elif defect == "phase":
        data["phase"] = "PASS"
    elif defect == "run":
        data["run"] = []
    elif defect == "supervision":
        data["supervision_lost"] = True
    elif defect == "not-object":
        data = []
    path.write_text(json.dumps(data))
    if defect == "mode":
        path.chmod(0o644)
    elif defect == "hardlink":
        os.link(path, tmp_path / "unrelated-link")
    with pytest.raises(controller.RegionalFixtureError):
        controller.RunJournal(path, first.scope).acquire()


@pytest.mark.parametrize("defect", ["type", "owner", "links", "mode"])
def test_journal_lock_rejects_unsafe_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    info = {"st_mode": stat.S_IFREG | 0o600, "st_uid": os.geteuid(), "st_nlink": 1}
    if defect == "type":
        info["st_mode"] = stat.S_IFIFO | 0o600
    elif defect == "owner":
        info["st_uid"] += 1
    elif defect == "links":
        info["st_nlink"] = 2
    else:
        info["st_mode"] |= 0o004
    monkeypatch.setattr(controller.os, "fstat", lambda _fd: SimpleNamespace(**info))
    with pytest.raises(controller.RegionalFixtureError, match="lock is invalid"):
        controller.RunJournal(tmp_path / "journal.json", {}).acquire()


def test_real_fresh_process_sees_crash_intent_and_releases_only_its_lock(
    tmp_path: Path,
) -> None:
    path = tmp_path / "journal.json"
    code = """
import json, os, sys
from pathlib import Path
from scripts.e2e.regional.destr014_recovery import RunJournal
journal = RunJournal(Path(sys.argv[1]), {"run": "owned-subprocess"})
journal.acquire()
journal.checkpoint(agent_disabled=True, injection_started=False)
os._exit(17)
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code, str(path)],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 17, completed.stderr
    fresh = controller.RunJournal(path, {"run": "owned-subprocess"})
    fresh.acquire()
    try:
        assert fresh.resumed and fresh.data["run"]["agent_disabled"] is True
        assert fresh.data["run"]["injection_started"] is False
        fresh.data["phase"] = "RECOVERY_REQUIRED"
        fresh.save()
    finally:
        fresh.close()


def test_a_closed_journal_of_another_attempt_is_archived_not_refused(
    tmp_path: Path,
) -> None:
    """Attempt 3 of 2026-09-18 met attempt 2's CLOSED journal (forensic
    tombstone) at the case's single journal path and was refused as "identity
    changed" before touching the node. A closed journal owns nothing: keep it
    under its run id and start the new attempt's journal fresh."""
    path = tmp_path / "journal.json"
    first = controller.RunJournal(path, {"run": "one", "run_id": "destr014-x-a2"})
    first.acquire()
    first.data["phase"] = "CLOSED"
    first.save()
    first.close()

    second = controller.RunJournal(path, {"run": "one", "run_id": "destr014-x-a3"})
    second.acquire()
    assert second.resumed is False, "the new attempt must not resume the tombstone"
    assert second.data["scope"]["run_id"] == "destr014-x-a3", second.data
    archived = sorted(tmp_path.glob("journal.closed-*.json"))
    assert [p.name for p in archived] == ["journal.closed-destr014-x-a2.json"], archived
    assert json.loads(archived[0].read_text())["phase"] == "CLOSED", archived


def test_controller_accepts_the_installer_restored_agent_on_a_new_boot(
    setup: Any,
) -> None:
    """Live: the runner's ``restore`` after the product's reboot met an Agent
    the node-installer had already re-enabled under a new link inode; the
    probe refused ("incarnation changed"), cleanup died at ``agent_recovery``
    and the scenario was never judged."""

    setup.window.arm(setup.host.scope)
    setup.window.disable()
    setup.host.reboot(renumber_devices=True)
    setup.host.installer_reinstall()
    restored = setup.window.restore()
    assert restored["phase"] == "RESTORED", restored
    assert restored["restored_by"] == "reboot-installer", restored
    assert restored["boot_id_observed"] == "boot-after", restored
    assert setup.journal.data["host_recovery_phase"] == controller.RESTORED_BY_REBOOT
    assert setup.journal.data["host_boot_id_observed"] == "boot-after"
    closed = setup.window.cleanup()
    assert closed["phase"] == "CLOSED" and closed["retired"] is True, closed
    saved = json.loads(setup.journal.path.read_text())
    assert saved["host_recovery_phase"] == "RESTORED_BY_REBOOT"
    assert saved["host_cleanup"]["restored_by"] == "reboot-installer"
    assert not (probe.SYSTEMD / setup.host.recovery.timer).exists(), (
        "the installer-restored Agent needs no persistent timer"
    )
    assert not setup.host.recovery.dropin.exists(), (
        "the 45 s start bound must not outlive the recovery"
    )


def test_controller_restore_by_the_saved_copy_on_a_new_boot_is_a_plain_restore(
    setup: Any,
) -> None:
    setup.window.arm(setup.host.scope)
    setup.window.disable()
    setup.host.reboot(renumber_devices=True)
    setup.host.installer_reinstall(enable=False)
    restored = setup.window.restore()
    assert restored["restored_by"] == "probe", restored
    assert restored["restore_reason"] == "controller", restored
    assert setup.journal.data["host_recovery_phase"] == "RESTORED"
    assert setup.window.cleanup()["phase"] == "CLOSED"
    assert setup.host.enable.exists() and setup.host.agent_active


def test_installer_restore_noticed_by_the_tick_is_not_the_safeguard_firing(
    setup: Any,
) -> None:
    setup.window.arm(setup.host.scope)
    setup.window.disable()
    setup.host.reboot()
    setup.host.installer_reinstall()
    setup.host.now = setup.host.scope["restore_at"] - 60
    ticked = setup.host.tick()
    assert ticked["phase"] == "RESTORED" and ticked["restore_reason"] == "reboot"
    restored = setup.window.restore()
    assert restored["restored_by"] == "reboot-installer", restored
    assert setup.journal.data["host_recovery_phase"] == "RESTORED_BY_REBOOT"
    assert setup.window.cleanup()["phase"] == "CLOSED"


def test_cleanup_records_the_reboot_restore_when_restore_never_ran(setup: Any) -> None:
    setup.window.arm(setup.host.scope)
    setup.window.disable()
    setup.host.reboot()
    setup.host.installer_reinstall()
    closed = setup.window.cleanup()
    assert closed["phase"] == "CLOSED" and closed["restored_by"] == "reboot-installer"
    assert setup.journal.data["host_recovery_phase"] == "RESTORED_BY_REBOOT"
    assert setup.journal.data["host_boot_id_observed"] == "boot-after"


@pytest.mark.parametrize(
    ("change", "gap"),
    [
        ({}, None),
        ({"phase": "CLOSED"}, None),
        ({"boot_id_observed": "boot-before"}, "unchanged"),
        ({"boot_id_observed": None}, "no boot id"),
        ({"phase": "DISABLED"}, "not RESTORED or CLOSED"),
        ({"restored_by": None}, "restoration receipt"),
        ({"disable_started": False}, "never disabled"),
    ],
)
def test_reboot_restore_proof_reads_only_the_host_record(
    change: dict[str, Any], gap: str | None
) -> None:
    report = {
        "phase": "RESTORED",
        "disable_started": True,
        "boot_id_observed": "boot-after",
        "restored_by": "probe",
        **change,
    }
    proof = controller.reboot_restore_proof(report, binding())
    assert proof["proven"] is (gap is None), proof
    assert proof["boot_id_before"] == "boot-before"
    if gap is not None:
        assert any(gap in item for item in proof["gaps"]), proof
    assert controller.reboot_restore_proof({}, binding())["proven"] is False
    assert (
        controller.reboot_restore_proof(
            report,
            binding(),
            agent_unit={"UnitFileState": "enabled", "ActiveState": "inactive"},
        )["proven"]
        is False
    ), "an inactive Agent snapshot is not a restored sibling"


@pytest.mark.parametrize("phase", ["OPEN", "RECOVERY_REQUIRED"])
def test_an_unfinished_journal_of_another_attempt_still_refuses(
    tmp_path: Path, phase: str
) -> None:
    path = tmp_path / "journal.json"
    first = controller.RunJournal(path, {"run": "one", "run_id": "a2"})
    first.acquire()
    first.data["phase"] = phase
    first.save()
    first.close()
    with pytest.raises(controller.RegionalFixtureError, match="identity changed"):
        controller.RunJournal(path, {"run": "one", "run_id": "a3"}).acquire()
    assert path.exists() and not list(tmp_path.glob("journal.closed-*")), (
        "an unfinished journal is never moved aside"
    )
