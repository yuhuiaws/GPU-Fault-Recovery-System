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
        # A foreign scope that never touched the host is retired, not refused
        # (see the never-armed tests below); drift only refuses once armed.
        data["scope"] = {"run": "other"}
        data["run"] = {"holder_armed": True}
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
    # controller.os is the global os module, which pytest's own tmp_path
    # teardown also uses; keep the fake to the call under test.
    with monkeypatch.context() as patch:
        patch.setattr(controller.os, "fstat", lambda _fd: SimpleNamespace(**info))
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
    """An unfinished journal that armed the GPU device holder may own host
    state; a later attempt is refused as identity drift, never moved aside."""

    path = tmp_path / "journal.json"
    first = controller.RunJournal(path, {"run": "one", "run_id": "a2"})
    first.acquire()
    first.data["phase"] = phase
    first.checkpoint(holder_armed=True)
    first.close()
    with pytest.raises(controller.RegionalFixtureError, match="identity changed"):
        controller.RunJournal(path, {"run": "one", "run_id": "a3"}).acquire()
    assert path.exists() and not list(tmp_path.glob("journal.*-*.json")), (
        "an unfinished journal armed on the host is never moved aside"
    )


@pytest.mark.parametrize("phase", ["OPEN", "RECOVERY_REQUIRED"])
def test_a_rebooted_unfinished_journal_is_retired_and_records_its_lineage(
    tmp_path: Path, phase: str
) -> None:
    """The live wedge (2026-09-18 attempt 4): a RECOVERY_REQUIRED journal bound
    to the sibling's original boot refused every later attempt as "identity
    changed" once the node rebooted. A journal armed on a boot the node no
    longer runs owns no host state -- the recovery probe's new-boot table
    already retired it -- so it is moved aside and the new attempt starts fresh,
    keeping the lineage for cleanup/verdict evidence."""

    path = tmp_path / "journal.json"
    first = controller.RunJournal(
        path, {"run": "one", "run_id": "a4", "boot_id": "boot-old"}
    )
    first.acquire()
    first.data["phase"] = phase
    first.data["host_request"] = "cleanup"
    first.save()
    first.close()

    second = controller.RunJournal(
        path, {"run": "one", "run_id": "a5", "boot_id": "boot-new"}
    )
    second.acquire()
    try:
        assert second.resumed is False, "a rebooted-away journal must not resume"
        assert second.data["scope"]["boot_id"] == "boot-new"
        archived = sorted(tmp_path.glob("journal.rebooted-*.json"))
        assert [p.name for p in archived] == ["journal.rebooted-a4.json"], archived
        assert json.loads(archived[0].read_text())["phase"] == phase
        assert not list(tmp_path.glob("journal.closed-*")), (
            "a rebooted journal uses the rebooted archive, not the closed one"
        )
        retired = second.data["run"]["retired_journals"]
        assert len(retired) == 1, retired
        entry = retired[0]
        assert entry["archive"] == str(archived[0])
        assert entry["old_run_id"] == "a4"
        assert entry["old_phase"] == phase
        assert entry["old_boot_id"] == "boot-old"
        assert entry["old_host_request"] == "cleanup"
        assert isinstance(entry["retired_at"], str) and entry["retired_at"]
    finally:
        second.close()


def test_an_unfinished_journal_on_the_same_boot_still_refuses(tmp_path: Path) -> None:
    """A journal whose boot id still matches and that disabled the sibling's
    Agent may own host state; a new run id on the same boot is refused as
    identity drift, never retired."""

    path = tmp_path / "journal.json"
    first = controller.RunJournal(
        path, {"run": "one", "run_id": "a4", "boot_id": "boot-same"}
    )
    first.acquire()
    first.data["phase"] = "RECOVERY_REQUIRED"
    first.checkpoint(agent_disabled=True)
    first.close()
    with pytest.raises(controller.RegionalFixtureError, match="identity changed"):
        controller.RunJournal(
            path, {"run": "one", "run_id": "a5", "boot_id": "boot-same"}
        ).acquire()
    assert path.exists() and not list(tmp_path.glob("journal.*-*.json")), (
        "a same-boot journal armed on the host is never moved aside"
    )


def test_a_rebooted_journal_that_lost_supervision_is_never_retired(
    tmp_path: Path,
) -> None:
    """Lost supervision demands independent review; even on a new boot such a
    journal is refused, never archived aside as a routine reboot retirement."""

    path = tmp_path / "journal.json"
    first = controller.RunJournal(
        path, {"run": "one", "run_id": "a4", "boot_id": "boot-old"}
    )
    first.acquire()
    first.data["phase"] = "RECOVERY_REQUIRED"
    first.data["supervision_lost"] = True
    first.save()
    first.close()
    with pytest.raises(controller.RegionalFixtureError):
        controller.RunJournal(
            path, {"run": "one", "run_id": "a5", "boot_id": "boot-new"}
        ).acquire()
    assert path.exists() and not list(tmp_path.glob("journal.rebooted-*")), (
        "a supervision-lost journal must not be retired on a new boot"
    )


def test_a_closed_journal_on_a_new_boot_uses_the_closed_archive(tmp_path: Path) -> None:
    """A CLOSED tombstone owns nothing regardless of boot; it keeps the closed
    archive name and is not treated as a reboot retirement."""

    path = tmp_path / "journal.json"
    first = controller.RunJournal(
        path, {"run": "one", "run_id": "a4", "boot_id": "boot-old"}
    )
    first.acquire()
    first.data["phase"] = "CLOSED"
    first.save()
    first.close()
    second = controller.RunJournal(
        path, {"run": "one", "run_id": "a5", "boot_id": "boot-new"}
    )
    second.acquire()
    try:
        assert second.resumed is False, "a closed tombstone must not resume"
        closed = sorted(tmp_path.glob("journal.closed-*.json"))
        assert [p.name for p in closed] == ["journal.closed-a4.json"], closed
        assert not list(tmp_path.glob("journal.rebooted-*")), (
            "a CLOSED journal is archived as closed, not rebooted"
        )
        assert "retired_journals" not in second.data["run"]
    finally:
        second.close()


@pytest.mark.parametrize(
    ("old_scope", "new_scope"),
    [
        (
            {"run": "one", "run_id": "a4"},
            {"run": "one", "run_id": "a5", "boot_id": "boot-new"},
        ),
        (
            {"run": "one", "run_id": "a4", "boot_id": "boot-old"},
            {"run": "one", "run_id": "a5"},
        ),
        (
            {"run": "one", "run_id": "a4", "boot_id": ""},
            {"run": "one", "run_id": "a5", "boot_id": "boot-new"},
        ),
    ],
)
def test_reboot_retirement_needs_a_string_boot_id_on_both_scopes(
    tmp_path: Path, old_scope: dict[str, Any], new_scope: dict[str, Any]
) -> None:
    """Without a non-empty boot id on both journals the reboot cannot be proven,
    so an unfinished journal that injected keeps refusing rather than being
    retired."""

    path = tmp_path / "journal.json"
    first = controller.RunJournal(path, old_scope)
    first.acquire()
    first.data["phase"] = "RECOVERY_REQUIRED"
    first.checkpoint(injection_started=True)
    first.close()
    with pytest.raises(controller.RegionalFixtureError, match="identity changed"):
        controller.RunJournal(path, new_scope).acquire()
    assert path.exists() and not list(tmp_path.glob("journal.*-*.json")), (
        "an armed journal without both boot ids is never retired"
    )


def unarmed_run(**overrides: Any) -> dict[str, Any]:
    """The ``run`` record of an attempt that died at the env-window step: the
    control-plane window was opened (the windows keep their own records and
    guards) but nothing was armed, disabled or written on the host."""

    return {
        "agent_disabled": False,
        "control_env_opened": True,
        "env_opened": False,
        "holder_armed": False,
        "injection_started": False,
        "physical_outcome_unknown": False,
        "incident_id": "",
        "marker": "",
        "follow_up_incident_id": "",
        "hold_reasons": None,
        "sibling_reboot_proof": {},
        "preflight": {"errors": []},
        "started_at": "started",
        **overrides,
    }


def seed_foreign_journal(
    path: Path,
    scope: dict[str, Any],
    *,
    phase: str = "RECOVERY_REQUIRED",
    run: dict[str, Any] | None = None,
    **records: Any,
) -> None:
    journal = controller.RunJournal(path, scope)
    journal.acquire()
    journal.data["phase"] = phase
    journal.data["run"] = unarmed_run() if run is None else run
    journal.data.update(records)
    journal.save()
    journal.close()


SAME_BOOT_OLD = {"run": "one", "run_id": "a6", "boot_id": "boot-same"}
SAME_BOOT_NEW = {"run": "one", "run_id": "a7", "boot_id": "boot-same"}


@pytest.mark.parametrize("phase", ["OPEN", "RECOVERY_REQUIRED"])
@pytest.mark.parametrize("env_opened", [False, True])
def test_a_never_armed_journal_on_the_same_boot_is_retired_and_records_its_lineage(
    tmp_path: Path, phase: str, env_opened: bool
) -> None:
    """The live wedge (2026-09-19 attempt 7): attempt 6 died at the env-window
    step before anything was armed on the sibling, so its RECOVERY_REQUIRED
    journal on the unchanged boot owned no host state, yet it refused every
    later attempt as "identity changed" -- the boot never changed, so the
    reboot rule could not retire it. A journal that provably never touched the
    host is moved aside and the new attempt starts fresh with the lineage."""

    path = tmp_path / "journal.json"
    seed_foreign_journal(
        path, SAME_BOOT_OLD, phase=phase, run=unarmed_run(env_opened=env_opened)
    )
    second = controller.RunJournal(path, SAME_BOOT_NEW)
    second.acquire()
    try:
        assert second.resumed is False, "a never-armed journal must not resume"
        assert second.data["scope"]["run_id"] == "a7"
        archived = sorted(tmp_path.glob("journal.abandoned-*.json"))
        assert [p.name for p in archived] == ["journal.abandoned-a6.json"], archived
        saved = json.loads(archived[0].read_text())
        assert saved["phase"] == phase and saved["scope"] == SAME_BOOT_OLD
        assert saved["run"]["control_env_opened"] is True, (
            "env-window flags do not block retirement; the windows have their "
            "own records and guards"
        )
        assert not list(tmp_path.glob("journal.rebooted-*")), (
            "an unchanged boot is not a reboot retirement"
        )
        assert not list(tmp_path.glob("journal.closed-*")), (
            "an unfinished journal never uses the closed archive"
        )
        retired = second.data["run"]["retired_journals"]
        assert len(retired) == 1, retired
        entry = retired[0]
        assert entry["archive"] == str(archived[0])
        assert entry["old_run_id"] == "a6"
        assert entry["old_phase"] == phase
        assert entry["old_boot_id"] == "boot-same"
        assert entry["old_host_request"] is None
        assert entry["reason"] == "never armed on the host"
        assert isinstance(entry["retired_at"], str) and entry["retired_at"]
        assert json.loads(path.read_text())["run"]["retired_journals"] == retired
    finally:
        second.close()


@pytest.mark.parametrize(
    "mark",
    [
        {"run": {"holder_armed": True}},
        {"run": {"injection_started": True}},
        {"run": {"agent_disabled": True}},
        {"run": {"physical_outcome_unknown": True}},
        {"run": {"incident_id": "inc-1"}},
        {"run": {"marker": "destr014-1-a6"}},
        {"run": {"follow_up_incident_id": "inc-support-after"}},
        {"host_binding": {"boot_id": "boot-same"}},
        {"host_ack": {"phase": "ARMED"}},
        {"host_request": "prepare"},
        {"host_cleanup": {"phase": "CLOSED"}},
    ],
    ids=[
        "holder_armed",
        "injection_started",
        "agent_disabled",
        "physical_outcome_unknown",
        "incident_id",
        "marker",
        "follow_up_incident_id",
        "host_binding",
        "host_ack",
        "host_request",
        "host_cleanup",
    ],
)
def test_a_journal_that_touched_the_host_is_never_retired_as_unarmed(
    tmp_path: Path, mark: dict[str, Any]
) -> None:
    """Every checkpoint the runner writes before a host mutation (arming the
    holder, disabling the Agent, writing the XIDs) and every record the
    recovery window writes keeps the same-boot journal refusing exactly as
    before; only a journal with none of them is retired."""

    path = tmp_path / "journal.json"
    records = {key: value for key, value in mark.items() if key != "run"}
    seed_foreign_journal(
        path, SAME_BOOT_OLD, run=unarmed_run(**mark.get("run", {})), **records
    )
    with pytest.raises(controller.RegionalFixtureError, match="identity changed"):
        controller.RunJournal(path, SAME_BOOT_NEW).acquire()
    assert path.exists() and not list(tmp_path.glob("journal.*-*.json")), (
        "a journal that touched the host is never moved aside"
    )
    assert json.loads(path.read_text())["scope"] == SAME_BOOT_OLD


def test_a_never_armed_journal_that_lost_supervision_is_never_retired(
    tmp_path: Path,
) -> None:
    """Lost supervision demands independent review even when no host flag was
    checkpointed: the journal is refused, never archived aside."""

    path = tmp_path / "journal.json"
    seed_foreign_journal(path, SAME_BOOT_OLD, supervision_lost=True)
    with pytest.raises(controller.RegionalFixtureError):
        controller.RunJournal(path, SAME_BOOT_NEW).acquire()
    assert path.exists() and not list(tmp_path.glob("journal.*-*.json")), (
        "a supervision-lost journal must not be retired as never armed"
    )


def test_a_never_armed_journal_from_a_replaced_boot_uses_the_rebooted_archive(
    tmp_path: Path,
) -> None:
    """The reboot rule runs first: a never-armed journal from a boot the node
    has replaced is retired once, as a reboot retirement, without a second
    abandoned archive or a second lineage entry."""

    path = tmp_path / "journal.json"
    seed_foreign_journal(path, {"run": "one", "run_id": "a6", "boot_id": "boot-old"})
    second = controller.RunJournal(
        path, {"run": "one", "run_id": "a7", "boot_id": "boot-new"}
    )
    second.acquire()
    try:
        archives = sorted(p.name for p in tmp_path.glob("journal.*-*.json"))
        assert archives == ["journal.rebooted-a6.json"], archives
        retired = second.data["run"]["retired_journals"]
        assert len(retired) == 1, retired
        assert retired[0]["old_boot_id"] == "boot-old"
        assert "reason" not in retired[0], retired
    finally:
        second.close()
