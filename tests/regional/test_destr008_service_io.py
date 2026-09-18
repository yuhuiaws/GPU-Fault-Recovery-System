from __future__ import annotations

import argparse
import copy
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.probes import warm_spare_node_probe as probe
from tests.regional._destr008_service_window import Host, write


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Host:
    monkeypatch.setenv("INVOCATION_ID", "a" * 32)
    return Host(tmp_path, monkeypatch)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("LoadState", "error"),
        ("Transient", "yes"),
        ("NeedDaemonReload", "yes"),
        ("UnitFileState", "unknown"),
        ("FragmentPath", "relative.service"),
        ("DropInPaths", "relative.conf"),
        ("DropInPaths", "/same.conf /same.conf"),
        ("EnvironmentFiles", "relative (ignore_errors=no)"),
        ("ExecStart", "unknown"),
        ("ExecStart", "{ path=relative ; argv[]=relative ; ignore_errors=no ; pid=0 }"),
        ("ExecStart", "{ path=/a ; argv[]=/b ; ignore_errors=no ; pid=0 }"),
    ],
)
def test_ambiguous_service_definition_is_rejected(
    host: Host, field: str, value: str
) -> None:
    host.overrides["kubelet.service"] = {field: value}
    with pytest.raises(probe.ProbeError):
        probe.capture("kubelet.service")
    assert host.mutations() == []


@pytest.mark.parametrize(
    "assignment", ["MALFORMED=one two", "not-an-assignment", "NODE_NAME=duplicate"]
)
def test_ambiguous_environment_identity_is_rejected(
    host: Host, assignment: str
) -> None:
    probe.AGENT_ENV.write_text(probe.AGENT_ENV.read_text() + "\n" + assignment)
    with pytest.raises(probe.ProbeError, match="assignments"):
        probe.capture("kubelet.service")
    assert host.mutations() == []


def test_empty_comments_do_not_change_identity_pins(host: Host) -> None:
    before = probe.capture("kubelet.service")
    probe.AGENT_ENV.write_text(probe.AGENT_ENV.read_text() + "\n\n# private comment\n")
    after = probe.capture("kubelet.service")
    assert before["pins"] == after["pins"]
    assert before["baseline_sha256"] != after["baseline_sha256"]


@pytest.mark.parametrize("value", ["", "unknown", "none", "uninitialized"])
def test_unknown_machine_identity_cannot_authorize_recovery(
    host: Host, value: str
) -> None:
    write(probe.MACHINE_ID, value)
    with pytest.raises(probe.ProbeError, match="incarnation"):
        probe.capture("kubelet.service")


def test_inactive_service_is_not_an_implicit_auto_recovery_request(host: Host) -> None:
    host.units["kubelet.service"].update(
        ActiveState="inactive", SubState="dead", MainPID="0"
    )
    with pytest.raises(probe.ProbeError, match="active and idle"):
        probe.capture("kubelet.service")
    with pytest.raises(probe.ProbeError, match="baseline"):
        host.window.prepare()
    assert host.mutations() == []


@pytest.mark.parametrize(
    "defect", ["host", "node", "boot", "baseline", "expired", "too-long", "helper"]
)
def test_prepare_rejects_drift_and_nonrenewable_deadline(
    host: Host, defect: str
) -> None:
    binding = dict(host.binding)
    if defect in {"host", "baseline", "helper"}:
        binding[
            {
                "host": "host_sha256",
                "baseline": "baseline_sha256",
                "helper": "helper_sha256",
            }[defect]
        ] = "f" * 64
    elif defect == "node":
        binding["node_uid"] = "another-node-uid"
    elif defect == "boot":
        binding["boot_id"] = "another-boot"
    else:
        binding["restore_at"] = int(host.now) + (1 if defect == "expired" else 1000)
        binding["expires_at"] = binding["restore_at"] + probe.RECOVERY_SECONDS
    window = probe.ServiceWindow(binding)
    with pytest.raises(probe.ProbeError):
        window.prepare()
    assert not any(c[0] in {"start", "stop"} for c in host.mutations()), (
        host.mutations()
    )


def test_service_snapshot_detects_changed_invocation_before_scheduling(
    host: Host,
) -> None:
    host.arm()
    host.units["kubelet.service"]["InvocationID"] = "f" * 32
    before = host.mutations()
    with pytest.raises(probe.ProbeError, match="invocation changed"):
        host.window.schedule_stop()
    assert host.mutations() == before


@pytest.mark.parametrize("clock", ["boot", "wall", "elapsed"])
def test_unknown_clock_is_not_authority(host: Host, clock: str) -> None:
    host.arm()
    if clock == "boot":
        write(probe.BOOT_ID, "changed-boot")
    elif clock == "wall":
        host.now = float("nan")
    else:
        host.elapsed = float("inf")
    with pytest.raises(probe.ProbeError, match="boot changed|clock is unavailable"):
        host.window.remaining("expires")


def test_owned_cgroup_empty_and_foreign_cgroup_are_distinguished(host: Host) -> None:
    host.arm()
    unit = copy.deepcopy(host.units[host.window.units["stop-service"]])
    unit["ControlGroup"] = f"/system.slice/{unit['Id']}"
    write(
        probe.CGROUP_ROOT / "system.slice" / unit["Id"] / "cgroup.events",
        "populated 0\nfrozen 0\n",
    )
    assert probe.idle(unit) is True
    unit["ControlGroup"] = "/system.slice/foreign.service"
    with pytest.raises(probe.ProbeError, match="cgroup identity"):
        probe.idle(unit)


def test_missing_journal_cannot_fence_foreign_loaded_units_or_files(host: Host) -> None:
    name = host.window.units["stop-service"]
    host.units[name] = host.empty(name, "loaded")
    with pytest.raises(probe.ProbeError, match="systemd cleanup"):
        host.window.cleanup()
    host.units[name] = host.empty(name)
    path = write(host.window.targets["claim"], "foreign claim")
    with pytest.raises(probe.ProbeError, match="resource cleanup"):
        host.window.cleanup()
    assert path.read_text() == "foreign claim"


def test_stolen_publication_source_cannot_be_used_for_restore(host: Host) -> None:
    host.stopped()
    source = Path(host.read()["resources"]["stop-service"]["source"])
    source.write_text("changed source")
    before = host.mutations()
    with pytest.raises(probe.ProbeError, match="source was replaced"):
        host.window.cleanup()
    assert host.mutations() == before


def test_unit_file_loss_between_validation_and_stop_refuses(
    host: Host, monkeypatch: pytest.MonkeyPatch
) -> None:
    host.arm()
    host.window.schedule_stop()
    own_state = host.window.own_state

    def replaced(role: str, **kwargs: Any) -> dict[str, str]:
        unit = own_state(role, **kwargs)
        host.window.targets[role].unlink()
        return unit

    monkeypatch.setattr(host.window, "own_state", replaced)
    before = host.mutations()
    with pytest.raises(probe.ProbeError, match="without its owned publication"):
        host.window.quiesce("stop-timer")
    assert host.mutations() == before


@pytest.mark.parametrize("defect", ["deadline", "state"])
def test_fsync_delay_or_concurrent_state_change_cannot_authorize_late_start(
    host: Host, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    host.stopped()
    save = host.window.save

    def change_after_intent() -> None:
        save()
        if host.window.record["start_intent"]:
            if defect == "deadline":
                host.now = host.binding["expires_at"] - probe.JOB_SECONDS - 20
            else:
                host.units["kubelet.service"]["InvocationID"] = "f" * 32

    monkeypatch.setattr(host.window, "save", change_after_intent)
    before = host.mutations()
    with pytest.raises(
        probe.ProbeError, match="during intent|changed before restoration"
    ):
        host.window.restore()
    assert not any(c[0] == "start" for c in host.mutations()[len(before) :]), (
        host.mutations()
    )


@pytest.mark.parametrize("defect", ["failed", "new-job", "new-invocation", "queued"])
def test_start_completion_requires_the_owned_job_and_invocation(
    host: Host, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    host.stopped()
    host.pending_start = defect != "failed"
    run = probe.systemctl
    sleep = host.sleep

    def start(*args: str) -> dict[str, str]:
        result = run(*args)
        if args[0] == "start" and args[-1] == "kubelet.service":
            if defect == "failed":
                host.units["kubelet.service"].update(
                    ActiveState="failed",
                    SubState="failed",
                    MainPID="0",
                    InvocationID="1" * 32,
                )
            elif defect == "queued":
                host.units["kubelet.service"]["InvocationID"] = "1" * 32
        return result

    def complete(seconds: float) -> None:
        sleep(seconds)
        unit = host.units["kubelet.service"]
        if defect == "new-job":
            unit["Job"] = "92"
        elif defect in {"new-invocation", "queued"}:
            unit.update(
                ActiveState="active",
                SubState="running",
                Job="0",
                InvocationID=("f" if defect == "new-invocation" else "2") * 32,
            )

    monkeypatch.setattr(probe, "systemctl", start)
    monkeypatch.setattr(probe.time, "sleep", complete)
    if defect == "queued":
        assert host.window.restore()["phase"] == "RESTORED"
    else:
        with pytest.raises(
            probe.ProbeError, match="availability|job was replaced|invocation changed"
        ):
            host.window.restore()
        assert host.read()["phase"] == "RESTORING"
    assert (
        host.mutations().count(
            ("start", "--no-block", "--job-mode=fail", "kubelet.service")
        )
        == 1
    )


@pytest.mark.parametrize("defect", ["reload", "target"])
def test_cleanup_requires_final_absence_and_unchanged_restore_receipt(
    host: Host, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    host.stopped()
    run = probe.systemctl

    def changed(*args: str) -> dict[str, str]:
        if args == ("daemon-reload",) and defect == "reload":
            return {}
        result = run(*args)
        if args == ("daemon-reload",) and defect == "target":
            host.units["kubelet.service"]["InvocationID"] = "f" * 32
        return result

    monkeypatch.setattr(probe, "systemctl", changed)
    with pytest.raises(
        probe.ProbeError, match="retirement is incomplete|proof was lost"
    ):
        host.window.cleanup()
    assert host.read()["phase"] == "CLOSING"
    assert host.window.targets["claim"].exists(), host.window.targets


def test_tick_of_closed_tombstone_does_not_rearm(host: Host) -> None:
    host.stopped()
    host.window.cleanup()
    before = list(host.calls)
    assert probe.ServiceWindow(host.binding).tick()["phase"] == "CLOSED"
    assert host.calls == before


def test_snapshot_and_compatibility_handlers_use_the_bound_protocol(
    host: Host, capsys: pytest.CaptureFixture[str]
) -> None:
    assert probe.main(["snapshot", "--service", "kubelet.service"]) == 0
    assert (
        json.loads(capsys.readouterr().out)["baseline_sha256"]
        == host.binding["baseline_sha256"]
    )
    host.arm()
    args = argparse.Namespace(binding=json.dumps(host.binding))
    probe.stop_with_failsafe(args)
    assert json.loads(capsys.readouterr().out)["phase"] == "SCHEDULED"
    probe.restore_service(args)
    assert json.loads(capsys.readouterr().out)["phase"] == "RESTORED"


def test_independent_key_cannot_select_a_different_binding(
    host: Host, capsys: pytest.CaptureFixture[str]
) -> None:
    host.arm()
    data = host.read()
    data["binding"]["owner"] = "f" * 32
    host.save(data)
    before = list(host.calls)
    assert probe.main(["watch", "--key", host.window.key]) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "ProbeError"
    assert host.calls == before


def test_non_object_private_record_is_not_proof(tmp_path: Path) -> None:
    path = write(tmp_path / "state.json", "[]")
    with pytest.raises(probe.ProbeError, match="malformed"):
        probe.private_read(path)


@pytest.mark.parametrize("defect", ["owner", "writable", "link"])
def test_file_identity_protects_owner_and_simulated_link_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    path = write(tmp_path / "owned", "regular test file")
    original = Path.lstat
    info = original(path)

    def metadata(self: Path) -> Any:
        if self != path:
            return original(self)
        return SimpleNamespace(
            st_dev=info.st_dev,
            st_ino=info.st_ino,
            st_uid=info.st_uid + (1 if defect == "owner" else 0),
            st_mode=stat.S_IFLNK | 0o777
            if defect == "link"
            else info.st_mode | (0o002 if defect == "writable" else 0),
        )

    monkeypatch.setattr(Path, "lstat", metadata)
    if defect == "link":
        monkeypatch.setattr(os, "readlink", lambda _path: "owned-target")
        assert probe.file_identity(path)["link"] == "owned-target"
    else:
        with pytest.raises(probe.ProbeError, match="owner|protected regular"):
            probe.file_identity(path)


@pytest.mark.parametrize("defect", ["open", "read"])
def test_regular_file_identity_rejects_replacement_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    path = write(tmp_path / "owned", "regular test file")
    original = os.fstat
    calls = 0

    def metadata(fd: int) -> Any:
        nonlocal calls
        calls += 1
        info = original(fd)
        values = {
            key: getattr(info, key)
            for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        }
        if defect == "open" or calls == 2:
            values["st_ino"] += 1
        return SimpleNamespace(**values)

    monkeypatch.setattr(os, "fstat", metadata)
    with pytest.raises(probe.ProbeError, match="changed while"):
        probe.file_identity(path)


def test_publication_and_deletion_are_create_only_and_receipt_bound(
    tmp_path: Path,
) -> None:
    source = write(tmp_path / "source", "owned")
    target = tmp_path / "target"
    receipt = probe.file_identity(source)
    assert probe.matches(target, receipt) is False
    probe.publish(source, target, receipt)
    with pytest.raises(FileExistsError):
        probe.publish(source, target, receipt)
    foreign = {**receipt, "ino": receipt["ino"] + 1}
    with pytest.raises(probe.ProbeError, match="source changed"):
        probe.publish(source, tmp_path / "other", foreign)
    with pytest.raises(probe.ProbeError, match="replaced"):
        probe.remove_owned(target, foreign)
    assert target.read_text() == "owned"
    probe.remove_owned(target, receipt)
    probe.remove_owned(target, receipt)
    assert not target.exists() and source.is_file()
