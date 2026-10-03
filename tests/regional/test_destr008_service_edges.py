from __future__ import annotations

import copy
import json
import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.probes import warm_spare_node_probe as probe
from tests.regional._destr008_service_window import Host, command_property, write


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Host:
    monkeypatch.setenv("INVOCATION_ID", "a" * 32)
    return Host(tmp_path, monkeypatch)


@pytest.mark.parametrize("key", sorted(probe.BINDING_KEYS))
def test_every_binding_field_is_required_before_host_io(host: Host, key: str) -> None:
    value = dict(host.binding)
    value.pop(key)
    before = list(host.calls)
    with pytest.raises(probe.ProbeError, match="incomplete"):
        probe.ServiceWindow(value)
    assert host.calls == before


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("case_id", "GF-REGIONAL-DESTR-014"),
        ("owner", "not-an-owner"),
        ("service", "sshd.service"),
        ("run_id", "unsafe;run"),
        ("node_uid", None),
        ("plan_sha256", "unknown"),
        ("restore_at", True),
        ("restore_at", 0),
        ("expires_at", 1),
        ("stop_delay_seconds", -1),
        ("stop_delay_seconds", 121),
    ],
)
def test_invalid_binding_never_constructs_authority(
    host: Host, key: str, value: Any
) -> None:
    binding = {**host.binding, key: value}
    before = list(host.calls)
    with pytest.raises(probe.ProbeError):
        probe.ServiceWindow(binding)
    assert host.calls == before


@pytest.mark.parametrize(
    "role",
    ["claim", "helper", "bound", "stop-service", "restore-service", "stop-timer"],
)
@pytest.mark.parametrize("after_link", [False, True])
def test_lost_publication_reply_can_be_cleaned_without_start_or_stop(
    host: Host, monkeypatch: pytest.MonkeyPatch, role: str, after_link: bool
) -> None:
    publish = probe.publish

    def interrupted(source: Path, target: Path, identity: dict[str, Any]) -> None:
        if after_link or target != host.window.targets[role]:
            publish(source, target, identity)
        if target == host.window.targets[role]:
            raise OSError("owned publication interrupted")

    monkeypatch.setattr(probe, "publish", interrupted)
    with pytest.raises(OSError, match="publication interrupted"):
        host.window.prepare()
    monkeypatch.setattr(probe, "publish", publish)
    assert host.read()["phase"] == "PREPARING"
    assert host.read()["stop_requested"] is False
    fresh = probe.ServiceWindow(host.binding)
    assert fresh.cleanup()["phase"] == "CLOSED"
    assert not any(c[0] in {"start", "stop"} for c in host.mutations()), (
        host.mutations()
    )
    assert not any((probe.SYSTEMD / n).exists() for n in fresh.units.values()), (
        fresh.units
    )


@pytest.mark.parametrize(
    "role", ["stop-timer", "stop-service", "restore-service", "bound", "claim"]
)
@pytest.mark.parametrize("after_unlink", [False, True])
def test_interrupted_retirement_resumes_without_restarting_service(
    host: Host, monkeypatch: pytest.MonkeyPatch, role: str, after_unlink: bool
) -> None:
    host.stopped()
    remove = probe.remove_owned

    def interrupted(path: Path, identity: dict[str, Any]) -> None:
        if after_unlink or path != host.window.targets[role]:
            remove(path, identity)
        if path == host.window.targets[role]:
            raise OSError("owned retirement interrupted")

    monkeypatch.setattr(probe, "remove_owned", interrupted)
    with pytest.raises(OSError, match="retirement interrupted"):
        host.window.cleanup()
    monkeypatch.setattr(probe, "remove_owned", remove)
    before = host.mutations()
    result = probe.ServiceWindow(host.binding).cleanup()
    assert result["phase"] == "CLOSED"
    assert not any(c[0] == "start" for c in host.mutations()[len(before) :]), (
        host.mutations()
    )
    assert not host.window.targets["claim"].exists(), host.window.targets


def test_interrupted_reload_after_unlink_retains_cleanup_ownership(host: Host) -> None:
    host.stopped()
    host.fail.add(("daemon-reload",))
    with pytest.raises(probe.ProbeError, match="systemctl failed"):
        host.window.cleanup()
    assert host.read()["retiring"] is True and host.read()["phase"] == "CLOSING"
    host.fail.clear()
    assert probe.ServiceWindow(host.binding).cleanup()["phase"] == "CLOSED"


@pytest.mark.parametrize("role", ["stop-timer", "stop-service", "restore-service"])
def test_preexisting_loaded_unit_is_not_claimed_by_matching_name(
    host: Host, role: str
) -> None:
    name = host.window.units[role]
    host.units[name] = host.empty(name, "loaded")
    before = host.mutations()
    with pytest.raises(probe.ProbeError, match="pre-existing"):
        host.window.prepare()
    assert host.mutations() == before and not host.window.targets["claim"].exists()


@pytest.mark.parametrize("role", ["claim", "helper", "stop-service"])
def test_preexisting_file_without_publication_receipt_is_not_touched(
    host: Host, role: str
) -> None:
    path = write(host.window.targets[role], "foreign")
    before = host.mutations()
    with pytest.raises(probe.ProbeError):
        host.window.prepare()
    assert path.read_text() == "foreign" and host.mutations() == before


@pytest.mark.parametrize("stage", ["prepare", "schedule", "owned-stop", "restore"])
def test_unknown_systemd_reads_never_authorize_new_target_actions(
    host: Host, stage: str
) -> None:
    if stage != "prepare":
        host.arm()
    if stage in {"owned-stop", "restore"}:
        host.window.schedule_stop()
    if stage == "restore":
        host.fire_stop()
    host.fail.add(
        next(
            c
            for c in reversed(host.calls)
            if c[0] == "show" and c[1] == host.binding["service"]
        )
    )
    before = host.mutations()
    with pytest.raises(probe.ProbeError, match="systemctl failed"):
        if stage == "owned-stop":
            host.window.stop_owned()
        else:
            getattr(host.window, {"schedule": "schedule_stop"}.get(stage, stage))()
    assert host.mutations() == before


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "old-wall",
        "old-elapsed",
        "future",
        "boot",
        "pid",
        "invocation",
        "dead",
        "job",
        "expiry",
        "nan",
    ],
)
def test_bad_independent_ack_cannot_schedule_stop(host: Host, defect: str) -> None:
    host.arm()
    record = host.read()
    ack = record["ack"]
    if defect == "missing":
        record.pop("ack")
    elif defect == "old-wall":
        ack["at"] -= probe.ACK_SECONDS + 1
    elif defect == "old-elapsed":
        ack["elapsed"] -= probe.ACK_SECONDS + 1
    elif defect == "future":
        ack["at"] += 1
    elif defect == "boot":
        ack["boot_id"] = "other-boot"
    elif defect == "pid":
        ack["pid"] += 1
    elif defect == "invocation":
        ack["invocation_id"] = "f" * 32
    elif defect in {"dead", "job"}:
        unit = host.units[host.window.units["restore-service"]]
        unit.update({"ActiveState": "inactive"} if defect == "dead" else {"Job": "5"})
    elif defect == "expiry":
        host.now = host.binding["restore_at"] - 1
    else:
        ack["elapsed"] = "nan"
    host.save(record)
    before = host.mutations()
    with pytest.raises(probe.ProbeError, match="ACK"):
        host.window.schedule_stop()
    assert host.mutations() == before


@pytest.mark.parametrize(
    "defect",
    [
        "fragment",
        "dropin",
        "transient",
        "reload",
        "description",
        "exec",
        "kill",
        "environment",
        "user",
    ],
)
def test_owned_service_manager_drift_blocks_mutation(host: Host, defect: str) -> None:
    host.arm()
    name = host.window.units["restore-service"]
    key, value = {
        "fragment": ("FragmentPath", "/foreign"),
        "dropin": ("DropInPaths", "/foreign/dropin"),
        "transient": ("Transient", "yes"),
        "reload": ("NeedDaemonReload", "yes"),
        "description": ("Description", "foreign"),
        "exec": ("ExecStart", command_property("/bin/false")),
        "kill": ("KillMode", "process"),
        "environment": ("Environment", "FOREIGN=yes"),
        "user": ("User", "foreign"),
    }[defect]
    host.overrides[name] = {key: value}
    before = host.mutations()
    with pytest.raises(probe.ProbeError, match="ownership|definition"):
        host.window.schedule_stop()
    assert host.mutations() == before


def test_target_timeout_override_is_required_before_any_stop(host: Host) -> None:
    host.arm()
    host.overrides[host.binding["service"]] = {"TimeoutStopUSec": "infinity"}
    before = host.mutations()
    with pytest.raises(probe.ProbeError, match="bounded"):
        host.window.schedule_stop()
    assert host.mutations() == before


@pytest.mark.parametrize(
    "defect", ["early", "foreign-invocation", "not-scheduled", "no-record"]
)
def test_stop_helper_rejects_unproven_or_early_execution(
    host: Host, defect: str
) -> None:
    if defect != "no-record":
        host.arm()
        if defect != "not-scheduled":
            host.window.schedule_stop()
        host.become("stop-service")
    if defect == "foreign-invocation":
        host.sleep(15)
        host.tick()
        host.become("stop-service")
        host.units[host.binding["service"]]["InvocationID"] = "f" * 32
    before = host.mutations()
    with pytest.raises(probe.ProbeError):
        host.window.stop_owned()
    assert host.mutations() == before


@pytest.mark.parametrize("method", ["restore", "status", "tick"])
def test_missing_host_journal_is_not_generic_recovery_authority(
    host: Host, method: str
) -> None:
    before = host.mutations()
    with pytest.raises(probe.ProbeError, match="journal|record"):
        getattr(host.window, method)()
    assert host.mutations() == before


@pytest.mark.parametrize(
    "state", ["new-job", "control-process", "unknown", "failed-start"]
)
def test_restore_does_not_accept_unknown_target_progress(
    host: Host, state: str
) -> None:
    host.stopped()
    service = host.binding["service"]
    if state == "control-process":
        host.units[service]["ControlPID"] = "100"
    elif state == "unknown":
        host.units[service]["ActiveState"] = "unknown"
    elif state == "new-job":
        host.pending_start = True
        host.overrides[service] = {"Job": "unknown"}
    else:
        host.fail.add(("start", "--no-block", "--job-mode=fail", service))
    with pytest.raises(probe.ProbeError):
        host.window.restore()
    assert host.window.targets["restore-service"].exists(), host.window.targets
    assert host.read()["phase"] != "RESTORED"


def test_pending_owned_start_waits_and_does_not_submit_twice(host: Host) -> None:
    host.stopped()
    host.pending_start = True
    host.complete_on_sleep = host.binding["service"]
    assert host.window.restore()["phase"] == "RESTORED"
    assert host.window.cleanup()["phase"] == "CLOSED"
    assert (
        host.mutations().count(
            ("start", "--no-block", "--job-mode=fail", host.binding["service"])
        )
        == 1
    )


def test_independent_watch_is_bounded_and_handles_only_lock_contention(
    host: Host, monkeypatch: pytest.MonkeyPatch
) -> None:
    host.stopped()
    host.become("restore-service")
    tick = host.window.tick
    count = 0

    def contended() -> dict[str, Any]:
        nonlocal count
        count += 1
        if count == 1:
            raise BlockingIOError("owned lock is busy")
        host.now = host.binding["restore_at"]
        host.elapsed = host.read()["clock"]["restore"]
        return tick()

    monkeypatch.setattr(host.window, "tick", contended)
    assert host.window.watch()["phase"] == "RESTORED"
    assert count == 2
    monkeypatch.setattr(host.window, "tick", lambda: {"phase": "ARMED"})
    with pytest.raises(probe.ProbeError, match="supervision exceeded"):
        host.window.watch()


@pytest.mark.parametrize("field", ["restore", "expires"])
@pytest.mark.parametrize("value", [None, True, "unknown"])
def test_elapsed_clock_record_is_strict(host: Host, field: str, value: Any) -> None:
    host.arm()
    record = host.read()
    record["clock"][field] = value
    host.save(record)
    with pytest.raises(probe.ProbeError, match="elapsed deadline"):
        host.window.restore()


@pytest.mark.parametrize("entry", ["source", "target", "identity", "unknown"])
def test_resource_receipt_shape_cannot_expand_cleanup(host: Host, entry: str) -> None:
    host.arm()
    record = host.read()
    resource = record["resources"]["stop-service"]
    resource[entry] = "/foreign" if entry != "identity" else []
    host.save(record)
    before = host.mutations()
    with pytest.raises(probe.ProbeError, match="receipt"):
        host.window.cleanup()
    assert host.mutations() == before


@pytest.mark.parametrize("mode", [0o644, 0o666])
def test_private_journals_reject_world_access(host: Host, mode: int) -> None:
    host.arm()
    host.window.state_path.chmod(mode)
    with pytest.raises(probe.ProbeError, match="not private"):
        host.window.cleanup()


def test_hardlinked_journal_and_missing_journal_with_residual_source_refuse(
    host: Host,
) -> None:
    host.arm()
    link = host.window.directory / "untrusted-alias"
    os.link(host.window.state_path, link)
    with pytest.raises(probe.ProbeError, match="not private"):
        host.window.cleanup()
    link.unlink()
    host.window.state_path.unlink()
    with pytest.raises(probe.ProbeError, match="missing beside"):
        host.window.cleanup()


@pytest.mark.parametrize("defect", ["mode", "owner", "kind", "links"])
def test_controller_independent_host_lock_metadata_is_checked(
    host: Host, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    values = {"st_mode": stat.S_IFREG | 0o600, "st_uid": os.geteuid(), "st_nlink": 1}
    if defect == "mode":
        values["st_mode"] |= 0o004
    elif defect == "owner":
        values["st_uid"] += 1
    elif defect == "kind":
        values["st_mode"] = stat.S_IFIFO | 0o600
    else:
        values["st_nlink"] = 2
    # The fake replaces the global os.fstat, which pytest's own tmp_path
    # teardown also uses; keep it to the call under test.
    with monkeypatch.context() as patch:
        patch.setattr(os, "fstat", lambda _fd: SimpleNamespace(**values))
        with pytest.raises(probe.ProbeError, match="lock is not private"):
            host.window.prepare()


@pytest.mark.parametrize("command", ["stop-with-failsafe", "restore-service"])
def test_legacy_names_never_authorize_a_service_action(
    host: Host, command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    before = list(host.calls)
    assert (
        probe.main([command, "--service", "kubelet.service", "--run-id", "old-run"])
        == 1
    )
    assert json.loads(capsys.readouterr().out)["recovery_required"] is True
    assert host.calls == before


@pytest.mark.parametrize(
    "command",
    [
        "prepare",
        "status",
        "stop-with-failsafe",
        "restore-service",
        "cleanup",
        "watch",
        "stop-owned",
    ],
)
def test_probe_entrypoint_uses_the_same_bound_state_machine(
    host: Host, command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    if command not in {"prepare", "cleanup"}:
        host.arm()
    if command == "stop-owned":
        host.window.schedule_stop()
        host.sleep(15)
        host.tick()
        host.become("stop-service")
    elif command == "watch":
        host.become("restore-service")
        host.now = host.binding["restore_at"]
        host.elapsed = host.read()["clock"]["restore"]
    argument = (
        ["--key", host.window.key]
        if command in {"watch", "stop-owned"}
        else ["--binding", json.dumps(host.binding)]
    )
    assert probe.main([command, *argument]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["binding_sha256"] == host.window.key
    assert report["phase"] in {
        "INSTALLED",
        "ARMED",
        "SCHEDULED",
        "STOPPING",
        "RESTORED",
        "CLOSED",
    }


@pytest.mark.parametrize(
    "args",
    [
        ["watch", "--key", "../foreign"],
        ["watch", "--binding", "{}"],
        ["watch"],
        ["cleanup", "--key", "e" * 64],
        ["cleanup", "--binding", "{}"],
        ["snapshot", "--restore-seconds", "59"],
        ["snapshot", "--stop-delay-seconds", "121"],
    ],
)
def test_invalid_probe_requests_do_not_reach_host_commands(
    host: Host, args: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    before = list(host.calls)
    assert probe.main(args) == 1
    assert json.loads(capsys.readouterr().out)["error"]
    assert host.calls == before


@pytest.mark.parametrize(
    ("output", "returncode"),
    [("unknown", 0), ("Job=0\nJob=1\n", 0), ("LoadState=loaded\n", 4), ("", 1)],
)
def test_systemctl_output_errors_are_fail_closed_and_withheld(
    host: Host,
    monkeypatch: pytest.MonkeyPatch,
    output: str,
    returncode: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess(
            [], returncode, output, "private diagnostic not for output"
        ),
    )
    with pytest.raises(probe.ProbeError):
        probe.systemctl("show", "kubelet.service")
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("missing", sorted(probe.UNIT_FIELDS | probe.SERVICE_FIELDS))
def test_only_identity_fields_are_required_others_default_empty(
    host: Host, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    # systemd 252 (Amazon Linux 2023, live 2026-09-19) prints no line for an empty
    # property, so a complete ``systemctl show`` legitimately omits any non-identity
    # field. ``unit_state`` therefore requires only the always-present identity
    # fields and reads every omitted property as empty.
    state = dict(host.units["kubelet.service"])
    state.pop(missing)
    monkeypatch.setattr(probe, "systemctl", lambda *_args: state)
    if missing in probe.IDENTITY_FIELDS:
        with pytest.raises(probe.ProbeError, match="properties are incomplete"):
            probe.service_snapshot("kubelet.service")
    else:
        snapshot = probe.service_snapshot("kubelet.service")
        if missing in probe.STATE_FIELDS:
            assert snapshot[missing] == ""


@pytest.mark.parametrize(
    "value", ["", "0", "0 /", "2", "2 /org/freedesktop/systemd1/job/2"]
)
def test_systemd_job_identity_is_parsed_exactly(value: str) -> None:
    assert probe.job_id({"Job": value}) == (2 if value.startswith("2") else 0)


@pytest.mark.parametrize(
    "value", ["unknown", "-1", "2 /org/freedesktop/systemd1/job/3"]
)
def test_ambiguous_job_identity_is_not_idle(value: str) -> None:
    with pytest.raises(probe.ProbeError, match="job identity"):
        probe.job_id({"Job": value})


@pytest.mark.parametrize(
    "body", ["", "populated unknown\n", "populated 0\npopulated 1\n", "populated\n"]
)
def test_missing_or_malformed_cgroup_evidence_is_not_quiescence(
    host: Host, body: str
) -> None:
    host.arm()
    unit = copy.deepcopy(host.units[host.window.units["stop-service"]])
    unit["ControlGroup"] = f"/system.slice/{unit['Id']}"
    write(probe.CGROUP_ROOT / "system.slice" / unit["Id"] / "cgroup.events", body)
    with pytest.raises(probe.ProbeError, match="cgroup"):
        probe.idle(unit)
