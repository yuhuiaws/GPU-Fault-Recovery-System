"""Owned collector-env recovery using private fixtures and a fake service manager."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast

import pytest

from scripts.e2e.regional.probes import collector_node_probe as probe

OWNER_NONCE = "a" * 32
ENV_TEXT = (
    "GPU_FAULT_EXPECTED_GPU_COUNT=8\n"
    "GPU_FAULT_HOST_INTERVAL_SECONDS=15\n"
    "GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES=2\n"
    "GPU_FAULT_CLUSTER_ID=cluster-fixture\n"
    "NODE_NAME=node-fixture\n"
    "GPU_FAULT_NODE_INSTANCE_ID=instance-fixture\n"
    "PRIVATE_FIXTURE_SENTINEL=never-return-this-content\n"
)


def forbidden(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("real host, process, or network operation is forbidden")


def execution(command: list[str]) -> str:
    return (
        f"{{ path={command[0]} ; argv[]={shlex.join(command)} ; ignore_errors=no ; }}"
    )


class EnvHost:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.run_id = "collect004-campaign-fixture-a1"
        self.now = 1000.0
        self.epoch = 1_789_603_200.0
        self.calls: list[list[str]] = []
        self.emitted: list[dict[str, Any]] = []
        self.callback_receipts: list[dict[str, Any]] = []
        self.loaded: dict[str, bytes] = {}
        self.dropin: bytes | None = None
        self.enabled: set[str] = set()
        self.active: set[str] = {probe.HOST_COLLECTOR_UNIT}
        self.invocations: dict[str, str] = {probe.HOST_COLLECTOR_UNIT: "host-0"}
        self.serial = 0
        self.pending_restart = False
        self.execute_recovery = True
        self.acknowledge_start = True
        self.overrides: dict[str, dict[str, str]] = {}
        self.failures: dict[tuple[str, ...], list[BaseException]] = {}
        self.before_call: Callable[[list[str]], None] | None = None
        runtime = probe.COLLECTOR_RUNTIME.resolve()
        device = runtime.stat().st_dev
        self.filesystem = {
            "uuid": "b953642a-7329-4d11-b89e-d7a6d9a3ae2f",
            "fstype": "ext4",
            "target": str(root),
            "fsroot": "/",
            "maj:min": f"{os.major(device)}:{os.minor(device)}",
        }
        self.findmnt_calls: list[list[str]] = []
        self.write_mountinfo()
        self.reload()

    def write_mountinfo(self) -> None:
        fields = self.filesystem

        def escape(value: str) -> str:
            return value.replace("\\", r"\134").replace(" ", r"\040")

        path = probe.PROC_ROOT / "self/mountinfo"
        if path.exists():
            path.chmod(0o600)
        path.write_text(
            f"7 1 {fields['maj:min']} {escape(fields['fsroot'])} "
            f"{escape(fields['target'])} rw - {fields['fstype']} /dev/fixture rw\n"
        )
        path.chmod(0o444)

    def findmnt(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        assert command[0].startswith("/proc/self/fd/"), (
            'findmnt: expected command[0].startswith("/proc/self/fd/")'
        )
        assert Path(command[0]).resolve() == probe.COLLECTOR_FINDMNT
        assert kwargs["pass_fds"] == (int(command[0].rsplit("/", 1)[1]),)
        assert kwargs["env"] == {"LC_ALL": "C", "PATH": "/usr/bin:/bin"}
        self.findmnt_calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, json.dumps({"filesystems": [self.filesystem]}), ""
        )

    def arguments(self, **changes: Any) -> argparse.Namespace:
        return argparse.Namespace(
            **{
                "run_id": self.run_id,
                "owner_nonce": OWNER_NONCE,
                "owner_nonce_file": "",
                "expected_env_sha256": hashlib.sha256(ENV_TEXT.encode()).hexdigest(),
                "expected_boot_id": "boot-fixture-a",
                "cluster_id": "cluster-fixture",
                "node_id": "node-fixture",
                "value": 9,
                "restore_seconds": 600,
                "automatic": False,
                **changes,
            }
        )

    def override(self, **changes: Any) -> dict[str, Any]:
        probe.override_expected_gpu_count(self.arguments(**changes))
        return self.emitted[-1]

    def restore(self, **changes: Any) -> dict[str, Any]:
        probe.restore_collector_env(self.arguments(**changes))
        return self.emitted[-1]

    def record(self, run_id: str | None = None) -> dict[str, Any]:
        backup, _ = probe.collector_restore_paths(run_id or self.arguments().run_id)
        return cast(
            dict[str, Any],
            json.loads(probe.collector_override_record(backup).read_text()),
        )

    def paths(self) -> tuple[Path, str]:
        return probe.collector_restore_paths(self.arguments().run_id)

    def reload(self) -> None:
        self.loaded = {
            path.name: path.read_bytes()
            for path in probe.SYSTEMD_UNIT_DIR.iterdir()
            if path.suffix in {".service", ".timer"}
        }
        dropin = (
            probe.SYSTEMD_UNIT_DIR
            / f"{probe.HOST_COLLECTOR_UNIT}.d"
            / "90-gpu-fault-collector-env.conf"
        )
        self.dropin = dropin.read_bytes() if dropin.exists() else None

    def state(self, name: str) -> dict[str, str]:
        state = {
            "Id": name,
            "LoadState": "not-found",
            "ActiveState": "inactive",
            "SubState": "dead",
            "MainPID": "0",
            "ControlPID": "0",
            "ControlGroup": "",
            "Job": "",
            "FragmentPath": "",
            "DropInPaths": "",
            "NeedDaemonReload": "no",
            "UnitFileState": "disabled",
            "InvocationID": self.invocations.get(name, ""),
            "Result": "success",
            "ExecMainStatus": "0",
            "Before": "",
            "After": "",
            "Requires": "",
            "ExecStartPre": "",
        }
        raw = self.loaded.get(name)
        if raw is None:
            state.update(self.overrides.get(name, {}))
            return state
        state["LoadState"] = "loaded"
        state["FragmentPath"] = str(probe.SYSTEMD_UNIT_DIR / name)
        if name in self.enabled:
            state["UnitFileState"] = "enabled"
        if name in self.active:
            state["ActiveState"] = "active"
            state["SubState"] = "waiting" if name.endswith(".timer") else "running"
            if name.endswith(".service"):
                state["MainPID"] = "1234"
        config = configparser.ConfigParser(interpolation=None)
        config.read_string(raw.decode())
        if name == probe.HOST_COLLECTOR_UNIT and self.dropin is not None:
            config.read_string(self.dropin.decode())
            state["DropInPaths"] = str(
                probe.SYSTEMD_UNIT_DIR / f"{name}.d" / "90-gpu-fault-collector-env.conf"
            )
        if "Unit" in config:
            for key in ("Before", "After", "Requires"):
                state[key] = config["Unit"].get(key, "")
        if "Service" in config:
            service = config["Service"]
            for key in (
                "Type",
                "Restart",
                "RestartPreventExitStatus",
                "KillMode",
                "SendSIGKILL",
            ):
                if key in service:
                    state[key] = service[key]
            for key in ("Restart", "TimeoutStart", "TimeoutStop"):
                if key + "Sec" in service:
                    state[key + "USec"] = service[key + "Sec"]
            for key in ("ExecStart", "ExecStartPre"):
                if key in service:
                    state[key] = execution(shlex.split(service[key]))
        if "Timer" in config:
            timer = config["Timer"]
            state.update(
                {
                    "Unit": timer["Unit"],
                    "NextElapseUSecMonotonic": timer["OnBootSec"],
                    "AccuracyUSec": timer["AccuracySec"],
                    "RandomizedDelayUSec": timer["RandomizedDelaySec"],
                }
            )
        state.update(self.overrides.get(name, {}))
        return state

    def recovery_start(self, name: str) -> None:
        if self.execute_recovery:
            config = configparser.ConfigParser(interpolation=None)
            config.read_string(self.loaded[name].decode())
            command = shlex.split(config["Service"]["ExecStart"])
            args = probe.parser().parse_args(command[5:])
            previous = len(self.emitted)
            try:
                args.handler(args)
            finally:
                self.callback_receipts.extend(self.emitted[previous:])
                del self.emitted[previous:]
        if self.acknowledge_start:
            self.serial += 1
            self.invocations[name] = f"recovery-{self.serial}"
        self.active.discard(name)

    def start_host(self) -> None:
        if self.dropin is not None:
            _, unit = self.paths()
            self.recovery_start(f"{unit}.service")
            config = configparser.ConfigParser(interpolation=None)
            config.read_string(self.dropin.decode())
            command = shlex.split(config["Service"]["ExecStartPre"])
            args = probe.parser().parse_args(command[5:])
            previous = len(self.emitted)
            try:
                args.handler(args)
            finally:
                self.callback_receipts.extend(self.emitted[previous:])
                del self.emitted[previous:]
        self.serial += 1
        self.active.add(probe.HOST_COLLECTOR_UNIT)
        self.invocations[probe.HOST_COLLECTOR_UNIT] = f"host-{self.serial}"
        self.pending_restart = False

    def run(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(command)
        if self.before_call is not None:
            self.before_call(command)
        for prefix, failures in self.failures.items():
            if tuple(command[: len(prefix)]) == prefix and failures:
                raise failures.pop(0)
        assert command[0] == "systemctl", command
        output = ""
        if command[1] == "show":
            output = "\n".join(
                f"{key}={value}" for key, value in self.state(command[2]).items()
            )
        elif command[1] == "daemon-reload":
            self.reload()
        elif command[1] in {"enable", "disable", "start", "stop"}:
            name = command[-1]
            if command[1] == "enable":
                self.enabled.add(name)
                target = (
                    "timers.target" if name.endswith(".timer") else "multi-user.target"
                )
                directory = probe.SYSTEMD_UNIT_DIR / f"{target}.wants"
                directory.mkdir(exist_ok=True)
                link = directory / name
                if not link.exists():
                    link.symlink_to(probe.SYSTEMD_UNIT_DIR / name)
                if "--now" in command:
                    self.active.add(name)
            elif command[1] == "disable":
                self.enabled.discard(name)
                target = (
                    "timers.target" if name.endswith(".timer") else "multi-user.target"
                )
                (probe.SYSTEMD_UNIT_DIR / f"{target}.wants" / name).unlink(
                    missing_ok=True
                )
                if "--now" in command:
                    self.active.discard(name)
            elif command[1] == "stop":
                self.active.discard(name)
            else:
                self.recovery_start(name)
        elif command[1:3] == ["--no-block", "restart"]:
            self.pending_restart = True
        elif command[1:] == ["restart", probe.HOST_COLLECTOR_UNIT]:
            self.start_host()
        else:
            raise AssertionError(f"unexpected fake service-manager call: {command}")
        return subprocess.CompletedProcess(command, 0, output, "")

    def mutations(self) -> list[list[str]]:
        return [call for call in self.calls if call[1] != "show"]


@pytest.fixture
def env_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> EnvHost:
    for name in ("run", "Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, forbidden)
    for name in ("socket", "create_connection", "getaddrinfo"):
        monkeypatch.setattr(socket, name, forbidden)
    for name in ("system", "kill", "fork", "execv", "execve"):
        monkeypatch.setattr(os, name, forbidden)
    for name, relative in {
        "COLLECTOR_ENV": "etc/collector.env",
        "ACCEPTANCE_STATE": "state",
        "SYSTEMD_UNIT_DIR": "systemd",
        "BOOT_ID_FILE": "boot-id",
        "COLLECTOR_RUNTIME": "current",
        "POWER_CGROUP_ROOT": "cgroup",
        "POWER_TRANSIENT_UNIT_DIR": "transient",
        "COLLECTOR_FINDMNT": "bin/findmnt",
        "PROC_ROOT": "proc",
    }.items():
        monkeypatch.setattr(probe, name, tmp_path / relative)
    probe.COLLECTOR_ENV.parent.mkdir()
    probe.COLLECTOR_ENV.write_text(ENV_TEXT)
    probe.COLLECTOR_ENV.chmod(0o600)
    probe.BOOT_ID_FILE.write_text("boot-fixture-a")
    probe.POWER_CGROUP_ROOT.mkdir()
    (probe.POWER_CGROUP_ROOT / "cgroup.controllers").write_text("cpu memory pids\n")
    probe.SYSTEMD_UNIT_DIR.mkdir()
    (probe.SYSTEMD_UNIT_DIR / probe.HOST_COLLECTOR_UNIT).write_text(
        "[Unit]\nAfter=network-online.target\n"
        "[Service]\nType=notify\nExecStart=/fixture/collector host\n"
    )
    (probe.SYSTEMD_UNIT_DIR / probe.HOST_COLLECTOR_UNIT).chmod(0o644)
    runtime = tmp_path / "release-fixture"
    (runtime / "venv/bin").mkdir(parents=True)
    runtime.chmod(0o755)
    python = runtime / "venv/bin/python"
    python.write_text("inert Python executable fixture\n")
    python.chmod(0o700)
    probe.COLLECTOR_RUNTIME.symlink_to(runtime, target_is_directory=True)
    probe.COLLECTOR_FINDMNT.parent.mkdir()
    probe.COLLECTOR_FINDMNT.parent.chmod(0o755)
    probe.COLLECTOR_FINDMNT.write_text("inert findmnt executable fixture\n")
    probe.COLLECTOR_FINDMNT.chmod(0o700)
    (probe.PROC_ROOT / "self").mkdir(parents=True)
    monkeypatch.setattr(probe, "sys", SimpleNamespace(executable=str(python)))
    # The only replaced privilege check: all files are private to the test UID.
    monkeypatch.setattr(probe, "collector_require_root", lambda: None)
    host = EnvHost(tmp_path)
    monkeypatch.setattr(probe, "run", host.run)
    monkeypatch.setattr(subprocess, "run", host.findmnt)
    monkeypatch.setattr(probe, "emit", host.emitted.append)
    monkeypatch.setattr(probe, "time", SimpleNamespace(monotonic=lambda: host.now))
    monkeypatch.setattr(
        probe,
        "datetime",
        SimpleNamespace(
            now=lambda tz: datetime.fromtimestamp(host.epoch + host.now, timezone.utc),
            fromisoformat=datetime.fromisoformat,
        ),
    )
    return host


def test_owned_env_cycle_has_bound_armed_and_cleaned_receipts(
    env_host: EnvHost,
) -> None:
    ack = env_host.override()
    record = env_host.record()
    assert record["schema_version"] == 3
    assert record["runtime_filesystem"] == {
        key: value for key, value in env_host.filesystem.items() if key != "maj:min"
    }
    backup, unit = env_host.paths()
    assert ack["mutation_started"] and ack["timer_armed"] and ack["boot_restore_armed"]
    assert ack["baseline_sha256"] == hashlib.sha256(ENV_TEXT.encode()).hexdigest()
    assert (
        ack["applied_sha256"]
        == hashlib.sha256(probe.COLLECTOR_ENV.read_bytes()).hexdigest()
    )
    assert ack["intent_sha256"] == record["intent_sha256"]
    assert record["state"] == "ACTIVE"
    assert backup.read_text() == ENV_TEXT
    assert backup.stat().st_mode & 0o777 == 0o600
    assert env_host.callback_receipts[0]["mutation_started"] is False
    assert probe.parse_env()["GPU_FAULT_EXPECTED_GPU_COUNT"] == "9"
    restored = env_host.restore()
    assert restored["state"] == "CLEANED" and restored["restored"]
    assert (
        restored["cleanup_verified"]
        and restored["timer_disarmed"]
        and restored["recovery_stopped"]
    )
    assert restored["intent_sha256"] == ack["intent_sha256"]
    assert probe.COLLECTOR_ENV.read_text() == ENV_TEXT
    assert not backup.exists(), (
        "test_owned_env_cycle_has_bound_armed_and_cleaned_receipts: expected no backup.exists()"
    )
    assert not (probe.SYSTEMD_UNIT_DIR / f"{unit}.service").exists(), (
        'test_owned_env_cycle_has_bound_armed_and_cleaned_receipts: expected no (probe.SYSTEMD_UNIT_DIR / f"{unit}.service").exists()'
    )
    assert not (probe.ACCEPTANCE_STATE / "collector-env-owner.json").exists(), (
        'test_owned_env_cycle_has_bound_armed_and_cleaned_receipts: expected no (probe.ACCEPTANCE_STATE / "collector-env-owner.json").exists()'
    )
    assert env_host.record()["state"] == "CLEANED"
    assert "PRIVATE_FIXTURE_SENTINEL" not in json.dumps(env_host.emitted)
    assert OWNER_NONCE not in json.dumps(env_host.emitted)
    assert OWNER_NONCE not in json.dumps(env_host.calls)


@pytest.mark.parametrize(
    "field,value",
    [
        ("owner_nonce", ""),
        ("owner_nonce", "A" * 32),
        ("owner_nonce", "a" * 31),
        ("expected_env_sha256", "f" * 64),
        ("expected_boot_id", "wrong-boot"),
        ("cluster_id", "wrong-cluster"),
        ("node_id", "wrong-node"),
        ("value", 10),
        ("restore_seconds", 0),
        ("restore_seconds", 901),
        ("restore_seconds", True),
    ],
)
def test_invalid_admission_has_no_env_or_service_mutation(
    env_host: EnvHost, field: str, value: Any
) -> None:
    with pytest.raises(probe.ProbeError):
        env_host.override(**{field: value})
    assert probe.COLLECTOR_ENV.read_text() == ENV_TEXT
    assert env_host.mutations() == []
    assert not (probe.ACCEPTANCE_STATE / "collector-env-owner.json").exists(), (
        'test_invalid_admission_has_no_env_or_service_mutation: expected no (probe.ACCEPTANCE_STATE / "collector-env-owner.json").exists()'
    )


@pytest.mark.parametrize("same_id", [True, False])
def test_foreign_run_or_nonce_cannot_restore_or_disarm(
    env_host: EnvHost, same_id: bool
) -> None:
    env_host.override()
    before = probe.COLLECTOR_ENV.read_bytes()
    backup, _ = env_host.paths()
    receipt = probe.collector_override_record(backup).read_bytes()
    changes = {"owner_nonce": "b" * 32}
    if not same_id:
        changes["run_id"] = "another-campaign-a1"
    env_host.calls.clear()
    for action in (env_host.override, env_host.restore):
        with pytest.raises(probe.ProbeError):
            action(**changes)
    assert env_host.mutations() == []
    assert probe.COLLECTOR_ENV.read_bytes() == before
    assert probe.collector_override_record(backup).read_bytes() == receipt


def test_unknown_run_restore_never_reads_live_environment(
    env_host: EnvHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = probe.collector_private_file

    def read(path: Path, **kwargs: Any) -> bytes:
        assert path != probe.COLLECTOR_ENV
        return original(path, **kwargs)

    monkeypatch.setattr(probe, "collector_private_file", read)
    result = env_host.restore()
    assert result["state"] == "NOT_STARTED" and result["no_mutation"]
    assert result["cleanup_verified"] and not result["restored"]
    assert env_host.calls == []


@pytest.mark.parametrize(
    "drift", ["extra-key", "node", "cluster", "instance", "runtime", "probe", "backup"]
)
def test_restore_rejects_identity_or_digest_drift(
    env_host: EnvHost, drift: str
) -> None:
    env_host.override()
    backup, _ = env_host.paths()
    if drift in {"extra-key", "node", "cluster", "instance"}:
        text = probe.COLLECTOR_ENV.read_text()
        if drift == "extra-key":
            text += "UNRELATED_FIXTURE_CHANGE=preserve\n"
        else:
            text = text.replace(f"{drift}-fixture", f"{drift}-changed")
        probe.COLLECTOR_ENV.write_text(text)
    elif drift == "runtime":
        probe.COLLECTOR_RUNTIME.unlink()
        probe.COLLECTOR_RUNTIME.symlink_to(env_host.root / "missing-runtime")
    else:
        changed = backup if drift == "backup" else backup.with_suffix(".probe.py")
        changed.chmod(0o600)
        changed.write_text("changed")
    before = probe.COLLECTOR_ENV.read_bytes()
    env_host.calls.clear()
    with pytest.raises((probe.ProbeError, OSError)):
        env_host.restore()
    assert probe.COLLECTOR_ENV.read_bytes() == before
    assert backup.exists(), (
        "test_restore_rejects_identity_or_digest_drift: expected backup.exists()"
    )
    assert env_host.mutations() == []


@pytest.mark.parametrize(
    "property_name,value",
    [
        ("DropInPaths", "/foreign/drop-in.conf"),
        ("ExecStart", execution(["/bin/false"])),
        ("UnitFileState", "disabled"),
        ("TimeoutStartUSec", "infinity"),
        ("RestartPreventExitStatus", ""),
        ("KillMode", "process"),
    ],
)
def test_recovery_service_drift_blocks_override(
    env_host: EnvHost, property_name: str, value: str
) -> None:
    _, unit = env_host.paths()
    env_host.overrides[f"{unit}.service"] = {property_name: value}
    with pytest.raises(probe.ProbeError):
        env_host.override()
    assert probe.COLLECTOR_ENV.read_text() == ENV_TEXT
    assert not env_host.record()["mutation_started"]
    assert not any(
        call[-1] == probe.HOST_COLLECTOR_UNIT for call in env_host.mutations()
    ), (
        "test_recovery_service_drift_blocks_override: expected no any( call[-1] == probe.HOST_COLLECTOR_UNIT for call in env_host.mutations() )"
    )


def test_missing_independent_armed_execution_blocks_write(env_host: EnvHost) -> None:
    env_host.acknowledge_start = False
    with pytest.raises(probe.ProbeError, match="acknowledge ARMED"):
        env_host.override()
    assert probe.COLLECTOR_ENV.read_text() == ENV_TEXT
    assert not env_host.record()["mutation_started"]


@pytest.mark.parametrize("failure", ["daemon-reload", "enable", "start"])
def test_partial_preparation_can_clean_only_its_owned_materials(
    env_host: EnvHost, failure: str
) -> None:
    env_host.failures[("systemctl", failure)] = [
        RuntimeError("fixture preparation failure")
    ]
    with pytest.raises(RuntimeError, match="preparation"):
        env_host.override()
    assert probe.COLLECTOR_ENV.read_text() == ENV_TEXT
    assert not env_host.record()["mutation_started"]
    env_host.calls.clear()
    result = env_host.restore()
    assert result["state"] == "CLEANED" and result["no_mutation"]
    assert result["cleanup_verified"]
    assert not any("restart" in call for call in env_host.calls), (
        'test_partial_preparation_can_clean_only_its_owned_materials: expected no any("restart" in call for call in env_host.calls)'
    )


def test_mutation_intent_survives_restart_failure(env_host: EnvHost) -> None:
    env_host.failures[("systemctl", "restart")] = [
        RuntimeError("fixture restart failure")
    ]
    with pytest.raises(RuntimeError, match="restart failure"):
        env_host.override()
    assert env_host.record()["mutation_started"]
    assert env_host.record()["state"] == "MUTATING"
    assert probe.parse_env()["GPU_FAULT_EXPECTED_GPU_COUNT"] == "9"
    result = env_host.restore()
    assert result["cleanup_verified"] and probe.COLLECTOR_ENV.read_text() == ENV_TEXT


@pytest.mark.parametrize("trigger", ["elapsed", "boot"])
def test_automatic_restore_keeps_fixed_deadline_and_materials(
    env_host: EnvHost, trigger: str
) -> None:
    ack = env_host.override()
    if trigger == "boot":
        probe.BOOT_ID_FILE.write_text("boot-fixture-b")
        env_host.now = 1.0
    else:
        env_host.now += 601
    result = env_host.restore(automatic=True)
    assert result["restored"] and not result["cleanup_verified"]
    assert result["restore_deadline_epoch"] == ack["restore_deadline_epoch"]
    assert result["restore_deadline_monotonic"] == ack["restore_deadline_monotonic"]
    assert probe.COLLECTOR_ENV.read_text() == ENV_TEXT
    assert env_host.paths()[0].exists(), (
        "test_automatic_restore_keeps_fixed_deadline_and_materials: expected env_host.paths()[0].exists()"
    )
    assert env_host.pending_restart, (
        "test_automatic_restore_keeps_fixed_deadline_and_materials: expected env_host.pending_restart"
    )
    assert env_host.restore()["cleanup_verified"]


def test_early_automatic_callback_does_not_mutate(env_host: EnvHost) -> None:
    env_host.override()
    before = probe.COLLECTOR_ENV.read_bytes()
    env_host.calls.clear()
    result = env_host.restore(automatic=True)
    assert result["deferred"] and not result["restored"]
    assert probe.COLLECTOR_ENV.read_bytes() == before
    assert env_host.mutations() == []


@pytest.mark.parametrize("failure", ["record", "backup", "skipped-recovery"])
def test_new_boot_cannot_start_collector_without_restoration(
    env_host: EnvHost, failure: str
) -> None:
    env_host.override()
    backup, _ = env_host.paths()
    probe.BOOT_ID_FILE.write_text("boot-fixture-b")
    env_host.active.discard(probe.HOST_COLLECTOR_UNIT)
    if failure == "record":
        probe.collector_override_record(backup).write_text("{truncated")
    elif failure == "backup":
        backup.write_text("wrong backup")
    else:
        env_host.execute_recovery = False
    with pytest.raises(probe.ProbeError):
        env_host.start_host()
    assert probe.HOST_COLLECTOR_UNIT not in env_host.active
    assert probe.parse_env()["GPU_FAULT_EXPECTED_GPU_COUNT"] == "9"


def test_start_check_rejects_expired_original_boot_override(env_host: EnvHost) -> None:
    env_host.override()
    env_host.now += 601
    with pytest.raises(probe.ProbeError):
        probe.collector_env_start_check(env_host.arguments())


def test_cleanup_stops_recovery_service_before_removing_backup(
    env_host: EnvHost,
) -> None:
    env_host.override()
    backup, unit = env_host.paths()
    observed: list[bool] = []

    def before(command: list[str]) -> None:
        if (
            command[:3] == ["systemctl", "disable", "--now"]
            and command[-1] == f"{unit}.service"
        ):
            observed.append(backup.exists())
            assert env_host.record()["state"] == "CLEANING"
            assert env_host.state(probe.HOST_COLLECTOR_UNIT)["Requires"] == ""

    env_host.before_call = before
    assert env_host.restore()["cleanup_verified"]
    assert observed == [True]


@pytest.mark.parametrize(
    "property_name,value",
    [
        ("MainPID", "123"),
        ("ControlPID", "123"),
        ("Job", "55"),
        ("ControlGroup", "/foreign/cgroup"),
    ],
)
def test_cleanup_retains_materials_when_service_stop_is_unproven(
    env_host: EnvHost, property_name: str, value: str
) -> None:
    env_host.override()
    backup, unit = env_host.paths()
    env_host.overrides[f"{unit}.service"] = {property_name: value}
    with pytest.raises(probe.ProbeError):
        env_host.restore()
    assert backup.exists(), (
        "test_cleanup_retains_materials_when_service_stop_is_unproven: expected backup.exists()"
    )
    assert env_host.record()["state"] != "CLEANED"


def test_owned_lock_blocks_concurrent_restore_without_late_write(
    env_host: EnvHost,
) -> None:
    env_host.override()
    before = probe.COLLECTOR_ENV.read_bytes()
    env_host.now += 601
    env_host.calls.clear()
    with probe.collector_env_lock():
        with pytest.raises(probe.ProbeError, match="still running"):
            env_host.restore(automatic=True)
        assert probe.COLLECTOR_ENV.read_bytes() == before
    assert env_host.restore()["cleanup_verified"]
    env_host.calls.clear()
    assert env_host.restore(automatic=True)["restored"]
    assert env_host.mutations() == []


def test_cleaned_receipt_is_retry_safe_and_rejects_nonce_reuse(
    env_host: EnvHost,
) -> None:
    env_host.override()
    first = env_host.restore()
    env_host.calls.clear()
    retry = env_host.restore()
    assert retry == first
    assert env_host.mutations() == []
    with pytest.raises(probe.ProbeError, match="owner"):
        env_host.restore(owner_nonce="b" * 32)
    with pytest.raises(probe.ProbeError, match="already exists"):
        env_host.override()


@pytest.mark.parametrize("material", ["env", "backup", "record", "nonce"])
def test_private_files_reject_public_permissions(
    env_host: EnvHost, material: str
) -> None:
    if material == "env":
        path = probe.COLLECTOR_ENV
    else:
        env_host.override()
        backup, _ = env_host.paths()
        path = {
            "backup": backup,
            "record": probe.collector_override_record(backup),
            "nonce": backup.with_suffix(".nonce"),
        }[material]
    path.chmod(0o644)
    with pytest.raises(probe.ProbeError):
        if material == "env":
            env_host.override()
        elif material == "nonce":
            env_host.restore(owner_nonce="", owner_nonce_file=str(path))
        else:
            env_host.restore()


def test_atomic_replacement_rechecks_drift_before_rename(
    env_host: EnvHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = tempfile.NamedTemporaryFile

    def create(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        probe.COLLECTOR_ENV.write_text(ENV_TEXT + "LATER_FIXTURE_CHANGE=preserve\n")
        return result

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", create)
    with pytest.raises(probe.ProbeError, match="changed before"):
        probe.replace_collector_env(
            b"replacement",
            expected_sha256=hashlib.sha256(ENV_TEXT.encode()).hexdigest(),
        )
    assert "LATER_FIXTURE_CHANGE" in probe.COLLECTOR_ENV.read_text()
    assert not list(probe.COLLECTOR_ENV.parent.glob(".collector-acceptance-*")), (
        'test_atomic_replacement_rechecks_drift_before_rename: expected no list(probe.COLLECTOR_ENV.parent.glob(".collector-acceptance-*"))'
    )


def test_guard_main_errors_are_structured_without_sensitive_input(
    env_host: EnvHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = env_host.arguments()
    arguments.command = "override-expected-gpu-count"
    arguments.handler = probe.override_expected_gpu_count
    arguments.expected_env_sha256 = "f" * 64
    monkeypatch.setattr(
        probe, "parser", lambda: SimpleNamespace(parse_args=lambda: arguments)
    )
    assert probe.main() == 78
    result = env_host.emitted[-1]
    assert result["error_kind"] == "collector_env_guard"
    assert not result["cleanup_verified"]
    assert "PRIVATE_FIXTURE_SENTINEL" not in json.dumps(result)
    assert OWNER_NONCE not in json.dumps(result)


def test_busy_automatic_callback_retries_without_writing(
    env_host: EnvHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_host.override()
    env_host.now += 601
    arguments = env_host.arguments(automatic=True)
    arguments.command = "restore-collector-env"
    arguments.handler = probe.restore_collector_env
    monkeypatch.setattr(
        probe, "parser", lambda: SimpleNamespace(parse_args=lambda: arguments)
    )
    before = probe.COLLECTOR_ENV.read_bytes()
    with probe.collector_env_lock():
        assert probe.main() == 1
    assert env_host.emitted[-1]["retryable"] is True
    assert probe.COLLECTOR_ENV.read_bytes() == before
    assert env_host.record()["state"] == "ACTIVE"


@pytest.mark.parametrize(
    "state",
    [
        {"ActiveState": "failed", "SubState": "failed", "MainPID": "0"},
        {"ActiveState": "activating", "SubState": "auto-restart", "MainPID": "0"},
        {"ActiveState": "active", "SubState": "running", "Job": "13"},
        {"ActiveState": "active", "SubState": "running", "MainPID": "0"},
    ],
)
def test_automatic_restore_rejects_unknown_or_degraded_host_state(
    env_host: EnvHost, state: dict[str, str]
) -> None:
    env_host.override()
    probe.BOOT_ID_FILE.write_text("boot-fixture-b")
    env_host.overrides[probe.HOST_COLLECTOR_UNIT] = state
    env_host.calls.clear()
    with pytest.raises(probe.ProbeError, match="state is unknown"):
        env_host.restore(automatic=True)
    assert env_host.paths()[0].exists(), (
        "test_automatic_restore_rejects_unknown_or_degraded_host_state: expected env_host.paths()[0].exists()"
    )
    assert not any("restart" in call for call in env_host.calls), (
        'test_automatic_restore_rejects_unknown_or_degraded_host_state: expected no any("restart" in call for call in env_host.calls)'
    )


def test_boot_restore_preserves_the_original_pending_start(env_host: EnvHost) -> None:
    env_host.override()
    probe.BOOT_ID_FILE.write_text("boot-fixture-b")
    env_host.active.discard(probe.HOST_COLLECTOR_UNIT)
    env_host.overrides[probe.HOST_COLLECTOR_UNIT] = {"Job": "17"}
    env_host.calls.clear()
    result = env_host.restore(automatic=True)
    assert result["restored"] and not result["cleanup_verified"]
    assert not any("restart" in call for call in env_host.calls), (
        'test_boot_restore_preserves_the_original_pending_start: expected no any("restart" in call for call in env_host.calls)'
    )
    assert not env_host.pending_restart, (
        "test_boot_restore_preserves_the_original_pending_start: expected no env_host.pending_restart"
    )
    assert env_host.state(probe.HOST_COLLECTOR_UNIT)["Job"] == "17"


def test_automatic_restore_refuses_public_recovery_directory(env_host: EnvHost) -> None:
    env_host.override()
    probe.ACCEPTANCE_STATE.chmod(0o755)
    before = probe.COLLECTOR_ENV.read_bytes()
    with pytest.raises(probe.ProbeError, match="directory is not private"):
        env_host.restore(automatic=True)
    assert probe.COLLECTOR_ENV.read_bytes() == before


def test_nonce_material_drift_cannot_trigger_restore_or_restart(
    env_host: EnvHost,
) -> None:
    env_host.override()
    backup, _ = env_host.paths()
    backup.with_suffix(".nonce").write_text("b" * 32)
    before = probe.COLLECTOR_ENV.read_bytes()
    env_host.calls.clear()
    with pytest.raises(probe.ProbeError, match="material identity"):
        env_host.restore()
    assert probe.COLLECTOR_ENV.read_bytes() == before
    assert env_host.mutations() == []


def test_missing_boot_enablement_is_rejected_before_mutation(env_host: EnvHost) -> None:
    _, unit = env_host.paths()

    def before(command: list[str]) -> None:
        if command == ["systemctl", "start", f"{unit}.service"]:
            (
                probe.SYSTEMD_UNIT_DIR / "multi-user.target.wants" / f"{unit}.service"
            ).unlink()

    env_host.before_call = before
    with pytest.raises(probe.ProbeError, match="boot enablement"):
        env_host.override()
    assert probe.COLLECTOR_ENV.read_text() == ENV_TEXT
    assert not env_host.record()["mutation_started"]


def test_missing_cgroup_stop_proof_rejects_before_mutation(env_host: EnvHost) -> None:
    (probe.POWER_CGROUP_ROOT / "cgroup.controllers").unlink()
    with pytest.raises(probe.ProbeError, match="requires cgroup v2"):
        env_host.override()
    assert probe.COLLECTOR_ENV.read_text() == ENV_TEXT
    assert env_host.mutations() == []


def restore_main(
    host: EnvHost,
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_id: str | None = None,
    owner_nonce: str = OWNER_NONCE,
    automatic: bool = False,
) -> int:
    argv = [
        "probe.py",
        "restore-collector-env",
        "--run-id",
        run_id or host.run_id,
        "--owner-nonce",
        owner_nonce,
    ]
    if automatic:
        argv.append("--automatic")
    monkeypatch.setattr(sys, "argv", argv)
    return probe.main()


@pytest.mark.parametrize(
    "health",
    [{"Job": "19"}, {"ActiveState": "activating", "SubState": "start", "MainPID": "0"}],
    ids=["pending-job", "not-active"],
)
def test_first_post_restart_guard_survives_later_cleaned_receipt(
    env_host: EnvHost, monkeypatch: pytest.MonkeyPatch, health: dict[str, str]
) -> None:
    env_host.override()
    backup, unit = env_host.paths()
    intent = env_host.record()["intent_sha256"]
    env_host.calls.clear()

    def pending_after_restart(command: list[str]) -> None:
        if command == ["systemctl", "restart", probe.HOST_COLLECTOR_UNIT]:
            env_host.overrides[probe.HOST_COLLECTOR_UNIT] = health

    env_host.before_call = pending_after_restart
    assert restore_main(env_host, monkeypatch) == 78
    failure = env_host.emitted[-1]
    assert failure["retryable"] is False
    assert failure["cleanup_verified"] is False
    assert failure["error_site"].startswith("collector_health:"), (
        'test_first_post_restart_guard_survives_later_cleaned_receipt: expected failure["error_site"].startswith("collector_health:")'
    )
    assert (
        failure["error_code"]
        == hashlib.sha256(
            b"collector service health or job completion is unknown"
        ).hexdigest()[:16]
    )
    assert failure["failure_receipt"]["status"] == "RECORDED"
    receipt = failure["failure_receipt"]["receipt"]
    assert receipt == {
        "error_code": failure["error_code"],
        "error_site": failure["error_site"],
        "command": "restore-collector-env",
        "automatic": False,
        "phase": "RESTORED",
        "recorded_at": datetime.fromtimestamp(
            env_host.epoch + env_host.now, timezone.utc
        ).isoformat(),
        "run_id": env_host.run_id,
        "intent_sha256": intent,
    }
    path = probe.collector_guard_failure_path(
        env_host.run_id, "restore-collector-env", failure["error_code"]
    )
    first = path.read_bytes()
    assert json.loads(first) == receipt
    assert path.stat().st_mode & 0o777 == 0o600
    assert env_host.record()["state"] == "RESTORED"
    assert env_host.record()["manual_restore"] is True
    assert probe.COLLECTOR_ENV.read_text() == ENV_TEXT
    for material in (
        backup,
        backup.with_suffix(".probe.py"),
        backup.with_suffix(".nonce"),
        backup.with_suffix(".armed.json"),
        probe.SYSTEMD_UNIT_DIR / f"{unit}.service",
        probe.SYSTEMD_UNIT_DIR / f"{unit}.timer",
    ):
        assert material.exists(), (
            "test_first_post_restart_guard_survives_later_cleaned_receipt: expected material.exists()"
        )
    assert not any("disable" in call for call in env_host.calls), (
        'test_first_post_restart_guard_survives_later_cleaned_receipt: expected no any("disable" in call for call in env_host.calls)'
    )

    env_host.before_call = None
    env_host.overrides.clear()
    env_host.now += 2
    assert restore_main(env_host, monkeypatch) == 0
    assert env_host.emitted[-1]["state"] == "CLEANED"
    assert env_host.emitted[-1]["cleanup_verified"] is True
    assert not backup.exists(), (
        "test_first_post_restart_guard_survives_later_cleaned_receipt: expected no backup.exists()"
    )
    assert not (probe.SYSTEMD_UNIT_DIR / f"{unit}.service").exists(), (
        'test_first_post_restart_guard_survives_later_cleaned_receipt: expected no (probe.SYSTEMD_UNIT_DIR / f"{unit}.service").exists()'
    )
    assert not (probe.ACCEPTANCE_STATE / "collector-env-owner.json").exists(), (
        'test_first_post_restart_guard_survives_later_cleaned_receipt: expected no (probe.ACCEPTANCE_STATE / "collector-env-owner.json").exists()'
    )
    assert path.read_bytes() == first
    assert OWNER_NONCE.encode() not in first
    assert b"PRIVATE_FIXTURE_SENTINEL" not in first
    assert b"never-return-this-content" not in first
    assert failure["error"].encode() not in first


@pytest.mark.parametrize("automatic", [False, True])
def test_failure_duplicates_keep_first_metadata_and_do_not_rewrite(
    env_host: EnvHost, automatic: bool
) -> None:
    env_host.override()
    arguments = env_host.arguments(command="restore-collector-env", automatic=automatic)
    first = probe.collector_guard_failure(
        arguments, error_code="1" * 16, error_site="collector_health:123"
    )
    assert first["status"] == "RECORDED"
    path = probe.collector_guard_failure_path(
        env_host.run_id, arguments.command, "1" * 16
    )
    original = path.read_bytes()
    before = path.stat()
    env_host.now += 15
    arguments.automatic = not automatic
    repeated = probe.collector_guard_failure(
        arguments, error_code="1" * 16, error_site="collector_check_start:456"
    )
    assert repeated == {"status": "EXISTS", "receipt": first["receipt"]}
    assert repeated["receipt"]["automatic"] is automatic
    assert path.read_bytes() == original
    assert path.stat().st_ino == before.st_ino
    assert path.stat().st_mtime_ns == before.st_mtime_ns


def test_failure_receipts_are_bounded_and_deduplicated_by_code_and_command(
    env_host: EnvHost,
) -> None:
    env_host.override()
    first: dict[str, Any] | None = None
    for index in range(probe.COLLECTOR_GUARD_FAILURE_LIMIT):
        command = "collector-env-start-check" if index % 2 else "restore-collector-env"
        result = probe.collector_guard_failure(
            env_host.arguments(command=command),
            error_code=f"{index // 2:016x}",
            error_site=None,
        )
        assert result["status"] == "RECORDED"
        assert result["receipt"]["automatic"] is (index % 2 == 1)
        if first is None:
            first = result["receipt"]
    assert len(list(probe.ACCEPTANCE_STATE.glob("*.guard-*.json"))) == 16
    arguments = env_host.arguments(command="restore-collector-env")
    assert probe.collector_guard_failure(
        arguments, error_code="f" * 16, error_site=None
    ) == {"status": "LIMIT_REACHED"}
    assert probe.collector_guard_failure(
        arguments, error_code="0" * 16, error_site=None
    ) == {"status": "EXISTS", "receipt": first}
    assert len(list(probe.ACCEPTANCE_STATE.glob("*.guard-*.json"))) == 16


@pytest.mark.parametrize(
    "invalid", ["nonce", "run", "missing-intent", "bad-intent", "missing-owner"]
)
def test_unowned_failures_cannot_write_diagnostics(
    env_host: EnvHost, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    env_host.override()
    backup, _ = env_host.paths()
    changes: dict[str, str] = {}
    if invalid == "nonce":
        changes["owner_nonce"] = "b" * 32
    elif invalid == "run":
        changes["run_id"] = "another-campaign-a1"
    elif invalid == "missing-intent":
        probe.collector_override_record(backup).unlink()
    elif invalid == "bad-intent":
        probe.collector_override_record(backup).write_text('{"state":"ACTIVE"}')
    else:
        (probe.ACCEPTANCE_STATE / "collector-env-owner.json").unlink()
    before = {
        path.name: path.read_bytes()
        for path in probe.ACCEPTANCE_STATE.iterdir()
        if path.is_file()
    }
    env_host.calls.clear()
    assert (
        restore_main(
            env_host,
            monkeypatch,
            run_id=changes.get("run_id"),
            owner_nonce=changes.get("owner_nonce", OWNER_NONCE),
        )
        == 78
    )
    assert env_host.emitted[-1]["failure_receipt"] == {
        "status": "OWNERSHIP_UNAVAILABLE"
    }
    assert not list(probe.ACCEPTANCE_STATE.glob("*.guard-*.json")), (
        'test_unowned_failures_cannot_write_diagnostics: expected no list(probe.ACCEPTANCE_STATE.glob("*.guard-*.json"))'
    )
    assert before == {
        path.name: path.read_bytes()
        for path in probe.ACCEPTANCE_STATE.iterdir()
        if path.is_file()
    }
    assert env_host.calls == []


def test_failure_without_intent_does_not_create_state_or_read_env(
    env_host: EnvHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = probe.collector_private_file

    def read(path: Path, **kwargs: Any) -> bytes:
        assert path != probe.COLLECTOR_ENV
        return original(path, **kwargs)

    monkeypatch.setattr(probe, "collector_private_file", read)
    assert probe.collector_guard_failure(
        env_host.arguments(command="restore-collector-env"),
        error_code="1" * 16,
        error_site="collector_health:123",
    ) == {"status": "OWNERSHIP_UNAVAILABLE"}
    assert not probe.ACCEPTANCE_STATE.exists(), (
        "test_failure_without_intent_does_not_create_state_or_read_env: expected no probe.ACCEPTANCE_STATE.exists()"
    )
    assert env_host.calls == []


@pytest.mark.parametrize("lost", [None, "nonce", "intent"])
def test_failure_ownership_is_rechecked_under_lock(
    env_host: EnvHost, monkeypatch: pytest.MonkeyPatch, lost: str | None
) -> None:
    env_host.override()
    original = probe.collector_load_record
    reads = 0

    def read(run_id: str, nonce: str) -> dict[str, Any] | None:
        nonlocal reads
        reads += 1
        if reads == 2:
            if lost is None:
                return None
            owner_path = probe.ACCEPTANCE_STATE / "collector-env-owner.json"
            owner = json.loads(owner_path.read_text())
            owner["owner_nonce_sha256" if lost == "nonce" else "intent_sha256"] = (
                "f" * 64
            )
            probe.power_write_json(owner_path, owner)
        return original(run_id, nonce)

    monkeypatch.setattr(probe, "collector_load_record", read)
    assert probe.collector_guard_failure(
        env_host.arguments(command="restore-collector-env"),
        error_code="1" * 16,
        error_site=None,
    ) == {"status": "OWNERSHIP_UNAVAILABLE"}
    assert reads == 2
    assert not list(probe.ACCEPTANCE_STATE.glob("*.guard-*.json")), (
        'test_failure_ownership_is_rechecked_under_lock: expected no list(probe.ACCEPTANCE_STATE.glob("*.guard-*.json"))'
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("error_code", "private-error"),
        ("error_site", "private-error"),
        ("command", "other-command"),
        ("automatic", 1),
        ("phase", "private-env-content"),
        ("recorded_at", "not-a-time"),
        ("recorded_at", "2026-09-17T12:00:00"),
        ("recorded_at", []),
        ("run_id", "another-run"),
        ("intent_sha256", "f" * 64),
        ("extra", "private-content"),
    ],
    ids=[
        "code",
        "site",
        "command",
        "mode",
        "phase",
        "time",
        "naive-time",
        "time-type",
        "run",
        "intent",
        "extra-field",
    ],
)
def test_malformed_or_foreign_failure_receipt_is_never_overwritten(
    env_host: EnvHost, field: str, value: Any
) -> None:
    env_host.override()
    arguments = env_host.arguments(command="restore-collector-env")
    assert (
        probe.collector_guard_failure(arguments, error_code="1" * 16, error_site=None)[
            "status"
        ]
        == "RECORDED"
    )
    path = probe.collector_guard_failure_path(
        env_host.run_id, arguments.command, "1" * 16
    )
    receipt = json.loads(path.read_text())
    receipt[field] = value
    path.write_text(json.dumps(receipt))
    before = path.read_bytes()
    assert probe.collector_guard_failure(
        arguments, error_code="1" * 16, error_site=None
    ) == {"status": "LOGGING_UNAVAILABLE"}
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "invalid",
    ["truncated", "not-object", "missing-field", "public", "symlink", "hardlink"],
)
def test_unsafe_failure_receipt_is_preserved_without_following_links(
    env_host: EnvHost, invalid: str
) -> None:
    env_host.override()
    arguments = env_host.arguments(command="restore-collector-env")
    path = probe.collector_guard_failure_path(
        env_host.run_id, arguments.command, "1" * 16
    )
    foreign = env_host.root / "foreign-receipt"
    foreign.write_text("foreign-state")
    if invalid == "symlink":
        path.symlink_to(foreign)
    elif invalid == "hardlink":
        os.link(foreign, path)
    else:
        probe.power_create_file(
            path,
            b"{"
            if invalid == "truncated"
            else b"[]"
            if invalid == "not-object"
            else b'{"error_code":"1111111111111111"}',
            0o600,
        )
        if invalid == "public":
            path.chmod(0o644)
    before = path.read_bytes()
    inode = path.lstat().st_ino
    assert probe.collector_guard_failure(
        arguments, error_code="1" * 16, error_site=None
    ) == {"status": "LOGGING_UNAVAILABLE"}
    assert path.read_bytes() == before
    assert path.lstat().st_ino == inode
    assert foreign.read_text() == "foreign-state"


@pytest.mark.parametrize("failure", ["create", "directory-sync"])
def test_logging_failure_preserves_primary_guard_and_exit_status(
    env_host: EnvHost, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    env_host.override()
    backup, _ = env_host.paths()
    backup.write_text("wrong fixture backup")
    if failure == "create":

        def create(path: Path, contents: bytes, mode: int) -> None:
            raise PermissionError("private-logging-error")

        monkeypatch.setattr(probe, "power_create_file", create)
    else:
        original_sync = probe.power_sync_directory

        def sync(directory: Path) -> None:
            if directory == probe.ACCEPTANCE_STATE and list(
                directory.glob("*.guard-*.json")
            ):
                raise OSError("private-logging-error")
            original_sync(directory)

        monkeypatch.setattr(probe, "power_sync_directory", sync)
    assert restore_main(env_host, monkeypatch) == 78
    result = env_host.emitted[-1]
    expected = "collector backup digest differs from its intent"
    assert result["error"] == expected
    assert result["error_code"] == hashlib.sha256(expected.encode()).hexdigest()[:16]
    assert result["error_site"].startswith("restore_collector_env:"), (
        'test_logging_failure_preserves_primary_guard_and_exit_status: expected result["error_site"].startswith("restore_collector_env:")'
    )
    assert result["retryable"] is False
    assert result["failure_receipt"] == {"status": "LOGGING_UNAVAILABLE"}
    assert "private-logging-error" not in json.dumps(result)
    assert OWNER_NONCE not in json.dumps(result)
    assert env_host.record()["state"] == "ACTIVE"
    assert backup.exists(), (
        "test_logging_failure_preserves_primary_guard_and_exit_status: expected backup.exists()"
    )


def test_failure_creation_fsyncs_file_and_directory_without_replacing(
    env_host: EnvHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_host.override()
    path = probe.collector_guard_failure_path(
        env_host.run_id, "restore-collector-env", "1" * 16
    )
    original_sync = os.fsync
    synced: list[int] = []

    def sync(descriptor: int) -> None:
        synced.append(os.fstat(descriptor).st_ino)
        original_sync(descriptor)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "replace", forbidden)
    result = probe.collector_guard_failure(
        env_host.arguments(command="restore-collector-env"),
        error_code="1" * 16,
        error_site=None,
    )
    assert result["status"] == "RECORDED"
    assert path.stat().st_ino in synced
    assert probe.ACCEPTANCE_STATE.stat().st_ino in synced
    assert path.stat().st_mode & 0o777 == 0o600
