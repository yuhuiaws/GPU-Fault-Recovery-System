"""Power-operation ownership and recovery, with no host or GPU commands."""

from __future__ import annotations

import argparse
import configparser
import json
import os
import runpy
import socket
import stat
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast

import pytest

from scripts.e2e.regional.probes import collector_node_probe as probe


def forbidden(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("external transport or host operation is forbidden")


class PowerHost:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.now = 1000.0
        self.epoch = datetime(2026, 9, 17, tzinfo=timezone.utc)
        self.calls: list[list[str]] = []
        self.executions: list[list[str]] = []
        self.emitted: list[dict[str, Any]] = []
        self.active: set[str] = set()
        self.enabled: set[str] = set()
        self.overrides: dict[str, dict[str, str]] = {}
        self.before_call: Callable[[list[str]], None] | None = None
        self.after_call: Callable[[list[str]], None] | None = None
        self.after_sleep: Callable[[], None] | None = None
        self.start_load = True
        self.power_output: str | None = None
        self.inventory_output: str | None = None
        self.power_returncode = 0
        self.inventory_returncode = 0
        self.command_returncodes: dict[tuple[str, ...], int] = {}
        self.ignore_writes = False
        self.gpus: list[dict[str, Any]] = [
            {
                "index": index,
                "uuid": f"GPU-{index + 1:08x}-0000-0000-0000-000000000000",
                "pci_bdf": f"0000:0{index + 1}:00.0",
                "power_draw_w": 100.0,
                "power_limit_w": 700.0,
                "power_min_limit_w": 200.0 + index * 50,
                "power_default_limit_w": 700.0,
                "utilization_percent": 0.0,
            }
            for index in range(2)
        ]

    def arguments(self, **changes: Any) -> argparse.Namespace:
        return argparse.Namespace(
            **{
                "run_id": "power-safety-a1",
                "gpu_index": 0,
                "load_seconds": 40,
                "restore_seconds": 100,
                **changes,
            }
        )

    def record(self) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            json.loads(probe.power_record_path(self.arguments().run_id).read_text()),
        )

    def utc_now(self) -> datetime:
        return self.epoch + timedelta(seconds=self.now)

    def sleep(self, seconds: float) -> None:
        assert seconds > 0
        self.now += seconds
        if self.after_sleep is not None:
            self.after_sleep()

    def execv(self, executable: str, argv: list[str]) -> None:
        assert executable == str(probe.POWER_TIMEOUT)
        self.executions.append(argv)

    def units(self) -> tuple[str, str, str]:
        unit = probe.power_limit_restore_unit(self.arguments().run_id)
        return unit + ".timer", unit + ".service", unit + "-load.service"

    def power_writes(self) -> list[list[str]]:
        return [call for call in self.calls if "-pl" in call]

    def power_csv(self) -> str:
        return "\n".join(
            ",".join(
                str(gpu[key])
                for key in (
                    "index",
                    "uuid",
                    "power_draw_w",
                    "power_limit_w",
                    "power_min_limit_w",
                    "power_default_limit_w",
                    "utilization_percent",
                )
            )
            for gpu in self.gpus
        )

    def unit_state(self, name: str) -> dict[str, str]:
        path = probe.SYSTEMD_UNIT_DIR / name
        state = {
            "Id": name,
            "LoadState": "not-found",
            "ActiveState": "inactive",
            "SubState": "dead",
            "MainPID": "0",
            "ControlPID": "0",
            "Job": "",
            "ControlGroup": "",
            "FragmentPath": "",
            "DropInPaths": "",
            "NeedDaemonReload": "no",
        }
        if not path.exists():
            return state
        config = configparser.ConfigParser(interpolation=None)
        config.read_string(path.read_text())
        state["LoadState"] = "loaded"
        state["FragmentPath"] = str(path)
        if name in self.active:
            state["ActiveState"] = "active"
            state["SubState"] = "waiting" if name.endswith(".timer") else "running"
            if name.endswith(".service"):
                state["MainPID"] = str(os.getpid())
                state["ControlGroup"] = f"/system.slice/{name}"
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
        if "Service" in config:
            service = config["Service"]
            command = service["ExecStart"]
            state["ExecStart"] = (
                f"{{ path={command.split()[0]} ; argv[]={command} ; "
                "ignore_errors=no ; start_time=[n/a] ; stop_time=[n/a] ; "
                "pid=0 ; code=(null) ; status=0/0 }}"
            )
            for key in (
                "Type",
                "Restart",
                "RestartPreventExitStatus",
                "KillMode",
                "SendSIGKILL",
            ):
                if key in service:
                    state[key] = service[key]
            for key in ("Restart", "TimeoutStart", "TimeoutStop", "RuntimeMax"):
                if key + "Sec" in service:
                    state[key + "USec"] = service[key + "Sec"]
        state.update(self.overrides.get(name, {}))
        return state

    def run(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(command)
        if self.before_call is not None:
            self.before_call(command)
        for prefix, code in self.command_returncodes.items():
            if tuple(command[: len(prefix)]) == prefix:
                return subprocess.CompletedProcess(command, code, "", "fixture refusal")
        stdout, returncode = "", 0
        if command[0] in {"nvidia-smi", str(probe.POWER_SMI)}:
            if "-pl" in command:
                record = self.record()
                assert record["mutation_started"], "write must follow durable intent"
                assert all(gpu["power_limit_w"] == 700 for gpu in record["baseline"]), (
                    'run: expected all(gpu["power_limit_w"] == 700 for gpu in record["baseline"])'
                )
                assert command[1] == "-i", "every write must bind an original UUID"
                matches = [gpu for gpu in self.gpus if gpu["uuid"] == command[2]]
                assert len(matches) == 1, "write must target exactly one original GPU"
                if not self.ignore_writes:
                    matches[0]["power_limit_w"] = float(command[-1])
            elif "power.draw" in command[1]:
                stdout = (
                    self.power_output
                    if self.power_output is not None
                    else self.power_csv()
                )
                returncode = self.power_returncode
            elif "pci.bus_id" in command[1]:
                stdout = (
                    self.inventory_output
                    if self.inventory_output is not None
                    else "\n".join(
                        f"{gpu['index']},{gpu['uuid']},{gpu['pci_bdf']},NVIDIA H100"
                        for gpu in self.gpus
                    )
                )
                returncode = self.inventory_returncode
            else:
                raise AssertionError(f"unexpected GPU query: {command}")
        elif command[:2] == ["systemctl", "show"]:
            stdout = "\n".join(
                f"{key}={value}" for key, value in self.unit_state(command[2]).items()
            )
        elif command[:2] == ["systemctl", "daemon-reload"]:
            pass
        elif command[0] == "systemctl" and command[1] in {
            "enable",
            "disable",
            "start",
            "stop",
        }:
            name = command[-1]
            if command[1] == "enable":
                self.enabled.add(name)
            elif command[1] == "disable":
                self.enabled.discard(name)
            if command[1] == "start" or command[1] == "enable" and "--now" in command:
                self.active.add(name)
                if name.endswith(".service"):
                    group = probe.POWER_CGROUP_ROOT / "system.slice" / name
                    group.mkdir(parents=True, exist_ok=True)
                    (group / "cgroup.events").write_text("populated 1\nfrozen 0\n")
                    if name.endswith("-load.service") and self.start_load:
                        probe.power_load_only(self.arguments())
            if command[1] == "stop" or command[1] == "disable" and "--now" in command:
                self.active.discard(name)
                group = probe.POWER_CGROUP_ROOT / "system.slice" / name
                if group.exists():
                    (group / "cgroup.events").write_text("populated 0\nfrozen 0\n")
        else:
            raise AssertionError(f"unexpected external command: {command}")
        if self.after_call is not None:
            self.after_call(command)
        return subprocess.CompletedProcess(
            command, returncode, stdout, "fixture failure"
        )


@pytest.fixture
def power_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> PowerHost:
    for name in ("run", "Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, forbidden)
    for name in ("socket", "create_connection", "getaddrinfo"):
        monkeypatch.setattr(socket, name, forbidden)
    for name in ("system", "kill", "fork", "execv", "execve"):
        monkeypatch.setattr(os, name, forbidden)
    paths = {
        "ACCEPTANCE_STATE": "state",
        "SYSTEMD_UNIT_DIR": "systemd",
        "BOOT_ID_FILE": "boot-id",
        "COLLECTOR_ENV": "collector.env",
        "POWER_CGROUP_ROOT": "cgroup",
        "POWER_TRANSIENT_UNIT_DIR": "transient",
        "POWER_SMI": "bin/nvidia-smi",
        "POWER_TIMEOUT": "bin/timeout",
    }
    for name, suffix in paths.items():
        monkeypatch.setattr(probe, name, tmp_path / suffix)
    for name in ("SYSTEMD_UNIT_DIR", "POWER_CGROUP_ROOT"):
        getattr(probe, name).mkdir()
    probe.BOOT_ID_FILE.write_text("11111111-1111-1111-1111-111111111111")
    probe.COLLECTOR_ENV.write_text("GPU_FAULT_EXPECTED_GPU_COUNT=2\n")
    (probe.POWER_CGROUP_ROOT / "cgroup.controllers").write_text("cpu memory pids\n")
    binary = tmp_path / "bin/dcgmproftester13"
    binary.parent.mkdir()
    for path in (binary, probe.POWER_SMI, probe.POWER_TIMEOUT):
        path.write_text("inert executable fixture\n")
        path.chmod(0o700)
    host = PowerHost(tmp_path)
    monkeypatch.setattr(probe, "proftester_binary", lambda: str(binary))
    monkeypatch.setattr(
        probe, "time", SimpleNamespace(monotonic=lambda: host.now, sleep=host.sleep)
    )
    monkeypatch.setattr(
        probe,
        "datetime",
        SimpleNamespace(
            now=lambda tz: host.utc_now(), fromisoformat=datetime.fromisoformat
        ),
    )
    monkeypatch.setattr(os, "execv", host.execv)
    monkeypatch.setattr(subprocess, "run", host.run)
    monkeypatch.setattr(probe, "emit", host.emitted.append)
    return host


def test_owned_power_cycle_is_per_gpu_and_verifies_cleanup(
    power_host: PowerHost,
) -> None:
    host = power_host
    probe.throttle_gpu(host.arguments())
    assert [call[-1] for call in host.power_writes()] == ["200.0", "250.0"]
    assert host.emitted[-1]["timer_proof"]["armed"]
    assert host.emitted[-1]["load_unit_active"]
    receipt_path = probe.power_load_start_path(host.arguments().run_id)
    receipt = receipt_path.read_bytes()
    assert host.emitted[-1]["load_start"] == json.loads(receipt)
    record = host.record()
    assert (
        record["load_hard_deadline_monotonic"] + 15
        == record["restore_deadline_monotonic"]
    )
    copied = probe.power_record_path(host.arguments().run_id).with_suffix(".py")
    assert copied.is_file(), (
        "test_owned_power_cycle_is_per_gpu_and_verifies_cleanup: expected copied.is_file()"
    )
    host.calls.clear()
    probe.restore_gpu_power_limit(host.arguments())
    result = host.emitted[-1]
    assert result["restored"] and result["cleanup_verified"]
    assert result["load_stopped"] and result["timer_disarmed"]
    assert not result["no_mutation"]
    assert all(gpu["power_limit_w"] == 700 for gpu in host.gpus), (
        'test_owned_power_cycle_is_per_gpu_and_verifies_cleanup: expected all(gpu["power_limit_w"] == 700 for gpu in host.gpus)'
    )
    first_write = next(i for i, call in enumerate(host.calls) if "-pl" in call)
    assert host.calls.index(["systemctl", "stop", host.units()[2]]) < first_write
    assert host.record()["phase"] == "CLEANED"
    assert not (probe.ACCEPTANCE_STATE / "gpu-power-owner.json").exists(), (
        'test_owned_power_cycle_is_per_gpu_and_verifies_cleanup: expected no (probe.ACCEPTANCE_STATE / "gpu-power-owner.json").exists()'
    )
    assert not copied.exists(), (
        "test_owned_power_cycle_is_per_gpu_and_verifies_cleanup: expected no copied.exists()"
    )
    assert receipt_path.read_bytes() == receipt
    assert not list(probe.SYSTEMD_UNIT_DIR.iterdir()), (
        "test_owned_power_cycle_is_per_gpu_and_verifies_cleanup: expected no list(probe.SYSTEMD_UNIT_DIR.iterdir())"
    )
    assert not host.active and not host.enabled
    host.calls.clear()
    probe.restore_gpu_power_limit(host.arguments())
    assert not host.power_writes(), "completed cleanup must be idempotent"


def test_restore_without_intent_does_not_query_or_mutate_host(
    power_host: PowerHost,
) -> None:
    host = power_host
    host.gpus[0]["power_limit_w"] = 500
    probe.restore_gpu_power_limit(host.arguments())
    assert host.calls == []
    assert host.gpus[0]["power_limit_w"] == 500
    assert host.emitted[-1]["no_mutation"] and host.emitted[-1]["cleanup_verified"]
    assert host.emitted[-1]["restored"] is False


def test_refused_custom_cap_is_not_overwritten_by_cleanup(
    power_host: PowerHost,
) -> None:
    host = power_host
    host.gpus[0]["power_limit_w"] = 500
    with pytest.raises(probe.ProbeError, match="nondefault"):
        probe.throttle_gpu(host.arguments())
    host.calls.clear()
    probe.restore_gpu_power_limit(host.arguments())
    assert host.calls == []
    assert host.gpus[0]["power_limit_w"] == 500
    assert not probe.power_record_path(host.arguments().run_id).exists(), (
        "test_refused_custom_cap_is_not_overwritten_by_cleanup: expected no probe.power_record_path(host.arguments().run_id).exists()"
    )


def test_exclusive_owner_rejects_other_run_and_foreign_cleanup(
    power_host: PowerHost,
) -> None:
    host = power_host
    probe.throttle_gpu(host.arguments())
    host.calls.clear()
    for operation in (probe.throttle_gpu, probe.restore_gpu_power_limit):
        with pytest.raises(probe.ProbeError, match="owns this node"):
            operation(host.arguments(run_id="another-run-a1"))
    assert host.calls == []
    assert host.units()[0] in host.active
    assert host.units()[2] in host.active


def test_same_id_cannot_restart_an_owned_operation(power_host: PowerHost) -> None:
    host = power_host
    probe.throttle_gpu(host.arguments())
    host.calls.clear()
    with pytest.raises(probe.ProbeError, match="owns this node"):
        probe.throttle_gpu(host.arguments())
    assert not host.calls, (
        "test_same_id_cannot_restart_an_owned_operation: expected no host.calls"
    )
    probe.restore_gpu_power_limit(host.arguments())
    with pytest.raises(probe.ProbeError, match="already been used"):
        probe.throttle_gpu(host.arguments())


def test_concurrent_operation_cannot_enter_process_lock(power_host: PowerHost) -> None:
    with probe.power_operation_lock():
        with pytest.raises(probe.ProbeError, match="still executing"):
            probe.throttle_gpu(power_host.arguments())
    assert not power_host.calls, (
        "test_concurrent_operation_cannot_enter_process_lock: expected no power_host.calls"
    )


def test_unowned_unit_collision_is_never_cleaned_up(power_host: PowerHost) -> None:
    host = power_host
    foreign = probe.SYSTEMD_UNIT_DIR / host.units()[0]
    foreign.write_text("unowned timer\n")
    with pytest.raises(probe.ProbeError, match="already exists"):
        probe.throttle_gpu(host.arguments())
    host.calls.clear()
    probe.restore_gpu_power_limit(host.arguments())
    assert not host.calls, (
        "test_unowned_unit_collision_is_never_cleaned_up: expected no host.calls"
    )
    assert foreign.read_text() == "unowned timer\n"


@pytest.mark.parametrize(
    "failure",
    [
        "query-failure",
        "inventory-failure",
        "empty",
        "malformed",
        "missing",
        "duplicate-uuid",
        "duplicate-index",
        "wrong-binding",
        "bad-bdf",
        "duplicate-bdf",
        "nan-draw",
        "inf-limit",
        "nan-minimum",
        "inf-default",
        "unknown-utilization",
        "negative-draw",
        "invalid-utilization",
        "expected-count",
        "nondefault",
    ],
)
def test_incomplete_or_unknown_power_never_authorizes_a_write(
    power_host: PowerHost, failure: str
) -> None:
    host = power_host
    if failure == "query-failure":
        host.power_returncode = 1
    elif failure == "inventory-failure":
        host.inventory_returncode = 1
    elif failure == "empty":
        host.power_output = ""
    elif failure == "malformed":
        host.power_output = host.power_csv() + "\nincomplete,row\n"
    elif failure == "missing":
        host.gpus.pop()
    elif failure == "duplicate-uuid":
        host.gpus[1]["uuid"] = host.gpus[0]["uuid"]
    elif failure == "duplicate-index":
        host.gpus[1]["index"] = 0
    elif failure == "wrong-binding":
        host.power_output = host.power_csv().replace(
            host.gpus[0]["uuid"], host.gpus[1]["uuid"]
        )
    elif failure == "bad-bdf":
        host.gpus[1]["pci_bdf"] = "unknown"
    elif failure == "duplicate-bdf":
        host.gpus[1]["pci_bdf"] = host.gpus[0]["pci_bdf"]
    elif failure == "expected-count":
        probe.COLLECTOR_ENV.write_text("GPU_FAULT_EXPECTED_GPU_COUNT=unknown\n")
    else:
        key, value = {
            "nan-draw": ("power_draw_w", float("nan")),
            "inf-limit": ("power_limit_w", float("inf")),
            "nan-minimum": ("power_min_limit_w", float("nan")),
            "inf-default": ("power_default_limit_w", float("inf")),
            "unknown-utilization": ("utilization_percent", "N/A"),
            "negative-draw": ("power_draw_w", -1),
            "invalid-utilization": ("utilization_percent", 101),
            "nondefault": ("power_limit_w", 500),
        }[failure]
        host.gpus[0][key] = value
    with pytest.raises(probe.ProbeError):
        probe.throttle_gpu(host.arguments())
    assert not host.power_writes(), (
        "test_incomplete_or_unknown_power_never_authorizes_a_write: expected no host.power_writes()"
    )
    assert not (probe.ACCEPTANCE_STATE / "gpu-power-owner.json").exists(), (
        'test_incomplete_or_unknown_power_never_authorizes_a_write: expected no (probe.ACCEPTANCE_STATE / "gpu-power-owner.json").exists()'
    )


@pytest.mark.parametrize(
    "property_name,value",
    [
        ("ActiveState", "inactive"),
        ("SubState", "elapsed"),
        ("Unit", "foreign.service"),
        ("NextElapseUSecMonotonic", "1099s"),
        ("NextElapseUSecMonotonic", "infinity"),
        ("AccuracyUSec", "1min"),
        ("RandomizedDelayUSec", "1s"),
        ("DropInPaths", "/foreign.conf"),
        ("NeedDaemonReload", "yes"),
    ],
)
def test_unverified_timer_never_authorizes_power(
    power_host: PowerHost, property_name: str, value: str
) -> None:
    host = power_host
    host.overrides[host.units()[0]] = {property_name: value}
    with pytest.raises(probe.ProbeError):
        probe.throttle_gpu(host.arguments())
    assert not host.power_writes(), (
        "test_unverified_timer_never_authorizes_power: expected no host.power_writes()"
    )
    assert not host.record()["mutation_started"]


@pytest.mark.parametrize(
    "target", ["restore-command", "load-command", "load-kill-policy"]
)
def test_owned_service_command_and_shutdown_policy_are_required(
    power_host: PowerHost, target: str
) -> None:
    host = power_host
    name = host.units()[1 if target == "restore-command" else 2]
    host.overrides[name] = (
        {"SendSIGKILL": "no"}
        if target == "load-kill-policy"
        else {"ExecStart": "{ path=/bin/true ; argv[]=/bin/true ; ignore_errors=no ; }"}
    )
    with pytest.raises(probe.ProbeError):
        probe.throttle_gpu(host.arguments())
    assert not host.power_writes(), (
        "test_owned_service_command_and_shutdown_policy_are_required: expected no host.power_writes()"
    )


def test_slow_arming_cannot_renew_the_fixed_deadline(power_host: PowerHost) -> None:
    host = power_host

    def delay(command: list[str]) -> None:
        if command[:3] == ["systemctl", "enable", "--now"]:
            host.now += 60

    host.after_call = delay
    with pytest.raises(probe.ProbeError, match="fixed recovery deadline"):
        probe.throttle_gpu(host.arguments())
    assert not host.power_writes(), (
        "test_slow_arming_cannot_renew_the_fixed_deadline: expected no host.power_writes()"
    )
    assert host.record()["restore_deadline_monotonic"] == 1100


@pytest.mark.parametrize("ack_lost", [False, True])
def test_interrupted_cap_remains_recoverable(
    power_host: PowerHost, ack_lost: bool
) -> None:
    host = power_host

    def interrupt(command: list[str]) -> None:
        if "-pl" in command:
            raise RuntimeError("lost cap acknowledgement")

    if ack_lost:
        host.after_call = interrupt
    else:
        host.before_call = interrupt
    with pytest.raises(RuntimeError, match="acknowledgement"):
        probe.throttle_gpu(host.arguments())
    assert host.record()["mutation_started"]
    assert host.units()[0] in host.active
    assert (
        probe.power_record_path(host.arguments().run_id).with_suffix(".py").is_file()
    ), (
        'test_interrupted_cap_remains_recoverable: expected probe.power_record_path(host.arguments().run_id).with_suffix(".py").is_file()'
    )
    host.before_call = host.after_call = None
    probe.restore_gpu_power_limit(host.arguments())
    assert all(gpu["power_limit_w"] == 700 for gpu in host.gpus), (
        'test_interrupted_cap_remains_recoverable: expected all(gpu["power_limit_w"] == 700 for gpu in host.gpus)'
    )
    assert host.emitted[-1]["cleanup_verified"]


def test_timer_arm_ack_loss_does_not_authorize_a_gpu_write(
    power_host: PowerHost,
) -> None:
    host = power_host

    def interrupt(command: list[str]) -> None:
        if command[:3] == ["systemctl", "enable", "--now"]:
            raise RuntimeError("lost timer acknowledgement")

    host.after_call = interrupt
    with pytest.raises(RuntimeError, match="timer acknowledgement"):
        probe.throttle_gpu(host.arguments())
    assert not host.power_writes() and not host.record()["mutation_started"]
    host.after_call = None
    probe.restore_gpu_power_limit(host.arguments())
    assert host.emitted[-1]["no_mutation"]
    assert host.emitted[-1]["cleanup_verified"]
    assert not host.power_writes(), (
        "test_timer_arm_ack_loss_does_not_authorize_a_gpu_write: expected no host.power_writes()"
    )


@pytest.mark.parametrize(
    "failure", ["stop-error", "populated", "job", "identity", "query", "readback"]
)
def test_uncertain_cleanup_preserves_recovery_and_cannot_report_success(
    power_host: PowerHost, failure: str
) -> None:
    host = power_host
    probe.throttle_gpu(host.arguments())
    host.calls.clear()
    host.emitted.clear()
    receipt_path = probe.power_load_start_path(host.arguments().run_id)
    receipt = receipt_path.read_bytes()
    if failure == "stop-error":
        host.command_returncodes[("systemctl", "stop")] = 1
    elif failure == "populated":

        def occupied(command: list[str]) -> None:
            if command[:2] == ["systemctl", "stop"]:
                path = probe.POWER_CGROUP_ROOT / "system.slice" / host.units()[2]
                (path / "cgroup.events").write_text("populated 1\n")

        host.after_call = occupied
    elif failure == "job":
        host.overrides[host.units()[2]] = {"Job": "123 /queued/job"}
    elif failure == "identity":
        host.gpus[0]["uuid"] = "GPU-ffffffff-0000-0000-0000-000000000000"
    elif failure == "query":
        host.power_returncode = 1
    elif failure == "readback":
        host.ignore_writes = True
    with pytest.raises(probe.ProbeError):
        probe.restore_gpu_power_limit(host.arguments())
    assert not host.emitted, (
        "test_uncertain_cleanup_preserves_recovery_and_cannot_report_success: expected no host.emitted"
    )
    assert host.units()[0] in host.active
    assert (probe.ACCEPTANCE_STATE / "gpu-power-owner.json").exists(), (
        'test_uncertain_cleanup_preserves_recovery_and_cannot_report_success: expected (probe.ACCEPTANCE_STATE / "gpu-power-owner.json").exists()'
    )
    assert (
        probe.power_record_path(host.arguments().run_id).with_suffix(".py").is_file()
    ), (
        "test_uncertain_cleanup_preserves_recovery_and_cannot_report_success: expected probe.power_record_path(host.arguments().run_id).with_suffi..."
    )
    assert receipt_path.read_bytes() == receipt
    assert not any(call[:2] == ["systemctl", "disable"] for call in host.calls), (
        'test_uncertain_cleanup_preserves_recovery_and_cannot_report_success: expected no any(call[:2] == ["systemctl", "disable"] for call in hos...'
    )
    if failure != "readback":
        assert not host.power_writes(), (
            "test_uncertain_cleanup_preserves_recovery_and_cannot_report_success: expected no host.power_writes()"
        )


def test_missing_gpu_at_restore_cannot_disarm_recovery(power_host: PowerHost) -> None:
    host = power_host
    probe.throttle_gpu(host.arguments())
    host.gpus.pop()
    host.calls.clear()
    with pytest.raises(probe.ProbeError, match="incomplete"):
        probe.restore_gpu_power_limit(host.arguments())
    assert not host.power_writes(), (
        "test_missing_gpu_at_restore_cannot_disarm_recovery: expected no host.power_writes()"
    )
    assert host.units()[0] in host.active


def test_external_limit_change_is_not_overwritten_on_restore(
    power_host: PowerHost,
) -> None:
    host = power_host
    probe.throttle_gpu(host.arguments())
    host.gpus[0]["power_limit_w"] = 450
    host.calls.clear()
    with pytest.raises(probe.ProbeError, match="drifted"):
        probe.restore_gpu_power_limit(host.arguments())
    assert not host.power_writes(), (
        "test_external_limit_change_is_not_overwritten_on_restore: expected no host.power_writes()"
    )
    assert host.gpus[0]["power_limit_w"] == 450
    assert host.units()[0] in host.active


def test_automatic_recovery_requires_manual_cleanup_confirmation(
    power_host: PowerHost,
) -> None:
    host = power_host
    probe.throttle_gpu(host.arguments())
    host.calls.clear()
    probe.restore_gpu_power_limit(host.arguments(automatic=True))
    assert not host.calls and host.emitted[-1]["deferred"]
    host.now = host.record()["restore_deadline_monotonic"]
    probe.restore_gpu_power_limit(host.arguments(automatic=True))
    result = host.emitted[-1]
    assert result["restored"] and result["load_stopped"]
    assert result["cleanup_deferred"] and not result["cleanup_verified"]
    assert host.units()[0] in host.active
    assert (
        probe.power_record_path(host.arguments().run_id).with_suffix(".py").exists()
    ), (
        'test_automatic_recovery_requires_manual_cleanup_confirmation: expected probe.power_record_path(host.arguments().run_id).with_suffix(".py"...'
    )
    assert (probe.ACCEPTANCE_STATE / "gpu-power-owner.json").exists(), (
        'test_automatic_recovery_requires_manual_cleanup_confirmation: expected (probe.ACCEPTANCE_STATE / "gpu-power-owner.json").exists()'
    )
    probe.restore_gpu_power_limit(host.arguments())
    assert host.emitted[-1]["cleanup_verified"] and host.emitted[-1]["timer_disarmed"]


def test_load_has_a_hard_kill_budget_and_refuses_late_start(
    power_host: PowerHost,
) -> None:
    host = power_host
    probe.throttle_gpu(host.arguments())
    assert host.executions[0][1:4] == ["--signal=TERM", "--kill-after=5s", "45s"]
    record = host.record()
    assert host.now + 45 + 5 < record["load_hard_deadline_monotonic"]
    host.now = record["load_hard_deadline_monotonic"] - 50
    with pytest.raises(probe.ProbeError, match="fixed recovery deadline"):
        probe.throttle_gpu(host.arguments(load_only=True))
    assert len(host.executions) == 1


def test_load_start_is_durable_and_bound_immediately_before_exec(
    power_host: PowerHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = power_host
    receipt_path = probe.power_load_start_path(host.arguments().run_id)
    fsync, link, sync_directory = os.fsync, os.link, probe.power_sync_directory
    synced: set[int] = set()
    events: list[str] = []
    receipts: list[dict[str, Any]] = []
    calls_at_exec: list[int] = []

    def sync_file(descriptor: int) -> None:
        fsync(descriptor)
        synced.add(os.fstat(descriptor).st_ino)

    def publish(source: Path, destination: Path) -> None:
        assert destination == receipt_path
        assert source.stat().st_ino in synced
        assert not destination.exists(), (
            "test_load_start_is_durable_and_bound_immediately_before_exec: expected no destination.exists()"
        )
        link(source, destination)
        events.append("publish")

    def sync_parent(directory: Path) -> None:
        if receipt_path.exists():
            with pytest.raises(BlockingIOError):
                probe.power_read_file(receipt_path, shared_lock=True)
            sync_directory(directory)
            events.append("durable")
        else:
            sync_directory(directory)

    def exec_load(executable: str, argv: list[str]) -> None:
        assert events == ["publish", "durable"]
        receipt = probe.power_read_json(receipt_path)
        record = host.record()
        assert receipt == {
            "run_id": record["run_id"],
            "boot_id": record["boot_id"],
            "intent_sha256": record["intent_sha256"],
            "gpu_index": 1,
            "gpu_uuid": host.gpus[1]["uuid"],
            "pid": os.getpid(),
            "started_at": host.utc_now().isoformat(),
            "started_monotonic": host.now,
        }
        assert receipt_path.stat().st_mode & 0o777 == 0o600
        assert receipt_path.stat().st_nlink == 1
        assert argv[-2:] == ["-i", "1"]
        receipts.append(receipt)
        calls_at_exec.append(len(host.calls))
        host.execv(executable, argv)
        host.now += 3

    monkeypatch.setattr(os, "fsync", sync_file)
    monkeypatch.setattr(os, "link", publish)
    monkeypatch.setattr(probe, "power_sync_directory", sync_parent)
    monkeypatch.setattr(os, "execv", exec_load)
    probe.throttle_gpu(host.arguments(gpu_index=1))
    assert host.emitted[-1]["load_start"] == receipts[0]
    assert receipts[0]["started_monotonic"] == host.now - 3
    assert not any(
        call[0] == str(probe.POWER_SMI) for call in host.calls[calls_at_exec[0] :]
    ), "the handoff must not come from a later observed-busy sample"


def test_load_start_receipt_is_never_overwritten_or_backdated(
    power_host: PowerHost,
) -> None:
    host = power_host
    probe.throttle_gpu(host.arguments())
    path = probe.power_load_start_path(host.arguments().run_id)
    original, inode = path.read_bytes(), path.stat().st_ino
    host.now += 1
    with pytest.raises(FileExistsError):
        probe.power_load_only(host.arguments())
    assert path.read_bytes() == original and path.stat().st_ino == inode
    assert len(host.executions) == 1
    assert host.units()[0] in host.active
    probe.restore_gpu_power_limit(host.arguments())
    assert host.emitted[-1]["cleanup_verified"]
    assert path.read_bytes() == original and path.stat().st_ino == inode


@pytest.mark.parametrize("symlink", [False, True])
def test_preexisting_load_start_receipt_refuses_all_mutation(
    power_host: PowerHost, symlink: bool
) -> None:
    host = power_host
    with probe.power_operation_lock():
        path = probe.power_load_start_path(host.arguments().run_id)
        if symlink:
            path.symlink_to(host.root / "absent-foreign-target")
        else:
            path.write_text("foreign receipt")
    with pytest.raises(probe.ProbeError, match="receipt already exists"):
        probe.throttle_gpu(host.arguments())
    assert not host.calls and not host.executions
    probe.restore_gpu_power_limit(host.arguments())
    assert not host.calls, (
        "test_preexisting_load_start_receipt_refuses_all_mutation: expected no host.calls"
    )
    assert path.is_symlink() if symlink else path.read_text() == "foreign receipt"


@pytest.mark.parametrize(
    "failure",
    [
        "run",
        "boot",
        "intent",
        "gpu-index",
        "gpu-uuid",
        "boolean-index",
        "pid",
        "boolean-pid",
        "missing-pid",
        "extra-field",
        "non-object",
        "invalid-json",
        "stale-monotonic",
        "future-monotonic",
        "deadline-monotonic",
        "nan-monotonic",
        "infinite-monotonic",
        "zero-monotonic",
        "negative-monotonic",
        "boolean-monotonic",
        "string-monotonic",
        "stale-wall",
        "future-wall",
        "naive-wall",
        "non-utc-wall",
        "invalid-wall",
        "non-string-wall",
    ],
)
def test_invalid_load_start_receipt_cannot_ack_or_erase_recovery(
    power_host: PowerHost, failure: str
) -> None:
    host = power_host
    path = probe.power_load_start_path(host.arguments().run_id)

    def alter_receipt(command: list[str]) -> None:
        if command[:2] != ["systemctl", "start"]:
            return
        receipt = json.loads(path.read_text())
        changes: dict[str, tuple[str, Any]] = {
            "run": ("run_id", "foreign-run"),
            "boot": ("boot_id", "22222222-2222-2222-2222-222222222222"),
            "intent": ("intent_sha256", "f" * 64),
            "gpu-index": ("gpu_index", 1),
            "gpu-uuid": ("gpu_uuid", host.gpus[1]["uuid"]),
            "boolean-index": ("gpu_index", False),
            "pid": ("pid", os.getpid() + 1),
            "boolean-pid": ("pid", True),
            "extra-field": ("unexpected", "untrusted"),
            "stale-monotonic": ("started_monotonic", host.now - 1),
            "future-monotonic": ("started_monotonic", host.now + 1),
            "deadline-monotonic": (
                "started_monotonic",
                host.record()["load_hard_deadline_monotonic"],
            ),
            "nan-monotonic": ("started_monotonic", float("nan")),
            "infinite-monotonic": ("started_monotonic", float("inf")),
            "zero-monotonic": ("started_monotonic", 0),
            "negative-monotonic": ("started_monotonic", -1),
            "boolean-monotonic": ("started_monotonic", True),
            "string-monotonic": ("started_monotonic", str(host.now)),
            "stale-wall": (
                "started_at",
                (host.utc_now() - timedelta(seconds=1)).isoformat(),
            ),
            "future-wall": (
                "started_at",
                (host.utc_now() + timedelta(seconds=1)).isoformat(),
            ),
            "naive-wall": (
                "started_at",
                host.utc_now().replace(tzinfo=None).isoformat(),
            ),
            "non-utc-wall": (
                "started_at",
                host.utc_now().astimezone(timezone(timedelta(hours=1))).isoformat(),
            ),
            "invalid-wall": ("started_at", "unknown"),
            "non-string-wall": ("started_at", 123),
        }
        if failure == "missing-pid":
            del receipt["pid"]
        elif failure == "non-object":
            receipt = []
        elif failure == "invalid-json":
            path.write_text("{")
            return
        else:
            key, value = changes[failure]
            receipt[key] = value
        path.write_text(json.dumps(receipt))

    host.after_call = alter_receipt
    with pytest.raises((probe.ProbeError, json.JSONDecodeError)):
        probe.throttle_gpu(host.arguments())
    original = path.read_bytes()
    assert not host.emitted, (
        "test_invalid_load_start_receipt_cannot_ack_or_erase_recovery: expected no host.emitted"
    )
    assert host.units()[0] in host.active and host.units()[2] in host.active
    assert (probe.ACCEPTANCE_STATE / "gpu-power-owner.json").exists(), (
        'test_invalid_load_start_receipt_cannot_ack_or_erase_recovery: expected (probe.ACCEPTANCE_STATE / "gpu-power-owner.json").exists()'
    )
    assert (
        probe.power_record_path(host.arguments().run_id).with_suffix(".py").is_file()
    ), (
        'test_invalid_load_start_receipt_cannot_ack_or_erase_recovery: expected probe.power_record_path(host.arguments().run_id).with_suffix(".py"...'
    )
    assert not host.record()["load_closed"]
    host.after_call = None
    probe.restore_gpu_power_limit(host.arguments())
    assert host.emitted[-1]["cleanup_verified"]
    assert all(gpu["power_limit_w"] == 700 for gpu in host.gpus), (
        'test_invalid_load_start_receipt_cannot_ack_or_erase_recovery: expected all(gpu["power_limit_w"] == 700 for gpu in host.gpus)'
    )
    assert path.read_bytes() == original, "invalid receipts remain audit-only"


def test_parent_waits_for_the_load_process_receipt(power_host: PowerHost) -> None:
    host = power_host
    host.start_load = False
    requested = host.now
    sleeps: list[float] = []

    def delayed_load() -> None:
        assert not host.emitted, (
            "test_parent_waits_for_the_load_process_receipt: expected no host.emitted"
        )
        sleeps.append(host.now)
        if host.now >= requested + 2:
            host.after_sleep = None
            probe.power_load_only(host.arguments())

    host.after_sleep = delayed_load
    probe.throttle_gpu(host.arguments())
    assert len(sleeps) > 1
    assert len(host.executions) == 1
    assert host.emitted[-1]["load_start"]["started_monotonic"] == host.now
    assert host.emitted[-1]["load_start"]["started_at"] == host.utc_now().isoformat()


@pytest.mark.parametrize("setup_delay", [0, 25])
def test_missing_load_receipt_wait_is_bounded_by_original_deadlines(
    power_host: PowerHost, setup_delay: int
) -> None:
    host = power_host
    host.start_load = False

    def slow_setup(command: list[str]) -> None:
        if command[:3] == ["systemctl", "enable", "--now"]:
            host.now += setup_delay

    host.after_call = slow_setup
    with pytest.raises(probe.ProbeError, match="deadline"):
        probe.throttle_gpu(host.arguments())
    assert host.now == 1000 + (30 if setup_delay == 0 else 35)
    assert not host.emitted and not host.executions
    assert host.record()["restore_deadline_monotonic"] == 1100
    assert host.record()["load_hard_deadline_monotonic"] == 1085
    assert host.units()[0] in host.active
    assert (probe.ACCEPTANCE_STATE / "gpu-power-owner.json").exists(), (
        'test_missing_load_receipt_wait_is_bounded_by_original_deadlines: expected (probe.ACCEPTANCE_STATE / "gpu-power-owner.json").exists()'
    )
    assert (
        probe.power_record_path(host.arguments().run_id).with_suffix(".py").is_file()
    ), (
        'test_missing_load_receipt_wait_is_bounded_by_original_deadlines: expected probe.power_record_path(host.arguments().run_id).with_suffix("....'
    )
    host.after_call = None
    probe.restore_gpu_power_limit(host.arguments())
    assert host.emitted[-1]["cleanup_verified"]
    assert all(gpu["power_limit_w"] == 700 for gpu in host.gpus), (
        'test_missing_load_receipt_wait_is_bounded_by_original_deadlines: expected all(gpu["power_limit_w"] == 700 for gpu in host.gpus)'
    )


@pytest.mark.parametrize(
    "failure", ["file-fsync", "publish", "directory-fsync", "exec"]
)
def test_failed_load_handoff_keeps_recovery_and_can_be_cleaned(
    power_host: PowerHost, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    host = power_host
    path = probe.power_load_start_path(host.arguments().run_id)
    fsync, link = os.fsync, os.link
    loading = False

    def starting(command: list[str]) -> None:
        nonlocal loading
        if command[:2] == ["systemctl", "start"]:
            loading = True

    def fail_sync(descriptor: int) -> None:
        directory = stat.S_ISDIR(os.fstat(descriptor).st_mode)
        if loading and (
            failure == "directory-fsync"
            and directory
            or failure == "file-fsync"
            and not directory
        ):
            raise OSError("receipt fsync failed")
        fsync(descriptor)

    def fail_publish(source: Path, destination: Path) -> None:
        if failure == "publish":
            raise OSError("receipt creation failed")
        link(source, destination)

    def fail_exec(executable: str, argv: list[str]) -> None:
        assert path.is_file(), "even an unsuccessful exec follows its durable receipt"
        raise OSError("load exec failed")

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail_sync)
        patch.setattr(os, "link", fail_publish)
        if failure == "exec":
            patch.setattr(os, "execv", fail_exec)
        host.before_call = starting
        with pytest.raises(OSError, match="failed"):
            probe.throttle_gpu(host.arguments())
    assert not host.emitted and not host.executions
    assert host.units()[0] in host.active
    assert (probe.ACCEPTANCE_STATE / "gpu-power-owner.json").exists(), (
        'test_failed_load_handoff_keeps_recovery_and_can_be_cleaned: expected (probe.ACCEPTANCE_STATE / "gpu-power-owner.json").exists()'
    )
    assert (
        probe.power_record_path(host.arguments().run_id).with_suffix(".py").is_file()
    ), (
        'test_failed_load_handoff_keeps_recovery_and_can_be_cleaned: expected probe.power_record_path(host.arguments().run_id).with_suffix(".py")....'
    )
    assert path.exists() == (failure in {"directory-fsync", "exec"})
    original = path.read_bytes() if path.exists() else None
    host.before_call = None
    probe.restore_gpu_power_limit(host.arguments())
    assert host.emitted[-1]["cleanup_verified"]
    assert all(gpu["power_limit_w"] == 700 for gpu in host.gpus), (
        'test_failed_load_handoff_keeps_recovery_and_can_be_cleaned: expected all(gpu["power_limit_w"] == 700 for gpu in host.gpus)'
    )
    assert (path.read_bytes() if path.exists() else None) == original


@pytest.mark.parametrize("late_stage", ["publication", "parent-verification"])
def test_load_receipt_cannot_extend_the_safe_start_deadline(
    power_host: PowerHost, monkeypatch: pytest.MonkeyPatch, late_stage: str
) -> None:
    host = power_host
    path = probe.power_load_start_path(host.arguments().run_id)
    sync_directory = probe.power_sync_directory

    def expire() -> None:
        record = host.record()
        host.now = record["load_hard_deadline_monotonic"] - record["load_seconds"] - 10

    def slow_publish(directory: Path) -> None:
        sync_directory(directory)
        if path.exists() and late_stage == "publication":
            expire()

    def slow_verify(command: list[str]) -> None:
        if (
            late_stage == "parent-verification"
            and path.exists()
            and command[:3] == ["systemctl", "show", host.units()[2]]
        ):
            expire()

    monkeypatch.setattr(probe, "power_sync_directory", slow_publish)
    host.after_call = slow_verify
    with pytest.raises(probe.ProbeError, match="fixed recovery deadline"):
        probe.throttle_gpu(host.arguments())
    assert not host.emitted, (
        "test_load_receipt_cannot_extend_the_safe_start_deadline: expected no host.emitted"
    )
    assert len(host.executions) == (late_stage == "parent-verification")
    assert path.is_file(), (
        "test_load_receipt_cannot_extend_the_safe_start_deadline: expected path.is_file()"
    )
    assert host.units()[0] in host.active
    assert (probe.ACCEPTANCE_STATE / "gpu-power-owner.json").exists(), (
        'test_load_receipt_cannot_extend_the_safe_start_deadline: expected (probe.ACCEPTANCE_STATE / "gpu-power-owner.json").exists()'
    )
    assert (
        probe.power_record_path(host.arguments().run_id).with_suffix(".py").is_file()
    ), (
        'test_load_receipt_cannot_extend_the_safe_start_deadline: expected probe.power_record_path(host.arguments().run_id).with_suffix(".py").is_...'
    )


def test_closed_load_cannot_start_again(power_host: PowerHost) -> None:
    host = power_host
    probe.throttle_gpu(host.arguments())
    probe.restore_gpu_power_limit(host.arguments())
    with pytest.raises(probe.ProbeError, match="owned start intent"):
        probe.throttle_gpu(host.arguments(load_only=True))


@pytest.mark.parametrize("failure", ["boot", "copy", "missing-owner"])
def test_recovery_identity_drift_never_writes_power(
    power_host: PowerHost, failure: str
) -> None:
    host = power_host
    probe.throttle_gpu(host.arguments())
    if failure == "boot":
        probe.BOOT_ID_FILE.write_text("22222222-2222-2222-2222-222222222222")
    elif failure == "copy":
        path = probe.power_record_path(host.arguments().run_id).with_suffix(".py")
        path.chmod(0o700)
        path.write_text("drift")
    else:
        (probe.ACCEPTANCE_STATE / "gpu-power-owner.json").unlink()
    host.calls.clear()
    with pytest.raises(probe.ProbeError):
        probe.restore_gpu_power_limit(host.arguments())
    assert not host.calls, (
        "test_recovery_identity_drift_never_writes_power: expected no host.calls"
    )
    assert host.units()[0] in host.active


def test_recovery_copy_runs_without_repository_imports(
    power_host: PowerHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    host = power_host
    probe.throttle_gpu(host.arguments())
    original_import = builtins.__import__

    def stdlib_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "scripts" or name.startswith(("scripts.", "gpu_fault")):
            raise AssertionError("recovery copy cannot import repository helpers")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", stdlib_import)
    copied = probe.power_record_path(host.arguments().run_id).with_suffix(".py")
    namespace = runpy.run_path(str(copied), run_name="isolated_power_probe")
    arguments = namespace["parser"]().parse_args(
        ["restore-gpu-power-limit", "--run-id", host.arguments().run_id, "--automatic"]
    )
    assert (
        arguments.automatic and arguments.handler.__name__ == "restore_gpu_power_limit"
    )


def test_new_boot_reports_manual_recovery_without_a_restart_loop(
    power_host: PowerHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = power_host
    probe.throttle_gpu(host.arguments())
    assert host.unit_state(host.units()[1])["RestartPreventExitStatus"] == "78"
    probe.BOOT_ID_FILE.write_text("22222222-2222-2222-2222-222222222222")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "restore-gpu-power-limit",
            "--run-id",
            host.arguments().run_id,
            "--automatic",
        ],
    )
    host.calls.clear()
    assert probe.main() == 78
    assert not host.calls, (
        "test_new_boot_reports_manual_recovery_without_a_restart_loop: expected no host.calls"
    )
    assert host.emitted[-1]["manual_confirmation_required"]
    assert not host.emitted[-1]["cleanup_verified"]
    assert not host.emitted[-1]["restored"]


def test_timer_verification_accepts_multiweek_systemd_uptime(
    power_host: PowerHost,
) -> None:
    host = power_host
    host.now = 3 * 604800 + 123
    host.overrides[host.units()[0]] = {"NextElapseUSecMonotonic": "3w 3min 43s"}
    probe.throttle_gpu(host.arguments())
    assert host.emitted[-1]["timer_proof"]["deadline_monotonic"] == host.now + 100


@pytest.mark.parametrize("field", ["baseline", "deadline", "owner"])
def test_immutable_intent_drift_is_refused(power_host: PowerHost, field: str) -> None:
    host = power_host
    probe.throttle_gpu(host.arguments())
    path = probe.power_record_path(host.arguments().run_id)
    record = host.record()
    if field == "baseline":
        record["baseline"][0]["power_limit_w"] = 450
    elif field == "deadline":
        record["restore_deadline_monotonic"] += 60
        record["load_hard_deadline_monotonic"] += 60
    else:
        owner = probe.ACCEPTANCE_STATE / "gpu-power-owner.json"
        owner.write_text(
            json.dumps({"run_id": host.arguments().run_id, "intent_sha256": "wrong"})
        )
    path.write_text(json.dumps(record))
    host.calls.clear()
    with pytest.raises(probe.ProbeError, match="record is invalid"):
        probe.restore_gpu_power_limit(host.arguments())
    assert not host.calls, (
        "test_immutable_intent_drift_is_refused: expected no host.calls"
    )
    assert host.units()[0] in host.active


def test_completed_receipt_cannot_hide_reappearing_units(power_host: PowerHost) -> None:
    host = power_host
    probe.throttle_gpu(host.arguments())
    probe.restore_gpu_power_limit(host.arguments())
    contents = probe.power_unit_contents(host.record())
    name = host.units()[2]
    (probe.SYSTEMD_UNIT_DIR / name).write_bytes(contents[name])
    host.active.add(name)
    host.calls.clear()
    host.emitted.clear()
    with pytest.raises(probe.ProbeError):
        probe.restore_gpu_power_limit(host.arguments())
    assert not host.power_writes(), (
        "test_completed_receipt_cannot_hide_reappearing_units: expected no host.power_writes()"
    )
    assert not host.emitted, (
        "test_completed_receipt_cannot_hide_reappearing_units: expected no host.emitted"
    )
    assert name in host.active, "an old completion does not own a new process"


def test_failed_disarm_is_not_success_and_cleanup_can_resume(
    power_host: PowerHost,
) -> None:
    host = power_host
    probe.throttle_gpu(host.arguments())
    host.command_returncodes[("systemctl", "disable")] = 1
    host.emitted.clear()
    with pytest.raises(probe.ProbeError):
        probe.restore_gpu_power_limit(host.arguments())
    assert not host.emitted, (
        "test_failed_disarm_is_not_success_and_cleanup_can_resume: expected no host.emitted"
    )
    assert all(gpu["power_limit_w"] == 700 for gpu in host.gpus), (
        'test_failed_disarm_is_not_success_and_cleanup_can_resume: expected all(gpu["power_limit_w"] == 700 for gpu in host.gpus)'
    )
    assert host.units()[0] in host.active
    assert (
        probe.power_record_path(host.arguments().run_id).with_suffix(".py").is_file()
    ), (
        'test_failed_disarm_is_not_success_and_cleanup_can_resume: expected probe.power_record_path(host.arguments().run_id).with_suffix(".py").is...'
    )
    host.command_returncodes.clear()
    host.calls.clear()
    probe.restore_gpu_power_limit(host.arguments())
    assert not host.power_writes(), (
        "test_failed_disarm_is_not_success_and_cleanup_can_resume: expected no host.power_writes()"
    )
    assert host.emitted[-1]["cleanup_verified"]


@pytest.mark.parametrize("transient", [False, True])
def test_orphan_recovery_for_a_different_run_blocks_admission(
    power_host: PowerHost, transient: bool
) -> None:
    host = power_host
    directory = probe.POWER_TRANSIENT_UNIT_DIR if transient else probe.SYSTEMD_UNIT_DIR
    directory.mkdir(exist_ok=True)
    foreign = directory / "gpu-fault-power-limit-restore-foreign.timer"
    foreign.write_text("unowned previous power recovery\n")
    with pytest.raises(probe.ProbeError, match="already exists"):
        probe.throttle_gpu(host.arguments())
    host.calls.clear()
    probe.restore_gpu_power_limit(host.arguments())
    assert not host.calls, (
        "test_orphan_recovery_for_a_different_run_blocks_admission: expected no host.calls"
    )
    assert foreign.exists(), (
        "test_orphan_recovery_for_a_different_run_blocks_admission: expected foreign.exists()"
    )


def test_missing_cgroup_population_cannot_authorize_restore(
    power_host: PowerHost,
) -> None:
    host = power_host
    probe.throttle_gpu(host.arguments())

    def missing(command: list[str]) -> None:
        if command[:2] == ["systemctl", "stop"]:
            path = probe.POWER_CGROUP_ROOT / "system.slice" / host.units()[2]
            (path / "cgroup.events").unlink()

    host.after_call = missing
    host.calls.clear()
    with pytest.raises(probe.ProbeError, match="population is unknown"):
        probe.restore_gpu_power_limit(host.arguments())
    assert not host.power_writes(), (
        "test_missing_cgroup_population_cannot_authorize_restore: expected no host.power_writes()"
    )
    assert host.units()[0] in host.active


@pytest.mark.parametrize(
    "value,seconds",
    [
        ("0", 0),
        ("1000000", 1),
        ("1ms", 0.001),
        ("1min 2.5s", 62.5),
        ("1h 1min 1s", 3661),
        ("1w 2d 3h 4min 5s", 788645),
        ("1y 2month 3w", 38631600),
    ],
)
def test_systemd_deadline_forms(value: str, seconds: float) -> None:
    assert probe.power_systemd_seconds(value) == seconds
