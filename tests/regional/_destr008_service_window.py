from __future__ import annotations

import configparser
import copy
import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import destr008_service_window as controller
from scripts.e2e.regional.host_probe_fixture import HostProbeSettings
from scripts.e2e.regional.probes import warm_spare_node_probe as probe

CASE = "GF-REGIONAL-DESTR-008"
PINS = {
    "cluster_id": "test-cluster",
    "node": "test-node",
    "node_uid": "test-node-uid",
    "artifact_sha256": "a" * 64,
    "bundle_sha256": "b" * 64,
    "profile_version": "test-profile",
}


def write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(content)
    path.chmod(0o600)
    return path


def command_property(command: str) -> str:
    return (
        f"{{ path={shlex.split(command)[0]} ; argv[]={command} ; ignore_errors=no ; "
        "start_time=[test] ; stop_time=[test] ; pid=100 ; code=(null) ; status=0/0 }"
    )


class Host:
    """All systemd and host identities are fake; files are owned regular files."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.now = 2_000_000_000.0
        self.elapsed = 1000.0
        self.calls: list[tuple[str, ...]] = []
        self.fail: set[tuple[str, ...]] = set()
        self.fail_after: set[tuple[str, ...]] = set()
        self.stuck: set[str] = set()
        self.overrides: dict[str, dict[str, str]] = {}
        self.units: dict[str, dict[str, str]] = {}
        self.pending_stop = False
        self.pending_start = False
        self.complete_on_sleep = ""
        self.roots = tmp_path / "host"
        root = self.roots / "var/acceptance/service-window"
        root.parent.parent.mkdir(parents=True, mode=0o700)
        systemd = self.roots / "run/systemd/system"
        systemd.mkdir(parents=True, mode=0o700)
        current = self.roots / "runtime/current"
        current.mkdir(parents=True, mode=0o700)
        marker = write(self.roots / "runtime/current-marker", "owned runtime pointer")
        executable = write(current / "venv/bin/gpu-fault-node-agent", "node agent")
        python = write(self.roots / "usr/bin/python3", "independent Python")
        env = write(
            self.roots / "etc/node-agent.env",
            "\n".join(f"{probe.PIN_KEYS[k]}={v}" for k, v in PINS.items()),
        )
        fake_paths = {
            "ROOT": root,
            "SYSTEMD": systemd,
            "CURRENT": current,
            "PYTHON": python,
            "AGENT_ENV": env,
            "BOOT_ID": write(self.roots / "proc/boot", "boot-one"),
            "MACHINE_ID": write(self.roots / "etc/machine", "machine-one"),
            "PRODUCT_ID": write(self.roots / "sys/product", "product-one"),
            "SELF_CGROUP": write(
                self.roots / "proc/self-cgroup", "0::/not-independent\n"
            ),
            "CGROUP_ROOT": self.roots / "sys/cgroup",
        }
        for name, path in fake_paths.items():
            monkeypatch.setattr(probe, name, path)
        identity = probe.file_identity

        def file_identity(path: Path) -> dict[str, Any]:
            # Simulate only the runtime symlink. No real symlink is created.
            return identity(marker) if path == current else identity(path)

        monkeypatch.setattr(probe, "file_identity", file_identity)
        clock = SimpleNamespace(
            time=lambda: self.now,
            monotonic=lambda: self.elapsed,
            clock_gettime=lambda _clock: self.elapsed,
            sleep=self.sleep,
            CLOCK_BOOTTIME=time.CLOCK_BOOTTIME,
        )
        self.clock = clock
        monkeypatch.setattr(probe, "time", clock)
        monkeypatch.setattr(controller, "time", clock)
        monkeypatch.setattr(probe, "subprocess", SimpleNamespace(run=self.run))
        for name in probe.ALLOWED_SERVICES:
            path = write(
                self.roots / "etc/systemd" / name, "[Service]\nExecStart=test\n"
            )
            unit = self.empty(name, "loaded")
            unit.update(
                ActiveState="active",
                SubState="running",
                MainPID="100",
                InvocationID="1" * 32,
                ExecMainStartTimestampMonotonic="100",
                FragmentPath=str(path),
                UnitFileState="enabled",
                ExecStart=command_property(str(executable)),
                Type="simple",
                EnvironmentFiles=f"{env} (ignore_errors=no)",
                JobTimeoutUSec="infinity",
                TimeoutStartUSec="1min 30s",
                TimeoutStopUSec="1min 30s",
            )
            self.units[name] = unit
        self.binding = self.make_binding()
        self.window = probe.ServiceWindow(self.binding)

    def empty(self, name: str, load: str = "not-found") -> dict[str, str]:
        unit = dict.fromkeys(probe.UNIT_FIELDS | probe.SERVICE_FIELDS, "")
        unit.update(
            Id=name,
            LoadState=load,
            ActiveState="inactive",
            SubState="dead",
            Transient="no",
            Job="0",
            NeedDaemonReload="no",
            UnitFileState="static",
            MainPID="0",
            ControlPID="0",
            KillMode="control-group",
            SendSIGKILL="yes",
        )
        return unit

    def make_binding(
        self, service: str = "kubelet.service", delay: int = 15
    ) -> dict[str, Any]:
        captured = probe.capture(service)
        return {
            "case_id": CASE,
            "run_id": "test-service-run",
            "owner": "c" * 32,
            **{k: PINS[k] for k in ("cluster_id", "node", "node_uid")},
            "release_id": "test-release",
            "plan_sha256": "d" * 64,
            "helper_sha256": hashlib.sha256(
                Path(probe.__file__).read_bytes()
            ).hexdigest(),
            **{k: captured[k] for k in ("baseline_sha256", "host_sha256", "boot_id")},
            "service": service,
            "restore_at": int(self.now) + 180 + delay,
            "expires_at": int(self.now) + 180 + delay + probe.RECOVERY_SECONDS,
            "stop_delay_seconds": delay,
        }

    def reload(self) -> None:
        for path in probe.SYSTEMD.iterdir():
            if path.suffix not in {".service", ".timer"}:
                continue
            parser = configparser.ConfigParser(interpolation=None)
            parser.read_string(path.read_text())
            unit = self.units.setdefault(path.name, self.empty(path.name))
            unit.update(
                LoadState="loaded",
                FragmentPath=str(path),
                Description=parser["Unit"]["Description"],
            )
            if path.suffix == ".service":
                unit.update(
                    ExecStart=command_property(parser["Service"]["ExecStart"]),
                    Type=parser["Service"]["Type"],
                    KillMode=parser["Service"]["KillMode"],
                    SendSIGKILL=parser["Service"]["SendSIGKILL"],
                )
        for name in list(self.units):
            if (
                name not in probe.ALLOWED_SERVICES
                and not (probe.SYSTEMD / name).exists()
            ):
                if self.units[name]["ActiveState"] in {"inactive", "failed"}:
                    self.units[name] = self.empty(name)
        for name in probe.ALLOWED_SERVICES:
            dropins = sorted((probe.SYSTEMD / f"{name}.d").glob("*.conf"))
            unit = self.units[name]
            unit["DropInPaths"] = " ".join(str(p) for p in dropins)
            unit.update(
                JobTimeoutUSec="45s" if dropins else "infinity",
                TimeoutStartUSec="30s" if dropins else "1min 30s",
                TimeoutStopUSec="30s" if dropins else "1min 30s",
            )

    def run(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        assert command[0] == "/bin/systemctl", "unexpected subprocess boundary"
        assert kwargs["timeout"] == 20 and kwargs["env"]["LC_ALL"] == "C"
        args = tuple(command[1:])
        self.calls.append(args)
        if args in self.fail:
            return subprocess.CompletedProcess(command, 1, "", "withheld failure")
        action = args[0]
        if action == "show":
            name = args[1]
            unit = {
                **self.units.get(name, self.empty(name)),
                **self.overrides.get(name, {}),
            }
            # systemd 252 (Amazon Linux 2023) prints no line for an empty list or
            # exec property: EnvironmentFiles= of a unit without one, ExecStart=
            # and EnvironmentFiles= of a unit it never loaded (live 2026-09-19,
            # DESTR-008 attempts 19 and 23).
            unit = {
                k: v
                for k, v in unit.items()
                if v != "" or k not in {"ExecStart", "EnvironmentFiles"}
            }
            output = "\n".join(f"{k}={v}" for k, v in unit.items())
            return subprocess.CompletedProcess(
                command, 4 if unit["LoadState"] == "not-found" else 0, output, ""
            )
        if action == "daemon-reload":
            self.reload()
        else:
            assert action in {"start", "stop"}, args
            name = args[-1]
            assert name in self.units and self.units[name]["LoadState"] == "loaded", (
                args
            )
            unit = self.units[name]
            if action == "start":
                if name.endswith(".timer"):
                    unit.update(ActiveState="active", SubState="waiting")
                else:
                    unit.update(
                        ActiveState="active",
                        SubState="running",
                        ControlPID="0",
                        Job="0",
                        MainPID=str(os.getpid())
                        if name not in probe.ALLOWED_SERVICES
                        else "200",
                        InvocationID="2" * 32
                        if name in probe.ALLOWED_SERVICES
                        else "a" * 32,
                    )
                    if name in probe.ALLOWED_SERVICES and self.pending_start:
                        unit.update(
                            ActiveState="activating", SubState="start", Job="91"
                        )
            elif name not in self.stuck:
                unit.update(
                    ActiveState="inactive",
                    SubState="dead",
                    MainPID="0",
                    ControlPID="0",
                    Job="0",
                )
                if name in probe.ALLOWED_SERVICES and self.pending_stop:
                    unit.update(
                        ActiveState="deactivating",
                        SubState="stop",
                        Job="81",
                        ControlPID="101",
                    )
        if args in self.fail_after:
            return subprocess.CompletedProcess(command, 1, "", "reply lost")
        return subprocess.CompletedProcess(command, 0, "", "")

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        self.elapsed += seconds
        if self.complete_on_sleep:
            unit = self.units[self.complete_on_sleep]
            if unit["ActiveState"] == "deactivating":
                unit.update(
                    ActiveState="inactive",
                    SubState="dead",
                    MainPID="0",
                    ControlPID="0",
                    Job="0",
                )
            else:
                unit.update(
                    ActiveState="active",
                    SubState="running",
                    MainPID="200",
                    ControlPID="0",
                    Job="0",
                )
            self.complete_on_sleep = ""

    def become(self, role: str) -> None:
        name = self.window.units[role]
        self.units[name].update(
            ActiveState="active",
            SubState="running",
            MainPID=str(os.getpid()),
            InvocationID="a" * 32,
            Job="0",
        )
        write(probe.SELF_CGROUP, f"0::/system.slice/{name}\n")
        os.environ["INVOCATION_ID"] = "a" * 32

    def tick(self) -> dict[str, Any]:
        self.become("restore-service")
        return self.window.tick()

    def arm(self) -> dict[str, Any]:
        self.window.prepare()
        return self.tick()

    def fire_stop(self) -> dict[str, Any]:
        self.sleep(max(1, self.binding["stop_delay_seconds"]))
        self.tick()
        self.become("stop-service")
        report = self.window.stop_owned()
        unit = self.units[self.window.units["stop-service"]]
        unit.update(
            ActiveState="inactive",
            SubState="dead",
            MainPID="0",
            ControlPID="0",
            Job="0",
        )
        return report

    def stopped(self) -> dict[str, Any]:
        self.arm()
        self.window.schedule_stop()
        return self.fire_stop()

    def read(self) -> dict[str, Any]:
        return probe.private_read(self.window.state_path)

    def save(self, data: dict[str, Any]) -> None:
        probe.write_record(self.window.state_path, data)

    def mutations(self) -> list[tuple[str, ...]]:
        return [c for c in self.calls if c[0] != "show"]


class Transport:
    def __init__(self, host: Host, directory: Path) -> None:
        self.h = host
        self.settings = HostProbeSettings(
            kubeconfig=write(directory / "kubeconfig", "test-only kubeconfig"),
            context="test-context",
            namespace="test-namespace",
            node=PINS["node"],
            image="test-image",
            case_id=CASE,
            run_id="test-service-run",
            probe_script=Path(probe.__file__),
            state_directory=directory,
        )
        self.state_path = directory / "host-proof.json"
        self.pod = "test-owned-pod"
        self.configmap = "test-owned-configmap"
        self.creates = 0
        self.cleanups = 0
        self.commands: list[str] = []
        self.lost_ack = ""
        self.arm_ack = True
        self.reports: dict[str, Any] = {}
        self.residual: dict[str, bool] = {
            f"pod/{self.pod}": False,
            f"configmap/{self.configmap}": False,
            "host_script": False,
            "creation_unresolved": False,
        }

    def create(self) -> None:
        self.creates += 1
        settings = self.settings
        scope = {
            "kubeconfig": str(settings.kubeconfig.resolve()),
            "context": settings.context,
            "namespace": settings.namespace,
            "node": settings.node,
            "image": settings.image,
            "case_id": settings.case_id,
            "run_id": settings.run_id,
            "script_sha256": hashlib.sha256(
                settings.probe_script.read_bytes()
            ).hexdigest(),
        }
        probe.write_record(
            self.state_path,
            {
                "schema_version": 1,
                "scope": scope,
                "owner": "e" * 32,
                "node_uid": PINS["node_uid"],
                "closed": False,
                "script_may_exist": False,
                "resources": {
                    k: {"name": n, "uid": f"test-{k}-uid", "create_started": True}
                    for k, n in (("pod", self.pod), ("configmap", self.configmap))
                },
            },
        )
        if self.lost_ack == "create":
            raise TimeoutError("create ACK lost")

    def execute(self, command: str, *args: str, timeout: int = 180) -> dict[str, Any]:
        assert timeout == 180
        self.commands.append(command)
        if command == "snapshot":
            assert args[0] == "--service"
            result = probe.capture(args[1])
        else:
            assert args[0] == "--binding"
            binding = json.loads(args[1])
            self.h.binding = binding
            self.h.window = probe.ServiceWindow(binding)
            method = {
                "stop-with-failsafe": "schedule_stop",
                "restore-service": "restore",
            }.get(command, command)
            result = getattr(self.h.window, method)()
            if command == "prepare" and self.arm_ack:
                self.h.tick()
        if self.lost_ack == command:
            self.lost_ack = ""
            raise TimeoutError("transport reply lost")
        return copy.deepcopy(self.reports.get(command, result))

    def cleanup(self) -> dict[str, bool]:
        self.cleanups += 1
        if self.state_path.exists() and not any(self.residual.values()):
            record = probe.private_read(self.state_path)
            record.update(closed=True, script_may_exist=False)
            probe.write_record(self.state_path, record)
        if self.lost_ack == "transport-cleanup":
            self.lost_ack = ""
            raise TimeoutError("cleanup ACK lost")
        return self.residual

    def residuals(self) -> dict[str, bool]:
        return self.residual
