from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.probes import destr014_recovery_probe as probe


def binding() -> dict[str, Any]:
    return {
        "case_id": probe.CASE,
        "run_id": "destr014-test-a1",
        "owner": "owner-one",
        "release_id": "release-test",
        "plan_sha256": "a" * 64,
        "helper_sha256": hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest(),
        "cluster_id": "cluster-test",
        "node": "node-test",
        "node_uid": "uid-test",
        "boot_id": "boot-before",
        "artifact_sha256": "b" * 64,
        "bundle_sha256": "c" * 64,
        "profile_version": "profile-test",
        "restore_at": 2000,
        "expires_at": 2180,
    }


class HostHarness:
    """Real private regular files, simulated links/systemd/host reads only."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.monkeypatch = monkeypatch
        self.now = 1000.0
        self.boot = "boot-before"
        self.links: dict[Path, str] = {}
        self.calls: list[tuple[str, ...]] = []
        self.agent_active = True
        self.agent_job = ""
        self.start_stuck = False
        self.stop_stuck = False
        self.lose_ack = ""
        self.invocation = "independent-invocation"
        self.cgroup = ""
        self.running: dict[str, bool] = {}
        self.overrides: dict[str, dict[str, str]] = {}
        # A reboot may enumerate the root NVMe device under another number;
        # every lstat on the new boot reports a shifted st_dev so the identity
        # checks are exercised the way the host would present them.
        self.device_shift = 0
        self.scope = binding()
        self.base = tmp_path
        root = tmp_path / "persistent"
        systemd = tmp_path / "systemd"
        systemd.mkdir(mode=0o700)
        (systemd / "multi-user.target.wants").mkdir(mode=0o700)
        (systemd / "timers.target.wants").mkdir(mode=0o700)
        self.fragment = systemd / probe.AGENT
        self.fragment.write_text("[Service]\nType=simple\n")
        self.fragment.chmod(0o600)
        env = tmp_path / "agent.env"
        env.write_text(
            "# installer literals\n\n"
            + "\n".join(
                f"{name}={self.scope[key]}" for key, name in probe.PIN_KEYS.items()
            )
        )
        env.chmod(0o600)
        current = tmp_path / "current"
        slot = tmp_path / "release"
        (slot / "venv/bin").mkdir(parents=True)
        (slot / "venv/bin/gpu-fault-node-agent").write_text("entrypoint")
        (slot / "venv/bin/gpu-fault-node-agent").chmod(0o600)
        python = tmp_path / "python"
        python.write_text("interpreter")
        python.chmod(0o600)
        self.enable = systemd / "multi-user.target.wants" / probe.AGENT
        monkeypatch.setattr(probe, "ROOT", root)
        monkeypatch.setattr(probe, "SYSTEMD", systemd)
        monkeypatch.setattr(probe, "ENABLE_LINK", self.enable)
        monkeypatch.setattr(probe, "AGENT_ENV", env)
        monkeypatch.setattr(probe, "CURRENT", current)
        monkeypatch.setattr(probe, "PYTHON", python)
        monkeypatch.setattr(probe, "boot_id", lambda: self.boot)
        monkeypatch.setattr(probe.time, "time", lambda: self.now)
        monkeypatch.setattr(probe.time, "monotonic", lambda: self.now)
        monkeypatch.setattr(probe.time, "sleep", self.sleep)
        monkeypatch.setattr(probe, "systemctl", self.systemctl)

        original_lstat = Path.lstat
        original_is_link = Path.is_symlink
        original_resolve = Path.resolve
        original_read = Path.read_text
        original_bytes = Path.read_bytes
        original_link = os.link
        original_readlink = os.readlink

        def lstat(path: Path) -> Any:
            info = original_lstat(path)
            if path in self.links:
                return SimpleNamespace(
                    st_mode=stat.S_IFLNK | 0o777,
                    st_uid=info.st_uid,
                    st_dev=info.st_dev + self.device_shift,
                    st_ino=info.st_ino,
                    st_nlink=info.st_nlink,
                )
            if self.device_shift:
                return SimpleNamespace(
                    st_mode=info.st_mode,
                    st_uid=info.st_uid,
                    st_dev=info.st_dev + self.device_shift,
                    st_ino=info.st_ino,
                    st_nlink=info.st_nlink,
                    st_mtime_ns=info.st_mtime_ns,
                    st_ctime_ns=info.st_ctime_ns,
                )
            return info

        def resolve(path: Path, *args: Any, **kwargs: Any) -> Path:
            if path in self.links and path.exists():
                return original_resolve(Path(self.links[path]), *args, **kwargs)
            return original_resolve(path, *args, **kwargs)

        def link(source: Any, target: Any, **kwargs: Any) -> None:
            original_link(source, target, **kwargs)
            if Path(source) in self.links:
                self.links[Path(target)] = self.links[Path(source)]

        def read_text(path: Path, *args: Any, **kwargs: Any) -> str:
            if path == Path("/proc/self/cgroup"):
                return self.cgroup
            return original_read(path, *args, **kwargs)

        def read_bytes(path: Path) -> bytes:
            if path == Path("/etc/machine-id"):
                return b"test-machine"
            if path == Path("/sys/class/dmi/id/product_uuid"):
                return b"test-hardware-identity"
            return original_bytes(path)

        monkeypatch.setattr(Path, "lstat", lstat)
        monkeypatch.setattr(
            Path,
            "is_symlink",
            lambda path: (
                path.exists() if path in self.links else original_is_link(path)
            ),
        )
        monkeypatch.setattr(Path, "resolve", resolve)
        monkeypatch.setattr(Path, "read_text", read_text)
        monkeypatch.setattr(Path, "read_bytes", read_bytes)
        monkeypatch.setattr(os, "link", link)
        monkeypatch.setattr(os, "symlink", self.symlink)
        monkeypatch.setattr(
            os,
            "readlink",
            lambda path, **kwargs: (
                self.links[Path(path)]
                if Path(path) in self.links
                else original_readlink(path, **kwargs)
            ),
        )
        self.symlink(str(slot), current)
        self.symlink(str(self.fragment), self.enable)
        self.recovery = probe.Recovery(self.scope)

    def symlink(self, source: Any, target: Any) -> None:
        path = Path(target)
        with path.open("x") as handle:
            handle.write("simulated symbolic link")
        path.chmod(0o600)
        self.links[path] = str(source)

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def systemctl(self, *args: str) -> dict[str, str]:
        self.calls.append(args)
        if args[0] == "show":
            name = args[1]
            if name == probe.AGENT:
                dropins = sorted((probe.SYSTEMD / f"{probe.AGENT}.d").glob("*.conf"))
                values = {
                    "LoadState": "loaded",
                    "FragmentPath": str(self.fragment),
                    "DropInPaths": " ".join(str(p) for p in dropins),
                    "UnitFileState": "enabled" if self.enable.exists() else "disabled",
                    "ActiveState": "active" if self.agent_active else "inactive",
                    "Job": self.agent_job,
                    "JobTimeoutUSec": "45s" if dropins else "infinity",
                    "TimeoutStartUSec": "30s" if dropins else "1min 30s",
                }
            else:
                timer = name.endswith(".timer")
                active = self.running.get(name, False)
                values = {
                    "LoadState": "loaded"
                    if (probe.SYSTEMD / name).exists()
                    else "not-found",
                    "FragmentPath": str(probe.SYSTEMD / name),
                    "ActiveState": ("active" if timer else "activating")
                    if active
                    else "inactive",
                    "SubState": "waiting" if active else "dead",
                    "UnitFileState": "enabled" if timer else "static",
                    "InvocationID": self.invocation,
                    "Job": "",
                }
            return {**values, **self.overrides.get(name, {})}
        if args[0] == "start":
            name = args[-1]
            if name == probe.AGENT:
                self.agent_active = not self.start_stuck
            else:
                self.running[name] = True
        elif args[0] == "stop":
            if not self.stop_stuck:
                self.running[args[1]] = False
        else:
            assert args == ("daemon-reload",), args
        if self.lose_ack and self.lose_ack in args:
            self.lose_ack = ""
            raise TimeoutError("simulated lost systemd ACK")
        return {}

    def tick(self) -> dict[str, Any]:
        fresh = probe.Recovery(self.scope)
        self.monkeypatch.setenv("INVOCATION_ID", self.invocation)
        self.running[fresh.service] = True
        self.cgroup = f"0::/system.slice/{fresh.service}\n"
        try:
            return fresh.tick()
        finally:
            self.running[fresh.service] = False

    def arm(self) -> dict[str, Any]:
        assert self.recovery.prepare()["phase"] == "INSTALLED"
        assert self.tick()["phase"] == "ARMED"
        return self.recovery.status()

    def read(self) -> dict[str, Any]:
        return probe.read_record(self.recovery.state_path)

    def update(self, **values: Any) -> None:
        data = self.read()
        data.update(values)
        probe.atomic_state(self.recovery.state_path, data)

    def reboot(self, *, renumber_devices: bool = False) -> None:
        self.boot = "boot-after"
        if renumber_devices:
            self.device_shift += 1
        self.agent_active = self.enable.exists()
        self.running.clear()
        self.running[self.recovery.timer] = (
            probe.SYSTEMD / "timers.target.wants" / self.recovery.timer
        ).exists()
        self.invocation = "post-reboot-invocation"

    def installer_reinstall(self, *, enable: bool = True) -> None:
        """The product's node-installer reconciler on the new boot: it rewrites
        the unit file and the runtime entrypoint (new content, new ctime) and,
        unless told otherwise, recreates the enable link under a fresh inode
        pointing at the installer unit and starts the Agent."""

        self.fragment.write_text("[Service]\nType=simple\n# reinstalled\n")
        entrypoint = Path(self.links[probe.CURRENT]) / "venv/bin/gpu-fault-node-agent"
        entrypoint.write_text("entrypoint reinstalled")
        if not enable:
            return
        if self.enable.exists():
            self.enable.unlink()
            self.links.pop(self.enable, None)
        self.symlink(str(self.fragment), self.enable)
        self.agent_active = True


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> HostHarness:
    return HostHarness(tmp_path, monkeypatch)


def test_recovery_survives_reboot_and_controller_loss_then_proves_cleanup(
    host: HostHarness,
) -> None:
    original = probe.file_identity(host.enable)
    assert host.arm()["ack"]["invocation_id"] == host.invocation
    assert host.recovery.disable()["phase"] == "DISABLED"
    assert not host.enable.exists() and host.agent_active
    host.reboot()
    assert not host.agent_active, "the disabled Agent must stay stopped after reboot"
    host.now = host.scope["restore_at"] - 1
    assert host.tick()["phase"] == "DISABLED"
    assert not host.enable.exists(), (
        "recovery must not enable the Agent before its deadline"
    )
    host.now += 1
    result = host.tick()
    assert result["phase"] == "RESTORED" and result["restore_reason"] == "deadline"
    assert probe.file_identity(host.enable) == original and host.agent_active
    assert not (probe.SYSTEMD / host.recovery.timer).exists(), (
        "successful recovery must remove its persistent timer"
    )
    assert not (probe.SYSTEMD / host.recovery.service).exists(), (
        "successful recovery must remove its service unit"
    )
    assert not host.recovery.dropin.exists(), (
        "successful recovery must release the owned timeout drop-in"
    )
    assert host.read()["phase"] == "RESTORED", (
        "self-exit is not a cleanup completion ACK"
    )
    assert probe.Recovery(host.scope).cleanup()["phase"] == "CLOSED"
    assert not (probe.ROOT / "active").exists(), (
        "closed recovery must release its active-owner claim"
    )
    assert probe.Recovery(host.scope).cleanup()["phase"] == "CLOSED"
    starts = [
        call for call in host.calls if call[0] == "start" and call[-1] == probe.AGENT
    ]
    assert starts == [("start", "--no-block", "--job-mode=fail", probe.AGENT)]
    assert all(
        call[0] in {"show", "start", "stop", "daemon-reload"} for call in host.calls
    ), "recovery must use only the bounded systemd command set"


def test_without_reboot_recovery_never_restarts_the_running_agent(
    host: HostHarness,
) -> None:
    host.arm()
    host.recovery.disable()
    host.now = host.scope["restore_at"]
    host.tick()
    assert host.agent_active and host.enable.exists()
    assert not [c for c in host.calls if c[0] == "start" and c[-1] == probe.AGENT]


def test_prepare_is_not_the_independent_arm_ack(host: HostHarness) -> None:
    host.recovery.prepare()
    with pytest.raises(probe.RecoveryError, match="independent"):
        host.recovery.disable()
    assert host.enable.exists(), (
        "preparation without an independent ACK must not disable the Agent"
    )
    with pytest.raises(probe.RecoveryError, match="cleanup only"):
        probe.Recovery(host.scope).prepare()
    assert probe.Recovery(host.scope).cleanup()["phase"] == "CLOSED"


@pytest.mark.parametrize(
    "defect",
    ["stale", "future", "wrong-boot", "reboot", "deadline", "missing-at", "nan"],
)
def test_disable_rejects_stale_or_unbound_ack_without_disabling(
    host: HostHarness, defect: str
) -> None:
    host.arm()
    ack = host.read()["ack"]
    if defect == "stale":
        host.now += probe.ACK_SECONDS + 1
    elif defect == "future":
        ack["at"] = host.now + 1
    elif defect == "wrong-boot":
        ack["boot_id"] = "other"
    elif defect == "reboot":
        host.boot = "other"
    elif defect == "deadline":
        host.now = host.scope["restore_at"] - 59
        ack["at"] = host.now
    elif defect == "missing-at":
        ack.pop("at")
    else:
        ack["at"] = float("nan")
    if defect == "nan":
        data = host.read()
        data["ack"] = ack
        host.recovery.state_path.write_text(json.dumps(data))
    else:
        host.update(ack=ack)
    before = len(host.calls)
    with pytest.raises(probe.RecoveryError, match="independent"):
        host.recovery.disable()
    assert host.enable.exists() and len(host.calls) == before


@pytest.mark.parametrize(
    "point",
    ["before-disable", "unlink-ack", "reload-ack", "restore-link-ack", "start-ack"],
)
def test_lost_acks_and_crashes_are_recoverable_without_replaying_disable(
    host: HostHarness, monkeypatch: pytest.MonkeyPatch, point: str
) -> None:
    host.arm()
    if point == "before-disable":
        host.update(phase="DISABLING", disable_started=True)
    elif point == "unlink-ack":
        original = probe.remove_owned

        def lost(path: Path, identity: dict[str, Any]) -> None:
            original(path, identity)
            if path == host.enable:
                raise TimeoutError("unlink ACK lost")

        with monkeypatch.context() as patch:
            patch.setattr(probe, "remove_owned", lost)
            with pytest.raises(TimeoutError):
                host.recovery.disable()
    elif point == "reload-ack":
        host.lose_ack = "daemon-reload"
        with pytest.raises(TimeoutError):
            host.recovery.disable()
    else:
        host.recovery.disable()
        host.reboot()
        if point == "restore-link-ack":
            original_publish = probe.publish

            def lost_link(
                source: Path, target: Path, identity: dict[str, Any], **kwargs: Any
            ) -> None:
                original_publish(source, target, identity, **kwargs)
                if target == host.enable:
                    raise TimeoutError("link ACK lost")

            with monkeypatch.context() as patch:
                patch.setattr(probe, "publish", lost_link)
                with pytest.raises(TimeoutError):
                    host.recovery.restore()
        else:
            host.lose_ack = probe.AGENT
            with pytest.raises(TimeoutError):
                host.recovery.restore()
    fresh = probe.Recovery(host.scope)
    assert fresh.cleanup()["phase"] == "CLOSED"
    assert host.enable.exists() and host.agent_active
    before = list(host.calls)
    with pytest.raises(probe.RecoveryError, match="independent"):
        probe.Recovery(host.scope).disable()
    assert host.calls == before
    assert len([c for c in host.calls if c[0] == "start" and c[-1] == probe.AGENT]) <= 1


def test_cleanup_tombstone_rejects_late_prepare_before_it_can_install_anything(
    host: HostHarness,
) -> None:
    assert host.recovery.cleanup()["phase"] == "CLOSED"
    with pytest.raises(probe.RecoveryError, match="cleanup only"):
        probe.Recovery(host.scope).prepare()
    assert host.calls == [] and host.enable.exists()
    assert not (probe.SYSTEMD / host.recovery.timer).exists(), (
        "a closed tombstone must not recreate a timer"
    )


@pytest.mark.parametrize(
    "target",
    ["agent", "enable", "claim", "helper", "timer", "dropin", "environment", "current"],
)
def test_recreated_or_changed_owners_are_never_overwritten(
    host: HostHarness, target: str
) -> None:
    host.arm()
    host.recovery.disable()
    targets = {
        "agent": host.fragment,
        "enable": host.enable,
        "claim": probe.ROOT / "active",
        "helper": host.recovery.directory / "recovery.py",
        "timer": probe.SYSTEMD / host.recovery.timer,
        "dropin": host.recovery.dropin,
        "environment": probe.AGENT_ENV,
        "current": probe.CURRENT,
    }
    path = targets[target]
    previous = path.read_bytes() if path.exists() else b"replacement"
    path.unlink(missing_ok=True)
    host.links.pop(path, None)
    path.write_bytes(previous)
    path.chmod(0o600)
    replacement = probe.file_identity(path)
    with pytest.raises(probe.RecoveryError):
        probe.Recovery(host.scope).cleanup()
    assert probe.file_identity(path) == replacement
    assert host.read()["phase"] != "CLOSED"


def test_other_run_cannot_take_the_claim_or_clean_the_original_units(
    host: HostHarness,
) -> None:
    host.arm()
    other = probe.Recovery({**host.scope, "owner": "owner-two"})
    with pytest.raises(probe.RecoveryError, match="another owner"):
        other.prepare()
    before = probe.file_identity(probe.SYSTEMD / host.recovery.timer)
    with pytest.raises(probe.RecoveryError, match="replaced"):
        other.cleanup()
    assert probe.file_identity(probe.SYSTEMD / host.recovery.timer) == before
    assert host.recovery.status()["phase"] == "ARMED"


@pytest.mark.parametrize(
    "resource", ["helper", "agent-start-bound", "service", "timer", "timer-boot-link"]
)
def test_install_link_ack_loss_has_durable_ownership_for_partial_cleanup(
    host: HostHarness, monkeypatch: pytest.MonkeyPatch, resource: str
) -> None:
    original = probe.publish

    def lost(
        source: Path, target: Path, identity: dict[str, Any], **kwargs: Any
    ) -> None:
        original(source, target, identity, **kwargs)
        if source.name == resource:
            raise TimeoutError("publish ACK lost")

    with monkeypatch.context() as patch:
        patch.setattr(probe, "publish", lost)
        with pytest.raises(TimeoutError):
            host.recovery.prepare()
    assert host.enable.exists(), (
        "lost publication ACK must not disable Agent boot activation"
    )
    assert probe.Recovery(host.scope).cleanup()["phase"] == "CLOSED"
    assert all(
        not Path(item["target"]).exists() for item in host.read()["resources"]
    ), "partial installation cleanup must remove every owned target"


def test_expired_recovery_never_issues_late_agent_start_and_cannot_report_cleanup(
    host: HostHarness,
) -> None:
    host.arm()
    host.recovery.disable()
    host.reboot()
    host.now = host.scope["expires_at"]
    assert host.tick()["phase"] == "EXPIRED"
    assert not host.enable.exists() and not host.agent_active
    assert not host.running[host.recovery.timer]
    with pytest.raises(probe.RecoveryError, match="restoration proof"):
        host.recovery.cleanup()
    assert host.read()["phase"] == "EXPIRED"
    assert not [c for c in host.calls if c[0] == "start" and c[-1] == probe.AGENT]


def test_agent_start_is_bounded_once_and_pending_jobs_prevent_cleanup(
    host: HostHarness,
) -> None:
    host.arm()
    host.recovery.disable()
    host.reboot()
    host.start_stuck = True
    with pytest.raises(probe.RecoveryError, match="not complete"):
        host.recovery.restore()
    with pytest.raises(probe.RecoveryError, match="not complete"):
        probe.Recovery(host.scope).cleanup()
    assert len([c for c in host.calls if c[0] == "start" and c[-1] == probe.AGENT]) == 1
    assert host.read()["phase"] == "RESTORING"
    host.agent_active = True
    host.agent_job = "41 /org/freedesktop/systemd1/job/41"
    with pytest.raises(probe.RecoveryError, match="not complete"):
        probe.Recovery(host.scope).cleanup()
    host.agent_job = ""
    assert probe.Recovery(host.scope).cleanup()["phase"] == "CLOSED"


def test_existing_unrelated_agent_job_is_not_replaced(host: HostHarness) -> None:
    host.arm()
    host.recovery.disable()
    host.reboot()
    host.agent_job = "7 /job/7"
    with pytest.raises(probe.RecoveryError, match="another Node Agent job"):
        host.recovery.restore()
    assert host.read()["start_requested"] is False
    assert not [c for c in host.calls if c[0] == "start" and c[-1] == probe.AGENT]


@pytest.mark.parametrize(
    "defect", ["invocation", "cgroup", "fragment", "active", "missing"]
)
def test_only_the_installed_independent_service_can_ack(
    host: HostHarness, defect: str
) -> None:
    host.recovery.prepare()
    host.monkeypatch.setenv("INVOCATION_ID", host.invocation)
    host.cgroup = f"0::/system.slice/{host.recovery.service}\n"
    if defect == "invocation":
        host.monkeypatch.setenv("INVOCATION_ID", "wrong")
    elif defect == "cgroup":
        host.cgroup = "0::/other-scope\n"
    elif defect == "fragment":
        host.overrides[host.recovery.service] = {
            "FragmentPath": "/run/transient.service"
        }
    elif defect == "active":
        host.running[host.recovery.service] = False
    else:
        host.monkeypatch.delenv("INVOCATION_ID")
    with pytest.raises(probe.RecoveryError, match="independent service"):
        host.recovery.tick()
    assert host.read()["phase"] == "INSTALLED"


@pytest.mark.parametrize(
    "defect",
    [
        "inactive",
        "not-waiting",
        "disabled",
        "transient",
        "unbounded-job",
        "unbounded-start",
    ],
)
def test_arm_verifies_persistent_timer_and_start_bounds(
    host: HostHarness, defect: str
) -> None:
    host.recovery.prepare()
    key, value = {
        "inactive": ("ActiveState", "inactive"),
        "not-waiting": ("SubState", "elapsed"),
        "disabled": ("UnitFileState", "disabled"),
        "transient": ("FragmentPath", "/run/unit"),
        "unbounded-job": ("JobTimeoutUSec", "infinity"),
        "unbounded-start": ("TimeoutStartUSec", "infinity"),
    }[defect]
    name = probe.AGENT if defect.startswith("unbounded") else host.recovery.timer
    host.overrides[name] = {key: value}
    with pytest.raises(probe.RecoveryError):
        host.tick()
    assert host.read()["phase"] == "INSTALLED" and host.enable.exists()


@pytest.mark.parametrize(
    "defect",
    [
        "disabled",
        "inactive",
        "job",
        "links",
        "relative-link",
        "boot",
        "window",
        "helper",
    ],
)
def test_prepare_preconditions_fail_before_agent_disable(
    host: HostHarness, defect: str
) -> None:
    if defect == "disabled":
        host.overrides[probe.AGENT] = {"UnitFileState": "disabled"}
    elif defect == "inactive":
        host.agent_active = False
    elif defect == "job":
        host.agent_job = "1 /job"
    elif defect == "links":
        host.symlink(str(host.fragment), probe.SYSTEMD / "alias.service")
    elif defect == "relative-link":
        host.links[host.enable] = "../" + probe.AGENT
    elif defect == "boot":
        host.boot = "changed"
    elif defect == "window":
        host.now = host.scope["restore_at"]
    else:
        host.scope["helper_sha256"] = "f" * 64
        host.recovery = probe.Recovery(host.scope)
    with pytest.raises(probe.RecoveryError):
        host.recovery.prepare()
    assert host.enable.exists(), (
        "failed preparation must leave Agent boot activation enabled"
    )
    assert not [c for c in host.calls if c[0] == "start" and c[-1] == probe.AGENT]


@pytest.mark.parametrize("defect", ["pin", "duplicate", "assignment", "unit"])
def test_runtime_binding_rejects_ambiguous_or_drifted_environment(
    host: HostHarness, defect: str
) -> None:
    if defect == "unit":
        host.overrides[probe.AGENT] = {"FragmentPath": "/other/unit"}
    else:
        with probe.AGENT_ENV.open("a") as handle:
            handle.write(
                {
                    "pin": "\nNODE_NAME=other",
                    "duplicate": "\nGPU_FAULT_NODE_CLUSTER_ID=cluster-test",
                    "assignment": "\nexport INVALID=value",
                }[defect]
            )
    with pytest.raises(probe.RecoveryError):
        host.recovery.prepare()
    assert host.enable.exists(), "runtime binding drift must not disable the Agent"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("extra", "field"),
        ("case_id", "other"),
        ("owner", ""),
        ("run_id", "bad\nid"),
        ("artifact_sha256", "not-a-digest"),
        ("restore_at", True),
        ("expires_at", 0),
        ("expires_at", 3000),
    ],
)
def test_binding_validation_is_strict(key: str, value: Any) -> None:
    with pytest.raises(probe.RecoveryError):
        probe.binding_key({**binding(), key: value})


def test_missing_and_malformed_journals_fail_closed(host: HostHarness) -> None:
    with pytest.raises(probe.RecoveryError, match="absent"):
        host.recovery.status()
    assert host.recovery.tick()["phase"] is None
    host.recovery.prepare()
    data = host.read()
    for altered in (
        {**data, "binding": {**host.scope, "owner": "other"}},
        {**data, "phase": "UNKNOWN"},
        {
            **data,
            "resources": [
                {"source": "/elsewhere", "target": "/etc/other", "identity": {}}
            ],
        },
        {**data, "disable_started": True, "resources": []},
    ):
        probe.atomic_state(host.recovery.state_path, altered)
        with pytest.raises(probe.RecoveryError):
            probe.Recovery(host.scope).cleanup()
    probe.atomic_state(host.recovery.state_path, data)
    assert host.recovery.cleanup()["phase"] == "CLOSED"


def test_failed_recovery_process_stop_preserves_unit_artifacts(
    host: HostHarness,
) -> None:
    host.arm()
    host.running[host.recovery.service] = True
    host.stop_stuck = True
    with pytest.raises(probe.RecoveryError, match="has not stopped"):
        host.recovery.cleanup()
    assert (probe.SYSTEMD / host.recovery.service).exists(), (
        "unproved recovery exit must retain the service unit"
    )
    assert (probe.SYSTEMD / host.recovery.timer).exists(), (
        "unproved recovery exit must retain the timer unit"
    )
    assert host.read()["phase"] == "RESTORED"
    host.stop_stuck = False
    assert probe.Recovery(host.scope).cleanup()["phase"] == "CLOSED"


def test_terminal_tick_resumes_retirement_after_crash_without_restarting_agent(
    host: HostHarness,
) -> None:
    host.arm()
    host.recovery.disable()
    host.recovery.restore()
    assert host.tick()["phase"] == "RESTORED"
    assert not (probe.SYSTEMD / host.recovery.timer).exists(), (
        "terminal recovery must retire its timer"
    )
    host.recovery.cleanup()
    before = list(host.calls)
    assert host.tick()["phase"] == "CLOSED"
    assert host.calls == before


def test_short_remaining_window_cannot_enqueue_an_unbounded_late_start(
    host: HostHarness,
) -> None:
    host.arm()
    host.recovery.disable()
    host.reboot()
    host.now = host.scope["expires_at"] - probe.START_SECONDS
    with pytest.raises(probe.RecoveryError, match="insufficient"):
        host.recovery.restore()
    assert host.read()["phase"] == "EXPIRED"
    assert not host.agent_active, "insufficient recovery time must not start the Agent"


def test_private_record_io_and_create_only_files(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    path = tmp_path / "state.json"
    probe.atomic_state(path, {"value": 1})
    assert probe.read_record(path) == {"value": 1}
    probe.atomic_state(path, {"value": 2})
    assert probe.read_record(path) == {"value": 2}
    assert not list(tmp_path.glob(".state-*")), (
        "atomic updates must leave no temporary state files"
    )
    path.chmod(0o644)
    with pytest.raises(probe.RecoveryError, match="private"):
        probe.read_record(path)
    path.chmod(0o600)
    path.write_text("[]")
    with pytest.raises(probe.RecoveryError, match="malformed"):
        probe.read_record(path)
    identity = probe.file_identity(path)
    target = tmp_path / "copy"
    probe.publish(path, target, identity)
    probe.publish(path, target, identity)
    assert probe.file_identity(target) == identity
    target.unlink()
    target.write_text("unrelated")
    target.chmod(0o600)
    with pytest.raises(probe.RecoveryError, match="another owner"):
        probe.publish(path, target, identity)
    with pytest.raises(probe.RecoveryError, match="replaced"):
        probe.remove_owned(target, identity)
    assert target.read_text() == "unrelated"
    probe.remove_owned(path, identity)
    probe.remove_owned(path, identity)
    with pytest.raises(probe.RecoveryError, match="source"):
        probe.publish(path, target, identity)


def test_systemctl_transport_is_bounded_and_never_echoes_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "ActiveState=active\nignored\n", "")

    monkeypatch.setattr(probe.subprocess, "run", run)
    assert probe.unit_state(probe.AGENT) == {"ActiveState": "active"}
    assert calls[0][0][0] == "/bin/systemctl"
    assert calls[0][1]["timeout"] == 20
    assert set(calls[0][1]["env"]) == {"PATH", "LC_ALL"}
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "sensitive output"),
    )
    with pytest.raises(probe.RecoveryError, match="withheld") as error:
        probe.systemctl("show", probe.AGENT)
    assert "sensitive output" not in str(error.value)


def test_main_handles_bound_commands_and_rejects_unbound_tick(
    host: HostHarness, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["--binding", json.dumps(host.scope)]
    assert probe.main(["prepare", *args]) == 0
    assert json.loads(capsys.readouterr().out)["phase"] == "INSTALLED"
    assert probe.main(["tick", *args]) == 1
    assert probe.main(["status", "--key", host.recovery.key]) == 1
    assert probe.main(["tick", "--key", "../outside"]) == 1
    host.monkeypatch.setenv("INVOCATION_ID", host.invocation)
    host.cgroup = f"0::/system.slice/{host.recovery.service}\n"
    assert probe.main(["tick", "--key", host.recovery.key]) == 0
    reports = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert all(item.get("recovery_required") is True for item in reports[:-1]), (
        "unbound recovery commands must report unresolved recovery"
    )
    assert reports[-1]["phase"] == "ARMED"
    assert probe.main(["cleanup", *args]) == 0
    assert json.loads(capsys.readouterr().out)["phase"] == "CLOSED"


def test_nonregular_file_and_unsafe_directory_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(probe.RecoveryError, match="regular file"):
        probe.file_identity(tmp_path)
    tmp_path.chmod(0o777)
    with pytest.raises(probe.RecoveryError, match="directory"):
        probe.require_system_directory(tmp_path)
    path = tmp_path / "unsafe"
    path.write_text("data")
    path.chmod(0o666)
    with pytest.raises(probe.RecoveryError, match="not owned"):
        probe.file_identity(path)


def test_tick_accepts_the_timer_running_its_own_unit(host: HostHarness) -> None:
    """While the independent tick executes, systemd reports the timer's
    SubState as ``running`` (verified on a HyperPod node); the tick must not
    refuse itself as "not armed" -- that refusal starved every ACK live."""

    host.recovery.prepare()
    host.overrides[host.recovery.timer] = {"SubState": "running"}
    report = host.tick()
    assert report["phase"] == "ARMED", report
    assert report["ack"] and report["ack"]["boot_id"], report
    assert report["last_tick_error"] is None, report


def test_a_refused_tick_leaves_its_reason_for_status(host: HostHarness) -> None:
    """The service discards stdout/stderr, so the reason a tick refused must
    survive in the journal where the runner's ``status`` poll can read it."""

    host.recovery.prepare()
    host.overrides[host.recovery.timer] = {"SubState": "elapsed"}
    with pytest.raises(probe.RecoveryError):
        host.tick()
    assert host.read()["last_tick_error"] == "persistent recovery timer is not armed"
    # Once the timer is healthy again, ``status`` still reports what refused.
    del host.overrides[host.recovery.timer]
    assert (
        host.recovery.status()["last_tick_error"]
        == "persistent recovery timer is not armed"
    )
    assert host.tick()["phase"] == "ARMED"


def _agent_starts(host: HostHarness) -> list[tuple[str, ...]]:
    return [c for c in host.calls if c[0] == "start" and c[-1] == probe.AGENT]


def _spy_systemctl(host: HostHarness) -> list[tuple[tuple[str, ...], bool]]:
    """Every systemctl call paired with whether the owned drop-in still existed."""

    seen: list[tuple[tuple[str, ...], bool]] = []
    original = host.systemctl

    def spy(*args: str) -> dict[str, str]:
        seen.append((args, host.recovery.dropin.exists()))
        return original(*args)

    host.monkeypatch.setattr(probe, "systemctl", spy)
    return seen


def test_new_boot_accepts_the_installer_restored_agent_and_retires(
    host: HostHarness,
) -> None:
    """Live: the product's reboot brought the node back, the node-installer
    reconciler re-enabled the Agent under a new link inode and rewrote the
    unit, and every tick on the new boot refused with "Node Agent or host
    incarnation changed" while the 45 s start bound stayed installed. A new
    boot cannot present the armed identity; the installer's Agent is the
    restoration and the probe's only remaining job is to retire itself."""

    host.arm()
    host.recovery.disable()
    host.reboot(renumber_devices=True)
    host.installer_reinstall()
    installer_link = probe.file_identity(host.enable)
    seen = _spy_systemctl(host)
    host.now = host.scope["restore_at"] - 60
    report = host.tick()
    assert report["phase"] == "RESTORED", report
    assert report["restored_by"] == "reboot-installer", report
    assert report["boot_id_observed"] == "boot-after", report
    assert report["retired"] is True, report
    assert probe.file_identity(host.enable) == installer_link, (
        "the installer's enable link is not the probe's to touch"
    )
    assert not host.recovery.dropin.exists(), (
        "the owned start bound must leave with the probe"
    )
    assert not (probe.SYSTEMD / host.recovery.timer).exists(), (
        "the installer-restored Agent needs no persistent timer"
    )
    assert not (probe.SYSTEMD / host.recovery.service).exists(), (
        "the recovery service unit must be retired"
    )
    assert host.systemctl("show", probe.AGENT)["JobTimeoutUSec"] == "infinity"
    assert (("daemon-reload",), False) in seen, (
        "systemd must reload after the drop-in is gone, before any later Agent start"
    )
    assert _agent_starts(host) == [], "an installer-restored Agent is never restarted"
    closed = probe.Recovery(host.scope).cleanup()
    assert closed["phase"] == "CLOSED" and closed["restored_by"] == "reboot-installer"
    assert probe.file_identity(host.enable) == installer_link, (
        "cleanup must not replace the installer's link with the saved inode"
    )


def test_new_boot_restores_a_missing_link_from_the_saved_copy_at_the_deadline(
    host: HostHarness,
) -> None:
    """The installer rewrote the unit on the new boot but did not re-enable it:
    the saved inode is still the only enable link this recovery may publish,
    and only from the deadline on, exactly as on the original boot."""

    original = probe.file_identity(host.enable)
    host.arm()
    host.recovery.disable()
    host.reboot(renumber_devices=True)
    host.installer_reinstall(enable=False)
    host.now = host.scope["restore_at"] - 1
    assert host.tick()["phase"] == "DISABLED"
    assert not host.enable.exists(), "no enable before the deadline on any boot"
    host.now += 1
    report = host.tick()
    assert report["phase"] == "RESTORED", report
    assert report["restored_by"] == "probe", report
    assert report["restore_reason"] == "deadline", report
    assert report["boot_id_observed"] == "boot-after" and report["retired"] is True
    saved = probe.file_identity(host.enable)
    assert {k: v for k, v in saved.items() if k != "dev"} == {
        k: v for k, v in original.items() if k != "dev"
    }, "the restored link must be the saved inode, not a fresh one"
    assert host.agent_active, "the deadline restore must start the Agent"
    assert _agent_starts(host) == [
        ("start", "--no-block", "--job-mode=fail", probe.AGENT)
    ]
    assert not host.recovery.dropin.exists(), "retirement must release the drop-in"
    assert probe.Recovery(host.scope).cleanup()["phase"] == "CLOSED"


def test_new_boot_expiry_retires_without_a_late_start_despite_identity_drift(
    host: HostHarness,
) -> None:
    host.arm()
    host.recovery.disable()
    host.reboot(renumber_devices=True)
    host.installer_reinstall(enable=False)
    host.now = host.scope["expires_at"]
    report = host.tick()
    assert report["phase"] == "EXPIRED", report
    assert report["retired"] is True and report["boot_id_observed"] == "boot-after"
    assert report["expired_at"] == host.now, report
    assert not host.enable.exists() and not host.agent_active
    assert _agent_starts(host) == [], "expiry must never issue a late Agent start"
    assert not (probe.SYSTEMD / host.recovery.timer).exists(), (
        "expiry must retire the persistent timer on the new boot"
    )
    assert not host.recovery.dropin.exists(), "expiry must release the owned drop-in"
    with pytest.raises(probe.RecoveryError, match="restoration proof"):
        probe.Recovery(host.scope).cleanup()


def test_new_boot_expired_recovery_closes_once_the_installer_brought_the_agent_back(
    host: HostHarness,
) -> None:
    host.arm()
    host.recovery.disable()
    host.reboot()
    host.now = host.scope["expires_at"]
    assert host.tick()["phase"] == "EXPIRED"
    host.installer_reinstall()
    report = probe.Recovery(host.scope).cleanup()
    assert report["phase"] == "CLOSED", report
    assert report["restored_by"] == "reboot-installer", report
    assert report["expired_at"] == host.scope["expires_at"], report
    assert _agent_starts(host) == []


def test_original_boot_keeps_refusing_identity_drift_and_recreated_links(
    host: HostHarness,
) -> None:
    host.arm()
    host.recovery.disable()
    host.installer_reinstall(enable=False)
    with pytest.raises(probe.RecoveryError, match="incarnation changed"):
        host.recovery.restore()
    assert not host.enable.exists(), "drifted identity must not publish the link"
    assert host.read()["phase"] == "RESTORING"
    host.fragment.write_text("[Service]\nType=simple\n")
    host.symlink(str(host.fragment), host.enable)
    recreated = probe.file_identity(host.enable)
    with pytest.raises(probe.RecoveryError):
        probe.Recovery(host.scope).cleanup()
    assert probe.file_identity(host.enable) == recreated, (
        "a recreated link on the original boot is never adopted or replaced"
    )
    assert host.read()["phase"] != "CLOSED"


def test_new_boot_foreign_enable_link_is_never_touched(host: HostHarness) -> None:
    host.arm()
    host.recovery.disable()
    host.reboot()
    host.symlink(str(probe.SYSTEMD / "other.service"), host.enable)
    foreign = probe.file_identity(host.enable)
    host.now = host.scope["restore_at"]
    with pytest.raises(probe.RecoveryError, match="unknown owner"):
        host.tick()
    assert host.read()["last_tick_error"] == (
        "Node Agent enable link has an unknown owner; manual recovery required"
    )
    assert probe.file_identity(host.enable) == foreign
    assert (probe.SYSTEMD / host.recovery.timer).exists(), (
        "an unresolved recovery keeps its timer until expiry"
    )
    host.now = host.scope["expires_at"]
    assert host.tick()["phase"] == "EXPIRED"
    assert probe.file_identity(host.enable) == foreign
    assert not (probe.SYSTEMD / host.recovery.timer).exists(), (
        "expiry retires the probe's own timer even beside a foreign link"
    )
    with pytest.raises(probe.RecoveryError, match="restoration proof"):
        probe.Recovery(host.scope).cleanup()
    assert _agent_starts(host) == []


def test_new_boot_before_disable_retires_the_armed_recovery(host: HostHarness) -> None:
    host.arm()
    host.reboot()
    with pytest.raises(probe.RecoveryError, match="independent"):
        host.recovery.disable()
    report = host.tick()
    assert report["phase"] == "RESTORED" and report["restored_by"] is None, report
    assert report["retired"] is True and host.enable.exists()
    assert probe.Recovery(host.scope).cleanup()["phase"] == "CLOSED"


def test_new_boot_installer_agent_still_starting_is_retried_not_adopted(
    host: HostHarness,
) -> None:
    host.arm()
    host.recovery.disable()
    host.reboot()
    host.installer_reinstall()
    host.agent_active = False
    with pytest.raises(probe.RecoveryError, match="not enabled, active and idle"):
        host.tick()
    assert host.read()["phase"] == "RESTORING"
    assert host.recovery.dropin.exists(), "an unproven Agent keeps the start bound"
    host.agent_active = True
    report = host.tick()
    assert report["phase"] == "RESTORED" and report["restored_by"] == "reboot-installer"
    assert _agent_starts(host) == []


def test_new_boot_report_carries_the_boot_and_retirement_receipts(
    host: HostHarness,
) -> None:
    host.arm()
    report = host.recovery.status()
    assert report["boot_id_observed"] is None and report["retired"] is False
    assert report["restored_by"] is None and report["expired_at"] is None
    host.recovery.disable()
    host.reboot()
    host.installer_reinstall()
    host.tick()
    status = probe.Recovery(host.scope).status()
    assert status["boot_id_observed"] == "boot-after"
    assert status["restored_by"] == "reboot-installer"
    assert status["retired"] is True


def test_prepare_reclaims_the_claim_of_a_recovery_dead_on_another_boot(
    host: HostHarness,
) -> None:
    """DESTR-014 attempt 8 was fenced by attempt 4's record: armed on the
    pre-reboot boot, stuck RESTORING after the installer replaced its units,
    still holding ``active``. A record whose boot is gone and whose resource
    targets are all absent can never resume, so its claim is released and the
    record is left as forensics."""

    host.boot = "boot-older"
    old = probe.Recovery({**host.scope, "owner": "owner-old", "boot_id": "boot-older"})
    old.prepare()
    for resource in json.loads(old.state_path.read_text())["resources"]:
        target = Path(resource["target"])
        host.links.pop(target, None)
        target.unlink(missing_ok=True)
    host.boot = "boot-before"

    report = host.recovery.prepare()

    assert report["reclaimed_stale_active"] == old.key, report
    assert probe.file_identity(probe.ROOT / "active") == host.read()["claim"], (
        "the new claim must own active after the dead one is released"
    )
    assert old.state_path.exists(), "the dead record stays as forensics"


def test_prepare_keeps_the_fence_while_a_stale_recovery_still_owns_units(
    host: HostHarness,
) -> None:
    host.boot = "boot-older"
    old = probe.Recovery({**host.scope, "owner": "owner-old", "boot_id": "boot-older"})
    old.prepare()
    before = json.loads(old.state_path.read_text())["claim"]
    host.boot = "boot-before"

    with pytest.raises(probe.RecoveryError, match="manual recovery"):
        host.recovery.prepare()

    assert probe.file_identity(probe.ROOT / "active") == before, (
        "a stale recovery whose units are still installed keeps its claim"
    )
