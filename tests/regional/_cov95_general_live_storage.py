"""A fake GPU node for the GF-GENERAL-LIVE storage probe, at the probe's own edges.

``FakeStorageHost`` answers ``systemctl``/``systemd-run``, the device tools
(``dmsetup``, ``losetup``, ``blockdev``, ``fallocate``, ``mkfs.*``, ``mount``,
``umount``, ``modprobe``) and ``nsenter`` at the probe's ``subprocess``
boundary; it keeps two mount tables -- PID 1's (``HOST_MOUNTINFO``) and the
probe's own (``/proc/self/mountinfo``) -- and a fake ``sysfs``, and answers
``os.statvfs`` per path from a table (a path that does not exist raises like the
kernel). Unlike the 006 fixture's fake it leaves the probe's ``emit``,
``tool_present``, ``backing_root``, ``mounts`` parser and ``foreign_holders``
real, so a test sees the probe's own JSON and guards: the host's ``/proc``
(PID 1's namespace links, other processes' mount tables) is a fake tree the
probe's ``Path("/proc")`` and ``os.readlink("/proc/...")`` resolve to.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_storage_probe as probe

RUN_ID = "gl004-cov-storage-a1-deadbeef"
DIGEST = probe.run_digest(RUN_ID)
NS = list(probe.HOST_NAMESPACES)
#: ``statvfs`` of a path the table does not name: 1 GiB, half used.
DEFAULT_FS = (262144, 131072, 65536, 65000, 0)
#: The namespaces of PID 1 in the fake ``/proc``.
HOST_MNT = "mnt:[4026531841]"
HOST_USER = "user:[4026531837]"


class FakeClock:
    """``time`` as the probe sees it; ``step`` seconds pass on every reading."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.step = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        self.now += self.step
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class FakeStorageHost:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.clock = FakeClock()
        self.calls: list[list[str]] = []
        self.timeouts: list[float] = []
        self.timers: set[str] = set()
        self.services: dict[str, str] = {}
        self.started: dict[str, list[str]] = {}
        #: ``MainPID`` of a unit (``<name>.service``) when it is not 0.
        self.main_pids: dict[str, str] = {}
        self.mappings: dict[str, int] = {}
        self.open_counts: dict[str, int] = {}
        self.loops: dict[str, str] = {}
        #: (source, target) in PID 1's mount namespace and in the probe's own.
        self.host_mounts: list[tuple[str, str]] = []
        self.own_mounts: list[tuple[str, str]] = []
        #: Targets a plain ``umount`` refuses as busy (a lazy one still works).
        self.busy: set[str] = set()
        #: Tools that fail with this return code.
        self.failing: dict[str, int] = {}
        self.timer_starts = True
        self.loadable = True
        self.found = {"nsenter", "dmsetup", "losetup"}
        #: What ``losetup --find --show`` prints / ``losetup -J -l`` answers.
        self.losetup_answer: str | None = None
        self.losetup_listing: str | None = None
        #: path -> (blocks, available blocks, inodes, free inodes, flags).
        self.filesystems: dict[str, tuple[int, int, int, int, int]] = {}
        self.synced = 0
        self.minor = 10
        self.sys_block = root / "sys-block"
        self.sys_module = root / "sys-module"
        self.host_mountinfo = root / "host-mountinfo"
        self.own_mountinfo = root / "self-mountinfo"
        self.backing = root / "var-lib" / "gpu-fault-acceptance"
        self.proc = root / "proc"
        self.sys_block.mkdir()
        self.sys_module.mkdir()
        self.backing.parent.mkdir(parents=True)
        self.add_process(1, mnt=HOST_MNT, comm="systemd")
        self.write_tables()

    def add_process(
        self,
        pid: int,
        *,
        mnt: str,
        user: str = HOST_USER,
        comm: str = "pause",
        mounts: tuple[tuple[str, str], ...] = (),
    ) -> Path:
        """A ``/proc/<pid>`` with its namespace links, ``comm`` and mount table."""

        directory = self.proc / str(pid)
        (directory / "ns").mkdir(parents=True)
        (directory / "ns" / "mnt").symlink_to(mnt)
        (directory / "ns" / "user").symlink_to(user)
        (directory / "comm").write_text(comm + "\n", encoding="utf-8")
        lines = [
            f"{index} 22 {devno} / {target} rw,relatime - ext4 /dev/dm rw"
            for index, (devno, target) in enumerate(mounts, start=60)
        ]
        (directory / "mountinfo").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return directory

    # ------------------------------------------------------------------ state
    def write_tables(self) -> None:
        for path, table in (
            (self.host_mountinfo, self.host_mounts),
            (self.own_mountinfo, self.own_mounts),
        ):
            lines = [
                "22 1 259:1 / / rw,relatime shared:1 - xfs /dev/nvme0n1p1 rw",
                "a line without the optional-fields separator",
            ]
            for index, (source, target) in enumerate(table, start=40):
                minor = self.mappings.get(source.rpartition("/")[2], index)
                lines.append(
                    f"{index} 22 253:{minor} / {target} rw,relatime "
                    f"shared:{index} - ext4 {source} rw"
                )
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def create_mapping(self, name: str) -> int:
        self.minor += 1
        self.mappings[name] = self.minor
        node = self.sys_block / f"dm-{self.minor}" / "dm"
        node.mkdir(parents=True)
        (node / "name").write_text(name + "\n", encoding="utf-8")
        return self.minor

    def mount_on_host(self, source: str, target: str) -> None:
        self.host_mounts.append((source, target))
        self.write_tables()

    def mutations(self) -> list[list[str]]:
        """Every call except ``systemctl show``/``list-units`` reads."""

        reads = (["systemctl", "show"], ["systemctl", "list-units"])
        return [call for call in self.calls if call[:2] not in reads]

    # -------------------------------------------------------------- os / time
    def statvfs(self, path: Any) -> os.statvfs_result:
        text = str(path)
        if not Path(text).exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), text)
        blocks, available, files, free, flag = self.filesystems.get(text, DEFAULT_FS)
        return os.statvfs_result(
            (4096, 4096, blocks, available, available, files, free, free, flag, 255)
        )

    def sync(self) -> None:
        self.synced += 1

    def readlink(self, path: Any) -> str:
        """``os.readlink`` with the host's ``/proc`` served from the fake one."""

        text = os.fspath(path)
        if text.startswith("/proc/"):
            text = str(self.proc) + text[len("/proc") :]
        return os.readlink(text)

    def path(self, *parts: Any) -> Path:
        """``Path`` as the probe builds it, with ``/proc`` the fake one."""

        return self.proc if parts == ("/proc",) else Path(*parts)

    # ------------------------------------------------------------- commands
    def run(
        self,
        command: list[str],
        *,
        input: str | None,
        text: bool,
        stdout: int,
        stderr: int,
        timeout: float,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert (text, stdout, stderr, check) == (
            True,
            subprocess.PIPE,
            subprocess.PIPE,
            False,
        ), "the probe captures text output and judges the return code itself"
        command = list(command)
        self.calls.append(command)
        self.timeouts.append(timeout)
        hosted = command[: len(NS)] == NS
        inner = command[len(NS) :] if hosted else command
        tool, arguments = inner[0], inner[1:]
        if tool in self.failing:
            code, out = self.failing[tool], ""
        else:
            code, out = self.answer(tool, arguments, hosted)
        error = f"{tool}: refused by the fake host\n" if code else ""
        return subprocess.CompletedProcess(command, code, stdout=out, stderr=error)

    def answer(self, tool: str, arguments: list[str], hosted: bool) -> tuple[int, str]:
        if tool == "systemctl":
            return self.systemctl(arguments)
        if tool == "systemd-run":
            unit = next(item for item in arguments if item.startswith("--unit="))[7:]
            if any(item.startswith("--on-active=") for item in arguments):
                if self.timer_starts:
                    self.timers.add(unit)
            else:
                self.services[unit] = "active"
            self.started[unit] = arguments
            return 0, ""
        if tool == "dmsetup":
            return self.dmsetup(arguments)
        if tool == "losetup":
            return self.losetup(arguments)
        if tool == "mount":
            if arguments[0] == "-o":
                return 0, ""  # ``mount -o remount,<mode> <target>``
            table = self.host_mounts if hosted else self.own_mounts
            table.append((arguments[-2], arguments[-1]))
            self.write_tables()
            return 0, ""
        if tool == "umount":
            return self.umount(arguments, hosted)
        if tool == "modprobe":
            if arguments[0] == "-r":
                shutil.rmtree(self.sys_module / arguments[1], ignore_errors=True)
            elif self.loadable:
                (self.sys_module / arguments[0]).mkdir(exist_ok=True)
            return 0, ""
        if tool == "fallocate":
            Path(arguments[-1]).write_bytes(b"")
            return 0, ""
        if tool == "blockdev":
            return 0, "4194304\n"
        if tool == "nsenter":
            return self.evict(arguments)
        if tool in {"mkfs.ext4", "mkfs.vfat"}:
            return 0, ""
        raise AssertionError(f"unexpected host command {[tool, *arguments]}")

    def systemctl(self, arguments: list[str]) -> tuple[int, str]:
        verb = arguments[0]
        if verb == "show":
            unit = arguments[1]
            base, _, suffix = unit.rpartition(".")
            if suffix == "timer":
                state = "active" if base in self.timers else "inactive"
            else:
                state = self.services.get(base, "inactive")
            load = (
                "loaded"
                if base in self.timers or base in self.services
                else "not-found"
            )
            pid = self.main_pids.get(unit, "0")
            return 0, (
                f"LoadState={load}\nActiveState={state}\nSubState=dead\n"
                f"MainPID={pid}\nwithout a separator\n"
            )
        if verb == "stop":
            base, _, suffix = arguments[1].rpartition(".")
            if suffix == "timer":
                self.timers.discard(base)
            else:
                self.services.pop(base, None)  # a ``--collect`` unit is gone
            return 0, ""
        if verb == "reset-failed":
            return 0, ""
        if verb == "list-units":
            prefix = arguments[-1].rstrip("*")
            listed = sorted(unit for unit in self.services if unit.startswith(prefix))
            return 0, "".join(
                f"{unit}.service loaded active running\n" for unit in listed
            )
        raise AssertionError(f"unexpected systemctl {arguments}")

    def dmsetup(self, arguments: list[str]) -> tuple[int, str]:
        verb, name = arguments[0], arguments[-1]
        if verb == "ls":
            listing = "".join(
                f"{key}\t(253:{value})\n" for key, value in self.mappings.items()
            )
            return 0, listing or "No devices found\n"
        if verb == "create":
            self.create_mapping(arguments[1])
            return 0, ""
        if verb == "remove":
            if name not in self.mappings:
                return 1, ""
            shutil.rmtree(self.sys_block / f"dm-{self.mappings.pop(name)}")
            return 0, ""
        if verb == "reload":
            name = arguments[1]
        if verb == "info" and "-o" in arguments:
            if name not in self.mappings:
                return 1, ""
            if arguments[arguments.index("-o") + 1] == "major,minor":
                return 0, f"253:{self.mappings[name]}\n"
            return 0, f"{self.open_counts.get(name, 0)}\n"
        if verb in {"resume", "suspend", "info", "reload"}:
            return (
                (0, f"Name: {name}\nState: ACTIVE\n")
                if name in self.mappings
                else (1, "")
            )
        raise AssertionError(f"unexpected dmsetup {arguments}")

    def losetup(self, arguments: list[str]) -> tuple[int, str]:
        if arguments[:2] == ["--find", "--show"]:
            device = self.losetup_answer or f"/dev/loop{len(self.loops)}"
            self.loops[device] = arguments[2]
            return 0, device + "\n"
        if arguments == ["-J", "-l"]:
            if self.losetup_listing is not None:
                return 0, self.losetup_listing
            devices = [
                {"name": key, "back-file": value} for key, value in self.loops.items()
            ]
            return 0, json.dumps({"loopdevices": devices})
        if arguments[0] == "-d":
            self.loops.pop(arguments[1], None)
            return 0, ""
        raise AssertionError(f"unexpected losetup {arguments}")

    def evict(self, arguments: list[str]) -> tuple[int, str]:
        """``nsenter --target PID --mount -- umount -l PATH`` of the holder sweep.

        The copy goes from every process of that mount namespace; one owned by
        another user namespace is locked and the unmount fails.
        """

        directory = self.proc / arguments[arguments.index("--target") + 1]
        assert arguments[-3:-1] == ["umount", "-l"], arguments
        if os.readlink(directory / "ns" / "user") != HOST_USER:
            return 32, ""
        namespace = os.readlink(directory / "ns" / "mnt")
        for other in self.proc.iterdir():
            try:
                if os.readlink(other / "ns" / "mnt") != namespace:
                    continue
            except OSError:
                continue
            table = other / "mountinfo"
            kept = [
                line
                for line in table.read_text(encoding="utf-8").splitlines()
                if line.split()[4:5] != [arguments[-1]]
            ]
            table.write_text("\n".join(kept) + "\n", encoding="utf-8")
        return 0, ""

    def umount(self, arguments: list[str], hosted: bool) -> tuple[int, str]:
        target = arguments[-1]
        if target in self.busy and "-l" not in arguments:
            return 32, ""
        table = self.host_mounts if hosted else self.own_mounts
        for index in reversed(range(len(table))):
            if table[index][1] == target:
                del table[index]
                self.write_tables()
                return 0, ""
        return 32, ""


def install_host(monkeypatch: pytest.MonkeyPatch, root: Path) -> FakeStorageHost:
    """Point the storage probe at ``root`` and a ``FakeStorageHost``."""

    host = FakeStorageHost(root)
    real_mounts = probe.mounts

    def mounts(table: Path = host.own_mountinfo) -> list[dict[str, str]]:
        # The probe's own ``/proc/self/mountinfo``, parsed by the probe.
        return real_mounts(table)

    monkeypatch.setattr(probe, "STATE_ROOT", root / "state")
    monkeypatch.setattr(probe, "MOUNT_ROOT", root / "mnt")
    monkeypatch.setattr(probe, "BACKING_ROOTS", (host.backing,))
    monkeypatch.setattr(probe, "HOST_MOUNTINFO", host.host_mountinfo)
    monkeypatch.setattr(probe, "SYS_BLOCK", host.sys_block)
    monkeypatch.setattr(probe, "SYS_MODULE", host.sys_module)
    monkeypatch.setattr(probe, "DEADLINE", [])
    monkeypatch.setattr(probe, "mounts", mounts)
    monkeypatch.setattr(
        probe, "subprocess", SimpleNamespace(run=host.run, PIPE=subprocess.PIPE)
    )
    monkeypatch.setattr(
        probe,
        "shutil",
        SimpleNamespace(
            which=lambda name: f"/usr/sbin/{name}" if name in host.found else None
        ),
    )
    overrides = {"statvfs": host.statvfs, "sync": host.sync, "readlink": host.readlink}
    monkeypatch.setattr(probe, "os", SimpleNamespace(**{**vars(os), **overrides}))
    monkeypatch.setattr(probe, "Path", host.path)
    monkeypatch.setattr(
        probe,
        "time",
        SimpleNamespace(monotonic=host.clock.monotonic, sleep=host.clock.sleep),
    )
    return host


def add_ready_device(
    host: FakeStorageHost,
    name: str = "io",
    *,
    fstype: str = "ext4",
    mounted: bool = False,
) -> dict[str, Any]:
    """A device record ``dm-setup`` finished, with its mapping, loop and mount."""

    dm = probe.dm_name(RUN_ID, name)
    target = probe.mountpoint(RUN_ID, name)
    target.mkdir(parents=True)
    directory = host.backing / f"gfrs-live-{DIGEST}"
    directory.mkdir(parents=True, exist_ok=True)
    backing = directory / f"{name}.img"
    backing.write_bytes(b"")
    loop = f"/dev/loop{len(host.loops)}"
    host.loops[loop] = str(backing)
    minor = host.create_mapping(dm)
    record: dict[str, Any] = {
        "dm": dm,
        "dm_path": f"/dev/mapper/{dm}",
        "setup_unit": probe.unit_name(RUN_ID, probe.setup_kind(name)),
        "backing": str(backing),
        "mountpoint": str(target),
        "loop": loop,
        "delay_ms": 0,
        "size_mib": 256,
        "fstype": fstype,
        "target": "data",
        "mounted": mounted,
        "phase": "ready",
        "step": None,
        "sectors": 4194304,
        "device": f"dm-{minor}",
        "fat_sectors": [8, 64] if fstype == "vfat" else None,
    }
    state = probe.load_state(RUN_ID)
    state["devices"][name] = record
    probe.save_state(RUN_ID, state)
    if mounted:
        host.mount_on_host(record["dm_path"], str(target))
    return record
