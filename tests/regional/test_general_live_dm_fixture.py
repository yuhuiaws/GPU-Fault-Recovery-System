"""The storage probe's GF-GENERAL-LIVE-006 device fixture, on a fake host.

A recording fake host answers systemd, ``nsenter``, ``dmsetup``, ``losetup``,
``blockdev``, ``fallocate``, ``mkfs.ext4``, ``mount``/``umount`` and
``modprobe`` at the probe's ``subprocess`` boundary -- over a fake sysfs, a fake
host mount table and a fake monotonic clock -- so the probe's own ``run`` and
its call budget run for real.

The fake keeps the property behind the 2026-10-25 live finding (``dm-create``
``TimeoutExpired`` on both EKS nodes before any product evidence): a ``dmsetup``
create/resume/remove issued outside PID 1's IPC namespace never receives the
host udev's cookie (``95-dm-notify.rules`` runs ``dmsetup udevcomplete`` in the
host IPC namespace, and the probe Pod is not hostIPC), so it hangs until its
timeout. The tests pin the fix: the call only starts a bounded setup unit,
every device command joins PID 1's IPC and mount namespaces, no call outlives
its budget, a failure names its step and site, and the cleanup -- the deadman's
command too -- undoes whatever the setup reached.
"""

from __future__ import annotations

import json
import re
import shutil
import struct
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_storage_probe as storep

RUN_ID = "gl006-short-flush-a1-deadbeef"
NS = list(storep.HOST_NAMESPACES)
DIGEST = storep.run_digest(RUN_ID)
DM = f"gfrs-{DIGEST}-io"
SETUP_UNIT = f"gfrs-{DIGEST}-io-io-setup"
WRITER_UNIT = f"gfrs-{DIGEST}-io-io"
DEADMAN = f"gfrs-{DIGEST}-sdeadman-dm-io"
#: The dmsetup verbs libdevmapper synchronises with udev (create, resume and
#: remove send a uevent carrying the cookie).
UDEV_SYNCED = frozenset({"create", "resume", "remove"})
DEVICE_TOOLS = frozenset(
    {
        "fallocate",
        "losetup",
        "blockdev",
        "dmsetup",
        "mkfs.ext4",
        "mkfs.vfat",
        "mount",
        "umount",
    }
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeDmHost:
    """One node's device tools, answered at the probe's ``subprocess`` boundary."""

    def __init__(self, root: Path, clock: FakeClock) -> None:
        self.clock = clock
        self.calls: list[list[str]] = []
        self.timeouts: list[float] = []
        self.services: dict[str, str] = {}
        self.timers: set[str] = set()
        self.started: dict[str, list[str]] = {}
        self.mappings: dict[str, int] = {}
        self.tables: dict[str, str] = {}
        self.loops: dict[str, str] = {}
        self.mounts: dict[str, str] = {}
        #: Mounts stacked on one target (``mounts`` keeps the top one), the
        #: mappings whose removal was deferred, and the foreign mount
        #: namespaces the probe's sweep finds (``evicted`` by ``nsenter``).
        self.stack: dict[str, list[str]] = {}
        self.deferred: list[str] = []
        self.holders: list[dict[str, Any]] = []
        self.sys_block = root / "sys-block"
        self.sys_module = root / "sys-module"
        self.mountinfo = root / "mountinfo"
        self.sys_block.mkdir()
        self.sys_module.mkdir()
        self.write_mountinfo()
        self.hang: set[str] = set()
        self.seconds_per_command = 0.0
        self.emitted: list[dict[str, Any]] = []
        self.minor = 6
        #: stdin of the command being answered (a multi-segment ``dmsetup
        #: reload`` table arrives there) and the ``open_count`` answers of
        #: ``dmsetup info -c`` per mapping, the last one repeating.
        self.stdin: str | None = None
        self.open_counts: dict[str, list[int]] = {}

    def write_mountinfo(self) -> None:
        lines = ["22 1 259:1 / / rw,relatime shared:1 - xfs /dev/nvme0n1p1 rw"]
        index = 40
        for target in sorted(self.stack):
            for source in self.stack[target]:
                minor = self.mappings.get(source.rpartition("/")[2], index)
                lines.append(
                    f"{index} 22 253:{minor} / {target} rw,relatime shared:{index} "
                    f"- ext4 {source} rw"
                )
                index += 1
        self.mountinfo.write_text("\n".join(lines) + "\n")

    def mount_devno(self, target: str) -> str:
        """``major:minor`` of the top mount on ``target`` (the mapping's minor)."""

        source = self.mounts[target].rpartition("/")[2]
        return f"253:{self.mappings[source]}"

    def run(
        self, command: list[str], *, timeout: float, input: str | None = None, **_: Any
    ) -> subprocess.CompletedProcess[str]:
        command = list(command)
        self.calls.append(command)
        self.timeouts.append(timeout)
        self.stdin = input
        hosted = command[: len(NS)] == NS
        inner = command[len(NS) :] if hosted else command
        stuck = inner[0] in self.hang or (
            inner[0] == "dmsetup" and inner[1] in UDEV_SYNCED and not hosted
        )
        if stuck or self.seconds_per_command > timeout:
            self.clock.now += timeout
            raise subprocess.TimeoutExpired(command, timeout)
        self.clock.now += self.seconds_per_command
        code, out = self.answer(inner)
        return subprocess.CompletedProcess(command, code, stdout=out, stderr="")

    def answer(self, inner: list[str]) -> tuple[int, str]:
        tool, args = inner[0], inner[1:]
        if tool == "systemctl":
            return self.systemctl(args)
        if tool == "systemd-run":
            unit = next(item for item in args if item.startswith("--unit="))[7:]
            if any(item.startswith("--on-active=") for item in args):
                self.timers.add(unit)
            else:
                self.services[unit] = "active"
            self.started[unit] = args
            return 0, ""
        if tool == "dmsetup":
            return self.dmsetup(args)
        if tool == "losetup":
            return self.losetup(args)
        if tool == "blockdev":
            return (0, "") if args[0] == "--flushbufs" else (0, "4194304\n")
        if tool == "fallocate":
            Path(args[-1]).write_bytes(b"")
            return 0, ""
        if tool in {"mkfs.ext4", "mkfs.vfat"}:
            return 0, ""
        if tool == "mount":
            self.stack.setdefault(args[-1], []).append(args[-2])
            self.mounts[args[-1]] = args[-2]
            self.write_mountinfo()
            return 0, ""
        if tool == "umount":
            target = args[-1]
            if not self.stack.get(target):
                return 32, ""
            self.stack[target].pop()
            if self.stack[target]:
                self.mounts[target] = self.stack[target][-1]
            else:
                del self.stack[target]
                del self.mounts[target]
            self.write_mountinfo()
            return 0, ""
        if tool == "nsenter":
            # The sweep's ``nsenter --target PID --mount -- umount -l PATH``
            # takes the holder's private copy down (a locked one stays).
            pid = int(args[args.index("--target") + 1])
            for holder in self.holders:
                if holder["pid"] == pid and not holder.get("locked"):
                    holder["gone"] = True
                    for name, answers in self.open_counts.items():
                        self.open_counts[name] = [max(0, answers[-1] - 1)]
            return 0, ""
        if tool == "modprobe":
            if args[0] == "-r":
                shutil.rmtree(self.sys_module / args[1], ignore_errors=True)
            else:
                (self.sys_module / args[0]).mkdir(exist_ok=True)
                (self.sys_module / args[0] / "refcnt").write_text("0\n")
            return 0, ""
        raise AssertionError(f"unexpected host command {inner}")

    def systemctl(self, args: list[str]) -> tuple[int, str]:
        verb = args[0]
        if verb == "stop":
            base, _, suffix = args[1].rpartition(".")
            if suffix == "timer":
                self.timers.discard(base)
            else:
                self.services.pop(base, None)
            return 0, ""
        if verb == "reset-failed":
            return 0, ""
        if verb == "show":
            base, _, suffix = args[1].rpartition(".")
            if suffix == "timer":
                state = "active" if base in self.timers else "inactive"
            else:
                state = self.services.get(base, "inactive")
            known = base in self.timers or base in self.services
            load = "loaded" if known else "not-found"
            return 0, f"LoadState={load}\nActiveState={state}\nMainPID=0\n"
        if verb == "list-units":
            prefix = args[-1].rstrip("*")
            listed = sorted(unit for unit in self.services if unit.startswith(prefix))
            return 0, "".join(
                f"{unit}.service loaded active running\n" for unit in listed
            )
        raise AssertionError(f"unexpected systemctl {args}")

    def dmsetup(self, args: list[str]) -> tuple[int, str]:
        verb, name = args[0], args[-1]
        if verb == "reload":
            name = args[1]
            if args[2:3] == ["--table"]:
                self.tables[name] = args[3]
            else:
                # ``dmsetup reload NAME`` without ``--table`` reads stdin: the
                # only way to load a table of more than one segment.
                assert self.stdin, args
                self.tables[name] = self.stdin.rstrip("\n")
            return 0, ""
        if verb == "info" and "-o" in args:
            if name not in self.mappings:
                return 1, ""
            if args[args.index("-o") + 1] == "major,minor":
                return 0, f"253:{self.mappings[name]}\n"
            answers = self.open_counts.get(name) or [0]
            return 0, f"{answers.pop(0) if len(answers) > 1 else answers[0]}\n"
        if verb == "create":
            name = args[1]
            if args[2:3] == ["--table"]:
                self.tables[name] = args[3]
            else:
                assert self.stdin, args  # a multi-segment table arrives on stdin
                self.tables[name] = self.stdin.rstrip("\n")
            self.minor += 1
            self.mappings[name] = self.minor
            node = self.sys_block / f"dm-{self.minor}" / "dm"
            node.mkdir(parents=True)
            (node / "name").write_text(name + "\n")
            return 0, ""
        if verb == "remove":
            if name not in self.mappings:
                return 1, ""
            if "--deferred" in args and (self.open_counts.get(name) or [0])[-1]:
                self.deferred.append(name)  # goes once the last holder closes it
                return 0, ""
            shutil.rmtree(self.sys_block / f"dm-{self.mappings.pop(name)}")
            return 0, ""
        if verb in {"resume", "suspend", "info"}:
            return (0, f"Name: {name}\n") if name in self.mappings else (1, "")
        if verb == "ls":
            listing = "".join(
                f"{key}\t(253:{value})\n" for key, value in self.mappings.items()
            )
            return 0, listing or "No devices found\n"
        raise AssertionError(f"unexpected dmsetup {args}")

    def losetup(self, args: list[str]) -> tuple[int, str]:
        if args[:2] == ["--find", "--show"]:
            device = f"/dev/loop{len(self.loops)}"
            self.loops[device] = args[2]
            return 0, device + "\n"
        if args == ["-J", "-l"]:
            devices = [
                {"name": key, "back-file": value} for key, value in self.loops.items()
            ]
            return 0, json.dumps({"loopdevices": devices})
        if args[0] == "-d":
            self.loops.pop(args[1], None)
            return 0, ""
        raise AssertionError(f"unexpected losetup {args}")

    def device_calls(self) -> list[list[str]]:
        return [call for call in self.calls if tool_of(call) in DEVICE_TOOLS]


def tool_of(call: list[str]) -> str:
    return call[len(NS)] if call[: len(NS)] == NS else call[0]


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeDmHost:
    clock = FakeClock()
    fake = FakeDmHost(tmp_path, clock)
    backing = tmp_path / "var-lib" / "gpu-fault-acceptance"
    backing.mkdir(parents=True)
    monkeypatch.setattr(storep, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(storep, "MOUNT_ROOT", tmp_path / "mnt")
    monkeypatch.setattr(storep, "BACKING_ROOTS", (backing,))
    monkeypatch.setattr(storep, "backing_root", lambda value, size: Path(value))
    monkeypatch.setattr(storep, "HOST_MOUNTINFO", fake.mountinfo)
    monkeypatch.setattr(storep, "SYS_BLOCK", fake.sys_block)
    monkeypatch.setattr(storep, "SYS_MODULE", fake.sys_module)
    monkeypatch.setattr(storep, "DEADLINE", [])
    monkeypatch.setattr(storep, "tool_present", lambda name: True)
    monkeypatch.setattr(storep, "emit", fake.emitted.append)
    # The fake device node cannot be read: the FAT16 geometry is pinned, an
    # O_DIRECT read answers from the mapping's table (EIO inside an ``error``
    # segment), a path's devno is the top mount's and ``/proc`` is the fake's.
    monkeypatch.setattr(storep, "fat_region", lambda device: list(FAT_SECTORS))

    def direct_read(device: str, sector: int) -> str:
        table = fake.tables.get(device.rpartition("/")[2], "")
        for line in table.splitlines():
            start, length, target = line.split()[:3]
            if int(start) <= sector < int(start) + int(length):
                return "OSError: 5" if target == "error" else "ok"
        return "ok"

    monkeypatch.setattr(storep, "direct_read", direct_read)
    monkeypatch.setattr(storep, "mount_devno", fake.mount_devno)
    monkeypatch.setattr(
        storep,
        "foreign_holders",
        lambda devnos: [
            {k: v for k, v in holder.items() if k != "gone"}
            for holder in fake.holders
            if not holder.get("gone")
        ],
    )
    monkeypatch.setattr(
        storep,
        "subprocess",
        types.SimpleNamespace(
            run=fake.run,
            PIPE=subprocess.PIPE,
            CompletedProcess=subprocess.CompletedProcess,
            TimeoutExpired=subprocess.TimeoutExpired,
        ),
    )
    monkeypatch.setattr(
        storep,
        "time",
        types.SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep),
    )
    return fake


def create(*extra: str, delay: int = 0) -> list[str]:
    return [
        "dm-create",
        "--run-id",
        RUN_ID,
        "--name",
        "io",
        "--size-mib",
        "2048",
        "--delay-ms",
        str(delay),
        "--backing",
        str(storep.BACKING_ROOTS[0]),
        "--deadman-seconds",
        "1440",
        *extra,
    ]


def call(subcommand: str, *extra: str) -> list[str]:
    return [subcommand, "--run-id", RUN_ID, *extra]


def device() -> dict[str, Any]:
    return storep.load_state(RUN_ID)["devices"]["io"]


def built(host: FakeDmHost) -> dict[str, Any]:
    assert storep.main(create()) == 0, host.emitted[-1]
    assert storep.main(call("dm-setup", "--name", "io")) == 0, host.emitted[-1]
    return host.emitted[-1]


def test_dm_create_only_arms_the_deadman_and_starts_the_bounded_setup_unit(
    host: FakeDmHost,
) -> None:
    assert storep.main(create()) == 0, host.emitted[-1]
    created = host.emitted[-1]
    assert (created["phase"], created["setup_unit"]) == ("setup", SETUP_UNIT)
    assert created["setup_max_seconds"] == storep.SETUP_MAX_SECONDS
    assert host.device_calls() == [], "the probe call builds nothing itself"
    started = [call for call in host.calls if call[0] == "systemd-run"]
    assert [item[1] for item in started] == [
        f"--unit={DEADMAN}",
        f"--unit={SETUP_UNIT}",
    ]
    deadman, setup = started
    assert deadman[-5:] == ["cleanup", "--run-id", RUN_ID, "--kind", "dm-io"]
    assert "--on-active=1440s" in deadman
    assert f"--property=RuntimeMaxSec={storep.SETUP_MAX_SECONDS}" in setup
    stop = f"--property=TimeoutStopSec={storep.UNIT_TIMEOUT_STOP_SECONDS}"
    assert stop in setup and stop in deadman, (setup, deadman)
    copy = storep.state_dir(RUN_ID) / "storage-probe.py"
    assert setup[-7:] == [
        sys.executable,
        str(copy),
        "dm-setup",
        "--run-id",
        RUN_ID,
        "--name",
        "io",
    ]
    assert copy.read_bytes() == Path(storep.__file__).read_bytes()
    entry = device()
    assert (entry["phase"], entry["setup_unit"], entry["loop"]) == (
        "setup",
        SETUP_UNIT,
        None,
    )


def test_the_setup_unit_builds_the_device_in_the_host_namespaces(
    host: FakeDmHost,
) -> None:
    assert storep.main(create()) == 0, host.emitted[-1]
    host.calls.clear()
    assert storep.main(call("dm-setup", "--name", "io")) == 0, host.emitted[-1]
    ready = host.emitted[-1]
    assert (ready["phase"], ready["device"], ready["loop"]) == (
        "ready",
        "dm-7",
        "/dev/loop0",
    )
    assert [tool_of(call) for call in host.calls] == [
        "fallocate",
        "losetup",
        "blockdev",
        "dmsetup",
        "mkfs.ext4",
        "mount",
    ]
    hosted = [call for call in host.calls if tool_of(call) in {"dmsetup", "mount"}]
    assert all(call[: len(NS)] == NS for call in hosted), hosted
    assert hosted[0][len(NS) :] == [
        "dmsetup",
        "create",
        DM,
        "--table",
        "0 4194304 linear /dev/loop0 0",
    ]
    entry = device()
    assert (entry["phase"], entry["device"], entry["mounted"]) == (
        "ready",
        "dm-7",
        True,
    )
    assert host.mounts == {entry["mountpoint"]: f"/dev/mapper/{DM}"}
    assert storep.main(call("dm-status", "--name", "io")) == 0, host.emitted[-1]
    status = host.emitted[-1]
    assert (status["phase"], status["device"], status["failed_step"]) == (
        "ready",
        "dm-7",
        None,
    )


def test_a_pod_side_dmsetup_reproduces_the_live_timeout_and_now_names_its_step(
    host: FakeDmHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Without the host namespaces the create waits for a cookie that never
    # comes, exactly as the live dm-create did -- but the worker now records
    # where, and the probe's error names the step instead of a bare class.
    monkeypatch.setattr(storep, "HOST_NAMESPACES", ())
    assert storep.main(create()) == 0, host.emitted[-1]
    assert storep.main(call("dm-setup", "--name", "io")) == 1
    failure = host.emitted[-1]
    assert failure["error_class"] == "TimeoutExpired", failure
    assert re.fullmatch(r"build_device:[1-9][0-9]*", failure["error_site"]), failure
    assert re.fullmatch(r"[0-9a-f]{16}", failure["error_code"]), failure
    entry = device()
    assert (entry["phase"], entry["step"], entry["setup_error_class"]) == (
        "failed",
        "dmsetup-create",
        "TimeoutExpired",
    )
    assert storep.main(call("dm-status", "--name", "io")) == 0, host.emitted[-1]
    status = host.emitted[-1]
    assert (status["phase"], status["failed_step"], status["setup_error_site"]) == (
        "failed",
        "dmsetup-create",
        failure["error_site"],
    )
    assert "error" not in status, "an error key would read as a failed call"


def test_a_setup_unit_that_ended_without_a_result_reads_failed(
    host: FakeDmHost,
) -> None:
    assert storep.main(create()) == 0, host.emitted[-1]
    storep.update_device(RUN_ID, "io", step="mkfs")
    host.services.pop(SETUP_UNIT)  # killed at RuntimeMaxSec before it recorded
    assert storep.main(call("dm-status", "--name", "io")) == 0, host.emitted[-1]
    status = host.emitted[-1]
    assert (status["phase"], status["failed_step"], status["setup_error_class"]) == (
        "failed",
        "mkfs",
        "SetupUnitEnded",
    )
    assert status["device"] is None


def test_a_running_setup_reads_setup_until_it_records_ready(host: FakeDmHost) -> None:
    assert storep.main(create()) == 0, host.emitted[-1]
    assert storep.main(call("dm-status", "--name", "io")) == 0, host.emitted[-1]
    assert host.emitted[-1]["phase"] == "setup"
    assert host.emitted[-1]["setup_unit"]["ActiveState"] == "active"


def test_fixed_path_calls_fit_their_commands_inside_the_budget(
    host: FakeDmHost,
) -> None:
    # The sum of the per-command timeouts on each call's longest path is its
    # documented worst case; the budget (and the runner's exec timeout above
    # it) must cover it, so the clamp never cuts a call that is within bounds.
    spent: dict[str, float] = {}

    def measure(argv: list[str]) -> None:
        start = len(host.timeouts)
        assert storep.main(argv) == 0, host.emitted[-1]
        spent[argv[0]] = sum(host.timeouts[start:])

    measure(create("--load-module", delay=200))
    measure(call("dm-status", "--name", "io"))
    measure(call("dm-setup", "--name", "io"))
    measure(call("io-start", "--name", "io", "--seconds", "60", "--direct"))
    measure(call("io-stop", "--name", "io"))
    measure(call("dm-suspend", "--name", "io", "--seconds", "90"))
    measure(call("dm-resume", "--name", "io"))
    measure(call("dm-recreate", "--name", "io"))
    assert set(spent) <= set(storep.CALL_BUDGET_SECONDS), spent
    over = {
        name: (seconds, storep.CALL_BUDGET_SECONDS[name])
        for name, seconds in spent.items()
        if seconds > storep.CALL_BUDGET_SECONDS[name]
    }
    assert over == {}, over
    steps = sum(storep.SETUP_STEP_SECONDS.values())
    assert spent["dm-setup"] == steps == storep.CALL_BUDGET_SECONDS["dm-setup"]
    assert storep.SETUP_MAX_SECONDS > storep.CALL_BUDGET_SECONDS["dm-setup"], (
        "the worker records its failure before RuntimeMaxSec kills it"
    )


def test_a_crawling_host_cannot_keep_a_call_past_its_budget(host: FakeDmHost) -> None:
    # Every command answers just inside its own timeout, so only the call's
    # budget can end it: the clamp cuts the last command to what is left and
    # the call answers with its error before the runner's exec timeout.
    built(host)
    host.seconds_per_command = storep.QUICK_SECONDS - 1
    start = host.clock.now
    assert storep.main(call("cleanup", "--kind", "dm-io")) == 1
    failure = host.emitted[-1]
    assert failure["error_class"] in {"BudgetExhausted", "TimeoutExpired"}, failure
    elapsed = host.clock.now - start
    assert elapsed <= storep.CALL_BUDGET_SECONDS["cleanup"], elapsed
    assert host.timeouts[-1] < storep.QUICK_SECONDS, "the last command was clamped"
    assert storep.DEADLINE == [], "the deadline ends with the call"
    # A later run on a responsive host (the deadman's retry) proves it clean.
    host.seconds_per_command = 0.0
    assert storep.main(call("cleanup", "--kind", "dm-io")) == 0, host.emitted[-1]
    assert host.emitted[-1]["clean"] is True, host.emitted[-1]["residue"]


def test_cleanup_undoes_a_half_built_device_and_the_module_it_loaded(
    host: FakeDmHost,
) -> None:
    assert storep.main(create("--load-module", delay=200)) == 0, host.emitted[-1]
    assert (host.sys_module / storep.DELAY_MODULE).is_dir(), "loaded for the run"
    assert storep.load_state(RUN_ID)["modules"] == [storep.DELAY_MODULE]
    host.hang.add("mkfs.ext4")
    assert storep.main(call("dm-setup", "--name", "io")) == 1
    assert host.mappings and host.loops, "the mapping and the loop exist"
    residue = storep.residue(RUN_ID)
    assert residue["units"] == [f"{SETUP_UNIT}.service"], residue
    assert residue["modules_loaded_by_run"] == [storep.DELAY_MODULE], residue
    host.hang.clear()
    assert storep.main(call("cleanup", "--kind", "dm-io")) == 0, host.emitted[-1]
    result = host.emitted[-1]
    assert result["clean"] is True, result["residue"]
    assert (host.mappings, host.loops, host.mounts) == ({}, {}, {})
    assert not (host.sys_module / storep.DELAY_MODULE).exists(), "unloaded again"
    assert host.timers == set() and host.services == {}, (host.timers, host.services)
    assert storep.load_state(RUN_ID)["devices"] == {}
    # Only ``dmsetup ls`` (no uevent, no cookie) may run in the Pod itself.
    plain = [call for call in host.calls if call[:1] == ["dmsetup"]]
    assert {tuple(call) for call in plain} == {("dmsetup", "ls")}, plain


def test_cleanup_disarms_the_deadman_of_a_device_that_was_never_recorded(
    host: FakeDmHost,
) -> None:
    host.timers.add(DEADMAN)
    assert storep.main(call("cleanup", "--kind", "dm-io")) == 0, host.emitted[-1]
    assert host.timers == set() and host.emitted[-1]["clean"] is True, host.emitted


def test_the_writer_must_stop_before_the_recreate(host: FakeDmHost) -> None:
    built(host)
    argv = call("io-start", "--name", "io", "--seconds", "60", "--direct")
    assert storep.main(argv) == 0, host.emitted[-1]
    host.calls.clear()
    assert storep.main(call("dm-recreate", "--name", "io")) == 1
    assert host.emitted[-1]["error_class"] == "ProbeError", host.emitted[-1]
    assert host.device_calls() == [], "refused before any unmount or remove"
    assert storep.main(call("io-stop", "--name", "io")) == 0, host.emitted[-1]
    assert storep.main(call("dm-recreate", "--name", "io")) == 0, host.emitted[-1]
    recreated = host.emitted[-1]
    assert recreated["device_before"] == "dm-7", recreated
    assert recreated["device_after"] == device()["device"], recreated
    commands = [call[len(NS) :][:2] for call in host.device_calls()]
    assert commands == [
        ["umount", device()["mountpoint"]],
        ["dmsetup", "remove"],
        ["dmsetup", "create"],
        ["mount", "-t"],
    ], commands


def test_suspend_and_resume_reach_the_host_and_bound_the_resume_deadman(
    host: FakeDmHost,
) -> None:
    built(host)
    host.calls.clear()
    assert storep.main(call("dm-suspend", "--name", "io", "--seconds", "90")) == 0
    resume = f"gfrs-{DIGEST}-sdeadman-resume-io"
    assert resume in host.timers and "--on-active=90s" in host.started[resume]
    assert storep.main(call("dm-resume", "--name", "io")) == 0, host.emitted[-1]
    assert resume not in host.timers
    dmsetup = [call for call in host.calls if tool_of(call) == "dmsetup"]
    assert [call[len(NS) + 1] for call in dmsetup] == ["suspend", "info", "resume"]
    assert all(call[: len(NS)] == NS for call in dmsetup), dmsetup


def test_io_start_needs_a_finished_mounted_device(host: FakeDmHost) -> None:
    assert storep.main(create()) == 0, host.emitted[-1]
    argv = call("io-start", "--name", "io", "--seconds", "60", "--direct")
    assert storep.main(argv) == 1
    assert host.emitted[-1]["error_class"] == "ProbeError", host.emitted[-1]
    assert WRITER_UNIT not in host.services
    assert storep.main(call("dm-setup", "--name", "io")) == 0, host.emitted[-1]
    assert storep.main(argv) == 0, host.emitted[-1]
    writer = host.started[WRITER_UNIT]
    assert "--property=RuntimeMaxSec=60" in writer and "--property=Nice=19" in writer
    stop = f"--property=TimeoutStopSec={storep.UNIT_TIMEOUT_STOP_SECONDS}"
    assert stop in writer, writer


def test_dm_create_refuses_an_unloaded_dm_delay_without_the_load_flag(
    host: FakeDmHost,
) -> None:
    assert storep.main(create(delay=200)) == 1
    assert host.emitted[-1]["error_class"] == "ProbeError", host.emitted[-1]
    assert host.calls == [], "refused before the deadman or any record"
    assert storep.load_state(RUN_ID)["devices"] == {}


def test_the_dm_name_comes_from_the_kernels_sysfs_name(host: FakeDmHost) -> None:
    (host.sys_block / "dm-3" / "dm").mkdir(parents=True)
    (host.sys_block / "dm-3" / "dm" / "name").write_text("other\n")
    (host.sys_block / "dm-11" / "dm").mkdir(parents=True)
    (host.sys_block / "dm-11" / "dm" / "name").write_text(f"{DM}\n")
    assert storep.dm_device_name(DM) == "dm-11"
    with pytest.raises(storep.ProbeError, match="no dm-N block device"):
        storep.dm_device_name("gfrs-missing-io")


def test_the_probe_copy_follows_the_current_script(host: FakeDmHost) -> None:
    copy = storep.probe_copy(RUN_ID)
    copy.write_text("# an older probe without dm-setup\n")
    assert storep.probe_copy(RUN_ID) == copy
    assert copy.read_bytes() == Path(storep.__file__).read_bytes()


# --------------------------------------------------------------------------- #
# 004 probe_error: FAT16 carrier and the error-table swap
# --------------------------------------------------------------------------- #
#: The FAT region ``mkfs.vfat -F 16`` leaves on the 256 MiB carrier (8 reserved
#: sectors, two FATs of 256 sectors), as ``fat_region`` reads it from the boot
#: sector -- the fake device node cannot be read, so the tests pin the value.
FAT_SECTORS = [8, 520]
FAT_ERROR_TABLE = (
    "0 8 linear /dev/loop0 0\n8 512 error\n520 4193784 linear /dev/loop0 520"
)
#: The 004 fault mapping: a second mapping over the same loop device.
FAULT = f"{DM}-fault"


def table_statvfs(host: FakeDmHost) -> Any:
    """A ``statvfs_reading`` that fails EIO while the mapping under the TOP mount
    of the path (each mapping gets its own superblock) has its FAT on ``error``."""

    def reading(path: str) -> dict[str, Any]:
        source = host.mounts.get(path, f"/dev/mapper/{DM}").rpartition("/")[2]
        if " error" in host.tables.get(source, ""):
            return {"path": path, "error": "OSError: 5"}
        return {"path": path, "used_percent": 1.0, "read_only": False}

    return reading


def build_vfat(host: FakeDmHost, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    synced: list[str] = []
    monkeypatch.setattr(storep, "statvfs_reading", table_statvfs(host))
    monkeypatch.setattr(storep, "fat_region", lambda device: list(FAT_SECTORS))
    monkeypatch.setattr(storep.os, "sync", lambda: synced.append("sync"))
    (host.sys_module / storep.VFAT_MODULE).mkdir(exist_ok=True)  # already loaded
    assert storep.main(create("--fstype", "vfat")) == 0, host.emitted[-1]
    assert storep.main(call("dm-setup", "--name", "io")) == 0, host.emitted[-1]
    return synced


def boot_sector(
    *, bytes_per_sector: int = 512, reserved: int = 8, fats: int = 2, per_fat: int = 256
) -> bytes:
    """A FAT16 BIOS parameter block with the given geometry."""

    sector = bytearray(512)
    sector[11:24] = struct.pack(
        "<HBHBHHBH", bytes_per_sector, 8, reserved, fats, 512, 0, 0xF8, per_fat
    )
    sector[54:62] = b"FAT16   "
    return bytes(sector)


def test_a_vfat_device_is_formatted_fat16_and_mounted_with_errors_continue(
    host: FakeDmHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_vfat(host, monkeypatch)
    ready = host.emitted[-1]
    assert ready["fstype"] == "vfat" and device()["fstype"] == "vfat", ready
    assert device()["target"] == "data", device()
    assert ready["fat_sectors"] == FAT_SECTORS == device()["fat_sectors"], ready
    mkfs = next(call for call in host.calls if call[0] == "mkfs.vfat")
    assert mkfs[1:4] == ["-F", "16", "-n"] and mkfs[4] == "GFRSIO", mkfs
    mount = next(call for call in host.device_calls() if tool_of(call) == "mount")
    assert mount[len(NS) :][:5] == [
        "mount",
        "-t",
        "vfat",
        "-o",
        "errors=continue,noatime",
    ], mount


def test_retarget_error_mounts_a_fault_mapping_over_the_healthy_one_and_data_restores(
    host: FakeDmHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 2026-10-25/26 live runs: a remount of the SAME device reused the
    superblock a private mount-namespace copy (the window probe Pod) kept
    alive, and statvfs answered from its cached count. The injection is now a
    second mapping over the same loop device whose FAT segment is ``error``,
    proved by O_DIRECT reads and mounted over the healthy mount: a new block
    device gets a new superblock, so the count has to read the FAT -- EIO.
    The healthy mapping is never unmounted or reloaded for it."""

    synced = build_vfat(host, monkeypatch)
    host.calls.clear()
    assert storep.main(call("dm-retarget", "--name", "io", "--target", "error")) == 0
    faulted = host.emitted[-1]
    assert faulted["faulted"] is True and faulted["attempts"] == 1, faulted
    assert faulted["statvfs"]["error"] == "OSError: 5" and faulted["mounted_on_host"]
    assert faulted["fat_sectors"] == FAT_SECTORS and faulted["diagnosis"] is None
    assert faulted["fault_dm"] == FAULT == device()["fault_dm"], faulted
    assert faulted["direct_reads"] == {"boot": "ok", "fat": "OSError: 5"}, faulted
    assert (
        faulted["mount_devno"]
        == faulted["fault_devno"]
        == f"253:{host.mappings[FAULT]}"
    )
    assert host.tables[FAULT] == FAT_ERROR_TABLE, host.tables
    assert host.tables[DM] == "0 4194304 linear /dev/loop0 0", (
        "the healthy mapping stays"
    )
    assert host.stack[device()["mountpoint"]] == [
        device()["dm_path"],
        f"/dev/mapper/{FAULT}",
    ]
    assert device()["target"] == "error", device()
    assert synced == [], "nothing is synced or unmounted for the injection"
    steps = [
        call[len(NS) :][:2] if call[: len(NS)] == NS else call[:2]
        for call in host.device_calls()
        if call[len(NS) :][:2] != ["dmsetup", "info"]  # devno lookups
    ]
    assert steps == [["dmsetup", "create"], ["mount", "-t"]], steps
    created = next(
        call
        for call in host.calls
        if call[len(NS) : len(NS) + 2] == ["dmsetup", "create"]
    )
    assert created[len(NS) :] == ["dmsetup", "create", FAULT], (
        "the table arrives on stdin"
    )
    mount = next(call for call in host.device_calls() if tool_of(call) == "mount")
    assert mount[len(NS) :] == [
        "mount",
        "-t",
        "vfat",
        "-o",
        "errors=continue,noatime",
        f"/dev/mapper/{FAULT}",
        device()["mountpoint"],
    ]
    assert all(call[: len(NS)] == NS for call in host.device_calls()), (
        "every device command joins the host namespaces"
    )
    host.calls.clear()
    assert storep.main(call("dm-retarget", "--name", "io", "--target", "data")) == 0
    restored = host.emitted[-1]
    assert restored["faulted"] is False and "error" not in restored["statvfs"]
    assert host.tables[FAULT] == "0 4194304 linear /dev/loop0 0", host.tables
    assert device()["target"] == "data" and device()["fault_dm"] == FAULT, device()
    reloaded = [
        call[len(NS) :][:3]
        for call in host.device_calls()
        if call[len(NS) :][:2] != ["dmsetup", "info"]
    ]
    assert reloaded == [
        ["dmsetup", "suspend", "--nolockfs"],
        ["dmsetup", "reload", FAULT],
        ["dmsetup", "resume", FAULT],
    ], "the fault mapping is restored in place; no remount, no touch of the device"


def test_retarget_reports_a_reading_that_still_answers_with_its_diagnosis(
    host: FakeDmHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A statvfs that answers although the fault mapping's FAT reads fail is
    reported ``faulted: false`` with the diagnosis (the runner's INCONCLUSIVE
    names it), after exactly one attempt -- no retry loop, no second mount."""

    build_vfat(host, monkeypatch)
    monkeypatch.setattr(
        storep, "statvfs_reading", lambda path: {"path": path, "used_percent": 1.0}
    )
    host.calls.clear()
    assert storep.main(call("dm-retarget", "--name", "io", "--target", "error")) == 0
    result = host.emitted[-1]
    assert result["faulted"] is False and result["attempts"] == 1, result
    assert result["direct_reads"]["fat"] == "OSError: 5", result
    assert result["diagnosis"] == (
        "statfs answered over a mapping whose FAT reads fail: superblock reused"
    ), result
    creates = [
        c for c in host.calls if c[len(NS) : len(NS) + 2] == ["dmsetup", "create"]
    ]
    assert len(creates) == 1 and host.tables[FAULT] == FAT_ERROR_TABLE, host.tables


def test_retarget_refuses_a_fault_mapping_whose_fat_segment_reads(
    host: FakeDmHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_vfat(host, monkeypatch)
    monkeypatch.setattr(storep, "direct_read", lambda device, sector: "ok")
    assert storep.main(call("dm-retarget", "--name", "io", "--target", "error")) == 1
    error = host.emitted[-1]
    assert error["error_class"] == "ProbeError", error
    assert error["error_site"].startswith("fault_mapping:"), error
    assert (
        device()["mountpoint"] not in host.stack
        or len(host.stack[device()["mountpoint"]]) == 1
    ), "nothing was mounted over the healthy mount"


def test_retarget_error_on_an_ext4_device_keeps_the_whole_device_table(
    host: FakeDmHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    built(host)
    monkeypatch.setattr(storep, "statvfs_reading", table_statvfs(host))
    host.calls.clear()
    assert storep.main(call("dm-retarget", "--name", "io", "--target", "error")) == 0
    assert host.tables[DM] == "0 4194304 error", host.tables
    tools = [tool_of(call) for call in host.device_calls()]
    assert tools == ["dmsetup", "dmsetup", "dmsetup"], "no remount for ext4"


def test_fat_region_reads_the_boot_sector_and_page_aligns_the_error_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(storep, "PAGE_SECTORS", 8)
    monkeypatch.setattr(storep, "MIN_FAT_ERROR_SECTORS", 16)
    device = tmp_path / "carrier"
    device.write_bytes(boot_sector())
    assert storep.fat_region(str(device)) == [8, 520], "mkfs.vfat 4.2 FAT16 on 256 MiB"
    # One reserved sector: the segment starts at the next page so a user-space
    # read of the boot sector's page never hits ``error``; the FAT tail that
    # does not fill a page stays readable too (the count reads the first FAT).
    device.write_bytes(boot_sector(reserved=1, per_fat=100))
    assert storep.fat_region(str(device)) == [8, 200]
    # 4 KiB sectors scale every offset to 512-byte sectors.
    device.write_bytes(boot_sector(bytes_per_sector=4096, reserved=1, per_fat=32))
    assert storep.fat_region(str(device)) == [8, 520]
    device.write_bytes(boot_sector(per_fat=4))
    with pytest.raises(storep.ProbeError, match="no error segment"):
        storep.fat_region(str(device))
    device.write_bytes(boot_sector()[:54] + b"FAT32   " + bytes(450))
    with pytest.raises(storep.ProbeError, match="FAT16 boot sector"):
        storep.fat_region(str(device))


def test_dm_table_puts_only_the_fat_segment_on_error() -> None:
    assert storep.dm_table(524288, "/dev/loop0", 0, error=True, fat=[8, 520]) == (
        "0 8 linear /dev/loop0 0\n8 512 error\n520 523768 linear /dev/loop0 520"
    )
    assert storep.dm_table(1024, "/dev/loop0", 0, error=True, fat=[0, 1024]) == (
        "0 1024 error"
    )
    assert storep.dm_table(1024, "/dev/loop0", 0, error=True) == "0 1024 error"
    assert storep.dm_table(1024, "/dev/loop0", 0, fat=[8, 520]) == (
        "0 1024 linear /dev/loop0 0"
    ), "the FAT segment applies to the error table only"


def test_retarget_refuses_a_running_writer_and_an_unfinished_device(
    host: FakeDmHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    (host.sys_module / storep.VFAT_MODULE).mkdir(exist_ok=True)
    assert storep.main(create("--fstype", "vfat")) == 0, host.emitted[-1]
    assert storep.main(call("dm-retarget", "--name", "io", "--target", "error")) == 1
    assert host.emitted[-1]["error_class"] == "ProbeError", host.emitted[-1]
    assert storep.main(call("dm-setup", "--name", "io")) == 0, host.emitted[-1]
    assert storep.main(call("io-start", "--name", "io", "--seconds", "60")) == 0
    host.calls.clear()
    assert storep.main(call("dm-retarget", "--name", "io", "--target", "error")) == 1
    assert host.device_calls() == [], "refused before any unmount or reload"


def test_cleanup_of_a_faulted_device_unmounts_the_stack_and_removes_both_mappings(
    host: FakeDmHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_vfat(host, monkeypatch)
    assert storep.main(call("dm-retarget", "--name", "io", "--target", "error")) == 0
    host.calls.clear()
    assert storep.main(call("cleanup", "--kind", "dm-io")) == 0, host.emitted[-1]
    assert host.emitted[-1]["clean"] is True, host.emitted[-1]["residue"]
    verbs = [
        (tool_of(call), inner[1] if inner[0] == "dmsetup" else "")
        for call in host.device_calls()
        for inner in [call[len(NS) :] if call[: len(NS)] == NS else call]
    ]
    assert verbs.count(("umount", "")) == 2, verbs
    removes = [
        inner[-1]
        for call in host.device_calls()
        for inner in [call[len(NS) :]]
        if inner[:2] == ["dmsetup", "remove"]
    ]
    assert removes == [FAULT, DM], "the fault mapping (on top) goes first"
    assert ("dmsetup", "reload") not in verbs, "no table is reloaded for the cleanup"
    assert (host.mappings, host.loops, host.mounts, host.stack) == ({}, {}, {}, {})
    destroyed = host.emitted[-1]["actions"][0]["device"]
    assert destroyed["holders"] == [] and destroyed["deferred"] == [], destroyed
    assert destroyed["directory_removed"] is True, destroyed


def test_cleanup_of_an_ext4_error_table_puts_the_data_table_back_first(
    host: FakeDmHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    built(host)
    monkeypatch.setattr(storep, "statvfs_reading", table_statvfs(host))
    assert storep.main(call("dm-retarget", "--name", "io", "--target", "error")) == 0
    host.calls.clear()
    assert storep.main(call("cleanup", "--kind", "dm-io")) == 0, host.emitted[-1]
    assert host.emitted[-1]["clean"] is True, host.emitted[-1]["residue"]
    verbs = [
        (tool_of(call), inner[1] if inner[0] == "dmsetup" else "")
        for call in host.device_calls()
        for inner in [call[len(NS) :] if call[: len(NS)] == NS else call]
    ]
    assert verbs.index(("dmsetup", "reload")) < verbs.index(("umount", "")), verbs


def test_destroy_evicts_a_private_copy_of_the_mount_from_another_namespace(
    host: FakeDmHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 2026-10-26 live cleanup: the window probe Pod, started after the
    mount with private propagation, kept a copy of it; the host unmount left
    the superblock alive, the mapping open (1), and ``dmsetup remove`` failed
    ``Device or resource busy``. The destroy now finds the namespace by the
    mappings' major:minor in ``/proc/<pid>/mountinfo``, lazily unmounts the
    copy there, and removes once the open count drops."""

    build_vfat(host, monkeypatch)
    assert storep.main(call("dm-retarget", "--name", "io", "--target", "error")) == 0
    host.holders = [
        {"pid": 4242, "comm": "pause", "locked": False, "mountpoints": ["/host/mnt/x"]}
    ]
    host.open_counts[DM] = [1]
    host.calls.clear()
    assert storep.main(call("dm-destroy", "--name", "io")) == 0, host.emitted[-1]
    result = host.emitted[-1]
    evictions = [
        call for call in host.calls if call[0] == "nsenter" and "1" not in call[:3]
    ]
    assert evictions == [
        ["nsenter", "--target", "4242", "--mount", "--", "umount", "-l", "/host/mnt/x"]
    ], evictions
    assert result["holders"] == [
        {
            "pid": 4242,
            "comm": "pause",
            "locked": False,
            "mountpoints": ["/host/mnt/x"],
            "evicted": True,
        }
    ], result
    assert result["deferred"] == [] and result["destroyed"] is True, result
    assert (host.mappings, host.deferred) == ({}, []), "both mappings removed at once"
    assert host.clock.now - 1000.0 == 0, "no waiting once the holder is gone"


def test_destroy_defers_the_removal_when_a_locked_copy_keeps_the_mapping_open(
    host: FakeDmHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copy owned by another user namespace cannot be unmounted from outside:
    the holder is named, the removal is deferred to the kernel (it completes
    when that namespace ends), nothing raises, and the residue stays honest."""

    built(host)
    host.holders = [
        {"pid": 777, "comm": "sleep", "locked": True, "mountpoints": ["/x/mnt"]}
    ]
    host.open_counts[DM] = [1]
    started = host.clock.now
    assert storep.main(call("cleanup", "--kind", "dm-io")) == 0, host.emitted[-1]
    report = host.emitted[-1]
    destroyed = report["actions"][0]["device"]
    assert destroyed["holders"][0]["locked"] and not destroyed["holders"][0]["evicted"]
    assert destroyed["deferred"] == [DM] and host.deferred == [DM], destroyed
    assert host.clock.now - started >= storep.REMOVE_WAIT_SECONDS
    assert report["clean"] is False and DM in report["residue"]["device_mapper"]
    assert "io" not in storep.load_state(RUN_ID)["devices"], "the record is gone"


def test_destroy_waits_for_the_mappings_openers_before_the_remove(
    host: FakeDmHost,
) -> None:
    """The 2026-10-25 live cleanup: ``dmsetup remove --retry`` right after the
    unmount failed ``Device or resource busy`` (a lazily unmounted copy in
    another mount namespace, udev's blkid). The destroy now polls the open
    count for a bounded while and removes once it reaches 0."""

    built(host)
    host.open_counts[DM] = [2, 1, 0]
    started = host.clock.now
    host.calls.clear()
    assert storep.main(call("cleanup", "--kind", "dm-io")) == 0, host.emitted[-1]
    assert host.emitted[-1]["clean"] is True, host.emitted[-1]["residue"]
    verbs = [
        (tool_of(call), inner[1] if inner[0] == "dmsetup" else inner[-1])
        for call in host.device_calls()
        for inner in [call[len(NS) :] if call[: len(NS)] == NS else call]
        if tool_of(call) == "dmsetup"
    ]
    assert verbs.count(("dmsetup", "info")) >= 3, verbs
    assert verbs.index(("dmsetup", "remove")) > verbs.index(("dmsetup", "info"))
    assert host.clock.now - started == 2 * storep.REMOVE_POLL_SECONDS
    assert DM not in host.mappings


def test_destroy_defers_an_unidentified_opener_after_the_wait(host: FakeDmHost) -> None:
    """An opener no mount namespace explains (a process holding the device node)
    is waited out, then the removal is deferred and reported -- never a raise
    that would leave the loop device, the backing file and the record behind."""

    built(host)
    host.open_counts[DM] = [1]
    started = host.clock.now
    assert storep.main(call("dm-destroy", "--name", "io")) == 0, host.emitted[-1]
    result = host.emitted[-1]
    assert result["holders"] == [] and result["deferred"] == [DM], result
    assert host.clock.now - started >= storep.REMOVE_WAIT_SECONDS
    assert host.deferred == [DM] and host.loops == {}, "loop detached, mapping deferred"
    assert "io" not in storep.load_state(RUN_ID)["devices"]


def test_dm_create_loads_vfat_under_the_flag_and_cleanup_unloads_it(
    host: FakeDmHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(storep, "statvfs_reading", table_statvfs(host))
    assert not (host.sys_module / storep.VFAT_MODULE).exists(), "vfat starts unloaded"
    assert storep.main(create("--fstype", "vfat")) == 1
    assert host.emitted[-1]["error_class"] == "ProbeError", host.emitted[-1]
    assert host.calls == [], "refused before the deadman or any record"
    assert storep.main(create("--fstype", "vfat", "--load-module")) == 0, host.emitted[
        -1
    ]
    created = host.emitted[-1]
    assert created["modules_loaded_by_run"] == [storep.VFAT_MODULE], created
    assert storep.load_state(RUN_ID)["modules"] == [storep.VFAT_MODULE]
    assert ["modprobe", storep.VFAT_MODULE] in [call[:2] for call in host.calls]
    assert (host.sys_module / storep.VFAT_MODULE).is_dir(), "loaded for the run"
    assert storep.main(call("dm-setup", "--name", "io")) == 0, host.emitted[-1]
    assert storep.residue(RUN_ID)["modules_loaded_by_run"] == [storep.VFAT_MODULE]
    assert storep.main(call("cleanup", "--kind", "dm-io")) == 0, host.emitted[-1]
    result = host.emitted[-1]
    assert result["clean"] is True, result["residue"]
    assert {"modules_unloaded": [storep.VFAT_MODULE]} in result["actions"], result
    assert not (host.sys_module / storep.VFAT_MODULE).exists(), "unloaded again"
    assert ["modprobe", "-r", storep.VFAT_MODULE] in [call[:3] for call in host.calls]


def run_directories() -> tuple[Path, Path]:
    return storep.BACKING_ROOTS[0] / f"gfrs-live-{DIGEST}", storep.MOUNT_ROOT / DIGEST


def test_cleanup_prunes_the_runs_empty_directories(host: FakeDmHost) -> None:
    """The run's backing directory and mount root go with its last device.

    ``cleanup --kind dm-io`` -- the runner's undo and the deadman's command --
    pruned them only for ``--kind all``, which nothing calls: on 2026-10-25
    both EKS nodes held one empty ``gfrs-live-<digest>`` backing directory and
    one empty ``/mnt/gfrs-live/<digest>`` per finished 006 run.
    """

    built(host)
    backing, mount_root = run_directories()
    assert backing.is_dir() and mount_root.is_dir()
    assert storep.main(call("cleanup", "--kind", "dm-io")) == 0, host.emitted[-1]
    result = host.emitted[-1]
    assert result["clean"] is True, result["residue"]
    assert result["residue"]["directories"] == []
    assert not backing.exists() and not mount_root.exists()


def test_a_run_directory_with_content_is_residue(host: FakeDmHost) -> None:
    built(host)
    backing, mount_root = run_directories()
    (backing / "stray.bin").write_bytes(b"x")
    assert storep.main(call("cleanup", "--kind", "dm-io")) == 0, host.emitted[-1]
    result = host.emitted[-1]
    assert result["clean"] is False
    assert result["residue"]["directories"] == [str(backing)]
    assert not mount_root.exists(), "the empty mount root is still pruned"


def test_dm_destroy_and_fs_destroy_prune_the_runs_directories(host: FakeDmHost) -> None:
    built(host)
    backing, mount_root = run_directories()
    assert storep.main(call("dm-destroy", "--name", "io")) == 0, host.emitted[-1]
    assert not backing.exists() and not mount_root.exists()
    arguments = [
        "fs-create",
        "--run-id",
        RUN_ID,
        "--name",
        "fs",
        "--size-mib",
        "512",
        "--backing",
        str(storep.BACKING_ROOTS[0]),
        "--deadman-seconds",
        "1440",
    ]
    assert storep.main(arguments) == 0, host.emitted[-1]
    assert backing.is_dir() and (mount_root / "fs").is_dir()
    assert storep.main(call("fs-destroy", "--name", "fs")) == 0, host.emitted[-1]
    assert not backing.exists() and not mount_root.exists()


def test_retarget_data_with_delay_ms_swaps_the_delay_table(host: FakeDmHost) -> None:
    """``dm-retarget --target data --delay-ms 0`` takes the injected latency
    out of the mounted device in place: same mapping, dm-N, mount and files.

    The 006 ``sustained_latency`` recovery proof needs it: the product emits
    no await sample without completed I/O, and with the 200 ms table still
    loaded any I/O would read above the threshold (live 2026-10-25 14:47Z).
    """

    assert storep.main(create("--load-module", delay=200)) == 0, host.emitted[-1]
    assert storep.main(call("dm-setup", "--name", "io")) == 0, host.emitted[-1]
    assert host.tables[DM] == "0 4194304 delay /dev/loop0 0 200", host.tables
    before = device()["device"]
    host.calls.clear()
    assert (
        storep.main(
            call("dm-retarget", "--name", "io", "--target", "data", "--delay-ms", "0")
        )
        == 0
    ), host.emitted[-1]
    answer = host.emitted[-1]
    assert answer["delay_ms"] == 0 and answer["target"] == "data", answer
    assert host.tables[DM] == "0 4194304 linear /dev/loop0 0", host.tables
    assert device()["delay_ms"] == 0 and device()["device"] == before, device()
    assert host.mounts, "the mount stays"
    steps = [call[len(NS) :][:2] for call in host.device_calls()]
    assert steps == [
        ["dmsetup", "suspend"],
        ["dmsetup", "reload"],
        ["dmsetup", "resume"],
    ], steps
    # Back to a delay table needs the module; without it the swap is refused.
    shutil.rmtree(host.sys_module / storep.DELAY_MODULE)
    assert (
        storep.main(
            call("dm-retarget", "--name", "io", "--target", "data", "--delay-ms", "50")
        )
        == 1
    )
    assert host.emitted[-1]["error"] == "ProbeError", host.emitted[-1]
    assert host.tables[DM] == "0 4194304 linear /dev/loop0 0", host.tables


def test_io_start_paces_the_writer(host: FakeDmHost) -> None:
    built(host)
    assert (
        storep.main(
            call(
                "io-start",
                "--name",
                "io",
                "--threads",
                "1",
                "--seconds",
                "120",
                "--direct",
                "--pace-ms",
                "100",
            )
        )
        == 0
    ), host.emitted[-1]
    answer = host.emitted[-1]
    assert answer["pace_ms"] == 100 and answer["threads"] == 1, answer
    writer = host.started[WRITER_UNIT]
    assert writer[-1] == "0.1", writer[-8:]  # the pace in seconds, after the span
    assert "if pace:" in writer[-7] or "pace" in writer[-7], "the writer honours it"
