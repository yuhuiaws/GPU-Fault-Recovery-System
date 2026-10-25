"""The shared-mount probe (GF-GENERAL-LIVE-005 client side) on a fake host.

A recording fake answers ``nsenter``, ``mount``/``umount``, systemd and
``modprobe`` at the probe's ``subprocess`` boundary over a fake host mount
table, and the host-namespace readings (``host_python``) are stubbed per
snippet. The tests pin: every mount and reading happens in PID 1's namespaces,
the deadman and the record precede the mount, only soft mounts of an IPv4
server and a test export path are made, the unmount cascade and the cleanup
leave nothing, and the module this run loaded is unloaded again. The reading
snippets themselves run once for real on ``tmp_path``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_shared_probe as sharedp

RUN_ID = "gl005-stat-timeout-a1-deadbeef"
NS = list(sharedp.HOST_NAMESPACES)
DIGEST = sharedp.run_digest(RUN_ID)
DEADMAN = f"gfrs-{DIGEST}-ndeadman-nfs-shared"
SERVER = "10.0.9.7"


class FakeHost:
    def __init__(self, root: Path) -> None:
        self.calls: list[list[str]] = []
        self.mounts: dict[str, str] = {}
        self.timers: set[str] = set()
        self.services: dict[str, str] = {}
        self.started: dict[str, list[str]] = {}
        self.mountinfo = root / "mountinfo"
        self.sys_module = root / "sys-module"
        self.sys_module.mkdir()
        self.umount_failures = 0
        self.emitted: list[dict[str, Any]] = []
        self.write_mountinfo()

    def write_mountinfo(self) -> None:
        lines = ["22 1 259:1 / / rw,relatime shared:1 - xfs /dev/nvme0n1p1 rw"]
        for index, target in enumerate(sorted(self.mounts), start=40):
            lines.append(
                f"{index} 22 0:{index} / {target} rw,relatime shared:{index} "
                f"- nfs4 {self.mounts[target]} rw,vers=4.1"
            )
        self.mountinfo.write_text("\n".join(lines) + "\n")

    def run(self, command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        command = list(command)
        self.calls.append(command)
        hosted = command[: len(NS)] == NS
        inner = command[len(NS) :] if hosted else command
        tool = inner[0]
        if tool in {"mount", "umount"}:
            assert hosted, f"{tool} must join the host namespaces: {command}"
        code, out = self.answer(inner)
        return subprocess.CompletedProcess(command, code, stdout=out, stderr="")

    def answer(self, inner: list[str]) -> tuple[int, str]:
        tool, args = inner[0], inner[1:]
        if tool == "mount":
            self.mounts[args[-1]] = args[-2]
            self.write_mountinfo()
            return 0, ""
        if tool == "umount":
            target = args[-1]
            if target not in self.mounts:
                return 32, ""
            if self.umount_failures and "-l" not in args:
                self.umount_failures -= 1
                return 32, "target is busy"
            del self.mounts[target]
            self.write_mountinfo()
            return 0, ""
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
        if tool == "modprobe":
            if args[0] == "-r":
                (self.sys_module / args[1]).rmdir()
            else:
                (self.sys_module / args[0]).mkdir(exist_ok=True)
            return 0, ""
        raise AssertionError(f"unexpected host command {inner}")

    def systemctl(self, args: list[str]) -> tuple[int, str]:
        verb = args[0]
        if verb == "stop":
            base, _, suffix = args[1].rpartition(".")
            (self.timers.discard if suffix == "timer" else self.services.pop)(
                base, None
            ) if (suffix != "timer") else self.timers.discard(base)
            return 0, ""
        if verb == "reset-failed":
            return 0, ""
        if verb == "show":
            base, _, suffix = args[1].rpartition(".")
            if suffix == "timer":
                state = "active" if base in self.timers else "inactive"
            else:
                state = self.services.get(base, "inactive")
            return 0, f"LoadState=loaded\nActiveState={state}\nMainPID=0\n"
        if verb == "list-units":
            pattern = args[-1]
            prefix = pattern.split("*")[0]
            listed = sorted(unit for unit in self.timers if unit.startswith(prefix))
            return 0, "".join(
                f"{unit}.timer loaded active waiting\n" for unit in listed
            )
        raise AssertionError(f"unexpected systemctl {args}")


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeHost:
    fake = FakeHost(tmp_path)
    monkeypatch.setattr(sharedp, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(sharedp, "MOUNT_ROOT", tmp_path / "mnt")
    monkeypatch.setattr(sharedp, "HOST_MOUNTINFO", fake.mountinfo)
    monkeypatch.setattr(sharedp, "SYS_MODULE", fake.sys_module)
    monkeypatch.setattr(sharedp, "tool_present", lambda name: True)
    monkeypatch.setattr(sharedp, "emit", fake.emitted.append)
    monkeypatch.setattr(
        sharedp,
        "subprocess",
        types.SimpleNamespace(
            run=fake.run,
            PIPE=subprocess.PIPE,
            CompletedProcess=subprocess.CompletedProcess,
            TimeoutExpired=subprocess.TimeoutExpired,
        ),
    )
    readings: dict[str, dict[str, Any]] = {
        sharedp.STATVFS_SNIPPET: {"total_bytes": 256 << 20, "used_percent": 1.0},
        sharedp.IO_SNIPPET: {"ok": True, "sha256_match": True},
        sharedp.FILL_SNIPPET: {"fill_bytes": 1, "statvfs": {"used_percent": 99.0}},
        sharedp.UNFILL_SNIPPET: {"removed": 1},
    }
    fake.readings = readings  # type: ignore[attr-defined]
    fake.python_calls = []  # type: ignore[attr-defined]

    def host_python(snippet: str, *arguments: str, timeout: float) -> dict[str, Any]:
        fake.python_calls.append((snippet, arguments, timeout))  # type: ignore[attr-defined]
        return dict(readings[snippet])

    monkeypatch.setattr(sharedp, "host_python", host_python)
    (fake.sys_module / sharedp.NFS_MODULE).mkdir()
    return fake


def mount(
    *extra: str, name: str = "shared", server: str = SERVER, export: str = "/a"
) -> list[str]:
    return [
        "nfs-mount",
        "--run-id",
        RUN_ID,
        "--name",
        name,
        "--server",
        server,
        "--export",
        export,
        "--deadman-seconds",
        "2700",
        *extra,
    ]


def call(subcommand: str, *extra: str) -> list[str]:
    return [subcommand, "--run-id", RUN_ID, *extra]


def target() -> str:
    return str(sharedp.MOUNT_ROOT / DIGEST / "shared")


def test_mount_arms_the_deadman_records_then_mounts_soft_in_the_host_namespace(
    host: FakeHost,
) -> None:
    assert sharedp.main(mount()) == 0, host.emitted[-1]
    result = host.emitted[-1]
    assert result["mountpoint"] == target() and result["host_mount"]["fstype"] == "nfs4"
    assert result["statvfs"]["total_bytes"] == 256 << 20
    assert result["module_loaded_by_run"] is False
    assert DEADMAN in host.timers, host.timers
    deadman = host.started[DEADMAN]
    assert "--on-active=2700s" in deadman, deadman
    assert "cleanup" in deadman and deadman[-2:] == ["--kind", "nfs-shared"], deadman
    assert str(sharedp.probe_copy(RUN_ID)) in deadman, (
        "the deadman runs a copy of this file"
    )
    order = [
        (
            "systemd-run"
            if c[0] == "systemd-run"
            else c[len(NS)]
            if c[: len(NS)] == NS
            else c[0]
        )
        for c in host.calls
    ]
    assert order.index("systemd-run") < order.index("mount"), order
    mounted = next(
        c for c in host.calls if c[: len(NS)] == NS and c[len(NS)] == "mount"
    )
    assert mounted[len(NS) :] == [
        "mount",
        "-t",
        "nfs4",
        "-o",
        sharedp.DEFAULT_OPTIONS,
        f"{SERVER}:/a",
        target(),
    ], mounted
    state = sharedp.load_state(RUN_ID)
    assert state["mounts"]["shared"]["mounted"] is True and state["modules"] == []
    assert Path(target()).is_dir(), "the mount point directory exists"


@pytest.mark.parametrize(
    "argv",
    [
        mount(server="export.internal"),
        mount(server="10.0.0.1", export="/../etc"),
        mount("--options", "vers=4.1,hard,timeo=600"),
        mount(name="Shared"),
    ],
    ids=["hostname", "export-path", "hard", "name"],
)
def test_mount_refuses_unsafe_inputs_before_any_host_command(
    host: FakeHost, argv: list[str]
) -> None:
    assert sharedp.main(argv) == 1
    assert host.emitted[-1]["error_class"] == "ProbeError", host.emitted[-1]
    assert host.calls == [], host.calls
    assert sharedp.load_state(RUN_ID)["mounts"] == {}


def test_mount_loads_nfsv4_only_under_the_flag_and_cleanup_unloads_it(
    host: FakeHost,
) -> None:
    (host.sys_module / sharedp.NFS_MODULE).rmdir()
    assert sharedp.main(mount()) == 1
    assert host.emitted[-1]["error_class"] == "ProbeError", host.emitted[-1]
    assert host.calls == [], "refused before the deadman"
    assert sharedp.main(mount("--load-module")) == 0, host.emitted[-1]
    assert host.emitted[-1]["module_loaded_by_run"] is True
    assert ["modprobe", "nfsv4"] in host.calls, host.calls
    assert sharedp.load_state(RUN_ID)["modules"] == ["nfsv4"]
    assert sharedp.residue(RUN_ID)["modules_loaded_by_run"] == ["nfsv4"]
    assert sharedp.main(call("cleanup")) == 0, host.emitted[-1]
    result = host.emitted[-1]
    assert result["clean"] is True, result["residue"]
    assert {"modules_unloaded": ["nfsv4"]} in result["actions"], result["actions"]
    assert not (host.sys_module / "nfsv4").exists(), (
        "(host.sys_module / 'nfsv4').exists() is still present"
    )


def test_umount_cascades_to_force_and_lazy_and_forget_removes_the_record(
    host: FakeHost,
) -> None:
    assert sharedp.main(mount()) == 0, host.emitted[-1]
    host.umount_failures = 2
    host.calls.clear()
    assert sharedp.main(call("nfs-umount", "--name", "shared")) == 0, host.emitted[-1]
    result = host.emitted[-1]
    assert [step["flags"] for step in result["umount"]["steps"]] == [[], ["-f"], ["-l"]]
    assert result["host_mount"] is None and result["forgotten"] is False
    assert sharedp.load_state(RUN_ID)["mounts"]["shared"]["mounted"] is False
    assert DEADMAN in host.timers, "the record and its deadman stay for a remount"
    assert sharedp.main(call("nfs-remount", "--name", "shared")) == 0, host.emitted[-1]
    assert host.emitted[-1]["host_mount"]["fstype"] == "nfs4"
    assert sharedp.main(call("nfs-umount", "--name", "shared", "--forget")) == 0
    assert sharedp.load_state(RUN_ID)["mounts"] == {}
    assert DEADMAN not in host.timers and not Path(target()).exists()


def test_cleanup_unmounts_everything_and_disarms_an_unrecorded_deadman(
    host: FakeHost,
) -> None:
    assert sharedp.main(mount()) == 0, host.emitted[-1]
    assert sharedp.main(mount(name="other", server="10.0.9.8")) == 0, host.emitted[-1]
    host.timers.add(f"gfrs-{DIGEST}-ndeadman-nfs-ghost")
    assert sharedp.main(call("cleanup", "--kind", "nfs-ghost")) == 0, host.emitted[-1]
    assert f"gfrs-{DIGEST}-ndeadman-nfs-ghost" not in host.timers
    assert host.emitted[-1]["clean"] is False, "two mounts remain"
    assert sharedp.main(call("cleanup")) == 0, host.emitted[-1]
    result = host.emitted[-1]
    assert result["clean"] is True, result["residue"]
    assert host.mounts == {} and host.timers == set()
    assert not (sharedp.MOUNT_ROOT / DIGEST).exists(), (
        "(sharedp.MOUNT_ROOT / DIGEST).exists() is still present"
    )
    assert sharedp.main(call("residue")) == 0 and host.emitted[-1]["clean"] is True


def test_readings_run_in_the_host_namespace_with_their_bounds(host: FakeHost) -> None:
    assert sharedp.main(mount()) == 0, host.emitted[-1]
    host.python_calls.clear()  # type: ignore[attr-defined]
    assert sharedp.main(call("nfs-stat", "--name", "shared", "--timeout", "7")) == 0
    snippet, arguments, timeout = host.python_calls[-1]  # type: ignore[attr-defined]
    assert snippet == sharedp.STATVFS_SNIPPET and arguments == (target(), "7.0")
    assert timeout == 22.0, "the subprocess backstop is the deadline plus 15 s"
    assert sharedp.main(call("nfs-io", "--name", "shared", "--mib", "4")) == 0
    snippet, arguments, _ = host.python_calls[-1]  # type: ignore[attr-defined]
    assert snippet == sharedp.IO_SNIPPET and arguments == (target(), "4", "0")
    assert host.emitted[-1]["io"]["ok"] is True
    assert sharedp.main(call("nfs-io", "--name", "shared", "--mib", "999")) == 1
    assert sharedp.main(call("nfs-fill", "--name", "shared", "--percent", "99")) == 0
    assert host.emitted[-1]["statvfs"]["used_percent"] == 99.0
    assert sharedp.main(call("nfs-fill", "--name", "shared", "--percent", "0")) == 1
    assert sharedp.main(call("nfs-unfill", "--name", "shared")) == 0
    assert host.emitted[-1]["removed"] == 1
    assert sharedp.main(call("nfs-stat", "--name", "ghost")) == 1
    assert host.emitted[-1]["error_class"] == "ProbeError"


def test_host_python_reports_a_timeout_as_a_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(command: list[str], **options: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, options["timeout"])

    monkeypatch.setattr(sharedp, "tool_present", lambda name: True)
    monkeypatch.setattr(sharedp, "run", run)
    reading = sharedp.host_python(sharedp.STATVFS_SNIPPET, "/x", "5", timeout=20)
    assert reading["timed_out"] is True and reading["elapsed"] == 20

    def garbage(command: list[str], **options: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command, 1, stdout="not json\n", stderr="boom"
        )

    monkeypatch.setattr(sharedp, "run", garbage)
    reading = sharedp.host_python(sharedp.IO_SNIPPET, "/x", timeout=20)
    assert "error" in reading and reading["stderr"] == "boom"


# --------------------------------------------------------------------------- #
# the reading snippets, for real
# --------------------------------------------------------------------------- #
def snippet(code: str, *arguments: str) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code, *arguments],
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    value = json.loads(completed.stdout.strip().splitlines()[-1])
    assert isinstance(value, dict), value
    return value


def test_statvfs_snippet_reads_a_mount_and_names_an_error(tmp_path: Path) -> None:
    reading = snippet(sharedp.STATVFS_SNIPPET, str(tmp_path), "5")
    assert reading["total_bytes"] > 0 and "used_percent" in reading, reading
    assert "timed_out" not in reading and "error" not in reading, reading
    missing = snippet(sharedp.STATVFS_SNIPPET, str(tmp_path / "gone"), "5")
    assert missing["error"] == "FileNotFoundError: 2", missing


def test_io_snippet_writes_fsyncs_reads_back_and_cleans_up(tmp_path: Path) -> None:
    reading = snippet(sharedp.IO_SNIPPET, str(tmp_path), "1", "0")
    assert reading["ok"] is True and reading["sha256_match"] is True
    assert reading["bytes"] == 1 << 20 and not list((tmp_path / "gfrs-io").iterdir())
    kept = snippet(sharedp.IO_SNIPPET, str(tmp_path), "1", "1")
    assert Path(kept["path"]).stat().st_size == 1 << 20
    blocker = tmp_path / "file.txt"
    blocker.write_text("not a directory")
    denied = snippet(sharedp.IO_SNIPPET, str(blocker / "below"), "1", "0")
    assert denied["ok"] is False and denied["errno"] in {20, 2}, denied


def test_fill_and_unfill_snippets_are_statvfs_driven(tmp_path: Path) -> None:
    # A target below the current usage writes nothing; the shape is what counts.
    filled = snippet(sharedp.FILL_SNIPPET, str(tmp_path), "0.0000001")
    assert filled["fill_bytes"] == 0 and filled["enospc"] is False
    assert "used_percent" in filled["statvfs"]
    (tmp_path / "gfrs-io").mkdir()
    (tmp_path / "gfrs-io" / "proof-1.bin").write_bytes(b"x")
    cleaned = snippet(sharedp.UNFILL_SNIPPET, str(tmp_path))
    assert cleaned["removed"] == 2 and not (tmp_path / "gfrs-io").exists()
    assert not (tmp_path / "gfrs-fill.bin").exists(), (
        "(tmp_path / 'gfrs-fill.bin').exists() is still present"
    )


# --------------------------------------------------------------------------- #
# mount failures (live attempt 5: every variant stopped on a withheld message)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("stderr", "needle"),
    [
        (
            "mount.nfs4: mounting 10.0.9.7:/a failed, reason given by server: "
            "No such file or directory",
            "ENOENT",
        ),
        ("mount.nfs4: access denied by server while mounting 10.0.9.7:/a", "denied"),
        ("mount.nfs4: Connection refused", "refused"),
        ("mount.nfs4: Connection timed out", "timed out"),
        ("mount.nfs4: No route to host", "route"),
        ("mount.nfs4: Operation not permitted", "privilege"),
        ("mount: /mnt/x: unknown filesystem type 'nfs4'.", "nfsv4 module"),
        ("mount.nfs4: Protocol not supported", "not supported"),
        ("mount.nfs4: an incorrect mount option was specified", "options"),
        ("something new from /secret/path token=abc", "unclassified"),
    ],
    ids=[
        "enoent",
        "denied",
        "refused",
        "timeout",
        "route",
        "eperm",
        "fstype",
        "proto",
        "options",
        "unknown",
    ],
)
def test_a_failed_mount_names_its_cause_in_a_constant_sentence(
    host: FakeHost, stderr: str, needle: str
) -> None:
    """The mount helper's stderr never leaves the node; the probe classifies it
    into one constant sentence (``error_kind: nfs_mount``) the runner may show,
    so a verdict reads e.g. "export server answered ENOENT" instead of a digest.
    """

    def failing(inner: list[str]) -> tuple[int, str, str]:
        if inner[0] == "mount":
            return 32, "", stderr
        code, out = host.answer(inner)
        return code, out, ""

    def run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        command = list(command)
        host.calls.append(command)
        inner = command[len(NS) :] if command[: len(NS)] == NS else command
        code, out, err = failing(inner)
        return subprocess.CompletedProcess(command, code, stdout=out, stderr=err)

    sharedp.subprocess.run = run  # type: ignore[attr-defined]
    assert sharedp.main(mount()) == 1, host.emitted[-1]
    result = host.emitted[-1]
    assert result["error_kind"] == "nfs_mount", result
    assert result["error_class"] == "MountError", result
    message = result["message"]
    assert needle.casefold() in message.casefold(), (message, stderr)
    assert message == sharedp.classify_mount_failure(stderr), (message, stderr)
    assert sharedp.CONSTANT_MESSAGE.fullmatch(message), message
    for secret in ("10.0.9.7", "/mnt", "/secret", "token"):
        assert secret not in message, message
    assert result["error_site"].startswith("nfs_mount:"), result
    state = sharedp.load_state(RUN_ID)
    assert state["mounts"]["shared"]["mounted"] is False, "recorded, never mounted"
    assert DEADMAN in host.timers, "the deadman stays armed until cleanup"


def test_every_mount_failure_sentence_is_constant_and_distinct() -> None:
    sentences = [sentence for _needles, sentence in sharedp.MOUNT_FAILURE_REASONS]
    sentences.append(sharedp.MOUNT_FAILURE_UNCLASSIFIED)
    assert len(set(sentences)) == len(sentences), sentences
    for sentence in sentences:
        assert sharedp.CONSTANT_MESSAGE.fullmatch(sentence), sentence
