"""Exercise the E2E probe against private files and a closed command transport."""

from __future__ import annotations

import errno
import json
import os
import runpy
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

import pytest

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
PROBE = lazy_script_module(ROOT / "scripts/e2e/regional/probes/e2e001_node_probe.py")
SERVICE: tuple[str, ...] = (
    "systemctl",
    "show",
    "gpu-fault-kernel-collector.service",
    "--property=ActiveState",
    "--property=SubState",
    "--property=NRestarts",
)
GPU: tuple[str, ...] = ("nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader")
MARKER = "fixture-owned:phase.1"


class FixtureHost:
    def __init__(self, root: Path) -> None:
        self.boot = root / "boot-id"
        self.boot.write_text("fixture-boot\n", encoding="utf-8")
        self.kmsg = root / "private-kmsg"
        self.kmsg.touch()
        self.responses = {
            SERVICE: subprocess.CompletedProcess(
                SERVICE,
                0,
                "ActiveState=active\nSubState=running\nNRestarts=4\n"
                "\nunstructured row\nOpaque=left=right\n",
                "",
            ),
            GPU: subprocess.CompletedProcess(
                GPU, 0, "\n 00000000:AF:00.0 \n00000000:BF:00.0\n", ""
            ),
        }
        self.commands: list[tuple[str, ...]] = []
        self.reads: list[str] = []
        self.accesses: list[tuple[str, int]] = []
        self.events: list[tuple[Any, ...]] = []
        self.descriptors: set[int] = set()
        self.writable = True
        self.failure: str | None = None

    def run(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        assert kwargs == {
            "text": True,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "check": False,
        }
        key = tuple(command)
        self.commands.append(key)
        assert key in self.responses, "an unexpected command must never execute"
        return self.responses[key]

    def path(self, value: str) -> Path:
        self.reads.append(value)
        assert value == "/proc/sys/kernel/random/boot_id"
        return self.boot

    def access(self, path: str, mode: int) -> bool:
        assert (path, mode) == ("/dev/kmsg", os.W_OK)
        self.accesses.append((path, mode))
        return self.writable

    def open(self, path: str, flags: int) -> int:
        assert (path, flags) == ("/dev/kmsg", os.O_WRONLY | os.O_CLOEXEC)
        self.events.append(("open", path, flags))
        if self.failure == "open":
            raise PermissionError(errno.EACCES, "fixture open refused")
        descriptor = os.open(self.kmsg, flags)
        self.descriptors.add(descriptor)
        return descriptor

    def write(self, descriptor: int, payload: bytes) -> int:
        assert descriptor in self.descriptors
        self.events.append(("write", descriptor, payload))
        if self.failure == "write":
            raise OSError(errno.EIO, "fixture write failed")
        return os.write(descriptor, payload)

    def close(self, descriptor: int) -> None:
        assert descriptor in self.descriptors
        self.events.append(("close", descriptor))
        os.close(descriptor)
        self.descriptors.remove(descriptor)


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FixtureHost]:
    fixture = FixtureHost(tmp_path)
    monkeypatch.setattr(PROBE, "Path", fixture.path)
    monkeypatch.setattr(
        PROBE, "subprocess", SimpleNamespace(run=fixture.run, PIPE=subprocess.PIPE)
    )
    monkeypatch.setattr(
        PROBE,
        "os",
        SimpleNamespace(
            access=fixture.access,
            open=fixture.open,
            write=fixture.write,
            close=fixture.close,
            W_OK=os.W_OK,
            O_WRONLY=os.O_WRONLY,
            O_CLOEXEC=os.O_CLOEXEC,
        ),
    )
    try:
        yield fixture
    finally:
        leaked = set(fixture.descriptors)
        for descriptor in leaked:
            fixture.close(descriptor)
        assert not leaked, "the probe must close every fixture descriptor"


@pytest.mark.parametrize("writable", [False, True])
def test_snapshot_preserves_identity_and_collector_properties(
    host: FixtureHost, writable: bool
) -> None:
    host.writable = writable

    assert PROBE.snapshot() == {
        "boot_id": "fixture-boot",
        "gpu_bdf": "0000:af:00.0",
        "kernel_collector": {
            "ActiveState": "active",
            "SubState": "running",
            "NRestarts": "4",
            "Opaque": "left=right",
        },
        "kmsg_writable": writable,
    }
    assert host.commands == [SERVICE, GPU]
    assert host.reads == ["/proc/sys/kernel/random/boot_id"]
    assert host.accesses == [("/dev/kmsg", os.W_OK)]
    assert host.events == []


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("\n 00000000:AF:00.0 \n00000000:BF:00.0\n", "0000:af:00.0"),
        ("0000:AB:01.2\n", "0000:ab:01.2"),
        ("AB:01.7\n", "ab:01.7"),
    ],
)
def test_gpu_identity_uses_the_first_nonempty_device(
    host: FixtureHost, output: str, expected: str
) -> None:
    host.responses[GPU] = subprocess.CompletedProcess(GPU, 0, output, "")
    assert PROBE.gpu_bdf() == expected
    assert host.commands == [GPU]


@pytest.mark.parametrize(
    "output", ["", " \n\t", "unavailable\n00000000:AF:00.0\n", "0000:AF:00.8", "AF:00"]
)
def test_gpu_identity_refuses_missing_or_invalid_first_device(
    host: FixtureHost, output: str
) -> None:
    host.responses[GPU] = subprocess.CompletedProcess(GPU, 0, output, "")
    with pytest.raises(PROBE.ProbeError, match="cannot parse GPU PCI BDF"):
        PROBE.gpu_bdf()
    assert host.commands == [GPU]
    assert host.events == []


@pytest.mark.parametrize("command", [SERVICE, GPU])
def test_snapshot_refuses_failed_command_without_writing(
    host: FixtureHost, command: tuple[str, ...]
) -> None:
    host.responses[command] = subprocess.CompletedProcess(
        command, 9, "untrusted partial output", " fixture permission denied \n"
    )
    with pytest.raises(
        PROBE.ProbeError, match=r"^command failed \(9\): fixture permission denied$"
    ):
        PROBE.snapshot()
    assert host.commands == ([SERVICE] if command == SERVICE else [SERVICE, GPU])
    assert host.accesses == []
    assert host.events == []


@pytest.mark.parametrize(
    ("marker", "bdf", "error"),
    [
        ("", "0000:af:00.0", "unsafe marker"),
        ("a" * 129, "0000:af:00.0", "unsafe marker"),
        ("../outside", "0000:af:00.0", "unsafe marker"),
        ("owned\n<3>forged", "0000:af:00.0", "unsafe marker"),
        ("-option", "0000:af:00.0", "unsafe marker"),
        (MARKER, "", "unsafe PCI BDF"),
        (MARKER, "0000:af:00.8", "unsafe PCI BDF"),
        (MARKER, "0000:af:00.0\n", "unsafe PCI BDF"),
        (MARKER, "af:00.0;touch /tmp/foreign", "unsafe PCI BDF"),
    ],
)
def test_invalid_injection_identity_is_rejected_before_io(
    host: FixtureHost, marker: str, bdf: str, error: str
) -> None:
    with pytest.raises(PROBE.ProbeError, match=f"^{error}$"):
        PROBE.write_xid11(marker, bdf)
    assert host.commands == []
    assert host.events == []
    assert host.kmsg.read_bytes() == b""


@pytest.mark.parametrize(
    ("bdf", "wire_bdf"), [("0000:AF:00.0", "0000:af:00"), ("AF:00.7", "af:00.7")]
)
def test_xid_writer_encodes_one_record_and_closes_its_only_descriptor(
    host: FixtureHost, bdf: str, wire_bdf: str
) -> None:
    expected = (
        f"<3>NVRM: Xid (PCI:{wire_bdf}): 11, Ch 00000001, "
        "regional E2E workload restart marker=fixture-owned:phase.1\n"
    ).encode()

    assert PROBE.write_xid11(MARKER, bdf) == {
        "marker": MARKER,
        "pci_bdf": bdf,
        "bytes_written": len(expected),
    }
    descriptor = host.events[1][1]
    assert host.events == [
        ("open", "/dev/kmsg", os.O_WRONLY | os.O_CLOEXEC),
        ("write", descriptor, expected),
        ("close", descriptor),
    ]
    assert host.kmsg.read_bytes() == expected
    assert host.descriptors == set()
    with pytest.raises(OSError) as closed:
        os.fstat(descriptor)
    assert closed.value.errno == errno.EBADF


@pytest.mark.parametrize("failure", ["open", "write"])
def test_main_reports_io_failure_and_releases_any_open_descriptor(
    host: FixtureHost,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    host.failure = failure
    monkeypatch.setattr(
        sys,
        "argv",
        ["probe", "write-xid11", "--marker", MARKER, "--pci-bdf", "0000:af:00.0"],
    )
    assert PROBE.main() == 1
    error = (
        PermissionError(errno.EACCES, "fixture open refused")
        if failure == "open"
        else OSError(errno.EIO, "fixture write failed")
    )
    assert json.loads(capsys.readouterr().out) == {"error": str(error)}
    assert [event[0] for event in host.events] == (
        ["open"] if failure == "open" else ["open", "write", "close"]
    )
    assert host.descriptors == set()
    assert host.kmsg.read_bytes() == b""


@pytest.mark.parametrize("command", ["snapshot", "write-xid11"])
def test_main_emits_only_the_result_of_the_selected_command(
    host: FixtureHost,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    arguments = (
        [] if command == "snapshot" else ["--marker", MARKER, "--pci-bdf", "af:00.0"]
    )
    monkeypatch.setattr(sys, "argv", ["probe", command, *arguments])
    assert PROBE.main() == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert captured.err == ""
    if command == "snapshot":
        assert result["boot_id"] == "fixture-boot"
        assert result["gpu_bdf"] == "0000:af:00.0"
        assert host.commands == [SERVICE, GPU]
        assert host.events == []
    else:
        assert result == {
            "marker": MARKER,
            "pci_bdf": "af:00.0",
            "bytes_written": len(host.kmsg.read_bytes()),
        }
        assert host.kmsg.read_bytes().endswith(b"marker=fixture-owned:phase.1\n"), (
            "the synthetic fixture event must retain its exact marker and terminator"
        )
        assert host.commands == []


def test_main_reports_missing_boot_evidence_without_querying_the_gpu(
    host: FixtureHost,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    host.boot.unlink()
    monkeypatch.setattr(sys, "argv", ["probe", "snapshot"])
    assert PROBE.main() == 1
    assert "No such file or directory" in json.loads(capsys.readouterr().out)["error"]
    assert host.commands == [SERVICE]
    assert host.accesses == []
    assert host.events == []


def test_script_entry_restricts_permissions_and_refuses_before_any_host_io(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    masks: list[int] = []
    forbidden: list[str] = []

    def refuse(*args: Any, **kwargs: Any) -> NoReturn:
        forbidden.append("host I/O")
        raise AssertionError("invalid input must not reach host I/O")

    monkeypatch.setattr(
        sys,
        "argv",
        ["probe", "write-xid11", "--marker", "../unsafe", "--pci-bdf", "af:00.0"],
    )
    with monkeypatch.context() as guarded:
        guarded.setattr(os, "umask", masks.append)
        guarded.setattr(os, "open", refuse)
        guarded.setattr(os, "access", refuse)
        guarded.setattr(Path, "read_text", refuse)
        guarded.setattr(subprocess, "run", refuse)
        with pytest.raises(SystemExit) as stopped:
            runpy.run_path(str(PROBE.path), run_name="__main__")
    assert stopped.value.code == 1
    assert masks == [0o077]
    assert forbidden == []
    assert json.loads(capsys.readouterr().out) == {"error": "unsafe marker"}
