"""GF-GENERAL-LIVE host probe: loads, deadmans, pause bookkeeping and cleanup.

Every command the runner sends (``oracle-*``, ``cpu-*``, ``page-cache-*``,
``memory-stop``, ``install-source``, ``cleanup``, ``residue``) runs through the
probe's own parser against ``NodeHost``: a recorded systemd whose transient
units come and go, with ``/proc`` and ``/run`` under ``tmp_path``. The tests
check what each command starts, records, refuses and leaves behind, and that
the file still runs as the script the deadman executes.
"""

from __future__ import annotations

import base64
import filecmp
import hashlib
import json
import os
import runpy
import shlex
import signal
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_node_probe as nodep
from tests.regional._cov95_glive_node import RUN, NodeHost, command, node_host

HOST_UNIT = "gpu-fault-host-collector.service"
KERNEL_UNIT = "gpu-fault-kernel-collector.service"
EXPORTER = ["/usr/bin/dcgm-exporter", "-f", "/etc/dcgm/counters.csv"]
GONE = {
    "LoadState": "not-found",
    "ActiveState": "inactive",
    "SubState": "dead",
    "MainPID": "0",
    "returncode": "0",
}


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> NodeHost:
    return node_host(monkeypatch, tmp_path)


def statvfs_answers(
    monkeypatch: pytest.MonkeyPatch, answers: dict[str, tuple[int, int]]
) -> None:
    """``os.statvfs`` for ``answers`` paths: (total, available) 4 KiB blocks."""

    real = os.statvfs

    def fake(path: Any) -> Any:
        if str(path) not in answers:
            return real(path)
        blocks, free = answers[str(path)]
        return os.statvfs_result((4096, 4096, blocks, free, free, 100, 90, 90, 0, 255))

    monkeypatch.setattr(nodep.os, "statvfs", fake)


def resumable(host: NodeHost, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """A fake ``os.kill``: SIGCONT lets the fake process run again."""

    sent: list[Any] = []

    def kill(pid: int, sig: int) -> None:
        sent.append((pid, sig))
        host.process(pid, EXPORTER, state="S")

    monkeypatch.setattr(nodep.os, "kill", kill)
    return sent


def deadman_stopped(host: NodeHost, kind: str) -> bool:
    timer = f"{nodep.unit_name(RUN, f'deadman-{kind}')}.timer"
    return ["systemctl", "stop", timer] in host.commands


# --------------------------------------------------------------------------- #
# state, the deadman's copy, the deadman itself
# --------------------------------------------------------------------------- #
def test_load_state_refuses_a_state_file_of_another_run(host: NodeHost) -> None:
    nodep.save_state(RUN, {"run_id": "gl-cov95-other-a1-cafe"})
    with pytest.raises(nodep.ProbeError, match="state identity mismatch"):
        nodep.load_state(RUN)
    (nodep.state_dir(RUN) / "state.json").write_text("[]")
    with pytest.raises(nodep.ProbeError, match="state identity mismatch"):
        nodep.load_state(RUN)
    nodep.save_state(RUN, {"run_id": RUN, "units": ["gfrs-x-cpu"]})
    assert nodep.load_state(RUN) == {
        "run_id": RUN,
        "units": ["gfrs-x-cpu"],
        "paused_units": {},
        "paused_processes": {},
        "page_cache_files": [],
        "log_collector": None,
    }, "an older state file gains the keys it lacks"


def test_probe_copy_installs_one_private_copy_for_the_deadman(host: NodeHost) -> None:
    copy = nodep.probe_copy(RUN)
    assert copy == nodep.state_dir(RUN) / "probe.py"
    assert filecmp.cmp(copy, nodep.__file__, shallow=False), "a byte copy"
    assert stat.S_IMODE(copy.stat().st_mode) == 0o700
    copy.write_text("# the copy an earlier step installed\n")
    assert nodep.probe_copy(RUN) == copy
    assert copy.read_text() == "# the copy an earlier step installed\n", (
        "an installed copy is kept, not copied again"
    )


def test_arm_deadman_starts_a_cleanup_timer_and_refuses_an_unarmed_one(
    host: NodeHost,
) -> None:
    name = nodep.arm_deadman(RUN, "pause-host", 180, "--unit", HOST_UNIT)
    assert name == nodep.unit_name(RUN, "deadman-pause-host")
    started = host.calls("systemd-run")[-1]
    assert started[:3] == ["systemd-run", f"--unit={name}", "--on-active=180s"]
    assert started[started.index(sys.executable) :] == [
        sys.executable,
        str(nodep.state_dir(RUN) / "probe.py"),
        "cleanup",
        "--run-id",
        RUN,
        "--kind",
        "pause-host",
        "--unit",
        HOST_UNIT,
    ]
    assert host.calls("systemctl", "stop") == [["systemctl", "stop", f"{name}.timer"]]
    host.timers_fail = True
    with pytest.raises(nodep.ProbeError, match=f"deadman {name} is not armed"):
        nodep.arm_deadman(RUN, "pause-host", 180)
    with pytest.raises(nodep.ProbeError, match="deadman seconds is outside 60..5400"):
        nodep.arm_deadman(RUN, "cpu", 59)
    assert len(host.calls("systemd-run")) == 2, "a refused bound starts nothing"


def test_disarm_deadman_never_stops_the_cleanup_it_is_running(host: NodeHost) -> None:
    name = nodep.unit_name(RUN, "deadman-memory")
    host.unit(f"{name}.service", MainPID="321")
    nodep.disarm_deadman(RUN, "memory")
    assert host.calls("systemctl", "stop") == [
        ["systemctl", "stop", f"{name}.timer"],
        ["systemctl", "stop", f"{name}.service"],
    ]
    assert host.commands[-1] == [
        "systemctl",
        "reset-failed",
        f"{name}.service",
        f"{name}.timer",
    ]
    host.commands.clear()
    host.unit(f"{name}.service", MainPID=str(os.getpid()))
    nodep.disarm_deadman(RUN, "memory")
    assert host.calls("systemctl", "stop") == [["systemctl", "stop", f"{name}.timer"]]


# --------------------------------------------------------------------------- #
# oracle sampler commands
# --------------------------------------------------------------------------- #
ORACLE = ["oracle-start", "--run-id", RUN, "--name", "disk", "--seconds", "600"]
SOURCES = ["--diskstats", "nvme1n1", "--netdev", "veth-gl0", "--statvfs", "/data"]


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--diskstats", "sda/../x", "unsafe block device name"),
        ("--netdev", "eth0;reboot", "unsafe interface name"),
        ("--statvfs", "relative/path", "statvfs path must be absolute"),
        ("--statvfs", "/data/../etc", "statvfs path must be absolute"),
    ],
)
def test_oracle_start_refuses_unsafe_sources_before_touching_the_node(
    host: NodeHost, flag: str, value: str, message: str
) -> None:
    with pytest.raises(nodep.ProbeError, match=message):
        command(*ORACLE, *SOURCES, flag, value)
    assert host.commands == [], "nothing is started"
    assert not nodep.state_dir(RUN).exists(), "nothing is written"


def test_oracle_start_runs_the_sampler_from_the_deadman_copy(host: NodeHost) -> None:
    command(*ORACLE, "--interval", "5", "--cpu", "--meminfo", *SOURCES)
    receipt = host.emitted[-1]
    unit = nodep.unit_name(RUN, "oracle-disk")
    path = nodep.state_dir(RUN) / "oracle-disk.jsonl"
    assert (receipt["unit"], receipt["path"]) == (unit, str(path))
    assert path.read_text() == "", "a fresh, empty oracle file"
    assert receipt["spec"] == {
        "cpu": True,
        "loadavg": False,
        "meminfo": True,
        "statvfs": ["/data"],
        "diskstats": ["nvme1n1"],
        "netdev": ["veth-gl0"],
        "snmp": False,
    }
    assert receipt["state"]["ActiveState"] == "active"
    started = host.calls("systemd-run", f"--unit={unit}")[0]
    assert "--property=RuntimeMaxSec=630" in started, "the sampler outlives its run"
    assert "--property=Nice=5" in started
    copy = str(nodep.state_dir(RUN) / "probe.py")
    loop = started[started.index(copy) :]
    assert loop[:8] == [
        copy,
        "sample-loop",
        "--output",
        str(path),
        "--interval",
        "5",
        "--seconds",
        "600",
    ]
    assert json.loads(loop[-1]) == receipt["spec"]
    assert nodep.load_state(RUN)["units"] == [unit]
    with pytest.raises(nodep.ProbeError, match="sampler interval is outside 1..60"):
        command(*ORACLE, "--interval", "61")
    assert len(host.calls("systemd-run")) == 1, "a refused interval starts nothing"


def test_oracle_read_pages_records_and_marks_malformed_lines(host: NodeHost) -> None:
    path = nodep.oracle_path(RUN, "disk")
    path.parent.mkdir(parents=True)
    path.write_text('{"t": 1}\n{"t": 2}\nnot json\n')
    command("oracle-read", "--run-id", RUN, "--name", "disk", "--offset", "1")
    assert host.emitted[-1] == {
        "offset": 1,
        "next_offset": 3,
        "total": 3,
        "records": [{"t": 2}, {"malformed": True}],
    }
    command("oracle-read", "--run-id", RUN, "--name", "net", "--offset", "-5")
    assert host.emitted[-1] == {
        "offset": 0,
        "next_offset": 0,
        "total": 0,
        "records": [],
    }, "a sampler that never wrote reads as empty"


def test_oracle_stop_stops_the_sampler_unit(host: NodeHost) -> None:
    unit = nodep.unit_name(RUN, "oracle-disk")
    host.unit(f"{unit}.service", MainPID="77")
    command("oracle-stop", "--run-id", RUN, "--name", "disk")
    assert host.emitted[-1] == {"stopped": GONE}
    index = host.commands.index(["systemctl", "stop", f"{unit}.service"])
    assert host.options[index]["timeout"] == 120
    with pytest.raises(nodep.ProbeError, match="unsafe resource name"):
        command("oracle-stop", "--run-id", RUN, "--name", "Disk1")


# --------------------------------------------------------------------------- #
# CPU, page cache and memory loads
# --------------------------------------------------------------------------- #
def test_cpu_start_runs_idle_class_burners_under_a_deadman(host: NodeHost) -> None:
    command("cpu-start", "--run-id", RUN, "--threads", "3", "--seconds", "600")
    receipt = host.emitted[-1]
    unit = nodep.unit_name(RUN, "cpu")
    assert (receipt["unit"], receipt["threads"], receipt["seconds"]) == (unit, 3, 600)
    assert receipt["deadman"] == nodep.unit_name(RUN, "deadman-cpu")
    assert receipt["state"]["ActiveState"] == "active"
    started = host.calls("systemd-run", f"--unit={unit}")[0]
    for item in ("RuntimeMaxSec=600", "CPUSchedulingPolicy=idle", "Nice=19"):
        assert f"--property={item}" in started, item
    assert started[-3:-1] == ["/bin/sh", "-c"] and "-lt 3 ]" in started[-1]
    deadman = host.calls("systemd-run", f"--unit={receipt['deadman']}")[0]
    assert "--on-active=720s" in deadman, "the deadman fires after the unit's cap"
    assert nodep.load_state(RUN)["units"] == [unit]
    limit = max(1, os.cpu_count() or 1) * nodep.MAX_THREADS_PER_CPU
    with pytest.raises(nodep.ProbeError, match=f"threads is outside 1..{limit}"):
        command(
            "cpu-start", "--run-id", RUN, "--threads", str(limit + 1), "--seconds", "60"
        )
    command("cpu-stop", "--run-id", RUN)
    assert host.emitted[-1] == {"stopped": GONE}
    assert deadman_stopped(host, "cpu"), host.commands


def test_backing_root_guards_the_backing_file_system(
    host: NodeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    var_root, nvme_root = nodep.BACKING_ROOTS
    with pytest.raises(nodep.ProbeError, match="parent does not exist on this node"):
        nodep.backing_root(str(nvme_root), 0)
    var_root.parent.mkdir(parents=True)
    statvfs_answers(monkeypatch, {str(var_root): (1000, 10)})
    with pytest.raises(nodep.ProbeError, match="no room for the test file"):
        nodep.backing_root(str(var_root), 10 * 4096)
    statvfs_answers(monkeypatch, {str(var_root): (0, 0)})
    with pytest.raises(nodep.ProbeError, match="no room for the test file"):
        nodep.backing_root(str(var_root), 0)
    statvfs_answers(monkeypatch, {str(var_root): (1000, 300)})
    with pytest.raises(nodep.ProbeError, match="would exceed its usage guard"):
        nodep.backing_root(str(var_root), 200 * 4096)
    assert nodep.backing_root(str(var_root), 50 * 4096) == var_root, "75 % after"
    assert var_root.is_dir(), "the allow-listed root is created"


def test_page_cache_start_reads_a_sparse_file_and_stop_removes_it(
    host: NodeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    host.write(
        "/proc/meminfo", f"MemTotal: {64 * 1024} kB\nMemAvailable: {48 * 1024} kB\n"
    )
    root = nodep.BACKING_ROOTS[0]
    root.parent.mkdir(parents=True)
    statvfs_answers(monkeypatch, {str(root): (1 << 20, 1 << 19)})
    argv = [
        "page-cache-start",
        "--run-id",
        RUN,
        "--seconds",
        "120",
        "--backing",
        str(root),
    ]
    with pytest.raises(nodep.ProbeError, match="fraction is outside 0.1..0.95"):
        command(*argv, "--fraction", "0.96")
    command(*argv, "--fraction", "0.5")
    receipt = host.emitted[-1]
    sparse = root / f"gfrs-live-{nodep.run_digest(RUN)}" / "page-cache.sparse"
    assert receipt["file"] == str(sparse)
    assert receipt["apparent_bytes"] == sparse.stat().st_size == 32 * 1024 * 1024
    assert receipt["allocated_bytes"] < receipt["apparent_bytes"], "a sparse file"
    assert receipt["deadman"] == nodep.unit_name(RUN, "deadman-pagecache")
    unit = host.calls("systemd-run", f"--unit={receipt['unit']}")[0]
    assert "--property=IOSchedulingClass=idle" in unit
    assert shlex.quote(str(sparse)) in unit[-1]
    command(*argv, "--fraction", "0.5")
    assert nodep.load_state(RUN)["page_cache_files"] == [str(sparse)], "recorded once"
    command("page-cache-stop", "--run-id", RUN)
    stopped = host.emitted[-1]
    assert stopped["removed"] == [str(sparse)] and not sparse.exists()
    assert stopped["memory"]["memory_used_percent"] == 25.0
    assert nodep.load_state(RUN)["page_cache_files"] == []
    assert deadman_stopped(host, "pagecache"), host.commands


def test_remove_page_cache_files_forgets_a_file_that_is_already_gone(
    host: NodeHost, tmp_path: Path
) -> None:
    kept = tmp_path / "kept.sparse"
    kept.write_bytes(b"")
    state = nodep.load_state(RUN)
    state["page_cache_files"] = [str(tmp_path / "gone.sparse"), str(kept)]
    nodep.save_state(RUN, state)
    assert nodep.remove_page_cache_files(RUN) == [str(kept)]
    assert not kept.exists(), "the file that was there is removed"
    assert nodep.load_state(RUN)["page_cache_files"] == []


def test_memory_stop_stops_the_allocator_and_disarms_its_deadman(
    host: NodeHost,
) -> None:
    host.write("/proc/meminfo", "MemTotal: 1000 kB\nMemAvailable: 900 kB\n")
    host.unit(f"{nodep.unit_name(RUN, 'memory')}.service", MainPID="55")
    host.unit(f"{nodep.unit_name(RUN, 'deadman-memory')}.timer")
    command("memory-stop", "--run-id", RUN)
    assert host.emitted[-1]["stopped"] == GONE
    assert round(host.emitted[-1]["memory"]["memory_used_percent"]) == 10
    assert deadman_stopped(host, "memory"), host.commands
    assert not any("deadman-memory" in name for name in host.units), host.units


# --------------------------------------------------------------------------- #
# pause bookkeeping the deadman needs this file for
# --------------------------------------------------------------------------- #
def test_stop_checks_read_the_process_state_and_its_start_ticks(host: NodeHost) -> None:
    host.process(4321, ["/usr/bin/python3"], state="T")
    host.unit(HOST_UNIT, MainPID="4321")
    assert nodep.unit_is_stopped(HOST_UNIT) is True
    for pid in ("0", "n/a"):
        host.unit(HOST_UNIT, MainPID=pid)
        assert nodep.unit_is_stopped(HOST_UNIT) is False, pid
    assert nodep.process_is_stopped(4321, "43210") is True
    assert nodep.process_is_stopped(4321, "999") is False, "a new process, old PID"
    host.write("/proc/77/stat", "77 (x) T 1\n")
    assert nodep.process_is_stopped(77, None) is True
    assert nodep.process_is_stopped(77, "770") is False, "no start ticks to compare"
    assert nodep.process_is_stopped(5555, None) is False, "a process that is gone"


def test_an_installed_pause_module_that_cannot_be_loaded_is_refused(
    host: NodeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, nodep.PROBE_ALIAS, nodep)
    installed = nodep.state_dir(RUN) / nodep.PAUSE_NAME
    installed.parent.mkdir(parents=True)
    installed.write_text("raise SystemExit('never executed')\n")
    with monkeypatch.context() as patched:
        patched.setattr(
            nodep.importlib.util, "spec_from_file_location", lambda *a, **k: None
        )
        with pytest.raises(nodep.ProbeError, match="pause module cannot be loaded"):
            nodep.load_pause_module(RUN)


CONTENT = b"VALUE = 1\n"
DIGEST = hashlib.sha256(CONTENT).hexdigest()
ENCODED = base64.b64encode(CONTENT).decode("ascii")
TOO_BIG = b"#" * (nodep.MAX_SOURCE_BYTES + 1)


@pytest.mark.parametrize(
    ("name", "sha256", "b64", "message"),
    [
        ("Pause.py", DIGEST, ENCODED, "unsafe source name"),
        ("pause.sh", DIGEST, ENCODED, "unsafe source name"),
        ("pause.py", "ABC", ENCODED, "a sha256 is required"),
        (
            "pause.py",
            hashlib.sha256(TOO_BIG).hexdigest(),
            base64.b64encode(TOO_BIG).decode("ascii"),
            "exceeds its size bound",
        ),
        ("pause.py", hashlib.sha256(b"x").hexdigest(), ENCODED, "does not match"),
    ],
    ids=["upper-case-name", "not-python", "no-sha256", "too-big", "sha-mismatch"],
)
def test_install_source_refuses_what_it_cannot_prove(
    host: NodeHost, name: str, sha256: str, b64: str, message: str
) -> None:
    argv = ["install-source", "--run-id", RUN, "--name", name]
    with pytest.raises(nodep.ProbeError, match=message):
        command(*argv, "--sha256", sha256, "--b64", b64)
    assert not nodep.state_dir(RUN).exists(), "a refused source writes nothing"


def test_resume_unit_reports_a_reload_that_failed_after_resuming(
    host: NodeHost,
) -> None:
    dropin = nodep.pause_dropin_path(RUN, KERNEL_UNIT)
    dropin.parent.mkdir(parents=True)
    dropin.write_text("[Service]\nExecStartPost=/usr/bin/kill -STOP $MAINPID\n")
    state = nodep.load_state(RUN)
    state["paused_units"][KERNEL_UNIT] = {
        "kind": "pause-kernel",
        "dropin": {"created_dir": True},
    }
    nodep.save_state(RUN, state)
    host.failures[("systemctl", "daemon-reload")] = 1
    with pytest.raises(nodep.ProbeError, match="did not reload after removing"):
        nodep.resume_unit(RUN, KERNEL_UNIT)
    assert not dropin.parent.exists(), "the drop-in and the directory it made"
    cont = ["systemctl", "kill", "--kill-whom=all", "--signal=SIGCONT", KERNEL_UNIT]
    assert cont in host.commands, "the unit is resumed before the error"
    assert nodep.load_state(RUN)["paused_units"] == {}
    assert deadman_stopped(host, "pause-kernel"), host.commands


def test_resume_processes_resumes_only_the_process_it_stopped(
    host: NodeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = resumable(host, monkeypatch)
    host.process(812, EXPORTER, state="T")
    host.process(813, EXPORTER, state="T")
    state = nodep.load_state(RUN)
    state["paused_processes"] = {
        "812": {"start_ticks": "8120"},
        "813": {"start_ticks": "1"},
        "9999": {"start_ticks": "5"},
    }
    nodep.save_state(RUN, state)
    assert nodep.resume_processes(RUN) == [
        {"pid": 812, "resumed": True, "stopped": False},
        {"pid": 813, "resumed": False, "gone": True},
        {"pid": 9999, "resumed": False, "gone": True},
    ], "813 is a new process under the PID; 9999 is gone"
    assert sent == [(812, signal.SIGCONT)]
    assert nodep.load_state(RUN)["paused_processes"] == {}
    assert deadman_stopped(host, "resume-exporter"), host.commands


def test_restore_log_collector_undoes_only_its_own_enable(host: NodeHost) -> None:
    log_unit = nodep.LOG_COLLECTOR_UNIT
    host.unit(log_unit, MainPID="61")
    state = nodep.load_state(RUN)
    state["log_collector"] = {"enabled_at": "2026-11-05T00:00:00+00:00"}
    state["paused_units"][log_unit] = {"kind": "pause-log"}
    nodep.save_state(RUN, state)
    restored = nodep.restore_log_collector(RUN)
    assert restored["restored"] is True
    assert restored["after"]["MainPID"] == "61", "systemctl disable is recorded only"
    disable = ["systemctl", "disable", "--runtime", "--now", log_unit]
    assert host.options[host.commands.index(disable)]["timeout"] == 120
    assert ["systemctl", "kill", "--kill-whom=all", "--signal=SIGCONT", log_unit] in (
        host.commands
    )
    after = nodep.load_state(RUN)
    assert (after["log_collector"], after["paused_units"]) == (None, {})
    assert deadman_stopped(host, "logcollector"), host.commands
    assert nodep.restore_log_collector(RUN) == {
        "restored": False,
        "reason": "not enabled by this run",
    }


# --------------------------------------------------------------------------- #
# cleanup and residue (the deadman's entry points)
# --------------------------------------------------------------------------- #
def test_cleanup_all_undoes_everything_the_run_recorded(
    host: NodeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = resumable(host, monkeypatch)
    digest = nodep.run_digest(RUN)
    cpu, oracle = nodep.unit_name(RUN, "cpu"), nodep.unit_name(RUN, "oracle-disk")
    host.unit(f"{cpu}.service", MainPID="11")
    host.unit(f"{oracle}.service", MainPID="12")
    for kind in ("cpu", "pagecache"):
        host.unit(f"{nodep.unit_name(RUN, f'deadman-{kind}')}.timer")
    host.unit(f"{nodep.unit_name(RUN, 'deadman-memory')}.service", MainPID="13")
    host.unit(f"{nodep.unit_name(RUN, 'deadman-cpu')}.scope")
    backing = [root / f"gfrs-live-{digest}" for root in nodep.BACKING_ROOTS]
    for directory in backing:
        directory.mkdir(parents=True)
    sparse = backing[0] / "page-cache.sparse"
    sparse.write_bytes(b"")
    host.process(812, EXPORTER, state="T")
    for unit in (HOST_UNIT, KERNEL_UNIT):
        nodep.pause_dropin_path(RUN, unit).parent.mkdir(parents=True)
        nodep.pause_dropin_path(RUN, unit).write_text("[Service]\n")
    nodep.save_state(
        RUN,
        {
            "run_id": RUN,
            "units": [cpu, oracle],
            "paused_units": {
                HOST_UNIT: {"kind": "pause-host", "dropin": {"created_dir": True}},
                "sshd.service": {"kind": "pause-sshd"},
            },
            "paused_processes": {"812": {"start_ticks": "8120"}},
            "page_cache_files": [str(sparse)],
            "log_collector": {"enabled_at": "2026-11-05T00:00:00+00:00"},
        },
    )
    command("cleanup", "--run-id", RUN)
    report = host.emitted[-1]
    assert report["kind"] == "all" and report["clean"] is True, report["residue"]
    kinds = [next(iter(action)) for action in report["actions"]]
    assert kinds[:3] == ["cpu", "memory", "pagecache"]
    assert kinds[-2:] == ["exporter", "log_collector"]
    resumed = {a["resumed"]["unit"] for a in report["actions"] if "resumed" in a}
    assert resumed == {HOST_UNIT, KERNEL_UNIT}, "the state's and the lost drop-in's"
    touched = {tuple(item[:2]) for item in host.commands if "sshd.service" in item}
    assert touched == {("systemctl", "show")}, "outside the allow-list: read only"
    assert ["systemctl", "stop", f"{oracle}.service"] in host.commands
    assert report["actions"][2]["removed"] == [str(sparse)]
    assert sent == [(812, signal.SIGCONT)]
    for kind in ("cpu", "memory", "pagecache", "pause-host", "logcollector"):
        assert deadman_stopped(host, kind), kind
    assert not any(directory.exists() for directory in backing), "empty dirs go"
    assert report["residue"]["units"] == [
        f"{nodep.unit_name(RUN, 'deadman-cpu')}.scope"
    ]


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--kind", "cpu"], ["cpu"]),
        (["--kind", "memory"], ["memory"]),
        (["--kind", "pagecache"], ["pagecache"]),
        (["--kind", "resume-exporter"], ["exporter"]),
        (["--kind", "logcollector"], ["log_collector"]),
        (["--kind", "pause-host", "--unit", HOST_UNIT], ["resumed"]),
        (["--kind", "oracle-disk"], []),
    ],
)
def test_cleanup_of_one_kind_undoes_only_that_kind(
    host: NodeHost, argv: list[str], expected: list[str]
) -> None:
    command("cleanup", "--run-id", RUN, *argv)
    report = host.emitted[-1]
    assert [next(iter(action)) for action in report["actions"]] == expected
    assert deadman_stopped(host, argv[1]), host.commands
    assert report["clean"] is True, report["residue"]


def test_cleanup_refuses_an_unsafe_kind_and_a_unit_outside_the_allow_list(
    host: NodeHost,
) -> None:
    with pytest.raises(nodep.ProbeError, match="unsafe cleanup kind"):
        command("cleanup", "--run-id", RUN, "--kind", "cpu; reboot")
    agent = "gpu-fault-node-agent.service"
    with pytest.raises(nodep.ProbeError, match="pause allow-list"):
        command("cleanup", "--run-id", RUN, "--kind", "pause-agent", "--unit", agent)
    assert host.commands == [], "a refused cleanup runs nothing"


def test_residue_command_reports_what_the_run_still_holds(
    host: NodeHost, tmp_path: Path
) -> None:
    held = tmp_path / "page-cache.sparse"
    held.write_bytes(b"")
    state = nodep.load_state(RUN)
    state["page_cache_files"] = [str(held), str(tmp_path / "gone.sparse")]
    nodep.save_state(RUN, state)
    cpu = f"{nodep.unit_name(RUN, 'cpu')}.service"
    host.unit(cpu, MainPID="9")
    host.hooks = {HOST_UNIT}
    command("residue", "--run-id", RUN)
    report = host.emitted[-1]
    assert report["clean"] is False
    assert report["residue"]["page_cache_files"] == [str(held)]
    assert report["residue"]["units"] == [cpu]
    assert report["residue"]["pause_hooks"] == [HOST_UNIT]


def test_the_probe_file_runs_as_the_deadmans_script(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["general_live_node_probe.py", "residue", "--run-id", "Bad Run"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(nodep.__file__, run_name="__main__")
    assert caught.value.code == 1
    out, err = capsys.readouterr()
    failure = json.loads(out.splitlines()[-1])
    assert failure["error_class"] == "ProbeError" and failure["command"] == "residue"
    assert failure["error_message"] == "unsafe run ID"
    assert failure["error_site"].startswith("show_residue:"), failure
    assert err == "ProbeError: unsafe run ID\n"
