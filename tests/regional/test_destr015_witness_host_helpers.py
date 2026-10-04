"""GF-REGIONAL-DESTR-015 detached witness: the host-facing helpers the fake
host normally replaces. Each one is driven with a stubbed ``subprocess.run``,
a real SQLite ledger or a real directory, so their refusals and read-only
fallbacks are pinned rather than assumed."""

from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.probes import destr015_witness_probe as probe
from tests.regional.test_destr015_witness_probe import owned_scope


class FakeSubprocess:
    """Answers ``subprocess.run`` with a scripted result and records the call."""

    def __init__(self, returncode: int = 0, stdout: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append({"command": list(command), **kwargs})
        return subprocess.CompletedProcess(command, self.returncode, self.stdout, "")


def test_run_refuses_a_failed_command_only_when_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeSubprocess(returncode=3, stdout="out")
    monkeypatch.setattr(probe.subprocess, "run", fake)
    with pytest.raises(probe.ProbeError, match=r"host command failed \(3\): false"):
        probe.run(["false", "--secret-argument"])
    unchecked = probe.run(["false", "--secret-argument"], check=False, timeout=5)
    assert unchecked.returncode == 3
    assert unchecked.stdout == "out"
    assert [call["timeout"] for call in fake.calls] == [60, 5]
    assert fake.calls[0]["env"] == {
        "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
        "LC_ALL": "C",
    }
    assert fake.calls[0]["check"] is False


def test_run_returns_a_successful_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probe.subprocess, "run", FakeSubprocess(stdout="ok\n"))
    assert probe.run(["true"]).stdout == "ok\n"


@pytest.mark.parametrize("value", ["../escape", "", 42, None, "a" * 129, "bad id"])
def test_safe_run_id_refuses_unsafe_values(value: Any) -> None:
    with pytest.raises(probe.ProbeError, match="unsafe run ID"):
        probe.safe_run_id(value)
    with pytest.raises(probe.ProbeError, match="unsafe run ID"):
        probe.run_directory(value)


def test_read_json_refuses_a_record_that_is_not_an_object(tmp_path: Path) -> None:
    record = tmp_path / "armed.json"
    record.write_text(json.dumps(["not", "a", "record"]), encoding="utf-8")
    with pytest.raises(probe.ProbeError, match="witness record is malformed"):
        probe.read_json(record)
    assert probe.read_json(tmp_path / "absent.json") is None


def test_unit_state_skips_lines_without_a_separator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeSubprocess(
        stdout="LoadState=loaded\nnot a property line\nActiveState=inactive\n\n"
    )
    monkeypatch.setattr(probe.subprocess, "run", fake)
    state = probe.unit_state("gpu-fault-destr015-witness-abc.service")
    assert state == {"LoadState": "loaded", "ActiveState": "inactive"}
    assert probe.unit_active(state) is False
    assert fake.calls[0]["command"][:3] == [
        "systemctl",
        "show",
        "gpu-fault-destr015-witness-abc.service",
    ]
    assert fake.calls[0]["timeout"] == 20


@pytest.mark.parametrize("pid_text", ["", "not-a-pid", "0", "-4"])
def test_agent_tracee_refuses_without_a_live_agent_pid(
    monkeypatch: pytest.MonkeyPatch, pid_text: str
) -> None:
    monkeypatch.setattr(probe.subprocess, "run", FakeSubprocess(stdout=pid_text))
    monkeypatch.setattr(
        probe,
        "process_identity",
        lambda pid: pytest.fail("no identity may be read without a pid"),
    )
    with pytest.raises(probe.ProbeError, match="no Node Agent process identity"):
        probe.agent_tracee()


def test_agent_tracee_reads_the_main_pid_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeSubprocess(stdout="4242\n")
    monkeypatch.setattr(probe.subprocess, "run", fake)
    monkeypatch.setattr(probe, "process_identity", lambda pid: {"pid": pid})
    assert probe.agent_tracee() == {"pid": 4242}
    assert fake.calls[0]["command"] == [
        "systemctl",
        "show",
        "--property=MainPID",
        "--value",
        probe.AGENT_UNIT,
    ]


def test_resolve_executable_requires_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(probe.shutil, "which", lambda name: None)
    with pytest.raises(probe.ProbeError, match="executable is unavailable"):
        probe.resolve_executable()
    real = tmp_path / "bin" / "nvidia-smi"
    real.parent.mkdir()
    real.write_bytes(b"fixture")
    link = tmp_path / "nvidia-smi"
    link.symlink_to(real)
    monkeypatch.setattr(probe.shutil, "which", lambda name: str(link))
    assert probe.resolve_executable() == real.resolve()


def test_gpu_present_reads_the_uuid_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        probe.subprocess, "run", FakeSubprocess(stdout="GPU-a \n GPU-b\n")
    )
    assert probe.gpu_present("GPU-a") is True
    assert probe.gpu_present("GPU-c") is False
    monkeypatch.setattr(
        probe.subprocess, "run", FakeSubprocess(returncode=9, stdout="GPU-a\n")
    )
    assert probe.gpu_present("GPU-a") is False, "a failed query proves nothing"


LEDGER_SCHEMA = (
    "CREATE TABLE results (command_id TEXT, attempt INTEGER, state TEXT, "
    "operation TEXT, started_at TEXT, completed_at TEXT, workflow_request_id TEXT, "
    "incident_id TEXT, fencing_token INTEGER, agent_generation INTEGER, "
    "gpu_uuids TEXT)"
)


def _ledger(path: Path, rows: list[tuple[Any, ...]]) -> Path:
    connection = sqlite3.connect(path)
    try:
        connection.execute(LEDGER_SCHEMA)
        connection.executemany(
            "INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows
        )
        connection.commit()
    finally:
        connection.close()
    return path


def test_reset_ledger_rows_reads_only_reset_rows(tmp_path: Path) -> None:
    ledger = _ledger(
        tmp_path / "node-actions.db",
        [
            (
                "cmd-2",
                1,
                "SUCCEEDED",
                "RESET_GPU",
                "2026-10-04T10:00:00Z",
                "2026-10-04T10:00:09Z",
                "workflow-2",
                "inc-2",
                7,
                3,
                json.dumps(["GPU-a"]),
            ),
            (
                "cmd-1",
                1,
                "FAILED",
                "RESET_GPU",
                "2026-10-04T09:00:00Z",
                "2026-10-04T09:00:05Z",
                "workflow-1",
                "inc-1",
                6,
                3,
                None,
            ),
            (
                "cmd-3",
                1,
                "SUCCEEDED",
                "RESTART_NODE",
                "2026-10-04T08:00:00Z",
                "2026-10-04T08:00:05Z",
                "workflow-0",
                "inc-0",
                5,
                2,
                json.dumps(["GPU-z"]),
            ),
        ],
    )
    rows = probe.reset_ledger_rows(ledger)
    assert [row["command_id"] for row in rows] == ["cmd-1", "cmd-2"]
    assert rows[0]["gpu_uuids"] == [], "a NULL gpu list reads as no GPUs"
    assert rows[1] == {
        "command_id": "cmd-2",
        "attempt": 1,
        "state": "SUCCEEDED",
        "operation": "RESET_GPU",
        "started_at": "2026-10-04T10:00:00Z",
        "completed_at": "2026-10-04T10:00:09Z",
        "workflow_request_id": "workflow-2",
        "incident_id": "inc-2",
        "fencing_token": 7,
        "agent_generation": 3,
        "gpu_uuids": ["GPU-a"],
    }


def test_reset_ledger_rows_treats_an_unreadable_ledger_as_none_seen(
    tmp_path: Path,
) -> None:
    assert probe.reset_ledger_rows(tmp_path / "absent.db") == []
    garbage = tmp_path / "garbage.db"
    garbage.write_bytes(b"this is not a sqlite database at all" * 4)
    assert probe.reset_ledger_rows(garbage) == []
    without_table = tmp_path / "empty.db"
    sqlite3.connect(without_table).close()
    assert probe.reset_ledger_rows(without_table) == []


def test_install_program_refuses_a_foreign_file_set(tmp_path: Path) -> None:
    sources = {name: "" for name in probe.PROGRAM_FILES[:-1]}
    with pytest.raises(probe.ProbeError, match="pinned file set"):
        probe.install_program(tmp_path / "program", sources)
    assert not (tmp_path / "program").exists(), "nothing is written before the check"


def test_install_program_replaces_a_stale_copy(tmp_path: Path) -> None:
    directory = tmp_path / "program"
    directory.mkdir()
    stale = directory / "stale.txt"
    stale.write_text("left over from an earlier arm", encoding="utf-8")
    sources = {name: f"# {name}\n" for name in probe.PROGRAM_FILES}
    digest = probe.install_program(directory, sources)
    assert digest == probe.program_digest(sources)
    assert not stale.exists(), "the stale copy is removed, not merged"
    for name in probe.PROGRAM_FILES:
        assert (directory / name).read_text(encoding="utf-8") == f"# {name}\n"
    for package in probe.PACKAGES:
        assert (directory / package / "__init__.py").is_file(), package


def test_attach_refuses_a_boot_or_agent_that_differs_from_the_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scope = owned_scope()
    monkeypatch.setattr(probe, "boot_id", lambda: "boot-other")
    monkeypatch.setattr(
        probe,
        "resolve_executable",
        lambda: pytest.fail("the executable is not resolved before the boot check"),
    )
    with pytest.raises(probe.ProbeError, match="host boot id differs"):
        probe.attach(tmp_path, scope, deadline=1.0)

    monkeypatch.setattr(probe, "boot_id", lambda: scope.boot_id)
    monkeypatch.setattr(probe, "resolve_executable", lambda: tmp_path / "nvidia-smi")
    monkeypatch.setattr(
        probe, "agent_tracee", lambda: SimpleNamespace(pid=1, boot_id="boot-other")
    )
    with pytest.raises(probe.ProbeError, match="Node Agent boot id differs"):
        probe.attach(tmp_path, scope, deadline=1.0)
    assert not (tmp_path / "trace").exists(), "no trace directory before the checks"


def test_reset_records_drops_a_stale_trace_directory(tmp_path: Path) -> None:
    trace = tmp_path / "trace"
    trace.mkdir()
    (trace / "events.bin").write_bytes(b"old")
    (tmp_path / "armed.json").write_text("{}", encoding="utf-8")
    probe.reset_records(tmp_path)
    assert not trace.exists(), "an earlier arm's trace is not reused"
    assert not (tmp_path / "armed.json").exists(), "the armed record is dropped"


@pytest.mark.parametrize("value", [None, ["scope"], "scope", 7])
def test_parse_scope_requires_an_object(value: Any) -> None:
    with pytest.raises(probe.ProbeError, match="carries no scope"):
        probe.parse_scope(value)


class RecordingTracer:
    """Stands in for the strace witness; only its lifecycle is observed."""

    def __init__(self, directory: Path, tracee: Any, **kwargs: Any) -> None:
        self.directory = directory
        self.kwargs = kwargs
        self.started = False

    def start(self) -> None:
        self.started = True


def test_attach_replaces_a_stale_trace_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scope = owned_scope()
    executable = tmp_path / "nvidia-smi"
    executable.write_bytes(b"fixture")
    stale = tmp_path / "trace" / "stale.bin"
    stale.parent.mkdir()
    stale.write_bytes(b"old")
    tracee = SimpleNamespace(
        pid=4242, boot_id=scope.boot_id, model_dump=lambda **_kw: {"pid": 4242}
    )
    monkeypatch.setattr(probe, "boot_id", lambda: scope.boot_id)
    monkeypatch.setattr(probe, "resolve_executable", lambda: executable)
    monkeypatch.setattr(probe, "agent_tracee", lambda: tracee)
    monkeypatch.setattr(probe, "AttachedExecWitness", RecordingTracer)
    monkeypatch.setattr(
        probe,
        "process_identity",
        lambda pid: SimpleNamespace(model_dump=lambda **_kw: {"pid": pid}),
    )
    witness, start = probe.attach(tmp_path, scope, deadline=100.0)
    assert isinstance(witness, RecordingTracer), "the tracer is the attached one"
    assert witness.started is True, "the tracer is started once attached"
    assert not stale.exists(), "the stale trace is removed before attaching"
    assert (tmp_path / "trace").is_dir(), "a fresh trace directory is created"
    assert witness.kwargs["deadline"] == 100.0 + probe.UNIT_SLACK_SECONDS // 2
    assert start["scope_sha256"] == scope.digest()
    assert start["executable_path"] == str(executable)
    assert start["tracee"] == {"pid": 4242}
