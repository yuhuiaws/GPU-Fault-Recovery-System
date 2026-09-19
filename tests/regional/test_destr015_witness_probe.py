"""The detached DESTR-015 witness probe: request handlers and the unit body.

Every host interaction is faked at the boundary the probe itself draws
(``run``, ``boot_id``, ``agent_tracee``, ``resolve_executable``,
``gpu_present``, ``AttachedExecWitness``); the state directory is a temporary
tree. Nothing here starts a process or touches a device.
"""

from __future__ import annotations

import io
import json
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import late_ownership_probe_bundle as bundle
from scripts.e2e.regional.destr015_physical_evidence import (
    ResetIntervalScope,
    evidence_digest,
)
from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied
from scripts.e2e.regional.probes import destr015_witness_probe as probe
from scripts.e2e.regional.probes.destr015_physical_probe import QUERY_ARGS
from tests.regional.test_acceptance_physical_interval_alignment import (
    interval_fixture,
    trace_process,
)

RUN_ID = "destr015-0b111dddd0a2-a1"
BOOT = "boot-node-a"
GPU = "GPU-node-a"
LIFETIME = 3600


def owned_scope(**changes: Any) -> ResetIntervalScope:
    _, scopes, _, _ = interval_fixture()
    return scopes["node-a"].model_copy(
        update={
            "run_id": RUN_ID,
            "maintenance_end": datetime.now(timezone.utc) + timedelta(hours=2),
            **changes,
        }
    )


def request(kind: str, bound: ResetIntervalScope, **payload: Any) -> dict[str, Any]:
    return {"kind": kind, "scope_sha256": bound.digest(), "payload": payload}


def arm_request(scope: ResetIntervalScope, lifetime: int = LIFETIME) -> dict[str, Any]:
    return request(
        "arm",
        scope,
        scope=scope.model_dump(mode="json"),
        lifetime_seconds=lifetime,
        runner_clock={"monotonic_ns": 1, "realtime_ns": 2},
    )


def armed_record(scope: ResetIntervalScope, executable: Path) -> dict[str, Any]:
    start = {
        "scope_sha256": scope.digest(),
        "witness_id": "w" * 48,
        "tracee": {"pid": 4242, "start_ticks": 7, "uid": 0, "boot_id": scope.boot_id},
        "producer": {"pid": 4300, "start_ticks": 9, "uid": 0, "boot_id": scope.boot_id},
        "executable_path": str(executable),
        "executable_sha256": "a" * 64,
    }
    return {
        "case_id": probe.CASE_ID,
        "record": "armed",
        "run_id": scope.run_id,
        "node": scope.node,
        "scope_sha256": scope.digest(),
        "unit": probe.unit_name(scope.run_id) + ".service",
        "invocation_id": "invocation-owned",
        "witness_id": start["witness_id"],
        "boot_id": scope.boot_id,
        "monotonic_ns": time.monotonic_ns(),
        "realtime_ns": time.time_ns(),
        "start": start,
        "ledger_baseline_command_ids": [],
    }


class FakeHost:
    """What the probe reads or drives on the node, answered without a node."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = tmp_path / "state"
        self.commands: list[list[str]] = []
        self.boot = BOOT
        self.agent_boot = BOOT
        self.gpus = {GPU}
        self.unit_active = False
        # What the transient unit does once started: reports armed, refuses, or
        # stays silent.
        self.unit_behaviour = "armed"
        self.executable = tmp_path / "nvidia-smi"
        self.executable.write_bytes(b"owned nvidia-smi fixture")
        monkeypatch.setattr(probe, "STATE_ROOT", self.root)
        monkeypatch.setattr(probe, "boot_id", lambda: self.boot)
        monkeypatch.setattr(probe, "agent_tracee", self.tracee)
        monkeypatch.setattr(probe, "resolve_executable", lambda: self.executable)
        monkeypatch.setattr(probe, "gpu_present", lambda uuid: uuid in self.gpus)
        monkeypatch.setattr(probe, "run", self.run)
        monkeypatch.setattr(probe, "ARM_WAIT_SECONDS", 1)
        monkeypatch.setattr(probe, "FINISH_WAIT_SECONDS", 1)

    def tracee(self) -> SimpleNamespace:
        return SimpleNamespace(
            pid=4242,
            boot_id=self.agent_boot,
            model_dump=lambda **_kw: {
                "pid": 4242,
                "start_ticks": 7,
                "uid": 0,
                "boot_id": self.agent_boot,
            },
        )

    def run(
        self, command: list[str], *, check: bool = True, timeout: int = 60
    ) -> subprocess.CompletedProcess[str]:
        self.commands.append(list(command))
        if command[0] == "systemd-run":
            directory = self.root / RUN_ID
            if self.unit_behaviour == "armed":
                probe.write_private_json(
                    directory / "armed.json",
                    armed_record(owned_scope(), self.executable),
                )
                self.unit_active = True
            elif self.unit_behaviour == "refuse":
                probe.write_private_json(
                    directory / "final.json",
                    {"record": "final", "refusal": "Agent is already traced"},
                )
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[:2] == ["systemctl", "show"]:
            state = "active" if self.unit_active else "inactive"
            return subprocess.CompletedProcess(
                command,
                0,
                f"LoadState=loaded\nActiveState={state}\nSubState=running\n"
                "MainPID=99\nInvocationID=invocation-owned\n",
                "",
            )
        if command[:2] == ["systemctl", "stop"]:
            self.unit_active = False
        return subprocess.CompletedProcess(command, 0, "", "")

    def stops(self) -> list[list[str]]:
        return [item for item in self.commands if item[:2] == ["systemctl", "stop"]]

    def starts(self) -> list[list[str]]:
        return [item for item in self.commands if item[0] == "systemd-run"]


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeHost:
    return FakeHost(tmp_path, monkeypatch)


# --------------------------------------------------------------------------- #
# arm
# --------------------------------------------------------------------------- #
def test_program_files_are_the_bundle_role(host: FakeHost) -> None:
    assert set(probe.PROGRAM_FILES) == set(
        bundle.probe_files("reset-interval-detached")
    ), "the unit's durable copy must be exactly the delivered bundle"
    assert bundle.ENTRIES["reset-interval-detached"] == probe.ENTRY


def test_arm_installs_the_program_starts_a_bounded_unit_and_binds_the_receipt(
    host: FakeHost,
) -> None:
    scope = owned_scope()
    response = probe.handle_request(arm_request(scope))
    assert response["kind"] == "armed", response
    assert response["scope_sha256"] == scope.digest()
    payload = response["payload"]
    unit = probe.unit_name(RUN_ID) + ".service"
    assert payload["run_id"] == RUN_ID and payload["node"] == "node-a"
    assert payload["unit"] == unit and unit.startswith(probe.UNIT_PREFIX)
    assert payload["state_dir"] == str(host.root / RUN_ID)
    assert payload["boot_id"] == BOOT and payload["lifetime_seconds"] == LIFETIME
    assert payload["armed"]["record"] == "armed"
    assert (
        payload["host_clock_start"]["monotonic_ns"]
        <= payload["armed"]["monotonic_ns"]
        <= payload["host_clock"]["monotonic_ns"]
    ), "the armed record must be written between the request and its response"
    assert (
        payload["program_sha256"] == bundle.probe_program("reset-interval-detached")[1]
    ), "the durable program copy must carry the delivered bundle's digest"
    assert payload["unit_state"]["ActiveState"] == "active"
    starts = host.starts()
    assert len(starts) == 1, host.commands
    command = starts[0]
    assert command[1:3] == ["--unit", probe.unit_name(RUN_ID)]
    assert f"--property=RuntimeMaxSec={LIFETIME + probe.UNIT_SLACK_SECONDS}" in command
    assert "--property=KillMode=control-group" in command
    program_root = host.root / RUN_ID / "program"
    assert command[-8:] == [
        "-I",
        "-u",
        str(program_root / probe.ENTRY),
        "witness",
        "--run-id",
        RUN_ID,
        "--state-root",
        str(host.root),
    ]
    for name in probe.PROGRAM_FILES:
        assert (program_root / name).is_file(), name
        assert (program_root / name).stat().st_mode & 0o777 == 0o600, name
    for package in probe.PACKAGES:
        assert (program_root / package / "__init__.py").is_file(), package
    state = json.loads((host.root / RUN_ID / "state.json").read_text())
    assert state["phase"] == "ARMED" and state["scope_sha256"] == scope.digest()
    assert state["unit"] == unit and state["lifetime_seconds"] == LIFETIME
    assert state["boot_id"] == BOOT and state["disarmed_at"] is None
    # A stale unit of the same name is cleared before the new one starts.
    stop = next(
        i for i, item in enumerate(host.commands) if item[:2] == ["systemctl", "stop"]
    )
    start = next(i for i, item in enumerate(host.commands) if item[0] == "systemd-run")
    assert stop < start, host.commands
    assert (host.root / RUN_ID).stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize(
    ("defect", "expected"),
    [
        ("lifetime-low", "lifetime is outside its bounds"),
        ("lifetime-high", "lifetime is outside its bounds"),
        ("window", "does not fit the maintenance window"),
        ("naive-window", "maintenance end is naive"),
        ("digest", "scope digest does not match"),
        ("boot", "host boot id differs"),
        ("agent-boot", "Node Agent boot id differs"),
        ("gpu", "approved GPU is not present"),
        ("armed-already", "already armed"),
    ],
)
def test_arm_refuses_before_starting_a_unit(
    host: FakeHost, defect: str, expected: str
) -> None:
    scope = owned_scope()
    lifetime = LIFETIME
    if defect == "lifetime-low":
        lifetime = probe.MIN_LIFETIME_SECONDS - 1
    elif defect == "lifetime-high":
        lifetime = probe.MAX_LIFETIME_SECONDS + 1
    elif defect == "window":
        scope = owned_scope(
            maintenance_end=datetime.now(timezone.utc) + timedelta(minutes=10)
        )
    elif defect == "naive-window":
        scope = owned_scope(maintenance_end=datetime(2099, 1, 1))
    elif defect == "boot":
        host.boot = "another-boot"
    elif defect == "agent-boot":
        host.agent_boot = "another-boot"
    elif defect == "gpu":
        host.gpus = set()
    elif defect == "armed-already":
        probe.write_private_json(
            host.root / RUN_ID / "state.json",
            {"run_id": RUN_ID, "phase": "ARMED", "disarmed_at": None},
        )
    message = arm_request(scope, lifetime)
    if defect == "digest":
        message["scope_sha256"] = "0" * 64
    response = probe.handle_request(message)
    assert response["kind"] == "refused", response
    assert expected in response["payload"]["reason"], response
    assert host.starts() == [], "a refused arm must not start a unit"
    if defect != "armed-already":
        assert not (host.root / RUN_ID / "program").exists(), (
            "a refused arm must not leave a program copy"
        )


@pytest.mark.parametrize("behaviour", ["refuse", "silent"])
def test_a_unit_that_refuses_or_stays_silent_is_stopped_and_reported(
    host: FakeHost, behaviour: str
) -> None:
    host.unit_behaviour = behaviour
    scope = owned_scope()
    response = probe.handle_request(arm_request(scope))
    assert response["kind"] == "refused", response
    reason = response["payload"]["reason"]
    if behaviour == "refuse":
        assert reason == "Agent is already traced"
    else:
        assert "did not report armed within" in reason
    unit = probe.unit_name(RUN_ID) + ".service"
    starts = [i for i, item in enumerate(host.commands) if item[0] == "systemd-run"]
    stops = [
        i
        for i, item in enumerate(host.commands)
        if item[:3] == ["systemctl", "stop", unit]
    ]
    assert starts and any(index > starts[0] for index in stops), (
        "the unit must be stopped after it failed to arm"
    )
    state = json.loads((host.root / RUN_ID / "state.json").read_text())
    assert state["phase"] == "REFUSED" and state["refusal"] == reason
    # The refusal stays readable for the runner while kubelet still answers.
    status = probe.handle_request(request("status", scope, run_id=RUN_ID))
    assert status["kind"] == "status"
    assert status["payload"]["state"]["refusal"] == reason
    assert status["payload"]["armed"] is None


# --------------------------------------------------------------------------- #
# collect / disarm / status
# --------------------------------------------------------------------------- #
def _armed(host: FakeHost) -> tuple[ResetIntervalScope, dict[str, Any]]:
    scope = owned_scope()
    response = probe.handle_request(arm_request(scope))
    assert response["kind"] == "armed", response
    return scope, response["payload"]


def _unit_answers_finish(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch, final: dict[str, Any]
) -> None:
    """The unit writes its final record once the finish request appears."""

    def sleep(_seconds: float) -> None:
        directory = host.root / RUN_ID
        if (directory / "finish.request").exists():
            probe.write_private_json(directory / "final.json", final)

    monkeypatch.setattr(
        probe,
        "time",
        SimpleNamespace(
            monotonic=time.monotonic,
            monotonic_ns=time.monotonic_ns,
            time_ns=time.time_ns,
            sleep=sleep,
        ),
    )


def test_collect_requests_the_finish_and_returns_every_record(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    scope, receipt = _armed(host)
    final = {"record": "final", "reason": "finish-request", "refusal": None}
    _unit_answers_finish(host, monkeypatch, final)
    response = probe.handle_request(request("collect", scope, run_id=RUN_ID))
    assert response["kind"] == "collected", response
    payload = response["payload"]
    assert payload["run_id"] == RUN_ID and payload["node"] == "node-a"
    assert payload["unit"] == receipt["unit"] and payload["boot_id"] == BOOT
    assert payload["armed"] == receipt["armed"]
    assert payload["final"] == final
    assert payload["finish_requested"] is not None
    assert payload["state"]["phase"] == "ARMED"
    assert (
        payload["host_clock_start"]["monotonic_ns"]
        <= payload["host_clock"]["monotonic_ns"]
    )
    written = json.loads((host.root / RUN_ID / "finish.request").read_text())
    assert written["run_id"] == RUN_ID
    # A final record that already exists is returned without a second request.
    again = probe.handle_request(request("collect", scope, run_id=RUN_ID))
    assert again["payload"]["finish_requested"] is None
    assert again["payload"]["final"] == final


def test_collect_without_a_final_record_reports_the_unit_not_a_proof(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    scope, _ = _armed(host)
    monkeypatch.setattr(
        probe,
        "time",
        SimpleNamespace(
            monotonic=time.monotonic,
            monotonic_ns=time.monotonic_ns,
            time_ns=time.time_ns,
            sleep=lambda _s: None,
        ),
    )
    monkeypatch.setattr(probe, "FINISH_WAIT_SECONDS", 0)
    response = probe.handle_request(request("collect", scope, run_id=RUN_ID))
    assert response["kind"] == "collected"
    assert response["payload"]["final"] is None
    assert response["payload"]["unit_state"]["ActiveState"] == "active"


@pytest.mark.parametrize(
    ("defect", "expected"),
    [
        ("boot", "host boot id changed since the witness was armed"),
        ("no-state", "no armed witness state"),
        ("scope", "belongs to another scope"),
    ],
)
def test_collect_refuses_a_changed_boot_id_and_unbound_state(
    host: FakeHost, defect: str, expected: str
) -> None:
    scope, _ = _armed(host)
    message = request("collect", scope, run_id=RUN_ID)
    if defect == "boot":
        host.boot = "boot-after-reboot"
    elif defect == "no-state":
        (host.root / RUN_ID / "state.json").unlink()
    else:
        message["scope_sha256"] = "1" * 64
    response = probe.handle_request(message)
    assert response["kind"] == "refused", response
    assert expected in response["payload"]["reason"], response
    assert not (host.root / RUN_ID / "finish.request").exists(), (
        "a refused collection must not ask the unit to finish"
    )


def test_disarm_stops_the_unit_drops_the_program_copy_and_keeps_the_records(
    host: FakeHost,
) -> None:
    scope, receipt = _armed(host)
    probe.write_private_json(host.root / RUN_ID / "final.json", {"record": "final"})
    response = probe.handle_request(request("disarm", scope, run_id=RUN_ID))
    assert response["kind"] == "disarmed", response
    payload = response["payload"]
    assert payload["state_present"] is True and payload["unit"] == receipt["unit"]
    assert payload["unit_state_before"]["ActiveState"] == "active"
    assert payload["unit_state"]["ActiveState"] == "inactive"
    assert ["systemctl", "stop", receipt["unit"]] in host.commands
    assert ["systemctl", "reset-failed", receipt["unit"]] in host.commands
    directory = host.root / RUN_ID
    assert not (directory / "program").exists(), "the program copy is the probe's own"
    assert (directory / "armed.json").is_file() and (directory / "final.json").is_file()
    state = json.loads((directory / "state.json").read_text())
    assert state["phase"] == "DISARMED" and state["disarmed_at"]
    first = state["disarmed_at"]
    again = probe.handle_request(request("disarm", scope, run_id=RUN_ID))
    assert again["kind"] == "disarmed"
    assert json.loads((directory / "state.json").read_text())["disarmed_at"] == first
    # A run that was never armed on this host is not an error either.
    other = owned_scope(run_id="destr015-never-armed-a1")
    fresh = probe.handle_request(
        request("disarm", other, run_id="destr015-never-armed-a1")
    )
    assert fresh["kind"] == "disarmed" and fresh["payload"]["state_present"] is False


def test_status_reads_without_touching_the_unit(host: FakeHost) -> None:
    scope, receipt = _armed(host)
    before = len(host.stops())
    response = probe.handle_request(request("status", scope, run_id=RUN_ID))
    assert response["kind"] == "status", response
    payload = response["payload"]
    assert payload["state"]["phase"] == "ARMED"
    assert payload["armed"] == receipt["armed"] and payload["final"] is None
    assert payload["unit_state"]["ActiveState"] == "active"
    assert payload["boot_id"] == BOOT
    assert len(host.stops()) == before, "status must never stop a unit"
    assert host.starts() == [host.starts()[0]], "status must never start a unit"


# --------------------------------------------------------------------------- #
# Request framing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "message",
    [
        "nope",
        {"kind": "arm"},
        {"kind": "explode", "scope_sha256": "a" * 64, "payload": {}},
        {"kind": "status", "scope_sha256": 7, "payload": {}},
        {"kind": "status", "scope_sha256": "a" * 64, "payload": []},
    ],
)
def test_malformed_requests_are_refused_without_a_handler(
    host: FakeHost, message: Any
) -> None:
    response = probe.handle_request(message)
    assert response["kind"] == "refused", response
    assert "unbound or unsupported" in response["payload"]["reason"]
    assert host.commands == [], "an unbound request must not reach the host"


def test_unexpected_failures_leave_only_their_type(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise ValueError("/etc/gpu-fault/node-agent.env leaked")

    monkeypatch.setitem(probe.HANDLERS, "status", explode)
    scope = owned_scope()
    response = probe.handle_request(request("status", scope, run_id=RUN_ID))
    assert response == {
        "kind": "refused",
        "scope_sha256": scope.digest(),
        "payload": {"reason": "ValueError"},
    }


def test_oversized_responses_are_replaced_by_a_refusal(
    capsys: pytest.CaptureFixture[str],
) -> None:
    probe.emit(
        {"kind": "status", "scope_sha256": "a" * 64, "payload": {"blob": "x" * 70000}}
    )
    printed = json.loads(capsys.readouterr().out.strip())
    assert printed["kind"] == "refused"
    assert "oversized" in printed["payload"]["reason"]


# --------------------------------------------------------------------------- #
# The unit
# --------------------------------------------------------------------------- #
class FakeTracer:
    calls: list[str] = []
    failures: dict[str, BaseException] = {}
    raw: bytes = b""

    def __init__(self, directory: Path, tracee: Any, **kwargs: Any) -> None:
        FakeTracer.calls.append("create")
        self.directory = directory
        self.deadline = kwargs["deadline"]

    def start(self) -> None:
        FakeTracer.calls.append("start")
        if "start" in FakeTracer.failures:
            raise FakeTracer.failures["start"]

    def check(self) -> None:
        FakeTracer.calls.append("check")
        if "check" in FakeTracer.failures:
            raise FakeTracer.failures["check"]

    def snapshot(self) -> bytes:
        FakeTracer.calls.append("snapshot")
        return FakeTracer.raw

    def close(self) -> None:
        FakeTracer.calls.append("close")


@pytest.fixture
def unit(host: FakeHost, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    scope = owned_scope()
    FakeTracer.calls = []
    FakeTracer.failures = {}
    FakeTracer.raw = trace_process(
        100, 1, 2, ("nvidia-smi", *QUERY_ARGS), executable=str(host.executable)
    ) + trace_process(
        101,
        4,
        9,
        ("nvidia-smi", "--gpu-reset", "-i", scope.gpu_uuid),
        executable=str(host.executable),
    )
    monkeypatch.setattr(probe, "AttachedExecWitness", FakeTracer)
    monkeypatch.setattr(
        probe,
        "process_identity",
        lambda pid: SimpleNamespace(
            model_dump=lambda **_kw: {
                "pid": pid,
                "start_ticks": 3,
                "uid": 0,
                "boot_id": host.boot,
            }
        ),
    )
    monkeypatch.setattr(probe, "reset_ledger_rows", lambda: [])
    directory = host.root / RUN_ID
    probe.write_private_json(
        directory / "state.json",
        {
            "case_id": probe.CASE_ID,
            "run_id": RUN_ID,
            "node": scope.node,
            "phase": "ARMED",
            "scope": scope.model_dump(mode="json"),
            "scope_sha256": scope.digest(),
            "boot_id": BOOT,
            "lifetime_seconds": LIFETIME,
            "unit": probe.unit_name(RUN_ID) + ".service",
        },
    )
    return SimpleNamespace(scope=scope, directory=directory, root=host.root, host=host)


def test_run_witness_records_armed_then_final_on_a_finish_request(
    unit: SimpleNamespace,
) -> None:
    (unit.directory / "finish.request").write_text("{}", encoding="utf-8")
    assert probe.run_witness(RUN_ID, state_root=unit.root) == 0
    armed = json.loads((unit.directory / "armed.json").read_text())
    final = json.loads((unit.directory / "final.json").read_text())
    assert armed["record"] == "armed" and final["record"] == "final"
    assert armed["run_id"] == final["run_id"] == RUN_ID
    assert armed["scope_sha256"] == final["scope_sha256"] == unit.scope.digest()
    assert armed["unit"] == final["unit"] == probe.unit_name(RUN_ID) + ".service"
    assert armed["boot_id"] == final["boot_id"] == BOOT
    assert armed["witness_id"] == final["witness_id"]
    assert armed["start"]["tracee"]["boot_id"] == BOOT
    assert final["start_sha256"] == evidence_digest(armed["start"])
    assert final["reason"] == "finish-request" and final["refusal"] is None
    assert final["trace_complete"] is True and final["closed"] is True
    assert final["lost_events"] == 0 and final["trace_bytes"] == len(FakeTracer.raw)
    assert final["calibration_execs"] == 1 and len(final["actions"]) == 1
    action = final["actions"][0]
    assert action["gpu_uuid"] == GPU and action["operation"] == "RESET_GPU"
    assert "argv" not in action, "raw argv never enters a record"
    assert armed["monotonic_ns"] <= final["monotonic_ns"]
    assert armed["realtime_ns"] <= final["realtime_ns"]
    assert final["wall_minus_monotonic_min_ns"] <= final["wall_minus_monotonic_max_ns"]
    assert FakeTracer.calls == ["create", "start", "snapshot", "close"]
    assert (unit.directory / "trace").stat().st_mode & 0o777 == 0o700
    for name in ("armed.json", "final.json"):
        assert (unit.directory / name).stat().st_mode & 0o777 == 0o600, name


def test_run_witness_stops_itself_at_its_deadline(unit: SimpleNamespace) -> None:
    state = json.loads((unit.directory / "state.json").read_text())
    state["lifetime_seconds"] = 0
    probe.write_private_json(unit.directory / "state.json", state)
    assert probe.run_witness(RUN_ID, state_root=unit.root) == 0
    final = json.loads((unit.directory / "final.json").read_text())
    assert final["reason"] == "deadline" and final["closed"] is True
    assert "check" not in FakeTracer.calls[:2]


@pytest.mark.parametrize(
    ("defect", "expected", "armed_written"),
    [
        ("start", "Agent is already traced", False),
        ("check", "exec witness continuity was lost", True),
        ("boot", "host boot id differs from the scope", False),
        ("no-state", "unit started without an armed state", False),
    ],
)
def test_run_witness_records_a_failure_and_never_a_proof(
    unit: SimpleNamespace, defect: str, expected: str, armed_written: bool
) -> None:
    if defect in {"start", "check"}:
        FakeTracer.failures[defect] = BoundaryDenied(expected)
    elif defect == "boot":
        unit.host.boot = "another-boot"
    else:
        (unit.directory / "state.json").unlink()
    assert probe.run_witness(RUN_ID, state_root=unit.root) == 1
    final = json.loads((unit.directory / "final.json").read_text())
    assert final["record"] == "final" and final["reason"] == "failure"
    assert final["refusal"] == expected
    assert final["trace_complete"] is False and final["actions"] == []
    assert (unit.directory / "armed.json").exists() is armed_written
    if defect in {"start", "check"}:
        assert FakeTracer.calls[-1] == "close", "a failed unit still detaches"
    else:
        assert FakeTracer.calls == [], "no tracer may exist before identity holds"


def test_run_witness_keeps_an_unproven_trace_as_a_refusal(
    unit: SimpleNamespace,
) -> None:
    FakeTracer.raw = trace_process(
        100, 1, 2, ("nvidia-smi", *QUERY_ARGS), executable=str(unit.host.executable)
    )
    (unit.directory / "finish.request").write_text("{}", encoding="utf-8")
    assert probe.run_witness(RUN_ID, state_root=unit.root) == 0
    final = json.loads((unit.directory / "final.json").read_text())
    assert final["actions"] == [] and final["calibration_execs"] == 0
    assert "calibration" in final["refusal"]
    assert final["closed"] is True


def test_run_witness_refuses_a_changed_executable(unit: SimpleNamespace) -> None:
    original = FakeTracer.snapshot

    def snapshot(self: FakeTracer) -> bytes:
        unit.host.executable.write_bytes(b"replaced nvidia-smi")
        return original(self)

    FakeTracer.snapshot = snapshot  # type: ignore[method-assign]
    try:
        (unit.directory / "finish.request").write_text("{}", encoding="utf-8")
        assert probe.run_witness(RUN_ID, state_root=unit.root) == 0
    finally:
        FakeTracer.snapshot = original  # type: ignore[method-assign]
    final = json.loads((unit.directory / "final.json").read_text())
    assert final["refusal"] == "physical reset executable changed"


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #
def test_main_runs_the_unit_or_answers_one_request(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        probe.main(["witness", "--run-id", RUN_ID, "--state-root", str(host.root)]) == 1
    )
    final = json.loads((host.root / RUN_ID / "final.json").read_text())
    assert final["refusal"] == "unit started without an armed state"
    scope = owned_scope(run_id="destr015-status-only-a1")
    message = request("status", scope, run_id="destr015-status-only-a1")
    monkeypatch.setattr(probe.sys, "stdin", io.StringIO(json.dumps(message) + "\n"))
    assert probe.main([]) == 0
    printed = json.loads(capsys.readouterr().out.strip())
    assert printed["kind"] == "status" and printed["payload"]["state"] is None
    monkeypatch.setattr(probe.sys, "stdin", io.StringIO(""))
    with pytest.raises(json.JSONDecodeError):
        probe.main([])
    monkeypatch.setattr(probe.sys, "stdin", io.StringIO("x" * 70000 + "\n"))
    assert probe.main([]) == 1
    printed = json.loads(capsys.readouterr().out.strip())
    assert (
        printed["kind"] == "refused"
        and "exceeds its bound" in printed["payload"]["reason"]
    )
