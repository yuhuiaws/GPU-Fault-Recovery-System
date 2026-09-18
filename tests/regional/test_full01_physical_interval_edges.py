from __future__ import annotations

import io
import json
import runpy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.destr015_physical_evidence import ResetIntervalScope
from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied
from scripts.e2e.regional.probes import destr015_physical_probe as probe
from tests.regional import (
    test_acceptance_physical_witness_lifecycle as witness_fixtures,
)
from tests.regional._cov95_identity_support import offline_guard as offline_guard
from tests.regional.test_acceptance_physical_interval_alignment import (
    BASE,
    SECOND,
    interval_fixture,
    trace_process,
)

node_witness = witness_fixtures.node_witness


def bound_scope(**changes: Any) -> ResetIntervalScope:
    _, scopes, _, _ = interval_fixture()
    return scopes["node-a"].model_copy(
        update={
            "maintenance_end": datetime.now(timezone.utc) + timedelta(hours=1),
            **changes,
        }
    )


def message(scope: ResetIntervalScope, kind: str, **changes: Any) -> dict[str, Any]:
    return {
        "kind": kind,
        "scope_sha256": scope.digest(),
        "payload": scope.model_dump(mode="json") if kind == "arm" else {},
        **changes,
    }


def set_input(monkeypatch: pytest.MonkeyPatch, *messages: dict[str, Any]) -> None:
    monkeypatch.setattr(
        probe.sys,
        "stdin",
        io.StringIO("".join(json.dumps(item) + "\n" for item in messages)),
    )


@pytest.mark.parametrize("defect", ["reversed", "interrupted"])
def test_clock_sample_refuses_an_unbounded_monotonic_bracket(
    monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    ticks = iter(
        [
            SECOND,
            SECOND - 1 if defect == "reversed" else SECOND + probe.CLOCK_MARGIN_NS + 1,
        ]
    )
    monkeypatch.setattr(
        probe,
        "time",
        SimpleNamespace(monotonic_ns=lambda: next(ticks), time_ns=lambda: BASE),
    )
    clock = probe.ClockEnvelope()
    with pytest.raises(BoundaryDenied, match="clock sampling was interrupted"):
        clock.sample()
    assert clock.previous is None and clock.minimum is None and clock.maximum is None, (
        "a rejected sample must not extend the trusted clock envelope"
    )


@pytest.mark.parametrize("defect", ["no-reset", "late-calibration"])
def test_completed_query_must_precede_a_real_reset_interval(defect: str) -> None:
    executable = Path("/unit/nvidia-smi")
    raw = trace_process(
        100, 10, 12, ("nvidia-smi", *probe.QUERY_ARGS), executable=str(executable)
    )
    if defect == "late-calibration":
        raw += trace_process(
            101,
            1,
            3,
            ("nvidia-smi", "--gpu-reset", "-i", "GPU-a"),
            executable=str(executable),
        )
    with pytest.raises(BoundaryDenied, match="no completed query calibration"):
        probe.reset_events(raw, executable=executable, gpu_uuid="GPU-a")


def test_foreign_executable_is_refused_by_the_real_action_parser() -> None:
    raw = trace_process(100, 1, 2, ("nvidia-smi", *probe.QUERY_ARGS))
    raw += trace_process(
        101,
        3,
        4,
        ("nvidia-smi", "--gpu-reset", "-i", "GPU-a"),
        executable="/unit/foreign-nvidia-smi",
    )
    with pytest.raises(BoundaryDenied, match="unapproved executable"):
        probe.reset_events(
            raw, executable=Path("/usr/bin/nvidia-smi"), gpu_uuid="GPU-a"
        )


def test_probe_entrypoint_rejects_incomplete_arm_input_without_starting_io(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(probe.sys, "stdin", io.StringIO('{"kind":"arm"}'))
    with pytest.raises(BoundaryDenied, match="arm request is incomplete"):
        runpy.run_path(str(probe.__file__), run_name="__main__")
    assert capsys.readouterr().out == "", (
        "the entrypoint must not announce an armed witness for incomplete input"
    )


@pytest.mark.parametrize("defect", ["kind", "scope"])
def test_arm_request_must_match_its_exact_scope_and_operation(
    node_witness: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    scope = bound_scope()
    initial = message(scope, "arm")
    initial["kind" if defect == "kind" else "scope_sha256"] = (
        "finish" if defect == "kind" else "0" * 64
    )
    set_input(monkeypatch, initial)
    with pytest.raises(BoundaryDenied, match="arm request is unbound"):
        probe.main()
    assert node_witness.calls == [], "an unbound request cannot allocate a witness"


@pytest.mark.parametrize("seconds", [-1, 7201], ids=["expired", "too-long"])
def test_arm_rejects_expired_and_excessive_maintenance_windows(
    node_witness: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, seconds: int
) -> None:
    now = datetime(2026, 9, 15, tzinfo=timezone.utc)
    monkeypatch.setattr(probe, "datetime", SimpleNamespace(now=lambda _timezone: now))
    scope = bound_scope(maintenance_end=now + timedelta(seconds=seconds))
    set_input(monkeypatch, message(scope, "arm"))
    with pytest.raises(BoundaryDenied, match="bounded maintenance window"):
        probe.main()
    assert node_witness.calls == [], "a refused window must not create a tracer"


@pytest.mark.parametrize(
    ("defect", "expected"),
    [
        ("executable", "reset executable is unavailable"),
        ("pid", "no Agent process identity"),
        ("boot", "Agent boot differs"),
    ],
)
def test_witness_requires_resolved_executable_pid_and_bound_boot_before_attach(
    node_witness: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
    expected: str,
) -> None:
    if defect == "executable":
        monkeypatch.setattr(probe.shutil, "which", lambda _name: None)
    elif defect == "pid":
        monkeypatch.setattr(
            probe.subprocess,
            "run",
            lambda *_args, **_kwargs: SimpleNamespace(stdout=""),
        )
    else:
        monkeypatch.setattr(
            probe,
            "process_identity",
            lambda _pid: SimpleNamespace(boot_id="other-boot"),
        )
    with pytest.raises(BoundaryDenied, match=expected):
        node_witness.run("abort")
    assert node_witness.calls == [], "identity refusal must precede witness creation"


def test_oversized_arm_response_closes_witness_without_publishing_partial_identity(
    node_witness: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scope = bound_scope(boot_id="owned-boot-" + "b" * (probe.MAX_MESSAGE_BYTES // 2))

    def identity(pid: int) -> SimpleNamespace:
        return SimpleNamespace(
            boot_id=scope.boot_id,
            model_dump=lambda **kwargs: {
                "pid": pid,
                "boot_id": scope.boot_id,
                "start_ticks": 1,
            },
        )

    initial = message(scope, "arm")
    assert len((json.dumps(initial) + "\n").encode()) < probe.MAX_MESSAGE_BYTES, (
        "the arm input must fit while its repeated process identities exceed output bounds"
    )
    monkeypatch.setattr(probe, "process_identity", identity)
    set_input(monkeypatch, initial)
    with pytest.raises(BoundaryDenied, match="response is oversized"):
        probe.main()
    assert capsys.readouterr().out == "", "no partial armed receipt may be emitted"
    assert [call[0] for call in node_witness.calls] == ["create", "start", "close"], (
        "output-bound failure must still close the owned witness"
    )


@pytest.mark.parametrize("defect", ["scope", "payload"])
def test_controller_request_drift_closes_without_a_completion_proof(
    node_witness: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    defect: str,
) -> None:
    scope = bound_scope()
    request = message(scope, "finish")
    request["scope_sha256" if defect == "scope" else "payload"] = (
        "0" * 64 if defect == "scope" else {"node": "foreign"}
    )
    set_input(monkeypatch, message(scope, "arm"), request)
    with pytest.raises(BoundaryDenied, match="witness request is unbound"):
        probe.main()
    messages = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [item["kind"] for item in messages] == ["armed"], (
        "a foreign controller request must not produce a completed capture"
    )
    assert node_witness.calls[-1] == ("close",), "request drift must detach the witness"


def test_idle_controller_poll_retains_supervision_until_a_bound_abort(
    node_witness: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    selections = iter([False, True])
    checks: list[str] = []

    def select(
        readers: list[Any], *_args: Any
    ) -> tuple[list[Any], list[Any], list[Any]]:
        return (readers if next(selections) else [], [], [])

    monkeypatch.setattr(probe.select, "select", select)
    monkeypatch.setattr(
        probe.AttachedExecWitness, "check", lambda _self: checks.append("checked")
    )
    assert node_witness.run("abort") == 0, "an idle read must not lose the later abort"
    messages = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert checks == ["checked", "checked"], "each poll must check witness supervision"
    assert [item["kind"] for item in messages] == ["armed", "closed"], (
        "an aborted capture must never be reported as finished"
    )
    assert messages[-1]["payload"] == {"closed": True, "proof_complete": False}, (
        "abort cleanup is not proof of a complete physical interval"
    )


def test_executable_byte_drift_after_snapshot_prevents_a_finished_receipt(
    node_witness: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original = probe.AttachedExecWitness.snapshot

    def snapshot(witness: Any) -> bytes:
        raw = original(witness)
        node_witness.executable.write_bytes(b"changed owned executable fixture")
        return raw

    monkeypatch.setattr(probe.AttachedExecWitness, "snapshot", snapshot)
    with pytest.raises(BoundaryDenied, match="reset executable changed"):
        node_witness.run("finish")
    messages = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [item["kind"] for item in messages] == ["armed"], (
        "trace parsing cannot authorize completion after executable identity drift"
    )
    assert node_witness.calls[-1] == ("close",), (
        "the changed executable must not leak a tracer"
    )


@pytest.mark.parametrize("command", ["abort", "finish"])
def test_witness_close_failure_never_emits_success_or_a_cleanup_receipt(
    node_witness: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    original = probe.AttachedExecWitness.close

    def close(witness: Any) -> None:
        original(witness)
        raise OSError("owned witness cleanup unavailable")

    monkeypatch.setattr(probe.AttachedExecWitness, "close", close)
    with pytest.raises(OSError, match="owned witness cleanup unavailable"):
        node_witness.run(command)
    messages = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [item["kind"] for item in messages] == ["armed"], (
        "a failed detach must not publish closed or finished evidence"
    )
    assert node_witness.calls[-2:] == [("close",), ("close",)], (
        "the finally path must retry owned witness cleanup and preserve its failure"
    )
