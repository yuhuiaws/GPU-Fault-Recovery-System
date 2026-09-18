from __future__ import annotations

import io
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import destr015_physical_intervals as controller
from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied
from scripts.e2e.regional.probes import destr015_physical_probe as node
from tests.regional.test_acceptance_physical_interval_alignment import (
    interval_fixture,
    trace_process,
)


@pytest.fixture
def witness_controller(monkeypatch):
    captures, scopes, _, _ = interval_fixture()
    capture, scope = captures["node-a"], scopes["node-a"]
    calls = []
    failures = {}

    class Stream:
        def send(self, kind, payload):
            calls.append(("send", kind, payload))
            if kind in failures:
                raise failures[kind]

        def receive(self, kind):
            calls.append(("receive", kind))
            if kind in failures:
                raise failures[kind]
            if kind == "closed":
                return {"closed": True, "proof_complete": False}
            if kind == "armed":
                return {"start": capture["start"], "monotonic_ns": 50}
            if kind == "finished":
                return {"end": capture["end"], "monotonic_ns": 60}
            return {"monotonic_ns": 55}

        def finish(self):
            calls.append(("finish",))

        def close(self):
            calls.append(("close",))

    def opened(**kwargs):
        calls.append(("open", kwargs))
        return object()

    monkeypatch.setattr(controller, "open_executor_stream", opened)
    monkeypatch.setattr(controller, "ProbeStream", lambda *a, **kw: Stream())
    monkeypatch.setattr(
        controller, "probe_program", lambda role: ("measured-code", "digest")
    )
    probe = SimpleNamespace(
        _check_target=lambda: calls.append(("identity",)),
        pod="owned-pod",
        settings=SimpleNamespace(
            kubeconfig="/fake-owned-kubeconfig",
            context="owned-context",
            namespace="owned-namespace",
        ),
    )
    return controller.ResetIntervalWitness(probe, scope), calls, failures


def test_physical_witness_binds_transport_and_collects_before_detaching(
    witness_controller,
):
    witness, calls, _ = witness_controller
    witness.start()
    witness.poll()
    capture = witness.finish()
    assert witness.close() == {"closed": True, "proof_complete": True}
    assert calls[0] == ("identity",)
    opened = calls[1][1]
    assert opened["chroot"] == "/host" and opened["container"] == "probe"
    assert opened["pod"] == "owned-pod" and opened["context"] == "owned-context"
    assert calls[2] == ("send", "arm", witness.scope.model_dump(mode="json"))
    assert len(capture["clock_exchanges"]) == 4
    assert capture["end"]["start_sha256"]
    assert calls[-1] == ("finish",)
    with pytest.raises(BoundaryDenied, match="reused"):
        witness.start()


def test_physical_witness_abort_never_claims_completed_proof(witness_controller):
    witness, calls, _ = witness_controller
    assert witness.close()["not_started"] is True
    witness.start()
    assert witness.close() == {"closed": True, "proof_complete": False}
    assert calls[-1] == ("close",)


@pytest.mark.parametrize("stage", ["armed", "clock", "finished", "closed"])
def test_physical_witness_transport_failure_cannot_be_a_cleanup_success(
    witness_controller, stage
):
    witness, calls, failures = witness_controller
    if stage == "armed":
        failures[stage] = BoundaryDenied("lost acknowledgement")
        with pytest.raises(BoundaryDenied):
            witness.start()
        assert witness.close()["proof_complete"] is False
        return
    witness.start()
    failures[stage] = BoundaryDenied("lost acknowledgement")
    with pytest.raises(BoundaryDenied):
        if stage == "clock":
            witness.poll()
        elif stage == "finished":
            witness.finish()
        else:
            witness.close()
    assert not witness.finished, "lost supervision cannot finalize a physical proof"
    if stage == "closed":
        assert calls[-1] == ("close",)


@pytest.fixture
def node_witness(tmp_path, monkeypatch):
    _, scopes, _, _ = interval_fixture()
    scope = scopes["node-a"].model_copy(
        update={"maintenance_end": datetime.now(timezone.utc) + timedelta(hours=1)}
    )
    executable = tmp_path / "nvidia-smi"
    executable.write_bytes(b"measured executable fixture")
    calls = []
    failures = {}
    raw = trace_process(
        100, 1, 2, ("nvidia-smi", *node.QUERY_ARGS), executable=str(executable)
    )
    raw += trace_process(
        101,
        4,
        9,
        ("nvidia-smi", "--gpu-reset", "-i", scope.gpu_uuid),
        executable=str(executable),
    )

    class Witness:
        def __init__(self, directory, tracee, **kwargs):
            calls.append(("create", tracee, kwargs))

        def start(self):
            calls.append(("start",))
            if "start" in failures:
                raise failures["start"]

        def check(self):
            if "check" in failures:
                raise failures["check"]

        def snapshot(self):
            return raw

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(node, "AttachedExecWitness", Witness)
    monkeypatch.setattr(node.shutil, "which", lambda value: str(executable))
    monkeypatch.setattr(
        node.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="10")
    )
    monkeypatch.setattr(
        node,
        "process_identity",
        lambda pid: SimpleNamespace(
            boot_id=scope.boot_id,
            model_dump=lambda **kw: {
                "pid": pid,
                "boot_id": scope.boot_id,
                "start_ticks": 1,
            },
        ),
    )
    monkeypatch.setattr(node.select, "select", lambda *args: ([node.sys.stdin], [], []))

    def request(kind, payload):
        return (
            json.dumps(
                {"kind": kind, "scope_sha256": scope.digest(), "payload": payload}
            )
            + "\n"
        )

    def run(*kinds):
        stream = request("arm", scope.model_dump(mode="json"))
        stream += "".join(request(kind, {}) for kind in kinds)
        monkeypatch.setattr(node.sys, "stdin", io.StringIO(stream))
        return node.main()

    return SimpleNamespace(
        run=run, calls=calls, failures=failures, executable=executable
    )


def test_node_witness_emits_only_bound_intervals_then_closes_its_tracer(
    node_witness, capsys
):
    assert node_witness.run("clock", "finish") == 0
    messages = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [message["kind"] for message in messages] == ["armed", "clock", "finished"]
    assert [message["sequence"] for message in messages] == [1, 2, 3]
    end = messages[-1]["payload"]["end"]
    assert end["closed"] is True and end["trace_complete"] is True
    assert len(end["actions"]) == 1 and end["calibration_execs"] == 1
    assert "argv" not in end["actions"][0]
    assert node_witness.calls[-1] == ("close",)


@pytest.mark.parametrize("stop", ["abort", "eof", "unsupported", "start", "check"])
def test_node_witness_never_signals_agent_or_invents_a_proof_after_failure(
    node_witness, capsys, stop
):
    if stop in {"start", "check"}:
        node_witness.failures[stop] = BoundaryDenied("lost coverage")
    if stop == "abort":
        assert node_witness.run("abort") == 0
    else:
        with pytest.raises(BoundaryDenied):
            commands = ("unsupported",) if stop == "unsupported" else ()
            node_witness.run(*commands)
    messages = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert "finished" not in [message["kind"] for message in messages]
    assert node_witness.calls[-1] == ("close",)
