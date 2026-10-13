"""Controller, verdict and runner-flow contracts of the detached DESTR-015 witnesses.

The controller talks to the host through one interactive exec session per arm
and collect (faked at the session opener: the request, its response, the clock
pings and the end message) and one-shot ``kubectl exec`` requests for disarm
and status (faked at the ``regional.kubectl`` boundary). The verdict functions
read only the durable records, and the runner flow is driven end to end through
the branch harness, whose probes and witnesses refuse exactly where a NotReady
node would refuse a real exec.
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import destr015_detached_witness as controller
from scripts.e2e.regional import destr015_verdicts as verdicts
from scripts.e2e.regional import run_destr015_parallel_branch_join as case
from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied
from scripts.e2e.regional.late_ownership_probe_bundle import probe_program, stdin_loader
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._cov95_destr_branches import NODES, BranchHarness
from tests.regional.test_acceptance_physical_interval_alignment import (
    SECOND,
    interval_fixture,
)

NOW = datetime(2026, 9, 15, 12, tzinfo=timezone.utc)
CASE_ID = "GF-REGIONAL-DESTR-015"


# --------------------------------------------------------------------------- #
# Record builders
# --------------------------------------------------------------------------- #
def witness_records(node: str = "node-a") -> SimpleNamespace:
    """A consistent arm receipt, collection and exchange pair for one node,
    folded from the interval fixture so the same interval arithmetic grades
    both the legacy capture and the detached records."""

    captures, scopes, workflow, hosts = interval_fixture()
    scope, capture = scopes[node], captures[node]
    start, end = capture["start"], capture["end"]
    wall = end["wall_minus_monotonic_min_ns"]
    unit = f"gpu-fault-destr015-witness-{'0' * 16}.service"

    def clock(seconds: float) -> dict[str, int]:
        mono = int(seconds * SECOND)
        return {"monotonic_ns": mono, "realtime_ns": wall + mono}

    common = {
        "case_id": CASE_ID,
        "run_id": scope.run_id,
        "node": node,
        "scope_sha256": scope.digest(),
        "unit": unit,
        "invocation_id": "invocation-owned",
        "witness_id": start["witness_id"],
        "boot_id": scope.boot_id,
    }
    armed = {
        **common,
        "record": "armed",
        **clock(49.5),
        "start": deepcopy(start),
        "ledger_baseline_command_ids": [],
    }
    final = {
        **common,
        "record": "final",
        **clock(149.5),
        "reason": "finish-request",
        **{key: deepcopy(end[key]) for key in verdicts.WITNESS_END_KEYS},
        "refusal": None,
        "ledger_reset_rows": [],
    }
    receipt = {
        "run_id": scope.run_id,
        "node": node,
        "unit": unit,
        "state_dir": f"/var/lib/gpu-fault-acceptance/destr015/{scope.run_id}",
        "lifetime_seconds": 3600,
        "boot_id": scope.boot_id,
        "program_sha256": "f" * 64,
        "armed": armed,
        "host_clock_start": clock(49),
        "host_clock": clock(50),
        "runner_clock": {"monotonic_ns": 0, "realtime_ns": 0},
        "unit_state": {"ActiveState": "active"},
    }
    collection = {
        "run_id": scope.run_id,
        "node": node,
        "unit": unit,
        "state_dir": receipt["state_dir"],
        "boot_id": scope.boot_id,
        "state": {
            "run_id": scope.run_id,
            "node": node,
            "phase": "ARMED",
            "scope_sha256": scope.digest(),
            "boot_id": scope.boot_id,
            "unit": unit,
            "lifetime_seconds": 3600,
        },
        "armed": deepcopy(armed),
        "final": final,
        "finish_requested": clock(149),
        "unit_state": {"ActiveState": "inactive"},
        "host_clock_start": clock(149),
        "host_clock": clock(150),
    }
    exchanges = [
        {
            "kind": kind,
            **item,
            "realtime_ns": wall + item["monotonic_ns"],
            "runner_sent_realtime_ns": 0,
            "runner_received_realtime_ns": 0,
        }
        for kind, item in zip(
            ("arm", "collect"), capture["clock_exchanges"], strict=True
        )
    ]
    return SimpleNamespace(
        scope=scope,
        receipt=receipt,
        collection=collection,
        exchanges=exchanges,
        workflow=workflow,
        hosts=hosts,
        capture=capture,
    )


def both_witnesses() -> tuple[
    dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]
]:
    records = {node: witness_records(node) for node in ("node-a", "node-b")}
    witnesses = {
        node: {
            "scope": item.scope,
            "receipt": item.receipt,
            "collection": item.collection,
            "exchanges": item.exchanges,
        }
        for node, item in records.items()
    }
    return witnesses, records["node-a"].workflow, records["node-a"].hosts


# --------------------------------------------------------------------------- #
# Verdicts
# --------------------------------------------------------------------------- #
def test_detached_records_fold_into_the_same_capture_the_stream_produced() -> None:
    records = witness_records()
    capture = verdicts.witness_capture(
        records.receipt, records.collection, records.exchanges
    )
    assert capture == {
        "start": records.capture["start"],
        "end": records.capture["end"],
        "clock_exchanges": records.capture["clock_exchanges"],
    }


def test_records_written_during_the_run_carry_no_error() -> None:
    records = witness_records()
    assert (
        verdicts.witness_record_errors(
            "node-a",
            scope=records.scope,
            receipt=records.receipt,
            collection=records.collection,
        )
        == []
    )
    witnesses, workflow, hosts = both_witnesses()
    assert (
        verdicts.physical_witness_errors(witnesses, workflow=workflow, hosts=hosts)
        == []
    )


@pytest.mark.parametrize(
    ("defect", "expected"),
    [
        ("no-receipt", "no armed record"),
        ("no-armed", "no armed record"),
        ("no-collection", "was not collected"),
        ("no-final", "left no final record"),
        ("refusal", "witness refused"),
        ("reason", "ended by failure"),
        ("scope-final", "not bound to this scope"),
        ("run-armed", "not bound to this scope"),
        ("boot-receipt", "boot id"),
        ("boot-final", "boot id"),
        ("boot-collection", "boot id"),
        ("boot-state", "boot id"),
        ("witness-id", "another witness"),
        ("unit", "another unit"),
        ("armed-before-arm", "arm..collect window"),
        ("final-after-collect", "arm..collect window"),
        ("final-before-armed", "arm..collect window"),
        ("clock-missing", "arm..collect window"),
        ("realtime", "realtime stamps"),
    ],
)
def test_records_outside_the_run_or_unbound_to_it_fail(
    defect: str, expected: str
) -> None:
    records = witness_records()
    receipt, collection = records.receipt, records.collection
    final, armed = collection["final"], receipt["armed"]
    if defect == "no-receipt":
        receipt = None
    elif defect == "no-armed":
        receipt["armed"] = None
    elif defect == "no-collection":
        collection = None
    elif defect == "no-final":
        collection["final"] = None
    elif defect == "refusal":
        final["refusal"] = "physical reset trace has no completed query calibration"
    elif defect == "reason":
        final["reason"] = "failure"
    elif defect == "scope-final":
        final["scope_sha256"] = "0" * 64
    elif defect == "run-armed":
        armed["run_id"] = "another-run"
        collection["armed"]["run_id"] = "another-run"
    elif defect == "boot-receipt":
        receipt["boot_id"] = "rebooted"
    elif defect == "boot-final":
        final["boot_id"] = "rebooted"
    elif defect == "boot-collection":
        collection["boot_id"] = "rebooted"
    elif defect == "boot-state":
        collection["state"]["boot_id"] = "rebooted"
    elif defect == "witness-id":
        final["witness_id"] = "another-witness"
    elif defect == "unit":
        final["unit"] = "gpu-fault-destr015-witness-ffffffffffffffff.service"
    elif defect == "armed-before-arm":
        armed["monotonic_ns"] = receipt["host_clock_start"]["monotonic_ns"] - 1
    elif defect == "final-after-collect":
        final["monotonic_ns"] = collection["host_clock"]["monotonic_ns"] + 1
    elif defect == "final-before-armed":
        final["monotonic_ns"] = armed["monotonic_ns"] - 1
    elif defect == "clock-missing":
        collection["host_clock"] = {"realtime_ns": 1}
    else:
        final["realtime_ns"] = collection["host_clock"]["realtime_ns"] + SECOND
    errors = verdicts.witness_record_errors(
        "node-a", scope=records.scope, receipt=receipt, collection=collection
    )
    assert any(expected in item and "node-a" in item for item in errors), (
        defect,
        errors,
    )


def test_lost_or_unbound_records_never_reach_the_overlap_arithmetic() -> None:
    witnesses, workflow, hosts = both_witnesses()
    witnesses["node-b"]["collection"]["final"]["monotonic_ns"] += 10 * SECOND
    errors = verdicts.physical_witness_errors(witnesses, workflow=workflow, hosts=hosts)
    assert any("node-b" in item and "arm..collect window" in item for item in errors), (
        errors
    )
    assert errors[-1].startswith(
        "the actual reset process intervals cannot be proven"
    ), errors
    witnesses, workflow, hosts = both_witnesses()
    action = witnesses["node-b"]["collection"]["final"]["actions"][0]
    action["started_ns"] += 17 * SECOND
    action["ended_ns"] += 9 * SECOND
    serial = verdicts.physical_witness_errors(witnesses, workflow=workflow, hosts=hosts)
    assert serial == [
        "the actual reset process intervals do not prove parallel execution"
    ]
    witnesses, workflow, hosts = both_witnesses()
    del witnesses["node-b"]
    assert verdicts.physical_witness_errors(
        witnesses, workflow=workflow, hosts=hosts
    ) == ["physical reset witnesses do not cover exactly the two approved nodes"]


def test_clock_binding_documents_the_offset_bound_the_arithmetic_uses() -> None:
    exchanges = witness_records().exchanges
    binding = verdicts.clock_binding(exchanges)
    span = exchanges[1]["received_ns"] - exchanges[0]["sent_ns"]
    drift = span * 200 // 1_000_000 + 50_000_000
    assert binding["bounded"] is True and binding["exchanges"] == 2
    assert binding["span_ns"] == span and binding["drift_allowance_ns"] == drift
    assert binding["offset_min_ns"] == 950 * SECOND - drift
    assert binding["offset_max_ns"] == 950 * SECOND + 20_000_000 + drift
    assert binding["uncertainty_ns"] == 20_000_000 + 2 * drift
    assert binding["max_clock_drift_ppm"] == 200 and binding["margin_ns"] == 50_000_000
    assert verdicts.clock_binding(exchanges[:1]) == {
        "bounded": False,
        "reason": "fewer than two integer clock exchanges",
    }
    reversed_pair = [dict(exchanges[1]), dict(exchanges[0])]
    assert (
        verdicts.clock_binding(reversed_pair)["reason"]
        == "controller clock is reversed"
    )
    stalled = deepcopy(exchanges)
    stalled[1]["monotonic_ns"] = stalled[0]["monotonic_ns"]
    stalled_binding = verdicts.clock_binding(stalled)
    assert stalled_binding["bounded"] is False
    assert stalled_binding["reason"] == "the exchanges admit no common clock offset"
    assert (
        verdicts.clock_binding([{"kind": "status", **exchanges[0]}])["bounded"] is False
    )
    pinged = [exchanges[0], {**exchanges[0], "kind": "arm-ping"}, exchanges[1]]
    assert verdicts.clock_binding(pinged)["exchanges"] == 3, "pings are clock samples"
    foreign = [*exchanges, {**exchanges[1], "kind": "status-ping"}]
    assert verdicts.clock_binding(foreign)["exchanges"] == 2, "only owned kinds count"


# --------------------------------------------------------------------------- #
# Controller
# --------------------------------------------------------------------------- #
class FakeRegional:
    """``kubectl exec -i`` at the boundary: records the request, answers by policy."""

    def __init__(self, respond: Any) -> None:
        self.respond = respond
        self.calls: list[tuple[str, tuple[str, ...], int | None]] = []
        self.requests: list[dict[str, Any]] = []

    def kubectl(
        self,
        plane: str,
        *args: str,
        input_text: str | None = None,
        timeout: int | None = None,
        **_kwargs: Any,
    ) -> str:
        self.calls.append((plane, args, timeout))
        assert input_text is not None, "witness requests travel on stdin"
        request = json.loads(input_text.rstrip("\n").rsplit("\n", 1)[-1])
        self.requests.append(request)
        return str(self.respond(request))


def response(kind: str, scope: Any, payload: dict[str, Any]) -> str:
    return json.dumps(
        {"kind": kind, "scope_sha256": scope.digest(), "payload": payload}
    )


class FakeStream:
    """The interactive exec at the boundary, answering as the probe would.

    The first line is the request: ``owner.answers`` gives the response line
    (``None`` stays silent) and, as the probe does, the process stays for pings
    only after an ``armed`` or ``collected`` response. Each ping is answered by
    ``owner.pong`` (``None`` stays silent) on a host clock that moves on from
    the response's; ``owner.end`` handles the end message.
    """

    def __init__(self, owner: SimpleNamespace, program: str) -> None:
        self.owner = owner
        self.program = program
        self.written: list[dict[str, Any]] = []
        self.stdout = ""
        self.stderr = ""
        self.open = True
        self.closed = False
        self.code: Any = 0
        self.host_ns = 0

    def write_stdin(self, data: str) -> None:
        assert self.open, "nothing is written once the probe has exited"
        assert data.endswith("\n") and data.count("\n") == 1, "one line per message"
        message = json.loads(data)
        self.written.append(message)
        if len(self.written) == 1:
            self._respond(message)
        elif message["kind"] == "ping":
            line = self.owner.pong(self, message)
            if line is not None:
                self.stdout += line + "\n"
        else:
            assert message["kind"] == "end" and message["payload"] == {}, message
            self.owner.end(self)

    def _respond(self, request: dict[str, Any]) -> None:
        self.stderr += self.owner.stderr
        text = self.owner.answers[request["kind"]](request)
        if text is None:
            return
        self.stdout += text + "\n"
        try:
            reply = json.loads(text)
            self.host_ns = reply["payload"]["host_clock"]["monotonic_ns"]
        except (ValueError, KeyError, TypeError):
            reply = {}
        if not self.owner.answers_pings or reply.get("kind") not in {
            "armed",
            "collected",
        }:
            self.open = False

    def read_stdout(self, timeout: float | None = None) -> str:
        assert timeout == 0, "the controller never blocks on a read"
        value, self.stdout = self.stdout, ""
        return value

    def read_stderr(self, timeout: float | None = None) -> str:
        assert timeout == 0, "the controller never blocks on a read"
        value, self.stderr = self.stderr, ""
        return value

    def is_open(self) -> bool:
        return self.open

    def update(self, timeout: float = 0) -> None:
        assert 0 <= timeout <= 1, "every wait is bounded"
        time.sleep(min(timeout, 0.001))

    @property
    def returncode(self) -> Any:
        if self.open:
            return None
        if isinstance(self.code, Exception):
            raise self.code
        return self.code

    def close(self) -> None:
        self.open = False
        self.closed = True


@pytest.fixture
def owned() -> SimpleNamespace:
    records = witness_records()
    wall = records.capture["end"]["wall_minus_monotonic_min_ns"]
    owned = SimpleNamespace(
        records=records,
        checks=[],
        answers={},
        sessions=[],
        stderr="",
        answers_pings=True,
    )
    probe: Any = SimpleNamespace(
        pod="probe-pod", _check_target=lambda: owned.checks.append("uid")
    )

    def respond(request: dict[str, Any]) -> str:
        return str(owned.answers[request["kind"]](request))

    def opener(program: str) -> FakeStream:
        stream = FakeStream(owned, program)
        owned.sessions.append(stream)
        return stream

    def pong_message(stream: FakeStream, message: dict[str, Any]) -> dict[str, Any]:
        stream.host_ns += 1_000_000
        return {
            "kind": "pong",
            "scope_sha256": message["scope_sha256"],
            "payload": {
                "sequence": message["payload"]["sequence"],
                "host_clock": {
                    "monotonic_ns": stream.host_ns,
                    "realtime_ns": wall + stream.host_ns,
                },
            },
        }

    def end(stream: FakeStream) -> None:
        stream.open = False

    owned.regional = FakeRegional(respond)
    owned.witness = controller.DetachedResetWitness(
        owned.regional,
        probe,
        records.scope,
        lifetime_seconds=3600,
        session_opener=opener,
    )
    owned.pong_message = pong_message
    owned.pong = lambda stream, message: json.dumps(pong_message(stream, message))
    owned.end = end
    records.receipt["program_sha256"] = owned.witness.bundle_sha256
    owned.answers["arm"] = lambda request: response(
        "armed", records.scope, records.receipt
    )
    owned.answers["collect"] = lambda request: response(
        "collected", records.scope, records.collection
    )
    owned.answers["disarm"] = lambda request: response(
        "disarmed",
        records.scope,
        {
            "run_id": records.scope.run_id,
            "host_clock": records.collection["host_clock"],
        },
    )
    owned.answers["status"] = lambda request: response(
        "status",
        records.scope,
        {
            "run_id": records.scope.run_id,
            "host_clock": records.collection["host_clock"],
        },
    )
    return owned


EXCHANGE_KEYS = {
    "kind",
    "sent_ns",
    "received_ns",
    "monotonic_ns",
    "realtime_ns",
    "runner_sent_realtime_ns",
    "runner_received_realtime_ns",
}


def test_arm_runs_one_session_with_clock_pings_and_binds_the_receipt(
    owned: SimpleNamespace,
) -> None:
    receipt = owned.witness.arm()
    assert receipt == owned.records.receipt
    assert owned.witness.receipt is receipt and owned.witness.armed_at is not None
    program, digest = probe_program(controller.ROLE)
    assert owned.witness.bundle_sha256 == digest
    assert owned.regional.calls == [], "arm never takes the one-shot exec"
    (session,) = owned.sessions
    assert session.program == program and session.closed is True
    scope = owned.records.scope.digest()
    request, *pings, end = session.written
    assert request["kind"] == "arm" and request["scope_sha256"] == scope
    assert request["payload"]["scope"] == owned.records.scope.model_dump(mode="json")
    assert request["payload"]["lifetime_seconds"] == 3600
    assert type(request["payload"]["runner_clock"]["monotonic_ns"]) is int
    assert pings == [
        {"kind": "ping", "scope_sha256": scope, "payload": {"sequence": index}}
        for index in range(controller.PING_COUNT)
    ]
    assert end == {"kind": "end", "scope_sha256": scope, "payload": {}}
    assert owned.checks == ["uid", "uid"], "identity is pinned before and after"
    exchange, *samples = owned.witness.exchanges
    assert set(exchange) == EXCHANGE_KEYS and exchange["kind"] == "arm"
    assert (
        exchange["monotonic_ns"] == owned.records.receipt["host_clock"]["monotonic_ns"]
    )
    assert exchange["realtime_ns"] == owned.records.receipt["host_clock"]["realtime_ns"]
    assert [item["kind"] for item in samples] == ["arm-ping"] * controller.PING_COUNT
    assert [item["sequence"] for item in samples] == list(range(controller.PING_COUNT))
    assert all(set(item) == EXCHANGE_KEYS | {"sequence"} for item in samples), samples
    stamps = [(item["sent_ns"], item["received_ns"]) for item in [exchange, *samples]]
    assert all(sent <= received for sent, received in stamps), stamps
    assert all(
        later[0] >= earlier[1]
        for earlier, later in zip(stamps, stamps[1:], strict=False)
    ), "every sample's round trip starts after the previous one ended"
    hosts = [item["monotonic_ns"] for item in [exchange, *samples]]
    assert hosts == sorted(set(hosts)), "the host clock only moves forward"
    assert owned.witness.clock_sessions == [
        {
            "kind": "arm",
            "pings": controller.PING_COUNT,
            "stopped": "complete",
            "returncode": 0,
        }
    ]
    with pytest.raises(BoundaryDenied, match="armed twice"):
        owned.witness.arm()
    assert len(owned.sessions) == 1, "a second arm opens no session"


def test_disarm_and_status_stay_one_shot_execs_of_the_pinned_bundle(
    owned: SimpleNamespace,
) -> None:
    owned.witness.arm()
    owned.witness.disarm()
    assert owned.witness.status()["run_id"] == owned.records.scope.run_id
    program, _ = probe_program(controller.ROLE)
    assert len(owned.sessions) == 1, "only arm and collect open a session"
    assert [item["kind"] for item in owned.regional.requests] == ["disarm", "status"]
    for plane, args, timeout in owned.regional.calls:
        assert plane == "gpu" and timeout == controller.REQUEST_TIMEOUT_SECONDS
        assert args == (
            "exec",
            "-i",
            "probe-pod",
            "-c",
            "probe",
            "--",
            "chroot",
            "/host",
            controller.HOST_PYTHON,
            "-I",
            "-u",
            "-c",
            stdin_loader(program),
        )
    assert owned.checks == ["uid"] * 6, "identity is pinned around every request"
    assert len(owned.witness.exchanges) == controller.PING_COUNT + 1
    assert len(owned.witness.clock_sessions) == 1, "a one-shot exec is no sample"
    owned.answers["status"] = lambda request: "\n"
    with pytest.raises(BoundaryDenied, match="status returned no response"):
        owned.witness.status()


def test_the_default_session_is_a_verified_exec_of_the_pinned_bundle(
    owned: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[dict[str, Any]] = []

    def open_executor_stream(**kwargs: Any) -> FakeStream:
        opened.append(kwargs)
        stream = FakeStream(owned, kwargs["program"])
        owned.sessions.append(stream)
        return stream

    monkeypatch.setattr(controller, "open_executor_stream", open_executor_stream)
    settings = SimpleNamespace(
        kubeconfig=Path("gpu-kubeconfig"),
        context="gpu-context",
        namespace="gpu-fault-acceptance",
    )
    probe: Any = SimpleNamespace(
        pod="probe-pod", settings=settings, _check_target=lambda: None
    )
    witness = controller.DetachedResetWitness(
        owned.regional, probe, owned.records.scope, lifetime_seconds=3600
    )
    assert witness.arm() == owned.records.receipt
    program, _ = probe_program(controller.ROLE)
    assert opened == [
        {
            "kubeconfig": "gpu-kubeconfig",
            "context": "gpu-context",
            "namespace": "gpu-fault-acceptance",
            "pod": "probe-pod",
            "container": "probe",
            "chroot": "/host",
            "python": controller.HOST_PYTHON,
            "program": program,
        }
    ]
    assert owned.regional.calls == [], "the session replaces the one-shot exec"
    assert witness.clock_sessions[0]["stopped"] == "complete"


@pytest.mark.parametrize(
    ("defect", "expected"),
    [
        ("refused", "arm refused: host boot id differs"),
        ("scope", "out of scope"),
        ("kind", "out of scope"),
        ("clock", "no clock binding"),
        ("garbage", "not a protocol message"),
        ("empty", "returned no response"),
        ("shape", "response is unbound"),
        ("receipt-boot", "arm receipt is unbound"),
        ("program", "does not match the delivered bundle"),
        ("lifetime", "lifetime differs"),
    ],
)
def test_arm_refuses_refused_unbound_or_foreign_responses(
    owned: SimpleNamespace, defect: str, expected: str
) -> None:
    receipt = owned.records.receipt
    scope = owned.records.scope
    if defect == "refused":
        owned.answers["arm"] = lambda request: json.dumps(
            {
                "kind": "refused",
                "scope_sha256": scope.digest(),
                "payload": {"reason": "host boot id differs from the scope"},
            }
        )
    elif defect == "scope":
        owned.answers["arm"] = lambda request: json.dumps(
            {"kind": "armed", "scope_sha256": "0" * 64, "payload": receipt}
        )
    elif defect == "kind":
        owned.answers["arm"] = lambda request: response("collected", scope, receipt)
    elif defect == "clock":
        owned.answers["arm"] = lambda request: response(
            "armed", scope, {**receipt, "host_clock": None}
        )
    elif defect == "garbage":
        owned.answers["arm"] = lambda request: "not json at all"
    elif defect == "empty":
        owned.answers["arm"] = lambda request: "\n\n"
    elif defect == "shape":
        owned.answers["arm"] = lambda request: json.dumps({"kind": "armed"})
    elif defect == "receipt-boot":
        receipt["boot_id"] = "another-boot"
    elif defect == "program":
        receipt["program_sha256"] = "0" * 64
    else:
        receipt["lifetime_seconds"] = 1800
    with pytest.raises(BoundaryDenied, match=expected):
        owned.witness.arm()
    assert owned.witness.receipt is None and owned.witness.armed_at is None
    (session,) = owned.sessions
    assert session.closed is True, "the session is closed on every refusal"
    if defect in {"refused", "scope", "kind", "clock", "garbage", "empty", "shape"}:
        assert owned.witness.exchanges == [], "a refused arm records no clock exchange"
        assert owned.witness.clock_sessions == [], "nor its session"
        assert [item["kind"] for item in session.written] == ["arm"], (
            "a refused or unbound response is never pinged"
        )


@pytest.mark.parametrize(
    ("defect", "expected"),
    [
        ("silent", "arm returned no response"),
        ("oversized", "arm output exceeds its bound"),
        ("diagnostics", "arm output exceeds its bound"),
        ("exit", "arm session exited with status 1"),
        ("target", "probe Pod was replaced"),
    ],
)
def test_a_session_that_is_not_bound_end_to_end_records_nothing(
    owned: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, defect: str, expected: str
) -> None:
    if defect == "silent":
        monkeypatch.setattr(controller, "REQUEST_TIMEOUT_SECONDS", 0.05)
        owned.answers["arm"] = lambda request: None
    elif defect == "oversized":
        owned.answers["arm"] = lambda request: "x" * (controller.MAX_MESSAGE_BYTES + 1)
    elif defect == "diagnostics":
        owned.stderr = "e" * (controller.MAX_MESSAGE_BYTES + 1)
    elif defect == "exit":

        def failed(stream: FakeStream) -> None:
            stream.open, stream.code = False, 1

        owned.end = failed
    else:
        checks: list[str] = []

        def replaced() -> None:
            checks.append("uid")
            if len(checks) > 1:
                raise BoundaryDenied("probe Pod was replaced during the request")

        owned.witness.probe = SimpleNamespace(pod="probe-pod", _check_target=replaced)
    with pytest.raises(BoundaryDenied, match=expected):
        owned.witness.arm()
    assert owned.witness.receipt is None and owned.witness.exchanges == []
    assert owned.witness.clock_sessions == [], "nothing of the session is recorded"
    (session,) = owned.sessions
    assert session.closed is True, "the session is closed on every refusal"


@pytest.mark.parametrize(
    ("defect", "expected"),
    [
        ("refused", "ping refused: witness ping is unbound or out of sequence"),
        ("scope", "ping is unbound"),
        ("sequence", "ping is unbound"),
        ("bool-sequence", "ping is unbound"),
        ("kind", "ping is unbound"),
        ("shape", "ping is unbound"),
        ("clock", "ping is unbound"),
        ("zero-clock", "ping is unbound"),
        ("garbage", "ping is unbound"),
        ("oversized", "output exceeds its bound"),
        ("unordered", "ping is unordered"),
    ],
)
def test_an_unbound_or_unordered_ping_refuses_the_whole_exchange(
    owned: SimpleNamespace, defect: str, expected: str
) -> None:
    def pong(stream: FakeStream, message: dict[str, Any]) -> str:
        reply = owned.pong_message(stream, message)
        payload = reply["payload"]
        if payload["sequence"] != 1:
            return json.dumps(reply)
        if defect == "refused":
            reply = {
                "kind": "refused",
                "scope_sha256": reply["scope_sha256"],
                "payload": {"reason": "witness ping is unbound or out of sequence"},
            }
        elif defect == "scope":
            reply["scope_sha256"] = "0" * 64
        elif defect == "sequence":
            payload["sequence"] = 2
        elif defect == "bool-sequence":
            payload["sequence"] = True
        elif defect == "kind":
            reply["kind"] = "armed"
        elif defect == "shape":
            payload["extra"] = None
        elif defect == "clock":
            payload["host_clock"] = {"monotonic_ns": 1.5, "realtime_ns": 1}
        elif defect == "zero-clock":
            payload["host_clock"]["monotonic_ns"] = 0
        elif defect == "garbage":
            return "not json at all"
        elif defect == "oversized":
            return "x" * (controller.MAX_MESSAGE_BYTES + 1)
        else:
            payload["host_clock"]["monotonic_ns"] -= 2 * SECOND
        return json.dumps(reply)

    owned.pong = pong
    with pytest.raises(BoundaryDenied, match=expected):
        owned.witness.arm()
    assert owned.witness.receipt is None and owned.witness.exchanges == [], (
        "one unbound ping refuses the response and every sample with it"
    )
    assert owned.witness.clock_sessions == []
    (session,) = owned.sessions
    assert session.closed is True, "the session is closed on every refusal"
    assert [item["kind"] for item in session.written] == ["arm", "ping", "ping"], (
        "no ping follows an unbound pong"
    )


@pytest.mark.parametrize(
    ("behaviour", "pings", "stopped", "returncode"),
    [
        ("silent", 0, "budget", 0),
        ("no-budget", 0, "budget", 0),
        ("exits", 3, "closed", 0),
        ("vanishes", 1, "closed", 0),
        ("lost", 1, "closed", None),
        ("old-probe", 0, "closed", 0),
        ("no-exit", controller.PING_COUNT, "complete", None),
        ("no-status", controller.PING_COUNT, "complete", None),
    ],
)
def test_a_short_or_missing_ping_series_keeps_the_bound_exchange(
    owned: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    behaviour: str,
    pings: int,
    stopped: str,
    returncode: int | None,
) -> None:
    def lost(stream: FakeStream) -> None:
        raise ConnectionResetError("exec stream lost")

    def pong(stream: FakeStream, message: dict[str, Any]) -> str | None:
        sequence = message["payload"]["sequence"]
        if behaviour == "vanishes" and sequence == 1:
            stream.open = False
            return None
        if behaviour == "lost" and sequence == 1:
            owned.end = lost
            lost(stream)
        line = json.dumps(owned.pong_message(stream, message))
        if behaviour == "exits" and sequence == 2:
            stream.open = False
        return line

    def no_status(stream: FakeStream) -> None:
        stream.open, stream.code = False, ValueError("no status on the error channel")

    owned.pong = pong
    if behaviour == "silent":
        monkeypatch.setattr(controller, "PING_BUDGET_SECONDS", 0.05)
        owned.pong = lambda stream, message: None
    elif behaviour == "no-budget":
        monkeypatch.setattr(controller, "PING_BUDGET_SECONDS", 0)
    elif behaviour == "old-probe":
        owned.answers_pings = False
    elif behaviour == "no-exit":
        monkeypatch.setattr(controller, "SESSION_CLOSE_SECONDS", 0.05)
        owned.end = lambda stream: None
    elif behaviour == "no-status":
        owned.end = no_status
    assert owned.witness.arm() == owned.records.receipt, "the exchange stays bound"
    assert [item["kind"] for item in owned.witness.exchanges] == [
        "arm",
        *["arm-ping"] * pings,
    ]
    assert owned.witness.clock_sessions == [
        {"kind": "arm", "pings": pings, "stopped": stopped, "returncode": returncode}
    ]
    (session,) = owned.sessions
    assert session.closed is True, "the session is always closed"


def test_collect_binds_the_records_to_the_arm_scope(owned: SimpleNamespace) -> None:
    with pytest.raises(BoundaryDenied, match="never armed"):
        owned.witness.collect()
    assert owned.sessions == [], "an unarmed witness is never execd"
    owned.witness.arm()
    collection = owned.witness.collect()
    assert collection == owned.records.collection
    assert owned.witness.collection is collection
    count = controller.PING_COUNT
    assert [item["kind"] for item in owned.witness.exchanges] == [
        "arm",
        *["arm-ping"] * count,
        "collect",
        *["collect-ping"] * count,
    ]
    assert owned.sessions[-1].written[0] == {
        "kind": "collect",
        "scope_sha256": owned.records.scope.digest(),
        "payload": {"run_id": owned.records.scope.run_id},
    }
    host_clock = owned.records.collection["host_clock"]
    assert (
        owned.witness.exchanges[count + 1]["monotonic_ns"]
        == (host_clock["monotonic_ns"])
    )
    assert owned.witness.clock_sessions == [
        {"kind": kind, "pings": count, "stopped": "complete", "returncode": 0}
        for kind in ("arm", "collect")
    ]
    owned.witness.disarm()
    owned.witness.status()
    assert len(owned.witness.exchanges) == 2 * (count + 1), (
        "disarm and status never extend the clock calibration"
    )
    evidence = owned.witness.evidence()
    assert evidence["receipt"] == owned.records.receipt
    assert evidence["collection"] == owned.records.collection
    assert evidence["lifetime_seconds"] == 3600
    assert evidence["bundle_sha256"] == owned.witness.bundle_sha256
    assert evidence["exchanges"] == owned.witness.exchanges
    assert evidence["clock_sessions"] == owned.witness.clock_sessions


@pytest.mark.parametrize(
    ("defect", "expected"),
    [
        ("boot", "host boot id at collection differs"),
        ("state-boot", "not this scope's"),
        ("armed", "armed record changed"),
        ("unit", "another unit"),
        ("run", "names another run"),
        ("refused", "collect refused: host boot id changed"),
    ],
)
def test_collect_refuses_a_changed_boot_id_or_unbound_records(
    owned: SimpleNamespace, defect: str, expected: str
) -> None:
    owned.witness.arm()
    collection = owned.records.collection
    if defect == "boot":
        collection["boot_id"] = "rebooted"
    elif defect == "state-boot":
        collection["state"]["boot_id"] = "rebooted"
    elif defect == "armed":
        collection["armed"]["witness_id"] = "another"
    elif defect == "unit":
        collection["unit"] = "gpu-fault-destr015-witness-ffffffffffffffff.service"
    elif defect == "run":
        collection["run_id"] = "another-run"
    else:
        owned.answers["collect"] = lambda request: json.dumps(
            {
                "kind": "refused",
                "scope_sha256": owned.records.scope.digest(),
                "payload": {
                    "reason": "host boot id changed since the witness was armed; "
                    "its records are not this boot's"
                },
            }
        )
    with pytest.raises(BoundaryDenied, match=expected):
        owned.witness.collect()
    assert owned.witness.collection is None


def test_source_drift_between_requests_is_refused(
    owned: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(controller, "probe_program", lambda role: ("changed", "0" * 64))
    with pytest.raises(BoundaryDenied, match="source changed"):
        owned.witness.arm()
    assert owned.regional.calls == [], "changed source must not be dispatched"
    assert owned.sessions == [], "nor opened as a session"


def test_witness_lifetime_is_bounded_by_the_window_and_the_case_budgets() -> None:
    assert (
        controller.witness_lifetime_seconds(
            now=NOW, maintenance_end=NOW + timedelta(hours=3)
        )
        == controller.MAX_LIFETIME_SECONDS
    )
    assert (
        controller.witness_lifetime_seconds(
            now=NOW, maintenance_end=NOW + timedelta(seconds=2400)
        )
        == 2400
    )
    with pytest.raises(RegionalFixtureError, match="needs at least"):
        controller.witness_lifetime_seconds(
            now=NOW, maintenance_end=NOW + timedelta(seconds=1700)
        )
    with pytest.raises(RegionalFixtureError, match="aware"):
        controller.witness_lifetime_seconds(
            now=NOW, maintenance_end=datetime(2026, 9, 15, 15)
        )


@pytest.mark.parametrize(
    ("defect", "expected"),
    [
        ("node", "another run or node"),
        ("boot", "boot id differs"),
        ("unit", "not a DESTR-015 witness unit"),
        ("state-dir", "outside the acceptance state"),
        ("no-clock", "no host clock exchange"),
        ("no-armed", "no armed record"),
        ("armed-scope", "not bound to this scope"),
        ("armed-boot", "not written on the scope's boot"),
        ("armed-unit", "names another unit"),
        ("armed-clock", "between the arm request and its response"),
    ],
)
def test_arm_receipt_errors_name_every_missing_binding(
    defect: str, expected: str
) -> None:
    records = witness_records()
    receipt = records.receipt
    if defect == "node":
        receipt["node"] = "node-z"
    elif defect == "boot":
        receipt["boot_id"] = "rebooted"
    elif defect == "unit":
        receipt["unit"] = "gpu-fault-destr016-holder-0000.service"
    elif defect == "state-dir":
        receipt["state_dir"] = "/tmp/elsewhere"
    elif defect == "no-clock":
        receipt["host_clock"] = {}
    elif defect == "no-armed":
        receipt["armed"] = {"record": "final"}
    elif defect == "armed-scope":
        receipt["armed"]["scope_sha256"] = "0" * 64
    elif defect == "armed-boot":
        receipt["armed"]["start"]["tracee"]["boot_id"] = "rebooted"
    elif defect == "armed-unit":
        receipt["armed"]["unit"] = "gpu-fault-destr015-witness-ffffffffffffffff.service"
    else:
        receipt["armed"]["monotonic_ns"] = receipt["host_clock"]["monotonic_ns"] + 1
    errors = controller.arm_receipt_errors(
        receipt, scope=records.scope, lifetime_seconds=3600, program_sha256="f" * 64
    )
    assert any(expected in item for item in errors), (defect, errors)
    assert (
        controller.arm_receipt_errors(
            witness_records().receipt,
            scope=records.scope,
            lifetime_seconds=3600,
            program_sha256="f" * 64,
        )
        == []
    )


# --------------------------------------------------------------------------- #
# Runner flow
# --------------------------------------------------------------------------- #
def _names(harness: BranchHarness) -> list[str]:
    return [name for name, _ in harness.calls]


def _indexes(harness: BranchHarness, name: str) -> list[int]:
    return [index for index, (item, _) in enumerate(harness.calls) if item == name]


def _node_of(detail: Any) -> Any:
    return detail.get("node") if isinstance(detail, dict) else detail


def test_witnesses_are_armed_before_the_injection_and_collected_after_both_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = BranchHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.pending_reads = 1
    code, report = h.execute(tmp_path)
    assert code == 0 and report["verdict"] == "PASS", report
    assert report["errors"] == report["cleanup"]["errors"] == [], report
    names = _names(h)
    arms, writes = _indexes(h, "witness.arm"), _indexes(h, "probe.write-xid46")
    waits, collects = _indexes(h, "node.wait_ready"), _indexes(h, "witness.collect")
    assert len(arms) == 2 and len(writes) == 2 and max(arms) < min(writes), names
    assert len(waits) >= 2 and max(writes) < min(waits) < min(collects), names
    assert len(collects) == 2 and names.count("witness.disarm") == 2, names
    for node in NODES:
        disarm = next(
            i for i, (n, d) in enumerate(h.calls) if n == "witness.disarm" and d == node
        )
        cleanup = next(
            i for i, (n, d) in enumerate(h.calls) if n == "probe.cleanup" and d == node
        )
        assert disarm < cleanup, "the witness is stopped before its probe Pod goes"
    case_dir = tmp_path / "cases" / case.CASE_ID
    for node in NODES:
        start = json.loads(
            (case_dir / f"physical-witness-start-{node}.json").read_text()
        )
        assert start["receipt"]["armed"]["record"] == "armed"
        assert start["lifetime_seconds"] >= controller.MIN_LIFETIME_SECONDS
        assert [item["kind"] for item in start["exchanges"]] == ["arm"]
    intervals = json.loads((case_dir / "physical-reset-intervals.json").read_text())
    assert set(intervals) == set(NODES)
    for node in NODES:
        assert intervals[node]["clock_binding"]["bounded"] is True, intervals[node]
        assert intervals[node]["collection"]["final"]["reason"] == "finish-request"
        assert [item["kind"] for item in intervals[node]["exchanges"]] == [
            "arm",
            "collect",
        ]
        calibration = intervals[node]["clock_calibration"]
        assert calibration == verdicts.clock_calibration(intervals[node]["exchanges"])
        assert calibration["samples"] == 2 and calibration["min_round_trip_ms"] > 0
    assert report["clock_calibration"] == {
        node: intervals[node]["clock_calibration"] for node in NODES
    }, "the verdict JSON itself carries each node's calibration summary"
    returned = json.loads((case_dir / "hosts-return.json").read_text())
    assert all(returned[node]["exec_allowed"] is True for node in NODES), returned
    assert report["cleanup"][f"witness_disarm:{NODES[0]}"]["state_present"] is True


def test_the_verdict_keeps_only_each_nodes_calibration_summary() -> None:
    summary = {"samples": 18, "min_round_trip_ms": 20.0, "offset_uncertainty_ms": 70.5}
    intervals = {
        NODES[0]: {"exchanges": [{"kind": "arm"}], "clock_calibration": summary},
        NODES[1]: {"exchanges": []},
    }
    assert controller.verdict_clock_calibration(intervals) == {
        NODES[0]: summary,
        NODES[1]: None,
    }, "the exchanges stay in physical-reset-intervals.json"


def test_a_witness_that_cannot_be_placed_refuses_before_the_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = BranchHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.arm_refusal[NODES[0]] = "approved GPU is not present on this host"
    code, report = h.execute(tmp_path)
    assert code == 1 and report["verdict"] == "FAIL", report
    assert "refusing before the injection" in report["error"], report
    assert "approved GPU is not present on this host" in report["error"], report
    assert not h.injected, "no fault may be written without both witnesses armed"
    names = _names(h)
    assert "probe.write-xid46" not in names
    assert names.count("witness.arm") == 1, names
    assert "witness.status" in names, (
        "the host's own account is read while kubelet answers"
    )
    case_dir = tmp_path / "cases" / case.CASE_ID
    refusal = json.loads(
        (case_dir / f"physical-witness-arm-refusal-{NODES[0]}.json").read_text()
    )
    assert (
        refusal["status"]["state"]["refusal"]
        == "approved GPU is not present on this host"
    )
    assert not (case_dir / f"physical-witness-start-{NODES[0]}.json").exists(), (
        "a refused arm must not publish a start receipt"
    )
    assert names.count("probe.cleanup") == 2 and names.count("witness.disarm") == 1, (
        names
    )
    assert report["cleanup"]["errors"] == [], report["cleanup"]


def test_a_maintenance_window_too_short_for_the_witness_refuses_before_any_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = BranchHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    code, report = h.execute(tmp_path, seconds=1700)
    assert code == 1 and "needs at least" in report["error"], report
    names = _names(h)
    assert "witness.arm" not in names and not h.injected, names
    assert names.count("probe.cleanup") == 2, names


def test_no_exec_reaches_a_node_between_the_injection_and_both_nodes_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = BranchHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.outage_seconds = 300
    h.pending_reads = 1
    code, report = h.execute(tmp_path)
    assert code == 0 and report["verdict"] == "PASS", report
    names = _names(h)
    last_write = max(_indexes(h, "probe.write-xid46"))
    first_wait = min(_indexes(h, "node.wait_ready"))
    between = [
        name
        for name in names[last_write + 1 : first_wait]
        if name.startswith(("probe.", "witness."))
    ]
    assert between == [], f"exec reached a quiesced node: {between}"
    case_dir = tmp_path / "cases" / case.CASE_ID
    returned = json.loads((case_dir / "hosts-return.json").read_text())
    for node in NODES:
        assert returned[node]["exec_allowed"] is True, returned
        assert returned[node]["node_ready"] is True, returned
    assert names.count("probe.create") == 4, (
        "both probe Pods are confirmed after the outage"
    )
    creates = _indexes(h, "probe.create")
    first_read = min(
        i
        for i, (n, d) in enumerate(h.calls)
        if n == "probe.snapshot" and "--since-epoch" in d["args"]
    )
    assert max(creates) < first_read, "no host read before both Pods are confirmed"
    assert names.count("witness.collect") == 2, names
    assert report["cleanup"]["errors"] == [], report["cleanup"]


def test_a_node_that_never_returns_loses_its_witness_and_is_never_execd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = BranchHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.permanent_outage = {NODES[1]}
    code, report = h.execute(tmp_path)
    assert code == 1 and report["verdict"] == "FAIL", report
    assert "did not return Ready" in report["error"] and NODES[1] in report["error"], (
        report["error"]
    )
    assert "treated as lost" in report["error"], report["error"]
    names = _names(h)
    assert "witness.collect" not in names, names
    assert not [
        d for n, d in h.calls if n == "probe.snapshot" and "--since-epoch" in d["args"]
    ], "no host may be read while a node is NotReady"
    last_write = max(_indexes(h, "probe.write-xid46"))
    lost_node_execs = [
        n
        for n, d in h.calls[last_write + 1 :]
        if n.startswith(("probe.", "witness.")) and _node_of(d) == NODES[1]
    ]
    assert lost_node_execs == [], lost_node_execs
    assert [d for n, d in h.calls if n == "witness.disarm"] == [NODES[0]], names
    assert [d for n, d in h.calls if n == "probe.cleanup"] == [NODES[0]], names
    cleanup = report["cleanup"]
    assert cleanup[f"host_reachability:{NODES[0]}"]["exec_allowed"] is True
    assert cleanup[f"host_reachability:{NODES[1]}"]["exec_allowed"] is False
    assert cleanup[f"witness_disarm:{NODES[1]}"]["disarmed"] == "unknown"
    assert cleanup[f"probe_cleanup_{NODES[1]}"]["deferred"] is True
    assert any(
        "never answered again" in item and NODES[1] in item
        for item in cleanup["errors"]
    ), cleanup["errors"]
    assert "prewarm.cleanup" in names and "runtime.verify" in names, names


@pytest.mark.parametrize("cause", ["reboot", "lifetime"])
def test_cleanup_assumes_a_lost_witness_only_on_a_new_boot_or_an_expired_lifetime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cause: str
) -> None:
    h = BranchHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.permanent_outage = {NODES[1]}
    if cause == "reboot":
        h.reboot_during_outage = {NODES[1]}
    else:
        h.advance_at["node.wait_ready"] = 2 * controller.MAX_LIFETIME_SECONDS + 200
    code, report = h.execute(tmp_path)
    assert code == 1 and report["verdict"] == "FAIL", report
    decision = report["cleanup"][f"witness_disarm:{NODES[1]}"]
    assert decision["disarmed"] == "assumed", decision
    assert ("boot id changed" if cause == "reboot" else "lifetime") in decision[
        "reason"
    ]
    assert [d for n, d in h.calls if n == "probe.cleanup"] == [NODES[0]]
    assert [d for n, d in h.calls if n == "witness.disarm"] == [NODES[0]]
    assert report["cleanup"][f"probe_cleanup_{NODES[1]}"]["deferred"] is True
