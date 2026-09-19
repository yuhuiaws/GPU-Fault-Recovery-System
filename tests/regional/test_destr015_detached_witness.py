"""Controller, verdict and runner-flow contracts of the detached DESTR-015 witnesses.

The controller talks to the host through one-shot ``kubectl exec`` requests
(faked at the ``regional.kubectl`` boundary), the verdict functions read only
the durable records, and the runner flow is driven end to end through the
branch harness, whose probes and witnesses refuse exactly where a NotReady node
would refuse a real exec.
"""

from __future__ import annotations

import json
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


@pytest.fixture
def owned() -> SimpleNamespace:
    records = witness_records()
    checks: list[str] = []
    probe = SimpleNamespace(pod="probe-pod", _check_target=lambda: checks.append("uid"))
    answers: dict[str, Any] = {}

    def respond(request: dict[str, Any]) -> str:
        return answers[request["kind"]](request)

    regional = FakeRegional(respond)
    witness = controller.DetachedResetWitness(
        regional, probe, records.scope, lifetime_seconds=3600
    )
    records.receipt["program_sha256"] = witness.bundle_sha256
    answers["arm"] = lambda request: response("armed", records.scope, records.receipt)
    answers["collect"] = lambda request: response(
        "collected", records.scope, records.collection
    )
    answers["disarm"] = lambda request: response(
        "disarmed",
        records.scope,
        {
            "run_id": records.scope.run_id,
            "host_clock": records.collection["host_clock"],
        },
    )
    answers["status"] = lambda request: response(
        "status",
        records.scope,
        {
            "run_id": records.scope.run_id,
            "host_clock": records.collection["host_clock"],
        },
    )
    return SimpleNamespace(
        witness=witness,
        regional=regional,
        records=records,
        checks=checks,
        answers=answers,
    )


def test_arm_delivers_the_pinned_bundle_and_binds_the_receipt(
    owned: SimpleNamespace,
) -> None:
    receipt = owned.witness.arm()
    assert receipt == owned.records.receipt
    assert owned.witness.receipt is receipt and owned.witness.armed_at is not None
    program, digest = probe_program(controller.ROLE)
    assert owned.witness.bundle_sha256 == digest
    plane, args, timeout = owned.regional.calls[0]
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
    request = owned.regional.requests[0]
    assert request["kind"] == "arm"
    assert request["scope_sha256"] == owned.records.scope.digest()
    assert request["payload"]["scope"] == owned.records.scope.model_dump(mode="json")
    assert request["payload"]["lifetime_seconds"] == 3600
    assert type(request["payload"]["runner_clock"]["monotonic_ns"]) is int
    assert owned.checks == ["uid", "uid"], "identity is pinned before and after"
    (exchange,) = owned.witness.exchanges
    assert exchange["kind"] == "arm"
    assert exchange["sent_ns"] <= exchange["received_ns"]
    assert (
        exchange["monotonic_ns"] == owned.records.receipt["host_clock"]["monotonic_ns"]
    )
    assert exchange["realtime_ns"] == owned.records.receipt["host_clock"]["realtime_ns"]
    with pytest.raises(BoundaryDenied, match="armed twice"):
        owned.witness.arm()


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
    if defect in {"refused", "scope", "kind", "clock", "garbage", "empty", "shape"}:
        assert owned.witness.exchanges == [], "a refused arm records no clock exchange"


def test_collect_binds_the_records_to_the_arm_scope(owned: SimpleNamespace) -> None:
    with pytest.raises(BoundaryDenied, match="never armed"):
        owned.witness.collect()
    owned.witness.arm()
    collection = owned.witness.collect()
    assert collection == owned.records.collection
    assert owned.witness.collection is collection
    assert [item["kind"] for item in owned.witness.exchanges] == ["arm", "collect"]
    assert owned.regional.requests[-1] == {
        "kind": "collect",
        "scope_sha256": owned.records.scope.digest(),
        "payload": {"run_id": owned.records.scope.run_id},
    }
    owned.witness.disarm()
    owned.witness.status()
    assert [item["kind"] for item in owned.witness.exchanges] == ["arm", "collect"], (
        "disarm and status never extend the clock calibration"
    )
    evidence = owned.witness.evidence()
    assert evidence["receipt"] == owned.records.receipt
    assert evidence["collection"] == owned.records.collection
    assert evidence["lifetime_seconds"] == 3600
    assert evidence["bundle_sha256"] == owned.witness.bundle_sha256


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
    returned = json.loads((case_dir / "hosts-return.json").read_text())
    assert all(returned[node]["exec_allowed"] is True for node in NODES), returned
    assert report["cleanup"][f"witness_disarm:{NODES[0]}"]["state_present"] is True


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
