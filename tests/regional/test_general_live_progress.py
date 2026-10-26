"""GF-GENERAL-LIVE-027 training-progress injector/verdict contract over fakes."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_progress as pr
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from scripts.e2e.regional.general_live_registry import build_registry, coverage_rows
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
    node_probe,
    store_poll_clock,
)

SPECS = {spec.variant: spec for spec in pr.VARIANTS}
NODE = "ip-10-0-43-5.us-west-2.compute.internal"
CLEAN_SURVEY = {
    "residue": {},
    "collector_env": {"GPU_FAULT_HOST_INTERVAL_SECONDS": "15"},
    "tools": {},
}


class FakeProgressJob:
    created: list["FakeProgressJob"] = []

    def __init__(
        self, ctx: Any, *, name: str, ranks: tuple[int, ...] = (0,), gpu_count: int = 8
    ) -> None:
        self.ctx = ctx
        self.name = name
        self.attempt_id = f"{name}-a001"
        self.ranks = ranks
        self.writes: list[tuple[int, dict[str, Any]]] = []
        self.stopped: list[int] = []
        FakeProgressJob.created.append(self)

    def start(self) -> dict[str, Any]:
        return {}

    def identity(self) -> dict[str, Any]:
        return {
            "job_id": self.name,
            "attempt_id": self.attempt_id,
            "ranks": list(self.ranks),
        }

    def write_progress(self, rank: int, **fields: Any) -> dict[str, Any]:
        self.writes.append((rank, fields))
        return {}

    def stop_reporter(self, rank: int) -> dict[str, Any]:
        self.stopped.append(rank)
        return {}

    def delete(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _patch_job(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeProgressJob.created = []
    monkeypatch.setattr(pr, "ManagedProgressJob", FakeProgressJob)


def _settings(spec: Any, tmp_path: Path):
    base = make_settings(spec, node=NODE)
    (tmp_path / "site.yaml").write_text("site: test\n")
    return type(base)(**{**base.__dict__, "site_file": tmp_path / "site.yaml"})


class RecordingStore:
    """A ``cpu_python`` double that serves the training-records documents in
    order (the last one repeats), answers the scan POST, and counts the reads --
    so a test can say how many store reads a verdict took."""

    def __init__(self, *documents: dict[str, Any]) -> None:
        self.documents = list(documents)
        self.reads = 0
        self.scans = 0

    def __call__(self, script: str, *arguments: str) -> dict[str, Any]:
        if "training-health" in script:
            self.scans += 1
            return {"status": 200, "body": {}}
        document = self.documents[min(self.reads, len(self.documents) - 1)]
        self.reads += 1
        return copy.deepcopy(document)


def _ctx(spec: Any, tmp_path: Path, doc: Any):
    responder = doc if callable(doc) else (lambda script, *a: doc)
    regional = FakeRegional(cpu_python=responder)
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={},
        settings=_settings(spec, tmp_path),
    )
    node_probe(ctx, FakeProbe({"survey": CLEAN_SURVEY}))
    return ctx


def _incident(kind: str, state: str = "ESCALATED", reasons=()):
    return {
        "incident_id": f"inc-training-{kind}",
        "event_id": f"training-{kind}-deadbeef",
        "state": state,
        "attempt_id": "a",
        "workflow_request_id": "wf",
        "reasons": list(reasons),
    }


def _workflow(ops, *, status: str = "SUCCEEDED", executed: Any = None):
    """A scripted workflow record. As live, a record carries its ``status`` and
    one execution per step that ran; ``executed`` is the ``(operation, status)``
    list of those, every planned step SUCCEEDED when omitted."""

    if executed is None:
        executed = [(op, "SUCCEEDED") for op in ops]
    return {
        "wf": {
            "status": status,
            "official_steps": [{"operation": op, "parameters": {}} for op in ops],
            "step_executions": [
                {"operation": op, "status": state, "step_index": index}
                for index, (op, state) in enumerate(executed)
            ],
        }
    }


NAN_CHAIN = (
    "FREEZE_EVIDENCE",
    "STOP_WORKLOADS",
    "COLLECT_DIAGNOSTIC_BUNDLE",
    "ESCALATE_SUPPORT",
)


def _nan_doc(*, state: str, status: str, executed: Any) -> dict[str, Any]:
    return {
        "progress_states": [],
        "incidents": [_incident("nonfinite-loss", state=state)],
        "workflows": _workflow(NAN_CHAIN, status=status, executed=executed),
        "decision": {"status": "NO_ACTION"},
        "notifications": [],
    }


#: The live 2026-10-26 nan record 4 s after the heartbeat: FREEZE_EVIDENCE and
#: STOP_WORKLOADS done, the remote COLLECT_DIAGNOSTIC_BUNDLE waiting on the
#: node (it took 22 s), the workflow RUNNING, the incident ACTION_PENDING.
NAN_IN_FLIGHT = _nan_doc(
    state="ACTION_PENDING",
    status="RUNNING",
    executed=[
        ("FREEZE_EVIDENCE", "SUCCEEDED"),
        ("STOP_WORKLOADS", "SUCCEEDED"),
        ("COLLECT_DIAGNOSTIC_BUNDLE", "WAITING"),
    ],
)
#: The same record once ESCALATE_SUPPORT completed it.
NAN_SETTLED = _nan_doc(state="ESCALATED", status="SUCCEEDED", executed=None)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_registry_covers_every_027_variant() -> None:
    rows = coverage_rows()
    build_registry(pr.VARIANTS, rows=rows, categories=[27])
    assert {s.variant for s in pr.VARIANTS} == set(rows["GF-GENERAL-LIVE-027"])
    assert all(s.implemented for s in pr.VARIANTS), (
        "every 027 variant is native/implemented"
    )


# --------------------------------------------------------------------------- #
# pure verdict helpers
# --------------------------------------------------------------------------- #
def test_chain_errors_requires_ops_and_forbids_restart() -> None:
    incident = _incident("nonfinite-loss")
    ok = _workflow(("FREEZE_EVIDENCE", "STOP_WORKLOADS", "ESCALATE_SUPPORT"))["wf"]
    assert (
        pr._chain_errors(
            incident,
            ok,
            kind="nonfinite-loss",
            required_ops=("FREEZE_EVIDENCE", "STOP_WORKLOADS"),
            allow_stop=True,
            states=frozenset({"ESCALATED"}),
        )
        == []
    )
    # A hang chain that planned STOP_WORKLOADS is a FAIL (no STOP for a stall).
    bad = _workflow(("FREEZE_EVIDENCE", "STOP_WORKLOADS", "VALIDATE_FABRIC"))["wf"]
    errors = pr._chain_errors(
        incident,
        bad,
        kind="hang",
        required_ops=("FREEZE_EVIDENCE", "VALIDATE_FABRIC"),
        allow_stop=False,
        states=frozenset(),
    )
    assert any("forbidden" in item for item in errors), errors


def test_chain_errors_missing_incident() -> None:
    assert pr._chain_errors(
        None, {}, kind="hang", required_ops=(), allow_stop=False, states=frozenset()
    ) == ["no training_hang finding reached an incident"]


def test_no_restart_and_checkpoint_helpers() -> None:
    records = {
        "incidents": [_incident("nonfinite-loss")],
        "workflows": _workflow(("FREEZE_EVIDENCE", "RESTART_WORKLOAD")),
        "decision": {"status": "NO_ACTION"},
    }
    assert pr._no_restart_errors(records) == [
        "a training workflow planned RESTART_WORKLOAD"
    ]
    bad = {
        "official_steps": [
            {
                "operation": "STOP_WORKLOADS",
                "parameters": {"checkpoint_disposition": "RESTORE"},
            }
        ]
    }
    assert pr._checkpoint_errors(bad) == [
        "checkpoint disposition is RESTORE, not application-validated"
    ]
    good = {
        "official_steps": [
            {
                "operation": "STOP_WORKLOADS",
                "parameters": {
                    "checkpoint_disposition": "REQUIRES_APPLICATION_VALIDATION"
                },
            }
        ]
    }
    assert pr._checkpoint_errors(good) == []


def test_incident_matches_by_kind_prefix() -> None:
    records = {"incidents": [_incident("reporter-silent"), _incident("hang")]}
    assert pr._incident(records, "hang")["event_id"] == "training-hang-deadbeef"
    assert pr._incident(records, "straggler") is None


# --------------------------------------------------------------------------- #
# execute paths
# --------------------------------------------------------------------------- #
def test_nan_passes_and_stops_without_restart(tmp_path: Path) -> None:
    doc = {
        "progress_states": [],
        "incidents": [_incident("nonfinite-loss")],
        "workflows": _workflow(
            ("FREEZE_EVIDENCE", "STOP_WORKLOADS", "ESCALATE_SUPPORT")
        ),
        "decision": {"status": "NO_ACTION"},
        "notifications": [],
    }
    result = SPECS["nan"].execute(_ctx(SPECS["nan"], tmp_path, doc))
    assert result["verdict"] == "PASS", result["errors"]
    job = FakeProgressJob.created[-1]
    assert job.writes and job.writes[0][1].get("numerical_error") is True


def test_nan_fails_when_a_restart_is_planned(tmp_path: Path) -> None:
    doc = {
        "progress_states": [],
        "incidents": [_incident("nonfinite-loss")],
        "workflows": _workflow(
            (
                "FREEZE_EVIDENCE",
                "STOP_WORKLOADS",
                "RESTART_WORKLOAD",
                "ESCALATE_SUPPORT",
            )
        ),
        "decision": None,
        "notifications": [],
    }
    result = SPECS["nan"].execute(_ctx(SPECS["nan"], tmp_path, doc))
    assert result["verdict"] == "FAIL"
    assert any(
        "RESTART_WORKLOAD" in item or "forbidden" in item for item in result["errors"]
    ), result["errors"]


def test_reporter_silent_fails_if_graded_as_hang(tmp_path: Path) -> None:
    doc = {
        "progress_states": [],
        "incidents": [_incident("reporter-silent"), _incident("hang")],
        "workflows": _workflow(("FREEZE_EVIDENCE", "VALIDATE_FABRIC")),
        "decision": None,
        "notifications": [],
    }
    result = SPECS["reporter_silent"].execute(
        _ctx(SPECS["reporter_silent"], tmp_path, doc)
    )
    assert result["verdict"] == "FAIL"
    assert any("hang" in item for item in result["errors"]), result["errors"]


def test_old_attempt_passes_when_nothing_is_stopped(tmp_path: Path) -> None:
    doc = {
        "progress_states": [],
        "incidents": [],
        "workflows": {},
        "decision": None,
        "notifications": [],
    }
    result = SPECS["old_attempt"].execute(_ctx(SPECS["old_attempt"], tmp_path, doc))
    assert result["verdict"] == "PASS", result["errors"]


# --------------------------------------------------------------------------- #
# settle before judging a state
# --------------------------------------------------------------------------- #
def test_settle_budget_is_the_step_cap_plus_one_poll_per_planned_step() -> None:
    # execution/config.py defaults: 600-s per-step cap, 5-s dispatcher poll;
    # the four-step nan chain gets 600 + 4 x 5 + the 180-s chain margin.
    assert pr.settle_budget_seconds(NAN_SETTLED["workflows"]["wf"]) == 800
    assert pr.settle_budget_seconds({}) == 600 + 5 + 180


def test_settled_needs_the_workflow_terminal_and_the_state_out_of_flight() -> None:
    running = {"status": "RUNNING"}
    succeeded = {"status": "SUCCEEDED"}
    in_flight = _incident("nonfinite-loss", state="ACTION_PENDING")
    escalated = _incident("nonfinite-loss", state="ESCALATED")
    assert not pr.record_settled(in_flight, running), "mid-flight is not settled"
    # The workflow's terminal write and the incident's state travel together;
    # a read that shows only one of them is not settled yet.
    assert not pr.record_settled(in_flight, succeeded), "state still in flight"
    assert not pr.record_settled(escalated, running), "workflow still running"
    assert pr.record_settled(escalated, succeeded), "both terminal: settled"
    for status in ("FAILED", "BLOCKED", "SUPERSEDED"):
        assert pr.record_settled(escalated, {"status": status}), status
    # No workflow to wait for: the plan compiled to nothing.
    assert pr.record_settled({**escalated, "workflow_request_id": None}, {}), (
        "no workflow named: the state alone decides"
    )
    assert not pr.record_settled({**in_flight, "workflow_request_id": None}, {}), (
        "no workflow named but the state is still in flight"
    )


@pytest.mark.parametrize("variant", ["nan", "checkpoint_unverified"])
def test_nan_judges_escalated_on_the_settled_record_not_the_first_read(
    tmp_path: Path, variant: str
) -> None:
    # Live 2026-10-26 replay: the first two reads show the chain mid-flight,
    # the third shows it complete. The old oracle failed on the first.
    store = RecordingStore(NAN_IN_FLIGHT, NAN_IN_FLIGHT, NAN_SETTLED)
    ctx = _ctx(SPECS[variant], tmp_path, store)
    result = SPECS[variant].execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["stages"]["settled"] == []
    assert store.reads == 3, "the detection read, then two settle polls"
    settled = result["settled"]
    assert settled["settled"] is True
    assert settled["state"] == "ESCALATED"
    assert settled["workflow_status"] == "SUCCEEDED"
    assert settled["budget_seconds"] == 800
    assert settled["waited_seconds"] > 0
    assert settled["polls"] == 2
    assert settled["executed"] == [f"{op} SUCCEEDED" for op in NAN_CHAIN]
    name = "nan" if variant == "nan" else "checkpoint"
    written = json.loads((ctx.case_dir / f"{name}-settled.json").read_text())
    assert written["executed"] == settled["executed"]
    timeline = json.loads((ctx.case_dir / f"{name}-settle-timeline.json").read_text())
    assert [entry["accepted"] for entry in timeline["entries"]] == [False, True]


def test_nan_settled_on_the_first_read_waits_for_nothing(tmp_path: Path) -> None:
    store = RecordingStore(NAN_SETTLED)
    result = SPECS["nan"].execute(_ctx(SPECS["nan"], tmp_path, store))
    assert result["verdict"] == "PASS", result["errors"]
    assert store.reads == 1
    assert result["settled"]["polls"] == 0
    assert result["settled"]["settled"] is True


def test_nan_in_flight_past_the_budget_is_inconclusive_not_fail(tmp_path: Path) -> None:
    store = RecordingStore(NAN_IN_FLIGHT)
    ctx = store_poll_clock(_ctx(SPECS["nan"], tmp_path, store))
    with pytest.raises(VariantInconclusive) as caught:
        SPECS["nan"].execute(ctx)
    message = str(caught.value)
    assert "did not settle within 800 s" in message
    assert "state ACTION_PENDING" in message
    assert "workflow wf RUNNING" in message
    assert "COLLECT_DIAGNOSTIC_BUNDLE WAITING" in message
    assert "planned FREEZE_EVIDENCE, STOP_WORKLOADS, COLLECT_DIAGNOSTIC_BUNDLE" in (
        message
    )
    # One read every STORE_POLL_SECONDS for the whole budget, then the record.
    assert store.reads >= 1 + 800 // 20
    written = json.loads((ctx.case_dir / "nan-settled.json").read_text())
    assert written["settled"] is False
    assert written["waited_seconds"] >= 800


def test_nan_fails_when_the_chain_settled_without_running_to_its_end(
    tmp_path: Path,
) -> None:
    # The bundle step failed; the product's failure exit still reads ESCALATED,
    # but the planned ESCALATE_SUPPORT hand-off never ran.
    failed = _nan_doc(
        state="ESCALATED",
        status="FAILED",
        executed=[
            ("FREEZE_EVIDENCE", "SUCCEEDED"),
            ("STOP_WORKLOADS", "SUCCEEDED"),
            ("COLLECT_DIAGNOSTIC_BUNDLE", "FAILED"),
        ],
    )
    result = SPECS["nan"].execute(_ctx(SPECS["nan"], tmp_path, RecordingStore(failed)))
    assert result["verdict"] == "FAIL"
    assert result["stages"]["chain"] == [], "the planned chain and state still hold"
    assert result["stages"]["settled"] == [
        "training_nonfinite-loss workflow settled FAILED, not SUCCEEDED (executed "
        "FREEZE_EVIDENCE SUCCEEDED, STOP_WORKLOADS SUCCEEDED, "
        "COLLECT_DIAGNOSTIC_BUNDLE FAILED)"
    ]


def test_nan_without_a_finding_settles_nothing(tmp_path: Path) -> None:
    empty = {
        "progress_states": [],
        "incidents": [],
        "workflows": {},
        "decision": None,
        "notifications": [],
    }
    store = RecordingStore(empty)
    ctx = store_poll_clock(_ctx(SPECS["nan"], tmp_path, store))
    result = SPECS["nan"].execute(ctx)
    assert result["verdict"] == "FAIL"
    assert result["errors"] == [
        "chain: no training_nonfinite-loss finding reached an incident"
    ]
    # The detection wait ran out (one read per 20-s poll over its 600-s budget);
    # the settle added no read and no wait (two clock readings, nothing between).
    assert store.reads == 600 // 20
    settled = result["settled"]
    assert settled.pop("waited_seconds") <= 20
    assert settled == {
        "budget_seconds": 600 + 5 + 180,
        "polls": 0,
        "incident_id": None,
        "settled": False,
    }


def _hang_doc(*, state: str, status: str, executed: Any) -> dict[str, Any]:
    return {
        "progress_states": [],
        "incidents": [_incident("hang", state=state)],
        "workflows": _workflow(
            ("FREEZE_EVIDENCE", "VALIDATE_FABRIC"), status=status, executed=executed
        ),
        "decision": None,
        "notifications": [],
    }


HANG_IN_FLIGHT = _hang_doc(
    state="ACTION_PENDING",
    status="RUNNING",
    executed=[("FREEZE_EVIDENCE", "SUCCEEDED"), ("VALIDATE_FABRIC", "WAITING")],
)
HANG_RECOVERED = _hang_doc(state="RECOVERED", status="SUCCEEDED", executed=None)


def test_resumed_progress_reads_after_the_new_progress_and_waits_for_the_settled_record(
    tmp_path: Path,
) -> None:
    store = RecordingStore(HANG_IN_FLIGHT, HANG_IN_FLIGHT, HANG_RECOVERED)
    ctx = _ctx(SPECS["resumed_progress"], tmp_path, store)
    result = SPECS["resumed_progress"].execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    job = FakeProgressJob.created[-1]
    assert job.writes[-1] == (0, {"step": 11, "loss": 1.0})
    # The detection read, then a fresh read after the new progress and one poll.
    assert store.reads == 3
    assert result["settled"]["polls"] == 2
    assert result["settled"]["state"] == "RECOVERED"
    assert store.scans >= 1, "the stall is minted by the scan"


def test_resumed_progress_fails_when_the_stall_settled_elsewhere(
    tmp_path: Path,
) -> None:
    escalated = _hang_doc(state="ESCALATED", status="FAILED", executed=None)
    store = RecordingStore(HANG_IN_FLIGHT, escalated)
    result = SPECS["resumed_progress"].execute(
        _ctx(SPECS["resumed_progress"], tmp_path, store)
    )
    assert result["verdict"] == "FAIL"
    assert result["errors"] == [
        "recovered: the stall incident is ESCALATED, not RECOVERED"
    ]


def test_resumed_progress_in_flight_past_the_budget_is_inconclusive(
    tmp_path: Path,
) -> None:
    store = RecordingStore(HANG_IN_FLIGHT)
    ctx = store_poll_clock(_ctx(SPECS["resumed_progress"], tmp_path, store))
    with pytest.raises(VariantInconclusive, match="VALIDATE_FABRIC WAITING"):
        SPECS["resumed_progress"].execute(ctx)
    written = json.loads((ctx.case_dir / "resume-settled.json").read_text())
    assert written["settled"] is False
    assert written["budget_seconds"] == 600 + 2 * 5 + 180


@pytest.mark.parametrize(
    "variant", ["stalled_step", "reporter_silent", "step_regression", "straggler_group"]
)
def test_scan_variants_judge_the_planned_chain_on_the_first_read(
    tmp_path: Path, variant: str
) -> None:
    # Their VALIDATE_FABRIC legitimately stays in flight until newer progress
    # arrives; they assert no state and wait for no settle.
    kind = {
        "stalled_step": "hang",
        "reporter_silent": "reporter-silent",
        "step_regression": "step-regression",
        "straggler_group": "straggler",
    }[variant]
    doc = {
        "progress_states": [],
        "incidents": [_incident(kind, state="ACTION_PENDING")],
        "workflows": _workflow(
            ("FREEZE_EVIDENCE", "VALIDATE_FABRIC"),
            status="RUNNING",
            executed=[("FREEZE_EVIDENCE", "SUCCEEDED"), ("VALIDATE_FABRIC", "WAITING")],
        ),
        "decision": None,
        "notifications": [],
    }
    store = RecordingStore(doc)
    result = SPECS[variant].execute(_ctx(SPECS[variant], tmp_path, store))
    assert result["verdict"] == "PASS", result["errors"]
    assert "settled" not in result
    assert store.reads == 1
