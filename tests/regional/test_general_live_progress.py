"""GF-GENERAL-LIVE-027 training-progress injector/verdict contract over fakes."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_progress as pr
from scripts.e2e.regional.general_live_fixture import VariantInconclusive, VariantNotRun
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
PEER = "ip-10-0-32-162.us-west-2.compute.internal"
#: The real job class, kept before the autouse fixture swaps the fake in.
RealManagedProgressJob = pr.ManagedProgressJob
MANIFESTS = Path(pr.__file__).with_name("manifests") / "training"
RUNTIME_IMAGE = "111122223333.dkr.ecr.us-west-2.amazonaws.com/gpu-fault@sha256:" + (
    "e" * 64
)
CLEAN_SURVEY = {
    "residue": {},
    "collector_env": {"GPU_FAULT_HOST_INTERVAL_SECONDS": "15"},
    "tools": {},
}


class FakeProgressJob:
    created: list["FakeProgressJob"] = []
    #: What ``reporter_logs`` answers (the straggler diagnostic).
    logs: dict[str, str] = {}

    def __init__(
        self, ctx: Any, *, name: str, ranks: tuple[int, ...] = (0,), gpu_count: int = 8
    ) -> None:
        self.ctx = ctx
        self.name = name
        self.attempt_id = f"{name}-a001"
        self.ranks = ranks
        self.rank_nodes = {
            rank: (ctx.settings.node if rank == 0 else ctx.settings.peer_node or PEER)
            for rank in ranks
        }
        self.writes: list[tuple[int, dict[str, Any]]] = []
        self.stopped: list[int] = []
        self.log_reads = 0
        FakeProgressJob.created.append(self)

    def start(self) -> dict[str, Any]:
        return {}

    def identity(self) -> dict[str, Any]:
        return {
            "job_id": self.name,
            "attempt_id": self.attempt_id,
            "ranks": list(self.ranks),
            "nodes": [self.rank_nodes[rank] for rank in self.ranks],
        }

    def write_progress(self, rank: int, **fields: Any) -> dict[str, Any]:
        self.writes.append((rank, fields))
        return {}

    def stop_reporter(self, rank: int) -> str:
        self.stopped.append(rank)
        return "4242\n\tT (stopped)\n"  # STOP_REPORTER's print: PID, State

    def reporter_logs(self, *, tail: int = 50) -> dict[str, str]:
        self.log_reads += 1
        return dict(FakeProgressJob.logs)

    def delete(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _patch_job(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeProgressJob.created = []
    FakeProgressJob.logs = {}
    monkeypatch.setattr(pr, "ManagedProgressJob", FakeProgressJob)


def _settings(spec: Any, tmp_path: Path):
    base = make_settings(spec, node=NODE)
    (tmp_path / "site.yaml").write_text("site: test\n")
    return type(base)(**{**base.__dict__, "site_file": tmp_path / "site.yaml"})


class RecordingStore:
    """A ``cpu_python`` double that serves the training-records documents in
    order (the last one repeats), answers the scan POST, and counts the reads --
    so a test can say how many store reads a verdict took -- keeping the node
    list each records read was scoped to (``TRAINING_PROBE``'s ``argv[2]``)."""

    def __init__(self, *documents: dict[str, Any]) -> None:
        self.documents = list(documents)
        self.reads = 0
        self.scans = 0
        self.node_arguments: list[str] = []

    def __call__(self, script: str, *arguments: str) -> dict[str, Any]:
        if "training-health" in script:
            self.scans += 1
            return {"status": 200, "body": {}}
        assert script is pr.TRAINING_PROBE, "a records read runs the probe script"
        self.node_arguments.append(arguments[1])
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


def _incident(kind: str, state: str = "ESCALATED", reasons=(), **fields: Any):
    return {
        "incident_id": f"inc-training-{kind}",
        "event_id": f"training-{kind}-deadbeef",
        "state": state,
        "attempt_id": "a",
        "workflow_request_id": "wf",
        "reasons": list(reasons),
        **fields,
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


# --------------------------------------------------------------------------- #
# straggler_group: two ranks are two Pods on two nodes, both advancing
# --------------------------------------------------------------------------- #
def _manifest(name: str) -> dict[str, Any]:
    import yaml

    return yaml.safe_load((MANIFESTS / name).read_text(encoding="utf-8"))


def _templates(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        role: replica["template"]["spec"]
        for role, replica in document["spec"]["pytorchReplicaSpecs"].items()
    }


def test_two_rank_job_renders_one_reporter_per_pod_on_two_nodes() -> None:
    document = _manifest(pr.TWO_NODE_MANIFEST)
    pr.pin_replicas(document, {"Master": NODE, "Worker": PEER})
    pr.install_reporter(
        document, "gl27-strag", {"Master": (0,), "Worker": (1,)}, RUNTIME_IMAGE
    )
    templates = _templates(document)
    assert {role: spec["nodeName"] for role, spec in templates.items()} == {
        "Master": NODE,
        "Worker": PEER,
    }
    # gpu-training-submit stamps rank-offset 0 on the Master and 1 on the
    # Worker: rank 0's reporter runs in the Master Pod, rank 1's in the Worker.
    assert {
        role: [item["name"] for item in spec["containers"]]
        for role, spec in templates.items()
    } == {
        "Master": ["pytorch", "progress-reporter-0"],
        "Worker": ["pytorch", "progress-reporter-1"],
    }
    worker = templates["Worker"]["containers"][1]
    assert worker["command"][worker["command"].index("--rank") + 1] == "1"
    assert worker["command"][-1] == "/var/run/gpu-fault/rank-1.json"
    assert worker["image"] == RUNTIME_IMAGE
    for spec in templates.values():
        assert all(
            any(mount["name"] == "gfrs-progress" for mount in item["volumeMounts"])
            for item in spec["containers"]
        ), "every container of a Pod shares that Pod's progress volume"
        assert {item["name"] for item in spec["volumes"]} >= {
            "gfrs-progress",
            "gfrs-control-plane-ca",
        }
    assert {item["name"] for item in templates["Worker"]["volumes"]} >= {
        "gfrs-reporter-1-state",
        "gfrs-reporter-1-tmp",
    }


def test_single_rank_job_keeps_its_reporter_on_the_one_pod() -> None:
    document = _manifest(pr.SINGLE_NODE_MANIFEST)
    pr.install_reporter(document, "gl27-nan", {"Master": (0,)}, RUNTIME_IMAGE)
    (spec,) = _templates(document).values()
    assert [item["name"] for item in spec["containers"]] == [
        "pytorch",
        "progress-reporter-0",
    ]


def test_pin_replicas_refuses_a_role_the_manifest_lacks() -> None:
    document = _manifest(pr.SINGLE_NODE_MANIFEST)
    with pytest.raises(ValueError, match="exactly one Worker"):
        pr.pin_replicas(document, {"Master": NODE, "Worker": PEER})


def test_two_rank_job_needs_the_peer_node(tmp_path: Path) -> None:
    ctx = _ctx(SPECS["straggler_group"], tmp_path, {})
    assert ctx.settings.peer_node == "", "the lane has no --node-b here"
    with pytest.raises(VariantNotRun) as caught:
        RealManagedProgressJob(ctx, name="gl27-strag-a1-deadbeef", ranks=(0, 1))
    assert caught.value.reason == "FIXTURE_NOT_READY"
    assert "--node-b" in caught.value.detail
    with pytest.raises(ValueError, match="rank 0 or ranks 0 and 1"):
        RealManagedProgressJob(ctx, name="gl27-strag-a1-deadbeef", ranks=(0, 2))


def test_straggler_group_is_a_two_node_variant() -> None:
    spec = SPECS["straggler_group"]
    assert "--node-b" in spec.injection and spec.max_minutes == 30, spec


def _straggler_doc(incidents: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "progress_states": [],
        "incidents": incidents,
        "workflows": _workflow(
            ("FREEZE_EVIDENCE", "VALIDATE_FABRIC"),
            status="RUNNING",
            executed=[("FREEZE_EVIDENCE", "SUCCEEDED"), ("VALIDATE_FABRIC", "WAITING")],
        ),
        "decision": None,
        "notifications": [],
    }


@pytest.mark.parametrize(
    "scope", [{"node_ids": [PEER]}, {}], ids=["lagging-ranks-node", "no-node_ids"]
)
def test_straggler_group_keeps_both_ranks_advancing_until_the_lagging_rank_is_flagged(
    tmp_path: Path, scope: dict[str, Any]
) -> None:
    empty = _straggler_doc([])
    # As live: the incident names the lagging rank's node (node-b) alone and
    # its reasons carry the finding's text, never a rank; a record without
    # node_ids is not the reference rank either.
    flagged = _straggler_doc(
        [
            _incident(
                "straggler",
                state="ACTION_PENDING",
                reasons=["training rank is lagging peer progress"],
                **scope,
            )
        ]
    )
    store = RecordingStore(empty, empty, flagged)
    ctx = _ctx(SPECS["straggler_group"], tmp_path, store)
    result = SPECS["straggler_group"].execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["stages"]["not_stalled"] == []
    assert result["stages"]["same_group_only"] == []
    job = FakeProgressJob.created[-1]
    assert (job.ranks, result["attempt"]["nodes"]) == ((0, 1), [NODE, PEER])
    # Every records read was scoped to both nodes of the job: live 2026-10-26
    # 17:43Z the lane-node-only read never listed rank 1's incident on node-b.
    assert store.node_arguments == [f"{NODE},{PEER}"] * 3
    # Every poll (scan + read) first wrote both ranks: rank 1 STRAGGLER_LAG
    # steps behind rank 0 and both STRAGGLER_ADVANCE further than last time.
    assert store.scans == store.reads == 3
    polls = [job.writes[index : index + 2] for index in range(0, len(job.writes), 2)]
    assert len(polls) == 3, job.writes
    for index, ((rank_a, a), (rank_b, b)) in enumerate(polls):
        assert (rank_a, rank_b) == (0, 1)
        assert a["step"] == 100 + index * pr.STRAGGLER_ADVANCE
        assert a["step"] - b["step"] == pr.STRAGGLER_LAG
        assert a["labels"] == b["labels"] == {"progress_group": "data-parallel"}
        assert a["loss"] == b["loss"] == 1.0
    advance = json.loads((ctx.case_dir / "straggler-advance.json").read_text())
    assert [item["steps"] for item in advance["writes"]] == [
        {"0": 100, "1": 40},
        {"0": 110, "1": 50},
        {"0": 120, "1": 60},
    ]
    assert job.log_reads == 0, "reporter logs are read only for a missing finding"


def test_straggler_group_fails_when_the_reference_rank_is_flagged(
    tmp_path: Path,
) -> None:
    # A straggler incident naming rank 0's node is the reference rank flagged
    # as its own straggler; the lagging rank's incident alone is the PASS.
    both = _straggler_doc(
        [
            _incident(
                "straggler",
                state="ACTION_PENDING",
                reasons=["training rank is lagging peer progress"],
                node_ids=[PEER],
            ),
            {
                **_incident(
                    "straggler",
                    state="ACTION_PENDING",
                    reasons=["training rank is lagging peer progress"],
                    node_ids=[NODE],
                ),
                "incident_id": "inc-training-straggler-reference",
                "event_id": "training-straggler-cafebabe",
            },
        ]
    )
    result = SPECS["straggler_group"].execute(
        _ctx(SPECS["straggler_group"], tmp_path, RecordingStore(both))
    )
    assert result["verdict"] == "FAIL"
    assert result["errors"] == [
        "same_group_only: the reference rank was flagged as its own straggler"
    ]


def test_single_rank_job_records_are_read_on_the_lane_node_only(tmp_path: Path) -> None:
    store = RecordingStore(NAN_SETTLED)
    result = SPECS["nan"].execute(_ctx(SPECS["nan"], tmp_path, store))
    assert result["verdict"] == "PASS", result["errors"]
    assert store.node_arguments == [NODE]


def test_straggler_group_without_the_finding_keeps_reporter_logs_and_names_a_stall(
    tmp_path: Path,
) -> None:
    # The live 2026-10-26 shape: no straggler finding, rank 0 graded a stall.
    FakeProgressJob.logs = {"progress-reporter-1": "heartbeat refused: accepted=False"}
    stalled = _straggler_doc(
        [
            _incident(
                "hang",
                state="ACTION_PENDING",
                reasons=["training step stopped advancing"],
            )
        ]
    )
    store = RecordingStore(stalled)
    ctx = store_poll_clock(_ctx(SPECS["straggler_group"], tmp_path, store))
    result = SPECS["straggler_group"].execute(ctx)
    assert result["verdict"] == "FAIL"
    assert result["errors"] == [
        "chain: no training_straggler finding reached an incident",
        "not_stalled: a lagging rank that keeps advancing was graded as a stall",
    ]
    assert FakeProgressJob.created[-1].log_reads == 1
    written = json.loads((ctx.case_dir / "straggler-reporters.json").read_text())
    assert written == FakeProgressJob.logs
    assert store.reads == 600 // 20, "one read per poll over the detection budget"
