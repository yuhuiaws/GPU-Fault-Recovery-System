"""GF-GENERAL-LIVE-011 EFA/NCCL-hung injector/verdict contract over fakes.

The node probes' ``efa-traffic`` counters follow the fake job: they climb at
the job's rate while it runs and stop while it is stalled, so the runner's own
baseline/drop oracle is exercised end to end next to the scripted product
records (the grouped hung incident, the per-node warnings, the workflow plan
and its step executions).
"""

from __future__ import annotations

import itertools
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.hung_summary_codec import encode_summary
from scripts.e2e.regional import general_live_nccl as nc
from scripts.e2e.regional import general_live_nccl_fr as nf
from scripts.e2e.regional import general_live_nccl_pair as pair
from scripts.e2e.regional import general_live_nccl_roles as roles
from scripts.e2e.regional import general_live_network_isolation as iso
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from scripts.e2e.regional.general_live_nccl_job import STALL_MAX_SECONDS
from scripts.e2e.regional.general_live_registry import build_registry, coverage_rows
from scripts.e2e.regional.general_live_store import ACTIVITY_PROBE
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
)

SPECS = {spec.variant: spec for spec in (*nc.VARIANTS, *pair.VARIANTS)}
NODE = "ip-10-0-43-5.us-west-2.compute.internal"
PEER = "ip-10-0-43-6.us-west-2.compute.internal"
CLEAN_SURVEY = {
    "residue": {},
    "collector_env": {"GPU_FAULT_HOST_INTERVAL_SECONDS": "15"},
    "tools": {},
}
ACTIVE_BPS = 2 * 1024 * 1024
LOW_BPS = 60.0
HUNG_REASON = (
    "EFA/RDMA traffic remains at zero long enough to suspect a hung training attempt"
)


class Traffic:
    """What the fake EFA counters do: the job's rate, or the stalled rate."""

    def __init__(self) -> None:
        self.active = ACTIVE_BPS
        self.low = LOW_BPS
        self.stalled = 0.0
        self.resumed: float | None = None
        #: Released readings (both nodes) before the resumed pair falls to zero.
        self.fades_after: int | None = None
        self.released_reads = 0
        self.state = "idle"

    def rate(self, keep_active: bool) -> float:
        running = self.active if keep_active else self.low
        if self.state == "stalled":
            return self.stalled
        if self.state == "released":
            self.released_reads += 1
            if self.fades_after is not None and self.released_reads > self.fades_after:
                return 0.0
            return running if self.resumed is None else self.resumed
        return running if self.state == "running" else 0.0


class FakeNcclJob:
    instances: list["FakeNcclJob"] = []
    traffic = Traffic()
    alive_value = True
    events: dict[str, list[str]] | None = None
    #: What ``flight_ledgers`` answers (a replayed recorder's responder ledgers).
    ledgers: dict[str, Any] | None = None
    #: What ``flight_recorder_forensics`` answers (the trainer's own FR files).
    forensics: dict[str, Any] | None = None

    def __init__(
        self,
        ctx: Any,
        *,
        name: str,
        flight_recorder: Any = False,
        keep_active: bool = True,
        stall_mode: str = "quiet",
    ) -> None:
        self.ctx = ctx
        self.name = name
        self.attempt_id = f"{name}-a001"
        self.flight_recorder = flight_recorder
        self.replay = None if isinstance(flight_recorder, bool) else flight_recorder
        self.keep_active = keep_active
        self.stall_mode = stall_mode
        self.stalled = False
        self.stall_times: list[datetime] = []
        self.released = 0
        #: ``released`` as of each forensics read: the files must be read while
        #: the stall still holds, before the Pods move on or go away.
        self.forensics_reads: list[int] = []
        FakeNcclJob.instances.append(self)

    def start(self) -> dict[str, Any]:
        self.traffic.state = "running"
        return {}

    def identity(self) -> dict[str, Any]:
        return {
            "job_id": self.name,
            "attempt_id": self.attempt_id,
            "nodes": [NODE, PEER],
            "stall_mode": self.stall_mode,
            "keep_active": self.keep_active,
        }

    def stall(self) -> dict[str, Any]:
        self.stalled = True
        self.stall_times.append(self.ctx.clock())
        self.traffic.state = "stalled"
        return {"pod": "master"}

    def release(self) -> dict[str, Any]:
        self.released += 1
        self.traffic.state = "released"
        return {"released": ["master", "worker"]}

    def alive(self) -> dict[str, Any]:
        return {"alive": self.alive_value}

    def flight_ledgers(self) -> dict[str, Any]:
        return dict(FakeNcclJob.ledgers or {})

    def flight_recorder_forensics(self) -> dict[str, Any]:
        self.forensics_reads.append(self.released)
        return dict(FakeNcclJob.forensics or {})

    def trainer_events(self) -> dict[str, list[str]]:
        """The trainer's log: every stall so far, each released one resumed."""

        if FakeNcclJob.events is not None:
            return FakeNcclJob.events
        mode = self.stall_mode
        events: dict[str, list[str]] = {"master": [], "worker": []}
        for index in range(max(1, len(self.stall_times))):
            step = 40 + 100 * index
            events["master"] += [
                f"STALL step={step} entered=extra_all_reduce watchdog_seconds=2700 rank=0",
                f"STALL step={step} held=before_extra_all_reduce mode={mode} rank=1",
            ]
            events["worker"].append(
                f"STALL step={step} held=before_extra_all_reduce mode={mode} rank=8"
            )
            if index < self.released:
                for pod, rank in (("master", 1), ("worker", 8)):
                    events[pod].append(
                        f"RESUMED step={step} reason=release held_seconds=900 rank={rank}"
                    )
        return events

    def delete(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _patch_job(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeNcclJob.instances = []
    FakeNcclJob.traffic = Traffic()
    FakeNcclJob.alive_value = True
    FakeNcclJob.events = None
    FakeNcclJob.ledgers = None
    FakeNcclJob.forensics = None
    monkeypatch.setattr(nc, "ManagedNcclJob", FakeNcclJob)


def efa_probe(node: str, *, survey: bool = False) -> FakeProbe:
    """A node probe whose counter pair moves at the fake job's current rate."""

    clock = itertools.count()
    state = {"total": 10_000_000.0}

    def reading(arguments: tuple[str, ...]) -> dict[str, Any]:
        tick = next(clock)
        job = FakeNcclJob.instances[-1] if FakeNcclJob.instances else None
        rate = FakeNcclJob.traffic.rate(job.keep_active if job else True)
        state["total"] += rate * 30 if tick else 0
        return {
            "observed_at": 1_800_000_000.0 + 30 * tick,
            "pair_bytes_total": int(state["total"]),
            "port_count": 16,
            "complete_pairs": True,
            "node": node,
        }

    script: dict[str, Any] = {"efa-traffic": reading}
    if survey:
        script["survey"] = CLEAN_SURVEY
    return FakeProbe(script)


def hung_records(
    since: str,
    *,
    state: str = "ESCALATED",
    ops: tuple[str, ...] = nc.DIAGNOSTIC_DAG,
    failed: dict[int, str] | None = None,
    reporters: tuple[str, ...] = (NODE, PEER),
    count: int = 1,
    warnings: tuple[str, ...] = (NODE, PEER),
    warning_after: float = 300,
    hung_after: float = 750,
    ranks: list[dict[str, Any]] | None = None,
    tag: str = "",
    workflow: str = "wf",
    predecessor: str | None = None,
    encoded: bool = True,
    triage_failures: dict[str, str] | None = None,
) -> dict[str, Any]:
    """One stall wave's records; ``tag`` keeps a later wave's ids apart.

    A triage node result is stored as the executor stores it: the summary
    codec's zlib/JSON envelope (``encoded=False``: a plain legacy document).
    ``triage_failures`` (node -> cause) is a first wave whose node results the
    executor rejected: ``node_results`` empty, the causes under
    ``node_failures`` -- the live HyperPod ``fr_enabled`` record.
    """

    started = datetime.fromisoformat(since)
    incidents = [
        {
            "incident_id": f"inc-hung{tag}-{index}",
            "event_id": f"host-{reporters[0]}-{tag}{1000 + index}{nc.HUNG_SUFFIX}",
            "event_type": "GPU_FAULT_GROUP",
            "state": state,
            "node_ids": [NODE, PEER],
            "reasons": [HUNG_REASON]
            + [f"{node}{nc.CONCURRENT}{HUNG_REASON}" for node in reporters[1:]],
            "workflow_request_id": workflow,
            "created_at": (started + timedelta(seconds=hung_after + index)).isoformat(),
        }
        for index in range(count)
    ] + [
        {
            "incident_id": f"inc-warning{tag}-{node}",
            "event_id": f"host-{node}-{tag}99{nc.WARNING_SUFFIX}",
            "event_type": "NODE_HEALTH",
            "state": "RECOVERED",
            "node_ids": [node],
            "reasons": [
                "EFA/RDMA traffic remains at zero for an active training attempt"
            ],
            "workflow_request_id": f"wf{tag}-{node}",
            "created_at": (started + timedelta(seconds=warning_after)).isoformat(),
        }
        for node in warnings
    ]
    executions = []
    for index, operation in enumerate(ops):
        details: dict[str, Any] = {}
        if operation == "COLLECT_HUNG_TRIAGE":
            # Wave 1 samples the reporting node; wave 2 found nothing to add.
            first = index == ops.index("COLLECT_HUNG_TRIAGE")
            result = {"node_id": NODE, "ranks": ranks or []}
            stored = encode_summary(result) if encoded else result
            details = {"node_results": {NODE: stored} if first else {}}
            if first and triage_failures:
                details = {
                    "node_results": {},
                    "node_failures": dict(triage_failures),
                    "undetermined_nodes": sorted(triage_failures),
                    "retained_bytes": 0,
                    "retained_raw_bytes": 0,
                }
        executions.append(
            {
                "step_index": index,
                "operation": operation,
                "status": (failed or {}).get(index, "SUCCEEDED"),
                "error": "agent unreachable" if index in (failed or {}) else None,
                "details": details,
                "updated_at": (started + timedelta(seconds=800 + index)).isoformat(),
            }
        )
    return {
        "incidents": incidents,
        "workflows": {
            workflow: {
                "official_steps": [{"operation": op, "parameters": {}} for op in ops],
                "step_executions": executions,
                "status": "SUCCEEDED",
                "predecessor_workflow_id": predecessor,
            }
        },
    }


def _settings(spec: Any, tmp_path: Path):
    base = make_settings(spec, node=NODE, peer=PEER)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "site.yaml").write_text("site: test\n")
    return type(base)(**{**base.__dict__, "site_file": tmp_path / "site.yaml"})


def _ctx(spec: Any, tmp_path: Path, *, none: bool = False, **records: Any):
    """A context whose product records are ``hung_records`` (or none at all)."""

    calls: list[tuple[str, ...]] = []

    def respond(script: str, *arguments: str) -> dict[str, Any]:
        calls.append(arguments)
        if none:
            return {"incidents": [], "workflows": {}}
        return hung_records(arguments[-1], **records)

    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=respond),
        probes={},
        settings=_settings(spec, tmp_path),
    )
    start = datetime(2026, 10, 24, tzinfo=timezone.utc)
    ticks = itertools.count()
    ctx.clock = lambda: start + timedelta(seconds=15 * next(ticks))
    ctx.probes[f"node@{NODE}"] = efa_probe(NODE, survey=True)  # type: ignore[assignment]
    ctx.probes[f"node@{PEER}"] = efa_probe(PEER)  # type: ignore[assignment]
    return ctx, calls


def _run(variant: str, tmp_path: Path, **records: Any) -> dict[str, Any]:
    ctx, _calls = _ctx(SPECS[variant], tmp_path, **records)
    return SPECS[variant].execute(ctx)


def _window(rates: list[float], *, start: float = 1_000.0) -> list[dict[str, Any]]:
    total = 5_000.0
    readings = []
    for index in range(len(rates) + 1):
        if index:
            total += rates[index - 1] * 30
        readings.append(
            {
                "observed_at": start + 30 * index,
                "pair_bytes_total": int(total),
                "port_count": 16,
                "complete_pairs": True,
            }
        )
    return readings


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_registry_covers_every_011_variant() -> None:
    rows = coverage_rows()
    variants = [*nc.VARIANTS, *nf.VARIANTS, *pair.VARIANTS, *roles.VARIANTS]
    build_registry(variants, rows=rows, categories=[11])
    assert {s.variant for s in variants} == set(rows["GF-GENERAL-LIVE-011"])
    assert {s.variant for s in nf.VARIANTS} == {
        "few_candidates",
        "ambiguous_candidates",
        "different_process_groups",
        "fr_temporal_completion",
        "fr_second_missing",
        "fr_whole_group",
        "fr_plus_process",
    }, "the FR module implements exactly the replayed forensics variants"
    assert [s.variant for s in pair.VARIANTS] == ["broken_pair"], (
        "the pair module carries exactly the broken_pair entry"
    )
    implemented = {s.variant for s in [*nc.VARIANTS, *pair.VARIANTS] if s.implemented}
    assert implemented == {
        "active_to_zero",
        "always_low",
        "write_progress",
        "phase_progress",
        "fr_enabled",
        "fr_disabled",
        "simultaneous_nodes",
        "repeat_after_recovery",
        "unreachable_node",
        "broken_pair",
    }
    pair_spec = SPECS["broken_pair"]
    assert pair_spec.proof_mode == "live-replay", "broken_pair replays inside the unit"
    assert pair_spec.approvals == ("HOST_SOURCE_REPLAY",), (
        "broken_pair replays the pair inside the host collector under the token"
    )
    assert {s.variant for s in roles.VARIANTS} == {
        "role_mismatch",
        "tid_namespace",
        "wchan_zero",
        "cpu_interval",
        "pyspy_nonzero",
        "attempt_change",
    }
    assert all(spec.implemented for spec in roles.VARIANTS), (
        "every role variant is executable"
    )
    assert all(
        lane == "hp"
        for spec in [*nc.VARIANTS, *nf.VARIANTS, *pair.VARIANTS]
        if spec.variant != "unreachable_node"
        for lane in spec.lanes
    ), "the stall variants and their FR/pair modules run on the HyperPod lane"
    assert SPECS["unreachable_node"].lanes == ("eks", "hp"), (
        "the Agent-channel variant runs on both clusters (two EFA nodes each)"
    )
    assert all(spec.lanes == ("eks", "hp") for spec in roles.VARIANTS), (
        "the role/attempt variants run on both clusters (two EFA nodes each)"
    )
    # Every deferred variant keeps an explicit NOT_RUN reason in the denominator.
    for spec in nc.VARIANTS:
        if not spec.implemented:
            assert spec.deferred.split(":", 1)[0] in {
                "FIXTURE_NOT_READY",
                "SCALE",
                "NOT_AUTHORIZED",
            }, spec.variant


def test_the_in_pod_watchdog_outlasts_the_runners_longest_wait() -> None:
    assert nc.RUNNER_STALL_SECONDS < STALL_MAX_SECONDS, (
        "a held rank must not release the stall while the runner still waits"
    )


# --------------------------------------------------------------------------- #
# the EFA counter oracle
# --------------------------------------------------------------------------- #
def test_efa_rates_drop_incomplete_reset_and_errored_intervals() -> None:
    readings = _window([ACTIVE_BPS, ACTIVE_BPS, ACTIVE_BPS, ACTIVE_BPS])
    assert nc.efa_rates(readings) == [pytest.approx(ACTIVE_BPS, rel=1e-6)] * 4
    readings[1] = {**readings[1], "complete_pairs": False}
    readings[3] = {**readings[3], "pair_bytes_total": 0}
    readings[4] = {"error": "OSError", "error_kind": "probe"}
    assert nc.efa_rates(readings) == [None, None, None, None]
    assert nc.efa_rates(_window([5.0])[::-1]) == [None], "time running backwards"


def test_an_inactive_or_too_short_baseline_is_not_an_active_one() -> None:
    minimum = nc.MIN_ACTIVE_BPS
    active = {NODE: _window([ACTIVE_BPS] * 4), PEER: _window([ACTIVE_BPS] * 4)}
    assert nc.baseline_errors(active, minimum_bps=minimum) == []
    dipped = {**active, PEER: _window([ACTIVE_BPS, 200_000, ACTIVE_BPS, ACTIVE_BPS])}
    [error] = nc.baseline_errors(dipped, minimum_bps=minimum)
    assert error.startswith(PEER) and "under the product's 1048576 B/s" in error
    short = {NODE: _window([ACTIVE_BPS, ACTIVE_BPS]), PEER: active[PEER]}
    assert nc.baseline_errors(short, minimum_bps=minimum) == [
        f"{NODE}: 2 usable EFA interval(s), need 3"
    ]
    low = {NODE: _window([LOW_BPS] * 4), PEER: _window([LOW_BPS] * 4)}
    assert nc.baseline_errors(low, minimum_bps=minimum, active=False) == []
    [error] = nc.baseline_errors(
        active | {PEER: low[PEER]}, minimum_bps=minimum, active=False
    )
    assert error.startswith(NODE) and "at or over" in error


def test_the_drop_must_reach_zero_on_both_nodes_for_the_hung_threshold() -> None:
    zero = {NODE: _window([0.0] * 25), PEER: _window([0.0] * 25)}
    assert nc.drop_errors(zero, zero_bps=1) == []
    moving = {**zero, PEER: _window([0.0] * 12 + [40.0] + [0.0] * 12)}
    [error] = nc.drop_errors(moving, zero_bps=1)
    assert error.startswith(PEER) and "kept moving at up to 40.0 B/s" in error
    brief = {NODE: _window([0.0] * 10), PEER: zero[PEER]}
    [error] = nc.drop_errors(brief, zero_bps=1)
    assert "covers 300 s, under the 600 s hung threshold" in error


def test_the_resume_reads_the_last_interval_against_the_floor() -> None:
    back = {NODE: _window([0.0, ACTIVE_BPS]), PEER: _window([ACTIVE_BPS] * 2)}
    assert nc.resume_errors(back, floor_bps=nc.MIN_ACTIVE_BPS) == []
    [error] = nc.resume_errors(
        {**back, PEER: _window([ACTIVE_BPS, 0.0])}, floor_bps=nc.MIN_ACTIVE_BPS
    )
    assert error.startswith(PEER) and "did not resume" in error


def test_the_trainer_account_must_show_the_requested_stall() -> None:
    good = FakeNcclJob(None, name="n", stall_mode="write").trainer_events()
    assert nc.stall_event_errors(good, "write") == []
    assert nc.stall_event_errors(good, "phase") == [
        "master: no rank reports holding before it in mode phase",
        "worker: no rank reports holding before it in mode phase",
    ]
    broken = {
        **good,
        "worker": good["worker"]
        + [
            "[default1]:STALL-ACTIVITY-ERROR mode=write error=OSError rank=9",
            "[default1]:RESUMED step=40 reason=watchdog",
        ],
    }
    errors = nc.stall_event_errors(broken, "write")
    assert any("activity failed" in item for item in errors), errors
    assert any("ended before" in item for item in errors), errors
    assert nc.stall_event_errors({"master": []}, "quiet")[0] == (
        "no rank reports entering the extra collective"
    )


# --------------------------------------------------------------------------- #
# the verdict over product records
# --------------------------------------------------------------------------- #
SINCE = "2026-10-24T00:00:00+00:00"
STARTED = datetime.fromisoformat(SINCE)


def test_dag_and_execution_errors_read_the_plan_and_every_step() -> None:
    assert nc.dag_errors(hung_records(SINCE)) == []
    assert nc.dag_errors({"incidents": [], "workflows": {}}) == [
        "no EFA hung_suspected finding reached an incident"
    ]
    wrong = hung_records(SINCE, ops=("FREEZE_EVIDENCE", "VALIDATE_FABRIC"))
    assert any("diagnostic DAG" in item for item in nc.dag_errors(wrong)), wrong
    forbidden = hung_records(SINCE, ops=(*nc.DIAGNOSTIC_DAG, "STOP_WORKLOADS"))
    assert any("forbidden" in item for item in nc.dag_errors(forbidden)), forbidden
    assert nc.execution_errors(hung_records(SINCE)) == []
    failed = hung_records(SINCE, failed={5: "FAILED"})
    assert nc.execution_errors(failed) == [
        "step 5 COLLECT_DIAGNOSTIC_BUNDLE ended FAILED: agent unreachable"
    ]
    planned_only = hung_records(SINCE)
    planned_only["workflows"]["wf"]["step_executions"].pop(3)
    assert nc.execution_errors(planned_only) == [
        "step 3 COLLECT_HUNG_TRIAGE never executed"
    ]


def test_escalated_grouped_and_both_nodes() -> None:
    assert nc.escalated_errors(hung_records(SINCE)) == []
    for state in ("ACTION_PENDING", "QUARANTINED"):
        assert nc.escalated_errors(hung_records(SINCE, state=state)) == [
            f"hung incident is {state}, expected ESCALATED"
        ]
    assert nc.grouped_errors(hung_records(SINCE, count=2), expect_single=True), (
        "two hung incidents for one stall wave must fail the merge check"
    )
    assert nc.grouped_errors(hung_records(SINCE), expect_single=True) == []
    merged = nc.hung_incident(hung_records(SINCE))
    assert merged is not None and nc.reported_nodes(merged) == {NODE, PEER}
    assert nc.both_nodes_errors(hung_records(SINCE), [NODE, PEER]) == []
    alone = hung_records(SINCE, reporters=(PEER,))
    assert nc.both_nodes_errors(alone, [NODE, PEER]) == [
        f"no hung evidence from {[NODE]} reached the merged incident"
    ]


def test_timing_needs_a_warning_per_node_after_60_s_and_hung_after_600_s() -> None:
    nodes = [NODE, PEER]
    assert nc.timing_errors(hung_records(SINCE), STARTED, nodes) == []
    early = hung_records(SINCE, hung_after=420)
    assert nc.timing_errors(early, STARTED, nodes) == [
        "HUNG_SUSPECTED 420 s after the stall, before the 600 s hung threshold"
    ]
    one = hung_records(SINCE, warnings=(NODE,))
    assert nc.timing_errors(one, STARTED, nodes) == [
        f"{PEER}: no ZERO_WARNING incident for the stall"
    ]
    rushed = nc.timing_errors(hung_records(SINCE, warning_after=20), STARTED, nodes)
    assert len(rushed) == 2 and all("warning threshold" in item for item in rushed)
    late = nc.timing_errors(hung_records(SINCE, warning_after=900), STARTED, nodes)
    assert late == [f"{node}: ZERO_WARNING only after HUNG_SUSPECTED" for node in nodes]


def test_fr_errors_read_the_triage_ranks() -> None:
    def triage(*rows: dict[str, Any]) -> dict[str, Any]:
        return hung_records(SINCE, ranks=list(rows))

    paired = {
        "flight_recorder": {"status": "dumped"},
        "flight_recorder_series": {"status": "PAIRED"},
    }
    off = {"flight_recorder": {"status": "not_configured"}}
    assert nc.fr_errors(triage(paired, paired), expect_configured=True) == []
    assert nc.fr_errors(triage(off, off), expect_configured=False) == []
    assert nc.fr_errors(triage(off), expect_configured=True) == [
        "FR was configured but 1 rank(s) read not_configured",
        "no sampled rank got two real FR dumps (statuses ['not_configured'], series ['None'])",
    ]
    single = {
        "flight_recorder": {"status": "dumped"},
        "flight_recorder_series": {"status": "SECOND_UNAVAILABLE"},
    }
    assert nc.fr_errors(triage(single), expect_configured=True), "one dump is not two"
    assert nc.fr_errors(triage(paired), expect_configured=False), (
        "a dump with FR off must be flagged"
    )
    assert nc.fr_errors(triage(), expect_configured=False) == [
        "COLLECT_HUNG_TRIAGE sampled no rank process"
    ]


def test_fr_errors_decode_the_envelope_each_triage_result_is_stored_in() -> None:
    """The executor keeps every triage node result as a zlib/JSON envelope.

    The oracle read ``ranks`` straight off it, so even a triage that sampled
    every rank read "sampled no rank process" (GF-GENERAL-LIVE-011.fr_enabled).
    A plain legacy document is read as it is; an envelope that does not decode
    is named, never silently empty.
    """

    paired = {
        "flight_recorder": {"status": "dumped"},
        "flight_recorder_series": {"status": "PAIRED"},
    }
    for encoded in (True, False):
        records = hung_records(SINCE, ranks=[paired], encoded=encoded)
        assert nc.fr_errors(records, expect_configured=True) == [], encoded
    broken = hung_records(SINCE, ranks=[paired])
    triage = next(
        item
        for item in broken["workflows"]["wf"]["step_executions"]
        if item["operation"] == "COLLECT_HUNG_TRIAGE"
    )
    triage["details"]["node_results"][NODE]["sha256"] = "0" * 64
    errors = nc.fr_errors(broken, expect_configured=True)
    assert errors[-1] == "COLLECT_HUNG_TRIAGE sampled no rank process", errors
    assert errors[0].startswith(
        f"COLLECT_HUNG_TRIAGE result of {NODE} is unreadable: "
    ), errors


# --------------------------------------------------------------------------- #
# execute paths
# --------------------------------------------------------------------------- #
def test_active_to_zero_passes_after_an_active_baseline_and_a_real_drop(
    tmp_path: Path,
) -> None:
    ctx, calls = _ctx(SPECS["active_to_zero"], tmp_path)
    result = SPECS["active_to_zero"].execute(ctx)

    assert result["verdict"] == "PASS", result["errors"]
    job = FakeNcclJob.instances[-1]
    assert job.stalled and job.released == 1, "stalled, then released once"
    assert ctx.cleanup.pending() == ["job-" + job.name], "the release is done"
    assert set(result["stages"]) == {
        "chain",
        "executed",
        "escalated",
        "timing",
        "merged",
        "resumed",
    }
    assert calls[-1][1] == f"{NODE},{PEER}", "both attempt nodes are read"
    baseline = ctx.evidence.case_dir / "a2z-active-efa.json"
    assert baseline.is_file() and (ctx.evidence.case_dir / "oracle.jsonl").is_file()


def test_no_active_baseline_is_inconclusive_and_never_stalls(tmp_path: Path) -> None:
    FakeNcclJob.traffic.active = 100_000.0
    ctx, _calls = _ctx(SPECS["active_to_zero"], tmp_path)
    with pytest.raises(VariantInconclusive, match="no active EFA baseline"):
        SPECS["active_to_zero"].execute(ctx)
    assert not FakeNcclJob.instances[-1].stalled, "no stall without a baseline"


def test_traffic_that_keeps_moving_through_the_stall_is_inconclusive(
    tmp_path: Path,
) -> None:
    FakeNcclJob.traffic.stalled = 500.0
    ctx, calls = _ctx(SPECS["active_to_zero"], tmp_path)
    with pytest.raises(VariantInconclusive, match="kept moving"):
        SPECS["active_to_zero"].execute(ctx)
    assert calls == [], "the product is not judged on an injection that did not happen"


@pytest.mark.parametrize(
    ("alive", "events", "message"),
    [
        (False, None, "did not stay alive"),
        (True, {"master": [], "worker": []}, "extra collective"),
    ],
)
def test_a_dead_job_or_a_trainer_that_never_stalled_is_inconclusive(
    tmp_path: Path, alive: bool, events: dict[str, list[str]] | None, message: str
) -> None:
    FakeNcclJob.alive_value = alive
    FakeNcclJob.events = events
    ctx, _calls = _ctx(SPECS["active_to_zero"], tmp_path)
    with pytest.raises(VariantInconclusive, match=message):
        SPECS["active_to_zero"].execute(ctx)


def test_a_valid_injection_with_no_hung_incident_fails(tmp_path: Path) -> None:
    result = _run("active_to_zero", tmp_path, none=True)
    assert result["verdict"] == "FAIL"
    assert (
        "chain: no EFA hung_suspected finding reached an incident" in result["errors"]
    )


def test_a_job_that_does_not_resume_fails(tmp_path: Path) -> None:
    FakeNcclJob.traffic.resumed = 0.0
    result = _run("active_to_zero", tmp_path)
    assert result["verdict"] == "FAIL"
    assert result["stages"]["resumed"], result["stages"]


def _unreachable_ctx(
    tmp_path: Path, *, gap: bool = True, fabricated: bool = False
) -> tuple[Any, dict[str, Any], FakeProbe]:
    """A context whose peer Agent block is a fake probe and whose hung records
    carry (or lack) the product's coverage-gap record for the peer."""

    spec = SPECS["unreachable_node"]
    start = datetime(2026, 10, 24, tzinfo=timezone.utc)
    marks: dict[str, Any] = {"restored": False}

    def respond(script: str, *arguments: str) -> dict[str, Any]:
        if script == iso.ISOLATION_RECORDS_PROBE:
            seen = start + timedelta(days=1) if marks["restored"] else start
            return {
                "heartbeat": None,
                "observations": [],
                "notifications": [],
                "agent": {
                    "node_id": PEER,
                    "endpoint": f"https://{PEER}:9099",
                    "last_seen_at": seen.isoformat(),
                    "lifecycle_state": "ACTIVE",
                    "lease_expires_at": None,
                },
            }
        records = hung_records(arguments[-1])
        for item in records["workflows"]["wf"]["step_executions"]:
            if item["operation"] != "COLLECT_HUNG_SUMMARY" or item["step_index"] != 1:
                continue
            results = {NODE: encode_summary({"node_id": NODE, "ranks": []})}
            if fabricated:
                results[PEER] = encode_summary({"node_id": PEER, "ranks": []})
            item["details"] = {"node_results": results}
            if gap:
                item["details"]["undetermined_nodes"] = [PEER]
                item["details"]["node_failures"] = {
                    PEER: "temporary transport failure: [Errno 111] Connection refused"
                }
        return records

    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=respond),
        probes={},
        settings=_settings(spec, tmp_path),
    )
    ticks = itertools.count()
    ctx.clock = lambda: start + timedelta(seconds=15 * next(ticks))
    ctx.probes[f"node@{NODE}"] = efa_probe(NODE, survey=True)  # type: ignore[assignment]
    ctx.probes[f"node@{PEER}"] = efa_probe(PEER)  # type: ignore[assignment]

    def block(arguments: tuple[str, ...]) -> dict[str, Any]:
        marks["blocked_at"] = ctx.clock()
        marks["block_arguments"] = arguments
        return {
            "rules": ["-A INPUT -p tcp --dport 9099 -m comment --comment t -j REJECT"],
            "deadman": {"timer": {"ActiveState": "active"}},
            "connectivity": False,
        }

    def cleanup(_arguments: tuple[str, ...]) -> dict[str, Any]:
        marks["restored"] = True
        marks["restored_at"] = ctx.clock()
        return {
            "clean": True,
            "kinds": {
                "agent-ingress": {
                    "remaining": [],
                    "timer": {"ActiveState": "inactive"},
                    "connectivity": True,
                }
            },
        }

    fake = FakeProbe({"block": block, "cleanup": cleanup})
    ctx.probes[f"isolation@{PEER}"] = fake  # type: ignore[assignment]
    return ctx, marks, fake


def test_unreachable_node_blocks_the_peer_agent_before_the_stall_and_passes_on_a_recorded_gap(
    tmp_path: Path,
) -> None:
    spec = SPECS["unreachable_node"]
    assert spec.approvals == ("NODE_AGENT_NETWORK_ISOLATION",)
    assert spec.flags == "X" and spec.risk == "H"
    ctx, marks, fake = _unreachable_ctx(tmp_path)
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert set(result["stages"]) == {
        "chain",
        "escalated",
        "timing",
        "coverage_gap",
        "no_isolation",
        "channel_restored",
        "resumed",
    }
    job = FakeNcclJob.instances[-1]
    assert marks["blocked_at"] < job.stall_times[0], "the block precedes the stall"
    assert marks["block_arguments"][:7] == (
        "block", "--kind", "agent-ingress", "--deadman-seconds",
        str(iso.AGENT_BLOCK_SECONDS), "--unit", iso.AGENT_UNIT,
    )  # fmt: skip
    assert "--port" in marks["block_arguments"] and "9099" in marks["block_arguments"]
    assert [call[0] for call in fake.calls] == ["block", "cleanup"]
    assert job.released == 1 and ctx.cleanup.pending() == ["job-" + job.name]
    assert result["unreachable_node"] == PEER
    assert result["later_steps"] == {
        "COLLECT_DIAGNOSTIC_BUNDLE": "SUCCEEDED",
        "VALIDATE_FABRIC": "SUCCEEDED",
        "ESCALATE_SUPPORT": "SUCCEEDED",
    }
    assert (ctx.evidence.case_dir / "unr-evidence-records.json").is_file(), (
        "the evidence-phase records are kept"
    )
    assert (ctx.evidence.case_dir / "isolation-agent-ingress-cleanup.json").is_file(), (
        "the probe's cleanup document is kept"
    )


@pytest.mark.parametrize(
    ("options", "needle"),
    [
        ({"gap": False}, "coverage gap is not recorded"),
        ({"fabricated": True}, "fabricated reading"),
    ],
    ids=["no-gap", "fabricated"],
)
def test_unreachable_node_fails_without_the_gap_record_or_with_a_fabricated_reading(
    tmp_path: Path, options: dict[str, bool], needle: str
) -> None:
    ctx, _marks, fake = _unreachable_ctx(tmp_path, **options)
    result = SPECS["unreachable_node"].execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any(needle in item for item in result["stages"]["coverage_gap"]), result
    assert [call[0] for call in fake.calls] == ["block", "cleanup"], "restored anyway"


def test_simultaneous_nodes_waits_for_both_reporters_and_needs_one_incident(
    tmp_path: Path,
) -> None:
    assert _run("simultaneous_nodes", tmp_path)["verdict"] == "PASS"
    result = _run("simultaneous_nodes", tmp_path / "two", count=2)
    assert result["verdict"] == "FAIL"
    assert any("one stall wave" in item for item in result["errors"]), result["errors"]
    alone = _run("simultaneous_nodes", tmp_path / "alone", reporters=(NODE,))
    assert alone["verdict"] == "FAIL"
    assert any("no hung evidence from" in item for item in alone["errors"]), alone


@pytest.mark.parametrize(
    ("variant", "mode"), [("write_progress", "write"), ("phase_progress", "phase")]
)
def test_progress_variants_stall_in_their_mode_and_still_need_the_chain(
    tmp_path: Path, variant: str, mode: str
) -> None:
    result = _run(variant, tmp_path)
    assert result["verdict"] == "PASS", result["errors"]
    assert FakeNcclJob.instances[-1].stall_mode == mode
    assert result["attempt"]["stall_mode"] == mode
    late = _run(variant, tmp_path / "early", hung_after=300)
    assert late["verdict"] == "FAIL", "progress may not shorten or bend the hung clock"


def test_fr_variants_configure_the_job_and_read_the_dumps(tmp_path: Path) -> None:
    paired = {
        "flight_recorder": {"status": "dumped"},
        "flight_recorder_series": {"status": "PAIRED"},
    }
    off = {"flight_recorder": {"status": "not_configured"}}
    result = _run("fr_enabled", tmp_path, ranks=[paired])
    assert result["verdict"] == "PASS", result["errors"]
    assert FakeNcclJob.instances[-1].flight_recorder is True
    assert _run("fr_enabled", tmp_path / "off", ranks=[off])["verdict"] == "FAIL"
    result = _run("fr_disabled", tmp_path / "disabled", ranks=[off])
    assert result["verdict"] == "PASS", result["errors"]
    assert FakeNcclJob.instances[-1].flight_recorder is False
    assert _run("fr_disabled", tmp_path / "dumped", ranks=[paired])["verdict"] == "FAIL"


def _torch29_forensics(*pods: str) -> dict[str, Any]:
    """The trainer's account on PyTorch 2.9: dumps under $HOME/.cache/torch."""

    return {
        pod: {
            "home": "/root",
            "files": [
                {
                    "path": f"/tmp/torch_nccl_debug_pipe_{rank}.pipe",
                    "type": "fifo",
                    "size": 0,
                    "mtime": 1,
                },
                {
                    "path": f"/root/.cache/torch/comm_lib_trace_rank_{rank}",
                    "type": "regular file",
                    "size": 48213,
                    "mtime": 2,
                },
            ],
            "log_lines": ["... ProcessGroupNCCL preparing to dump debug info ..."],
            "error": None,
        }
        for rank, pod in enumerate(pods)
    }


def test_fr_variants_keep_the_trainers_own_fr_files_as_evidence_and_attribute_a_miss(
    tmp_path: Path,
) -> None:
    missing = {
        "flight_recorder": {
            "status": "dump_missing",
            "pipe": "/proc/3024782/root/tmp/torch_nccl_debug_pipe_0.pipe",
        }
    }
    FakeNcclJob.forensics = _torch29_forensics("master", "worker")
    ctx, _calls = _ctx(SPECS["fr_enabled"], tmp_path, ranks=[missing])
    result = SPECS["fr_enabled"].execute(ctx)
    assert result["verdict"] == "FAIL"
    assert result["stages"]["fr_dumps"] == [
        "no sampled rank got two real FR dumps (statuses ['dump_missing'], series ['None'])",
        "the trainer wrote 2 FR dump file(s) under /root/.cache/torch/comm_lib_trace_rank_*"
        " (the PyTorch >= 2.9 default) and 0 under /tmp/nccl_trace_rank_* (the only"
        " default the product reads); 2 'preparing to dump debug info' line(s)",
    ]
    job = FakeNcclJob.instances[-1]
    assert job.forensics_reads == [0], "read once, while the stall still holds"
    evidence = json.loads((ctx.case_dir / "fron-fr-forensics.json").read_text())
    assert evidence == FakeNcclJob.forensics

    paired = {
        "flight_recorder": {"status": "dumped"},
        "flight_recorder_series": {"status": "PAIRED"},
    }
    result = _run("fr_enabled", tmp_path / "paired", ranks=[paired])
    assert result["verdict"] == "PASS", result["errors"]
    assert result["stages"]["fr_dumps"] == [], "a paired read needs no attribution"

    FakeNcclJob.forensics = {
        "master": {"home": "/root", "files": [], "log_lines": [], "error": None}
    }
    ctx, _calls = _ctx(
        SPECS["fr_disabled"],
        tmp_path / "disabled",
        ranks=[{"flight_recorder": {"status": "not_configured"}}],
    )
    result = SPECS["fr_disabled"].execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert FakeNcclJob.instances[-1].forensics_reads == [0]
    evidence = json.loads((ctx.case_dir / "froff-fr-forensics.json").read_text())
    assert evidence == FakeNcclJob.forensics


def test_always_low_passes_on_a_quiet_job_with_no_suspicion(tmp_path: Path) -> None:
    result = _run("always_low", tmp_path, none=True)
    assert result["verdict"] == "PASS", result["errors"]
    job = FakeNcclJob.instances[-1]
    assert job.keep_active is False and job.stalled and job.released == 1


def test_always_low_fails_on_a_warning_or_a_hung_suspicion(tmp_path: Path) -> None:
    result = _run("always_low", tmp_path, reporters=(NODE,), count=1)
    assert result["verdict"] == "FAIL"
    assert any("zero-traffic suspicion" in item for item in result["errors"]), result


def test_always_low_is_inconclusive_when_the_job_went_active(tmp_path: Path) -> None:
    FakeNcclJob.traffic.low = 3_000_000.0
    ctx, _calls = _ctx(SPECS["always_low"], tmp_path, none=True)
    with pytest.raises(VariantInconclusive, match="always-low job reached the active"):
        SPECS["always_low"].execute(ctx)


# --------------------------------------------------------------------------- #
# repeat_after_recovery
# --------------------------------------------------------------------------- #
REPEAT = SPECS["repeat_after_recovery"]
REPEAT_STAGES = [
    "chain",
    "executed",
    "escalated",
    "timing",
    "merged",
    "repeat",
    "repeat_chain",
    "repeat_executed",
    "repeat_escalated",
    "repeat_timing",
    "repeat_merged",
    "resumed",
]


class RepeatProduct:
    """The product's records over the repeat variant's two stalls.

    The first stall gets the usual grouped incident and warnings; once the fake
    job stalled again the second gets what ``second`` names: ``own`` (its own
    incident and workflow, after the first's), ``missing``, ``merged`` (only
    concurrent evidence on the first incident), ``duplicate``, ``shared`` (on
    the first workflow) or ``unchained`` (no predecessor).
    """

    def __init__(self, second: str = "own", *, first: bool = True) -> None:
        self.second = second
        self.first = first

    def __call__(self, script: str, *arguments: str) -> dict[str, Any]:
        stalls = FakeNcclJob.instances[-1].stall_times
        if not self.first:
            return {"incidents": [], "workflows": {}}
        records = hung_records(stalls[0].isoformat())
        if len(stalls) < 2 or self.second == "missing":
            return records
        if self.second == "merged":
            records["incidents"][0]["reasons"].append(
                f"{NODE}{nc.CONCURRENT}{HUNG_REASON}"
            )
            return records
        later = hung_records(
            stalls[1].isoformat(),
            tag="2",
            workflow="wf" if self.second == "shared" else "wf-2",
            predecessor=None if self.second == "unchained" else "wf",
            count=2 if self.second == "duplicate" else 1,
        )
        return {
            "incidents": records["incidents"] + later["incidents"],
            "workflows": {**later["workflows"], **records["workflows"]},
        }


def _repeat(tmp_path: Path, product: RepeatProduct | None = None) -> Any:
    ctx, _calls = _ctx(REPEAT, tmp_path)
    ctx.regional.cpu_python_responder = product or RepeatProduct()
    return ctx


def test_the_rearm_window_is_the_collectors_recovery_window_plus_margin(
    tmp_path: Path,
) -> None:
    ctx = _repeat(tmp_path)
    ctx.survey = CLEAN_SURVEY
    assert nc.rearm_seconds(ctx) == 180 + 3 * 15, "120 s + two 30 s spacings"
    prefix = "GPU_FAULT_EFA_TRAFFIC_RECOVERY_"
    ctx.survey = {
        "collector_env": {
            "GPU_FAULT_HOST_INTERVAL_SECONDS": "10",
            f"{prefix}SECONDS": "300",
            f"{prefix}MAX_GAP_SECONDS": "20",
        }
    }
    assert nc.rearm_seconds(ctx) == 300 + 2 * 10 + 3 * 10, "spacing is half the gap"
    ctx.survey = {"collector_env": {f"{prefix}MIN_SAMPLES": "10"}}
    assert nc.rearm_seconds(ctx) == 10 * 30 + 3 * 15, "or more samples, if longer"


def test_a_later_stall_reads_only_its_own_trainer_lines() -> None:
    job = FakeNcclJob(None, name="n")
    job.stall_times = [STARTED, STARTED + timedelta(seconds=1500)]
    job.released = 1
    events = job.trainer_events()
    assert nc.stall_step(events) == 140
    assert nc.stall_step({"master": []}) is None
    assert any(
        "ended before" in item for item in nc.stall_event_errors(events, "quiet")
    ), "unfiltered, the first stall's RESUMED lines would end the second"
    assert nc.stall_event_errors(nc.events_after(events, 40), "quiet") == []
    stepless = {"master": ["STALL-ACTIVITY-ERROR mode=write error=OSError rank=1"]}
    assert nc.events_after(stepless, 40) == stepless


def test_episode_keeps_each_stall_waves_own_incidents() -> None:
    second_at = STARTED + timedelta(seconds=1500)
    later = hung_records(second_at.isoformat(), tag="2", workflow="wf-2")
    records = hung_records(SINCE)
    records["incidents"] += later["incidents"]
    first = nc.episode(records, STARTED, second_at)
    second = nc.episode(records, second_at)
    assert {item["incident_id"] for item in first["incidents"]} == {
        "inc-hung-0",
        f"inc-warning-{NODE}",
        f"inc-warning-{PEER}",
    }
    assert {item["incident_id"] for item in second["incidents"]} == {
        "inc-hung2-0",
        f"inc-warning2-{NODE}",
        f"inc-warning2-{PEER}",
    }
    assert first["workflows"] == records["workflows"], "workflows stay as read"


def test_repeat_errors_need_a_first_incident_and_judge_absorption_against_it() -> None:
    empty: dict[str, Any] = {"incidents": [], "workflows": {}}
    assert nc.repeat_errors(empty, empty, empty) == [
        "no first hung incident to repeat after"
    ]
    first = hung_records(SINCE)
    # Its own first-wave peer evidence is not the second stall's.
    assert nc.repeat_errors(first, empty, first) == [
        "the second stall raised no hung incident of its own"
    ]


def test_repeat_after_recovery_passes_with_a_second_collection_of_its_own(
    tmp_path: Path,
) -> None:
    ctx = _repeat(tmp_path)
    result = REPEAT.execute(ctx)

    assert result["verdict"] == "PASS", result["errors"]
    assert list(result["stages"]) == REPEAT_STAGES
    job = FakeNcclJob.instances[-1]
    assert len(job.stall_times) == 2 and job.released == 2, "two stalls, both released"
    assert ctx.cleanup.pending() == ["job-" + job.name], "every release is done"
    assert result["rearm_seconds"] == 225
    case = ctx.evidence.case_dir
    recovered = json.loads((case / "rar-recovered-baseline.json").read_text())
    assert recovered == {"minimum_bps": nc.MIN_ACTIVE_BPS, "seconds": 225, "errors": []}
    gap = (job.stall_times[1] - job.stall_times[0]).total_seconds()
    assert gap > nc.ZERO_HUNG_SECONDS + nc.RESUME_SECONDS + 225, gap
    trainer = json.loads((case / "rar2-stalled-job.json").read_text())["trainer"]
    assert any("RESUMED step=40" in line for line in trainer["master"]), (
        "the second stall was judged on a log that still holds the first"
    )
    for name in ("hung-records", "hung-before-repeat", "hung2-records"):
        assert (case / f"{name}.json").is_file(), name


@pytest.mark.parametrize(
    ("second", "message"),
    [
        ("missing", "repeat: the second stall raised no hung incident of its own"),
        (
            "merged",
            "repeat: the first hung incident inc-hung-0 absorbed the second stall's",
        ),
        ("duplicate", "repeat_merged: 2 hung incidents for one stall wave"),
        ("shared", "repeat: the second hung incident has no workflow of its own: wf"),
        ("unchained", "repeat: the second hung workflow wf-2 follows None, not"),
    ],
)
def test_a_suppressed_merged_or_duplicated_second_episode_fails(
    tmp_path: Path, second: str, message: str
) -> None:
    result = REPEAT.execute(_repeat(tmp_path, RepeatProduct(second)))
    assert result["verdict"] == "FAIL"
    assert any(item.startswith(message) for item in result["errors"]), result["errors"]
    assert len(FakeNcclJob.instances[-1].stall_times) == 2


@pytest.mark.parametrize(
    ("resumed", "fades_after", "message"),
    [(300_000.0, None, "did not resume"), (None, 10, "fell to 0 B/s")],
)
def test_a_recovery_that_is_not_proven_is_inconclusive_and_never_stalls_again(
    tmp_path: Path, resumed: float | None, fades_after: int | None, message: str
) -> None:
    FakeNcclJob.traffic.resumed = resumed
    FakeNcclJob.traffic.fades_after = fades_after
    ctx = _repeat(tmp_path)
    with pytest.raises(VariantInconclusive, match=f"rearm window: .*{message}"):
        REPEAT.execute(ctx)
    job = FakeNcclJob.instances[-1]
    assert len(job.stall_times) == 1 and job.released == 1, "no second stall"


def test_a_failed_first_episode_fails_without_a_second_stall(tmp_path: Path) -> None:
    result = REPEAT.execute(_repeat(tmp_path, RepeatProduct(first=False)))
    assert result["verdict"] == "FAIL"
    assert (
        "repeat: not attempted: the first stall's hung chain failed"
        in (result["errors"])
    )
    job = FakeNcclJob.instances[-1]
    assert len(job.stall_times) == 1 and job.released == 1, "released, not repeated"


# --------------------------------------------------------------------------- #
# broken_pair: the pair broken inside the host collector around a real stall
# --------------------------------------------------------------------------- #
PAIR_DEV = "rdmap79s0"
PAIR_DEST = f"/sys/class/infiniband/{PAIR_DEV}/ports/1/hw_counters/tx_bytes"
PAIR_INVENTORY = {
    "root": "/sys/class/infiniband",
    "devices": {
        PAIR_DEV: {
            "driver": "efa",
            "uevent_drivers": ["efa"],
            "efa": True,
            "netdevs": [],
            "ports": {
                "1": {
                    "state": "4: ACTIVE",
                    "phys_state": "5: LinkUp",
                    "counters": {},
                    "hw_counters": {
                        "rx_bytes": "100",
                        "tx_bytes": "200",
                        "rx_drops": "0",
                    },
                }
            },
        }
    },
    "physical_netdevs": ["ens5"],
    "ethtool": {"path": None, "statistics": {}},
}


def _pair_ctx(
    tmp_path: Path,
    *,
    warn_after: float = 90.0,
    warn_while_broken: bool = False,
    warn_never: bool = False,
    invalid: bool = True,
) -> tuple[Any, FakeProbe, dict[str, Any]]:
    spec = SPECS["broken_pair"]
    state: dict[str, Any] = {}
    ctx_box: dict[str, Any] = {}

    def warning(created: datetime, node: str = NODE) -> dict[str, Any]:
        return {
            "incident_id": f"inc-warn-{node}",
            "event_id": f"host-{node}-77{nc.WARNING_SUFFIX}",
            "event_type": "NODE_HEALTH",
            "state": "RECOVERED",
            "node_ids": [node],
            "reasons": [
                "EFA/RDMA traffic remains at zero for an active training attempt"
            ],
            "workflow_request_id": "wf-warn",
            "created_at": created.isoformat(),
        }

    def respond(script: str, *arguments: str) -> dict[str, Any]:
        if script == ACTIVITY_PROBE:
            samples = [
                {
                    "name": "efa_traffic_observation_valid",
                    "device": None,
                    "value": 0 if invalid else 1,
                    "labels": {},
                }
            ]
            errors = [
                f"_rdma: HostObservationError: {PAIR_DEV}/1: incomplete traffic counter pair (invalid counter at {PAIR_DEST})"
            ]
            if not invalid:
                samples.append(
                    {
                        "name": "efa_traffic_bytes_per_second",
                        "device": None,
                        "value": 0.0,
                        "labels": {},
                    }
                )
                errors = []
            return {
                "incidents": [],
                "workflows": [],
                "batches": [
                    {
                        "observed_at": "2026-10-24T00:05:00+00:00",
                        "batch_id": "host-b1",
                        "edge_filter_reasons": [],
                        "collection_errors": errors,
                        "samples": samples,
                    }
                ],
                "markers": [],
                "commands": [],
                "health_signals": {},
                "notifications": [],
                "collector_statuses": [],
                "metric_latest": [],
            }
        # The peer's pair is real and its zero is graded meanwhile: its own
        # ZERO_WARNING is never this node's streak.
        peer = [warning(ctx_box["ctx"].clock(), PEER)]
        if warn_never:
            return {"incidents": peer, "workflows": {}}
        if "closed_at" in state:
            return {
                "incidents": [
                    *peer,
                    warning(state["closed_at"] + timedelta(seconds=warn_after)),
                ],
                "workflows": {},
            }
        if warn_while_broken:
            return {
                "incidents": [*peer, warning(ctx_box["ctx"].clock())],
                "workflows": {},
            }
        return {"incidents": peer, "workflows": {}}

    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=respond),
        probes={},
        settings=_settings(spec, tmp_path),
    )
    ctx_box["ctx"] = ctx
    start = datetime(2026, 10, 24, tzinfo=timezone.utc)
    ticks = itertools.count()
    ctx.clock = lambda: start + timedelta(seconds=15 * next(ticks))
    ctx.probes[f"node@{NODE}"] = efa_probe(NODE, survey=True)  # type: ignore[assignment]
    ctx.probes[f"node@{PEER}"] = efa_probe(PEER)  # type: ignore[assignment]

    def close(_arguments: tuple[str, ...]) -> dict[str, Any]:
        state["closed_at"] = ctx.clock()
        return {"clean": True}

    def read_source(arguments: tuple[str, ...]) -> dict[str, Any]:
        view = {"readable": True, "value": "200", "device": "21"}
        return {
            "path": arguments[2],
            "unit": view,
            "host": dict(view),
            "same": True,
            "window_open": False,
        }

    source = FakeProbe(
        {
            "inventory": PAIR_INVENTORY,
            "install-source": {"bytes": 7},
            "open-window": {
                "after": {"ActiveState": "active"},
                "proof_scope": "live-replay",
                "verification": {"ok": True},
            },
            "close-window": close,
            "read-source": read_source,
        }
    )
    ctx.probes[f"source-replay@{NODE}"] = source  # type: ignore[assignment]
    return ctx, source, state


def test_broken_pair_starts_the_streak_only_on_real_readings(tmp_path: Path) -> None:
    ctx, source, state = _pair_ctx(tmp_path)
    result = SPECS["broken_pair"].execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert (
        result["proof_scope"] == "live-replay"
        and result["target"]["counter"] == "tx_bytes"
    )
    assert ("install-source",) == source.calls[1][:1] and source.calls[1][
        2
    ] == "efa-tx-bytes"
    opened = source.calls[2]
    assert opened[0] == "open-window" and f"efa-tx-bytes:{PAIR_DEST}" in opened, opened
    order = [call[0] for call in source.calls]
    assert order.index("close-window") < order.index("read-source")
    assert "closed_at" in state
    job = FakeNcclJob.instances[-1]
    assert job.stalled and job.released == 1, (
        "the job was stalled while broken and released after"
    )
    assert result["stages"]["settled"] == [] and result["stages"]["resumed"] == []
    ctx.cleanup.run()
    assert ctx.cleanup.failed == []


def test_broken_pair_fails_on_a_suspicion_while_broken(tmp_path: Path) -> None:
    ctx, _source, _state = _pair_ctx(tmp_path, warn_while_broken=True)
    result = SPECS["broken_pair"].execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any("no_streak_while_broken" in error for error in result["errors"]), result[
        "errors"
    ]


def test_broken_pair_fails_when_the_warning_comes_too_early_or_never(
    tmp_path: Path,
) -> None:
    ctx, _source, _state = _pair_ctx(tmp_path, warn_after=20)
    result = SPECS["broken_pair"].execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any(
        "warning_after_restore" in e and "streak began on the broken pair" in e
        for e in result["errors"]
    ), result["errors"]
    ctx, _source, _state = _pair_ctx(tmp_path, warn_never=True)
    result = SPECS["broken_pair"].execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any("real zero was not graded" in error for error in result["errors"]), (
        result["errors"]
    )


def test_broken_pair_fails_when_the_product_reads_the_broken_pair_as_a_rate(
    tmp_path: Path,
) -> None:
    ctx, _source, _state = _pair_ctx(tmp_path, invalid=False)
    result = SPECS["broken_pair"].execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any("observation_invalid" in e for e in result["errors"]), result["errors"]
    assert any("no_rate_from_broken_pair" in e for e in result["errors"]), result[
        "errors"
    ]


def test_broken_pair_is_inconclusive_when_the_pair_keeps_moving(tmp_path: Path) -> None:
    ctx, source, _state = _pair_ctx(tmp_path)
    FakeNcclJob.traffic.stalled = ACTIVE_BPS
    with pytest.raises(
        VariantInconclusive, match="zero EFA traffic while the pair was broken"
    ):
        SPECS["broken_pair"].execute(ctx)
    ctx.cleanup.run()
    assert [call[0] for call in source.calls][-1] == "close-window", (
        "the LIFO cleanup still closes the window"
    )
    assert ctx.cleanup.failed == []
