"""GF-GENERAL-LIVE-001 injector/verdict contract over scripted product records."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.host_health import NodeHealthPolicy
from scripts.e2e.regional import general_live_host as host
from scripts.e2e.regional import general_live_progress as progress
from scripts.e2e.regional.general_live_common import STORE_POLL_SECONDS
from scripts.e2e.regional.general_live_fixture import VariantInconclusive, VariantNotRun
from scripts.e2e.regional.general_live_progress_probes import TRAINING_PROBE
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    host_batch,
    incident,
    make_context,
    store_poll_clock,
    store_script,
    workflow,
)

SPECS = {spec.variant: spec for spec in host.VARIANTS}
NODE = "ip-10-0-43-5.us-west-2.compute.internal"


def survey_response(cpu_count: int = 8) -> dict[str, Any]:
    return {
        "cpu_count": cpu_count,
        "tools": {"systemd-run": True, "systemctl": True},
        "modules": {},
        "backing": [],
        "collector_env": {
            "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
            "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "300",
        },
        "residue": {"units": [], "paused_units": {}, "paused_processes": {}},
    }


def oracle_records(
    value: float, count: int = 40, key: str = "cpu_busy_percent"
) -> dict[str, Any]:
    records = [{key: value} for _ in range(count)]
    return {"records": records, "next_offset": count, "total": count}


def busy_node_probe(busy: float = 99.0) -> FakeProbe:
    return FakeProbe(
        {
            "survey": survey_response(),
            "cpu-start": {"unit": "gfrs-x-cpu"},
            "cpu-stop": {"stopped": {"ActiveState": "inactive"}},
            "oracle-start": {"unit": "gfrs-x-oracle"},
            "oracle-stop": {"stopped": {}},
            "oracle-read": [
                oracle_records(busy),
                {"records": [], "next_offset": 40, "total": 40},
            ],
        }
    )


def chain_activity(
    metric: str, *, device: str = "node", rule_id: str | None = None
) -> dict[str, Any]:
    inc = incident(
        "inc-1", metric, device=device, rule_id=rule_id, state="ACTION_PENDING"
    )
    wf = workflow(
        "wf-1",
        "inc-1",
        ["FREEZE_EVIDENCE", "VALIDATE_HOST"],
        status="FAILED",
        probe_metric=metric,
        executions=[
            {
                "operation": "VALIDATE_HOST",
                "status": "FAILED",
                "details": {"host_diagnostic_failure": True},
            }
        ],
    )
    # One document that satisfies detection, the diagnostic chain, recovery and
    # settlement at once: an ACTION_PENDING incident whose terminal (FAILED)
    # workflow executed VALIDATE_HOST as a host_diagnostic_failure, a breach
    # batch and three fresh below-threshold batches.
    batches = [host_batch("2026-10-24T00:02:00+00:00", metric, 99.0, device=None)]
    batches += [
        host_batch(f"2026-10-24T00:2{index}:00+00:00", metric, 1.0, device=None)
        for index in range(3)
    ]
    return {
        "incidents": [inc],
        "workflows": [wf],
        "batches": batches,
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
    }


def test_cpu_busy_passes_the_diagnostic_chain(tmp_path: Path) -> None:
    spec = SPECS["cpu_busy"]
    regional = FakeRegional(
        cpu_python=store_script({"activity": chain_activity("cpu_usage_percent")})
    )
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"node@{NODE}": busy_node_probe()},
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert (ctx.case_dir / "oracle.jsonl").is_file(), (
        "(ctx.case_dir / 'oracle.jsonl').is_file()"
    )


def test_cpu_busy_is_inconclusive_when_the_oracle_never_crossed(tmp_path: Path) -> None:
    spec = SPECS["cpu_busy"]
    probe = busy_node_probe()
    probe.script["oracle-read"] = [oracle_records(70.0)]
    regional = FakeRegional(
        cpu_python=store_script({"activity": chain_activity("cpu_usage_percent")})
    )
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={f"node@{NODE}": probe}
    )
    # Never a verdict on a load the oracle did not see cross the threshold,
    # whatever the product recorded: the ramp wait runs out first.
    with pytest.raises(VariantInconclusive, match="condition not established"):
        spec.execute(store_poll_clock(ctx))
    assert not [call for call in probe.calls if call[0] == "cpu-stop"][1:], (
        "the load is stopped once, by the cleanup stack"
    )


def test_cpu_busy_fails_when_a_hardware_step_is_planned(tmp_path: Path) -> None:
    spec = SPECS["cpu_busy"]
    document = chain_activity("cpu_usage_percent")
    document["workflows"][0]["official_steps"].append(
        {"operation": "RESET_GPU", "parameters": {}}
    )
    regional = FakeRegional(cpu_python=store_script({"activity": document}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"node@{NODE}": busy_node_probe()},
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any("RESET_GPU" in error for error in result["errors"]), (
        "any('RESET_GPU' in error for error in result['errors'])"
    )


def test_cpu_busy_fails_when_validate_host_succeeded_while_loaded(
    tmp_path: Path,
) -> None:
    spec = SPECS["cpu_busy"]
    document = chain_activity("cpu_usage_percent")
    document["workflows"][0]["step_executions"] = [
        {"operation": "VALIDATE_HOST", "status": "SUCCEEDED", "details": {}}
    ]
    regional = FakeRegional(cpu_python=store_script({"activity": document}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"node@{NODE}": busy_node_probe()},
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any("succeeded while the fault" in error for error in result["errors"]), (
        "any('succeeded while the fault' in error for error in result['errors']"
    )


def test_load_average_passes_and_binds_its_own_signal(tmp_path: Path) -> None:
    spec = SPECS["load_average"]
    probe = busy_node_probe()
    probe.script["oracle-read"] = [
        oracle_records(2.4, key="load1_per_cpu"),
        {"records": [], "next_offset": 40, "total": 40},
    ]
    document = chain_activity("load1_per_cpu", rule_id=None)
    regional = FakeRegional(cpu_python=store_script({"activity": document}))
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={f"node@{NODE}": probe}
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_short_burst_passes_when_no_finding_appears(tmp_path: Path) -> None:
    spec = SPECS["short_burst"]
    empty = {
        "incidents": [],
        "workflows": [],
        "batches": [host_batch("2026-10-24T00:01:00+00:00", "cpu_usage_percent", 99.0)],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
    }
    probe = busy_node_probe()
    probe.script["oracle-read"] = [oracle_records(99.0, count=12)]
    regional = FakeRegional(cpu_python=store_script({"activity": empty}))
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={f"node@{NODE}": probe}
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_short_burst_fails_when_a_finding_appears(tmp_path: Path) -> None:
    spec = SPECS["short_burst"]
    probe = busy_node_probe()
    probe.script["oracle-read"] = [oracle_records(99.0, count=12)]
    regional = FakeRegional(
        cpu_python=store_script({"activity": chain_activity("cpu_usage_percent")})
    )
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={f"node@{NODE}": probe}
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"


class FakePhaseJob:
    """The managed progress job double: what the runner wrote, no cluster."""

    created: list["FakePhaseJob"] = []

    def __init__(self, ctx: Any, *, name: str, **_: Any) -> None:
        self.name = name
        self.attempt_id = f"{name}-a001"
        self.writes: list[tuple[int, dict[str, Any]]] = []
        self.deleted = False
        FakePhaseJob.created.append(self)

    def start(self) -> dict[str, Any]:
        return {}

    def identity(self) -> dict[str, Any]:
        return {"job_id": self.name, "attempt_id": self.attempt_id}

    def write_progress(self, rank: int, **fields: Any) -> dict[str, Any]:
        self.writes.append((rank, fields))
        return {}

    def delete(self) -> None:
        self.deleted = True


@pytest.fixture
def phase_job(monkeypatch: pytest.MonkeyPatch) -> type[FakePhaseJob]:
    FakePhaseJob.created = []
    monkeypatch.setattr(progress, "ManagedProgressJob", FakePhaseJob)
    return FakePhaseJob


#: The attempt's training records once the reporter posted the declared phase.
CHECKPOINTING = {
    "progress_states": [
        {
            "rank": 0,
            "step": 100,
            "labels": {"phase": "CHECKPOINT_SAVE"},
            "observed_at": "2026-10-24T00:20:00+00:00",
        }
    ],
    "observation": {"workload_phase": "RUNNING"},
    "incidents": [],
    "workflows": {},
    "decision": None,
    "notifications": [],
}


def with_training(
    activity: Any, training: Any = CHECKPOINTING
) -> Any:  # a cpu_python double
    """``activity`` answers the node-activity reads, ``training`` (a document,
    or a list consumed per read) the attempt's TRAINING_PROBE reads."""

    reads = {"training": 0}

    def respond(script: str, *arguments: str, **options: Any) -> dict[str, Any]:
        if script is TRAINING_PROBE:
            documents = training if isinstance(training, list) else [training]
            index = min(reads["training"], len(documents) - 1)
            reads["training"] += 1
            return dict(documents[index])
        return activity(script, *arguments, **options)

    return respond


def test_recovery_and_checkpoint_phase_pass(
    tmp_path: Path, phase_job: type[FakePhaseJob]
) -> None:
    for variant in ("recovery", "checkpoint_phase"):
        spec = SPECS[variant]
        regional = FakeRegional(
            cpu_python=with_training(
                store_script({"activity": chain_activity("cpu_usage_percent")})
            )
        )
        ctx = make_context(
            spec,
            tmp_path=tmp_path,
            regional=regional,
            probes={f"node@{NODE}": busy_node_probe()},
        )
        result = spec.execute(ctx)
        assert result["verdict"] == "PASS", (variant, result["errors"])
    assert any("phase correlation" in note for note in result["notes"]), (
        "any('phase correlation' in note for note in result['notes'])"
    )
    (job,) = phase_job.created
    assert job.writes == [
        (0, {"step": 100, "loss": 1.0, "labels": {"phase": "CHECKPOINT_SAVE"}})
    ], "the attempt declares the checkpoint phase before the load"


def checkpoint_ctx(tmp_path: Path, training: Any, activity: Any = None) -> Any:
    spec = SPECS["checkpoint_phase"]
    activity = activity or store_script(
        {"activity": chain_activity("cpu_usage_percent")}
    )
    return make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=with_training(activity, training)),
        probes={f"node@{NODE}": busy_node_probe()},
    )


def test_checkpoint_phase_never_declared_is_inconclusive_before_any_load(
    tmp_path: Path, phase_job: type[FakePhaseJob]
) -> None:
    # The product never shows the phase: the stimulus did not reach it (the
    # old variant declared nothing at all and passed on the cpu chain).
    undeclared = {**CHECKPOINTING, "progress_states": []}
    ctx = checkpoint_ctx(tmp_path, undeclared)
    with pytest.raises(VariantInconclusive, match="never recorded"):
        SPECS["checkpoint_phase"].execute(ctx)
    calls = [call[0] for call in ctx.probes[f"node@{NODE}"].calls]
    assert "cpu-start" not in calls, calls


@pytest.mark.parametrize(
    ("after", "error"),
    [
        (
            {**CHECKPOINTING, "observation": {"workload_phase": "STOPPED"}},
            "the checkpointing attempt is STOPPED, not RUNNING",
        ),
        (
            {
                **CHECKPOINTING,
                "progress_states": [
                    {**CHECKPOINTING["progress_states"][0], "labels": {}}
                ],
            },
            "the attempt no longer declares CHECKPOINT_SAVE",
        ),
        (
            {
                **CHECKPOINTING,
                "progress_states": [
                    {
                        **CHECKPOINTING["progress_states"][0],
                        "observed_at": "2026-10-24T00:01:00+00:00",
                    }
                ],
            },
            "the attempt's last CHECKPOINT_SAVE heartbeat predates the cpu finding",
        ),
    ],
    ids=["stopped", "phase-dropped", "stale"],
)
def test_checkpoint_phase_fails_when_the_phase_was_not_held_through_the_finding(
    tmp_path: Path, phase_job: type[FakePhaseJob], after: Any, error: str
) -> None:
    ctx = checkpoint_ctx(tmp_path, [CHECKPOINTING, after])
    result = SPECS["checkpoint_phase"].execute(ctx)
    assert result["verdict"] == "FAIL", result
    assert result["stages"]["phase_held"] == [error], result["stages"]


def test_checkpoint_phase_fails_when_the_cpu_chain_stops_the_attempt(
    tmp_path: Path, phase_job: type[FakePhaseJob]
) -> None:
    stopped = chain_activity("cpu_usage_percent")
    stopped["workflows"][0]["official_steps"].insert(
        1, {"operation": "STOP_WORKLOADS", "parameters": {}}
    )
    ctx = checkpoint_ctx(tmp_path, CHECKPOINTING, store_script({"activity": stopped}))
    result = SPECS["checkpoint_phase"].execute(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("STOP_WORKLOADS" in item for item in result["stages"]["chain"]), result[
        "stages"
    ]


def test_checkpoint_phase_without_a_site_file_is_not_run(tmp_path: Path) -> None:
    # The real job: a managed progress job needs the site file (the recipe
    # must pass --site-file); nothing is created before the refusal.
    ctx = checkpoint_ctx(tmp_path, CHECKPOINTING)
    with pytest.raises(VariantNotRun, match="--site-file"):
        SPECS["checkpoint_phase"].execute(ctx)


def test_invalid_delta_is_an_implemented_live_replay_variant() -> None:
    spec = SPECS["invalid_delta"]
    assert spec.proof_mode == "live-replay", "invalid_delta is a live-replay variant"
    assert spec.implemented, (
        "invalid_delta is now implemented by the /proc/stat source bind"
    )


# --------------------------------------------------------------------------- #
# 001 invalid_delta: a reviewed static /proc/stat bind -> no busy sample
# (lane K live-replay). A source-replay FakeProbe stands in for the bind host.
# --------------------------------------------------------------------------- #
def _source_probe() -> FakeProbe:
    return FakeProbe(
        {
            "install-source": {"name": "proc-stat"},
            "open-window": {
                "after": {"ActiveState": "active"},
                "proof_scope": "live-replay",
            },
            "status": {"fixtures": {"proc-stat": "cpu  100 0 50 1000 0 0 0 0 0 0"}},
            "close-window": {"clean": True},
        }
    )


def _invalid_delta_probes() -> dict[str, FakeProbe]:
    return {
        f"node@{NODE}": FakeProbe({"survey": survey_response()}),
        f"source-replay@{NODE}": _source_probe(),
    }


def _empty_doc() -> dict[str, Any]:
    return {
        "incidents": [],
        "workflows": [],
        "batches": [],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "metric_latest": [],
        "notifications": [],
        "collector_statuses": [],
    }


def test_invalid_delta_emits_no_busy_sample(tmp_path: Path) -> None:
    spec = SPECS["invalid_delta"]
    regional = FakeRegional(cpu_python=store_script({"activity": _empty_doc()}))
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes=_invalid_delta_probes()
    )
    result = host.invalid_delta(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["proof_scope"] == "live-replay", result


def test_invalid_delta_fails_if_a_busy_sample_appears(tmp_path: Path) -> None:
    spec = SPECS["invalid_delta"]
    doc = _empty_doc()
    doc["batches"] = [
        host_batch("2026-10-24T00:03:00+00:00", "cpu_usage_percent", 99.0)
    ]
    regional = FakeRegional(cpu_python=store_script({"activity": doc}))
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes=_invalid_delta_probes()
    )
    result = host.invalid_delta(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("cpu_usage_percent samples" in error for error in result["errors"]), (
        result["errors"]
    )


# --------------------------------------------------------------------------- #
# Detection timed from the oracle's crossing over two host health summaries
# (the 003 high_page_cache lesson of 2026-10-25, applied to the 120-s cpu rules)
# --------------------------------------------------------------------------- #
SITE = SimpleNamespace(host_summary=300, host_interval=15)


def empty_activity() -> dict[str, Any]:
    return {
        "incidents": [],
        "workflows": [],
        "batches": [],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
    }


def oracle_reads(
    *phases: list[float], key: str = "cpu_busy_percent"
) -> list[dict[str, Any]]:
    """One ``oracle-read`` answer per phase: the readings since the last read."""

    entries: list[dict[str, Any]] = []
    total = 0
    for values in phases:
        total += len(values)
        entries.append(
            {
                "records": [{key: value} for value in values],
                "next_offset": total,
                "total": total,
            }
        )
    return entries


#: The oracle reads 99 % on its third read (the SCHED_IDLE threads took two
#: polls to be scheduled); the product's records -- the finding and its
#: executed VALIDATE_HOST -- appear 40 store polls, 800 s, after the crossing:
#: the edge batch, the first summary at least 120 s later (up to 450 s after
#: the edge) that mints the finding, and the next summary the validation is
#: judged on. The old 510-s budget, counted from the load's start, ended
#: before that second summary while the load was already being stopped.
CPU_RAMP_READS = oracle_reads([70.0] * 3, [70.0] * 3, [99.0] * 24, [99.0] * 60)
LOAD_RAMP_READS = oracle_reads(
    [0.4] * 3, [1.5] * 3, [2.4] * 3, [2.4] * 60, key="load1_per_cpu"
)
MINTED_AT_POLL = 40


def ramp_run(
    tmp_path: Path,
    variant: str,
    oracle: list[dict[str, Any]],
    minted_at_poll: int | None,
    *,
    metric: str = "cpu_usage_percent",
) -> tuple[dict[str, Any], Any, list[tuple[str, ...]]]:
    spec = SPECS[variant]
    polls = itertools.count(1)

    def product(_script: str, *_arguments: str, **_: Any) -> dict[str, Any]:
        poll = next(polls)
        if minted_at_poll is not None and poll >= minted_at_poll:
            return chain_activity(metric)
        return empty_activity()

    probe = busy_node_probe()
    probe.script["oracle-read"] = oracle
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=with_training(product)),
        probes={f"node@{NODE}": probe},
    )
    result = spec.execute(store_poll_clock(ctx))
    return result, ctx, probe.calls


@pytest.mark.parametrize("variant", ["cpu_busy", "recovery", "checkpoint_phase"])
@pytest.mark.usefixtures("phase_job")
def test_cpu_variants_time_detection_from_the_oracles_crossing(
    tmp_path: Path, variant: str
) -> None:
    result, ctx, calls = ramp_run(tmp_path, variant, CPU_RAMP_READS, MINTED_AT_POLL)
    assert result["verdict"] == "PASS", (variant, result["errors"])
    ramp = json.loads((ctx.case_dir / "cpu-ramp-timeline.json").read_text())
    assert [entry["accepted"] for entry in ramp["entries"]] == [False, False, True]
    started = json.loads((ctx.case_dir / "cpu-start.json").read_text())
    assert started["detection_budget_seconds"] == host.detection_budget(ctx) == 915
    assert started["ramp_budget_seconds"] == host.CPU_RAMP_BUDGET
    assert started["sustain_seconds"] == host.SUSTAIN_SECONDS
    assert (started["host_summary_seconds"], started["host_interval_seconds"]) == (
        300,
        15,
    )
    # The load and the oracle outlive the ramp and the detection budget.
    for subcommand in ("cpu-start", "oracle-start"):
        call = next(item for item in calls if item[0] == subcommand)
        seconds = int(call[call.index("--seconds") + 1])
        assert seconds >= host.CPU_RAMP_BUDGET + host.detection_budget(ctx), (
            subcommand,
            seconds,
        )


def test_cpu_busy_fails_when_the_product_misses_two_summaries(tmp_path: Path) -> None:
    # Minted only after the cadence-sized budget (the fake survey's 300-s
    # summaries, 15-s interval): the product's FAIL, not the runner's budget.
    late = host.detection_budget(SITE) // STORE_POLL_SECONDS + 3
    result, _ctx, _calls = ramp_run(tmp_path, "cpu_busy", CPU_RAMP_READS, late)
    assert result["verdict"] == "FAIL", result
    assert result["stages"]["chain"] == [
        "no cpu_usage_percent finding reached an incident"
    ], result["stages"]


def test_cpu_busy_is_inconclusive_when_the_ramp_runs_out(tmp_path: Path) -> None:
    with pytest.raises(VariantInconclusive, match="condition not established"):
        ramp_run(tmp_path, "cpu_busy", oracle_reads([70.0] * 3, [71.0] * 3), None)


def test_load_average_times_detection_from_the_load1_crossing(tmp_path: Path) -> None:
    result, ctx, calls = ramp_run(
        tmp_path,
        "load_average",
        LOAD_RAMP_READS,
        MINTED_AT_POLL,
        metric="load1_per_cpu",
    )
    assert result["verdict"] == "PASS", result["errors"]
    ramp = json.loads((ctx.case_dir / "load-ramp-timeline.json").read_text())
    assert [entry["accepted"] for entry in ramp["entries"]] == [False, False, True]
    started = json.loads((ctx.case_dir / "cpu-start.json").read_text())
    assert started["ramp_budget_seconds"] == host.LOAD_RAMP_BUDGET
    call = next(item for item in calls if item[0] == "cpu-start")
    seconds = int(call[call.index("--seconds") + 1])
    assert seconds >= host.LOAD_RAMP_BUDGET + host.detection_budget(ctx), seconds


def test_load_average_is_inconclusive_when_load1_never_converges(
    tmp_path: Path,
) -> None:
    reads = oracle_reads([0.4] * 3, [1.5] * 3, key="load1_per_cpu")
    with pytest.raises(VariantInconclusive, match="load1_per_cpu to 2"):
        ramp_run(
            tmp_path, "load_average", reads, MINTED_AT_POLL, metric="load1_per_cpu"
        )


def test_short_burst_watches_the_sustain_plus_one_summary_after_the_burst(
    tmp_path: Path,
) -> None:
    spec = SPECS["short_burst"]
    probe = busy_node_probe()
    probe.script["oracle-read"] = [oracle_records(99.0, count=12)]
    regional = FakeRegional(cpu_python=store_script({"activity": empty_activity()}))
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={f"node@{NODE}": probe}
    )
    result = spec.execute(store_poll_clock(ctx))
    assert result["verdict"] == "PASS", result["errors"]
    window = host.quiet_window(ctx)
    assert result["negative_window_seconds"] == window == 120 + 300 + 15 + 180
    assert result["host_summary_seconds"] == 300
    # The watch is bounded and, after the burst, really runs its window: the
    # recovered edge and at least one health summary have been seen.
    watched = json.loads((ctx.case_dir / "negative-window-timeline.json").read_text())
    assert len(watched["entries"]) >= window // STORE_POLL_SECONDS, len(
        watched["entries"]
    )
    calls = [call[0] for call in probe.calls]
    assert calls.index("cpu-stop") < calls.index("oracle-read"), (
        "the burst is over before the negative watch begins"
    )


def test_host_budgets_follow_the_products_rule_and_fit_the_variant_windows() -> None:
    assert host.SUSTAIN_SECONDS == NodeHealthPolicy.HOST_RESOURCE_SUSTAIN_SECONDS
    budget = host.detection_budget(SITE)
    assert budget == host.SUSTAIN_SECONDS + 2 * 300 + 15 + 180 == 915
    assert budget > 120 + 300 + 90, "the fixed budget that ran out live"
    recovery = 2 * SITE.host_summary + 6 * SITE.host_interval
    # checkpoint_phase first starts its managed job (the 900-s start wait)
    # and waits for the declared phase.
    for variant, ramp, tail in (
        ("cpu_busy", host.CPU_RAMP_BUDGET, 240),
        ("recovery", host.CPU_RAMP_BUDGET, 60),
        (
            "checkpoint_phase",
            host.CPU_RAMP_BUDGET,
            120 + 900 + host.PHASE_DECLARE_BUDGET,
        ),
        ("load_average", host.LOAD_RAMP_BUDGET, 240),
    ):
        assert SPECS[variant].max_minutes * 60 >= ramp + budget + recovery + tail, (
            variant,
            SPECS[variant].max_minutes,
        )
    window = host.quiet_window(SITE)
    assert SPECS["short_burst"].max_minutes * 60 >= host.SHORT_BURST_SECONDS + 10 + (
        window + 120
    )
    assert SPECS["invalid_delta"].max_minutes * 60 >= 4 * 15 + 30 + window + 120
