"""GF-GENERAL-LIVE-002 injector/verdict contract over scripted product records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_memory as mem
from scripts.e2e.regional.general_live_fixture import (
    VariantBlocked,
    VariantInconclusive,
    VariantNotRun,
)
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    host_batch,
    incident,
    make_context,
    make_settings,
    store_script,
    workflow,
)

SPECS = {spec.variant: spec for spec in mem.VARIANTS}
NODE = "ip-10-0-43-5.us-west-2.compute.internal"


def survey_response(swap_total: int = 0) -> dict[str, Any]:
    return {
        "cpu_count": 8,
        "tools": {"systemd-run": True, "systemctl": True},
        "modules": {},
        "backing": [],
        "memory": {"MemTotal": 2_000_000_000_000, "SwapTotal": swap_total},
        "collector_env": {
            "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
            "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "300",
        },
        "residue": {"units": [], "paused_units": {}, "paused_processes": {}},
    }


def oracle_records(value: float, count: int = 40) -> dict[str, Any]:
    records = [{"memory_used_percent": value} for _ in range(count)]
    return {"records": records, "next_offset": count, "total": count}


def mem_node_probe(used: float = 97.0) -> FakeProbe:
    return FakeProbe(
        {
            "survey": survey_response(),
            "memory-start": {"unit": "gfrs-x-memory"},
            "memory-stop": {"stopped": {"ActiveState": "inactive"}},
            "oracle-start": {"unit": "gfrs-x-oracle"},
            "oracle-stop": {"stopped": {}},
            "oracle-read": [
                oracle_records(used),
                {"records": [], "next_offset": 40, "total": 40},
            ],
        }
    )


def chain_activity(metric: str = "memory_used_percent") -> dict[str, Any]:
    inc = incident("inc-1", metric, state="ACTION_PENDING")
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
    batches = [host_batch("2026-10-24T00:02:00+00:00", metric, 99.0)]
    batches += [
        host_batch(f"2026-10-24T00:2{index}:00+00:00", metric, 50.0)
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


def test_memory_pressure_passes_the_diagnostic_chain(tmp_path: Path) -> None:
    spec = SPECS["memory_pressure"]
    regional = FakeRegional(cpu_python=store_script({"activity": chain_activity()}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"node@{NODE}": mem_node_probe()},
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert (ctx.case_dir / "oracle.jsonl").is_file(), (
        "(ctx.case_dir / 'oracle.jsonl').is_file()"
    )


def test_memory_pressure_needs_the_approval(tmp_path: Path) -> None:
    spec = SPECS["memory_pressure"]
    settings = make_settings(spec, node=NODE)
    object.__setattr__(settings, "approvals", ())
    regional = FakeRegional(cpu_python=store_script({"activity": chain_activity()}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"node@{NODE}": mem_node_probe()},
        settings=settings,
    )
    with pytest.raises(VariantNotRun, match="HOST_MEMORY_PRESSURE"):
        spec.execute(ctx)


def test_memory_pressure_is_inconclusive_when_the_oracle_never_crossed(
    tmp_path: Path,
) -> None:
    spec = SPECS["memory_pressure"]
    probe = mem_node_probe()
    probe.script["oracle-read"] = [oracle_records(70.0)]
    regional = FakeRegional(cpu_python=store_script({"activity": chain_activity()}))
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={f"node@{NODE}": probe}
    )
    # Live 2026-10-24: a 2 TiB host was still at 76 % when detection timed
    # out; the ramp is now waited for (and named) before detection is timed.
    with pytest.raises(VariantInconclusive, match="did not raise memory_used"):
        spec.execute(ctx)


def test_memory_pressure_fails_when_a_hardware_step_is_planned(tmp_path: Path) -> None:
    spec = SPECS["memory_pressure"]
    document = chain_activity()
    document["workflows"][0]["official_steps"].append(
        {"operation": "RESET_GPU", "parameters": {}}
    )
    regional = FakeRegional(cpu_python=store_script({"activity": document}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"node@{NODE}": mem_node_probe()},
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any("RESET_GPU" in error for error in result["errors"]), result["errors"]


def test_memory_pressure_fails_when_validate_host_succeeded_while_loaded(
    tmp_path: Path,
) -> None:
    spec = SPECS["memory_pressure"]
    document = chain_activity()
    document["workflows"][0]["step_executions"] = [
        {"operation": "VALIDATE_HOST", "status": "SUCCEEDED", "details": {}}
    ]
    regional = FakeRegional(cpu_python=store_script({"activity": document}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"node@{NODE}": mem_node_probe()},
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any("succeeded while the fault" in error for error in result["errors"]), (
        result["errors"]
    )


def test_recovery_passes_and_settles(tmp_path: Path) -> None:
    spec = SPECS["recovery"]
    regional = FakeRegional(cpu_python=store_script({"activity": chain_activity()}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"node@{NODE}": mem_node_probe()},
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_no_swap_passes_on_a_swapless_node(tmp_path: Path) -> None:
    spec = SPECS["no_swap"]
    empty = {
        "incidents": [],
        "workflows": [],
        "batches": [host_batch("2026-10-24T00:01:00+00:00", "swap_used_percent", 0.0)],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
        "metric_latest": [],
    }
    probe = FakeProbe({"survey": survey_response(swap_total=0)})
    regional = FakeRegional(cpu_python=store_script({"activity": empty}))
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={f"node@{NODE}": probe}
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_no_swap_blocks_when_the_node_has_swap(tmp_path: Path) -> None:
    spec = SPECS["no_swap"]
    probe = FakeProbe({"survey": survey_response(swap_total=8_000_000_000)})
    regional = FakeRegional(cpu_python=store_script({"activity": {}}))
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={f"node@{NODE}": probe}
    )
    with pytest.raises(VariantBlocked, match="swapless"):
        spec.execute(ctx)


# --------------------------------------------------------------------------- #
# cgroup_only: an OOM inside a small test cgroup is not host pressure
# --------------------------------------------------------------------------- #
def oom_status(
    result: str = "oom-kill", state: str = "failed", used: float = 60.0
) -> dict[str, Any]:
    return {
        "unit": {
            "ActiveState": state,
            "Result": result,
            "ExecMainCode": "killed",
            "ExecMainStatus": "9",
        },
        "memory": {"memory_used_percent": used},
    }


def cgroup_node_probe(
    statuses: list[dict[str, Any]] | None = None, used: float = 60.0
) -> FakeProbe:
    return FakeProbe(
        {
            "survey": survey_response(),
            "memory-start": {
                "unit": "gfrs-x-memory",
                "memory_max_bytes": 2 << 30,
                "cgroup_oom": True,
            },
            "memory-status": statuses or [oom_status("", "active"), oom_status()],
            "memory-stop": {"stopped": {"ActiveState": "inactive"}},
            "oracle-start": {"unit": "gfrs-x-oracle"},
            "oracle-stop": {"stopped": {}},
            "oracle-read": [
                oracle_records(used),
                {"records": [], "next_offset": 40, "total": 40},
            ],
        }
    )


def quiet_activity(
    used: float = 60.0,
    incidents: list[dict[str, Any]] | None = None,
    workflows: list[dict[str, Any]] | None = None,
    batches: int = 3,
) -> dict[str, Any]:
    return {
        "incidents": incidents or [],
        "workflows": workflows or [],
        "batches": [
            host_batch(f"2026-10-24T00:0{index}:00+00:00", "memory_used_percent", used)
            for index in range(batches)
        ],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
        "metric_latest": [],
    }


def log_incident(action: str, workflow_id: str = "wf-log") -> dict[str, Any]:
    return {
        "incident_id": "inc-log",
        "event_id": "log-oom-abc",
        "event_type": "NODE_HEALTH",
        "state": "ACTION_PENDING",
        "effective_action": action,
        "created_at": "2026-10-24T00:03:00+00:00",
        "workflow_request_id": workflow_id,
        "node_ids": [NODE],
    }


def cgroup_context(
    tmp_path: Path, document: dict[str, Any], probe: FakeProbe | None = None
) -> Any:
    spec = SPECS["cgroup_only"]
    regional = FakeRegional(cpu_python=store_script({"activity": document}))
    return make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"node@{NODE}": probe or cgroup_node_probe()},
    )


def test_cgroup_only_passes_without_a_memory_finding(tmp_path: Path) -> None:
    probe = cgroup_node_probe()
    ctx = cgroup_context(tmp_path, quiet_activity(), probe)
    result = SPECS["cgroup_only"].execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    starts = [call for call in probe.calls if call[0] == "memory-start"]
    assert len(starts) == mem.CGROUP_OOM_REPEATS
    assert all(
        "--cgroup-oom-bytes" in call and "--approved" in call for call in starts
    ), starts
    assert all(
        call[call.index("--cgroup-oom-bytes") + 1] == str(mem.CGROUP_OOM_CAP_BYTES)
        for call in starts
    ), "all((call[call.index('--cgroup-oom-bytes') + 1] == str(mem.CGROUP_OOM_"
    assert "--target-used-percent" not in starts[0]
    stops = [call for call in probe.calls if call[0] == "memory-stop"]
    assert len(stops) == mem.CGROUP_OOM_REPEATS, "every OOM-killed unit is reset"
    assert "memory" not in ctx.cleanup.pending()
    assert len(result["cgroup_oom_kills"]) == mem.CGROUP_OOM_REPEATS
    assert all(item["result"] == "oom-kill" for item in result["cgroup_oom_kills"]), (
        "all((item['result'] == 'oom-kill' for item in result['cgroup_oom_kills"
    )
    assert result["host_used_percent"]["max"] < mem.USED_THRESHOLD
    assert round(result["cgroup_cap_percent_of_host"], 2) == 0.11
    assert (ctx.case_dir / "oracle.jsonl").is_file(), "oracle evidence written"


def test_cgroup_only_needs_the_approval(tmp_path: Path) -> None:
    spec = SPECS["cgroup_only"]
    settings = make_settings(spec, node=NODE)
    object.__setattr__(settings, "approvals", ())
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=store_script({"activity": quiet_activity()})),
        probes={f"node@{NODE}": cgroup_node_probe()},
        settings=settings,
    )
    with pytest.raises(VariantNotRun, match="HOST_MEMORY_PRESSURE"):
        spec.execute(ctx)


def test_cgroup_only_is_inconclusive_when_the_unit_was_not_oom_killed(
    tmp_path: Path,
) -> None:
    probe = cgroup_node_probe(statuses=[oom_status("success", "inactive")])
    ctx = cgroup_context(tmp_path, quiet_activity(), probe)
    with pytest.raises(VariantInconclusive, match="instead of oom-kill"):
        SPECS["cgroup_only"].execute(ctx)
    assert [call[0] for call in probe.calls if call[0] == "memory-stop"] == [
        "memory-stop"
    ], "the unit is reset before the verdict is abandoned"
    assert "memory" not in ctx.cleanup.pending()


def test_cgroup_only_is_inconclusive_when_the_unit_never_ended(tmp_path: Path) -> None:
    probe = cgroup_node_probe(statuses=[oom_status("", "active")])
    ctx = cgroup_context(tmp_path, quiet_activity(), probe)
    with pytest.raises(VariantInconclusive, match="still running"):
        SPECS["cgroup_only"].execute(ctx)


def test_cgroup_only_is_inconclusive_when_the_host_itself_was_under_pressure(
    tmp_path: Path,
) -> None:
    ctx = cgroup_context(tmp_path, quiet_activity(), cgroup_node_probe(used=96.0))
    with pytest.raises(VariantInconclusive, match="under pressure"):
        SPECS["cgroup_only"].execute(ctx)


def test_cgroup_only_is_inconclusive_without_a_fresh_host_sample(
    tmp_path: Path,
) -> None:
    ctx = cgroup_context(tmp_path, quiet_activity(batches=0))
    with pytest.raises(VariantInconclusive, match="silent channel"):
        SPECS["cgroup_only"].execute(ctx)


def test_cgroup_only_fails_when_the_product_minted_a_memory_finding(
    tmp_path: Path,
) -> None:
    ctx = cgroup_context(tmp_path, chain_activity())
    result = SPECS["cgroup_only"].execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any(
        "unexpected memory_used_percent incident" in e for e in result["errors"]
    ), result["errors"]


def test_cgroup_only_fails_when_an_oom_log_line_isolated_the_node(
    tmp_path: Path,
) -> None:
    document = quiet_activity(
        incidents=[log_incident("QUARANTINE")],
        workflows=[
            workflow(
                "wf-log",
                "inc-log",
                ["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"],
            )
        ],
    )
    result = SPECS["cgroup_only"].execute(cgroup_context(tmp_path, document))
    assert result["verdict"] == "FAIL"
    assert any("graded QUARANTINE" in e for e in result["errors"]), result["errors"]
    assert any("isolation operations" in e for e in result["errors"]), result["errors"]


def test_cgroup_only_accepts_a_diagnostic_only_oom_log_finding(tmp_path: Path) -> None:
    document = quiet_activity(
        incidents=[log_incident("RUN_DIAGNOSTICS")],
        workflows=[workflow("wf-log", "inc-log", ["FREEZE_EVIDENCE", "VALIDATE_HOST"])],
    )
    result = SPECS["cgroup_only"].execute(cgroup_context(tmp_path, document))
    assert result["verdict"] == "PASS", result["errors"]
    stopped = quiet_activity(
        incidents=[log_incident("RUN_DIAGNOSTICS")],
        workflows=[
            workflow("wf-log", "inc-log", ["FREEZE_EVIDENCE", "STOP_WORKLOADS"])
        ],
    )
    result = SPECS["cgroup_only"].execute(cgroup_context(tmp_path, stopped))
    assert result["verdict"] == "FAIL"
    assert any("stopped a workload" in e for e in result["errors"]), result["errors"]


def test_cgroup_only_spec_reuses_the_memory_approval() -> None:
    spec = SPECS["cgroup_only"]
    assert spec.implemented and spec.approvals == ("HOST_MEMORY_PRESSURE",)
    assert spec.lanes == ("eks", "hp") and spec.proof_mode == "native"


# --------------------------------------------------------------------------- #
# the real probe's memory-start behind the fake (live 2026-10-25: the variant
# asked for a 1980 s hold and the probe's 1800 s cap refused it before any
# product evidence, three times on two 2 TiB nodes)
# --------------------------------------------------------------------------- #
def real_memory_start(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A ``memory-start`` that runs the real probe function over a p5en-shaped host."""

    import subprocess

    from scripts.e2e.regional.probes import general_live_node_probe as nodep

    gib = 1024**3
    monkeypatch.setattr(nodep, "STATE_ROOT", tmp_path / "node-state")
    monkeypatch.setattr(
        nodep,
        "run",
        lambda command, **_: subprocess.CompletedProcess(command, 0, "", ""),
    )
    monkeypatch.setattr(nodep, "arm_deadman", lambda run_id, kind, seconds, *a: kind)
    monkeypatch.setattr(
        nodep,
        "meminfo",
        lambda: {"MemTotal": 2000 * gib, "MemAvailable": 1928 * gib, "SwapTotal": 0.0},
    )
    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(nodep, "emit", emitted.append)

    def start(arguments: tuple[str, ...]) -> dict[str, Any]:
        parsed = nodep.parser().parse_args(list(arguments))
        parsed.handler(parsed)  # raises ProbeError exactly as on the node
        return dict(emitted[-1])

    return start


@pytest.mark.parametrize("variant", ["memory_pressure", "recovery", "cgroup_only"])
def test_memory_variants_ask_the_real_probe_for_an_accepted_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    probe = mem_node_probe() if variant != "cgroup_only" else cgroup_node_probe()
    probe.script["memory-start"] = real_memory_start(monkeypatch, tmp_path)
    document = chain_activity() if variant != "cgroup_only" else quiet_activity()
    regional = FakeRegional(cpu_python=store_script({"activity": document}))
    ctx = make_context(
        SPECS[variant],
        tmp_path=tmp_path,
        regional=regional,
        probes={f"node@{NODE}": probe},
    )
    result = SPECS[variant].execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    starts = [call for call in probe.calls if call[0] == "memory-start"]
    assert starts, "the variant started the memory unit through the real probe"
    seconds = [int(call[call.index("--seconds") + 1]) for call in starts]
    assert all(value <= mem.PROBE_MAX_SECONDS for value in seconds), seconds
    if variant == "memory_pressure":
        assert seconds == [1980], "the 14-minute ramp plus detection hold"
    if variant == "recovery":
        assert seconds == [1800], "exactly the old cap: recovery never failed live"


def test_memory_pressure_fails_fast_when_the_hold_exceeds_the_probe_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.e2e.regional.regional_commands import RegionalFixtureError

    monkeypatch.setattr(mem, "PROBE_MAX_SECONDS", 1800)  # the live cap of 2026-10-25
    probe = mem_node_probe()
    regional = FakeRegional(cpu_python=store_script({"activity": chain_activity()}))
    ctx = make_context(
        SPECS["memory_pressure"],
        tmp_path=tmp_path,
        regional=regional,
        probes={f"node@{NODE}": probe},
    )
    with pytest.raises(
        RegionalFixtureError, match="1980 s memory hold exceeds .* 1800 s"
    ):
        SPECS["memory_pressure"].execute(ctx)
    assert [call[0] for call in probe.calls] == ["survey"], (
        "refused before the oracle or the memory unit touched the node"
    )


def test_low_available_memory_hold_fits_the_probe_cap() -> None:
    from scripts.e2e.regional import general_live_utilization as util

    requested = util.MEMORY_RAMP_BUDGET + (util.MEMORY_DURATION + 300 + 120) + 120
    assert requested <= mem.PROBE_MAX_SECONDS, requested
