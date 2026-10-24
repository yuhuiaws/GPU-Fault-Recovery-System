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
