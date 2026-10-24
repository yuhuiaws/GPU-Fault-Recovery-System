"""GF-GENERAL-LIVE-001 injector/verdict contract over scripted product records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_host as host
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    host_batch,
    incident,
    make_context,
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
    assert (ctx.case_dir / "oracle.jsonl").is_file()


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
    with pytest.raises(VariantInconclusive):
        spec.execute(ctx)


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
    assert any("RESET_GPU" in error for error in result["errors"])


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
    assert any("succeeded while the fault" in error for error in result["errors"])


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


def test_recovery_and_checkpoint_phase_pass(tmp_path: Path) -> None:
    for variant in ("recovery", "checkpoint_phase"):
        spec = SPECS[variant]
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
        assert result["verdict"] == "PASS", (variant, result["errors"])
    assert any("phase correlation" in note for note in result["notes"])


def test_invalid_delta_is_a_deferred_live_replay_variant() -> None:
    spec = SPECS["invalid_delta"]
    assert spec.deferred.startswith("FIXTURE_NOT_READY")
    assert spec.execute is None
