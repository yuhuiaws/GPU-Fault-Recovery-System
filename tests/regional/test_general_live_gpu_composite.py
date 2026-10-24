"""GF-GENERAL-LIVE-024 composite injector/verdict contract (scripted)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_gpu_composite as comp
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings
from tests.regional._general_live_gpu_fakes import (
    audit_records,
    dcgm_probe,
    gpu_cpu_python,
    gpu_incident,
    gpu_workflow,
)

SPECS = {spec.variant: spec for spec in comp.VARIANTS}
G0 = "GPU-00000000-0000-4000-8000-000000000000"
G1 = "GPU-11111111-1111-4111-8111-111111111111"


def _ctx(tmp_path: Path, variant: str, *, probe: Any, cpu: Any):
    spec = SPECS[variant]
    regional = FakeRegional(cpu_python=cpu)
    settings = make_settings(spec, node=regional.node_snapshot("")["name"])
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )
    ctx.probes[f"dcgm@{ctx.settings.node}"] = probe
    return ctx


def _activity(incidents: list, workflows: list = ()) -> dict[str, Any]:
    return {
        "incidents": incidents,
        "workflows": list(workflows),
        "batches": [{"observed_at": "t"}],
        "markers": [],
        "commands": [],
        "notifications": [],
        "metric_latest": [],
        "collector_statuses": [],
        "agent": None,
    }


def test_registry_covers_024() -> None:
    from scripts.e2e.regional import general_live_registry as reg

    assert set(SPECS) == set(reg.coverage_rows()["GF-GENERAL-LIVE-024"])
    for variant in ("thermal_warning", "power_performance", "independent_xid"):
        assert SPECS[variant].deferred.startswith("FIXTURE_NOT_READY"), SPECS[
            variant
        ].deferred


def test_corrected_memory_forms_a_diagnostic_composite(tmp_path: Path) -> None:
    inc = gpu_incident(
        "inc", f"b-{G1}-composite-CORRECTABLE_MEMORY_DEGRADATION", state="RECOVERED"
    )
    wf = gpu_workflow("wf", "inc", ["FREEZE_EVIDENCE", "VALIDATE_GPU"], status="FAILED")
    cpu = gpu_cpu_python(
        activity=_activity([inc], [wf]),
        gpu_state={"active_findings": [], "batches": [{"observed_at": "t"}]},
    )
    probe = dcgm_probe(
        audit=audit_records(
            {
                f"{G1}/ecc_sbe_volatile_total": 1.0,
                f"{G1}/row_remap_correctable_total": 1.0,
            }
        )
    )
    out = comp.corrected_memory(
        _ctx(tmp_path, "corrected_memory", probe=probe, cpu=cpu)
    )
    assert out["verdict"] == "PASS", out["errors"]


def test_wrong_scope_source_fails_if_a_composite_forms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(comp, "_validated_restore", lambda ctx, incidents: [])
    bad = gpu_incident(
        "inc-c",
        f"b-{G0}-composite-GPU_MEMORY_DEGRADATION",
        action="QUARANTINE",
        workflow="wf",
    )
    wf = gpu_workflow("wf", "inc-c", ["MARK_UNSCHEDULABLE", "QUARANTINE"])
    cpu = gpu_cpu_python(
        activity=_activity([bad], [wf]), gpu_state={"active_findings": []}
    )
    out = comp.wrong_scope_source(
        _ctx(
            tmp_path,
            "wrong_scope_source",
            probe=dcgm_probe(
                audit=audit_records({f"{G0}/ecc_dbe_volatile_total": 1.0})
            ),
            cpu=cpu,
        )
    )
    assert out["verdict"] == "FAIL" and any(
        "another GPU" in e for e in out["errors"]
    ), out["errors"]


def test_wrong_scope_source_passes_when_no_composite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(comp, "_validated_restore", lambda ctx, incidents: [])
    inc = gpu_incident(
        "inc", f"b-{G0}-ecc_dbe_volatile_total", action="QUARANTINE", workflow="wf"
    )
    wf = gpu_workflow("wf", "inc", ["MARK_UNSCHEDULABLE", "QUARANTINE"])
    cpu = gpu_cpu_python(
        activity=_activity([inc], [wf]), gpu_state={"active_findings": []}
    )
    out = comp.wrong_scope_source(
        _ctx(
            tmp_path,
            "wrong_scope_source",
            probe=dcgm_probe(
                audit=audit_records({f"{G0}/ecc_dbe_volatile_total": 1.0})
            ),
            cpu=cpu,
        )
    )
    assert out["verdict"] == "PASS", out["errors"]


def test_memory_failure_is_critical_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(comp, "_validated_restore", lambda ctx, incidents: [])
    inc = gpu_incident(
        "inc",
        f"b-{G1}-composite-GPU_MEMORY_DEGRADATION",
        state="QUARANTINED",
        action="QUARANTINE",
        workflow="wf",
    )
    wf = gpu_workflow(
        "wf",
        "inc",
        ["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"],
        status="BLOCKED",
    )
    cpu = gpu_cpu_python(
        activity=_activity([inc], [wf]), gpu_state={"active_findings": []}
    )
    out = comp.memory_failure(
        _ctx(
            tmp_path,
            "memory_failure",
            probe=dcgm_probe(
                audit=audit_records({f"{G1}/ecc_dbe_volatile_total": 1.0})
            ),
            cpu=cpu,
        )
    )
    assert out["verdict"] == "PASS", out["errors"]
