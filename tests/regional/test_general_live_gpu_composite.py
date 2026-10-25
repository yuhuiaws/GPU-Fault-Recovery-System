"""GF-GENERAL-LIVE-024 composite injector/verdict contract (scripted)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_gpu_composite as comp
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
)
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
    from scripts.e2e.regional import general_live_gpu_environment as envc
    from scripts.e2e.regional import general_live_registry as reg

    environment = {spec.variant for spec in envc.VARIANTS}
    assert environment == {"thermal_warning", "power_performance"}
    assert set(SPECS) | environment == set(reg.coverage_rows()["GF-GENERAL-LIVE-024"])
    assert [variant for variant, spec in SPECS.items() if spec.deferred] == []
    assert SPECS["independent_xid"].approvals == ("KERNEL_SOURCE_REPLAY",)
    assert SPECS["independent_xid"].risk == "M"


def _xid_decision(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "event_id": "kernel-log-kmsg-boot-900000064-xid-92-cluster-abc",
        "xid": comp.INDEPENDENT_XID,
        "gpu_uuid": G1,
        "event_source": "KERNEL_LOG",
        "decision": True,
        "disposition": "MONITOR_ONLY",
        "official_action": "IGNORE",
        "action": "NO_ACTION",
        "safety_action": None,
        "requires_operator": False,
        "incident_id": "inc-xid-92",
        "incident_state": "RECOVERED",
        "workflow_request_id": None,
        "duplicate": False,
        "correlated_event_id": None,
        "reasons": [],
    }
    base.update(overrides)
    return base


def _kernel_script() -> dict[str, Any]:
    return {
        "install-records": {"records": 1},
        "open-window": {"after": {"ActiveState": "active"}},
        "arm": {"armed": True, "lines_written": 6},
        "status": {"injected": {"mode": "kmsg", "fixture_lines": 6}},
        "close-window": {"clean": True},
    }


def _independent_ctx(tmp_path: Path, decisions: list[dict[str, Any]]) -> Any:
    from scripts.e2e.regional.general_live_kernel_xid import XID_DECISION_PROBE

    inc = gpu_incident(
        "inc-c", f"b-{G1}-composite-CORRECTABLE_MEMORY_DEGRADATION", state="RECOVERED"
    )
    wf = gpu_workflow(
        "wf", "inc-c", ["FREEZE_EVIDENCE", "VALIDATE_GPU"], status="FAILED"
    )
    gpu_state = {
        "active_findings": [],
        "findings": [
            {
                "finding_id": f"b-{G1}-composite-CORRECTABLE_MEMORY_DEGRADATION",
                "correlation_rule_id": "CORRECTABLE_MEMORY_DEGRADATION",
                "component_metrics": list(comp.CORRECTABLE_COMPONENTS),
                "component_finding_ids": [
                    f"b-{G1}-{name}" for name in comp.CORRECTABLE_COMPONENTS
                ],
                "gpu_uuid": G1,
                "observed_at": "t",
            }
        ],
        "batches": [{"observed_at": "t"}],
    }
    records = gpu_cpu_python(activity=_activity([inc], [wf]), gpu_state=gpu_state)

    def cpu(script: str, *arguments: str) -> dict[str, Any]:
        if script == XID_DECISION_PROBE:
            return {"decisions": decisions}
        return records(script, *arguments)

    probe = dcgm_probe(
        audit=audit_records(
            {
                f"{G1}/ecc_sbe_volatile_total": 1.0,
                f"{G1}/row_remap_correctable_total": 1.0,
            }
        )
    )
    ctx = _ctx(tmp_path, "independent_xid", probe=probe, cpu=cpu)
    ctx.probes[f"kernel-replay-kmsg@{ctx.settings.node}"] = FakeProbe(_kernel_script())
    return ctx


def test_independent_xid_keeps_its_own_decision_beside_the_weak_composite(
    tmp_path: Path,
) -> None:
    ctx = _independent_ctx(tmp_path, [_xid_decision()])
    out = comp.independent_xid(ctx)
    assert out["verdict"] == "PASS", out["errors"]
    assert out["xid"] == 92 and out["pci_bdf"] == "0000:02:00"
    assert out["kernel_replay"]["mode"] == "kmsg"
    kernel = ctx.probes[f"kernel-replay-kmsg@{ctx.settings.node}"]
    verbs = [call[0] for call in kernel.calls]
    assert verbs == ["install-records", "open-window", "arm", "status", "close-window"]
    assert set(out["stages"]) == {
        "composite_chain",
        "composite_read_only",
        "independent",
        "not_covered",
        "hardware",
        "isolation",
    }


def test_independent_xid_fails_when_absorbed_into_the_composite_incident(
    tmp_path: Path,
) -> None:
    ctx = _independent_ctx(tmp_path, [_xid_decision(incident_id="inc-c")])
    out = comp.independent_xid(ctx)
    assert out["verdict"] == "FAIL"
    assert any("absorbed" in error for error in out["errors"]), out["errors"]


def test_independent_xid_fails_when_the_xid_is_dropped(tmp_path: Path) -> None:
    out = comp.independent_xid(_independent_ctx(tmp_path, []))
    assert out["verdict"] == "FAIL"
    assert any("reached no decision" in error for error in out["errors"]), out["errors"]


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
