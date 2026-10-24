"""GF-GENERAL-LIVE-019 injector/verdict contract over scripted records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_gpu_memory as mem
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings
from tests.regional._general_live_gpu_fakes import (
    audit_records,
    dcgm_probe,
    gpu_cpu_python,
    gpu_incident,
    gpu_workflow,
    survey,
)

SPECS = {spec.variant: spec for spec in mem.VARIANTS}
GPU = "GPU-11111111-1111-4111-8111-111111111111"


def _ctx(tmp_path: Path, variant: str, *, probe: Any, cpu: Any, settings: Any = None):
    spec = SPECS[variant]
    regional = FakeRegional(cpu_python=cpu)
    settings = settings or make_settings(spec, node=regional.node_snapshot("")["name"])
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )
    ctx.probes[f"dcgm@{ctx.settings.node}"] = probe
    return ctx


def test_registry_covers_every_019_variant() -> None:
    from scripts.e2e.regional import general_live_registry as reg

    rows = reg.coverage_rows()["GF-GENERAL-LIVE-019"]
    assert set(rows) == set(SPECS), set(rows) ^ set(SPECS)
    assert all(spec.proof_mode == "live-replay" for spec in mem.VARIANTS), (
        "every 019 variant is a live-replay"
    )


def test_corrected_ecc_is_a_clean_diagnostic_chain(tmp_path: Path) -> None:
    incident = gpu_incident(
        "inc-ecc", f"b-{GPU}-ecc_sbe_volatile_total", state="RECOVERED"
    )
    workflow = gpu_workflow(
        "wf-1",
        "inc-ecc",
        ["FREEZE_EVIDENCE", "VALIDATE_GPU"],
        status="FAILED",
        executions=[
            {
                "operation": "VALIDATE_GPU",
                "status": "FAILED",
                "details": {"host_diagnostic_failure": True},
            }
        ],
    )
    activity = {**_activity([incident], [workflow]), "batches": [{"observed_at": "t"}]}
    probe = dcgm_probe(audit=audit_records({f"{GPU}/ecc_sbe_volatile_total": 1.0}))
    ctx = _ctx(
        tmp_path,
        "corrected_ecc",
        probe=probe,
        cpu=gpu_cpu_python(
            activity=activity,
            gpu_state={"active_findings": [], "batches": [{"observed_at": "t"}]},
        ),
    )
    out = mem.corrected_ecc(ctx)
    assert out["verdict"] == "PASS", out["errors"]
    assert out["proof_scope"] == "live-replay"
    # the replay was deployed, armed and closed (restore before stop).
    calls = [c[0] for c in probe.calls]
    assert (
        calls.index("arm")
        < calls.index("restore-collector")
        < calls.index("stop-proxy")
    )


def test_diagnostic_fails_when_a_reset_is_planned(tmp_path: Path) -> None:
    incident = gpu_incident(
        "inc-ecc", f"b-{GPU}-ecc_sbe_volatile_total", action="RESET_GPU"
    )
    workflow = gpu_workflow(
        "wf-1", "inc-ecc", ["FREEZE_EVIDENCE", "RESET_GPU", "VALIDATE_GPU"]
    )
    activity = _activity([incident], [workflow])
    ctx = _ctx(
        tmp_path,
        "aggregate_dbe",
        probe=dcgm_probe(audit=audit_records({f"{GPU}/ecc_dbe_aggregate_total": 1.0})),
        cpu=gpu_cpu_python(activity=activity, gpu_state={"active_findings": []}),
    )
    out = mem.aggregate_dbe(ctx)
    assert out["verdict"] == "FAIL"
    assert any("RESET_GPU" in e for e in out["errors"]), out["errors"]


def test_inconclusive_when_the_replay_never_served(tmp_path: Path) -> None:
    from scripts.e2e.regional.general_live_fixture import VariantInconclusive

    ctx = _ctx(
        tmp_path,
        "retired_delta",
        probe=dcgm_probe(audit=audit_records()),
        cpu=gpu_cpu_python(),
    )
    with pytest.raises(VariantInconclusive, match="never served"):
        mem.retired_delta(ctx)


def test_inconclusive_when_the_replay_aborted(tmp_path: Path) -> None:
    from scripts.e2e.regional.general_live_fixture import VariantInconclusive

    records = audit_records(
        {f"{GPU}/ecc_sbe_volatile_total": 1.0}, aborted="real upstream change"
    )
    ctx = _ctx(
        tmp_path, "corrected_ecc", probe=dcgm_probe(audit=records), cpu=gpu_cpu_python()
    )
    with pytest.raises(VariantInconclusive, match="abort"):
        mem.corrected_ecc(ctx)


def test_volatile_dbe_quarantines_without_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incident = gpu_incident(
        "inc-dbe",
        f"b-{GPU}-ecc_dbe_volatile_total",
        state="QUARANTINED",
        action="QUARANTINE",
        workflow="wf-q",
    )
    workflow = gpu_workflow(
        "wf-q",
        "inc-dbe",
        ["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"],
        status="BLOCKED",
    )
    activity = _activity([incident], [workflow])
    restored = {}
    monkeypatch.setattr(
        mem,
        "_validated_restore",
        lambda ctx, incidents: (restored.update(called=incidents), [])[1],
    )
    ctx = _ctx(
        tmp_path,
        "volatile_dbe",
        probe=dcgm_probe(audit=audit_records({f"{GPU}/ecc_dbe_volatile_total": 1.0})),
        cpu=gpu_cpu_python(activity=activity),
    )
    out = mem.volatile_dbe(ctx)
    assert out["verdict"] == "PASS", out["errors"]
    assert restored["called"][0]["incident_id"] == "inc-dbe"


def test_volatile_dbe_fails_if_a_reset_appears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incident = gpu_incident(
        "inc-dbe",
        f"b-{GPU}-ecc_dbe_volatile_total",
        state="QUARANTINED",
        action="QUARANTINE",
        workflow="wf-q",
    )
    workflow = gpu_workflow(
        "wf-q", "inc-dbe", ["MARK_UNSCHEDULABLE", "QUARANTINE", "RESET_GPU"]
    )
    activity = _activity([incident], [workflow])
    monkeypatch.setattr(mem, "_validated_restore", lambda ctx, incidents: [])
    ctx = _ctx(
        tmp_path,
        "volatile_dbe",
        probe=dcgm_probe(audit=audit_records({f"{GPU}/ecc_dbe_volatile_total": 1.0})),
        cpu=gpu_cpu_python(activity=activity),
    )
    out = mem.volatile_dbe(ctx)
    assert out["verdict"] == "FAIL" and any("RESET_GPU" in e for e in out["errors"])


def test_pending_supported_requires_a_reset_chain(tmp_path: Path) -> None:
    spec = SPECS["pending_supported"]
    assert spec.approvals == ("GPU_RESET",) and spec.flags == "X" and spec.risk == "H"
    incident = gpu_incident(
        "inc-pend", f"b-{GPU}-row_remap_pending", action="RESET_GPU", workflow="wf-r"
    )
    workflow = gpu_workflow(
        "wf-r",
        "inc-pend",
        [
            "QUIESCE_GPU_SERVICES",
            "VERIFY_NO_GPU_CLIENTS",
            "RESET_GPU",
            "RESTORE_GPU_SERVICES",
            "VALIDATE_GPU",
        ],
        status="RUNNING",
    )
    activity = _activity([incident], [workflow])
    settings = make_settings(spec, node="ip-10-0-43-5.us-west-2.compute.internal")
    object.__setattr__(settings, "approvals", ("GPU_RESET",))
    ctx = _ctx(
        tmp_path,
        "pending_supported",
        probe=dcgm_probe(audit=audit_records({f"{GPU}/row_remap_pending": 1.0})),
        cpu=gpu_cpu_python(activity=activity),
        settings=settings,
    )
    out = mem.pending_supported(ctx)
    assert out["verdict"] == "PASS", out["errors"]


def test_pending_rejects_a_non_modern_gpu(tmp_path: Path) -> None:
    from scripts.e2e.regional.general_live_fixture import VariantNotRun

    probe = dcgm_probe(survey(model="NVIDIA L40S"))
    ctx = _ctx(tmp_path, "pending_supported", probe=probe, cpu=gpu_cpu_python())
    with pytest.raises(VariantNotRun, match="modern"):
        mem.pending_supported(ctx)


def test_deferred_variants_record_their_reason() -> None:
    for variant in ("pending_unsupported", "observe_profile"):
        spec = SPECS[variant]
        assert spec.deferred.startswith("CAPABILITY_ABSENT"), spec.deferred
        assert spec.execute is None


def _activity(incidents: list, workflows: list) -> dict[str, Any]:
    return {
        "incidents": incidents,
        "workflows": workflows,
        "batches": [],
        "markers": [],
        "commands": [],
        "notifications": [],
        "metric_latest": [],
        "collector_statuses": [],
        "agent": None,
    }
