"""GF-GENERAL-LIVE-020/022 injector/verdict contract over scripted records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.e2e.regional import general_live_gpu_links as links
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings
from tests.regional._general_live_gpu_fakes import (
    audit_records,
    dcgm_probe,
    gpu_cpu_python,
    gpu_incident,
    gpu_workflow,
)

SPECS = {spec.variant: spec for spec in links.VARIANTS}
G0 = "GPU-00000000-0000-4000-8000-000000000000"
G1 = "GPU-11111111-1111-4111-8111-111111111111"


def _ctx(
    tmp_path: Path,
    variant: str,
    *,
    probe: Any,
    cpu: Any,
    approvals: tuple[str, ...] = (),
):
    spec = SPECS[variant]
    regional = FakeRegional(cpu_python=cpu)
    settings = make_settings(spec, node=regional.node_snapshot("")["name"])
    object.__setattr__(settings, "approvals", approvals)
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )
    ctx.probes[f"dcgm@{ctx.settings.node}"] = probe
    return ctx


def _activity(incidents: list, workflows: list = (), **extra: Any) -> dict[str, Any]:
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
        **extra,
    }


def test_registry_covers_020_and_022() -> None:
    from scripts.e2e.regional import general_live_registry as reg

    rows = reg.coverage_rows()
    assert {s.variant for s in links.VARIANTS if s.case_id.endswith("020")} == set(
        rows["GF-GENERAL-LIVE-020"]
    )
    assert {s.variant for s in links.VARIANTS if s.case_id.endswith("022")} == set(
        rows["GF-GENERAL-LIVE-022"]
    )


def test_crc_is_a_clean_nvlink_diagnostic(tmp_path: Path) -> None:
    canonical = "nvlink_crc_flit_error_total"
    inc = gpu_incident("inc", f"b-{G1}-{canonical}", state="RECOVERED")
    wf = gpu_workflow(
        "wf-1",
        "inc",
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
    cpu = gpu_cpu_python(
        activity=_activity([inc], [wf]),
        gpu_state={"active_findings": [], "batches": [{"observed_at": "t"}]},
    )
    out = links.crc(
        _ctx(
            tmp_path,
            "crc",
            probe=dcgm_probe(audit=audit_records({f"{G1}/{canonical}": 1.0})),
            cpu=cpu,
        )
    )
    assert out["verdict"] == "PASS", out["errors"]


def test_at_rate_limit_passes_when_no_finding(tmp_path: Path) -> None:
    cpu = gpu_cpu_python(
        activity=_activity([]),
        gpu_state={"active_findings": [], "batches": [{"observed_at": "t"}]},
    )
    out = links.at_rate_limit(
        _ctx(
            tmp_path,
            "at_rate_limit",
            probe=dcgm_probe(audit=audit_records({f"{G1}/pcie_replay_total": 2.0})),
            cpu=cpu,
        )
    )
    assert out["verdict"] == "PASS", out["errors"]


def test_at_rate_limit_fails_if_a_finding_appears(tmp_path: Path) -> None:
    inc = gpu_incident("inc", f"b-{G1}-pcie_replay_total")
    wf = gpu_workflow("wf-1", "inc", ["FREEZE_EVIDENCE", "VALIDATE_GPU"])
    cpu = gpu_cpu_python(
        activity=_activity([inc], [wf]),
        gpu_state={"active_findings": [], "batches": [{"observed_at": "t"}]},
    )
    out = links.at_rate_limit(
        _ctx(
            tmp_path,
            "at_rate_limit",
            probe=dcgm_probe(audit=audit_records({f"{G1}/pcie_replay_total": 2.0})),
            cpu=cpu,
        )
    )
    assert out["verdict"] == "FAIL" and any("limit" in e for e in out["errors"])


def test_same_gpu_xid_requires_the_composite(tmp_path: Path) -> None:
    comp = gpu_incident(
        "inc-c",
        f"b-{G1}-composite-PCIE_XID_LINK_FAILURE",
        action="DRAIN",
        workflow="wf-c",
    )
    wf = gpu_workflow(
        "wf-c", "inc-c", ["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"]
    )
    cpu = gpu_cpu_python(
        activity=_activity([comp], [wf]),
        gpu_state={"active_findings": [], "batches": [{"observed_at": "t"}]},
    )
    out = links.same_gpu_xid(
        _ctx(
            tmp_path,
            "same_gpu_xid",
            probe=dcgm_probe(audit=audit_records({f"{G1}/pcie_replay_total": 9.0})),
            cpu=cpu,
            approvals=("DCGM_XID_GAUGE",),
        )
    )
    assert out["verdict"] == "PASS", out["errors"]


def test_wrong_gpu_xid_fails_if_the_composite_forms(tmp_path: Path) -> None:
    comp = gpu_incident(
        "inc-c",
        f"b-{G1}-composite-PCIE_XID_LINK_FAILURE",
        action="DRAIN",
        workflow="wf-c",
    )
    wf = gpu_workflow("wf-c", "inc-c", ["MARK_UNSCHEDULABLE", "QUARANTINE"])
    cpu = gpu_cpu_python(
        activity=_activity([comp], [wf]),
        gpu_state={"active_findings": [], "batches": [{"observed_at": "t"}]},
    )
    out = links.wrong_gpu_xid(
        _ctx(
            tmp_path,
            "wrong_gpu_xid",
            probe=dcgm_probe(audit=audit_records({f"{G1}/pcie_replay_total": 9.0})),
            cpu=cpu,
            approvals=("DCGM_XID_GAUGE",),
        )
    )
    assert out["verdict"] == "FAIL" and any("different GPU" in e for e in out["errors"])


def test_expired_xid_is_expected_fail_while_the_composite_joins(tmp_path: Path) -> None:
    spec = SPECS["expired_xid"]
    assert spec.expected_outcome == "FAIL" and "gauge fallback" in spec.product_gap
    comp = gpu_incident(
        "inc-c",
        f"b-{G1}-composite-PCIE_XID_LINK_FAILURE",
        action="DRAIN",
        workflow="wf-c",
    )
    wf = gpu_workflow("wf-c", "inc-c", ["MARK_UNSCHEDULABLE", "QUARANTINE"])
    cpu = gpu_cpu_python(
        activity=_activity([comp], [wf]),
        gpu_state={"active_findings": [], "batches": [{"observed_at": "t"}]},
    )
    out = links.expired_xid(
        _ctx(
            tmp_path,
            "expired_xid",
            probe=dcgm_probe(audit=audit_records({f"{G1}/pcie_replay_total": 9.0})),
            cpu=cpu,
            approvals=("DCGM_XID_GAUGE",),
        )
    )
    assert out["verdict"] == "FAIL", out
    assert "expired" in out["errors"][0]


def test_counter_reset_passes_with_no_finding(tmp_path: Path) -> None:
    cpu = gpu_cpu_python(
        activity=_activity([]),
        gpu_state={"active_findings": [], "batches": [{"observed_at": "t"}]},
    )
    out = links.counter_reset(
        _ctx(
            tmp_path,
            "counter_reset",
            probe=dcgm_probe(
                audit=audit_records({f"{G1}/nvlink_crc_flit_error_total": 5.0})
            ),
            cpu=cpu,
        )
    )
    assert out["verdict"] == "PASS", out["errors"]


def test_independent_fatal_event_is_deferred_to_lane_k() -> None:
    spec = SPECS["independent_fatal_event"]
    assert spec.deferred.startswith("FIXTURE_NOT_READY") and "lane K" in spec.deferred
