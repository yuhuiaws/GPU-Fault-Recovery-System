"""GF-GENERAL-LIVE-021 fabric-state injector/verdict contract (scripted)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_gpu_fabric as fab
from scripts.e2e.regional.general_live_fixture import VariantNotRun
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings
from tests.regional._general_live_gpu_fakes import (
    audit_records,
    dcgm_probe,
    gpu_cpu_python,
    gpu_incident,
    gpu_workflow,
    survey,
)

SPECS = {spec.variant: spec for spec in fab.VARIANTS}
GPUS = [
    "GPU-00000000-0000-4000-8000-000000000000",
    "GPU-11111111-1111-4111-8111-111111111111",
]


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


def _served() -> list[dict[str, Any]]:
    return audit_records({f"{gpu}/fabric_manager_status": 2.0 for gpu in GPUS})


def _activity(incidents: list, workflows: list = ()) -> dict[str, Any]:
    return {
        "incidents": incidents,
        "workflows": list(workflows),
        "batches": [],
        "markers": [],
        "commands": [],
        "notifications": [],
        "metric_latest": [],
        "collector_statuses": [],
        "agent": None,
    }


def test_registry_covers_021_and_reset_variants_need_approval() -> None:
    from scripts.e2e.regional import general_live_registry as reg

    assert set(SPECS) == set(reg.coverage_rows()["GF-GENERAL-LIVE-021"])
    for variant in (
        "not_started",
        "in_progress",
        "failed",
        "full_reset",
        "every_gpu_success",
        "recovery_failure",
    ):
        spec = SPECS[variant]
        assert (
            "GPU_FULL_FABRIC_RESET" in spec.approvals
            and spec.flags == "X"
            and spec.lanes == ("hp",)
        )
    assert "NODE_REBOOT" in SPECS["recovery_failure"].approvals


def test_before_window_passes_with_no_finding(tmp_path: Path) -> None:
    cpu = gpu_cpu_python(activity=_activity([]), gpu_state={"active_findings": []})
    out = fab.before_window(
        _ctx(tmp_path, "before_window", probe=dcgm_probe(audit=_served()), cpu=cpu)
    )
    assert out["verdict"] == "PASS", out["errors"]


def test_before_window_fails_if_a_fabric_finding_appears(tmp_path: Path) -> None:
    inc = gpu_incident("inc", "b-node-fabric_manager_status", action="RESET_GPU")
    cpu = gpu_cpu_python(
        activity=_activity([inc], [gpu_workflow("wf", "inc", ["FREEZE_EVIDENCE"])]),
        gpu_state={"active_findings": []},
    )
    out = fab.before_window(
        _ctx(tmp_path, "before_window", probe=dcgm_probe(audit=_served()), cpu=cpu)
    )
    assert out["verdict"] == "FAIL" and any(
        "under the window" in e for e in out["errors"]
    )


def test_ungraded_product_requires_the_label(tmp_path: Path) -> None:
    cpu = gpu_cpu_python(activity=_activity([]), gpu_state={"active_findings": []})
    out = fab.ungraded_product(
        _ctx(tmp_path, "ungraded_product", probe=dcgm_probe(audit=_served()), cpu=cpu)
    )
    assert out["verdict"] == "PASS", out["errors"]


def test_full_reset_requires_the_full_fabric_chain(tmp_path: Path) -> None:
    inc = gpu_incident(
        "inc", "b-node-fabric_manager_status", action="RESET_GPU", workflow="wf"
    )
    executions = [{"operation": "RESET_ALL_GPUS_NVSWITCHES", "status": "SUCCEEDED"}]
    wf = gpu_workflow(
        "wf", "inc", list(fab.FULL_RESET), status="SUCCEEDED", executions=executions
    )
    cpu = gpu_cpu_python(
        activity=_activity([inc], [wf]), gpu_state={"active_findings": []}
    )
    out = fab.full_reset(
        _ctx(
            tmp_path,
            "full_reset",
            probe=dcgm_probe(audit=_served()),
            cpu=cpu,
            approvals=("GPU_FULL_FABRIC_RESET",),
        )
    )
    assert out["verdict"] == "PASS", out["errors"]


def test_full_reset_fails_without_the_reset_step(tmp_path: Path) -> None:
    inc = gpu_incident(
        "inc", "b-node-fabric_manager_status", action="RESET_GPU", workflow="wf"
    )
    wf = gpu_workflow(
        "wf", "inc", ["FREEZE_EVIDENCE", "QUARANTINE"], status="SUCCEEDED"
    )
    cpu = gpu_cpu_python(
        activity=_activity([inc], [wf]), gpu_state={"active_findings": []}
    )
    out = fab.full_reset(
        _ctx(
            tmp_path,
            "full_reset",
            probe=dcgm_probe(audit=_served()),
            cpu=cpu,
            approvals=("GPU_FULL_FABRIC_RESET",),
        )
    )
    assert out["verdict"] == "FAIL", out


def test_graded_product_is_required(tmp_path: Path) -> None:
    probe = dcgm_probe(survey(model="NVIDIA L40S"))
    cpu = gpu_cpu_python()
    with pytest.raises(VariantNotRun, match="NVSwitch"):
        fab.before_window(_ctx(tmp_path, "before_window", probe=probe, cpu=cpu))
