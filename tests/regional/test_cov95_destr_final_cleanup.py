from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from scripts.e2e.regional import run_destr001_gpu_reset as reset
from scripts.e2e.regional import run_destr003_warm_spare_failover as failover
from scripts.e2e.regional import run_destr009_workload_restart as workload
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._cov95_destr_actions import ActionHarness
from tests.regional._cov95_destr_agent import AgentHarness
from tests.regional._cov95_destr_fence import FenceHarness
from tests.regional._cov95_destr_idle import IdleHarness
from tests.regional._cov95_destr_lifetime import LifetimeHarness
from tests.regional._cov95_destr_warm import WarmHarness


@pytest.mark.parametrize("defect", ["unready", "tainted"])
def test_lifetime_restore_must_leave_a_ready_untainted_node(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = LifetimeHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    original = h.warm.wait_workflow_id

    def restored(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(*args, **kwargs)
        if defect == "unready":
            h.node["ready"] = "False"
        else:
            h.node["taints"] = [{"key": "foreign", "effect": "NoSchedule"}]
        return result

    monkeypatch.setattr(h.warm, "wait_workflow_id", restored)
    code, report = h.execute(tmp_path)
    assert code == 1 and report["cleanup"]["errors"], report
    expected = (
        "not Ready after cleanup" if defect == "unready" else "still carries taints"
    )
    assert any(expected in error for error in report["cleanup"]["errors"]), report
    assert any(name == "window.close" for name, _ in h.calls), h.calls


@pytest.mark.parametrize("defect", ["workload", "node"])
def test_agent_restart_final_audit_refuses_late_workload_or_node_failure(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AgentHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    original = h.call

    def boundary(name: str, detail: Any = None) -> None:
        original(name, detail)
        if name == "agent.disarm-restore":
            if defect == "workload":
                h.workloads = [{"name": "late-workload"}]
            else:
                h.node["ready"] = "False"

    monkeypatch.setattr(h, "call", boundary)
    code, report = h.execute(tmp_path)
    assert code == 1 and report["cleanup"]["errors"], report
    expected = (
        "acquired a non-system workload"
        if defect == "workload"
        else "Node state differs from baseline"
    )
    assert any(expected in error for error in report["cleanup"]["errors"]), report
    assert h.injected and h.restarted, h.calls


@pytest.mark.parametrize("defect", ["prewarm-error", "prewarm-residual", "identity"])
def test_workload_cleanup_failure_cannot_leave_a_successful_verdict(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ActionHarness(workload, tmp_path, monkeypatch)
    h.plan(tmp_path)
    if defect == "prewarm-error":
        h.failures["prewarm.cleanup"] = RuntimeError("fake prewarm cleanup failed")
    elif defect == "prewarm-residual":
        h.transforms["prewarm.cleanup"] = lambda *_a: {"pod": True}
    else:

        def verified(result: Any, _args: Any, kwargs: Any) -> Any:
            if kwargs["stage"] == "after DESTR-009 cleanup":
                raise RegionalFixtureError("fake final runtime drift")
            return result

        h.transforms["regional.verify_runtime_identity"] = verified
    code, report = h.execute(tmp_path)
    assert code == 1 and report["workload_residual"] is False, report
    if defect == "identity":
        assert any("fake final runtime drift" in error for error in report["errors"]), (
            report
        )
    else:
        assert any(report["prewarm_residuals"].values()), report
    assert any(name == "workload.delete" for name, _, _ in h.calls), h.calls


@pytest.mark.parametrize(
    ("family", "phase"),
    [
        ("reset", "host.write-xid46"),
        ("failover", "replacement.post"),
        ("fence", "host.write-xid"),
        ("lifetime", "injector.write-xid46"),
        ("agent", "agent.restart-agent"),
    ],
)
def test_lost_transport_supervision_propagates_without_a_pass_artifact(
    family: str, phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if family == "reset":
        h: Any = IdleHarness(reset, tmp_path, monkeypatch)
    elif family == "failover":
        h = WarmHarness(failover, tmp_path, monkeypatch)
    elif family == "fence":
        h = FenceHarness(tmp_path, monkeypatch)
    elif family == "lifetime":
        h = LifetimeHarness(tmp_path, monkeypatch)
    else:
        h = AgentHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    original = h.call
    lost = False
    refused: list[str] = []

    def guarded(name: str, detail: Any = None) -> None:
        nonlocal lost
        if lost:
            refused.append(name)
            raise ProcessSupervisionLost("fake process tree remains unproved")
        original(name, detail)
        if name == phase:
            lost = True
            raise ProcessSupervisionLost("fake process tree remains unproved")

    monkeypatch.setattr(h, "call", guarded)
    with pytest.raises(ProcessSupervisionLost, match="unproved"):
        h.execute(tmp_path)
    assert lost and refused, {"phase": phase, "refused": refused, "calls": h.calls}
    assert h.calls[-1][0] == phase, h.calls
    verdict_files = list((tmp_path / "cases").glob("GF-REGIONAL-DESTR-*/*.json"))
    assert all(
        json.loads(path.read_text()).get("verdict") != "PASS"
        for path in verdict_files
        if path.stem.startswith("GF-REGIONAL-DESTR-")
    ), verdict_files
