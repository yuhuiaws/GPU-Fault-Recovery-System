from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_destr022_spare_reservation_reclaim as case
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureAbort,
    RegionalFixtureError,
)
from tests.regional._cov95_destr_reclaim import SPARE, ReclaimHarness, verdicts


def test_reclaim_uses_owned_conditional_patch_and_keeps_spare_cordoned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ReclaimHarness(tmp_path, monkeypatch)
    preflight = h.plan(tmp_path)
    assert preflight["errors"] == [], preflight
    h.calls.clear()
    code, report = h.execute(tmp_path)
    assert code == 0 and report["errors"] == report["cleanup"]["errors"] == [], report
    assert report["reclaimed_at"] is not None, report
    assert report["synthetic_incident_id"] == h.incident_id, report
    assert h.node["unschedulable"] is True, h.node
    assert h.node["annotations"] == h.baseline["annotations"], h.node
    assert sum(name == "node.patch" for name, _ in h.calls) == 1, h.calls
    assert report["cleanup"]["other_nodes"] == {"count": 2}, report


@pytest.mark.parametrize("defect", ["timeout", "uncordon", "drop-patch", "ack-loss"])
def test_reclaim_failure_restores_only_its_recorded_metadata(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ReclaimHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    if defect == "timeout":
        h.reclaim_after_reads = 10000
    elif defect == "uncordon":
        h.unexpected_uncordon = True
    elif defect == "drop-patch":
        h.drop_patch = True
    else:
        h.reclaim_after_reads = 10000
        h.patch_ack_lost = True
    code, report = h.execute(tmp_path)
    assert code == 1 and report["verdict"] == "FAIL", report
    if defect in {"timeout", "ack-loss"}:
        assert h.node["annotations"] == h.baseline["annotations"], h.node
        assert report["cleanup"]["errors"] == [], report
    elif defect == "uncordon":
        assert "became schedulable" in report["error"], report
    else:
        assert "patch was not observed" in report["error"], report


@pytest.mark.parametrize(
    "phase", ["executor.logs", "provider.events", "runtime.verify"]
)
def test_reclaim_phase_failure_still_checks_owned_metadata_and_other_nodes(
    phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ReclaimHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures[phase] = RuntimeError(f"fake failure at {phase}")
    code, report = h.execute(tmp_path)
    assert code == 1 and phase in str(report), report
    assert report["cleanup"]["other_nodes"] == {"count": 2}, report


@pytest.mark.parametrize("defect", ["counter", "stale", "provider", "log-failure"])
def test_reclaim_needs_causal_executor_evidence_and_no_provider_action(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ReclaimHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    if defect == "counter":
        h.counter_delta = 0
    elif defect == "stale":
        h.breadcrumb_stale = True
    elif defect == "provider":
        h.events = [{"event_name": "BatchReplaceClusterNodes"}]
    else:
        h.extra_logs = ["WARNING spare reservation sweep failed"]
    code, report = h.execute(tmp_path)
    assert code == 1 and report["errors"], report
    assert h.node["unschedulable"] is True, h.node


def test_reclaim_abort_after_patch_preserves_cleanup_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ReclaimHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures["executor.logs"] = RegionalFixtureAbort(2)
    with pytest.raises(RegionalFixtureAbort):
        h.execute(tmp_path)
    assert h.node["annotations"] == h.baseline["annotations"], h.node
    assert any(name == "runtime.verify" for name, _ in h.calls), h.calls


def test_reclaim_uid_drift_between_plan_and_execute_refuses_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ReclaimHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.node["uid"] = "new-node"
    with pytest.raises(RegionalFixtureError, match="plan drifted"):
        h.execute(tmp_path)
    assert not h.injected, h.calls


def test_reclaim_expired_window_never_patches_the_spare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ReclaimHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    code, report = h.execute(tmp_path, seconds=-1)
    assert code == 1 and "window ended" in report["error"], report
    assert not h.injected, h.calls


def test_reclaim_rechecks_spare_readiness_after_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ReclaimHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    read = h.warm.node_snapshot
    count = 0

    def snapshot(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal count
        count += 1
        if count >= 3:
            h.node["ready"] = "False"
        return read(*args, **kwargs)

    monkeypatch.setattr(h.warm, "node_snapshot", snapshot)
    code, report = h.execute(tmp_path)
    assert code == 1 and "spare changed since preflight" in report["error"], report
    assert not h.injected, h.calls


def test_executor_probe_rejects_non_object_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ReclaimHarness(tmp_path, monkeypatch)
    h.executor_value = []
    with pytest.raises(RegionalFixtureError, match="JSON object"):
        case.executor_probes(h.regional)


def test_reclaim_cleanup_does_not_overwrite_foreign_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ReclaimHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    original = h.regional.provider_events

    def events(*args: Any) -> list[dict[str, Any]]:
        result = original(*args)
        h.node["annotations"][verdicts.SPARE_RESERVATION_ANNOTATION] = (
            "foreign-incident"
        )
        return result

    monkeypatch.setattr(h.regional, "provider_events", events)
    code, report = h.execute(tmp_path)
    assert code == 1 and "tracked node field changed" in str(
        report["cleanup"]["errors"]
    ), report
    assert (
        h.node["annotations"][verdicts.SPARE_RESERVATION_ANNOTATION]
        == "foreign-incident"
    ), h.node


def test_reclaim_detects_changes_to_other_nodes_in_finally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ReclaimHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    before = deepcopy(h.nodes)
    original = h.regional.provider_events

    def events(*args: Any) -> list[dict[str, Any]]:
        result = original(*args)
        h.nodes[1]["unschedulable"] = True
        return result

    monkeypatch.setattr(h.regional, "provider_events", events)
    code, report = h.execute(tmp_path)
    assert code == 1 and report["cleanup"]["errors"], report
    assert h.nodes[0]["name"] == SPARE and h.nodes[1] != before[1], h.nodes
