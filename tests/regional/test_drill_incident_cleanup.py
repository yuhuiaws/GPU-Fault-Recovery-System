"""Drill incident closure: closable states close through the product path with
node evidence, planning states and foreign incidents are reported, never
forced; and the runners that inject faults call it from their cleanup."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import drill_incident_cleanup as closure
from scripts.e2e.regional import run_ha003_aurora_failover_reset as ha003
from scripts.e2e.regional import run_ha004_waiting_reclaim_reset as ha004

NODE = "hyperpod-i-fault"
OTHER = "hyperpod-i-other"


class FakeWarm:
    def __init__(
        self,
        state: str,
        *,
        node_ids: list[str] | None = None,
        refusal: str | None = None,
    ) -> None:
        self.state = state
        self.node_ids = [NODE] if node_ids is None else node_ids
        self.refusal = refusal
        self.closes: list[dict[str, Any]] = []

    def incident_by_id(self, incident_id: str) -> dict[str, Any]:
        return {
            "incident_id": incident_id,
            "state": self.state,
            "node_ids": self.node_ids,
        }

    def close_incident_with_evidence(self, incident_id: str, **kwargs: Any) -> dict:
        self.closes.append({"incident_id": incident_id, **kwargs})
        if self.refusal:
            return {"closed": False, "refusal": self.refusal, "state": self.state}
        return {"closed": True, "refusal": None, "state": "RECOVERED"}


class FakeRegional:
    def __init__(self, *, isolated: bool) -> None:
        self.isolated = isolated
        self.reads: list[str] = []

    def node_snapshot(self, node: str) -> dict[str, Any]:
        self.reads.append(node)
        if not self.isolated:
            return {"unschedulable": False, "taints": [], "ownership_annotations": {}}
        return {
            "unschedulable": True,
            "taints": [
                {
                    "key": closure.QUARANTINE_TAINT,
                    "value": "abc",
                    "effect": "NoSchedule",
                }
            ],
            "ownership_annotations": {"gpu-fault.io/incident-id": "inc-a"},
        }


def close(warm: FakeWarm, regional: FakeRegional) -> dict[str, Any]:
    return closure.close_drill_incident(
        warm,
        regional,
        "inc-a",
        reason="HA-003 drill cleanup",
        reference="ha003-run-1",
        nodes=(NODE,),
    )


def test_escalated_incident_closes_through_the_product_path_with_evidence() -> None:
    warm = FakeWarm("ESCALATED")
    regional = FakeRegional(isolated=False)

    report = close(warm, regional)

    assert report["closed"] is True, report
    assert report["state_after"] == "RECOVERED"
    assert "residual" not in report
    assert warm.closes == [
        {
            "incident_id": "inc-a",
            "reason": "HA-003 drill cleanup",
            "operator": closure.DEFAULT_OPERATOR,
            "reference": "ha003-run-1",
            "evidence": [
                {
                    "node_id": NODE,
                    "exists": True,
                    "unschedulable": False,
                    "quarantine_taint_value": None,
                    "isolation_annotations": {},
                }
            ],
        }
    ], "the close carries the drill's reason, reference and fresh node evidence"


def test_quarantined_incident_closes_only_once_its_nodes_are_released() -> None:
    held = FakeWarm("QUARANTINED")
    report = close(held, FakeRegional(isolated=True))
    assert report["closed"] is False
    assert "still isolated" in report["residual"], report
    assert held.closes == [], "an isolated node is released by a restore, not a close"

    released = FakeWarm("QUARANTINED")
    report = close(released, FakeRegional(isolated=False))
    assert report["closed"] is True, report
    assert len(released.closes) == 1


@pytest.mark.parametrize("state", ["DETECTED", "ACTION_PENDING", "SAFETY_PENDING", ""])
def test_planning_states_are_reported_not_forced(state: str) -> None:
    warm = FakeWarm(state)
    regional = FakeRegional(isolated=False)

    report = close(warm, regional)

    assert report["closed"] is False
    assert "workflow that ends it" in report["residual"], report
    assert warm.closes == [] and regional.reads == [], "no close, no node read"


def test_recovered_incident_is_left_alone() -> None:
    warm = FakeWarm("RECOVERED")

    report = close(warm, FakeRegional(isolated=False))

    assert report == {
        "incident_id": "inc-a",
        "state_before": "RECOVERED",
        "closed": False,
        "already_terminal": True,
    }
    assert warm.closes == []


def test_an_incident_over_other_nodes_is_never_closed() -> None:
    warm = FakeWarm("ESCALATED", node_ids=[NODE, OTHER])

    report = close(warm, FakeRegional(isolated=False))

    assert report["closed"] is False
    assert "foreign incident" in report["residual"], report
    assert warm.closes == []


def test_a_service_refusal_becomes_the_residual() -> None:
    warm = FakeWarm(
        "ESCALATED", refusal="incident inc-a still has an open workflow wf-1"
    )

    report = close(warm, FakeRegional(isolated=False))

    assert report["closed"] is False
    assert report["refusal"] == report["residual"]
    assert "open workflow" in report["residual"]


def test_batch_close_dedupes_ids_and_collects_residuals() -> None:
    warm = FakeWarm("ESCALATED", refusal="refused")

    reports = closure.close_drill_incidents(
        warm,
        FakeRegional(isolated=False),
        ["inc-a", None, "", "inc-a"],
        reason="r",
        reference="ref",
        nodes=(NODE,),
    )

    assert list(reports) == ["inc-a"], "each id is closed once"
    assert closure.residual_incidents(reports) == {"inc-a": "refused"}


# --- runner wiring -----------------------------------------------------------------


class Host:
    def execute(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {}

    def cleanup(self) -> dict[str, bool]:
        return {"pods": False}


class Regional:
    def node_snapshot(self, node: str) -> dict[str, Any]:
        assert node == NODE, node
        return {"ownership_annotations": {}, "unschedulable": False, "taints": []}


def spy(calls: list[dict[str, Any]], residual: str | None):
    def close_drill_incident(warm: Any, regional: Any, incident_id: str, **kwargs: Any):
        calls.append({"incident_id": incident_id, **kwargs})
        report = {
            "incident_id": incident_id,
            "state_before": "ACTION_PENDING",
            "closed": False,
        }
        if residual:
            report["residual"] = residual
        return report

    return close_drill_incident


@pytest.mark.parametrize("module", [ha003, ha004])
def test_ha_cleanup_closes_its_own_incident_and_names_a_residual(
    module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        module, "close_drill_incident", spy(calls, "incident is ACTION_PENDING")
    )
    monkeypatch.setattr(module, "WarmSpareLiveFixture", lambda *_a: SimpleNamespace())
    extra: dict[str, Any] = {}
    if module is ha004:
        baseline = {"replicas": 1, "env": {}}
        extra["timing"] = SimpleNamespace(
            restore=lambda: dict(baseline), baseline=baseline
        )

    result = module.cleanup_case(
        settings=SimpleNamespace(node=NODE),
        regional=Regional(),
        host=Host(),
        preflight={"store": {"profile": {"profile_version": "v1"}}},
        incident_id="inc-a",
        run_id="run-1",
        sampler_started=False,
        **extra,
    )

    assert result["errors"] == [], result
    assert calls == [
        {
            "incident_id": "inc-a",
            "reason": f"{module.CASE_ID.rsplit('-', 2)[-2]}-{module.CASE_ID.rsplit('-', 1)[-1]} drill cleanup",
            "reference": "run-1",
            "nodes": (NODE,),
        }
    ], "the drill's own incident is closed with the run id as reference"
    assert result["residual_incident"] == {
        "incident_id": "inc-a",
        "state": "ACTION_PENDING",
        "reason": "incident is ACTION_PENDING",
    }, "a non-closable leftover is a visible finding, not a silent one"


@pytest.mark.parametrize("module", [ha003, ha004])
def test_ha_cleanup_without_an_incident_closes_nothing(
    module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(module, "close_drill_incident", spy(calls, None))
    extra: dict[str, Any] = {}
    if module is ha004:
        baseline = {"replicas": 1, "env": {}}
        extra["timing"] = SimpleNamespace(
            restore=lambda: dict(baseline), baseline=baseline
        )

    result = module.cleanup_case(
        settings=SimpleNamespace(node=NODE),
        regional=Regional(),
        host=Host(),
        preflight={"store": {}},
        incident_id="",
        run_id="run-1",
        sampler_started=False,
        **extra,
    )

    assert calls == [] and "residual_incident" not in result, result
