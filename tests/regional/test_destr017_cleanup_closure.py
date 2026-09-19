"""DESTR-017 cleanup closes the fenced incident on node isolation evidence.

Split out of ``test_destr017_out_of_band_reboot_fence.py`` (at the architecture
size ceiling). The fenced workflow's quiesce is never restored by the product --
the generation fence refused the compensation and the out-of-band reboot
retired it on the host -- so the record that stays QUARANTINED over a node
nobody isolates is closed the way ``workflow-reconcile --close-quarantined``
closes it, on the runner's own read of the node; a validated restore is kept
for whichever incident still owns isolation.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import destr017_verdicts as verdicts
from scripts.e2e.regional import run_destr017_out_of_band_reboot_fence as destr017

# --------------------------------------------------------------------------- #
# Cleanup: the fenced incident is closed on isolation evidence, not restored
# --------------------------------------------------------------------------- #
# After the out-of-band reboot the fenced workflow's quiesce was never restored
# by the product (the generation fence refused the compensation) and the reboot
# retired it on the host. The support successor owns the node's isolation and a
# validated restore lifts it; the fenced record then stays QUARANTINED over a
# node nobody isolates, and the product's exit for that is the evidence-based
# close ``workflow-reconcile --close-quarantined`` performs -- not another
# validated restore, which the fixture's own quiesce guard refuses.


class _CleanupWarm:
    def __init__(self, states: dict[str, str], refusal: str | None = None) -> None:
        self.states = dict(states)
        self.refusal = refusal
        self.restores: list[str] = []
        self.closes: list[dict[str, Any]] = []

    def wait_incident_idle(self, incident_id: str) -> None:
        return None

    def incident_by_id(self, incident_id: str) -> dict[str, Any]:
        return {"incident_id": incident_id, "state": self.states[incident_id]}

    def create_restore_workflow(self, **kwargs: Any) -> dict[str, Any]:
        self.restores.append(str(kwargs["incident_id"]))
        return {"workflow_request_id": f"restore-{kwargs['incident_id']}"}

    def wait_workflow_id(self, workflow_id: str) -> dict[str, Any]:
        incident_id = workflow_id.removeprefix("restore-")
        # The successor's restore lifts the isolation it owned.
        self.states[incident_id] = "RECOVERED"
        return {"status": "SUCCEEDED"}

    def close_incident_with_evidence(
        self, incident_id: str, **kwargs: Any
    ) -> dict[str, Any]:
        self.closes.append({"incident_id": incident_id, **kwargs})
        if self.refusal:
            return {"closed": False, "refusal": self.refusal, "state": None}
        self.states[incident_id] = "RECOVERED"
        return {"closed": True, "refusal": None, "state": "RECOVERED"}


class _CleanupRegional:
    def __init__(self, snapshots: list[dict[str, Any]]) -> None:
        self.snapshots = list(snapshots)

    def node_snapshot(self, node: str) -> dict[str, Any]:
        if len(self.snapshots) > 1:
            return self.snapshots.pop(0)
        return self.snapshots[0]


def _cleanup_run(warm: _CleanupWarm, regional: _CleanupRegional) -> Any:
    return SimpleNamespace(
        warm=warm,
        regional=regional,
        settings=SimpleNamespace(node="node-a"),
        preflight={"store": {"profile": {"profile_version": "hyperpod-v1"}}},
        successor_incident_id="inc-support-after-workflow-1",
        incident_id="inc-fenced-1",
    )


def _isolated_snapshot() -> dict[str, Any]:
    return {
        "unschedulable": True,
        "taints": [{"key": verdicts.QUARANTINE_TAINT, "value": "abc"}],
        "ownership_annotations": {
            "gpu-fault.io/incident-id": "inc-support-after-workflow-1"
        },
    }


def _clean_snapshot() -> dict[str, Any]:
    return {"unschedulable": False, "taints": [], "ownership_annotations": {}}


def test_cleanup_restores_the_isolating_successor_then_closes_the_fenced_record_on_evidence() -> (
    None
):
    warm = _CleanupWarm(
        {"inc-support-after-workflow-1": "ESCALATED", "inc-fenced-1": "QUARANTINED"}
    )
    # Isolated while the successor owns the node, clean once its restore ran.
    regional = _CleanupRegional(
        [_isolated_snapshot(), _isolated_snapshot(), _clean_snapshot()]
    )
    report = destr017.restore_isolation(_cleanup_run(warm, regional))
    assert warm.restores == ["inc-support-after-workflow-1"], (
        "the validated restore must name the successor that owns the isolation"
    )
    assert [item["incident_id"] for item in warm.closes] == ["inc-fenced-1"], (
        "the fenced record must be closed on evidence, not restored again"
    )
    evidence = warm.closes[0]["evidence"]
    assert evidence == [
        {
            "node_id": "node-a",
            "exists": True,
            "unschedulable": False,
            "quarantine_taint_value": None,
            "isolation_annotations": {},
        }
    ], "the evidence must be the runner's own read of a node carrying no isolation"
    assert report["inc-support-after-workflow-1"] == "SUCCEEDED", (
        "the successor restore result is recorded"
    )
    assert report["inc-fenced-1"] == "CLOSED_ON_EVIDENCE:RECOVERED", (
        "the fenced record's evidence close is recorded with the resulting state"
    )


def test_cleanup_still_restores_a_fenced_incident_that_owns_the_isolation() -> None:
    warm = _CleanupWarm(
        {"inc-support-after-workflow-1": "RECOVERED", "inc-fenced-1": "QUARANTINED"}
    )
    regional = _CleanupRegional([_isolated_snapshot()])
    report = destr017.restore_isolation(_cleanup_run(warm, regional))
    assert warm.restores == ["inc-fenced-1"] and warm.closes == [], (
        "a node still isolated is restored through the validated-restore workflow"
    )
    assert report["inc-support-after-workflow-1"] == "RECOVERED", (
        "an already RECOVERED incident is recorded, not touched"
    )


def test_cleanup_reports_a_refused_evidence_close_as_a_failure() -> None:
    warm = _CleanupWarm(
        {"inc-support-after-workflow-1": "RECOVERED", "inc-fenced-1": "QUARANTINED"},
        refusal="incident inc-fenced-1 still owns isolation on node-a",
    )
    regional = _CleanupRegional([_clean_snapshot()])
    with pytest.raises(
        destr017.RegionalFixtureError,
        match="evidence close of inc-fenced-1 was refused",
    ):
        destr017.restore_isolation(_cleanup_run(warm, regional))
    assert warm.restores == [], (
        "a refused close must not fall through to a restore that would also be refused"
    )


def test_isolation_evidence_projects_the_quarantine_taint_and_ownership_annotations() -> (
    None
):
    evidence = destr017.isolation_evidence("node-a", _isolated_snapshot())
    assert evidence == {
        "node_id": "node-a",
        "exists": True,
        "unschedulable": True,
        "quarantine_taint_value": "abc",
        "isolation_annotations": {
            "gpu-fault.io/incident-id": "inc-support-after-workflow-1"
        },
    }, "the evidence must carry exactly what the closure service judges"
