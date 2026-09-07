from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault_release import regional_release_probes as PROBES
from gpu_fault_release import regional_release_workflow_safety as SAFETY

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_RESOLUTION = ROOT / "src/gpu_fault/workflow_resolution.py"


class Runner:
    dry_run = False

    def __init__(self, result: str) -> None:
        self.result = result

    def run(self, args, **_kwargs):
        return "cpu-pod" if "get" in args and "pod" in args else self.result


def release(result: str):
    return SimpleNamespace(
        runner=Runner(result),
        config=SimpleNamespace(namespace="gpu-fault-system"),
        _cpu=lambda *args: ["kubectl", *args],
    )


def test_workflow_safety_accepts_blocked_workflows_closed_by_restore() -> None:
    result = SAFETY.workflow_safety_snapshot(
        release(
            '{"blocker_count":0,"blockers":[],"resolved_blocked_count":4,'
            '"resolved_blocked":["workflow-a"]}'
        )
    )

    assert result["resolved_blocked_count"] == 4


def test_workflow_safety_rejects_unresolved_destructive_workflows() -> None:
    with pytest.raises(SAFETY.ReleaseError, match="block release"):
        SAFETY.workflow_safety_snapshot(
            release(
                '{"blocker_count":1,"blockers":["workflow-a"],'
                '"resolved_blocked_count":0,"resolved_blocked":[]}'
            )
        )


def test_workflow_safety_accepts_generations_the_incident_replanned_away_from() -> None:
    """A record whose incident moved on is paperwork, not a live remediation.

    This is the case that wedged the 2026-09-04 release: a PENDING workflow at
    generation 1, no step ever handed to an adapter, whose incident had advanced
    to generation 4 and named a different workflow. Counting it as a blocker
    stopped every release -- including the one carrying the fix that
    terminalizes it -- so the gate has to pass on a zero blocker count while
    still reporting what it set aside.
    """

    result = SAFETY.workflow_safety_snapshot(
        release(
            '{"blocker_count":0,"blockers":[],'
            '"resolved_blocked_count":0,"resolved_blocked":[],'
            '"abandoned_generation_count":1,'
            '"abandoned_generation":["workflow-bdcf2823"]}'
        )
    )

    assert result["abandoned_generation_count"] == 1
    assert result["abandoned_generation"] == ["workflow-bdcf2823"]


def test_workflow_safety_still_rejects_when_a_real_blocker_accompanies_one() -> None:
    """Setting one record aside must not clear the ones beside it."""

    with pytest.raises(SAFETY.ReleaseError, match="block release"):
        SAFETY.workflow_safety_snapshot(
            release(
                '{"blocker_count":1,"blockers":["workflow-live"],'
                '"resolved_blocked_count":0,"resolved_blocked":[],'
                '"abandoned_generation_count":1,'
                '"abandoned_generation":["workflow-bdcf2823"]}'
            )
        )


# The guard fields that make an abandoned generation provably dead. Both copies
# of the predicate have to read every one of them: dropping any single field
# turns the test into "PENDING and behind", which would terminalize a workflow
# that is mid-step, holding a lease, or holding a fleet-wide budget claim.
ABANDONED_GENERATION_GUARDS = (
    "step_executions",
    "completed_step_indexes",
    "completed_operations",
    "remediation_budget_claims",
    "execution_owner_id",
    "execution_lease_expires_at",
    "fencing_token",
    "workflow_request_id",
)


def test_the_probe_and_the_product_agree_on_what_makes_a_generation_abandoned() -> None:
    """The probe inlines the predicate, so something has to hold the copies together.

    It cannot import ``workflow_resolution.abandoned_generation_successor``: the
    engine ships this file to the Pod as source and it runs against the
    *already deployed* ``gpu_fault``, which for the release that first carries
    the fix is a module without it. That is a deliberate duplication, and this
    is the test that stops it drifting into two different definitions of dead.
    """

    preflight = PROBES.probe_source("workflow_safety")
    resolution = WORKFLOW_RESOLUTION.read_text(encoding="utf-8")

    for source, label in ((preflight, "probe"), (resolution, "product")):
        assert "abandoned_generation" in source, label
        for guard in ABANDONED_GENERATION_GUARDS:
            assert guard in source, f"{label} copy stopped reading {guard}"
        assert "WorkflowStatus.PENDING" in source, label

    assert "abandoned_generation_successor" not in preflight, (
        "importing the product helper would make the probe fail on exactly the "
        "deployment that needs it -- the one that has not shipped the fix yet"
    )


def test_preflight_and_fleet_gate_require_verified_restore_evidence() -> None:
    # Both gates ship their program to the Pod as source, so the evidence they
    # demand is a property of the probe body, not of the engine module that
    # sends it -- that is where these assertions have to look.
    preflight = PROBES.probe_source("workflow_safety")
    fleet = PROBES.probe_source("rollout_wave_safety")
    resolution = WORKFLOW_RESOLUTION.read_text(encoding="utf-8")

    assert "IncidentState.RECOVERED" in preflight
    for source in (fleet, resolution):
        assert "verified_restore_successor" in source
    assert "IncidentState.RECOVERED" in resolution
    assert "WorkflowStatus.SUCCEEDED" in resolution
    assert "WorkflowOperation.RESTORE_SCHEDULING" in resolution
