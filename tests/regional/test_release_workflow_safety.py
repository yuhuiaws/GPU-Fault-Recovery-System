from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault_release import regional_release_probes as PROBES
from gpu_fault_release import regional_release_workflow_safety as SAFETY
from tests.regional._release_orchestrator_support import ingress_pod_list_json

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_RESOLUTION = ROOT / "src/gpu_fault/workflow_resolution.py"


class Runner:
    dry_run = False

    def __init__(self, result: str, *, installed: bool = True) -> None:
        self.result = result
        self.installed = installed
        self.runs: list[list[str]] = []

    def run(self, args, **_kwargs):
        self.runs.append(list(args))
        if "get" in args and "pod" in args:
            return ingress_pod_list_json("cpu-pod") if "json" in args else "cpu-pod"
        return self.result

    def probe_output(self, args, **_kwargs):
        assert "deployment" in args
        if self.installed == "scaled-to-zero":
            return 0, "0", ""
        if self.installed:
            return 0, "3", ""
        return (
            1,
            "",
            'Error from server (NotFound): deployments.apps "gpu-fault-api-ha" not found',
        )


def release(result: str, *, installed: bool = True):
    return SimpleNamespace(
        runner=Runner(result, installed=installed),
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


def test_workflow_safety_sets_aside_a_workflow_blocked_at_compile_time() -> None:
    """A compiler refusal that never ran is the dispatcher's to close, not a blocker.

    The record observed live -- ``REMEDIATE_EFA_DRIVER`` BLOCKED with ``no
    executable owner for efaDriverRemediation`` -- held the release that would
    have added the owner. The sweep that closes it ships in that release, so the
    gate has to roll past it while still reporting what it set aside.
    """

    result = SAFETY.workflow_safety_snapshot(
        release(
            '{"blocker_count":0,"blockers":[],'
            '"resolved_blocked_count":0,"resolved_blocked":[],'
            '"abandoned_generation_count":0,"abandoned_generation":[],'
            '"compile_blocked_count":1,"compile_blocked":["workflow-efa"]}'
        )
    )

    assert result["compile_blocked_count"] == 1
    assert result["compile_blocked"] == ["workflow-efa"]


# The guard fields that make a compile-time BLOCKED record provably a no-op.
# Both copies of the predicate have to read every one of them: dropping any
# single field turns the test into "BLOCKED with a reason", which would set
# aside -- and let the dispatcher close -- a record an operator is meant to act
# on (the dispatcher's own INTERNAL_ERROR BLOCKED, which a claim has stamped).
COMPILE_BLOCKED_GUARDS = (
    "blocked_reasons",
    "execution_epoch",
    "step_executions",
    "completed_step_indexes",
    "completed_operations",
    "execution_owner_id",
    "source_plan_id",
    "remediation_budget_claims",
    "IncidentState.RECOVERED",
    "IncidentState.ESCALATED",
    "RemoteCommandStatus.PENDING",
    "RemoteCommandStatus.LEASED",
    "RemoteCommandStatus.WAITING",
)
COMPILE_BLOCKED = ROOT / "src/gpu_fault/compile_blocked.py"


def test_the_probe_and_the_product_agree_on_what_makes_a_record_compile_blocked() -> (
    None
):
    """The probe inlines the predicate; this holds the two copies together.

    Same arrangement as ``abandoned_generation`` above and for the same reason:
    the probe runs against the *already deployed* ``gpu_fault``, which for the
    release that first carries the sweep has no ``gpu_fault.compile_blocked``.
    """

    preflight = PROBES.probe_source("workflow_safety")
    product = COMPILE_BLOCKED.read_text(encoding="utf-8")

    for source, label in ((preflight, "probe"), (product, "product")):
        assert "compile_blocked" in source, label
        for guard in COMPILE_BLOCKED_GUARDS:
            assert guard in source, f"{label} copy stopped reading {guard}"
        assert "WorkflowStatus.BLOCKED" in source, label

    assert "gpu_fault.compile_blocked" not in preflight.replace(
        "``gpu_fault.compile_blocked", ""
    ), (
        "importing the product module would make the probe fail on exactly the "
        "deployment that needs it -- the one that has not shipped the sweep yet"
    )
    assert "compile_blocked_count" in preflight


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


def test_workflow_safety_sets_aside_a_blocked_workflow_of_a_recovered_incident() -> (
    None
):
    """An Always-Fatal SXID's RESTART_BM workflow stayed BLOCKED after an operator
    closed its incident RECOVERED, and held the release carrying the fix for the
    replayed log line that had opened it. Nothing waits on such a record; the
    gate rolls past it and reports it, as it does for the compile-time shape."""

    result = SAFETY.workflow_safety_snapshot(
        release(
            '{"blocker_count":0,"blockers":[],'
            '"resolved_blocked_count":0,"resolved_blocked":[],'
            '"abandoned_generation_count":0,"abandoned_generation":[],'
            '"compile_blocked_count":0,"compile_blocked":[],'
            '"settled_incident_blocked_count":1,'
            '"settled_incident_blocked":["workflow-restart-bm"]}'
        )
    )

    assert result["settled_incident_blocked_count"] == 1
    assert result["settled_incident_blocked"] == ["workflow-restart-bm"]


# The guards that make a BLOCKED record of a RECOVERED incident provably dead:
# nothing live on it, and an incident nobody waits on. ESCALATED is deliberately
# absent -- that incident still awaits an operator who may act on the record.
SETTLED_INCIDENT_GUARDS = (
    "IncidentState.RECOVERED",
    "execution_owner_id",
    "remediation_budget_claims",
    "source_plan_id",
    "RemoteCommandStatus.PENDING",
    "RemoteCommandStatus.LEASED",
    "RemoteCommandStatus.WAITING",
)


def test_the_probe_and_the_product_agree_on_what_makes_a_settled_incident_record() -> (
    None
):
    preflight = PROBES.probe_source("workflow_safety")
    product = COMPILE_BLOCKED.read_text(encoding="utf-8")

    for source, label in ((preflight, "probe"), (product, "product")):
        assert "settled_incident_blocked" in source, label
        for guard in SETTLED_INCIDENT_GUARDS:
            assert guard in source, f"{label} copy stopped reading {guard}"
    assert "settled_incident_blocked_count" in preflight
    assert "is not IncidentState.RECOVERED" in preflight, (
        "the probe must set aside RECOVERED only, never ESCALATED"
    )


def test_workflow_safety_passes_on_a_first_bootstrap_without_a_control_plane() -> None:
    # The first regional preflight runs before gpu-fault-api-ha exists, so the
    # in-Pod probe cannot run; with no control plane nothing destructive can be
    # active, and the check must not fail the bootstrap.
    target = release("unused", installed=False)

    snapshot = SAFETY.workflow_safety_snapshot(target)

    assert snapshot["blocker_count"] == 0
    assert snapshot["control_plane"] == "not installed"
    assert target.runner.runs == []


def test_workflow_safety_still_fails_when_the_deployment_is_unreadable() -> None:
    target = release("unused")
    target.runner.probe_output = lambda *_args, **_kwargs: (
        1,
        "",
        "Unable to connect to the server: dial tcp: i/o timeout",
    )

    with pytest.raises(SAFETY.ReleaseError, match="CPU ingress Deployment"):
        SAFETY.workflow_safety_snapshot(target)
    assert target.runner.runs == []


def test_workflow_safety_passes_over_a_control_plane_a_cleanup_scaled_to_zero() -> None:
    """Live 2026-09-13: a failed bootstrap's cleanup leaves the Deployments in
    place at zero replicas; the retry's preflight must not ask a Pod that cannot
    exist. Scaled to zero is "nothing running", like not installed."""

    target = release("unused", installed="scaled-to-zero")

    snapshot = SAFETY.workflow_safety_snapshot(target)

    assert snapshot["blocker_count"] == 0
    assert snapshot["control_plane"] == "not installed"
    assert target.runner.runs == []
