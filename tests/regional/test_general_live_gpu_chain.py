"""``chain_errors`` judges executed steps, not planned ones (live 2026-10-26)."""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional.general_live_gpu_verdicts import (
    GpuChain,
    chain_errors,
    chain_executed,
    never_executed_errors,
)
from tests.regional._general_live_gpu_fakes import gpu_incident, gpu_workflow

GPU = "GPU-11111111-1111-4111-8111-111111111111"
DRAIN_STEPS = [
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "QUARANTINE",
    "COLLECT_DIAGNOSTIC_BUNDLE",
    "RUN_FIELD_DIAGNOSTIC",
    "VALIDATE_GPU",
    "ESCALATE_SUPPORT",
]
CHAIN = GpuChain(
    label="row_remap_failure",
    operations=("FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"),
    forbidden=frozenset({"RESET_GPU"}),
    incident_states=frozenset({"ESCALATED", "ACTION_PENDING", "QUARANTINED"}),
)
DIAGNOSTIC = GpuChain(
    label="ecc_sbe_volatile_total",
    operations=("FREEZE_EVIDENCE", "VALIDATE_GPU"),
    validation="VALIDATE_GPU",
    validation_must_fail=True,
)
#: The record live 2026-10-26 06:25Z left: BLOCKED at compile time, superseded by
#: the dispatcher sweep 35 ms later, no step ever executed.
HOLLOW = {
    "status": "SUPERSEDED",
    "blocked_kind": "NEEDS_OPERATOR",
    "blocked_reasons": ["no executable owner for memoryDiagnostics"],
    "preemption_reason": (
        "dispatcher reconciliation: closed compile-time BLOCKED workflow that never "
        'changed a node; blocked_reasons=["no executable owner for memoryDiagnostics"]'
    ),
}


def _executed(*operations: str, failed: str | None = None) -> list[dict[str, Any]]:
    return [
        {
            "operation": operation,
            "status": "FAILED" if operation == failed else "SUCCEEDED",
            "updated_at": f"2026-10-26T06:25:{index:02d}+00:00",
            "error": "node agent refused" if operation == failed else None,
        }
        for index, operation in enumerate(operations)
    ]


def _activity(incident: dict[str, Any], workflow: dict[str, Any]) -> dict[str, Any]:
    return {"incidents": [incident], "workflows": [workflow]}


def _case(workflow_extra: dict[str, Any], **incident: Any) -> tuple[Any, list, Any]:
    inc = gpu_incident(
        "inc-remap",
        f"b-{GPU}-row_remap_failure",
        state=incident.get("state", "ESCALATED"),
        action="DRAIN",
        workflow="wf-drain",
    )
    wf = {**gpu_workflow("wf-drain", "inc-remap", DRAIN_STEPS), **workflow_extra}
    return _activity(inc, wf), [inc], wf


def test_the_compile_time_blocked_record_is_a_hollow_chain() -> None:
    """The planned steps were all there; none ran. The old plan-only reading
    passed this record and the node nothing had touched."""

    activity, incidents, wf = _case(HOLLOW)
    errors = chain_errors(activity, incidents, CHAIN)
    assert errors == [
        "row_remap_failure chain planned but never executed: compile-time BLOCKED "
        "(no executable owner for memoryDiagnostics)"
    ], errors
    assert never_executed_errors(wf, CHAIN) == errors, "the one rule names it once"
    assert chain_executed(activity, incidents, CHAIN) is True, (
        "an ended record is judgeable at once: nothing more will execute"
    )


def test_blocked_reasons_without_any_execution_are_the_hollow_chain_too() -> None:
    # Before the sweep ran: still BLOCKED, no preemption reason yet.
    activity, incidents, _wf = _case(
        {
            "status": "BLOCKED",
            "blocked_kind": "NEEDS_OPERATOR",
            "blocked_reasons": ["no executable owner for memoryDiagnostics"],
        }
    )
    (error,) = chain_errors(activity, incidents, CHAIN)
    assert "planned but never executed: compile-time BLOCKED" in error, error
    assert "memoryDiagnostics" in error, error


def test_an_ended_record_needs_a_succeeded_execution_of_every_isolating_step() -> None:
    activity, incidents, _wf = _case(
        {
            "status": "BLOCKED",
            "blocked_kind": "NEEDS_OPERATOR",
            "step_executions": _executed("FREEZE_EVIDENCE"),
        }
    )
    errors = chain_errors(activity, incidents, CHAIN)
    assert errors == [
        "row_remap_failure chain planned but never executed: MARK_UNSCHEDULABLE has "
        "no execution",
        "row_remap_failure chain planned but never executed: QUARANTINE has no "
        "execution",
    ], errors


def test_an_executed_containment_passes_even_when_parked_needs_operator() -> None:
    """The post-fix shape (b192): containment executed, then the record parks
    BLOCKED/NEEDS_OPERATOR with its blocked reason; blocked_reasons beside
    executions are not a hollow chain."""

    activity, incidents, _wf = _case(
        {
            "status": "BLOCKED",
            "blocked_kind": "NEEDS_OPERATOR",
            "blocked_reasons": ["no executable owner for memoryDiagnostics"],
            "step_executions": _executed(
                "FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"
            ),
        }
    )
    assert chain_errors(activity, incidents, CHAIN) == [], "executed steps pass"


def test_a_failed_containment_step_is_its_own_error() -> None:
    activity, incidents, _wf = _case(
        {
            "status": "RUNNING",
            "step_executions": _executed(
                "FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", failed="MARK_UNSCHEDULABLE"
            ),
        }
    )
    errors = chain_errors(activity, incidents, CHAIN)
    assert errors == [
        "row_remap_failure MARK_UNSCHEDULABLE ended FAILED: node agent refused"
    ], errors


def test_a_record_that_may_still_execute_is_judged_on_what_it_did() -> None:
    activity, incidents, _wf = _case(
        {"status": "RUNNING", "step_executions": _executed("FREEZE_EVIDENCE")},
        state="ACTION_PENDING",
    )
    assert chain_errors(activity, incidents, CHAIN) == [], (
        "a step still ahead of a RUNNING record is not a missing execution"
    )
    assert chain_executed(activity, incidents, CHAIN) is False, (
        "the caller that needs the executions in hand keeps waiting"
    )
    executed, executed_incidents, _wf = _case(
        {
            "status": "RUNNING",
            "step_executions": _executed(
                "FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"
            ),
        },
        state="ACTION_PENDING",
    )
    assert chain_executed(executed, executed_incidents, CHAIN) is True, (
        "terminal executions of every isolating step make the chain judgeable"
    )
    assert chain_executed(activity, [], CHAIN) is False, "no incident, nothing to judge"


def test_diagnostic_only_chains_are_unchanged() -> None:
    inc = gpu_incident("inc-ecc", f"b-{GPU}-ecc_sbe_volatile_total", state="RECOVERED")
    wf = gpu_workflow(
        "wf-1",
        "inc-ecc",
        ["FREEZE_EVIDENCE", "VALIDATE_GPU"],
        status="FAILED",
        executions=[{"operation": "VALIDATE_GPU", "status": "FAILED"}],
    )
    assert chain_errors(_activity(inc, wf), [inc], DIAGNOSTIC) == [], (
        "no isolating step: only the validation's execution is judged, as before"
    )
    never_ran = gpu_workflow("wf-1", "inc-ecc", ["FREEZE_EVIDENCE", "VALIDATE_GPU"])
    errors = chain_errors(_activity(inc, never_ran), [inc], DIAGNOSTIC)
    assert errors == ["ecc_sbe_volatile_total workflow never executed VALIDATE_GPU"], (
        errors
    )
    assert chain_executed(_activity(inc, never_ran), [inc], DIAGNOSTIC) is True, (
        "a chain without isolating steps has no containment to wait for"
    )


def test_an_unexecuted_operation_that_ran_is_an_error() -> None:
    """RUN_FIELD_DIAGNOSTIC is planned by the DRAIN but has no executable owner
    on this site (memoryDiagnostics OBSERVE); a record that ran it anyway fails."""

    chain = GpuChain(
        label="row_remap_failure",
        operations=("FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"),
        forbidden=frozenset({"RESET_GPU"}),
        unexecuted=frozenset({"RUN_FIELD_DIAGNOSTIC"}),
    )
    planned_only, incidents, _wf = _case(
        {
            "status": "BLOCKED",
            "blocked_kind": "SAFETY_SETTLED",
            "blocked_reasons": ["no executable owner for memoryDiagnostics"],
            "step_executions": _executed(
                "FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"
            ),
        },
        state="QUARANTINED",
    )
    assert chain_errors(planned_only, incidents, chain) == [], (
        "planned but not executed is exactly what the OBSERVE capability means"
    )
    ran, incidents, _wf = _case(
        {
            "status": "BLOCKED",
            "blocked_kind": "SAFETY_SETTLED",
            "step_executions": _executed(
                "FREEZE_EVIDENCE",
                "MARK_UNSCHEDULABLE",
                "QUARANTINE",
                "RUN_FIELD_DIAGNOSTIC",
            ),
        },
        state="QUARANTINED",
    )
    assert chain_errors(ran, incidents, chain) == [
        "row_remap_failure workflow wf-drain executed RUN_FIELD_DIAGNOSTIC "
        "(SUCCEEDED); it must not run on this site"
    ], chain_errors(ran, incidents, chain)


def test_a_safety_plan_lists_its_containment_under_safety_steps() -> None:
    """b192 runs the containment prefix as a SAFETY plan; whether the record
    lists it under safety_steps or official_steps, the chain reads both."""

    inc = gpu_incident(
        "inc-remap",
        f"b-{GPU}-row_remap_failure",
        state="QUARANTINED",
        action="DRAIN",
        workflow="wf-drain",
    )
    wf = {
        **gpu_workflow(
            "wf-drain",
            "inc-remap",
            ["RUN_FIELD_DIAGNOSTIC", "VALIDATE_GPU", "ESCALATE_SUPPORT"],
            status="BLOCKED",
            executions=_executed("FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"),
        ),
        "blocked_kind": "SAFETY_SETTLED",
        "safety_steps": [
            {"operation": op, "parameters": {}}
            for op in ("FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE")
        ],
    }
    assert chain_errors(_activity(inc, wf), [inc], CHAIN) == [], (
        "the containment planned as safety steps and executed is the chain"
    )
