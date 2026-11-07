"""DESTR-017: a fenced chain that still holds its node owes one support hand-off.

Split out of ``test_destr017_out_of_band_reboot_fence.py`` (at the architecture
size ceiling). From b262 (R23) the product hands the node a FAILED record still
holds to support (``failure_outcome.held_chain_hand_off``, stage
``held_chain_step``): a reset chain whose GPU-client check failed after its
cordon has no rung, so even when the RESTORE_GPU_SERVICES compensation
SUCCEEDED the node keeps the chain's cordon and gets exactly one
``workflow-support-after-<id>`` with the containment hand-off's plan. The case
owes that successor too -- and still refuses one for a chain that released the
node, and every hardware rung either way.
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional import destr017_verdicts as verdicts
from tests.regional.test_destr017_out_of_band_reboot_fence import (
    NODE,
    REQUEST,
    fenced_incident,
    fenced_workflow,
    support_incident,
    support_workflow,
)


def _forbidden(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {"reboot": None, "replace": None, "drain": None}
    value.update(overrides)
    return value


def held_chain_workflow(**overrides: Any) -> dict[str, Any]:
    """The fenced record whose restore compensation SUCCEEDED: the waiting
    GPU-client check failed on the generation fence, the node keeps the
    chain's cordon (MARK_UNSCHEDULABLE completed, RESTORE_SCHEDULING never
    ran)."""

    workflow = fenced_workflow(**overrides)
    for item in workflow["step_executions"]:
        if item["operation"] == verdicts.COMPENSATION_OPERATION:
            item.update(status="SUCCEEDED", error=None)
    return workflow


def _successor_errors(evidence: dict[str, Any], **successor: Any) -> list[str]:
    """``successor_errors`` wired the way the runner wires it from the fence
    evidence (``run_destr017_out_of_band_reboot_fence.execute_case``)."""

    return verdicts.successor_errors(
        successor.get("workflow", {}),
        successor.get("incident", {}),
        node=NODE,
        predecessor_request_id=REQUEST,
        forbidden_escalations=successor.get("forbidden", _forbidden()),
        compensation_failed="FAILED" in (evidence.get("compensation_statuses") or []),
        incident_recovered=bool(successor.get("recovered", False)),
        node_held=evidence.get("node_held") is True,
    )


def test_fence_evidence_says_whether_the_chain_still_holds_the_node() -> None:
    held = verdicts.fence_evidence(
        {"completed_operations": ["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE"]}
    )
    released = verdicts.fence_evidence(
        {"completed_operations": ["MARK_UNSCHEDULABLE", "RESTORE_SCHEDULING"]}
    )
    never_cordoned = verdicts.fence_evidence({"completed_operations": []})
    assert held["node_held"] is True
    assert released["node_held"] is False
    assert never_cordoned["node_held"] is False
    assert verdicts.fence_evidence(fenced_workflow())["node_held"] is True


def test_a_held_fenced_chain_with_a_restored_quarantine_window_hands_the_node_over() -> (
    None
):
    """The compensation SUCCEEDED but the chain's cordon was never released:
    exactly one support successor of the support plan is owed, and its absence
    leaves the cordoned node with nobody."""

    assert (
        verdicts.successor_errors(
            support_workflow(),
            support_incident(),
            node=NODE,
            predecessor_request_id=REQUEST,
            forbidden_escalations=_forbidden(),
            compensation_failed=False,
            node_held=True,
        )
        == []
    )
    errors = verdicts.successor_errors(
        {},
        {},
        node=NODE,
        predecessor_request_id=REQUEST,
        forbidden_escalations=_forbidden(),
        compensation_failed=False,
        node_held=True,
    )
    assert any("still held the node" in item for item in errors), errors
    assert not any("failed restore compensation" in item for item in errors), errors


def test_the_r23_record_passes_with_its_held_chain_step_successor() -> None:
    """The R23 shape end to end: the fenced record (generation fence in the
    record, compensation SUCCEEDED, node held) is a valid fence, and the
    runner's wiring owes and accepts the one support-after successor."""

    workflow = held_chain_workflow()
    evidence = verdicts.fence_evidence(workflow)
    assert evidence["compensation_statuses"] == ["SUCCEEDED"]
    assert evidence["node_held"] is True
    assert verdicts.workflow_errors(workflow, fenced_incident(), node=NODE) == []
    successor = {"workflow": support_workflow(), "incident": support_incident()}
    assert _successor_errors(evidence, **successor) == []
    missing = _successor_errors(evidence)
    assert any("held_chain_step" in item for item in missing), missing


def test_the_held_chain_successor_is_held_to_the_support_plan() -> None:
    """The held-chain hand-off is judged like the containment one: this
    record's support-after, the four-step plan, ESCALATE_OPERATOR, the node."""

    evidence = verdicts.fence_evidence(held_chain_workflow())
    errors = _successor_errors(
        evidence,
        workflow=support_workflow(
            request_id="workflow-support-after-someone-else",
            official_action="RESTART_NODE",
            official_steps=[],
        ),
        incident=support_incident(node_ids=["node-c"]),
    )
    assert any("not this workflow's support escalation" in item for item in errors), (
        errors
    )
    assert any("not the support plan" in item for item in errors), errors
    assert any("not ESCALATE_OPERATOR" in item for item in errors), errors
    assert any(f"does not name {NODE}" in item for item in errors), errors


def test_a_released_chain_still_refuses_a_support_successor() -> None:
    """A chain whose RESTORE_SCHEDULING released the node owes nothing: a
    support escalation without a failed containment step stays a failure."""

    workflow = held_chain_workflow(
        completed_operations=[
            "FREEZE_EVIDENCE",
            "MARK_UNSCHEDULABLE",
            "QUIESCE_GPU_SERVICES",
            "RESTORE_SCHEDULING",
        ]
    )
    evidence = verdicts.fence_evidence(workflow)
    assert evidence["node_held"] is False
    assert _successor_errors(evidence) == []
    errors = _successor_errors(
        evidence, workflow=support_workflow(), incident=support_incident()
    )
    assert any("released the node" in item for item in errors), errors


def test_hardware_rungs_stay_forbidden_and_a_recovery_owes_nothing() -> None:
    evidence = verdicts.fence_evidence(held_chain_workflow())
    for rung in ("reboot", "replace", "drain"):
        errors = _successor_errors(
            evidence,
            workflow=support_workflow(),
            incident=support_incident(),
            forbidden=_forbidden(**{rung: {"incident": {}}}),
        )
        assert any("hardware escalation was opened" in item for item in errors), (
            rung,
            errors,
        )
    assert _successor_errors(evidence, recovered=True) == []
