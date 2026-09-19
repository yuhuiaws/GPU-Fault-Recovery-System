from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.e2e.regional.destr014_cleanup import _cleanup
from scripts.e2e.regional.destr014_verdicts import (
    operator_hold_reasons,
    product_hold_reasons,
    recovery_cleanup_hold,
    workflow_errors,
)
from tests.regional.test_destr014_branch_exhaustion import (
    FAULT,
    SIBLING,
    happy_incident,
    happy_workflow,
)


def unknown_reboot():
    workflow = happy_workflow()
    workflow.update(
        status="BLOCKED",
        blocked_kind="NEEDS_OPERATOR",
        branch_escalation_counts={FAULT: 1},
        official_steps=workflow["official_steps"][:21],
        superseded_step_indexes=[
            index for index in workflow["superseded_step_indexes"] if index < 21
        ],
        step_executions=[
            item for item in workflow["step_executions"] if item["step_index"] < 21
        ],
    )
    reboot = next(
        item for item in workflow["step_executions"] if item["step_index"] == 10
    )
    reboot["details"] = {
        "step_waiting_timeout_seconds": 300,
        "outcome_unknown": True,
        "manual_confirmation_required": True,
    }
    return workflow


def errors(workflow, *, scenario="unknown"):
    return workflow_errors(
        workflow,
        happy_incident(),
        fault_node=FAULT,
        sibling_node=SIBLING,
        failure_reason=f"node branch escalation exhausted: branch:{SIBLING}",
        reboot_outcome=scenario,
    )


def test_unknown_reboot_requires_operator_hold_not_replacement():
    assert errors(unknown_reboot()) == []


@pytest.mark.parametrize(
    "defect", ["terminal", "kind", "replacement", "restored", "no-proof"]
)
def test_unknown_reboot_cannot_be_relabelled_known_or_released(defect):
    workflow = unknown_reboot()
    if defect == "terminal":
        workflow["status"] = "FAILED"
    elif defect == "kind":
        workflow["blocked_kind"] = "SAFETY_SETTLED"
    elif defect == "replacement":
        workflow["official_steps"].append(
            {"operation": "REPLACE_NODE", "node_ids": [SIBLING]}
        )
    elif defect == "restored":
        workflow["step_executions"].append(
            {"operation": "RESTORE_SCHEDULING", "step_index": 14, "status": "SUCCEEDED"}
        )
    else:
        workflow["step_executions"][7]["details"] = {
            "step_waiting_timeout_seconds": 300
        }
    assert errors(workflow), defect


def test_confirmed_reboot_refusal_is_a_separate_replacement_exhaustion_contract():
    workflow = happy_workflow()
    reboot = next(
        item for item in workflow["step_executions"] if item["step_index"] == 10
    )
    reboot["details"] = {"node_action_not_started": True}
    reboot["error"] = "provider explicitly refused submission"
    assert errors(workflow, scenario="confirmed-failure") == []
    reboot["details"] = {"step_waiting_timeout_seconds": 300}
    assert errors(workflow, scenario="confirmed-failure"), (
        "a timeout alone must not prove confirmed reboot refusal"
    )


def test_hold_reasons_name_the_product_hold_and_the_missing_sibling_proof():
    """The runner's hold has two sources and the report must say which one is
    left: the product still owning recovery, and the sibling's reboot not being
    proven by the host record."""
    settled = {"workflow": happy_workflow(), "commands": []}
    assert product_hold_reasons(settled) == []
    assert recovery_cleanup_hold(settled) is False
    blocked = {"workflow": unknown_reboot(), "commands": [{"status": "WAITING"}]}
    reasons = product_hold_reasons(blocked)
    assert recovery_cleanup_hold(blocked) is True
    assert any("BLOCKED" in r and "NEEDS_OPERATOR" in r for r in reasons), reasons
    assert any("not terminal" in r for r in reasons), reasons
    assert product_hold_reasons({"workflow": None, "commands": []}) == [
        "workflow or remote command inventory is incomplete"
    ]
    proven = {"proven": True, "gaps": []}
    unproven = {"proven": False, "gaps": ["boot id unchanged"]}
    assert operator_hold_reasons([], proven) == []
    assert operator_hold_reasons(reasons, proven) == [f"product: {r}" for r in reasons]
    assert operator_hold_reasons([], unproven) == [
        "sibling reboot is unproven: boot id unchanged"
    ]
    assert len(operator_hold_reasons(reasons, unproven)) == len(reasons) + 1


@pytest.mark.parametrize(
    "defect", ["missing", "leased", "unknown-command", "unknown-step", "open"]
)
def test_reboot_cleanup_requires_a_complete_known_terminal_read(defect):
    state = {"workflow": happy_workflow(), "commands": []}
    assert recovery_cleanup_hold(state) is False
    if defect == "missing":
        state.pop("commands")
    elif defect == "leased":
        state["commands"].append({"status": "LEASED"})
    elif defect == "unknown-command":
        state["commands"].append(
            {"status": "FAILED", "result_details": {"outcome_unknown": True}}
        )
    elif defect == "unknown-step":
        state["workflow"]["step_executions"][0]["details"] = {
            "manual_confirmation_required": True
        }
    else:
        state["workflow"]["status"] = "RUNNING"
    assert recovery_cleanup_hold(state) is True


def test_unknown_reboot_cleanup_preserves_workload_and_isolation(tmp_path):
    warm, workload, recovery = Mock(), Mock(), Mock()
    probe = Mock()
    probe.cleanup.return_value = {}
    prewarm = Mock()
    prewarm.cleanup.return_value = {}
    result = _cleanup(
        regional=Mock(),
        warm=warm,
        workload=workload,
        prewarm=prewarm,
        fault_probe=probe,
        sibling_probe=probe,
        inject_fault=probe,
        inject_sibling=probe,
        settings=SimpleNamespace(sibling_node=SIBLING, fault_node=FAULT),
        run_id="owned-run",
        incident_id="owned-incident",
        env_baseline=tmp_path / "executor.json",
        env_opened=True,
        control_env_baseline=tmp_path / "cpu.json",
        control_env_opened=True,
        holder_armed=False,
        agent_disabled=True,
        profile_version="profile",
        recovery_window=recovery,
        operator_hold=True,
    )
    assert result["operator_hold_preserved"] and result["errors"]
    assert result["workload_cleanup_deferred"]
    workload.delete.assert_not_called()
    warm.create_restore_workflow.assert_not_called()
    warm.reactivate_agent.assert_not_called()
    recovery.cleanup.assert_called_once()
