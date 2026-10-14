"""The product's account of a reset that did not hold, read from the store.

The state is shaped like the store snapshot DESTR-001's runner got on plain EKS
when RESET_GPU failed busy and the product escalated the branch to RESTART_NODE.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from scripts.e2e.regional.reset_escalation_evidence import (
    TEXT_LIMIT,
    escalation_errors,
    isolation_errors,
    post_case_isolation,
    product_escalation,
    product_reboot,
    reboot_proof_error,
)
from tests.regional._destructive_acceptance_builders import (
    PLANTED_SECRET,
    _reset_state,
    escalated_reset_state,
)


def test_summary_reads_the_escalated_reset_of_the_live_store_shape() -> None:
    summary = product_escalation(escalated_reset_state())

    assert summary["escalated"] is True, summary
    assert summary["workflow_status"] == "SUCCEEDED", summary
    assert summary["node_lifecycle_completed"] == ["RESTART_NODE"], summary
    (escalation,) = summary["escalations"]
    reason = escalation.pop("reason")
    assert escalation == {
        "from_operation": "RESET_GPU",
        "to_operation": "RESTART_NODE",
        "node_id": "node-a",
        "rung_count": 1,
        "max_rungs": 2,
        "step_index": 4,
        "at": "2030-01-01T00:01:00Z",
    }, escalation
    assert reason.startswith("RESET_GPU failed on node-a (node agent node-a:"), reason
    assert len(reason) == TEXT_LIMIT and reason.endswith("…"), reason
    (failed,) = summary["failed_steps"]
    assert (failed["operation"], failed["step_index"]) == ("RESET_GPU", 4), failed
    assert failed["error"].startswith("node agent node-a: ResetProgressError"), failed
    assert "\n" not in failed["error"] and len(failed["error"]) <= TEXT_LIMIT, failed
    assert PLANTED_SECRET not in json.dumps(summary), (
        "no lease digest, fencing token, challenge nonce or signature is copied"
    )


def test_escalation_error_names_the_agent_failure_and_the_target_rung() -> None:
    summary = product_escalation(escalated_reset_state())

    (line,) = escalation_errors(summary)

    assert line.startswith(
        "RESET_GPU failed on node-a: ResetProgressError: nvidia-smi exited with "
        "status 255: The following GPUs could not be reset: GPU 00000000:59:00.0: "
        "In use by another client"
    ), line
    assert line.endswith("; the product escalated the branch to RESTART_NODE"), line


def test_escalation_error_falls_back_to_the_escalation_reason() -> None:
    state = escalated_reset_state()
    workflow = state["workflow"]
    workflow["step_executions"] = [
        item for item in workflow["step_executions"] if item["status"] != "FAILED"
    ]

    (line,) = escalation_errors(product_escalation(state))

    assert line.startswith(
        "RESET_GPU failed on node-a: RESET_GPU failed on node-a ("
    ), line
    assert line.endswith("escalated the branch to RESTART_NODE"), line


@pytest.mark.parametrize("source", ["completed_operations", "executions", "commands"])
def test_a_lifecycle_rung_counts_from_any_store_record_of_its_success(
    source: str,
) -> None:
    state = escalated_reset_state(restart_status="FAILED")
    workflow = state["workflow"]
    if source == "completed_operations":
        workflow["completed_operations"].append("REPLACE_NODE")
    elif source == "executions":
        workflow["step_executions"].append(
            {"operation": "REPLACE_NODE", "step_index": 13, "status": "SUCCEEDED"}
        )
    else:
        state["commands"].append(
            {
                "status": "SUCCEEDED",
                "step": {"operation": "VALIDATE_GPU"},
                "batched_steps": [{"step": {"operation": "REPLACE_NODE"}}],
            }
        )

    assert product_escalation(state)["node_lifecycle_completed"] == ["REPLACE_NODE"]


def test_a_reset_that_held_has_nothing_to_report() -> None:
    summary = product_escalation(_reset_state())

    assert summary == {
        "escalated": False,
        "escalations": [],
        "failed_steps": [],
        "node_lifecycle_completed": [],
        "workflow_status": "SUCCEEDED",
    }, summary
    assert escalation_errors(summary) == [], summary
    assert product_escalation({}) == {**summary, "workflow_status": None}


@pytest.mark.parametrize(
    ("restart", "before", "after", "expected"),
    [
        ("SUCCEEDED", "boot-1", "boot-1", "RESTART_NODE completed"),
        (
            "SUCCEEDED",
            "boot-1",
            "boot-2",
            "RESTART_NODE completed; boot id changed from boot-1 to boot-2",
        ),
        ("FAILED", "boot-1", "boot-2", "boot id changed from boot-1 to boot-2"),
        ("FAILED", "boot-1", "boot-1", None),
        ("FAILED", None, "boot-2", None),
    ],
)
def test_a_reboot_is_a_completed_rung_or_a_new_boot_id(
    restart: str, before: str | None, after: str | None, expected: str | None
) -> None:
    summary = product_escalation(escalated_reset_state(restart_status=restart))

    reason = product_reboot(summary, {"boot_id": before}, {"boot_id": after})

    assert reason == expected
    if reason is not None:
        who = (
            "the product rebooted the node"
            if restart == "SUCCEEDED"
            else "the node rebooted during the case"
        )
        assert reboot_proof_error(summary, reason) == (
            "the physical single-GPU reset proof is unavailable because "
            f"{who} ({reason}): the host probe Pod and the reset sampler ended "
            "with that boot"
        )


def _node(**changes: Any) -> dict[str, Any]:
    return {
        "unschedulable": False,
        "taints": [
            {"key": "nvidia.com/gpu", "value": "present", "effect": "NoSchedule"}
        ],
        "ownership_annotations": {},
        **changes,
    }


def test_a_released_node_is_not_isolated() -> None:
    isolation = post_case_isolation(_node(), "inc-case")

    assert isolation == {
        "unschedulable": False,
        "taints": [],
        "ownership_incident": None,
        "owned_by_case_incident": False,
        "other_incident_owner": None,
    }, isolation
    assert isolation_errors(isolation) == []


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        (
            _node(ownership_annotations={"gpu-fault.io/incident-id": "inc-other"}),
            "owned by another incident inc-other; no gpu-fault.io taint; schedulable",
        ),
        (
            _node(
                unschedulable=True,
                taints=[{"key": "gpu-fault.io/quiesce", "effect": "NoExecute"}],
            ),
            "no incident ownership annotation; taint gpu-fault.io/quiesce:NoExecute; "
            "unschedulable",
        ),
    ],
)
def test_isolation_error_says_who_and_what_still_holds_the_node(
    node: dict[str, Any], expected: str
) -> None:
    errors = isolation_errors(post_case_isolation(node, "inc-case"))

    assert errors == [
        "target node is still isolated at the end of the case: " + expected
    ], errors
