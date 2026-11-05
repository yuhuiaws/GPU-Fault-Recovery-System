"""b220's workload-domain oracle on records that break the domain contract.

``test_workload_domain_oracle.py`` judges the parallel, serialized, tripped and
unreadable layouts; this module reuses its record builders (``part``,
``snapshot``, ``judge``) and bends one domain at a time the ways a product can:
two domains on one workflow, an incident of another cluster/workflow/node/job,
a plan that acts beyond its node or misses its job tokens, an operator's
reconcile, claims beyond the domain or a wider own limit, a wait without its
scope or a refusal without its event, a breaker window that counted the wrong
node or domain, an open incident the case left behind -- plus the oracle's own
input checks and time parsing.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from typing import Any

import pytest

from gpu_fault.models import WorkflowEvent, WorkflowEventKind
from gpu_fault.remediation_storm import StormEntry
from scripts.e2e.regional import workload_domain_oracle as oracle
from tests.regional.test_workload_domain_oracle import (
    CLUSTER,
    at,
    judge,
    snapshot,
    two_idle,
    two_jobs,
)


def _contract(parts: tuple[Any, ...], value: dict[str, Any] | None = None) -> tuple:
    verdict = judge(value or snapshot(*parts), parts, counted=2)
    assert (verdict.verdict, verdict.outcome) == (oracle.FAIL, oracle.CONTRACT), (
        verdict.errors,
        verdict.inconclusive,
    )
    return verdict.errors


def _step(workflow: dict[str, Any], operation: str) -> dict[str, Any]:
    return next(
        item for item in workflow["official_steps"] if item["operation"] == operation
    )


# --------------------------------------------------------------------------- #
# The oracle's own inputs
# --------------------------------------------------------------------------- #
def test_inputs_the_oracle_cannot_judge_are_refused_or_read_as_unknown() -> None:
    with pytest.raises(ValueError, match="a node-group minimum must be positive"):
        oracle.node_group_gate([], minimum=0, cluster_id=CLUSTER)
    with pytest.raises(ValueError, match="unknown overlap span 'sideways'"):
        judge(snapshot(*two_idle()), two_idle(), overlap="sideways")
    assert oracle.job_scopes(CLUSTER, "", ["default/pytorchjob/a", ""]) == {
        f"job:{CLUSTER}:default/pytorchjob/a"
    }, "no job id: no job-id token"
    naive = datetime(2026, 11, 1, 12, 0)
    assert oracle.parse_time(naive) == naive.replace(tzinfo=timezone.utc)
    assert oracle.parse_time("11/01/2026 12:00") is None
    assert oracle.parse_time(None) is None
    assert (
        oracle.execution_span([{"operation": "RESET_GPU", "status": "RUNNING"}])
        == "RESET_GPU has no terminal step execution"
    )
    assert (
        oracle.execution_span(
            [
                {
                    "operation": "RESET_GPU",
                    "status": "SUCCEEDED",
                    "started_at": "yesterday",
                    "updated_at": at(9).isoformat(),
                }
            ]
        )
        == "a step execution has no readable started_at/updated_at"
    )


# --------------------------------------------------------------------------- #
# One domain, one workflow, one incident
# --------------------------------------------------------------------------- #
def test_two_domains_on_one_workflow_are_merged_faults() -> None:
    first, second = two_idle()
    second.expectation = dataclasses.replace(
        second.expectation, workflow_id=first.expectation.workflow_id
    )
    errors = judge(snapshot(first, second), (first, second), counted=2).errors
    assert (
        "two domains name one workflow: the faults were merged, not repaired as "
        "separate domains"
    ) in errors, errors


def test_an_incident_of_another_cluster_workflow_node_or_job_breaks_the_contract() -> (
    None
):
    first, second = two_idle()
    second.incident.update(
        cluster_id="other-cluster",
        workflow_request_id="workflow-elsewhere",
        node_ids=["node-z"],
        job_id="job-q",
    )
    errors = _contract((first, second))
    assert errors[:4] == (
        "node-b: incident belongs to other-cluster",
        "node-b: incident incident-node-b names workflow workflow-elsewhere, not "
        "workflow-node-b",
        "node-b: incident nodes ['node-z'] are not the faulted node node-b",
        "node-b: incident job is job-q, expected none (an idle node)",
    ), errors


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #
def test_an_idle_plan_must_reset_its_own_node_and_stop_no_workload() -> None:
    first, second = two_idle()
    steps = second.workflow["official_steps"]
    _step(second.workflow, "RESET_GPU")["node_ids"] = ["node-b", "node-c"]
    _step(second.workflow, "VALIDATE_GPU")["node_ids"] = ["node-b", "node-z"]
    steps.append({**steps[0], "operation": "STOP_WORKLOADS", "node_ids": ["node-b"]})
    second.workflow["events"].append(
        WorkflowEvent(
            kind=WorkflowEventKind.OPERATOR_RECONCILED, at=at(30), code="manual-close"
        ).model_dump(mode="json")
    )
    errors = _contract((first, second))
    assert errors[:5] == (
        "node-b: expected one RESET_GPU step on node-b, found [['node-b', 'node-c']]",
        "node-b: RESET_GPU acts on ['node-c'] beyond its own node",
        "node-b: VALIDATE_GPU acts on ['node-z'] beyond its own node",
        "node-b: an idle node's workflow stops or restarts a workload: "
        "['STOP_WORKLOADS']",
        "node-b: an operator reconciled the workflow (['manual-close']); the "
        "recovery must be the product's own",
    ), errors


def test_a_job_plan_needs_both_of_its_workload_tokens() -> None:
    first, second = two_jobs()
    second.workflow["official_steps"] = [
        item
        for item in second.workflow["official_steps"]
        if item["operation"] != "RESTART_WORKLOAD"
    ]
    errors = _contract((first, second))
    assert "job-b: the job workflow has no ['RESTART_WORKLOAD']" in errors, errors


# --------------------------------------------------------------------------- #
# Claims and holds
# --------------------------------------------------------------------------- #
def test_a_claim_beyond_the_domain_and_a_wider_own_limit_fail() -> None:
    first, second = two_idle()
    own = f"node:{CLUSTER}:node-b"
    second.workflow["remediation_budget_claims"] = sorted(
        [own, f"node:{CLUSTER}:node-z"]
    )
    second.workflow["remediation_budget_limits"] = {own: 2}
    errors = _contract((first, second))
    assert errors[:2] == (
        f"node-b: claims node:{CLUSTER}:node-z, outside its own node",
        f"node-b: {own} limit is 2, not 1",
    ), errors


def test_a_wait_without_a_scope_fails_and_a_refusal_without_its_event_is_unknown() -> (
    None
):
    first, second = two_idle()
    second.workflow["events"].append(
        WorkflowEvent(
            kind=WorkflowEventKind.HOLD,
            at=at(1),
            code=oracle.WAIT_CODE,
            actor="remediation-budget",
            reason="held",
            details={"reason": oracle.WAIT_CODE},
        ).model_dump(mode="json")
    )
    assert "node-b: a budget wait names no scope" in _contract((first, second))

    first, second = two_idle()
    second.workflow["remediation_budget_wait_count"] = 2
    verdict = judge(snapshot(first, second), (first, second), counted=2)
    assert verdict.verdict == oracle.INCONCLUSIVE, verdict.errors
    assert verdict.inconclusive == (
        f"node-b: refused 2 time(s) without a {oracle.WAIT_CODE} event (a release "
        "before the wait-age change refused it)",
    )


# --------------------------------------------------------------------------- #
# The breaker's count and the residue
# --------------------------------------------------------------------------- #
def _entry(item: StormEntry, **changes: Any) -> StormEntry:
    return StormEntry(**{**item.model_dump(), **changes})


def test_the_breaker_must_count_each_domain_on_its_node_in_its_own_domain() -> None:
    first, second = two_idle()
    second.window = [_entry(second.window[0], domain="job:job-q")]
    errors = _contract((first, second))
    assert (
        "node-b: the breaker counted it in workload domain job:job-q, not "
        "incident:incident-node-b"
    ) in errors, errors

    first, second = two_idle()
    second.window = [_entry(second.window[0], node_id="node-z")]
    errors = _contract((first, second))
    assert (
        "node-b: the breaker counted ['node-z'] for it, not its faulted node node-b"
    ) in errors, errors


def test_an_open_incident_the_case_left_behind_is_residue() -> None:
    parts = two_idle()
    value = snapshot(*parts)
    value["open_incidents"] = [
        {
            "incident_id": "incident-new",
            "state": "ESCALATED",
            "node_ids": ["node-b"],
            "created_at": at(30).isoformat(),
        }
    ]
    assert _contract(parts, value) == (
        "residue: incident incident-new is still ESCALATED on ['node-b']",
    )
