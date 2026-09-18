from __future__ import annotations

import pytest

from gpu_fault.models import WorkflowStatus, WorkflowStepStatus
from tests.execution._cov95_runtime_triage import BUNDLE, signal, triage_harness


def test_unrelated_bundle_is_not_rewritten_by_a_triage_step() -> None:
    h = triage_harness({}, linked_bundle=False)
    result = h.execute()
    assert result.status is WorkflowStatus.RUNNING, result
    saved = h.store.get_workflow(h.workflow.request_id)
    assert saved.dag_revision == 0, saved
    assert saved.official_steps[1].parameters["hung_triage_target_pending"] is True, (
        saved
    )
    assert len(h.adapter.calls) == 2, h.adapter.calls


def test_matching_attempt_supplies_missing_rank_and_gpu_identity() -> None:
    h = triage_harness(
        {
            "node_results": {
                "node-a": {"ranks": [None, signal(100, None, 9)]},
                "node-b": {"ranks": [signal(200, None)]},
                "node-c": {"ranks": [signal(300, None)]},
            },
            "not_sampled_nodes": ["node-z", "", "node-z"],
        }
    )
    result = h.execute()
    assert result.status is WorkflowStatus.RUNNING and result.waiting_step_index == 1, (
        result
    )
    saved = h.store.get_workflow(h.workflow.request_id)
    bundle = saved.official_steps[1]
    assert bundle.node_ids == ["node-a", "node-b"], bundle
    assert bundle.gpu_uuids == ["GPU-node-a", "GPU-node-b"], bundle
    assert bundle.parameters["target_pids_by_node"] == {
        "node-a": [100],
        "node-b": [200],
    }, bundle
    decision = bundle.parameters["hung_triage_decision"]
    assert decision["classification"] == "CONFIRMED" and decision["culprit_ranks"] == [
        0
    ], decision
    assert decision["not_sampled_nodes"] == ["node-z"], decision
    triage = saved.step_executions[0]
    assert triage.status is WorkflowStepStatus.SUCCEEDED, triage
    assert triage.details["hung_triage_decision"] == decision, triage
    assert (
        next(call for call in h.adapter.calls if call.step.operation is BUNDLE).step
        == bundle
    ), h.adapter.calls


@pytest.mark.parametrize(
    "unknown_context", ["no-attempt", "no-observation", "foreign-job"]
)
def test_unknown_attempt_context_never_invents_rank_or_gpu_ownership(
    unknown_context: str,
) -> None:
    h = triage_harness(
        {"node_results": {"node-a": {"ranks": [signal(100, None)]}}},
        incident_attempt=None if unknown_context == "no-attempt" else "attempt-a",
        observe_job=(
            None
            if unknown_context == "no-observation"
            else "other-job"
            if unknown_context == "foreign-job"
            else "job-a"
        ),
    )
    result = h.execute()
    assert result.status is WorkflowStatus.RUNNING, result
    bundle = h.store.get_workflow(h.workflow.request_id).official_steps[1]
    decision = bundle.parameters["hung_triage_decision"]
    assert decision["classification"] == "UNDETERMINED", (
        "an attempt ID alone cannot authorize another job's rank and GPU attribution",
        decision,
    )
    assert bundle.parameters["capture_process_state"] is False, bundle
    assert bundle.parameters["target_pids_by_node"] == {}, bundle
    assert bundle.parameters["operator_escalation_required"] is True, bundle


def test_unreachable_rank_retains_known_identity_and_explicit_uncertainty() -> None:
    h = triage_harness(
        {
            "node_results": {
                "node-a": {"ranks": [signal(100, None)]},
                "node-b": {"ranks": [signal(200, None)]},
            },
            "undetermined_nodes": ["node-c", "node-c"],
        }
    )
    result = h.execute()
    assert result.status is WorkflowStatus.RUNNING, result
    bundle = h.store.get_workflow(h.workflow.request_id).official_steps[1]
    decision = bundle.parameters["hung_triage_decision"]
    assert decision["classification"] == "PLAUSIBLE", decision
    assert decision["undetermined_nodes"] == ["node-c"], decision
    assert decision["culprit_ranks"] == [2] and decision["control_ranks"] == [0], (
        decision
    )
    assert bundle.parameters["target_pids_by_node"] == {
        "node-a": [100],
        "node-c": [300],
    }, bundle


def test_symmetric_rank_evidence_escalates_without_attaching_to_every_process() -> None:
    h = triage_harness(
        {
            "node_results": {
                f"node-{letter}": {"ranks": [signal((rank + 1) * 100, rank)]}
                for rank, letter in enumerate("abc")
            }
        }
    )
    result = h.execute()
    assert result.status is WorkflowStatus.RUNNING, result
    bundle = h.store.get_workflow(h.workflow.request_id).official_steps[1]
    assert (
        bundle.parameters["hung_triage_decision"]["classification"]
        == "FABRIC_SUSPECTED"
    ), bundle
    assert bundle.parameters["capture_process_state"] is False, bundle
    assert bundle.parameters["max_processes"] == 0, bundle
    assert bundle.parameters["strace_sample_count"] == 0, bundle


def test_invalid_pids_are_not_sent_as_process_collection_targets() -> None:
    h = triage_harness(
        {
            "node_results": {
                "node-a": {"ranks": [signal("invalid", 0, 9)]},
                "node-b": {"ranks": [signal(200, 1)]},
                "node-c": {"ranks": [signal(300, 2)]},
            }
        },
        observe_job=None,
        incident_job=None,
    )
    result = h.execute()
    assert result.status is WorkflowStatus.RUNNING, result
    bundle = h.store.get_workflow(h.workflow.request_id).official_steps[1]
    assert bundle.parameters["target_pids_by_node"] == {"node-b": [200]}, bundle
    assert bundle.parameters["target_gpu_uuids_by_pid_by_node"] == {}, bundle
    assert bundle.gpu_uuids == [], bundle
