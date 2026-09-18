from __future__ import annotations

import pytest

from gpu_fault.hyperpod import (
    HyperPodAction,
    HyperPodAdapterError,
    HyperPodLifecycleAdapter,
    HyperPodSubmissionRecord,
    HyperPodSubmissionResult,
    HyperPodWorkflowDispatcher,
)
from gpu_fault.models import WorkflowOperation, WorkflowStatus
from gpu_fault.store import InMemoryStore
from tests._builders import workflow_step
from tests.hyperpod._cov95_provider_extra_root import OWNER, TARGET, RootHarness
from tests.hyperpod._cov95_provider_extra_safety import (
    provider_extra_isolation as provider_extra_isolation,
)


def test_process_local_idempotency_cannot_reuse_the_key_for_another_request():
    h = RootHarness(durable=False)
    first = h.submit()
    with pytest.raises(HyperPodAdapterError, match="already used.*different"):
        h.submit(["worker-group-2"], isolation_verified_nodes=["worker-group-2"])
    assert len(h.client.reboot_requests) == 1, (
        "conflicting idempotency replay submitted again"
    )
    assert first.requested_node_logical_ids == [TARGET], (
        "the original request was overwritten"
    )


def test_lost_success_record_leaves_an_intent_that_blocks_another_provider_submission(
    caplog,
):
    class UnwritableOutcome(InMemoryStore):
        def save_hyperpod_submission(self, record):
            raise OSError("unit outcome persistence unavailable")

    h = RootHarness(store=UnwritableOutcome())
    first = h.submit()
    assert first.submitted is True, (
        "a completed provider call lost its returned outcome"
    )
    record = h.store.get_hyperpod_submission(
        "hp-cluster", h.arguments["idempotency_key"]
    )
    assert record.state == "INTENDED", (
        "outcome failure reopened the reserved submission key"
    )
    retry = HyperPodLifecycleAdapter(h.config, client=h.client, store=h.store)
    with pytest.raises(HyperPodAdapterError, match="already reserved"):
        retry.submit(HyperPodAction.REBOOT, [TARGET], **h.arguments)
    assert len(h.client.reboot_requests) == 1, (
        "lost persistence caused a second provider action"
    )
    assert any(
        "could not persist" in record.getMessage() for record in caplog.records
    ), "the operator received no durable-outcome persistence diagnostic"


def test_unknown_provider_outcome_remains_nonretryable_across_a_new_adapter():
    h = RootHarness()
    h.client.submit_error = TimeoutError("unit provider acknowledgement lost")
    with pytest.raises(TimeoutError, match="acknowledgement lost"):
        h.submit()
    record = h.store.get_hyperpod_submission(
        "hp-cluster", h.arguments["idempotency_key"]
    )
    assert record.state == "UNKNOWN", (
        "an ambiguous provider result was treated as unsubmitted"
    )
    h.client.submit_error = None
    retry = HyperPodLifecycleAdapter(h.config, client=h.client, store=h.store)
    with pytest.raises(HyperPodAdapterError, match="unknown outcome"):
        retry.submit(HyperPodAction.REBOOT, [TARGET], **h.arguments)
    assert len(h.client.reboot_requests) == 1, "an unknown outcome was submitted again"


def test_concurrent_reservation_winner_is_replayed_without_a_loser_submission(
    monkeypatch,
):
    h = RootHarness()
    winner = HyperPodSubmissionResult(
        operation_id="other-replica-operation",
        idempotency_key=h.arguments["idempotency_key"],
        action=HyperPodAction.REBOOT,
        cluster_name="hp-cluster",
        requested_node_logical_ids=[TARGET],
        successful_node_logical_ids=[TARGET],
    )
    original = h.store.reserve_hyperpod_submission

    def reserve(record):
        saved, created = original(record)
        assert created is True, (
            "the test's simulated concurrent winner did not reserve its key"
        )
        h.store.save_hyperpod_submission(
            saved.model_copy(update={"state": "SUBMITTED", "result": winner})
        )
        return original(record)

    monkeypatch.setattr(h.store, "reserve_hyperpod_submission", reserve)
    result = h.submit()
    assert result.duplicate is True, (
        "the losing replica did not identify the durable replay"
    )
    assert result.operation_id == winner.operation_id, (
        "the losing replica invented another operation"
    )
    assert h.client.reboot_requests == [], (
        "the reservation loser submitted a provider action"
    )


def test_result_keeps_both_provider_failure_formats_without_claiming_all_nodes_succeeded():
    h = RootHarness()
    h.client.submit_response = {
        "SuccessfulNodeLogicalIds": [],
        "FailedNodeLogicalIds": [{"NodeLogicalId": TARGET, "ErrorCode": "Conflict"}],
        "Failed": [{"NodeId": "legacy-node", "Message": "unit legacy rejection"}],
    }
    result = h.submit()
    assert result.successful_node_logical_ids == [], (
        "rejected nodes were recorded as successful"
    )
    assert [(item.node_logical_id, item.node_id) for item in result.failures] == [
        (TARGET, None),
        (None, "legacy-node"),
    ], "one of the documented provider failure formats was dropped"
    saved = h.store.get_hyperpod_submission(
        "hp-cluster", h.arguments["idempotency_key"]
    )
    assert saved.result == result, (
        "the durable record lost the provider rejection evidence"
    )


@pytest.mark.parametrize(
    "status", [WorkflowStatus.BLOCKED, WorkflowStatus.FAILED, WorkflowStatus.SUCCEEDED]
)
def test_dispatcher_refuses_nonexecutable_workflow_state_before_provider_reads(status):
    h = RootHarness()
    workflow = h.workflow(status=status)
    with pytest.raises(HyperPodAdapterError, match="PENDING or RUNNING"):
        HyperPodWorkflowDispatcher(h.adapter).preflight(
            workflow, 0, isolation_verified_nodes=[TARGET]
        )
    assert h.client.cluster_reads == [], (
        "a nonexecutable workflow reached provider preflight"
    )
    assert h.client.reboot_requests == [], "a nonexecutable workflow submitted a reboot"


@pytest.mark.parametrize("index", [1, 20])
def test_dispatcher_rejects_indexes_past_the_plan_before_provider_reads(index):
    h = RootHarness()
    with pytest.raises(HyperPodAdapterError, match="out of range"):
        HyperPodWorkflowDispatcher(h.adapter).preflight(
            h.workflow(), index, isolation_verified_nodes=[TARGET]
        )
    assert h.client.cluster_reads == [], (
        "an invalid step index triggered provider reads"
    )


def test_dispatcher_rejects_negative_step_indexes_instead_of_selecting_the_last_step():
    h = RootHarness()
    with pytest.raises(HyperPodAdapterError, match="step index|out of range"):
        HyperPodWorkflowDispatcher(h.adapter).submit(
            h.workflow(),
            -1,
            isolation_verified_nodes=[TARGET],
            confirm_cluster_name="hp-cluster",
            expected_fencing_token=7,
        )
    assert h.client.reboot_requests == [], (
        "negative indexing dispatched an unintended plan step"
    )


def test_dispatcher_cannot_route_a_nonprovider_operation_even_with_the_provider_owner():
    h = RootHarness()
    workflow = h.workflow(
        official_steps=[workflow_step(WorkflowOperation.FREEZE_EVIDENCE, OWNER)]
    )
    with pytest.raises(HyperPodAdapterError, match="only supports"):
        HyperPodWorkflowDispatcher(h.adapter).preflight(
            workflow, 0, isolation_verified_nodes=[TARGET]
        )
    assert h.client.cluster_reads == [], (
        "an unsupported operation reached provider preflight"
    )


def test_submission_record_identity_is_order_independent_but_does_not_change_its_request():
    record = HyperPodSubmissionRecord(
        cluster_name="hp-cluster",
        idempotency_key="unit-identity",
        action=HyperPodAction.REBOOT,
        requested_node_identifiers=["node-b", "node-a"],
    )
    assert record.request_identity == (HyperPodAction.REBOOT, ("node-a", "node-b")), (
        "durable identity depends on the caller's node ordering"
    )
    assert record.requested_node_identifiers == ["node-b", "node-a"], (
        "computing identity mutated the stored original request"
    )
