"""Run provider-correlation behavior against an explicitly allocated PostgreSQL."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from contextlib import closing

import pytest

from gpu_fault.app import default_simulated_profile
from gpu_fault.app.ingest.faults import FaultIngestionService
from gpu_fault.models import FaultIncident
from gpu_fault.store import PostgresStore
from gpu_fault.store.contracts import ControlPlaneStore
from gpu_fault.store.shared.errors import StaleWriteError
from tests._builders import build_context
from tests.orchestration.test_provider_correlation_arbitration import (
    test_access_marker_does_not_hide_trunk_reset_kind_or_gpu_b,
    test_companion_event_id_wins_over_an_unrelated_semantic_marker,
    test_companion_link_to_an_old_diagnostic_cannot_suppress_the_required_reset,
    test_correlated_trunk_candidate_never_rewrites_a_started_access_reset,
    test_exact_companion_still_shares_one_reset_with_aggregation_disabled,
    test_exact_event_replay_returns_current_execution_without_replanning,
    test_execution_progress_after_marker_lookup_is_seen_by_atomic_arbitration,
    test_failed_candidate_merge_does_not_mark_the_event_handled_and_retry_is_idempotent,
    test_fresh_new_attempt_xid48_does_not_bind_to_old_attempt_warning,
    test_later_access_marker_does_not_downgrade_the_trunk_reset,
    test_later_sbe_warning_preserves_the_xid48_reset,
    test_missing_correlated_workflow_cannot_bypass_the_unknown_workload_gate,
    test_new_access_event_with_intersecting_scope_resets_both_gpus,
    test_sbe_marker_cannot_swallow_xid48_reset,
    test_stale_semantic_event_is_fenced_without_overwriting_the_associated_incident,
    test_terminal_marker_incumbent_does_not_count_as_new_reset_coverage,
    test_unsafe_exact_companion_cannot_bypass_a_fresh_candidates_evidence_gate,
)
from tests.orchestration.test_provider_marker_binding import (
    observation_event,
    test_correlated_event_link_waits_for_both_snapshot_guards,
    test_equivalent_provider_observations_share_an_unstarted_plan,
    test_failed_incident_ingestion_does_not_publish_a_marker,
    test_finalized_xid_marker_belongs_to_its_new_attempt_recovery,
    test_marker_write_failure_refuses_finalization_and_retries_same_workflow,
    test_progressed_provider_plan_cannot_absorb_another_observation,
    test_retained_policy_must_match_the_correlated_execution_owner,
    test_semantic_drift_does_not_reuse_an_independent_provider_plan,
)
from tests.store._postgres_processor_claim_support import _truncate

__all__ = [
    "test_access_marker_does_not_hide_trunk_reset_kind_or_gpu_b",
    "test_companion_event_id_wins_over_an_unrelated_semantic_marker",
    "test_companion_link_to_an_old_diagnostic_cannot_suppress_the_required_reset",
    "test_correlated_event_link_waits_for_both_snapshot_guards",
    "test_correlated_trunk_candidate_never_rewrites_a_started_access_reset",
    "test_equivalent_provider_observations_share_an_unstarted_plan",
    "test_exact_companion_still_shares_one_reset_with_aggregation_disabled",
    "test_exact_event_replay_returns_current_execution_without_replanning",
    "test_execution_progress_after_marker_lookup_is_seen_by_atomic_arbitration",
    "test_failed_candidate_merge_does_not_mark_the_event_handled_and_retry_is_idempotent",
    "test_failed_incident_ingestion_does_not_publish_a_marker",
    "test_finalized_xid_marker_belongs_to_its_new_attempt_recovery",
    "test_fresh_new_attempt_xid48_does_not_bind_to_old_attempt_warning",
    "test_later_access_marker_does_not_downgrade_the_trunk_reset",
    "test_later_sbe_warning_preserves_the_xid48_reset",
    "test_marker_write_failure_refuses_finalization_and_retries_same_workflow",
    "test_missing_correlated_workflow_cannot_bypass_the_unknown_workload_gate",
    "test_new_access_event_with_intersecting_scope_resets_both_gpus",
    "test_progressed_provider_plan_cannot_absorb_another_observation",
    "test_retained_policy_must_match_the_correlated_execution_owner",
    "test_sbe_marker_cannot_swallow_xid48_reset",
    "test_semantic_drift_does_not_reuse_an_independent_provider_plan",
    "test_stale_semantic_event_is_fenced_without_overwriting_the_associated_incident",
    "test_terminal_marker_incumbent_does_not_count_as_new_reset_coverage",
    "test_unsafe_exact_companion_cannot_bypass_a_fresh_candidates_evidence_gate",
]

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)


@pytest.fixture
def correlation_store() -> Iterator[ControlPlaneStore]:
    assert POSTGRES_URL is not None, "the fixture requires an allocated test database"
    _truncate()
    store = PostgresStore(POSTGRES_URL)
    store.save_profile(default_simulated_profile())
    try:
        yield store
    finally:
        store.close()
        _truncate()


@pytest.fixture
def binding_store(correlation_store: ControlPlaneStore) -> ControlPlaneStore:
    return correlation_store


def test_postgres_correlated_observation_refuses_a_concurrent_workflow_claim(
    correlation_store: ControlPlaneStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert POSTGRES_URL is not None, "the test requires an owned PostgreSQL allocation"
    context = build_context(store=correlation_store)
    service = FaultIngestionService(context)
    first = service.ingest_xid(observation_event("before-concurrent-claim"))
    assert first.workflow_request_id is not None, "the initial fault requires a plan"
    workflow = correlation_store.get_workflow(first.workflow_request_id)
    event = observation_event("after-concurrent-claim", seconds=1)
    original = correlation_store.save_incident
    claimed = False

    with closing(PostgresStore(POSTGRES_URL, initialize_schema=False)) as other:

        def claim_before_binding(
            incident: FaultIncident,
            *,
            expected: FaultIncident | None = None,
            extra_event_ids: Sequence[str] = (),
        ) -> None:
            nonlocal claimed
            if not claimed and expected is not None:
                other.claim_workflow(
                    workflow.request_id, "other-executor", workflow.fencing_token
                )
                claimed = True
            original(incident, expected=expected, extra_event_ids=extra_event_ids)

        with monkeypatch.context() as race:
            race.setattr(correlation_store, "save_incident", claim_before_binding)
            with pytest.raises(StaleWriteError):
                service.ingest_xid(event)

        assert claimed, "the independent connection must claim after the coverage read"
        assert correlation_store.get_incident_by_event(event.event_id) is None, (
            "the failed workflow CAS must roll back the event association"
        )
        assert (
            other.get_workflow(workflow.request_id).execution_owner_id
            == "other-executor"
        ), "the correlation transaction must not overwrite the concurrent claimant"

        retried = service.ingest_xid(event)
        assert retried.workflow_request_id != workflow.request_id, (
            "retry must re-evaluate the now-started incumbent before linking the event"
        )
