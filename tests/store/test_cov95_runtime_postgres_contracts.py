"""Shared public behavior contracts, repeated against the guarded local PG slot."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gpu_fault.models import WorkflowStatus
from gpu_fault.remote_command_models import RemoteCommandStatus
from tests.store import _cov95_runtime_postgres as postgres
from tests.store import test_cov95_compat_control_records as controls
from tests.store import test_cov95_compat_fleet_records as fleet
from tests.store import test_cov95_compat_notification_retention as notifications
from tests.store import test_cov95_compat_remote_lifecycle as commands
from tests.store import test_cov95_compat_retained_control as retention
from tests.store import test_cov95_compat_signal_state as signals
from tests.store import test_cov95_compat_workflow_budgets as budgets
from tests.store import test_cov95_compat_workflow_reads as workflows
from tests.store._postgres_processor_claim_support import postgres_store_instance


@pytest.fixture
def store(monkeypatch):
    postgres.validated_url()
    monkeypatch.setenv("GPU_FAULT_POSTGRES_HOT_STATE_MODE", "legacy")
    yield from postgres_store_instance()


@pytest.mark.parametrize(
    "contract",
    [
        controls.test_restart_budget_is_scoped_immutable_and_idempotently_releasable,
        controls.test_completion_retention_skips_pins_and_never_spends_limit_on_missing_events,
        controls.test_preempting_successor_read_ignores_nonpending_and_unrelated_children,
        controls.test_hyperpod_identity_and_intent_records_do_not_invoke_provider_actions,
        retention.test_inactive_marker_retention_preserves_incident_pins_and_live_markers,
        retention.test_recent_node_markers_use_trust_activity_and_boot_identity_before_limit,
        fleet.test_agent_compare_and_set_preserves_foreign_and_newer_records,
        fleet.test_fleet_deployment_cas_and_retention_keep_live_waves,
        fleet.test_registry_publish_rejects_missing_generation_without_partial_installation,
        fleet.test_member_cleanup_is_bounded_and_preserves_new_acknowledgements,
        notifications.test_retention_preserves_pins_and_open_work_and_releases_deleted_keys,
        notifications.test_watermark_suppresses_only_unfinished_history_and_fences_its_leases,
        notifications.test_backlog_opt_in_is_sticky_and_does_not_consume_delivery_state,
        workflows.test_incident_state_reads_filter_cluster_nodes_and_states_before_limit,
        workflows.test_active_workflow_reads_bind_job_and_node_without_leaking_foreign_pairs,
        workflows.test_job_recovery_reads_find_live_attempts_and_completed_restart_evidence,
        workflows.test_unhandled_failure_reads_are_bounded_and_do_not_reopen_handled_work,
        workflows.test_paired_saves_preserve_all_event_aliases_across_followup_updates,
        workflows.test_missing_workflow_amend_and_event_link_do_not_create_records,
        budgets.test_workflow_claim_keeps_one_live_owner_and_rotates_epoch_at_expiry,
        budgets.test_budget_extension_refusal_preserves_held_scopes_until_capacity_is_available,
        signals.test_xid_retention_and_time_bounds_do_not_cross_node_or_cluster,
        signals.test_due_correlation_lease_takeover_rejects_the_previous_owner,
        signals.test_xid_occurrence_identity_uses_link_bit_and_event_not_arrival_count,
        signals.test_xid_metric_clear_rearms_a_transition_without_replaying_stale_data,
        signals.test_efa_decision_rejects_missing_state_or_a_different_active_event,
        signals.test_receive_clock_orders_health_signals_without_obeying_a_regressed_node_clock,
        commands.test_unclaimed_expiry_is_bounded_and_counts_only_pending_work,
        commands.test_stale_fence_sweep_preserves_live_and_current_generation_leases,
        commands.test_terminal_command_retention_uses_update_time_and_stable_limits,
    ],
    ids=lambda contract: contract.__name__.removeprefix("test_"),
)
def test_postgres_matches_shared_public_contract(store, contract):
    contract(store)


@pytest.mark.parametrize("invalid", ["unclaimed", "owner", "expired"])
def test_budget_extension_requires_a_current_execution_lease(store, invalid):
    budgets.test_budget_extension_requires_a_current_execution_lease(store, invalid)


@pytest.mark.parametrize("wrong_pointer", ["incident", "workflow"])
def test_paired_write_refuses_inconsistent_identity_atomically(store, wrong_pointer):
    workflows.test_paired_write_rejects_mismatched_identity_without_publishing_half_a_pair(
        store, wrong_pointer
    )


@pytest.mark.parametrize("status", list(RemoteCommandStatus))
def test_bulk_command_cancellation_preserves_inflight_and_terminal_contract(
    store, status
):
    commands.test_workflow_cancellation_distinguishes_queued_from_inflight_commands(
        store, status
    )


@pytest.mark.parametrize("status", list(RemoteCommandStatus))
def test_single_command_preemption_is_state_guarded(store, status):
    commands.test_single_command_preemption_refuses_inflight_and_terminal_work(
        store, status
    )


def test_empty_workflow_and_command_scopes_do_not_expand_the_query(store):
    now = datetime.now(timezone.utc)
    assert store.list_workflows(set()) == []
    assert store.count_held_workflows(set(), dispatchable_at=now) == {}
    assert store.count_held_workflows(None, dispatchable_at=now) == {}
    assert (
        store.list_recent_workflows(set(WorkflowStatus), updated_since=None, limit=10)
        == []
    )
    assert store.list_remote_commands(workflow_request_ids=[]) == []
    assert (
        store.claim_remote_commands("missing", "executor", limit=0, lease_seconds=10)
        == []
    )
    assert (
        store.claim_remote_commands(
            "missing",
            "executor",
            limit=1,
            lease_seconds=10,
            execution_owners=set(),
            accept_batched_steps=False,
        )
        == []
    )
    assert (
        store.find_remote_command_covering_step(
            "missing", 0, "official", fencing_token=1
        )
        is None
    ), "an absent workflow has no compound command covering its step"
    with pytest.raises(ValueError, match="must not be negative"):
        store.list_notifications(limit=-1)


def test_event_read_bounds_and_inactive_signal_batches_remain_narrow(store):
    event = signals.event("event-without-retention")
    assert store.save_xid_event_if_absent(event) is True
    assert store.get_xid_events([]) == {}
    assert store.list_xid_events_for_scopes([]) == {}
    assert store.list_xid_events_for_scopes(
        [(event.cluster_id, event.node_id, None, None)]
    ) == {(event.cluster_id, event.node_id): [event]}
    assert store.list_xid_events(
        event.cluster_id, event.node_id, observed_before=event.observed_at
    ) == [event]
    assert store.list_xid_events(
        event.cluster_id, event.node_id, observed_after=event.observed_at
    ) == [event]
    assert store.claim_health_signal_transitions([]) == []
    assert store.claim_health_signal_transitions(
        [("inactive", False, event.observed_at, 0)]
    ) == [False], "a first inactive sample must not invent an activation"
    assert store.get_health_signal_state("inactive") is None, (
        "an inactive first sample must not allocate persistent signal state"
    )
