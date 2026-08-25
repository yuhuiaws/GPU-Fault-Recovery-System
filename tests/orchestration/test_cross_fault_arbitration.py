"""Stable pytest nodeids for split test_cross_fault_arbitration.py cases."""

# ruff: noqa: F401
from ._cross_fault_arbitration_cases_1 import (
    test_attempt_observation_gpu_count_prefers_declared_resources,
    test_cross_node_sxid_upgrade_preserves_prior_xid_gpu,
    test_dcgm_kernel_and_fabric_manager_share_attempt_workflow,
    test_delayed_xid_replay_uses_explicit_attempt_allocation,
    test_disjoint_node_scope_uses_independent_workflows_or_shared_dag,
    test_gpu_ownership_disambiguates_multiple_active_attempts,
    test_later_weaker_event_does_not_reverse_existing_preemption,
    test_multiple_active_attempts_disable_cross_type_grouping,
    test_node_reboot_is_not_downgraded_by_sxid_reset,
    test_running_stronger_workflow_ignores_later_weaker_action,
    test_running_workflow_absorbs_same_action_and_scope,
    test_running_workflow_widens_scope_before_reset_is_submitted,
    test_running_xid_workflow_adds_terminal_quarantine_branch,
    test_same_rank_incompatible_actions_do_not_absorb,
    test_three_collectors_concurrently_merge_atomically,
    test_xid_and_sxid_share_one_attempt_workflow,
)
from ._cross_fault_arbitration_cases_2 import (
    test_cross_node_sxid_only_resets_its_hardware_node,
    test_dag_appends_same_node_successor_after_branch_started,
    test_dag_compares_recovery_rank_within_the_target_node_branch,
    test_dag_widens_unstarted_same_rank_branch_for_another_gpu,
    test_event_after_current_attempt_start_is_processed,
    test_fresh_sxid_waits_for_running_previous_attempt,
    test_fresh_weaker_diagnostic_can_run_with_previous_attempt,
    test_late_node_after_restart_finalization_uses_new_workflow,
    test_missing_attempt_start_time_keeps_fail_safe_behavior,
    test_preemption_marks_stronger_successor_and_reuses_containment,
    test_running_same_action_widens_unsubmitted_reset_for_new_gpu,
    test_running_weaker_workflow_queues_stronger_successor,
    test_source_event_time_takes_precedence_over_collection_time,
    test_stale_same_rank_xid_is_ignored_after_restart,
    test_stale_stronger_action_queues_successor,
    test_stale_weaker_sxid_is_ignored_after_restart,
)
