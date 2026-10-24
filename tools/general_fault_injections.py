"""Executable software checks for each of the 32 manual acceptance categories.

Keep segmented coverage explicit: several older tests inject at separate
collector, API and adapter boundaries and do not prove one continuous workflow.
"""

from __future__ import annotations

from dataclasses import dataclass

HOST = "tests/host_health/test_general_fault_observation_e2e.py::"
EXTENDED = "tests/host_health/test_general_fault_extended_injection.py::test_extended_host_injection"
HEALTH = "tests/execution/test_general_health_policy_e2e.py::"
OBSERVATION = "tests/execution/test_observation_recovery_e2e.py::"
NCCL_CHAIN = "tests/execution/test_nccl_continuous_e2e.py::"
NCCL_REVIEW = "tests/execution/test_nccl_review_e2e.py::"
NCCL_BOUNDARIES = "tests/execution/test_nccl_review_boundaries.py::"
NCCL_BUDGET = "tests/node_agent/test_nccl_collection_budget.py::"
NCCL_OWNERSHIP = "tests/execution/test_nccl_watcher_ownership.py::"


@dataclass(frozen=True)
class Injection:
    number: int
    source: str
    checks: tuple[str, ...]
    continuous: bool = True


INJECTIONS = (
    Injection(
        1,
        "procfs CPU counters",
        (
            HOST + "test_resource_fault_reaches_real_validation[unhealthy-cpu]",
            HOST + "test_resource_fault_reaches_real_validation[recovered-cpu]",
        ),
    ),
    Injection(
        2,
        "procfs meminfo",
        (
            HOST + "test_resource_fault_reaches_real_validation[unhealthy-memory]",
            HOST + "test_resource_fault_reaches_real_validation[recovered-memory]",
            HOST + "test_resource_fault_reaches_real_validation[recovered-swap]",
        ),
    ),
    Injection(
        3,
        "procfs page cache; managed-attempt utilization samples",
        (
            EXTENDED + "[003]",
            "tests/host_health/test_low_utilization_requires_managed_attempt.py::test_running_attempt_on_the_node_fires_once_per_device_with_its_identity",
        ),
    ),
    Injection(
        4,
        "statvfs capacity",
        (HOST + "test_hardware_fault_reaches_real_quarantine_adapter[filesystem]",),
    ),
    Injection(5, "NFS mountinfo and statvfs capacity", (EXTENDED + "[005]",)),
    Injection(6, "procfs diskstats cumulative counters", (EXTENDED + "[006]",)),
    Injection(
        7,
        "smartctl JSON and query failure",
        (
            HOST + "test_hardware_fault_reaches_real_quarantine_adapter[smart]",
            HOST + "test_observation_failure_does_not_isolate_node[smart-query]",
        ),
    ),
    Injection(
        8,
        "net sysfs operstate and missing required interface",
        (
            HOST + "test_hardware_fault_reaches_real_quarantine_adapter[net-down]",
            HOST + "test_hardware_fault_reaches_real_quarantine_adapter[net-missing]",
            HOST + "test_observation_failure_does_not_isolate_node[net-unknown]",
        ),
    ),
    Injection(9, "net sysfs receive error counter", (EXTENDED + "[009]",)),
    Injection(10, "RDMA sysfs port_rcv_errors counter", (EXTENDED + "[010]",)),
    Injection(
        11,
        "RDMA/CRI files through API, signed NodeAction, archives and real fabric validation; device/notification boundaries simulated",
        (
            NCCL_CHAIN
            + "test_raw_rdma_to_real_collection_archive_and_operator_handoff",
            NCCL_CHAIN
            + "test_expansion_budget_coverage_and_bundle_are_persisted_end_to_end",
            NCCL_CHAIN
            + "test_partial_evidence_still_reaches_review_without_claiming_coverage",
            NCCL_CHAIN
            + "test_fabric_waits_for_missing_telemetry_and_never_closes_from_process_data_alone",
            NCCL_CHAIN + "test_bundle_failure_cannot_be_reported_as_training_recovered",
            NCCL_CHAIN + "test_raw_counter_negative_paths_do_not_start_hung_collection",
            NCCL_CHAIN + "test_nccl_log_is_evidence_only_without_rdma_history",
            NCCL_CHAIN
            + "test_multiple_reporting_nodes_share_one_raw_triggered_collection",
            NCCL_CHAIN
            + "test_executor_restart_retains_completed_nodes_and_pending_deadlines",
            NCCL_CHAIN + "test_long_observation_gap_is_not_counted_as_continuous_idle",
            NCCL_CHAIN
            + "test_lost_response_and_agent_restart_replay_without_recapturing",
            NCCL_CHAIN
            + "test_missing_workload_scope_skips_attachments_and_requires_operator_review",
            NCCL_CHAIN
            + "test_whole_task_communication_summary_prioritizes_an_unsampled_lagging_node",
            NCCL_CHAIN
            + "test_summary_byte_budget_exhaustion_is_unknown_not_full_coverage",
            NCCL_REVIEW
            + "test_inherited_rank_and_same_start_tick_child_do_not_remove_training_rank",
            NCCL_REVIEW
            + "test_process_bound_registration_resolves_rank_without_environment",
            NCCL_REVIEW
            + "test_pg_status_by_numeric_id_and_both_original_dumps_survive_rotation",
            NCCL_REVIEW
            + "test_missing_clock_bound_cannot_authorize_cross_node_targeted_capture",
            NCCL_REVIEW
            + "test_management_link_is_not_a_substitute_for_task_nic_binding",
            NCCL_REVIEW + "test_explicit_errors_are_evidence_only_while_processes_live",
            NCCL_REVIEW + "test_recheck_is_a_barrier_before_any_active_collection",
            NCCL_BOUNDARIES
            + "test_pg_status_is_versioned_and_never_indexed_by_group_name",
            NCCL_BOUNDARIES
            + "test_persistent_collective_semantics_are_not_just_sequence_numbers",
            NCCL_BOUNDARIES
            + "test_p2p_evidence_does_not_trigger_topology_guessing_or_rank_attribution",
            NCCL_BOUNDARIES
            + "test_two_sweep_barrier_repairs_disjoint_batches_without_removing_time_checks",
            NCCL_BOUNDARIES
            + "test_training_nic_down_or_missing_cannot_be_hidden_by_management_link",
            NCCL_BOUNDARIES
            + "test_offline_official_analyzer_missing_dependency_is_not_a_success",
            NCCL_BOUNDARIES + "test_error_classification",
            NCCL_BUDGET
            + "test_real_owned_diagnostic_subprocess_is_time_and_output_bounded",
            NCCL_BUDGET
            + "test_blocked_reads_hold_their_slot_but_return_incomplete_without_thread_growth",
            NCCL_BUDGET
            + "test_per_attempt_frequency_is_persistent_and_node_concurrency_is_one",
            NCCL_OWNERSHIP
            + "test_logs_then_nonzero_exit_follow_generic_watcher_containment_and_restart",
            NCCL_OWNERSHIP + "test_known_exit_or_cleanup_stops_new_hung_collection",
            NCCL_OWNERSHIP
            + "test_ordinary_log_is_retained_without_creating_fault_markers",
            NCCL_OWNERSHIP
            + "test_persistent_rdma_suspicion_remains_independent_of_ordinary_errors",
            NCCL_OWNERSHIP
            + "test_peer_failure_during_low_traffic_window_does_not_launch_hung",
            "tests/execution/test_hung_episode_e2e.py::test_api_sqlite_noise_dedup_stable_rearm_and_second_evidence_round",
            "tests/execution/test_hung_episode_e2e.py::test_raw_counter_recovery_with_default_edge_filter_rearms_without_extra_probes",
            "tests/execution/test_hung_episode_e2e.py::test_recovered_first_reporter_allows_new_wave_on_a_different_node",
            "tests/execution/test_partial_hung_evidence.py::test_partial_timeout_is_retained_without_counting_node_as_full_coverage",
            "tests/node_agent/test_communication_monitor.py::test_real_local_ras_protocol_two_and_json_response",
            "tests/node_agent/test_communication_monitor.py::test_owned_worker_returns_partial_at_deadline",
            "tests/node_agent/test_communication_history_pipeline.py::test_delayed_survey_reads_the_requested_common_epochs_not_latest_cache",
            "tests/node_agent/test_communication_history_pipeline.py::test_ras_local_rank_is_joined_to_verified_pid_without_global_rank",
            "tests/node_agent/test_communication_scope.py::test_authenticated_heartbeat_returns_only_current_task_scope_and_legacy_stays_compatible",
            "tests/node_agent/test_passive_hung_summary.py::test_exited_gpu_process_is_not_a_live_communication_participant",
            "tests/regional/test_efa_traffic_harness_main.py::test_efa_counter_harness_runs_its_real_main_with_isolated_api",
            "tests/execution/test_hung_workflow_hardening.py::test_efa_detection_to_evidence_and_operator_review_never_recovers_on_fabric_alone",
            "tests/execution/test_hung_survey_pipeline.py::test_real_node_summary_drives_two_disjoint_active_waves_and_bounded_bundle",
            "tests/node_agent/test_flight_temporal_collection.py::test_two_real_fifo_requests_are_fresh_process_bound_and_time_bounded",
            "tests/node_agent/test_flight_temporal_collection.py::test_real_fifo_temporal_evidence_drives_persisted_targeted_bundle",
            "tests/execution/test_flight_temporal.py::test_same_enqueue_sequence_can_hide_persistent_completion_lag",
            "tests/execution/test_flight_temporal.py::test_every_member_stalled_is_reported_without_inventing_a_culprit",
            "tests/node_agent/test_process_diagnostics_e2e.py::test_real_capture_checks_exit_code_json_shape_and_does_not_parse_error_text",
            "tests/node_agent/test_process_diagnostics_e2e.py::test_unknown_rank_storage_wait_reaches_workflow_without_strong_attribution",
            "tests/execution/test_process_diagnosis.py::test_legitimate_role_or_window_differences_are_not_peer_outliers",
            "tests/execution/test_process_diagnosis.py::test_fr_completion_progress_does_not_silence_process_io_evidence",
            HEALTH
            + "test_hung_validation_requires_new_application_progress_not_healthy_fabric",
        ),
        True,
    ),
    Injection(
        12,
        "exporter device loss; separate post-reboot inventory validation",
        (
            "tests/collectors/test_gpu_resilience.py::test_dcgm_vanished_device_is_reported_as_device_lost",
            "tests/execution/test_validation.py::test_post_reboot_gpu_inventory_requires_fresh_expected_count",
        ),
        False,
    ),
    Injection(
        13,
        "EFA PCI and driver sysfs; separate unknown-inventory policy",
        (
            "tests/collectors/test_host.py::test_efa_inventory_distinguishes_unbound_driver_from_missing_pci",
            "tests/host_health/test_efa_observation_validity.py::test_unknown_inventory_failure_is_diagnostic_not_reboot",
            "tests/execution/test_validation.py::test_targeted_inventory_recovery_satisfies_validation_gate",
        ),
        False,
    ),
    Injection(
        14,
        "Kubernetes node allocatable JSON; separate restart adapter gate",
        (
            "tests/collectors/test_cloud.py::test_kubernetes_node_resource_collector_detects_allocatable_loss",
            "tests/execution/test_restart_device_plugin_gate.py::test_a_pinned_restart_waits_until_kubelet_reports_the_devices_healthy",
        ),
        False,
    ),
    Injection(
        15,
        "ipmitool four-column sensor output and query failure",
        (
            HOST + "test_hardware_fault_reaches_real_quarantine_adapter[bmc]",
            HOST + "test_observation_failure_does_not_isolate_node[bmc-query]",
        ),
    ),
    Injection(
        16,
        "raw journal JSON and untrusted CRI stdout",
        (
            HEALTH + "test_fatal_host_log_stops_the_owned_attempt_before_quarantine",
            HEALTH + "test_untrusted_fatal_text_cannot_stop_training",
            "tests/collectors/test_node_log_hardening.py::test_old_boot_evidence_cannot_quarantine_current_boot",
            "tests/collectors/test_node_log_hardening.py::test_corrected_context_never_masks_a_fatal_rule",
        ),
    ),
    Injection(
        17,
        "raw journal PID; CRI P/F records across restart",
        (
            HEALTH + "test_application_log_preserves_attribution_and_watcher_ownership",
            "tests/collectors/test_node_log_hardening.py::test_cri_partial_join_survives_restart_and_preserves_identity",
            "tests/collectors/test_node_log_hardening.py::test_exited_rank_keeps_identity_without_live_process_capture",
            "tests/collectors/test_node_log_hardening.py::test_old_or_unbound_training_log_cannot_target_current_attempt",
            "tests/collectors/test_node_log_hardening.py::test_timestamp_ties_resume_by_cursor_across_restart",
            "tests/collectors/test_node_log_hardening.py::test_vacuumed_cursor_replays_timestamp_window",
            "tests/collectors/test_node_log_hardening.py::test_deferred_new_file_keeps_its_first_baseline",
            "tests/collectors/test_node_log_hardening.py::test_missing_journal_is_unknown",
            "tests/collectors/test_node_log_hardening.py::test_oversize_line_resumes_with_bounded_raw_reads",
            "tests/collectors/test_node_log_hardening.py::test_tracking_cap_reports_incomplete_coverage",
            "tests/collectors/test_node_log_hardening.py::test_stalled_file_io_does_not_block_journal_or_spawn_more_workers",
            "tests/collectors/test_node_log_hardening.py::test_cri_continuation_follows_inode_rotation",
            "tests/collectors/test_node_log_hardening.py::test_rejected_cri_batch_preserves_the_durable_fragment",
            "tests/collectors/test_node_log_hardening.py::test_file_read_error_restores_partial_line_before_retry",
        ),
    ),
    Injection(
        18,
        "DCGM exposition temperature and missing-field intervals",
        (
            HEALTH + "test_confirmed_thermal_fault_requires_review_after_cooling",
            HEALTH + "test_unknown_observation_breaks_thermal_confirmation",
            HEALTH + "test_device_temperature_limits_reach_real_containment",
        ),
    ),
    Injection(
        19,
        "DCGM exposition volatile DBE",
        (HEALTH + "test_uncorrectable_ecc_stops_for_analysis_without_automatic_rma",),
    ),
    Injection(
        20,
        "DCGM exposition NVLink CRC counter",
        (HEALTH + "test_counter_diagnostics_never_dispatch_hardware_recovery[nvlink]",),
    ),
    Injection(
        21,
        "DCGM fabric status edge; separate fabric persistence validation",
        (
            "tests/collectors/test_dcgm_fabric_manager_edges.py::test_a_state_change_is_delivered_at_once",
            "tests/metrics/test_fabric_manager_state.py::test_a_fabric_held_in_progress_past_the_window_is_one_node_finding",
            "tests/execution/test_validation_fabric_state_loss.py::test_a_stopped_fabric_manager_around_a_per_gpu_reset_fails_the_node",
        ),
        False,
    ),
    Injection(
        22,
        "DCGM exposition PCIe replay counter",
        (HEALTH + "test_counter_diagnostics_never_dispatch_hardware_recovery[pcie]",),
    ),
    Injection(
        23,
        "DCGM exposition power, cap, utilization and duration in nanoseconds",
        (
            HEALTH + "test_counter_diagnostics_never_dispatch_hardware_recovery[power]",
            HEALTH + "test_power_brake_is_not_power_cap_and_never_reboots_the_node",
            HEALTH
            + "test_thermal_warning_and_duration_execute_only_read_only_diagnostics",
        ),
    ),
    Injection(
        24,
        "DCGM exposition corrected ECC and remapped rows",
        (
            HEALTH
            + "test_counter_diagnostics_never_dispatch_hardware_recovery[correctable-composite]",
        ),
    ),
    Injection(
        25,
        "DCGM exposition missing required field and later restoration",
        (
            HEALTH
            + "test_incomplete_gpu_collection_refreshes_observation_without_hardware_ladder",
        ),
    ),
    Injection(
        26,
        "nvidia-smi inventory for unknown instance type; separate identity storage",
        (
            "tests/collectors/test_gpu_resilience.py::test_gpu_inventory_unknown_instance_type_leaves_expected_count_unset",
            "tests/metrics/test_gpu_metrics.py::test_gpu_inventory_snapshot_persists_latest_per_node",
        ),
        False,
    ),
    Injection(
        27,
        "training-progress API numerical_error and stalled step",
        (
            HEALTH
            + "test_nonfinite_loss_stops_only_its_attempt_and_does_not_restart_bad_checkpoint",
            HEALTH
            + "test_hung_validation_requires_new_application_progress_not_healthy_fabric",
        ),
    ),
    Injection(
        28,
        "Kubernetes terminated-container status; separate failure API",
        (
            "tests/completion/test_completion_controller.py::test_nonzero_exit_submits_terminal_once",
            "tests/completion/test_completion_controller.py::test_normal_failure_captures_logs_before_containment_post",
            "tests/completion/test_completion_service.py::test_no_marker_restarts_once_within_budget",
        ),
        False,
    ),
    Injection(
        29,
        "exhausted durable restart budget and executable recovery workflow",
        (
            "tests/execution/test_restart_budget_exhausted_preflight.py::test_exhausted_budget_runs_the_chain_and_withholds_only_the_restart",
            "tests/execution/test_restart_budget_exhausted_preflight.py::test_exhausted_budget_notification_is_saved_once_across_reclaims",
        ),
        False,
    ),
    Injection(
        30,
        "Kubernetes spare Ready=False through spare controller",
        (OBSERVATION + "test_not_ready_spare_does_not_invent_hardware_reboot",),
    ),
    Injection(
        31,
        "live Agent lease with absent required collector channels",
        (
            OBSERVATION
            + "test_silent_collector_refresh_is_bounded_and_requires_fresh_success",
        ),
    ),
    Injection(
        32,
        "unknown Xid policy input and unknown-product source discovery",
        (
            "tests/policy/test_policy.py::test_unknown_xid_is_quarantined_and_idempotent",
            "tests/collectors/test_gpu_identity_refresh.py::test_kernel_collector_stamps_the_late_product_on_the_first_xid",
        ),
        False,
    ),
)
