"""Stable pytest nodeids for split test_kubernetes.py cases."""

# ruff: noqa: F401
from ._kubernetes_cases_1 import (
    test_fabric_manager_restart_sends_one_idempotent_email,
    test_failure_after_isolation_keeps_incident_quarantined,
    test_kubernetes_adapter_hashes_long_incident_id_for_taint,
    test_kubernetes_adapter_preserves_initial_schedulability_across_isolation_steps,
    test_kubernetes_adapter_preserves_provider_taint,
    test_kubernetes_adapter_rejects_active_isolation_takeover,
    test_kubernetes_adapter_restarts_efa_device_plugin,
    test_kubernetes_adapter_retries_node_isolation_conflict,
    test_kubernetes_adapter_takes_over_terminal_isolation,
    test_kubernetes_adapter_treats_absent_node_as_isolated,
    test_kubernetes_adapter_waits_when_device_plugin_pod_is_absent,
    test_kubernetes_stop_deletes_active_suspended_pytorch_pods,
    test_kubernetes_stop_waits_for_job_to_be_inactive,
    test_reboot_workflow_restarts_workload_only_after_node_recovers,
    test_restore_removes_stale_quarantine_taint_after_ownership_check,
    test_same_incident_stale_workload_generation_is_rejected,
    test_storeless_adapter_takes_over_terminal_node_isolation,
    test_successful_quarantine_keeps_incident_quarantined,
    test_terminal_quarantine_hold_rejects_automatic_reset_takeover,
)
from ._kubernetes_cases_2 import (
    test_kubernetes_auto_resume_guard_reports_only_violating_workloads,
    test_kubernetes_auto_resume_guard_runs_before_pod_termination,
    test_kubernetes_refuses_workload_with_hyperpod_auto_resume,
    test_kubernetes_restart_clones_terminal_job_idempotently,
    test_kubernetes_stop_captures_training_log_before_delete,
    test_kubernetes_stop_rejects_active_incident_takeover,
    test_kubernetes_stop_retries_workload_patch_conflict,
    test_kubernetes_stop_takes_over_terminal_incident,
    test_kubernetes_stops_workload_without_hyperpod_auto_resume,
    test_pytorch_restart_rebinds_node_placement_to_spare,
    test_workload_log_s3_upload_is_gzipped,
)
