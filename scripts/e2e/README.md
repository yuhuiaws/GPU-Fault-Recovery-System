# E2E drivers and probes

This directory contains live-cluster drivers and focused audit probes.
Kubernetes inputs are isolated under `manifests/`; follow that directory's
rendering and target-node instructions before applying anything.

## Scenario drivers

- `run_hyperpod_dcgm_metrics_e2e.py`
- `run_hyperpod_efa_traffic_e2e.py`
- `run_hyperpod_three_source_fault_e2e.py`
- `run_hyperpod_xid74_case.py`
- `run_cap005_postgres_suite.py`
- `run_ha008_processor_exit_probe.py`
- `run_regional_boot_guard_cases.sh`

## Audit probes

Files prefixed `audit_` verify one production contract, usually PostgreSQL,
collector, processor, evidence or policy behavior. They are not generic test
runners and must be invoked only by the procedure that names them.

`audit_regional_command_protocol_live.py` is piped into a Ready control-plane
Pod after the production Executor has been scaled to zero. It creates
content-addressed, non-destructive protocol commands and deletes every
temporary command/workflow/incident between cases.

## Q118 helpers

- `q118_fallback_marker.py`
- `q118_requeue_containment.py`
- `audit_q118_gpu_result.py`

Their Kubernetes fixtures live in `manifests/q118-*.yaml`.

## Recovery helper

`restore_validated_quarantine.py` performs the documented, validated
quarantine restore path. It must not be replaced with direct taint deletion.
