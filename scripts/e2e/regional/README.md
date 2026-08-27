# Regional acceptance fixtures

This directory owns the executable fixtures referenced by
`docs/区域模式端到端验收测试用例.md`.

All regional-only executable fixtures belong here. Do not add new regional
probes under `scripts/`, `tests/manifests/`, or the `scripts/e2e/` root.

## Live and staging drivers

- `run_regional_boot_guard_cases.sh`
- `run_cap005_postgres_suite.py`
- `run_ha008_processor_exit_probe.py`
- `run_boot019_admin_lifecycle.py`
- `run_boot020_release_rolling.py`

The BOOT-019/020 runners are plan-only by default. Their live paths require
both `--execute` and the case-specific confirmation string, keep resumable
evidence under a caller-supplied `--run-dir`, and remain `manual` in the case
catalog because they attach/remove clusters or roll real releases.

## Audit probes

Files prefixed with `audit_` are focused probes named by an acceptance
procedure. Notable entry points:

- `audit_regional_command_protocol_live.py`
- `audit_executor_local_guards.py`
- `audit_executor_readiness.py`
- `audit_collector_outbox.py`

## Recovery helper

`restore_validated_quarantine.py` creates the validation-first workflow used to
restore a quarantined node. Direct taint or ownership-annotation deletion is
not an equivalent cleanup.

## Manifests

All regional E2E Kubernetes inputs live under `manifests/`, split into
`fault-injection/` and `training/` where appropriate.
`manifests/regional-isolation-gpu-job.yaml` is a source workload. Render it
with `gpu-training-submit` into a test namespace; do not apply it directly.
