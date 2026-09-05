# Regional acceptance fixtures

This directory owns the executable fixtures referenced by
`docs/区域模式端到端验收测试用例.md`.

All regional-only executable fixtures belong here. Do not add new regional
probes under `scripts/`, `tests/manifests/`, or the `scripts/e2e/` root.

## Site values: `--site-profile`

These runners must not contain site topology — account IDs, kubeconfig paths,
EKS contexts, node names — and
`tests/regional/test_collector_and_multicluster_fixtures.py::test_collector_promoted_scripts_contain_no_site_specific_topology`
enforces it. Supply those values with one file the operator keeps outside the
repository, normally in the private state directory:

```yaml
# /secure/.../acceptance-site-profile.yaml   (0600)
arguments:            # keys are the runner's long flags, minus the leading --
  cluster-id: ...
  gpu-context: ...
  node: ...           # a list here becomes repeated --node flags
environment:          # only keys that are not already exported
  GPU_FAULT_ACCEPTANCE_WINDOW_END: ...
```

```bash
python scripts/e2e/regional/run_collector_acceptance.py \
  --site-profile /secure/.../acceptance-site-profile.yaml \
  --run-dir ... --case GF-REGIONAL-COLLECT-002 --plan
```

Precedence is the one the runners already use for environment fallbacks: an
explicit flag on the command line wins, then the profile, then the ambient
environment. A flag typed on the line suppresses the profile's entry for it
entirely — including `append` flags such as `--node`, so an explicit `--node`
replaces the profile's list rather than adding to it.

`install_site_profile()` is the first statement of every runner's `main`,
because flags such as `--maintenance-window-end` read their default from the
environment while the parser is being built.
`build_plan` records the profile's path and sha256 in `plan.json`, and
`authorize_execution` refuses an `--execute` whose profile differs from the one
`--plan` was built against — a plan names the target it was approved for.

The profile is refused if it is group- or world-writable, or if it has a section
other than `arguments`/`environment`: it decides which node a destructive case
reboots, and a typo in a section name would silently drop every value in it.

## Live and staging drivers

- `run_regional_boot_guard_cases.sh`
- `run_cap005_postgres_suite.py`
- `run_ha008_processor_exit_acceptance.py`
- `run_ha007_control_worker_shutdown.py`
- `run_net002_command_recovery.py`
- `run_net003_result_retry.py`
- `run_ha001_control_plane_failover.py`
- `run_ha002_pdb_topology.py`
- `run_ha003_aurora_failover_reset.py`
- `run_ha004_waiting_reclaim_reset.py`
- `run_ha005_rollout_continuity.py`
- `run_ha006_executor_takeover.py`
- `run_ha009_aurora_credential_rotation.py`
- `run_boot019_admin_lifecycle.py`
- `run_boot020_release_rolling.py`
- `run_boot_acceptance.py`
- `run_identity_acceptance.py`
- `run_workload_acceptance.py`
- `run_preemption_contracts.py`
- `run_preempt012_acceptance.py`
- `run_notification_acceptance.py`
- `run_capacity_acceptance.py`
- `run_net001_collector_replay.py`
- `run_blast_acceptance.py`
- `run_destr001_gpu_reset.py`
- `run_destr002_hyperpod_reboot.py`
- `run_destr003_warm_spare_failover.py`
- `run_destr008_warm_spare_shortage.py`
- `run_destr009_workload_restart.py`
- `run_destr010_fabric_manager_restart.py`
- `run_destr012_managed_recovery_guard.py`
- `run_collector_acceptance.py`
- `run_collector_destructive.py`
- `run_collect016_training_recovery.py`
- `run_collect017_efa_plugin.py`
- `run_iso006_cluster_offline.py`
- `run_e2e002_multicluster_fault.py`

The BOOT-019/020 runners are plan-only by default. Their live paths require
both `--execute` and the case-specific confirmation string, keep resumable
evidence under a caller-supplied `--run-dir`, and remain `manual` in the case
catalog because they attach/remove clusters or roll real releases.
BOOT-020 consumes distinct CPU-only, Executor-only, Agent-only and FULL
configs, and records component rollback scope plus `T_safe`, `T_full` and
interrupted-resume RTO evidence.

The grouped BOOT-011..018, AUTH/ISO, WORKLOAD/E2E, PREEMPT-012, NOTIFY,
CAP-001..004 and NET-001 drivers use the same plan/execute guard. They derive
their formal predecessor from `testcases/regional-execution-order.yaml` and
default to the preceding evidence file under the same run directory.
`run_preemption_contracts.py` is the deterministic non-live command entry for
PREEMPT-001..009. `run_blast_acceptance.py` executes one explicit read-only
BLAST case.

`run_ha008_processor_exit_probe.py` is the fatal child process used internally
by the HA-008 acceptance wrapper. It is not the public case entry point.

## Audit probes

Files prefixed with `audit_` are focused probes named by an acceptance
procedure. Notable entry points:

- `audit_regional_command_protocol_live.py`
- `audit_executor_local_guards.py`
- `audit_executor_readiness.py`
- `audit_net004_dependency_boundary.py`
- `audit_collector_outbox.py`
- `audit_destr011_provider_replace.py`
- `audit_destr013_replacement_invariant.py`
- `audit_warm_spare_guardrails.py`

## Fixture lifecycle

Regional case code has four distinct ownership classes:

| Class | Location | Catalog automation | Requirements |
|---|---|---|---|
| pytest regression | `tests/` | `pytest` or `related_pytest` | deterministic, isolated and safe for CI |
| reusable audit | `scripts/e2e/regional/audit_*.py` | `command` | read-only or isolated, environment-driven and credential-safe |
| reusable live driver | `scripts/e2e/regional/run_*.py` | usually `manual` | plan-only by default, explicit `--execute`, exact confirmation, rollback and caller-owned evidence directory |
| task-local investigation | ignored `.codex/` or private evidence storage | never referenced by the catalog | site-specific, experimental or one-off; not a release asset |

A task-local runner is promoted only after all of these are true:

1. Region, contexts, cluster IDs, nodes and paths come from explicit arguments
   or documented environment variables; no customer topology is embedded.
2. The default invocation is read-only or plan-only. Every mutation requires
   an explicit execute flag and a case-specific confirmation value.
3. Cleanup is retryable and independently verifies zero database, registry and
   Kubernetes residuals. Live network or scheduling changes have an automatic
   rollback path.
4. Output excludes Secret values and can pass `scripts/check-artifacts.py`.
5. The regional specification, case catalog and fixture contract tests point
   to the same command.

`related_pytest` remains supporting logic coverage. It does not convert a
manual live case into executed acceptance evidence.

Current classification:

- `NET-004` and `NET-005` have reusable read-only/isolated audit commands.
- BOOT-011..018 share `run_boot_acceptance.py`. Greenfield actions remain
  plan-only; runtime CA, SES, dispatcher and owner checks cover every selected
  live replica. BOOT-018 performs reproducible builds and removes the isolated
  site after live identity verification unless explicitly retained.
- AUTH-007/010/012..015 and ISO-003..005 share
  `run_identity_acceptance.py`. Registry, token and key documents remain only
  in memory and are restored in `finally`; Secret values never enter evidence.
- WORKLOAD-001/002, ISO-001 and E2E-001 share
  `run_workload_acceptance.py`. It owns prewarm, managed submission,
  observation checks and workload cleanup. E2E-001 emits the execution card
  and CPU baseline consumed by BLAST-001.
- PREEMPT-001..009 use `run_preemption_contracts.py`; PREEMPT-012 combines a
  real fail-safe quiesce/restore cycle with the deployed executor state
  machine and live PostgreSQL.
- NOTIFY-001..005 share `run_notification_acceptance.py`. Reset/restart email
  cases use labeled drills rather than repeating physical actions.
- CAP-001..004 share `run_capacity_acceptance.py`, promoted from the isolated
  disposable-control-plane harness. NET-001 uses
  `run_net001_collector_replay.py` and arms a host rollback timer before any
  TCP/443 rule is added.
- `HA-007` and `HA-008` have reusable isolated subprocess drivers.
- `NET-002/003` and `HA-001/002/005/006` have reusable manual live drivers.
  Their generalized source has not yet been rerun live, so existing private
  evidence is not yet declared equivalent to these final entry points.
- HA-009 credential rotation has a reusable manual live driver with the same
  plan/execute guard and roll-forward recovery contract.
- DESTR-011 has a reusable read-only/isolated guard audit.
- DESTR-010 has a reusable manual live driver. It is plan-only by default,
  pins one idle GPU node, writes exactly one XID 45 through a temporary
  host probe, verifies the two-step workflow and Node Agent ledger replay,
  and restores Fabric Manager plus deletes probe resources in `finally`.
- DESTR-001/002/009/012 have reusable manual live drivers. Each is plan-only
  by default and validates a predecessor evidence file before mutation:
  `010 -> 001 -> 002 -> 009 -> 012`. DESTR-001/002 use the shared node-pinned
  host probe; DESTR-009/012 share image prewarm and managed workload fixtures.
  DESTR-012 executes groups B, A and D in order. Optional group C remains
  `NOT_RUN` unless a separately reviewed isolated control plane and database
  are provisioned; the runner never edits the production Runtime Profile.
- DESTR-003/008 have reusable manual warm-spare live drivers. DESTR-003
  requires DESTR-012 PASS and one already-declared, cordoned, topology-matched
  healthy spare. DESTR-008 requires DESTR-003 PASS and runs six independently
  cleaned shortage scenarios; a subset is useful for debugging but cannot
  produce a PASS verdict. Both use the execution-token protected synthetic
  replacement signal to drive the real workflow and real Kubernetes
  mutations, and explicitly do not claim a real hardware fault.
- HA-003/004 have reusable destructive live drivers. HA-003 triggers Aurora
  failover only after the real RESET_GPU command is observed LEASED. HA-004
  temporarily rolls the two-replica executor Deployment to 10s/2s
  lease/poll settings, arms a detached rollback watchdog, force deletes the
  first command owner and requires the same command ID to be reclaimed while
  the Node Agent ledger records one physical reset.
- DESTR-013 has a reusable read-only final invariant audit. It requires an
  explicit start/end window plus HA-004 PASS evidence and enumerates exact
  CloudTrail mutation verbs, running executor environments, NodeRecovery,
  IAM simulation and deploy YAML invariants.
- COLLECT manual cases use two shared entry points:
  `run_collector_acceptance.py` for 001/002/003/005/009/010/011/012 and
  `run_collector_destructive.py` for 004/008/013/014/015. COLLECT-016 and
  COLLECT-017 have dedicated managed-training and EFA/device-plugin drivers.
  Every driver is plan-only by default and requires the previous formal case
  evidence before mutation.
- ISO-006 and E2E-002 have dedicated two-physical-cluster drivers.
  `multi_cluster_fixture.py` rejects duplicate cluster identities/contexts;
  ISO-006 uses per-node 443 blocks with host rollback timers, and E2E-002
  submits the same job/attempt identity independently to both clusters.
- DESTR-005/006/007 use `audit_warm_spare_guardrails.py` as their supported
  command entry. It executes the guard against the deployed component wheel,
  blocks a pre-existing quarantine/ownership baseline, and compares every GPU
  node's scheduling state in `finally`. The obsolete workflow that first
  isolated a real node and then deliberately failed the guard must not be run.
- Provider mutations and destructive-action orchestrators remain task-local/
  manual until they satisfy the same promotion gates.

Files under `probes/` are implementation support for a live driver. They are
not standalone catalog commands.

`host_probe_fixture.py` is the shared controller for node-level live cases.
It creates a node-pinned, time-bounded privileged Pod and ConfigMap from an
allowlisted probe script, then verifies both resources are absent after cleanup.

`regional_live_fixture.py` owns credential-safe CPU/GPU kubectl, store,
CloudTrail, node and predecessor-evidence reads. `managed_workload_fixture.py`
owns digest-pinned image prewarm, `gpu-training-submit`, Pod/heartbeat
observation and retryable workload cleanup for the workload restart cases.

`warm_spare_fixture.py` owns spare pool/node/provider/Agent/notification
snapshots, synthetic replacement submission, incident-scoped spare release,
Agent reactivation and validation-first fault-node restoration.
`probes/warm_spare_node_probe.py` can stop only kubelet or the Node Agent and
always arms a bounded host-side automatic restore timer before doing so.

`collector_acceptance_fixture.py` and `probes/collector_node_probe.py` own
marker/evidence/workflow lookup plus allowlisted XID, SXID, collector cursor,
EFA and expected-count operations. `multi_cluster_fixture.py` owns the
redacted two-cluster registration and context binding used by ISO/E2E cases.

## Recovery helper

`restore_validated_quarantine.py` creates the validation-first workflow used to
restore a quarantined node. Direct taint or ownership-annotation deletion is
not an equivalent cleanup.

## Manifests

All regional E2E Kubernetes inputs live under `manifests/`, split into
`fault-injection/` and `training/` where appropriate.
`manifests/regional-isolation-gpu-job.yaml` is a source workload. Render it
with `gpu-training-submit` into a test namespace; do not apply it directly.
