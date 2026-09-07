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
- `run_net006_lease_loss_withheld_result.py`
- `run_net008_outbox_dead_letter.py`
- `run_cmd017_barrier_hold.py`
- `run_cmd018_open_sibling_hold.py`
- `run_notify007_delivery_states.py`
- `run_preempt037_dispatcher_liveness.py`
- `run_preempt038_evidence_pins.py`
- `run_collect018_rejected_event.py`
- `run_collect019_nvidia_smi_hang.py`
- `run_collect020_gpu_identity.py`
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
- `run_destr015_parallel_branch_join.py`
- `run_destr016_preempting_reboot.py`
- `run_destr017_out_of_band_reboot_fence.py`
- `run_destr018_lifetime_deadline.py`
- `run_destr019_agent_restart_ledger.py`
- `run_destr020_identity_mismatch_isolation.py`
- `run_destr021_adversarial_node_metadata.py`
- `run_destr022_spare_reservation_reclaim.py`
- `run_ha010_aurora_blackout_liveness.py`
- `run_boot023_release_history.py`
- `run_preempt036_stuck_workflow_reconcile.py`
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
- PREEMPT-036 uses `run_preempt036_stuck_workflow_reconcile.py`. It is fully
  isolated and never contacts a cluster: per reconcile mode it creates its own
  store (a throwaway `postgres:16` container on a random loopback port, removed
  with `docker rm -f` in `finally`; a temporary SqliteStore when docker cannot
  start one, recorded as `store_backend` in the evidence), seeds the stuck
  workflow shape through Store APIs only, and then executes the text the admin
  layer actually ships -- the reconcile module's own source plus its
  stdin/stdout driver, the same string `workflow-reconcile` sends into the CPU
  ingress Pod -- in a local subprocess. The release verdict is measured with the
  shipped `workflow_safety` and `remote_command_stats` probe sources before and
  after each apply. `--run-dir` is required and evidence lands in
  `<run-dir>/cases/GF-REGIONAL-PREEMPT-036/`; unlike HA-008 there is no
  temporary-directory default, because the evidence is the point of the case.
  Its verdicts and seeding live in `preempt036_verdicts.py` and are unit tested
  in `tests/regional/test_preempt036_stuck_workflow_reconcile.py`. The runner
  uses only store URLs it created itself and refuses any other, including an
  ambient `GPU_FAULT_STORE_URL`: a fixture whose job is to manufacture wedged
  workflows must not be able to aim at a real database.
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
- DESTR-015 has a reusable manual live driver for the happy side of the
  two-node job DAG that DESTR-014 fails on purpose. It requires DESTR-012
  PASS, pins one two-node 16-GPU PyTorchJob, writes one XID 46 to each
  node's `/dev/kmsg` concurrently so both faults land in one DAG inside the
  aggregation window, and requires two parallel reset branches, a single
  join that restarts the job once, and both nodes schedulable afterwards.
  Its verdicts live in `destr015_verdicts.py`; a node a failed branch left
  isolated is restored only through the validation-first workflow.
- DESTR-016 has a reusable manual live driver for the second fault that arrives
  while a physical step is already in progress. It requires DESTR-002 PASS,
  pins one idle GPU node, and writes one XID 46 through the shared host probe
  to open a RESET_GPU workflow. A bounded GPU device holder, armed off *this
  drill's* QUIESCE_GPU_SERVICES ledger row by a second probe
  (`probes/destr016_node_probe.py`), keeps VERIFY_NO_GPU_CLIENTS WAITING with
  GPU services already quiesced -- the dirty boundary the case needs. It then
  writes a second XID 46, which must be absorbed by the same workflow without
  adding a step, and one XID 79, which must preempt: the reset workflow goes
  SUPERSEDED, its barrier command is cancelled with
  `status_source=workflow-preempted`, and the successor RESTART_NODE workflow
  carries `quiesce_handoff_from_workflow_id` plus a
  `preemption_quiesce_handoff_after_reboot` RESTORE_GPU_SERVICES step that runs
  *after* the real HyperPod reboot. The runner asserts exactly one
  BatchRebootClusterNodes by the executor role, that the holder vanished with
  the reboot, that the Node Agent re-registered with a new boot id, and that no
  successful RESET_GPU row ever reached the on-node ledger. Its verdicts live in
  `destr016_verdicts.py`. Both probe Pods are recreated after the reboot
  (`restartPolicy: Never`), the holder is disarmed idempotently before anything
  else in cleanup, and a node the case left isolated is restored only through
  the validation-first workflow -- never by deleting a taint.
- DESTR-017 has a reusable manual live driver for the generation fence. It
  requires DESTR-002 PASS, pins one idle GPU node, writes one XID 46 through
  the shared host probe to open a RESET_GPU workflow, holds the workflow in
  WAITING with a bounded GPU device holder armed off this drill's
  `QUIESCE_GPU_SERVICES` ledger row, and then reboots the node out of band
  with a bounded `systemd-run --on-active` timer -- no provider API is
  involved, and the runner asserts the CloudTrail mutation verb set is empty.
  The node returns with a new boot id and the Node Agent re-registers one
  generation higher than the one `QUIESCE_GPU_SERVICES` pinned into the
  maintenance window, so no RESET_GPU is ever posted on the new boot and the
  old generation's command is never completed as SUCCEEDED. Its verdicts live
  in `destr017_verdicts.py`, its node-local work in
  `probes/destr017_node_probe.py` (the reboot must be armed and returned from,
  never run synchronously, because it kills the probe's own exec channel), and
  the retired-generation reconcile view is run in plan mode only and recorded
  `NOT_APPLIED`. The isolation the failed workflow leaves behind is restored
  only through the validation-first workflow, successor incident first.
- DESTR-018 has a reusable manual live driver for the workflow lifetime hard
  deadline. It requires DESTR-010 PASS, opens a temporary env window on the CPU
  control-worker Deployment that compresses
  `GPU_FAULT_NODE_WORKFLOW_MAX_LIFETIME_SECONDS` and
  `GPU_FAULT_WORKFLOW_EXECUTION_TIMEOUT_SECONDS` to 180s, holds one
  `/dev/nvidiaN` open on one idle node so `VERIFY_NO_GPU_CLIENTS` stays WAITING
  without burning its 60 attempts, writes one XID 46 to `/dev/kmsg`, and
  requires the whole workflow to fail on the lifetime, the in-flight command to
  end FAILED with `status_source=workflow-timeout`, exactly one successful
  `RESTORE_GPU_SERVICES`, an `ESCALATE_SUPPORT` handoff, and a later XID 79 to
  be absorbed record-only. It never lowers
  `GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS`: the runner measures the real
  redispatch cadence from the ledger and refuses to execute unless
  `lifetime < attempts x cadence` holds with margin. Its verdicts live in
  `destr018_verdicts.py`, including the three data-plane verdicts that read the
  Node Agent ledger against the cancellation moment. A node the deadline left
  isolated is restored only through the validation-first workflow.
- DESTR-019 has a reusable manual live driver for what a Node Agent restart
  must leave behind. It requires DESTR-010 PASS, refuses to restart while
  any remote command is open, arms a bounded `systemd-run --on-active` start
  of `gpu-fault-node-agent.service` *before* `systemctl restart`, and then
  writes one solo XID 45 so the Fabric Manager restart DESTR-010 proved runs
  again on the restarted Agent. It judges the Agent's `/healthz`
  (`ledger.writable`, heartbeat, counters), the three structured journald
  lines the command leaves, the ledger row's audit columns and
  `PRAGMA user_version`, and drills the pre-audit ledger schema's in-place
  migration on a scratch copy under `/var/lib/gpu-fault-acceptance` with the
  deployed wheel. Its verdicts live in `destr019_verdicts.py`, its node-local
  work in `probes/destr019_node_probe.py` (the only mutating verbs are
  `restart`/`start` on the Agent unit and `stop`/`reset-failed` on the
  fail-safe unit named after the run). It deliberately never restarts the
  Agent with a command in flight: an interrupted action is failed closed as
  `manual_confirmation_required`, and on a RESET_GPU workflow that climbs to
  a provider reboot nothing on the node can stop.
- DESTR-020 has a reusable manual live driver for the isolation identity
  check (ARCH-A5). It requires DESTR-001 PASS and posts one API-replay XID 46
  for an alias node id that neither Kubernetes nor the fleet knows, so the
  RESET_GPU workflow's `MARK_UNSCHEDULABLE` reads a 404 and must fail closed
  (`safety_rejection`, `absent`) instead of recording the node as "already
  isolated"; no real node may change and no later step may be reached. The
  alias has no Node Agent and no provider node, so no reset or reboot can be
  armed from it. Its verdicts live in `destr020_verdicts.py`.
- DESTR-021 has a reusable manual live driver for adversarial node metadata
  (ARCH-A2/A3/A8). It requires DESTR-020 PASS, reuses COLLECT-017 A段's EFA
  unbind injection on one idle node, pre-writes a stale foreign
  `efa-plugin-restart-*` annotation set that `RESTART_EFA_DEVICE_PLUGIN` must
  take over and clear, and runs a bounded local writer
  (`probes/destr021_annotation_writer.py`) that patches a harmless tick
  annotation every 0.5 s so `MARK_UNSCHEDULABLE`/`RESTORE_SCHEDULING` meet
  real 409s; both steps must report `node_baselines[node].before/after`, the
  restore must never fail, and the node must end clean. Its verdicts live in
  `destr021_verdicts.py`.
- DESTR-022 has a reusable manual live driver for the warm-spare reservation
  sweep (ARCH-A4b/A4c). It requires DESTR-008 PASS, writes a synthetic
  incident's stale reservation (`spare-reserved-at` two days old) on the
  declared spare without uncordoning it, and waits for the deployed cluster
  executor's `SpareReservationSweep` to release it, judged by the node
  annotations, the executor log line and the claim-state breadcrumb counter
  (`probes/destr022_executor_probe.py`). Its verdicts live in
  `destr022_verdicts.py`; cleanup restores the recorded baseline.
- HA-010 has a reusable manual live driver for control-plane liveness across
  an Aurora writer failover (ARCH-H1/H3). It requires HA-001 PASS, refuses to
  run with any remote command open, calls `failover-db-cluster` exactly as
  HA-003 does, deletes one `gpu-fault-api-ha` replica right after the request,
  and samples `/livez` and `/healthz` inside every CPU Pod
  (`probes/ha010_probe.py`): liveness must stay 200, readiness may only 503 and
  must return within the stale window, no pre-existing container may restart,
  the replacement must not CrashLoop, and `secret_drift` must stay false. Its
  verdicts live in `ha010_verdicts.py`.
- BOOT-023 has a reusable manual live driver for the release audit surfaces
  (ARCH-H2/H3/H5). It requires BOOT-020 PASS and the `noop` config that
  `boot020_release_candidates.py configs` materialises, refuses anything that
  does not classify as NOOP, runs `RegionalRelease.noop` once, and reads back
  the append-only `gpu-fault-release-history` ConfigMap and its mirror, the
  previous-snapshot retention, the per-Pod Secret-vs-durable registry digests
  (`probes/boot023_registry_probe.py`) and the parser's refusal of
  `database.rollback_compatible: true`. Its verdicts live in
  `boot023_verdicts.py`.
- COLLECT-018/019/020 and NET-008 share `collector_window_fixture.py` and
  `probes/collector_window_probe.py`: one host probe that can open a bounded
  change on one allow-listed collector unit of one idle node -- a systemd
  drop-in under `/run/gpu-fault-acceptance/<digest>` that appends an env file
  (only a probe-generated wrong cluster token), unsets
  `GPU_FAULT_EXPECTED_GPU_COUNT`, or puts a `nvidia-smi` shadow first on PATH
  (`hang:<s>` sleeps past the collector timeout; `drop-uuid:<uuid>:1` hides
  one GPU line from exactly one inventory query, below the host collector's
  mismatch threshold) -- with a `systemd-run --on-active` deadman that closes
  it. It also writes labelled user-space `/dev/kmsg` lines (monitor-only XID
  63, or an `Xid` line with no code), posts one deliberately incompatible
  event through the node's own sink, drives the shipped
  `gpu-fault-collector outbox` CLI, seeds/purges a retired-channel outbox
  record only while the owning unit is stopped, and reuses NET-001's tagged
  iptables reject. The Node Agent is not in its unit allow-list. Verdicts are
  `collect018_verdicts.py`, `collect019_verdicts.py`, `collect020_verdicts.py`
  and `net008_verdicts.py`.
- CMD-018 (`run_cmd018_open_sibling_hold.py`, `cmd018_verdicts.py`) drives the
  deployed `RegionalRemoteWorkflowAdapter` inside a CPU API Pod for a
  `perf-cap-000` workflow: dispatch, rewrite the step's parameters through a
  guarded `save_workflow`, dispatch again (held `OPEN_SIBLING_COMMAND`), let
  `probes/cmd018_ledger_executor.py` finish the open sibling once, then show
  the hold lifts and cancel what it mints. It reuses the registry, Pod and
  postflight parts of `seeded_command_fixture.py`.
- NOTIFY-007 (`run_notify007_delivery_states.py`,
  `probes/notify007_delivery_drill.py`, `notify007_verdicts.py`) runs the
  shipped `dispatch_outbox` in a control-worker Pod against an isolated
  in-memory store with the Pod's SES notifier wrapped to fail once, then
  always: RETRY without a result row, then SENT; DEAD with a terminal FAILED.
- PREEMPT-037 (`run_preempt037_dispatcher_liveness.py`,
  `preempt037_verdicts.py`) sets `GPU_FAULT_ENABLE_WORKFLOW_DISPATCHER=false`
  on the control-worker Deployment (baseline recorded first, restored
  exactly) and evaluates `GpuFaultWorkflowDispatcherStalled` on the replicas'
  `/metrics` with the rule file's threshold and `for`. It refuses while any
  workflow is in flight.
- PREEMPT-038 (`run_preempt038_evidence_pins.py`, `preempt038_verdicts.py`)
  ships `audit_raw_evidence_periodic_cleanup.py` into a control-worker Pod:
  two expired `audit-*` evidence rows and one ESCALATED `audit-cluster`
  incident, the deployed sweep deletes the unrelated row, keeps the pinned
  one until the incident is RECOVERED, and logs the deleted keys.
- NET-006 and CMD-017 share `seeded_command_fixture.py`, the NET-002 shape
  made reusable: a synthetic `perf-cap-000` registry entry that expires in 30
  minutes, one remote command seeded straight into the store for node ids no
  cluster carries, one probe executor Pod on the GPU plane running the
  deployed `ClusterActionExecutor`, and a purge that reads back zero. NET-006
  (`probes/net006_executor.py`) holds its action past the lease while a
  loopback proxy refuses the control plane, so the executor must withhold the
  result (`results_withheld_total`, `lease_lost_total`, no 409) and reclaim it
  from the idempotency ledger. CMD-017 (`probes/cmd017_barrier_executor.py`)
  seeds a two-node `RESET_ALL_GPUS_NVSWITCHES` for a stand-in adapter with
  `barriers=None`; the executor must hold it at the claim boundary
  (`status_source=executor-barrier-unavailable`,
  `barrier_unavailable_holds_total`) and never reach the adapter. The
  natural two-node SXID injection is deliberately not used: past
  `VERIFY_NO_GPU_CLIENTS` a regressed claim boundary would arm two real
  full-fabric resets with no on-node stop.
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

`control_plane_env_window.py` opens and closes the one env window DESTR-018
needs. It is read-only unless `--open` or `--close` is given, each with its own
confirmation string (`OPEN_CONTROL_PLANE_ENV_WINDOW` /
`CLOSE_CONTROL_PLANE_ENV_WINDOW`). It touches exactly one container
(`control-worker` of the CPU-plane `gpu-fault-control-worker` Deployment),
accepts only the two allowlisted variable names as positive integers of at least
60 seconds, records the pre-window env plus the Deployment's template digest in a
`--baseline` file, and restores exactly that on close — a variable that was unset
before the window is unset again, not set to the shipped default. It also
surveys the values every ready replica actually runs, so a half-finished rollout
is reported as disagreement rather than averaged.

`probes/destr018_node_probe.py` holds one GPU device open. `arm-holder` starts a
bounded systemd transient unit that opens `/dev/nvidiaN` and sleeps, so the
Agent's device-client check keeps seeing the same pid on the same device;
`watch-ledger`, `holder-status`, `disarm-holder` and `snapshot` report and
release it. Devices are allowlisted to `/dev/nvidia<N>` (never `nvidiactl`,
`nvidia-uvm` or a block device), the unit name is a digest of the run id so the
probe can only stop units it created, and the hold is capped so a lost runner
cannot leave a device held. XID injection is not its job: DESTR-018 runs a
second `host_probe_fixture.py` instance with
`probes/destructive_node_probe.py` for that.

## Recovery and preparation helpers

`restore_validated_quarantine.py` creates the validation-first workflow used to
restore a quarantined node. Direct taint or ownership-annotation deletion is
not an equivalent cleanup.

`declare_warm_spare.py` declares or releases the one warm spare node
`DESTR-003`/`DESTR-008` require. It is read-only unless `--declare` or
`--release` is given, each with its own confirmation string, and it records the
node's pre-declaration labels and cordon state in a `--baseline` file so the
release restores exactly that — a node that was already cordoned stays cordoned.

It mutates only the spare label and the cordon. It does not write
`gpu-fault.io/spare-pool-state`, which the control plane owns and DESTR-003
accepts absent. Nor is the declaration folded into DESTR-003 itself:
`preflight_errors` asserts the declared spare set is exactly the requested node
in order to catch a site with a stray label elsewhere, and a case that created
its own spare would satisfy that assertion by construction even if
`HyperPodSpareCoordinator.allocate(local_only=True)` had drifted away from it.

`synthetic_replacement_route.py` opens and closes DESTR-003's other precondition:
`POST /v1/admin/test/node-replacement` answers 404 unless
`GPU_FAULT_ENABLE_SYNTHETIC_REPLACEMENT_TESTS=true` on every API replica, and the
shipped regional manifest never sets it. `--open` records the Deployment's env
first, then sets the variable and refuses unless the rollout reached every ready
replica; `--close` restores what the baseline recorded, so a variable that was
absent goes back to absent rather than to `false`. The variable name is compiled
in — this is a test-window switch for one route, not a way to set arbitrary env
on a live control plane.

The route fabricates a `REPLACE_NODE` finding for any node in the cluster. It is
protected by the workflow execution token, which is why a bounded window is
acceptable at all, but leaving it open is a standing way to have a healthy
machine quarantined and failed over — hence the recorded baseline rather than a
hand `kubectl set env` and a note to remember.

## Manifests

All regional E2E Kubernetes inputs live under `manifests/`, split into
`fault-injection/` and `training/` where appropriate.
`manifests/regional-isolation-gpu-job.yaml` is a source workload. Render it
with `gpu-training-submit` into a test namespace; do not apply it directly.
