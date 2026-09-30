# Change Impact and Test Selection

English edition of `docs/变更影响与测试选择.md`; the Chinese file remains the source of record until both are maintained together.

This document defines the reverse impact index "code path/module -> static checks + pytest + regional acceptance cases".
It serves daily development and release candidates, and does not change the source of truth of formal regional acceptance:

- Case fields: `testcases/fault-scenarios.yaml`
- Formal order: `testcases/regional-execution-order.yaml`
- Reverse impact rules: `testcases/change-impact.yaml`

The selector only generates a regional case plan; it never automatically executes live, destructive, reboot, reset,
warm-spare or workload restart.

## Execution tiers

| Scenario | Requirement |
|---|---|
| Daily development | Related static checks + related pytest |
| PR / release candidate | Daily development gates + plan of affected non-destructive regional cases |
| Administrator selective execution | Run only the explicitly selected cases; the report is marked selective and does not stand for a complete phase pass |
| Formal first go-live / major architecture change | Full acceptance per `regional-execution-order.yaml` |
| destructive / live case | Executed only when the corresponding implementation is affected and a maintenance window approval has been obtained |

Common entry points:

```bash
make test-impact BASE=origin/main
make regional-impact-plan BASE=origin/main

# plan only, do not run pytest
python3 scripts/select-affected-tests.py --base origin/main
```

The selector merges the following differences so that uncommitted changes are not missed:

1. committed changes in `BASE...HEAD`;
2. working tree changes;
3. staged changes;
4. untracked files that are not ignored by Git.

The unified staging build uses `--write-plan` to save one selection result to
`dist/staging-impact-plan.json`, then executes it with `--read-plan`. The plan file is bound to `BASE`, the changed
files and the content SHA; on read, the changed files are fetched again and required to match exactly, so reusing a plan does not accept source
drift. The same file also provides the regional safe/approval case lists, so the rule matching is no longer
re-run just to output the regional plan.

The cases output by the selector can be executed one by one by the administrator under selective scope. Selective scope does not automatically make up
the other cases in the formal order, nor can it serve as the predecessor of a later formal case; to declare that complete regional acceptance
has passed, the formal campaign must still be executed per `regional-execution-order.yaml`.

## Meaning of the output

`Changed domains` are the impact domains that were hit. `Run static checks` and `Run pytest` can be executed directly.
`Regional safe cases` only represents the non-destructive impact plan; it does not mean anything has been executed.
The cases under `Require staging/live approval` must go through the approval of the corresponding environment and maintenance window.
`Not selected regional families` states explicitly which families have no implementation path hit.

`make test-impact` runs the static checks and pytest, but does not run regional cases. When a real
PostgreSQL stress test is needed and the environment does not set `GPU_FAULT_TEST_POSTGRES_URL`, the command fails closed.
The unified staging entry point first reads the `postgres` flag in the plan and creates the isolated
PostgreSQL 16 only when that flag is true.

## Main CI content-identity reuse

The `unit` gate of main CI does not use the Git commit directly as the sole reuse key; instead it generates four logical content domains:

1. `runtime`: application Runtime source, ordinary Runtime tests, shared fixtures and dependencies;
2. `deployment`: Runtime shared source, deployment/release/deploy-host-only source and their tests;
3. `fault_runner`: Runtime shared source, catalog/scheduler/runner and runner tests;
4. `postgres`: Runtime shared source, the four PostgreSQL contract tests, the actual PostgreSQL image;
5. each identity is additionally bound to the Python/Runner, the installed distributions, the coverage protocol and the worker configuration.

The runtime domain is further split into `runtime_0..2` by a hash of the stable pytest nodeid, so each run actually produces six
physical shard artifacts. Each nodeid of a parametrized test enters only one runtime sub-shard, and the union of the three keeps
the full Runtime test set.

A historical shard artifact must come from a completed main push whose corresponding shard job succeeded, and must pass the Cosign signature verification of the fixed workflow
identity, the producer run, the evidence SHA and the current content-identity check; failures of other jobs in the same run
do not invalidate an independent shard that was already successfully signed. On a hit, the current run uses the historical
coverage/pytest/duration evidence to generate a new shard gate carrying `reused_from` and re-signs it; shards that miss
execute independently. Only after the six physical shards complete are `coverage combine`, the 78% floor,
the per-module floor of deployment-only modules and the fault report run together.

When the GitHub artifact query or download is temporarily unavailable, the current shard safely falls back to fresh execution; an already downloaded gate
whose signature, digest, producer or content identity is inconsistent continues to fail closed, and the network fallback must not mask invalid
evidence.

Therefore:

- docs/CI tooling changes are executed by the current static job, and all six shards can be reused;
- deploy-host-only administrator source or corresponding test changes only rerun the deployment shard;
- fault catalog/runner changes only rerun the fault_runner shard;
- PostgreSQL contract tests or related Runtime source changes make the postgres shard rerun;
- Runtime shared source changes conservatively invalidate the multiple shards that depend on it, but these fresh shards execute in parallel.
- CI gate, artifact download, coverage/unit signing and candidate recovery belong to the explicit `ci-trust` domain; changes to these files
  continue to escalate to the full gates and are no longer caught by the unclassified runtime fallback.

This mechanism cannot reuse results from failed, cancelled, PR or other-branch runs, nor can it skip the current commit's static,
artifact, unified coverage floor or PostgreSQL stress.

## Default module mapping

| Change scope | Main regional cases |
|---|---|
| Release, renderer, Manifest, Pin, rollback | BOOT-018..022, HA-005, HA-007..009 |
| Pre-state snapshot (`regional_release_admin_capture.py`, `test_release_rollback_state.py`): the Aurora window in the snapshot comes only from the release state record value or the candidate expected value | BOOT-018..022; rollback does not move the Aurora window (see Administrator Capacity Configuration §7) |
| deploy-host CLI, source identity, candidate recovery | deployment tests; when the application identity is unchanged the deploy is a host-only NOOP |
| Environment isolation, parallel preparation, tool cache and failure wrap-up of the online deploy-host install entry point | `tests/test_unified_deploy_host_setup.py` uses an isolated installer and download fixtures to verify the real Make entry point's parallel overlap, pre-start path validation, failure waiting and third-party-package-free tool-only checks; no live deploy is executed |
| First deploy, join/remove, uninstall, registry | BOOT-016, BOOT-019, BOOT-024..029, BOOT-032, ISO-001..006, E2E-002 |
| BOOT-018's runtime identity comparison takes the manifest of this reproducible build as the expected value (no longer reads dist/current-release.json of the checkout directory) and reads the check items from `health.checks` of `status --full` (`boot_acceptance_lifecycle.py`, `test_boot_acceptance_lifecycle.py`) | BOOT-018 |
| BOOT-015's remote command metric probe now reads control-worker (8081; after the role split, ingress no longer serves these gauges), and the two samples are polled according to the metric scan cache TTL (60 s) (`boot_acceptance_runtime.py`, `test_boot_acceptance_runtime.py`) | BOOT-015 |
| BOOT-012's empty-CA negative probe accepts both the handshake-time CERTIFICATE_VERIFY_FAILED and the OpenSSL 3.5 load-time NO_CERTIFICATE_OR_CRL_FOUND (`boot_acceptance_runtime.py`, `test_boot_acceptance_runtime.py`) | BOOT-012 |
| The first deploy after an uninstall COMPLETED retires the site record to `retired-<time>/` (`deploy_command.retire_site_after_uninstall`, `test_admin_deploy_refuses_during_uninstall.py`) | BOOT-024..029 |
| The failure-domain map refresh at join/remove wrap-up waits for the control-worker rollout to converge (`failure_domain_map.py`); the uninstall preflight waits for the control plane to converge and selects only Ready, not-being-deleted database Pods (`prepare-clean-redeploy.sh`, `test_clean_redeploy_database_pod.py`) | BOOT-026..029 |
| Site-wide batch DRAINING keeps unselected durable states, CPU is kept until strictly drained, consumers stop before ingress; the journal binds the exact phase_order and does not import batch SQL leftovers as failures (`drain_registry_clusters`, `cleanup_state.py`, `test_cleanup_registry_drain.py`). Kubernetes cleanup order of uninstall: first publish `DRAINING` in the registry (`CLUSTERS_DRAINING`; the API refuses new events and new claims, in-flight leases may still complete), drain waits for the queue to empty while ingress and consumers are still running, then stops consumers, ingress and the GPU Executor in turn, and node components are uninstalled in a separate `NODE_RUNTIMES_STOPPED` phase; cleanup state `schema_version=2`, old records are read-only and not resumed (`prepare-clean-redeploy.sh`, `test_clean_redeploy_script.py`, `test_cleanup_state.py`) | BOOT-027, BOOT-029 S4 |
| A lifecycle-driven first-deploy resume must have the original inputs and the exit receipt; the Aurora final snapshot policy is bound to the sequence and the uninstall journal, and an explicit skip does not stand for snapshot-retention acceptance (`admin-lifecycle-sequence.sh`, `test_admin_lifecycle_sequence_script.py`); S5 strictly follows the README three-step deploy: command text parsed from the README by heading, a pristine source copy, an allowlisted environment, a `readme` receipt (`boot029_readme.py`, `test_boot029_readme.py`) | BOOT-029; the retain branch references the BOOT-027 snapshot check |
| Manifest-owning source root and normalized artifact paths: the artifact relative paths of the release manifest are resolved against the repository root containing the manifest (`gpu_fault_release.containing_repository_root`, `regional_release_config.load_release_artifacts`, `test_release_artifact_roots.py`); signature verification, digest and independent OCI checks are still kept; a failure to load the candidate release during join validation is classified as BootstrapError, and the failure reason is written to `failure` in `join-cluster/<id>/state.json` (`cluster_join.py`, `cluster_join_state.py`) | BOOT-019, BOOT-025, BOOT-026, BOOT-029 S3 |
| Scale-out/empty-cluster upgrade is allowed only after all first-deploy targets are managed or released by a full-identity removal proof: the first-deploy target checkpoint constrains the resume target only when the site document does not exist, and reaches COMPLETE when the site already manages the committed set or the committed cluster has been removed by `remove-cluster`, after which `site.yaml` is authoritative; ordinary join failure reasons are persisted redacted (`bootstrap_site.bind_initial_deploy_target`, `cluster_join_state.py`, `test_admin_bootstrap_site.py`, `test_main_sync_lifecycle_bootstrap.py`, `test_main_sync_lifecycle_join.py`) | BOOT-024..029 (`deploy --state-dir` after remove, `deploy --gpu-cluster-arn NEW` immediately after the first deploy) |
| Registry expired members, members new after publication and persistent heartbeat readiness (`registry_revision_missing_member_ids`, `test_regional_registry_convergence.py`, `test_regional_registry_convergence_api.py`) | BOOT-019, BOOT-024..029, HA-001..002 |
| Runtime Profile, capability owner | BOOT-015, BOOT-022; action implementations are appended by other domains |
| Authentication, token, TLS, route authorization | AUTH-001..015, BLAST-002..004 |
| Remote command, lease, Executor | CMD-001..016, HA-004, HA-006, NET-002..003, DESTR-018 |
| Workflow, DAG, preemption, resource claim | PREEMPT-001..036, DESTR-015..018 |
| Store, schema, Processor, concurrency | BOOT-030..031, CAP-001..005, HA-003, HA-008..009 |
| Completion Watcher, training restart | WORKLOAD-001..002, CMD-015, DESTR-009/012/015, E2E-001 |
| Collector, XID/SXID, GPU/Host policy | COLLECT-001..017 |
| Node Agent, reset, quiesce, reset barrier | AUTH-015, CMD-011, DESTR-001/010/014..018, COLLECT-013/014 |
| HyperPod reboot / warm spare / reboot confirmation | DESTR-002/003/005..008/011/013/014/016, ISO-004 |
| Branch in-place escalation ladder (`execution/branch_escalation.py`) | DESTR-014 (not counted as a separate domain; overlaps with the Workflow domain) |
| Pre-submission re-check of node actions under a STOP receipt and boot-change proof (`adapters/kubernetes/stop_ownership.py`, `adapters/kubernetes/stop_boot_transition.py`: confirmed reboot chain, plus the sibling node reboot under the same receipt that is not this target, already submitted by the product and still in flight) | DESTR-014 (two nodes share the receipt; the fault node must still be able to reboot while the sibling reboot is in flight), DESTR-015..017, HA-004; run only the corresponding pytest: `tests/regional/test_stop_ownership_reboot_transition.py`, `tests/regional/test_late_ownership_*.py`, `tests/regional/test_reboot_submission_replay_authority.py`, `tests/regional/test_regional_executor_restart_authorization.py` |
| Operations reconcile (`workflow_reconcile.py`, `workflow_resolution.py`, `retired_generation.py`, `admin/workflow_reconcile.py`) | PREEMPT-036 (isolated command case, self-built storage); run only the corresponding pytest (previously fell into the full fallback) |
| Manual confirmation of node actions with unknown results, per-node restore, verified-restore close-out of non-plan records (`execution/node_action_confirmation.py`, `execution/node_action_uncertainty.py`, `admin/remediation_plans.py`, `admin/remediation_script.py`, `admin/submit_remediation.py`, the third shape of `compile_blocked.py`, the `apply-verified-restore` bridge of `admin/workflow_reconcile.py`) | DESTR-014, HA-004 (release path of a stopped record); run only the corresponding pytest: `tests/execution/test_node_action_confirmation.py`, `tests/admin/test_admin_confirm_node_action.py`, `tests/admin/test_admin_submit_remediation.py`, `tests/admin/test_admin_workflow_reconcile.py`, `tests/execution/test_compile_blocked_sweep.py`, `tests/test_workflow_reconcile_non_plan_successor.py`, `tests/orchestration/test_confirmed_node_action_release_path.py` |
| Notification, outbox, mail templates | NOTIFY-001..006, BOOT-013/014 |
| Network, retry, offline cache | NET-001..005, HA-005, AUTH-013/014 |
| Metrics, capacity, connection pool, alert rules | CAP-001..005, directly related HA cases |
| Release engine kubeconfig token cache (`regional_kubeconfig_cache.py`, the pre-command refresh hook of `Runner`) | No regional case: pure in-process optimization on the deploy host; run only `tests/regional/test_kubeconfig_cache.py` and the rollout entry point tests (`test_rollout_readonly_modes.py`, `test_release_inflight_install_gate.py`) |
| Release engine kubectl batched reads and release-history in-process cache (`regional_release_state.config_maps_data`, `regional_release_history`, `verify_control_plane_role_split.py`) | Run only the corresponding pytest: `tests/regional/test_release_batched_reads.py`, `tests/deploy/test_verify_control_plane_role_split_*.py`; semantics unchanged, BOOT-018..022 serve as regression |
| Release preflight concurrency, early end of the stability window (`regional_release_preflight_concurrency.py`, `regional_release_validation.py`) | Run only the corresponding pytest (`test_release_preflight_concurrency.py`, `test_release_stability_early_exit.py`, `test_release_stability_window.py`); BOOT-018..022 cover it in a real deploy |

The specific paths and pytest targets are governed by `change-impact.yaml`. Regional case references support
the `GF-REGIONAL-CMD-001..016` range syntax; the selector expands them, sorts them by the formal execution order, and automatically adds
the `pytest_nodeid` or `related_pytest` the case already has.

## Fail-closed escalation

Any of the following escalates to `make check` and the full regional plan:

- changes to shared contracts such as `models.py`, the operation/channel registry, the Store contract, the authorization registry;
- changes to the Python version, dependency lock, runtime image or release identity;
- changes to the PostgreSQL schema, transactions, lease or fencing;
- one change spanning more than three business impact domains;
- implementation files exist that no ordinary rule matches;
- changes to the impact matrix, the selector, the case catalog or the regional execution order itself.

The full regional plan is still only a plan. `GF-REGIONAL-DESTR-004` never enters the selection result; other
destructive/live cases appear only in the approval section and are not executed by `test-impact`.

## Rule maintenance

Each rule contains:

```yaml
- id: remote-command
  description: Remote command protocol and Executor.
  paths:
    - src/gpu_fault/cluster_executor/**
    - src/gpu_fault/store/**/remote_commands.py
  pytest:
    - tests/hyperpod/test_cluster_executor.py
  regional_cases:
    - GF-REGIONAL-CMD-001..016
  checks:
    - config-check
```

When adding a rule, run:

```bash
make impact-check
make docs-check
python3 -m pytest -q tests/test_change_impact.py
```

`impact-check` validates duplicate keys, paths, pytest targets, case existence, the formal order and
the `DO_NOT_RUN` constraint. Any path whose safety cannot be proven should go into a fail-closed rule rather than be silently ignored.
