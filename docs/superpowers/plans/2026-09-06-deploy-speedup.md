# Deploy Speed-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Work packages (WP) have disjoint file scopes and may run in parallel; the coordinator owns WP-F and every shared file.

**Goal:** Cut the wall-clock of `gpu-fault-admin deploy` (measured 40 min for a 4-node repeat deploy, 25.6 min of it in `rollout deploy`) by removing duplicated work, dead caches, an always-firing probe defect, and unnecessary serialization, without weakening any fail-closed gate.

**Architecture:** The deploy is a chain `cli.py → scripts/staging_deploy.py → cli.py --prepared-source-release → admin/bootstrap.py → scripts/release_deploy.py → deploy/control-plane/regional/rollout_regional_release.py`. Each WP below edits one link of that chain. Speed-ups come from (1) fixing defects that make work re-run every time, (2) caching/reusing evidence inside one transaction, (3) running independent steps concurrently while keeping the hard ordering `schema → ingress-staged → data plane → cpu-finalize`, and (4) letting wave size follow the scale cap the code already has.

**Tech Stack:** Python 3.12, pytest (`make test-parallel PYTEST_XDIST_WORKERS=16`), bash, kubectl/aws CLIs behind `CommandRunner`; ten static gates via `make check-static`.

**Spec:** the analysis in this session (2026-09-06), recorded in memory `deploy-timing-profile-and-waste.md`; measured evidence in `/secure/gpu-fault-bootstrap/logs/deploy-20260905T122848Z-2028213.log` and `deploy-20260905T150707Z-2277970.log`.

## Global Constraints

- Red-then-green: write the failing test first, run it, then implement. Run only the touched test files per task; the coordinator runs the full suite once per batch.
- Never widen a fail-closed gate into a fail-open one. Every removed probe must be replaced by evidence already proven in the same transaction, and the replacement must be asserted by a test.
- Do not edit shared files from a WP: `Makefile`, `*-baseline.json`, `docs/**`, `deploy/control-plane/base/*.yaml`, `deploy/dataplane/*.yaml`, `docs/review/IMPLEMENTATION-LOG.md`. Record documentation needs in your fragment; the coordinator applies them.
- Each WP writes its log to `docs/review/fragments/DEPLOY-SPEED-<WP>.md` with sections 改动 / 未做 / 红灯 / 绿灯 (test node ids and counts only).
- Ratchets that bite: ruff I001 import order inside `import (...)` blocks; `scripts/check-mypy-baseline.py` (annotate locals, not just signatures); `scripts/check-test-private-coupling.py` (drive public entrypoints, do not monkeypatch `obj._private`). Do not run `--write-baseline`.
- Tests must not read ambient `GPU_FAULT_ADMIN_LOG`; clear it with `monkeypatch.delenv`.
- Use `.venv/bin/python -m pytest <files>` for targeted runs.
- No commits: the working tree is a shared dirty tree; the user has not asked to commit.

---

## WP-A: source preparation (`scripts/staging_deploy.py`, `scripts/staging_gate_caches.py`, `scripts/setup_deploy_host.py`, `tests/test_staging_deploy.py`, `tests/test_deploy_host_setup.py`)

### Task A1: one `status` per deploy, lock first

**Files:** Modify `scripts/staging_deploy.py` and `:1209-1300`; Test `tests/test_staging_deploy.py`.

Behaviour today: `deploy()` collects live evidence (`gpu-fault-admin status`, ~45 s) before the site lock, `apply_source_deploy` collects it again under the lock, and again after the admin deploy. Three `status` runs per deploy.

Target:
- `deploy()` takes `site_operation_lock(state_dir, wait=True)` **before** the first `collect_live_deploy_evidence` and holds it through `apply_source_deploy` (pass `lock_fd` down; `apply_source_deploy` must accept an already-held lock instead of acquiring its own).
- The pre-apply classification runs once; the "classification changed while waiting for apply lock" re-check and its `status` call are removed (the lock is held, nothing can change).
- The post-deploy `collect_live_deploy_evidence` stays only when `mode != "UNCHANGED"` (it is the evidence recorded as success); when `mode == "UNCHANGED"` reuse the pre-apply evidence.

- [ ] Test: `test_deploy_collects_live_evidence_once_under_lock` — stub `collect_live_deploy_evidence` and `site_operation_lock`; assert the lock is entered before the first evidence call and that evidence is collected exactly once before `run_admin_deploy` and once after (total 2 for APPLICATION_RELEASE, 1 for UNCHANGED).
- [ ] Run, expect FAIL (currently 3 calls / lock entered after first call).
- [ ] Implement; run, expect PASS. Run the whole `tests/test_staging_deploy.py`.

### Task A2: wire the gate tool caches into the APPLICATION_RELEASE path

**Files:** Modify `scripts/staging_deploy.py` (`run_admin_deploy`); Test `tests/test_staging_deploy.py`.

`tool_cache_environment(state_dir)` (`scripts/staging_gate_caches.py`) is only applied in `run_source_impact_gate`. `run_admin_deploy` passes `{**os.environ, SITE_OPERATION_LOCK_FD_ENV: ...}` so mypy/ruff/pytest caches are cold on every real deploy.

- [ ] Test: `test_run_admin_deploy_passes_tool_cache_environment` — capture the env of the spawned command; assert `MYPY_CACHE_DIR` (and every key `tool_cache_environment` produces) is present and points under `state_dir`.
- [ ] Implement: `env={**tool_cache_environment(state_dir), SITE_OPERATION_LOCK_FD_ENV: str(lock_fd)}` (tool_cache_environment already merges os.environ; verify and keep that contract).
- [ ] Run, expect PASS.

### Task A3: fix the copy-pasted venv branch

**Files:** Modify `scripts/staging_deploy.py`; Test `tests/test_staging_deploy.py`.

The `elif not (venv / "bin/gpu-fault-admin").is_file()` branch sets `mode = "APPLICATION_RELEASE"` but `prepared_mode` (captured at :1355) stays QUALITY_ONLY; `apply_source_deploy` reclassifies to QUALITY_ONLY and runs no deploy, while the impact gate at :1411 was skipped because `mode` was APPLICATION_RELEASE. Result: nothing gated, nothing deployed, success recorded.

- [ ] Test: `test_missing_venv_promotes_prepared_mode_and_deploys` — QUALITY_ONLY change, venv missing → `run_admin_deploy` is called (or the impact gate runs); success is not recorded without either.
- [ ] Implement: extract the bundle+venv block into one helper `_ensure_deploy_host(...)`; when the venv is missing set both `mode` and `prepared_mode` to APPLICATION_RELEASE.
- [ ] Run, expect PASS.

### Task A4: prune state-dir caches

**Files:** Modify `scripts/staging_deploy.py` (after `record_successful_source_deploy`) and `scripts/setup_deploy_host.py` (venv versions); Test `tests/test_staging_deploy.py`, `tests/test_deploy_host_setup.py`.

Production state dir holds 59 `source-snapshots/*` (3.5 GB) and 12 GB of `.deployer-venv.versions/*`. Nothing prunes them.

- Keep: the snapshot named in `source-deploy.json`, the one in `source-deploy-success.json` (`prepared_repository_root`), and the newest `GPU_FAULT_SOURCE_SNAPSHOT_RETAINED` (default 5) by mtime. Remove others with `git worktree remove --force` when they are worktrees of the source repo, else `shutil.rmtree`.
- Keep venv versions referenced by the current `deployer-venv` binding JSON plus the newest 3.
- Never prune while a lock is not held; run inside the site lock at the end of a successful deploy.

- [ ] Test: `test_prune_source_snapshots_keeps_referenced_and_newest` — create 8 fake snapshot dirs with mtimes; referenced ones older than the newest 5 survive; the rest are removed.
- [ ] Test (setup_deploy_host): `test_prune_venv_versions_keeps_bound_and_newest_three`.
- [ ] Implement both; run, expect PASS. Note the new env var for the coordinator (env reference regen).

### Task A5: fewer redundant artifact verifications (optional, do last)

`build-release-artifacts.py` runs 3× and `load_component_artifacts` re-hashes AST closures each time (`scripts/component_wheels.py`, `component_artifacts.py`). Add an `functools.lru_cache` keyed on `(path, mtime_ns, size)` for `component_source_digest` per process only if the test `tests/test_component_artifacts.py` still passes unchanged. Skip if it needs more than 30 lines.

---

## WP-B: bootstrap re-validation (`src/gpu_fault/admin/bootstrap_platform_probes.py`, `bootstrap_services.py`, `bootstrap_checkpoint.py`, `deploy/observability/install-amp-monitoring.sh`, `tests/admin/test_admin_bootstrap_probes.py`, `tests/admin/test_admin_bootstrap*.py`)

### Task B1: aurora-refresh probe jsonpath

**Files:** Modify `src/gpu_fault/admin/bootstrap_platform_probes.py`; Test `tests/admin/test_admin_bootstrap_probes.py`.

`_AURORA_REFRESH_POD = "{.spec.jobTemplate.spec.template.spec}"` is concatenated with `"{.containers[0].image}"`. kubectl evaluates both against the root, prints the whole pod spec, and the comparison with `runtime_image` never matches, so the ensure path (apply + verify Job + `kubectl wait --timeout=420s`) runs on every deploy (visible in every production log, e.g. line 925-943 of `deploy-20260905T150707Z-2277970.log`).

- [ ] Test: replace the substring stub (`"image}" in " ".join(arguments)`) with a stub that emulates kubectl: it answers only when the single argument equals exactly `jsonpath={.spec.jobTemplate.spec.template.spec.containers[0].image}` (and the two volume/env projections likewise), and returns the **full spec JSON** for the old concatenated form. Add `test_aurora_refresh_probe_passes_when_live_matches` (no `BootstrapMutationRequired`) and keep the drift test.
- [ ] Run: the new pass test FAILS on current code.
- [ ] Implement: `projection(expression)` builds `f"jsonpath={_AURORA_REFRESH_POD}{expression}"` where `_AURORA_REFRESH_POD = "{.spec.jobTemplate.spec.template.spec"` (no closing brace) and each expression starts with `.` and ends with `}`: `.containers[0].image}`, `.volumes[?(@.name=="artifact")].configMap.name}`, etc. Update every caller in the file.
- [ ] Run, expect PASS.

### Task B2: read the pod-identity agent and each IAM role once per run

**Files:** Modify `src/gpu_fault/admin/bootstrap_services.py`; Test `tests/admin/test_admin_bootstrap*.py`.

- `_ensure_pod_identity_agent` runs in `revalidate_pod_identity_agent` and again inside `control_plane_role` (and LBC), each with an `aws eks wait addon-active`. Cache the successful result on the bootstrap context for the process lifetime (dict keyed by `(cluster_name, region)`); second call returns without IO.
- `_ensure_role` reads `iam get-role` and `get-role-policy` twice (silent existence probe, then logged read). Read once and reuse the parsed document.

- [ ] Test: `test_pod_identity_agent_is_probed_once_per_run` — count `describe-addon`/`wait addon-active` invocations across the foundation tasks; expect 1 each.
- [ ] Test: `test_ensure_role_reads_each_role_once`.
- [ ] Implement; run, expect PASS.

### Task B3: checkpoint digests stop embedding orchestration source files

**Files:** Modify `src/gpu_fault/admin/bootstrap_checkpoint.py`; Test `tests/admin/test_admin_bootstrap_probes.py` and the checkpoint tests.

Digests for `nlb_network`, `pki`, `aurora`, `pod_identity_agent` embed `bootstrap.py`/`bootstrap_services.py`/`bootstrap_common.py`/`bootstrap_checkpoint.py` bytes. Any refactor re-runs the heavy ensure paths (unconditional `rds modify-db-subnet-group`, `rds wait`). Keep digests over inputs that describe the *desired resource* (identities, manifests, assets, admin config, wheel bytes for `aurora_refresh`); drop the module-source components. One-time effect: the first deploy after this change re-runs each ensure once because the digest format changed — record that in the fragment.

- [ ] Test: `test_task_digests_ignore_orchestration_source_edits` — build digests, mutate the recorded source sha of `bootstrap.py`, rebuild; digests equal.
- [ ] Implement; run, expect PASS (existing digest tests must still pass; `aurora_refresh` must still track the wheel bytes).

### Task B4: AMP installer writes only on drift

**Files:** Modify `deploy/observability/install-amp-monitoring.sh` and the restart/`rollout status` block; Test: bash-level test under `tests/` following the existing `_script_loader.py` pattern (see `tests/test_deploy_host_setup.py` for shell testing style), or a Python test that drives the script with a fake `aws`/`kubectl` on `PATH`.

- `iam put-role-policy`: fetch `get-role-policy`, compare canonical JSON, skip when equal.
- `sns create-topic`/`set-topic-attributes`: skip when `get-topic-attributes` matches.
- ADOT restart: only when `kubectl apply` output contains `configured`/`created` for the ConfigMap or Deployment, or when `GPU_FAULT_FORCE_ADOT_RESTART=true`.
- Emit `amp-step-elapsed` lines on stdout so `runner.run` (`bootstrap_services.py`) keeps them in the deploy log.

- [ ] Test: fake CLIs record calls; unchanged inputs → zero `put-role-policy`, zero `set-topic-attributes`, zero `rollout restart`.
- [ ] Implement; run, expect PASS. `bash -n` and shellcheck must stay green (`make shell-check` is coordinator's).

---

## WP-C: rollout ordering and checkpoints (`deploy/control-plane/regional/regional_release_orchestration.py`, `regional_release_mutation_preflight.py`, `regional_release_progress.py`, `tests/regional/test_regional_release_orchestrator.py`, `tests/regional/_release_orchestrator_support.py`, `tests/regional/test_release_progress_persistence.py`, `tests/regional/test_release_phase_narration.py`)

Hard ordering that must survive every task here: `schema-ready` → `registry-staged` → **ingress** `cpu-staged` → `upgrade_gpu_clusters` → `data-converged` → `cpu-finalized` → `verified`. Everything else is movable.

### Task C1: node candidate-preflight runs concurrently with schema/registry/CPU stage

**Files:** Modify `regional_release_orchestration.py`.

Today `preflight_upgrade_mutations` (read-only preflight Jobs on GPU nodes, 100-170 s) runs before schema. It touches only GPU clusters; schema/registry/CPU stage touch only the CPU cluster.

- Start `preflight_upgrade_mutations(self, plan)` in a `ThreadPoolExecutor(max_workers=1)` future right after `uploaded`; run schema → registry → cpu-staged on the main thread; `future.result()` before `upgrade_gpu_clusters`; then `checkpoint("candidate-preflight-ready")`.
- All `_save_state` calls from both threads go through `state_transaction(self)` (already used by `record_progress`). Wrap the main-thread `checkpoint` and `run_component` saves in it too.
- On resume (`"candidate-preflight-ready" in completed_phases`) do not re-run; on `data-converged` completed, skip entirely (existing rule).
- If the preflight future fails, the CPU stage that already ran is compensated by the existing rollback plan (CPU_STAGE has a STARTED/COMPLETED marker); nothing new needed, but add a test that a preflight failure after cpu-staged still records `original_failure` and leaves `cpu-staged` in completed_phases.

- [ ] Test: `test_candidate_preflight_overlaps_cpu_stage` — use the orchestrator support fakes; make the preflight fake block on an `Event` that the CPU stage fake sets; assert the deploy completes (would deadlock/timeout if serialized) and phase order in narration is `uploaded, schema-ready?, registry-staged?, cpu-staged, candidate-preflight-ready, ...`.
- [ ] Test: `test_gpu_rollout_waits_for_candidate_preflight` — data plane fake asserts the preflight fake has finished.
- [ ] Implement; run `tests/regional/test_regional_release_orchestrator.py`, expect PASS.

### Task C2: only the ingress role blocks the data plane

**Files:** Modify `regional_release_orchestration.py` and `rollout_regional_release.py` (`_apply_cpu` accepts `roles: tuple[str, ...] | None`).

`control_plane_role_targets(diff)` may contain `spool`, `worker`, `ingress`. Split the stage:
- `_apply_cpu(finalize=False, roles=("ingress",))` + heartbeat barrier on the main thread (unchanged semantics) when ingress is targeted.
- `_apply_cpu(finalize=False, roles=<worker/spool subset>)` in a background future started right after the ingress stage; joined before `data-converged`. When ingress is not targeted the worker/spool apply runs in the background from the start of the stage.
- `apply-control-plane-role-split.sh` already takes `GPU_FAULT_CONTROL_PLANE_ROLE_TARGETS`; pass the subset. `force_restart=registry_staged` applies to both calls.
- Progress: record `CPU_STAGE` STARTED once, COMPLETED when both halves finish; FAILED if either fails (join the future inside `run_component`'s action so the try/except covers it).

- [ ] Test: `test_worker_spool_stage_overlaps_data_plane` — fakes record the role sets passed to `_apply_cpu`; the data-plane fake starts before the worker/spool fake finishes.
- [ ] Test: `test_ingress_stage_precedes_data_plane` — unchanged ordering when ingress is targeted.
- [ ] Implement; run, expect PASS.

### Task C3: endpoint alongside CPU stage, observability alongside the data plane

**Files:** Modify `regional_release_orchestration.py`.

- `ENDPOINT` (`_apply_nlb`: NLB Service + elbv2/route53 waits) runs in a future started after `registry-staged`, joined before `upgrade_gpu_clusters` (per-cluster `_verify_gpu_control_plane_endpoint` needs the endpoint).
- `OBSERVABILITY` (`_apply_observability`, ~118 s) runs in a future started after `endpoint-ready`, joined before `cpu-finalized`.
- Checkpoints `endpoint-ready`/`observability-ready` are written when their futures complete, via `state_transaction`.

- [ ] Test: `test_observability_overlaps_data_plane` (blocking-Event pattern as in C1).
- [ ] Test: `test_endpoint_ready_before_gpu_rollout`.
- [ ] Implement; run, expect PASS.

### Task C4: half the state ConfigMap writes

**Files:** Modify `regional_release_orchestration.py`; Test `tests/regional/test_release_progress_persistence.py`, `tests/regional/test_release_phase_narration.py`.

- Skip `checkpoint("schema-ready")` when `not plan.has(ReleaseComponent.SCHEMA)`; downstream `"schema-ready" not in completed_phases` checks must treat "not planned" as satisfied (add a helper `phase_done(name)` that returns True when the phase's component is not in the plan).
- `run_component(component, action, *, completes_phase: str | None = None)`: when `completes_phase` is given, the STARTED marker and the preceding phase checkpoint are written in **one** `_save_state(completes_phase, ..., component_progress=STARTED)`. Replace every `checkpoint("x"); run_component(Y, ...)` pair accordingly. Crash semantics are unchanged: after the single write, the state says phase x is complete and Y has started.
- Keep the FAILED-marker save.

- [ ] Test: `test_started_marker_shares_the_phase_checkpoint_write` — count `_save_state` calls for a FULL deploy with the fakes; expect the count to drop by the number of components (assert exact expected number, derive it in the test from the plan).
- [ ] Test: `test_schema_ready_not_written_without_schema_component`.
- [ ] Test: resume with a state that has `schema-ready` absent and SCHEMA not planned proceeds.
- [ ] Implement; run, expect PASS.

### Task C5: skip the pre-finalize heartbeat barrier when convergence evidence is fresh

**Files:** Modify `regional_release_orchestration.py`, `regional_release_progress.py` (`cluster_attempts_with` details).

Every cluster's `wait_agents` (agent_convergence) already proves every node reports the candidate identity. Record on CONVERGED: `converged_at_epoch` and `identity_verified: true` in `cluster_attempts[cluster]`. In `apply_final_cpu`, run the `required_identity` barrier only if any cluster lacks `identity_verified` or `now - converged_at_epoch > FINALIZE_EVIDENCE_MAX_AGE_SECONDS` (default 300, env `GPU_FAULT_FINALIZE_EVIDENCE_MAX_AGE_SECONDS`). The post-finalize barrier stays unconditional.

- [ ] Test: `test_finalize_skips_pre_barrier_with_fresh_convergence` — one barrier call (post) instead of two.
- [ ] Test: `test_finalize_runs_pre_barrier_when_evidence_stale` — set `converged_at_epoch` 10 min ago → two calls.
- [ ] Implement; run, expect PASS. Note the env var for the coordinator.

### Task C6: canary-then-parallel cluster rollout when parallelism is enabled

**Files:** Modify `regional_release_orchestration.py`.

When `upgrade_max_parallel_clusters > 1`, roll the first pending cluster (site order) alone; if it converges, submit the remaining clusters to the pool. Default stays 1 (unchanged blast radius by default; documented reason: a global failure rolls back converged clusters, see session analysis).

- [ ] Test: `test_parallel_clusters_start_after_first_converges`.
- [ ] Implement; run, expect PASS.

### Task C7: agent convergence timeout is a cluster-local failure

**Files:** Modify `deploy/control-plane/regional/regional_release_agent_convergence.py` (this file is shared with WP-D; WP-C owns only this one-line change plus its test; WP-D must not edit line 227).

`ReleaseError(f"{cluster_id} agents did not converge")` → `ClusterLocalReleaseError`. Effect: a single cluster's convergence timeout pauses the release (`PAUSED`, resume from that cluster) instead of rolling back every converged cluster.

- [ ] Test in `tests/regional/test_regional_release_orchestrator.py`: convergence timeout on cluster B after cluster A converged → lifecycle `PAUSED`, A stays CONVERGED, no rollback invoked.
- [ ] Implement; run, expect PASS.

---

## WP-D: data-plane waves (`deploy/control-plane/regional/regional_release_fleet_rollout.py`, `regional_release_node_runtime_rollout.py`, `regional_release_agent_convergence.py` except line 227, `regional_release_config.py`, `src/gpu_fault/admin/site.py`, `tests/regional/test_release_fleet_wave_narration.py`, `tests/regional/test_release_rollout_wait.py`, `tests/admin/test_admin_site*.py`, `tests/regional/test_env_validation.py`)

### Task D1: one wave-safety probe per wave

**Files:** Modify `regional_release_fleet_rollout.py`.

The second `ensure_rollout_wave_safe(timeout_seconds=0, margin=ROLLOUT_AGENT_POST_LEASE_MARGIN_SECONDS)` after `next-wave` re-execs the same probe ~5 s later. Replace it with a local check: the first probe's snapshot must expose `minimum_lease_remaining_seconds` (extend `probes/rollout_wave_safety.py` output if it does not); assert `first_snapshot_min_lease - elapsed_since_probe >= ROLLOUT_AGENT_POST_LEASE_MARGIN_SECONDS`, else run the probe again (fallback keeps fail-closed).

- [ ] Test: `test_wave_runs_one_safety_probe_when_lease_margin_holds` — count probe execs per wave: 1.
- [ ] Test: `test_wave_reprobes_when_lease_margin_too_small`.
- [ ] Implement; run, expect PASS.

### Task D2: wave size follows the scale cap by default

**Files:** Modify `regional_release_config.py`, `src/gpu_fault/admin/site.py`, `regional_release_fleet_rollout.py`.

`upgrade_max_unavailable` default 1 defeats the built-in cap (4/8/16/32). Make `0` mean "auto" (= size cap): config accepts `0..32`; site default `upgradeMaxUnavailable: 0`; `node_rollout_policy` uses `size_cap` when the configured value is 0. First wave stays 1 (canary). Rollback policy unchanged.

- [ ] Test: `test_auto_upgrade_max_unavailable_uses_size_cap` — 4 nodes → 4; 64 → 8; 256 → 16; explicit 2 → 2; UNKNOWN domain → 1.
- [ ] Test: site default renders `upgrade_max_unavailable == 0` and release config validates it.
- [ ] Implement; run, expect PASS. Docs to update (coordinator): `docs/安全与参数参考.md:151`, `docs/开发者部署实现.md:858`, `docs/部署和运维手册.md:4557`.

### Task D3: `wait_agents` polls only the wave's nodes and reports installer Failed fast

**Files:** Modify `regional_release_agent_convergence.py` (not line 227's exception type — WP-C changes that).

`kubectl get nodes -l <cluster>` per 5 s poll lists the whole fleet; use `--field-selector metadata.name=...` is not multi-valued, so pass the explicit node names via `kubectl get nodes <n1> <n2> ...` when the wave is ≤ 16 nodes; otherwise keep the label listing. Poll interval stays 5 s.

- [ ] Test: `test_wait_agents_lists_only_wave_nodes_for_small_waves`.
- [ ] Implement; run, expect PASS.

### Task D4 (coordinator applies the manifest; WP-D writes the test): Reconciler poll 15 s → 5 s

`deploy/dataplane/node-installer-reconciler.yaml` `GPU_FAULT_RECONCILE_SECONDS` 15→5 halves the two poll boundaries per wave (measured 23-48 s install, ~15 s of it is polling). WP-D adds/adjusts any unit test that pins the value (grep `RECONCILE_SECONDS` in tests). Coordinator edits the manifest and runs `make deployment-contracts-update`.

---

## WP-E: driver, verify and stability (`scripts/release_deploy.py`, `deploy/control-plane/regional/regional_admin_commands.py`, `regional_release_validation.py`, `regional_validation_evidence.py`, `tests/admin/test_release_deploy.py`, `tests/admin/test_admin_quick_validation_evidence.py`, `tests/regional/test_regional_release_validation*.py`)

### Task E1: quick validation evidence is reusable after a real deploy

**Files:** Modify `regional_admin_commands.py` (`next_deploy`), `scripts/release_deploy.py`.

After `rollout deploy` the state is `phase=complete, transaction_committed=False`; `next_deploy` labels that `resume: True`, so `_is_clean_noop_diff` rejects it and the log always says "quick validation evidence was not reusable". Verify then re-runs the two read-only verifier scripts (~26 s) that the deploy's quick gate ran 90 s earlier.

- `next_deploy` adds `"pending_commit": True` and `"release_id": <state release id>` when `phase == "complete" and transaction_committed is False`.
- `_is_clean_noop_diff(diff, prepared)` accepts `pending_commit is True and diff["release_id"] == prepared.release_id and diff["state_sha256"] == expected_state_sha256` (the sha the deploy just wrote; read it from the evidence file, which cli writes — verify field names in `cli.py`).

- [ ] Test (`test_admin_quick_validation_evidence.py`): `test_evidence_finalizes_when_deploy_completed_but_not_committed`.
- [ ] Test (`test_release_deploy.py`): verify is invoked with the reusable evidence env and the verifier scripts are marked `reused`.
- [ ] Implement; run, expect PASS.

### Task E2: verify and stability run concurrently

**Files:** Modify `scripts/release_deploy.py`.

`cli verify` (44 s) and `rollout stability` (128 s, mostly sleeping) are independent read-only phases. Run both in a `ThreadPoolExecutor(2)`; if either fails, cancel/ignore the other's result and raise the first failure (rollback path unchanged). Both reports are written as today; `_update_phase` VERIFIED then STABLE order preserved after both complete.

- [ ] Test: `test_verify_and_stability_overlap` — fakes with Events; total fake time < sum.
- [ ] Test: stability failure while verify passes → rollback invoked once; verify failure → rollback invoked once.
- [ ] Implement; run, expect PASS.

### Task E3: stability queue criterion and restart-alert grace

**Files:** Modify `regional_release_validation.py` and the critical-alert check at `:565-570`.

- Queue growth: replace `depth0 < depth1 <= depth2 and age0 < age1 <= age2` with: fail only if `depth[-1] - depth[0] >= max(GPU_FAULT_RELEASE_QUEUE_GROWTH_MIN_DEPTH (default 8), 0.5 * depth[0])` **and** depth is non-decreasing over the last three samples **and** `oldest_age[-1] > 2 * sample_seconds`. Small absolute wobble (0→1→1) after a fleet restart no longer rolls back.
- Critical alerts: alerts whose names are in `GPU_FAULT_RELEASE_STABILITY_GRACE_ALERTS` (default `GpuFaultCollectorSilent,GpuFaultCollectorMetricsSnapshotStale`) are ignored while `now - data_plane_restart_epoch < 600 s`; the release records the restart epoch in state at `data-converged` (WP-C writes `converged_at_epoch`; read the max of those). Any other critical alert still fails immediately.

- [ ] Test: `test_stability_ignores_small_absolute_queue_wobble` (0,1,1 → healthy).
- [ ] Test: `test_stability_fails_on_sustained_relative_growth` (10,20,40 with age > 60 → fail).
- [ ] Test: `test_stability_grace_for_collector_alerts_after_restart` and `test_stability_no_grace_for_other_critical_alerts`.
- [ ] Implement; run, expect PASS. Note both env vars for the coordinator.

---

## WP-F (coordinator): shared files, manifests, docs, verification

- [ ] `Makefile:9` `PYTEST_XDIST_WORKERS ?= $(shell $(PYTHON) -c "import os;print(max(4,min(16,(os.cpu_count() or 4)//4)))")` and `scripts/run_release_gates.py` pytest parallelism reading the same rule (the release gate spawns `make test-parallel-release`, pass `PYTEST_XDIST_WORKERS` explicitly).
- [ ] `deploy/dataplane/node-installer-reconciler.yaml` `"15"` → `"5"`; `deploy/control-plane/base/control-plane-deployment.yaml` `terminationGracePeriodSeconds: 120` → `60`, preStop `sleep 20` → `sleep 10` (NLB deregistration delay is what the preStop covers; check `regional-control-plane-nlb.yaml` deregistration_delay before choosing 10). Run `make deployment-contracts-update`.
- [ ] Env reference: `make env-doc-check` → regenerate with `scripts/generate-env-reference.py` for the new variables from A4, C5, E3.
- [ ] Docs: update the three `upgradeMaxUnavailable` default lines (D2), the phase table in `docs/管理员快速部署.md` (candidate-preflight overlaps CPU stage; observability overlaps data plane), and `docs/管理员日常运维.md:223` (convergence timeout is now cluster-local).
- [ ] Merge fragments into `docs/review/IMPLEMENTATION-LOG.md` as one section.
- [ ] Verification, once: `make check-static`, `make test-parallel PYTEST_XDIST_WORKERS=16`, `GPU_FAULT_ADMIN_LOG=/tmp/outer.log make check PYTEST_XDIST_WORKERS=16` (the deploy's own gate environment).
- [ ] Update memory `deploy-timing-profile-and-waste.md` with what was implemented.

## Self-review

- Spec coverage: every recommendation from the session's three answers maps to a task above except: caching `_ensure_contexts` across the standalone preflight and deploy (8 s, dropped as not worth a new evidence channel); skipping the paused/steady Reconciler redeploys (needs a manifest-identity contract, deferred with reason in WP-D fragment); cross-cluster default parallelism (kept 1, C6 makes >1 safer).
- Type consistency: `run_component(component, action, *, completes_phase=None)` (C4) is the only signature change other WP tasks in the same file depend on; C1-C3 must call it with `completes_phase`.
- Placeholder scan: none of "TBD/TODO/similar to".
