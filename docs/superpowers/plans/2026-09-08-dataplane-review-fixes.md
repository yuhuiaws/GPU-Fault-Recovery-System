# Data-Plane Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. Work packages (WP) have disjoint file scopes and may run in parallel; the coordinator owns every shared file and runs the full suite once per wave.

**Goal:** Close the defects found by the 2026-09-08 read-only review of every data-plane component (node collectors, collector shared layer, Node Agent, completion watcher, cluster executor, HMA watcher, installer reconciler, manifests): 2 P0, 21 P1, the P2 robustness/performance items the owner approved, and the uncovered processing paths that go with them.

**Architecture:** Nothing structural changes. Each task fixes one causal path inside one component and adds the red test that exercises that path. Cross-component themes: (T1) callers of `HttpEventSink.post` migrate to `deliver_event` and treat `BUFFERED` as delivered; (T2) destructive operations become exactly-once on the node and in the executor; (T3) every blocking call gets a hard bound and every long-running Pod gets a liveness signal; (T4) outboxes stop buffering oversized payloads; (T5) manifests stop hard-coding one instance type.

**Tech Stack:** Python 3.12, FastAPI/uvicorn (Node Agent), kubernetes client, pydantic v2, pytest (`.venv/bin/python -m pytest`), bash + systemd units, Kubernetes YAML.

**Spec:** the review reports. Consolidated: `.superpowers/sdd/2026-09-08-dataplane-review-fixes/reports/00-consolidated.md`; per-group reports in the same `reports/` directory (`node-agent-core.md`, `node-agent-operations.md`, `cluster-executor.md`, `completion-watcher.md`, `collector-shared-layer.md`, `host-collector.md`, `gpu-metrics-collectors.md`, `log-collectors.md`, `hma-installer-deploy.md`). A task's "Evidence" line names the report and finding (e.g. `node-agent-core §F1`); read that section before starting — it carries the quoted code, the reproduced scenario and the coverage table.

**Owner decisions (2026-09-08, binding):**
1. T1: keep `post()` semantics; migrate the raw `sink.post` callers to `deliver_event`.
2. Idle-cluster UNKNOWN: in scope — watcher coverage heartbeat + control-plane `covered` resolution (Task 8).
3. DCGM cadence: exporter gets `-c 15000` on both launch paths; collector stays at 15 s.
4. Delete the legacy sync route `POST /v1/node-actions`.
5. node-log collector (default-off) P1/P2 are fixed this round.
6. Installer GPU-health gates become WARN; software-integrity gates stay fatal.

## Global Constraints

- **Red-then-green.** Write the failing test first, run it, show the failure, then implement. Run only the touched test files per task with `.venv/bin/python -m pytest <files> -q -p no:cacheprovider`. The coordinator runs the full suite (`make test-parallel PYTEST_XDIST_WORKERS=16`) once per wave, not per task.
- **Never widen a fail-closed path into a fail-open one.** Where a task relaxes a gate (Task 17 WARN, Task 3 retry), the replacement must be asserted by a test and the fail-closed remainder must be named in the test.
- **Stay inside your WP's file scope.** Do not edit: `Makefile`, `*-baseline.json`, `docs/**` (other than fragments the coordinator asked for), `pyproject.toml`, `src/gpu_fault/channel_registry.py`, `src/gpu_fault/collector_registry.py`, `deploy/control-plane/**`, or any file another WP owns. If your fix needs one of them, stop and report `NEEDS_CONTEXT` naming the file and the exact change.
- **Ratchets that bite** (run `make check-static` only if you touched >3 files; otherwise the coordinator runs it per wave): ruff format + `ruff check` incl. I001 import order inside `import (...)` blocks; `scripts/check-mypy-baseline.py` (annotate locals, not only signatures; no new `Any` leaks); `scripts/check-test-private-coupling.py` (tests drive public entry points, never monkeypatch `obj._private`); `scripts/check-test-source-assertions.py` and `check-assert-messages.py` (every `assert` in tests carries a message). Never run any `--write-baseline`.
- **Contract tests that scan your outputs:** `tests/test_deploy_layout.py`, `tests/test_installation_resources.py`, `tests/deploy/**` for manifests and systemd units; `tests/test_collector_registry.py` for collector wiring; `tests/test_documentation_contracts.py` for route inventories; `tests/test_env_reference.py` for new `GPU_FAULT_*` env vars (a new env var needs a row in `docs/环境变量参考.md` — report it, the coordinator adds it). Run the relevant one before reporting.
- **Tests must not read ambient env:** clear `GPU_FAULT_*` vars you depend on with `monkeypatch.delenv(..., raising=False)`.
- **Commits:** one or more commits per task on the current branch (`worktree-dataplane-review`), imperative subject under 72 chars, body says what path was broken and how the test proves the fix. End every commit message with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Never push, never rebase, never touch another branch, never `git stash`.
- **No new dependencies.** Standard library + what `pyproject.toml` already lists.
- **Report file** (path given in your dispatch) must contain: 改动 / 未做 / 红灯 (test ids + the failing assertion text) / 绿灯 (test ids + counts) / 文档需要 (env vars, manifests, runbook lines the coordinator must add).

---

## WP-N: Node Agent (`src/gpu_fault/node_agent/**`, `tests/node_agent/**` except `test_node_installer_reconciler.py` and `test_node_deployment.py`, `tests/fleet/_fleet_cases_1.py` line 197 only)

### Task 1: ledger exactly-once for destructive node actions; delete the sync route

**Files:** Modify `src/gpu_fault/node_agent/executor.py`, `ledger.py`, `app.py`; Test `tests/node_agent/test_service.py`, `test_ledger_audit.py`, `test_agent_observability.py`, `test_protocol.py`, `test_misc.py`, `tests/fleet/_fleet_cases_1.py` (the node-action URL string).
**Evidence:** `node-agent-core §F1 (P0), §F2, §F3, §F4, §F5`.

Behaviour today:
- `ledger.get()` (`ledger.py:186-187`) returns `None` for an IN_PROGRESS row. If `ledger.save` raises after the handler ran (`executor.py:394-400`), the marker stays; a resubmit computes `attempt = 1` (`executor.py:324`), overwrites the audit row and **re-runs the destructive handler**; `/result` keeps answering 404. In the async path `finalize_action` (`app.py:279-290`) records `FAILED retryable=False attempt=1` for a reset that succeeded.
- `/submit` (`app.py:381-383`) returns any existing row including `FAILED retryable=True`, so the attempt+1 retry contract (`transport.py:338-348`) never fires over HTTP.
- A `ValueError` raised when the pooled `execute` re-validates (expiry while queued) is persisted by `finalize_action` as a permanent FAILED row under the deterministic `command_id`.
- Replay compares `command_id` only (`executor.py:318-321`); `ledger` stores `parameters_digest`/`gpu_uuids` but never compares them. Lines 312-314 are dead.
- `POST /v1/node-actions` (`app.py:359-367`) has no product caller and disagrees with `/submit` about in-flight work.

Target:
- `ledger.latest_row(command_id) -> (state, attempt, result|None)` that does not hide IN_PROGRESS. `execute` derives `attempt` from it. A stale IN_PROGRESS row with no in-flight `Event` is converted to `INTERRUPTED` (`error="result could not be persisted"`, `retryable=False`) and returned; the handler is never re-run.
- On `save` failure: retry the write up to 3 times with 0.2 s sleep (injectable), then write the INTERRUPTED marker; keep the in-memory result in the future so `/result` can still return it.
- `finalize_action` reuses the in-progress attempt number; a `ValueError` from the pooled execute is a rejection: do not persist, drop the future, log once; the next `/result` 404s and the transport resubmits. Count `rejected` once (validation runs twice today).
- `/submit`: an existing `FAILED retryable=True` row falls through to the pool (execute already increments the attempt).
- Replay: on an existing row compare operation, `parameters_digest` and sorted `gpu_uuids`; mismatch → `ValueError("node action command_id reused for a different command")` → 409 `COMMAND_ID_REUSED`, `requires_new_command=True` in `_node_action_rejection`. Delete dead lines 312-314.
- Remove the `POST /v1/node-actions` route. Rewrite the 4 tests that use it to submit + poll; delete the `test_misc.py:180-189` assertion that the route is synchronous; change the fleet URL string to `/v1/node-actions/submit`. Check the route inventory in `tests/test_documentation_contracts.py` still passes.

- [ ] Tests (red): `test_ledger_save_failure_after_reset_never_reexecutes` (flaky save raising once; resubmit twice; exactly one `--gpu-reset`; poll answers INTERRUPTED, not 404, not PENDING); `test_submit_reruns_a_retryable_failure_as_attempt_two` (HTTP surface); `test_expiry_while_queued_is_a_rejection_not_a_ledger_result`; `test_same_command_id_with_different_body_is_rejected_not_replayed` (409 COMMAND_ID_REUSED); `test_sync_node_action_route_is_gone`.
- [ ] Run, expect FAIL for each with the assertion text recorded.
- [ ] Implement; run the whole `tests/node_agent/` directory and `tests/fleet/`, expect PASS.

### Task 2: one reset per quiesce window; quiesce retry is not a dead end

**Files:** Modify `src/gpu_fault/node_agent/operations/reset.py`, `src/gpu_fault/node_agent/quiesce.py`; Test `tests/node_agent/test_quiesce.py`, `test_remediation.py` (or the file that holds reset tests; `_support.py` `FakeRunner`).
**Evidence:** `node-agent-operations §F1 (P1), §F3, §F5`.

Behaviour today:
- `_run_checked` (`reset.py:33-34`) converts only `CalledProcessError`; `subprocess.TimeoutExpired` from `nvidia-smi --gpu-reset` (120/180 s) escapes and `_retryable_action_error` (`executor.py:455-464`) classifies it retryable. The control plane resubmits; `assert_quiesced` passes (state still QUIESCED); a second reset is issued. A different `command_id` for the same incident also passes.
- After any stop/sweep failure the state becomes `QUIESCE_FAILED` (`quiesce.py:528-532`); a retry of the same QUIESCE raises non-retryable "quiesce is incomplete" (`:603-606`) until the 420 s fail-safe timer restores; the workflow is dead while kubelet/FM/DCGM stay down.
- `ctr tasks kill` with `check=True` (`quiesce.py:256-285`) fails quiesce for a task that exited between list and kill.

Target:
- In `_run_reset_with_busy_retry` (and the RESET_ALL path) catch `TimeoutExpired` and raise `RuntimeError("gpu reset outcome unknown after <n>s; refusing to retry automatically")` (non-retryable).
- Under the quiesce lock, before invoking `nvidia-smi --gpu-reset`, record `reset_issued = {"command_id", "issued_at"}` in the quiesce state file; `assert_quiesced(..., for_reset=True)` raises non-retryable when `reset_issued.command_id` exists and differs from the caller's. `restore` clears it.
- In `quiesce()`, when the existing state is `QUIESCE_FAILED` or `RESTORE_FAILED`, call `restore_state_file` inline (idempotent; cancels the timer), then proceed with a fresh quiesce instead of raising "incomplete".
- `ctr tasks kill` uses `check=False`; rely on `_wait_task_stopped`; raise only if still RUNNING after SIGKILL.

- [ ] Tests (red): `test_gpu_reset_timeout_is_not_retryable_and_runs_once` (FakeRunner raising `TimeoutExpired` on `--gpu-reset`; `result.retryable is False`; re-executing the same envelope returns the stored result; one `--gpu-reset` call); `test_second_command_id_cannot_reset_inside_one_quiesce_window`; `test_failed_quiesce_can_be_retried_after_inline_restore` (`ServiceRunner(fail_stop="nvidia-dcgm")` first, clear, second succeeds, `systemd-run` appears twice); `test_ctr_kill_of_an_exited_task_does_not_fail_quiesce`.
- [ ] Run, expect FAIL.
- [ ] Implement; run `tests/node_agent/`, expect PASS. Also run `tests/node_agent/test_quiesce_boot_reconcile.py` (state-file schema gained a key).

### Task 23: Node Agent performance and small operation defects

**Files:** Modify `src/gpu_fault/node_agent/heartbeat.py`, `quiesce.py` (boot reconcile), `operations/clients.py`, `operations/hung_process.py`, `operations/diagnostics.py`, `operations/flight_recorder.py`, `app.py` (reconcile threading only); Test `tests/node_agent/**`.
**Evidence:** `node-agent-core §F8`; `node-agent-operations §F2, §F4, §F6, §F7, §F8, §F9`.

Target:
- Heartbeat: cache `systemctl is-enabled` per unit (refresh only when the unit set changes or every 30 ticks); re-read the XID policy file only when its mtime changes.
- Boot reconcile (`quiesce.py:676-682,766`, `app.py:192-195`): `reconcile_after_boot` skips `_wait_containers_restored` (a new boot recreated the containers) or runs in a daemon thread while `QUIESCE_GPU_SERVICES` is fenced until it finishes. Pick the skip unless a test in `test_quiesce_boot_reconcile.py` asserts the wait; if one does, use the thread + fence.
- `clients.py:46-60`: build uuid→device path from `nvidia-smi -q -x` `minor_number` (like `collectors/gpu/discovery.py:107`); index path only as fallback. Cache device paths once per `_verify_no_clients`.
- `diagnostics.py:183-199`: the Field Diagnostic failure heuristic must not fire on `Error count: 0` / `ERROR: none detected`; extend the negative pattern.
- `diagnostics.py:431`: include `dcgm-quick-diagnostic-*.json` in the retention sweep.
- `hung_process.py:343-467`: bound total strace time (parallel per process or a total budget ≤ 60 s).
- `flight_recorder.py:233-251`: wrap `is_file`/`stat` in `try/except OSError` → `parse_error` entry; one vanished dump must not abort the triage.

- [ ] Tests (red): `test_heartbeat_caches_unit_enablement_between_ticks`; `test_boot_reconcile_does_not_wait_for_containers` (or the fence variant); `test_device_path_uses_minor_number_not_index` (XML fixture: index 0 → minor 3 → `/dev/nvidia3`); parametrised `test_field_diagnostic_benign_error_lines_are_not_failures`; `test_diagnostic_retention_prunes_quick_diag_json`; `test_flight_recorder_vanished_dump_is_a_parse_error_not_a_crash`.
- [ ] Run, expect FAIL; implement; run `tests/node_agent/`, expect PASS.

---

## WP-X: Cluster executor (`src/gpu_fault/cluster_executor.py`, `src/gpu_fault/adapters/node_action/**`, `tests/execution/test_cluster_executor*.py`, `tests/execution/test_node_action*.py`, `tests/execution/_node_action_cases_*.py`, `tests/hyperpod/test_cluster_executor*.py`)

### Task 3: result reporting survives transport errors; a poison command does not sink its batch

**Files:** Modify `src/gpu_fault/cluster_executor.py`; Test `tests/execution/test_cluster_executor_lease_and_report.py` (+ new file if it grows past ~400 lines).
**Evidence:** `cluster-executor §F1 (P1), §F3, §F9, §F10, §F11`.

Behaviour today: `_post`/`_get` (`cluster_executor.py:183,206`) wrap only `HTTPError`; `_execute_and_report` (`:987`) catches only `ClusterExecutorError`. A `URLError`/timeout on `complete` escapes the worker, `run_once` raises, the command stays LEASED until expiry and is re-claimed and **re-executed**. `RemoteCommandClaim.model_validate(response)` (`:234`) fails the whole claim batch when one command is unparseable, after the control plane committed the leases. Counters are mutated from several threads without a lock; 404 is detected by `"(404)" in str(exc)` (`:353`); `run()` logs every `run_once` exception as "claim failed" (`:1084`).

Target:
- `_post`/`_get` wrap `URLError`, `TimeoutError`, `socket.timeout`, `ssl.SSLError`, `ConnectionError` into `ClusterExecutorError(..., status_code=None)`.
- `_execute_and_report`: retry `complete` with jittered backoff (0.5 s, 1 s, 2 s, cap 3 attempts, injectable sleep) while `watch.hold_reason() is None`; retryable = `status_code is None` or in {408, 425, 429, 500, 502, 503, 504}; on final failure count `reported_failures` and return `WAITING`; `except Exception` at the boundary so nothing escapes to `run_once`.
- Claim parsing: validate each item; for an item that fails, post `RemoteCommandResult(status=FAILED, status_source="executor-rejected", error=<validation summary>)` using the raw dict's `command_id`/`lease_token`, and process the rest.
- Counters: one `threading.Lock` around every `+= 1` on shared counters (or a small `_Counters` helper).
- `exc.status_code == 404` instead of string match. `run()` distinguishes "claim failed" from "report failed" in its log.

- [ ] Tests (red): `test_a_transport_error_on_complete_is_retried_under_a_live_lease` (FakeExecutorClient.complete raises `URLError` once then succeeds; one execution, one successful post, `run_once` returns normally, siblings intact); `test_a_report_failure_never_escapes_run_once` (complete raises `ssl.SSLError` every time → `reported_failures == 1`, `run_once` returns, no re-execution on the next cycle within lease); `test_a_malformed_command_in_a_claim_is_failed_without_dropping_its_siblings`.
- [ ] Run, expect FAIL; implement; run the WP-X test files, expect PASS.

### Task 4: WAITING results keep the adapter's continuation state; early exits keep per-node results

**Files:** Modify `src/gpu_fault/cluster_executor.py` (`_execute` around `:1160`), `src/gpu_fault/adapters/node_action/step_execution.py` (`:230,254,270,311`); Test `tests/execution/test_node_action.py` (or `_node_action_cases_*.py`), `tests/execution/test_cluster_executor_lease_and_report.py`.
**Evidence:** `cluster-executor §F2, §F4`.

Behaviour today: executor-manufactured WAITING results (fleet preflight hold, `_retryable_result`, `_retryable_control_plane_result`) carry only their own keys and `complete_remote_command` replaces `result_details`, wiping `agent_baselines`, `spare_failover_pending`, `gpu_client_quiesce_attempt`. In `_fold_result`, early returns for FAILED/INTERRUPTED/retryable nodes assemble `details` without `node_results`, so a terminal FAILED never reports the nodes that succeeded.

Target: in `_execute`, every executor-manufactured WAITING result's details = `{**(command.result_details or {}), **own_details}`. In `_fold_result`, every early-return outcome includes `"node_results": dict(state.node_results)` and `"completed_nodes": sorted(state.node_results)`.

- [ ] Tests (red): `test_a_retryable_adapter_error_keeps_the_previous_cycles_details` (command with `result_details={"agent_baselines": {...}}`, adapter raises a retryable error; posted details still contain `agent_baselines`); `test_a_failed_second_node_still_reports_the_first_nodes_result`.
- [ ] Run, expect FAIL; implement; run the WP-X test files, expect PASS.

### Task 21: executor execution deadline, liveness, and round-trip reduction

**Files:** Modify `src/gpu_fault/cluster_executor.py`, `src/gpu_fault/adapters/node_action/transport.py`, `barriers.py`, `deploy/dataplane/cluster-action-executor.yaml`; Test WP-X test files + `tests/deploy/**` or `tests/test_deploy_layout.py` if they assert this manifest.
**Evidence:** `cluster-executor §F5, §F6, §F7, §F8`.

Target:
- `GPU_FAULT_CLUSTER_EXECUTOR_MAX_EXECUTION_SECONDS` (default 1800): when exceeded, stop renewing, post `FAILED` with `status_source="executor-execution-timeout"`, log the stuck thread's operation. Report the new env var in 文档需要.
- Liveness: an exec probe on breadcrumb age (< 300 s) in `cluster-action-executor.yaml`; the breadcrumb must be written by the poll loop, not by workers.
- SIGTERM handler: set a stop flag checked by `run()`; in-flight commands stop renewing so the lease lapses fast.
- Fetch the `AgentRecord` once per node per cycle and pass it to `_secret_for_node` / `_ssl_context`; cache `SSLContext` per `(node_id, sha256(certificate))`; skip the fleet preflight when `command.result_details` already carries `node_action_command_id` (the mutation has begun).

- [ ] Tests (red): `test_a_stuck_adapter_stops_being_renewed_after_the_execution_cap`; `test_sigterm_stops_claiming_and_stops_renewing`; `test_one_get_agent_per_node_per_cycle`; `test_ssl_context_is_reused_across_sends`; manifest test asserting a `livenessProbe` on the executor container.
- [ ] Run, expect FAIL; implement; run the WP-X test files and the manifest tests, expect PASS.

---

## WP-W: Completion watcher (`src/gpu_fault/completion_controller.py`, `completion_outbox.py`, `completion_observation.py`, `completion_attempt_state.py`, `attempt_observation_state.py`, `completion_metrics_server.py`, `watcher.py`, `deploy/dataplane/completion-watcher.yaml`, `tests/completion/**`; Task 8 additionally owns the control-plane files it names)

### Task 5: failure-detected must be deliverable; the WAL never vetoes the live post

**Files:** Modify `src/gpu_fault/completion_outbox.py`, `src/gpu_fault/completion_controller.py`, `completion_metrics_server.py`; Test `tests/completion/test_completion_outbox.py`, `test_completion_controller.py`, `test_completion_metrics_server.py`.
**Evidence:** `completion-watcher §F1 (P0), §F8, §F12`.

Behaviour today: `KubernetesCompletionOutbox.post` runs `_append` **before** `sink.post` (`completion_outbox.py:337`); `_write_data` raises `CompletionOutboxFull` when the ConfigMap payload exceeds `max_bytes` (900 000). Each snapshot carries a `tail` up to `workload_log_max_bytes` (262 144) and the controller attaches every Pod's snapshot to the failure event (`completion_controller.py:396,714-728`). A 4-rank attempt with normal logs is ~918 KB → every `_deliver_failure_containment` raises inside `_append`, zero POSTs, and after 30 s `KubernetesWorkloadStopper.stop` suspends the workload with no incident recorded. Also the live path and `replay()` both retry the same buffered record every pass.

Target:
- The buffered (WAL) record for a critical event is pointer-sized: strip `workload_log_snapshots[*].tail` down to at most 8 KB (keep `record_id`, `s3_uri`, `sha256`, `truncated` flags) before `_append`; the live POST still sends the full event. On replay the pointer-sized record is what is sent (document that in the model docstring).
- If `_append` raises (`CompletionOutboxFull`, `ApiException`, `OSError`), log once per key, increment `outbox_append_failures_total`, and **still attempt the live POST**. Expose the counter on `/metrics`.
- If `_append` finds the key already buffered, return without a live POST and let `replay()` own delivery (F8).

- [ ] Tests (red): `test_failure_detected_with_large_log_tails_reaches_the_sink` (controller + `KubernetesCompletionOutbox` + `KubernetesWorkloadStopper` fakes, 4 Pods, 260 KB logs each; the inner sink receives failure-detected; no emergency stop); `test_outbox_append_failure_does_not_veto_the_live_post` (fake core_api raising `ApiException(409/500)` on replace; sink still called; counter == 1); `test_already_buffered_record_is_left_to_replay`; metrics test asserting `outbox_append_failures_total` is exported.
- [ ] Run, expect FAIL; implement; run `tests/completion/`, expect PASS.

### Task 6: watch stream timeout, liveness, and reconcile overlap

**Files:** Modify `src/gpu_fault/completion_controller.py`, `completion_metrics_server.py`, `deploy/dataplane/completion-watcher.yaml`; Test `tests/completion/test_completion_controller.py`, `test_completion_controller_resilience.py`, `test_completion_metrics_server.py`, plus the manifest test that covers `deploy/dataplane/*.yaml` (`tests/test_deploy_layout.py` or `tests/deploy/**` — find it with `grep -rl completion-watcher tests`).
**Evidence:** `completion-watcher §F3 (P1), §F5, §F11, §F12`.

Target:
- `watcher.stream(...)` passes `_request_timeout=(5, self.watch_timeout_seconds + 15)`.
- `/metrics` exports `completion_watcher_last_cycle_completed_timestamp` (and `reconcile_runs_total`, outbox depth/quarantined/oldest-age from `outbox.stats()`); the Deployment gets a `livenessProbe` (httpGet on the metrics port hitting a `/healthz` that fails when cycle age > 3 × watch timeout) and a `readinessProbe`.
- `reconcile_lock` becomes an instance attribute taken around the initial full pass and every debounce-timer reconcile; the `finally` joins the in-flight timer thread before `watcher.stop()`; `watcher.stop()` runs even if `flush_pending()` raises (nested try/finally).

- [ ] Tests (red): FakeWatch asserts the `_request_timeout` kwarg; `test_metrics_export_cycle_age_and_outbox_stats`; `test_healthz_fails_when_the_cycle_is_stale`; `test_timer_reconcile_never_overlaps_the_full_reconcile` (slow sink + stream ending right after an event; instrument via a public hook, not `_private`); `test_watcher_stops_even_when_flush_raises`; manifest test asserting liveness + readiness on the watcher container.
- [ ] Run, expect FAIL; implement; run `tests/completion/` + the manifest test, expect PASS.

### Task 7: a false STOPPED tombstone must not blind a resumed attempt; GC'd finishes are not user stops

**Files:** Modify `src/gpu_fault/completion_observation.py`, `completion_controller.py`, `completion_attempt_state.py`; Test `tests/completion/test_completion_controller.py`, `test_completion_terminal_recovery.py`, `test_workload_context_freshness.py`.
**Evidence:** `completion-watcher §F2 (P1), §F7, §F10`.

Behaviour today: `_terminal_observations.get` short-circuits before Pods are looked at (`completion_observation.py:121-127`); `_evict_pruned_attempts` never evicts while Pods exist (`completion_controller.py:663-665`). All Pods absent > 30 s (`cleanup_timeout_seconds`, a container-cleanup budget) → `STOPPED initiator=None` → workflows withdrawn as a user stop; Pods returning under the same attempt-id are invisible forever. Restored RUNNING attempts whose Pods were GC'd during watcher downtime are tombstoned STOPPED instead of SUCCEEDED/FAILED. A Pending Pod without `containerStatuses` is published RUNNING.

Target:
- Record the origin on cached terminal observations (`origin="missing-tombstone"` vs `"observed"`). When live Pods are grouped for an attempt whose cached terminal is a missing-tombstone, drop the cache and `watcher.reset_attempt(...)` the way `_take_over_attempt_spec` does, then observe normally.
- Separate `attempt_missing_grace_seconds` (default 300, env `GPU_FAULT_COMPLETION_ATTEMPT_MISSING_GRACE_SECONDS`) from `cleanup_timeout_seconds`. Report the env var.
- Before tombstoning a *restored* attempt with no Pods, read the workload object's status/conditions through the stopper's clients; `succeeded>0` → SUCCEEDED, `failed>0` → FAILED, else STOPPED.
- Phase is RUNNING only when some container status is `running` or `terminated`; otherwise PENDING.

- [ ] Tests (red): `test_resumed_attempt_after_tombstone_reports_its_failure` (Pods vanish → tombstone → same attempt-id Pods return and rank 0 exits 1 → failure-detected + FAILED terminal posted); `test_missing_grace_is_independent_of_cleanup_timeout`; `test_restored_attempt_with_gc_pods_and_succeeded_job_is_succeeded`; `test_pending_pod_without_statuses_is_pending`.
- [ ] Run, expect FAIL; implement; run `tests/completion/`, expect PASS.

### Task 8: coverage heartbeat — an idle, fully watched cluster is IDLE, not UNKNOWN

**Files:** Modify `src/gpu_fault/completion_controller.py`, `completion_metrics_server.py`; **control plane:** the topology resolver (`grep -rn "class WorkloadTopologyService" src/gpu_fault` — expected under `src/gpu_fault/app/` or `src/gpu_fault/telemetry.py` (the UNKNOWN/covered resolution around lines 203-271)), the ingest route that receives attempt observations (`grep -rn "attempts/observation\|workload-observation" src/gpu_fault/app/routes`), the store method that persists observations (`src/gpu_fault/store/**`), and `src/gpu_fault/telemetry.py`; Test `tests/completion/**`, the topology tests (`grep -rl "is_unknown_not_idle\|WorkloadTopology" tests`), and the store contract tests for the new column/table.
**Evidence:** `completion-watcher §F4 (P1)`; memory `idle-cluster-workload-state-unknown` (live: DESTR-016 attempt 4 was BLOCKED on an idle cluster).

Behaviour today: the watcher posts observations only for `attempt_ids` it sees (`completion_controller.py:633-650`); with no managed job for 10 min `covered=False` → `UNKNOWN` → every node-mutating plan is QUARANTINE + BLOCKED (`telemetry.py:203-207`, `:266-271`; `test_no_observations_for_the_cluster_is_unknown_not_idle`).

Target:
- Watcher: on every completed full pass post a coverage heartbeat `{"cluster_id", "observed_at", "watched_pods", "watched_attempts", "resource_version", "watcher_instance"}` to a new route `POST /v1/attempts/coverage` (cluster-token bucket, same auth as the observation route), through the same sink (not through the critical outbox: a lost heartbeat is fine, the next pass resends). Interval = the pass; no heartbeat is sent when the pass did not complete.
- Control plane: persist the latest heartbeat per cluster (one row per cluster, upsert); `WorkloadTopologyService.resolve` treats a heartbeat fresher than `coverage_freshness_seconds` (default = the existing 600 s UNKNOWN threshold) with zero observations as `IDLE covered=True`; stale heartbeat + no observations stays `UNKNOWN`. Observations still win when present.
- Register the route in `channel_registry`/`collector_registry` **only if** the validators require it — if they do, stop and report `NEEDS_CONTEXT` (the coordinator owns those files). Add the schema migration the store convention requires (see `src/gpu_fault/schema_migrations.py`; if it needs a version bump, say so in the report — schema releases need fail-forward, the coordinator handles deploy).
- Keep `test_no_observations_for_the_cluster_is_unknown_not_idle` passing (no heartbeat → UNKNOWN).

- [ ] Tests (red): `test_watcher_posts_a_coverage_heartbeat_after_each_full_pass` (and not after an aborted pass); route test: 401 without token, 202 with; `test_fresh_coverage_heartbeat_makes_an_empty_cluster_idle`; `test_stale_coverage_heartbeat_leaves_the_cluster_unknown`; store round-trip test.
- [ ] Run, expect FAIL; implement; run `tests/completion/`, the topology tests, the store tests and `tests/test_documentation_contracts.py` (route inventory), expect PASS.

### Task 22: completion outbox capacity and write path

**Files:** Modify `src/gpu_fault/completion_outbox.py`, `completion_attempt_state.py`, `completion_controller.py`; Test `tests/completion/**`.
**Evidence:** `completion-watcher §F6, §F9`.

Target: persist per attempt only the spec plus `pod uid/rank/node` (rebuild the rest from Pods on restore); track outbox depth in-process and skip the replay GET when known empty; split routine `active-attempts` state and critical WAL into separate ConfigMaps (`<name>` and `<name>-active`) so routine state cannot starve critical delivery; log a permanently malformed Pod once per (attempt, error digest) at ERROR, then DEBUG.

- [ ] Tests (red): `test_a_thousand_running_pods_fit_the_active_state_and_a_terminal_still_delivers`; `test_replay_skips_the_get_when_the_outbox_is_known_empty`; `test_malformed_pod_logs_one_error_then_debug`.
- [ ] Run, expect FAIL; implement; run `tests/completion/`, expect PASS. Note any RBAC change (a second ConfigMap name) in 文档需要; the Role in `completion-watcher.yaml` must list it (`resourceNames`).

---

## WP-C: Collectors (`src/gpu_fault/collectors/**`, `src/gpu_fault/collectors_cli.py`, `src/gpu_fault/transport/http_client.py`, `tests/collectors/**`; Task 11 also owns `deploy/systemd/gpu-fault-host-collector.service`)

### Task 9: finish ARCH-G3 — every collector treats BUFFERED as delivered

**Files:** Modify `src/gpu_fault/collectors/host/collector.py` (the post at lines 425-441), `gpu/dcgm.py:372`, `gpu/nvidia_smi.py:212,299`, `gpu/discovery.py:383` (`deliver_gpu_inventory`), `training_progress.py:78`, `cloud/kubernetes.py:59,95-116,402`, `cloud/cloudwatch.py:91,179`, `logs/fabric_manager.py:166` (health summary), `logs/kernel.py:303` (stats); Test `tests/collectors/test_host.py`, `test_gpu.py`, `test_cloud.py`, `test_logs.py`, `test_delivery_result.py`.
**Evidence:** `collector-shared-layer §F1 (P1), §F3 (P1)`; `host-collector §F5`; `log-collectors §F13`; `hma-installer-deploy §F2`.

Behaviour today: `HttpEventSink.post` raises `CollectorError(buffered=True)` after the outbox took the record; the 13 raw `sink.post` callers treat it as failure: bookkeeping (`_last_delivered_at`, edge state, next-summary time) is skipped so the host collector re-blocks ~47 s and re-buffers a full batch every tick; the node-resources collector aborts its whole node loop on the first buffered post and commits `_last_state`/`_mismatch_counts` before delivery; the HMA watcher's initial LIST + posts are outside its `try` so a buffered post crash-loops the process.

Target: each caller uses `deliver_event(...)`; `DELIVERED` and `BUFFERED` both advance the caller's bookkeeping; `FAILED` keeps today's error behaviour. Node-resources: commit `_last_state` only after a non-FAILED result; per-node `try/except Exception: log; continue`. HMA watcher (`cloud/kubernetes.py:94-98`): LIST + initial posts inside the guarded block; the run loop never exits on a sink error; skip `DELETED` events and drop their `_resource_versions` entry (F13). Fabric-manager health summary and kernel stats posts wrapped like `kernel.py:272-282`.

- [ ] Tests (red), using `_BufferingSink` from `tests/collectors/test_delivery_result.py`: host `collect_once` under a buffering sink sets `_last_delivered_at` (observe via the next tick not posting an unchanged batch — public behaviour, no private access); node-resources `collect_once` over two nodes with a buffering sink delivers both; with a rejecting sink on node A, node B is still sampled and A's edge re-fires next cycle; `test_kubernetes_hma_run_survives_sink_failure_during_relist` (fake api/watch, sink raising once); `test_kubernetes_collector_ignores_deleted_node_events`; fabric-manager health summary under a buffering sink does not log "log collection failed".
- [ ] Run, expect FAIL; implement; run `tests/collectors/`, expect PASS.

### Task 10: sink hardening

**Files:** Modify `src/gpu_fault/collectors/sinks.py`, `src/gpu_fault/transport/http_client.py` (only if the id-key list lives there); Test `tests/collectors/test_http.py`, `test_http_replay.py`, `test_outbox_dead_letter.py`.
**Evidence:** `collector-shared-layer §F2 (P1), §F5, §F6, §F10, §F11, §F12, §F13`.

Target:
- `_kick_outbox_replay`: the synchronous `_replay_outbox()` is wrapped in `try/except OSError` (log, mark inactive, return); replay failure never surfaces through `post()`/`deliver()`; the background replay thread logs and continues on the next kick rather than stopping.
- Non-JSON 2xx body → `CollectorError(status_code=<2xx>)` (or `{}`), never `JSONDecodeError`.
- `Retry-After`: `delay = min(retry_after, 30) + jitter`; HTTP-date form → 0.
- Idempotency-key derivation recognises `snapshot_id` and `log_event_id`; for HMA node events derive `node/<name>/<resourceVersion>`; the stale-keep-alive retry applies to them.
- Receipt poll: retryable HTTP errors (`is_retryable_collector_status`) continue polling until the deadline; only non-retryable codes fail.
- Receipt GET does not send `Content-Type`, `Content-Encoding`, `Idempotency-Key`.
- `_replay_outbox` releases `_outbox_lock` while performing network I/O (read under lock, deliver outside, rewrite under lock).

- [ ] Tests (red): `OutboxFile.write` raising `OSError(28)` → `deliver()` returns `DELIVERED` and the record stays; `test_non_json_2xx_is_a_collector_error`; `Retry-After: 3600` → sleep ≤ 30 + jitter; parametrised retry test for `snapshot_id`/`log_event_id`; receipt poll with one 503 then 200 succeeds; GET header assertion; a live `post` during replay does not wait for the replay's network call (use a sink whose replay delivery blocks on an Event; assert the buffer call returns first).
- [ ] Run, expect FAIL; implement; run `tests/collectors/`, expect PASS.

### Task 11: host collector cannot wedge; progress-file errors are not hangs

**Files:** Modify `src/gpu_fault/collectors/host/collector.py`, `host/system_metrics.py`, `host/network.py`, `host/inventory.py`, `training_progress.py`, `deploy/systemd/gpu-fault-host-collector.service`; Test `tests/collectors/test_host.py` (+ new `test_host_bounds.py` if it grows), the systemd unit test in `tests/deploy/**` or `tests/node_agent/test_node_deployment.py` (`grep -rl WatchdogSec tests` first).
**Evidence:** `host-collector §F1 (P1), §F2 (P1), §F3 (P1), §F4, §F6`.

Target:
- `_nvidia_smi` (`collector.py:346`): `Popen` + `communicate(timeout=...)`; on timeout `kill()`, `wait(timeout=1)`, swallow the second timeout, hand the child to a daemon reaper thread, set the breaker → `CollectorError` within `timeout + 2 s`.
- `_shared_filesystems`/`_filesystems`: `statvfs` in a worker thread with `future.result(timeout=5)`; on timeout emit `shared_filesystem_unavailable=1`, skip used-percent, do not re-probe that mount this tick.
- Unit: `Type=notify`, `WatchdogSec=90`; `run()` calls `sd_notify("READY=1")` once and `WATCHDOG=1` each completed tick via the `NOTIFY_SOCKET` datagram (stdlib socket; no dependency).
- `training_progress.py:361-389`: parse/validate inside `try/except (OSError, ValueError, ValidationError)`; still post a heartbeat with progress fields `None` and `labels["progress_error"]=<class name>`; warn once per distinct error.
- `_network`: per-interface `try/except OSError: continue`; skip interfaces without `/sys/class/net/<if>/device` unless in `required_interfaces`; prune `net/` keys not seen this tick.
- `inventory.py:142`: on non-zero nvidia-smi exit still emit inventory samples from `/proc/driver/nvidia/gpus/*` (discovered) vs UUIDs parsed from partial stdout (observed) with `failure_mode=DRIVER_QUERY_FAILED`, so `gpu_inventory_mismatch` can reach the consumer.

- [ ] Tests (red): fake Popen whose `wait()` blocks on an Event → `_nvidia_smi` raises `CollectorError` within the bound and the breaker is set; `statvfs` sleeping 10 s for `/fsx` → `_shared_filesystems` returns within 6 s with value 1; malformed progress file variants (`{"labels": "x"}`, `{"step": -1}`, truncated JSON) → heartbeat posted with `step is None`; fake sysfs with a vanishing interface and a veth → physical interface survives, veth absent; runner rc=255 with 7 of 8 UUIDs → `gpu_inventory_mismatch` reaches 1 after `inventory_mismatch_consecutive_samples`; unit test asserting `WatchdogSec` + `Type=notify`.
- [ ] Run, expect FAIL; implement; run `tests/collectors/` + the unit-file test, expect PASS.

### Task 12: DCGM collector delivers through failures and grades duty cycles correctly

**Files:** Modify `src/gpu_fault/collectors/gpu/dcgm.py`, `gpu/discovery.py`, `gpu/nvidia_smi.py`; Test `tests/collectors/test_gpu.py`, `test_gpu_resilience.py`.
**Evidence:** `gpu-metrics-collectors §F1 (P1), §F2 (P1), §F3 (P1), §F5, §F6, §F7, §F9`.

Target:
- `query_nvidia_temperature_limits` converts `OSError`/`TimeoutExpired` to `CollectorError` (as `query_gpu_inventory` does); `deliver_gpu_inventory` converts pydantic `ValidationError` to `CollectorError`; in DCGM mode a limits/inventory failure never blocks the scrape; after 3 consecutive failures the limits query backs off to the inventory interval.
- Duty cycle: store `(value, observed_at)` per key and compute `elapsed` from that key's own previous observation; `_implausible_duty_cycle_keys` entries expire after 10 × interval.
- DCGM `run()` mirrors nvidia-smi mode: `_report_collection_error` batch on failure and `next_interval_seconds` backoff.
- nvidia-smi fallback: one merged `--query-gpu` per round (split only on non-zero exit); `observed_at` stamped after the query returns; `parse_csv` skips blank rows.
- Remove the dead `_history` deque in dcgm.py (and its `context_history=[]` plumbing if nothing else reads it — check the consumer contract in `gpu_metrics.py` first; if the field is part of the model keep the field, drop the deque).

- [ ] Tests (red): `test_dcgm_scrape_is_delivered_when_temperature_limit_query_hangs`; `test_dcgm_metrics_survive_inventory_validation_error` (`GPU_FAULT_EXPECTED_GPU_COUNT=4`, 8 GPUs); `test_duty_cycle_uses_the_counters_own_previous_observation` (same value twice at 15 s spacing then a 30 s delta → graded once at the true value, not blacklisted); `test_implausible_duty_cycle_blacklist_expires`; `test_dcgm_run_reports_scrape_failure_as_error_batch`; `test_nvidia_smi_round_issues_one_query`.
- [ ] Run, expect FAIL; implement; run `tests/collectors/`, expect PASS.

### Task 13: kernel and fabric-manager collectors

**Files:** Modify `src/gpu_fault/collectors/logs/kernel.py`, `logs/fabric_manager.py`; Test `tests/collectors/test_kernel_resilience.py`, `test_logs.py`, `test_xid_kmsg_catalog_replay.py`.
**Evidence:** `log-collectors §F2 (P1), §F3, §F4, §F5, §F6, §F8, §F9, §F12 (fabric part), §F15`.

Target:
- fabric_manager file `record_id` = sha256 over `cluster_id, node_id, boot_id, st_dev, st_ino, offset` (mirror `node.py:610-617`); `evidence_ref` includes `node_id`.
- `journalctl ... --output=json --all` in fabric_manager (and kernel if it shells out to journalctl anywhere).
- Rename-rotation: on a new path key, look up a tracked record with the same `(st_dev, st_ino)`, inherit its offset, drop the old key.
- `path.open()` and reads inside the `try/except OSError` guard; warn and `continue`, never starve the journal source.
- Kernel delivery decoupled from the kmsg read loop: a bounded in-process queue (default 2048, drop-oldest counted in stats) drained by one delivery thread; the reader never blocks on the sink.
- Kernel: `deliver_event` wrapped in `except Exception` inside `collect_lines` (count `delivery_failures`, continue) so a non-`CollectorError` never reopens `/dev/kmsg`.
- `unknown-boot` fallback → `f"unknown-{uuid4().hex[:12]}"` computed once per process.
- fabric_manager file tails read in binary with byte offsets, decode per line.
- Kernel: replace per-line `CollectorStats.model_copy` with plain counters, build the model on read.

- [ ] Tests (red): two fabric collectors (`node_id` a/b, same file) → different `record_id`; `--all` present in both journal commands; rename-rotation delivers the appended SXID; unreadable configured file + a journal SXID → SXID delivered; a sink whose `post` blocks on an Event → 3 kmsg records read before the Event is set; sink raising `ValueError` on the first post → `opens == 1`, second record delivered, counter == 1; two collectors with unreadable boot_id → different `record_id`; fabric offset test with a multibyte line.
- [ ] Run, expect FAIL; implement; run `tests/collectors/`, expect PASS.

### Task 14: node-log collector (default-off) — scan budget, streaming, robustness

**Files:** Modify `src/gpu_fault/collectors/logs/node.py`; Test `tests/collectors/test_node_log_*.py`, `test_logs.py`.
**Evidence:** `log-collectors §F1 (P1), §F3 (node part), §F4 (node part), §F7, §F10, §F11, §F12 (node part), §F14`.

Target:
- Journal read via `Popen`, streamed line by line; stop after a byte/time budget (4 MiB or 200 ms per source per poll, constants at module top) or after `max_entries_per_batch` **matched + context** entries; `consumed_until` = last entry read; `terminate()` the child.
- Training-log tails: same budget; count only matched+context entries against the cap.
- `--all` on the journal command; per-entry `try/except` for malformed journal entries → `_record_discard("unparseable-journal-entries")`, cursor advances.
- Rename-rotation inherits the offset by `(st_dev, st_ino)`.
- Self-unit exclusions counted separately from loss; `_discarded` snapshot/restore with the cursor on delivery failure.
- Binary reads with byte offsets; `_save_state` fsyncs file and directory.
- Add a test with a buffering/failing sink (the report notes the rollback path has none).

- [ ] Tests (red): 3000 non-matching journal lines then one MCE → MCE in the batch and `journal_since == until`; same for a 3000-line training log; runner yielding 50 k lines → reading stops at the cap and memory does not hold them all (assert on the number of lines consumed from the fake stream); one `"1"` line among valid entries → others delivered, cursor advances; `_BufferingSink` → cursor advances; error batch does not mention `self-unit-entries`.
- [ ] Run, expect FAIL; implement; run `tests/collectors/`, expect PASS.

### Task 19: host collector performance

**Files:** Modify `src/gpu_fault/collectors/host/system_metrics.py`, `host/network.py`, `host/collector.py`; Test `tests/collectors/test_host.py`.
**Evidence:** `host-collector §F7, §F8, §F9`.

Target: cache SMART results ≥ 300 s (rescan only when the device list changes); emit zero-valued deltas only for the names the consumer needs to clear signals (`rdma_errors_delta`, `efa_*_errors_delta`, `network_*_delta`), drop the per-port traffic deltas when zero; delete the unread `_history` (keep `context_history` field if the model requires it); `gpu_rank.py:195,251` reset `_rank_progress_at` per attempt change; `/proc/stat` `guest`/`guest_nice` excluded from the total.

- [ ] Tests (red): counting runner, two ticks 15 s apart → one SMART scan; sample-name set of a quiet tick has no zero traffic deltas; cpu total excludes guest fields.
- [ ] Run, expect FAIL; implement; run `tests/collectors/`, expect PASS.

### Task 20: collector outbox write path

**Files:** Modify `src/gpu_fault/collectors/sinks.py` (`OutboxFile`, `_buffer_event`, `_replay_outbox`), `src/gpu_fault/collectors_cli.py` (`requeue-dead`); Test `tests/collectors/test_outbox_dead_letter.py`, `test_http_replay.py`, `test_cli.py`.
**Evidence:** `collector-shared-layer §F4, §F7, §F9`.

Target: `_buffer_event` appends one line (`open(..., "a")`) and compacts only when depth > `outbox_max_records` or at replay; `OutboxFile.write` fsyncs the temp file and the directory before `os.replace`; an `fcntl.flock` on `<outbox>.lock` guards every read-modify-write in both the sink and the CLI; 413-rejected dead records store a digest + the first 4 KB of payload, not the whole body.

- [ ] Tests (red): N buffered events → O(1) `write` calls (count via a fake `OutboxFile` subclass exposed for tests or via file mtime/line-count checks); `os.fsync` called (monkeypatch `os.fsync`); interleaved `requeue_dead` between read and write loses no record; 413 dead record is bounded.
- [ ] Run, expect FAIL; implement; run `tests/collectors/`, expect PASS.

### Task 24: HMA watcher dedupe and SQS consumer error handling

**Files:** Modify `src/gpu_fault/collectors/cloud/kubernetes.py` (HMA watcher class only), `cloud/cloudwatch.py`; Test `tests/collectors/test_cloud.py`.
**Evidence:** `hma-installer-deploy §F10, §F12, §F13`; `collector-shared-layer §F14`.

Target: dedupe on a hash of the HMA-relevant subset (labels/annotations ∩ `HMA_KEYS`, HMA taint, true conditions) rather than `resourceVersion`; strip `status.images` before posting; resume the watch from the last `resourceVersion` and relist only on 410; paginate the relist like the sibling collector; prune `_resource_versions` on DELETED. `SqsHmaConsumer.run()`: `try/except` with backoff around `receive_message`; a message received ≥ 5 times (`ApproximateReceiveCount`) is deleted with a warning.

- [ ] Tests (red): `test_kubernetes_collector_skips_unchanged_hma_content_across_resource_versions`; `test_hma_watcher_resumes_from_resource_version_and_relists_on_410`; `test_sqs_consumer_survives_receive_error`; `test_sqs_poison_message_is_dropped_after_five_receives`.
- [ ] Run, expect FAIL; implement; run `tests/collectors/`, expect PASS.

### Task 25: migrate the duplicated BUFFERED/FAILED handling to `deliver_or_raise`

**Files:** Modify `src/gpu_fault/collectors/host/collector.py`, `gpu/dcgm.py`, `gpu/discovery.py`, `gpu/nvidia_smi.py`, `training_progress.py`, `cloud/kubernetes.py`, `cloud/cloudwatch.py`; Test `tests/collectors/**` (existing tests must keep passing; add one test per family asserting the shared warning text).
**Evidence:** Task 9 review (collector-shared-layer) Minor: the six-line `result.raise_for_failure()` + "persisted to the collector outbox" warning block is repeated 8× across the callers; Task 10 added the public `deliver_or_raise(sink, path, payload, *, logger, what) -> DeliveryResult` in `sinks.py` for exactly this. Coordinator ruling (2026-09-08): migrate after Tasks 11/12/13/24 landed so no file is edited concurrently.

Target: each of the 8 call sites calls `deliver_or_raise(...)` and drops its local BUFFERED-warning block; behaviour is unchanged (FAILED still raises the original `CollectorError`; BUFFERED logs exactly one warning with the id and `result.error`); no caller keeps a hand-rolled copy. `grep -rn "persisted to the collector outbox" src/gpu_fault/collectors` afterwards hits only `sinks.py`.

- [ ] Tests (red): one test per family (host, gpu, cloud, training-progress) asserting the BUFFERED warning comes from `deliver_or_raise` (single record, the shared text) — red because the local text differs or is duplicated.
- [ ] Implement; run `tests/collectors/`, expect PASS.

---

## WP-D: Deploy, installer, reconciler, HMA ingest (`deploy/dataplane/**` except `completion-watcher.yaml` and `cluster-action-executor.yaml`, `deploy/systemd/**` except `gpu-fault-host-collector.service`, `deploy/node/**`, `src/gpu_fault/node_installer_reconciler.py`, `src/gpu_fault_release/regional_gpu_bootstrap.py`, `src/gpu_fault/hma.py`, `src/gpu_fault/app/routes/gpu_events.py`, `tests/node_agent/test_node_installer_reconciler.py`, `tests/node_agent/test_node_deployment.py`, `tests/deploy/**`, `tests/test_deploy_layout.py`, `tests/test_installation_resources.py`, `tests/hma/**`)

### Task 15: installer reconciler — budget, isolation, backoff

**Files:** Modify `src/gpu_fault/node_installer_reconciler.py`; Test `tests/node_agent/test_node_installer_reconciler.py`.
**Evidence:** `hma-installer-deploy §F3 (P1), §F5, §F6 (reconciler part), §F8, §F14`.

Target: pre-count active installer Jobs (label `INSTALLER_JOB_LABEL=true`, not Complete/Failed) before the node loop so `max_unavailable` is never overshot; per-node `try/except Exception: log; count "error"; continue`; every Kubernetes call passes `_request_timeout=(5, 60)`; skip the node patch when the annotation already equals the target; track attempt count in an annotation and back off failures (300 s × 2^n, cap 3600 s); a Job whose Pod never started (`CreateContainerConfigError`, e.g. missing node-action key) is classified `"unsupported"` with a clear annotation and not retried until the key Secret's resourceVersion changes (if the reconciler cannot read Secrets, use the Job Pod's waiting reason and back off with the same curve — say which in the report). At the end of every reconcile pass write the heartbeat file `/tmp/reconciler-heartbeat` (touch/utime; constant path, no env var) so Task 16's liveness probe can read its age.

- [ ] Tests (red): `test_reconcile_pass_touches_the_heartbeat_file`; `test_running_job_on_a_later_node_blocks_creation_on_an_earlier_node` (per-name Job map fixture); `test_a_failing_node_does_not_block_later_nodes`; `test_reconcile_passes_request_timeouts`; `test_failed_node_is_not_repatched_every_pass`; `test_node_without_action_key_is_reported_not_retried`.
- [ ] Run, expect FAIL; implement; run `tests/node_agent/test_node_installer_reconciler.py`, expect PASS.

### Task 16: manifests and systemd units

**Files:** Modify `deploy/dataplane/hyperpod-dcgm-exporter.yaml`, `deploy/dataplane/node-installer-reconciler.yaml`, `deploy/dataplane/kubernetes-node-resource-collector.yaml`, `deploy/dataplane/optional/*.yaml`, the historical in-cluster GPU metrics DaemonSet manifest `gpu-metrics-collector.yaml` under `deploy/dataplane/` (delete, and fix the tests that assert it), `deploy/systemd/gpu-fault-dcgm-exporter.service`, `deploy/systemd/gpu-fault-node-agent.service`, `src/gpu_fault_release/regional_gpu_bootstrap.py`; Test `tests/test_deploy_layout.py`, `tests/test_installation_resources.py`, `tests/deploy/**`, `tests/regional/**` tests that render the exporter manifest.
**Evidence:** `gpu-metrics-collectors §F4 (P1), §F7, §F8`; `hma-installer-deploy §F1 (P1), §F6 (probes), §F7, §F11, §F16, §F18`; `node-agent-core §F7`; owner decision 3.

Target:
- Exporter DaemonSet: replace the `nodeSelector` with `nodeAffinity … In [<every instance type in node_installer_reconciler._INVENTORY>]`, rendered by `regional_gpu_bootstrap.py` from that one table (import it; do not duplicate the list); `-c 15000` in the DaemonSet `args` **and** in `gpu-fault-dcgm-exporter.service`; bind `-a 127.0.0.1:9400` in the DaemonSet as well (hostNetwork); `priorityClassName: system-node-critical`.
- Reconciler Deployment: tolerations = the watcher's two (`quarantined`, `unschedulable`) + `node.kubernetes.io/not-ready` and `unreachable` with `tolerationSeconds: 60`; a liveness probe on a per-pass heartbeat file age (< 300 s; the reconciler writes it — coordinate with Task 15 which owns the code: the file path is `/tmp/reconciler-heartbeat`, written at the end of each pass; if Task 15 has not landed, add the write yourself in a minimal way and say so).
- Node-resources Deployment: `livenessProbe` on the collector's existing metrics/health surface if one exists; otherwise an exec probe on a heartbeat file the collector writes each cycle (add the write in `cloud/kubernetes.py` only if no other surface exists — that file is WP-C's; if you must touch it, stop and report `NEEDS_CONTEXT`).
- Optional HMA manifests: mirror `kubernetes-node-resource-collector.yaml` (secretKeyRef URL/token/cluster-id, `SSL_CERT_FILE`, CA volume); drop the unused `artifact` volume in the CloudWatch consumer; `priorityClassName: system-cluster-critical` on singletons.
- Delete the historical `gpu-metrics-collector.yaml` DaemonSet manifest under `deploy/dataplane/` and update the tests that assert it (`test_in_cluster_collectors_mount_writable_state_directory` etc.).
- `gpu-fault-node-agent.service`: `TimeoutStopSec=1900` (≥ the longest handler timeout), `KillMode=mixed`.

- [ ] Tests (red): `test_dcgm_exporter_daemonset_matches_every_reconciler_instance_type`; `test_dcgm_exporter_collects_every_15_seconds_on_both_launch_paths`; `test_exporter_binds_loopback`; extend `test_data_plane_collectors_tolerate_cordoned_nodes` to the reconciler and assert no bare `operator: Exists`; `test_every_data_plane_deployment_has_a_liveness_probe` (executor and watcher probes are other WPs' work — this test may need to land last; if it fails only on those two manifests, mark those cases `xfail(strict=True)` with the task number and say so); `test_optional_hma_manifests_use_the_regional_connection_secret`; node-agent unit test asserting `TimeoutStopSec`.
- [ ] Run, expect FAIL; implement; run the listed test files, expect PASS. Also run `make yaml-check` equivalent if present (`grep -n yaml-check Makefile`).

### Task 17: installer GPU-health gates become WARN; agent restart is safe

**Files:** Modify `deploy/node/install-gpu-fault-collector.sh`, `deploy/node/verify-gpu-fault-collector.sh`; Test `tests/node_agent/test_node_deployment.py`, `tests/deploy/**` (find the shell-script contract tests: `grep -rl verify-gpu-fault-collector tests`).
**Evidence:** `hma-installer-deploy §F9`; `node-agent-core §F7`; owner decision 6.

Target:
- Installer: `nvidia-smi -L` failing to enumerate stays fatal only when it enumerates **zero** GPUs; persistence-mode not Enabled on some GPU → `WARN` line + write `/var/lib/gpu-fault/installer-degraded-gpu.json` (`{"observed_at", "gpus": [{"index","uuid","persistence_mode","error"}]}`), continue.
- `verify`: "GPU persistence mode" and "DCGM exporter supported metrics" become `WARN` (non-fatal, counted separately, printed); "metrics delivered to control plane", every `systemctl is-active` check, the Agent readiness probe and the wheel/digest checks remain `FAIL` (fatal). `verify` exits 0 with WARNs; exit 1 only on FAILs. Print a final `SUMMARY pass=<n> warn=<n> fail=<n>` line.
- Installer restart of the Agent during an upgrade: before `systemctl restart gpu-fault-node-agent`, query the ledger for IN_PROGRESS rows (`sqlite3` if present, else the Python venv) and wait up to `TimeoutStopSec` for them to clear; if still present, `die` with the command ids.
- The degraded-GPU marker is read by nothing yet: note in 文档需要 that the host collector should surface it (follow-up, WP-C).

- [ ] Tests (red): script contract tests: `verify` exits 0 and prints `WARN  GPU persistence mode` when one GPU is not Enabled (stub `nvidia-smi` via PATH); `verify` still exits 1 when `systemctl is-active` fails; installer writes the degraded marker; installer refuses to restart the agent while a row is IN_PROGRESS (stub the ledger query).
- [ ] Run, expect FAIL; implement; run the script tests + `make shell-check` equivalent (`grep -n shell-check Makefile`), expect PASS.

### Task 18: an HMA "Unschedulable" node without a parsable code opens a WARNING finding

**Files:** Modify `src/gpu_fault/hma.py` (`normalize_node`, `normalize_cloudwatch`), `src/gpu_fault/app/routes/gpu_events.py` (the `_ingest_hma` path); Test `tests/hma/test_hma.py`, `tests/hma/test_unparsed_fault_lines.py`, the route test file that exercises `/v1/provider-events/hyperpod-hma/*`.
**Evidence:** `hma-installer-deploy §F4`.

Target: `normalize_node`/`normalize_cloudwatch` add an unresolved reason when `health_status == "Unschedulable"` (or the HMA taint is present) and no XID/SXID code was extracted; `_ingest_hma` calls `dependencies.ingest_unresolved_signals(normalized, batch_id=signal.signal_id)` the way the kernel and fabric-manager routes do (`collector_events.py:214,273`); the response reports `unresolved=<n>`.

- [ ] Tests (red): `test_hma_unschedulable_node_without_code_opens_warning_finding` (asgi client; a WARNING `NodeHealthFinding` and the counter); `test_hma_node_with_a_parsed_xid_does_not_double_count_as_unresolved`.
- [ ] Run, expect FAIL; implement; run `tests/hma/` + the route tests, expect PASS.

---

## Coordinator-owned (not dispatched)

- Wave boundaries: after each wave, `make test-parallel PYTEST_XDIST_WORKERS=16`, then `make check-static`; fix ratchet fallout by resuming the owning implementer.
- Docs from every report's 文档需要: `docs/环境变量参考.md` rows (new env vars from Tasks 7, 8, 21), runbook lines for the WARN/degraded marker (Task 17), route inventory for `/v1/attempts/coverage` (Task 8), and `COLLECTORS.md` if a collector's deployment contract changed (exporter `-c`, deleted manifest).
- `channel_registry.py` / `collector_registry.py` edits requested by Task 8 or Task 9.
- Memory update: `dataplane-review-20260908.md` progress line per wave.
