# Regional Acceptance Runner Review

Date: 2026-09-12. Original review: 184 cases and four additions. The subsequent
coverage-hardening work adds BOOT-026, HA-011 and NOTIFY-008, bringing the current
catalog at that review stage to 191 regional cases, including nine retired cases.

Integration identity notice: main independently published BOOT-024 through
BOOT-029 for administrator lifecycle cases. The unpublished WIP table audits
formerly called BOOT-024/025 are now BOOT-030/031, and WIP full uninstall
BOOT-026 is now BOOT-032. The merged catalog has 197 regional cases. Historical
headings, review counts and verification records below keep their original WIP
identities; they neither describe main's same-numbered cases nor validate the
merged source. Current runner/procedure bindings come from the case catalog.

This is a code-review record, not a LIVE acceptance report. No AWS, Kubernetes,
GPU, workload or deployment action was executed by this review. Historical
findings and reviewer handoffs below describe the implementation at review time.
Current Disposition records later integration where applicable. The authoritative
case catalog remains testcases/fault-scenarios.yaml.

The verification numbers below are preserved results of the original review.
They are not a fresh full-suite measurement of the later coverage-hardening
changes. The [scenario matrix](scenario-coverage.md) keeps design, implementation,
local verification and unmeasured LIVE evidence separate.

## Verification

### Current Alignment Contracts

The subsequent alignment review compares the catalog, procedure, runner oracle
and current runtime separately. Old coverage percentages and historical PASS
records below do not establish that these contracts were executed live.

| Area | Current Contract |
|---|---|
| Terminal quarantine | Phase/index/operation-bound successful receipts discharge only their node's hold. Proven spare recovery may recover the incident without releasing the original node; a recovered sibling or partial completion index cannot authorize takeover. |
| Reset evidence | One command/ledger attempt and one successful physical invocation are distinct from bounded NVIDIA busy refusals. Additional attempts require structured consistent success/refusal accounting; unresolved or partial execution cannot pass. |
| Reset cleanup | `run_destr001_gpu_reset` observes product compensation and cleans its own probe/sampler. It does not directly restore quiesce or clear the product's reset claim. |
| Collector census | COLLECT-019 requires present, finite, nonnegative scoped samples on each observed replica; missing census is not zero. |
| GPU identity finding | COLLECT-020 checks the matching marker's critical severity and UUID, and structured metric/event identities for negative inventory findings. Incident prose and a diagnostic workflow are insufficient. |
| Kernel evidence clocks | COLLECT-018 applies the existing allowance once, to an exact marker-bound raw line only. Incident/workflow windows are not widened. |
| Training continuity | COLLECT-017 requires every original Pod to advance valid NCCL collective steps during loss and after restoration, with unchanged node/attempt identities and a bound incident. Stable UIDs alone cannot pass. |
| Passive completion | COLLECT-021 waits for the source decision/plan/workflow, any required successful containment, the restart reservation, and a fresh replacement Observation matching every new Pod UID. Final node validation follows legitimate passive containment. |
| BOOT database isolation | The procedure streams `boot_guard_isolation.py` into the installed component Python, using the same projected-credential and database-identity checks as the runner. Passing a renamed URL alone cannot isolate a Store. |
| Recovery uncertainty | CPU restore and runner cleanup share `recovery_safety`, including legacy and nested unknown outcomes. DESTR-014 retains its operator hold after SQL cancellation or a later terminal status. |
| FM delivery | Bounded producer receipts prove source handoff and progress within an observed journal window. Store deduplication alone cannot prove no replay; observed truncation advances durable cursor generation. |
| Persistent thermal signals | Producer and ingestion share device-derived warning/critical semantics; a persistent severity/action change is delivered without requiring an intervening healthy sample. |
| Administrator lifecycle | Fresh operation journals and saved EKS/HyperPod identity mappings bind resume. Maintenance DSN reads require a successful current projected-file read; runtime pool fallback is a separate contract. |
| Rotation | AUTH-016 uses the production administrator transaction and every credential consumer. Same-release Node key delivery is operational activation, never a substitute for AUTH-015's independent signed witness. |
| HA and capacity | Role-specific metrics, foreground/replay receipts and pooled plus LISTEN budgets replace unlabelled or row-count assumptions. Queued commands retain their original lease deadline and must renew before adapter entry. |
| Local and physical proof | PREEMPT local contracts require complete source-bound receipts. DESTR-015 measures independently traced reset process intervals, not whole-branch duration or literal silicon overlap. |

`testcases/scenario-requirements.yaml` binds the associated new negative controls
without removing prior obligations or the unimplemented external-writer atomic
ownership requirement. Local assertions, deployment proofs and physical/live
evidence remain separate.

### Historical Verification

```json
{
  "coverage": {
    "administrator_state_binding_file_percent": 100,
    "metric": "covered lines plus covered branches over measurable lines plus branches",
    "module_floors": "PASS",
    "perf_dependencies_after_percent": 53.521412624315,
    "production_contractual_scope_percent": 85.92151018380527,
    "production_floor_percent": 78,
    "regional_runners_after_percent": 67.90237241850146,
    "runner_and_tools_after_percent": 68.37451569859975,
    "runner_and_tools_before_percent": 55.19118194099105,
    "runner_and_tools_gain_percentage_points": 13.1833337576087,
    "scope_note": "Before/after comparisons include scripts/e2e/regional and tools. scripts/perf is measured separately after changes. The 78% production gate measures src/gpu_fault, src/gpu_fault_release and deploy/control-plane/tools.",
    "tools_after_percent": 76.0309991696651
  },
  "execution_limits": [
    "No live, staging, AWS, Kubernetes, GPU, workload or deployment actions were executed.",
    "Local test counts are overlapping verification groups and must not be summed as one unique suite.",
    "Unsafe or incomplete live proof variants remain explicitly non-PASS; deferred coverage is listed separately.",
    "No commit, staging or push was performed; changes remain in the requested worktree."
  ],
  "ordinary_full_suite": {
    "artifact_opt_in_skips": 4,
    "failed": 0,
    "passed": 14779,
    "seconds": 295.4,
    "skip_resolution": "All artifact checks were separately executed successfully.",
    "workers": 16
  },
  "other_checks": {
    "artifact_tests_passed": 6,
    "ci_tooling_tests_passed": 82,
    "documentation_tests_passed": 265,
    "final_static_and_documentation_repeat": "PASS"
  },
  "postgres_full_suite": {
    "execution": "serial, owned loopback PostgreSQL 16",
    "failed": 0,
    "initial_non_postgres_variant_skips": 3,
    "passed": 1533,
    "seconds": 987.19,
    "skip_resolution": "PostgreSQL-only xmin/depth assertions now collect only PostgreSQL parameters; the two assertions were rerun successfully. Applicable collection is 1533 tests across 87 files. General cross-backend contracts remain unchanged."
  },
  "review": {
    "active_cases": 179,
    "added_cases": 4,
    "original_cases": 184,
    "retired_cases": 9,
    "total_cases": 188
  },
  "runner_suite": {
    "failed": 0,
    "passed": 5988,
    "postgres_tests_deferred_to_serial_suite": 44,
    "seconds": 181.16,
    "workers": 16
  },
  "status": "LOCAL_VERIFICATION_PASSED",
  "supplemental_tests": {
    "coverage_note": "The initial coverage command failed a per-file module floor. Supplemental coverage was appended against unchanged production source, and the module-floor gate then passed.",
    "failed": 0,
    "new_administrator_binding_tests": 17,
    "passed": 19,
    "production_source_changed_after_full_suite": false,
    "rechecked_postgres_assertions": 2,
    "skipped": 0
  }
}
```

Before: 4035 passed, 16 skipped; scripts/e2e/regional + tools line/branch
coverage 55.19118194%. This is not repository-wide coverage and not the 78% gate.

## Subsequent Cases

These reviewed implementations have local regression coverage and remain `manual`
and `NOT_RUN`. Registration alone does not close a scenario implementation gap;
neither registration nor local regressions establish deployed execution.

| Case | Distinct Assertion | Runner And Boundaries |
|---|---|---|
| BOOT-026 (historical WIP; now BOOT-032) | Full uninstall, interrupted native teardown, fresh-process resume and protected-site preservation | The current entry is `run_boot032_full_uninstall.py`. It uses a separately provisioned sacrificial site, native confirmation/state binding/locks/cleanup, an available final database snapshot and independent resource readback. It does not reuse or destroy the ongoing acceptance site. |
| HA-011 | Busy processor and spool takeover with a stale callback while the replacement is still executing | `run_ha011_busy_cpu_takeover.py` uses a private namespace, the deployed runtime image and isolated PostgreSQL. Only probe-owned processes are killed; it is not a business-Pod failure or CPU-saturation SLO test. |
| NOTIFY-008 | Provider acceptance before durable commit, independently observed across actual process loss | `run_notify008_commit_ambiguity.py` uses private PostgreSQL and a separate SIMULATED provider. It distinguishes the three commit windows and does not claim external exactly-once delivery, real SNS/SES mail or Aurora acceptance. |

The new tests exposed additional failure boundaries: opaque spool claim fences,
known create-ACK UIDs retained before fallible readback, global IAM identities
across Regions, compatible shared administrator modules in the control-plane
wheel, and bounded private HTTP proof transport. Each is reviewed with its
corresponding source owner; focused results are not added to the historical full
suite above.

## Duplicate Decisions

- Restore HA-005: HA-009 deliberately proves zero rollout and is not a rollout-continuity superset.
- Keep AUTH-010, NOTIFY-002, DESTR-011 and PREEMPT-013/023/034 retired only with their surviving assertions retained.
- AUTH-012 is a changed rotation contract, not evidence that the retired outage case ran.
- BOOT-006 is obsolete; DESTR-004 remains prohibited.
- Shared helpers, input types or action names alone do not make two cases duplicates.
- Existing PostgreSQL pool rotation tests already cover long-lived pool reconnect; do not add a duplicate case merely because the review proposal missed them.

## Safety Limits

- New and affected LIVE cases remain NOT_RUN; 43 manual and three live-command PASS bindings were invalidated after guard/verdict/cleanup changes. NOTIFY-006's changed local selection also requires a fresh binding.
- DESTR-014 and three expiring DESTR-008 variants require independent cancellation/fencing before they can safely be enabled.
- AUTH-015 cannot claim complete installation-time custody or deployed signature coverage from point-in-time scans.
- Software API/kmsg/private-log inputs are not physical hardware faults.
- PostgreSQL and runner-scope measurements must remain distinct from Aurora LIVE evidence.
- Optional DESTR-012/C remains NOT_RUN without a physically isolated negative-cluster fixture.

## Deferred Coverage

- Post-STOP late physical sibling injection, live provider-submit ownership drift and runner loss during an expiring shortage need a deterministic, independently fenced injection/recovery contract. They were identified but no runnable hardware case or PASS evidence was invented.
- The original busy processor/spool takeover gap is addressed by HA-011's isolated runtime implementation and local process regressions. Business-Pod failover and CPU-saturation SLOs remain outside that proof.
- Notification provider acceptance followed by a crash before durable commit remains ambiguous without provider idempotency or reconciliation. Existing replay/delivery-state tests do not establish unconditional exactly-once delivery across that window.
- Nested mixed-identity ingestion and deployed command/result-query node-signature probes were proposed as additional boundaries. Existing local authorization/signature tests are not represented as fresh deployed proof.
- The four registered additions address deployed state-table wiring, asymmetric cluster recovery and private FM cursor failure/recovery; other proposals are not silently counted as implemented cases.

## Shared Fixes

- Plan schema 3 binds arguments, kubeconfig contents, actual preflight, source including untracked inputs, and site scope.
- Formal scheduling cannot advance on a local proxy or analysis PASS; reports reject incomplete/duplicate/mismatched results.
- Host probes use durable ownership receipts and UID-fenced cleanup; failed or ambiguous cleanup cannot become PASS.
- CPU/Executor environment windows persist intent before conditional updates and verify full convergence before closing.
- Pod probes use the correct installed component interpreter; standalone launch resolves this checkout.
- Command supervision loss is durable and prevents unverified retries; reports invalidate old PASS on a new attempt.

## Per-Case Analysis

### GF-REGIONAL-AUTH-001

缺少集群 header 返回 401

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: A valid cluster token without the cluster header must receive 401 with the exact required-header detail. Claim is the representative endpoint; the specification additionally names the protected route sweep.
- Runner: scripts/e2e/regional/audit_auth_boundary.py
- Entry Points: audit_auth_boundary.main; audit_auth_boundary.run_matrix; audit_auth_boundary.case_documents
- Assertions: matrix_errors compares AUTH-001 status to 401; it does not compare its detail.; case_documents writes a per-case PASS when that present entry has no status error; release identity is optional without CPU kubeconfig.; post performs verified TLS and propagates transport failures rather than treating them as 401.
- Safety And Cleanup: Negative claim advertises ACCEPTANCE_PROBE_OWNER and has no intended mutation.; The matrix directly executes without the guarded case plan/predecessor workflow; an exception before document creation leaves no structured FAIL.
- Necessity: Keep this credential-present/header-absent case: it distinguishes missing routing identity from absent bearer credentials and from an unregistered identity.
- Overlap: case_ids: GF-REGIONAL-AUTH-002; GF-REGIONAL-AUTH-014; classification: complementary; rationale: AUTH-014 removes all credentials, whereas this case preserves a valid bearer and removes only the cluster header; the authentication precedence is different.
- Tests: existing: tests/regional/test_regional_auth_boundary_acceptance.py::test_auth001_a_protected_route_without_the_cluster_header_answers_401; tests/regional/test_identity_multicluster_contracts.py::test_audit_writes_one_verdict_per_case_and_needs_the_store_for_negatives; missing_behavioral: Wrong 401 detail must fail the actual matrix evaluator.; Transport/JSON failure must preserve failed-case evidence and stop later requests.; Evidence with missing release/site binding must not satisfy formal execution.
- Findings At Review: The live verdict accepts unrelated 401 responses that the application-level unit test rejects.; No valid-token/header-absent sweep of the other protected routes.; Parent must align the multi-case matrix with formal order and evidence identity; current matrix jumps past AUTH-007.
- Reviewer Handoff: FIXED: matrix_errors and case_documents require the complete case entry set, exact integer 401 and exact required-header body. Behavioral tests remove each entry and substitute wrong denials. Full guarded per-case origin/release/order integration of the standalone matrix remains a parent proposal.

### GF-REGIONAL-AUTH-002

缺少或格式错误的 Bearer token 返回 401 与 403 的精确分界

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: Absent Authorization and Basic must produce 401 bearer-required; empty Bearer must produce 403 authentication-failed. The 401/403 boundary is explicitly operationally significant.
- Runner: scripts/e2e/regional/audit_auth_boundary.py
- Entry Points: audit_auth_boundary.run_matrix; audit_auth_boundary.expected_statuses; audit_auth_boundary.case_documents
- Assertions: Three payloads are issued with a valid cluster header.; matrix_errors compares the three statuses but not their details.; case_documents only collects errors for entries present in results, so a partial three-entry case can PASS.
- Safety And Cleanup: Requests use the non-leasing probe owner; no Secret is modified.; Basic and empty Bearer are explicit authorization strings, separate from the normal token constructor.
- Necessity: Keep all three variants within one case: they exercise parsing of the credential scheme independently from validation of a syntactically present credential.
- Overlap: case_ids: GF-REGIONAL-AUTH-001; GF-REGIONAL-AUTH-004; classification: complementary; rationale: AUTH-001 removes identity and AUTH-004 uses non-empty wrong credentials; neither covers the empty-Bearer parsing boundary or Basic scheme.
- Tests: existing: tests/regional/test_regional_auth_boundary_acceptance.py::test_auth002_absent_malformed_and_empty_bearer_stay_distinguishable; missing_behavioral: Parameterize deletion of each required result entry and require FAIL.; Use fake HTTP responses to verify exact headers and exact 401/403 details.; Reject null/string status values without crashing or accepting partial evidence.
- Findings At Review: A returned 401/403 from an unrelated authorization layer is accepted.; The evidence validator does not require all three variants, despite expected_statuses knowing their names.
- Reviewer Handoff: FIXED: all three credential syntax variants are mandatory, with exact 401/403 and denial bodies; missing, non-object and malformed-status replies fail. Covered by parameterized behavioral matrix tests.

### GF-REGIONAL-AUTH-003

未注册集群 ID 返回 403

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: A real token paired with an unregistered cluster ID must receive 403 and the exact unregistered-cluster detail.
- Runner: scripts/e2e/regional/audit_auth_boundary.py
- Entry Points: audit_auth_boundary.run_matrix; audit_auth_boundary.matrix_errors
- Assertions: run_matrix combines token A with the fixed cluster-not-registered header.; Only status 403 is required; unknown registration is not distinguished from a wrong token.; Current factory.authenticate_regional_cluster separately emits unregistered and authentication-failed details.
- Safety And Cleanup: No resource is created or modified by the intended request.; No preliminary proof establishes that the fixed probe cluster ID is absent.
- Necessity: Keep as the registration-existence branch. It tests that possession of a valid token cannot authorize an arbitrary routing identity.
- Overlap: case_ids: GF-REGIONAL-AUTH-004; GF-REGIONAL-AUTH-008; classification: complementary; rationale: AUTH-004 targets an existing registration and AUTH-008 crosses two existing registrations; absence is a distinct authorization branch.
- Tests: existing: tests/regional/test_regional_control_plane.py::test_regional_api_authenticates_cluster_and_rejects_spoofing; missing_behavioral: The matrix must reject authentication-failed detail when unregistered detail is required.; A populated fixed probe registration must prevent execution or use a verified absent run-scoped ID.; Missing case identity fields must prevent formal PASS.
- Findings At Review: A configured registration accidentally named cluster-not-registered can make the case pass for the wrong reason.; The runner's expected response is weaker than both the specification and current runtime response.
- Reviewer Handoff: FIXED: an existing probe registration or a generic wrong-token denial cannot substitute for the exact unregistered-cluster response. Wrong-denial and missing-entry tests cover the live evaluator; no registry mutation or new identity is created.

### GF-REGIONAL-AUTH-004

错误 token 返回 403 且不泄露差异

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: Entirely wrong and one-character-near tokens must both produce the identical generic 403 denial, without length/prefix information; runtime comparison must remain timing-safe.
- Runner: scripts/e2e/regional/audit_auth_boundary.py
- Entry Points: audit_auth_boundary.run_matrix; audit_auth_boundary.matrix_errors; audit_auth_boundary.redact_body
- Assertions: Both present replies must have status 403 and detail regional cluster authentication failed.; The complete bodies are not compared, so additional discriminatory fields can differ.; RegionalClusterRegistration._matched_slot uses compare_digest for all accepted slots; this is static runtime confirmation, not a timing experiment.
- Safety And Cleanup: Tokens are read from files into memory, and the near token is constructed in memory.; Missing/empty token input is not validated before token_a[-1]; only lease_token is redacted from arbitrary response bodies.
- Necessity: Keep the near-miss and all-wrong injections together: equivalence is the security assertion, unlike a generic wrong-token denial.
- Overlap: case_ids: GF-REGIONAL-AUTH-002; GF-REGIONAL-AUTH-003; classification: complementary; rationale: Those cases test parsing or registration existence, not whether response differences disclose how close a token is.
- Tests: existing: tests/regional/test_regional_auth_boundary_acceptance.py::test_auth004_a_near_miss_token_is_denied_exactly_like_a_wrong_one; tests/regional/test_regional_acceptance_fixtures.py::test_auth_boundary_fixture_validates_precise_results; missing_behavioral: Require the full denial body contract and reject differing extra fields.; Reject a missing zero/near result even when the other succeeds.; Empty/short token files and adversarial echoed credentials must fail without disclosure.
- Findings At Review: Full-body non-disclosure is asserted by the application proxy test but not by the live evaluator.; A partial result set can satisfy the per-case evidence builder.
- Reviewer Handoff: FIXED: both wrong and near-token responses must equal the entire generic denial body, including absence of extra discriminator fields. Non-JSON responses are hashed, and malformed/partial replies fail. Runtime timing-safe comparison is unchanged.

### GF-REGIONAL-AUTH-005

header 与 payload 的 cluster_id 不一致返回 403

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: A-authenticated workload observation carrying B's cluster ID must fail with the exact payload-binding 403 and leave B without the injected record.
- Runner: scripts/e2e/regional/audit_auth_boundary.py
- Entry Points: audit_auth_boundary.run_matrix; audit_auth_boundary.STORE_NEGATIVE_PROBE; audit_auth_boundary.store_negative_errors
- Assertions: The live request uses only cluster_id=B in the workload-observations body.; 403 and unchanged aggregate B observation count are checked when --cpu-kubeconfig is supplied.; No store snapshot makes this case FAIL, but empty/malformed snapshot dictionaries can compare equal and pass.
- Safety And Cleanup: This is an intended middleware denial, not a valid observation injection.; The store probe only reads counts; it neither cleans up an unexpected accepted record nor records a run-specific observation identity.
- Necessity: Keep the workload-observation boundary: false ownership here can authorize recovery against another cluster even without a collector event.
- Overlap: case_ids: GF-REGIONAL-AUTH-009; GF-REGIONAL-ISO-001; classification: complementary; rationale: AUTH-009 samples event ingestion and ISO-001 uses valid local identities; neither specifically rejects forged ownership submitted through workload-observations.
- Tests: existing: tests/regional/test_regional_auth_boundary_acceptance.py::test_auth005_a_payload_for_another_cluster_is_403_and_writes_nothing; tests/regional/test_identity_multicluster_contracts.py::test_audit_store_negative_errors_catch_new_records_and_leased_commands; missing_behavioral: Exercise the exact workload-observations route, not only terminal/collector-health routes used by the named test.; Same-count overwritten observation and malformed/missing B snapshots must fail.; Assert exact binding detail and preserve evidence on transport failure.
- Findings At Review: Aggregate count equality cannot prove an existing B observation was not overwritten.; The named application test does not submit to this case's workload-observations route.; No exact binding-detail check in the runner; no run-scoped injected identity.
- Reviewer Handoff: FIXED: exact payload-binding denial and explicit typed Store measurements are required. Observation fingerprints detect same-count updates, not only insertions; absent or invalid snapshots fail. Concurrent legitimate target changes conservatively fail this no-write proof.

### GF-REGIONAL-AUTH-006

嵌套 heartbeat 中的 cluster_id 同样被校验

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: A credential with heartbeat.cluster_id=B must be denied by cluster binding before signature validation and must not create or change B's agent.
- Runner: scripts/e2e/regional/audit_auth_boundary.py
- Entry Points: audit_auth_boundary.run_matrix; audit_auth_boundary.matrix_errors; audit_auth_boundary.store_negative_errors
- Assertions: The nested heartbeat deliberately has an invalid signature and a fixed probe-node identity.; 403 is checked, but the payload-binding detail that establishes ordering is not.; B agent generations are compared as node_id:generation strings.
- Safety And Cleanup: No signed action or heartbeat is authorized.; No-change evidence ignores lifecycle, endpoint, and other agent fields that can change without changing generation.
- Necessity: Keep this nested identity injection because top-level-only binding can pass AUTH-005 and still permit fleet spoofing.
- Overlap: case_ids: GF-REGIONAL-AUTH-005; GF-REGIONAL-ISO-003; classification: complementary; rationale: AUTH-005 is top-level ownership and ISO-003 is local client rejection; this is server middleware precedence over nested signed-heartbeat handling.
- Tests: existing: tests/regional/test_regional_auth_boundary_acceptance.py::test_auth006_a_nested_heartbeat_cluster_id_is_checked_before_the_signature; missing_behavioral: Wrong 403 reason and signature-driven denial must fail the live evaluator.; Changes to an existing B agent with unchanged generation must fail no-write proof.; Missing B agent snapshot must not count as an unchanged empty fleet.
- Findings At Review: The exact response-order guarantee exists only in the runtime proxy test.; The aggregate agent representation does not establish complete no-write behavior.; Nested lists and clusterId/cluster aliases need separate behavior coverage, not a claimed duplicate retirement.
- Reviewer Handoff: FIXED: exact payload-binding denial distinguishes rejection before signature validation. Agent-record fingerprints supplement generation counts and reject unmeasured or same-generation updates. No signed heartbeat or action is manufactured.

### GF-REGIONAL-AUTH-007

被停用的集群立即失去认证能力

Current catalog: live-service-action; manual; NOT_RUN.

- Purpose: Disable only the isolated secondary registration, prove B=403 while A=200, restore B=200, and measure isolation propagation. This is a maintenance-window service action.
- Runner: scripts/e2e/regional/run_identity_acceptance.py; scripts/e2e/regional/identity_acceptance_auth.py; scripts/e2e/regional/identity_acceptance_common.py
- Entry Points: run_identity_acceptance.main; identity_acceptance_auth.run_auth007; IdentitySite.write_registry; IdentitySite.rollout_control
- Assertions: run_auth007 checks B disabled 403, A 200, restored B 200, full registry equality, and cleanup_errors.; Durable registry mode publishes a revision and waits for missing_member_ids to become empty; the measured delay is POST-to-ACK.; Claim probes use the acceptance-only execution owner, avoiding real command leases.
- Safety And Cleanup: run_cleanup_steps attempts registry restore, propagation, and recovery claim even after an earlier cleanup error.; write_registry fetches a fresh head generation for an old whole-registry snapshot, so concurrent unrelated registration changes can be overwritten.; There is no runtime proof that the selected secondary is an isolated test target; node/runtime identity and plan drift need parent-level coordination.
- Necessity: Keep this disable/re-enable transition independently of token rotation: it proves emergency isolation, not credential overlap.
- Overlap: case_ids: GF-REGIONAL-AUTH-016; GF-REGIONAL-ISO-006; classification: complementary; rationale: AUTH-016 changes credential slots and ISO-006 blocks transport; neither exercises enabled-state authorization propagation.
- Tests: existing: tests/regional/test_identity_multicluster_contracts.py::test_auth007_failure_runs_every_restore_and_raises_the_original; tests/regional/test_identity_multicluster_contracts.py::test_write_registry_measures_post_to_last_ack_and_rollout_is_noop; missing_behavioral: Full success and wrong-denial behavior through run_auth007.; Concurrent registry generation/content drift must stop restore without overwriting peer entries.; Lost revision ACK, failed recovery claim, already-disabled secondary, and restored registry mismatch.
- Findings At Review: Specification/catalog still describe Secret editing and CPU rollouts; durable mode no longer does that.; Fresh-generation CAS does not protect the original snapshot from concurrent change.; The 403 reason is not checked, and positive claim command_count is not required to be zero.
- Reviewer Handoff: FIXED: distinct targets, initially enabled secondary, durable registry, exact denial and zero-command positive claims are required. Registry writes bind reviewed content plus generation; restore accepts only the original or this runner's last published content and refuses concurrent drift. Behavioral CAS, restore and failure-path tests pass.

### GF-REGIONAL-AUTH-008

集群 A 的凭证不能领取集群 B 的动作

Current catalog: live-non-destructive; manual; NOT_RUN.

- Purpose: With a real pending B command, A claims and a spoofed executor label must remain A-scoped; crossed tokens must be 403 and B's command must remain pending.
- Runner: scripts/e2e/regional/audit_auth_boundary.py
- Entry Points: audit_auth_boundary.claim_payload; audit_auth_boundary.run_matrix; audit_auth_boundary.probe_claims_leased_nothing; audit_auth_boundary.store_negative_errors
- Assertions: Normal/spoofed A claims must be 200 with no foreign commands; crossed header/token pairs must be 403.; A separate check requires both probe-owner claims to return no commands.; The before/after probe compares any pre-existing PENDING command, but does not require even one B command.
- Safety And Cleanup: The non-production execution owner avoids leasing live work, which is a necessary safety property.; There is no controlled probe-owner backlog or release/retirement path if an unexpected claim returns a command.
- Necessity: Keep the pending-command isolation test, but the current empty-owner probe is not sufficient evidence for it.
- Overlap: case_ids: GF-REGIONAL-ISO-002; GF-REGIONAL-E2E-002; classification: complementary; rationale: ISO-002 rejects a command locally after dispatch and E2E-002 observes successful local recovery; neither directly tests claim selection against an available foreign candidate and spoofed executor label.
- Tests: existing: tests/regional/test_identity_multicluster_contracts.py::test_control_plane_answers_probe_owner_claim_with_200_and_no_commands; tests/regional/test_identity_multicluster_contracts.py::test_audit_store_negative_errors_catch_new_records_and_leased_commands; missing_behavioral: Seed a B-only command for the dedicated probe owner and show an omitted cluster filter makes the case fail.; Require a nonempty B-specific pending baseline and exact unchanged command identity.; Cover all four claim combinations, malformed commands body, and safe cleanup after an unexpected lease.
- Findings At Review: The advertised owner has no commands in either cluster, so removing cluster filtering can still yield two empty successful claims.; No B-pending precondition; an idle site can receive PASS.; Global command movement can cause false failures unrelated to the probe.; Fake executor_id is a fixed string, not the actual B executor identity.
- Reviewer Handoff: FAIL_CLOSED_PRECONDITION: AUTH-008 now requires a B PENDING command for the non-adapter acceptance owner and verifies that exact backlog did not move. Normal production-owner backlog or an idle B no longer satisfies the case. A controlled, owned seed/retire fixture is still required for formal execution; parent proposal IDENTITY-AUTH008-CANDIDATE defines it.
- Current Disposition: Run-owned B backlog with probe-only owner, crossed-token negatives and a positive B claim; owned lease-fenced retirement replaces a potentially empty backlog.

### GF-REGIONAL-AUTH-009

集群 A 的凭证不能提交集群 B 的事件

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: A credentials must not write B through every named collector/event/terminal/progress endpoint; exact 403s and unchanged B collector counters/evidence are required.
- Runner: scripts/e2e/regional/audit_auth_boundary.py
- Entry Points: audit_auth_boundary.run_matrix; audit_auth_boundary.expected_statuses; audit_auth_boundary.STORE_NEGATIVE_PROBE; audit_auth_boundary.case_documents
- Assertions: The current runner issues ten hard-coded paths, preserving the user's HMA-to-gpu-inventory replacement.; Expected AUTH-009 names are derived from results actually returned, so missing routes are invisible to completeness validation.; store_negative_errors checks observation count and agent generations, not collector sample_count or raw evidence.
- Safety And Cleanup: Payloads only carry foreign cluster_id and are intended to be denied before model validation.; No event marker is attached; accepted asynchronous work or pre-existing-row changes cannot be attributed reliably.
- Necessity: Keep the ingress-wide binding sweep, with completeness derived from current route/schema contracts and case-specific no-write evidence.
- Overlap: case_ids: GF-REGIONAL-AUTH-005; GF-REGIONAL-AUTH-006; GF-REGIONAL-AUTH-014; classification: partial-overlap-not-superset; rationale: All exercise authorization, but AUTH-014 is anonymous, AUTH-005 is ownership ingestion, and AUTH-006 tests nested signature precedence. None replaces authenticated foreign event payloads.
- Tests: existing: tests/regional/test_regional_auth_boundary_acceptance.py::test_auth009_every_cluster_token_write_route_rejects_a_foreign_payload; tests/regional/test_identity_multicluster_contracts.py::test_audit_writes_one_verdict_per_case_and_needs_the_store_for_negatives; missing_behavioral: Missing one required route must fail; one AUTH-009 result must not represent the whole case.; Collector counter/evidence changes with unchanged observation/agent snapshots must fail.; Distributed nested events and clusterId/cluster aliases; exact error detail and asynchronous no-write proof.
- Findings At Review: The current named unit fixture includes only one AUTH-009 route and still accepts its case document.; Catalog text still says eleven endpoints while current spec/runner list ten.; The asserted store invariant is different from the case's collector/evidence invariant.; The fixed list does not follow new channel/route additions automatically.
- Reviewer Handoff: FIXED: ten AUTH009_PATHS form a fixed required set, every reply must be the exact payload-binding denial, and B collector counters plus raw-evidence count/fingerprint must remain unchanged. Preserved the user's HMA-to-gpu-inventory change. Parent must reconcile stale eleven-route catalog text.

### GF-REGIONAL-AUTH-010

已并入：受保护路径矩阵完整性

Current catalog: read-only-signal-replay; manual; SUPERSEDED.

- Purpose: Already SUPERSEDED by AUTH-014 and excluded by do_not_run. Former obligations were the cluster-token anonymous matrix, no public/metrics write buckets, and absence of execution token from GPU Secrets.
- Runner: scripts/e2e/regional/run_identity_acceptance.py; scripts/e2e/regional/identity_acceptance_auth.py
- Entry Points: run_identity_acceptance.CASE_IDS; identity_acceptance_auth.run_auth010; regional_case_contract.formal_predecessor
- Assertions: The legacy helper still computes PASS/FAIL and status=superseded, and the parser still lists this ID.; The normal main entry calls predecessor_path, whose formal_predecessor rejects DO_NOT_RUN before dispatch.; run_auth010 explicitly rejects public/metrics/None write buckets; run_auth014 lacks that complete check.
- Safety And Cleanup: Do not execute or re-enable this retired ID.; Its legacy helper remains callable outside main; central tooling must not interpret its legacy PASS as new formal execution.
- Necessity: The existing retirement is sensible only after restoring the complete survivor assertion; repair AUTH-014 rather than inventing a second route-sweep case.
- Overlap: case_ids: GF-REGIONAL-AUTH-014; classification: existing-retirement-with-unproven-implementation-superset; rationale: The injection and intended scope are included by AUTH-014, but the live survivor can accept a metrics-bucket write route returning 403. That loses AUTH-010's bucket-safety assertion.
- Tests: existing: tests/regional/test_regional_auth_boundary_acceptance.py::test_auth010_every_route_answers_the_denial_its_bucket_declares; tests/regional/test_identity_multicluster_contracts.py::test_auth010_records_that_it_is_superseded; missing_behavioral: Behaviorally verify retired main is refused before any environment reads or requests.; AUTH-014 must fail a public/metrics write declaration even when the anonymous HTTP status matches that bucket.; Remove obsolete helper/choice only with parent-owned central entry-test coordination.
- Findings At Review: The current retirement unit test only searches source strings.; The CLI still advertises a case that cannot pass the formal predecessor resolver.; No new retirement proposal is justified by this incomplete survivor.
- Reviewer Handoff: RETIRED_PRESERVED: no new execution or reactivation. AUTH-014 now retains AUTH-010's unsafe public/metrics write-bucket assertion and execution-token sweep, closing the lost survivor obligation. Parent may remove the stale CLI choice; formal DO_NOT_RUN remains authoritative.

### GF-REGIONAL-AUTH-011

健康与指标端点不需要集群凭证但也不泄露集群清单

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: From a non-loopback data-plane peer, healthz remains public, metrics is forbidden, and neither anonymous nor cluster-token clients may read the regional cluster inventory or leaked labels.
- Runner: scripts/e2e/regional/audit_auth_boundary.py
- Entry Points: audit_auth_boundary.run_matrix; audit_auth_boundary.get; audit_auth_boundary.matrix_errors
- Assertions: Checks 200 health and 403 metrics/inventory responses.; Does not inspect denied response or health bodies for other cluster IDs/labels.; get assumes every successful/failed response is JSON, including metrics, so an exposed Prometheus response raises before structured verdict creation.
- Safety And Cleanup: Read-only HTTP requests require TLS validation.; The standalone matrix is executed from its calling host, not guaranteed to be an executor Pod or the intended non-loopback path.
- Necessity: Keep as an information-disclosure boundary: denying writes does not prove metrics and inventory cannot reveal the regional fleet.
- Overlap: case_ids: GF-REGIONAL-AUTH-014; classification: partial-overlap-not-superset; rationale: AUTH-014's anonymous route sweep overlaps status checks, but this case also sends a valid cluster token to operator inventory and forbids leaked cluster labels.
- Tests: existing: tests/regional/test_regional_auth_boundary_acceptance.py::test_auth011_health_and_metrics_do_not_expose_the_regional_inventory; missing_behavioral: A 403 body containing B identity must fail.; Exposed text/plain metrics must produce a structured FAIL, not abort without evidence.; Require all four entries and bind the data-plane origin.
- Findings At Review: Status-only success misses information disclosure in denied bodies.; The live HTTP decoder cannot record the most important metrics failure mode.; Origin, release, and formal-order binding need the parent-supported guarded matrix integration.
- Reviewer Handoff: FIXED: all four entries are required; metrics and both inventory denials must be exact execution-token denial bodies, so leaked extra fields cannot PASS. Exposed non-JSON metrics are hashed and judged FAIL without dumping their text. Network-origin binding remains part of the parent matrix integration proposal.

### GF-REGIONAL-AUTH-012

已废弃：集群 token 轮换的失败窗口测量

Current catalog: live-service-action; manual; SUPERSEDED.

- Purpose: Already SUPERSEDED by AUTH-016. Its former single-token premise expected a rotation outage and measured its duration; overlap-window support intentionally changed the acceptance semantics.
- Runner: scripts/e2e/regional/run_identity_acceptance.py; scripts/e2e/regional/identity_acceptance_auth.py
- Entry Points: run_identity_acceptance.CASE_IDS; identity_acceptance_auth.run_auth016; RegionalClusterRegistration.accepted_token_slots
- Assertions: No AUTH-012 handler or parser choice exists.; Catalog and order preserve SUPERSEDED/do_not_run and point to AUTH-016.; Current token-slot implementation accepts retiring credentials only before their deadline.
- Safety And Cleanup: No old single-token cutover may be executed or recorded as a fresh PASS.; Its replacement performs a separate authorized rotation and restoration; it is not evidence that this retired case ran.
- Necessity: Preserve this historical ID and its retired outcome. Measuring an expected failure window is incompatible with the current zero-interruption goal.
- Overlap: case_ids: GF-REGIONAL-AUTH-016; classification: semantic-replacement-not-duplicate; rationale: The token topology and expected outcome changed, so this is a versioned semantic replacement, not proof of an injection/assertion superset suitable for additional retirements.
- Tests: existing: tests/regional/test_regional_token_rotation.py::test_overlap_window_accepts_both_tokens; tests/regional/test_complete_acceptance_entries.py::test_reusable_entries_follow_the_formal_predecessor_chain; missing_behavioral: Central routing must continue to reject AUTH-012 before side effects.; Replacement tests must cover expiry and removal, not manufacture evidence for the retired ID.
- Findings At Review: No new runner is needed for AUTH-012.; AUTH-016's current gap-tolerant sampling cannot yet provide the promised replacement proof; that defect belongs to AUTH-016.
- Reviewer Handoff: RETIRED_PRESERVED: no handler added and no obsolete outage expected-result restored. AUTH-016 is the overlap-window semantic replacement, not fabricated execution of AUTH-012.

### GF-REGIONAL-AUTH-013

私有 CA 与主机名校验不可绕过

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: Prove scoped private-CA trust, correct SAN, rejection of empty CA and wrong hostname, more than 30 days of validity, and an armed host certificate-expiry timer.
- Runner: scripts/e2e/regional/identity_acceptance_auth.py; scripts/e2e/regional/probes/auth013_certificate_probe.py; scripts/e2e/regional/run_identity_acceptance.py
- Entry Points: identity_acceptance_auth.TLS_BOUNDARY_PROBE; identity_acceptance_auth.run_auth013; identity_acceptance_auth.certificate_alert_checks; auth013_certificate_probe.scan
- Assertions: Requires a successful normal TLS handshake, exact hostname SAN, CA-file presence, and no global CA override variables.; The wrong-hostname probe treats any SSLError or OSError as a correct rejection, including timeout/reset unrelated to certificate verification.; The empty-CA path checks SSLContext construction, not an actual rejected remote handshake.; No host probe makes the timer check false, but a cleanup_error-only residual object does not make the verdict fail.
- Safety And Cleanup: The Pod probe creates /tmp/auth013-empty-ca.pem without removing it.; HostProbeFixture creates a privileged host Pod and writes a host script, despite the read-only-signal-replay classification.; run_auth013 catches cleanup exceptions but excludes cleanup_error from its residual-failure condition.
- Necessity: Keep as a transport identity test. HTTP authorization cannot compensate for trusting the wrong TLS peer.
- Overlap: case_ids: GF-REGIONAL-AUTH-014; classification: complementary; rationale: AUTH-014's network reachability denial does not establish CA or hostname verification, and a TCP failure is specifically not a hostname-verification result.
- Tests: existing: tests/regional/test_identity_multicluster_contracts.py::test_certificate_alert_checks_read_the_node_timer_not_the_site_file; tests/regional/test_identity_multicluster_contracts.py (historical pre-fix symbol: test_tls_probe_accepts_any_ssl_or_os_error_and_records_default_handshake); tests/regional/test_regional_auth_boundary_acceptance.py::test_auth013_the_executor_trusts_the_private_ca_without_replacing_aws_trust; missing_behavioral: Invoke the probe with fake socket/SSL boundaries and distinguish verification mismatch from generic OSError/timeouts.; Run the handler with probe.cleanup raising or returning residuals and require FAIL.; Exercise host timer disabled/missing/unreadable cases and wrong returned host/node binding.
- Findings At Review: Current source-text test explicitly enshrines the overly broad hostname denial.; A probe cleanup failure can coexist with PASS.; The validity threshold comes from site config rather than an explicit case minimum; parent must reconcile risk/approval text for the privileged host probe.
- Reviewer Handoff: FIXED: only OpenSSL hostname-mismatch verification code 62 proves hostname rejection; transport/reset/unrelated TLS errors do not. Empty-CA scratch files are scoped and removed, validity is at least 30 days, cleanup errors/residuals fail. HostProbeSettings receives required case_dir/host-probes; handler and socket-boundary behavioral tests cover these branches.

### GF-REGIONAL-AUTH-014

互联网可达 NLB 上的端点鉴权全量清单

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: Enumerate the deployed route/method/bucket surface; deny anonymous protected access, restrict cluster-token high-risk routes, prove outside-allowlist network denial, and forbid execution-token presence on the data plane.
- Runner: scripts/e2e/regional/identity_acceptance_auth.py; scripts/e2e/regional/run_identity_acceptance.py
- Entry Points: identity_acceptance_auth.run_auth014; identity_acceptance_auth.ROUTE_INVENTORY_PROBE; identity_acceptance_auth.outside_probe; identity_acceptance_auth.data_plane_execution_token_hits
- Assertions: Checks anonymous statuses against buckets and records the public documentation surface.; Checks only declared buckets for the three high-risk routes, not their authenticated cluster-token behavior.; Outside evidence requires target_host, time, and connection_blocked=true, but no proof of probe origin or actual transport result.; No security groups yields no broad rules and therefore passes the world-open checks.; The inventory omits /livez and uses a path-keyed bucket dictionary, not method-specific declarations.
- Safety And Cleanup: Secret values stay in memory; output reports names/key matches and digests.; Token sweep is limited to the selected namespace/cluster and literal Pod env, not all enabled GPU clusters.; Only read operations are intended; anonymous write success is treated as failure but unexpected effects have no cleanup path.
- Necessity: Keep the survivor full-surface audit, but strengthen its proof rather than declaring overlap based on route counts.
- Overlap: case_ids: GF-REGIONAL-AUTH-010; GF-REGIONAL-AUTH-011; GF-REGIONAL-BLAST-004; classification: partial-overlap-with-retired-predecessor; rationale: The anonymous route matrix subsumes AUTH-010's injection only when unsafe write buckets and the Secret check are retained. BLAST-004 additionally covers all-cluster token uniqueness and broader configuration surfaces.
- Tests: existing: tests/regional/test_identity_multicluster_contracts.py::test_high_risk_routes_are_judged_by_bucket_not_counted; tests/regional/test_identity_multicluster_contracts.py::test_outside_probe_requires_host_age_and_records_digest; tests/regional/test_identity_multicluster_contracts.py::test_execution_token_hits_compare_values_not_only_key_names; tests/regional/test_regional_auth_boundary_acceptance.py::test_auth014_the_internet_facing_surface_is_exactly_its_declared_buckets; missing_behavioral: Missing route/method, duplicate bucket collision, metrics-bucket write, and no-SG must fail.; Exercise actual cluster-token fleet foreign-query and operator-write negatives through the runner.; Reject outside evidence with DNS/TLS failure, unproven origin, wrong release/run, or merely asserted connection_blocked.; Cross-namespace/peer-cluster execution-token hits must be detected.
- Findings At Review: The runtime proxy test covers cluster-token high-risk behavior that the live runner never invokes.; Missing full response-set and explicit safe-write-bucket checks leave AUTH-010 retirement incomplete.; External evidence is self-asserted and not bound to release/run/approved source network.; Parent should supply a structured outside-network probe runner and central evidence contract.
- Reviewer Handoff: FIXED: deployed inventory includes livez and per-endpoint buckets; every route/method result must be present; unsafe write buckets fail independently of HTTP status; missing NLB security groups fail. Execution-token scanning covers all namespaces of the selected GPU cluster. Parent still owns attested outside-network probe/origin proof and fleet-wide formal scope.

### GF-REGIONAL-AUTH-015

fleet master 永不进入 GPU 节点

Current catalog: non-destructive; manual; NOT_RUN.

- Purpose: Fleet master must never reach either GPU node during installation; node-scoped v2 keys must reject cross-node heartbeat/command/result signatures, and rotating A must preserve B.
- Runner: scripts/e2e/regional/identity_acceptance_auth.py; scripts/e2e/regional/probes/auth015_node_probe.py; scripts/e2e/regional/run_identity_acceptance.py
- Entry Points: identity_acceptance_auth.run_auth015; identity_acceptance_auth.master_reference_scan; identity_acceptance_auth.auth015_focused_tests; auth015_node_probe.scan
- Assertions: Checks zero master digest matches, key-A change, unchanged peer keys, B generation/ACTIVE/advanced heartbeat, restored CPU/GPU key digests, and probe removal.; HostProbeFixture.execute chroots to /host, but candidate_files/process_environments still use /host/tmp, /host/etc, /host/var, /host/proc. Missing roots and zero scanned values are accepted.; No installer resource is recorded as not_evaluated but excluded from verdict checks.; Cross-node proof runs two local fleet heartbeat tests; no deployed command or result-query negative is exercised.
- Safety And Cleanup: The master is supplied from a restricted deploy-host file and only its digest enters the probe.; Current provision_node_action_keys.provision changes Secrets only; no installer or node-key activation is driven by this runner.; restore_secret uses apply of the old whole Secret without UID/resourceVersion or concurrent-change ownership protection.; A failed focused test or a before-scan master hit does not stop the subsequent key rotation.
- Necessity: Keep this key-isolation gate, but it currently cannot prove the installation-time and deployed node-key claims it names.
- Overlap: case_ids: GF-REGIONAL-BLAST-004; classification: complementary; rationale: BLAST-004 separates execution/cluster tokens between clusters; AUTH-015 concerns a stronger fleet-master capability and lateral movement between nodes within one cluster.
- Tests: existing: tests/regional/test_identity_multicluster_contracts.py::test_master_reference_scan_says_when_no_installer_was_present; tests/regional/test_identity_multicluster_contracts.py::test_auth015_creates_probes_in_parallel_and_checks_the_node_set; tests/fleet/test_fleet.py::test_derived_node_key_cannot_sign_for_another_node; tests/fleet/test_fleet.py::test_node_specific_key_can_rotate_without_changing_peer; missing_behavioral: Model the actual chroot path layout and require readable roots/nonempty scans; missing/unreadable scan surfaces must fail.; No installer, projected/envFrom master reference, failed focused tests, or a pre-scan hit must stop before rotation.; Run fake provisioning/agent snapshots through the handler and verify A activation, peer continuity, cleanup failures, and Secret UID/CAS drift.; Cover command and result-query cross-node signatures without physical actions.
- Findings At Review: Confirmed vacuous host scan due to the /host-after-chroot mismatch.; No installer observation is allowed to PASS; reference scan misses projected Secret and envFrom forms.; Tests are local heartbeat-only evidence and do not use live node keys.; The supplied master fingerprint is not proven to identify the actual trusted CPU fleet master.; Secret rollback lacks identity-fenced ownership and node-runtime convergence proof.
- Reviewer Handoff: FIXED_FAIL_CLOSED: host scan uses post-chroot paths and refuses empty, missing, unreadable or oversized scan surfaces. Projected/envFrom master references are covered; absent/unsafe installer evidence, failed signature tests or a dirty pre-scan stop before rotation. Distinct nodes and CPU/GPU key agreement are required. Restores use UID/resourceVersion/content tests; unknown ACK-loss changes are deferred, never overwritten. Case-private HostProbe state is supplied. Installation-time coverage and deployed cross-node command/result signatures remain explicit parent case proposals, not claims from local heartbeat tests.
- Current Disposition (2026-09-13): Implemented prospective independently signed custody in the real provisioning path, explicit administrator enrollment/preparation/resume, and a deployed Agent witness of rotated-key acceptance plus retired/sibling command/result rejection. The runner does not rotate Secrets, install nodes, or reconstruct historical custody from snapshots. Initial activation is only a prerequisite, not a full-case PASS. Missing/incomplete provenance and unknown create ACKs remain fail-closed. Focused fake-I/O and real local crypto/Agent tests pass; LIVE remains NOT_RUN. Protocol and trust limits: [Node Key Custody Evidence](node-key-custody-evidence.md).

### GF-REGIONAL-AUTH-016

集群 token 重叠窗口的零中断轮换

Current catalog: live-service-action; manual; NOT_RUN.

- Purpose: Zero-interruption overlapping-token rotation: old and new credentials work during overlap, new works after cutover, old becomes 403 when retired, commands survive, and credentials are restored.
- Runner: scripts/e2e/regional/identity_acceptance_auth.py; scripts/e2e/regional/identity_acceptance_common.py; scripts/e2e/regional/run_identity_acceptance.py
- Entry Points: identity_acceptance_auth.run_auth016; identity_acceptance_auth.auth016_result; identity_acceptance_auth.direct_claim; IdentitySite.write_registry
- Assertions: Samples one current token every request duration plus two seconds across baseline/overlap/cutover.; auth016_result filters out all non-integer outcomes, allowing probe-unavailable/transport errors alongside one 200 per phase.; New token is sampled only once during overlap, and the completed phase's new-token samples are not judged.; Remote-command baseline scans the whole region and accepts SUCCEEDED as survival; no selected-cluster binding is retained.
- Safety And Cleanup: Uses a non-leasing owner; sensitive tokens are sent through script stdin rather than argv.; Stops/joins the sampler then attempts every credential restore; restoration still proceeds even if the sampler outlives the join.; Whole-registry restore can overwrite concurrent membership/rotation changes; token Secret patch has no UID/CAS precondition.; Baseline failures do not prevent proceeding into mutation, and the caller discards the returned maintenance deadline.
- Necessity: Keep as the current rotation acceptance, but zero interruption must mean complete measured availability, not successful HTTP samples after removing all gaps.
- Overlap: case_ids: GF-REGIONAL-AUTH-012; GF-REGIONAL-AUTH-007; classification: semantic-replacement-and-complementary; rationale: It replaces the obsolete single-token-outage premise of AUTH-012 and separately exercises overlapping credential slots rather than AUTH-007's registration disable bit.
- Tests: existing: tests/regional/test_regional_token_rotation.py::test_overlap_window_accepts_both_tokens; tests/regional/test_identity_multicluster_contracts.py::test_direct_claim_uses_the_probe_owner_and_a_single_attempt; tests/regional/test_identity_multicluster_contracts.py::test_commands_not_misterminated_needs_a_baseline_and_accepts_succeeded; tests/regional/test_identity_registry_durable.py::test_plaintext_tokens_become_digests_and_refresh_updated_at; missing_behavioral: A phase containing one 200 plus timeout/unavailable must not PASS.; Require bounded cadence/duration and both tokens during overlap; wrong new-token status after retirement fails.; Use target-scoped commands and prove an unrelated cluster baseline cannot satisfy the case.; Fake-clock full rotation with failed baseline, failed revision ACK, lost sampler, restore failures, and registry/Secret drift.
- Findings At Review: Current zero-interruption verdict ignores the interruption samples it records.; Only a minimum of one HTTP sample per phase is enforced; no complete two-token overlap observation.; An idle target can use another cluster's open command as its survival evidence.; Node collectors/watchers are not moved to the new token before retiring old credentials; their outage is outside the executor-only sampling.
- Reviewer Handoff: FIXED: non-HTTP gaps remain failures; both old/new credentials are sampled in overlap and completed-phase new-token samples are judged. Failed baseline/overlap/cutover prevents advancement. Command survival baseline is target-cluster scoped. Registry content/generation and token Secret UID/content are fenced; no restore runs under a live sampler, and unsafe restore sequencing is deferred. Fake-thread lifecycle tests exercise all stages and failures. Executor sampling does not prove migration of every node collector; parent rotation-coverage proposal remains.

### GF-REGIONAL-BLAST-001

控制面 EKS 从不被 GPU 故障 workflow taint、cordon 或 drain

Current catalog: live-non-destructive; manual; NOT_RUN.

- Purpose: Reuse this run's approved data-plane mutation and prove CPU nodes unchanged, no GPU-fault Jobs or eviction events, with prior E2E operations showing the exercised blast-radius boundary.
- Runner: scripts/e2e/regional/run_blast_acceptance.py; scripts/e2e/regional/blast_acceptance_base.py; scripts/e2e/regional/blast_acceptance_cases_1.py; scripts/e2e/regional/run_workload_acceptance.py
- Entry Points: BlastCasesOne.blast_001; BlastCasesOne.node_security_snapshot; BlastCasesOne.managed_field_writes; run_workload_acceptance.run_e2e001
- Assertions: Compares taints, cordon, gpu-fault labels/annotations; checks new relevant managedFields, current labelled Jobs created after start, and current eviction events after start.; Requires operation names MARK_UNSCHEDULABLE, STOP_WORKLOADS, RESTART_WORKLOAD anywhere in declared or executed workflow lists, without successful-step/command correlation.; The normal RESTART_APP branch of WorkflowBuilder.catalog_operations contains FREEZE_EVIDENCE, STOP_WORKLOADS, RESTART_WORKLOAD, not MARK_UNSCHEDULABLE; E2E-001 can legitimately succeed while the BLAST consumer refuses.; The execution card/baseline/workflow files are not bound by digest, run, release, cluster, or successful E2E verdict; end time is recorded but not used.
- Safety And Cleanup: Read-only audit, no extra fault or reset is performed.; Empty/malformed node lists can normalize to equal empty dictionaries; Node UIDs are not compared.; Event retention or transient changes reverted before observation are not proven absent by the current read.
- Necessity: Keep the runtime blast-radius audit, but require evidence from an actually exercised operation set instead of fabricating a cordon step or weakening the stated scope.
- Overlap: case_ids: GF-REGIONAL-E2E-001; GF-REGIONAL-BLAST-002; classification: shared-evidence-distinct-scope; rationale: E2E-001 already checks CPU before/after state, while BLAST-001 additionally consumes managedFields and labelled Job/event evidence. BLAST-002 checks capability, not what changed during a fault.
- Tests: existing: tests/regional/test_blast_acceptance_runner.py::test_blast001_reads_the_window_and_the_workflow_steps_e2e001_hands_over; tests/regional/test_blast_acceptance_runner.py::test_managed_field_writes_ignore_entries_already_in_the_baseline; missing_behavioral: Producer-to-consumer test with the real RESTART_APP compiled operation shape, not a handcrafted extra MARK step.; Reject wrong release/run/cluster, future/naive/reversed windows, changed baseline bytes, failed-only declarations, and empty/replaced node sets.; Exercise new Job/eviction/managedField changes and post-window unrelated changes separately.
- Findings At Review: The existing handoff test fabricates the very MARK operation the producer is not required to emit.; External --e2e-dir and --trusted-cpu-baseline can be unrelated to the admitted predecessor.; Time-window end and UID continuity are not enforced.; Parent must resolve whether this case consumes a containment-capable prior case or narrows its documented operation coverage; no unapproved extra mutation.
- Reviewer Handoff: FIXED_FAIL_CLOSED: consumes only successful, cleaned, release/cluster-bound E2E handoffs with baseline/workflow digests and a valid completed observation window. CPU Node UIDs and nonempty inventories are required; only SUCCEEDED step executions prove operations. The documented MARK_UNSCHEDULABLE requirement is not fabricated for RESTART_APP, so a normal E2E-001 alone can still fail full BLAST-001. Parent must resolve the containment-source contract rather than weaken the assertion silently.
- Current Disposition: Moved after DESTR-001 to consume real successful containment, retaining E2E-001 workload evidence without another mutation.

### GF-REGIONAL-BLAST-002

控制面不持有任何 GPU 集群的写入凭证

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: CPU processes must have no GPU kubeconfig or Kubernetes write capability and no HyperPod mutations; SageMaker is read-only and SES sending is scoped to the verified sender.
- Runner: scripts/e2e/regional/blast_acceptance_base.py; scripts/e2e/regional/blast_acceptance_cases_1.py
- Entry Points: BlastCasesOne.blast_002; BlastCasesOne.auth_can_i; BlastCasesOne.cpu_control_plane_role_arn; BlastCasesOne.iam_role_policies; BlastCasesOne.sagemaker_read_only
- Assertions: Checks home .kube absence, KUBECONFIG unset, and an all-namespace 4x3 write matrix for the hard-coded CPU ServiceAccount.; Finds exactly one Pod Identity association and inspects its IAM policies, but does not prove the selected CPU Pod uses that ServiceAccount/role.; Forbids four named HyperPod actions and any Allow/NotAction; literal sagemaker: patterns must begin Describe/List.; SES Resource may contain identity/* and any nonempty Condition, without equality to the configured sender or ses:FromAddress.
- Safety And Cleanup: Only read operations and access reviews, but command() has no subprocess timeout.; Cached preflight binds only site pathname, cluster IDs, and age; content/role/context/release changes can be reused.; kubeconfig_binding returns ca_matches=false without rejecting a missing CA, and does not reject insecure-skip-tls-verify.
- Necessity: Keep capability inspection separate from observed no-mutation evidence. It tests whether a runtime bug could bypass the CPU/GPU split.
- Overlap: case_ids: GF-REGIONAL-BLAST-001; GF-REGIONAL-BLAST-003; classification: complementary; rationale: A quiet CPU during one fault does not prove it lacks write credentials, and the executor intentionally has a different allowed capability set.
- Tests: existing: tests/regional/test_regional_least_privilege_acceptance.py::test_blast002_the_control_plane_declares_no_gpu_write_permission; tests/regional/test_blast_acceptance_runner.py::test_sagemaker_read_only_accepts_a_role_with_no_sagemaker_permission; tests/regional/test_blast_acceptance_runner.py::test_preflight_is_reused_within_the_window_for_the_same_site; missing_behavioral: Drive blast_002 with fake live Pods/RBAC/IAM; wrong actual SA/role, identity/* SES resource, unrelated Condition, and service-wildcard mutation patterns must fail.; Namespace-specific write grants must not disappear behind --all-namespaces queries.; Missing CA, insecure TLS, drifted cache contents, malformed association/policy pages, and command timeout must fail closed.
- Findings At Review: Filesystem presence checks do not prove no GPU credential mounted elsewhere.; Role inspection is not bound to the running CPU identity; only Pod Identity is supported although the spec names IRSA too.; SageMaker service wildcards and sender/Condition semantics can evade the custom pattern checks.; All-namespace access review is not a proof of no namespaced RoleBinding permission.
- Reviewer Handoff: FIXED: preflight cache binds site bytes, targets, namespaces, kubeconfig paths and current release; missing/mismatched CA, insecure TLS and missing release identity fail. Ready Pod identity and actual ServiceAccount are checked. Namespaced write permissions supplement all-namespace checks; can-i exit/status pairs are strict. SageMaker wildcard services cannot evade denial and SES requires the exact sender ARN plus ses:FromAddress condition. Focused permission/cache/TLS behavior tests cover positive and negative outcomes.

### GF-REGIONAL-BLAST-003

GPU executor 权限最小化且不含 SES

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: Executor RBAC must match least privilege: exact node/read-only workload ClusterRole, no sensitive-object access or Pod creation, writes only in allowed namespaces, and no SES capability.
- Runner: scripts/e2e/regional/blast_acceptance_base.py; scripts/e2e/regional/blast_acceptance_cases_1.py; scripts/e2e/regional/blast_acceptance_cases_2.py
- Entry Points: BlastCasesTwo.blast_003; blast_acceptance_cases_2.expected_executor_role; BlastCasesTwo.normalized_role_rules
- Assertions: Reads the shipped ClusterRole, checks live normalized rule equality, exact nodes/pods can-i columns, sensitive-resource denials, and IRSA annotation equality.; The queried jobs/pytorchjobs/jobsets matrix columns are never judged; a second binding can grant cluster-wide writes while the named ClusterRole still matches.; Namespace Role/RoleBinding grants and actual running Pod ServiceAccount are not inspected.; Normalization drops resourceNames and nonResourceURLs; IAM audit forbids SES patterns and two provider actions but does not enforce the existing complete executor IAM allowlist/resource scope.
- Safety And Cleanup: Read-only across all site targets; no node patch or permission mutation is attempted.; Shared BLAST preflight cache and CA/timeout gaps apply.; The report correctly states that local nodes/patch cannot be constrained by label selector.
- Necessity: Keep this separate least-privilege gate because it defines the executor's permitted blast radius even when CPU delegation is correct.
- Overlap: case_ids: GF-REGIONAL-ISO-005; GF-REGIONAL-BLAST-002; GF-REGIONAL-NOTIFY-004; classification: partial-overlap-not-superset; rationale: ISO-005 tests runtime namespace guards and NOTIFY-004 mail separation; neither covers actual Kubernetes RBAC and the local executor IAM identity together.
- Tests: existing: tests/regional/test_regional_least_privilege_acceptance.py::test_blast003_the_executor_rbac_and_iam_stay_minimal; tests/regional/test_blast_acceptance_runner.py::test_expected_executor_role_is_parsed_from_the_shipped_manifest; missing_behavioral: Call blast_003 with extra job/CRD write grants, namespace-only Secret permissions, extra bindings, and wrong Pod ServiceAccount.; Reject policies allowing cross-cluster reboot, unrelated services, wildcard service/action spellings, and hidden SES actions.; Reject live role fields omitted by normalization and exercise each target independently.
- Findings At Review: Actual live permission columns are collected but ignored.; The declaration-level proxy test is stronger than the live runner's IAM and namespaced RBAC assertions.; Current-source manifest equality is not a signed deployed-release binding.; A new broad source manifest can redefine expected permissions without enough fixed safety invariants in the runner.
- Reviewer Handoff: FIXED: all jobs/PyTorchJob/JobSet permission columns are judged, complete Ready executor Pods use the audited ServiceAccount, and namespace-specific sensitive/foreign write grants are audited. Named-role normalization refuses unsupported scopes rather than discarding them. IAM permits only the known executor SageMaker actions and the actual local HyperPod ARN, with no NotAction/NotResource. Behavioral tests exercise permissions and identity; no manifest/baseline changes.

### GF-REGIONAL-BLAST-004

执行 token 与集群 token 的分离

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: Execution token must be absent from every GPU Secret, ConfigMap, and Pod environment; each physical GPU cluster must have a different valid cluster token and no peer token leakage.
- Runner: scripts/e2e/regional/blast_acceptance_cases_2.py; scripts/e2e/regional/blast_acceptance_base.py
- Entry Points: BlastCasesTwo.blast_004; BlastCasesTwo.scan_gpu_objects; BlastCasesTwo.execution_token_digest
- Assertions: Scans all namespaces for exact execution-token value digests and key/object/reference names, including ConfigMap binaryData and init/ephemeral containers.; Reads runtime env names from one Ready executor, compares each canonical connection Secret hash to execution hash, and compares canonical cluster hashes pairwise.; Raw-byte hashes are used while cluster authentication strips bearer whitespace; whitespace-distinct copies of one token appear unique.; No minimum/nonempty cluster-token length or secondary-copy search is performed; one cluster makes uniqueness vacuously true with only a limitation.
- Safety And Cleanup: Only digests/names/locations enter the intended evidence, never decoded Secret values.; Value checks match entire fields only; embedded config values, image-level env under unrelated names, and peer cluster-token copies outside the canonical Secret are not covered.; All targets come from the site rather than a fresh complete enabled-registry inventory.
- Necessity: Keep both credential-class separation and per-cluster uniqueness, with explicit insufficient-topology status for the cross-cluster half.
- Overlap: case_ids: GF-REGIONAL-AUTH-014; GF-REGIONAL-AUTH-015; classification: partial-overlap-not-superset; rationale: AUTH-014 only sweeps one selected namespace/cluster as part of the route audit; AUTH-015 addresses the fleet master and per-node keys. Neither proves pairwise cluster-token uniqueness or all-namespace execution-token absence.
- Tests: existing: tests/regional/test_regional_least_privilege_acceptance.py::test_blast004_the_execution_token_stays_on_the_control_plane; missing_behavioral: Run scan_gpu_objects/blast_004 against fake multi-cluster Secret/ConfigMap/Pod objects with adversarial key names, binary values, whitespace, empty tokens, and peer copies.; Runtime actual values must be digest-checked without disclosure; missing runtime streams must fail.; Single-cluster scope, missing enabled cluster, duplicate canonical Secret, malformed base64, and evidence redaction.
- Findings At Review: There is no direct live-runner behavior test for this scanner.; Different raw token bytes need not mean different accepted credentials.; Pairwise canonical Secret hashes do not establish that A's token is absent from all B objects.; One-cluster PASS overstates the two-cluster part of the acceptance unless the parent contract explicitly separates partial proof.
- Reviewer Handoff: FIXED: token comparisons normalize accepted whitespace, reject short/missing/duplicate canonical credentials, scan peer-token copies, and verify runtime environment value digests plus actual cluster identity. Missing runtime inventory fails. Fewer than two physical targets cannot PASS pairwise uniqueness. All evidence contains locations/digests, not credential values; focused scanner tests cover copied and normalized credentials.

### GF-REGIONAL-BOOT-001

区域模式缺少集群注册表必须启动失败

Current catalog: non-destructive; manual; NOT_RUN.

- Purpose: Regional startup without GPU_FAULT_REGIONAL_CLUSTERS_JSON must refuse service, never become Ready or answer healthy, and stay outside production Service endpoints. Catalog: manual, live, non-destructive; first startup-guard entry after BOOT-023.
- Runner: scripts/e2e/regional/run_regional_boot_guard_cases.sh; scripts/e2e/regional/boot_guard/derive.sh; scripts/e2e/regional/boot_guard/mutate.py; scripts/e2e/regional/boot_guard/assert.sh
- Entry Points: shell begin_case 1 -> reset_probe -> apply_mutation del -> assert_probe -> endpoint check -> delayed Ready recheck -> pass_case; ControlPlaneSettings.from_mapping rejects the missing registry before ApplicationContext._base_context
- Assertions: assert.sh requires the exact missing-registry log message and Ready != True on the single selected Pod; Runner rejects guard-probe names in gpu-fault-api endpoints; One readinessProbe.periodSeconds later the Pod must still not be Ready; final text marker is VERDICT PASS
- Safety And Cleanup: Every shell invocation initializes gpu_fault_guardprobe before any case; explicit PostgresStore URL is overridden by inherited production GPU_FAULT_STORE_URL_FILE, violating isolation; EXIT cleanup invokes cleanup.sh --drop-database and removes the local derived manifest, but cleanup status is suppressed; No plan/execute/maintenance or BOOT-023 predecessor gate; evidence is cases/<id>.txt, not the canonical JSON consumed downstream
- Necessity: A necessary distinct fail-closed configuration guard. Two readiness samples and one Service check are narrower than the specification's never-healthy and all-production-endpoints claims; the shared isolation defect must be fixed before a live run.
- Overlap: case_ids: GF-REGIONAL-BOOT-002; GF-REGIONAL-BOOT-010; GF-REGIONAL-BOOT-016; classification: complementary; rationale: Missing registry differs from malformed/explicit-empty BOOT-002; BOOT-010 observes valid production settings; BOOT-016 rejects empty first deployment at the administrator boundary.
- Tests: existing: tests/regional/test_regional_boot_guard_acceptance.py::test_boot001_regional_mode_without_a_registry_refuses_to_start; tests/regional/test_regional_boot_guard_runner.py::test_every_case_begins_and_ends_with_one_verdict; assessment: ApplicationContext refusal is behavioral; shell coverage is syntax/source-text assertions and does not exercise credential routing, failed reads, or final cleanup.; planned_behavior: Mocked shell transport: missing-registry log plus API read failure must FAIL; Mock PostgresStore construction and derived Secret mount to prove actual isolated DSN selection before schema initialization; Transient Ready and membership in either production Service must prevent PASS
- Findings At Review: P0: BC-SHARED-001 database isolation; P1: legacy shell lacks formal authorization/evidence binding; P2: no continuous health or all-Service endpoint observation
- Reviewer Handoff: Owned safety/authorization/evidence defects fixed and tested, including real isolated PG16. Parent documentation must use the new guarded shell CLI.

### GF-REGIONAL-BOOT-002

非法注册表必须失败且显式空列表保持健康

Current catalog: non-destructive; manual; NOT_RUN.

- Purpose: Three invalid registry shapes (truncated JSON, object, scalar array item) must fail startup with distinct errors; explicit [] must start and return authenticated GET /v1/regional/clusters = 200 []. Catalog: manual, live, non-destructive.
- Runner: scripts/e2e/regional/run_regional_boot_guard_cases.sh; scripts/e2e/regional/boot_guard/reset.sh; scripts/e2e/regional/boot_guard/assert.sh
- Entry Points: begin_case 2 iterates three bad Secret payloads with reset_probe between each; final [] branch waits through probe_pod_name and executes an authenticated loopback cluster-list request; ControlPlaneSettings.from_mapping validates JSON/list/object shapes; RegionalRegistryRuntime.bootstrap creates the first durable revision
- Assertions: Each malformed branch must match its own exact error through assert_probe; Positive branch requires HTTP 200 and payload == []; pass_case occurs only after all four branches
- Safety And Cleanup: Bad registry is a temporary Secret; reset deletes the Deployment between branches, not the database/durable registry; Inherited production DSN file breaks isolation as in BC-SHARED-001; The [] positive branch seeds a durable empty head reused by later positive BOOT-007/008 probes unless round isolation is added
- Necessity: The malformed-versus-empty distinction is valid and must not be reverted to rejecting legitimate zero-member steady state. Current reuse of one database across successful startup variants undermines later registration/authentication probes.
- Overlap: case_ids: GF-REGIONAL-BOOT-001; GF-REGIONAL-BOOT-016; GF-REGIONAL-BOOT-019; classification: complementary_shared_empty_state; rationale: BOOT-002 checks startup/parser behavior; BOOT-016 checks initial lifecycle admission; BOOT-019 must prove reaching [] through supported removal. None is a substitute for the others.
- Tests: existing: tests/regional/test_regional_boot_guard_acceptance.py::test_boot002_a_malformed_registry_fails_and_an_explicit_empty_one_boots; assessment: Tests the three parser messages and empty in-process Store, but not the shell's authenticated positive request or cross-round durable-head persistence.; planned_behavior: Run mocked four-branch dispatch and require reset before every payload; Exercise [] followed by nonempty registry against a fake durable-head Store to expose stale bootstrap input; Reject positive response 200 with nonempty/malformed payload and preserve FAIL evidence
- Findings At Review: P0: shared DSN-file isolation; P1: reset does not reset durable registry state; positive cases are not independent; P1: read failures may be treated as empty Pod listings by reset.sh
- Reviewer Handoff: Owned isolation, durable-state reset and failed-read handling fixed; no live verdict claimed.

### GF-REGIONAL-BOOT-003

区域模式与单集群变量互斥

Current catalog: non-destructive; manual; NOT_RUN.

- Purpose: Regional mode plus GPU_FAULT_HYPERPOD_CLUSTER naming the CPU HyperPod must fail before local-cluster action targeting is possible. Catalog: manual, live, non-destructive.
- Runner: scripts/e2e/regional/run_regional_boot_guard_cases.sh; scripts/e2e/regional/boot_guard/mutate.py; scripts/e2e/regional/boot_guard/assert.sh
- Entry Points: begin_case 3 -> set GPU_FAULT_HYPERPOD_CLUSTER from CPU_HYPERPOD_CLUSTER -> exact mutual-exclusion error; ControlPlaneSettings.from_mapping rejects the value before constructing Store/adapters
- Assertions: Log must contain 'regional mode uses the cluster registry; GPU_FAULT_HYPERPOD_CLUSTER must be unset'; Single probe Pod must not be Ready at assertion time; No explicit action-call spy or post-error healthy check in the shell branch
- Safety And Cleanup: Only a disposable Deployment should be mutated, but unsafe shared database initialization occurs before this early settings guard; EXIT cleanup and evidence shortcomings are shared with BOOT-001; CPU_HYPERPOD_CLUSTER is required but not bound to an approved site/Region identity by this runner
- Necessity: Distinct and important CPU/GPU isolation guard. Specification correctly locates refusal in settings, but nearby prose saying BOOT-004/005/006 later persist registry is stale and even references retired BOOT-006.
- Overlap: case_ids: GF-REGIONAL-BOOT-004; GF-REGIONAL-BOOT-005; GF-REGIONAL-BOOT-010; classification: complementary; rationale: BOOT-003 rejects an identity source; BOOT-004/005 reject local mutation adapters; BOOT-010 verifies actual valid deployment values.
- Tests: existing: tests/regional/test_regional_boot_guard_acceptance.py::test_boot003_regional_mode_and_the_single_cluster_variable_are_exclusive; assessment: Behavioral refusal is covered. No mocked shell case execution or proof that the rejected startup never reaches Store/adapters.; planned_behavior: Spy ApplicationContext._base_context at the invocation boundary and require no call on contradictory identity; Mock shell command failure after the expected log and require final FAIL, not bare log PASS
- Findings At Review: P0: unsafe common initialization is reached even though this particular guard is pre-Store; P1: no formal target/approval binding; P2: stale specification references and limited shell behavior coverage
- Reviewer Handoff: Owned common safety and target/evidence binding fixed; original configuration-refusal semantics retained.

### GF-REGIONAL-BOOT-004

区域模式禁止 in-cluster Kubernetes adapter

Current catalog: non-destructive; manual; NOT_RUN.

- Purpose: A regional CPU control plane must reject GPU_FAULT_ENABLE_KUBERNETES_ADAPTER=true and never use its own Kubernetes service account for GPU mutation. Catalog: manual, live, non-destructive.
- Runner: scripts/e2e/regional/run_regional_boot_guard_cases.sh; scripts/e2e/regional/boot_guard/assert.sh
- Entry Points: begin_case 4 -> apply_mutation set GPU_FAULT_ENABLE_KUBERNETES_ADAPTER true -> assert_probe; ControlPlaneSettings.from_mapping checks regional_mode and kubernetes before ApplicationContext adapter construction
- Assertions: Exact 'regional control plane must not enable the in-cluster KubernetesWorkflowAdapter' log; Not-Ready observation through assert.sh; Branch writes final text PASS only after assert_probe pipeline succeeds
- Safety And Cleanup: No legitimate call to a KubernetesWorkflowAdapter is needed; disposable probe only; Shared setup still risks production Store and cleanup/read failures remain fail-open; No independent taint/cordon audit; this is configuration rejection, not the BLAST live node audit
- Necessity: Keep as a separate negative adapter guard. It demonstrates startup refusal, not all CPU blast-radius guarantees.
- Overlap: case_ids: GF-REGIONAL-BOOT-003; GF-REGIONAL-BOOT-005; GF-REGIONAL-BOOT-010; GF-REGIONAL-BLAST-001; classification: complementary_layers; rationale: Shares the shell fixture with other guards but tests a different forbidden adapter; BLAST-001 inspects actual CPU node isolation outcomes.
- Tests: existing: tests/regional/test_regional_control_plane.py::test_regional_context_rejects_local_kubernetes_adapter; tests/regional/test_regional_boot_guard_runner.py::test_case_specific_additions_are_present; assessment: Runtime predicate covered; source-text shell checks cannot detect a broken refusal/read/cleanup sequence.; planned_behavior: Patch adapter constructors to fail on invocation and prove settings refusal precedes construction; Drive the shell case with exact log plus Ready=True and assert nonzero/final FAIL
- Findings At Review: P0: BC-SHARED-001; P1: no authoritative JSON evidence or approved-window guard; P2: no branch-level shell behavioral tests
- Reviewer Handoff: Owned shell safety/provenance fixes complete; no Kubernetes adapter or live node action executed.

### GF-REGIONAL-BOOT-005

区域模式禁止中央 HyperPod mutation adapter

Current catalog: non-destructive; manual; NOT_RUN.

- Purpose: Regional CPU startup must reject the central HyperPod mutation adapter; CloudTrail must show no CPU-role node mutation, with immediate lookup explicitly provisional and final confirmation deferred to DESTR-013. Catalog: manual, live, non-destructive.
- Runner: scripts/e2e/regional/run_regional_boot_guard_cases.sh; scripts/e2e/regional/boot_guard/assert.sh
- Entry Points: begin_case 5 -> enable central HyperPod adapter -> exact settings refusal -> aws cloudtrail lookup-events -> optional Username substring filter -> length == 0; ControlPlaneSettings.from_mapping rejects regional_mode and hyperpod
- Assertions: Required refusal log and not Ready; CloudTrail enumerates RebootClusterNodes, BatchDeleteClusterNodes, BatchReplaceClusterNodes, UpdateClusterSoftware; Records cloudtrail_provisional=true and requires filtered event array empty
- Safety And Cleanup: CloudTrail query is read-only, but this task has not executed it; Optional CONTROL_PLANE_ROLE_NAME filter matches Username substrings rather than structured principal ARN/sessionIssuer; Common probe DB initialization and cleanup remain unsafe; final absence proof belongs to the later approved DESTR-013 window
- Necessity: The startup guard is sound, while CloudTrail is supporting eventual-consistency evidence. The enumerated mutation names omit BatchRebootClusterNodes, which the actual provider reboot path uses; absence cannot be claimed from this list.
- Overlap: case_ids: GF-REGIONAL-BOOT-010; GF-REGIONAL-DESTR-013; classification: partial_overlap_complementary_timing; rationale: BOOT-005 proves an invalid CPU configuration is refused; DESTR-013's complete delayed audit supersedes only the provisional CloudTrail observation, not the startup refusal.
- Tests: existing: tests/regional/test_regional_boot_guard_acceptance.py::test_boot005_regional_mode_refuses_the_central_hyperpod_adapter; tests/regional/test_regional_boot_guard_runner.py::test_case_specific_additions_are_present; assessment: No behavior test supplies CloudTrail events or principal identities.; planned_behavior: Mock BatchRebootClusterNodes from the CPU principal and require FAIL; Use structured CloudTrail principal records including assumed-role Username mismatch; CloudTrail permission/malformed response must stop without converting to an empty event list
- Findings At Review: P0: shared database isolation; P1: BatchRebootClusterNodes omitted from mutation audit; P1: role attribution may hide events or reject unrelated actors; P2: final DESTR-013 audit linkage is prose-only
- Reviewer Handoff: Owned mutation-name/filter and common isolation defects fixed. Delayed provider audit remains a separate live obligation.

### GF-REGIONAL-BOOT-006

【已作废】区域模式禁止 in-cluster quick diagnostics

Current catalog: non-destructive; manual; NOT_RUN.

- Purpose: Retired without replacement because the in-cluster quick-diagnostics path and GPU_FAULT_ENABLE_QUICK_DIAGNOSTICS were deleted. Catalog remains manual/NOT_RUN; execution order lists DO_NOT_RUN.
- Runner: scripts/e2e/regional/run_regional_boot_guard_cases.sh; testcases/regional-execution-order.yaml
- Entry Points: No begin_case 6 and no active handler; Shell resume explicitly skips case 6; regional_case_contract.formal_predecessor rejects a DO_NOT_RUN case
- Assertions: No runtime PASS should be produced; The shell currently prints aggregate 'BOOT-001..010 PASS' despite only running nine active cases, which is misleading for the retired slot
- Safety And Cleanup: No feature resurrection, flag injection, live execution, or synthetic replacement case proposed; No cleanup is needed for a retired case
- Necessity: Retirement is correct and must be retained. Some BOOT-003/010 prose still mentions the old flag/case and should be cleaned centrally without changing the retirement decision.
- Overlap: case_ids: ; classification: retired_no_replacement; rationale: The underlying capability was removed, not superseded by another acceptance case.
- Tests: existing: tests/regional/test_regional_boot_guard_runner.py::test_every_case_begins_and_ends_with_one_verdict; assessment: Current test checks source text for active case numbers; a behavioral dispatch/predecessor refusal test is stronger.; planned_behavior: Mocked case selection must reject BOOT-006 before any transport call; Aggregate completion output must enumerate only active executed IDs
- Findings At Review: P2: aggregate shell PASS label includes a retired ID; P2: stale cross-references in BOOT-003/010
- Reviewer Handoff: Retirement preserved; no PASS or active handler for006.

### GF-REGIONAL-BOOT-007

集群 token 长度下界精确为 32 字符

Current catalog: non-destructive; manual; NOT_RUN.

- Purpose: Cluster token length floor is exactly 32 characters: 31 fails startup, 32 yields a Ready Pod and HTTP 200 health. Catalog: manual, live, non-destructive.
- Runner: scripts/e2e/regional/run_regional_boot_guard_cases.sh; scripts/e2e/regional/boot_guard/registry.py; scripts/e2e/regional/boot_guard/reset.sh
- Entry Points: begin_case 7 -> probe_registry 31 -> refusal -> probe_registry 32 -> probe_pod_name -> loopback /healthz; ControlPlaneSettings registry validation calls cluster_token_sha256; registry.py uses a synthetic ID and TEST-NET-1 Agent CIDR
- Assertions: 31-character branch requires 'cluster token must contain at least 32 characters'; 32-character branch waits on Deployment Available, prints Pod readiness, and asserts /healthz status 200; Resume from 7 requires prior text final PASS markers for 1..5, skipping retired 6
- Safety And Cleanup: Synthetic token only, not a production credential; Inherited production DSN file is unsafe, and a prior [] durable head can make the 32-character probe test the wrong registration state; No explicit per-Pod Ready wait after Deployment Available as requested by the spec; final printed readiness is not asserted
- Necessity: Exact boundary is worth keeping. A healthy empty registry is insufficient evidence that the 32-character credential actually registered and authenticates; a targeted authenticated positive request would make the test meaningful after durable registry was introduced.
- Overlap: case_ids: GF-REGIONAL-BOOT-002; GF-REGIONAL-BOOT-008; classification: complementary_shared_registry_fixture; rationale: Length validation differs from JSON/schema validation and disabled-member authentication, although all depend on fresh probe registry state.
- Tests: existing: tests/regional/test_regional_boot_guard_acceptance.py::test_boot007_a_cluster_token_shorter_than_32_characters_stops_startup; assessment: Behaviorally verifies 31 refusal and 32 health plus Store registration; shell does not assert that final registration fact.; planned_behavior: Mock final Pod not Ready despite Deployment availability; Assert positive token authentication and expected synthetic registration after an earlier [] round; Fail a stale/other-cluster text resume record
- Findings At Review: P0: BC-SHARED-001; P1: durable head survives reset and can invalidate the intended token branch; P2: Deployment availability substituted for explicit Pod readiness
- Reviewer Handoff: Owned shared isolation/reset/resume defects fixed; token-boundary behavior preserved.

### GF-REGIONAL-BOOT-008

注册表字段完整性与 enabled=false 语义

Current catalog: non-destructive; manual; NOT_RUN.

- Purpose: Missing eks_cluster_arn and unknown alowed_namespaces must be rejected; enabled=false must still boot but reject its known cluster token with 403 authentication-failed. Catalog: manual, live, non-destructive.
- Runner: scripts/e2e/regional/run_regional_boot_guard_cases.sh; scripts/e2e/regional/boot_guard/registry.py; scripts/e2e/regional/boot_guard/assert.sh
- Entry Points: begin_case 8 -> three probe_registry variants -> disabled /v1/regional/executors/claim on loopback; RegionalClusterRegistration StrictModel validates required/extra fields; authenticates rejects disabled registration
- Assertions: Missing-field probe only checks generic RegionalClusterRegistration validation text, not eks_cluster_arn/Field required; Misspelling probe checks Extra inputs are not permitted; Disabled probe requires 403 and authentication-failed substring, after Deployment Available
- Safety And Cleanup: Claim is sent only to the disposable probe with predictable synthetic token; Shared database/file DSN and persisted durable registry can redirect or invalidate disabled-member semantics; No asserted positive enabled control in the shell to distinguish disabled membership from a wrong token/unregistered ID
- Necessity: Schema strictness and disabled authentication are related but distinct subcases; both remain needed. Generic validation text can pass for an unrelated prerequisite error, and authentication must be tied to the freshly configured disabled member.
- Overlap: case_ids: GF-REGIONAL-BOOT-007; GF-REGIONAL-BOOT-019; GF-REGIONAL-BOOT-021; classification: complementary; rationale: BOOT-007 tests length, BOOT-019 revocation after removal, and BOOT-021 incorrect readiness credentials; disabled-at-startup is neither removal nor malformed readiness.
- Tests: existing: tests/regional/test_regional_boot_guard_acceptance.py::test_boot008_registry_field_integrity_and_disabled_cluster_semantics; assessment: Existing behavior test includes an enabled 200 control and disabled 403; live shell lacks equivalent identity control.; planned_behavior: Reject generic schema error unrelated to eks_cluster_arn; Use fresh enabled/disabled registry controls and require exact response detail; Verify case-8 resume cannot use stale final PASS from another identity
- Findings At Review: P0: shared isolation; P1: durable registry state not reset; P2: nonspecific missing-field verdict and incomplete positive control
- Reviewer Handoff: Owned unsafe setup and stale durable-state paths fixed; no live disabled-member authentication claim.

### GF-REGIONAL-BOOT-009

managed recovery observer 的依赖门禁

Current catalog: non-destructive; manual; NOT_RUN.

- Purpose: Managed recovery observer enabled without Agent registry must fail startup. Unlike BOOT-004/005 it reaches ApplicationContext._managed_observer after Store/context setup. Catalog: manual, live, non-destructive.
- Runner: scripts/e2e/regional/run_regional_boot_guard_cases.sh; scripts/e2e/regional/boot_guard/derive.sh; scripts/e2e/regional/boot_guard/assert.sh
- Entry Points: begin_case 9 only flips GPU_FAULT_ENABLE_AGENT_REGISTRY=false; ApplicationContext.from_environment -> _base_context -> _base_adapters -> _managed_observer checks context.fleet_registry
- Assertions: Requires exact 'HyperPod managed recovery observer requires GPU_FAULT_ENABLE_AGENT_REGISTRY=true' log; Not-Ready at the single assertion point; Does not first assert that inherited ENABLE_HYPERPOD_MANAGED_OBSERVER is true
- Safety And Cleanup: This case must have a genuinely isolated database because registry/context work precedes the target guard; After case 9 the runner explicitly calls cleanup.sh --drop-database, disables the EXIT trap, then begins production gate 10; Cleanup script prints residual Pods/Secrets but does not require their absence and ignores many command failures
- Necessity: Necessary dependency guard, not an adapter authorization test. The single-variable mutation is valid only when the inherited observer-enabled precondition is proven.
- Overlap: case_ids: GF-REGIONAL-BOOT-010; classification: negative_positive_pair; rationale: BOOT-009 proves refusal of the invalid dependency pair; BOOT-010 confirms the live valid pair is present.
- Tests: existing: tests/regional/test_regional_boot_guard_acceptance.py::test_boot009_the_managed_observer_requires_the_agent_registry; assessment: Correct behavioral refusal with explicitly enabled observer; no shell fixture/cleanup transport tests.; planned_behavior: Observer false baseline must be rejected as not exercising this case; Verify explicit and EXIT cleanup both propagate a failed Pod/Secret/database absence check; Mock context construction and prove it never opens the production DSN
- Findings At Review: P0: target guard is after the broken isolation boundary; P1: failed cleanup can still be followed by BOOT-010; P2: observer-enabled prerequisite implicit
- Reviewer Handoff: Owned P0 isolation and cleanup false-success defects fixed.

### GF-REGIONAL-BOOT-010

生产三副本已按区域模式正确启动

Current catalog: non-destructive; manual; NOT_RUN.

- Purpose: All three live API replicas must agree on regional mode, four dangerous switches false, managed observer/Agent registry true, no HyperPod cluster variable, active-active processor, registered-cluster status 200, and placement on three CPU nodes. Catalog: manual, live, non-destructive.
- Runner: scripts/e2e/regional/run_regional_boot_guard_cases.sh
- Entry Points: begin_case 10 after probe cleanup -> compare Deployment generation/readyReplicas baseline -> per-Pod printenv filter -> expect_env -> cluster-status loopback requests -> unique node count
- Assertions: Requires all listed environment lines and equality across every listed API Pod; Requires no GPU_FAULT_HYPERPOD_CLUSTER entry and nonempty Secret registry; Calls collector-status for each Secret cluster ID through one selected Ready API Pod; Requires unique spec.nodeName count == 3, but does not require exactly three Pods or Ready=True on each
- Safety And Cleanup: Read-only production observations after cleanup; no GPU mutation needed; Baseline equality can preserve an already degraded baseline and Pod list is not checked for deletion/current ReplicaSet; Only flat text evidence is emitted, so AUTH-001's canonical JSON predecessor cannot be satisfied directly
- Necessity: Useful positive smoke gate, but its hard-coded three-node topology and API-only role sampling are narrower than the current role-split control plane. Missing/extra/not-Ready replicas need explicit failure, not set cardinality alone.
- Overlap: case_ids: GF-REGIONAL-BOOT-003; GF-REGIONAL-BOOT-004; GF-REGIONAL-BOOT-005; GF-REGIONAL-BOOT-009; GF-REGIONAL-BOOT-018; classification: complementary_positive_baseline; rationale: Negative guards prove invalid inputs fail; this observes valid live settings. BOOT-018 checks content identities rather than these environment invariants.
- Tests: existing: tests/regional/test_regional_control_plane.py::test_regional_context_loads_registry_and_remote_adapter; tests/regional/test_regional_boot_guard_runner.py::test_case_specific_additions_are_present; assessment: Neither executes the production-replica observation logic with missing, duplicate, terminating, or unhealthy Pods.; planned_behavior: Mock four Pods on three nodes and require FAIL; Mock unchanged Deployment generation with one not-Ready/old ReplicaSet Pod and require FAIL; Require collector-status coverage for all intended replicas or explicitly narrow the specification
- Findings At Review: P1: replica count/readiness/UID ownership not proven; P1: noncanonical evidence blocks the next formal family; P2: only one API Pod checks the loaded registry and current role-split coverage is incomplete
- Reviewer Handoff: Owned replica, credential-transport and canonical evidence defects fixed; no live production probe executed.

### GF-REGIONAL-BOOT-011

数据面 executor 清单的 Secret key 完整性

Current catalog: non-destructive; manual; NOT_RUN.

- Purpose: Prove greenfield Executor deployability: required Secret keys are derivable by the manual, optional endpoints may be absent, no database credentials, Pod Ready within 180s, isolated cluster/token/namespace/IRSA, and no production claims. Catalog manual; formal sequence starts here despite its stated BOOT-016 prerequisite.
- Runner: scripts/e2e/regional/run_boot_acceptance.py; scripts/e2e/regional/boot_acceptance_common.py; scripts/e2e/regional/boot_acceptance_runtime.py
- Entry Points: run_boot_acceptance.main -> SiteFixture -> run_boot011; secret_key_contract parses all YAML documents and regex-matches key creation in the manual; run_boot011 reads existing Deployment/Secret/Pods/IRSA and production lease owners; it never performs a fresh deployment
- Assertions: required_secret_keys_present, optional_endpoint_key, no_database_credentials, executor_ready_within_180s; ready_within counts Ready transitions relative to creation; foreign_lease_owners compares both lease_owner and last_lease_owner; dedicated_irsa_effective only compares role ARN to the supplied target and rejects any '*' in serialized trust; isolated_registry_only is only {selected_cluster_id}.isdisjoint(production_ids), not a live registry query
- Safety And Cleanup: Runner observes an already provisioned site; no runner-owned namespace/RBAC/IRSA creation or cleanup; Production Secret values are not written, but isolated Secret JSON is read in memory; Guard environment binds site path and cluster ID but not --production-site or complete site content; evidence_identity_for silently drops identity checks on read errors
- Necessity: Static key checks and existing-Pod inspection are useful but cannot prove the manual's fresh derivation/deployment sequence. Disjoint names alone do not prove endpoint/token/IRSA/namespace separation.
- Overlap: case_ids: GF-REGIONAL-BOOT-012; GF-REGIONAL-BOOT-016; GF-REGIONAL-BOOT-018; classification: partial_overlap_complementary; rationale: Shared Secret/TLS inputs overlap BOOT-012; BOOT-016 performs actual deployment and BOOT-018 checks artifact identity. None currently proves this case's independent credential derivation.
- Tests: existing: tests/regional/test_boot_acceptance_runtime.py::test_ready_within_needs_replicas_and_a_ready_transition_inside_the_limit; tests/regional/test_boot_acceptance_runtime.py::test_foreign_lease_owners_recognise_the_isolated_executor_identity; tests/regional/test_production_safety_config.py::test_boot011_documents_all_required_executor_inputs; assessment: No test calls run_boot011. Helper coverage does not test full isolation assertions or live command failure handling.; planned_behavior: Mock run_boot011 with reused production IRSA, extra live registrations, shared endpoint, and missing expected Secret reference; Reject negative Ready duration, terminating/extra unhealthy Pods, and missing readiness evidence; Reject GPU_FAULT_POSTGRES_POOL_* and DSN-file/volume credentials, not only the current incorrect POSTGRES_POOL_ prefix; Compare required (Secret name, key) pairs, not any matching key string anywhere in the manual
- Findings At Review: P1: no actual greenfield runner path for the specified creation sequence; P1: live registry and dedicated IRSA trust subject are not verified; P1: required keys from every Secret are checked against one Secret only; pool-prefix test misses GPU_FAULT_POSTGRES_POOL_*; P1: BC-SHARED-002 formal prerequisite conflict and identity fail-open
- Reviewer Handoff: Owned known replica/read/error/DSN-prefix defects fixed. Initial greenfield isolation proof breadth is explicitly limited, with fixture creation delegated to016.

### GF-REGIONAL-BOOT-012

executor 的私有 CA 信任链必须显式且局部配置

Current catalog: non-destructive; manual; NOT_RUN.

- Purpose: Prove scoped private-CA trust without replacing AWS public trust, positive authenticated readiness, empty-CA rejection, STS success, five-minute clean logs, and no unexplained unclaimed backlog. BOOT-021's readiness matrix may be recorded from the same per-Pod run. Catalog manual/live.
- Runner: scripts/e2e/regional/run_boot_acceptance.py; scripts/e2e/regional/boot_acceptance_runtime.py; scripts/e2e/regional/boot_acceptance_common.py; scripts/e2e/regional/audit_executor_readiness.py
- Entry Points: main -> run_boot012 -> ca_file_contract -> SiteFixture.pods -> environment/readiness/empty-CA/STS/readiness-audit/log probes; RegionalExecutorClient.__init__ builds ssl.create_default_context(cafile=ca_file); readiness_matrix_verdict writes optional BOOT-021 case evidence
- Assertions: scoped_ca compares env path and summarized mount, rejects SSL_CERT_FILE/REQUESTS_CA_BUNDLE; authenticated_readiness, sts_public_trust, stale_claim_503, recent TLS/error count zero; empty_ca_rejected requires CERTIFICATE_VERIFY_FAILED specifically; Full audit subprocess validates all five status/body/reason combinations, but BOOT-012 consumes only stale_claim_503 rather than a complete matrix result
- Safety And Cleanup: Only Ready Pods are returned by SiteFixture.pods; a missing/unhealthy Executor replica is not counted; Empty /tmp/boot012-empty-ca.pem is created inside each Pod and never removed; Failed log reads use check=False and can look like zero TLS errors; Optional BOOT-021 evidence is emitted immediately without checking its BOOT-020 formal predecessor
- Necessity: Scoped TLS and AWS trust are distinct useful checks. An empty CA normally fails at SSL context loading with NO_CERTIFICATE_OR_CRL_FOUND, which the spec explicitly accepts but the runner rejects; unrelated nonzero failures must still not pass.
- Overlap: case_ids: GF-REGIONAL-BOOT-011; GF-REGIONAL-BOOT-018; GF-REGIONAL-BOOT-021; classification: BOOT021_strict_subset_plus_complementary_checks; rationale: BOOT-012 executes the exact BOOT-021 matrix per Ready Executor Pod; 021 is fully duplicated execution, while TLS/STS/log checks are unique here. Artifact source closure belongs to 018.
- Tests: existing: tests/regional/test_boot_acceptance_runtime.py::test_ca_file_contract_matches_the_manifest_against_itself; tests/regional/test_boot_acceptance_runtime.py::test_ca_file_contract_rejects_drift_and_global_trust_overrides; tests/regional/test_boot_acceptance_runtime.py::test_readiness_matrix_verdict_requires_every_replica_and_status; tests/regional/test_regional_acceptance_fixtures.py::test_executor_readiness_fixture_rejects_a_generic_503; assessment: No test executes run_boot012. Existing matrix helper fixtures contain statuses only; precise response checks are covered separately in the shared audit.; planned_behavior: Mock accepted empty-CA SSL error variants versus unrelated failures; Require all desired nonterminating replicas and fail unreadable logs; Exercise all five matrix failures through run_boot012 and its optional evidence writer; Require exact CA key-to-relative-path projection and cleanup of the temporary file
- Findings At Review: P1: empty-CA false failure on valid NO_CERTIFICATE_OR_CRL_FOUND; P1: Ready-only sampling and ignored log-read failures; P1: absent backlog age <=600/idle observation; P1: early BOOT-021 formal evidence; BC-SHARED-002; P2: summarized mount omits projected item paths and temporary file remains
- Reviewer Handoff: Owned TLS/replica/log/matrix/cleanup and early-future-PASS defects fixed.

### GF-REGIONAL-BOOT-013

控制面缺少 AWS_REGION 时邮件必然失败

Current catalog: non-destructive; manual; NOT_RUN.

- Purpose: Every API Pod must expose target AWS region, construct an SES client with it, reject an intentionally invalid send locally, and fail NoRegionError without region. ALLOW_EMAIL must agree with config; actual email is optional and needs separate consent. Catalog manual/live/non-destructive.
- Runner: scripts/e2e/regional/run_boot_acceptance.py; scripts/e2e/regional/boot_acceptance_runtime.py; scripts/e2e/regional/boot_acceptance_common.py
- Entry Points: main -> run_boot013 -> BOOT013_PROBE per SiteFixture.pods('cpu', 'gpu-fault-api-ha'); SesNotificationConfig.from_environment -> SesEmailNotifier._create_client -> invalid ToAddresses [1] local ParamValidationError; Child removes AWS_REGION/AWS_DEFAULT_REGION and redirects AWS_CONFIG_FILE to /dev/null
- Assertions: Nonempty results, region_present_all, config/client region == fixture.region; execution_enabled equals parsed allow_email; local_param_validation and NoRegionError text from child stderr
- Safety And Cleanup: No accepted SES payload or email-switch mutation; invalid parameters should stop before SES transport; Only current Ready API replicas are sampled; not the worker role that actually dispatches asynchronously; Child subprocess has no inner timeout, relying on the outer Pod exec timeout; results omit Pod names
- Necessity: A good zero-delivery configuration check, not notification delivery acceptance. It needs explicit replica completeness and declared-role scope to avoid proving only the healthy subset.
- Overlap: case_ids: GF-REGIONAL-BOOT-014; GF-REGIONAL-NOTIFY-001; classification: complementary; rationale: Region construction and dispatcher configuration are independent failure modes; NOTIFY-001's real delivery/dedup path is intentionally not exercised here.
- Tests: existing: tests/regional/test_regional_boot_guard_acceptance.py::test_boot013_the_ses_client_region_comes_only_from_the_declared_environment; assessment: Covers config and mocked boto3 creation, not BOOT013_PROBE or run_boot013's fleet aggregation. Repository search found no direct run_boot013 call in tests.; planned_behavior: Execute the probe with mocked boto3 transport and require ParamValidationError before any send; Feed mismatched replica region/config flags and missing desired replicas through run_boot013; Check child nonzero exit and exact NoRegionError classification instead of accepting stderr text alone; Record Pod names for per-replica diagnostic provenance
- Findings At Review: P1: Ready-only replica sampling and ingress-only scope; P2: no declared Manifest/live value cross-check; P2: no end-to-end probe aggregation tests or named replica evidence
- Reviewer Handoff: Owned replica aggregation and deployed-interpreter correctness fixed; zero-delivery scope retained.

### GF-REGIONAL-BOOT-014

通知 dispatcher 的开关与邮件开关必须一致

Current catalog: non-destructive; manual; NOT_RUN.

- Purpose: Asynchronous email must have its dispatcher enabled; compare every replica's environment with service attributes and prove no notification lacks a result. Real delivery/toggling is optional, separately approved. Catalog manual/live/non-destructive.
- Runner: scripts/e2e/regional/run_boot_acceptance.py; scripts/e2e/regional/boot_acceptance_runtime.py; scripts/e2e/regional/boot_acceptance_common.py
- Entry Points: main -> run_boot014 -> BOOT014_PROBE constructs ApplicationContext and reads advisory_notifications plus notification/results; run_boot014 normalizes five service fields and calculates backlog from values[:1]
- Assertions: Requires at least one Ready API result and identical normalized service values; allow_email == dispatcher_enabled and no async/email-on/dispatcher-off combination; backlog_without_result_zero only uses the first replica's count difference; Reads notification_result rather than nonexistent notification.status
- Safety And Cleanup: No explicit dispatch or switch changes; Unlike the specification's InMemoryStore plus raising notifier, the probe constructs a full production ApplicationContext, including Store/registry setup and real notifier construction; No direct comparison of async_delivery/dispatcher_enabled/backlog settings against raw env; only ALLOW_EMAIL is independently read
- Necessity: Necessary complement to region validation. Service-object agreement is not the same as environment-to-service agreement, and an existing FAILED/PENDING result is not evidence of healthy delivery; the spec currently defines the narrower missing-result check.
- Overlap: case_ids: GF-REGIONAL-BOOT-013; GF-REGIONAL-NOTIFY-001; classification: complementary; rationale: Shared symptom is missing mail, but independent configuration causes warrant both BOOT cases; actual delivery remains NOTIFY coverage.
- Tests: existing: tests/regional/test_regional_boot_guard_acceptance.py::test_boot014_the_dangerous_delivery_switch_combination_is_named_not_silent; assessment: Exercises AdvisoryNotificationService combinations with a fake sender; no run_boot014 or BOOT014_PROBE behavior test.; planned_behavior: Feed environment/service disagreement and asymmetric replica backlog through the full handler; Ensure an unhealthy/missing desired replica cannot be skipped; Use a mocked raising notifier and explicit read-only Store access so probing cannot invoke real delivery
- Findings At Review: P1: all-replica and environment/service correspondence not fully checked; P2: first-replica-only backlog observation; P2: full context creation broadens a nominally read-only probe
- Reviewer Handoff: Owned environment correspondence and asymmetric-backlog false-PASS defects fixed.

### GF-REGIONAL-BOOT-015

runtime profile 声明的 owner 必须有对应的 executor adapter

Current catalog: non-destructive; manual; NOT_RUN.

- Purpose: Compare runtime Profile remote owners with actual Executor owners, inject one old unclaimable synthetic command, prove readiness denial/PENDING metrics/AMP firing, then remove the command and observe alert resolution. Catalog still describes no deadline/alerts although current runtime has a 900s unclaimed deadline and owner-aware readiness.
- Runner: scripts/e2e/regional/run_boot_acceptance.py; scripts/e2e/regional/boot_acceptance_runtime.py
- Entry Points: run_boot015 -> actual executor_from_environment().execution_owners -> PROFILE_OWNER_PROBE -> static alert gate -> baseline -> REMOTE_INJECT_PROBE; TRIGGER_HEALTH_SNAPSHOT uses nonexistent owner/node and created_at now-610s; readiness/log/metric probes -> wait_amp_alert -> REMOTE_STATE_PROBE -> finally boot015_cleanup
- Assertions: Remote owner subset and identical advertised owner sets; All sampled Executors must reject readiness with owner-backlog reason; Two metrics requests <5s, PENDING>=1, oldest increases, command remains PENDING/unleased; No command ID in sampled logs and AMP alert firing; Cleanup requires injected ID absent, no new IDs and readiness restored, but does not require AMP resolve_observed.matched
- Safety And Cleanup: Injected flag becomes true only after cpu_python(REMOTE_INJECT_PROBE) returns; committed write plus ACK/transport loss skips deletion entirely; Command ID uses attempt+integer time, with no existence/ownership collision check; deletion is raw _delete by ID; Baseline/open-workload/owner preconditions are not hard-gated before injection; script can deliberately drop readiness on production Executors; Error log reads use check=False; raw diagnostic tails/error text are persisted without a dedicated redaction layer
- Necessity: The synthetic unowned command is a reasonable bounded observation technique only with approved service-impact scope and verified cleanup. Current catalog non-destructive wording understates dropping every Executor's readiness; missing Profile rows make the subset check vacuously true.
- Overlap: case_ids: GF-REGIONAL-BOOT-021; GF-REGIONAL-CAP-002; classification: complementary_negative_readiness_and_alerting; rationale: BOOT-021 tests a bad advertised owner request without creating backlog; BOOT-015 tests a real unowned backlog and its alert. CAP-002 tests a different Store-capacity signal.
- Tests: existing: tests/regional/test_boot_acceptance_runtime.py::test_boot015_cleanup_records_each_step_and_verifies_deletion_by_listing; tests/regional/test_boot_acceptance_runtime.py::test_boot015_writes_details_and_cleans_up_when_a_probe_raises; tests/regional/test_regional_boot_guard_acceptance.py::test_boot015_a_command_no_adapter_owns_is_skipped_silently_but_counted; assessment: Some cleanup paths are behaviorally covered, but existing tests explicitly accept unresolved alerts and only fail after an acknowledged insert; insertion-ACK loss is uncovered.; planned_behavior: Simulate Store insert success followed by transport exception; require ownership-bound deletion/readback and FAIL evidence; Require no injection when Profile set is empty, owners disagree, or prerequisite observations fail; Require bounded target-alert resolution and disappearance of the unclaimed metric; Test command ID collision, read failures, and exception preservation through independent cleanup steps
- Findings At Review: P1: ACK-loss cleanup bypass leaves a nonterminal command and persistent readiness failure; P1: unresolved AMP alert can still produce PASS, contradicting catalog and global cleanup rule; P1: vacuous Profile subset and preconditions not enforced before mutation; P1: parent must reconcile stale catalog risk/problem text and formal identity binding
- Reviewer Handoff: Owned ACK-loss, vacuous Profile and cleanup false-PASS defects fixed.

### GF-REGIONAL-BOOT-016

ARN单命令首次部署先生成完整registry且零集群稳态可运行

Current catalog: live-non-destructive; manual; NOT_RUN.

- Purpose: Fresh ARN-based deployment must create complete registrations before workloads, reject empty first-bootstrap input before mutation, converge verified release, and rerun as NOOP; use isolated site and preserve underlying clusters. Catalog/manual spec still requires signed build before non-ECR infrastructure, unlike current concurrent bootstrap.
- Runner: scripts/e2e/regional/run_boot_acceptance.py; scripts/e2e/regional/boot_acceptance_lifecycle.py; scripts/e2e/regional/boot_acceptance_common.py
- Entry Points: run_boot016 rejects nonempty state dir -> _boot016_verified_site -> admin_command deploy -> one local pytest empty-bootstrap guard -> status --full -> same deploy; Reads registry Secret, release state and Deployment creation timestamps/generations; finally cleanup_isolated_site retains PASS for 017/018, uninstalls failed site if site.yaml exists
- Assertions: 0700 state and 0600 site, nonempty cluster list; Registry IDs == site IDs and registry creation <= all observed workload creation; Local empty-bootstrap pytest returncode == 0; Release phase complete/completed, status exit zero, rerun exit zero, Deployment generations identical
- Safety And Cleanup: New state directory is not proof CPU/GPU ARNs belong to an isolated site; no production baseline comparison; admin_command uses raw subprocess.run timeout without the administrator process-tree completion proof; timeout can trigger uninstall while descendants remain unproven; If infrastructure is created before site.yaml is committed, cleanup_isolated_site assumes nothing can be removed; Default uninstall --cpu-cluster keep preserves Aurora unless reset/delete policy explicitly says otherwise, so full AWS teardown is not proven
- Necessity: Core lifecycle case is essential, but present runner substitutes one mocked pytest for the live empty-bootstrap branch and checks only a subset of deployment outputs. Parent must update obsolete build/infrastructure ordering without undoing prior authorized concurrency work.
- Overlap: case_ids: GF-REGIONAL-BOOT-002; GF-REGIONAL-BOOT-011; GF-REGIONAL-BOOT-017; GF-REGIONAL-BOOT-019; GF-REGIONAL-BOOT-020; classification: shared_fixture_partial_overlap; rationale: 017 explicitly reuses this deployed site; 011 needs it earlier in the order. 002/019 prove empty steady state, not empty initial bootstrap. 020's NOOP overlap is release-upgrade idempotence, not infrastructure creation.
- Tests: existing: tests/regional/test_boot_acceptance_lifecycle.py::test_boot016_uninstalls_the_site_after_a_deploy_timeout; tests/regional/test_boot_acceptance_lifecycle.py::test_boot016_honours_retain_and_never_uninstalls_a_pass; tests/regional/test_regional_admin_commands.py::test_deploy_rejects_empty_cluster_set_during_bootstrap; assessment: Cleanup tests mock admin_command and create site.yaml before error; none exercise _boot016_verified_site or pre-site infrastructure residue.; planned_behavior: Mock full deploy/status/rerun observations and independently fail each verdict; Prove isolation target binding before first admin call and reject conflicting identities; Test interrupted deployment without site.yaml but with a bootstrap resource journal; Do not auto-uninstall without confirmed child process completion; coordinate supported admin recovery with parent
- Findings At Review: P1: live empty-bootstrap negative, signature-failure side effects, ECR ownership/cache/reuse, completed cluster set, Agent identity and stability report are not asserted; P1: raw subprocess timeout and pre-site cleanup blind spot; P1: formal fixture lifecycle/order conflicts (BC-SHARED-002); P2: generation equality is weaker than NOOP classification plus unchanged runtime identity
- Reviewer Handoff: Owned process-supervision and identity fail-open defects fixed. Parent owns pre-site administrator recovery and final epoch/spec integration.
- Current Disposition: Order starts with BOOT-016; foundations remain parallel with build. Persistent acceptance site is retained.

### GF-REGIONAL-BOOT-017

部署和运维手册的命令顺序可执行性干跑

Current catalog: non-destructive; manual; NOT_RUN.

- Purpose: Manual command-order static gate plus three deliberate bad-order mutations must pass; reuse BOOT-016's successful greenfield deployment/status for the live component. Missing valid BOOT-016 reuse is PARTIAL, not PASS. Catalog manual with check-manual-command-order.py gate.
- Runner: scripts/e2e/regional/run_boot_acceptance.py; scripts/e2e/regional/boot_acceptance_lifecycle.py; scripts/check-manual-command-order.py
- Entry Points: run_boot017 executes manual-order --verbose and test_manual_command_order_gate_is_not_constant_green; boot016_reuse reads the predecessor document and compares state_dir; returns NOT_EVALUATED/boolean facts; run_boot017 maps failed local gates to FAIL, valid reuse+status to PASS, absent reuse to PARTIAL
- Assertions: Both local exit codes must be zero; Predecessor wrapper verdict PASS, evidence case_id BOOT-016, same resolved state_dir, and checks.status_passed truthy; No new admin status or manual step-by-step deployment is executed
- Safety And Cleanup: This handler performs local checks and consumes existing isolated-site evidence; site_identity performs a best-effort live read and returns None on error, so stale health can be reused after identity failure; Manual checker/tool ownership stays with parent
- Necessity: The explicitly documented reuse is reasonable for avoiding duplicate deployment, and PARTIAL behavior is correct. However old file evidence is not proof the site remains untouched, and the reread document's own verdict is not revalidated.
- Overlap: case_ids: GF-REGIONAL-BOOT-016; classification: live_component_duplicate_static_component_unique; rationale: Its live part is exactly BOOT-016 reuse; command dependency parsing and constant-green selftests are the independent value. Do not count it as a second fresh installation.
- Tests: existing: tests/regional/test_boot_acceptance_lifecycle.py::test_boot016_reuse_is_not_evaluated_under_selective_scope; tests/regional/test_boot_acceptance_lifecycle.py::test_boot016_reuse_requires_the_same_state_dir; tests/regional/test_boot_acceptance_lifecycle.py::test_boot017_is_partial_when_the_greenfield_run_cannot_be_evaluated; assessment: Good PARTIAL/identity-path tests but no stale/reread-verdict drift or unreadable live identity case.; planned_behavior: Pass wrapper with reread FAIL document must not satisfy reuse; Require exact boolean status_passed, not truthy strings; Reject identity drift/read failure before promoting reused deployment evidence
- Findings At Review: P1: predecessor/identity freshness is not enforced by owned best-effort helpers; P2: file reread does not require document.verdict == PASS; P2: manual static gate cannot independently prove each manual command ran live
- Reviewer Handoff: Owned reuse and identity drift defects fixed; explicit016 dependency supplied to parent.
- Current Disposition: Explicit predecessor is BOOT-016; intermediate read-only cases no longer invalidate this handoff.

### GF-REGIONAL-BOOT-018

发布制品 Node bundle 与运行进程四面一致

Current catalog: non-destructive; manual; NOT_RUN.

- Purpose: Four-way artifact identity: reproducible component wheels/bundle under umask 077/002, matching source closures, image descriptors/isolated runtimes, every live Pod and unexpired ACTIVE Agent, release config/key pins, plus a content-tamper negative. Catalog manual/staging/non-destructive, though runner normally uninstalls the isolated site.
- Runner: scripts/e2e/regional/run_boot_acceptance.py; scripts/e2e/regional/boot_acceptance_lifecycle.py; scripts/e2e/regional/boot_acceptance_common.py
- Entry Points: boot018_body -> two parallel build_release_under_umask copies -> artifact pytest -> compare local Manifest SHA/digests -> modify copied source -> negative pytest; admin status --full with quick-evidence reuse disabled -> _live_release_identity -> runtime_identity_matches_release; run_boot018 finally cleanup_isolated_site
- Assertions: Local rebuilt release_id, bundle SHA and three component wheel SHA/module digests equal across masks; Any nonzero tamper pytest exit passes the negative check; Live report digests/metadata/ACTIVE Agent artifact+compatibility match the live Manifest; Cleanup exit status can turn PASS into FAIL
- Safety And Cleanup: Temporary checkouts are removed, but build subprocesses use preexec_fn umask from threads; Default final uninstall is a service action not reflected in the non-destructive catalog label; --retain-bootstrap-site bypasses it; ACTIVE_AGENT_PROBE does not collect config digest/key version or the expected target node set
- Necessity: High-value release acceptance, but the current implementation proves two separate consistent systems: local source/build agreement and live Manifest/runtime agreement. It never compares rebuilt component identities to the live Manifest, so a completely different consistent live release can pass.
- Overlap: case_ids: GF-REGIONAL-BOOT-012; GF-REGIONAL-BOOT-016; GF-REGIONAL-BOOT-020; classification: complementary_supply_chain_with_shared_fixture; rationale: 012 covers TLS/readiness, 016 deployment, 020 transitions/rollback; none replaces source-to-live identity proof. Uninstall overlaps 019 and disrupts its next-case prerequisite.
- Tests: existing: tests/regional/test_boot_acceptance_lifecycle.py::test_runtime_identity_passes_only_when_every_replica_matches_the_release; tests/regional/test_boot_acceptance_lifecycle.py::test_runtime_identity_fails_on_missing_fields_instead_of_skipping; tests/regional/test_boot_acceptance_lifecycle.py::test_boot018_builds_the_two_umask_checkouts_concurrently; tests/regional/test_boot_acceptance_lifecycle.py::test_boot018_records_cleanup_for_every_verdict_and_fails_a_bad_uninstall; assessment: Identity helper and cleanup have tests; the concurrent-build test intentionally ends at FileNotFoundError, before source-to-live comparison. No full mocked successful/negative boot018_body test.; planned_behavior: Use equal local builds and a different internally consistent live Manifest; require FAIL; Require config/key pin agreement and exact expected Agent/Pod coverage; Distinguish tamper assertion failure from pytest collection/import failure or skipped test; Verify image descriptor/system-Python isolation facts and restore-after-tamper proof, not merely source-only wheel tests
- Findings At Review: P1: no bridge between rebuilt artifacts and live release; P1: Agent config/key and expected-node completeness absent; P1: default cleanup violates the following site's prerequisite unless parent defines separate epochs; P2: generic negative pytest failure can falsely prove tamper detection; v4 split-image spec text is stale
- Reviewer Handoff: Owned source/live identity, tamper verdict and persistent-fixture teardown defects fixed; formal retain is enforced.
- Current Disposition: Formal invocation requires retaining the persistent acceptance site.

### GF-REGIONAL-BOOT-019

管理员集群接入、注销与最后集群空稳态

Current catalog: live-service-action; manual; NOT_RUN.

- Purpose: Join a second isolated GPU cluster, fail on both sides of the irreversible boundary, resume, reconcile four member sets, remove second/last clusters, reject old token and uninstall while preserving underlying clusters. Spec also requires no CPU restart and accepted first heartbeat/claim.
- Runner: scripts/e2e/regional/run_boot019_admin_lifecycle.py
- Entry Points: main now uses schema-3 build_plan/authorize_execution and canonical case_evidence_path; run_admin_lifecycle -> LiveAdminLifecycleBackend.join/remove/snapshot/probe_revoked_token/uninstall; join now wraps cluster_join_commit.complete_step at REGISTRY_UPDATED and ACTIVATION_STARTED
- Assertions: Baseline exactly one GPU; site/Secret/release-state/installation-registry member sets agree; Before activation: ROLLED_BACK; after activation intent: FAILED_AFTER_ACTIVATION; resume: COMPLETED; Removed token 403 with exact detail; final registry empty; API desired replicas positive and all ready; Uninstall preserves Aurora and at least two resources, with only DELETED/DETACHED/PRESERVED final statuses; PASS is persisted after sensitive cleanup; legacy site-commit stage records are rejected
- Safety And Cleanup: Captured token remains in memory and is cleared in finally; Site bytes are hashed into the approved plan; acceptance_contract=2 prevents legacy replay; Administrator subtransactions own compensation; no case-wide resource cleanup after an arbitrary mid-case failure; Memory-only captured token cannot survive process loss between capture/removal and revocation probe
- Necessity: Essential membership coverage. The old live backend immediately raised AttributeError on moved _commit_site, while fake backend tests stayed green. Current irreversible boundary is ACTIVATION_STARTED, not site commit. No-restart/first-heartbeat claims still lack observations; failure-domain-map changes may legitimately roll the worker.
- Overlap: case_ids: GF-REGIONAL-BOOT-002; GF-REGIONAL-BOOT-016; GF-REGIONAL-BOOT-020; GF-REGIONAL-BOOT-023; classification: complementary_lifecycle_shared_state; rationale: 002 checks empty startup, 016 first deployment, 020 releases, 023 audit consistency. This uniquely tests membership removal/revocation; its teardown cannot silently provide a deployed prerequisite for 020.
- Tests: existing: tests/regional/test_boot019_admin_lifecycle.py::test_lifecycle_completes_and_cleans_up_last; current credential-custody successor: tests/regional/test_boot019_admin_lifecycle.py::test_live_backend_protects_revocation_credentials_until_the_case_completes; tests/regional/test_regional_acceptance_fixtures.py::test_boot019_runner_executes_and_records_the_full_lifecycle; added: tests/regional/test_boot019_admin_lifecycle.py::test_live_backend_injects_at_the_current_irreversible_boundary; tests/regional/test_boot_lifecycle_entrypoints.py::test_lifecycle_entrypoint_uses_a_bound_plan_and_canonical_evidence; tests/regional/test_boot_lifecycle_entrypoints.py::test_changed_lifecycle_input_file_requires_a_new_plan; assessment: Two new hook tests reproduced AttributeError, then exercised actual complete_step persistence with mocked join and verified restoration. Existing broad fake never reached the removed symbol.
- Findings At Review: Fixed: moved-symbol failure and obsolete failure boundary; Fixed: unguarded legacy main and absent canonical verdict; Remaining: CPU UID/restart/healthz drift and first-heartbeat/claim observations; Remaining: cross-process token-capture resume and limited stage live revalidation; Parent: BC-SHARED-002 lifecycle order and BC-SHARED-005 mixed tests/spec
- Reviewer Handoff: Owned moved-symbol, irreversible-boundary, protected-epoch and canonical-evidence safety defects fixed. Parent central order/docs integration acknowledged.
- Current Disposition: Disposable lifecycle target is physically disjoint from mandatory --protected-site; current activation failure boundary and canonical evidence are tested.

### GF-REGIONAL-BOOT-020

发布差异分类、两阶段Pin、失败续跑与自动回滚

Current catalog: live-service-action; manual; NOT_RUN.

- Purpose: Chained NOOP, CPU-only, Executor-only, Agent-only and FULL candidates must select correct scope, stage/finalize pins, rollback within fixed RTOs, resume interrupted Executor phase and reapply FULL without trusting an old completed Installer Job.
- Runner: scripts/e2e/regional/run_boot020_release_rolling.py; scripts/e2e/regional/boot020_release_candidates.py
- Entry Points: main now admits schema-3 plans bound to all five config SHA values and writes canonical PASS evidence; run_release_rolling -> five _stage_* handlers; LiveReleaseRollingBackend.deploy injects after RegionalRelease._save_state and calls noop/upgrade; resume_release_rolling discards incomplete records and reconverges the previous scenario
- Assertions: Exact classification; NOOP live/generation equality; CPU-only GPU equality; Executor rollback/staged/finalized required/compatible SHA windows and exact Executor-only GPU generation changes; Nonempty rollback plan, scenario scope and T_safe/T_full limits; Executor resume <=1200s without second CPU generation change; FULL rollback restores prior live snapshot; reapply changes live state and next classification is NOOP; --start-stage now requires every skipped *_passed marker, not merely *_after
- Safety And Cleanup: Release engine owns rollback; driver records failure without inventing compensation after unrelated exceptions; Approved config bytes and source must remain unchanged; legacy plans refused; Candidate clean removes matching symlinks only, but candidate build deletes/recreates wt-* directories and needs explicit ownership; No user BOOT-020 resource/config/evidence paths were inspected
- Necessity: Useful composite release suite, but phase flags and generation comparisons do not prove all claims. Agent-only can roll an unrelated GPU Deployment without a dedicated veto; FULL lacks observed phase order, Installer Job UID recreation and exact final Agent heartbeat checks.
- Overlap: case_ids: GF-REGIONAL-BOOT-016; GF-REGIONAL-BOOT-018; GF-REGIONAL-BOOT-023; classification: partial_overlap_different_claims; rationale: NOOP overlaps first-deploy idempotence and audit NOOP, while rollback/pin/resume is unique. Candidate comment edits change module digests, so they do not cover equal-digest metadata-only wheel SHA changes required by the spec.
- Tests: existing: tests/regional/test_boot020_runner.py::test_boot020_runner_covers_diff_resume_and_rollback; tests/regional/test_boot020_release_rolling.py::test_executor_stage_checks_the_pin_window_at_each_checkpoint; tests/regional/test_boot020_release_candidates.py::test_chain_changes_one_dimension_per_step; added: tests/regional/test_boot020_runner.py::test_boot020_start_stage_cannot_skip_an_unpassed_terminal_assertion; tests/regional/test_boot_lifecycle_entrypoints.py::test_lifecycle_entrypoint_uses_a_bound_plan_and_canonical_evidence; tests/regional/test_boot_lifecycle_entrypoints.py::test_changed_lifecycle_input_file_requires_a_new_plan; assessment: New test reproduced snapshot-only skipping. Most stage tests use a fake backend; actual deploy injection/capture paths need additional adapter-level mocks.
- Findings At Review: Fixed: --start-stage skipping failed terminal assertions; Fixed: missing plan/config-content binding and canonical PASS; Remaining: Agent-only/all-cluster scope, physical-SHA-only candidate, actual FULL phase/wave/Job identity proof; Remaining: candidate check prints classifications without rejecting unexpected kinds; Parent: initial noop.json no longer classifies NOOP after final FULL
- Reviewer Handoff: Owned resume, missing-generation and predecessor-evidence defects fixed; remaining candidate/proof breadth is explicit, not a claim of live completion.
- Current Disposition: Explicit predecessor is BOOT-018 on the persistent target, not the disposable BOOT-019 target.

### GF-REGIONAL-BOOT-021

Executor认证readiness负向矩阵

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: Authenticated readiness matrix: valid 200/ready; wrong token 403; wrong artifact, no owner and stale claim age each 503 with exact reasons; no command creation.
- Runner: scripts/e2e/regional/audit_executor_readiness.py; scripts/e2e/regional/boot_acceptance_runtime.py; scripts/e2e/regional/run_boot_acceptance.py
- Entry Points: audit_executor_readiness.main reads deployed identity/owners and sends five POSTs; validate_readiness_matrix uses explicit _require and survives python -O; run_boot012 optionally writes BOOT-021 evidence via readiness_matrix_verdict
- Assertions: Valid ready=True, no reasons/unsupported owners; Wrong-token exact authentication detail; Wrong-pin exact reason shape and echoed artifact; no-owner exact reason; stale-age exact echo/regex; Owned BOOT-012 now requires the complete matrix and full desired replica set
- Safety And Cleanup: Readiness requests only; no commands/nodes/workloads changed; API positive control sets claim age=0; real claim freshness is separately checked by BOOT-012 readiness CLI; Shared audit belongs to parent and was not edited; Early co-recording does not prove the later formal BOOT-020 prerequisite
- Necessity: Strong API matrix, wholly duplicated by BOOT-012. Keep a public case only with explicit scoped evidence reuse, not a second purported live execution.
- Overlap: case_ids: GF-REGIONAL-BOOT-012; GF-REGIONAL-BOOT-015; classification: strict_subset_of_BOOT012; rationale: 012 runs this exact audit per Executor. 015 is complementary because it creates actual unowned backlog instead of altering request fields.
- Tests: existing: tests/regional/test_regional_acceptance_fixtures.py::test_executor_readiness_fixture_validates_precise_reasons; tests/regional/test_regional_acceptance_fixtures.py::test_executor_readiness_fixture_rejects_a_generic_503; tests/regional/test_boot_acceptance_runtime.py::test_readiness_matrix_verdict_requires_every_replica_and_status; added: tests/regional/test_boot_acceptance_behavior.py::test_boot012_requires_the_whole_readiness_matrix; tests/regional/test_boot_acceptance_behavior.py::test_boot012_accepts_only_a_specific_empty_ca_tls_refusal; assessment: Existing precise reason tests are useful; added tests now exercise matrix consumption and evidence output through run_boot012.
- Findings At Review: Parent: approved reuse semantics or recording at the formal 021 slot; Parent: standalone guarded 021 runner is missing; raw audit or 012 option only; API age=0 is not itself successful production-claim freshness
- Reviewer Handoff: Standalone canonical021 implemented; early future PASS path removed.
- Current Disposition: Standalone guarded entry; BOOT-012 cannot pre-issue its later PASS.

### GF-REGIONAL-BOOT-022

Runtime Profile内容寻址不可变与site版本绑定

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: Content-addressed immutable Runtime Profile, approval on policy change and site-bound training/annotation versions. Catalog command runs three local pytest nodeids; spec additionally requests live site/Agent/Watcher/workload version comparison.
- Runner: testcases/fault-scenarios.yaml; tests/admin/test_release_deploy.py; tests/regional/test_regional_release_config_and_profile.py; tests/hyperpod/test_training_submit_cli.py
- Entry Points: Three-nodeid pytest wrapper: profile-change approval, same-version runtime drift refusal and training site-version resolver; regional_case_contract.is_pytest_wrapper excludes this from canonical live predecessor selection
- Assertions: Mocked policy change generates content-derived version, approval/consumption and site update; ensure_runtime_profile refuses changed content under the same version without posting; training_submit_cli.resolve_runtime_profile_version uses site and rejects explicit mismatch; No runner performs the extra live comparison
- Safety And Cleanup: Temporary files and mocked release/API calls only for catalog command; No canonical live case evidence from the pytest wrapper; Central command selection and these non-owned tests remain parent-owned
- Necessity: Keep the offline contract offline. Selected nodeids do not cover every stated claim: normalized unchanged-policy NOOP, snapshot-file collision, workload-annotate mismatch or live identity comparison.
- Overlap: case_ids: GF-REGIONAL-BOOT-018; GF-REGIONAL-BOOT-020; classification: complementary_offline_policy_contract; rationale: 018 checks content identities and 020 transitions; neither replaces immutable planning. Conversely three local tests cannot prove deployed Profile agreement.
- Tests: existing: tests/admin/test_release_deploy.py::test_profile_change_requires_approval_and_generates_version; tests/regional/test_regional_release_config_and_profile.py::test_runtime_profile_drift_requires_a_new_version; tests/hyperpod/test_training_submit_cli.py::test_site_supplies_runtime_profile_and_rejects_mismatch; assessment: Read all three bodies. The training test stubs load_site and checks only a resolver. Adjacent idempotent registration test exists but is not selected.; planned_behavior: Parent expand offline selection with idempotence, snapshot collision and annotation CLI tests; Add a guarded read-only runner only if the live comparison remains required
- Findings At Review: Missing runner for additional live step; Incomplete offline nodeid selection; Do not make this a live predecessor without defining real evidence
- Reviewer Handoff: Offline contract preserved; no owned live execution implied. Concrete missing live-profile case proposed.

### GF-REGIONAL-BOOT-023

发布审计历史只追加、注册表 durable 发布与 Secret 一致、rollback_compatible 声明在计划期被拒

Current catalog: live-service-action; manual; NOT_RUN.

- Purpose: After BOOT-020, run NOOP and prove bounded append-only history/mirror, <=3 unchanged previous snapshot groups, per-CPU-Pod Secret/durable agreement, stable head generation and rejection of rollback_compatible=true.
- Runner: scripts/e2e/regional/run_boot023_release_history.py; scripts/e2e/regional/boot023_verdicts.py; scripts/e2e/regional/probes/boot023_registry_probe.py
- Entry Points: CaseRunner -> read_only_preflight -> classification/history/groups/probe/runtime reads and focused pytest; execute_case repeats preflight/plan_identity, invokes run_noop_release and pure verdict functions; finally verify_runtime_identity_allowing permits only updated_at_epoch
- Assertions: NOOP, phase complete, valid predecessor/tests/probes; Ring suffix/prefix alignment, appended SHA/operator/command/last phase and exact mirror tail; Unchanged snapshot group names and bound; unchanged per-Pod head generation; Registry validator currently permits missing healthz secret digest and does not require positive/cross-Pod matching head generation
- Safety And Cleanup: Retains history as audit data; final runtime comparison can turn PASS into FAIL; Release config targets are not bound to the separate RegionalLiveSettings used for probes; RegionalRelease.noop applies Watcher state ConfigMaps/RBAC; history-only/no-GPU-write plan text is inaccurate; No direct plan/execution invoked in this review
- Necessity: History validation is substantive. Actual orchestration, target binding and fail-closed registry proof are the largest gaps. Original 020 noop config points at the pre-FULL release and needs re-materialization.
- Overlap: case_ids: GF-REGIONAL-BOOT-019; GF-REGIONAL-BOOT-020; classification: complementary_audit_subset; rationale: Only NOOP history and registry consistency are checked; durable publication and history during membership/upgrade/rollback require 019/020 assertions, not another rollout.
- Tests: existing: tests/regional/test_boot023_release_history.py::test_a_full_ring_that_dropped_its_oldest_entry_still_counts_as_growth; tests/regional/test_boot023_release_history.py::test_each_registry_probe_failure_is_named; tests/regional/test_boot023_release_history.py::test_a_plan_that_drifted_from_its_preflight_is_refused; assessment: Frozen branch coverage is 95.45% for verdicts, 40.65% for runner. No execute_case test; uncovered orchestration is more important than more formatting examples.; planned_behavior: Bind ReleaseConfig target to fixture before Runner creation; Reject missing health digest/head identity and cross-Pod disagreement; Mock execute success/error/cleanup drift and interruption evidence; Bind runtime/history contents, not only counts, into plan identity
- Findings At Review: P1: unbound release/probe targets; P1: history-only mutation claim disagrees with noop call chain; P1: missing registry evidence can pass; P2: shape-only plan identity and missing execute behavior coverage
- Reviewer Handoff: Owned target/evidence/interpreter safety fixes complete; parent history-only API dependency resolved.
- Current Disposition: History-only noop explicitly disables prerequisite repair while retaining release validation and history persistence.

### GF-REGIONAL-BOOT-024

Historical WIP identity; the current case is
[GF-REGIONAL-BOOT-030](../区域模式端到端验收测试用例.md#gf-regional-boot-030).

部署后remote_command专表模式与双写一致性

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: 本机迁移测试不能证明部署后每个CPU角色连接同一writer并使用与release匹配的专表schema。
- Runner: scripts/e2e/regional/run_state_table_acceptance.py
- Entry Points: guarded main -> read-only preflight -> plan/execute -> evidence and cleanup
- Assertions: 全部启用CPU角色具有完整Ready副本，执行前后Deployment和Pod身份不变; schema版本、连续migration校验和与结构校验均匹配已部署release; 所有副本的数据库摘要、模式和revision一致，模式等于显式批准的expected-mode; dual要求回填完成，missing、mismatched、extra、invalid和noncanonical均为零; dedicated允许尚未清理的历史行，但legacy_purged标记与实际零行必须一致; 不执行DDL、模式切换、回填、旧行删除或任何AWS及Kubernetes mutation
- Safety And Cleanup: Explicit target and current plan; no LIVE executed by this review.
- Necessity: Deployed remote-command state table, explicit migration mode and full dual-write comparison.
- Overlap: classification: added_missing_boundary; rationale: Deployed remote-command state table, explicit migration mode and full dual-write comparison.
- Tests: tests/regional/test_state_table_acceptance.py
- Findings At Review: Approved LIVE validation remains NOT_RUN; local tests are not hardware evidence.
- Reviewer Handoff: Implemented; final local verification is recorded in the verification section.

### GF-REGIONAL-BOOT-025

Historical WIP identity; the current case is
[GF-REGIONAL-BOOT-031](../区域模式端到端验收测试用例.md#gf-regional-boot-031).

部署后workflow专表模式与双写一致性

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: remote_command迁移成功不代表workflow的独立模式、列重建和回填状态正确，必须分别核对。
- Runner: scripts/e2e/regional/run_state_table_acceptance.py
- Entry Points: guarded main -> read-only preflight -> plan/execute -> evidence and cleanup
- Assertions: 正式前置BOOT-024通过，release和cluster身份仍一致; 全部启用CPU副本的workflow数据库摘要、schema版本、模式和revision一致; dual要求workflow回填完成并通过完整payload一致性校验，不以行数相等代替内容相等; 错误模式、未知状态、读副本连接、缺副本、版本漂移和查询失败全部fail closed; 保持只读；不测HOT比例，不自动推进迁移，不清理legacy行或索引
- Safety And Cleanup: Explicit target and current plan; no LIVE executed by this review.
- Necessity: Workflow has its own migration mode/revision and payload projection; remote-command success cannot substitute.
- Overlap: classification: added_missing_boundary; rationale: Workflow has its own migration mode/revision and payload projection; remote-command success cannot substitute.
- Tests: tests/regional/test_state_table_acceptance_postgres.py
- Findings At Review: Approved LIVE validation remains NOT_RUN; local tests are not hardware evidence.
- Reviewer Handoff: Implemented; final local verification is recorded in the verification section.

### GF-REGIONAL-CAP-001

单集群队列打满时返回 429 且不影响其他集群

Current catalog: live-non-destructive; manual; NOT_RUN.

- Purpose: Disposable control plane: A queue cap20, A50QPS/60s, B1QPS. A gets429/Retry-After, B all succeeds within2x baseline p95, global queue stays bounded and only A rejection counter grows.
- Runner: scripts/e2e/regional/run_capacity_acceptance.py; scripts/e2e/regional/capacity_acceptance_base.py; scripts/e2e/regional/capacity_acceptance_cases.py; scripts/e2e/regional/probes/cap_probe_supervisor.py
- Entry Points: Guarded main -> case_001 -> deploy_probe('CAP001'); 15 B-only samples, 3000 A and60 B scheduled POSTs, metrics monitor, two zero-depth drain samples -> cap001_failures; CapHarnessBase.run writes PENDING then final verdict after cleanup and production baseline comparison
- Assertions: A429 present, Retry-After exactly2; B baseline15x202 and storm60x202; A has no integer5xx; B p95 <= factor times baseline; queue_depth<=25, A rejection max>0, B rejection max0, queue drained; Monitor read errors are skipped; missing metric series default0; transport-error strings in A status counts are not rejected by cap001_failures
- Safety And Cleanup: Synthetic registry/disposable database; no intended GPU or production registry mutation; Owned cleanup now tracks partial creation, exact run-case Pod label, foreground delete/absence and database drop; resource names include random entropy; Probe app label matches production worker for ADOT discovery; production Service/selector isolation needs an explicit check; Probe mounts only scripts/work, not live RDS CA bundle required by verify-full DSN
- Necessity: Useful tenancy admission test with a real B baseline. Its custom all-role/spool-disabled probe is not the deployed role-split/spool path, so results cannot establish production SLO. It also accepts direct processor mode even though this queue-cap scenario needs queued admission.
- Overlap: case_ids: GF-REGIONAL-CAP-002; GF-REGIONAL-CAP-003; classification: complementary_capacity_dimensions; rationale: 001 tests per-cluster429 isolation;002 global Store503 backpressure;003 empty-claim scaling. Same disposable harness does not make these duplicate cases.
- Tests: existing: tests/regional/test_capacity_acceptance_harness.py::test_cap001_passes_when_b_stays_within_the_baseline_factor; tests/regional/test_capacity_acceptance_harness.py::test_cap001_flags_b_latency_beyond_the_baseline_factor; tests/regional/test_capacity_acceptance_harness.py::test_cap001_queue_depth_bound_is_the_a_cap_plus_slack; added: tests/regional/test_capacity_acceptance_cleanup.py::test_probe_cleanup_checks_the_label_it_actually_created; tests/regional/test_capacity_acceptance_cleanup.py::test_partial_probe_creation_is_tracked_for_cleanup; tests/regional/test_run_capacity_acceptance.py::test_plan_binds_arguments_and_cannot_override_failed_preflight; assessment: Existing tests exercise pure result verdict, not the 3060-request case body. Shared cleanup regressions are now behaviorally covered.; planned_behavior: Reject unexpected/transport status, nonfinite latency and absent queue metrics; Mock full case scheduling/aggregation and verify all A/B payloads, final verdict and cleanup
- Findings At Review: Fixed: partial-create/label/delete false-cleanup and infinite latency factor; Fixed: schema-3 argument binding and exact preflight bool; Remaining: A transport-status handling, unknown metrics and full-body coverage; Parent: isolated CA/ADOT/Service topology; production role-split/spool gap
- Reviewer Handoff: Owned cleanup, topology isolation and false-PASS verdict defects fixed. Production-topology coverage remains a distinct proposal.

### GF-REGIONAL-CAP-002

Store I/O 饱和时返回 503 且告警链路端到端触发

Current catalog: live-non-destructive; manual; NOT_RUN.

- Purpose: Four Store I/O slots with1s admission timeout,20 concurrent claims; some503/Retry-After2, no500, eventual successful retry, and real GpuFaultStoreIoSaturated firing/resolution.
- Runner: scripts/e2e/regional/run_capacity_acceptance.py; scripts/e2e/regional/capacity_acceptance_base.py; scripts/e2e/regional/capacity_acceptance_cases.py; scripts/e2e/regional/probes/cap_probe_app.py
- Entry Points: case_002_v2 rejects preexisting target alert -> deploy_probe ingress -> _cap002_behavior; Loopback /__cap__/hold starts four Store I/O Event.wait tasks, then HTTP clients issue claims and retry; cap002_alert holds4x600s, queries AMP every15s, releases holds and _cap002_wait_resolved precedes probe deletion
- Assertions: Initial503>0, no500, Retry-After2, all20 retries200, capacity rejection counter>0; Local Store I/O ratio>0.9; Firing comes from global alert_states(alert_name); target AMP vector presence/value is recorded but not required; Resolution requires no firing before delete, but pending/unknown states are treated as resolved
- Safety And Cleanup: Hold controls are loopback-only middleware and do not invoke GPU adapters; Finally releases both hold tags and deletes probe; release errors are swallowed and failed behavior can skip the resolution wait; Owned shared cleanup now rejects unverified resource/database absence; Live alert queries and claims are not executed in this task
- Necessity: Separating behavior and alert phases is reasonable. The case tests manual HTTP retry after release, not production Executor poll/backoff behavior. A global alert unrelated to the probe can satisfy firing because the target query is ignored.
- Overlap: case_ids: GF-REGIONAL-CAP-001; GF-REGIONAL-BOOT-015; classification: complementary_backpressure_and_alert_path; rationale: 001 concerns queue429, not Store503;015 tests unclaimed-command alert. They share AMP transport but have different failure conditions and cleanup.
- Tests: existing: tests/regional/test_capacity_acceptance_harness.py::test_cap002_alert_phase_holds_four_slots; tests/regional/test_capacity_acceptance_harness.py::test_cap002_waits_for_the_alert_to_resolve_before_deleting_the_probe; added: tests/regional/test_capacity_acceptance_cleanup.py::test_common_cleanup_attempts_both_resources_and_reports_errors; assessment: Positive alert test stubs a global firing state and a placeholder query result, so it cannot expose missing target attribution. Behavior/retry phase lacks direct test coverage.; planned_behavior: Empty/zero/NaN target vector plus unrelated firing must fail; Require no pending/firing states on resolution; Exercise20-client status aggregation and retry sequence with MockTransport; Preserve cleanup and resolution evidence on each exception path
- Findings At Review: P1: global firing can pass without target saturation evidence; P1: failed case path may delete the probe before observing alert resolution; P2: manual retry does not prove production Executor poll behavior; Shared: verify-full CA and probe network/selector isolation
- Reviewer Handoff: Owned target-attribution, invalid status and failed-path alert/resource cleanup defects fixed.

### GF-REGIONAL-CAP-003

N 集群规模下的 claim 放大基线

Current catalog: live-non-destructive; manual; NOT_RUN.

- Purpose: Measure empty-claim overhead at N1/5/10/20 with production-equivalent cadence, latency percentiles, CPU/Store wait/Aurora connections; p95<1s/no5xx, connection-budget accounting and actionable poll/batch recommendations.
- Runner: scripts/e2e/regional/capacity_acceptance_cases.py; scripts/e2e/regional/capacity_acceptance_base.py; scripts/e2e/regional/run_capacity_acceptance.py
- Entry Points: case_003 creates one custom probe and runs60 polls per identity for each N; Captures cgroup CPU delta, database connections before/after and Store metric; connection_budget parses role replica/process/pool limits; cap003_knee and cap003_recommendations judge results and suggest polling
- Assertions: First non200 or p95>=1000ms is a knee; all four points must have no knee; Raw CloudWatch CPU/DatabaseConnections datapoints are saved but emptiness/coverage is not required; Missing Store wait metric defaults0; connection_budget counts configured CPU roles but not missing expected roles or effective env overrides
- Safety And Cleanup: One disposable database, synthetic IDs, no GPU workload; Owned shared cleanup now verifies residuals; production raw-DSN queries still need fresh file credentials in connection_budget; HTTP client close is after the whole loop, not finally; a failed request exits through outer resource cleanup; No real load, database or CloudWatch call made here
- Necessity: Measures isolated claim latency, not a fleet-wide limit. It registers20 identities from the beginning and changes active clients, so registry-size scaling is not tested. No N>20 is sampled; no observed beyond-threshold knee should be invented.
- Overlap: case_ids: GF-REGIONAL-CAP-001; GF-REGIONAL-CAP-004; GF-REGIONAL-CAP-005; classification: complementary_scale_baseline; rationale: 001 is event admission,004 nonempty command batch execution,005 transaction correctness. Empty pull traffic is a separate capacity dimension.
- Tests: existing: tests/regional/test_capacity_acceptance_harness.py::test_cap003_without_a_knee_recommends_from_the_largest_tested_count; tests/regional/test_capacity_acceptance_harness.py::test_cap003_knee_at_the_latency_threshold_drives_the_recommendation; tests/regional/test_capacity_acceptance_harness.py::test_cap003_knee_on_a_non_200_answer; assessment: Only pure knee/recommendation logic is tested. Full loop, role budget reads and CloudWatch completeness are uncovered.; planned_behavior: Mock all2160 requests and assert N-specific counts, latencies and cleanup; Reject missing role/pool/metric evidence and nonfinite latency; Include actual executor replica count in headroom recommendation and report BATCH_SIZE; Use isolated credential-file-aware readback for max_connections
- Findings At Review: P1: absent metrics/CloudWatch evidence can look healthy; P2: fixed20 registry rather than N-sized registry; P2: recommendations omit BATCH_SIZE and do not account for two Executor replicas in target headroom; P2: only tests<=20; beyond-threshold knee remains unmeasured
- Reviewer Handoff: Owned resource/readback/measurement false-success defects fixed; no fabricated capacity measurements.

### GF-REGIONAL-CAP-004

大批量command并发执行、续租与吞吐

Current catalog: live-non-destructive; manual; NOT_RUN.

- Purpose: 25 non-destructive commands with production Executor concurrency5, long actions spanning leases, per-command renewal, one renewal failure, no duplicate physical execution and payload/memory/throughput evidence.
- Runner: scripts/e2e/regional/capacity_acceptance_cases.py; scripts/e2e/regional/capacity_acceptance_base.py; scripts/e2e/regional/probes/cap_seed_commands.py
- Entry Points: case_004 seeds25 synthetic commands, manually claims them and calls _cap004_execute_claimed; That function implements its own five-worker ThreadPoolExecutor,25 renewal threads, competitor client and manual result POSTs; No ClusterActionExecutor.run_once or deployed executor code is invoked
- Assertions: Exactly25 claimed, max_active5, >=5 durations>=10s; Some renewal success/failure, one injected403/409, competitor claimed none, no counted competitor transport errors; 25 result200 and final claim empty; each recorded lease progression renewal_count>0; Latest expiry need not exceed initial; empty progression list passes all(); competitor HTTP non200 is ignored; payload bytes have no budget ceiling
- Safety And Cleanup: All action bodies are local sleep and simulated result submission, not physical GPU execution; Synthetic IDs/nonexistent nodes and fake owner; no Node Agent connection; Renewal threads only join5s although HTTP timeout20s; background work may survive into database cleanup; Owned shared cleanup now enforces resource/Pod/database absence
- Necessity: It exercises server claim/renew/result APIs but proves its own hard-coded thread count, not production Executor concurrency/renewal/idempotence. Calling this production execution acceptance is too strong; use the real library with a nonphysical adapter and observable ledger.
- Overlap: case_ids: GF-REGIONAL-CAP-003; GF-REGIONAL-CAP-005; classification: complementary_but_current_runner_scope_mismatch; rationale: 003 empty pulls and005 Store correctness cannot establish real executor behavior. A separate duplicate simulator case would not fill this gap; repair the existing CAP-004 runner.
- Tests: existing: tests/regional/test_capacity_acceptance_harness.py::test_verdict_is_pass_only_when_everything_held; added: tests/regional/test_capacity_acceptance_cleanup.py::test_harness_rejects_a_missing_or_failed_case_result; assessment: No direct case_004 or _cap004_execute_claimed test existed in the frozen suite; generic harness tests cannot detect fake-concurrency or expiry-proof gaps.; planned_behavior: Require unique complete per-command expiry progress with later timestamps; Treat competitor403/503/malformed responses as unknown duplicate proof and FAIL; Verify worker/renewal shutdown before teardown; Use actual ClusterActionExecutor with fake nonphysical adapter; assert action ledger and state transitions
- Findings At Review: P1: missing real production Executor runner; P1: competitor errors/lease progress can be silently unproven; P1: renewal thread completion not established before cleanup; P2: response byte budget and exactly-once ledger not asserted
- Reviewer Handoff: Owned cleanup and incomplete-proof bugs fixed; production-Executor gap is fail-closed in code, not merely documented. Formal CAP004 remains blocked on BC-PROP-005.
- Current Disposition: Production Executor/client and nonphysical ledger; API25 measurement/WAITING handback is separate from batch5/concurrency5. At least five uncached successful actions exceed their original lease. Terminal renewal rejections are reconciled only against the same acknowledged completion lease, including reversed ACK arrival. 31 core-instrumented local tests and one real PostgreSQL test passed.

### GF-REGIONAL-CAP-005

生产 PostgreSQL 测试零 skip 门禁

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: PostgreSQL16 isolated database, full Store/claim/lease/reconnect/cleanup/migration stress at8x40, completion concurrency2/4, zero JUnit skips/failures and no database residue. New user constraint explicitly forbids Aurora stress.
- Runner: scripts/e2e/regional/run_cap005_postgres_suite.py
- Entry Points: main -> run_suite -> validate_database_url -> validate_server -> unique database and per-run JUnit directory; _run_pytest invokes make test-postgres-stress and Store contract using current interpreter, with xdist workers0; finally terminates connections, drops database and verifies absence
- Assertions: Reject before connect unless PostgreSQL URL uses loopback host, explicit port and /postgres with no target override query; Server version must16; aurora_version function refuses even a loopback Aurora tunnel; Child receives only generated GPU_FAULT_TEST_POSTGRES_URL plus scrubbed AWS/Kubernetes/libpq environment; Each JUnit must exist, contain tests>0, failures/errors/skips0; both exit codes0; database absence verified
- Safety And Cleanup: The original intent-before-CREATE handling could delete a database without proven creation. Follow-up cleanup requires acknowledged creation; a definite refusal or uncertain acknowledgement preserves the database and remains non-PASS. Exception types omit credential-bearing text, and fresh per-run report directories prevent stale JUnit reuse. Original review connections and subprocesses were mocked.
- Necessity: Essential production-backend correctness gate, but database-level isolation alone never authorized stress on an Aurora writer. The new loopback/server guard meets the explicit task boundary; actual server allocation remains an operator/parent prerequisite.
- Overlap: case_ids: GF-REGIONAL-CAP-001; GF-REGIONAL-CAP-002; GF-REGIONAL-CAP-003; GF-REGIONAL-CAP-004; classification: complementary_backend_gate; rationale: Real transaction tests support all capacity cases but do not measure their live latency or alerts. Do not run against a shared Aurora endpoint under the assumption an independent database isolates server load.
- Tests: existing: tests/regional/test_cap005_postgres_suite.py::test_a_red_contract_suite_still_reports_its_junit_counts; tests/regional/test_cap005_postgres_suite.py::test_a_drop_failure_is_merged_into_the_report; tests/store/test_postgres_processor_claim.py::test_completion_cluster_concurrency_runs_real_transactions_in_parallel; safety: tests/regional/test_cap005_safety.py::test_remote_or_redirected_database_is_rejected_before_connect; tests/regional/test_cap005_safety.py::test_create_ack_loss_preserves_unproven_database_ownership; tests/regional/test_cap005_safety.py::test_child_cannot_inherit_another_store_or_cloud_environment; tests/regional/test_cap005_safety.py::test_stale_junit_cannot_substitute_for_this_run; tests/regional/test_cap005_safety.py::test_server_must_be_postgres16_and_not_an_aurora_tunnel; follow-up: tests/regional/test_boot_cap_postgres_isolation.py::test_real_cap005_creation_refusal_preserves_the_existing_database; the original review had 21 passing targeted tests; that result does not verify later source changes. The current acknowledgement-loss contract preserves unproven ownership instead of deleting by intent.
- Findings At Review: Fixed: remote/Aurora authorization, inherited DSN-file/credential routing, stale JUnit, creation ACK-loss and missing exception report; Remaining: actual isolated PostgreSQL16 stress not run; Parent: catalog/spec must remove Aurora as an allowed stress endpoint; Parent: JUnit should prove required test identities/areas without freezing a numeric count
- Reviewer Handoff: Owned safety and real local PostgreSQL lifecycle validation complete; no stress-suite or live acceptance claim.
- Current Disposition: The full local PostgreSQL group passed 1533 tests. Three inapplicable Memory/SQLite variants of PostgreSQL-only assertions are no longer collected; the two PostgreSQL assertions were rerun with zero skips. Applicable collection remains 1533. Generic cross-backend contracts and the strict local-server/zero-skip runner policy are preserved.
- Integration Follow-Up: The direct contract child uses a complete pytest receipt, explicit JUnit/serial arguments and a validated generated-database opt-in. Make/stress also strips inherited filters and worker identities. Refused or uncertain CREATE does not authorize DROP. The preceding PostgreSQL count is historical, not fresh verification of these changes.

### GF-REGIONAL-CMD-001

claim 的 max_commands 边界值

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: API max_commands bounds 0/1/25/26/-1/string5/5.5; return granted leases as WAITING.
- Runner: scripts/e2e/regional/audit_regional_command_protocol_live.py
- Entry Points: manual catalog; LiveProtocolAudit.run_001 via batch main/run; not a pytest wrapper
- Assertions: 422 for invalid values; 200 with bounded returned count; IDs belong to three seeded commands; hand-back returns 200
- Safety And Cleanup: Test-only owner and empty-queue/executor assertion preflight; per-case object deletion; PROTO-005 and CENTRAL-003 affect batch cleanup/order
- Necessity: Useful deployed API validation; three seeded rows cannot prove the 25-command cap under a backlog larger than 25, and empty 200 responses can pass.
- Overlap: case_ids: GF-REGIONAL-CMD-002; classification: shared_helper_distinct_scenario; rationale: Same claim endpoint but independent batch-size and lease-window constraints.
- Tests: tests/regional/test_regional_command_protocol_acceptance.py::test_cmd001_max_commands_is_enforced_at_both_ends_of_its_range; tests/regional/test_command_protocol_audit_contracts.py::test_cmd001_hands_back_every_lease_as_waiting_and_seeds_only_its_owner; tests/regional/test_command_protocol_audit_contracts.py::test_cmd001_refuses_a_lease_on_a_command_it_did_not_seed
- Findings At Review: Runner does not prove 25 cap with 26+ backlog; product proxy does; Failure paths can lease foreign commands before refusing and do not hand those leases back
- Reviewer Handoff: fixed_owned; exact_HTTP_codes_IDs_counts_and_26_command_backlog_locally_tested

### GF-REGIONAL-CMD-002

claim 的 lease_seconds 边界值

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: Lease bounds 10..7200, deadline within 3 seconds, stale short lease completion rejected.
- Runner: scripts/e2e/regional/audit_regional_command_protocol_live.py
- Entry Points: manual catalog; LiveProtocolAudit.run_002
- Assertions: 9/7201/0 -> 422; 10/600/601/7200 -> one lease, before/after deadline bounds; 11-second runner wait after 10-second lease -> 409
- Safety And Cleanup: Valid leases handed back; final expired record deleted by batch cleanup; Common audit preflight and PROTO-005 apply
- Necessity: Distinct lease-duration boundary proof; 11-second wait proves expiry but differs from documented 15 seconds.
- Overlap: case_ids: GF-REGIONAL-CMD-006; GF-REGIONAL-NET-002; classification: partial_overlap_complementary; rationale: CMD006 covers ownership replacement; NET002 adds transport outage/reclaim. Short-lease expiry assertion is intentionally repeated.
- Tests: tests/regional/test_regional_command_protocol_acceptance.py::test_cmd002_lease_seconds_bounds_and_the_deadline_it_returns
- Findings At Review: No direct run_002 behavioral test for correct ID, deadline or cleanup failure; Returned command count checked but returned ID not explicitly bound to seed
- Reviewer Handoff: fixed_owned; exact_claim_and_result_contract_locally_tested

### GF-REGIONAL-CMD-003

execution_owners 校验与空列表 fail-closed 回退

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: Validate owners and prove empty owner list only falls back to Kubernetes adapter.
- Runner: scripts/e2e/regional/audit_regional_command_protocol_live.py
- Entry Points: manual catalog; LiveProtocolAudit.run_003
- Assertions: Blank/whitespace/duplicate/33 owners rejected; 32 accepted; Empty owners excludes node-agent and includes exactly the seeded Kubernetes command
- Safety And Cleanup: Seeds real owner names; production claimant exclusion is critical; Only target cluster empty-queue assertion, no live verification of pod deletion; PROTO-005
- Necessity: Both negative and positive default-owner controls are useful and not duplicate with explicit-owner filtering.
- Overlap: case_ids: GF-REGIONAL-CMD-004; classification: shared_helper_distinct_scenario; rationale: Default fallback and input validation vs explicit owner partition.
- Tests: tests/regional/test_regional_control_plane.py::test_remote_claim_filters_execution_owners_and_legacy_api; tests/regional/test_command_protocol_audit_contracts.py::test_claim_defaults_to_the_test_only_owner
- Findings At Review: No direct run_003 matrix/failure-path fake test; Ready replicas=0 alone does not establish no terminating claimant
- Reviewer Handoff: fixed_owned_claim_validation; deployed_claimant_quiescence_is_a_separate_limit

### GF-REGIONAL-CMD-004

只领取匹配 owner 的 command

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: Three adapter owners partition into node-agent only then Kubernetes plus HyperPod, no duplicate claim.
- Runner: scripts/e2e/regional/audit_regional_command_protocol_live.py
- Entry Points: manual catalog; LiveProtocolAudit.run_004
- Assertions: First returned owners exactly [node-agent]; Second owners compared as a set to Kubernetes/HyperPod
- Safety And Cleanup: Real owner commands need production claimant exclusion; Hand-back and common object cleanup
- Necessity: Valid routing test but runner set comparison can accept duplicate second-batch rows and does not bind claims to the three seed IDs.
- Overlap: case_ids: GF-REGIONAL-CMD-003; classification: partial_overlap_complementary; rationale: Both owner filtering; this adds explicit multi-owner partition and uniqueness.
- Tests: tests/regional/test_regional_command_protocol_acceptance.py::test_cmd004_a_claim_only_takes_commands_for_the_owners_it_asked_for
- Findings At Review: Missing runner assertion for exact IDs, batch length and no duplicates; No direct runner fake covering duplicate/foreign rows
- Reviewer Handoff: fixed_owned; exact_owner_ID_count_and_duplicate_refusal

### GF-REGIONAL-CMD-005

领取顺序为 FIFO

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: FIFO by created_at and deterministic command_id tie-break.
- Runner: scripts/e2e/regional/audit_regional_command_protocol_live.py
- Entry Points: manual catalog; LiveProtocolAudit.run_005
- Assertions: Five constructed timestamps plus reverse-seeded equal-time pair; exact returned ID sequence
- Safety And Cleanup: Test-only owner; all seven leased rows handed back and deleted
- Necessity: Constructing distinct timestamps avoids unnecessary live waits while testing the same persisted ordering contract.
- Overlap: case_ids: GF-REGIONAL-CMD-001; classification: shared_helper_distinct_scenario; rationale: Same claim batch helper, ordering vs capacity boundary.
- Tests: tests/regional/test_regional_command_protocol_acceptance.py::test_cmd005_the_claim_order_is_fifo_and_ties_break_deterministically
- Findings At Review: Direct run_005 wrong-order/tie/missing-row tests absent; Real PostgreSQL concurrency/SQL mode proof not run in this task
- Reviewer Handoff: reviewed_and_locally_tested; FIFO_contract_unchanged

### GF-REGIONAL-CMD-006

lease 过期后可被重新领取且只有新持有者能写结果

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: WAITING hand-back, lease replacement invalidates A1, A2 expires, A3 alone completes.
- Runner: scripts/e2e/regional/audit_regional_command_protocol_live.py
- Entry Points: manual catalog; LiveProtocolAudit.run_006
- Assertions: A1 WAITING=200; old A1=409; expired A2=409 after15s; A3 SUCCEEDED and last_lease_owner=A3
- Safety And Cleanup: Test-only owner and synthetic node; command removed after case
- Necessity: Useful three-owner protocol sequence; it proves no physical action uniqueness by itself.
- Overlap: case_ids: GF-REGIONAL-CMD-002; GF-REGIONAL-NET-002; GF-REGIONAL-NET-006; classification: partial_overlap_complementary; rationale: Shared lease rules; network cases separately exercise reporting/reclaim under transport failures.
- Tests: tests/regional/test_regional_control_plane.py::test_remote_command_lease_and_result_advance_adapter; tests/regional/test_regional_acceptance_fixtures.py::test_cmd006_fixture_waits_for_the_documented_lease_expiry
- Findings At Review: Runner does not assert successful claim status/exact seeded ID before indexing; Existing direct fake only checks duration and canned completions, not real transitions
- Reviewer Handoff: fixed_owned; exact_claim_and_result_contract_locally_tested

### GF-REGIONAL-CMD-007

重复提交终态结果幂等且不可覆盖

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: Terminal result immutable across same-token, wrong-token and contradictory FAILED replay.
- Runner: scripts/e2e/regional/audit_regional_command_protocol_live.py
- Entry Points: manual catalog; LiveProtocolAudit.run_007
- Assertions: Initial completion=200; three replay bodies exactly equal initial body
- Safety And Cleanup: Test-only owner; terminal command deleted; Assertion error currently interpolates the request payload including a lease token
- Necessity: Distinct terminal protocol matrix; live runner does not explicitly require initial SUCCEEDED/error=null, so consistent wrong response could pass.
- Overlap: case_ids: GF-REGIONAL-NET-003; GF-REGIONAL-NOTIFY-001; classification: partial_overlap_complementary; rationale: NET003 adds response-loss transport; NOTIFY001 owns actual notification dedup. Retired NOTIFY002 is not a runnable predecessor.
- Tests: tests/regional/test_regional_command_protocol_acceptance.py::test_cmd007_a_terminal_result_is_idempotent_and_cannot_be_rewritten; tests/regional/test_command_protocol_audit_contracts.py::test_cmd007_records_the_replay_statuses_it_observed_and_fails_on_a_rewrite
- Findings At Review: Initial status/error assertion absent in runner; No notification side effect proof; spec references retired NOTIFY002; No failure-output credential-redaction test
- Reviewer Handoff: fixed_owned; terminal_status_immutability_and_sanitized_errors_locally_tested

### GF-REGIONAL-CMD-008

RemoteCommandResult 的取值约束

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: Reject non-result statuses, missing error/token and extra fields without changing the command; accept WAITING.
- Runner: scripts/e2e/regional/audit_regional_command_protocol_live.py
- Entry Points: manual catalog; LiveProtocolAudit.run_008
- Assertions: Six invalid payloads ->422 and stored status remains LEASED; Final WAITING ->200 with WAITING body
- Safety And Cleanup: Invalid requests do not intentionally mutate; legal hand-back then object cleanup
- Necessity: Appropriate payload boundary test. Snapshot comparison would prove unchanged state more strongly than status alone.
- Overlap: case_ids: GF-REGIONAL-CMD-007; classification: shared_helper_distinct_scenario; rationale: Input schema validation before transition vs terminal idempotence after transition.
- Tests: tests/regional/test_regional_command_protocol_acceptance.py::test_cmd008_an_illegal_result_is_rejected_and_changes_nothing
- Findings At Review: Runner does not compare lease/result fields after422; No direct runner fake for field mutation while status stays LEASED
- Reviewer Handoff: fixed_owned; invalid_results_preserve_full_command_record

### GF-REGIONAL-CMD-009

不存在的与跨集群的 command_id 都返回 404

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: Unknown and foreign command IDs yield indistinguishable not-found contract; foreign record unchanged.
- Runner: scripts/e2e/regional/audit_regional_command_protocol_live.py
- Entry Points: manual catalog; LiveProtocolAudit.run_009
- Assertions: Both404; erroneously compares literal detail equality across different IDs; Foreign status remains PENDING
- Safety And Cleanup: Foreign record uses synthetic test owner; created in other cluster; deleted by batch cleanup
- Necessity: Security scenario is necessary; PROTO-001 makes current runner reject the correct API.
- Overlap: case_ids: GF-REGIONAL-ISO-002; GF-REGIONAL-CMD-011; classification: different_layer; rationale: API information disclosure vs executor-local identity/fence refusal.
- Tests: tests/regional/test_regional_command_protocol_acceptance.py::test_cmd009_an_unknown_and_a_foreign_command_are_the_same_404; tests/regional/test_command_protocol_audit_contracts.py::test_cmd009_requires_the_unknown_and_foreign_404_to_be_indistinguishable
- Findings At Review: PROTO-001; Direct audit test covers distinguishable failure only, not healthy caller-derived404 success
- Reviewer Handoff: fixed_owned; caller_derived_404_contract_ASGI_tested

### GF-REGIONAL-CMD-010

fencing token 陈旧的 command 既不可领取也不可完成

Current catalog: live-non-destructive; manual; NOT_RUN.

- Purpose: Stale fencing command cannot overwrite current workflow or be reclaimed.
- Runner: scripts/e2e/regional/audit_regional_command_protocol_live.py
- Entry Points: manual catalog; LiveProtocolAudit.run_010
- Assertions: Advances persisted workflow+incident generation; expects obsolete409/LEASED; checks no reclaim and no registered probe agent
- Safety And Cleanup: Synthetic node; future not_before; object cleanup omits incident event link
- Necessity: Required fence defense but current acceptance semantics are stale; do not alter runtime safety to satisfy them.
- Overlap: case_ids: GF-REGIONAL-CMD-006; GF-REGIONAL-CMD-011; classification: different_fences; rationale: Workflow-generation fence vs lease-owner fence vs embedded executor three-way fence.
- Tests: tests/store/test_remote_command_stale_fence.py; tests/regional/test_command_protocol_audit_contracts.py (no direct run_010 success regression)
- Findings At Review: PROTO-002; PROTO-005; No catalog related_pytest entry for current stale-fence suite
- Reviewer Handoff: fixed_owned; stale_fence_FAILED_and_late_result_ASGI_tested; parent_spec_sync

### GF-REGIONAL-CMD-011

executor 本地校验 fencing token 三方一致

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: Executor rejects mismatched command/workflow/incident fencing locally before adapter execution.
- Runner: scripts/e2e/regional/audit_executor_local_guards.py
- Entry Points: catalog command -> main/evaluate -> constructed ClusterActionExecutor._execute; entirely local
- Assertions: FAILED/executor-rejected and exact fence-mismatch reason; unexpected_failures=0
- Safety And Cleanup: No Store, no real adapters, no cluster client; only optional evidence files
- Necessity: Appropriate isolated malformed-command proof, not live deployment proof. Only incident differs in the current fixture.
- Overlap: case_ids: GF-REGIONAL-ISO-002; classification: shared_runner_distinct_scenario; rationale: Same two-command fixture separately tests cluster mismatch and fence mismatch, not duplicate coverage.
- Tests: tests/regional/test_command_protocol_audit_contracts.py::test_local_guard_fixture_writes_evidence_for_iso002_and_cmd011; tests/regional/test_command_protocol_audit_contracts.py::test_local_guard_fixture_reports_a_guard_that_stopped_refusing; tests/regional/test_regional_acceptance_fixtures.py::test_local_executor_guard_fixture_covers_iso002_and_cmd011
- Findings At Review: Each fence mismatch permutation and valid control not exercised by this fixture; Writes both case documents even for a selected single case; coordinate evidence scoping with parent
- Reviewer Handoff: reviewed_and_locally_tested; no_hardware_proof_claim

### GF-REGIONAL-CMD-012

command 身份幂等与冲突拒绝

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: Retry command identity stable; changed generation mints a distinct remote-24hex ID; conflicts rejected.
- Runner: scripts/e2e/regional/audit_regional_command_protocol_live.py
- Entry Points: manual catalog; run_012 directly invokes deployed RegionalRemoteWorkflowAdapter, not dispatcher retry
- Assertions: Three executions reuse first ID and created_at; bumped fence gets different valid24hex ID
- Safety And Cleanup: Node-agent owner on nonexistent node; command cleanup; Adapter-created command tracking occurs after returned outcome, leaving ACK-loss gap
- Necessity: Runner transparently labels dispatcher retry as NOT_EXERCISED; digest includes step content, not only documented triple.
- Overlap: case_ids: GF-REGIONAL-CMD-018; classification: partial_overlap_complementary; rationale: Identity reuse/fence change vs changed-step open-sibling suppression.
- Tests: tests/regional/test_regional_command_protocol_acceptance.py::test_cmd012_the_command_identity_is_reused_on_retry_and_new_on_a_fence; tests/regional/test_regional_command_protocol_acceptance.py::test_cmd012_a_reused_command_id_with_different_content_is_refused
- Findings At Review: No deliberate identity-conflict branch in live audit; No actual dispatcher retry or log scan; recorded limitation covers only first; Central docs omit full-step digest input
- Reviewer Handoff: reviewed_and_locally_tested; cleanup_identity_tracked; parent_documentation_proposal

### GF-REGIONAL-CMD-013

WAITING 可重新领取并携带上一轮结果

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: Twenty WAITING reclaims preserve prior result_details and final SUCCEEDED.
- Runner: scripts/e2e/regional/audit_regional_command_protocol_live.py
- Entry Points: manual catalog; LiveProtocolAudit.run_013
- Assertions: Exactly one claim per round; previous details survive; final round20 details and SUCCEEDED
- Safety And Cleanup: Synthetic command with test-only owner; terminal row cleanup
- Necessity: Useful command-level progress test. It does not prove workflow lifetime is unbounded, because no persisted live workflow is driven.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-032; classification: complementary_lifetime_layers; rationale: Per-command progress can be unbounded in rounds while workflow deadline still cancels it.
- Tests: tests/regional/test_regional_command_protocol_acceptance.py::test_cmd013_waiting_can_be_reclaimed_and_carries_the_previous_details
- Findings At Review: No direct runner success/missing-details fake; Catalog/spec statement that workflow can hang forever contradicts implemented lifetime deadline
- Reviewer Handoff: reviewed_and_locally_tested; parent_must_distinguish_command_rounds_from_workflow_deadline

### GF-REGIONAL-CMD-014

控制面在委托期间明确声明未由控制面提交变更

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: Persisted delegated WAITING step explicitly records no control-plane mutation and remote identity/status.
- Runner: scripts/e2e/regional/audit_regional_command_protocol_live.py
- Entry Points: manual catalog; run_014 persists pair, uses ProductionWorkflowExecutor.execute, reads Store
- Assertions: Exactly one persisted step0; WAITING; valid remote ID; cluster/status/no-CPU-mutation fields
- Safety And Cleanup: Future not_before, synthetic node; created workflow/incident/command removed; PROTO-005 event-link and failed-command-tracking gaps
- Necessity: Stronger than in-memory adapter proxy, but reads Store rather than documented workflow GET API.
- Overlap: case_ids: GF-REGIONAL-CMD-012; classification: shared_helper_distinct_scenario; rationale: Persistent audit metadata vs command identity.
- Tests: tests/regional/test_regional_command_protocol_acceptance.py::test_cmd014_the_delegating_step_states_the_control_plane_did_not_mutate
- Findings At Review: No direct live-runner fake exercising ProductionWorkflowExecutor/persisted details; Exception after command creation before tracking can leave untracked row
- Reviewer Handoff: fixed_owned; persisted_delegation_and_ACK_loss_cleanup_locally_tested

### GF-REGIONAL-CMD-015

restart 预算在委托前被预留且耗尽时不下发 command

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: Claim preflight exhausts restart budget; dispatch without reservation and missing safety context fail before command creation.
- Runner: scripts/e2e/regional/audit_regional_command_protocol_live.py
- Entry Points: manual catalog; run_015 -> reserve_job_restart/reserve_restart_budgets/RegionalRemoteWorkflowAdapter
- Assertions: Exhausted1/1 with unchanged sole reservation; unreserved dispatch fails; missing source_gpu_count fails; no commands across synthetic incidents
- Safety And Cleanup: Synthetic job and nonexistent workload; budget deletion in finally; leaked commands tracked only after all assertions
- Necessity: Good separation of sole reservation writer and read-only issuance; not workload restart proof.
- Overlap: case_ids: GF-REGIONAL-CMD-012; GF-REGIONAL-PREEMPT-025; classification: different_contract; rationale: Restart authorization budget vs identity and completion causal acknowledgement.
- Tests: tests/execution/test_restart_budget_preflight.py (to resolve exact matching test coverage); tests/regional/test_regional_control_plane.py (restart authorization coverage, follow-up required)
- Findings At Review: No direct run_015 behavioral fake located; Positive one-reservation path not exercised here; If refusal incorrectly creates command then earlier expect raises, leak discovery is skipped
- Reviewer Handoff: fixed_owned_failure_intent_tracking; restart_budget_refusals_ASGI_tested

### GF-REGIONAL-CMD-016

HyperPod 提交幂等记录的 reserve/get/outcome 协议边界

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: HyperPod submission reserve/get/outcome idempotence, cluster binding, missing records and anonymous refusal, no provider mutation.
- Runner: scripts/e2e/regional/audit_regional_command_protocol_live.py
- Entry Points: manual catalog; run_016 -> run_hyperpod_submission_case -> deployed HTTP endpoints
- Assertions: First/duplicate reserve; foreign/mismatched403; missing GET twice null; SUBMITTED; invalid outcome409; anonymous401/403
- Safety And Cleanup: No provider client invoked by the audit; Submission record is not tracked or cleaned; PROTO-005
- Necessity: Useful database-backed protocol audit. It marks case PASS with CloudTrail NOT_EVALUATED, so cannot establish the full catalog provider-negative assertion.
- Overlap: case_ids: GF-REGIONAL-CMD-007; GF-REGIONAL-AUTH-014; classification: different_resource_same_patterns; rationale: Submission-record idempotence and auth extend, not duplicate, command terminal and route-bucket tests.
- Tests: tests/regional/test_regional_control_plane.py::test_remote_hyperpod_submission_endpoints_enforce_cluster_binding
- Findings At Review: Missing owned submission cleanup/ACK-loss tests; CloudTrail reason lists UpdateClusterSoftware instead of full reboot/replace/delete set; No comprehensive run_hyperpod_submission_case fake matrix located
- Reviewer Handoff: fixed_owned; exact_401_and_submission_cleanup_ASGI_tested; provider_proof_NOT_EVALUATED

### GF-REGIONAL-CMD-017

多节点 barrier 命令在区域 executor 的 claim 边界被 fail-closed 持有，计数且永不到达适配器

Current catalog: live-non-destructive; manual; NOT_RUN.

- Purpose: Synthetic two-node barrier command held WAITING at deployed executor claim boundary, never adapter-executed.
- Runner: scripts/e2e/regional/run_cmd017_barrier_hold.py; scripts/e2e/regional/cmd017_verdicts.py; scripts/e2e/regional/seeded_command_fixture.py; scripts/e2e/regional/probes/cmd017_barrier_executor.py
- Entry Points: manual; PlainCaseRunner main -> plan/authorize -> run_case; probe ClusterActionExecutor.run
- Assertions: >=2 holds, reason/source/operation/nodes, counters+breadcrumb, log, no marker, final nonterminal and Running probe
- Safety And Cleanup: Synthetic cluster, no agents, stand-in adapter, pod deadline and registry expiry; Finally uses seeded.cleanup; PROTO-004 and CENTRAL-002 apply
- Necessity: Appropriate safe deployed-executor defense proof, not real barrier coordination or physical reset proof.
- Overlap: case_ids: GF-REGIONAL-NET-006; GF-REGIONAL-CMD-018; classification: shared_fixture_distinct_scenario; rationale: Same isolated seed/probe infrastructure; barrier refusal vs lease loss vs sibling hold.
- Tests: tests/regional/test_cmd017_barrier_hold.py; tests/execution/test_cluster_executor_lease_guard.py::test_regional_wiring_either_coordinates_or_refuses_barrier_operations
- Findings At Review: final_command_errors accepts absent/unknown status; seed_errors treats missing registered_agents as empty and does not bind cluster/two distinct exact nodes; Missing orchestration/finally failure injection tests; tests mostly pure verdicts; Marker read errors can be interpreted as no marker
- Reviewer Handoff: fixed_owned_unknown_state_scoped_cleanup_and_UID_receipts; perf_registration_ACK_CAS_boundary_verified_with_fakes

### GF-REGIONAL-CMD-018

同一 workflow step 的开放兄弟命令在飞时，参数被改写后的再次派发被持有为 WAITING、计数，节点只执行一次

Current catalog: live-non-destructive; manual; NOT_RUN.

- Purpose: Changed step held behind open sibling A; A local ledger executes once; after A terminal new B minted and cancelled before claim.
- Runner: scripts/e2e/regional/run_cmd018_open_sibling_hold.py; scripts/e2e/regional/cmd018_verdicts.py; scripts/e2e/regional/seeded_command_fixture.py; scripts/e2e/regional/probes/cmd018_ledger_executor.py
- Entry Points: manual; PlainCaseRunner -> run_case -> in-Pod dispatch/hold -> ledger probe -> release/cancel
- Assertions: WAITING OPEN_SIBLING_COMMAND points to A/different held ID; one open command; count1; ledger1; A success; B distinct and cancelled; metric family
- Safety And Cleanup: Synthetic cluster/nodes, seed lease, local ledger; stop probe before B is intended but delete uses check=False; Mode-aware existing SQL preserved; PROTO-003/004 and CENTRAL-002 apply
- Necessity: Distinct open-sibling protection. In-Pod adapter counter plus exported metric family does not prove the live dispatcher's own increment; limitation already documented.
- Overlap: case_ids: GF-REGIONAL-CMD-012; GF-REGIONAL-NET-003; classification: partial_overlap_complementary; rationale: Step-content identity/sibling gate differs from stable identity and terminal response replay.
- Tests: tests/regional/test_cmd018_open_sibling_hold.py; tests/store/test_postgres_find_open_remote_command.py
- Findings At Review: PROTO-003 cleanup tracking after last stage only; No B lease_owner/last_lease_owner check; cancelled command ID not bound to minted B in verdict; Probe-ready identity unchecked; ledger snapshot may precede latest recorder tick; No fake orchestration test for failed dispatch/probe/delete/release
- Reviewer Handoff: fixed_owned_seed_intent_A_tracking_and_UID_bound_stop_before_purge; real_SQL_verified_legacy_dual_dedicated

### GF-REGIONAL-COLLECT-001

edge filter 在真机上确实生效的稳态抑制比

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: Manual, read-only-signal-replay: measure deployed DCGM/host edge-filter cadence for two live summary periods and reject routine-summary persistence or invented recovery edges.
- Runner: scripts/e2e/regional/run_collector_acceptance.py; scripts/e2e/regional/collector_acceptance_fixture.py; scripts/e2e/regional/probes/collector_node_probe.py
- Entry Points: read_only_preflight; execute_case; run_collect001; collector_statuses; recent_evidence
- Assertions: Uses each chain's live interval/summary, batch prefixes, deduplicated observed_at, >=2 deliveries, <=duration//summary+1, and max gap >=summary-2*interval.; Rejects persisted health-summary-only records and recovered: resource reasons.; Does not assert either EDGE_FILTER_ENABLED flag or establish the healthy-resource premise used by the recovery-edge check.
- Safety And Cleanup: No fault injection; execution still creates a privileged host probe and reads an allowlisted collector.env projection.; Probe finally is present, but a post-handler cpu_blast_snapshot exception can retain PASS (COLLECT-REVIEW-003).
- Necessity: Useful deployed-config evidence, not equivalent to a local filter test. Positive, discriminating timing and a healthy opening state must be proved; summary equal to interval cannot demonstrate suppression.
- Overlap: case_ids: GF-REGIONAL-COLLECT-002; GF-REGIONAL-COLLECT-003; GF-REGIONAL-COLLECT-019; classification: complementary; rationale: Routine suppression, anomaly delivery, inventory consistency, and error visibility are distinct claims despite reusable baseline reads.
- Tests: selected: tests/collectors/test_gpu.py::test_dcgm_edge_filter_suppresses_unchanged_healthy_samples; tests/collectors/test_host.py::test_host_edge_filter_suppresses_health_and_delivers_edges; runner_coverage: tests/regional/test_collector_runner_contracts.py::test_collect001_judges_each_chain_by_its_own_interval_and_summary; observation: Catalog related_pytest is only registry membership. Missing behavioral negatives for disabled filters, unhealthy opening state, nondiscriminating intervals, and final blast-read failure.; execution: Not run during parent baseline measurement.
- Findings At Review: No current-finding or collector-error health baseline.; Only dcgm- is accepted; require the deployed mode explicitly instead of timing out on nvidia-smi mode.; The shared latest HOST_TELEMETRY row can hide host deliveries overwritten by k8s-efa-.; Evidence scans are bounded without a truncation indication for absence claims.
- Reviewer Handoff: Bounded shared collector safety fixed: final blast-read failures force FAIL, finite intervals and controlled HostProbe state are enforced. Healthy-cadence premise and lossless producer history remain explicit acceptance-coverage proposals (COLLECT-PARENT-005), not claimed fixed.

### GF-REGIONAL-COLLECT-002

edge filter异常沿必须在确认样本数加一个采集周期内投递

Current catalog: live-non-destructive; manual; NOT_RUN.

- Purpose: Manual, live-non-destructive: lower enforced power limits, load one GPU, require a correlated anomaly within interval*(confirmation+1), and restore all limits.
- Runner: scripts/e2e/regional/run_collector_acceptance.py; scripts/e2e/regional/probes/collector_node_probe.py
- Entry Points: run_collect002; gpu_metrics_stamp; throttle_gpu; restore_gpu_power_limit
- Assertions: Aligns to a changed dcgm- status and uses the live confirmation count for latency.; Requires some candidate/threshold GPU_METRICS evidence and equal common default/limit sets after restoration.; Does not bind the candidate to the loaded UUID, power_limit_correlation, or the delivered batch.
- Safety And Cleanup: Probe arms restore before capping, but throttle-gpu is outside the runner cleanup try.; restore_gpu_power_limit disarms the deadman before defaults/query/set/readback can fail.; Generic preflight does not reject active workloads although every GPU's power limit is changed.; Baseline custom limits are replaced by driver defaults without a baseline-mode precondition.
- Necessity: A real software-controlled throttle/load test, not hardware damage. Idle-node authorization, baseline preservation, and explicit partial-failure rollback are required.
- Overlap: case_ids: GF-REGIONAL-COLLECT-001; GF-REGIONAL-COLLECT-019; classification: complementary; rationale: An anomaly edge is not covered by healthy cadence or tool-timeout reporting; quiet-phase setup can be reused.
- Tests: selected: tests/collectors/test_gpu.py::test_dcgm_edge_filter_emits_counter_delta_and_candidate_recovery; runner_coverage: tests/regional/test_collector_and_multicluster_fixtures.py::test_collector_probe_arms_the_power_limit_restore_before_capping; tests/regional/test_collector_runner_contracts.py::test_collect002_restores_the_power_limit_and_keeps_the_original_error; tests/regional/test_collector_runner_contracts.py::test_collect002_reports_a_failed_restore_as_a_case_error; observation: Existing cleanup tests start after successful throttle. No partial-throttle, deadman-disarm-before-failed-restore, custom-baseline, or unrelated-candidate regression.; execution: Not run.
- Findings At Review: Negative/stale delivery latency is not rejected.; UUID membership/count is not compared before and after.; Load safety compares load_seconds, not the subprocess timeout upper bound load_seconds+60.; No deadline recheck after waiting a summary interval before mutation.
- Reviewer Handoff: Fixed partial throttle rollback, default-limit prerequisite, arm-before-cap, verify-before-disarm, negative latency and admission after summary wait. Unit failure/abort/deadman tests pass. Candidate-to-loaded-UUID/batch correlation remains a documented coverage proposal.

### GF-REGIONAL-COLLECT-003

EXPECTED_GPU_COUNT 与 EXPECTED_EFA_DEVICE_COUNT 真机核对

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: Manual, read-only-signal-replay: actual GPU and active EFA counts must match collector.env, with no persistent inventory-mismatch false alarm.
- Runner: scripts/e2e/regional/run_collector_acceptance.py; scripts/e2e/regional/probes/collector_node_probe.py
- Entry Points: run_collect003; gpu_inventory; efa_inventory; collector_setting
- Assertions: Compares GPU inventory length to EXPECTED_GPU_COUNT.; EFA probe correctly requires efa driver and active port state=4/phys_state=5, including the documented missing-phys_state default.; No control-plane query checks the third required fact: absence of a persistent mismatch finding.
- Safety And Cleanup: No injection; probe finally is present.; Generic preflight does not prove collector freshness or absence of mismatch incidents.
- Necessity: A cheap model/configuration consistency check, but the implementation verifies only two of the specified three facts.
- Overlap: case_ids: GF-REGIONAL-COLLECT-001; GF-REGIONAL-COLLECT-004; GF-REGIONAL-COLLECT-020; classification: shared_preflight_not_duplicate; rationale: 003 establishes healthy configuration; 004 tests persistent count debounce and reboot, while 020 tests transient UUID identity changes and diagnostics.
- Tests: selected: tests/collectors/test_host.py::test_efa_inventory_ignores_non_efa_rdma_devices; tests/collectors/test_gpu.py::test_host_collector_reports_persistent_gpu_and_efa_card_loss; runner_coverage: tests/regional/test_collector_and_multicluster_fixtures.py::test_collector_setting_names_the_missing_key; observation: Production inventory semantics are tested; no direct runner test rejects matching counts when a fresh mismatch finding remains.; execution: Not run.
- Findings At Review: Missing control-plane false-alarm/freshness check.; Require successful inventory parsing and unique UUIDs instead of treating malformed output as merely a count.
- Reviewer Handoff: Review complete; controlled HostProbe state and finite settings/final-blast safety integrated. Persistent mismatch false-alarm/freshness assertion remains COLLECT-PARENT-005; matching inventory counts alone is not claimed as full acceptance.

### GF-REGIONAL-COLLECT-004

inventory mismatch 去抖阈值的真机验证

Current catalog: destructive-provider-reboot; manual; NOT_RUN.

- Purpose: Manual, destructive-provider-reboot: expected GPU count baseline+1, first-sample suppression and second-sample finding in one process, exactly one successful reboot, and byte-exact config restoration.
- Runner: scripts/e2e/regional/run_collector_destructive.py; scripts/e2e/regional/probes/collector_node_probe.py
- Entry Points: run_collect004; wait_mismatch_finding; debounce_errors; wait_planned_workflow; override_expected_gpu_count; restore_collector_env
- Assertions: Checks finding consecutive count and latency, restores env once reboot is planned, requires changed boot, one SUCCEEDED workflow, matching env digest, and one executor-role reboot event.; Latency starts at runner wall time; no first non-finding batch or common collector InvocationID is required.
- Safety And Cleanup: Configuration surrogate can cause a real provider reboot; risk must remain unchanged.; Env replacement precedes timer arming, and override call is outside runner try/finally.; Temporary env replacement does not preserve restrictive permission bits on a file that can contain cluster credentials.; OnBootSec=1 can already be elapsed when enabling the timer on a running node.; Finally restores env only, not failed workflow isolation, pending commands, or post-reboot services.
- Necessity: The surrogate is explicitly high risk. Current safety and debounce evidence are insufficient for unattended command promotion.
- Overlap: case_ids: GF-REGIONAL-COLLECT-003; GF-REGIONAL-COLLECT-015; GF-REGIONAL-DESTR-002; classification: shared_reboot_effect_distinct_detection; rationale: 004 uniquely tests persistent count debounce; reboot-effect helpers overlap other cases, but the detection premise does not.
- Tests: selected: tests/collectors/test_gpu.py::test_host_collector_reports_persistent_gpu_and_efa_card_loss; runner_coverage: tests/regional/test_collector_probe_env_restore.py; tests/regional/test_collector_runner_contracts.py::test_debounce_errors_demand_the_second_sample_at_about_two_intervals; tests/regional/test_collector_runner_contracts.py::test_collect004_tracks_reboot_without_modifying_the_production_collector; tests/regional/test_collector_runner_contracts.py::test_collect004_rejects_config_drift_and_unsuccessful_product_recovery; observation: Timer tests inspect generated unit text and stub systemctl. They cannot detect elapsed OnBootSec, pre-arm failure, file-mode loss, or a missing first sample.; execution: Not run.
- Findings At Review: Pre-action NodeRecovery/replacement/auto-resume/capability/target binding is absent in this handler.; Need same-process first-sample evidence and collector-timestamp-only debounce measurement.; Reused run ID can overwrite the original backup/override record.; Do not weaken risk or timing assertions to hide failures.
- Reviewer Handoff: Fixed partial env override rollback, restrictive mode preservation, immutable original backup, arm-before-mutation, same-boot deferred OnBootSec recovery and retained retryable watchdog. Admission checks precede override. First-sample/same-process debounce evidence and provider prerequisite coverage remain explicit parent/DESTR followups.

### GF-REGIONAL-COLLECT-005

fabric-manager collector 的采集层与游标持久化

Current catalog: non-destructive; manual; NOT_RUN.

- Purpose: Manual, catalog non-destructive: missing/corrupt FM state must baseline EOF; append one synthetic unknown Non-fatal SXID99999, prove cursor persistence/no replay, and no mutation workflow.
- Runner: scripts/e2e/regional/run_collector_acceptance.py; scripts/e2e/regional/collector_acceptance_fixture.py; scripts/e2e/regional/probes/collector_node_probe.py
- Entry Points: run_collect005; wait_fm_cursor; cursor_errors; append_sxid; fabric_manager_cursor; STORE_PROBE
- Assertions: Requires one evidence row before and across three post-restart samples, cursor offset=EOF, matching inode, forward progress, and active services.; Does not exercise missing/corrupt-state initialization or compare cursor device identity.; No-workflow check uses marker-only STORE_PROBE, which cannot join FM records to their workflow without an injection-time fallback.
- Safety And Cleanup: Append-only synthetic FM record, not a real switch fault; log lines are preserved.; Collector restart failure has no explicit start/recovery cleanup.; Hardcodes FM log/state paths instead of first proving the deployed configured source.
- Necessity: Useful harmless cursor testing, but PASS overstates EOF initialization and the current no-workflow assertion.
- Overlap: case_ids: GF-REGIONAL-COLLECT-011; classification: partial_overlap_keep_safe_cursor_case; rationale: 011 promises cursor checks under fatal-scope injection, which is not a replacement for safe unknown-code and missing-state tests.
- Tests: selected: tests/collectors/test_logs.py::test_fabric_manager_file_collector_persists_offsets; tests/collectors/test_logs.py::test_fabric_manager_file_cursor_rolls_back_on_delivery_failure; runner_coverage: tests/regional/test_collector_runner_contracts.py::test_collect005_polls_the_cursor_and_samples_the_replay_count; tests/regional/test_collector_runner_contracts.py::test_cursor_claims_compare_the_persisted_cursor_to_the_log; tests/regional/test_collector_probe_env_restore.py::test_fabric_manager_cursor_reads_the_persisted_offsets; observation: Cursor mocks exist, but no behavioral fake-Store test proves an unexpected FM workflow is found. Source-string probe assertions are inadequate for this join.; execution: Not run.
- Findings At Review: Preserve earlier FM/HMA retirement changes; never restore historical replay behavior.; Implement an isolated missing/corrupt-state exercise or ask parent to narrow the prose.; Bind event/record/workflow and cursor device/inode/offset to the appended record.
- Reviewer Handoff: Fixed exact FM event/workflow association and seed registration before append. Bad initial collection refuses collector restart; elapsed action windows refuse it too. Missing/corrupt cursor-state scope is proposed as a separate unit-only case instead of editing live cursor state.

### GF-REGIONAL-COLLECT-006

NVLink5 解码表 95 条规则表驱动全覆盖

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: Component pytest, non-destructive: all 95 NVLink5 rules across both driver branches/status values, independent expected actions, missing-evidence gates, secondary actions, and reachability.
- Runner: tests/policy/test_policy.py; tests/policy/_policy_cases_2.py; tests/policy/_support.py
- Entry Points: test_every_nvlink5_decode_rule_resolves_as_the_catalog_says; _nvlink5_decode_cases; _nvlink5_expected_action
- Assertions: Selected matrix uses an independent mask/status/action oracle, asserts official_action and matched subcode, and rejects BLOCKED_MISSING_EVIDENCE.; Exercises driverBoundary-5 and driverBoundary and expands slash-separated status values.; The separate 95-rule/reachability and missing-evidence tests are not selected by the catalog nodeid.
- Safety And Cleanup: Pure synthetic model events; no live collection, systemd, GPU, or cluster calls.; This pytest entry produces no formal case JSON predecessor evidence.
- Necessity: Good semantic matrix design, but its selected nodeid is not the full advertised decoder/gate suite.
- Overlap: case_ids: GF-REGIONAL-COLLECT-007; classification: complementary_policy_matrices; rationale: 006 supplies detailed decode evidence; 007 covers per-rule product applicability and generic outcomes, often with decode evidence absent.
- Tests: selected: tests/policy/test_policy.py::test_every_nvlink5_decode_rule_resolves_as_the_catalog_says; additional: tests/policy/test_policy.py::test_nvlink5_decode_table_has_no_unreachable_rule; tests/policy/test_policy.py::test_nvlink5_decodes_driver_specific_bits_but_does_not_guess; observation: Main-pattern witnesses set wildcard bits to zero. Explicit secondary-action positive/negative and invalid-mask/status witnesses need auditing; one main-pattern witness is not every branch.; execution: Not run; no coverage percentage claimed.
- Findings At Review: Parent should select all promised gate/invariant tests or narrow the spec.; Independently exercise missing driver, intr_info, error_status and action2/conflict outcomes.; Never label component replay as deployed hardware-fault evidence.
- Reviewer Handoff: Review complete, no owned runner mutation. Preserve the independent 95-rule decoder oracle; parent must include the separately promised reachability/missing-evidence tests in catalog selection (COLLECT-PARENT-006).
- Current Disposition: Catalog also selects independent reachability and missing-evidence decoder checks.

### GF-REGIONAL-COLLECT-007

XID 目录 172 条规则乘 4 个产品族判定层全覆盖

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: Component pytest, non-destructive: 172 catalog rules by four product columns, plus all advertised version/workflow/containment/unowned-action branches.
- Runner: tests/policy/test_policy.py; tests/policy/_policy_cases_1.py; tests/policy/_policy_cases_2.py; tests/policy/_support.py
- Entry Points: test_every_catalog_rule_resolves_as_the_catalog_says; _catalog_matrix_cases; _expected_xid_outcome
- Assertions: Selected 688-row matrix uses high driver/CUDA versions and independently computes official_action/disposition/action.; Inapplicable rows require action=None and safety quarantine.; Version boundaries, dynamic XID154, companions, UVM, containment, unknown XID, and unowned-action set are separate unselected tests.
- Safety And Cleanup: Only local policy engines and synthetic model events.; Cannot supply predecessor JSON for COLLECT-009.
- Necessity: The product matrix is valuable, but the one catalog nodeid does not prove the full 14-branch list or advertised mutation-sensitive checks.
- Overlap: case_ids: GF-REGIONAL-COLLECT-006; GF-REGIONAL-COLLECT-008; GF-REGIONAL-COLLECT-012; GF-REGIONAL-COLLECT-013; classification: semantic_baseline_not_live_duplicate; rationale: Exhaustive semantics avoid per-code hardware resets; live representatives add ingestion/identity/action effects rather than more catalog rows.
- Tests: selected: tests/policy/test_policy.py::test_every_catalog_rule_resolves_as_the_catalog_says; additional: tests/policy/test_policy.py::test_version_gate_branches_are_exact; tests/policy/test_policy.py::test_xid_154_covers_every_official_recovery_action; tests/policy/test_policy.py::test_xid_45_and_48_workflow_branches_are_exact; tests/policy/test_policy.py::test_containment_exceptions_are_pinned; tests/policy/test_policy.py::test_only_the_known_unowned_workflows_reach_no_executor; observation: The independent oracle explicitly assumes full versions and no extra evidence. Those assumptions explain why separate branch tests must be included.; execution: Not run; no mutation experiment performed.
- Findings At Review: Parent should select a complete bounded suite for promised behavior.; Do not claim version/product/containment mutations were killed without executing those tests.; Parent prose contains stale source-line references.
- Reviewer Handoff: Review complete, no owned runner mutation. Preserve the independent 688-row product oracle; parent must include version/XID154/companion/UVM/containment/unowned-owner tests or narrow its advertised contract (COLLECT-PARENT-006).
- Current Disposition: Catalog also selects version gates, XID154, companion, UVM, containment and unowned-workflow branches.

### GF-REGIONAL-COLLECT-008

WORKFLOW_XID_48 真 kmsg 双注入驱动 DRAIN_AND_RESET

Current catalog: destructive; manual; NOT_RUN.

- Purpose: Manual, destructive: user-space XID63 then XID48 writes to real kmsg must correlate on one GPU within 30 seconds, select DRAIN_AND_RESET, execute one physical reset, and recover.
- Runner: scripts/e2e/regional/run_collector_destructive.py; scripts/e2e/regional/run_destr001_gpu_reset.py; scripts/e2e/regional/probes/destructive_node_probe.py
- Entry Points: run_collect008; companion_xid_errors; wait_xid_workflow; workflow_errors; host_errors; write_generic_xid
- Assertions: Sleeps six seconds between writes; requires XID48/DRAIN_AND_RESET and shared exact reset-step/completion checks.; Separately reads the XID63 marker and requires XID63/kmsg://, MONITOR_ONLY, and no workflow.; Physical proof is one new RESET_GPU ledger row plus aggregate GPU-count dip/recovery, not target UUID evidence.; Does not bind companion event IDs, boot, numeric sequences, UUID, product/driver/CUDA, or measured event-time separation.
- Safety And Cleanup: User-space kmsg replay is not genuine hardware fault evidence.; Sampler stop is in finally; node quiesce and validated scheduling restore are absent from failure cleanup.; Host baseline/after snapshots are not persisted; shared stop deletes the trace.; Final blast-read exception can retain PASS.
- Necessity: The companion-policy delta is meaningful, but requires correlation evidence beyond the shared reset shape.
- Overlap: case_ids: GF-REGIONAL-COLLECT-013; GF-REGIONAL-COLLECT-016; GF-REGIONAL-DESTR-001; classification: substantive_physical_overlap_distinct_policy; rationale: All reuse single-GPU reset proof; 008 uniquely needs the 63-to-48 correlation/pre_actions. Identical step counts do not justify distinct policy PASS.
- Tests: selected: tests/orchestration/test_xid.py::test_xid_48_solo_and_companion_compile_the_same_containment; runner_coverage: tests/regional/test_collector_runner_contracts.py::test_collect008_reads_the_solo_xid63_by_marker; tests/regional/test_collector_runner_contracts.py::test_run_single_reset_stops_the_sampler_on_every_path; observation: Mocks cover marker routing and sampler stop, not wrong-boot/GPU companions, late writes, action binding, physical evidence retention, or failed-reset recovery.; execution: Not run.
- Findings At Review: Assert correlated_event_id, XID63 IGNORE/NO_ACTION, and XID48 exact disposition/action/pre_actions.; Persist physical evidence before teardown.; Coordinate shared reset proof with separate DESTR owner.
- Reviewer Handoff: Fixed before-ACK seed ownership, per-write admission, retained host/audit snapshots, signed physical reset identity/UUID/fence/result proof and terminal-only quiesce-before-validation cleanup. Full companion temporal/provenance coverage and target-UUID sampler output remain explicit followups, not hardware claims.

### GF-REGIONAL-COLLECT-009

CHECK_MECHANICALS等待与人工确认闭环真机跑通

Current catalog: live-non-destructive; manual; NOT_RUN.

- Purpose: Manual, live-non-destructive: XID54 creates FREEZE_EVIDENCE then CHECK_MECHANICALS; wrong fence stays WAITING, correct bound acknowledgement succeeds, node stays untouched.
- Runner: scripts/e2e/regional/run_collector_acceptance.py; scripts/e2e/regional/collector_acceptance_fixture.py; scripts/e2e/regional/probes/collector_node_probe.py
- Entry Points: configure; run_collect009; waiting_workflow; mechanical_acknowledgement_details; observe_marker; CaseCleanup.remove_annotation
- Assertions: Follows remote result_details pointers; writes wrong:0, observes 15 seconds, then writes returned expected value.; Requires some SUCCEEDED workflow and checks Ready/schedulability/gpu-fault taints, boot/services, and provisional provider-event absence.; Does not check exact decision/steps, actual incident:fencing equality, ledger/timers, or GPU UUID identity.
- Safety And Cleanup: Annotation is registered before writing, but removal ignores command failure and lacks readback.; WAITING timeout before state registration leaves no cleanup incident.; wrong:0 changes both incident and fence, so does not isolate wrong-fence rejection for the correct incident.
- Necessity: Distinct operator workflow. Local predecessor 007 is wrong: formal order requires 005, while 007 emits no predecessor case JSON; the shared wrapper does not normalize the local mapping.
- Overlap: case_ids: GF-REGIONAL-COLLECT-010; classification: distinct_operator_and_firmware_gates; rationale: 009 waits for human acknowledgement without isolation; 010 blocks absent firmware authorization and can quarantine.
- Tests: selected: tests/execution/test_misc.py::test_mechanical_inspection_waits_for_incident_fenced_annotation; runner_coverage: tests/regional/test_collect009_acknowledgement_details.py; tests/regional/test_collector_runner_contracts.py::test_collect009_observes_the_refusal_and_reads_the_node_state; tests/regional/test_collector_runner_contracts.py::test_collect009_fails_a_node_that_was_tainted_or_mutated; observation: No behavioral negatives for correct-incident/wrong-fence, failed deletion, missing/FAILED instead of WAITING, exact step shape, or formal predecessor resolution.; execution: Not run.
- Findings At Review: Use parent's formal_predecessor contract from owned configure.; Require every wrong-ack sample to remain the same fenced WAITING step.; Use deployed dispatcher timing and bind zero-action evidence to the window.
- Reviewer Handoff: Fixed formal predecessor selection, before-write seed ownership, failed-wait incident rediscovery, annotation-removal error retention and deadline admission before both acknowledgement writes. Correct-incident/wrong-fence discrimination and complete untouched-node evidence remain documented coverage gaps.

### GF-REGIONAL-COLLECT-010

UPDATE_SWFW 未配置固件目标版本时必须 fail-closed

Current catalog: live-isolation; manual; NOT_RUN.

- Purpose: Manual, live-isolation: with firmware target/allow/commands absent, user-space XID78 must block for missing owner/target, never quiesce or flash, then restore through validation.
- Runner: scripts/e2e/regional/run_collector_acceptance.py; scripts/e2e/regional/collector_acceptance_fixture.py; scripts/e2e/regional/probes/collector_node_probe.py
- Entry Points: run_collect010; inject_blocked_xid; CaseCleanup.restore
- Assertions: Requires first workflow BLOCKED and no UPDATE_SOFTWARE_FIRMWARE official step.; Checks unchanged boot and originally active services, then validated restore.; Omits exact blocking reasons, decision, dispatcher observation window, ledger/timers, and UUID checks.
- Safety And Cleanup: No pre-injection safe projection proves CPU target version absent and Node Agent firmware allow/commands disabled.; The negative test can therefore enter an enabled firmware path before post-hoc assertions fail.; Injection/wait errors occur before incident cleanup registration.
- Necessity: Necessary fail-closed coverage, but the missing negative-premise preflight is a safety blocker. Never enable firmware capability or change risk to pass it.
- Overlap: case_ids: GF-REGIONAL-COLLECT-009; GF-REGIONAL-COLLECT-011; classification: complementary_negative_premises; rationale: Different authorization/evidence gates must each be asserted; a generic BLOCKED status is not equivalent coverage.
- Tests: selected: tests/orchestration/test_misc.py::test_failed_firmware_remediation_escalates_directly_to_support; runner_coverage: tests/regional/test_collector_runner_contracts.py::test_collect010_registers_the_quarantine_before_it_judges_it; observation: Selected production test concerns failed firmware escalation, not absence of pre-action authorization. Runner test begins after a returned state and misses timeout-before-registration.; execution: Not run.
- Findings At Review: Read safe booleans/presence projections from relevant replicas before injection.; Require the intended blocked reasons and continuous no-physical-action evidence.; Register injection identity before waiting, then rediscover only this case's incident for cleanup.
- Reviewer Handoff: Fixed: pre-injection all-replica/running-Agent/OBSERVE premise, exact missing-target+missing-owner verdict, no non-containment execution, seed registration before write/wait and validated cleanup. Only unit projections/protocols executed; no firmware or live actions.

### GF-REGIONAL-COLLECT-011

SXID SCOPE_DEPENDENT_RESET 两个方向都 fail-closed

Current catalog: live-isolation; manual; NOT_RUN.

- Purpose: Manual, live-isolation: synthetic FM SXID11001 on two idle nodes distinguishes ACCESS without participants from UNKNOWN scope, blocks with exact fields, proves cursor behavior, and restores both nodes.
- Runner: scripts/e2e/regional/run_collector_acceptance.py; scripts/e2e/regional/collector_acceptance_fixture.py; scripts/e2e/regional/probes/collector_node_probe.py
- Entry Points: run_collect011; nvswitch_pci_bdf; select_workflow; STORE_PROBE; restore_incidents
- Assertions: Uses two nodes and a synthetic NVSwitch PCI slot distinct from GPU slots.; Injects two SXID11001 directions and checks some scope-dependent workflow is BLOCKED plus unchanged boot/services.; Omits ACCESS/UNKNOWN distinction, scope source, switch/port, participants, reasons, exact steps, A3/A4 codes, and cursor/state-loss checks.
- Safety And Cleanup: No idle-workload or deployed trusted-topology gate before injection.; Ingest resolves trusted topology first and can derive ACCESS participants from active workload; a non-GPU PCI address alone is insufficient.; Both injections precede either verdict, so a bad first direction does not stop the second.; Cleanup state is registered only after successful waits; workflow association is broad node/time matching.
- Necessity: Two distinct negative premises are justified, but neither is adequately proved or asserted by the current runner.
- Overlap: case_ids: GF-REGIONAL-COLLECT-005; GF-REGIONAL-COLLECT-014; classification: partial_fm_path_overlap_distinct_scope_gate; rationale: 005 safely tests cursor semantics; 014 tests inventory sufficiency. 011 must assert exact scope/participant results to justify its extra isolation exposure.
- Tests: selected: tests/orchestration/test_sxid_topology.py::test_conflicting_or_untrusted_topology_fails_closed; runner_coverage: tests/regional/test_collector_runner_contracts.py::test_collect011_judges_the_scope_dependent_workflow_not_the_latest; tests/regional/test_collector_and_multicluster_fixtures.py::test_collect011_injects_an_nvswitch_address_that_is_not_a_gpu_slot; observation: Selection/free-slot tests miss executable reset caused by workload/topology, wrong direction, subsequent mutation after failure, and wait-failure cleanup.; execution: Not run.
- Findings At Review: Prove negative premises before each injection.; Bind event/decision/incident/actions and assert each direction independently.; Parent must reconcile four-record/cursor prose with the two-record implementation.
- Reviewer Handoff: Fixed: fresh IDLE/topology premise using deployed enrichment, ACCESS without participants versus UNKNOWN exact policy reasons, normalized input to persisted decision/incident/workflow joins, before-write seeds and stop after failed direction. Parent must reconcile two executed directions with four-record/cursor prose.

### GF-REGIONAL-COLLECT-012

XID RESTART_APP 从 API 重放换成真 kmsg 注入

Current catalog: live-kernel-log-injection; manual; NOT_RUN.

- Purpose: Manual, live-kernel-log-injection: write identical XID13 twice and XID31 once on an idle node; distinct kernel sequences/current identity, MONITOR_ONLY/NO_ACTION, recovered incidents, no workflow.
- Runner: scripts/e2e/regional/run_collector_acceptance.py; scripts/e2e/regional/collector_acceptance_fixture.py; scripts/e2e/regional/probes/collector_node_probe.py
- Entry Points: run_collect012; wait_monitor_only; kmsg_record_errors; restart_app_monitor_only_errors; write_xid
- Assertions: Requires >=3 record IDs and current-boot prefix for the first two writes, distinct nonempty refs, idle RESTART_APP decision, no workflows, and some RECOVERED incident.; Final node checks cover boot/services/schedulability/gpu-fault taints.; Does not parse/ref-bind numeric sequence, node, boot, product/driver/CUDA, or resolved GPU UUID for every event.
- Safety And Cleanup: User-space writes to real kmsg, not real hardware faults.; Generic preflight does not prove idle before injecting an application-restart class XID.; Registers no cleanup state for an unexpected quarantine/workflow.; All three injections precede verdict evaluation.
- Necessity: Idle policy and sequence behavior are distinct from actual restart; absence of a managed workload must be a precondition, not an expected-result assumption.
- Overlap: case_ids: GF-REGIONAL-COLLECT-016; GF-REGIONAL-COLLECT-021; GF-REGIONAL-COLLECT-007; classification: distinct_workload_states; rationale: 012 starts idle; 016 starts active and tests proactive restart; 021 terminates an attempt before late XID and tests passive recovery.
- Tests: selected: tests/orchestration/test_misc.py::test_restart_app_on_an_idle_node_does_not_open_a_workflow; runner_coverage: tests/regional/test_collector_runner_contracts.py (historical pre-fix symbol: test_kmsg_records_for_identical_lines_share_the_boot_prefix); tests/regional/test_collector_runner_contracts.py::test_collect012_judges_the_idle_node_monitor_only_without_a_workflow; tests/regional/test_collector_runner_contracts.py::test_restart_app_monitor_only_errors_names_every_regression; observation: Covers old BLOCKED behavior/duplicate IDs, not forged refs, identity mismatch, active preflight, or unexpected-workflow recovery.; execution: Not run.
- Findings At Review: Same-original-sequence isolated replay is not part of this runner/focused selection.; Wait should require the matching recovered incident.; Do not reintroduce retired HMA paths or old idle quarantine behavior.
- Reviewer Handoff: Fixed mutating idle-node preflight, real nested RawEvidenceRecord shape, cluster/node/boot/numeric sequence/current GPU/software/event/decision/incident binding, pre-ACK seeds and stop before subsequent injections after a bad sample. Existing HMA retirement preserved.

### GF-REGIONAL-COLLECT-013

XID RESET_GPU 真 kmsg 注入并真的执行单卡 reset

Current catalog: destructive; manual; NOT_RUN.

- Purpose: Manual, destructive: one XID109 or XID62 user-space kmsg injection, one physical single-GPU reset, detached evidence, automatic recovery without reboot.
- Runner: scripts/e2e/regional/run_collector_destructive.py; scripts/e2e/regional/run_destr001_gpu_reset.py; scripts/e2e/regional/probes/destructive_node_probe.py
- Entry Points: run_collect013; run_single_reset; workflow_errors; host_errors; stop_sampler
- Assertions: Allowlisted XID and exactly one reset cycle.; Shared verdict checks exact reset/completion sequence, remote success, one ledger row, service/timer recovery, and aggregate count dip.; Lacks current boot/sequence/product/UUID and ledger-to-workflow/target binding.
- Safety And Cleanup: Sampler finally exists, but no restore-quiesce/validated scheduling recovery on failure.; run_single_reset discards host baseline/after snapshots; shared stop deletes sampler trace.; Shared DESTR helper changes require coordination, not unilateral edits.
- Necessity: One representative direct-reset class is justified; per-code resets are not. Physical-action evidence must be independent of synthetic injection text.
- Overlap: case_ids: GF-REGIONAL-COLLECT-008; GF-REGIONAL-COLLECT-016; GF-REGIONAL-DESTR-001; classification: substantive_physical_overlap; rationale: Physical proof overlaps 008/DESTR-001; direct-class ingestion remains distinct from 008 correlation. Reuse proof only with parent-approved identity binding.
- Tests: selected: tests/node_agent/test_remediation.py::test_gpu_reset_is_idempotent_and_rechecks_clients; runner_coverage: tests/regional/test_collector_runner_contracts.py::test_collect013_runs_one_reset_cycle_for_the_selected_xid; tests/regional/test_collector_runner_contracts.py::test_run_single_reset_stops_the_sampler_on_every_path; observation: Covers single-cycle dispatch/sampler stop, not partial reset recovery, evidence persistence, or stale/wrong-target action records.; execution: Not run.
- Findings At Review: Persist physical evidence before teardown.; Recheck idle/capability/deadline before injection.; Parent prose's manual uncordon fallback conflicts with validation-first recovery.
- Reviewer Handoff: Fixed one-reset caller lifecycle: owned seed before write, fresh action admission, retained host and reset-audit snapshots, signed ledger-to-workflow/fence/UUID/result proof, and terminal-command quiesce-before-validation cleanup. Shared aggregate sampler limitation remains explicitly separate.

### GF-REGIONAL-COLLECT-014

SXID整机GPU与NVSwitch reset的双证据门与正向

Current catalog: destructive; manual; NOT_RUN.

- Purpose: Manual, destructive: synthetic API SXID10003 must block without complete inventory; restore isolation, then synthetic FM injection causes one full GPU/NVSwitch reset with unchanged boot/full UUID set.
- Runner: scripts/e2e/regional/run_collector_destructive.py; scripts/e2e/regional/collector_acceptance_fixture.py; scripts/e2e/regional/probes/collector_node_probe.py; scripts/e2e/regional/probes/destructive_node_probe.py
- Entry Points: run_collect014; fail_closed_timing_errors; GPU_INVENTORY_SNAPSHOT; post_fabric_event; STORE_PROBE
- Assertions: Eight-minute backdate and 90-second future margin against only dedicated inventory; blocked reason and no ledger/boot change, restore before positive.; Positive checks SUCCEEDED full-reset step, one new ledger row, equal GPU counts, unchanged boot.; No full UUID equality, exact owner/parameters/sequence, ledger success-state binding, or physical sampler proof.
- Safety And Cleanup: Ingest can fall back to legacy GPU metrics when dedicated snapshot is missing/ineligible; current guard ignores that source and treats missing snapshot as safe.; Therefore the intended negative can execute a full reset. API source currently claims an FM path without an explicit synthetic origin annotation.; Positive restore is after sampler finally, so a raised wait skips node restore; negative wait failures also bypass it.; Successful failed-premise verdict stops the positive path, but only after the first mutation could already happen.
- Necessity: Full-fabric coverage is distinct, but a temporal negative premise must include every runtime evidence source or use an isolated non-executable replay. No timestamp guess authorizes a supposedly safe mutation.
- Overlap: case_ids: GF-REGIONAL-COLLECT-011; GF-REGIONAL-COLLECT-013; classification: distinct_scope_with_shared_reset_lifecycle; rationale: Full-fabric inventory equality differs from untrusted port scope and single-GPU reset; common lifecycle proof is reusable.
- Tests: selected: tests/node_agent/test_remediation.py::test_full_fabric_reset_rejects_inventory_mismatch; tests/node_agent/test_remediation.py::test_full_fabric_reset_verifies_inventory_and_is_idempotent; runner_coverage: tests/regional/test_collector_and_multicluster_fixtures.py::test_collect014_never_injects_the_positive_sxid_after_a_bad_fail_closed; tests/regional/test_collector_and_multicluster_fixtures.py::test_collect014_stops_the_sampler_when_the_positive_path_raises; tests/regional/test_collector_runner_contracts.py::test_collect014_refuses_to_post_when_the_stored_inventory_is_too_fresh; observation: Mocks prove snapshot timing and stop-after-detected-failure, not legacy fallback, evidence source drift, wrong UUID sets, or validated restore after a thrown wait.; execution: Not run.
- Findings At Review: Catalog says two-minute backdate and 10003+19084 positives; implementation uses eight minutes and one 10003 positive.; Parent should keep 19084 nonfatal equivalence in safe policy replay, not silently authorize another physical reset.; Persist full host proof before deleting sampler.
- Reviewer Handoff: Fixed dedicated+legacy inventory negative premise, explicit API-replay source, caller-known seed before API ACK and FM write, exact SXID joins, failure-path registered cleanup, stop before positive on negative failure, retained audit and exact full-inventory signed reset proof. Eight-minute temporal contract and one positive reset need parent catalog/spec reconciliation.

### GF-REGIONAL-COLLECT-015

SXID ALWAYS_FATAL 经 provider 真机整机重启

Current catalog: destructive-provider-reboot; manual; NOT_RUN.

- Purpose: Manual, destructive-provider-reboot, last in COLLECT phase: synthetic ALWAYS_FATAL FM SXID23001 selects RESTART_BM/REBOOT_NODE, one provider reboot, no replacement, unchanged provider/node identity, recovered services/scheduling.
- Runner: scripts/e2e/regional/run_collector_destructive.py; scripts/e2e/regional/probes/collector_node_probe.py; scripts/e2e/regional/warm_spare_fixture.py
- Entry Points: run_collect015; collect015_workflow_errors; HYPERPOD_SUBMISSION; EXECUTOR_REPLACE_FLAG
- Assertions: Requires changed boot, SUCCEEDED restart workflow/step, SUBMITTED provider record, equal provider inventory, one executor-role reboot, provisional zero replacement events.; NodeRecovery and replacement flag are checked only after injecting and waiting for reboot.; No exact seven-step/decision proof, Kubernetes Node UID comparison, uptime reset, postboot GPU UUID/service/timer verification, or sampler.
- Safety And Cleanup: Synthetic FM log replay causes an actual provider reboot; never a physical switch fault.; No pre-action auto-resume/NodeRecovery/replacement/capability gate in this handler.; Outer finally removes probes but cannot recover failed reboot/isolation workflows.; No separate pre-action provider target/action evidence binding beyond shared helper behavior.
- Necessity: ALWAYS_FATAL ingestion is distinct from XID reboot, but the safety premise must precede mutation and final physical/node recovery must be proved.
- Overlap: case_ids: GF-REGIONAL-COLLECT-004; GF-REGIONAL-DESTR-002; classification: substantive_provider_effect_overlap_distinct_signal; rationale: Same provider reboot mechanism, different detecting signals. Additional codes should stay policy-only rather than repeat reboot.
- Tests: selected: tests/policy/test_policy.py::test_always_fatal_sxid_uses_host_restart_branch; runner_coverage: tests/regional/test_collector_and_multicluster_fixtures.py::test_collect015_matches_the_reboot_workflow_by_injection_time; tests/regional/test_collector_runner_contracts.py (historical pre-fix symbol: test_collect015_control_plane_readings); observation: Tests check returned control-plane fields, not refusal before injection, missing/unknown safety values, wrong provider target, or post-reboot cleanup failure.; execution: Not run.
- Findings At Review: Missing NodeRecovery currently passes as None, contrary to fail-closed unknown-state semantics.; CloudTrail negative is correctly marked provisional; formal final audit must occur after this phase, not rely on earlier DESTR-013.; Retain no-provider-replace invariant and route shared binding changes to parent.
- Reviewer Handoff: Fixed exact SXID joins, pre-ACK seed ownership, admission and final-blast failure semantics. Reboot/provider capability prerequisites, complete postboot host proof and event actor/target binding remain DESTR/parent integration work; no live reboot acceptance claimed.

### GF-REGIONAL-COLLECT-016

真实分布式训练下的 RESTART_APP 与 RESET_GPU 执行段

Current catalog: destructive; manual; NOT_RUN.

- Purpose: Manual, destructive: active 24-GPU training, proactive RESTART_APP, budget exhaustion without mutation, then new-job STOP/quiesce/no-client/single-reset/restart.
- Runner: scripts/e2e/regional/run_collect016_training_recovery.py; scripts/e2e/regional/run_collector_destructive.py; scripts/e2e/regional/managed_workload_fixture.py; scripts/e2e/regional/run_destr009_workload_restart.py; scripts/e2e/regional/run_destr001_gpu_reset.py
- Entry Points: run_restart_budget_sections; restart_budget_section_errors; run_reset_section; execute_case; run_single_reset
- Assertions: Registers resources before creation and uses distinct A/B versus D job IDs.; A reuses restart verdict but only requires nonempty identity source; B checks FAILED/budget reason/no commands/count=1.; Omits exact sole-active identity, event boot/product/UUID/attempt, baseline budget=0, B unchanged Pods/suspend/completed operations, and sole restarted-observation gate.; D reuses idle host_errors that rejects all compute clients even after training restarts.
- Safety And Cleanup: D still runs when A/B returns errors.; No per-mutation deadline/selected-node gate after long waits.; Deletes workload/probes/prewarm but does not quiesce failed workflows or restore partial reset isolation.; Shared workload delete ignores command failures and lacks residual proof; parent/DESTR-owner issue.
- Necessity: Distinct active ownership/budget/ordering proof. Persistent-client negatives must remain mocked, never a live device holder.
- Overlap: case_ids: GF-REGIONAL-COLLECT-012; GF-REGIONAL-COLLECT-013; GF-REGIONAL-COLLECT-021; GF-REGIONAL-DESTR-009; classification: partial_effect_overlap_distinct_active_binding; rationale: Restart/reset effects overlap, but deployed kmsg enrichment and budget refusal add claims absent from idle/API cases.
- Tests: selected: tests/hyperpod/test_e2e_distributed_xid_reset.py::test_three_node_job_stops_before_two_fault_gpu_resets; tests/execution/test_restart_safety.py::test_restart_budget_blocks_second_restart_for_same_job; runner_coverage: tests/regional/test_collect016_training_recovery.py; observation: Resource registration and missing-field tests exist; no full execute stop-on-error, wrong nonempty identity, B stale observation, or active D host-verdict compatibility test.; execution: Not run.
- Findings At Review: Catalog includes DCGM CONFIG-failure/client-intersection claims not directly exercised here.; No final CPU-blast comparison.; Coordinate shared workload/DESTR helper changes.
- Reviewer Handoff: Fixed stop after A/B or A failure, pre-ACK seeds, deadline checks before later actions, controlled probe state, retained reset audit and terminal-command cleanup. Shared idle host_errors currently conflicts with restarted training clients; parent/DESTR must settle that contract before live use.

### GF-REGIONAL-COLLECT-017

EFA 分层恢复与 Kubernetes accelerator allocatable

Current catalog: live-node-mutation; manual; NOT_RUN.

- Purpose: Manual, live-node-mutation: one EFA driver unbind/agent rebind, NVIDIA plugin availability loss, and EFA plugin loss during managed training without workload restart.
- Runner: scripts/e2e/regional/run_collect017_efa_plugin.py; scripts/e2e/regional/run_collector_destructive.py; scripts/e2e/regional/probes/collector_node_probe.py; scripts/e2e/regional/managed_workload_fixture.py
- Entry Points: run_efa_unbind; efa_unbind_errors; run_gpu_plugin; run_training_plugin; DevicePluginFixture.exclude_node; DevicePluginFixture.restore; execute_case
- Assertions: A checks seven planned steps, driver-unbound reason, remote exact BDF rebind, inventory recovery, no fail-safe rescue, and recovered incident.; B/C check exact steps/statuses; C checks Pod UIDs but not fresh collective progression or bound job/attempt observation.; A chooses a remote result by operation/node, not the exact execution command ID.
- Safety And Cleanup: Unbind call is outside try despite armed host timer.; Plugin exclusion overwrites original required affinity terms; merge-patch restore can leave added keys behind.; Template patch can trigger cluster-wide plugin rollout, with no durable affinity watchdog.; B/C continue after earlier verdict errors and failed incident isolation is not restored.
- Necessity: Driver binding and GPU/EFA registration are distinct layers; retain no physical PCI removal/link damage/reboot fallback.
- Overlap: case_ids: GF-REGIONAL-DESTR-021; GF-REGIONAL-COLLECT-020; classification: substantive_A_segment_overlap_only; rationale: DESTR-021 reuses A plus stale/concurrent annotations, not B/C; 020 tests inventory identity, not plugin registration.
- Tests: selected: tests/collectors/test_host.py::test_efa_inventory_distinguishes_unbound_driver_from_missing_pci; tests/orchestration/test_health.py::test_idle_node_resource_group_upgrades_plugin_to_driver_remediation; runner_coverage: tests/regional/test_collect017_efa_verdicts.py; tests/regional/test_collector_and_multicluster_fixtures.py::test_device_plugin_discovery_ignores_daemonsets_that_schedule_nowhere; observation: Good A negatives/restore-on-wait-failure; missing structured affinity roundtrip, rollback conflict/watchdog, partial unbind, wrong command ID, and stop-after-phase-error tests.; execution: Not run.
- Findings At Review: C uses submit() while prose explicitly requires rendered managed kubectl apply; parent integration issue.; Preflight omits agent/profile efaDriverRemediation owner and C three-node readiness.; Common namespace/budget/teardown issues go to parent.
- Reviewer Handoff: Fixed partial EFA rollback, stop after failed phases, conjunctive affinity preservation, UID/RV-fenced patch and Pod deletion, durable GPU-local restoration watchdog with independent worker protocol test, retained proof on cleanup uncertainty and explicit OnDelete-only deployment policy. Watchdog uses the installed immutable Executor image and component interpreter; no custom DaemonSet strategy is modified.

### GF-REGIONAL-COLLECT-018

202 之后被拒的故障层事件被计数、告警并落到节点采集器状态；无编号的 Xid 行成为 operator-review finding 而非静默丢弃

Current catalog: live-kernel-log-injection; manual; NOT_RUN.

- Purpose: Manual, live-kernel-log-injection: synthetic API post rejected after 202, unparsed user-space Xid becomes freeze-only operator review, accepted batch clears erroring.
- Runner: scripts/e2e/regional/run_collect018_rejected_event.py; scripts/e2e/regional/collect018_verdicts.py; scripts/e2e/regional/collector_window_fixture.py; scripts/e2e/regional/probes/collector_window_probe.py
- Entry Points: execute; post_rejected_event; rejected_status_errors; rejection_log_errors; unparsed_finding_errors; recovery_errors; run_window_case
- Assertions: Correctly labels API/user-space origins; checks 422 row, counters, some request log, freeze inclusion, review notification, marker evidence, and success timestamp.; Freeze-only uses incomplete denylist: QUARANTINE misspelled QUARANTINE_NODE; several real mutations are allowed.; Post request/time not bound to status/log; stale or unrelated evidence can satisfy verdict.
- Safety And Cleanup: Deadline discarded; errors bypass recovery batch.; Unexpected mutations not restored.; Window predecessor and final envelope omit release/cluster binding.
- Necessity: Distinct rejection/unresolved-line coverage; require exact freeze-only behavior, not a partial destructive denylist.
- Overlap: case_ids: GF-REGIONAL-COLLECT-019; GF-REGIONAL-NET-008; classification: shared_plumbing_distinct_failure; rationale: Processor rejection differs from tool timeout and transport/outbox outage.
- Tests: selected: tests/regional/test_collect018_rejected_event.py::test_the_rejected_status_contract_passes_on_a_rejected_event_row; runner_coverage: tests/regional/test_collect018_rejected_event.py; tests/regional/test_collector_window_probe.py; observation: Pure tests reject MARK_UNSCHEDULABLE but miss QUARANTINE/full-reset/workload mutation and failed freeze. No full execute failure cleanup/request-binding test.; execution: Not run.
- Findings At Review: Untouched-node check omits Ready/boot/taints/GPU identity.; Heartbeat alignment timeout does not stop posting.; Activity notification join is node text, not cluster-bound incident identity.
- Reviewer Handoff: Fixed exact FREEZE_EVIDENCE-only success contract, schema-3 plan inputs and release/cluster envelope, caller-known record-ID intent before post, heartbeat alignment refusal, phase-stop and deadline admission. Broader notification/finding provenance limits remain stated; all injection evidence is labelled synthetic/user-space.

### GF-REGIONAL-COLLECT-019

nvidia-smi 挂起超过超时多轮，host collector 不重启、持续投递带断路器原因的错误批次，控制面报 erroring 而非 silent

Current catalog: live-non-destructive; manual; NOT_RUN.

- Purpose: Manual, live-non-destructive: unit-only nvidia-smi shadow hangs; timeout then breaker, same active process, erroring not silent, close and recover.
- Runner: scripts/e2e/regional/run_collect019_nvidia_smi_hang.py; scripts/e2e/regional/collect019_verdicts.py; scripts/e2e/regional/collector_window_fixture.py; scripts/e2e/regional/probes/collector_window_probe.py
- Entry Points: execute; open_window_or_rollback; open_window; close_window; erroring_status_errors; service_errors; gauge_errors
- Assertions: Checks timing/deadman, accumulated timeout/breaker, same PID/NRestarts, gauges, removed root/dropin, and success after error.; Missing PID/NRestarts compare equal; errors need not be fresh/ordered/current.; HOST_TELEMETRY recovery can come from another producer.
- Safety And Cleanup: Synthetic query hang, not hung hardware.; Partial-open rollback catches Exception but not abort BaseException.; Close stops timer before restart succeeds and can remove recovery state despite bad final unit.; Close is not idempotent after prior close/deadman; deadline discarded.
- Necessity: Useful narrow process-resilience test; durable retryable close and process/status identity are prerequisites for the watchdog claim.
- Overlap: case_ids: GF-REGIONAL-COLLECT-001; GF-REGIONAL-COLLECT-018; GF-REGIONAL-NET-008; classification: complementary_shared_window; rationale: Responsive erroring process differs from healthy cadence, payload rejection, and network loss.
- Tests: selected: tests/regional/test_collect019_nvidia_smi_hang.py::test_the_erroring_status_contract_passes_on_timeout_then_breaker; runner_coverage: tests/regional/test_collect019_nvidia_smi_hang.py; tests/regional/test_collector_window_probe.py; observation: Status/service and failed-open wrapper tests exist; no real-helper failure matrix, abort, retained watchdog, or missing process-ID negatives.; execution: Not run.
- Findings At Review: Shared release/cluster predecessor/result binding absent.; Retain recovery state until active readback and cleanup succeed.; Preserve NET-008 behavior while fixing the shared collector-specific window helper.
- Reviewer Handoff: Fixed partial-open abort rollback, pre-mutation watchdog arming, retained state until verified close, unit-bound idempotent close and automatic retry, missing process identity refusal, finite metrics/intervals and action deadline admission. No hung-driver/hardware proof claimed.

### GF-REGIONAL-COLLECT-020

未配置期望 GPU 数时按实例类型推导；一次 inventory 查询少一个 UUID 即 CRITICAL identity finding、诊断而非重启，restore 后 marker 带退役字段

Current catalog: live-isolation; manual; NOT_RUN.

- Purpose: Manual, live-isolation: hide one UUID in one inventory query, derive expected count, critical identity diagnostics without reboot/reset, validated restore, explicit marker retirement.
- Runner: scripts/e2e/regional/run_collect020_gpu_identity.py; scripts/e2e/regional/collect020_verdicts.py; scripts/e2e/regional/collector_window_fixture.py; scripts/e2e/regional/probes/collector_window_probe.py
- Entry Points: execute; _restore; inventory_evidence_errors; identity_finding_errors; dcgm_notification_errors; marker_retirement_errors; SHADOW_SCRIPT
- Assertions: Preconditions ensure known count and one drop; checks forbidden plans and final boot/node.; Inventory accepts empty/wrong sets omitting target UUID; diagnostics accept either required operation instead of both.; DCGM notice may belong to another incident, markers may be absent, critical severity not checked.
- Safety And Cleanup: Query shadow only, never physical device removal.; Window finally exists but exceptions skip later incident restore.; Close verdict requires only dropin removal; hard-stop is not applied continuously during later waits.; Malformed shadow counter content raises instead of safely passing through; deadline discarded.
- Necessity: Distinct identity/retirement regression; exact inventory and action/notification binding must replace existence-only evidence.
- Overlap: case_ids: GF-REGIONAL-COLLECT-003; GF-REGIONAL-COLLECT-004; GF-REGIONAL-COLLECT-017; classification: distinct_transient_identity; rationale: One-query UUID changes differ from config counts, persistent loss/reboot, and plugin allocatable changes.
- Tests: selected: tests/regional/test_collect020_gpu_identity.py::test_the_identity_finding_contract_passes_on_a_diagnostic_workflow; runner_coverage: tests/regional/test_collect020_gpu_identity.py; tests/regional/test_collector_window_probe.py::test_the_drop_uuid_shadow_drops_one_query_and_then_passes_through; tests/regional/test_collector_window_probe.py::test_the_drop_uuid_shadow_passes_through_when_its_counter_cannot_be_kept; observation: Missing empty/wrong inventory, missing one diagnostic step, unrelated DCGM, absent markers, malformed counter, and wait-failure restore negatives.; execution: Not run.
- Findings At Review: Shared identity envelope absent.; Require all case incidents restored and expected markers present.; Retain live-isolation risk and one-hidden-query limit.
- Reviewer Handoff: Fixed exact inventory UUID/count checks, both diagnostic operations, related critical notification and marker requirements, malformed shadow-counter fail-safe, durable window close and deadline admission. Unexpected-isolation recovery after arbitrary later read failure remains an explicit workflow-coverage followup; physical removal is never claimed.

### GF-REGIONAL-COLLECT-021

Pod 先死、RESTART_APP XID 后到——空闲判定 MONITOR_ONLY，被动路径仍重启

Current catalog: live-workload-restart; manual; NOT_RUN.

- Purpose: Manual, live-workload-restart: kill selected Pod processes, prove its nonzero terminal exit before late user-space XID13, idle/no workflow, passive budgeted new attempt.
- Runner: scripts/e2e/regional/run_collect021_late_xid_after_pod_death.py; scripts/e2e/regional/probes/collector_node_probe.py; scripts/e2e/regional/managed_workload_fixture.py; scripts/e2e/regional/run_destr009_workload_restart.py
- Entry Points: run_late_xid_section; wait_death_observation; node_reads_idle; proactive_errors; passive_errors; workload_processes; kill_workload
- Assertions: Good exact proactive decision/no-command/no-workflow and boot/taint/ownership checks.; Passive accepts FAILED or STOPPED, disjoint UIDs, changed attempt, count=1, any non-source pending/running observation.; Death wait ignores pod_uid; unknown phase reads idle and absent target containers can pass.; No exact kmsg provenance or event/decision/incident identity linkage.
- Safety And Cleanup: Resources registered before creation; shared workload deletion lacks residual proof.; Process selection uses arbitrary pod<UID> substrings and later numeric-PID kill without lifetime/cgroup revalidation.; No deadline recheck after long waits or validated unexpected-isolation recovery.
- Necessity: Distinct terminal-before-XID race, but death must be this selected failed container, not unknown state/user STOPPED. Never substitute Pod deletion for the injection.
- Overlap: case_ids: GF-REGIONAL-COLLECT-012; GF-REGIONAL-COLLECT-016; classification: distinct_ordering_and_passive_recovery; rationale: 012 has no failed job; 016 injects while active. Neither proves this ordering/passive recovery.
- Tests: selected: tests/completion/test_pod_dies_before_xid_ingest.py; runner_coverage: tests/regional/test_collect021_late_xid_after_pod_death.py; observation: Good proactive/process basics, but an existing test explicitly accepts STOPPED without the killed container; no death-wait identity/PID-reuse/provenance negatives.; execution: Not run.
- Findings At Review: Fail closed on wrong job/attempt/Pod/node, unknown phase, missing nonzero exit, and stale observations.; Bind one new RUNNING attempt across Pods and observation.; Common workload deletion/namespace/budget issues go to parent.
- Reviewer Handoff: Fixed exact failed job/attempt/Pod/node death proof, nonzero termination, UNKNOWN/STOPPED refusal, pre-ACK late-XID seed ownership and later action admission with controlled probe state. PID-lifetime revalidation and sole-active restarted-attempt/provenance completeness remain documented followups; no live process kill executed.

### GF-REGIONAL-COLLECT-022

部署后FM采集器私有游标丢失损坏及日志轮转恢复

Current catalog: live-non-destructive; manual; NOT_RUN.

- Purpose: Run the deployed FabricManagerLogCollector in separate guarded host processes against nonce-owned private log/checkpoint files and a non-forwarding sink. Check missing/corrupt cursor recovery, rename rotation and same-inode truncation without touching existing collector files or services.
- Runner: scripts/e2e/regional/run_collect022_fm_cursor_recovery.py; scripts/e2e/regional/probes/collect022_fm_cursor_probe.py
- Entry Points: read_only_preflight; execute_case; phase_errors; truncation_errors; run_action; truncate_private; cleanup
- Assertions: Current bound COLLECT-005 evidence and approved node/release/Agent generation must match preflight.; Every phase retains the initialized root, runtime module, service invocation and boot identity, and proves a distinct reader process.; Historical data is skipped when the checkpoint is missing or corrupt; fresh markers A/B/C/R/N/T are delivered exactly once.; Rename rotation preserves the old unread inode/tail and collects the new active file; the following restart does not replay either marker.; Truncation requires an intact EOF checkpoint for the existing private rotated.log and an appended record shorter than the saved offset.; Truncation records the same-inode zero-length intermediate snapshot, leaves the old cursor intact until collection, advances the cursor to the new EOF and does not replay T on restart.; Exact marker counts, private path/device/inode/offset joins, complete cursor file sets and globally non-repeated delivered record IDs are retained.; Any malformed proof or unresolved private-file/HostProbe cleanup residual fails the case.
- Safety And Cleanup: No configurable private file path; only nonce-owned allowlisted files under the fixed private root.; Private root/owner document/inode binding, nonblocking ownership lock and file type/link/permission/size checks are retained.; Persist local intent before initialization and phase intent/receipts before acknowledging completion; unresolved phases are not replayed.; Cleanup runs after lost ACK, exception, abort or rejected proof and never adopts an unproved directory or follows an outside symlink.; No forwarding sink, journal command, collector service mutation, real collector state deletion or GPU action is used by the private reader.
- Necessity: Private missing/corrupt checkpoints and rotation exercise deployed cursor behavior not covered by retained-state restart.
- Overlap: case_ids: GF-REGIONAL-COLLECT-005; classification: complementary; rationale: COLLECT-005 covers the normal deployed collection chain and intact cursor restart; COLLECT-022 separately exercises private cursor loss, corruption, rotation and truncation.
- Tests: selected: tests/regional/test_collect022_fm_cursor_recovery.py::test_deployed_reader_private_missing_corrupt_rotation_and_restart_protocol; truncation_positive: tests/regional/test_collect022_fm_cursor_recovery.py::test_private_truncation_delivers_once_and_advances_the_same_inode_cursor; module_passed: 59; truncation_subset_passed: 13; truncation_subset_included_in_module_count: True; failed: 0; skipped: 0
- Findings At Review: Only mocked/temp-file verification was executed; no deployed-host or hardware acceptance verdict is claimed.; CPU ingestion, notifications and control-plane deduplication are outside the non-forwarding reader case.; The truncation target is the existing private rotated.log; its byte-zero historical record was skipped at baseline, preserving the prior delivered-record identity checks.
- Reviewer Handoff: Implemented and focused-tested; not executed on a deployed host.

### GF-REGIONAL-DESTR-001

经 executor 与 Node Agent 的真实 GPU reset

Current catalog: destructive; manual; NOT_RUN.

- Purpose: Phase 13 after 010: one synthetic userspace kmsg XID46 drives the real eight-step idle-node reset. Require current Node Agent owner, one physical target-GPU reset, per-GPU post-readiness validation, observed isolation/restoration, no provider mutation and unchanged CPU state. This is software injection, not hardware damage.
- Runner: scripts/e2e/regional/run_destr001_gpu_reset.py; scripts/e2e/regional/probes/destructive_node_probe.py; scripts/e2e/regional/host_probe_fixture.py
- Entry Points: main -> run_standard_case(CASE); read_only_preflight -> preflight_errors; execute_case -> workflow_errors / host_errors
- Assertions: workflow_errors checks xid=46, evidence_ref=kmsg://, official_action=RESET_GPU, exact official/completed operation order and SUCCEEDED remote operations, including batched_steps via command_operations.; waiting_details_present accepts WAITING mutation_submitted_by_control_plane=false or a terminal remote/ adapter ID, without binding that ID to an exact command result.; host_errors requires one new RESET_GPU row, attempt=1, restored active services/timer inventory, no quiesce state, sampler sample_count>=2, min count below expected and final count equal expected.
- Safety And Cleanup: admission: Exact confirmation/deadline and same release/cluster predecessor; plan binds Node UID, Agent generation and profile version, not full runtime templates/artifact or Agent lease freshness.; cleanup: Sampler cleanup is owed before start, including timeout. finally restores quiesce only if incident_id was learned, deletes probes and reads final Ready.; all_exit_gap: Wait timeout before terminal incident assignment skips quiesce recovery. Final scheduling/taints/UID are not revalidated after cleanup. Abort BaseException skips the result write after finally.
- Necessity: Retain as basic reset proof. Total enumeration dip is corroboration, not target identity: an nvidia-smi error or sibling disappearance must not prove this GPU was reset.
- Overlap: case_ids: GF-REGIONAL-DESTR-023; GF-REGIONAL-COLLECT-008; GF-REGIONAL-COLLECT-013; GF-REGIONAL-COLLECT-016; classification: shared_action_contract_not_duplicate; rationale: Other cases add heartbeat-only idle, companion/correlation or managed-workload inputs. Reuse one assertion library without counting repeated reset checks as new target-identity proof.
- Tests: existing: tests/regional/test_destructive_acceptance_fixtures.py::test_destr001_requires_the_exact_reset_contract; tests/regional/test_destructive_acceptance_fixtures.py::test_physical_reset_is_proven_by_the_sampler_dip_not_by_journal_text; tests/regional/test_destructive_review_fixes.py::test_destr001_stops_a_sampler_whose_start_did_not_return; parent_coverage: runner 74%; destructive_node_probe 26%; needed: FAILED/missing-state RESET row must reject despite sampler dip.; Sibling UUID disappearance, nonzero nvidia-smi exit and new attempt under an old command ID must reject.; Mock accepted-injection timeout, abort, failed restore and final taint/UID drift; assert cleanup and persisted FAIL with no extra action.
- Findings At Review: id: D001-LEDGER; severity: P1; symbol: host_errors; finding: A FAILED attempt=1 RESET_GPU row plus unrelated enumeration dip passes; success, target and incident identity are unchecked.; id: D001-TARGET; severity: P1; symbol: destructive_node_probe.gpu_sample / ledger_rows; finding: Sampler records counts, not UUID/BDF presence; ledger omits gpu_uuids/fence/workflow. Nonzero nvidia-smi with empty stdout yields count=0 and can satisfy the dip.; id: D001-VALIDATION; severity: P2; symbol: workflow_errors; finding: Spec readiness_after per-GPU samples, node_baselines and executor Kubernetes timeout values are not asserted.
- Reviewer Handoff: state: fixed_owned_evidence; implementation: Successful attempt-one reset ledger, exact target UUID disappearance with siblings present, unchanged final UUID/BDF inventory, incident/workflow binding, deadline gates and case-scoped HostProbe ownership.; tests: tests/regional/test_destructive_evidence_review.py::test_reset_requires_a_successful_ledger_result; tests/regional/test_destructive_evidence_review.py::test_sibling_gpu_disappearance_cannot_prove_the_target_reset; remaining: No live per-GPU readiness telemetry or physical reset was observed in this review.; live_execution: NOT_RUN; prior_pass_reusable: False

### GF-REGIONAL-DESTR-002

经 provider 的真实节点 reboot

Current catalog: destructive-provider-reboot; manual; NOT_RUN.

- Purpose: Phase 13 after 001: executor-role HyperPod reboot of exactly the approved node, NodeRecovery=None, freshly observed scheduler isolation, durable SUBMITTED record and duplicate replay after replacing the submitting executor. No replace/delete; same Node UID/artifact, changed boot, all three validations before restored scheduling.
- Runner: scripts/e2e/regional/run_destr002_hyperpod_reboot.py; scripts/e2e/regional/probes/destructive_node_probe.py
- Entry Points: PREFLIGHT_PROBE / preflight_probe_errors; wait_for_submission -> restart_executor -> STORE_REPLAY_PROBE; workflow_errors / submission_target_errors / node_recovery_errors
- Assertions: Preflight distinguishes correct isolation, missing/wrong isolation and disabled reboot; requires NodeRecovery=None.; Submission wait stops at SUBMITTED or terminal workflow; submission_target_errors rejects extra targets instead of accepting overlap.; CloudTrail reboot count is polled and actor checked by shared role matcher; final Node UID/GPU capacity compared.
- Safety And Cleanup: admission: Exact confirmation, bound predecessor, node boot/UID/generation/artifact/profile plan. No renewed deadline before executor deletion; runner does not judge actual observed_isolation proof.; cleanup: Deletes host probe and reads final Ready; failed reboot remains isolated for operator, never replacement.; all_exit_gap: Pod deletion uses name without UID precondition. Full final runtime identity and abort result persistence share early-runner gaps.
- Necessity: Retain normal reboot and durable idempotency. A Store read alone is not execution of product dedupe and cannot establish replay behavior.
- Overlap: case_ids: GF-REGIONAL-DESTR-004; GF-REGIONAL-DESTR-016; GF-REGIONAL-DESTR-017; GF-REGIONAL-COLLECT-015; classification: retired_assertion_transfer_and_complements; rationale: 002 carries retired submission/idempotency assertions; 016 adds preemption, 017 is OS reboot without provider submit, COLLECT015 changes collector input. None licenses replacement.
- Tests: existing: tests/regional/test_destr002_review_fixes.py::test_destr002_replays_the_durable_record_from_the_replacement_executor; tests/regional/test_destructive_review_fixes.py::test_destr002_submission_must_name_exactly_the_target; tests/regional/test_destructive_acceptance_fixtures.py::test_destr002_duplicate_replay_requires_exact_submitted_record; parent_coverage: runner 76%; needed: Execute probe with fake installed executor/lifecycle Store and a submit tripwire; assert real product dedupe and zero provider calls.; Pin replay to replacement Pod UID; wrong/ambiguous owner prevents deletion/replay.; Reject missing result, wrong action/node/key/fence, missing validation execution and stale observed isolation.
- Findings At Review: id: D002-REPLAY; severity: P1; symbol: STORE_REPLAY_PROBE / execute_case; finding: Probe computes duplicate=all(checks) itself, never calls product dedupe, and executor_python chooses any Ready Pod rather than the proven replacement. Exact-record test asserts source strings.; id: D002-EXECUTIONS; severity: P2; symbol: workflow_errors; finding: Validation names plus workflow SUCCEEDED pass without successful validation executions or rejecting extra physical operations; missing final Agent boot_id can escape changed-boot comparison.
- Reviewer Handoff: state: fixed_owned_evidence; implementation: Invokes installed lifecycle durable replay without provider submission, pins replacement executor UID, requires successful ordered reboot/validation/scheduling executions and a nonempty changed Agent boot ID; deadline checks before executor restart.; tests: tests/regional/test_destructive_acceptance_fixtures.py::test_destr002_duplicate_replay_requires_exact_submitted_record; tests/regional/test_destr002_review_fixes.py::test_destr002_replays_the_durable_record_from_the_replacement_executor; remaining: Live provider/IRSA and physical boot evidence NOT_RUN; shared executor deletion ownership remains parent-owned.; live_execution: NOT_RUN; prior_pass_reusable: False

### GF-REGIONAL-DESTR-003

warm spare 方式的节点替换只使用客户自备的热备 GPU 节点

Current catalog: destructive-warm-spare; manual; NOT_RUN.

- Purpose: Phase 14 after 012: synthetic replacement finding drives real 8-GPU workload failover onto the only predeclared healthy matching spare. Exact local rebind, Agent revoke, one restart and unchanged provider instances/count; no provider replace or hardware damage.
- Runner: scripts/e2e/regional/run_destr003_warm_spare_failover.py; scripts/e2e/regional/warm_spare_fixture.py; scripts/e2e/regional/declare_warm_spare.py; scripts/e2e/regional/synthetic_replacement_route.py
- Entry Points: read_only_preflight -> preflight_errors; execute_case -> replacement_payload / workflow_errors; cleanup_case -> recover_incident_id / restore_fault_node
- Assertions: Checks exact declared spare, matching group/type, Ready+cordon, ACTIVE agents, NodeRecovery=None, per-replica spare/remote enabled and replace=false.; Exact operation order/strategy, SPARE_FAILOVER, activated_spare_nodes, provider_mutation_submitted=false, rebind and persisted notification.; 8->8 GPU restart context, budget=restart_count=1, workload on spare, fault quarantine/Agent REVOKED, unchanged provider inventory.
- Safety And Cleanup: admission: Declaration/route opening are separate approved operations. Plan binds both Node UIDs/provider inventory, not full runtime templates or command quiescence.; cleanup: Delete workload, release incident reservation, reactivate fault Agent, validated restore through current quarantine owner; lost incident resolved by event.; all_exit_gap: Delete/release/reactivation can precede quiescence. Any current owner is treated as successor without lineage proof. Caller must close synthetic route and eventually release spare declaration.
- Necessity: Essential positive test of the only replacement strategy. External declaration preserves stray-spare detection. Cleanup must not claim authority over an unrelated current owner.
- Overlap: case_ids: GF-REGIONAL-DESTR-004; GF-REGIONAL-DESTR-008; GF-REGIONAL-DESTR-022; GF-REGIONAL-DESTR-014; classification: positive_negative_and_lifecycle_complements; rationale: 003 proves allocation/rebinding; 008 shortage/rollback; 022 stale reservation; 014 branch exhaustion. 003 covers retired 004 revoke/rebind without destroying an instance.
- Tests: existing: tests/regional/test_destructive_acceptance_fixtures.py::test_destr003_requires_local_warm_spare_rebinding_and_notification; tests/regional/test_destr003_warm_spare_failover.py::test_restore_goes_through_the_successor_that_owns_the_node; tests/regional/test_destr003_warm_spare_failover.py::test_a_lost_incident_id_is_recovered_from_the_injected_event; parent_coverage: runner 38%; warm_spare_fixture 38%; needed: Protocol-backed success/response loss/wait timeout/release failure/foreign-owner flows.; Assert no deletion/release/reactivation until original and successor commands are terminal and lineage proven.; Execute STORE_PROBE against mock command models; output must exclude lease credentials.
- Findings At Review: id: D003-CLEANUP; severity: P1; symbol: cleanup_case / restore_fault_node; finding: No pre-delete/release quiescence barrier; foreign quarantine owner need not descend from this run.; id: D003-CREDENTIAL; severity: P1; symbol: warm_spare_fixture.STORE_PROBE; finding: Emits RemoteActionCommand.model_dump without excluding lease_token, unlike regional probe; ordinary workflow evidence can contain a live lease credential.; id: D003-SCOPE; severity: P2; symbol: preflight_errors / workflow_errors; finding: Warnings/queue/readiness/local_only proof incomplete; rebindings may contain extra mappings beyond the approved fault-to-spare pair.
- Reviewer Handoff: state: fixed_owned_cleanup_and_disclosure; implementation: Lease credential excluded before probe serialization; explicit queue evidence required; workload/spare release waits for terminal workflows and zero open commands, including lost trigger ACK; NodeMutationFixture uses UID/resourceVersion CAS and verifies observed writes.; tests: tests/regional/test_warm_spare_fixture_safety.py::test_store_probe_never_serializes_the_command_lease_credential; tests/regional/test_warm_spare_fixture_safety.py::test_failover_cleanup_does_not_release_resources_after_unproved_quiescence; tests/regional/test_warm_spare_fixture_safety.py::test_restore_recovers_an_applied_patch_whose_ack_was_lost; remaining: Managed workload ownership is identity-agent owned. Successor isolation ownership needs parent integration before accepting cross-incident cleanup as complete.; live_execution: NOT_RUN

### GF-REGIONAL-DESTR-004

【已作废】HyperPod 主动 replace 由 provider 销毁并重建实例

Current catalog: destructive; manual; SUPERSEDED.

- Purpose: Permanently prohibited provider replacement. Catalog SUPERSEDED by 013; authoritative order DO_NOT_RUN. No mode, override or maintenance approval may execute it.
- Runner: tools/regional_acceptance_plan.py; tools/run_regional_acceptance.py; tools/run_fault_test_cases.py; scripts/e2e/regional/regional_case_contract.py
- Entry Points: Parent-owned retirement validation / ExecutorKind.DO_NOT_RUN; No destructive runner exists for 004
- Assertions: Current catalog/order agree on manual/SUPERSEDED/DO_NOT_RUN.; Planner rejects overrides and produces blocked entry with no dependencies.; No plan or live entrypoint was executed by this reviewer.
- Safety And Cleanup: admission: Unconditional prohibition, not an approval gate.; cleanup: No action/cleanup; never synthesize PASS.; all_exit_gap: Parent should test an immutable ID-level prohibition even under inconsistent or simultaneously edited catalog/order.
- Necessity: Retirement is mandatory. All useful safety assertions can be proved without a replacement API or destruction of a GPU instance.
- Overlap: case_ids: GF-REGIONAL-DESTR-002; GF-REGIONAL-DESTR-003; GF-REGIONAL-DESTR-013; classification: prohibited_assertions_redistributed; rationale: 002 owns durable provider idempotency; 003 revoke/rebind; 013 configuration/IAM/CloudTrail prohibition. Keep 004 solely as immutable prohibited identifier.
- Tests: existing: tests/test_regional_acceptance_plan.py::test_override_rejects_unknown_and_do_not_run_cases; tests/regional/test_regional_execution_order.py::test_do_not_run_matches_regional_superseded_cases; tests/hyperpod/test_hyperpod.py::test_replace_env_is_rejected_by_the_design_invariant; parent_coverage: No owned executable path; needed: Parent protocol tests for formal/selective/explicit-case/override/inconsistent metadata: action spy stays empty and 004 always refuses.
- Findings At Review: id: D004-PROHIBITION; severity: P1; symbol: parent planner/dispatcher gates; finding: Do not make 004 runnable through ordinary configurable metadata. Add immutable-ID regression centrally; never add a replacement runner.
- Reviewer Handoff: state: prohibition_preserved; implementation: No executable replacement case added; immutable DO_NOT_RUN requirement handed to the parent catalog/guard owner.; live_execution: DO_NOT_RUN

### GF-REGIONAL-DESTR-005

warm spare step 被路由到 managed 或 provider adapter 时必须失败

Current catalog: live-non-destructive; command; NOT_RUN.

- Purpose: First entry of phase13: in-memory HEALTHY_WARM_SPARE_ONLY REPLACE_NODE handed to deployed ManagedRecoveryObserverAdapter must fail with the exact delegation refusal. No persistent workflow, workload, scheduler or provider mutation.
- Runner: scripts/e2e/regional/audit_warm_spare_guardrails.py
- Entry Points: main -> run_audit; deployed_managed_owner_probe; probe_errors / node_preflight_errors / node_state_drift
- Assertions: Deployed probe and fixed pytest require FAILED and healthy warm-spare replacement cannot be delegated to managed/provider node recovery.; Node baseline/postflight compare UID, Ready, GPU allocatable, cordon, taints and ownership; dirty baseline fails.; Case-specific probe outcome is separate from shared reads; recent empty CloudTrail reads are provisional.
- Safety And Cleanup: admission: Explicit CPU/GPU contexts; read-only/in-memory probe. No mutation is intended. Formal admission/maintenance window is parent-controlled.; deadline: No mutation interface should be reachable; requires a protocol tripwire test rather than assuming a Python exec is read-only.; cleanup: Common postflight is sequenced at the end, not protected by an outer finally; an abort or artifact writer failure can bypass it.
- Necessity: Retain deployed-wheel routing denial. Old real-workload/quarantine setup is invalid and unnecessary.
- Overlap: case_ids: GF-REGIONAL-DESTR-006; GF-REGIONAL-DESTR-007; GF-REGIONAL-DESTR-013; classification: distinct_guard_layers; rationale: 005 blocks wrong owner; 006 wrong provider recovery mode; 007 missing dependencies; 013 configuration/IAM/full-window denial.
- Tests: existing: tests/execution/test_misc.py::test_managed_recovery_rejects_warm_spare_replacement; tests/regional/test_warm_spare_guardrails.py::test_one_probe_failure_fails_only_the_case_that_depends_on_it; parent_coverage: audit 68%; needed: Execute captured probe against fake installed modules with Store/Kubernetes/provider mutation tripwires.; Dirty baseline must prevent deployed calls; abort must still attempt postflight.; Parent formal execution must stop before006 after005 failure, separately from diagnostic aggregation.
- Findings At Review: id: D005-FAILSTOP; severity: P2; symbol: run_audit; finding: Collected preflight errors do not stop probes. Default multi-case invocation aggregates failures instead of formal fail-stop semantics.; id: D005-PROVIDER; severity: P2; symbol: replace_events; finding: Only replace verbs are checked; an unexpected reboot/delete is not judged under the no-mutation promise.
- Reviewer Handoff: state: reviewed_parent_integration; implementation: Audited shared warm-spare guardrail runner and existing behavioral tests; no equivalence with destructive failover claimed.; remaining: Parent-owned shared audit admission/fail-stop and all-provider-event checks.; live_execution: NOT_RUN
- Current Disposition: Shared read-only audit now binds durable registry/release/predecessor, checks complete Ready population and Pod UID, validates provider-record identity/freshness/pagination, records all provider mutation verbs and refuses implicit multi-case CLI execution.

The follow-up applies to DESTR-005/006/007 together. Local prerequisites now use
supervised source-bound pytest receipts with complete discovery and all three
passing phases, not exit status or parsed failure text. Independent cases retain
their own results, but collection loss, source drift and unexplained process
failure cannot establish PASS. Both node snapshots require nonblank, unique names
and UIDs before comparison; preflight refuses ambiguous identities before probes.
GPU resource declarations retain zero-allocatable nodes in the comparison;
unreadable quantities or incomplete inventory cannot shrink the audited population.
The CLI also binds the durable supervision marker before external reads, with a
current non-PASS result already recorded; a fresh retry cannot bypass prior loss
of command custody or reuse an old PASS after configuration refusal.
The regressions in `tests/regional/test_cov95_warm_guard_reads.py`,
`tests/regional/test_cov95_warm_guard_flow.py` and
`tests/regional/test_cov95_warm_pytest_boundary.py` use fake live reads and isolated
local pytest children. `tests/regional/test_cov95_warm_peer_boundaries.py` adds
fresh-process checks of the real supervision recorder and retry gate. They do not
establish physical failover or LIVE acceptance.

### GF-REGIONAL-DESTR-006

warm spare 要求 HyperPod NodeRecovery=None

Current catalog: live-non-destructive; command; NOT_RUN.

- Purpose: After005: record three read-only SageMaker APIs from an already-Automatic unmanaged cluster, replay into deployed executor, and distinguish warm-spare NodeRecovery refusal from the otherwise-identical non-warm-spare control refusal. Never change NodeRecovery or enlarge IRSA.
- Runner: scripts/e2e/regional/audit_warm_spare_guardrails.py
- Entry Points: record_provider_snapshot; deployed_automatic_recovery_probe / RecordedProviderClient / guard_outcome; probe_errors
- Assertions: Projected snapshot carries recorded_at and payload_digest; client exposes only DescribeCluster, paged ListClusterNodes and DescribeClusterNode.; Require observed_node_recovery=Automatic, exact warm-spare failure and different control error containing HyperPod automatic node recovery is enabled.; Managed cluster remains None and GPU node state must compare equal.
- Safety And Cleanup: admission: Read-only auditor recording avoids expanding executor IRSA to external clusters.; deadline: No planned mutation; tripwire tests must prove replay has no mutation method even on unexpected data.; cleanup: Common postflight gap applies; exclusion of negative cluster from registry and its unchanged state are not actually read/compared.
- Necessity: Paired control makes refusal attributable. Replaying authentic provider observations is preferable to changing managed recovery settings.
- Overlap: case_ids: GF-REGIONAL-DESTR-005; GF-REGIONAL-DESTR-007; GF-REGIONAL-BOOT-005; GF-REGIONAL-DESTR-013; classification: deployed_preflight_negative; rationale: Tests adapter behavior on observed Automatic recovery, not only startup/config/IAM denial.
- Tests: existing: tests/execution/test_node_action.py::test_warm_spare_rejects_automatic_node_recovery_before_allocation; tests/regional/test_warm_spare_guardrails.py::test_the_negative_cluster_is_read_from_the_recorded_snapshot_not_described_twice; parent_coverage: audit 68%; needed: Two-page provider fixture, wrong-cluster response, missing NodeRecovery, bad digest, repeated page token and stale snapshot.; Execute both adapter strategies over the same recording and assert zero mutations.; Registered negative cluster must refuse before replay.
- Findings At Review: id: D006-IDENTITY; severity: P2; symbol: configure / record_provider_snapshot / deployed_automatic_recovery_probe; finding: No registry exclusion proof or independent recording freshness/digest verification; most tests replace the deployed probe with a prebuilt outcome dictionary.
- Reviewer Handoff: state: reviewed_parent_integration; implementation: Reviewed the real negative NodeRecovery guard and isolated-cluster boundary; retained separately from configuration-only 005/007.; remaining: Identity-owner proof that the negative cluster is excluded from the live registry and that the recording is independently fresh.; live_execution: NOT_RUN
- Current Disposition: Shared read-only audit now binds durable registry/release/predecessor, checks complete Ready population and Pod UID, validates provider-record identity/freshness/pagination, records all provider mutation verbs and refuses implicit multi-case CLI execution.

### GF-REGIONAL-DESTR-007

warm spare 要求 spare coordinator 与远端状态均已启用

Current catalog: live-non-destructive; command; NOT_RUN.

- Purpose: After006: isolated executor startup must reject spare_failover=true/remote_state=false; warm-spare step without coordinator must fail before allocation/provider submit. Production replicas/nodes stay unchanged.
- Runner: scripts/e2e/regional/audit_warm_spare_guardrails.py
- Entry Points: deployed_executor_guard_probes; case_definitions / run_focused_pytest / probe_errors; run_audit
- Assertions: Startup failure names required GPU_FAULT_CLUSTER_EXECUTOR_REMOTE_STATE=true.; Step outcome is FAILED with healthy warm-spare replacement is required but the spare coordinator is disabled.; Two running executor replicas report spare_failover=true, remote_state=true, allow_replace=false; node snapshots identical.
- Safety And Cleanup: admission: Overrides/stubs are process-local, never a Deployment rollout or durable command.; deadline: No intended mutation; assert no AWS/Store/Kubernetes write construction/reachability using probe protocol tests.; cleanup: Environment is sampled once; no before/after full replica proof. Common abort/postflight gap applies.
- Necessity: Both subguards are needed and distinct. The retired live-isolation setup must not return.
- Overlap: case_ids: GF-REGIONAL-DESTR-005; GF-REGIONAL-DESTR-006; GF-REGIONAL-DESTR-008; classification: dependencies_vs_capacity; rationale: 007 proves dependencies cannot silently degrade; 008 proves real allocation refusal when dependencies exist.
- Tests: existing: tests/hyperpod/test_cluster_executor.py::test_executor_spare_failover_requires_remote_state; tests/execution/test_node_action.py::test_warm_spare_requires_enabled_coordinator_before_submission; tests/regional/test_warm_spare_guardrails.py::test_the_focused_tests_run_in_one_process_and_are_attributed_per_case; parent_coverage: audit 68%; needed: Run embedded bootstrap/adapter scripts with injected dependency protocols.; Replica/environment change and failed postflight must fail.; Literal two replicas should be reconciled with desired/Ready Deployment membership, never silently skip an unready replica.
- Findings At Review: id: D007-REPLICAS; severity: P2; symbol: executor_env / run_audit; finding: One environment read and literal replica count cannot prove all actual replicas remained consistent throughout the case.
- Reviewer Handoff: state: reviewed_parent_integration; implementation: Reviewed installed executor safety environment checks; no new broad action authorization.; remaining: Parent-owned shared audit must enumerate every current replica rather than rely on a literal expected count.; live_execution: NOT_RUN
- Current Disposition: Shared read-only audit now binds durable registry/release/predecessor, checks complete Ready population and Pod UID, validates provider-record identity/freshness/pagination, records all provider mutation verbs and refuses implicit multi-case CLI execution.

### GF-REGIONAL-DESTR-008

热备池不足时失败并告警绝不降级为 provider replace

Current catalog: destructive-warm-spare; manual; NOT_RUN.

- Purpose: Phase14 after003: all six shortage causes must fail REPLACE_NODE after successful STOP, with distinct reasons, no provider fallback, original isolation and no partial reservation. S1 has zero shortage alerts; S2-S6 one; only the injected finding marker belongs to the incident.
- Runner: scripts/e2e/regional/run_destr008_warm_spare_shortage.py; scripts/e2e/regional/warm_spare_fixture.py; scripts/e2e/regional/probes/warm_spare_node_probe.py
- Entry Points: ScenarioFixture.apply / apply_late / restore; run_scenario -> scenario_errors / bound_errors; execute_case
- Assertions: Matrix includes no-spare, topology mismatch, NotReady, reserved-by-other, active GPU Pod and Agent unavailable; first non-PASS stops subsequent scenarios; partial selection cannot PASS.; Finite outages are armed only after workload Running. wait_timeout_seconds/bound_errors require replacement conclusion before outage expiry.; Assert real STOP success, exact strategy, cause-specific failure, alert count, sole finding marker, no owned reservation/ALLOCATED residue and unchanged provider inventory.
- Safety And Cleanup: admission: Only the predeclared spare is perturbed; service unit allowlist and 60..600-second recovery timer; delayed kubelet stop keeps response channel alive.; deadline: Added checks before prewarm, shortage setup, workload submission and replacement trigger after apply_late. Existing check remains before bounded service action. Per-command expiration during helpers needs parent supervision.; cleanup: Per-scenario workload/service/metadata/spare restore is in finally, but lacks original-command quiescence proof. Restoring availability after runner timeout can let an in-flight replacement succeed.
- Necessity: Retain six distinct refusal causes. Fixed allocator-conflict pytest is complementary, not live atomicity evidence. Runner timeout cannot be equated with cancelled allocation.
- Overlap: case_ids: GF-REGIONAL-DESTR-003; GF-REGIONAL-DESTR-014; GF-REGIONAL-DESTR-022; classification: negative_matrix_and_lifecycle_complements; rationale: 008 allocator refusal; 014 embeds zero-spare in branch exhaustion; 022 TTL reclamation. Each has a separate causal claim.
- Tests: existing: tests/regional/test_destr008_shortage_matrix.py::test_a_service_scenario_stops_the_unit_only_after_the_workload_is_running; tests/regional/test_destr008_shortage_matrix.py::test_the_workflow_wait_never_outlives_the_shortage_bound; tests/regional/test_destr008_shortage_matrix.py::test_notification_semantics_distinguish_no_pool_from_shortage; parent_coverage: runner35%; service probe75%; needed: Full six-scenario protocol simulation with first-failure/partial-selection checks.; Response loss or timeout must not restore capacity while a command can allocate it.; Restore failure must preserve recovery timer; UID/CAS/foreign reservation change must prevent metadata overwrite.
- Findings At Review: id: D008-INFLIGHT; severity: P1; symbol: run_scenario finally / ScenarioFixture.restore; finding: Restoring shortage does not first establish original workflow/commands terminal; expiry timer independently restores it too. Needs shared cancellation/quiescence design.; id: D008-TIMER; severity: P1; symbol: warm_spare_node_probe.restore_service; finding: Cancels fallback timer before active-state verification succeeds; parent common-helper handoff.; id: D008-RUN-ID; severity: P2; symbol: ScenarioFixture/prewarm run_id construction; finding: Scenario+attempt omits run-dir identity and can collide across campaigns.
- Reviewer Handoff: state: fixed_fail_closed_bounded_variants_unavailable; implemented: UID/CAS/ACK-loss guards, unique run identity, per-stage deadlines, quiet workflow/command barrier before deleting workload or restoring shortage, fallback start timer retained until service is actually active. active-gpu-pod, kubernetes-not-ready and agent-unavailable now refuse before injection because their independent expiry can enable a still-running replacement.; tests: tests/regional/test_destructive_runner_review.py::test_expiring_shortage_refuses_without_independent_cancellation; tests/regional/test_warm_spare_fixture_safety.py::test_cleanup_waits_for_commands_even_after_the_workflow_is_terminal; tests/regional/test_warm_spare_node_probe.py::test_failed_restore_keeps_the_independent_start_timer; remaining: A proven independent cancellation/fencing design is required to enable the three bounded variants; see proposal DESTR-PROP-003.; live_execution: NOT_RUN
- Current Disposition (2026-09-13): All requests carry immutable activation inhibition, including old-protocol claim exclusion and cached confirmation protection. The three expiring variants now compose a separately running CPU cancellation observer, full source/producer/quiet-state receipts, a Node-bound GPU fence and independently owned fixture recovery. Workload readiness precedes arming; fence/observer closure precedes restoration. The GPU fence is defense in depth, not proof of cluster-wide admission convergence. Valid direct create ACKs preserve deletion-only custody when spec approval fails; unknown ACKs cannot be adopted. Existing runs are cleanup-only, arbitrary quarantine owners are not treated as descendants, and every scenario's resource-closure proof is required for matrix PASS. Local causal and native PostgreSQL tests are not LIVE acceptance. See [DESTR008 Cancellation And Inhibition](destr008-cancellation.md).

### GF-REGIONAL-DESTR-009

workload 停止与重启经 executor 本地执行

Current catalog: live-workload-restart; manual; NOT_RUN.

- Purpose: Phase13 after002: API-replayed XID11 stops/restarts a real three-node24-GPU PyTorchJob through GPU executor. One shared stop/restart, old/new Pod UIDs disjoint, NCCL recovery, budget+1, no reset/reboot/spare, unchanged CPU EKS and runtime identity throughout.
- Runner: scripts/e2e/regional/run_destr009_workload_restart.py; scripts/e2e/regional/managed_workload_fixture.py
- Entry Points: execute_case -> xid11_payload / workflow_errors; wait_observation / remote_execution_evidence; cleanup_case -> wait_for_cleanup_quiescence
- Assertions: OWN workload capabilities, empty GPU workloads/fault queue, three candidate nodes, observation with24 distinct UUIDs.; Core operations exact once, forbidden node mutations absent, fast step requires terminal remote/ ID and successful command.; New UIDs disjoint, three-node distribution, NCCL heartbeat/all-reduce and budget1/1; runtime checked before replay, after restart and after cleanup.; Log verdict rejects empty/inconclusive reads for inspected Pods.
- Safety And Cleanup: admission: Schema3 parent approval binds arguments and successful preflight; execute compares candidate UIDs and runtime identity.; deadline: Added prewarm/submission checks; existing injection check remains after runtime verification. Parent transport supervision must cover individual mutations inside long helpers.; cleanup: Requires terminal source observation/workflow/commands and15s stable state before delete; lack of proof defers deletion and fails. Prewarm cleanup remains independent.
- Necessity: Retain workload-owner baseline; label API replay accurately. Quiescence deferral is necessary to avoid racing an active restart.
- Overlap: case_ids: GF-REGIONAL-DESTR-012; GF-REGIONAL-DESTR-015; GF-REGIONAL-COLLECT-016; classification: prerequisite_reuse_and_complementary_paths; rationale: 012A consumes009 evidence;015 adds parallel hardware branches/join;COLLECT016 adds collection-path proof.
- Tests: existing: tests/regional/test_destr009_review_fixes.py::test_destr009_evidence_records_the_step_owners_of_the_matched_workflow; tests/regional/test_destructive_acceptance_fixtures.py::test_destr009_cleanup_defers_delete_without_quiescence; tests/regional/test_destructive_acceptance_fixtures.py::test_destr009_rejects_fast_step_without_remote_command_evidence; parent_coverage: runner81%; needed: Unrelated command cannot satisfy remote evidence.; Parsed executing container image must match digest; a comment/sidecar digest must not suffice.; Missing expected log app or failed residual read must be inconclusive/FAIL; workload helper owned by identity agent.
- Findings At Review: id: D009-IMAGE; severity: P2; symbol: read_only_preflight; finding: Text membership of image digest is weaker than parsing the actual executing containers.; id: D009-LOGS; severity: P2; symbol: log_write_snapshot / log_write_errors; finding: An expected app with zero Pods is omitted when other apps return lines; missing/unknown log verdict is not rejected.
- Reviewer Handoff: state: fixed_owned_deadline_parent_workload_integration; implemented: Deadline gates before initial workload mutations retained; current managed-workload/prewarm ownership interfaces exercised through protocol fakes.; remaining: Image/container and workload-ownership validation is identity-agent owned; parent consolidates log-population completeness.; live_execution: NOT_RUN

### GF-REGIONAL-DESTR-010

Fabric Manager 重启与 GPU 服务静默经 Node Agent

Current catalog: live-service-action; manual; NOT_RUN.

- Purpose: Phase13 before001: solo kmsg XID45 resolves to RESTART_FM and exactly FREEZE_EVIDENCE+RESTART_FABRIC_MANAGER. One Node Agent action, service restart and ACTION_COMPLETED notification; direct cached replay must leave ledger, PID/InvocationID/start count unchanged and never isolate/quiesce.
- Runner: scripts/e2e/regional/run_destr010_fabric_manager_restart.py; scripts/e2e/regional/probes/node_host_probe.py
- Entry Points: fabric_probe / validate_preflight; execute_case -> workflow_errors / host_errors / node_state_errors; replay_command -> REPLAY_SCRIPT
- Assertions: Quiet companion window, correct OWN/allowlist, executable RESTART_FM with kmsg evidence, exact steps/owners and unique successful command.; Replay is one attempt; notification IDs unchanged.; Now compares whole ledger across replay, InvocationID, MainPID, active state and journal started_count=1; repeated attempt under same command ID cannot disappear in a set comparison.
- Safety And Cleanup: admission: Plan binds Node UID/generation/profile/release; full template/lease proof remains weaker than009.; deadline: Added probe-creation and direct replay checks; initial injection was already checked. Fresh generation proof immediately before replay is still needed.; cleanup: finally ensures FM active, removes probe and rechecks node state. Independent recovery if runner/host disappears and abort-result persistence remain gaps.
- Necessity: Retain the lowest service-action prerequisite. Cached replay is useful only when all physical and ledger observations are unchanged, not merely command-ID sets.
- Overlap: case_ids: GF-REGIONAL-DESTR-019; GF-REGIONAL-COLLECT-005; classification: same_action_distinct_claims; rationale: 019 uses FM to audit Agent restart/schema/health;COLLECT005 validates raw collection. Shared action does not replace those assertions.
- Tests: existing: tests/regional/test_destr010_review_fixes.py::test_execute_case_replays_the_command_once_after_the_first_action; tests/regional/test_destr010_review_fixes.py::test_replay_command_runs_the_adapter_with_a_single_attempt; added: tests/regional/test_destructive_evidence_review.py::test_fabric_replay_cannot_hide_an_attempt_under_an_existing_command_id; tests/regional/test_destructive_evidence_review.py::test_fabric_replay_compares_service_and_ledger_evidence; parent_coverage: runner74%
- Findings At Review: id: D010-REPLAY; severity: P1; symbol: host_errors; finding: Original set comparison hid attempt2 and changed rows/journal/InvocationID.; resolution: Fixed and behaviorally reproduced.; id: D010-FRESHNESS; severity: P2; symbol: execute_case -> replay_command; finding: Deadline now rechecked; exact current lease/generation/artifact should also be observed before replay.
- Reviewer Handoff: state: fixed_owned_replay_evidence; implemented: Replay compares complete attempt ledger, MainPID, InvocationID, active service state and exactly one journal start, with renewed deadline gate and case-scoped HostProbe state.; tests: tests/regional/test_destructive_evidence_review.py::test_fabric_replay_cannot_hide_an_attempt_under_an_existing_command_id; tests/regional/test_destructive_evidence_review.py::test_fabric_replay_compares_service_and_ledger_evidence; remaining: Live Agent lease/release identity and service recovery need renewed approved execution.; live_execution: NOT_RUN; prior_pass_reusable: False

### GF-REGIONAL-DESTR-011

已并入：provider replace应用不变量与IAM拒绝

Current catalog: non-destructive; manual; SUPERSEDED.

- Purpose: Retired duplicate of013: authoritative order DO_NOT_RUN, catalog SUPERSEDED. Legacy read-only diagnostics must never create a separate formal acceptance PASS.
- Runner: scripts/e2e/regional/audit_destr011_provider_replace.py; scripts/e2e/regional/audit_destr013_replacement_invariant.py
- Entry Points: Legacy diagnostic run_case only; Formal dispatcher uses013 and refuses011
- Assertions: 013 contains independently enabled reboot, per-replica replace=false, scoped IAM denial and full-window CloudTrail absence.; Legacy output now has verdict=SUPERSEDED, diagnostic_verdict separate, diagnostic_only=true and formal_sequence_satisfied=false.
- Safety And Cleanup: admission: Never a formal case; diagnostics still access live APIs and were not executed here.; deadline: No mutation permitted; separate diagnostic status cannot authorize a downstream formal case.; cleanup: Read-only inventory comparison; no physical resources.
- Necessity: Keep retired. Separate formal PASS would inflate coverage without a distinct safety assertion.
- Overlap: case_ids: GF-REGIONAL-DESTR-013; classification: fully_superseded_duplicate; rationale: 013 strictly contains useful checks over a wider explicit window with route closure and positive control.
- Tests: existing: tests/regional/test_audit_destr013_replacement_invariant.py::test_destr011_is_recorded_as_superseded; tests/hyperpod/test_hyperpod.py::test_provider_replace_can_be_disabled_without_disabling_reboot; added: tests/regional/test_destructive_runner_review.py::test_retired_diagnostic_never_writes_a_formal_pass; reproduction: Both healthy and failed mocked diagnostic paths formerly emitted PASS/FAIL; now always SUPERSEDED with separate diagnostic result.
- Findings At Review: id: D011-VERDICT; severity: P2; symbol: run_case; finding: Legacy formal-looking PASS/FAIL could be counted despite superseded_by.; resolution: Fixed without running live diagnostics.
- Reviewer Handoff: state: fixed_retired_verdict; implementation: Diagnostic PASS cannot become formal PASS; record remains SUPERSEDED with formal_sequence_satisfied=false.; live_execution: NOT_RUN; catalog_verdict: SUPERSEDED

### GF-REGIONAL-DESTR-012

任务自动恢复由本方案独占且 HyperPod auto-resume 必须处于禁用状态

Current catalog: live-workload-restart; manual; NOT_RUN.

- Purpose: Phase13 after009; strict B->A->D->C. B proves all allowed namespaces/profiles comply, A consumes formal same-release009 owner evidence, D rejects auto-resume conflict before writes then permits a fresh retry after removal; C is optional independent-control-plane DELEGATE proof and never production Profile mutation.
- Runner: scripts/e2e/regional/run_destr012_managed_recovery_guard.py; scripts/e2e/regional/run_destr009_workload_restart.py; scripts/e2e/regional/managed_workload_fixture.py
- Entry Points: group_b_audit; group_a_from_evidence / explicit run_group_a; run_group_d -> group_d_failure_errors; execute_case / groups_not_run
- Assertions: B records profile SHA per API/control-worker replica and namespace/object annotations; A requires evidence_valid and Kubernetes STOP/RESTART owners.; D requires FAILED STOP, only violating workload, exact required annotation/value/remediation command, unchanged suspend/Pod UIDs and conclusive clean executor logs.; Retry has a new record marker and must finish real restart. Runtime identity is checked around A/D and cleanup; failed A stops D.
- Safety And Cleanup: admission: Schema3 plan/arguments and release-template identity. Never enable provider recovery or change production Profile.; deadline: Added submission/annotation/prewarm/retry checks; expired second phase cannot inject. Annotation removal remains allowed cleanup after expiry.; cleanup: Only group quiescence gate deletes workload; outer finally audits residuals and runtime. Annotation-removal failures are now recorded, not swallowed.
- Necessity: B/A/D jointly prove exclusive recovery, deny unsafe configuration and allow corrected state again. A reuse eliminates duplicate24-GPU mutation; C stays explicitly optional.
- Overlap: case_ids: GF-REGIONAL-DESTR-009; GF-REGIONAL-DESTR-013; classification: evidence_reuse_plus_distinct_guard; rationale: A intentionally reuses009; B/D add task recovery ownership. 013 covers node/provider invariants, not auto-resume.
- Tests: existing: tests/regional/test_destr012_review_fixes.py::test_execute_reads_group_a_from_evidence_unless_the_rerun_flag_is_set; tests/regional/test_destr012_review_fixes.py::test_group_d_deletes_its_workload_only_through_the_quiescence_gate; added: tests/regional/test_destr012_review_fixes.py::test_group_d_never_reinjects_after_the_negative_guard_failed; tests/regional/test_destr012_review_fixes.py::test_group_d_rechecks_the_window_before_the_remediated_injection; parent_coverage: runner68%; reproduction: Both new tests failed before fix: second injection occurred after guard failure and after window expiry.
- Findings At Review: id: D012-FAILSTOP; severity: P1; symbol: run_group_d; finding: Continued after negative assertion failure and expired retry deadline.; resolution: Fixed; mock call counts assert one injection and cleanup.; id: D012-C; severity: P2; symbol: execute_case group_c; finding: C has no real isolated runner path; retained as explicit optional NOT_RUN, separate proposed integration test.
- Reviewer Handoff: state: fixed_owned_fail_stop; implemented: D refuses the second injection after any failed negative assertion or expired deadline; cleanup failures cannot be hidden. B/A/D/C ordering unchanged.; tests: tests/regional/test_destr012_review_fixes.py; remaining: Optional C remains explicitly NOT_RUN until an isolated-cluster fixture exists; identity agent and parent own namespace/replica integration.; live_execution: NOT_RUN

### GF-REGIONAL-DESTR-013

硬约束否证 NodeRecovery 恒为 None 且全程不存在 BatchReplaceClusterNodes

Current catalog: non-destructive; manual; NOT_RUN.

- Purpose: Last entry of phase15 after HA004: read-only falsification of NodeRecovery!=None and every provider replacement. Requires actual per-replica replace=false/reboot=true, closed synthetic route, cluster-scoped IAM deny/positive reboot control, unchanged inventory and source manifest invariants over the complete earlier destructive time window.
- Runner: scripts/e2e/regional/audit_destr013_replacement_invariant.py
- Entry Points: main; environment_errors / synthetic_route_errors; iam_decisions / cloudtrail_events / window_errors / coverage_errors / manifest_invariants
- Assertions: Requires InService/NodeRecovery=None; every observed executor denies replacement and independently enables reboot; every API replica has synthetic-route variable absent.; IAM simulation uses the actual supplied cluster ARN, not '*'; replacement deny and reboot allowed are distinct assertions.; Explicit forbidden verbs are checked, including delete/update; recent window refuses unless marked provisional; missing positive reboot control needs explicit waiver.; All readable recorded destructive timestamps must lie in supplied window; manifest overrides and inventory changes fail.
- Safety And Cleanup: admission: No fault injection or resource mutation; parent must bind predecessor, actual GPU cluster and actual IRSA role to the supplied identities.; deadline: Read-only; query window completeness/freshness is the relevant bound, not mutation approval.; cleanup: No resource cleanup. Failure/abort should still leave a scoped incomplete audit record.
- Necessity: Required final prohibition proof, not a replacement exercise. Empty recent or wrong-cluster CloudTrail is never sufficient evidence of absence.
- Overlap: case_ids: GF-REGIONAL-DESTR-004; GF-REGIONAL-DESTR-011; GF-REGIONAL-BOOT-005; classification: final_umbrella_falsification; rationale: Replaces004 entirely and subsumes011. Individual case provider reads remain provisional until this full-window audit.
- Tests: existing: tests/regional/test_audit_destr013_replacement_invariant.py::test_an_empty_window_needs_a_positive_control_or_an_explicit_waiver; tests/regional/test_audit_destr013_replacement_invariant.py::test_the_window_must_cover_the_runs_recorded_timestamps; tests/regional/test_audit_destr013_replacement_invariant.py::test_every_executor_replica_must_independently_enable_reboot; parent_coverage: audit61%; needed: Mock main with correct IAM answers for the wrong role/cluster and require rejection.; Malformed/unreadable case evidence must make coverage incomplete, not disappear.; Wrong-cluster positive reboot must not establish CloudTrail coverage; complete timestamps and mature window are mandatory.
- Findings At Review: id: D013-BINDING; severity: P1; symbol: main / cloudtrail_events; finding: Predecessor call omits expected release/cluster and audit arguments are not independently bound to actual Kube/IRSA identity. Region-wide positive control can come from another cluster.; id: D013-COVERAGE; severity: P2; symbol: run_timestamps / coverage_errors; finding: Unreadable JSON or unparseable timestamps are skipped; any nonempty remaining timestamp set can pass while another case is unaccounted for.
- Reviewer Handoff: state: reviewed_parent_audit_integration; implemented: Reviewed provider-replacement invariant audit and its event-window tests; no provider mutation or live diagnostic was run.; remaining: Parent must bind the aggregate audit to release/cluster/IRSA and reject partial/unreadable predecessor timestamp coverage.; live_execution: NOT_RUN
- Current Disposition: Final integration binds actual registration/EKS/context/IRSA/release and complete Ready populations, rejects incomplete timestamp or CloudTrail identity/pagination evidence, persists initial FAIL and supervision loss, and never promotes provisional windows to formal PASS. 108 focused tests passed.

### GF-REGIONAL-DESTR-014

同一作业两节点并行修复：一支就地升级到真实 reboot 并收口，另一支耗尽后整条 FAILED 且不重启作业

Current catalog: destructive-provider-reboot; manual; NOT_RUN.

- Purpose: Phase14 after008 and spare release: two16-GPU job branches. Fault branch reset refusal escalates to real reboot and restores; sibling reboot returns Node Ready without Agent, times out then exhausts zero-spare replacement. Whole workflow FAILED, no join/restart, two executor reboots, no replace, bounded cleanup.
- Runner: scripts/e2e/regional/run_destr014_branch_exhaustion.py; scripts/e2e/regional/destr014_verdicts.py; scripts/e2e/regional/probes/destr014_node_probe.py
- Entry Points: preflight_errors / _arm_and_inject / execute_case / _cleanup; workflow_errors / host_errors / cloudtrail_errors / follow_up_errors; disable_agent_restart / restore_agent
- Assertions: Exact branch escalation counts, exhausted sibling, failed reset with client refusal, successful fault reboot/validations, bounded sibling timeout and zero-spare failure; no RESTART_WORKLOAD execution.; Host/Node boot changes, Agent disable/restore state, isolated sibling vs restored fault, budget unchanged and exactly two reboot records.; Critical missing safeguard is now explicit preflight failure; disable_agent_restart refuses before any command.
- Safety And Cleanup: admission: BLOCKED: no implemented reboot-surviving independent recovery for disabled Node Agent. Ordinary transient timer cannot survive the reboot this case intentionally causes.; deadline: Added case-setup/configuration checks; cleanup flags are recorded before env/holder/Agent mutation calls, including lost ACK.; cleanup: Old restore-agent path retained. Cleanup now treats true residual maps as errors. New unsafe Agent disable cannot occur; positive live case requires a reviewed independent recovery implementation.
- Necessity: Branch-exhaustion claim is valuable and distinct, but current physical construction was unsafe. Fail-closed blocking is required rather than inventing a timer guarantee or weakening failure assertions.
- Overlap: case_ids: GF-REGIONAL-DESTR-015; GF-REGIONAL-DESTR-008; GF-REGIONAL-DESTR-002; classification: negative_join_and_escalation_complement; rationale: 015 proves successful join,014 proves exhausted sibling prevents join;008 supplies zero-spare semantics and002 normal reboot.
- Tests: existing: tests/regional/test_destr014_branch_exhaustion.py::test_destr014_workflow_contract_requires_sibling_exhaustion_without_restart; tests/regional/test_destr014_node_probe.py::test_agent_baseline_record_and_restore_actions; added: tests/regional/test_destr014_branch_exhaustion.py::test_preflight_refuses_without_a_reboot_surviving_recovery_safeguard; tests/regional/test_destructive_runner_review.py::test_agent_disable_refuses_without_a_reboot_surviving_safeguard; tests/regional/test_destructive_runner_review.py::test_expired_case_does_not_start_mutation_but_still_cleans_up; parent_coverage: runner42%; verdict94%; probe56%
- Findings At Review: id: D014-FAILSAFE; severity: P1; symbol: disable_agent_restart; finding: Original function only disabled Agent; no independent recovery existed.; resolution: Unsafe entrypoint and preflight now refuse; restoring older runs remains supported.; id: D014-AUDIT; severity: P2; symbol: cloudtrail_errors / host_errors; finding: Two events alone do not bind one reboot per node or actor; detailed holder chronology/journal proof and full runtime identity remain incomplete.; id: D014-SPEC; severity: P2; symbol: central spec/catalog; finding: 300/600-second bounds, reset barrier versus commit, and promised failsafe diverge; parent must document blocked status.
- Reviewer Handoff: state: fixed_fail_closed_unavailable; implemented: Both preflight and disable-agent-restart refuse without a reboot-surviving independent safeguard. Legacy restore remains available. ACK-loss obligations are set before RPC, residual cleanup maps are checked, and workload deletion/isolation cleanup defer if command quiescence cannot be proved.; tests: tests/regional/test_destructive_runner_review.py::test_agent_disable_refuses_without_a_reboot_surviving_safeguard; tests/regional/test_destructive_runner_review.py::test_expired_case_does_not_start_mutation_but_still_cleans_up; remaining: Do not enable live DESTR014 until an independently owned reboot-surviving recovery mechanism is implemented and audited; transient systemd-run is insufficient.; live_execution: NOT_RUN
- Current Disposition (2026-09-13): The guarded entry now owns a persistent systemd recovery service/timer and stdlib helper independent of the runner, probe Pod and Agent venv. A separate service invocation must acknowledge the original Node/boot/runtime binding before the original enable-link inode is disabled. The fixed restore/expiry deadlines cannot be renewed. Recovery before scenario completion is failure; return after expiry requires manual reconciliation. UID/inode drift, pending jobs and unknown supervision block cleanup. Existing attempts resume cleanup only, and unbound legacy Agent actions now refuse. Focused reboot/controller-loss tests pass; no real reboot was performed. See [DESTR-014 Recovery Safeguard](destr014-recovery-safeguard.md).

### GF-REGIONAL-DESTR-015

同一作业两节点同窗故障：双分支并行 GPU reset，单次 join 重启作业

Current catalog: destructive; manual; NOT_RUN.

- Purpose: Phase13 after012: two same-window kmsg XID46 events for one16-GPU job merge into one DAG, run two reset branches concurrently, share STOP once and restart exactly once after both scheduling tails, with no escalation or provider action.
- Runner: scripts/e2e/regional/run_destr015_parallel_branch_join.py; scripts/e2e/regional/destr015_verdicts.py; scripts/e2e/regional/probes/destructive_node_probe.py
- Entry Points: preflight_errors / _inject_both / execute_case / _cleanup; injection_errors / spread_errors / workflow_errors / host_errors / workload_errors
- Assertions: Checks same workflow, kmsg RESET_GPU decisions, event-time spread, one shared STOP/join, each branch operation successful once and join dependencies on both tails.; Now requires complete aware ordered execution timestamps before accepting branch overlap or join-after-release.; Ledger reset successes, no reboot/full reset/residue, new Pod UIDs on same nodes, one budget consumption, CPU/provider/runtime checks.
- Safety And Cleanup: admission: Arbiter/DNS placement, correct capabilities, lifetime/window headroom, idle queues and plan identity.; deadline: Added prewarm/submission/setup checks; simultaneous injections already gate deadline. In-helper per-command bound belongs to parent supervision.; cleanup: Failed quiescence now defers workload deletion and isolation restore instead of continuing through a guard that only logged the error. Probe/prewarm cleanup remains independent; inconclusive CPU logs now fail.
- Necessity: Retain core positive multi-node promise. Missing timing evidence cannot mean parallel, and shared-step presence alone does not prove exactly-once execution.
- Overlap: case_ids: GF-REGIONAL-DESTR-014; GF-REGIONAL-DESTR-009; GF-REGIONAL-PREEMPT-017; classification: live_positive_complement; rationale: 014 negative exhausted join,009 workload restart, PREEMPT017 synthetic DAG;015 combines real two-node repair with one join.
- Tests: existing: tests/regional/test_destr015_parallel_branch_join.py::test_the_join_must_start_after_both_branches_released_their_nodes; current physical-proof successor: tests/regional/test_destr015_parallel_branch_join.py::test_dispatch_intervals_cannot_prove_physical_parallelism; added: tests/regional/test_destructive_evidence_review.py::test_parallel_proof_requires_every_execution_timestamp; tests/regional/test_destructive_evidence_review.py::test_parallel_join_requires_its_own_start_timestamp; tests/regional/test_destructive_runner_review.py::test_expired_case_does_not_start_mutation_but_still_cleans_up; parent_coverage: runner45%; verdict90%
- Findings At Review: id: D015-TIME; severity: P1; symbol: workflow_errors; finding: Absent timestamps skipped overlap and join checks.; resolution: Fixed with missing/invalid/naive timestamp counterexamples.; id: D015-CLEANUP; severity: P1; symbol: _cleanup; finding: Previously deleted workload after failed quiescence.; resolution: Now defers deletion and isolation restoration.; id: D015-LATE-SIBLING; severity: P2; symbol: central case contract; finding: Post-STOP sibling fault can form independent node workflow racing restart; no defined live case yet.
- Reviewer Handoff: state: fixed_owned_join_evidence; implemented: Missing/invalid/naive execution timestamps fail closed; cleanup requires quiescence before deleting the workload or restoring isolation; inconclusive CPU logs fail; deadline and HostProbe ownership propagated.; tests: tests/regional/test_destructive_evidence_review.py::test_parallel_proof_requires_every_execution_timestamp; tests/regional/test_destructive_evidence_review.py::test_parallel_join_requires_its_own_start_timestamp; remaining: Post-STOP late sibling is a distinct missing case, not coverage supplied by this two-pre-STOP injection case; see DESTR-PROP-001.; live_execution: NOT_RUN; prior_pass_reusable: False

### GF-REGIONAL-DESTR-016

同一节点第二个故障在物理步骤进行中到达：同级 XID 被吸收，更高级 XID 79 抢占 reset 工作流、接管 quiesce 并完成真实 reboot

Current catalog: destructive-provider-reboot; manual; NOT_RUN.

- Purpose: Phase13 after same-node002: park RESET_GPU at VERIFY_NO_GPU_CLIENTS with quiesce applied, absorb same-rank XID46, then stronger XID79 supersedes reset and hands quiesce to one real provider reboot. Cancel old command, never dispatch reset, restore on new boot and validate/release.
- Runner: scripts/e2e/regional/run_destr016_preempting_reboot.py; scripts/e2e/regional/destr016_verdicts.py; scripts/e2e/regional/probes/destr016_node_probe.py
- Entry Points: _arm_and_park / _absorb / _escalate / execute_case / _cleanup; waiting_boundary_errors / cancelled_command_errors / restore_after_reboot_errors; watch_ledger / arm_race_lost / target_bdf / holder_device
- Assertions: Same incident/workflow/unchanged step set for absorb; explicit stronger-rank successor links, inherited containment and graph-rewired quiesce handoff.; Old step stays WAITING while its command becomes FAILED/workflow-preempted; reset dispatch now checked in batched_steps too.; One role-attributed provider reboot, changed boot/same Node UID/artifact, handoff restore after reboot and no successful reset ledger rows.
- Safety And Cleanup: admission: Lifetime/step-window headroom and minimum maintenance window; holder device now must correspond to approved BDF in actual inventory.; deadline: Setup and arm window checks; holder obligation recorded before RPC. Scheduled XIDs still need absolute on-node expiry and exact drill binding, not only delay bounds.; cleanup: Holder disarm precedes quiet/validated restore; losing arming race now stops holder and returns without scheduling later XIDs. Early failure before incident capture remains a cleanup gap.
- Necessity: Important dirty-boundary preemption proof. It does not interrupt an executing GPU reset; it preempts the preceding waiting barrier.
- Overlap: case_ids: GF-REGIONAL-DESTR-002; GF-REGIONAL-DESTR-015; GF-REGIONAL-DESTR-017; classification: distinct_boundary_cases; rationale: 002 normal reboot,015 sibling branch,017 external boot-generation change;016 tests same-node stronger action and handoff.
- Tests: existing: tests/regional/test_destr016_preempting_reboot.py::test_destr016_workflow_contract_preempts_the_reset_and_reboots_once; tests/regional/test_destr016_node_probe.py::test_a_verification_that_succeeded_before_the_holder_loses_the_race; added: tests/regional/test_destructive_runner_review.py::test_lost_holder_race_never_schedules_escalation; tests/regional/test_destructive_runner_review.py::test_holder_is_bound_to_the_explicit_faulted_gpu; tests/regional/test_destructive_runner_review.py::test_expired_case_does_not_start_mutation_but_still_cleans_up; parent_coverage: runner46%; verdict93%; probe65%
- Findings At Review: id: D016-RACE; severity: P1; symbol: watch_ledger; finding: Originally scheduled escalation before checking lost race.; resolution: Check now precedes scheduling; failing race cannot schedule XIDs.; id: D016-DEVICE; severity: P1; symbol: target_bdf / holder_device; finding: Explicit target could differ from default held GPU or not exist.; resolution: Inventory and BDF/device binding enforced.; id: D016-DELAYED; severity: P1; symbol: on-node scheduled injections; finding: Relative delay and broad post-arm ledger match do not guarantee exact workflow/fence or absolute maintenance expiry; separate on-node admission proof still required.
- Reviewer Handoff: state: fixed_owned_barrier_authorization; implemented: Holder race loss stops before scheduling. Target BDF/device bound. No delayed XID timer exists until the controller authorizes the exact WAITING drill/workflow; firing rechecks local ledger command identity, incident/fence/generation, boot, quiesce scope, deadlines, script SHA and single-use intent. Failed absorb proof stops escalation.; tests: tests/regional/test_destr_barrier_authorization.py::test_controller_proof_uses_exact_batched_step_identity; tests/regional/test_destr_barrier_authorization.py::test_on_node_guard_reads_real_audit_rows_and_quiesce_state; tests/regional/test_destr_barrier_authorization.py::test_delayed_xid_consumes_intent_before_lost_ack_and_never_repeats; remaining: If kubelet quiesce prevents delivery of post-WAITING authorization, the case refuses; never revert to an unbound pre-armed timer. Live transport/timing NOT_RUN.; live_execution: NOT_RUN; prior_pass_reusable: False

### GF-REGIONAL-DESTR-017

带外 reboot 后的 generation fence：节点在 RESET_GPU 等待期间被系统外重启，旧代际的在飞命令被拒绝，新 boot 上不执行任何 reset

Current catalog: destructive; manual; NOT_RUN.

- Purpose: Phase13 after same-node002: OS reboot during quiesced RESET_GPU barrier changes Agent generation exactly N->N+1. Old maintenance commands and compensation fail closed, no reset/provider action, one support handoff and validated cleanup. Spec requires actual pinned maintenance window and old/new boot witnesses.
- Runner: scripts/e2e/regional/run_destr017_out_of_band_reboot_fence.py; scripts/e2e/regional/destr017_verdicts.py; scripts/e2e/regional/probes/destr017_node_probe.py
- Entry Points: _start_probes / _wait_for_maintenance_pin / _arm_out_of_band_reboot / execute_case; workflow_errors / command_errors / agent_errors / boot_errors / successor_errors; place_reboot_timer / fire_reboot / cancel_reboot
- Assertions: Checks reset workflow FAILED, no reset execution, a generation fence error, old verify not successful, new generation and one retired incarnation, independent boot observations and no provider mutations.; Negative command check now includes batched_steps.; Timer now invokes guarded fire_reboot rather than bare systemctl. Under a file lock it rechecks absolute deadline, same boot, same run, cancellation and single-use intent before issuing the one OS reboot.
- Safety And Cleanup: admission: Plan/UID/boot/generation/lifetime proof plus observed maintenance pin. Runtime code also accepts RECOVERED self-heal, whereas current spec/catalog describe QUARANTINED support path; parent must reconcile this drift.; deadline: Added setup/arm/injection checks, propagated absolute window to host, rejected timer placement beyond deadline and rechecked on actual fire.; cleanup: Reboot/holder obligations set before RPC, cancellation/disarm retained, validated restore and host final checks. Owned state clear-state promised in spec remains absent; early lost incident identification and precise pin-vs-fire timing need further proof.
- Necessity: Retain distinct generation-fence acceptance; normal provider reboot does not test an external boot while old commands remain in flight. Do not infer actual fence timing solely from later boot change.
- Overlap: case_ids: GF-REGIONAL-DESTR-016; GF-REGIONAL-DESTR-002; GF-REGIONAL-DESTR-019; classification: distinct_incarnation_transitions; rationale: 016 planned successor reboot,002 normal provider reboot,019 Agent-only same-generation restart;017 external OS reboot fences old generation.
- Tests: existing: tests/regional/test_destr017_out_of_band_reboot_fence.py::test_the_fenced_workflow_contract_accepts_the_expected_terminal_record; tests/regional/test_destr017_out_of_band_reboot_fence.py::test_the_agent_must_advance_exactly_one_generation; tests/regional/test_destr017_node_probe.py::test_the_reboot_is_armed_on_node_and_records_its_boot_id_first; added: tests/regional/test_destructive_delayed_actions.py::test_timer_fire_rechecks_deadline_boot_and_single_use; tests/regional/test_destructive_delayed_actions.py::test_fire_intent_is_persisted_before_the_only_reboot_request; tests/regional/test_destructive_delayed_actions.py::test_timer_cannot_be_placed_beyond_the_approved_window; parent_coverage: runner35%; verdict95%; probe60%
- Findings At Review: id: D017-EXPIRY; severity: P1; symbol: place_reboot_timer / fire_reboot; finding: Bare timer had no absolute expiry/cancelled-boot/single-use check.; resolution: Guarded fire entry added, with durable intent under lock.; id: D017-CONTRACT; severity: P2; symbol: ACCEPTED_AFTERMATH_STATES / successor_errors; finding: Code permits RECOVERED with lighter successor; central spec still requires QUARANTINED and support. No assertion was weakened in this review.; id: D017-PIN; severity: P2; symbol: watch_ledger / _arm_out_of_band_reboot; finding: Node timer is placed at quiesce, before controller observes WAITING. Exact case/fence/pinned-window attribution and cleanup of probe state still need stronger protocol evidence.
- Reviewer Handoff: state: fixed_owned_barrier_and_cleanup; implemented: Reboot timer is armed only after controller WAITING authorization, not by quiesce alone; exact local barrier and script identity rechecked at fire, absolute window and boot checked, intent consumed before reboot request. Cancellation is persisted under the same lock; clear-state refuses active/unknown units. HostProbe state is case-scoped.; tests: tests/regional/test_destr_barrier_authorization.py::test_controller_refuses_ambiguous_or_advanced_barrier; tests/regional/test_destructive_delayed_actions.py::test_timer_fire_rechecks_deadline_boot_and_single_use; tests/regional/test_destr_barrier_authorization.py::test_probe_state_cannot_be_cleared_while_a_timer_is_active; remaining: Post-WAITING transport may be unavailable and must refuse. Parent must reconcile existing RECOVERED-vs-QUARANTINED aftermath text; this review did not broaden accepted aftermath.; live_execution: NOT_RUN; prior_pass_reusable: False
- Current Disposition: Catalog/spec now describe both existing aftermath branches without changing the verdict code: the original fenced workflow must fail; RECOVERED additionally needs a successful non-preempting descendant with no forbidden action or foreign node.

### GF-REGIONAL-DESTR-018

工作流生命周期硬截止真机验收：RESET_GPU 持续 WAITING 到生命周期到期，整条 FAILED、在飞命令 workflow-timeout、只升级到人工，节点侧 ledger 在取消后不再新增 RESET 行

Current catalog: live-service-action; manual; NOT_RUN.

- Purpose: Phase13 after010: compress lifetime/execution to180s, hold GPU client, let VERIFY wait until lifetime failure, cancel remote command, run only one restore compensation and escalate to operator. Later XID79 must be record-only; no reset after cancellation, no reboot/replace, restore env and quarantine safely.
- Runner: scripts/e2e/regional/run_destr018_lifetime_deadline.py; scripts/e2e/regional/destr018_verdicts.py; scripts/e2e/regional/probes/destr018_node_probe.py
- Entry Points: _open_and_arm / _inject_and_observe / _escalation_and_absorb_errors / _cleanup; lifetime_margin_errors / cancellation_moment / remote_command_errors / data_plane_errors / escalation_errors
- Assertions: Requires stamped lifetime, correct failing VERIFY details, cancellation source and one compensation, no forbidden physical/escalation operations.; Ledger checks started_at vs cancellation, at most one straddling attempt and agreement with cancelled command/kernel. Compensation must now start at/after cancellation, not just finish afterwards.; Env identity permits only intended worker rollout/generation delta and requires exact baseline template restored.
- Safety And Cleanup: admission: Closed window, explicit timing relationships/cadence/attempt headroom, no active workflow and idle queues.; deadline: Added case setup and checks in both actual XID injections; failed control/data-plane assertions stop the later injected event. Window/holder obligations are recorded before RPC.; cleanup: Disarm holder, quiesce-aware validated restore/incident close, restore env, delete probes, verify runtime and node. Parent owns env-window mechanics; cleanup remains allowed after approval expiry.
- Necessity: Retain real cancellation/compensation proof. Compressed timing only proves classification at that timing; it is not default-production-duration evidence.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-032; GF-REGIONAL-DESTR-016; GF-REGIONAL-DESTR-017; classification: deadline_vs_preemption_vs_generation; rationale: Same barrier but different terminal cause: lifetime, stronger action, or changed incarnation. Synthetic032 cannot replace node ledger cancellation proof.
- Tests: existing: tests/regional/test_destr018_lifetime_deadline.py::test_the_lifetime_contract_passes_on_the_intended_terminal_state; tests/regional/test_destr018_lifetime_deadline.py::test_a_mutating_row_without_started_at_is_undecidable_not_a_pass; tests/regional/test_destr018_lifetime_deadline.py::test_a_support_workflow_that_inherited_an_expired_lifetime_fails_the_case; added: tests/regional/test_destructive_evidence_review.py::test_compensation_must_start_after_cancellation_not_merely_finish_after; tests/regional/test_destructive_runner_review.py::test_expired_case_does_not_start_mutation_but_still_cleans_up; parent_coverage: runner48%; verdict90%; probe62%
- Findings At Review: id: D018-COMPENSATION; severity: P1; symbol: compensation_row_errors; finding: Pre-cancel restore start could pass as post-cancel compensation.; resolution: Fixed.; id: D018-MARGIN; severity: P2; symbol: worst_case_verify_attempts; finding: 90s assumed containment reserve is not a proven lower bound; faster containment permits more attempts. Use zero/proven minimum for a genuinely conservative admission bound.; id: D018-IDENTITY; severity: P2; symbol: straddling_row_errors / metric_errors; finding: Assumes remote and Node Action command_id match, though real node IDs include deterministic suffixes; negative metric deltas are tolerated despite strict spec. Needs protocol fixture, not source text.
- Reviewer Handoff: state: fixed_owned_deadline_evidence; implemented: Compensation must start after cancellation; post-cancel injection is gated by renewed deadline and successful prerequisite evidence, including support/quarantine proof. Attempt bound uses zero unproved containment allowance. Node Action correlation uses exact per-node/generation idempotency identity, including batches. Decreasing/nonfinite metric evidence cannot pass.; tests: tests/regional/test_destructive_evidence_review.py::test_compensation_must_start_after_cancellation_not_merely_finish_after; tests/regional/test_destructive_evidence_review.py::test_node_ledger_correlation_uses_the_actual_node_action_id; tests/regional/test_destr018_lifetime_deadline.py::test_the_shipped_window_leaves_the_attempt_margin_it_promises; remaining: Real deadline cancellation and compensation timing NOT_RUN; env-window mechanics remain parent-owned.; live_execution: NOT_RUN; prior_pass_reusable: False

### GF-REGIONAL-DESTR-019

Node Agent 空闲重启后同代际回归、/healthz 可写与计数、账本就地迁移、下一条真实命令留下完整审计轨迹

Current catalog: live-service-action; manual; NOT_RUN.

- Purpose: Phase13 after010: restart idle Node Agent once behind fail-safe start, keep boot/incarnation/generation, verify healthy writable ledger/heartbeat/counters, migrate a scratch legacy ledger, then real solo XID45 leaves exactly one audited FM command across journal/ledger/health. Never restart Agent with a command in flight.
- Runner: scripts/e2e/regional/run_destr019_agent_restart_ledger.py; scripts/e2e/regional/destr019_verdicts.py; scripts/e2e/regional/probes/destr019_node_probe.py
- Entry Points: _baseline / _migration_drill / _quiet_control_plane / _restart / _audit_errors / execute_case; restart_errors / agent_record_errors / health_errors / journal_errors / ledger_row_errors / migration_errors
- Assertions: Same incarnation/generation/boot with changed PID/InvocationID and fresh heartbeat; health counters0 then accepted/completed1 and failed/rejected0.; Scratch0->current LEDGER_SCHEMA_VERSION migration, compound PK, readable old rows, preserved fencing and second attempt; scratch removed.; Two-step FM workflow, nine journal identifier fields/no sensitive fields, one successful ledger attempt with matching incident/workflow/fence and digests.
- Safety And Cleanup: admission: No open commands before restart; allowlisted FM; same release/Node plan fields. Full runtime identity is read but not entirely included in plan_identity.; deadline: Added setup and before-XID check after restart. Health/migration/restart errors now stop instead of proceeding to another live action.; cleanup: Restore obligation set before restart; ensure Agent active then disarm and ensure FM active; final node/runtime/probe checks. Scratch migration cleanup on exceptional exit remains incomplete.
- Necessity: Retain safe idle restart/audit. In-flight Agent interruption is a separate uncertain-outcome contract and must not be introduced into this case.
- Overlap: case_ids: GF-REGIONAL-DESTR-010; GF-REGIONAL-DESTR-017; classification: same_action_for_new_audit_claim; rationale: FM is a known safe command for observability;017 changes boot/generation while019 explicitly must not.
- Tests: existing: tests/regional/test_destr019_agent_restart_ledger.py::test_the_ledger_contract_passes_on_one_audited_attempt_row; tests/regional/test_destr019_node_probe.py::test_the_migration_drill_report_satisfies_the_case_verdict; added: tests/regional/test_destructive_runner_review.py::test_expired_case_does_not_start_mutation_but_still_cleans_up; parent_coverage: runner31%; verdict83%; probe59%; needed: Restart ACK loss and baseline/health refusal must prohibit XID but still recover/disarm.; Migration exceptions must remove only owned scratch files; fresh row generation/gpu_count must match actual command rather than merely type-check.
- Findings At Review: id: D019-FAILSTOP; severity: P1; symbol: execute_case; finding: Bad baseline/migration/restart accumulated errors but still injected FM event.; resolution: Now aborts before later mutation; deadline rechecked.; id: D019-SCRATCH; severity: P2; symbol: migration_drill_report; finding: Scratch unlink is after the try/finally that only closes ledger; constructor/read/append failure can leave scratch files.; id: D019-SCHEMA; severity: P2; symbol: catalog/spec; finding: Catalog still says ledger version2 while source/spec use3; parent central correction needed.
- Reviewer Handoff: state: fixed_owned_fail_stop_and_scratch_cleanup; implemented: Failed baseline health, migration or restart stops before later actions. Deadline renewed before restart and XID. Unknown command queue refuses restart. Scratch migration uses a private temporary directory and cannot overwrite an existing path; exceptions clean only owned scratch files.; tests: tests/regional/test_destructive_runner_review.py::test_agent_restart_case_stops_after_any_failed_prerequisite; tests/regional/test_destructive_runner_review.py::test_scratch_migration_failure_removes_only_the_owned_directory; tests/regional/test_destructive_runner_review.py::test_scratch_migration_does_not_overwrite_existing_data; remaining: Parent catalog must say Node Action ledger schema3, distinct from unchanged PostgreSQL schema17. Live Agent restart NOT_RUN.; live_execution: NOT_RUN
- Current Disposition: Catalog ledger version now follows the installed LEDGER_SCHEMA_VERSION (currently 3), independently of the PostgreSQL schema version.

### GF-REGIONAL-DESTR-020

隔离是观测出来的不是记住的：Kubernetes 不认识的别名节点让 MARK_UNSCHEDULABLE fail closed，任何节点都不被 cordon

Current catalog: live-non-destructive; manual; NOT_RUN.

- Purpose: Phase13 after001: API replay of unknown XID999 for an absent alias must create safety-only plan, fail MARK_UNSCHEDULABLE with observed absence, touch no real node and escalate exactly once to FREEZE_EVIDENCE->ESCALATE_SUPPORT. No physical operation or second-order escalation.
- Runner: scripts/e2e/regional/run_destr020_identity_mismatch_isolation.py; scripts/e2e/regional/destr020_verdicts.py
- Entry Points: alias_facts / kubernetes_node_present / read_only_preflight; _inject_and_observe / _escalation_chain / execute_case; decision_errors / workflow_errors / remote_command_errors / escalation_errors
- Assertions: Alias absent in Kube/fleet/GPU inventory; receipt200, SITE_SAFETY/QUARANTINE/action=None; failed isolation with absent/safety_rejection and exact node.; Support plan two control-only steps succeeds; bounded watch rejects second-order pair; real-node scheduling snapshot unchanged.; Kubernetes absence read now uses check=True with ignore-not-found, so auth/transport failure is not an absent node.
- Safety And Cleanup: admission: No physical target/Agent exists; unknown policy only safety steps. Direct build_plan now passes parsed arguments and real preflight bool to parent schema3.; deadline: Existing before-injection check retained; public expired-case behavioral test proves no action and cleanup runs. Long helper internals remain parent-supervised.; cleanup: Only fleet/runtime comparison and scoped leftover IDs; no Node mutation or direct DB deletion allowed.
- Necessity: Useful negative observation proof, but not a substitute for refusing a real node whose ownership/taint changed immediately before provider submit.
- Overlap: case_ids: GF-REGIONAL-DESTR-002; GF-REGIONAL-DESTR-021; classification: absent_identity_vs_live_metadata; rationale: 002 positive observed isolation,020 absent alias refusal,021 real-node metadata conflicts. Propose a separate safe mocked real-node ownership-drift case.
- Tests: existing: tests/regional/test_destr020_identity_mismatch_isolation.py::test_the_old_memory_based_isolation_is_a_failure; tests/regional/test_destr020_identity_mismatch_isolation.py::test_one_succeeded_support_handoff_without_isolation_is_the_contract; added: tests/regional/test_destructive_runner_review.py::test_alias_absence_requires_a_successful_kubernetes_read; tests/regional/test_destructive_runner_review.py::test_custom_plan_call_binds_parsed_arguments_and_real_preflight_verdict; parent_coverage: runner34%; verdict97%
- Findings At Review: id: D020-ABSENCE; severity: P1; symbol: kubernetes_node_present; finding: check=False converted empty failed read to absence.; resolution: Fixed; transport refusal is observable.; id: D020-BINDING; severity: P2; symbol: predecessor/plan_identity/fleet_snapshot; finding: Existing predecessor call omits release/cluster expectations; snapshot compares scheduling but not every real Node UID/boot. Parent identity integration still required.
- Reviewer Handoff: state: fixed_owned_absence_and_plan_binding; implemented: Failed Kubernetes reads cannot prove alias absence; custom plan passes parsed Namespace and actual preflight boolean to schema3 admission.; tests: tests/regional/test_destructive_runner_review.py::test_alias_absence_requires_a_successful_kubernetes_read; tests/regional/test_destructive_runner_review.py::test_custom_plan_call_binds_parsed_arguments_and_real_preflight_verdict; remaining: Parent/identity owner consolidates full predecessor/runtime/fleet binding.; live_execution: NOT_RUN; prior_pass_reusable: False

### GF-REGIONAL-DESTR-021

对抗性节点元数据下的 EFA 分层恢复：陈旧外来 plugin-restart 注解被接管、并发 409 下 RESTORE_SCHEDULING 仍收敛、隔离/恢复带前后快照

Current catalog: live-node-mutation; manual; NOT_RUN.

- Purpose: A real EFA driver-unbind drill races bounded harmless Node annotation writes against ownership takeover, isolation and recovery. Old plugin-restart annotations must not suppress remediation; the writer must actually stop and all owned metadata must be restored.
- Runner: scripts/e2e/regional/run_destr021_adversarial_node_metadata.py; scripts/e2e/regional/destr021_verdicts.py; scripts/e2e/regional/probes/destr021_annotation_writer.py; scripts/e2e/regional/warm_spare_fixture.py
- Entry Points: execute_case -> _preseed -> _inject_and_observe -> _judge -> _cleanup; AnnotationWriter.run_loop / stop / clear / report; writer_errors / takeover_errors / expected_count_errors / restore_errors
- Assertions: Requires workflow ownership takeover and the exact baseline EFA resource count, not a generic readiness flag.; Writer evidence requires successful patches, finite positive elapsed time, bounded cadence, running=false, stopped=true, cleared=true and no loop error.; NodeMutationFixture tracks the actual plugin annotation keys and refuses UID, resourceVersion, ownership or reservation drift.
- Safety And Cleanup: admission: Closed maintenance window and unknown command queue refuse injection; per-case HostProbe ownership propagated through CollectorAcceptanceFixture.; cleanup: Restore EFA first, stop/join writer, then clear tick and restore tracked annotations. A still-running writer forbids clear and defers annotation restore. Writer's default transport uses the public supervised regional command helper and is UID-bound.
- Necessity: Retain: resourceVersion contention and stale plugin ownership are different from an absent-node alias or ordinary EFA recovery. Concurrent writes do not guarantee a particular number of HTTP409 responses.
- Overlap: case_ids: GF-REGIONAL-DESTR-020; GF-REGIONAL-COLLECT-017; classification: distinct_metadata_race; rationale: 020 has no real target and COLLECT017 lacks the adversarial metadata writer.
- Tests: existing: tests/regional/test_destr021_adversarial_node_metadata.py; added: tests/regional/test_destructive_runner_review.py::test_writer_cannot_clear_while_a_patch_is_still_running; tests/regional/test_destructive_runner_review.py::test_writer_records_transport_failure_instead_of_silently_dying; tests/regional/test_destructive_runner_review.py::test_writer_calls_the_public_supervised_command_helper; tests/regional/test_destructive_evidence_review.py::test_writer_requires_a_stopped_observed_healthy_loop
- Findings At Review: id: D021-STOP; severity: P1; finding: stop event was previously accepted as proof that an in-flight writer thread had exited.; resolution: Actual thread/loop completion gates cleanup and verdict.; id: D021-ELAPSED; severity: P1; finding: Missing or nonfinite elapsed-time evidence previously skipped cadence judgment.; resolution: Unknown timing fails closed.
- Reviewer Handoff: state: fixed_owned_writer_and_cleanup; implemented: Observed writer completion, supervised transport, UID-bound writes, finite evidence and EFA-before-writer cleanup ordering.; remaining: Collector injection internals belong to the Collector owner; live EFA unbind NOT_RUN.; live_execution: NOT_RUN; prior_pass_reusable: False

### GF-REGIONAL-DESTR-022

部署 executor 的 spare 预留巡检按 TTL 回收陈旧预留，热备全程保持 cordon

Current catalog: live-node-mutation; manual; NOT_RUN.

- Purpose: Backdate one synthetic reservation on the declared cordoned spare. The installed executor must actually reclaim it, name exactly that node/incident in its log, and rewrite a fresh counter breadcrumb showing exactly one increment. No incident or workflow is created.
- Runner: scripts/e2e/regional/run_destr022_spare_reservation_reclaim.py; scripts/e2e/regional/destr022_verdicts.py; scripts/e2e/regional/probes/destr022_executor_probe.py; scripts/e2e/regional/warm_spare_fixture.py
- Entry Points: execute_case -> _inject -> _wait_reclaim -> _executor_evidence -> _cleanup; counter_errors / timeline_errors / log_errors / store_errors / final_errors; NodeMutationFixture.restore
- Assertions: Spare stays cordoned at every observation; reservation and reserved-at clear, pool state becomes AVAILABLE, and no foreign reservation is reclaimed.; Executor before/after identities are nonempty, unique and the same Pod UID set. Counters are real nonnegative integers; all judged records must be rewritten after reclaim and no later than their probe timestamp.; A readable exact reclaim log and total counter increment of one are required; zero fresh records cannot pass.
- Safety And Cleanup: admission: Synthetic incident absence, explicit idle queue, independent spare scope, same-release schema3 plan arguments.; cleanup: UID/resourceVersion CAS and pre-transport intent cover applied-but-lost ACK. Only the exact recorded reclaimed-reservation transition may be restored; a foreign reservation or quarantine owner is never overwritten.
- Necessity: Retain as the positive stale-reservation sweep test. DESTR008 reserved-by-other is a negative eligibility gate and cannot prove reclaim.
- Overlap: case_ids: GF-REGIONAL-DESTR-008; classification: positive_sweep_vs_negative_eligibility; rationale: Different trigger, clock boundary, product path and observable counter.
- Tests: existing: tests/regional/test_destr022_spare_reservation_reclaim.py; added: tests/regional/test_destructive_evidence_review.py::test_spare_reclaim_needs_at_least_one_fresh_counter_witness; tests/regional/test_destructive_evidence_review.py::test_reclaim_counters_require_the_same_fresh_executor; tests/regional/test_destructive_evidence_review.py::test_reclaim_counter_is_an_observed_nonnegative_integer; tests/regional/test_warm_spare_fixture_safety.py::test_a_proven_reclaimed_spare_can_return_to_its_declared_baseline
- Findings At Review: id: D022-FRESH; severity: P1; finding: A stale breadcrumb set with zero judged replicas previously passed.; resolution: Actual post-reclaim records and exact UID/counter identity now required.
- Reviewer Handoff: state: fixed_owned_fresh_record_evidence; implemented: Strict timestamp, Pod UID and counter evidence; checked log reads; CAS restoration and schema3 plan binding.; remaining: Real executor sweep/log propagation NOT_RUN.; live_execution: NOT_RUN; prior_pass_reusable: False

### GF-REGIONAL-DESTR-023

真正空闲集群的自主复位：无受管作业、观测流全部过期时，watcher 覆盖心跳让节点保持 IDLE，XID 46 走完八步 RESET_GPU 而不被 UNKNOWN 阻塞

Current catalog: destructive; manual; NOT_RUN.

- Purpose: Prove a genuinely idle cluster resolves IDLE from a fresh zero-work coverage heartbeat alone, with no managed Pods, no canary and no fresh attempt observations, then run the DESTR001 target-reset contract.
- Runner: scripts/e2e/regional/run_destr023_idle_cluster_reset.py; scripts/e2e/regional/destr023_verdicts.py; scripts/e2e/regional/run_destr001_gpu_reset.py; scripts/e2e/regional/probes/destructive_node_probe.py
- Entry Points: coverage_probe / watcher_deployment / wait_for_coverage / execute_case; coverage_supported_errors / fresh_coverage_errors / watcher_errors / reset_errors
- Assertions: Explicit nonnegative integer heartbeat and observation counts; required aware timestamps; finite nonnegative ages exactly consistent with their timestamps; future, missing and contradictory evidence fails.; Watcher Deployment UID/generation and observedGeneration are explicit, all replica counts are exact, and live Pod inventory uses strict Ready conditions and complete container readiness.; IDLE must not be explainable by a fresh attempt observation; post-reset target UUID and ledger identity use the strengthened 001 proof.
- Safety And Cleanup: admission: Recheck deadline before probe creation and XID; wrong-cluster heartbeat refuses.; cleanup: Sampler obligation is recorded before start, with case-scoped persistent HostProbe ownership; normal reset cleanup still uses validation-first restoration.
- Necessity: Retain: normal reset cases can be accidentally made IDLE by a leftover workload observation. This case isolates the heartbeat-only premise.
- Overlap: case_ids: GF-REGIONAL-DESTR-001; GF-REGIONAL-DESTR-024; classification: shared_reset_with_distinct_coverage_premise; rationale: 001 proves basic reset;024 is the negative UNKNOWN boundary.
- Tests: existing: tests/regional/test_destr023_idle_cluster_reset.py; added: tests/regional/test_destructive_evidence_review.py::test_idle_coverage_requires_explicit_heartbeat_fields; tests/regional/test_destructive_evidence_review.py::test_idle_coverage_rejects_unusable_heartbeat_age; tests/regional/test_destructive_evidence_review.py::test_coverage_cannot_infer_missing_inventory_fields; tests/regional/test_destructive_evidence_review.py::test_coverage_age_must_match_the_observed_timestamp; tests/regional/test_destructive_evidence_review.py::test_ready_count_alone_does_not_prove_watcher_recovery
- Findings At Review: id: D023-UNKNOWN; severity: P1; finding: Missing counts defaulted to zero and NaN/future ages could pass.; resolution: Explicit consistent records required.; id: D023-READY; severity: P1; finding: Desired plus ready replica count alone could hide an incomplete rollout.; resolution: Exact Deployment and strict Pod population proof required.
- Reviewer Handoff: state: fixed_owned_idle_and_readiness_evidence; implemented: Strict heartbeat-only IDLE evidence, complete readiness and target reset identity, deadline and HostProbe ownership propagation.; remaining: Live heartbeat attribution and physical reset NOT_RUN.; live_execution: NOT_RUN; prior_pass_reusable: False

### GF-REGIONAL-DESTR-024

watcher 停机时 fail-closed：覆盖心跳过期后节点为 UNKNOWN，XID 46 只做隔离并 BLOCKED，不 quiesce 不 reset；watcher 恢复后心跳回来，隔离经 validated restore 解除

Current catalog: live-isolation; manual; NOT_RUN.

- Purpose: Stop the completion watcher behind an independently armed restore watchdog; after heartbeat/observations expire, one synthetic XID46 must be safety-contained but BLOCKED by UNKNOWN workload state with no physical operation. Restore the watcher and prove a genuinely new heartbeat before validation-first isolation cleanup.
- Runner: scripts/e2e/regional/run_destr024_watcher_down_fail_closed.py; scripts/e2e/regional/destr024_verdicts.py; scripts/e2e/regional/destr023_verdicts.py
- Entry Points: WatcherScaleFixture.arm / scale_down / restore; execute_case / validated_restore; watcher_absent_errors / blocked_workflow_errors / host_untouched_errors / heartbeat_recovered_errors / restore_errors
- Assertions: Watcher absence needs explicit zero replicas and no Pods, never an empty unknown response.; Any physical step execution, including FAILED, is disallowed. A new attempt under an existing ledger command cannot be hidden; GPU inventory identity stays unchanged.; Actual cordon, quarantine NoSchedule taint and incident ownership must all exist. Recovery needs a fresh advanced heartbeat after restore began, exact successful validation-first executions, unchanged final Node UID and no residual isolation.
- Safety And Cleanup: admission: Dead watchdog refuses scale-down; deadlines rechecked before watcher mutation and XID; strict coverage expiry proof.; cleanup: Watchdog disarms only after complete observed readiness. Failed or ambiguous stop cannot be recorded as restored. Final isolation always fails even if an earlier restore succeeded.
- Necessity: Retain as the required fail-closed half of023. Restoring desired replicas is not recovery evidence and an old fresh-looking heartbeat is insufficient.
- Overlap: case_ids: GF-REGIONAL-DESTR-023; classification: paired_positive_negative_boundary; rationale: 023 requires IDLE and reset;024 requires UNKNOWN and zero physical action, then observes recovery.
- Tests: existing: tests/regional/test_destr024_watcher_down_fail_closed.py; added: tests/regional/test_destructive_runner_review.py::test_dead_watchdog_cannot_authorize_watcher_scale_down; tests/regional/test_destructive_evidence_review.py::test_watcher_down_rejects_a_new_attempt_under_an_old_physical_command; tests/regional/test_destructive_evidence_review.py::test_a_heartbeat_from_before_restore_is_not_recovery; tests/regional/test_destructive_evidence_review.py::test_ready_count_alone_does_not_prove_watcher_recovery
- Findings At Review: id: D024-RESTORE; severity: P1; finding: Failed watchdog stop and post-restore re-isolation could still pass.; resolution: Stop uncertainty and every final isolation fail.; id: D024-PHYSICAL; severity: P1; finding: FAILED physical executions and repeated attempts under an old command ID were ignored.; resolution: All physical executions and new attempts are rejected.
- Reviewer Handoff: state: fixed_owned_fail_closed_and_recovery_evidence; implemented: Strict absence/readiness, no-physical-action evidence, actual fresh heartbeat and validation-first restoration, final UID/isolation checks and truthful watchdog-stop result.; remaining: Live watchdog descendant/host-loss recovery and Kubernetes availability NOT_RUN; parent must retain any broader supervision limitations in consolidated acceptance notes.; live_execution: NOT_RUN; prior_pass_reusable: False

### GF-REGIONAL-E2E-001

单集群 Collector 到控制面到本地 executor 全链路

Current catalog: live-workload-restart; manual; NOT_RUN.

- Purpose: Per physical cluster, drive real user-space /dev/kmsg XID11 through Collector, policy, incident/workflow, local executor restart, notifications, CPU blast checks, and verified cleanup with changed attempt identity.
- Runner: scripts/e2e/regional/run_workload_acceptance.py; scripts/e2e/regional/probes/e2e001_node_probe.py; scripts/e2e/regional/run_destr009_workload_restart.py; scripts/e2e/regional/managed_workload_fixture.py
- Entry Points: run_workload_acceptance.e2e_preflight; run_workload_acceptance.run_e2e001; run_workload_acceptance.notification_errors; run_workload_acceptance.gpu_nodes_clean; e2e001_node_probe.write_xid11
- Assertions: Preflight checks collector kinds but not freshness, and active-agent count rather than an exact per-node identity set; absent executor Pods or failed log reads can appear clean.; Checks real kmsg bytes, kmsg:// evidence_ref, nonempty source_boot_id/source_monotonic_us, RESTART_APP workflow, changed Pod UIDs/single new attempt, two SENT notification categories, CPU comparison, and GPU cleanliness.; The event boot ID is not compared with host.snapshot boot_id, and no raw evidence record_id/kernel-sequence or exact attempt/workload linkage is read.; Reused workflow_errors proves only matching operation categories, not an exact remote/<command-id> to command identity bijection; forbidden operations are checked only in official_steps.; gpu_nodes_clean accepts an empty node list and a newly disappeared node is not detected by its all() check.
- Safety And Cleanup: A host probe performs a user-space log injection, not hardware damage; this distinction is correctly recorded.; Cleanup defers workload deletion when the source workflow cannot be proven quiescent, then checks GPU readiness/taints.; Shared workload deletion can silently fail; unchanged node checks cannot prove the running training resources disappeared or observations terminalized.; Target node is chosen only after scheduling, is not bound to the approved plan, and the passed maintenance deadline is used in evidence rather than rechecked before injection.; The BLAST handoff files lack release/digest binding and may be written before cleanup later changes the case verdict.
- Necessity: Keep as the per-cluster admission gate. Unlike API replay, it exercises the installed kernel collector, cursor, and source metadata path.
- Overlap: case_ids: GF-REGIONAL-DESTR-009; GF-REGIONAL-ISO-001; GF-REGIONAL-BLAST-001; classification: shared-recovery-helper-distinct-injection; rationale: DESTR-009/ISO-001 replay through HTTP and bypass the collector's real kmsg path. BLAST-001 consumes a different evidence surface and cannot replace this complete chain.
- Tests: existing: tests/regional/test_workload_acceptance_runner.py (historical pre-fix symbol: test_e2e001_source_checks_attempt_id_scope_and_nodes_after_cleanup); tests/regional/test_workload_acceptance_runner.py::test_gpu_nodes_clean_tolerates_only_the_cordoned_spare; tests/regional/test_workload_acceptance_runner.py::test_control_plane_blast_errors_flag_new_evictions_but_not_aged_out_ones; missing_behavioral: Drive run_e2e001 with fake host/collector/store/notification/workload/cleanup boundaries and inspect the written handoff.; Stale/missing collector, wrong active-agent node set, absent/unreadable executors, expired maintenance window, or node UID drift must prevent injection.; Wrong boot/sequence/attempt/workload/command/notification identity must fail even with successful statuses.; Empty/disappeared GPU node sets, ignored delete failure, new target Observation not terminal, or stale handoff must fail.
- Findings At Review: E2E-specific high-risk control flow is currently covered mostly by source assertions.; The twelve claimed layers lack complete source/evidence identity, running executor proof, and post-delete resource verification.; No Fabric Manager service check on every GPU node or sustained successful claim sampling.; Parent/DESTR-family must repair shared observation freshness, command correlation, and owned cleanup; parent owns the BLAST handoff contract.
- Reviewer Handoff: FIXED: nonempty node/executor/Agent preflight; fresh Pod-bound Observation; pre-injection node UID/deadline check; event/decision/incident/workflow/remote-step/budget and raw kmsg record/node/boot/sequence/GPU linkage. Notifications require unique IDs, incident binding and both SENT categories. Cleanup is ownership/quiescence fenced and node identities are rechecked; BLAST handoff is written after cleanup with final verdict and file digests. Case-private HostProbe state is supplied. This remains user-space kmsg software-chain evidence, never real hardware damage.

### GF-REGIONAL-E2E-002

两个集群并发同类故障互不干扰

Current catalog: live-workload-restart; manual; BLOCKED.

- Purpose: Run identical fresh job/attempt IDs on two physical clusters, inject both XID11 faults within 60 seconds, and prove independent scoped incidents/workflows/commands/budgets/notifications, failure isolation, stable CPU replicas, and complete cleanup.
- Runner: scripts/e2e/regional/run_e2e002_multicluster_fault.py; scripts/e2e/regional/multi_cluster_fixture.py; scripts/e2e/regional/run_destr009_workload_restart.py; scripts/e2e/regional/managed_workload_fixture.py
- Entry Points: run_e2e002_multicluster_fault.read_only_preflight; run_e2e002_multicluster_fault.execute_case; run_e2e002_multicluster_fault.command_scope_errors; run_e2e002_multicluster_fault.notification_errors
- Assertions: Runs preparation/injection/settlement concurrently; checks different incident/workflow IDs, shared workflow success contract, both budgets=1, scoped commands/notifications, and CPU container differences.; The 60-second measurement covers metadata retrieval and completed receipt polling, not just the two injection start times.; command_scope_errors accepts any owner when the known-identity set is empty, and does not establish exact workflow/incident/step bindings.; notification_errors does not require SENT, MessageId, exact count/category, or nonempty IDs; budget checks only read restart_count.; Both branches must succeed; no failed-A/healthy-B variant tests the explicit failure-isolation expectation.
- Safety And Cleanup: Finally deletes both workloads immediately even if one workflow is still active; it does not reuse cleanup_workload/wait_for_cleanup_quiescence.; Shared delete suppresses errors and has no ownership/UID or absence proof; no final Pod/Observation/taint/cordon/command verification is performed.; No fresh empty control-record baseline is checked before submission, so reused explicit IDs can consume historical evidence.; Plan/run deadline and node/runtime release checks are only initial; plan node UIDs are not compared before mutation.
- Necessity: Keep concurrent symmetric recovery separate from the asymmetric isolation control; add the missing failed-branch variant without silently calling two successful restarts failure isolation.
- Overlap: case_ids: GF-REGIONAL-ISO-001; GF-REGIONAL-ISO-006; GF-REGIONAL-E2E-001; classification: complementary-not-retirable; rationale: ISO-001 leaves B untouched, ISO-006 removes its network, and E2E-001 tests one collector chain. Simultaneous same-ID recovery tests different shared-state contention and cannot retire any of them.
- Tests: existing: tests/regional/test_identity_multicluster_contracts.py::test_e2e002_notification_and_command_scope_judgments; tests/regional/test_identity_multicluster_contracts.py (historical pre-fix symbol: test_e2e002_follows_iso001_and_runs_the_clusters_in_parallel); tests/regional/test_collector_and_multicluster_fixtures.py::test_multi_cluster_runners_are_plan_only; missing_behavioral: Full fake execute_case with one branch failing before/after injection, and prove the other is observed to terminal plus both cleanups are safe.; Missing known executor, wrong workflow/incident/step/budget cluster, unsent/duplicate/empty notification, and reused identity must fail.; Actual injection timestamps, long receipt processing, and deadline/node-UID drift need fake-clock coverage.; A failed delete or remaining workload/command/Observation must keep FAIL and auditable deferred cleanup.
- Findings At Review: Order and code require ISO-001; specification prose still says ISO-006 and both per-cluster E2E-001 records. Parent must reconcile them.; Predecessor checks release only, not the same A/B cluster pair, and final evidence has only A as its top-level cluster identity.; The promised notification delivery and failed-branch independence are not tested.; Cleanup can delete a workload while its executor is still acting and then report success despite suppressed delete errors.
- Reviewer Handoff: FIXED: fresh same-ID baselines, complete executor identities, source Observation binding, actual injection-start timing and per-cluster causal command/budget/notification checks. A lost injection ACK still observes both started branches; branch failures retain sibling outcomes and complete quiescence-aware owned cleanup. Full fake lifecycle tests cover ACK loss, one failed branch, dirty baseline and wrong causal proof. A deliberately failed-A/healthy-B live experiment is a separate parent proposal, not asserted from two successful recoveries.

### GF-REGIONAL-HA-001

删除 active-active 控制面副本后 API 与 dispatcher 继续工作

Current catalog: live-service-action; manual; NOT_RUN.

- Purpose: contract: Delete an ingress replica and two original active-consumer workers under claim and host-telemetry load; bound failures, retain ingress quorum, drain queue and complete one XID11 recovery workflow without duplicate steps.; catalog: manual, live-service-action; YAML still describes all replicas as active consumers, unlike ingress/worker role split.; order: Phase 12 first entry after NET-008; implementation has no predecessor check.
- Runner: scripts/e2e/regional/run_ha001_control_plane_failover.py; scripts/e2e/regional/probes/ha001_probe.py; scripts/e2e/regional/ha_store_probe.py
- Entry Points: main -> custom build_plan(schema3) -> shared authorize_execution(schema2) -> execute_case; execute_case -> create_probe -> role_snapshot -> observe_phase -> delete_and_observe x2 -> seed_closure/wait_closure -> delete third target -> cleanup_steps; Probe uses RegionalExecutorClient/ClusterActionExecutor plus SimulatedAdapter; seeded workflow is deliberately BLOCKED.
- Assertions: Derived replica floors, preStop/graceful-shutdown plus assumed 30s NLB budget, endpoint/queue samples and replacement settle window.; Roles before/after, claim/health success cycle counts, no 401/403/500 and synthetic remote-command success counts/ledger operation sequence.; No host telemetry is injected. The XID11 collector/processor/dispatcher/workload path and workflow step_executions are not exercised.; role validation demands leadership=null, but current healthz emits processor_mode/processor_role and no leadership field.
- Safety And Cleanup: Custom owner isolates synthetic commands from production adapters; no real workload restart or GPU action occurs.; Target UID is re-read before name-only deletion, but no atomic delete precondition exists.; Probe Pod/ConfigMap use fixed names, pre-delete/apply and unconditional final deletion without run ownership.; cleanup_steps continues after exceptions, but legacy check=False reads previously interpreted API failure as absence; probe resource reads now propagate failure.; SQL deletion now retains existing Store-v17 control-state routines and resolves projected credentials through ha_store_probe.
- Necessity: Meaningful CPU replica/claim continuity smoke test, not the full specified recovery acceptance. Fix entry/health contracts and add real processor load before using its result as HA-010 prerequisite.
- Overlap: case_ids: GF-REGIONAL-HA-002; GF-REGIONAL-HA-005; GF-REGIONAL-E2E-001; classification: partial shared sampling; no superset; rationale: 002 injects CPU cordon/PDB eviction; 005 injects Deployment rollout with event outbox traffic; E2E-001 covers workload recovery without these CPU failure timings. None duplicates assertion, injection and scope together.
- Tests: existing: tests/regional/test_ha001_control_plane_failover.py: limits, role checks, settle/cap, synthetic ledger verdict and per-step cleanup; missing_behavioral: Real main plan-to-authorize round trip, current healthz shape, below-quorum baseline rejection.; Failed get/zero observations, stale probe counters, sustained fault queue growth and mutation UID race.; Nonzero processor workload through deletion, exact step executions and failure-safe seeded-row ownership cleanup.; execution: Owned health, schema3, nonce/UID, component Python and cleanup tests passed within final269. No live Pod deletion or real workflow recovery.
- Findings At Review: id: H001-1; severity: P1; description: Custom schema3 plan has no shared details/digest envelope, so normal execute is rejected by schema2 authorization.; code: build_plan/main; owner: owned runner; parent shared contract coordination; id: H001-2; severity: P1; description: Current healthz lacks leadership, so validate_roles rejects healthy role-split processes.; code: validate_roles; src/gpu_fault/app/routes/admin.py::healthz; owner: owned runner; id: H001-3; severity: P1; description: No host telemetry, XID11 or completed workflow; isolated BLOCKED workflow plus simulated commands cannot prove worker queue takeover or real recovery closure.; code: seed_closure; probes/ha001_probe.py::main; owner: owned runner/probe; parent spec/catalog; id: H001-4; severity: P1; description: No formal predecessor/result release identity, atomic deletion UID or run ownership for fixed probe resources; deadline is not enforced per deletion.; code: main/delete_and_observe/create_probe/cleanup_steps; owner: owned caller; parent shared mutation contract
- Reviewer Handoff: Owned schema3 round trip, active-active health shape, strict ReadyPods, projected credential SQL, release/cluster predecessor, per-deletion deadline, UID DeleteOptions and nonce/UID probe ownership cleanup implemented. Supervision loss records FAIL without further remote commands. Busy queue/workflow survival and real XID closure are not implied by its synthetic command drill.

### GF-REGIONAL-HA-002

CPU节点维护时三层角色的PDB与拓扑约束生效

Current catalog: live-service-action; manual; NOT_RUN.

- Purpose: contract: Cordon one CPU node; exercise ingress, worker and enabled spool PDB eviction/rejection independently while traffic remains healthy; restore scheduling and declared topology.; catalog: manual, live-service-action.; order: Phase 12 after HA-001; runner does not read that predecessor.
- Runner: scripts/e2e/regional/run_ha002_pdb_topology.py; scripts/e2e/regional/run_ha001_control_plane_failover.py
- Entry Points: main -> custom schema3 build_plan -> shared schema2 authorization -> execute; _run_disruptions -> shared HA001 probe -> detached uncordon watchdog -> cordon -> _evict_role for each enabled role -> uncordon -> recovery; _evict_role -> real Eviction -> wait/re-read PDB closed -> second real Eviction -> require rejection
- Assertions: PDB re-read rejects known reopened budget; second refusal must mention disruption budget.; Availability floors, short observation windows, restored replicas, queue <=5 and scheduling/taint equality.; balanced_distribution only includes occupied nodes; all replicas on one node can be called balanced. Spool distribution is recorded but not validated.; No notification lease or telemetry-spool record conservation assertions and no injected processor load.
- Safety And Cleanup: Selected role Pods only, but plan stores target Pod names without their UIDs; Eviction has no UID preconditions.; Node UID is checked before execution, not atomically with cordon/uncordon or by the detached watchdog.; Cleanup previously stopped watchdog even when uncordon failed. It now retains watchdog and refuses a detected replacement Node; raw watchdog still lacks UID binding.; pod_by_name now propagates API failures and treats only successful ignore-not-found output as absence.; PDB can reopen after final GET and before the second real Eviction; unexpected success can still remove another Pod.
- Necessity: Useful PDB/topology scenario, but must not lose its restoration fallback or infer balanced topology from occupied nodes alone. Current schema/role mismatch blocks ordinary execution.
- Overlap: case_ids: GF-REGIONAL-HA-001; GF-REGIONAL-HA-005; classification: complementary maintenance failure mode; rationale: Node cordon and PDB admission are absent from Pod deletion and rollout cases. Shared claim probe does not make the injection or topology assertions redundant.
- Tests: existing: tests/regional/test_ha002_pdb_topology.py: PDB recheck, occupied-node distribution, capacity recommendation and cleanup; missing_behavioral: Plan/execute with current shared guard; target Pod/Node UID changes and concurrent cordon ownership.; Eligible node with zero Pods, one-node pileup, spool imbalance and actual topology key/selector.; API refusal/timeouts vs absent resources; failed uncordon retains a UID-safe watchdog; unexpected second Eviction success stops all later mutation.; Spool and notification lease continuity with nonempty fixtures.; execution: Owned schema3, strict ReadyPods, topology, UID/owner restoration and dry-run Eviction tests passed within final269. No cordon or live Eviction.
- Findings At Review: id: H002-1; severity: P1; description: Same schema3/schema2 plan and obsolete health leadership mismatch as HA-001.; code: build_plan/main; COMMON.validate_roles; owner: owned runners; id: H002-2; severity: P1; description: Watchdog has no UID/ownership checks and second Eviction can succeed after a PDB race. Restore previously disarmed even after failure.; code: start_uncordon_watchdog/_evict_role/cleanup_case; owner: owned runner; shared atomic mutation support parent; id: H002-3; severity: P2; description: Topology excludes zero-count eligible nodes and never validates spool spread; empty traffic gives no queue/spool/notification survival proof.; code: balanced_distribution/_ha002_result; owner: owned runner/probes; id: H002-4; severity: P2; description: No formal predecessor identity or ongoing maintenance-window enforcement.; code: main/_run_disruptions; owner: owned caller
- Reviewer Handoff: Owned schema3/health/predecessor fixes plus zero-count eligible-node topology implemented. Node cordon and delayed restore use UID/owner CAS, failed restore keeps watchdog armed, Pod eviction has UID preconditions, and second rejection probe uses server dry-run so a reopened PDB cannot remove another Pod. No unowned probe deletion on refusal.

### GF-REGIONAL-HA-003

Aurora 故障转移期间不重复执行破坏性 step

Current catalog: destructive; manual; NOT_RUN.

- Purpose: contract: Real GPU reset on a pre-cordoned idle node overlaps an Aurora writer failover; one physical action/terminal command/successful step; no permanent BLOCKED workflow or dropped processor requests.; catalog: manual, destructive, staging-level; related pytest checks reset claim capture.; order: Phase 15 first entry with explicit DESTR-008 predecessor.
- Runner: scripts/e2e/regional/run_ha003_aurora_failover_reset.py; scripts/e2e/regional/run_destr001_gpu_reset.py (other owner, consumed only); scripts/e2e/regional/host_probe_fixture.py (shared)
- Entry Points: run_standard_case -> read_only_preflight/focused tests -> verify_plan_identity -> execute_case; Host probe write-xid46 -> wait_reset_claim(LEASED or WAITING) -> aws_rds failover -> wait_rds_failover with Store samples -> wait_for_workflow -> evaluate_reset; Production: Node Agent command-id in-flight/ledger, executor lease/result, Store fencing and transient processor replay; CPU does not submit GPU mutation.
- Assertions: Predecessor release/cluster binding, node Ready/cordoned/idle, agent ACTIVE, profile gpuReset OWN and quiet fault/command queues.; Shared reset workflow/host checks, one RESET_GPU success and command ID, final node baseline, RDS writer transition and overlap samples.; verdict_for prioritizes FAIL over INCONCLUSIVE; no observed overlap cannot PASS.; Shared host evidence currently uses one ledger row plus inventory dip/recovery and at most one journal reset line, not the specified exact physical reset count.; Result log parser counts only recognizable rejected HTTP codes from surviving Ready Executors; failed/empty log reads previously looked like no failures.
- Safety And Cleanup: Requires explicit destructive confirmation/window; injection is synthetic kmsg XID driving a real authorized reset, not hardware damage.; Quiesce restoration and validated node restore are attempted only when incident_id is captured. It was set only after workflow wait, losing the target on expected Store outage.; Incident capture now happens at claim return, but failures before that return and during timeline capture need incremental recovery identity.; Host probe removal precedes ownership restore; final UID/quiesce/terminality and provider-negative visibility need stronger proof.; RDS cluster endpoint is read but not proven to be the CPU Store's database; arbitrary supplied cluster ID could fail over an unrelated database.
- Necessity: Distinct, high-value destructive concurrency boundary. Failover must be real and overlap must be observed, but expected Store transport loss should be recorded and retried without discarding cleanup identity or being misclassified as evidence.
- Overlap: case_ids: GF-REGIONAL-HA-004; GF-REGIONAL-HA-010; GF-REGIONAL-DESTR-001; classification: complementary failure injection; rationale: DESTR-001 supplies reset assertions without DB failover; 004 injects Executor owner loss; 010 injects DB failover without GPU commands and checks process health. No superset.
- Tests: existing: tests/regional/test_ha_reset_window_fixtures.py: LEASED/WAITING capture, terminal-too-late refusal, WAITING accumulation, overlap verdict and RDS/sample pairing; Focused runtime tests listed by focused_tests cover transient Store failure, replay retry and Node Agent reset idempotency; missing_behavioral: Store/read transport interruption during every wait, with retained incident identity, bounded recovery and no false overlap from unknown RDS status.; Failed/missing Executor logs vs observed zero rejected results; command-scoped parsing and retry paths.; Cleanup after injection, claim, failover, host-sampler or workflow failures; changed Node UID and in-flight physical action refusal.; Isolated PostgreSQL failover/fencing tests require parent setup; live physical evidence remains unexecuted.; execution: Owned polling and early-failure/fail-stop orchestration tests passed within final269; required private HostProbe directory checked by fakes. No GPU/RDS action.
- Findings At Review: id: H003-1; severity: P1; description: Expected Store outage aborts observe() without bounded retry; early exceptions lose cleanup incident identity.; code: wait_rds_failover/execute_case/cleanup_case; owner: owned runner; parent transport classification; id: H003-2; severity: P1; description: No binding of RDS endpoint/resource to actual CPU Store, and mutation after long capture may outlive approval.; code: rds_snapshot/read_only_preflight/execute_case; owner: owned caller; parent site identity helper; id: H003-3; severity: P1; description: Inventory min/count and journal<=1 cannot prove exactly one target-GPU physical reset; missing sampler transport could resemble an inventory dip.; code: run_destr001_gpu_reset.py::host_errors and its probe; owner: DESTR helper owner; id: H003-4; severity: P2; description: Focused-test reuse omits tests/execution, tests/processor and tests/node_agent; production log/queue semantics need separate truthful coverage.; code: focused_tests; live_driver_guard.source_digest; owner: parent shared digest
- Reviewer Handoff: Owned polling distinguishes unavailable Store observations from reset evidence and waits for a successful recovered read; incident identity retained incrementally before later failure. Required private HostProbe state_directory passed. Cleanup/FAIL evidence survives early errors; ProcessSupervisionLost stops remote cleanup. Store-to-RDS identity and target-specific physical reset proof remain parent/DESTR-owner integration, not verified LIVE.
- Current Disposition: Shared read-only Aurora binding now connects CPU EKS/release/Secret/projected DSN/SQL writer to the selected RDS incarnation. It is rechecked before reset and failover; reset must remain open after revalidation.

### GF-REGIONAL-HA-004

双 executor 副本对 WAITING command 重领时不产生重复物理动作

Current catalog: destructive; manual; NOT_RUN.

- Purpose: contract: Two real Executors reclaim one in-flight WAITING/expired reset command after owner termination; remote and deterministic Node command IDs stay fixed and physical reset happens once; restore lease/poll/replicas.; catalog: manual, destructive; effective Store polling about 6s is explicitly acknowledged.; order: Phase 15 immediately after HA-003.
- Runner: scripts/e2e/regional/run_ha004_waiting_reclaim_reset.py; scripts/e2e/regional/executor_env_window.py (shared); scripts/e2e/regional/run_destr001_gpu_reset.py (other owner)
- Entry Points: run_standard_case -> preflight/identity -> ExecutorTimingFixture.apply -> host write-xid46 -> command_timeline kill callback -> wait_for_workflow -> evaluate_reclaim -> cleanup_case; Timing env delegates to executor_env_window; local runner owns replica scale and delayed rollback.
- Assertions: Two Ready Executors, pinned node/agent/profile/deployment/env and predecessor release/cluster identity.; One timeline command ID, at least two owners and either two token digests or a changed owner; shared reset workflow/host checks and deterministic Node ledger ID.; No proof that second lease occurred only after kill, no affirmative normal-renewal observation, and final command ID is not compared to the timeline's unique ID.; No explicit every-step success-count check; final Node generation is used to construct Node command ID rather than proving baseline generation stability.
- Safety And Cleanup: Watchdog duration is derived to 3600s; env window restores original absent/literal values and exceptions preserve the fallback.; Restore disarms watchdog before checking restored UID/env/replica evidence; raw delayed commands act by Deployment name and can overwrite a replacement.; Owner mapping was substring-based; it now matches the exact final owner path component and rejects ambiguity.; Incident was captured after workflow wait; claim/timeline result now captures it sooner, but incremental observation is still needed on timeline failure.; Forced Pod deletion has no UID precondition; node quiesce/ownership restore needs final baseline identity and terminality checks.
- Necessity: Do not merge with HA-006: this is the actual Node Agent idempotency boundary. Retain coarse-sampling limitations and verify second-owner timing and safe restoration explicitly.
- Overlap: case_ids: GF-REGIONAL-HA-003; GF-REGIONAL-HA-006; GF-REGIONAL-CMD-006; classification: partial lease overlap, distinct physical boundary; rationale: CMD-006 rejects stale results without proving physical idempotence; 006 uses a synthetic notification-dedup ledger; 003 loses database availability instead of the Executor. No superset.
- Tests: existing: tests/regional/test_ha_reset_window_fixtures.py: stable timeline IDs, WAITING owner kill, token hashing, owner-change lease inference, watchdog budgets and process-group stop; missing_behavioral: Wrong final command, generation drift, second owner observed before kill, ambiguous owner name and no normal-renewal proof.; Failed/partial env restore and UID replacement must retain a safe watchdog; exact restored live replicas/values before disarm.; Injection/timeline exceptions retain recovery incident and audit failed cleanup.; True concurrency and fencing with new isolated PostgreSQL, plus separately approved live Node Agent evidence.; execution: Owned actual schema2 window integration, stdin CAS, post-kill identity and HostProbe failure cleanup tests passed within final269. No Executor rollout or GPU action.
- Findings At Review: id: H004-1; severity: P1; description: Timing restore/watchdog use name-only mutation and may overwrite replacement deployment; fallback disarm precedes final equality proof.; code: ExecutorTimingFixture.start_watchdog/restore; owner: owned wrapper; shared env window parent; id: H004-2; severity: P1; description: Early timeline failures lose cleanup incident and can skip quiesce/ownership restoration.; code: command_timeline/execute_case/cleanup_case; owner: owned runner; id: H004-3; severity: P2; description: Reissue predicate is not tied to a post-kill sample and final command/generation identity; normal renewal not observed.; code: lease_reissue_observed/evaluate_reclaim; owner: owned runner; id: H004-4; severity: P1; description: Shared physical reset proof is weaker than exactly one target reset, as detailed in H003-3.; code: reset_case.host_errors; owner: DESTR helper owner
- Reviewer Handoff: Owned exact owner lookup, incremental incident capture, post-kill same-command identity proof, UID Pod deletion and actual schema2 env-window integration implemented. Replica changes use stdin CAS; watchdog invokes bound window close and UID scale restore; restoration is verified before disarming. Required HostProbe state_directory and fail-stop cleanup are covered by fakes. Physical reset proof remains DESTR-owned.

### GF-REGIONAL-HA-005

控制面滚动升级期间数据面只观察到可重试错误

Current catalog: live-service-action; manual; NOT_RUN.

- Purpose: contract: Restored rollout continuity case: formal execution requires --all-deployments over every enabled CPU role. Only transient transport/503-with-Retry-After errors; no 401/403/500; event accounting, durable replay and every accepted processor request finishes internally 200.; catalog: Parent accepted reactivation; live-service-action, fresh LIVE NOT_RUN. Ingress-only mode is selective scope, not full formal PASS.; order: Active phase12 after HA002 and before HA006; central order/spec are parent-owned.
- Runner: scripts/e2e/regional/run_ha005_rollout_continuity.py; scripts/e2e/regional/probes/ha005_probe.py
- Entry Points: main remains executable despite retirement; defaults ingress rollout, --all-deployments rolls every enabled CPU role.; _run_rollout_case -> synthetic registry -> GPU-side host telemetry/claim probe -> _restart_and_observe -> receipts/continuity -> teardown; HA-009 imports continuity_errors and probe setup, but never _restart_and_observe.
- Assertions: Waits for baseline event/claim attempt counters; checks updated/available/Ready replicas, observed generation and old UID removal.; Event attempts-accepted=failures, buffered=failures, empty outbox, nonempty accepted IDs, returned receipt status COMPLETED/200 and no 401/403/500.; Does not require receipt ID set/count equality, successful claims during rollout, failed/replayed event conservation or Retry-After.; No event failures is explicitly recorded as outbox not exercised, but can still PASS the overall continuity smoke test.
- Safety And Cleanup: Synthetic cluster has token Secret and TTL; fixture does not intentionally submit physical operations.; Originally --all-deployments could differ from approved plan because current details were not checked; scope equality guard now added.; Fixed probe/Secret names and unconditional cleanup can delete pre-existing resources even after preflight refusal.; Uncaught logging/deletion exceptions in finally can skip teardown and final evidence; failed resource reads are now checked.; 900s probe lifetime and 30min registration do not cover the full three-role worst path.
- Necessity: HA-005 is still needed as rollout coverage. Preserve HA-009's correct zero-rollout credential design and let parent restore or replace the retired rollout case rather than making rotation roll again.
- Overlap: case_ids: GF-REGIONAL-HA-009; GF-REGIONAL-HA-001; classification: supersession disproven by injection/scope; rationale: Assertion overlap exists through continuity_errors. Injection fails the superset test: 005 calls rollout restart and requires old UIDs gone, whereas 009 calls credential refresh and fails on any generation/UID change. Scope also differs: 005 can cover all enabled roles, 009 only hot credential reload. 001 deletes individual Pods, not a Deployment rolling strategy.
- Tests: existing: tests/regional/test_ha005_rollout_continuity.py: receipt arithmetic, outbox vacuity, rollout predicate, baseline polling and target selection; missing_behavioral: All-deployments plan drift, retired CLI refusal, no claim success and malformed/partial receipt identities.; Dropped replayed event, 503 without Retry-After, 400/404/502 and final stop/drain race.; Preflight foreign resources, cleanup after partial create, teardown despite log/delete timeout and UID ownership.; In-process probe loop with fake sink/HTTP and finite stop, so behavior not source text is exercised.; execution: Owned continuity/cleanup/component Python and genuine schema3 all-deployments tests passed within final269. Formal ingress-subset refusal passed before any reads. No live rollout.
- Findings At Review: id: H005-1; severity: P1; description: HA-009 no longer injects rollout; retirement leaves rollout continuity unexecuted in formal order.; code: testcases catalog/order; HA009._run_rotation_case; owner: parent catalog/order/spec; id: H005-2; severity: P1; description: Weak receipt/error accounting permits missing accepted or replayed requests and permanent claim failure to pass.; code: continuity_errors; probes/ha005_probe.main; owner: owned runner/probe; id: H005-3; severity: P1; description: Unowned fixed-resource cleanup and an early cleanup exception can remove foreign resources or leave registered synthetic clusters.; code: _cleanup_rollout_case; owner: owned caller; parent registry helper; id: H005-4; severity: P2; description: Probe/registration lifetimes and window checks do not cover all-deployment worst path.; code: pod_manifest/_run_rollout_case; owner: owned runner
- Reviewer Handoff: Parent ACCEPTED active phase12 restoration after HA002 before HA006. Formal main/run_case now require --all-deployments before any reads; ingress-only mode requires selective scope and cannot provide full formal PASS. Added predecessor identity, exact accepted receipt accounting/claim success, stopped-probe final evidence, all-role budget/TTL coverage and nonce/UID cleanup. HA009 zero-rollout is not a substitute. All owned focused tests passed; central order/spec remain parent-owned.
- Current Disposition: Reactivated after HA-002; formal scope requires every enabled CPU Deployment.

### GF-REGIONAL-HA-006

executor Pod 被强杀后动作被接管而不是丢失

Current catalog: live-non-destructive; manual; NOT_RUN.

- Purpose: contract: Force-delete current Executor during a non-destructive remote command; distinguish LEASED-expiry versus immediate WAITING takeover, final command success and exactly one physical action.; catalog: manual, live-non-destructive; describes Node Agent ledger despite synthetic implementation.; order: Phase 12 after HA-002; no predecessor check in runner.
- Runner: scripts/e2e/regional/run_ha006_executor_takeover.py; scripts/e2e/regional/probes/ha006_executor.py
- Entry Points: run_case -> synthetic registry -> two temporary Executors -> seed RUN_DCGM_DIAGNOSTIC with test-only owner -> wait_first_owner -> force delete -> wait_terminal -> takeover_errors; SharedLedgerAdapter emits WAITING then uses notification dedup as an atomic simulated-action ledger; no DCGM or Node Agent operation occurs.
- Assertions: Final survivor identity, cached ledger reuse, summed simulated counters ==1, stable notification ID and suppressed drill result.; Takeover bound uses post-kill lease expiration and Store updated_at, with poll/backoff tolerance.; WAITING observations are informational; no timed WAITING kill scenario or required alternate owner.; Previously missing lease bypassed every integrity assertion. Integrity now runs first; only clean missing-lease evidence is INCONCLUSIVE.
- Safety And Cleanup: Temporary test-only owner and synthetic cluster avoid physical mutation; fixture Pods have active deadlines.; Force deletion and cleanup use fixed names, not captured UID. API deletion is not proof the original process has stopped.; State snapshot taken before kill may miss later simulated actions; killed owner's counter is not observed again.; Finally can still abort on an early log/delete timeout; failed residual reads now propagate.; SQL probes use projected credentials and existing v17 delete routines.
- Necessity: Valuable remote lease failover smoke test, but the report must label simulation and must not use an unobserved kill branch to hide integrity defects or claim Node Agent physical proof.
- Overlap: case_ids: GF-REGIONAL-HA-004; GF-REGIONAL-CMD-006; GF-REGIONAL-HA-001; classification: complementary synthetic lease timing; rationale: 004 uses a real GPU reset and Node Agent ledger; CMD-006 focuses stale completion rejection; 001 deletes CPU Pods. Different injection/scope, not duplicates.
- Tests: existing: tests/regional/test_ha006_executor_takeover.py: two-round adapter, missing lease, timing, WAITING metadata and summed counters; missing_behavioral: No lease plus duplicate/wrong-owner/FAILED result must be FAIL, not INCONCLUSIVE.; Same-owner repeated and concurrent adapter calls count one action.; Negative chronology, survivor re-lease mistaken for killed lease, stale counter and exact kill UID.; Both branch orchestrations and all cleanup failure points via mocked commands/Store; real PostgreSQL lease expiration separately.; execution: Owned same-owner concurrent replay, timing, ownership and fail-stop tests passed within final269. No test Pod was created outside fake APIs.
- Findings At Review: id: H006-1; severity: P1; description: Missing-lease branch used to hide integrity errors behind INCONCLUSIVE.; code: run_case/takeover_errors; owner: owned runner; status: fixed; broader orchestration regression pending; id: H006-2; severity: P2; description: Same-owner replay considered the same notification ID a fresh win and incremented action count again.; code: SharedLedgerAdapter.execute; owner: owned probe; status: initial fix added; concurrent behavior needs follow-up; id: H006-3; severity: P1; description: Post-kill lease may belong to survivor; no genuine WAITING kill/timing branch or Node Agent physical action proof.; code: run_case/waiting_branch; owner: owned runner; parent spec/catalog; id: H006-4; severity: P1; description: Fixed-name, unchecked cleanup and missing UID/deadline/identity constraints remain.; code: run_case finally/main; owner: owned caller; parent shared helpers
- Reviewer Handoff: Owned replay serialized per adapter and counted once; integrity judged before INCONCLUSIVE; timing uses the killed owner's pre-kill lease, not a survivor lease. Missing cross-replica WAITING observation is inconclusive, not PASS. Added schema3 identity/predecessor, nonce/UID temporary resources and fail-stop cleanup. Physical action is simulated and a forced WAITING-owner-kill timing extension is still distinct.

### GF-REGIONAL-HA-007

control-worker 带合法长请求滚动退出

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: contract: Local subprocess uses production ShutdownCoordinator to drain simulated 30/70/110s requests under SIGTERM; default also deliberately exercises 140s over-budget failure. Not a Kubernetes rollout test.; catalog: command, non-destructive; YAML omits the 140s negative branch and still claims every request completes/no orphan LEASED rows.; order: Phase 12 after HA-006; catalog writes under a fixed artifacts path, not the formal run's cases/<id> path.
- Runner: scripts/e2e/regional/run_ha007_control_worker_shutdown.py
- Entry Points: main -> run_probe -> isolated _child -> daemon blocking thread -> SIGTERM event -> ShutdownCoordinator.join/raise_if_failed
- Assertions: Reads generated lifespan budget/grace (130/240s), verifies completed count, return code, failure thread and elapsed shutdown.; Over-budget branch expects zero completion and a coordinator failure, unlike catalog's unconditional completion language.; No actual processor request, lease, replay, application lifespan or duplicate external side effect exists.; New checks reject signal-killed children, empty durations and nonfinite/negative budgets.
- Safety And Cleanup: Local synthetic child only; no cluster or provider access.; Old fixed started/result paths could reuse stale readiness/results; per-probe unique directories now prevent that.; Old readiness exception could leave child until its self-timeout; parent now kills/waits its own child in finally.; Child self-times-out when parent never signals; inherited environment still requires sanitized invocation.
- Necessity: Appropriate subprocess timing test if honestly scoped. It cannot certify lease drainage or Kubernetes termination and must separate the intentional negative branch from positive criteria.
- Overlap: case_ids: GF-REGIONAL-HA-008; GF-REGIONAL-HA-005; classification: complementary graceful-shutdown path; rationale: 008 invokes fatal deadline exit and Store lease release; 005 observes live Deployment rollout. 007 directly drives ShutdownCoordinator under OS signals.
- Tests: existing: tests/regional/test_ha007_control_worker_shutdown.py: real short subprocess, budget/deadline, no-signal self-exit and clean termination; missing_behavioral: Fresh second run cannot read first run's markers/results; all abnormal parent paths reap children.; Full worker lifespan with a real isolated processor lease and exactly-once observable side effect.; Formal evidence path/source identity and positive-versus-negative catalog assertions.; execution: Short real local shutdown subprocesses and canonical-bound/unbound main tests passed within final269. No live control-worker lifecycle claim.
- Findings At Review: id: H007-1; severity: P1; description: Stale state files plus nonzero-overbudget acceptance could falsely pass a child killed before installing its handler.; code: run_probe/evaluate_run; owner: owned runner; status: fixed; id: H007-2; severity: P2; description: Simulated sleeping thread has no processor lease; 140s branch conflicts with catalog positive assertions.; code: _child; fault-scenarios.yaml; owner: owned test/runner; parent catalog/spec; id: H007-3; severity: P2; description: Fixed/default output does not produce formal predecessor evidence under the regional run directory.; code: catalog command/run_probe; owner: parent tools/catalog
- Reviewer Handoff: Owned unique child state, finally/reaping, signal-kill rejection and bounded numeric validation implemented and isolated tests exercised. Main writes canonical case path; optional release/cluster arguments bind the enclosing run and predecessor, while unbound local output explicitly cannot satisfy a formal predecessor. Parent tools own argument/sidecar integration; full busy processor lease survival is a separate extension.

### GF-REGIONAL-HA-008

processor deadline fatal-exit 与 lease 释放

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: contract: Independent worker subprocess crosses real fatal deadline callback, exits70, releases request if possible; replacement process claims/completes and stale old lease is refused; failed release has genuine ERROR evidence.; catalog: command, non-destructive; command omits --run-dir so output uses a deleted temporary directory.; order: Phase 12 after HA-007; subsequent formal case assumes this can produce case evidence.
- Runner: scripts/e2e/regional/run_ha008_processor_exit_probe.py; scripts/e2e/regional/run_ha008_processor_exit_acceptance.py
- Entry Points: run_acceptance -> _run_branch twice -> real ProcessorFactory/ProcessorCoordinator subprocess -> _mark_execution_deadline_exceeded -> _recycle thread -> os._exit(70); Parent opens private SQLite, waits lease/not_before, claims as another owner, tests stale completion and completes replacement.
- Assertions: Both branches exit70; normal PENDING versus failed-release LEASED; replacement owner differs, stale result rejected, final COMPLETED/200.; Requires ERROR from gpu_fault.processor.coordinator, not merely injected traceback text.; Second owner is currently the parent process, not a replacement subprocess or application; no health endpoint is probed.; Fatal threshold is explicitly1 instead of production multiple-strike default, which is reasonable branch isolation but should remain documented.
- Safety And Cleanup: SQLite and raw lease claim stay in a private TemporaryDirectory and are removed; ordinary evidence contains no lease token.; subprocess.run timeout terminates its child; failure before report write can leave no structured case verdict.; No live DB or cloud credential required; inherited environment must be sanitized.
- Necessity: Real exit70 and Store fencing evidence is stronger than a callback mock. Replacement-process/lifespan and durable PostgreSQL claims remain narrower than the spec.
- Overlap: case_ids: GF-REGIONAL-HA-007; GF-REGIONAL-CMD-006; classification: distinct fatal processor failure domain; rationale: 007 is graceful thread drain; CMD lease checks concern remote commands, not processor lane/request leases and fatal process recycling.
- Tests: existing: tests/regional/test_ha008_processor_exit.py: log classification and one failed-branch verdict; tests/processor/test_processor_leadership.py::test_processor_abandons_in_flight_request_lease; missing_behavioral: Run real fatal and replacement subprocesses against isolated SQLite in focused tests, verify private temp cleanup and no secret-bearing output.; Timeout/no-claim/malformed evidence branches produce FAIL and reap owned processes.; Replacement worker health/lifespan and serial isolated PostgreSQL lane/fence behavior.; execution: Real fatal and independent takeover subprocesses on isolated SQLite passed for both branches within final269, with subprocess coverage enabled. No direct acceptance CLI or live deployment call.
- Findings At Review: id: H008-1; severity: P2; description: Replacement claim/completion is in the parent, not the specified second process; app health path is absent.; code: _run_branch; owner: owned runner/probe; id: H008-2; severity: P2; description: Catalog omits a persistent formal run path; TemporaryDirectory evidence is deleted and cannot satisfy following formal predecessor.; code: main; catalog command; owner: parent tools/catalog; id: H008-3; severity: P2; description: Unit tests do not execute _run_branch or the fatal probe; baseline probe coverage is0%.; code: tests/regional/test_ha008_processor_exit.py; owner: owned tests
- Reviewer Handoff: Owned real replacement subprocess implemented and exercised against isolated SQLite for both fatal-exit branches. Distinct child/replacement/parent PIDs, stale completion rejection, final completion and private-temp cleanup checked. Canonical main evidence and optional enclosing release/cluster binding added; failures persist FAIL. No live health endpoint or PostgreSQL acceptance claimed.

### GF-REGIONAL-HA-009

Aurora凭据轮换期间零滚动、投影凭据与全角色新连接验证

Current catalog: live-service-action; manual; NOT_RUN.

- Purpose: contract: Rotate Aurora master credentials, verify before Secret update, project DSN to every CPU role, reload without any rollout, preserve request/command/notification semantics, second refresh NOOP.; catalog: manual, live-service-action; YAML still incorrectly requires all roles to roll.; order: Phase 12 after HA-008; no predecessor check in runner. Alleged successor to HA-005.
- Runner: scripts/e2e/regional/run_ha009_aurora_credential_rotation.py; scripts/e2e/regional/run_ha005_rollout_continuity.py; scripts/e2e/regional/ha_store_probe.py
- Entry Points: _run_rotation_case -> master_secret_arn/version metadata -> synthetic traffic/records -> detached refresh watchdog -> Secrets Manager rotate-secret -> refresher Job -> projected digest -> idle observation -> second Job; Runtime StoreCredentials reloads the projected DSN only on new connection attempts; current healthz reads cached registry/processor readiness, not a direct SQL reconnect.
- Assertions: AWSCURRENT changes, old becomes AWSPREVIOUS; Secret digest changes; first log rotated=True/restarted=False, UID/generation/restart counts steady.; Per-Pod projected DSN digests, post-idle health200 and pool error metric presence, zero auth log markers, HA005 continuity checks and one suppressed notification marker.; No proof that command/notification crossed the rotation window, no actual notification send, no lane-completion cardinality.; No rejected-candidate/failure Warning Event path is injected.; First fixes add actual role ports, exact expected Pod-map coverage, metrics HTTP status, finite values and refusal when idle budget cannot cover max_idle.
- Safety And Cleanup: Secret values are hashed in memory; provider version IDs/digests rather than passwords are reported.; Authorized zero-rollout Store and v17 changes are preserved. SQL probes now reload current projected DSN instead of connecting with stale Pod env passwords.; Emergency refresh is forward recovery, not restoring an obsolete password; watchdog remains armed if emergency refresh fails.; Uncaught logging/deletion exceptions can still skip teardown; fixed Job/probe names and delayed CronJob creation lack UID/run ownership.; RDS/Secret/CronJob/CPU Store identities are not cross-bound before rotation; provider API for an RDS-managed Secret needs a verified supported contract.
- Necessity: Keep the correct no-rollout design. A mounted file plus health200 after sleep does not prove each worker process opened a fresh authenticated connection; max_idle shrinks idle pools and does not force all connections to recycle.
- Overlap: case_ids: GF-REGIONAL-HA-005; GF-REGIONAL-HA-010; GF-REGIONAL-NOTIFY-007; classification: HA005 supersession invalid; other contracts complementary; rationale: Same continuity predicate is not the same injection: 009 explicitly forbids the rollout that 005 requires. 010 changes writer availability, and 007 exercises outbox retry/terminal states rather than credential reload.
- Tests: existing: tests/regional/test_ha009_aurora_credential_rotation.py: steady/rolled predicates, TTL/watchdog, idle observation verdict; tests/regional/test_aurora_credential_refresh.py: verify-before-Secret and no-rollout unit contract; tests/regional/test_ha_runner_fail_closed.py: projected DSN, role port, failed auth log reads; missing_behavioral: Whole rotation orchestrator, absent/partial maps, empty workers, malformed Job outcomes and warning/no-change failure branch.; Direct SQL reconnection on long-lived Pod environment after file rotation with real isolated PostgreSQL and no password output.; Per-process fresh connection proof and error-counter deltas; not a sleep/max_idle assumption.; All failed cleanup stages attempted and no foreign resource deleted after preflight refusal.; execution: Owned credential/health/no-rollout and fake SQL handshake tests passed within final269. The two parent-authorized promoted helper tests also passed: fully mocked in-process schema3 plan and zero-rollout rejection. No Secret/RDS/Kubernetes access.
- Findings At Review: id: H009-1; severity: P1; description: Probe used8080 for workers/spool and SQL cleanup used stale environment DSN after hot rotation.; code: probe_pod; BASE.cpu_python; owner: owned runner/helper; status: fixed; id: H009-2; severity: P1; description: Missing observation maps or failed log reads could pass as no errors; idle window silently clipped below configured max_idle.; code: rotation_errors/auth_failures_in_logs/idle_wait_seconds; owner: owned runner; status: fixed; id: H009-3; severity: P1; description: No fresh connection proof, command/notification overlap or lane exactly-once evidence; current healthz does not authenticate to DB on every call.; code: _run_rotation_case/seed_runtime_records/probe_pod; owner: owned runner; parent runtime proof integration; id: H009-4; severity: P1; description: Cleanup can abort before teardown and delayed mutation lacks resource identity. Database/Secret/CronJob binding and supported RDS-managed rotation API are not proven.; code: _cleanup_rotation/master_secret_arn/start_refresh_watchdog; owner: owned caller; parent provider identity support; id: H009-5; severity: P2; description: Catalog/order supersession and rollover wording are stale; compatibility wait_deployments still references removed consumer_rollout budget.; code: wait_deployments; catalog/order/spec; owner: owned dead helper; parent contracts
- Reviewer Handoff: Owned role ports/projected credentials/complete health and metrics populations fixed. Each CPU Pod exec child now performs a fresh SQL handshake; output explicitly does not claim every application pool recycled. Zero-rollout UID/generation/replica invariants remain. RDS-managed rotation uses ModifyDBCluster.RotateMasterUserPassword and refuses pending unrelated changes. Owned probes/refresh Jobs have nonce/UID receipts; failures retain forward-recovery and cleanup evidence. Cross Store/RDS/Secret/CronJob binding and fallback-watchdog identity still require parent-supported validation before LIVE.
- Current Disposition: Catalog now matches the existing zero-rollout contract. Shared Aurora binding also verifies refresh program/ServiceAccount/RBAC/IAM. Refresh Jobs clone the verified template; unstarted rotation disarms its watchdog while ambiguous submission retains recovery. Log evidence is a digest and parsed booleans, not raw output.

### GF-REGIONAL-HA-010

Aurora 写入端故障转移只让控制面 NotReady 而不重启任何 Pod，启动退避有界

Current catalog: live-service-action; manual; NOT_RUN.

- Purpose: contract: Quiet control plane undergoes a real writer failover and one ingress deletion; surviving liveness200, readiness only200/503 and timely recovery, no existing restarts, bounded replacement startup and no registry Secret drift.; catalog: manual, live-service-action; prior PASS not reused as new verification.; order: Phase12 last entry with explicit HA-001 predecessor.
- Runner: scripts/e2e/regional/run_ha010_aurora_blackout_liveness.py; scripts/e2e/regional/ha010_verdicts.py; scripts/e2e/regional/probes/ha010_probe.py; scripts/e2e/regional/run_ha003_aurora_failover_reset.py (RDS helpers)
- Entry Points: run_standard_case -> preflight -> plan identity -> samplers -> quiet recheck -> RDS failover + ingress delete -> replacement/rollout waits -> sampler verdict -> cleanup; Probe independently queries local livez/healthz using actual named HTTP ports.
- Assertions: Real RDS writer change, available status, minimum3 ingress, ready roles, registry ready/no drift/digest; before/after UID and restart invariants.; Sample time span, every surviving livez200, readiness error policy, recovery budget and final registry readiness.; No readiness outage or startup retry is required; a fast failover can legitimately keep readiness200 but does not exercise stale-readiness/failure handler.; Initial fixes enforce finite budgets, monotonic gap-bounded samples, readiness200/503 only, healthy payload on200 and correct irregular outage integration.; Sampler exit errors now fail; cleanup waits for local child exit and continues to other samplers.
- Safety And Cleanup: No GPU or workflow injection. Preflight/failover checks remote queue but uses shared fault-only backlog semantics rather than literal all-queue emptiness.; Predecessor/result identity was absent; now release/cluster binding is passed and emitted.; Deletion still uses a stale preflight Pod name without atomic UID; RDS cluster is not bound to Store endpoint.; No postflight quiet queue/command check, all-role budget consensus or monitoring of replacement before RDS wait finishes.; Default420s sampling cannot cover worst900s failover plus readiness budget; this yields insufficient evidence, not success.
- Necessity: High-value non-GPU availability case, complementary to destructive HA-003. Distinguish actual writer failover from an exercised readiness refusal and do not accept gaps or dead samplers as continuous observation.
- Overlap: case_ids: GF-REGIONAL-HA-003; GF-REGIONAL-HA-001; GF-REGIONAL-HA-009; classification: complementary liveness/readiness scope; rationale: 003 proves physical action dedup under failover, 001 CPU deletion under traffic, 009 credential hot reload without writer failover. None includes all this case's health/UID/startup assertions and quiet scope.
- Tests: existing: tests/regional/test_ha010_aurora_blackout_liveness.py: extensive pure verdicts, port extraction, RDS return unpacking and real local sampler stdin; missing_behavioral: Full execute/cleanup sequence with fake RDS/Kubernetes, every failing observation, exact expected sampler/Pod identity and deadline changes.; Missing replica samples, failed initial metrics/health reads, replacement events during RDS wait and UID-safe delete.; Use a short local HTTP server to exercise probe.fetch/main through real status/timeout behavior.; Separate readiness-refusal/startup-retry proof with isolated PostgreSQL; no need to cause a longer production blackout.; execution: Owned timeline/status, loopback HTTP sampler, UID request and cleanup tests passed within final269. No live writer failover.
- Findings At Review: id: H010-1; severity: P1; description: Sparse samples and nonfinite budgets could evade continuous observation; some invalid readiness statuses passed.; code: ha010_verdicts.sampler_errors/positive_seconds; owner: owned verdicts; status: fixed; id: H010-2; severity: P1; description: Local sampler kill was not waited and a false stopped flag did not fail cleanup.; code: Sampler.stop/_cleanup; owner: owned runner; status: fixed; id: H010-3; severity: P1; description: Predecessor lacked identity binding, fixed; stale name deletion, Store/RDS binding and postflight quietness remain.; code: read_only_preflight/request_failover/execute_case; owner: owned caller; parent identity support; id: H010-4; severity: P2; description: No observed readiness503 or startup retry is required, and long RDS wait outlives sampler; actual fault effect should be reported separately.; code: _start_samplers/_wait_replacement/ha010_verdicts; owner: owned runner; parent spec/catalog
- Reviewer Handoff: Owned gap-bounded finite timeline/readiness verdicts, bound predecessor/result, UID Pod deletion, postflight quietness and sampler reaping implemented. Supervision loss still stops local samplers but suppresses remote cleanup. Real writer outage, readiness refusal/startup retry and Store/RDS binding remain unexecuted; fast/no-refusal failover must not be described as refusal coverage.
- Current Disposition: The same Store/RDS binding is present in preflight and rechecked immediately before this runner's independent failover call. The three bound HA callers and helper passed 261 focused tests; this is not LIVE or application-pool-wide reconnection evidence.

### GF-REGIONAL-ISO-001

两个集群使用相同 node name、job ID 和 attempt ID 时状态完全隔离

Current catalog: live-workload-restart; manual; NOT_RUN.

- Purpose: Use identical job/attempt/virtual node names across A and B with fresh empty control-record baselines; inject XID11 only into A and prove A restarts once while B's workload, budget, incident scope, and executor remain unaffected.
- Runner: scripts/e2e/regional/run_workload_acceptance.py; scripts/e2e/regional/managed_workload_fixture.py; scripts/e2e/regional/multi_cluster_fixture.py; scripts/e2e/regional/run_destr009_workload_restart.py
- Entry Points: run_workload_acceptance.run_iso001; run_workload_acceptance.virtual_isolation_errors; run_workload_acceptance.executor_logs_since; run_workload_acceptance.cleanup_workload
- Assertions: Checks empty initial identities, two physical registrations, same virtual node with local Pod names/UIDs and disjoint GPUs, A workflow/budget=1, B absent budget/decision/workflow, unchanged B Pod UIDs, and no A command ID in B logs.; The virtual-state assertions are evaluated only after fault injection and both recovery waits, so an already-failed isolation check does not stop the injection.; executor_logs_since uses check=False and does not require any Ready B executor/log stream, making missing logs an apparently clean negative.; wait_observation validates RUNNING and 24 GPU counts/UUIDs but not freshness or exact source Pod identities; shared helper belongs to the other family.
- Safety And Cleanup: A cleanup uses wait_for_cleanup_quiescence and defers deletion if it cannot prove quiet; B is deleted without checking whether a cross-cluster regression created work there.; Both workloads are constructed before the precondition checks, and finally invokes delete even if a pre-existing GPU workload or dirty identity caused refusal before submission.; ManagedWorkloadFixture.delete uses fixed names/job labels without UID ownership, suppresses delete errors, and does not confirm disappearance.; No final ISO-001 node baseline/taint/cordon or Observation-terminal proof is performed.
- Necessity: Keep the asymmetric A-fault/B-control case. A clean B counterfactual is the strongest signal that state keys are actually cluster-scoped.
- Overlap: case_ids: GF-REGIONAL-E2E-002; GF-REGIONAL-AUTH-005; GF-REGIONAL-ISO-002; classification: complementary-not-retirable; rationale: E2E-002 faults both sides and therefore cannot prove B remains unchanged when only A faults; AUTH cases deny invalid credentials rather than exercising valid same-ID workload ownership.
- Tests: existing: tests/regional/test_workload_acceptance_runner.py::test_virtual_isolation_compares_each_clusters_own_pods_and_gpus; tests/regional/test_workload_acceptance_runner.py::test_identity_baseline_must_be_empty_or_the_operator_changes_the_identity; tests/regional/test_workload_acceptance_runner.py::test_cleanup_workload_defers_the_delete_when_quiescence_is_unproven; tests/regional/test_workload_acceptance_runner.py::test_iso001_prepares_both_clusters_in_a_thread_pool; missing_behavioral: Drive run_iso001 through fake preparation, virtual posts, injection, recovery, and failure cleanup.; A failed pre-submit baseline must cause zero workload delete/mutation calls.; A mismatched virtual observation must stop before post_xid_event.; Missing/unreadable B logs, stale observations, wrong refreshed Pods, or B in-flight commands must not pass.; Verify run-owned residuals and node/Observation state after both cleanups.
- Findings At Review: Actual runner requires two physical clusters and three-node/24-GPU jobs, while the spec allows logical identities and describes one-node jobs.; Manual virtual observation may race the real watcher; posts/readbacks are not tied to an acknowledged observation revision.; B control evidence can be vacuously clean when logs are unavailable, and limited Store queries can omit unrelated/terminal cross-scope effects.; Shared cleanup ownership and source-observation freshness need parent/DESTR-family coordination.
- Reviewer Handoff: FIXED: dirty baselines prevent all workload mutation; source/refreshed Observations must be fresh and bind the physical Pods. Failed virtual isolation stops before fault injection and restores physical Observations. Injection checks the maintenance deadline; recovery binds event/incident/workflow/commands/budget. B logs must be available, B commands absent, and both cleanup paths verify quiescence and owned UID-fenced resource removal. Full fake lifecycle covers normal, dirty, virtual and wrong-causal-evidence paths.

### GF-REGIONAL-ISO-002

executor 拒绝执行不属于本集群的 command

Current catalog: live-non-destructive; manual; NOT_RUN.

- Purpose: Executor A locally rejects a B command with FAILED/executor-rejected and the exact cluster-mismatch reason; a live crossed-token claim provides the separate control-plane defense.
- Runner: scripts/e2e/regional/audit_executor_local_guards.py
- Entry Points: audit_executor_local_guards.evaluate; audit_executor_local_guards.write_evidence; CommandDispatch._validate
- Assertions: evaluate constructs the real executor with no adapters and invokes _execute with a foreign command.; Checks FAILED, executor-rejected, exact reason substring, and zero unexpected_failures.; The same invocation independently checks CMD-011 fencing and writes both case documents.; No live crossed-token request or node baseline check is performed.
- Safety And Cleanup: No store, queue, adapter, or cluster is used by this fixture; no cleanup is necessary.; release_id is optional and caller-supplied, and cluster B identity is absent from emitted evidence.
- Necessity: The local no-adapter fixture is a strong safe regression test for the second defense, but must not be promoted as full live acceptance without the separate control-plane evidence.
- Overlap: case_ids: GF-REGIONAL-AUTH-008; GF-REGIONAL-CMD-011; classification: shared-fixture-distinct-assertions; rationale: AUTH-008 checks server claim selection, CMD-011 injects stale fencing, and ISO-002 injects foreign cluster identity. Sharing a fixture does not establish an assertion superset.
- Tests: existing: tests/regional/test_regional_acceptance_fixtures.py::test_local_executor_guard_fixture_covers_iso002_and_cmd011; tests/regional/test_command_protocol_audit_contracts.py; missing_behavioral: Mutate the fake execution result reason/status/source and confirm each case independently fails.; Explicitly prove adapter selection is never reached for foreign cluster commands.; Central orchestration must join the local outcome with current live AUTH mismatch evidence and release identity.
- Findings At Review: Catalog/spec mix a local guard test and live no-mutation assertions, but the runner emits PASS from only the local half.; Shared ISO-002/CMD-011 audit ownership requires coordination before editing.; No reason to retire either ID.
- Reviewer Handoff: PARENT_SHARED_CONTRACT: retained the safe local no-adapter guard and did not edit the shared ISO-002/CMD-011 audit. Local foreign-command rejection is not a full live AUTH-008/ISO-002 proof. Parent must compose current authenticated claim evidence with the release-bound local guard; no retirement or unsafe adapter added.

### GF-REGIONAL-ISO-003

Fleet 代理拒绝跨集群读取与状态转换

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: The deployed Fleet proxy must reject cross-cluster list_agents and revoke_agent locally with precise errors, leaving B's agent state unchanged.
- Runner: scripts/e2e/regional/identity_acceptance_iso.py; scripts/e2e/regional/run_identity_acceptance.py
- Entry Points: identity_acceptance_iso.FLEET_CROSS_CLUSTER_PROBE; identity_acceptance_iso.run_iso003; RegionalFleetRegistry.list_agents; RegionalFleetRegistry._transition
- Assertions: Checks both local ClusterExecutorError messages and rejected flags.; Reads B agents before/after but explicitly removes equality from the checks and returns PASS even if it is false.; There is no transport spy in the deployed probe to prove the exceptions happened before any HTTP request.
- Safety And Cleanup: The current runtime correctly rejects before HTTP, confirmed in source and by the application proxy's RecordingClient.; If that ordering regresses, the acceptance probe can call revoke on fixed probe-node without recording ownership or cleanup.
- Necessity: Keep this local-client boundary separately from HTTP authorization; it constrains what a compromised or incorrect executor client can request.
- Overlap: case_ids: GF-REGIONAL-AUTH-014; GF-REGIONAL-AUTH-006; classification: complementary; rationale: Those server checks cannot prove that the client never emits a cross-cluster read or transition request in the first place.
- Tests: existing: tests/regional/test_regional_auth_boundary_acceptance.py::test_iso003_the_fleet_proxy_refuses_cross_cluster_reads_and_transitions; tests/regional/test_identity_multicluster_contracts.py (historical pre-fix symbol: test_iso003_records_the_vacuous_check_as_not_evaluated); missing_behavioral: Exercise the runner probe with a recording transport and reject send-then-raise behavior.; Foreign agent mutation must not be discarded as not_evaluated.; Positive local listing establishes a usable client separately from the exact negative exceptions.
- Findings At Review: The live verdict assumes the runtime ordering that the case is meant to verify.; A source-string test enshrines removal of the unchanged-state assertion.; The before/after read is retained at live cost without contributing an invariant.
- Reviewer Handoff: FIXED: the deployed probe traps HTTP GET/POST before transport and records zero requests; exact local denials and unchanged secondary agents are required. Real local guard execution and send-then-reject/mutation negatives are behaviorally tested.

### GF-REGIONAL-ISO-004

spare health 端点拒绝跨集群查询

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: A spare-health request for B with A credentials must fail exact cluster binding; a same-cluster request for a nonexistent node must be 200, ready=false, with the missing-agent reason.
- Runner: scripts/e2e/regional/identity_acceptance_iso.py; scripts/e2e/regional/run_identity_acceptance.py
- Entry Points: identity_acceptance_iso.SPARE_HEALTH_PROBE; identity_acceptance_iso.run_iso004
- Assertions: One executor process sends both requests with its actual local token/header.; Checks cross 403 and precise binding-message substring, local 200, literal false readiness, and expected one matching agent, found 0.; Transport exceptions propagate rather than becoming an expected denial.
- Safety And Cleanup: Only spare-health reads; no reserve/allocate or node mutation.; The nonexistent node alias is fixed and not pre-verified; returned claim identity is not attested to the selected target.
- Necessity: This is one of the stronger paired-negative/positive runner cases: the local missing-node answer demonstrates the handler is reachable, while the foreign request exercises binding.
- Overlap: case_ids: GF-REGIONAL-AUTH-005; GF-REGIONAL-ISO-003; classification: complementary; rationale: A generic payload mismatch or Fleet proxy rejection does not exercise the spare-health endpoint or distinguish missing-node health from unrelated pool failure.
- Tests: existing: tests/regional/test_regional_control_plane.py::test_regional_spare_health_uses_fleet_registry; tests/regional/test_identity_multicluster_contracts.py::test_iso004_asserts_the_missing_agent_reason_in_one_process; missing_behavioral: Call run_iso004 with all valid paired results and vary each status/detail/readiness/reason.; Wrong primary/secondary response keys, null body, transport exception, and malformed reasons must fail clearly.; Verify a wrong executor Pod cluster identity is refused before the query.
- Findings At Review: The runner-specific existing test is primarily source text, not handler behavior.; Fixed alias and unverified deployed Pod cluster identity weaken reproducibility and evidence attribution; parent identity contract should bind them.
- Reviewer Handoff: VERIFIED_WITH_BEHAVIOR_TESTS: paired foreign 403 and local 200/ready=false/missing-agent reason are tested through run_iso004. Wrong status, reason, readiness and target shape fail. Transport errors are not expected denials; no write or spare allocation is added.

### GF-REGIONAL-ISO-005

namespace allowlist 在控制面与数据面两侧都独立生效

Current catalog: live-non-destructive; manual; NOT_RUN.

- Purpose: Independently prove control-plane and executor namespace allowlists reject a forbidden workload without changing its Deployment/Pod identity. Old spec widens the live control-plane allowlist, conflicting with current hard constraints.
- Runner: scripts/e2e/regional/identity_acceptance_iso.py; scripts/e2e/regional/identity_acceptance_common.py; scripts/e2e/regional/run_identity_acceptance.py
- Entry Points: identity_acceptance_iso.run_iso005; identity_acceptance_iso.ALLOWLIST_WORKFLOW_PROBE; identity_acceptance_iso.REMOTE_COMMAND_RETIRE_PROBE
- Assertions: Creates a fixed forbidden namespace/pause Deployment, calls the deployed adapter with an unsaved workflow, then widens the real registry and waits for one executor rejection.; Requires the control-plane failure/no command, one FAILED executor command with namespace error, and unchanged workload snapshot.; Raw command model dumps and cleanup.read_registry are retained in evidence; the latter can contain plaintext tokens in bootstrap mode.
- Safety And Cleanup: Unconditionally deletes the fixed namespace in finally even when create failed because it already existed; no UID/ownership precondition.; Delete and confirming get use check=False, so an API failure with empty output can look absent.; Lost creation ACK means command_ids can stay empty and orphan the unsaved-workflow command.; Retirement catches every read error as absence, uses stale status after cancellation, ignores cancellation failure, then calls store._delete; a concurrently leased command can be deleted.; Restores the entire old registry against a freshly read generation and thus can overwrite concurrent changes.
- Necessity: Keep the two independent defenses, but replace this unsafe exercise with an isolated deployed-code/local-adapter fixture that never expands a live allowlist.
- Overlap: case_ids: GF-REGIONAL-ISO-002; GF-REGIONAL-AUTH-008; classification: complementary; rationale: Those cases inject foreign cluster identity, not an in-cluster forbidden namespace. Namespace defense must remain separately tested on both sides.
- Tests: existing: tests/regional/test_identity_multicluster_contracts.py (historical pre-fix symbol: test_iso005_retires_commands_settle_cancel_then_delete); missing_behavioral: Pre-existing namespace, create ACK loss, namespace UID replacement, and read failure must never authorize deletion.; Record a command before dispatch and prove cleanup cannot delete a leased/drifting command.; Reject bootstrap plaintext registry and lease-token leakage in all evidence paths.; Execute isolated control-plane/executor namespace guards with real models and recording adapters; assert no live registry mutation.
- Findings At Review: Confirmed violation of the no-online-allowlist-expansion invariant.; Confirmed namespace ownership/data-loss cleanup bug.; Source-order assertions do not test the read/cancel/delete race or failed lookups.; No fail-fast stop after a failed first defense; unsafe escalation to the second stage is still attempted.; Risk/catalog/spec require parent revision; do not weaken the hard safety rule to match the old procedure.
- Reviewer Handoff: FIXED: replaced online allowlist broadening and namespace creation/deletion with deployed CPU adapter/in-memory Store plus deployed GPU no-adapter executor guards. A permitted-namespace positive control proves each guard is reached; exact forbidden-namespace denial and unchanged registry are required. No production command row, workload or namespace is created/deleted. Behavioral tests cover wrong denials, namespace drift, positive-control failure and concurrent registry change. Parent must describe this honest component scope in catalog/spec.
- Current Disposition: Online allowlist widening removed; deployed-code isolated positive and negative guards, zero adapters and unchanged registry.

### GF-REGIONAL-ISO-006

一个 GPU 集群整体离线不阻塞其他集群

Current catalog: live-workload-restart; manual; BLOCKED.

- Purpose: Isolate all of physical cluster B from control-plane TCP443, prove A's p95 grows less than 20% with no errors while an A XID11 recovery completes, observe CPU pressure/restarts, then prove B recovers within five minutes.
- Runner: scripts/e2e/regional/run_iso006_cluster_offline.py; scripts/e2e/regional/probes/cluster_network_probe.py; scripts/e2e/regional/multi_cluster_fixture.py; scripts/e2e/regional/identity_acceptance_common.py
- Entry Points: run_iso006_cluster_offline.read_only_preflight; run_iso006_cluster_offline.execute_case; run_iso006_cluster_offline.latency_errors; run_iso006_cluster_offline.cut_proven; cluster_network_probe.block; cluster_network_probe.unblock
- Assertions: Requires two registry records with distinct EKS/HyperPod strings, both initial claims 200, and DESTR-013 PASS bound only by release.; Five A baseline samples and a default 900-second block replace the spec's 60 samples and 1800 seconds; no B latency/collector-growth baseline exists.; latency_errors rejects transport failure and 5xx, but accepts 401/403/429 and does not judge bad baseline statuses.; No A workload submission, XID injection, workflow, or restart check exists in execute_case.; Pressure comparison uses only initial/final parsed counters; missing metrics default to zero, and intermediate CPU readiness snapshots are recorded but not evaluated.
- Safety And Cleanup: Host timers are armed before OUTPUT/FORWARD REJECT rules; no provider mutation is called.; blocked becomes true only after a successful block receipt, so a first-node ACK loss can skip explicit unblock.; unblock suppresses iptables failures, stops the restore timer, and returns blocked=false without reading rules back.; Returned nonempty probe residuals do not affect PASS unless cleanup raises.; The maintenance deadline is checked only before preparation; rule installation and the observation window can outlive it.
- Necessity: Keep offline-cluster isolation separately from concurrent-fault recovery, but the current runner proves only a narrower REJECT-based claim-path experiment.
- Overlap: case_ids: GF-REGIONAL-E2E-002; GF-REGIONAL-CAP-001; GF-REGIONAL-NET-001; classification: complementary; rationale: Whole-cluster loss, per-cluster load, and a single collector's replay failure differ in scope and pressure mechanism. E2E-002 never cuts B off and cannot replace this counterfactual.
- Tests: existing: tests/regional/test_identity_multicluster_contracts.py::test_iso006_cut_must_be_proven_at_the_transport; tests/regional/test_identity_multicluster_contracts.py::test_iso006_latency_judgment_is_p95_within_twenty_percent_and_no_5xx; tests/regional/test_collector_and_multicluster_fixtures.py::test_cluster_network_probe_arms_restore_before_block; tests/regional/test_collector_and_multicluster_fixtures.py::test_cluster_network_probe_cleans_partial_block_failure; missing_behavioral: 401/403/429, malformed/non-finite latency, missing samples/counters, and incomplete CPU container sets must fail.; Full fake-clock execute_case must prove block-before-measure and cleanup after first block ACK loss.; Residual rule/probe objects or an expired timer/window must fail; failed removal must retain the recovery timer.; Integrate an A recovery callback and require its complete workflow proof before PASS.
- Findings At Review: Required A recovery is entirely missing.; A TLS/JSON exception also satisfies cut_proven, so generic transport failure does not prove the intended network cutoff.; REJECT fails immediately and does not reproduce SG/NACL DROP timeout accumulation; preserve this limitation rather than claim equivalence.; Node/context identity is compared by configuration strings, not live UID/ARN binding; predecessor omits both cluster identities.; Catalog risk and phase maintenance=no conflict with network/service disruption and the missing workload-restart stage.
- Reviewer Handoff: FIXED_FULL_CLOSED_LOOP: A's run-owned three-node/24-GPU workload is prepared before the baseline and recovered from one software XID11 replay while B's host iptables REJECT cut is verified. Exact cut checks bracket A injection/restart; event/incident/workflow/command/budget/notification and new-attempt evidence are required inside the cut window. Latency, CPU continuity, B recovery/Node identity, timer/readback and first-block ACK-loss cleanup protections remain. Site/job/attempt inputs join the existing bound image/target/window settings. Full fake orchestration plus actual helper behavior tests pass; no claim-only PARTIAL substitute remains.
- Current Disposition: Verified host REJECT window now contains a causally bound A workload recovery; do not equate REJECT with SG/NACL DROP timeout accumulation.

### GF-REGIONAL-ISO-007

A集群恢复被预算拒绝时B集群仍独立完成恢复

Current catalog: live-workload-restart; manual; NOT_RUN.

- Purpose: 两个集群同时恢复成功不能证明其中一侧明确失败时另一侧不被全局等待或共享预算阻塞。
- Runner: scripts/e2e/regional/run_iso007_failed_recovery_isolation.py
- Entry Points: guarded main -> read-only preflight -> plan/execute -> evidence and cleanup
- Assertions: E2E-002前置绑定同一release和A/B物理集群对，不使用同一EKS的两个逻辑身份; A工作流因RESTART_BUDGET_EXHAUSTED终态FAILED，无远端RESTART_WORKLOAD，原任务已停止，预算及重启数均为0; B工作流独立终态SUCCEEDED且仅一次本集群任务重启; 事件、incident、workflow、command、预算与通知身份均与各自集群和attempt一致; 两边通知和终态清理均可审计，不留下非终态记录、测试workload或节点元数据漂移; 这是软件事件回放与声明预算的拒绝，不代表真实硬件故障或网络分区
- Safety And Cleanup: Explicit target and current plan; no LIVE executed by this review.
- Necessity: A denied recovery and B successful recovery differ from E2E-002's two successes and ISO-006's network cut.
- Overlap: classification: added_missing_boundary; rationale: A denied recovery and B successful recovery differ from E2E-002's two successes and ISO-006's network cut.
- Tests: tests/regional/test_iso007_failed_recovery_isolation.py
- Findings At Review: Approved LIVE validation remains NOT_RUN; local tests are not hardware evidence.
- Reviewer Handoff: Implemented; final local verification is recorded in the verification section.

### GF-REGIONAL-NET-001

网络中断期间 Collector 持久缓存且恢复后按幂等键补交

Current catalog: live-kernel-log-injection; manual; NOT_RUN.

- Purpose: Ten-minute node egress interruption, three user-space monitor-only kmsg events buffered, unique replay across four channels and short reconnect without duplication.
- Runner: scripts/e2e/regional/run_net001_collector_replay.py; scripts/e2e/regional/probes/net001_node_probe.py
- Entry Points: manual -> main plan/authorize -> Runner.run -> privileged node probe; live-kernel-log-injection, not hardware fault
- Assertions: 25-minute initial window; active services/no NRestarts delta; Ten outage snapshots contain target markers and no CPU records; Three unique evidence IDs, no observed workflow, four replayable queues drain; reconnect preserves IDs
- Safety And Cleanup: 720s/60s host firewall rollback timers, tracked tags, bounded kubectl and pod lifetime; Finally removes tags/probe resources; failed Kubernetes GET can falsely prove absence; cleanup_tag does not validate timer stopped; No explicit abort-signal installation; arm ACK loss can miss active_tags tracking
- Necessity: Distinct long interruption/recovery proof. Injected kmsg is explicitly synthetic. Preflight does not prove node Ready, fresh idle ownership or absence of Pending/Terminating workload.
- Overlap: case_ids: GF-REGIONAL-NET-005; GF-REGIONAL-NET-008; classification: partial_overlap_complementary; rationale: Same outbox transport; local boundary/capacity and auth/dead-letter/stream-identity add different failures.
- Tests: tests/regional/test_net001_collector_replay.py
- Findings At Review: Final evidence counts three IDs but does not require exactly one per requested marker; No-mutation verdict can pass with no incidents yet, before delayed policy processing; No final timer/host-file absence proof; no failure-safe per-resource cleanup test; Maintenance deadline only checked at start, not before each mutation
- Reviewer Handoff: supervised_public_command_boundary_and_Plan3_updated; existing_delivery_proof_limits_retained

### GF-REGIONAL-NET-002

中断期间已领取的 command 被安全回收

Current catalog: live-non-destructive; manual; NOT_RUN.

- Purpose: Expired command result returns409 after interruption; same executor reclaims and idempotent ledger acts once.
- Runner: scripts/e2e/regional/run_net002_command_recovery.py; scripts/e2e/regional/net_command_fixture.py; scripts/e2e/regional/seeded_command_fixture.py; scripts/e2e/regional/probes/net002_executor.py
- Entry Points: manual -> fixture.run_main -> predecessor gate/authorize -> run_case -> synthetic probe executor
- Assertions: 70s block >60s lease; configured gate; lease unchanged until unblock; claimed_total2/reported_failures1/renewal failure; stale409 log; cached final result and ledger1; probe Running
- Safety And Cleanup: Only loopback proxy affected; 100s unblock watchdog; 900s pod deadline; 30min registry/seed lease; Unconditional shared cleanup can mutate on failed predecessor; PROTO-004
- Necessity: Distinct late-result path, but custom client complete gate blocks before product HTTP submission. 180s timeout is intentionally nonproduction and not itself proof of actual HTTP delay.
- Overlap: case_ids: GF-REGIONAL-CMD-006; GF-REGIONAL-NET-006; GF-REGIONAL-NET-003; classification: shared_fixture_distinct_scenario; rationale: Server refuses late result vs client withholds under lost lease vs committed response lost.
- Tests: tests/regional/test_net002_command_recovery.py; tests/regional/test_net_probe_contracts.py::test_net002_gated_client_holds_the_result_until_the_block_lifts
- Findings At Review: No stage/finally fake coverage for run_case; Ready identity and command IDs not bound across snapshots; network result gate is client-side, not a held HTTP POST; documentation must be precise; Predecessor only checked for PASS/case by helper, no expected release/cluster supplied; evidence_identity can return None release/cluster without failing
- Reviewer Handoff: fixed_owned_UID_scoped_cleanup_and_no_implicit_SQL_replay; perf_ACK_CAS_boundary_verified_with_fakes

### GF-REGIONAL-NET-003

结果提交时中断重试后幂等

Current catalog: live-non-destructive; manual; NOT_RUN.

- Purpose: First result commits upstream then connection resets; one terminal replay must preserve result and single suppressed notification.
- Runner: scripts/e2e/regional/run_net003_result_retry.py; scripts/e2e/regional/net_command_fixture.py; scripts/e2e/regional/probes/net003_executor.py
- Entry Points: manual -> fixture.run_main -> run_case -> InterruptingRegionalExecutorClient and TLS byte relay
- Assertions: Forwarded application data and response bytes before reset; original update predates replay; One claim/physical action/replay; no409/renewal rejection; same updated_at; one notification/delivery/SKIPPED result
- Safety And Cleanup: Synthetic command and drill-suppressed notification; no actual email proof; PROTO-004 broad cleanup and ACK retry; mode-aware SQL preserved; PROTO-006 missing workflow lease
- Necessity: The network response-loss scenario is distinct and useful. PROTO-007 means current acceptance proves a test-client replay, not production report retry.
- Overlap: case_ids: GF-REGIONAL-CMD-007; GF-REGIONAL-NET-002; GF-REGIONAL-NOTIFY-001; classification: partial_overlap_complementary; rationale: Broader replay matrix, stale-lease rejection, and actual notification delivery are separate.
- Tests: tests/regional/test_net003_result_retry.py; tests/regional/test_net_probe_contracts.py
- Findings At Review: PROTO-006; PROTO-007; TLS1.3 outer application record may be handshake/KeyUpdate/ticket, not HTTP response; tests use synthetic records only; Probe state writes are not atomic and may be observed partial; Notification query by one exact key cannot detect an extra differently keyed notification; No embedded seed/cleanup script or full fake run coverage
- Reviewer Handoff: fixed_owned; production_executor_retry_seed_lease_and_UID_cleanup_locally_tested; real_SQL_verified_legacy_dual_dedicated

### GF-REGIONAL-NET-004

跨 VPC 单向依赖、鉴权边界与超时配置

Current catalog: read-only-signal-replay; command; NOT_RUN.

- Purpose: Read-only deployed DNS/TLS/NLB timeout/NAT allowlist and CPU-to-GPU credential/network boundary.
- Runner: scripts/e2e/regional/audit_net004_dependency_boundary.py
- Entry Points: catalog command -> configure/audit; actual kubectl get/exec and AWS describe; no plan gate
- Assertions: NLB active TLS443, matching DNS, mode-appropriate IPs/NAT; Verified TLS and executor timeout below NLB idle; One CPU ingress Pod named environment/mount checks; public GPU endpoint401/403 or private unreachable
- Safety And Cleanup: Read-only calls; no resources created; optional output only; No private profile inspected or command executed in this review
- Necessity: Useful deployed boundary probe, but a single ingress Pod and name heuristics do not prove every CPU role lacks GPU credentials.
- Overlap: case_ids: GF-REGIONAL-AUTH-014; GF-REGIONAL-BLAST-001; classification: different_layer; rationale: Transport topology and timeout vs route authorization and permission scope.
- Tests: tests/regional/test_net004_dependency_boundary.py; tests/regional/test_regional_acceptance_fixtures.py::test_net004_dependency_audit_is_environment_driven
- Findings At Review: Private-endpoint probe treats TLS verification/DNS failures as accepted unreachability; Only one Running CPU ingress Pod, not all CPU roles/Ready replicas; envFrom/projected mounts not fully checked; Allowlist checks ignore IPv6/group-source rules; No standard cases/<id> evidence or release/cluster binding; Unit tests concentrate on timeout arithmetic, not credential/security negatives
- Reviewer Handoff: reviewed_unmodified; central_evidence_and_security_proof_limits_retained

### GF-REGIONAL-NET-005

Collector有界outbox容量、dead-letter与实时事件优先

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: Isolated HttpEventSink live-first replay, bounded outbox, write failure and permanent400 dead-letter behavior.
- Runner: scripts/e2e/regional/audit_collector_outbox.py
- Entry Points: catalog command -> main; real sink with fake urlopen and TemporaryDirectory, not live proof
- Assertions: [1,2,1] ordering; background cross-batch drain; keep newest3;400 nonreplayable; unwritable buffered=false
- Safety And Cleanup: Global urlopen restored in finally; TemporaryDirectory removes files; background wait is bounded
- Necessity: Appropriate deterministic boundary case; distinct from network on real node. Not a pytest wrapper, but still local/integration evidence.
- Overlap: case_ids: GF-REGIONAL-NET-001; GF-REGIONAL-NET-008; classification: different_layer; rationale: Local capacity/live-first scheduling vs actual deployed buffering and auth/dead-letter CLI.
- Tests: tests/regional/test_regional_acceptance_fixtures.py::test_collector_outbox_fixture_covers_current_delivery_contract; tests/collectors/test_http.py (sink behavior)
- Findings At Review: No zero-progress-stop/re-wake or new live event during background replay in this audit, despite spec; 400 nonreplayable checked once but no actual later replay attempt proves exclusion; Failure while background worker remains active can race restoration of global HTTP function; Only PASS text/dict output, no standard per-case JSON or failed verdict
- Reviewer Handoff: reviewed_unmodified; local_proof_only_and_parent_evidence_proposal

### GF-REGIONAL-NET-006

动作执行期间丢失 command lease，executor 不在丢失的 lease 下提交结果并计数，重新 claim 后由幂等账本收尾

Current catalog: live-non-destructive; manual; NOT_RUN.

- Purpose: Lease lost during100s action under120s loopback outage; withhold result and reclaim cached once with counters/breadcrumb.
- Runner: scripts/e2e/regional/run_net006_lease_loss_withheld_result.py; scripts/e2e/regional/net006_verdicts.py; scripts/e2e/regional/seeded_command_fixture.py; scripts/e2e/regional/probes/net006_executor.py
- Entry Points: manual -> PlainCaseRunner -> synthetic seed -> production ClusterActionExecutor with HoldingLedgerAdapter
- Assertions: Loss/withhold counters and logs; no409/cancellation; claim2/ledger1; Action-thread lease hold reason; nonterminal while blocked; final cached SUCCEEDED by same executor; Block duration, no watchdog, breadcrumb counter equality
- Safety And Cleanup: Synthetic cluster, no Node Agent, 150s watchdog/900s Pod/seed lease; PROTO-004 and CENTRAL-002 affect cleanup, predecessor and identity
- Necessity: Distinct client-side lease guard proof. Ledger action is simulated; not two-executor or actual NodeAction proof.
- Overlap: case_ids: GF-REGIONAL-NET-002; GF-REGIONAL-HA-006; GF-REGIONAL-CMD-017; classification: shared_fixture_distinct_scenario; rationale: Different moment of lease loss; HA006 adds two claimant identities; CMD017 barrier hold uses same skeleton only.
- Tests: tests/regional/test_net006_lease_loss.py; tests/execution/test_cluster_executor_lease_guard.py
- Findings At Review: Missing/null error counters default0; invalid expiry string skips expiry comparison; no command-ID binding across snapshots; No direct HoldingLedgerAdapter/RefusingProxy lifecycle or runner cleanup fake tests; Maintenance timing does not account for entire registration/probe/teardown time
- Reviewer Handoff: fixed_owned_UID_scoped_cleanup_and_ACK_readback; existing_counter_and_identity_proof_limits_retained

### GF-REGIONAL-NET-007

节点隔离步骤遭遇瞬时 Kubernetes API 故障时步骤 WAITING 重试而非 FAILED，故障消退后工作流照常收尾

Current catalog: live-node-mutation; manual; NOT_RUN.

- Purpose: Node-scoped executor-only webhook returns transient500 during EFA remediation; same step WAITING then full recovery without escalation.
- Runner: scripts/e2e/regional/run_net007_transient_api_outage.py; scripts/e2e/regional/net007_verdicts.py
- Entry Points: manual -> CaseRunner -> read_only_preflight -> collector fixture/deadman/webhook -> EFA unbind/observe
- Assertions: Classifier and server>=30, target idle/Ready, queue gates; Webhook round-trip; paired step/command WAITING retryable marker; exact seven-step success/RECOVERED; no forbidden provider events
- Safety And Cleanup: Live EFA/node mutation, no reset/reboot intended; host bind fail-safe plus deadman; PROTO-008; deadman log read failure can prevent outage cleanup; Preflight checks attempt1 webhook name even for later attempt
- Necessity: Distinct real Kubernetes adapter classification proof layered on COLLECT017 injection. Current substring classifier probe and webhook guard are weaker than behavioral/exact capability proof.
- Overlap: case_ids: GF-REGIONAL-COLLECT-017; classification: shared_injection_distinct_scenario; rationale: Same EFA recovery path, but new narrowly scoped API outage; do not run concurrently.
- Tests: tests/regional/test_net007_transient_api_outage.py; tests/execution/test_retryable_adapter_errors.py::test_transient_adapter_error_becomes_a_waiting_step_not_a_failure
- Findings At Review: PROTO-008; Outage evidence not bound to exact command/step/workflow through recovery; retryable marker does not require actual webhook500; Observed minimum hold duration and continued webhook presence not verdict-gated; No predecessor/release binding in result; current node UID not compared to plan; No OutageFixture mutation/cleanup failure tests
- Reviewer Handoff: fixed_owned_guard_deadman_ACK_UID_cleanup_full_hold_and_same_command_recovery; focused_tested

### GF-REGIONAL-NET-008

Collector outbox 的死信语义：403 保持可重放并在 token 恢复后送达，回放 4xx 死信并可 requeue，中断期间不重开 /dev/kmsg

Current catalog: live-kernel-log-injection; manual; NOT_RUN.

- Purpose: Wrong-token403 remains replayable, retired-channel404 dead-letter/requeue, blackout keeps kernel stream identity and delivers once.
- Runner: scripts/e2e/regional/run_net008_outbox_dead_letter.py; scripts/e2e/regional/net008_verdicts.py; scripts/e2e/regional/collector_window_fixture.py (shared read-only dependency); scripts/e2e/regional/probes/collector_window_probe.py (shared read-only dependency)
- Entry Points: manual -> run_window_case preflight/authorize -> execute three phases -> node probe
- Assertions: Marked403 event count including batching, no dead count, service active; Each marker delivered once;404 seed/list metadata/requeue confirmation; PID/invocation/fd/position continuity; block timer/connectivity; one seed purged
- Safety And Cleanup: 600s token window,300s firewall timer and fail-safe collector start; PROTO-009; flags set after ACK; cleanup is sequential and may leave service stopped; Shared wrapper only removes probe resources, not remaining host mutations
- Necessity: Distinct auth/error classification and retained stream test. Wrong token is not real rotation; kmsg is user-space synthetic.
- Overlap: case_ids: GF-REGIONAL-NET-001; GF-REGIONAL-NET-005; GF-REGIONAL-AUTH-016; GF-REGIONAL-COLLECT-018; classification: partial_overlap_complementary; rationale: Long buffering, local capacity, actual token rotation and processor handler failures remain separate scenarios.
- Tests: tests/regional/test_net008_outbox_dead_letter.py
- Findings At Review: PROTO-009; Missing invocation/position can pass; empty connectivity maps vacuously pass; final timer stop not checked; No before/after replayable counts from real stats passed to some verdicts; No phase/cleanup failure injection tests; execute discards deadline; second collector restart has no settled-reader proof before wake event
- Reviewer Handoff: fixed_owned_real_stats_phase_stop_ACK_intent_and_independent_restore; focused_tested

### GF-REGIONAL-NOTIFY-001

关键动作各有一封 SENT 邮件且含非空 SES MessageId

Current catalog: live-non-destructive; manual; NOT_RUN.

- Purpose: contract: One SENT RESET_GPU completion and one SENT RESTART_WORKLOAD completion with SES/inbox proof; four identical remote-result submissions must not create another notification or provider send. Reuse approved actions, otherwise use an isolated drill, never repeat a GPU action for email.; catalog: manual, live-non-destructive; related_pytest is the workload-restart API notification test.; order: Phase 9 first entry, after BLAST-004; NOTIFY-002 is excluded as superseded.
- Runner: scripts/e2e/regional/run_notification_acceptance.py; scripts/e2e/regional/probes/notification_drill.py
- Entry Points: main -> predecessor_path/predecessor_evidence(release_id, cluster_id) -> authorize_execution -> run_notify001; run_notify001 -> live_action_completed_records -> select_live_record or drill; notification_drill.main -> RestartGuardEmailBuilder -> InMemoryStore.save_notification_if_absent -> AdvisoryNotificationService.send x4; Production result path is AdvisoryNotificationService.dispatch_remote_completion, not the path this drill invokes.
- Assertions: Live candidates are re-read from Store; selected rows must be non-drill SENT with a provider-id-present flag and same_incident_kind_count == 1.; Fallback sends labelled GPU-reset/workload-restart drills; the reset drill is always exercised, including when both live completions exist.; Four statuses, a nonempty first provider id, stable provider ids, and external receipt/dedup valid flags determine the result.; No actual remote-result submissions, notification-count delta, provider call count, or per-message external receipt binding is measured.
- Safety And Cleanup: The drill has a separate in-memory Store and does not execute a GPU or workload action; it can send one or two real emails.; Only provider-id presence/stability is emitted by the probe; production credentials are not intentional output.; The runner resolves a Ready worker once, but does not validate DRILL subject/body markers before delegating the real notifier.; The authorization deadline is discarded, so late sends can occur after the maintenance window.; External JSON is copied wholesale into the result, including arbitrary unvalidated fields.
- Necessity: Useful mail-transport smoke test, but it can hide a failed real completion with a passing drill and cannot presently claim the remote-result exactly-once contract.
- Overlap: case_ids: GF-REGIONAL-NOTIFY-002; GF-REGIONAL-NOTIFY-007; classification: NOTIFY-002 helper-level duplicate only; original remote-result scope not covered; NOTIFY-007 complementary; rationale: 001 and legacy 002 invoke the identical direct-send drill and the identical four-status/provider-id/external-window checks. That is a superset of the weakened 002 runner, not of 002's remote-result injection. 007 drives async outbox RETRY/DEAD instead of synchronous send caching.
- Tests: existing: tests/regional/test_notification_acceptance_runner.py: external windows, live selection, fallback, dedicated reset drill, probe output; tests/regional/test_regional_control_plane.py::test_regional_api_sends_executor_workload_restart_notification; tests/regional/test_regional_control_plane.py::test_regional_api_sends_gpu_reset_completion_once; missing_behavioral: Reject external valid/errors override, received='false', fractional/boolean counters, absent or invalid timestamps, future or partial windows.; Reject FAILED/PENDING/missing real completion instead of masking it with a drill; reject candidates from failed or wrong-release evidence.; Exercise real result handler four times with a RecordingNotifier and count notification rows/provider calls; test retry and concurrent completion.; Bind receipt and dedup observations to run/drill/provider identities and the full first-send/replay interval.; Test complete main orchestration and deadline refusal; replace the existing source-text identity assertion.; execution: Owned case tests passed within final269 focused tests, including both actual completion-handler replay kinds. No SES or live result endpoint called.
- Findings At Review: id: N001-1; severity: P1; description: validate_external_evidence returns computed valid/errors before **value, allowing input JSON to overwrite its verdict. Truthiness and int coercion also accept malformed evidence.; code: run_notification_acceptance.py::validate_external_evidence; owner: owned runner; id: N001-2; severity: P1; description: Four service.send calls on one pre-built in-memory notification bypass completion handler, builder re-entry, lease/fence validation, durable dedup and notification-row creation. Spec says four /result submissions.; code: probes/notification_drill.py::main; run_notification_acceptance.py::run_notify001; owner: owned runner/probe; parent spec/catalog integration; id: N001-3; severity: P1; description: select_live_record filters out failed, pending or missing records and run_notify001 falls back to a successful isolated drill even when this run did produce a real completion that failed delivery.; code: run_notification_acceptance.py::select_live_record/run_notify001; owner: owned runner; id: N001-4; severity: P2; description: Candidate extraction checks cluster/category but not evidence PASS/release identity. External windows cover only a start timestamp and are not tied to actual message IDs or replay end; delta=0 over a window including the first send is ambiguous.; code: action_completed_records_from_evidence; validate_external_evidence; notification_drill.main; owner: owned runner/probe; parent evidence contract
- Reviewer Handoff: Owned fixes implemented: computed external verdict cannot be overwritten; typed counts/timestamps and complete live Store reads; failed or wrong-release candidates cannot be replaced by drills; actual complete_remote_command handler replayed four times with one notification and one provider invocation; DRILL/deadline guards. Real HTTP authorization, durable PostgreSQL replay and provider-acceptance crash ambiguity are separate validation boundaries.

### GF-REGIONAL-NOTIFY-002

已并入：重复 remote result 不产生第二封邮件

Current catalog: live-non-destructive; manual; SUPERSEDED.

- Purpose: contract: Superseded duplicate remote-result case. Preserve the requirement that repeated SUCCEEDED completion payloads create one notification and one send.; catalog: manual, live-non-destructive, evidence SUPERSEDED, superseded_by NOTIFY-001.; order: do_not_run; not a formal predecessor.
- Runner: scripts/e2e/regional/run_notification_acceptance.py; scripts/e2e/regional/probes/notification_drill.py
- Entry Points: run_notify002 directly calls the same GPU-reset drill as run_notify001 and can return a reproduction PASS with superseded_by.; Normal main calls predecessor_path first; formal_predecessor rejects this do_not_run case before the legacy handler is reached.
- Assertions: Legacy helper checks four statuses, provider-id presence/stability and external dedup evidence.; It neither submits remote results nor counts notification rows or real sends.; The callable helper and the CLI disagree on whether superseded reproduction is supported.
- Safety And Cleanup: Direct helper use sends one labelled email from an isolated Store.; No hardware action or persistent Store cleanup is required.; No separate formal execution should be re-enabled without parent catalog/order review.
- Necessity: Avoiding a duplicate mail smoke test is reasonable. Retiring the original remote-result contract is not justified by the current direct-send drill.
- Overlap: case_ids: GF-REGIONAL-NOTIFY-001; classification: exact weakened-runner duplicate; intended-contract superset proof fails; rationale: Assertion: same four-status/provider-id checks and same external validator. Injection: same notification_drill.main direct-send x4. Scope: same in-memory Store and real SES notifier. These prove helper equivalence, but neither runner injects the remote-result API path required by the original case.
- Tests: existing: tests/regional/test_notification_acceptance_runner.py::test_notify002_declares_that_notify001_supersedes_it; tests/regional/test_regional_control_plane.py::test_regional_api_sends_gpu_reset_completion_once; missing_behavioral: Verify the superseded CLI refuses before email/provider invocation and cannot emit formal PASS.; Carry result-route replay, notification-row-count and provider-call-count tests into NOTIFY-001.; execution: Owned legacy-helper tests passed within final269; formal CLI remains retired and no provider was contacted.
- Findings At Review: id: N002-1; severity: P2; description: Comment and helper advertise callable reproduction, but main always asks formal_predecessor for a do_not_run case and is rejected. Keep retirement explicit rather than claiming an executable reproduction path.; code: run_notification_acceptance.py::main/run_notify002; regional_case_contract.py::formal_predecessor; owner: owned runner; shared contract parent-owned; id: N002-2; severity: P1; description: The supersession claim inherits N001-2: no remote-result injection or durable notification creation is covered.; code: probes/notification_drill.py::main; owner: owned probe; parent spec/catalog/order
- Reviewer Handoff: Keep SUPERSEDED. CLI now refuses before site/provider access. NOTIFY001 carries actual isolated result-handler replay, notification cardinality and provider-call assertions; the legacy imported helper is not a formal execution entrypoint.

### GF-REGIONAL-NOTIFY-003

首次启用 dispatcher 不倒灌历史积压

Current catalog: live-non-destructive; manual; NOT_RUN.

- Purpose: contract: First dispatcher enable suppresses historical PENDING backlog, sends fresh grace-window notifications, permits explicit resend and starts dispatch only on worker roles.; catalog: manual, live-non-destructive; historical PASS is not new execution evidence.; order: Phase 9 after NOTIFY-001; formal predecessor NOTIFY-001.
- Runner: scripts/e2e/regional/run_notification_acceptance.py
- Entry Points: main plan can execute notify003_focused_tests; execute reuses passing source-digest result or reruns tests.; run_notify003 -> deployment_notification_config -> Deployment/envFrom ConfigMaps and Ready Pod environment reads.; Runtime: AdvisoryNotificationService._establish_watermark; lifespan.start_notification_worker gate; explicit route delegates to requeue.
- Assertions: Three focused tests pass, declared service roles match, dispatcher/async strings agree across roles, observed Ready Pod env agrees with templates.; No assertion requires dispatcher_enabled or async_delivery to be true.; No assertion requires desired worker replicas >0 or Ready count to equal desired; all([]) treats an absent role's replicas as agreement.; No live backlog injection, suppressed count, fresh-send result, manual requeue API or dispatcher-thread/lease observation is performed.
- Safety And Cleanup: The runner itself does not toggle live dispatcher settings; isolated tests run locally.; Plan is not read-only in behavior because it runs pytest and writes logs; direct --plan has not been invoked in this review.; A shared focused-test reuse digest excludes tests/notifications and untracked files, so changed exercised tests may not invalidate reuse.
- Necessity: The isolated watermark tests are meaningful, but a configuration survey cannot justify the specified live enable/rollout acceptance or a PASS when dispatch is disabled.
- Overlap: case_ids: GF-REGIONAL-NOTIFY-006; GF-REGIONAL-NOTIFY-007; classification: partial role/lease overlap, distinct watermark contract; rationale: 006 enters real lifespan but does not seed historical suppression; 007 drives retry/dead delivery states. None is a superset of first-enable watermark and operator resend.
- Tests: existing: tests/notifications/_notifications_cases_2.py::test_first_dispatch_suppresses_the_pre_enable_backlog (Memory and SQLite); test_suppressed_backlog_can_be_requeued_on_demand; test_notification_worker_reclaims_expired_lease_and_stops; tests/regional/test_notification_acceptance_runner.py::test_notify003_focused_tests_reuse_the_plan_result; missing_behavioral: Disabled-but-consistent config, missing/zero workers, missing Ready pods and mixed-role settings must refuse.; Run isolated deployed-code watermark drill with historical and fresh rows and an observable notifier; bind to exact runtime identity.; Boundary tests at grace cutoff and concurrent first owner/watermark creation on an isolated PostgreSQL instance.; Test source-digest invalidation when tests/notifications or untracked probe/test files change (parent-owned).; execution: Owned fail-closed/configuration tests passed within final269. A live enable/rollout was not performed.
- Findings At Review: id: N003-1; severity: P1; description: A site with all notification flags false, or no worker replicas, can satisfy all current checks and receive PASS.; code: run_notification_acceptance.py::run_notify003/deployment_notification_config; owner: owned runner; id: N003-2; severity: P2; description: Live specification claims enable/rollout/backlog/requeue behavior, but runner combines local tests with env snapshots. Main plan exit status ignores focused-test failure.; code: run_notification_acceptance.py::main/run_notify003; owner: owned runner; parent spec/catalog; id: N003-3; severity: P2; description: Reusable focused-test digest omits the exact tests/notifications sources run by this case and ignores untracked files.; code: live_driver_guard.py::SOURCE_DIGEST_PATHS/source_digest; owner: parent shared helper
- Reviewer Handoff: Owned false-PASS gates fixed: enabled async dispatch and positive complete Ready populations required; failed focused tests invalidate schema3 plans. Parent source_digest now covers exercised and untracked sources. Current scope remains isolated watermark tests plus live configuration survey; no live enable/rollout proof is claimed.

### GF-REGIONAL-NOTIFY-004

GPU executor 无 SES 权限

Current catalog: read-only-signal-replay; manual; NOT_RUN.

- Purpose: contract: A GPU Executor's SES SendEmail attempt must fail with an authorization denial, not message validation or missing-identity errors.; catalog: manual, read-only-signal-replay; historical PASS is not new execution evidence.; order: Phase 9 after NOTIFY-003.
- Runner: scripts/e2e/regional/run_notification_acceptance.py
- Entry Points: run_notify004 -> IdentitySite.any_executor_pod -> pod_json(SES_DENIAL_PROBE) -> boto3 sesv2.send_email
- Assertions: Requires result DENIED and code AccessDenied/AccessDeniedException/UnauthorizedOperation.; MessageRejected, NotFound, successful send, malformed JSON and uncaught transport exceptions do not satisfy checks.; Only one Ready Executor is tested, without proving all replicas use the intended service account/IRSA and release identity.
- Safety And Cleanup: Uses example.invalid sender/recipient so no valid mailbox is intentionally contacted.; Provider errors are reduced to code, not secret-bearing client diagnostics.; This still invokes the SendEmail API, not a read-only signal replay. Keep it behind explicit execute approval.; No persistent row or workload is created; boto3 retry/timeouts are not tied to the authorized deadline.
- Necessity: The negative result distinguishes IAM rejection from meaningless SES validation failures. Coverage should match the per-cluster/per-replica identity boundary, and risk metadata should describe an attempted send.
- Overlap: case_ids: GF-REGIONAL-BLAST-002; GF-REGIONAL-BLAST-003; GF-REGIONAL-BLAST-004; classification: complementary IAM boundary; no duplicate proven; rationale: Related least-privilege tests inspect multiple data-plane policies; this case actually calls SES from an Executor process. Different injection and scope.
- Tests: existing: tests/regional/test_regional_least_privilege_acceptance.py::test_notify004_no_data_plane_identity_can_send_mail; missing_behavioral: Table-driven run_notify004 tests: AccessDenied accepted; ALLOWED, MessageRejected, NotFound, timeout, malformed and missing code rejected.; Replica identity drift and missing Executor refuse before a provider attempt.; Test bounded SDK failure and no credential/error-body leakage into evidence.; execution: Owned per-replica denial classification tests passed within final269 using mocked provider transport.
- Findings At Review: id: N004-1; severity: P2; description: One arbitrary Ready Executor is insufficient to reject a drifted second replica or mismatched IRSA. Static IAM tests are not runtime identity proof.; code: run_notification_acceptance.py::run_notify004; owner: owned runner; parent IdentitySite helper as needed; id: N004-2; severity: P2; description: risk=read-only-signal-replay does not describe a live SendEmail attempt. Provider timeout/retry and window stop behavior lack focused behavioral tests.; code: testcases/fault-scenarios.yaml; SES_DENIAL_PROBE; owner: parent catalog; owned probe
- Reviewer Handoff: Owned runner now attempts the denial probe on each observed Ready Executor, only accepts authorization denial, and bounds SDK connect/read/retry and per-call maintenance deadline. ALLOWED/validation errors/missing codes fail in behavioral tests. Parent owns attempted-provider-action risk wording and full desired-population/IRSA identity coverage.

### GF-REGIONAL-NOTIFY-005

低利用率类邮件按节点或任务聚合

Current catalog: live-non-destructive; manual; NOT_RUN.

- Purpose: contract: One 8-GPU node then three 8-GPU nodes with low-utilization workloads produce at most one notification per node, each selected node is observed, messages aggregate multiple GPUs, latches re-arm between phases, and test workloads are removed.; catalog: manual, live-non-destructive; historical PASS is not new execution evidence.; order: Phase 9 after NOTIFY-004; predecessor for NOTIFY-007 because 006 is pytest-only.
- Runner: scripts/e2e/regional/run_notification_acceptance.py; scripts/e2e/regional/managed_workload_fixture.py (shared, read-only)
- Entry Points: run_notify005 -> foreign_gpu_reservations -> wait_low_utilization_latch_disarmed -> low_utilization_manifest -> ManagedWorkloadFixture.submit; wait_running_pods -> wait_low_utilization_notifications -> count/scope/device assertions -> fixture.delete; Runtime node_health ingestion groups host resource findings by cluster/node/metric/workloads and aggregates device identifiers before notification generation.
- Assertions: Checks independent Master/Worker metadata, <=1 and <=3 notifications, per-phase node coverage, exactly one matched node per message and >1 GPU UUID.; Does not require SENT or provider_message_id although the query returns both.; Stops sampling once every node has any notification, so later duplicate growth and actual dispatch completion are not observed.; Notification query uses substring matches in text and no cluster binding; no live proof of eight GPUs per node.
- Safety And Cleanup: Workloads reserve one GPU and perform CPU work, restart budget 0; no real reset requested.; Foreign reservation guard ignores terminating/non-Running pods and checks only regular container requests.; No explicit Node UID/Ready/cordon/inventory preflight is performed before single-node nodeName placement.; Shared ManagedWorkloadFixture.submit begins with label/name deletion; delete ignores nonzero command returns and has no UID preconditions.; Wait-for-Pod-removal timeout is not an assertion. Finally calls delete again but does not prove absence, terminal observations, re-armed latches or cleared alerts.; If the phase throws, cleanup_errors added to a local result are lost when main catches the original exception.; Maintenance deadline is not passed to either long phase; a failed first phase is only judged after the second phase runs.
- Necessity: The positive node-coverage and latch polling prevent vacuous zero-notification success. Missing cluster binding, send-state checks and verifiable cleanup still permit misleading acceptance and resource residue.
- Overlap: case_ids: GF-REGIONAL-COLLECT-016; GF-REGIONAL-NOTIFY-001; classification: shared workload/notification machinery, distinct aggregation/latch contract; rationale: 001 checks completion email dedup; collector cases check event detection or lifecycle. Neither provides the one-node/three-node device aggregation and latch reset scope.
- Tests: existing: tests/regional/test_notification_acceptance_runner.py: independent templates, YAML alias rejection, injected role/rank metadata; missing_behavioral: Run both phases through fakes: per-node events, SENT convergence, late duplicates, silent nodes and latch timeout.; Reject same-name nodes from another cluster and substring collisions; bind GPU sets to validated eight-device inventories.; Test terminating/init-container/limits-only reservations and wrong Node UID/cordon/Ready state.; Stop before phase two after phase-one failure or expired maintenance window.; Delete failure, still-present workload/Pods, cleanup after injection exception and UID-replacement refusal must be visible in final FAIL evidence.; execution: Owned two-phase, late-duplicate and cleanup failure tests passed within final269 using fake managed workloads.
- Findings At Review: id: N005-1; severity: P1; description: Can pass with all low-utilization notifications unsent, or with wrong-cluster/same-substring node matches; can miss later duplicate messages.; code: NOTIFICATION_QUERY_PROBE; wait_low_utilization_notifications; run_notify005 checks; owner: owned runner/probe; id: N005-2; severity: P1; description: Shared delete silently ignores failure and lacks UID/ownership preconditions; runner has no final absence/terminality proof and loses cleanup diagnostics on a phase exception.; code: managed_workload_fixture.py::ManagedWorkloadFixture.delete/submit; run_notify005; owner: parent shared fixture plus owned runner; id: N005-3; severity: P1; description: No node identity/scheduling/inventory gate and no ongoing maintenance deadline; phase-two mutation proceeds before phase-one verdict is evaluated.; code: run_notification_acceptance.py::main/run_notify005/foreign_gpu_reservations; owner: owned runner; parent shared preflight integration
- Reviewer Handoff: Owned runner checks node UID/Ready/8-GPU baseline, cluster-bound SENT/provider records and exact node matching, observes late duplicates, refuses phase two after a failed phase, and retains cleanup errors plus absence checks. Shared ManagedWorkloadFixture identity remains with its owner; no shared fallback deletion added.

### GF-REGIONAL-NOTIFY-006

通知 dispatcher 的 service-role 装配与重启接管

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: contract: Only worker lifespan starts notification dispatch; one notification is sent once, a replacement owner reclaims after expiry, shutdown leaves no threads and throttled replicas have independent retry schedules.; catalog: pytest, non-destructive; only one lifespan nodeid is mapped.; order: Phase 9 between 005 and 007, skipped when resolving formal predecessor because pytest writes no per-case JSON.
- Runner: tests/notifications/test_notifications.py; tests/notifications/_notifications_cases_2.py; src/gpu_fault/app/lifespan.py (runtime, read-only); src/gpu_fault/app/lifespan_workers.py (runtime, read-only)
- Entry Points: Catalog pytest_nodeid -> test_only_the_worker_role_dispatches_the_outbox_through_the_lifespan; create_app -> real lifespan -> start_notification_worker -> AdvisoryNotificationService.dispatch_outbox
- Assertions: The mapped test shares one Store and PENDING notification across ingress, spool-worker and worker lifespans with all flags enabled.; Asserts no dispatcher/send for ingress/spool, at least one for worker, exactly one send, SENT result and thread count returns to baseline.; Separate tests cover manually expired Memory lease takeover and jittered throttle waits; neither is in the catalog nodeid.; No mapped test terminates an actual throttled worker then starts a replacement owner.
- Safety And Cleanup: Uses RecordingNotifier and isolated in-process Store; no SES or cluster action.; Existing real lifespan test verifies dispatcher shutdown and has a useful thread-baseline assertion.; No production Store guarantee should be inferred from Memory-only lease manipulation.
- Necessity: Strong test for role assembly and thread lifecycle. Catalog scope overstates what its single mapped nodeid executes.
- Overlap: case_ids: GF-REGIONAL-NOTIFY-003; GF-REGIONAL-NOTIFY-007; classification: complementary lifecycle coverage; not redundant; rationale: 003 checks historical watermark, 007 checks terminal delivery states. 006 owns runtime lifespan, thread lifecycle and owner takeover.
- Tests: existing: test_only_the_worker_role_dispatches_the_outbox_through_the_lifespan; test_notification_worker_reclaims_expired_lease_and_stops; test_throttled_replicas_back_off_on_independent_schedules; missing_behavioral: One isolated full-lifespan scenario: provider throttles, owner stops, another owner takes over same row, old lease completion is refused and notifier sends once.; Deterministic random injection for jitter sequences rather than probabilistic inequality alone.; Real PostgreSQL lease-expiry/fencing coverage through the dispatcher after parent provides an isolated database.; Parent catalog mapping must execute all behaviors it claims, without silently changing pytest-wrapper evidence semantics.; execution: All three existing lifespan, expired-lease takeover and jitter tests explicitly passed in the final269 selection. No PostgreSQL or LIVE process acceptance performed.
- Findings At Review: id: N006-1; severity: P2; description: Catalog selects only role/lifespan test but expected includes restart takeover and throttling; separate supporting tests are not part of formal case execution.; code: testcases/fault-scenarios.yaml GF-REGIONAL-NOTIFY-006; owner: parent catalog/tools; owned notification tests; id: N006-2; severity: P2; description: Existing takeover edits an expired Memory lease directly, not a stopped throttled lifespan on a durable Store. Provider acceptance-before-result-write ambiguity is not an exactly-once guarantee.; code: tests/notifications/_notifications_cases_2.py; owner: owned notification tests; parent isolated PostgreSQL
- Reviewer Handoff: No separate owned live runner. Existing lifespan, lease-takeover and jitter tests are included together in final focused validation. Parent should map all three into the existing case instead of treating one role test as their superset. Durable crash/fencing is a separate proposed extension.
- Current Disposition: Catalog now selects lifespan, lease takeover and jitter tests together; no provider acceptance/commit crash guarantee.

### GF-REGIONAL-NOTIFY-007

通知投递状态：瞬态失败是 RETRY 而非 FAILED，耗尽重试才是 DEAD 与终态 FAILED

Current catalog: live-non-destructive; manual; NOT_RUN.

- Purpose: contract: Isolated deployed outbox drill: transient failure leaves RETRY without result, retry yields SENT with provider id, exhaustion yields DEAD plus FAILED. Production metrics/families must exist and FAILED/DEAD must not increase.; catalog: manual, live-non-destructive; historical PASS is not new execution evidence.; order: Phase 9 last entry; actual predecessor is NOTIFY-005, not pytest-only 006.
- Runner: scripts/e2e/regional/run_notify007_delivery_states.py; scripts/e2e/regional/notify007_verdicts.py; scripts/e2e/regional/probes/notify007_delivery_drill.py
- Entry Points: main -> predecessor_path -> authorize_execution -> execute; execute -> control_plane_metrics -> run_drill -> retry/dead/exported/production/alert verdict functions; Probe -> InMemoryStore -> real AdvisoryNotificationService.dispatch_outbox; first failure wrapper then SES; permanent wrapper with max_attempts=1
- Assertions: RETRY=1, no SENT/DEAD or result after first failure; SENT=1, RETRY=0 and provider id after second cycle; wrapper attempts=2 and cycle timestamp >0.; Permanent failure requires DEAD=1, FAILED result, dead_lettered_total=1 and wrapper attempts=1.; Metrics families/statuses are unioned across snapshots; production FAILED/DEAD compare global maxima only when both samples exist.; Rule YAML is parsed but PromQL and runbook anchoring use string checks; no AMP evaluation is claimed.
- Safety And Cleanup: One labelled email and an isolated Memory Store, no production row mutation.; Ready worker and ingress metrics are read, but spool role and per-replica identity/completeness are omitted.; main neither binds predecessor release/cluster nor emits current evidence_identity, unlike NOTIFY-001..005.; No fresh runtime identity proof before sending, and authorization deadline is not propagated.
- Necessity: The deployed-code Memory drill honestly isolates production data and exercises real delivery logic. Missing observations can still be accepted, and it cannot validate PostgreSQL-specific delivery semantics.
- Overlap: case_ids: GF-REGIONAL-NOTIFY-001; GF-REGIONAL-NOTIFY-003; GF-REGIONAL-NOTIFY-006; classification: complementary async state contract; rationale: Its fault injection is retryable/permanent notifier failure with outbox transitions, not cached synchronous send, historical suppression or lifespan shutdown. No superset duplication.
- Tests: existing: tests/regional/test_notify007_delivery_states.py: executes real Memory drill with RecordingNotifier and fake sleep; tests RETRY/DEAD and malformed result cases, metrics and rule helper checks; tests/notifications/test_delivery_state.py; tests/store/test_postgres_notification_delivery_lock.py; missing_behavioral: Missing before/after FAILED or DEAD series, missing replica, identity changes, nonfinite metric values and an increase hidden by another replica's maximum must refuse.; Require all nonselected delivery statuses/counts to be zero and validate phase/notification identity.; Test main predecessor identity/output identity and no send on runtime drift or expired window.; Exercise RETRY/DEAD contract on isolated PostgreSQL, not only Memory.; execution: Owned metric/verdict/schema3/identity tests passed within final269; no production delivery-state drill executed.
- Findings At Review: id: N007-1; severity: P1; description: production_untouched_errors silently accepts absent samples; max-over-replicas can hide one replica's increase, and exported_family_errors accepts partial coverage across replicas.; code: notify007_verdicts.py::production_untouched_errors/exported_family_errors; run_notify007_delivery_states.py::control_plane_metrics; owner: owned runner/verdicts; id: N007-2; severity: P1; description: Predecessor is not bound to release/cluster and result omits identity; a wrong-site predecessor may be accepted and the next identity-checking case cannot trust this evidence.; code: run_notify007_delivery_states.py::main; owner: owned runner using parent helper; id: N007-3; severity: P2; description: PREDECESSOR_CASE_ID constant/tests still say 006 although actual contract correctly resolves 005; phase checks do not enforce complete mutually exclusive status totals.; code: notify007_verdicts.py; tests/regional/test_notify007_delivery_states.py; owner: owned helper/tests
- Reviewer Handoff: Owned fixes implemented: complete per-replica metric families and growth checks, exclusive delivery-state totals, schema3 genuine preflight result, formal NOTIFY005 predecessor and release/cluster result binding. Empty/partial evidence cannot pass. Production delivery and isolated PostgreSQL validation remain unexecuted.

### GF-REGIONAL-PREEMPT-001

抢占开关关闭时保持原串行语义

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: Disabled preemption keeps RUNNING reset serial and queues a nonpreempting stronger successor.
- Runner: scripts/e2e/regional/run_preemption_contracts.py; tests/execution/_executor_cases_1.py
- Entry Points: catalog command -> run_case -> pytest test_preemption_config_disabled_keeps_predecessor_running; InMemoryStore/FakeAdapter only
- Assertions: Selected test disables ProductionExecutorConfig preemption and asserts predecessor remains RUNNING and its FREEZE_EVIDENCE adapter runs
- Safety And Cleanup: No live clients or cluster actions; subprocess/evidence only
- Necessity: Valid executor switch regression but does not construct the documented reset/reboot event sequence or prove planner successor.preempt_predecessor=false.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-004; classification: distinct_gate; rationale: Configuration disable vs strict recovery-rank eligibility.
- Tests: tests/execution/test_executor.py::test_preemption_config_disabled_keeps_predecessor_running; tests/regional/test_preemption_contracts_evidence.py
- Findings At Review: No planner-disabled event arrival or serial successor completion in selected nodeid; Wrapper uses exit0 only, with no collected/passed/skip proof; local JSON must not be treated as live proof
- Reviewer Handoff: reviewed_local_contract; coverage_selection_proposal_for_parent

### GF-REGIONAL-PREEMPT-002

同 attempt clean boundary 抢占与步骤继承

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: Same-attempt clean boundary reuses MARK/STOP inside one DAG and supersedes unstarted lower branch; catalog still describes old cross-record successor.
- Runner: scripts/e2e/regional/run_preemption_contracts.py; tests/orchestration/_cross_fault_arbitration_cases_2.py; tests/execution/_executor_cases_1.py
- Entry Points: run_case -> two pytest nodeids; ASGI synthetic fault ingestion plus independent executor boundary fixture
- Assertions: Same workflow ID, completed containment preserved, reset index superseded, one STOP and PREEMPTION event; Separate seeded cross-record predecessor becomes SUPERSEDED without adapter call
- Safety And Cleanup: Memory contexts/fake adapters only
- Necessity: Both modern branch merge and retained cross-record boundary are useful, but combining them does not prove the old catalog inheritance assertion.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-012; GF-REGIONAL-PREEMPT-022; classification: partial_overlap_complementary; rationale: Clean branch preemption vs independent real-quiesce fixture and multi-node branch-local rank.
- Tests: tests/orchestration/test_cross_fault_arbitration.py::test_preemption_marks_stronger_successor_and_reuses_containment; tests/execution/test_executor.py::test_executor_supersedes_at_clean_step_boundary
- Findings At Review: CENTRAL-001 catalog/spec mismatch; Selected tests do not execute successor branch to one final join or prove all noncontainment operations cannot inherit
- Reviewer Handoff: reviewed; parent_catalog_alignment_required

### GF-REGIONAL-PREEMPT-003

无 job 的同节点抢占与跨节点隔离

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: Idle same-node stronger action preempts; disjoint node, mixed job/no-job and distinct jobs do not.
- Runner: scripts/e2e/regional/run_preemption_contracts.py; tests/orchestration/test_merge.py
- Entry Points: run_case -> idle-node ingestion test and four scope-boundary parameter variants
- Assertions: Reboot has predecessor/reset flag and inherits MARK; Four preemption_scope_matches outcomes
- Safety And Cleanup: Synthetic incidents, InMemoryStore, no live operations
- Necessity: Reasonable focused scope regression; event ingestion and pure predicate controls complement each other.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-016; GF-REGIONAL-PREEMPT-028; classification: partial_overlap_complementary; rationale: Pairwise preemption scope vs independent node dispatch and abnormal attempt ownership.
- Tests: tests/orchestration/test_merge.py::test_idle_node_stronger_action_requests_safe_preemption; tests/orchestration/test_merge.py::test_preemption_scope_boundaries
- Findings At Review: Scope predicate test lacks cross-cluster and same-job/different-attempt variants; these need call-path-bound isolation tests, not just helper assumptions
- Reviewer Handoff: reviewed_local_contract; no_runner_specific_fix_required

### GF-REGIONAL-PREEMPT-004

recovery rank 门禁只允许严格更高级抢占

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: Only strictly higher recovery rank can preempt.
- Runner: scripts/e2e/regional/run_preemption_contracts.py; tests/orchestration/test_merge.py
- Entry Points: run_case -> WorkflowMergeService.prepare_preempting_successor with lower/equal/higher candidates
- Assertions: Lower/equal flag false; higher true and rank30->50 reason
- Safety And Cleanup: Pure models and merger, no Store or live I/O
- Necessity: Correct rank gate test; no need for a destructive live duplicate.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-001; GF-REGIONAL-PREEMPT-022; GF-REGIONAL-PREEMPT-024; classification: distinct_arbitration_axes; rationale: Enabled flag, local rank and domain dominance are separate gates.
- Tests: tests/orchestration/test_merge.py::test_preemption_requires_strictly_higher_recovery_rank; tests/orchestration/test_preemption_guards.py
- Findings At Review: Selected nodeid does not cover candidate non-PENDING status or candidate with no node-exclusive claims; existing guard suite is not selected
- Reviewer Handoff: reviewed_local_contract; coverage_selection_proposal

### GF-REGIONAL-PREEMPT-005

未被领取的 remote command 在抢占前原子取消

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: PENDING remote command must be cancelled atomically before preemption and never claimable afterwards.
- Runner: scripts/e2e/regional/run_preemption_contracts.py; tests/execution/_node_action_cases_1.py
- Entry Points: run_case -> real InMemoryStore command creation and ProductionWorkflowExecutor with fake adapter
- Assertions: Predecessor SUPERSEDED; command FAILED/workflow-preempted; all lease fields null; subsequent claim empty; no adapter call
- Safety And Cleanup: Memory-only command/incident/workflow, no data-plane claimant
- Necessity: Selected assertions match deterministic protocol. Single-thread Memory does not prove database cancellation-vs-claim atomicity.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-009; classification: shared_transition_distinct_initial_state; rationale: Unclaimed PENDING vs classified remote WAITING/LEASED behavior.
- Tests: tests/execution/test_node_action.py::test_unclaimed_remote_command_is_cancelled_before_preemption; tests/execution/test_preemption_cancel_live_check.py; tests/store/test_remote_command_claim_cancellation.py
- Findings At Review: Race/later-claim tests exist outside selected wrapper; real PostgreSQL concurrency not executed in this review
- Reviewer Handoff: reviewed_local_contract; no_runtime_or_DDL_change

### GF-REGIONAL-PREEMPT-006

已提交 reset 被 reboot 抢占时只执行一次

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: Reset already LEASED when reboot arrives finishes once, compensates RESTORE, cuts lower tail and executes successor once.
- Runner: scripts/e2e/regional/run_preemption_contracts.py; tests/execution/_misc_cases_1.py
- Entry Points: run_case -> test_submitted_reset_restores_before_reboot_preemption
- Assertions: Test seeds reset as already SUCCEEDED and RESTORE as remote LEASED; validates compensation, skipped validation, successor reboot call once
- Safety And Cleanup: FakeAdapter and InMemoryStore; no NodeAction ledger or provider invocation
- Necessity: Useful post-reset handoff test but starts after the documented critical reset-in-flight boundary.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-008; GF-REGIONAL-PREEMPT-009; GF-REGIONAL-DESTR-016; classification: complementary_boundary_states; rationale: After reset vs before reset vs generic in-flight hold; DESTR016 is the actual reset/reboot proof.
- Tests: tests/execution/test_misc.py::test_submitted_reset_restores_before_reboot_preemption; tests/execution/test_preemption_boundary.py; tests/execution/test_preemption_cancel_live_check.py
- Findings At Review: Selected wrapper does not register higher action while RESET_GPU is LEASED; Physical exactly-once deliberately belongs to DESTR016, not this PASS
- Reviewer Handoff: reviewed; concrete_missing_scenario_proposal

### GF-REGIONAL-PREEMPT-007

已提交 reboot 被 replace 抢占时不并发调用 provider

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: Warm-spare replacement must wait for submitted reboot and never invoke provider replace.
- Runner: scripts/e2e/regional/run_preemption_contracts.py; tests/execution/_misc_cases_1.py
- Entry Points: run_case -> test_submitted_reboot_finishes_before_replace_successor; fake lifecycle outcomes
- Assertions: Reboot first WAITING then success; predecessor SUPERSEDED without validation; replacement called after test manually completes reboot; submission row unchanged
- Safety And Cleanup: All provider and replacement operations are fake; no AWS or actual spare
- Necessity: Tests terminal handoff order but never attempts the successor while reboot is pending, so it cannot catch premature successor admission.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-006; GF-REGIONAL-DESTR-003; classification: different_physical_boundary; rationale: Provider lifecycle unresolved vs reset handoff; actual spare success is separate.
- Tests: tests/execution/test_misc.py::test_submitted_reboot_finishes_before_replace_successor
- Findings At Review: Must try dispatching successor before reboot terminal and assert no replacement call; Fake operation_id warm-spare/simulated is not provider-negative or spare-availability proof
- Reviewer Handoff: reviewed; concrete_missing_negative_control_proposal

### GF-REGIONAL-PREEMPT-008

QUIESCE 后抢占先补偿 RESTORE 再跳过低级 reset

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: After QUIESCE but before reset, hand off safely or restore before supersession; no low reset.
- Runner: scripts/e2e/regional/run_preemption_contracts.py; tests/execution/_misc_cases_1.py; tests/execution/_executor_cases_1.py
- Entry Points: run_case -> two fake-adapter executor tests for safe handoff and nonhandoff compensation
- Assertions: No low reset; handoff marker/dependent RESTORE appended; nonhandoff path calls only RESTORE before SUPERSEDED
- Safety And Cleanup: No actual service stop/timer/state file in these tests
- Necessity: Useful two-way state-machine coverage, explicitly not physical service recovery.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-012; GF-REGIONAL-PREEMPT-006; classification: different_layer_or_boundary; rationale: PREEMPT012 separately runs real quiesce;006 tests already completed reset.
- Tests: tests/execution/test_misc.py::test_reset_to_reboot_hands_off_quiesce_and_skips_low_reset; tests/execution/test_executor.py::test_executor_restores_when_successor_cannot_take_quiesce_handoff; tests/orchestration/test_quiesce_handoff_on_preemption.py
- Findings At Review: Selected safe handoff stops at reboot WAITING, not final cleanup; PENDING reset cancellation is not included in the quiesced seed; state/timer/service checks absent by construction
- Reviewer Handoff: reviewed_local_contract; parent_proof_layer_alignment

### GF-REGIONAL-PREEMPT-009

四类 WAITING 的可抢占性分类

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: Classify local validation, readonly remote WAITING, destructive LEASED, and workload restart WAITING preemption.
- Runner: scripts/e2e/regional/run_preemption_contracts.py; tests/execution/test_validation.py; tests/execution/_node_action_cases_1.py
- Entry Points: run_case -> four pytest nodes; ProductionWorkflowExecutor with Memory and fake adapters
- Assertions: Local validation supersedes without call; VERIFY WAITING cancelled; RESTART_NODE LEASED remains RUNNING; RESTART_WORKLOAD WAITING remains RUNNING and command preserved
- Safety And Cleanup: Memory-only leases and synthetic restart reservation
- Necessity: Four separate state classes, not four duplicate scenarios.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-005; GF-REGIONAL-PREEMPT-006; GF-REGIONAL-PREEMPT-007; classification: shared_rule_complementary_transitions; rationale: Classification vs cancellation atomicity and terminal handoff sequences.
- Tests: tests/execution/test_validation.py::test_local_validation_waiting_is_safe_to_preempt; tests/execution/test_node_action.py::test_safe_remote_waiting_is_cancelled_before_preemption; tests/execution/test_node_action.py::test_remote_waiting_step_is_not_preempted; tests/execution/test_node_action.py::test_restart_workload_remote_waiting_is_not_preempted
- Findings At Review: Readonly selection only VERIFY, not CHECK_MECHANICALS/diagnostics matrix; Unknown/malformed remote state and concurrent sibling claim tests outside wrapper
- Reviewer Handoff: reviewed_local_contract; additional_negative_selection_proposal

### GF-REGIONAL-PREEMPT-010

前驱终态回写不得覆盖 incident 的 successor 指针

Current catalog: non-destructive; pytest; NOT_RUN.

- Purpose: Predecessor SUCCEEDED/FAILED/SUPERSEDED save cannot overwrite successor incident pointer or fence.
- Runner: tests/execution/_misc_cases_1.py
- Entry Points: catalog automation pytest -> test_predecessor_terminal_save_preserves_successor_incident_pointer[3 variants]
- Assertions: Requested predecessor terminal state, successor request_id and fencing preserved, incident ACTION_PENDING
- Safety And Cleanup: Memory-only; no standalone live runner or standard case evidence
- Necessity: Correct focused contract for three terminal states.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-014; classification: partial_overlap_complementary; rationale: Chain checks final pointer only; this adds success/failure/supersession save matrix.
- Tests: tests/execution/test_misc.py::test_predecessor_terminal_save_preserves_successor_incident_pointer
- Findings At Review: No concurrent PostgreSQL save/failover exercised by this selected nodeid
- Reviewer Handoff: reviewed_local_contract; no_live_claim

### GF-REGIONAL-PREEMPT-011

多副本竞争、租约与 Aurora failover 下的抢占唯一性

Current catalog: non-destructive; pytest; NOT_RUN.

- Purpose: Lease/epoch fencing preserves preemption and exactly-once successor across control replica takeover and Aurora failover.
- Runner: tests/execution/_executor_cases_2.py
- Entry Points: catalog pytest -> test_preemption_survives_store_failover_and_lease_takeover
- Assertions: Fake psycopg.OperationalError after A lease; dispatcher waiting not failed; synthetic expiry; B epoch advances; stale A write raises; successor once
- Safety And Cleanup: InMemoryStore, fake adapter, controlled exception; no database/network/provider action
- Necessity: Good deterministic failover recovery regression, but not actual Aurora or concurrent two-process validation.
- Overlap: case_ids: GF-REGIONAL-HA-003; GF-REGIONAL-HA-010; GF-REGIONAL-PREEMPT-010; classification: different_layer; rationale: Simulated store interruption vs real HA cases and terminal-pointer save.
- Tests: tests/execution/test_executor.py::test_preemption_survives_store_failover_and_lease_takeover
- Findings At Review: No actual PostgreSQL/Aurora test in the case; No per-case JSON produced; cannot be PREEMPT012 hardcoded evidence predecessor
- Reviewer Handoff: reviewed; parent_evidence_and_wording_proposal

### GF-REGIONAL-PREEMPT-012

clean 与 dirty 两种边界的真机抢占验收

Current catalog: destructive; manual; NOT_RUN.

- Purpose: Live clean and post-QUIESCE preemption groups with no low physical action and fully restored node/command/workflow state.
- Runner: scripts/e2e/regional/run_preempt012_acceptance.py; scripts/e2e/regional/probes/preempt012_node_probe.py
- Entry Points: manual; main -> predecessor/plan/authorization -> real host quiesce cycle plus separate in-Pod CONTROL_AUDIT with fake adapters
- Assertions: Synthetic clean/dirty supersession and inheritance; audit time lies inside real quiesce; no audit remote command/provider mutation; services/GPU count/node readiness restored
- Safety And Cleanup: 180s host failsafe,45s quiesce hold, single-attempt control script, host cleanup in finally; Unknown timer originally accepted; fixed and regression tested; Control audit creates event links but only deletes objects; partial seed tracking after all writes
- Necessity: Independently timed real quiesce and synthetic preemption do not prove causal workflow-to-NodeAction handoff or real ledger zero. Current limitations partly disclose this, but full spec proof remains missing.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-002; GF-REGIONAL-PREEMPT-008; GF-REGIONAL-DESTR-016; classification: partial_overlap_missing_integration; rationale: Reuses local cross-record semantics; DESTR016 owns submitted-reset boundary. A causal pre-reset live handoff is not currently implemented.
- Tests: tests/regional/test_preempt012_acceptance_runner.py; tests/regional/test_protocol_fail_closed_review.py::test_quiesce_cleanup_does_not_accept_an_unknown_timer
- Findings At Review: PROTO-010 predecessor hardcode; No NodeAction ledger before/after or actual successor service-restore causality; Empty final_nodes/services and unknown cleanup state originally pass; Host arm still uses legacy /opt/gpu-fault/venv/bin/python rather than current slot
- Reviewer Handoff: fixed_owned_predecessor_timer_control_ACK_cleanup_and_required_host_state_directory; causal_NodeAction_limit_retained

### GF-REGIONAL-PREEMPT-013

三级连环抢占只让最高级执行

Current catalog: non-destructive; manual; SUPERSEDED.

- Purpose: Retired three-level chain, superseded by PREEMPT014 five-level/two-arrival-order coverage.
- Runner: tests/execution/_executor_cases_1.py; tests/execution/_support.py
- Entry Points: catalog SUPERSEDED/manual; formal do_not_run; only retained regression nodeid [3]
- Assertions: Retained test asserts chain predecessors, lower supersession/no calls, highest once, incident pointer
- Safety And Cleanup: No standalone execution authorized; keep ID retired
- Necessity: Retirement correct; three-level chain is a strict scenario subset, though retained regression remains useful.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-014; classification: retired_strict_subset; rationale: Five-level test uses same chain/verification helpers and both timing variants.
- Tests: tests/execution/test_executor.py::test_dispatcher_resolves_chained_preemptions_to_highest_action[3]
- Findings At Review: None requiring a new live runner; do not re-enable to inflate case count
- Reviewer Handoff: retired_preserved

### GF-REGIONAL-PREEMPT-014

五级连环抢占与两种到达时序等价

Current catalog: non-destructive; pytest; NOT_RUN.

- Purpose: Five-level strictly stronger chain converges to highest only under preregistered and incremental arrival.
- Runner: tests/execution/_executor_cases_1.py; tests/execution/_support.py
- Entry Points: catalog pytest -> test_five_level_preemption_is_arrival_order_independent -> _run_chained_preemptions twice
- Assertions: First4 SUPERSEDED/no step executions; highest SUCCEEDED/called once; predecessor chain intact; incident final pointer/fence; equal outcome/calls across timing
- Safety And Cleanup: Memory and FakeAdapter; no physical mutation
- Necessity: Appropriate deterministic superset of013. Incremental shape is seeded/reset directly, not real event arrival after observed RUNNING.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-013; GF-REGIONAL-PREEMPT-010; classification: superset_plus_shared_invariant; rationale: Replaces013; overlaps final pointer with010 but not its full terminal-state matrix.
- Tests: tests/execution/test_executor.py::test_five_level_preemption_is_arrival_order_independent
- Findings At Review: No event-ingestion arbitration through all five ranks; Incremental helper manually saves successor RUNNING after ticks; does not prove runtime arrival observability
- Reviewer Handoff: reviewed_local_contract; no_new_live_case_required_without_parent_design

### GF-REGIONAL-PREEMPT-015

高级 workflow 运行后到达的低级事件被吸收不降级

Current catalog: non-destructive; pytest; NOT_RUN.

- Purpose: Weaker late event and duplicate cannot downgrade or undo active higher-action preemption.
- Runner: tests/orchestration/_cross_fault_arbitration_cases_1.py
- Entry Points: catalog pytest -> ASGI XID48/SXID/high action then weaker XID11 and duplicate
- Assertions: Same incident/workflow, lower reset stays retired, node-action branch list unchanged, STOP/RESTART each1, actions/pointer unchanged, duplicate true
- Safety And Cleanup: Memory context and local ASGI; no executor action
- Necessity: Strong branch-local regression, but catalog language still asks preserving nonempty predecessor/inheritance chain that this seed never has.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-002; GF-REGIONAL-PREEMPT-022; classification: distinct_arrival_sequence; rationale: Later weaker event must be absorbed after prior upgrade, not merely allow the upgrade.
- Tests: tests/orchestration/test_cross_fault_arbitration.py::test_later_weaker_event_does_not_reverse_existing_preemption
- Findings At Review: Separate nonempty cross-record preemption metadata preservation not covered by current selected nodeid
- Reviewer Handoff: reviewed_local_contract; parent_wording_alignment

### GF-REGIONAL-PREEMPT-016

不同节点与 warm spare 并发时不得互相抢占

Current catalog: non-destructive; pytest; NOT_RUN.

- Purpose: Disjoint no-job nodes dispatch independently; shared-attempt nodes form one DAG and single STOP/restart.
- Runner: tests/orchestration/_cross_fault_arbitration_cases_1.py
- Entry Points: catalog pytest -> independent ASGI contexts then shared attempt ingestion
- Assertions: No predecessor/preemption for independent nodes, distinct incident/workflow/lease owners; Same-attempt DAG has one STOP/restart, at most checkpoint1, per-node restores and join dependencies
- Safety And Cleanup: InMemoryStore and synthetic observations only
- Necessity: Good paired grouping/independence test; no actual warm-spare reservation or concurrent executor threads.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-003; GF-REGIONAL-PREEMPT-017; classification: partial_overlap_complementary; rationale: Scope predicate and executor fanout complement planner grouping here.
- Tests: tests/orchestration/test_cross_fault_arbitration.py::test_disjoint_node_scope_uses_independent_workflows_or_shared_dag
- Findings At Review: Title mentions warm spare but seed is generic node-b reboot; Lease claims are sequential; node ownership side effects not exercised
- Reviewer Handoff: reviewed_local_contract

### GF-REGIONAL-PREEMPT-017

同 job 双节点并行 branch 与单次 join

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: Same-job two branches both dispatch in one cycle despite WAITING and join/restart once.
- Runner: tests/execution/_executor_cases_1.py
- Entry Points: catalog command is python3 -m pytest wrapper, not live runner
- Assertions: FakeAdapter receives reset/reboot on first execute; both WAITING; join absent; second execute completes both and restart once
- Safety And Cleanup: Memory, precompleted STOP, no remote queue or node action
- Necessity: Tests executor DAG fanout, but fake operation IDs remote/reset-a and remote/reboot-b do not create remote commands as spec claims.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-016; GF-REGIONAL-PREEMPT-018; GF-REGIONAL-DESTR-015; classification: planner_executor_live_layers; rationale: Planner grouping, dynamic branch append and physical two-node recovery are distinct.
- Tests: tests/execution/test_executor.py::test_dag_fans_out_node_branches_and_joins_once
- Findings At Review: No RegionalRemoteWorkflowAdapter/actual command rows; Checkpoint and per-node RESTORE_SCHEDULING absent; STOP seeded completed rather than executed
- Reviewer Handoff: reviewed; missing_remote_fanout_contract_proposal

### GF-REGIONAL-PREEMPT-018

3 节点与 5 节点动态追加 branch

Current catalog: non-destructive; pytest; NOT_RUN.

- Purpose: Append nodes2..5 during RUNNING, preserve completed step indices, revise DAG monotonically and keep unique join.
- Runner: tests/orchestration/test_merge.py
- Entry Points: catalog pytest -> DagBrancher.append_parallel_job_branch
- Assertions: Revisions1..4; completed STOP/execution unchanged; unique branch IDs; restart dependency count3/5; duplicate node event unchanged
- Safety And Cleanup: Pure models and brancher, no persistence or live actions
- Necessity: Correct deterministic branch compilation test.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-017; GF-REGIONAL-PREEMPT-021; classification: different_growth_dimension; rationale: New nodes vs fixed two-branch execution vs second GPU on same node.
- Tests: tests/orchestration/test_merge.py::test_parallel_job_dag_accepts_three_and_five_node_branches
- Findings At Review: No Store CAS/lease concurrency during append or full dispatcher continuation
- Reviewer Handoff: reviewed_local_contract

### GF-REGIONAL-PREEMPT-019

branch 失败时 join 门禁不得放行恢复

Current catalog: non-destructive; pytest; NOT_RUN.

- Purpose: Failed branch does not prevent sibling settlement, but blocks join/readmission/restart.
- Runner: tests/execution/_executor_cases_1.py
- Entry Points: catalog pytest -> fake reset failure and reboot success in one DAG executor call
- Assertions: Both branch adapters called; restart never called; workflow FAILED, incident ESCALATED; rerun adds no calls
- Safety And Cleanup: Memory and synchronous FakeAdapter only
- Necessity: Correct join gating at executor layer; does not cover outstanding remote sibling convergence after failure.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-017; GF-REGIONAL-PREEMPT-026; classification: distinct_terminal_gate; rationale: Branch failure vs ordinary fanout and final quarantine suppressing restart.
- Tests: tests/execution/test_executor.py::test_dag_branch_failure_still_dispatches_other_ready_branch
- Findings At Review: No RESTORE_SCHEDULING step to prove readmission is withheld; No remote LEASED/WAITING sibling or physical idempotency key settlement
- Reviewer Handoff: reviewed_local_contract; coverage_gap_recorded

### GF-REGIONAL-PREEMPT-020

finalization 开始后迟到的节点另起 workflow

Current catalog: non-destructive; pytest; NOT_RUN.

- Purpose: Late new node after restart WAITING/SUCCEEDED cannot modify original DAG; queue independent successor.
- Runner: tests/orchestration/_cross_fault_arbitration_cases_2.py
- Entry Points: catalog pytest -> two parametrized restart states with ASGI late node event
- Assertions: Original complete model unchanged; new workflow ID/predecessor; no preemption; hardware scope exactly late node
- Safety And Cleanup: InMemoryStore and local API, no workload mutation
- Necessity: Strong planner immutability test for both documented boundary states.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-026; classification: different_late_event; rationale: Late node recovery vs late final quarantine after restart.
- Tests: tests/orchestration/test_cross_fault_arbitration.py::test_late_node_after_restart_finalization_uses_new_workflow
- Findings At Review: Does not dispatch successor while old restart is pending to prove serial execution gate
- Reviewer Handoff: reviewed_local_contract

### GF-REGIONAL-PREEMPT-021

同节点同级不同 GPU 合并为一个 reset step

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: Same-node equal-rank second GPU widens unsubmitted reset, otherwise appends protected successor branch.
- Runner: tests/orchestration/_cross_fault_arbitration_cases_2.py
- Entry Points: catalog command is python3 -m pytest with two nodeids, three parameter variants
- Assertions: Single reset with both GPUs before submission, including completed QUIESCE; After WAITING execution, old step/execution untouched, second GPU reset appended and dependent; one STOP/restart
- Safety And Cleanup: Synthetic ASGI/Memory, no actual remote command or GPU reset
- Necessity: Useful precise submitted/unsubmitted distinction; shared helper use is not duplicate coverage.
- Overlap: case_ids: GF-REGIONAL-CMD-018; GF-REGIONAL-PREEMPT-018; classification: different_layer; rationale: Planner preserves submitted step vs adapter holds sibling identity; dynamic nodes vs GPU scope.
- Tests: tests/orchestration/test_cross_fault_arbitration.py::test_dag_widens_unstarted_same_rank_branch_for_another_gpu; tests/orchestration/test_cross_fault_arbitration.py::test_dag_appends_same_node_successor_after_branch_started
- Findings At Review: No actual LEASED command raced against merge; WAITING execution is seeded
- Reviewer Handoff: reviewed_local_contract

### GF-REGIONAL-PREEMPT-022

仲裁必须是 branch-local，不被别的节点污染

Current catalog: non-destructive; pytest; NOT_RUN.

- Purpose: Branch-local rank lets node-a reset upgrade to reboot despite node-b already rebooting.
- Runner: tests/orchestration/_cross_fault_arbitration_cases_2.py
- Entry Points: catalog pytest -> ASGI node-a reset/node-b reboot/node-a reboot
- Assertions: Same workflow; exactly one reboot per node; node-a reset superseded; STOP/restart each1
- Safety And Cleanup: Memory planner-only, no hardware actions
- Necessity: Distinct regression against global-rank pollution.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-002; GF-REGIONAL-PREEMPT-004; classification: composition_case; rationale: Composes clean branch preemption and rank gate across competing node branches.
- Tests: tests/orchestration/test_cross_fault_arbitration.py::test_dag_compares_recovery_rank_within_the_target_node_branch
- Findings At Review: No in-flight node-a reset variant; not needed for this unstarted-step expectation but separate boundary coverage remains necessary
- Reviewer Handoff: reviewed_local_contract

### GF-REGIONAL-PREEMPT-023

跨故障域的 action dominance 与并行边界

Current catalog: non-destructive; manual; SUPERSEDED.

- Purpose: Retired cross-domain dominance case, incorporated as an exact pytest nodeid in024.
- Runner: tests/orchestration/test_merge.py
- Entry Points: SUPERSEDED/manual catalog and do_not_run; no independent runner
- Assertions: Retained domain/rank/mutation conflict matrix is selected by024
- Safety And Cleanup: Remain retired; local regression retained
- Necessity: Exact duplicate selection correctly removed.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-024; classification: retired_exact_test_subset; rationale: Same test_same_node_cross_domain_actions_require_explicit_dominance nodeid.
- Tests: tests/orchestration/test_merge.py::test_same_node_cross_domain_actions_require_explicit_dominance
- Findings At Review: No missing standalone live runner; do not revive
- Reviewer Handoff: retired_preserved

### GF-REGIONAL-PREEMPT-024

持续采集与按需诊断的边界

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: Existing collector evidence is frozen once; only bounded live-process capture delays shared STOP; domain dominance explicit.
- Runner: tests/orchestration/test_health.py; tests/orchestration/test_merge.py
- Entry Points: catalog python3 -m pytest wrapper -> three nodeids, collector categories parametrized
- Assertions: FREEZE first and no recapture bundle for CPU/memory/storage/RDMA/NCCL; Cross-domain concurrency/dominance matrix; Before STOP capture dependency; after STOP capture disabled with reason and FREEZE retained
- Safety And Cleanup: Models/merger and Memory orchestrator only
- Necessity: Appropriate superset of retired023 with added evidence/STOP ordering.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-023; GF-REGIONAL-PREEMPT-017; classification: superset_and_shared_DAG; rationale: Includes023 exactly; fanout case exercises executor rather than capture dependency compilation.
- Tests: tests/orchestration/test_health.py::test_collector_backed_diagnostics_do_not_recapture_evidence; tests/orchestration/test_merge.py::test_same_node_cross_domain_actions_require_explicit_dominance; tests/orchestration/test_merge.py::test_stop_waits_only_for_available_live_process_capture
- Findings At Review: No actual evidence-ref freeze readback or time-budget bound execution; selected tests inspect plans, not running diagnostics
- Reviewer Handoff: reviewed_local_contract; retirement_correct

### GF-REGIONAL-PREEMPT-025

本方案主动终止任务后 Watcher 终态走因果 ack 快速路径

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: Controller-initiated termination remains Watcher acknowledgement and cannot recurse into new recovery or consume restart budget.
- Runner: tests/completion/test_completion_controller.py; tests/completion/test_completion_service.py
- Entry Points: catalog python3 -m pytest wrapper; fake watcher Pod/sink then real completion service on Memory
- Assertions: Watcher terminal POST retains initiator/no stopper call; NO_ACTION/no recovery plan/marker match; incident/workflow/marker counts and budget/workflow unchanged
- Safety And Cleanup: Fake Kubernetes and local Store; no stop/restart
- Necessity: Good coverage of both ends of the causal acknowledgement contract.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-033; classification: distinct_initiator; rationale: Our own stop acknowledges recovery; external stop withdraws recovery.
- Tests: tests/completion/test_completion_controller.py::test_incident_termination_does_not_suspend_workload; tests/completion/test_completion_service.py::test_controller_initiated_failed_exit_does_not_recurse
- Findings At Review: Two isolated endpoints are not linked into one deployed Watcher/processor/Store chain; no live-runner proof claimed
- Reviewer Handoff: reviewed_local_contract

### GF-REGIONAL-PREEMPT-026

QUARANTINE 终止共享 restart 而不伪造取消

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: Final quarantine suppresses unsubmitted readmission/restart while started branches settle; late quarantine after restart gets new generation.
- Runner: tests/orchestration/test_merge.py; tests/execution/_executor_cases_1.py
- Entry Points: catalog python3 -m pytest wrapper -> two planner tests plus terminal DAG executor test
- Assertions: Unstarted restore/restart superseded; in-flight reset record unchanged; Late restart WAITING original unchanged/new queued successor; Final quarantine workflow SUCCEEDED, incident QUARANTINED, no restart adapter
- Safety And Cleanup: Memory/models/fake adapter only
- Necessity: Complementary planning and terminal execution controls; not a duplicate of ordinary branch failure.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-019; GF-REGIONAL-PREEMPT-020; GF-REGIONAL-PREEMPT-031; classification: shared_invariant_distinct_trigger; rationale: Failure blocks join, late node makes new generation, host critical finding triggers quarantine.
- Tests: tests/orchestration/test_merge.py::test_quarantine_branch_suppresses_shared_restart_and_readmission; tests/orchestration/test_merge.py::test_quarantine_after_restart_submission_queues_new_generation; tests/execution/test_executor.py::test_terminal_quarantine_dag_finishes_without_shared_restart
- Findings At Review: No actual remote in-flight sibling cancellation/settlement loop
- Reviewer Handoff: reviewed_local_contract

### GF-REGIONAL-PREEMPT-027

跨 group 到达顺序对称合并为同一 DAG

Current catalog: non-destructive; pytest; NOT_RUN.

- Purpose: XID and replacement group arrival order both join one attempt DAG with distinct node branches and single STOP/restart.
- Runner: tests/orchestration/test_xid.py
- Entry Points: catalog pytest -> replacement_first true/false; orchestrator ingestion
- Assertions: Within each run same incident/workflow, DAG enabled/no predecessor, STOP/restart1, reset node-a/replace node-b
- Safety And Cleanup: Memory and synthetic findings, no actual replacement
- Necessity: Useful symmetric grouping test; equality is within each order, not requiring deterministic IDs across independent runs.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-016; classification: distinct_group_key_ordering; rationale: Same-attempt grouping across different persistent group types, not just independent node scope.
- Tests: tests/orchestration/test_xid.py::test_xid_and_replacement_arrival_order_joins_same_attempt_dag
- Findings At Review: No concurrent group creation/Store transaction test in selection; No physical replacement or executor join execution
- Reviewer Handoff: reviewed_local_contract

### GF-REGIONAL-PREEMPT-028

不同 attempt 独立与节点独占不变量

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: Different attempts never merge; stale generation rejected; ambiguous shared-node ownership audited and alarmed.
- Runner: tests/orchestration/_cross_fault_arbitration_cases_1.py; tests/orchestration/test_health.py; tests/orchestration/test_multi_node_sxid.py; tests/orchestration/test_merge.py; tests/metrics/test_metric_contributors.py; tests/regional/test_production_safety_config.py
- Entry Points: catalog python3 -m pytest wrapper -> eight nodes; Memory plus one SQLite case
- Assertions: Different incident/workflows for ambiguous/replacement/SXID attempts; Old boot and aged unbound event fence; Node-exclusive conflict lookup scoped; Ambiguity counters/logs, exported metrics and optional PrometheusRule expressions
- Safety And Cleanup: Local ASGI, fake observations; SQLite close in finally
- Necessity: Good multi-layer invariant suite, but three real scheduling/Terminating race windows are not represented as complete live sequences.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-003; GF-REGIONAL-PREEMPT-035; classification: different_ownership_boundary; rationale: Preemption scope and placement hold complement attempt ambiguity/fencing.
- Tests: tests/orchestration/test_cross_fault_arbitration.py::test_multiple_active_attempts_disable_cross_type_grouping; tests/orchestration/test_health.py::test_replacement_findings_for_different_attempts_do_not_merge; tests/orchestration/test_multi_node_sxid.py::test_sxids_from_different_attempts_do_not_merge; tests/orchestration/test_merge.py::test_fault_action_generation_fence_blocks_previous_boot; tests/orchestration/test_merge.py::test_fault_action_generation_fence_blocks_aged_event_without_boot; tests/orchestration/test_health.py::test_node_predecessor_requires_same_node_conflicting_claims; tests/metrics/test_metric_contributors.py::test_ambiguous_attempt_metric_is_exported; tests/regional/test_production_safety_config.py::test_ambiguous_attempt_ownership_has_a_prometheus_alert
- Findings At Review: Selected alert check reads optional processor-alerts.yaml, not deployed AMP rules; No actual Pod transition/concurrent ownership test or live alert delivery
- Reviewer Handoff: reviewed_local_contract; parent_alert_selection_proposal
- Current Disposition: Ownership alert regression validates the production AMP rules as well as optional PrometheusRule rules.

### GF-REGIONAL-PREEMPT-029

warm-spare 替换成功的固定邮件与去重

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: Warm-spare success notification includes rebinding/identities/no provider mutation and deduplicates through regional storeless path.
- Runner: tests/execution/_node_action_cases_1.py; tests/notifications/_notifications_cases_1.py; tests/regional/_regional_control_plane_cases_2.py
- Entry Points: catalog python3 -m pytest wrapper -> five adapter/template/API tests
- Assertions: Fake spare allocation1/no provider calls, expected email fields and duplicate ID; Remote success persists outbox and invokes sender once; Template dedup suffix; Regional registry forwards API; ASGI duplicate shortage email dedups
- Safety And Cleanup: Fake provider/spares/Kubernetes, Memory outbox; no email delivery
- Necessity: Useful component coverage, but regional notification route test uses shortage template rather than warm-spare success end to end.
- Overlap: case_ids: GF-REGIONAL-NOTIFY-001; GF-REGIONAL-DESTR-003; classification: different_layer; rationale: Notification content/dedup components vs actual mail and physical spare workflow.
- Tests: tests/execution/test_node_action.py::test_hyperpod_replace_records_activated_spare_nodes; tests/execution/test_node_action.py::test_control_plane_persists_remote_warm_spare_success_email; tests/notifications/test_notifications.py::test_warm_spare_replacement_email_lists_rebinding; tests/regional/test_regional_control_plane.py::test_regional_registry_persists_notification_over_api; tests/regional/test_regional_control_plane.py::test_regional_executor_can_page_operator_on_spare_shortage
- Findings At Review: No single store=None success adapter -> regional client -> actual success outbox chain; Provider message delivery/exactly-once actual email not proven
- Reviewer Handoff: reviewed_local_contract

### GF-REGIONAL-PREEMPT-030

reboot 后主动触发 health snapshot 再放行 validation

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: After reboot stabilization trigger health snapshot; WAITING holds, SUCCEEDED records details, FAILED falls back.
- Runner: tests/execution/_node_action_cases_2.py
- Entry Points: catalog pytest -> three snapshot_status variants of fake lifecycle + FleetRegistry test
- Assertions: New Agent incarnation confirms reboot; snapshot WAITING then success, success details, failure error fallback; snapshot calls scoped
- Safety And Cleanup: Fake lifecycle and NodeAction adapter; no reboot or snapshot on a node
- Necessity: All three snapshot outcomes covered, but stabilization is explicitly configured0 in this selected test.
- Overlap: case_ids: GF-REGIONAL-DESTR-002; classification: component_vs_live; rationale: Deterministic snapshot state machine vs actual provider reboot/health proof.
- Tests: tests/execution/test_node_action.py::test_hyperpod_reboot_auto_confirms_new_ready_agent_incarnation; tests/execution/test_node_action.py::test_hyperpod_reboot_waits_for_post_reboot_stabilization
- Findings At Review: The second listed stabilization test is not selected by catalog; No real60s/minutes cadence measurement or deployment coverage
- Reviewer Handoff: reviewed; parent_stabilization_test_selection_proposal
- Current Disposition: The stabilization test is now selected alongside snapshot outcome cases.

### GF-REGIONAL-PREEMPT-031

critical host quarantine 并入同节点 active workflow

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: Critical host findings join active GPU recovery and preserve quarantine against later automatic takeover.
- Runner: tests/orchestration/_misc_cases_1.py; tests/execution/_kubernetes_cases_1.py; tests/regional/_regional_control_plane_cases_2.py
- Entry Points: catalog python3 -m pytest wrapper -> host category matrix, idle-node merge, fake Kubernetes takeover and ownership API/client
- Assertions: Eight critical finding variants same DAG with both causes, suppressed restore/restart; Idle MCE same-node merge; Terminal quarantine rejects reset but permits quarantine-preserving takeover; API/client retains quarantine_hold and unknown/cross-cluster guards
- Safety And Cleanup: Synthetic findings/fake Kubernetes/Memory; no real host fault or taint
- Necessity: Strong component coverage of merge plus regional ownership reporting.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-026; GF-REGIONAL-PREEMPT-028; classification: distinct_trigger_and_takeover; rationale: Same suppression invariant, but host critical evidence and terminal quarantine ownership are unique.
- Tests: tests/orchestration/test_misc.py::test_critical_host_quarantine_joins_running_attempt_recovery; tests/orchestration/test_misc.py::test_idle_host_quarantine_joins_active_node_reset; tests/execution/test_kubernetes.py::test_terminal_quarantine_hold_rejects_automatic_reset_takeover; tests/regional/test_regional_control_plane.py::test_incident_ownership_route_answers_takeover_question; tests/regional/test_regional_control_plane.py::test_regional_ownership_provider_reads_the_route
- Findings At Review: Critical classification is supplied as finding, not generated from real collector data; Idle path covers MCE only; no complete storeless adapter -> API -> patch rejection chain
- Reviewer Handoff: reviewed_local_contract

### GF-REGIONAL-PREEMPT-032

工作流生命周期硬截止：到点失败、取消在飞命令、只升级到人工、后续事件只记录

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: Workflow hard lifetime expires, cancels pending remote work, compensates only restore and absorbs later stronger fault without new recovery.
- Runner: tests/execution/test_workflow_lifecycle_scenarios.py; tests/orchestration/test_job_workflow_lifetime.py
- Entry Points: catalog python3 -m pytest wrapper -> SQLite API/executor scenario plus complete lifetime unit module
- Assertions: First claim stamps/caps deadline; after1.3s fails command/workflow with lifetime marker; no new action except RESTORE; support-only escalation; later XID stays same failed record; Boundary instant, inherited deadline, no branch escalation and post-expiry absorb covered
- Safety And Cleanup: SQLite close in finally and Memory fakes; no node command/client
- Necessity: Meaningful deterministic lifecycle integration with explicit live supplement belonging to DESTR018.
- Overlap: case_ids: GF-REGIONAL-CMD-013; GF-REGIONAL-DESTR-018; classification: command_workflow_physical_layers; rationale: Unlimited WAITING rounds do not remove workflow deadline; live ledger cancellation is DESTR018.
- Tests: tests/execution/test_workflow_lifecycle_scenarios.py::test_preempt032_a_node_workflow_past_its_lifetime_ends_with_the_operator; tests/orchestration/test_job_workflow_lifetime.py
- Findings At Review: Real kernel ledger/cancellation boundary and AMP alert firing not this pytest proof; Fixed sleep could be replaced by injected time in a future narrow test improvement
- Reviewer Handoff: reviewed_local_contract; do_not_duplicate_DESTR018

### GF-REGIONAL-PREEMPT-033

作业被外部停止后工作流收口：在飞动作跑完、已 cordon 节点恢复调度、不再重启作业

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: External STOPPED withdraws DAG, finishes in-flight branch, releases only touched node and never restarts stopped job.
- Runner: tests/execution/test_workflow_lifecycle_scenarios.py; tests/execution/test_workload_withdrawal.py
- Entry Points: catalog python3 -m pytest wrapper -> completion/executor integration and withdrawal module; optional external PG parametrization
- Assertions: withdrawn timestamp; node-b reset/restore only; node-c steps/restart superseded; workflow SUPERSEDED/incident RECOVERED; amend merge revision, later event record-only and Store persistence
- Safety And Cleanup: Fake adapters, Memory/SQLite; optional PG fixture skips without explicit URL and must run separately when enabled
- Necessity: Useful local external-stop lifecycle test, distinct from own-stop acknowledgement025.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-025; GF-REGIONAL-PREEMPT-035; classification: distinct_user_intent; rationale: Own stop ack vs external withdrawal vs placement-time corrective stop.
- Tests: tests/execution/test_workflow_lifecycle_scenarios.py::test_preempt033_a_stopped_job_winds_its_workflow_down_without_restarting_it; tests/execution/test_workload_withdrawal.py
- Findings At Review: No dedicated deployed PyTorchJob remains-suspended/real node readmission runner; Plain pytest wrapper can report PASS while optional PG cases skip; backend proof must be explicit
- Reviewer Handoff: reviewed; new_live_supplement_proposal_for_parent_only

### GF-REGIONAL-PREEMPT-034

新作业工作流等待节点修复：最多等待 300 秒，超时只执行 STOP_WORKLOADS 并 FAILED/ESCALATED

Current catalog: non-destructive; manual; SUPERSEDED.

- Purpose: Retired node-busy wait/timeout STOP case, subsumed by035 placement hold.
- Runner: tests/execution/test_node_busy_wait.py; tests/execution/test_workflow_lifecycle_scenarios.py
- Entry Points: SUPERSEDED/manual catalog and do_not_run; retained regressions only
- Assertions: Retained tests wait while node busy then STOP-only failure, node owner unchanged and timeout count1
- Safety And Cleanup: Memory/SQLite fake adapters; no independent acceptance execution
- Necessity: Retirement consistent with stronger035 scenario; retain lower-level regression tests.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-035; classification: retired_scenario_subset; rationale: Same timeout path;035 adds hold creation and successful dissolve.
- Tests: tests/execution/test_node_busy_wait.py; tests/execution/test_workflow_lifecycle_scenarios.py::test_preempt034_a_job_on_a_node_under_repair_waits_then_stops_the_job
- Findings At Review: Historic300s wording is retired, not permission to change active035's240s contract
- Reviewer Handoff: retired_preserved

### GF-REGIONAL-PREEMPT-035

新作业被观察到落在修复中节点：开出安置保留，窗口内节点释放即消散，超时只执行 STOP_WORKLOADS 并 FAILED/ESCALATED

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: Observed placement on repairing node opens deduplicated STOP-only hold; node free within window dissolves; timeout stops once and fails.
- Runner: tests/execution/test_placement_hold_dispatch.py; tests/orchestration/test_placement_hold.py
- Entry Points: catalog python3 -m pytest wrapper -> full SQLite dispatcher and Memory placement-hold modules
- Assertions: Hold identity/policy/HOLD event, initiator and scope; repeated/terminal/own-initiator/read-only/resolved cases do not duplicate; node_busy then dissolve without adapter call or timeout STOP failure; late-free race cannot dissolve decided timeout
- Safety And Cleanup: SQLite close in finally; Memory fakes; no actual stop
- Necessity: Good deterministic superset of034. The success path intentionally finishes node repair, so spec's statement node workflow remains RUNNING in both paths is contradictory.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-034; GF-REGIONAL-PREEMPT-028; classification: superset_and_complementary_invariant; rationale: Subsumes034; ownership anomalies motivate but do not duplicate corrective placement hold.
- Tests: tests/execution/test_placement_hold_dispatch.py; tests/orchestration/test_placement_hold.py
- Findings At Review: Tests call hold_attempt_on_repairing_nodes directly, not POST /v1/workload-observations chain; Timeout measured from first HOLD event, not only created_at as prose suggests; No dedicated live workload stop/no-stop supplement; no actual metrics endpoint assertion
- Reviewer Handoff: reviewed; parent_spec_and_live_supplement_proposal

### GF-REGIONAL-PREEMPT-036

卡死工作流由 dispatcher 自动关闭：compile-blocked、orphaned-commands、retired-generation 三种形态在每轮清扫中各自收敛，孪生的活记录逐字段不动，再跑一轮幂等

Current catalog: non-destructive; command; NOT_RUN.

- Purpose: Isolated compile-blocked/orphaned/retired-generation sweep closes only eligible records, preserves twins/all fields, keeps manual blockers and reruns idempotently.
- Runner: scripts/e2e/regional/run_preempt036_stuck_workflow_reconcile.py; scripts/e2e/regional/preempt036_verdicts.py
- Entry Points: catalog command -> main auto provisions own PG16 container or SQLite -> run_shape -> production sweep_stuck_records and shipped release probes
- Assertions: Three shape-specific status/audit/cancellation/withheld sets; Exact release blocker/compile/resolved lists and counts; open-command delta; rerun unchanged; Full canonical record digest added to snapshots to cover previously omitted fields
- Safety And Cleanup: Only provisioner-owned URLs, no ambient Store; no adapters or cluster clients; Current run_shape lacks Store.close finally; SQLite work files not temporary; unchecked docker removal/no subprocess timeout
- Necessity: Appropriate isolated Store integration, not live proof; backend explicitly recorded. No DDL source modified or real DB started in this review.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-032; GF-REGIONAL-BOOT-020; classification: different_lifecycle_and_release_gate; rationale: Predicate-driven stale-record closure vs hard timeout; release preflight proof is reused, not duplicate scenario.
- Tests: tests/regional/test_preempt036_stuck_workflow_reconcile.py; tests/regional/test_protocol_fail_closed_review.py::test_sweeper_snapshot_detects_changes_outside_its_old_projection
- Findings At Review: PROTO-011 snapshot gap now fixed/tested; Single --shape can write full-case PASS; source/probe coverage runner was0% baseline; Resource lifecycle/failure fake tests needed; optional real PG mode coverage requires parent coordination
- Reviewer Handoff: fixed_owned_full_record_digest_Store_close_and_all_three_shape_gate; all_SQLite_shapes_tested; optional_PG_provisioner_not_validated

### GF-REGIONAL-PREEMPT-037

调度循环判活：关闭 workflow dispatcher 后 last-cycle 时间戳老化，GpuFaultWorkflowDispatcherStalled 表达式在其 for 内成立，恢复后归零

Current catalog: live-service-action; manual; NOT_RUN.

- Purpose: Temporarily disable dispatcher on all CPU workers, prove stall expression for duration while periodic loop lives, restore exact env/process values.
- Runner: scripts/e2e/regional/run_preempt037_dispatcher_liveness.py; scripts/e2e/regional/preempt037_verdicts.py
- Entry Points: manual -> plan/authorize/predecessor -> execute -> Deployment env mutation and per-Ready-Pod metrics -> finally restore
- Assertions: Rule threshold/for/severity/anchor; quiescence; false env in replicas; Continuous observed stall period with periodic freshness; exact env restore and fresh dispatch after recovery
- Safety And Cleanup: Initial window budget and finally restore; no independent deadman; no UID/resourceVersion binding to deployment
- Necessity: Distinct process-alive/loop-stalled test, not AMP alert service proof. Missing/NaN/future timestamps now rejected rather than counted as alert/recovery.
- Overlap: case_ids: GF-REGIONAL-HA-007; classification: distinct_failure_mode; rationale: Loop disabled while process lives vs graceful worker process exit.
- Tests: tests/regional/test_preempt037_dispatcher_liveness.py; tests/regional/test_protocol_fail_closed_review.py::test_missing_metric_is_not_proof_that_the_alert_expression_fired; tests/regional/test_protocol_fail_closed_review.py::test_invalid_or_future_dispatch_metric_cannot_prove_recovery
- Findings At Review: PROTO-012 latest500 active-state omission remains to fix; Supported rule shape not validated; defaults can silently reinterpret changed PromQL; No independent rollback; no exact desired replica census; sparse timeline can claim continuous duration; Result lacks release/cluster identity
- Reviewer Handoff: fixed_owned_complete_census_exact_rule_finite_metrics_timeline_and_UID_image_bound_watchdog; local_tested_and_census_SQL_verified_in_all_three_modes

### GF-REGIONAL-PREEMPT-038

原始证据钉在 incident 生命周期上：过期行在同集群开放 incident 点名其节点时不被清理，RECOVERED 后才删，且每次清理记录被删的键

Current catalog: live-non-destructive; manual; NOT_RUN.

- Purpose: Deployed periodic sweep deletes unrelated expired evidence, retains node-window pin until incident RECOVERED, then deletes/logs own keys.
- Runner: scripts/e2e/regional/run_preempt038_evidence_pins.py; scripts/e2e/regional/audit_raw_evidence_periodic_cleanup.py; scripts/e2e/regional/preempt038_verdicts.py
- Entry Points: manual -> guarded main -> control-worker pod_python audit -> two-phase SQL pin observation -> logs
- Assertions: Unrelated deleted<=180s; pinned survives twice; after recovery deleted<=180s; owned suffix residual0; positive-count log names both keys
- Safety And Cleanup: Synthetic cluster/node rows, exact keys/suffix; SQL finally cleanup; no DDL; PROTO-013 mutation ACK/aborted-transaction cleanup; wrapper generic retry and no caller-bound suffix
- Necessity: Distinct deployed periodic retention proof; attempt pin and cap policy deliberately not live-tested here.
- Overlap: case_ids: GF-REGIONAL-PREEMPT-036; classification: different_cleanup_domain; rationale: Raw-evidence retention/pinning vs workflow/command stuck-state reconciliation.
- Tests: tests/regional/test_preempt038_evidence_pins.py; tests/store/test_evidence_pinned_to_incident.py; tests/regional/test_protocol_fail_closed_review.py::test_empty_evidence_keys_cannot_match_arbitrary_cleanup_logs
- Findings At Review: Empty-key log check fixed/tested; No SQL probe behavior tests at baseline0% coverage; caller-known IDs/transaction recovery needed; Log search only current Ready Pods/keys[:20] may miss actual sweep evidence; Report verdict and numeric timing types/bounds not yet checked; result identity absent
- Reviewer Handoff: fixed_owned_caller_seed_identity_transaction_recovery_and_report_binding; local_tested_and_real_SQL_verified_in_all_three_modes

### GF-REGIONAL-WORKLOAD-001

用 gpu-training-submit 提交三节点 managed PyTorchJob

Current catalog: live-non-destructive; manual; NOT_RUN.

- Purpose: Submit a managed three-node PyTorchJob through gpu-training-submit, prove all metadata, 24 NCCL ranks and three decreasing-loss steps, watcher-owned terminal Observation/event, and deletion monotonicity without manual observation posts.
- Runner: scripts/e2e/regional/run_workload_acceptance.py; scripts/e2e/regional/managed_workload_fixture.py
- Entry Points: run_workload_acceptance.run_workload_baseline; run_workload_acceptance.wait_finite_workload; run_workload_acceptance.metadata_errors; run_workload_acceptance.loss_errors; run_workload_acceptance.WORKLOAD_STORE_PROBE
- Assertions: Requires three succeeded Pods on distinct nodes and one SUCCESS/all_reduce=300.0 substring per Pod.; loss_errors only checks non-increasing values for whichever ranks/steps occur; a single point or flat loss passes and full 24x3 coverage is absent.; Metadata check omits role and some workload-level fields; Observation ranks are checked but expected_critical_ranks and exact Pod UID binding are not.; A deterministic decision.event_key is used as proof of exactly one terminal event; no event count or retained three-container post-delete assertion exists.
- Safety And Cleanup: No fault injection or manual observation post is intended.; Shared submit deletes a fixed resource name/job label before creating without UID ownership; final deletes suppress command failure and do not verify absence.; No fresh empty control-record baseline is checked for this workload case, unlike ISO-001/E2E-001.; WORKLOAD_STORE_PROBE dumps full remote commands, and its substring filter can omit commands even when workflow IDs are known.
- Necessity: Keep as a prerequisite proving the normal submit path and watcher discovery before any restart acceptance depends on ownership.
- Overlap: case_ids: GF-REGIONAL-WORKLOAD-002; GF-REGIONAL-E2E-001; classification: shared-fixture-distinct-entrypath; rationale: WORKLOAD-002 bypasses the submission entrypoint after annotation, while E2E-001 uses a long-running job and fault recovery. Neither proves this finite direct-submit terminal path.
- Tests: existing: tests/regional/test_workload_acceptance_runner.py::test_loss_errors_require_non_increasing_loss_per_rank; tests/regional/test_workload_acceptance_runner.py::test_observation_ranks_must_be_exactly_the_expected_set; tests/regional/test_workload_acceptance_runner.py::test_cleanup_workload_deletes_after_quiescence_or_without_an_injection; missing_behavioral: Full fake run_workload_baseline success and failure lifecycle, including no manual observation post.; Require ranks 0..23, steps 0..2 exactly once per rank, finite losses with final strictly below initial, and correct per-rank collective output.; Wrong role/profile/expected-rank/Pod UID, duplicate terminal event, dropped post-delete containers, stale identity, and failed cleanup must fail.; Prove final batched verification includes all named workload, command, node, and watcher-terminal invariants.
- Findings At Review: Current loss unit test explicitly allows incomplete rank/step sets and unchanged loss.; The named runner lifecycle has no behavioral end-to-end fake test; helpers and source strings dominate.; After deletion, only Observation phase is checked; retained containers, event idempotency, actual poll progress, node state, and Pod/PyTorchJob absence are not.; Spec's claimed batch cleanup verification is not implemented by this runner.
- Reviewer Handoff: FIXED: all 24 ranks need three finite strictly decreasing loss steps and complete collective evidence. Fresh control-record baseline, metadata, exact Pod UID/rank/terminal container sets and post-delete retention are checked. Shared fixture uses the existing submit run API with a create-only callback, run ownership marker, UID/resourceVersion cleanup and ACK-loss recovery; preexisting names/labels never authorize deletion. Full finite lifecycle tests cover success, malformed proof and residuals.

### GF-REGIONAL-WORKLOAD-002

gpu-fault-workload-annotate 后 kubectl apply 等价于直接提交

Current catalog: live-non-destructive; manual; NOT_RUN.

- Purpose: Render with gpu-fault-workload-annotate, prove byte equality with submit --dry-run, perform server-side dry-run and independent apply, then satisfy the same watcher/finite-training/cleanup invariants as WORKLOAD-001.
- Runner: scripts/e2e/regional/run_workload_acceptance.py; scripts/e2e/regional/managed_workload_fixture.py
- Entry Points: run_workload_acceptance.run_workload_baseline; run_workload_acceptance.admin_status_policy
- Assertions: Invokes annotate and submit dry-run separately, compares bytes, then runs server dry-run, delete, and apply.; A false render_equivalence is recorded but does not stop before live apply.; The direct-submit path is not used for actual submission, matching the intended integration difference.; All finite-workload/rank/terminal-event weaknesses are inherited; admin status runs here by default, but --skip-admin-status excludes it from PASS checks.
- Safety And Cleanup: Uses a fixed rendered PyTorchJob name and shared non-UID-fenced deletion.; Render mismatch or malformed metadata is only judged after the workload has run.; Cleanup suppresses underlying delete errors through the shared fixture and does not prove final resource absence.
- Necessity: Keep this integration route separately: external schedulers depend on annotation being a reusable pre-apply transformation, not an inseparable submit side effect.
- Overlap: case_ids: GF-REGIONAL-WORKLOAD-001; classification: shared-assertions-distinct-injection; rationale: The final watcher expectations are shared, but annotate plus independent apply is a different submission boundary with its own byte-equivalence and admission checks.
- Tests: existing: tests/regional/test_workload_acceptance_runner.py (historical pre-fix symbol: test_workload002_deletes_the_same_named_job_before_apply); tests/regional/test_workload_acceptance_runner.py::test_admin_status_runs_only_on_the_group_closing_case_unless_skipped; missing_behavioral: Record the actual fake command sequence: annotate, submit dry-run, compare, server dry-run, owned delete/create, and never submit's live path.; Render mismatch or either dry-run failure must prevent all subsequent mutations.; Full lifecycle tests for shared rank/terminal semantics, admin-status failure/skip classification, and cleanup errors.
- Findings At Review: The current source-order test cannot detect that byte mismatch still applies a workload.; The documented two-stage rendering guarantee is only a final diagnostic, not a fail-closed gate.; Formal acceptance needs an explicit parent policy for skipped group-closing status; it must not silently stand for a complete group.
- Reviewer Handoff: FIXED: byte mismatch or either dry-run failure stops before submission. The independently rendered workload goes through submit_rendered/create-only ownership after server dry-run, not name deletion followed by apply. Same UID/ACK/terminal/loss cleanup protections as WORKLOAD-001. Behavioral command-order and refusal tests replace the old source-order assertion. Parent must update obsolete pre-apply-delete wording.
