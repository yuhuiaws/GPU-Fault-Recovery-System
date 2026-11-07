English edition of `docs/故障模拟测试手册.md`; the Chinese file remains the source of record until both are maintained together.

# GPU Fault Simulation Test Manual

## 1. Goal

The test system records fault scenarios in `testcases/fault-scenarios.yaml`. Every test case must contain:

- a unique ID, title, category, verification level and risk level;
- a problem statement, the fault injection method and an auditable expectation;
- a `pytest`, a controlled `command` or a manual `procedure`;
- an optional public `evidence.verdict`.

A simulated event is not the same as a real hardware fault. Reports must distinguish the following verification levels:

| Level | Meaning |
|---|---|
| `unit` | A single function or model boundary |
| `component` | Deterministic tests of a real policy, Collector, Store or adapter |
| `integration` | Multi-module tests of API, workflow, adapter or the durable Store |
| `end-to-end` | A closed loop from event input to decision, workflow, notification or Agent barrier |
| `staging` | Controlled acceptance in an isolated HyperPod or plain EKS (`eks-managed`) test cluster |
| `live` | Manual or explicit command acceptance in a real HyperPod or plain EKS environment |

Full acceptance of the split deployment of the regional CPU control plane and the GPU data plane is in
[Regional E2E Acceptance Test Cases](regional-e2e-acceptance-test-cases.md). `GF-REGIONAL-*`
cases are not all executed manually: the current catalog contains all three automation types, `manual`, `pytest` and `command`.
Acceptance of real processes, identity, network and blast radius is usually manual or an explicit command; pure protocol and
state-machine contracts can be covered by pytest.

## 2. Safety Boundaries

By default the runner executes only scenarios with `risk: non-destructive` and `automation: pytest`, does not connect to
production Kubernetes, and calls no GPU reset, node reboot or warm-spare switch.

A real-cluster command must satisfy all of the following:

1. `--include-live` is used;
2. `--case` is supplied explicitly;
3. the case is declared `automation: command` in the catalog.

For manual scenarios, even with `--include-manual`, the runner only produces `NOT_RUN` records and executes no command.

The promoted regional runner uses the `formal` scope by default and requires the formal predecessor. When an administrator
needs to execute only explicitly selected cases, use:

```bash
GPU_FAULT_ACCEPTANCE_EXECUTION_SCOPE=selective \
GPU_FAULT_ACCEPTANCE_SELECTION_REFERENCE=CHG-12345 \
python3 scripts/e2e/regional/<runner>.py --case <case> ...
```

selective only skips the formal-order predecessor; it does not skip that case's own preflight, maintenance window,
confirmation, node idleness, identity/pin, stop conditions and cleanup. The report must keep
`execution_scope=selective`, the audit reference and `formal_sequence_satisfied=false`; that PASS cannot
be used by the formal runner as predecessor evidence for later cases.

The only vocabulary of risk levels is in
[Regional E2E Acceptance Test Cases §2.2](regional-e2e-acceptance-test-cases.md#22-risk-level-vocabulary-the-single-source-of-values)
and `RISK_VALUES` in `tools/run_fault_test_cases.py`. This document does not copy a second complete vocabulary.

The node lifecycle supports only the following production paths:

| Risk | Action | Constraints |
|---|---|---|
| `destructive` | The Node Agent performs a GPU/fabric reset | Must quiesce, have no GPU clients, be signed, fenced and validated after the action |
| `destructive-provider-reboot` | Reboot of the original node: HyperPod `BatchRebootClusterNodes`, plain EKS EC2 `RebootInstances` on the bound instance | HyperPod `NodeRecovery=None` (plain EKS: the node group's EKS node auto repair off), secondary cluster confirmation and provider idempotency |
| `destructive-warm-spare` | `HEALTHY_WARM_SPARE_ONLY` local switch | Uses only managed healthy spares; the faulty instance stays in existence and isolated |

`destructive-provider-replace` is neither a legal risk nor an executable test path.
`BatchReplaceClusterNodes` may only serve as a refutation target: that call must not appear anywhere in the window; when
warm spares are insufficient it must fail and alert, never fall back to provider replace.

Every real-hardware test must:

- state the target Region, context, cluster ID, nodes and maintenance window explicitly;
- confirm HyperPod `NodeRecovery=None`, and that plain EKS node groups have EKS node auto repair off;
- record the current release, Runtime Profile, required/compatible pins and node Agent generation;
- use isolated nodes or a controlled training job, with stop conditions and rollback defined in advance;
- write the raw report to `artifacts/fault/`, a CI artifact or the private evidence store;
- never describe a user-space `/dev/kmsg` write as a real NVIDIA hardware fault.

## 3. Viewing and Running Test Cases

List the cases executable by default:

```bash
python3.12 tools/run_fault_test_cases.py --list
```

Filter by category or level:

```bash
python3.12 tools/run_fault_test_cases.py \
  --category restart-guard

python3.12 tools/run_fault_test_cases.py \
  --level end-to-end
```

Run specific cases:

```bash
python3.12 tools/run_fault_test_cases.py \
  --case GF-RST-001 \
  --case GF-RST-002 \
  --report "artifacts/fault/restart-guard-$(date -u +%Y%m%dT%H%M%SZ).json"
```

Run all default non-destructive pytest scenarios:

```bash
make fault-test-cases
```

The default is still a single worker in series. Enable resource-locked parallelism only for cases the runner judges
parallel-safe:

```bash
make fault-test-cases FAULT_TEST_WORKERS=4
```

Non-destructive pytest uses the `local-test: shared` lock by default and clears
`GPU_FAULT_STORE_URL` and `GPU_FAULT_TEST_POSTGRES_URL` in the child process so that environment variables on the
admin machine cannot make a local case connect to the live database by mistake. command and manual cases use the global
exclusive lock by default; only a command that explicitly declares `execution.parallel_safe=true` in the catalog and lists
its resource locks may run in parallel. `execution` may also declare `depends_on` (the cases it depends on; when a selected
dependency fails the case is recorded `BLOCKED`) and `failure_scope` (`case`, `branch` or `global`, default `branch`; a
command/manual case without `execution` counts as `global`: once it FAILs every case not yet started is recorded
`BLOCKED`).

CAP-005 uses a random temporary database but still shares the CPU, I/O, connections, WAL and lock manager of the same
PostgreSQL/Aurora server, so it declares:

```yaml
execution:
  parallel_safe: true
  environment: inherit
  failure_scope: branch
  locks:
  - {resource: postgres-server, mode: exclusive}
  - {resource: cap005-workdir, mode: exclusive}
```

It can run alongside isolated local pytest but not in parallel with CAP, HA, CMD, processor, notification or live Store
tests that use the same database server. Recommended entry point:

```bash
GPU_FAULT_STORE_URL='<validation PostgreSQL/Aurora admin DSN>' \
  make fault-test-cases-with-cap005
```

This entry point explicitly adds CAP-005 to the default pytest set and schedules with 4 workers. `--include-live` remains
the safety confirmation for command cases; `--also-case` does not unlock other command cases that were not named.

Run an explicitly registered live command:

```bash
python3.12 tools/run_fault_test_cases.py \
  --case GF-LIVE-000 \
  --include-live \
  --report "artifacts/fault/live-passive-e2e-$(date -u +%Y%m%dT%H%M%SZ).json"
```

Add manual scenarios to the report without executing them:

```bash
python3.12 tools/run_fault_test_cases.py \
  --include-manual \
  --report "artifacts/fault/all-with-manual-$(date -u +%Y%m%dT%H%M%SZ).json"
```

## 4. Reports and Evidence

The runner starts a separate process for every pytest or command case (`--batch-pytest` runs a compatible pytest
selection in one pytest process and still reports per case; `--pytest-results` reuses the per-node results
`tools/pytest_case_reporter.py` wrote, which is what CI's `make fault-test-cases-ci` does), and any `FAIL` or `BLOCKED` case
makes the final process return non-zero. Every result contains:

- case ID, level, risk, problem, injection method and expectation;
- `PASS`, `FAIL`, `BLOCKED` (not run after a failed dependency or a global failure) or `NOT_RUN`;
- the pytest node ID or command, duration and raw output;
- the `processing_trace` produced by cases that require a trace.
- the queued/start/completed times, lock wait, worker, resource locks and environment mode in `scheduling`.

The default of `--workers` is 1, so existing commands and report order are unchanged. Under parallel execution the
`results` in the report stay in declaration order and are not reordered by actual completion.

The report-level verdicts are:

| verdict | Meaning |
|---|---|
| `PASS` | All selected cases executed and passed |
| `PASS_WITH_LIMITATIONS` | The executed cases passed, but the report contains `NOT_RUN` |
| `FAIL` | At least one case is `FAIL` or `BLOCKED` |
| `NOT_RUN` | No case passed (and none is `FAIL`/`BLOCKED`) |

A case that requires a processing trace is judged `FAIL` when it outputs no valid structure, even if the pytest exit code
is 0.

Default report location:

```text
artifacts/fault/fault-tests-<UTC timestamp>.json
```

CI executes the unit and component levels; the artifact is named `fault-test-report`, retained for 30 days, and uploaded
even on failure.

`evidence` in the public catalog allows only `verdict` and `verified`:

```yaml
evidence:
  verdict: PASS
  verified:
    case_digest: <sha256>
    components:
      control_plane: <sha256>
      executor: <sha256>
      node_runtime: <sha256>
```

The legal values are `PASS`, `NOT_RUN`, `BLOCKED` and `SUPERSEDED`. Execution time, environment, cluster names, operator,
report paths and site notes do not enter the catalog; they belong to the runner report or the private evidence store.
`SUPERSEDED` must use the top-level `superseded_by` to point at another case.

`verified` may appear only on `PASS`, and `PASS` must carry it: `case_digest` is the digest of this case's normative
fields, so rewriting the assertions immediately invalidates the old verdict (hard failure at load time); `components` are
the source digests of the verified components, and drift is derived as `STALE` by
`scripts/build-fault-evidence-index.py`, which requires re-execution in the next maintenance window. The historical live
cases in `EVIDENCE_UNBOUND_PASS_CASES` carry only `case_digest` and report `UNBOUND`; that list is a one-way ratchet that
may only shrink after a re-run on real hardware, and digests may not be backfilled after the fact. The full semantics are
in `docs/evidence/fault/README.md`.

Public long-term evidence is registered only as redacted reports through `docs/evidence/fault/manifest.yaml`. The public
manifest may currently be empty; environment-specific JSON not included in the manifest must not be cited in this
document. The normative index is generated by:

```bash
python3 scripts/build-fault-evidence-index.py
python3 scripts/build-fault-evidence-index.py --check
```

## 5. Test Case Maintenance Rules

1. When adding a fault handling branch, add or update the catalog case first.
2. An ID must not be reused once published; create a new ID when the semantics change substantially.
3. A pytest case must reference a real, collectable node ID that verifies exactly one explicit scenario.
4. A command case must use an argument array and must not depend on an implicit shell or the current kubectl context.
5. A manual case's procedure must point at a Markdown heading anchor that really exists.
6. `expected` must describe an externally observable result, never just "the system is normal".
7. When evidence is missing, expect `INCONCLUSIVE`, `BLOCKED` or `QUARANTINE`; never guess success.
8. A destructive case must record the maintenance window, isolated nodes, rollback method and operation evidence.
9. A split-deployment case must prove that the CPU control plane did not directly change GPU nodes or workloads.
10. A set-consistency assertion must bound its scope first, for example comparing only the owners that
    correspond to `GPU_FAULT_REMOTE_EXECUTION_OWNERS`.
11. When reading a process's actual configuration, construct the object in the target Pod, read its attributes and verify per replica; never infer only from env.
12. A non-destructive case must not modify running adapter switches, allowlists or release pins in order to manufacture a failure.
13. When injecting non-terminal records, the same case must provide the cleanup step so that alerts are not triggered long-term.
14. Code references cite complete Python symbols first; source line numbers, which drift easily, must not be maintained.
15. An alert case must distinguish two independent paths:
    - administrator action mail: the code generates a notification and sends it through SES;
    - operational metric alerts: `/metrics -> ADOT -> AMP -> SNS`.
16. An AMP alert must verify metric export, ADOT keep, the actually loaded AMP rule, the SigV4 query, firing and resolve.
17. A Deployment with 0 replicas may still reference a ConfigMap; the cleanup check must include ReplicaSets and rollback references.
18. The risk, level, automation and evidence fields use only the closed sets defined by the runner.
19. The parallel strategy must declare `execution.parallel_safe`, `environment` and `locks` explicitly;
    an undeclared command/manual case is globally exclusive by default. When adding a resource lock, add the scheduler conflict test first.

`AdvisoryNotificationService.dispatch_remote_completion` generates an action-completion notification only for a
successful remote command. Therefore "a command stays PENDING for a long time" must be covered by remote-command metric
alerts; an SES action-completion mail must not be expected.

## 6. Public Coverage and Sources of Truth

The current public catalog covers:

- per-item kmsg-format replay of the NVIDIA XID Catalog, XID 154, companion XIDs and NVLink5 decode;
- SXID, DCGM, GPU inventory, host, EFA/RDMA, training progress and log Collectors;
- training terminal, allocation, restart budget, idempotency and the attempt generation fence;
- regional boot guards, authentication, remote command, HA, capacity, notifications and least privilege;
- manual destructive acceptance such as reset, reboot and warm spare;
- the end-to-end specifications of the 32 general fault categories (`GF-GENERAL-E2E-*`), their software tests
  (`GF-GENERAL-SW-*`, `GF-GENERAL-FIX-*`) and live acceptance (`GF-GENERAL-LIVE-*`), and the live acceptance of GPU
  temperature, power and throttling (`GF-ENV-LIVE-*`); the software tests are pytest cases, the rest are manual, and
  the live ones are run by their own runners and approvals (`scripts/e2e/regional/run_general_live.py`,
  `scripts/e2e/regional/run_env_live.py`).

The order of authority is:

1. `testcases/fault-scenarios.yaml`: case ID, risk, automation and the current public verdict;
2. `tools/run_fault_test_cases.py`: the acceptable schema and runner behavior;
3. [Regional E2E Acceptance Test Cases](regional-e2e-acceptance-test-cases.md): the regional risk vocabulary and detailed acceptance;
4. this document: general execution method and real-hardware procedure anchors.

This document records no customer environment, cluster name, node ID, release digest or PASS/FAIL details of any single
execution. Raw results are kept in CI artifacts or the private evidence store.

## 7. Real XID Verification Tiers

Chapter 7 provides only the current manual procedures. During execution the training submission tool reads the Profile
declared by the site; hard-coding versions is forbidden:

```bash
STATE_DIR="${GPU_FAULT_STATE_DIR:?set GPU_FAULT_STATE_DIR}"

gpu-fault-admin status --full --state-dir "${STATE_DIR}"
```

The `status --full` checks must confirm that the Profile is registered, has no warnings, and agrees with the CPU side,
Collector, Watcher, Installer and node Agent. Real-hardware reports are written to `artifacts/fault/` or the private evidence store and do
not modify source files outside the public catalog.

### 7.1 kmsg Format Replay

`GF-XID-KMSG-001` through `GF-XID-KMSG-172` and the corresponding B200 family are expanded by the catalog generator.
Every case hands an NVIDIA-format record as in-memory input to the `KernelLogCollector`, which then flows into the
kernel ingestion API, normalizer, fixed Catalog and policy engine.

This tier verifies the parser, PCI BDF, boot ID, monotonic timestamp, evidence reference and policy branch; it does not
write the real `/dev/kmsg` and does not change GPU state.

```bash
python3.12 tools/run_fault_test_cases.py \
  --case GF-XID-KMSG-094 \
  --report "artifacts/fault/xid-kmsg-094-$(date -u +%Y%m%dT%H%M%SZ).json"

python3.12 tools/run_fault_test_cases.py \
  --category xid-kmsg-replay \
  --report "artifacts/fault/xid-kmsg-all-$(date -u +%Y%m%dT%H%M%SZ).json"

python3.12 tools/run_fault_test_cases.py \
  --category xid-kmsg-replay-b200 \
  --report "artifacts/fault/xid-kmsg-b200-all-$(date -u +%Y%m%dT%H%M%SZ).json"
```

### 7.2 HyperPod Three-Node XID 11 Restart

`GF-LIVE-XID11-001` verifies that XID 11 triggers a training workload restart without executing a GPU reset or node action.

```bash
gpu-training-submit \
  scripts/e2e/regional/manifests/training/xid11-three-node-pytorchjob.yaml \
  --site "${SITE_FILE}" \
  --job-id <unique-job-id> \
  --attempt-number 1 \
  --restart-budget 1
```

Before execution confirm that the three Pods sit on different GPU nodes, the 24-rank NCCL heartbeat is normal, and the
attempt observation contains 24 real GPU UUIDs. Replay the XID 11 format event only to the control plane; do not write the
node's `/dev/kmsg`.

Success criteria:

- the Catalog decision is `RESTART_APP`;
- the original three Pod UIDs disappear and the new attempt is redistributed across three nodes;
- the restart budget increases only once;
- NCCL all-reduce recovers;
- no GPU reset, node reboot or replacement occurs.

### 7.3 KernelLogCollector Isolated FIFO Verification

`GF-LIVE-KERNEL-COLLECTOR-001` verifies the Collector binary actually installed on the node without writing the real
`/dev/kmsg`.

1. Confirm the systemd Collector is enabled/active with no restarts.
2. Confirm with a read-only debug Pod that the main process holds the `/dev/kmsg` file descriptor.
3. Render `scripts/e2e/regional/manifests/kernel-collector-passive-validation.yaml` to `/tmp`;
   the temporary control plane stays in simulation with the dispatcher off.
4. Replay a normal record and an XID 11 with a duplicate sequence through a FIFO with mode `0600` and a one-shot Collector.
5. Verify that the normal record is filtered, the duplicate sequence produces only one piece of evidence, and the boot ID and attempt correlation are correct.
6. Confirm the production training Pod UIDs are unchanged, then delete the temporary resources.

### 7.4 Real /dev/kmsg User-Space Write

`GF-LIVE-KMSG-XID11-001` may only be executed on an isolated node with no running managed workload. Before writing,
record the Collector PID, restart count, boot ID, collector status and evidence baseline.

```bash
printf '%s\n' \
  '<3>NVRM: Xid (PCI:0000:59:00): 11, pid=4242, name=python, Invalid or corrupted push buffer stream' \
  | sudo tee /dev/kmsg >/dev/null
```

Do not add a `test_id` or write in a loop. New records must be identified by the boot ID, kernel sequence, monotonic
timestamp and observed time produced by the Collector. Without a workload, no workload restart, GPU reset, node reboot or
replacement may execute; the node must end Ready with no leftover taint.

This test can only prove that the production Collector read the real `/dev/kmsg`; it cannot prove that the driver or the
hardware produced the XID.

### 7.5 HyperPod Standalone XID 45 Fabric Manager Restart

A standalone XID 45 (no companion) is decided as `official_action=RESTART_FM` (the Xid catalog's Investigatory Action is
"Solo: RESTART_FM / Not Solo: IGNORE (follow other Xid)"); the shape it runs in depends on whether the control plane
grades the target node's NVLink fabric (a graded NVSwitch node: a H100/H200/H800/H20 or HGX B200/B300 product that is not
a PCIe/NVL variant, whose DCGM exporter field 170 reports a fabric state 1-4; HGX B200/B300 not live-verified), and on the
control plane's site switch `GPU_FAULT_XID45_SOLO_NVSWITCH_RESTART_FM`:

- a node that is not graded (A100, PCIe/NVL, GH200, GB200/GB300, B100 and the other catalog parts the fabric grading leaves out, a node that reports no field 170; a part in no catalog product family is NOT_APPLICABLE at the product gate and isolated first):
  whatever the switch says, the workflow is `FREEZE_EVIDENCE` → `RESTART_FABRIC_MANAGER`, a Fabric Manager restart
  only, with no isolation, no quiesce and no reset;
- a graded node with the switch unset (the default): observe-only (our deviation from the catalog's "Solo: RESTART_FM"). The decision is `MONITOR_ONLY`/`NO_ACTION` with
  `official_action` still `RESTART_FM`; the incident is recorded `RECOVERED` with its evidence, its reasons say why (with
  the graded GPU count) and how to opt in, and its `policy_reference` is the NVIDIA page below; only the XID
  investigatory notification is sent (subject `[GPU FAULT][XID 45 investigatory action] <cluster> <node>`,
  `FAULT_DETECTED`), with no workflow, no node command and no isolation. Evidence: NVIDIA GPU Debug Guidelines, GPU Node
  Triage, "Understanding Xid Messages", XID table row "45 Robust Channel Preemptive Removal"
  (https://docs.nvidia.com/deploy/gpu-debug-guidelines/gpu-node-triage.html#understanding-xid-messages, docs version
  615, page "Last updated on Oct 04, 2026"): "No action, informative only. Indicates channels affected by another
  failure. On A100, this error could be seen by itself due to unexpected Fabric Manager shutdown when FM is running in
  the same OS environment as the GPU. Otherwise, this error is safe to ignore as an informational message.";
- a graded node with the switch `true`: stopping Fabric Manager loses the fabric state and every GPU of the fabric must
  be reset after it restarts, so the workflow is the full GPU and NVSwitch reset chain: `FREEZE_EVIDENCE`,
  `MARK_UNSCHEDULABLE`, (`STOP_WORKLOADS` with an ACTIVE workload, preceded by `CHECKPOINT_WORKLOADS` when a checkpoint
  manifest exists), `QUIESCE_GPU_SERVICES` (stops Fabric Manager),
  `VERIFY_NO_GPU_CLIENTS`, `RESET_ALL_GPUS_NVSWITCHES` (every graded GPU of the node), `RESTORE_GPU_SERVICES` (starts
  Fabric Manager), `VALIDATE_GPU`, `VALIDATE_FABRIC`, `RESTORE_SCHEDULING`, (`RESTART_WORKLOAD` when the event names
  affected workloads). The `official_action` stays `RESTART_FM` and the completion notification is the GPU/NVSwitch reset
  notification.

`GF-REGIONAL-DESTR-010` verifies all three shapes: its preflight reads the control plane's own grading of the node and
what every ready `gpu-fault-control-worker` replica reads for the switch (on a graded node under the default a
disagreement refuses), records the phases in the plan, and execute refuses a node whose grading or phases moved. Under the default a graded node runs two phases: one
injection first proves observe-only (incident `RECOVERED`, a single investigatory notification, no remote command, Fabric
Manager's MainPID/InvocationID and the Agent ledger unchanged, the node never isolated), then the runner opens an
environment window `GPU_FAULT_XID45_SOLO_NVSWITCH_RESTART_FM=true` on `gpu-fault-control-worker`
(`scripts/e2e/regional/control_plane_env_window.py`, through `scripts/e2e/regional/xid45_opt_in.py`; one control-worker
roll), injects a second time after the first XID's companion window to run the full chain, and closes the window exactly
to its baseline once the workflow is terminal (a second roll); a default that did not hold opens no window and runs no
chain. While the window is open every graded node's solo XID 45 on the site runs the full chain, so no other regional
case may run at the same time. A site that opted in for good, or a control plane that predates the default, runs the
chain at once. The former `GF-LIVE-KMSG-XID45-SOLO-FM-20260724` and the mail case `GF-LIVE-KMSG-XID45-FM-EMAIL-20260724`
covered the two-step shape only; they are `SUPERSEDED` and no longer run.

Preconditions:

- a node that is not graded: `fabricManagerRestart` in the current Profile is OWNed by `gpu-fault-node-agent` and the
  allowlist contains `RESTART_FABRIC_MANAGER`;
- a graded node (also one that is observe-only first): `gpuReset` and `fabricReset` in the current Profile are OWNed by
  `gpu-fault-node-agent` and the allowlist contains `QUIESCE_GPU_SERVICES`, `VERIFY_NO_GPU_CLIENTS`,
  `RESET_ALL_GPUS_NVSWITCHES` and `RESTORE_GPU_SERVICES`; the ready `gpu-fault-control-worker` replicas agree on the
  switch;
- the Agent is ACTIVE and the artifact/config/profile pins agree;
- the Fabric Manager MainPID has been recorded; `GF-REGIONAL-DESTR-010` requires an idle node.

Write the standard XID 45 to a single target node only (the lines carry `marker=` and `drill_id=`; under the default a
graded node gets two lines, one per phase, the second after the first one's companion window plus 5 s) and wait for the
full correlation window each time. The success criteria of
the observe-only default phase are `FINALIZED` with no companion, an incident `RECOVERED` without a workflow whose reasons
carry the observe-only reason and the way to opt in, a single investigatory notification, an unchanged MainPID and an
unchanged node scheduling state. The success criteria of the full chain and of the Fabric Manager restart are
`FINALIZED` with no companion, the workflow completing every step of the node's shape, a changed MainPID, and no
`QUARANTINE`, reboot or replace. A node that is not graded is neither isolated nor quiesced, its scheduling fields match the baseline and no non-system
workload appears (the preflight requires only system Pods on the node); on a graded
node `RESET_ALL_GPUS_NVSWITCHES` runs once, Fabric Manager is stopped once and started once, every graded GPU's fabric
state is back to Success after the reset, and the node is schedulable again after `RESTORE_SCHEDULING`. The injected
lines carry `drill_id` (`<run_id>` for the chain and the Fabric Manager restart, `<run_id>-observe` for the observe-only
phase), so under the default policy the shape's fixed-template notification and the observe-only phase's investigatory
notification are recorded `SKIPPED` by the drill policy, and on a site with `GPU_FAULT_NOTIFICATION_DELIVER_DRILLS=true`
they are `SENT`. The runner judges the delivery result: `SENT` must carry a non-empty `provider_message_id`; `SKIPPED`
passes only on a notification that carries the run's own `drill_id` and whose reason is the drill policy (containing
`GPU_FAULT_NOTIFICATION_DELIVER_DRILLS`), and a `SKIPPED` for any other reason (expired, a disabled channel, ...) or on
a notification without that `drill_id` is FAIL; a result not yet recorded or still being delivered (no result, `QUEUED`,
a retrying `FAILED`) is re-read from the Store for at most 120 s and is FAIL if it is still not `SENT`; any other status
is FAIL at once. An idempotent replay must not deliver twice.

### 7.5.1 XID 45 Cross-Replica and Companion Correlation on Real Hardware

This section covers:

- `GF-LIVE-XID45-AURORA-HA-20260724`: different control-plane replicas receive XID 45 and XID 14; verifies Aurora state
  and the finalize lease;
- `GF-LIVE-KMSG-XID45-XID14-20260724`: the same node's `/dev/kmsg` first receives XID 45, then an XID 14 with the same
  PCI BDF.

Both cases must wait for the full window and confirm that XID 45 inherits the `IGNORE/NO_ACTION` of XID 14, reuses the same
incident and creates no workflow. The temporary canary and injection Pods must be deleted after the report is complete.

### 7.6 Kernel Log SXID Closed Loop

`GF-LIVE-SXID-KMSG-001` verifies, in a maintenance window, the chain from SXID 10003 in the real `/dev/kmsg` to the
all-GPU/NVSwitch reset and training recovery.

The training attempt, Pod UIDs, GPU inventory, Fabric Manager PID, Collector baseline and restart budget must be saved.
The compiled chain starts `FREEZE_EVIDENCE` → `COLLECT_DIAGNOSTIC_BUNDLE` → `MARK_UNSCHEDULABLE`, and a code listed in
`GPU_FAULT_SXID_DRIVER_REMEDIATION_CODES`/`GPU_FAULT_SXID_FIRMWARE_UPDATE_CODES` adds `REMEDIATE_DRIVER`/
`UPDATE_SOFTWARE_FIRMWARE` before the reset; the execution order must include:

```text
STOP_WORKLOADS
-> QUIESCE_GPU_SERVICES
-> VERIFY_NO_GPU_CLIENTS
-> RESET_ALL_GPUS_NVSWITCHES
-> RESTORE_GPU_SERVICES
-> VALIDATE_GPU
-> VALIDATE_FABRIC
-> RESTORE_SCHEDULING
-> RESTART_WORKLOAD
```

If any node lacks a complete GPU inventory, its generation changes, the no-client gate fails or a service cannot be
restored, the whole workflow fails closed; a partial reset is not allowed.

### 7.7 Two-Node SXID Quiesce Lease

`GF-LIVE-SXID-MULTINODE-LEASE-001` verifies that after quiesce stops kubelet, the control plane completes the two-node
barrier within a bounded maintenance window using the same persisted Agent generation.

It must cover:

1. continuing is allowed when quiesce succeeded and the generation is unchanged;
2. the heartbeat is stale while quiesce is incomplete, and the reset must not be issued;
3. past the maintenance window, the reset must not be issued;
4. if any node's generation/incarnation changes or its Agent is not ACTIVE, the whole barrier aborts.

### 7.7.1 Three-Way Concurrent Aggregation and Reset on Three-Node GPU Training

`GF-LIVE-THREE-CONCURRENT-RESET-001` uses a real three-node, 24-GPU managed PyTorchJob to verify that when a workload
observation, an XID and an SXID arrive concurrently they form only one incident, one workflow and one reset command.

Allocation or GPU UUIDs must not be faked. The success criteria include:

- all three requests succeed and the processor has no deadline exceeded;
- a single workflow completes stop, quiesce, reset, restore and GPU/Fabric validation;
- source/target GPU counts are both 24;
- all three old Pod UIDs are replaced and the new attempt recovers 24-rank NCCL;
- the restart budget increases only once;
- the nodes end Ready, schedulable and with unchanged inventory.

### 7.7.2 Two-Node XID 95 Reset and Training Resume

`GF-LIVE-XID95-DUAL-RESET-RESUME-20260724` uses a two-node, 16-GPU managed PyTorchJob. Two `synthetic=true` XID 95 events
are submitted at once through the distributed XID API, each node naming one real GPU that participates in training.

Only after the workload has fully stopped may the two barrier participants each perform the no-client check and reset.
Both must be `COMMITTED`, with `reset_gpu_uuids` in the receipt matching the request; if any participant did not PREPARE,
its generation changed or its mapping is incomplete, the whole barrier aborts. After validation passes a new attempt
starts and the restart budget increases only once.

### 7.8 Real Hardware Fault Injection Is Not Implemented

This repository provides no fault injection tool, Manifest or administrator command that makes GPU hardware or the
NVIDIA driver genuinely produce a specified XID. `GF-LIVE-004` is a `BLOCKED` capability boundary, not an executable
acceptance method.

None of the following existing capabilities can substitute for real hardware injection:

- the synthetic XID/SXID API;
- in-memory or FIFO format replay;
- user-space writes to `/dev/kmsg`;
- DCGM or NVIDIA Field Diagnostic;
- GPU reset, node reboot or warm-spare recovery.

Administrators must not download or run external injection tools that are not part of the release, not digest-pinned and
not security-reviewed in order to satisfy this case. Only after the repository formally delivers a reviewed
implementation, permission model, maintenance window, rollback and evidence contract may `GF-LIVE-004` change from
`BLOCKED` to an executable state.

### 7.9 Restart Attempt Generation Time Fence

`GF-ATTEMPT-GENERATION-FENCE-*` is a pytest scenario group verifying that a delayed XID/SXID does not repeat an action on
a new attempt. The main branches include:

- a same-rank or weaker old event only links to the original incident and adds no workflow;
- a strictly stronger old event queues only one successor;
- new events after the new attempt starts are processed normally;
- fail-safe is kept when a trusted started_at is missing;
- `source_event_time` takes precedence over the collection time;
- a node-exclusive successor waits for a still-running predecessor;
- a workload-only recovery does not inherit old node actions.

Run them by category or case ID from the catalog; no real-hardware injection is needed. The report must keep all
timestamps, predecessor/successor and action ranks.

### 7.10 HyperPod XID 74 Register Branch Verification on Real Hardware

These cases use a privileged Pod to write user-space NVRM-format records into the target node's real `/dev/kmsg`; they
verify only the parser, policy, workflow, Node Agent and notification chain and do not prove a physical NVLink fault.

```bash
python3.12 scripts/e2e/hyperpod/run_hyperpod_xid74_case.py \
  --case <operator_case> \
  --cluster-id <cluster-id> \
  --job-id <gpu-fault.io/job-id> \
  --node <kubernetes-node-name> \
  --pci-bdf <0000:bb:dd> \
  --context <kube context of the target cluster> \
  --confirm-context <the same context> \
  --live-run \
  --report "artifacts/fault/<case>-$(date -u +%Y%m%dT%H%M%SZ).json"
```

`--context` pins every kubectl call to the target cluster (`--kubeconfig` names a kubeconfig), `--confirm-context` must
repeat it exactly and `--live-run` acknowledges that the case changes a live cluster; without any of them the case
refuses to run. Support-flow cases must pass `--allow-support` explicitly and `--require-email-sent` when delivery is required; cases that
may execute a real GPU reset must pass `--allow-reset`.

Monitoring cases must not create a workflow or change Pod UIDs, the attempt, taints or GPU annotations. Reset cases must
verify that old Pods are replaced, the new attempt recovers, the node is Ready and GPU/Fabric validation passes. Every case
must use the current Profile declared by `SITE_FILE` and its actual capabilities, and must not overwrite the content of a
same-version Profile.

## 8. Overview of Fault Injection Methods

This chapter lists, by the layer at which a signal enters the system, every injection method the repository actually uses
today. It answers "how does this case manufacture the fault"; it neither replaces the per-case procedures of Chapter 7 nor
changes the safety boundaries of Chapter 2. Code references cite only files and symbols, never line numbers.

### 8.1 Seven Kinds of Injection

| Kind | Method | Signals covered |
|---|---|---|
| Forge raw logs | A privileged Pod or probe writes one NVIDIA-format record into the real `/dev/kmsg` or the Fabric Manager log | Kernel log, FM log; runs the full chain from collection to recovery |
| Create real hardware or configuration anomalies | Change real state on the node: cap the power limit, unbind the EFA driver, tamper with the expected GPU count, shadow `nvidia-smi`, hold GPU devices, stop services, out-of-band reboot | DCGM metrics, host telemetry, Node Agent checks, node status |
| Deliver synthetic events to the control-plane API | Call collector-events, distributed XID and the synthetic node-replacement endpoint with a real token, without touching the host | Policy engine and orchestration layer, bypassing the collection layer |
| Break infrastructure and network | iptables blocking 443, deleting or rolling Pods, cordon eviction, Aurora failover, secret rotation, changing Deployment environment variables | Control-plane high availability, executor protocol, timeout and recovery paths |
| Seed state in the database | Register `synthetic` clusters and write remote commands, stuck workflows and expired evidence rows directly | dispatcher, command protocol, cleanup and reconciliation logic, with no real node |
| Notification drills and in-process simulation | Mail with `drill_id`; fake DCGM endpoint, fake EFA counter directory and in-memory kmsg lines inside an isolated process | Notification delivery path, collector parsing logic, zero side effects |
| Replay raw readings on a real node (live-replay) | Replace only one raw reading a product collector reads: DCGM exporter fields, procfs/sysfs files or trees, `smartctl`/`ipmitool`/`ethtool`/`nvidia-smi` output, kernel or journal records, a CRI Pod log, a flight recorder dump; bounded, reversible, restored by a deadman | The live-replay variants of GF-GENERAL-LIVE and GF-ENV-LIVE: collection, grading, orchestration and recovery are all the product's own; they prove the processing chain, never a physical fault |

The conclusion of Section 7.8 stands: none of the above can make GPU hardware or the driver genuinely produce a specified
XID. The approvals, windows and verdicts of every GF-GENERAL-LIVE and GF-ENV-LIVE fixture are in
[general fault live acceptance](../components/general-fault-live-acceptance.md) and
[GPU environment live acceptance](../components/gpu-environment-live-acceptance.md).

### 8.2 Common Preconditions

Apart from the XID 74 runner's own kmsg Pod (see 8.3), all node-level injections enter the host through the privileged Pod (privileged, hostPID, hostNetwork, hostPath root)
created by `scripts/e2e/regional/host_probe_fixture.py::HostProbeFixture`, chroot and run the probe subcommand with the
venv of the node Agent's current signed slot (`/opt/gpu-fault/current/venv/bin/python`; a node without the Agent falls
back to the system `python3`, the probe scripts being stdlib-only). Every injection that changes node state first schedules a recovery timer with
`systemd-run` and only then applies the change; Manifest-style injections must first label the target node with
`gpu-fault.io/e2e-target-a=true` or `-b=true`, see
`scripts/e2e/regional/manifests/fault-injection/README.md`.

The two markers have different semantics and must not be mixed in reports:

- `synthetic` marks objects or events constructed by the test (`XidEvent`, `SxidEvent`, cluster registrations, synthetic
  node-replacement findings); the control plane uses it to skip expired synthetic clusters and to use the dedicated
  `policy_source` for synthetic node replacement;
- `drill_id` is parsed from the kmsg line text and affects only notification delivery: by default it is suppressed by
  `GPU_FAULT_NOTIFICATION_DELIVER_DRILLS` and given a `[DRILL:..]` subject marker; it does not change actual actions such as
  isolation, reset or reboot.

The only synthetic feature switch in the whole repository is `GPU_FAULT_ENABLE_SYNTHETIC_REPLACEMENT_TESTS`, and it must be
combined with the execution token. No CLI has an inject or simulate subcommand; all injection is done by e2e probe scripts
or control-plane routes.

The managed training fixtures keep their GPUs busy between two heartbeats. The four long-running heartbeat jobs under
`scripts/e2e/regional/manifests/training/` (`xid11-three-node-pytorchjob.yaml`, `two-node-replace-pytorchjob.yaml`,
`single-node-warm-spare-pytorchjob.yaml`, `xid11-single-node-job.yaml`) run fp16 matmuls on every GPU they hold for
`GPU_FAULT_ACCEPTANCE_GPU_DUTY_PERCENT` (default `30`, range `0` to `50`) of every 10 ms period, call
`torch.cuda.synchronize` after every eight, and stop at the deadline of the ~2 s heartbeat interval: the `HEARTBEAT`/`SUCCESS`
lines, the ~2 s cadence, the step counts and the total duration are unchanged, the load holds about 24 MiB per GPU and
draws far less than the power limit by design (not a hardware measurement), and SIGTERM finishes the batch in flight,
synchronizes and exits 143 (with the main thread stuck in a collective the signal thread exits 143 itself after 1 s), so a stop
leaves no kernel running on the GPU. The reason: a real training job keeps kernels running, while a heartbeat that only
all-reduces one value and sleeps reads about 0 % `utilization.gpu` in `nvidia-smi`, which the product's
`LOW_GPU_UTILIZATION` rule (any GPU a live managed attempt holds on the node at or below 5 % for 300 s by default,
each GPU graded on its own) correctly reports in any case that runs that long. The 10 ms period is far shorter than the shortest sample period
(1/6 s), so every sample reads about the duty. `0` restores the idle heartbeat; a case that needs idle GPUs sets it in its
rendered manifest (`scripts/e2e/regional/acceptance_gpu_load.py::set_gpu_duty_percent`). The NOTIFY-005 CPU-only load,
the warm-spare GPU holder Pod and the late-ownership CUDA holder stay GPU-idle by design; the one-GPU
`xid11-single-node-job.yaml` keeps only its own GPU busy, and the node's seven other GPUs, which the attempt does not hold,
are not graded; and
COLLECT-021 deliberately SIGKILLs the training processes, which can land inside a batch. The copies in the manifests
equal the canonical text in `scripts/e2e/regional/acceptance_gpu_load.py`, checked by
`python3 -m scripts.e2e.regional.acceptance_gpu_load` (`--write` rewrites the copies).

Evidence: NVIDIA nvidia-smi documentation, GPU ATTRIBUTES > Utilization > GPU
(https://docs.nvidia.com/deploy/nvidia-smi/index.html, nvidia-smi v610 Update, newest change-log entry "Changes between
nvidia-smi v610 Update and v595"): "Percent of time over the past sample period during which one or more kernels was
executing on the GPU. The sample period may be between 1 second and 1/6 second depending on the product." Evidence for
leaving no kernel running at a stop: NVIDIA Xid catalog (docs version 615,
https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html, "Xids" sheet of the linked Xid-Catalog.xlsx, code 45
Trigger Conditions): "This event is logged when the user application aborts and the kernel driver tears down the GPU
application running on the GPU. Control-C, GPU resets, sigkill are all examples where the application is aborted and this
event is created."

### 8.3 Node Kernel and Driver Log Layer

| Injection method | Real signal produced | Mechanism | Typical cases | Code reference |
|---|---|---|---|---|
| Write an XID line to `/dev/kmsg` | One `NVRM: Xid (PCI:..): <n>` kernel log line, annotated with `marker=` or `drill_id=` | Direct write via `os.open("/dev/kmsg")`; each probe keeps an XID allowlist | DESTR-001/002/010/014/015/016/017/018/019, DESTR-023..029, HA-003/004, COLLECT-008/009/010/012/013/016/021, E2E-001 | `scripts/e2e/regional/probes/destructive_node_probe.py::write_xid`, `scripts/e2e/regional/probes/collector_node_probe.py::write_xid`, `scripts/e2e/regional/probes/e2e001_node_probe.py::write_xid11`, `scripts/e2e/regional/probes/node_host_probe.py::write_xid45` |
| XID written by a systemd timer after quiesce | XID 46 or 79 written on schedule once kubelet is stopped and the exec channel is unavailable | `systemd-run --on-active` schedules the probe's write-xid | DESTR-016 | `scripts/e2e/regional/probes/destr016_node_probe.py` |
| Write user-space annotated lines | A monitor-only XID 63, or a deliberately malformed Xid line missing its code | Direct write via `os.open("/dev/kmsg")`, kind limited to `xid63` and `unparsed-xid` | NET-001, COLLECT-018, NET-008 | `scripts/e2e/regional/probes/net001_node_probe.py::write_kmsg`, `scripts/e2e/regional/probes/collector_window_probe.py::write_kmsg` |
| Privileged busybox Pod writes kmsg | Single-node XID combinations (45 then 14, 63 then 48, 54) and the XID 74 register variants | busybox `printf > /host-dev-kmsg` with hostPath `/dev/kmsg` (CharDevice); a Manifest's target node must carry `gpu-fault.io/e2e-target-a=true`, the XID 74 runner builds its own Pod pinned by nodeName (not through HostProbeFixture) | GF-LIVE-KMSG-XID45-XID14-20260724, the by-hand forms of COLLECT-008/009, the GF-LIVE-XID74 series | The `scripts/e2e/regional/manifests/fault-injection/` directory, `scripts/e2e/hyperpod/run_hyperpod_xid74_case.py` |
| Append an SXID line to the Fabric Manager log | A `nvidia-nvswitchN: SXid (PCI:..): <sxid>, <Fatal/Non-fatal>, Link <port> …` record (prefix `NVSwitch ` with `--no-include-switch`) | Append write plus fsync; SXID allowlist | COLLECT-005/011/014/015 | `scripts/e2e/regional/probes/collector_node_probe.py::append_sxid` |
| Replay kernel-source records through the product's readers | A trusted `_TRANSPORT=kernel` record read by the node log collector (a non-kernel transport stays untrusted), or an XID/SXID record in `/dev/kmsg` syntax read by the kernel collector | Journal mode: a `/run` drop-in puts a pass-through `journalctl` wrapper first on the PATH of `gpu-fault-log-collector.service` only, which after `arm` appends the reviewed JSON lines after the real output; kmsg mode: a drop-in points the kernel collector's `GPU_FAULT_KMSG_PATH` at a reviewed file under `/run` and `arm` appends the records; the real `/dev/kmsg` is never written, a `systemd-run` deadman restores | GF-GENERAL-LIVE-016, 017 (journal variants), 020 (`independent_fatal_event`), 024 (`independent_xid` and `xid_twin_*`), 029 (`hardware_chain_continues`), 032 | `scripts/e2e/regional/probes/general_live_kernel_source_probe.py` |
| Give the node log collector one reviewed CRI Pod log | A managed attempt's CUDA OOM text and a collective-timeout line written in two parts, with the real Pod UID and container name (the cgroup OOM, NCCL and torch.distributed texts are journal replays, the row above) | A `/run` drop-in makes `GPU_FAULT_TRAINING_LOG_PATHS` of `gpu-fault-log-collector.service` match only this run's fixture file and runtime-enables the unit; the fixture file lives in the real log directory kubelet made for the runner's own Pod | GF-GENERAL-LIVE-017 | `scripts/e2e/regional/probes/general_live_applog_probe.py` |

### 8.4 Node Hardware and Configuration Layer

| Injection method | Real signal produced | Mechanism | Typical cases | Code reference |
|---|---|---|---|---|
| Lower the GPU power cap and run at full load | Real power violation combined with high utilization | `nvidia-smi -pl` plus `dcgmproftester`, restored on a timer | COLLECT-002 | `scripts/e2e/regional/probes/collector_node_probe.py::throttle_gpu` |
| Unbind the EFA driver | `efa_inventory_mismatch` with failure_mode `DRIVER_UNBOUND` | Write `/sys/bus/pci/drivers/efa/unbind`, rebind on a timer | COLLECT-017, DESTR-021, NET-007 | `scripts/e2e/regional/probes/collector_node_probe.py::unbind_efa` |
| Tamper with the collector's expected GPU count | On a type in the instance table the control plane grades it as a configuration contradiction: WARNING `gpu_expected_count_contradicted`, evidence only, no reboot; only a type outside the table still goes `gpu_inventory_mismatch` → reboot | Change `/etc/gpu-fault/collector.env` to the current value plus one, restored on a timer (no runner takes this path any more: COLLECT-004 uses the next row, GF-GENERAL-LIVE-012 `expectation_wrong` the env mode of the host-collector source replay; old override journals are still cleaned up by their protocol) | (none) | `scripts/e2e/regional/probes/collector_node_probe.py::override_expected_gpu_count` |
| One real UUID fewer in the isolated collector instance's `nvidia-smi` listing | `gpu_inventory_mismatch` (expected = configured, observed = expected-1), genuinely reaching REBOOT_NODE on a type in the table; the production collector, its configuration and the node's listing are untouched | The probe's `DroppedUuidListing` as the isolated `HostTelemetryCollector`'s runner drops the same real UUID row on both ticks (the GF-GENERAL-LIVE-012 `drop_row` idiom); refused when the real listing lacks the configured number of UUIDs | COLLECT-004 | `scripts/e2e/regional/probes/collector_node_probe.py::sample_gpu_inventory` |
| Shadow `nvidia-smi` | Collection timeout, or one GPU missing from the first inventory | A systemd drop-in puts the shadow script first on PATH | COLLECT-019, COLLECT-020 | `scripts/e2e/regional/probes/collector_window_probe.py::open_window` |
| Collector environment window | 403 from a wrong cluster token (NET-008, kernel collector), a missing expected GPU count (COLLECT-020); the same window also appends `GPU_FAULT_FILESYSTEMS` and `gfrs*` `GPU_FAULT_REQUIRED_INTERFACES` for the GF-GENERAL-LIVE storage and NIC variants | EnvironmentFile or UnsetEnvironment in a systemd drop-in, deadman timer closes the window | NET-008, COLLECT-020, GF-GENERAL-LIVE-004/008/009 | `scripts/e2e/regional/probes/collector_window_probe.py::open_window` |
| Seed an outbox dead-letter record | Replay hits a retired channel and returns 404 | Append ndjson under `/var/lib/gpu-fault/outbox/` | NET-008 | `scripts/e2e/regional/probes/collector_window_probe.py::seed_outbox_record` |
| Hold a GPU device file | `VERIFY_NO_GPU_CLIENTS` stays WAITING | A `systemd-run` transient unit opens `/dev/nvidiaN` and sleeps | DESTR-014/016/017/018/025 | `scripts/e2e/regional/probes/destr014_node_probe.py` and sibling probes |
| SIGKILL the training processes | A managed attempt's critical container processes exit, the Pod is not deleted | SIGKILL of the training processes in the Pod cgroup through the host PID namespace | COLLECT-021 | `scripts/e2e/regional/probes/collector_node_probe.py::kill_workload` |
| Out-of-band reboot | A node reboot that bypasses the HyperPod API | `systemd-run --on-active` runs `systemctl reboot` after a delay | DESTR-017 | `scripts/e2e/regional/probes/destr017_node_probe.py::arm_reboot` |
| Disable or restart the Node Agent, stop warm-spare services | The hot spare becomes unavailable (DESTR-008), the Agent does not come back after a reboot (DESTR-014), the Agent restarts (DESTR-019) | DESTR-008 arms the recovery timer first, then `systemctl stop`s the units of its service window; DESTR-014, after arming a persistent recovery safeguard that survives a reboot, removes only the Node Agent's enable link and leaves the running Agent alone; DESTR-019 `systemctl restart`s the Agent behind a systemd-run fail-safe start | DESTR-008, DESTR-014, DESTR-019 | `scripts/e2e/regional/probes/warm_spare_node_probe.py::stop_with_failsafe`, `scripts/e2e/regional/probes/destr014_recovery_probe.py::Recovery`, `scripts/e2e/regional/probes/destr019_node_probe.py::restart_agent` |
| Real quiesce and restore cycle | Stop every unit of `DEFAULT_QUIESCE_SERVICES` and container of `DEFAULT_QUIESCE_CONTAINERS`, then restore | Call the Node Agent's `GpuServiceQuiesceManager` | PREEMPT-012 | `scripts/e2e/regional/probes/preempt012_node_probe.py`, `DEFAULT_QUIESCE_SERVICES` in `src/gpu_fault/node_agent/common.py` |
| Real low-utilization training load | The `LOW_GPU_UTILIZATION` sustained rule (`LOW_CPU_UTILIZATION` stays silent) | Submit a CPU-only burn job that holds all 8 GPUs of each node and creates no CUDA context (a `batch/v1` Job on one node, a PyTorchJob on several) | NOTIFY-005 | `scripts/e2e/regional/notify005_checks.py::low_utilization_manifest` |
| Replay the host collector's raw readings | Variants of the CPU, memory, NIC and EFA counter, PCI/InfiniBand tree, `/proc/driver/nvidia`, SMART, BMC, `ethtool` and `nvidia-smi` readings | Only inside the mount namespace of `gpu-fault-host-collector.service` (the command mode can also open on `gpu-fault-kernel-collector.service` under the `KERNEL_COLLECTOR_SOURCE_REPLAY` approval, for 032 `missing_version`): `BindReadOnlyPaths=` over an allow-listed file (bind) or a whole directory (tree, the real tree copied with reviewed mutations), a pass-through command wrapper that rewrites only allow-listed fields (command, `tools/host_command_replay.py`), a file or tool made unopenable (inaccessible), and for that unit only an expected-GPU-count override (env, 012 `expectation_wrong`) or a fixture PCI tree (pci-root, 013); a `/run` drop-in plus a deadman restores | GF-GENERAL-LIVE-001/002/007/008/009/010/011/012/013/015/032 | `scripts/e2e/regional/probes/general_live_source_probe.py`, `scripts/e2e/regional/probes/general_live_source_builders.py` |
| Replay DCGM exporter fields and the GPU inventory reading | Memory ECC/remap, NVLink counters, fabric state, PCIe, composites, field-structure gaps and GPU identity changes | A loopback replay tool on the node (`tools/gpu_dcgm_field_replay.py` with its per-category profiles; `tools/gpu_environment_replay.py` for the thermal/power parts) and a `/run` drop-in pointing the single metrics collector at it; the inventory variants add a pass-through `nvidia-smi` wrapper (`tools/nvidia_smi_inventory_replay.py`); a deadman restores | GF-GENERAL-LIVE-019/020/021/022/024/025/026 | `scripts/e2e/regional/probes/general_live_dcgm_probe.py`, `scripts/e2e/regional/general_live_gpu_replay.py` |
| Replay temperature, power and throttle readings | GPU and memory temperature, power limit, thermal violation time, Power Brake and other DCGM readings | The loopback proxy `tools/gpu_environment_replay.py` passes the real exporter through (`127.0.0.1:9400` → `127.0.0.1:9411`) and injects only after `arm`; the deadman is armed first, then a `/run` drop-in points `gpu-fault-metrics-collector.service` at the proxy | GF-ENV-LIVE-001..012, GF-GENERAL-LIVE-024 (`thermal_warning`/`power_performance`) | `scripts/e2e/regional/probes/env_replay_proxy.py`, `scripts/e2e/regional/run_env_live.py` |
| Bounded host load and producer pauses | CPU saturation, memory pressure, page cache, low utilisation; collector silence; a DCGM collection gap | Only transient systemd units named after the run digest and bounded by `RuntimeMaxSec` (CPU burners, a reader of a sparse file filling the page cache, a capped memory allocator); a SIGSTOP of every process of one allow-listed collector unit (a `/run` drop-in stops a restarted main process too) or of the node's DCGM exporter process, resumed by a deadman; a runtime enable of the node log collector behind an approval | GF-GENERAL-LIVE-001/002/003/025/030 (`stale_metrics`)/031 | `scripts/e2e/regional/probes/general_live_node_probe.py`, `scripts/e2e/regional/probes/general_live_node_pause.py` |
| Test file systems and block devices | Capacity and inode exhaustion, read-only, a vanished mount, statvfs errors, IO saturation and latency | A loop file system or a loop + device-mapper device mounted under `/mnt/gfrs-live/<digest>/`, a bounded O_DIRECT writer unit, the `probe_error` fault mapping; recorded before created, undone by a deadman | GF-GENERAL-LIVE-004, 006 | `scripts/e2e/regional/probes/general_live_storage_probe.py` |
| Test NFS exports | Shared-storage timeouts, full, vanished mounts, a cross-node failure domain | A userspace NFSv4 server Pod (memory-backed emptyDir) serves the export; the node mounts it `soft` under `/mnt/gfrs-live/<digest>/` in PID 1's mount namespace, a deadman unmounts | GF-GENERAL-LIVE-005 | `scripts/e2e/regional/shared_export_fixture.py`, `scripts/e2e/regional/probes/general_live_shared_probe.py` |
| Test network interfaces | A required interface down or missing, an unknown state, error and drop deltas, TCP-only loss, counter resets | Only `gfrs<digest><slot>` test interfaces (a drop generator on a veth pair, a `dummy` interface, a netns peer with `tc netem` loss), declared required to the collector by a collector window; an interface with a routable address or a real device is always refused | GF-GENERAL-LIVE-008, 009 | `scripts/e2e/regional/probes/general_live_net_probe.py` |
| EFA driver rebind failure | The bind failure and the post-repair validation failure of `REMEDIATE_EFA_DRIVER` | Before one allow-listed EFA function is unbound its `driver_override` is set to a driver name no module provides, so the kernel refuses to bind it to `efa` again; the post-repair variant sets no override and unbinds the function once more after the product's bind, when the product's post-repair snapshot restarts the host collector (a read-only watch of that unit's state); a deadman restores | GF-GENERAL-LIVE-013 | `scripts/e2e/regional/probes/general_live_efa_probe.py` |
| Freeze a device plugin | `gpu_device_plugin_unresponsive`/`efa_device_plugin_unresponsive` (invisible to kubelet itself) | A SIGSTOP of the plugin container's own process: the socket stays open, the Pod stays Running, allocatable unchanged; a deadman sends SIGCONT | GF-GENERAL-LIVE-014 | `scripts/e2e/regional/probes/general_live_device_plugin_probe.py` |
| Temporary test packages | None (only makes a variant's tool or kernel module usable: `tc`/`sch_netem`, `smartmontools`, `dosfstools`) | Only the reviewed packages and their closures, previewed with `--assumeno` first; what this run installed is removed afterwards | GF-GENERAL-LIVE-004, 007, 009 | `scripts/e2e/regional/probes/general_live_deps_probe.py` |

### 8.5 Control-Plane API Layer

| Injection method | Real signal produced | Mechanism | Typical cases | Code reference |
|---|---|---|---|---|
| API replay of an XID | A synthetic XID event without touching the host | Deliver to `/v1/collector-events/nvidia-kernel` with the deployed `HttpEventSink` from inside an executor Pod | DESTR-009/012/020, E2E-002, ISO-001/006/007 | `scripts/e2e/regional/regional_live_fixture.py::RegionalLiveFixture.post_xid_event` |
| API replay of an SXID | A synthetic SXID event | As above, path `/v1/collector-events/fabric-manager` | COLLECT-014 | `scripts/e2e/regional/run_collector_destructive.py::post_fabric_event` |
| Distributed XID batch | A multi-node XID batch with `synthetic=true` | Direct POST to `/v1/gpu-events/xid/distributed` | The case of Section 7.7.2 | `src/gpu_fault/app/routes/gpu_events.py::evaluate_distributed_xids` |
| Synthetic node-replacement finding | A CRITICAL `REPLACE_NODE` that drives a hot-spare switch | POST `/v1/admin/test/node-replacement`, requires the switch and the execution token | DESTR-003, DESTR-008 | `src/gpu_fault/app/routes/collector_events.py::inject_test_node_replacement`, `scripts/e2e/regional/warm_spare_fixture.py::WarmSpareLiveFixture.post_synthetic_replacement`, `scripts/e2e/regional/synthetic_replacement_route.py` |
| Submit an event with an illegal field | Enqueued successfully but rejected by the processor, producing a rejected event | Deliver a payload with an unknown field through the real sink inside the node venv | COLLECT-018 | `scripts/e2e/regional/probes/collector_window_probe.py::post_rejected_event` |
| Redeliver the product's own recorded terminal | The same TerminalEvent arriving a second time | In an API Pod, under the product's interpreter, the stored TerminalEvent of a test attempt is handed to `CompletionService.handle_terminal` (the entry of the `/v1/attempts/terminal` route) | GF-GENERAL-LIVE-028 (`old_terminal`), 029 (`duplicate_event`) | `scripts/e2e/regional/general_live_terminal_replay.py` |

### 8.6 Kubernetes and Network Layer

| Injection method | Real signal produced | Mechanism | Typical cases | Code reference |
|---|---|---|---|---|
| Adversarial node metadata mutation | Delete the hot-spare label, change the instance group, write stale or foreign annotations, roll back the reservation time, high-frequency patches producing 409 | kubectl patch, restored exactly after recording a baseline | DESTR-008/021/022 | `scripts/e2e/regional/run_destr008_warm_spare_shortage.py::ScenarioFixture`, `scripts/e2e/regional/probes/destr021_annotation_writer.py::patch_command`, `scripts/e2e/regional/run_destr022_spare_reservation_reclaim.py` |
| Hold a GPU on the hot spare | The hot spare is judged unavailable | Create a Pod requesting `nvidia.com/gpu` | DESTR-008 | `scripts/e2e/regional/warm_spare_fixture.py::GpuHolderFixture` |
| iptables blocking control-plane 443 | Collector, executor and control plane disconnected | `iptables` rejects 443 and rolls back on a timer | NET-001/008, ISO-006 | `scripts/e2e/regional/probes/net001_node_probe.py`, `scripts/e2e/regional/probes/collector_window_probe.py::block`, `scripts/e2e/regional/probes/cluster_network_probe.py::block` |
| Executor proxy blocking or packet loss | Lease renewal fails, retry after a lost result | A real ClusterActionExecutor with a loopback proxy running inside a Pod | NET-002/003/006 | `scripts/e2e/regional/probes/net002_executor.py`, `scripts/e2e/regional/probes/net003_executor.py`, `scripts/e2e/regional/probes/net006_executor.py` |
| Delete or roll control-plane and executor Pods | Process-level failover and takeover | `kubectl delete pod`, `rollout restart`, `scale` | HA-001/004/005/006/010, DESTR-024 (the watcher scaled to 0) | `scripts/e2e/regional/run_ha001_control_plane_failover.py` and sibling runners |
| cordon plus PDB eviction | Topology and availability constraint verification | Cordon a CPU node, then call the Eviction API | HA-002 | `scripts/e2e/regional/run_ha002_pdb_topology.py` |
| Deployment environment variable window | Compress workflow lifetimes, lower validation counts, turn off the dispatcher, the XID 45 opt-in | `kubectl patch deployment --type=json` (uid/resourceVersion `test` operations guard concurrency, the full env goes through stdin), with a compiled-in variable allowlist and a required confirmation string (`OPEN_/CLOSE_EXECUTOR_ENV_WINDOW`, `OPEN_/CLOSE_CONTROL_PLANE_ENV_WINDOW`) | DESTR-010/014/018/019/025, HA-004, PREEMPT-037 | `scripts/e2e/regional/deployment_window_guard.py`, `ALLOWED_VARIABLES` in `scripts/e2e/regional/executor_env_window.py`, `scripts/e2e/regional/control_plane_env_window.py` |
| Single-variable mutation of a derived guard-probe Deployment | A boot-guard refusal | `boot_guard/derive.sh` derives a one-off guard-probe Deployment (isolated database), `mutate.py` changes exactly one env variable (del/set/sref), run as a batch by `run_regional_boot_guard_cases.sh --confirm BOOT_GUARD_EXECUTE` | BOOT-001..010 | `scripts/e2e/regional/boot_guard/mutate.py`, `scripts/e2e/regional/run_regional_boot_guard_cases.sh` |
| A temporary webhook that refuses node updates | The executor's node patches answer HTTP 500 | A ValidatingWebhookConfiguration matching only the target node's UPDATE by the executor service account, `failurePolicy=Fail`, pointing at a Service that does not exist; a deadman Job removes it | NET-007 | `scripts/e2e/regional/run_net007_transient_api_outage.py`, `scripts/e2e/regional/probes/net007_deadman.py` |
| Host-level network isolation | The completion watcher's POSTs to the control plane fail at once; the executor cannot push to the Node Agent | Tagged `REJECT --reject-with tcp-reset` rules: FORWARD from the watcher Pod's IP to the control-plane endpoint, or INPUT to the Agent's listening port on a GPU node; the intent is recorded and the deadman armed before the rules go in | GF-GENERAL-LIVE-028 (`api_unreachable`), 011 (`unreachable_node`) | `scripts/e2e/regional/probes/general_live_isolation_probe.py` |
| Managed test jobs | A critical container's non-zero exit, stop or hang; training progress NaN, stall or a silent reporter; a two-node NCCL stall | Disposable jobs submitted through `ManagedWorkloadFixture`: every replica of the failing job runs the variant's exit/stop/hang command; the progress job runs a `gpu-fault-collector training-progress` sidecar in the training Pod over the progress file the test application writes; the NCCL job's rank 0 stops on the runner's pause file, so every rank blocks in the next all-reduce | GF-GENERAL-LIVE-011, 027, 028, 029 | `scripts/e2e/regional/general_live_fail_job.py`, `scripts/e2e/regional/general_live_progress.py`, `scripts/e2e/regional/general_live_nccl_job.py` |

### 8.7 Cloud Service and Data Layer

| Injection method | Real signal produced | Mechanism | Typical cases | Code reference |
|---|---|---|---|---|
| Aurora writer failover | Database primary switch | `aws rds failover-db-cluster` | HA-003, HA-010 | `scripts/e2e/regional/run_ha003_aurora_failover_reset.py`, `scripts/e2e/regional/run_ha010_aurora_blackout_liveness.py` |
| Aurora master password rotation | Credential invalidation and refresh | `aws rds modify-db-cluster --rotate-master-user-password --apply-immediately` rotates the RDS-managed master secret, then the credential-refresh Job runs from the `gpu-fault-aurora-credential-refresh` CronJob (a watchdog backs it up) | HA-009 | `scripts/e2e/regional/run_ha009_aurora_credential_rotation.py` |
| Synthetic cluster seeded commands | Register a `synthetic` cluster and seed remote commands directly, claimed by a probe executor | Written through the Store API, no real node | CMD-017/018, NET-002/003/006, AUTH-016 | `scripts/e2e/regional/seeded_command_fixture.py::seed_command`, `scripts/e2e/regional/seeded_command_fixture.py::register_synthetic_cluster` |
| Isolated database seeds | Stuck workflows | A separate Postgres container or temporary SQLite, refusing the ambient store URL | PREEMPT-036 | `scripts/e2e/regional/run_preempt036_stuck_workflow_reconcile.py` |
| Synthetic audit rows | Expired evidence rows | From inside a control-worker Pod, only `audit-*` expired rows of the synthetic `audit-cluster` are written to the production store and deleted in `finally` | PREEMPT-038 | `scripts/e2e/regional/audit_raw_evidence_periodic_cleanup.py` |
| An unclaimable seeded command | One remote_command no owner can claim, `created_at` set back 610 s | Written into the running control plane, never claimed, deleted in `finally` | BOOT-015 | `scripts/e2e/regional/boot_acceptance_runtime.py::run_boot015` |

### 8.8 Notification and In-Process Simulation Layer

| Injection method | Real signal produced | Mechanism | Typical cases | Code reference |
|---|---|---|---|---|
| drill notification | A real mail with the `[DRILL:..]` marker that triggers no action | Built with the email builder and sent through the site's configured notification channel (`GPU_FAULT_NOTIFICATION_CHANNEL`: sns by default, ses optional) | NOTIFY-001/007 (NOTIFY-002 is SUPERSEDED) | `scripts/e2e/regional/probes/notification_drill.py::build_notification`, `scripts/e2e/regional/probes/notify007_delivery_drill.py` |
| In-process synthetic signals | Fake DCGM metrics endpoint, fake EFA counter directory, in-memory kmsg lines | A local HTTP server and temporary directories fed to the deployed collector, then delivered to an isolated control plane; the in-memory kmsg replay of section 7.1 runs in pytest (`tests/collectors/test_xid_kmsg_catalog_replay.py`) | Section 7.1 and the hyperpod E2E | `scripts/e2e/hyperpod/run_hyperpod_dcgm_metrics_e2e.py`, `scripts/e2e/hyperpod/run_hyperpod_efa_traffic_e2e.py`, `scripts/e2e/hyperpod/run_hyperpod_three_source_fault_e2e.py` |
| Flight recorder responder inside the managed job | Flight recorder dumps shaped by the variant: progress that advances, an operation left pending, enqueue without completion, a rank one collective behind, a withheld second dump | The two-node NCCL stall job runs with `TORCH_NCCL_TRACE_BUFFER_SIZE=0`; the responder owns every rank's FIFO and writes a dump on each `COLLECT_HUNG_TRIAGE` request; the FIFO protocol, namespace mapping and parsing are the product's | GF-GENERAL-LIVE-011 | `scripts/e2e/regional/probes/general_live_fr_responder.py`, `scripts/e2e/regional/general_live_fr_replay.py` |

### 8.9 Maintenance Rules

The HyperPod software harness uses the complete fake DCGM field set, duration counter inputs are in ns, and policy time
advances in explicit 15-second samples; per-sample policy replay is reported separately from the sustained Collector
edge-filter verification of `DCGM-E2E-013`, and the former must not be passed off as a confirmation/suppression test. The
old and new exporter names of one field (`CLOCKS_EVENT_REASONS`/`CLOCK_THROTTLE_REASONS`,
`GPU_TEMP_MARGIN_CELSIUS`/`GPU_TEMP_LIMIT`) carry the same value (the harness derives every old/new exporter name pair
from `DCGM_METRICS`; these are two of them) and the Collector keeps one sample. In `DCGM-E2E-013` a
warning temperature still needs the three-sample edge confirmation; a critical temperature candidate is uploaded on every
cycle and the control plane confirms it as CRITICAL on the third comparable sample, without re-alerting afterwards. Thermal,
NVLink-counter and correctable-memory correlations only request diagnostics, and a cumulative counter that drops is not a
recovery. The three-source harness first establishes an EOF baseline for the private FM log and then appends the SXID, binds
responses by channel and batch, and does not read the host GPU inventory.
The current DAG for EFA hung traffic is `FREEZE_EVIDENCE -> COLLECT_HUNG_SUMMARY(BASELINE) ->
COLLECT_HUNG_SUMMARY(RECHECK) -> COLLECT_HUNG_TRIAGE(first wave) -> COLLECT_HUNG_TRIAGE(expansion) ->
COLLECT_DIAGNOSTIC_BUNDLE`, with `VALIDATE_FABRIC` in parallel after RECHECK and `ESCALATE_SUPPORT` after both; only the two
TRIAGE waves capture process state. It is not the removed completion-phase quick-triage.
All of these verify only software inputs and an isolated control plane; they are not proof of hardware damage or
real-hardware recovery.

- When adding an injection method, update the tables in this chapter first, then the corresponding case's `injection` field; the two descriptions must agree.
- Implement injection only inside dedicated probe subcommands, with an allowlist, recovery timer and state snapshot; never write injection into deployment manifests or the production CLI.
- For injections that change node state, the report must record whether the recovery timer fired and whether the node returned to baseline.
