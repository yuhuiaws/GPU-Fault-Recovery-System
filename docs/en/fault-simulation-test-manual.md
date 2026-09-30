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
| `staging` | Controlled acceptance in an isolated HyperPod test cluster |
| `live` | Manual or explicit command acceptance in a real HyperPod environment |

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
| `destructive-provider-reboot` | HyperPod reboot of the original node | `NodeRecovery=None`, secondary cluster confirmation and provider idempotency |
| `destructive-warm-spare` | `HEALTHY_WARM_SPARE_ONLY` local switch | Uses only managed healthy spares; the faulty instance stays in existence and isolated |

`destructive-provider-replace` is neither a legal risk nor an executable test path.
`BatchReplaceClusterNodes` may only serve as a refutation target: that call must not appear anywhere in the window; when
warm spares are insufficient it must fail and alert, never fall back to provider replace.

Every real-hardware test must:

- state the target Region, context, cluster ID, nodes and maintenance window explicitly;
- confirm HyperPod `NodeRecovery=None`;
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
its resource locks may run in parallel.

CAP-005 uses a random temporary database but still shares the CPU, I/O, connections, WAL and lock manager of the same
PostgreSQL/Aurora server, so it declares:

```yaml
execution:
  parallel_safe: true
  environment: inherit
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

The runner starts a separate process for every pytest or command case, and any failure makes the final process return
non-zero. Every result contains:

- case ID, level, risk, problem, injection method and expectation;
- `PASS`, `FAIL` or `NOT_RUN`;
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
| `FAIL` | At least one executed case failed |
| `NOT_RUN` | No case actually executed |

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
- manual destructive acceptance such as reset, reboot and warm spare.

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

`verify` must confirm that the Profile is registered, has no warnings, and agrees with the CPU side, Collector, Watcher,
Installer and node Agent. Real-hardware reports are written to `artifacts/fault/` or the private evidence store and do
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

`GF-LIVE-KMSG-XID45-SOLO-FM-20260724` and the mail case verify the complete `RESTART_FABRIC_MANAGER` chain for an
XID 45 without a companion.

Preconditions:

- `fabricManagerRestart` in the current Profile is OWNed by `gpu-fault-node-agent`;
- the Agent is ACTIVE and the artifact/config/profile pins agree;
- the allowlist contains `RESTART_FABRIC_MANAGER`;
- the training heartbeat is normal and the Fabric Manager MainPID has been recorded.

Write one standard XID 45 to a single target node only and wait for the full correlation window. The success criteria are
`FINALIZED` with no companion, the workflow completing `FREEZE_EVIDENCE` and `RESTART_FABRIC_MANAGER`, a changed
MainPID, and unchanged training Pods and attempt. With mail enabled, the fixed-template notification must also be verified
as `SENT` and an idempotent replay must not deliver twice.

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
The execution order must include:

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
  --report "artifacts/fault/<case>-$(date -u +%Y%m%dT%H%M%SZ).json"
```

Support-flow cases must pass `--allow-support` explicitly and `--require-email-sent` when delivery is required; cases that
may execute a real GPU reset must pass `--allow-reset`.

Monitoring cases must not create a workflow or change Pod UIDs, the attempt, taints or GPU annotations. Reset cases must
verify that old Pods are replaced, the new attempt recovers, the node is Ready and GPU/Fabric validation passes. Every case
must use the current Profile declared by `SITE_FILE` and its actual capabilities, and must not overwrite the content of a
same-version Profile.

## 8. Overview of Fault Injection Methods

This chapter lists, by the layer at which a signal enters the system, every injection method the repository actually uses
today. It answers "how does this case manufacture the fault"; it neither replaces the per-case procedures of Chapter 7 nor
changes the safety boundaries of Chapter 2. Code references cite only files and symbols, never line numbers.

### 8.1 Six Kinds of Injection

| Kind | Method | Signals covered |
|---|---|---|
| Forge raw logs | A privileged Pod or probe writes one NVIDIA-format record into the real `/dev/kmsg` or the Fabric Manager log | Kernel log, FM log; runs the full chain from collection to recovery |
| Create real hardware or configuration anomalies | Change real state on the node: cap the power limit, unbind the EFA driver, tamper with the expected GPU count, shadow `nvidia-smi`, hold GPU devices, stop services, out-of-band reboot | DCGM metrics, host telemetry, Node Agent checks, node status |
| Deliver synthetic events to the control-plane API | Call collector-events, distributed XID and the synthetic node-replacement endpoint with a real token, without touching the host | Policy engine and orchestration layer, bypassing the collection layer |
| Break infrastructure and network | iptables blocking 443, deleting or rolling Pods, cordon eviction, Aurora failover, secret rotation, changing Deployment environment variables | Control-plane high availability, executor protocol, timeout and recovery paths |
| Seed state in the database | Register `synthetic` clusters and write remote commands, stuck workflows and expired evidence rows directly | dispatcher, command protocol, cleanup and reconciliation logic, with no real node |
| Notification drills and in-process simulation | Mail with `drill_id`; fake DCGM endpoint, fake EFA counter directory and in-memory kmsg lines inside an isolated process | Notification delivery path, collector parsing logic, zero side effects |

The conclusion of Section 7.8 stands: none of the above can make GPU hardware or the driver genuinely produce a specified
XID.

### 8.2 Common Preconditions

All node-level injections enter the host through the privileged Pod (privileged, hostPID, hostNetwork, hostPath root)
created by `scripts/e2e/regional/host_probe_fixture.py::HostProbeFixture`, chroot into the venv of the node's current
signed slot and run the probe subcommand. Every injection that changes node state first schedules a recovery timer with
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

### 8.3 Node Kernel and Driver Log Layer

| Injection method | Real signal produced | Mechanism | Typical cases | Code reference |
|---|---|---|---|---|
| Write an XID line to `/dev/kmsg` | One `NVRM: Xid (PCI:..): <n>` kernel log line, annotated with `marker=` or `drill_id=` | Direct write via `os.open("/dev/kmsg")`; each probe keeps an XID allowlist | DESTR-001/002/010/014/015/016/018/019, COLLECT-008/009/010/012/013/016, E2E-001 | `scripts/e2e/regional/probes/destructive_node_probe.py::write_xid`, `scripts/e2e/regional/probes/collector_node_probe.py::write_xid`, `scripts/e2e/regional/probes/e2e001_node_probe.py::write_xid11`, `scripts/e2e/regional/probes/node_host_probe.py::write_xid45` |
| XID written by a systemd timer after quiesce | XID 46 or 79 written on schedule once kubelet is stopped and the exec channel is unavailable | `systemd-run --on-active` schedules the probe's write-xid | DESTR-016 | `scripts/e2e/regional/probes/destr016_node_probe.py` |
| Write user-space annotated lines | A monitor-only XID 63, or a deliberately malformed Xid line missing its code | As above, kind limited to `xid63` and `unparsed-xid` | NET-001, COLLECT-018, NET-008 | `scripts/e2e/regional/probes/net001_node_probe.py::write_kmsg`, `scripts/e2e/regional/probes/collector_window_probe.py::write_kmsg` |
| Manifest privileged Pod writes kmsg or the FM log | Single- or two-node XID combinations, cross faults inside and outside the time window, SXID 10003 | busybox `printf > /host-dev-kmsg`; hostPath CharDevice or File | Manual COLLECT, GF-CROSS-FAULT, dual-node, the GF-LIVE-XID74 series | The `scripts/e2e/regional/manifests/fault-injection/` directory, `scripts/e2e/hyperpod/run_hyperpod_xid74_case.py` |
| Append an SXID line to the Fabric Manager log | A `nvidia-nvswitchN: SXid (PCI:..): <sxid>, Fatal/Non-fatal` record | Append write plus fsync; SXID allowlist | COLLECT-005/011/013/014/015 | `scripts/e2e/regional/probes/collector_node_probe.py::append_sxid` |

### 8.4 Node Hardware and Configuration Layer

| Injection method | Real signal produced | Mechanism | Typical cases | Code reference |
|---|---|---|---|---|
| Lower the GPU power cap and run at full load | Real power violation combined with high utilization | `nvidia-smi -pl` plus `dcgmproftester`, restored on a timer | COLLECT-002 | `scripts/e2e/regional/probes/collector_node_probe.py::throttle_gpu` |
| Unbind the EFA driver | `efa_inventory_mismatch` with failure_mode `DRIVER_UNBOUND` | Write `/sys/bus/pci/drivers/efa/unbind`, rebind on a timer | COLLECT-017, DESTR-021 | `scripts/e2e/regional/probes/collector_node_probe.py::unbind_efa` |
| Tamper with the collector's expected GPU count | `gpu_inventory_mismatch`, genuinely reaching a node reboot | Change `/etc/gpu-fault/collector.env` to the current value plus one, restored on a timer | COLLECT-004 | `scripts/e2e/regional/probes/collector_node_probe.py::override_expected_gpu_count` |
| Shadow `nvidia-smi` | Collection timeout, or one GPU missing from the first inventory | A systemd drop-in puts the shadow script first on PATH | COLLECT-019, COLLECT-020 | `scripts/e2e/regional/probes/collector_window_probe.py::open_window` |
| Collector environment window | 403 from a wrong cluster token, or a missing expected GPU count | EnvironmentFile or UnsetEnvironment in a systemd drop-in, deadman timer closes the window | COLLECT-018/019/020, NET-008 | As above |
| Seed an outbox dead-letter record | Replay hits a retired channel and returns 404 | Append ndjson under `/var/lib/gpu-fault/outbox/` | NET-008 | `scripts/e2e/regional/probes/collector_window_probe.py::seed_outbox_record` |
| Hold a GPU device file | `VERIFY_NO_GPU_CLIENTS` stays WAITING | A `systemd-run` transient unit opens `/dev/nvidiaN` and sleeps | DESTR-014/016/017/018 | `scripts/e2e/regional/probes/destr014_node_probe.py` and sibling probes |
| Out-of-band reboot | A node reboot that bypasses the HyperPod API | `systemd-run --on-active` runs `systemctl reboot` after a delay | DESTR-017 | `scripts/e2e/regional/probes/destr017_node_probe.py::arm_reboot` |
| Disable the Node Agent or kubelet | The Agent stops registering, the node goes NotReady, the hot spare becomes unavailable | Arm the recovery timer first, then `systemctl disable` or `stop` | DESTR-008, DESTR-014, DESTR-019 | `scripts/e2e/regional/probes/warm_spare_node_probe.py::stop_with_failsafe`, `scripts/e2e/regional/probes/destr014_node_probe.py::disable_agent_restart`, `scripts/e2e/regional/probes/destr019_node_probe.py::restart_agent` |
| Real quiesce and restore cycle | Stop fabricmanager, dcgm, persistenced and kubelet, then restore | Call the Node Agent's quiesce manager; or run `systemctl stop` in a chroot from a Manifest and restore on a timer | PREEMPT-012, the dual-node quiesce Manifest | `scripts/e2e/regional/probes/preempt012_node_probe.py`, `DEFAULT_QUIESCE_SERVICES` in `src/gpu_fault/node_agent/common.py` |
| Real low-utilization training load | The `LOW_GPU_UTILIZATION` sustained rule | Submit a CPU-only burn PyTorchJob | NOTIFY-005 | `scripts/e2e/regional/notify005_checks.py::low_utilization_manifest` |

### 8.5 Control-Plane API Layer

| Injection method | Real signal produced | Mechanism | Typical cases | Code reference |
|---|---|---|---|---|
| API replay of an XID | A synthetic XID event without touching the host | Deliver to `/v1/collector-events/nvidia-kernel` with the deployed `HttpEventSink` from inside an executor Pod | DESTR-009/012/020, E2E-002 | `scripts/e2e/regional/regional_live_fixture.py::RegionalLiveFixture.post_xid_event` |
| API replay of an SXID | A synthetic SXID event | As above, path `/v1/collector-events/fabric-manager` | COLLECT-014 | `scripts/e2e/regional/run_collector_destructive.py::post_fabric_event` |
| Distributed XID batch | A multi-node XID batch with `synthetic=true` | Direct POST to `/v1/gpu-events/xid/distributed` | The case of Section 7.7.2 | `src/gpu_fault/app/routes/gpu_events.py::evaluate_distributed_xids` |
| Synthetic node-replacement finding | A CRITICAL `REPLACE_NODE` that drives a hot-spare switch | POST `/v1/admin/test/node-replacement`, requires the switch and the execution token | DESTR-003, DESTR-008 | `src/gpu_fault/app/routes/collector_events.py::inject_test_node_replacement`, `scripts/e2e/regional/warm_spare_fixture.py::WarmSpareLiveFixture.post_synthetic_replacement`, `scripts/e2e/regional/synthetic_replacement_route.py` |
| Submit an event with an illegal field | Enqueued successfully but rejected by the processor, producing a rejected event | Deliver a payload with an unknown field through the real sink inside the node venv | COLLECT-018 | `scripts/e2e/regional/probes/collector_window_probe.py::post_rejected_event` |

### 8.6 Kubernetes and Network Layer

| Injection method | Real signal produced | Mechanism | Typical cases | Code reference |
|---|---|---|---|---|
| Adversarial node metadata mutation | Delete the hot-spare label, change the instance group, write stale or foreign annotations, roll back the reservation time, high-frequency patches producing 409 | kubectl patch, restored exactly after recording a baseline | DESTR-008/021/022 | `scripts/e2e/regional/run_destr008_warm_spare_shortage.py::ScenarioFixture`, `scripts/e2e/regional/probes/destr021_annotation_writer.py::patch_command`, `scripts/e2e/regional/run_destr022_spare_reservation_reclaim.py` |
| Hold a GPU on the hot spare | The hot spare is judged unavailable | Create a Pod requesting `nvidia.com/gpu` | DESTR-008 | `scripts/e2e/regional/warm_spare_fixture.py::GpuHolderFixture` |
| iptables blocking control-plane 443 | Collector, executor and control plane disconnected | `iptables` rejects 443 and rolls back on a timer | NET-001/008, ISO-006, COLLECT-018/019/020 | `scripts/e2e/regional/probes/net001_node_probe.py`, `scripts/e2e/regional/probes/cluster_network_probe.py::block` |
| Executor proxy blocking or packet loss | Lease renewal fails, retry after a lost result | A real ClusterActionExecutor with a loopback proxy running inside a Pod | NET-002/003/006 | `scripts/e2e/regional/probes/net002_executor.py`, `scripts/e2e/regional/probes/net003_executor.py`, `scripts/e2e/regional/probes/net006_executor.py` |
| Delete or roll control-plane and executor Pods | Process-level failover and takeover | `kubectl delete pod`, `rollout restart`, `scale` | HA-001/004/005/006/010 | `scripts/e2e/regional/run_ha001_control_plane_failover.py` and sibling runners |
| cordon plus PDB eviction | Topology and availability constraint verification | Cordon a CPU node, then call the Eviction API | HA-002 | `scripts/e2e/regional/run_ha002_pdb_topology.py` |
| Deployment environment variable window | Compress workflow lifetimes, lower validation counts, turn off the dispatcher, single-variable mutation | `kubectl set env`, with a compiled-in variable allowlist and a required confirmation string | DESTR-014/018, PREEMPT-037, BOOT-001..010 | `ALLOWED_VARIABLES` in `scripts/e2e/regional/executor_env_window.py`, `scripts/e2e/regional/control_plane_env_window.py`, `scripts/e2e/regional/boot_guard/mutate.py` |

### 8.7 Cloud Service and Data Layer

| Injection method | Real signal produced | Mechanism | Typical cases | Code reference |
|---|---|---|---|---|
| Aurora writer failover | Database primary switch | `aws rds failover-db-cluster` | HA-003, HA-010 | `scripts/e2e/regional/run_ha003_aurora_failover_reset.py`, `scripts/e2e/regional/run_ha010_aurora_blackout_liveness.py` |
| Aurora master password rotation | Credential invalidation and refresh | Run the credential-refresh Job after `aws secretsmanager rotate-secret` | HA-009 | `scripts/e2e/regional/run_ha009_aurora_credential_rotation.py` |
| Synthetic cluster seeded commands | Register a `synthetic` cluster and seed remote commands directly, claimed by a probe executor | Written through the Store API, no real node | CMD-017/018, NET-006, BOOT-015, CAP-001..005 | `scripts/e2e/regional/seeded_command_fixture.py::seed_command`, `scripts/e2e/regional/seeded_command_fixture.py::register_synthetic_cluster` |
| Isolated database seeds | Stuck workflows, expired evidence rows | A separate Postgres container or temporary SQLite, refusing the ambient store URL | PREEMPT-036/038 | `scripts/e2e/regional/run_preempt036_stuck_workflow_reconcile.py`, `scripts/e2e/regional/audit_raw_evidence_periodic_cleanup.py` |

### 8.8 Notification and In-Process Simulation Layer

| Injection method | Real signal produced | Mechanism | Typical cases | Code reference |
|---|---|---|---|---|
| drill notification | A real mail with the `[DRILL:..]` marker that triggers no action | Built with the email builder and sent through real SES | NOTIFY-001/002/007 | `scripts/e2e/regional/probes/notification_drill.py::build_notification`, `scripts/e2e/regional/probes/notify007_delivery_drill.py` |
| In-process synthetic signals | Fake DCGM metrics endpoint, fake EFA counter directory, in-memory kmsg lines | A local HTTP server and temporary directories fed to the deployed collector, then delivered to an isolated control plane | Section 7.1 and the hyperpod E2E | `scripts/e2e/hyperpod/run_hyperpod_dcgm_metrics_e2e.py`, `scripts/e2e/hyperpod/run_hyperpod_efa_traffic_e2e.py`, `scripts/e2e/hyperpod/run_hyperpod_three_source_fault_e2e.py` |

### 8.9 Maintenance Rules

The HyperPod software harness uses the complete fake DCGM field set, duration counter inputs are in ns, and policy time
advances in explicit 15-second samples; per-sample policy replay is reported separately from the sustained Collector
edge-filter verification of `DCGM-E2E-013`, and the former must not be passed off as a confirmation/suppression test. The
three-source harness first establishes an EOF baseline for the private FM log and then appends the SXID, binds responses by
channel and batch, and does not read the host GPU inventory.
The current DAG for EFA hung traffic is `FREEZE_EVIDENCE -> COLLECT_HUNG_TRIAGE ->
{COLLECT_DIAGNOSTIC_BUNDLE, VALIDATE_FABRIC}`, which is not the removed completion-phase quick-triage.
All of these verify only software inputs and an isolated control plane; they are not proof of hardware damage or
real-hardware recovery.

- When adding an injection method, update the tables in this chapter first, then the corresponding case's `injection` field; the two descriptions must agree.
- Implement injection only inside dedicated probe subcommands, with an allowlist, recovery timer and state snapshot; never write injection into deployment manifests or the production CLI.
- For injections that change node state, the report must record whether the recovery timer fired and whether the node returned to baseline.
