English edition of `docs/性能压测验收方案.md`; the Chinese file remains the source of record until both are maintained together.

# GPU Fault Recovery Control Plane Performance Load-Test Acceptance Plan

## 1. Goal

This document defines a repeatable capacity model, thresholds, command entry points and
evidence format. It stores no cluster names, node IDs, account numbers, private network
addresses, release digests or un-redacted run results from any customer environment.

Real load-test results must go into CI artifacts or an access-controlled private evidence
store. The public repository keeps only redacted conclusions and never treats a historical
PASS as capacity proof for a new release.

The current capacity targets must use the shared connection budget of
[Administrator Capacity Configuration](administrator-capacity-configuration.md), counting each
process's pool and dedicated LISTEN connections. The enabled preset defaults to 5 control-workers,
the disabled preset keeps 6; the historical 6-worker enabled load-test results in this document are
not rewritten as measurements of the new topology. Short-poll latency and the raw wait plus extra
overhead of the production 20-second long poll must be recorded separately.

## 2. Acceptance Scale

Three topologies must be covered:

| Topology | Baseline | Headroom |
|---|---:|---:|
| Multi-cluster regional control plane | 32 GPU clusters | 50 GPU clusters |
| Single very large GPU cluster | 1000 nodes | Increase to the target customer scale |
| Complex recovery actions | 32 clusters of concurrent workflows | 50 clusters of concurrent workflows |

The multi-cluster capacity model assumes 256 nodes per cluster and 8 GPUs per node:

```text
32 clusters: 8192 nodes / 65536 GPUs
50 clusters: 12800 nodes / 102400 GPUs
```

The single-cluster 1000-node scenario must use one and the same `cluster_id`; it must not be
substituted by creating several logical clusters, otherwise the single-cluster lane, quota and
hot rows are not covered.

## 3. Release Prerequisites

Every formal load-test round must record:

- wheel SHA-256, node bundle SHA-256, `module_digest` and schema version;
- replica counts and service role of ingress, control-worker and spool-worker;
- Aurora min/max ACU, instance status, connection limit and deadlock counts before and after the test;
- spool enabled/disabled, worker count, batch size and in-flight byte limit;
- processor completion cluster concurrency, retry backoff and lane policy distribution;
- the five remediation budget tiers: Region, cluster, node, failure domain and resource class;
  when the release lacks this capability record `unsupported` explicitly, never fabricate defaults;
- GPU cluster count, node count, request mix, duration and concurrency;
- load generator CPU/memory limits, actual usage and the dedicated node it runs on;
- start/end UTC times and cleanup result.

### 3.1 Load-Test Notifications Must Never Leave the Site

- Every synthetic performance cluster must use the reserved prefix `perf-cap-`; the Store must
  tag every notification produced by that prefix with `drill_id=perf-capacity`, including
  Collector, Agent, capacity and silence alert notifications that have no persisted Incident.
- AMP alerts do not pass through the Store: Alertmanager delivers straight to the site SNS topic.
  The Alertmanager definition must therefore route alerts with `cluster_id=~"perf-cap-.*"` to the
  `gpu-fault-drill-sink` receiver, which has no delivery configuration (the alerts remain
  queryable in AMP); the runner reads the live definition before registration to check that
  route, fails closed when it is missing, and writes the result to
  `notification-safety-preflight.json`.
- `GPU_FAULT_NOTIFICATION_DELIVER_DRILLS` must be unset or `false`. Before modifying the registry
  or creating load, the performance runner must check that value on every non-zero CPU Pod one by
  one; if any replica allows drill delivery it fails closed.
- The number of new mails in the administrator mailbox during the load test must be `0`. On any
  synthetic mail, stop generating load immediately, pause the dispatcher, clean up this round's
  `perf-cap-*` notification/result/delivery rows, and judge the round `INVALID`; do not continue on
  the grounds that "mail does not affect throughput".
- Do not rely only on the `drill_id` text in XID/SXID messages. Routine telemetry, Collector
  silence, Agent and action completion may also create notifications; the drill tag must be
  completed at the single entry point that persists notifications.

Control plane and data plane must use the same release. When the version or digest differs,
the round's results are invalid.

### 3.2 Registry and Control-Plane Isolation

Capacity load tests may by default target only an isolated control plane and an isolated
registry. The default namespace is `gpu-fault-perf-system`; the isolated control plane must
mount its own registry Secret, and the connection Secret in the data-plane namespace must
point at that isolated control plane.

```bash
export GPU_FAULT_PERF_CONTROL_NAMESPACE=gpu-fault-perf-system
export GPU_FAULT_PERF_DATAPLANE_NAMESPACE=gpu-fault-perf-system
export GPU_FAULT_PERF_REGISTRY_SECRET=gpu-fault-regional-clusters
export GPU_FAULT_PERF_CONNECTION_SECRET=gpu-fault-regional-connection
```

The isolated control plane must also use a separate, disposable PostgreSQL database; changing
only the namespace and registry Secret while continuing to write to the production database
does not constitute isolation. That database is still created by the separate schema ensure
Job, and business Pods must not enable auto schema init. After acceptance ends, first confirm
that all synthetic rows, Jobs and connections have terminated, then delete the database.

The GPU perf namespace deploys only the connection Secret, the load-test ServiceAccount/RBAC
and the simulated Jobs; it does not deploy a real Cluster Executor, Watcher, Collector or Node
Runtime. That namespace and the load ServiceAccount `gpu-fault-completion-watcher` are created
by the runner when missing (the namespace carries the `gpu-fault.io/perf-namespace` label; the
ServiceAccount is created as a per-round resource and torn down with the round), and the
connection Secret is mirrored from the identity namespace every round; both survive across
sites and may also be created manually in advance. The Action runner derives a short-lease
synthetic Agent identity and simulated Executor claims from the isolated CPU control plane's
release state and the current process's required pins; when the two disagree it fails closed.
Only live registry mode still requires the real data-plane Executor Deployment and the active
Agent identity to match the release pin.

Only when the formal `gpu-fault-system` control plane is genuinely approved for use may the
command supply both:

```text
--allow-live-registry
--confirm-live-registry ALLOW_PERF_CAPACITY_LIVE_REGISTRY
```

If either confirmation is missing it must fail before modifying the registry. Merely changing
the Secret name in the formal namespace cannot be claimed as isolation.

Every logical cluster registry record must contain:

- `synthetic=true`;
- the `synthetic_run_id` corresponding to this round's `suite_id`;
- a timezone-aware `synthetic_expires_at`;
- a placeholder identity that does not point at a real EKS/HyperPod.

Expired synthetic records must not keep authenticating and must not enter the runtime durable
registry. Legacy or expired `perf-cap-*` residue is cleaned automatically before the load test;
when another not-yet-expired run is found, stop rather than preempt it. After the load test this
round's records must be deleted, the registry re-read to confirm that no synthetic residue
remains, and `registry-preflight.json` and `registry-postflight.json` saved.
Before registration the runner records the digest and serialized shape of `clusters.json` in
the registry Secret (`registry-secret-baseline.json`, which holds only the digest and shape,
never tokens); at teardown it restores the Secret byte for byte to its pre-run content in that
shape (`registry-secret-restore.json` records `byte_identical`).
An alignment check runs once before registration and once after teardown, writing
`registry-alignment-preflight.json` / `registry-alignment-postflight.json`: the Secret's
configuration digest must equal the durable head's configuration digest, and every Running
`gpu-fault-api-ha` replica's `/healthz?verbose=1` must report
`regional_registry.secret_drift=false`. Misalignment before registration refuses the run (the
drift is not attributed to this round); misalignment after teardown fails this round's cleanup
and lists the differences and drifted replicas field by field.

## 4. Traffic Model

### 4.1 Three-Priority Mixed Traffic

| Priority | Requests |
|---|---|
| P0 | XID, SXID and other real fault channels |
| P50 | GPU/host suspicious evidence with an edge-filter reason |
| P100 | Routine state such as inventory, GPU metrics and host telemetry |

The fixed formal request matrix is as follows. Do not guess the P50 count from CLI defaults:

| Scale | P0: XID + SXID | P50: GPU + Host evidence | P100: three node telemetry kinds | Total requests | Ratio P0/P50/P100 |
|---|---:|---:|---:|---:|---:|
| 32 clusters × 256 nodes | 500 + 500 | 500 + 500 | 8192 × 3 = 24576 | 26576 | 3.76% / 3.76% / 92.48% |
| 50 clusters × 256 nodes | 781 + 781 | 781 + 781 | 12800 × 3 = 38400 | 41524 | 3.76% / 3.76% / 92.48% |

32/50 clusters must each be tested with:

1. telemetry spool disabled;
2. telemetry spool enabled;
3. enabled with a dedicated spool-worker;
4. at least one continuous input run during a control-plane Pod exit or rolling update.

### 4.2 Single Cluster, 1000 Nodes

Send a mix within the same cluster:

- 1000 node heartbeats;
- inventory, GPU metrics and host telemetry;
- XID, SXID and P50 evidence;
- workload observations;
- multi-node faults on the same attempt and independent faults on different attempts.

Spool disabled and enabled must be verified separately.

### 4.3 Complex Recovery Actions

Action load tests use a simulated executor and never call a real reset, reboot or provider
mutation. Every workflow contains at least:

- evidence;
- scheduler containment;
- workload stop/restart;
- a diagnostic branch;
- remote command claim/result;
- DAG dependency, lease and fencing.

A release that includes the durable remediation budget must split the Action load test into
two tiers:

1. **Production-budget tier**: use the production defaults about to be released and verify
   that fault storms are throttled, waiting workflows call no adapter, and everything
   eventually completes without starvation once quota is released;
2. **Raw-capacity tier**: in the isolated load-test environment, raise the budget through the
   formal release configuration until it is no longer a bottleneck, and measure the raw
   throughput of dispatcher, Store and Executor.

The current production defaults are:

| scope | Default |
|---|---:|
| Region | 20 |
| Per cluster | 5 |
| Per node | 1 |
| Per failure domain | 1 |
| Per resource class | 2 |

Every workflow in the current Action model requests `WORKLOAD_MUTATION`,
`GPU_RUNTIME_MUTATION` and `NODE_LIFECYCLE_MUTATION` at the same time. Therefore
`--workflows-per-cluster 4` in the production-budget tier actually has at most 2 same-cluster
workflows active simultaneously, and 32/50 clusters × 4 workflows are also executed in waves
under the Region cap of 20. This is the budget taking effect, not dispatcher throughput
regression.

The current seeded synthetic steps set no `failure_domain`, `fabric_partition` or
`availability_zone`, so they cannot prove that the failure-domain budget takes effect. Formal
budget acceptance must give synthetic workflows deterministic domains covering both directions:
same-domain mutual exclusion and different-domain parallelism.

### 4.4 Whole-Cluster Correlated Fault on N x 500+ Node Single-AZ Clusters (planned, not yet executed)

**No round of this section's scenario has ever been executed; what follows is only the scenario
definition and acceptance criteria and must not be cited as capacity proof.**

The current product requirement is N GPU clusters of 500+ nodes each, each in a single AZ.
Neither topology in §2 covers it: the multi-cluster model assumes 256 nodes per cluster, and
the single-cluster 1000-node model has only one `cluster_id`. What sets it apart from the
existing scenarios is that one whole-cluster correlated fault (such as a single-AZ network or
power event) produces ≥500 fault-priority requests simultaneously on ≥500 different node lanes
of the same cluster, whereas the 1000-node model of §13.4 sends load in matrix batches, not as a
whole-cluster wave in the same second.

The baseline topology matches the release defaults: 4 clusters x 512 nodes = 2048 managed
nodes, one AZ per cluster, AdminConfig `largestClusterNodeCount=512`, `managedNodeCount=2048`,
`maxClusterQueueDepth=4096`, `aurora.minAcu>=20.5` (derivation rules in Administrator Capacity
Configuration §6). Before execution the values above must be converged through the formal
configuration entry point, and `GPU_FAULT_PROCESSOR_FAULT_RESERVED_CLUSTER_DEPTH >= 512` must be
confirmed in the ConfigMaps of the three CPU roles.

Traffic model: against the background of the §4.1 three-priority mixed traffic running steadily,
inject 512 XID fault events into one of the 512-node clusters within the same second (one per
node, each on its own node lane), then let the resulting workflows run evidence, containment,
stop/restart and remote command per §4.3. The other 3 clusters keep normal telemetry traffic to
prove that a single-cluster storm does not spill over.

Acceptance criteria (each corresponds to one derivation rule or runtime self-check in
Administrator Capacity Configuration §6):

1. **The fault reserve holds the whole wave**: all 512 fault requests of the injected cluster
   get 202, and
   `gpu_fault_processor_admission_rejections_by_cluster_total{scope="cluster_reserved"}`
   is 0 for that cluster; at the same moment that cluster's routine telemetry may show 429
   (this is the reserve taking effect, not a failure).
2. **Single-cluster depth does not spill over**: for the other 3 clusters
   `gpu_fault_telemetry_spool_rejected_total{scope="cluster"}` and processor 429 are both 0,
   and `gpu_fault_processor_admission_rejections_total{scope="global"}` is 0.
3. **The Aurora floor holds**: HTTP 503 is 0 during the whole wave, and the actual Aurora ACU
   does not exceed 100% of the capacity pre-provisioned by Min 20.5 (if it does, the
   `managedNodeCount/100` ratio needs correction and must be recorded).
4. **Runtime self-check**: deliberately change `GPU_FAULT_PROCESSOR_FAULT_RESERVED_CLUSTER_DEPTH`
   to 128 and roll one role; the Pod must refuse to start with
   `fault-reserved cluster depth ... cannot hold one correlated whole-cluster fault wave`;
   `gpu_fault_capacity_largest_cluster_node_count` and
   `gpu_fault_capacity_managed_node_count` on `/metrics` must equal the AdminConfig declared values.
5. **Completion side**: all 512 fault incidents enter workflows, P0 completion p99 is compared
   with the §13.4 complete model and recorded, and the remediation budget executes in waves
   without starvation (§6.3).

The execution entry point, artifact format and redaction requirements follow §8.4 and §11; the
results are added as a new subsection of §13 after the first execution, and this section
remains the scenario definition.

## 5. Load Generator Gates

- Never generate load from inside the API Pod under test.
- Use at least 4 independent load-generator Pods.
- Load-generating Pods should be spread across different nodes.
- No-op calibration throughput must reach twice the target peak.
- Average CPU usage on the load generator stays below 50%; results are invalid if any resource exceeds 70%.
- Test at least the 16 and 32 concurrency levels for the same model.
- Timeouts, connection exhaustion or scheduling jitter on the load generator itself must not be attributed to the control plane.
- Before the start gate, the synchronous burst performs at most 5 bounded resolutions of the control-plane hostname and caches and rotates the results within each load-generating process. Every request still opens a new TCP/TLS connection and completes HTTP Host, TLS SNI and certificate verification with the original hostname; a request matrix on the order of 40,000 must not create 40,000 CoreDNS queries at once. Formal results must record the number of DNS resolution attempts, addresses and cache hits; if pre-resolution needs retries, any request bypasses the cache, or a transport retry occurs, the round fails.

## 6. Acceptance Thresholds

### 6.1 Ingestion and Admission

```text
P0 acceptance rate = 100% while the fault hard cap is not reached
P0 HTTP 429 = 0
P0 API p99 must be recorded and compared with the latency budget approved for the round
sustained steady-state processor oldest age < 5s
```

Once routine telemetry reaches the non-fault quota it must receive reserved rejections; P0 may
still enter the reserved capacity, and rejections are allowed only once the global/cluster hard
cap is reached.
The formal report must record the P0 ingress latency budget adopted for the round and its
basis; without a pre-approved uniform budget, p99 serves only as a capacity characteristic and a
version regression indicator and cannot decide PASS or FAIL on its own.

### 6.2 Queue and Completion Latency

```text
processor queue returns to the pre-test steady state within 60s after the burst ends
spool depth eventually returns to steady state when spool is enabled
fault completion p99 does not exceed the approved budget
no permanent starvation between priorities
```

"The queue drained quickly" alone cannot establish a pass: when many requests are rejected with
429/503 the queue may be empty just the same. Accepted, completed and failed counts and the
final state must be checked together.

### 6.3 Recovery Actions

```text
first isolation step p99 < 10s for a single-node fault that did not wait on the remediation budget
first isolation step < aggregation window + 10s for a multi-node fault that did not wait on the remediation budget
adapter/remote command starts = 0 while the budget is full
after the budget is released every waiting workflow eventually gets its turn, no permanent starvation
duplicate claim = 0
stale fencing write = 0
PostgreSQL deadlock = 0
```

Budget wait time must be counted separately; it must not be folded into the dispatcher's first
isolation step latency and then judged as an implementation regression. The production-budget
tier is about throttling correctness, fairness of waiting and eventual convergence; only the
raw-capacity tier compares first-action and workflow duration of different releases under the
same concurrency topology.

### 6.4 High Availability

- Deleting one ingress or worker Pod loses no P0.
- No destructive step is executed twice after an Aurora writer failover.
- Old and new replicas are protocol-compatible during a rolling upgrade, and a failure before finalize can roll back.
- Module digest drift on any replica invalidates the round's results.

## 7. Required Metrics

Save at least:

```text
gpu_fault_processor_queue_depth
gpu_fault_processor_queue_oldest_age_seconds
gpu_fault_processor_lane_wait_seconds
gpu_fault_processor_admission_rejections_by_path_total
gpu_fault_store_io_admission_wait_seconds
gpu_fault_postgres_pool_checkout_wait_seconds
gpu_fault_processor_claim_seconds
gpu_fault_telemetry_spool_depth
gpu_fault_telemetry_spool_in_flight_bytes
gpu_fault_event_loop_lag_seconds
gpu_fault_remediation_budget_active_claims
gpu_fault_remediation_budget_wait_total
gpu_fault_remediation_budget_waiting_workflows
```

Also collect:

- HTTP status, client latency and Server-Timing;
- load generator DNS mode, pre-resolution duration/attempt count, address count and in-process cache hits;
- processor priority completion p50/p95/p99/max;
- per-cluster p99 minimum, maximum and mean;
- Aurora ACU, CPU, DBLoad, connection count and commit/rollback;
- PostgreSQL table sizes, live/dead tuples and deadlocks;
- Pod CPU throttling, memory and restart counts.

## 8. Execution Entry Points

### 8.1 Multi-Cluster and Single-Cluster Capacity

```bash
export GPU_FAULT_CONTROL_KUBECONFIG='<cpu-control-plane-kubeconfig>'
export GPU_FAULT_DATAPLANE_CONTEXT='<gpu-data-plane-context>'
export GPU_FAULT_PERF_AWS_REGION='<aws-region>'
export GPU_FAULT_PERF_CONTROL_NAMESPACE=gpu-fault-perf-system
export GPU_FAULT_PERF_DATAPLANE_NAMESPACE=gpu-fault-perf-system

scripts/perf/regional_capacity_suite.py all \
  --case burst \
  --clusters 32 \
  --nodes-per-cluster 256 \
  --xid-total 500 \
  --sxid-total 500 \
  --gpu-evidence-total 500 \
  --host-evidence-total 500 \
  --workers 256 \
  --duration-seconds 60 \
  --artifact-root artifacts/perf
```

`all` and `run` must execute an idempotent teardown in `finally` regardless of success,
non-Complete, exception or interruption. Teardown deletes the Job/ConfigMap/Secret, purges
database records as needed, deregisters this round's synthetic registry, and retries at most
3 times on failure. The exception path must not merely write `aborted` and skip registry cleanup.

Staged execution is only for controlled debugging:

```bash
SUITE_ID='<approved-run-id>'

scripts/perf/regional_capacity_suite.py register \
  --clusters 32 \
  --suite-id "${SUITE_ID}" \
  --synthetic-ttl-seconds 3600 \
  --keep-registration

scripts/perf/regional_capacity_suite.py run \
  --case burst \
  --clusters 32 \
  --suite-id "${SUITE_ID}"
```

`run` requires the registration count, run ID and TTL to all match and cleans up by default
when it finishes. A standalone `register` must refuse to execute without an explicit
`--keep-registration`.

Registration and cleanup first update the bootstrap Secret, then atomically publish a new
generation through `/v1/regional/registry/revisions` and wait for all CPU processes to ACK the
same digest; the registry must no longer be brought into effect through a rollout restart.

50-cluster headroom:

```bash
scripts/perf/regional_capacity_suite.py all \
  --case burst \
  --clusters 50 \
  --nodes-per-cluster 256 \
  --xid-total 781 \
  --sxid-total 781 \
  --gpu-evidence-total 781 \
  --host-evidence-total 781 \
  --workers 256 \
  --duration-seconds 60 \
  --artifact-root artifacts/perf
```

The single-cluster 1000-node test uses the following fixed model, and all records must use one
and the same `cluster_id`:

```bash
scripts/perf/regional_capacity_suite.py run \
  --case burst \
  --clusters 1 \
  --nodes-per-cluster 1000 \
  --xid-total 500 \
  --sxid-total 500 \
  --gpu-evidence-total 500 \
  --host-evidence-total 500 \
  --training-heartbeat-total 1000 \
  --workload-observation-total 500 \
  --correlate-attempt-faults \
  --workers 256 \
  --duration-seconds 60 \
  --artifact-root artifacts/perf
```

Each of the 500 workload observations declares two nodes/critical ranks, and the 1000
training-progress heartbeats align with them rank by rank. Each attempt receives one XID and one
SXID on two separate nodes, forming 500 same-attempt multi-node fault groups; different groups
use independent `job_id/attempt_id/workload_id`. Together with per-node inventory, GPU metrics
and host telemetry the total request count is 6500. The heartbeats here are cleanable training
progress heartbeats, not signed Node Agent heartbeats that would be written into the production
Fleet.

### 8.2 Action Capacity

```bash
scripts/perf/regional_action_capacity_suite.py \
  --clusters 32 \
  --workflows-per-cluster 4 \
  --nodes-per-workflow 4 \
  --executor-workers 8 \
  --artifact-root artifacts/perf
```

A release with a remediation budget must run the production-budget tier and the raw-capacity
tier separately with the same request model. The raw-capacity tier must satisfy at least:

```text
Region limit >= clusters * workflows_per_cluster
cluster limit >= workflows_per_cluster
resource class limit >= workflows_per_cluster
node limit = 1 (only when every workflow uses non-overlapping nodes)
failure domain limit >= number of workflows planned to run concurrently within the same domain
```

Therefore in the fixed model:

- 32 clusters × 4 workflows: Region at least 128, cluster/resource class at least 4;
- 50 clusters × 4 workflows: Region at least 200, cluster/resource class at least 4.

These values are allowed only for the isolated raw-capacity tier and are not production default
recommendations. The configuration switch must go into the formal release/site configuration and
complete a rollout; it must not be applied to some Pods temporarily with `kubectl set env`.
`run.json` must save the five budget tiers, the tested tier and whether workflows were given a
failure domain. The runner reads the five budget tiers from one control-worker Pod, classifies the
run as `raw-capacity`, `production-budget` or `unbounded` per the table above and writes it to
`run.json.remediation_budget`; in both tiers the executor concurrency lower bound only requires
progress (1), and each executor's concurrency peak `max_concurrent_commands` is recorded as
evidence rather than a pass condition — the per-node and failure-domain limits deliberately
serialize same-cluster workflows. The executor wall clock and the Job `activeDeadlineSeconds`
follow `--timeout-seconds`; the production-budget tier 32×4 model takes about 16 minutes and
should be given `--timeout-seconds 2400` explicitly.

Every entry point must name the target kubeconfig/context explicitly. Production registry tokens
and Secrets must not be written into artifacts; the registry baseline may keep only cluster IDs
and token digests.
Action capacity teardown must also delete the synthetic workflows matching
`workflow-actionperf-*` that carry no `cluster_id`, and then delete the `perf-cap-*` incident,
command, Agent and queue records; otherwise orphan workflows with missing incidents remain and
keep triggering dispatcher errors.
After the executor Job is deleted, the processor queue is drained and registry ownership is
verified, teardown may also delete this round's own non-terminal workflow/command records (the
`forced_nonterminal` count is written into the cleanup result): the LEASED/PENDING commands left
by an aborted run have no executor that could complete them and would otherwise refuse every
later round's registration. During the run the runner refreshes this round's synthetic Agent
heartbeats and leases every 30 seconds (the seeding script's `run-heartbeats` mode); otherwise
the fleet compatibility preflight withholds destructive steps once the heartbeats expire.

### 8.3 Live correlated-action

Putting correlated ingestion and directly seeded Action workflows in the same report alone
cannot prove the causal chain from event to escalated action. Full acceptance must also run:

```bash
scripts/perf/regional_correlated_action_suite.py \
  --clusters 32 \
  --scenario-max-seconds 1800 \
  --terminal-drain-seconds 600 \
  --artifact-root artifacts/perf
```

This scenario establishes two independent attempts per synthetic cluster:

1. Aggregation and preemption chain: first submit two same-rank `XID 48 RESET_GPU` events in
   succession for the same attempt, node and GPU; both must reuse the same incident and
   workflow. After the earlier workflow completes `MARK_UNSCHEDULABLE` and `STOP_WORKLOADS`,
   submit `SXID 11001 RESET_ALL_GPUS_NVSWITCHES`. Containment is complete and no physical step is
   in flight, so this is a clean boundary: the stronger event preempts within the same workflow
   record, the weaker plan's not-yet-executed steps go into `superseded_step_indexes`, and the
   fabric reset executes immediately as that node's `branch:<node>:successor:1`, inheriting
   only those two still-valid containment steps; if the preemption lands on an in-flight physical
   step, it is still a cross-record successor with `preempt_predecessor`, and the audit counts the
   two shapes separately.
   The simulated executor returns `FAILED` for the full fabric reset; that step sits on the
   node branch of the DAG record, and the control plane must escalate that branch to a
   `RESTART_NODE` rung within the same workflow record (a `BRANCH_ESCALATED` event), then
   complete the following `VALIDATE_GPU/VALIDATE_HOST/VALIDATE_FABRIC/RESTORE_SCHEDULING` after the
   node returns, and the record ends `SUCCEEDED`.
2. Precise reset escalation chain: the second attempt creates a `RESET_GPU` workflow on its own,
   and the simulated executor returns `FAILED` for the `RESET_GPU` command; that record is a
   linear single-node plan, and the control plane must first grow it in place into a
   single-branch DAG, then likewise escalate within the record to a `RESTART_NODE` rung and end
   `SUCCEEDED` after the node returns.

Simulated nodes produce no telemetry of their own, and post-action validation after a reboot
accepts only fresh samples later than the reboot barrier; the runner therefore refreshes this
round's synthetic Agent heartbeats every 30 seconds while waiting, and at the refresh time
publishes that node's healthy GPU/NVLink metrics (GPU UUIDs consistent with the attempt
identity), host metrics and all collector success statuses to simulate "the node coming back".
The refresh writes only this round's `corr-node-*` nodes and does not change production node
data. The audit requires 2 incidents and 2 workflows per cluster (plus 1 for every cross-record
preemption and plus 1 for every reboot-after successor), every failed reset bound to exactly one
reboot (an in-record `RESTART_NODE` rung or a `workflow-reboot-after-<id>` successor) with all of
them `SUCCEEDED` and validated; ownerless reboots and `REPLACE_NODE` rungs number 0. Normally
both chains are in-record rungs and the successor count is 0; when the cluster remediation
budget cannot accommodate a new rung the control plane fails closed: the branch is exhausted
(reason mentioning remediation budget), the record is `FAILED`, and the reboot is then completed
by the whole-record escalation `workflow-reboot-after-<id>`; the audit accepts that shape, but
every exhausted branch must be a budget denial and its reboot must be validated.

### 8.4 Live Mixed Traffic and Causal Action Integrated Acceptance

The formal CPU control-plane capacity conclusion uses one and the same runner round and no
longer stitches together the three results of ingestion, Action seed and correlated-action:

This entry point modifies the live CPU registry, creates simulated Jobs in the GPU perf
namespace, and may adjust Aurora to `Min=124, Max=128`. Before execution the Region, CPU
kubeconfig, GPU context, Aurora cluster ID, maintenance window, stop conditions and restore
values must be stated and approved explicitly. The current workspace must already have been
converted into a clean, signed release that completed verify/stability; a formal capacity
conclusion cannot be produced directly from a dirty checkout.

```bash
export STATE_DIR=/secure/gpu-fault
export GPU_FAULT_CONTROL_KUBECONFIG='<cpu-control-plane-kubeconfig>'
export GPU_FAULT_DATAPLANE_CONTEXT='<approved-gpu-perf-context>'
export GPU_FAULT_PERF_AWS_REGION='<aws-region>'
export GPU_FAULT_PERF_CONTROL_NAMESPACE=gpu-fault-system
export GPU_FAULT_PERF_DATAPLANE_NAMESPACE=gpu-fault-perf-system
export GPU_FAULT_PERF_IDENTITY_NAMESPACE=gpu-fault-system
export AURORA_CLUSTER_ID='<aurora-cluster-id>'
```

`gpu-fault-perf-system` may hold only the connection Secret, the load-generator
ServiceAccount/RBAC and the simulated Executor Job; no real Cluster Executor, Watcher, Collector
or Node Runtime may be deployed there. Before the formal start run:

```bash
gpu-fault-admin status --full --state-dir "${STATE_DIR}"
```

The four-round matrix is as follows; do not continue when the previous round did not pass or
its cleanup is incomplete:

| Round | AdminConfig preset | clusters | spool | P0 | P50 | P100 | Total requests | Workflow | command |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `32-disabled` | 32 | disabled | 1000 | 1000 | 24576 | 26576 | 128 | 960 |
| 2 | `32-enabled` | 32 | enabled | 1000 | 1000 | 24576 | 26576 | 128 | 960 |
| 3 | `50-disabled` | 50 | disabled | 1562 | 1562 | 38400 | 41524 | 200 | 1500 |
| 4 | `50-enabled` | 50 | enabled | 1562 | 1562 | 38400 | 41524 | 200 | 1500 |

Each round first converges the control plane through the formal configuration entry point,
for example:

```yaml
# ${STATE_DIR}/admin-config.yaml
apiVersion: gpu-fault.aws/v1alpha1
kind: AdminConfig
spec:
  capacity:
    preset: 32-disabled
```

```bash
gpu-fault-admin config \
  --state-dir "${STATE_DIR}" \
  --reference PERF-32-DISABLED
```

After the configuration command completes verify/stability, run the runner for the same round:

```bash
scripts/perf/regional_integrated_workflow_capacity_suite.py \
  --clusters 32 \
  --spool-mode disabled \
  --aurora-cluster-id "${AURORA_CLUSTER_ID}" \
  --configure-aurora-capacity \
  --confirm-aurora-scaling SET_AURORA_MIN_124_MAX_128 \
  --workflows-per-cluster 4 \
  --ingress-workers 256 \
  --executor-workers 8 \
  --lead-seconds 20 \
  --timeout-seconds 1800 \
  --executor-idle-seconds 300 \
  --synthetic-ttl-seconds 3600 \
  --allow-live-registry \
  --confirm-live-registry ALLOW_PERF_CAPACITY_LIVE_REGISTRY \
  --label integrated-32-disabled \
  --artifact-root artifacts/perf
```

The other three rounds must replace the preset, `--clusters`, `--spool-mode`, `--label` and the
approval reference together. The preset switch between rounds is a CONTROL_PLANE_ONLY release:
when switching from `*-enabled` to `*-disabled`, the role rollout script first rolls ingress to
spool=false, then reads `gpu_fault_telemetry_spool_depth/leased` and similar from the
spool-worker Pod's `http` container port until they reach 0 before scaling spool-worker to 0; a
Pod that has not answered yet simply keeps being polled. The automatic rollback on a failed
release re-resolves the ingress Pods instead of reusing Pod names remembered during the upgrade
phase.
The formal value of `--workflows-per-cluster` is fixed at 4 and must not be guessed from the CLI
default. Each synthetic cluster picks 3 XIDs and 1 SXID from the original matrix, covering two
resets, one reboot and one diagnostic/fabric reset respectively. The step counts compiled by the
current live policy are `10/10/9/12`; the 9-step reboot branch includes post-reboot
`VALIDATE_HOST`, and the 12-step SXID branch includes diagnostic, stop, fabric validation and
workload restart at once.
The attempt/workload and synthetic Agent context is pre-seeded before the start gate and is not
counted in the matrix.
The runner uses the same cold-connect model as the formal four-round baseline; before the start
gate and every 30 seconds during execution it refreshes the synthetic Agent heartbeat, healthy
GPU/NVLink/RDMA metrics, `load1_per_cpu`, `memory_used_percent`, `filesystem_used_percent` and
all collector success statuses that Agent actually requires. All samples use the current
refresh time and must be later than the completed reset/reboot barrier, ensuring post-action
validation verifies genuinely fresh data. The refresh only updates existing synthetic nodes and
does not change identity, workflow or production node data.

The simulated executor must claim real remote commands, maintain lease/fencing, wait an
approximation of the action duration and return `SUCCEEDED`. Defaults are `RESET_GPU=8s` and
`RESTART_NODE=120s`, all results carry `simulated=true`; calling a real Node Agent, Kubernetes
mutation or provider API is forbidden. The report uniformly records ingress/processor,
workflow/command p50/p95/p99, duplicate, fencing, budget wait, Aurora, queue drain and
cleanup. A load generator retry that succeeds after a first connection error must also be
counted in each path's `transport_retries`, not only the final HTTP success; the simulated
executor must aggregate `claim_error_counts` by HTTP status or exception type and save bounded
time samples that contain no URL, token, cluster ID or exception body.

Before registering the synthetic registry the runner records the original Min/Max and raises
Aurora to `Min=124, Max=128` only when below the floor (a window above the floor is left as is),
then hard-verifies that both `db.serverless` instances are `available` and that the actual
CloudWatch capacity of each is not below 124 ACU; at the same time it reads
`GPU_FAULT_REMEDIATION_MAX_ACTIVE_*` from every control-worker Pod one by one and requires them
to match the 32/50-cluster budgets of the table below exactly. The results are written to
`aurora-preflight.json` and `remediation-budget-preflight.json` respectively.
Before registration the runner also mirrors the connection Secret from the data-plane identity
namespace (`GPU_FAULT_PERF_IDENTITY_NAMESPACE`) into the perf namespace (the perf namespace is
recreated across sites and would otherwise retain the previous site's CA and token); it fails
closed when the identity namespace lacks that Secret, and the mirror result is written to
`connection-secret-preflight.json`.

The spool mode must come from a formal config-only release. The runner checks the actual
environment of every ingress Pod, the spool-worker replica count and the five remediation budget
tiers of every control-worker one by one; online `kubectl set env` is not allowed. The report
records the code release ID, the global configuration SHA and the three role configuration
SHAs. The runner must also check that the live ingress start command contains no
`--limit-max-requests` or the corresponding jitter parameter; when an old release still recycles
by request count, it fails closed before registering synthetic clusters, and "restart the
workers first every round" to zero the accumulated request count cannot be used to claim
steady-state capacity passed. The check result is written to
`ingress-process-model-preflight.json`.

The integrated runner's final verdict must verify:

```text
fixed P0/P50/P100 request counts match exactly
mixed ingress Indexed Job = Complete
P0 (NVIDIA_KERNEL, FABRIC_MANAGER_LOG) and evidence paths: HTTP 429/503 and client errors are all 0
routine telemetry paths (GPU_INVENTORY, GPU_METRICS, HOST_TELEMETRY) may show the reserved 503 of §6.1 at a share ≤ 2% (recorded in ingress.reserved_rejections / reserved_rejection_share); 429 and other non-2xx are 0
first transport retry = 0 on every path
incident/workflow count = clusters × 4
every action workflow = SUCCEEDED
remote command count = clusters × 30
every command terminal with result_details.simulated = true
workflow step count within 8..12
reset, fabric reset, reboot, containment and restore operations all covered
duplicate claim/idempotency/workflow step = 0
fencing mismatch, claim/result error, permanent budget waiter = 0
control-plane Pod set unchanged, container restart delta = 0 and all Ready at the end
at least one lease renewal on a long action
processor queue and spool back to the depth from before the round started
```

To additionally verify automatic escalation to reboot after a `RESET_GPU` or fabric reset
failure, same-rank aggregation and preemption, the `regional_correlated_action_suite.py` of §8.3
must be run separately. That suite expects 2 initial incidents and 2 workflows per cluster
(plus 1 for every cross-record preemption or reboot-after successor following a budget
denial), and 2 failed reset/fabric chains each bound to one successful, validated reboot (an
in-record `RESTART_NODE` rung; a `workflow-reboot-after-<id>` successor only on budget denial);
that failure-chain criterion must not be mixed into this section's integrated runner count of
successful Workflows.

The scenario allows only `perf-cap-*` synthetic registrations and the simulated executor, and
calls no real reset, reboot, cordon, taint, workload mutation or provider API. The runner deletes
the Jobs, ConfigMaps, synthetic registry and this round's database records in `finally`; any
round may continue only after the following evidence is confirmed:

```text
status.json = ok
summary.json.status = PASS
Job/Pod in cleanup-workloads.json are all empty
incident/workflow/command in cleanup-residuals.json are all 0
registry-postflight.json contains no synthetic registration from this round
every workflow and command is terminal
```

Once the runner enters `finally` it must temporarily mask `SIGINT`, `SIGTERM` and `SIGHUP` and
explicitly wait until both Jobs and all their Pods are deleted. The first termination signal only
changes the round's result to aborted and cannot interrupt the bounded teardown; any remaining
resource or database/registry residue still fails the round.

After the measurement window ends, the runner reads load/executor Pod logs with at most 16-way
concurrency and sorts them by Indexed Job completion index before writing artifacts and
aggregating. That concurrency only shortens the summary time and does not change the load
window, request model, latency samples or verdict; any read exception still fails the round
closed.

On any verdict failure, critical alert, undrained queue, cleanup residual, unconverged registry
or unavailable Aurora, stop the remaining rounds immediately. Failed artifacts go into
`aborted/` and cannot serve as formal PASS evidence.

The runner records the original Min/Max in `aurora-preflight.json.initial_scaling` but does not
restore Aurora capacity automatically. After all rounds end, the original Min/Max must be
restored per that record through the approved AWS/IaC process, and the cluster and both
`db.serverless` instances verified available again; at the same time run
`gpu-fault-admin config` once with the AdminConfig saved before the load test to restore the
site's original configuration and run `verify` again.

## 9. spool A/B Method

spool enabled/disabled must use:

- the same wheel, schema and runtime profile;
- the same cluster count, node count, total requests and random seed;
- the same Aurora ACU and control-plane resources;
- the same load-generator configuration;
- separate run directories and UTC times.

The configuration switch must go through the formal rollout process and must not merely modify
the environment variables of some Pods. Before each round, confirm that the service role, spool
setting and module digest of all non-zero replicas are exactly the same.

## 10. Verdicts

Formal results are one of:

- `PASS`: every threshold met;
- `FAIL`: any threshold failed;
- `INVALID`: version drift, load generator saturation, missing metrics, failed cleanup or an inconsistent test model;
- `PASS_WITH_LIMITATIONS`: allowed only for explicit limitations that do not affect the core thresholds.

The report must also record the direction of failure and cannot offer only averages or
throughput. Capacity may never be declared passed in the following situations:

- higher throughput obtained through large numbers of 429/503;
- latency counted only over successful requests;
- a first transport error that succeeded after an internal retry not counted as a client error;
- latency written as 0 when everything failed;
- several logical clusters used in place of a single-cluster hot-spot test;
- only ingress observed with spool enabled, not the spool-worker;
- results of a historical release used in place of the current release.
- a release that supports the remediation budget ran only the production-budget tier or only the raw-capacity tier;
- the five budget tiers not recorded, or budget active/wait metrics missing;
- a claim to have verified the failure-domain budget while the synthetic workflows have no domain field.

## 11. Evidence and Retention

Each round's output directory contains at least:

```text
run.json
status.json
summary.json
metrics-before.json
metrics-after.json
cgroup-before.json
cgroup-after.json
control-pods-before.json
control-pods-after.json
ingress-process-model-preflight.json
postgres-before.json
postgres-after.json
queue-drain.json
processor-priority-latency.json
aurora.json
pods/
executors/
```

Every ingress path in `summary.json` must include classified `transport_retries` counts; the
executor must include `claim_error_counts` and bounded, redacted `claim_error_samples`. The
integrated runner must also record `control_plane_pod_lifecycle`, containing at least the total
restart delta, the Pods that restarted, Pods that appeared/disappeared during the run, counter
regressions and Pods not Ready at the end. If any item is non-empty or non-zero it cannot count
as a formal PASS.

An Action load test with a remediation budget must also save `remediation-budget.json`,
containing at least the five-tier configuration, peak active claims per scope, wait counts,
longest wait, the number of workflows that never got to execute and the number of lease-expiry
releases. When an old release does not export these fields, write `unsupported` explicitly; an
all-zero file must not be substituted.

Raw logs and site identifiers are kept only in CI artifacts or private object storage. The
public repository may publish reviewed summary conclusions but must remove account numbers,
ARNs, cluster names, node IDs, IPs, tokens, Secrets and internal DNS.

## 12. Isolation and Cleanup

- Store micro load tests use a separate temporary schema.
- API composite load tests use a separate audit cluster and deterministic event IDs.
- No real reset, reboot or replace is executed in capacity tests.
- After the test, unconditionally delete Jobs, ConfigMaps, Secrets, temporary registry and database records.
- When cleanup fails, `status.json` must be marked `aborted`; an incomplete directory must not be used as formal evidence.

## 13. Redacted Reference Baselines

This section keeps only reference snapshots free of infrastructure identity, for judging
capacity trends and regression direction. It is not PASS proof for the current or any future
release; a formal release must still re-run the commands above and save the raw artifacts.
The earlier snapshots in §13.2–§13.4 were produced before the remediation budget entered the
release and can only continue to serve as ingestion, Store and processor trend baselines. The
newer snapshot in §13.5 records the budget configuration but is still a standard ingestion burst
and cannot prove that the production-budget tier or the raw Action capacity tier passed.

### 13.1 Common Configuration and Request Samples

Unless a table says otherwise, the reference experiments use:

- 3 ingress Pods, 4 Uvicorn processes per Pod;
- 6 control-worker Pods, 4 Uvicorn processes per Pod;
- 3 dedicated spool-worker Pods when enabled, 1 process per Pod;
- when disabled, ingress PostgreSQL pool max 40 per process and spool-worker 0;
- when enabled, ingress pool max 48 per process and spool-worker 3;
- control-worker pool max 24 per process, spool-worker pool max 12 per process;
- 256 load workers, 60 seconds, cold connections;
- spool replay 8 workers per process, batch max 64 items;
- processor cluster depth 1024, fault reserved cluster depth 128;
- spool cluster depth 1024 except in the single-cluster depth4096 experiment;
- PostgreSQL deadlock 0 in all listed experiments.

The two single-cluster samples are:

| Model | P0 | P50 | P100 | Total requests | Notes |
|---|---:|---:|---:|---:|---|
| priority-only | 1000 | 1000 | 3000 | 5000 | Only XID/SXID, the two evidence kinds and the three node telemetry kinds |
| complete-attempt | 1000 | 1500 | 4000 | 6500 | Additionally 500 workload observations and 1000 training heartbeats |

In `complete-attempt`, each of the 500 observations declares two nodes; the first node of the
same attempt receives the XID and the second node receives the SXID. P50 contains 1000 evidence
items and 500 observations; P100 contains 3000 node telemetry items and 1000 training heartbeats.

### 13.2 32/50 Clusters, Max ACU 64

The test declared `Min=0.5, Max=64` and warmed the actual capacity to at least 32 ACU before
each round. `P0/P50/P100 p99` in the table below is processor completion latency; P100 goes
through the spool when enabled and is therefore recorded as `N/A`.

| Clusters | Spool | HTTP 503 | Fault ingress p99 | P0/P50/P100 completion p99 | Throughput req/s | Aurora max ACU / CPU |
|---:|---|---:|---:|---:|---:|---:|
| 32 | disabled | 81 | 12.04s | 14.86s / 40.55s / 55.91s | 891 | 48.0 / 74.6% |
| 32 | enabled | 0 | 10.07s | 11.60s / 11.40s / N/A | 1289 | 61.5 / 95.9% |
| 50 | disabled | 2352 | 10.89s | 21.00s / 69.20s / 78.96s | 1345 | 63.5 / 98.6% |
| 50 | enabled | 1291 | 16.41s | 18.67s / 17.42s / N/A | 1304 | 64.0 / 100% |

All four groups accepted all of P0. 32-enabled eliminated routine telemetry rejections;
50-enabled reduced 503s by 45.1% and markedly improved P50 completion latency, but once the
database saturated the Fault ingress p99 was 50.7% higher than disabled.

### 13.3 50-Cluster ACU Ladder

`low-start` means only Max was raised and the run started from about 32 ACU actual;
`pre-provisioned` means Min was raised first and the run waited for actual capacity to arrive.
All request matrices are 41524 requests.

| Min/Max ACU | Spool | Actual max ACU / CPU | HTTP 503 | Fault p99 | P0/P50/P100 completion p99 | Throughput req/s |
|---|---|---:|---:|---:|---:|---:|
| 0.5/96 low-start | disabled | 45.5 / 46.9% | 3076 | 15.61s | 31.02s / 103.11s / 113.22s | 1017 |
| 0.5/96 low-start | enabled | 40.5 / 41.8% | 1942 | 17.52s | 21.09s / 19.78s / N/A | 1105 |
| 92/96 pre-provisioned | enabled | 96.0 / 100% | 0 | 10.74s | 10.46s / 10.13s / N/A | 1758 |
| 124/128 pre-provisioned | disabled | 124.0 / 88.6% | 0 | 8.62s | 12.03s / 65.16s / 67.27s | 1745 |
| 124/128 pre-provisioned | enabled | 128.0 / 100% | 0 | 10.81s | 10.11s / 9.58s / N/A | 1932 |
| 124/128 + tuned delays | enabled | 128.0 / 100% | 0 | 10.51s | 8.67s / 8.17s / N/A | 1925 |

Raising Max alone did not let the short burst use the added capacity; the actual peak was only
40.5–45.5 ACU. After pre-provisioning Min, 503s dropped to zero and throughput rose markedly.
128-enabled still reached 100% Aurora CPU, showing that the write path can keep consuming
database capacity at this scale, but Fault p99 did not fall linearly with ACU.

The tuned group used:

```text
GPU_FAULT_PROCESSOR_ADMISSION_BATCH_DELAY_SECONDS=0.05
GPU_FAULT_PROCESSOR_FAULT_ADMISSION_BATCH_DELAY_SECONDS=0.002
GPU_FAULT_PROCESSOR_EVIDENCE_ADMISSION_BATCH_DELAY_SECONDS=0.01
GPU_FAULT_TELEMETRY_SPOOL_BATCH_DELAY_SECONDS=0.05
```

Compared with the default delays at the same ACU, P0/P50 completion p99 improved by
14.3%/14.7%, Fault p99 improved by 2.8%, throughput was essentially unchanged, and drain
improved by 7.1%. But PostgreSQL `xact_rollback` rose from 4 to 59; there was no deadlock or
SQL exception, yet it still needs several repeated rounds before it can become a production
default.

### 13.4 Single Cluster, 1000 Nodes

| Model | Spool cluster depth | HTTP 429/503 | Fault p99 | P0/P50/P100 completion p99 | Throughput req/s | Aurora max ACU / CPU |
|---|---:|---:|---:|---:|---:|---:|
| priority-only | 1024 | 243 / 0 | 4.41s | 4.53s / 2.58s / N/A | 428 | 124 / 60.4% |
| complete-attempt | 4096 | 0 / 0 | 12.83s | 13.17s / 5.66s / 8.80s | 234 | 124 / 39.2% |

All 243 of the first group's 429s came from
`gpu_fault_telemetry_spool_rejected_total{scope="cluster"}`: 3000 P100 items shared one
cluster and the peak tripped spool cluster depth 1024. After temporarily raising it to 4096, all
6500 requests of the complete model returned 202, and the processor final responses were all 200.

Persistence checks of the complete model:

- 500 observations, 500 independent attempts, 1000 container nodes;
- 1000 training heartbeats, 500 attempts, 1000 nodes;
- 1000 fault incidents, exactly 2 per attempt;
- all 500 XID and 500 SXID evidence items carry attempt identity;
- processor, spool, object, observation and heartbeat test data all 0 after cleanup.

This model adds 1500 non-spool processor requests and attempt correlation, so its latency cannot
be read directly against priority-only as "depth4096 got slower". The only direct conclusion
about the depth change is: single-cluster spool 429s fell from 243 to 0.

### 13.5 Latest 32/50-Cluster spool A/B

This group used the currently signed release, Aurora `Min=124, Max=128`, and the Pod, process,
connection pool and dedicated spool-worker configuration of §13.1. The load generator used 256
workers, 60 seconds, cold connections; the request matrix used the fixed 32/50-cluster model of
§4.1.

All four rounds loaded and held the following remediation budget:

| Clusters | Region | Per cluster | Per resource class | Per node | Per failure domain |
|---:|---:|---:|---:|---:|---:|
| 32 | 128 | 4 | 4 | 1 | 1 |
| 50 | 200 | 4 | 4 | 1 | 1 |

Both spool modes of the 32-cluster rounds used:

```bash
GPU_FAULT_REMEDIATION_MAX_ACTIVE_REGION=128
GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_CLUSTER=4
GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_RESOURCE_CLASS=4
GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_NODE=1
GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_FAILURE_DOMAIN=1
```

Both spool modes of the 50-cluster rounds used:

```bash
GPU_FAULT_REMEDIATION_MAX_ACTIVE_REGION=200
GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_CLUSTER=4
GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_RESOURCE_CLASS=4
GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_NODE=1
GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_FAILURE_DOMAIN=1
```

With spool disabled, the actual ingress environment was:

```bash
GPU_FAULT_SERVICE_ROLE=ingress
GPU_FAULT_TELEMETRY_SPOOL=false
GPU_FAULT_POSTGRES_POOL_MIN_SIZE=2
GPU_FAULT_POSTGRES_POOL_MAX_SIZE=40
GPU_FAULT_STORE_IO_WORKERS=28
```

With spool enabled, the actual ingress environment was:

```bash
GPU_FAULT_SERVICE_ROLE=ingress
GPU_FAULT_TELEMETRY_SPOOL=true
GPU_FAULT_POSTGRES_POOL_MIN_SIZE=16
GPU_FAULT_POSTGRES_POOL_MAX_SIZE=48
GPU_FAULT_STORE_IO_WORKERS=24
```

In both modes the control-worker was 6 Pods with 4 processes per Pod, actual environment:

```bash
GPU_FAULT_SERVICE_ROLE=worker
GPU_FAULT_POSTGRES_POOL_MIN_SIZE=2
GPU_FAULT_POSTGRES_POOL_MAX_SIZE=24
GPU_FAULT_STORE_IO_WORKERS=8
```

The enabled mode additionally used 3 dedicated spool-workers with 1 process per Pod:

```bash
GPU_FAULT_SERVICE_ROLE=spool-worker
GPU_FAULT_TELEMETRY_SPOOL=true
GPU_FAULT_POSTGRES_POOL_MIN_SIZE=2
GPU_FAULT_POSTGRES_POOL_MAX_SIZE=12
GPU_FAULT_STORE_IO_WORKERS=8
```

Both modes loaded the same spool boundary parameters; disabled simply does not enable writes and
does not run the spool-worker:

```bash
GPU_FAULT_TELEMETRY_SPOOL_ADMISSION_PARTITIONS=8
GPU_FAULT_TELEMETRY_SPOOL_BATCH_DELAY_SECONDS=0.01
GPU_FAULT_TELEMETRY_SPOOL_BATCH_GROUPS=8
GPU_FAULT_TELEMETRY_SPOOL_BATCH_SIZE=64
GPU_FAULT_TELEMETRY_SPOOL_MAX_CLUSTER_DEPTH=1024
GPU_FAULT_TELEMETRY_SPOOL_MAX_DEPTH=65536
GPU_FAULT_TELEMETRY_SPOOL_MAX_ITEM_BYTES=4194304
GPU_FAULT_TELEMETRY_SPOOL_STORE_IO_MAX_IN_FLIGHT=256
GPU_FAULT_TELEMETRY_SPOOL_STORE_IO_WORKERS=8
```

These values only prove that the configuration was consistent during the test. This traffic
created no recovery actions that needed executing, so it cannot be used to judge budget wait,
fairness, remote command capacity or workflow terminal states.

| Clusters | Spool | Requests / HTTP errors | Throughput req/s | Fault p99 | Inventory / GPU metrics / Host p99 | Drain | Aurora max ACU / CPU | `xact_rollback` |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 32 | disabled | 26576 / 0 | 1663.6 | 7.00s | 9.93s / 9.91s / 9.88s | 3.85s | 124.0 / 94.6% | 29 |
| 32 | enabled | 26576 / 0 | 1685.8 | 7.85s | 9.22s / 9.22s / 9.23s | 3.55s | 128.0 / 100.0% | 108 |
| 50 | disabled | 41524 / 0 | 1792.3 | 9.35s | 16.10s / 16.39s / 16.37s | 3.56s | 124.0 / 91.6% | 15 |
| 50 | enabled | 41524 / 0 | 1947.5 | 10.72s | 15.08s / 14.94s / 14.21s | 3.66s | 128.0 / 100.0% | 50 |

All requests in the four rounds returned HTTP 202, HTTP 429/503 were 0, and PostgreSQL deadlocks
were 0; when sampling stopped, everything was 0 except a spool depth of 1 in 50-enabled, and
cleanup then completed.

The changes of enabled relative to disabled at the same scale are as follows. A positive
Throughput value means an increase; negative latency and drain values mean improvement.

| Clusters | Throughput | Fault p99 | Inventory p99 | GPU metrics p99 | Host p99 | Drain | Aurora CPU | `xact_rollback` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | +1.3% | +12.0% | -7.2% | -7.0% | -6.6% | -7.7% | +5.7% | +272.4% |
| 50 | +8.7% | +14.7% | -6.3% | -8.8% | -13.2% | +2.8% | +9.2% | +233.3% |

Reading:

- enabled lowered the p99 of all three routine telemetry kinds at both 32 and 50; the 50-cluster
  throughput gain of 8.7% is clearly higher than the 1.3% at 32 clusters;
- enabled Fault p99 was 12.0% and 14.7% higher than disabled; the spool mode must not be chosen
  on total throughput alone;
- both enabled samples pushed Aurora CPU to about 100%; the current configuration has no
  obvious database headroom;
- compared with the previous 50-cluster sample at the same `Min=124, Max=128`, disabled
  throughput rose 2.7% but all four p99s regressed; enabled throughput rose 0.8%, Fault p99
  improved 0.9%, and the three routine telemetry p99s improved 4.4%–9.6%;
- `xact_rollback` rose markedly, especially in the enabled samples. Although there was no
  deadlock or HTTP error, A/A and A/B should still be repeated and the rollback causes recorded
  before it becomes the production default.

This group is judged only as a valid ingestion/spool A/B reference result. Event aggregation,
conflict handling, action escalation, every workflow/command terminal state, duplicate, fencing
and budget wait must still be verified by the separate four-group composite/action-capacity
experiments.

### 13.6 Latest Four Formal Integrated Workflow Rounds

This group used the same signed release and the formal runner of §4.2, and on top of the fixed
ingress matrix of §13.5 created 4 workflows per cluster with full attempt/workload causal
identity at the same time. 32 clusters executed 128 workflows and 960 simulated remote commands
in total; 50 clusters executed 200 workflows and 1500 simulated remote commands. Resets and
reboots were only returned as `simulated=true` by the isolated executor after representative
durations; no real Node Agent, Kubernetes mutation or provider reboot was called.

All four rounds used a window not below Aurora `Min=124, Max=128` (the `32-*` rounds wrote
`aurora: 124/128` explicitly in AdminConfig, the `50-*` rounds inherited the preset floor
`128/128`, and the runner does not lower a window above the floor back to 124), and confirmed
before generating load that the actual capacity of both writer and reader was not below 124 ACU.
The load model stayed cold-connect; the control-plane hostname was resolved once before the start
gate with a process-level address cache, while HTTP Host, TLS SNI and certificate verification
still used the original hostname.

| Clusters | Spool | Requests / HTTP errors | Throughput req/s | Fault p99 | Inventory / GPU metrics / Host p99 | Workflow / Command p99 | Drain | Aurora max ACU / CPU | `xact_rollback` |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | disabled | 26576 / 0 | 1437.8 | 6.70s | 10.00s / 9.73s / 9.78s | 206.45s / 122.29s | 4.32s | 124.0 / 87.6% | 92612 |
| 32 | enabled | 26576 / 0 | 1579.6 | 9.23s | 5.35s / 5.40s / 5.40s | 194.28s / 123.52s | 3.45s | 128.0 / 100.0% | 85532 |
| 50 | disabled | 41524 / 0 | 1312.5 | 12.68s | 12.36s / 12.41s / 12.46s | 235.08s / 121.70s | 3.91s | 124.0 / 91.2% | 98948 |
| 50 | enabled | 41524 / 0 | 1533.1 | 14.53s | 7.11s / 7.11s / 7.12s | 229.17s / 121.87s | 3.85s | 128.0 / 100.0% | 93160 |

The public verdict of all four rounds is `PASS`:

- both 32-cluster rounds succeeded with 128/128 workflows and 960/960 commands;
- both 50-cluster rounds succeeded with 200/200 workflows and 1500/1500 commands;
- HTTP 429/503, final non-2xx, transport retries and executor claim/result errors were all 0;
- duplicate workflow steps, duplicate claims, fencing mismatches and permanent budget waiters were all 0;
- PostgreSQL deadlocks and the control-plane Pod restart delta were both 0;
- the maximum DNS resolution attempts was 1 in every round, and the process cache covered all requests;
- after cleanup, Job, Pod, incident, workflow, command and synthetic registry residue were all 0.

The changes of enabled relative to disabled at the same scale are as follows. A positive
Throughput value means an increase; negative latency and drain values mean improvement.

| Clusters | Throughput | Fault p99 | Inventory p99 | GPU metrics p99 | Host p99 | Workflow p99 | Drain |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | +9.9% | +37.6% | -46.5% | -44.5% | -44.8% | -5.9% | -20.1% |
| 50 | +16.8% | +14.6% | -42.5% | -42.7% | -42.9% | -2.5% | -1.6% |

Reading:

- enabled markedly lowered the p99 of the three routine telemetry kinds at both scales and
  improved workflow completion p99; the 50-cluster throughput gain is higher than at 32 clusters;
- enabled also raised Fault ingress p99, showing that spool writes and fault admission still
  compete for Aurora or Store I/O resources; the mode must not be chosen on total throughput alone;
- both enabled samples reached 128 ACU and 100% Aurora CPU; there is no obvious database headroom
  under the formal burst;
- scaling from 32 to 50 clusters lowered disabled throughput by 8.7% and enabled throughput by
  2.9% while Fault and workflow p99 kept rising, showing that this fixed resource configuration
  does not scale linearly with cluster count;
- the absolute `xact_rollback` values of this group are high, but there were no deadlocks, HTTP
  errors or workflow failures; before becoming the production default the rollbacks must still be
  broken down by SQL cause and A/A and A/B repeated at least three times;
- the single DNS pre-resolution and process cache eliminated the DNS query storm the cold-connect
  load Pods used to produce; this group had no recovered or final transport retries at all.

These four rounds are the current release's formal integrated ingress, workflow and remote-command
evidence, but each mode still has only one sample. The trend can be used for regression and
capacity judgement and cannot replace multi-round variance, cost and target SLO review.

## 14. Reading the Reference Results

The redacted experiments above are for capacity trends and regression comparison and do not
directly replace the formal PASS/FAIL of the current release. No uniform fixed veto line on P0
ingress p99 is adopted; each round's result should be judged separately against acceptance rate,
rejections, queue recovery, action thresholds and the latency budget approved in advance for that
round:

- the 50-cluster, Max ACU 64 samples contain large numbers of 503s and fail the ingestion and
  admission thresholds;
- the high-Min 96/128 ACU samples eliminated rejections, but their second-level admission wait
  should be compared with that round's approved budget and the historical regression baseline,
  not judged FAIL merely for exceeding some uniform fixed value;
- the single-cluster complete model achieved zero rejections with a P0 completion p99 of
  13.17 seconds; that value is for judging completion-stage capacity and version regression and is
  not the same as a uniform ingress veto item.

Aurora CPU in the complete single-cluster model was only 39.2%, but the processor lane deltas were:

```text
node lane: 435 waits, average 2.45s
attempt lane: 374 waits, average 1.72s, maximum 10.92s
cluster lane: 0 waits
retry reschedule: 0
```

This shows the main bottleneck of this model has moved from Aurora capacity to the node/attempt
ordering lanes and admission holder time. The total CPU throttling seen on the control-worker is
small and cannot explain second-level latency.

## 15. Performance Recommendations

1. **Do not just raise Max ACU for short bursts.** Pre-provision Min ACU before known drills or
   large job windows, and generate load only after `ServerlessDatabaseCapacity` has actually
   arrived. Restore afterwards per the cost policy; do not stop at `modify-db-cluster` having
   returned.
2. **Prefer enabled + dedicated spool-worker at 50 clusters.** It markedly improves P50
   completion latency and routine telemetry acceptance, but may raise Fault admission latency when
   the database saturates; Aurora CPU, DBLoad and Fault p99 must be watched together.
3. **A spool cluster depth of at least 4096 is recommended for a 1000-node single cluster.**
   3000 synchronous P100 items trip the expected 429 at depth1024. 4096 should only be a capacity
   value in the very-large-cluster profile and must not be turned into the global default for all
   sites without verification; the per-cluster cap must also be kept.
4. **Shorten the node/attempt lane holder first.** Move parallelizable parsing, evidence writes
   and policy precomputation outside the lane, and commit incident/workflow only in a short
   transaction; pure telemetry/evidence should be separated from node-control by conflict domain.
5. **Add complete performance evidence for the lane holder.** Artifacts must collect the path
   label of `gpu_fault_processor_lane_holder_seconds` and write the before/after delta into the
   summary, not only the cumulative lane wait.
6. **Continue A/A and A/B on batch delay as a candidate configuration.** The current values
   improve completion p99 but increase rollbacks; repeat at least 3 rounds and check rollback
   causes, pool checkout and admission batch expired before it becomes the production default.
7. **Distinguish cold and warm connection SLOs.** Cold results include TCP/TLS setup, warm results
   measure steady-state service capability; both must be saved, and warm results must not replace
   the public/NLB first-request threshold.
8. **The load tool must verify the processor final response.** An ingress 202 only means
   enqueued; a dry run once found background 422s. The formal tool should judge any non-2xx
   receipt/final response as FAIL directly.
9. **Isolate periodic cloud queries for the synthetic registry.** An audit cluster with a
   placeholder HyperPod name produces `ResourceNotFound` logs. Add an explicit synthetic marker so
   the identity refresher skips these entries and avoids polluting logs and a small amount of
   control-plane resources.
10. **Before raising ACU further, check whether the bottleneck is still the database.** When
    Aurora CPU is already below about 70%–80% while lane wait is still at the second level, adding
    ACU has limited benefit; turn to transaction, lane and completion parallelism optimization.
11. **Do not raise the production remediation budget just to pass the raw-capacity tier.** The
    production-budget tier first proves that the defaults limit fault storms without starvation;
    the raw-capacity tier raises the quota only in the isolated environment to expose the next
    layer of dispatcher, Store and Executor bottlenecks. If the business genuinely requires
    repairing more than 2 same-resource-class workflows per cluster at once, that must be approved
    jointly on multi-round capacity evidence, failure-domain risk and recovery time objectives.
