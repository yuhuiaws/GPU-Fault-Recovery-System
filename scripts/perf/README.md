# Performance tooling

The supported regional capacity entry point is:

```bash
export GPU_FAULT_PERF_CONTROL_NAMESPACE=gpu-fault-perf-system
export GPU_FAULT_PERF_DATAPLANE_NAMESPACE=gpu-fault-perf-system
export GPU_FAULT_AURORA_CLUSTER_ID=<aurora-cluster-id>

scripts/perf/regional_capacity_suite.py all \
  --case burst \
  --clusters 32
```

`all` and `run` refuse to start without `GPU_FAULT_AURORA_CLUSTER_ID`: the run
resolves the Aurora writer from that cluster before it registers anything,
reads `aurora.json` for that writer, and records the writer instance, its
Availability Zone and the control-plane Pods' node zones in
`run.json.placement` and `summary.json.placement`
(`regional_capacity_placement.py`). The integrated runner records the same
from `--aurora-cluster-id`; the action suite records it when the variable is
set and otherwise marks it `recorded: false`.

The default target is an isolated control-plane/data-plane namespace. The
isolated control plane must mount its own regional registry Secret and the
data-plane namespace must contain a connection Secret pointing to that control
plane. It must also use a disposable PostgreSQL database; a separate namespace
that still writes the production database is not isolated. The names can be
overridden with:

```text
GPU_FAULT_PERF_REGISTRY_SECRET
GPU_FAULT_PERF_CONNECTION_SECRET
```

The perf data-plane namespace needs only the connection Secret, load-generator
ServiceAccount/RBAC, and simulator Jobs. It must not run a real Cluster
Executor, Watcher, Collector, or Node Runtime. In isolated mode, the action
suites derive short-lived synthetic Agent and Executor identities from the CPU
release state and its active required pins. Any pin mismatch fails closed.
Live-registry mode retains the real Agent and Executor Deployment cross-checks.
The Executor pin cross-check reads the real Deployment from
`GPU_FAULT_PERF_IDENTITY_NAMESPACE` (default `gpu-fault-system`); load and
simulator Jobs remain in `GPU_FAULT_PERF_DATAPLANE_NAMESPACE`.

Mutating the production `gpu-fault-system` registry requires both
`--allow-live-registry` and
`--confirm-live-registry ALLOW_PERF_CAPACITY_LIVE_REGISTRY`.

Synthetic registrations carry a run ID and expiration. `all`, `run`, and the
action, integrated and correlated suites always execute the same idempotent
teardown in `finally`; registry updates publish one CAS revision and wait for
all CPU process ACKs without restarting CPU Deployments. The purge and the
deregistration both run inside that teardown, which is retried up to three
times (after 15, 45 and 90 seconds) and verified against both the bootstrap
Secret and runtime registry; no attempt deregisters while the run's data is
still present. When the retries are spent the run keeps its receipts,
`status.json` is `aborted` with `cleanup_error` and `recovery_command`, and the
log prints `recovery command: ...`, one line that runs
`regional_capacity_suite.py teardown` with the run's `--suite-id` and `--run-dir`
(integrated and correlated runs: the directory under `_aborted/`). A purge whose
final read still finds the run's data deletes nothing; its error counts what is
left per record kind (appeared after its walk / still present from it) and per
table, without keys or payloads. The purge waits for the run's data to settle
first and probes again 30 s after it, purging what landed late (a
collector-silence scan that read the run's Agents before the commit), at most
three more times, each one logged by kind; a registration adopts residue only
when it is provably an earlier run's drill residue and runs that run's purge
(`regional_capacity_settle.py`). Every suite's
`run.json` carries the resume fields that command checks. A standalone
`register` command requires `--keep-registration`, a shared `--suite-id`, and
must be followed by `run` or `teardown`. Separate `run`, `purge`, and `teardown`
commands require both the original `--suite-id` and `--run-dir`; the directory
contains the registration intent, exact cluster inventory, release identity,
and resource ownership receipts. A new directory or a cluster-name prefix is
not sufficient authorization to resume or delete a previous run.

Kubernetes resources are created with run ownership, and deletion requires the
observed UID. An unacknowledged create remains unresolved until its identity
can be proved; cleanup never treats an unknown resource as absent. Database
cleanup resolves the current projected credentials inside the CPU component,
checks run ownership and terminal state, and handles the active legacy, dual,
or dedicated workflow/remote-command representation. It does not remove another
run's records or publish a successful verdict after an incomplete cleanup.

Supported entry points:

- `regional_integrated_workflow_capacity_suite.py`: formal live CPU-control-plane
  run combining the fixed 26,576-request matrix with four causally linked
  action workflows per synthetic cluster and simulated destructive execution.
- `regional_capacity_suite.py`: registration, mixed/burst/multicluster load,
  metrics, Aurora sampling, teardown and artifacts.
- `regional_action_capacity_suite.py`: regional action execution capacity; it
  records the five remediation budget limits in `run.json` and fails a run
  (exit 3) when any `history.json` sample holds the region, a cluster or a
  resource class with more RUNNING workflows than its limit. Every run writes
  `remediation-budget.json` (`regional_remediation_budget.py`): the five
  tiers, the per-scope peaks of that verdict, the refused claims, waited and
  never-executed workflows and the longest wait from the run's workflow rows,
  and `lease_expiry_releases: not-applicable` (claims are held until a
  workflow ends).
- `regional_correlated_action_suite.py`: live synthetic event-to-action
  same-rank aggregation, preemption and failure-escalation chains, including
  budget-wave-aware scenario timeouts, a separate terminal-drain phase, exact
  workflow relationships and complete workflow/command terminal-state audit.
- `seed_regional_action_workflows.py` and
  `benchmark_regional_action_executor.py`: action-side data and load drivers.
- `capture_processor_inflight.py`: optional in-flight sampling attached to a
  suite artifact directory.
- `control_plane_capacity_probe.py`: control-plane capacity probe helper.
- `ingress_event_loop_burst.py`: local-only reproduction of the ingress liveness
  kill under a cold-connect burst - the real application in the rendered ingress
  environment under uvicorn's production flags, a CPU-burning flush stub, a
  kubelet-like `/livez` prober and per-process samplers (event-loop lag, GIL
  wait, GC pauses, RSS, accept queue). It touches no cluster and no database.

`payload-templates.json` and the three benchmark modules selected by
`regional_capacity_suite.py` are implementation inputs, not standalone entry
points.

Any script that opens psycopg inside a control-plane Pod -- an embedded
`cpu_python(...)` literal or a whole file piped to `python3 -` -- must take the
DSN from `store_dsn()` (the `STORE_DSN_SNIPPET` in `regional_capacity_registry.py`,
or a verbatim local copy in a stand-alone tool) so it re-reads
`GPU_FAULT_STORE_URL_FILE` across an Aurora password rotation instead of the
`GPU_FAULT_STORE_URL` value frozen at Pod start.

`benchmark_correlated_action_scenario.py` is likewise an implementation input
for `regional_correlated_action_suite.py`. It must not be launched without the
suite's synthetic registry, token, TLS, cleanup and evidence guards.

The formal integrated run is:

```bash
export GPU_FAULT_CONTROL_KUBECONFIG=<cpu-control-plane-kubeconfig>
export GPU_FAULT_DATAPLANE_CONTEXT=<approved-gpu-perf-context>
export GPU_FAULT_PERF_AWS_REGION=<aws-region>
export GPU_FAULT_PERF_CONTROL_NAMESPACE=gpu-fault-system
export GPU_FAULT_PERF_DATAPLANE_NAMESPACE=gpu-fault-perf-system
export GPU_FAULT_PERF_IDENTITY_NAMESPACE=gpu-fault-system

scripts/perf/regional_integrated_workflow_capacity_suite.py \
  --clusters 32 \
  --spool-mode disabled \
  --aurora-cluster-id <aurora-cluster-id> \
  --configure-aurora-capacity \
  --confirm-aurora-scaling SET_AURORA_MIN_124_MAX_128 \
  --workflows-per-cluster 4 \
  --allow-live-registry \
  --confirm-live-registry ALLOW_PERF_CAPACITY_LIVE_REGISTRY
```

It targets the live CPU ingress/worker/Aurora path. Synthetic cluster identity,
Agent records and the executor isolate GPU mutations; reset and reboot commands
sleep for configured representative durations and return simulated results.
Each claimed command reports when its own duration elapses and its worker slot
is claimed again at once, so a 120 s reboot no longer holds the next commands
of the cluster's other workflows (`completion_mode: per-command`).
Before registration it first records the current Aurora scaling range, sets
`Min=124, Max=128` when needed, then requires two available `db.serverless`
instances with observed capacity at least 124 ACU and exact live remediation
budgets for the selected 32/50-cluster matrix.
The formal matrix contains four runs: 32/50 clusters crossed with
`--spool-mode disabled/enabled`. The runner verifies every live ingress Pod and
the dedicated spool-worker replica count before registering synthetic clusters.
It also rejects a live ingress command containing Uvicorn
`--limit-max-requests`; request-count recycling can overlap during a burst and
remove serving capacity even though Kubernetes reports no container restart.
The result is written to `ingress-process-model-preflight.json` before Aurora
scaling or synthetic registry registration.
Apply the matching `32-disabled`, `32-enabled`, `50-disabled`, or `50-enabled`
AdminConfig preset before each run: set `spec.capacity.preset` in
`<state-dir>/admin-config.yaml` and run `gpu-fault-admin config --state-dir
<state-dir>` (there are no capacity flags; the YAML is the only input). The
current command totals are 30 per cluster: 960 for 32 clusters and 1500 for 50.

For the whole workflow phase `regional_validation_telemetry.py` keeps one
`kubectl exec -i` session of the seeding script's `integrated-validation-watch`
mode in an api-ha Pod: as soon as the control plane records a node's
reset/reboot step it publishes that node's healthy telemetry, stamped with the
actual publish time (post-action validation accepts only samples newer than
that record). The program arrives length-prefixed on stdin and closing stdin
stops the session; `summary.json.validation_telemetry` holds its evidence.
As the fallback, before the start gate and every 30 seconds during workflow
drain, the runner refreshes synthetic Agent leases, required collector success,
healthy GPU/NVLink/RDMA samples, and fresh `load1_per_cpu`,
`memory_used_percent`, and `filesystem_used_percent` host samples, each node
stamped with its own write time. Host sample timestamps must cross the
completed reboot barrier so `VALIDATE_HOST` cannot pass on stale telemetry.

Integrated cleanup ignores subsequent `SIGINT`, `SIGTERM`, and `SIGHUP` while
its bounded `finally` block deletes both Jobs and their Pods, purges database
rows, and removes the synthetic registry revision. A completed teardown writes
`cleanup-workloads.json`, `cleanup-residuals.json`, and
`registry-postflight.json`; any residual keeps the run aborted.

The integrated evidence distinguishes final client failures from recovered
first-attempt transport failures. Each ingress path reports
`transport_retries`; executor claim failures report bounded, redacted
`claim_error_counts` and `claim_error_samples`. The runner also writes
`control-pods-before.json` and `control-pods-after.json` and rejects any
control-plane Pod replacement, container restart, counter regression, or Pod
that is not Ready after the run. These checks do not change request, timeout,
concurrency, or capacity thresholds.

The synchronized ingress generator resolves the control-plane hostname with a
bounded retry before the start gate and installs a process-local rotating
address cache. Requests still use the declared `cold-connect` TCP/TLS model and
the original hostname for HTTP Host, TLS SNI, and certificate verification, but
the burst no longer turns every request into a separate CoreDNS query. Evidence
records DNS resolution attempts, address count, and cache hits; the integrated
verdict rejects a pre-resolution retry, missing cache, incomplete cache
coverage, or any request transport retry.

The runner records the previous Aurora Min/Max in
`aurora-preflight.json.initial_scaling` but does not restore it. After the final
round, restore Aurora through the approved AWS/IaC path and reapply the pre-run
AdminConfig. Do not continue after a failed verdict, cleanup residual, registry
residual, nonterminal workflow/command, or critical alert.

After the measured window closes, load and executor Pod logs are fetched with a
bounded pool of at most 16 concurrent `kubectl logs` calls. Results are sorted
by Indexed Job completion index before artifact writes and aggregation, so this
reduces operator wait time without changing the measured window or verdict.

The supported CLI remains in `regional_capacity_suite.py`; its implementation
is split by responsibility:

- `regional_capacity_job.py`: Indexed Job and start-gate manifest construction.
- `regional_capacity_database.py`: Aurora and PostgreSQL observations.
- `regional_capacity_timeseries.py`: one sampling session per control-plane Pod
  through the load and the drain (processor in-flight, workers, queue depth and
  consumer cycles once per second), and the drain probe that starts at the
  load stop.
- `regional_capacity_placement.py`: the run's Aurora writer, its Availability
  Zone and the control-plane node zones.
- `regional_validation_telemetry.py`: the integrated and correlated runners'
  event-driven validation telemetry session (the correlated one publishes a
  node's node-return package right after its reboot step is recorded; the
  30 s heartbeat refresh stays the fallback, every node stamped with its own
  write time).
- `regional_capacity_registry.py`: isolated/live target selection, synthetic
  registry metadata, residual detection and verified cleanup.
- `regional_capacity_results.py`: metric parsing, aggregation and artifact
  lifecycle.
- `regional_capacity_resources.py`: durable resource receipts and UID-fenced
  creation/cleanup.
- `regional_capacity_data.py`: run-scoped database census and terminal cleanup
  using current projected credentials.

Additional focused utilities:

- `benchmark_processor_admission_storm.py`
- `generate_watcher_pod_storm.py`
- `validate_hot_state_defaults.py`

Superseded one-Job-per-benchmark harnesses are retained only in the private
archive and are not part of the public performance tooling.
