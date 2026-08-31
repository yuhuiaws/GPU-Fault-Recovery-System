# Performance tooling

The supported regional capacity entry point is:

```bash
export GPU_FAULT_PERF_CONTROL_NAMESPACE=gpu-fault-perf-system
export GPU_FAULT_PERF_DATAPLANE_NAMESPACE=gpu-fault-perf-system

scripts/perf/regional_capacity_suite.py all \
  --case burst \
  --clusters 32
```

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

Mutating the production `gpu-fault-system` registry requires both
`--allow-live-registry` and
`--confirm-live-registry ALLOW_PERF_CAPACITY_LIVE_REGISTRY`.

Synthetic registrations carry a run ID and expiration. `all`, `run`, and the
action-capacity suite always execute idempotent teardown in `finally`; registry
updates publish one CAS revision and wait for all CPU process ACKs without
restarting CPU Deployments. Cleanup is retried and verified against both the
bootstrap Secret and runtime registry. A standalone
`register` command requires `--keep-registration`, a shared `--suite-id`, and
must be followed by `run` or `teardown`.

Supported entry points:

- `regional_integrated_workflow_capacity_suite.py`: formal live CPU-control-plane
  run combining the fixed 26,576-request matrix with four causally linked
  action workflows per synthetic cluster and simulated destructive execution.
- `regional_capacity_suite.py`: registration, mixed/burst/multicluster load,
  metrics, Aurora sampling, teardown and artifacts.
- `regional_action_capacity_suite.py`: regional action execution capacity.
- `regional_correlated_action_suite.py`: live synthetic event-to-action
  same-rank aggregation, preemption and failure-escalation chains, including
  budget-wave-aware scenario timeouts, a separate terminal-drain phase, exact
  workflow relationships and complete workflow/command terminal-state audit.
- `seed_regional_action_workflows.py` and
  `benchmark_regional_action_executor.py`: action-side data and load drivers.
- `capture_processor_inflight.py`: optional in-flight sampling attached to a
  suite artifact directory.
- `control_plane_capacity_probe.py`: control-plane capacity probe helper.

`payload-templates.json` and the three benchmark modules selected by
`regional_capacity_suite.py` are implementation inputs, not standalone entry
points.

`benchmark_correlated_action_scenario.py` is likewise an implementation input
for `regional_correlated_action_suite.py`. It must not be launched without the
suite's synthetic registry, token, TLS, cleanup and evidence guards.

The formal integrated run is:

```bash
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
Before registration it first records the current Aurora scaling range, sets
`Min=124, Max=128` when needed, then requires two available `db.serverless`
instances with observed capacity at least 124 ACU and exact live remediation
budgets for the selected 32/50-cluster matrix.
The formal matrix contains four runs: 32/50 clusters crossed with
`--spool-mode disabled/enabled`. The runner verifies every live ingress Pod and
the dedicated spool-worker replica count before registering synthetic clusters.

The supported CLI remains in `regional_capacity_suite.py`; its implementation
is split by responsibility:

- `regional_capacity_job.py`: Indexed Job and start-gate manifest construction.
- `regional_capacity_database.py`: Aurora and PostgreSQL observations.
- `regional_capacity_registry.py`: isolated/live target selection, synthetic
  registry metadata, residual detection and verified cleanup.
- `regional_capacity_results.py`: metric parsing, aggregation and artifact
  lifecycle.

Additional focused utilities:

- `benchmark_processor_admission_storm.py`
- `generate_watcher_pod_storm.py`
- `validate_hot_state_defaults.py`

Superseded one-Job-per-benchmark harnesses are retained only in the private
archive and are not part of the public performance tooling.
