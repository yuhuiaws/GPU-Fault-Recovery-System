English edition of `docs/管理员容量配置.md`; the Chinese file remains the source of record until both are maintained together.

# Administrator Capacity Configuration

This document defines the formal entry point through which administrators adjust the capacity of the regional control plane and Aurora. A capacity change does not modify source code or
rebuild images or wheels; the CLI reuses the currently signature-verified release, performs an audited RDS change on Aurora, and rolls the affected CPU roles only for configuration that
needs a process reload.

Administrators need to do only two things: edit `<state-dir>/admin-config.yaml`, and run
`gpu-fault-admin config --state-dir <state-dir>`.

## 1. Supported Scope

Administrators can currently configure:

- The control-worker replica count;
- Site topology: the maximum single GPU cluster node count (`largestClusterNodeCount`) and the total managed node count
  (`managedNodeCount`);
- telemetry spool enabled/disabled, the spool-worker replica count and the single-cluster spool depth
  `maxClusterDepth` (64..65536, default 1024; beyond that depth a single cluster's routine telemetry receives reserved 429.
  Per the complete model of a 1000-node single cluster in Performance Acceptance Plan §13.4: at 1024, 1150 of 3000 routine telemetry requests got 429;
  at 4096 all 6500 requests got 202 with P0 and evidence unchanged. 4096 is recommended when a single cluster has >= 512 nodes;
  the default 1024 corresponds to the preset 256-node cluster model);
- The Region, cluster and resource class caps of the remediation budget;
- Aurora Serverless v2 Min/Max ACU;
- Processor total queue, single-cluster queue, client backoff, persistent retry and completed record retention;
- Workflow scan period and dispatcher thread count;
- Notification batch size and maximum attempts;
- Raw evidence retention hours and per-node record cap.

The formal template contains 22 administrator-configurable leaf fields in total. The node and failure-domain remediation budgets
are safety constants fixed at 1 at runtime and do not appear in the template; if an old file still has
`maxActivePerNode: 1` or `maxActivePerFailureDomain: 1` it is accepted and ignored, and any other value
is refused.

Credentials, Region, cluster identity, Runtime Profile, operation allowlist, release pins,
destructive capability switches and safety invariants are not general administrator configuration. Only the fields declared by the formal template may be changed;
arbitrary `GPU_FAULT_*` variables must not be passed through to Pods.

## 2. Release Template

The source release contains:

```text
config/admin-config.example.yaml
```

After the signed deploy-host bundle is installed, the same template is at:

```text
<deploy-host-venv>/share/gpu-fault/admin-config.example.yaml
```

The template is a read-only release asset with mode `0644`. The administrator's actual input must live in a controlled directory with mode `0600`.

## 3. First Deployment

When the administrator has already decided the capacity values before the first deployment, prepare your own configuration file from the release template, then apply it directly with one
deploy command:

```bash
gpu-fault-admin deploy \
  --cpu-cluster-arn <cpu-arn> \
  --gpu-cluster-arn <gpu-arn> \
  --state-dir /secure/gpu-fault \
  --admin-email <operations-email> \
  --config /secure/gpu-fault-input/admin-config.yaml
```

The CLI validates the configuration before the source build or any AWS/Kubernetes access, writes the normalised values into
`<state-dir>/admin-config/desired.json`, and writes the editable copy to:

```text
<state-dir>/admin-config.yaml
```

A first deployment does not need `config` to run first. Without `--config` the release defaults are used and a default
`<state-dir>/admin-config.yaml` is generated.

## 4. Modifying an Existing Site

### 4.1 Command

```text
gpu-fault-admin config --state-dir STATE_DIR [--file ADMIN_CONFIG] [--dry-run] [--reference REFERENCE]
```

Under the same verb there is also a sub-action `gpu-fault-admin config spare --state-dir STATE_DIR --node NODE [--declare|--release ...]` for declaring or releasing warm-spare GPU nodes; it does not involve the capacity configuration of this document, see the task table of [Administrator Operations](administrator-operations.md).

After editing `<state-dir>/admin-config.yaml` the administrator simply runs:

```bash
gpu-fault-admin config --state-dir /secure/gpu-fault
```

- `--file`: temporarily use another `0600` file as input; on success the CLI writes the normalised result back to
  `<state-dir>/admin-config.yaml` so it always equals the online desired values.
- `--dry-run`: reads only the local `desired.json` and YAML, prints the change plan (affected roles, before/after values per field,
  whether Aurora changes) and exits. No Cosign signature verification, no cluster reads, no executor identity resolution, no state file written,
  and no migration of a legacy `desired.json`. A missing input file is refused outright without creating the template or changing
  directory permissions; the ordinary non-dry-run command can still initialise a missing default template.
- `--reference`: optional change ticket or maintenance window number, 3 to 128 letters, digits or `. _ : / -`.
  When absent it is recorded as `<executor identity>:<start time>`. The executor identity is always taken automatically from
  the caller ARN of `aws sts get-caller-identity` (degrading to `user@host` when unobtainable) and enters the audit record together with the
  reference; administrators need not and cannot fill in the identity by hand.

Fields absent from the file inherit the currently persisted values. The CLI refuses unknown fields, wrong types, a spool switch inconsistent with its replicas,
cluster/resource-class greater than Region, `managedNodeCount` less than
`largestClusterNodeCount`, combinations exceeding the accepted PostgreSQL fleet connection budget, and combinations violating the §5
topology derivation rules (for example raising the managed node count to 2048 without raising `aurora.minAcu` to at least 20.5).
The error message gives the exact next command to run.

### 4.2 Apply Flow

Without `--dry-run`, the whole command holds the site operation lock (the same lock as `deploy` and `join-cluster`)
and, in order:

1. Reads `desired.json` (a legacy record is first verified per §4.5 and atomically completed) and the YAML, computes the target configuration and validates it
   fully; with no change it prints `NOOP` and exits without writing any file.
2. Checks the content-addressed Manifest against the Cosign attestation and reads the live release
   state in the CPU cluster: it must be the current signed release and already committed, otherwise fail closed (except the resume exception of §4.3).
3. Writes the audit: `history/<start time>-<target configuration digest>/before.json` (the
   `desired.json` before the change), `pending.json` (configuration before and after, site identity, signed release identity,
   executor, reference, start time), then changes `desired.json` to the target configuration.
4. When Aurora Min/Max changed, first calls RDS `modify-db-cluster` and waits for the cluster, writer and reader
   to return to `available` on the new window (three consecutive observations).
5. Rolls only the affected CPU roles and completes verify/stability/commit.
6. When raising Min ACU, only now waits for the `ServerlessDatabaseCapacity` of both writer and reader to reach the new
   Min. This step and the role rollout are independent; it is placed after the rollout so that the scale-up ramp time does not serially occupy the whole
   command.
7. Writes `history/<...>/after.json` and `result.json` (`APPLIED`), deletes `pending.json`,
   and writes the normalised complete 22 items back to `<state-dir>/admin-config.yaml`.

When steps 4 to 6 fail, the CLI re-reads the live release state; Aurora cannot be restored on the exit code of the release subprocess alone.
Only when it is proven that the roles are still on the originally committed configuration, that a matching configuration release has not yet started role changes, or that a matching
rollback has been verified and cleaned up does it restore the pre-change Min/Max and `desired.json`.
`result.json` is recorded as `FAILED` with `details.release_recovery` and the actual Aurora compensation result;
`pending.json` is kept, and the administrator's YAML is not rewritten.

A failed finalisation of a committed candidate, disabled automatic rollback, a failed rollback, an incomplete state or a read failure all keep the pending
target and do not automatically lower the database capacity. If the release subprocess has partially aligned the local management configuration, the CLI writes back
the canonical target and original approval information from pending, avoiding misrecording an unknown outcome as the old configuration. First reconcile the release per the error hint,
then rerun `config`; legacy or incomplete recovery evidence cannot be used to presume success.
When only step 6 times out waiting for the actual ACU ramp and the role release already succeeded, the Aurora window and target configuration are not reverted;
rerunning the same `config` completes the wait. If the Aurora compensation itself fails, the target is likewise kept and it fails closed.

### 4.3 Resuming After an Interruption

When the same command is interrupted at any step, rerunning with the same `--state-dir` (and the same YAML) resumes:
`pending.json` is matched by site identity, signed release identity and target configuration digest. A rerun of an attempt already recorded as failed
is a new attempt with its own `history/<new start time>-<digest>/` directory, executor and reference;
an attempt that crashed mid-way without a `result.json` continues in the original directory. A rerun with a different target configuration marks the old
`pending.json` `SUPERSEDED` and replaces it.

The requirements on live state for a resume are unchanged: an uncommitted live release in the CPU cluster is accepted only if it is
`CONTROL_PLANE_ONLY`, its `admin_config_sha256` equals the target configuration digest in `pending.json`, and it is in a
phase the release engine can resume; any other uncommitted state fails closed and suggests first finishing that release with
`gpu-fault-admin deploy --state-dir <state-dir>`. On the Aurora side the CLI accepts only a
live window that is still the pre-change value or the target value; other live values fail closed as drift. A `desired.json` changed during the interruption
is also refused, naming the file to check.

### 4.4 State Files

```text
<state-dir>/admin-config/
  desired.json                      Authoritative desired configuration: configuration, configuration digest, three role digests,
                                    source, reference, executor identity, update time
  pending.json                      The in-progress or interrupted application; deleted on success
  history/<start time>-<configuration digest>/
    before.json                     desired.json before this application started
    after.json                      desired.json after this application ended
    result.json                     APPLIED / FAILED / SUPERSEDED, release ID,
                                    affected roles, executor, reference, error and Aurora details
```

Both `desired.json` and `pending.json` carry digest self-checks; changed content fails closed.

### 4.5 Upgrading Legacy Records

When upgrading from a desired state containing only `capacity`, the legacy 18 fields, or the 20 fields lacking the node count fields, the CLI first doubly verifies
by the legacy configuration digest and the three role digests, then atomically completes the missing fields inside the site operation lock. When an old site lacks
the Aurora fields they are kept at the historical `0.5/8`, so a software upgrade does not silently change the database; new sites default to `8/32`. When an old record
lacks node counts, the topology is inferred from its own declared single-cluster queue depth (`min(512, depth / 4)`; the historical
`1024` means 256 nodes, and the managed node count takes the same value), so a cluster that depth cannot hold is not read in at the new default 512. An old
record with unknown structure, modified content or any digest mismatch still fails closed. `--dry-run` only reads and does not
complete.

Reading persisted or online-captured state performs only structural validation without applying the empirical floors of §5: an old site below the floors is
a fact, and refusing to read would only prevent the administrator from loading the state to plan a fix. Every entry point that generates a new desired (file,
preset, plan) validates fully, and a combination below the floors is refused naming both fields.

### 4.6 Capacity Presets

Presets are declared through the same YAML; there is no separate command-line parameter:

```yaml
apiVersion: gpu-fault.aws/v1alpha1
kind: AdminConfig
spec:
  capacity:
    preset: 32-disabled
```

The `preset` is applied first, then other explicit fields in the same file override on top; non-capacity fields inherit from the currently persisted
AdminConfig. A preset names a topology, and the topology decides the Aurora Min ACU floor (see §5): when the current
Min ACU is below that preset's floor, the plan raises Min ACU to the floor and Max ACU to `max(current Max, 128)`,
entering the audit and RDS path like any other Aurora change; an Aurora already above the floor is not lowered. The `124/128` used by the formal stress
test is above the `82` floor of `32-*` and must still be set explicitly; it cannot be inferred from `32-enabled`;
the floor of `50-*` is exactly `128/128`. `preset: default` explicitly restores the release default capacity.

Supported presets (multi-cluster topologies count 256 nodes per cluster, see Performance Acceptance Plan §2):

| preset | worker replicas | Region budget | per cluster | per resource class | spool | max cluster / managed nodes | Aurora floor |
|---|---:|---:|---:|---:|---|---:|---:|
| `default` | 6 | 20 | 5 | 2 | disabled, 0 replicas | 512 / 512 | 8/32 (template value) |
| `32-disabled` | 6 | 128 | 4 | 4 | disabled, 0 replicas | 256 / 8192 | 82/128 |
| `32-enabled` | 5 | 128 | 4 | 4 | enabled, 3 replicas | 256 / 8192 | 82/128 |
| `50-disabled` | 6 | 200 | 4 | 4 | disabled, 0 replicas | 256 / 12800 | 128/128 |
| `50-enabled` | 5 | 200 | 4 | 4 | enabled, 3 replicas | 256 / 12800 | 128/128 |

### 4.7 PostgreSQL Connection Budget

The runtime, the administrator validation and the capacity runner share
`postgres_capacity.PostgresPoolCapacity`, computing the total cap from each role's replica count, processes per Pod,
pool cap and separate LISTEN connections. Each worker process additionally needs 1 processor queue and
2 dispatcher connections; the spool-worker needs 1 spool connection. Each regional process further reserves 1 on-demand
remote-command claim connection, which cannot be omitted just because no long poll has been received yet.

The cap of the default disabled combination is 1164, below the original 1200 budget; the enabled preset with 3 spool replicas plus 5 workers
is 1190, below the original 1240 budget. enabled with the inherited 6 workers is 1302 and cannot pass the new target validation.
The default worker count of a preset is computed only when the preset is chosen explicitly; an administrator's explicit override is not silently shrunk, and an ordinary upgrade
does not automatically rewrite the existing topology. Old configurations may be loaded read-only to plan the correction; structure, digest and new-target validation are unchanged.

This is the connection cap of this solution's processes, not measured concurrency, and it reserves no capacity for Aurora management connections, release surge,
query tools or other applications. The 6-worker enabled results of historical stress tests remain a historical configuration
and do not represent the current presets or authorise raising the 1200/1240 budgets.

## 5. Configuration File Format

The current template content is:

```yaml
apiVersion: gpu-fault.aws/v1alpha1
kind: AdminConfig
spec:
  capacity:
    controlWorkerReplicas: 6
    largestClusterNodeCount: 512
    managedNodeCount: 512
    telemetrySpool:
      enabled: false
      replicas: 0
      maxClusterDepth: 1024
    remediation:
      maxActiveRegion: 20
      maxActivePerCluster: 5
      maxActivePerResourceClass: 2
  aurora:
    minAcu: 8
    maxAcu: 32
  processor:
    maxQueueDepth: 65536
    maxClusterQueueDepth: 4096
    retryAfterSeconds: 2
    retryBackoffSeconds: 1
    retryBackoffMaxSeconds: 30
    completedRetentionSeconds: 600
  workflow:
    pollIntervalSeconds: 5
    dispatcherWorkers: 8
  notificationDelivery:
    batchSize: 25
    maxAttempts: 8
  evidence:
    retentionHours: 24
    maxRecordsPerNode: 10000
```

The field names in the YAML are the camelCase spelling of the snake_case field names in `desired.json`; both derive from the same
schema, declared once. Fields absent from the file inherit the currently persisted values. An ordinary code upgrade keeps the current
configuration and does not accidentally restore defaults. Internal parameters such as fault reserved depth, the global admission guard, retryable
response age, cleanup batches, workflow batch and notification poll/lease/backlog grace are derived by the code from the public fields or fixed production contracts and do not enter the administrator template.

`largestClusterNodeCount` and `managedNodeCount` are topology inputs, not tuning knobs; the four derivation rules
all come from this repository's stress-test evidence (Performance Acceptance Plan §8.4, §13.3-§13.4) and are shared by the same code for CLI validation,
the two manifest renderers and the runtime startup self-check:

| Rule | Basis |
|---|---|
| `processor.maxClusterQueueDepth >= 4 x largestClusterNodeCount` | §13.4: a single 1000-node cluster produced 243 HTTP 429 at depth 1024, and 0 after raising to 4096 |
| `aurora.minAcu >= ceil(managedNodeCount / 100, 0.5 ACU)` | §13.3: about 12800 nodes produced 3076 HTTP 503 under `0.5/96 low-start`, and 0 with Min pre-provisioned at about 124-128 |
| `GPU_FAULT_INGRESS_NORMAL_CONCURRENCY = clamp(ceil(managedNodeCount / 5), 1000, 3584)` (ingress role only) | §8.4: the 50-cluster preset (12,800 nodes) at the fixed 1000 per process / 2 s wait produced a 0.58-3.85 % reserved 503 share on normal telemetry across six live runs, the 32-cluster preset (8,192 nodes) 0; the 3584 ceiling keeps it plus the 256 fault slots under uvicorn `--limit-concurrency 4096` |
| `GPU_FAULT_PROCESSOR_FAULT_RESERVED_CLUSTER_DEPTH = max(maxClusterQueueDepth / 8, largestClusterNodeCount)` | Product requirement: a cluster-wide correlated fault is one fault-priority request per node landing on as many node lanes, and the reservation must hold a whole wave |

The defaults `512 / 512 / 4096 / 8` satisfy all four (the Aurora floor for 512 nodes is 5.5 ACU; the normal-tier concurrency sits at its 1000 floor). A site with 4 clusters of 512 nodes
must write `managedNodeCount` as 2048 and raise `minAcu` to at least 20.5 at the same time. The node counts are rendered together with
`GPU_FAULT_PROCESSOR_FAULT_RESERVED_*` into the ConfigMaps of the three CPU roles
(`GPU_FAULT_CAPACITY_LARGEST_CLUSTER_NODE_COUNT`,
`GPU_FAULT_CAPACITY_MANAGED_NODE_COUNT`); the runtime refuses at startup an environment whose reservation is smaller than the largest cluster node count,
and `/metrics` exports the two gauges `gpu_fault_capacity_largest_cluster_node_count` and
`gpu_fault_capacity_managed_node_count`. When rollback captures the online AdminConfig, it
reads only the node counts actually present in the ConfigMap; when production declares none, it infers from the captured depth per the legacy record rule above,
without filling in the release default.

## 6. Precedence and Omitted Fields

Runtime configuration precedence from high to low is:

1. Safety invariants and dedicated approvals;
2. Secrets, release identity and compatibility pins;
3. The normalised AdminConfig;
4. role-split renderer production defaults;
5. `os.getenv` fallback defaults in the application code.

The AdminConfig overrides same-named shell variables when merging the deployment environment and is baked into the role ConfigMaps. Duplicate explicit `env`
and `envFrom`, unknown fields or out-of-range combinations all fail before application.

A first deployment has no historical `desired.json`, and omitted fields in the configuration file use the AdminConfig defaults. On an existing site
omitted fields inherit the current `desired.json` values, suited to changing only a few fields with a partial `--file`. After a successful application
`<state-dir>/admin-config.yaml` always writes out the complete 22 items; to restore defaults, fill in the template defaults explicitly,
or write `preset: default` for the capacity as a whole.

## 7. Taking Effect and Rollback

The configuration digest is independent of the code release ID, image digest and wheel digest. A config-only release:

- Uploads or rebuilds no code artefact;
- On Aurora field changes first calls the RDS API and waits for the cluster, writer and reader to be available on the new window, then rolls
  the roles; when raising Min, waits after the role rollout for both instances' actual ACU to land;
- Runs no schema, GPU data-plane, Node Runtime or Runtime Profile phase;
- Publishes no cluster registry revision;
- A remediation budget or worker replica change rolls only the control-worker; a node count change changes the fault reserved depth shared by the three roles,
  and therefore rolls all three CPU roles;
- Enabling spool rolls the spool-worker first, then ingress;
- Disabling spool first turns off ingress admission, waits for spool depth and leased to both be 0, then scales down
  the spool-worker;
- The release state and Deployment metadata record the global configuration digest, and the Pod template records the role digest, so a
  ConfigMap change is not misjudged as `NOOP`.

The releaser saves the role ConfigMaps and the old AdminConfig before the change. On a verify or stability failure it follows the existing
CPU automatic rollback, restoring the old configuration, old replica topology and old role processes; the outer CLI must verify the recovery evidence per §4.2
before compensating Aurora Min/Max. When the target is committed, the rollback unfinished or the outcome unknown, no automatic compensation may happen.
A failed Aurora compensation is recorded separately and fails closed. Running only `kubectl rollout undo` does not constitute a complete configuration rollback.
The releaser's automatic rollback and `deploy --rollback` restore only the role configuration, replica topology and role processes and do not change
the Aurora capacity window: `aurora` in `desired.json` keeps the pre-rollback desired value, `aurora` in the previous release snapshot
comes from the release state of that time (no longer filled in by parser defaults), and the rollback does not write it back either.

## 8. Hot-Update Boundary

CPU configuration is loaded from environment variables at process startup, so it takes effect via Kubernetes rollouts, and generic restart-free
hot updates are not supported. Aurora Min/Max is a separate AWS resource change that can be adjusted online after deployment with the same administrator command, but
still must go through validation, audit, convergence verification and rollback. Replica counts, Uvicorn processes, thread pools, the PostgreSQL pool
and the spool topology are not suitable for in-process hot changes.

If in future some thresholds really must take effect immediately without a restart, configuration revisions, all-process ACK,
a mixed-revision gate, failure rollback and state metrics must be implemented separately. That mechanism must not reuse the cluster registry records, nor
bypass the field whitelist and audit of this document.

## 10. Control Record Retention

archive-first retention of control records (incidents/workflows and their associated markers, notifications, completion records) is **enabled by default**
(the decision of the control-plane component review of 2026-09-08); administrators need to change no file; the default `site.yaml` is equivalent to:

```yaml
spec:
  retention:
    controlRecordRetentionDays: 30          # default 30; only an explicit 0 disables
    # archiveS3Uri: s3://<bucket>/<prefix>  # optional; derived by default from account/Region/site
    # archiveIntervalSeconds: 600           # optional; archive scan period, default 600
```

When `archiveS3Uri` is not written, both site loading and bootstrap resolve the target as
`s3://gpu-fault-control-records-<account>-<region>/<site>/control-record-archive`
(the account from `spec.cpu.eksArn`; from the CPU cluster identity when no site.yaml exists yet at the first bootstrap). The first bootstrap and every `gpu-fault-admin deploy --state-dir`:

1. The bootstrap task `control_record_archive_bucket` creates the bucket (when absent; in the site Region, us-east-1 without
   LocationConstraint) and on every run enforces versioning, SSE-S3 default encryption and Block Public Access,
   converging an existing bucket to these three as well; `head-bucket` treats only 404/NoSuchBucket as "absent", and AccessDenied errors outright.
2. The control-plane pod identity role is granted `s3:PutObject` on that prefix (the archiver only writes; it neither reads nor deletes objects).
3. `GPU_FAULT_CONTROL_RECORD_RETENTION_DAYS / ARCHIVE_S3_URI / ARCHIVE_INTERVAL_SECONDS` are rendered only into
   the control-worker (only it runs the archiver); `GPU_FAULT_CONTROL_RECORD_ARCHIVE_BATCH_SIZE` defaults to 200.
4. The deploy preflight `control_record_archive_bucket` checks with the read-only `s3api get-bucket-location` that the bucket exists and is in the site
   Region, refusing the release otherwise (the bootstrap task runs before the preflight, and normally has already created it).

When the bucket is missing, the archive put failure only counts in `gpu_fault_control_record_archive_errors_total` and retries in the next round;
it **does not** delete any live record (archive first, verify the SHA, then delete).

**Disabling**: explicitly declare `spec.retention.controlRecordRetentionDays: 0` in `site.yaml` and deploy; the archiver and
incident/workflow deletion stop entirely; the bucket and objects are kept. `GPU_FAULT_MARKER/NOTIFICATION/COMPLETION_RECORD/REGISTRY_MEMBER_RETENTION_SECONDS`
are independent archive-free sweeps (defaults 30 days / 30 days / 30 days / 1 day), each disabled with `<=0`, obeying the rule "kept as long as the incident
still exists".
