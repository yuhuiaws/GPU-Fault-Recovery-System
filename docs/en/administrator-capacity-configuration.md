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
- Processor total queue, single-cluster queue, client backoff, persistent retry, completed record retention and the
  ingress's routine admission batch delay (`admissionBatchDelaySeconds`, 0..0.5 s, default 0.002; the default is
  omitted from records and YAML so live digests do not move, see §4.6);
- The site-wide range of concurrent ingress admission flushes: the ceilings `admissionFlushConcurrency` /
  `faultAdmissionFlushConcurrency` / `evidenceAdmissionFlushConcurrency` (1..1024 each, defaults 96/48/24) and the
  floors `admissionFlushConcurrencyFloor` / `faultAdmissionFlushConcurrencyFloor` /
  `evidenceAdmissionFlushConcurrencyFloor` (1..1024 each, defaults 96/48/12, each at most its ceiling), shared by the
  3 api-ha replicas x 4 processes so each process takes ceil(value/12) of both and moves its in-flight flush count
  between them on flush latency (the routine and fault floors equal their ceilings by default - the proven fixed caps,
  so the controller is a constant there); and the adaptive cap of the batch delay (`admissionBatchDelayMaxSeconds`,
  0..0.5 s, default 0.05); the defaults of all seven are likewise omitted from records and YAML, see §4.6;
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
   to return to `available` on the new window (three consecutive observations); when it did not change, checks
   read-only that the live window equals the pre-change value (`verify_aurora_capacity`), so an apply is never
   recorded `APPLIED` over a window somebody else moved.
5. Rolls only the affected CPU roles and completes verify/stability/commit. A schedulability preflight runs first:
   per live node, allocatable CPU/memory minus the requests of every scheduled Pod, modelling the roll one role at a
   time (ingress's surge Pod needs one free 4-core/4Gi slot, the consumer tier's added replicas take headroom for
   good); a step that cannot fit fails closed before anything in the cluster changes, naming the missing cores and
   every node's headroom, with the evidence written to `<state-dir>/control-plane-apply/` (Deployment and Operations
   Manual §2.1). The ingress's surge-first roll is also replayed old Pod by old Pod: with exactly one free slot (a
   change disabling the spool rolls api-ha first, while the spool-workers still occupy CPU) the apply writes the
   deletion order onto the old Pods as `controller.kubernetes.io/pod-deletion-cost` annotations before the new
   template (the old Pod sharing the surge node last -- otherwise the controller always deletes it first and the next
   new Pod has nowhere to go), recomputed from where the new Pods actually land during the rollout wait; only when
   even the steered order cannot progress is the change refused, naming the node short of cores and by how much
   ("free one 4-core / 4Gi slot on a second node or add a CPU node"). The roles roll one at a time, each waiting for
   its own rollout: ingress -> control-worker ->
   spool-worker.
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

**Aurora drift** (the live window is neither the pre-change value nor the target, e.g. moved by a tool without a
reviewed record) is refused before any modify with `AuroraCapacityDriftError`: nothing was changed by this apply, so
no Aurora compensation is made, `desired.json` is restored to the pre-change configuration and `result.json` records
`FAILED` with the three windows. Two remedies: `gpu-fault-admin deploy --state-dir <dir>` converges Aurora per
`desired.json` (deploy has no reviewed record and converges live to desired), or write the live window explicitly into
admin-config.yaml to bring it under review (desired equal to live is accepted). The performance runner therefore only
verifies Aurora and never modifies it (performance acceptance plan §8.4).

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

Both `desired.json` and `pending.json` carry digest self-checks; changed content fails closed. The schedulability
preflight's evidence (headroom per node, each role's surge/added Pods, the order actually rolled, and the ingress
step's `steering`: `nodes_with_a_free_slot`, the predicted `surge_nodes`, the `deletion_order` with its `cost`s,
`progress_guaranteed`, the unsteered `without_steering.stalling_order`) is
`<state-dir>/control-plane-apply/schedulability-<release>-<time>-<pid>-<n>.json`, one per preflight (`<n>` numbers
them within one apply): one per CPU role apply, plus one when a rollback restarts the tiers that could not prove the
pins on `/healthz` (the fallback restart runs the preflight again).

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

Node sizing: `32-enabled`/`50-enabled` settle at 3 api-ha x 4 cores + 5 control-worker x 3 cores + 3 spool-worker x
1 core = 30 cores (plus about 3 cores of system Pods, the HyperPod operator and kueue), which 3 x `ml.c5.4xlarge`
(about 15.9 allocatable cores per node) carry; the tightest step of the roll is ingress's surge Pod, which needs some
node with `free >= 4 cores and >= 4Gi` (free = allocatable - the requests of every Pod on that node). `config` runs
the schedulability preflight by the same formula before the apply (§4.2 step 5): exactly one free slot rolls (the old
Pods' deletion order is steered); no free slot at all, or a new Pod with nowhere to go even steered, is refused naming
the node and the missing cores -- add CPU nodes first, then apply.

Admission batch delay: the presets leave `processor.admissionBatchDelaySeconds` at the release default 0.002 (2 ms).
The tuned group of Performance Acceptance Plan §13.3 improved completion p99 by about 14% at 0.05 with unchanged
throughput, but PostgreSQL `xact_rollback` rose from 4 to 59 and §15 item 6 requires repeated rounds before it becomes
the default, so the default is not changed silently. With the ingress's routine admission batch cap of 64, a delay of
20-50 ms at about 1,000 req/s raises the average flush from 9.7 items (25k items took 2,553 transactions) to about
29-48, cutting admission transactions 3-5x at the cost of at most the delay in added acceptance latency; changing the
field rolls the three roles (the processor section is part of every role digest) through the same audit and rollback
path as any other field.

Admission flush concurrency range: `processor.admissionFlushConcurrency` (routine telemetry, default 96),
`faultAdmissionFlushConcurrency` (the P0 fault tier, default 48) and `evidenceAdmissionFlushConcurrency` (the evidence
tier, default 24) are **site-wide ceilings** -- the most admission INSERT transactions all api-ha processes together may
run against `gpu_fault_processor_queue` at once; `admissionFlushConcurrencyFloor` / `faultAdmissionFlushConcurrencyFloor`
/ `evidenceAdmissionFlushConcurrencyFloor` (defaults 96/48/12, validated to be at most their ceilings) are the **floors**
every process starts at and runs at under moderate load. The renderer ships both ends together with the divisor
(`GPU_FAULT_PROCESSOR_{,FAULT_,EVIDENCE_}ADMISSION_FLUSH_CONCURRENCY{,_FLOOR}` and the ingress's
`GPU_FAULT_INGRESS_WRITER_PROCESSES` = 3 replicas x 4 processes = 12); each process takes `ceil(value / 12)` of each
(floors 8/4/1 and ceilings 8/4/2 by default; a value that does not divide takes effect as the rounded-up fleet total),
and no per-process `*_ADMISSION_BATCH_GROUPS` literal is rendered any more (an explicit GROUPS value still pins both
ends per process). A floor equal to its ceiling makes the controller a constant: by default the routine and fault tiers
are exactly the fixed caps that passed live (8 and 4 per process), and only the evidence tier (1..2 per process) and a
range an administrator opens deliberately adapt. A controller inside each process (`app/admission_concurrency.py`)
decides on every dispatch pass: while the batcher is saturated (every slot busy, work still queued) and the
flush-duration EWMA is below the open threshold
(`GPU_FAULT_PROCESSOR_{,FAULT_,EVIDENCE_}ADMISSION_FLUSH_CONCURRENCY_OPEN_SECONDS`, default 0.15 s) it opens one more
slot; once the EWMA passes the close threshold (`..._CLOSE_SECONDS`, default 0.30 s) it halves the headroom above the
floor (additive increase, multiplicative decrease); between the thresholds it holds; and when the limit stops binding
(slots idle) it gives back one slot, so a burst that has passed leaves the process at the floor and an idle site never
holds more concurrency than the floor. Every change -- a raise as much as a close or a decay -- waits until a generation
of flushes (`max(3, current limit)`) has completed at the current level, so the EWMA describes the level being judged,
one slow sample cannot cascade a wide limit down to the floor, and the limit cannot be raised and lowered on adjacent
passes. The arrival-time deadline projection counts the slot wait in rounds of the concurrency the controller is
**about to open** (the ceiling while the EWMA is below the open threshold, the live limit otherwise), so capacity that
is opening is not counted as missing. `/metrics` exports
`gpu_fault_processor_{,fault_,evidence_}admission_batch_in_flight_limit` (the live value) and
`..._in_flight_limit_raised_total` / `..._lowered_total`.

Where the numbers come from (this site's integrated-32 perf suite on schema v22 -- the same 32-cluster burst and the
same 12 ingress processes throughout; undated): the caps used to be per-process literals (16 + 8 + 4), up to 336
concurrent INSERT transactions from 12 processes on one table, and the first 30-65 s of the load phase became a convoy
of 134-182 concurrent queue INSERTs with 85% of active statements waiting on `LWLock:BufferContent`, inside which a P0
request's own INSERT reached its 30 s deadline (503) -- the amplifier was the v21 wakeup probe's cached plan, fixed by
schema v22. Three calibration points followed. A **fixed 24** (2 per process) traded the convoy for shed telemetry:
every process sat at its cap for the whole load phase, 762-801 entries queued per process, 4,601 requests expired in
the queue (18.7% reserved 503s for routine telemetry against a 2% budget), 517 req/s, 0.59 s per 16-item flush --
queueing, not the store. A **fixed 96/48** (8/4 per process) passed: 1,051 req/s, zero reserved 503s, zero P0 503s,
81 ms per flush, 60-69 entries queued per process at the peak, a peak of 113 active statements for about 5 s and 1%
`BufferContent` waits. An **adaptive 48..192** (4..16 per process, fault 24..96, open/close thresholds 1.0/2.0 s)
failed: the controller let the site reach ~250 concurrent flushes and the page-lock convoy on the queue table returned
(a 69 s storm, 142 active statements at the peak, `BufferContent` 36%, `XactSync` 23%); a flush averaged 1.08 s, which
is below the 2.0 s close threshold, so the slots stayed open while telemetry was shed on the deadline projection
(543 req/s, 35.8% reserved 503s, P0 503s too, fault p99 28.4 s) and the limit flapped (81 raises and 51 lowers in 14
minutes); starting at 4 per process also shed during the first second of the burst. Conclusion: on this schema the
store tolerates about 100 concurrent admission flushes and convoys above that -- page contention on
`gpu_fault_processor_queue` is the physical source of the ceiling -- and the 1,400+ req/s this site once reached at
192 slots predates the v20 per-writer ledger (the single counter row serialised the writers, which happened to throttle
page contention) and is not reachable by concurrency alone on this schema. Hence the default ceilings are the fixed
caps that passed, 96/48/24; the routine and fault floors equal their ceilings (the first second of a burst must be
carried by the floor without shedding) and evidence keeps 12..24; the thresholds are ~2x and ~4x the uncontended 81 ms
flush (0.15/0.30 s), so a 1.08 s average flush closes hard instead of holding. The fault tier has its own range, and
its own batcher and Store I/O threads inside each process, so a telemetry burst never takes a fault slot; the depth
reserve (`GPU_FAULT_PROCESSOR_FAULT_RESERVED_*`) keeps its semantics. The recommended site values are the defaults; the
acceptance target for the default configuration is this site's calibrated 1,000+ req/s with zero reserved 503s and
zero P0 503s, `..._in_flight_limit` constant at 8/4 on the routine/fault tiers and `..._raised_total` /
`..._lowered_total` at 0. The adaptive cap of the batch delay, `admissionBatchDelayMaxSeconds` (default 0.05, 0..0.5): once
the flush-duration EWMA passes 100 ms the batcher raises the coalescing window linearly from `admissionBatchDelaySeconds`
toward this cap (fully at an EWMA of 200 ms) and lets it fall back as the store recovers; a cap at or below the base
delay switches the adaptation off (the runtime takes the larger of the two, so raising the base alone stays valid). The window is not a lever on items per flush: a flush is one transaction
for one cluster scope, so what it carries is the number of that cluster's requests queued inside the process at
dispatch, not what arrived during the window. In a closed-loop simulation of one process (32 clusters, 8 slots, an
81 ms store round trip) 16 / 96 / 400 requests in flight gave 1.0 / 3.0 / 12.5 items per flush whatever the window did,
while a 20 ms base window cost 17% of the throughput and a 50 ms window held under saturation cost 35% -- the wait
before a dispatch pass idles the slot a completed flush just freed. The live 2.5 items per flush were ~65 entries
pending per process spread over 32 clusters; 8-16 items per flush would need more of the burst inside each process at
once (requests in flight), not a longer window. The base window therefore stays at 2 ms and the adaptive stretch only
happens once the flush EWMA passes 100 ms (the store is the bottleneck and statements themselves are the cost), ahead
of the controller's hold band (0.15 s) and close threshold (0.30 s). The
convoy's amplifier -- the v21 wakeup trigger's `NOT EXISTS` anti-join, whose plan a backend cached while the queue was
empty as a hash over the whole backlog (a 10-row INSERT read 49k pages in 250-320 ms until autoanalyze replaced the
plan) -- is fixed by schema v22 (Detailed Design §3.1.3). The defaults of all seven fields are omitted from records and
YAML (live digests do not move); a change rolls the three roles as well.

Per-process thread counts are not the lever for liveness under a burst. In the local reproduction of the same
50-cluster cold-connect burst (`scripts/perf/ingress_event_loop_burst.py`), 32 vs 8 decode threads and 16 vs 8 flush
slots gave identical probe and event-loop latencies (longest stall 0.25-0.6 s, always one gen-2 GC); the mechanism of
the kill is the service listening socket's accept queue (see the liveness section of the deployment manual). This
configuration therefore introduces no "per-process CPU-thread budget" field and does not let the concurrency controller
close slots on event-loop lag -- closing slots only moves the queue from inside the process to the kernel's accept queue
(at 2 slots the probe median was 94 ms but 1,074 requests got 503), trading 503s for no shorter stall. The three related
runtime variables are set by the renderer and are not AdminConfig fields: `GPU_FAULT_LIVENESS_PORT` (service port + 10
per role, the dedicated liveness listener), `GPU_FAULT_GC_FREEZE_AFTER_STARTUP=true` (`gc.freeze()` at the end of
lifespan start-up: 9.0 s -> 6.6 s of GC time per process under the same burst, +7% throughput, longest pause unchanged)
and `GPU_FAULT_GC_GEN2_THRESHOLD` (CPython's 10 by default; raising it to 100 cut gen-2 collections from 45 to 4-5 and
total GC time by another 30% but stretched each pause to 0.45-0.59 s -- a worse trade for an event loop, so set it only
for an A/B).

The service port's accept queue is split per worker as well: the three roles launch through `python -m gpu_fault.app.serve`,
each worker binds the service port itself with `SO_REUSEPORT`, and the completed connections a Pod can queue grow from
`somaxconn` (4096) to `--workers x 4096`. This is not a throughput lever (in the same reproduction, at 4,250 concurrent
cold connections, throughput and median latency are within noise and the tail is +3-11%) but a capacity lever: past 4096
connections the shared socket drops SYNs (5,500 concurrent: 961 req/s, 73 503s, client p99 27.9 s) while the per-worker
sockets do not (1,117 req/s, 0 non-2xx, p99 5.35 s); the 50-cluster preset's ~4,266 cold connections per Pod sit exactly
across that line. With it, `--timeout-keep-alive 5 -> 100` (renderer constant `KEEP_ALIVE_TIMEOUT_SECONDS`, above the
node-side pool's 90 s idle ceiling): the node Agent heartbeat (30 s) and the GPU inventory (60 s) each keep one long-lived
connection per process, the 300 s health summaries still reconnect. The capacity arithmetic must remember that uvicorn's
`--limit-concurrency 4096` counts idle keep-alive connections: the design scale, 12,800 nodes x 2 channels = 25,600
long-lived connections / 12 ingress processes ~ 2,133 per process, leaves ~1,960 for bursts; making the 300 s channels
long-lived too would add 4 per node (~4,267 per process) and exceed the limit, which is why the client's idle ceiling
stays at 90 s by default. The node-side jitter (`GPU_FAULT_COLLECTOR_CADENCE_JITTER_RATIO` 0.1,
`GPU_FAULT_COLLECTOR_STARTUP_JITTER_SECONDS` 20) spreads the first posts after a fleet-wide restart from a 15 s window to
35 s and requires every silence threshold >= cadence x 1.1 + 35 + 47 s (all defaults satisfy it; held by
`tests/collectors/test_cadence_jitter.py`).

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

### 4.8 Hot-Table Physical Layout and Background Compaction

The capacity presets assume compact hot-table indexes and small tables in the Store. One mass delete (the teardown of a
formal 32-cluster perf round, a cluster leaving the fleet) leaves the 9 indexes of `gpu_fault_processor_queue`, the 2
indexes of `gpu_fault_gpu_metric_latest` and the three small tables lanes/queue counts/priority shards at their
high-water mark: on one release with identical admission defaults, ingress throughput fell from 1,572 req/s on a fresh
store to 538 req/s (49 % of the active statements waiting on `LWLock:BufferContent`) and returned to 1,141 req/s once
only the indexes were rebuilt and the three small tables `VACUUM FULL`ed. The control worker's periodic job
`store-compaction` (`GPU_FAULT_STORE_COMPACTION_INTERVAL_SECONDS`, default 300 s, `0` disables) does exactly that from
catalog facts and exports `gpu_fault_store_hot_table_heap_pages_per_live_row{table}`,
`gpu_fault_store_hot_table_index_pages_per_live_row{table}` and
`gpu_fault_store_hot_table_index_min_entries_per_page{table}`, `gpu_fault_store_hot_table_index_max_pages{table}` and
`gpu_fault_store_hot_table_due_actions{table}`; the values depend on no planner statistics target or cost
model, and the perf runners write them into every round's `summary.json` as `store_bloat`. The verdict is made on each
index's own density among indexes of at least 64 pages, never on index pages per live row -- a queue left with two live
rows reads 74.5 index pages per live row for 149 pages while no index reaches the floor, and `due_actions` of 0 is the
conclusion (Administrator Operations §10.3). The thresholds (an index of
at least 64 pages below 8 live entries per page; a small table of at least 16 pages below 4 live rows per page) are not
capacity parameters and do not vary with the preset; the large tables' heaps and TOAST are reclaimed only by
`gpu-fault-admin store-maintenance` in a maintenance window (Administrator Operations §10.3) and cost roughly a quarter
of throughput. The variable is a ConfigMap environment variable, not a capacity-preset field of `admin-config.yaml`.

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
    admissionBatchDelaySeconds: 0.002
    admissionBatchDelayMaxSeconds: 0.05
    admissionFlushConcurrency: 96
    admissionFlushConcurrencyFloor: 96
    faultAdmissionFlushConcurrency: 48
    faultAdmissionFlushConcurrencyFloor: 48
    evidenceAdmissionFlushConcurrency: 24
    evidenceAdmissionFlushConcurrencyFloor: 12
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
- The roles roll one at a time, each waiting for its own rollout, in the order ingress -> control-worker ->
  spool-worker (ingress's surge Pod needs a free slot on some node, which the consumer tier's added replicas would
  take; the preflight decides the deletion order of ingress's old Pods and writes it as `pod-deletion-cost`, and an
  expired rollout wait prints the stall diagnosis first); enabling spool follows the same order: the spool is a durable table, requests ingress spools before the
  spool-worker is ready wait there, and requests beyond the depth cap are returned with 503 Retry-After for the Agent
  to retry; only when that order cannot fit and a role's replica reduction would free the room does the preflight
  move the shrinking role to the front, recording the order actually rolled in the evidence; on a rerun (§4.3) a role
  whose render equals the live template rolls no more, and the preflight asks no slot for it;
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
