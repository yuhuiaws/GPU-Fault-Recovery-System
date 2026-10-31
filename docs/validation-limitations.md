# Validation limitations

## Host observation and targeted validation

The [host hardening batch](components/host-observation-hardening.md) separates
unreadable EFA/SMART/BMC data from hardware faults and binds new host diagnostic
plans to their triggering metric/device. Local proc/sys fixtures and a loopback
API verify software behavior only. Physical PCI loss, real driver behavior,
storage probes and production recovery remain separate acceptance work.
The broader thermal/RMA and application-fault policy recommendations are not
claimed as completed by this batch.

## Xid linkage and Drain P2P recovery

Xid 154 linkage versions are reporting metadata, not applicability gates for
the original Xid. NVLink5 still needs its version-specific decode evidence.
Drain P2P now uses `RECOVER_DRAIN_P2P` after cordon, workload stop, service
quiesce and GPU-client verification. It requires an explicit disabled
persistence reading and an identifiable recovery-action report for every target.
Unknown or unsupported reports fail closed; an explicit reboot request or
a known failed reset authorizes the next reboot rung. A failed post-recovery
validation follows the existing reset-validation escalation path. The operation
reuses the single-reset quiesce claim and ledger, and keeps Fabric Manager running.

These changes require matched control-plane, executor and Node Runtime releases
and the updated operation allowlists. Old agents reject the new operation.
Drain pending new-operation workflows before rollback to an older release.
Local tests use synthetic driver XML, fake system commands and real local
ledgers/quiesce files. They do not establish physical UVM persistence release,
driver XML compatibility on every supported GPU, or live reset/reboot recovery.
Independent UVM persistence detection/release is not implemented, rather than
merely untested. The [software E2E specification](components/xid154-recovery-e2e.md)
registers this as a separate BLOCKED requirement. Its automated chain uses real
API/correlation/executor/agent logic with local SQLite and scripted external I/O.
Terminal reset-timeout compensation can restore services under existing policy;
scheduling and training remain held and the workflow requires confirmation.

Deployment regression tests use fake AWS/Kubernetes runners, plus real local
Python virtual environments for node-slot integrity checks. They cover unhealthy
status diagnostics, fail-closed CronJob reads, full Aurora refresher snapshots and
restore-before-refresh ordering. They do not measure live provisioning speed or
prove an Aurora password rotation/rollback against a production cluster.

## Acceptance Alignment Boundaries

The current alignment fixes are verified by local behavior tests before a fresh
integrated gate. Their focused counts overlap and are not a new full-suite
result. Historical PASS bindings for the changed AUTH-016 and BOOT-020 contracts
are preserved separately; their current claims require new evidence.

Shared CPU/runner recovery guards reject legacy, nested and cancelled unknown
physical outcomes. Phase-bound successful readmission cannot clear a sibling's
hold. A verified spare recovery may recover its incident while the original node
remains isolated; an explicit persistent hold still wins. DESTR-014 keeps its
operator hold even if later SQL reads are terminal. Local tests use fake physical
adapters and do not establish that a real uncertain reset/reboot has completed.

FM producer receipts distinguish source handoff from Store deduplication and
sink retry, within the bounded observed journal window. Omitted, incomplete or
exhausted receipts refuse proof. Observed truncation advances cursor generation;
a polling reader cannot recover an undetectable truncate/rewrite between reads.
DCGM thermal tests exercise persistent warning-to-critical delivery with actual
collector/ingestion logic, not physical overheating.

The GF-ENV-LIVE acceptance runner (`scripts/e2e/regional/run_env_live.py`) drives
the audited loopback replay proxy and repoints only the single metrics collector
through a bounded, deadman-restored maintenance change (a `/run` drop-in handing
the unit a run-scoped `EnvironmentFile=`, the unit restarted, the endpoint read
back from the new process's `/proc/<pid>/environ` and that process's first
pass-through read of the proxy awaited before anything is armed -- a repoint
that does not take effect is rolled back on the node and fails the fixture with
its reason); it proves the
normally-installed collector, real control plane and scheduling run end to end,
never that a sensor truly reported the fault or that hardware was repaired. Its
unit tests use fakes for the device, Kubernetes and systemd I/O. Every
GF-ENV-LIVE case stays manual/NOT_RUN until run live; a passing unit test is
supporting logic coverage, not a LIVE PASS. The runner decides per device which
severe-temperature variant can run on the real same-scale limits and leaves the
others NOT_RUN rather than fabricating a scale: on a T.Limit-only p5en H200 the
absolute-core and absolute-shutdown (GF-ENV-LIVE-011) variants are NOT_RUN,
GF-ENV-LIVE-002 reuses GF-REGIONAL-COLLECT-002's controlled power path.
GF-ENV-LIVE-010 runs only with a control node (`--node-b`), `--site-file` and a
multi-GPU target node: an independent control task (the node-pinned single-node
training fixture, GPU-active at the acceptance GPU duty) runs on the control node
while one GPU of the target node is replayed, and the scope judge grades the
control GPUs, the control
node and the task as untouched and the fault as one merged workflow / one sent
notification; without those it is NOT_RUN. The target node carries no owned
distributed job, so the STOP scope of an affected job is reported as not measured.
The runner carries no HyperPod-specific step: on a host-mode node without an
exporter Pod (plain EKS) the device limits are read by a transient host probe
that is deleted again, otherwise by an exec into the node's exporter Pod.
After arming, the runner polls the proxy status and the product's records within
a deadline derived from the node's collector interval, the confirmation, cooldown
and VALIDATE_GPU windows and a scan margin, capped by the proxy's replay window
and the collector deadman. A replay that was not consumed or whose ingestion no
accepted record shows is INCONCLUSIVE, never PASS; a reaction the product needs
longer than that window for is INCONCLUSIVE as well. Non-brake bits and unusable
fields are no edge-filter delivery edge, so they reach the control plane only when
a health summary falls inside their short plans. GF-ENV-LIVE-009 pauses the metrics
collector read-synchronously on the node (a waiter unit SIGSTOPs it the moment the
second severe read is consumed, a deadman resumes it), resumes it past the 60 s
confirmation gap and restarts it once at a normal temperature; the judge requires
the post-gap sample to count 1/N again, no reading inside the gap, no finding
cleared by silence, one producer and a fresh initial-baseline batch. A waiter that
does not run, times out or pauses late makes the case INCONCLUSIVE. The DCGM to
nvidia-smi source switch (installation only) and the cross-boot path are not
driven and are recorded as gaps. The preflight admits only a node without
business workloads, so task progress and suspension are reported as not measured.

The product records a confirmation-count change only in the current finding
state (`gpu_finding_state`, overwritten by the field's next accepted reading);
the finding history is written on a severity or automatic-action change alone,
and no incident reason, marker or notification follows a count change. The
post-gap regrade "2/3 -> 1/3" of GF-ENV-LIVE-009 is therefore a state that lives
one collector interval (15 s) until the clearing read replaces it, while one
poll of the runner (its sleep plus five `kubectl exec` reads) costs 46-56 s on a
live site: the first live run ended INCONCLUSIVE "no finding graded it yet"
although the product had regraded correctly. The runner now schedules a second
read-synchronous pause (the recount hold, `pause-collector --label hold`, its
own waiter, deadman and record) while the gap pause still holds the collector,
one read past the gap pause's target: the collector stops again as soon as it
has posted the first post-gap read, the regraded state stays readable, and the
judge resumes the hold the moment it has read a finding on that sample (bounded
by two polling rounds, 120 s; the proxy window and the collector deadman grow by
the same amount for 009). Every pause now lands a 2 s settle after its target
read so the read's batch has reached the control plane. The proof is still the
product's own current state, read while current; a hold the node refuses or
that never lands is recorded, the resume goes ahead and the records decide. A
durable product record of a count change (a history row, an incident reason) is
a suggestion, not implemented here.

The replay tool matches exporter samples the way the collector's parser does:
dcgm-exporter declares the thermal and power violation durations `counter`
(`deploy/dataplane/dcgm-counters.csv`), the Prometheus text parser renames those
samples `<name>_total`, and the tool now falls back to the bare name
(`canonical_mapping`). Before that fallback (live 2026-10-26, GF-ENV-LIVE-006
`hw_thermal` on plain EKS) every armed read was answered 503 "requested fields
are absent", the collector logged `cannot scrape DCGM exporter` each tick, the
proxy's read count froze at the arm, and the case ended "consumed 0 of 7" while
the repoint proof itself had held; the two duration plans and the 024
`thermal_warning`/`power_performance` plans share the fix. The proxy `/status`
now reports `failed_reads`/`last_error`. The ENV-LIVE windows are sized from the
node's collector cadence like the repoint's first-read budget (reads x interval,
the cadence jitter over them, the start-up spread and start-up jitter a
collector restarted inside the window needs, the delivery margin), the probe's
preflight names leftovers of an earlier run (an open replay window, a loaded
repoint drop-in, a `gpu-fault-env-*` transient unit, a collector reading a
foreign endpoint) and the runner deploys nothing while any is present, the
repoint proof is required before the arm and re-checked against the status read
right before it, and an armed proxy that counts no read by the time the first
one is due has the unit read back once and the reason named (endpoint
reverted, drop-in gone, unit restarted or inactive, proxy refused the reads,
collector silent). All of these are INCONCLUSIVE with the reason, never a
product verdict; the proof scope stays `live-telemetry-replay`.

GF-ENV-LIVE-010's control task runs the acceptance GPU duty
(`acceptance_gpu_load.DEFAULT_GPU_DUTY_PERCENT`, 30 %), written explicitly into
the rendered manifest (`env_live_scope.CONTROL_TASK_GPU_DUTY_PERCENT`) and into
the `control_task_before` receipt. Its first valid plain-EKS run (2026-10-26,
`core_tlimit/severe`, release R13g) FAILed on the control stage alone, with every
other stage clean: the task was rendered at duty 0 ("idle"), and the product's
`LOW_GPU_UTILIZATION` / `LOW_CPU_UTILIZATION` rules -- the two
`active_workload_only` rules, which time a window while exactly one managed
attempt with a live container holds the node's GPUs and open an incident once
utilization stays at or below 5 % for 300 s -- graded the harness's own control
task about nine minutes after it started (two `low_gpu_utilization` /
`low_cpu_utilization` incidents, one workflow and three notifications on the
control node), which the scope judge counted as activity the control node
"received". The product behaved as designed (the same rule is what the
acceptance training workloads were made GPU-active for); the oracle's fixture
contradicted the product's rules. In the same live rounds no other case produced
a low-utilization incident. The duty cannot coincide with the replayed plan: the
scope checks read the target node's batches and findings only (its other GPUs
run nothing), the control node is read for records only, and the 010 plans serve
on the one target GPU a temperature one degree past the device's own slowdown or
maximum-operating limit -- a control GPU reporting such a value would be a real
thermal fault, not a product of a bounded load sized for a fraction of the power
limit. A control node that still receives a low-utilization incident FAILs the
control stage as before; the error text now adds the diagnosis (the product
graded the control task's own utilization: check its GPU duty) and the evidence
lists those incidents (`control_utilization_incidents`). The unit tests pin the
rendered duty and the diagnosis; the case is to be re-run live on both clusters
(harness only, no product release).

AUTH-016 follows the real administrator rotation contract using controlled I/O.
Same-release Node key activation tests bind durable progress, projected key
ordering and actual local Agent/FleetRegistry behavior. They are not deployed
Kubelet timing or independent LIVE activation evidence. Platform clocks, kernel
and trusted Kubernetes identity reports remain assumptions.
The writer checks its fixed activation deadline and owned wave before each
Secret submission. That blocks an already-expired or already-drifted submission,
not an arbitrary external change after the last read or a server completion
after the client's deadline. Unknown acknowledgements retain incomplete evidence.

HA/capacity probes distinguish role-local metrics, routine-spool coalescing,
foreground/replay admission and simulated execution from physical recovery.
CAP-004's separate queued-lease phase exercises a two-claim/one-executor boundary;
it is neither a production throughput result nor a repeated hardware-action
experiment. SQLite concurrency regressions concern a compatibility backend only;
the production business Store remains PostgreSQL.

NOTIFY-007 records invalid metric-role evidence as a failed preflight plan;
execution still requires a valid plan and strict role/family evidence.
HA-007's local signal handoff polls the event flag under the same deadline, and its
handler sets the flag only once, so a second SIGTERM that lands inside the first
handler's `Event.set()` cannot deadlock the child.
A deterministic condition-timeout test demonstrates that `Event.wait()` may
return false even when a signal callback has set the flag. That does not prove
the exact scheduling history of the failed full run or a live Kubelet shutdown;
signal delivery, real shutdown budgets, coordinator failures and child cleanup
remain separate checks.

`VALIDATE_GPU` has a bounded WAITING grace for a single all-WARNING class:
thermal stress, correctable-memory degradation, or power-limit throttling.
The power class includes `power_violation_total_us` and
`composite:POWER_LIMIT_THROTTLING`. Grace is not successful validation or proof
that hardware will recover. Mixed classes, critical findings and findings that
persist beyond the existing window receive no new exemption; this change does
not grant a grace window to `VALIDATE_FABRIC`.

For an incident's original targets, validated restore can use complete,
unambiguous per-node inventory to route GPU/FABRIC validation to the correct node.
It never removes an incident GPU
because that UUID is absent from the inventory: absence can itself be a fault.
Every required GPU still needs fresh telemetry, and all validation precedes
restoring scheduling. Old records with unproven cross-node GPU scope remain
fail-closed and need authoritative reconciliation, not automatic scope shrinkage.

An escalation successor places each source GPU before it builds its steps: a
single-node step names only GPUs placed on its node (the node's own inventory
snapshot or latest per-GPU telemetry, the attempt's container observations, then
the workflow's own per-node lists). A GPU nothing places is kept when every node
of the source scope failed; in a wider scope it is named on no successor step and
each failed node validates its full known inventory instead. That drop is a
deliberate liveness choice: such a GPU was never reported by any scope node, so
requiring it could only time out on a healthy node. The validated restore removes
an incident GPU only on positive evidence that it sits elsewhere: the inventory of
a release-only node of the incident or, for an escalation successor, of a node on
its escalation lineage lists it and no restored node's inventory does. That
lineage reading is the authoritative reconciliation above; absence alone still
removes nothing. Residuals: with every store read failing, the workflow's own
per-node lists decide alone, so a step narrowed to one node that kept the union of
GPUs still reads as that node's.

A merge that widens a multi-node node-action step without `gpu_uuids_by_node`
places that step's GPUs by the workflow's own per-node lists only
(`RecoveryArbiter.merged_gpu_scope`, `arbitration.step_gpu_shares`; no store
read), so a single-node step that kept the union of GPUs can still mislead it; a
GPU those lists put on two of the step's nodes counts as unplaced, and one they
put on the wrong node reaches an Agent that refuses it, as before. When the
lists cannot say which node holds each GPU, the step stays without a mapping
through every later merge: a multi-node reset is then refused before any send,
while the other node actions still hand each node the step's whole list, as they
do for an unmerged step, and rely on the Agent refusing a GPU it does not have.
None of such a step's GPUs reaches other steps through the merged scope, and a
later fault on one of its nodes whose GPU its flat list names counts as covered
(`fault_scope_covered`) and is absorbed into a step that will be refused, so the
ladder, not that step, repairs it. Mappings an earlier release's widen already
wrote are explicit and kept. No current producer builds such a RESET step; local
tests construct one.

The restore's `--plan` preview and the in-Pod write call the builder with the
same restored and release-only nodes. The Pod re-reads the inventories when it
writes, so an inventory that changes between the inspection and the write can
still build other steps: the result names them (`the created workflow's steps
differ from the plan`) and does not refuse. A deployed image whose builder
predates release-only restores refuses those restores, writing nothing, until the
release is deployed. Local tests use the in-memory store and fake inventories;
they do not measure a live escalation or a live restore.

Collector acceptance waits account for the leased metrics census and preceding
attempt-missing grace. They are bounded by the existing operation and maintenance
deadlines, reject incomplete evidence, and never extend the approved window.
COLLECT-021 drains predecessors before killing its workload, then rechecks before
the late-XID submission; these sequential observations are not an atomic fence
against independent external writers. Local tests do not measure deployed watcher
or metrics refresh timing.

Workflow/remote-command table migration is tested with isolated PostgreSQL 16,
not Aurora. Coverage includes mixed migration modes, batched backfill, canonical
JSON validation, old-writer cutover fences, lease/CAS behavior, notifications,
archive-first deletion, index plans and read-only schema drift rejection.
Synthetic large-payload renewal tests observe HOT updates without TOAST growth;
this is not a production throughput, WAL-volume or Aurora-cost benchmark.
Production activation still requires separate remote-command and workflow
maintenance windows, drained execution, validation and stable release periods.
Ordinary deployment does not activate or purge either migration.

The following cases are not claimed as fully validated by the current
staging environment as of 2026-08-27:

- The Kubernetes shutdown budget is checked arithmetically, but the revised
  `GF-REGIONAL-HA-007` still needs a staging run that holds legal 30/70/110
  second requests while a control-worker terminates. The current worker uses a
  130-second lifespan budget and a 240-second Kubernetes termination grace;
  exactly-once completion or explicit lease release must be proven live.
- The fatal processor deadline callback has not been run in a subprocess that
  verifies exit code 70, lease release, and takeover by a replacement worker
  (`GF-REGIONAL-HA-008`).
- Notification outbox behavior is well covered as a service, but the
  service-role/lifespan wiring that starts the production worker dispatcher
  has not been exercised end to end. Restart takeover while the provider is
  throttling is also unverified (`GF-REGIONAL-NOTIFY-006`).
- PostgreSQL-specific test count is intentionally not pinned in this document.
  Any test skipped because `GPU_FAULT_TEST_POSTGRES_URL` is absent means the
  default local suite is not a production Aurora Store gate
  (`GF-REGIONAL-CAP-005`).
- Remote commands have no command-level `max_waiting_rounds` or deadline.
  `GF-REGIONAL-CMD-013` successfully reclaimed the same command after each
  of 20 consecutive WAITING results, carrying the prior result details every
  time. A permanently WAITING adapter can therefore keep a workflow open until
  the separate workflow execution deadline intervenes; command-level
  dead-letter behavior is not implemented.
- `GF-REGIONAL-HA-006` measured the default executor-crash takeover delay:
  after the owning process was SIGKILLed, a completed Node Agent diagnostic
  remained behind the old 120-second command lease and reached terminal state
  125.981 seconds after the kill. This is expected by the current lease model
  but remains a recovery-latency limitation for time-critical actions.
- `HttpEventSink` now has a per-collector bounded disk outbox. Network errors,
  429 and exhausted 5xx retries are replayable; permanent 4xx records remain
  non-replayable. The residual limitation is bounded capacity, unwritable or
  damaged local storage, and a Collector process restarted mid-outage waiting
  for that channel's next live post (delivered or failed) to re-arm replay.
  Once armed -- by a delivered post or by a failed post the outbox took -- a
  single background worker continues bounded batches without waiting for
  another collection cycle; a zero-progress round waits out a bounded backoff
  (10 s doubling to a 60 s cap, jittered) rather than stopping, so the backlog
  drains within about one interval of connectivity returning (GF-REGIONAL-NET-001
  on 2026-10-01 showed the fabric-manager channel lagging by its full 300 s
  cadence before this). A later successful live post wakes it early. Completion Watcher
  critical failure/terminal events now use a ConfigMap-backed write-ahead
  outbox and replay independently after restart; its residual bound is the
  configured ConfigMap record/byte limit.

- `COLLECT-004` debounce was validated through the documented safe substitute:
  expected GPU count was temporarily set to 9, the first mismatch established
  a baseline, and the second sample 15.35 seconds later emitted the threshold.
  A physical GPU removal was intentionally not performed.
- XID154 `IGNORE` and XID74 reset-first were driven through real `/dev/kmsg`.
  The remaining destructive XID154 labels reuse already validated stop/reset/
  reboot execution chains and were not each repeated as separate physical
  actions. XID151 `RESTART_VM` was driven through real `/dev/kmsg` and mapped
  to a successful HyperPod node reboot. XID159 `CHECK_UVM` is applicable only
  to B100/GB200 and is `NOT_APPLICABLE` on this H200 fleet; a test must not
  falsify the product identity.
- Multi-cluster isolation is supported by the regional control plane, but the
  current environment has only one enabled, formally registered physical GPU
  cluster. A disabled temporary `b300-isolation-test` registration is not a
  second production cluster. The user explicitly deferred ISO-006/E2E-002.
- Provider node replacement is intentionally prohibited and is validated by
  configuration, IAM denial, and CloudTrail absence rather than by executing
  `BatchReplaceClusterNodes`.

Historical observations from the 2026-08-11 rerun:

These observations predate the current source review. They are not verification
of this worktree, and cannot close the implementation gaps in the current
[scenario matrix](components/scenario-coverage.md). In particular, historical
AUTH-015 observations do not replace installation-time custody evidence and
deployed command/result-signature checks for a new release.

- BOOT-018 compares source, wheel, node bundle, runtime image component venvs,
  every running control/data plane process, release pins, and every live Agent
  by content digest.
- AUTH-015 now provisions per-node keys on the trusted deployment host. The
  GPU cluster contains no fleet master; installer Jobs and Cluster Executor
  mount only node-scoped keys, cross-node forgeries fail, and one node key was
  rotated without restarting its peer.
- HA-003 recovered from a real RESET=`LEASED` Aurora failover without duplicate
  action; restore, validation, and scheduling restoration succeeded.
- 8e/8f/8g/8h/8i were exercised through real events and cluster state.
- Fabric Manager journal used the production collector/control-plane sink,
  persisted its cursor, and did not replay after collector restart.
- Drill IDs now propagate from markers, XID/SXID raw messages, and node-health
  findings to incidents and notification subjects/bodies.
- DESTR-006 fail-closed preflight was exercised against an existing
  `NodeRecovery=Automatic` HyperPod without provider mutation.

These limitations must remain visible in acceptance reports. They are not
equivalent to PASS results.

The current AUTH-015 runner can additionally bind an explicit signed schema-4
release descriptor to deployed command/result-query signature challenges. Correct
key controls and sibling-key refusals use a non-dispatching operation and an
independently impossible command time window. HTTP proof workers cap bytes and
wall time, scrub credentials from argv/environment and expire independently after
parent loss. Local TLS, process and real Node Agent handler tests are not a LIVE
run. The optional signature-only subproof still does not establish historical
installation custody or activation of a newly rotated key.

The complete AUTH-015 path now requires prospective, independently signed
authorization/start/completion/activation receipts from the real trusted
provisioning path. Explicit administrator enrollment prepares actual identities,
pauses for independent authorization and resumes normal deploy/join; it cannot
promote old checkpoints or current Secret contents into provenance. A subsequent
authorized rotation must be deployed before the same pinned Agent endpoint
accepts the new key and rejects retired/sibling keys. The runner itself neither
rotates Secrets nor restarts nodes. Initial activation alone remains full-case
FAIL. Real local crypto and Node Agent/fake administrator tests verify the code,
not a LIVE installation or activity outside the trusted provisioning path.
See [Node Key Custody Evidence](components/node-key-custody-evidence.md).

## Deployment scheduling regression coverage

The deployment performance changes have hermetic regression tests for checkpoint
invalidation after a slow release build, independent Grafana scheduling, monitoring
rollback ownership, bounded node preflight and runtime identity fan-out, and the
site-wide Installer wave budget. Shell tests use explicit fake Kubernetes clients
to check node input binding, dependency-specific runtime slots, failed Job detection,
and one final CPU Pod template per selected role. Release orchestration tests keep
candidate validation before schema/CPU mutation and DNS publication after CPU
readiness.

The follow-up regressions cover live monitoring drift entering the release plan,
explicit absence snapshots versus denied reads, foreground failure while a build
still writes state, actual RECORD file/bytecode validation, and two local wheel
versions sharing one immutable dependency layer. They also cover a slow node not
blocking the next preflight slot, no new nodes after an observed failure, atomic
legacy environment cleanup with UID/resourceVersion conflicts, and transient Job
API failures under a shared deadline. A successful watch returns its UID-bound
completion object; no extra GET is required at the timeout boundary.
Preflight also rejects invalid budgets and reaps forcibly terminated local workers
even when they cannot emit their normal completion notice.

The access graph tests deliberately block GPU access and Pod Identity until Aurora
starts, proving the CPU-ready-only scheduling boundary without calling AWS.
CommandRunner counts exclude nested subprocesses and SDK-internal API retries.
The separate subprocess API ledger measures admitted child commands and observed
CSM retry events, not every network request. CSM is a UDP lower bound, not a complete
request count or evidence for increasing concurrency. Protocol v4 uses temporary
deploy-host SQLite only for local coordination; the production business Store
remains Aurora.

Local process regressions exercise dual-subreaper ownership, loss of either owner,
refusal after all completion proofs are lost, absolute command deadlines, and
whole-operation cleanup across repeated SIGINT. Cancellation targets supervised
child trees; it does not hard-preempt arbitrary Python work in parent threads or
make uninterruptible kernel waits safe. Site-lock release still waits for cleanup.
A captured command's stderr is proven to hold only the command's own output: the
regression runs a real command past a shortened heartbeat interval and checks
that the heartbeat reached the operator channel instead, and the AWS absence
matcher is exercised with the live stderr shape (heartbeat, blank line, exact
NoSuchEntity error) together with its prose and other-code controls.
API reservation cleanup distinguishes a procfs file disappearing during its read
from an unreadable or malformed process identity. Explicit disappearance can prove
exit; unknown reads keep capacity charged. PID/start-time and ancestor checks,
capacity limits and deadlines are unchanged.
The private-stdio native HTTP tests use loopback peers and fake credential providers
to exercise stalled headers, chunk framing/trailers, output limits, TLS rejection,
and bounded `aws configure export-credentials` before frozen-credential signing.
They are not live AWS credential-provider or AMP/Grafana acceptance evidence.

Prerequisite-repair regressions check that the original refresher snapshot and
candidate/database binding are journaled before Store-dependent gates, that the
one-shot refresh cannot restart business Deployments, and that the application
transaction adopts the original snapshot. Internal `preflight --for-deploy` may
perform this preparation and create proof Jobs; public preflight remains read-only.
Job tests cover lost create acknowledgements, spec/UID drift, failure and interrupt
cleanup, and retrying cleanup before another Job. Proof Jobs have one non-controller
owner reference to the managed Aurora refresher CronJob's exact UID, also bound in
the journal and checked in admission previews and live Job reads. This provides
parent garbage-collection coverage without making the proof a scheduled/active
CronJob run; parent GC and TTL do not replace explicit journal/UID cleanup.
Discovery fixtures treat only a sole matching registered CronJob owner in the same
namespace as covered. Orphans, recreated parents, unregistered parents and ambiguous
ownership remain unregistered and block cleanup preflight, without blind deletion
or adopting customer-namespace Jobs. These local fixtures do not prove live
Kubernetes garbage-collection timing.
The checks verify the admitted Job template, not subsequent Pod admission changes;
CPU Pod admission remains trusted.
Early rollback tests reject invalid previous template content before CPU,
refresher or Profile mutation when Agent/Reconciler compensation is pending.

`tests/regional/test_bootstrap_store_proof_postgres.py` requires a separate serial
run against an isolated local PostgreSQL instance. It exercises real read-only
empty/partial-schema checks, schema identity and active-record refusal; missing
database configuration causes skips, not a successful database gate. Neither
absent business Pods nor a historical `CREATED` marker proves an empty database.
An origin proof authorizes the bootstrap continuation only; every retry needs a
fresh database read. These tests do not prove production Aurora bootstrap,
credential rotation or rollback.

These tests do not measure AWS provisioning, image transfer, Aurora DDL, node pip
installation, or real Kubernetes rollout latency. First-deploy and upgrade speedups,
API throttling at the concurrency limits, and production rollback with these changes
still need an explicitly approved live validation window. No end-to-end duration or
percentage speedup is claimed from local test timings.

Aurora creation tests model the documented AWS requirement that both the cluster
and primary be available before adding a reader. They cover the ordered create
barrier, interrupted creation, and a promoted primary whose identifier ends in
`-reader`; they do not replace a real Serverless v2 provisioning run.

## Administrator lifecycle validation

The lifecycle review separates local regression coverage from integrated release
acceptance. Focused results from overlapping test groups must not be added together
or described as a clean full run. A fixed failure followed by a passing focused run
does not erase the failed full-run result; final integration must be reported
separately after all command and shared-helper changes settle.

The [regional review record](components/regional-acceptance-review.md) retains
the prior integrated test results. Those results are not a fresh full-run
verification of later coverage-hardening changes. The current-source gate must
run after all participating edits stop. Existing migration tests and
administrator fixes do not establish a stable production release cycle.

Join/removal/uninstall regressions use controlled AWS/Kubernetes runners to cover
identity-bound retries, partial acknowledgements, namespace recreation, dependency
ordering, and terminal-state replay. Join tests distinguish compensation before
activation from fail-forward after its persisted intent; batch verification may
account for a completed sibling without refreshing the original proof age.
Removal tests require namespace disappearance before credential/AWS revocation and
allow a partial site commit to resume only through its exact recorded ARN.
Namespace deletion carries the saved UID as an API precondition, and a recreated
UID blocks its waiter. Node metadata cleanup verifies the complete saved Node UID
set before issuing UID/resource-version guarded patches, with at most eight
metadata workers. This is API concurrency, not a node-unavailability allowance.
Uninstall tests preserve immutable registry identity while applying the authorized
effective cleanup policy, retain Kubernetes evidence across late retries, and keep
Aurora behind the non-database cleanup barrier. The known legacy LBC attachment
edge is normalized only in an ephemeral graph shared by validation and deletion:
detach the role's policy, delete the role, then delete the policy. Persisted
dependencies and all other immutable identity fields remain unchanged. Retained
consumers, missing dependencies, cycles and the cleaner's attached-policy veto
remain fail-closed.

Node-key helper regressions cover CPU union CAS, preservation of unrelated keys
and metadata, exact GPU membership, existing random rotation values, pending
rotation recovery, lost acknowledgements and fresh-process retries. Both native
`List` and typed `NodeList` wrappers are accepted only with valid Node members.
The parent probe reads raw Secret JSON privately and compares digests of the
actual decoded bytes, not names alone; any pending rotation marker requires
ensure, even when both maps match. Join binds the helper to a fresh NodeName-to-UID
map and records initial actual GPU key digests for compensation. CPU rollback
requires the matching current digest and resourceVersion CAS; equal key bytes
alone do not establish cross-cluster NodeName ownership. Bundle inclusion,
TaskInputSpec and release identity bindings are present in the source; focused
helper tests are not a completed signed-release or live node lifecycle gate.

The installed-resource tests include actual local `kubectl` against loopback TLS
fixtures, native exec-protocol handling, and temporary component wheels/Node
bundles. These exercise real client JSON list behavior and packaging boundaries,
not a live EKS API or AWS IAM token service. The deploy tool manually caches only
standard `aws eks get-token` credentials with explicit expiry; generic exec plugins
and other exec options remain under native kubectl, including
`KUBERNETES_EXEC_INFO` and `provideClusterInfo`. Unknown historical registry rows
must stop synchronization before any rewrite even if the object is already absent;
a self-computed SHA is not source authorization.
The collector also retains the full synchronized document for each raw context
under `gpu.by_context[context]`, including its actual `.resources`. It rejects
missing or duplicate contexts before CPU writes. The top-level resource union
does not authorize cleanup in every GPU cluster; multi-cluster cleanup requires
the selected contexts' ownership documents.

`tests/admin/test_registry_sync_postgres.py` separately exercises real PostgreSQL
registry writes, atomic immutable-identity conflict handling and whole-batch
rollback. `tests/admin/test_uninstall_probe_postgres.py` separately covers read-only
cleanup probes, unknown/orphan active records, missing tables and scoped queue
checks. Each requires an explicitly configured isolated PostgreSQL database and a
serial run; missing configuration means skipped coverage, not a database PASS.
These tests do not prove Aurora network/TLS behavior, credential rotation, final
snapshot availability or a production database deletion.

Token-rotation regressions pin `keep_window`, `window_minutes`, `quiet_seconds`
and `acceptance_timeout_seconds`, including in-progress and cleanup-only resumes.
The acceptance helper fixes each probe's quiet-window boundary and verifies
stable Ready ingress Pod/api-container identities. Before window reads, it takes
a no-since, explicit `--tail=-1` current-log prefix capped at 4096 bytes and
requires a complete first record no later than the window start. After all window
reads, the exact complete-prefix byte length and digest must still match.
Normal append is allowed; unprovable retention, incomplete records, changed
prefixes, future timestamps and late responses fail closed. Receipts are local
to the probe, not a new runtime metric or persisted raw-log artifact.

`tests/admin/test_admin_rotate_token_kubectl.py` runs actual kubectl against an
explicit credential-free loopback HTTP fixture to check framing, prefix byte
caps and fixed `sinceTime` query behavior. It is not a real Kubelet/CRI server,
production TLS acceptance or a live token rotation. The node-key and registry
loopback TLS tests likewise establish only their local client contracts.

Remaining operational limits:

- The canonical state-directory `flock` is local coordination, not a distributed
  lock across deployment hosts or copied state directories. FD ownership and
  thread-exclusion tests do not replace registry CAS, Store fencing or node
  resource-version checks.
- Older in-flight journals without the required site, cluster, namespace, node,
  resource or database-incarnation binding need explicit reconciliation.
  Removal v1 journals cannot resume as v2 without that review. A valid file digest
  cannot reconstruct missing authority. Do not edit checkpoints or discard
  evidence to make a retry appear new.
- Token-rotation prefix receipts depend on trusted normal append/rotate
  current-log semantics. They cannot rule out silent Kubelet/CRI omissions or
  malicious same-prefix rewrites, or demonstrate migration of
  offline callers that made no requests during the window. No old complete first
  record within the 4096-byte bound means refusal, not quiet. Older active
  journals without pinned policy require explicit reconciliation. The persisted
  local token-write intent remains an irreversible fail-forward boundary,
  including lost ACKs.
- Namespace and temporary cleanup-resource UID checks are modeled locally. They
  do not establish live admission behavior, controller convergence, garbage
  collection timing or node systemd teardown under real EKS failures.
- AWS cleanup tests check exact error handling, current ownership, reader-before-
  writer deletion, nodegroup request waves and serial Fargate profile deletion.
  They do not measure AWS eventual consistency or throttling under those limits.
  Shared IAM, DNS, SQS, LBC and network dependencies still require explicit
  resolution when ownership cannot be proved.
- A local timeout or lost process-supervision proof does not prove a remote action
  was canceled. In particular, collector-outbox polling can finish while its
  workflow remains active; inspect that workflow before submitting another.

No live join/removal/uninstall, cloud resource deletion, Aurora lifecycle test,
GPU reset/reboot, warm-spare mutation or workload restart is established by this
review. Stage tables describe dependency and concurrency contracts, not measured
end-to-end speedups. Store migration verification is a separate gate and is not
implied by administrator-command checks.

## Release signing: Rekor 409 recovery

`scripts/cosign_sign_blob.py`, the wrapper every `cosign sign-blob` of the release tooling runs through, is covered
hermetically by `tests/test_cosign_sign_blob.py`: a fake `cosign` and a fake Rekor on localhost exercise the pass-through
paths, the 409 recovery and each binding it requires (artifact digest, signing key, UUID leaf hash, entry kind, inclusion
proof), a rebuilt bundle that `cosign verify-blob` rejects, the bounded Rekor read retries and keyless signing. The suite
never signs, uploads or reaches the public Rekor. The bundle mapping was also checked once offline against a real conflicting
entry of the public log: the rebuilt bundle passed `cosign verify-blob --key` and `scripts/verify-release-attestation.py`. A
live signing that hits the 409 cannot be provoked on demand; the first such event in a release build or deploy is its live
evidence.

## Optional HMA ingestion retirement

The custom Kubernetes HMA watcher and CloudWatch/Lambda/SQS pipeline, their four
HTTP routes, commands and deployment assets are retired. Kernel/Fabric Manager
normalization moves to `gpu_fault.nvidia_logs` without changing wire fields, event
identities or timestamp semantics. Regression coverage retains parsing, source
correlation, unresolved-line handling and generic sink behavior, and checks retired
route rejection, legacy data decoding and cleanup-only resource discovery.

The candidate preflight refuses an upgrade while either historical producer
Deployment exists or its absence cannot be established. This check does not prove
that historical AWS queues, DLQs, subscriptions or outboxes have been drained.
Their retirement and data retention require a separately approved procedure using
the old release. No live HMA teardown, AWS resource deletion or current-cluster
quiesce validation was performed. AWS HMA, Node Agent quiesce and warm-spare
health-label safety checks are unchanged.

## Code and scenario coverage

The 95% improvement objective separately measures statement and branch coverage
for production code and regional runners. A combined percentage is not enough:
entirely unexecuted branch alternatives must remain in the denominator.
`tools.coverage_objectives` checks the complete current source-file inventory and
reports both metrics without lowering existing CI floors.

The independent mechanism inventory in `testcases/scenario-requirements.yaml`
retains known implementation gaps and validates operation/channel membership.
`tools.scenario_coverage` distinguishes design, reviewed implementation bindings
and fresh local test verification; it never produces LIVE verification. A matrix
row is not evidence that its implementation handles every possible timing,
hardware failure or production topology. Rule replay counts cannot mask critical
mechanism gaps. See [the definitions](components/scenario-coverage.md).

Scenario result aggregation also validates discovered tests and explicit clean
collection metadata through the shared receipt parser. Expected parameter counts
alone are insufficient: every discovered variant must have passed all three
phases, possibly across complementary source-matching shards. Path aliases cannot
hide a failed result. These checks make local statistics stricter; they do not
close the independent physical-safeguard or installation-custody gaps.
Real serial and xdist child regressions check an alternate `--rootdir`: an ignored
passing copy cannot verify a same-named source test, while running the actual
source test under a different collection root remains valid. A clean checkout
started from its ignored subdirectory is still bound to the Git top-level root.
These are local
test-provenance checks, not deployed execution evidence.
Reporter path mappings are established during collection and reused during
execution, including tests that replace process/filesystem APIs. A new pytest
configuration clears these mappings; results cannot reuse another root's paths.
Reporter self-tests use an isolated module and no inherited report destination.
Their synthetic failures/skips must not reset the active suite's discovery,
delete worker reports or appear as real suite results.
Executor concurrency tests use a shared entry barrier and require both successful
completions. A serial execution cannot satisfy that proof; a fixed subsecond wall
clock under a loaded parallel test host is not a production latency guarantee.
CAP-004 PostgreSQL tests distinguish the one deliberately withheld result from
late renewals rejected after an acknowledged terminal commit. Counter totals
must equal the injected rejection plus one explained rejection per same
acknowledged lease, with no unconfirmed rejections. Deterministic tests cover
both ACK orderings and reject wrong-owner, unacknowledged and later-lease
explanations. Production counters and runner reconciliation are unchanged.

Capacity probes require complete production Deployment/Pod inventories before
load and after cleanup. Disabled zero-replica roles remain explicit, and both
main-container and init-container restarts enter the comparison. Empty or partial
API responses do not establish an unchanged baseline. These are local behavioral
checks of the runner, not measured production capacity or a fleet-wide SLO.

Common-model regression tests also exercise zero/negative reason budgets and
history containing repeated truncation markers. Zero retains no reasons; negative
budgets are rejected. History compaction retains one accumulated truncation marker
without indexing past the remaining real events or duplicating them. Normal
incident/workflow history budgets and recovery policy are unchanged.

## Runtime boundary regressions

The coverage-hardening review also exercises real recovery and transport logic
behind fake I/O boundaries. These are functional regression tests, not additional
physical-fault or provider acceptance results.

- A placement hold whose dissolution cannot be claimed remains held for another
  dispatcher tick. The failed claim does not authorize a STOP.
- Kubernetes restart observations match both job and attempt identity. An
  unknown lookup failure cannot stand in for an explicit HTTP 404 and authorize
  creating a replacement workload. Kubernetes timeouts must be finite and positive.
- Unknown HyperPod instance group or instance type is not compatible spare
  capacity. No provider replacement or automatic recovery capability is enabled.
  A live fault marker requires the matching cluster, incident, workflow pointer
  and recovery fencing token before successful recovery can clear its spare veto.
  Contradictory bound identities raise an evidence error; the scan stays SUSPECT
  without a new workflow or scheduler write. They are not a new hardware finding.
- An explicitly supplied empty HyperPod node snapshot stays empty; it cannot
  silently refetch a different inventory. Negative workflow step indexes are
  rejected before dispatch rather than selecting Python's final list element.
  Reboot/replacement stabilization restarts after an observed health interruption.
  Replacement confirmation requires scheduler-isolation proof, including legacy
  observation paths; it does not authorize provider replacement submission.
- Training-health event IDs include cluster identity using the existing canonical
  record-ID helper. Placement-hold IDs no longer flatten ambiguous separators;
  legacy hold reuse verifies persisted cluster, job, attempt and workflow identity.
  Notification latch keys remain stable across the event-ID change.
  Old binaries still generate the ambiguous IDs; these tests do not establish
  safe mixed-version training/observation writers. Drain and upgrade those writers
  together, rather than creating cross-tenant aliases for old IDs.
- DCGM warning/message string arrays retain their bounded text for recommendation
  selection. This fixes lost diagnostic evidence without adding recovery actions.
  A malformed recommended-action list is refused without losing the diagnostic
  handoff. NVLink primary and secondary bit patterns are independently matched,
  with the same required error-status gate.
- Redelivery after raw XID retention preserves the existing finalized decision
  and incident binding. Distributed reset incidents retain the validated job and
  attempt identity so terminal withdrawal can find their workflows.
- Preemption compensation that returned WAITING resumes restoration before the
  ordinary step cursor. It cannot run a pending reset before cleanup finishes;
  successful or failed compensation preserves the successor and its audit.
  A DAG's selected restore is not counted as an unrelated in-flight sibling
  that blocks its own compensation; every other unresolved sibling still holds
  finalization.
- Terminal incident closure takes node scope from proven successful
  RESTORE_SCHEDULING steps, including supported sequential legacy receipts.
  STOP scope, allocation, diagnostics, planned releases and skipped releases
  cannot close an independent node's incident or marker.
- Completion success requires complete successful workload-object evidence.
  Emergency-stop ownership remains on the workload after Pod garbage collection,
  and a configured training container must exist in its Pod.
- Unknown or interrupted multi-node action results retain manual-confirmation
  requirements. A later barrier retry cannot convert an uncertain physical
  outcome into successful preparation or commit.
- Corrupt quiesce state is reported while independent saved windows are still
  examined for restoration. The unreadable state itself is not declared restored.
- An unreadable global process inventory cannot prove that no GPU device clients
  exist. The node operation refuses with a retryable read failure before reset.
- Independent identity registries are refreshed even when another cluster fails.
  The scheduled job still reports failure; partial progress is not full success.
  Indexed endpoint-network lookups use the current registry, not startup CIDRs.
  Malformed persisted heads or revisions immediately invalidate readiness until
  a valid refresh; errors contain no raw model payload. Only transient I/O may
  use the existing bounded cache grace.
- Received HTTP errors are not retried as broken keep-alive connections. Node
  error bodies that decode as JSON arrays or scalars retain their HTTP refusal
  classification rather than causing an unrelated attribute error.
- Prometheus path labels use the shared newline, quote and backslash escaper.
  No new metric family or recovery policy is introduced by the encoding fix.
- Processor renewal-start and batch-encoding failures release recoverable claims
  and clear local execution bookkeeping. Durable lease fencing remains required;
  these tests do not establish takeover timing in a deployed CPU Pod.
- PostgreSQL spool duplicates inherit their winning sample's admission result.
  If the global or cluster cap rejects that winner, the duplicates cannot be
  acknowledged as coalesced without a stored sample. Memory and SQLite already
  enforce the same admission contract.
- Spool completion, retry, drop and abandonment also require the current opaque
  claim token, independently of payload revision. Tests exercise an old callback
  while the replacement is still busy, same-owner reclaim, claim refund and
  deleted/recreated lanes. A completed replacement followed by a harmless old
  callback would not prove this interval. The existing owner column stores the
  owner-qualified token, without DDL changes; old unfenced consumers must be
  drained and upgraded before claiming full fencing protection. SQLite composes
  the Memory spool and does not establish PostgreSQL durability.
- Orphan-command cleanup rechecks the complete terminal workflow snapshot under
  its write lock and commits cancellation with the workflow audit. A concurrent
  workflow change or already completed command is not overwritten; leased
  commands receive a cancellation request, not a fabricated physical-stop result.
  PostgreSQL regressions cover commit, rollback and concurrent writers in
  legacy, dual and dedicated state-table modes.
- User-stop withdrawal failures do not commit a terminal decision that would
  suppress retry. Independent workflow updates are still attempted, but the
  first failure is propagated so the completion transaction can roll back.
  Previously cached decisions from an older release are not repaired by replaying
  an unverified incoming duplicate body.
- Completion marker reads filter the cluster before their result limit, across
  Memory, SQLite and PostgreSQL. A foreign or unbound marker cannot supply node
  remediation evidence to the selected cluster. A local marker referencing a
  foreign incident is contradictory identity and is refused, not converted into
  a no-hardware-evidence restart. Passive containment also binds cluster, job
  and attempt before it can reinterpret a terminal event as a controller stop.
- Malformed collector checkpoints and non-object journal/notification payloads
  are handled without treating their contents as valid telemetry.
- SNS uses bounded SDK calls and a 1024-entry process-local deduplication hint.
  Store delivery state remains authoritative; cache eviction and the
  provider-accept-before-commit window cannot promise external exactly-once mail.
  A failed throttled-claim release does not prevent releasing sibling claims.
- Aurora credential replacement preserves escaped usernames and IPv6 authority.
  The final retry error does not expose the raw driver exception chain.

The scenario inventory still retains the stronger post-read physical ownership
fencing requirement and separates implementation from deployed proof. Increasing
Python branch coverage cannot remove that requirement or authorize a blocked runner.

BOOT-032, HA-011 and NOTIFY-008 provide reviewed isolated runners with local
regressions for full native uninstall/resume, busy worker takeover and notification
commit ambiguity. They remain manual and NOT_RUN in the regional catalog.
The integrated cleanup contract now uses schema-v2 and an explicit
`NODE_RUNTIMES_STOPPED` phase after control consumers and GPU Executors stop.
It relies on the original saved fleet/UID proof, not later database reads,
broad orphan SQL or a guessed failed checkpoint. Schema-v1 requires explicit
reconciliation. Earlier full-suite receipts do not validate this changed order;
neither the new phase nor local tests establish LIVE teardown success.
NOTIFY-008 uses independent SIMULATED provider receipts and real local PostgreSQL
process-loss tests; it does not establish SNS/SES acceptance or exactly-once mail.
The four-mechanism implementation pass adds persistent independent reboot recovery,
expiring-fixture cancellation/inhibition and prospective key custody with deployed
activation checks. DESTR-014 owns its
systemd service/timer and original Agent enable-link inode; a separate service
invocation must acknowledge the exact binding before disable. A fixed expiry,
identity drift or lost supervision cannot authorize late recovery, and existing
attempts are cleanup-only. This was verified with fake host/systemd I/O and owned
local process-loss tests, not a real reboot. The current
[requirement matrix](../testcases/scenario-requirements.yaml) retains the remaining
unfenced post-read physical-race gap; the controlled companion does not close that
stronger guarantee.
Historical custody cannot be reconstructed by this new implementation.

The physical late-ownership companion uses a native post-queue/pre-spawn Agent
challenge, a short-lived node-key permit and continuous independent exec tracing.
Its controlled change occurs after queueing and before permit issuance.
Sequential Kubernetes reads, permit issuance and the eventual system call are
not atomic; no guarantee about arbitrary post-read ownership changes is implied.
Local kernel-trace controls execute only a harmless owned child, not a GPU reset.
The three manual companion variants share one mechanism evidence set across
DESTR-015 and PREEMPT-033 and never overwrite or promote an ordinary case result.
See [the precise guarantee](components/late-ownership-acceptance.md#exact-guarantee)
and its cleanup/reconciliation limits. Implementation bindings and fresh full
verification remain separate from unmeasured LIVE behavior.

DESTR-008 now binds all six synthetic requests to immutable activation inhibition
and excludes old-protocol claimants. Three expiring fixtures additionally require
an independently armed CPU cancellation observer, complete source/drain evidence
and run-owned GPU/host protections. Local causal tests use the real watchdog and
Store logic; Kubernetes, service changes and workloads remain controlled test I/O.
Native PostgreSQL claim/history tests are not Aurora acceptance. A matching UID
does not excuse changed declared spec or owner ancestry, and an unknown create
ACK cannot acquire deletion authority from a later snapshot. Closed and partially
retired journals retain their original Plan/control/receipt bindings. No live
expiry, admission convergence or hardware-failover result is claimed.
Capability inspection is acceptance source delivered through a bounded,
SHA-256-checked stdin loader. It does not live in the shared production protocol
module: dynamic literal imports are real packaging dependencies, not an isolation
boundary. Component-wheel isolation and the Node Runtime dependency lock remain
separate enforced checks.

BOOT-016 requires nonempty, uniquely named deployment inventories and positive
integer generations before comparing deployment snapshots. Equal missing values
cannot establish a NOOP. Acceptance output redaction is exercised on long
non-secret and credential-shaped text, including whitespace after `Bearer`;
report-file permission failures also close their temporary descriptors.

Local pytest prerequisites launched through the shared regional command boundary
must produce complete source-bound receipts. Exit zero with skips, xfails,
filtered discovery or missing phases is rejected. Live credentials and inherited
pytest filters/partitions are removed from these local test processes; their
private receipt cannot overwrite an outer suite's report. The supervised process
and timeout boundary remains in force. CAP-005 passes its validated, generated
loopback PostgreSQL database through an explicit opt-in and supplies JUnit and
serial-worker options on the direct pytest command line. Its make/stress subprocess
also strips inherited pytest filters and worker identities. Neither path restores
ambient production database or cloud credentials.
Only an acknowledged CAP-005 database creation authorizes its cleanup. A refused
or uncertain CREATE remains non-PASS and preserves unproven ownership for manual
inspection. The local PostgreSQL lifecycle tests distinguish that case from
cleanup after a failed test child; they do not execute the full CAP-005 stress
runner or establish Aurora acceptance.

The DESTR-005/006/007 read-only warm-spare audit uses the same supervised,
source-bound pytest receipts, with complete parameter and setup/call/teardown
evidence attributed to each independent prerequisite. An exit-zero skipped or
filtered run cannot establish a guard. Node snapshots also require nonblank,
unique names and UIDs before preflight and before postflight comparison; duplicate
names cannot hide a replaced node when rows are indexed. Tests use fake deployed
reads and small isolated pytest children, not physical warm-spare failover.
GPU membership follows resource declarations, including zero allocatable GPUs;
a quarantined zero-allocatable neighbor cannot disappear from either snapshot.
Malformed resource quantities, incomplete inventories and missing known-GPU
allocatable declarations are refused. Existing readiness and business-taint
policy is unchanged.
The entrypoint writes a current non-PASS result before binding command supervision
and validating configuration. The real durable loss marker blocks a fresh retry
before external reads and cannot leave a previous canonical PASS in place.

The node-holder scope probe bounds startup readiness by five seconds and 4096
combined stdout/stderr bytes, and requires its child's exact PID/device receipt.
It checks the deadline again immediately before accepting a complete receipt;
a late wakeup cannot transfer ownership as successful startup.
Failure while starting any holder cleans that child and attempts cleanup of all
previously started holders. Success is printed only after cleanup completes.
Regression tests use owned local children and temporary regular files, never GPU
device operations. They do not establish real device-holder quiesce behavior,
kernel responsiveness, or the missing independent physical recovery safeguards.

The post-delivery Node Action regressions distinguish an unconfirmed response
from a proven pre-dispatch rejection. Exact-command read-only polling cannot
resubmit under a changed intent or adopt a different phase's result. Native
refusal tests retain earlier granted checkpoints without claiming completed
physical effects. Executor tests keep uncertain node actions occupied as
`BLOCKED / NEEDS_OPERATOR` and prevent timeout-driven service restoration.
These controls do not establish a cluster-wide ownership fence, managed EKS
admission activation, or physical completion after an independent failsafe.
The product does not require external writers to adopt a coordinating protocol,
change permissions, or pause their writes. Existing safety prerequisites and
fail-closed checks remain unchanged; undetected post-read races are still a
limitation, not a passing scenario. The original gap, critical classification,
and denominator remain intact.
The explored coordinator is not adopted under this boundary; its unresolved
platform requirements are recorded in
[Atomic Ownership Fence Design](components/atomic-ownership-fence-design.md).

## NVLink fabric state across GPU resets

No NVIDIA counter exposes multicast-team health: after Fabric Manager is
stopped and started around a per-GPU reset, `nvidia-smi -q` Fabric State and
Status, DCGM field 170 and unicast P2P keep reading healthy while every NVLink
multicast team with a GPU the restarted instance does not know fails (NVLS
lost for the node). The product therefore detects a lost fabric state only
through its own quiesce evidence: VALIDATE_GPU's `nvlink_fabric_state_lost`
reads which window stopped Fabric Manager and what it reset. A Fabric Manager
restart outside the product (an operator, systemd restarting a crashed unit)
is not detected, and nothing re-validates a node between workflows. The Fabric
Manager log collector forwards SXid lines only; turning its "failed to find the
member GPU handle ... multicast team setup request" and "All GPUs in the
partition need to be reset to recover" lines into a node-level finding is a
proposal, not implemented. The keep-running policy is verified by unit tests
with fake systemd runners and the live precondition (with NVLink Inband Fabric
Manager holds no `/dev/nvidia[0-7]`) on Hopper HGX only; Blackwell NVSwitch
nodes are covered by the same NVIDIA rule but were not observed live.

HGX B200/B300 (fourth-generation NVSwitches, NVL5; driver product strings
`NVIDIA B200` and `NVIDIA B300 SXM6 AC`) are graded NVSwitch nodes by the same
rules as Hopper SXM, but the site has no p6-b200/p6-b300 node and none was
observed live. The 300 s `fabric_manager_status` window, VALIDATE_GPU's fabric
and Fabric Manager unit checks, the `nvlink_fabric_state_lost` guard, the
per-GPU reset window's `fabric_manager=keep` and the display-stack bystander
rule are all unverified on NVL5: unit tests grade B200/B300 parts (299 s held:
no finding; 300 s: one node finding) and run VALIDATE_GPU, the state-loss guard
and the RESTART_FM chain on them, but nothing ran on an NVL5 node. On NVL5 one
unit, `nvidia-fabricmanager.service`, starts the NVLink Subnet Manager (NVLSM)
and then Fabric Manager (NVIDIA Fabric Manager user guide, "The Fabric Manager
Service": "The FM service unit file is also updated to start the NVLSM process
if applicable. In this case, the FM systemd service status indicates the
process status for FM and NVLSM, and operations such as systemd start, stop,
and so on will operate on both processes."), so keeping or stopping the unit
keeps or stops both.

NVLSM blind spot: the R580 unit's `PIDFile` is Fabric Manager's
(`/var/run/nvidia-fabricmanager/nv-fabricmanager.pid`; NVLSM's PID is in
`/var/run/nvidia-fabricmanager/nvlsm.pid`), so systemd tracks only Fabric
Manager's PID. An NVLSM that died beside a running Fabric Manager does not show
in the unit's state, and VALIDATE_GPU's unit check cannot see it; DCGM field
170 or the Fabric Manager log may. No product component checks the NVLSM
process itself.

The quiesce never signals a process named `nvlsm` or any process of the Fabric
Manager unit's cgroup, but the display-stack bystander rule excuses Fabric
Manager itself only (`nv-fabricmanage` holding nothing but the fabric's own
device nodes). An `nvlsm`, or an NVL5 Fabric Manager, holding any other NVIDIA
device node is not a display-release bystander: the display stack stays
loaded, so a per-GPU reset the driver refuses as busy escalates up the ladder
(fail closed). Which device nodes the NVL5 processes hold was not observed.

Fabric Manager start budget: the Agent waits 780 s for one `systemctl start`
or `restart` of `nvidia-fabricmanager` (the R580 unit's `TimeoutStartSec=720`
plus a minute) and RESTORE_GPU_SERVICES/RESTART_FABRIC_MANAGER may wait up to
1200 s, but the workflow execution budget
(`GPU_FAULT_WORKFLOW_EXECUTION_TIMEOUT_SECONDS`, 1800 s) was not changed. A
Fabric Manager start near NVIDIA's 720 s (on NVL5 the start script polls up to
60 s for the InfiniBand device, then starts NVLSM and Fabric Manager) keeps the
full GPU + NVSwitch reset chain close to that budget. No NVL5 start time was
measured. The same 780 s bounds the Agent's boot reconcile, which restores a
quiesce state file a reboot orphaned before the Agent serves its first command
or sends its first heartbeat: a `systemctl start nvidia-fabricmanager` there
joins the unit's own boot start job, so an NVL5 Fabric Manager that takes long
to start delays the Agent's startup up to 780 s (120 s before) after a reboot
that interrupted a quiesce window. On Hopper the start returns in about half a
minute, or fails; neither case changed.

## Solo XID 45 on graded NVSwitch nodes

A solo XID 45 on a graded NVSwitch node (Hopper SXM, HGX B200/B300) is
observe-only by default: MONITOR_ONLY / NO_ACTION with RESTART_FM kept as the
official action, an incident recorded RECOVERED, the XID investigatory
notification, and no workflow, node action or cordon.
`GPU_FAULT_XID45_SOLO_NVSWITCH_RESTART_FM=true` on the control plane restores
the catalog's RESTART_FM, which compiles there to the full GPU + NVSwitch reset
chain. The default relies on the field-170 grading of the node's latest
samples: a Hopper or Blackwell NVSwitch node whose exporter reports no field
170 (the shipped `dcgm-counters.csv` exports 170 and 171) is not graded and
keeps FREEZE_EVIDENCE → RESTART_FABRIC_MANAGER whatever the switch says: a
Fabric Manager restart, which loses the fabric state there, in a two-step chain
that runs no VALIDATE_GPU. Every other ungraded node (A100, PCIe/NVL, GH200,
GB200/GB300, unknown parts) keeps the Fabric Manager restart as well.

Evidence: NVIDIA GPU Debug Guidelines, GPU Node Triage, section
"Understanding Xid Messages", row "45 Robust Channel Preemptive Removal",
docs version 615, page "Last updated on Oct 04, 2026":
https://docs.nvidia.com/deploy/gpu-debug-guidelines/gpu-node-triage.html#understanding-xid-messages
"No action, informative only. Indicates channels affected by another failure. On A100, this error could be seen by itself due to unexpected Fabric Manager shutdown when FM is running in the same OS environment as the GPU. Otherwise, this error is safe to ignore as an informational message."
NVIDIA XID Catalog, docs version 615, page "Last updated on Sep 09, 2026":
https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html -- code
45, Investigatory Action
"Solo: RESTART_FM / Not Solo: IGNORE (follow other Xid)"; the pinned catalog
data is unchanged, and A100 keeps RESTART_FM.

## GF-GENERAL-LIVE live acceptance runner (phase 1)

The generic `scripts/e2e/regional/run_general_live.py` runner and its
per-category injector/verdict modules (`scripts/e2e/regional/general_live_*.py`,
`probes/general_live_*_probe.py`) add live acceptance for the implemented
phase-1 categories (001/003/004/006/008/009/025/028/029/031). They are
plan-only by default and change no cluster state until an operator runs
`--execute` with the case confirmation and, for a dangerous branch, its
`--approve` token. No GF-GENERAL-LIVE case has live evidence yet: every case
stays `manual` / `NOT_RUN` in `testcases/fault-scenarios.yaml`, and the
module tests under `tests/regional/test_general_live_*.py` exercise only the
injection/verdict logic against scripted product records with fakes for
kubectl and the host probes -- they are not proof of a physical experiment.

Deferred phase-1 variants remain in the denominator with an explicit reason:
live-replay variants that need a product-owned read fixture (a
`GPU_FAULT_HOST_PROC_ROOT` / net-class / exporter-scrape substitution), the
capability-absent variants (no SR-IOV VF netdev, no inactive cluster, no safe
way to fail the live SES channel) and the dangerous branches that need an
approval not arranged this phase (watcher network isolation, control-plane
restart). The implemented verdicts record the known product gaps the spec
expects to FAIL rather than masking them; since batch 172 only the empty
required-interface list remains one. The weighted-time disk await does not: the
host collector reports iostat's `await` (diskstats fields 4+8 over 1+5, with an
interval spanning a counter restart re-baselined instead of rated), so 006
`reference_iostat` expects the product's value and the oracle's standard await
to agree within the tolerance. Inode exhaustion is no longer one of them: the
product grades `filesystem_inodes_used_percent` like capacity, so 004
`inode_full` now expects the full MARK_UNSCHEDULABLE/QUARANTINE chain on the
inode signal and readmits the node through the validated restore, exactly as
`capacity_full` does. Nor is the capacity-only reading of a read-only, vanished
or unreadable configured mount: batch 163 reports each as
`filesystem_unavailable` (READ_ONLY / MOUNT_MISSING / PROBE_FAILED), a
VALIDATE_HOST diagnostic that holds the node while the mount stays unusable and
never resets, reboots or quarantines, so 004 `read_only` and
`required_mount_missing` now expect that chain and readmit the node once the
mount is restored and the signal clears.
Batch 163 took the read-only flag from the collector's own `statvfs`, inside the
unit's `ProtectSystem=strict` sandbox where every mount is a read-only bind: in
that release every GPU node reported `/` and `/var` READ_ONLY (and the 004 test mount
would have read READ_ONLY before the injection and after the restore). Batch
166 takes the verdict from the host's `/proc/1/mountinfo` -- per-mount or
superblock `ro` on the mount holding the path -- which the 004 `read_only`
injection (a superblock `remount,ro` from the storage probe's Pod namespace)
sets and the restore clears; an unreadable host table reports no READ_ONLY,
and a path the sandbox hides (`ProtectHome=`, a `PrivateTmp=` directory, a
mount that never reached the collector's namespace) is not reported at all.

The native storage variants (003 `scratch_full`, 004
`capacity_full`/`duplicate`/`below_threshold`/`inode_full`, and the 006 disk
I/O writers) build a dedicated loop or device-mapper test filesystem under
`/mnt/gfrs-live/<digest>/<name>` and declare it to the host collector through a
`GPU_FAULT_FILESYSTEMS` window drop-in. That mount is only useful if the
*installed* collector can see it: the collector is a systemd unit whose mount
namespace is cloned from the host root namespace at each (re)start, and the
006 I/O writer runs as a `systemd-run` unit in that same host root namespace,
while the storage probe runs in its own privileged Pod mount namespace. A plain
`mount` in the probe therefore stayed invisible to both, and the collector's
`statvfs` of the appended path resolved to the parent (root) filesystem -- the
2026-10-24 live finding where the reading read ~6 % used / ~0.1 % inodes
instead of the filled loop fs, so the variants could not fire. The storage
probe's Pod now mounts the host root with `Bidirectional` mount propagation
(the standard privileged node-plugin mechanism), so the test mount propagates
out of the Pod into the host root namespace and is present when the collector
restart clones its namespace and when the writer unit opens its files. This
needs the privileged probe container (already set) and a node whose `/` is
shared, which the systemd-managed EKS/AL2023 GPU nodes provide; it changes no
product code and leaves the collector unit's own sandbox
(`ProtectSystem=strict`, `PrivateTmp=true`) untouched. Only the storage probe
asks for propagation; every read-only probe keeps its Pod-private default.

The 031 silence variants judge recovery on the product's own success signal,
which is not one poll away for every channel: GPU_METRICS posts every scrape
and GPU_INVENTORY every minute, while the host and node-log collectors post
only on an edge or on their 300 s health summary, and the node log collector's
summary slot counts as a success only when that poll read the journal cleanly
(an error-only batch does not move `last_success_at`). On 2026-10-26 (plain
EKS, release R13g) `logs_silent` held 900 s blind after enabling the collector,
read an UNKNOWN that came from a success five hours old, and then waited 600 s
-- one summary slot and a half -- for a success that never came, because the
R13g collector passes `--since` together with `--after-cursor`, systemd refuses
every poll that holds a cursor and every other poll fails for the life of the
process (`journalctl-failures=72` in 1457 s; batch b190, shipped in R13h). The
variant now waits for the first success after the enable and does not pause a
channel that never succeeded, accepts only the silence's own
`STALE_OBSERVATION` as the pause's UNKNOWN, sizes every recovery wait from the
node's collector environment (two posts at their longest plus start-up spread,
poll, silence scan and retry ladder; 812 s for NODE_LOGS, 817 s for
HOST_TELEMETRY, the former 600 s as the floor), and names b190's refusal
pattern in the stage text when the channel's status shows it. That is still a
FAIL: on R13g a resumed node log collector reports a fresh success only when a
summary slot happens to land on a clean poll, so reruns of `logs_silent` on
either cluster may fail with that text until R13h is live.

031 `disabled_unit`, `expired_agent` and `new_success` could not fail.
`disabled_unit` matched unit names against advisory keys, which name channels
(`collector-silent/<cluster>/<node>/<channel>/<bucket>`); `expired_agent` read
notifications over a window of zero length and required only that an Agent
record existed; `new_success` discarded its UNKNOWN wait and passed on the
recovery alone. `disabled_unit` (read-only) now needs a gpu-fault collector
unit disabled or masked on the node (otherwise NOT_RUN(CAPABILITY_ABSENT)) that
the Agent's own record also reports intentionally disabled -- the product's
input, judged with the product's `CollectorServiceState` rule and mapped to
channels through `collector_kinds` (otherwise INCONCLUSIVE) -- and across two
silence scans: the node's other channels re-recorded during the window (the
scan ran; otherwise INCONCLUSIVE), no observation recorded and no
collector-silent advisory for the excluded channels. `expired_agent`
(read-only) lists the cluster's Agent records with the product's
`agent_is_current` verdict (new read-only `AGENT_CURRENCY_PROBE`), takes one
that is not current -- a lease that expired with a departed node, or a
retired Agent -- and across two scans needs the lane node's current Agent
re-recorded (positive control), no observation and no advisory for the
non-current node, and that node still non-current at the end (otherwise
INCONCLUSIVE). A cluster with no such Agent is NOT_RUN(FIXTURE_NOT_READY):
expiring a live Agent's lease needs a Node Agent heartbeat cut, the pausable
units deliberately exclude the Node Agent (every other case relies on its
lease) and an expired lease trips `gpu_fault_stale_agents`; no fixture does
that. `new_success` now grades the silence first: the paused channel must read
the silence's own `STALE_OBSERVATION` UNKNOWN within its threshold plus the
scan margin (stage `unknown`), then leaving it needs a new success time. The
module tests run the product's own silence scan (`notify_silent_collectors`)
over an in-memory store and read it with the harness's real probes: the
disabled unit and the expired Agent are excluded, and the same graders FAIL
the records of a scan that did not exclude them.

The 006 disk-I/O variants use the same probe for a loop + device-mapper test
device (`dm-N`, the name `/proc/diskstats` and the product report) and a
`systemd-run` O_DIRECT writer, but mount visibility was not their blocker. On
2026-10-25 every 006 variant still failed on both plain-EKS nodes with `host
probe failed (exit 1; error TimeoutExpired)` about 290 s into the run, before
any product evidence and with an empty cleanup stack. The class came from the
probe's own JSON, so a command inside the probe timed out, not the runner's
`kubectl exec`: `dmsetup create` in `dm-create`, at the probe's 180 s default
command timeout. libdevmapper makes every create, resume and remove wait until
the host's systemd-udevd confirms a notification cookie -- a SysV semaphore that
`95-dm-notify.rules` completes with `dmsetup udevcomplete $DM_COOKIE` -- and SysV
IPC objects are per IPC namespace. The probe Pod is hostPID/hostNetwork but has
its own IPC namespace, so udev's completion in the host namespace never finds
the Pod's semaphore and the create waits forever; 004's loop file system never
calls `dmsetup`, which is why it worked. Both nodes' journals confirm the
mechanism: seconds after each Pod-side `dm-create`, `(udev-worker)` logged
`dm-N: Process '/usr/sbin/dmsetup udevcomplete <cookie>' failed with exit code
1` (ip-10-0-43-5 at 07:40/07:45/07:51Z, ip-10-0-32-162 at 07:38/07:44/07:49Z,
one per loop attach), and each run's deadman `cleanup --kind dm-io` later
removed the mapping, loop and backing file (`clean: true`). The
earlier `sustained_latency` `ProbeError` fits a second gap: the variant requires
`KERNEL_MODULE_LOAD`, but nothing loaded `dm_delay`, so the probe refused the
delay table. Neither depends on a loaded `dm_delay`, so the previous
`NOT_RUN(FIXTURE_NOT_READY)` deferral of `sustained_util`/`sustained_latency` is
withdrawn.

The harness now builds the device in the host's namespaces. `dm-create` arms the
deadman, records the device and only starts a bounded transient setup unit
(`RuntimeMaxSec`; `fallocate`, `losetup`, `dmsetup create`, `mkfs.ext4`, `mount`,
each step recorded before it runs) and returns; the runner polls `dm-status`
until the record reads `ready`, or `failed` with the step, the error class, a
digest and the probe site. Every other `dmsetup`, `mount` and `umount` of the
device joins PID 1's IPC and mount namespaces through `nsenter` (a node whose
survey lacks `nsenter` or `systemd-run` is `NOT_RUN(CAPABILITY_ABSENT)`). Each 006
probe call has a wall-clock budget the probe clamps every command to, and the
runner's `kubectl exec` timeout is that budget plus a 60 s transport margin, so a
call always answers with its own error first. Every undo is on the LIFO stack
before the call it undoes; the device undo is the probe's `cleanup --kind dm-io`
(also the deadman's command), which stops the setup and writer units, removes
the mapping, detaches only loop devices backed by the run's own file, unloads a
`dm_delay` the run loaded (`--load-module`, recorded before the load) and fails
the cleanup unless the residue is empty. `counter_reset` now stops the writer
before it removes and recreates the mapping (the writer's open files kept the
mount busy) and restarts it afterwards. A probe failure now reads `error <class>
<digest> at <function:line>` instead of a bare class. The first live runs on
the fixed code (2026-10-25) passed: `sustained_util`, `reference_iostat` and
`counter_reset` on ip-10-0-43-5, `short_flush`, `zero_completed_io` and
`recovery` on ip-10-0-32-162, the setup unit building `dm-0` in about 14 s;
`sustained_latency` is registered as runnable with `KERNEL_MODULE_LOAD` and
waits for a window without open incidents on the node. Its first live run in
that window (2026-10-25 14:47Z, R13e) failed at the recovery stage ("no fresh
sample below threshold"): after `io-stop` the dm-delay device completed no I/O
for the whole 690 s wait (the oracle's 2 s samples from 14:53:55Z to 15:05Z
read util 0 and no await), so the product -- by its zero-completed-I/O rule --
emitted no `disk_io_await_ms` sample and the signal could only stay active;
with the 200 ms table still loaded, any I/O would have read >= 200 ms. A
structural fixture gap, not a product gap, and not the 300 s health-summary
cadence (the wait covers two summaries); the device was torn down only after
the wait. The await recovery proof now swaps the data table back to linear in
place (`dm-retarget --target data --delay-ms 0`: same dm-N, mount and files)
and keeps a paced single writer (one 64 KiB O_DIRECT write every 100 ms, far
below 1 % util) on the device so the product samples a below-threshold await
there, stopping it after the wait; the util variant needs neither (an idle
device reads 0 % from field 10 over elapsed time). The second live run, with
that recovery proof (2026-10-25 17:04Z, eksb, R13e), failed at the detection
stage ("no disk_io_await_ms on dm-0 finding reached an incident"). The
product's records (read-only `ACTIVITY_PROBE` replay): the edge batch at
17:04:25Z (`threshold:disk_io_await_ms`, await 211 ms), 17:05:11Z
(`threshold:disk_io_util_percent`), the health summary at 17:08:30Z (util 100,
await 210) -- which minted the util finding (199 s after its edge, past the
120 s sustain) but not await, whose first active sample was earlier. The cause:
`reference_iostat` on the same node (14:56-15:05Z, dm-delay 200 ms) had minted
an await incident at 15:03:10Z, which set the device-scoped signal
`.../disk_io_await_ms/dm-0` active and notified; it then stopped its writer and
destroyed the device without one below-threshold await sample ever delivered
(the 15:03:40Z `recovered` batch cleared util only), so the signal stayed
active and notified until 17:14:22Z and the new run's active samples read as
the same notified episode in `store/shared/xid.py` `_next_health_signal_state`
(`previously_notified`: nothing minted). A fixture-hygiene gap: every variant
that drives await over the threshold now clears the signal before teardown
(`reference_iostat` takes the delay out, runs the paced writer and waits for
the product's recovery samples; its `signal_cleared` stage fails when they do
not clear it; `_io_metric_variant` raises its INCONCLUSIVE only after the
recovery proof). Worth recording as a product-gap candidate: a device-scoped
signal whose device stops reporting never resets its notified latch, and
every later episode of that key is swallowed. The detection budget also moved
to the cadence model 003 proved: from the oracle's first reading at the
threshold, `sustain + 2 x health summary + collector interval + 180 s` (915 s
on this site; before, 510 s from the load start, while a finding can take up
to ~435 s after the edge -- tens of seconds of slack), with the writer and the
oracle covering ramp (180 s) + budget + 240 s; `sustained_util`,
`sustained_latency` and `recovery` share it (45-minute windows) and nothing is
waited on to settle when no incident exists. `reference_iostat`
takes an optional `--approve KERNEL_MODULE_LOAD`: its device gets the 200 ms
dm-delay table (a loaded `dm_delay`, or one loaded for the run and unloaded in
cleanup) so the awaits sit above the 5 ms tolerance floor -- a linear device on
NVMe reads sub-millisecond awaits that no formula difference can lift past it
(2026-10-25 on ip-10-0-43-5 the product's 0.30-0.44 ms against the oracle's
0.33-0.76 ms read PASS on this expected-FAIL variant); such readings, and a run
with no await samples to compare, are `INCONCLUSIVE` now. Two more follow-ups
from those runs: the runner reads a `HostProbeError`/`RegionalFixtureError` raised
by its own fixture as `INCONCLUSIVE` (`detail: harness fixture failed: ...`),
no longer as `FAIL` -- the pre-fix `reference_iostat` on ip-10-0-32-162, an
expected-FAIL variant, had been shown as its registered product gap with an
empty note -- while an unclean undo after such a failure still fails the
variant; and `cleanup`/`dm-destroy`/`fs-destroy` prune the run's emptied
backing directory and mount root (`residue.directories` names any still
present) instead of only `--kind all`, which nothing called -- both nodes
carried dozens of empty `gfrs-live-<digest>` directories.

The native network variants (008 NIC link down, 009 NIC errors/drops) build
host-local `gfrs<digest><slot>` test interfaces (a `veth` pair, a `dummy`) with
`general_live_net_probe.py` and declare them required to the host collector
through a `GPU_FAULT_REQUIRED_INTERFACES` window drop-in; the collector reads
them from `/sys/class/net`. Unlike the storage mount-namespace bug this has no
network-namespace analogue: the host probe Pod already runs `hostNetwork: true`,
so `ip link add` creates the test interface in the host network namespace where
the installed collector reads it -- visibility was never the defect. The defect
was that `create_veth`'s before/after "management plane untouched" digest hashed
each real interface's per-interface statistics counters (`rx_packets`,
`tx_packets`, `rx_errors`, `tx_errors`, `rx_dropped`, `tx_dropped`), the full
`ip route` table and every pod veth. Those move on their own on a live node --
the management ENI alone carries the probe's own `kubectl exec` stream and the
node's control-plane traffic -- so the before/after equality tripped on ordinary
traffic and the probe raised `ProbeError("creating the test veth changed a real
interface digest")`, surfaced as `HostProbeError: host probe failed (exit 1;
error ProbeError)` before any product evidence (the 2026-10-24 live finding on
008 `required_link_down`). The digest now hashes only the *stable configuration*
(`operstate`, hardware `address`, `mtu`, `carrier`, `ifindex`, `flags` and the
assigned addresses) of the real interfaces the probe already refuses to touch --
the addressed or `device`-backed management plane (the management ENI, the
VPC-CNI ENI, the EFA netdevs) -- dropping the volatile counters, the per-Pod CNI
route table and the ephemeral pod veths. It still changes when a real interface
is downed, re-addressed or renumbered (the safety property the comparison exists
for), so this only widens the tolerance to normal traffic, touches no product
code and does not change which interfaces the probe operates on (still only
`gfrs*`). Every 008/009 variant that creates a `veth` is recovered by it: 008
`required_link_down`, `interface_missing`, `optional_interface`, `recovery` and
the live-replay `statistics_failed`, and 009 `errors_delta`, `sustained_drops`,
`short_drops`, `counter_reset`, `wrong_interface` and `recovery`. 008
`unknown_state` (a `dummy`, which never computes the digest) and the
observation/capability variants (`empty_required_list`, `bond_vf`) were never
affected. 009 `tcp_only` was recovered from the `ProbeError` but stayed
`INCONCLUSIVE`: its `tc netem` TCP-loss flow in a test netns was not wired as a
probe command, so the oracle saw no RetransSegs rise -- a separate harness gap,
independent of interface visibility, closed by the fixture described below.

008 `empty_required_list` decides from the node whether an empty
`GPU_FAULT_REQUIRED_INTERFACES` is a site FAIL at all. The net probe's read-only
`inventory` lists every real netdev (not `lo`, not the run's test interfaces)
with its sysfs `device` link, driver, RDMA device, global addresses, whether it
carries the main table's default route and which policy routing tables default
through it; `interface_roles` then classifies generically: the default-route
interface is the management plane, a device-backed netdev that is the default
route of a policy table (the pod network's secondary ENIs -- the VPC CNI keeps
one table each) belongs to the CNI, a netdev without a device link is a
container/virtual interface, and every other device-backed netdev is a
dedicated data NIC the list could name. A node without one (this site's p5en:
primary ENA, VPC-CNI secondary ENIs, EFA-only `rdmap` devices with no netdev)
records NOT_RUN(CAPABILITY_ABSENT) -- an empty list is the correct
configuration there (user decision 2026-10-25) -- while a node with a dedicated
data netdev keeps the registered expected FAIL naming that netdev.

A second `ProbeError` on the veth path survived that digest change and was caught
live on 008 `optional_interface` (2026-10-24, after the digest fix shipped):
`forbid_real` -- the guard the probe runs before `set-link down` so it never
downs a real NIC -- rejected the test veth as colliding with a real device.
Bringing a `veth` up makes the kernel auto-assign it an IPv6 link-local
(`fe80::`, `scope link`), and `interface_addresses` counted that as a real-device
address, so `set-link down` on the probe's own interface refused itself. The same
link-local also let the host-side pod veths (which carry only a link-local on the
host side) back into `management_plane`, re-exposing the digest to Pod churn.
`interface_addresses(..., include_link_local=False)` now drops `scope link`
addresses from both the `forbid_real` collision test and the management-plane
membership/digest, so the probe tolerates its own and the pod veths' kernel
link-locals while a routable (global) address or a `device` link still marks a
real device. This recovers the three `set-link down` variants (008
`required_link_down`, `optional_interface`, `recovery`); the other veth variants
never call `set-link down` and were already recovered by the digest fix. Only 009
`tcp_only` remained `INCONCLUSIVE`, for the unrelated netem gap above.

The next live pass (2026-10-25, plain EKS) ran 008 `required_link_down` end to
end and still read FAIL, "no network_link_down on gfrs0d3ec4a finding reached
an incident", while the product had done what the variant expects. The first
host-collector tick after `set-link down` was delivered at once (a new
`threshold:network_link_down` edge; the rule has no sustain window) and minted
`inc-<that batch>-network_link_down-gfrs0d3ec4a`; MARK_UNSCHEDULABLE and
QUARANTINE isolated the node under it, and the runner's validated restore
later readmitted the node through that incident, found by the node's ownership
annotation. Both ends of the test veth pair are declared required, and a veth
end set down leaves its peer `lowerlayerdown` (NO-CARRIER; for a required
interface the collector reads every operstate but `up` as down), so the same
tick carried a second CRITICAL QUARANTINE finding, for the peer. The product
folds a critical quarantine that arrives while the node's isolation workflow is
in flight into that incumbent (`orchestration/families/drain.py`,
`ingest_terminal_node_quarantine`): one incident and one isolation, the peer's
event linked to it, the record re-labelled `NODE_HEALTH_GROUP` with its id and
minting event id unchanged. The runner's matcher took only `NODE_HEALTH`
incidents, polled its whole 420 s budget and then waited another 420 s for the
incident it had not found to settle. `finding_incidents` now takes every label
the product gives the record a node-health finding minted -- `NODE_HEALTH`,
`NODE_HEALTH_GROUP`, `NODE_RESOURCE_GROUP`, and `GPU_FAULT_GROUP` for a finding
on a node a managed attempt holds -- because the event id, not the label, says
which finding minted an incident. The 008 link variants (`required_link_down`,
`interface_missing`) also grade the isolation chain on the read in which its
workflow is terminal (the detection read can catch the incident
`ACTION_PENDING` seconds before it is `QUARANTINED`), skip that wait when
nothing was detected, and hand back through the validated restore every
`network_link_down` incident of either end of the pair.

The same pass ran 008 `interface_missing` on the other plain EKS node and the
opposite happened: the runner found the incident of the deleted end, and the
node was QUARANTINED, but the validated restore of the node's owning incident
ended FAILED. Two product defects and one runner defect met there. The peer's
finding had not joined the first end's isolation: the node's merge-group link
still named an isolation an earlier case had merged and ended, the
terminal-quarantine route compared the new finding against that ended pair
and opened a second MARK_UNSCHEDULABLE + QUARANTINE beside the in-flight one,
and that second incident took the node (the contrast with the first node,
whose link was clean, is exactly that link). Both drain routes (the critical
quarantine and the device-resource finding) now treat a linked pair whose
workflow has ended as no link and merge into the live incumbent. The
restore then failed on the pair's last reading: the runner had closed the
collector window with the pair still deleted, so the newest readings of both
ends stayed "missing" (`network_link_down=1`, `network_link_up=0`), inside the
validation window, and VALIDATE_HOST judged them although the collector no
longer reported either interface. VALIDATE_HOST and VALIDATE_FABRIC now drop
a link reading older than the node's newest host batch without a collection
error -- that batch reads every interface in the collector's scope -- so an
interface taken out of `GPU_FAULT_REQUIRED_INTERFACES`, or deleted, no longer
blocks a restore, while one the newest batch reads down still fails it and a
batch with a collection error still makes the step wait. And the runner now
restores the original interface configuration before the controlled
readmission, as the specification asks: `interface_missing` recreates the pair
while it is still declared required and waits for the product to read both
ends up again before it closes the window; `required_link_down` and
`recovery`, which closed the window seconds after `set-link up` and passed
only because a collector tick happened to fall in between, wait for that
reading too. The regression tests run the three variants over the real
product -- a real host collector over a fake `/sys/class/net` whose pair goes
`down` / `lowerlayerdown` or disappears and comes back, ticking between the
runner's reads, a real control plane over an in-memory store, the real
executor on the simulated Runtime Profile and the real node-activity read,
once on a fresh node and once after an earlier merged isolation -- and fail
with the live messages before the change.

Once the veth path ran, live pass 3 recorded 009 `tcp_only` as
`NOT_RUN(CAPABILITY_ABSENT)` "the node lacks tc": the AL2023 NVIDIA EKS AMI
ships no `iproute-tc` (its survey showed `sch_netem` available but not loaded).
The variant now has a real native fixture (`general_live_tcp.py`). The net
probe's `tcp-setup` builds a `gfrs<digest>ns` test netns and a gfrs veth pair
through the same guarded creation path as the other veth variants (test names
only, refused when a name is a device-backed or routably addressed interface,
management-plane digest before/after, deadman and state recorded first),
creates the peer directly inside the netns, gives both ends static IPv6
link-local test addresses -- no routable address and no route on any other
interface -- and puts `tc netem loss 5%` on the host-side veth's egress only.
`tcp-start` runs a bounded TCP sender in the host netns and a receiver in the
test netns as `RuntimeMaxSec` transient units, so the retransmissions are
counted in the host `/proc/net/snmp` the host collector reads. A named netns is
a bind mount under `/run/netns`; the probe Pod's `/host` is not propagated, so a
netns added from the chroot would be an empty file to the host's systemd (the
network-namespace analogue of the storage mount-propagation finding) -- every
netns command therefore runs in PID 1's mount namespace via `nsenter`, where
the receiver unit and the `cleanup --kind tcp` deadman (flow, qdisc, netns,
veth) see it. Only the host-side end is declared required: the peer is invisible
to the host collector, and declaring it would read as a missing required
interface (`network_link_down`, QUARANTINE). The oracle needs the host
`RetransSegs` to rise by at least 20, netem to drop packets and the receiver to
get data, while the test veth's error and drop counters and every real
interface's error counters stay flat (`INCONCLUSIVE` otherwise). The negative
itself is decided on records the product produces for a *quiet* window.
`tcp_retransmits_delta` has no rule, so a batch carrying it is never an edge
(`collectors/host/collector.py` `_edge_reasons`): inside the flow the host
collector delivers only its health summaries, and `app/ingest/telemetry.py`
keeps a summary with no finding and no collection error as no raw evidence.
Live 2026-10-25, both lanes (releases 13daa08b2227 and d5d7610460e0): 22952 and
23608 retransmissions, 16 polls, not one stored batch -- the previous oracle
waited for raw evidence the product by design never writes for this metric.
What every accepted batch does leave is the `HOST_TELEMETRY` collector status
row (`_record_collector_status`: `host-<node>-<observed_at us>`, upserted per
accepted batch) and the latest-wins records of the required test veth's
`network_errors_delta`/`network_drops_delta` (`evaluate_metrics` over
`LATEST_METRICS_REQUIRED`; a zero ships for them). A *witness* is such a record
observed inside the flow window `[start + interval, stop]`, confirmed against
the runner's own RetransSegs samples: the collector interval that ended at its
`observed_at` counted retransmissions, so the batch it stands for carried a
positive `tcp_retransmits_delta` (`_tcp` emits it unconditionally once it has
a baseline) and the control plane graded it; a raw-evidence batch that happens
to fall inside the flow with a positive value counts too. Both records are
latest-wins, so the runner polls every two collector intervals while the flow
runs and keeps every record it saw (`tcp-window.json`). PASS needs a witness
and none of: a NIC/TCP/RDMA finding, a stored batch delivered on a NIC or
fabric `threshold:` edge, a non-zero veth counter in the product's own
readings, a hardware step, an isolation; any of those is FAIL (a finding fails
even without a witness). `INCONCLUSIVE` is left to a failed stimulus or to a
window in which the product accepted no host batch at all (a silent channel).

`tc` and `sch_netem` are used as the node has them. Without the optional
`TEMP_TEST_PACKAGE_INSTALL` approval a missing `tc` keeps the pass-3
`NOT_RUN(CAPABILITY_ABSENT)`. With it the separate
`general_live_deps_probe.py` installs, for this run only, exactly `iproute-tc`
(and `kernel-modules-extra-$(uname -r)` when `sch_netem` cannot be loaded for
the running kernel) after a `dnf --assumeno` preview, and loads `sch_netem` only
if it is not loaded (`KERNEL_MODULE_LOAD`). Its limits are deliberate:

- The previewed transaction must consist of `Installing` rows of exactly those
  packages, plus `Installing dependencies` rows of their reviewed closure only
  (`PACKAGE_CLOSURES`, same build; `iproute-tc` has none). A kernel, an upgrade
  (for example an `iproute` bump that a newer `iproute-tc` would require), a
  downgrade, a removal, an obsolete or any dependency outside the closure makes
  the variant `NOT_RUN(FIXTURE_NOT_READY)` naming the unexpected packages ("dnf
  would also install the dependencies ..."), with the transaction in
  `tcp-dependencies.json`; widening the allow-list is a separate review.
- The module package is only the running kernel's `kernel-modules-extra`
  build; a `kernel6.x` flavour would need `kernel6.x-modules-extra` and is
  reported as "no package ... in the enabled repositories", never guessed.
- What the run installs (NEVRAs from the rpm transaction that installed them)
  and which module it loads are recorded on the node *before* the change, and
  `cleanup` -- the runner's LIFO step after the fixture is gone, and a deadman
  timed to fire after the fixture's even when the install took its whole
  budget -- unloads only a module this run loaded once it is idle, quiets any
  unit the packages brought, and `dnf remove`s exactly the recorded packages,
  last installed first, one previewed transaction each that must remove
  nothing else (no dependent package, no "unused" dependency). A
  cleanup that cannot finish keeps its deadman armed and fails the variant's
  cleanup. A node reboot before either runs loses the `/run` record and the
  timers; the installed NEVRAs stay in the case evidence for a manual removal.

The same step is shared -- `general_live_deps.ensure_dependencies`, one helper
over the same probe, never a copy per category -- by the six 007 SMART replay
variants (`smartctl` -> `smartmontools`) and 004 `probe_error` (`mkfs.vfat` ->
`dosfstools`), the two other tools the AL2023 EKS GPU AMI lacks (pass 5 recorded
all of them NOT_RUN(CAPABILITY_ABSENT)). Without the optional token the NOT_RUN
stays and its detail names the token and the package(s); with it exactly that
package -- with its reviewed closure -- is installed before the fixture that
needs the tool and removed as the last LIFO cleanup after that fixture is gone
(the package deadman is timed after the fixture's). The allow-list is the
probe's `TOOL_PACKAGES` plus the closures of `PACKAGE_CLOSURES`. The one closure
is `smartmontools` -> `smartmontools-selinux`: the rpm requires
`(smartmontools-selinux if selinux-policy-targeted)` and AL2023 nodes ship
`selinux-policy-targeted`, so dnf always adds the subpackage -- the same source
build's noarch SELinux policy module, whose scriptlets load the `smartmon`
module at priority 200 (over the base policy's priority-100 `smartmon`) and
relabel only changed file contexts; nothing runs. It is accepted only as an
`Installing dependencies` row of the same build as `smartmontools`; a preview
that lists anything else is refused by name. Removal walks the record
backwards (`smartmontools`, then `smartmontools-selinux`; the reverse would
cascade), and the residue proves both packages gone, `smartd.service` -- which
AL2023's `90-default.preset` enables on install and the probe disables at
once, the fixture needing `smartctl`, never the daemon -- neither active nor
enabled, and the priority-200 policy module unloaded. `ipmitool`
is deliberately not on it: 015 serves it from the wrapper with
`--allow-missing-real`, so no real tool is needed on this BMC-less site.
`nfs-utils` for 005 is not either: its transaction pulls rpcbind, libtirpc,
gssproxy and more, which the exact-package rule refuses; widening the rule to
a reviewed dependency list is a separate review. No collector is restarted for
an install: `_smart` and `_bmc` run `shutil.which` on every tick, the host
collector unit sets no PATH of its own and its `ProtectSystem=strict` is a
read-only view of the live `/usr`.

## GF-GENERAL-LIVE host/device categories (phase 2)

Phase 2 adds the host/device categories 002 memory/swap, 005 shared storage,
007 SMART, 010 EFA/RDMA counters, 012 GPU missing and 015 BMC
(`scripts/e2e/regional/general_live_memory.py`, `general_live_shared.py`,
`general_live_devices.py` for 010/012, `general_live_smart.py` for 007,
`general_live_bmc.py` for 015). Every required variant is in the registry so none
drops out of the denominator. Executable without any raw-source replay fixture
are 002 `memory_pressure`/`recovery`
(native host memory pressure in a capped cgroup, behind `HOST_MEMORY_PRESSURE`)
and `no_swap`, 002 `cgroup_only` (the negative control: the node probe's
`memory-start --cgroup-oom-bytes` exhausts a 2 GiB test cgroup -- no swap,
`OOMPolicy=kill`, refused above 1 % of MemTotal or the host's headroom -- three
times across the rule's sustain plus one health summary, and `memory-status`
reads systemd's `Result=oom-kill` on the uncollected unit as the proof the OOM
stayed inside the cgroup, also behind `HOST_MEMORY_PRESSURE`), plus 007
`healthy` (a read-only observation that the real SMART reading mints no
finding). `cgroup_only` proves only that the product does not grade a
cgroup-local OOM as host pressure (no `memory_used_percent` /
`memory_available_percent` finding, no STOP, isolation or hardware step while
the host collector keeps delivering normal samples); it is a transient systemd
unit, not a kubelet-managed container, so it says nothing about an
`OOMKilled` Pod status or the completion watcher, and the kernel collector
ignores the kernel's OOM line (it extracts XID/SXID records only), so the
RUN_DIAGNOSTICS grading of that line is exercised only where the node log
collector is enabled. Their module tests under
`tests/regional/test_general_live_memory.py` and
`tests/regional/test_general_live_phase2.py` run the verdict logic against
scripted product records with fakes; they are not proof of a physical fault.
`memory_pressure` has no live evidence yet: on 2026-10-25 it failed three times
on two plain-EKS nodes before any product record because the runner asked the
node probe for a 1980 s hold (the 2 TiB host's 14-minute ramp plus detection)
and the probe's `memory-start` cap was 1800 s; the cap now equals the other
load units' 3600 s, the runner checks the requested hold against it before a
probe Pod touches the node, and a probe failure's JSON names its step
(`error_code`/`error_site`), so the memory tests now also run the real
`memory_start` over a p5en-shaped fake host to pin the accepted hold.

The 007 SMART and 015 BMC replay variants run on the shared host source replay
(`general_live_source_replay.py`, `--approve HOST_SOURCE_REPLAY`, both lanes):
the reviewed `smartctl` / `ipmitool` wrapper first on the host collector unit's
PATH rewrites one allow-listed field of the product's own query (007
`health_failed` `smart_status.passed=false`, `unsupported_device` without
`smart_status`, `query_failed` a verdict-less JSON with exit 2, `malformed_json`
truncated JSON, `scan_failed` a non-zero `--scan-open`) or hides `smartctl`
inside the unit (`tool_missing`); 015 serves a reviewed `ipmitool sensor` table
or exit status with `--allow-missing-real` (EC2 has neither a BMC nor
`ipmitool`; everything but `sensor` answers "command not found"). The drive is
the device of the product's latest `smart_health_failed` reading, the verdict
reads only the product's records (marker + `_smart`/`_bmc` collection error for
UNKNOWN, marker without error for NOT_SUPPORTED, `MARK_UNSCHEDULABLE` +
`QUARANTINE` and the validated restore for `health_failed` / `critical` /
`nonrecoverable`, zero critical for `wrong_column` / `unavailable_sensor`), the
wrapper journal is the oracle that the product asked (otherwise INCONCLUSIVE),
and every variant closes its window and proves the real tool answers again. On
a node without `smartctl` the six 007 replay variants are
NOT_RUN(CAPABILITY_ABSENT) naming `TEMP_TEST_PACKAGE_INSTALL`; with that
optional token `smartmontools` is installed first (exactly that package and its
reviewed dependency `smartmontools-selinux`, the same build's SELinux policy
module that AL2023's `selinux-policy-targeted` makes dnf pull), the
runner waits up to one health summary plus six ticks for the product's first
`smart_health_failed` reading -- the host collector finds the tool on its next
tick, no restart -- and names that drive, the window then restarts the unit on
open and close as always, and the packages are removed as the last LIFO cleanup
after the window closed, `smartmontools` first, so the product reports
`smart_status_unsupported` TOOL_MISSING again. A replay proves the processing chain only -- no drive fails and no BMC exists, so
`proof_scope` is `live-replay`; the OPEN_FAILED drive and a hung `-H` query stay
software-tested; 015 `unavailable_sensor` uses one `na` row among healthy ones
(a table of only `na` rows is UNKNOWN in the product -- a `_bmc` collection
error and no sample -- and stays software-tested). 015 `critical` and
`nonrecoverable` demand the sensors' identity where the product keeps it: the
`bmc_critical_sensor` sample's `sensors` / `sensor_alerts` labels (name,
reading, units, status and the threshold columns ipmitool printed), the
incident `reasons` (`BMC reports 2 critical sensors: CPU1 Temp=96.000 degrees C
[cr; unc 90.000, ucr 95.000, unr 100.000]; Fan1=...`, the text the QUARANTINE
step's hardware escalation mail lists) and the marker `raw_reason`; the bare
count the collector used to post names nothing and was the registered gap. 015
`command_failure` serves two exits in turn: a BMC that does not answer (`Get
Device ID command failed`) must be a `_bmc` collection error with no sample, a
host without an IPMI device node (`Could not open device at /dev/ipmi0 ...: No
such file or directory`, what an installed `ipmitool` prints on EC2) must be
`bmc_status_unsupported` NO_DEVICE with the error gone. The capability marker
is read from two product records (`marker_witness`): the accepted batches
inside the read -- the tick on which a marker appears, changes its `reason` or
clears is delivered once with a `not-supported:...` reason
(`host_capability_markers`) -- and the node's `telemetry_metric_latest` record,
which every accepted batch moves (the health summary included, which writes no
raw evidence) and which is the only witness of a steady marker. The second
phase admits a latest record stamped up to one collector interval before the
read's start (`MARKER_LOOKBACK_TICKS`): the collector stamps a tick when it
starts and runs `_bmc` last, so the tick that answers the replan can carry a
stamp from before the plan landed -- the first live run on plain EKS did
exactly that: that tick was the only delivery carrying NO_DEVICE, every tick
after it was the unchanged marker (no edge), and the read held no batch at
all. The result's `no_device_marker.source` and notes say whether the marker
came from evidence or from `metric_latest`; a latest record with another
reason or older than the allowance, a finding or an isolation still FAIL. The
phase order needs no change: the window restarts the collector unit, the first
phase's collection error is the active reason, and the NO_DEVICE marker of the
second is a new reason against it whatever the node's marker was before the
window (TOOL_MISSING without `ipmitool`, NO_DEVICE with it). 015
`unsupported_bmc` is the read-only capability record of the BMC-less site (like
002 `no_swap`): the product must mark the node `bmc_status_unsupported`
TOOL_MISSING (no `ipmitool`) or NO_DEVICE (`ipmitool`, no BMC) with no `_bmc`
collection error; a steady marker is no edge and the summary is not evidence,
so it is normally read from the `telemetry_metric_latest` record that moved
inside the read (the first live run saw zero batches in a whole summary
period; `capability.marker_source` records the source); FAIL without the marker
(no batch carries it and no latest record inside the read) or with the
permanent error, BLOCKED if a BMC answers. Both variants need the release that
makes the markers edges and latest records: without it `unsupported_bmc`
cannot PASS on a quiet node, and `command_failure` passes only when the
delivered tick's stamp happens to fall after the read's start.

The 010 EFA/RDMA counter variants are wired on the shared host source replay
(`scripts/e2e/regional/general_live_efa.py` on the reusable
`general_live_efa_fixture.py`; every one `live-replay` behind
`HOST_SOURCE_REPLAY`, on both clusters). Each starts from the node's real RDMA
inventory (the probe's read-only `inventory`: driver identity, owned netdevs,
port states, every `counters`/`hw_counters` file, the real `ethtool -S` names)
and replaces exactly one reading inside the host collector unit: a real port
counter file bound at its real value and stepped (`rdma_error`), the port
`state` bound to `1: DOWN` for a bounded exposure (`link_inactive`: the probe
watches the collector's own ticks on the node -- `await-ticks` reads the unit's
`WatchdogTimestampMonotonic`, the last `WATCHDOG=1` the collector's main
process sent, one per contributor and one per posted batch, ticks separated by
a whole interval of silence, counting only stamps after
`ActiveEnterTimestampMonotonic` because systemd stamps the watchdog once more
at start-post on `READY=1` (the 013 eks INCONCLUSIVE of 2026-10-26, below) --
and closes the window right after the tick the
product needs to confirm the shortfall (`GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES`,
2 by default; `general_live_source_replay.observed_exposure`), so the first
tick mints the `rdma_link_down` WARNING, the second is the fresh reading its
VALIDATE_FABRIC must fail on and the one that confirms the
`efa_inventory_mismatch` LINK_INACTIVE co-finding, and no third tick sees the
port down; the window's deadman is only the fallback for a runner that
disappears (`exposure_fallback_seconds`: startup phase + startup jitter +
N x (interval + 10 s) + 20 s, 105 s on this site). Until 2026-10-26 the
deadman itself was the restore, 1.7 intervals (26 s) after the *restart*; a
restarted host collector first sleeps a node phase of up to
`GPU_FAULT_COLLECTOR_STARTUP_SPREAD_SECONDS` (15 s) plus a `[0, 20 s)` start-up
jitter before its first tick, so that bound was a lottery on the phase -- the
two 010 passes had their first tick 8.4 s in, the 013 run that failed 13 s in;
since batch 175 the co-finding's diagnostic judges fresh readings
and would escalate a port still inactive to REBOOT_NODE by design, so it only
ever sees the restored port: SUCCEEDED, RECOVERED, no RESTART_NODE; a link
verdict reached after the restore, an inactive reading after it or no
`rdma_link_down` sample inside the window are INCONCLUSIVE),
or the reviewed `ethtool` wrapper serving, withdrawing or failing a group
counter on the netdev an EFA function owns (`rnr`/`retry`/`cq_error`,
`missing_counter` -> `efa_ethtool_counters_unsupported` MISSING_COUNTER, read
from the batch the changed marker is delivered in (its `not-supported:*` edge;
the first live run found the changed marker never left the node),
`ethtool_failed` -> a collection error, never a zero). Where no EFA function
owns a netdev the fixture tree attaches one of the node's physical netdevs to
an EFA device (`netdev_source: fixture`) so the product's ethtool path has a
reading to replay; the netdev itself is untouched. Cleanup closes the window
and `read-source` proves the unit reads the real sysfs value again.
`counter_reset` ramps both readers past their WARNING, resets them below the
baseline and ramps on; since batch 173 (one reset rule for the three readers,
`HostNetworkMixin._monotonic_delta` / `_reset_problem`) it expects PASS and
asserts the rule: per reader a collection error naming the reader (`_rdma` /
`_efa_network`), the port or netdev, the counter that fell and `delta unknown`,
no reading of that counter's delta or its aggregate (`rdma_errors_delta`,
`rdma_<counter>_delta`, `efa_<group>_errors_delta`) in that batch, no more
incidents per reader than the episodes the product's own readings show (a
runner step slower than the tick ships a legitimate 0 reading that ends an
episode and the next step opens one; the first live run read those as clamps
and four incidents per reader as re-minted while the reset tick was UNKNOWN on
both readers), no negative delta and no reading in the reset tick's batch. 009
`counter_reset` (`general_live_network_reset.py`, the drop interface deleted
and recreated) grades the same way on `_network` and `network_drops_delta`, and
relies on the delivery contract rather than on timing: a collection error is
an edge, so the reset tick is delivered and stored as raw evidence; a second
short burst on the recreated pair ships a `threshold:network_drops_delta` edge
whose delta counts from the new baseline (the `rearmed` stage: positive, never
negative, never more than the oracle counted on the new pair, then a zero), so
the old clamp is observable as readings after the recreate with no reset error
ever named (FAIL). The host collector keys baselines by interface *name*, prunes
the baseline of a name a tick does not list and reads a missing required name
as CRITICAL `network_link_down`, so the recreate is one net-probe command
(`recreate-veth`: `ip link del` then `ip link add` in the same process, the gap
reported in milliseconds with the recreate's node-clock `recreated_at`) rather
than two probe round trips (about 15 s each live, most of a collector
interval). Should a tick still see the pair missing, the product's own
`network_link_down` incident is the witness: the variant waits for both ends'
link readings to recover, closes the window, readmits the node through the
validated restore and is INCONCLUSIVE ("straddled a collector tick"), never
FAIL. The stimulus is the runner's netdev oracle (drops rising before the
recreate, the probe's ifindex change, drops rising again on the new pair) and
every product-side step is polled on a cadence budget (six intervals plus
60 s), never read once after a fixed hold. The 2026-10-25 HyperPod
INCONCLUSIVE (release d5d7610460e0, harness 26f860e4) was the earlier shape:
two round trips, one read 60 s after the recreate, neither an error nor a
reading and no record of what the collector or the counters did; the EKS PASS
of 2026-10-24 (release bf1bb0037847) came from the older oracle that graded no
hardware step only and noted the clamp as an observation -- it did not test
the reset rule, so timing luck played no part in it either. 011
`broken_pair` (`general_live_nccl_pair.py`, both lanes) uses the same bind
on one port's `hw_counters/tx_bytes` around a real two-node stall: while broken
the product must record `efa_traffic_observation_valid=0` with no rate and
start no zero streak although the probe's own pair oracle reads a real zero;
once the pair reads again the still-stalled job's ZERO_WARNING must come no
earlier than the warning threshold after the restore. `non_efa` stays
CAPABILITY_ABSENT (no IB-only RDMA device on this site).

The rest stay `NOT_RUN` for a missing capability or vendor lab only; no
`FIXTURE_NOT_READY` is left in these categories (the 007/010/015 command
replays and the 002/012 `/proc/meminfo` / `nvidia-smi` / `/proc/driver/nvidia`
read replays are all wired on the shared raw-source replay described below,
which replaces a reading only inside the one host-collector unit's private
environment, and 005 runs natively on the test export fixture), never
replacing a host binary or a global mount; `CAPABILITY_ABSENT` where the site
genuinely lacks the capability (no non-EFA RDMA device, no BMC on EC2 for 015
`healthy`); and `VENDOR_LAB_ONLY` for the
012 variants that need a real GPU off the bus and an approved provider reboot
(this lane never unbinds or removes a real GPU). The 012 read paths and the 002
variants are wired on that replay: `single_sample` (one real UUID dropped from
exactly one `uuid,utilization.gpu` query), `query_failed` (an NVML
initialisation failure for four ticks), `proc_unreadable` (the same failure
while the unit reads a copy of `/proc/driver/nvidia` without `gpus`, so its
listing fails -- the two-read-failure case), `read_failure` (an unopenable
device node bound over `/proc/meminfo` inside the unit; systemd's
`InaccessiblePaths=` is no read failure for a root unit, which reads its
mode-0000 node as an empty file and lists its directory node as empty) and
`swap_pressure` (the node's real `/proc/meminfo` with only `SwapTotal`/
`SwapFree` rewritten, since the site runs no swap and business swap must not
be enabled for a test); each is judged on the product's own records --
failure-mode context and a collection error, never a zero count, a healthy
value, a finding or a hardware step -- with the wrapper's journal as the
raw-source oracle. `expectation_wrong` (the recorded expected GPU count plus
one in the unit's environment) expects PASS since batch 174 and needs
`HOST_SOURCE_REPLAY` alone (`S`, risk M): the control plane re-reads every
`gpu_inventory_mismatch` against its instance table and, with the node listing
exactly the type's GPUs and the driver-bound count agreeing, grades the
configured count as a `gpu_expected_count_contradicted` WARNING /
COLLECT_EVIDENCE. The window closes as soon as the product minted a finding on
the override; PASS is no `gpu_inventory_mismatch` incident, one contradiction
incident with a FREEZE_EVIDENCE-only workflow and `effective_action`
COLLECT_EVIDENCE whose reason names the configured count, the instance type and
its count, the listed and the bound count, exactly one OPERATOR_REVIEW
notification, no RESTART_NODE or isolation, and the real expectation read back;
an override the product never recorded is INCONCLUSIVE. The spec's other expected product gaps are recorded
in the variants' reasons rather than hidden. Of the three closed in the product,
the failed SMART scan and the failed drive query are now proven live by 007
`scan_failed` / `query_failed` (`smart_query_failed` plus a collection error,
never zero drives), the EFA `ethtool -S` deltas reported only from real readings
(failures and resets are UNKNOWN, missing counters
`efa_ethtool_counters_unsupported`) by 010 `ethtool_failed` / `missing_counter` /
`counter_reset`, the same reset rule on the sysfs RDMA port counters and the netdev
`statistics` counters (a counter below its baseline is a collection error for that
tick, no `rdma_errors_delta` / `network_errors_delta` reading, never a clamped 0), and an NVML failure with an unreadable `/proc/driver/nvidia/gpus`
being `INVENTORY_UNAVAILABLE` rather than zero GPUs by 012 `proc_unreadable`; an
unopenable drive and a hung drive query stay software-tested.

005 shared storage no longer waits for a site export: `general_live_shared.py`
brings a disposable test export (`shared_export_fixture.py`: one Pod per export
running a userspace NFSv4 server, NFS-Ganesha over a 256 MiB memory-backed
`emptyDir` whose `sizeLimit` is the whole capacity, `activeDeadlineSeconds` as
the cluster-side deadman, deleted by UID precondition; the export root is
`chmod 1777` and `chown 65534:65534` -- with `fs.protected_regular >= 1` a
squashed root's `open(O_WRONLY|O_CREAT)` in a root-owned sticky directory is
refused EACCES client-side, leaving 0-byte files, the image smoke test's first
symptom -- `Attr_Expiration_Time` sits in the EXPORT block, which NFS-Ganesha
4.3 reads it from, and `NFS_KRB5 { Active_krb5 = false; }` silences the
start-up CRIT; the site image is
`<account-id>.dkr.ecr.us-west-2.amazonaws.com/gfrs-test/nfs-ganesha@sha256:d354efdc8364e4ba17b83c4aa27cdef4001a474fc950b07547e7fcf9d1e14d7f`,
Ubuntu 24.04 with NFS-Ganesha 4.3, `/usr/bin/ganesha.nfsd`,
`/usr/lib/x86_64-linux-gnu/ganesha/libfsalvfs.so` and dash as `/bin/sh`) and
mounts it `soft` on the client node in PID 1's mount namespace through
`probes/general_live_shared_probe.py` (record and `systemd-run --on-active`
deadman before the mount, `umount`/`-f`/`-l` cascade, bounded `statvfs` and
write/fsync/read-back readings in the host namespace; the mount is
node-scoped under `/mnt/gfrs-live/<digest>/` and its cleanup unmounts and
removes the directory, which makes the product forget it). The `soft` option
is not what bounds a reading: the smoke test saw a paused export fail neither
`statvfs` within 185 s nor writes within 125 s under `soft,timeo=100,retrans=2`
(the NFSv4.1 session keeps retrying), so the reader's own thread deadline --
the product's 5 s per call, the probe's worker -- is the bound, and the option
only keeps a dead export from parking threads forever. The product has no
shared-filesystem setting -- the host collector discovers `nfs`/`nfs4`/`lustre`
mounts by type from its own mount table -- so the export is deliberately not
added to `GPU_FAULT_FILESYSTEMS` (that would grade it under the local-disk
QUARANTINE/MOUNT_MISSING rules). A hang is `SIGSTOP` on the ganesha child
(TCP still accepted, nothing answered), recovery `SIGCONT` with the same file
handles. `stat_timeout`, `capacity_full`, `healthy_other_mount` and `recovery`
expect PASS (diagnostic-only chain on `shared_filesystem_unavailable` /
`shared_filesystem_used_percent`, other host metrics still delivered, the
healthy second mount never stands in, fresh I/O after the resume).
`mount_disappears` and `io_only_failure` were registered as expected FAILs
naming their product gaps; batch 176 closes both in the product and they now
expect PASS: each NFS/Lustre mount of the host's table gets one bounded,
read-only directory-entry read after its `statvfs` (`IO_ERROR` with the errno
for the server-side `chmod 000`, or `IO_TIMEOUT` as missing evidence --
unavailable, never healthy), and a node-scoped shared mount the table no
longer lists is `MOUNT_MISSING` from a memory kept in
`/var/lib/gpu-fault/host-collector-shared-filesystems.json` across collector
restarts, until it is remounted or its mount point directory is removed (a
teardown -- which is why the fixture's `nfs-umount` keeps the directory for
the injection and its cleanup removes it). Pod volumes and autofs children
are sampled while present but never remembered. `shared_failure_domain`
expects PASS since batch 177 (the correlation that was its product gap): both
clients' baselines must carry one and the same `source` label for the mount
(a missing label or two different sources is NOT_RUN(FIXTURE_NOT_READY) --
nothing to correlate), and after both per-node findings both nodes are read
again for exactly one `event_type=SHARED_STORAGE_FAILURE_DOMAIN` incident
(`policy_reference=shared-filesystem:<source>`, both nodes in `node_ids`, no
workflow or action, a first reason naming the export and the nodes, a
`member <node>: incident <id>` line per node), a `common failure domain <id>:`
reason back on each per-node incident, exactly one
`SHARED_STORAGE_FAILURE_DOMAIN` notification, both per-node chains still
diagnostic-only, and the record `RECOVERED` after the resume. All need
`KERNEL_MODULE_LOAD` (a missing `nfsv4` is loaded and
unloaded), `shared_failure_domain` holds both EKS lane locks (`X`) and needs
`--node-b`. Live prerequisites: an immutable-digest server image shipping
`ganesha.nfsd` and `/bin/sh` (`--export-image` /
`GPU_FAULT_GENERAL_LIVE_EXPORT_IMAGE`), `nfs-utils` on the node AMI, TCP/2049
from the node to the Pod IP, and a kubelet that honours memory `emptyDir`
`sizeLimit` (the first client `statvfs` verifies the capacity). These are
fakes-based module tests until the live run; the fixture proves the processing
chain on a real NFS mount, not a production file server's failure modes.

The first live run of 005 ended all six single-client variants on both EKS lanes
INCONCLUSIVE with `host probe failed (exit 1; error ProbeError <digest> at
nfs_mount:558); output withheld`: the export Pod was Running and Ready on the
other GPU node within three seconds, `nfs-utils` (`mount.nfs4`) and the `nfsv4`
module were present on the AL2023 node AMI, and the client's
`mount -t nfs4 <pod ip>:/a` got `mount.nfs4: mounting <pod ip>:/a failed, reason
given by server: No such file or directory` (the six digests decode to exactly
that line). NFS-Ganesha answers the path walk of a pseudo path with NFS4ERR_NOENT
when the export's root never initialised (`init_export_root` fails and the export
is reverted) or when the client may not see it; the same image, `ganesha.conf`,
entry script, capabilities and mount options serve `/a` to an AL2023 `nfs-utils`
client off-cluster, and the Pod was deleted with its log, so the server-side
reason stayed unrecorded. Three harness changes close that gap: `ganesha.conf`
logs the `CONFIG`/`EXPORT` components at INFO and `wait_ready` requires the
server's own `Export 1 created at pseudo (/a) with path (/export/a)` line beyond
the TCP readiness (a `Lookup failed on path, ExportId=1`, `Config File (` or
`Could not create export` line fails the fixture with that line; a Ready Pod whose
log never shows the export is a fixture error, not a product one); a failed client
mount saves the export Pod log as `export-<suffix>-log-at-mount-failure.json`
before the LIFO cleanup deletes the Pod, and the error names node, server and
file; and the probe classifies the mount helper's stderr into one constant
sentence (`error_kind: nfs_mount`, e.g. `export server answered ENOENT for the
export path (export not initialised or hidden; read the export Pod log)`,
`TCP/2049 connection refused at the export server address`, `TCP/2049 connection
to the export server timed out (security group or network policy)`) that the
host-probe fixture shows under `nfs mount failed:` -- the stderr itself (server
address, mount point) still never leaves the node. HyperPod nodes carry the same
client prerequisites (`mount.nfs4`, a loadable `nfsv4`), so their 005 runs show
the same server-side reason until it is read from the saved log.

The second live run (`stat_timeout`) got that server line from the new readiness
check: `fsal_cfg_commit :CONFIG :CRIT :Could not create export for (/a) to
(/export/a)`. A local kind cluster (containerd + kubelet) with the same image,
ConfigMap and entry script reproduced it and showed the line before it:
`populate_posix_file_systems :FSAL :CRIT :Error 2 in setmntent(/etc/mtab): No
such file or directory`. NFS-Ganesha's FSAL_VFS locates the export's file system
by reading the mount table through glibc's `MOUNTED` (`/etc/mtab`); neither layer
of the Ubuntu 24.04 image ships that file, Docker synthesises `/etc/mtab ->
/proc/mounts` in every container it creates (which is why the image's docker
smoke test exported `/a`), and containerd/CRI does not -- so the export was never
created and the server answered the `/a` path walk with NFS4ERR_NOENT (the first
run's `No such file or directory` on all six variants). The fix is in the
fixture's entry script, the image digest unchanged: `test -e /etc/mtab || ln -sfn
/proc/self/mounts /etc/mtab` before `ganesha.nfsd`. In the kind reproduction the
fixed Pod logs `Export 1 created at pseudo (/a) with path (/export/a)`, a client
mount with the probe's options succeeds (`statvfs` 256 MiB, a root write lands as
65534:65534) while the unfixed Pod still answers `reason given by server: No such
file or directory`. A failed `wait_ready` now journals the server's log tail and
reason into `export-<suffix>.json` (`export_failure`, `log`), and the error carries
the FSAL/EXPORT/CONFIG CRIT/MAJ lines logged within eight lines before the marker
(`preceded by: ...`), never the DBUS start-up noise.

The third live run (both lanes' `stat_timeout`, then every 005 variant) mounted
the export and read it healthy, yet ended INCONCLUSIVE with `the product never
delivered a healthy ... reading ...: the test export was not discovered as a
shared mount`. The cause was the verdict's read, not the product: `await_baseline`
took the baseline from stored raw batches only, while a newly appearing healthy
shared mount is no edge for the host collector (`_edge_reasons` knows threshold
crossings, collection errors, the collector's own first tick and a cleared
reason), so its first readings travel with the next health summary -- a batch
`app/ingest/telemetry.py` by design never writes as raw evidence (only the
`health-summary` reason, no finding, no collection error). What that batch does
move is the `metric_latest` record of every `LATEST_METRICS_REQUIRED` sample, both
shared metrics among them: in a quiet window the product's only per-mount
witness. The fix reads the baseline from evidence batch samples plus the latest
records that moved inside the window (`baseline_readings`, the rule 002's
`memory_used_percent` baseline already follows), reads the `shared_failure_domain`
`source` label from them too, and keeps the budget at one summary period plus
margin. On timeout the error now tells three cases apart and saves the last read
as `baseline-<node>-last-read.json`: no `host-` HOST_TELEMETRY delivery inside the
read (a silent collector), a mount read but never healthy (its readings listed),
or a delivering collector whose shared latest records name other mounts only --
the real "not discovered", the evidence of a sandbox view or a filtered root.

The first live run in which 005 `capacity_full` reached its fill (plain EKS,
release R13g) ended INCONCLUSIVE on `harness fixture failed: HostProbeError:
host probe failed (exit 0); output withheld` -- no fill evidence, no
`probe-error.json`, no token -- while the cleanup's unfill removed one file from
an export that read 0 % used afterwards. The cause was the probe's result shape,
not the fill: the fill snippet printed an `error` key on every run (`null` on
success, `OSError: 28` on a failure), `nfs_fill` spread that reading into its
result, and `host_probe_fixture.execute` classifies a result by the *presence*
of an `error` key, then shows only whitelisted tokens; `null` matched none. The
fixture keeps presence as the signal -- a probe's failure line whose error is
null or empty must never be graded as a success payload, and the other probes'
failure lines are unchanged -- and now says `error key null` for that case. The
probe side changed: the fill and unfill snippets nest their records under
`fill` / `unfill` beside the top-level `statvfs` (no snippet field can collide
with the fixture's vocabulary), catch every `OSError` and name the first one by
class, `errno` and `enospc` inside that record, and a fill the export stopped
short (ENOSPC before the target, a quota) is a reading the runner grades --
`VariantInconclusive("the test export only reached 61.2 % used (fill wrote 160
of 251 MiB; OSError errno 28 ENOSPC)")` -- not a failed call. What is a failed
call -- the fill's subprocess bound (`timed_out`), a snippet that printed no
JSON, an unfill that could not remove the fill file -- raises `ProbeError` with
a reason token (`fill_timeout`, `fill_unreadable`, `unfill_timeout`,
`unfill_unreadable`, `unfill_failed`) the fixture shows and records with the
message and the reading in `probe-error.json`. Tests:
`tests/regional/test_general_live_shared_probe.py` runs the snippets for real
(a `/dev/full` fill for the ENOSPC reading, a fill root that is a file, a fill
file that is a directory) and the reason tokens through `main`;
`tests/regional/test_host_probe_error_key.py` pins the fixture's classification
of null, empty, class-shaped and sentence-shaped `error` values, the probe's
failure line, and passes the real fill and unfill readings through the real
fixture; `tests/regional/test_general_live_shared.py` grades the short fill and
the fill without a statvfs reading.

The first live run of `healthy_other_mount` (plain EKS, release R13g) ended FAIL
on two harness errors about a right product. (1) `bounded_delivery: only 1 host
batch(es) were accepted after the injection`: the export was paused at 14:18:31Z,
the finding accepted at 14:18:51Z, the export resumed at 14:19:03Z, and the
oracle counted the host batches between the pause and the finding, wanting two.
The host collector ships a held breach once, as its crossing's edge
(`collect_once`: `delivery_reasons = reasons - _active_edge_reasons`), then only
on its health summary, so a product that detects within 20 s leaves exactly one
batch in that window and a stalled collector looks the same. `stat_timeout` and
`healthy_other_mount` (the two variants that grade the claim) now keep the export
paused until a second host batch after the pause is in the store -- the summary,
kept as evidence because the finding rides along -- for at most one summary
period plus the interval and the margin (`sustained_detection_budget(ctx, 0,
summaries=1)`, 495 s on the site's 300 s / 15 s cadence; the polls are saved as
`hang-delivery-timeline.json`), resume after it, and grade the delivery, the
chain and the other mount on that last read under the hang, which no post-resume
batch can reach; a collector the hang stalled runs the budget out and fails the
stage. (2) `cleanup did not prove the node clean: mount-other@...: ... left
residue: {'deadman_timers': ['...-ndeadman-nfs-shared.timer'],
'modules_loaded_by_run': ['nfsv4'], 'mount_points': ['.../shared'], 'mounts':
['.../shared']}` while the next entry, `mount-shared@...`, reported clean: the
LIFO stack unmounts the second mount first and the probe's residue check was
run-scoped, so a two-mount run always blamed the first cleanup for its
still-mounted sibling, the sibling's deadman timer and the run's module. `cleanup
--kind nfs-<name>` now answers for its own scope -- the mount (any NFS mount at or
below its mount point), its deadman timer, its mount point directory -- and the
run-level undo (unloading the module the run loaded; removing the run's root
directory `/mnt/gfrs-live/<digest>`, which the fixture never removed before, so a
node kept one empty directory per run) belongs to the run's last cleanup:
`--kind all`, or the per-mount cleanup after which the state records no other
mount (the first-mounted share's on the stack, or a deadman firing last). The
receipt says which it was (`scope.mounts`, `scope.run_level`,
`scope.remaining_mounts`) and what a non-last cleanup left to the last one
(`deferred`); a run-level residue adds `modules_loaded_by_run` and `run_dir`,
and the runner's error names the scope. The delivery hold and the recovery reads
moved to `general_live_shared_recovery.py` (the variants module sits at the
1500-line cap). Tests: `tests/regional/test_general_live_shared_recovery.py` (new; the
variants module's own test file is at the 1500-line cap as well, and now
reads `COMPANION_METRIC` from the recovery module) holds the
hang over a scripted edge-only product (two polls, the resume after the second
batch), runs a stalled collector out of the budget, and counts the window between
the pause and the resume; `tests/regional/test_general_live_shared_probe.py` runs
a two-mount LIFO cleanup without false residue, a per-mount leftover (an NFS
mount below the mount point) still reported, the run-level leftovers (a module
that would not unload, a run directory that is not empty) reported by the last
cleanup alone, and its ghost-kind cleanup now asserts the per-mount scope instead
of the run-scoped `clean=false`.

The same day `mount_disappears` (plain EKS, release R13g) ended FAIL on
`recovery: shared_filesystem_unavailable on ... did not recover: no fresh 0
reading with a cleared health signal, nor two fresh readings` after 34 polls over
twelve minutes -- and the product was right. The umount receipt is stamped
14:33:25Z, the batch of 14:33:28Z carried `shared_filesystem_unavailable=1`, the
batch of 14:33:43Z carried the recovery (`0` and a used %), and the remount
receipt is stamped 14:33:45Z: the collector's tick landed between the mount
syscall and the probe's answer, the variant took the recovery's `since` after
that answer, and the only 0 evidence the product ever writes predated it (the
recovery is one `recovered` edge, then summaries that move only the latest-metric
record); the health signal cleared at 14:42:54Z, so `cleared` held with `fresh`
empty. Every undo now stamps its `since` first (the remount; the SIGCONT of
`stat_timeout`, `healthy_other_mount` and `shared_failure_domain`; the chmod back
of `io_only_failure`; the unfill of `capacity_full`; `recovery` already did), and
`await_clear` / `await_used_recovery` accept, besides two fresh evidence readings,
one reading observed at or after `since` -- an evidence batch's or the
latest-metric record's, which every accepted batch moves, summaries included --
together with the health signal cleared at or after `since`, so a tick that beats
the stamp by clock skew is still proven by the next summary. Tests:
`tests/regional/test_general_live_shared_recovery.py` replays the chronology against
`await_clear` (1.0 at T, 0.0 at T+15 s, `since` T+17 s, the signal cleared at
T+9 min with the latest record it moved: accepted; without the record the budget
runs out) and drives five variants' `utc_now` so that the recovered edge at
T+45 s is fresh only against a stamp taken before the undo.

The hold's own count then failed a right product on HyperPod
(`healthy_other_mount`, 2026-10-26 19:46-19:55Z, release 37db33f254a1) and on
plain EKS (`stat_timeout`, 15:30Z): `bounded_delivery: only 1 host batch(es)
were accepted after the injection ... the collector's next scheduled delivery
never arrived while the mount hung` after the full 495 s. The collector was
ticking -- the product-side audit is lane b200's hung-mount summary-cadence
test of the host collector (on its branch until it lands: only the
`gpu-fault-mount-probe` worker parks on the hung mount; two summaries go out
under the hang) -- and the oracle counted the wrong record. It counted
raw-evidence batches (`list_raw_evidence`), and the summary that re-reports an
already claimed `shared_filesystem_unavailable=1` is never one:
`app/ingest/telemetry.py` `_persist_host_telemetry` keeps a batch as evidence
only with a finding, a collection error or a reason other than
`health-summary`, and `evaluate_metrics` mints a finding only on a claim
transition. The run's own records show it: the pause at 19:46:30Z, the
`threshold:` edge batch at 19:46:43Z, then no batch until the `recovered` edge
at 19:55:32Z after the 19:55:24Z SIGCONT, while the node's summaries fall on
its stable phase grid (`stable_phase_seconds` of the `host-telemetry` channel,
75 s on that node: 19:41:15 / 19:46:15 / 19:51:15 / 19:56:15, jitter +-15 s)
and the 19:51:17Z summary moved the HOST_TELEMETRY collector status
(`observed_at`, `batch_id`) and the latest-metric rows -- the hung mount's
`shared_filesystem_unavailable=1` `PROBE_TIMEOUT` reading with a new stamp,
`memory_used_percent` too -- as the baseline read before the pause had already
shown (`batches=[]` since 19:41:04Z, status and latest rows at the 19:46:12Z
summary of 105 samples). `hold_hang_until_delivered` and `delivery_errors` now
count with `deliveries`: the records the store keeps of one accepted batch,
folded by the batch's `observed_at` (the status spells it `...Z`, the latest
rows `...+00:00`; the fold compares moments) -- a raw-evidence batch, the
node's HOST_TELEMETRY status (`host-` batch ids only; the `k8s-efa-` producer
shares the channel) advanced to a stamp after the pause, or the latest-metric
row of the hung mount's `shared_filesystem_unavailable` / of
`memory_used_percent` moved there -- and want two of them, `memory_used_percent`
known to be carried by one (an evidence sample or a moved latest row). The two
latest-only records witness a delivery only while it is the newest, so the hold
reads them under the hang; the stages are graded on that last read, and a bound
`until` (the resume) excludes a later status. `other_still_reported` likewise
accepts the healthy mount's latest 0 record (`baseline_readings`). The budget
stays `sustained_detection_budget(ctx, 0, summaries=1)` on the survey's
summary/interval seconds (495 s on 300 s / 15 s): because the summaries sit on
the phase grid, the next one is up to interval + jitter + tick (330 s) after the
pause, not one interval after the edge; the hold writes the deliveries it saw
and the node's computed grid (`summary_schedule`: channel, phase, jitter, the
due phases inside the budget) as `hang-delivery-deliveries.json`, and a stalled
collector -- status and rows pinned at the edge -- still runs the budget out and
fails the stage, the error now naming the records read and the last batch they
witness. Tests: `tests/regional/test_general_live_shared_recovery.py` replays
the HyperPod chronology over both variants with the status advancing, the latest
rows moving, or both (one hold poll with the edge alone, the second accepted,
the resume after it), runs the pinned records out of the budget (25 polls, the
same sentence plus the witness), folds the three record spellings, bounds the
window at the resume, drops `k8s-efa-` and other channels' statuses, ignores
latest rows in a document without `since`, reads the healthy mount's latest
record, and pins the live node's phase grid (75 s; one due phase, 19:51:15,
inside the budget after a 19:46:28 pause).

004 `probe_error` has its faulting mount: the storage probe's loop +
device-mapper device carrying FAT16 (`dm-create --fstype vfat`; the FAT region
is read from the boot sector's BPB at build time, page-aligned inwards, and
recorded as `fat_sectors`). `dm-retarget --target error` creates a SECOND
mapping over the same loop device, `<dm>-fault`, whose three-segment table
has only the FAT segment on the `error` target built into `dm_mod` (boot
sector, root directory and data stay `linear`), proves the table with O_DIRECT
reads before anything mounts it (boot sector readable, first FAT sector EIO --
otherwise the fixture raises), and mounts it over the declared mount point, on
top of the healthy mount: the mount stays, reads read-write in the host table,
and every `statvfs` fails EIO. The mechanism: FAT counts its free clusters
from the FAT on the device until one count succeeds and then answers from
memory, and the kernel reuses a superblock per block device (`sget`). Both
live attempts of 2026-10-25/26 (the whole-device `error` table; the FAT-segment
table with `blockdev --flushbufs` and a remount) unmounted and remounted the
SAME mapping -- but the window probe Pod, created right after the test device
was mounted with its hostPath `/` on the default private propagation, held a
private copy of that mount, so the host unmount never ended the superblock,
the remount reused it with its cached count, statvfs kept answering, and the
mapping kept exactly one opener (the cleanup's `dmsetup remove` failed `Device
or resource busy` on both clusters). A new mapping is a new block device, so
its mount gets a new superblock whose first count must read the FAT through
the fault mapping: EIO, independent of the page cache and of whoever pins the
old superblock (reproduced here with `unshare -m --propagation private`: three
rounds of three statvfs, all EIO, while the pin was alive). `--target data`
reloads the fault mapping's table to `linear` in place (the next count
succeeds, no remount). The probe's own reading is the oracle: no EIO on its
one attempt is `INCONCLUSIVE`, and the runner now writes `injection.json`
with the reading, the `direct_reads` and a `diagnosis` ("statfs answered over
a mapping whose FAT reads fail: superblock reused" when the mount point
resolves to the fault mapping, otherwise "the mount point does not resolve to
the fault mapping"). The error segment is neither written nor page-read from
user space -- a failed write would log `Buffer I/O error`, which the node log
collector's storage rule grades QUARANTINE; the kernel only logs `FAT-fs: FAT
read failed` lines, which that rule does not match -- and the variant still
refuses to run while that collector is active. The destroy unmounts the whole
stack on the mount point, finds every other mount namespace that still holds
a copy of either mapping (`/proc/<pid>/mountinfo` by major:minor), lazily
unmounts it there with `nsenter --mount` and records the holders (pid, comm,
`locked`, `evicted`) in the cleanup result, waits (bounded, 90 s) for the
open count to reach 0 and removes; a copy owned by another user namespace is
locked and cannot be unmounted from outside, so that mapping is removed
`--deferred` (it goes when the holder exits) and reported as residue instead
of raising. The read-only probe Pods' hostPath `/` now uses `HostToContainer`
(rslave) so the host's mounts flow in and out of them and a probe started
mid-case can no longer pin a superblock. It needs `KERNEL_MODULE_LOAD` for a
missing `vfat` and `mkfs.vfat` on the node; without the tool it is
NOT_RUN(CAPABILITY_ABSENT) unless the optional `TEMP_TEST_PACKAGE_INSTALL` is
given, which installs exactly `dosfstools` for the run before the device is
built and removes it as the last LIFO cleanup after the device is destroyed.

`healthy_other_mount` on HyperPod (2026-10-26 19:46-19:55Z, release
37db33f254a1) failed `bounded_delivery` with the oracle that holds the export
paused until the store shows a second host batch: none appeared in 495 s, read
as "the collector's next scheduled delivery never arrived while the mount
hung". That is not a collector stall but what the oracle counts. The API keeps
a host batch as raw evidence only when it carries findings, collection errors
or a reason other than `health-summary` (`app/ingest/telemetry.py`,
`_persist_host_telemetry`), and `evaluate_metrics` emits a finding only on a
health-signal claim transition, so a summary that re-reports an already
claimed `shared_filesystem_unavailable=1` is never evidence and
`list_raw_evidence` -- the activity probe's `batches` -- cannot count it; the
run's own baseline read shows the rule (no batch since 19:41:04 while the
HOST_TELEMETRY collector status and the latest-metric rows named the 19:46:12
summary of 105 samples). The chronology is a ticking collector on a 15 s
interval: this node's summaries fall on 19:41:15 / 19:46:15 / 19:51:15 /
19:56:15 (`stable_phase_seconds` 75, jitter +-15 s); the detection edge went
out at 19:46:43; the 19:51 summary moved only the collector status and the
latest-metric rows; the `recovered` edge at 19:55:32 carried no
`health-summary` reason (its phase had been served) and the collector status
read 19:56:18 afterwards. A tick blocked until the 19:55:24 SIGCONT could not
have been followed by a tick starting at 19:55:32. The collector side is
pinned by `tests/collectors/test_host_hung_mount_summary_cadence.py`: every
`os`/`io` path call under a hung mount is trapped and only the
`gpu-fault-mount-probe` worker may park there (one worker for the whole hang);
two health summaries go out under the hang with the neighbour and the
configured filesystem still read, and a bare `stat` of the mount from the
tick's thread fails the test by name. What proves a summary during a hang is
the HOST_TELEMETRY collector status (`observed_at`, `batch_id`) or the
latest-metric record of the hung mount's `shared_filesystem_unavailable`
moving to a later `observed_at` at value 1 -- both latest-only, so the hold
has to read them while the export is paused; raw-evidence `batches` can only
ever hold the detection edge. Until the oracle reads those records,
`healthy_other_mount` and `stat_timeout` fail on a right product.

## GF-GENERAL-LIVE 029 store-level restart-budget variants

029 `concurrent_claim` and `missing_context` run no managed job. The harness
(`scripts/e2e/regional/general_live_restart_budget.py`) ships a probe into an
API Pod under the product's interpreter and drives the product's own ledger API
(`reserve_job_restart`/`release_job_restart`/`get_restart_budget`) and claim
preflight (`reserve_restart_budgets`, `issue_restart_authorization`) on a test
job id (`gl29…`) that no training job carries; the incident and workflow the
preflight is handed exist in memory only and are never saved. What this proves
is the ledger layer against the live store -- the same-reservation idempotency,
the over-budget refusal, the exactly-once release and the context check before
any budget read, serialised by the Postgres advisory lock the product uses --
and not two real executor replicas racing a claimed workflow through the
dispatcher (a watcher restart inside the restart transaction is 029
`worker_restart`, below). The
product offers no way to delete a ledger
row, so the cleanup releases this run's reservations through the API and then
removes the emptied test row with the store's `_delete` primitive (the one
`cleanup_audit_records` uses for the command-protocol audit's `restart_budget`
rows); a row that still holds a reservation this run did not make is left and
fails the cleanup. The module tests run the real probe in-process over an
`InMemoryStore` with the three key/value shims the SQL stores have, so the
advisory-lock path itself is exercised only live.

## GF-GENERAL-LIVE 029 hardware chain under an exhausted budget

029 `hardware_chain_continues` (`scripts/e2e/regional/general_live_attempt_hardware.py`;
`--approve GPU_RESET --approve KERNEL_SOURCE_REPLAY`, list flag `X`, both lanes)
is the one GENERAL-LIVE variant that drives a fatal-grade kernel XID live. What
it proves, and what it does not:

- The fault is a replay. One reviewed `NVRM: Xid (PCI:<bdf>): 46, …` line
  (`ROBUST_CHANNEL_GPU_TIMEOUT_ERROR`, pinned catalog
  `src/gpu_fault/data/nvidia-xid-catalog-610.generated.yaml` lines 462-473,
  `immediateAction: RESET_GPU`, H100 family; the code GF-REGIONAL-DESTR-001
  proved the reset chain with on this hardware) enters through the kernel
  collector's kmsg source, so the evidence says `proof_scope: live-replay` and
  nothing physically failed. The actions are the product's own node actions --
  cordon, stop of the managed job, quiesce, one reset of one H200, restore,
  validate, uncordon -- read as SUCCEEDED step receipts. XID 46 alone is
  exercised: nothing is said about the other RESET_GPU codes (62, 109, 110,
  119/120, 136, 140, 143, 156, 158), about XID 95's pre-actions or about the
  XID 154 dynamic recovery actions.
- The exhausted budget is seeded, not consumed. `ExhaustedRestartBudget`
  (built on the store-level variants' `SeededRestartBudget`, same `gl29hc-…`
  test job id and cleanup) reserves `restart_count == budget` through
  `store.reserve_job_restart` -- the call the executor's claim preflight makes
  -- and records the product's refusal of one more reservation. The managed
  training job is created first under that id with `restart_budget` 1 (the
  runner's managed-job contract: its `kubectl create` is the variant's first
  cluster mutation and no store record is read before it; the ledger cleanup
  is armed ahead of the job so the LIFO stack releases it last), the seed
  follows once the job is RUNNING. The product cannot tell the seed from a
  consumed budget, but the spec's
  "consume the budget through legitimate failures" sentence is proved by
  `budget_consumed` / 028 `one_restart`, not here.
- The contrast with GF-REGIONAL-ISO-007 (software XID 11, budget 0, never run
  live) is asserted explicitly: both end FAILED on `RESTART_BUDGET_EXHAUSTED`
  with that one FAILED step and no remote restart command; this variant
  additionally requires RESET_GPU to have executed, every planned repair step
  SUCCEEDED, the node uncordoned by the product's RESTORE_SCHEDULING and the
  incident ESCALATED (QUARANTINED, or an ESCALATED incident on a still
  isolated node, is a FAIL and the node is handed back through the validated
  restore). It proves nothing about a second physical cluster or ISO-007's
  cross-cluster independence.
- The attempt is graded on the completion records: terminal STOPPED with the
  hardware incident as initiator, decision NO_ACTION, no passive
  `TrainingAttemptFailureDetected` containment, the ledger still the seeded
  1/1 (a withheld restart reserves nothing), exactly one
  `restart-budget-exhausted` notification for the incident -- the one the
  withheld record names. GPU health is fresh accepted GPU_METRICS after the
  RESET_GPU receipt with no active finding; a reset that released the idle
  display stack must report it restored (`display_stack_restored`), and the
  product's busy-reset retries are recorded (`gpu_reset_busy_attempts`), not
  judged.
- Cleanup is LIFO -- the kmsg window (closed as soon as the decision exists;
  its `systemd-run --on-active` deadman restores the collector if the runner
  dies), the test job and its prewarm Pods, then the seeded reservations and
  the emptied test ledger row (a row that is not provably gone raises and
  fails the variant). The job and the ledger row are cluster/control-plane
  test objects with no deadman of their own: a run that dies leaves a
  `gl29hc-…` row that withholds restarts of that test job only, and the next
  attempt refuses it (`NOT_RUN(FIXTURE_NOT_READY)`) while its cleanup removes
  it -- the same residue rule as the store-level variants.
- Live prerequisites: an idle target node (no GPU Pods; the variant creates
  the job itself), `--site-file`, a maintenance window (the node is cordoned
  for the duration of the chain, hence flag `X`); HyperPod additionally
  `NodeRecovery=None`. The module tests run the seed and notification probes
  in-process over an `InMemoryStore` and the whole variant over scripted
  product records; the Postgres advisory lock, the kernel collector's reopen
  of the fixture file and the reset itself are exercised only live.

## GF-GENERAL-LIVE watcher isolation and restart (028 api_unreachable, 011 unreachable_node, 029 worker_restart)

Three dangerous native variants cut or replace a product component instead of
a node resource, each behind its own `--approve` token and each graded on the
product's records only (`scripts/e2e/regional/general_live_network_isolation.py`,
`general_live_watcher_restart.py`, node probe
`probes/general_live_isolation_probe.py`).

028 `api_unreachable` (`WATCHER_NETWORK_ISOLATION`) rejects, on the node that
runs the one completion-watcher Pod, TCP from that Pod's IP to the control-plane
endpoint (FORWARD rules, tcp-reset, the run's tag in the comment match) while a
managed job exits non-zero on the runner's trigger file. The scope is the Pod
IP: no other Pod, host unit or destination is affected, and the systemd deadman
(a copy of the probe running `cleanup --strict`) fires within 540 s, under the
600 s after which the control plane reads the cluster's coverage UNKNOWN. What
is proven: the watcher's emergency fallback suspends the workload itself
(annotated `emergency-fallback`, naming the containment incident), nothing
reaches the control plane while the block holds (the coverage heartbeat and the
job's workload observations stop advancing and the attempt has no
event/decision -- otherwise INCONCLUSIVE), and after the restore every action
is recorded once: one containment with one successful STOP_WORKLOADS, one
decision (PLAN_CREATED: a STOPPED terminal initiated by the passive containment
is still the failure's terminal and restarts once within budget), at most one
recovery workflow with at most one RESTART_WORKLOAD, a ledger whose count
equals its reservations, no notification twice, and a write of the watcher's
own after the restore. Two readings the first HyperPod run (2026-10-27) got
wrong and the oracle now makes: the recovery history is what the terminal
decision opened -- the store's job/attempt listing
(`list_job_recovery_workflow_incidents`, terminal workflows included) returns
the passive containment's own STOP workflow beside it, which the attempt probe
reports under `containment_workflows` only, so one stop and one restart are
not "2 recovery workflows"; and the watcher's return is its heartbeat *or* the
job's latest workload observation, because a coverage heartbeat is the idle
statement only (`publish_coverage_heartbeat` says nothing from a pass that saw
a running Pod or a live attempt) and the restart the variant expects keeps the
cluster busy until the cleanup -- that run waited 300 s on a heartbeat row that
could not move while the restart copy's RUNNING observations and the attempt's
terminal had all been delivered. Limits: the block holds the resolved IPv4 addresses of the
endpoint at block time (a CIDR list can be given instead); a DNS change during
the window is reported by the probe's `status` and would show up as an
ineffective block (INCONCLUSIVE), never as a product verdict. The REJECT makes
the watcher fail fast; a silent DROP (hung POSTs until their timeout) is not
exercised.

011 `unreachable_node` (`NODE_AGENT_NETWORK_ISOLATION`) rejects TCP to the Node
Agent's listening port on `--node-b` -- the channel the executor pushes node
actions to -- from before the stall until the hung chain's summary/triage
executions are recorded, then restores it so the chain's later steps run
against a reachable node. The Agent's outbound heartbeats, the collectors and
kubelet are untouched, so the control plane still believes the node is covered
and has to record the gap itself: the summary/triage details must name the
node as undetermined/failed/not responding and carry no reading for it, the
reachable node's evidence must be present, the incident must still end
ESCALATED and nothing may be cordoned, stopped or reset. The restore proof is
the probe's own connect to the Agent port plus the Agent's `last_seen_at`
advancing in the control plane. Not exercised: an Agent that is down (its
heartbeat lost), a node whose host collector is also unreachable, and the
10-minute waiting cap a node action would hit if the channel stayed cut.

029 `worker_restart` (`CONTROL_PLANE_RESTART`) deletes the watcher Pod fenced on
its UID/resourceVersion, with no grace, inside the restart transaction of the
managed job, timed from the source failure rather than from an "open claim".
The reservation on the ledger is open for one control-loop round trip only: the
executor reserves the restart in the tick that claims the recovery workflow and
mints the data-plane command in the same tick (`regional.py:945-1020`), the
cluster executor is woken by the command's NOTIFY, creates the restart copy and
returns SUCCEEDED in the same call (`adapters/kubernetes/workload_operations
._set_workloads`, 156-235: no Pod of the copy is read, no wait for scheduling
or Running), and the command's terminal status wakes the dispatcher
(`execution/dispatcher_wakeups.WAKING_REMOTE_COMMAND_STATUSES`), which records
the step SUCCEEDED. Live it lasts one to a few seconds (HyperPod, 2026-10-27:
open and closed between two records reads 10 s apart), shorter than one
records read (a `kubectl exec` probe, about 5 s), so no poll can gate on it and
no start barrier on the restart copy's Pod can widen it. The transaction as a
whole -- failure, publish, containment, terminal, decision, recovery, settle --
is tens of seconds, and its start is visible without the store: the runner
reads the source Pod and its PyTorchJob through the GPU-cluster API every
second (every five while it still starts; one labelled `get pod,pytorchjob` --
both carry the `gpu-fault.io/job-id`/`attempt-id` labels the product's submit
path stamps on the templates and the object; `wait_source_failure`, no store
read) and takes the source's end from whichever of three signals it observes
first: a terminated critical container or a Failed phase (`pod-terminated`), a
`Failed` (or `Succeeded`) condition on the PyTorchJob (`job-ended`, the
training operator's `PyTorchJobFailed`, which outlives the Pod and also covers
a Pod that fails before it was ever seen Running), or a Pod no longer listed
after it was seen Running (`pod-absent`: the containment's STOP_WORKLOADS
deletes the attempt's Pods with no grace within a second of the publish,
`workload_operations._delete_terminating_pods`; live on HyperPod, 2026-10-27
12:40, the source went from Running straight to absent between two one-second
reads and no terminated state was ever listed, so the previous watch, which
knew only the terminated state, waited the full 900 s). Every change on either
side is written with its source to `source-failure-timeline.json`; none of the
three within 900 s is a fixture failure (INCONCLUSIVE) naming the last Pod
state and the last job condition. The watcher Pod is deleted at that
observation (`DELETE_OFFSET_SECONDS` = 0). The job runs `exit_once:1` rather
than `exit:1`: the `exit:1` copy failed again 30 s after its start and the
product's second containment deleted its Pod, a race against the replacement
watcher's cold start that `attempt_once` could lose; the `exit_once` copy runs
`COPY_RUN_SECONDS` = 240 s (`GFRS_COPY_RUN_SECONDS`) and exits 0 -- no second
failure, escalation or notification on the job, the copy's Pod stays
(cleanPodPolicy None) for the replacement to observe, and its terminal reading
is a watcher write after the replacement is Ready, inside the 300 s
watcher-restored wait. The delete therefore lands inside
the transaction by construction: before the watcher published the failure, and
the replacement must publish it (from its write-ahead outbox or its own
observation) and still yield exactly one decision, one reservation and one
restart; or after it, and the replacement's re-observation must not
re-publish, re-decide or re-reserve -- both are the product's claims. The
replacement is then awaited (a new Ready UID within 600 s, the startupProbe
window) and the recovery's settle under a bound derived from the product's
constants (`settle_bound`: the 600 s replacement window + one step cap each
for STOP_WORKLOADS and RESTART_WORKLOAD, 600 s with no override for either
(`execution/config.py:293`, `:175-182`) + two fallback dispatcher polls of 5 s
(`:522`) + one watcher poll of 5 s (`completion_controller.py:146`) + a 120 s
margin = 1935 s from the delete). Where the delete landed is dated afterwards
from the records (`transaction_timeline`: the containment incident's
`created_at` is the publish, the recovery incident's `created_at` the
decision, the recovery workflow's first CLAIM event the reservation, the
terminal RESTART_WORKLOAD record's and the workflow's terminal `updated_at` the
restart's end and the settle) and the delete request's bracket is classified
BEFORE_PUBLISH / BETWEEN_PUBLISH_AND_RESTART / AFTER_RESTART_BEFORE_SETTLE /
AFTER_SETTLE (`classify_delete`; `delete-timing.json`, and `delete_landed` /
`delete_timing` in the result). Only AFTER_SETTLE, a fenced delete that did not
happen and a replacement that never became Ready are INCONCLUSIVE. The
replacement restores its state from the product's own ConfigMaps and must not
re-decide, re-reserve or re-send: the ledger keeps exactly the one reservation
of the recovery workflow (not lost, not doubled), one RESTART_WORKLOAD succeeds,
one STOP_WORKLOADS, one decision, exactly one restart attempt is observed
besides the source (by attempt id, whichever watcher instance published it), no
notification is recorded twice and a write from the replacement follows (its
coverage heartbeat once the cluster is idle, or its observation of the restart
attempt). The module tests drive the sequence over fakes (a Pod inventory that
fails the source and flips the watcher on the fenced delete, a store whose
records date the transaction around the delete); the live run is the only proof
of the real outbox replay.

Both watcher variants start their managed job first -- every managed-job
variant's first cluster mutation is the job pinned to `--fault-node` (the
prewarm Pod or the PyTorchJob) -- and discover the completion-watcher Pod and
the control-plane endpoint afterwards. A watcher that is then not a Ready
singleton (none, two during a rollout, hostNetwork) is a fixture failure --
INCONCLUSIVE, the job cleaned up -- rather than a NOT_RUN, because a mutation
has already happened; the read-only `--plan` does not check the watcher.

## GF-GENERAL-LIVE live acceptance runner (phase 2)

Phase 2 adds the workload- and spare-centred categories through
`general_live_nccl` (011 EFA stall / NCCL hung), `general_live_progress` (027
training progress) and `general_live_spares` (030 warm spare unhealthy), with
fakes-based module tests under `tests/regional/test_general_live_{nccl,progress,
spares}.py` and an `efa-traffic` read-only command on the node probe. These are
still plan-only by default and change no cluster state until an operator runs
`--execute`; the module tests exercise only the injection/verdict logic against
scripted product records and are not proof of a physical experiment.

011 and 030 run on both lanes (011 needs `--node-b`, the lane's second EFA
node -- the other plain-EKS node, or the HyperPod profile's node-b: a dedicated
two-node NCCL job, rendered from the two-node replace fixture's Pod templates,
all-reduces 1.5 MiB every 2 s, sized to 1-4 MiB/s per node from the product's
1 MiB/s active threshold and 4x spike ratio, and on the runner's pause file one
rank enters an extra collective while the others hold before it until a release
file or a bounded in-Pod watchdog. The runner's own `/sys/class/infiniband`
counter reads must show the pair at or above the active threshold before the
stall and at or below the zero threshold for at least 600 s after it, on both
nodes and with the job alive, or the variant is INCONCLUSIVE; the source
fixture's own trainer moved about 74 KB per node over a whole live case, so a
FAIL against it measured an injection that never happened;
`repeat_after_recovery` also needs the pair active between its two stalls for
the product's rearm window; 030 declares its spare candidate --
`--spare-candidate`, by default the lane's `--node-b` -- as the warm spare
through the admin CLI under `DECLARE_WARM_SPARE` (label plus cordon, keyed to
the provider's failure-domain group: HyperPod's instance group or the EKS
managed node group, graded by the same `HyperPodSpareCoordinator` the plain-EKS
profile's `nodeReplace` warm-spare handover uses), holds every lock (`X`)
because on plain EKS that node is the other lane's own, proves it idle at plan
time and again before declaring -- Ready, schedulable, no gpu-fault
ownership/taints/spare marks, only infrastructure Pods, no GPU-requesting Pod,
an ACTIVE Agent with fresh channels, no open incident or workflow, and on EKS
the GPU node groups' `nodeRepairConfig` not enabled (the product's own join
preflight requires node auto repair off) -- and is INCONCLUSIVE otherwise,
never cordoning a busy node; it undoes its injection and releases the spare
before its verdict, recording the candidate's labels, taints, cordon and spare
annotations before and after, the release receipt and a cluster-wide
spare-label census, any of which failing the `restored` stage fails the
variant; `warm_spare_residue.py` is the same read-only census for the lane
runner's heal between cases (exit 3 on a leftover label). `not_ready` stops
kubelet through the DESTR-008 service window on both providers (the Agent
identity it pins is the EC2 instance id on eks-managed; the on-node restore
timer bounds the stop at 600 s), waits for the node to leave Ready, grades,
then restores kubelet and waits for the node and every Pod on it Ready before
the release; `stale_metrics` waits for the channels fresh again after the
resume so the other lane's next preflight does not read the pause as its own
stale collector). 027 enables the optional reporter only for its disposable test
job (a Pod-local `training-progress` sidecar) and triggers a single per-cluster
scan, never the global monitor. The flight-recorder forensics variants of 011
(`general_live_nccl_fr`: `fr_temporal_completion`, `fr_second_missing`,
`fr_whole_group`, `few_candidates`, `ambiguous_candidates`,
`different_process_groups`, `fr_plus_process`) are live-replay: the stall job
runs with PyTorch's recorder off and a reviewed test dump responder
(`probes/general_live_fr_responder.py`, started inside each `pytorch`
container before `torchrun`, bounded by its own `--max-seconds`, removing its
FIFOs and dumps when it ends) answers the Node Agent's own FIFO requests with
dumps of the variant's shape; the FIFO protocol, namespace mapping, parsing,
two-dump series, classification and bundle binding are the product's, so these
variants prove the processing chain and never PyTorch's recorder or an actual
NCCL fault. Two limits follow from the product's own rules: the hung DAG of
this site carries no `clock_uncertainty_seconds`, so a group spanning both
nodes is never time-comparable and the groups that yield leads or a
whole-group stall are node-local (eight ranks each; the cross-node default
group is present and its per-rank watermarks are still read); and a D-state
wait cannot be forced natively, so `fr_plus_process` keeps the write-mode
checkpoint I/O (`HOST_WRITES_OBSERVED`) as its independent process evidence.
On the first live runs these variants exposed a node-side contention gap, not
an oracle error: a node's single isolated observation worker slot is shared by
the 30-second passive communication sampler, and a triage or bundle that
arrived while the sample held it was refused `WORKER_STILL_RUNNING` on arrival
with no wait. Live, `fr_temporal_completion` (hp, 2026-10-26 04:21Z) landed
1.45 s into an epoch and both nodes refused the triage (the decision saw no
process, the oracle's `fr_paired`/`watermarks` read "sampled no ranked
process"); `fr_second_missing` (hp, 04:46Z) sampled every rank but its bundle
was refused on the shared node `hyperpod-i-08d8…` alone -- the node that also
hosts the other lane's prewarm/training Pods, so its sampler had more targets
holding the slot -- and the workflow failed and escalated. The fix makes a
control-plane command wait for the slot (inside the control plane's 30-second
per-node wait) and preempt the passive sample, makes a refusal after the whole
wait a retryable failure the survey re-sends, and makes a worker past its
deadline hand back its last partial checkpoint; the oracle is unchanged
(`fr_temporal_completion`/`fr_second_missing` stay expected PASS). The other
FR-replay variants run the same shared slot and the same budget: `fr_whole_group`
and `fr_plus_process` pair two dumps on eight ranks exactly as
`different_process_groups` does (measured ~4.5 s inside the 20 s budget) and
`fr_enabled`/`fr_disabled` take one slot likewise, so any of them can be
refused the same way if it lands on a busy slot -- their holds lift with this
node_runtime roll rather than needing a separate product change. Re-run
`fr_temporal_completion` and `fr_second_missing` on the hp lane (the eks lane
pre-claimed) once the GPU nodes' Agents carry this fix.
The role, thread, process and attempt-change
variants of 011 (`general_live_nccl_roles`, on both lanes like every 011
variant) are the same stall on the
trainer's role modes (`general_live_nccl_role_trainer`): each rank declares
PP/EP roles and container-native thread roles through its `rank-progress/v1`
`diagnostic_context`, a keeper thread keeps the snapshot inside the product's
30 s freshness window while the rank is held or blocked, helper threads exist
from the start and switch behaviour with the stall, `cpu` adds an unranked GPU
sharer process and `pyspy` a compiled non-Python CUDA context holder the Node
Agent's own py-spy legitimately fails on (no tool is shadowed; an image without
a C compiler makes the variant INCONCLUSIVE). The verdict grades the stored
triage rows and the `hung_triage_decision` the product bound the bundle step
with against the trainer's own `ROLES`/`SHARER`/`HOLDER` lines; a same-role
`ROLE_COHORT_*` lead is legitimate, a cross-role one is the defect.
`attempt_change` deletes the stalled attempt A once both nodes warned and
under the hung threshold, resubmits the job as attempt B
(`--attempt-id <job>-a002`) and reads each node's EFA traffic state row. The
module tests check these graders against the product's own classification of
synthetic triage rows, not against live evidence. Variants that need a
128-node/1024-process scale or the HyperPod spare-health controller (disabled
in this regional control plane, `GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER=false`)
stay `NOT_RUN` with an explicit reason in the denominator on both clusters'
boards rather than being dropped (every plan-only variant records its reason on
both lanes: a site-wide gap is a fact about each cluster). No GF-GENERAL-LIVE
case has live evidence yet.

Every variant's optional `preflight` hook takes `(settings, regional)` -- the
fixture is the second argument so a hook can read the cluster (the 030 spare
candidate) -- and `read_only_preflight` calls every hook that way; a framework
test pins the signature of each registered hook, after the 017 application-log
hooks still took one argument and five variants ended NO_EVIDENCE on
2026-10-26 with a `TypeError` before any evidence existed.

The 016/017 journal replays write a synthetic cursor that carries the run id
(`gfrs-replay-<run>-<index>-<boot>`). The cursor is the collector's entry id and
the control plane derives an incident's event id from (cluster, node, entry id),
so a cursor reused on the same node and boot merged a later run's finding into
the earlier run's incident -- created before the run began, hence unreadable by
the grade -- and 016 `kernel_lockup` / `rdma_fatal` and 017 `cgroup_oom` ended
"no severe kernel-log finding reached a NODE_HEALTH incident" on 2026-10-26
without a product defect. Their reviewed lines now follow the kernel's real
print formats (Linux v6.12): the `kernel/watchdog.c` soft-lockup line, the EFA
`efa_com.c` completion timeout through the `__ibdev_printk` prefix and the
`mm/oom_kill.c` "Memory cgroup out of memory: Killed process" line without a
`kernel:` tag, so the rules are exercised on what a node really logs.

016 `cursor_pagination` read FAIL on every live run -- HyperPod and plain EKS
on 2026-10-26 (R13g) and plain EKS on 2026-10-27 (R13h, after b190 changed the
collector's journalctl positioning) -- with "only 0/3 severe kernel records
reached an incident; the cursor starved the rest", while the product had done
what the variant exists to prove. On R13h the first poll after `arm` read the
three reviewed records (one wrapper call, one timestamp, three cursors) in one
batch. The first finding (the MCE record) minted `inc-log-<node>-<its event
id>` at 18:46:32Z and opened the node's isolation (FREEZE_EVIDENCE,
MARK_UNSCHEDULABLE, QUARANTINE); the storage and lockup findings joined that
isolation while it was in flight (`orchestration/families/drain.py`
`ingest_terminal_node_quarantine`): one record, re-labelled
`NODE_HEALTH_GROUP`, carrying the three findings' reasons and three markers,
and the node stayed quarantined until the runner's validated restore at
18:56:35Z. Both R13g runs left the same record (HyperPod 11:00:13Z with all
three reasons; plain EKS 05:13:07Z with two, the MCE record's cursor not yet
carrying the run id and so routed to an earlier run's incident). The grade
took only `NODE_HEALTH` incidents and wanted one per record, which the product
does not mint while the node's isolation is in flight, by design. b190 (journalctl positioned by its cursor
or by a time window, never both) fixed a real collector defect, but not this
one: the records were delivered on both releases. The variant now follows each
record through its own finding's marker -- the product writes one per
finding, naming the entry it was read from in `raw_evidence_ref`
(`dmesg://<node>/<replay cursor>`) and the incident ingestion routed it to --
and passes when every record's marker points at a node-log incident minted
since the injection, under any label the product gives it
(`NODE_HEALTH_EVENT_TYPES`); the isolation chain is graded on the incident the
burst minted first. The 016 incident matcher takes the same labels, and
`minted_in_read`, for every variant. The replay cannot place a page boundary
between same-timestamp entries (all three land in one read), so the
collector's resume across one stays proven by
`tests/collectors/test_node_log_journal_positioning.py`;
`tests/regional/test_general_live_kernel_burst.py` drives the variant's own
records through the probe's wrapper, the real collector, the control-plane API
and the activity probe, so the grade is tested against what the product writes
rather than a scripted record.

Every GF-GENERAL-LIVE matcher of node-health incidents now takes one label set,
`NODE_HEALTH_EVENT_TYPES` in `scripts/e2e/regional/general_live_verdicts.py`.
The product mints a node-health finding's incident as `NODE_HEALTH`
(`orchestration/families/health.py`) and re-labels that record when it routes
more evidence into it instead of opening another: `NODE_HEALTH_GROUP` when a
critical quarantine joins the node's in-flight isolation (`families/drain.py`
`ingest_terminal_node_quarantine`), `NODE_RESOURCE_GROUP` when a
device-resource finding joins it (`ingest_grouped_node_resource_finding`),
`GPU_FAULT_GROUP` for a finding of a managed attempt
(`families/grouped_health.py`); a reboot or replacement finding of a running
attempt opens `NODE_HEALTH_GROUP` itself (`families/node_lifecycle.py`, under a
derived `incident-<digest>` id and its own event id). 008 and 016 took the set
after live runs read FAIL over records the product had grouped; the 017
journal variants, 002 `cgroup_only`'s log grading and the DCGM and inventory
selectors of `general_live_gpu_verdicts` (019, 020, 021, 022, 024, 026) still
took `NODE_HEALTH` alone, so a grouped record was invisible to them: a positive
stage found no incident for a finding the record held (the DCGM chain's "no
... finding reached an incident"), and an evidence-only stage read PASS over a
node-log isolation opened inside its window. They take the set now. The event id still says which
finding minted a record, and a finding that joined another finding's record is
followed through its marker (`marker-<event id>`, whose `incident_id` names the
incident ingestion routed it to), as 016 `cursor_pagination` does. Checked
variant by variant, no other variant's own injection lets two of its findings
share a record (one finding, or none that claims the node exclusively or is a
critical quarantine); the one exception, 024 `wrong_scope_source`, would group
only if the exporter emitted the pending-repair reading before the
uncorrectable one, and the site's `deploy/dataplane/dcgm-counters.csv` lists
the volatile DBE first, so it keeps its per-GPU incidents.
`tests/regional/test_general_live_node_health_labels.py` drives the product's
own ingestion (kernel lines and DCGM scrapes through the control-plane API)
through every routing, pins the labels they give to the set exactly -- the
product's other record labels (the shared-storage failure domain, the placement
hold, operator maintenance, XID/SXID) stay out of it -- and grades each matcher
on a product-made record two findings share, which every one of them missed
under `NODE_HEALTH` alone.

017 `old_pod` writes its reviewed line under the UID of a Pod the product has
already restarted away, in a directory kubelet is going to remove: kubelet's
container GC (one-minute `ContainerGCPeriod`) deletes every `/var/log/pods`
directory whose Pod UID it no longer knows. The first live run (HyperPod,
2026-10-26) lost that race between two separate probe calls -- `create-log`
succeeded, `write` found the file gone six seconds later -- and read only
"host probe failed (exit 1); output withheld", because the probe's failure
line carried the class under `error` alone. The node's layout was not the
cause: the probe's `inspect` showed the same `/var/log/pods/<ns>_<pod>_<uid>/
<container>/0.log` and cgroup v2 layout as plain EKS. The fixture now seeds
directory, file and line in one probe call (`seed-log`) right after the GC tick
it watches for on the old Pod's own directory (at most 75 s), repeats a round
the GC still takes (the probe's `fixture_gone` / `directory_removed` reason
tokens, read back as `HostProbeError.reason`) and fails on any other reason by
name; every refusal of this probe carries a token from a closed vocabulary and
its sentence lands in the case's `probe-error.json`. Before a window opens, the
probe's read-only `preflight` must show kubelet's root, a kubelet-shaped Pod
directory under it and the installed collector unit, or the variant is
`NOT_RUN(CAPABILITY_ABSENT)` naming the gap. Neither cluster has an `old_pod`
PASS yet; the rerun needs nothing prepared on the node.

017 `cri_partial_rotation` failed its first live run (HyperPod, 2026-10-26)
only on the two stages that expected "one NO_ACTION evidence incident
attributed to the attempt"; the stitched record itself was delivered from the
rotated inode with the attempt's Pod UID/container and the P fragment's time,
so the collector's CRI discovery and stitching work on HyperPod and the batch
carrying the line was accepted (b190's journal refusals did not block the CRI
tail). The expectation was wrong: `app/ingest/node_health.py` keeps a
`SITE_APPLICATION_LOG` NO_ACTION finding as the NODE_LOGS record only -- no
incident, marker or workflow ("log evidence is already durably captured by the
ingest transaction; workload errors are not hardware markers or a second
recovery owner"). The verdict now reads the record: no incident for it
(`evidence_only`; one minted anyway is the failure) and its Pod UID/container
is a live container of the attempt in the completion watcher's observation
(`attributable`), with no incident wait. The same fact means the journal
variants `nccl_error` / `distributed_error` wait their 420 s incident budget
out by design and pass only on the absence of adverse handling; reading the
product's delivery of a journal line needs a store probe extension (open).

011 `fr_enabled` (the Flight Recorder configured as the operations manual
prescribes: trace buffer, pipe, dump-file prefix and dump-on-timeout; at least
one PID must show two real dumps as `PAIRED`) is registered as an expected FAIL
naming a product gap. The first gap -- the Node Agent reader knew only the
legacy PyTorch variable names and default dump location -- was closed by batch
182 and proven live. The gap registered now is `FR_RESULT_BUDGET_GAP`: on the
2026-10-25 HyperPod run the triage found and dumped every rank twice, but the
triage result of 8 ranks x 2 dumps (about 1.2 MB) exceeded the observation
worker's 1 MiB single-line result limit, both nodes handed back only
`ObservationIncomplete: RESULT_SIZE_LIMIT` with `node_results={}`, and the
runner read "sampled no rank process". The runner now distinguishes a lost
triage result (`<node>: no COLLECT_HUNG_TRIAGE result reached the record:
<cause>`) from dumps the product never read; batch 186 trims an over-budget
triage result stepwise on the node instead of dropping it, batch 186b gives
that ladder a `skeleton` floor (every rank's identity, status, watermarks and
raw-dump reference survive; the stand-in names its ranks in `ranks_summary`),
and once that Node Agent is live and the variant passes on both clusters its
expectation flips back to PASS.

## GF-GENERAL-LIVE-014 Kubernetes allocatable (native device-plugin freeze)

The five native 014 variants (`gpu_plugin`, `efa_plugin`, `one_sample`,
`replacement_uid`, `validation_failure`) live in
`scripts/e2e/regional/general_live_device_plugin.py` with the probe
`probes/general_live_device_plugin_probe.py` and the `DEVICE_PLUGIN_RESTART`
approval; four execute, `validation_failure` is `NOT_RUN(CAPABILITY_ABSENT)`
(below). The spec forbids patching `Node.status` or faking allocatable, and the
COLLECT-017 B/C fixture excludes the DaemonSet by affinity and restores it
*before* the product acts (so the product takes the `already_healthy` branch and
never deletes a Pod); neither reaches the product's own pod-delete/re-register
path. The injection instead SIGSTOPs the plugin container's own process -- only
a process whose cgroup names the plugin Pod UID *and* whose argv basename is
allow-listed (the static list plus the plugin container's own command/image
basenames read from the discovered Pod spec, since the HyperPod EFA plugin's
process basename was none of the original static names: live on the HyperPod
lane the Pod's command/image basename is `aws-efa-k8s-device-plugin` while
the container process is `efa-k8s-device-plugin`, which the static list now
carries), deadman-backed with SIGCONT. The
probe confirms the task actually reached the kernel stopped state (`T`) before
it returns -- `os.kill` only queues SIGSTOP, so a read on the next line races
the transition and can report `stopped:false` for a process that did freeze; a
freeze the probe cannot confirm stopped is refused immediately with that reason,
never waited out.

What the freeze does and does not break (live result, HyperPod, lane S45, and
kubelet as written). The harness first assumed kubelet would de-register the
resource and allocatable would decay while the Pod stays `Running`; it does not:
the `gpu_plugin` freeze left `nvidia.com/gpu` at 8 for about four frozen
minutes, so the allocatable oracle returned `INCONCLUSIVE`. kubelet's device
manager dials the plugin without keepalive and runs `ListAndWatch` under
`context.Background()`, blocking in `stream.Recv()` until an error
(`pkg/kubelet/cm/devicemanager/plugin/v1beta1/client.go`, `dial`/`Run`); only
an error ends the client and reaches `PluginDisconnected` →
`markResourceUnhealthy` (`manager.go`), and capacity/allocatable disappear only
after `endpointStopGracePeriod` (5 min, `types.go`). A stopped process keeps the
socket open and the kernel keeps the connection, so nothing ever errors. What a
frozen plugin does break is admission: `endpoint.allocate` calls the plugin's
`Allocate` with `context.Background()` (`endpoint.go`; `GetPreferredAllocation`
likewise, `PreStartContainer` alone has 30 s), reached from topology manager
`admitPolicyNone` → device manager `Allocate` on the kubelet sync loop
(`kubelet.go` `syncLoopIteration` → `HandlePodAdditions` → `canAdmitPod`), so the
first Pod that asks for the resource wedges the whole sync loop: no
`UnexpectedAdmissionError` event (that reason is produced only when `Allocate`
*returns* an error), the Pod stays Pending, the node stays Ready, and kubelet's
`/healthz` `syncloop` check fails only after 5 min. The product (R13h) detects
the hang from the node: the host collector probes each plugin socket with a
bounded `GetDevicePluginOptions` (`collectors/host/device_plugin.py`,
`device_plugin_probe.py`) and raises `gpu_device_plugin_unresponsive` /
`efa_device_plugin_unresponsive` after two consecutive timed-out ticks; the
planner stamps the `RESTART_*_DEVICE_PLUGIN` step `plugin_unresponsive=true` so
it deletes the Pod although allocatable is full (receipt
`restart_reason=DEVICE_PLUGIN_UNRESPONSIVE`) and confirms only a Ready Pod of
another UID with the full allocatable (`already_healthy=false`). The sample
carries `consecutive_timeout_samples` (1 after the first timed-out tick, the
finding after the second) and the finding on that metric plans
`RESTART_{GPU,EFA}_DEVICE_PLUGIN` (lane b204).

The variants assert that probe path (lane S47); the effectiveness oracle is the
probe sample, not kubelet's advertisement, and S45's "frozen but connected"
`INCONCLUSIVE` is retired for the product that has the probe. Before the freeze
the fixture decides which product it faces (`ProductGeneration`, evidence
`product-generation.json`): the probe's read-only `feature` reports whether the
node's host-collector venv (the unit's `ExecStart` path, default
`/opt/gpu-fault/current/venv`) carries `collectors/host/device_plugin.py`,
`device_plugin_probe.py` and `device_plugin_liveness.py`, plus the unit's
`GPU_FAULT_DEVICE_PLUGIN_*` settings; an API Pod answers whether the control
plane's `NodeHealthPolicy.METRIC_RULES` grades the family's
`*_device_plugin_unresponsive` metric. The installed package is the signal
rather than the node's latest-metric set because the liveness metric is not in
`LATEST_METRICS_REQUIRED` and a healthy tick's value-0 sample travels only in
health-summary batches, which are graded but kept as no raw evidence. A node
that probes under a control plane without the rule, or a node whose unit sets
`GPU_FAULT_DEVICE_PLUGIN_SOCKET_DIR` empty, is `INCONCLUSIVE` before anything is
frozen. Against the liveness-probe product `gpu_plugin`/`efa_plugin` assert, in
stages: `detected` -- within two host ticks (each stretched by the 5 s probe
timeout) plus the timeout plus 60 s of ingest, a `*_device_plugin_unresponsive`
= 1 sample with `probe_state=TIMEOUT`, `transport_answered=false` (a stopped
process sends no HTTP/2 frame), `failure_mode=DEVICE_PLUGIN_UNRESPONSIVE` and
the frozen plugin's socket (`nvidia-gpu.sock` / `aws-efa-device-plugin.sock`,
verified live on both clusters) in `sockets`; `restart_chain` -- that metric's
incident runs `RESTART_*_DEVICE_PLUGIN` → `TRIGGER_HEALTH_SNAPSHOT` →
`VALIDATE_{GPU,FABRIC}` all SUCCEEDED and ends RECOVERED; `deleted_original` --
`deleted_pod_uid` is the frozen Pod (or, without any receipt, the census shows
it gone) and, when the delete pass's own receipt was caught,
`restart_reason=DEVICE_PLUGIN_UNRESPONSIVE` (the executor replaces the step
record on every pass, so a 5 s poll can miss that pass; the result then carries
a note, and the later passes' `deleted_pod_uid`, which
`node_operations.deleted_plugin_pod_receipt` carries forward, proves the
delete); `replacement_ready` -- `replacement_pod_uids` names another UID, the
one Ready plugin Pod on the node is it, and the SUCCEEDED receipt says
`already_healthy=false` (a restart that read the full allocatable as healthy and
deleted nothing FAILs here); `allocatable_full_while_frozen` -- every raw kubelet
reading the runner took between the freeze and the product's finding equals the
baseline (a drop would mean the hung-but-connected premise did not hold on that
node, and is reported as such); `allocatable_restored`, `no_reboot`,
`not_isolated`; `replacement_uid` adds `no_redelete`. The result records the
detection latency, the readings while frozen and the host collector's own probe
journal (`probe-log`: one WARNING per timed-out tick). `one_sample` is decided
on the node: `freeze --resume-after-first-timeout --window-seconds <tick + 5 s +
20 s>` tails `gpu-fault-host-collector.service`'s journal after the SIGSTOP and
SIGCONTs the plugin the instant the first timed-out tick is logged (`(1
consecutive; finding after 2)`), well under a second later -- a control-plane
round trip could not win that window, because the first tick's value-0 sample
never reaches the store as evidence and the second tick follows one interval
later; the variant then watches two ticks plus 60 s and asserts no
`*_device_plugin_unresponsive` = 1 sample, no finding, no
`RESTART_GPU_DEVICE_PLUGIN`, no isolation, with the journal showing exactly one
timed-out tick. No logged tick within the window, a second tick logged before
the resume, or a journal with two ticks is `INCONCLUSIVE` (the window was not
achieved), not FAIL. Against the earlier product (no probe modules on the node)
the freeze is confirmed, allocatable is watched for eight ticks plus 120 s, and
when it does not drop the variant is `INCONCLUSIVE` with the kubelet mechanism
and the generation cited -- a product that lacks the feature by design is not
FAILed; if allocatable did drop, the allocatable-mismatch chain is graded as
before (`one_sample` has no allocatable-rule window any more and is
`INCONCLUSIVE` either way). Before R13h is deployed the live checkout carrying
this harness therefore records the four executable variants `INCONCLUSIVE` with
that reason; the rerun after the R13h deploy grades the probe path.

`validation_failure` is `NOT_RUN(CAPABILITY_ABSENT)` on both products, with the
mechanism in its reason. Its former definition (re-freeze every replacement so
the restart's allocatable snapshot never confirms) has no kubelet-level
mechanism left: a re-frozen replacement is Ready with the full allocatable, so
R13h's restart confirms it and the next probe tick opens a *new* finding instead
of failing this chain. The step's real failure path for a hang is a kubelet
already blocked in the frozen plugin's `Allocate`, which neither kills the frozen
container nor starts the replacement, so the step FAILs at
`restart_timeout_seconds` and escalates to `RESTART_NODE` under
`failure_escalation_action=REBOOT_NODE`; producing it means admitting a Pod that
requests the resource onto the node while the plugin is frozen, which wedges
kubelet's sync loop for every Pod on that node until the product's reboot -- not
something the harness may do to a live shared node. A replacement that never
becomes Ready cannot be made with a SIGSTOP (the plugin DaemonSets carry no
readiness probe, so a Pod is Ready as soon as its container runs), and excluding
the DaemonSet is COLLECT-017's path on a cluster-wide object the other lane
shares. The reboot escalation stays covered by the product's own unit tests of the
restart step and of the planner's `plugin_unresponsive` stamp (R13h).

Cleanup resumes any still-frozen process and proves allocatable at the baseline
with exactly one plugin Pod on the node. `expected_unknown` / `mig_timeslicing`
(CAPABILITY_ABSENT) and `running_training` (COVERED_BY_REGIONAL) stay `NOT_RUN`
in the denominator. All variants are registered on both lanes: nothing in the
freeze is provider-specific (the plugin Pods are found by their kubelet
device-plugin socket mount, the EFA plugin's DaemonSet name comes from the
node's provider, the SIGSTOP targets the plugin container's own process, and the
product's probe and restart run on both providers), HyperPod's
health-monitoring agent watches GPU/EFA hardware rather than the plugin, and
the lane preflight requires `NodeRecovery=None` there.

Live 2026-10-27 10:53Z the plain-EKS `efa_plugin` ended BLOCKED
`FIXTURE_NOT_READY expected one hyperpod-dependencies-aws-efa-k8s-device-plugin
DaemonSet on the cluster, found 0` because the fixture read the product helper
`node_provider.efa_device_plugin_name()` with the coordinator process's own
environment, which carries no `GPU_FAULT_CLUSTER_PROVIDER` and therefore means
HyperPod; the name is now resolved from the target node's provider (its labels,
the lane preflight's own test) through the product's per-provider table, the
discovery cross-checks the selected DaemonSet's chart identity
(`app.kubernetes.io/name=aws-efa-k8s-device-plugin` or the image basename) and
the product's `name=` restart selector and lists the DaemonSets actually
carrying that identity when they differ, and a product-side
`GPU_FAULT_EFA_DEVICE_PLUGIN_NAME` override is not read (the harness has no
reader of the executor's effective environment; the per-provider default is
what it expects) -- `scripts/e2e/regional/general_live_device_plugin_targets.py`.

The socket names are verified: a read-only listing of
`/var/lib/kubelet/device-plugins/` on the HyperPod node and on the plain-EKS node
(2026-10-27) shows exactly `aws-efa-device-plugin.sock`, `nvidia-gpu.sock`,
`kubelet.sock` and `kubelet_internal_checkpoint`, no MIG sockets, so the default
patterns resolve to one socket per family (`gpu=nvidia-*.sock` →
`nvidia-gpu.sock`, `efa=*efa*.sock` → `aws-efa-device-plugin.sock`) and the
frozen socket's basename the variants should expect in `sockets` is one of
those two.

Live R13h (2026-10-27, plain EKS `ip-10-0-43-5`, case 19:21:28Z-19:30:47Z) the
`efa_plugin` variant FAILed on a product defect, not the harness: the probe path
worked (SIGSTOP 19:22:36Z, `efa_device_plugin_unresponsive` with two timed-out
ticks at 19:23:03Z) and RESTART_EFA_DEVICE_PLUGIN deleted the frozen Pod, but
with grace 0. The force delete let the DaemonSet start the replacement at once,
the EFA replacement (hostNetwork, no init container) registered before kubelet's
SIGKILL ended the stopped container, and when the frozen stream ended kubelet's
device manager (up to release-1.36 it disconnects the client registered last
under the resource name, kubernetes issue 121994) dropped the replacement:
allocatable 0 behind a Ready replacement Pod, the restart FAILED at 180 s
(`final_allocatable: 0`, `restart_error: device plugin did not restore
vpc.amazonaws.com/efa`), the escalation QUARANTINE + REBOOT rebooted the node
(Ready again 19:27:56Z with 16 EFA devices), and the variant's cleanup probe,
running on the rebooting node, ended in `HostProbeMissingResponseError`. The
same freeze of the NVIDIA plugin passed on both clusters only because that
replacement (pod network, NVML start-up) registered after the frozen container
had died. Lane b212 deletes the plugin Pod gracefully (5 s,
`node_operations.PLUGIN_POD_DELETE_GRACE_SECONDS`): the Pod stays terminating
until kubelet stopped its containers, a non-surge DaemonSet starts no
replacement meanwhile, and the replacement registers only after the frozen
stream ended (unit model and evidence:
`tests/execution/test_device_plugin_restart_graceful_delete.py`, detailed design
v2 §3.8). The rerun on the release carrying b212 should see the delete pass's
receipt with `delete_grace_period_seconds=5`, `replacement_pod_uids` and the
full allocatable; HyperPod's `efa_plugin` runs the same eks-charts EFA plugin
(hostNetwork) and would have failed the same way on R13h.

## GF-GENERAL-LIVE live acceptance runner (phase 2, lane K)

Phase 2 adds reviewed `live-replay` categories that drive the product's own
readers. The reusable kernel-log source replay
(`scripts/e2e/regional/general_live_kernel_replay.py` and
`probes/general_live_kernel_source_probe.py`) installs a reviewed set of
kernel-transport records for one node and boot and reads back what the product
ingested, two ways: a `journalctl` pass-through wrapper on the node log
collector (journal mode) and a `GPU_FAULT_KMSG_PATH` repoint of the kernel
collector (kmsg mode). A sibling host-collector source replay
(`general_live_source_replay.py` / `probes/general_live_source_probe.py`,
command wrapper `tools/host_command_replay.py`, fixture builders
`probes/general_live_source_builders.py` -- a probe ships as one file, so the
controller installs the builders through `install-source`, sha256-checked like
the wrapper, before the first `build-*` and the probe imports that copy; the
window, deadman and cleanup commands need only the probe) is the shared
raw-source fixture of the host/device categories (002/007/010/011/012/013/015/032). It
changes only the private environment of `gpu-fault-host-collector.service`
through one `/run` drop-in that cleanup and a deadman (armed before the drop-in)
remove, with fixture content installed base64 + sha256 under
`/run/gfrs-live/source-<digest>/`: a reviewed procfs/sysfs *file* bound
read-only into the unit's mount namespace (`/proc/stat`, `/proc/meminfo`,
`/proc/driver/nvidia/version`, a `gfrs*` test netdev statistic, an EFA port's
`hw_counters`/`counters`/`state`/`phys_state`), a whole reviewed *tree*
(`/sys/class/infiniband` built from the real tree with reviewed mutations,
`/proc/driver/nvidia`), a reviewed pass-through *command wrapper* first on the
unit's PATH for `smartctl`, `ipmitool`, `ethtool` and the host collector's
`nvidia-smi` queries (per-tool, per-query, per-field allow-list; bounded calls
and deadline; a failed real query is passed through as it failed; every
catalogued call is journaled), a path or tool binary made *inaccessible* inside
the unit, an expected GPU/EFA count override, or the `GPU_FAULT_PCI_DEVICES_ROOT`
repoint. The count override is handed to the unit as an `EnvironmentFile=`
(the run's `/run/gfrs-live/source-<digest>/env`), never as `Environment=`: the
installer writes the real counts into the unit's own
`EnvironmentFile=/etc/gpu-fault/collector.env` (`GPU_FAULT_EXPECTED_GPU_COUNT="8"`,
derived from the instance type, older than the reachable history) and systemd
applies every `EnvironmentFile=` after every `Environment=`, later-listed files
winning, with drop-ins parsed after the unit file (systemd.exec(5)
`EnvironmentFile=`, systemd.unit(5)); `UnsetEnvironment=` is applied last of
all, so no file resurrects an unset variable. Live 2026-10-26 (eksb, 012
`expectation_wrong`) the drop-in `Environment=GPU_FAULT_EXPECTED_GPU_COUNT=9`
lost to the unit file's `"8"`: the window opened, the unit restarted, the
process still read 8 and the verification closed the window (`ProbeError
ce53d15331ea1884 at open_window:1161`); a systemd 255 user-unit experiment on
the controller host reproduced all four rules (drop-in `Environment=` loses,
drop-in `EnvironmentFile=` wins, `UnsetEnvironment=` removes a file-set key, a
file line `X=` yields the empty string). After the unit restarts the probe
verifies through `nsenter` that each replacement is visible inside the unit
and not to PID 1, or closes the window. The close (`close-window`, the
deadman, `await-ticks --close`) removes the drop-in, reloads and reads
`DropInPaths`/`NeedDaemonReload` back -- a removed drop-in stays listed until
the next reload (systemd 255; the listing SSM found after the live 013 close
on 2026-10-26) -- reloading once more if it is still listed, restarts the unit
and reads the real configuration back (`restore.checks`: every replaced source
as PID 1 sees it, every touched variable at the `baseline_env` recorded at the
open, unset ones absent; `restore.ok`, `closed_at`, `reloads`). A failed probe
call still surfaces only whitelisted tokens in the exception text (class,
16-hex digest, `function:line`), but the probe's own message and structured
detail (`ProbeError.detail`, e.g. the failed verification's checks) are now
written, redacted (credential-shaped tokens and long opaque blobs masked,
bounded) to the case directory as `probe-error.json`, which the exception text
names; two lanes had brute-forced the digest that day.
No host binary is replaced, no global mount is made, the Node Agent is never in
the allow-list and `py-spy` (run only by the Node Agent) is refused by name.
Everything beyond the phase-1 binds and the PCI-root repoint needs
`--approve HOST_SOURCE_REPLAY`. Every replay only replaces
the raw reading; parsing, trust classification, grading, the database and the
approved node actions stay real, so the evidence is always `proof_scope:
live-replay` -- it proves the processing chain, never the physical fault.

Implemented live-replay variants: 016 severe kernel log (all nine, including the
trusted-source and current-boot gates that must *not* isolate on a user-space or
old-boot line), 017 application error logs (the OOM diagnostic and the NCCL /
distributed evidence-only paths), 032 unknown error (`unknown_xid` graded
SITE_SAFETY, its dedup, and `missing_topology`: a reviewed fatal Fabric Manager
SXID summary line naming no link on a node whose `GPU_FAULT_NVSWITCH_TOPOLOGY_
COMMAND` is unset -- the installer default on both clusters -- so no trusted
`nvswitch_port_topology` exists and the H200 access-only invariant cannot
apply; the product must answer BLOCKED_MISSING_EVIDENCE "requires a trusted
access/trunk port-scope mapping" with the full fabric reset withheld, and the
site-safety quarantine is handed back through the validated restore), 013 EFA
inventory `unreadable_state` (UNKNOWN, not zero cards) plus the whole-tree
`/sys/class/infiniband` replays `unknown_inventory` (an ambiguous driver
identity is UNAVAILABLE, no count), `excess_device` (a duplicated real device
is EXCESS_DEVICE, RUN_DIAGNOSTICS whose VALIDATE_FABRIC concludes on fresh
evidence -- SUCCEEDED on the restored count or FAILED on a fresh excess reading
with nothing escalated -- with `details.inventory_evidence` recorded, never the
stale `missing_inventory_action_completion` wait, RECOVERED, no reboot or
isolation) and `link_inactive` (one real port DOWN/Disabled is LINK_INACTIVE,
diagnose first), and the phase-1 host file-source variants 001 `invalid_delta`,
008 `statistics_failed` and 009 `errors_delta`. `link_inactive` expects PASS
since batch 175 under `HOST_SOURCE_REPLAY` alone (`S`, risk M; the old
deadline-then-RESTART_NODE gap and the `NODE_REBOOT` token are gone): because
the diagnostic now judges fresh readings and a port still inactive is the
designed REBOOT_NODE, the inactive port is shown for a bounded exposure only --
the probe watches the collector's own ticks on the node (`await-ticks`, the
unit's `WatchdogTimestampMonotonic`) and closes the tree window right after
the tick that confirms the shortfall and mints the finding
(`GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES`, the second by default),
before a third could be read (`observed_exposure`; the evidence
`link-inactive-exposure.json` records when each observed tick ended,
`closed_at` and `closed_by: probe`, or `deadman` when the fallback fired
first). Neither the runner's store polling (20 s polls plus probe round trips
against a 15 s tick) nor a fixed bound from the restart can do that: the
deadman that closed the window 1.7 intervals (26 s) after it opened lost to
the collector's startup phase live on 2026-10-26 (eksb: first tick 13 s after
the open, deadman before the second, no finding, INCONCLUSIVE); it is now the
fallback `exposure_fallback_seconds` (105 s here). PASS is a first workflow
that diagnoses (FREEZE_EVIDENCE + VALIDATE_FABRIC, no hardware or isolation
step), its VALIDATE_FABRIC SUCCEEDED with
`details.inventory_evidence[node][efa_inventory_active_count]` at the restored
count observed after the restore, no stale gate, the incident RECOVERED, no
RESTART_NODE and the healthy count read back; no finding or fewer shortfall
readings than confirming ticks inside the exposure (the message names the
observed ticks, the exposure's length and who closed it), or a shortfall
reading recorded after the restore (the bound did not hold and the designed
escalation applies), is INCONCLUSIVE rather than a verdict on the product.

The second live run of that design on eks (2026-10-26 18:59Z, R13g, lane node
ip-10-0-43-5) was INCONCLUSIVE again -- a 21.4 s exposure, `closed_by: probe`
after two "ticks", one shortfall reading and no finding inside it -- while hpb
PASSed the same morning on the same harness: a timing exposure in the probe's
tick count, not in the design. systemd stamps a `Type=notify` unit's
`WatchdogTimestampMonotonic` when the unit enters start-post on `READY=1`, 30 us
before `ActiveEnterTimestampMonotonic` and before the collector's startup phase
(reproduced on systemd 255 with a transient user unit), so a probe loop that
starts before the collector's first real tick reads that stamp as a ping: on
eks the restarted collector's READY came 1.9 s after the restart, the first real
tick 12.6 s after it, and the loop began 11.1 s in (the exec round trip) --
the READY stamp became tick 1 and the window closed after one real read. On
hpb the first real tick (10 s in) preceded the loop start (12 s in) and had
already overwritten the stamp. The product's own records show which reading
was missing: `rdma_link_down` fired from the tick observed at 18:59:13.83Z
(`observed_at` is the tick's start, stamped before any contributor reads --
not the delivery time), nothing was read near the 18:59:03.1Z "tick", so no
inside reading was recorded late; the second confirming read never happened
before the close. `await-ticks` now counts only stamps strictly after
`ActiveEnterTimestampMonotonic` (`STAMP_PROPERTIES`; the receipt and
`link-inactive-exposure.json` carry `unit_active_at`), the deadman stays the
hard bound, and the INCONCLUSIVE names the missing reading
(`general_live_source_replay.exposure_reading_gaps`, also on 010
`link_inactive`'s "no sample": a tick with no batch is "not a collector read,
or its batch was never delivered", a batch without the shortfall "read no
shortfall on it", a shortfall observed after the close "read after the restore,
outside the bound", then how many confirming readings never happened before
the close). The tests replay the eks chronology on the probe's fake systemd
(`FakeHost.ready_offset`: the READY stamp 1.9 s in, real ticks 12.6-13.0 s
and 27.6-28.0 s, the loop starting 11.1 s in; the close must follow the second
real tick and precede the third at 42.6 s -- the old rule closed 19-21 s in)
and on the product's records (one shortfall batch 12.6 s in, no finding), and
pin the three gap lines.
013 `pci_missing`
(`scripts/e2e/regional/general_live_efa_pci.py`, both clusters, risk H, every
lock, `--approve HOST_SOURCE_REPLAY NODE_REBOOT`, expected PASS) is the
PCI_DEVICE_MISSING reboot chain on a live-replay: first the spec's negative
stage -- `build-pci-fixture --unreadable` makes the PCI-root fixture a regular
file, so the product's listing fails (ENOTDIR) under any privilege, held for
four host ticks (twice the two-sample confirmation window); the product must
record `EFA inventory observation unavailable` and no inventory count, mismatch,
finding, hardware step or isolation -- only the inventory's own `efa_inventory_*`
samples are graded (`EFA_INVENTORY_METRICS`): the RDMA tree is real in that
window, so `_rdma`'s per-port `rdma_link_down` keeps flowing and is recorded
(oracle `rdma_samples_recorded`), not counted; live hp 2026-10-27 the 16
`rdma_link_down=0` of every delivered tick were graded as counts from the
unreadable root and failed a correct product -- (its error path also resets
the confirmation streak,
pinned by running the real collector over missing -> unreadable -> missing ->
missing: only the fourth tick confirms); a finding from the unreadable root is
FAIL, its chain is waited out and the positive stage is skipped. Then the node's
real PCI root is copied without one real EFA function (`--source
/sys/bus/pci/devices --drop <BDF>`, the BDF from the probe's `inventory`, which
now names each RDMA device's PCI function) together with the real RDMA tree
without that device, in one mixed window closed the moment the finding exists,
so every read behind the reboot is real. The product's chain is MARK_UNSCHEDULABLE,
QUARANTINE, one RESTART_NODE through the provider, VALIDATE_GPU/HOST/FABRIC on
post-reboot evidence (`EFA_INVENTORY_RECOVERY_OPERATIONS` lists RESTART_NODE)
and RESTORE_SCHEDULING; the runner waits for the reboot like COLLECT-004 (new
boot id and Ready on the same Node UID, `wait_node_settled_after_reboot`),
re-creates the probe Pod that died with the node and proves on the new boot
that no drop-in, deadman, state directory or fixture PCI root survived
(`snapshot`) and that the unit reads the real RDMA tree again (`read-source`).
Graded on the product's records only: the confirmed sample's failure mode and
counts, exactly one SUCCEEDED RESTART_NODE execution and node-scoped remote
command, no REPLACE_NODE or second reboot, the incident RECOVERED, the node
Ready and schedulable with the EFA and GPU allocatable at baseline, the
post-reboot inventory on the signals the product really records -- (1) the
latest `efa_inventory_active_count` (in
`host_health.NodeHealthPolicy.LATEST_METRICS_REQUIRED`, so every delivered
batch, health summaries included, advances its latest-value row) newer than
the SUCCEEDED RESTART_NODE execution's `updated_at`, the anchor the product's
own inventory gate reads from
(`adapters/gpu_validation.inventory_observation_window`), waited at most one
more health-summary phase (`GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS` + 2 ticks);
(2) the new boot's first inventory batch -- the collector's `baseline`
delivery, always raw evidence -- reading `efa_inventory_mismatch` /
`efa_inventory_missing_count` 0, taken as the latest reading after the
RESTART_NODE execution's `started_at` (the old boot delivers at most one more
tick, then the new boot's): neither gauge has a latest-value row and a steady
healthy node delivers health summaries only, which the ingest never writes as
raw evidence, so "a mismatch=0 sample after the reboot completed" does not
exist in the product's record (live hp 2026-10-27 r3: SUCCEEDED chain, fresh
`active_count`, yet "no fresh efa_inventory_mismatch=0 ... (latest None)");
(3) the chain's SUCCEEDED VALIDATE_FABRIC
`details.inventory_evidence[node][efa_inventory_active_count]` (batch 175: the
gate records the reading every verdict rests on) at the expected count,
observed after that completion. The requested/completed/new-boot moments, the
wait's polls, the new boot's readings and the gate's judged reading are
recorded in `fresh-inventory-anchor.json` (the earlier anchor, the runner's
clock after its post-reboot settle, probe re-create, snapshot and read-back,
failed even the `active_count` half on hp 2026-10-27 r2 with "latest None")
--, the provider's CloudTrail reboot record (EC2
`RebootInstances`, SageMaker `BatchRebootClusterNodes`) naming this node and
no other -- EC2 by instance id, SageMaker by the `NodeLogicalId` the product
submits (`hyperpod.py` `NodeLogicalIds`; `request_node_ids` is null in the
receipt), which the runner binds to the instance before the window opens with
`sagemaker list-cluster-nodes --include-node-logical-ids`
(`hyperpod_logical_id`, recorded as `hyperpod_binding` in `node-before.json`;
an unresolved or ambiguous binding fails the receipt stage by name; live hp
2026-10-27 the instance-id match graded the receipt that named this very node
as "another target") --, and `--node-b`
untouched. Reboot safety: the close precedes RESTART_NODE; if the product is
faster, nothing of the window lives outside `/run`, a deadman firing during
shutdown can only fail a `systemctl restart` the shutdown kills, and the new
boot's snapshot is the proof. A Node UID change during the wait (Spot reclaim)
is INCONCLUSIVE; a node left isolated is handed back through the validated
restore. It needs at least 70 minutes of maintenance window; run it on EKS
first, on HyperPod only with `NodeRecovery=None`. 032 `missing_version` runs on
the kernel collector's own command window (`--approve
KERNEL_COLLECTOR_SOURCE_REPLAY` plus `KERNEL_SOURCE_REPLAY`, both lanes): the
NVLink5 decoder gates on the `driver_branch` the *kernel collector's*
CollectorContext stamps on the XID event from its `nvidia-smi
--query-gpu=driver_version` at unit start, so the reviewed wrapper is placed
first on that one unit's PATH and answers only that query with a failure
(`GPU_FAULT_DRIVER_BRANCH` is unset in the unit's environment for the window;
every other `nvidia-smi` call passes through; no file, tree, override or
PCI-root mode exists for that unit, `HOST_SOURCE_REPLAY` never unlocks it and
the new token never unlocks the host collector). The runner proves from the
wrapper journal that the restarted collector asked and was refused and from the
unit's own journal (`journal-read`) that no branch was discovered, then replays
the real B-series XID 149 `NETIR_LINK_EVT ... (0x004505c6 0x00000000)` line
through the kmsg replay and grades the product's records: `intr_info` and
`error_status` parsed, `driver_branch` null, BLOCKED_MISSING_EVIDENCE naming the
driver branch, WORKFLOW_NVLINK5_ERR withheld, operator required, no reset or
reboot, quarantine handed back through the validated restore. A window that did
not take effect is INCONCLUSIVE and nothing is injected (under the pre-R575 V1
decode the same line is RESET_GPU). On this site the variant records
`NOT_RUN(CAPABILITY_ABSENT)` before any mutation: `_resolve_xid` runs the
Catalog's product gate before the workflow resolver and XID 144-150 apply to the
B100/GB200 families only, so every such line is NOT_APPLICABLE on an H200
whatever the branch (verified with the real parser and engine) -- the version
gate is reachable only on a Blackwell node, where the same code runs for real.

The variants that stay `NOT_RUN` on this single-node site keep their place in the
denominator with an explicit reason: the 014 `expected_unknown` / `mig_timeslicing`
(no instance type outside `INSTANCE_ACCELERATOR_COUNTS`, no MIG/time-slicing; `running_training`
is `COVERED_BY_REGIONAL`, and 014's native device-plugin freezes are implemented, see above), (013 `pci_missing`, `bind_failure` and `post_repair_failure` are implemented above;
`driver_unbound` is `COVERED_BY_REGIONAL`), the catalog/authority-bound 032 variants (a genuinely unknown product

The two native 013 EFA driver-binding negatives are implemented (not a replay):
`bind_failure` and `post_repair_failure` (`general_live_efa_driver.py` with the
on-node `probes/general_live_efa_probe.py`, `EFA_DRIVER_UNBIND` + `NODE_REBOOT`,
flag X, both clusters, run last). Both reuse the COLLECT-017 A / DESTR-021 real
unbind of one allow-listed EFA data function and its `systemd-run --on-active`
rebind deadman. `bind_failure` holds that one function un-rebindable (its
`driver_override` set to a name no module provides, so the kernel refuses every
bind of it to `efa` with ENODEV) and unbinds it: the product's
`REMEDIATE_EFA_DRIVER` records an independent FAILED bind receipt naming the
BDF, and -- the step carrying `failure_escalation_action=REBOOT_NODE` for a
DRIVER_UNBOUND finding -- escalates to RESTART_NODE, the approved branch for a
bind that will not take (and the one that clears the per-device override). No
global `efa` unload ever happens: the module stays loaded and every other
function keeps its binding. `post_repair_failure` uses no override; a bounded
one-shot trap re-unbinds the function a few seconds after the product's bind, so
`REMEDIATE_EFA_DRIVER` records SUCCEEDED (`rebound_pci_bdfs=[BDF]`) and a later
post-repair step (`RESTART_EFA_DEVICE_PLUGIN` or `VALIDATE_FABRIC`)
independently FAILS on fresh evidence and escalates to the same reboot -- the
independent failure receipt and the escalation the spec requires. Both grade
only on the product's own records (the incident reasons, the step receipts, the
EFA inventory, a changed boot id, no isolation beyond the chain's own cordon --
the reboot successor's QUARANTINE included, see below), are scoped to one
function, node-reversible (one write to
`driver_override`, or a reboot) and deadman-restorable, and their cleanup binds
the function back to `efa` or fails loudly. The probe's "no override" is the
kernel's own read-back: the attribute's show callback prints the
`pdev->driver_override` pointer with `"%s\n"` (`drivers/pci/pci-sysfs.c`
`driver_override_show()`: `sprintf(buf, "%s\n", pdev->driver_override)` at
v3.16, `sysfs_emit(...)` at v5.18 and v6.1) and vsprintf renders a NULL `%s`
argument as the literal `(null)` (`lib/vsprintf.c`: `string()` at v4.19,
`if ((unsigned long)s < PAGE_SIZE) s = "(null)";`; `check_pointer_msg()` at
v6.1, `if (!ptr) return "(null)";`), so an unset override reads `(null)`, never
an empty string -- the store side turns `""` and `"\n"` into a NULL pointer
(`driver_override_store()` through v5.18, `drivers/base/driver.c`
`driver_set_override()` from v5.19, read at v6.12: `/* Empty string passed -
clear override */`, `/* "\n" passed - clear override */`;
`Documentation/ABI/testing/sysfs-bus-pci`: "may be cleared with an empty string
(echo > driver_override)") and the attribute is unconditional
(`DEVICE_ATTR_RW(driver_override)` in `pci_dev_attrs[]`). Any other read-back
(`efa`, `vfio-pci`, the ABI's own opt-out example `none`) is a stored override
and the probe refuses to inject (`reason driver_override_set`). The first live
run (HyperPod, kernel 6.12.103) compared `(null)` against a made-up `(none)` and
refused every real node (`bind_failure` INCONCLUSIVE; the cleanup misreported
"not restored" on the same comparison); one `KERNEL_NO_OVERRIDE` /
`override_name()` in the probe now decides, the survey keeps the raw read-back
as `driver_override_sysfs`, and `tests/regional/test_general_live_efa_probe.py`
pins it. The probe's refusal message and structured detail travel with its
failure line into the case directory's `probe-error.json` (the source-replay
probe's lane; a later failure keeps the earlier record under `previous`), the
trap unit starts as systemd-run's default `Type=simple` service (`Type=oneshot`
would block `systemd-run` until the process exits and `RuntimeMaxSec` has no
effect on it -- systemd-run(1), systemd.service(5)) and waits for the fixture's
own unbind before it waits for the product's bind. 032 `missing_version` stays
`FIXTURE_NOT_READY`: the NVLink5 decoder gates on the `driver_branch` the
*kernel collector's* own CollectorContext stamps on the XID event, and the host
source replay is allow-listed for `gpu-fault-host-collector.service` only.

Live 2026-10-27 06:55-07:06Z (HyperPod, R13g, harness int/r13c-live24)
`bind_failure` ran end to end for the first time: the survey read 16 functions
bound and `driver_override_sysfs` `(null)`, the hold made the product see
DRIVER_UNBOUND, its `REMEDIATE_EFA_DRIVER` bind of the BDF failed with ENODEV
(`bind_errors=<BDF>=OSError: [Errno 19]`), the record ended FAILED with the
incident parked QUARANTINED, and the ladder -- the step's
`failure_escalation_action=REBOOT_NODE` (`orchestration/families/health.py`
`_efa_parameters`; `orchestration/escalation.py` `_classify`, stage
`efa_recovery`) -- opened `workflow-reboot-after-<record>` 23 ms later with the
plan every reboot/replacement successor gets: FREEZE_EVIDENCE,
MARK_UNSCHEDULABLE, **QUARANTINE**, RESTART_NODE, VALIDATE_GPU, VALIDATE_HOST,
VALIDATE_FABRIC, RESTORE_SCHEDULING (`escalation.py` `_escalation_operations`).
The node rebooted (new boot id), came back Ready with 16/16 bound and was
released, and the predecessor incident was closed RECOVERED by the successor's
restore (`incident_closure.py` `_close_escalation_predecessors`, "node restored
via escalation successor") -- the designed path, not a product defect. The
variant still graded FAIL on two of the harness's own expectations:
`not_isolated_beyond_chain` read the successor's QUARANTINE as isolation beyond
the chain, and the cleanup's `release` went through the probe Pod the reboot had
terminated (`HostProbeTransportError: owned host probe Pod has terminated`).
Both variants are reboot-aware now (`general_live_efa_escalation.py`):
`escalation_shape` asserts, on the record before and at the reboot, one
REMEDIATE attempt, the remediation record FAILED without restoring scheduling,
exactly one reboot-after successor planning that REBOOT_NODE chain,
RESTART_NODE executed exactly once and SUCCEEDED, no REPLACE_NODE and no
successor of the successor; `not_isolated_beyond_chain` counts only a QUARANTINE
in the remediation record itself, a STOP_WORKLOADS naming no workload, or any
isolation step of a workflow outside the chain; `recovered` requires, on the
settled record, a new boot id, a node free of the product's isolation, the
successor SUCCEEDED and both incidents RECOVERED (the predecessor's closure is
given up to 180 s more). The waits follow: `efa-driver-escalation.json` is
written once the remediation record is terminal and its successor is on the
record (the record before/at the reboot), and a ladder that stopped on two
consecutive polls without a RESTART_NODE returns the settled wait at once so the
stages name the deviation instead of waiting out the reboot window. Cleanup: a
`release` the probe answers with `HostProbeError` goes through
`close_after_probe_loss` (the DCGM replay's close): a node back on the same
boot gets the retry through the same Pod; a boot id changed against the one the
fixture was armed on, with the probe still refusing, means the sysfs override,
the `systemd-run` deadman/trap units and the `/run` state went with the boot --
and the product's own inventory of the new boot (an
`efa_inventory_active_count` at the expected count with `mismatch=0` observed
after the node's Ready transition) must prove the function bound, or the
cleanup fails as unproven; without a reboot the probe failure stays the cleanup
error. `post_repair_failure` is aligned the same way: the trap's record is read
every 10 s while the product repairs, before the reboot can take the probe
(`summary` carries it as `trap_fired_at`, the JSON text the trap unit writes --
the earlier grading read a `trap_fired` dict key that does not exist there and
would have reported "trap did not fire" live), and a probe lost before that read
leaves the product's own SUCCEEDED-then-post-repair-FAILED record as the witness
(`trap_witness=product record`). `tests/regional/test_general_live_efa_driver.py`
pins it on fake timelines in the live shape: with and without the reboot, the
chain's own versus a foreign QUARANTINE, a blind reboot loop, REPLACE_NODE, a
ladder that never reboots, a probe that dies with the reboot, and the cleanup
after the reboot.

The same day's `post_repair_failure` (07:08-07:51Z, the same node on its new
boot) took the product's other designed path: REMEDIATE_EFA_DRIVER bound the
function back at 07:10:00 (the trap unit saw the bind, re-unbound after its 5 s
grace at 07:10:04 and recorded `fired=true`, read by the second summary at
07:10:08), RESTART_EFA_DEVICE_PLUGIN passed as `already_healthy` on the
plugin's unchanged allocatable of 16 (`adapters/kubernetes/node_operations.py`
`_restart_device_plugin_node`: `allocatable >= expected` restarts nothing), and
VALIDATE_FABRIC never got an `efa_inventory_active_count` newer than the last
recovery step within its 600 s per-step cap (`adapters/gpu_validation.py`
`inventory_observation_window`), failing with "stayed non-terminal for 602s,
past the 600s per-step cap". By `validation_evidence.withheld_escalation` a
validation a bound failed before it observed any node failure is not granted
the REBOOT_NODE it asked for; the ladder's VALIDATION_EVIDENCE_MISSING_STAGE
opened `workflow-support-after-<record>` (FREEZE_EVIDENCE + ESCALATE_SUPPORT)
and the node kept the chain's own cordon for an operator -- **no reboot**. The
earlier grading turned that into three errors of its own: `action_succeeded`
read `commands[*].operation`, while a `RemoteActionCommand` (`regional.py`)
names its operation only under `step.operation`, so the receipt was always
empty ("rebound None", "already bound"); `trap_fired` read a `trap_fired` dict
key where the probe writes `trap_fired_at` JSON text; `escalation`/`recovered`
waited the whole 40 min for a reboot. Now the receipt is read from the step
execution's `details.node_results[node]` or the command whose `step.operation`
matches (`node_result`); the post-repair failure is classified by the
product's own rule (`post_repair_verdict`: a non-validation step, or a
validation with `node_failures`/`failed_nodes`, is `observed`; a validation
with only `node_pending`/`step_waiting_timeout_seconds`/the deadline is
`withheld`); on `withheld`, `escalation_shape` grades the hand-off's shape
(exactly one support-after successor planning FREEZE_EVIDENCE +
ESCALATE_SUPPORT, no reboot-after, no RESTART_NODE executed, no REPLACE_NODE),
`recovered`/`escalation` no longer demand a reboot, and the one failing stage
is `post_repair_failure_observed` -- the fresh-evidence failure the variant
claims was not demonstrated -- with the product-side question in
`product_gap`: TRIGGER_HEALTH_SNAPSHOT had restarted the host collector
(`node_agent/operations/diagnostics.py` `_trigger_health_snapshot`) and the
function was unbound again, so why did no inventory sample newer than the
recovery steps' completion arrive within 600 s (the step's
`details.node_pending`, which `efa-driver-settled.json` now records together
with `inventory_evidence` and `step_waiting_timeout_seconds`). The cleanup
gained the hand-off's exit: `hand_back` runs after the fixture's release
(LIFO) and, when the node is still isolated and the run has incidents, hands
it back through the product's validated restore (`restore_incidents`); on the
reboot path the product released the node and it is skipped. The HyperPod node
most likely still carries the chain's cordon and an ESCALATED incident after
that live run and needs this exit.
needs other hardware; `observe_profile` / `permitted_isolation` need a Runtime
Profile authority the site does not configure, AUTH-015), the 017 attribution
variants (a managed attempt with CRI Pod logs, lane N territory), and 004
`probe_error` (the product reports a `statvfs` error as `filesystem_unavailable`
since batch 163, but injecting a real one needs a faulting test mount -- a
read-only file bind cannot, statvfs resolving the mount, not the bound file).

Live 2026-10-27 (R13g, both clusters: HyperPod 07:08-07:51Z, plain EKS
ip-10-0-43-5 10:18-10:33Z) `post_repair_failure` took the product's other path
and exposed a product gap behind it. REMEDIATE_EFA_DRIVER rebound the function
(SUCCEEDED 10:19:46.7Z), the trap re-unbound it 4 s later (10:19:50.3Z),
RESTART_EFA_DEVICE_PLUGIN had already passed on the plugin's unchanged
allocatable (10:19:48.2Z), TRIGGER_HEALTH_SNAPSHOT restarted both collectors
(10:20:01.7Z), and VALIDATE_FABRIC waited out the 600 s cap (10:30:06.2Z) with
`node_pending = {node: {missing_or_stale_collectors: [HOST_TELEMETRY]}}`; the
product withheld the REBOOT_NODE the plan asked for (`validation_evidence_missing`)
and opened `workflow-support-after-<record>`, leaving the node cordoned for an
operator. The missing evidence was not missing. The restarted host collector's
first tick (the `host.request` force snapshot skips the start-up spread) read 15
of 16 RDMA devices and delivered `efa_inventory_active_count=15` with the
DRIVER_UNBOUND verdict seconds after the last recovery step -- exactly the sample
`inventory_observation_window` asks for -- and kept delivering it (the mismatch
edge on the second tick, the health summary every 300 s +-10 %). But the same
batches carried the RDMA traffic contributor's `incomplete EFA traffic coverage:
15/16 devices` as a collection error (`collectors/host/network.py` `_rdma`), the
ingest records an erroring batch with `last_success_at=None`
(`app/ingest/telemetry_context.py` `_record_collector_status`), so HOST_TELEMETRY's
status read COLLECTION_ERROR / NO_POST_ACTION_OBSERVATION for as long as the
function was unbound, and the validation adapter's collector check
(`adapters/gpu_validation.py` `_recent_host_metrics`,
`observation_health.observation_reason`) returned `missing_or_stale_collectors`
before it read a single row. The edge filter was not the blocker (the hypothesis
the harness recorded as `product_gap`), and neither was the cadence: an unbound
function was itself reported as a collector failure, so a post-repair validation
of a node that stays unbound could never reach its inventory verdict on either
cluster -- deterministically, not by timing. Two changes, one per half of the
root cause. The inventory gate now fails a node on an inventory row its host
collector delivered since the action even when a collector the step waits for is
stale or erroring (`adapters/inventory_verdict.py` `inventory_failure_despite_collection_error`,
`observation_health.reported_since`: the newest batch of either kind, within the
post-action age and after the last recovery step; `details.host_collector_errors`
keeps the batch's errors beside the verdict). It passes nothing on such a batch
-- the rows an erroring batch lacks would read as clean, so the clean path is
still the only way to SUCCEEDED -- and reads no verdict from a batch whose
inventory contributor itself failed (`inventory_contributor_error`: a
`_gpu_inventory` query that exited non-zero after listing the GPUs that still
answer ships the partial count with its error; an unreadable `_efa_inventory`
source ships none), so the unreadable-source UNKNOWN of `pci_missing` /
`unreadable_state` stays UNKNOWN and the pre-action row stays excluded. And the
RDMA traffic contributor no longer classifies the node's own inventory shortfall
as a collection error: the aggregate it cannot complete is still withheld
(`efa_traffic_observation_valid=0`), the shortfall is the inventory contributor's
`efa_inventory_mismatch` (DRIVER_UNBOUND / PCI_DEVICE_MISSING / EXCESS_DEVICE),
and HOST_TELEMETRY is no longer erroring -- with its hourly
`collector_observation_unknown` refresh finding -- for the lifetime of a missing
device or of a mistyped `expected_efa_device_count`. With the adapter change
alone the live chain ends FAILED on `efa_inventory_active_count=15,expected=16`
at the validation's first attempt after the restarted collector's first batch,
`validation_without_observed_failure` is false, and the plan's REBOOT_NODE opens
the reboot rung the variant expects; `bind_failure`'s reboot-after
VALIDATE_FABRIC is unchanged (after the reboot every function is held, the batch
is clean, the clean path judges it).
`tests/execution/test_validation_inventory_despite_collection_error.py` runs the
live shape (the chain's three completed steps, the sticky `last_success_at` from
before the unbind, the erroring post-restart batch with the 15 row) to a verdict
the ladder owes, and pins the preserved behaviours (an erroring batch with the
expected count still waits, a count from before the last recovery step is no
verdict, an unreadable inventory source stays UNKNOWN, a partial GPU query is no
verdict while a coexisting `_smart` error is, a stale erroring status is no
verdict, a dark DCGM does not hide a fresh host inventory failure);
`tests/collectors/test_host_rdma_coverage_is_inventory.py` runs the real host
collector over a fake sysfs with one of two functions unbound (no collection
error, `efa_inventory_mismatch=1` DRIVER_UNBOUND, traffic withheld) and with
both held (complete traffic on the second tick). The harness grading on
`int/r13c-live26` (S49) reads the withheld hand-off as the product's path and
fails `post_repair_failure_observed` alone; with this release the variant's
chain reaches the reboot rung and the same grader's `post_repair_verdict` reads
`observed`.

A sixth deferral reason, `COVERED_BY_REGIONAL: <case ids>`, names GF-REGIONAL
cases that already PASSED live on this site with the very injection and
assertions the variant would make, so the variant gets no fixture of its own
and stays `NOT_RUN` with the cross-reference (its registry entry keeps lanes,
tokens and expectations for a later re-enable). Three variants carry it: 013
`driver_unbound` -- the same real `efa` unbind of one BDF and the same
REMEDIATE_EFA_DRIVER 7-step chain assertions passed as GF-REGIONAL-DESTR-021
(eks, 2026-10-22), GF-REGIONAL-COLLECT-017 stage A (hp, 2026-10-22) and
GF-REGIONAL-NET-007 (eks); 014 `running_training` -- GF-REGIONAL-COLLECT-017
stage C (hp, 2026-10-22) removed the EFA device plugin under a managed training
job and proved the Pod UID set and the progress preserved with no
STOP/RESTART_WORKLOAD; 030 `no_spare` -- GF-REGIONAL-DESTR-008 S1 (hp,
2026-10-22) removed every spare label and saw REPLACE_NODE refused with the
exact no-fallback literal and no BatchReplaceClusterNodes in CloudTrail.

The first live pass corrected three runner judgments that contradicted the
product's documented behavior, not the product: a diagnostic-only chain
(evidence collection plus a validation) whose validation fails while the
fault is held ends `RECOVERED` with reason `diagnostic inconclusive`
(detailed design §6.3), so the runners accept `ACTION_PENDING` (still running)
or that `RECOVERED` and flag `ESCALATED` or a `RECOVERED` without the reason;
the host collector delivers an unchanged reading only once per health summary
(`GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS`) and keeps normal summaries out of
the stored evidence, so a recovery wait now spans two summary periods and
accepts either two fresh delivered samples below the threshold or the
recovered edge's fresh sample together with the product's cleared health
signal for the metric (host CPU/load, storage and host memory alike), read
from the moment the stop was requested because that edge is often delivered
while the stop is still in progress; the host-memory variants (002 and 003
`low_available_memory`) wait for the allocator to reach the threshold (up to
18 minutes on a 2 TiB host) before timing detection; and the
GPU environment runner
treats a collector unit that carries no endpoint override after the restore as
restored to `collector.env`'s endpoint, the state its repoint started from.
A failed validation also counts as judging the held fault when its
`node_failures` name the metric or its `host_diagnostic:<metric>` probe: the
product sets `host_diagnostic_failure` only on health contracts that forbid
hardware escalation, and the load-average validation runs on another one.

A later pass (2026-10-25, 003 `low_available_memory`) failed "no
memory_available_percent finding reached an incident" while the product had
opened LOW_MEMORY_AVAILABLE, and the co-firing `memory_used_percent`
incident, and closed both `RECOVERED` "diagnostic inconclusive" within the
same second, exactly the diagnostic-only chain the variant expects. The node
activity read lists incidents in every state; the runner's matcher did not
take it. A sustained finding's event id ends with the product's rule id
(`-low_memory_available`), and the runner looked for the rule's threshold
family (`-memory`); `high_page_cache` (`page_cache` for HIGH_PAGE_CACHE) and
`scratch_full` (`local_filesystem` for HIGH_LOCAL_FILESYSTEM_USAGE) could
not match either, while the low-utilisation variants already named their
rule. The fakes built the event id from the runner's own argument, so the
variant tests agreed with the runner. The 003 variants now name the product's
rule ids, and a test pins them to `host_health.py`'s sustained rule table and
selects the incident of each sustained finding the product's own policy
mints. The matcher still takes an incident in any state, but only one the read
selected itself: created at or after the read's start, on the read's node and
cluster. The read also adds the incident of any workflow created since its
start however old that incident is (a validated restore names one), and such
an incident is no finding of the run. The regression test replays the live
failure through the real activity read over an in-memory store: the pair
appears already `RECOVERED` on the third poll and the variant passes; an
earlier run's pair pulled in by a restore, the pair on another node, or only
other metrics' incidents leave it failing "no memory_available_percent
finding reached an incident".

003 `high_page_cache` read FAIL in that pass with nothing to show the page
cache had ever been high, and the product opened no incident in the window.
The variant had no oracle: the "accepted" flag of its detection timeline is
the incident match (which the threshold-family rule id above also made
impossible), and the settled wait after it can never accept without an
incident, so it ran out its 12 minutes as well. The variant now samples
`/proc/meminfo` with the node probe's oracle, using the product's formula
`page_cache_percent` = (Cached + SReclaimable - Shmem) / MemTotal. It is
`INCONCLUSIVE` "condition not established" unless that reading held at or
above HIGH_PAGE_CACHE's 90 % for the whole 300-s sustain window; only then
is a missing finding the product's FAIL. The fill itself is unchanged, and
on a 2 TiB host it is unlikely to establish the condition. The sparse file
is 0.92 x MemTotal (1.98 TB), so at least 97.8 % of it must stay cached at
once, and everything else in memory must fit in the remaining 8 %. One
`cat` reads holes CPU-bound at a few GB/s (2.4 GB/s measured
single-threaded on an ext4 host), about 13 to 14 minutes for the 1.93 TB the
threshold needs. Detection stops 720 s after the fill starts and needs 300 s
of that above the threshold. A fixture that can establish the condition
needs three things: a ramp wait timed from the oracle's crossing, as
`low_available_memory` has; parallel readers over disjoint ranges; and a
size computed from MemAvailable with headroom above the threshold.

The next pass with the oracle (2026-10-25, release 13daa08b2227, attempt 5
on the plain-EKS p5en) showed the fill is not the weak layer: one `cat` read
the holes at about 7.5 GB/s and the oracle went from 0.15 % to 90.2 % in 277
s, then held 92.15 % for 505 s, so the condition was established and the
variant read FAIL "no page_cache_percent finding reached an incident". The
product's own records explain the missing finding, and it is the runner's
budget, not the product. The host collector's edge filter ships a sustained
breach once, as the crossing's edge (`sustained:page_cache_percent`, a
reason that stays in `_active_edge_reasons` while the breach holds), and
after that only on its health summaries, 300 s apart with +-5 % jitter
(`collectors/host/collector.py`, `collect_once`); a summary-only batch with
no finding is evaluated but not kept as raw evidence
(`app/ingest/telemetry.py`, `_persist_host_telemetry`). The control plane
mints a sustained finding on the first *later* active sample at least the
rule's 300 s after the first (`store/shared/xid.py`,
`_next_health_signal_state`: `duration >= minimum_active_seconds`), so
HIGH_PAGE_CACHE can only come from the first summary 300 s or more after the
edge, up to about 630 s after it, and VALIDATE_HOST is then judged on
evidence newer than itself, one more summary. The store holds exactly one
active batch for the window, the 15:57:53 edge at 92.15 %, then the 16:06:19
`recovered` batch at 0.15 %: the runner's 720-s budget was counted from
before the ramp, ended 16:05:55 (482 s after the edge), and the fill was
stopped at 16:06:12, which reset the signal before the summary that could
have claimed it. The pass-4 reading of the timelines, "never accepted a
reading", was a misreading: the timeline's `accepted` is whether that poll
found the chain judgeable, not a reading's acceptance.

The variant now waits for its own oracle to read the threshold (a ramp budget
of 1080 s, `INCONCLUSIVE` "condition not established" if the read never gets
there) and only then times detection, over a budget sized from the node's
collector environment: the 300-s duration plus two host health summaries plus
one collector interval plus a 180-s margin for jitter, ingest and dispatch
(1095 s on this site). The fill and the oracle outlive ramp and detection,
`page-cache-start.json` records both budgets and the cadence they came from,
and the variant's window is 45 minutes. The settle wait after a verdict with
no incident is now one read, for every variant: `settled` needs a workflow to
judge, so an empty incident list could never accept and burned its whole
budget. The tests replay the live shape -- the oracle crossing on the third
read, the product's records appearing 40 polls (800 s) after it -- and pin
the budget to the product's rule duration and the fake survey's cadence.

The same cadence rule holds for every variant whose PASS depends on a
sustained host rule, and an audit of the harness on 2026-10-25 found the
other budgets sized the way 003's had been: from fixed numbers (the rule's
sustain plus 300 s plus a few ticks) and counted from the load's start. 001
`cpu_busy`/`recovery`/`checkpoint_phase` (the 120-s `cpu_usage_percent`
rule) had 510 s, the load held for 750 s and VALIDATE_HOST waited for in a
second 510-s wait after the finding, so the validation's summary could fall
after the load had been stopped -- read as "succeeded while the fault was
still held", or as no finding at all; `load_average` had 930 s including
load1's roughly three-minute convergence. 002 `memory_pressure`/`recovery`
already timed detection from the oracle's crossing but with 540 s, and the
allocator's hold (ramp 1080 + 540 + 240) could end before the validation's
summary. 009 `sustained_drops`/`wrong_interface`/`recovery` (the 60-s
`network_drops_delta` rule) had 450 s from the generator's start, and the
required-interface drop-in's deadman (1020 s) was shorter than a full run.
005 `capacity_full` (98 % `shared_filesystem_used_percent`, no sustain)
graded VALIDATE_HOST on the first snapshot that showed the incident, while
the step was still WAITING for the summary after it.

They now share one helper, `general_live_common.sustained_detection_budget`
-- sustain + 2 x `GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS` +
`GPU_FAULT_HOST_INTERVAL_SECONDS` + a 180-s margin, read from the node's
collector environment: 915 s for the 120-s rules, 855 s for the 60-s drops
rule, 1095 s for 003's 300-s rule, and with `summaries=1` 495 s for a rule
with no sustain, whose finding is the edge batch itself -- and one crossing
wait, `await_oracle_crossing`, that times detection from the runner's own
oracle and makes the variant INCONCLUSIVE "condition not established" when
the oracle never crosses within the ramp budget (120 s for the CPU loop,
600 s for load1, 1080 s for the allocator, 90 s for the drop generator). The
stimulus and the oracle outlive ramp + budget + the grading tail, one wait
covers the finding and the terminal validation (`wait_for_incident(...,
validation=...)`), and the injection's `*-start.json` records the budgets
and the cadence they came from. The negative variants (001 `short_burst`
and `invalid_delta`, 009 `short_drops`, 008 `optional_interface`) watch
`sustained_negative_window` after the burst ended -- sustain + one summary +
interval + margin, 615 / 555 / 495 s here, bounded and recorded in the
result -- so the recovered edge and at least one summary after it have been
seen before the negative verdict is read. 005 `capacity_full` waits for
VALIDATE_HOST to be terminal (one summary) before grading its chain. The
variant windows grew with the budgets: 001 cpu_busy and recovery 35 min,
checkpoint_phase 50 (it starts a managed job first), load_average 45,
short_burst and invalid_delta 16;
002 memory_pressure 55, recovery 50, swap_pressure 25 (the allocator's hold
is 2355 s, under the node probe's 3600-s unit cap); 009 sustained_drops 30,
wrong_interface and recovery 25, short_drops 16; the required-interface
window's deadman is max_minutes x 60 + 600, capped at the window probe's
1800 s. The tests replay the live shape for each -- the oracle crossing on a
later read, the product's records 40 store polls (800 s) after it, a PASS
inside the new budget and a FAIL only past it -- and pin every budget to the
product's rule table and the fake survey's cadence. Not changed here, with
the same exposure, are 003 `low_available_memory` (480 s from the crossing
for a 60-s rule plus VALIDATE_HOST; 855 s needed), `scratch_full` (720 s
from the fill for the 60-s local filesystem rule plus the validation) and
`gpu_idle`/`rank_cpu_idle` (300-s rules: a 345-s hold, then 720 s), 006
`sustained_util`/`sustained_latency` (510 s from the load's start for the
120-s disk I/O rules; being moved to the same helper) and 004's
unusable-filesystem diagnostics (420 s for a no-sustain rule plus the
validation's summary, 90 s of slack). Their pass-5 FAIL or INCONCLUSIVE
verdicts may be these budgets rather than the product.

001 `checkpoint_phase` never declared a checkpoint phase: it ran the
`cpu_busy` load and chain with no managed attempt on the node. It now starts
027's managed progress job on the lane node (rank 0, the product's own
training-progress reporter; it needs `--site-file`, otherwise
NOT_RUN(FIXTURE_NOT_READY)), writes `labels.phase=CHECKPOINT_SAVE` into its
progress file and starts the load only once the product's training-progress
state shows that phase (otherwise INCONCLUSIVE within 120 s, no load
started). It passes when the cpu finding stays the diagnostic chain --
FREEZE_EVIDENCE then a VALIDATE_HOST that fails while loaded on the
`cpu_usage_percent` probe, the incident in a diagnostic state -- with no
hardware step and no STOP, restart or isolation planned for the
checkpointing attempt, and the attempt, read after the finding, still
RUNNING, still declaring the phase, its latest heartbeat no older than the
finding. The absent phase correlation stays a note, not a FAIL.

The same review of every diagnostic-only verdict found one more incident
the matcher could not see: 025 `exporter_timeout`, `sustained_unknown` and
`post_refresh_success` waited for a `dcgm_field_completeness` incident with
a host-health event id (`-dcgm_field_completeness-`). The product books a
failed scrape as one finding per error streak, `<first batch>-dcgm-fields`,
and hands it to the node-health path as `gpu-<finding id>`, so its incident
is `inc-gpu-dcgm-<node>-<us>-dcgm-fields`. The metric name is not in it.
The matcher now knows that shape, and the 025 tests mint the event id
through the product's own collection-error code. The other diagnostic-only
waits (001 CPU and load, 002 memory, 004 unusable filesystem, 006 disk I/O,
009 drops) match event ids the product does mint.

The 10:25Z pass of the same day (release 6a95b90fcd82, plain-EKS nodes) read
FAIL for the two attempt-bound 003 variants, and both were runner judgments.
`gpu_idle` reported "VALIDATE_GPU ended WAITING": the first node-activity
read that showed the incident (the third poll, 10:33:16Z) found the
validation step WAITING and the chain was judged off that snapshot; the step
was terminal by the settled wait's second poll, 10:33:47Z. A validation waits
for evidence newer than itself -- GPU_METRICS and HOST_TELEMETRY collector
statuses within the 2-minute sample age, a sample of its host diagnostic
probe observed after the finding, fresh GPU metrics -- and the executor
records it WAITING until that arrives (`adapters/gpu_validation.py`,
`host_diagnostics.py`); a step still running is not a verdict.
`rank_cpu_idle` reported "VALIDATE_HOST ended FAILED: validation failed on
nodes: ... (host_diagnostic:training_rank_cpu_percent)": the validation's
host diagnostic probe re-read the metric while the ranks still slept and
failed naming it -- the product judging the held fault, the same flow as
`low_available_memory` -- but the variant's expectation did not say the
validation must fail, so the failure was the error. Both variants now expect
what the product does with an idle attempt: FREEZE_EVIDENCE then
VALIDATE_GPU / VALIDATE_HOST carrying a `host_diagnostic_probe` on the metric
(at most 5 %), the probe failing `host_diagnostic:<metric>` while the attempt
stays idle, the workflow FAILED and the incident closed RECOVERED "diagnostic
inconclusive" (never ESCALATED, which would hold the job's restart), no
RUN_DCGM_DIAGNOSTIC, no hardware step, the attempt id on the incident; the
tests pin each of those to the product's own code (`diagnostic_probe`,
`validate_diagnostic_probe`, `failure_incident_state`). Every wait that
precedes a validation judgment -- 001 load average and checkpoint phase, the
003 variants, 004 unusable mounts, 006 disk I/O, 009 drops, 025 exporter
timeout -- now polls until the validation has a terminal execution or the
workflow is terminal (`chain_judgeable`), within the same detection budget,
and the judged incidents and workflows are kept in the case directory as
`<wait>-chain.json`. A validation still WAITING when judged reads "did not
finish" with what it waited for, whether or not the expectation says it must
fail (a WAITING step used to satisfy a must-fail expectation). The recorded
runs replay through the real activity probe: gpu_idle's chain appears WAITING
on the third read and FAILED on the fifth and the variant passes;
rank_cpu_idle's appears FAILED on the third and passes; a validation that
passes while the attempt is idle, or never finishes within the budget, fails
the variant.

The managed test jobs of 003, 011, 027, 028 and 029 are thin wrappers over
the image prewarm and the managed workload fixture, and the variant tests
replace both with fakes, so the live pass was the first to start one: 003
handed the prewarm its regional settings object instead of the node name (the
Pod intent was journaled, then could not be serialized, and cleanup refused
"creation outcome is unknown" for a Pod that was never sent), 028/029 handed
it the cluster id, and 027's reporter sidecars made the Pod template
multi-container, which `gpu-training-submit` refuses without
`--training-container`. The wrappers now pass the target nodes and 027 names
its `pytorch` container. The prewarm refuses a node that is not a name or not
a node of the GPU cluster, and both fixtures serialize the request before they
journal its intent, so a failed preparation leaves a journal that proves
nothing was created and its cleanup reports `nothing created`; a create whose
acknowledgement is lost still refuses.
`tests/regional/test_general_live_managed_job_start.py` starts every
managed-job variant with the runner's own settings, fixtures and journals --
only the kubectl subprocess, the store read and the host probes are fakes --
up to its first cluster mutation.

A rerun of a managed-job variant by the same lane then met its own earlier
journals. Every managed-job identity is stable per lane and attempt
(`run_identity`: case number, job suffix, attempt and a digest of the case
directory, e.g. `gl17-op-a6-227ee78d`; 003's `gl3-idle-<attempt>` and the
`gl28..-<digest>-<attempt>` ids of 028/029 likewise), and a journaled fixture
is create-once: an identity whose `.ownership.json` / `.prewarm.json` already
exist in the case directory is cleanup-only. Right for a crashed run, that
refused every deliberate rerun of a variant the same lane had already run and
cleaned up within one campaign attempt -- live 2026-10-26 on the HyperPod lane
hpb, 017.old_pod, 017.cri_partial_rotation and 017.live_pid, each rerun released
by a landing after its fix, INCONCLUSIVE "harness fixture failed:
RegionalFixtureError: an existing fixture is cleanup-only" at the prewarm's
`begin`, while the morning's run of the same variant had completed both
journals; reruns the other lane picked up had another digest and passed, so it
looked sporadic. The identity rule is now rerun-aware
(`fixture_ownership.rerun_identity`): a run uses the base identity, or
`<base>-r2`, `-r3`, ... -- the first without journals in the case directory --
provided every earlier one is closed, where closed is a completed journal or
one never begun that journaled nothing (the fixtures' own proof that nothing
was created); an earlier journal begun and not completed keeps its identity,
so the fixture still refuses it as cleanup-only and a crashed run's custody is
never skipped (its cleanup-only resumption removes what the crashed run left,
and the rerun after that starts fresh under `-r2`). The derivation reads the
journals and writes nothing, and touches no cluster, so plan and execute,
minutes apart in one lane, agree. `VariantContext.identity` memoises the rule
per run and base -- 011's second attempt of a job still names the same job --
and every naming site goes through it: the `_job` helpers of 011, 017, 027,
028/029, 003's `_managed_job`, 024's `start_job` and the restart-ledger id of
`SeededRestartBudget` (the ledger row and the managed job stay one identity; a
variant without a managed job has no journals and keeps its id), and the
GF-ENV-LIVE-010 control task (`env_live_scope.ControlTask`; its run id already
carries the start time, the rule covers a reuse). A derived identity is a
label value of at most 58 characters so its `-a<nnn>` attempt id fits the
63-character label, and the workload name is its own digest
(`managed_workload_name`). The result records `identities` (base to the
identity used).

The run id itself has the same rule over the host-probe ownership records.
Every host-probe Pod of a run keeps a record under
`cases/<key>/host-probes/<pod>.json` (030's spare service under
`spare-service/`), its name a digest of the run id, node, case and probe
script, and the fixture compares the record's whole scope -- the probe
script's digest and the image included -- before `create` looks at anything
else, so a rerun of the same lane and attempt under the stable run id met the
earlier run's record; when a landing had changed the probe script in between,
it refused "host probe ownership record does not match this run" (live
2026-10-26 22:40Z, HyperPod lane hpb, 022.recovery_evidence rerun after its
14:1xZ run, whose record was still open because that run's result was lost to
the coordinator's disk-full event). `rerun_run_id` names the run `base` or
`base-r<n>`: the first run id without records in the case directory, provided
every earlier one's records are closed; a record still open keeps its run id,
so the fixture meets it and resumes or refuses it exactly as before (that live
022 record is open, so its lane stays refused until the record is closed or
removed by hand -- by design). Everything derived from the run id follows:
the host-probe record and Pod names, the node probe's on-node state directory
and deadman units, the DCGM replay and source-replay run directories, the
prewarm's `run_id`. The plan records the run id the execute will use
(`details.run_id`); both derive it from the same case directory and agree.
`tests/regional/test_general_live_rerun_identity.py` runs 028.nonzero_exit
three times in one case directory against the fake cluster (the base, `-r2`,
`-r3`, each a PASS under its own job, the earlier journals untouched), reruns
over a crashed run's open journals (refused as cleanup-only, the crashed
custody removed, the next rerun under `-r2`), the control task's second
construction, every wrapper's journal names, the journal states the rule
reads, and the host-probe side with the real fixture over the in-memory API:
a closed record and a changed probe script give a new run id and a fresh
record, an open record keeps the run id and is refused as before until its
own cleanup-only resumption closes it, and the runner names the run after the
records it finds.

Past that mutation, all nine executable 027 variants then failed before any
fault (live 2026-10-24, plain-EKS lane): "managed workload did not become
healthy", the job's only Pod Running and NotReady for the whole 900-s start
wait while its trainer heartbeated. The progress-reporter sidecar ran the Node
Agent's host install path (`/opt/gpu-fault/current/venv/bin/gpu-fault-collector`)
inside the public PyTorch training image, which ships no product, so the
container never started, `restartPolicy: Never` left it terminated, and a Pod
is Ready only when every container is; it also lacked the control-plane
connection `gpu-fault-collector` refuses to start without. The fixture's Ready
requirement was right -- a Pod whose reporter is dead cannot run any 027
variant. The sidecar now runs `/opt/gpu-fault/executor/bin/gpu-fault-collector`
from the GPU cluster's own runtime image (read from the
`gpu-fault-cluster-executor` Deployment and required to be digest-pinned) with
the URL, cluster id and cluster token from `gpu-fault-regional-connection`, its
CA through `SSL_CERT_FILE`, a writable per-reporter state directory for the
durable outbox, and the in-cluster collector's security context. One more
defect on the same path would have failed the variants next: the runner's
progress write piped its JSON through `kubectl exec` without `-i`, so the
trainer wrote empty progress files (the reporter reported only
`progress_error=JSONDecodeError`). The write now passes `-i`. A start that
still times out names each container that is not Ready with the kubelet's
reason.
`tests/regional/test_general_live_progress_job.py` starts the job with the
runner's own objects against a kubectl fake whose kubelet starts a container
only if its image ships the program it executes and the product CLI's own
start-up accepts the rendered arguments and environment, then writes progress
through the exec path and reads it back with the reporter that start-up built.
Not covered offline: whether a GPU node can pull the runtime image (the
executor's Pods may run elsewhere), and the product-side outcome of each 027
variant.

The first live pass with that job (2026-10-26, plain EKS, release
37db33f254a1) failed 027 `nan` with "training_nonfinite-loss incident is
ACTION_PENDING, expected ['ESCALATED']" while the product did exactly what the
variant expects: the heartbeat at 09:35:53Z minted the incident and its
STOP_WORKLOAD chain within 60 ms, FREEZE_EVIDENCE and STOP_WORKLOADS succeeded
within 4 s, COLLECT_DIAGNOSTIC_BUNDLE (a remote command to the Node Agent) ran
for 22 s, and ESCALATE_SUPPORT completed the workflow SUCCEEDED and moved the
incident to ESCALATED at 09:36:19Z. The runner's detection wait accepted the
first store read that showed the incident at all (09:35:57Z, the bundle step
WAITING, the workflow RUNNING) and the verdict judged the incident state on
that snapshot; the incident later read RECOVERED only because the runner's
post-case step closed the escalation. ACTION_PENDING is the state of a record
whose workflow is still executing, and ESCALATED is the terminal write of the
chain's last step, so no first read can show it. `nan` and
`checkpoint_unverified` now wait for the record to settle before judging the
state -- the incident's workflow terminal (SUCCEEDED/FAILED/BLOCKED/SUPERSEDED)
and the incident out of DETECTED/ACTION_PENDING/SAFETY_PENDING -- within a
budget derived from the product's step cadence: the per-step cap
`GPU_FAULT_WORKFLOW_STEP_TIMEOUT_SECONDS` (600 s, past which the product fails
the step itself) plus one dispatcher poll interval
`GPU_FAULT_WORKFLOW_POLL_INTERVAL_SECONDS` (5 s) per planned step plus the
180-s chain margin, 800 s for the four-step chain, read every 20 s. They
record how long they waited, the reads, the state and the planned and executed
steps (`<variant>-settled.json`, the polls in `<variant>-settle-timeline.json`)
and report a record still in flight at the end of that budget as INCONCLUSIVE
naming the observed state and the executed steps (for example
`COLLECT_DIAGNOSTIC_BUNDLE WAITING`) rather than a FAIL; a record that settled
without the workflow SUCCEEDED -- an ESCALATED reached through a failed or
blocked step, not the planned ESCALATE_SUPPORT hand-off -- is a FAIL that names
the executed steps (stage `settled`). The required-operations, STOP allowance,
checkpoint-disposition and no-restart checks are unchanged. `resumed_progress`
reads afresh after the original rank's new progress and waits for RECOVERED
with the same settle semantics (another settled state is its FAIL, an in-flight
record past the budget INCONCLUSIVE). The scan-driven variants
(`stalled_step`, `reporter_silent`, `step_regression`, `straggler_group`) judge
the planned chain and the finding's classification on the first read and
assert no state: their VALIDATE_FABRIC legitimately stays in flight until newer
progress arrives, which `resumed_progress` is the variant for. The module tests
replay the live sequence through a recording store double (two in-flight reads,
then the settled one), and a scripted workflow record now carries a terminal
`status` and its step executions, as live records do; a record without them
would never settle.

027 `old_attempt` never posted its stimulus: its "foreign" heartbeat was one
more write of the live attempt's own progress file, so the live reporter
posted a heartbeat of the LIVE attempt and "nothing was stopped, no finding"
held by construction. The variant now first requires the live attempt's own
heartbeat in the store (otherwise INCONCLUSIVE: a refusal would prove
nothing), then rank 0's reporter sidecar sends one heartbeat claiming
`<job>-a000` -- the job's id with an attempt number the product never ran,
the live Pod's identity, `numerical_error=true` (the payload that stops an
attempt once accepted) -- through the product's ingest route with the
product's own heartbeat model and collector client (`HEARTBEAT_POST`: the
sidecar's connection, CA and Pod identity, one delivery, no outbox, no
retry). It passes only when the ingest answers `accepted=false`, the store
keeps no progress for that attempt (the records probe's new
`other_progress_states`), no training finding, STOP or restart follows, and
the live attempt stays RUNNING with its own heartbeats still accepted after
the foreign one; an answer that is not the ingest's grading (a transport or
HTTP error, unreadable output) is INCONCLUSIVE. A module test runs the
script against the product's `TrainingHealthService.ingest` and the real
records probe over an in-memory store: refused for the never-run attempt,
accepted for the live one with the same identity. `resumed_progress`'s
progress check returned nothing in every case; a RECOVERED stall now needs a
SUCCEEDED VALIDATE_FABRIC whose recorded `validated_attempt` (the job's
attempt), `validated_rank` (0) and `validated_step` (past the stalled step)
come from the product's training validation (`validate_training`), or the
variant FAILs: the stall was not cleared by the original rank's newer
progress.

The same pass failed 027 `straggler_group` (2026-10-26 09:54-10:06Z, plain EKS)
with "no training_straggler finding reached an incident", and the records show
why no product rule could have fired: the attempt's `progress_states` held rank
0 only (step 100, `progress_group` data-parallel), rank 1's heartbeat never
reached the store, and at 09:57:33Z -- 145 s after rank 0's single write -- the
product minted `training_hang` ("training step stopped advancing") for rank 0
instead. The fixture had run both reporter sidecars in one single-node Pod. The
product binds one rank per Pod: the completion watcher's rank is the Pod's
replica index plus the `gpu-fault.io/rank-offset` that `gpu-training-submit`
stamps per role (`completion_pod_parsing.py`, `training_submit_cli.py`: Master
0, Worker 1), the progress ingest answers `accepted=False` to a heartbeat whose
rank has no container in the attempt's allocation (`training_health.py`
`ingest` -> `_allocation`), and the scan compares only the allocated ranks'
fresh heartbeats, two at least per group (`scan`, `_group_stragglers`). So a
single-Pod job can never show the product a straggler, and a rank written once
and left is a stall by the product's own 120-s rule, which is the correct
grading of what the fixture did. This is a fixture defect, not a product gap.
`straggler_group` is now a two-node variant (`--node-b`, as 011's stall job):
the two-node manifest with rank 0's Master Pod on the lane node and rank 1's
Worker Pod on node-b, each Pod carrying only its own rank's reporter, two Pods
and 16 GPUs expected; without `--node-b` it is NOT_RUN(FIXTURE_NOT_READY). On
every poll (before the scan) the runner now writes both ranks -- rank 1 60
steps behind rank 0 (30 behind the two-rank group's median, above
`GPU_FAULT_TRAINING_MAX_STEP_LAG` 20), both 10 steps further than the previous
poll, so `last_progress_at` keeps moving and neither rank stalls
(`straggler-advance.json`); a new stage `not_stalled` fails the variant if a
`training_hang` finding appears, and when the straggler finding is missing the
tail of every reporter sidecar's log is kept as `straggler-reporters.json`
(the ingest's refusal is answered to the reporter, which logs it, never to the
runner). The module tests render both manifests through the real
`_pin_replicas`/`_install_reporter` (one reporter per Pod, pinned nodes, `--rank
1` and `rank-1.json` on the Worker), start the two-rank job against the
in-memory cluster with a second node (one Pod per node, each rank's reporter
posting that Pod's identity, the trainer's collective at the 16-GPU world
size), drive the variant through a recording store (three polls, both ranks
written each time, the lag constant) and replay the live shape (a hang instead
of a straggler: FAIL naming the stall, reporter logs kept). On plain EKS node-b
is the other lane's node, so the variant needs both lanes' claims at once, as
011's two-node job does.

The two-node `straggler_group` then failed again on HyperPod (2026-10-26
17:30-17:43Z, release 37db33f254a1) with the same "no training_straggler finding
reached an incident", this time with both ranks in the store: rank 0 at step
240 on the lane node, rank 1 at step 180 on node-b, both in `progress_group`
data-parallel, both fresh, a lag of 30 against the pair's median. The product
had graded the pair. A `training_straggler` finding is bound to the rank it
grades -- its `node_id` is that rank's node (`training_health.py` `_claim`) --
and the incident the scan route raises from it names that node alone
(`node_ids`). The fixture's records probe read the incidents scoped to the lane
node (`list_incidents_by_state(..., node_ids={<lane node>})`), so rank 1's
incident on node-b was never listed; the straggler signal stayed active, and a
claim emits on the transition only (`claim_health_signal_transition`), so every
later scan body showed `findings: []`. This is a harness defect, not a product
gap. The product's behaviour is pinned by
`tests/execution/test_general_health_policy_e2e.py` on the real planner: two
Pods on two nodes in one attempt, the fixture's labels and steps, one straggler
finding on the lagging rank's node emitted once, its incident naming that node
only with FREEZE_EVIDENCE and VALIDATE_FABRIC planned and no STOP_WORKLOADS,
and the lane-node-scoped read returning nothing while the two-node read returns
the incident. The records probe (`general_live_progress_probes.py`, split out
of the fixture module for the 1500-line cap) now takes every node the job's
ranks run on, as 011's `HUNG_PROBE` does, lists each incident's `node_ids`, and
records the attempt's observation (phase, first and last observed, containers
with rank, node, Pod UID and container name) and the control plane's clock at
the read, so a binding, phase or clock cause is visible in
`training-records.json` next time. `same_group_only` judges a straggler
incident's `node_ids` against rank 0's node: an incident's `reasons` carry the
finding's text and never the rank, so the former "rank-0 in reasons" check could
not fire. The module tests assert the two-node variant's reads name both nodes
and a single-rank job's read names the lane node, that a straggler incident on
rank 0's node fails the variant, and the scope rule on live-shaped incidents.
A records read never raises over the job's shape: a job that names no nodes
(no `rank_nodes`, or none for a rank) is read on the lane node, and the job
doubles of the 027 test modules carry the real job's `ranks`/`rank_nodes`
surface -- the live20 pre-gate's `reporter_silent` replay ran a
`StoppedReporterJob` without it and crashed in `_job_nodes` before any
verdict; its records read is now asserted to name the lane node.

Every managed job also rendered its manifest's fixed name
(`gpu-fault-single-node-warm-spare`, `gpu-fault-two-node-replace`), so one
leftover refused all later jobs whatever their job id: live 2026-10-25 every
003 managed-job variant failed before any fault with "managed workload name or
job identity already exists", and 003 `ambiguous_ownership`, which runs two
jobs at once, could never have submitted its second. The leftover was the
Suspended PyTorchJob of an earlier 028 `nonzero_exit`. The product had
restarted that job in place (same UID, attempt id `<source>-r-<8 hex>`), and
the case's cleanup refused it ("managed workload ownership or UID changed")
before deleting anything: a journaled fixture can bind no restart, and its
cleanup kept the strict adoption rule. A managed workload is now named
`<manifest name>-<sha256(job id), 10 hex>`, at most 43 characters so the
product's `-r-<8 hex>` restart copy and the training operator's `-master-0`
replica names stay DNS-1035 labels, and every lookup (status reads, the job-id
Pod selector, the `training.kubeflow.org/job-name` Service selector, cleanup)
follows from that name. A journaled cleanup now deletes what the product
restarted of its own submission under the restart-lineage rule the unjournaled
cleanup already had (this fixture's owner nonce and job id, an attempt id
derived from its own, a chain to a UID it created or to a RESTART_WORKLOAD
copy) and journals each such object; a journal a later process reopened keeps
the strict rule. Both halves of a job's cleanup always run, its record lists
every object deleted and which of them were the product's restart, and a
failure carries both halves' records. A create whose name or job id is still
held refuses and names each holder (kind and name, UID, job id, attempt id,
owner nonce, state, and whether it is another acceptance run's leftover);
nothing is deleted on that ground. 003's job ids (`gl3-<variant>-<attempt>`)
repeat every pass, so a leftover of the same 003 job still refuses a later
pass, now by name. `tests/regional/test_general_live_managed_job_cleanup.py`
runs the wrappers, fixtures, journals and runner against an in-memory
namespace with a training operator and the product's in-place and copy
restarts; on the old code all of its cases fail, the in-place restart with the
live cleanup record.

With the names apart, 003 `ambiguous_ownership` then failed on its first job
(live 2026-10-25, as it already had on 2026-10-24 under the fixed name):
"managed workload did not become healthy", the PyTorchJob annotated with the
completion watcher's `inc-attempt-failure-*` incident and its
`workflow-attempt-stop-*/1/STOP_WORKLOADS`, no Pod. The variant runs two
4-GPU jobs on one 8-GPU node, and its 4-GPU rendering changed only the GPU
request: the trainer still ran `torchrun --nproc-per-node=8`, so local ranks
4-7 had no device, torchrun exited non-zero, the product's passive containment
stopped the attempt and suspended the job, and the start wait saw no Pod for
900 s. Each job also kept the manifest's 16 EFA devices, the node's whole
complement, so the second job could never have been scheduled beside the
first; and a job that had run would have heartbeated at world size 8 against
the fixture's expected 4. The 4-GPU rendering now fits the trainer's workers
and the EFA request to the GPU count (8 EFA for 4 GPUs). The start-wait
failure also printed `suspend: None` for a PyTorchJob the product had
suspended, because the snapshot read `spec.suspend`; it reads
`runPolicy.suspend` for a PyTorchJob now. The fake namespace of
`tests/regional/test_general_live_managed_job_cleanup.py` schedules Pods
against the node's GPUs and EFA devices, runs a torchrun only when every
worker has a GPU, and stops a failed attempt as the completion watcher does;
the variant runs through `execute_variant` on it, and the old rendering
reproduces the live failure text.

028 `nonzero_exit` failed in its cleanup (live 2026-10-25, pass 4, plain
EKS): the product restarted the failed job as a copy, the copy's own attempt
failed 33 s later (it runs the same command), and the product's STOP_WORKLOADS
suspended and annotated the copy in the second after the cleanup had retained
it and deleted the source job; the delete of the copy was refused as a change
of ownership and the copy stayed (the leftover PyTorchJob
`...-c098a41205-r-893dc810`). The cleanup custody no longer binds the
product's own bookkeeping on a managed workload -- the `gpu-fault.io/*` step
annotations, the suspend toggle, the Pod's initiator annotation and the
attempt-id relabel of an in-place restart -- while the UID, namespace, owner
nonce, job id, every other label and annotation and the spec still bind it;
and the lineage rule for an unbound restart copy accepts the STOP_WORKLOADS
step key the copy's own containment writes over the RESTART_WORKLOAD key,
since only the product derives a `-r-` attempt id under this fixture's nonce.
The fake namespace reproduces both refusals on the old code (the containment
inside the cleanup and before it).

Two more 028 variants failed in the same pass for want of an injection:
`cleanup_timeout` ran one Pod whose only container ignored TERM and slept,
`controller_stop` one that waited for a stop nobody sent, and both read the
product's records for the whole 900-s decision wait ("terminal was None", "a
controller stop produced None"). The product's TIMED_OUT (`watcher.py`) needs
a detected failure first -- the cleanup deadline starts at the
FailureDetectedEvent -- and a critical rank still not terminated when it
passes, so `cleanup_timeout` now runs two ranks on the node, a Master that
exits non-zero and a Worker that ignores the stop (two 4-GPU Pods), with no
restart budget: the restart is `nonzero_exit`'s and `one_restart`'s proof.
`controller_stop` suspends its own running job the way a job controller such
as Kueue does (`spec.runPolicy.suspend`, fenced on the UID and resourceVersion
the fixture adopted); the operator deletes the Pod, the product reads a stop
by someone else, and the verdict wants the STOPPED terminal and NO_ACTION.
Neither variant proves what the product does with a stop its own control
plane initiates; that is the containment path of the node-fault categories.
The two-rank and suspend injections run through the real wrappers and
fixtures in the fake namespace.

Two more 028 variants passed vacuously and were given their stimulus or
deferred. `old_terminal` runs the managed job in the `exit_once` mode: the
first attempt exits non-zero after 30 s, and the product's RESTART_WORKLOAD
copy -- whose Pod template the product relabels with the `-r-` attempt id it
derives -- reads that id through the downward API (`GFRS_ATTEMPT_ID` from
`metadata.labels['gpu-fault.io/attempt-id']`) and keeps running. Once the
product's RESTART_WORKLOAD execution recorded `restart_attempt_id` and a
watcher pass has observed the copy, `TERMINAL_REPLAY_PROBE` (an API Pod probe
like the 029 ledger harness, `gl28` test attempts only) reads the product's
own stored TerminalEvent of the old attempt and redelivers it through
`CompletionService.handle_terminal`, the entry the `/v1/attempts/terminal`
route calls for the watcher -- nothing hand-made. The verdict wants the
original decision back with `duplicate=true` (same status, recovery plan and
event key), exactly one RESTART_WORKLOAD workflow and one budget deduction for
the old attempt, and the new attempt without a terminal, a decision or a
containment of its own; no decision, no restart copy or no recorded terminal is
INCONCLUSIVE, and the existing managed-job cleanup removes the running copy.
`missing_allocation` is `NOT_RUN(FIXTURE_NOT_READY)`: on Kubernetes the watcher
builds a terminal's allocation from `spec.nodeName` of every terminated
critical container (`completion_pod_parsing.observe_containers` ->
`watcher._terminal_event`); a terminated container always has a node and a
workload FAILED without Pods never reaches a terminal, so a watcher-observed
terminal always carries an allocation, and only a hand-made TerminalEvent
posted to `/v1/attempts/terminal` could lack one -- which the spec (watcher-
observed terminals only) and the lane rules forbid. The product path
(`service._decide` -> `planner.from_missing_allocation`) stays software-tested.

029 `duplicate_event` is the same redelivery against the SAME attempt, so the
product receives one terminal twice: the job runs in `exit_once` mode, the
runner waits for the decision, the recorded `restart_attempt_id` and one
watcher pass (the first delivery settled, the restart copy keeps running so
nothing else moves the records), then `TERMINAL_REPLAY_PROBE` (now in
`general_live_terminal_replay.py`, shared with 028, `gl28`/`gl29` test
attempts only) hands the stored TerminalEvent to
`CompletionService.handle_terminal` again. The verdict reads the attempt's
records before and after: the answer is the original decision with
`duplicate=true`; the stored decision row is unchanged and not rewritten as a
duplicate; exactly one RESTART_WORKLOAD workflow; the restart ledger at one
deduction with the same reservation ids; no new notification for the
attempt's incidents (the attempt probe now returns the notifications of the
attempt's containment and recovery incidents, read-only); the restart copy
untouched -- one decision, one deduction, one notification. Before this the
variant passed without any redelivery.

028 `nonzero_exit`/`one_restart` and 029 `budget_consumed`, `restart_withheld`
and `single_notification` graded the first read that showed a completion
decision. The product records the decision (PLAN_CREATED) soon after the
failure, but dispatches the recovery workflow only once the containment ended
and reserves the budget at that workflow's claim
(`restart_budget_preflight.reserve_restart_budgets`), so the ledger helper
accepted `restart_count` 0 (a reservation not yet taken, or released
unattempted) as "consumed once", an empty recovery history as "stop then
restart", a WAITING or absent RESTART_WORKLOAD as "withheld", and nothing read
the notifications. The variants now wait for the outcome -- every recovery
workflow of the attempt terminal, within the product's own caps (one WAITING
STOP_WORKLOADS and RESTART_WORKLOAD of 600 s each, a dispatcher poll per
claim, the watcher poll and the margin: `settle_bound` without the watcher
replacement, 1335 s), INCONCLUSIVE when it does not settle -- and read what
the product recorded (`general_live_restart_outcomes.py`). Within budget:
one recovery workflow SUCCEEDED with one RESTART_WORKLOAD SUCCEEDED naming the
copy's `restart_attempt_id`, the containment's STOP_WORKLOADS succeeded once,
and the ledger holds exactly that workflow's one reservation (`restart_count`
1); `one_restart` now injects a single non-zero exit (`exit_once`, the copy
keeps running), requires the watcher's RUNNING observation of the copy and
re-reads one watcher pass later (still one restart, one deduction, the copy
without a terminal, decision or containment of its own). With budget 0 the
preflight withholds the restart (`withhold_exhausted_restart`): the variants
need one RESTART_WORKLOAD FAILED for `RESTART_BUDGET_EXHAUSTED` at 0 of 0, no
restart, restart attempt or reservation, the containment's STOP still
succeeded and no new training job; a recovery that planned no
RESTART_WORKLOAD never reached the budget gate (INCONCLUSIVE).
`single_notification` also needs exactly one budget-exhausted notice of the
job (`<cluster>/<job>/restart-budget-exhausted/0`) among the attempt's
incidents' notifications, the one the withheld restart names
(`details.notification_id`), still the only one a watcher pass later. A
missing decision, or one that planned no restart, is graded at once (FAIL),
with no wait. The managed failing job's fixture moved to
`general_live_fail_job.py` (the 1500-line cap); its names stay importable
from `general_live_attempts`.

GF-GENERAL-LIVE-030 (warm spare unhealthy) is the same story on the HyperPod
lane: its `SpareSession` keeps the spare node *name* in `self.node` and also
defined a `node()` accessor, so the instance attribute shadowed the method and
every `session.node()` / `self.node()` tried to call a string (live 2026-10-24,
`not_ready` and `stale_metrics`, `TypeError: 'str' object is not callable`). The
accessor is now `node_snapshot()`; `stop_kubelet` also passed the deadline as an
ISO string and an empty plan digest to the service window, both of which the
rename then exposed, and now passes the deadline `datetime` and a real digest of
the approved stop. The declared spare is always released with an auditable
receipt: the release cleanup is registered before the declaration can change the
cluster, reads the on-disk record to decide whether anything was in fact
declared, and the receipt records `released`/`restored` and the declared-spare
set the node is now out of; a declare refused before any cluster change cleans
up as "nothing declared" and leaves the node untouched. Driving `not_ready`
through the real host path then exposed one more layer: the on-node service
window that stops kubelet (with an independent restart timer, exactly what
`SPARE_KUBELET_STOP` authorises) is granted by `warm_spare_node_probe` only to
the DESTR-008 service-window fixture identity (`case_id` `GF-REGIONAL-DESTR-008`,
the shared fixture DESTR-008 introduced). `stop_kubelet` had stamped the window
with its own running-case key, which the probe rejects as "service window is not
a DESTR008 fixture"; it now opens the window under that fixture identity (the
acceptance evidence still records the running 030 case, and `run_id` keeps this
run's window and journals distinct from a real DESTR-008 run on the same node).
`tests/regional/test_general_live_spare_session_start.py` drives all four
executable 030 variants with the runner's own objects -- only the kubectl
subprocess, `cpu_python` and the host probes are fakes, and the warm-spare node
patches apply in memory so a spare is really declared and released -- up to the
first host-probe mutation, asserts the release bookkeeping, and pins the window
to the fixture identity `warm_spare_node_probe.binding_key` grants.

The kubelet undo of 030 `not_ready` is awaited through the node, not read
through the stopped kubelet (`service_window_node_restore.py`). Every request
the service window makes to the node (`restore-service`, `cleanup`) travels
over `kubectl exec` into the host-probe Pod, and the API server hands an exec
to that node's own kubelet; with kubelet stopped the channel is gone, the exec
returns no output and the fixture raises `HostProbeMissingResponseError`, which
says nothing about the node. Live 2026-10-27 (plain EKS, release 37db33f254a1,
attempt 7) the explicit undo was issued about 80 s into the stop, got exactly
that, and the variant judged `restored: the candidate is not Ready after the
release (Unknown)` and `cleanup did not prove the node clean` at 14:19:20Z,
while the window's own on-node restore unit brought kubelet back at `restore_at`
14:27:22Z and the node was Ready at 14:27:23Z -- the node-side deadman worked;
the harness judged the hand-back eight minutes early (the next EKS case started
on a NotReady node and ended NOT_RUN(FIXTURE_NOT_READY)). The undo now attempts
the explicit undo first (with kubelet up -- stop timer not yet fired, restore
unit already done, HyperPod or EKS alike -- it cancels or closes the window at
once); a missing response or transport error while the node is not Ready is
recorded as the expected shape, the undo is deferred to the on-node unit, the
node is polled through the API until Ready with the deadline `expires_at` (the
unit's authority bound, `restore_at` + 180 s) + 120 s, and the explicit undo is
then repeated over the restored channel -- the window's CLOSED report (kubelet
running, stop and restore units quiescent, `restore_reason` naming who restored
it) is the proof; then every Pod on the node is awaited Ready before the spare
is released and `restored` is judged. Only a node that has not returned by that
deadline fails the undo, with a message naming `restore_at`, `expires_at`, the
wait deadline and the last node condition observed (`spare-kubelet-restore.json`
carries the poll timeline). The oracle is tightened to the claim -- the product
graded the spare while kubelet was down: the node is read again after the grade
and must be not Ready on both sides of it, and the grade must precede
`restore_at` (`spare-grading-window.json`), else INCONCLUSIVE. The sibling
variants (`stale_metrics`, `configuration_error`, `reserved_spare`) never stop
kubelet; GF-REGIONAL-DESTR-008 S3 always waited for the node Ready before its
`restore()` and did not have the defect.

The rerun of 030 `not_ready` on the next release was refused by the earlier
run's custody, not by anything on the node. Live 2026-10-28 (plain EKS, release
e78e0049b586, the same lane and attempt) it ended INCONCLUSIVE `service
controller identity or state changed` 8 s after declaring the spare. That
check compares the local controller journal's scope, not a systemd identity:
the 2026-10-27 run's undo (before `restore_through_node`) had left its journal
at CLEANUP_REQUIRED and its host-probe record open; an open record keeps its
run id (`rerun_run_id`), so the rerun computed the same journal path, and the
journal's scope differed in `release_id` and `maintenance_expires_at` -- values
only the execution that wrote it holds, so it can neither resume nor reuse that
window. On the node the rerun touched nothing: kubelet still ran the invocation
the earlier window's restore unit had started at its `restore_at`, while that
window's drop-in, three units, kubelet claim, window directory and host-probe
script copy remained, with its host-probe Pod (DeadlineExceeded) and ConfigMap
in the cluster. On HyperPod (2026-10-24/25) a window left the same way had
already turned the next run's window on that node into a second one nobody
could close: its `prepare` refused the claim only after writing its own
journal, and its cleanup stopped at the same claim. Now 030 `not_ready` reads
`spare-service/` first and, before anything is declared, retires a window a
journal there still holds open on the candidate (below) -- or, when it cannot be
proven stale, is BLOCKED(FIXTURE_NOT_READY) naming the journal, the window's
drop-in, units, claim and directory, its host probe and every failed proof; the
controller's journal refusal names each differing scope key (values for ids and
the expiry only) and what the journal records; the warm-spare probe's snapshot
reports the service's claim holder and the controller refuses a claimed service
before anything is bound, so the refused window publishes nothing and closes;
a missing host-probe response names the probe Pod, the node, where kubectl said
the exec channel broke (a fixed label, never kubectl's stderr) and the node's
Ready condition; and the close over the restored channel is retried a few times
before failing as `NodeRestoreUnconfirmed`.

Retiring a window left like this is a harness fixture step, through the
harness's own host probe like every other fixture action
(`service_window_retire.retire_journal`; the 030 `not_ready` pre-step, and
`scripts/e2e/regional/retire_service_window.py` for journals no pending 030 run
will reach -- plan-only by default, `--execute --confirm
RETIRE_STALE_SERVICE_WINDOW` with a maintenance window). It acts only on proof,
and any failed proof is a named refusal with nothing changed: locally the
journal is CLEANUP_REQUIRED or CREATING for that node, supervision was never
lost, its binding is valid and its own, no process holds it, the Node still has
the recorded UID, and the earlier host probe's Pod and ConfigMap are proven that
record's by labels, owner annotation, script digest, UID and node; on the node a
dedicated retire probe (the warm-spare probe's own source plus
`probes/warm_spare_retire.py`, one stdlib-only script) proves the host journal
RESTORED (or never published, or already closed), the service running on the
machine the window was bound on under exactly the invocation the window restored
(its start receipt), every artifact it removes named by the window's key and
matching its publication receipt -- drop-in `90-gpu-fault-destr008-<key32>.conf`,
the three `gpu-fault-destr008-<key32>-*` units (inactive, no job, process or
cgroup member), the claim naming that window, the window directory holding only
the window's own files -- no other live window or drop-in for the service, and
the earlier probe's `/run` script copy carrying that probe's script digest. The
baseline fingerprints a later node Agent wave rewrites (`node-agent.env`,
`/opt/gpu-fault/current`) are deliberately not compared: every release changes
them, and the window's own cleanup refusing exactly that is what left it stale.
The actions: mark the host journal CLOSING with a retirement marker, remove the
units and the drop-in by receipt, `systemctl daemon-reload` (the service is not
restarted, and that is verified), remove the claim and the script copy, mark the
host journal CLOSED and remove the window directory with the journal last; then
delete the earlier probe's Pod and ConfigMap under UID preconditions, close its
ownership record, clean the retire probe, and mark the controller journal CLOSED
with a retirement record (who -- the operator identity --, when, by what, the
host's proofs). An interrupted retirement resumes from its marker. After it the
variant opens its own window under the next free run id (`<run id>-r2`): the
retired journal is create-once under the old one. A window that published
nothing (HyperPod 2026-10-25: `prepare` met the claim after writing its own
journal) retires the same way without touching the claim holder.

030 `stale_metrics` then failed identically on both clusters live 2026-10-27
(HyperPod 14:47Z, plain EKS 14:51Z, release 37db33f254a1): the first step after
the declaration -- the metrics pause -- ended `HostProbeError: host probe failed
(exit 1; error ProbeError 6420bbc8692c83af); output withheld`, INCONCLUSIVE,
with neither `spare-metrics-pause.json` nor `probe-error.json` in the case
directory. The digest is the first 16 hex digits of the sha256 of the node
probe's `pause seconds is outside 60..1500`: `SpareSession.pause_metrics`
shipped the variant's budget, `max_minutes * 60`, as `unit-pause --seconds`, and
once the 030 spec budget grew from 25 to 30 minutes (2026-10-26) that was
1800 s, past the pause module's 60..1500 s bound, so the probe refused before
stopping anything. Nothing provider-specific was involved: on both providers the
metrics collector is the same host systemd unit,
`gpu-fault-metrics-collector.service`, paused by SIGSTOP under the pause drop-in
(the dcgm-exporter container is 025's target, not this variant's), and the
2026-10-25 HyperPod attempt had paused it with the then 1500 s budget. The pause
is now sized as 031 sizes its pauses, from the threshold it waits for:
`STALE_HOLD_SECONDS` (460 s, the 420 s GPU_METRICS threshold plus one scan) +
120 + 300 = 880 s of on-node deadman (`METRICS_PAUSE_SECONDS`, pinned by test
inside `general_live_node_pause.MAX_PAUSE_SECONDS` = 1500); a unit that did not
stop is INCONCLUSIVE with the resume still queued; and
`general_live_fixture.pause_bound` refuses an out-of-bound `--seconds` by name
on the runner before the pause module ships. The second gap: the node probe's
failure line carried `error_code`/`error_site` but no `error_message`, so the
host fixture's `probe-error.json` record never fired for it, and because the
pause commands are dispatched through `_shipped`'s lambda while
`exception_site` admitted only the probe file's own frames, the site landed on
`<lambda>`, which the fixture's whitelist drops -- the exception text showed
the digest alone. The failure line now carries `error_message` (truncated on
the node, redacted by the fixture) and `exception_site` admits the installed
pause module's frames, so the text reads `error ProbeError 6420bbc8692c83af at
unit_pause:NN; detail in probe-error.json` and every node-probe failure leaves
`probe-error.json` in the case directory. `tests/regional/test_general_live_probes.py`
pins the live digest to the refused budget and drives the recorded failure
through the host fixture; `tests/regional/test_general_live_spares.py` drives
the real session's pause over a scripted node probe.

027 `reporter_silent`'s injection stopped nothing live 2026-10-26 (plain EKS,
release 37db33f254a1): the runner's `kubectl exec -c progress-reporter-0 --
/bin/sh -c 'kill -STOP 1 || true'` targeted the reporter sidecar as its
container's PID 1, and pid_namespaces(7) drops a SIGSTOP sent to a PID
namespace's init from inside that namespace (`kill` returns 0). The reporter
kept posting `step=10` every 15 s for thirteen minutes, and the product graded
the stall it was shown -- `training_hang`, "training step stopped advancing",
the `stalled_step` contract -- which the variant read as "reporter silence
produced an NCCL/hang finding instead". The product was right; the fixture had
injected nothing and never checked. The reporter container now runs a
`/bin/sh` launcher as PID 1 with the reporter as its child and its PID on the
container's `/tmp` emptyDir; `stop_reporter` SIGSTOPs that PID and prints its
`/proc/<pid>/status` State, which `reporter_silent` journals
(`silent-stop.json`) and judges -- a reporter not `T (stopped)` makes the
variant INCONCLUSIVE at the stop rather than FAIL on the product.
`tests/regional/test_general_live_progress_reporter_stop.py` runs the rendered
launcher and stop script against a real local process through the kubelet fake
of `test_general_live_progress_job.py` (the reporter's stand-in is the
launcher's child, the stop lands, SIGTERM ends the launcher at once) and
drives the variant over fakes for both outcomes. Not covered offline: the
kernel rule itself on the node's container runtime (the launcher removes the
dependency on it rather than testing it).

## GF-GENERAL-LIVE DCGM / GPU-inventory replay (phase 2)

Phase 2 adds the DCGM-field and GPU-inventory categories (019 ECC/row remap/
page retirement, 020 NVLink counters, 021 NVLink fabric state, 022 PCIe
replay, 024 composites, 026 GPU identity, and the five 025 field variants
phase 1 left `FIXTURE_NOT_READY`). The injection is `live-replay` only: one
raw reading changes and the real collector parser, transport, grading,
database, scheduling and approved node actions stay real. It is driven by a
reviewed DCGM-field replay tool (`tools/gpu_dcgm_field_replay.py`, its own
per-category field allow-lists, separate from the environment tool's) and a
reviewed `nvidia-smi` inventory pass-through wrapper
(`tools/nvidia_smi_inventory_replay.py`), both installed under `/run` with a
sha256 check and run loopback-only / first-on-PATH by one host controller
(`probes/general_live_dcgm_probe.py`) that repoints the single metrics
collector through a `/run` drop-in with a deadman restore. A replay only
proves the processing chain, never the physical fault, so every verdict is
`proof_scope: live-replay` and the module tests
(`tests/regional/test_general_live_gpu_*.py`, `tests/test_gpu_dcgm_field_replay.py`,
`tests/test_nvidia_smi_inventory_replay.py`) run the verdict logic over
scripted product records with fakes -- they are not proof of a physical
experiment.

The fixture's first live contact (2026-10-25, plain EKS, release
6a95b90fcd82, 019 `corrected_ecc` on ip-10-0-43-5 and `volatile_dbe` on
ip-10-0-32-162) failed inside the fixture before any product evidence:
`repoint-collector` had armed the deadman, written the drop-in and restarted
the collector, then judged the repoint failed because it read
`/proc/<MainPID>/environ` as soon as the `Type=simple` unit was active -- at
that moment the process is still systemd's pre-exec child (the sandbox,
`ProtectSystem=strict` over every mount of a Kubernetes node, is built between
fork and exec) and carries systemd's environment, not the unit's. The runner's
cleanup then skipped `restore-collector` (its `repointed` flag is set only when
the probe call returns), `stop-proxy` rightly refused while the collector read
the replay, and the drop-in, proxy unit and run directory stayed until the
deadman fired. Three rules now hold, pinned by
`tests/regional/test_general_live_gpu_probe_cleanup.py` and
`test_general_live_gpu_replay_cleanup.py` over a recorded fake systemd: the
environment of a restarted collector is read only once the process has exec'd
(bounded settle window); `DcgmReplay.close` runs the probe's idempotent
`close-run` whatever step raised and names every object the probe still finds
(the deadman stays armed while anything is left); and the survey refuses to
start -- naming each drop-in, unit, run directory and deadman fire time -- while
an earlier run's objects are on the node, reclaiming only this fixture's own
leftovers whose deadman has elapsed, never a pending one and never a foreign
drop-in or directory.

The second live contact (2026-10-25 11:09-11:29Z, the same release, both
plain-EKS nodes) got through the repoint and ended every 019 replay variant
(`corrected_ecc`, `remap_failure`, `volatile_dbe`, `aggregate_dbe`,
`pending_supported`, `post_reset`) INCONCLUSIVE "the replay never served
`<GPU>/<field>`" about 30 s after `arm`, with the node clean afterwards. The
replay journals held exactly two armed reads, both in the plan's three
pass-through baseline reads (`baseline()`, which serve nothing by design): the
collector scrapes every 15 s and the first replayed reading comes with its
fourth armed scrape, but `arm_and_confirm` judged the journal at once. Nothing
below the runner failed -- the proxy unit, the drop-in, the collector's reads
of the replay and the close are all in the receipts -- and no product record
was ever consulted; the variants carry no product evidence either way.
`arm_and_confirm` now waits for the first replayed reading
(`DcgmReplay.wait_served`, polled through `audit-read`, bounded by the plan's
leading pass-through reads plus three scrapes at the repointed collector's own
`GPU_FAULT_METRICS_INTERVAL_SECONDS` -- read from its process environment in
the repoint receipt, default 15 s -- plus 90 s; 180 s on this site), ends at
once on a replay abort, and a deadline names the armed reads the journal did
hold and the plan step they reached, so "the collector never scraped the
replay" reads differently from "the step was not reached yet"
(`tests/regional/test_general_live_gpu_replay_wait.py`, over the live journal
shape). Every replay-based DCGM category (019-022, 024, 025) goes through the
same gate; 025's keyless variants keep their own settle hold. All six 019
variants are to be re-run; they still have no product evidence.

The reset-driving variants need an explicit approval and an exclusive node:
019 `pending_supported` / `post_reset` need `--approve GPU_RESET` (a
single-GPU service-window reset); every 021 finding drives a full GPU +
NVSwitch reset, so those variants need `--approve GPU_FULL_FABRIC_RESET` (and
`NODE_REBOOT` for `recovery_failure`) and run on each lane's exclusive node on
both clusters (the `X` flag holds every lock, so a plain-EKS lane's node is
alone on its cluster while the fabric resets). 022's XID-join variants replay
the DCGM last-XID gauge only as
XID 32 (`--approve DCGM_XID_GAUGE`); 019 `remap_failure` needs
`GPU_FIELD_DIAGNOSTIC`; 019 `pending_unsupported` (no legacy GPU) and
`observe_profile` (no OBSERVE reset profile) are `CAPABILITY_ABSENT`.

022's three XID-join variants (`same_gpu_xid`, `wrong_gpu_xid`,
`expired_xid`) have no live evidence yet. On 2026-10-26 (plain EKS, both
lanes, release R13g) all three ended INCONCLUSIVE at `deploy-proxy` --
`HostProbeError: dcgm deploy-proxy: host probe failed (exit 1; error
ProbeError ae2c099c21f98eb7 at tool_command:800); output withheld`. The
node-side controller (`probes/general_live_dcgm_probe.py`) accepted only
`[a-z]{3,16}` as the field tool's `--profile`, and these are the only plans on
the `pcie_xid` profile, the one underscored name in
`tools/gpu_dcgm_field_replay.py`; the neighbours' `pcie` and `composite`
deployed. The digest is the first 16 hex of the refusal sentence's sha256; the
probe sent no message and no `probe-error.json` was written. The controller
now accepts the tool's profile-name shape (`PROFILE_NAME`), the runner's
`plan()` checks a profile against both the tool's `PROFILES` and that shape
before the node is touched, a refused deploy names a fixed reason token in the
error (`reason profile_shape` / `port_busy` / `unit_down` / `no_status` /
`armed_at_start` / `source_missing`) and the probe's message lands, redacted,
in the case's `probe-error.json`
(`tests/regional/test_general_live_dcgm_deploy_profile.py`). The trio is to be
re-run on both clusters.

022's XID-join trio then ran on both HyperPod lanes (2026-10-26 14:17-14:21Z,
release R13g, harness live18) and ended INCONCLUSIVE at the replay's first
armed read -- `the replay aborted: targeted reading absent upstream; refusing
invented samples: GPU-80f702c5-…/xid_last_error` (`same_gpu_xid` and
`expired_xid` on hp; `wrong_gpu_xid` on hpb, GPU-036afe17-…) -- and the same
three were held on plain EKS, whose exporter publishes the same field set. The
node was not the cause: the survey saw 28 `DCGM_FI_*` names on each of the 16
GPUs and `DCGM_FI_DEV_XID_ERRORS` on none, although
`deploy/dataplane/dcgm-counters.csv` line 14 requests it. dcgm-exporter
publishes a field only when DCGM returns an OK, non-blank value: tag 4.8.4
(2026-09-18; the deployed image is
`nvcr.io/nvidia/k8s/dcgm-exporter:4.6.1-4.8.4-distroless`),
`internal/pkg/collector/gpu_collector.go`, `toString` (line 1022: `if
value.Status != dcgm.DCGM_ST_OK { return skipDCGMValue }`, `if isInt64Blank(v)
{ return skipDCGMValue }`) and `toMetric` (lines 931-940: `// Filter out
counters with no value and ignored fields for this entity` … `continue`),
https://github.com/NVIDIA/dcgm-exporter/blob/4.8.4/internal/pkg/collector/gpu_collector.go
(`internal/pkg/collector/utils.go` `isInt64Blank`: "These values indicate no
valid data is available."). And DCGM fills field 230 only from an XID event:
v4.6.1 `dcgmlib/src/DcgmCacheManager.cpp` lines 13759-13761 (`case
DCGM_FI_DEV_XID_ERROR: case DCGM_FI_DEV_NVLINK_ERROR: break; /* These are
handled by the NVML event thread (m_eventThread) */`) and `RecordXidForGpu`
(line 7535, `AppendEntityInt64(threadCtx, eventData, 0, now, expireTime)`, on
`nvmlEventTypeXidCriticalError` and from the kmsg parser),
https://github.com/NVIDIA/DCGM/blob/v4.6.1/dcgmlib/src/DcgmCacheManager.cpp;
`dcgmlib/dcgm_fields.h` lines 803-806 define it as `/** XID errors. The value
is the specific XID error */ #define DCGM_FI_DEV_XID_ERROR 230`
(`DCGM_FI_DEV_XID_ERRORS` is the compatibility alias, line 3659), and the DCGM
API reference "Field Identifiers"
(https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html,
"Last updated on 2026-06-17") carries the same sentence. So on a healthy GPU
the gauge's absence *is* its reading -- the product knows
(`src/gpu_fault/adapters/gpu_validation.py`: "DCGM omits XID_ERRORS on healthy
GPUs on some driver/DCGM combinations"; deployment manual A.2: "an optional
persistent-state supplement and is therefore not listed as a default hard
requirement of the exporter") -- and a plan that must put it could never find
a real sample to derive from: the trio could not arm on any healthy node. The
vendor text does not itself say "blank until an XID"; that follows from the
cache-manager source and the live scrapes.

A plan may now declare, per field and only for the tool's `ABSENT_WHEN_HEALTHY`
allow-list (the gauge alone), `fields: {"xid_last_error": {"absent_when_healthy":
true}}` -- written explicitly into the plan JSON on the node -- so the baseline
accepts the missing reading; a `set` then synthesises the line from the same
GPU's own published label set (its `DCGM_FI_DEV_PCIE_REPLAY_COUNTER` line first:
gpu, UUID, pci_bus_id, device, modelName, Hostname, DCGM_FI_DRIVER_VERSION, plus
the `err_code` / `err_msg` labels the exporter adds to that value, `xid_errors.go`)
under the exporter's `# HELP DCGM_FI_DEV_XID_ERRORS Last XID error code` /
`# TYPE … gauge` header; a real appearance of the gauge upstream aborts the
run like any real change; every other field keeps the "refusing invented
samples" refusal; and the runner's `ComponentPair` accepts the gauge as an
unpublished companion (`absent_when_healthy` in the evidence). Two product
facts shape the plans. The product mints a gauge-change event only against a
stored baseline (`observe_xid_metric`: `previous is not None and xid > 0 and
xid != previous.xid`) and baselines a first-seen value silently, so a first 32
on this exporter -- or a 32 served again after an earlier run left the
baseline at 32 -- would join nothing; every plan therefore serves the gauge at
`0` (the product's own "no XID" value; whether any exporter generation
publishes `0` while healthy is not stated by the vendor text read) beside an
under-limit PCIe step for three delivered reads before the 32, and
`expired_xid` holds the 32 alone for `expired_gap_reads` (4 reads at 15 s, one
more than the 45 s correlation window) before its ramp. And the synthesised
line vanishes with the step that stops setting it, the disarm, the deadline or
the plan's end, where the real exporter keeps the last XID published for as
long as DCGM's cache holds the sample -- its watch asks for the legacy
10-minute keep-age (dcgm-exporter 4.8.4 `internal/pkg/devicewatcher/const.go`
lines 23-24: `maxKeepAge = 600.0 // How long to keep data for this field in
seconds`, `maxKeepSamples = 0`; `device_watcher.go` line 160: "preserves the
legacy 10-minute age and unlimited-sample retention policy"); whether DCGM
drops the lone event sample at that age was not verified, and a hostengine
restart clears it: the plans serve it through every later step, both
oracles are judged during the held step and the recovery wait reads only
`pcie_replay_total` findings, so the vanishing changes no verdict -- its one
trace is the product's gauge baseline left at 32, which the next plan's `0`
resets (`tests/test_gpu_dcgm_field_replay.py`,
`tests/regional/test_general_live_gpu_links.py`). The trio is to be re-run on
both clusters.

021's three no-finding variants (`before_window`, `ungraded_product`,
`monitor_restart`) have no live evidence yet either. On 2026-10-26 (HyperPod,
the hpb lane's exclusive node, release R13g, harness live17)
`before_window` ended INCONCLUSIVE at `deploy-proxy` with `reason unit_down`
and a `probe-error.json` that said only "the replay unit did not stay up". The
node was not the cause: the three plans' held steps asked 400 reads
(`monitor_restart` 800) while `tools/gpu_dcgm_field_replay.py` serves at most
`MAX_READS_PER_STEP` 200 per step (and `MAX_TOTAL_READS` 400 per plan), so the
tool refused the plan at its start (`error: reads must be an integer in
1..200`, exit 2) and the transient unit died; the `nvlink` plans (3/40 reads)
on the same node had deployed minutes earlier. The runner's `plan()` now runs
the tool's own `validate_plan` before the node is touched (a refusal is
INCONCLUSIVE with the tool's sentence), the held steps' read bounds come from
`held_reads(interval, seconds)` -- twice the reads of the hold at the
collector's interval plus four, refused rather than clipped above 200 because
a replay ending under a running hold would grade the real reading and pass
hollow -- and on `unit_down` / `no_status` the controller puts the unit's
`Result` / `ExecMainStatus` and the last 40 journal lines (300 chars each,
redacted by the fixture) into `probe-error.json`
(`tests/regional/test_general_live_fabric_plan_bounds.py`). The three are to be
re-run on both clusters.

021 `recovery_failure` graded the product's designed reboot as a failure (live
2026-10-27 05:04-05:26Z, the HyperPod lane's exclusive node, release R13g,
harness int/r13c-live23: FAIL "a fabric that stayed failed after the reset did
not escalate/hold isolation"). The record shows the product right: the replay
kept serving FAILURE (4) after RESET_ALL_GPUS_NVSWITCHES, the post-reset
VALIDATE_GPU failed at once (`gpu_fabric_manager.fabric_validation`: a FAILURE
state fails immediately, 1/2 wait as training for the window),
`orchestration/escalation.next_rung` sent the failed validation to
RESTART_NODE and `execution/branch_escalation` appended that rung with its
VALIDATE_GPU/HOST/FABRIC + RESTORE_SCHEDULING tail to the same record (a flat
single-node plan is adopted into a one-branch DAG). RESTART_NODE is a reboot
of the same instance on both providers (HyperPod `BatchRebootClusterNodes`,
plain EKS EC2 `RebootInstances`; `node_provider.restart_vm_effective_action`):
the node came back on boot id b7e017d9 (Ready 05:19:58Z), every replay object
lives under `/run` and went with the boot, the post-reboot validations graded
the node's real, trained fabric and the incident RECOVERED. The oracle asked
the settled record for ESCALATED / QUARANTINED -- the ladder's end state,
reached only when the fabric is still failed AFTER the reboot -- which a
`/run`-based replay cannot show on either provider and must not be made to
show (a replay that survives the boot drives the product into a
reboot/quarantine loop: 024 `wrong_scope_source`, 2026-10-26 14:54Z). Now
`recovery_failure` grades the escalation on the record the product writes
before and at the rung (`held_failure_errors`, stage `recovery_failure`): the
post-reset validation FAILED on the held fault, the node was not released
while the fabric read failed (no RESTORE_SCHEDULING SUCCEEDED before
RESTART_NODE, none at all without a rung) and the rung followed -- a
RESTART_NODE execution, or an operator hand-off (ESCALATED / QUARANTINED) on a
site that does not reboot; it then waits for the record to settle and judges
the exit (`escalation_exit_errors`, stage `escalation_exit`): exactly one
RESET_ALL, at most one RESTART_NODE, no REPLACE_NODE, a SUCCEEDED RESTART_NODE
that changed the boot id, and a settled record whose incident matches it
(SUCCEEDED -> RECOVERED with no residual isolation on the node; FAILED/BLOCKED
-> ESCALATED/QUARANTINED). The evidence file `fabric-escalation.json` carries
the incident and workflow states, every reset/validation/rung/release
execution and both boot ids; the notes say the post-reboot branch is
unobservable by design. The five reset-only `X` variants share the exposure:
a held read not consumed before the quiesce is served to the first scrape
after RESTORE_GPU_SERVICES, and for `failed` (state 4) that fails VALIDATE_GPU
at once and the product reboots the node (1/2 wait as training, which the
advance normally beats). Their reset-done wait now polls every 10 s, the
`advance` retries on `HostProbeError` for up to 180 s while kubelet is still
down after the quiesce (INCONCLUSIVE with the reason if it never lands), and a
record that carries a RESTART_NODE rung or a changed boot id is INCONCLUSIVE
with the reason -- never PASS on a reboot, never FAIL on the product's
designed response (`reboot_exposure`); the chain check includes
MARK_UNSCHEDULABLE, the cordon of the product's catalog chain.
`recovery_failure` is to be re-run on both clusters (no plain-EKS record yet),
the five reset-only variants too; no 021 variant becomes NOT_RUN by this
change (`tests/regional/test_general_live_fabric_reboot.py`).

021's advance after the reset is refused by the tool, not unanswered by the
node, on a plain-EKS node (live 2026-10-27 13:14-13:28Z, ip-10-0-43-5, release
R13g, harness int/r13c-live27; `in_progress` repeated it at 13:44-13:59Z).
`not_started` ended INCONCLUSIVE "the node did not answer the probe in 13
attempts over 187 s" while every attempt was answered: `probe-error.json`
shows `loopback:430`, "loopback POST failed: HTTP Error 409: Conflict", the
same from 13:25:19Z (8 ms after reset-done was accepted) to 13:28:26Z. The 409
is the replay tool's own answer -- `tools/gpu_dcgm_field_replay.py` `do_POST`
turns the `ValueError` of `Replay.advance()` into 409, and `Replay.advance()`
says "the replay is not on a step" whenever `Replay.active()` is `None`. The
tool's journal (the case's `oracle.jsonl`) shows why: the collector's first
scrape after RESTORE_GPU_SERVICES read one GPU's real `fabric_manager_status`
2.0 -- Fabric Manager re-training after the reset -- against the baseline 3.0,
so `Replay._check_baseline` aborted ("a real reading is never masked"); from
then on `active()` is `None` for good, the real readings pass through and no
advance can land; the old `_advance_when_reachable` took every
`HostProbeError` for an unanswered node and retried for 180 s. The collector
itself pauses during the quiesce on BOTH providers (verified by product lane
b208): `gpu-fault-metrics-collector` is in the Agent's DEFAULT_QUIESCE_SERVICES
(`src/gpu_fault/node_agent/common.py`, lines 77-85), an eks-managed quiesce protects
only kubelet and containerd (EKS_MANAGED_PROTECTED_QUIESCE_SERVICES, `src/gpu_fault/node_agent/config.py::protected_quiesce_services`, lines 195-209),
the services stop last configured first
(`src/gpu_fault/node_agent/quiesce_persistence.py::TargetPersistence.stop_services`, lines 240-257) and restore FM ->
dcgm -> persistence -> collector -> host-collector -> (HyperPod) kubelet
(`src/gpu_fault/node_agent/quiesce.py::GpuServiceQuiesceManager._restore_locked`, lines 985-1000); both EKS journals show the
same ~12-read (~3 min) gap as HyperPod's, 27 reads at the advance on each. The
single upstream `Connection refused` (journal read 27) sits at the window's
START, 13:22:34Z = detection accepted: eks-managed taints the node
`gpu-fault.io/quiesce:NoExecute` before it dispatches the quiesce
(`src/gpu_fault/adapters/node_action/quiesce_taint.py::apply_quiesce_taint`, module docstring lines 12-14), the plain-EKS
dcgm-exporter DaemonSet does not tolerate that key
(`docs/components/plain-eks-gpu-cluster.md:67-68`; the HyperPod manifest keeps
`operator: Exists`, `deploy/dataplane/hyperpod-dcgm-exporter.yaml:59-60`), so
the exporter Pod is evicted seconds before the Agent stops the collector and
one scrape fails. What differs is the exporter's return after RESTORE: on
plain EKS kubelet never stopped and the Pod is recreated as soon as the taint
is lifted ahead of the restore, so the restored collector's first scrape
catches Fabric Manager re-training (2.0); on HyperPod the exporter container
waits for kubelet, the last restored unit, so the first scrape comes later and
already reads 3.0 -- the advance landed at read 27 (journal `ADVANCE read 27
step 2`) and the variant passed at 12:11Z. The product's grading is the same
on both -- its post-reset validation reads a trained fabric, served or real. Now a probe that ran and
exited 1 (not a transport failure) makes the runner read the tool's `/status`
and classify the refusal (`plan_release`): an abort or a lapsed hold (read
bound or deadline) is the plan's release -- PASS with a note that names the
409's meaning; a plan already past the held step is an advance whose receipt
was lost; a disarmed or unarmed tool is INCONCLUSIVE with the meaning ("the
replay is not on a held step"); a plan still on its held step is retried;
kubelet-down transport failures retry as before without consulting the tool.
`recovery_failure` released this way is INCONCLUSIVE (the FAILURE cannot be
held past the real transition where the exporter returns before the
collector's first scrape, so the RESTART_NODE rung cannot be driven there),
never FAIL on the product. The held step's read bound was a near miss of its
own: 36 reads ((300 + 180) / 15 + 4) plus the 3 baseline reads stand for
585 s of scrapes while arm -> reset-done took 569 s (plain EKS) and 589 s
(HyperPod); the live runs stayed at 27 reads only because the quiesce pauses
the collector for ~3 min, and a detection that takes the health summary's
full latency (300 + 300 s = 40 reads at 15 s) lapses the bound before the
quiesce even starts; `reset_hold_reads` now sizes it with
`held_reads(interval, 300 + health_summary + 300)` (124 reads at 15 s) over
the detection horizon and the chain. Evidence: `plan-advance.json` carries `release` (kind, reason, the
tool's status). The six reset variants are to be re-run on plain EKS
(`tests/regional/test_general_live_fabric_reboot.py`,
`tests/regional/test_general_live_fabric_plan_bounds.py`).

019 `remap_failure`'s recovery stage is the exit the settled record asks
for (live 2026-10-26, plain EKS ip-10-0-43-5, release R13g; product lane b192,
commit 69f87b5d, ships in R13h). On R13g the product detected the replayed
`row_remap_failure` and planned the RMA-class drain (FREEZE_EVIDENCE,
MARK_UNSCHEDULABLE, QUARANTINE, COLLECT_DIAGNOSTIC_BUNDLE, RUN_FIELD_DIAGNOSTIC,
VALIDATE_GPU, ESCALATE_SUPPORT), but the record was BLOCKED at compile time --
`blocked_reasons` "no executable owner for memoryDiagnostics": both live
Runtime Profiles declare memoryDiagnostics OBSERVE with a null adapter,
deliberately (no DCGM level 2/3 on this site) -- and the dispatcher sweep
superseded it 35 ms after creation. No command reached the node, the Node
Agent journal stayed silent, no cordon, taint or annotation was ever applied,
and the incident was left ESCALATED. The variant's chain stage still read
PASS, because `chain_errors` judged only the PLANNED steps (`official_steps`)
and the incident state -- a hollow pass that threatened every GpuChain user
(019/020/022/024) -- and the fixture's validated-restore script was then
refused ("workflow still owns recovery; operator hold retained"), the one
honest FAIL of the run. b192 makes a plan whose only compile errors are "no
executable owner for <capability>" run its containment prefix
(FREEZE_EVIDENCE, MARK_UNSCHEDULABLE, QUARANTINE, COLLECT_DIAGNOSTIC_BUNDLE,
TRIGGER_HEALTH_SNAPSHOT) as a SAFETY plan: SAFETY_PENDING, then
BLOCKED/SAFETY_SETTLED with that blocked reason, the incident QUARANTINED (not
ESCALATED), the node cordoned with the incident's `gpu-fault.io/quarantined`
taint and ownership annotation, a real COLLECT_DIAGNOSTIC_BUNDLE command on
the node, the training job untouched, and a `contained-operator-handoff-zh-v1`
notification (category OPERATOR_REVIEW, "[GPU FAULT][contained, operator
action required] <cluster>: <node>"). The documented exit for that shape is
the validated restore (`gpu-fault-admin submit-remediation --disposition
restore`; `recovery_safety.workflow_recovery_error` accepts SAFETY_SETTLED),
and `IncidentClosureService.close_incident` refuses it ("only an ESCALATED
incident can be closed"); until R13h ships `remap_failure` stays held on both
clusters.

Three harness rules follow. (1) `chain_errors` judges executed steps
(`never_executed_errors`): a `preemption_reason` naming the compile-time
close, or `blocked_reasons` on a record with no step execution at all, is
"chain planned but never executed: compile-time BLOCKED (<reasons>)"; every
isolating step of the chain (MARK_UNSCHEDULABLE, QUARANTINE, STOP_WORKLOADS,
CHECKPOINT_WORKLOADS) needs a SUCCEEDED final execution once the record has
ended, a FAILED one is its own error, a record that may still execute is
judged on what it has executed so far -- so 019's two containment variants
wait for the executions before judging (`chain_executed`,
`CONTAINMENT_SECONDS` on top of the detection budget) -- and an operation the
chain lists as `unexecuted` that did run (RUN_FIELD_DIAGNOSTIC here) is an
error; planned steps are read from `safety_steps` and `official_steps`
together, and diagnostic-only chains are unchanged
(`tests/regional/test_general_live_gpu_chain.py`). (2) The exit follows the
settled state (`_contained_exit`, after the replay has ended -- the restore's
VALIDATE_GPU would rightly refuse while row_remap_failure=1 is still served):
the node read first must carry the chain's containment (`containment_missing`:
the cordon, the incident's quarantine taint, the incident-id annotation), else
"chain never isolated the node" and nothing is restored or closed. QUARANTINED
-> the product's handoff notification must be in its notification records
(`handoff` stage: OPERATOR_REVIEW with the "contained" marker, polled 120 s)
and the node is readmitted through the validated restore (`restore_incidents`:
VALIDATE_GPU/HOST/FABRIC then RESTORE_SCHEDULING SUCCEEDED, no isolation left,
the incident RECOVERED on a fresh read). ESCALATED (the product handed the
node to an operator; on R13g that record is the hollow one and already fails
the chain) -> the operator close through `IncidentClosureService.close_incident`
(the call `gpu-fault-admin workflow-reconcile --close-incident` makes, with
the node's isolation evidence, operator `acceptance-fixture`, the run id as
reference, a reason naming the case's own injection), proven on fresh,
bounded reads: incident RECOVERED, the owning workflow no longer open, no
residue on the node. Any other settled state, a refusal on either path, or
residue is the recovery error and nothing is retried
(`tests/regional/test_general_live_gpu_memory.py`, over recorded fakes). (3)
The plan's expected text says so: containment first, SAFETY_SETTLED, incident
QUARANTINED, operator notified, readmitted by the validated restore. The
runner's post-case close of ESCALATED incidents lists only what is still
ESCALATED, so it has nothing to do after either exit. What the proof does not
cover: the restore (or the close) is the product's record exit; it does not
show that an RMA happened, and the replay never damaged memory. The
same finding exposed a reporting defect: `failure_class()` classified a Pod
exec's stderr from its last line, which for every `kubectl exec` failure is
kubectl's own trailer `command terminated with exit code N`, so the exception
class (`[RuntimeError]`) was missing from every Pod-exec failure's evidence;
kubectl's trailer and `Defaulted container ...` notice are now skipped before
classifying (`tests/regional/test_live_command_boundaries.py`).

The two variants that need a trusted independent kernel XID beside a held
DCGM fault (020 `independent_fatal_event`, 024 `independent_xid`) reuse lane
K's kmsg-mode kernel-source replay through
`scripts/e2e/regional/general_live_kernel_xid.py` (`--approve
KERNEL_SOURCE_REPLAY`). They deliberately replay a *monitor-grade* catalog
entry -- XID 137 `NVLINK_PRIV_ERR` beside the NVLink counter, XID 92
`EXCESSIVE_SBE_INTERRUPTS` beside the weak correctable-memory composite; both
are catalog Immediate Action `IGNORE`, which the engine resolves to
`NO_ACTION` / `MONITOR_ONLY` whatever the workload state, and the helper
refuses any other grade before touching the node. What they prove is the
independence of the trusted-source path under a concurrent diagnosis: the
XID's own catalog decision, its own `RECOVERED` incident, no workflow, no
correlation, no merge into the counter/composite incident, no XID component
inside the composite, the counter chain undisturbed. What they do **not**
prove is the spec's fatal wording -- that a RESET/QUARANTINE-grade dispatch
survives the concurrent diagnostic workflow (node-scoped merge, exclusive
operations). That branch needs the reset approvals and belongs to the
dangerous kmsg-XID lanes; a PASS here must be read with that limit. The
kmsg probe now gives every run its own sequence band (`kmsg_sequence`): the
product dedups the XID event id derived from `kmsg-<boot>-<sequence>`, so a
second run on the same boot used to be graded a duplicate of the first.
Through release R13g no kmsg-mode replay could reach the product at all: the
kernel collector took a regular `GPU_FAULT_KMSG_PATH` file's EOF for an ended
stream and reopened it at its end every second, skipping whatever the probe
appended in between (live 2026-10-26, plain EKS: 024 `independent_xid` XID 92,
032 `unknown_xid` XID 987 and 032 `missing_topology` SXID 11001 each left five
or six copies of the reviewed line in the fixture and no XID/SXID event in the
store). The collector now follows a regular file like `tail -f`
(`collectors/logs/kernel.py` `_follow_regular_file`,
`tests/collectors/test_kernel_regular_file_follow.py`); every kmsg-mode PASS
needs a release that carries it, and the probe's `status` reads back whether
the collector holds the fixture open at the end of what was appended
(`consumed`) so a window the collector never read is INCONCLUSIVE, not a
grading FAIL.

The 020 NVLink counter variants had no live coverage until the harness
followed the exporter's counter generation. On 2026-10-26 (06:45-06:58Z,
plain EKS, p5en.48xlarge, 8x H200, DCGM host engine 4.6/4.7, release R13g)
all eight variants ended `NOT_RUN(CAPABILITY_ABSENT)` ("no GPU on the node
reports nvlink_crc_flit_error_total" / `..._replay_error_total` /
`..._recovery_error_total`, "fewer than two GPUs report NVLink counters"):
the variants hard-coded the legacy per-class canonical names, but a
Hopper/NVLink4 exporter publishes only the DL aggregates
(`DCGM_FI_DEV_NVLINK_ERROR_DL_CRC` / `_REPLAY` / `_RECOVERY`, read as
`nvlink_*_aggregate_error_total`; field 409 and the per-class counters print
N/A on this part), and the HyperPod cluster has the same GPU generation. Each
variant now names an NVLink error *family* (`gpu_fault.dcgm_fields`
`DCGM_FABRIC_CANONICAL_GROUPS`: crc, replay, recovery) and the survey picks
the member the node really reports (`GpuSurvey.nvlink_counter`: the legacy
per-class counter where published, else the DL aggregate; `multiple_gpus` the
member the most GPUs report); the chosen canonical name and its exporter
metric are in the result's `canonical_name` / `exporter_metrics` and notes.
The product grades every member of a family alike -- the counter set, the
delta threshold, the counter-reset and zero-delta recovery handling, the
GPU_FABRIC class and both NVLink composites key on `nvlink_*_error_total` or
list all seven names (pinned by `tests/regional/test_general_live_gpu_links.py`)
-- and the replay tool's `nvlink` / `composite` profiles serve and alter the
DL aggregates exactly as the legacy counters (`tests/test_gpu_dcgm_field_replay.py`
over a typed Hopper scrape) while still refusing unknown fields. On this site
the variants therefore exercise `nvlink_crc_aggregate_error_total`,
`nvlink_replay_aggregate_error_total` and
`nvlink_recovery_aggregate_error_total`; they are to be re-run on both
clusters and still have no product evidence. The flit/data split and the
per-class counters themselves stay unexercised until a NVLink3 node joins.

The 024 memory composites replayed a second component the exporter may not
have. On 2026-10-26 (07:41Z, plain EKS, p5en H200, R13g) `fresh_components`
ended INCONCLUSIVE "the replay aborted: targeted reading absent upstream;
refusing invented samples: GPU-.../retired_pages_sbe_total": the variants
hard-coded `retired_pages_sbe_total` beside `ecc_sbe_volatile_total`, but
Hopper exposes no page retirement at all (row remapping replaced it; 019
`retired_delta` is CAPABILITY_ABSENT on this site), so the replay tool refused
to invent the sample and aborted after arm, and with both clusters on H200
`fresh_components` and `reused_components` had no coverage. The second
component is now chosen from the node's survey (`GpuSurvey.component_pair`):
`retired_pages_sbe_total` where the part reports it (pre-Hopper), else
`row_remap_correctable_total` -- both members of the product's
CORRECTABLE_MEMORY_DEGRADATION rule (`gpu_metrics._add_memory_composites`
takes any two of ecc_sbe_volatile_total, ecc_sbe_aggregate_total,
retired_pages_sbe_total, row_remap_correctable_total; pinned by ingesting both
pairs) -- on the same target GPU; `corrected_memory` and `independent_xid`
take the same pair, and `memory_failure` / `wrong_scope_source` take the
pending-repair flag the node reports (`row_remap_pending` on Ampere and later,
else `retired_pages_pending`). The chosen component is in the result's
`second_component` / `component_metrics` / `exporter_metrics` and notes, and
every composite plan is checked against the survey before anything is
installed (`composite_replay`): a plan naming a reading the node does not
report is NOT_RUN(CAPABILITY_ABSENT) listing it, never INCONCLUSIVE mid-run.
On this site the variants exercise `ecc_sbe_volatile_total` +
`row_remap_correctable_total`; the retired-page pair stays unexercised until a
pre-Hopper node joins. `fresh_components` and `reused_components` are to be
re-run on both clusters.

024 `wrong_scope_source` FAILED live (2026-10-26 07:51Z, eksb, R13g) on its
hardware stage alone: the repair GPU's own `row_remap_pending` incident ran the
product's authorized single-GPU service-window reset chain (FREEZE_EVIDENCE,
MARK_UNSCHEDULABLE, QUARANTINE, QUIESCE_GPU_SERVICES, VERIFY_NO_GPU_CLIENTS,
RESET_GPU, RESTORE_GPU_SERVICES, VALIDATE_GPU, RESTORE_SCHEDULING, every step
SUCCEEDED 07:47:57-07:50:03Z -- a real reset of one GPU) and ended RECOVERED
by itself, while the error GPU's incident was isolated and later released by
the validated restore and no cross-GPU composite formed. That is the product's
policy (a pending repair flag on a modern GPU is WARNING RESET_GPU, the chain
019 `pending_supported` expects under `GPU_RESET`), contradicted only by the
variant's unscoped `no_hardware` oracle. The variant now declares `GPU_RESET`
(flags `X`, risk H, like 019's reset variants: it really resets a GPU and
belongs with the dangerous, run-last variants), judges hardware per incident
(the repair incident's workflows may plan QUIESCE_GPU_SERVICES and RESET_GPU
and nothing else; the error GPU's incident and every other workflow on the
node plan no hardware operation), waits for the repair incident to settle
through its own chain (every workflow terminal, incident RECOVERED; a stalled
or parked chain is the `repair_settled` stage's error and the fixture never
restores that incident) and restores only the error GPU's incident.
`memory_failure` (both signals on one GPU) is unchanged: there the components
are suppressed into the composite and no reset is planned (PASS 2026-10-26).

024 `wrong_scope_source` FAILED again live (2026-10-26 14:51-14:55Z, the
HyperPod lane's node, R13g, harness live17) as "cleanup did not prove
the node clean: dcgm-replay: HostProbeMissingResponseError: dcgm close-run:
host probe response is missing (exit 1)", after its oracle gate's `audit-read`
had failed the same way at 14:54:25Z. The plan held the pair
(`ecc_dbe_volatile_total` +1 on one GPU, `row_remap_pending=1` on another, 60
reads, `until_advance`) and relied on the runner to close the replay after the
detection. Both incidents were minted at 14:54:03Z, and on HyperPod the repair
GPU's reset chain's QUIESCE_GPU_SERVICES stops kubelet about 20 s later
(`adapters/node_action/quiesce_taint`: a plain-EKS node keeps kubelet and is
tainted `gpu-fault.io/quiesce` instead, which the probe Pod tolerates), so the
probe Pod could not be reached, the replay stayed armed, and the chain's
post-reset VALIDATE_GPU re-graded a reading that still said
`row_remap_pending=1`: QUARANTINE, then RESTART_NODE (NotReady ~14:57Z, Ready
14:59:45Z; the workflow finished 15:06Z and lifted the quarantine) -- the
product's documented policy for a pending remap a reset did not clear, right
on what it was shown. The variant's subject is composite scoping, not that
escalation, so three things changed in the harness. (1) The plan is bounded
to exactly the one scrape that mints both findings (`MINTING_READS`, never a
spare), and the tool is back on the real readings by itself -- as a reset
applies a pending remap -- before the chain's quiesce stops the collector.
A window of reads would not do (S36's finding on 019): QUIESCE_GPU_SERVICES
stops `gpu-fault-metrics-collector` and the exporter container on both
providers (`node_agent/common.py` DEFAULT_QUIESCE_SERVICES /
DEFAULT_QUIESCE_CONTAINERS) within ~10 s of the mint (eksb 07:48Z journal:
read 4 served the pair, the next scrape 15 s later was "Connection refused"),
the field tool consumes a read only on a scrape it could serve
(`make_server`: a failed upstream fetch is journaled as an error and `reads`
does not advance), so any fault read left unconsumed is served to the first
scrape after RESTORE_GPU_SERVICES -- the batch the post-reset VALIDATE_GPU
grades (fresh metrics after the RESET_GPU readiness barrier, then the GPU's
active findings, no grace for `row_remap_pending`) -- and
`escalation.next_rung` turns a validation failed after a reset into
RESTART_NODE. One scrape mints every record the variant judges
(`gpu_memory_policy.memory_decision`: DBE > 0 is CRITICAL, the pending flag
WARNING RESET_GPU, no delta, confirmation or sustain; the same-GPU composite
forms on that first batch too -- pinned by ingesting one batch through
`GpuMetricsService`). The records are judged once the repair record has
ended (`scoping_errors`: no GPU_MEMORY_DEGRADATION finding or incident on
either GPU, the error GPU's own incident present, no incident of the other
GPU's reading on either GPU; hardware per incident; the repair incident
RECOVERED by itself), the replay is closed after the chain -- never inside
it, where its collector restart would fight the quiesce -- and the spec text
now says the chain ends RECOVERED with no RESTART_NODE escalation. (2) The oracle gate falls back to the product's records when the
journal cannot be read because the node is away (`wait_served_or_records`,
evidence `replay-gate-deferred`) and the journal is read at the close; any
other probe failure stays the fixture failure it was. (3) `DcgmReplay.close`
no longer fails on a probe the product took away
(`general_live_gpu_replay_close`): it polls the Node object inside the
variant's remaining budget (15 min at most, 3 min kept back), waits for Ready
to have held 45 s with no quiesce taint, retries `close-run` with the journals
read first, and judges a node whose boot id changed since the deploy clean by
its reboot (`judged_by: node reboot`: the drop-in, run directory, transient
proxy unit and deadman live under /run and cannot survive a boot); only a node
that never comes back without rebooting leaves the close unproven, and the
error names the boot id and readiness it saw. Tests:
`tests/regional/test_general_live_gpu_composite.py` (the installed plan driven
through the field tool -- three pass-through reads, one served read, then the
exporter's own scrape byte for byte; the product facts the one-read bound
rests on; the close after the settled record; the records gate; the scoping
oracle over findings and incidents) and
`tests/regional/test_general_live_gpu_replay_reboot_close.py` (the quiesce
return, the Ready settle, the quiesce taint, the reboot verdict, unproven
inside and outside the budget). The variants whose finding plans no reset --
`memory_failure` (QUARANTINE only), `corrected_memory`, `fresh_components`,
`reused_components`, `independent_xid`, `nvlink_multiple_gpus` (diagnostic
composites) -- plan no reset, and only the reset family's chain
(`orchestration/families/health._reset_operations`) carries the quiesce, so
their collector never stops under a held plan and no fault read can be
served late; they keep their held plans and gain the reboot-aware close. 019
`pending_supported` / `post_reset` held `row_remap_pending=1` into the same
chain and advanced or closed it through the probe (`post_reset` after
RESET_GPU SUCCEEDED, while kubelet is still down on HyperPod) -- the same
exposure on both providers; S36 bounded them the same way (`pending_fault_reads`,
since S46 the node's edge-filter confirmation count of reads rather than one),
gated on the journal or the records, judged on the settled record and closed
after the chain.

024 `wrong_scope_source` FAILED a third time live (2026-10-27 14:33-14:41Z, the
HyperPod lane's node, R13g 37db33f254a1, harness int/r13c-live28, the one-read
plan): "repair_settled: the row_remap_pending incident ... is QUARANTINED, not
RECOVERED by its own chain" -- the harness's oracle contradicting the product's
design, not a product fault. The journal shows read 4 serving the pair
(replay-served accepted 14:36:20Z, detection 14:36:26Z) and reads 5-9 the
exporter's own scrape byte for byte with no upstream error; the repair-settled
polls 14:36:32-14:38:17Z found the record running and 14:38:43Z found every
workflow ended with the incident QUARANTINED, no rung planned (the hardware
stage was empty) and NO reboot (the preflight at 14:33:02Z, the survey and the
post-restore Node object all carry boot id cc70db9b, the 14:11Z boot of an
earlier case; Ready 14:37:40Z is kubelet back after the quiesce, which is why
the `/run` objects -- drop-in, audit file -- were still there for the
14:39:06Z close); the validated restore released the repair incident first, as
the node's isolation owner, then the error GPU's, leaving the node Ready,
schedulable and unannotated. Root cause (S54 follow-up, reproduced through the
product's own path -- `GpuMetricsService.ingest`, `gpu_node_health_findings`,
`NodeHealthIngestionService.ingest` -- identically on this tree and the R13h
candidate): the error GPU's CRITICAL DBE plans FREEZE_EVIDENCE,
MARK_UNSCHEDULABLE, QUARANTINE, whose QUARANTINE step carries
`terminal_quarantine_node_ids` (`orchestration/families/health._enrich_step`
for a QUARANTINE/DRAIN finding) -- a terminal hold on the node -- and the
repair GPU's RESET_GPU plan on the same node inherits it
(`workflow_quarantine.inherit_terminal_quarantine`, "a subsequent recovery
cannot silently undo a node's persistent hold": a QUARANTINE inserted,
RESTORE_SCHEDULING superseded; the product pins it in
`test_reset_inherits_quarantine_without_reintroducing_retired_steps`), so
`terminal_state.terminal_incident_state` settles it QUARANTINED although every
executed step SUCCEEDED. The pending flag alone (019) has no sibling hold and
settles RECOVERED (live PASS on both clusters). The 2026-10-26 eksb record read
as "RECOVERED by itself" does not support that reading either: its
validated-restore.json releases the repair incident first too. The property
product gap 30 names (VALIDATE_GPU grants a pending flag neither the grace nor
b202's newer-batch wait -- true on both trees, probed) is latent, not this
FAIL's cause: the one-read plan, like 019's, lets the real reading clear the
flag before the post-reset validation. So the generation gate of 96a1e5f7 /
5a257ea is gone (neither 019 nor 024 is gated) and `repair_settled` judges the
product's design: every workflow of the repair incident terminal, the latest
SUCCEEDED having executed RESET_GPU and VALIDATE_GPU (both SUCCEEDED), and the
incident RECOVERED -- its own RESTORE_SCHEDULING released the node -- or
QUARANTINED under the error GPU's terminal hold: the chain's QUARANTINE carries
the node in `terminal_quarantine_node_ids`, no RESTORE_SCHEDULING of it
SUCCEEDED, and an error-GPU incident holds the node under its own terminal
quarantine. A FAILED/BLOCKED chain, a QUARANTINED without the hold, a hold
without a source or a chain that never reset stays the stage's error. The
validated restore releases the error GPU's incident and a repair incident the
ended chain left isolated (a running chain is not restored over), so the node
ends clean either way, and `repair-record.json` keeps both incidents' full
record (workflows, superseded steps, hold parameters, step executions,
timestamps). With the findings ingested the other way round (the repair
finding first) the product's persistent quarantine replaces the unstarted
reset plan and the DBE finding joins the repair incident -- no reset, one
incident -- which the scoping stage reports; both live records show the
error-first order. Tests: `tests/regional/test_general_live_gpu_terminal_hold.py`
(the product's ingest path pinning the inherited hold and the QUARANTINED, the
flag alone RECOVERED; the variant over the product's own records: the live
shape PASSES with both incidents released; a failed VALIDATE_GPU, no hold, no
source, no reset -> FAIL; a running chain is not restored over) and
`tests/regional/test_general_live_gpu_memory_reset.py` (019's two reset
variants ungated, reading the GPU state and the node activity only).

019 `pending_supported` and `post_reset` are now bounded and judged on the
records (S36, 2026-10-26; both are X variants under `GPU_RESET` that have never
run live). Their plan held `row_remap_pending=1` for 120 reads
(`until_advance`) and advanced or closed it through the probe -- `post_reset`
right after RESET_GPU SUCCEEDED, which on HyperPod is inside the quiesce window
with kubelet and the probe Pod gone. Sizing the bound from the product and the
live journals showed that a window of reads is not enough. The quiesce stops
`gpu-fault-metrics-collector` itself and the exporter container on both
providers (`node_agent/common.py` `DEFAULT_QUIESCE_SERVICES` /
`DEFAULT_QUIESCE_CONTAINERS`), and the field tool counts only the scrapes it
could serve (`tools/gpu_dcgm_field_replay.py` `make_server`: a failed upstream
fetch is journaled as an error and `reads` does not advance), so every fault
read the quiesce finds unconsumed is served to the first scrape after
RESTORE_GPU_SERVICES -- the batch the post-reset VALIDATE_GPU grades
(`adapters/gpu_validation.py`: fresh GPU metrics after the RESET_GPU readiness
barrier, then the GPU's active findings, no grace for the pending flag) -- and
`orchestration/escalation.next_rung` turns a validation that fails after a
reset into RESTART_NODE. The eksb 2026-10-26 07:48Z replay journal (024
`wrong_scope_source`, the held plan) shows read 4 serving the pair and
"Connection refused" at the very next scrape 15 s later: the exporter was gone
within ~10 s of the mint. The product grades the first replayed scrape within
~4 s (`gpu_memory_policy.memory_decision`: the flag mints on one reading > 0 --
no delta, confirmation or sustain -- and clears on the next accepted reading of
0). S36 therefore served the flag for exactly the one scrape that mints it --
and live 2026-10-27 04:05-04:18Z (HyperPod, the hp lane's fault node,
R13g, harness int/r13c-live23) that one read minted nothing: the replay journal
shows read 5 serving `row_remap_pending=1` and reads 6-43 the device's own,
no incident, no error record, the node untouched (FAIL "no row_remap_pending
... finding reached an incident"). The ingest grades a *delivered* batch on one
reading, but the batch has to leave the node first, and the node's collector
delivers a changed `row_remap_pending` only once the flag has been a "breached"
candidate on `GPU_FAULT_DCGM_EDGE_CONFIRMATION_SAMPLES` consecutive scrapes
(`collectors/gpu/dcgm.py` `_candidate_keys` / `_should_deliver`
`candidate-confirmed`; 3 on the live nodes, per the case's `gpu-survey.json`
`collector_env`). The flag is a gauge -- not a counter (`_is_counter`:
`*_total`, the `counter-increased` edge) and not a state gauge
(`STATE_CHANGE_REASONS`: the last XID and the fabric state) -- so one read of
it is no delivery edge at all and the batch is suppressed; S36's bound had been
pinned against `GpuMetricsService.ingest` alone. 024 `wrong_scope_source`
mints on one read only because the DBE counter in the same batch is a
`counter-increased` edge (eksb 2026-10-26 07:48Z, read 4). So (S46, 2026-10-27)
the flag is served for exactly the node's confirmation count of consecutive
scrapes (`pending_fault_reads(confirmation)`: the last one is the delivery and
the mint; fewer never reach the control plane, a spare is the escalation; one
read where the survey shows `GPU_FAULT_DCGM_EDGE_FILTER_ENABLED` off, because
then every batch is delivered), the plan is checked against the node's count
before anything is deployed (`require_pending_plan`, a `RegionalFixtureError`),
the plan is never held, and the tool is back on the device's real reading
before the collector stops. The reads before the confirmation must not be
delivered early either: the one delivery an idle node makes by itself is the
health summary, which would mint the flag on the first flag read and leave the
two behind it for the first scrape after RESTORE_GPU_SERVICES (the RESTART_NODE
exposure). Summaries go out on the node's restart-stable phase
(`collectors/scheduling.next_stable_phase` of cluster, node and the
`gpu-metrics` channel), moved by at most `GPU_FAULT_COLLECTOR_CADENCE_JITTER_
RATIO` (0.1, not set by this site) x 300 / 2 = 15 s and delivered on the first
scrape at or after the moved phase, so the arm waits (`summary_clear_wait_
seconds`, evidence `summary-phase-clearance`) until the pre-confirmation reads'
window -- `(A + 3 x 15 s, A + 6 x 15 s]` for an arm at `A`, the tick phase
unknown and one scrape of slack -- meets no phase's `[P - 25 s, P + 40 s]`;
at most 109 s, most of the time nothing. A stray counter edge on the idle node
during those two scrapes would still deliver early; the plan cannot rule that
out and the records would show it as the escalation. Both
variants deploy, arm, gate on the journal or -- when the quiesce has already
taken the probe -- on the product's records (`wait_served_or_records`), wait
for the incident and then for the record to END (`RESET_SETTLE_SECONDS` 600 s
beyond the health summary: the chain took 2 min 6 s live, its RESTART_NODE
escalation 12 min), and only then close the replay through the reboot-aware
close; nothing is advanced. The oracle is the records: `pending_supported`
proves the policy and the executed chain (QUIESCE_GPU_SERVICES,
VERIFY_NO_GPU_CLIENTS, RESET_GPU, RESTORE_GPU_SERVICES, VALIDATE_GPU
SUCCEEDED; RESET_GPU executed; the incident RECOVERED; no workflow of the
incident planning RESET_ALL_GPUS_NVSWITCHES / RESTART_NODE / REPLACE_NODE; no
other hardware step on the node), `post_reset` the post-action state on top
(`post_reset_errors`: exactly one SUCCEEDED RESET_GPU, VALIDATE_GPU SUCCEEDED
and not before the reset completed, a single workflow for the incident -- a
successor is the ladder's rung and the blind reboot loop the variant rules
out), and both require no `row_remap_pending` finding active after the chain
(`recovered_errors`). The other 019 plans are bounded by their own read counts
too (`fault_reads`: the wait that needs the fault plus two reads -- the
diagnostic counters for the detection wait, the isolating `volatile_dbe` /
`remap_failure` for detection plus `CONTAINMENT_SECONDS`; `remap_failure`
plans no reset at all: its CRITICAL DRAIN has no quiesce, the failure is
served for the containment wait because a remap failure does not clear by
itself, and the replay ends before the exit as before), and every 019 replay
records the node's boot id for the close. The same reading applies to 024
`wrong_scope_source`'s three-read window: its second and third fault reads
fall after the exporter is gone and would be served to the post-reset scrape;
that is left to its own lane. Tests:
`tests/regional/test_general_live_gpu_memory_reset.py` (the installed plan driven
through the field tool -- three pass-through reads, the flag on exactly the
node's three confirmation scrapes, then the exporter's own scrape byte for byte,
`advance` refused; one read where the edge filter is off; the plan guard's
refusals; the summary-clear wait against the product's scheduler for every
second of a cycle and the variant holding before its arm; the records gate,
settle wait, close and no advance across a simulated kubelet gap; the
RESTART_NODE successor record failing every stage but recovery although the
incident ends RECOVERED; a finding still active; two resets or a validation
before the reset; a chain still running at the settle bound; the product facts
the read bound rests on -- the flag's mint and clear of a *delivered* reading
through `GpuMetricsService.ingest`, the quiesce service and container lists,
the readiness barrier, `next_rung`, and the tool's server refusing a dead
upstream without consuming a read),
`tests/regional/test_general_live_pending_edge.py` (the harness's plan through
the real field tool, the real `DcgmMetricsCollector` delivery filter and the
real `GpuMetricsService.ingest`: the confirmation count of flag reads is
delivered on the third and mints the RESET_GPU finding, the first real reading
clears it; S36's one-read shape ships nothing but the baseline and never mints
-- the live 2026-10-27 failure; a spare read rides the restarted collector's
initial baseline to the post-reset VALIDATE_GPU with the finding still active;
a health summary on a pre-confirmation read mints early, and the arm's
clearance moves that window past the phase) and
`tests/regional/test_general_live_gpu_memory.py` (no 019 plan holds a step).

024 `independent_xid` FAILED live (2026-10-26 08:07Z, eks ip-10-0-43-5, R13g,
harness live14) with two CORRECTABLE_MEMORY_DEGRADATION incidents for one held
pair: `inc-gpu-dcgm-<node>-1793001279120143-GPU-053ab270-...-composite-CORRECTABLE_MEMORY_DEGRADATION`
(batch 07:54:39.12Z, the first replayed scrape, WARNING) and
`...-1793001309238684-...` (batch 07:55:09.24Z, the third consecutive scrape).
Both lie inside the run's window (started 07:52:34Z) and precede the kernel
XID injection (07:55:57Z), and the replay journal shows both counters rising
by one on every read with no gap or abort, so neither the activity window nor
the XID replay minted the second one. It is the product's own escalation:
`GpuMetricsService._add_memory_composites` (`gpu_metrics.py`) turns the
composite CRITICAL at `correctable_memory_drain_consecutive_samples` (3)
consecutive breaches, the store's `update_gpu_findings` treats a severity
change as a new activation (`store/memory/telemetry.py`,
`store/postgres/gpu_telemetry.py`), the composite finding id is stamped with
the batch (`gpu_composites.composite_finding`), and the ingest mints one
NODE_HEALTH incident per activated finding id (`app/ingest/gpu_findings.py`,
`event_id gpu-<finding id>`), so a sustained correctable-memory episode opens
a second incident while the first is still open -- against the 024 rule that
sustained (reused) components keep one composite incident
(`reused_components`, whose 80-read hold crosses the same third sample);
`corrected_memory` judges the detection document before it and passed. Fixed
in the product (lane b194, for R13h): a composite keeps one finding id -- one
event id, one incident -- for its episode
(`gpu_composites.composite_episode_identity`); the third sample's CRITICAL is
reported as `updated_composite_findings` and lands on the open incident as
one bounded reason (`composite:CORRECTABLE_MEMORY_DEGRADATION escalated to
CRITICAL: ...`) with the marker rewritten at the new severity; a breach after
a clear, after a changed action, or after the GPU stopped reporting for
longer than the correlation window opens a new incident (Detailed Design
§2.4.2, "Composite identity"). `tests/metrics/test_composite_episode_identity.py`
drives the live shape through the service, both key/value stores and the
ingestion route; `tests/regional/test_general_live_gpu_composite.py` pins the
single activation per episode. The oracle of `independent_xid` and
`reused_components` is unchanged (one composite incident); both are to be
re-run on both clusters on R13h, together with the XID 92 decision check of
the same run. The same run's XID 92 reached no decision within 420 s although the
kmsg fixture was armed (07:55:57Z, the run's own sequence band, a current
monotonic stamp, the window's open and arm receipts clean); the product's XID
event records and the kernel collector's log for that window are to be read
before the variant is re-run -- the fixture side shows nothing wrong.

025 `exporter_timeout` FAILED live (2026-10-26 08:34-08:48Z, eksb
ip-10-0-32-162, R13g, harness live14) with "VALIDATE_HOST did not finish:
still WAITING (health_evidence_pending, host_diagnostic_failure,
missing_observations, step_waiting_seconds, step_waiting_slow)" while its
hardware, unknown and recovery stages were clean. That was the runner's order,
not the product: under the OBSERVABILITY recovery contract VALIDATE_HOST
passes only on a GPU_METRICS success newer than the finding and at most 120 s
old (`health_evidence_validation.validate_observability`), so with the node's
only exporter paused it waits by design, latches `step_waiting_slow` at 300 s
(the step-waiting alert over `gpu_fault_workflow_step_waiting_seconds` is the
escalation) and fails "diagnostic inconclusive" at the executor's 600 s
per-step cap (`execution/step_bounds.bounded_waiting_outcome`; no override for
VALIDATE_* on this site) -- bounded, no product gap. The variant paused the
exporter for 900 s and demanded a terminal validation inside a 720 s wait,
which could not happen before its own resume (last read 08:47:22Z, resume
08:47:43Z, fresh success 08:48:10Z). It now judges in two reads: while paused,
the incident, the planned bounded chain (TRIGGER_HEALTH_SNAPSHOT +
VALIDATE_HOST, nothing hardware) and a validation that has not passed on
evidence that cannot exist; after `exporter-resume`, the validation's
completion within 300 s -- SUCCEEDED, dated after the resume, not through the
cap -- with the product's own waiting reasons and the waited seconds kept in
the result (`validation`). The deadman stays 900 s; `max_minutes` is 20.
`sustained_unknown` and `post_refresh_success` never required a terminal
validation and are unchanged. The WAITING summary of every chain verdict now
prints the health-evidence reason and the missing observations instead of the
detail keys. To be re-run on both clusters.

025 `conflicting_alias` ended INCONCLUSIVE live (2026-10-26 08:53Z, eksb
ip-10-0-32-162, R13g): "the replay aborted: DCGM_FI_DEV_CLOCK_THROTTLE_REASONS
is not a distinct alias of DCGM_FI_DEV_CLOCK_THROTTLE_REASONS". The variant
hard-coded the injected alias, and this node's dcgm-exporter 4.6.1 publishes
`clock_throttle_reasons` under exactly that (legacy) name, so the tool's alias
op refused it by design. No exporter publishes two names of one field (the
survey's `exporter.gpus[uuid].metric_names` carries one name per canonical
field on the H200 nodes); a distinct alias pair is the published name plus the
product's other name for the same canonical field
(`collectors/gpu/dcgm.DCGM_METRICS` inverted, limited to the completeness
profile's fields -- only `clock_throttle_reasons`:
DCGM_FI_DEV_CLOCK_THROTTLE_REASONS / DCGM_FI_DEV_CLOCKS_EVENT_REASONS; the
T.Limit pair is a thermal-safety reading the profile excludes). The pair is now
chosen from the survey before anything is installed (`alias_pair`, on the
highest-index GPU reporting the field) and recorded in the result
(`alias_pair`: field, published, injected, value); a node with no usable pair
is NOT_RUN(CAPABILITY_ABSENT, "no canonical field has two distinct exporter
aliases on this node: ...") instead of an abort mid-run. To be re-run on both
clusters (live16).

025 `conflicting_alias` FAILed on its first live run with a distinct pair
(2026-10-26 15:30-15:35Z, HyperPod, the lane's H200 node, R13g, harness
live18): `ambiguous_rejected` held (the product's "invalid or
ambiguous GPU observation: <gpu>/clock_throttle_reasons" was in the window)
and `no_silent_overwrite` read "a conflicting alias slot was accepted as a
value: 1.0". That 1.0 was the published reading from *before* the conflict:
the result window opens before the arm, the repoint's collector restart
delivers its initial-baseline scrape with the real reading, the plan's three
pass-through reads follow (15:31:43-15:32:24) and the alias was served from
read 5 (~15:32:30), yet the stage graded every accepted sample of the slot
since the window opened. The product's semantics run the other way: two
exporter names of one canonical field that disagree on one GPU are one
`(device, canonical)` slot seen twice -- `gpu_environment_policy.
deduplicate_dcgm_aliases` collapses an alias pair only when values and labels
are equal, `gpu_observation.normalize_gpu_observation` marks every sample of
a slot counted twice invalid and keeps none, and `GpuMetricsService.ingest`
upserts `gpu_metric_latest` for accepted samples only -- so from the first
conflicting batch the product recorded (15:33:01Z here: the node collector
confirms the alias's clock-reason candidate over three scrapes before it ships
the edge, so reads 5 and 6 never reached the control plane) neither the
injected value nor the published reading is a sample, and the slot's latest
row stays the pre-conflict reading. The variant now anchors on that batch
(`first_conflicting_batch_at`) and grades no sample before it; the injected
value as a sample or as the latest row in any batch is a FAIL, as is any
accepted sample of the slot or a latest row timestamped inside the window;
the retained reading goes into `notes`. A product that never reported the
ambiguity leaves no batch to anchor on, and the window then opens at the
oracle gate's confirmation that the alias was served (`window_opened_at`;
every later read is on the held step). The sibling field variants were read
for the same exposure: `duplicate_slot`'s `others_usable` accepted any latest
row the other GPU ever had, from before the arm as well, and now requires the
other GPU's reading accepted at or after the first batch that rejected the
duplicate (its anchor is recorded as `first_replayed_batch_at`);
`missing_field` accepts only the scrape-wide message naming
`pcie_replay_total` (a real gap of another field inside the window used to
satisfy it) and is NOT_RUN(CAPABILITY_ABSENT) when no GPU reports the field;
`one_gpu_missing` gates on the served omit like `missing_field` instead of a
journal peek that asserted nothing. `nan_negative` already anchored on the
product's rejection; `exporter_timeout`, `sustained_unknown` and
`post_refresh_success` pause the exporter at once and have no pass-through
phase. Harness oracle: `tests/regional/test_general_live_gpu_fields.py` (the
live chronology, which FAILs on the previous oracle and PASSes now, the
published reading continuing, the injected value accepted, the latest row
moving, the gate-opened window, the duplicate and omission siblings); product
pin: `tests/collectors/test_dcgm_per_gpu_completeness.py` (served text ->
`DcgmMetricsCollector` -> control-plane ingest -> store: both samples
forwarded, the scrape delivered at the candidate's confirmation, the slot
dropped whole, the earlier reading and its time retained, no finding, the
one-name scrape as the recovery edge). To be re-run on both clusters with the
runner alone; no product release is needed.

024 `thermal_warning` and `power_performance` reuse the GF-ENV-LIVE
telemetry replay unchanged (`tools/gpu_environment_replay.py`,
`probes/env_replay_proxy.py`, `env_live_fixture.EnvReplayController`) through
`general_live_env_replay.py` and the reviewed plans
`env_live_plans.thermal_stress_plan` / `power_limit_throttling_plan`; the
environment tool's `ALLOWED_FIELDS` is not widened. The thermal plan holds a
warning-band core temperature computed from the device's own limits with a
rising thermal violation under the `0x20` software reason (never `0x80`); the
power plan pins the draw just under the real enforced limit at high
utilization with a rising power violation and serves no clock reason, because
a thermal or brake reason is a temperature finding the power composite excludes
by design. Every wait stays inside the proxy's 600 s window and the endpoint is
restored by the runner, not the deadman. The de-duplication grade reads the
product's own records (one composite incident, no component incident minted
after it formed, components only its own metrics, severity WARNING and
RUN_DIAGNOSTICS on the finding and the incident). A device without a gradable
core scale, without the power fields or with a real thermal/brake reason at
preflight is `NOT_RUN`.

Their first live runs (2026-10-26, plain EKS, release R13g) ended INCONCLUSIVE
with "the collector consumed 0 of the 12 armed replay reads inside the budget",
and no product behaviour was graded: the shared env-replay probe
(`probes/env_replay_proxy.py`) repointed `gpu-fault-metrics-collector.service`
through a drop-in `Environment=GPU_FAULT_DCGM_METRICS_URL=<proxy>` and did
restart the unit (the receipts carry a new InvocationID and MainPID), but the
unit's own `EnvironmentFile=/etc/gpu-fault/collector.env` -- where the node
installer writes the real exporter URL -- overrides any `Environment=`
(systemd.exec(5)), so the new process kept reading the real exporter while
`systemctl show`, which lists `Environment=` directives only, reported the
override as effective and the probe's verification passed. The same probe
serves every GF-ENV-LIVE case, whose HyperPod round of 2026-10-24 ended with
the identical "consumed 0 of N reads" signature; the earlier GF-ENV-LIVE PASS
verdicts graded no served reads. The probe now hands the override to the unit
as a run-scoped `EnvironmentFile=` (later-listed files win), reads the endpoint
back from the restarted process and waits for that process's first pass-through
read of the proxy (start-up spread + start-up jitter + one interval, at most
90 s) before the fixture arms; an override the process did not pick up or a
collector that never read the proxy is rolled back on the node and reported as
`repointed: false` with a constant reason, which the fixture raises as a
harness failure (INCONCLUSIVE "harness fixture failed: ... did not take effect")
instead of spending the replay budget. The restore is idempotent (nothing is
restarted on a clean unit) and runs whenever a repoint was attempted. Both
variants are to be re-run on both clusters; they still have no product
evidence. The four product gaps these variants
first recorded as expected FAIL -- the whole-batch (not per-GPU) DCGM
completeness check (025 `one_gpu_missing`), the PCIe<->XID gauge join with no
age check (022 `expired_xid`), the UUID-set-only identity comparison (026
`bdf_changed`) and the inventory baseline that never compared the source boot
id (026 `stale_boot`) -- are fixed by batch 160, so the variants now expect
PASS; a product that regressed would FAIL them.

026's inventory oracle gate is now waited for. On 2026-10-26 (plain EKS,
release R13g) `bdf_changed` (ip-10-0-43-5, 09:14-09:18Z) and `count_unchanged`
(ip-10-0-32-162, 09:18-09:21Z) ended INCONCLUSIVE "the inventory replay was not
effective: the wrapper never rewrote a real inventory query" with no product
record consulted: `require_inventory` read the wrapper's journal exactly once,
14-15 s after `repoint-collector` had restarted the collector, while the
restarted collector queries the inventory on its first tick -- the start-up
spread (`GPU_FAULT_COLLECTOR_STARTUP_SPREAD_SECONDS`, 15 s) plus the start-up
jitter (`GPU_FAULT_COLLECTOR_STARTUP_JITTER_SECONDS`, `[0, 20 s)`) -- and then
every `GPU_FAULT_GPU_INVENTORY_INTERVAL_SECONDS` (60 s here). The first real
query of the `bdf_changed` run was rewritten 20 s after the restart, 5 s after
the verdict (its record is in the case's `oracle.jsonl`); `uuid_changed` had
passed at 09:03Z with a first tick 6 s after the restart. The reviewed plans'
deadlines (`not_after` 09:41:45Z and 09:48:55Z) had not expired. The gate
(`DcgmReplay.wait_inventory_rewritten`, the same for all seven variants) now
polls the journal from the repoint on, bounded by spread + jitter + two
inventory intervals + 30 s, all read from the survey's collector environment
(185 s on this site); a real query the wrapper passed through unchanged (an
unreadable or expired plan, an index not in the real inventory, a replay UUID
naming a real GPU) is final and INCONCLUSIVE at once with the wrapper's reason,
a failed real query is retried by the product and waited out, and a deadline
names whether any query reached the wrapper. The plan's deadline is composed
before the first source lands on the node (the probe refuses a second copy) as
a 600 s deploy allowance plus the variant's hold, so the plan stays valid for
the whole hold after the repoint; a deploy that overran the allowance is a
named fixture failure ("harness fixture failed: the reviewed inventory plan
stays valid N s after the repoint, below ..."), never a plan expiring under the
running variant (`inventory-gate.json` records the window, the deploy duration
and the remaining validity; `tests/regional/test_general_live_gpu_identity.py`).
All 026 variants are to be re-run on both clusters with the runner alone.

022 `counter_reset` and `source_boot_change` first ran live on 2026-10-26
(plain EKS, release R13g, one node each) and FAILed on their own plans, not on
the product. `counter_reset` raised the counter by 50 per 15-s read (200/min
against the 8/min rule) and read the WARNING that storm minted as "a PCIe
counter reset minted a finding"; `source_boot_change` ramped +5 per read
(20/min, the `over_rate_limit` increment) inside the replayed boot after the
repointed collector's pass-through first scrape, so the only boundary it
crossed was (0, real boot) -> (0, replayed boot) and the same-generation storm
was read as "a cross-generation PCIe rate". The product's records grade both
boundaries as unknown: the drop is delivered as a `counter-reset` edge, has no
delta, mints nothing, neither clears nor re-mints an active finding and becomes
the baseline; a sample whose `source_boot_id` differs from the previous
accepted one has no delta and becomes the new generation's baseline, after
which same-generation rates grade as always
(`gpu_composites.metric_counter_delta`; pinned on the real collector filter
and ingest by `tests/metrics/test_pcie_counter_boundaries.py`, no product
change). The variants now drive the boundary itself. `counter_reset` raises by
`under_limit_increment` (half the 8/min budget, at least one: one per read on
a 15-s collector) before the clear and grades, on the product's records, the
accepted raise, the accepted drop with its `counter-reset` edge reason, the
latest reading equal to the reset value and every `pcie_replay_total` finding
by its place (during the raise, or at/after the drop), plus no active finding
and no hardware step (evidence `reset-readings`). `source_boot_change` arms the
plan before the repoint (`arm_before_repoint`; `DcgmReplay.deploy` is now
`serve` + `repoint`), so the restarted collector's first scrape under the
replayed boot id is already a `cross_boot_jump` -- over 8/min against any
reading of the real boot (twice the health summary plus a scrape and a
minute), held afterwards -- and grades no finding, a reading accepted under
the replayed boot and the latest reading being that first reading; a first
new-boot reading that is not the jump is INCONCLUSIVE (the boundary was
crossed at the real value and not exercised) (evidence `boot-readings`;
`tests/regional/test_general_live_pcie_boundaries.py`). Both variants are
rerun with the runner alone; no product release is needed.

022 `source_boot_change` then ran its new order live for the first time on
2026-10-26 (14:12Z HyperPod, the same minute on plain EKS; release R13g,
harness live17) and ended INCONCLUSIVE at the fixture, not at the product:
"harness fixture failed: HostProbeError: dcgm arm: host probe failed (exit 1;
error ProbeError 98b43a124d35de96 at arm:1092)", `probe-error.json` saying
"refusing to arm: the collector does not read the replay". The host
controller's one `arm` guard required the running collector to already read
this run's replay -- the repoint-then-arm order every other replay variant
uses (the sibling 022 variants PASSed on the same node minutes before) --
while `arm_before_repoint` arms first by design, and the unit tests mock the
probe, so the two orders first met on the node. The probe's `arm` is now
mode-aware (`--before-repoint`, threaded from `DcgmReplay.arm` only by
`arm_before_repoint`): the default mode still requires the repointed endpoint;
the before-repoint mode requires the collector to still read the original
exporter (not this run's replay, not another run's proxy), this run's proxy
unit to be active and answering `/status`, unarmed and with no read served,
and records `before_repoint`, the endpoints, the unit state and the read count
in the receipt. Every refusal names the mode and the endpoint it saw and
carries the same facts as its detail. Arming before any scrape burns no plan
step: the tool advances steps per `/metrics` read it serves (`Replay.active`
counts `reads - start_read`; `/status` does not count), so the repointed
collector's first scrape is the plan's first read as b193 intended
(`tests/regional/test_general_live_dcgm_probe_arm.py`, the wiring in
`tests/regional/test_general_live_pcie_boundaries.py`). The variant is rerun
with the runner alone on both clusters; no product release is needed.

020 `counter_reset` first ran live on 2026-10-26 (HyperPod, H200, the DL
aggregate `nvlink_crc_aggregate_error_total`) and FAILed on its own plan and
oracle, not on the product. Unlike the PCIe rate, the NVLink rule has no
under-limit raise: `gpu_metrics._link_decision` grades every increase of an
NVLink error counter (`delta >= nvlink_error_delta_critical`, default 1) as
the WARNING, so the variant's +5 per read raise minted the one incident the
oracle then blamed on the reset ("a counter reset minted a finding"), and its
`active` stage demanded no active finding after the reset -- the opposite of
the product's contract, by which a decrease is an invalid delta that keeps the
finding (neither a new fault nor a recovery; only a later zero-delta batch
recovers it, which `recovered` proves). Replayed through the real collector
filter and ingest, the live reads (0, 0, 0, 5, 10, 15, 0, ...) mint at the
first raised read and nothing at the reset tick, which ships as a
`counter-reset` edge and leaves the finding active with the latest reading at
0. The variant now raises by one per read (the smallest honest increase) and
grades the snapshot of the product's records that first holds the drop:
`raise` (the raised readings accepted and one finding before the drop),
`reset` (the drop delivered with its `counter-reset` reason, the latest
reading equal to the reset value), `no_fault` (no finding at or after the
drop, no second incident), `no_recovery` (the raise's finding still active --
judged only while no later batch of the counter, such as a health summary,
intervened in the snapshot; otherwise recorded in `reset-readings` as not
judged) and `hardware`. `stale_sample`, `recovered` and
`independent_fatal_event` raise in order to mint and are unaffected;
`tests/regional/test_general_live_gpu_links_reset.py`. The variant is rerun
with the runner alone; no product release is needed.

## GF-GENERAL-LIVE verdict strength: must-fail validations, negative windows and positive controls

A stale-PASS audit of the board found oracles in the current runner that a
rerun would pass the same way whatever the product did. Each one below was
confirmed in the runner's code and is fixed in the harness; the product is
unchanged.

**019/020/022 diagnostic chains: "must fail" needs a verdict.**
`general_live_gpu_verdicts.chain_errors` accepted any VALIDATE_GPU status but
SUCCEEDED when the chain said the validation must fail, and 019
`corrected_ecc`/`aggregate_dbe`/`retired_delta`, 020
`crc`/`replay`/`recovery_counter`, the counter chain of 020
`independent_fatal_event` and 022 `over_rate_limit` judged the first read that
showed the incident. The product records VALIDATE_GPU WAITING until evidence
newer than the step exists (`adapters/gpu_validation`; a WARNING memory finding
first waits its 120 s transient-warning grace, then the next batch that
re-judges it), so a step that had judged nothing satisfied "must fail". The
general verdict path had been fixed the same way before; the GPU path had not.
Now the detection wait polls until the judged workflow has a finished
VALIDATE_GPU or has itself ended (`validation_judgeable`), within its budget
(019's diagnostic budget adds the 120 s grace, `VALIDATION_GRACE_SECONDS`, and
the counter rises for as long). A FAILED step satisfies "must fail" only when it
names the held reading -- `active_gpu_health_findings` in
`details.node_failures` and the held counter's canonical name on the held GPU in
`details.node_findings` (`held_reading_named`); a failure on another reading is
a FAIL. A step still WAITING when the wait ends, or one the executor's per-step
cap ended (`step_waiting_timeout_seconds`), never judged the reading: the
variant is INCONCLUSIVE with that reason (`ChainJudgement.unfinished`,
`inconclusive_unless_failed`) unless another stage holds a definite error,
which is a FAIL; `chain_errors`, for callers that take no INCONCLUSIVE, reads it
as an error. `tests/regional/test_general_live_gpu_must_fail.py` pins the
product side through the real DCGM collector delivery, `GpuMetricsService.ingest`
and `GpuValidationAdapter.execute` (corrected ECC WAITING on its first attempt
with `transient_gpu_warning_cooldown`, FAILED naming the held finding once a
newer batch re-judged it; the aggregate DBE, NVLink and PCIe counters FAILED
naming theirs) and runs every affected variant over the old shape -- the
incident with VALIDATE_GPU WAITING on every read, which passed -- now
INCONCLUSIVE.

**020 `recovered` / 022 `recovery_evidence`: the detection counts.** Both
held a counter fault, discarded the detection read and passed when
`recover_counter`'s last read had any accepted batch and no active finding of
any kind -- true of a product that never detected the fault. Now no incident on
the held counter is INCONCLUSIVE (there was no fault to recover from), the
detection is judged as the sibling diagnostic variant's (`crc`, `replay`,
`over_rate_limit`: the read-only chain whose VALIDATE_GPU failed on the held
reading; a wrong one is a FAIL), and the recovery needs a new normal reading
(`renewal_errors`): when the replay ends, the first batch at the device's own
reading is a `counter-reset` edge whose invalid delta keeps the finding active,
and only a later normal batch clears it, so the finding must be inactive and at
least two readings below the replayed peak accepted after it. "Real
communication" is read as the device's own readings accepted after the replay;
no traffic is driven. `tests/regional/test_general_live_gpu_recovery.py` pins the
drop and the clearing batch through the real DCGM collector, control-plane API,
store and `GPU_STATE_PROBE`, and runs both variants over the old shape (no
incident -> INCONCLUSIVE), a wrong detection, a validation passed while held, a
recovery on the drop alone and a finding that never cleared.

**019 recovery.** The diagnostic variants' `recovery` stage was
`[] if ... else []`, and `recover_counter` returns its last read when its wait
runs out, so "cleared by a fresh batch" was never checked. The stage
(`cleared_errors`) now requires that no finding of the counter is active on the
GPU and that the product accepted a reading of it below the replayed peak after
the peak.

**016 `user_text` / `old_boot`: the negative is proved on what the product
received.** The only stimulus witness was the journal wrapper's emission count,
and `no_finding` counted QUARANTINE incidents only. Both variants now read the
control plane's own stored NODE_LOGS batches (`NODE_LOG_ENTRY_PROBE`,
read-only): the reviewed record must be there under this run's replay cursor
with the trait the gate refuses -- a `journal` origin whose transport is not
kernel; a `dmesg` line stamped with the reviewed prior boot, not the
collector's -- or the variant is INCONCLUSIVE. A finding of any grade the
record produced (a marker naming its entry) fails the variant, as do a
QUARANTINE node-log incident in the window and an isolated node.
`tests/regional/test_general_live_kernel_negatives.py` runs both through the
real journalctl wrapper, `NodeLogCollector`, control-plane API and in-memory
store: the product stores the batch and grades it into nothing and the variant
passes; a batch that never reached the control plane (the old PASS) is
INCONCLUSIVE; a gate that let the line through as a diagnostic finding (the old
PASS again) fails.

**025 `sustained_unknown`.** The detection read was thrown away
(`_ = detection`), so a product that minted nothing passed on the UNKNOWN
observation alone. The variant now waits until the bounded chain has reached
its validation, as `exporter_timeout` does, and, with the exporter still
paused, requires the completeness incident, the planned TRIGGER_HEALTH_SNAPSHOT
then VALIDATE_HOST with nothing hardware, and a VALIDATE_HOST that ran and has
not passed; what the validation recorded at detection and after the sustain is
kept in the result (`validation`).
`tests/regional/test_general_live_sustained_unknown.py`.

**003 `ambiguous_ownership`: the negative window.** It read the node once,
LOW_DURATION plus three intervals (345 s) after the two jobs ran.
LOW_GPU_UTILIZATION is sustained (`GPU_FAULT_LOW_UTILIZATION_DURATION_SECONDS`,
300 s) on samples an idle node delivers only with its host health summaries
(270-330 s apart): the first sample graded for an attempt can come one summary
after the jobs ran, and the control plane mints on the first later sample at
least 300 s after it (`store/shared/xid.py` `_next_health_signal_state`) -- up
to 300 s plus two summaries, which the positive `gpu_idle` budget already
allowed. A wrongly attributed finding minted after 345 s passed. The variant now
watches `low_finding_window` (300 s + two summaries + one interval + 180 s
margin, 1095 s at the defaults), ends at the first such incident (FAIL), and
requires the product to have graded a low (at most 5 %)
`host_gpu_utilization_percent` sample on the jobs' GPUs at least 300 s into the
window (the latest-wins record); without one it is INCONCLUSIVE. Its
`max_minutes` is 35. `tests/regional/test_general_live_ambiguous_ownership.py`.

The DCGM replay, 003 and 016 result builders now record an INCONCLUSIVE's
reasons (`inconclusive`, `detail` and `inconclusive: ...` errors). The variants
above are re-run live on both clusters.

## GF-GENERAL-LIVE host-probe handle ownership and the dead-record sweep

Read-only on both clusters on 2026-10-26 (22:4xZ): seven host-probe Pods in
`gpu-fault-system` in phase Failed (container exit 143 -- killed by their own
`activeDeadlineSeconds` 30-45 min after start) with their ConfigMaps still
present, each with an ownership record `host-probes/<name>.json` left
`closed: false` in its case directory. Two leak paths, read from the records
and the lane logs. (1) The 024 composites (`thermal_warning`,
`power_performance`, HyperPod and plain EKS; five Pods) PASSed while their
`probe_residuals` named only `dcgm@<node>`: the leaked record's script digest
is `probes/env_replay_proxy.py`. `_composite_flow` closes the replay and
discards its undo (`replay.close(); ctx.cleanup.discard(CLEANUP_NAME)`);
`EnvReplay.close` restores the endpoint and stops the proxy but never deletes
the probe, `EnvReplay.cleanup` -- the one caller of `controller.host.cleanup()`
-- was the discarded undo, and the replay controller had built its
`HostProbeFixture` itself, outside `ctx.probes`, so the runner's teardown had
no handle to clean. The CRI Pod-log, kernel-replay and source-replay fixtures
built their probes the same way. (2) 022 `recovery_evidence` (HyperPod 14:19Z)
died of `OSError: [Errno 28] No space left on device` on the runner host inside
`_teardown` -- its dcgm handle was registered, but `_delete_owned` saves the
record before it deletes, so ENOSPC stopped the cleanup before the Pod delete
and killed the process before `result.json`. In both cases the rerun of the
case in the same run directory derives the same probe name (`run_identity` is
stable), met the open record carrying the earlier probe script's digest, and
was refused: `dcgm survey: host probe ownership record does not match this
run` (022, 22:40Z). 022's probe was a registered handle; only its teardown was
cut short.

Fix, harness only. Every host probe is the context's: `VariantContext.host_probe`
is the one way a variant or fixture gets a handle (`<purpose>@<node>`; an
existing handle under the key -- a test's fake, an earlier call's handle -- is
returned, a new one registered), `ctx.probe` and the environment-replay,
CRI Pod-log, kernel-replay, source-replay, EFA-driver, isolation and
device-plugin fixtures all go through it, and `ctx.probes` is a `ProbeRegistry`:
assigning a different handle under a held key cleans the earlier handle first
(two handles of one identity would compete for one record) and keeps it as
`<key>#<n>`, and the teardown cleans and lists every handle, live or replaced,
in `probe_residuals`. A handle that proved itself clean (a fixture's own LIFO
cleanup) answers the teardown's second `cleanup` from that proof until it
creates or executes again, and is uncreated afterwards, so a fixture that
recreates its probe after a node reboot gets a new Pod under a new record.
`HostProbeFixture.cleanup` goes on deleting what it owns when the record cannot
be rewritten (ENOSPC) and raises `host probe ownership record could not be
saved (...); the probe resources were deleted` afterwards. At variant start
`run_general_live` runs `scripts/e2e/regional/host_probe_sweep.py` over the case
directory's records (evidence `host-probe-sweep`, result `host_probe_sweep`):
an open record whose Pod is gone or terminal (Failed/Succeeded) is a dead
probe's -- its Pod and ConfigMap are deleted under the fixture's own proof
(case/run labels, owner annotation, script digest, node and node UID, UID
precondition) and the record closed with a `swept` note; a record whose Pod
still runs (Pending/Running) is reported and left exactly as it is, and the
fixture's refusal of a changed script against a live record stands; a record
another process holds, a terminal Pod or ConfigMap that is not the record's, or
a delete not acknowledged keep the record open and are reported with the
reason. The script copy a dead probe installed under `/run` cannot be removed
without a Pod: the swept record keeps its digest as `stale_host_script_sha256`,
the record the rerun mints carries it, and `HostProbeFixture.execute` lets its
install step overwrite a copy with exactly that digest (any other copy at the
path is still rejected) and drops the marker once a call installed the current
script. Tests: `tests/regional/test_general_live_probe_ownership.py` runs the
runner's `execute_variant` over an in-memory cluster with the dcgm probe and
each wrapped probe on one node and the wrapper's cleanup discarded -- zero Pods,
zero ConfigMaps, zero host-script copies, both records closed, both handles
listed with all-false residuals (red before this change for all four
wrappers); a replaced handle cleaned at replacement and listed as `#1`; a
proven-clean handle costing the teardown no API call; a rerun after a dead
probe sweeping its record and proceeding; a live probe of an earlier attempt
reported and untouched. `tests/regional/test_host_probe_sweep.py` covers the
sweep's outcomes (dead, absent with ConfigMap, running, pending, foreign Pod or
ConfigMap, unacknowledged delete, locked record, other context, closed and
unreadable records), the changed-script rerun refused before the sweep and
clean after it with the stale digest passed to the install step, a foreign
copy still rejected, and the out-of-disk cleanup that deletes, fails and is
closed by the next sweep. The seven leaked Pods and ConfigMaps were deleted by
hand before this change; no product release is needed.
