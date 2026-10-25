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
through a bounded, deadman-restored maintenance change; it proves the
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
training fixture, idle GPU duty) runs on the control node while one GPU of the
target node is replayed, and the scope judge grades the control GPUs, the control
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
device reads 0 % from field 10 over elapsed time). `reference_iostat`
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
interface's error counters stay flat (`INCONCLUSIVE` otherwise); the product
must also have accepted a host batch with a positive `tcp_retransmits_delta`
from inside the flow, because that metric deliberately has no rule and a "no
finding" the product never had the data for proves nothing.

`tc` and `sch_netem` are used as the node has them. Without the optional
`TEMP_TEST_PACKAGE_INSTALL` approval a missing `tc` keeps the pass-3
`NOT_RUN(CAPABILITY_ABSENT)`. With it the separate
`general_live_deps_probe.py` installs, for this run only, exactly `iproute-tc`
(and `kernel-modules-extra-$(uname -r)` when `sch_netem` cannot be loaded for
the running kernel) after a `dnf --assumeno` preview, and loads `sch_netem` only
if it is not loaded (`KERNEL_MODULE_LOAD`). Its limits are deliberate:

- The previewed transaction must consist of `Installing` rows of exactly those
  packages. A kernel, an upgrade (for example an `iproute` bump that a newer
  `iproute-tc` would require), a downgrade, a removal, an obsolete or any other
  dependency makes the variant `NOT_RUN(FIXTURE_NOT_READY)` with the transaction
  in `tcp-dependencies.json`; widening the allow-list is a separate review.
- The module package is only the running kernel's `kernel-modules-extra`
  build; a `kernel6.x` flavour would need `kernel6.x-modules-extra` and is
  reported as "no package ... in the enabled repositories", never guessed.
- What the run installs (NEVRAs from the rpm transaction that installed them)
  and which module it loads are recorded on the node *before* the change, and
  `cleanup` -- the runner's LIFO step after the fixture is gone, and a deadman
  timed to fire after the fixture's even when the install took its whole
  budget -- unloads only a module this run loaded once it is idle, and
  `dnf remove`s exactly the recorded packages after a preview that
  must remove nothing else (no dependent package, no "unused" dependency). A
  cleanup that cannot finish keeps its deadman armed and fails the variant's
  cleanup. A node reboot before either runs loses the `/run` record and the
  timers; the installed NEVRAs stay in the case evidence for a manual removal.

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
and every variant closes its window and proves the real tool answers again. A
replay proves the processing chain only -- no drive fails and no BMC exists, so
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
`bmc_status_unsupported` NO_DEVICE with the error gone. 015 `unsupported_bmc` is
the read-only capability record of the BMC-less site (like 002 `no_swap`): the
product must mark the node `bmc_status_unsupported` TOOL_MISSING (no
`ipmitool`) or NO_DEVICE (`ipmitool`, no BMC) on its routine host batches with
no `_bmc` collection error; FAIL without the marker or with the permanent
error, BLOCKED if a BMC answers.

The 010 EFA/RDMA counter variants are wired on the shared host source replay
(`scripts/e2e/regional/general_live_efa.py` on the reusable
`general_live_efa_fixture.py`; every one `live-replay` behind
`HOST_SOURCE_REPLAY`, on both clusters). Each starts from the node's real RDMA
inventory (the probe's read-only `inventory`: driver identity, owned netdevs,
port states, every `counters`/`hw_counters` file, the real `ethtool -S` names)
and replaces exactly one reading inside the host collector unit: a real port
counter file bound at its real value and stepped (`rdma_error`), the port
`state` bound to `1: DOWN` for a bounded exposure (`link_inactive`: the
window's own deadman closes it 1.7 collector intervals after it opened --
`general_live_source_replay.bounded_exposure_seconds`, 26 s at 15 s ticks --
so the first tick mints the `rdma_link_down` WARNING, the second is the fresh
reading its VALIDATE_FABRIC must fail on and the one that confirms the
`efa_inventory_mismatch` LINK_INACTIVE co-finding, and no third tick sees the
port down; since batch 175 the co-finding's diagnostic judges fresh readings
and would escalate a port still inactive to REBOOT_NODE by design, so it only
ever sees the restored port: SUCCEEDED, RECOVERED, no RESTART_NODE; a link
verdict reached after the restore, an inactive reading after it or no
`rdma_link_down` sample inside the window are INCONCLUSIVE),
or the reviewed `ethtool` wrapper serving, withdrawing or failing a group
counter on the netdev an EFA function owns (`rnr`/`retry`/`cq_error`,
`missing_counter` -> `efa_ethtool_counters_unsupported` MISSING_COUNTER,
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
`rdma_<counter>_delta`, `efa_<group>_errors_delta`) in that batch, one incident
per reader, no negative delta and never a clamped 0 after the reset. 009
`counter_reset` (the drop interface deleted and recreated) grades the same way
on `_network` and `network_drops_delta`, and is INCONCLUSIVE when the recreate
straddled a tick and the baseline was pruned (neither an error nor a reading). 011
`broken_pair` (`general_live_nccl_pair.py`, HyperPod lane) uses the same bind
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
with `/proc/driver/nvidia/gpus` hidden from the unit -- the two-read-failure
case), `read_failure` (`/proc/meminfo` hidden from the unit) and
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

004 `probe_error` has its faulting mount: the storage probe's loop +
device-mapper device carrying FAT16 (`dm-create --fstype vfat`) whose table
`dm-retarget --target error` swaps to the `error` target built into `dm_mod`
right after a fresh mount, so the mount stays, reads read-write in the host
table, and every `statvfs` fails EIO (FAT counts its free clusters from the
device at the first `statfs` after a mount; ext4 answers from memory and never
notices dm-error); `--target data` restores it and the probe's own reading is
the oracle (no EIO is `INCONCLUSIVE`). The swap is made with nothing dirty so
the error table is never written -- a failed write would log `Buffer I/O
error`, which the node log collector's storage rule grades QUARANTINE -- and
the variant refuses to run while that collector is active. It needs
`mkfs.vfat` on the node and `KERNEL_MODULE_LOAD` for a missing `vfat`.

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
dispatcher (a watcher restart mid-claim is 029 `worker_restart`, below). The
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
reaches the control plane while the block holds (the coverage heartbeat stops
advancing and the attempt has no event/decision -- otherwise INCONCLUSIVE), and
after the restore every action is recorded once: one containment with one
successful STOP_WORKLOADS, one decision (PLAN_CREATED: a STOPPED terminal
initiated by the passive containment is still the failure's terminal and
restarts once within budget), at most one RESTART_WORKLOAD, a ledger whose
count equals its reservations, no notification twice, and a coverage heartbeat
after the restore. Limits: the block holds the resolved IPv4 addresses of the
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
its UID/resourceVersion, with no grace, while the recovery workflow's
RESTART_WORKLOAD reservation is open (reserved on the ledger, step WAITING on
the replica); a claim already closed at the first read is INCONCLUSIVE, since
the replica's start decides the window. The replacement restores its state
from the product's own ConfigMaps and must not re-decide, re-reserve or re-send:
the ledger keeps exactly the one reservation of the recovery workflow (not lost,
not doubled), one RESTART_WORKLOAD succeeds, one STOP_WORKLOADS, one decision,
exactly one restart attempt is observed, no notification is recorded twice and
a coverage heartbeat follows from the replacement. The module tests drive the
sequence over fakes (a phase-driven store, an inventory that flips on the fenced
delete); the live run is the only proof of the real outbox replay.

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

011 and 030 run on the HyperPod lane (011 needs `--node-b`: a dedicated
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
the product's rearm window; 030 declares
one warm spare through the admin CLI under `DECLARE_WARM_SPARE` and always
releases it). 027 enables the optional reporter only for its disposable test
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
The role, thread, process and attempt-change
variants of 011 (`general_live_nccl_roles`, registered on both lanes: the
plain-EKS cluster's second EFA node is `--node-b`) are the same stall on the
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
stay `NOT_RUN` with an explicit reason in the denominator rather than being
dropped. No GF-GENERAL-LIVE case has live evidence yet.

## GF-GENERAL-LIVE-014 Kubernetes allocatable (native device-plugin freeze)

The five native 014 variants (`gpu_plugin`, `efa_plugin`, `one_sample`,
`replacement_uid`, `validation_failure`) are implemented in
`scripts/e2e/regional/general_live_device_plugin.py` with the probe
`probes/general_live_device_plugin_probe.py` and the `DEVICE_PLUGIN_RESTART`
approval. The spec forbids patching `Node.status` or faking allocatable, and the
COLLECT-017 B/C fixture excludes the DaemonSet by affinity and restores it
*before* the product acts (so the product takes the `already_healthy` branch and
never deletes a Pod); neither reaches the product's own pod-delete/re-register
path. The injection instead SIGSTOPs the plugin container's own process -- only
a process whose cgroup names the plugin Pod UID *and* whose argv basename is
allow-listed, deadman-backed with SIGCONT -- so the plugin stops answering
kubelet's `ListAndWatch`, kubelet de-registers the resource, allocatable decays
while the Pod stays `Running`, and the product must delete and replace the Pod
itself. `gpu_plugin`/`efa_plugin` grade the full `RESTART_*_DEVICE_PLUGIN` +
`TRIGGER_HEALTH_SNAPSHOT` + `VALIDATE_{GPU,FABRIC}` chain SUCCEEDED, a new Ready
`replacement_pod_uids` different from the deleted Pod, allocatable back to
baseline, incident RECOVERED and no reboot; `one_sample` resumes the plugin after
one mismatch sample to prove the `required_consecutive_samples=2` rule leaves no
finding; `replacement_uid` grades the new UID is recorded, differs from the
deleted one and is not re-deleted; `validation_failure` keeps the registration
broken across the replacement (re-freezing each new Pod) so the restart's own
allocatable snapshot never confirms, the step fails at `restart_timeout_seconds`
and -- tagged `failure_escalation_action=REBOOT_NODE` by
`orchestration/families/health._resource_escalation` -- escalates to
`RESTART_NODE` under the separate `NODE_REBOOT` authorization (flag `X`). The
effectiveness oracle is the raw kubelet allocatable reading: a freeze that does
not reduce it is `INCONCLUSIVE`, not a pass. Cleanup resumes any still-frozen
process and proves allocatable back to baseline with exactly one plugin Pod on
the node. Observed product gap: the final `RESTART_*_DEVICE_PLUGIN` step receipt
persists only `replacement_pod_uids`; the deleted Pod's `deleted_pod_uid` lives
only in the transient delete-attempt receipt
(`node_operations._restart_device_plugin_node`'s terminal branch drops it and it
is not in `step_bounds.ATTEMPT_DETAIL_KEYS`), so the runner grades the deleted
UID from the live-captured receipt when caught and otherwise from the Pod census
(the original Pod gone, a new Ready Pod present). `expected_unknown` /
`mig_timeslicing` (CAPABILITY_ABSENT) and `running_training` (FIXTURE_NOT_READY,
a managed scale attempt, lane N) stay `NOT_RUN` in the denominator. These run on
EKS first (no SageMaker health-monitoring agent) and on HyperPod with
`NodeRecovery=None`; no live evidence yet.

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
repoint. After the unit restarts the probe verifies through `nsenter` that each
replacement is visible inside the unit and not to PID 1, or closes the window.
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
the tree window's own deadman closes it 1.7 collector intervals (26 s) after it
opened, after the second tick that confirms the shortfall and mints the finding
and before a third could be read, which the runner's own polling (20 s store
polls plus two probe round trips against a 15 s tick) could not guarantee; the
runner's later close only confirms it (`closed_by`). PASS is a first workflow
that diagnoses (FREEZE_EVIDENCE + VALIDATE_FABRIC, no hardware or isolation
step), its VALIDATE_FABRIC SUCCEEDED with
`details.inventory_evidence[node][efa_inventory_active_count]` at the restored
count observed after the restore, no stale gate, the incident RECOVERED, no
RESTART_NODE and the healthy count read back; no finding inside the exposure,
or a shortfall reading recorded after the restore (the bound did not hold and
the designed escalation applies), is INCONCLUSIVE rather than a verdict on the
product. 013 `pci_missing`
(`scripts/e2e/regional/general_live_efa_pci.py`, both clusters, risk H, every
lock, `--approve HOST_SOURCE_REPLAY NODE_REBOOT`, expected PASS) is the
PCI_DEVICE_MISSING reboot chain on a live-replay: first the spec's negative
stage -- `build-pci-fixture --unreadable` makes the PCI-root fixture a regular
file, so the product's listing fails (ENOTDIR) under any privilege, held for
four host ticks (twice the two-sample confirmation window); the product must
record `EFA inventory observation unavailable` and no count, mismatch, finding,
hardware step or isolation (its error path also resets the confirmation streak,
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
Ready and schedulable with the EFA and GPU allocatable at baseline, fresh
post-reboot `efa_inventory_active_count`/`efa_inventory_mismatch`, the
provider's CloudTrail reboot record (EC2 `RebootInstances`, SageMaker
`BatchRebootClusterNodes`) naming this instance and no other, and `--node-b`
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
EFA inventory, a changed boot id, no isolation beyond the remediation chain's
own cordon), are scoped to one function, node-reversible (one write to
`driver_override`, or a reboot) and deadman-restorable, and their cleanup binds
the function back to `efa` or fails loudly. 032 `missing_version` stays
`FIXTURE_NOT_READY`: the NVLink5 decoder gates on the `driver_branch` the
*kernel collector's* own CollectorContext stamps on the XID event, and the host
source replay is allow-listed for `gpu-fault-host-collector.service` only.
needs other hardware; `observe_profile` / `permitted_isolation` need a Runtime
Profile authority the site does not configure, AUTH-015), the 017 attribution
variants (a managed attempt with CRI Pod logs, lane N territory), and 004
`probe_error` (the product reports a `statvfs` error as `filesystem_unavailable`
since batch 163, but injecting a real one needs a faulting test mount -- a
read-only file bind cannot, statvfs resolving the mount, not the bound file).

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
`NODE_REBOOT` for `recovery_failure`) and run only on the HyperPod lane's
exclusive node. 022's XID-join variants replay the DCGM last-XID gauge only as
XID 32 (`--approve DCGM_XID_GAUGE`); 019 `remap_failure` needs
`GPU_FIELD_DIAGNOSTIC`; 019 `pending_unsupported` (no legacy GPU) and
`observe_profile` (no OBSERVE reset profile) are `CAPABILITY_ABSENT`.

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
preflight is `NOT_RUN`. No case has live evidence yet. The four product gaps these variants
first recorded as expected FAIL -- the whole-batch (not per-GPU) DCGM
completeness check (025 `one_gpu_missing`), the PCIe<->XID gauge join with no
age check (022 `expired_xid`), the UUID-set-only identity comparison (026
`bdf_changed`) and the inventory baseline that never compared the source boot
id (026 `stale_boot`) -- are fixed by batch 160, so the variants now expect
PASS; a product that regressed would FAIL them.
