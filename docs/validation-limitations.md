# Validation limitations

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
HA-007's local signal handoff polls the event flag under the same deadline.
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
  damaged local storage, and needing one successful live post to wake replay.
  After that wake-up a single background worker continues bounded batches
  without waiting for another collection cycle; it stops on a zero-progress
  round and a later successful live post wakes it again. Completion Watcher
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
