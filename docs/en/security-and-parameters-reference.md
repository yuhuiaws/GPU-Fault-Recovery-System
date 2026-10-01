# GPU Fault Recovery System Security and Parameters Reference

English edition of `docs/安全与参数参考.md`; the Chinese file remains the source of record until both are maintained together.

This document collects the hard constraints and configuration boundaries that administrators must approve. For everyday lookups, start with the category-organised
[Administrator Environment Variables Reference](administrator-environment-variables.md); the complete machine-generated inventory is provided by the code-generated
[Environment Variables Reference](environment-variables-reference.md).

## 1. Production Constraints That Cannot Be Relaxed

1. Managed GPU HyperPod clusters must be EKS-orchestrated with `NodeRecovery=None`.
2. This solution never calls `BatchReplaceClusterNodes`;
   `GPU_FAULT_ALLOW_HYPERPOD_REPLACE` is always `false`.
3. HyperPod Job Auto Restart, EKS auto-resume and Slurm auto-resume must be disabled.
4. Unknown policy, missing evidence, version mismatch or unknown workload state must fail closed.
5. The CPU control plane holds no GPU kubeconfig; each cluster's Executor operates only on its local GPU EKS.
6. Each GPU cluster uses its own token, IRSA, namespace allowlist and Node Action key.
7. `curl -k`, `verify=false`, unsigned Node Actions and online widening of the allowlist are forbidden.
8. Production Agent endpoints must use HTTPS, and the control plane trusts only the node
   certificate carried by a signed heartbeat; every `RegionalClusterRegistration.agent_endpoint_allowed_cidrs` must be narrowed to
   the node subnet of that GPU cluster.
9. Workload ownership for node mutations accepts only active Observations refreshed within 120 seconds;
   stale records must be excluded from ownership and alerted separately. Fresh overlap is currently critical, and
   Alertmanager uses a 5-second group wait; a stale Observation is warning.

## 2. Destructive Capability Phases

| Phase | Capability | Proof required before going live |
|---|---|---|
| A | Observation, evidence collection, simulation | Events, allocation, Collector, Agent and evidence chain complete |
| B | Isolation, stopping and restarting workloads | Old and new attempts do not overlap, provider taint is not removed |
| C | quiesce, no-GPU-client validation, GPU/fabric reset | fail-safe timer, signatures, fencing and post-action verification |
| D.1 | HyperPod reboot | `NodeRecovery=None`, second cluster confirmation, fencing, provider idempotency and post-reboot verification |
| D.2 | warm-spare `nodeReplace` | `HEALTHY_WARM_SPARE_ONLY`, healthy spare of the same specification, atomic full allocation and post-replacement verification |

The control-plane allowlist, Runtime Profile owner or maintenance window must not be skipped just because the node-side switch is already on.
In the current `regional-hyperpod-safe` Profile, `nodeReboot` and `nodeReplace` are both
OWNed by `gpu-fault-hyperpod-adapter`, so D.1/D.2 are separately accepted sub-phases, not two independent
environment switches. When only D.1 is approved, a new Profile version must be released that keeps `nodeReplace` at
`OBSERVE`; do not rely on disabling the spare coordinator to prevent replacement.

D.2 only permits a local switch from the managed pool of healthy warm spares; the faulty node stays isolated and is not destroyed.
`GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER=true` is a safety lock that refuses to fall back to the provider replace API,
not a standalone action authorization; at the same time
`GPU_FAULT_ALLOW_HYPERPOD_REPLACE=false` must be kept. When healthy spares are insufficient the whole transaction fails; it must not be partially
activated, nor may `BatchReplaceClusterNodes` be called.

`BatchReplaceClusterNodes` is forbidden not only by environment variables and IAM: the runtime provider client
protocol has removed that method, and any `HyperPodAction.REPLACE` entering provider submit fails directly.

### 2.1 Regional Acceptance Fixture Boundaries

The plan of a live runner must bind the target, the kubeconfig content, the current source code and a successful preflight;
read-only analysis or a local-proxy PASS does not authorize real actions. Loss of supervision, target UID drift, read failure or uncleaned
residue all forbid continuing, and the formal sequence must not be resumed by hand-editing a verdict.

WORKLOAD-001/002 must execute the real training submission entry point from the README; internal functions or
a create stand-in may not be used to prove the apply path. Create a temporary namespace-scoped get/create identity for this test only,
first prove it has no patch/update/delete permission, then perform the submission through a private impersonation kubeconfig;
the original deployment identity is used only for the preflight and for cleaning up the resources owned by this run by UID/resourceVersion.
Stop when a same-named object appears, permissions widen, a read fails or the cleanup identity is unclear; do not modify existing workloads,
component ServiceAccounts or the namespace allowlist.

Cleanup of a managed test workload first keeps the verified delete-only record, then removes the controller with UID/resourceVersion
conditions and the `Orphan` policy, and finally explicitly deletes the Pods with zero grace period, so that the Job controller does not
recreate Pods. Cleanup after the parent chain has disappeared still validates the saved UID, content digest and owner references;
only the normal orphan removal of owner references and runtime finalizer changes are allowed, not new ownership or content drift.
Version conflicts are re-read a limited number of times only within the same verified object; unknown create results, read failures and identity changes still stop.
Before submission, only the empty annotations map that Kubernetes would normalise away is omitted; non-empty declarations must still be kept.
A controller whose creation is confirmed but whose declaration validation failed may only be rolled back with foreground cascade
while the current content still matches the original ACK, and its child resources are not registered as approved on the basis of its parent chain.

Hot-registration credentials of the CMD synthetic identity enter the CPU probe only through protected stdin and are never written to argv, the environment,
source code or logs; they must match the current registry revision, a separate current token and this run/validity period.
The probe may only use a synthetic identity with loopback Agent scope and no Agent registration, and cannot use it to assert that no physical consumer
exists. A forced termination by the CPU-local watchdog is not proof of cleanup, and the cumulative cleanup budget cannot be extended;
data and registration deletion still depend on the original registration receipt and evidence of a stopped producer.

The probes of NOTIFY-008 and HA-011 use only a PriorityClass that is exclusive to this run, priority 0, non-default and non-preempting;
borrowing a system component class or removing the non-preempting constraint to bypass admission is not allowed. The class's namespace owner,
creation UID and pre-deletion version must be checked, and the disappearance of the namespace does not excuse skipping the class residue check.

CAP-002 uses an exclusive ADOT scraper and the probe's own token file; copying the production execution token
into the capacity fixture or switching to anonymous metrics is forbidden. The scraper reads only this run's probe and does not modify production ADOT, IAM or
Secrets; the plan must bind the collector identity in the deployment. First prove that this run's metrics are fresh, return to zero after release and that the
alert resolves, then clean up the scraper and dependent resources by UID. When process stop cannot be proven, dependencies must not be deleted before reporting
success. The workspace's global capacity alert may still send notifications along the existing SNS route.

The temporary plugin affinity experiment of COLLECT-017 supports only DaemonSets pre-approved and configured as `OnDelete`;
the runner is not allowed to convert `RollingUpdate` online. First confirm that the GPU local recovery watchdog is
ARMED, then perform the UID/resourceVersion conditional update; the original affinity must be fully restored and verified before the
watchdog is cancelled. Approve only least privilege for this run's resources/nodes, keep the recovery log on failure and do not swallow cleanup errors.

COLLECT-022 uses only this run's private FM log, checkpoint and a non-forwarding sink. Testing is no reason to
delete the real collector cursor, modify its service, or describe a private software log as a real hardware fault.
The dedicated-table acceptance of BOOT-030/031 only connects to the writer read-only and does not automatically advance migrations or clean up old data.

The live publish of COLLECT-004 still triggers a real reboot; it is not a read-only case. The new path uses `baseline+1` only in an isolated Collector
instance, and the private sink has no network access; only after sampling completes may the two saved batches, bound to the original Node,
boot, production configuration digest and collection process, be delivered to the real entry point. The production `collector.env` and Host
Collector service must not be modified for this, and a second continuously forwarding producer must not be started.
The publish must be explicitly confirmed and match the sampling digest, keeping the original batch IDs; on a lost response only the persistent evidence may be queried, without blindly
resending, fabricating receipt or cleaning up unknown workflows. The end state still requires exactly one recovery workflow SUCCEEDED, a new boot,
an unchanged configuration digest and complete probe cleanup. The CloudTrail proof is strictly bound to the approved cluster, a single instance, the full calling
role and the successful submission window; counting alone or comparing role short names is not enough.
The recovery compatibility path of the legacy override journal still requires the original ownership, a private backup, a fixed deadline, the node lock and
complete CLEANED proof; the old guards must not be relaxed, nor past identities fabricated, because the new case no longer depends on that path.
Isolated sampling, a simulated lifecycle and local platform checks do not constitute a COLLECT-004 live PASS.

All synthetic insufficient-warm-spare requests of DESTR-008 must keep the behaviour-limiting-only `activation_forbidden=true`.
Old Executor protocols must not claim such commands; no real activation or cached activation refusal may
impersonate a passing health check of the scenario itself. The three time-limited scenarios also need an independent CPU cancellation watchdog and a Node-UID-bound
GPU admission guard; an admission dry-run does not mean the whole API Server cache has converged, so it
cannot replace the product's internal immutable inhibition. The fixture is not restored while the original producer is not withdrawn, the POST origin is unknown or the command is not
silenced; the original run allows only cleanup resume. See
[DESTR008 Cancellation](../components/destr008-cancellation.md).

DESTR-014 may remove the original Agent enablement link only after the persistent independent recovery service/timer has explicitly reported ARMED for the original Node, boot, release and runtime.
The fixed recovery deadline cannot be extended; after loss of contact only the
same Agent is restored, with no authorization for reset, reboot, warm-spare switch or workload restart. Expiry, identity drift,
unknown supervision or an unstopped systemd job cannot issue cleanup success. An ordinary transient timer and
the old unbound restore command cannot replace this protocol; see
[Reboot-Surviving Recovery](../components/destr014-recovery-safeguard.md).

The physical-ownership companion of DESTR-015/PREEMPT-033 must wait for the ordinary DESTR-015 to pass and finish cleanup,
then be approved separately for the three independent variants; PREEMPT-033 references the same set of evidence and does not repeat the same physical experiment.
Production Agent protocol 4 waits at the post-dequeue physical checkpoint for a single-use permit signed with the node key;
the challenge lasts at most 90 seconds and no longer than the original command deadline, and the permit at most 3 seconds. Before issuance the lease and the workload/Pod/Node ownership of all
participating nodes are rechecked, and before execution the local generation, fencing,
quiesce, clients and permit validity are checked as well. A refusal result cannot trigger reboot/replace escalation.

A malformed response after the permit may have been delivered, a failed result query and contradictory responses must keep the original command ID,
intent digest and result-unknown state; only the original command is queried, and no substitute command is resubmitted. When the original phase, fencing,
parameters or GPU scope change, an old result cannot be taken as success of a new action. A negative permit does not override execution facts the Agent has already returned;
when a later check refuses, the previously permitted checkpoints are kept and it cannot be claimed that the whole command never started.
A step or workflow timeout does not prove the physical action has stopped. While unconfirmed node actions remain, the control plane must not automatically restore
GPU service or release the node occupation into an ordinary terminal state; it should keep `BLOCKED / NEEDS_OPERATOR` and wait for reconciliation.
A confirmed ordinary failure or an explicit pre-execution refusal can still be compensated; the node's existing quiesce failsafe is unchanged,
but it cannot be treated as proof that the physical action completed.
WAITING records of regional single-step and compound commands must carry these physical states. A LEASED change may have been submitted later even if the last preflight
had not yet started; after cancellation and before a terminal state/automatic recovery, the remote evidence must be rechecked.
A transient read failure permits only retry and keeping the occupation; it does not authorize restoring service. DAG dispatch must also use the current plan's
dependencies, target scope and decommission state, and may not submit actions already withdrawn from a cached ready list.

An `nvidia-smi` timeout of both a single-GPU and a whole-node fabric reset saves
`outcome_unknown/manual_confirmation_required` together with the actual per-GPU progress, carried through the Agent,
adapter and workflow. In mixed old/new version operation, a non-empty `reset_outcome_unknown` in an old result is also treated as unknown;
old whole-node results without structured progress recognise only the explicit `ResetOutcomeUnknown` exception category and do not infer the execution result from arbitrary
timeout wording. An unknown state keeps the node occupied and waits for manual reconciliation, without automatically restoring GPU service or escalating the action.
The retained non-regional multi-node barrier compatibility path uses the same normalisation rules and keeps per-GPU progress of failed nodes in
`node_failure_details`; it cannot downgrade a legacy unknown result into an ordinary failure that can be compensated automatically.

A confirmed reboot in the same workflow can explain a boot change, but the original STOP receipt cannot be overwritten or reissued.
The `stop_reboot_authorization_v1` recorded at submission must bind the original receipt digest, Node UID, old boot,
workflow/phase/fencing and execution epoch/owner, and then correspond to the post-confirmation Agent/provider observation.
Polling reuses the original binding; old code lacking that proof, an external reboot, Node re-creation or workload/Pod drift is still refused.
When the provider has accepted the reboot but the original resume proof is lost, even if an idempotent replay returns a cached result, the proof cannot be back-signed from later
Kubernetes reads; return `STOP_REBOOT_AUTHORIZATION_MISSING`, keep the result unknown,
wait for manual reconciliation, and neither resubmit the reboot nor escalate to replacement.
This does not constitute atomic mutual exclusion between external ownership writes and provider calls.

Distributed reset computes the expanded size of the per-node DAG at compile time, sharing the 256-step cap with the executor.
Exceeding the cap does not truncate the node set, nor is it saved as an executable reset; only evidence collection,
cordoning and isolation are performed when the original Profile allows, then `BLOCKED / SAFETY_SETTLED` is kept. When isolation cannot be performed, it goes directly to
`BLOCKED / NEEDS_OPERATOR`, and in both cases the manual recovery reason is kept.

Terminal quarantine is not a read-only event, and it cannot be overridden by the temporary isolation before a reboot.
`terminal_quarantine_node_ids` in the merged and successor records binds the faulty nodes; unsubmitted recovery steps can be narrowed or withdrawn,
but steps already submitted, executed or decommissioned must not rewrite identity. Warm-spare recovery must follow the binding of the real node replacement, and
isolation and evidence collection added later still target the original faulty node. A REPLACE plan or success marker alone, without a real
move to a non-isolated target, cannot release recovery scheduling of the original node; the refusal happens before any patch or no-op.
Terminal-state and ownership reports also use node scope: A's recovery cannot offset B's terminal quarantine, and unexecuted or decommissioned
recovery steps do not count as completion proof. An incident historically mislabelled RECOVERED cannot by itself authorize takeover,
and the successor must still keep the isolation constraints of the actually managed nodes.
Recovery completion proof must bind the phase, step index and the successful execution receipt of the operation; a partial completion index,
an old digest without phases or a standalone success status cannot release isolation. A verified SPARE_FAILOVER can make the
incident recover and allow dependent tasks to continue on the warm spare, but the original faulty node still keeps ownership isolation; explicit
continued isolation or other unrecovered branches are not offset by warm-spare success.

A successful reset result distinguishes `reset_successes` from `reset_busy_refusals`. A busy retry still performs the original
identity and client checks and does not add to the budget of successful physical actions; when multiple calls lack consistent structured counts,
have unknown/failed/unexecuted GPUs, or exceed the fixed retry cap, acceptance cannot judge it a single successful reset.
The runner does not restore quiesce directly to complete cleanup; it keeps the product's own compensation, failsafe and manual reconciliation boundaries.
The CPU recovery entry point and the runner use the same `recovery_safety` criterion, including the legacy
`reset_outcome_unknown` and nested node results. A SQL timeout cancellation or terminal state does not equal physical completion, and
the manual protection already recorded by DESTR-014 is not lifted on its own by later terminal-state reads.
Cleanup of an existing incident must obtain real drain proof; a read failure cannot authorize deleting the
workload or restoring isolation because the holder is not enabled. Only the preparation-failure path with no incident yet and the holder not enabled may be cleaned up directly.

This experiment introduces a controlled change after enqueue and before permit issuance, and uses an independent continuous execution trace to judge whether the action boundary was crossed.
The sequential Kubernetes read, permit issuance and the actual system call are not an atomic transaction; it cannot be claimed on that basis that
all changes occurring after any given read can be prevented. Missing trace or cleanup proof must not issue a PASS, and the companion
must not override the ordinary case result. The exact boundary and manual reconciliation conditions are in
[Physical Late-Ownership Acceptance](../components/late-ownership-acceptance.md).
This solution adds no new requirement for external Operators, schedulers or business clients to join a unified ownership write protocol,
change their write entry point or modify permissions; it cannot be assumed that they will cooperate by pausing writes. The existing safety prerequisites such as `NodeRecovery=None` and
disabled auto-restart are unchanged, and detected ownership drift, read failures and unknown results are still refused
or kept for manual reconciliation under the existing rules. This boundary does not mean that all changes after any given read can be detected.
Full atomic mutual exclusion is still not implemented; the evaluated but not adopted coordinator design and its platform prerequisites are in
[Atomic Ownership Fence Design](../components/atomic-ownership-fence-design.md);
it is not a current deployment prerequisite, a deployed capability or proof of whole-API-Server effect. The original scenario gaps and acceptance gates
remain unchanged and are not counted as passed because that architecture is not adopted.

## 3. Configuration Sources of Truth

| Configuration type | Normal source of truth |
|---|---|
| Regional topology, identities and health targets | The internal site generated by the CLI in the `state-dir` |
| AWS site infrastructure | ARN bootstrap state and the Aurora installation resource registry |
| Low-level release inputs | The `regional-release.json` temporarily generated by the CLI |
| Runtime Profile | Developer-maintained template; `release-deploy` generates an immutable source snapshot and a content-addressed version |
| Runtime Profile approval | `state-dir/release-deploy/profile-plan.json`, the active approval and the consumption records archived by plan digest |
| release artifact | `dist/current-release.json` |
| GitHub main CI candidate read-only credential | `GPU_FAULT_GITHUB_TOKEN_FILE` with permission `0600`; used only for clean `HEAD == origin/main` candidate recovery |
| Administrator capacity desired values | `state-dir/admin-config/desired.json`; generated by the CLI, digest-verified, contains no Secrets |
| Administrator capacity plans and approvals | `state-dir/admin-config/pending.json` (the one in-flight or interrupted apply), `desired.json` and the audit archived by plan SHA |
| CPU non-sensitive configuration | ConfigMaps split by core/postgres/processor/telemetry/notification/recovery |
| Processor retry and completion concurrency | Generated from the role-based processor ConfigMap; production backoff `1..30s`, completion cluster concurrency `1`; CAP-005 must pass before raising it to 2/4 |
| Remediation concurrency budget | `GPU_FAULT_REMEDIATION_MAX_ACTIVE_*`; production defaults are fixed in code and only explicit overrides enter the configuration; Region, cluster, node, failure domain and resource class claims persist with the workflow lease. The failure domain comes from the fabric partition of the SXID and the node mapping ConfigMap `gpu-fault-failure-domain-map` rendered automatically by deploy/join/remove (label keys taken from site.yaml `spec.failureDomainLabels`; `GPU_FAULT_REMEDIATION_FAILURE_DOMAIN_MAP`, optional mount, must be valid when present, and the render side validates it with the same loader before apply) |
| Secret | Kubernetes Secret, Secrets Manager or controlled file references |
| Regional registry runtime | Aurora immutable revision/head; the Secret is for bootstrap/disaster recovery only |
| Node configuration | `/etc/gpu-fault/*.env` generated by the Installer |
| Common administrator parameters | [Administrator Environment Variables Reference](administrator-environment-variables.md) |
| Complete machine inventory | [Environment Variables Reference](environment-variables-reference.md) |

Administrators must not hand-edit the internal site, the low-level release JSON or `regional-env.sh`, or use online
`kubectl set env` to change identities or policy. `regional-env.sh` and the direct release JSON are only for the detailed
manual procedure and break-glass.

Unified startup validation requires numeric environment variables to be finite numbers; `NaN`, `Inf` and `-Inf` cannot be used as timeouts, capacities or
thresholds to bypass range checks. Errors report only the variable name and constraint, never echoing the value; `warn`/`ignore` affect only unknown variable names
and cannot let an unusable numeric value through.

The complete environment variable inventory is not a general-purpose write interface. Administrator configuration exposes only the 22 control-plane capacity,
Aurora Min/Max, queue/retry, Workflow scheduling, notification delivery and evidence retention fields of the formal template; credentials, identities,
Runtime Profile, operation
allowlist, release pin, destructive switches and safety invariants must continue to use their own dedicated sources of truth. A first deploy can import a strict YAML with permission `0600` via
`deploy --config`; an existing site edits
`<state-dir>/admin-config.yaml` and then runs `gpu-fault-admin config` with an approval reference. The plan SHA
is generated and archived internally by the CLI and is not an administrator parameter.

CPU capacity configuration takes effect through a rolling restart and does not support general no-restart hot updates; Aurora Min/Max is adjusted online through an audited RDS transaction
that requires both instances to converge and evidenced failure compensation. Failure of the release subprocess does not prove the original role has recovered;
when the candidate has been submitted, the rollback is incomplete or the live result is unknown, the pending target is kept and Aurora capacity is not lowered automatically.
Only matching unchanged or completed-rollback evidence can authorize restoring the old window. Only after independent configuration revisions, ACK from all CPU processes,
mixed-revision gates and rollback are explicitly implemented in the future may other allowlisted fields be marked hot-reloadable;
cluster registry records must not be reused to bypass these gates.

## 4. Release and Runtime Profile

- The Control Plane, Executor, Node Runtime wheel and Node bundle must come from the same
  signed schema v4 release manifest; the Manifest must also cover the rendered resource digests, the renderer,
  `uv.lock`, the Node template, the image digest of every role and the node offline dependency inventory. Old schema v3
  shared-image releases continue to be read and rolled back, but no new independent image identity may be guessed for them.
- The runtime image must have its locked dependencies pre-installed by CI. Production Pods and the Schema Job must not run
  `pip install` at startup; the deploy host verifies the attestation signature and digest. In schema v4 the CPU and Executor each use an independent
  OCI image and `/opt/gpu-fault/control-plane`, `/opt/gpu-fault/executor`, containing only their own
  component wheel; the system Python must not fall back to the full project package. Independent images do not change the required/compatible
  pin or the protocol window; when the previous snapshot lacks the v4 independent Executor identity, rollback must be refused.
- The node wheelhouse is distributed as a digest-pinned OCI artifact and mounted read-only; before installation, the
  inventory bound by the signed Manifest, every file, the platform and the lock in the candidate bundle are verified, `--no-index --require-hashes` is enforced,
  and py-spy still has its binary digest checked. An old bundle must use the old dependency identity and must not be mixed into the candidate wheelhouse.
  Even when the bundle content is identical, rollback selects the old dependency image through the explicit previous target; a missing or mismatched
  cluster/bundle/dependency binding must fail. The deploy-host image cache only accepts private HMAC receipts and fresh pinned OCI
  identity checks; its authentication key must not enter source code, ordinary artifacts or release assets, and the cache does not replace release signature verification.
- previous also stores the SHA of the complete template content already trusted by the old Reconciler, kept distinct from the template source SHA.
  Both capture and restore verify the actual Job bytes of the ConfigMap as well as the dependency image, inventory and read-only mount; the current
  live content must not be re-hashed and then treated as old-version authorization. An explicit template override must carry a trusted content pin, and
  preflight checks the template actually selected; managed entry points clear inherited template override variables and fail closed when proof is missing.
  Before restoring the CPU, the refresher or the Profile, rollback first verifies all cluster templates still awaiting AGENT/RECONCILER
  compensation; it re-verifies before each per-target execution, and a CPU-only scope does not read unrelated GPU templates.
- The "already installed / reusable" hint returned by the product is not a security proof. The Shell entry point must still perform the full Node key byte
  digest/ownership check and verify the trusted content pin of the template actually selected; no hint may skip any gate.
- Dirty source code may only be turned into an isolated `staging_only=true` test release by the internal preparer of the unified
  `gpu-fault-admin deploy`. Ordinary
  `release-deploy` and production signature verification refuse that level by default; only the final clean commit and Release CI may
  produce a production release.
- Runtime image reuse may only be based on the complete image input digest: Dockerfile, pinned base digest,
  runtime lock, platform, build args, control-plane/executor wheel SHA and module
  digest. When the tag, platform or any config label in the Registry does not match, it must fail closed.
- The BuildKit cache must be separated from the immutable runtime repository. The cache repository allows mutable
  tags but only for layer acceleration; ARN bootstrap only cleans untagged cache artifacts older than 7 days and keeps
  the current cache tag. When an existing lifecycle policy is inconsistent it must fail closed, and the release Manifest may only ever
  reference verified immutable OCI digests.
- Capacity stress tests use an isolated control plane/registry by default. Synthetic clusters must carry a run ID and TTL;
  live registry mutation requires double confirmation, and every execution path must complete a retryable,
  verifiable teardown in `finally`. Residue must not be hidden by making status ignore `perf-cap-*`.
- The production deploy host must install its Python environment from a Cosign-signed offline deploy-host bundle. The bundle contains only
  hash-locked wheels, the project wheel, the tool manifest and optional audited binaries, and must not contain credentials. Online initialization is
  only for development checkouts; `gpu-fault-admin deploy` must not install or upgrade host software.
- The administrator CLI must come from the independent `gpu-fault-deploy-host` distribution; the production Control Plane
  Runtime wheel/image must not contain the `gpu-fault-admin` entry point, `admin_cli.py` or
  `deploy-host-tools.json`. Deploy-host code changes must not masquerade as application Runtime changes.
- The locally built temporary PostgreSQL listens only on loopback; the password must not enter Docker arguments, URLs or ordinary artifacts;
  a private authorization directory outside the repository is used. Even after the exact CID, image and ownership are confirmed, cleanup must still prove the resource no longer exists;
  an unknown creation, lost supervision or failed cleanup does not allow the build to be declared successful, and refusals cannot be bypassed by taking over by name or deleting the directory.
  PG test subprocesses use their own credential HOME and do not change the HOME used for ECR, Docker builds and signing.
- The GitHub candidate token must not be written to the site, source code, ordinary artifacts, logs or command arguments; when candidate recovery lacks credentials
  or the API is temporarily unavailable it may fall back to the full local gates, but when the signature, repository, commit/tree,
  gate or file manifest of an already downloaded candidate does not match it must fail closed.
- The physical wheel SHA-256 proves the actual file; the component `module_digest` proves the behavioral content; the Agent and
  Executor required/compatible windows verify both.
- The Runtime Profile is shared across the regional control plane by `profile_version`.
- A Profile policy change must first have a plan generated by deploy, then be written as a one-time state approval and continue the deployment through the same
  `gpu-fault-admin deploy --state-dir ... --approve-profile-plan <plan_sha256> --reference ...`
  command; the command must compare the `plan_sha256` that the administrator reviewed and recorded in the change ticket with
  the current plan under lock, and only allow the write when they match. There is no separate approval verb;
  environment variables and hand-written JSON must not replace this explicit argument. The plan must directly display
  `site_identity.site_name/aws_region/cpu_eks_arn` and verify them with `site_identity_sha256`;
  the approval is also bound to the source/target versions, the policy/template/snapshot digests and the live baseline, and any drift must invalidate it and require re-review.
- The Profile approval is consumed only after verify, stability and commit succeed. When the release fails and the plan has not drifted it is kept for
  idempotent resume; after successful consumption or replacement by a new plan it must not be replayed. Approval and release must be serialized by the same state-dir file
  lock. For the full administrator commands and exception handling see
  [Runtime Profile Change Approval](administrator-profile-change-approval.md).
- The PostgreSQL schema is currently v18 (as defined by `schema_migrations.LATEST_POSTGRES_SCHEMA_VERSION`); production business Pods run with
  `GPU_FAULT_POSTGRES_AUTO_SCHEMA_INIT=false`, and DDL may only be executed by the ensure Job.
- The workflow/remote_command dedicated tables stay legacy by default. Data migration is executed explicitly by the audited Store CLI and
  is not switched automatically by deploy; before dedicated, the canonical JSON/missing/difference checks must complete and active commands and
  workflow leases must be drained. After the switch, old-table writers are refused by the database barrier and old writes cannot be restored through a rollback mode;
  old rows and indexes may only be decommissioned explicitly after the stability period.
- The dedicated-table maintenance entry point holds the schema shared barrier and verifies the release version, the complete migration history and the state table definitions.
  Old tools, missing or disabled triggers and function drift must not enter the mode change, backfill or decommission flows.
  A mode switch that encounters an active writer refuses immediately and asks for a retry instead of queueing into a lock loop of old writers;
  entering dual clears the non-authoritative copy through a restricted DELETE and no longer uses TRUNCATE. When locked copy rows
  cannot be cleared the whole transaction rolls back; partial activation is not allowed.
- Regional authentication and Processor ingestion share a bounded request-body reader; `Content-Length` is only used for early refusal, and
  chunked requests and requests without a length declaration are still memory-bounded by the bytes actually received. Reads use the smaller of the remaining request budget and
  a 30-second read ceiling; a timeout returns a retryable 503 and no half event is accepted. gzip is limited separately on wire bytes
  and on decompressed bytes, returning 413 when exceeded; a successful read still keeps downstream decoding bound to the cluster identity.
- Content of the same version cannot be overwritten; a policy change must use a new version.
- The CPU required profile, the Collector, the Watcher workload annotation and the Node Agent config
  must reference the same version.
- Before a Profile change finalizes, the old Profile's non-terminal workflows and PENDING/RUNNING workloads must
  all reach zero; the current implementation does not allow an implicit two-version transition.
- When a reset, fabric reset or failed post-action verification escalates to reboot/replace, if the new workflow contains
  `RESTART_WORKLOAD`, it must inherit the same set of `cluster_id`, `job_id`,
  `source_attempt_id`, `source_gpu_count` and `restart_budget` from the failed workflow. When a field is missing or multiple sources disagree,
  the escalation workflow must stay `BLOCKED` and must not use default values to re-authorize a training restart.
- Node Runtime upgrades advance by `FleetDeployment.max_unavailable/waves` while also verifying the
  artifact, bundle, template and signed ACTIVE heartbeat.
- A shared release upgrades only one GPU cluster at a time by default; cross-cluster parallelism is enabled explicitly by the site-configured
  `upgradeMaxParallelClusters` (default 1, maximum 8); it is not state semantics and does not change
  automatic rollback: rollback always runs serially in reverse order of actual mutation time. A failure in any cluster blocks the clusters not yet started, and
  clusters already in progress finish their current wave sequence and then uniformly enter `PAUSED` or `FAILED`. In the site configuration,
  `upgradeMaxUnavailable` defaults to 0 (auto, taking the scale ceiling) with a maximum of 32; the first upgrade wave is fixed at 1; an UNKNOWN failure domain forces 1,
  a known failure domain adapts the single-cluster ceiling to 4, 8, 16 or 32 by node count and distributes evenly, and an explicit 1..32 takes the smaller of the configured value and the ceiling.
  `rollbackMaxUnavailable` defaults to 2 with a maximum of 4; the first rollback wave is fixed at 1; fewer than 6 nodes, an UNKNOWN
  failure domain or incomplete evidence forces 1, 6 to 31 nodes allow at most 2, and 4 is only allowed with at least 32 nodes; with multiple failure domains, at most 1 per domain per wave.
- A candidate that touches the GPU or the Node Runtime must complete
  `candidate-preflight-ready` before the schema Job and the CPU rollout: run a server-side dry-run against the candidate GPU/DCGM/Reconciler/Installer Manifests,
  verify the Action key/Secret references, and prove the bundle/wheel
  digests, disk, Python/systemd/driver tools, quiesce state and old runtime rollback slot through a read-only host preflight. When that gate fails,
  CPU/GPU processes must not be rolled, the Reconciler must not be applied and no Fleet wave may be created.
- GPU/DCGM Manifests must be server-side dry-run again before the actual apply, to prevent admission or RBAC drift that occurred after the early gate
  from causing partial resource mutation.
- The read-only candidate preflight runs at most 4 clusters in parallel and at most 8 nodes of Host preflight Jobs concurrently per cluster; the private topology snapshot contains no credentials,
  may only be used for rendering and read-only preflight, and cannot replace fresh Node, workflow and lease checks before the real installation.
  Batched server-side dry-runs are limited to 4 nodes, 8 Jobs and 1MiB per batch; the host
  preflight Jobs are created only after all objects pass, and a successful batch cannot mask a partial object failure. admission runs at most 8 batches in parallel independently, still sharing the whole-deploy
  kubectl quota; this read-only request window cannot replace the Host execution window or the real node unavailability budget.
- A nested CLI on the deploy host that borrows and returns API quota must verify PID/starttime, ancestry and paused state; it must not trust the PARENT
  environment variable and let the call through directly. It may only pause CLI processes spawned by this run whose identity matches, and must not operate on GPU Pods or host services;
  without quota it must not resume a paused caller. Anonymous authentication output may only be staged in a 0600 temporary file, stdout/stderr
  combined must not exceed 8MiB, and when exceeded the partial output is cleaned up and discarded and does not enter the ledger, ordinary artifacts or logs.
- The API-quota SQLite file is only for temporary cross-process coordination within a single deploy-host command; it is not the production Store and is not deployed to
  the CPU/GPU clusters; production business data still uses Aurora. Protocol v4 persists `stopping`, which still holds quota, before STOP, and only records
  `parked` after all threads have stopped; returning first commits `resuming`, which counts against capacity, and then sends SIGCONT; crash recovery
  or sibling branches may not bypass that order. The ledger stores no credentials, is deleted on normal exit, and abnormal leftovers cannot prove that a subsequent deploy succeeded.
- The inner and outer subreaper layers of process supervision each provide an independent completion proof; when one layer is lost, the surviving owner must prove
  that the whole descendant tree has stopped and been reaped before ordinary failure compensation may begin. When all valid proofs are lost,
  `ProcessSupervisionLost` is raised, the current process is forbidden to keep issuing commands and to auto-rollback, "the supervisor process has exited"
  or "the output pipe has closed" must not be treated as successful cleanup, and restarting the CLI alone must not be used to presume that the original command stopped.
- The whole operation is protected by `interruption_scope`; SIGINT requests cancellation of supervised non-cleanup commands and fails at a safe
  boundary; repeated SIGINT cannot release the site lock early or interrupt descendant reaping. Command environment preparation, quota waiting and startup
  share an absolute `expires_at`; compensation/cleanup may only use its own bounded window and cannot bypass supervision loss.
  This contract does not promise hard preemption of arbitrary threads in the parent process or of non-terminable kernel waits.
- Grafana/AMP networking and HTTP framing execute under supervision in the private stdio worker of `native_http` with `-I -S -B`;
  authentication headers, bodies, proxy authentication and responses must not enter argv, the ledger, ordinary artifacts or logs. AMP's
  `aws configure export-credentials` is likewise bound by supervision, admission and the output ceiling; after export only the frozen
  credentials are used for signing, and no new environment variables are written nor is an SDK credential refresh re-triggered. Credential resolution and the AMP request share a query window of at most 30 seconds,
  and private messages and responses have an 8MiB ceiling; a timeout, ceiling breach or TLS verification failure must not be let through.
- ADOT/AMP candidate updates for an existing release must take place after previous is captured and must not be overwritten early by bootstrap.
  Read-only drift must enter the release repair plan and be re-checked after completion. Objects confirmed missing are recorded as explicit absence;
  ADOT absence must first prove the CPU namespace anchor, and permission/network errors must not count as absence.
- Monitoring IAM, Pod Identity and SNS publish permissions are owned by bootstrap and use the existing site-level role; the release's
  runtime install mode may only verify these prerequisites and must not create another role or rewrite base permissions. The SNS shared statement is fixed as
  `AllowAmpAlertmanagerPublish`, comes from the same `amp-sns-publish-policy.json`, binds this account,
  this workspace and the topic, and preserves other statements; when the topic identity, generation or policy read is unclear it
  stops and does not treat a read error as first creation. The internal `--runtime-only` verifies these prerequisites before apply and does not call
  the write interfaces of IAM, Pod Identity association or SNS policy.
- The Aurora refresher is likewise owned by the release transaction: the snapshot saves the old CronJob, ServiceAccount and RBAC,
  and does not save the database password. Rollback first restores the old refresher program, then synchronizes `AWSCURRENT`, and only then allows the CPU restart;
  changing the image alone does not substitute for a complete restore. A failed CronJob query or an unclear identity must stop; the refresh cannot be skipped silently.
- The public `preflight` is read-only. The internal `preflight --for-deploy` allows an authorized deploy to perform pre-repair on the
  refresher actually selected by the plan; it is not a switch administrators can use to relax preflight. Before the Store dependency gate,
  `aurora_prerequisite_repair` must first be persisted, bound to the candidate pin, database identity, original state and complete old snapshot, and then the
  `AWSCURRENT` verification Job that forbids consumer restarts is run. The Store gate cannot be omitted, and DDL, business restarts, the Profile
  and the GPU rollout must not be brought forward; a failure keeps a resumable journal, and the formal previous inherits the pre-repair objects.
  An orphan journal left by a candidate abandoned after `READY` may only be taken over by the next candidate, after the site has been committed, the baseline matches exactly and
  the database identity is consistent, by first restoring the original refresher, or be restored independently by `deploy --rollback`; the old binding is not rewritten.
- The first bootstrap must prove an actually empty database with a CPU read-only database Job, and must not infer it from missing Pods, release state or a
  `CREATED` marker. The proof binds the Aurora/namespace UID, the candidate schema, the run/Job UID and a fresh time;
  the DSN comes only from the CPU Secret, TLS must be `verify-full`, and tables must not be created automatically. A bound origin only allows resume,
  and still checks schema/migration and active records every time; views or unknown tables in an uninitialized database, as well as unknown state,
  schema or identity drift all fail closed.
  The proof Job takes the actual UID of the managed Aurora refresh CronJob as its sole owner with a fixed non-controller reference, and is
  neither scheduled by the CronJob nor managed as an active Job; before creation the journal registers the run/spec identity and
  `owner_uid`, and both preview and live Jobs must match. Cleanup uses a UID precondition and foreground confirmation;
  a lost creation ACK, a failure or an interruption must all clean up their own Job, must not delete replaced or re-owned objects, and
  parent resource garbage collection and TTL cannot replace explicit cleanup proof.
  When all local supervision completion proofs are lost, no further cleanup commands may be issued, nor may the remote Job be claimed as cleaned up.
- installed-resource discovery includes the scheme Job; only a sole owner exactly matching a registered
  CronJob UID in the same namespace counts as a covered child resource. Orphans, same-name parent object recreation, unregistered parent objects, cross-namespace or
  multiple owners are all treated as unregistered resources that block the cleanup preflight and cannot be blindly deleted by name; customer-namespace Jobs are still not taken over.
  Job admission validation does not endorse rewrites for subsequent Pod admission, and the Pod admission
  policy of the CPU namespace remains within the trusted boundary.
- installed-resource cleanup authorization comes only from the trusted-source inventory, including explicit retired entries, not from
  a SHA computed by a mutable ConfigMap itself. An old row of unknown origin must, even when the object no longer exists, stop before any registry
  rewrite and keep the original record; review it first and bring the explicit retired entry into the trusted-source manifest of the signed release;
  objects whose complete `scope + kind + namespace + name` identity differs do not share authorization.
  `gpu.by_context[context]` of a multi-GPU snapshot must keep each cluster's complete synchronization document and its actual `.resources`;
  the top-level union does not authorize cross-cluster cleanup. The context is non-empty and unique and is verified before the CPU registry is written; when the selected context is missing,
  ownership must not be guessed from same-name objects or the union.
  Only the standard `aws eks get-token` is cached manually by the deploy tooling with an explicit validity period; without a validity period it is not reused across requests;
  generic exec plugins and other exec options are handed to native kubectl, preserving the `KUBERNETES_EXEC_INFO` and
  `provideClusterInfo` protocols. A failed refresh does not reuse the old token and does not output credentials.
  API-not-found and temporarily-unavailable are accepted only through the exact stdout/exit-status protocol of controlled scripts; arbitrary error text is not parsed
  to bypass the API; success must additionally verify the complete resource confirmation, and direct SQL must confirm the exact row count. The compatible Aurora backfill is still bound by
  immutable resource identity and whole-batch transaction constraints, uses the currently mounted DSN and executes no DDL.
- The optional HMA forwarding entry point has been decommissioned. When candidate preflight finds an old HMA producer Deployment or cannot confirm
  its state, the upgrade is refused; drain or archive events with the old release first, then decommission the resources in an approved maintenance window.
  AWS HMA, CloudFormation stacks, SQS/DLQ or historical evidence are not deleted automatically, nor are the hot-spare health label,
  quiesce, `NodeRecovery=None` and the permanently disabled provider replace constraints relaxed.
- After the CPU candidate is released it must wait for Agents to refresh their heartbeats against the candidate control plane again; each node wave runs a
  bounded-retry safety gate before claim, requires Agents outside the wave to keep sufficient lease margin, and re-checks with a smaller margin immediately after claim.
  A one-off instantaneous check must not be the basis for starting node mutation.
- Before each rollback wave starts it must prove that all Executors are Ready, Agents outside the wave are ACTIVE with valid heartbeats,
  remote commands are 0 and there is no active destructive workflow; a failure on any node must not open the next wave.
- upgrade, join and rollback must additionally run a dynamic read-only barrier before the first node mutation: re-verify the
  target Node UID/Ready/cordon/isolation taint and failure domain, and verify the Executor, remote commands,
  destructive workflows and Agent leases. The deterministic candidate and host facts for upgrade must already be proven before the CPU rollout;
  when join and rollback have no reusable current candidate proof they still run the same read-only host
  preflight first within the local flow. The read-only host preflight Job may create and clean up its own Pods/Jobs, but the host root must be read-only, and writing node
  files, annotations or systemd is forbidden.
- A Node Runtime candidate must first be installed to
  `/opt/gpu-fault/releases/<artifact-sha>-<dependency-sha>/venv` and verified,
  and only then atomically switch `/opt/gpu-fault/current`. Old slots and the first legacy `/opt/gpu-fault/venv`
  must not be deleted or overwritten in place within the rollback window.
  The dependency identity binds the lock bytes, Python/ABI, platform, libc and the interpreter binary; when missing or mismatched
  the slot must not be reused. Shared dependency layers are kept immutable under that identity; before reuse, the real files, version closure and layer references are verified,
  without first executing the candidate `.pth` or application code; a corrupted published layer must not be rebuilt in place.
  The old rollback slot must also pass verification of the actual files, the old dependency closure and the entry module during candidate preflight and before the service is stopped;
  the existence of an executable file by itself does not prove rollback is possible. A legitimate Python alias only accepts a symlink chain to the same binary,
  and does not relax the interpreter version or system site-packages isolation requirements.
- The terminal event is the monotonic terminal barrier of the attempt Observation. Saving the terminal event must move the corresponding Observation to a terminal state in the same
  Store transaction; a late-arriving `PENDING/RUNNING` must not restore active ownership.
- Only `ACTIVE` members of the cluster registry may dispatch operations; `PENDING/FAILED/ROLLED_BACK` retain only
  join, verification and audit capabilities. Concurrent joins must use a single-cluster CAS transition; fully overwriting the registry revision from an expired candidate site
  is forbidden. Join CLIs of the same site are serialized across the whole transaction; discovery and prerequisite work within a single batch is at most 4-way,
  the rolling parallelism takes the site's `upgradeMaxParallelClusters`, and batch and upgrade share that knob (default 1, ceiling 8).
  The site-wide Installer budget is fixed at 64 nodes, and each cluster's wave is further bounded by `floor(64 / cluster parallelism)`; cluster and node parallelism cannot be
  scaled up separately. The Secret disaster-recovery copy must also use `resourceVersion` CAS.
- registry convergence cannot check only the member set captured at release time. All currently active CPU processes, including late joiners,
  must confirm the same generation and digest with valid persistent heartbeats; future heartbeats and missing required records still block.
  Known expired members may leave the wait only while there is still an active fleet that has fully ACKed; total loss of contact is not success.
  Runtime service eligibility is bound to both a successful refresh and a heartbeat write; a successful database read cannot extend an already expired heartbeat lease.
  This rule depends on the same release and a consistent stale window, and does not claim to have observed that old Pods actually terminated.
- join candidate verify must bind the source/candidate site, live release identity, registry
  generation/content and complete cluster state, and commit within 15 minutes. Site, release state and
  Aurora resource registration complete first; after `ACTIVATION_STARTED` only fail-forward is allowed, and an ACTIVE
  cluster must not be transitioned to FAILED/ROLLED_BACK and then cleaned up.
- join compensation must wait for already started parallel tasks to wind down, and only handles the namespace,
  IAM and network changes this attempt has proven to own; on identity drift or when command termination cannot be proven it does not clean up automatically. Unfinished compensation is resumed first; a new attempt cannot be
  opened at the same time; a cluster token still referenced by a rolled-back registry must not be discarded and replaced with new credentials.
- The Node key probe privately reads the complete Secret JSON and computes the digest over the actual base64-decoded bytes, without stripping first and
  without comparing key names only. The GPU key set must exactly match the current target node; the CPU merges by CAS and preserves the other keys
  and metadata of the whole site; a conflicting unknown key of the same name must be refused, and an existing legitimate randomly rotated value must not be overwritten by a re-derived value.
  A pending rotation marker must enter ensure even when both digests match, and is cleared by CAS only after the Node UID, CPU image
  and GPU identity are verified again; the marker must not be treated as proof of membership. The helper and wrapper are delivered together and bound to the
  TaskInputSpec/release identity, execute only in the trusted deploy-host environment, and do not widen Node Runtime privileges.
  join first excludes cross-cluster NodeName conflicts, and `BoundNodeKeyRunner` carries a fresh NodeName-to-UID mapping;
  compensation is allowed only for targets where key writing may already have started and complete Node/key proof exists, uses the originally bound actual GPU
  key digest to verify the current CPU value, and then deletes with a resourceVersion CAS. On digest drift or without proof, keys must not be deleted by name.
- The first multi-ARN deploy creates the token, IAM, NAT allowlist and Private Hosted Zone
  VPC association only for the first baseline GPU cluster. Associations are deduplicated by VPC; remove and join share the site transaction lock, and removal may be committed only after the Route53 ChangeInfo reaches `INSYNC`
  and the actual association has disappeared.
  A DNS association may only be detached with an exact `CREATED/DETACH` registration record; CPU-native, shared or external associations are preserved.
  List position, member count, name and the observed existence of an object are none of them proof of creation origin; when an old record's identity or origin
  is unclear, reconcile first; the immutable key/ownership must not be rewritten to obtain deletion rights.
  The initial target commitment may be released only when all targets are managed, or when the missing targets have a completed removal journal with complete identity that matches the current site/release.
  Malformed targets, removal rows that match by name only, unfinished or old-version journals, and
  `COMPLETE` with a missing site must all block subsequent deploys.
- deploy/bootstrap, Profile approval, AdminConfig, join/remove and uninstall must compete for the same
  state-dir mutation lock; after taking the lock, reload the site and verify the CPU stable anchor; concurrent overwriting of the site,
  bootstrap state or release state is forbidden.
- An inherited lock FD must actually hold the exclusive flock on that file; a matching inode by itself is not proof of holding the lock; mutual exclusion must
  also hold across threads, and the lock path refuses symlinks and FIFOs. `config` must not initiate Aurora
  modifications or plan writes from an old site target read outside the lock. Rotation, hot-spare writes, manual disposition, workflow convergence, the remote outbox and explicit rollback
  are likewise bound by the site lock; the local file lock provides no distributed mutual exclusion across deploy hosts or replicated state directories.
- The failure-domain mapping refresh after membership must not treat a worker read error as non-existence, nor silently drop
  Node records with missing or conflicting identity. The managed call requires the worker to exist, updates the template with UID/resourceVersion
  guards and waits for the rollout; only the explicit internal bootstrap path may accept a worker not yet created.
- remove first sets the target to `DRAINING`, drains and uninstalls the nodes/Executor, and only after confirming the namespace has actually disappeared
  allows parallel revocation of the registry/key and exclusive AWS resources. VPC/NAT and shared resource dependencies still used by the CPU or other GPU clusters
  must be preserved; node key deletion binds the original Secret UID, the original key digest and fresh site-wide NodeName ownership,
  uses UID/resourceVersion CAS and reads back to confirm.
  Resume binds the complete EKS/HyperPod ARN, cluster/namespace/Node identity and the original site/release digest;
  a target already moved out of the site may only be restored by its exact committed journal, not guessed from same-name objects. Old in-flight journals lacking these bindings
  must be reconciled explicitly, without deleting state or hand-patching hashes to let them through.
- namespace deletion must send the proven UID to the API as `DeleteOptions.preconditions.uid` and refuse same-name recreation while waiting;
  having merely read the UID before deletion is not enough. Node Installer annotation cleanup first verifies the complete target
  UID set, then modifies with a UID/resourceVersion JSON Patch; the at most 8 metadata workers are not a
  node unavailability budget and do not authorize widening the Fleet wave.
- The uninstall `keep/delete/reset` mode, effective policy and source snapshot are bound to the same transaction. Compatibility handling of historical
  `REUSED/PRESERVE` only forms an execution plan and does not rewrite the registry's immutable ownership or delete
  policy. Historical LBC attachment edges are normalized only in the temporary execution graph shared by pre-verification/deletion; the original dependencies remain immutable;
  the role's own attachment is detached and the role deleted first, then the policy; remaining consumers, missing dependencies and cycles all block,
  and the attachment veto on policy deletion cannot be relaxed. Live ownership, the calling account and shared dependencies must still be verified.
  Only after the Kubernetes/non-Aurora/CPU verification passes and
  the pre-Aurora proof is saved may the database be deleted; the final snapshot must bind the original database incarnation and confirm available.
  A rerun of a completed uninstall only re-verifies, and cannot use old cleanup evidence to delete a same-name rebuilt site.
- Site-wide cleanup first sets the targets to DRAINING under the same registry revision, then stops the producers; the ingress and CPU
  consumers keep running until strictly drained, after which the consumers, the ingress and the GPU Executor are stopped in turn, finally entering
  `NODE_RUNTIMES_STOPPED`, and only then are auxiliary resources and objects cleaned up. The node phase uses only the fleet/UID proof saved in the original journal,
  the database must not be read again after the CPU consumers stop, and bulk orphan SQL settlement does not replace drain.
  The schema-v2 journal fixes the new order with phase_order; old versions or old orders must be reconciled explicitly, and changing the version,
  the order array, fabricating phases or guessing the latest
  failed checkpoint is forbidden. Parked spares are restored only by the explicitly owned `previous-unschedulable` baseline,
  using Node UID/resourceVersion with read-back; a missing baseline blocks, and uncordon must not be the default.
- Dynamic workload RBAC is included in the original per-context cleanup snapshot only after the complete renderer policy, namespace/SA UID and foreign-binding checks pass;
  labels or the snapshot's self-hash are not deletion authorization and do not exempt orphans. remove and
  uninstall share the subsequent deletion path, wait for the real drain and the GPU Executor to stop, and before deleting the ownership anchor
  use the original UID/current resourceVersion and confirm disappearance. A `DRAINING` ACK cannot replace these prerequisite proofs.
  The shutdown re-verification refuses Pods that still exist and does not let them through because the replica count is zero. The skip check after the CPU has been deleted is only for read-only re-verification where the original journal
  cleanup is complete and every RBAC deletion has been confirmed item by item; remaining deletions must still verify the CPU and all shutdown prerequisites.
- A local supervision loss in remove/uninstall must be persisted in the original journal as a refusal to resume; after restarting the CLI, mutation must not continue because
  the in-memory supervision marker has disappeared. Confirm the original operation has ended and reconcile first; this marker is not cleared automatically.
- Token rotation records intent before the Secret update, the data-plane rollout, the node wave and the local file replacement; once the local
  `TOKEN_FILE_WRITTEN` intent is on disk, rollback is forbidden, and even when file replacement completion has not yet been recorded only
  fail-forward is allowed. Rollback covers changes that started but whose ACK was lost; the terminal state is written to disk before the temporary token is deleted.
  The journal fixes the values and types of `keep_window/window_minutes/quiet_seconds/acceptance_timeout_seconds`, and
  in-flight or terminal cleanup resume must not modify them; old journals missing fields must be reconciled explicitly, without guessing defaults.
  Quiet observation fixes the ingress Deployment UID/generation/desired replica count before waiting, and requires a complete healthy
  Pod set whose ownership matches; missing replicas, scale-down, recreation or read failures cannot be judged quiet from the logs of the surviving Pods alone.
  It also verifies the stably Ready Pods and the api container identity, fixes the window boundary of each round, and is bound by the overall deadline.
  The current log prefix without a since filter is at most 4096 bytes and must contain a complete record no later than the window start; after the window read,
  it re-reads by the original receipt length and verifies integrity and digest. Normal appends are allowed; unprovable retention, source drift,
  and incomplete/future/late logs all fail closed. This check does not prove that the Kubelet/CRI has not silently under-reported or maliciously kept the same
  prefix while rewriting logs, nor that offline callers have migrated; a real kubectl loopback test is not live Kubelet acceptance.
- Hot-spare claim/release must bind the cluster, Node UID and the original label/cordon baseline; the modification carries resourceVersion and
  reads back to confirm; unknown pool state, concurrent reservation, isolation or node recreation are all refused. Mechanical check confirmation annotations must also
  bind the node identity and resourceVersion, and may not treat a matching name as the same node.
- commit/rollback first persists the committed/rolled-back terminal state and then cleans up the temporary Secret backup. A failed cleanup may only resume
  cleanup idempotently and must not re-mark a release that has passed the point of no return as rollbackable.
- Deleting incident/workflow/plan database rows directly or hand-editing workflow JSON is forbidden. Historical `BLOCKED` records already converged by a recovery
  successor may only be handled through
  `gpu-fault-admin workflow-reconcile` (`--dry-run` shows the plan only; drop it to execute); fencing,
  lease, remote commands, provider actions, the source plan and GPU node isolation state must be re-verified, and on success they transition only to
  `SUPERSEDED` with the complete audit preserved; final deletion still goes through S3 archive-first.
- When any of the Agent artifact/compatibility digest, config digest, protocol or Runtime Profile
  drifts, node actions must be refused by the compatibility gate; before claiming a command the Executor likewise verifies the
  artifact, compatibility digest and protocol.

## 5. Credential Boundaries

| Credential | Allowed locations | Forbidden locations |
|---|---|---|
| execution token | CPU Secret, CPU Pod | GPU EKS, documentation, logs |
| cluster token | CPU registry, the corresponding GPU Secret, target node env | other GPU clusters |
| fleet master | CPU Secret, short-lived file on the trusted deploy host | GPU Secret, nodes |
| per-node action key | CPU/GPU key map, the corresponding node | other nodes |
| CA private key | PKI storage such as Secrets Manager | ConfigMap, nodes |
| CA public key | GPU Secret, node trust bundle | none |
| Aurora password | Secrets Manager, Kubernetes DSN Secret | shell arguments, ordinary artifacts |

Install-time Node key custody is registered explicitly by `gpu-fault-admin node-key-custody configure`,
relying on an externally trusted asymmetric signature authorization, actual write start/completion and independent runtime activation.
Registration does not rotate the key and does not call KMS signing; an unauthorized request only prepares the actual identity and stops. Old checkpoints,
the current Secret shape or a point-in-time scan cannot fabricate historical custody proof. AUTH-015 consumes the complete signature chain and verifies the new key, old key and sibling key against the real
Agent; it does not itself hand out the master, nor restart or reinstall nodes.
That evidence covers only the trusted provisioning path and does not endorse unobserved historical activity; see
[Node Key Custody Evidence](../components/node-key-custody-evidence.md) for details.

Replacement nodes obtain their key through `POST /v1/regional/node-action-keys` (cluster-token
bucket; the payload's `cluster_id` is subject to cluster binding, at most 64 node ids). Threat
boundary: an executor holding one cluster's cluster token can only obtain v2 keys for **nodes the
control plane has already seen in that cluster's own data-plane evidence**, and keys are derived
scoped by cluster_id, so it cannot fetch keys for another cluster or for invented node names; the
fleet master never leaves the CPU, values never enter logs or audit, only the cluster and node names
enter the audit row. On the data-plane side the executor's Kubernetes permission is the namespaced
Role `gpu-fault-cluster-executor-node-keys`: `secrets` `get`/`patch` with `resourceNames`
containing only `gpu-fault-node-action-keys`; no list/watch/create/delete, the sync only appends
missing keys and uses a `resourceVersion` precondition so it cannot overwrite a concurrent write.
Clusters not in the ACTIVE lifecycle state are refused on this path.

### 5.1 Mail Notification Boundaries

- `adminEmail` is the site operations contact; `emailSender` is the verified SES identity;
  `emailRecipients` is an independent, multi-valued recipient list; the implementation must not force the three to be equal.
- `emailSubjectPrefix` is only used to identify the site environment. The SES sending layer uniformly adds the site, Region,
  AWS account and cluster context so that mail from same-name clusters is not misjudged.
- The optional `sesConfigurationSet` takes effect only under `channel: ses`; the name is limited to 1 to 64 ASCII
  letters, digits, hyphens or underscores. Its value is declared by site.yaml, rendered into the
  `GPU_FAULT_SES_CONFIGURATION_SET` of the three CPU role ConfigMaps and enters the rollback environment; when not declared the variable is cleared explicitly and must not inherit
  the value from the deploy-host shell. Once declared, preflight must confirm that the SES configuration set exists, the name matches and sending is allowed;
  it does not create it automatically or widen IAM permissions. Rollback restores the actually captured old ConfigMaps and per-role env; the candidate configuration must not
  overwrite the old notification environment.
- The sender of a native SNS email subscription is the AWS-managed `no-reply@sns.amazonaws.com`;
  `emailSender` only controls the SES fault/action notifications sent directly by this solution.
- The SNS confirmation request binds the Topic generation, endpoint, real Subscription ARN and a 48-hour window;
  `Subscribe` must not be repeated while pending. When the same Topic/mailbox shows multiple confirmed or
  confirmed+pending entries it must fail closed and not guess automatically which one to delete. Ordinary deploy/rollback
  must not delete the site alert Topic; only a formal uninstall handles it according to the installation registry.
- Node, workload, Incident and Workflow identifiers must come from the current event or persisted state; production templates must not
  write example node names, fixed HyperPod instance IDs or test fixture identifiers.
- kubectl approval commands in mail must explicitly carry the context; when the context is unknown, use the
  `GPU_FAULT_KUBE_CONTEXT` required-value protection; falling back to the current kubectl context is forbidden.
- `vendor-ticket-*` is this solution's internal support escalation record and does not mean an AWS Support Case has been created.
  Templates may only list the Workflow operations that actually FAILED and must not sweepingly claim that reset, reboot and replace
  have all been executed and failed.
- When a fixed subject, disposition instruction or field changes, `template_version` must be raised so that dedup and audit can
  distinguish old and new templates.

## 6. Launch Veto Items

If any of the following fails, active must not be entered:

- the production attestation is neither bound to a signature-verified main CI gate nor comes from the trusted local full gates
  equivalent to `make check` plus the PostgreSQL stress;
- `make release-preflight` or the CAP-005 PostgreSQL 16 stress suite
  (8x40, JUnit zero skip) did not pass;
- the Runtime Profile has warnings;
- the required/compatible pin does not match the release phase;
- SES or AMP/SNS has no usable receiving endpoint;
- Executor claim, Agent heartbeat or Collector freshness is not satisfied;
- NLB/TLS, Aurora credential refresh or certificate validity has not been verified;
- a destructive step lacks the adapter owner, signature, fencing or post-action verification;
- the Agent endpoint CIDR is empty, HTTP is used, or the signed heartbeat does not carry a TLS certificate;
- the capacity test shows sustained 429/503, queue growth or database connection exhaustion.
- the 120 to 300 second stability window after release shows new Pod restarts, sustained queue growth, non-terminal remote commands or critical
  firing alerts. Before the formal stability window, only a bounded convergence wait of at most 420 seconds for `GpuFaultStoreIoRejected` is allowed,
  and it must first prove a complete fresh zero view of all running CPU Pods: aggregator degraded=0,
  the bounded slot count matches process-count, and every slot fully publishes finite non-negative zero values for the three reasons.
  Missing, old merged-format, duplicate, malformed or non-zero data cannot authorize the wait, and process_id cannot replace the slot; other
  critical alerts, restarts, non-terminal commands or timeouts still veto immediately, and the wait cannot be written as an alert exemption.
  The Executor internal error alert must be based on the latest error timestamp and must not use counter `increase()` on the retained record count,
  which declines as retention cleanup runs.

For detailed gates see the [Deployment and Operations Manual](deployment-and-operations-manual.md) §2, §6, §7 and §10.
