English edition of `docs/管理员日常运维.md`; the Chinese file remains the source of record until both are maintained together.

# GPU Fault Handling System Administrator Operations

This document starts after the site's first go-live acceptance is complete, and is organised by "task or symptom"; it does not require administrators to understand the internal module
call chains. First site build or application-layer rebuild is in [Administrator Quick Deploy](administrator-quick-deploy.md); low-level commands, security
rationale and special recovery steps are kept in the [Deployment and Operations Manual](deployment-and-operations-manual.md).

## 1. Common Task Entry Points

| Task | Normal entry point | Detailed reference |
|---|---|---|
| First deployment or application-layer rebuild | [Administrator Quick Deploy](administrator-quick-deploy.md) | One `deploy`, the training smoke and go-live evidence |
| View regional health | `gpu-fault-admin status --state-dir ...`; add `--full` for the complete read-only acceptance (TLS, Agent/Collector, NLB, Aurora, alerts) | REG-8, REG-14 |
| Upgrade all managed clusters | `gpu-fault-admin deploy --state-dir ...` (one parameter; ARNs, email and Grafana are all restored from the state-dir) | REG-8.1.3/8.1.4, §4 |
| Resume after an interruption | Rerun the same `deploy --state-dir ...` as is; there is no separate `resume` | REG-8.1.4, §4 |
| Roll back to the previous committed release | `gpu-fault-admin deploy --state-dir ... --rollback` | REG-8.1.4, REG-8.2, §4 |
| Adjust administrator configuration | `gpu-fault-admin config --state-dir ... [--file F] [--dry-run] --reference ...` | [Administrator Capacity Configuration](administrator-capacity-configuration.md) |
| Declare/release a warm-spare GPU node (the failover target of `REPLACE_NODE`) | `gpu-fault-admin config spare --state-dir ... --node <node>` for the read-only check; add `--declare --reference ... --confirm DECLARE_WARM_SPARE_CORDON` to declare (label + cordon), `--release --reference ... --confirm RELEASE_WARM_SPARE_UNCORDON` to restore per the baseline; recorded in `<state-dir>/warm-spares/<node>.json` | Manual, "Adding an independently repaired and re-accepted node to the spare pool" |
| Approve a pending Profile plan | The same `deploy --state-dir ... --approve-profile-plan <plan_sha256> --reference ...`; there is no separate approval verb | [Runtime Profile Change Approval](administrator-profile-change-approval.md) |
| Release across schema versions | `deploy --state-dir ... --accept-schema-change` (takes an Aurora snapshot first; `--accept-schema-change-without-snapshot` skips the snapshot) | §4.7 |
| Join a GPU cluster | `deploy --state-dir ... --gpu-cluster-arn <all managed ARNs> --gpu-cluster-arn <new ARN>` (the given set must be the managed set or a superset of it; the extra ARNs are the joins; giving fewer is refused with `site manages GPU clusters the command omits ... use remove-cluster`; when there are code changes release first, then join), or the join-only `gpu-fault-admin join-cluster --state-dir ... --gpu-cluster-arn <new ARN>` (does not trigger a site release; already managed ARNs return `ALREADY_MANAGED`) | REG-8.1.5, §5 |
| Remove a GPU cluster | `gpu-fault-admin remove-cluster --state-dir ... --gpu-cluster-arn <ARN> --confirm REMOVE_GPU_CLUSTER` | REG-8.1.6, REG-12, §5 |
| Reinstall while keeping the CPU/GPU EKS | `gpu-fault-admin uninstall --state-dir ... --cpu-cluster keep --confirm UNINSTALL_GPU_FAULT` followed by a fresh `deploy`; keep preserves Aurora by default (historical incidents/workflows are not lost), and only an explicit `--reset-database` wipes the database | §3 |
| Permanently decommission and delete the CPU cluster | `gpu-fault-admin uninstall --state-dir ... --cpu-cluster delete --confirm DELETE_CPU_CONTROL_PLANE` (`--aurora-final-snapshot retain|skip` decides the final snapshot) | REG-13, §3 |
| Troubleshoot PENDING/LEASED commands, Pods not Ready, certificate/Agent version drift | `gpu-fault-admin status --full` (the former `doctor` idea has been folded into status), then follow the symptom table | §7, REG-14.2/14.3 |
| Rotate the cluster token | `gpu-fault-admin rotate-token --state-dir ... --gpu-cluster-arn <ARN>` (one command completes the registry overlap window, GPU Secret, data-plane rollout and node Agent switch) | §6.1, REG-9 |
| Explicitly register Node key custody | `gpu-fault-admin node-key-custody configure --state-dir ... --file ... --trust-sha256 ...`; after registration the original `deploy` / `join-cluster` still perform the authorised preparation and resume; Secrets are not rotated directly | [§6.2](#62-node-key-custody) |
| Submit an explicit remediation after CHECK_MECHANICALS; trigger a validated restore after a QUARANTINED node is repaired; manually confirm a node action with unknown outcome | `gpu-fault-admin submit-remediation --state-dir ... --incident-id ... --disposition {inspected,reset-gpu,reboot-node,quarantine,restore,confirm-node-action} [--node <node>] [--plan]` (`restore` see §9.2, `confirm-node-action --node` see §9.4) | §9, §9.2, §9.4, REG 8.3 |
| Remotely view/requeue a node collector's dead-letter outbox | `gpu-fault-admin collector-outbox --state-dir ... --cluster-id C --node N --collector kernel --action stats\|list\|requeue-dead [--path /v1/...] [--yes] --reference CHG-...` (no login to the GPU node; only metadata is returned; `requeue-dead` must carry `--yes`; there is no remote `--force`) | §9.3, §8 `GpuFaultCollectorSilent` |
| Data migration | `gpu-fault-store-migrate` plus the stop-write state machine | §10 |
| Archive and delete expired incidents/workflows | **Enabled** by default (30 days, bucket created automatically by deploy); disabled only when `site.yaml` explicitly declares `spec.retention.controlRecordRetentionDays: 0` | §10 |
| Converge historical workflows that recovered but stay BLOCKED | `gpu-fault-admin workflow-reconcile --state-dir ... --dry-run` to see the plan, then drop `--dry-run` and add `--reference` to execute once | §10.2 |
| List and close ESCALATED incidents (node repaired but the record still only logs) | `GET /v1/incidents?state=ESCALATED` or `gpu-fault-admin workflow-reconcile --state-dir ... --close-escalated --dry-run` to list the queue; `--close-incident <id>` / `--close-escalated` with `--reason` `--reference` to close | §9.1 |
| Node failure-domain (rack/instance group) mapping for remediation budget throttling | Automatic: `deploy`/`join-cluster`/`remove-cluster` render and apply it; read-only view with `gpu-fault-admin failure-domain-map --state-dir ... [--output ...]` | §5.1 |
| Handle one AMP alert | Enter the triage card by alert name | §8 |

Ordinary deployment and operations entry points restore the internal site through `state-dir` and require no site, artifact or release JSON path.
Node key custody registration additionally requires an explicit evidence selection file and an externally approved trust pin, see §6.2;
it does not change the site restoration entry point of normal deploy/join.
Resume is rerunning `deploy --state-dir` as is, rollback is the same command plus `--rollback`; no path may
depend on the current default kubectl context.

### 1.1 The Site-Bound CLI

The first `deploy` installs under the state-dir a deployment-host CLI bound to that site: `<state-dir>/deployer-venv`
(the binding is recorded in `deployer-venv/gpu-fault-managed-state-dir.json`, 0600); when the deploy-host code changes,
`deploy` automatically switches the version and the symlink, with no manual maintenance. Later join, remove, uninstall, configuration apply,
rotation and rollback use this CLI; you can add its `bin` directory to PATH. All `gpu-fault-admin ...` examples in this document
refer to this CLI, and the two forms are equivalent:

```bash
# Run directly after activation
. /secure/gpu-fault/deployer-venv/bin/activate
gpu-fault-admin status --state-dir /secure/gpu-fault --full

# Without activation, using the full path
/secure/gpu-fault/deployer-venv/bin/gpu-fault-admin join-cluster \
  --state-dir /secure/gpu-fault --gpu-cluster-arn <ARN>
```

It runs the code of the site's current release, resolves scripts and artefacts from the source snapshot recorded in `site.yaml`, and accepts only
its own bound directory: a `--state-dir` pointing elsewhere is refused (`installed deploy-host is bound to --state-dir <X>;
refusing ...`), and omitting `--state-dir` is refused likewise, so it cannot accidentally operate on another site. The binding check precedes log creation,
so it cannot leave logs or execute changes in another site's directory. The `.venv/bin/gpu-fault-admin` built by `make deploy-host-setup-online` in a development
checkout is not bound to a site and runs the working tree's current source; it is only for source staging iteration; do not use it for daily
operations. Do not manually install a venv into the state-dir with `setup-deploy-host.sh` as the bound CLI -- the binding file is written only by
`deploy`; to refresh it, rerun `deploy --state-dir`.

Misusing the unbound CLI from a checkout is blocked: when the target state-dir already has a bound CLI, the unbound development checkout allows only
ordinary source deploy (which prepares and re-enters the bound CLI itself) and genuinely read-only query paths; the other verbs are refused with
exit code 2 before the command log is opened, printing the bound CLI path to use instead, and the refused call leaves no log under any state-dir.
`deploy --rollback`, explicit site files and the internal prepared-source form do not enjoy the ordinary source deploy exemption.
Read-only includes status/preflight/verify, `config --dry-run`, the warm-spare check without declare/release,
`workflow-reconcile --dry-run` and `submit-remediation --plan`.
Remote outbox stats/list create control records and are not part of the read-only exemption. A permitted checkout is usually ahead of the site's deployed
release, and the image locked by its `dist/current-release.json` is no longer the one the site deployed: modes that apply or validate the image
(deploy, upgrade, plan, verify, remove-cluster, etc.) refuse accordingly and print both image references; drain-cluster, which only publishes the
registry and consumes no image, is unaffected, so uninstall's `CLUSTERS_DRAINING` is not blocked by the image lock
(`src/gpu_fault_release/regional_release_config.py`).

The binding file must be a private, readable, valid record not replaced by a symlink, and the two site selectors must point at the same
canonical site. An existing installation lacking the binding or with a damaged CLI is refused without falling back to the checkout; this version provides no
environment variable bypass. Repository overrides must agree with the site record; only a checked ordinary source deployment and its bound continuation
may choose a new candidate snapshot, and the existing release, target and approval gates continue to apply.

## 2. Fixed Runbook Format

Every new administrator runbook must contain the following seven items:

1. **Applicability**: in what state execution is allowed.
2. **Scope of impact**: which CPU, GPU clusters, training jobs and external resources are affected.
3. **Read-only checks**: the gates and expected output that must be satisfied before execution.
4. **Command**: the single formal entry point; when dry-run is the default this must be stated.
5. **Success criteria**: not judged by process exit code or Pod Ready alone.
6. **Rollback**: recovery order, state files and irreversible boundaries.
7. **Evidence**: change ticket, release/profile, command output and final state.

"Widening permissions, disabling TLS, deleting fencing, modifying terminal records" must not be written as troubleshooting steps.

### 2.1 Capacity Configuration

Ordinary capacity changes neither rebuild code nor publish a dynamic Aurora revision. Edit
`<state-dir>/admin-config.yaml` and run one command:

```bash
# First see what would change: reads only the local file, no signature verification, no cluster connection, no state written
gpu-fault-admin config --state-dir /secure/gpu-fault --dry-run

# Apply; --reference is optional and defaults to "executor identity:start time"
gpu-fault-admin config --state-dir /secure/gpu-fault --reference CHG-12345
```

The command automatically records the executor identity (STS caller ARN), the configuration before and after and the result in
`<state-dir>/admin-config/history/<start time>-<configuration digest>/`. This path reuses the current signed image and
wheel, first issues the Aurora capacity change, then rolls only the affected ingress, control-worker or spool-worker,
and finally waits for the Aurora scaling to land. After an interruption, rerunning the same command resumes. Complete fields, templates, first deployment and rollback
semantics are in [Administrator Capacity Configuration](administrator-capacity-configuration.md).

### 2.2 Same-Site Concurrency and Recovery

deploy, configuration/Profile approval, Node key custody registration, join/remove, uninstall, token rotation,
warm-spare declaration/release, manual
remediation writes, workflow convergence and explicit rollback share one mutation lock on the canonical `state-dir`.
When the CLI finds it held it reports the holding process immediately and does not queue silently. After taking the lock it re-reads the internal site and checks the CPU stable identity;
it must not continue with the stale membership or Aurora target from command start. Remote outbox `stats/list`, although not modifying the node
outbox, also creates a maintenance workflow and therefore holds the lock as well.

The lock only constrains managed entry points on the same deployment host and the same state directory; it is not a distributed lock; do not copy the state to another
machine and operate the same site concurrently. An inherited FD or a hand-created file of the same name is no proof of holding the lock, and a lock file in use must not be deleted to unlock.
Parallelism is scheduled only inside the command by resource dependency and budget; several CLIs must not be started to stack concurrency.

Rerunning a checkpoint is a recovery order, not a skip of the current identity and safety checks. When an old in-flight journal lacks the required binding, the
target has been rebuilt under the same name, or the source/release identity has drifted, keep all evidence and reconcile explicitly; do not edit phases, back-fill
digests or delete directories to force a resume. `SUPERVISION_LOST` means it is still unproven whether the original command stopped, and compensation must not begin merely because
the CLI was restarted. All real-machine writes still require prior confirmation of Region, context, cluster, node scope,
maintenance window, stop conditions and rollback plan.

A warm-spare declaration first saves the owning cluster, Node UID and original label/cordon baseline, then modifies with resourceVersion and
reads back; release restores only that baseline. On a resourceVersion conflict during modification (the node concurrently updated between the check and the modification,
e.g. kubelet heartbeats, device plugin re-registration) it re-reads the node and retries with the new version, at most 3 attempts,
and only when the spare label, cordon, spare-reservation annotation and isolation ownership are all unchanged, otherwise it refuses rather than overwriting.
An interrupted declaration whose patch never took effect (the record has no `declared_state` and the baseline still equals the node's present state) is archived by the next
`--declare` into the record's `superseded_interrupted_declarations` and re-declared directly,
without a prior `--release`.
A node originally cordoned stays cordoned; unknown pool state, reservation/isolation,
cross-cluster records, missing baseline or Node UID change all refuse release. The read-only `config spare` check is not a declaration authorisation,
nor does it change `NodeRecovery=None` or the provider replace prohibition.

## 3. Complete Uninstall or Permanent Decommissioning

**Applicability**

- New training submissions have stopped, and active workflows, remote commands and node isolation states satisfy the cleanup gates;
- The internal site in the `state-dir` contains all currently managed GPU clusters;
- The executing identity can still reach the CPU/GPU Kubernetes APIs and the AWS resources in the registry.

**Scope of impact**

- Both modes uninstall this solution's CPU/GPU-side components, node systemd, NLB, solution IAM/monitoring/PKI and
  solution-specific network objects;
- All GPU EKS/HyperPod clusters and customer training workloads are always kept;
- Warm spares registered with `config spare --declare` are released before the Kubernetes cleanup according to their declaration records
  (`<state-dir>/warm-spares/`): the spare label is removed, the pre-declaration scheduling state is restored, and it is recorded in the uninstall journal's
  `released_warm_spares`; a spare reserved by an incident, ALLOCATED or under isolation makes the uninstall fail closed.
- `--cpu-cluster keep` is a **reinstall**: it keeps the CPU EKS/HyperPod and also keeps the solution-specific Aurora
  (the site's incident/workflow/registry records; the isolation taints on GPU nodes are keyed by the incident ids
  inside it). The next `deploy` takes over this existing Aurora (CPU-2), and records are not lost. Only an explicit
  `--reset-database` deletes Aurora too, letting the next `deploy` start from an empty database;
- `--cpu-cluster delete` is **decommissioning**: it deletes the CPU EKS/HyperPod (not its external VPC) and
  Aurora, keeping one final snapshot as audit evidence by default; `--aurora-final-snapshot skip` is valid with
  `delete` and can also be combined with `keep --reset-database` (no audit snapshot needed when wiping the database for a reinstall); combined with `keep` alone it is refused.
- When an uninstall gets stuck midway on a defect of the cleanup tools shipped with the deployed snapshot, `deploy` refuses to run (uninstall in progress);
  in that case resume with `gpu-fault-admin uninstall --state-dir ... --repo-root <reviewed source tree> \
  --accept-repository-root-override ...`: the path of that tree and the digests of its `deploy/` and `scripts/` are written into
  `repository_root_overrides` of `uninstall/state.json`; it is accepted only while an uninstall is already in progress,
  and a brand-new uninstall cannot use it to bypass the deployed snapshot. The cleanup tools accept only this one digest of the release configuration re-materialised at that point,
  and record old and new digests in `config_sha256_history` of `kubernetes-cleanup.json`.

A reinstall is authorised by the complete uninstall record and the original Aurora instance identity, and binds a new namespace and installation generation;
it is not an override of the "database non-empty" check. Old deletion plans, cleanup logs and transaction directories are archived as a whole and are not used for a new round of
uninstall. The new generation of the resource registry is separated from the old AWS ID history, while the actual AWS site tags stay unchanged; do not hand-edit
the installation ID or registry scope to resolve conflicts.

The uninstall relies on three tiers of resource sources of truth:

The site-wide cleanup first sets the target GPU clusters to `DRAINING` in the same registry revision, then stops the producers.
During the drain the ingress and consumers keep running to finish existing requests; workflows, remote commands,
processor and spool must pass the strict zero-activity check, and only then are the consumers, ingress, Executors and nodes stopped.
An expired lease or "nobody can claim" does not prove the physical action has ended; blocking must not be cleared by judging failure through bulk SQL.
Resume also validates the original journal's `phase_order` and identity, and does not fabricate completed phases for an old order.

| Inventory | Location | Purpose |
|---|---|---|
| Kubernetes installation inventory | `gpu-fault-system/gpu-fault-installed-resources` in each CPU/GPU EKS | Deletes the actually installed workloads, RBAC, Secrets, ConfigMaps and namespace |
| Node installation inventory | `/opt/gpu-fault/installed-units.txt`, with digest and full list synchronised to the Aurora AgentRecord | Uninstalls the actually installed systemd units |
| AWS solution resource registry | `installation_resource` in Aurora `gpu_fault_objects` | Records ARN/ID, `CREATED/EXTERNAL`, `DELETE/DETACH/PRESERVE` and dependencies |

**Command**

```bash
# Reinstall: keep the CPU cluster and the Aurora records, then simply deploy again
gpu-fault-admin uninstall \
  --state-dir /secure/gpu-fault \
  --cpu-cluster keep \
  --confirm UNINSTALL_GPU_FAULT

# Reinstall but start from an empty database: explicitly delete Aurora (one final snapshot is still kept by default)
gpu-fault-admin uninstall \
  --state-dir /secure/gpu-fault \
  --cpu-cluster keep \
  --reset-database \
  --confirm UNINSTALL_GPU_FAULT

# Permanently decommission the CPU control plane
gpu-fault-admin uninstall \
  --state-dir /secure/gpu-fault \
  --cpu-cluster delete \
  --aurora-final-snapshot retain \
  --confirm DELETE_CPU_CONTROL_PLANE
```

**Success criteria**

- `gpu-fault-installed-resources` and the solution namespace of the CPU and every GPU EKS do not exist;
- All registry `DELETE`/`DETACH` entries are confirmed absent by real AWS queries;
- `PRESERVE` entries still exist, especially all GPU EKS/HyperPod clusters, and in `keep` mode without
  `--reset-database` the whole Aurora set (cluster, writer/reader, managed
  master password, subnet group/SG/parameter group);
- `uninstall/state.json` is `COMPLETED`;
- `delete_policy_residuals` in `uninstall/state.json` and the command output is `0`;
  in `uninstall/installation-resources-final.json` (with `.sha256`, file SHA verified) every resource's
  status is `DELETED`/`DETACHED`/`PRESERVED`;
- `aurora_cluster` in the output matches the chosen mode: `preserved` for `keep`,
  `deleted` for `keep --reset-database` and `delete`.

The AWS registry in Aurora must be exported first when the uninstall starts. Only after the verification of Kubernetes, non-Aurora resources and CPU
preservation or deletion is all complete, `installation-resources-pre-aurora-delete.json` is saved
and `READY_TO_DELETE_AURORA` entered, may the `--reset-database` and `delete` modes delete
the database. Aurora deletion must not be moved ahead of, or in parallel with, these proofs to save time. `keep` without
reset only checks the whole Aurora set still exists and marks it `PRESERVED`.
The delete mode first binds the original database incarnation and the final snapshot policy, then proceeds by the current actual reader, writer,
cluster and dependencies; it enters the next phase only after the previous instance wave is confirmed gone. The final snapshot must belong to the original
database and reach available; the subnet group/SG/cluster parameter group `<cluster-name>-pg`
(`aws/aurora/parameter-group`) are cleaned last by dependency. If any step fails, the uninstall must not report completion;
after the database is deleted it uses the saved checked cleanup proof and independent resource read-backs, not depending on the destroyed database.

The fixed phases and reasons for waiting are:

| Phase | Proof that must be completed | Performance and parallelism boundary |
|---|---|---|
| Registry export | site/account/Region, complete CPU/GPU/Aurora identity, dependencies and the effective policy of this run agree; records to delete are `DELETE_PENDING` | Pins the original inventory first; resume reuses the bound snapshot and does not rebuild authorisation ad hoc |
| Stop producers and drain | Active workflows/commands and queues at zero; an unknown state cannot be treated as an empty queue | Stops producers, closes ingress and drains; nodes and Executors may still finish in-flight work during this time; records are not emptied via bulk orphan SQL |
| Stop consumers and GPU Executors | First `CONTROL_CONSUMERS_STOPPED`, then `GPU_EXECUTORS_STOPPED` | Prevents further dispatch; both must precede node component uninstall |
| Node and Kubernetes uninstall | `NODE_RUNTIMES_STOPPED` uses the original journal's fleet/UID inventory; then auxiliary, registered objects and both sides' solution namespaces are cleaned up (within each dependency wave all objects are issued for deletion first, then their disappearance is awaited once), and both sides' installed-resource ConfigMaps, registered objects and namespaces are confirmed absent | No database reads after the consumers stop; bulk re-verification by context/kind/namespace; a successful delete request does not equal object disappearance |
| Non-Aurora resource deletion | NLB/listener/target group, DNS, PKI (ACM certificate), LBC Helm release, monitoring, SES identity, ECR repository, IAM, EKS addons and solution-specific network resources are cleaned by dependency, and external users must not be affected | Resources of the same priority without dependencies at most 4-way per batch; the CPU Helm/LBC must be verified while the CPU API is still available |
| CPU keep or delete | `keep` confirms the CPU is still present; `delete` deletes the CPU HyperPod first, then EKS; all GPU clusters are still present | nodegroups may be requested for deletion in the same wave; Fargate profiles are deleted one by one with waits, not blindly in parallel |
| Aurora final phase | Saves `installation-resources-pre-aurora-delete.json` and enters `READY_TO_DELETE_AURORA` | `keep` without reset only checks the whole Aurora set; the delete mode first binds the database incarnation and snapshot policy, then reader, writer, cluster and dependencies |
| Final re-verification | The retained snapshot belongs to the original database and is available, `installation-resources-final.json` with SHA-256 is written, the final resource states are complete and residuals are 0 | `COMPLETED` is written only after all preceding proofs pass; local duration is not a cloud performance commitment |

**Resume after failure**

Rerun the command with exactly the same parameters. It reuses `uninstall/state.json`, the Aurora export/deletion plan and
`kubernetes-cleanup.json` only when that state file still exists and the site, target, `--cpu-cluster`
mode, snapshot policy and `--reset-database` match the last run (a parameter or effective policy change refuses the resume outright); a
`kubernetes-cleanup.json` not yet at `CLEANUP_COMPLETED` is still the original journal of this transaction, and the resume continues on it without renaming, rescanning
or looking for another checkpoint.
Completed phases do not re-execute mutations but still re-verify the necessary real resource states; all delete APIs are executed idempotently.
Once CPU deletion has been proven, Helm is no longer verified via its vanished API; while Aurora is still deleting or the final snapshot still being created
it keeps a bounded wait, and a rerun seeing the cluster and instances in `deleting` only waits, without re-issuing deletion to original objects already confirmed deleting.
An old in-flight state lacking the currently required identity/cleanup proof needs explicit reconciliation and cannot be assumed safe.
Rerunning `COMPLETED` only re-verifies; it is not an authorisation to uninstall again a site rebuilt under the same name; once `uninstall/` has been archived as a whole by a later `deploy`
(see "The first `deploy` after an uninstall" below), a new uninstall starts from an empty directory. If `uninstall/`
contains only the previous round's working files (`installation-resources-*.json(.sha256)`, `kubernetes-cleanup.json`)
and no `state.json`, a brand-new uninstall refuses with `uninstall transaction files lack their original journal`
and does not rename or archive automatically and continue; the administrator handles those leftover files first. Do not delete the `uninstall/` state directory
or hand-edit the checkpoint and resume by hand.

The cleanup journal uses schema-v2 and explicitly records `NODE_RUNTIMES_STOPPED`. Only this run's original journal
and the complete fleet/UID binding may be used to resume; another failed checkpoint must not be picked by file time. The old order of schema-v1
needs explicit reconciliation; the version number must not be changed, phases back-filled or files deleted to "resume from scratch". Stop when unfinished or unknown commands
are found; do not change them to FAILED via bulk SQL.
Parked spares restore their original scheduling state only per this run's owned `previous-unschedulable` baseline;
Node UID and resourceVersion must both protect the write and read-back. An original value of true stays cordoned, and an unknown baseline
blocks the cleanup outright rather than defaulting to uncordon. This change adds no Kubernetes resources.

When the old online version explicitly lacks `/v1/installation-resources`, the CLI can rebuild the inventory from the same-site bootstrap state,
the live NLB and AWS tags, and backfill through the currently mounted DSN of a running CPU Pod. Only the controlled script's
exact old-API/temporarily-unavailable protocol is accepted; ordinary 404/503 error text, authentication or TLS failures are not treated as compatibility signals.
API synchronisation needs a complete read-back confirmation; direct backfill needs exact row counts, an immutable identity conflict rolls back the whole batch, and no DDL is executed.

Solution-specific IAM, SGs, PKI, monitoring or LBC marked `REUSED/PRESERVE` in a legacy registry are no longer preserved;
the uninstaller treats them as same-site leftover resources. Whether Aurora stays is decided only by the mode (`keep` preserves,
`--reset-database`/`delete` deletes), independent of the old policy in the registry. Only cluster base capabilities such as EKS/HyperPod,
external public subnets, the VPC IGW, the OIDC Provider and the pre-installed Pod Identity Agent may continue to be
`PRESERVE`.

The "effective policy" here is saved in this uninstall's plan/audit and does not rewrite the immutable ownership,
delete policy or dependencies in the Aurora registry; live ownership must still be proven before deletion. Cases such as IAM still used by instance profiles/other principals,
LBC still serving other Services/Ingresses, or SQS statements mixing permissions of other topics must be reconciled first.
Historical LBC role/policy attachments are adjusted only in the temporary execution graph to first detach the role's own attachment and delete the
role, then delete the policy; when the role is kept, its policy must not be deleted, nor other principals' attachments forcibly detached.
A SHA computed by the registry or local files on their own only shows integrity and grants no deletion right.

Historical rows of unknown origin in the Kubernetes installation inventory must stop the synchronisation and keep the original
registry even when the corresponding object no longer exists; a developer reviews, adds an explicit retired entry in the trusted source inventory and ships it with a signed release;
rows must not be hand-deleted to pass the cleanup gate. The cleanup identity includes scope, kind, namespace and name; a same-named object in a historical namespace
cannot be authorised by the current namespace.
Multi-GPU cleanup also uses the inventory actually installed on each cluster in `gpu.by_context[context].resources`;
the top-level aggregate does not mean every cluster owns all of those objects. A missing selected context or duplicate contexts refuse;
ownership evidence cannot be completed from same-named resources of other clusters or the site-wide union.

`uninstall` reads the site only from `site.yaml` under `--state-dir`; a directory without an internal site is not
a managed site, and the command refuses outright.

**The first `deploy` after an uninstall.** While `uninstall/state.json` exists but has not reached `COMPLETED`,
`deploy --state-dir` refuses outright (`an uninstall of this site is in progress (phase <phase>)`);
first finish the uninstall with the same `uninstall` command; do not use `deploy` to refresh the deployment host while the uninstall is unfinished. The first `deploy` after the uninstall
reaches `COMPLETED` first turns this directory back into a new site: with a retirement journal carrying persisted intent and checkpoints it moves
`site.yaml`, `bootstrap-state.json`, `source-deploy.json`, `source-deploy-success.json`,
`source-deploy-success.sigstore.json`, `installation-resources.json(.sha256)`,
`quick-validation.json`, `public-release-verdict.json`, the old `release-deploy/`, `join-cluster/`,
`remove-cluster/` transaction directories and the whole `uninstall/` into
`<state-dir>/retired-<UTC time>-<random id>/` (kept for audit, not deleted, and not just renaming the main state file), writes a new
installation ID (`keep` mode also records `retained_uninstall` for taking over the retained Aurora), prints
`completed installation retired into retired-...; bootstrap will revalidate retained resources` on stderr,
and then recreates all resources along the first-deployment path. `admin-config/`, the release signing keys, `deployer-venv`,
kubeconfig and logs are untouched; the reinstall does not need these inputs prepared again. Do not delete these records by hand to "clean up" the directory.

## 4. Regional Upgrade

**Applicability**

- The configuration contains all managed GPU clusters;
- Remote command `PENDING/LEASED/WAITING` is 0;
- No `PENDING`/`WAITING` driver/firmware/EFA installation step (`REMEDIATE_DRIVER` /
  `UPDATE_SOFTWARE_FIRMWARE` / `REMEDIATE_EFA_DRIVER`): if there is one, the engine refuses to open the upgrade/rollback
  transaction (automatic rollback refuses likewise) and names the workflow, because the new control plane would derive another command id for that step
  and submit the installation again on the node still installing; when you really must release with an in-flight installation, add
  `--allow-inflight-installs` on the same command (`--rollback` also accepts only this one extra parameter), and the engine instead records what it skipped;
- The `gpu-fault-aurora-credential-refresh` CronJob can run through: both deploy and rollback first
  run it synchronously once (`create job --from=cronjob`, wait for Complete, default 300 seconds); if the Job does not
  Complete it fails closed and rolls no component;
- The Runtime Profile content is unchanged, or a new plan has been generated per this section and approved independently.
- If the change touches `ddl*.py`, Processor claim/retry/completion or the PostgreSQL pool,
  the CAP-005 runner has passed with zero skips on an isolated PostgreSQL 16 database.

**Read-only checks**

**Command**

```bash
gpu-fault-admin deploy --state-dir /secure/gpu-fault
```

An existing site needs only this one parameter: the CPU/GPU ARNs and administrator email are read from `site.yaml` (ARNs passed again must match
the managed set or be a superset of it, see §5). `deploy` automatically chooses bootstrap or upgrade from the live release state.
Within the first minute the command checks that the SNS subscription is confirmed (sites with `spec.notifications.channel: ses` also check the SES sending identity); when the online transaction stopped at
`failed/partial-convergence` with a different release as candidate, or the candidate crosses a schema version, it refuses with the engine's own words as soon as the release build finishes
and names the parameter to add (`--supersede-failed-transaction` / `--accept-schema-change`);
command-line parameters take precedence over leftover environment variables.
The current PostgreSQL schema baseline is v14 (`objects-wakeup-notify-trigger`; what v12-v14 each changed is in the special note of §4.7); a schema change must be classified `FULL` and can only be released fail-forward: add `--accept-schema-change` on the deploy command (§4.7); do not change `site.yaml`. The schema phase of `deploy`
(`deploy/control-plane/tools/ensure-postgres-schema.sh`; the single-cluster `deploy.sh` of Appendix A uses the same order) runs three one-shot Jobs in order before rolling
the business processes, stopping on any failure without rolling:

1. `gpu-fault-postgres-index-build` (`gpu-fault-store-migrate --build-indexes-concurrently`):
   runs `CREATE INDEX CONCURRENTLY` one by one for indexes declared by the code but missing or invalid in the database, see §4.2.
2. `gpu-fault-postgres-schema-ensure` (`--ensure-schema`): idempotent DDL and migration registration, advancing
   `gpu_fault_schema_migrations` to the version the wheel requires. The indexes were built in the previous step, so here they are just skipped.
   Since v12 this step no longer takes any table lock when there is "no change": every `CREATE INDEX` / `ADD COLUMN` /
   trigger / `ENABLE TRIGGER` first queries the system catalogs before deciding whether to execute (the bare `IF NOT EXISTS` form still takes a ShareLock on the table
   until the transaction commits when the index already exists, and `ALTER TABLE` is ACCESS EXCLUSIVE), and the `partition_id`
   sweep runs only when the database version is < 5. A busy-hour deployment no longer blocks all writes or hits `lock_timeout` because of a routine ensure.
3. `gpu-fault-postgres-schema-preflight` (`--schema-preflight`, read-only): see the gate checklist of §4.3;
   the JSON report is printed in the deployment log.

Pods must not run DDL themselves at startup: `GPU_FAULT_POSTGRES_AUTO_SCHEMA_INIT` must stay `false`.

`deploy` writes the quick-validation evidence to `<state-dir>/quick-validation.json` (`<state-dir>` is the managed state directory holding
`site.yaml`, not under `release-deploy/<release_id>/`; the release driver writes the online release state digest in place
after the deployment completes, mode 0600, and evidence judged non-reusable is deleted
rather than left behind). A subsequent `gpu-fault-admin status --full` -- whether the site is specified with `--state-dir` or `-f`
-- reads it from the same path, reuses the read-only checks that just ran (the control plane role split
and each GPU cluster's data plane executor probe), and lists
`reused_validation_checks` in the report. Reuse has hard conditions: release id, delivery digest, site identity and online
release state digest must all agree, and the evidence must be no older than 10 minutes; if any is unmet the probes are rerun,
with the reason given in `validation_evidence_fallback`. `status` without `--full` runs only the two lightweight checks
`cpu_workloads` and `control_api` (`health_scope: quick`) and does not touch
these probes at all. The verify gate inside the release driver likewise reuses the just-written evidence after a successful finalize (the driver passes the file path to verify via
`GPU_FAULT_QUICK_VALIDATION_EVIDENCE`, with the same hard conditions as `status --full`);
a manual verify/probe without that environment variable always reruns.

`status` first prints five human-readable summary lines on stderr (healthy or not, online release id and phase,
the next deployment classification, failed check names, where the JSON went), then outputs the complete JSON unchanged to stdout; scripts parse
stdout as before.
Synchronisation of the installation resource registry is also no longer on the critical path of `deploy`'s return code:
it is performed by the release driver after the release commit, and a failure is recorded only as
`installation_registry.status=UNAVAILABLE` in `state.json` and one `completion_warnings` entry; the release itself is still
`COMPLETED`.

If the command stops because of a Runtime Profile policy change, the stop message has already printed
the review fields of `<state-dir>/release-deploy/profile-plan.json`, the real `plan_sha256` and the resume
command. After review and change-ticket approval, run that resume command as is (replacing only the change ticket number):

```bash
gpu-fault-admin deploy \
  --state-dir /secure/gpu-fault \
  --approve-profile-plan <plan_sha256> \
  --reference CHG-12345
```

This command approves first and then continues the same deployment; there is no separate approval verb. `--approve-profile-plan` accepts only
the digest of the current pending plan; with no plan or a mismatched digest it refuses and gives the current pending digest. The approval is bound to
the specific plan and live baseline; plan drift voids the old approval and stops again printing a new digest, a failed resume needs no
repeated approval, and after success the active record is consumed and cannot be replayed.

| Classification | Actual action |
|---|---|
| `NOOP` | No upload, no migration, no rollout; only the CPU/GPU verifiers run |
| `CONTROL_PLANE_ONLY` | Uploads only the Control Plane wheel and rolls the three CPU tiers |
| `DATA_PLANE_COMPATIBLE` | Uploads only the changed Executor/Node artefacts and proceeds in GPU cluster order |
| `FULL` | Performs the complete two-phase release when schema, protocol, Agent configuration, Profile or cluster set changes |

The physical SHA-256 confirms the actual files installed on Pods/nodes, and the component `module_digest` judges the behavioural content.
Both Agent and Executor carry the artifact SHA, compatibility digest and protocol; the required/compatible window opens only when these
values actually change. A shared release advances one GPU cluster at a time by default;
within a single cluster the Executor converges first, Watcher/Collector run in parallel in the same wave, then the Reconciler and Node Runtime follow.
The first upgrade wave is fixed at 1, then adapts by node scale and failure domains to 4, 8, 16 or an explicit 32; the first rollback wave
is fixed at 1, with later waves conservatively adapting to 1, 2 or 4.

An upgrade touching the GPU or Node Runtime submits the candidate preflight as a background phase after `uploaded`, running in parallel with the schema Jobs and
the CPU stage, and joins before entering the data plane, writing `candidate-preflight-ready` (the join is the gate, not the
submission): the candidate Manifests complete a server-side dry-run, and a per-node read-only host
probe proves the artefact digests, disk, Python/systemd/driver and rollback slot. Before the first node mutation
it still re-checks Node state, Executor, remote commands, destructive workflows and Agent leases.
An early gate failure triggers no CPU/GPU rollout; a late dynamic gate failure trims the rollback by existing transaction evidence.
A resume that has not reached `data-converged` re-runs the read-only candidate/host preflight and never permanently trusts an old checkpoint.

When a GPU cluster hits local capacity, an Agent safety gate or convergence timeout, or an Installer failure, the site release enters `PAUSED`: clusters already
`CONVERGED` stay on the candidate version, the currently failed cluster keeps its checkpoint, and later clusters stay `PENDING`.
After fixing the cause, rerun the same four-parameter deploy; the system re-validates the release, execution plan, previous digest,
CPU compatibility window, cluster identities and node sets, then resumes from the failed cluster. Unknown scope or a global fault still fails closed.

**Success criteria**

- The three CPU tiers, the GPU Executor/Watcher/Resource Collector/Reconciler, and the node
  systemd Collectors/Agents have all converged;
- Control Plane, Executor and Node Runtime each point at the target component;
- Agent/Executor required artifact and compatibility digest are the new version, and compatible is empty;
- Runtime Profile, Agent config digest and protocol agree;
- The verifier passes and remote commands have no new backlog.

**Rollback**

```bash
gpu-fault-admin deploy --state-dir /secure/gpu-fault-bootstrap --rollback
```

`--rollback` goes back only one step: the target is always `previous` in the release state, i.e. the release running online before the current release
was committed; there is no release id parameter and no confirmation literal is required. The command first reads the online
state and refuses with a reason in these cases: no `previous` (after the first bootstrap or a NOOP release);
a transaction still in progress (any intermediate upgrade/rollback phase, `complete` not committed, commit cleanup not
finished -- first rerun `deploy --state-dir X` as is to let it resume/finish); a PostgreSQL schema crossed between the previous release and the current release
(the engine refuses, and the command relays the engine's words and the snapshot id).

After the rollback completes, the state is `rolled-back`, `previous` is the version now online, earlier committed releases
are no longer recorded, so **a second consecutive `--rollback` is refused**; to go back to an earlier version, check out the commit that produced that old
release and `deploy` via the ordinary flow (the engine's words: `deploy the older commit as an ordinary release`).
Changing only `dist/current-release.json` is useless: reusing it requires the signed attestation's `source.git_commit`
to equal HEAD with a clean working tree, otherwise the build regenerates and overwrites it. The
Secret backups needed for rollback are kept at commit until the next transaction commits, so the rollback of a committed release can
find its own backups. After the engine finishes, the command aligns the management side like an automatic rollback (changing `site.yaml` back
and `sync-state`), and `gpu-fault-admin status` then judges health by the rolled-back online version.

Running only `kubectl rollout undo` does not restore the release metadata, installer bundle or node
artefacts and does not constitute a regional rollback.

If the rollback target is an old single-wheel release without separate component pins, the orchestrator automatically rolls the Agent
compatibility digest back to the old artifact SHA and removes the 6 component-pin environment variables that only the new version recognises before applying the old Deployment;
these variables must not be left for the old process by hand.

After an upgrade failure or automatic rollback, simply rerun the same four-parameter `gpu-fault-admin deploy`. A
`failed` transaction not yet rolled back allows only the same release to resume with the original `release_diff`; once rollback has been verified and the
`rolled-back` terminal state entered, the next candidate reuses the conservative diff set but opens a new transaction from the current online baseline, not treating the old
release checkpoint as resume state. A new transaction clears the previous release's rollback phase and cluster
checkpoints, avoiding misjudging a historical rollback as the current rollback having completed. Do not delete the release state ConfigMap.
Rollback compensates only components and clusters with mutation evidence and restores in reverse of the actual upgrade order; an Executor-only failure
does not reinstall Agents, and a GPU-only transaction that did not change CPU pins does not roll the CPU pointlessly. After the rollback verify passes,
the management site, capacity desired configuration and release state are aligned to the previous version, so
`gpu-fault-admin status` should report health by the current rolled-back online version rather than keep comparing the failed candidate.
`partial-convergence` continues the original upgrade checkpoint; any `rollback-*` or
`rollback-failed` only continues the rollback. If the online state is already commit/rolled-back and only the temporary Secret backup
cleanup failed, a rerun only completes the cleanup without rolling components again. When the site sets `autoRollback=false`, a verify or
stability failure keeps the candidate intermediate state and marks `SKIPPED_POLICY`.

**deploy says "a failed fail-forward transaction only resumes the release that failed".**
This is the refusal you see when a fail-forward transaction stopped at `failed`/`partial-convergence` and you fixed the defect and deploy again:
the fix produced a new release id, and the failed transaction allows only the same release to resume. The error states the failed
release id, the current candidate id and the way out -- add `--supersede-failed-transaction` on the same four-parameter deploy:

```bash
gpu-fault-admin deploy \
  --cpu-cluster-arn <cpu-arn> \
  --gpu-cluster-arn <gpu-arn> \
  --state-dir /secure/gpu-fault \
  --admin-email <operations-email> \
  --supersede-failed-transaction
```

It opens a **new transaction** for the new candidate: the rollback baseline inherits the failed transaction's `previous` (the previous committed release and its Secret
backups, not the present mixed state, so an automatic rollback lands on the known-good version); the components to roll are "the new candidate's changes relative to the failed candidate
∪ the components the failed transaction already touched", so anything the failed candidate rolled is re-rolled even when the new candidate equals the baseline; the superseded transaction is recorded in
the `superseded_transaction` field of the release state ConfigMap (`release_id`, `phase`, `superseded_at`,
`original_failure`, etc.); `live_release` in `status` currently does not show it, so read the release state to see it. A schema change already accepted by the failed transaction
with the same version is carried over automatically without passing `--accept-schema-change` again. In any other state
(`complete`, a mid-way checkpoint, `rollback-*`, `rolled-back`, the candidate being the failed release itself) the parameter is refused with
a reason; rerunning the same release id neither needs nor may carry this parameter -- that is an ordinary resume. Do not edit the
release state ConfigMap or hand-write scripts calling the orchestrator to bypass the refusal. Details are in `部署和运维手册.md` (Deployment and Operations Manual) REG-8.1.4.

**Aurora password rotation and `password authentication failed`.** RDS rotates the Aurora
master password every 7 days, and the `gpu-fault-aurora` Secret is synchronised only by the `gpu-fault-aurora-credential-refresh`
CronJob (now hourly at `17 * * * *`). Running Pods re-read the DSN through the `/etc/gpu-fault/aurora/postgres-url`
mount: after the Secret refresh, kubelet sync (about 1-2 minutes) restores them, and connections rebuilt after `max_idle` reclamation before the refresh
fail authentication and are retried by the pool with backoff, no longer losing slots (control-plane review 2026-09-08, CP-3); the refresher by default
no longer rolls the Deployments, and the three create/wait/delete lines in the release output are unchanged. Between the rotation and the refresh, every **newly created**
control-plane Pod still exits with `FATAL: password authentication failed` --
and a release is exactly a batch of new Pods. On 2026-09-07 12:13Z a production transaction ran into this window: the re-apply
failed, the automatic rollback restarted control-worker, the new Pods likewise could not authenticate, the transaction stopped at `rollback-failed`,
and from the release output it was indistinguishable from a genuinely broken release. Therefore deploy (including resume) and rollback now
both synchronously run this CronJob once before rolling any component and fail closed:

- Normally the release output gains three lines `+ kubectl ... create job --from=cronjob/...`, `wait`,
  `delete job`, taking a few seconds;
- If the Job does not Complete within the limit, the transaction fails with
  `Aurora credential refresh Job gpu-fault-system/gpu-fault-aurora-credential-refresh-<suffix> did not complete within 300s`,
  writing no release state, rolling no component, and leaving the Job in place. Read
  `kubectl -n gpu-fault-system logs job/<name>` and `describe job/<name>` by the Job name in the error message, fix it (usually
  Pod Identity/KMS permission or the RDS CA bundle), then `delete job` and rerun the same command;
- The limit is overridden with `GPU_FAULT_RELEASE_AURORA_REFRESH_WAIT_SECONDS` (positive integer seconds) on the deployment host;
  old sites without the CronJob print `aurora-credential-refresh skipped` and skip.

If a transaction is still stopped at `rollback-failed` and the new Pod logs read `password authentication failed`
(`PoolTimeout` is only the surface symptom), do not touch the release: run by hand
`kubectl -n gpu-fault-system create job --from=cronjob/gpu-fault-aurora-credential-refresh <name>`
(about 5 seconds, Pods Ready in about 2 minutes), then rerun the same deploy command to resume. Diagnosis and recovery details are in
CPU-2R of the Deployment and Operations Manual.

Changes to the regional cluster registry (`clusters`, i.e. the cluster set) are not transactional, and
with `autoRollback=true` the deployment fails before rolling any component with
`automatic rollback is not yet transactional for: <field>`. Such a release must be switched to fail-forward explicitly by the administrator:
set `spec.autoRollback` to `false` in `site.yaml` and rerun the same deploy
command. Redeploying re-renders the site but preserves the administrator-declared `autoRollback` and does not quietly change it back to
`true`; missing is treated as `true`, and a non-boolean is refused by site validation (`spec.autoRollback must be a boolean`).
A failure during fail-forward does not roll back automatically; the recovery means are
to fix and rerun the same deploy command to resume, or to roll back explicitly to the previous committed release with `deploy --state-dir X --rollback` from above
-- but **a release across schema versions cannot be rolled back this way**; the explicit rollback likewise refuses with
`rollback across this PostgreSQL schema change is not declared backward-compatible`,
see §4.7. After the release commits,
`spec.autoRollback` should be changed back to `true` so that subsequent transactional releases regain automatic rollback.

The ADOT Manifest and ADOT image do not belong to this class; changing only ADOT needs no fail-forward: before touching the
collector, the upgrade reads the online ADOT objects into the observability snapshot, and rollback restores those objects from the snapshot, deletes the objects newly added by the
candidate, and restarts the collector waiting for it to be Ready again, restoring the configuration and image the previous release itself applied. When the snapshot lacks the collector Deployment, or the old state has no ADOT objects, rollback fails closed and does not claim
to have compensated a plane it did not restore. When resuming a failed transaction started before this capability went live, its previous snapshot has no ADOT
objects, and the deployment fails before rolling any component with
`previous observability snapshot predates ADOT rollback capture`; that one transaction still resumes via the fail-forward flow above,
and after commit `spec.autoRollback` is changed back to `true`. The endpoint (NLB Service and
Route53 record) is now likewise compensated by snapshot and belongs to this class; when previous has no endpoint snapshot there is the same-shaped refusal
`previous snapshot predates endpoint rollback capture; deploy this release with spec.autoRollback: false`,
handled as above. When the old previous lacks the container environment snapshot, rollback does not refuse; it only records
`snapshot-predates-capture` at the `rollback-cpu-container-env` step and starts the old image with the current template's env names.

### 4.0 Agent Disappears After HyperPod Re-images the Node (UpdateClusterSoftware)

`UpdateClusterSoftware` reinstalls the GPU node's system disk in place: the Node object and UID are unchanged and all annotations preserved,
but `/opt/gpu-fault`, `/var/lib/gpu-fault`, `/etc/gpu-fault` and all `gpu-fault-*` systemd
units disappear, and the Agent lease expires accordingly. The node-installer-reconciler used to compare only annotations and Node UID,
so it judged such a node `current` and did not reinstall; the release then failed at "previous Agent identity capture" with
`<cluster> has no active Agent identity to capture`.

The reconciler now binds the installation record to `status.nodeInfo.bootID` (annotation
`gpu-fault.io/installer-boot-id`): after a boot id change it first probes the Agent port
(`GPU_FAULT_NODE_AGENT_PORT`, default 9099) -- a response means an ordinary reboot, and the boot id is re-stamped;
no response with the node Ready for more than `GPU_FAULT_INSTALLER_REBOOT_GRACE_SECONDS` (default 600) marks
`Retrying` and reinstalls in the regular waves (the two counters `rebooted`/`recovering` appear in the reconcile log).
Nodes installed by an old version lack the annotation; the first reconcile only adds the annotation without reinstalling.

If a re-image already happened before the release carrying this fix was deployed, trigger the reinstall through the reconciler's own retry channel:

```bash
kubectl --context "${GPU_CONTEXT}" annotate node <node> \
  gpu-fault.io/installer-state=Retrying --overwrite
```

The reconciler creates an installer Job for that node (bound by `max-unavailable`, proceeding node by node), and after completion
the Agent lease recovers. This changes only the installation bookkeeping annotation; the isolation/incident annotations or taints must not be changed the same way.

The reconciler and cluster executor now each have their own `/metrics` (9110 / 9111, scraped into AMP by each GPU cluster's
`gpu-fault-adot-dataplane`, series carrying `gpu_cluster`), so checking "is the loop still turning" no longer needs
`kubectl exec` to read the heartbeat file: `gpu_fault_node_installer_reconcile_passes_total` flat across several periods, or
`time() - gpu_fault_node_installer_last_pass_completed_timestamp` growing continuously = the reconciler loop is not doing work;
`gpu_fault_node_installer_nodes_pending > 0` with `budget_in_flight == 0` unchanged for a long time = no installation is being scheduled.
`gpu_fault_cluster_executor_claim_loop_alive == 0` or `last_loop_iteration_timestamp` older than 300 s
= the claim loop has stopped (the same signal as the exec liveness); loop alive but `last_claim_timestamp` aging = the control plane is unreachable,
not a broken executor. `gpu_fault_cluster_executor_consecutive_transport_degraded_cycles > 0` persisting = the executor's own network is degraded and
the claim loop is backing off; `retryable_transport_errors_total` rising while `retryable_adapter_errors_total` does not = the problem is on the executor side, not the
target side; `batched_progress_failures_total` rising only means control-plane latency, no lost steps; `results_withheld_total` rising = the lease was lost before the result,
and the next holder redoes that step. Variables and prerequisites are in the path B section of the Deployment and Operations Manual.
### 4.1 Reading Durations in the Deployment Log

One upgrade prints more than three thousand lines; previously only `release-phase` carried second-level timestamps, and any question of "why was this one slow"
meant subtracting two timestamps by hand, while the duration of a single command was nowhere to be found. Now read it in three layers:

| Line | Meaning |
|---|---|
| `release-begin <stamp> mode=<mode> dry_run=<bool>` | This script invocation starts. One `gpu-fault-admin deploy` drives this script a dozen or so times (plan/preflight/deploy/verify/commit), so the log contains several begin/end segments |
| `release-end <stamp> mode=<mode> exit_code=<n> total=<seconds>s` | This invocation ends, including failure paths. No end line means the process is still stuck in the last command, or has been killed |
| `release-phase <stamp> <phase> ... elapsed=<seconds>s total=<seconds>s` | `elapsed` is the cost from the previous checkpoint to this checkpoint, `total` is the cumulative time of this invocation |
| `command-elapsed <seconds>s <command>` | Duration of a single command. Printed only above 15 seconds, otherwise thousands of sub-second kubectl calls would drown the command trace; `probe`-class read-only checks, which normally do not echo the command, also use this line |

A phase is the interval between checkpoints, and the longest stretch of an upgrade falls inside **one** phase:
`data-plane-progress` checkpoints only once per cluster, so a four-wave cluster prints two phase lines seven and a half minutes apart,
with nothing but `get nodes`/`get jobs` polling in between. That stretch now has its own lines, all with `total`
but without `elapsed` -- they print inside the phase, and taking the phase's split would record a seven-minute phase as
its last thirty seconds:

| Line | Meaning |
|---|---|
| `fleet-wave <stamp> cluster=<id> phase=<phase> wave=<i>/<n> ready=<ready>/<total> nodes=<nodes>` | A wave starts. `wave` is its position in this cluster's wave sequence, `ready` is the number of nodes already on the new identity before the rollout |
| `fleet-wave-done <stamp> cluster=<id> wave=<i>/<n> ready=<ready>/<total> safety=<seconds>s handoff=<seconds>s install=<seconds>s elapsed=<seconds>s` | A wave ends, splitting its duration into three parts: `safety` is the wait for Agent leases and unfinished commands, `handoff` is the Reconciler rollout plus cleanup of the previous wave's leftover Jobs, `install` is the nodes installing themselves |
| `installer-wait <stamp> cluster=<id> nodes=<n> aligned=<n> waiting=installers\|heartbeats pending=<node>:<state>,... elapsed=<seconds>s` | The wait inside a wave. `waiting=installers` means nodes are still installing, and `pending` lists each node name with its `gpu-fault.io/installer-state` (`<none>` means the Reconciler has not created the Job yet; the next reconcile pass will); `waiting=heartbeats` means the nodes have installed the new identity and are waiting for the Agent heartbeat to report. Printed whenever the state changes, otherwise repeated every 30 seconds; with more than 5 nodes only the first 5 are listed, ending with `+N` |

Only the three-part split of `fleet-wave-done` tells whether a slow wave is worth optimising: a large `install` is the cluster doing
the upgrade's real work; large `safety`/`handoff` is the control plane's own overhead. `installer-wait` polls every 5 seconds
(each poll is one `get nodes` of about 2 seconds); the former 15-second backoff has been removed -- when a wave rolls one
node, the backoff would sleep through the whole tail of the next node's installation. This only reduces the idle gap between state convergence and the next read,
and does not shorten node installation or the safety window.

The ten static gates of `make check` run in parallel, and wall time equals the slowest gate, so each gate prints
`static-gates: <gate> passed in <seconds>s` at the end; when a gate contains several commands, any command over 5 seconds is printed separately as
`static-gates: <gate> step took <seconds>s: <command>`, and failure lines also carry the command name and duration.

Durations are measured with a monotonic clock, so a clock jump mid-way does not produce negative phases. Timing resets with every process invocation,
so `total` is the duration of this invocation, not of the whole deploy. `release-end` of parallel or nested invocations cannot
simply be added; for end-to-end duration look at the top-level start and end times, and for phase comparison distinguish execution, quota waits, cloud convergence and retries.
The phase necessity and parallelism boundaries of join/remove/uninstall are in §3, §5 and
[Developer Deployment Implementation](developer-deployment-implementation.md#81-join-phases-performance-and-compensation-boundaries); locally simulated durations do not represent AWS performance.

**Boundaries**

- These lines are observation only and are not gates. Do not raise a timeout, skip a probe or delete a gate because some phase was slow;
  the same constraint of §8 applies here.
- The duration line of `<sensitive command>` likewise does not expand the arguments. Do not print the arguments of sensitive commands to see a slow command more clearly;
  commands containing passwords/tokens must stay invisible.
- Do not raise the slow-command threshold to make the log "clean": that re-hides exactly the slow commands you are looking for.
- When `installer-wait` stays on the same `pending` node for a long time, deal with that node (look at its installer
  Job and `installer-state`), not by shortening the `wait_agents` timeout or relaxing the convergence decision: the membership decision of `pending`
  and the gate use the same `agents_converged`; changing one changes the gate.

### 4.2 The Three-Step Index Method (Why DDL Must Not Build Indexes)

- The idempotent DDL runs in **one transaction**, and PostgreSQL forbids `CREATE INDEX CONCURRENTLY` inside a transaction;
  if the ensure Job or Pod startup were to build a new index, it could only use a plain `CREATE INDEX`, during which
  writes to `gpu_fault_objects` (all workflows / incidents) or `gpu_fault_processor_queue` (the processor queue)
  are locked, and admission, claiming, completion and dispatch all stall, for a duration depending on table size.
- Hence the rule: (1) the `gpu-fault-postgres-index-build` Job builds online with `CONCURRENTLY`; (2) the DDL first queries
  `to_regclass` before deciding whether to issue `CREATE INDEX IF NOT EXISTS` -- the bare `IF NOT EXISTS` still takes a ShareLock on the table until commit when the index already exists,
  so "skip if exists" must happen outside the statement; (3) every replica validates at startup that all indexes declared by the code
  exist, refusing to start when any is missing (`PostgreSQL indexes are missing`) -- better not to start than to let hot queries
  degrade to full table scans unnoticed.
- A failed `CONCURRENTLY` index build (connection dropped, timeout) leaves an invalid index with `indisvalid = false`;
  the index-build Job first runs `DROP INDEX CONCURRENTLY` and then rebuilds. Manual check:

```sql
SELECT c.relname, i.indisvalid
FROM pg_class c JOIN pg_index i ON i.indexrelid = c.oid
WHERE c.relname LIKE 'gpu_fault_%' ORDER BY 1;
```

- When manual execution is needed (for example the Job cannot run), run
  `gpu-fault-store-migrate --build-indexes-concurrently --postgres-url <url>` on the writer one by one, or hand-write
  `CREATE INDEX CONCURRENTLY IF NOT EXISTS` per the definitions in `src/gpu_fault/store/postgres/ddl*.py`.

### 4.3 Schema Preflight Checklist Before a FULL Release

The `gpu-fault-postgres-schema-preflight` Job checks the following three items read-only, exiting non-zero and stopping the deployment when any is unmet;
it can also be run by hand at any time with `gpu-fault-store-migrate --schema-preflight --postgres-url <url>`:

| Check | Pass condition | What to do when it fails |
| --- | --- | --- |
| Schema version | The maximum version of `gpu_fault_schema_migrations` equals the version the wheel requires, and the history checksums agree | Behind: run the ensure Job; ahead: the wheel is older than the database, switch to the correct release; checksum mismatch: stop and compare the migration registrations by hand |
| Indexes | All indexes declared by the code exist and are `indisvalid` | Run the index-build Job (§4.2) |
| In-flight safety flows | No `RUNNING`/`SAFETY_PENDING` workflow with non-empty `blocked_reasons` and a missing `safety_only` field | Wait for these workflows to finish (usually a few minutes) before rolling; they are old records written before the `safety_only` field appeared, and the new code cannot tell which set of steps they should run |

The report also contains a **non-blocking** section `diagnostics` / `warnings`: the current values of `log_lock_waits`, `deadlock_timeout`,
`log_min_duration_statement`, `shared_preload_libraries`, and whether `pg_stat_statements` is installed.
With `log_lock_waits` off, the `40P01` deadlocks the store retries away are invisible in the database log; without `pg_stat_statements`
there is no per-statement ACU view. Both are parameter group matters (§4.6); once the parameters are in place, use
`gpu-fault-store-migrate --ensure-diagnostics --postgres-url <url>` to create the extension in the database and print the same diagnostics.

The corresponding read-only query (run on the writer or a read replica):

```sql
SET default_transaction_read_only = on;
SELECT key, payload->>'status'
FROM gpu_fault_objects
WHERE kind='workflow'
  AND payload->>'status' IN ('RUNNING','SAFETY_PENDING')
  AND jsonb_array_length(coalesce(payload->'blocked_reasons','[]'::jsonb)) > 0
  AND coalesce(payload->>'safety_only','false') <> 'true';
```

### 4.4 Configuration Items Added in This Round (Defaults Are Production-Ready)

All of the following variables have code defaults, and the deployment configuration needs no new keys; they are listed so you know they exist and when to adjust them.
The complete list is in `管理员环境变量参考.md` (Administrator Environment Variables Reference).

| Variable | Default | Meaning | When to adjust |
| --- | --- | --- | --- |
| `GPU_FAULT_WORKFLOW_INTERNAL_ERROR_BACKOFF_SECONDS` | 60 | How long a workflow backs off before retrying when dispatch hits an unrecognised internal error (previously went straight to BLOCKED) | Lower it when internal errors are mostly transient and faster retry is wanted |
| `GPU_FAULT_PROCESSOR_DEADLINE_EXCEEDED_PROCESS_THRESHOLD` | 3 | How many distinct requests must exceed their execution deadline before the processor process concludes the fault is its own and self-destructs | Take 2 to be more conservative; a single poison request can no longer restart the whole fleet |
| `GPU_FAULT_PROCESSOR_UNHEALTHY_TTL_SECONDS` | 300 | How long the process unhealthy latch persists before clearing automatically | Especially important when the self-destruct switch is off, otherwise the background services stall permanently |
| `GPU_FAULT_JOB_WORKFLOW_MAX_LIFETIME_SECONDS` | 3600 | Hard cap of a job workflow (including job restart) from first claim; on expiry it fails and escalates to a human, and later events of the same job are only recorded | Raise for large jobs or long spare-switch chains |
| `GPU_FAULT_NODE_WORKFLOW_MAX_LIFETIME_SECONDS` | 3600 | Hard cap of a single-node jobless workflow, shared by the whole reset->reboot->replace chain | As above |
| `GPU_FAULT_BRANCH_ESCALATION_MAX_RUNGS` | 2 | Maximum number of in-place escalation rungs of a single node branch inside a job workflow (reset->reboot->replace) | Set to 1 when automatic spare switching is not wanted |
| `GPU_FAULT_JOB_WORKFLOW_NODE_BUSY_WAIT_SECONDS` | 240 | The single window of rule A: when a training job's node is occupied by another repair, the job waits only this long. If the repair finishes within the window the job continues (the queued job workflow is dispatched; the `after_incident` restart restarts the job); beyond the window the job is not restarted -- a still-running job is stopped with a STOP_WORKLOADS carrying the control-plane initiator marker, a dead job's restart plan fails, the workflow is FAILED, the incident ESCALATED, and a job the system itself stopped is never restarted by anyone. The dispatcher (job workflows not yet started) and the executor (the precondition of a restart already running) read the same variable | Raise when single-node repairs often exceed 4 minutes; unequal values at the two read sites refuse startup |
| `GPU_FAULT_PROCESSOR_LEASE_RECLAIM_SECONDS` | 30 | How often the periodic task scans the processor queue to return requests still LEASED with an expired lease to PENDING; each reclamation counts in `gpu_fault_processor_expired_leases_reclaimed_total` | Usually untouched; may be relaxed in step when the request lease itself (`GPU_FAULT_PROCESSOR_REQUEST_LEASE_SECONDS`, default 120) is lengthened |
| `GPU_FAULT_PROCESSOR_COUNTER_DRIFT_SCAN_SECONDS` | 60 | How often the periodic task compares the processor queue shard counters with the real PENDING/LEASED row counts, refreshing `gpu_fault_processor_counter_drift_abs` and `gpu_fault_processor_counter_mismatched_clusters` (not computed in the scrape) | Lengthen when the queue table is very large and this COUNT visibly slows Aurora; the alert window is 5 minutes, do not approach it |
| `GPU_FAULT_WORKFLOW_INVARIANT_CHECKS` | log | After every lease-bearing write the executor checks the workflow record's state-machine invariants (indexes ordered and unique, DAG join markers, no owner/lease in terminal states, `events` bounded and ordered, etc.); `off` skips, `log` records an error log and continues, `raise` fails that write | Keep `log` in production; use `raise` in staging and acceptance drills to expose malformed records on the spot |

**Ordering relationships between timing parameters.** The wait-class parameters in the table above and elsewhere in §4 are not independent of each other; the code has an implicit ordering,
checked item by item at startup by `gpu_fault.execution.config.validate_timing_relationships` (a violation refuses startup;
items that cannot be checked are given as `warning:` logs): every step wait cap (default `GPU_FAULT_WORKFLOW_STEP_TIMEOUT_SECONDS`
and each override, including the managed recovery window of `GPU_FAULT_HYPERPOD_MANAGED_RECOVERY_TIMEOUT_SECONDS`) <= the single-node lifetime
`GPU_FAULT_NODE_WORKFLOW_MAX_LIFETIME_SECONDS` (otherwise the lifetime fails the step before the step cap, and the adapter's own timeout handling --
the managed recovery escalation notification -- never runs; the human confirmation cap of `CHECK_MECHANICALS` is the exception, and at claim time the lifetime takes it as a floor);
managed recovery window + one reboot quota <= the job lifetime `GPU_FAULT_JOB_WORKFLOW_MAX_LIFETIME_SECONDS`, and the job lifetime >=
`GPU_FAULT_BRANCH_ESCALATION_MAX_RUNGS` × the managed recovery window (an escalated branch performs at most that many delegated recoveries); the execution timeout
`GPU_FAULT_WORKFLOW_EXECUTION_TIMEOUT_SECONDS` <= both lifetimes (claim takes the smaller of the two; a larger setting is as good as none); the lease
`GPU_FAULT_WORKFLOW_LEASE_DURATION_SECONDS` < the execution timeout; `GPU_FAULT_JOB_WORKFLOW_NODE_BUSY_WAIT_SECONDS`
must be equal in the executor configuration and the dispatcher configuration (rule A has only one window, both read the same variable, and inequality means someone configured separate copies bypassing the environment variable), and <
`GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS` × `GPU_FAULT_WORKFLOW_POLL_INTERVAL_SECONDS` (the moment the job workflow gives up waiting and
STOPs the job must precede the moment the node repair's VERIFY gives up, otherwise the STOP can never help the VERIFY in time);
`GPU_FAULT_WORKFLOW_DISPATCH_LEASE_SECONDS` >= 3 × the poll interval (the default is `max(15, 3×poll)`; validated when overridden explicitly);
`GPU_FAULT_WORKFLOW_DISPATCH_CYCLE_SECONDS`, if > 0, >= the poll interval. Before adjusting any one of them, look at its neighbours against this relationship table.

### 4.5 Check `/metrics` for the Processor Counter Mode Before Deciding Whether to Finalize

Which counter table the per-cluster depth of processor admission reads is decided by the single row
`gpu_fault_processor_counter_mode`, with only two values: `dual` (the default during greenfield initialisation and rolling upgrades) and `partitioned`.
`dual` is not "half open": every enqueue still locks that cluster's row in `gpu_fault_processor_queue_counts` until commit,
and the 16 priority shards are merely an extra copy, relieving row-lock contention not at all; only `partitioned` makes the shards the source of truth.
When the production database was inspected on 2026-09-07 it had stayed at `dual` all along, and no metric had exposed that before.

`/metrics` now exports the two series `gpu_fault_processor_counter_mode{mode="dual"}` and
`{mode="partitioned"}`, with the active one at 1 (SQLite/in-memory stores do not export). The routine check after an upgrade:

```bash
# control-worker's /metrics is on 8081 (8080 is the api-ha ingress, 8082 is the spool-worker)
curl -s http://<control-worker>:8081/metrics | grep gpu_fault_processor_counter_mode
```

When you see `mode="dual"} 1` and all replicas have rolled to the same module digest, finalize in the order of CPU-4U of `部署和运维手册.md` (Deployment and Operations Manual):
first confirm the processor queue's `PENDING/LEASED` are 0 (both `expected_total` and `counter_total` of `gpu-fault-store-migrate
--processor-counter-shard-status` are 0), then run
`postgres-counter-shards-finalize-job.yaml` (`--finalize-processor-counter-shards`).
finalize re-seeds the shards under the advisory lock and the queue table lock, and refuses outright when the queue is not empty; do not rely on
repeated retries. After the switch every process reads the new mode within at most 0.25 seconds and the metric flips accordingly; if some replica does not flip for a long time, check its
`gpu_fault_periodic_job_errors_total` first; do not restart it to "refresh".

The rollback direction is `--restore-legacy-processor-counters`, likewise requiring an empty queue, and it must precede rolling back to an old wheel that does not
understand priority shards.

### 4.6 Aurora Parameter Group and Log Export: deploy Aligns Automatically, Only the Restart Window Remains Manual

When the store hits `40P01` (deadlock detected) it retries automatically and succeeds, so deadlocks are invisible to the business; in the Aurora
default parameter group `log_lock_waits` is off, so they are invisible in the database log too. `pg_stat_statements` is the only view that shows per
statement where the ACU goes, and it is not installed by default either.

**None of this needs doing by hand any more.** The regional bootstrap builds Aurora itself (`src/gpu_fault/admin/bootstrap.py`
`_ensure_aurora`) and now creates the database with the dedicated cluster parameter group `<cluster-name>-pg` and `postgresql` log export; for a cluster already
online, every `gpu-fault-admin deploy` reconciles just as it corrects ACU capacity: a missing parameter group is created, deviating parameters (four of them) are changed,
and a cluster not attached to this group or without export enabled gets one `modify-db-cluster --apply-immediately`. Everything takes effect online, restarts no
instance and breaks no connection; an already aligned database receives not a single modification command. The four parameters:

| Parameter | Value | How it takes effect |
| --- | --- | --- |
| `log_lock_waits` | 1 | Effective on attaching the group |
| `log_min_duration_statement` | 1000 | Effective on attaching the group |
| `pg_stat_statements.track` | all | Effective on attaching the group |
| `shared_preload_libraries` | pg_stat_statements | **Needs a writer restart**; until then the cluster shows `pending-reboot`, which is the expected state |

Read-only check (Aurora is in the cluster's region; mind `--region`; the deployment host's default region may not be it):

```bash
aws rds describe-db-clusters --db-cluster-identifier <aurora-cluster-id> --region <region> \
  --query 'DBClusters[0].[DBClusterParameterGroup,EnabledCloudwatchLogsExports]'
aws rds describe-db-instances --region <region> \
  --query "DBInstances[?DBClusterIdentifier=='<aurora-cluster-id>'].[DBInstanceIdentifier,DBParameterGroups[0].ParameterApplyStatus]"
```

**The only manual step: get the `pg_stat_statements` library loaded.** Both instances are promotion tier 0 and each other's failover targets;
pick a window with no live acceptance case and no `LEASED` workflow:

1. `aws rds reboot-db-instance --db-instance-identifier <reader>`, wait for it to be `available`;
2. `aws rds failover-db-cluster --db-cluster-identifier <aurora-cluster-id>`; the reader is promoted to writer,
   business connections drop once at that moment, for seconds to tens of seconds, and the store's connection pool self-heals;
3. `aws rds reboot-db-instance --db-instance-identifier <original writer>`.
4. Afterwards `gpu-fault-store-migrate --ensure-diagnostics --postgres-url <url>` runs
   `CREATE EXTENSION IF NOT EXISTS pg_stat_statements` in the database; in the deployment's `--schema-preflight` report
   `diagnostics.pg_stat_statements_installed` should be true.

After instance names are truncated to 63 characters, the `-writer` / `-reader` suffix may be displaced by the digest; read the role from
`DBClusterMembers[].IsClusterWriter`.

Once log export is on, RDS pushes `postgresql.log` to the log group `/aws/rds/cluster/<aurora-cluster-id>/postgresql`,
one log stream per instance; set a retention period on this log group (30 days recommended), otherwise it is kept forever by default. In normal operation this database runs tens of
statements per second, the four parameters produce log only on lock waits and slow statements, and the log volume is measured in KB/day. Deadlocks appear in that log group as
`process ... detected deadlock while waiting`, queryable with Logs Insights
`filter @message like /deadlock detected/`, or alertable via a metric filter; statements over 1 second enter the log with the full SQL
-- note that workflow/command ids appear in the log, and per the "Fault evidence privacy constraint" do not paste these lines into external tickets.

### 4.7 Releases That Change the Schema Version: One Command, One Parameter

**How to know this release changed the schema.** When the release's `database_schema_version` differs from production, `deploy`
stops before rolling any component, reporting
`automatic rollback across this PostgreSQL schema change is not declared backward-compatible`,
and tells you which parameter to add. This is by design: at startup the new wheel requires `gpu_fault_schema_version.version`
to exactly equal the code, and a Pod rolled back to the old wheel refuses to start on the new-version database. Therefore
`gpu-fault-admin deploy --state-dir X --rollback` also refuses with
`rollback across this PostgreSQL schema change is not declared backward-compatible`.
**No switch can declare this schema change "rollback-capable"**: `database.rollback_compatible: true` in the release manifest is refused outright by `deploy` (the runtime requires the schema version and migration history to be exactly equal, and an old wheel on a new schema inevitably CrashLoops); the only way out is the parameter below.

**Only one parameter needs adding to the same deploy command:**

```bash
gpu-fault-admin deploy \
  --cpu-cluster-arn <cpu-arn> \
  --gpu-cluster-arn <gpu-arn> \
  --state-dir /secure/gpu-fault \
  --admin-email <operations-email> \
  --accept-schema-change
```

This parameter expresses one thing only: I know this crosses a schema version and cannot be rolled back. The rest is done by the release engine within **this one transaction**:

1. Opens that gate and writes the acceptance record (mode, time, schema version) into the release state.
2. Before the three schema Jobs index-build -> ensure -> preflight, creates a manual snapshot of the Aurora cluster
   (named `gpu-fault-pre-v<version>-<first 12 characters of the release id>`) and waits for it to be `available` before continuing.
   An existing snapshot of the same name is reused directly, so a resume does not take another. The snapshot id is written into the release state
   and is visible in `live_release.schema_change_acceptance` of `gpu-fault-admin status`.
3. A later verify or stability failure is handled fail-forward: the candidate keeps its intermediate state, the reason of `SKIPPED_POLICY`
   states the schema acceptance, and no rollback that would certainly be refused is attempted.
4. `site.yaml` is not changed by one character. After the transaction commits, the next deploy reads `spec.autoRollback` as usual (still `true`),
   and automatic rollback is naturally still there -- there is no "change it back" step.
5. After a failure, rerun the same command (with or without the parameter): the engine reads the acceptance from the release state and does not snapshot again.

**What if the snapshot cannot be taken.** The cluster id is taken from `health.aurora_cluster_id`, or derived from the writer address of the `gpu-fault-aurora`
Secret when unconfigured. The deployment host role needs `rds:DescribeDBClusterSnapshots`,
`rds:CreateDBClusterSnapshot`, `rds:AddTagsToResource`; with insufficient permission the command stops before the schema Jobs,
database and Pods untouched, and the error states what is missing. The wait cap is `GPU_FAULT_RELEASE_SCHEMA_SNAPSHOT_WAIT_SECONDS`
(default 900 seconds). For databases nobody will restore (staging) you can forgo the snapshot with
`--accept-schema-change-without-snapshot`; do not use it in production.

**Two paths after a failure.** Fix the cause and rerun the same `deploy` to resume; or restore the database to the old version from the snapshot
(`aws rds restore-db-cluster-from-snapshot`, then switch the application to the restored cluster), at which point the database matches the old wheel,
then check out the commit that produced the old release and `deploy` via the ordinary flow (the "go back to an earlier version" path of the rollback section of §4).
`deploy --state-dir X --rollback` **always refuses** on a cross-schema transaction: it only looks at whether `release_diff.changed`
contains `database_schema` and does not detect whether the database has been restored; the second half of the error text "restore ... first, then rerun the
rollback" does not work. The engine does not restore the database for you: that is a lossy operation and a human must decide.

**Closing check:**

```sql
SELECT version FROM gpu_fault_schema_version;
SELECT max(version) FROM gpu_fault_schema_migrations;
```

Both equal the new version; `gpu-fault-admin status` is healthy; run
`gpu-fault-store-migrate --schema-preflight --postgres-url <url>` once more, with `ok` true.

**v12 special note.** v12 adds 6 indexes on `gpu_fault_objects`, which the index-build Job builds before the rollout;
the ensure phase drops `gpu_fault_remote_command_workflow`, superseded by `gpu_fault_remote_command_workflow_all`.
With a small database (a hundred thousand-odd rows) the whole schema phase completes in one or two minutes. From v12 on, `--ensure-schema` with no changes
no longer takes table locks (start of §4), so the ensure phase of a routine release no longer affects business writes.

**v13 special note (control-plane component review).** v13 adds 9 text-expression indexes on `gpu_fault_objects`
(archive candidates, fleet_deployment terminal state, marker/notification/event retention, workflow/incident pointers, etc.) and sets more aggressive
autovacuum table parameters on `gpu_fault_objects`. The release is still just that one `--accept-schema-change`
parameter: the index-build Job `CREATE INDEX CONCURRENTLY`s all 9 indexes before the rollout, with no need to first run
`gpu-fault-store-migrate --build-indexes-concurrently` by hand; but after v13 the startup validation also checks that the indexes are **valid**
(`indisvalid`) and their definitions match the code; an INVALID index left by an interrupted CONCURRENTLY makes a replica refuse to start,
and the preflight reports `indexes.drifted` -- only then does the index-build Job need a manual rerun (§4.2).

**v14 special note (wakeup trigger).** v14 (`objects-wakeup-notify-trigger`) only adds
a row trigger `gpu_fault_objects_notify_wakeup_trigger` on `gpu_fault_objects`, pushing workflow / remote command
state changes to the dispatcher and Executor with `pg_notify` so they no longer rely on 5 s / 2 s polling; there is no table or column change, the index-build Job
has nothing to do, and `--ensure-schema` installs idempotently via `CREATE OR REPLACE FUNCTION` plus a catalog comparison before creating the trigger;
a rerun on an existing database takes no table lock. The version number still exists because the migration registration checksums every `ddl*.py` file, so the release
is still that one `--accept-schema-change` parameter (and still takes a snapshot). The new wheel validates at startup that the trigger exists, refusing to start when missing
(`PostgreSQL objects wakeup trigger is missing; run gpu-fault-store-migrate --ensure-schema`).

### 4.8 Two Automatic Checks Around a Release: the Archive Bucket and the ADOT Self-Monitoring Metric Names

These two used to live only in release records and human memory; both are now check items in the `gpu-fault-admin deploy` report, and either failing stops:

| Check | Phase | Pass condition | What to do when it fails |
| --- | --- | --- | --- |
| `control_record_archive_bucket` | Preflight | `GPU_FAULT_CONTROL_RECORD_ARCHIVE_S3_URI` is `s3://bucket/prefix`, and the bucket exists in the site Region (read-only `s3api get-bucket-location`) | Normally the bootstrap task `control_record_archive_bucket` has created and hardened the bucket before the preflight; a failure here means the bucket was deleted, is in another Region, or the bootstrap task failed on permissions -- look at that task's error in the deploy output. The control-plane pod identity's `s3:PutObject` on that prefix is granted by bootstrap per `spec.retention`. With the bucket missing the archiver only counts `gpu_fault_control_record_archive_errors_total` and deletes no live rows, but retention has silently stopped |
| `adot_self_metrics` | Post-deployment verify | Reads AMP with a SigV4 instant query: `up{job="gpu-fault-adot-self"}` is 1, and every `otelcol_*` series referenced by the alert rules and the keep list exists under that job (no longer reads the Pod's `:8889` through the API server proxy -- that reader is deliberately bound to 127.0.0.1, and proxying to the Pod IP inevitably gets connection refused, which is exactly how the verify of deploy #25 on 2026-09-09 failed; reading AMP also proves the whole chain of self-scrape, keep list and remote_write) | Means the collector image changed the metric naming (most likely added a suffix like `_bytes`): change the keep regex of `deploy/observability/adot-control-plane.yaml` and the `GpuFaultAdotMemoryHigh` / remote_write alerts in `amp-rules.yaml` in step, then rerun verify. Until then ADOT's own alerts are mute |

`adot_self_metrics` is rerun separately with `gpu-fault-admin status --full`; `control_record_archive_bucket`
runs only in the preflight of `deploy`, so to rerun it rerun `deploy`. `preflight` / `verify` are internal pass-throughs of the release driver,
absent from `--help`, and not open to administrators.

## 5. Joining and Removing GPU Clusters

Giving `deploy` one more ARN is a join: pass all managed GPU ARNs plus the new cluster; `deploy` checks that the
set is a superset of the managed set, releases (or skips the rollout directly when the release is unchanged) and then hands the extra clusters to the same
batch join; giving fewer clusters is refused with a hint to use `remove-cluster`. `join-cluster` performs a join only for the given clusters
and does not trigger a site release (the engine only runs `preflight` / `join-cluster` / `activate-cluster` in per-cluster mode);
to upgrade code at the same time take the `deploy` superset path above. Giving it an already managed ARN is not an error and returns `ALREADY_MANAGED`:

```bash
gpu-fault-admin deploy \
  --state-dir /secure/gpu-fault \
  --gpu-cluster-arn arn:aws:eks:us-west-2:123456789012:cluster/gpu-prod-a \
  --gpu-cluster-arn arn:aws:eks:us-west-2:123456789012:cluster/gpu-prod-b
# Join only, without releasing; already managed ARNs return ALREADY_MANAGED
gpu-fault-admin join-cluster \
  --state-dir /secure/gpu-fault \
  --gpu-cluster-arn \
    arn:aws:eks:us-west-2:123456789012:cluster/gpu-prod-b \
  --gpu-cluster-arn \
    arn:aws:eks:us-west-2:123456789012:cluster/gpu-prod-c
```

The command auto-discovers the EKS/HyperPod identity and validates same account, same Region and `NodeRecovery=None`, then updates
the separate kubeconfig, creates the cluster token, Executor IAM/OIDC, NLB NAT EIP whitelist,
Route53 VPC association and node Action keys. It performs the low-level join with the candidate site, then
verifies at cluster scope; only after everything passes does it atomically update the formal `site.yaml` and write the EKS, HyperPod, Executor IAM and DNS
association into the Aurora registry. After a failed join is rolled back automatically, `phase` in `join-cluster/<arn-hash>/state.json`
(`<arn-hash>` is the first 12 characters of the sha256 of the ARN as given, see below)
is `ROLLED_BACK`, and the `failure` field records the reason (`error`) and the last completed step
(`after_step`); after the fix, rerunning with the same parameters starts a new attempt.

The join's commit path pays only for the new cluster, and the `evidence` in `state.json` shows the scope of every step:

- `COLLECTORS_READY`: waits only for the "fast" collectors (configured unconditional reporting period <= 60 s, currently
  `GPU_INVENTORY`, `HOST_TELEMETRY`) to report once on every node, and requires the Agent row to be alive
  (`/v1/fleet/readiness`: lease not expired, identity is the current release); kinds with a longer period
  (the 300 s default health summaries: `GPU_METRICS`, `FABRIC_MANAGER_LOG`, `NODE_LOGS`,
  `NVIDIA_KERNEL`) only have their systemd unit checked as active, recorded as `verified_as: scheduled`,
  with the first report confirmed by the later verify/`status` path. The fast/slow split comes from the period
  variables written by the collector installation script (`GPU_FAULT_*_HEALTH_SUMMARY_SECONDS`, `GPU_FAULT_GPU_INVENTORY_INTERVAL_SECONDS`,
  `GPU_FAULT_HOST_INTERVAL_SECONDS`), not a hard-coded list. Evidence fields: `waited_kinds`,
  `deferred_kinds` (each kind's period, scheduled/reported node counts), `fleet_ready`. A fast kind
  not reporting within 600 s is still a hard failure.
- `VERIFIED`: the candidate verify runs only the checks that read the new cluster (`regional_contexts`, `cpu_secrets`,
  `cpu_workloads`, `runtime_profile`, `read_only_verifiers`, `runtime_component_identity`,
  `control_api` and each cluster's `gpu_cluster:*`), skipping six site-level checks unrelated to the new cluster
  (`monitoring`, `adot_self_metrics`, `aurora`, `aurora_credential_refresh`,
  `email_notifications`, `nlb_runtime` -- they read only control-plane/AWS state, and join does not modify those objects;
  they are still covered by `gpu-fault-admin status --full`). `control_api` judges collector readiness for the new cluster by the fast/slow rule above;
  `evidence.VERIFIED.verify` records the checks run, the checks skipped and the relaxed
  kinds (`relaxed_collector_readiness`).
- `RELEASE_STATE_UPDATED`: `sync-state` re-captures only the new cluster's snapshot (`capture_scope:
  cluster:<id>`), merges it with the committed release state and still validates the site-level invariants -- each cluster's live
  Agent set equals its HyperPod node set, and the CPU/new cluster/recorded runtime images agree.
- `REGISTRY_UPDATED`: writes only the resource rows added by this join (`delta_keys`, i.e. EKS, HyperPod,
  Executor role/OIDC, ADOT writer role, VPC association) into Aurora, with a consistency check against
  the snapshot taken at `DISCOVERED`; `FINAL_VERIFIED` still reads the online registry to confirm these keys are
  ACTIVE.

Without registered Node key custody, when the first deploy is given several GPU ARNs at once, the first cluster forms a stable baseline first
and the remaining clusters automatically use batch
join: read-only discovery and each cluster's prerequisite resources (IAM, network whitelist, node keys) are prepared in parallel, and the whole batch shares
one candidate preflight and one final verify; a separate baseline verify of the existing site is no longer done, since the candidate verify covers
all clusters. How many clusters roll at once is decided by the release engine's `spec.release.upgradeMaxParallelClusters`
(default 1, maximum 8); batch join reads that value directly and no longer has its own concurrency or node budget.
Each target still goes through `PENDING -> ACTIVE` independently; a cluster failing enters `FAILED/ROLLED_BACK` without
undoing the successful clusters of the same batch. Each cluster's record is `join-cluster/<arn-hash>/state.json`, and the batch
summary (joined/already_managed/failed with each cluster's record path) is printed directly in the command result, and on partial failure
also output together with the error message. When several join CLIs are started concurrently, only one operation on the same site gets the site lock,
and the rest fail immediately reporting the lock holder's pid and command, never waiting silently.
The first multi-ARN target is written into the bootstrap state. When rerunning the original command after a mid-way failure, the already successful members may be a subset of the original target,
and the join continues only for the missing members; a change of the original target identity, order or CPU anchor fails closed.

Only after the original targets are all managed, or the missing targets have a completed removal record with full identity matching the current site/release,
does the first-target checkpoint stop blocking normal upgrades and expansions. A partial site cannot end the multi-target commitment early;
a same-named removal record alone does not release it. An empty-cluster upgrade after removing the last GPU cluster still keeps the CPU identity check,
which cannot be bypassed by deleting the checkpoint.

Sites with explicitly registered Node key custody switch to serial per-cluster join; when authorised preparation or evidence blocks, later targets stop.
A custody pause keeps the already prepared namespace and original inputs and does not enter the automatic compensation branch of ordinary failures,
so as not to change UIDs pending or already authorised. Registration, independent approval and resume of the original command are in [§6.2](#62-node-key-custody).

The first infrastructure phase creates only the namespace, token, IAM, NAT whitelist
and Private Hosted Zone VPC association needed by the CPU and the first baseline GPU cluster; other GPU ARNs get no network permission in advance. Each later join adds only
its own exclusive resources. When several clusters share the same VPC it is associated only once, and removing one of them does not disassociate a VPC still used by other
clusters.

State lives in `join-cluster/<arn-hash>/state.json` next to `site.yaml` (`<arn-hash>` = the first 12 characters of the sha256 of the ARN
as given, not the internal cluster id; the same cluster written as EKS ARN and as HyperPod ARN lands in
different directories). A failure before the activation intent
(including the phase where the site is written but not yet activated), when `autoRollback=true` and both identity and command stop can be proven, compensates this round's
data-plane, membership, IAM and network changes and publishes FAILED/ROLLED_BACK; the commit phase merges site,
release state and Aurora registry incrementally inside the membership lock without overwriting other members.
Ordinary failures of single-cluster and batch join both pass the original exception explicitly from the caller, writing a redacted, length-capped reason into
`failure.error` of `state.json` before compensation, and recording the last step with proven completion time as
`failure.after_step` and `failure.recorded_at`; after automatic rollback `phase` is `ROLLED_BACK`, and after the fix rerunning with the same parameters
starts a new attempt. Compensation keeps the original failure reason, and the new attempt archives it with the old
state; raw subprocess stderr or credentials are never written directly into that field. Identity, custody and supervision loss are still handled by
their dedicated blocking states and cannot be auto-retried or cross the activation barrier because of the added diagnostics.
The candidate verify evidence is valid for 15 minutes; before commit it checks that the candidate/source site digests and cluster set have not drifted, drift of release state and
registry is caught by comparing runtime snapshots before and after verify, and `FINAL_VERIFIED` reads the online registry once more to re-check. Only after these
compensable commits does it enter `ACTIVATION_STARTED -> ACTIVATED -> FINAL_VERIFIED` (the cluster's lifecycle in the registry is
`PENDING -> ACTIVE`); a failure after that point keeps the deployed resources and requires
rerunning the same command fail-forward, with no further rollback. The success criterion is that the registry, connection
Secret, Executor, Collector, Reconciler, all current Agents and the Aurora resource keys are ready at the same time; when the failure-domain mapping digest
changes, join also waits for the control-worker rollout to complete (`rollout status`, <=600 s) before returning COMPLETED,
and remove-cluster likewise.

| Join phase | Why it is needed | Reusable or parallel scope |
|---|---|---|
| Discovery and old-attempt recovery | Prevents joining the CPU, rebinding HyperPod or applying old cleanup to a new namespace | Read-only discovery at most 4-way, the batch shares registry/network reads; a new attempt cannot start while old compensation is unfinished |
| Credentials, namespace and prerequisite resources | Establishes the target's independent token/keys, IAM and network access | kubeconfig writes serial; independent prerequisite tasks may run in parallel, the ADOT role waits for Executor/OIDC to complete |
| Candidate preflight and rollout | Proves the current signed release is installable, with waves still controlled by the dynamic node safety gate | Shares the candidate preflight; cluster/node budgets are uniformly limited by the release |
| Candidate verify | Verifies all old and new members and the complete runtime identity | Shares one verify; on 15-minute expiry one more round may be verified for the remaining targets, not merely refreshing the timestamp |
| Commit, activation and terminal re-verification | Guarantees the management site, release, Aurora and ACTIVE membership agree | Commits serially per cluster, stopping on other member/release drift; after the activation intent only fail-forward |

A rerun of an unfinished transaction re-checks the recorded GPU identity. `ROLLBACK_STARTED/ROLLBACK_FAILED` first complete the original compensation,
`BLOCKED_IDENTITY` must first resolve the identity uncertainty, and `SUPERVISION_LOST` cannot be retried automatically.
A token still referenced by the rollback registry must stay in the controlled path; do not delete credentials by hand to let a new attempt regenerate them.
Once `ACTIVATION_STARTED` is recorded, even if the original verify has expired, activation and final re-verification complete only per the original binding,
without returning to `PENDING` verification and cleaning up already enabled resources.
A completed member still in the site may return idempotently; that is not a fresh live health proof, and the current state needs an independent check with `status --full`.

Before joining, NodeNames conflicting with existing members, Agents, CPU keys or same-batch targets are refused, and the target Node UIDs
are passed into the key preparation flow. Existing legitimate randomly rotated keys are kept as is; compensation uses the digest of the actual GPU key bytes originally bound,
first re-verifying the Node UID and the CPU's current digest, then CAS-deleting this attempt's CPU entry. Key writing not started,
missing ownership proof or a changed digest cannot revoke an existing key by name.

Before removal you must confirm the target cluster has no active workflows or remote commands. The normal entry point is:

```bash
gpu-fault-admin remove-cluster \
  --state-dir /secure/gpu-fault \
  --gpu-cluster-arn \
    arn:aws:eks:us-west-2:123456789012:cluster/gpu-prod-b \
  --confirm REMOVE_GPU_CLUSTER
```

`--gpu-cluster-arn` uses the same ARN as `deploy` and `join-cluster` (either the GPU EKS or the HyperPod
ARN); the command maps it to the internal cluster id in the site itself, and the administrator need not know that id. When the ARN
is not a current member, only a persistent record exactly matching the same removal attempt whose site commit has started allows resume;
other cases refuse, without guessing the target by name or ARN suffix.

The command completes the stop of the target GPU cluster's data sources, the queue drain and the node systemd uninstall via the state machine.
join/remove use the same site transaction lock. Disassociating an exclusive Route53 VPC association saves the ChangeInfo, waits for
`INSYNC` and confirms the association has really disappeared from the hosted zone; when other remaining GPU clusters still use the same VPC the association is kept.
Disassociation also requires a `CREATED/DETACH` resource registration with full zone/Region/VPC identity; CPU-native associations and
external/reused associations are kept. Old numeric or cluster-form keys are not renamed on membership changes; read failures, unknown origin or
duplicate physical records must be reconciled first, and "the association exists" cannot be taken as this site's right to delete.
Roles/RoleBindings rendered by the engine into workload namespaces (including system namespaces and kube-system)
enter the original per-context cleanup snapshot only after the renderer policy, namespace/SA UID and full foreign-binding checks pass;
`gpu-fault.io/workload-namespace-rbac=true` by itself authorises no deletion or orphan exemption.
remove-cluster and site-wide uninstall share this read-only inspection and the subsequent checked deletion flow;
RBAC is deleted only after the drain, GPU Executor and the other required stop phases complete, and before its
SA/Deployment/namespace anchors are cleaned. RoleBindings precede Roles, writes carry the original UID and current
resourceVersion, and progress is recorded in the same journal only after disappearance is confirmed. Manual resources and foreign bindings are not
pulled into the deletion scope for carrying the same label; unknown ownership must stop.
After the Agent/Executor stop, namespace deletion is requested and its disappearance confirmed, and only then are CPU registry/key
cleanup and IAM/network dependency-wave cleanup run in parallel. Finally the target's absence, the registry, EKS/HyperPod preservation and the remaining
site's health are verified in parallel. The CPU control plane, the other GPU clusters and the target GPU EKS/HyperPod are all kept.
Components in the namespace may still use IAM, network or keys while terminating, so the namespace's absence must be confirmed first,
and these dependencies must not be revoked early to reduce waiting.
Namespace deletion carries the original UID as a precondition; a same-named rebuild during deletion, a read failure or an invalid response all stop.
Node Installer annotation cleanup first reads and checks the complete Node UID set once, then protects every patch with UID/resourceVersion.
The at most 8-way metadata cleanup only optimises API waits and does not widen the node simultaneous-unavailability or reinstall budget.

| Removal phase | Why it is needed | Parallel or resumable scope |
|---|---|---|
| Discovery and binding | Checks the ARN, EKS endpoint/creation identity, namespace/Node UIDs, token digest, registry and shared dependencies | Network discovery deduplicated, at most 4-way; each resume re-reads, identity drift stops immediately |
| `DRAINING`, stop and drain | Blocks new command claims, keeps the ability to finish in-flight work, then stops Executor and Node Runtime | The `DRAINING` ACK is not a drain proof; RBAC, token, keys or IAM must not be revoked early |
| Namespace deletion complete | Deletes by recorded UID, proving the target Pods and their dependencies no longer run | Must wait for real absence and refuse same-named rebuilds; cannot overlap with later credential/AWS revocation |
| registry/key and AWS revocation | Handles only the target's exclusive resources; VPCs or NAT shared with the CPU/other GPUs are kept | Two branches in parallel, AWS reverse-dependency waves at most 4-way; per-EIP confirmation, Route53 waits for `INSYNC` and real absence |
| Aurora/site/release commit | Synchronises state incrementally, removes the local member and token, refreshes the failure-domain mapping | Serial commit, recording `SITE_COMMIT_STARTED` before writing the site |
| Final re-verification | Proves the removal target is absent and the clusters to be kept are still healthy | Read-only checks may run in parallel; any failure writes no `COMPLETED` |

The last GPU cluster may be removed; the CPU control plane keeps running with an explicit empty registry `[]`, for
re-joining later. After the last cluster is removed, the release state sync (`sync-state`) has no GPU cluster from which to read the Node Installer image,
and uses the image recorded in the release state (otherwise the configured value), no longer judging the empty set as "inconsistent". The Runtime Profile registration anchor is a stable identifier and is not rewritten because the current GPU set is empty.
A regular `deploy --state-dir` upgrade after removal follows `site.yaml`: the "initial deployment targets" checkpoint the first deployment writes into `bootstrap-state.json`
constrains the target clusters only while the site document does not yet exist (resuming an interrupted first deployment); once the site already manages the target,
or a cluster in the target has been removed by `remove-cluster`, the checkpoint automatically settles to COMPLETE and no longer refuses
empty-cluster or add-cluster upgrades against the first deployment's cluster set.

State and evidence live in
`remove-cluster/<cluster-id>/` next to `site.yaml` (cluster-id is the internal id the site records for that ARN), with each attempt
using a separate evidence directory. After a partial site commit, still rerun with the original full ARN and confirmation parameter; an old v1 journal lacking
the bindings currently required by v2 needs explicit reconciliation, without deleting the directory. When a removed cluster is later re-joined, a new attempt must be created;
the old `COMPLETED` cannot be reused to skip cleanup; merely rerunning an old completed record also refuses a new namespace or node identity drift.

### 5.1 Node Failure-Domain Mapping (Automatic)

The remediation concurrency budget `GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_FAILURE_DOMAIN` (fixed at 1 in production)
takes effect only for failure domains the Executor knows. One GPU cluster is one AZ, so the zone label cannot distinguish any two nodes in the same
cluster; the meaningful failure domain is the HyperPod instance group, rack/PDU or network leaf.
The control plane itself has no node topology; the mapping ConfigMap `gpu-fault-failure-domain-map` is rendered, validated, applied and rolled by the product from the node labels of each GPU
cluster, and **administrators need to run no command**:

- `deploy` (upgrade and bootstrap) reads the nodes of every managed cluster inside the control-plane apply step
  (sharing the same `get nodes` as the rolling upgrade), renders the mapping, first validates it with the same loader the worker uses at startup
  (an invalid mapping is refused before apply and never reaches the worker), then applies the ConfigMap and writes the content digest
  into the control-worker Pod template annotation `gpu-fault.io/failure-domain-map-sha256`: the worker rolls only when the mapping
  changed, stays put when not, with no extra `rollout restart`.
- `join-cluster` / `remove-cluster` likewise re-render and apply after the membership is settled; after nodes are added or removed
  the next `deploy` refreshes automatically. Before refreshing, the control-worker must be readable; a query failure or missing worker
  stops the managed command and is not treated as "not yet installed" and skipped. An unchanged digest is not re-patched; when an update is needed it carries
  UID/resourceVersion guards, and both commands wait for the control-worker `rollout status` to complete (<=600 s) before returning.
  **There is no step to run manually after a node change.**
- The default label priority is `sagemaker.amazonaws.com/instance-group-name` ->
  `topology.kubernetes.io/zone` -> `failure-domain.beta.kubernetes.io/zone`. When the fleet carries EC2
  topology labels, declare them in `spec.failureDomainLabels` of `site.yaml` with "the finest domain first" (which layer
  is the rack depends on the instance family, hence no default), for example:

  ```yaml
  spec:
    failureDomainLabels:
      - topology.k8s.aws/network-node-layer-3
      - sagemaker.amazonaws.com/instance-group-name
  ```

  The field is persisted with the site and read on every render; the per-failure-domain cap of the rolling upgrade (`upgradeMaxUnavailable`
  related) reads the same field and the same function, so the two definitions of "failure domain" cannot diverge.
- Nodes without any of the selected labels do not enter the mapping: they occupy no `domain:` quota, are bound only by the node/cluster/region
  quotas, and do not block remediation. The rolling-upgrade side records the same batch of nodes as `UNKNOWN` topology and tightens each wave to 1 node.
- The control-worker's `GPU_FAULT_REMEDIATION_FAILURE_DOMAIN_MAP` takes its value from the ConfigMap's `map-path`
  key and is mounted `optional`: a site without labels gets an empty mapping, with the same effect as no ConfigMap
  (no per-failure-domain throttling); a mapping with invalid content is never applied. The role-split verifier at the end of deploy
  resolves this item with kubelet's `optional` semantics.
- To see what the product would render, use the read-only debug command
  `gpu-fault-admin failure-domain-map --state-dir "$STATE_DIR" [--output PATH]`:
  it prints each cluster's mapping summary and `unmapped` (nodes without any selected label); with `--output` it also writes out
  the ConfigMap manifest and prints its content digest `sha256`. It applies nothing.

## 6. Credential and Certificate Lifecycle

| Object | Current behaviour | Administrator requirement |
|---|---|---|
| Aurora master password | Rotated by Secrets Manager, synchronised to Kubernetes by the CronJob, running Pods re-read through the mounted file (no rollout) | Watch the age of `last-refresh-status.json` and `GpuFaultAuroraCredentialRefreshStale/Failing`; after a rotation check the three roles' mounted file digests updated and no authentication failure logs |
| cluster token | One current token per cluster; during rotation one additional retiring token with a bounded expiry may be accepted | Use `gpu-fault-admin rotate-token` to complete the registry overlap window, GPU Secret, data plane and node Agent switch in one go (§6.1, REG-9) |
| Node Action key | Node-level key version 2; existing legitimate randomly rotated values are kept | custody must be explicitly registered in advance and independently authorised; the deployment and witness after a single-node rotation cannot be replaced by a Secret snapshot or reinstall (§6.2) |
| Node Agent TLS | A certificate is generated per node at installation and pinned by the signed heartbeat | A certificate change raises the Agent generation; rotate via a formal node rollout, never fall back to HTTP |
| NLB server certificate | The certificate self-issued by bootstrap and **imported** into ACM (issued by a private root CA: server certificate valid 825 days, root CA 3650 days, nodes hold the CA public key). ACM does not auto-renew imported certificates; `status` alerts ahead of time on this 825-day server certificate per `certificateMinValidityDays` (default 30) | Before expiry re-issue and import the certificate, update the data-plane trust chain and switch via a formal node rollout; there is no `renew-certificate` verb and none is planned; `curl -k` is forbidden |
| execution token | Exists only on the CPU control plane | Must not be distributed to the GPU data plane |
| Notification email channel | Default `channel: sns`: the administrator address is the only confirmed subscription on the site SNS topic, shared with AMP alerts, and the control-plane role holds only `sns:Publish`; `channel: ses` sites additionally have a verified SES sending identity | When the subscription lapses, the monitoring check of `status` gives WARN, and rerunning `deploy` resends the confirmation email; switching from SES to SNS only needs `spec.notifications.channel` in `site.yaml` changed to `sns` and a `deploy` rerun, and the old SES identity is deleted by `uninstall` per the registry |
- `GPU_FAULT_OPERATOR_ACKNOWLEDGEMENT_TIMEOUT_SECONDS` (default 86400): the wait cap of steps awaiting human confirmation
  (currently only `CHECK_MECHANICALS`), and at the same time the floor of the execution deadline and lifetime of workflows containing that step.
  Sized as "one working day": a mechanical inspection naturally waits hours, and the 600-second generic step cap would keep COLLECT-009 from ever passing;
  it is not infinite so that workflows eventually close. Tighten it to shorten the wait; it must not be below the generic step cap.

Node key synchronisation requires the GPU set to exactly match the currently verified nodes, while the CPU keeps the site-wide key union; the probe reads the
Secret privately and compares the actual byte digests, not treating a complete key set as success. Partial rotation resumes using the
`gpu-fault.io/node-action-key-rotation` marker on the existing GPU Secret, and even when both digests are the same it completes only after confirming the Node UID,
the CPU mirror and the marker's removal. Read failures, same-named rebuilds or unknown key conflicts must stop; do not hand-delete
the marker, overwrite the whole CPU Secret or re-derive existing random keys to clear the error; the original Secret must not be output as evidence.

### 6.1 Cluster Token Rotation with an Overlap Window

Each GPU cluster in the registry may declare both the current token and one **retiring** token at the same time:
`token_sha256` is the new credential, effective immediately; `retiring_token_sha256` is the old credential, accepted only until
`token_rotation_expires_at`. Both must appear together or be absent together, the digests must differ,
the expiry must carry a timezone and must not exceed
`gpu_fault.regional.MAX_TOKEN_ROTATION_WINDOW` (7 days) relative to `updated_at`. After the window expires only
`token_sha256` is accepted, so forgetting the finalisation does not disconnect Executors that have already switched; it only makes callers that have not yet switched
fail closed. The authoritative runtime registry is the Aurora revision; the bootstrap/disaster-recovery
`GPU_FAULT_REGIONAL_CLUSTERS_JSON` takes effect only while there is no durable revision.

The only formal entry point for rotation (administrators no longer compute digests, publish revisions or roll the data plane by hand):

```bash
gpu-fault-admin rotate-token \
  --state-dir /secure/gpu-fault \
  --gpu-cluster-arn arn:aws:eks:us-west-2:123456789012:cluster/gpu-prod-a \
  --reference CHG-12345
```

1. **Applicability**: the target cluster has no `PENDING`/`LEASED`/`WAITING` remote commands and no unfinished
   release transaction; fault handling need not be stopped.
2. **Scope of impact**: only the target GPU cluster's registry entry, connection Secret, the three Deployments Executor /
   Completion Watcher / Resource Collector, and all node Agents (reinstalled wave by wave via the same fleet wave as a
   release).
3. **Read-only checks**: `gpu-fault-admin status` healthy; a first rotation has no conflicting `IN_PROGRESS` transaction,
   and interrupted recovery of the same rotation must reuse `<state-dir>/rotate-token/<cluster-id>/state.json`,
   which must not be deleted first.
4. **Command**: as above. The command executes in order: publish the overlap
   revision carrying `retiring_token_sha256` -> update the GPU connection Secret -> roll the data plane -> reinstall nodes by wave -> wait for the control-plane
   logs to pass the current retention check with no
   `regional cluster <id> authenticated with the retiring token` within the `--quiet-seconds` (default 180) window -> atomically write
   `secure/<cluster-id>.token` (the old file kept as `.retired-<time>`) -> publish the revision dropping the retiring
   digest. `--window-minutes` (default 120, set in minutes or hours, not days), `--keep-window`,
   `--rollback` and the state machine and resume rules are in REG-9.
5. **Success criteria**: output `status=COMPLETED`; in `GET /v1/regional/clusters` that cluster has
   `retiring_token_sha256_present=false` (the two digests themselves are always redacted); the token file digest equals the output's
   `new_token_sha256`; `warnings` is empty.
6. **Rollback**: before the `TOKEN_FILE_WRITTEN` start intent is recorded, the same command with `--rollback` can be used
   (Secret -> data plane -> nodes that started or completed the switch -> registry containing only the old digest).
   Once that write intent is on disk, even if the file replacement ACK is lost you can only continue the original rotation without `--rollback` and cannot
   revert; to change the credential again after completion, start a new rotation. Once a rollback has started, keep finishing it with `--rollback`.
7. **Evidence**: `<state-dir>/rotate-token/<cluster-id>/state.json` and the command output JSON, containing digests only.

The rotation does not modify the execution token, nor does it distribute any cluster token to other GPU clusters. Real-machine acceptance is
`GF-REGIONAL-AUTH-016`.

The quiet window must be shorter than `--acceptance-timeout-seconds` (default 1200 seconds); the first wait starts after the last
data-plane mutation, and log reads and polling share the remaining deadline. Before and after reading it checks that all ingress
Pods and their `api` containers stay Ready with consistent UID, container ID, start time and restart count.
Each probe round fixes the window start, first reads the complete prefix of up to 4096 bytes of every current log, requiring the first record to be no later than
the window start; after reading the window it re-checks that prefix digest at the original length. Normal appends do not affect the check; a missing or
changed prefix, unprovable retention, an incomplete log, future timestamps or a timeout all stop and cannot be treated as "no old-token calls".
This relies on the normal log semantics of a trusted Kubelet/CRI and does not prove silent under-reporting, malicious rewrites with the same prefix or offline callers
having abandoned the old token; a local real-kubectl test is not production Kubelet retention acceptance either.

The same transaction pins `--keep-window`, `--window-minutes`, `--quiet-seconds` and
`--acceptance-timeout-seconds` into the journal, and they must not change even when only cleanup continues after commit.
An old in-flight journal lacking the bindings needs explicit reconciliation; defaults must not be filled in or the quiet window shortened to bypass a failure;
before the point of no return `--rollback` may be added while the policy is unchanged.
A failed resume re-checks the site and node set and cannot process new nodes with old waves; the terminal state is written to disk before the temporary token is deleted,
a cleanup failure only continues the cleanup and does not turn an already committed rotation back into a rollback-capable state.
`--keep-window` keeps the retiring digest until expiry and produces a warning, which does not meet the complete retirement success criterion of no warnings by default above.
### 6.2 Node Key Custody

This entry point establishes evidence for controlled key delivery in the future; it is off by default and is not a tool for retroactively signing proof for old installations.
The complete fields and trust boundary are in the [Node Key Custody evidence contract](../components/node-key-custody-evidence.md).

**Applicability**

The first evidence capture must be registered before the target keys are written: the GPU key Secret does not yet exist, and the CPU key map has no entries for these
target nodes. Existing identical derived values, the current Secret, node scans, heartbeats or unsigned digests
cannot prove the custody of a historical installation. A later single-node rotation requires a predecessor chain bound to the same release/site/complete Node UID and
already proven activated by an independent witness; v1 does not automatically migrate old installations or inherit evidence across releases/identities.
During the evidence capture a quiet window for CPU key-map writes must be coordinated; although CAS preserves unrelated keys, another writer changing their digest
fails the completion of the capture, and such changes cannot be attributed to this transaction.

**Scope of impact**

`configure` holds the normal site lock and writes only
`<state-dir>/node-key-custody/registration.json` with mode `0600` on the deployment host; it issues no authorisation, calls no KMS Sign, creates no
namespace and modifies no Secret. The real preparation and delivery still go through the approved deploy/join.
approval, provisioner and witness must use controlled identities that are mutually independent and different from the release signing key;
their KMS signing permissions must not be distributed to GPU Pods, Executors or nodes. The fleet master may still exist only in the CPU Secret and
in controlled files on the trusted deployment host; GPUs receive only node-level keys.

**Read-only checks**

On an existing site first check runtime health with the normal `status --full`, but a health result cannot replace custody evidence.
The independent approver must confirm the SHA-256 pin and public key of the trust document; an ad hoc self-computed digest cannot be taken as approval.
The custody read-only probe verifies the signed completion chain, the current release, the cluster/namespace/Node UIDs and the actual CPU/GPU
key byte digests; with missing evidence it refuses, generating no preparation, signing nothing, and not relying on old key-shape
checkpoints to pass. A completed receipt can be re-verified after the outer bootstrap checkpoint is lost.

**Command**

The selection file is strict JSON with mode `0600` on the deployment host, for example:

```json
{
  "schema_version": 1,
  "trust": "/secure/custody/trust.json",
  "allow_staging": false,
  "clusters": {
    "arn:aws:eks:us-west-2:123456789012:cluster/gpu-prod-a": null
  }
}
```

`clusters` must be non-empty, with keys being the full GPU **EKS ARN**, not the internal cluster ID or the HyperPod ARN;
the value is `null` or a provisioning request path. Paths in the selection file resolve relative to that file;
`allow_staging` may be omitted and defaults to `false`, and does not replace the existing release grade and approval gates.
No environment variable can enable custody implicitly.

```bash
gpu-fault-admin node-key-custody configure \
  --state-dir /secure/gpu-fault \
  --file /secure/custody/selection.json \
  --trust-sha256 <externally-approved-trust-sha256>
```

1. First register the targets with `null`, then run the original formal entry point: the four-parameter `deploy` for a first site build,
   `deploy --state-dir ...` for an existing site, the original `join-cluster --state-dir ... --gpu-cluster-arn ...` for a join.
   Normal resource preparation and the release build may still write to the environment; it pauses only before the key write, outputting
   `node-key-custody/preparations/<sha256>.json` with `authorization=false`.
2. The independent approver reviews the real UIDs, master source digest, release and producer/witness
   source identities in the preparation and provides a fresh signed authorization. Neither the CLI nor the preparation itself can sign on their behalf.
3. Prepare the request per the evidence contract, change the target's `null` to the request path, then run the same `configure`.
   The registration pins the content of the request, authorization, trust/public key, Manifest and predecessor chain, not just paths.
4. Rerun the same deploy/join as is. Only after the inputs and online identity are re-verified is the trusted provisioner
   called to write and sign the completion receipt; there is no separate custody execute or resume verb.

**Success criteria**

`CONFIGURED` only means the registration succeeded. The signed `Completed` of the key task proves controlled delivery and actual read-back,
with `runtime_activation_proved=false`; the administrator probe likewise reports only `runtime_activation=NOT_PROVED`.
After a normal deployment an independent witness must additionally be approved to verify, against the release-bound HTTPS Agent and fresh heartbeats,
that the new key authenticates and the wrong key is rejected. The initial `Activated` is only the precondition for rotation; the complete
`GF-REGIONAL-AUTH-015` is still FAIL; only after the deployment of an authorised single-node rotation, verifying the new key works,
the old key and sibling keys are rejected on both the command and result-query paths, the sibling identities are unchanged, and the independent signature is completed,
is the complete case satisfied. The witness does not rotate Secrets, build host probes, restart or dispatch physical actions.
Witnessing still needs its own live plan and maintenance-window approval, requests an independent KMS signature and writes a public receipt;
it is not offline signature verification or an approval-free operation.

**Rollback and interruption**

A join's `CUSTODY_AWAITING_AUTHORIZATION` or `CUSTODY_BLOCKED` must be handled while keeping the original state, namespace and
inputs; do not delete the directory, rebuild UIDs or hand-edit the phase to continue. An existing `Started` without a valid `Completed`,
an unknown write ACK, changed inputs or identity drift all require explicit reconciliation; a completion proof cannot be rebuilt from the current identical key.
An unstarted request may be explicitly re-registered; a started request can only be continued by a new independently authorised rotation,
and the predecessor must match the signed completion and independent activation chain of the original registered transaction. Registered clusters must not be removed nor the trust pin replaced
to downgrade to the old path, and there is no CLI to automatically cancel unknown writers or revoke a custody transaction.
Reinstalling a node is not a rotation; hand-editing Secrets, deleting rotation markers or re-deriving keys to bypass an evidence failure is forbidden.

**Evidence**

Keep the registration file, the content-addressed preparation, the independent authorization, `<transaction_id>.started.json`,
`<transaction_id>.chain.json` and the public signature chain output by the witness. They contain no raw keys or master;
the receipt directory must be a controlled `0700` directory. The rotation's `retired_key_file` is used only for negative authentication witnessing and stays in
a `0600` file in a separately controlled directory on the trusted deployment host, never in the receipt directory, ordinary artefacts, the release or the GPU,
and its retention and deletion follow the approved rotation process. The signed proof covers only the trusted delivery path and cannot prove the master never appeared at unobserved locations;
local hermetic tests are not LIVE installation, KMS IAM or complete AUTH case acceptance evidence either.

## 7. Troubleshooting by Symptom

| Symptom | First check | Detailed reference |
|---|---|---|
| CPU Pod CrashLoopBackOff | registry, Aurora DSN, startup guard, duplicate configuration | REG-14.1 |
| remote commands stay PENDING | Executor claim, token, CA, cluster ID, owner, protocol | REG-14.2 |
| command LEASED for a long time | Executor Pod, lease renewal, result reporting | REG-14.3 |
| step FAILED with an unclear reason | `status_source` and the executor traceback | REG-14.4 |
| All Agents non-ACTIVE | node token, systemd, artifact/config/profile pins | REG-14.5 |
| operator API returns 403 | Whether the execution token and cluster token are mixed up | REG-14.6 |
| HyperPod outcome unknown | CloudTrail, actual node state, idempotent submission record | REG-14.7 |
| Profile content drift within the same version | Release a new profile version; do not overwrite the original version | REG-8.1.3 |
| Agents not converged after an upgrade | Keep the compatible window, check the Reconciler/Installer, do not force finalize | REG-8.1.4 |
| Executor verifier fails | IRSA, token, CA, owner, artifact and protocol | REG-14 |
| SNS confirmation email arrives repeatedly | Duplicate confirmed/pending on the same Topic, bootstrap pending checkpoint | REG-7 |
| StaleAttemptObservation | Training Pod, Completion Watcher, Observation delivery/outbox | REG-14 |
| workflow FAILED but the incident is `RECOVERED`, reason contains `diagnostic inconclusive` | By design: the workflow contains only forensic/diagnostic/validation steps, failure means "diagnostic inconclusive", the incident is closed, the incident's markers retired, and a `DIAGNOSTIC_INCONCLUSIVE` email sent; the node was not isolated and needs no release, and the job may restart as usual | `详细设计-v2.md` §6.3 |
| All XID plans `BLOCKED NEEDS_OPERATOR` with reason `workload state is UNKNOWN` | The cluster had neither an attempt observation nor a Completion Watcher coverage heartbeat within 600 s (the age of `observed_at` of `get_workload_coverage_heartbeat`, with `watched_pods`/`watched_attempts` at 0); a stopped watcher meaning UNKNOWN is expected; fix the watcher rather than releasing | `部署和运维手册.md` `GPU_FAULT_WORKLOAD_CONTEXT_FRESHNESS_SECONDS` |

The troubleshooting order is fixed as:

```text
release/module digest
  -> identity and TLS
  -> queue/lease
  -> Executor owner and protocol
  -> Node Agent/Collector
  -> provider actual state
```

This section is entered by the symptom the administrator observes. If the entry point is an AMP alert, go straight to the triage card of §8 by alert name;
do not first reverse-look-up the symptom in this table.

## 8. Alert Handling Index

This section is the recipient entry point for every alert in `deploy/observability/amp-rules.yaml`. Each alert's
`annotations.runbook_url` points at the same-named sub-section of this section, and `scripts/verify-regional-alerting.py`
checks consistency in both directions: an alert missing `runbook_url`, a `runbook_url` not matching the alert name, a pointed-to sub-section that does not exist,
or a sub-section here with no alert pointing to it all fail acceptance.

Sub-section headings use the alert name directly rather than `8.1` numbering, because the anchor of `runbook_url` must be mechanically
derivable from the alert name (lower-case, non-alphanumerics removed), and numbering would drift when alerts are inserted.

This section is three-line triage cards, not the seven-item runbooks of §2: each card gives only "meaning / first check / boundary"; when an action
is needed, jump to the formal runbook the card references. No alert may be cleared by raising a threshold, deleting Store records or
silencing the rule.

### 8.0 Dashboards

`deploy/observability/dashboards/*.json` are 8 Grafana dashboards: 1 overview plus 1 for each of the 7 alert groups of
`amp-rules.yaml`. They are generated by `scripts/build-grafana-dashboards.py`
from the panel table in `scripts/grafana_dashboard_catalog.py`, and `make grafana-dashboards-check`
guarantees the checked-in JSON is byte-for-byte identical to the generated result. The generator reads `amp-rules.yaml`: whenever an alert expression ends in
`> N` / `>= N` / `< N`, the panel drawing that metric gets a threshold line of the same value (critical red, warning
orange); each panel's description lists the alert names it covers, their `for` durations and the anchors of the cards in this section, so it is one step from a panel to
a triage card. Each of the 83 alerts lands on at least one panel (`tests/test_grafana_dashboards.py`
iterates over all rules asserting this, and asserts the thresholds too); a threshold changed in the rules without regenerating turns the gate red.

Import path: the Grafana provisioning step of `gpu-fault-admin deploy` imports them automatically at deployment into
the "GPU Fault Recovery" folder and binds the Prometheus (AMP) data source with uid `gpu-fault-amp`;
the manual fallback is Grafana -> Dashboards -> New -> Import -> upload the corresponding JSON -> choose the AMP data source.
Every dashboard has `control_plane_cluster` / `region` variables at the top, and those drawing per-cluster metrics also have
a `cluster_id` multi-select variable (default All).

**Login and authorisation.** The deployment-side import uses a short-lived token of the workspace service account and depends on nobody's login permission;
but a human opening the dashboards still goes through the workspace's own authentication (Amazon Managed Grafana supports only IAM Identity Center
or SAML, with no local username/password). The workspace id and address are in the bootstrap state's
`resources.monitoring_install.grafana.dashboards_url` (of the form
`https://<workspace-id>.grafana-workspace.<cpu-region>.amazonaws.com/dashboards/f/gpu-fault-recovery`).
Before opening for the first time, your own Identity Center user needs a role on the workspace. deploy does this automatically after the import:
it finds the user in IAM Identity Center by the site administrator email (`emails.value`, falling back to `userName`) and grants
ADMIN, skipping when ADMIN is already held; on success stderr prints `Grafana ADMIN granted to <email> (Identity Center
user <id>)`, and the result is in the bootstrap state's `resources.monitoring_install.grafana.admin_grant`; the administrator
can then add VIEWER for on-call colleagues in the Grafana UI (Administration -> Users). When the user cannot be derived (no or several
Identity Center instances in that Region, no user whose email equals the administrator email, the deployment host lacking `sso:ListInstances` /
`identitystore:GetUserId` / `ec2:DescribeRegions` / `grafana:ListPermissions`) the deployment is unaffected; when Identity Center
is enabled in another Region, deploy automatically scans the account's enabled Regions to find it and records the home Region in
`spec.health.identityCenterRegion` of site.yaml (pre-setting that key asks only that Region); otherwise there are two other ways:

1. Pass `gpu-fault-admin deploy … --grafana-viewer <sso-user-id>` directly at deployment, and after the import deploy grants
   VIEWER on your behalf (a wrong user id fails the deployment; correct it and run once more).
2. Without it, deploy prints at the end on stderr the dashboard address, the one command for this site's workspace and a sentence about what is missing; replace
   `<sso-user-id>` with your own Identity Center user id and run it as is (in the CPU cluster's Region); or add the user with the administrator email in
   Identity Center and rerun deploy; the probe retries the derivation on every deployment:

```bash
aws grafana update-permissions --region <cpu-region> --workspace-id <workspace-id> \
  --update-instruction-batch '[{"action":"ADD","role":"VIEWER","users":[{"id":"<sso-user-id>","type":"SSO_USER"}]}]'
```

The user id is on the user detail page of the IAM Identity Center console; the command-line lookup is `aws identitystore list-users
--identity-store-id <store-id> --filters AttributePath=UserName,AttributeValue=<username>`,
with `<store-id>` from `aws sso-admin list-instances` in the home Region where the organisation enabled Identity Center.

Role choice: on-call dashboard viewing uses `VIEWER`; give `EDITOR` / `ADMIN` only to change panels or data sources, and such changes are overwritten by the checked-in JSON at the next
`gpu-fault-admin deploy` re-import -- a lasting change must be made in
`scripts/grafana_dashboard_catalog.py` and regenerated. To revoke, replace `"action":"ADD"` with `"REVOKE"`.
After authorisation log in via the Identity Center portal or by opening the address above directly; `Alerting -> Alert rules` with the AMP data source
shows the live state of the 83 rules (7 groups), corresponding one-to-one to the cards below.

Reading the dashboards is consistent with the cards: only staying above the threshold line (or below it for `< N`-type panels) for longer than the `for` duration
is an alert; the overview's "Alerts firing" panel reads AMP's `ALERTS{alertstate="firing"}` directly,
and the names are the card names below.

| Dashboard (uid) | Alert group | Panel row -> card |
| --- | --- | --- |
| GPU Fault · Overview (`gpu-fault-overview`) | Cross-group closed-loop essentials | Alerts -> all firing cards; Closed loop -> `GpuFaultIncidentsAwaitingOperator`, `GpuFaultOrphanWorkflows`; Processor queue -> `GpuFaultProcessorQueueDepthHigh`, `GpuFaultProcessorQueueDepthCritical`, `GpuFaultProcessorOldestRequestDelayed`, `GpuFaultProcessorClusterBacklogHigh`; Store I/O -> `GpuFaultStoreIoSaturated`; the Remediation budget, Declared topology and workflow/notification outbox panels have no dedicated card and are trend readings |
| GPU Fault · Collector health (`gpu-fault-collector-health`) | `gpu-fault-collector-health` | Collector delivery -> `GpuFaultCollectorSilent`, `GpuFaultCollectorCollectionErrors`; Freshness -> `GpuFaultCollectorMetricsSnapshotStale` |
| GPU Fault · Telemetry pipeline (`gpu-fault-telemetry-pipeline`) | `gpu-fault-telemetry-pipeline` | Remote write -> `GpuFaultTelemetryRemoteWriteFailing`, `GpuFaultTelemetryRemoteWriteQueueSaturated`, `GpuFaultAdotMemoryHigh`; Scrape liveness -> `GpuFaultAdotSelfMetricsMissing`, `GpuFaultDataplaneCollectorMissing`; Completion Watcher -> `GpuFaultCompletionActiveStateUnavailable`, `GpuFaultCompletionOutboxAppendFailures` |
| GPU Fault · Control-plane capacity (`gpu-fault-control-plane-capacity`) | `gpu-fault-control-plane-capacity` | Scrape liveness -> `GpuFaultControlPlaneMetricsMissing`, `GpuFaultControlPlaneReplicaScrapeFailing`, `GpuFaultMetricsContributorFailing`, `GpuFaultProcessorConsumerStalled`; PostgreSQL pool and credentials -> `GpuFaultPostgresPoolCheckoutQueueing`, `GpuFaultPostgresPoolConnectionErrors`, `GpuFaultAuroraCredentialRefreshStale`, `GpuFaultAuroraCredentialRefreshFailing`, `GpuFaultAuroraCredentialRefreshStatusUnknown` (the oversubscription ratio and the per-consumer demand breakdown are readings without a card); Control-loop review counters -> `GpuFaultRemoteCommandStaleFenceSwept`, `GpuFaultNotificationDeliveryErrors`, `GpuFaultControlRecordArchiveErrors`; Processor queue -> `GpuFaultProcessorQueueDepthHigh`, `GpuFaultProcessorQueueDepthCritical`, `GpuFaultProcessorOldestRequestDelayed`, `GpuFaultProcessorClusterBacklogHigh`; Metric scan truncation -> `GpuFaultWorkflowMetricScanTruncated`, `GpuFaultAttemptObservationMetricScanTruncated`; Admission and Store I/O -> `GpuFaultProcessorAdmissionRejected`, `GpuFaultStoreIoSaturated`, `GpuFaultStoreIoRejected`, `GpuFaultStoreIoBackendUnavailable`; Latency, leases and counters -> `GpuFaultProcessorP95Slow`, `GpuFaultProcessorExpiredLeasesReclaimed`, `GpuFaultProcessorCounterDrift`, `GpuFaultProcessorCounterDriftScanUnavailable`; Telemetry spool -> `GpuFaultTelemetrySpoolListenerDown`, `GpuFaultTelemetrySpoolDataLoss`, `GpuFaultTelemetrySpoolAdmissionRejected`, `GpuFaultTelemetrySpoolReplayError`, `GpuFaultTelemetrySpoolDrainDelayed`; Loop liveness -> `GpuFaultWorkflowDispatcherStalled`, `GpuFaultPeriodicRunnerStalled`, `GpuFaultProcessorUnhealthy`, `GpuFaultProcessorNoActiveConsumer`, `GpuFaultTelemetrySpoolConsumerStalled`, `GpuFaultTelemetrySpoolConsumerStopped`; Processor and periodic errors -> `GpuFaultProcessorLeaseRenewalFailing`, `GpuFaultProcessorFaultEventsRejected`, `GpuFaultPeriodicServiceErrors`, `GpuFaultMetricsAggregationIncomplete` |
| GPU Fault · Remote command (`gpu-fault-remote-command`) | `gpu-fault-remote-command` | Claim latency -> `GpuFaultRemoteCommandUnclaimed`, `GpuFaultRemoteCommandExecutorInternalError`; Command census -> `GpuFaultRemoteCommandUnclaimedExpired` (the per-state/per-cluster command count panels are trend readings) |
| GPU Fault · Policy coverage (`gpu-fault-policy-coverage`) | `gpu-fault-policy-coverage` | Catalog coverage -> `GpuFaultUnknownProductFamily` (CRITICAL findings without an incident is a trend reading) |
| GPU Fault · Orchestration invariants (`gpu-fault-orchestration-invariants`) | `gpu-fault-orchestration-invariants` | Node ownership -> `GpuFaultExclusiveNodeOwnershipInvariantViolation`, `GpuFaultStaleAttemptObservation`; Fleet rollout fence -> `GpuFaultFleetRolloutFenceStuck`; Pointers, agents and registry -> `GpuFaultIncidentDanglingWorkflowPointers`, `GpuFaultStaleAgents`, `GpuFaultFleetPinAheadOfFleet`, `GpuFaultRegionalRegistrySecretDrift` |
| GPU Fault · Recovery outcome (`gpu-fault-recovery-outcome`) | `gpu-fault-recovery-outcome` | Workflows -> `GpuFaultRecoveryBlockedBacklog`, `GpuFaultWorkflowPendingAge`, `GpuFaultWorkflowDeadlineNotEnforced`; Step waiting -> `GpuFaultWorkflowStepStalled`; Counters (30m increase) -> `GpuFaultWorkflowLifetimeExceeded`, `GpuFaultJobWorkflowNodeBusyTimeout`, `GpuFaultProcessorRetryHorizonFailures`, `GpuFaultBranchEscalationBudgetRefused`; Closed loop and notifications -> `GpuFaultClosedLoopSlow`, `GpuFaultClosedLoopWindowIncomplete`, `GpuFaultNotificationDeliveryFailing`; Remediation budget -> `GpuFaultRemediationBudgetStarved`, `GpuFaultRemediationBudgetClusterSaturated`; Completion and incidents -> `GpuFaultCompletionEventsWithoutDecision`, `GpuFaultIncidentsAwaitingOperator`; Dispatcher errors and notification outbox -> `GpuFaultWorkflowDispatchInternalErrors`, `GpuFaultWorkflowFailureHandlingAbandoned`, `GpuFaultNotificationUndeliveredTooLong`, `GpuFaultNotificationOutboxNotDraining`, `GpuFaultNotificationExpiredUnsent` |

The last row of the overview, "Declared topology", is `gpu_fault_capacity_largest_cluster_node_count` /
`gpu_fault_capacity_managed_node_count`: the topology declared by the release, from which the fault reservation and the Aurora capacity floor are derived;
a reading of 0 means that release declared none. Every metric family on the dashboards must be in the keep list of
`adot-control-plane.yaml` (asserted by `tests/test_grafana_dashboards.py`), otherwise the panel stays empty forever.

### GpuFaultCollectorSilent

- **Meaning**: some collector channel of some GPU node has not successfully delivered a batch beyond the silence threshold; that node's
  fault signals are untrustworthy at this moment, and the policy is blind to it.
- **First check**: node systemd service state, Node Agent TLS and the private CA trust chain, cluster token,
  collector outbox backlog, timestamp of the last successful batch. To look at the outbox, prefer not logging into the node:
  `gpu-fault-admin collector-outbox --state-dir "$STATE_DIR" --cluster-id <cluster> --node <node>
  --collector kernel --action stats|list --reference CHG-xxxx`, and for requeue add `--action requeue-dead --yes`
  (§9.3; it sends the same command to the Node Agent as a `COLLECTOR_OUTBOX_MAINTENANCE` single-step workflow, returns only metadata,
  and takes the lock strictly). When the Node Agent predates that operation, or the collector has stopped and left a `.lock`, fall back to
  `gpu-fault-collector outbox --collector kernel stats|list|requeue-dead --yes` on the node (`list`/`stats` show only
  metadata and print no payload). Next to the outbox file is a same-named `.lock`: `requeue-dead` tries to take the lock non-blockingly, retrying at most 10 times,
  about 5 s in total, and on failure errors out giving the `.lock` path without hanging; the error and the WARNING of `--force` attach the holder
  recorded in the lock file: `recorded holder pid <n> (collector|cli:<subcommand>), alive, since <time>`,
  `stale recorded holder pid <n> (<role>, gone)` (the recorded process is gone, meaning the real holder wrote no identity line)
  or `holder unknown`; it is a hint written by the lock taker, not the lock itself, and pid reuse or the window between release and rewrite can let it
  lag behind reality; when unsure verify with `lsof <outbox>.lock`; `--force` likewise makes this bounded round of attempts first, and on failure rewrites without the lock with a
  WARNING -- if the collector is still running at that point, records it appends between the CLI's read and replace may be lost, so
  `--force` is used only when the collector has stopped, and for that reason **exists only in the node-local CLI**: the remote path cannot see whether the collector has stopped,
  always takes the lock strictly, and on failure returns the recorded holder as the failure reason; dead letters refused with 413 keep only a digest and the first 4 KB,
  cannot be requeued, and need re-collection at the source. `unlocked_writes_total` in `stats` is a counter inside the collector
  process, always 0 when read by the CLI; for lock unavailability look at the `outbox lock is unavailable` warning in the collector log. For "is it delivering, how often" see
  `GET /v1/collector-status/{cluster_id}`: `collector_status` is updated on every accepted batch,
  while raw evidence is written only when a batch carries a finding, a collection error or a reason other than `health-summary` -- a healthy
  node having 0 `GPU_METRICS` evidence records is by design, not a dead collector; "what the collector saw" is what
  evidence is for.
- **Boundary**: do not verify connectivity with `curl -k`, and do not raise the silence threshold to clear the alert. Repair entry points REG-6,
  REG-14.5.

### GpuFaultCollectorCollectionErrors

- **Meaning**: on some collector channel a node is "delivering on time, but what it delivers is failures" --
  `last_error_at` is not earlier than `last_success_at`. This and `GpuFaultCollectorSilent` are two
  different states: such a node has normal connectivity and batches are not late, so the silence alert does not see it until the last success ages past the silence threshold
  (a quarter of an hour for node-logs), while the node is in fact already losing data.
- **First check**: read the channel's `errors` field in `GET /v1/collector-status/{cluster_id}`,
  which is the reason the node itself wrote (for example a journalctl exit code, training log rotation, an abandoned time window); then look at
  the collector's journal on the node.
- **Boundary**: the only way this alert self-heals is the node really delivering successfully once -- do not change thresholds, do not delete
  `collector_status` records. Repair entry points REG-6, REG-14.5.

### GpuFaultCollectorMetricsSnapshotStale

- **Meaning**: the collector silence snapshot in `/metrics` has not refreshed for more than 120 seconds, meaning
  `GpuFaultCollectorSilent` itself has lost its data source.
- **First check**: whether the scrape target points at the ingress replicas rather than the workers; connectivity from ingress to the shared Store
  and the Aurora credential refresh CronJob.
- **Boundary**: do not judge the silence of `GpuFaultCollectorSilent` as "recovered" before the snapshot recovers.
  See REG-14.1 for details.

### GpuFaultTelemetryRemoteWriteFailing

- **Meaning**: remote_write from ADOT to AMP is dropping points. Batches that have not succeeded after the 120s retry window are
  discarded outright and never resent, so the data in AMP has holes: all the other alerts of this section evaluate on those holes,
  and "the graphs look clean" does not mean "the fleet is fine".
- **First check**: `Exporting failed. Dropping data.` and its `error` in
  `kubectl -n gpu-fault-system logs deploy/gpu-fault-adot`; the sigv4 role and the AMP workspace's
  write throttling. The rule does not select by job, so each GPU cluster's data-plane collector `gpu-fault-adot-dataplane`
  is also in scope: first look at the `gpu_cluster` label of the raw series to locate which cluster's collector it is, then go to that GPU
  cluster's `logs deploy/gpu-fault-adot-dataplane`; that side's IRSA role has only `aps:RemoteWrite`,
  and permission-class failures are usually a wrong role trust or workspace ARN. The metric carries the two labels `error_type` / `error_permanent`; splitting by them distinguishes
  rejections (permission/throttling) from transient failures.
- **Boundary**: the counter stopping does not mean recovery -- a collector restart also resets it to zero. Confirm with
  `otelcol_exporter_sent_metric_points` rising again. Do not clear the alert by relaxing
  `max_elapsed_time`: that only trades dropped points for an OOMKill (container limit 256Mi).
- **Note**: these `otelcol_*` metric names have no `_total` suffix; they were measured from `:8889` of the pinned image
  (aws-otel-collector v0.49.0), differing from the suffixed spelling in upstream documentation.
  When changing the image version they must be re-measured, otherwise the alerts silently stop working.

### GpuFaultTelemetryRemoteWriteQueueSaturated

- **Meaning**: ADOT's retry queue is over 80% used; writing to AMP cannot keep up with scraping, and the next wave of backlog
  will be dropped rather than delivered late. This is the only early warning before `GpuFaultTelemetryRemoteWriteFailing`.
- **First check**: the trend of `otelcol_exporter_queue_size` / `otelcol_exporter_queue_capacity`;
  the AMP workspace's ingestion throttling; whether the control-plane replica count was just scaled up, raising the number of time series.
- **Boundary**: do not clear the alert by enlarging `sending_queue` -- the container limit is 256Mi, `GOMEMLIMIT` 220MiB,
  and the queue is paid for with an OOMKill.

### GpuFaultAdotSelfMetricsMissing

- **Meaning**: ADOT's own telemetry (`up{job="gpu-fault-adot-self"}`) is missing from AMP.
  Either the collector is not running or remote_write is failing completely -- and in the latter case
  `GpuFaultTelemetryRemoteWriteFailing` cannot fire, because its counter travels the same path.
- **First check**: first compare with `GpuFaultControlPlaneMetricsMissing`. Both firing: the collector
  or the write path as a whole is down; check Pod status and export error logs. Only this one firing: the self telemetry
  reader (`service.telemetry` bound to 127.0.0.1:8889) or the `gpu-fault-adot-self` keep
  filter has been broken.
- **Boundary**: while this alert exists do not conclude "no fault", for the same reason as
  `GpuFaultControlPlaneMetricsMissing`. It looks only at the control-plane collector: the data-plane collector's own job is deliberately named
  `gpu-fault-adot-dataplane-self` (the same name would let one live collector hide another dead one), and the absence of the data-plane collector is reported by
  the next card, `GpuFaultDataplaneCollectorMissing`.

### GpuFaultDataplaneCollectorMissing

- **Meaning**: the data-plane ADOT collector (one per GPU cluster, `deploy/dataplane/adot-dataplane.yaml`, job
  `gpu-fault-dataplane`) has written no `up == 1` series into AMP for 15 minutes in some `gpu_cluster` that is **expected to have a collector**,
  or in some `gpu_cluster` **all** the targets it scrapes are `up == 0` -- the collector is not running, cannot reach targets, or cannot write to AMP.
  At that point that cluster's `GpuFaultCompletionActiveStateUnavailable` / `GpuFaultCompletionOutboxAppendFailures` and
  the reconciler / executor self-metrics are all blind: a quiet dashboard is not evidence.
- **Which metric it reads**: the same alert name has two sources. The static rule (`amp-rules.yaml`) has only
  `max by (control_plane_cluster, region, gpu_cluster) (up{job="gpu-fault-dataplane"}) == 0` (a live collector cannot scrape
  a single target in some cluster; `max` rather than `min`: one port at `up == 0` does not count, all must be unreachable). The absence half is
  **rendered per cluster** by the release: one
  `absent(up{job="gpu-fault-dataplane", gpu_cluster="<cluster_id>"} == 1)` (labelled with `gpu_cluster`) per cluster configured with `adot_irsa_role_arn`, placed in the separate
  AMP rule namespace `gpu-fault-dataplane-expected` (`regional_dataplane_observability`), rewritten every time the observability digest
  moves; when no cluster has a role configured that namespace is deleted. The former global
  `absent(up{job="gpu-fault-dataplane"} == 1)` has been removed -- it fired forever on a site "with a workspace but no role configured yet",
  sending one SNS message every 4 hours.
- **First check**: first determine whether it was **deliberately skipped** by the release -- the `plan` mode of the release engine
  (`deploy/control-plane/regional/rollout-regional-release.sh plan`; `gpu-fault-admin` has no `plan` verb)
  outputs `dataplane_observability` listing the render result per cluster, and `[]` means skipped; clusters without `adot_irsa_role_arn` (site.yaml `spec.clusters[].adotIrsaRoleArn`)
  or without `health.amp_workspace_id` are skipped by design, and the release prints
  `<cluster>: data-plane ADOT collector not applied (...)`. After adding the role rerun
  `gpu-fault-admin deploy --state-dir STATE_DIR`: the role enters the `observability_adot` digest, this release is just one
  `CONTROL_PLANE_ONLY` round that reruns the AMP installation and applies the collector on every GPU cluster, touching nothing else. With the role configured, go to that GPU cluster and look at
  `kubectl -n gpu-fault-system get deploy gpu-fault-adot-dataplane` and the Pod logs: the IRSA token not obtainable (the role's
  trust must allow `system:serviceaccount:gpu-fault-system:gpu-fault-adot-dataplane`, `aud = sts.amazonaws.com`),
  sigv4 rejected (the policy needs only `aps:RemoteWrite` to that workspace ARN; a wrong ARN is the most common), `/health` probe failing; then check whether the scraped
  Pods still carry the `prometheus.io/scrape` annotation and a port named `metrics`.
- **Boundary**: warning rather than critical -- the watcher, reconciler and executor themselves are still running; what is lost is alert visibility. Repair via
  `gpu-fault-admin deploy`; **do not** `kubectl apply` the raw manifest (placeholders would remain, and the next release overwrites it), **do not** widen
  the collector's Role (it needs only get/list/watch on `pods` in its own namespace, no ClusterRole). With several GPU clusters, one collector dead and
  another alive: the dead one produces no series, the static rule cannot see it, and the **per-cluster rendered `absent` rule** exists precisely for it -- the alert instance's
  `gpu_cluster` label names the cluster. After deliberately removing a cluster's role, rerun `gpu-fault-admin deploy`: the release scales that cluster's collector
  to 0 and re-renders the rules (no longer including it); without the rerun that cluster's collector keeps running with invalid credentials and the rule keeps firing. Automatic rollback restores the collector objects per cluster
  from the snapshot (clusters without a collector before the release have the candidate-created objects deleted) and restores `gpu-fault-dataplane-expected`
  to its pre-release definition; the data-plane half is a separate rollback phase, placed after the control-plane restore (`rollback-cpu-restored`), one cluster's failure
  does not affect the other clusters and the rule namespace, and the phase raises a single error at the end; `path` in the rollback record's `rollback_timing.phases.dataplane_observability_restore.details`
  states whether the `snapshot` path or the `previous-image` fallback of old states was taken, and `clusters.<cluster_id>` records per cluster
  `restored` / `restored-unchanged` (apply all unchanged, not restarted) / `removed` / `scaled-to-zero` / `failed: …`.

### GpuFaultControlPlaneMetricsMissing

- **Meaning**: `up{job="gpu-fault-control-plane"}` is missing; the whole CPU control plane is invisible to monitoring,
  and all the other alerts of this file are disabled at the same time.
- **First check**: ingress Pod status and startup guard logs, Service/target discovery, the
  namespace and port in the scrape configuration.
- **Boundary**: while this alert exists do not conclude "no fault". See REG-14.1 for details.

### GpuFaultProcessorQueueDepthHigh

- **Meaning**: the processor global queue is over 70% of capacity; not yet rejecting, but queueing latency is already growing.
- **First check**: control-worker replica count and lane occupancy, Aurora latency, whether there is a single-cluster fault storm
  (compare `GpuFaultProcessorClusterBacklogHigh`).
- **Boundary**: scaling goes through `gpu-fault-admin config` of [Administrator Capacity Configuration](administrator-capacity-configuration.md);
  do not lower the alert threshold, and do not open the queue bypass.

### GpuFaultMetricsAggregationIncomplete

- **Meaning**: a scrapable control-plane Pod has lacked complete, valid process metrics for 5 consecutive minutes. A live process's shared sample
  missing, malformed, older than 60 seconds, timestamped in the future or a shared directory unavailable all count as unknown coverage.
- **First check**: `gpu_fault_metrics_aggregation_degraded` and `aggregation_processes`, shared
  directory permissions, the publishing thread and process slots. The process count counts only fresh complete samples; a still-existing PID is no substitute.
- **Boundary**: do not read a missing process's counts as zero activity. Restore metric coverage first, then interpret health and event counts;
  do not fake completeness by deleting shared files or restarting processes.

### GpuFaultProcessorQueueDepthCritical

- **Meaning**: the queue is over 90% of capacity, one step from admission rejection.
- **First check**: same as `GpuFaultProcessorQueueDepthHigh`, and confirm whether workers are blocking each other on lanes
  (`gpu_fault_processor_lane_wait_seconds_max`).
- **Boundary**: this is one of the few cases where scaling worker replicas outside a maintenance window is allowed, but it still needs a change ticket and a rollback plan.

### GpuFaultProcessorOldestRequestDelayed

- **Meaning**: the oldest queued request has waited more than 30 seconds. The queue depth may be normal -- this one catches long-tail starvation,
  usually a lane held by a single slow request.
- **First check**: `gpu_fault_processor_phase_oldest_seconds` to locate the phase, whether
  `gpu_fault_processor_claimed_not_started` is non-zero, Aurora connection pool waits.
- **Boundary**: do not clear the queue by restarting workers; first obtain the stuck phase and request ID.

### GpuFaultProcessorClusterBacklogHigh

- **Meaning**: a single GPU cluster's queue is over 70% of its quota, meaning the backlog is concentrated in one cluster rather than global overload.
- **First check**: whether that cluster's collectors are replaying, whether the Executor is claiming, whether there is a node-level
  fault storm. Then look at `gpu_fault_processor_cluster_queue_oldest_age_seconds{cluster_id}`:
  the claim window is region-wide FIFO, one cluster's storm slows the others, and this metric gives each cluster's oldest request wait per cluster.
- **Boundary**: the per-cluster quota is an isolation measure; do not widen that cluster's quota to clear the alert and squeeze out other clusters.

### GpuFaultWorkflowMetricScanTruncated

- **Meaning**: the number of workflows that are non-terminal (PENDING/RUNNING/SAFETY_PENDING/BLOCKED) or updated within the scan window
  `GPU_FAULT_METRICS_WORKFLOW_SCAN_WINDOW_SECONDS` (default 7 days, exported as
  `gpu_fault_workflow_scan_window_seconds`) exceeded
  `GPU_FAULT_METRICS_WORKFLOW_SCAN_LIMIT`, and the three families `gpu_fault_workflow_step_total`,
  `gpu_fault_workflow_duration_seconds`, `gpu_fault_closed_loop_milestone_seconds`
  cover only part of that set. These three families are recomputed from the scan slice on every scrape, so they inherently
  describe only the closed loop within the window, not the full table history; terminal old rows outside the window are not scanned and do not trigger this alert.
  `gpu_fault_workflow_total` is still a database-side aggregate and is unaffected.
- **First check**: this is a real workflow storm or widespread stall, not audit history growth. Look at
  the counts of the non-terminal states in `gpu_fault_workflow_total` (are thousands of PENDING/BLOCKED
  piling up), the `gpu_fault_processor_*` queue and claim metrics, whether some rule is mass-producing new workflows
  per node; then compare `gpu_fault_workflow_scan_size` with
  `gpu_fault_workflow_scan_limit` to estimate the overshoot.
- **Boundary**: deal with the source producing workflows first. Merely raising
  `GPU_FAULT_METRICS_WORKFLOW_SCAN_LIMIT` shifts the cost to Aurora and every scrape; setting the window
  to 0 reverts to the old "latest N rows" behaviour, making this alert fire with table size again instead of in-flight count;
  do not use it to clear the alert.

### GpuFaultAttemptObservationMetricScanTruncated

- **Meaning**: the attempt observation table has exceeded the scan budget of `/metrics`, and
  `gpu_fault_ambiguous_attempt_ownership_current` and
  `gpu_fault_stale_attempt_observations` cover only the latest segment. The exclusive-ownership invariant is then
  evaluated on the slice, ownership conflicts among older observations are not reported, and
  `GpuFaultExclusiveNodeOwnershipInvariantViolation` may therefore be silent.
- **First check**: the gap between `gpu_fault_attempt_observation_scan_size` and
  `gpu_fault_attempt_observation_scan_limit`, whether the observation retention cleanup
  (`GPU_FAULT_ATTEMPT_OBSERVATION_MAX_AGE_SECONDS`, default 7 days) is still running,
  whether an attempt storm or a Completion Watcher that stopped terminalising is keeping the row count from converging.
- **Boundary**: the slice keeps the latest rows, and the signals of both the ownership and staleness families are at the recent end, so this is a
  warning, not an invariant failure. Merely raising `GPU_FAULT_METRICS_OBSERVATION_SCAN_LIMIT`
  shifts the cost to Aurora and every scrape; first find out why the table no longer converges.

### GpuFaultProcessorAdmissionRejected

- **Meaning**: requests are already being rejected. Rejected fault events depend on client resubmission, and there is a real detection-loss window.
- **First check**: the distribution of rejections by path and cluster
  (`gpu_fault_processor_admission_rejections_by_path_total`, `_by_cluster_total`),
  whether the telemetry spool is rejecting at the same time.
- **Boundary**: critical. Restore capacity first, then trace back whether any XID/SXID went unprocessed in this window.

### GpuFaultStoreIoSaturated

- **Meaning**: Store I/O lane usage is over 90% of capacity; the processor may start refusing Store operations at any moment.
- **First check**: Aurora CPU and connection pool waits, whether there are full-table-scan-type requests,
  `gpu_fault_store_io_admission_wait_seconds_max`.
- **Boundary**: before widening the lane confirm it is not a single slow query; widening the lane only shifts pressure to Aurora.

### GpuFaultStoreIoRejected

- **Meaning**: the Store I/O lane is full -- requests waiting for a slot beyond the admission timeout are rejected
  (`gpu_fault_store_io_rejections_total{reason="capacity"}`). The write path is then incomplete, and
  orchestration state may lag behind the real cluster state. Only the single reason `capacity` triggers this alert;
  a briefly unreachable writer or a request's own timeout is `GpuFaultStoreIoBackendUnavailable`, not here.
- **First check**: same as `GpuFaultStoreIoSaturated` (whether `gpu_fault_store_io_in_flight` hits
  `gpu_fault_store_io_max_in_flight`, `gpu_fault_store_io_admission_wait_seconds_max`,
  Aurora CPU and connection pool waits, full-table-scan-type requests); and check whether any workflow entered
  retry or fail-closed because of it.
- **Boundary**: critical. After recovery you must check that the workflows in that window agree with the actual node state.
  Do not raise `GPU_FAULT_STORE_IO_*` to clear the alert: widening the lane only shifts pressure to Aurora; first find
  the slow query filling the slots.

### GpuFaultStoreIoBackendUnavailable

- **Meaning**: within 10 minutes more than 5 Store calls were answered 503 + `Retry-After` because the PostgreSQL writer was briefly unreachable
  (`reason="backend_unavailable"`: failover, credential rotation, retryable errors after the server closed idle connections)
  or the request's own deadline came first (`reason="deadline"`), persisting for 5 minutes.
  The lane itself is not full; clients replay from their own outbox, and with a low count there is no data loss.
- **First check**: whether Aurora just had a failover or credential rotation (RDS events, whether
  `GpuFaultAuroraCredentialRefreshFailing` fires at the same time); the frequency of
  `Remote end closed connection` in the control-worker logs -- about once a minute is the known baseline, not a fault;
  split `gpu_fault_store_io_rejections_total` by `reason` to see which is rising; a continuously rising `deadline`
  means the request budget is being eaten by upstream queueing; look at `GpuFaultProcessorP95Slow`.
- **Boundary**: warning. Baseline-level occasional closed connections need no handling; escalate only when persistently above the threshold or when appearing together with
  `GpuFaultStoreIoSaturated`/`GpuFaultStoreIoRejected`. Do not treat
  these 503s as a capacity problem and tune `GPU_FAULT_STORE_IO_*`.

### GpuFaultProcessorP95Slow

- **Meaning**: request processing p95 exceeds 5 seconds for 10 minutes, the earliest leading indicator among the capacity alerts.
- **First check**: event loop lag, Aurora connection pool waits, whether a release just happened (compare release digests).
- **Boundary**: alone it is a warning; when it appears together with queue or Store alerts, handle by the latter's priority.

### GpuFaultProcessorExpiredLeasesReclaimed

- **Meaning**: a queued request was claimed by a worker (LEASED) and not completed beyond the lease, and the periodic reclaim task returned it to
  PENDING. The request is not lost, but waited at least one extra lease; every non-zero increase corresponds to a processor worker that died or got stuck
  mid-way. Queue-depth alerts cannot see it -- the depth is unchanged before and after the reclaim.
- **Which metric it reads**: `gpu_fault_processor_expired_leases_reclaimed_total` (a per-process counter,
  taking the 10-minute increase per separate process slot first, then summing by `control_plane_cluster, region`).
- **First check**: whether control-worker Pods restarted in the same window (the RESTARTS column of `kubectl -n <ns> get pod`
  against release times); whether some phase of `gpu_fault_processor_phase_oldest_seconds` stays high for long;
  whether the reclaim task itself reports errors in `gpu_fault_periodic_job_errors_total{periodic_job=…}`.
  One or two occurrences during a rolling release are expected; sustained growth outside a release window is workers dying.
- **Boundary**: do not lengthen `GPU_FAULT_PROCESSOR_REQUEST_LEASE_SECONDS` to clear the alert (that only makes stuck requests wait
  longer); do not hand-edit LEASED rows back to PENDING either -- the reclaim task exists for exactly that.

### GpuFaultProcessorCounterDrift

- **Meaning**: the processor queue's per-cluster depth reads the shard counters rather than `COUNT(*)`, and the two have disagreed for 5 consecutive minutes.
  At that point `gpu_fault_processor_cluster_queue_depth` and the per-cluster-quota admission decisions use a wrong
  number. The periodic task compares once every `GPU_FAULT_PROCESSOR_COUNTER_DRIFT_SCAN_SECONDS`, and a single sample may
  straddle a bulk commit, so the rule takes the minimum over a 5-minute window.
- **Which metric it reads**: `gpu_fault_processor_counter_drift_abs` (|expected total - counter total|)
  and `gpu_fault_processor_counter_mismatched_clusters` (how many clusters' shard counters differ from the real
  row count); either persistently non-zero counts. Both, together with the scan time and validity, are paired per process, taking the latest valid
  scan first and then computing the window, so that a non-zero value kept by an old task owner does not override a zero after the fix.
- **First check**: whether the drift is a stable constant or growing -- a constant usually means one transaction wrote the table without the counter, growth means
  the write path is still drifting; check whether `GpuFaultProcessorClusterBacklogHigh` is simultaneously false-positive or missing;
  `gpu_fault_periodic_job_errors_total` for failures of the comparison task. First look at which side of `gpu_fault_processor_counter_mode`
  is currently 1: under `dual` the legacy row is compared, under `partitioned` the shards; a one-off drift in the first
  scan period right after a mode switch is expected; only 5 minutes of persistence counts.
- **Boundary**: counters are rebuilt from the queue table by the backfill path; do not hand-edit counter rows; do not restart workers
  to clear the alert either, a restart does not rebuild counters.

### GpuFaultProcessorCounterDriftScanUnavailable

- **Meaning**: the control-worker is still scrapable, but for 5 consecutive minutes there has been no complete, valid latest counter scan.
  A missing item, a negative unknown value, a future time or an expired scan cannot prove the counters agree.
- **First check**: `gpu_fault_processor_counter_drift_scan_timestamp_seconds` and
  `gpu_fault_processor_counter_drift_scan_max_age_seconds`, the `counter_drift` task lease and
  `gpu_fault_periodic_job_errors_total{periodic_job="counter_drift"}`.
- **Boundary**: the scan validity is three configured periods, at least 120 seconds, 180 seconds by default. Only a real successful scan
  updates the time; republishing an old sample does not extend validity. While this alert exists, a silent drift alert does not mean agreement.

### GpuFaultTelemetrySpoolListenerDown

- **Meaning**: the telemetry spool consumer's `LISTEN/NOTIFY` has disconnected and degraded to polling. Function is still correct,
  latency is larger.
- **First check**: whether the Aurora connection is being cut by an intermediary, the spool worker Pod restart history,
  the growth rate of `gpu_fault_telemetry_spool_fallback_polls_total`.
- **Boundary**: do not clear the alert by disabling notifications; polling is the degraded path, not the target state.

### GpuFaultTelemetrySpoolDataLoss

- **Meaning**: the spool dropped routine samples. Dropping applies only to routine telemetry, not fault events, but
  host and network trends will have holes.
- **First check**: whether `gpu_fault_telemetry_spool_depth` and `_bytes` hit the cap, whether
  `GpuFaultTelemetrySpoolDrainDelayed` appeared before the drop.
- **Boundary**: critical. Host health conclusions within the drop window cannot be used for after-the-fact judgement.

### GpuFaultTelemetrySpoolAdmissionRejected

- **Meaning**: the spool is rejecting routine samples at the entry, one link earlier than `DataLoss`.
- **First check**: entry batching parameters, the in-flight byte cap, whether a single cluster is flooding.
- **Boundary**: as above; restore capacity first, then assess the extent of the telemetry hole.

### GpuFaultTelemetrySpoolReplayError

- **Meaning**: spool replay failed. Samples already persisted did not make it into the processing path.
- **First check**: the exception type in the spool worker logs, the ratio of `_abandoned_total` to `_superseded_total`,
  whether the Store schema version matches the wheel.
- **Boundary**: critical. A replay failure may be version drift; first do the REG-10A artefact consistency check.

### GpuFaultTelemetrySpoolDrainDelayed

- **Meaning**: the oldest sample in the spool has not drained for more than 60 seconds, a leading indicator of drops and rejections.
- **First check**: spool worker replica count and lane occupancy, Aurora write latency.
- **Boundary**: warning, but when persistent handle it as a capacity problem; do not wait for `DataLoss` to act.

### GpuFaultRemoteCommandUnclaimed

- **Meaning**: some PENDING remote command has not been claimed by any Executor for more than 600 seconds. The periodic sweep sets it to FAILED after
  `GPU_FAULT_REMOTE_COMMAND_CLAIM_DEADLINE_SECONDS` (default 900 seconds) and triggers
  `GpuFaultRemoteCommandUnclaimedExpired`; this alert is the earlier signal while the command can still be claimed.
- **First check**: whether the step's runtime profile owner is advertised by an adapter actually constructed by the target cluster's Executor,
  the Executor's TLS to the control plane, whether the cluster ID and owner match, whether the protocol
  version is within the compatible window.
- **Boundary**: critical. Do not hand-edit the command to a terminal state; troubleshoot item by item per REG-14.2.

### GpuFaultRemoteCommandExecutorInternalError

- **Meaning**: the Executor failed a command because of its own defect, not by refusing an unsafe action.
- **First check**: the traceback in the Executor Pod logs; read the stack before retrying. The alert uses the latest error timestamp within the retention
  period, so cleanup of terminal records is not misread as the counter resetting.
- **Boundary**: critical. This is the code-defect channel; the fix goes through the release flow and is not masked by retries. See
  REG-14.4 for details.

### GpuFaultUnknownProductFamily

- **Meaning**: a GPU product not mapped to an NVIDIA Catalog family appeared; automatic XID recovery on that product
  is fail-closed.
- **First check**: the alert's `product` label, the currently pinned catalog version, the node instance type.
- **Boundary**: add the product prefix to `spec.catalog.productFamilies` of the pinned catalog and go through the formal
  release; do not relax the policy online or guess the family for an unknown product.

### GpuFaultExclusiveNodeOwnershipInvariantViolation

- **Meaning**: the same GPU node is claimed by several "fresh" active attempts at the same time, and node changes have
  failed closed. Stale observations do not count towards ownership and have their own alert.
- **First check**: the `cluster_id` and `gpu_node` in the alert labels, whether the training Pods of these attempts
  all really exist, whether Completion Watcher delivery is duplicated.
- **Boundary**: critical. Do not hand-craft or delete attempts/allocations to resolve the ambiguity; handling must reference
  real workload observations. See §9.

### GpuFaultStaleAttemptObservation

- **Meaning**: the node has `PENDING/RUNNING` attempt observations not refreshed for more than 120 seconds. The value is the number of attempts,
  not the number of alert instances merged into the email.
- **First check**: whether the training Pods still exist, the Completion Watcher's Ready and logs, the contents of
  `gpu-fault-completion-watcher-outbox` (WAL) and `-outbox-active` (restart memory), and the watcher's
  `gpu_fault_completion_active_state_unavailable` on 9109 `/metrics` (scraped by the data-plane ADOT `gpu-fault-adot-dataplane`, series carrying
  `gpu_cluster`) (1 means the `-active` object is unreadable and the watcher runs without restart memory; check the Role's `resourceNames` and whether the object
  exists),
  `gpu_fault_completion_outbox_depth` and `gpu_fault_completion_outbox_quarantined_depth`
  (quarantined greater than 0 means records were quarantined: `rejected` = the control plane answered a non-retryable status, `expired` =
  the buffer was full for 24 hours without delivery, counted in `gpu_fault_completion_outbox_expired_total`. An `expired` record is removed from the WAL right after that
  ERROR (with key, payload digest, last status), and a one-shot replay cannot save it; if the attempt is still held by the
  watcher, the live path resends from the cache as usual. `rejected` records stay in the WAL as evidence and are evicted oldest-first to make room only when a critical event append hits
  the 256-record or byte cap (the ERROR carries key/digest/reason/last status, counted in
  `gpu_fault_completion_outbox_quarantine_evictions_total`); only when the WAL is all live records does it refuse the write and count
  `append_failures_total` (eviction makes room only for critical failure/terminal events; a routine workload
  observation's latest-wins write hitting the cap is still simply abandoned and goes only through live delivery). After a watcher restart `rejected` records can only rely on the one-shot replay:
  `kubectl -n gpu-fault-system exec deploy/gpu-fault-completion-watcher -- gpu-fault-completion-watcher --replay-quarantined`;
  exit code 1 means records remain undelivered or this replay expired and removed some live record (the ERROR summarises the counts), so run again or investigate.
  `rejected` records **never expire by age**: when the control plane is temporarily unreachable during the one-shot replay, only
  `last_transient_status/last_transient_error/last_transient_at` are appended to the record, the original verdict's `last_status`/`last_error`
  (such as 422 and its reason) are kept, and the closing ERROR states this was a transient failure to be rerun after the control plane recovers. With quarantined records present
  **do not** restart the watcher); `gpu_fault_completion_controller_resumed_attempts_total` greater than 0 means a `STOPPED` was once sent for an attempt that was actually
  still alive.
- **Boundary**: wait for the formal `STOPPED/FAILED/SUCCEEDED` observation to override; do not delete Store
  records directly to clear the alert.
### GpuFaultFleetRolloutFenceStuck

- **Meaning**: the cluster has a non-terminal fleet deployment that is already more than 2 hours old. The fleet fence holds **all**
  destructive remediation on that cluster (GPU reset, reboot, isolation restore) behind it, so the real meaning of this
  alert is "automatic recovery on this cluster is currently stalled", not "an upgrade was slow".
- **Why age rather than count**: a normal regional roll naturally leaves the deployment non-terminal; that is exactly
  why the fence exists. The longest roll observed on four p5en nodes was about 7 minutes; 2 hours is far outside that.
- **Which metric it reads**: `gpu_fault_fleet_rollout_fence_age_seconds{cluster_id}` is the age of the oldest
  non-terminal deployment. The same cluster's
  `gpu_fault_fleet_rollout_never_started{cluster_id}` is the triage bit:
  - **non-zero** -- the record never started a single wave (`DeploymentStatus.PLANNED` is a derived state that holds only when all
    nodes are still `PENDING`). It means the release that created it died right after writing the record, and nothing
    is in flight. 2026-09-03 was this class; the fence locked for 36 hours.
  - **zero** -- the rollout is really stuck in the middle of some wave. First check that wave's nodes' Agent leases and the
    `gpu-fault-node-installer-reconciler` logs, then decide retry or rollback.
- **First check**: `gpu-fault-admin status` **does not report** fleet deployments; look at the two
  metrics above in `/metrics`, or read `list_active_fleet_deployments(<cluster_id>)` inside a control-plane Pod.
- **Which commands can cross the fence**: only node actions "already accepted by the Node Agent" skip the fence on retry, and judged per
  **node**: a multi-node step skips only when every node has accepted; a step accepted by half still waits for the fence as a whole (the accepted
  nodes wait at most 420 s more, and the quiesce fail-safe restores them). In-flight records left from before upgrading to this version have no
  node acceptance list and are always treated as not accepted, so you will see one more fleet preflight held during the upgrade window.
- **Boundary**: do not delete Store rows, do not disable the fence, and do not raise the 2-hour threshold to clear the alert.
  Terminalisation goes only through the release engine: for an abandoned record, the `rollback-rollout-cleaned` phase of the next
  `rollout-regional-release` sweeps the leftovers of the same release, and when a new deployment is created
  `supersede_never_started_deployments` sets the same cluster's never-started old records to terminal.

### GpuFaultWorkflowLifetimeExceeded

- **Meaning**: a recovery workflow exceeded its lifetime cap from first claim (job workflows
  `GPU_FAULT_JOB_WORKFLOW_MAX_LIFETIME_SECONDS`, single-node workflows
  `GPU_FAULT_NODE_WORKFLOW_MAX_LIFETIME_SECONDS`, both 1 hour by default); the control plane has written it FAILED,
  cancelled in-flight remote commands, and the incident entered ESCALATED; from then on events of the same node/job are only recorded to that incident
  without automatic repair, until operations finish handling and the incident is closed.
- **Which metric it reads**: `gpu_fault_workflow_lifetime_exceeded_total` (executor counter, cumulative per process).
- **First check**: `gpu-fault-admin status` **does not list** incidents. The incident_id comes from the
  hardware-escalation notification email of this escalation (body `Incident: <id>`), the count from the Grafana `gpu-fault-overview`
  Closed loop panel or `gpu_fault_incidents_by_state{state="ESCALATED"}` in `/metrics`; then
  `GET /v1/incidents/{incident_id}`, `GET /v1/workflows/{workflow_id}` to read the `workflow_lifetime_exceeded` in the workflow's last
  FAILED step record and the cancelled remote commands; see which escalation rungs it went through
  (`branch_escalation_counts`).
- **Handling**: a human judges whether the node needs replacing or repair; after handling, close the incident, and only then will the node's new events
  trigger automatic repair again. Do not "fix" this alert by extending the lifetime.
- **Adjacent counters**: `gpu_fault_hardware_escalation_chain_terminated_total` counts "the escalated
  support workflow itself failed too, but no third one was opened" -- the escalation chain caps at the second level and the incident stays
  ESCALATED awaiting operations; `gpu_fault_hardware_escalation_containment_refused_total` counts the cases where all containment steps
  failed on safety refusals (for example the target node does not exist in Kubernetes) and the support workflow compiled only FREEZE_EVIDENCE
  and ESCALATE_SUPPORT. Both are cumulative per process; any growth means an incident needs a human to judge the node identity.

### GpuFaultJobWorkflowNodeBusyTimeout

- **Meaning**: rule A fired. A training job's node is occupied by another repair workflow, and the job waits only the single window
  `GPU_FAULT_JOB_WORKFLOW_NODE_BUSY_WAIT_SECONDS` (default 4 minutes); if the repair finishes within the window
  the job continues (the queued job workflow is dispatched, the `after_incident` restart restarts the job). Past the window the job is no longer restarted:
  a still-running job is stopped with STOP_WORKLOADS (carrying the control-plane initiator marker `termination_initiator_incident_id`),
  a dead job's restart plan fails; the workflow is FAILED, the incident ESCALATED. A job the system itself stopped is not
  restarted -- the completion service sees the initiator marker and records only NO_ACTION. The repair workflow occupying the node is unaffected.
  The dispatcher (job workflow not started) and the executor (the `requires_incident_state` precondition of `RESTART_WORKLOAD` already waiting)
  read the same variable, and the latter's step record ends with `reason=NODE_REMEDIATION_TIMEOUT`. The window counts from the **first**
  `HOLD(node under remediation)` event the dispatcher recorded for that row (`details.held_since`), not from `created_at`
  -- time blocked by the aggregation window `not_before` or a non-terminal predecessor does not count as waiting (D-11); the executor-side sequential loop
  re-checks at every step whether the job has been withdrawn. Consecutive waits of the same step fold into one STEP_ATTEMPT, `attempt` is "the number of
  attempts with a change", and when the wait is too long `details.step_waiting_slow=true` (RF-2).
- **Which metric it reads**: `gpu_fault_workflow_dispatch_node_busy_timeouts_total` (dispatcher giving up); the executor-side
  give-up is the `NODE_REMEDIATION_TIMEOUT` code of the STEP_ATTEMPT event.
- **First check**: read the failed workflow's `terminal_failure_reason`, which lists which nodes are held by which workflow;
  confirm whether that node workflow is still progressing normally (if not, it triggers the lifetime alert on its own).
- **Handling**: after the node repair completes the job owner simply resubmits -- the control plane does not restart it for them; if jobs repeatedly land on nodes under
  repair, check why the scheduler placed the job on a cordoned node.

### GpuFaultProcessorRetryHorizonFailures

- **Meaning**: some request in the processor queue kept failing to execute or complete, and the cumulative time it was released for retry exceeded
  `GPU_FAULT_PROCESSOR_RETRYABLE_RESPONSE_MAX_AGE_SECONDS` (default 300 seconds); the control plane completed it with
  503 and ended its circulation, so the lane is unblocked. The request's business side effects were not confirmed.
- **Which metric it reads**: `gpu_fault_processor_retry_horizon_failures_total`; alongside see
  `gpu_fault_processor_completion_failure_releases_total` (each failure release with backoff).
- **First check**: the processor log line `completed as failed past its retry horizon` gives
  request_id, path, failure and retry_count; read the JSON in the completion body by request_id.
- **Handling**: deterministic failures are usually a handler missing a parameter, a refused store write or a 5xx from the data plane; after fixing the root cause
  the data plane resubmits events of the same kind; do not hand-edit queue rows.

### GpuFaultRecoveryBlockedBacklog

- **Meaning**: `BLOCKED` is the fail-closed terminal state -- evidence, identity or workload state is insufficient to continue changing
  the node. A single one is normal safety behaviour; a persistent backlog means nobody is handling them, and each corresponds to a GPU node long absent from the training pool.
- **Which metric it reads**: `gpu_fault_workflow_blocked_unreconciled`, i.e. the count of "`BLOCKED` without
  a verified restore successor", the same criterion the release preflight uses to decide whether workflows block a release. It **is not** `gpu_fault_workflow_total{status="BLOCKED"}`: the latter counts the whole historical table,
  and `BLOCKED` is terminal, so that number only falls through `workflow-reconcile` or archival -- it does not fall even after a successor of the same incident
  has put the node back into the training pool, and an alert on it would keep resending per `repeat_interval`. So the correct meaning of this alert being silent is "no node is still held by a BLOCKED record", not "no
  `BLOCKED` rows in the table".
- **First check**: read each workflow's blocking reason and incident; distinguish "awaiting administrator decision"
  (see §9) from "awaiting evidence".
- **Boundary**: do not clear `BLOCKED` by changing Store rows or hand-editing workflow JSON, and do not raise the remediation budget to clear the alert.
  Historical records already converged by a successful successor of the same incident go only through
  `gpu-fault-admin workflow-reconcile`, see §10.2; that command requires a verified successor to exist and
  `restore_reconciliation_reasons` to be empty, refusing to write otherwise -- by design.

### GpuFaultClosedLoopSlow

- **Meaning**: within a six-hour event-time window, the mean time from workflow creation to first successful containment
  exceeds 900 seconds for 15 minutes; this is not the detection time from the raw hardware signal, nor does it alone prove
  that new training attempts still exist on the faulty node.
- **First check**: `gpu_fault_closed_loop_milestone_window_mean_seconds`, the corresponding
  `window_count/window_complete` and `gpu_fault_closed_loop_window_end_timestamp_seconds`;
  compare with capacity alerts, budget waits and `GpuFaultRemoteCommandUnclaimed`, then check the specific steps.
- **Boundary**: each workflow's first milestone counts once, and an inherited STOP is not a new event. Only complete non-empty windows
  sampled within the last 120 seconds take part in the alert; an empty window is count=0, mean=NaN, complete=1.
  Historical snapshot summaries no longer take part in delta computation. While the completeness alert below exists, silence does not mean compliance.

### GpuFaultClosedLoopWindowIncomplete

- **Meaning**: at least one control-worker's containment window is incomplete, expired, timestamped in the future or
  missing metrics for 5 minutes; healthy replicas cannot mask the unknown state of other replicas.
- **First check**: scan truncation and window configuration, milestone history truncation, metric contributor errors, clock and the actual
  archive retention period. The supported archive retention is at least one day; the scan window must not be shorter than six hours. A normal cache keeps
  the original scan time and does not refresh the timestamp on every scrape.
- **Boundary**: an incomplete window is mean/count=NaN, complete=0. Do not delete records, relax caps or clear the alert directly
  to fake recovery success; restore complete observation first, then interpret ClosedLoopSlow.

### GpuFaultNotificationDeliveryFailing

- **Meaning**: business notification delivery failed. Administrator decision requests (GPU count change, insufficient spares, mechanical inspection conclusion)
  did not arrive, and the human half of the closed loop has silently stopped.
- **First check**: SNS topic subscription state and whether there are duplicate confirmed/pending (see REG-7),
  `gpu_fault_notification_outbox_depth`, whether the workflow of the failed notification is waiting for a decision.
- **Metric semantics**: `gpu_fault_notification_terminal_failure_last_seen_timestamp_seconds`
  is the terminal failure time actually recorded inline or in the outbox; the rule uses a 15-minute event window with 5 minutes of confirmation.
  Deleting old FAILED rows or continuing to keep old DEAD rows creates no new failures. The outbox depth includes only
  PENDING/RETRY/LEASED, not DEAD.
- **Boundary**: critical. Before redelivery do not judge workflows awaiting a decision as "no-progress faults".
  Events already scraped by Prometheus may survive restarts within the query window; events of a process that died before publishing or scraping
  are not guaranteed to be rebuilt from this metric, and the persisted notification results and metric coverage alerts must still be checked.

After confirming the channel and permissions are restored and the notification still needs delivery, the operator calls
`POST /v1/advisory-notifications/{notification_id}/requeue` with the execution token, then observes the dispatcher's
actual result and provider MessageId. That action explicitly resets the retry budget of DEAD/RETRY;
`/send` does not revive DEAD, and an already SENT notification is not resent. Do not bypass the retry budget by deleting notification rows,
modifying Store JSON or repeating `send`; an isolated notifier result in acceptance does not represent real email delivery.

### GpuFaultBranchEscalationBudgetRefused

- **Meaning**: in a multi-node job repair, some node branch wanted to escalate in place to the next rung (reset -> reboot -> warm-spare replacement),
  but the concurrency budget of that action's resource class (such as `NODE_LIFECYCLE_MUTATION`) in this cluster was full, and the control plane chose to hand the branch
  to a human (branch exhausted, workflow finally FAILED / ESCALATED) rather than exceed the concurrency cap.
- **Which metric it reads**: `gpu_fault_workflow_branch_escalation_budget_refusals_total` (executor counter).
- **First check**: the incident_id comes from the escalation notification email (`status` does not list incidents; the count is in
  `gpu_fault_incidents_by_state{state="ESCALATED"}`), `GET /v1/workflows/{workflow_id}`
  reads the workflow's `exhausted_branch_ids`
  and the "remediation budget cannot take the next rung" in the last failure record; check
  `GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_RESOURCE_CLASS` (default 2) against the reboots/replacements in flight at the time.
- **Handling**: after the in-flight reboots/replacements close out, a human decides whether that node is rebooted, replaced or sent for repair; if it occurs frequently and
  hardware capacity is confirmed to allow it, then evaluate raising the concurrency cap of that resource class.

### GpuFaultWorkflowStepStalled

- **Meaning**: some recovery action (the `operation` label says which) has waited longer than **its own**
  expected cap without reaching a terminal state. The faulty node is still held and its training job still suspended while every component is healthy -- which
  is exactly why this alert exists: before it, "step stuck" and "step waiting normally" were indistinguishable in monitoring.
- **Which metric it reads**: `gpu_fault_workflow_step_waiting_seconds{operation=…}` minus
  `gpu_fault_workflow_step_waiting_warning_seconds{operation=…}`; the former is the maximum age of steps still in
  `WAITING` in non-terminal workflows, the latter the threshold applicable to that operation. Age rather than count: waiting
  itself is normal, and a count cannot distinguish "the node is rebooting" from "the node will never answer".
- **Why the threshold is a metric rather than a constant in the rule**: the threshold is tiered by operation. `REPLACE_NODE` /
  `RESTART_NODE` are handed to HyperPod managed recovery, waiting on the provider, so the cap is
  `GPU_FAULT_HYPERPOD_MANAGED_RECOVERY_TIMEOUT_SECONDS` (default 1800 seconds); the cap of the other actions is
  `GPU_FAULT_WORKFLOW_STEP_TIMEOUT_SECONDS` (default 600 seconds). A single number hard-coded in the rule could only be right for one
  tier: 600 seconds would false-alarm on every **normal** node replacement, 1800 seconds would give up alerting on all other actions.
  The control plane exports the applicable threshold under the same labels and the rule subtracts, so no adjustment of either window makes
  this rule stale. The lead time of the alert is the difference between `GPU_FAULT_WORKFLOW_STEP_TIMEOUT_SECONDS` and
  `GPU_FAULT_WORKFLOW_STEP_WARNING_SECONDS` (default 300 seconds), shifted by the same lead time
  onto higher-tier operations, so it always fires while the step is still waiting, not after the cap has already failed the step.
- **First check**: read the step's `details`, focusing on whether the object it waits on **has already given an answer**.
  The instance of 2026-09-05: the node agent had clearly answered "there are someone else's GPU processes on the spare", but that answer was
  mapped to `WAITING`; at the same time the step's retry count was read back from a record whose key never advanced, so the
  count stayed at 1 and `verify_max_attempts=60` was never reached. If the evidence in `details` (PIDs, command
  records, snapshot times) is clearly older than now, the engine is replaying an already finished record.
- **Boundary**: do not raise `GPU_FAULT_WORKFLOW_STEP_TIMEOUT_SECONDS`,
  `GPU_FAULT_WORKFLOW_STEP_WARNING_SECONDS` or
  `GPU_FAULT_HYPERPOD_MANAGED_RECOVERY_TIMEOUT_SECONDS` to clear the alert, and do not hand-edit the workflow's
  `step_executions`. Once the cap is reached the step fails with an explicit error and goes through the normal compensation and failure handling path; that is
  expected behaviour, not an incident to intervene in ahead of it. A managed recovery timeout additionally produces a notification (with an AWS Support
  case draft); that notification is the only actionable product of such a timeout; if you see only the step failure without a notification, the generic timeout
  preempted the observer, which is a configuration defect, not a runtime incident.

### GpuFaultWorkflowDeadlineNotEnforced

- **Meaning**: a non-terminal workflow has been past its own `execution_deadline` by more than 5 minutes. This alert reports
  not "some recovery is slow" but **the failure of the fallback mechanism itself**: the deadline passed yet it was neither executed by the lease holder
  nor closed by the watchdog.
- **Which metric it reads**: `gpu_fault_workflow_overdue_seconds`, the overdue seconds of the most overdue non-terminal workflow.
  Normally it should be a steady 0: once the deadline passes, the next dispatch has the lease-holding
  executor fail the current step and terminalise.
- **First check**: read the workflow's `execution_owner_id` and `execution_lease_expires_at`.
  - Both present, lease in the future -> the executor keeps renewing but does not enforce the deadline; check whether the line `workflow execution deadline exceeded` appears in the control-worker
    logs.
  - Lease expired or owner empty -> the watchdog should close it; look for
    the WARNING `workflow is past its execution deadline but held by its executor`.
    Before 2026-09-05 this was a silent `continue`, and the only symptom was a workflow never ending.
- **Boundary**: **do not** hand-modify `execution_deadline`, do not delete workflow rows, and do not restart control-plane Pods to make this
  number zero (a restart only switches to another owner that keeps renewing). The real fix is in code; the operations side
  only collects evidence.

### GpuFaultRemediationBudgetStarved

- **Meaning**: a workflow has waited for remediation budget for more than 30 minutes. The budget limits the number of nodes changed simultaneously;
  a short wait is correct overload protection, a long wait means quota is not being released.
- **First check**: `gpu_fault_remediation_budget_active_claims` by scope type against the configured cap;
  confirm whether the workflows holding claims are still progressing and whether their execution leases have expired.
- **Boundary**: do not raise the budget directly to clear the alert; first confirm there is no "claim holder dead but lease not expired"
  situation. Quota adjustment goes through [Administrator Capacity Configuration](administrator-capacity-configuration.md).

### GpuFaultRemediationBudgetClusterSaturated

- **Meaning**: the remediation budget of one GPU cluster (`cluster_id` label) has been full for 30 minutes:
  its active claims equal the single-cluster cap (`GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_CLUSTER`, factory 5),
  while workflows are also waiting because refused on this cluster's scope. Under a cluster-wide correlated fault (one AZ, 500+ nodes)
  this is **expected**: the cluster repairs one by one at the concurrency cap, and the drain time is roughly "waiting count ÷ cap × single repair
  duration", tens of hours for a few hundred nodes. The value of this alert is saying which cluster and how long to wait, not that the system is wrong.
- **First check**: `gpu_fault_remediation_budget_cluster_active_claims{cluster_id}` against
  `gpu_fault_remediation_budget_cluster_limit`, confirming it really sits at the cap;
  `gpu_fault_remediation_budget_cluster_waiting_workflows{cluster_id}` for the cluster's waiting count,
  `gpu_fault_remediation_budget_waiting_workflows_by_scope` for whether the refusal was cluster or
  region/node/domain scope; estimate the drain time with the mean of `gpu_fault_workflow_duration_seconds`;
  `gpu_fault_workflow_pending_age_seconds_max` finds the oldest waiting workflow; read its
  `remediation_budget_last_blocked_reason` / `remediation_budget_last_blocked_scope`.
  If a claim-holding workflow no longer progresses (lease never expiring), handle as `GpuFaultRemediationBudgetStarved` instead.
- **Boundary**: the single-cluster cap is the safety cap on nodes changed simultaneously; do not raise it to clear the alert; when the waiting count falls
  at the expected rate, just wait and tell the business side the drain time. Quota adjustment goes through [Administrator Capacity Configuration](administrator-capacity-configuration.md) and needs
  a change record and a rollback plan.

### GpuFaultCompletionEventsWithoutDecision

- **Meaning**: there are completion events with no decision row at all. Events and decisions should land in the same transaction;
  an event alone means it was recorded but never judged, and downstream never looks at it again; a failed attempt may thus have
  no recovery plan at all. An event may precede its decision by one scrape, never by half an hour.
- **Which metric it reads**: `gpu_fault_completion_events_without_decision`.
- **First check**: exceptions after the completion event write in the control-worker / ingress logs (a failed decision write leaves a stack);
  map these events back to their attempts and confirm whether the corresponding nodes have already been handled by another workflow.
- **Boundary**: do not delete event rows; finish the reconciliation above before replaying events, otherwise you open another
  workflow for an already recovered node.

### GpuFaultCompletionActiveStateUnavailable

- **Meaning**: the Completion Watcher writes two ConfigMaps, only one of which is required at runtime -- the write-ahead log carries critical completion
  events, while `gpu-fault-completion-watcher-outbox-active` carries the routine attempt state to be read back after a restart.
  The delivery path never touches the latter, so when the object is missing, or the ClusterRole's `resourceNames` does not name it, the watcher only
  degrades and does not crash, and this gauge is the only report. The rule's `for` of 15 minutes is about fifteen of the watcher's own re-probes: one
  ConfigMap re-apply or one rolling update does not bother anyone, while a genuinely unapplied manifest reaches a human within one shift.
  warning, not critical: as long as the watcher does not restart, nothing is lost.
- **Which metric it reads**: `gpu_fault_completion_active_state_unavailable` (0/1 gauge, 1 = that object
  is refusing reads/writes with 404/403); for the write-ahead log side also see `gpu_fault_completion_outbox_depth`.
  These metrics are exposed by the watcher in the data-plane cluster on 9109 and scraped by the data-plane ADOT (`gpu-fault-adot-dataplane`,
  one per GPU cluster, writing into the same AMP workspace); the rule groups by `control_plane_cluster, region, gpu_cluster`,
  and the alert instance names the cluster directly. A cluster without `clusters[].adot_irsa_role_arn` (site.yaml `adotIrsaRoleArn`) configured gets no
  collector, and this alert still does not fire on that cluster (`GpuFaultDataplaneCollectorMissing` does) -- there you can only
  read `curl :9109/metrics` manually on the watcher Pod.
- **First check**: on the affected cluster `kubectl -n gpu-fault-system get cm
  gpu-fault-completion-watcher-outbox-active` -- absence is the 404 side (the ClusterRole has only
  `get/update/patch` on configmaps, no `create`, and the watcher cannot recreate it itself); object present but 403 -> check whether
  `resourceNames` in `kubectl -n gpu-fault-system get clusterrole gpu-fault-completion-watcher -o yaml`
  lists this name. The active-state line in the watcher log states which kind of refusal it is.
  The fix goes through the release path, not `kubectl apply`. On a clean checkout on the deployment host (the working tree must have no changes; a dirty tree becomes
  a real release) first run `gpu-fault-admin status --state-dir <state-dir>`, checking
  `live_release.phase == complete`, `transaction_committed == true`, `next_deploy.kind == NOOP` --
  if not NOOP, stop; the checkout differs from production; first switch to the commit of the deployment that produced `live_release.release_id`.
  Then `gpu-fault-admin deploy --state-dir <state-dir>`: expect `status: SKIPPED_NOOP`, one line
  `configmap/gpu-fault-completion-watcher-outbox-active created` per GPU cluster in the log, and the ClusterRole and
  ClusterRoleBinding `unchanged` (`configured` means the online RBAC had been hand-edited and has now been restored -- the NOOP
  release reasserts the watcher's pair of RBAC objects, and manual relaxations are not kept). The NOOP path only fills in the missing state ConfigMap
  and does not touch the Deployment; afterwards `kubectl -n gpu-fault-system get cm gpu-fault-completion-watcher-outbox-active`
  should show the object, the gauge returns to 0 within 1 minute, and the watcher re-probes every minute by itself with no restart needed. **Do not** `kubectl apply -f deploy/dataplane/completion-watcher.yaml`:
  the Deployment in the raw manifest is unrendered (placeholder image and wheel volume name); Recreate would CrashLoop the only replica and lose
  the attempt memory -- exactly the restart forbidden in "Boundary" below.
- **Boundary**: until the alert clears **do not** restart or reschedule the Completion Watcher -- only a restart reads an
  "attempt whose Pod has already disappeared" as IDLE instead of ACTIVE, making the subsequent reset plan omit `STOP_WORKLOADS`; do not hand-craft a same-named
  ConfigMap or grant `create` on the ClusterRole to clear the alert; both void the RBAC scope narrowing.

### GpuFaultCompletionOutboxAppendFailures

- **Meaning**: the write-ahead log (`gpu-fault-completion-watcher-outbox`) is the only means by which critical completion events survive a watcher restart:
  buffered before the POST, cleared after the POST succeeds. A buffer failure no longer vetoes the POST, so the event is delivered as usual, and this
  counter is the only trace that "the log did not catch it". Any increase is a hole, so the threshold is `> 0` in a 10-minute window; the `for`
  of 15 minutes lets a single write conflict during a rolling update clear itself, while an unwritable ConfigMap keeps the alert up.
- **Which metric it reads**: `gpu_fault_completion_outbox_append_failures_total` (a monotonic counter; the rule reads
  `increase(...[10m])`); alongside see `gpu_fault_completion_outbox_depth` and
  `gpu_fault_completion_outbox_quarantined_depth` (quarantine semantics and the one-shot replay command are in
  the first check of the GpuFaultStaleAttemptObservation card).
  Same as the previous card: scraped by the data-plane ADOT (`gpu-fault-adot-dataplane`), the rule groups by `gpu_cluster` and the instance names the cluster;
  on clusters without a collector read manually on the watcher Pod, and for the absence of the collector itself see `GpuFaultDataplaneCollectorMissing`.
- **First check**: `kubectl -n gpu-fault-system get cm gpu-fault-completion-watcher-outbox -o json | wc -c`
  to see whether it approaches the ConfigMap's 1 MiB object cap (full means the 413 side); the append-failure line in the watcher log
  states whether it is 413/409/403; on the 403 side check whether the ClusterRole still has `update/patch` on that ConfigMap and whether the object
  was deleted by someone -- the ClusterRole has no `create`, and once deleted it cannot be recreated.
- **Boundary**: until the alert clears **do not** restart or reschedule the Completion Watcher -- a restart is the only action that turns this gap into real
  lost events or duplicate delivery; do not hand-edit the outbox ConfigMap's content to "empty" it either; the buffer may still hold undelivered
  critical events.

### GpuFaultIncidentsAwaitingOperator

- **Meaning**: `ESCALATED` is the parked state of an incident after the control plane hands the decision to a human -- the concurrency budget refused the next
  rung, the workflow lifetime ran out, a mechanical inspection is needed. It is a correct terminal state by design and is **not**
  closed automatically, so having a count is not itself a fault; but an escalation nobody has touched for a day is a GPU node (or a job) everyone forgot.
  This is a reminder that "a human queue exists", warning, 24 hours.
- **Which metric it reads**: `gpu_fault_incidents_by_state{state="ESCALATED"}` (a gauge by
  `IncidentState`).
- **First check**: this alert has only a count, and `gpu-fault-admin status` does not list incidents either; the queue itself has two entry points,
  either of which lists the ids:
  - `GET /v1/incidents?state=ESCALATED` (with the execution token; `&cluster_id=...`, `&node_id=...`,
    `&limit=N` may be added, default 200, at most 1000; `truncated=true` in the response means more were not listed) -- each row is
    `incident_id`, cluster, nodes, `created_at`, `workflow_request_id` and the first reason; after reading, as needed
    `GET /v1/incidents/{incident_id}` for the full text;
  - `gpu-fault-admin workflow-reconcile --state-dir "$STATE_DIR" --close-escalated --dry-run`
    (writes nothing) -- lists every ESCALATED incident site-wide from oldest to newest by created_at and whether it can be closed now
    (`would-close | refused(<reason>)`); the refused reason is what it is still waiting for.
  Each one also still corresponds to a hardware-escalation notification email (body `Incident: <id>`); the email is only the fallback.
  Route by the business handling of §9 (GPU count change approval, spare reservation, repair, `submit-remediation`); for nodes already repaired by a human
  whose record is still ESCALATED, close per §9.1 (one by one with `--close-incident`, or in bulk with `--close-escalated`).
  This is a handling process not closed out, not a monitoring false positive.
- **Boundary**: only an operator action moves an incident out of this state; the retention policy does not; do not change Store
  rows or silence the rule to clear the alert -- silencing amounts to admitting this queue needs no one watching it.

### GpuFaultWorkflowPendingAge

- **Meaning**: the oldest `PENDING` workflow has waited more than 30 minutes without any dispatcher claiming it. The dispatcher polls at second-level
  ticks, so 30 minutes is not queueing latency but "no dispatcher will take it": a predecessor workflow not terminalised, a retired generation
  awaiting an operator, or the dispatcher itself stuck. The dispatcher observes this age on every tick and alerts once when over the threshold
  (`gpu_fault_workflow_pending_age_warnings_total`); this rule is the AMP version of the same fact.
- **Which metric it reads**: `gpu_fault_workflow_pending_age_seconds_max`; alongside
  `gpu_fault_workflow_retired_generation_awaiting_operator` (the number of retired generations still needing a human to close them)
  and `gpu_fault_workflow_dispatch_preemption_pending_seen_total`.
- **First check**: read-only first -- `scripts/e2e/regional/audit_stuck_workflow_baseline.py` runs three read-only
  criteria; `gpu_fault_workflow_retired_generation_awaiting_operator` shows whether a retired generation was
  refused closure by the dispatcher and left to a human (see §10.1); `GET /v1/workflows/{request_id}`
  reads the workflow's `predecessor_workflow_id`, then whether the predecessor is terminal. Predecessor non-terminal and itself unclaimed ->
  handle the predecessor per `GpuFaultWorkflowDeadlineNotEnforced`; everything normal but still unclaimed -> look at the control-worker's
  dispatcher log and `gpu_fault_periodic_lease_errors_total`. When the predecessor is `BLOCKED(NEEDS_OPERATOR)`
  first see **whether it ran any step**: since the control-plane review of 2026-09-08 (C-03), a never-executed NEEDS_OPERATOR record
  no longer serves as a predecessor -- the next event on the same node recompiles it in place (`REPLACE_IN_PLACE`, same id, generation +1);
  for a BLOCKED record that ran steps: when the incident is already RECOVERED it is closed automatically by the sweep in the third row of §10.1; when the incident is still
  ESCALATED first close the incident per §9.1 (the closure itself settles that row).
- **Boundary**: do not delete or hand-edit PENDING workflow rows, and do not hand-set the predecessor to terminal; retired generations are revoked automatically by the dispatcher's
  sweep (§10.1), and `gpu_fault_workflow_retired_generation_awaiting_operator` counts records refused
  revocation (operations that already changed nodes were completed), handled separately per §8; no command closes them for you.

### GpuFaultNotificationUndeliveredTooLong

- **Meaning**: the oldest undelivered notification (delivery in PENDING / RETRY / LEASED) has waited more than
  30 minutes. It still has margin before being dropped outright by the shelf life (`GPU_FAULT_NOTIFICATION_TTL_SECONDS`, default 6 hours)
  and can still be saved now. The age counts from enqueue or the most recent manual requeue; retry backoff does not reset it.
- **First check**: `gpu_fault_notification_delivery_total{status=...}` to see which state it is stuck in --
  mostly `RETRY` means SNS/SES keeps refusing (see `last_error` on the delivery row; for SNS check the topic exists and the role's `sns:Publish`, for SES the identity and quota);
  mostly `PENDING`/`LEASED` means the dispatcher has not reached it; go to `GpuFaultNotificationOutboxNotDraining`.
  `GET /v1/advisory-notifications` can read the notification body.
- **Boundary**: critical. Do not shorten the TTL to make it expire faster and clear the alert; redelivery goes through
  `POST /v1/advisory-notifications/dispatch`, without changing Store rows.

### GpuFaultNotificationOutboxNotDraining

- **Meaning**: the outbox has PENDING/RETRY notifications, but no replica started a dispatch cycle within 5 minutes.
  This is exactly the combination warned about by the `notification delivery:` startup log line -- emails are queued in the outbox
  but no process is draining it.
- **First check**: the conclusion of `notification delivery:` in the control-worker Pod startup logs;
  whether `GPU_FAULT_NOTIFICATION_DISPATCHER_ENABLED` is true on at least one replica;
  a replica whose `gpu_fault_notification_dispatch_last_cycle_timestamp_seconds` is 0 has never dispatched,
  and a non-zero but stale one has a dead `gpu-fault-notification-dispatcher` thread (look for
  `notification dispatch cycle failed` in that Pod's log).
- **Boundary**: warning. A deployment with inline delivery (async off) has no undelivered rows, and this alert is naturally silent for it;
  its appearance means a notification really has nobody delivering it.

### GpuFaultNotificationExpiredUnsent

- **Meaning**: within the last 30 minutes a notification was past its shelf life when claimed and was retired directly as SKIPPED without being
  sent. Expiry is deliberate (mailing about a node recovered six hours ago is just noise), but it is still a notification nobody received,
  and persistent occurrence means the outbox drains slower than it fills.
- **First check**: `gpu_fault_notification_expired_total` for the count; check whether
  `GpuFaultNotificationOutboxNotDraining` or `GpuFaultNotificationUndeliveredTooLong`
  fired first in that period, and find what blocked the outbox (dispatcher off, SNS/SES refusing, a replica stuck). The notification body can still be read at
  `GET /v1/advisory-notifications`.
- **Boundary**: warning. Find the blocking cause first, then consider relaxing the TTL; do not bulk-requeue expired notifications to clear the alert.

### GpuFaultWorkflowDispatcherStalled

- **Meaning**: no control-worker replica started a workflow dispatch cycle within 5 minutes.
  `gpu_fault_workflow_pending_age_seconds_max` and all dispatch counters are frozen values from that moment --
  the thread writing them is not running; nobody claims PENDING workflows and nobody enforces deadlines.
- **First check**: whether the control-worker Pods are Ready; whether the `gpu-fault-workflow-dispatcher`
  thread inside the Pod is alive (py-spy dump); a Pod running but with a static timestamp is a hung Store call.
  `gpu_fault_workflow_dispatch_last_cycle_timestamp_seconds` by `pod` shows which replica stopped.
- **Boundary**: critical. Capture the thread stack before restarting; do not lengthen the dispatch lease to clear the alert.

### GpuFaultPeriodicRunnerStalled

- **Meaning**: no replica's periodic service thread ticked within 5 minutes. Training health, spare health, identity refresh,
  cleanup, archival, collector silence detection, lease reclamation and counter drift all run on this one thread; when it stops the counters of these eight
  no longer move.
- **First check**: `gpu_fault_periodic_job_last_run_timestamp_seconds{periodic_job=...}` shows which job ran last --
  usually the stuck one; control-worker Pod logs and thread stacks.
- **Boundary**: critical. One blocking Store call inside one job drags the other seven; locate first, then restart.

### GpuFaultProcessorUnhealthy

- **Meaning**: some control-worker replica marked its own processor unhealthy and has not recovered for 5 minutes. It refuses to claim
  processor requests and takes part in no periodic task; its share of the queue is carried by the other replicas.
- **First check**: the reason in that Pod's `/healthz` body, processor logs; commonly a lost Store connection or
  a schema version mismatch with the wheel (REG-10A artefact consistency check).
- **Boundary**: critical. Do not mask with repeated restarts; a schema mismatch is handled through the release flow.

### GpuFaultProcessorNoActiveConsumer

- **Meaning**: no replica is consuming the processor queue. Fault events are still accepted and enqueued but nobody claims them -- detection has
  stopped while all ingress health checks are still green.
- **First check**: whether the control-worker Deployment has Ready replicas, whether `GPU_FAULT_SERVICE_ROLE` is
  worker, whether the startup guard let it through; the queue depth alerts follow soon after.
- **Boundary**: critical. This is not a capacity problem; do not solve it by adding ingress replicas.

### GpuFaultTelemetrySpoolConsumerStopped

- **Meaning**: some telemetry-spool-worker Pod is running, but its spool consumer loop has exited.
  Routine telemetry spooled to it is no longer replayed, and `GpuFaultTelemetrySpoolDrainDelayed` follows.
- **First check**: the exception ending `run_telemetry_spool` in that Pod's log; the loop does not restart in place.
- **Boundary**: critical. This is the code-defect channel; read the stack before restarting the Pod.

### GpuFaultTelemetrySpoolConsumerStalled

- **Meaning**: some spool consumer loop claims to be running but has not issued a claim for 10 minutes. Even an idle loop
  claims once every `GPU_FAULT_TELEMETRY_SPOOL_NOTIFICATION_FALLBACK_SECONDS` (2-5 seconds),
  so this is stuck, not idle.
- **First check**: that Pod's thread stack (py-spy) -- usually stuck in a Store call or a replay
  future that never completes; `gpu_fault_telemetry_spool_claim_rounds_total` confirms the count is really static.
- **Boundary**: critical. Capture the stack before restarting, otherwise the stuck call can never be known.

### GpuFaultProcessorLeaseRenewalFailing

- **Meaning**: within 15 minutes a processor lease renewal was refused (another owner holds the lane; the same request may have run on
  two workers) or the Store errored during renewal. Previously only counted, watched by nobody.
- **First check**: compare with `GpuFaultProcessorExpiredLeasesReclaimed` -- reclaimed first then refused means a worker
  stalled on a lease and woke up; `gpu_fault_processor_renewal_errors_total` rising while fenced does not
  is a Store connection problem.
- **Boundary**: warning. Do not lengthen `GPU_FAULT_PROCESSOR_REQUEST_LEASE_SECONDS` to clear the alert. The counter is per
  process, the control-worker has 4 processes per Pod, and `/metrics` aggregates by SUM inside the Pod (no longer randomly hitting
  one process); but a process restart makes the Pod sum fall, and `increase()` reads that as a reset (6 times for 6 replicas during a rollout);
  once the processor exposes a last-seen timestamp this rule will read the timestamp instead.

### GpuFaultPeriodicServiceErrors

- **Meaning**: within 15 minutes some periodic task raised an exception (the `job` label says which), or the Store errored while taking the task lease.
  The runner continues and the job's tick is skipped -- the service it provides is late by one interval. After the control-plane
  review of 2026-09-08 (F-1), **failures of the job body itself** (exceptions raised by cleanup jobs, archival, scans) also count in
  `periodic_job_errors_total` and trigger this alert; previously only lease- and runner-level errors were visible, and a silently failing job body
  was indistinguishable from one that never ran.
- **First check**: the traceback of `periodic service <job> failed` / `periodic task lease
  <key> could not be taken` in the control-worker logs; `gpu_fault_periodic_job_errors_total{periodic_job}`
  for frequency.
- **Boundary**: warning. Persistent occurrence is handled by the corresponding job's runbook; do not remove the job from `_JOBS`.

### GpuFaultWorkflowDispatchInternalErrors

- **Meaning**: within 15 minutes a dispatch attempt raised an exception the dispatcher does not recognise (F-B4). The workflow stays executable with backoff,
  recovery is delayed rather than lost, but the same attempt keeps hitting the same defect until it is fixed.
- **First check**: `workflow dispatch hit an internal error` and its traceback in the control-worker logs;
  `gpu_fault_workflow_dispatch_internal_errors_total` for the count.
- **Boundary**: warning. This is a wiring defect of this replica (missing adapter, missing capability); the fix goes through a release,
  not through hand-editing workflow rows.

### GpuFaultWorkflowFailureHandlingAbandoned

- **Meaning**: within 30 minutes the failure handler of a failed workflow was abandoned after repeatedly raising (F-A6). The workflow is terminal,
  but the failure's side effects -- escalation, incident state, notification -- did not happen: a node still out of the pool that nobody knows about.
- **First check**: the workflow id in `failure handling abandoned` in the control-worker logs;
  `GET /v1/workflows/{request_id}` and the state of the corresponding incident.
- **Boundary**: critical. Manually check incident state and whether the node is still out of the pool, one by one, handling per §8/§9;
  `workflow-reconcile` handles only BLOCKED records already restored by a successor, not FAILED records,
  and the dispatcher's sweep (§10.1) does not close it out either; do not delete records.

### GpuFaultOrphanWorkflows

- **Meaning**: there are workflows still PENDING / SAFETY_PENDING past the aggregation window whose incident no longer exists or points at
  another workflow, and no successor names them as predecessor (F-B3). No dispatch scan reaches them, no fence covers them,
  no sweep closes them; they may still hold node claims.
- **First check**: the records behind the `gpu_fault_orphan_workflows` count; `GET /v1/workflows/{request_id}`
  reads their incident and predecessor (whether the incident no longer exists or points at another workflow).
- **Boundary**: warning. Currently no path closes them -- `workflow-reconcile` handles only BLOCKED records,
  the dispatcher's sweep (§10.1) closes only retired generations, compile-time BLOCKED and orphan remote commands, none of which include
  such PENDING orphans. Record the request_id as a product defect to follow up; do not delete rows or hand-edit state.

### GpuFaultIncidentDanglingWorkflowPointers

- **Meaning**: an incident's `workflow_request_id` points at a non-existent workflow row (F-B3). Later events on the same
  node find nothing following the pointer, and the duplicate-event fast path can only rebuild the chain
  (`gpu_fault_ingest_stale_event_link_repairs_total`).
- **First check**: `GET /v1/incidents/{incident_id}` to see whether the row `workflow_request_id` points at exists;
  check whether `GpuFaultOrphanWorkflows` fires at the same time (usually two faces of the same incomplete write).
- **Boundary**: warning. `workflow-reconcile` does not repair incident pointers; the next event on the same node rebuilds the chain via the fast path
  (counted in `gpu_fault_ingest_stale_event_link_repairs_total`). Do not hand-edit incident rows.

### GpuFaultStaleAgents

- **Meaning**: a Node Agent with lifecycle ACTIVE has had no valid heartbeat lease for more than 15 minutes. The control plane still
  treats these nodes as repairable targets but cannot reach them: every node action on them fails readiness, and the collector silence
  alerts follow. A few minutes is a reboot; a quarter of an hour is an agent that did not come back or a broken token/TLS.
- **First check**: `GET /v1/fleet/agents` lists the stale ones; on the node the `gpu-fault-node-agent` unit state,
  cluster token, TLS trust chain (REG-6).
- **Boundary**: warning. Nodes that no longer exist go through drain/revoke (§5); do not delete agent records.

### GpuFaultFleetPinAheadOfFleet

- **Meaning**: no live agent runs the agent version / artifact sha256 / policy / profile / config digest pinned by the control plane,
  so the pin changed rather than the nodes lagging (the `PIN_AHEAD_OF_FLEET` hint in readiness).
  Every node action on that cluster fails readiness.
- **First check**: `gpu_fault_fleet_pin_drift_nodes{cluster_id,kind}` -- `PIN_AHEAD_OF_FLEET`
  means fix the pin or roll the build out first; `NODE_STALE` means nodes lag their peers; upgrade or restart the agent on the node.
  The value is the result of that cluster's latest readiness evaluation and clears at the next evaluation after the fix.
- **Boundary**: warning. Do not change the pin to "any version" to clear the alert; pin changes go through the §4 release flow.

### GpuFaultRemoteCommandUnclaimedExpired

- **Meaning**: a remote command remained unclaimed beyond `GPU_FAULT_REMOTE_COMMAND_CLAIM_DEADLINE_SECONDS`
  (default 900 seconds) and was set to FAILED by the periodic sweep with `status_source=
  unclaimed-deadline-exceeded` -- the corresponding workflow failed because of it, and no node action was ever attempted.
  `GpuFaultRemoteCommandUnclaimed` is the earlier signal while the command could still be claimed.
- **First check**: the `execution_owner` written in the command's `error`, and whether the adapter that should advertise it was constructed by the target
  cluster's Executor (REG-14.2); `gpu_fault_remote_command_unclaimed_expired` is
  the number of such commands within the retention period and falls with retention cleanup.
- **Boundary**: critical. The workflow is already ESCALATED, and clearing the alert does not retry it; after fixing the owner handle per §9.

### GpuFaultProcessorFaultEventsRejected

- **Meaning**: within 15 minutes a fault-tier collection event got 202 at the entry and was then completed with 4xx during processor replay -- the node
  reported a fault, and the control plane dropped it without making any policy decision. Previously such 4xx were only a log line.
- **First check**: the `rejected-event` line in the processor log gives request_id, path, status and the Pydantic
  detail (only `loc/msg/type`, no payload); the node's collector status `errors` carries the same prefix,
  and that entry stays until the channel sees one success **later than the rejection** -- the
  `errors=[]` status the node sink reports right after only updates its own reason and does not overwrite the processor's verdict.
  Use `gpu_fault_processor_completions_by_path_status_total{path,status_class}` to see which channel:
  a path starting to produce 4xx after a release is a schema drift between collector and control plane, not a node fault.
- **Boundary**: warning. The counter is per process, the control-worker has 4 processes per Pod, and `/metrics` aggregates by SUM inside
  the Pod; `increase()` reads it as a reset only when some process restarts and the Pod sum falls (under-reporting,
  not false positives).
  Do not relax the entry validation to clear the alert;
  rejected events are not replayed automatically; after fixing the drift the data plane resubmits events of the same kind.

### GpuFaultRegionalRegistrySecretDrift

- **Meaning**: the regional registry Secret digest a role's process read at startup has disagreed with the durable head
  in Aurora for 15 consecutive minutes. The head is authoritative (join/remove publish there), and the running registry is right;
  what is stale is the release configuration -- a release that only rewrites the Secret changed nothing, while a Pod restarted from this Secret
  runs on the old cluster list until its first refresh.
- **First check**: compare `secret_config_sha256` with `durable_config_sha256` in the `regional_registry` payload of
  `/healthz`; `gpu_fault_regional_registry_secret_drift{service_role}` says which
  role. The startup logs of api-ha and control-worker give both digests in the line `regional registry Secret drifts`.
- **Boundary**: warning. Do not hand-edit the Secret or write the head directly; let the release configuration catch up with the head -- go through
  join/remove (§5) for the diverging clusters, or run a release with the REGISTRY component that lands the Secret on the head.

### GpuFaultControlPlaneReplicaScrapeFailing

- **Meaning**: some control-plane Pod has not answered the ADOT scrape (timeout or non-200) for 5 consecutive minutes. `absent(up==1)` fires only when all Pods disappear, so a single Pod's scrape failure used to be a blind spot -- and `min by (pod)` rules such as `GpuFaultProcessorUnhealthy` are exactly what cannot see the missing Pod.
- **Which metric it reads**: `up{job="gpu-fault-control-plane"} == 0`, with labels `pod`, `service_role` from target relabelling.
- **First check**: that Pod's readiness; `kubectl exec` into the Pod and `curl /metrics` on loopback to see whether it is a timeout or a 500; whether neighbouring Pods' `gpu_fault_metrics_contributor_errors_total` rises at the same time (a store-side cause).
- **Handling**: for the pool losing slots (after a password rotation) look at `GpuFaultPostgresPoolConnectionErrors` first; for a single hung Pod restart that Pod. Until it recovers, treat its processor health and loop liveness as "unknown", not "healthy".

### GpuFaultMetricsContributorFailing

- **Meaning**: some contributor of `/metrics` raised exceptions several times within 10 minutes. Rendering is now isolated per contributor: the failing family is skipped and counted, the other families are output as usual, so in-process gauges no longer disappear together with one store exception.
- **Which metric it reads**: `increase(gpu_fault_metrics_contributor_errors_total{contributor,pod}[10m]) > 3`.
- **First check**: the first traceback of `gpu_fault.app.metric_contributors` in that Pod's log (only one warning line afterwards); commonly a store read failure (pool timeout, failover, schema drift).
- **Handling**: treat the missing family as "unknown", not 0; after fixing the store-side cause the count stops, with no restart needed.

### GpuFaultPostgresPoolCheckoutQueueing

- **Meaning**: some Pod (`pod`/`service_role` labels, the 4 processes in the Pod already aggregated by SUM) has had callers waiting for a pool connection in every sample for 2 consecutive minutes (psycopg_pool `requests_waiting`): the pool is full or losing slots, and the next step is `PoolTimeout` -> 503 or worker threads stalling.
- **Which metric it reads**: `gpu_fault_postgres_pool_requests_waiting{service_role,pod}`; together with `gpu_fault_postgres_pool_size` vs `gpu_fault_postgres_pool_max_size`.
- **First check**: `pool_size < pool_max_size` with waiting -> the pool is losing slots (see `GpuFaultPostgresPoolConnectionErrors`); `pool_size == max` -> real saturation, look at Aurora latency and `gpu_fault_processor_lane_wait_seconds_max`.
- **Handling**: saturation goes through a configuration change (pool cap / thread count; first read the oversubscription reading below); lost slots are handled per the next card.
- **The oversubscription ratio is a reading, not an alert**: `gpu_fault_postgres_pool_oversubscription_ratio` (per role) is a ratio of configuration constants
  -- the number of threads of that role that "can hold a pool connection at the same time" (Store I/O threads + processor workers + dispatcher workers + periodic services)
  divided by the pool cap; both roles are > 1 under the current configuration and have never queued because of it. It is read only on the Control-plane capacity dashboard
  row "PostgreSQL pool and credentials" (the ratio, the `gpu_fault_postgres_pool_demand_connections{consumer=}`
  breakdown, `gpu_fault_postgres_pool_max_size` and `gpu_fault_postgres_pool_headroom_connections`),
  and at process startup `admission_runtime` prints the same estimate as one WARNING as a preflight signal. `> 1` alone is no basis for action;
  the real consequences are carried by this card (someone is waiting) and `GpuFaultPostgresPoolConnectionErrors` (slots are being lost). When adjusting the pool cap or
  thread counts, first see which consumer class contributes most, then check that role's `GPU_FAULT_PROCESSOR_WORKERS`,
  `GPU_FAULT_WORKFLOW_DISPATCHER_WORKERS` and pool cap; do not change the metric's accounting to make the ratio look good.
  LISTEN connections are outside the pool; see also `gpu_fault_postgres_unpooled_connections`.

### GpuFaultPostgresPoolConnectionErrors

- **Meaning**: some Pod (`pod`/`service_role` labels) has continuously failed to create new pool connections within 5 minutes. After a master password rotation this is the shape of the pool redialling with the old password; on every failure the pool re-reads the mounted Secret file (`gpu_fault_postgres_credential_rotations_total` +1 when the file changes).
- **Which metric it reads**: `increase(gpu_fault_postgres_pool_connections_errors_total[5m]) > 0`, by `pod`.
- **First check**: `credential_rotations_total` not moving -> the Secret was not updated; look at `GpuFaultAuroraCredentialRefreshStale/Failing` and the refresh CronJob; during a failover it rises briefly and stops; still rising with the Secret unchanged -> DNS / security group / Aurora availability.
- **Handling**: when the Secret lags, run the refresh Job once by hand; the pool self-heals with no restart; network causes are investigated as Aurora connectivity.

### GpuFaultAuroraCredentialRefreshStale

- **Meaning**: the hourly `gpu-fault-aurora-credential-refresh` CronJob has not succeeded for more than 3 hours, and the mounted Secret may not keep up with the next (7-day) master password rotation.
- **Which metric it reads**: `gpu_fault_aurora_credential_refresh_last_success_age_seconds > 10800`, from the `last-refresh-status.json` the refresher writes into the Secret.
- **First check**: `gpu_fault_aurora_credential_refresh_last_run_ok` at 0 means it is failing (see the next card); at 1 with a large age means the CronJob is not being scheduled; `kubectl get secret gpu-fault-aurora -o jsonpath='{.data.last-refresh-status\.json}' | base64 -d`.
- **Handling**: after fixing the cause run the Job once by hand; confirm the CronJob is not suspended and `startingDeadlineSeconds` was not missed.

### GpuFaultAuroraCredentialRefreshFailing

- **Meaning**: the refresher's last run ended non-ok, with no later success within 15 minutes. The status file records only the last run, so on failure the success-age series disappears; this card is its dual.
- **Which metric it reads**: `gpu_fault_aurora_credential_refresh_last_run_ok == 0`;
  across replicas the minimum is taken, so an old success projection cannot hide another replica's known failure.
- **First check**: the `error` in the status file (password redacted) and the latest Job log; commonly Secrets Manager reads, the verify connection (network / CA bundle), the patch RBAC on the Secret.
- **Handling**: after the fix run the Job by hand and watch `last_run_ok` return to 1.

### GpuFaultAuroraCredentialRefreshStatusUnknown

- **Meaning**: a role with a DSN file configured has been unable to verify the refresh status file for 15 consecutive minutes: missing, unreadable,
  malformed content or a completion time in the future. It is neither a refresh success nor proof the Job has failed.
- **First check**: `gpu_fault_aurora_credential_refresh_status_unreadable`, the status file projection,
  file permissions, clock and the refresher Job. Only status fields are checked; DSN, password or the full Secret are never output.
- **Boundary**: when unknown, the success age and `last_run_ok` are not exported; roles without a DSN file configured export none of these
  metrics and do not trigger this alert. Restore the evidence first; a missing field must not be hand-filled as success.

### GpuFaultAdotMemoryHigh

- **Meaning**: the ADOT collector's resident memory has been above 200MiB for 10 minutes (container limit 256Mi, `GOMEMLIMIT` 220MiB). Further up is an OOMKill loop -- each restart still lands a batch of data, `absent(up)` and the drop counters do not fire, and AMP is left with only gaps.
- **Which metric it reads**: `otelcol_process_memory_rss` (self job).
- **First check**: the ADOT Pod's `restartCount`; whether the cardinality of the newly added `cluster_id` / `pod` labels is growing.
- **Handling**: understand the cardinality source before considering raising the limit; do not shrink the families the alerts depend on in the keep list to make the alert disappear.

### GpuFaultProcessorConsumerStalled

- **Meaning**: some control-worker reports the processor consumer loop is not running, or has not started a new claim round for more than 30 seconds, persisting for 2 minutes. Previously `gpu_fault_processor_healthy` stayed 1 when the loop thread died.
- **Which metric it reads**: `gpu_fault_processor_consumer_running == 0` or `gpu_fault_processor_consumer_last_cycle_age_seconds > 30`, by `pod`.
- **First check**: that Pod's `/livez` lists `processor-consumer` as a dead thread and kubelet restarts it; on recurrence look at `gpu_fault_processor_consumer_cycle_errors_total` and the store call the loop hangs on in the processor log.
- **Handling**: a one-off restart suffices; on recurrence handle by the store-side cause (pool, failover).

### GpuFaultRemoteCommandStaleFenceSwept

- **Meaning**: the periodic sweep set to FAILED (`status_source=stale-fence`) remote commands that were LEASED while the workflow changed generation (fencing token mismatch), whose lease expired and for which nobody reported a result. This means no acceptable terminal result was received; it does not prove the physical action completed or never started.
- **Which metric it reads**: `increase(gpu_fault_periodic_cleanup_rows_total{periodic_job="stale_fence_remote_commands"}[30m]) > 0`.
- **First check**: the command's `result_details.stale_fence_swept`; a result the executor reports afterwards is kept in
  `result_details.post_stale_fence_status` / `post_stale_fence_error`, recorded only when the issued lease identity of the original LEASED command
  is still verifiable. A stale fence does not authorise another lease to overwrite the result, and a terminal command is not
  rewritten because of a cluster token; check whether an action overlapping the current generation
  happened on the node. Incidentally: the workflow lease of a WAITING step is now `max(3 × poll, 30 s)` (D-7),
  so a near `execution_lease_expires_at` is **not** an anomaly.
- **Handling**: trust the current generation only after manually checking the node state; this is an audit signal and needs no configuration change.

### GpuFaultNotificationDeliveryErrors

- **Meaning**: within 15 minutes a notification delivery raised an exception in the provider call or the bookkeeping after it; the row is released for retry and the rest of the batch continues, so it is "slower" rather than "lost" -- unless it keeps firing.
- **Which metric it reads**: `increase(gpu_fault_notification_delivery_errors_total[15m]) > 0`.
- **First check**: the exception in the `gpu_fault.notification_service` logs; commonly SES timeouts (5s/20s/2 attempts) and Secret/IAM errors.
- **Handling**: fix the external dependency; revive DEAD notifications explicitly with `requeue` (`send` no longer revives DEAD).

### GpuFaultControlRecordArchiveErrors

- **Meaning**: the control record archiver raised a non-safety-refusal exception for a candidate incident in the past 1 hour; the round continues and that incident is left to the next round. An S3 permission or bucket error makes every candidate fail, and retention silently stops reclaiming.
- **Which metric it reads**: `sum by (control_plane_cluster, region) (increase(gpu_fault_control_record_archive_errors_total[1h])) > 0`. The metric itself carries `reason` (exception type); the rule sums it away, and the alert instance carries no `reason`; to see the exception type read the raw series.
- **First check**: the archiver logs; whether `gpu_fault_control_record_archive_archived_total` is still growing.
- **Handling**: fix S3/IAM; `GPU_FAULT_CONTROL_RECORD_ARCHIVE_S3_URI` and the IRSA permissions are in the deployment manual 8.4.
## 9. Administrator Business Remediation

GPU count change approval, degraded recovery after insufficient spares and the explicit action after a mechanical inspection each have a detailed runbook:

- [GPU count change approval](deployment-and-operations-manual.md#81-gpu-count-change-approval) -- one `kubectl annotate`;
  there is no dedicated verb and none is needed;
- [Administrator manual degraded recovery when spares are insufficient](deployment-and-operations-manual.md#82-administrator-manual-degraded-recovery-when-spares-are-insufficient) --
  spare reservation is still a manual step (a product automation gap, see §12);
- [Explicit remediation after CHECK_MECHANICALS](deployment-and-operations-manual.md#83-administrator-submits-an-explicit-remediation-after-check_mechanicals) --
  `gpu-fault-admin submit-remediation --state-dir ... --incident-id ... --disposition
  {inspected,reset-gpu,reboot-node,quarantine,restore,confirm-node-action} --reference ...`; `--plan` only prints what would be submitted;
  `restore` is the validated restore after a QUARANTINED node is repaired (§9.2, judged per node),
  `confirm-node-action --node <node>` is the manual confirmation of a node action with unknown outcome (§9.4).

Administrators must not hand-craft non-existent allocations, attempt IDs or fencing tokens. Change decisions must reference
the current incident, real workload observations and the approval ticket.

### 9.1 Closing ESCALATED Incidents (Node Recovered but the Incident Still "Records Only")

**When to use**: `GET /v1/incidents/{id}` shows `state=ESCALATED` (reset/reboot handed to operations because of a lifetime timeout or support escalation), the node has been restored through the support incident's validated restore (`kubectl get node` shows no cordon and no `gpu-fault.io/` taint), but the incident still absorbs later XIDs on the same node (`gpu_fault_workflow_merge_record_only_total{reason="lifetime_exceeded"}` in `/metrics` grows). **Normally no manual action is needed**: when the support incident's restore workflow SUCCEEDED, the executor automatically writes ESCALATED incidents of the same cluster whose nodes are fully covered and which have no open workflow as RECOVERED (`gpu_fault_incident_auto_closed_by_restore_total` +1, reasons appended with `node restored via incident <id>`). Manual closure is for the scenario where the node was repaired by the vendor/manually without going through a restore workflow.

**Preconditions**: incident `state=ESCALATED`; the incident has no PENDING/RUNNING/SAFETY_PENDING or placeholder BLOCKED workflow (otherwise 409/refused; wait first or handle with `workflow-reconcile`). QUARANTINED incidents cannot be closed manually by default -- the node is still isolated and must go through the validated restore workflow (§9.2); the only exception is when the isolation on the node is already gone (released after a later incident took over, or lifted during acceptance cleanup), in which case close with node evidence per the "QUARANTINED branch" below.

**CLI (recommended)**
```bash
# See the conclusion first, writing nothing
gpu-fault-admin workflow-reconcile --state-dir "$STATE_DIR" \
  --close-incident inc-xxx [--close-incident inc-yyy] \
  --reason "node repaired by vendor, validated by ops" --dry-run
# Output, one per line: inc-xxx: would-close | already-recovered | refused(<reason>)

# Execute (--reference is the approved change/ticket number; the operator identity comes from aws sts get-caller-identity and is written into the audit event)
gpu-fault-admin workflow-reconcile --state-dir "$STATE_DIR" \
  --close-incident inc-xxx \
  --reason "node repaired by vendor, validated by ops" \
  --reference CHG-2026-0908-01
# Output: inc-xxx: closed | already-recovered | refused(<reason>); any refused gives exit code 1
# Result archive: $STATE_DIR/workflow-reconcile/incident-close/history/<UTC time>-<digest>.json

# When the id is unknown, or the queue is long (GpuFaultIncidentsAwaitingOperator gives only a count): --close-escalated
# auto-discovers every ESCALATED incident site-wide (all registered clusters), sorted by created_at then id,
# then makes exactly the same decision as --close-incident for each; mutually exclusive with --close-incident.
# See the conclusion first: no --reason/--reference needed, writes nothing
gpu-fault-admin workflow-reconcile --state-dir "$STATE_DIR" --close-escalated --dry-run
# First line: discovered N ESCALATED incident(s) across M cluster(s): <cluster list>
# Then one per line: inc-xxx: would-close | refused(<reason>) (QUARANTINED is not discovered -- see --close-quarantined below)

# Execute: likewise needs --reason and --reference; --max-items processes only the oldest N, leaving the rest for next time
gpu-fault-admin workflow-reconcile --state-dir "$STATE_DIR" --close-escalated --max-items 20 \
  --reason "rack repaired by vendor, nodes validated by ops" \
  --reference CHG-xxxx
# First line: discovered N ... ; processing 20 (capped by --max-items 20)
# Then one per line: inc-xxx: closed | refused(<reason>); any refused gives exit code 1
# Archive file as above, with the additional selector (states/cluster_ids/max_items) and discovered_incident_ids
```

**QUARANTINED branch (the isolation on the node is already gone)**

**When to use**: `state=QUARANTINED`, but `kubectl get node <node> -o json` shows `spec.unschedulable` empty/false, no `gpu-fault.io/quarantined` taint whose value equals this incident, and the annotation `gpu-fault.io/incident-id` is not this incident -- typically because a later incident took over the node and its restore released it, or acceptance cleanup lifted the isolation. A node really still isolated by this incident with merely the hardware repaired does not go here; it goes through the validated restore of §9.2.

**Nodes whose taint was removed by hand**: when only `kubectl taint`/`uncordon` released the isolation without touching the annotations, the node still carries this incident's `gpu-fault.io/incident-id`, `gpu-fault.io/fencing-token`, `gpu-fault.io/previous-unschedulable`. These three annotations are the bookkeeping `RESTORE_SCHEDULING` removes together with the taint, not the isolation itself; in that state the restore of §9.2 says there is no isolation to restore, while the evidence decision refuses because the annotations still point at this incident. `--close-quarantined` (and `--close-incident` pointing at a QUARANTINED incident) judges this shape -- no cordon, no taint of this incident, yet annotations of this incident -- as this incident's own residue: at execution it first strips the three annotations through the GPU kubeconfig with a merge patch carrying a `resourceVersion` precondition (if the node has been re-isolated by a new incident the patch conflicts and the incident is refused), then re-reads the node and hands it to the Pod for decision; `--dry-run` decides on the post-strip shape and truthfully reports `would strip` without touching the node. Annotations belonging to **other** incidents are not stripped.

**Evidence rules** (decided inside the Pod by the same service function; the CLI only reads nodes): for **every** node in the incident's `node_ids`, closure is possible only with `unschedulable=false`, no `gpu-fault.io/quarantined` taint whose value equals `quarantine_taint_value(<incident_id>)` (i.e. `incident-<sha256(incident_id)[:24]>`; the bare incident_id written by old versions also counts as this incident), and no `gpu-fault.io/incident-id=<incident_id>` isolation annotation. Taints/annotations held by other incidents do not block closure (that is their own isolation) but are named in reasons with their owner. A node not in the cluster, any node still carrying this incident's isolation, or a still-open workflow: refused. The API does not accept caller-supplied evidence (it has no kubeconfig), so QUARANTINED can only be closed from the CLI.

```bash
# Auto-discover every QUARANTINED incident site-wide (same order as --close-escalated), see the conclusion first:
# the CLI reads nodes with the site's own GPU kubeconfig (when the site has no rendered GPU kubeconfig that incident is refused; the default kubeconfig is never used),
# then hands the evidence to the in-Pod script for decision; writes nothing. Mutually exclusive with --close-escalated / --close-incident
gpu-fault-admin workflow-reconcile --state-dir "$STATE_DIR" --close-quarantined --dry-run
# First line: discovered N QUARANTINED incident(s) across M cluster(s): <cluster list>
# Then one per line: inc-xxx: would-close (isolation absent on <node list>) | refused(<reason>)
#   Nodes whose taint was removed by hand with annotations remaining: would-close (isolation absent on <n>; would strip orphaned isolation annotations on <n>)
#   Common refused reasons: node <n> still carries the gpu-fault.io/quarantined taint of incident <id> (go to §9.2) /
#   node <n> is still cordoned / node <n> is not in the cluster / still has an open workflow

# Execute: --reason and --reference as above; the node evidence read for each incident is archived with the result (isolation_evidence)
gpu-fault-admin workflow-reconcile --state-dir "$STATE_DIR" --close-quarantined \
  --reason "isolation released by successor incident; node schedulable and healthy" \
  --reference CHG-xxxx
# Then one per line: inc-xxx: closed (isolation absent on <node list>) | refused(<reason>); any refused gives exit code 1
#   Where orphaned annotations were stripped: closed (isolation absent on <n>; orphaned isolation annotations stripped on <n>), with the additional stripped_isolation_nodes in the archive
# --close-incident inc-xxx pointing at a QUARANTINED incident makes the same evidence decision (first judged in the Pod as needing evidence, then reads the nodes and decides a second time)
```

**Effect (QUARANTINED branch)**: same as ESCALATED, additionally appending per node `isolation no longer present on node <node>` to reasons (with the parenthetical `owned by incident <id>` when another party holds the isolation); the audit event details carry `previous_incident_state=QUARANTINED` and `isolation_evidence` (per node unschedulable / taint value / isolation annotation); likewise counted in `gpu_fault_incident_operator_closed_total`.

**API (equivalent; the CLI calls the same service function inside the Pod)**
```bash
# List the queue: state is required (repeatable or comma-separated), optional cluster_id / node_id / limit (default 200, at most 1000)
curl -sS "$CONTROL_PLANE_URL/v1/incidents?state=ESCALATED" \
  -H "X-GPU-Fault-Execution-Token: $GPU_FAULT_EXECUTION_TOKEN"
# 200 {"incidents":[{"incident_id":"inc-xxx","state":"ESCALATED","cluster_id":"...","node_ids":[...],
#      "created_at":"...","updated_at":"...","event_type":"XID","official_action":...,"effective_action":...,
#      "workflow_request_id":"wf-...","first_reason":"...","reasons_count":N}, ...],
#      "truncated":false,"states":["ESCALATED"],"cluster_ids":[...],"limit":200}
# 422: no state given / state is not an IncidentState value / limit out of range   403: missing or wrong execution token

curl -sS -X POST "$CONTROL_PLANE_URL/v1/incidents/inc-xxx/close" \
  -H "X-GPU-Fault-Execution-Token: $GPU_FAULT_EXECUTION_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"reason":"node repaired by vendor","operator":"alice@example.com","reference":"CHG-2026-0908-01"}'
# 200 {"incident":{...,"state":"RECOVERED","reasons":[..., "operator closed: node repaired by vendor by alice@example.com"]},"closed":true}
# 200 closed=false: already RECOVERED, idempotent, nothing written
# 409 detail: still has an open workflow <id> (<status>) / state is not ESCALATED (QUARANTINED: go through the validated restore of §9.2,
#   or when the isolation is already gone close with node evidence via the CLI --close-quarantined; the API never accepts self-reported evidence) / concurrent modification, retry
# 404: incident does not exist   403: missing or wrong execution token   422: missing reason/operator
```

**Effect**: incident -> RECOVERED, reasons appended with `operator closed: <reason> by <operator>`; the events of the incident's last workflow get `kind=OPERATOR_RECONCILED code=INCIDENT_CLOSED actor=<operator> details.reference=<ref>` appended; all its markers are retired (retired_by=<operator>); the next XID on the node for that fault scope regenerates a new incident/workflow. `gpu_fault_incident_operator_closed_total` +1.

### 9.2 Triggering a Validated Restore After a QUARANTINED Node Is Repaired

**When to use**: `state=QUARANTINED`, the node still carries the incident's `gpu-fault.io/quarantined` taint (value `incident-<sha256(incident_id)[:24]>`) and the `gpu-fault.io/incident-id=<incident_id>` annotation, the hardware has been repaired by the vendor/manually, and the node needs product verification before returning to scheduling. There is only one exit path: create under the **same** incident and the same fencing token a validated restore workflow `VALIDATE_GPU -> VALIDATE_HOST -> VALIDATE_FABRIC -> RESTORE_SCHEDULING` (the first three steps the validation adapter, the last the kubernetes adapter, which lifts cordon/taint/annotations after checking the node's isolation ownership by incident id + fencing token). Do not hand-edit the incident state and do not uncordon by hand -- that leaves the incident in QUARANTINED forever with the node ownership annotation still present.

```bash
# Read-only: prints the workflow that would be created (steps, owner, nodes, GPU scope, fencing token), writing nothing
gpu-fault-admin submit-remediation --state-dir "$STATE_DIR" \
  --incident-id inc-xxx --disposition restore --reference CHG-xxxx --plan

# Execute: writes incident+workflow under the product identity inside the CPU ingress Pod (one transaction) and wakes the dispatcher
gpu-fault-admin submit-remediation --state-dir "$STATE_DIR" \
  --incident-id inc-xxx --disposition restore --reference CHG-xxxx
# message: workflow workflow-validated-restore-<uuid> (fencing token N, PENDING) runs VALIDATE_GPU -> ... ; incident inc-xxx is now ACTION_PENDING
# Idempotent: rerunning while that restore workflow is still PENDING/RUNNING -> no_op=true, and message gives the existing id
# Evidence: $STATE_DIR/submit-remediation/<incident-id>/restore-<time>.json
```

**Preconditions (any unmet refuses and writes nothing)**: incident `state=QUARANTINED`; the incident has no PENDING/RUNNING/SAFETY_PENDING workflow (BLOCKED records do not count as open; those with unknown outcome are confirmed first per §9.4); every node of the incident is in the cluster; the incident and its last workflow have the same fencing token (otherwise a stale generation); **at least one node is still isolated by this incident**. Isolation ownership is judged node by node, and `node_decisions` of `--plan` gives restore/skip and the reason per node: nodes carrying a `gpu-fault.io/quarantined` taint whose value equals this incident, or carrying this incident's `gpu-fault.io/incident-id` annotation (taint/cordon removed by hand, annotation still present -- RESTORE_SCHEDULING checks ownership by exactly that annotation and clears it too) enter the restore; already clean nodes and nodes isolated by another incident (naming that incident, released by its own restore) are skipped, no longer refusing the whole incident because of the first clean node (exactly the shape when a multi-node job DAG's workflow restored one node itself and stalled on another). Nodes left with only a bare cordon and no gpu-fault isolation metadata are named in `warnings`: the restore does not release a cordon it does not own. When no node is still isolated by this incident it refuses and suggests closing via the QUARANTINED branch of §9.1. Validation of a subset restore does not carry the incident's GPU scope (full per-node validation): the GPUs named by the incident may be on already restored sibling nodes, and a scope that can only time out looks like a bad node.

**Effect**: incident -> ACTION_PENDING, `workflow_request_id` points at the new workflow, reasons appended with `operator restore requested by <operator ARN> (<reference>)`; after the workflow completes RESTORE_SCHEDULING the executor writes the incident RECOVERED as usual and incidentally closes other ESCALATED incidents on the same node (§9.1 automatic closure). If validation fails the workflow is FAILED and the incident returns to operations; after handling the failure reason, `--disposition restore` may be run again.

The order above denotes validation phases and does not guarantee a multi-node incident has exactly four steps. When the inventory can completely and uniquely prove
which node a GPU belongs to, GPU/Fabric validation expands per node, and all must still pass before scheduling is restored. GPUs the incident requires
are not excluded because the current inventory lacks that UUID; such a failure should first be investigated as a lost card or a wrong historical scope,
and must not be released by narrowing the validation scope or lifting isolation by hand.

### 9.3 Remotely Requeueing Collector Dead Letters

**When to use**: after `GpuFaultCollectorSilent`/`GpuFaultCollectorCollectionErrors` triage or a token rotation or control-plane outage,
some node's collector outbox holds dead letters with `replayable=false` (the control plane once refused with a non-transient 4xx), the root cause has been removed,
and the collector needs to redeliver. Previously this required logging into the GPU node to run `gpu-fault-collector outbox`; now the same command can be issued from
the deploy host: it creates an incident with `policy_source=OPERATOR` (`inc-operator-outbox-<node>-<action>-<time>`)
and a single-step `COLLECTOR_OUTBOX_MAINTENANCE` workflow (NODE_ACTION, non-destructive, does not take the node exclusively, allowed while training runs),
writes them in one transaction inside the CPU ingress Pod and wakes the dispatcher, then waits for the terminal state and prints the node's answer.

```bash
# Read-only: depth, replayable/dead split, number of records whose payload was truncated, earliest failure time, lock holder
gpu-fault-admin collector-outbox --state-dir "$STATE_DIR" --cluster-id gpu-a \
  --node ip-10-0-0-7.ec2.internal --collector kernel --action stats --reference CHG-xxxx

# Read-only: one line of metadata per record (request_id, path, dead/replayable, payload_truncated, error prefix, failed_at)
gpu-fault-admin collector-outbox --state-dir "$STATE_DIR" --cluster-id gpu-a \
  --node ip-10-0-0-7.ec2.internal --collector kernel --action list [--path /v1/...] --reference CHG-xxxx

# Write: flips dead letters back to replayable (those whose payload was truncated to a digest stay dead and are counted); --yes is mandatory
gpu-fault-admin collector-outbox --state-dir "$STATE_DIR" --cluster-id gpu-a \
  --node ip-10-0-0-7.ec2.internal --collector kernel --action requeue-dead --yes --reference CHG-xxxx
# message: requeued N dead record(s) ...; M left dead because only a payload digest was kept; dead X -> Y
# Evidence: $STATE_DIR/collector-outbox/<node>/<action>-<time>.json; on the node the requeued records get the error prefix
#       requeued by operator: <operator ARN> (<reference>): <original error>
```

**Boundary**: `--collector` accepts only the node collectors `kernel|dcgm|nvidia-smi|host|logs|fabric-manager`; `list`/`stats`
never return payload bytes (ARCH-G2); `--reference` is mandatory for all three actions. **There is no remote `--force`**: the Node Agent always takes
`<outbox>.lock` strictly (bounded retry of about 5 s), and on failure ends FAILED returning the line `recorded holder pid <n> (collector|cli:...)`
-- when the collector is buffering or replaying, simply retry later; only when the collector has stopped and left the lock do you log into the node and use `--force` of the node-local CLI.
When the Node Agent heartbeat does not declare `COLLECTOR_OUTBOX_MAINTENANCE` (the node is still on an old release) the command is refused before creating the incident,
suggesting to roll that node or use the node-local CLI instead. **Terminal states**: step succeeded -> incident RECOVERED; step failed (lock held, file unreadable,
Agent refused) -> workflow FAILED, the incident likewise converges to RECOVERED with the failure reason recorded in reasons (a single-step read-only workflow is
diagnostic-only to the executor and leaves no ESCALATED record needing manual closure), and the command exits non-zero printing the reason.

`--wait-seconds` bounds every status read and the whole poll, and a success returned after the deadline is not accepted. Every result must
correspond to the submitted workflow request ID and a known status; `SUCCEEDED` must additionally carry the target node and the result of the corresponding
action, otherwise the CLI exits non-zero. A timeout does not cancel the remote workflow; first query the terminal state by the printed ID before
deciding whether to resubmit, and do not treat the end of the local wait as the node action having stopped. Evidence keeps only metadata, without the outbox
payload or the raw output of the authentication helper.

### 9.4 Node Actions with Unknown Outcome: Manual Confirmation (confirm-node-action)

**When to use**: a workflow is `BLOCKED / NEEDS_OPERATOR`, some node action step (typically `RESTART_NODE`) FAILED
with details carrying `outcome_unknown: true, manual_confirmation_required: true` -- the executor got no receipt by the step cap
(the Agent happened to be stopped while the node rebooted), and the product stopped the record fail-closed by design. That record occupies all nodes of the incident:
`restore`, `--close-quarantined` and `workflow-reconcile` all refuse, and new faults on the two nodes no longer open incidents. On site you see
the node Ready on a new boot with the Agent back, but the product had no entry point to record that fact -- this command is that entry point.

```bash
# Read-only: prints the confirmation that would be written (step, command, boot id before/after the reboot, Agent generation, node/Agent evidence), writing nothing
gpu-fault-admin submit-remediation --state-dir "$STATE_DIR" \
  --incident-id inc-xxx --disposition confirm-node-action --node <node_id> --reference CHG-xxxx --plan

# Execute: writes that step record inside the Pod with compare-and-set and appends the audit event
gpu-fault-admin submit-remediation --state-dir "$STATE_DIR" \
  --incident-id inc-xxx --disposition confirm-node-action --node <node_id> --reference CHG-xxxx
# message: confirmed step N RESTART_NODE (boot <old> -> <new>) on node <node> of workflow <id> ...; the record stays BLOCKED until ...
# Evidence: $STATE_DIR/submit-remediation/<incident-id>/confirm-node-action-<time>.json
```

**Evidence rules (any unmet refuses and names what is missing)**: the record must be BLOCKED, have no execution owner, an expired lease, no WAITING
step and no open remote command; the node must be in the incident's node list, and the record must really have a pending node action on that node (otherwise it lists
which steps are pending -- a wrong node does not confirm something else). `RESTART_NODE`: node `Ready`, Node Agent record
`ACTIVE`, lease valid, a heartbeat after the action started, kubelet and the Agent reporting the **same** boot id, which differs from the boot id recorded before the
action (taken in order from `agent_baselines` on the step or its remote command, the STOP ownership authorisation `stop_reboot_authorization_v1`,
the single-node incident's `source_boot_id`; refused when none exists -- the record cannot prove the reboot). Other node actions (such as `RESET_GPU`):
the remote command carries the Agent's own terminal answer (`node_results`, SUCCEEDED/FAILED; the Agent refusing before the action also counts as terminal),
**or** a later validated restore of the same incident has completed `VALIDATE_GPU` and `RESTORE_SCHEDULING` on that node, and the incident is
RECOVERED and points at it. A BLOCKED record's deadline being long past is no grounds for refusal -- that is exactly why it was stopped.

**Effect**: changes only that step record: the uncertainty flags `outcome_unknown`/`manual_confirmation_required` etc. are set to false,
`operator_confirmed` records verbatim the operator ARN, `--reference`, time, boot id before and after, Agent generation, node and Agent evidence,
and the replaced old flags; the step stays FAILED (it really failed at the cap; only the outcome is now known); the workflow gets a
`kind=OPERATOR_RECONCILED code=NODE_ACTION_CONFIRMED` event appended; the record stays BLOCKED -- what ends it is the restore of §9.2
plus the workflow-reconcile of §10.2 (or closure with node evidence). Idempotent: rerunning for an already confirmed node returns `no_op=true` and names the earlier
`--reference`. The command-line side computes the decision, and the in-Pod script uses only store primitives present in the running release to re-read the record, compare the confirmed step
verbatim, re-check the Agent record and boot id, then write, so it is usable on the very release whose deployment the stuck record blocks.

## 10. Data and Audit Maintenance

**Control record retention is driven by site configuration and enabled by default.** The runtime default of the container environment variable `GPU_FAULT_CONTROL_RECORD_RETENTION_DAYS`
is still `0` (bare code deletes nothing), but when `gpu-fault-admin deploy` renders the control-plane manifests it fills in this variable
per `spec.retention` of `site.yaml`, and `spec.retention` defaults to 30 days: incidents, workflows
and their associated records are first archived to S3 after 30 days and deleted from Aurora only after verification passes. Disabling must be declared explicitly by the site administrator.

archive-first retention (first write the whole incident bundle to S3 and verify the SHA-256, then delete the live rows) is **enabled by default**,
with no file for the administrator to change: the default `spec.retention` is equivalent to

```yaml
spec:
  retention:
    controlRecordRetentionDays: 30         # default 30; only an explicit 0 disables
    # archiveS3Uri: s3://<bucket>/<prefix>  # optional; derived by default as
    #   s3://gpu-fault-control-records-<account>-<region>/<site>/control-record-archive
    # archiveIntervalSeconds: 600           # optional, archive scan period
```

The first bootstrap and every later `gpu-fault-admin deploy --state-dir <state-dir>`: create and harden the bucket
(bootstrap task `control_record_archive_bucket`: versioning, SSE-S3, Block Public Access; an existing bucket is likewise
converged); add `s3:PutObject` on that prefix to the control-plane IRSA role (the archiver only writes; it neither reads nor deletes objects); render the three variables only into
the control-worker Deployment's container env (ingress/spool do not run the archiver). To **disable**, declare
`controlRecordRetentionDays: 0` explicitly in `site.yaml` and deploy; to change the bucket, declare `archiveS3Uri`.


### 10.1 Stuck Records the Product Closes Itself (No Command Needed)

The following automatic close-out shapes keep their own complete safety predicates; the former `workflow-reconcile --mode ...` served to let the release
preflight pass. Their decisions all read the Store only, so they are now closed automatically by the sweep the dispatcher runs before every dispatch scan
(`WorkflowDispatcher.sweep_stuck_records`): the decision is re-derived from scratch, never by "age",
each record is written individually with compare-and-set (a row changed by someone else is refused rather than overwritten, and re-read next round), and no
row is deleted.

| Shape | Decision (all must hold) | Action | Audit location |
| --- | --- | --- | --- |
| Retired generation | The incident has advanced to a higher fencing generation and points at another workflow; the old workflow is still `PENDING`/`RUNNING`/`SAFETY_PENDING`/`BLOCKED` and has **not** completed any node-changing operation | First cancels its remote commands, then in the next round turns it `SUPERSEDED` (`preempted_by_workflow_id` pointing at the successor), releasing unused restart reservations | `preemption_reason` and the incident's `reasons`; control-worker log `retired generation revoked` |
| Compile-time BLOCKED | `BLOCKED` with non-empty `blocked_reasons`, never claimed (`execution_epoch` is 0), zero step executions, no completed operation, no owner, no source plan, no budget claim, no open remote command, and the incident is `ESCALATED`/`RECOVERED` | Turns it `SUPERSEDED`, `preemption_reason` carrying the original `blocked_reasons`; **does not write the incident** | An `OPERATOR_RECONCILED` event on the workflow with `actor` `dispatcher`; log `compile-time BLOCKED workflow closed by the dispatcher` |
| BLOCKED of a settled incident | `BLOCKED`, its incident already `RECOVERED` (`ESCALATED` does not count), no owner, unexpired lease, source plan or open remote command, and node occupancy, unconfirmed node actions and `WAITING` provider actions excluded; may have run steps, but must pass the complete safety predicate. Only then are residual budget claims released with the closure, with `released_budget_claims` recorded in the event details | Sets `SUPERSEDED`, `preemption_reason` carrying `closed BLOCKED workflow of a RECOVERED incident` and the original `blocked_reasons` | An `OPERATOR_RECONCILED` event, actor `dispatcher`; when an operator closes an incident with `--close-quarantined`/`--close-escalated`/`--close-incident` the same predicates apply |
| Remote commands left by a terminal workflow | The workflow is already `FAILED`/`SUCCEEDED`/`BLOCKED`/`SUPERSEDED` while its commands are still `PENDING`/`LEASED`/`WAITING` | Calls the Store's `cancel_remote_commands_for_workflow`: `PENDING`/`WAITING` become `FAILED` immediately (`status_source=workflow-timeout`), `LEASED` gets one cancellation request settled on the Agent side; workflow and incident untouched | An `OPERATOR_RECONCILED` event on the workflow (`actor=dispatcher`, `details.action=cancelled orphaned remote commands`, `details.command_ids`); log `orphaned remote commands cancelled by the dispatcher` |

How to confirm it happened: the events above appear in `events` of `GET /v1/workflows/{request_id}` (or
the revocation text appears in `preemption_reason`/incident `reasons`); the open counts by state of `gpu_fault_remote_command_total`
fall with the cancellations; `gpu_fault_workflow_retired_generation_awaiting_operator`
counts only retired generations the dispatcher **refused** to close -- they have completed a node-changing operation and have real state to compensate;
such records are still located by what they did per §8 and handled individually; no command closes them for you. The release preflight
`workflow_safety` lists compile-time BLOCKED records under `compile_blocked` and BLOCKED records of settled incidents under
`settled_incident_blocked` rather than `blockers`, so the release carrying the sweep logic passes its own gate without first closing records by hand.

The classification above does not include `BLOCKED / NEEDS_OPERATOR` with a still-unknown physical outcome. Incident terminal state, step timeout,
command cancellation or lease expiry cannot alone prove the node action has stopped; such records still occupy the node, and the release probe must
keep the blocker; automatic close-out must not release the occupancy or trigger compensation.

Commands left by `PENDING`/`RUNNING`/`SAFETY_PENDING` workflows are never touched by this sweep:
that is live work, and only the lease-holding executor may settle it; the duration of operator confirmations such as
`CHECK_MECHANICALS` is controlled by `GPU_FAULT_OPERATOR_ACKNOWLEDGEMENT_TIMEOUT_SECONDS` (default 86400). Likewise, records the dispatcher
itself wrote `BLOCKED` because of an internal error (`blocked_kind=INTERNAL_ERROR`, already claimed) are not part of the
compile-time shape and are handled per the `GpuFaultWorkflowDispatchInternalErrors` card.

### 10.2 Closing Historical Records That Recovered but Stay BLOCKED (workflow-reconcile)

The only manual shape kept: records whose successful recovery successor of the same incident has completed validation and
`RESTORE_SCHEDULING` (`terminalization` is `verified-restore`), or which never changed a node
(`never-changed`, requiring the incident to be `ESCALATED`/`RECOVERED`), while the historical predecessor is still
`BLOCKED`. The control plane cannot close it itself, because the evidence is on the node: the GPU node no longer has this solution's cordon,
`gpu-fault.io/quarantined` taint and isolation annotations. The command reads that evidence with the GPU kubeconfig rendered by the site
(refusing outright when the site has no GPU kubeconfig, never falling back to the shell's default kubeconfig), then hands the
decision to `gpu_fault.workflow_reconcile` inside the Pod to write. Do not delete database rows directly or hand-edit workflow
JSON.

See the plan first, writing nothing:

```bash
gpu-fault-admin workflow-reconcile --state-dir /secure/gpu-fault --dry-run
```

Without `--workflow-id` it scans all `BLOCKED` (by `updated_at` from oldest to newest, at most 10,000; exceeding that
is not refused, and `discovery.scan_truncated` is true). `--incident-id <id>` (repeatable) looks at one
incident only, `--max-items N` limits the batch size, and `discovery.remaining` tells you how many remain afterwards; these two
selectors are only for discovery, and giving them together with `--workflow-id` is refused rather than ignored. Each item's
`eligible`/`reasons` and `scheduling_evidence` (whether the node has scheduling restored) say whether it can be closed and why.

After confirming, execute once -- plan, re-check and write complete in the same invocation:

```bash
gpu-fault-admin workflow-reconcile \
  --state-dir /secure/gpu-fault \
  --incident-id <incident-id> \
  --reference CHG-12345
```

- Without `--dry-run`, `--reference` must be given (the change ticket number, written into the workflow's audit event and the source
  plan's `reconciliation_reference`), otherwise refused;
- Before writing, the command produces the plan **again** and compares it field by field with the plan just produced; if any field (fencing token,
  execution epoch, successor, node scheduling state...) changed it refuses by name
  (`workflow-...: fencing_token 7 -> 8`) and writes nothing; `workflow_updated_at` takes no part in the comparison,
  so a write changing only the timestamp does not void the plan;
- If a record named with `--workflow-id` is ineligible, the whole invocation refuses and lists the reasons; in discovery mode ineligible
  records are skipped and listed under `ineligible` in the result;
- Each record is written individually: `applied_workflow_ids`/`failed_workflow_ids`/`failures` say which succeeded,
  which did not and why; with failures the exit code is 1, what was written is not rolled back, and simply rerun -- closed records no longer
  appear in discovery results. The verified-restore shape writes workflow, incident and source
  plan in the Store transaction at once and keeps the incident pointing at the recovery successor; the never-changed-node shape writes only the workflow
  (`preemption_reason`) and the source plan, without touching the incident;
- Result and plan are archived together under `workflow-reconcile/history/<plan_sha256>/` (`plan.json`,
  `applied.json`); `actor` in `applied.json` is the executor's STS caller ARN (`user@host` when unobtainable),
  and the same identity and `admin_plan_sha256` are also written on the workflow's `OPERATOR_RECONCILED`
  event;
- `records_deleted` is always 0; only when the site has enabled `spec.retention` per §10 is the whole incident bundle
  archived and deleted by archive-first after the retention period, otherwise it stays in the Store permanently.

Non-plan-driven records (`source_plan_id` empty: node branches of job DAGs, workflows submitted directly with `--disposition`)
can also be closed out -- as long as they have a verified restore successor (`terminalization` is `verified-restore`).
The successor proves the recovery, the audit lands on the workflow's `OPERATOR_RECONCILED` event and the incident's reasons, and having no
source plan to write is no reason to refuse; `resolved_plan_ids` is empty for such records. The `never-changed` shape still requires a
source plan: there, no successor can prove anything. When the running release still carries the old rule (the record in the plan has `eligible=false`,
`reasons` contains only `workflow has no source recovery plan`, `terminalization=verified-restore`), the command-line side
promotes it to `verified-restore-non-plan` and writes through an in-Pod `apply-verified-restore` that uses only existing store primitives (the parser decision minus that one
reason, `amend_workflow` with an audit event, compare-and-set `save_incident`)
-- so this lever is usable on the very release whose preflight such records block; after the release the dispatcher's sweep also automatically closes out
the same shape (except node actions with a still-unknown outcome; confirm first per §9.4). Do not change Store rows to make the count fall.

When a manual migration or purge is required, first record the replicas, stop ingress, drain the consumers, then stop all writers.
Recovery is in the reverse order: consumers first, ingress last. The state file must not be deleted before all original replicas are
Ready.

## 11. Centralised Container Logs (CloudWatch Container Insights)

Not installed by default. Logs stopping at the node is a written-down trade-off ([Detailed Design v2 §10.1](detailed-design-v2.md#101-logging)),
and this section is the delivered path for shipping control-plane container logs out of the cluster.

**Applicability**

The site has completed bootstrap (the `eks-pod-identity-agent` addon is ACTIVE), and acceptance of
CloudWatch Logs ingestion and storage costs has been confirmed. This path **targets the CPU control-plane cluster only**; GPU clusters need explicit
release, for the reasons in "Scope of impact" below.

**Scope of impact**

- Installs the `amazon-cloudwatch-observability` EKS addon, creating the `amazon-cloudwatch`
  namespace and its own DaemonSet. Changes no rendered manifest of this repository, does not enter `module_digest`,
  is unrelated to release transactions, and can be installed and removed in a separate window.
- Creates four log groups: `/aws/containerinsights/<cluster>/{application,dataplane,host,performance}`.
  `application` carries the stdout/stderr of **every** container in the cluster; on the control plane that is this system's own
  logs, so the central redaction of `logging_setup` is a hard prerequisite.
- The scope of the `host` group is larger than its name suggests. Measured, `host-log.conf` is three journald inputs:
  `_TRANSPORT=kernel` (i.e. the kernel messages dmesg sees), `PRIORITY=0-6`,
  `SYSLOG_FACILITY=10`; the only `grep` exclusion is by syslog facility (mail/cron/authpriv),
  **not by unit**. So the `PRIORITY=0-6` input is unit-agnostic: as long as any `gpu-fault-*`
  systemd unit runs on the node, its journal also enters CloudWatch. Audit scope and the bill must be computed on that basis.
- **It does not** collect training log files -- Fluent Bit has no arbitrary-path tail input, only
  `/var/log/containers/*.log`. Nor does it collect `PRIORITY=7`.
- Whether collected or not, **the fault evidence channel is unchanged**: the control plane never reads CloudWatch, and
  `NodeLogBatch.collection_errors` remains the only channel that delivers collection gaps to the control plane
  (see §8 `GpuFaultCollectorCollectionErrors`).
- All inputs start from the **tail** (`READ_FROM_HEAD=Off`, `READ_FROM_TAIL=On`); logs written before
  installation are never backfilled.
- For billing, know that kubelet's journal is collected **twice**: `dataplane` collects by
  `_SYSTEMD_UNIT=kubelet.service`, and `host`'s `PRIORITY=0-6` collects it again (verified on
  2026-09-04 with the same `SyncLoop ADD` line hitting once in each of the two groups). This is a property of the upstream Container
  Insights configuration itself, not introduced by this script; count the duplication when estimating ingestion volume.
- Three extra consequences on GPU clusters, hence refused by default: `application` ships every training job's entire stdout
  into CloudWatch; `host` ships another copy of the kernel XID lines and the `gpu-fault-*` unit journals,
  both of which the node side already collects; and the agent's bundled dcgm-exporter contends for host port 9400, which
  is already taken by `gpu-fault-dcgm-exporter.service`. Release requires explicitly setting
  `GPU_FAULT_CLOUDWATCH_ALLOW_GPU_CLUSTER=true` while keeping
  `GPU_FAULT_CLOUDWATCH_ACCELERATED_METRICS=false`; without the release switch the script refuses clusters with GPU nodes
  with exit 2 and lists the three consequences.

**Read-only checks**

The default action is read-only; first look at the current state (addon, pod identity association, each log group's
retention and last event time, DaemonSet ready count):

```bash
AWS_REGION=<region> \
EKS_CLUSTER=<cpu-eks-cluster> \
EKS_KUBECONFIG=/secure/gpu-fault-bootstrap/control-plane.kubeconfig \
  deploy/observability/install-cloudwatch-observability.sh
```

Any log group showing `retention=None` means "never expires" and must be fixed before continuing.

**Command**

```bash
AWS_REGION=<region> \
EKS_CLUSTER=<cpu-eks-cluster> \
EKS_KUBECONFIG=/secure/gpu-fault-bootstrap/control-plane.kubeconfig \
SITE_ID=<site-id> \
GPU_FAULT_CLOUDWATCH_ACTION=install \
GPU_FAULT_CLOUDWATCH_RETENTION_DAYS=30 \
  deploy/observability/install-cloudwatch-observability.sh
```

The script is idempotent: it first creates the IAM role (tagged `gpu-fault:site-id`) and the pod identity association,
then sets the retention of the four log groups, and only then creates/updates the addon -- the order avoids leaving a
"never expires" window. When the kubeconfig and the cluster ARN of `EKS_CLUSTER` disagree it refuses to run.

**Success criteria**

Not judged by addon ACTIVE or Pod Ready. The script requires a log stream to appear in
`/aws/containerinsights/<cluster>/application` before it counts as success -- with a missing association
or a role that lost its policy, the agent is just as Ready yet every PutLogEvents fails. Then re-check with the read-only action that
the retention of the four log groups is the expected number of days.

Do not judge by "every Pod has a log stream": Fluent Bit tails from the end of the file and **does not backfill** logs written before
installation, so a Pod already quiet before the installation (an `api-ha` with no requests for a long time, a Completed
Job) has no log stream until it next writes a log. One missing stream cannot imply a broken chain; judge the chain by whether the
`application` group as a whole keeps receiving events.

**Rollback**

```bash
AWS_REGION=<region> EKS_CLUSTER=<cpu-eks-cluster> \
EKS_KUBECONFIG=... SITE_ID=<site-id> \
GPU_FAULT_CLOUDWATCH_ACTION=uninstall \
  deploy/observability/install-cloudwatch-observability.sh
```

Deletes the addon (without `--preserve`) and waits for addon-deleted, deletes the association, and deletes the IAM role only when it carries this
site's tag. **The log groups are deliberately kept**: they are the only copy of what the cluster said before it stopped shipping,
and retention already bounds their size; to discard them too, run the `delete-log-group` printed at the end of the script separately.
Irreversible boundary: already ingested logs are billed per retention, and uninstalling does not refund.

Note the relationship with §3: `gpu-fault-admin uninstall --cpu-cluster keep` **does not** clear this
addon (it is not in the resource registry, which is exactly the price of it not entering release transactions); run the uninstall above first;
`--cpu-cluster delete` deletes it together with the cluster.

**Evidence**

The change ticket number, the two outputs of the read-only action before and after installation, the addon's `--configuration-values`, the retention of the four log groups,
and the timestamp of the first log stream in the `application` group.

## 12. Commands Currently Pending Productisation

All verbs delivered by `gpu-fault-admin` (consistent with the parser; there is no second list):

```text
deploy               first/upgrade/resume/rollback/join/approve Profile plan/cross-schema release, all the same verb plus parameters
status [--full]      read-only health; --full is the complete standalone acceptance (the former verify and the former doctor idea both folded in here)
config               administrator capacity configuration (plan and execution in the same command, --dry-run shows only the plan)
join-cluster         joins only the given GPU clusters without triggering a site release (to upgrade at the same time use the deploy superset path)
remove-cluster       removes one GPU cluster by --gpu-cluster-arn
uninstall            uninstall; --cpu-cluster keep|delete, only --reset-database wipes the database
workflow-reconcile   converges historical workflows that recovered but stay BLOCKED (--dry-run / execute);
                     --close-incident <id> / --close-escalated close ESCALATED incidents,
                     --close-quarantined closes QUARANTINED incidents whose isolation is no longer on the node (§9.1)
rotate-token         one command completes the cluster token overlap-window rotation
node-key-custody     configure explicitly registers forward-looking custody; authorised delivery is still performed by deploy/join (§6.2)
submit-remediation   explicit remediation after CHECK_MECHANICALS
collector-outbox     remotely view/requeue a node collector's dead-letter outbox (§9.3)
failure-domain-map   read-only debug: prints the failure-domain mapping the product would render
```

The following verbs once listed as "roadmap" **will not** appear, each for its own reason:

| Verb once envisaged | Conclusion |
|---|---|
| `preflight` / `verify` / `approve-profile` / `resume` / `rollback` | All folded into `deploy` (automatic preflight, `--approve-profile-plan`, rerun is resume, `--rollback`) or `status --full`; no longer separate verbs |
| `doctor` | Folded into `status --full`: Pods, certificates, Agent/Collector versions, remote command backlog are all in that one report |
| `renew-certificate` | Not done. The NLB certificate is an 825-day server certificate issued by bootstrap with a private root CA (3650 days) and imported into ACM; ACM does not renew imported certificates; before expiry re-issue, import and go through a node rollout per the §6 table |
| `rotate-node-key` | There is no rotation verb that changes the Secret directly. Existing legitimate v2 keys are kept; a custody rotation first gets independent authorisation and registers the request, then goes through the original deploy/join and witness. A node reinstall constitutes neither a rotation nor historical proof (§6.2) |
| `refresh-aurora-credential` | Not done. Aurora credential refresh is already automatic: the hourly CronJob plus one synchronous run before every deploy/rollback (§4) |
| `approve-gpu-change` | Not done. GPU count change approval is one `kubectl annotate` (REG 8.1); wrapping it in a verb gains nothing |
| `reserve-spare` | Not a verb but a product automation gap: degraded recovery after insufficient spares is currently still performed manually per REG 8.2 |

Other current gaps: automatic evidence bundles and admission policy are not yet productised (the operations dashboards are provisioned automatically by deploy, see §8.0).

| Priority | Goal |
|---|---|
| Done P0 | schema-typed `site.yaml`, live preflight, one `deploy`, complete acceptance in `status --full` |
| Done P0.1 | Aurora AWS registry, single-command uninstall, CPU keep/delete choice and item-by-item residue verification |
| Done P1 | `rotate-token`, `submit-remediation`, single-mode `workflow-reconcile`, failure-domain mapping automation, automatic Grafana dashboard provisioning |
| P2 | Automatic evidence bundles, admission policy, spare reservation automation |

The acceptance standard for later administrator capabilities is unchanged: no Secrets printed, no manual `sed` required, no direct editing of solution
Manifests or online Deployments; declarative site configuration should have dry-run, persistent state, resume, rollback and evidence.
