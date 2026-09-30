# GPU Fault Recovery System Developer Deployment Implementation

English edition of `docs/开发者部署实现.md`; the Chinese file remains the source of record until both are maintained together.

This document is the production delivery contract, aimed only at developers who modify the
solution source code, production Manifests, renderer, configuration model or resource
lifecycle. Before adding an operation, channel, Store, route, plugin or metric, read the
[Extension Guide](extension-guide.md) first; follow this document when the extension needs to enter production.

The end-to-end order in which GitHub Actions builds the Runtime Image, the three component wheels, the Node bundle,
attestations and the deploy-host offline bundle from a source commit is described in [CI Release Process](ci-release-process.md). This document goes on to
define the render, deploy, upgrade and rollback contracts once these artifacts enter a site.
Developers who only need to complete "modify code, test on staging, promote to production" along the main path should first read
[EC2 Source Staging Unified Deployment Process](ec2-source-staging-reproduction.md) and return to this document when they hit Profile, resource or rollback branches.
First and subsequent deploys of a single-EC2 source staging are wrapped by
[EC2 Source Staging Unified Deployment Process](ec2-source-staging-reproduction.md) into the same four-parameter
`gpu-fault-admin deploy` entry point.
Both clean and dirty working trees are first copied into a content-addressed isolated checkout under the canonical `state-dir`.
A dirty working tree is then turned into a temporary commit that does not modify the development branch, and produces a `staging_only` candidate. A clean formal
commit attempts to automatically download and verify the signature of the same-commit main CI candidate only when `HEAD == refs/remotes/origin/main`;
on a hit it uses `release-build-promoted`; a personal clean commit, or an unavailable candidate, uses the local
`release-build`. GitHub Release also uses the promoted entry point; the three kinds of attestation are not interchangeable.
The generated internal site permanently references that isolated checkout and must not point back to the development working tree, which keeps changing afterwards.

| Source state | Build entry point | Mandatory gates | Release level |
|---|---|---|---|
| dirty development working tree | `release-build-staging` | public scan, impact tests, regional impact plan; escalate to the full suite when uncertain | `staging_only=true` |
| clean personal commit or no CI candidate | `release-build` | after static, parallel plain pytest / PostgreSQL stress / source-only artifact, then full artifact consistency last | `staging_only=false` |
| GitHub main CI signed candidate | `release-build-promoted` | CI gate signature verification and source/manifest match, final artifact consistency | `staging_only=false` |

The staging attestation is additionally bound to the impact-test `BASE`; when the source commit is the same but `BASE` changes, impact
selection must also be re-executed. Normal production deploys and signature verification refuse the staging-only level by default.

main CI splits unit evidence into four logical domains: `runtime`, `deployment`, `fault_runner` and `postgres`;
runtime is further split by a stable pytest nodeid hash into three signed execution shards, producing six physical shards in total.
Each shard is bound to the source, tests, dependencies and Python/Runner environment it consumes; the PostgreSQL shard is additionally
bound to the actual PostgreSQL image. On a shard miss only that domain is rerun; the six data sets are finally merged with
`coverage combine` and checked against the production 78% combined floor, the per-module
floors declared in `coverage.module_floors`, and the 95% statement and branch gates for production and runner respectively (deployment-only modules are visible only in the merged report). Deploy-host-only administrator source changes invalidate only the
deployment shard and do not require retesting the unchanged application Runtime; changes to runtime shared source
conservatively invalidate the multiple shards that depend on it. Documentation-only or `.github`-only changes are verified by the current static run and may reuse
all shards. The current commit always re-executes static, documentation/CI contract and artifact checks.

Administrators deploying an existing release use [Administrator Quick Deploy](administrator-quick-deploy.md) and
[Administrator Operations](administrator-operations.md); they do not read this document to apply solution resources one by one.
Administrators and customers do not write solution Manifests; the customer's own training workload YAML is the explicit exception.

## 1. Support Boundary and Normal Entry Point

The only production topology is a regional CPU EKS/HyperPod control plane, Aurora PostgreSQL, a TLS NLB and one or
more HyperPod EKS GPU clusters with `NodeRecovery=None`. Generic Kubernetes, Slurm and
CPU/GPU single-cluster all-in-one are not delivery targets for new production features.

The normal administrator entry point is:

```bash
gpu-fault-admin deploy \
  --cpu-cluster-arn <cpu-eks-or-hyperpod-arn> \
  --gpu-cluster-arn <gpu-eks-or-hyperpod-arn> \
  --state-dir /secure/gpu-fault \
  --admin-email <operations-email>
gpu-fault-admin status --state-dir /secure/gpu-fault
gpu-fault-admin status --full --state-dir /secure/gpu-fault
gpu-fault-admin workflow-reconcile --state-dir /secure/gpu-fault --dry-run
gpu-fault-admin join-cluster --state-dir /secure/gpu-fault \
  --gpu-cluster-arn <existing-gpu-eks-or-hyperpod-arn>
gpu-fault-admin remove-cluster --state-dir /secure/gpu-fault \
  --gpu-cluster-arn <gpu-eks-or-hyperpod-arn> --confirm REMOVE_GPU_CLUSTER
gpu-fault-admin uninstall --state-dir /secure/gpu-fault \
  --cpu-cluster keep --confirm UNINSTALL_GPU_FAULT
```

First deploys, subsequent upgrades and resumes after failure all use the first four-parameter command. `release-ref`, artifact, site,
`release-build/release-deploy`, bundle and venv paths are internal implementation and not part of the public CLI contract.
The hidden site mode is reserved only for internal orchestration and compatibility audits.

`resume` and `rollback` remain low-level regional orchestrator commands. Developers may use them in staging and break-glass acceptance,
but must not turn the low-level release JSON into the administrator's day-to-day source of truth. The high-level
`join-cluster` must enter the persistent state machine from the internal site in `state-dir` together with the GPU ARN.

The deploy-host installation produced by a unified source deploy is bound to one canonical `state-dir`. The binding record is independent of
the signed bundle content and has `0600` permissions; the installed administrator CLI fails closed for a different state directory, and the refusal
happens before log creation, source preparation, and any AWS or Kubernetes action. Reading the binding refuses non-private, unreadable,
symlinked, duplicate-field, unknown-schema or inconsistent-path records; an incomplete installation of an existing site must not be treated as unbound.
The local development `.venv` has no such binding, which does not affect unit tests or explicit low-level development commands; but against a
state directory that already has a bound CLI, an unbound development `.venv` may only run a plain source deploy (which prepares and re-enters the bound CLI itself) or genuinely read-only
subcommands; every other verb is refused with exit code 2 before the command log is opened and prints the bound CLI path that should be used instead
(`gpu_fault.admin.deploy_host_binding`). rollback, the explicit site form and the prepared-source form are not
plain source deploys. The existing release/pin, target, source and approval gates are not relaxed by this, and no inherited environment variable is offered as a bypass.
The admitted checkout is usually ahead of the deployed release, so `GPU_FAULT_*_IMAGE` (the images the site has deployed) and the manifest's locked digests
therefore disagree: any mode that would apply or verify images fails closed on that basis and prints both image references at the same time; drain-cluster, which only publishes the registry
and consumes no image, uses a context that consumes no image, so uninstall's `CLUSTERS_DRAINING` is not held up by the image
lock (`src/gpu_fault_release/regional_release_config.py`, see also §8.3).

The source preparer computes separate input identities for the deploy-host payload, deploy-host orchestration, Control Plane, Executor,
Node Runtime and release/Manifest. When only the deploy-host identity of an existing site changes, only the
content-addressed bundle/overlay venv is updated, and the impact gates and read-only preflight run; when the application identities are completely identical, no
Runtime Image or application release is generated and no GPU rollout is entered. Test-only/documentation-only changes execute only the impact gates.
The last successful identity comes from the local Cosign-signed authorization record; when it is missing or does not match, the process conservatively falls back to the application release flow.

## 2. Roles and Sources of Truth

| Role | Maintains |
|---|---|
| Administrator | existing cluster ARNs, state-dir, administrator email, approvals and acceptance evidence |
| Developer | source code, source Manifests, renderer, site schema, lifecycle contracts, tests and documentation |
| Automation | internal `site.yaml`, private release JSON, generated Manifests, build-time cleanup inventory and the runtime registry |
| Customer | their own training workload YAML; does not modify the solution's Manifests |

Administrators do not maintain the internal `site.yaml`. Administrators do not maintain `regional-release.json`. The CLI generates the low-level configuration from the internal site in state into a
`0700/0600` temporary directory; the direct release JSON and `regional-env.sh` are used only for the detailed
manual procedure or break-glass.

Production resources have three layers of source of truth:

| Layer | Source of truth | Purpose |
|---|---|---|
| Kubernetes | `gpu-fault-installed-resources` in each CPU/GPU EKS | uninstall the actually installed workload, RBAC and namespace resources |
| Node | `/opt/gpu-fault/installed-units.txt` and the AgentRecord digest | uninstall the actually installed systemd units |
| AWS | `installation_resource` objects in Aurora | record ARN/ID, ownership, delete policy, dependencies and status |

No second hand-maintained inventory of Kubernetes, unit or AWS resources may be added.

The runtime source of truth for the regional cluster registry is the immutable, monotonic-generation
revision/head in Aurora. The `gpu-fault-regional-clusters` Secret serves only as first bootstrap input, deploy input and
disaster-recovery copy; once a durable head exists, process startup must not use the Secret content to delete or overwrite the runtime
revision. Bringing the watcher/ACK protocol online for the first time still needs one CPU rollout; after that,
`join-cluster/remove-cluster` publish
`PENDING/ACTIVE/FAILED/ROLLED_BACK/DRAINING/REVOKED` revisions through the execution-token API and wait for every currently active
CPU process to ACK, including processes that joined after publication, with no restart of the CPU Deployment. An ACK must match the current
generation and content digest, and both the ready and the durable heartbeat must be valid. The `required_member_ids` snapshot taken at publication
stays immutable; a known record in it that exceeds the stale window is treated as having left the fleet, and neither counts as an ACK nor keeps blocking a fleet that is still active and
fully ACKed: an old process terminating during a rolling restart slows convergence by at most one stale window, instead of running out the convergence
timeout and rolling back. Missing records, future heartbeats or fleet-wide expired heartbeats cannot count as convergence; only when both the required set and the member records are empty
is it the empty-fleet success of a first bootstrap. Runtime readiness is constrained by the freshness of both the last successful refresh and the last successfully written heartbeat;
continuously reading the database alone cannot extend service eligibility. This contract requires the same release and consistent stale configuration,
and does not describe heartbeat expiry as proof that a Kubernetes Pod exited. Only `ACTIVE` clusters may claim remote commands; a failed join only transitions
the target cluster's lifecycle and does not overwrite other concurrently joining clusters with the candidate site. Single-cluster updates of the Secret disaster-recovery copy
must use Kubernetes `resourceVersion` CAS retries and must not perform concurrent unconditional read-modify-write.

## 3. Source File and Generated Artifact Boundaries

Production source files are placed by owner:

| Scope | Source files |
|---|---|
| GPU data plane | `deploy/dataplane/*.yaml`, `deploy/dataplane/optional/*.yaml` |
| CPU control plane | `deploy/control-plane/base/`, regional patches, `render_control_plane_role_split.py` |
| Observability | `deploy/observability/` |
| Node installation | `deploy/node/`, `deploy/systemd/` |
| AWS auxiliary resources | `deploy/aws/` and `src/gpu_fault/admin/bootstrap*.py` |
| Data migrations | `deploy/migrations/`, `src/gpu_fault/schema_migrations.py` |
| Runtime image | `deploy/image/` |
| Administrator resource lifecycle | `src/gpu_fault/admin/resource_registry.py`, `src/gpu_fault/admin/cluster_removal.py`, `src/gpu_fault/admin/uninstall.py` |

The following content is maintained by generators or the build process and must not be edited directly:

```text
deploy/control-plane/regional/generated/
deploy/control-plane/regional/cleanup-inventory.json
docs/环境变量参考.md
docs/管理员环境变量参考.md
src/gpu_fault/data/env-inventory.json
src/gpu_fault/data/nvidia-xid-catalog-610.generated.yaml
```

The Catalog, the two environment variable references and the regional Manifests each use their own generation entry point. The full boundary of the deploy directory is described in
[deploy/README](../../deploy/README.md).

Existing in `deploy/dataplane/` does not mean a regional production deploy will deploy it. The production rollout set is determined by the source Manifest's
`gpu-fault.io/deploy-mode` and the generated
`regional_deployment_inventory.GPU_ROLLOUT_DEPLOYMENTS`; currently these are only the
Completion Watcher, Kubernetes Node Resource Collector and Cluster Executor, plus the
Node Installer Reconciler. The historical in-cluster GPU metrics DaemonSet manifest has been deleted;
the GPU metrics collector is currently installed by node systemd, and a DaemonSet of the same kind and the systemd
producer must not be deployed at the same time.

## 4. Configuration, Profile and Release Contracts

New configuration must first settle on its single source of truth:

- Regional topology, identity, artifact references and health targets go into `RegionalSite`/`site.yaml`;
- Runtime Profile policy goes into versioned Profile YAML; content within the same `profile_version` cannot be overwritten;
- Non-sensitive CPU configuration goes into ConfigMaps split by domain;
- Administrator-mutable configuration goes into `state-dir/admin-config/desired.json`, normalized by the strict `AdminConfig`
  schema, which produces the global and per-role digests; the public template currently contains 22 fields covering control-plane capacity/topology/remediation budget,
  Aurora Min/Max, Processor, Workflow, notification delivery and evidence retention;
- Tokens, passwords and private keys go into Kubernetes Secrets, Secrets Manager or controlled files outside the repository;
- Node configuration is generated by the Installer as `/etc/gpu-fault/*.env`.

Processor parameters must be generated per role by the renderer; the worker's long lease/deadline must not be
copied to ingress/spool-worker, nor may `kubectl set env` be run online. The current regional worker uses
request lease/renew/deadline `150/10/120s`, ingress/spool use `30/5/20s`;
retry backoff is `1..30s`, and completion cluster concurrency is fixed at 1 in production.
All three CPU roles forbid Uvicorn `--limit-max-requests`: overlapping recycles of ingress during a burst reduce the
serving processes and amplify admission waits, and worker recycles interrupt claim/lease. Process replacement must go through a checked
Deployment rollout, Pod replacement or liveness.
When capacity acceptance changes the telemetry spool, control-worker replicas, remediation budget or Aurora
Min/Max, the administrator edits `<state-dir>/admin-config.yaml` and runs `gpu-fault-admin config`.
Within the same command the CLI generates and archives the plan SHA, while validating field upper/lower bounds, the ACU step size and Min/Max relationship,
spool switch/replica consistency and the PostgreSQL fleet connection budget, and then writes the normalized configuration into the low-level release JSON.
The publisher reuses the current signed images and wheels and invokes the same role-split renderer in a temporary directory, without modifying the checked-in
default generated artifacts. The global digest goes into release state and Deployment metadata, the per-role digests go into the Pod
template; a CPU configuration-only change is classified as `CONTROL_PLANE_ONLY`, executes no schema, GPU, Node Runtime,
Profile or cluster registry revision, and rolls only the affected CPU roles by digest. Release delivery additionally
maintains separate ingress, worker and spool Manifest digests; when a role Manifest changes alone, only that role is rendered/applied and
waited for, while changes to shared CPU inputs, the Control Plane wheel or pins still conservatively roll all CPU roles.
The baseline also reads the `GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION` actually loaded by one Running api-ha Pod
(`-config-core` is only a startup snapshot for the process): when it disagrees with the target version while the ConfigMap already agrees, this is classified as
a `cpu_pod_runtime_profile` change under `CONTROL_PLANE_ONLY` and forces a roll of the CPU roles; with no Running
replica it is treated as unknown and does not count as a change.
Aurora-only changes first
converge through the RDS API, after which the release goes through `NOOP` verification and refreshes the global configuration digest; for mixed changes, when the subsequent release
fails, the original Aurora Min/Max is restored only when it is proven that the original roles are unchanged or that a matching rollback completed.
`admin.config_recovery` binds the admission baseline of this run, the target full configuration and per-role digests, the previous snapshot, and either
explicit not-started component progress or completed rollback evidence. If management alignment has already cleared the rollback journal, it must additionally check
the driver receipts for the same site and same release within this invocation's time range, confirming that the rollback PASSED and the management state has been synchronized.
When the candidate is already committed, the change is partial, the rollback policy refuses, or live state cannot be read, the pending target is kept and the command fails;
the failure audit records the recovery decision, the reason and the digest of the live state that was read, without creating another journal.
`kubectl set env` must not be run online, nor may RDS be changed directly bypassing the administrator transaction.

The Aurora ACU window has only one write entry point: `gpu_fault.admin.aurora_capacity`.
`create_db_cluster_arguments(AuroraClusterSpec)` produces the `create-db-cluster` arguments for bootstrap
(the engine version constant `AURORA_ENGINE_VERSION`, and the two tags `gpu-fault:site-id` and `Application`),
`_modify_window` is the only `modify-db-cluster` capacity call, and the three public entry points share it:
`reconcile_aurora_capacity` performs the modification and settle in one go (bootstrap, the load-test fixture, and the
Aurora rollback of the administrator command), while `request_aurora_capacity`/`await_aurora_capacity` split the same change into two halves, "initiate and wait for
RDS to be in place" and "wait for the instances to actually run on the new ACU", so that `config` can roll the affected control-plane roles in between; `expected` may be omitted
(bootstrap and the load-test fixture have no change plan record) and `aws_json` may be injected (bootstrap passes
`CommandRunner.aws_json`; `config --dry-run` only prints the plan and does not call RDS). The legacy `deploy.sh` and the operations manual
call the same module through `python -m gpu_fault.admin.aurora_capacity create|reconcile`;
`tests/admin/test_admin_aurora_capacity.py` scans the whole repository to guarantee that this RDS parameter appears only in this module.
bootstrap's capacity reconciliation for an existing cluster is placed after the writer/reader are in place, because the shared reconciler must prove on both
instances that the window has taken effect.

**The writer shares the CPU control plane's availability zone.** Before creating the Aurora instances, bootstrap reads the
`topology.kubernetes.io/zone` labels of the CPU HyperPod nodes and puts the zone with the most nodes first: the first instance is the
initial writer (promotion tier 0) and the reader lands in the other private-subnet zone (tier 1). `aurora_ready` checks again once
both instances are available; when the writer sits outside the control-plane zone and an available member sits inside it, it issues
`failover-db-cluster` and waits for RDS to report the new writer, recording the outcome under `writer_zone_affinity`. A cross-zone
writer adds a hop to every database round-trip of the API Pods; under the §8.4 50-cluster load the reserved 503 share of normal
telemetry measured about three times higher.

Before `config` creates a plan or a NOOP audit, it must simultaneously verify that
`dist/current-release.json` and `dist/<release-id>/release.json` in the isolated checkout are byte-for-byte identical, use the
Cosign public key in state to verify that the attestation is indeed bound to that Manifest, and read the CPU cluster's
`gpu-fault-regional-release-state`, requiring the transaction to be committed, the phase complete, and the live release
ID identical. A drifted `dist/` in the development checkout, a modified snapshot, a signature mismatch or an inconsistent live release
all fail closed; the audit must not be bound to the wrong release.

The relative paths of wheels/bundles in a Manifest are resolved against **the repository root that contains the manifest** (the source snapshot at site deploy time),
not the checkout the running release engine lives in: `regional_release_config.load_release_artifacts` first
normalizes the Manifest's real path, then uses `gpu_fault_release.containing_repository_root` to locate the source snapshot it belongs to.
Locating requires both the repository directory anchors and the actual regional rollout entry file, and does not accept a same-named directory in place of a complete source tree. The relative paths of the three component wheels and the Node
bundle are all resolved from that root, and the same set of normalized paths is used for the existence check, digest verification and the return value,
avoiding a symlink being re-resolved after verification and selecting a different file. When the located snapshot lacks an artifact, it fails directly, without falling back
to the running engine's checkout; a legacy Manifest outside the repository keeps the original engine-root fallback, and configurations that use no Manifest still take
the configuration file's directory as the base. Therefore commands such as `join-cluster` that run the engine in-process inside the development checkout verify the
artifacts in the snapshot, and the checkout's `dist/` neither needs nor should hold links pointing at the snapshot. This locating does not replace signature verification and does not change schema-v4's
independent OCI and offline dependency identities.

A new AdminConfig plan can only be created from a committed live release. The idempotent resume after an interrupted release is the only exception:
the active plan and approval must still exist and be bound to the same release/target config, and the live state must be
`CONTROL_PLANE_ONLY`, with the target configuration digest exactly identical and not in rollback. When all conditions hold,
`transaction_committed` is allowed to be not yet restored and the original release continues; a missing approval, target drift or rollback state still
fails closed immediately.

spool enable follows the spool-worker → ingress order; disable first turns off ingress admission, waits for
`gpu_fault_telemetry_spool_depth` and `gpu_fault_telemetry_spool_leased` to drop to zero, and then scales
spool-worker down. Configuration hot reload is not part of the current contract; processes still read environment variables at startup.
### 4.1 What the Runtime Profile Does

The Runtime Profile is not an ordinary parameter file; it is the regional control plane's authorization and
routing contract for "who may execute which kind of capability". The policy engine first produces a recovery intent, and the workflow then uses the Profile to decide whether that capability is observe-only, executed by this
solution, or delegated to another implementation.

The Profile is split into two kinds of files:

| File | Maintainer | Purpose |
|---|---|---|
| `runtimeProfile.templateSource` | Developer | The only editable policy template |
| `runtimeProfile.source` | `release-deploy` | Immutable snapshot generated from the policy content |

Developers must not directly modify the `source` snapshot, the Profile version in `site.yaml`, or the Agent config digest.
`release-deploy` automatically generates the version, the snapshot and the related site fields from the template content.

The Profile is registered in the regional control plane only once per `profile_version` and is shared by every GPU cluster that references that version.
Modifying the Profile affects every training job and node in the whole site that uses the new version, not one particular registration anchor cluster.

### 4.2 Profile Fields and Modes

| Field | Meaning | Modification rule |
|---|---|---|
| `cluster_id` | Stable anchor used at first registration | The template keeps a placeholder; the releaser injects it from the site, and it does not change as GPU clusters are added or removed |
| `environment` | Orchestration environment the Profile applies to | Currently fixed to `hyperpod-eks` in production |
| `profile_version` | Immutable policy version | The template value is not used as the release version; it is generated automatically from the content digest |
| `claims[]` | Declares the expected mode, owner and adapter for each capability | The main place to edit for policy changes |
| `observed[]` | Declares whether the owner implementation is available and its implementation version | `OWN/DELEGATE` must have a matching fact with `available=true` |

Capability mode:

| mode | Meaning | Constraint |
|---|---|---|
| `DISABLED` | Capability fully off; does not enter the effective Profile | Used to explicitly delete a capability |
| `OBSERVE` | Only recognize, record and notify; no action may be executed | Default phase for new and high-risk capabilities |
| `AUGMENT` | Multiple detection sources jointly supplement evidence | Allowed only for non-destructive capabilities |
| `OWN` | The owner designated by this solution is the sole executor | Normal production form for destructive capabilities |
| `DELEGATE` | Delegate the capability to an external owner | Current production hard constraints usually forbid delegating managed recovery |

Destructive capabilities cannot use `AUGMENT`, nor may they have two active writers. When `OWN/DELEGATE` lacks
a matching observed fact it is downgraded to `OBSERVE` and produces a warning; the unified release entry point refuses a
Profile that carries warnings.

### 4.3 Every Capability in the Current Profile

| capability | Current mode/owner | Functional meaning | When to modify and possible impact |
|---|---|---|---|
| `gpuDetection` | `AUGMENT/hyperpod-hma` | This solution's Kernel/DCGM/FM Collectors supplement evidence beyond the AWS built-in detection; does not mean the HMA forwarding chain is attached | Keep the provider responsibility declaration; a change of detection source requires reviewing the event source and deduplication scope, and does not directly authorize recovery actions |
| `providerHealthIsolation` | `OBSERVE/hyperpod-hma` | Observe provider health-isolation facts | Currently kept at OBSERVE; switching to an execution mode may introduce a second lifecycle writer |
| `evidenceCapture` | `OWN/gpu-fault-control-plane` | Freeze events, metrics, allocation and diagnostic references | Modify when the evidence storage implementation or owner migrates; misconfiguration leaves the workflow without audit evidence |
| `schedulerDrain` | `OWN/gpu-fault-kubernetes-adapter` | cordon, quarantine and restoring node scheduling | Modify when the Kubernetes isolation implementation changes; affects whether nodes are schedulable, and is a destructive capability |
| `checkpointRestore` | `OWN/gpu-fault-kubernetes-adapter` | Manage checkpoint-related steps before stopping training and restarting | Modify when integrating a new training framework checkpoint protocol; an incorrect implementation may lose training state |
| `workloadStop` | `OWN/gpu-fault-kubernetes-adapter` | Stop the current managed training attempt | Must currently be OWNed by this solution; once disabled, training cannot be reliably stopped before hardware actions |
| `workloadRestart` | `OWN/gpu-fault-kubernetes-adapter` | Create a new attempt according to the restart budget | Must currently be OWNed by this solution; changes affect job recovery and the GPU-count gate |
| `mechanicalInspection` | `OWN/gpu-fault-kubernetes-adapter` | Move the workflow into a step that waits for manual mechanical inspection and explicit confirmation | Modify when the CHECK_MECHANICALS state machine changes; does not complete physical repair automatically |
| `deepDiagnostics` | `OWN/gpu-fault-validation-adapter` | Perform post-action validation of GPU, Host and Fabric actions | Modify when the validator or health-criteria version changes; failure blocks restoring scheduling |
| `supportEscalation` | `OWN/gpu-fault-support-escalation` | Generate fixed-template support escalation records and notifications | Modify when integrating a real ticketing system or when the template protocol changes; currently does not mean an external ticket has been created |
| `diagnosticBundleCapture` | `OWN/gpu-fault-node-agent` | Collect node logs, processes, DCGM and diagnostic bundles | Enabled by default in regional production; used as audit evidence before and after reset/fabric recovery |
| `gpuReset` | `OWN/gpu-fault-node-agent` | quiesce, no-client check, GPU reset and service recovery | Enabled by default in regional production; still constrained by the policy evidence, workload, signature, fencing and post-action validation gates |
| `fabricManagerRestart` | `OWN/gpu-fault-node-agent` | Controlled restart of the Fabric Manager and verification of the new PID/active state | Enabled by default in regional production; executed only by the fixed policy and the Node Agent |
| `fabricReset` | `OWN/gpu-fault-node-agent` | Execute a full reset barrier over all GPUs/NVSwitches of the node | Enabled by default in regional production, but must satisfy the full inventory, global training stop and barrier gates |
| `nvlinkDiagnostics` | `OBSERVE/gpu-fault-node-agent` | Run the NVLink Field Diagnostic with Link ID and a fixed SHA | Switch to OWN after the tool, parameters and SHA have been validated on the site; may occupy GPUs for a long time |
| `memoryDiagnostics` | `OBSERVE/gpu-fault-node-agent` | Run whole-card GPU memory/ECC/row-remap diagnostics | Switch to OWN after the memory diagnostic tool has been validated; a failed diagnostic usually keeps the node isolated |
| `driverRemediation` | `OBSERVE/gpu-fault-node-agent` | Run GPU driver remediation with a fixed command and target branch | Switch to OWN only when the vendor conclusion, target version and command SHA are all present; failure may leave a partially updated state |
| `efaDriverRemediation` | `OWN/gpu-fault-node-agent` | When the EFA PCI function exists but the `efa` driver is not bound, the Node Agent rebinds the driver (`REMEDIATE_EFA_DRIVER`) | Node Agent default `ALLOW_EFA_DRIVER_REMEDIATION=true`; before 2026-09-06 the regional template did not declare this capability, so all such incidents could only fail closed |
| `softwareFirmwareUpdate` | `OBSERVE/gpu-fault-node-agent` | Update and verify GPU or related firmware | Switch to OWN only after full acceptance in an isolated maintenance environment; an irreversible high-risk action |
| `nodeReboot` | `OWN/gpu-fault-hyperpod-adapter` | Invoke the HyperPod in-place node reboot and wait for the new incarnation | Modify when the adapter protocol changes or phase D.1 is withdrawn; reboots the physical node |
| `nodeReplace` | `OWN/gpu-fault-hyperpod-adapter` | Perform only the `HEALTHY_WARM_SPARE_ONLY` local switch-over | Modify when the warm-spare protocol changes or phase D.2 is withdrawn; must not call `BatchReplaceClusterNodes` |

If any Node Agent capability is changed from `OBSERVE` to `OWN`, you must also:

1. configure the correct `adapter` on the claim;
2. add an `available: true` entry with the implementation version for the same capability/owner to `observed[]`;
3. confirm that the Node Agent allowlist, installation parameters and heartbeat actually advertise that capability;
4. complete the maintenance-window acceptance for the corresponding destructive phase in the [Security and Parameters Reference](security-and-parameters-reference.md).

### 4.4 When Not to Modify the Profile

The following changes usually do not change capability authorization and do not modify the template:

- ordinary bug fixes or internal refactoring;
- rebuilding the Control Plane, Executor or Node Runtime;
- wheel, bundle, Schema or protocol version changes;
- adding or removing GPU nodes, or joining/removing GPU clusters;
- changing timeouts, queues, Collector periods or resource requests/limits;
- releases in which the capability owner, mode, adapter and implementation version all stay the same.

These changes are expressed by the release manifest, the environment configuration or the Agent config digest.

### 4.5 Profile Modification and Automatic Release Flow

Developers edit only the YAML pointed to by `templateSource`; they do not modify the site or historical snapshots. Recommended flow:

1. Modify the `mode/owner/adapter` of the target claim and the corresponding observed facts.
2. Run the relevant unit tests and the Profile compilation tests:

   ```bash
   .venv/bin/python -m pytest -q tests/regional/test_capabilities.py
   ```

3. CI builds, tests and pushes the immutable runtime image:

   ```bash
   make PYTHON=.venv/bin/python release-build \
     RUNTIME_IMAGE_REPOSITORY=<registry/repository>
   ```

4. Run the target site's existing deploy command. When the Profile content has changed, the first run writes the diff to
   `<state-dir>/release-deploy/profile-plan.json` and stops. Review its
   `change_kind`, capability changes, original/target versions, live baseline, policy digest and target snapshot digest.
5. After obtaining a change ticket or maintenance-window approval, rerun the same deploy once with the approval parameter, which writes a one-time state approval bound to that plan
   digest and continues the same deployment:

   ```bash
   gpu-fault-admin deploy \
     --state-dir /secure/gpu-fault \
     --approve-profile-plan "$(jq -er '.plan_sha256' \
       /secure/gpu-fault/release-deploy/profile-plan.json)" \
     --reference CHG-12345
   ```

6. There is no separate approval verb; when approval fails or the plan drifts, re-review using the current digest given in the error.
   Subsequent ordinary upgrades remain `gpu-fault-admin deploy --state-dir ...` without the approval parameter.
7. The script automatically:
   - normalizes claims/observed, ignoring YAML order and `cluster_id/profile_version`;
   - computes the policy digest and generates the `regional-hyperpod-<digest12>` version;
   - writes the immutable snapshot `<site directory>/profiles/<version>.yaml`;
   - updates the site's `source/version` and the Agent config digest;
   - verifies the CI attestation signature, Manifest SHA and delivery identity, without rebuilding on the deploy host;
   - lets the component DAG select CPU, Executor, Watcher, Collector, DCGM, Reconciler and Agent;
   - on first deploy writes only the first GPU cluster into the baseline site and completes the CPU-plus-one-GPU baseline; the remaining GPU ARNs enter
     independent batch joins after the baseline verify, rather than binding multiple GPU clusters into one atomic bootstrap transaction;
   - lets the Node Runtime converge node systemd Collectors/Agents by Fleet waves;
   - before Profile finalize, refuses non-terminal workflows or active workloads that still reference the old Profile;
   - runs the quick gate and produces structured evidence bound to the release ID, delivery, site/cluster identity, time and live release
     state SHA; the full verify that immediately follows reuses only the
     CPU role-split, per-GPU Executor verifier and `runtime_component_identity` checks the evidence explicitly covers; other checks
     still run, and when the evidence is stale or mismatched the full checks are rerun; the first bootstrap uses the same evidence protocol;
     if only
     `GpuFaultStoreIoRejected` remains, and every running CPU Pod has proven a complete, fresh zero-valued process view,
     waits up to 420 seconds for the old 5-minute rate window and Alertmanager state to clear;
   - before upgrade, freezes both the retained count of Executor internal errors and the latest error timestamp; the retained count is a
     gauge that drops as terminal commands are cleaned up, so the full verify judges whether this release introduced new defects by whether the latest error timestamp moved forward. `GpuFaultRemoteCommandExecutorInternalError` likewise fires on whether a new error appeared within the last 900 seconds,
     and no longer applies Prometheus `increase()` to the retained count;
   - starts the 120-to-300-second stability window only after criticals have cleared. In each round the read-only sampling of the CPU, each GPU cluster, the Store and AMP
     runs at most 8-way parallel, then produces a lightweight summary. Other
     criticals, new Pod restarts, non-terminal remote commands or wait timeouts still restore the online release according to `autoRollback`.

The plan shows the stable site identity directly via `site_identity.site_name`, `site_identity.aws_region` and
`site_identity.cpu_eks_arn`, and verifies that readable object with `site_identity_sha256`. The approval record also binds the registration anchor, original/target Profile versions,
normalized policy digest, template/snapshot digests and the live Profile SHA at approval time. The administrator must explicitly pass the reviewed
`plan_sha256` back via `deploy --approve-profile-plan`; the command compares the current plan inside the same file lock and only writes the
approval and continues the release once they agree. On a failed resume where the live baseline has not changed, the approval is kept; when the site identity, template, target content or live
baseline changes, the old approval is automatically archived as `SUPERSEDED` and re-review is required. Only after the release has completed
verify, stability and commit is the `CONSUMED` audit written and the active plan and approval deleted; it cannot be replayed.
Approval and release share the `state-dir` file lock; these JSON records must not be hand-edited or copied.
For the administrator operation contract see [Runtime Profile Change Approval](administrator-profile-change-approval.md).

The regional release diff compares the normalized policy digest, not the Profile file path or the raw YAML SHA.
When `source` switches from the template to the immutable snapshot, the field order changes or the registration anchor is replaced, as long as the normalized policy and
version are unchanged it is classified as `NOOP`. The raw source/template SHAs are still written into the release state, for audit
and old-state migration only.
If the site's `source` path has drifted but
the byte SHA of `<state-dir>/profiles/<current-version>.yaml` exactly matches the live release state,
the release planner may use that local immutable snapshot to restore a trusted baseline and atomically correct the site; when the snapshot is missing or the SHA mismatches
it is still classified as `UNKNOWN_BASELINE` and requires a stop, which cannot be bypassed through approval.

Training submission no longer fills in the Profile version by hand:

```bash
gpu-training-submit --site /path/to/site.yaml training.yaml
gpu-fault-workload-annotate --site /path/to/site.yaml training.yaml
```

When an explicit `--runtime-profile-version` is used together with `--site` they must match exactly, otherwise the command refuses to submit.

New configuration must additionally:

1. update `site.example.yaml`, schema parsing, the redacted audit digest and the negative tests;
2. regenerate `src/gpu_fault/data/env-inventory.json` and the two environment variable references;
3. refuse unknown spellings, cross-Region ARNs, leftover placeholders and unsafe Secret locations;
4. enter at least one observable check in `preflight`, `verify` or `status`.

The `value_kinds` in the inventory are the types the generator infers from the conversion at the read site (`int(...)`, `float(...)`,
comparison against `0/1/true/false/yes/no/on/off`); the five runtime processes validate values against it at startup and fail
closed. Therefore, when adding a numeric or switch variable, keep a single way of reading it and do not wrap the parse in
`except ValueError` -- a variable with conflicting read sites or a swallowed parse falls back to "untyped, unvalidated", and regenerating the inventory silently
yields one entry fewer instead of reporting an error. The enabling words of a switch also come from the code: a switch that only tests equality with `"true"` will not accept `yes`;
if it should, change the read site at the same time, not just the documentation or configuration.

`site.yaml` allows `spec.clusters: []`, because a completed site may deregister its last GPU cluster and keep
an idle control plane. That semantics cannot be used for the first bootstrap: the regional deployment state machine requires at least one GPU cluster when there is no release state or it is still in a
`bootstrap-*` recovery phase, and fails closed before the first Kubernetes apply.

Aurora credential refresh is cross-role configuration: `preflight/verify` must parse both the CronJob's
`GPU_FAULT_AURORA_RESTART_DEPLOYMENTS` and the live Role's Deployment
`resourceNames`, and fail closed when the two sets are not exactly equal. Allowing only ingress while missing worker
creates the hidden drift of "Secret updated, some consumers still using the old password". Since the control-plane review of 2026-09-08 (CP-3 /
H1) the refresher no longer rolls Deployments by default -- the three roles mount the whole `gpu-fault-aurora` Secret at
`/etc/gpu-fault/aurora`, and the connection pool re-reads the DSN via `GPU_FAULT_STORE_URL_FILE`;
`GPU_FAULT_AURORA_RESTART_DEPLOYMENTS` drives rolling only under the `--restart-deployments` (or
`GPU_FAULT_AURORA_REFRESH_RESTART_DEPLOYMENTS=true`) compatibility switch, but the RBAC consistency
requirement still stands as before, because the switch may be turned on at any time. The
`require_aurora_credential_mount` in `deploy/control-plane/tools/verify_control_plane_role_split.py` checks the three roles' volume/secretName/no items/mountPath/read-only/no
subPath and the source of `GPU_FAULT_STORE_URL_FILE`.

The versioned resources of the credential refresher are rendered uniformly by `regional_release_aurora_refresh`, and the first-install and verification Jobs
are likewise owned by the regional release. ARN bootstrap converges only the peripheral IAM/Pod Identity on both first and existing sites, and
does not apply the candidate ServiceAccount, Role, RoleBinding or CronJob ahead of time.
`aurora-refresh` is an independent release component: the Manifest,
Control Plane wheel, runtime image or read-only drift decides whether it runs; a pure refresher change does not roll CPU/GPU
and does not execute DDL. Only the master Secret ARN reference of `gpu-fault-aurora` is read for rendering; the database password is not
read into the ordinary release snapshot.

The public `preflight` stays read-only; it does not repair the refresher, create proof Jobs or bypass unknown states. The internal
`preflight --for-deploy` is called only by the authorized deploy driver and is not an alias of the public read-only preflight:
`regional_admin_commands.build_deploy_preflight_report` selects the refresher/monitoring items allowed to be repaired from the real release plan, and validates the non-Store prerequisites first; only when the plan includes `AURORA_REFRESH` is the workflow read deferred,
and that hard gate is restored after the refresh completes. A WARN on a monitoring item also only means it has been verified and the plan will repair it; it does not mean a read
failure is ignored or ADOT/AMP is updated early.

`src/gpu_fault_release/regional_release_prerequisite_repair.py` establishes an independent `aurora_prerequisite_repair` journal before any Store-dependent gate,
solving the ordering problem where the password has already been rotated and the old refresher cannot repair
itself. It first binds the candidate Manifest pin, CPU EKS/namespace, Aurora resource identity, original release
state digest and the complete snapshot of the old refresher, then persists `PREPARED`; it cannot apply the candidate first and capture previous afterwards.
When an update is needed it first installs the candidate CronJob with `suspend=true`, uses a one-off Job to prove that it supports the interface that forbids consumer restarts,
synchronizes `AWSCURRENT` and verifies a fresh refresh state, and only on success resumes the candidate schedule and records `READY`.
The one-off Job directly imports `aurora_credential_refresh.refresh_once` from the candidate wheel and confirms that its
`restart_deployments` parameter exists before running the entry point; the regression test executes the actually generated Python program and cannot merely
simulate Job completion, otherwise a runtime API rename could pass the tests yet fail at container startup.
This prerequisite transaction only prepares the CA, the candidate wheel and the refresher; it executes no DDL, business Pod restart, Profile or GPU rollout.
After the Store gate passes, the apply transaction takes over the original old-refresher snapshot and component progress, and must not mistake the repaired candidate for the old version.
On failure or incomplete cleanup the journal is retained; the resume validates the same candidate, database and baseline and refreshes again,
without reusing the old password-validity proof. A repair not yet taken over by the apply transaction can independently restore the original objects without marking the candidate as committed.

The bootstrap prerequisite repair separately saves the phase, cluster completion list, cleanup progress, cleanup failure information and the raw representation of the resume
phase; these type- and range-validated progress writes do not change the original full baseline digest, and the other fields
must still match exactly. An old first-install journal is compatible only when the full SHA matches exactly after the known initial progress has been restored;
unknown fields, apply completion evidence or genuine identity drift cannot be ignored.
Direct bootstrap also checks for prerequisite repairs not yet taken over before writing the resource and candidate checkpoints.

Switching to another candidate is allowed only for a bootstrap that has been cleaned up, not taken over and has no apply deployment completion evidence:
re-validate the original baseline, original candidate binding, CPU/namespace UID, database, cluster membership and snapshot,
first complete cleanup under the original Job identity and restore the full refresher, then atomically record the new candidate baseline and the new repair journal,
keeping the predecessor audit link. A restore or persistence failure keeps a resumable record and does not modify the old binding directly to impersonate the new candidate.
The new candidate must re-validate the Manifest pin, credentials and Store; Manifest drift within the same release is still refused,
and this path does not widen the supersede or commit-live permissions.

An upgrade candidate can likewise abort after the repair reaches `READY` and before the apply transaction starts (live 2026-09-18: the candidate was
refused at the workflow gate), and since the release id is a digest of the source tree the original candidate will never appear again; if the journal accepted only the original candidate,
a committed site would be locked forever by that orphan journal, and even `deploy --rollback` could not get in. Therefore
`validate_upgrade_repair_handoff` allows the **next candidate** to take over a non-bootstrap orphan journal, provided all of the following hold:
the current state minus the journal exactly matches the original baseline's full SHA (the apply transaction never started); both the state and the
baseline are `phase=complete`, `transaction_committed=true`, `release_lifecycle=COMMITTED`,
`commit_cleanup_completed=true`, and the live release id is not the orphan candidate; the binding is `bootstrap=false`,
`commit_live=false`, with an execution plan containing only the refresher; the CPU EKS/namespace/database identities match; the snapshot digest and content
pass the refresher snapshot validation; the Job ledger fields are complete. The takeover first re-checks cleanup under the original Job identity, restores the full snapshot of the original refresher and
re-verifies the database identity, then atomically writes a journal bound to the new candidate: it continues to carry the very first old-refresher snapshot (it does not recapture
the orphan candidate's objects), and `supersedes` stores the predecessor audit and the pre-takeover state digest -- the state the deploy driver pinned at
release-diff time is precisely the one containing the orphan journal, and the expected-state gate of the apply phase bridges on that basis.
A restore or persistence failure keeps the original journal and the restore is repeated next time. resume, supersede and commit-live still refuse
(`candidate binding differs`); orphans on a `rolled-back` site are outside this path.
`deploy --rollback` performs the same validation on this kind of orphan, independently restores the original refresher and deletes the journal, then stops with the original prompt
and does not roll back the application.

The first bootstrap has no business Pod available to run the Store probe, so
`src/gpu_fault_release/regional_release_store_proof.py` creates a read-only
database proof Job with the fixed candidate CPU image, executing `regional_release_store_probe.py` from the same trusted deployment source.
The Job keeps the `PATH` in the template that points at the component venv, so that `python -I` loads the same CPU component wheel;
this is also compatible with old images whose base Python contains only the shared dependencies. The default `PATH` of the new image already points to the single application environment,
yet the explicit component path is kept; the base interpreter cannot stand in for that environment to complete the Store check.
The proof binds the namespace UID, Aurora ARN/resource ID, endpoint, database, username, master
Secret reference, schema target, run ID, Job UID and completion time, and re-verifies the database identity before and after reading.
The Job references the DSN Secret and RDS CA only in the CPU namespace and does not mount a ServiceAccount token; the connection enforces
`verify-full`, a read-only repeatable-read transaction and connection/statement/lock-wait limits, and does not instantiate the business Store that would create tables automatically.
The first time it must actually prove that the database is uninitialized and empty: partially created tables that are declared but still have no rows are allowed; unknown
tables/views, RLS or unverified execution policies are not accepted. Neither Pods, a missing release state nor a historical `CREATED` marker can
prove an empty database. An ordinary bootstrap may resume the check against an already-initialized database only with a bound `bootstrap_database_origin` empty-database proof,
and each time it still freshly reads the schema version, the continuous migration identity and the two workflow/remote-command/Observation
states; non-terminal, unknown states or schema drift all block. Until the password and database safety proofs complete, neither schema
ensure nor business Pod startup may run; a historical origin is not a cache of the current "no active records".

When a managed proof Job fails, `regional_release_probe_job` collects diagnostics before the UID-scoped cleanup with a budget of at most 20 seconds
that does not exceed the existing deadline. It reads at most two Pods belonging to that Job and at most two
containers per Pod, re-verifies the Pod and Job UIDs after reading, and outputs only the allowed exit statuses, signals and Python error categories.
Raw logs, Pod specs and Secret contents are neither output nor written into the release state. An ordinary diagnostic read failure does not replace the
original failure; loss of supervision or a deadline anomaly still fails closed, and a cleanup failure cannot be recorded as completed.
The Job identity comparison allows the API Server to omit an empty `args: []`; neither representation overrides the image's default arguments;
adding, removing or replacing non-empty arguments still counts as an execution spec change and must be refused before the Job is created.
A Job's `Complete` status does not mean the database safety proof passed: the probe may catch an error and return `safe=false`.
`regional_release_store_probe` outputs only fixed execution phases and allowed error categories; the wrapper still validates the original
identity, time, schema and blocking-record conditions; on refusal it reports only whether fields exist and the already evaluated boolean conditions,
without outputting raw results, identity values or exception text, and without treating unevaluated conditions as failed.

Reinstallation after a managed `uninstall --cpu-cluster keep` completes takes a different path,
`regional_release_retained_database`: the completed uninstall record, the original Aurora incarnation, the site identity
and the new installation ID form an independent handoff, and the first use also pins a new namespace UID.
This path stores `retained_database_origin`; it cannot copy the old namespace's empty-database origin, nor does it claim the retained database
is empty. Every continuation re-reads the current safety state. Only the reviewed v17-to-v18 precheck may validate the full
historical prefix and the physical heap/columns without executing old stored functions, after which the normal schema Job ensures up to the candidate version,
and only then are business Pods allowed; older, future, missing or drifted schemas get no generic compatibility.

Rollback first restores the full objects of the old refresher according to component progress, then lets the old program synchronize `AWSCURRENT` in Secrets Manager,
and only then may the control plane be restarted. The old database password is never restored, nor is anything quietly repaired in verify via `set image`.
When an object did not originally exist, its absence may be recorded, and the candidate object deleted at rollback, only if the CPU namespace anchor has been verified.
An old transaction lacking a full snapshot cannot claim to have restored a modified refresher; automatic rollback refuses before the mutation,
and a validated old snapshot must be used for the restore. Only a successful CronJob query with an empty result means "not installed";
Forbidden, timeout, malformed JSON or identity mismatch all stop the release and cannot be interpreted as "skip the refresh".

### 4.6 Mail Notification Template Contract

Mail routing is declared by the site: `adminEmail`, `emailSender`, `emailRecipients[]` and
`emailSubjectPrefix` express the operations contact, the SES identity, the recipient list and the environment prefix respectively; the optional
`sesConfigurationSet` is the SES v2 configuration set name, rendered with the release config into the three roles'
`…-config-notification` ConfigMap (`GPU_FAULT_SES_CONFIGURATION_SET`); when not declared it stays empty
and does not change the notification digest of existing sites. The ARN-only
entry point allows passing the sender and multiple recipients independently; only when an old site declares no recipient list does it fall back for compatibility to
`adminEmail`.

The SES adapter uniformly adds site, Region, AWS account and cluster to the outgoing Subject and body, and does not require
every fault template to maintain these fields again. Each Builder accepts only the current event, Incident, Workflow and node
parameters; production templates must not contain concrete node names or test fixtures. When generating kubectl approval commands, explicit
context protection must be used. `support_case_draft` and `vendor-ticket-*` are only internal escalation records; only after the external ticket
adapter returns a real case ID may an AWS Support Case be claimed as created.

A change to the template body or Subject protocol must bump the corresponding `template_version` and update the Builder tests,
preview tests and deduplication keys in step. Fixed product copy may stay in the versioned template; site identity, node, execution result
and failed operation must be rendered dynamically.

New releases use release manifest schema v4, which covers both the runtime artifacts and the delivery identity;
source-only builds and old shared-image releases continue to support schema v3:

| Component | Allowed runtime surface |
|---|---|
| `control_plane` wheel | CPU ingress/worker/spool, schema and credential refresh |
| `executor` wheel | GPU Executor, Watcher, Resource Collector, Reconciler |
| `node_runtime` wheel | GPU node systemd Agent and Collector |
| `node_bundle` | Node install scripts, unit templates and the exact Node Runtime wheel |

In addition, it must record the rendered CPU/GPU/Schema/NLB/Observability digests, renderer inputs,
`uv.lock` and hashed requirements, the Node template SHA, and the immutable image digests of the CPU runtime, Executor,
Node Installer, node offline dependencies, DCGM and ADOT. Only a CI-generated image-set descriptor may mark the
Manifest as `deployable=true`; source-only artifacts cannot enter deployment.
The current runtime base layer is fixed to the immutable digest of UBI 9 Python 3.12 minimal; the image build stage installs the locked dependencies as
root, and the final runtime stage is fixed to UID 1001 and clears the base image's S2I default entry point.
Each CPU/Executor image installs all hash-locked dependencies and
its own component wheel only in the isolated venv `/opt/gpu-fault/runtime`; the default `PATH`/`VIRTUAL_ENV` point there, system-site-packages is not enabled,
and no packages are borrowed from the base Python via `.pth`. The base interpreter, standard library and bundled packaging tools remain, but neither
this solution's runtime dependencies nor the project wheels are installed there. The CPU's `/opt/gpu-fault/control-plane` and the Executor's
`/opt/gpu-fault/executor` are merely symlinks to their respective runtime directories, kept for compatibility with existing Manifests, probes and CLIs,
and do not constitute multiple package environments. The third-party dependency layer sits before the component parameters and wheel, so cache reuse across images is preserved.
Content validation actually starts the default Python and the `-I` interpreter at the compatibility path, loads the console entry points and runs `pip check`;
besides the module digest, it also requires that the dependencies, package search path and entry interpreter all belong to the same venv. Known old Dockerfile
digests continue to be validated with the old layout; old signed images are not required to grow new directories out of thin air, nor is the gate for new images lowered.
The two images each bind their own component wheel,
independently content-addressed, validated and reused; a CPU image change no longer automatically selects a GPU rollout. An Executor wheel change must still
perform the necessary CPU compatibility pin stage/finalize; the protocol window cannot be skipped just because the images were split.
schema v3 keeps the original shared-image interpretation; schema v4 lacking an independent Executor identity must fail and must not be guessed as the CPU image.
The previous snapshot, resume, rollback and the post-rollback administrator site all save or restore their respective image identities.
The node dependency target explicitly distinguishes candidate from previous via `NodeDependencyTarget`, carried through host preflight,
every wave and the steady-state Reconciler restore; even when two releases have the same bundle SHA, that cannot be used to select the
candidate dependency image. The previous's cluster, bundle and v4 dependency identities must match completely, otherwise rollback is refused.

The three wheels must come from the same build, but must not be installed interchangeably. Each component records the physical
`wheel_sha256` and the behavioral `module_digest`: the former identifies the actual file, drives artifact difference classification and the strict
artifact pin; the latter judges whether the implementation protocol is behaviorally compatible and serves as the compatibility pin. Even if the code
`module_digest` is unchanged, as long as the license, packaging metadata or other wheel content changes `wheel_sha256`,
it must not be classified as `NOOP`. Collector, Watcher, Installer and training annotation must still use the same
Runtime Profile version.

`verify/status` must check the component digest actually loaded by every running Pod; it cannot sample or look only at the
Deployment image/annotation. The CPU and GPU contexts stay isolated; the Pod list reads and
`kubectl exec` of all Deployments share one thread pool of at most 8 ways; even single-replica Deployments run in parallel across roles and clusters.
After all Pods complete, the output is sorted by Pod name. Any replica error is aggregated and then
fails closed; concurrency only optimizes wait time and does not reduce coverage.

Kubernetes JSON reads within the same `verify/status` or previous snapshot capture share one bounded
read snapshot. The health check runs `get deployment -o json` once per CPU/GPU context and
fills the list items into a thread-safe cache; subsequent reads of a Deployment by name use an independent deep copy directly, with no further remote
calls. The cache must not be reused across snapshots between different reports, different release phases or after any mutation.

The release build must be independent of the caller's shell `umask`. Component builds always use
`setuptools==84.0.0`, `SOURCE_DATE_EPOCH` and `umask 022`; the Python, packaged data
and LICENSE copied into the temporary project are uniformly `0644`; Node bundle directories are uniformly `0755`, regular files
`0644` and scripts `0755`. `artifact-check` must yield the same release ID under `umask 077` and `umask 002`,
and must not fabricate a spurious upgrade because ZIP external attributes changed.

The Agent and the Regional Executor each use an artifact SHA, compatibility digest and protocol;
the Agent additionally has a config digest. Together they use the two-phase required/compatible pin:

1. the steady-state old version stays required, and the candidate version is temporarily added to compatible;
2. before GPU apply, check that the candidate Executor's artifact, compatibility and protocol have been accepted by the
   required/compatible window; if not, fail immediately instead of waiting for the Deployment timeout;
3. roll the CPU and all GPU clusters, and wait for all Agents to converge exactly;
4. promote the candidate to required and clear compatible;
5. on failure, restore the previous pin, the corresponding component wheel, bundle, Profile and node artifacts.

The strict required pin must not be switched before installing nodes, nor may the compatible window be cleared while the Fleet has not converged.
When rolling back to a single-wheel release of **release manifest schema v1** (not PostgreSQL
schema v1), the old Agent compatibility digest is equivalent to the old artifact SHA; the old process's environment
variable allowlist does not recognize component pins, so
`filter_legacy_release_env.py` must structurally remove these env entries before apply, rather than relying on empty strings.

`RegionalRelease.noop` still validates and completes the Watcher's prerequisite state resources by default.
The BOOT-023 release-history acceptance explicitly passes `allow_prerequisite_repair=False`, doing only validation and history appends;
when prerequisite resources are missing it must fail, and must not build GPU ConfigMaps/RBAC inside an acceptance window that claims to change only history.
This parameter does not skip release validation, nor does it allow a non-NOOP diff.

`NOOP/CONTROL_PLANE_ONLY/DATA_PLANE_COMPATIBLE/FULL` remain administrator summary classifications; the actual execution
is decided by the component DAG. A pure Watcher or Collector Manifest change rolls only the corresponding Deployment; Node bundle,
template or Agent pin changes enter the Reconciler/Agent nodes; schema, CPU, endpoint, observability
and DCGM are each independent. New fields must register their dependency nodes; `FULL` cannot be equated with redoing everything.
A single rollout of a CPU role or a GPU Executor/Watcher/Collector is capped at 5 minutes; the Node Installer
Job has an active deadline and a host `flock`, and the Reconciler handles only the Fleet's current wave. A deterministic pin,
protocol or compatibility mismatch must fail before apply and cannot consume the full rollout window.
The Node Installer ConfigMap name uses the digest of the rendered Job YAML; node annotations and the Agent
heartbeat use the template source digest provided by the release. The two must not reuse the same self-referencing variable.
previous separately stores the `GPU_FAULT_INSTALLER_TEMPLATE_CONTENT_SHA256` the old Reconciler already trusted,
and uses it to validate the full bytes of the actual `job.yaml` in the ConfigMap as well as the old dependency image/inventory/read-only mounts;
the digest must not be recomputed over the mutable live Job and then authorized as the old version. An explicit template override must also provide that
trusted content pin, host preflight validates the template actually selected, and the managed entry point clears override variables inherited from the caller.

Each Fleet wave does not redeploy the Reconciler Deployment: the release first waits for the Reconciler
rollout to be healthy, then verifies that the bundle/template identity in the container env still equals the
identity recorded at this pause, then merge-patches the
`gpu-fault-node-installer-wave` pointed to by `GPU_FAULT_INSTALLER_WAVE_CONFIG_MAP` (`allowed-nodes`, `max-unavailable`, `generation`),
and reads it back to confirm the patch took effect. The Reconciler re-reads that ConfigMap on every reconcile round, so the allowed-node
set is hot-updated and is not part of the installer identity. The identity check happens before the patch, and the Installer Job's
cancel/retry is the same as on the original path. The Reconciler uses `strategy: Recreate`, and rebuilding it wave by wave would leave the cluster without an installer controller
between waves; only when the running Reconciler has no wave ConfigMap env (deployed before this
contract) does it fall back to redeploying wave by wave.

## 5. Kubernetes Lifecycle Contract

Runtime Deployments, DaemonSets and CronJobs must declare at the top level:

| annotation | Legal values and purpose |
|---|---|
| `gpu-fault.io/cleanup-phase` | CPU: `ingress/consumer/auxiliary`; GPU: `producer/executor` |
| `gpu-fault.io/cleanup-order` | Non-negative, stable shutdown order within the same phase |
| `gpu-fault.io/cleanup-action` | `delete/namespace/reset`, defaults to `delete` |
| `gpu-fault.io/deploy-mode` | GPU Deployments may use `regional_rollout/regional_reconciler` |

ServiceAccounts, RBAC, PDBs and ordinary Services are usually derived as `support/delete`; the CPU NLB Service uses
`nlb/reset`; ConfigMaps, Secrets and Namespaces are covered by the namespace lifecycle. Do not pick a wrong phase
for a resource just to make the generator pass.

Resource kind, name and scope are derived by the generator from the source YAML. The only update entry point is:

```bash
make PYTHON=.venv/bin/python deployment-contracts-update
```

It renders the three CPU roles, generates the build-time candidate `cleanup-inventory.json`, validates the allowlist and runs the contract
tests. The release entry point then refreshes the CPU/GPU
`gpu-fault-installed-resources` from the actual results of `kubectl get`. Decommissioning a resource must keep an explicit retired entry in the trusted-source inventory
until the live object is really deleted; you cannot simply delete the manifest and let a historical row in the mutable ConfigMap keep authorizing the deletion.
An entry's `provenance_sha256` is only for content-integrity diagnostics; it is not a signature or a proof of ownership. An unknown old row, even if
its self-digest is correct or the object no longer exists, stops before any registry rewrite, keeps the original record, and requires review first and registration of the decommission in the
trusted-source inventory. Resource identity comprises scope, kind, namespace and name; an object of the same name in a historical namespace
cannot be overridden by the entry of the current namespace.

Run before committing:

```bash
make PYTHON=.venv/bin/python deployment-contracts-check
```

The cleanup preflight also scans historical namespaces and cluster-scoped RBAC for unregistered
`gpu-fault-*` objects; a discovered out-of-band resource must fail closed and be classified first.
The namespace inventory only covers persistable API resources: objects of projected API groups such as `metrics.k8s.io` and `metrics.eks.amazonaws.com` (PodMetrics/NodeMetrics) are computed views without a uid, are not installed resources, and the inventory simply skips them instead of treating the missing uid as an incomplete identity. The controller mirrors of a registered Service, namely the `v1 Endpoints` of the same name and the TargetGroupBinding that the AWS Load Balancer Controller binds to it by `spec.serviceRef` and the stack label, are regarded as owned by that Service and disappear with it; the same kinds of objects pointing at an unregistered Service still fail closed. The schema Job is deleted by `ensure-postgres-schema.sh` right after it prints the completion log, so that a completed Job is not treated as an unregistered resource that blocks cleanup at uninstall.
Discovery and synchronization run through the same supervised command entry point; resource reads only accept explicit absence and never treat a tool, permission or
network error as non-existence. Reads are grouped by `scope + kind + namespace`, and kind, namespace and
name are verified item by item; objects of the same name are not merged across scopes or namespaces.
`collect_installed_resource_registry.py` also stores, under `gpu.by_context[context]`, the complete synchronization document of each original GPU
context, whose `.resources` lists only the trusted resources actually installed on that cluster. The context must be
non-empty and unique, and the check happens before the CPU registry is written. The top-level `gpu.resources` is an aggregated view and does not prove
that every item in it belongs to every GPU cluster; cleanup and residue verification run against the document of the corresponding context. When a multi-cluster live inventory
lacks the selected context, cleanup is refused; it cannot fall back to guessing ownership from the union.
The deploy tooling only builds a process-private `0600` temporary kubeconfig cache for the standard `aws eks get-token`: the return value must
carry an explicit, timezone-aware `expirationTimestamp` to be reused across requests, and it is refreshed 60 seconds early; without a validity period a fresh token is fetched for every request, and after a failed refresh the old token must not
continue to be used. Other exec plugins, and exec options that need `provideClusterInfo`
or do not meet that standard path, are still left to native kubectl, preserving
`KUBERNETES_EXEC_INFO`, interactive mode and the certificate-credential protocol, with no manual caching by the deploy tooling.
Authentication diagnostics do not echo plugin output, and the private configuration is cleaned up after completion or failure. This optimization does not change the trusted kubeconfig or the
release signature-verification boundary, nor does it provide a cross-command authentication cache.

The one-off CPU Jobs for the prerequisite refresh and the database proof are managed by
`src/gpu_fault_release/regional_release_probe_job.py`. Before creation it reads the actual UID of the managed Aurora refresh
CronJob and sets a unique `ownerReferences` entry, fixing `controller=false` and
`blockOwnerDeletion=false`: that reference hooks the Job into the garbage-collection chain of an already registered parent resource, without handing the proof over to the
CronJob as a scheduled run or an active Job to manage. Creation is refused when the parent object is missing or being deleted.
The dynamic identity and `owner_uid` are first written to `jobs` in the release journal, with state moving from `PLANNED` to `RUNNING` and then to
`REMOVED`. The server-side dry-run and the post-creation read-back both check the complete Job spec, accept only explicitly listed API
defaults, and verify name, namespace, run/spec markers and owner references. The run/spec digest, the Job UID and the
parent UID bound in the journal jointly constrain the completion proof and the deletion. When the creation ACK is lost, the read-back uses the same identity; it must not delete
blindly by name, trust a replaced Job, or delete an object whose owner has been rewritten.
While local command supervision remains trustworthy, success, failure and interruption all attempt a bounded foreground, UID-precondition cleanup
and confirm disappearance; once all completion proof is lost, issuing further cleanup commands is forbidden. A failure keeps the cleanup record, and the next run must finish the cleanup
before creating a new Job. Parent-resource garbage collection, the active deadline and the 300-second TTL are safety nets; they do not replace explicit cleanup and the
`REMOVED` proof. The database probe only outputs status, counts and identity digests; driver exceptions output only their type, and the DSN is never logged.
The resident refresh CronJob, ServiceAccount and RBAC are still managed by the existing installation resource registry;
the temporary Job covers its resource lifecycle through that registered parent object, without introducing a second hand-written resource inventory.
The discovery scope of `deploy/control-plane/tools/collect_installed_resource_registry.py` includes Jobs:
only when the apiVersion, CronJob name and UID of the unique owner reference and the Job namespace exactly match the currently registered parent object
is the Job regarded as a covered child resource. Orphans, old UIDs left after a parent of the same name was recreated, unregistered parents,
cross-namespace or multi-owner solution Jobs are all reported as `unregistered_resources`, on which the cleanup preflight
fails closed and requires classification first, with no automatic claiming or blind deletion. The existing name/namespace scope is not widened, and Jobs in customer
namespaces are not taken over.
Job template validation is not an independent proof against later Pod admission rewrites; the Pod admission
policy of the CPU namespace must still be trusted.

The current regional GPU rollout inventory must remain:

```text
completion-watcher.yaml
kubernetes-node-resource-collector.yaml
cluster-action-executor.yaml
```

`node-installer-reconciler.yaml` is marked separately as `regional_reconciler`. To re-enable any
DaemonSet collector, you must first remove the systemd producer with the same function, define migration/rollback and double-write detection;
merely adding the historical YAML back to the inventory is not allowed.

`completion-watcher.yaml` also declares
two ConfigMaps: `gpu-fault-completion-watcher-outbox` holds only the write-ahead log, and
`gpu-fault-completion-watcher-outbox-active` holds the steady-state attempt state. failure/terminal events
must first be written to the former, then sent to the control plane and deleted after acknowledgement; ordinary `workload-observations` are sent directly when healthy, and
only on failure are they written latest-wins by `cluster_id + attempt_id` into the same WAL and replayed in the next round, so that a brief control-plane
write failure does not leave the CPU-side Observation permanently in `RUNNING`. The `active-attempts.json` in the `-active` object
only saves active Observations when an attempt structure first appears or changes, and for each Pod stores only the attribution identity outside the spec
(uid/rank/node plus `gpu_uuids`, `container_id`, `cgroup_path`, `host_pid`, about 833 B/Pod;
the 900 KB ceiling holds about 1080 managed Pods, and attempts beyond that only lose restart memory without affecting delivery); after a Watcher
restart the remaining fields are rebuilt from live Pods, and when a Pod has already disappeared `STOPPED` is published per the cleanup timeout. The reason for the split is
the 1 MiB ceiling of a ConfigMap: when the steady-state state and the WAL share one object, observations of thousands of Pods squeeze terminal-event
writes into `CompletionOutboxFull`. When upgrading to this version, the Watcher's first read migrates the
`attempt:*` keys of the old object to `-active` in one pass (write the new object first, then delete the old keys; a failed delete is compensated by retries on later writes,
merging by the newer `observed_at`), and the first round after migration rewrites `-active` once per attempt. The migration is one-way: after rolling back to the previous watcher, the old object no longer has attempt state and restart memory can only be rebuilt from live Pods. The Role's
`resourceNames` must list both objects and must not grant `create` (RBAC cannot restrict create by name);
the objects are created by the publisher; when `-active` reads 403/404 the Watcher runs degraded (no restart memory), retries once every 60 s,
emits at most one ERROR per hour, and sets `gpu_fault_completion_active_state_unavailable` to 1. When the publisher
finds that either ConfigMap already exists it must omit that document from the rollout manifest; overwriting the
outbox/attempt state with default empty data is forbidden. Reverting to Pod memory or
`emptyDir` is not allowed. The `site.yaml` of every GPU cluster
must provide `agentEndpointAllowedCidrs`. That field goes into the corresponding
`RegionalClusterRegistration`, and heartbeat selects the CIDRs by the authenticated `cluster_id`, without merging them into a regional
allowlist. The node installer generates a local TLS cert/key automatically by default, advertises HTTPS, and carries the certificate in the signed heartbeat
for the Executor to pin.

When the Watcher produces a terminal event, the published Observation must be mapped to a terminal state at the same time:
`TIMED_OUT/FAILED -> FAILED`, `SUCCEEDED -> SUCCEEDED`, `STOPPED -> STOPPED`,
and unknown still-running ranks must not be retained. When the Store saves the terminal event it terminalizes the
Observation in the same attempt transaction; any `PENDING/RUNNING` Observation arriving afterwards, single or batched, is refused as an overwrite. The periodic cleanup
only selects contradictory records that are still missing or still active, so it is not starved for long, under the limit, by earlier terminal events that are already
consistent.

## 6. Node systemd Lifecycle

When adding a unit:

1. put the template in `deploy/systemd/gpu-fault-*.service|timer`;
2. `install-gpu-fault-collector.sh` copies or renders the template explicitly;
3. after installation the script scans the actual units and writes `/opt/gpu-fault/installed-units.txt`;
4. the Agent reports the complete list and its SHA-256 at startup or when the inventory changes, and stable heartbeats report only the digest;
5. after uninstall, confirm that no units, `/opt/gpu-fault`, `/etc/gpu-fault` or `/var/lib/gpu-fault` remain.

The Node Runtime uses content-addressed A/B slots:

- the candidate is first installed and verified into `/opt/gpu-fault/releases/<artifact-sha>-<dependency-sha>/venv`;
- systemd only executes `/opt/gpu-fault/current/venv/bin/*`, and `current` is switched by an atomic symlink;
- on the first migration from the old version, the real `/opt/gpu-fault/venv` is kept for the old installer's rollback;
- when a service fails to start or verify after the switch, the original `current`, configuration and unit enable/active states are restored;
- the old signed slot is kept until the rollback window ends and is not deleted early on the release path.

The dependency identity in `runtime-slot.sh` binds the `node-runtime.lock` bytes, the Python implementation/version, ABI,
platform, libc and the interpreter binary digest. The independent dependency venv is stored at
`/opt/gpu-fault/dependencies/<dependency-sha>/venv`, and slots of different application wheels reference the same immutable dependency layer through a checked `.pth`;
updating only the wheel no longer reinstalls the whole lock. A lock or interpreter change still creates a new layer.
`runtime_integrity.py` is executed by the host Python with `-I -S`; it first verifies, file by file, the RECORD, the version closure, extra files,
bytecode-to-source consistency, the interpreter and layer references, and only then allows the candidate Python to start; it never executes an unverified `.pth` first.
The RECORD text digest is only an additional seal and does not replace the actual file digests. A corrupted published dependency layer is refused for reuse;
a layer that may still be referenced by the active/rollback slot must not be rebuilt in place; only an interrupted build carrying `.building` and never published may be rebuilt.
An old slot lacking the new identity is not treated as a reusable candidate but is still kept for rolling back to the old release. When the active slot fails verification the process stops;
deleting or overwriting in place is forbidden.
The read-only candidate host preflight also checks the current rollback slot file by file through the trusted helper in the candidate bundle, including the old dependency layer,
the RECORD seal, the dependency closure and the entry modules; it verifies once more before the real installation, and both happen before the old services are stopped.
The old slot is verified by its own lock and interpreter identity; the candidate lock is not used to judge the old dependencies; the legacy standalone venv and old
artifact-only slots still support actual file and closure verification. Legitimate Python symlink paths are recognized by the same interpreter chain,
and the actual binary, version and isolation configuration are still required to match.

The candidate node environment fully validates the Collector plugins after RECORD, the dependency closure and `pip check` pass and before activation;
the candidate Executor image runs the installed environment's
`python -I -B -m gpu_fault.collectors_cli validate-plugins` in a network-less, read-only container. On-demand loading by the ordinary CLI
cannot replace this step. When an old node slot has no Collector registry interface, it is compatible only if no
`gpu_fault.collectors` plugin is installed, and rollback to the old version is not broken by the absence of the new CLI subcommand.
The image check first creates a container with an independent ownership label, checks the returned full CID against the private CID file, and only then starts
the check command; completion, timeout and main-thread interruption all clean up the container and anonymous volumes by the full identity and confirm the container no longer exists.
When the creation result or the cleanup result is unknown the gate fails, keeping the private ownership record for reconciliation; an identical name or label cannot replace
the original container identity, nor can that record be treated as proof of successful cleanup.

The node candidate preflight first renders one shared Job template, then `regional_node_batch` reads the private node snapshot
and the template in a single pass; `node_installer_rendering` shares the node-binding logic with the running Reconciler and verifies, object by object,
name, UID, InternalIP, instance type and the node-level Secret key. Only after all Jobs complete the server-side dry-run
does it start the read-only host preflight with at most 8 ways; on failure it stops submitting new nodes and cleans up the preflight Jobs already created.
The dry-run sends a bounded JSON List over stdin, at most 4 nodes, 8 Jobs and 1 MiB per batch, with a 60-second command budget;
admission runs at most 8 batches in parallel independently and shares the deploy-wide kubectl quota; the host execution window is still limited separately to at most 8 nodes.
Batching reduces CLI startups and repeated discovery, not the per-object admission checks;
if any batch fails no host Job may start, and batches already submitted must still be wound up.
When a creation request loses its ACK, it likewise reads the run-bound Job and confirms the node/artifact identity, deletes with a UID precondition,
and does not delete replaced objects; duplicate Node UIDs are refused before rendering. Failure logs are read in limited quantity and redacted.
It no longer starts a shell rendering process per node or re-parses the full node snapshot. The binder, the instance-type table and the DCGM
sampling period all enter the Node template identity; the binding logic cannot change while the old template pin is kept.

The schema v4 node dependencies are distributed from a separate OCI image and do not enter a ConfigMap. `node_wheelhouse` downloads, on a Linux
Python 3.12 build machine, hash-pinned Linux amd64 wheels per `node-runtime.lock` and `node-tools.lock`, the latter including py-spy; the inventory binds the platform, the locks and each file's digest/size, and its digest enters the signed
delivery. The non-root initContainer only copies the wheelhouse into a size-capped emptyDir and mounts neither the host nor
Secrets; the Installer uses it inside the chroot through a read-only mount. Both the read-only host preflight and the real installation check the
inventory, every file, the host platform and the locks in the candidate bundle, and then install with `--no-index --require-hashes`.
py-spy keeps its independent binary digest check. A first deploy still needs to pull the OCI image and install the local wheels, but no longer contacts PyPI
per node; subsequent deploys reuse the original immutable dependency layer. Rollback uses previous's dependency identity and cannot hand the candidate wheelhouse
to the old bundle. The original schema v3 rollback path stays compatible and does not fabricate an offline dependency image that the old version never declared.

EFA driver remediation stays enabled by default. The Installer Job's
`GPU_FAULT_ENABLE_NODE_EFA_DRIVER_REMEDIATION` accepts only `true` or `false`; an explicit `false`
passes `--disable-efa-driver-remediation` and removes the `REMEDIATE_EFA_DRIVER` capability, and omitting the
enable parameter does not fall back to the installer default. A collector-only installation still must not start the Node Agent.
The node action ledger is retained for `2592000` seconds (30 days) by default, and the help text matches the actual default.

Run at least:

```bash
.venv/bin/python -m pytest -q \
  tests/node_agent/test_node_deployment_*.py \
  tests/node_agent/test_service.py \
  tests/fleet/test_installation_inventory.py
```

### 6.1 Node Action Key Preparation and Synchronization

`deploy/node/provision-node-action-keys.sh` calls
`provision_node_action_keys.py` in the same directory and runs in the trusted deploy-host Python environment, adding no administrator dependency to the Node Runtime.
The Node bundle must carry both the wrapper and the helper; `TaskInputSpec` binds both, together with
`src/gpu_fault/admin/node_key_proof.py` and the target cluster, into the `node_keys:*` task digest, and the helper also
enters the release's `node_template_inputs` identity; you cannot update only the script while keeping the old checkpoint or template pin.

Secret reads use a supervised private JSON channel that distinguishes present, confirmed absent and read failure; only a successful and empty
`--ignore-not-found` result allows creation. Actual values are handled only in private memory/stdio, and identity verification covers
the Secret type, namespace/name, UID and resourceVersion. SHA-256 is computed over the raw bytes after base64 decoding,
without stripping first; the raw JSON, keys and fleet master must not enter logs or ordinary artifacts.
The GPU key set must be exactly equal to the currently verified target nodes, while the CPU key map is the site-wide union. The CPU merges the target entries with a
UID/resourceVersion-protected CAS, preserving unrelated keys and metadata; an existing legitimate randomly
rotated GPU value is reused as is, and v2 keys are derived only for missing nodes. A read failure, a recreated Secret of the same name or an unknown CPU key conflict
can never authorize an overwrite; identical key bytes by themselves do not prove that a NodeName belongs to that cluster.

`gpu-fault.io/node-action-key-rotation` on the GPU Secret is the resume marker of a partial rotation, not a proof of membership
ownership, and not a new resource. The marker binds the cluster, Node UID, CPU Secret scope/UID and the byte digests before and after rotation;
the presence of any marker must enter ensure, even if the CPU/GPU digests are already identical. Before completion it re-verifies the current
Node UID set, the CPU mirror value and the GPU identity, then removes the marker by CAS; it cannot regenerate random keys after a CPU write failure.
Each CAS phase gets at most 3 attempts, 20 seconds per command, 15 seconds per Kubernetes request and a 300-second total helper window, still subject to the parent
deadline. The fresh NodeName-to-UID binding passed in by join and the CPU compensation digest guard are described in §8.1.

### 6.2 Node Key Custody

custody is forward-looking forensics over the trusted delivery path above, not a second key synchronizer. The administrator procedure is in
[Administrator Operations §6.2](administrator-operations.md#62-node-key-custody); the complete signature schema and proof boundary are in
[Node Key Custody Evidence Contract](../components/node-key-custody-evidence.md).

The public entry point is `gpu-fault-admin node-key-custody configure`, which explicitly requires `--state-dir`, `--file`
and `--trust-sha256`. `src/gpu_fault/admin/cli.py` parses the arguments, and
`src/gpu_fault/admin/node_key_custody_admin_config.py::configure_admin_custody`
verifies while holding the site mutation lock and writes the private `node-key-custody/registration.json`. The selection file must be a
controlled `0600` file; `AdminCustodySelection` requires a non-empty mapping of complete GPU EKS ARNs whose values are `null`
or request paths, `trust` and request paths are resolved relative to the selection file, and `allow_staging` defaults to `false`.
There is no environment-variable opt-in; when unregistered the original synchronization behaviour is kept, and when the registration directory exists but the registration is corrupted/missing it must
fail closed and cannot fall back to legacy.

Registration pins the content identity of the selection, request, authorization, trust and its public key, manifest and predecessor chain.
Loading and consumption recompute and compare; replacing the content at the same path also requires explicit re-registration. The externally approved trust
SHA-256 is independent of the evidence provision; the P-256 public-key identities of approval, provisioner and witness differ from each other,
and none may reuse the release signing key. The provisioner cannot obtain approval signing rights; all forensic signing rights
stay in their respective controlled deploy/witness roles and cannot be handed down to GPU nodes or runtime components.
`configure` itself generates no authorization, does not call KMS Sign, creates no namespace and rotates no Secret.

| Phase | Actual entry point and evidence | Boundary that must hold |
|---|---|---|
| Explicit preparation | `node_key_custody_admin.provision_admin_custody` generates a `CustodyPreparation` through the current binding when the target request is `null` | Only the execution path of a normal deploy/join may write the preparation; `authorization=false`, and it cannot pose as an independent approval |
| Resume after independent authorization | `configure` again to register the signed request, rerun the same deploy/join | The real helper is admitted only after the current release, source, cluster/namespace/Node UID, master source and request snapshot all match |
| Trusted delivery | `Started` precedes the key write; `Completed` is signed after the actual CAS and read-back succeed | genesis requires that the target key did not exist before; rotation requires an activated predecessor and advances only one node, and cannot backfill history from the current identical key |
| Independent runtime witness | The deployed HTTPS Agent receives a fresh challenge and an independent witness signs `Activated` | The initial activation is only a prerequisite, not a complete AUTH-015 rotation PASS; the Secret completion state does not replace endpoint verification |

`src/gpu_fault/admin/node_key_custody_admin.py::write_preparation` outputs
`<state-dir>/node-key-custody/preparations/<statement-sha256>.json`, binding the signature-verified schema v4
candidate release, site/Region and CPU/GPU EKS identities, the cluster/namespace UIDs of both sides, the complete NodeName-to-
UID set, the CPU master Secret UID/key/byte digest, and the producer/witness source identities.
Resource preparation and builds still run under normal authorization; the preparation pause only guarantees that the target key has not yet been written and does not claim that the whole deploy is read-only.
The independent authorization binds these facts, the purpose, the predecessor and a window of at most 24 hours, and the real write still re-verifies the window phase by phase.

`src/gpu_fault/admin/bootstrap_services.py::provision_node_action_keys` enters the custody branch only when it receives a
`custody_context`, and first checks the context against the actual call parameters; unselected targets keep the
original synchronization implementation. The checked `AdminNodeKeyContext` is constructed by bootstrap and join, not guessed from the CLI environment.
The helper is still `deploy/node/provision-node-action-keys.sh` and its Python implementation; the administrator invocation adds
`--custody-request`, `--custody-trust-sha256` and `--custody-input-sha256`.
The last one pins the complete input content before and after the subprocess handoff; it is an integrity guard, not a signing authorization.
Sensitive parameters are passed only as controlled file/configuration references; the raw master and keys must not enter argv, ordinary artifacts or logs,
and the GPU still receives only node-level keys.
The completion receipt also pins the digest of the CPU's unrelated key set; internal serialization is not a cross-deploy-host lock, and the silence windows of other
writers must be coordinated. CAS preserving unrelated values does not make a concurrent digest change automatically part of this forensic record.

`src/gpu_fault/admin/node_key_custody_admin_probe.py::verify_completed_custody`
is the signed-receipt probe; it verifies the complete chain and the current release/UID binding, privately reads the CPU/GPU Secrets,
compares the actual byte digests, Secret UIDs and the absence of a pending rotation, and does not pass on a complete set of key names alone.
`provision_admin_custody(..., probe_only=True)` and the public preflight can neither call KMS Sign,
create a preparation nor execute the helper; missing evidence requires preparation/reconciliation. A valid completion chain can satisfy the probe again after the outer bootstrap
checkpoint is lost, and an old shape-only checkpoint cannot skip custody.
`Completed.runtime_activation_proved=false`, and the administrator task result also explicitly returns
`runtime_activation=NOT_PROVED`; configuration or delivery success cannot be upgraded into a runtime proof.

A subsequent rotation on the same release must still go through
`node_key_custody_activation.CustodyActivation`. Before the key write it records the binding snapshot,
runs the safety gates and drains the designated Agent, taking over the Installer wave of the single node; only then does it deliver the key,
refresh the Executor signature, install the designated node, refresh the CPU verification process, observe the fresh Agent and restore the wave.
The intent is persisted before every mutation, with a fixed deadline not exceeding the independent authorization expiry and the 1800-second budget;
the actual key writer carries the path and digest of the persisted activation state and, before every Secret commit, re-verifies
the same deadline, the monotonic clock and the managed wave UID/complete data. Resumes after FENCED, the CPU/GPU rollouts
and node annotation commits also re-verify the wave; an early restore to the original open wave cannot serve as a proof of ownership.
Only the last UNFENCED phase that has already started may confirm its own lost ACK, without reopening earlier mutations.
An interruption only advances along the original journal or stays in manual reconciliation; it is not skipped by NOOP, and keys are neither regenerated nor automatically written back to the old key.
The read-only probe of a rotation also checks that activation state; Secret completion by itself is insufficient.
The completed operation state is `DEPLOYED_NOT_WITNESSED`, which does not change the evidentiary meaning of the signed completion above;
an independent witness must still verify the real Agent's new key, old key and sibling refusal. The new node activation
annotation is registered in the existing Installer annotation cleanup inventory, without adding another resource source of truth.

When `Started` exists without a valid completion chain, the Secret creation ACK is unknown, a read fails, the source is replaced or the UID drifts,
the intent is kept and `CustodyReconciliationRequired` is raised; identical current bytes and a legacy rotation marker can neither
authorize reopening the writer nor manufacture a completion receipt. `cluster_join` records
`CUSTODY_AWAITING_AUTHORIZATION`/`CUSTODY_BLOCKED` respectively and keeps the namespace and inputs, without running the automatic compensation of an ordinary
failure. An unstarted request may be explicitly re-registered; a started request can only be continued by a rotation with a new independent authorization, and its signed completion and activated predecessor must match the originally registered transaction. Registration cannot remove a cluster,
replace the trust pin or implicitly downgrade; v1 offers no cancellation of unknown writers, retroactive forensics or automatic migration across releases/identities.

The independent witness uses the release-bound Agent, a fresh heartbeat, the TLS certificate and actual protocol results to prove that
the new key authenticates, that the old key/sibling key are refused on both the command and result-query paths, and that the sibling runtime identity is unchanged.
The challenge uses a non-admissible time window and the forbidden `FREEZE_EVIDENCE`, the verification command never exists,
and it dispatches no physical action, rotates no Secret, builds no host probe, and does not reboot or restore.
It still requests an independent KMS signature and writes a public receipt, and cannot bypass the original live runner's plan, target, source, connection identity and maintenance-window approval.
Even if the initial witness signs `Activated`, the complete `GF-REGIONAL-AUTH-015` remains FAIL; an authorized single-node rotation must be deployed and
witnessed; see [AUTH-015 witness](../components/node-key-custody-evidence.md#auth-015-witness).
The standalone decommissioned key file is only for negative challenges and stays at a controlled `0600` path on the trusted deploy host; it does not enter the receipt directory,
the release or the GPU. A signature does not prove unobserved host behaviour; hermetic tests do not constitute LIVE installation/KMS IAM evidence.

The custody modules belong to the deploy-host import closure and enter the delivered Node template/source identity;
they are not added to the Control Plane, Executor or Node Runtime wheels. Related source changes must re-verify explicit
registration, unsigned preparation, resume after independent authorization, the signed probe, old checkpoint refusal, the different verdicts of the initial activation and
a complete rotation, and that unregistered sites keep the original behaviour.

## 7. AWS Resource Lifecycle

A normal first deploy creates Aurora, AMP, SNS, the runtime/cache ECR, PKI, IAM and the solution
security groups through the ARN entry point, and writes every resource into the bootstrap state and the installation registry. The ARN entry point is the
only writer of site AWS infrastructure in this solution.

The ARN bootstrap sets a fixed lifecycle policy for the mutable BuildKit cache ECR: it deletes only untagged cache artifacts older than 7 days and keeps the current tagged cache manifests. A missing policy is filled in idempotently; when an existing
policy disagrees with the source of truth it fails closed and is not overwritten online. The policy digest and retention days are written to the bootstrap state, and
the runtime ECR does not receive that cleanup policy.

The SNS email subscription uses an independent persistent state machine. The Topic carries a random generation tag, so after a Topic of the same name is
deleted and recreated the old pending checkpoint is invalidated even if the ARN is unchanged. Before calling `Subscribe` it first writes
`REQUESTING`, and on success saves the real ARN returned by `--return-subscription-arn` and a 48-hour confirmation window;
the same generation must not call again before confirmation or window expiry. When `list-subscriptions-by-topic` finds one
confirmed entry it is reused directly; multiple confirmed entries for the same mailbox, or confirmed and pending coexisting, fail closed,
and the administrator explicitly keeps one before rerunning. The email subscription enters the installation
registry as a `DETACH` resource; ordinary deploy and rollback must not delete the Topic or the subscription.

The Kubernetes cleanup registry does not cover Aurora, NLB, ACM, Route53, SNS, AMP, IAM, SG,
subnets or other AWS resources. A new resource must create an `InstallationResource` record containing at least:

- a stable `resource_key` and ARN/ID;
- Region and account;
- `CREATED/EXTERNAL` ownership;
- `DELETE/DETACH/PRESERVE` delete policy;
- the other `resource_key`s it depends on;
- non-sensitive attributes and status.

Special case for the Route53 VPC association row: a resumed cold deploy that lost the associate receipt records the
GPU VPC association already present on its own zone as `EXTERNAL/PRESERVE` (bootstrap does not fabricate a creation proof). The zone is always created by the solution
and is a deletion target, and the association cannot outlive the zone, so uninstall detaches such rows as `DETACH` before deleting the
zone instead of refusing with "retained resource depends on a deletion target"; the native CPU VPC
association never enters the registry, so the last VPC of the private zone is never detached.

`site_id + resource_key` is the stable key; provider/type/ID/ARN/Region/account/ownership/
delete policy/dependencies belong to the immutable identity. attributes forbid secret/token/password/
credential-class keys; the registry is not a credential store.

Registry API compatibility access is provided by the self-contained script in `src/gpu_fault/admin/resource_registry_scripts.py`,
which recognizes an old API or a retryable 503 only when explicit machine return codes and stdout markers match; a 404/503
string anywhere in stderr is not allowed to trigger the fallback. The snapshot is sent over private stdin and does not enter the command arguments. The direct Aurora
backfill for old control planes likewise makes an atomic conflict decision by the immutable identity, and any conflict rolls back the whole batch; an unconditional upsert rewriting resource IDs or
delete policies is not allowed. The connection prefers the currently mounted `GPU_FAULT_STORE_URL_FILE`; when the file is configured but unreadable it
refuses to fall back to the old password from startup; only old versions without that mount keep using the original environment reference. Connections, statements, locks and the outer command
all have time limits, and database errors output only their category. This compatibility path only writes registration records and executes no DDL.
A successful exit is by itself not a proof of synchronization: the complete resource set returned by the API must match this snapshot item by item, and the direct SQL
backfill must return the exact row count; a missing, extra or mismatched confirmation all fail. The effective
delete policy computed by uninstall from the approval mode exists only in the execution plan and the audit; it is not written back to the resource's immutable ownership, delete policy or dependencies;
registry updates only advance the corresponding state and cannot use the historical `REUSED/PRESERVE` compatibility handling to loosen identity constraints.
The local snapshot's file digest and `source_sha256` can only detect content corruption; before deletion the site identity must still be bound and the
real resource ownership verified; a new bootstrap record also checks the CPU identity in `initial_deploy_target`.

After deploying or taking over legacy resources of the same site, `gpu_fault.admin.resource_registry`'s
`sync_installation_resource_snapshot` (kubectl exec into the CPU Pod to call
`/v1/installation-resources/sync`; it falls back to a direct write only when the controlled script above reports the exact old-API or retryable-503 protocol)
synchronizes to Aurora
`gpu_fault_objects(kind=installation_resource)`. Administrators do not hand-write these
fields in `site.yaml`, and must not add a second AWS inventory file.

A new AWS foundation resource must be hooked into the ARN bootstrap and emit the same site contract. Every application-owned AWS resource must
also be hooked into:

1. ARN discovery, deterministic naming, site ownership tags and `site.yaml` generation;
2. ownership tag verification and fail-closed on external same-name conflicts;
3. `preflight/status/verify`;
4. `remove-cluster` (if cluster-specific);
5. `uninstall` dependency ordering, real deletion queries and final residue verification;
6. the registry snapshot, SHA-256 and the final Aurora deletion phase.

The unified uninstall first exports the Aurora registry, then marks the entries `DELETE_PENDING`. Only after Kubernetes, non-Aurora
resources and CPU retention or deletion are all verified complete does it save the pre-Aurora cleanup proof and enter
`READY_TO_DELETE_AURORA`, and only then does delete mode allow deleting Aurora in the actual order of readers, writers, cluster and dependencies;
database deletion cannot run in parallel with the preceding verification. The final
`installation-resources-final.json` must contain only `DELETED/DETACHED/PRESERVED`, with
`delete_policy_residuals=0`.

Cover at least:

```text
tests/admin/test_admin_resource_registry.py
tests/admin/test_admin_cluster_removal.py
tests/admin/test_admin_uninstall.py
```

`cleanup_state.py` is still usable for break-glass phase state, but it cannot replace the Aurora
`installation_resource` registry.
## 8. Administrator Interface Completion Criteria

A new feature cannot ship as nothing more than hand-run `kubectl`, `aws` or `curl` snippets. The completion criteria include:

- the normal release enters the regional orchestrator from `site.yaml`;
- `preflight` checks inputs, identity and dependencies before the environment is modified;
- `deploy` is idempotent and has persistent state plus a recovery or rollback boundary;
- `verify` independently and read-only validates real functionality, not just Pod Ready;
- `status` outputs structured health, release/Profile/pin and failure reasons;
- cluster-specific resources go into `remove-cluster`, site-wide resources go into `uninstall`;
- dangerous operations have dry-run, approval, success criteria, rollback and evidence;
- new credentials have creation, rotation, revocation, least privilege and log-redaction definitions;
- administrator documents describe only decisions and formal entry points; internal details stay in the developer documents.

The formal entry point for adding a GPU cluster is:

```bash
gpu-fault-admin join-cluster \
  --state-dir /secure/gpu-fault \
  --gpu-cluster-arn <gpu-arn>
```

When the first four-parameter deploy receives several GPU ARNs, the first cluster is used to establish and verify the baseline and the remaining clusters enter a bounded
batch join: only one baseline verify, one merged candidate preflight and one final full verify are executed; discovery and
the per-cluster preparatory work (IAM role, network allow-list, node key) run at most 4-way parallel, while the actual rolling parallelism is taken from the site's
`spec.release.upgradeMaxParallelClusters` (default 1, upper bound 8); join and the shared release upgrade use the same
knob. The site-level Installer budget is fixed at 64 nodes; the per-cluster wave ceiling is further taken as `floor(64 / cluster parallelism)`,
and the minimum is taken with the scale, the explicit `upgradeMaxUnavailable` and the failure-domain limit; the first wave remains 1, and node waves remain
constrained by the release engine's adaptive max-unavailable.
Each cluster still has independent state, a `PENDING -> ACTIVE` transaction and compensation; the formal site,
release state, bootstrap state and the Aurora resource registry are committed incrementally cluster by cluster inside the membership lock.
Multiple `join-cluster` CLIs on the same site are serialized by a full-transaction file lock and do not add up the parallelism of separate processes
to break through the site parallelism and global budget.
The batch parallelism above applies to the default path with no registered custody; a registered site has `cluster_batch_join` call the locked single join
one target at a time and stop at the first custody preparation or blocking condition without continuing to later targets, see §6.2.
The identities and order of the first target GPUs are persisted to the bootstrap checkpoint; on a failed rerun, the current members are allowed to be a monotonic subset of the original targets,
bootstrap only reconciles members that are already managed, and missing members continue through independent join. The original commitment counts as complete only when every original target
is managed, or a missing target has a `COMPLETED` removal journal with exact EKS/HyperPod/CPU identity that matches the current site/release;
a partial site or a merely same-named `removed_clusters` row cannot release the commitment.
After the commitment completes, the checkpoint is set to `COMPLETE`; from then on deploy only checks that the requested set is the managed set or a superset of it and no longer
compares against the birth targets; new targets awaiting join form the next checkpoint, and omitting a still-managed member still requires
`remove-cluster` first. A malformed checkpoint, identity drift, or `COMPLETE` with a missing site are all refused; there is no restart by deleting the old record.
The checkpoint also binds the targets, the release Manifest, the AdminConfig and the bootstrap implementation digest. State schema v3 stores an
independent input digest per task and invalidates only tasks whose inputs or implementation source changed: the digests of tasks that do not depend on the release are bound after scope discovery and
before the task graph reads the checkpoint (`bind_bootstrap_inputs(..., release=None)`); the digests of `release`, `monitoring_install` and
`aurora_refresh` are bound later by the same function with the actual signed candidate once the signed release build finishes; each binding
only adjudicates the tasks it names and never clears tasks that this process has already completed. Pod Identity Agent, Load Balancer
Controller, Control Plane/Executor IAM role, notification and monitoring resources still run a read-only live probe even on a checkpoint hit.
Only when the probe confirms that the addon, IAM trust/policy/tag, Pod Identity association,
Helm values, SES Secret, AMP/SNS and subscriptions are all consistent is the checkpoint reused directly; only when drift is found
does the task enter ensure and allow mutation, so a healthy site does not repeat apply/update/Subscribe on every deploy.
AWS base resources, cluster join and platform prerequisite tasks are scheduled in one dependency graph (`bootstrap_tasks.TaskSpec`,
`bootstrap_access.ClusterAccessPlan`, `run_parallel`, a single 8-thread pool). A task's ensure,
probe, re-verification policy, dependencies and failure policy live in the same TaskSpec; duplicate names, unknown registrations and cyclic dependencies
are refused before execution. Only tasks that currently have a free slot and whose dependencies are ready are submitted to the pool; nothing is queued in an unbounded pending queue.
Input identity is defined uniformly by `bootstrap_task_inputs.TaskInputSpec`; a TaskSpec must declare a fingerprint
or an explicit reason for re-verifying every time; the checkpoint computes digests from the same directory and no longer maintains a second task-name mapping.
A graph whose inputs are not yet bound may not reuse the checkpoint. The render, preflight, registry, artifact and credential-refresh interfaces use the explicit
RegionalRelease type; external JSON is still validated at the boundary, and a missing dynamic method cannot be used to skip the drift check.
Legacy Aurora configuration is first migrated to the normalized AdminConfig, then the early inputs are bound and the build is started; the dependency graph consumes the capacity values from that same
parse, so that a migration changing `desired.json` does not make the two input identities conflict. A task this same process has just completed
must also re-check its digest; "already executed this run" is not an exemption from invalidation.
As soon as the CPU kubeconfig/namespace/base Secret are complete, `aurora` is allowed to create the cluster and instances, no longer
waiting for the GPU kubeconfig/namespace or Pod Identity; writes to the shared GPU kubeconfig remain serial.
`nlb_network`, `pki`, monitoring resources and independent IAM roles may overlap with these join chains. The CPU role and LBC
wait for CPU join and Pod Identity; the ADOT writer role still waits for the corresponding Executor OIDC and AMP workspace.
`monitoring_install` keeps its original checkpoint name but only prepares the site-level amp-writer role and Pod Identity,
waiting for CPU join, Pod Identity and `monitoring_resources`, no longer for `release`. The monitoring resources task also converges the SNS AMP publish permission through
the shared `amp-sns-publish-policy.json`; it keeps the other statements unchanged, verifies
account, workspace, topic, site tag and generation, does not write when already consistent, and stops when it observes a concurrent change.
`grafana_install` waits only for
`monitoring_resources`, not for the image build or the ADOT install; the `aurora` task creates instances through
`ensure_serverless_instances(wait=False)`, still constrained by the real primary and cluster readiness barriers:
the writer is created first, and the reader is created only after the actual primary and the cluster are confirmed available. `wait=False` only hands the final
two-instance readiness check to `aurora_ready`; it does not skip the preconditions for creating the replica.
The actual primary is read before a replica is added; after a resume or a failover the role is not guessed from the instance name.
While the first instance is still `creating`, waiting is allowed for its member record to turn from an unspecified writer into the unique primary;
after the instance becomes available, member information is synchronized at most 6 more reads at 5-second intervals, bounded by the parent deadline.
An available cluster with no primary, unknown members, several primaries, or an identity that changed after waiting still refuses to create the reader.
This is the [AWS prerequisite for adding an Aurora Replica](https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/aurora-replicas-adding.html)
and cannot be bypassed with a parallel waiter. The RDS instance and cluster waiters explicitly use a 2100-second budget, covering the default 30-second interval,
60 checks and API overhead, and no longer fall into the generic 900-second cap. `bootstrap_aurora` reads the acceptors, interval and attempt cap from the local botocore waiter
model; each describe is admitted separately through the supervised CommandRunner, and the polling interval
does not consume AWS CLI quota; no SDK client is created behind the runner's back. The actual wait is still the smaller of the operation budget and the parent task's remaining
time, and instance waiters in the thread pool inherit the same parent deadline. Each request, and the success judgement after it returns,
re-check the remaining time; terminal failure, error responses and identity mismatches stop immediately.
`aurora_ready` waits for `aurora` and CPU join, verifies that both instances are finally available, reads the master Secret with bounded retries (18 x 10 seconds)
and then writes the `gpu-fault-aurora` Secret;
managed bootstrap explicitly uses `admin.rds_ca_bundle.RDS_CA_BUNDLE_PATH` to generate a
`sslmode=verify-full` DSN, consistent with the non-optional read-only CA mount of the CPU, the schema Job and the refresher,
neither depending on the deploy host exporting the CA path beforehand nor accepting an inherited environment override of that path. The CA install still checks the official bundle's
fixed digest; a missing mount or an untrusted certificate still refuses the connection, and a legacy manual call without the CA input still refuses to generate the DSN.
`aurora_refresh` likewise only prepares the refresher's IAM access, waiting for `aurora_ready` and Pod Identity, not for
the candidate wheel or image; the first-install CronJob/ServiceAccount/RBAC are also deployed by the release.
By default `node_keys:*` waits for CPU/GPU join and the fleet master. When custody is explicitly registered, all node key tasks sharing the CPU
key map (including unselected legacy targets) must additionally run serially after `release` completes;
unrelated IAM and resource preparation still runs independently in parallel. Tasks whose dependencies failed are recorded as skipped rather than failing individually; a join or release-build failure also
stops independent tasks that have not yet started, while started tasks must finish. `aws-infrastructure-ready` is recorded when the base subset
completes and `platform-prerequisites-ready` when the whole graph completes; in the subsequent release rollout,
the first NLB Service is created before the schema/CPU deployment so that AWS's asynchronous preparation overlaps the CPU work;
healthy targets, TLS and DNS verification and CNAME publication must still wait for the CPU to be ready.
Once the CPU, Profile and endpoint gates pass, the first-install control-plane monitoring configuration runs in parallel with the GPU bootstrap;
`regional_release_bootstrap.bootstrap_observability` waits for both sides to
finish before verification, failure persistence and cleanup. The parallel phase uses the installer's leave-as-is mode and does not read, publish or delete the expected-collector
rule; that rule is published separately only after both the GPU and the control-plane monitoring succeed, skipping a definition that is already ACTIVE with identical content.
The GPU checkpoint is still written by the original calling thread; configuration completion does not mean the Collector is already healthy.
The signed release build (ECR repositories, release gates, wheel, runtime image) starts on an independent thread immediately after scope discovery
(`release_repositories.SignedReleaseBuild`), in parallel with this graph: it only reads the source snapshot, and the AWS base
resources do not read it; the `release` dependency node waits for the build to finish and binds the candidate inputs. `monitoring_install`,
`aurora_refresh`, `aurora_ready` and, when no custody is registered, `node_keys:*` start immediately according to their own resource dependencies
and do not consume build outputs. Custody node key tasks must take the signature-verified release and cannot reuse that default exception.
The input digests of infrastructure tasks are bound before the build is started/scheduled; after a dependency task completes, whether the checkpoint is
valid is re-read, and the old completed set copied when scheduling started cannot be reused. The release consent check runs once the first time the
build result is taken, so a non-consented release is refused before any platform task uses it. A first site install
thereby overlaps the two longest stretches, the Aurora wait and the cold build.
`SignedReleaseBuild` finishes through `finally` on the success, precondition-error, graph-failure and interruption paths; work that has not
started may be cancelled, while a running build must finish before the site lock is released. SIGINT requests cancellation of the supervised subcommands
but does not hard-preempt the parent process's Python threads. The final failure phase is written after finishing, preventing a late
`release-ready` from overwriting the failure; a cancellation request cannot be equated with all work having stopped immediately.
The scheduler outputs task start/cache-hit/end; `task_reports` in `bootstrap-state.json` keeps this run's
execute/probe mode, start and end time, duration and CommandRunner command counts. The counts are classified as AWS/kubectl/total
and contain no arguments or credentials; `runner_commands` counts only direct calls. The new `subprocess_api` separately records the subprocess API commands of this phase
and of the inherited deployment scope, the time spent waiting for capacity and the observed SDK retries.
Each failed task also immediately saves its exception type and a bounded, redacted reason, without waiting for independent long tasks to end and without keeping only the first failure;
the progress stream outputs only the exception type, detailed reasons are read from the private state, and the original exception is still raised after in-flight tasks finish.
`admin.api_budget` constrains the AWS, kubectl and HTTP concurrency of the whole deployment with a private cross-process ledger, at
8, 8 and 4 respectively; Shell subcommands use the same budget, and the atomically maintained PID/starttime leases do not occupy capacity by mistake through PID reuse.
When lease reclamation (`_reap_leases`) uses `/proc/<pid>/stat` to decide whether a CLI has exited, **the zombie state (Z) by itself is not
evidence of exit**: snap-packaged aws/kubectl are exec'd by the Go launcher (`snap run`, `snap-exec`) from a non-main thread, and
before the exec'ing thread takes over the pid the kernel reports the already-dead main thread as Z under the same pid and the same starttime, for a window
of one scheduling delay. Deploy #18 on 2026-09-19 was exactly this: one shim judged a sibling shim's just-started `aws` as exited,
found it running on re-check, and then failed its own innocent command with `cannot release a running API command`.
Now, when `_identity_gone` sees Z, it first checks whether `/proc/<pid>/task/` still has live threads, then re-reads according to
`EXIT_CONFIRMATION_DELAYS` (2/5/10/20/30 ms), and only a consistently Z or vanished entry counts as exited;
when the reclaimer (not the lease owner) finds on re-check that the command is still running it only keeps its capacity and no longer raises, while an owner releasing a still-running command in
`_finish_slot` is still refused.
The failure line for a shim process exit appends the class name and text of the underlying exception after the generic text
(`…capacity/deadline: <Error>: <detail>`): a release that failed only because of a shim otherwise leaves no diagnosable information
(on 2026-09-18, 2 of the 20 concurrent kubectl calls in the finishing identity validation left only the bare generic line). The finishing runtime component identity validation
(`validate_runtime_component_identity`) retries the read-only Pod list and module digest probes once in place after the first failure,
counting an error only if they fail again: one shim hiccup must not judge a candidate that has already fully rolled out as failed and trigger a rollback.
The ledger here is `budget.sqlite3` inside the single command's temporary directory on the deploy host, using the Python standard library SQLite
to coordinate local subprocesses; it is not the production Store, is not pushed to the CPU/GPU clusters, stores no business objects or credentials, and is
deleted on normal exit. The production control plane's business storage remains Aurora PostgreSQL. A temporary ledger left over from an abnormal exit cannot serve as proof of success or reusable state for the next
deploy.
S3 high-level transfers force the classic client and 4-way concurrency and occupy 4 AWS capacity units; the user's AWS configuration is not modified.
Ledger protocol v4 checks the schema and owner cap of the inherited scope and refuses before the command starts when incompatible; a separate
independent budget cannot be created to bypass the global limit. A nested CLI's PARENT is only an identity hint; the PID/starttime and the ancestor chain must be verified.
Before quota is borrowed on the same backend, the parent CLI that spawned this command is paused via pidfd and all of its threads are confirmed stopped; the same parent CLI
allows only one borrowing branch, at most 8 levels deep, and a branch keeps the highest ancestor weight so it can be returned. Before STOP, the
`stopping` state that still occupies quota and the borrower identity are persisted first; `parked` is committed only after all threads have stopped; on return, `resuming`, which counts toward capacity, is committed first,
and only then is SIGCONT sent, with the borrow marker cleared last. A mid-way crash is recovered by other admitters or a surviving shim; a recovered
process must not remain undercounted as `parked`, and a sibling branch must not start a new STOP before the recovery commit.
When cancelled and unable to return quota, the paused caller is terminated; it is never resumed without quota. What is paused here is the deploy host's
AWS/kubectl subprocess, not a GPU Pod, HMA container or node service.
Nested CLI output is staged in an anonymous 0600 file, stdout/stderr combined at most 8MiB, checked before writing;
exceeding the limit terminates the subcommand and discards the partial output, without limiting the size of normal download target files. Output is relayed only after the caller is resumed,
so a paused caller is never unable to drain the pipe. Temporary output does not enter the ledger or ordinary artifacts.
A custom helper that needs the paused parent process to keep providing stdin/IPC
is not part of this non-interactive authentication helper contract.
The budget limits command/transfer concurrency and smooths the command start rate; it is not a precise per-HTTP-request RPS limiter.
Pausing only constrains runnable CLI work; it does not withdraw or freeze HTTP requests already issued.
AWS CLI internal calls and retries are collected through loopback CSM, and only numeric counts enter the ledger; raw CSM may contain
credentials and must never be persisted or output. UDP may drop packets, so the report explicitly marks `observed-csm-lower-bound`;
telemetry not received cannot be interpreted as zero retries, and it cannot be used to automatically widen concurrency or infer AWS billing.
The Python concurrency budget is centralized in `admin.deploy_limits.DEPLOY_CONCURRENCY`; the Shell preflight entry point additionally enforces
a per-cluster cap of 1 to 8 nodes and refuses invalid budgets; read-only queries, candidate preflight and the actual Installer/unavailable-node
budgets cannot substitute for one another.
The server-side dry-run of `regional_node_batch` independently runs at most 8 batches in parallel, each batch still at most 4 nodes, 8 Jobs
and 1MiB, continuing to share the deployment-wide kubectl quota and start interval across clusters; the
Host preflight Job is created only after every admission succeeds. The Host execution window is still limited separately to 1 to 8 nodes per cluster; batch concurrency cannot be treated as widening the Host
or the real Installer unavailable budget. The batch scheduling gain is validated by a deterministic start/per-object RTT model and is not equivalent to
a real AWS end-to-end time measurement.
`admin.deadlines` provides one clock for native HTTP, quota waits and `admin.execution` subcommands;
execution keeps its original public entry points. The whole deploy, each TaskSpec and each subcommand set their own deadlines, and a child cannot extend its parent;
the default total window is 6 hours, the release build 2 hours, Aurora tasks 1 hour, and other tasks follow their registered policy.
Ordinary commands are bound by the normal deadline; what the lifecycle driver waits for is the hard cutoff after the reserved compensation time,
so the parent process does not exit first while the child begins a rollback. The hard cutoff is inherited level by level, and a nested driver may not add another two hours;
rollback takes at most two hours and may not exceed that hard cutoff, and nested compensation may not extend the current compensation deadline.
Read-only administrator commands reserve no compensation window. Cleanup still uses an independent, bounded short window.
`run_command` computes the absolute monotonic cutoff before environment/protocol preparation and passes it as `expires_at` to
`run_owned_command`; preparation, admission and startup overhead cannot regain the full window, and the remaining time is re-verified at launch and before returning success.
`execution.deployment_deadline` wraps the whole operation's `interruption_scope(wait_all=True)`:
SIGINT only registers a cancellation request and notifies registered non-cleanup commands via pidfd, and the safe boundary then raises
`KeyboardInterrupt`; all supervisors are waited on before exit. Repeated SIGINT cannot asynchronously interrupt cleanup, and
`allow_interrupted` is only for limited compensation/cleanup and does not bypass supervision loss or extend the deadline.
Known command stdin is handed to the subprocess through an anonymous `0600` file descriptor and is not written to a named file or log.
This way Manifests, dashboards and private HTTP messages larger than the pipe capacity do not depend on `communicate()` polling to keep writing,
nor receive only the first half of the input because the subprocess started a little slowly. Preparing the input still counts against the original deadline, and the time limit and interruption state are checked again before launch;
termination, reaping and completion proof of the subprocess and its descendants are unchanged.
Grafana and AMP native HTTP run DNS, TLS, HTTP framing and response reading in an isolated `-I -S -B` Python
worker via `src/gpu_fault/admin/native_http.py`; the whole worker tree is bound by the same process supervision and absolute deadline;
blocking calls or continuous chunked trailers cannot be handled by in-thread `read1` polling alone. The worker recomputes the timeout after obtaining the same HTTP/AWS
quota, a single window is at most 30 seconds, chunked responses and private JSON messages are each limited to at most 8MiB, and a late
response may not succeed. URL, authentication header, body and response travel only over private stdin/captured stdout,
never in argv, the ledger, ordinary logs or artifacts.
Within the same 30-second query window AMP calls
`aws configure export-credentials --format process` through a supervised worker, following the original AWS shim's admission and credential chain;
stdout/stderr combined at most 8MiB. The exported temporary credentials are used only in memory and private stdio, SigV4 signing is done with the frozen credentials,
they are not handed to an SDK refresh chain that might block again, and they are not written to new environment variables. TLS verification stays enabled.
This isolation covers only the I/O handed to the worker and does not promise hard preemption of arbitrary Python or system calls in the parent thread.
Each command is supervised by two independent inner and outer Linux subreaper layers from `admin.process_supervisor`, which use PID identity and
pidfd to terminate and reap the complete descendant tree; a new session, a double fork or a killed intermediate Python process cannot let descendants escape.
When the inner layer is killed the outer layer takes over the orphans; when the outer layer is killed the inner layer finishes via the parent-exit signal. The two layers have independent
completion report pipes, and the actual command does not inherit the report write end; only a surviving owner that proves complete reaping may report an ordinary failure
or enter compensation, and the supervisor process exiting by itself cannot be treated as command completion.
The deploy host must support Linux subreaper and pidfd; when supervision cannot be established, the command fails before the target command is launched.
When all completion proofs are lost, `ProcessSupervisionLost` is raised, refusing to let the current process keep issuing commands or enter automatic
rollback; even if an orphan still holds the output pipe, it must not block indefinitely on the output read and then misreport an ordinary failure. This state
does not mean cleanup succeeded, and restarting the CLI alone cannot be used to presume the original command has stopped. An unkillable kernel wait is likewise never
falsely reported as a successful cleanup.
The JSON driver's stdout stays machine-readable; stderr emits limited structured phase progress in real time plus a 30-second wait heartbeat;
other diagnostics keep the whole JSON and multi-line credential context and are then redacted as a unit, and beyond the size cap only the safe error category is kept.
Real-time pipe/terminal output uses an independent non-blocking descriptor and does not change the blocking state of other writers; a closed or full
console cannot make the supervisor exit or block reaping. Regular file diagnostics are deferred to output at command end, and the diagnostics reader's finishing
also has its own cap; machine-readable stdout is neither dropped nor mixed with progress information.
Credential rotation, operation locks, approval, stability windows and node safety barriers are not omitted because of this.
Positive proof caches are reused only within the same process and within a bounded validity: tools 60 seconds, artifact extraction verification 60 seconds, CPU credential
structure verification 30 seconds. The key binds scope, content and validator identity; resource proofs additionally bind the freshly read UID and
resourceVersion. Failures are not cached, the same key is single-flight, different keys may run in parallel. A credential structure proof does not mean the
database password is still AWSCURRENT; the synchronous Aurora refresh is not skipped. Resource queries uniformly distinguish exists, confirmed absent and
read failure; only a successful empty `--ignore-not-found` result may take the absent branch.
The monitoring base resources, Aurora instance readiness, refresher IAM and Node Action key tasks are likewise probe-before-ensure.
`TaskInputSpec` binds each task's true inputs, and ordinary infrastructure tasks do not bind the whole release identity: monitoring resources bind the alert mailbox,
the SNS policy helper and the shared asset; the monitoring install/refresher base tasks bind the CPU identity and the bootstrap implementation;
the default node key binds the wrapper, the helper, the private proof implementation and the target cluster; with registered custody it also binds the full registration,
the candidate release and the custody/bootstrap-related implementation digests. `bootstrap_custody_profile` refuses unresolved
Profile/template policy changes and may not pre-authorize by the old Profile; the first run follows the normal `hyperpod-v1` and
config digest computation by default. Source deploy classification must also send registered custody down the application/bootstrap path and
may not skip evidence verification as unchanged or deploy-host-only. The ARN bootstrap probe checks the IAM,
Pod Identity, SNS and similar prerequisites;
the release additionally checks the ADOT/AMP definitions, available replicas and the refresh CronJob's image/wheel/master Secret references through `observability_drift` and `aurora_refresh_drift`,
bringing runtime resource drift into the release plan. A base
task checkpoint hit cannot replace runtime resource verification. The node key probe privately reads the full
Secret JSON through `read_node_key_proof`, requires the GPU key set to match the current HyperPod nodes exactly, and compares node by node the
actual byte digest of the corresponding entry in the CPU union; it is not a names-only check. Missing or inconsistent entries, a pending rotation marker or an explicit single-node
rotation request all enter ensure; invalid JSON, a read failure or an identity conflict refuse outright and cannot be treated as confirmed absence.
The product's reuse hint cannot replace this full key proof; the Shell entry point likewise keeps the full byte check and the trusted content pin check of the actually selected template,
and does not turn the product's "already verified" hint into authorization to skip the key or template check.
The release's monitoring install script itself also compares per object before writing: the AMP rule groups
namespace and Alertmanager definition are first `describe`d and the online blob decoded; when it is byte-for-byte identical to the file about to be written
and the state is `ACTIVE`, the put is skipped, thereby also skipping the wait for AMP's asynchronous validation `UPDATING -> ACTIVE`;
on the runtime-only path the ADOT collector is `rollout restart`ed only when the apply result of the ConfigMap/Deployment is not `unchanged`
or the Deployment that should be running is currently unavailable, otherwise only `rollout status` is confirmed;
`GPU_FAULT_FORCE_ADOT_RESTART=true` forces a restart. IAM inline policies and SNS topic attributes are converged by drift by the base
tasks and are not written by runtime-only. The script prints `amp-step-elapsed <seconds> <step>` per step on stdout,
so the time of this phase can be attributed directly in the release log.
Once an existing release has taken over monitoring, bootstrap still does not overwrite ADOT/AMP ahead of time. Ownership is judged from the CPU cluster's release
state; a missing local site cannot prove this is a first install. Release classification runs a read-only check of ADOT declaration differences, available replicas and AMP
definitions through `regional_observability_drift.observability_drift`; drift selects the OBSERVABILITY node, and even when the application source is unchanged the deploy-host-only/NOOP shortcut may not be taken.
A legitimate unhealthy JSON that status returns with exit code 1 can still be used for release difference diagnosis, including monitoring and refresher drift;
it does not constitute health evidence and cannot be reused as a passed verify. Invalid output and other exit codes are not accepted as normal diagnostics.
The actual repair still happens after the previous snapshot; afterwards no-drift is proved again. When ADOT is missing, the CPU namespace
anchor must be verified first before an explicit absence can be recorded in the snapshot; AMP is recorded as absent only on ResourceNotFound, and permission/read
failures do not equal non-existence. Rollback restores the original object or the original absence and does not guess the old state from the candidate Manifest.

Both the first and existing application releases have bootstrap converge the monitoring periphery IAM, Pod Identity and SNS publish permission without
overwriting the candidate ADOT/AMP configuration ahead of time. The release verifies the existing IAM, associations
and SNS conditions through the installer's internal `--runtime-only` mode and updates only the versioned runtime resources; it does not create or switch to a second region-level role.
These versioned resources are updated after previous is captured, guaranteeing that the rollback snapshot really belongs to the old version. The actual ADOT render and
the drift check use the same namespace, including the Kubernetes discovery scope embedded in the collector.
Grafana uses an independent `grafana_install` checkpoint and reads the legacy `monitoring_install.grafana` for compatibility;
modifying only dashboards or Grafana options no longer triggers a monitoring reinstall.
When the auto-discovered/created/imported presentation chain is temporarily unavailable, `DEGRADED` and a task warning are recorded and the next deploy
retries; AMP, SNS and the alert subscription remain hard gates. Inconsistent ownership, several workspaces impersonating the same site, or wrong explicit
workspace/viewer input must fail and cannot bypass the identity and authorization checks through presentation degradation.
Exhausting the 30-second window of a single Grafana HTTP request is treated as a presentation fault only while the outer task/deployment deadline is still valid and process supervision is safe;
outer expiry, interruption, supervision loss or a timeout of unknown origin is still raised and may not be downgraded to success.

Schema and credential refresh share `wait-for-kubernetes-job.sh`: it keeps the original timeout budget while also watching the Job's
`Failed/FailureTarget` and deterministic container configuration errors, and refuses a Job UID that was replaced during the wait.
A GET that hits explicit throttling, a transient service fault or a network timeout retries only within the same total deadline; permission, TLS trust,
NotFound and unknown errors refuse immediately. A successful watch returns directly and verifies the Complete object and UID, without misreporting success as a timeout
at the timeout edge because of one extra GET.
Failed Jobs on the ordinary schema/refresh paths are kept for diagnosis; the proof Job of the prerequisite transaction follows the journal and UID
cleanup contract of section 5 and does not bypass cleanup because of a failure. release history reads the ConfigMap only once per run; when unreadable it
alerts without overwriting the old audit.

deploy/bootstrap, Profile approval, AdminConfig, Node key custody registration, join, remove and uninstall compete for the same
state-dir mutation lock. Time-consuming source preparation may run outside the lock; after taking the lock, the site is re-read and the site name,
Region and CPU EKS anchor are checked, and the stale member snapshot from CLI start may not continue to be used.
`config` also re-reads the actual Aurora and other target references inside the lock, refusing before the plan or configuration file is written when identity drift has
occurred. The inherited lock check not only inspects the file inode but also confirms through Linux fdinfo that the open file description
really holds the exclusive flock; independently opening the same file, an old FD that was already unlocked, or another thread's inherited FD cannot bypass
mutual exclusion. A valid lock held by the same calling thread may still be reused by controlled nested commands.
The lock path must be a regular file; symlinks and FIFOs are refused. Token rotation, spare declaration/release, manual disposition writes,
workflow convergence and explicit rollback also use the same lock and re-read the site inside it; the remote collector-outbox
`stats/list` also creates a persistent workflow and is therefore also a locked administrator transaction.
This is the file lock of the single deploy host's canonical `state-dir`, not a distributed lock across deploy hosts or between replicated state directories;
the runtime registry CAS, Store atomic operations, node UID/resourceVersion and fencing remain each necessary.

The AWS/Kubernetes mutations of a new site likewise cover only the first baseline GPU cluster: GPU kubeconfig context,
namespace, token, Executor IAM, Node Action key, NLB NAT EIP and Private Hosted Zone
VPC association may none of them be created ahead of time for a cluster awaiting join. The remaining ARNs are kept only as batch join input. Route53
associations are deduplicated by `Region + VPC ID`; when several GPU clusters share a VPC it is registered only once, and remove disassociates only when no
remaining cluster references that VPC.
`resource_registry_dns` generates association records centrally: the CPU native association is marked explicitly and cannot be inferred from list position;
new GPU associations use a stable physical key, while existing numeric or cluster-form keys are kept by full zone/Region/VPC identity,
so member reordering does not rewrite the immutable ownership, delete policy or dependencies. The association origin must be explicit:
only a successful, verified creation response records `CREATED`; merely observing an existing object grants no deletion right; once the creation response is lost,
read-only discovery cannot fabricate an origin. PKI preparation writes confirmed associations to the same checkpoint promptly, while task completion is still registered by the
scheduler; the DNS helper digest is part of the PKI task input.
Registry synchronization distinguishes a confirmed empty inventory from a read failure. Old associations missing CPU/origin proof, identity conflicts, or duplicate immutable rows for the same physical
association must be reconciled; ownership is not guessed from the one remaining cluster. remove verifies the exact
`CREATED/DETACH` authorization and reverse dependencies before drain; the low-level network disassociation does not delete DNS by default, only when the supervised call explicitly enables it.
CPU native associations, shared VPCs and `EXTERNAL/REUSED/PRESERVE` associations are none of them revoked by removing a single cluster.

Candidate verify evidence binds the source/candidate site digest, the live release identity, the full registry
cluster state, the generation/content digest and the verification time, and is valid for 15 minutes; drift during verify or before commit
fails immediately. The commit order is fixed: site, release state, Aurora resource registration and other compensable steps, then
`ACTIVATION_STARTED` is persisted and the idempotent `PENDING -> ACTIVE` is executed. That marker is the point of no return; a failure after it can only
resume fail-forward and may not clean up an already ACTIVE data plane.

Whoever modifies this flow must at the same time maintain ARN discovery, the candidate site, pre-commit compensating rollback, post-commit fail-forward,
Aurora resource registration and the low-level release JSON contract.

### 8.1 Join Phases, Performance and Compensation Boundaries

The following order comes from `src/gpu_fault/admin/cluster_join.py` and
`src/gpu_fault/admin/cluster_batch_join.py`; it is not a task list that may be freely reordered.

| Phase | Necessity | Performance and parallelism boundary |
|---|---|---|
| Hold lock, discover and recover old attempt | Bind CPU anchor, GPU EKS/HyperPod, account/Region and `NodeRecovery=None`; finish unfinished compensation first | The batch shares one Aurora registry export and network baseline, discovery at most 4-way; batches may not mix sites |
| Local inputs and prerequisite resources | Prepare an independent token, managed namespace, node key and dedicated network/IAM | Writes to the default shared GPU kubeconfig are serial; Executor role, network and node key may run in parallel, the ADOT role waits for Executor/OIDC; for additional custody boundaries see §6.2 |
| Candidate preflight and data-plane install | Install the current signed release only for `PENDING` members, run node dynamic barriers and Fleet waves | The batch shares the candidate preflight; cluster parallelism and the 64-node total budget follow the release, no second set of parallelism parameters is added |
| Full candidate verify | Verify original and new members together, binding site, live release, the full registry lifecycle and generation/content | One full verify is shared, the baseline verify is not repeated; evidence is valid for 15 minutes, on expiry at most one extra verification round for the targets awaiting commit forms a new proof, the old timestamp cannot simply be changed |
| Batch failure-domain barrier | Generate the merged mapping for fully verified members awaiting join, save the publish intent first, then publish and check worker UID, template digest, replica convergence and registry stability | One batch shares one node inventory build and worker convergence; activation proceeds cluster by cluster only after the barrier passes, and may not continue if it fails |
| Commit and activate | First site/bootstrap, release state, Aurora resource registration, then `ACTIVATION_STARTED` and `ACTIVE`, finally re-verify | The batch commits cluster by cluster serially; the expected revision change of an already committed sibling may advance the remaining proofs but may not hide drift of other members or of the release |
| Failure compensation or fail-forward | When not activated and `autoRollback=true`, compensate this attempt; after the activation intent only move forward | Wait for started parallel tasks to finish before compensating; do not undo already successful siblings, and do not start the next attempt in parallel with the old compensation |

A custody wait for authorization or an evidence block does not take the ordinary failure compensation in the table; the prepared namespace/UID and the
original inputs must be kept; the next probe must not be made to pass by deleting resources, automatic recreation or repeated key writes.

An unfinished transaction resume rediscovers and compares the persisted GPU identity; when a CPU is taken for a GPU, HyperPod is rebound or the namespace UID changes it
stops, and new objects may not be cleaned up with old evidence. Network mutations first record the intent and the completed exclusive changes; when it cannot be confirmed whether a request
landed, cleanup may not be marked successful. `ROLLBACK_STARTED/ROLLBACK_FAILED` must first continue the original compensation;
when the remote compensation is complete but the local credential cleanup is not, only the local finishing continues. A cluster token still referenced by the rolled-back registry
is kept at a controlled path bound to the cluster/EKS/HyperPod identity; it may not be deleted and a new token auto-generated to reuse the old registration;
the temporary fleet master is not
kept as ordinary evidence. `SUPERVISION_LOST` forbids automatic retry or compensation; commands not proven stopped must be explicitly reconciled first.
A resume after `ACTIVATION_STARTED` allows the old verify to exceed 15 minutes, but still checks the original bindings and completes the ACTIVE terminal
re-verification; it may not require the target to go back to `PENDING` or take the withdrawal path again.
A member that is complete and still in the local site may return idempotently, which does not mean a real-time verify was re-executed; current health is still checked independently by
`status --full`.

Every in-batch commit still re-verifies the worker/mapping identity of the failure-domain barrier and the current member state; proof expiry or failure recovery
may require a new barrier for the remaining verified members. The barrier may not be skipped to reduce rollouts, and candidate members may not be
marked ACTIVE early. Single-cluster join/remove continue their original mapping refresh and convergence checks.

Before node key preparation, site-wide NodeName conflicts are excluded through the current member Node set, Agent ownership, CPU key occupancy and the targets in the same batch,
and the full NodeName-to-UID binding of `NODE_NAMES_VERIFIED` is saved.
`BoundNodeKeyRunner` passes that mapping to the helper through the internal `GPU_FAULT_NODE_KEY_EXPECTED_NODES_JSON`;
passing only a name list would miss a same-name recreation between discovery and write. The `expected_key_sha256` in the initial proof
uses the actual raw byte digest of the GPU's existing legitimate key, and only missing entries use the v2 derived value; existing randomly rotated keys
can therefore be kept and correctly compensated. A resume may not overwrite the original proof with newly observed values. Only when a key write may have begun and
the Node/key ownership proof is complete does compensation withdraw the CPU entries; the Node UID is re-verified first, then the CPU's current actual
digest is compared item by item and resourceVersion CAS is used. A digest change or unknown ownership stops; foreign keys may not be deleted by NodeName.
Compensation of a join that already committed the site must complete the release state repair even if it finds the local site already recovered; cleaned-up
resources must send explicit terminal updates such as `DELETED/DETACHED` under their original immutable identity, and the Aurora record cannot be assumed withdrawn just because it was
omitted from the upsert list.

### 8.2 Removal Phases, Performance and Resume Boundaries

The public entry point selects the target by full GPU ARN; `src/gpu_fault/admin/cluster_removal.py`,
`src/gpu_fault/admin/cluster_removal_state.py` and their network/resources/Kubernetes helpers
maintain the transaction together.

| Phase | Necessity | Performance and parallelism boundary |
|---|---|---|
| Discovery and binding | Check CPU/GPU EKS endpoint, creation identity, HyperPod ARN, namespace UID, Node UID set, token digest and registry | CPU, target and remaining-cluster network queries are deduplicated, at most 4-way; a resume must read afresh, no long-lived cache serves as deletion authorization |
| `CONTROL_REGISTRY_DRAINING` | Stop the target from claiming new commands, keep the access needed for draining and cleanup | Serial with the later stop and drain, credentials cannot be revoked first |
| `KUBERNETES_QUIESCED` | Stop producers, drain the queues, uninstall node units, stop the Executor and verify bound cleanup evidence | Reuse the verified cleanup checkpoint; exit code 0 cannot replace the full cleanup state |
| `KUBERNETES_REMOVED` | Delete by the recorded namespace UID and confirm actual disappearance, preventing terminating Pods from racing revocation/IAM deletion | After requesting deletion, absence must be awaited; this wait cannot overlap with credential or AWS revocation |
| registry/key revocation and `AWS_DETACHED` | Revoke the target membership, remove only the target node keys, clean up exclusive IAM/NAT/VPC associations | The two branches may run in parallel; AWS runs in reverse-dependency waves, at most 4-way per wave, a failure does not enter the next wave; Secret writes use resourceVersion CAS |
| Aurora, site and release commit | Merge this run's state delta into the freshly read registry, remove the member, token and bootstrap references, refresh the failure-domain mapping | Committed serially after all prerequisite branches succeed; `SITE_COMMIT_STARTED` precedes the site write, other entries may not be overwritten with an old snapshot |
| `VERIFIED/COMPLETED` | Prove the namespace and registry target are absent, CPU/remaining members are healthy, GPU EKS/HyperPod still exist | Independent read-only checks may run in parallel; unconfirmed or invalid JSON does not equal absence |

`src/gpu_fault/admin/cluster_removal_kubernetes.py` uses
`DeleteOptions.preconditions.uid` and foreground propagation in the namespace deletion request; a UID already checked at read time cannot
replace this API precondition. A lost deletion ACK may continue only after a renewed successful read confirms absence;
a new same-name UID appearing during the wait stops, and the new namespace cannot be treated as the old deletion still in progress.
Node Installer annotation cleanup reads the target HyperPod's complete Node set once and, after checking the saved
Node UIDs and resourceVersions, submits JSON Patches with at most 8 `metadata_cleanup` workers,
each patch again testing UID/resourceVersion; when the initial set or identity does not match, no patch is submitted.
A later conflict stops unstarted tasks and waits for started tasks to finish. These 8 lanes are metadata API concurrency, unrelated to Host preflight,
Fleet waves and the node unavailability budget, and do not authorize stopping or reinstalling 8 nodes at the same time.

Route53 associations are handled by `Region + VPC ID`, not owned by whichever cluster key registered them first.
A VPC still used by the CPU or another GPU cluster and a shared NAT EIP may not be revoked; exclusive EIPs are revoked one by one with residue checks,
and one that no longer exists cannot cause the remaining entries to be skipped. An exclusive VPC association must wait for ChangeInfo to be `INSYNC` and
re-confirm that the actual association is gone. When retained resources still depend on the target's IAM or other resources, removal is refused before the data plane is stopped.

When the site has already removed the target but the later synchronization was interrupted, the original full EKS or HyperPod ARN may continue through the exact
tombstone of the same attempt; the target may not be guessed by name or ARN suffix. An old `COMPLETED` record does not authorize skipping a re-joined
member; the old attempt must be archived and rediscovered. The current remove state requires schema v2 and full binding;
an old journal lacking bindings needs explicit reconciliation; digests may not be filled in by hand, the state directory may not be deleted, and it may not be treated as a first run.
Node key cleanup also binds the saved CPU Secret UID and the raw key byte digest, and only after re-verifying site-wide NodeName ownership
and Node UIDs does it delete with UID/resourceVersion CAS and read back to confirm; the raw key is not stored.
When removal or uninstall hits a local `ProcessSupervisionLost`, a refusal marker must be persisted in the original journal, and a later
process may not resume automatically even though its in-memory supervision state has been reset; it must first confirm the original operation stopped and complete reconciliation.

The failure-domain refresh after join/removal is executed by `src/gpu_fault/admin/failure_domain_map.py`, with all external
commands under the unified supervision and deadline. Missing identities, duplicate names or HyperPod ownership conflicts in the Node list must be
refused; rows may not be silently dropped and an incomplete mapping published as success. Before writing the ConfigMap, it is first proved that the control-worker is readable
and its identity complete; a permission error does not equal absence, and managed membership operations do not accept a missing worker either. Only an explicit
internal bootstrap call may leave a not-yet-created worker to a later deployment. When the template digest already matches, no patch is repeated;
on change, the update is guarded by the Deployment UID/resourceVersion, and after waiting for the rollout to complete the same
Deployment UID and target digest are checked again before returning; waiting for readiness by name alone cannot prove the original object finished updating.

### 8.3 Uninstall and AWS Deletion Phases

`src/gpu_fault/admin/uninstall.py` uses monotonic phases and calls
`deploy/control-plane/regional/prepare-clean-redeploy.sh` for the Kubernetes/node part;
`src/gpu_fault/admin/aws_cleanup.py` only performs AWS deletion by registered type and is not a public standalone command.

The administrator uninstall journal's schema-v2 binds the installation ID, site identity and original inventory.
Only after the uninstall is explicitly COMPLETED does the next deploy archive, with persisted intent and checkpoints, the whole
`uninstall/` directory plus the old release/join/remove transaction directories and site records, rather than just renaming the main state file;
a new install may not continue to use the old namespace UID and old deletion plan. Administrator configuration, credentials, signing keys and
the audit archive are kept. An unfinished old journal or one with lost bindings must still be reconciled and may not be automatically initialized into a new cleanup.

The resource registry isolates the storage scope by installation generation: an old site's first registration keeps using the original site_name, and
`<site_name>:installation:<installation_id>` is assigned only after full decommission. All current build/fetch/sync,
join/remove/uninstall uniformly read `RenderedSite.registry_site_id`; AWS names, tags and scope
checks still use the original logical site_name. Historical resource IDs in the old database remain unchanged, and the new generation may register re-created
AWS IDs; identity conflicts within the same generation are still refused, and the immutable constraint cannot be bypassed by temporarily switching scope.

| Phase | Necessity | Performance and parallelism boundary |
|---|---|---|
| `REGISTRY_EXPORTED` | Check the site digest, full CPU/GPU/Aurora identity, account/Region, dependencies and effective policy; save the original inventory and pending-deletion state first | Retries reuse the bound snapshot; the installed-resource registry is not rewritten during destructive cleanup to eliminate unknown entries |
| Kubernetes preflight and drain | Save the cleanup inventory, namespace objects, cluster/Node UIDs and the full fleet/unit inventory; refuse active or unknown records | Site-wide, first set the targets to DRAINING in the same revision, then stop the GPU producers; ingress and consumers keep running until strictly drained, no bulk orphan SQL release |
| Consumer and Executor stop | Complete `CONTROL_CONSUMERS_STOPPED`, `INGRESS_STOPPED` and `GPU_EXECUTORS_STOPPED` in order | CPU roles stop only once no active records remain; must precede node uninstall, so that tearing down nodes does not create commands nobody executes |
| `NODE_RUNTIMES_STOPPED` | Uninstall the Node Runtime and Agent by the fleet/UID inventory saved in the original journal | The database is no longer read after the CPU consumers stop; existing temporary node DaemonSets stay bound to run/spec/UID, no new resource kinds are added |
| auxiliary, object and namespace cleanup | Stop auxiliary components, delete registered objects and managed namespaces and verify item by item | Parked spares restore only the explicitly owned original scheduling baseline, written with Node UID/resourceVersion and read back; an unknown baseline blocks, customer objects are not deleted |
| `NON_AURORA_VERIFIED` | Finish the non-database resources such as edge, monitoring, IAM and network first, keeping the dependencies needed for later verification | Independent resources of reverse dependency and the same priority at most 4-way per batch; CPU Helm/LBC and the like must be proven deleted while the CPU API is available |
| `CPU_VERIFIED` | Keep or delete the CPU cluster according to `keep/delete` and re-check that the GPU clusters are kept | CPU HyperPod before CPU EKS; nodegroups may all be requested first and then awaited one by one, Fargate profiles must be requested and awaited one at a time, addons are also handled one by one |
| `READY_TO_DELETE_AURORA` | Pin the preceding residue at 0 and the pre-Aurora snapshot, keeping the final database audit boundary | `keep` without reset only verifies the whole Aurora set; delete mode first persists the database incarnation and the final snapshot policy |
| Aurora deletion and `COMPLETED` | Delete mode finishes by the current actual reader, writer, cluster and dependencies, checking that the retained snapshot is available and the final residue | In the same reader wave, send the requests first and touch the writer only after all have disappeared; a writer identity change stops immediately, roles are not guessed by name; cannot run in parallel with earlier phases |

Before every AWS deletion, the scope, current calling account, live tag or exact association is re-checked; only the calling account already checked within the current cleaner
is cached, resource ownership is not cached. Unknown types, missing identity, permission/TLS/resolution errors or shared dependencies
all stop. NotFound accepts only the exact error code of the corresponding service and the checked exit status, and cannot match arbitrary error substrings.
IAM instance profile users, still-attached policies, OIDC providers trusted by other roles, IGWs used by external
route tables, and an LBC still serving customer Services/Ingresses all block the related deletion.
The Route53 CNAME must point exactly at the NLB owned by this site; SQS removes only the exact topic statement, and mixed or wildcard
shared authorizations must be reconciled manually first, other permissions are not cleared in bulk.

The policy dependency role edge of historical LBC registration rows is normalized to role-uses-policy only in the temporary execution graph of `uninstall._execution_dependencies`,
serving the same pre-validation and deletion scheduling: the role deletion flow first detaches its own
attachment and deletes the role, then the policy is deleted. The dependencies, ownership and delete policy contained in
`InstallationResource.immutable_identity()` in the exported snapshot, deletion plan or API/SQL may not be rewritten.
Keeping a role while deleting its policy must be refused before cleanup; deleting a role while keeping the policy may proceed; missing dependencies,
cycles and other retained consumers must still be refused. The policy deleter continues to refuse any remaining attachment and does not force-detach.

Every AWS waiter request and successful return is bound by the same remaining deadline; a long resource wait does not mean a single AWS CLI
may run indefinitely. Retrying an object already in `deleting` only keeps waiting and verifying and does not re-send the deletion; when keeping the final snapshot,
even if the cluster has already disappeared it must be proved that the snapshot belongs to the original database and is available. A snapshot name conflict or database
incarnation drift is refused before instance deletion. A `COMPLETED` rerun only re-verifies and does not authorize cleaning up same-name re-created resources;
old in-flight state lacking the original site/inventory/target binding must be explicitly reconciled and cannot be presumed safe because its hashes are self-consistent.

The Kubernetes cleanup journal is currently schema-v2 and always includes the independent `NODE_RUNTIMES_STOPPED` phase.
The site-wide order adds `CLUSTERS_DRAINING`, preserving the full durable state of unselected clusters with a bulk CAS;
other clusters may not be set back to ACTIVE using the candidate site. The journal's `phase_order` must match the current order exactly;
the old v2 order, like schema-v1, needs explicit reconciliation, not just filling in phases or updating a self-computed digest.
A resume accepts only the original journal and its run/site/namespace/Node UID and fleet bindings; it does not look for or guess a "latest failed"
checkpoint, re-initialize the original identity or fabricate a drain completion record. schema-v1 belongs to the old order and must be explicitly reconciled; changing only the version number,
fabricating new phases or re-initializing the same path to continue is not allowed. The `previous-unschedulable` of a parked spare must come from
an explicit baseline owned by this cleanup; when missing or of unknown ownership, uncordon may not be the default.

A resume with consumers not yet stopped re-proves DRAINING and the already completed queue drain; an old completion marker cannot serve as a
continuing admission barrier. A resume after the consumers stopped uses only the original fleet/UID proof and no longer reads the database. Pure registry
drain uses a context that consumes no images; normal `RegionalRelease` construction and direct calls still check the schema-v3/v4
image lock, and the image usage path cannot be bypassed through a CLI branch or an environment variable.

Dynamic workload RBAC is checked read-only through
`src/gpu_fault/admin/cluster_removal_rbac.py::inspect_workload_namespace_rbac`.
The collector's `attach_workload_rbac`, after the static synchronization, appends the proven
Role/RoleBindings to the same GPU context snapshot, with rows marked `guarded_delete=workload-rbac`;
the original UIDs and the namespace/SA/Deployment anchors are saved in that snapshot's proof, are not written back to the install ConfigMap,
and orphans are not exempted by label or by a self-computed digest.
remove-cluster and site-wide uninstall both enter the same `prepare-clean-redeploy.sh` flow.
`cleanup_kubernetes.delete_workload_rbac`, after verifying the original journal's drain/stop phases and the actual quiescence,
calls
`src/gpu_fault/admin/cluster_removal_rbac.py::delete_recorded_workload_namespace_rbac` inside `APPLICATION_OBJECTS_DELETED`, before the ownership anchors are deleted.
It re-verifies the renderer, scope, original UIDs and foreign bindings, deletes by UID/current resourceVersion and confirms disappearance;
the caller writes the deletion progress item by item into the original journal. A row inconsistent with the proof, an unknown read, a checkpoint failure or a same-name
recreation all stop. Early pruning, an independent RBAC journal or a deletion path based on `DRAINING` alone may not be restored.
The stop re-verification must also confirm that all Pod objects of the corresponding Deployment are gone, including Pods still terminating; a replica count of zero
cannot prove the stop by itself. After the CPU cluster has been deleted per the uninstall plan, the internal `verify-targets --skip-cpu` permits only
a read-only re-verification when the original journal has completed `CLEANUP_COMPLETED` and every RBAC deletion has a checkpoint;
it continues to verify the GPU clusters and the absence of the original resources, and may not be used to launch remaining deletions or omit the CPU checks of in-flight cleanup.

### 8.4 Phase Boundaries of Other Administrator Commands

| Command | Required phases and success conditions | Performance, parallelism and failure boundary |
|---|---|---|
| `node-key-custody configure` | Explicit private selection file and external trust pin, verified under lock with the content identity pinned; only records the registration | Does not sign or modify Secrets; execution after preparation and authorization is still done by the original deploy/join, a corrupt registration cannot implicitly exit custody (§6.2) |
| `rotate-token` | Under lock re-read the site and release gates, publish the overlapping revision, update the GPU Secret, roll the data plane/nodes, observe the quiet window, commit the local token and final revision | Nodes follow the release waves; Secret/rollout record the start intent first, compensation covers waves that started but were not confirmed complete. Writing the `TOKEN_FILE_WRITTEN` intent is the point of no return; after a lost ACK verify the old and new digests and fail-forward; the terminal state precedes temporary token deletion |
| `config spare` | Read-only health check; declaration first saves the cluster/Node UID and the original label/cordon baseline, then modifies with UID/resourceVersion and reads back; release restores only that baseline | Single-node locked mutation, spare allocation parallelism is not widened; unknown pool state, reservation/quarantine, cross-cluster records or a Node UID change refuse release |
| `workflow-reconcile` | Build the plan, read node evidence via the target GPU kubeconfig, rebuild the plan before apply and check the Store atomic conditions, archive the result; a non-plan record that a running release refuses only for a missing source plan but that has a verified restore successor is promoted on the command-line side to `verified-restore-non-plan` and written through the in-Pod `apply-verified-restore`, which uses only existing store primitives | Node evidence of the same cluster is read once and reused; the plan is not an action authorization cache, apply remains serial under lock, partial failures must keep their results; an explicitly blank selector may not widen to site-wide; promoted items and the runtime apply are sent separately, and the runtime subset rebuilds its plan separately to match its digest |
| Incident close | Select an explicit incident or a limited discovery scope and judge item by item on the CPU; QUARANTINED additionally obtains GPU isolation evidence before a second judgement | Same locked entry point, closed/refused and evidence recorded per item, any refused yields a non-zero exit; the whole batch is not one atomic transaction |
| `submit-remediation` | `--plan` is read-only; apply re-reads the site and incident/nodes under lock and, by selection, creates the supervised workflow, writes the mechanical check confirmation, or (`confirm-node-action --node`) writes a manual confirmation for a node action with unknown result on a BLOCKED record | The mechanical check annotation carries resourceVersion, incomplete or conflicting node identity fails; multiple nodes are not one Kubernetes atomic transaction, nodes already written must be audited before a resume; the manual confirmation is judged on the command-line side, the in-Pod script uses only existing store primitives to compare the confirmed step's original text, re-check the Agent record and boot id and then writes with `save_workflow(expected=)` |
| `collector-outbox` | Submit a single-step maintenance workflow, poll the corresponding request ID/known status with bounds, verify the node results of the target action, archive metadata | `stats/list` does not modify the outbox but still writes a control record; each poll reads the shared remaining deadline, a timeout does not cancel the remote workflow. Success only on `SUCCEEDED` with valid node results and no error |
| `deploy --rollback` | Under lock re-read the live state, take only the committed previous, run the rollback under supervision, then align the management site/configuration and sync-state | No independent build, no second lifecycle transaction started in parallel; neither an engine failure nor a management-plane alignment failure may report success, and the old transaction should first resume in its original direction |

The token journal pins the values and types of `keep_window`, `window_minutes`, `quiet_seconds` and
`acceptance_timeout_seconds`. An in-flight resume and a post-commit resume with only temporary file cleanup left must both
be consistent; an old journal missing fields needs explicit reconciliation, defaults may not be guessed in. The `--rollback` direction is not part of these four
policy bindings, so before the intent is written a rollback under the same policy is still possible; past the point of no return or the terminal state the original direction is followed.

`submit-remediation restore` decides isolation ownership per node: nodes carrying this incident's quarantine taint or this incident's isolation
annotation enter restore; nodes already clean or isolated by another incident are skipped with the reason given in `node_decisions`;
when no node is still isolated by this incident it refuses and suggests closing the incident. When only some nodes are restored, the step carries no incident GPU scope and does a per-node
full verification. `submit-remediation restore` reads the incident nodes' GPU inventory and uses it only as routing for the per-node verification.
When the inventory is complete and every incident GPU has a unique owner, the GPU/Fabric verification may be split into per-node steps; the single `RESTORE_SCHEDULING` runs only after
all verifications pass. When the inventory is missing, incomplete or ambiguous, the original verification scope is kept;
the verification requirement may not be deleted because a UUID is temporarily not enumerable, and an empty intersection may not be downgraded to a success check with no GPU scope.
A single-node escalation incident prefers the node/GPU correspondence already on the original step and may not inherit the GPUs of an unrelated sibling node;
an unprovable historical scope error needs supervised reconciliation and is not guessed and corrected at restore time.

Quiet observation requires `0 < quiet-seconds < acceptance-timeout-seconds`; the first wait follows the last
data-plane mutation, reads and polls share the remaining deadline, and each probe round pins `quiet_start`.
Before the first wait, the UID, generation and desired replica count of the managed ingress Deployment are pinned and a complete,
healthy Pod set with a matching ownership chain is required; when probing ends, the same Deployment and Pod set must still match. Logs may not be taken only from
the Pods that still exist while missing replicas that disappeared within the quiet window. Before and after, all ingress Pod UIDs and the
`api` container's Ready, container ID, start time and restart count are checked together; a missing replica, a scale-down or a recreation refuses.
Each source first reads the current log prefix without a since filter, with `--container=api --prefix --timestamps --tail=-1` and
`--limit-bytes=4096`; the first complete record must be no later than `quiet_start`; the byte length and SHA-256 are recorded only for complete prefix
records, and the receipt is used only in memory within this round. Only a trailing fragment that exactly touches the byte cap may be
truncated; below the cap but incomplete, no old complete record, or an empty response all refuse.
Then the window is read with `--since-time` floored to the second, and decommissioned-token records are filtered on the original fixed nanosecond boundary.
After all windows are read, the prefix is re-read at each receipt's exact length and must match in length, completeness and digest, after which the source identity is re-verified.
A normal append does not invalidate the receipt; prefix loss/rotation, future timestamps, incomplete or late responses
can none of them be interpreted as quiet. This proof relies on trusted normal append/rotate semantics and does not prove that the Kubelet/CRI did not silently
under-report, that a malicious same-prefix rewrite did not happen, or that an offline caller that sent no request within the window has switched tokens.
The auxiliary commands share supervised, bounded CPU exec/node queries and redacted diagnostics, and do not put raw authentication helper errors or
credentials into argv, ordinary evidence or logs. The implementation basis is the same-named command modules in `src/gpu_fault/admin/` plus
`rotate_token_policy.py` and `rotate_token_acceptance.py`; for the validation boundary see
[Validation limitations](../validation-limitations.md).
## 9. Change and Acceptance Process

The production deploy host consumes only signed offline bundles; `deploy-host-setup-online` is for development checkouts only.
This online entry point first performs a read-only check of the paths and version configuration, then two bounded Make tasks prepare, in parallel, the pinned Python/admin CLI and
the independent supply-chain tool environment, without the caller adding `make -j`. Both paths use the system Python 3.12;
`DEPLOY_HOST_BOOTSTRAP_PYTHON` may specify the system interpreter path, which does not depend on the main venv being created.
The tool branch downloads and verifies the Prometheus archive while preparing the Python tools, and installs and verifies promtool only after both have finished;
pip writes into the same venv remain serial. The tool-only version check depends only on the Python standard library.
Once the pinned versions, dependency closure and archive digest checks pass, the tools are reused; when missing or mismatched they are repaired.
If any step fails, it waits for the tasks already started to finish and then returns failure; it does not roll back the other path's completed environment. CI keeps an independent tool target;
the signed offline install does not go online and adds no promtool requirement. The online Python initialisation is still rebuilt by the original procedure and may go online again.

The deploy-host wheel (`gpu-fault-deploy-host`) is packaged by `scripts/component_wheels.py` by import closure;
the closure follows both `gpu_fault` and `gpu_fault_release`: the administrator CLI imports the release engine at module level
(status header, deploy consent, schema change acceptance, rollback), so the two must be installed together with the wheel at the same version.
deploy-host also delivers the existing `gpu-training-submit` and `gpu-fault-workload-annotate` entry points and their closures;
the compatibility entry points in the Control Plane are kept, and this adds no administrator permissions and changes no submission security checks.
The release engine's non-Python inputs remain under `deploy/` and are located by `gpu_fault_release.repository_root()`:
when run from a checkout it is that checkout; when installed in a venv it takes, in order, `GPU_FAULT_REPOSITORY_ROOT`
(set by `setup_deploy_host` during the `gpu-fault-admin --help` install smoke test) and the venv-bound
`gpu-fault-managed-state-dir.json` → `site.yaml`'s `spec.repositoryRoot`.
The bound site is read with a restricted YAML parser that accepts legal line folding, quoting and trailing comments; a relative repositoryRoot is resolved against
the site file's directory, while non-string, empty and missing repository anchors are still refused.
The bundle must bind the OS, CPU architecture, Python 3.12 ABI/sysconfig platform, libc and the digests of all payloads;
schema v2 matches by the payload identity of the current clean checkout, with commit authorization carried by the independent signed CI gate or a local
success record, while the old schema v1 still matches the embedded commit exactly. setup verifies the signature first, then installs with `--no-index`
into a temporary venv, and switches atomically only after the dependency and system tool checks
all pass. New bundles also record a `dependency_identity_sha256` computed from `build.lock`, `deploy-host.lock` and the platform:
each identity installs the shared dependency venv only once, and different project wheels reuse that dependency layer through
a lightweight overlay venv and `.pth`. A missing, corrupted or identity-mismatched shared layer fails closed;
old bundles without that field keep using the full install. The same bundle is only verified and reused, and the original venv is kept on failure.
The initialiser must not call the system package manager, and reports must not contain credentials. Adding a build-time Python dependency requires updating
`requirements/deploy-host.lock`; adding a system tool requires updating
`src/gpu_fault/data/deploy-host-tools.json`, the bundle build, the initialisation checks and the tests.
The offline venv install must remove the inherited `PYTHONPATH/PYTHONHOME` and force-install the project wheel from the bundle,
never picking up the workspace egg-info just because the semantic version is the same; an incomplete venv with the same bundle digest but a missing CLI or state file
must be rebuilt automatically, and only a complete venv with a healthy dependency check may be reused.
When the source preparer switches to the verified deploy-host CLI, it also prepends the selected venv's `bin` to the subprocess PATH;
subsequent site commands are bound to the current CLI's interpreter directory too. Directory selection does not resolve the final target of Python symlinks,
which would wrongly fall back to the system environment; the inherited `PYTHONHOME` is removed so it cannot override the verified venv's package paths.
The verified executor then puts the API quota shim first; the operation lock and deadline continue to be passed through;
there is no need to activate the internal content-addressed venv, and the interpreter selection on GPU nodes is unchanged.
The internal hand-off that starts the verified CLI also removes the old `PYTHONPATH` and explicitly binds this run's repositoryRoot,
so an old checkout cannot shadow the installed code; later site scripts use only the source paths of the bound repository.
Removal/rotation entry points that receive a complete environment must not layer the original environment on top again and bring back the removed `PYTHONHOME`;
when an in-process rotation scope exits, the caller's complete original environment is still restored.
The bundle also records `project_distribution=gpu-fault-deploy-host`, the project wheel SHA and
`project_module_digest`; before activation, setup reads the installed distribution and recomputes
`gpu_fault.module_digest()`, forbidding the Control Plane project wheel from being installed as the administrator CLI by mistake.
`COSIGN_PASSWORD` may only enter the final `cosign sign-blob` process; the quality gates, PostgreSQL tests,
image/artifact builds, attestation generation and deploy-host bundle build must explicitly remove that variable.

### 9.1 Build and Deploy Upgrade After an Ordinary Code Change

This procedure applies to ordinary code changes that do not modify Profile templates, site topology or AWS resources.

A code candidate for an existing site can run build independently:

```bash
COSIGN_SIGNING_KEY=/secure/release/cosign.key \
make PYTHON=.venv/bin/python release-build \
  RUNTIME_IMAGE_REPOSITORY=<registry/repository>
```

This target requires a clean source tree; it runs static first, then runs ordinary pytest,
PostgreSQL stress and the source-only artifact with 1 to 3 lanes according to CPU capacity. Once all pass, it uses the hashed requirements exported from `uv.lock`
to prepare the runtime image. The first ARN deploy creates ECR and completes the login first, then builds the control-plane/executor wheels with the same
target; an independent build provides the repository explicitly. The image identity computes a canonical image input digest from the
Dockerfile, pinned base image digest, runtime lock, platform, build args, wheel SHAs and
module digest, and uses the full digest to generate an immutable tag.
Internally, static splits Ruff, mypy, compile, architecture, contracts, security, deployment configuration, docs, YAML and Shell into
at most 10 lanes; an ordinary `make check` runs, after static passes, the artifact check and the full pytest that excludes the duplicate artifact tests in parallel.
The quality gate subprocess is isolated from the outer deploy's deadline, compensation markers, API budget and shim PATH, so that the tests'
simulated clock or fake CLI cannot use the real deployment context; supervision and timeout of the outer build process still apply, and the subsequent real image
build and release continue to use the original deployment budget.

When a local release-build needs PostgreSQL and no test connection is explicitly provided,
`admin.release_postgres.isolated_postgres_allocation` creates an isolated database. It pins the local Docker
socket, first resolves and binds the PostgreSQL 16 image ID, then persists the creation intent, and starts only after checking the full CID, CID file,
ownership label and loopback port; readiness checks and cleanup both use supervised bounded commands.
The password enters only a private file, never Docker arguments or a URL. Authorization and credentials for real builds live outside the repository in
`<state-dir>/release-postgres` with permission 0700, files 0600.

The locally managed PostgreSQL gate is split by `scripts/run_postgres_shards.py` into independent instance shards;
`POSTGRES_TEST_WORKERS` defaults to a quarter of the core count clamped to 4..16 (the same derivation as `PYTEST_XDIST_WORKERS`) and allows 1 to 16. The public production build and the staging impact gate pass
`POSTGRES_TEST_PARALLEL=1` to the original `test-postgres-stress` target only when
a valid local allocation authorization exists; the original quality gate name, 8 contending workers and 40 stress rounds are unchanged.
Each shard uses an independent process to create its own PostgreSQL 16 and private authorization, and pytest stays at `-n 0`.
The full collected list, the actually selected duplicate-free union, a successful test phase and zero skips must all be verified;
one shard succeeding or a seemingly identical total does not replace complete proof. This local report is not a CI shard receipt.
An allocation that already exists in the outer layer is still kept by the original deployment scope; shards must not take it over or clean it up;
the shards' own identity, stop and cleanup keep following the managed PostgreSQL lifecycle and do no global Docker cleanup.
Manual external connections, the CI coverage path and explicit custom impact test commands do not change how they execute.
When a staging impact plan requires the full gates and PostgreSQL, the path with a valid local authorization and the default stress command
reuses `run_release_gates` directly: static first, then ordinary pytest, PostgreSQL shards and the source-only
artifact in parallel, no longer finishing `make check` before starting PostgreSQL. Other impact plans still execute by the original selection. Changes that touch only the deploy-host layer or are QUALITY_ONLY go through the source impact gate of `staging_deploy.py`; when the impact plan requires PostgreSQL, that gate likewise allocates the isolated gate PostgreSQL before executing the plan, and there is no path that runs without a database.

The build inherits only the password-less URL and the authorization reference; `postgres_test_environment` selects a private HOME/PGPASSFILE only for the release-gates
PostgreSQL branch and the impact tests' PostgreSQL subprocess; the HOME of ordinary tests,
artifact, ECR/Docker and signing tools is unchanged. The native tests still verify the CID, ownership, port and
server version independently, and do not skip verification because the parent process has created a database.
On exit, the container and volume are deleted by exact CID, and the private directory is deleted only after confirming that neither the CID nor the name exists.
When the creation result is unknown, cleanup is unconfirmed or supervision is lost, the whole build fails and the ownership evidence is kept; subsequent builds or artifact
reuse also refuse an unresolved `release-postgres` directory. First confirm process and resource ownership by the original operational recovery procedure,
then reconcile under control; the refusal must not be bypassed by simply deleting a same-named container or wiping the directory. An explicit external test connection is still the caller's
responsibility to isolate and authorize; this helper does not take over or delete that service.

Shell scripts and Python share the same size ratchet: `scripts/check-python-architecture.py` records the file line counts and `name() {` function line counts of `*.sh` under
`deploy/`, `scripts/`, `tools/` and `tests/` in `architecture-baseline.json`, with the same thresholds as Python (1500 lines per file, 200 per function); entries over the limit
may only be registered with `--write-baseline` after review, and thereafter may only shrink, never grow. `make shell-check` and the shellcheck in the CI static job
run with `--severity=info`, and all scripts are warning-free at that level; remote payloads and JMESPath queries that genuinely need a literal `$` or
backslash may only use per-line `# shellcheck disable=` with the reason stated; file-level exclusions are not accepted.

If the tag already exists in the registry, its OCI digest is reused only when the target platform and all `gpu-fault.*` config labels match this run's inputs
exactly; if the tag exists but any label differs, it fails closed. On a miss it runs
build/push and reads the manifest digest and labels back from the registry again. The optional BuildKit remote cache uses an independent
mutable cache repository; it only speeds up layer builds and cannot enter the Manifest or replace the verification above.
Only a registry missing response matching the target reference may enter a cold build; a missing Docker credential helper,
authentication, TLS, network, process exit or output parsing errors must stop and must not be inferred from generic `not found` text.
Error messages keep the safe tool category and troubleshooting direction, and do not echo helper output that may contain credentials.
`release_image_set` parses the BuildKit options as a CSV structure, appends
`-control-plane`, `-executor` and `-node-dependencies` respectively to the registry cache tag, and uses matching subdirectories for the local cache,
so that parallel exports do not overwrite the same cache index. An ordinary cache import tries both the component target and the original shared target;
imports with a pinned digest, a selector or another backend keep their original semantics and do not guess new addresses. split export
supports only explicit registry refs and local dests; duplicate, conflicting, ambiguous or unsupported targets are refused before the build.

The `artifact-check` in `make check` first builds this run's single canonical set of the three component wheels and the
Node bundle. The Runtime Image and final artifact phases verify the source identity, physical SHAs,
`module_digest` and the wheels embedded in the bundle, then reuse that set of artifacts to generate the deployable Manifest v4 and
attestation, and within the same `release-build` use cosign to generate
`dist/current-attestation.bundle.json`. PostgreSQL concurrency, lease, counter or fencing changes
must still pass CAP-005.

The image-set descriptor schema v3 must record the control-plane/executor wheel SHAs and module digests separately,
plus the node offline dependency inventory SHA; each image records its full image input and its SHA.
The three builds run with at most 3 lanes, share the same immutable ECR repository and add no AWS resources. Image reuse still verifies
the registry labels; before emitting the descriptor, the builder starts a network-less, read-only container through `verify_release_images` to
check the component entry point, the actual module digest, that the other component's directory really is absent, and the digests of every offline dependency file.
The node dependency image additionally keeps a deploy-host local build receipt at
`~/.cache/gpu-fault/node-dependency-images-v1/`. The directory permission is 0700 and files 0600;
an independent local key authenticates the receipt with HMAC, and the key or receipt must never be added to the source code, release artifacts or a ConfigMap.
The receipt binds the repository, both locks, platform, base, Dockerfile and the builder/publisher digests; on reuse it still
re-verifies the pinned OCI digest, platform and all expected labels against the registry. Only when both authentication and identity checks
pass are the wheel download and build skipped; a missing cache or an image that is definitely absent takes the cold path, while corruption, permission or identity errors
must fail, and an ordinary inventory file cannot be trusted. The receipt does not replace the final release signature and image content checks.
The wheel names, versions, platform tags and file hashes in the inventory are also bound item by item to the current lock; pip's requirements
parsing and compatible tag computation run in an isolated subprocess with a 30-second cap, so its global logging configuration cannot pollute the deploy host or the application.
A signature-verified split release checks the CPU, Executor and dependency image digests in batch within the same ECR repository, and
verifies the complete returned set of registry/repository/digest; a read failure is not treated as an absent image.
The single-image entry point for schema v3 shared images reuses the same verifier and binds the Region and registry ID explicitly; an empty set,
wrong account, wrong digest, duplicate entries or unfinished pagination can never count as a reusable image.
`make runtime-image-check` performs the same local build/content checks without pushing.
The final artifact build recomputes and compares the canonical
artifacts item by item instead of serially rebuilding a second and third set of wheels; on mismatch it must not generate a deployable Manifest.

The application release identity and the deploy-host bundle identity are independent: changes to the deploy-host lock, setup script or tool
inventory rebuild only the deploy-host bundle and do not force a new Runtime Image or application release.
Application Python behaviour continues to be bound by the three component wheel SHAs and `module_digest`; removing the broad
`src/**/*.py` from `runtime_image_inputs` does not bypass source verification.

`gpu_fault.admin.cli` and its dependency closure belong to the independent `gpu-fault-deploy-host` distribution.
The `gpu-fault-control-plane` wheel and the Runtime Image must not contain the `gpu-fault-admin` entry point,
`admin_cli.py` or `deploy-host-tools.json`. Therefore administrator CLI, bootstrap and source deploy
code changes update the deploy-host bundle and the deployment-domain tests, but do not change the application component digests;
when the Manifest, renderer output, Node bundle and Runtime components are all unchanged, the cluster application release is classified as
`NOOP` and only the final verify runs.

The source-only component artifacts of a dirty staging are also written to
`<state-dir>/component-artifacts/<component-build-identity>/`. The identity covers Python, the
platform, `pyproject.toml`, the pinned build frontend, the source closures of the three components and every input of the Node bundle.
Later delivery-only or deploy-host-only snapshots may reuse the physical wheels and Node bundle, but must regenerate
the current delivery Manifest; any component, node script, systemd unit, cleanup inventory or
build lock change uses a new identity. When the cache is corrupted, that entry is ignored and rebuilt, and the delivery identity is not inherited from an old Manifest.

Deployment phase:

```bash
make PYTHON=.venv/bin/python release-deploy \
  SITE=/path/to/site.yaml \
  COSIGN_KEY=/secure/release/cosign.pub
```

By default `release-deploy` reads `dist/current-attestation.json` and
`dist/current-attestation.bundle.json` from the same checkout, and does not run `make check`, a Docker build or dependency downloads.
When duties are separated on an existing site, Release CI runs build and the administrator runs deploy; the fixed artifact file names are unchanged.
The deployment phases are:

1. cosign verifies the signature, CI identity and OIDC issuer, then checks the
   `dist/current-release.json`, release ID and delivery SHA bound by the attestation.
2. Compare the Profile template with the live Profile; when a policy change lacks a matching state approval, only
   `profile-plan.json` is generated and the run stops. The administrator reruns the same deploy with
   `--approve-profile-plan <plan_sha256> --reference <ref>`, which approves and continues the deployment.
3. Atomically prepare the Profile snapshot, Agent config digest, site release and per-role image digests.
4. Determine NOOP through `release-diff`, which reads only release state; that decision does not require the old and new release IDs to be the same.
   Non-runtime changes such as `release_delivery/rendered_manifests` can still be classified as
   NOOP when the release ID changes. Non-NOOP runs preflight and the component DAG:

   | Code change | Typical classification | Actual action |
   |---|---|---|
   | Artifact files, configuration and Profile all unchanged | `NOOP` | Fast path with no rollout, only one final verify |
   | Only the Control Plane closure changed | `CONTROL_PLANE_ONLY` | Roll only the CPU ingress/worker/spool |
   | Wheel licence or packaging metadata changed, code digest unchanged | `DATA_PLANE_COMPATIBLE` | Upload the changed artifacts by physical SHA, establish the compatible window, then a minimal rollout |
   | Compatible Executor or Node Runtime change | `DATA_PLANE_COMPATIBLE` | Roll only the affected GPU data plane or node Agent |
   | Schema, protocol, Agent configuration, Profile or cluster set change | `FULL` | Digest classification; the DAG executes only dependent nodes and keeps the two-phase pin |

5. Each phase and cluster writes a checkpoint; a shared release advances GPU clusters serially in site order by default, and
   each cluster records `PENDING -> RUNNING -> CONVERGED`, with `FAILED/PAUSED` on exceptions.
   `upgradeMaxParallelClusters` (`upgrade_max_parallel_clusters` in the release config,
   default 1, cap 8) turns cross-cluster progress into bounded parallelism: all clusters share one re-entrant state lock, and
   `component_progress`, `cluster_attempts` and the checkpoint are completed within the same read-modify-write,
   so parallel clusters do not overwrite each other's subtrees. The first failure closes the gate immediately; clusters not yet started are written to
   `not_started_cluster_ids`, and clusters already advancing finish their current wave sequence before the summary;
   `failure_scope` still decides `PAUSED` or `FAILED` by whether everything is cluster-local.
   Automatic rollback does not use this switch; `component_progress.clusters[*].updated_at_epoch` guarantees serial restoration in reverse order of the actual
   mutation time. Nodes use
   a deterministic FleetDeployment ID and `waves/next-wave`, and reuse the original wave after an interruption.
   `upgradeMaxUnavailable` defaults to 0 (auto, taking the size cap), cap 32, and the first wave is fixed at 1; a topology containing UNKNOWN forces 1, while under known
   failure domains the per-cluster cap adapts to 4, 8, 16 or 32 by node count and is distributed evenly, and an explicit value takes the smaller of itself and the cap.
   `rollbackMaxUnavailable` defaults to 2, cap 4, and the first rollback wave is fixed at 1; fewer than 6 nodes or UNKNOWN
   forces 1, 6 to 31 nodes allow at most 2, only 32 or more nodes allow 4, and with multiple failure domains each domain gets at most 1 per wave.
   After the affected artifacts are uploaded and before the schema Job and CPU rollout,
   `candidate-preflight-ready` is completed first: the candidate GPU/DCGM/Reconciler/Installer Manifests run a
   server-side dry-run, the node-level Action key and regional Secret references are checked read-only, and a read-only
   host-root preflight Job proves the bundle/wheel digests, disk, Python 3.12, systemd, NVIDIA/EFA tools,
   quiesce residue and the old runtime rollback slot. Cluster preflight runs at most 4 lanes, admission independently at most 8 batches, each batch at most
   4 nodes/8 Jobs/1MiB; the read-only host Job uses a sliding window of at most 8 nodes per cluster, and as soon as any node finishes
   the next node starts, without waiting for the slowest node of the whole batch.
   Host Jobs start only after all dry-runs pass; after a failure, no new nodes are started and the nodes already started are allowed to finish.
   A forcibly terminated subprocess is identified through a bounded liveness check rather than relying only on the exit notification, so as not to wait for a notification that can never arrive.
   The private input snapshot stores only the topology and
   the verified Secret references, never credentials; `--node-inputs` is only for rendering or read-only preflight and cannot authorize
   a real install. The node inventory cache is isolated per thread context. Results are summarised in node order, and any node failure fails the whole phase.
   When that phase fails, the CPU/GPU processes,
   the Reconciler and the Fleet wave have all had no mutation. Before the first node's mutation a dynamic barrier runs again, re-reading the target Node set/UIDs/failure
   domains, Ready, cordon/quarantine taints, and verifying the Executor, remote commands, destructive workflows and
   Agent leases; these facts, which depend on candidate processes or the latest runtime state, cannot simply reuse an early checkpoint.
   The only exemption from that barrier is **a node parked in the hot spare pool**: for an unreserved node with
   `gpu-fault.io/spare-pool-state=AVAILABLE`, its cordon and the derived
   `node.kubernetes.io/unschedulable` taint do not count as blockers. A hot spare must stay cordoned
   (`HyperPodSpareCoordinator` refuses a schedulable candidate with "unreserved spare is schedulable"),
   so reading every cordon as "someone is draining or repairing this machine" would make any site that declares a hot spare pool
   unable to upgrade -- the barrier fails, the release rolls back, and the customer can only choose between "keep the hot spare" and "accept the fix".
   A parked hot spare carries no workload and can only be assigned while the node agent stays fleet-ready,
   so it must be upgraded with the waves; the installer itself already tolerates the cordon taint, and the rollout never
   uncordons any node, so the pool invariant is unaffected. Nodes that are reserved, in a non-AVAILABLE pool state,
   carrying a quarantine taint, not Ready, being deleted or with an installer still running block as before.
   As long as `data-converged` has not been reached, an interrupted resume re-runs the read-only candidate/host preflight; the persisted
   checkpoint is used to recover ordering and cannot prove indefinitely that the disk, rollback slot or resource references have not drifted.
   The GPU/DCGM Manifests are server-side dry-run once more before the actual apply, to prevent partial resource mutation caused by
   admission or RBAC drift after the early proof; the read-only host probe is not repeated within the same normal transaction.
   The remaining phases also pay no needless waiting for inputs that "have not changed": `endpoint-ready` first reads the live CNAME, and
   when Name/Type/TTL/Value exactly match the expectation it submits no UPSERT and therefore does not wait for Route53 INSYNC propagation
   (rollback writes a different value and still submits and waits as usual); the role-split script called by `cpu-staged`/`cpu-finalized`
   reads `gpu-fault-release-metadata` only once and takes all pin fields locally,
   and composes all ConfigMaps of the selected role into one multi-document stream applied once. The normal path writes the wheel, image,
   notification/capacity configuration, failure-domain and pin digests into the final Pod template in one go, no longer apply first and patch/restart afterwards.
   Pin changes no longer roll the control plane: `gpu-fault.io/pin-config-sha256` is recorded in the Deployment's metadata annotations,
   control-plane processes hot-reload pins from that ConfigMap (`gpu_fault.fleet_pins`), and the script then uses
   `deploy/control-plane/tools/wait_control_plane_pins.py` to wait until every Pod of the selected role reports the same digest at `/healthz`;
   an explicit force uses a new restart timestamp; non-force keeps the original timestamp. When historical manual environment variables exist, `apply_control_plane_deployment.py`
   first obtains the final object from a server-side dry-run, removes only fields that are retired and not declared by the candidate, then performs one replace by UID and
   resourceVersion; on conflict it retries the full preview a bounded number of times, without an extra `set env` rollout.
   Other live fields are kept, and old environment variables in an explicit rollback snapshot are not filtered. When the Fleet wave waits for
   Agent convergence it polls `get nodes` at a fixed 5 seconds, no longer backing off to 15 seconds (the back-off would only idle up to 10 more seconds
   after the installer had already finished). None of this changes any gate's decision criteria; it only removes the idling between gates.
6. A Profile change queries the Aurora Store before CPU finalize; it fails if the old Profile's activity has not dropped to zero.
7. Component `STARTED` and `FAILED` are written to release state immediately; multiple components within the same GPU action/wave share
   one timestamp and are merged into a single checkpoint write. `COMPLETED` is first updated in memory and merged to disk by the immediately following phase
   or cluster checkpoint. A process exiting between the two only replays that component idempotently and never loses
   rollback evidence of a mutation that has begun. Then the change-specific quick gate runs, followed by exactly one full verify with at most 8
   parallel lanes, saving `verification-report.json`. That verify and the previous snapshot capture pre-read the Deployment
   list per context and reuse the JSON result within the report scope, never across a mutation. The quick gate's independent verifiers run in parallel and
   evidence is written only after all succeed; the runtime identity check covers all Pods and shares the global 8-lane budget.
8. A release with actual changes runs the critical convergence gate before the formal stability window. Only when the current critical set is exactly
   `GpuFaultStoreIoRejected`, and `gpu_fault_store_io_rejections_total` of every running CPU Pod is
   fully covered by process and reason and is 0, may it wait at most 420 seconds
   for the old series to leave the 5-minute rate window; any other critical fails immediately. After clearing, a 120 to 300 second
   stability window runs, checking Pod restarts, Ready, the Processor queue trend, remote commands and critical
   alerts, and saves `stability-report.json`. verify and stability are independent of each other, run in parallel, and either failure
   triggers only one rollback. The queue trend criterion is relative growth: sustained growth is declared only when the last three samples are non-decreasing, the difference between the last and first sample is at least
   `max(GPU_FAULT_RELEASE_QUEUE_GROWTH_MIN_DEPTH (default 8), half of the first sample)` and the oldest request age
   exceeds two sampling intervals; within 600 seconds after the data plane has just converged, the critical alerts listed in
   `GPU_FAULT_RELEASE_STABILITY_GRACE_ALERTS` (default
   `GpuFaultCollectorSilent,GpuFaultCollectorMetricsSnapshotStale`) are not counted, while
   other criticals still fail immediately. When `release_diff.kind` is `CONTROL_PLANE_ONLY` the window may end as early as 60 seconds
   (two clean samples after the baseline), provided every sample is clean (no critical alert at all, including the grace-listed ones) and
   all control-plane Deployments have converged (`replicas == updatedReplicas == readyReplicas ==
   spec.replicas`, `observedGeneration == generation`); the report explains the applicable window with `early_exit` and
   `early_exit_reason`. The quick verification evidence written by deploy at `complete` while not yet committed can be reused by the following
   verify (`pending_commit` with the same release ID), skipping only the read-only verifier scripts. When verify or
   stability fails and the site has no rollback (fail-forward release), the transaction stops at `complete` with
   `transaction_committed=false`: rerunning deploy with the same candidate resumes into commit; a deploy with a **different candidate**
   first commits that live transaction under its own identity (`commit_live_release`→`save_recorded_state`, not
   `save_state`, otherwise the candidate's digests would overwrite the state describing the live release), then opens its own transaction with it as `previous`, and
   stderr prints `commit-live-release release_id=… reason=…`. The verify of deploy #25 on 2026-09-09 failed because
   the check itself was defective; the old candidate could never pass verify and the new candidate was refused by the engine with
   `resume release_id does not match the candidate`; this path was added for exactly that.
9. `release-summary.json` is generated only once, at the end; NOOP only registers the new release state and runs one full
   verify, skipping the mutating deploy, the repeated quick verifiers and the stability window.

Failure behaviour:

- CI gate, OCI push, attestation or signing failure: no deployable release is produced;
- signature verification failure: the deploy host modifies neither the site nor the cluster;
- preflight failure: no rollout starts;
- upgrade failure with `autoRollback=true`: restore the previous stable artifacts, pins, Profile, regional cluster
  registry, CPU role ConfigMaps, DCGM and node identities. previous is captured only for the surfaces the execution plan will
  mutate; CPU-only does not read the GPU Deployment, Installer or Agent rollback identities, and does not read the rules blob when AMP is
  unchanged. The canonical previous JSON is gzipped and written to immutable, content-addressed, chunked
  ConfigMaps; the main `gpu-fault-regional-release-state` keeps only the digest and chunk references, avoiding the
  ConfigMap 1 MiB limit; any missing chunk, non-canonical content or digest drift fails closed. The old embedded state
  stays compatible, and a later successful commit keeps only the snapshot referenced by the current state plus the 2 most recent old snapshots
  (`PREVIOUS_SNAPSHOTS_RETAINED=3`), deleting only older ones;
- every `save_state` also appends an audit record to the append-only `gpu-fault-release-history` ConfigMap
  (`history.ndjson`, keeping the most recent 200 entries): release ID, phase, sha256 of the state and
  execution plan, UTC timestamp, operator identity (the ARN from `aws sts get-caller-identity`,
  or `user@host` when unavailable) and the redacted command line; `gpu-fault-admin` mirrors the same record through
  `GPU_FAULT_RELEASE_HISTORY_DIR` into `history/history.ndjson` in the state directory.
  The record holds digests only and never a token or the state body; a write failure only warns and does not block the release;
- with `autoRollback=false`, a high-level verify or stability failure only persists the failure and
  `rollback.status=SKIPPED_POLICY`, keeping the original transaction for a resume after the fix; it must not roll back on its own;
- when the low layer has completed the automatic rollback but exits non-zero, the high layer only aligns the formal site, capacity desired and release state,
  and does not start a second rollback. commit and rollback first persist the irreversible terminal state, then idempotently delete the temporary
  Secret backups; a cleanup failure only resumes the cleanup and must not reverse-roll back a committed transaction;
- rollback compensates, based on the persisted `execution_plan`, component progress and cluster checkpoints, only the components
  that actually began mutation; CPU-only must not touch the GPU, and Executor-only must not reinstall the Agent. The low layer first
  validates the scope covered by the compensation plan, and the high layer runs the full site verify after completing the management-plane alignment. GPU clusters
  are restored serially in reverse order of actual mutation time; the candidate Installer Job and
  Agent FleetDeployment are cancelled only within the corresponding cluster/component scope, and the old/new A/B slots and immutable
  artifacts are kept until the rollback window ends. `rollback_timing` records failure detected, controller,
  per-cluster/per-wave, restore, verify and `T_safe/T_full`;
- before the first migration from a legacy release that installs wheels at start-up to an immutable runtime image, `sync-state`
  must record both the live real old image and the scanned rollback-compatible image with the old release CLI preinstalled. If at the start of the upgrade
  the real old image has drifted or the compatible image is not an immutable digest, it must fail before mutation;
- rollback restores the GPU data plane and nodes first, then the CPU; a legacy node that still satisfies the old artifact, config,
  node UID and Succeeded status and has no Installer Job skips the new Fleet protocol, and the old Agent is not forced to back-report
  the newly added bundle/template fields.
- when a legacy node has already been partially updated by the candidate, the candidate CPU and candidate Reconciler install the previous
  artifacts by Fleet wave; each wave exactly checks the protocol, agent version, artifact,
  compatibility, policy, Profile, config, Node Action key and node UID supported by the old version.
  bundle/template continue to be restored from the previous template and install package, but stay `None` in the old Agent heartbeat.
- rollback must reuse the immutable Installer template ConfigMap recorded by previous; it cannot re-render
  a template with different content and a different name. `regional_release_rollback_target.validate_rollback_node_template`
  validates, at the rollback entry, every cluster that is not yet complete and needs AGENT/RECONCILER compensation, before the CPU,
  refresher or Profile restore; each target is re-validated before it is actually compensated. A CPU-only rollback does not read unrelated GPU templates,
  and when the old trusted content pin or dependency binding is missing it must not restore other components first and fail afterwards.
  Idempotent replay is allowed while `rollback-data-restored` has not yet passed verify.
- every non-resume upgrade and `sync-state` must establish a new transaction state, clearing the previous release's
  `rollback_completed_phases`, cluster checkpoints, failure and result fields; only when the same
  uncommitted release is in `failed` or has a persisted upgrade checkpoint may an explicit resume reuse the current
  transaction checkpoint. resume must rebuild from persisted state and validate the retry diff, and must not fall back to the default
  `FULL`. resume validation must be checkpoint-aware: the release ID, execution plan, previous snapshot digest,
  CPU required/compatible window, cluster identities and node set must not drift; completed clusters must match the candidate,
  clusters not started must match previous, and failed or paused clusters may only be in the mixed states defined by the state machine.
- the installed-resource registry queries resources in batch by `scope + kind + namespace`, and uses a private `0600` temporary kubeconfig only for the standard
  `aws eks get-token` within its explicit validity period, reducing repeated authentication; without a
  validity period nothing is reused across requests. Generic exec plugins and other exec options keep the native kubectl protocol, with no manual caching.
- automatic rollback is forbidden when a `database_schema_version` change is not declared backward compatible; a change only to the ensure Job manifest or
  image executes idempotently at the same target version and is not treated as a DDL version jump; AMP rules and Alertmanager save the original blob before
  mutation and can be compensated automatically; the ADOT collector is likewise compensable: the snapshot reads the live ServiceAccount, ConfigMap, Deployment, PDB, Role and RoleBinding in the
  object order declared by the Manifest, strips the
  API server's own fields, records objects newly added by the candidate as `absent`, and on rollback applies the old objects, deletes the `absent` ones in reverse order
  and converges the collector with `rollout restart` plus `rollout status`, so ADOT Manifest and ADOT image changes
  no longer refuse automatic rollback; when the live site lacks a collector Deployment, the CPU namespace anchor must be verified and the
  explicit absence recorded, otherwise the snapshot fails closed. Resuming an old transaction whose previous
  snapshot has no ADOT objects refuses automatic rollback before rolling any component, rather than failing only at the observability
  restore. The endpoint (NLB Service + Route53 record) and cluster registry also have a compensation surface:
  the previous snapshot first captures the Service and CNAME record, rollback restores in reverse apply order, and the registry is republished after the
  rollback; only old transactions whose previous snapshot predates the endpoint capture contract still refuse automatic rollback and must be resumed with
  `spec.autoRollback: false`;
- the original upgrade exception and the rollback exception are saved separately, and neither overwrites the other;
- verify or stability failure: first atomically write `FAILED`, `failed_at`, the original exception and
  `rollback.status=IN_PROGRESS`, then call the low-level rollback; after the rollback completes, update to
  `PASSED/FAILED`. When the process is interrupted, the `IN_PROGRESS` evidence must be kept; it must not revert to a false error-free
  `VERIFIED` state, nor claim a successful release;
- Profile release failure: the active approval bound to the same plan and live baseline is kept for an idempotent resume; when the plan or
  baseline changes, the old approval is archived and invalidated;
- release summary failure: keep `COMPLETED` and the verify evidence already checked in, and also write
  `release_summary.status=UNAVAILABLE` and `completion_warnings`;
- after the fix, rerun the same `make release-deploy`. A transaction that is `failed` or interrupted at an upgrade checkpoint
  may only be resumed by the same release from the persisted checkpoint; `rolled-back` is a verified terminal state, and later candidates reuse
  the conservative diff set but open a new transaction from the current live baseline, not treating the old release ID as a resume target.
- after a verified rollback completes, the releaser must synchronise the formal site, the administrator capacity desired configuration and the release state
  to the previous live version. When a historical `rolled-back` state has not yet completed that management-plane synchronisation, `status` uses
  the previous immutable Manifest and Profile to build a temporary read-only baseline; it must not keep using the failed candidate to judge live health.
  The previous snapshot records the path from which that Manifest was read (`release_manifest_path`): rollback alignment first checks the
  identity by that path, then scans `source-snapshots/`; only when neither is found does it refuse, listing the search scope in the error
  (BOOT-020 chained candidates live under `boot020-work-*/wt-*/dist`, which the scan never sees).
- a historical `BLOCKED` predecessor may stop blocking the Profile/release gates only when the same incident is `RECOVERED`, the current successor is
  `SUCCEEDED` and `RESTORE_SCHEDULING` is complete. The public
  `workflow-reconcile` (`--dry-run`/execution are the same command) must also bind fencing, expired leases, no open command,
  no WAITING provider action, a FAILED source plan and no residual quarantine on GPU nodes; apply updates the predecessor, incident and plan through atomic Store
  operations, never deletes records directly, and deletion is still performed only archive-first.
- `partial-convergence` and upgrade checkpoints take the upgrade resume; `rollback-*`,
  `rollback-failed` and rolled-back-but-cleanup-incomplete states take only the rollback resume. A committed `complete` state whose backup cleanup
  is incomplete only retries the commit cleanup.

The capability matrix of the first legacy migration is a code contract:

| Surface | Actually supported by legacy | Not supported by legacy | rollback user |
|---|---|---|---|
| FleetRequest | protocol, agent version, artifact, compatibility, policy, Profile, config, node list, max unavailable | deterministic deployment ID, bundle, template | candidate CPU Fleet |
| Agent heartbeat / record | protocol, version, artifact, compatibility, policy, Profile, config, Node Action key | bundle, template | previous Agent |
| Node annotation | version, artifact, config, node UID, Succeeded | bundle, template | previous node |
| Node Installer Reconciler | version, artifact, config, node UID, Succeeded | allowed-node wave, bundle/template identity, controller max-unavailable | candidate Reconciler executes the wave; switch to the previous Reconciler after completion |

`previous.agent_identities` must be captured before mutation from real Agent records that are ACTIVE with a valid lease;
records within the same cluster must be fully consistent, and the node set must equal the target HyperPod nodes. The release metadata has no
agent version or policy version, and guessing them from candidate code or version numbers is forbidden.

During a release there must be no further build, and no manual editing of Deployments, pins, generated Manifests or
`agentConfigDigest`.

### 9.2 Development Process for Changes Involving Configuration, Profile or Resources

1. Determine whether the change is a code extension, a production delivery or both, and enter the corresponding document via the selection table at the end of this document.
2. Determine whether the component lives on the CPU or GPU, its runtime phase, permissions, Profile owner, pin, rollback and uninstall boundary.
3. Modify the source code, source Manifests, renderer, site schema and resource registration logic.
4. Fill in resources, probes, termination grace, security context and negative tests.
5. Run the required generators and review the generated Manifests, cleanup inventory and environment variable reference.
   A PostgreSQL DDL change also appends a consecutive schema migration; the current baseline is
   `schema_migrations.LATEST_POSTGRES_SCHEMA_VERSION` (currently v18), the new migration version number
   must be consecutive +1 (the next version is v19), and it must pass `make test-postgres-stress` or the CAP-005 runner.
6. Run the first-step build and sign:

   ```bash
   COSIGN_SIGNING_KEY=/secure/release/cosign.key \
   make PYTHON=.venv/bin/python release-build \
     RUNTIME_IMAGE_REPOSITORY=<registry/repository>
   ```

   Then run `release-deploy` directly; by default it uses the signed `dist/current-*` artifacts generated in the first step.
   Developers only modify
   the policy template pointed to by `runtimeProfile.templateSource`; the script automatically generates an immutable
   Profile version and snapshot from the canonical content. When the policy really changes, the first deploy generates the plan and stops; after review, rerun the same deploy with
   the approval arguments, which approves and continues the deployment:

   ```bash
   gpu-fault-admin deploy \
     --state-dir /secure/gpu-fault \
     --approve-profile-plan "$(jq -er '.plan_sha256' \
       /secure/gpu-fault/release-deploy/profile-plan.json)" \
     --reference CHG-12345
   ```

   When the content is unchanged no approval record is needed. Hidden arguments and environment variables must not replace this state approval.
7. `release-build` runs static first, then ordinary pytest, PostgreSQL stress and the source-only
   artifact in parallel within budget, and after all pass performs OCI reuse/build and signing;
   `release-deploy` verifies the signature, computes the production Agent config digest, generates
   `<site directory>/profiles/<content-digest>.yaml`, atomically prepares `site.yaml`, then runs
   `deploy -> verify -> stability -> release-summary`. The reports are saved as
   `verification-report.json`, `stability-report.json` and `release-summary.json` respectively;
   the full `status` is not executed repeatedly
   within the same release. When the component diff is clearly `NOOP`, even if the release ID changes because of the delivery identity, the
   `deploy` phase still records
   `SKIPPED_NOOP` and goes straight into the parallel verify; any uncertainty falls back to the full deploy. The actual rollout scope
   is still classified automatically by the regional release diff as
   `NOOP/CONTROL_PLANE_ONLY/DATA_PLANE_COMPATIBLE/FULL`.
8. On staging, additionally verify `join-cluster/remove-cluster/uninstall` according to the change scope, and use the low-level
   `plan/upgrade/resume/rollback` to rehearse failure resume and rollback; do not substitute
   `kubectl rollout undo` for the regional rollback.
9. Submit the source code, source Manifests, generated artifacts, tests, administrator documentation and evidence notes in the same PR.

The release state and site backups live at:

```text
<site directory>/release-deploy/<release-id>/
```

This directory contains at least `plan.json`, `state.json`, `verification-report.json`,
and, for actual changes, `stability-report.json` and `release-summary.json`. When the administrator later runs
`gpu-fault-admin status` separately, it still re-fetches the
full live health state and does not reuse historical verify reports.

A build failure does not modify the site; a deployment failure keeps the target site and the `FAILED` state, the online release is handled by the existing
`autoRollback` policy, and after the fix simply rerun the same command.

### 9.3 Supply Chain Checks

The CI static job runs `make ci-supply-chain-tools` after yamllint (it installs the pip-audit and cyclonedx-bom versions pinned at the top of the Makefile
into an independent venv without touching the hash-locked environment; it also downloads Prometheus's `promtool` by version number and
the SHA-256 of the release tarball), followed by
`make promtool-check` (`scripts/check-alert-rules.py`), which hands `deploy/observability/amp-rules.yaml`
and `deploy/control-plane/regional/processor-alerts.yaml` (after stripping the PrometheusRule shell) to
`promtool check rules` -- `verify-regional-alerting.py` only checks runbook/annotation/aggregation shape;
whether the PromQL itself parses is decided by promtool alone. The target also runs the complete `PROMQL_TESTS` set declared by the
Makefile with the real tool at the pinned version,
requiring a fully successful pytest receipt. These two groups of tests belong to the always-re-executed static gate and do not reuse the
coverage shard cache; a missing local promtool likewise fails and is not recorded as a skip.
`test`, `test-shuffled`, `test-parallel` and `test-parallel-release` run
`promtool-preflight` before pytest; the check/release modes of `run_release_gates` do the same pre-check before the expensive gates.
Make uniformly exports `PROMTOOL`, by default from the independent tool environment; an explicit configuration must still pass the executability and exact version
checks. `check-alert-rules.py --tool-only --version ...` checks only the tool, reads no rules, starts no
pytest, and does not treat a missing tool as skippable. When missing or the version mismatches, it points to the verified install entry point and does not install from the network automatically.
This prerequisite applies to source builds that actually run local tests; deployment paths that only consume signed CI artifacts and run no local tests
require no extra tool because of it. Test subprocesses still clear the signing password and unauthorized database connection environment.
Targeted impact tests decide from Make's `PROMQL_TESTS` list whether the selected files, directories or node IDs need the tool,
and when they do run the same pre-check before other checks and pass the returned verified absolute path to the actual pytest subprocess;
they do not rely on an earlier Make subprocess having modified the Python parent process's environment. Selections without PromQL add no tool dependency, and
the original test selection, risk classification and independent PostgreSQL authorization are unchanged.
`make pip-audit-check` audits with `--require-hashes` the five shipped locks listed in the Makefile's `SHIPPED_LOCKS`
(`requirements/build.lock`, `requirements/runtime.lock`, `requirements/deploy-host.lock`,
`requirements/node-runtime.lock`, `requirements/node-tools.lock`); accepted vulnerability ids may only be written in
`requirements/pip-audit-ignore.txt` with the fixed version noted. `make lazy-export-check`
(`scripts/check-lazy-exports.py`) parses every `_EXPORTS` lazy export table and the declared entry points and imports them one by one;
`make doc-facts-check` (`scripts/check-doc-facts.py`) runs with `make docs-check` to verify that the declared
documentation assertions match the code. The three `release-build*` targets run `make sbom` before generating the attestation, writing each
lock's CycloneDX SBOM to `dist/sbom/`, and `--sbom-dir` binds each file's SHA-256 into the cosign-signed
attestation; when the tools are missing locally these targets only print a skip, while under `CI=true` a missing tool fails.

After the optional HMA CloudWatch chain was decommissioned, the last CloudFormation template and its dedicated cfn-lint
install/check targets were deleted together; the YAML, lazy export, pip-audit, SBOM and promtool gates remain.
The retired resource table of `generate-cleanup-inventory.py` generates only historical cleanup entries and cannot carry a deploy
mode or reappear in the deployable Manifests. The candidate `preflight_retired_collectors` continues applying changes only after confirming on each GPU cluster
that the old producer Deployment has been decommissioned; for the handling order of old queues and evidence see the Deployment and Operations
Manual §7.4. Nothing here changes the Runtime Profile's AWS provider responsibilities or any recovery authorization.

When fixing a transitive dependency reported by pip-audit, do not hand-edit the locks: `uv.lock` is the single source, and the three locks are exported by the commands recorded in their own file
headers. Closing the 6 PYSEC entries of urllib3 2.3.0 on 2026-09-07 followed exactly this route: urllib3 is a transitive dependency of
botocore/requests/kubernetes, pinned by the `urllib3<2.4.0` of `kubernetes==34.1.0`,
so `kubernetes>=31,<35` in `pyproject.toml` was relaxed to `<36`, and uv 0.8.12 (the same exporter as the existing locks,
able to reproduce both exports byte for byte) ran `uv lock --upgrade-package kubernetes --upgrade-package
urllib3`, yielding kubernetes 35.0.0 and urllib3 2.7.0 and dropping google-auth,
cryptography, cffi, pycparser, pyasn1 and pyasn1-modules, which kubernetes 35 no longer depends on; then each lock was re-exported verbatim with its file header command:
`uv export --frozen --no-dev --no-emit-project --extra collectors --extra postgres --extra performance --format requirements-txt --output-file requirements/runtime.lock`,
`uv export --frozen --no-dev --no-emit-project --extra dev --extra collectors --extra postgres --extra performance --format requirements-txt --output-file requirements/deploy-host.lock`,
`uv pip compile --generate-hashes requirements/build.in --output-file requirements/build.lock`
(the latter has no urllib3, zero changes); the fourth, node-side lock is exported with
`uv export --frozen --no-dev --no-emit-project --extra collectors --format requirements-txt --output-file requirements/node-runtime.lock`
without any `--no-emit-package` (see the node installation section of the Deployment and Operations Manual); `git diff` should show only the version/hash lines of the target packages and the dependencies forced to move with them; once each lock
passes `pip-audit --require-hashes` with zero findings, delete the ids from `pip-audit-ignore.txt`. A change to `deploy-host.lock` or
`build.lock` changes `dependency_identity_sha256`, so the signed deploy-host bundle must be rebuilt
and re-signed before the next live deployment; a `runtime.lock` change goes through the normal Runtime Image release.

### 9.4 The Regional Release Orchestrator Is a Package; Coverage Floors for Deployment-side Source

The regional release orchestrator (rollout / rollback / join / status and so on, 75 modules, about 31 thousand lines) is the
`src/gpu_fault_release/` package, whose entry module is `src/gpu_fault_release/rollout.py`; modules import each other exclusively with
`from gpu_fault_release.<module> import …` absolute imports. It used to be
a pile of scripts in `deploy/control-plane/regional/*.py` importing each other by bare name, loadable only by pushing that directory into
`sys.path`; `tests/test_release_architecture.py` now guards this boundary -- no
file in the repository may import an orchestrator module by bare stem, nor push the package directory into `sys.path`. The administrator CLI
(`src/gpu_fault/admin/cli.py`, `site.py`, `cluster_join.py`, `cluster_removal.py`,
`rollback_command.py`) and `scripts/release_deploy.py` still exec
`deploy/control-plane/regional/rollout-regional-release.sh` by path; it is now a thin shell: it puts
`<repo>/src` into `PYTHONPATH` and then `exec python3 -m gpu_fault_release.rollout "$@"`,
passing argv and the environment through unchanged, so a clean checkout without the wheel installed still runs. The non-Python inputs did not move:
`deploy/control-plane/regional/generated/`, the patch/prerequisites YAML,
`cleanup-inventory.json` and `probes/` (which the orchestrator sends to Pods as source text for execution, not imported)
remain under the deploy tree, and the orchestrator resolves the repository root through `gpu_fault_release.repository_root()` (resolution order at the
start of §9): from the current source checkout or the site bound to the verified venv, not copied into the business Runtime image;
wheel/bundle relative paths inside the Manifest resolve against the repository root where the manifest lives, so probe changes take effect with the next deploy's
snapshot. The data-plane observability install scripts and manifests are also read through this locator and must not guess the deploy tree location from a fixed parent of the wheel's
`site-packages` directory. The import closure of `scripts/component_wheels.py` tracks both
`gpu_fault` and `gpu_fault_release`, and the orchestrator Python is installed together with the independent `gpu-fault-deploy-host` wheel
(`component_wheels.LOCAL_PACKAGES`), while at runtime it is still started by the thin shell from the snapshot `src`; it does not enter the application
`module_digest()` but is hashed by the deploy-host identity `scripts/deploy_source_identity.py`'s
`DEPLOY_HOST_ORCHESTRATION_INPUTS`, whose inputs cover `src/gpu_fault_release/*.py`,
`deploy/control-plane/regional/*.sh` and `probes/*.py`, `deploy/control-plane/tools/*`,
the start-up scripts and bounded Job waiter (including the prerequisite repair, database proof and dynamic Job management helpers), and the release driver
scripts (`scripts/release_deploy*.py`, `run_*_gates.py`, `ci_gate*.py`, `staging_*.py` and so on);
therefore a probe-only change is also classified as `DEPLOY_HOST_ONLY`. `native_http`, process supervision and the SNS policy helper are also in the
deploy-host closure and do not enter the three business component wheels. The shared `amp-sns-publish-policy.json` is included by
`config/release-identity.yaml` in both the Manifest inputs and the observability component identity, and
`bootstrap_task_inputs` also binds that asset and `monitoring_policy.py` so that the owning checkpoint is invalidated.
The `deploy/migrations/postgres-schema-preflight-job.yaml` reused by the database proof is covered by the same Manifest
input rule; probe source sent to CPU Jobs must come from this run's trusted deployment source and must not take an unbound script from elsewhere.

The SES configuration set maps from `spec.notifications.sesConfigurationSet` to the low-level
`notifications.ses_configuration_set`, validated by the SES naming rules;
`render_notification_config_maps` is the structured YAML modification entry point shared by plan and CPU apply.
All selected CPU roles consume the same value, which is explicitly cleared when undeclared; it cannot inherit notification or old snapshot control from the shell.
The notification digest keeps the old format when the field is undeclared; rollback prefers the really captured old ConfigMaps and per-role env.
preflight only queries the declared existing SES configuration set and verifies its name and sending status; it creates no resources and extends no permissions.

`probes/workflow_safety.py` is the `workflow_safety` gate of the release preflight:
destructive workflows in `PENDING`/`SAFETY_PENDING`/`RUNNING`/`BLOCKED` count as
`blockers` by default. They may be excluded only when the complete safety predicate of `resolved_blocked`, `abandoned_generation`, `compile_blocked`
or `settled_incident_blocked` respectively holds; the latter two are also closed by the
dispatcher sweep. `settled_incident_blocked` requires the incident to be `RECOVERED`;
`ESCALATED` must not be treated as recovered; an existing owner, an unexpired lease, a source plan, an unsettled remote
command, a still-occupied node or an unconfirmed node/provider action cannot release the occupation on the incident's terminal state alone.
The probe and runtime predicates must stay consistent,
verified by `tests/regional/test_release_workflow_safety.py`. Source snapshots and deploy-host
delivery do not change these runtime safety conditions.

`coverage.sources` in `config/ci-unit-gate.json` (schema 4) lists five source roots under test:
the production scope is `src/gpu_fault`, `src/gpu_fault_release`, `deploy/control-plane/tools`, and the
runner scope is `scripts/e2e/regional`, `tools`. The two scopes are measured independently and each enforces 95% statement/branch;
the production scope's 78% combined floor and the original per-module floors are kept. The two deployment-only roots also
appear in `coverage.deployment_only_sources` and are measured only by the `deployment` shard (the `source` of the runtime,
fault_runner and postgres shards does not contain these two roots at all, so their identities and floors are unaffected by
deployment-side source changes; if files from these two roots appear in a non-deployment shard's data, `verify_coverage_data`
refuses outright). Tests `from gpu_fault_release import …` like any package, and coverage records by path.
`deploy/control-plane/regional/probes/*.py` is outside the floors of these five source roots, but that must not be taken to mean
AST validation suffices. `test_store_io_zero_view_probe` actually executes the standard-library script sent to the Pod, covering legal, incomplete, malformed and read-failure scenarios with
a replaced urllib transport; other probes must also get behaviour tests according to risk.

The Store-I/O alert-clear wait accepts only a complete zero view: the aggregator must report degraded=0 and an
integer process-count=1..16 uniquely and without labels; the number of bounded process slots must match, and each slot has exactly the three finite non-negative zero counters capacity,
deadline and backend_unavailable. Slots may be non-contiguous and existing process_ids
may coexist, but they cannot replace slots or treat duplicate slots/reasons as different series. Unknown, missing, duplicate, non-zero,
malformed or over-8MiB responses are all refused; the HTTP response is closed on every path and raw data is not echoed.
The old merged counter format is insufficient to prove completeness and does not authorize the fast alert-clear wait; this constrains only the existing
transient critical convergence branch and does not change general rollback, the no-critical path or the established wait/stability windows.

`regional_release_interfaces` provides narrow Protocols for the snapshot, CPU rollout and GPU rollout;
upgrade/rollback options and node sets use explicit types, and at key boundaries the typed methods of `RegionalRelease` delegate to
the original implementation. JSON continues to be validated at runtime at the original persistence entry points; TypedDict does not replace signature verification, state validation or the rollback barrier.

Every module of the deployment-side source must belong to a `coverage.module_floors` family
(guarded by `tests/test_ci_unit_gate.py::test_module_floors_cover_every_deployment_only_source_file`):
`regional-release-orchestrator` (`src/gpu_fault_release/regional_release_*.py`,
`rollout.py` and `__init__.py`), `regional-deployment-support` (the remaining
`src/gpu_fault_release/regional_*.py` listed by name), `control-plane-tools`
(`deploy/control-plane/tools/*.py`). Floors come from the first measurement: the family floor is the measured value rounded down minus 2, and
the file floor is the lowest file in the family rounded down; the purpose is to stop the numbers sliding, not to claim compliance. To raise a floor: run
`make coverage-shard COVERAGE_SHARD=deployment COVERAGE_SHARD_ROOT=/tmp/x`, generate the report with
`coverage json` (with the three roots in `source`), confirm that `python scripts/ci_coverage_gate.py module-floors
--coverage-json <report>` passes, then write the new integer into the corresponding family; it may only go up. When adding a
`gpu_fault_release` module, add it to the `globs` of `regional-deployment-support` (or create a new family),
otherwise that guard test fails.

### 9.5 mypy Convergence Direction

`mypy-baseline.json` ratchets item by item on (file, error code); it can only prevent growth and does not express "what to fix first".
`mypy-convergence.json` adds the direction: per directory it gives the cap (`ceiling`) on the strict error count, the priority and the rationale;
the current five directories in priority order are `src/gpu_fault/store`, `src/gpu_fault/orchestration`,
`src/gpu_fault/app`, `src/gpu_fault_release`, `deploy` -- they carry the vast majority of the baseline's
`no-untyped-def` / `type-arg` / `no-any-return`, and are exactly the three layers plus the release/rollback path where "wrongly shaped dict" is the main failure mode. Rules: a directory's
total exceeding its cap fails (the hint is to fix that directory's errors, not to raise the cap); being below the cap is not slack, and
`python scripts/check-mypy-baseline.py --write-baseline` tightens the cap to the current value and never lets it rise again;
`--report` prints `current/ceiling (rationale)` per directory. Adding a directory only needs one more line in that JSON, with the cap starting at the current baseline
total.

## 10. Document Selection

| Change | Required reading |
|---|---|
| Adding an operation, channel, Store, route, plugin or metric | [Extension Guide](extension-guide.md) |
| Modifying Manifests, renderer, systemd, AWS resources or the administrator CLI | This document |
| Adding both a capability and production resources | Read both and run both sets of gates |
| Only deploying an existing release | [Administrator Quick Deploy](administrator-quick-deploy.md) |

For detailed renderer and break-glass commands see
[the configuration management chapter of the Deployment and Operations Manual](deployment-and-operations-manual.md).
