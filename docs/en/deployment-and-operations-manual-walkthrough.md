English edition of `docs/部署和运维手册逐章解读.md`; the Chinese file remains the source of record until both are maintained together.

# Chapter-by-Chapter Walkthrough of the GPU Fault Handling System "Deployment and Operations Manual"

This document is a reading aid for [`部署和运维手册.md` (Deployment and Operations Manual)](deployment-and-operations-manual.md) and does not replace the
commands, parameters and acceptance criteria of the original manual. The original manual answers "what exactly to run"; this document answers "why this chapter exists,
when to run it, and what result you should get after running it".

For normal deployment administrators first read [Administrator Quick Deploy](administrator-quick-deploy.md), and for daily operations first read
[Administrator Operations](administrator-operations.md). Only when shared infrastructure is missing, the unified orchestrator fails, or an
item-by-item audit is needed do you enter the full chapters via this document. Security approval and parameter boundaries are in
[Security and Parameters Reference](security-and-parameters-reference.md), and Manifest/renderer changes in
[Developer Deployment Implementation](developer-deployment-implementation.md).

The accounts, clusters, VPCs, subnets, nodes and endpoints in the public manual are all variables or placeholders. Real environment
values and per-round execution results are kept only in the private evidence store.

## I. Choose a Reading Route First

### Route A: Regional Production Deployment

The normal path first runs
`deploy (with preflight built in) -> status --full -> training acceptance` from the "Administrator Quick Deploy".
Only when the prerequisite infrastructure is not yet prepared do you complete the detailed steps in the order
`0 -> 1 -> 2 -> 3 -> 4 -> 5.1 -> 5.3 -> 5.4 -> REG-8`.
This is the only production delivery form at present.

### Route B: Historical Single-Cluster Migration

Read Appendix A and §5.2 only when maintaining an old environment or migrating.
The single-cluster path has not passed complete acceptance at the same level as the regional form and is not used for new production deployments.

### Route C: Regional Data-Plane Node Delivery

REG-6 uses the Installer/Reconciler by default; when privileged Jobs cannot run, read
§5.5. Both install the same set of Collector/Node Agent.

### Route D: Regional Daily Operations and Fault Handling

First enter the corresponding runbook from the task/symptom index of "Administrator Operations"; when low-level evidence is needed, read
`REG-8 to 14, 6, 7, 8, 9, 10, EFA/RDMA, host resource monitoring, heterogeneous inventory,
configuration management and client identity constraints`. Do not widen the allowlist or disable fail-closed gates ad hoc for troubleshooting.

## II. Global Concepts

- **Greenfield deployment**: a new environment establishing Aurora, NLB, Secrets, schema and CPU/GPU workloads for the first time.
- **Application-layer bootstrap**: the shared EKS, Aurora, ACM, subnets and SGs already exist; only this solution's objects are deployed.
- **required pin**: the Agent artifact/config/protocol that must be used in steady state.
- **compatible pin**: a candidate or old version accepted temporarily only in the rolling-upgrade window.
- **fail closed**: stop the action, rather than guess, when evidence, identity, version or execution owner is incomplete.
- **Three-tier control plane**: ingress receives requests; control-worker handles faults and workflows;
  spool-worker optionally handles routine telemetry.
- **Regional Executor**: runs in the GPU EKS, claims the control plane's remote commands and operates the local
  Kubernetes, Node Agent or HyperPod API.

## III. Chapter-by-Chapter Walkthrough

## 0. Reading Order and Table of Contents

This is the entry point of the whole manual, resolving "which deployment path to take" and "why the chapters cross-reference".

### 0.1 Overall Order of a Regional Greenfield Deployment

- **Purpose**: make clear that a regional deployment includes both the AWS infrastructure layer and the application layer.
- **Order**: tests -> CPU-1 to 3 -> REG-1/2 -> CPU-4 -> CPU-5 to 9 ->
  REG-3 to 8 -> the two training submission methods -> final acceptance.
- **Key conclusion**: Aurora Serverless v2 and the public TLS NLB are both required components of the regional solution.
- **Pitfall**: the low-level `rollout-regional-release.sh bootstrap` is responsible only for the application layer;
  only the ARN-only `gpu-fault-admin deploy` first automatically creates the Aurora, LBC/IAM,
  NLB/PKI and monitoring resources exclusive to the current site; same-site tagged objects are taken over only as uncleaned leftovers.

### 0.2 Persisting Regional Variables and Restoring a Session

- **Purpose**: prevent the many shell variables from being lost on disconnect or carried as empty values into `sed/aws/kubectl`.
- **Method**: write the non-sensitive variables into a `0600 regional-env.sh` outside the repository and re-source it on every login.
- **Region**: filled in explicitly by the operator, not inferred from the shell or kubectl context; the CPU/GPU EKS ARNs
  must agree with that Region.
- **Sensitive values**: tokens, CA private keys, the fleet master and database passwords only enter Secrets,
  Secrets Manager or `${SECURE_DIR}`.
- **Pass criterion**: the variable, Region and EKS ARN gates all succeed, and CPU/GPU `/readyz` are reachable.

### 0.3 Keeping the EKS Clusters and Reinstalling

- **Purpose**: wipe this solution's state and reinstall, while keeping only the CPU EKS and all GPU EKS clusters.
- **Entry point**: normally use `gpu-fault-admin uninstall --cpu-cluster keep`; the low-level
  `prepare-clean-redeploy.sh --mode reset --node-mode uninstall` is only for break-glass.
- **What is deleted**: both sides' solution namespaces/components, node installations, the whole dedicated Aurora Serverless v2,
  NLB/SG, ACM/PKI and LBC/IAM.
- **Safety boundary**: the script calls no CPU/GPU EKS delete API; finding a GPU node still under quarantine
  refuses the reset. Aurora deletion requires an explicit choice on whether to keep a final snapshot.
- **Redeployment entry point**: return to §0.1; CPU-2 rebuilds Aurora, CPU-3/5 rebuild LBC/NLB,
  REG-3 to 8 reinstall on the original GPU EKS.

### 0.4 Permanent Decommissioning Plan

- **Purpose**: the solution will no longer be used.
- **What is deleted**: CPU EKS/HyperPod, Aurora, NLB, IAM, monitoring and all persistent state.
- **GPU boundary**: all GPU EKS/HyperPod clusters must be kept, with only this solution's components uninstalled.
- **Entry point**: `gpu-fault-admin uninstall --cpu-cluster delete`; the Aurora registry is exported first,
  and the Aurora final phase is entered only after the non-Aurora resources and the CPU cluster are verified.
- **Entry point**: REG-13; do not run the §0.3 reset first, or the final Aurora snapshot loses its audit data.

## 1. Scope

- **Purpose**: define the only production form and the non-delivery boundary.
- **Production form**: a regional CPU control plane separated from one or more HyperPod EKS GPU clusters.
- **Non-production/undelivered**: single cluster only as a historical transition; generic Kubernetes not implemented;
  systemd is a regional data-plane installation method.
- **Key conclusion**: the regional control plane must not directly apply the renderer inputs in `deploy/control-plane/base/`, nor run the single-cluster `deploy.sh`
  on the CPU control-plane cluster.
- **Sample resources**: the sample Region names, VPCs and EKS clusters are only validated samples and cannot be copied to a new environment.

## 2. Pre-Go-Live Decisions

- **Purpose**: fix the globally stable identity and recovery responsibility before creating resources.
- **Must decide**: `cluster_id`, Runtime Profile, Store, control-plane URL, DCGM endpoint,
  lifecycle owner, the initial operation allowlist.
- **Key principle**: HyperPod managed recovery and this system's mutations must not become dual writers.

### 2.1 Mandatory Pre-Go-Live Checklist

- **Code gate**: `make check` and the real PostgreSQL tests pass.
- **Release gate**: the release metadata is complete, and the compatible fields are empty in steady state.
- **Secret gate**: execution token, processor replay secret and node action secret
  are all present and different.
- **Runtime gate**: exit budgets, PDBs, email/ADOT, Agent key version and collector readiness
  are all satisfied.
- **Significance**: this section is a go-live veto list, not a suggestion list; any failure forbids the release.

### 2.2 Hard Constraints of the Regional Production Solution

- HyperPod `NodeRecovery` must be `None`.
- This solution's code and execution roles never call `BatchReplaceClusterNodes`, and
  `GPU_FAULT_ALLOW_HYPERPOD_REPLACE` is always `false`.
- Job automatic resume must be disabled; training recovery is the responsibility of this solution alone.
- Only HyperPod EKS is currently supported, not HyperPod Slurm.

## 3. Software and Network Prerequisites

### 3.1 General

Lists the minimum dependencies such as Python, PostgreSQL, the NVIDIA driver, `nvidia-smi`, `/dev/kmsg`, Node Agent
9099 and DCGM 9400. When these fail, Kubernetes objects should not be created.

### 3.2 Stable Private DNS, Public NLB and Private CA

- **Purpose**: when the GPU VPC and CPU VPC are separate or their CIDRs conflict, establish a restricted TLS channel via public NLB + NAT.
- **Requirements**: the certificate SAN must cover the stable private hostname; the CNAME is published only after the NLB is active, the raw
  DNS resolves and the targets are healthy. GPU Pods must complete
  DNS/TLS/`/healthz` verification with the private CA; `curl -k` is forbidden.
- **Boundary**: the CA private key never enters a Kubernetes ConfigMap; the GPU data plane gets only the CA public key.

### 3.3 HyperPod EKS

Checks AWS CLI, kubectl, Helm, IAM, EKS subnets, Load Balancer Controller permissions and
HyperPod HMA/DCGM. The CPU/GPU contexts must be specified explicitly.

### 3.4 Boundary Between Solution Component Images and Customer Training Images

- **Solution images**: the Python runtime, node installation Job, DCGM Exporter, ADOT.
- **Training images**: entirely decided by the customer workload, never replaced by the control plane.
- **Enterprise images**: unify images or pin digests through the four `GPU_FAULT_*_IMAGE` variables.
- **Key conclusion**: the business version is determined by the content-addressed wheel, not by the Python base image tag.

## 4. Build and Test

- **Purpose**: obtain from CI a signed deployable schema v3 release containing separate wheels/
  bundle, rendered digests, dependency lock, Node template and immutable OCI image digests.
- **Gate contents**: Ruff, the full mypy/Mixin contract ratchet, compileall,
  the architecture ratchet, deploy layout,
  bash/ShellCheck, environment variable reference, case index, generated config, artifact consistency,
  the full pytest.
- **PostgreSQL**: use the CAP-005 runner to create an isolated PostgreSQL 16 database, run
  the 8×40 lock contention, completion concurrency 2/4, counter/fencing/deadlock tests and require
  zero skips in JUnit; `make check` itself does not connect to an external database.
- **Artefact discipline**: `make check` validates source-only artefacts; `release-build` generates and signs the
  deployable Manifest after pushing the OCI image, and the deployment host only verifies signatures and applies. The build pins the timestamp and
  archive order, and repeated builds of the same source must yield the same component SHAs and release ID.

## 5. Regionally Separated Production Deployment (the Only Production Form)

This chapter puts the CPU control plane, GPU data plane, training jobs and node delivery under one continuous numbering and is the only main deployment chapter a new production
environment needs to execute.

### 5.1 Regional CPU Control-Plane Greenfield Deployment

This is the AWS infrastructure and CPU application-layer runbook of regional mode.

| Section | Walkthrough |
|---|---|
| `CPU-1. Pin the CPU EKS Context and Check the Three Nodes` | Pins the kubeconfig and confirms three Ready CPU nodes, permissions and anti-affinity capacity. |
| `CPU-2. Create or Take Over the Same-Site Legacy Aurora Serverless v2` | Creates the subnet group, SG, Aurora cluster, writer, reader and the Kubernetes DSN Secret. |
| `2b. Deploy the Aurora Credential Refresh CronJob (Required Step When Keeping Rotation)` | Synchronises AWSCURRENT from Secrets Manager into the Kubernetes Secret; running Pods re-read the DSN through the mounted file, no rollout by default (`--restart-deployments` is a compatibility switch), and every run writes its conclusion into `last-refresh-status.json`. |
| `CPU-2R. Aurora Credential Recovery for an Already Deployed Environment` | When the password has rotated and Pods CrashLoop, compares AWSCURRENT/AWSPREVIOUS and safely repairs the DSN. |
| `CPU-3. Install the AWS Load Balancer Controller` | Lets a Kubernetes LoadBalancer Service create and continuously maintain the AWS NLB/IP targets. |
| `CPU-4. Create the Control-Plane Secrets and Deploy the Three-Tier Roles` | Configures the control-plane IAM, email, Secrets, registry, release pins and schema. |
| `CPU-4a. Create the Control-Plane Credentials and Confirm the First Registry` | Creates three different keys; the first-deployment registry must contain the first GPU, and later decommissioning of a single cluster allows an explicit empty array. |
| `CPU-4b. Create the Release Metadata and Initialise the Schema` | Reuses the tested wheel, computes the artifact/config/protocol pins, and runs the schema Jobs first. |
| `CPU-4c. Deploy and Verify the Three-Tier Role Split` | Deploys in the order spool-worker -> control-worker -> ingress and runs the verifier. |
| `CPU-4d. Register the Regionally Shared Runtime Profile` | Uses the stable cluster ID as the registration anchor; registers when missing, requires a new version when the same version's content drifts. |
| `CPU-4P. Wheel Reclamation After the Release Completes` | Deletes only unreferenced old ConfigMaps, protecting ReplicaSets, 0-replica Deployments and recent versions. |
| `CPU-4U. Counter Shards Upgrade for an Already Deployed Environment` | Switches the priority counters through dual/finalize/rollback while queue=0. |
| `CPU-5 to CPU-8. Automated NLB, DNS and TLS Gates` | hosted zone/certificate -> NLB Service -> active/TLS listener -> raw DNS -> healthy targets -> CNAME -> INSYNC -> GPU Pod DNS/TLS -> Executor. |
| `Legacy Reference: CPU-5. TCP Bootstrap of the Public NLB` | Only for recognising and migrating legacy environments; a new deployment must not substitute a non-empty hostname for the complete gate. |
| `Legacy Reference: CPU-6. Issue a Private Certificate for the NLB AWS DNS` | The old flow bound the certificate to the raw NLB DNS; it must not be mixed with the current stable private CNAME path. |
| `Legacy Reference: CPU-7. Whitelist the GPU NAT EIPs` | Network whitelist reference of the old flow; the current ARN-only deployment maintains the solution-specific SG automatically. |
| `Legacy Reference: CPU-8. Switch the NLB to a TLS Listener` | TCP->TLS switch reference of the old flow; the current Service uses the pre-issued certificate from first creation. |
| `CPU-9. Verify Pod IP Targets, TLS and Processor Active-Active Consumption` | Verifies the 3 ingress IP targets, the worker active consumer, no global leader, TLS, NLB failover and the 300-second continuous probe. |

### 5.2 Migrating from an In-Cluster Control Plane to the Regional Control Plane

- **Purpose**: move the all-in-one control plane inside the original GPU EKS to a standalone CPU EKS.
- **Order**: first build the regional control plane and data-plane credentials, then stop the old writer/Service/Secret.
- **Core risks**: two control planes writing Aurora at the same time, a leftover old execution token, the old Service still taking traffic.
- **Pass criterion**: the GPU cluster keeps only data-plane components, and the regional control plane becomes the only source of truth.

### 5.3 Static Registration of Regional GPU Clusters

| Section | Walkthrough |
|---|---|
| `REG-1. Pin the Control-Plane and Target GPU Cluster Contexts` | Operates on exactly one explicit CPU kubeconfig and one GPU context at a time. |
| `REG-2. Generate the Per-Cluster Token and Update the Control-Plane Registry` | Tokens are independent per cluster; other registry entries are kept; all non-zero CPU roles are rolled. |
| `REG-3. Prepare the Node Action Keys and the Private CA` | Derives node-specific keys from the CPU fleet master and distributes only the public CA and per-node keys to the GPU side. |
| `REG-4. Create the Connection Secret in the GPU EKS` | Writes the URL, cluster ID/token, namespace allowlist, HyperPod double-confirmation name and CA. |
| `REG-5. Sync the Executor Wheel and Deploy the Cluster Action Executor` | Uploads the separate Executor wheel, configures the data-plane-specific IRSA, and verifies the artifact, compatibility digest and real claim capability. |
| `REG-6. Install the Collector and Node Agent on Every GPU Node` | Enables the 4 collector systemd services and 1 Node Agent with the same bundle/pins; the NodeLogCollector unit stays disabled, and the Reconciler is enabled. |
| `REG-7. Verify from the GPU Data Plane` | Verifies TLS, token, Executor logs, collector status and Agent ACTIVE. |
| `REG-8. Regional Deployment Acceptance Checklist` | Verifies NLB, authentication, registry, Agents, Executor owners, backlog age and alerts. |
| `REG-8.1 Unified Regional Lifecycle Orchestrator` | Unifies first deployment, upgrade, resume, rollback, join/remove cluster. |
| `REG-8.1.1 Management Boundary` | The script creates no customer shared infrastructure such as EKS/VPC/Aurora/ACM. |
| `REG-8.1.2 Read-Only Checks` | `plan/status` read the release and per-cluster state without modifying resources. |
| `REG-8.1.3 First Deployment` | Uploads artefacts, schema, CPU/GPU Deployments, Reconciler, and waits for Agent convergence. |
| `REG-8.1.4 Upgrade, Resume and Rollback` | required keeps the old stable value, the candidate goes into compatible; finalize after the whole fleet succeeds. |
| `REG-8.1.5 Joining a Second GPU Cluster to an Existing Control Plane` | Appends to the registry without rebuilding Aurora/NLB, requiring the current global required release. |
| `REG-8.1.6 Removing a GPU Cluster` | Stops the target data plane first, then deletes the registry entry and rolls the CPU; removing the last GPU cluster is allowed, and the CPU control plane stays runnable with the explicit empty registry `[]`. |
| `REG-8.2 Manual Release Object Inventory and Three-Tier Rollback` | Centrally records CPU/GPU wheels, bundle, template and node annotations; rollback cannot undo only ingress. |
| `REG-9. Cluster Token Rotation` | One `gpu-fault-admin rotate-token` completes the overlap revision, GPU Secret, data-plane rollout, node fleet-wave reinstall, retirement log confirmation, atomic token file replacement and finalisation; resumable, `--rollback`-capable, never prints the token. |
| `REG-10. Bulk Migration of Node Endpoints and Credentials` | A one-shot privileged DaemonSet rewrites the CA, URL and host wheel; token rotation is now done by the REG-9 command. |
| `REG-10A. Component Artefact Consistency Check (Mandatory Before Every Deployment and Troubleshooting)` | Compares the module digest of the Control Plane, Executor and Node Runtime source closures with the running artefacts separately, ruling out "same version name, different content" first. |
| `REG-11. Cluster Action Executor Upgrade and Rollback` | Rolls while remote commands are 0; the old protocol must be in the compatible window. |
| `REG-12. GPU Cluster Deregistration` | Stops the Executor/Reconciler/node reporting, then revokes the registry/token and the NLB whitelist. |
| `REG-13. Permanent Decommissioning of the Regional Solution (Delete the CPU Cluster, Keep the GPU Clusters)` | Takes down in the reverse order GPU components -> CPU workloads -> NLB -> Aurora -> CPU EKS/IAM, proving each GPU EKS still exists. |
| `REG-14. Regional Mode Troubleshooting` | The regional-specific fault entry point. |
| `REG-14.1 CPU API Pod CrashLoopBackOff` | Check registry, Secrets, release metadata, Aurora DSN and schema. |
| `REG-14.2 Commands Stay PENDING (Most Common)` | Check Executor Ready, token, owner advertisement, protocol and unclaimed age. |
| `REG-14.3 Commands Stuck in LEASED` | Check lease renewal, Pod force-kill, adapter idempotency and timeouts. |
| `REG-14.4 A Step FAILED with an Unclear Cause` | Locate layer by layer from workflow step, remote result, executor/Agent logs. |
| `REG-14.5 All Nodes Become Non-ACTIVE` | Check token rotation omissions, heartbeats, artifact/config pins and node services. |
| `REG-14.6 Receiving 403 "regional mode requires a valid X-GPU-Fault-Execution-Token"` | Distinguish the operator execution token from the cluster token. |
| `REG-14.7 HyperPod Reboot Reports "unknown outcome" or "already reserved"` | Check the reboot provider observation window, idempotency keys and the real CloudTrail result; provider replace is always disabled. |

### 5.4 Regional Data-Plane Training Job Submission and Acceptance

The training workloads, Completion Watcher and Cluster Executor all live in the GPU EKS; the CPU control plane
only receives observations, terminal events and fault events.

#### TRAIN-1. Using the Unified Training Submission Command

- **Direct submission**: `gpu-training-submit` injects the metadata and calls `kubectl apply`.

##### TRAIN-1a. Inject the YAML First, Then Submit Separately

`gpu-fault-workload-annotate` only generates the YAML,
  suited to approval, scanning, GitOps and server-side dry-run.

##### TRAIN-1b. Customer-Defined Training Images

The system does not modify image/command/args; the customer is responsible for CUDA/NCCL/EFA,
  pull Secrets, resources and checkpoints.

##### TRAIN-1c. Three-Node 24 GPU / 48 EFA Example

Really executes 24-rank NCCL, DDP forward/backward,
  SGD parameter updates and loss decrease; the managed `expected-critical-ranks=3` means three critical Pods,
  PyTorch `world_size=24` means 24 processes; the large image must be pre-pulled before submission.

### 5.5 Regional Data-Plane Node systemd Installation in Detail

This is the alternative node delivery method of REG-6, not a separate architecture. Nodes still connect to the regional control plane and use the same set of
cluster token, CA, wheel, profile and node-specific keys.

#### NODE-1. Recommended Installation Command

Shows the complete command for installing the Collector and signed Node Agent on a standalone node, including CA, token, wheel,
DCGM, Node UID and advertise URL.

#### NODE-2. Installer Parameters

Explains the kernel, FM, metrics, host, node-log, Agent, reset, quiesce, diagnostic and inventory
parameters. Destructive parameters must be used together with `--enable-node-agent`.

#### NODE-3. Generated Files

Lists `/etc/gpu-fault`, `/opt/gpu-fault`, systemd units, state, ledger and diagnostic directories,
for backup, permission audit and uninstall.

#### NODE-4. NodeLogCollector Deployment Policy

Production disables the NodeLogCollector by default, avoiding duplicate log collection and capacity risk; enable explicitly and accept separately when needed.

#### NODE-5. Node Verification

Runs `/opt/gpu-fault/verify`, `systemctl is-active/is-enabled`, journal and network connectivity checks.
When node verification fails, the control-plane pin must not be promoted to required.


> The historical §6 and §7 corresponded to the undelivered generic Kubernetes deployment and have been removed from the public manual.
> The current §6 and §7 are the control-plane parameters and Collector/Watcher parameters after consecutive renumbering; the main manual keeps
> the old §8 to §13 anchors as compatibility entry points.

## 6. Control-Plane Parameters

### 6.1 Startup and Storage

Defines active/simulation, Store, execution/replay secrets, allowlist, workflow lease,
processor queue, remote command and Agent endpoint security parameters. Retryable replay backs off persistently for 1-30 seconds via
`not_before/retry_count`; STRICT lanes keep order, and only
latest-wins P100 requests may be reordered. Completion cluster concurrency is 1 in production; 2/4 only canary after
CAP-005 passes. The focus remains the mandatory items of active mode and the phased allowlist.

### 6.2 Adapters and Owners

Explains the switches and owners of Kubernetes, Node Action, Evidence, Validation and Managed Recovery.
The Runtime Profile owner and the running Adapter owner must match exactly.

### 6.3 Fleet compatibility

Defines Agent version, artifact, policy, profile, config digest, heartbeat age and the maintenance window.
Together they decide whether a node may execute a NodeAction.

### 6.4 Diagnostics, Training and Evidence

Configures the training health scan, heartbeat, step lag, throughput ratio and evidence TTL.
The Training Health Monitor is currently off by default.
That feature is not yet fully implemented and tested as a deliverable capability and currently cannot be used in production.

### 6.5 GPU Thresholds

Covers policies for temperature, PCIe replay, NVLink, ECC, row-remap, power/thermal violation and so on.
The defaults are a conservative site policy and must not be claimed as official NVIDIA RMA thresholds.

### 6.6 HyperPod

Configures the HyperPod adapter, managed observer, reboot/replace switches, spares, processor,
telemetry spool, PostgreSQL pool, NVSwitch topology and remediation. This is the densest parameter table for regional capacity and
recovery behaviour.

## 7. Collector and Watcher Parameters

### 7.1 Common Collector Parameters

Defines URL, cluster/profile, GPU discovery, sampling periods, filesystems, outbox, gzip and connection reuse.
The Collector must discover all GPUs; taking only the first card is not allowed.

### 7.2 Completion Watcher

Defines the watched namespace, poll/watch timeout, cleanup, log snapshots, GPU UUID discovery,
restart budget and emergency fallback. The Watcher is not a continuous logging platform.
### 7.3 Training Progress

Keeps the progress file protocol and reporter command, but production disables it by default. Before enabling, the sidecar,
atomic file updates and end-to-end validation must be completed.
That feature is not yet fully implemented and end-to-end tested as a deliverable production capability and currently cannot be used in production.

### 7.4 Collector Context and HMA

Explains workload context, KMSG, collector readiness and the retirement order of the historical HMA forwarding chain.
Both forwarding chains and their entry points have been deleted; the AWS built-in HMA and node safety checks remain.

#### HMA Forwarding Chain Retirement and Upgrade

First stop the flow, drain or archive evidence under the old release, then retire the self-built producers; the new version's preflight refuses to continue when it finds leftover
Deployments and does not automatically delete AWS stacks, queues or the AWS built-in HMA.

#### XID 154 Evidence and Correlation Acceptance

Verifies the 30-second correlation between the seven-register evidence of XID 144-150 and the XID 154 action code.
Missing registers or an unknown label must yield `BLOCKED_MISSING_EVIDENCE`.

### 7.5 Complete Node Agent Parameters

Covers node-specific HMAC keys, allowlist, reset/quiesce, diagnostics, driver/firmware, TLS,
heartbeat, endpoint SSRF constraints, CIDR whitelist and key rotation. It is the final safety boundary of node destructive
capabilities.

### 7.6 Fleet CLI transport Variables

The Fleet deployment transport reads node, cluster, artifact and so on from `GPU_FAULT_FLEET_*` environment variables,
avoiding concatenating unescaped values into shell.

## 8. Email Notifications

- **Role**: writes notifications asynchronously into the Aurora outbox, then the dispatcher sends via SES.
- **Key gates**: sender/recipient, SES region, dispatcher, watermark, deduplication.
- **Reading**: a notification ID or QUEUED does not equal delivered; wait for `SENT` and the provider ID.

### 8.1 GPU Count Change Approval

Compares source/target GPU counts before a training restart. Shrinking or growing must be approved by the administrator with an exact `source:target` annotation;
unknown counts cannot be approved automatically.

### 8.2 Administrator Manual Degraded Recovery When Spares Are Insufficient

| Sub-section | Walkthrough |
|---|---|
| `8.2.1 Scope and System Behaviour` | Automatic replacement requires one-to-one spares; when insufficient the original workflow stays failed and a human creates a new attempt. |
| `8.2.2 Step 1: Confirm the Automatic Flow Has Stopped` | Confirm REPLACE_NODE failed for lack of spares and the old attempt has stopped. |
| `8.2.3 Step 2: Keep the Faulty Nodes Isolated` | Faulty nodes stay cordoned/quarantined and are not reused because of the reduced recovery. |
| `8.2.4 Step 3: Re-check and Manually Reserve a Healthy Spare` | Verify HyperPod/K8s/Agent health and write the reservation before uncordoning. |
| `8.2.5 Step 4: Modify the Training YAML and Training Parameters` | Change node count, world size, parallelism, batch, LR, checkpoint and affinity. |
| `8.2.6 Step 5: Generate and Review the New Attempt` | New name, new attempt ID, reusing the job ID and the existing restart budget. |
| `8.2.7 Step 6: Verify the Reduced Attempt` | Verify scheduled nodes, GPU/EFA, checkpoint, NCCL, loss, Watcher observation. |
| `8.2.8 Failure Rollback and Spare Release` | When the new job fails, cordon the spare first, then safely clear the reservation and return it to the pool. |

### 8.3 Administrator Submits an Explicit Remediation After CHECK_MECHANICALS

`CHECK_MECHANICALS` only means investigation. `gpu-fault-admin submit-remediation` derives the annotation value, trusted marker and terminal trigger from the incident record:
`inspected` only confirms, `reset-gpu`/`reboot-node`/
`quarantine` compile a new fenced workflow without rewriting the original incident; idempotent, and any inconsistency fails closed.

<a id="84-手工删除无-ttl-的-incident-审计记录"></a>
### 8.4 Manually Deleting Incident Audit Records Ahead of Time

archive-first automatic retention is on by default, keeping 30 days; only explicitly setting
`spec.retention.controlRecordRetentionDays: 0` disables it. The archive bucket and the `s3:PutObject` permission on the target prefix
are prepared automatically by deploy. When early deletion is needed you must still preview, export NDJSON and SHA first,
then stop the writers and use the transactional purge; the script refuses to delete data with active workflows/commands, referenced by other incidents
or with inconsistent evidence.
Mail archives and Aurora snapshots need separate handling.

## 9. Data Migration

- **Purpose**: SQLite -> PostgreSQL or PostgreSQL -> PostgreSQL.
- **Regional stop-write order**: record replicas -> stop ingress -> wait for the processor/spool to drain ->
  stop consumers.
- **Recovery order**: first verify with 1 worker + 1 ingress, then consumers -> ingress restored to the original replicas.
- **Key requirement**: the target database is empty; the state file is kept until all original replicas are Ready.

## 10. Enabling Destructive Capabilities in Phases

### Phase A: Observation

Forensics, collection and simulation only; no workload or node is modified.

### Phase B: Isolation and Training Control

Opens isolation, stopping and restarting workloads, first proving that old and new attempts do not overlap.

### Phase C: GPU Reset

Enabled in the order Node Agent readiness -> drain -> quiesce -> no-client -> reset -> restore ->
validation -> scheduling/workload restore.

#### Phase C.0: XID 74 NVLink Workflow

Requires complete registers, Link ID, diagnostic command and SHA. Unknown/all-zero/secondary-only fail closed;
reset or support escalation only once the Catalog conditions are met.

#### Phase C.1: H200 SXID Recovery

Ordinary Fatal Trunk/full-reset uses the all-GPU/NVSwitch reset barrier; Always-Fatal enters
node reboot. The Kernel Collector is the production main path, the FM Collector a supplementary source.

#### Phase C.2: Explicitly Enabling Driver/Firmware Remediation

May be enabled only when the site has validated the target version, command and SHA. Any version, upload, verification or client
gate failure keeps isolation and escalates to support.

### Phase D: Node Lifecycle (Provider Replace Always Disabled)

The current production Profile OWNs both `nodeReboot` and `nodeReplace`, so the two capabilities must be accepted separately;
they are not two independent environment switches.

#### Phase D.1: HyperPod Reboot

Provider reboot is opened only with `NodeRecovery=None` and the profile OWNing it. Every call must have
the cluster second confirmation, incident/workflow fencing and an idempotency key; the old mutation master switch must not be set.

#### Phase D.2: warm-spare nodeReplace

`nodeReplace` always uses `HEALTHY_WARM_SPARE_ONLY`, activating only healthy spares of the same specification, and requires
full atomic allocation, post-action GPU/fabric validation and workload restart. Provider replace is always off;
with insufficient healthy spares it keeps isolation and alerts, never partially replacing or falling back to
`BatchReplaceClusterNodes`.

## 11. Warning on Using the Sample Manifests

Before applying any YAML scan for `REPLACE_WITH`, test clusters, fixed nodes, old wheels and the canary
Service. Samples are not a source of production parameters.

## Configuring EFA/RDMA Traffic and Hung Diagnostics

- Configures the EFA traffic baseline, SPIKE/DROP/ZERO/HUNG thresholds and the suppression budget.
- Hung diagnostics collect GPU processes, Python stacks, `/proc`, strace, RDMA counters,
  ethtool and log snapshots.
- A real hang versus checkpoint/graph compilation must be distinguished with training progress and per-rank liveness combined.

### Per-Rank Liveness Decision Before Escalation (Independent of Application Instrumentation)

The Host Collector judges whether a rank is advancing from each GPU process's write rate, CPU activity and GPU idleness.
Missing signals are not silent; suppression has a maximum budget to avoid permanently masking a hang.

## Configuring Host Resource Persistent Hazard Monitoring

- Monitors low CPU/GPU utilisation, low memory, high page cache, disk usage and EFA/RDMA state.
- Low utilisation is timed only when the control plane resolves, through attempt observations, exactly one running managed training attempt on the node; the ACTIVE state self-reported by the node collector is only a pre-filter; memory/disk anomalies do not require training to exist.
- The action is fixed as diagnostics, with no automatic restart; emails are deduplicated with a cooldown per node/job/signal.

### Downloading and Extracting Diagnostic Bundles on Windows

Explains downloading from S3, checking size/SHA, the two extractions of the outer tar.gz and the inner
`nvidia-bug-report.log.gz`, and how to distinguish an ordinary diagnostic bundle from a hung bundle.

## Configuring Heterogeneous Node GPU/EFA ACTIVE Inventory Checks

### 1. Understand the Configuration Scope

expected GPU/EFA are node-level values written into collector.env per instance type, not global control-plane values.

### 2. Check Node Labels Before Deployment

Check instance type, cluster label, `nvidia-smi`, EFA PCI/driver, port
ACTIVE/LinkUp.

### 3. Configuration Parameters

Defines instance type, expected GPU, expected ACTIVE EFA, consecutive anomaly count and sampling period.

### 4. Deploying for Heterogeneous HyperPod Nodes

Reads the instance type node by node and generates installation Jobs. Unknown instance types must provide GPU/EFA counts explicitly and cannot be skipped silently.
This section also explains the three-tier control-plane resources, connection budget, schema smoke and capacity probe.

### 5. Verify the Final Configuration of Every Node

Check collector.env, Host Collector logs, expected/active/missing/excess and mismatch.

### 6. Control-Plane Behaviour After an Anomaly

Chooses plugin restart, driver remediation, reboot or replace according to missing GPU, missing EFA PCI, unbound driver, inactive port, missing allocatable;
re-verifies the inventory after the action.

## Configuration Management and Pre-Go-Live Validation

- Non-sensitive parameters are loaded via `envFrom` from the core/postgres/processor/telemetry/notification/recovery ConfigMaps.
- Secrets, tokens, passwords and release pins keep using `valueFrom`.
- `generate-env-reference.py` maintains the environment variable reference.
- `gpu_fault.config_cli validate` checks sensitive configuration, exit budgets, role boundaries, pins and the database connection budget.

### Production Manifest Modification and Auto-Generation

- This section is only for solution developers; on the normal path administrators/customers maintain only `site.yaml`, Secrets and AWS inputs,
  deploy via `gpu-fault-admin`, and neither write nor apply solution Manifests one by one.
- GPU components modify the source YAML in `deploy/dataplane/`; CPU components modify the base, regional patch
  or renderer; editing generated directly is forbidden.
- Running resources declare the stop phase, order within the phase and cleanup action via `gpu-fault.io/cleanup-*` annotations;
  GPU Deployments managed by the regional orchestrator additionally declare `deploy-mode`.
- `make deployment-contracts-update` automatically re-renders the CPU manifests, generates the build-time candidate inventory from the source Manifests,
  validates the configuration and runs the contract tests.
- CI runs the read-only `deployment-contracts-check`; any inconsistency among source annotations, generated output and the registry
  blocks the merge.
- First deployment and upgrades automatically create/refresh the CPU and GPU
  `gpu-fault-installed-resources` ConfigMaps from live objects; cleanup reads the runtime registries, and developers do not maintain
  resource names.

#### Administrator/Customer Boundary: Do Not Write Solution Manifests

Administrators use
`gpu-fault-admin deploy/status/join-cluster/remove-cluster/uninstall`;
only resume and rollback still use the low-level checked commands. Editing generated, the cleanup inventory or
online Deployments/RBAC directly is forbidden. The customer's training job YAML is a business workload and the only explicit exception.

#### Example: Adding a GPU PCIe Collector

Declare the producer phase and order within the phase on the new GPU Deployment; declare `regional_rollout` additionally when the regional orchestrator must apply it.
SA/RBAC are derived by the generator as support/delete.

#### Example: Modifying a CPU Control-Plane Role

Modify the base/patch/renderer; the renderer automatically generates the correct phase for ingress, control-worker and spool-worker;
generated files must not be hand-edited.

#### Complete Developer Change Steps

Complete in order ownership/lifecycle design, source code and source Manifests, runtime constraints and annotations, auto-generation,
review of generated results, contracts and the full gate, staging release/rollback, and commit everything in the same PR.

#### One Command Completes the Update

`make deployment-contracts-update` writes the generated artefacts,
`make deployment-contracts-check` validates read-only and is run by CI.

## Hard Constraints on Client Identity and Migration Switches

- Uvicorn must use `--no-proxy-headers`; internal replay trusts only the direct peer.
- Unsigned NodeAction result queries may be enabled only in a short migration window carrying the release ID and expiry.
- Collector gauges are exported only from ingress; the scrape configuration must include ingress.
- Internal detail metrics need the execution token and do not enter the AMP default scrape.
- A non-regional active executor must explicitly confirm `GPU_FAULT_ALLOW_SINGLE_CLUSTER=true`.

## IV. Recommended Daily Usage

1. **First regional deployment**: follow §0.1 strictly; do not run only bootstrap.
2. **Daily releases**: first `plan/status`, confirm remote open=0, then `upgrade`.
3. **Adding a GPU cluster**: use `gpu-fault-admin join-cluster --state-dir ...
   --gpu-cluster-arn ...`; do not rebuild the CPU control plane.
4. **Node addition/replacement**: installed automatically by the Reconciler according to Node UID and the artifact/config pins.
5. **Troubleshooting**: check the wheel/module digest first, then authentication, queue, Executor, Agent.
6. **Destructive actions**: open in phases A -> D, each phase needing a maintenance window and rollback evidence.
7. **Training jobs**: customer images are custom; the sample PyTorch Training DLC is only for the NCCL/DDP smoke.
8. **Acceptance records**: a production declaration must keep version, scale, time window, command output and cleanup results together.

## Appendix A. Legacy HyperPod Single Cluster (Non-Production)

This appendix is only for historical environment maintenance, a migration starting point and limited Canaries; it is not a production deployment entry point.

### A.1 Setting Parameters

- **Purpose**: configure the capacity, Aurora, processor, alerting, recovery and collection parameters of the single-cluster automated deployment.
- **Focus**: processor workers, Store I/O, PostgreSQL pool and exit budgets must be accounted for together.
- **Database upgrade procedure for existing deployments**: schema ensure -> backfill -> dual ->
  status check -> dedicated -> purge; phases cannot be skipped.

### A.2 Choosing the DCGM Exporter Mode

- `auto`: probes existing endpoints, deploying the managed exporter when none exists.
- `existing`: only reuses the customer exporter, failing on missing fields.
- `managed`: this solution deploys the exporter, failing on a port conflict.
- `disabled`: allowed only when the nvidia-smi fallback is explicitly enabled.
- **Pass criterion**: all nodes adopt a consistent mode, with the required DCGM fields complete.

### A.3 Running the Deployment

`deploy/hyperpod/deploy.sh deploy` is responsible for kubeconfig, Aurora, wheel/bundle,
the three-tier control plane, Runtime Profile, Watcher, Exporter, node installation and the non-destructive E2E.

### A.4 Observing the Deployment

Look at Pods, Deployments, DaemonSets, PDBs, logs and `/healthz`. The single-cluster path keeps
the `gpu-fault-api-canary` compatibility Service; the regional path uses `gpu-fault-api`.

### A.5 Runtime Profile Selection

- **Role**: declares for each capability whether this system OWNs it, DELEGATEs it, OBSERVEs it or disables it.
- **Key requirement**: events, Collectors, Agents, Watcher and training jobs must reference the same registered
  profile version.
- **Common fault**: referencing an unregistered profile from a sample file leaves the Watcher at 404 forever.

### A.6 Transitional Single-Cluster Acceptance Checklist (Non-Production)

This section is only for old single-cluster environments and pre-migration Canaries; production regional mode uses REG-8. Checks API, fleet, collectors, DCGM,
PDB, RBAC, idempotency, lease fencing and the non-destructive E2E.

### A.7 Transitional Single-Cluster Troubleshooting (Non-Production)

#### API Cannot Start

Check the Store URL, execution token, allowlist, Secrets and Pod logs.

#### Agent Readiness Fails

Compare Agent version, wheel SHA, policy, profile, config digest, Node UID, boot ID,
generation and heartbeat age.

#### No DCGM Data

Check the 9400 endpoint, required counters and collector logs; the nvidia-smi fallback
cannot replace complete DCGM.

#### Workflow Stays WAITING

Check the waiting step, old Pods, HyperPod identity, Agent compatibility and remote commands.

#### Workflow BLOCKED

This is a safe terminal state or a safe wait. Complete evidence/owner/freshness per `blocked_reasons`; do not widen the allowlist.

#### Administrators Must Not Call HyperPod Provider Replace

- `BatchReplaceClusterNodes`, `BatchDeleteClusterNodes` or an equivalent provider replace
  is forbidden for automation and administrators alike.
- `REPLACE_NODE` means only `HEALTHY_WARM_SPARE_ONLY`. With a qualified spare continue the checked
  warm-spare workflow; when insufficient perform the manual degraded new attempt of §8.2 or support escalation.
- The faulty node stays isolated. Only after a node disappears through independent repair or permanent decommissioning are stale
  fleet records cleaned via drain -> revoke; before re-admitting it as a spare, hardware acceptance, Agent convergence, the spare label,
  the `AVAILABLE` state and the cordon must be completed.
- The generic `MARK_UNSCHEDULABLE` completion notification is still a gap, but any later notification must explicitly say
  "warm-spare only, provider replace forbidden".

The sub-sections kept in the main manual are for executing the alternative path and cleanup, not a provider replace SOP:

| Sub-section | Current meaning |
|---|---|
| `When an Alternative Remediation May Start: Preconditions` | First prove that isolation and stopping training are complete |
| `No Need to Avoid the Warm Spare Allocation Window` | The faulty node and the spare reservation are different objects |
| `Permitted Remediation Steps` | Only warm-spare or the §8.2 manual degradation is allowed |
| `Cleaning Up Fleet Records After a Node Disappears Through External Repair or Permanent Decommissioning` | drain -> revoke stale Agents |
| `Adding an Independently Repaired and Re-accepted Node to the Spare Pool` | Complete the health, Agent, label, AVAILABLE and cordon gates |
| Manual cleanup of isolation residue (only when the workflow never reached `RESTORE_SCHEDULING`) | Clean only this solution's taints/annotations |
| `Known Gap: No Notification When Isolation Completes` | Do not rely on emails to judge whether remediation may proceed |
