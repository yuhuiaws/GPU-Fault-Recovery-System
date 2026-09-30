English edition of `docs/管理员快速部署.md`; the Chinese file remains the source of record until both are maintained together.

# GPU Fault Handling System Administrator Quick Deploy

This document describes only the successful path of the first site build or application-layer rebuild of the regional production solution. The administrator supplies existing CPU/GPU EKS or HyperPod ARNs, and a single command automatically creates the site AWS resources, builds the signed release and deploys the application. This path does not require administrators to hand-stitch commands following CPU-1 to 9 or REG-1 to 8, nor to write or apply solution Manifests one by one. For object-by-object audit, special handling of shared infrastructure or break-glass, go to the [Deployment and Operations Manual](deployment-and-operations-manual.md); post-go-live upgrades, cluster changes, rotation, troubleshooting and decommissioning uniformly use [Administrator Operations](administrator-operations.md).

## 1. Scope

The only production form is:

```text
Regional CPU EKS control plane + Aurora PostgreSQL + TLS NLB
                    |
                    +-- HyperPod EKS GPU cluster A
                    +-- HyperPod EKS GPU cluster B
                    +-- ...
```

Managed GPU HyperPod clusters must be EKS-orchestrated with `NodeRecovery=None`. Generic Kubernetes,
HyperPod Slurm and the single-cluster all-in-one are not part of the new production deployment path.

## 2. Current Administrator Entry Point

The first four parameters are stored in the private `initial-deploy-request.json` before source preparation; afterwards only `--state-dir` is passed, and even when `site.yaml` has not been generated the ARN order and email are restored from that record.
Conflicting input is refused; when old state lacks that record the original parameters must be supplied, without hand-fabricating or guessing the identity.

```bash
gpu-fault-admin deploy \
  --cpu-cluster-arn <cpu-eks-or-hyperpod-arn> \
  --gpu-cluster-arn <gpu-eks-or-hyperpod-arn> \
  --state-dir /secure/gpu-fault \
  --admin-email <operations-email>
gpu-fault-admin deploy --state-dir /secure/gpu-fault   # resume / upgrade
```
For several GPU clusters repeat `--gpu-cluster-arn`; after the first GPU cluster completes the baseline deployment and verification, the remaining clusters automatically enter
a separate batch join, rolling at the site cluster parallelism `spec.release.upgradeMaxParallelClusters` (default 1, cap 8); a single failure does not undo the successful clusters of the same batch.
ARNs passed again on an existing site must be the managed set or a superset of it: extra GPU ARNs are joined automatically after the release (join only, no rollout, when the release is unchanged); giving fewer clusters or changing the CPU is
refused with a hint to use `remove-cluster`. The same command with `--rollback` goes back one step, with `--approve-profile-plan <sha>
--reference CHG-<id>` approves the Profile plan and continues. Administrators supply no release-ref, artifact, site,
release-build/release-deploy, bundle or venv path. The command automatically creates or reuses the signing material and
site-tagged ECR, runs the required gates, pushes and verifies the release, at the same time creates or reuses
Aurora, AMP/SNS, PKI/IAM and other resources per the dependency graph, and imports `deploy/observability/dashboards/*.json` into Amazon Managed Grafana in the Region of the CPU cluster: taking in turn `--grafana-workspace-id`, the workspace with the site tag, the only ACTIVE workspace in that Region, and creating a new one for the customer when there is none (IAM Identity Center login; uninstall deletes it); a refused creation or failed import only warns and does not block. After the import, deploy finds the corresponding user in IAM Identity Center by the administrator email (`--admin-email`) and grants Grafana ADMIN (the deployment host needs `sso:ListInstances`, `identitystore:GetUserId`, `grafana:ListPermissions`, `grafana:UpdatePermissions`); when that user cannot be found it does not block, and at the end of the deployment prints the dashboard address, the one `aws grafana update-permissions` command to authorise your own login, and what is missing; after adding the user simply rerun deploy; you can also pass `--grafana-viewer <Identity Center user id>` directly to let deploy grant VIEWER on your behalf.
Steps needing candidate artefacts still have to wait for the build and signature verification to complete. When
`spec.health.identityCenterRegion` is not pre-set, the deployment host also needs `ec2:DescribeRegions` permission to scan and discover
the enabled Region where Identity Center lives; with the field pre-set no other Regions are scanned.

The VPC and base routing of EKS/HyperPod themselves are part of the already built clusters. The GPU VPC must already have a usable NAT egress; when the CPU VPC lacks public subnets for the NLB, the script creates two solution-specific `/28` subnets from free CIDR.

Before deploying, install and activate `gpu-fault-admin` per the offline bundle rules of the [CI Release Process](ci-release-process.md); a development checkout runs
`make deploy-host-setup-online` to prepare the CLI and the isolated tool environment in parallel, and deploy continues only after both succeed.
The first command determines the source root from the current trusted checkout; later runs restore it from internal state; the generated site and repository paths do not enter the public parameters.

The release provides `config/admin-config.example.yaml`; in the deploy-host venv the same template is at `share/gpu-fault/admin-config.example.yaml`. A first site build can apply the administrator-edited configuration directly with
`deploy --config <0600-admin-config.yaml>`; when not passed, the release defaults are persisted and `<state-dir>/admin-config.yaml` generated.
On an existing site edit that file and run
`gpu-fault-admin config --state-dir ... --reference ...`; the complete commands are in
[Administrator Capacity Configuration](administrator-capacity-configuration.md).

| Internal phase | Meaning |
|---|---|
| source preparation | Pins the Git identity, prepares or reuses the signed release and the deployment environment |
| preflight | Read-only checks of AWS identity, clusters, IAM, capacity, Aurora, LBC, ACM and monitoring |
| bootstrap/upgrade | Without a site, creates in parallel per the dependency graph (monitoring, Node keys and Aurora instance readiness do not wait on each other); with an existing site, verifies cluster identity first, then upgrades |
| verify/stability | Verifies CPU/GPU, Profile, TLS, Agents, Aurora and the stability window |

## 3. Pre-Deployment Inputs

**One email confirmation is a prerequisite: the site has exactly one email channel.** The site SNS topic carries both the AMP Alertmanager metric alerts and the fault-handling notifications sent by the control plane itself (incident escalations, GPU count change approvals, mechanical inspection requests, including retries and receipts; plain text, subject <= 100 ASCII characters, sender `no-reply@sns.amazonaws.com`).
`--admin-email` is the SNS subscription address; deploy sends and checks this confirmation within its first minute, exiting with code 2 and listing the resume command when unconfirmed, or add `--wait-for-email-confirmation <minutes>`. No SES identity needs prior verification; sites needing rich-text email can declare `spec.notifications.channel: ses` in `site.yaml` (which then additionally requires a verified SES sending identity, see the operations manual §8).

**Grafana dashboard prerequisites (without them the deployment still succeeds; just nobody can open the dashboards).** After importing the dashboards into Amazon Managed Grafana, deploy grants workspace ADMIN to "the IAM Identity Center user whose email equals `--admin-email`"; Managed Grafana accepts only Identity Center/SAML login, so first satisfy: (1) the account has IAM Identity Center enabled (deploy first looks for an instance in the CPU cluster Region, and if none is found automatically scans the account's other enabled Regions and records the found home Region in `spec.health.identityCenterRegion` of site.yaml; that key can also be pre-set to skip the scan);
(2) Identity Center has a user whose email equals the administrator email and who can already log in (has set a password or completed email verification; this step can only be done in the Identity Center console); (3) the deployment host identity has `grafana:*`, `sso:ListInstances`, `identitystore:GetUserId`, and, when `spec.health.identityCenterRegion` is not pre-set, the `ec2:DescribeRegions` needed to scan enabled Regions.

The first ARN command generates in the state an internal `RegionalSite` with mode `0600` as the source of truth for application topology, identity, artefacts and health checks.
Administrators do not create, edit or pass in that file by hand. Its internal content includes:

| Configuration | Requirement |
|---|---|
| `spec.awsRegion` | The Region explicitly chosen by the operator |
| `spec.cpu` | kubeconfig, EKS ARN, CPU HyperPod name |
| `spec.release` | `dist/current-release.json` and the Agent config digest |
| `spec.runtimeProfile` | Editable template, immutable source snapshot, automatic version and stable registration anchor |
| `spec.nlb` | NLB name, public subnets, Security Group, ACM certificate ARN |
| `spec.health` | Aurora, AMP, SNS and health thresholds |
| `spec.notifications` | Notification channel (`channel: sns` default / `ses`), administrator email, Subject prefix and external alert acknowledgement; `ses` additionally has sender, recipient list and the optional `sesConfigurationSet` |
| `spec.clusters[]` | Per-cluster unique ID, context, EKS ARN, HyperPod name, IRSA, namespace allowlist, GPU node subnets `agentEndpointAllowedCidrs` |
| Administrator configuration | The 22 normalised non-sensitive configuration items and role digests in `state-dir/admin-config/desired.json` |
| Credential references | Per-cluster token files, CA file and the controlled fleet master file |

The CLI refuses layer by layer unknown fields, wrong types, duplicate clusters, cross-Region ARNs, non-HTTPS URLs and leftover
`REPLACE_*`, and generates the low-level release configuration in a private directory. Administrators do not maintain these files.

The current version still references the token, CA and fleet master through controlled files. The files must be outside the repository, with directory mode
`0700` and sensitive file mode `0600`, and must not enter Git, ordinary logs or build artefacts.

## 4. First-Deployment Gates

1. An existing 3-node CPU EKS/HyperPod cluster.
2. At least one existing GPU EKS/HyperPod cluster with `NodeRecovery=None`.
3. The GPU VPC already has a usable NAT egress.
4. The executing identity has ECR, EC2, RDS, AMP, SNS, EKS, Kubernetes and IAM permissions.
5. The deployment host has the Python 3.12 project environment installed plus `aws`, `cosign`, Docker Buildx, `kubectl` (>=1.28: the release engine's wait loops use `kubectl wait --for=jsonpath=...` (including value-less field-existence checks) and `kubectl rollout status` instead of fixed-interval sleeps), `helm`, `curl`, `jq`, `openssl`, `sha256sum` and `make`; the ARN entry point verifies these commands before any AWS mutation but does not install system tools automatically.

`spec.clusters` of the first application bootstrap must be non-empty; only a completed site can enter an explicit `[]` via the formal
`remove-cluster`, and an empty list on a new site fails before the first Kubernetes apply.

A dirty working tree and a clean personal commit do not query GitHub; only a clean `HEAD == origin/main` without a local
signed release attempts to consume the main CI candidate of the same commit. Credential boundaries are in the [CI Release Process](ci-release-process.md).

The ARN bootstrap creates or reuses, exclusive to the current `site-id`:

- Aurora Serverless v2 writer/reader, subnet group and SG;
- The immutable runtime ECR and the mutable BuildKit cache ECR repository; the cache repository keeps the current
  `buildcache-linux-amd64`, and untagged historical cache artefacts expire after 7 days;
- The NLB SG and the GPU egress CIDR whitelist;
- The AMP workspace, Alertmanager, SNS topic and email subscription;

EKS/HyperPod are explicit ARN inputs; VPC/subnets/node CIDRs are discovered from the clusters. The command automatically creates the separate token,
CA, fleet master and kubeconfig files. When the GPU VPC has no NAT egress the deployment stops without modifying existing routes.
When an existing cache repository lacks that lifecycle policy the ARN entry point completes it; when an existing policy differs in content the deployment fails closed
and does not silently widen or rewrite the cleanup scope. The executing identity needs read and write permission on ECR lifecycle policies.
The ARN entry point is the only writer of site AWS infrastructure in this solution. The source scan, preflight and cluster checks are all run internally by
the unified command, and no AWS or Kubernetes mutation starts when any required check fails.

## 6. Running the Deployment

The first run creates resources and builds the release; repeated runs skip completed resources according to the bootstrap state and site tags,
reuse the registry digest when the runtime image inputs are unchanged, and the same release is judged `NOOP` by the rollout.
Administrators must not rewrite the release ID or component digests.
When the component `release-diff` is explicitly classified `NOOP`, even if the release ID changed only because of delivery identity,
this entry point skips the preflight and the NOOP verifier inside deploy, only registers the new release state, runs one
full `verify` in parallel and saves the report, without waiting for the stability window; an administrator running `status` still re-queries live state.
deploy-host-only only updates the deployment host and runs the read-only preflight; test/documentation changes run only the affected gates; neither
creates an application release, touches the GPU rollout or waits for the stability window.

When the Runtime Profile policy changes, this deploy writes out
`<state-dir>/release-deploy/profile-plan.json` before applying the release and stops; the stop message directly prints the fields of the table below and the resume command with the real
`plan_sha256`, with no `jq` needed. Complete operation, evidence and exception branches are in
[Runtime Profile Change Approval](administrator-profile-change-approval.md).

| Field | What the administrator checks |
|---|---|
| `site_identity.site_name` | Is it the target site |
| `site_identity.aws_region` | Is it the approved Region |
| `site_identity.cpu_eks_arn` | Is it the site's stable CPU control plane |
| `site_identity_sha256` | Machine check digest of the readable identity above |
| `current_version` / `desired_version` | Does the Profile version migration match the change ticket |
| `change_kind` / `changes[]` | capability, mode, owner, adapter changes and risks |
| `live_profile_sha256` | The online Profile baseline at approval time |
| `policy_digest` / `source_sha256` / `snapshot_sha256` | Target policy, template and candidate snapshot evidence |
| `plan_sha256` | The approval binding value of the whole reviewed plan; must be written into the change ticket and passed back to the approval command |

Write the printed `plan_sha256` into the change ticket; after obtaining the change ticket or maintenance window approval, run the resume command at the end of the stop message
as is, replacing only `CHG-<id>` with the real change ticket number:

```bash
gpu-fault-admin deploy \
  --state-dir /secure/gpu-fault \
  --approve-profile-plan <plan_sha256> \
  --reference CHG-12345
```

This command approves first and then continues the same deployment. `--approve-profile-plan` accepts only the digest of the current pending plan:
without `profile-plan.json` or with a mismatched digest it refuses and gives the current pending digest, and it cannot be carried on the first
deploy. The approval is valid only for that plan: a change of template, target policy or live baseline archives and voids the old approval
and stops again; when the release failed and the plan is unchanged, rerunning `deploy --state-dir` resumes; after the complete verify,
stability and commit succeed the active approval is consumed once. Do not edit
`profile-plan.json`, `profile-approval.json` or the approval archive.

`deploy` runs the live preflight again, avoiding drift of the environment after the check, then:
1. Runs the first bootstrap when there is no release state.
2. With existing release state, compares the wheels/bundle, rendered Manifests,
   renderer/dependency lock, image digests, schema, protocols, Profile, endpoint and cluster set in Manifest v3.
   The four change kinds are summaries only; the component DAG chooses the actual path; the CPU maintains separate ingress, worker and
   spool Manifest digests, and a single-role change waits only for that role. finalize rolls the CPU only when pins actually change.
   The Profile is compared by the canonical policy digest; a path-only or YAML-order-only change does not trigger `FULL`.
3. A failed bootstrap state re-runs the idempotent bootstrap.
4. Uploads the three content-addressed wheels and the Node bundle as needed; when the GPU or Node Runtime is involved, completes `candidate-preflight-ready` with a server-side dry-run and per-node read-only host probes, joining before schema/registry/CPU modifications. The read-only preflight runs at most 4 clusters in parallel with at most 8 nodes per cluster, omitting no target node.
5. On a schema change runs the three Jobs index-build, schema ensure (advancing to the target schema version) and schema preflight in order; stops if the preflight fails. A schema version change can only be fail-forward: add `--accept-schema-change` on the same deploy, the engine first snapshots Aurora (`gpu-fault-pre-v<target version>-<release>`), and `site.yaml` is untouched.
6. Deploys the three-tier CPU control plane (candidate compatibility window), checks the Profile and waits for Agent heartbeats; observability and data-plane waves run in parallel. Before a node wave it re-checks Nodes, Executor, commands, workflows and leases.
7. Creates the NLB Service after verifying the private hosted zone exists, the ACM certificate is `ISSUED` and its SAN covers the control-plane domain. The first deployment requests creation before schema/CPU, overlapping the CPU work, without publishing the CNAME at that point.
8. Strictly waits for the NLB `state=active`, the TLS 443 listener using the target certificate, the raw NLB DNS
   resolving, and all three API Pod targets `healthy`.
9. Only after the gates above pass does it `UPSERT` the private CNAME and wait for the Route53 change to become explicitly
   `INSYNC`.
10. The first deployment writes only the first GPU cluster into the CPU registry as the site baseline and converges it; the other targets get no pre-created
   token, IAM, Private Hosted Zone VPC association or data-plane resources.
11. The other GPU clusters are then joined as separate join transactions: only discovery and per-cluster prerequisite work run at most 4-way in parallel, and joins and upgrades are both bound by the site-wide 64-node Installer budget;
    rollout parallelism uses the same knob as the shared release upgrade: serial per cluster by default, and only `spec.release.upgradeMaxParallelClusters`
    (default 1, cap 8) allows parallelism, with the first cluster completing alone first as the canary. Any cluster failure immediately blocks clusters not yet started and records `failed_cluster_ids` and `not_started_cluster_ids`;
    a single cluster's Executor not Ready, an Agent safety gate or convergence timeout, or an Installer failure enters resumable `PAUSED`; only a global fault rolls back automatically, always restoring serially in reverse order.
12. Before finalizing a Profile change, old-Profile workflows/workloads must be at zero.
13. bootstrap/quick evidence is reused only when bound to the current release state and within ten minutes (including quick evidence of a deploy completed but not committed), and reruns otherwise. verify runs in parallel with the stability window; only `GpuFaultStoreIoRejected` may wait up to 420 seconds, after which the 120 to 300 second stability window is still
    observed (a `CONTROL_PLANE_ONLY` release rolling only the control plane ends as early as 60 seconds after two clean samples and all control-plane Deployments converged); queue trends are judged by relative growth, and Collector-type criticals within 600 seconds after the data plane just converged do not count.

`plan/status` shows the next deployment classification: `NOOP` does only one parallel read-only verification; `CONTROL_PLANE_ONLY`
rolls only the CPU; `DATA_PLANE_COMPATIBLE` rolls only the changed data plane, serial across GPU clusters by default; `FULL` keeps the two-phase
pins. The `plan` output directly writes out the cluster parallelism in effect for this site; batch join uses the same configuration, 1-way by default.
When an upgrade fails, rerunning the same `deploy` reuses the persisted diff plan: the same
release still in `failed` resumes from its checkpoint; a `rolled-back` state that has completed and verified automatic rollback opens a new transaction from the current online baseline,
so a new release produced after fixing the code does not wrongly continue the old release ID.

Failure of any phase stops the subsequent phases. In particular, no new CNAME is published while the NLB targets are unhealthy;
the Executor of a first deployment or a re-join flow is not started while Route53 is not `INSYNC` or the GPU Pod's DNS/TLS verification fails.
Rerunning the same `deploy` command continues from the idempotent state.

Administrators must not run CPU-4, REG-1 to 8 or online
`kubectl edit/set env/apply` while the command runs.

## 7. Post-Deployment Verification

Before the unified command returns successfully it has completed the full verify and the stability window, including:

- The three CPU tiers, PDBs, ADOT and the Aurora refresh CronJob;
- The Runtime Profile registered with no content drift;
- GPU Executor IRSA, Secrets, wheel and adapter switches;
- Real Executor readiness, the TLS private CA;
- Agents ACTIVE, Fleet readiness and Collector freshness;
- Remote command backlog age and internal errors;
- The NLB TLS listener, certificate and three healthy targets;
- Aurora writer/reader;
- The AMP rule namespace, Alertmanager and the confirmed SNS subscription.

Then perform production acceptance per the [Regional deployment acceptance checklist](deployment-and-operations-manual.md#reg-8-regional-deployment-acceptance-checklist), verifying separately with
different job/attempt IDs:

```bash
SITE_FILE=/secure/gpu-fault/site.yaml

gpu-training-submit customer-job.yaml \
  --site "${SITE_FILE}" \
  --job-id deployment-smoke-direct \
  --attempt-number 1

gpu-fault-workload-annotate customer-job.yaml \
  --site "${SITE_FILE}" \
  --job-id deployment-smoke-reviewed \
  --attempt-number 1 \
  --output /tmp/customer-job.managed.yaml

kubectl apply --dry-run=server -f /tmp/customer-job.managed.yaml
kubectl apply -f /tmp/customer-job.managed.yaml
```

Only after both training runs are correctly observed by the Watcher may destructive capabilities be opened in phases A to D.

### 7.1 Opening the Grafana Dashboards

At the end deploy prints the workspace address and `Grafana ADMIN granted to <email> (Identity Center user <id>)`: log in with that Identity Center user; on the left, Dashboards -> the `GPU Fault Recovery` folder holds the 8 dashboards (overview, recovery-outcome, orchestration-invariants, remote-command, collector-health, telemetry-pipeline, control-plane-capacity, policy-coverage), with the data source `GPU Fault AMP (<region>)`; when the rules change the next deploy re-imports automatically.
To grant colleagues permission choose any one of: `deploy --state-dir X --grafana-viewer <Identity Center user ID>` (grants VIEWER; the ID is a UUID, found under "User ID" on the user detail page of the Identity Center console or via `aws identitystore list-users --identity-store-id <d-xxxx>`), the Grafana UI Administration -> Users, or running the `aws grafana update-permissions …` command deploy printed.
When the automatic grant did not succeed deploy prints `Grafana ADMIN was not granted automatically: <reason>`: no user with that email -> add it in Identity Center, complete the first login and rerun deploy (the read-only probe completes the grant without rerunning the installation); `no IAM Identity Center instance is visible from <region> or the N other enabled regions scanned` -> the account has not enabled Identity Center yet; enable it and rerun (if you know the home Region and the scan cannot reach it, write `spec.health.identityCenterRegion` and rerun); `AccessDenied` -> add the deployment host permissions. Details are in the Grafana section of the Deployment and Operations Manual.

## 8. Failure Handling

| Scenario | Handling |
|---|---|
| Email unconfirmed (exit code 2, within the first minute) | Click the link in the SNS confirmation email (sites with `channel: ses` also click the SES verification email), and rerun the printed command as is; or add `--wait-for-email-confirmation <minutes>` |
| Internal preflight fails | Fix the source, credentials or shared infrastructure per the check name without modifying the clusters; retirement of old HMA producers must be approved separately per the operations manual §7.4 |
| ARN bootstrap fails | Keep the private state directory and rerun `deploy --state-dir`; completed resources are skipped |
| release build fails | Only the created ECR is kept; fix the source/gates/credentials and rerun the same command |
| First application deployment fails | Look at the release state; after the automatic fail-closed fix the cause, then rerun `gpu-fault-admin deploy` |
| verify/stability fails | Do not enter production traffic or the destructive capability phases; fix and rerun `deploy --state-dir` |
| Prompted to add `--supersede-failed-transaction`/`--accept-schema-change` | Refused as soon as the build finishes; after confirming, add that parameter to the same deploy and rerun (the parameter takes precedence over leftover environment variables) |
| Profile plan awaiting approval | Review the printed plan fields and run the `deploy --approve-profile-plan` resume command given in the stop message |
| Profile plan or live baseline drift | The old approval has been archived and voided; review the new plan and approve again |
| Profile content differs within the same version | Must not overwrite; release a new profile version |
| Grafana import failed (warning only) or cannot be opened | No Identity Center instance / API refused: create an Identity Center-authenticated workspace in the console and rerun with `--grafana-workspace-id`, or paste `deploy/observability/dashboards/*.json` in the Grafana UI via Dashboards -> New -> Import; for permission problems handle by the reasons of §7.1 |

Do not make the deployment green by lowering readiness, widening the allowlist, disabling signing or skipping TLS. Post-go-live handling such as interrupted upgrades, unconverged Agents and Executor verifier failures is in the regional upgrade and troubleshooting-by-symptom chapters of [Administrator Operations](administrator-operations.md).

## 9. Go-Live Evidence

Every production deployment keeps at least:

- The release manifest, the SHAs and `module_digest` of the three component wheels/Node bundle, and the control-plane/executor isolated venvs inside the runtime image with their component digests;
- The site YAML digest and the generated configuration digest, without Secret content;
- The `deploy` and `status --full` output (including the preflight/verify reports inside deploy);
- The CPU/GPU verifier results;
- The Runtime Profile version and effective capabilities;
- The Profile plan digest, approval reference and the `CONSUMED`/`SUPERSEDED` audit, without copying Secret content;
- NLB/TLS, Aurora refresh, AMP/SNS, CAP-005 zero skips and the installation resource registry digest;
- The job/attempt IDs of the training smoke;
- The change ticket number, maintenance window and operator.

Evidence capture commands and redaction requirements are in the [Deployment and Operations Manual](deployment-and-operations-manual.md).

## 10. Follow-Up Operations Entry Points

Later operations uniformly enter [Administrator Operations](administrator-operations.md):

| Follow-up task | Operations chapter |
|---|---|
| Health checks and repeated acceptance | Common task entry points |
| Regional upgrade, resume and rollback | Regional upgrade |
| Joining or removing GPU clusters | `gpu-fault-admin join-cluster/remove-cluster` |
| Token, certificate and node key rotation | Credential and certificate lifecycle |
| Agent, Executor, queue or provider faults | Troubleshooting by symptom |
| Data migration, archival and purge | Data and audit maintenance |
| Reinstall keeping the clusters, complete uninstall or permanent decommissioning | Complete uninstall or permanent decommissioning |
| Current capability boundaries and commands pending productisation | Commands currently pending productisation |
This file does not repeat the uninstall state machine, legacy environment discovery, cluster removal or the roadmap; no solution uninstall command may delete a GPU EKS or GPU HyperPod cluster.
