English edition of `docs/管理员Profile变更审批.md`; the Chinese file remains the source of record until both are maintained together.

# Runtime Profile Change Approval

This document is for regional site administrators and explains, after a deploy stops because of a Runtime Profile policy change, how to review the plan,
bind an external change ticket, let the same deploy continue and keep the audit evidence. The whole procedure has only two steps and needs no `jq`,
internal site path, `PROFILE_APPROVAL` environment variable or hidden parameter.

The Profile is an authorisation policy. Once a new version of the site profile (`spec.runtimeProfile`) takes effect it
is used by every GPU cluster that **follows the site profile**; a cluster may also own its profile
(`spec.clusters[].runtimeProfile`, filled in by `join-cluster` from the provider's example profile when the site
profile's `nodeReboot` owner cannot serve the cluster's provider: HyperPod uses `gpu-fault-hyperpod-adapter`/`hyperpod-v1`,
plain EKS `eks-managed` uses `gpu-fault-eks-adapter`/`eks-v1`). Every version in use is registered once in the control
plane; the cluster entry's `agentConfigDigest` is the Agent configuration digest computed for its version; the cluster
registration carries that version and digest and the fleet pin is checked per cluster. GPU cluster joins and
deregistrations continue to be handled by `join-cluster/remove-cluster` and do not enter the stable site identity digest
of the site profile approval, but **a cluster's own new profile version is reviewed the same way**: the join writes the
same `release-deploy/profile-plan.json` (`change_kind: CLUSTER_PROFILE_NEW`, `registration_cluster_id` naming the
cluster), continues after `join-cluster --approve-profile-plan <plan_sha256> --reference <CHG>` (or the same flags on
`deploy`), and once the approval is consumed the immutable snapshot `profiles/<version>.yaml` is written and reused by
later clusters of the same provider.
When several clusters need the same version, one plan covers them all and one approval approves it (see 1.2).

## 1. Shortest Operating Path

Step one: run the four-parameter deploy as usual.

```bash
gpu-fault-admin deploy \
  --cpu-cluster-arn <cpu-arn> \
  --gpu-cluster-arn <gpu-arn> \
  --state-dir /secure/gpu-fault \
  --admin-email <operations-email>
```

When the Profile changes, the command writes out `/secure/gpu-fault/release-deploy/profile-plan.json` and stops,
and the stop message directly prints the review fields and the resume command:

```text
Runtime Profile policy changed; the deploy stopped for review.
Review /secure/gpu-fault/release-deploy/profile-plan.json:
  site_identity:
    site_name: prod
    aws_region: us-east-1
    cpu_eks_arn: arn:aws:eks:us-east-1:123456789012:cluster/control
  version: regional-hyperpod-3f2a1c9b7d40 -> regional-hyperpod-8e51b0c2a7f3
  change_kind: EXPANSIVE
  changes:
    - gpuReset: mode OBSERVE->OWN
  live_profile_sha256: ...
  policy_digest: ...
  snapshot_sha256: ...
  plan_sha256: <plan_sha256>
Record plan_sha256 on the change request; once it is approved, rerun:
  gpu-fault-admin deploy --state-dir /secure/gpu-fault --approve-profile-plan <plan_sha256> --reference CHG-<id>
```

Step two: review these fields per section 4, write `plan_sha256` into the change ticket, and after obtaining approval run the printed
resume command as is, replacing only `CHG-<id>` with the real change ticket number:

```bash
gpu-fault-admin deploy \
  --state-dir /secure/gpu-fault \
  --approve-profile-plan <plan_sha256> \
  --reference CHG-12345
```

This command first completes the approval on the same state-dir, then continues the original deployment until verify, stability and commit finish.
There is no step three.

The site profile's version name follows the registration cluster's provider family: when the policy is that family's
example template's and the declared snapshot, if present (`profiles/eks-v1.yaml` / `profiles/hyperpod-v1.yaml`), carries
the same policy, the version returns to the declared `eks-v1` / `hyperpod-v1` and reuses the snapshot as is (coming back
from a live base of the other family, the plan reads `regional-hyperpod-<digest12> -> eks-v1`); otherwise it is
`regional-<provider>-<digest12>`. A template whose `nodeReboot` owner is the other provider's adapter is refused by
`deploy` before any plan is written.

### 1.1 A cluster's own profile on a deployed site

A GPU cluster whose provider the site profile cannot reboot (a plain EKS cluster on a HyperPod site, or
the reverse) is pinned to its own profile version, recorded on `spec.clusters[].runtimeProfile`
(`version`, `source`, `templateSource`, `agentConfigDigest`). Its first version is the provider's declared
one (`eks-v1` / `hyperpod-v1`) and is reviewed by `join-cluster` (`change_kind: CLUSTER_PROFILE_NEW`,
`registration_cluster_id: <cluster>`). The snapshot `profiles/<version>.yaml` is immutable: a version never
names two policies.

When the provider's example profile changes policy after the pin (`templateSource` differs from the pinned
snapshot), `deploy` does not reuse the version. It computes the successor version
`regional-<provider>-<digest12>` (the first 12 hex digits of the template's policy digest), writes the plan
and stops before the release bundle and its gates:

```text
Runtime Profile policy changed; the deploy stopped for review.
Review /secure/gpu-fault/release-deploy/profile-plan.json:
  ...
  cluster: gpu-eks-cluster
  version: eks-v1 -> regional-eks-8e51b0c2a7f3 (this cluster only)
  change_kind: CLUSTER_PROFILE_NEW
  changes:
    - cluster gpu-eks-cluster (eks-managed) moves from its own Runtime Profile eks-v1 to regional-eks-8e51b0c2a7f3: ...
  plan_sha256: <plan_sha256>
Record plan_sha256 on the change request; once it is approved, rerun:
  gpu-fault-admin deploy --state-dir /secure/gpu-fault --approve-profile-plan <plan_sha256> --reference CHG-<id>
```

The resume command approves the plan, writes `profiles/regional-eks-<digest12>.yaml`, re-pins the cluster
entry (version, source, templateSource, the release's `agentConfigDigest`), registers the version with the
release and continues; the node wave re-pins the cluster's Agents. `eks-v1` keeps its snapshot untouched.
A cluster joining after the change derives the same successor version and reuses its approved snapshot.
Do not delete the pinned snapshot to "free" the version name: the snapshot is the proof of the policy that
was approved under it.

### 1.2 Several clusters needing one version

The site has one plan file, so a cluster's own profile is planned per **version**: one `CLUSTER_PROFILE_NEW` plan names
every cluster of the operation that needs the version -- the several `--gpu-cluster-arn` of one `join-cluster`, the new
GPU clusters of `deploy --gpu-cluster-arn` (both run the batch join), or every member a `deploy` moves to the same
successor. `registration_cluster_id` and the snapshot anchor are the smallest cluster id among them, so whichever command
lists the clusters, in whatever order, computes the same plan -- the deploy's early preflight and the site finalization
included -- and one approval covers all of them. The stop then prints `clusters: <id>, <id>` and `(these clusters only)`,
`changes` carries one line per cluster, and the last line states that the version is one approval and one immutable
snapshot shared by these clusters.

A batch join reviews exactly like a single join: it stops before it mutates anything, the clusters the plan covers record
the stop in their join journals, the other clusters of the batch are left as they were, and the printed rerun (the same
`--gpu-cluster-arn` list plus `--approve-profile-plan <plan_sha256> --reference <CHG>`) continues. A managed cluster that
has no profile of its own and no approved snapshot (a plain EKS cluster a batch join completed before batch joins planned
cluster profiles) gets its first own version planned the same way by `deploy` (reviewed against the site profile); the
approval writes the snapshot and pins the entry. A pinned version whose snapshot is missing is still refused.

An approval covers exactly the plan it was given for: when the plan changes before it is consumed (for example the set of
clusters needing the version changed), the old approval is archived as `SUPERSEDED`, the new plan is written and the
command stops and prints the new digest. The control plane stores a version once, under the cluster that first registered
it (`cluster_id`); the release compares a registered profile without that registration anchor, so a join appending a
cluster, a batch join or commit sorting by cluster id, a `deploy` ordering by EKS ARN or a `remove-cluster` taking the
anchor away never reads as policy drift, and a `join-cluster` release registers a new version under the joining cluster.
Stored profiles are read as they are and never rewritten.

## 2. State Transitions

```text
four-parameter deploy
  |
  +-- Profile unchanged ---------------------------> normal release
  |
  +-- Profile changed -> profile-plan.json -> stop and print the resume command -> human review
                                      |
                                      +-- deploy --approve-profile-plan <sha> --reference <ref>
                                            |
                                            +-- plan not drifted -> approval takes effect -> release continues -> CONSUMED
                                            +-- plan drifted -> old approval SUPERSEDED -> stops again and prints the new digest
                                            +-- release failed and plan not drifted -> rerunning the original four-parameter deploy resumes
```

Administrators do not create the approval archive directory or copy the plan file. The resume command automatically creates
`release-deploy/profile-approvals/<plan_sha256>/` and writes the audit files.

## 3. Review Prerequisites

Before running the resume command confirm:

1. The same `state-dir` and administrator identity as when the plan was generated are used.
2. The release comes from trusted signed artefacts, and the ordinary deploy did not enter production with a staging-only candidate.
3. The external change ticket or maintenance window has clearly stated the target Profile and risk scope.
4. If the change opens reset, reboot, warm spare or other destructive capabilities, the corresponding maintenance window and phase acceptance are satisfied.
5. No other deploy command is using the same `state-dir`.
6. `profile-plan.json` has not been hand-edited, copied back or had its permissions changed.

## 4. Reviewing the Plan

The stop message has already printed the fields the administrator must review; for the complete content look at the JSON file directly:

```text
<state-dir>/release-deploy/profile-plan.json
```

| Field | Review requirement |
|---|---|
| `site_identity.site_name` | Must be the target site |
| `site_identity.aws_region` | Must be the approved Region |
| `site_identity.cpu_eks_arn` | Must be the site's stable CPU control plane |
| `site_identity_sha256` | Recomputed by the program to prevent the readable identity and digest disagreeing |
| `registration_cluster_id` | The site's current registration cluster (`spec.runtimeProfile.registrationClusterId`), which new versions register under; `remove-cluster` moves it to a remaining member of the same provider when it removes that cluster. Registered versions and their immutable snapshots keep the anchor of their first registration, not required to still belong to the current GPU set |
| `current_version` / `desired_version` | Must match the version migration described in the change ticket |
| `change_kind` | Must be consistent with the risk and maintenance window |
| `changes[]` | Every capability, mode, owner and adapter change must be explained |
| `live_profile_sha256` | The online Profile baseline at approval time |
| `policy_digest` | The canonical target policy digest |
| `source_sha256` | Digest of the developer template's original content |
| `snapshot_sha256` | Digest of the immutable Profile snapshot about to be released |
| `plan_sha256` | The binding value of the whole reviewed plan; must be written into the change ticket and passed unchanged to `--approve-profile-plan` |

Meaning of `change_kind`:

| Value | Administrator judgement |
|---|---|
| `EXPANSIVE` | Expands capabilities or execution permissions; focus on destructive capabilities and the maintenance window |
| `OWNER_CHANGE` | Owner or adapter change; confirm there is no second writer |
| `RESTRICTIVE` | Tightens or disables capabilities; confirm it does not break the current recovery SLO |
| `IMPLEMENTATION_CHANGE` | Observed implementation or version change; confirm the implementation has been accepted |
| `UNKNOWN_BASELINE` | The online baseline cannot be proven; must not be approved; fix the evidence or state first |

`UNCHANGED` does not generate a Profile change awaiting approval.

A deployment that changes only the template, with code and site unchanged, generates a plan just the same: before judging `UNCHANGED`, the staging layer first compares
the template with the online Profile using `plan_runtime_profile`, and when a change awaiting approval exists it goes straight down the complete
application release path.

## 5. Binding the External Approval and Continuing the Deployment

`--approve-profile-plan` accepts only the `plan_sha256` of the current plan awaiting approval:

- When there is no `profile-plan.json` under the state-dir the command refuses to run -- the first deploy cannot carry this parameter.
- When the passed value does not match the pending plan's digest the command refuses to run and gives the current pending digest in the error;
  no active approval is written.
- Once the same plan is approved it cannot be overwritten with a different `--reference`, so confirm the change ticket number is correct before running.

The approval completes inside the same file lock, recording the approver identity (STS caller ARN), change ticket number, approval time and site
identity, and then the same process continues the release. The releaser recomputes the site identity, Profile target and live baseline,
and only when all match does it generate the immutable snapshot, update the internal site and perform deploy, verify, stability and commit.

If the template, target policy or live baseline changes again between approval and release, the old approval is archived as
`SUPERSEDED`, and the command stops again printing a new `plan_sha256` and resume command; review again per section 4.

## 6. Success Evidence

The resume command automatically creates:

```text
<state-dir>/release-deploy/profile-approvals/<plan_sha256>/
  plan.json
  approval.json
```

After a successful release it automatically adds:

```text
consumed.json
```

On plan drift it automatically adds:

```text
superseded.json
```

The corresponding release state is at:

```text
<state-dir>/release-deploy/<release-id>/state.json
```

Success criteria:

1. The release state is `COMPLETED`.
2. `profile_approval_audit.status=CONSUMED`.
3. The `plan_sha256` in the audit exactly matches the change ticket.
4. The active `profile-plan.json` and `profile-approval.json` have been deleted.
5. `gpu-fault-admin status` confirms the target Profile is registered without drift.

## 7. Exception Handling

| Symptom | Handling |
|---|---|
| deploy failed but there is no `profile-plan.json` | Not an approval pause; fix per the original error and rerun without `--approve-profile-plan` |
| `--approve-profile-plan` refused: no plan awaiting approval | First run the four-parameter deploy without that parameter to let it generate and print the plan |
| `--approve-profile-plan` refused: digest mismatch | The plan changed after review; review again per the current digest given in the error and update the change ticket |
| `site_identity` or its digest mismatch | Stop; confirm the state-dir and CPU control-plane identity |
| The site `source` path drifted, but the local version snapshot SHA equals the live SHA | deploy automatically restores the baseline from that immutable snapshot and corrects the site |
| `UNKNOWN_BASELINE` | The local snapshot is missing or the SHA disagrees; do not approve; first restore trusted live Profile evidence |
| Release failed, approval still active and plan unchanged | Fix the release problem and rerun the original four-parameter deploy; no repeated approval needed |
| Old approval marked `SUPERSEDED` | Review again per the newly printed `plan_sha256`, then run the new resume command |
| Release completed but archive finalisation failed | Rerun the original four-parameter deploy; the system completes consumption with `ALREADY_APPLIED` |
| Message that an approval or release is in progress | Wait for the current process to finish; do not delete the lock file |
| GPU cluster join or deregistration | Use the formal join/remove commands; the GPU set does not change the stable Profile approval identity |
| `approved Runtime Profile snapshot ... has a different policy than ...` on a cluster-owned version | The template policy changed under a pinned version; rerun `deploy` -- it plans the successor `regional-<provider>-<digest12>` (section 1.1) and prints the resume command. Do not remove the pinned snapshot |

## 8. Forbidden Operations

- Do not create the `profile-approvals/<plan_sha256>/` directory by hand.
- Do not copy, move or edit `plan.json`, `approval.json`, `consumed.json` by hand.
- Do not delete the active approval, the lock or a failed release state to force a rerun.
- Do not inject the approval reference through environment variables, hidden parameters or by calling the internal `release-deploy` directly.
- Do not run two deploy commands against the same `state-dir` in parallel within the same maintenance window.
- Do not treat `plan_sha256` as a Secret; it is an audit identity, not a credential.
