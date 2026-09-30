English edition of `docs/管理员Profile变更审批.md`; the Chinese file remains the source of record until both are maintained together.

# Runtime Profile Change Approval

This document is for regional site administrators and explains, after a deploy stops because of a Runtime Profile policy change, how to review the plan,
bind an external change ticket, let the same deploy continue and keep the audit evidence. The whole procedure has only two steps and needs no `jq`,
internal site path, `PROFILE_APPROVAL` environment variable or hidden parameter.

The Profile is a site-level authorisation policy. Once a new version takes effect it is used by all GPU clusters the site currently manages; GPU cluster
joins and deregistrations continue to be handled by `join-cluster/remove-cluster` and do not enter the stable site
identity digest of the Profile approval.

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
| `registration_cluster_id` | The Profile's stable registration anchor; not required to still belong to the current GPU set |
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

## 8. Forbidden Operations

- Do not create the `profile-approvals/<plan_sha256>/` directory by hand.
- Do not copy, move or edit `plan.json`, `approval.json`, `consumed.json` by hand.
- Do not delete the active approval, the lock or a failed release state to force a rerun.
- Do not inject the approval reference through environment variables, hidden parameters or by calling the internal `release-deploy` directly.
- Do not run two deploy commands against the same `state-dir` in parallel within the same maintenance window.
- Do not treat `plan_sha256` as a Secret; it is an audit identity, not a credential.
