# Atomic Ownership Fence Design

Status: DESIGN ONLY; NOT ADOPTED for the current product boundary. No fence,
resource, activation proof, or LIVE evidence is implemented here. The original
stronger ownership requirement remains unimplemented.

## Product Boundary Decision

The product must not require external controllers, schedulers, or workload clients
to adopt a shared ownership protocol, replace their write entrypoints, or change
their Kubernetes permissions for this guarantee. External writes remain
independently concurrent; assumed cooperation or temporary silence is not evidence
of exclusion. Do not pursue the mandatory external-writer coordinator below as
the implementation plan.

Retain the existing fresh ownership checks, node-local quiesce/client checks,
signed one-use permits, and conservative handling of uncertain physical outcomes.
Detected drift or unverifiable state still refuses action under those checks.
They cannot promise detection of every change after a read or make the eventual
physical call atomic with Kubernetes writes.

This decision does not relax existing safety prerequisites, including
`NodeRecovery=None` and disabled automatic restart/resume. It adds no production
authorization or LIVE evidence. Preserve the original scenario assertion, gap,
critical classification, and denominator; declining this architecture does not
turn an unimplemented guarantee into a passing result.

The remaining sections record the explored alternative and its unresolved
requirements, not current deployment prerequisites or an approved rollout plan.

## Original Required Guarantee

An ownership-changing Kubernetes write and a protected physical action require
a defined order. A write that wins must make original-STOP validation reject
the action; an action that wins must prevent conflicting commits until physical
completion. This includes writes admitted before the final read but committed
afterward, not only changes introduced before permit issuance.

Protect workload UIDs/complete owner chains, original participant Pods, and Node
UID/boot identities, including new siblings/bindings outside the allowlist.
The trust boundary includes API servers, storage, admission, and guarded Agents;
out-of-band etcd writes, compromised administrators, and arbitrary host-root
actions are excluded. No supported enforcement-disable bypass is proposed.

## Current Implementation

- `src/gpu_fault/adapters/kubernetes/stop_ownership.py::KubernetesStopOwnershipValidator.check`
  verifies STOP ownership and all participants through sequential API reads.
- `src/gpu_fault/adapters/node_action/transport.py::NodeActionTransportMixin._final_ownership_permit`
  answers the native Agent challenge after another ownership/lease check.
- `src/gpu_fault/node_agent/late_ownership.py::execute_with_final_ownership`
  validates local state and a one-use permit, without holding Kubernetes writers.
- `scripts/e2e/regional/destr008_admission.py::ActivationFence` is a run-owned
  spare-activation policy, not a persistent ownership journal or global barrier.

Current [bounded behavior](late-ownership-acceptance.md) has no persistent coordinator.

## Proposed Journal And RV Drain

Mandatory admission and the coordinator share a GPU-local, durable, linearizable
journal. Process locks, informer caches, and expiring Leases are not authority.
Missing, corrupt, replaced, or overflowing state refuses grants, never resets empty.

Before Allow, persist request identity, cluster/incarnation, resource/subresource,
verb, UID, observed resourceVersion (RV), intended change, and scope. Retries must
be idempotent but re-evaluate the fence epoch, never replay cached Allow.
Cover owner/namespace deletion, Pod CREATE/binding/eviction/resize/ephemeral
containers, and relevant Node/workload updates. Removable labels cannot opt out.

An action reservation moves OPEN to DRAINING through the same journal CAS,
blocking new overlapping writers and resolving earlier admitted intents before
original-STOP validation. Existing-object RV drain requires UID/RV-conditional
writes that advance storage state: a no-op is not a barrier. RVs are opaque
preconditions, not sortable cross-resource timestamps.

An old writer must have committed before final validation, or be unable to commit
its old attempt and face mandatory revalidation. Prove this for each API path.
CREATE has no existing-object RV to advance. Unresolved writers block acquisition
without independent commit/abort proof; absence, same-name GET, elapsed time,
and client timeout are insufficient. Untracked writers invalidate completeness.

After drain and fresh complete validation, DRAINING may become HELD. The grant
binds journal UID/epoch, release, protected identities, command, Agent incarnation,
and challenge. Retain HELD through physical spawn/write and completion, not just
permit delivery. Authenticated terminal evidence permits RELEASING then CLOSED.
Caller loss, unknown execution, or expiry never unlocks the scope. Recovery cannot
renew the original command deadline or adopt changed ownership.

## Kubernetes v1.33 Source Basis

Official v1.33.0 source motivates this design, not proof for every EKS build:

- [Generic registry Store.Update][generic-store] places `updateValidation`
  inside `Storage.GuaranteedUpdate`. Conflict retries can re-evaluate validation;
  unconditional updates may adopt the latest RV, so a conflict alone is not refusal.
- [etcd3 store.conditionalDelete][etcd-store] checks preconditions and
  `validateDeletion`, performs revision-conditional `OptimisticDelete`, and
  retries against current state on conflict. UID-only deletion is not an RV lock.
  `store.GuaranteedUpdate` can return without writing unchanged bytes.
- [BindingREST.Create and setPodNodeAndMetadata][pod-storage] validate the
  Binding before its internal `GuaranteedUpdate`. They do not rerun ordinary
  Pod-update admission inside that loop. Pod UID/RV preconditions are supplied
  only when present on the Binding; the proposed protocol must require/prove
  appropriate preconditions or retain the unresolved binding intent.

Allow is not commit acknowledgement or a stock webhook completion callback.
Prove each writer/subresource/retry path, including delete-during-update.

## Unresolved EKS Activation Contract

The journal is complete only if every serving API server already invokes the
mandatory admission path. A stored webhook/policy configuration is not evidence
that every server's admission cache enforces it.

An authoritative activation contract must bind the cluster incarnation,
complete serving membership epoch and server incarnations, effective admission
configuration/scope, TLS trust, journal identity, release, and implementation.
It must establish enforcement on every serving member, account for requests
admitted earlier, and prevent replacements/new members from serving affected
writes before the same enforcement and drain requirements hold.

A ready flag, load-balanced dry-runs, delay, or self-signed operator boolean is
not proof. No repository EKS interface proves complete serving membership and
effective admission state. This alternative would require a platform contract
with an authority that observes/controls it; an upload field is no solution.
Those unresolved requirements do not create a new product startup stage.
Stronger action authority and atomic enforcement claims remain unavailable.

Reuse [Node Key Custody Evidence](node-key-custody-evidence.md) mechanics:
`src/gpu_fault/admin/node_key_custody_crypto.py::CustodyCrypto` and
`src/gpu_fault/admin/node_key_custody_chain.py::verify_chain` illustrate external
trust pins and separate approval/provisioning/witness authorities.
Define distinct fence statements; signatures authenticate supported observations,
not unobserved API-server coverage. Do not copy optional custody enablement.

## Deployment, Rollback, And Uninstall

Proposed resources: GPU-local HA Deployment/Service, dedicated SA/RBAC, durable
journal, mandatory admission configuration, TLS/grant credentials, PDB, and
restricted networking. Admission markers may require a mandatory mutating webhook.
Require non-target GPU-EKS capacity that survives quiesce, not just anti-affinity.
No GPU kubeconfig on CPU; no fleet master or CPU database credentials in the fence.

Provision a dedicated Service-DNS certificate and verify the actual serving
chain, SAN, expiry, and API-server-to-backend reachability. Rotate with verified
trust overlap; retain authority history required by active fences. Public
preflight does not issue certificates, sign activation, or acquire action grants.

Integrate `ReleaseComponent`/`build_execution_plan`, `bootstrap_gpu_target`,
`RegionalRelease.join_cluster`, and `upgrade_gpu_target` before Agent activation.
Bind source/manifests in `scripts/component_wheels.py` and `config/release-identity.yaml`.
Extend inventory kinds/phases; preserve `regional_deployment_inventory.GPU_EXECUTORS`.
Hold new protected actions per cluster, drain accepted physical work, establish
admission authority, stage pins, and roll Executor then mandatory guarded Agents.
Legacy peers must reject new protected submissions; retain safe result polling
and restoration. Finalize pins and verify every participant before lifting holds.

`rollback_target` must reject a target lacking fencing capability. Never restore
live journal state, erase uncertain holds, or serialize private keys through
`regional_manifest_snapshot`. Missing historical state cannot authorize removal.
Extend `cleanup_state.PHASES` and the existing admin removal/uninstall paths:
seal new grants, prove physical quiescence and resolved writer intents, then
retire admission before its backend and state, using UID/RV ownership checks.
An active or unknown fence blocks retirement and namespace deletion.

## Required Test Matrix

| Interleaving or failure | Required observation |
| --- | --- |
| Writer first, delayed commit | Drain accounts for the write; changed STOP identity prevents any physical action. |
| Action first | Conflicting writes cannot commit while HELD; release follows physical completion. |
| Update/delete retry or no-op drain | Retry revalidates the epoch; unchanged storage is not accepted as a barrier. |
| Binding without or with stale UID/RV | Refuse or retain unresolved intent; no reliance on Pod-update admission. |
| Untracked writer or API-server cache lag | Activation/completeness remains unproved; no action grant. |
| API-server replacement | Cannot enter the serving set without the authoritative enforcement contract. |
| Unknown CREATE or lost acknowledgement | No snapshot adoption, timeout unlock, or fabricated completion. |
| Caller/coordinator loss or journal replacement | Hold survives uncertainty; no replay, duplicate action, or reset-to-empty. |
| Rollback/uninstall with active fence | Preserve enforcement/state and reject unsafe retirement. |

Use owned hermetic API-server/storage and harmless physical-boundary tests.
Fake I/O cannot prove EKS activation; this design asserts no deployment/LIVE result.

[generic-store]: https://github.com/kubernetes/kubernetes/blob/v1.33.0/staging/src/k8s.io/apiserver/pkg/registry/generic/registry/store.go
[etcd-store]: https://github.com/kubernetes/kubernetes/blob/v1.33.0/staging/src/k8s.io/apiserver/pkg/storage/etcd3/store.go
[pod-storage]: https://github.com/kubernetes/kubernetes/blob/v1.33.0/pkg/registry/core/pod/storage/storage.go
