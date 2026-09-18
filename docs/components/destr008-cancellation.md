# DESTR008 Cancellation And Inhibition

This is the acceptance-only composition used by
`run_destr008_warm_spare_shortage.py`. It does not authorize a live drill.
The existing maintenance approval, predecessor, EKS-only HyperPod,
`NodeRecovery=None`, disabled auto-resume and no-provider-replacement rules remain.

## Independent Authorities

Every synthetic shortage request carries the immutable, solely inhibiting
`activation_forbidden: true` authority. The request accepts only literal true;
normal requests omit the field and retain their behavior. Presence in stored
step parameters, including malformed values, requires Executor protocol 4 and
refuses allocation/activation, including cached reservations and confirmation
paths. Ordinary health/topology/occupancy checks still run first.

The case must observe its intended health failure. An
`ACTIVATION_FORBIDDEN:` error, `activation_inhibited` or
`cached_activation_rejected` outcome means FAIL, not a successful shortage test.
The marker is not permission to act, does not widen an allowlist, and cannot
be removed by merging with an ordinary pending replacement.

For expiring fixtures, a run-owned GPU ValidatingAdmissionPolicy and Binding
add defense in depth for the selected Node UID. The policy is not an atomic
cluster-wide allocator lock and a successful dry-run does not prove that every
API server informer has converged. The immutable product-side inhibition is
therefore the primary protection against another/new/cached spare being selected.

A separate CPU Job owns deadline revocation and command-drain observation.
Its immutable Plan binds the original run, event, job/attempt, release, workload
IDs, fault/spare Nodes and fence UIDs. The mutable producer control uses
ConfigMap UID/resourceVersion CAS. One claim permits one POST; a lost response
cannot grant another POST or be replaced by a current snapshot.

The installed-capability inspection lives in the acceptance-only
`probes/destr008_inhibition_probe.py`, not in the shared production protocol
module. `destr008_capabilities.py` sends bounded source over stdin to the selected
component interpreter with isolation enabled. The loader verifies its SHA-256
before execution, and the returned source digest must match. The proof still
inspects the actual installed API or Executor code without constructing services.
Keeping the role-specific inspection outside shared runtime imports prevents CPU
routes and database implementations from entering Executor or Node Runtime wheels.

## Ordering And Bounds

1. Verify complete, stable API and Executor populations and their actual installed
   inhibition capability. Verify the CPU worker image, config and scoped
   terminal-history Store implementation, including every actual Ready worker.
2. Submit the unique workload and obtain its Running observation under the
   approved Profile. This scheduling wait does not consume the fixture outage.
3. Arm the GPU fence, then create the CPU watchdog supporting resources and Job.
   Validate the admitted Job and its exact owned Pod before UID/RV removal of
   the scheduling gate. Only a fresh, complete ARMED receipt authorizes a fixture.
4. Bind the fixture to the Plan. The service window has independently acknowledged
   host recovery; the GPU holder has create-only UID custody and a fixed deadline.
   At least 120 seconds of cancellation headroom must remain after startup.
5. Claim the producer immediately before the single synthetic POST. Validate its
   exact response before recording the incident/workflow acknowledgement.
6. Preserve at least a 60-second margin between cancellation and fixture expiry.
   The cancellation deadline also ends at least 60 seconds before maintenance
   expiry. These times cannot be extended by a retry.
7. Revoke the producer and prove source completion, no pending creation, zero
   active owned commands/workflows, and a stable quiet interval. Only then close
   the GPU fence and retire the CPU observer resources.
8. Delete the workload, restore the fixture, and use the product's validated
   restoration workflow for the proven incident family. Repeat Node UID and
   quarantine-owner checks before mutations. Every scenario must report complete
   cleanup before the full six-scenario case can pass.

The Store inventory uses the existing cluster/job/attempt-scoped history API,
including terminal rows and exact successor lookups. It does not repeatedly scan
the global workflow table. A cancelled or expired lease is not physical
completion. Unknown source/command outcomes keep the protection and make the
case fail.

## Recovery And Custody

The normal administrator site lock and private run locks exclude local concurrent
controllers. Journals bind source, kubeconfig contents, context, namespace, host,
release and Node identities. They are not a distributed lock across copied state
directories or multiple deployment hosts.

Successful create responses supply the original UID before readback. A valid
identity with an unapproved spec confers deletion-only custody of that unchanged
object, never execution authority. Unknown create ACKs are not adopted from
matching public labels. Deletion uses UID and the resourceVersion of the fully
validated object; new same-name objects are refused.

Existing run state enters cleanup-only recovery before normal healthy preflight.
It retains the original attempt and expiry, starts no workload, applies no
shortage and sends no new POST. An explicit replacement CPU cleanup observer
must first stop prior owned observer Jobs; it cannot reopen producer authority.
The original failure is retained even if fresh cleanup succeeds. Separate
`cleanup-attempt-<n>.json` reports do not overwrite or reissue execution PASS.
Another execution requires a new approved run directory.

Unresolved journals, source/identity drift, or unproven command quiescence require
operator reconciliation. Interrupted metadata-only variants do not reconstruct
their original mutation baseline from current labels. Independent service
recovery remains bounded even when the controller is unavailable; see
[Service Window](destr008-service-window.md).

## Verification Limits

Focused tests compose the real watchdog, Memory Store, resource lifecycle,
control protocol and main runner with controlled Kubernetes/host/workload I/O.
Native PostgreSQL tests cover scoped history and protocol-filtered claims.
These are local implementation tests, not Aurora, EKS admission, GPU occupancy
or real service-recovery acceptance. The catalog remains `NOT_RUN` until a
separately authorized live execution produces complete evidence.
