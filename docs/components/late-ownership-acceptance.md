# Physical Late-Ownership Acceptance

The connected companion runner is
`scripts/e2e/regional/run_late_ownership_acceptance.py`, assembled by
`late_ownership_entry.CASE` and `late_ownership_live.LiveBoundaryIO`.
It covers the controlled late-ownership boundary for
`GF-REGIONAL-PREEMPT-033` and `GF-REGIONAL-DESTR-015`.
It does not replace the ordinary DESTR-015 parallel two-node recovery and single
restart. Companion results never promote the ordinary case automatically.

No LIVE execution has been performed as part of this implementation. Local
tests, including real tracing of harmless owned children, remain `LOCAL_TEST`.
The parent integration owns the scenario matrix, catalogs, and formal sequence.

## Exact Guarantee

An Executor read **before HTTP submission** does not protect a command waiting
inside the Node Agent. The product therefore also enforces this sequence:

1. The signed command is accepted and dequeued by the Agent.
2. A supported handler reaches its concrete physical checkpoint and publishes a
   fresh `AGENT_PRE_SPAWN` challenge. No protected spawn/write crosses that
   checkpoint while it waits.
3. The Executor checks its current lease and freshly reads workload UID,
   complete owner references, STOP ownership, original participants, Node
   incarnations, and GPU Pods on every participant node.
4. The Executor signs a challenge-bound permit using the existing node-specific
   Node Action key. The Agent rechecks command validity, generation, fencing,
   quiesce state, and current clients as applicable. It checks permit expiry
   again after those local reads, before the physical call.

The challenge is bounded by 90 seconds and the original command expiry.
A permit expires no later than three seconds after the Executor issues it and
never later than the challenge. Guarded command expiry is also capped by the
workflow lifetime. A slow final client check does not extend that permit.
Existing full sampled client preflight remains; the post-permit client check is
additional. Each reset attempt and protected install/bind checkpoint rechecks.

The deterministic experiment introduces the owner change or late sibling
**after queueing and before permit issuance**, while the native callback is
parked. A fresh product rejection and an independent continuous exec trace
establish that this controlled change did not cross the reset-exec boundary.

This is **not atomic Kubernetes-to-kernel fencing**. Kubernetes reads are
sequential. Ownership can change after an individual read, after permit
issuance, or during OS scheduling before the spawn. Three seconds bounds permit
validity, not the age of every individual Kubernetes read or an atomic
read-to-syscall transaction. The mechanism does not claim to reject every such
arbitrary post-read change. Closing that stronger TOCTOU property requires an
additional coordinating fence; this implementation introduces no global
admission resource and must not close a matrix requirement with that claim.

## Live Variants

The guarded entry requires explicit `--case`, `--scenario`, and
`--ordinary-destr015-evidence` arguments. It retains
the original site, predecessor, release, GPU context, cluster, namespace, two
nodes, maintenance deadline, source-digest and plan checks. Execute confirmation
is `PHYSICAL_LATE_OWNERSHIP_EXECUTE`. A separate reviewed plan is required for
each variant; neither this document nor a local test authorizes execution.

| Scenario | Controlled live I/O | Required result |
| --- | --- | --- |
| `unchanged-owner` | Preserve the owned workload and participant identities; release the parked callback into fresh checks. | Exactly one completed reset exec on each approved node. |
| `ownership-drift` | Create one run-owned **suspended** Job anchor; add its non-controller owner reference to the existing PyTorchJob using UID/resourceVersion JSON Patch tests; read it back. The workload UID is not replaced. | `STOP_OWNERSHIP_DRIFT`; zero reset attempts on both nodes. |
| `late-sibling` | Create one same-source-owner Pod with a new UID on node B, one GPU, the already approved training image, no service-account token, a bounded CUDA allocation, readiness acknowledgement and a 240-second lifetime; observe its GPU PID/cgroup. Node B is not quiesced in this negative variant. | `STOP_PARTICIPANTS_CHANGED`; zero reset attempts on both nodes, including candidate node A. |

Wrong/replaced workload UIDs are additional local negative controls. They are
not described as a live UID-replacement experiment.

### Formal Sequence And Storage

This physical companion cannot run in PREEMPT-033's ordinary phase-6 slot.
The manual session takes place **after the ordinary DESTR-015 has completed
successfully and its cleanup has been verified**, in an approved physical
maintenance window. Do not insert this runner into the ordinary PREEMPT-033
slot or silently replace the normal DESTR-015 command.

The current normal runner's predecessor constant is DESTR-012. Preserve that
precondition and provide its ordinary-run artifact explicitly through
`--predecessor-evidence`; a separate companion directory has no implicit copy.
Also provide the completed ordinary DESTR-015 result explicitly through
`--ordinary-destr015-evidence`, using its canonical path
`<ordinary-run>/cases/GF-REGIONAL-DESTR-015/GF-REGIONAL-DESTR-015.json`.
There is no sibling-path inference or selective-mode bypass. Before protocol
inspection or physical setup, the companion requires PASS, no execution errors,
confirmed command quiescence, workload absence, resource cleanup and node
restoration. Each host-probe cleanup must account for its Pod, ConfigMap, host
script and unresolved creation state; an empty or partial probe inventory is
not closure. Empty image-prewarm cleanup is valid when both images were cached.
It compares the ordinary release, cluster, ordered node scope,
Node UID/boot identities, runtime profile and deployment identity to current
preflight. The bounded regular-file read refuses a changed or symlinked result.
The exact evidence bytes' digest is included in the reviewed companion plan,
so replacing the ordinary proof after planning requires a new plan.
Use a separate, non-nested run directory for **each variant**.
The entry refuses ordinary result storage, nesting in either direction between
the ordinary and companion roots, and predecessor evidence inside the companion
output root. Ordinary plans, results and PASS files remain untouched.

The canonical companion uses `--case GF-REGIONAL-DESTR-015` and retains all
three reviewed variant runs once: `unchanged-owner`, `ownership-drift`, and
`late-sibling`. PREEMPT-033 cross-references these same mechanism receipts.
Its runner alias remains available for optional investigation, not as a second
required set of physical trials.
Each trial keeps its own plan/source pins, scope, mutation journal, physical
receipts and cleanup result. Stop on a failed or incomplete trial.

The companion's files are
`<variant-run>/cases/<case>/late-ownership/<scenario>/result.json` and
`receipts.json`, not the ordinary `<case>.json`. Retain a manual evidence index
listing the three variant paths and digests together with the original ordinary case
results. There is no automatic case-PASS roll-up. One variant, a case alias,
or a local test is not complete formal case execution coverage; matrix updates
must preserve the bounded temporal guarantee stated above.

`OwnedMutation` records creation intent before sending and the acknowledged UID
before readback. Unknown create acknowledgements cannot be adopted by name.
After the native STOP receipt has been validated, `acknowledge_stop` binds only
the product's expected STOP annotations and suspension transition to the original
declared source. It does not adopt the current GET as a new baseline. All three
variants and cleanup use that acknowledged transition; unrelated source changes
remain a refusal. Ordinary product STOP changes are not foreign ownership drift.
Restoration tests the original workload UID, current resourceVersion and the
injected owner before changing it. Foreign replacement is never overwritten.

## Concrete I/O

- **CPU:** `late_ownership_control` uses existing Store methods in the CPU Pod.
  It creates an auditable RUNNING holder whose lease outlives the fixed mutation
  lifetime. Fresh, nonce-bound holder checks verify workflow, incident, owner,
  epoch, fence, lease and lifetime before forward steps and during the native
  permit exchange. Exact owned completion records SUPERSEDED/withdrawn, never a
  workload restart. Partial completion can be reconciled without reopening it.
- **GPU Executor:** one UID/container-pinned, TLS-verified Kubernetes exec
  channel runs the installed Kubernetes and Node Action adapters. Only the
  measured acceptance code is supplied over framed stdin; product modules are
  not replaced. Node keys remain in this GPU Executor. No CPU GPU kubeconfig,
  execution token, or new credential is introduced. The selected Ready Pod's
  GET must match its name, namespace and UID and report the running container's
  imageID. Exact ReplicaSet and Deployment owner identities and all three
  executor image declarations must agree; a declared SHA-256 image pin must
  also agree with the running imageID. Missing or mismatched identity is refused
  before exec, and identity is checked again after protocol inspection.
- **Nodes:** an independently owned, privileged host-PID witness Pod runs its
  observer as its **main process**, not as a long-lived exec session. This is
  necessary because quiesce stops kubelet. The daemon uses a private Unix
  sequenced socket, downward-API Pod UID, peer PID/start time/boot checks and a
  fixed deadline. RPCs occur before quiesce or after services return.

Both observers attach and physically calibrate through their actual Agent
processes before STOP is enabled. After actual STOP, the Executor acknowledges
containment; independent source-Pod GPU-client checks happen on both nodes while
kubelet is still available. Only then does the Executor proceed to quiesce and
the native post-queue reset checkpoint.

The STOP receipt identifies the queued command, Agent generation and challenge,
the source participants, and both calibrated witnesses. The mutation receipt
is linked to it. `RECHECK_ONLY` releases the acceptance rendezvous into product
validation; it is not the signed Node Action authorization.

## Independent Witness

`late_ownership_trace.AttachedExecWitness` attaches `strace -f -ttt -xx` to the
exact Agent process and checks every thread's kernel `TracerPid`. The live
observation domain is the resolved `nvidia-smi` executable, filtered by its
exact path and hash-pinned before and after observation. Recognized read-only
queries are distinct from reset attempts.

Raw exec data stays in an anonymous, size-bounded regular file. Only sanitized
receipts leave the observer. Complete exec/exit pairs, failed execs and killed
action processes are accounted for. Missing calibration, process replacement,
truncation, unknown invocation forms, unfinished records or lost tracing cause
failure, never a zero-action receipt. The tracer's parent-death guard and
cleanup target only its owned pidfd, never the observed Agent.

An empty ledger, FAILED workflow, STOP timestamp, model UID, successful cleanup,
or zero model counter is not independent no-action evidence. Local controls
exercise the actual Agent checkpoint and a real kernel trace while deliberately
leaving a model counter at zero.

The witness proves attempts at this executable boundary, not arbitrary NVML
ioctls, physical chip damage, unrelated processes, or provider-side reboots.
This companion does not submit provider reboot/replacement commands. The
ordinary case's runtime/health checks and existing provider safeguards remain
necessary.

## Refusal And Compatibility

`KubernetesStopOwnershipValidator` is wired by the cluster Executor bootstrap
and dispatch, including batched steps. The Node Action transport preserves
polls of accepted commands while checking each new submission and native
challenge. Direct HyperPod mutation checkpoints also check ownership; their
activation-inhibition guard remains independent and composes with this check.

Node Agent fleet protocol is 4; the challenge protocol is
`node-final-ownership/v1`. Normal Agent configuration always enables final
ownership enforcement. A new Agent refuses legacy protected submissions. A new
Executor sends a signed ownership marker that an old strict Agent cannot
silently ignore. Missing capabilities, receipts or validators fail closed.
An unmarked/unsupported mutating handler is refused before invocation.
Explicit legacy dependency construction exists for local unit compatibility,
not as a production environment escape hatch.

Ownership denials, expired permits, lost callers, unsupported boundaries and
unverifiable protocol results carry `safety_rejection` and
`manual_confirmation_required`; they are not retryable hardware-reset failures.
Both the in-place branch ladder and whole-workflow classifier must route them
to an operator, not reboot/replacement. Tests assert zero provider calls.

An unreadable or contradictory response after possible delivery is not proof of
pre-dispatch refusal. The transport retains the exact command and intent digest
as an unknown outcome, polls without issuing a replacement submission, and
refuses to adopt an older phase or changed intent as current success.
A negative permit records the caller's decision, not what the Agent already did;
returned execution evidence is not overwritten by that intended denial.

An Agent refusal includes its earlier granted checkpoints. A granted checkpoint
does not prove physical completion, but it prevents the entire command from
being described as never started. Only an explicit first-checkpoint refusal
with empty checkpoint history can provide that native no-action proof.

Software workload restart retains source UID/owner/participant checks and the
existing signed restart authorization. It does not freeze old Node
incarnations after legitimate recovery or require unrelated jobs to stay quiet.
Hardware actions retain strict Node UID/boot and all-node containment checks.
Compensating service/scheduling restoration retains its existing ownership
safeguards and is not blocked by a new-action ownership refusal.

## Cleanup And Limits

Successful closure requires authenticated terminal results for every accepted
Node command, matching command IDs and operations. GPU services are restored;
actual host state, boot, inventory and timers are checked before scheduling is
restored. CPU holder completion is confirmed before continuous witnesses close.
The Executor process must acknowledge completion **and exit successfully**.
Witness Pods/mailboxes, injected resources, the owned workload, prewarm and
original host probes are then cleaned with their ownership checks.

Cleanup has a bounded compensation window; it cannot extend action authority.
An ownership refusal may already have spent the quiesce window's reset
allowance. That allowance is not reopened automatically.

Step and workflow deadlines do not prove that an outstanding Node Action stopped.
The control-plane executor preserves unresolved action details, refuses automatic
GPU service restoration while non-restoration actions remain uncertain, and uses
`BLOCKED / NEEDS_OPERATOR` instead of releasing the workflow's node occupancy.
Ordinary confirmed failures and explicit no-action refusals retain compensation.
An existing restore can still be polled; an uncertain restore cannot justify
releasing node occupancy either. This is control-plane safety bookkeeping, not a
Kubernetes write fence. Existing node-local quiesce failsafes remain unchanged
and are not physical-completion receipts.

Lost controllers, unknown acknowledgements, foreign UIDs, broken traces,
unconfirmed terminal commands, restoration failure or incomplete cleanup prevent
PASS. After action start, unproven quiescence defers resource destruction.
Existing Node quiesce failsafes remain enabled, but a failsafe is not a cleanup
receipt. A failed or interrupted experiment can require operator reconciliation
using its private scope, mutation journal, host-probe records and command IDs;
there is no claim of unattended recovery from every partial state.

An operator must resolve pending commands, restore/verify the original nodes,
and reconcile exact owned resources before another trial. A same-name
replacement is not cleanup authority. Missing physical evidence cannot be
reconstructed from counters or replayed model results.

## Local Verification

`tests/regional/test_late_ownership_*` covers the connected assembly with fake
I/O, UID/owner/lease loss, late callbacks, cleanup failures, batching, protocol
compatibility and no-escalation controls. Physical local controls use private
sockets, regular files and owned children executing only `/usr/bin/true`.
They invoke no live-runner CLI, GPU command, Kubernetes/AWS mutation, host service
operation, or PostgreSQL connection.
