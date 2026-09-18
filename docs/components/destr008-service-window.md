# DESTR008 Service Window

This acceptance-only safeguard owns one bounded stop of `kubelet.service` or
`gpu-fault-node-agent.service`. It is not a production recovery operation, a
generic service probe, or authorization to run DESTR008. Local tests do not
establish LIVE behavior. EKS-only HyperPod, `NodeRecovery=None`, disabled job
auto-restart/auto-resume, and the prohibition on provider replacement remain
unchanged.

## Controller API

The parent records the original arguments before creating the fixture:

```python
service = WarmSpareServiceFixture(
    warm,
    node=node,
    image=host_probe_image,
    case_id="GF-REGIONAL-DESTR-008",
    run_id=run_id,
    state_directory=private_state_directory,
    node_uid=approved_node_uid,
    plan_sha256=approved_plan_sha256,
    release_id=original_release_id,
    maintenance_expires_at=original_aware_datetime,
)
try:
    service.create()
    scheduled = service.stop(
        "kubelet.service", restore_seconds=180, delay_seconds=15
    )
    # The parent separately observes the scenario's actual shortage.
finally:
    service.close()
```

`close()` only releases the local controller journal lock. It does not restore,
cancel a systemd job, stop the independent process, or delete host/Kubernetes
resources. The parent must call it from `finally`, including failed stop and
failed restore paths. Ordinary successful cleanup uses `restore()` followed by
`cleanup()`; `cleanup()` itself also requires host recovery to complete before
deleting the transport.

For cleanup-only resumption, construct the same fixture with the **original**
arguments and call `resume_cleanup()`. Do not call `create()` or `stop()`, set a
service name, recompute the maintenance deadline, or substitute a new Node UID,
release, plan, image, source or kubeconfig. Construction performs no remote
actions. `journal_path` exposes the deterministic private controller journal
location. The `service` property is read-only and comes from the loaded binding.

`resume_cleanup()` returns:

```python
{
    "phase": "CLOSED",
    "resumed": True,
    "host": bound_host_closure_report,
    "probe_residuals": verified_transport_residuals,
}
```

The host report has a binding digest, target service, absolute deadlines,
baseline/restored state, and separate timer, stop-process and restore-process
quiescence receipts. Missing, malformed, foreign or unresolved evidence raises;
there is no boolean success fallback. No host window requested is explicitly
`NOT_REQUESTED`, not evidence that a service was stopped and restored.

## Ownership And Arming

The controller uses a private, flock-protected journal with atomic replacement
and file/directory fsync. Its stable filename depends on the case, run and node,
not mutable source/configuration inputs. The journal binds the approved plan,
release, Node UID, transport owner, source digests, GPU context/namespace and
kubeconfig digest. Mutating request intent is saved before transport submission.
The existing `HostProbeFixture` retains its own Pod/ConfigMap UID and owner
receipts. Controller resumption cannot manufacture those receipts.

The stdlib host helper holds a node-local lock and a create-only per-service
claim. Before arming it captures the active, idle service invocation; unit
fragment/drop-ins; executable and runtime-slot identities; Node Agent identity
pins; environment-file fingerprints; machine identity and boot ID. Secret
contents are never returned or journaled. Unsupported or ambiguous systemd
properties, environment layouts, identities and file types are rejected.

The host publishes a copied helper, one temporary target timeout drop-in, an
independent restore service, and a separate stop service/timer. Each publication
has a private source inode and a durable receipt before create-only linking.
Same-name objects, including byte-identical replacements with different inodes,
are not adopted. The helper and receipts live under
`/var/lib/gpu-fault-acceptance/service-window/<binding-digest>/`; unit files live
under `/etc/systemd/system/`. These units are never boot-enabled. Service
enablement and all unrelated service configuration remain unchanged.

Only the independent restore service can ACK arming: its systemd invocation,
main PID, exact executable arguments and cgroup membership must agree. Both
controller and host require a fresh original-boot ACK before scheduling stop.
A timer name, process existence, or generic readiness boolean is insufficient.
Both targets use the owned asynchronous stop service, including zero-delay
requests. Kubelet's controller transport is never asked to carry an inline stop.

## Nonrenewable Bounds

`restore_seconds` is an integer in `60..600`; `delay_seconds` is in `0..120`.
The absolute `restore_at` is fixed before preparation to the controller's current
time plus these intervals. Arming consumes part of that interval; ACK or request
retries do not extend it. Zero delay schedules no earlier than one second.
The host also retains equivalent same-boot elapsed deadlines, so moving the wall
clock backward cannot renew authority.

Recovery has an additional fixed 180 seconds, and the whole window must fit both
the approved maintenance expiry and the host's 900-second maximum. The target
job is bounded by an owned `JobTimeoutSec=45s` drop-in with 30-second start/stop
timeouts. The independent restore service has a bounded runtime and no restart
policy. A late stop is rejected; a late start is rejected if its bounded job
cannot fit the remaining recovery authority.

`failsafe_at` is the fixed restoration **start**, not a guarantee of healthy
service at that instant. The parent must keep its activation/cancellation
protection until its own workflow and command quiescence checks complete.
Service recovery does not prove that a warm-spare scenario remains blocked or
that a late allocation is safe.

## Restoration And Cleanup

Restoration first durably seals the window against every later stop request.
It verifies the complete owned resource set, stops the delayed timer, stops its
service if necessary, and proves both have no pending systemd job. The stop
service must have no main/control process and an empty cgroup. The target's
pending stop job must also finish before a start is permitted. Failed timer
cancellation, a populated cgroup, an in-flight target job, or an unknown read
cannot produce restoration or disarm success.
If target stop submission began, its systemd acknowledgement must also have
been durably recorded. An exited client without that receipt is not proof that
an unconfirmed request cannot arrive later; recovery remains unresolved.

Only the unchanged baseline service may be started. An already active original
invocation needs no start. An independently restored invocation requires its
durable receipt. A lost target-start ACK without such a receipt is unresolved,
not permission to adopt another invocation or retry a start blindly.

The independent process may report `RESTORED`, but cannot attest its own exit.
Controller cleanup separately stops and verifies that process, verifies target
availability and job quiescence, then removes only the recorded unit/drop-in
inodes and reloads systemd. Explicit unit absence and the same restored target
invocation are required before `CLOSED`. Partial publication, lost replies and
interrupted retirement retain their receipts for retry. A minimal host
tombstone, source receipts and helper remain to fence late requests; they are
not enabled recovery units. The per-service claim is released only after closure.

The controller closes only after host closure and verified owned transport
cleanup. A fresh controller cannot create, rearm or re-stop an existing attempt.
Command supervision loss is journaled and forbids further remote commands,
including fresh-process automatic cleanup. The parent retains the original case
failure; cleanup-only closure never awards a new scenario PASS.

## Assumptions And Limits

- Filesystems, systemd, the interpreter and the local clock are trusted.
  Publication sources and destinations must support hard links across their
  locations; failure occurs before stop. No concurrent privileged administrator
  or runtime installer is allowed during the window. File identity checks are
  not transactions against a privileged process racing a systemd request.
- This window survives controller, Pod transport and Node Agent loss, not a
  reboot. A changed boot identity is a refusal, not a new recovery opportunity.
  Persistent unit files are not boot-enabled and must be reconciled explicitly
  after such an interruption.
- Hardware loss, uninterruptible processes, expired authority, replaced files
  and unknown invocation ownership may require operator reconciliation. The
  helper never broadens service scope, enables automatic recovery, resets a GPU,
  reboots, changes scheduling, restarts workloads or calls AWS APIs.
- The acceptance resource inventory is the host publication journal plus the
  existing HostProbe controller journal. These are not new production Node
  Runtime installer resources.

Implementation: `warm_spare_node_probe.ServiceWindow`,
`destr008_service_window.ServiceWindowController`, and
`warm_spare_fixture.WarmSpareServiceFixture` under `scripts/e2e/regional/`.
Focused `test_destr008_service_*` tests use fake systemd, regular temporary files
and an owned child process for crash/lock recovery. No host, Kubernetes, AWS,
GPU, reboot, reset or LIVE validation is implied.
