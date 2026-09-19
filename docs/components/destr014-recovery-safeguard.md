# DESTR-014 Reboot-Surviving Recovery

This is an acceptance-only safeguard, not a production recovery operation or
permission to run DESTR-014. The case still requires the approved physical plan,
target identity, maintenance window, EKS HyperPod `NodeRecovery=None`, no warm
spares, and provider replacement disabled. Local tests are not LIVE evidence.

## Protocol

1. The runner saves a private, locked controller journal before configuration or
   workload changes. It binds the attempt, approved plan, release and target.
2. After baseline capture, it journals a nonrenewable host recovery binding:
   run owner, cluster, Node UID, original boot ID, release, Agent artifact/bundle
   and Profile pins, helper digest, restoration time and expiry.
3. The host probe takes a node-local filesystem lock and an exclusive persistent
   claim. It verifies the installed Agent environment, runtime slot, unit files,
   interpreter, machine ID and hardware identity. Only the standard installer's
   single absolute `multi-user.target.wants` Agent link is supported.
4. It retains the original Agent enable-link inode and publishes run-owned unit
   files using create-only hard links. Sources and ownership receipts are
   persisted before publication. It installs a copied stdlib-only helper under
   `/var/lib/gpu-fault-acceptance/destr014-recovery/<binding-digest>/`, a persistent
   service/timer under `/etc/systemd/system/`, and a timer boot link. The timer
   uses both `OnBootSec` and `OnCalendar`, with `Persistent=true`; it does not
   depend on a transient `systemd-run` unit, the probe Pod, the controller, or
   the Agent's virtual environment.
5. Only an invocation in the installed recovery service's cgroup, with matching
   systemd `InvocationID`, can acknowledge arming. A separate controller read
   observes this ACK. Both controller and host require a fresh original-boot ACK
   before disabling Agent boot activation. The host rechecks the clock and
   identity after verification.
6. Disable removes only the recorded Agent enable-link inode and reloads
   systemd. It never stops the running Agent. The host records disable intent
   first, then verifies `disabled` and still `active` before acknowledging.
   The runner cannot arm the GPU holder or inject either software fault until
   this ACK is accepted.
7. At the fixed restoration time, the independent helper restores the original
   enable link with create-only linking, and starts the same Agent if necessary.
   An owned drop-in bounds the systemd start job to 45 seconds and startup to
   30 seconds. There is at most one start submission, including lost-ACK cases.
   No reset, reboot, replacement, scheduling change or workload restart is
   available to this helper.
8. Restoration seals the host state against late disable requests. Normal
   controller restoration uses the same protocol. Automatic recovery before
   scenario completion makes the case fail, even if Agent health later returns.

## Cleanup And Resumption

The host journal is atomically replaced with file and directory fsync. The
controller uses the repository's atomic JSON writer plus directory fsync.
Mutating requests are recorded before submission, including requests whose
responses may be lost. A missing arm response never authorizes injection.

Reexecuting an attempt with a controller journal is cleanup-only. It validates
the original release and Node UIDs, reconstructs fixture ownership from durable
journals, and does not reopen windows, resubmit the workload, rearm or reinject.
Missing incident/quiescence evidence defers workload and environment cleanup.
Loss of command supervision is durably recorded and forbids further commands,
including cleanup from a fresh process; independent host recovery remains armed.
Such runs require separate operator review, not deletion of the refusal marker.
An unfinished journal from an earlier attempt whose recorded boot ID the node
has since replaced owns no host state on the current boot -- the installer
recreated the Agent and the probe's new-boot table already retired that boot's
state -- so it is archived aside (`<stem>.rebooted-<run id>.json`) and the new
attempt starts fresh, recording the retired journal's lineage under the new
journal's `run.retired_journals`. A journal on the same boot, or one that lost
supervision, still refuses.

Cleanup verifies the whole persistent resource ownership set before deleting
anything. It stops the timer and proves the independent service has stopped and
has no pending job. The Agent start job must also be gone before its timeout
drop-in can be removed. Only matching original inodes/content are removed. A
recreated enable link, runtime file, claim, service, timer or drop-in is never
overwritten or adopted. An independent service cannot attest its own exit:
it records restoration and retires its timer, while controller cleanup supplies
the final completion proof. Interrupted retirement is resumable.
Reports identify all nonclosed states as `UNFINISHED_RECOVERY`; only a verified
`CLOSED` record is classified as `FORENSIC_TOMBSTONE`.

The controller records `CLOSED` only after host restoration, actionable unit
removal, owned probe cleanup, and the other case cleanup checks succeed.
The final case PASS is written only afterwards. A cleanup-only replay does not
produce another PASS. A minimal persistent host tombstone and private source
receipts remain to reject late requests; they are audit/fencing state, not an
enabled service. Absence of a journal beside residual files is not cleanup proof.

The acceptance resource inventory is the host journal's `resources` list plus
the existing HostProbeFixture Pod/ConfigMap journal. These resources are created
and removed by this case, not added to the production Node Runtime installer.
The legacy unbound `disable-agent-restart` and `restore-agent` probe entries
refuse Agent changes. Older baseline-only records require manual reconciliation.

## Bounds And Assumptions

- Host storage and `/etc` must survive reboot and support hard links between the
  recorded sources and destinations. Unsupported layouts fail before disable.
- Systemd, the host interpreter, local filesystem integrity and UTC clock are
  trusted. No concurrent administrator/runtime installation is allowed during
  the drill. Filesystem identity checks detect drift; they are not transactions
  against a privileged administrator racing an unlink.
- The hold is derived from the case duration estimate plus 120 seconds; recovery
  gets a fixed additional 180 seconds. The complete window must fit the approved
  maintenance deadline and the host's 7200-second maximum. It cannot be renewed.
- An unavailable host cannot be physically recovered by local software.
  On return before expiry, persistent scheduling resumes recovery. On return
  after expiry, the helper records `EXPIRED`, retires its owned timer and does
  not submit late Agent actions. Manual recovery and reconciliation are required.
- Runtime/host identity drift, an unrelated pending Agent job, a failed bounded
  start, missing supervision, or replaced resources prevent successful cleanup.
  The helper never compensates by resetting hardware or replacing the node.
- The independent helper restores only Agent service availability. Workflow
  quiescence, isolation, workload deletion and CPU/GPU environment restoration
  remain the existing guarded runner/fixture responsibilities.

## Hermetic Verification

The focused `test_destr014_recovery*` tests execute the real journal/state machine
against fake systemd/host I/O and temporary regular files. Symbolic-link semantics
are simulated; no real systemd, namespace, AWS, Kubernetes or GPU commands run.
Owned subprocess tests exercise process death and fresh-process journal recovery.
Negative controls cover stale/forged ACKs, reboot and runner loss, clock expiry,
lost create/disable/restore/cleanup ACKs, foreign owners, partial publication,
unknown journals, stuck jobs, supervision loss, cleanup residuals and premature
PASS. These tests do not claim a real HyperPod reboot has been validated.

Implementation: `destr014_recovery_probe.Recovery`,
`destr014_recovery.AgentRecoveryWindow`, `destr014_recovery.RunJournal`, and
`run_destr014_branch_exhaustion.execute_case` under `scripts/e2e/regional/`.
