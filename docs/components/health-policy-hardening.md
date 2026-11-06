# Twelve General Health Policy Changes

This batch implements the twelve rows previously marked pending in the
[general-fault audit](general-fault-e2e-audit.md): 016, 017, 018, 019, 020,
022, 023, 024, 025, 027, 030 and 031. It changes detection and workflow
semantics, not merely the acceptance documentation.

It does not complete the other twelve partially addressed categories, nor
prove a physical GPU, provider or NCCL recovery. The full manual cases keep
their `NOT_RUN` status. New `GF-GENERAL-FIX-*` cases describe software proof
separately.

## Implemented Behavior

Temperature, power and throttling were subsequently revised against NVIDIA
and NVSentinel sources; see [environment policy review](gpu-environment-policy-review.md).

| Row | Change |
|---|---|
| 016 | Trusted kernel MCE/storage/RDMA/lockup findings isolate the node and stop its freshly attributed managed attempt before quarantine. Unknown attribution permits containment but does not invent a job to stop. Ordinary journal text is no longer kernel authority; corrected MCE goes to diagnostics, graded from the kernel's own header (IA32_MCi_STATUS.UC clear, `mce_grading`) and words, with the record's detail lines producing no finding of their own. |
| 017 | Journal PID/transport/cgroup/container metadata is retained. The control plane matches PID against observed containers and distinguishes live, exited and unresolved attribution. NCCL errors use bounded triage; application log diagnostics end with operator review instead of claiming that healthy fabric repaired the application. |
| 018 | Sustained critical temperature requires three valid observations, except an absolute shutdown-temperature breach. New workflows stop attributed work, quarantine, collect evidence and request operator review. Idle cooling does not authorize automatic readmission, reset or RMA. |
| 019 | Volatile DBE stops work and quarantines for analysis, not automatic RMA. SBE/aggregate/retirement increments request diagnostics. Modern row-remap pending and supported legacy page-retirement pending select service-window reset; unknown architecture requests review. Confirmed row-remap failure retains field-diagnostic/RMA-review handling on supported products. |
| 020 | NVLink counters alone, including multi-GPU counter correlation, request diagnostics rather than permanent DRAIN/RMA. Independently received Xid/SXid faults retain their catalog recovery actions. |
| 022 | Counter deltas are invalid across reset, boot, GPU identity or metric-source changes. Negative/ambiguous data cannot clear an existing fault as a healthy zero delta. PCIe/Xid joins keep the existing trusted-device/time-window gates. |
| 023 | Duration alone remains diagnostic with the existing noise/duty-cycle bounds. Normal power limiting is not thermal failure. HW Power Brake (0x80) is distinct from SW cap (0x04), requires sustained confirmation, and isolates the node under `ENVIRONMENT_REVIEW` for power-delivery review (official action `CONTACT_SUPPORT`) without reset or reboot. |
| 024 | Composite inputs cannot come from the future or a different metric source. Reusing the same component findings does not advance the consecutive counter. Correctable-only correlation does not authorize DRAIN; memory composites preserve the strongest supported component action and its official procedure. |
| 025 | Empty, non-finite, negative or duplicate metric slots become incomplete observations; valid independent samples remain usable. Sustained collection errors request a bounded health snapshot and require a new successful observation, rather than running GPU stress diagnostics or escalating hardware recovery. |
| 027 | Unallocated, terminated, stale-attribution or identity-mismatched progress cannot stop a job. NaN stops only the attributed attempt and requests evidence/operator review; an unvalidated checkpoint is never automatically selected. Reporter silence is distinguished from a fresh heartbeat with stalled steps. Phase hints are bounded and stragglers are compared only inside an explicit progress group. |
| 030 | NotReady, provider state, HMA labels, generic warnings and missing telemetry are not sufficient reboot evidence. Hardware escalation requires an explicit fresh hardware finding or executable trusted hardware marker. Missing inventory/metrics makes the spare ineligible without manufacturing a reboot. Local and regional health paths use the same classification. |
| 031 | The silence scanner records `collector_observation_valid` with `HEALTHY`/`UNKNOWN` and a reason, and creates at most one snapshot-refresh workflow per node/hour, for silent channels only: a channel that delivered no batch, clean or erroring, within its threshold. A collector that keeps posting batches naming its collection errors (a hung `nvidia-smi`'s timeout/breaker batches) stays UNKNOWN but is not refreshed: the snapshot restarts the metrics and host collectors, which cannot clear an error the source keeps raising and would reset the collector's breaker (GF-REGIONAL-COLLECT-019). It respects active membership, valid Agent lease, disabled collectors and Profile ownership. Actual monitoring restart remains the existing systemd policy or operator action; snapshot authority does not grant permission to restart arbitrary services. |

## Safety Boundaries

Newly compiled steps carry `health_recovery_contract`. The validation adapter,
branch escalation and hardware escalation classifier consume the same intent.
The guard applies even when validation times out and its final error has no
special details. Existing workflows without this metadata retain their original
contract; this change does not rewrite in-flight plans.

All original capability, owner, allowlist, workload-state, signing, generation,
restart-budget and terminal-isolation constraints remain in force. Field
diagnostics marked OBSERVE are not enabled by this batch.

New environmental workflows use `ENVIRONMENT_REVIEW`: containment remains until
operator review, even after fresh cool readings. Old persisted `THERMAL_COOLDOWN`
workflows are not rewritten; inventory and review them before rollout. The old
300-second cooldown is not proof of repaired cooling under load.

The NVLink and corrected-memory changes do not weaken independent Xid/SXid
handling. These telemetry counters do not by themselves distinguish
recoverable link noise from a physically failed component. An independently
classified hardware fault still has its own policy/incident path.

## Observation State

`observation_reason` is shared by validation, spare admission and the silence
scanner. Missing success, stale/future success, a current collection error or
missing post-action evidence is UNKNOWN. A recent success preceding a newer
error is not healthy.

The scanner publishes the derived state as an existing host-telemetry latest
sample (`collector_observation_valid`, device = collector kind). This is not a
new global Prometheus family, channel, endpoint or Store kind. Its timestamp is
the scan time; normal scanner cadence can delay the displayed transition.
Safety consumers evaluate the canonical collector status directly, without
waiting for that cached display sample. Late ingestion therefore cannot
overwrite current health by publishing a newly timestamped old status.

Refresh means `TRIGGER_HEALTH_SNAPSHOT` followed by read-only validation. It does
not mean "restart the GPU" or "restart all monitoring services". A stopped
collector that does not respond remains UNKNOWN and requires its existing
supervisor or operator to restore it. A GPU observation with zero samples is
not accepted as a successful recovery.

## Training Protocol

No heartbeat schema change is required. Optional existing `labels` carry:

- `phase`: `TRAINING`, `INITIALIZING`, `CHECKPOINT_LOAD`, `CHECKPOINT_SAVE` or
  `RESTORING`. Unknown values do not suppress hang detection.
- `progress_group`: the application-defined group whose steps and throughput
  are comparable. Missing/different groups are not compared as stragglers.

Initialization can defer stalled-step detection for at most 300 seconds;
checkpoint/restore phases for at most 1,800 seconds. The phase window is durable.
Changing between pause phases does not restart it, and the step-progress anchor
also caps the deferral. A missing heartbeat is not a valid phase-progress proof.
The training application must actually report these labels; GFRS does not infer
checkpoint phases from low RDMA traffic.

A declared restore with a checkpoint reference may move the step backwards
inside its bounded phase. That reference is evidence only: this batch does not
download, execute or trust its contents. NaN recovery requires application
validation of a checkpoint and the normal authorized restart path; it does not
blindly retry a numerical failure.

Training validation requires newer, finite application progress from the
original attempt/rank/Pod. Healthy GPU or network metrics alone cannot satisfy
it. Logs without enough process attribution still produce evidence/review,
not a fabricated suspicious rank.

## Compatibility and Verification

Roll out matching control-plane, executor and node-runtime artifacts before
relying on these contracts. No new operation or broad service-mutation
permission is introduced. The metric policy and correlation policy versions
change with these semantics; historical decisions retain their recorded
version.

Software cases exercise the real API, policies, SQLite, workflow executor and
validation adapter. Scheduler node changes use the actual Kubernetes adapter
with a local client. Workload/provider/device commands and diagnostic payloads
are controlled I/O boundaries, not real hardware. Separate boundary tests cover
counter discontinuities, phase budgets, source attribution, duplicate samples,
stale inputs and repeated composite evidence.

```bash
.venv/bin/python -m pytest \
  tests/execution/test_general_health_policy_e2e.py \
  tests/execution/test_observation_recovery_e2e.py \
  tests/metrics/test_general_health_policy_boundaries.py \
  tests/completion/test_training_phase_recovery.py -q -o addopts=''
```

The full manual acceptance matrix still needs an isolated, approved hardware
window. No deployment, real GPU/NIC operation, Kubernetes mutation or cloud
recovery was performed by the local implementation tests.
