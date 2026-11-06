# Xid Linkage and Drain P2P E2E Tests

Chinese source: [test specification](../../components/xid154-recovery-e2e.md).

These cases address risk 1, interpreting Xid 154 linkage as minimum versions for
the original Xid, and risk 3, the incomplete Drain P2P recovery chain.

## Scope and Execution

The software E2E uses the real event API, policy, correlation coordinator,
workflow builder, application-wired production executor and escalation,
NodeActionWorkflowAdapter, signed commands, NodeActionExecutor, quiesce manager,
and temporary SQLite control-plane/agent ledgers.

Scheduler, workload, provider, validation, node transport and host command I/O
are scripted boundaries. No subprocess, real GPU or cluster mutation runs.
The suite does not validate collectors, regional queues/TLS, Kubernetes,
DCGM health, persistence daemon RPC, actual reset/reboot or checkpoint recovery.

```bash
.venv/bin/python -m pytest tests/execution/test_xid_recovery_e2e.py -n 0 -q -o addopts=''
```

The six automated catalog entries cover 16 variants. The UVM gap has a separate
manual BLOCKED entry, not a passing skip or xfail. Case definitions and independent
requirements live in `testcases/fault-scenarios.yaml` and
`testcases/scenario-requirements.yaml`.

## Cases

| ID | Input and procedure | Required outcome |
|---|---|---|
| GF-E2E-XID-LINKAGE-001 | H100 Xid 95 with R550/CUDA 12.4, missing versions, or R575/CUDA 12.9; Xid 48 with R550/CUDA 12.4 | EXECUTABLE/RESET_GPU; stop and release before one target-only reset; restart once and incident RECOVERED |
| GF-E2E-DRAIN-P2P-001 | Raw Xid 154 Drain P2P log; fresh XML reports None, Reset, Drain and Reset, or Drain P2P | Stop must finish before node actions; release before query; None skips reset, others reset once; validation precedes readmission; duplicate event/command cannot repeat effects |
| GF-E2E-DRAIN-P2P-002 | Raw Drain P2P log, driver readback Reboot | Real result propagation and escalation reach one simulated node reboot, no GPU reset, then validation and workload restart |
| GF-E2E-DRAIN-P2P-003 | N/A, missing target, persistence stuck Enabled, query timeout, or reset timeout | No scheduling/workload readmission or automatic reboot; no reset except the single timed-out reset; unknown outcome remains BLOCKED and cannot be resubmitted |
| GF-E2E-DRAIN-P2P-004 | IDLE node, readback None | Release/query succeeds without stopping or restarting an invented workload |
| GF-E2E-DRAIN-P2P-005 | Remove RECOVER_DRAIN_P2P from the agent allowlist | Recovery cannot succeed or fall back to reset/reboot/workload restart |

Xid 154 tests start with its raw log, not a trusted client action field. They
observe PENDING_CORRELATION with no workflow/command, advance the coordinator's
test clock, run its public worker method, and read the finalized decision through
the API again. The negative cases must reach RECOVER_DRAIN_P2P, not fail at an
unrelated prerequisite.

The success matrix also checks that the other GPU keeps persistence enabled,
Fabric Manager is neither stopped nor restarted, target persistence is restored,
and the real agent ledger answers command replay without new host calls.
The compatibility case proves allowlist refusal, not actual old-agent deployment.

For a terminal reset-timeout response, existing failure compensation permits
service restoration while keeping the workflow BLOCKED and the node unschedulable.
Services being active is neither proof of hardware recovery nor permission to
restart training. An in-flight command is a different case.

The old GF-POL-011 expectation is corrected. GF-REGIONAL-COLLECT-007 changes its
definition and selector, so its old PASS binding is no longer applicable and is
reset to NOT_RUN. Historical digests are not re-signed as new evidence. Product
applicability and genuine NVLink decoder version requirements remain covered by
the existing policy tests.

## Live Acceptance Prerequisites

Live acceptance of the full chain is specified but not run, and no dedicated
automatic destructive runner exists for it. GF-GENERAL-LIVE-024's
`xid_twin_drain_p2p` variant (`scripts/e2e/regional/general_live_xid_twins.py`,
separately approved as `GPU_DRAIN_P2P`) moves DCGM field 230 to 154 on an idle
node and then writes one reviewed XID 154 Drain P2P kernel line, expecting
exactly one executed RECOVER_DRAIN_P2P, a RECOVERED incident and no planned
QUARANTINE. It exercises the cross-source pairing and the chain on a real node
while the driver still decides the actual GPU Recovery Action, so it covers
neither the Reset/Reboot branches below nor training recovery. Use isolated nodes, a disposable training job and an approved maintenance
window. Record workload/attempt identity, GPU UUID/PCI, driver/CUDA, boot ID,
persistence, service state and scheduling state first. Prove independent bounded
recovery and operator takeover before authorizing stop/reset/reboot.

Do not downgrade production drivers for the older-version case. Missing matching
hardware or an independently controllable driver recovery state means NOT_RUN.
Injecting an Xid log does not change the actual driver GPU Recovery Action; fake
XML is software-test evidence only.

Correlate raw events, correlation records, workflow steps, signed command IDs,
agent ledger records, original nvidia-smi output, actual host/provider calls and
new workload attempts. Verify order, target scope, exactly-once effects,
GPU/fabric health, service/persistence restoration, checkpoint identity and
continued training progress. On uncertainty preserve isolation and confirmation;
do not uncordon, restart training or erase ledgers merely to clean up the test.

## UVM Persistence Gap

GF-E2E-DRAIN-P2P-UVM is BLOCKED: independent UVM persistence detection/release is
not implemented. Ordinary persistence Disabled or an empty temporary proc tree
does not prove UVM release. A future acceptance case must include ordinary
persistence disabled while UVM persistence remains enabled, and require verified
release or an explicit refusal to proceed. This is an implementation gap, not
only missing live measurement.
