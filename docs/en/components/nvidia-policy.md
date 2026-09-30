# NVIDIA Official Policy Implementation Audit

English edition of `docs/components/nvidia-policy.md`; the Chinese file remains the source of record until both are maintained together.

## Pinned Upstream

- Catalog: NVIDIA Xid Catalog 610
- Official address:
  `https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html`
- XLSX SHA-256:
  `7f70ce9684be0c9d98a367770341ccc1d8bca4465cebe42dab7e23d57df5d5a5`
- Generated artifact canonical SHA-256:
  `ffb82509abb574577db3c5d759cec94df40ea25edaf4691e7bd4ba0a12ba0462`
- Fetch and generation date: 2026-07-20

The generated artifact
`src/gpu_fault/data/nvidia-xid-catalog-610.generated.yaml`
contains 172 XID entries, 95 XID 144-150 decode entries and 32 Resolution Buckets.

Regenerate:

```bash
python tools/generate_nvidia_xid_policy.py \
  Xid-Catalog.xlsx \
  src/gpu_fault/data/nvidia-xid-catalog-610.generated.yaml \
  --catalog-version 610 \
  --expected-sha256 \
  7f70ce9684be0c9d98a367770341ccc1d8bca4465cebe42dab7e23d57df5d5a5
```

When the digest does not match, generation fails outright; silently accepting an upstream change is forbidden.
Even when `--expected-sha256` is omitted, it defaults to the pinned XLSX digest above.
Integer cells reject boolean values, so `TRUE`/`FALSE` cannot be taken as an XID number; integers, integral floats and integer text
keep the original normalisation behaviour. The read-only workbook is closed both after success and after a parse failure, so repeated generation does not accumulate file handles.

Resident gate that needs no XLSX:

```bash
PYTHONPATH=src python tools/generate_nvidia_xid_policy.py --check
```

This gate checks the canonical format, the upstream digest, the 172/95/32 counts and
`metadata.generatedSha256`. The runtime loader recomputes the digest of the generated artifact;
`mapping_version` uses the first 16 characters of the generated artifact's digest, so a hand-edited rule fails to load outright,
or, after the digest is explicitly updated, triggers a deployment pin mismatch.

The upstream XLSX does not enter the source tree. Before regenerating, store the verified file under
`artifacts/upstream/nvidia/xid-catalog-610/<source-sha256>/` and archive that directory to
controlled object storage; `artifacts/` is only a local workspace.

## Decision Semantics

The policy results are stored separately:

- `official_action`: the original NVIDIA Immediate Action/workflow name.
- `investigatory_action`: the original NVIDIA Investigatory Action.
- `action`: the execution action the current control plane has approved and can express.
- `safety_action`: the site's fail-closed action when the official workflow is blocked because evidence or the executor is
  incomplete; currently only `QUARANTINE` is allowed.
- `source`: `NVIDIA_CATALOG`, `NVIDIA_XID_154`,
  `NVIDIA_FABRIC_MANAGER` or `SITE_SAFETY`.
- `disposition`: executable, monitor only, blocked on missing evidence, blocked on missing workflow, or not applicable.

Only the following direct mappings enter a recovery plan automatically:

| NVIDIA Immediate Action | Control-plane action |
|---|---|
| `IGNORE` | `NO_ACTION` |
| `RESTART_APP` | `RESTART_WORKLOAD` |
| `RESET_GPU` | `RESET_GPU` |
| `RESTART_BM` | `REBOOT_NODE` |

In the `hyperpod-eks` and `hyperpod-slurm` runtime profiles,
`RESTART_VM` means rebooting the current HyperPod node; the runtime compiler keeps
`official_action=RESTART_VM`, sets `effective_action=REBOOT_NODE`, and
`BatchRebootClusterNodes` executes it. This mapping does not mean an EC2 stop/start; the host placement, local storage and identity semantics of stop/start and
node replacement remain independent actions.

`CONTACT_SUPPORT`, `CHECK_MECHANICALS` and `UPDATE_SWFW` are not crudely replaced with another
"official action": the policy keeps the original action and returns EXECUTABLE, and the workflow compiler lands them respectively as
`ESCALATE_SUPPORT` (fixed-template mail, no isolation), `CHECK_MECHANICALS` (notify and wait for manual confirmation)
and a downtime chain containing `UPDATE_SOFTWARE_FIRMWARE`. Other workflow names with no registered resolver keep
the original action, return `BLOCKED_WORKFLOW`, and block further scheduling through the independent `safety_action=QUARANTINE`.
The per-XID verdicts and steps are in
[Fault Categories and Actions](../fault-categories-and-actions.md).

Unknown XIDs use the `SITE_SAFETY` source and `safety_action=QUARANTINE`, stating explicitly that this is this system's fail-closed
safety policy, not an NVIDIA Catalog recommendation.

The Catalog's `A100/H100/B100/GB200` columns are interpreted by product family. A concrete SKU is normalised as
`A* -> A100`, `H*/GH* -> H100`, `B* -> B100`, `GB* -> GB200`, and the applicability gate then runs against the families
each XID allows. For example, H200/H800 belong to the H100 column, B200 belongs to the B100 column, and
GB300 belongs to the GB200 column.

## Implemented Official Workflows

- XID 45: first persisted as `PENDING_CORRELATION`, waiting for the 30-second window to close; when accompanied by other XIDs
  it reuses the incident/workflow of the most severe accompanying XID; when solo it keeps `RESTART_FM` and compiles it to
  `RESTART_FABRIC_MANAGER`, executed by the Node Agent's `fabricManagerRestart` capability. The shared
  Store lease supports HA, out-of-order arrival and Pod restarts.
- XID 48: solo executes `RESET_GPU`; when accompanied by XID 63/64 it executes
  `DRAIN_AND_RESET`.
- XID 94/95: application/all-applications containment are stored separately; XID 95 requires the affected workload to be stopped before
  the reset.
- XID 154: the driver-reported action takes precedence and is mapped as-is.
- XID 159 `CHECK_UVM`: reset when UVM/vGPU use is explicitly confirmed, otherwise ignore; blocked when evidence is missing.
- XID 144-150: selects the V1/V2 `IntrInfo` pattern by the driver R575 boundary, matching
  Error Status and Action 2 at the same time. The primary action and Action 2 may use different bit patterns and are matched separately;
  an RXPIPE secondary-pattern hit must still satisfy the Error Status of the same table entry, and cannot first require the primary pattern to hit as well.
  When no official table entry matches, it blocks and does not guess an action.

When the correlated raw record reaches its retention limit and the same event is re-delivered, the surviving final decision and incident binding must be reused.
A purged raw record does not mean the event was never handled; an uncorrelated cached policy result must not overwrite the final decision;
records still in `PENDING_CORRELATION` continue through the original window repair flow.

## SXID

SXID recovery follows the Fabric Manager User Guide:

- non-fatal: informational, keep monitoring.
- fatal access link: the affected GPU and the workload participating GPUs must already be resolved,
  then the job is stopped and the whole GPU group is reset.
- fatal trunk and 10003/19084: require fabric_partition and the complete node GPU inventory; when both are present
  `RESET_ALL_GPUS_AND_NVSWITCHES` is executed (Node Agent `fabricReset` capability, multi-node barrier);
  if either is missing the result is `BLOCKED_MISSING_EVIDENCE`, and it never degrades to a single-GPU reset.
- always-fatal (the 20 codes pinned by Table 23): executes `REBOOT_NODE`, cordoning and stopping the job beforehand;
  an event that declares Always-Fatal but is not in the table blocks.
- B200/B300: the traditional fatal/non-fatal SXIDs do not apply; DCGM/NVSDM telemetry is required.

The SXID `classification_source` must be `NVIDIA_FABRIC_MANAGER`; otherwise it blocks because the evidence source
is untrusted.
