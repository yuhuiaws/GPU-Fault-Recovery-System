# NVIDIA Official Policy Implementation Audit

English edition of `docs/components/nvidia-policy.md`; the Chinese file remains the source of record until both are maintained together.

## Pinned Upstream

- Catalog: NVIDIA Xid Catalog 610
- Official address:
  `https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html`
- XLSX SHA-256:
  `7f70ce9684be0c9d98a367770341ccc1d8bca4465cebe42dab7e23d57df5d5a5`
- Generated artifact canonical SHA-256:
  `ffb82509abb574577db3c5d759cec94df40ea25edaf4691e7bd4ba0a12ba0462` (the artifact's `metadata.generatedSha256`)

The generated artifact
`src/gpu_fault/data/nvidia-xid-catalog-610.generated.yaml`
contains 172 XID entries, 95 XID 144-150 decode entries and 32 Resolution Buckets.

Regenerate:

```bash
PYTHONPATH=src python tools/generate_nvidia_xid_policy.py \
  Xid-Catalog.xlsx \
  src/gpu_fault/data/nvidia-xid-catalog-610.generated.yaml \
  --catalog-version 610 \
  --expected-sha256 \
  7f70ce9684be0c9d98a367770341ccc1d8bca4465cebe42dab7e23d57df5d5a5
```

When the digest does not match, generation fails outright; silently accepting an upstream change is forbidden.
Even when `--expected-sha256` is omitted, it defaults to the pinned XLSX digest above; an omitted output path writes the
generated artifact above.
Integer cells reject boolean values, so `TRUE`/`FALSE` cannot be taken as an XID number; integers, integral floats and integer text
keep the original normalisation behaviour. The read-only workbook is closed both after success and after a parse failure, so repeated generation does not accumulate file handles.

Resident gate that needs no XLSX:

```bash
PYTHONPATH=src python tools/generate_nvidia_xid_policy.py --check
```

This gate checks the canonical format, the upstream digest, the 172/95/32 counts and
`metadata.generatedSha256` (`--generated-policy` points it at another artifact). The runtime loader recomputes the digest of the generated artifact;
`mapping_version` is `nvidia-xid-catalog-610/sha256:<first 16 characters of the generated artifact's digest>`, so a hand-edited rule fails to load outright,
or, after the digest is explicitly updated, triggers a deployment pin mismatch.

The upstream XLSX does not enter the source tree. Before regenerating, store the verified file under
`artifacts/upstream/nvidia/xid-catalog-610/<source-sha256>/` and archive that directory to
controlled object storage; `artifacts/` is only a local workspace.

### DCGM Exporter

- Image: `nvcr.io/nvidia/k8s/dcgm-exporter:4.6.1-4.8.4-distroless` (DCGM 4.6.1 client library, exporter 4.8.4, distroless
  variant; the release uses go-dcgm `v1.4611.2`)
- Manifest list SHA-256:
  `sha256:148b0c025e5f2850256816fa33754fbf4733b7a086697a95613fc0ba3db5b003` (`4.6.1-4.8.4`, `4.6.1-4.8.4-distroless`,
  `4.8.4` and `latest` all name it)
- linux/amd64 manifest SHA-256:
  `sha256:804c17f67ab63e8a4dd5301546beefaa104129f833dae4a68f5ff7c4819f471b`
- Image config: `com.nvidia.dcgm.version=4.6.1`, `org.opencontainers.image.version=4.8.4`, base
  `nvcr.io/nvidia/distroless/cc:v4.1.4`; the layers ship `libdcgm.so.4.6.1` and `/etc/ld.so.cache`
- Where the product holds it: the one source `src/gpu_fault/admin/dcgm_exporter_release.py`; the `dcgm_exporter` entry of
  `config/release-images.lock.json` (`reference` the manifest-list digest, `source` the tag); the plain EKS host-mode host
  DCGM `4.6.1-1` (`DCGM_HOST_PACKAGE_VERSION`) is derived from it; the rule is in `src/gpu_fault/dcgm_versions.py`

NVIDIA's rules, verbatim (from the page version "Last updated on 2026-09-10", the text the product and its tests pin):

- "Install DCGM Exporter", "Requirements and Version Compatibility"
  (<https://docs.nvidia.com/datacenter/dcgm/latest/installation/install-dcgm-exporter.html#requirements-and-version-compatibility>):
  - "DCGM Exporter releases pair a DCGM version with an exporter version. Container image tags encode the tested pair as `<dcgm-version>-<dcgm-exporter-version>-<image-variant>`."
  - "Run DCGM Exporter with the DCGM version paired with that exporter release. Mismatched combinations might function, but they are not tested or supported. When troubleshooting, first update to matching, current versions of DCGM and DCGM Exporter."
  - "For a separate DCGM host engine, the exporter’s DCGM client library version must also be greater than or equal to the DCGM host engine version. Meeting this connection requirement does not make a mismatched release pair supported."
  - Troubleshooting, same page: "For a separate DCGM host engine, verify the endpoint, transport, socket mount, and client/host-engine version relationship."
- Same page, "Connect to a DCGM Host Engine" (<https://docs.nvidia.com/datacenter/dcgm/latest/installation/install-dcgm-exporter.html#connect-to-a-dcgm-host-engine>):
  "The URI forms require DCGM 4.5 or later." (the product keeps the `host:port` form `127.0.0.1:5555`)
- README (<https://github.com/NVIDIA/dcgm-exporter>, "Version compatibility"): "Run DCGM Exporter with the DCGM version paired with that exporter release."
- GitHub release (<https://github.com/NVIDIA/dcgm-exporter/releases/tag/4.8.4>): "Release DCGM Exporter 4.8.4 with DCGM 4.6.1 and go-dcgm v1.4611.2."
- DCGM release notes (<https://docs.nvidia.com/datacenter/dcgm/latest/release-notes/changelog.html>), 4.6.1,
  Bug Fixes, Core: "Add safeguards against potential memory errors when using mismatched clients and hostengines."

Why this release: it is the newest exporter release; DCGM 4.7.0 packages are published, but no exporter paired with DCGM
4.7.x exists on the registry or on GitHub (go-dcgm's `v1.4701.*` bindings exist, no exporter release uses them).
`4.6.0-4.8.3-distroless`, the release paired with DCGM 4.6.0, ships no `/etc/ld.so.cache`, and its
start-up check for `libdcgm.so.4` then exits with "exit status 1" unless the container runtime's NVIDIA hook writes one
(github.com/NVIDIA/dcgm-exporter issue #612, fixed by PR #724 in 4.8.4). Since 4.8.2 NVIDIA publishes the distroless variant
only. The pairing per cluster (plain EKS host mode is a supported pair; the HyperPod engine is the AMI's, whose version the AMI
decides, the HyperPod AMI's own engine (currently DCGM 4.7.0) newer than the 4.6.1 client, the connection requirement broken with no product
remedy, advised about until NVIDIA publishes an exporter on DCGM 4.7.0 or later), why the product advises rather than refuses,
and the upgrade steps are in the [Deployment and Operations Manual](../deployment-and-operations-manual.md) §6.5, "DCGM
exporter version and the host hostengine pairing".

Re-verify (NGC registry v2 API, anonymous pull token):

```bash
token="$(curl -s 'https://nvcr.io/proxy_auth?scope=repository:nvidia/k8s/dcgm-exporter:pull' \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["token"])')"
registry=https://nvcr.io/v2/nvidia/k8s/dcgm-exporter
# manifest-list digest of each tag: one value for 4.6.1-4.8.4, 4.6.1-4.8.4-distroless, 4.8.4 and latest
for tag in 4.6.1-4.8.4 4.6.1-4.8.4-distroless 4.8.4 latest; do
  curl -sI -H "Authorization: Bearer ${token}" \
    -H 'Accept: application/vnd.docker.distribution.manifest.list.v2+json' \
    "${registry}/manifests/${tag}" | grep -i docker-content-digest
done
# the linux/amd64 manifest of the list, then its image config labels
curl -s -H "Authorization: Bearer ${token}" \
  -H 'Accept: application/vnd.docker.distribution.manifest.list.v2+json' \
  "${registry}/manifests/4.6.1-4.8.4-distroless" \
  | python3 -c 'import json, sys; print([m["digest"] for m in json.load(sys.stdin)["manifests"] if m["platform"]["architecture"] == "amd64"][0])'
config="$(curl -s -H "Authorization: Bearer ${token}" \
  -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
  "${registry}/manifests/sha256:804c17f67ab63e8a4dd5301546beefaa104129f833dae4a68f5ff7c4819f471b" \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["config"]["digest"])')"
curl -sL -H "Authorization: Bearer ${token}" "${registry}/blobs/${config}" \
  | python3 -c 'import json, sys; labels = json.load(sys.stdin)["config"]["Labels"]; print({key: labels[key] for key in ("com.nvidia.dcgm.version", "org.opencontainers.image.version", "org.opencontainers.image.base.name")})'
```

A release change follows the upgrade steps of the deployment and operations manual §6.5: the source, the literals and the
image lock above change together (`tests/regional/test_dcgm_exporter_release_pins.py`), and the counters field table is read
again from `pkg/dcgm/const_fields.go` of the new release's go-dcgm (its `go.mod`) (`PINNED_FIELD_IDS` in
`tests/regional/test_dataplane_dcgm_counters.py`).

## Decision Semantics

The policy results are stored separately:

- `official_action`: the original NVIDIA Immediate Action/workflow name; for an SXID the branch this system picked from the
  Fabric Manager table (an Always-Fatal `RESTART_HOST_OR_SBR` is recorded as `RESTART_BM`, scope-dependent codes as
  `RESET_PARTICIPATING_GPUS`, `RESET_ALL_GPUS_AND_NVSWITCHES` or `RESOLVE_TRUSTED_LINK_SCOPE`).
- `investigatory_action`: the original NVIDIA Investigatory Action.
- `action`: the execution action the current control plane has approved and can express.
- `safety_action`: the site's fail-closed action when the official workflow is blocked because evidence or the executor is
  incomplete; currently only `QUARANTINE` is allowed.
- `source`: `NVIDIA_CATALOG`, `NVIDIA_XID_154`,
  `NVIDIA_FABRIC_MANAGER` or `SITE_SAFETY`.
- `disposition`: pending correlation (`PENDING_CORRELATION`), executable (`EXECUTABLE`), monitor only (`MONITOR_ONLY`),
  blocked on missing workflow (`BLOCKED_WORKFLOW`), blocked on missing evidence (`BLOCKED_MISSING_EVIDENCE`), not applicable
  (`NOT_APPLICABLE`) or site safety (`SITE_SAFETY`); the last four without an executable action carry
  `safety_action=QUARANTINE`.

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
`BatchRebootClusterNodes` executes it; the `eks` (plain EKS, `eks-managed`) profile maps it to `REBOOT_NODE` as well, executed
by EC2 `RebootInstances` on the instance the Node providerID binds (`node_provider.restart_vm_effective_action`); a
`kubernetes` profile maps nothing and keeps the engine's BLOCKED verdict. The mapping applies to an XID (XID 151) only; the
`RESTART_VM` of SXID 11004/12028 is not rewritten. This mapping does not mean an EC2 stop/start; the host placement, local storage and identity semantics of stop/start and
node replacement remain independent actions.

`CONTACT_SUPPORT`, `CHECK_MECHANICALS` and `UPDATE_SWFW` are not crudely replaced with another
"official action": the policy keeps the original action and returns EXECUTABLE, and the workflow compiler lands them respectively as
`ESCALATE_SUPPORT` (fixed-template mail, no isolation), `CHECK_MECHANICALS` (notify and wait for manual confirmation)
and a downtime chain containing `UPDATE_SOFTWARE_FIRMWARE` (it needs `GPU_FAULT_TARGET_FIRMWARE_VERSION`; without it the plan
blocks and only the safety chain FREEZE_EVIDENCE → MARK_UNSCHEDULABLE → QUARANTINE runs). Other workflow names with no registered resolver keep
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
  it reuses the incident/workflow of the most severe accompanying XID; when solo it keeps `RESTART_FM`. A node of a catalog product family that is not
  graded -- A100 (NVIDIA's own solo case), PCIe/NVL variants, GH200, GB200/GB300, B100 and the other parts the fabric grading
  leaves out, and a Hopper/Blackwell node whose exporter reports no field 170 (or only NOT_SUPPORTED/UNRECOGNIZED/NVML_TOO_OLD)
  -- compiles it to `RESTART_FABRIC_MANAGER` (`FREEZE_EVIDENCE → RESTART_FABRIC_MANAGER`)
  whatever the switch below says, executed by the Node Agent's `fabricManagerRestart` capability. On a graded NVSwitch node
  (a Hopper SXM or HGX B200/B300 part whose latest DCGM samples report field 170), a Fabric Manager restart alone
  loses the fabric state, so the product compiles `RESTART_FM` there as NVIDIA's H100-and-later procedure, the full GPU + NVSwitch reset;
  NVIDIA's GPU debug guidelines, on the other hand, read such a solo XID 45 as informational (see "Evidence" below). Such a
  node is therefore **observe-only** by default (`orchestration/xid45_policy.py`): the decision becomes `MONITOR_ONLY` /
  `NO_ACTION`, `official_action` stays `RESTART_FM` and `investigatory_action` is unchanged, the marker is INFO and not
  active; the incident is recorded RECOVERED with its evidence, its `policy_reference` is the debug-guidelines URL and its
  reasons say why and name the switch; the XID investigatory notification is sent as usual; no workflow is built, no node
  action is sent and nothing cordons the node. The control-plane environment variable
  `GPU_FAULT_XID45_SOLO_NVSWITCH_RESTART_FM=true` (the mechanism the SXID remediation lists already use for site variants of
  catalog actions; unset by default = observe-only) restores the catalog behaviour: `RESTART_FM` compiles to the full GPU +
  NVSwitch reset chain on the graded GPUs (`RESET_ALL_GPUS_NVSWITCHES` inside a quiesce window that stops and restarts
  Fabric Manager, then VALIDATE_GPU and VALIDATE_FABRIC), under the `fabricReset` capability. The shared
  Store lease supports HA, out-of-order arrival and Pod restarts.
- XID 48: solo executes `RESET_GPU`; with an XID 63/64 already stored for the same GPU and boot within 30 seconds it is
  recorded as `DRAIN_AND_RESET` (the same RESET_GPU chain, cordoned first) and merges only into an XID 64's incident, never
  into an XID 63's; a later XID 64 links back to that XID 48, so the GPU is reset once.
- XID 94/95: application/all-applications containment are stored separately; XID 95 requires the affected workload to be stopped before
  the reset.
- XID 154: the driver-reported action takes precedence. The catalog's `Xid 154 linkage` describes the
  version from which XID 154 is reported alongside, not the original XID's minimum supported version; an old driver or
  missing version information does not block the original rules of XID 48/63/79/94/95 and the others. The NVLink5 R575
  bit-format boundary and its required decode evidence remain. DCGM field 1523 holds the same value as a
  state: values 1-4 take the XID 154 decision of the same label, and the value is evidence only while an XID
  154 owns the GPU's recovery (see "Evidence: the DCGM field audit rules" below).
- `DRAIN_P2P`: after cordoning, stopping the associated jobs, quiescing and confirming that no GPU client is left, a
  dedicated `RECOVER_DRAIN_P2P` operation checks that the target GPUs' persistence is off and re-reads the GPU Recovery
  Action. `None` resets nothing; `Reset`, `Drain and Reset` and a still-`Drain P2P` reuse the existing guarded GPU reset;
  when any target asks for `Reboot`, no other target is reset first and the node is rebooted. An unknown action, a missing
  or duplicate GPU, or persistence not released authorises no reset or reboot and goes to an operator. The dedicated
  operation lets an old Agent refuse a request it does not know, so it can never silently degrade to `STOP_WORKLOADS` or an
  unconditional reset.
- XID 159 `CHECK_UVM`: reset when the event carries `uvm_in_use=true`, ignore on `false`; without that evidence
  `BLOCKED_MISSING_EVIDENCE` + `safety_action=QUARANTINE` (this system's fail-closed choice). The product's own collectors do
  not provide the field, so a collected XID 159 is currently handled as missing evidence.
- XID 144-150: selects the V1/V2 `IntrInfo` pattern by the driver R575 boundary, matching
  Error Status and Action 2 at the same time. The primary action and Action 2 may use different bit patterns and are matched separately;
  an RXPIPE secondary-pattern hit must still satisfy the Error Status of the same table entry, and cannot first require the primary pattern to hit as well.
  When no official table entry matches, it blocks and does not guess an action.

When the correlated raw record reaches its retention limit and the same event is re-delivered, the surviving final decision and incident binding must be reused.
A purged raw record does not mean the event was never handled; an uncorrelated cached policy result must not overwrite the final decision;
records still in `PENDING_CORRELATION` continue through the original window repair flow.

### Evidence: the observe-only default for a solo XID 45 on a graded NVSwitch node

Both pages were verified against NVIDIA's live pages:

- NVIDIA GPU Debug Guidelines -- GPU Node Triage, section "Understanding Xid Messages", XID table row
  "45 Robust Channel Preemptive Removal"; docs version 615, page "Last updated on Oct 04, 2026":
  `https://docs.nvidia.com/deploy/gpu-debug-guidelines/gpu-node-triage.html#understanding-xid-messages`
  Verbatim:
  "No action, informative only. Indicates channels affected by another failure. On A100, this error could be seen by itself due to unexpected Fabric Manager shutdown when FM is running in the same OS environment as the GPU. Otherwise, this error is safe to ignore as an informational message."
- NVIDIA XID Catalog, docs version 615, page "Last updated on Sep 09, 2026":
  `https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html`; spreadsheet
  `https://docs.nvidia.com/deploy/xid-errors/_downloads/4586dadb59119a55d1e93a181caa4272/Xid-Catalog.xlsx`
  (sha256 `7f70ce9684be0c9d98a367770341ccc1d8bca4465cebe42dab7e23d57df5d5a5`, byte-identical to the XLSX pinned above).
  "Xids" sheet, Code 45: Immediate Action `WORKFLOW_XID_45`; Investigatory Action
  "Solo: RESTART_FM / Not Solo: IGNORE (follow other Xid)"; Trigger:
  "This event is logged when the user application aborts and the kernel driver tears down the GPU application running on the GPU. Control-C, GPU resets, sigkill are all examples where the application is aborted and this event is created. In many cases, this is not indicative of a bug but rather a user or system action."

Conclusion (our reading and our deviation): NVIDIA's own triage guide explains the catalog's "Solo: RESTART_FM" as the A100 case (an unexpected Fabric
Manager shutdown with Fabric Manager in the GPU's OS) and reads every other solo XID 45 as informational. The default
departs from the catalog's investigatory action only where NVIDIA's own debug guidance reads the solo event as
informational -- graded NVSwitch Hopper/Blackwell nodes; the catalog data itself is unchanged (the generated artifact, its
digests and `mapping_version` stay as they are); A100 keeps `RESTART_FM` (NVIDIA's own solo case). An observe-only
incident's `policy_reference` is the debug-guidelines URL above; a site that wants the catalog behaviour sets
`GPU_FAULT_XID45_SOLO_NVSWITCH_RESTART_FM=true`.

### Evidence: the post-reset VALIDATE_GPU re-judges a pending row remap / page retirement on a newer batch

Every page below was verified against NVIDIA's live page (docs version 615):

- NVIDIA GPU Memory Error Management, section "Error Recovery and Response Flags", item "Row-remapping pending flag";
  page "Last updated on Sep 09, 2026":
  `https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/error-recovery-and-response-flags.html#error-recovery-and-response-flags`
  Verbatim:
  "This flag indicates that row-remapping will happen at the next GPU reset."
- The same manual, section "Row Remapping":
  `https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/row-remapping.html#row-remapping`
  Verbatim: "The process of row-remapping requires a GPU reset to take effect and will remain persistent throughout the life of the life of the GPU."
  Table 2 "Page Retirement vs. Row-Remapping", row "Application of pending changes": Ampere and later "GPU reset is required.";
  legacy page retirement "Needed a kernel module reload or driver re-initialization or GPU reset".
- NVIDIA Dynamic Page Retirement, section "Visibility" > "nvidia-smi"; page "Last updated on Sep 09, 2026":
  `https://docs.nvidia.com/deploy/dynamic-page-retirement/visibility.html#nvidia-smi`
  Verbatim: "The nvidia-smi “Pending Page Blacklist” field indicates whether a page has been recently retired and will be blacklisted on the next system reboot or driver load."
  The same document, "Blacklisting and ECC Error Recovery" > "Reattaching the GPU"
  (`https://docs.nvidia.com/deploy/dynamic-page-retirement/blacklisting-and-ecc-error-recovery.html#reattaching-the-gpu`),
  lists the ways to apply pending retirements, in order of cost: "Re-attach the GPUs (persistence mode disabled only)",
  "Reset the GPUs", "Reload the kernel module (nvidia.ko)", "Reboot the machine (or VM)".
- NVIDIA GPU Debug Guidelines -- GPU Node Triage, "Understanding Xid Messages", row 63 "A100: Row-remapping recording
  event"; page "Last updated on Oct 04, 2026":
  `https://docs.nvidia.com/deploy/gpu-debug-guidelines/gpu-node-triage.html#understanding-xid-messages`
  Excerpt: "until there is a convenient time to reset the GPU (refer to GPU reset capabilities/limitations section below) or reboot for row remapping to activate."

Conclusion: a pending row remap or page retirement takes effect at the next GPU reset (or node reboot), so after this
workflow's `RESET_GPU` (on that GPU), `RESET_ALL_GPUS_NVSWITCHES` or `RESTART_NODE` a pending finding whose reading is not
newer than that completion is no verdict on the reset GPU: as long as every blocking finding is a pending-repair flag
(`row_remap_pending`/`retired_pages_pending`), `VALIDATE_GPU` keeps `WAITING` (`awaiting_newer_gpu_metrics` and
`pending_repair_reset` in `node_pending.<node>`, `policy_references` the flag's URL above) until a GPU metrics batch newer
than the reset re-judges it; cleared succeeds, still set fails on the existing path; the bound is the executor's per-step
cap; with any other finding beside it the step fails at once as before. `RECOVER_DRAIN_P2P` (it can succeed with `reset_skipped`), `REMEDIATE_DRIVER`, `UPDATE_SOFTWARE_FIRMWARE` and
`REPLACE_NODE` do not count as applying the repair. Implementation `src/gpu_fault/adapters/pending_repair_rejudge.py`; the
quotes are pinned by `tests/execution/test_validation_pending_repair_after_reset.py`. The case has not occurred on real
hardware; unit tests cover the behaviour (see the pending repair flag paragraph of [Detailed Design](../detailed-design.md) §2.4.1, "GPU Metrics `gpu_metrics.py`").

### Evidence: the DCGM field audit rules

Every page below was re-fetched and checked; NVML and NVIDIA GPU Memory Error Management are docs version 615 (pages
"Last updated on Sep 09, 2026"), the DCGM pages "Version: 4.6 (Latest)", the DCGM source tag v4.7.0. The full quotes and the
decisions are in the detailed design §2.4.1, "The DCGM field audit's fields"; each quote is pinned by
`tests/test_dcgm_gap_field_evidence.py`.

- GPU recovery action (1523): GPU Memory Error Management, "Error Recovery and Response Flags"
  `https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/error-recovery-and-response-flags.html#error-recovery-and-response-flags`
  "Recovery action flag will be triggered for uncorrectable errors."; NVML API Reference, "vGPU Structs",
  `nvmlDeviceGpuRecoveryAction_t`
  `https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlVgpuStructs.html#_CPPv429nvmlDeviceGpuRecoveryAction_t`
  "NVML_GPU_RECOVERY_ACTION_RECOVER_IMEX_DOMAIN Recover IMEX Domain.", "NVML_GPU_RECOVERY_ACTION_BUS_RESET Reset the GPU’s
  PCIe bus.", "NVML_GPU_RECOVERY_ACTION_SYSTEM_REBOOT Reboot the system."; DCGM Release Notes 4.7.0
  `https://docs.nvidia.com/datacenter/dcgm/latest/release-notes/changelog.html` "Added DCGM_FR_GPU_RECOVERY_IMEX_DOMAIN for
  the separate case in which driver recovery requires restoring an IMEX domain."; NVML API Reference change log
  `https://docs.nvidia.com/deploy/nvml-api/latest/change-log.html`, "Changes between NVML v610 and v615": "Added
  NVML_GPU_RECOVERY_ACTION_BUS_RESET and NVML_GPU_RECOVERY_ACTION_SYSTEM_REBOOT to nvmlDeviceGpuRecoveryAction_t."; nvidia-smi
  documentation
  `https://docs.nvidia.com/deploy/nvidia-smi/index.html`, "-r, --gpu-reset": "Reset triggered without extra arguments, will
  be a default Function Level Reset (FLR). To issue a Bus Reset, use -r bus."
- PCIe link (235-238): nvidia-smi documentation, GPU ATTRIBUTES › PCI › GPU Link information: "The current link generation
  and width. These may be reduced when the GPU is not in use.", "The maximum link generation and width possible with this
  GPU and system configuration."; NVIDIA DGX B300 User Guide
  `https://docs.nvidia.com/dgx/dgxb300-user-guide/introduction-to-dgxb300.html#key-enhancements-over-previous-dgx-systems`
  "ConnectX-8 networking connects directly to GPUs via PCIe Gen6."
- PCIe replay (202) and correctable errors (1501): DCGM Feature Overview, "Background Health Checks" › "Failure Conditions"
  `https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html#background-health-checks` "There were
  more than 8 PCIe replays in the last minute."; DCGM v4.7.0 `modules/health/DcgmHealthWatch.cpp`
  `https://github.com/NVIDIA/DCGM/blob/v4.7.0/modules/health/DcgmHealthWatch.cpp` "Expected replay rate = Replay rate per
  lane * Num of pcie Lanes. Note that value is ceiled."; DCGM v4.7.0 `nvvs/plugin_src/pcie/Pcie.cpp`
  `https://github.com/NVIDIA/DCGM/blob/v4.7.0/nvvs/plugin_src/pcie/Pcie.cpp` "/* Default is hopper, but fall back for other
  devices */"; NVML API Reference, "Field Value Enums"
  `https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlFieldValueEnums.html#c.NVML_FI_DEV_PCIE_COUNT_CORRECTABLE_ERRORS`
  "PCIe Correctable Errors Counter." (no health threshold).
- SRAM threshold (346), no spare rows (389), uncorrectable remaps (393): GPU Memory Error Management, "RMA Policy"
  `https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/rma-policy-thresholds-for-row-remapping.html#gpu-l2-memory-rma-policy`
  "Any of the following events will trigger the SRAM Threshold Exceeded flag:"; the same page, "GPU DRAM Memory RMA Policy"
  `https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/rma-policy-thresholds-for-row-remapping.html#gpu-dram-memory-rma-policy`: "For Blackwell GPUs, a third remapping attempt for an uncorrectable memory error on a bank
  invokes HBM channel repair if there is a spare channel available."; DCGM Release Notes 4.6.0 "Health monitoring now warns
  when total uncorrectable row remaps reach 512 on Ampere and newer GPUs, allowing administrators to run field diagnostics
  before GPU memory health degrades further."
- HW slowdown times (1423/1424) and clock-event bits: NVML API Reference, "Field Value Enums"
  `https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlFieldValueEnums.html` "Field values for Clock Throttle
  Reason Counters All counters are in nanoseconds."; NVML API Reference, "Clocks Event Reasons"
  `https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlClocksEventReasons.html`, `nvmlClocksThrottleReasonHwSlowdown`:
  "May be also reported during PState or clock change This behavior may be removed in a later release."
- ECC mode (300/301) and InfoROM: NVML API Reference, "Device Queries"
  `https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html` "Requires ECC Mode to be enabled.";
  Dynamic Page Retirement, "Implementation"
  `https://docs.nvidia.com/deploy/dynamic-page-retirement/implementation.html#implementation` "These addresses are stored in
  the InfoROM."

Conclusion: values 1-4 of field 1523 take the XID 154 decision of the same label; RECOVER_IMEX_DOMAIN goes to an operator;
BUS_RESET, SYSTEM_REBOOT and undocumented values fail closed (NVIDIA documents only one-line enum meanings for 6 and 7, and
the product's GPU reset is an FLR, not a bus reset). The PCIe replay threshold follows DCGM's health formula for the current
link (8/min only without a link reading; a configured `GPU_FAULT_PCIE_REPLAY_RATE_WARNING_PER_MINUTE` other than 8 wins); the correctable-error limit is the lower of that formula and DCGM's PCIe
diagnostic limit. Blackwell (B200/B300) behaviour is unit-tested from NVIDIA documentation, not live-verified (no B-series
hardware).

### Evidence: the HGX B200/B300 DCGM rules

The pages below were re-fetched and checked: NVML API Reference and NVIDIA GPU Memory Error Management docs version 615,
the DCGM pages "Version: 4.6 (Latest)", the DCGM source tag v4.7.0, the Fabric Manager User Guide "Last updated on Oct 07,
2026", the nvidia-smi documentation of the v610 era, and NVIDIA's `nvidia-ml-py` 13.615.71 for the fabric-health bit
values. The full quotes and the decisions are in the detailed design §2.4.1, "HGX B200/B300 DCGM fields"; each quote is
pinned by `tests/test_blackwell_dcgm_evidence.py`. Blackwell (B200/B300) behaviour is unit-tested from NVIDIA
documentation, not live-verified (no B-series hardware).

- NVLink completeness (1211-1213 as alternatives of the data-link counters 497-499): NVML API Reference, "Field Value
  Enums", `NVML_FI_DEV_NVLINK_ERROR_DL_CRC`
  `https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlFieldValueEnums.html#c.NVML_FI_DEV_NVLINK_ERROR_DL_CRC`
  "This is unsupported for Blackwell+ Please use NVML_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_*".
- NVLink5 counters (1204-1209, 1211-1214, 1219, 1220): DCGM Field Identifiers
  `https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html` "Returns aggregate value across all
  links. Not supported on NVLink4 and earlier."; DCGM Feature Overview, "Background Health Checks" › "Failure Conditions"
  `https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html#background-health-checks` "An NVLink
  reported 1 or more errors. This includes datalink CRC, recovery, and replay errors (pre-Blackwell GPUs) and link recovery
  events (Blackwell+ GPUs)."; DCGM v4.7.0 `modules/health/DcgmHealthWatch.cpp`
  `https://github.com/NVIDIA/DCGM/blob/v4.7.0/modules/health/DcgmHealthWatch.cpp`: 1212 above 0 FAIL, 1211 above 0 WARN,
  any change within a minute of 1204-1209, 1213, 1214, 1219 or 1220 FAIL, next step "Run a field diagnostic on the GPU."; DCGM Release
  Notes 4.7.0 `https://docs.nvidia.com/datacenter/dcgm/latest/release-notes/changelog.html` "Removed NVLink bit-error-rate
  thresholds that caused false health and diagnostic failures on healthy GB200 systems." (BER is not graded).
- Fabric health (174/175): NVML API Reference, "Fabric Definitions"
  `https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlFabricDefs.html` "Fabric Health Mask: Route is
  unhealthy.", "Fabric Health Mask: Bandwidth degraded.", "Fabric Health Summary: Unhealthy.", "Fabric Health Summary:
  Limited Capacity."; DCGM Field Identifiers
  `https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html#c.DCGM_FI_DEV_FABRIC_HEALTH_MASK` "Use
  DCGM_GPU_FABRIC_HEALTH_TEST macro to check the different health statuses."; DCGM v4.7.0 `MonitorFabricFields` next step
  "Check DCGM and system logs for errors. Reset GPU. Restart DCGM. Rerun diagnostics."; nvidia-smi documentation
  `https://docs.nvidia.com/deploy/nvidia-smi/index.html`, "Fabric" › "Health": "Summary - Summary of Fabric Health
  <Healthy, Unhealthy, Limited Capacity>".
- Memory repair: GPU Memory Error Management, "RAS Repair" › "GPU Memory Repair"
  `https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/hbm-channel-repair.html#gpu-memory-repair` "The process of channel
  repair requires a reboot to take effect."; nvidia-smi documentation, "ECC Errors": "If one of the repair flags is pending,
  check the GPU Recovery action and take the appropriate steps."; DCGM Field Identifiers
  `https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html#c.DCGM_FI_DEV_MEMORY_UNREPAIRABLE`
  "Unrepairable memory flag indicating if memory has unrepairable errors 1=yes, 0=no"; NVML API Reference change log
  `https://docs.nvidia.com/deploy/nvml-api/latest/change-log.html`, "Changes between v580 and v590": "Added
  nvmlDeviceGetUnrepairableMemoryFlag_v1() to report whether or not there is unrepairable memory."

Conclusion: on HGX B200/B300 the NVLink5 link-recovery counters answer for the NVLink fields NVML no longer reports; a
failed link recovery (1212) drains, the other NVLink5 counters take the NVLink family's WARNING, an increment across a
product window grades nothing and one an XID 144-150/155 owns is evidence; a DCGM field 230 XID change and the kernel line
on the same GPU within 120 s are one occurrence (see Part IV of [fault categories and actions](../fault-categories-and-actions.md)),
and a change of XID 144-150 waits up to 120 s for its kernel line. An unhealthy fabric (summary UNHEALTHY, route unhealthy
or an incorrect configuration) held ≥ 60 s outside any product fabric window is confirmed and takes NVIDIA's full GPU +
NVSwitch reset; a route or access-timeout recovery in progress is evidence only and is treated the same after 300 s; a
degraded fabric gets a diagnostic and an operator notice, NOT_SUPPORTED nothing; the rule covers every graded NVSwitch part
(Hopper SXM included). DCGM records FAIL at once for the NVLink5 error counters and these fabric states: the product's
WARNING grading, its confirmation wait and never resetting for a degradation are our deviation. A pending memory repair is an operator's reboot unless the GPU recovery action
decides; unrepairable memory takes the row-remap failure's RMA drain. Per-link fields, the NVSDM switch fields and the
ConnectX bridge fields are deferred ([validation limitations](../../validation-limitations.md)).

## SXID

SXID recovery follows the Fabric Manager User Guide (`https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html`;
the artifact `src/gpu_fault/data/nvidia-fabric-manager-sxid-2025-11-14.yaml` pins Tables 21-24 of its "2025-11-14" edition):

- non-fatal: the catalog's IGNORE codes and non-fatal codes outside the tables are informational, keep monitoring; the non-fatal
  19084, 11004/12028 and 20012 follow their own rules below.
- fatal access link: the access scope must come from a trusted NVSwitch port topology (`TRUSTED_NVSWITCH_TOPOLOGY`) or the
  HGX H100/H200 product invariant (`NVIDIA_PRODUCT_INVARIANT`), otherwise `BLOCKED_MISSING_EVIDENCE` + QUARANTINE; with a
  non-empty participating GPU set the node is cordoned, the job stopped and that group reset (`RESET_PARTICIPATING_GPUS`),
  with an empty one it blocks. The set is the union of the port's GPU from the trusted topology, the GPUs in the
  NVSwitch's PCI slot and every GPU of the node's active managed jobs -- the jobs the same workflow stops; without a named
  GPU the affected job cannot be told from the others. Evidence: Fabric Manager User Guide, "Runtime NVSwitch and GPU
  Errors" > "NVSwitch SXid Errors" > "Fatal SXid Errors" (page "Last updated on Oct 07, 2026"),
  `https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html#fatal-sxid-errors`: "When a fatal SXid
  error is reported on a NVSwitch port that connects a GPU and an NVSwitch, the corresponding error will be propagated to
  the GPU. The CUDA jobs that are running on that GPU will be aborted" and "Reset the specified GPU and all the
  participating GPUs in the affected workload" (pinned by `tests/app_services/test_access_link_sxid_scope.py`).
- fatal trunk and 10003/19084: require fabric_partition and the complete node GPU inventory; when both are present
  `RESET_ALL_GPUS_AND_NVSWITCHES` is executed (Node Agent `fabricReset` capability, multi-node barrier);
  if either is missing the result is `BLOCKED_MISSING_EVIDENCE`, and it never degrades to a single-GPU reset.
  Complete means exactly the node's declared expected GPU count (the collector's `GPU_FAULT_EXPECTED_GPU_COUNT` or
  instance-type count, the `expected_gpu_count` of its inventory snapshots): a fresh inventory snapshot (one
  nvidia-smi enumeration) that lists fewer GPUs than it declares is not; without a fresh snapshot, the latest per-GPU
  metric rows around the event count only when they cover exactly the count the node's latest snapshot declares --
  they are no single delivery, and for an event older than the newest deliveries the window keeps exactly the rows
  that stopped refreshing (for example the `pcie_link_downtrain` verdict row of an idle GPU stays at its last delivery
  after a collector restart); the GPU a trusted port topology names for the SXID's port is not either. An incomplete inventory
  leaves the event without participating GPUs and takes the existing evidence, cordon, quarantine and operator-review
  path.
- always-fatal (the 20 codes pinned by Table 23): executes `REBOOT_NODE`, cordoning and stopping the job beforehand, and
  needs neither fabric_partition nor a GPU inventory; a code outside the pinned tables that declares Always-Fatal or Fatal
  always blocks (`BLOCKED_MISSING_EVIDENCE`): a client-supplied severity never authorises a destructive recovery.
- `RESTART_VM` (11004, 12028, the Shared NVSwitch/vGPU case): bare metal has no guest VM lifecycle, so `BLOCKED_WORKFLOW` +
  `safety_action=QUARANTINE` (the isolation is this system's fail-closed choice); `CHECK_MECHANICALS` (20012):
  `ESCALATE_OPERATOR`, compiled to FREEZE_EVIDENCE → CHECK_MECHANICALS (notify and wait for an operator's acknowledgement,
  no cordon, no stop).
- The Blackwell families (B100/GB200 product families: B200/B300, GB200/GB300): the traditional SXIDs do not apply,
  `NOT_APPLICABLE` (investigatory `USE_DCGM_NVSDM_TELEMETRY`) + `safety_action=QUARANTINE` (this system's fail-closed choice);
  a product in no known family is `BLOCKED_MISSING_EVIDENCE` (`VERIFY_GPU_PRODUCT_GENERATION`).

The SXID `classification_source` must be `NVIDIA_FABRIC_MANAGER` (or `NVIDIA_FABRIC_MANAGER_CATALOG`,
`NVIDIA_FABRIC_MANAGER_RUNTIME`); otherwise it blocks because the evidence source is untrusted, and so does a
fatal/non-fatal observed on the event that conflicts with the pinned table.
