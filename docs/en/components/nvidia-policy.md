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

NVIDIA's rules, verbatim:

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
decides, 4.7.0 as the live Agent heartbeats report, newer than the 4.6.1 client, the connection requirement broken with no product
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
  it reuses the incident/workflow of the most severe accompanying XID; when solo it keeps `RESTART_FM`. A node that is not
  graded -- A100 (NVIDIA's own solo case), PCIe/NVL variants, GH200, GB200/GB300, unknown parts, and a Hopper/Blackwell node
  whose exporter reports no field 170 -- compiles it to `RESTART_FABRIC_MANAGER` (`FREEZE_EVIDENCE → RESTART_FABRIC_MANAGER`)
  whatever the switch below says, executed by the Node Agent's `fabricManagerRestart` capability. On a graded NVSwitch node
  (a Hopper SXM or HGX B200/B300 part whose latest DCGM samples report field 170), a Fabric Manager restart alone
  loses the fabric state, so `RESTART_FM` there can only be NVIDIA's H100-and-later procedure, the full GPU + NVSwitch reset;
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

Conclusion: NVIDIA's own triage guide explains the catalog's "Solo: RESTART_FM" as the A100 case (an unexpected Fabric
Manager shutdown with Fabric Manager in the GPU's OS) and reads every other solo XID 45 as informational. The default
departs from the catalog's investigatory action only where NVIDIA's own debug guidance reads the solo event as
informational -- graded NVSwitch Hopper/Blackwell nodes; the catalog data itself is unchanged (the generated artifact, its
digests and `mapping_version` stay as they are); A100 keeps `RESTART_FM` (NVIDIA's own solo case). An observe-only
incident's `policy_reference` is the debug-guidelines URL above; a site that wants the catalog behaviour sets
`GPU_FAULT_XID45_SOLO_NVSWITCH_RESTART_FM=true`.

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
