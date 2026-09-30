English edition of `docs/故障类别与处置动作总表.md`; the Chinese file remains the source of record until both are maintained together.

# Fault Categories and Remediation Actions

This document is the item-by-item expansion of [High-Level Design v2](high-level-design-v2.md) §2 "Fault classification and policy matrix": it puts, for every fault category,
the detection rule, the official NVIDIA recommendation, this solution's policy decision and the actually compiled execution steps into one table.
The source of fact is the code; this document only organises it:

- policy decision: `src/gpu_fault/policy/engine.py`, `src/gpu_fault/policy/nvlink74.py`;
- official catalogs: `src/gpu_fault/data/nvidia-xid-catalog-610.generated.yaml`,
  `src/gpu_fault/data/nvidia-fabric-manager-sxid-2025-11-14.yaml`;
- step compilation: `src/gpu_fault/orchestration/workflow_builder.py`,
  `src/gpu_fault/orchestration/coordinator.py`, `src/gpu_fault/orchestration/families/health.py`;
- non-XID rules: `src/gpu_fault/host_health.py`, `src/gpu_fault/gpu_metrics.py`;
- capability switches: `config/runtime-profile.regional-hyperpod-safe.example.yaml`.

If rule thresholds, action names or step order disagree with the code, the code prevails and this document is revised;
the NVIDIA catalog supply chain and gate semantics are in [NVIDIA policy supply chain and implementation](components/nvidia-policy.md).

Reading conventions:

- The "NVIDIA official recommendation" column quotes the Xid Catalog / Fabric Manager User Guide text verbatim; "This solution's decision" is the disposition and `RecoveryAction` output by the policy engine; "Actual execution" is the compiled `WorkflowOperation` step chain.
- `[...]` denotes a conditional step: `STOP_WORKLOADS` is inserted only when workload_state is ACTIVE; `CHECKPOINT_WORKLOADS` is inserted only when a checkpoint_manifest_ref exists (it only records the reference, no real checkpoint is taken); `RESTART_WORKLOAD` is appended only when there are affected_workload_ids.
- Hardware, driver and workload mutations belonging to `NODE_MUTATING_OPERATIONS` are refused at compile time when workload_state is UNKNOWN (fail closed); pure scheduling isolation is not forbidden by this. Safety isolation for BLOCKED-type decisions must still be authorised by the Profile; once granted it executes FREEZE_EVIDENCE → MARK_UNSCHEDULABLE → QUARANTINE and waits for a human.
- NVIDIA `RESTART_APP` (including the `restart app` label of XID 154 and the fallback of NVLink5 `XID_154_EVAL` without an accompanying 154) is re-decided by the policy engine as MONITOR_ONLY / `NO_ACTION` when workload_state is IDLE: there is no managed application to restart, only an INFO marker and an investigation notification are written, no workflow is built and no cordon. The "Actual execution" column of the RESTART_APP rows in the table below describes the ACTIVE case where attribution to a workload succeeds; ACTIVE but not attributable to a workload still goes through SAFETY_PENDING isolation, and UNKNOWN still fails closed per the previous item.
- Chain A = GPU reset chain, chain B = node reboot chain; definitions are in the RecoveryAction landing table at the end of Part I.

## I. General fault categories (non-XID/SXID)

| Fault category | Data source / detection method | Decision rule | This solution's RecoveryAction | Actual execution |
|---|---|---|---|---|
| CPU overload | Host collector `/proc/stat`, `/proc/loadavg`, 15s period | `cpu_usage_percent` ≥98; `load1_per_cpu` ≥2.0 | WARNING → `RUN_DIAGNOSTICS` | FREEZE_EVIDENCE → VALIDATE_HOST (read-only verification of control-plane telemetry) |
| Memory/swap exhaustion | `/proc/meminfo` | `memory_used_percent` ≥95; `swap_used_percent` ≥80 | WARNING → `RUN_DIAGNOSTICS` | Same as above |
| Low utilisation / persistently abnormal resources (`SITE_HOST_RESOURCE_HEALTH`) | Same as above + nvidia-smi utilisation | `LOW_CPU_UTILIZATION`/`LOW_GPU_UTILIZATION` ≤5% for 300s (timed only when the control plane resolves exactly one running managed attempt on the node; the collector self-reporting ACTIVE is not enough to start it); `LOW_MEMORY_AVAILABLE` ≤5% for 60s; `HIGH_PAGE_CACHE` ≥90% for 300s; `HIGH_LOCAL_FILESYSTEM_USAGE` ≥90% for 60s | WARNING → `RUN_DIAGNOSTICS` | FREEZE_EVIDENCE → VALIDATE_HOST (for the GPU item RUN_DCGM_DIAGNOSTIC + VALIDATE_GPU) + aggregated advisory mail |
| Local disk full | `statvfs` on `/`,`/var`,`/tmp` | `filesystem_used_percent` ≥98 | CRITICAL → `QUARANTINE` | FREEZE_EVIDENCE → MARK_UNSCHEDULABLE → QUARANTINE (cordon + taint `gpu-fault.io/quarantined`, jobs not stopped) |
| Shared storage unavailable/full | statvfs on lustre/nfs mounts from mountinfo | `shared_filesystem_unavailable` ≥1 (CRITICAL); `shared_filesystem_used_percent` ≥98 (WARNING) | `RUN_DIAGNOSTICS` | FREEZE_EVIDENCE → VALIDATE_HOST |
| Disk IO saturation | `/proc/diskstats` | `disk_io_util_percent` ≥98 or `disk_io_await_ms` ≥100, for 120s | WARNING → `RUN_DIAGNOSTICS` | Same as above |
| SMART failure | `smartctl -H -j` | `smart_health_failed` ≥1 | CRITICAL → `QUARANTINE` | cordon + QUARANTINE |
| NIC link down | `/sys/class/net/*/operstate`, required interfaces only | `network_link_down` ≥1 | CRITICAL → `QUARANTINE` | cordon + QUARANTINE |
| NIC errors/drops | rx/tx errors, drops deltas | `network_errors_delta` ≥1; `network_drops_delta` ≥1 for 60s; TCP retransmits deliberately have no rule | WARNING → `RUN_DIAGNOSTICS` | FREEZE_EVIDENCE → VALIDATE_FABRIC |
| EFA/RDMA counter errors | `/sys/class/infiniband/*/ports/*/{counters,hw_counters}`, `ethtool -S` | `rdma_errors_delta`, `rdma_link_down`, `efa_rnr_errors_delta`, `efa_retry_errors_delta`, `efa_cq_errors_delta` ≥1 | WARNING → `RUN_DIAGNOSTICS` | FREEZE_EVIDENCE → VALIDATE_FABRIC |
| EFA traffic state machine (`SITE_EFA_TRAFFIC`) | EFA hw_counters byte rate, baseline EMA; evaluated only when the node has exactly one active attempt | `SPIKE` ≥4× baseline; `DROP` ≤0.25×; `ZERO_WARNING` zero traffic ≥60s; `HUNG_SUSPECTED` zero traffic ≥600s (rank progress/training progress can suppress) | First three WARNING → `RUN_DIAGNOSTICS`; HUNG_SUSPECTED CRITICAL → `RUN_DIAGNOSTICS`(capture_process_state) | HUNG: FREEZE_EVIDENCE → COLLECT_HUNG_TRIAGE (read-only sampling on ≤8 nodes: py-spy, NCCL flight recorder, /proc, EFA counters) → COLLECT_DIAGNOSTIC_BUNDLE → VALIDATE_FABRIC; no isolation, jobs not stopped |
| GPU dropped | `nvidia-smi --query-gpu=uuid` count vs instance type expectation | `gpu_inventory_mismatch` for 2 consecutive samples | CRITICAL → `REBOOT_NODE` | Node reboot chain B; can be absorbed by a RUNNING workflow on the same node |
| EFA dropped | PCI vendor 0x1d0f/0xefa*, driver binding, ports ACTIVE vs expected count | 2 consecutive samples, by failure_mode: `PCI_DEVICE_MISSING`, unknown → `REBOOT_NODE`; `DRIVER_UNBOUND` → `REMEDIATE_EFA_DRIVER`; `LINK_INACTIVE`/`EXCESS_DEVICE` → `RUN_DIAGNOSTICS` | All CRITICAL; repair failure escalates to `REBOOT_NODE` | REMEDIATE: FREEZE_EVIDENCE → MARK_UNSCHEDULABLE → [STOP_WORKLOADS] → REMEDIATE_EFA_DRIVER (`modprobe efa` + write `/sys/bus/pci/drivers/efa/bind`) → RESTART_EFA_DEVICE_PLUGIN → TRIGGER_HEALTH_SNAPSHOT → VALIDATE_FABRIC → RESTORE_SCHEDULING → [RESTART_WORKLOAD] |
| K8s allocatable under-reported | Cluster collector reads Node `status.allocatable` | `efa_kubernetes_allocatable_mismatch` / `gpu_kubernetes_allocatable_mismatch` for 2 consecutive samples | CRITICAL → `RESTART_EFA_DEVICE_PLUGIN` / `RESTART_GPU_DEVICE_PLUGIN`; failure escalates to `REBOOT_NODE` | FREEZE_EVIDENCE → delete that node's device-plugin Pod and wait for allocatable to recover (180s) → TRIGGER_HEALTH_SNAPSHOT → VALIDATE_FABRIC/VALIDATE_GPU; no cordon |
| BMC sensor alarm | `ipmitool sensor` | `bmc_critical_sensor` ≥1 | CRITICAL → `QUARANTINE` | cordon + QUARANTINE |
| Logs: MCE / storage read-only / RDMA fatal / kernel lockup | `NodeLogCollector` (disabled by default) journald regex | `MCE`, `STORAGE`, `RDMA`, `SYSTEM_LOG` rule hit | CRITICAL → `QUARANTINE` | cordon + QUARANTINE |
| Logs: OOM / NCCL / torch.distributed errors | Same as above | `MEMORY`, `NCCL`, `TRAINING` rule hit | WARNING → `RUN_DIAGNOSTICS` | FREEZE_EVIDENCE → VALIDATE_HOST or VALIDATE_FABRIC |
| GPU temperature | DCGM exporter / nvidia-smi | GPU: device slowdown/max-operating limits (85/90 ℃ if absent); memory: limits or 90/95 ℃ | WARNING → `RUN_DIAGNOSTICS`; CRITICAL → `DRAIN` | DRAIN: FREEZE_EVIDENCE → MARK_UNSCHEDULABLE → [STOP_WORKLOADS] → QUARANTINE → COLLECT_DIAGNOSTIC_BUNDLE → [RUN_FIELD_DIAGNOSTIC] → VALIDATE_GPU → ESCALATE_SUPPORT; the node stays isolated and scheduling is not restored automatically; incident terminal state `ESCALATED`, closed by a human after RMA |
| GPU ECC / row remap / page retirement | DCGM | `ecc_dbe_volatile_total` >0, `row_remap_failure` >0, `retired_pages_dbe_total` delta → CRITICAL; SBE, aggregate DBE, correctable remap, SBE retired page delta → WARNING; `row_remap_pending`/`retired_pages_pending` >0 → WARNING | CRITICAL → `DRAIN` (row_remap_failure appends RUN_FIELD_DIAGNOSTIC; in the example profile memoryDiagnostics=OBSERVE, which fails closed); WARNING → `RUN_DIAGNOSTICS`; pending → `RESET_GPU` | DRAIN chain / RUN_DCGM_DIAGNOSTIC + VALIDATE_GPU / GPU reset chain A |
| NVLink error counters | DCGM `nvlink_*_error_total` | Any delta ≥1 | CRITICAL → `DRAIN` | DRAIN chain |
| PCIe replay | DCGM `pcie_replay_total` | >8/min | WARNING → `RUN_DIAGNOSTICS`; combined with XID 32/79 as `PCIE_XID_LINK_FAILURE` → CRITICAL `DRAIN` | RUN_DCGM_DIAGNOSTIC + VALIDATE_GPU / DRAIN chain |
| Throttling / thermal violation / power | DCGM `clock_throttle_reasons` (bits 0x60), `thermal_violation_total_us`, `power_violation_total_us` | Single occurrence WARNING; thermal violation 2 consecutive or compound `THERMAL_STRESS` 2 consecutive → CRITICAL; power requires ≥95% of limit and utilisation ≥80% | WARNING → `RUN_DIAGNOSTICS`; CRITICAL → `DRAIN` | Same as above |
| DCGM compound rules (`SITE_DCGM_CORRELATION`, 45s window) | Combinations of the metrics above | `GPU_MEMORY_DEGRADATION`, `NVLINK_LINK_DEGRADATION`, `MULTI_GPU_NVLINK_FABRIC_FAILURE` → CRITICAL; `CORRECTABLE_MEMORY_DEGRADATION` 3 consecutive → CRITICAL; `POWER_LIMIT_THROTTLING` → WARNING | CRITICAL → `DRAIN`; WARNING → `RUN_DIAGNOSTICS` | DRAIN chain / diagnostics |
| DCGM collection incomplete | Batch collection_errors | Missing fields / scrape failure | WARNING → `RUN_DIAGNOSTICS`; cleared automatically by the next clean batch | Diagnostics |
| GPU identity change / expected count unknown | Inventory snapshot | `gpu_inventory_identity_changed` (CRITICAL); `gpu_expected_count_unknown` (WARNING) | `RUN_DIAGNOSTICS` / `COLLECT_EVIDENCE` | Diagnostics / FREEZE_EVIDENCE only + operator-review advisory |
| Training progress | `training-progress` reporter (off by default) | `training_nonfinite-loss`, `training_hang` (heartbeat >120s or steps not advancing) → CRITICAL; `training_step-regression`, `training_straggler` → WARNING | All `RUN_DIAGNOSTICS` | FREEZE_EVIDENCE → VALIDATE_FABRIC |
| Training process failure / attempt terminal state | Completion Watcher watches container exit codes and Job phase | Critical container non-zero exit or FAILED (not initiated by this solution) | `STOP_WORKLOAD` (incident `TRAINING_ATTEMPT_FAILURE_DETECTED`); after the terminal state a plan is generated from the active marker's recommended_action; with no marker, one in-place restart per restart_budget; no allocation blocks automatic restart | FREEZE_EVIDENCE → STOP_WORKLOADS (suspend + capture pod logs + delete pod with grace 0); on control-plane contact timeout the node side performs an emergency stop |
| Restart budget exhausted | `gpu-fault.io/restart-budget` annotation, default 1 | Control-plane claim preflight `reserve_restart_budgets` reservation fails | workflow FAILED, incident `ESCALATED` | No adapter call, RESTART_WORKLOAD not executed, the preflight sends mail |
| Warm spare node unhealthy | spare health check (K8s Ready, HyperPod status, marker, GPU finding, HMA label) | HARDWARE-class reason 2 consecutive times | CRITICAL → `REBOOT_NODE`; still failing → `UNAVAILABLE` (1h re-check, 24h alert) | Node reboot chain B |
| Collector silence | Control plane reads collection state, default scan interval 60s (not a loss-of-contact threshold) | In active clusters, an Agent that is ACTIVE with a valid heartbeat lease has no success record for a required channel, or the time since the last successful report exceeds the threshold: GPU_INVENTORY default 180s, GPU_METRICS/HOST 420s, log channels 900s; explicitly disabled/masked services excepted | No RecoveryAction; only an AdvisoryNotification via the notification channel | Missing monitoring is not treated as health; no unified UNKNOWN state is written, no incident or recovery workflow is created; Agent heartbeat loss is decided separately |
| Unknown XID / unknown GPU product / missing evidence | Policy engine | Not in catalog, product family unrecognised, version-gate evidence missing | `SITE_SAFETY` / BLOCKED + safety `QUARANTINE` | FREEZE_EVIDENCE → MARK_UNSCHEDULABLE → QUARANTINE, wait for a human |

When the same Host metric stays abnormal but the action, severity or failure_mode changes, a finding is produced again,
including EFA `LINK_INACTIVE -> PCI_DEVICE_MISSING`, without first returning to zero. Confirmation of a submitted incident
binds the original receipt moment and the remediation fingerprint; an old delivery cannot confirm a new round's semantics.

A correlated marker does not mean the old action suffices: a later XID/SXID must still verify the full action, GPU scope and attempt.
GPU and memory temperature both belong to the thermal fault category; kernel XID 32/79 joins the combined rule only when a PCIe threshold breach
on the same trusted device exists within the valid window; a kernel XID arriving alone still follows its independent policy.
Persistent isolation also differs from the temporary isolation in the reboot flow; after merging, the constrained faulty node must not be re-admitted;
recovery of a healthy warm spare must target the node actually substituted.

### How RecoveryAction lands

| RecoveryAction | Actual mechanism |
|---|---|
| `MARK_UNSCHEDULABLE` / `QUARANTINE` | The same K8s patch: `spec.unschedulable=true`, taint `gpu-fault.io/quarantined=sha256(incident)[:24]:NoSchedule`, write incident/fencing annotations; refused if isolated by another incident that is not terminal |
| `STOP_WORKLOAD` | patch Job/PyTorchJob/JobSet `suspend=true`, capture pod logs as evidence, delete pods with grace 0, wait for non-active; `enable-job-auto-resume=true` goes straight to FAILED |
| `RESTART_WORKLOAD` | Control-plane claim preflight reserves the restart budget, the data plane compares the `RestartAuthorization`; a terminal Job is recreated as `<name>-r-<sha8>`, otherwise suspend is cancelled; applies warm-spare rebinding and NotIn anti-affinity; a GPU count change requires the approval annotation |
| `RESET_GPU` (chain A) | FREEZE_EVIDENCE → MARK_UNSCHEDULABLE → [CHECKPOINT] → [STOP_WORKLOADS] → QUIESCE_GPU_SERVICES (systemctl stop fabricmanager/dcgm/persistenced/collector/kubelet, ctr kill the HMA and device-plugin containers, sweep `/dev/nvidia*` holders, 420s self-recovery timer on failure) → VERIFY_NO_GPU_CLIENTS → RESET_GPU (`nvidia-smi --gpu-reset -i <uuid>`) → RESTORE_GPU_SERVICES → VALIDATE_GPU → RESTORE_SCHEDULING → [RESTART_WORKLOAD]; failure escalates to REBOOT_NODE |
| `REBOOT_NODE` (chain B) | FREEZE_EVIDENCE → MARK_UNSCHEDULABLE → [QUARANTINE, health path] → [CHECKPOINT] → [STOP_WORKLOADS] → RESTART_NODE (`sagemaker.batch_reboot_cluster_nodes`, requires NodeRecovery≠Automatic, waits for boot_id change + 60s stable) → VALIDATE_GPU → VALIDATE_HOST → VALIDATE_FABRIC → RESTORE_SCHEDULING → [RESTART_WORKLOAD]; failure escalates to REPLACE_NODE |
| `REPLACE_NODE` | Calls no provider replace API; picks one healthy warm spare with `gpu-fault.io/spare=true` in the same instance group/type, isolates the spare and returns node_rebindings, and subsequent VALIDATE_* and RESTART_WORKLOAD move to the spare; without a spare FAILED + mail; failure escalates to ESCALATE_OPERATOR |
| `DRAIN` | cordon → [stop] → QUARANTINE → COLLECT_DIAGNOSTIC_BUNDLE (nvidia-smi -q/nvlink/topo, dcgmi diag -r 1, journal, nvidia-bug-report) → [RUN_FIELD_DIAGNOSTIC, only with `RUN_FIELD_DIAGNOSTIC_FOR_RMA`] → VALIDATE_GPU → ESCALATE_SUPPORT (SES fixed-template mail + vendor-ticket record); no RESTORE_SCHEDULING, the node keeps cordon+taint, incident terminal state `ESCALATED`, closed via `POST /v1/incidents/{id}/close` after manual RMA |
| `RUN_DIAGNOSTICS` / `VALIDATE_NODE` | Read-only: VALIDATE_GPU/HOST/FABRIC verify fresh telemetry in the control plane; the GPU class adds `dcgmi diag -r 1 -j`, and FAIL escalates to DRAIN |
| `REMEDIATE_EFA_DRIVER` / `RESTART_*_DEVICE_PLUGIN` | See the EFA rows in the table above; the node needs `GPU_FAULT_NODE_ALLOW_EFA_DRIVER_REMEDIATION`, the profile needs `efaDriverRemediation: OWN` |
| `ESCALATE_OPERATOR` | FREEZE_EVIDENCE → cordon → [stop] → QUARANTINE → ESCALATE_SUPPORT (SES fixed-template mail, no external ticketing API) |
| `RESTORE_SCHEDULING` | After verifying annotation ownership, remove the taint and restore the original unschedulable value |
| `COLLECT_EVIDENCE` / `NO_ACTION` | FREEZE_EVIDENCE only (records the reference) / no workflow built |

## II. XID table (NVIDIA Xid Catalog 610, 172 entries)

Engine order: catalog lookup → product gate (A*→A100, H*/GH*→H100, B*→B100, GB*→GB200) → version gate (when xid154Linkage carries driver/CUDA requirements the event must carry driver_branch and cuda_version; missing → BLOCKED_MISSING_EVIDENCE, below the requirement → NOT_APPLICABLE) → the driver-reported action of an accompanying XID 154 within 30s overrides this XID's Immediate Action → direct mapping (IGNORE→NO_ACTION, RESTART_APP→RESTART_WORKLOAD, RESET_GPU→RESET_GPU, RESTART_BM→REBOOT_NODE) or a dedicated workflow.

**The 61 unused XIDs (marked Unused in the catalog, no applicable product)**: 1-7, 9, 10, 12, 15-24, 26-30, 33-36, 42, 47, 49-53, 55-59, 61, 65, 73, 81, 87, 90, 91, 111-118, 122-125, 138. NVIDIA official: CONTACT_SUPPORT; this solution: product gate fails → NOT_APPLICABLE + safety `QUARANTINE`, executing FREEZE_EVIDENCE → isolate the node, wait for a human.

NVLink5 (XID 144-150) decode notes: the event must carry driver_branch, intr_info and error_status; missing any one → BLOCKED_MISSING_EVIDENCE + QUARANTINE. The V1/V2 IntrInfo pattern is chosen by the driver R575 boundary, then the Error Status and Action2 patterns are matched. If the hit set contains RESET_GPU → `RESET_GPU`; if it contains XID_154_EVAL → follow the action of the accompanying XID 154 (PENDING while the window is not closed, RESTART_WORKLOAD when there is no 154); all IGNORE → `NO_ACTION`; no table entry hit or conflicting actions → BLOCKED + QUARANTINE.

| XID | Name | Applicable products | NVIDIA Immediate Action | NVIDIA Investigatory Action | Version gate | This solution's decision | This solution's actual execution |
|---|---|---|---|---|---|---|---|
| 8 | ROBUST_CHANNEL_FIFO_ERROR_IDLE_TIMEOUT | A100,H100,B100,GB200 | RESTART_APP | CONTACT_SUPPORT |  | EXECUTABLE / `RESTART_WORKLOAD` | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 11 | ROBUST_CHANNEL_GR_ERROR_MISSING_HW | A100,H100,B100,GB200 | RESTART_APP | CHECK_APP/CUDA |  | EXECUTABLE / `RESTART_WORKLOAD` | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 13 | ROBUST_CHANNEL_GR_EXCEPTION / ROBUST_CHANNEL_GR_ERROR_SW_NOTIFY | A100,H100,B100,GB200 | RESTART_APP | WORKFLOW_XID_13 |  | EXECUTABLE / `RESTART_WORKLOAD` | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 14 | ROBUST_CHANNEL_FAKE_ERROR | A100,H100,B100,GB200 | IGNORE | CONTACT_SUPPORT |  | MONITOR_ONLY / `NO_ACTION` | INFO marker only, no workflow built |
| 25 | ROBUST_CHANNEL_GR_ILLEGAL_NOTIFY | A100,H100,B100,GB200 | RESTART_APP | CHECK_APP/CUDA |  | EXECUTABLE / `RESTART_WORKLOAD` | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 31 | ROBUST_CHANNEL_FIFO_ERROR_MMU_ERR_FLT | A100,H100,B100,GB200 | RESTART_APP | WORKFLOW_XID_31 |  | EXECUTABLE / `RESTART_WORKLOAD` | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 32 | ROBUST_CHANNEL_PBDMA_ERROR | A100,H100,B100,GB200 | RESTART_APP | CHECK_APP/CUDA |  | EXECUTABLE / `RESTART_WORKLOAD` | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 37 | ROBUST_CHANNEL_FECS_ERR_UNIMP_FIRMWARE_METHOD | A100,H100,B100,GB200 | IGNORE | CHECK_APP/CUDA |  | MONITOR_ONLY / `NO_ACTION` | INFO marker only, no workflow built |
| 38 | ROBUST_CHANNEL_FECS_ERR_WATCHDOG_TIMEOUT | A100,H100,B100,GB200 | IGNORE | CONTACT_SUPPORT |  | MONITOR_ONLY / `NO_ACTION` | INFO marker only, no workflow built |
| 39 | ROBUST_CHANNEL_CE0_ERROR | A100,H100,B100,GB200 | RESTART_APP | CONTACT_SUPPORT |  | EXECUTABLE / `RESTART_WORKLOAD` | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 40 | ROBUST_CHANNEL_CE1_ERROR | A100,H100,B100,GB200 | RESTART_APP | CONTACT_SUPPORT |  | EXECUTABLE / `RESTART_WORKLOAD` | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 41 | ROBUST_CHANNEL_CE2_ERROR | A100,H100,B100,GB200 | RESTART_APP | CONTACT_SUPPORT |  | EXECUTABLE / `RESTART_WORKLOAD` | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 43 | ROBUST_CHANNEL_RESETCHANNEL_VERIF_ERROR | A100,H100,B100,GB200 | IGNORE | CONTACT_SUPPORT |  | MONITOR_ONLY / `NO_ACTION` | INFO marker only, no workflow built |
| 44 | ROBUST_CHANNEL_GR_FAULT_DURING_CTXSW | A100,H100,B100,GB200 | IGNORE | CONTACT_SUPPORT |  | MONITOR_ONLY / `NO_ACTION` | INFO marker only, no workflow built |
| 45 | ROBUST_CHANNEL_PREEMPTIVE_REMOVAL | A100,H100,B100,GB200 | WORKFLOW_XID_45 | Solo: RESTART_FM / Not Solo: IGNORE (follow other Xid) |  | PENDING_CORRELATION 30s; when accompanied by other XIDs, follow the decision of the most severe accompanying XID and merge incidents; solo: EXECUTABLE `RESTART_FM` | solo: FREEZE_EVIDENCE → RESTART_FABRIC_MANAGER (Node Agent `systemctl restart nvidia-fabricmanager`) |
| 46 | ROBUST_CHANNEL_GPU_TIMEOUT_ERROR | A100,H100,B100,GB200 | RESET_GPU | CONTACT_SUPPORT |  | EXECUTABLE / `RESET_GPU` | GPU reset chain A |
| 48 | ROBUST_CHANNEL_GPU_ECC_DBE | A100,H100,B100,GB200 | WORKFLOW_XID_48 | WORKFLOW_XID_48 | CUDA 12.7; R565 | solo: `RESET_GPU`; accompanied by XID 63/64 within 30s: `DRAIN_AND_RESET` (RESET_GPU + cordon beforehand), sharing the incident with XID 64 to avoid a duplicate reset | GPU reset chain A |
| 54 | SILENT_RUNNING_PWR_REDUCED_CLOCKING | A100,H100,B100 | CHECK_MECHANICALS | CONTACT_SUPPORT |  | EXECUTABLE `CHECK_MECHANICALS`, requires_operator | FREEZE_EVIDENCE → CHECK_MECHANICALS (mail notification, wait for manual annotate confirmation, no cordon, jobs not stopped) |
| 60 | ROBUST_CHANNEL_SEC2_ERROR | A100,H100,B100,GB200 | RESTART_APP | INVESTIGATE_SW |  | EXECUTABLE / `RESTART_WORKLOAD` | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 62 | PMU_HALT_ERROR | A100,H100,B100,GB200 | RESET_GPU | CONTACT_SUPPORT | CUDA 12.7; R565 | EXECUTABLE / `RESET_GPU` | GPU reset chain A |
| 63 | INFOROM_DRAM_RETIREMENT_EVENT | A100,H100,B100,GB200 | IGNORE | IGNORE | CUDA 12.7; R565 | MONITOR_ONLY / `NO_ACTION` | INFO marker only, no workflow built |
| 64 | INFOROM_DRAM_RETIREMENT_FAILURE | A100,H100,B100,GB200 | RESET_GPU | CONTACT_SUPPORT | CUDA 12.7; R565 | EXECUTABLE / `RESET_GPU`; links back to an XID 48 within the previous 30s and shares its incident | GPU reset chain A |
| 66 | ROBUST_CHANNEL_FECS_ERR_REG_ACCESS_VIOLATION | A100,H100,B100,GB200 | IGNORE | INVESTIGATE_SW |  | MONITOR_ONLY / `NO_ACTION` | INFO marker only, no workflow built |
| 67 | ROBUST_CHANNEL_FECS_ERR_VERIF_VIOLATION | A100,H100,B100,GB200 | IGNORE | CONTACT_SUPPORT |  | MONITOR_ONLY / `NO_ACTION` | INFO marker only, no workflow built |
| 68 | ROBUST_CHANNEL_NVDEC0_ERROR | A100,H100,B100,GB200 | RESTART_APP | CONTACT_SUPPORT |  | EXECUTABLE / `RESTART_WORKLOAD` | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 69 | ROBUST_CHANNEL_GR_CLASS_ERROR | A100,H100,B100,GB200 | RESTART_APP | CHECK_APP/CUDA |  | EXECUTABLE / `RESTART_WORKLOAD` | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 70 | ROBUST_CHANNEL_CE3_ERROR | A100,H100,B100,GB200 | RESTART_APP | CONTACT_SUPPORT |  | EXECUTABLE / `RESTART_WORKLOAD` | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 71 | ROBUST_CHANNEL_CE4_ERROR | A100,H100,B100,GB200 | RESTART_APP | CONTACT_SUPPORT |  | EXECUTABLE / `RESTART_WORKLOAD` | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 72 | ROBUST_CHANNEL_CE5_ERROR | A100,H100,B100,GB200 | RESTART_APP | CONTACT_SUPPORT |  | EXECUTABLE / `RESTART_WORKLOAD` | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 74 | NVLINK_ERROR | A100,H100 | WORKFLOW_NVLINK_ERR | CONTACT_SUPPORT | CUDA 12.7; R565 | Only the H100 family decodes the 7 register bits: only safe_ignore/corrected_threshold bits → `NO_ACTION`; repeats on the same link reaching the threshold (ecc_parity ≥3; report_if_repeated, field_diag_if_repeated, mechanical ≥2) or mechanical_or_hardware → `RESET_GPU`; unknown bits, secondary appearing alone, unexpected_production, marginal_channel, fabric_reset_required → `ESCALATE_OPERATOR`; link-level bits missing nvlink_link_id → BLOCKED + QUARANTINE; registers ≠7 → BLOCKED + STOP_WORKLOAD; non-H100 family → `ESCALATE_OPERATOR` | reset path: cordon → STOP → [COLLECT_DIAGNOSTIC_BUNDLE] → QUIESCE → VERIFY_NO_GPU_CLIENTS → [RUN_NVLINK74_WORKFLOW; fails closed when nvlinkDiagnostics=OBSERVE in the example profile] → RESET_GPU → RESTORE_GPU_SERVICES → VALIDATE_GPU → VALIDATE_FABRIC → RESTORE_SCHEDULING → RESTART_WORKLOAD; escalation path: [cordon+stop] → QUARANTINE → ESCALATE_SUPPORT; fabric_reset_required: COLLECT_DIAGNOSTIC_BUNDLE → ESCALATE_SUPPORT |
| 75 | ROBUST_CHANNEL_CE6_ERROR | A100,H100,B100,GB200 | RESTART_APP | CONTACT_SUPPORT |  | EXECUTABLE / `RESTART_WORKLOAD` | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 76 | ROBUST_CHANNEL_CE7_ERROR | A100,H100,B100,GB200 | RESTART_APP | CONTACT_SUPPORT |  | EXECUTABLE / `RESTART_WORKLOAD` | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 77 | ROBUST_CHANNEL_CE8_ERROR | A100,H100,B100,GB200 | RESTART_APP | CONTACT_SUPPORT |  | EXECUTABLE / `RESTART_WORKLOAD` | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 78 | VGPU_START_ERROR | A100,H100,B100,GB200 | UPDATE_SWFW | UPDATE_SWFW |  | EXECUTABLE `UPDATE_SWFW` | cordon → [CHECKPOINT] → STOP → QUIESCE → VERIFY_NO_GPU_CLIENTS → UPDATE_SOFTWARE_FIRMWARE → RESTORE_GPU_SERVICES → VALIDATE_GPU → VALIDATE_HOST → RESTORE_SCHEDULING → RESTART_WORKLOAD; requires GPU_FAULT_TARGET_FIRMWARE_VERSION; with softwareFirmwareUpdate=OBSERVE in the example profile → no owner, fails closed |
| 79 | ROBUST_CHANNEL_GPU_HAS_FALLEN_OFF_THE_BUS | A100,H100,B100,GB200 | RESTART_BM | CONTACT_SUPPORT | CUDA 12.7; R565 | EXECUTABLE / `REBOOT_NODE` | Node reboot chain B |
| 80 | PBDMA_PUSHBUFFER_CRC_MISMATCH | A100,H100 | RESTART_APP | CHECK_APP/CUDA |  | EXECUTABLE / `RESTART_WORKLOAD` | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 82 | ROBUST_CHANNEL_NVJPG0_ERROR | A100,H100,B100,GB200 | RESTART_APP | CONTACT_SUPPORT |  | EXECUTABLE / `RESTART_WORKLOAD` | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 83 | ROBUST_CHANNEL_NVDEC1_ERROR | A100,H100,B100,GB200 | RESTART_APP | CONTACT_SUPPORT |  | EXECUTABLE / `RESTART_WORKLOAD` | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 84 | ROBUST_CHANNEL_NVDEC2_ERROR | A100,H100,B100,GB200 | RESTART_APP | CONTACT_SUPPORT |  | EXECUTABLE / `RESTART_WORKLOAD` | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 85 | ROBUST_CHANNEL_CE9_ERROR | A100,H100,B100,GB200 | RESTART_APP | CONTACT_SUPPORT |  | EXECUTABLE / `RESTART_WORKLOAD` | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 86 | ROBUST_CHANNEL_OFA0_ERROR | A100,H100,B100,GB200 | RESTART_APP | CONTACT_SUPPORT |  | EXECUTABLE / `RESTART_WORKLOAD` | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 88 | ROBUST_CHANNEL_NVDEC3_ERROR | A100,H100,B100,GB200 | RESTART_APP | CONTACT_SUPPORT |  | EXECUTABLE / `RESTART_WORKLOAD` | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 89 | ROBUST_CHANNEL_NVDEC4_ERROR | A100,H100,B100,GB200 | RESTART_APP | CONTACT_SUPPORT |  | EXECUTABLE / `RESTART_WORKLOAD` | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 92 | EXCESSIVE_SBE_INTERRUPTS | A100,H100,B100,GB200 | IGNORE | CONTACT_SUPPORT |  | MONITOR_ONLY / `NO_ACTION` | INFO marker only, no workflow built |
| 93 | INFOROM_ERASE_LIMIT_EXCEEDED | A100 | IGNORE | CONTACT_SUPPORT |  | MONITOR_ONLY / `NO_ACTION` | INFO marker only, no workflow built |
| 94 | ROBUST_CHANNEL_CONTAINED_ERROR | A100,H100,B100,GB200 | RESTART_APP | IGNORE (sympathetic) | CUDA 12.7; R565 | EXECUTABLE / `RESTART_WORKLOAD`, containment APPLICATION | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 95 | ROBUST_CHANNEL_UNCONTAINED_ERROR | A100,H100,B100,GB200 | RESET_GPU | IGNORE (sympathetic) | CUDA 12.7; R565 | EXECUTABLE / `RESET_GPU`, pre_actions MARK_UNSCHEDULABLE + STOP_WORKLOAD, containment ALL_APPLICATIONS | GPU reset chain A |
| 96-98 | ROBUST_CHANNEL_NVDEC5/6/7_ERROR | H100,B100,GB200 | RESTART_APP | CONTACT_SUPPORT |  | EXECUTABLE / `RESTART_WORKLOAD` | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 99-105 | ROBUST_CHANNEL_NVJPG1..7_ERROR | H100,B100,GB200 | RESTART_APP | CONTACT_SUPPORT |  | EXECUTABLE / `RESTART_WORKLOAD` | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 106 | SMBPBI_TEST_MESSAGE | A100,H100,B100,GB200 | IGNORE | IGNORE |  | MONITOR_ONLY / `NO_ACTION` | INFO marker only, no workflow built |
| 107 | SMBPBI_TEST_MESSAGE_SILENT | A100,H100,B100,GB200 | IGNORE | IGNORE |  | MONITOR_ONLY / `NO_ACTION` | INFO marker only, no workflow built |
| 108 | NVLINK_REMOTE_TRANSLATION_ERROR | A100,H100,B100,GB200 | IGNORE | XID_137_FLOW |  | MONITOR_ONLY / `NO_ACTION` | INFO marker only, no workflow built |
| 109 | ROBUST_CHANNEL_CTXSW_TIMEOUT_ERROR | A100,H100,B100,GB200 | RESET_GPU | CONTACT_SUPPORT | CUDA 12.7; R570 | EXECUTABLE / `RESET_GPU` | GPU reset chain A |
| 110 | SEC_FAULT_ERROR | H100,B100,GB200 | RESET_GPU | INVESTIGATE_SW | CUDA 12.7; R565 | EXECUTABLE / `RESET_GPU` | GPU reset chain A |
| 119 | GSP_RPC_TIMEOUT | A100,H100,B100,GB200 | RESET_GPU | INVESTIGATE_SW |  | EXECUTABLE / `RESET_GPU` | GPU reset chain A |
| 120 | GSP_ERROR | A100,H100,B100,GB200 | RESET_GPU | INVESTIGATE_SW | CUDA 12.7; R565 | EXECUTABLE / `RESET_GPU` | GPU reset chain A |
| 121 | C2C_ERROR | GB200 | IGNORE | CONTACT_SUPPORT |  | MONITOR_ONLY / `NO_ACTION` | INFO marker only, no workflow built |
| 126-135 | ROBUST_CHANNEL_CE10..19_ERROR | B100,GB200 | RESTART_APP | CONTACT_SUPPORT |  | EXECUTABLE / `RESTART_WORKLOAD` | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 136 | ALI_TRAINING_FAIL | H100 | RESET_GPU | INVESTIGATE_LINK_SI | CUDA 12.7; R565 | EXECUTABLE / `RESET_GPU` | GPU reset chain A |
| 137 | NVLINK_PRIV_ERR | A100,H100,B100,GB200 | IGNORE | XID_137_FLOW |  | MONITOR_ONLY / `NO_ACTION` | INFO marker only, no workflow built |
| 139 | ROBUST_CHANNEL_OFA1_ERROR | B100,GB200 | RESTART_APP | CONTACT_SUPPORT |  | EXECUTABLE / `RESTART_WORKLOAD` | FREEZE_EVIDENCE → STOP_WORKLOADS → RESTART_WORKLOAD (subject to the restart budget) |
| 140 | UNRECOVERABLE_ECC_ERROR_ESCAPE | A100,H100,B100,GB200 | RESET_GPU | CONTACT_SUPPORT |  | EXECUTABLE / `RESET_GPU` | GPU reset chain A |
| 141 | ROBUST_CHANNEL_FAST_PATH_ERROR | H100,B100,GB200 | IGNORE | CONTACT_SUPPORT |  | MONITOR_ONLY / `NO_ACTION` | INFO marker only, no workflow built |
| 142 | ROBUST_CHANNEL_NVENC3_ERROR | GB200 | CONTACT_SUPPORT | (none) |  | EXECUTABLE `CONTACT_SUPPORT`, requires_operator | FREEZE_EVIDENCE → ESCALATE_SUPPORT (mail escalation, no cordon) |
| 143 | GPU_INIT_ERROR | H100,B100,GB200 | RESET_GPU | CONTACT_SUPPORT | CUDA 12.9; R575 | EXECUTABLE / `RESET_GPU` | GPU reset chain A |
| 144 | NVLINK_SAW_ERROR | B100,GB200 | WORKFLOW_NVLINK5_ERR | WORKFLOW_NVLINK5_ERR | CUDA 12.7; R565 | NVLink5 decode (see notes above); 6 rules: errorStatus 0x2/0x10 → `RESET_GPU`, otherwise `NO_ACTION` | GPU reset chain A / no action / BLOCKED+QUARANTINE |
| 145 | NVLINK_RLW_ERROR | B100,GB200 | WORKFLOW_NVLINK5_ERR | WORKFLOW_NVLINK5_ERR | CUDA 12.7; R565 | 33 rules: RLW_REMAP 0x1-0x20, RLW_SRC_TRACK 0x4/0x8 → XID_154_EVAL (follow the accompanying XID 154 action, RESTART_WORKLOAD when there is no 154); RLW_RXPIPE 0x1/0x2/0x4 IGNORE but an Action2 bit hit → `RESET_GPU`; each Fatal item → `RESET_GPU`; otherwise `NO_ACTION` | GPU reset chain A / no action / BLOCKED+QUARANTINE |
| 146 | NVLINK_TLW_ERROR | B100,GB200 | WORKFLOW_NVLINK5_ERR | WORKFLOW_NVLINK5_ERR | CUDA 12.7; R565 | 20 rules: each sub-unit errorStatus 0x4 → `RESET_GPU`; 0x1/0x2/0x80000000 → `NO_ACTION` | GPU reset chain A / no action / BLOCKED+QUARANTINE |
| 147 | NVLINK_TREX_ERROR | B100,GB200 | WORKFLOW_NVLINK5_ERR | WORKFLOW_NVLINK5_ERR | CUDA 12.7; R565 | 2 rules: all IGNORE → `NO_ACTION` | INFO marker only |
| 148 | NVLINK_NVLPW_CTRL_ERROR | B100,GB200 | WORKFLOW_NVLINK5_ERR | WORKFLOW_NVLINK5_ERR | CUDA 12.7; R565 | 1 rule: IGNORE → `NO_ACTION` | INFO marker only |
| 149 | NVLINK_NETIR_ERROR | B100,GB200 | WORKFLOW_NVLINK5_ERR | WORKFLOW_NVLINK5_ERR | CUDA 12.7; R565 | 31 rules: most of NETIR_INT / NETIR_LINK_DOWN / NETIR_MFDE → `RESET_GPU`; NETIR_BER_EVENT and 2 LINK_DOWN entries → `NO_ACTION` | GPU reset chain A / no action |
| 150 | NVLINK_MSE_ERROR | B100,GB200 | WORKFLOW_NVLINK5_ERR | WORKFLOW_NVLINK5_ERR | CUDA 12.7; R565 | 2 rules (MSE Degraded / MSE_WATCHDOG) → `RESET_GPU` | GPU reset chain A |
| 151 | ROBUST_CHANNEL_KEY_ROTATION_ERROR | H100,B100,GB200 | RESTART_VM | CONTACT_SUPPORT |  | The engine decides BLOCKED_WORKFLOW + QUARANTINE; under the HyperPod profile the coordinator sets effective_action to `REBOOT_NODE` (BatchRebootClusterNodes, not EC2 stop/start) | Node reboot chain B |
| 152 | ROBUST_CHANNEL_DLA_SMMU_ERROR | (none) | IGNORE | CONTACT_SUPPORT |  | NOT_APPLICABLE + safety `QUARANTINE` (no applicable product, gate fails closed) | FREEZE_EVIDENCE → QUARANTINE |
| 153 | ROBUST_CHANNEL_DLA_TIMEOUT | (none) | IGNORE | CONTACT_SUPPORT |  | NOT_APPLICABLE + safety `QUARANTINE` (no applicable product, gate fails closed) | FREEZE_EVIDENCE → QUARANTINE |
| 154 | GPU_RECOVERY_ACTION_CHANGED | A100,H100,B100,GB200 | XID_154 | N/A Informational only regarding another Xid |  | By the driver-reported label: none → `NO_ACTION`; drain p2p → `STOP_WORKLOAD`; drain and reset → `RESET_GPU`(+cordon+stop); gpu reset required → `RESET_GPU`; node reboot required → `REBOOT_NODE`; label missing → BLOCKED + QUARANTINE; 30s window not closed → PENDING_CORRELATION. This action overrides the Immediate Action of accompanying XIDs on the same GPU within 30s | Corresponding STOP_WORKLOADS / chain A / chain B |
| 155 | NVLINK_SW_DEFINED_ERROR | B100,GB200 | RESET_GPU | INVESTIGATE_SW_USER | CUDA 12.7; R565 | EXECUTABLE / `RESET_GPU` | GPU reset chain A |
| 156 | RESOURCE_RETIREMENT_EVENT | H100,B100,GB200 | RESET_GPU | IGNORE | CUDA 12.7; R565 | EXECUTABLE / `RESET_GPU` | GPU reset chain A |
| 157 | RESOURCE_RETIREMENT_FAILURE | H100,B100,GB200 | IGNORE | CONTACT_SUPPORT |  | MONITOR_ONLY / `NO_ACTION` | INFO marker only, no workflow built |
| 158 | GPU_FATAL_TIMEOUT | A100,H100,B100,GB200 | RESET_GPU | CONTACT_SUPPORT |  | EXECUTABLE / `RESET_GPU` | GPU reset chain A |
| 159 | ROBUST_CHANNEL_CHI_NON_DATA_ERROR | B100,GB200 | CHECK_UVM | SYMPATHETIC_REPORT_SOLO |  | uvm_in_use=true → `RESET_GPU`; false → `NO_ACTION`; missing evidence → BLOCKED + QUARANTINE | GPU reset chain A / no action |
| 160 | CHANNEL_RETIREMENT_EVENT | B100,GB200 | IGNORE | INVESTIGATE_SW | CUDA 12.9; R575 | MONITOR_ONLY / `NO_ACTION` | INFO marker only, no workflow built |
| 161 | CHANNEL_RETIREMENT_FAILURE | B100,GB200 | IGNORE | INVESTIGATE_SW | CUDA 12.9; R575 | MONITOR_ONLY / `NO_ACTION` | INFO marker only, no workflow built |
| 162-165 | PSHC_REENGAGED / DISENGAGED / LOW_LIFETIME / ZERO_LIFETIME | B100,GB200 | (none) | (none) |  | MONITOR_ONLY / `NO_ACTION` (no action defined in the catalog) | INFO marker only |
| 166 | NVLINK_SECURE_CRYPTO_ERR | B100,GB200 | (none) | (none) |  | MONITOR_ONLY / `NO_ACTION` (no action defined in the catalog) | INFO marker only |
| 167 | PCIE_FATAL_TIMEOUT | H100,B100,GB200 | (none) | (none) |  | MONITOR_ONLY / `NO_ACTION` (no action defined in the catalog) | INFO marker only |
| 168 | REDUCED_GPU_MEMORY_CAPACITY | A100,H100,B100,GB200 | (none) | (none) |  | MONITOR_ONLY / `NO_ACTION` (no action defined in the catalog) | INFO marker only |
| 169 | SEC2_HALT_ERROR | H100,B100,GB200 | (none) | (none) |  | MONITOR_ONLY / `NO_ACTION` (no action defined in the catalog) | INFO marker only |
| 170 | NVLINK_SECURE_OTHER | B100,GB200 | (none) | (none) |  | MONITOR_ONLY / `NO_ACTION` (no action defined in the catalog) | INFO marker only |
| 171 | UNCORRECTABLE_DRAM_ERROR | A100,H100,B100,GB200 | (none) | (none) |  | MONITOR_ONLY / `NO_ACTION` (no action defined in the catalog) | INFO marker only |
| 172 | UNCORRECTABLE_SRAM_ERROR | A100,H100,B100,GB200 | (none) | (none) |  | MONITOR_ONLY / `NO_ACTION` (no action defined in the catalog) | INFO marker only |

## III. SXID table (Fabric Manager User Guide 2025-11-14, Tables 21-24)

Preliminary gates: GPU is of the B100/GB200 family → NOT_APPLICABLE + QUARANTINE (DCGM/NVSDM telemetry is required instead); product unrecognised → BLOCKED; classification_source must come from the Fabric Manager tables; the fatal/non-fatal observed on the event conflicting with the table → BLOCKED. The workflows for FATAL/ALWAYS_FATAL and for 10003/19084 all prepend COLLECT_DIAGNOSTIC_BUNDLE. link_scope is determined by the NVSwitch port topology (`TRUSTED_NVSWITCH_TOPOLOGY`) or by the product invariant of HGX H100/H200 without trunks (`NVIDIA_PRODUCT_INVARIANT`); other sources are untrusted.

| SXID | NVIDIA classification | NVIDIA officialAction | NVIDIA Investigatory Action | Applicability | This solution's decision | This solution's actual execution |
|---|---|---|---|---|---|---|
| 11004 | NON_FATAL | RESTART_VM | VALIDATE_NVSWITCH_FABRIC_ROUTING | SHARED_NVSWITCH_OR_VGPU | BLOCKED_WORKFLOW + safety `QUARANTINE` (bare metal has no guest VM lifecycle; SXID is not rewritten to REBOOT_NODE) | FREEZE_EVIDENCE → QUARANTINE, wait for a human |
| 12028 | NON_FATAL | RESTART_VM | NONE | SHARED_NVSWITCH_OR_VGPU | Same as above | Same as above |
| 11012, 11021, 11022, 11023, 12021, 12023, 15008, 15011, 19049, 19055, 19057, 19059, 19062, 19065, 19068, 19071, 24001, 24002, 24003 | NON_FATAL | IGNORE | MONITOR_CORRECTED_ECC | ALL | MONITOR_ONLY / `NO_ACTION` | INFO marker only; the notification includes the investigatory action |
| 20001 | NON_FATAL | IGNORE | MONITOR_NVLINK_THROUGHPUT | ALL | MONITOR_ONLY / `NO_ACTION` | Same as above |
| 22013 | NON_FATAL | IGNORE | NONE | ALL | MONITOR_ONLY / `NO_ACTION` | Same as above |
| 10001, 10002 | NON_FATAL | IGNORE | CORRELATE_FOLLOWING_FATAL_EVENTS | ALL | MONITOR_ONLY / `NO_ACTION` | Same as above |
| 10004 | NON_FATAL | IGNORE | CHECK_SYSTEM_COOLING | ALL | MONITOR_ONLY / `NO_ACTION` | Same as above |
| 10005 | NON_FATAL | IGNORE | VERIFY_THERMAL_EVENT_CLEARED | ALL | MONITOR_ONLY / `NO_ACTION` | Same as above |
| 20012 | NON_FATAL | CHECK_MECHANICALS | CHECK_LINK_MECHANICAL_CONNECTIONS | ALL | EXECUTABLE / `ESCALATE_OPERATOR` (official CHECK_MECHANICALS, requires_operator) | FREEZE_EVIDENCE → CHECK_MECHANICALS (mail + wait for manual annotate confirmation, no cordon, jobs not stopped) |
| 19084 | NON_FATAL | RESET_ALL_GPUS_AND_NVSWITCHES | CORRELATE_COMPANION_FATAL_SXID | ALL | EXECUTABLE `RESET_ALL_GPUS_AND_NVSWITCHES` (a full reset is executed even though NON_FATAL); missing fabric_partition or the complete node GPU inventory → BLOCKED + QUARANTINE; pre_actions cordon + stop | FREEZE_EVIDENCE → COLLECT_DIAGNOSTIC_BUNDLE → MARK_UNSCHEDULABLE → [CHECKPOINT] → STOP_WORKLOADS → QUIESCE_GPU_SERVICES → VERIFY_NO_GPU_CLIENTS → [REMEDIATE_DRIVER / UPDATE_SOFTWARE_FIRMWARE, only when the code is listed in GPU_FAULT_SXID_DRIVER_REMEDIATION_CODES / _FIRMWARE_UPDATE_CODES] → RESET_ALL_GPUS_NVSWITCHES (the Node Agent stops FM then `nvidia-smi --gpu-reset` on all cards, requiring the UUID set to match the local inventory, multi-node barrier) → RESTORE_GPU_SERVICES → VALIDATE_GPU → VALIDATE_FABRIC → RESTORE_SCHEDULING → [RESTART_WORKLOAD] |
| 10003 | FATAL | RESET_ALL_GPUS_AND_NVSWITCHES | NONE | ALL | Same as 19084 | Same as 19084 |
| 11001, 11009, 11013, 11018, 11019, 11020, 12001, 12002, 12022, 12024, 12025, 12026, 12027, 12030, 12031, 12032, 14017, 15001, 15006, 15009, 15010, 15012, 15013, 19047, 19048, 19054, 19056, 19058, 19060, 19061, 19063, 19064, 19066, 19067, 19069, 19070, 20034, 22012, 24004, 24005, 24006, 24007 | FATAL | SCOPE_DEPENDENT_RESET | NONE | ALL | By link_scope: ACCESS → EXECUTABLE / `RESET_GPU` (official RESET_PARTICIPATING_GPUS: the affected GPU + all participating GPUs of the same job; missing participating_gpu_uuids → BLOCKED); TRUNK → EXECUTABLE `RESET_ALL_GPUS_AND_NVSWITCHES` (missing fabric_partition / inventory → BLOCKED); UNKNOWN or untrusted source → BLOCKED + QUARANTINE; all with pre_actions cordon + stop | ACCESS: FREEZE_EVIDENCE → COLLECT_DIAGNOSTIC_BUNDLE → MARK_UNSCHEDULABLE → [CHECKPOINT] → STOP_WORKLOADS → QUIESCE → VERIFY_NO_GPU_CLIENTS → RESET_GPU (participating GPU group) → RESTORE_GPU_SERVICES → VALIDATE_GPU → RESTORE_SCHEDULING → [RESTART_WORKLOAD]; TRUNK: same chain as 19084 |
| 12020, 22003, 22011, 23001, 23002, 23003, 23004, 23005, 23006, 23007, 23008, 23009, 23010, 23011, 23012, 23013, 23014, 23015, 23016, 23017 | ALWAYS_FATAL | RESTART_HOST_OR_SBR | NONE | ALL | EXECUTABLE / `REBOOT_NODE` (official RESTART_BM; the 20 codes of Table 23 are pinned; an event declared Always-Fatal but not in the table → BLOCKED); pre_actions cordon + stop; containment NODE | FREEZE_EVIDENCE → COLLECT_DIAGNOSTIC_BUNDLE → MARK_UNSCHEDULABLE → [CHECKPOINT] → STOP_WORKLOADS → RESTART_NODE (BatchRebootClusterNodes) → VALIDATE_GPU → VALIDATE_HOST → VALIDATE_FABRIC → RESTORE_SCHEDULING → [RESTART_WORKLOAD] |
| SXID codes outside the tables | Per the fatal/non-fatal in the FM log | (none) | (none) | (none) | NON_FATAL → MONITOR_ONLY / `NO_ACTION` (investigatory is MONITOR_PROGRESS_AND_PERFORMANCE); FATAL → the SCOPE_DEPENDENT path above | Same as the corresponding row |

## IV. Implementation notes

- `CONTACT_SUPPORT`, `CHECK_MECHANICALS` and `UPDATE_SWFW` are EXECUTABLE rather than BLOCKED at the policy layer:
  `CONTACT_SUPPORT` compiles to `ESCALATE_SUPPORT` (fixed-template mail, no cordon), `CHECK_MECHANICALS`
  compiles to a notification and waits for manual confirmation, `UPDATE_SWFW` compiles to the complete shutdown chain containing `UPDATE_SOFTWARE_FIRMWARE`.
  Only missing evidence, inapplicable product, unknown XID and workflows without a resolver go to BLOCKED + `QUARANTINE`.
- Always-Fatal SXIDs execute `REBOOT_NODE` directly; fatal trunk and 10003/19084 execute
  `RESET_ALL_GPUS_AND_NVSWITCHES` directly. Both require fabric_partition and the complete node GPU inventory;
  missing either → BLOCKED.
- XID 152/153 are officially IGNORE, but the catalog lists no applicable product, so the product gate decides NOT_APPLICABLE and isolates the node,
  stricter than the official recommendation.
- The example profile declares `nvlinkDiagnostics`, `memoryDiagnostics`, `driverRemediation` and
  `softwareFirmwareUpdate` as OBSERVE, so the field-diagnostic branch of XID 74, XID 78,
  the `RUN_FIELD_DIAGNOSTIC` appended by `row_remap_failure`, and the SXID driver/firmware repair steps under that profile
  fail closed with "no executable owner" and are handed to a human.
- While a node is inside a reboot window initiated by the product itself (a `RESTART_NODE`/`REPLACE_NODE` step has been dispatched and the subsequent
  `RESTORE_SCHEDULING` has not yet completed), kmsg XID/SXID and node-health markers on that node no longer open a new incident
  but are linked as evidence to the incident that owns the reboot; kmsg events carrying the old boot_id (the agent already heartbeats with the new boot) are treated as
  pre-reboot residue, land as RECOVERED and produce no workflow. The rules are in `orchestration/reboot_window.py`.
