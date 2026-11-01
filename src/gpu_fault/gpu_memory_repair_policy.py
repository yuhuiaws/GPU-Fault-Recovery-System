"""Blackwell GPU memory repair: the repair-pending and unrepairable flags (batch 235).

**Sources.** The DCGM collector reads ``<channel_repair_pending>``,
``<tpc_repair_pending>`` and ``<unrepairable_memory>`` from ``nvidia-smi -q -x``
at the inventory cadence (``collectors/gpu/smi_health``; the tag names are the
live H200 R580 output, driver 580.178.04, XML lines 194-196 of the per-GPU
memory section, all ``No``) and publishes ``Yes`` as 1, ``No`` as 0 and
``N/A`` or a missing tag as nothing. DCGM field 1507
(``DCGM_FI_DEV_MEMORY_UNREPAIRABLE_FLAG``, DCGM 4.6 name
``DCGM_FI_DEV_MEMORY_UNREPAIRABLE``) is the second source of the unrepairable
flag; it needs NVML v590 (NVML change log, "Changes between v580 and v590":
:data:`NVML_UNREPAIRABLE_CHANGE_LOG_QUOTE`) and reads N/A on R580, while the
R580 nvidia-smi already prints the tag (live; nvidia-smi's own change log lists
it under "Changes between nvidia-smi v590 Update and v580"). A GPU reporting
both is published once, at the higher reading. Exported, never required.

Evidence -- every quote checked against the live page named:

* NVIDIA GPU Memory Error Management (docs version 615), "RAS Repair" >
  "GPU Memory Repair" (:data:`NVIDIA_MEMORY_REPAIR_REFERENCE`):
  :data:`NVIDIA_MEMORY_REPAIR_QUOTE` and :data:`NVIDIA_REPAIR_REBOOT_QUOTE`.
* nvidia-smi(1) (v610-era page), "ECC Errors"
  (:data:`NVIDIA_SMI_REFERENCE`): :data:`NVIDIA_SMI_REPAIR_QUOTE`, and the flag
  lines of :data:`NVIDIA_SMI_FLAG_QUOTES`; change log "Changes between
  nvidia-smi v580 Update and v575": :data:`NVIDIA_SMI_REPAIR_CHANGE_LOG_QUOTE`.
* DCGM Field Identifiers (docs "Version: 4.6 (Latest)"),
  ``DCGM_FI_DEV_MEMORY_UNREPAIRABLE 1507``: :data:`DCGM_UNREPAIRABLE_QUOTE`;
  DCGM v4.7.0 ``DcgmHealthWatch::MonitorMemUnrepairableFlag`` FAILs on any
  non-zero value with ``DCGM_FR_FAULTY_MEMORY`` (next step "Run a field
  diagnostic on the GPU.").

**Grades.**

* Channel or TPC repair pending (1) while the GPU's recovery action (DCGM field
  1523, the same batch) is None: WARNING ``ESCALATE_OPERATOR``, official
  ``REBOOT_IN_SERVICE_WINDOW`` -- the repair takes effect at the next reboot,
  which an operator schedules -- and a persistent advisory that VALIDATE_GPU
  does not fail on (``gpu_field_rules.ADVISORY_GPU_FINDING_NAMES``; only the
  reboot clears it). Without a 1523 reading the same advisory.
* Repair pending while 1523 is not None: evidence only (``NO_ACTION``) --
  NVIDIA: "check the GPU Recovery action and take the appropriate steps" --
  and batch 233's 1523 rule (``gpu_recovery_action_policy``) decides.
* Unrepairable memory (the XML ``Yes`` or field 1507 = 1): CRITICAL ``DRAIN``,
  official ``RUN_FIELD_DIAGNOSTIC_FOR_RMA``, on the memory path
  (``gpu_memory_policy.memory_decision``, the row-remap failure's RMA drain,
  including the containment prefix a ``memoryDiagnostics: OBSERVE`` profile
  runs before it blocks).
* XID 160 (``CHANNEL_RETIREMENT_EVENT``) and 161 (``CHANNEL_RETIREMENT_FAILURE``)
  keep their catalog action (IGNORE, B100/GB200 parts); an XID 160/161 of the
  GPU the node's history holds is named in these findings as evidence
  (:func:`repair_xid_evidence`).

Which recovery action accompanies a channel repair NVIDIA does not document,
nor whether every Blackwell part supports it (Supported GPUs: "Support is not
guaranteed for all Blackwell GPUs."). Unit-tested from NVIDIA documentation,
not live-verified (no B-series hardware).
"""

from __future__ import annotations

from typing import Any

from gpu_fault.gpu_memory_policy import (
    DCGM_UNREPAIRABLE_REFERENCE as DCGM_UNREPAIRABLE_REFERENCE,
)
from gpu_fault.gpu_metric_models import GpuHealthSeverity
from gpu_fault.gpu_pcie_link_policy import device_values
from gpu_fault.gpu_recovery_action_policy import (
    GPU_RECOVERY_ACTION,
    recovery_action_label,
)
from gpu_fault.models import IncidentState, RecoveryAction

MEMORY_CHANNEL_REPAIR_PENDING = "memory_channel_repair_pending"
MEMORY_TPC_REPAIR_PENDING = "memory_tpc_repair_pending"
MEMORY_UNREPAIRABLE = "memory_unrepairable"
REPAIR_PENDING_NAMES = frozenset(
    {MEMORY_CHANNEL_REPAIR_PENDING, MEMORY_TPC_REPAIR_PENDING}
)
#: The nvidia-smi XML tags -> the canonical names.
XML_REPAIR_TAGS = {
    "channel_repair_pending": MEMORY_CHANNEL_REPAIR_PENDING,
    "tpc_repair_pending": MEMORY_TPC_REPAIR_PENDING,
    "unrepairable_memory": MEMORY_UNREPAIRABLE,
}
REPAIR_OFFICIAL_ACTION = "REBOOT_IN_SERVICE_WINDOW"
NVIDIA_MEMORY_REPAIR_REFERENCE = (
    "https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/hbm-channel-repair.html"
    "#gpu-memory-repair"
)
NVIDIA_MEMORY_REPAIR_QUOTE = (
    "GPU memory repair is another resiliency feature available on select "
    "Blackwell products. GPU memory repair consists of swapping spare DRAM "
    "channels or L2 cache slices."
)
NVIDIA_REPAIR_REBOOT_QUOTE = (
    "XID 160 will be printed if a DRAM channel or L2 slice is successfully marked "
    "for repair. The process of channel repair requires a reboot to take effect."
)
NVIDIA_SMI_REFERENCE = "https://docs.nvidia.com/deploy/nvidia-smi/index.html"
NVIDIA_SMI_REPAIR_QUOTE = (
    "If one of the repair flags is pending, check the GPU Recovery action and take "
    "the appropriate steps."
)
NVIDIA_SMI_FLAG_QUOTES = {
    MEMORY_CHANNEL_REPAIR_PENDING: "Indicates if a Channel repair is pending",
    MEMORY_TPC_REPAIR_PENDING: "Indicates if a TPC repair is pending",
    MEMORY_UNREPAIRABLE: "Indicates if there is unrepairable memory",
}
NVIDIA_SMI_REPAIR_CHANGE_LOG_QUOTE = (
    "Added Channel/TPC repair pending flags to ECC field: 'nvidia-smi -q -d ECC'"
)
NVML_CHANGE_LOG_REFERENCE = (
    "https://docs.nvidia.com/deploy/nvml-api/latest/change-log.html"
)
NVML_UNREPAIRABLE_CHANGE_LOG_QUOTE = (
    "Added nvmlDeviceGetUnrepairableMemoryFlag_v1() to report whether or not there "
    "is unrepairable memory."
)
DCGM_UNREPAIRABLE_QUOTE = (
    "Unrepairable memory flag indicating if memory has unrepairable errors 1=yes, 0=no"
)
#: XID 160 / 161 (XID catalog, docs version 615): their catalog names.
REPAIR_XIDS = {160: "CHANNEL_RETIREMENT_EVENT", 161: "CHANNEL_RETIREMENT_FAILURE"}
REPAIR_XID_LOOKUP_INCIDENT_LIMIT = 50


def _decision_xid(decision: Any) -> int | None:
    reason = str(getattr(getattr(decision, "marker", None), "raw_reason", "") or "")
    kind, _, code = reason.partition(" ")
    return int(code) if kind == "XID" and code.isdigit() else None


def repair_xid_evidence(store: Any, batch: Any, sample: Any) -> str | None:
    """The newest XID 160/161 of the sample's GPU in the node's history.

    Read from the node's incidents (any state, newest first): XID 160/161 are
    IGNORE in the catalog, so their incidents end RECOVERED at once and outlive
    the XID event table's few minutes. ``None`` when there is none.
    """

    for incident in store.list_incidents_by_state(
        batch.cluster_id,
        set(IncidentState),
        node_ids={batch.node_id},
        limit=REPAIR_XID_LOOKUP_INCIDENT_LIMIT,
    ):
        if incident.event_type != "XID" or (
            incident.gpu_uuids and sample.gpu_uuid not in incident.gpu_uuids
        ):
            continue
        xid = _decision_xid(store.get_xid_policy_decision(incident.event_id))
        if xid in REPAIR_XIDS:
            return (
                f"XID {xid} {REPAIR_XIDS[xid]} at {incident.created_at.isoformat()} "
                f"(incident {incident.incident_id})"
            )
    return None


def with_repair_xid_evidence(
    store: Any, batch: Any, sample: Any, decision: dict[str, Any] | None
) -> dict[str, Any] | None:
    """``decision`` with the GPU's XID 160/161, when the history has one."""

    if decision is None or sample.canonical_name not in XML_REPAIR_TAGS.values():
        return decision
    evidence = repair_xid_evidence(store, batch, sample)
    if evidence is None:
        return decision
    return {**decision, "reason": f"{decision['reason']}; evidence: {evidence}"}


def repair_pending_decision(batch: Any, sample: Any) -> dict[str, Any] | None:
    """A channel or TPC repair pending (see the module docstring)."""

    if sample.canonical_name not in REPAIR_PENDING_NAMES or sample.value != 1:
        return None
    what = (
        "channel" if sample.canonical_name == MEMORY_CHANNEL_REPAIR_PENDING else "TPC"
    )
    action = device_values(batch, sample).get(GPU_RECOVERY_ACTION)
    common = {
        "severity": GpuHealthSeverity.WARNING,
        "threshold_value": 1.0,
        "policy_source": "SITE_GPU_MEMORY_EVIDENCE",
        "policy_reference": NVIDIA_MEMORY_REPAIR_REFERENCE,
    }
    if action is not None and action != 0:
        return {
            **common,
            "reason": (
                f"GPU memory {what} repair pending (nvidia-smi); the driver's GPU "
                f"recovery action {recovery_action_label(action)} (DCGM field "
                "1523) decides, as NVIDIA says for a pending repair flag"
            ),
            "automatic_action": RecoveryAction.NO_ACTION.value,
            "threshold_source": "NVIDIA_MEMORY_REPAIR_FOLLOWS_RECOVERY_ACTION",
        }
    reading = (
        "None (DCGM field 1523)"
        if action is not None
        else "not reported (no DCGM field 1523 reading)"
    )
    return {
        **common,
        "reason": (
            f"GPU memory {what} repair pending (nvidia-smi) with the GPU recovery "
            f"action {reading}: the repair takes effect at the next reboot; "
            "reboot the node in a service window"
        ),
        "automatic_action": RecoveryAction.ESCALATE_OPERATOR.value,
        "threshold_source": "NVIDIA_MEMORY_REPAIR_PENDING",
        "official_action": REPAIR_OFFICIAL_ACTION,
    }
