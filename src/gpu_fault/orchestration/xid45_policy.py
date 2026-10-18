"""A solo XID 45 on a graded NVSwitch node is observe-only by default (batch 131).

Evidence -- NVIDIA, both verified against the live pages:

* Xid catalog, docs version 615 (page "Last updated on Sep 09, 2026"):
  https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html and its
  spreadsheet
  https://docs.nvidia.com/deploy/xid-errors/_downloads/4586dadb59119a55d1e93a181caa4272/Xid-Catalog.xlsx
  (sha256 7f70ce9684be0c9d98a367770341ccc1d8bca4465cebe42dab7e23d57df5d5a5, the
  source of this product's pinned ``data/nvidia-xid-catalog-610.generated.yaml``;
  the catalog data is unchanged). "Xids" sheet, code 45: Immediate Action
  WORKFLOW_XID_45, Investigatory Action "Solo: RESTART_FM / Not Solo: IGNORE
  (follow other Xid)"; trigger: "This event is logged when the user application
  aborts and the kernel driver tears down the GPU application running on the
  GPU. Control-C, GPU resets, sigkill are all examples where the application is
  aborted and this event is created. In many cases, this is not indicative of a
  bug but rather a user or system action."
* GPU debug guidelines, GPU node triage, section "Understanding Xid Messages",
  row "45 Robust Channel Preemptive Removal", docs version 615 (page "Last
  updated on Oct 04, 2026"):
  https://docs.nvidia.com/deploy/gpu-debug-guidelines/gpu-node-triage.html#understanding-xid-messages
  -- "No action, informative only. Indicates channels affected by another
  failure. On A100, this error could be seen by itself due to unexpected Fabric
  Manager shutdown when FM is running in the same OS environment as the GPU.
  Otherwise, this error is safe to ignore as an informational message."

So NVIDIA's own triage explains the catalog's "Solo: RESTART_FM" as an A100
case (an unexpected Fabric Manager shutdown with Fabric Manager in the GPU's OS)
and reads every other solo XID 45 as informational. On a graded NVSwitch node
(Hopper SXM, HGX B200/B300: ``gpu_fabric_manager.graded_fabric_gpus``) batch
128 compiles RESTART_FM into the full GPU + NVSwitch reset -- a Fabric Manager
restart alone loses the fabric state -- which cordons the node, stops its
workloads and resets every GPU: every killed job (Control-C, sigkill) would
have reset the node's whole fabric. There, and only there, the default
effective action is observe-only: the decision becomes MONITOR_ONLY /
NO_ACTION, so the incident and its evidence are recorded and the XID
investigatory notification is sent exactly as for an IGNORE-class XID, nothing
touches or cordons the node, and the official action RESTART_FM stays on the
decision, the incident and the notification. The incident's reasons say why
and how to opt in, and its ``policy_reference`` is the debug-guidelines row.

The site switch ``GPU_FAULT_XID45_SOLO_NVSWITCH_RESTART_FM=true`` -- an
environment variable of the control plane, like the SXID remediation lists --
restores the catalog behaviour on those nodes: RESTART_FM compiles to batch
128's full fabric chain again. Every other node -- A100, NVIDIA's own solo case;
PCIe/NVL variants, GH200, GB200/GB300, unknown parts; a node whose exporter
reports no field 170 -- keeps FREEZE_EVIDENCE -> RESTART_FABRIC_MANAGER whatever
the switch says. A non-solo XID 45 follows its companion (the policy engine)
and is not touched here.
"""

from __future__ import annotations

from collections.abc import Sequence

from gpu_fault.models import RecoveryAction, Severity
from gpu_fault.policy import (
    ActionDisposition,
    ActionSource,
    FaultPolicyDecision,
    SxidEvent,
    XidEvent,
)

#: The site switch: ``true`` runs the catalog's RESTART_FM on graded NVSwitch
#: nodes (the full GPU + NVSwitch reset); unset or ``false`` is observe-only.
XID45_SOLO_RESTART_FM_ENV = "GPU_FAULT_XID45_SOLO_NVSWITCH_RESTART_FM"
#: The catalog's solo action for XID 45.
RESTART_FM = "RESTART_FM"
NVIDIA_XID_CATALOG_REFERENCE = (
    "https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html"
)
NVIDIA_XID_CATALOG_SPREADSHEET = (
    "https://docs.nvidia.com/deploy/xid-errors/_downloads/"
    "4586dadb59119a55d1e93a181caa4272/Xid-Catalog.xlsx"
)
#: GPU debug guidelines, GPU node triage, "Understanding Xid Messages", row
#: "45 Robust Channel Preemptive Removal": what every observe-only incident
#: carries as its ``policy_reference``.
NVIDIA_GPU_DEBUG_XID45_REFERENCE = (
    "https://docs.nvidia.com/deploy/gpu-debug-guidelines/gpu-node-triage.html"
    "#understanding-xid-messages"
)
NVIDIA_GPU_DEBUG_XID45_QUOTE = (
    "No action, informative only. Indicates channels affected by another "
    "failure. On A100, this error could be seen by itself due to unexpected "
    "Fabric Manager shutdown when FM is running in the same OS environment as "
    "the GPU. Otherwise, this error is safe to ignore as an informational "
    "message."
)
OBSERVE_ONLY_REASON = (
    "observe-only: a solo XID 45 on a graded NVSwitch node ({count} GPUs report "
    "an NVLink fabric state; Hopper SXM or HGX B200/B300) takes no node action "
    "and no cordon by default, because NVIDIA's GPU debug guidelines read it as "
    "informational except on A100 with Fabric Manager in the GPU's OS "
    f"({NVIDIA_GPU_DEBUG_XID45_REFERENCE}); the catalog's RESTART_FM is recorded, "
    "not run: on this node it is the full GPU + NVSwitch reset"
)
OPT_IN_REASON = (
    f"to run the catalog's RESTART_FM on such nodes, set {XID45_SOLO_RESTART_FM_ENV}"
    "=true on the control plane: a solo XID 45 then compiles to the full GPU + "
    "NVSwitch reset chain"
)


def solo_restart_fm(event: XidEvent | SxidEvent, decision: FaultPolicyDecision) -> bool:
    """Whether ``decision`` is the engine's executable solo XID 45 (RESTART_FM)."""

    return (
        isinstance(event, XidEvent)
        and event.xid == 45
        and decision.official_action == RESTART_FM
        and decision.disposition is ActionDisposition.EXECUTABLE
        and decision.source is ActionSource.NVIDIA_CATALOG
    )


def observe_only(
    decision: FaultPolicyDecision, graded_gpus: Sequence[str]
) -> FaultPolicyDecision:
    """``decision`` as a solo XID 45 on a graded node is by default: MONITOR_ONLY.

    The shape of every IGNORE-class XID (NO_ACTION, INFO marker, not active),
    with RESTART_FM kept as the official action and two reasons: why nothing
    ran, and the switch that opts in.
    """

    marker = decision.marker.model_copy(
        update={
            "severity": Severity.INFO,
            "recommended_action": RecoveryAction.NO_ACTION,
            "action_disposition": ActionDisposition.MONITOR_ONLY.value,
            "active": False,
        }
    )
    reasons = [
        *decision.reasons,
        OBSERVE_ONLY_REASON.format(count=len(graded_gpus)),
        OPT_IN_REASON,
    ]
    return decision.model_copy(
        update={
            "disposition": ActionDisposition.MONITOR_ONLY,
            "action": RecoveryAction.NO_ACTION,
            "safety_action": None,
            "pre_actions": [],
            "severity": Severity.INFO,
            "reasons": list(dict.fromkeys(reasons)),
            "marker": marker,
        }
    )


def observe_only_reference(decision: FaultPolicyDecision) -> str | None:
    """The incident's ``policy_reference`` for an observe-only solo XID 45.

    Only this rewrite pairs RESTART_FM with MONITOR_ONLY: the engine finalizes
    a solo XID 45 as EXECUTABLE and a non-solo one as its companion's action.
    """

    if (
        decision.official_action == RESTART_FM
        and decision.disposition is ActionDisposition.MONITOR_ONLY
    ):
        return NVIDIA_GPU_DEBUG_XID45_REFERENCE
    return None
