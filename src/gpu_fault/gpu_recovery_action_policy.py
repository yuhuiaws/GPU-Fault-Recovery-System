"""DCGM field 1523: the driver's GPU recovery action as a state (batch 233).

XID 154 reports a change of the driver's recovery action as a kernel event;
field 1523 (``DCGM_FI_DEV_GET_GPU_RECOVERY_ACTION``, DCGM 4.6 name
``DCGM_FI_DEV_GPU_RECOVERY_ACTION``) holds the same NVML value as a state. It
is the backstop for a lost XID 154 line (a ``kmsg`` overflow, a line emitted
while a quiesce window had stopped the collectors), names a GPU that was
already pending when the collector started, and proves the return to None
after a reset or a reboot.

Evidence -- every quote checked against the live page on the docs version
named:

* DCGM Field Identifiers (docs "Version: 4.6 (Latest)"), entry
  ``DCGM_FI_DEV_GPU_RECOVERY_ACTION 1523``:
  :data:`DCGM_RECOVERY_ACTION_FIELD_REFERENCE` --
  :data:`DCGM_RECOVERY_ACTION_FIELD_QUOTE`.
* NVIDIA GPU Memory Error Management (docs version 615), "Error Recovery and
  Response Flags", item "Recovery action flag":
  :data:`NVIDIA_RECOVERY_ACTION_REFERENCE` --
  :data:`NVIDIA_RECOVERY_ACTION_QUOTE`; its entries map the NVML values to the
  user actions (Reset: "Terminate all GPU processes, reset the GPU ...";
  Reboot: "Reboot the operating system ..."; Drain P2P: "Once all
  peer-to-peer traffic are drained, query NVML_FI_DEV_GET_GPU_RECOVERY_ACTION
  again, which will return one of the other actions."; Drain and Reset: "After
  all existing work have drained, reset the GPU to regain its full capacity.").
* The values (``nvmlDeviceGpuRecoveryAction_t``): NONE 0, GPU_RESET 1,
  NODE_REBOOT 2, DRAIN_P2P 3, DRAIN_AND_RESET 4, RECOVER_IMEX_DOMAIN 5 (DCGM
  v4.7.0 ``sdk/nvidia/nvml/dcgm_nvml.h``), BUS_RESET 6 and SYSTEM_REBOOT 7
  (NVIDIA's NVML bindings ``nvidia-ml-py`` 13.615.71, ``pynvml.py``). NVML API
  Reference (docs v615, "Last updated on Sep 09, 2026"), page "vGPU Structs",
  enum ``nvmlDeviceGpuRecoveryAction_t``
  (:data:`NVML_RECOVERY_ACTION_ENUM_REFERENCE`): one line per value, for 5-7
  :data:`NVML_RECOVERY_ACTION_ENUM_MEANINGS`. NVML change log: "Changes
  between NVML v590 and v595" -- :data:`NVML_IMEX_CHANGE_LOG_QUOTE`; "Changes
  between NVML v610 and v615" -- :data:`NVML_RECOVERY_ACTION_CHANGE_LOG_QUOTE`.

Values 1-4 follow the XID 154 path exactly
(``policy/engine.GpuFaultPolicyEngine._dynamic_recovery_resolution``):
:func:`xid154_outcome` is that mapping, and a test runs the engine on the
kernel label of every value to pin it. Value 5 follows what NVIDIA documents
for it -- DCGM Release Notes 4.7.0, "Background health checks (dcgmi health)":
:data:`DCGM_IMEX_RECOVERY_QUOTE`; DCGM v4.7.0 ``dcgmlib/dcgm_errors.h``
``DCGM_FR_GPU_RECOVERY_IMEX_DOMAIN_NEXT``: :data:`DCGM_IMEX_RECOVERY_NEXT` --
a recovery this product has no operation for: CRITICAL ``ESCALATE_OPERATOR``
(cordon, the support hand-off). For 6 and 7 NVIDIA documents only the one-line
enum meanings quoted above ("Reset the GPU’s PCIe bus.", "Reboot the
system."): no procedure, no nvidia-smi label (nvidia-smi and GPU Memory Error
Management list only None, Reset, Reboot, Drain P2P and Drain and Reset). Both
fail closed like a value NVIDIA does not document: CRITICAL ``QUARANTINE``
(freeze, cordon, quarantine, an operator decides) -- the shape the XID path
gives an XID 154 line without a label. 6 is not the product's ``RESET_GPU``:
the node action runs ``nvidia-smi --gpu-reset -i <uuid>`` with no reset type,
a Function Level Reset -- nvidia-smi(1) (v610-era page), "-r, --gpu-reset":
:data:`NVIDIA_SMI_RESET_TYPE_QUOTE`. 7 is not the node reboot of value 2,
which nvidia-smi qualifies as "OS warm reboot is sufficient (no need for AC/DC
cycle)"; what a system reboot adds NVIDIA does not say. DCGM's own health
check is no backstop for them: DCGM v4.7.0
``modules/health/DcgmHealthWatch.cpp`` logs any value above
RECOVER_IMEX_DOMAIN as :data:`DCGM_HEALTH_RECOVERY_RANGE_QUOTE` and ignores
it. A Blackwell part reports the same field from the same NVML enum; that
mapping is unit-tested from NVIDIA documentation, not live-verified (no
B-series hardware). While an XID 154 owns the
GPU's recovery -- a kernel XID 154 event inside :data:`XID154_EVIDENCE_WINDOW`
(still in its correlation window, or just finalized) or an open incident the
XID 154 path decided -- the value is evidence only: the finding's automatic
action is ``NO_ACTION`` and no incident or workflow is opened for it
(``app/ingest/gpu_findings.gpu_node_health_findings``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import timedelta
from typing import Any

from gpu_fault.gpu_metric_models import GpuHealthSeverity
from gpu_fault.models import (
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStepStatus,
)
from gpu_fault.pci_identity import same_pci_device
from gpu_fault.policy.models import DIRECT_ACTION_MAP, DynamicRecoveryAction

GPU_RECOVERY_ACTION = "gpu_recovery_action"
GPU_RECOVERY_ACTION_POLICY_SOURCE = "NVIDIA_DCGM_GPU_RECOVERY_ACTION"
GPU_RECOVERY_ACTION_POLICY_VERSION = "nvidia-gpu-recovery-action/615"
NVIDIA_RECOVERY_ACTION_DOCS_VERSION = "615"
NVIDIA_RECOVERY_ACTION_REFERENCE = (
    "https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/"
    "error-recovery-and-response-flags.html#error-recovery-and-response-flags"
)
NVIDIA_RECOVERY_ACTION_QUOTE = (
    "Recovery action flag will be triggered for uncorrectable errors."
)
DCGM_RECOVERY_ACTION_FIELD_REFERENCE = (
    "https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/"
    "dcgm-api-field-ids.html#c.DCGM_FI_DEV_GPU_RECOVERY_ACTION"
)
DCGM_RECOVERY_ACTION_FIELD_QUOTE = (
    "GPU Recovery Action (see nvmlDeviceGpuRecoveryAction_t for return values)"
)
NVML_RECOVERY_ACTION_CHANGE_LOG_REFERENCE = (
    "https://docs.nvidia.com/deploy/nvml-api/latest/change-log.html"
)
NVML_RECOVERY_ACTION_CHANGE_LOG_QUOTE = (
    "Added NVML_GPU_RECOVERY_ACTION_BUS_RESET and "
    "NVML_GPU_RECOVERY_ACTION_SYSTEM_REBOOT to nvmlDeviceGpuRecoveryAction_t."
)
NVML_IMEX_CHANGE_LOG_QUOTE = (
    "Added new NVML_GPU_RECOVERY_ACTION_RECOVER_IMEX_DOMAIN enum to "
    "nvmlDeviceGpuRecoveryAction_s."
)
DCGM_IMEX_RECOVERY_REFERENCE = (
    "https://docs.nvidia.com/datacenter/dcgm/latest/release-notes/changelog.html"
)
DCGM_IMEX_RECOVERY_QUOTE = (
    "Added DCGM_FR_GPU_RECOVERY_IMEX_DOMAIN for the separate case in which driver "
    "recovery requires restoring an IMEX domain."
)
DCGM_IMEX_RECOVERY_NEXT = "Recover or re-establish the IMEX domain."
NVML_RECOVER_IMEX_DOMAIN = 5
#: ``nvmlDeviceGpuRecoveryAction_t`` value -> the label ``nvidia-smi -q`` prints
#: ("Possible values: None, Reset, Reboot, Drain P2P, Drain and Reset"), and the
#: NVML enum name of the values nvidia-smi does not list.
NVML_GPU_RECOVERY_ACTION_LABELS = {
    0: "None",
    1: "Reset",
    2: "Reboot",
    3: "Drain P2P",
    4: "Drain and Reset",
    5: "RECOVER_IMEX_DOMAIN",
    6: "BUS_RESET",
    7: "SYSTEM_REBOOT",
}
#: NVML API Reference (v615) "vGPU Structs", ``nvmlDeviceGpuRecoveryAction_t``.
NVML_RECOVERY_ACTION_ENUM_REFERENCE = (
    "https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlVgpuStructs.html"
    "#_CPPv429nvmlDeviceGpuRecoveryAction_t"
)
#: The one-line enum meanings (verbatim) of the values nvidia-smi does not list.
NVML_RECOVERY_ACTION_ENUM_MEANINGS = {
    5: "Recover IMEX Domain.",
    6: "Reset the GPU’s PCIe bus.",
    7: "Reboot the system.",
}
#: Values NVIDIA documents only by that one-line meaning (NVML v615): no
#: procedure, no nvidia-smi label -- fail closed.
NVML_ENUM_ONLY_RECOVERY_ACTIONS = frozenset({6, 7})
#: nvidia-smi(1), "-r, --gpu-reset": the product's reset passes no reset type.
NVIDIA_SMI_RESET_TYPE_QUOTE = (
    "Reset triggered without extra arguments, will be a default Function Level "
    "Reset (FLR). To issue a Bus Reset, use -r bus. For certain platforms only "
    "Function Level Reset is possible."
)
#: DCGM v4.7.0 ``DcgmHealthWatch.cpp``: what DCGM's health check logs for a
#: value above ``NVML_GPU_RECOVERY_ACTION_RECOVER_IMEX_DOMAIN`` (and ignores).
DCGM_HEALTH_RECOVERY_RANGE_QUOTE = "Invalid GPU recovery action value"
#: The value -> the XID 154 action of the same label
#: (``policy.models.XID154_ACTION_LABELS``: "gpu reset required", "node reboot
#: required", "drain p2p", "drain and reset").
GPU_RECOVERY_ACTION_DYNAMIC = {
    1: DynamicRecoveryAction.RESET_GPU,
    2: DynamicRecoveryAction.RESTART_BM,
    3: DynamicRecoveryAction.DRAIN_P2P,
    4: DynamicRecoveryAction.DRAIN_AND_RESET,
}
#: How far around the batch a kernel XID 154 for the GPU makes the value
#: evidence: its 30 s correlation window plus the DCGM path's own lag (the
#: exporter's 15 s refresh, the collector's 15 s scrape) and slack. Older XID
#: 154s count only through their open incident.
XID154_EVIDENCE_WINDOW = timedelta(seconds=120)
#: What the XID 154 path writes as an incident's ``policy_source``
#: (``ActionSource.NVIDIA_XID_154``) and as the official action of an XID 154
#: line it could not read (``XID_154_ACTION``).
XID154_POLICY_SOURCE = "NVIDIA_XID_154"
XID154_OFFICIAL_ACTION = "XID_154"
OPEN_INCIDENT_STATES = frozenset(set(IncidentState) - {IncidentState.RECOVERED})
#: A recovery action the driver reports while another owner recovers the GPU.
EVIDENCE_ONLY_SOURCE = "XID_154_OWNS_RECOVERY"
UNKNOWN_ACTION_SOURCE = "NVIDIA_GPU_RECOVERY_ACTION_UNKNOWN"
#: The steps after which VALIDATE_GPU requires the GPU's recovery action to be
#: None again, read after the step: a GPU reset (also the full GPU + NVSwitch
#: reset and the drain-P2P recovery, which resets the GPUs the driver asks
#: for), a reboot, a replacement.
RECOVERY_ACTION_RESET_OPERATIONS = frozenset(
    {
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        WorkflowOperation.RECOVER_DRAIN_P2P,
        WorkflowOperation.RESTART_NODE,
        WorkflowOperation.REPLACE_NODE,
    }
)


def recovery_action_label(value: float) -> str:
    """The nvidia-smi label of ``value``, or ``unknown(<value>)``."""

    if math.isfinite(value) and value.is_integer():
        label = NVML_GPU_RECOVERY_ACTION_LABELS.get(int(value))
        if label is not None:
            return label
    return f"unknown({value:g})"


def xid154_outcome(action: DynamicRecoveryAction) -> tuple[RecoveryAction, str]:
    """``(automatic action, official action)`` the XID 154 path gives ``action``.

    ``GpuFaultPolicyEngine._dynamic_recovery_resolution``: DRAIN_P2P stops the
    workload under its own official action (the drain-P2P chain), DRAIN_AND_RESET
    resets the GPU, the others take their direct catalog action.
    """

    if action is DynamicRecoveryAction.DRAIN_P2P:
        return RecoveryAction.STOP_WORKLOAD, action.value
    mapped = (
        DynamicRecoveryAction.RESET_GPU
        if action is DynamicRecoveryAction.DRAIN_AND_RESET
        else action
    )
    return DIRECT_ACTION_MAP[mapped.value], action.value


def _same_gpu(gpu_uuid: Any, pci_bdf: Any, sample: Any) -> bool:
    """Whether an XID event's GPU is the sample's; an event naming no GPU is
    node-wide and owns every GPU (fail toward evidence, not a second plan).
    A PCI address names a device in any spelling (batch 236): the kernel
    line's ``0000:59:00`` is the exporter's ``00000000:59:00.0``."""

    if gpu_uuid and sample.gpu_uuid:
        return bool(gpu_uuid == sample.gpu_uuid)
    if pci_bdf and sample.pci_bdf:
        return same_pci_device(str(pci_bdf).lower(), str(sample.pci_bdf).lower())
    return not gpu_uuid and not pci_bdf


def xid154_owner(store: Any, batch: Any, sample: Any) -> str | None:
    """What owns the GPU's recovery through the XID 154 path, or ``None``.

    A kernel XID 154 event for the GPU inside :data:`XID154_EVIDENCE_WINDOW`
    of the batch (the XID event table keeps eight 30 s windows) -- usually
    still pending its correlation window, because the line reaches the control
    plane before the exporter's next refresh -- or an open incident of the node
    that the XID 154 path decided for the GPU (or for no GPU in particular).
    """

    events = store.list_xid_events(
        batch.cluster_id,
        batch.node_id,
        observed_after=batch.observed_at - XID154_EVIDENCE_WINDOW,
        observed_before=batch.observed_at + XID154_EVIDENCE_WINDOW,
    )
    for event in sorted(events, key=lambda item: (item.observed_at, item.event_id)):
        if event.xid == 154 and _same_gpu(event.gpu_uuid, event.pci_bdf, sample):
            return f"XID 154 event {event.event_id}"
    for incident in store.list_incidents_by_state(
        batch.cluster_id, OPEN_INCIDENT_STATES, node_ids={batch.node_id}
    ):
        if (
            incident.policy_source == XID154_POLICY_SOURCE
            or incident.official_action == XID154_OFFICIAL_ACTION
        ) and (not incident.gpu_uuids or sample.gpu_uuid in incident.gpu_uuids):
            return f"XID 154 incident {incident.incident_id}"
    return None


def _fail_closed_reason(code: int | None) -> str:
    """Why a value without an XID 154 action is quarantined (module docstring)."""

    if code is None or code not in NVML_ENUM_ONLY_RECOVERY_ACTIONS:
        return "is not a value NVIDIA documents"
    meaning = NVML_RECOVERY_ACTION_ENUM_MEANINGS[code]
    detail = (
        "the product's GPU reset is a Function Level Reset, not a bus reset"
        if code == 6
        else "not the node reboot of value 2"
    )
    return (
        f'is documented only by its one-line NVML meaning ("{meaning}"): no '
        f"procedure, no nvidia-smi label; {detail}"
    )


def recovery_action_decision(
    store: Any, batch: Any, sample: Any
) -> dict[str, Any] | None:
    """The grade of one field-1523 sample; ``None`` for None (0).

    See the module docstring: the XID 154 mapping, evidence while an XID 154
    owns the GPU, fail closed on a value this product cannot name.
    """

    if sample.canonical_name != GPU_RECOVERY_ACTION:
        return None
    value = float(sample.value)
    if value == 0:
        return None
    label = recovery_action_label(value)
    common = {
        "threshold_value": value,
        "policy_source": GPU_RECOVERY_ACTION_POLICY_SOURCE,
        "policy_version": GPU_RECOVERY_ACTION_POLICY_VERSION,
        "policy_reference": NVIDIA_RECOVERY_ACTION_REFERENCE,
    }
    owner = xid154_owner(store, batch, sample)
    if owner is not None:
        return {
            **common,
            "severity": GpuHealthSeverity.WARNING,
            "reason": (
                f"driver GPU recovery action {label} (DCGM field 1523) is evidence "
                f"for {owner}, which owns this GPU's recovery: no second workflow"
            ),
            "automatic_action": RecoveryAction.NO_ACTION.value,
            "threshold_source": EVIDENCE_ONLY_SOURCE,
        }
    code = int(value) if math.isfinite(value) and value.is_integer() else None
    if code == NVML_RECOVER_IMEX_DOMAIN:
        return {
            **common,
            "severity": GpuHealthSeverity.CRITICAL,
            "reason": (
                f"driver GPU recovery action {label} (DCGM field 1523): driver "
                "recovery requires restoring the IMEX domain, which no product "
                "step does; an operator recovers or re-establishes it"
            ),
            "automatic_action": RecoveryAction.ESCALATE_OPERATOR.value,
            "official_action": "RECOVER_IMEX_DOMAIN",
            "threshold_source": "NVIDIA_GPU_RECOVERY_ACTION",
            "policy_reference": DCGM_IMEX_RECOVERY_REFERENCE,
        }
    action = GPU_RECOVERY_ACTION_DYNAMIC.get(code) if code is not None else None
    if action is None:
        return {
            **common,
            "severity": GpuHealthSeverity.CRITICAL,
            "reason": (
                f"driver GPU recovery action {label} (DCGM field 1523) "
                + _fail_closed_reason(code)
                + "; fail closed: quarantine for an operator"
            ),
            "automatic_action": RecoveryAction.QUARANTINE.value,
            "threshold_source": UNKNOWN_ACTION_SOURCE,
            **(
                {"policy_reference": NVML_RECOVERY_ACTION_ENUM_REFERENCE}
                if code in NVML_ENUM_ONLY_RECOVERY_ACTIONS
                else {}
            ),
        }
    automatic, official = xid154_outcome(action)
    return {
        **common,
        "severity": GpuHealthSeverity.CRITICAL,
        "reason": (
            f"driver GPU recovery action {label} (DCGM field 1523); following it "
            f"as the XID 154 path does ({official})"
        ),
        "automatic_action": automatic.value,
        "official_action": official,
        "threshold_source": "NVIDIA_GPU_RECOVERY_ACTION",
    }


def last_recovery_reset(context: Any, node_id: str) -> Any | None:
    """The last reset, reboot or replacement this workflow completed on
    ``node_id`` before the current step, or ``None``."""

    steps = context.workflow.official_steps
    resets = [
        execution
        for execution in context.workflow.step_executions
        if execution.step_index < context.step_index
        and execution.status is WorkflowStepStatus.SUCCEEDED
        and execution.operation in RECOVERY_ACTION_RESET_OPERATIONS
        and (
            execution.step_index >= len(steps)
            or not steps[execution.step_index].node_ids
            or node_id in steps[execution.step_index].node_ids
        )
    ]
    return max(resets, key=lambda execution: execution.updated_at, default=None)


def recovery_action_validation(
    context: Any,
    node_id: str,
    latest: Sequence[Any],
    targets: Sequence[str],
    failures: dict[str, list[str]],
) -> dict[str, Any] | None:
    """VALIDATE_GPU's check that the driver asks for nothing after a reset.

    After this workflow reset (or rebooted, or replaced) ``node_id``, every
    target GPU's field-1523 value read *after* that step must be None (0); the
    XML ``<gpu_recovery_action>`` the DRAIN_P2P step reads is the same NVML
    value, read before its reset, so the post-step reading comes from DCGM.
    Returns the WAITING detail while a target has no sample newer than the
    step; appends ``gpu_recovery_action_not_none(...)`` to ``failures`` when one
    still asks for an action. Nothing to judge -- no such step, or an exporter
    that does not serve the field (exported, never required) -- is ``None``.
    """

    reset = last_recovery_reset(context, node_id)
    if reset is None:
        return None
    samples = [
        item for item in latest if item.sample.canonical_name == GPU_RECOVERY_ACTION
    ]
    if not samples:
        return None
    wanted = list(targets) or sorted(
        {str(item.sample.gpu_uuid) for item in samples if item.sample.gpu_uuid}
    )
    fresh = {
        str(item.sample.gpu_uuid): item
        for item in samples
        if item.sample.gpu_uuid and item.observed_at > reset.updated_at
    }
    asking = {
        gpu: recovery_action_label(float(fresh[gpu].sample.value))
        for gpu in wanted
        if gpu in fresh and fresh[gpu].sample.value != 0
    }
    if asking:
        failures.setdefault(node_id, []).append(
            "gpu_recovery_action_not_none("
            + ", ".join(f"{gpu}={label}" for gpu, label in sorted(asking.items()))
            + f"; read after step {reset.step_index} {reset.operation.value})"
        )
        return None
    waiting = [gpu for gpu in wanted if gpu not in fresh]
    if waiting:
        return {
            "awaiting_gpu_recovery_action": waiting,
            "gpu_recovery_action_after": reset.updated_at.isoformat(),
        }
    return None
