"""NVLink fabric registration state: DCGM field 170, graded per node.

On a graded NVSwitch node -- a Hopper SXM part on third-generation NVSwitches
(H100/H200/H800/H20 SXM) or an HGX B200/B300 on fourth-generation NVSwitches
(NVL5; the driver names the parts ``NVIDIA B200`` and ``NVIDIA B300 SXM6 AC``)
-- every GPU registers with the NVLink fabric that Fabric Manager trains; until
a GPU's registration completes, CUDA does not start on it (NVIDIA Fabric Manager
user guide, "Initializing NVSwitch and NVLink"). DCGM reports the registration
state per GPU as ``DCGM_FI_DEV_FABRIC_MANAGER_STATUS`` (170), a
``dcgmFabricManagerStatus_t`` (``dcgm_structs.h``, identical in DCGM 4.4.1 and
4.7.0), and the DCGM return code of a completed registration as
``DCGM_FI_DEV_FABRIC_MANAGER_ERROR_CODE`` (171), blank -- no exporter sample --
until it completes.

On NVL5 the same ``nvidia-fabricmanager`` unit also runs the NVLink Subnet
Manager (NVLSM): "The FM service unit file is also updated to start the NVLSM
process if applicable. In this case, the FM systemd service status indicates
the process status for FM and NVLSM, and operations such as systemd start,
stop, and so on will operate on both processes." (Fabric Manager user guide,
"The Fabric Manager Service"). Keeping or stopping the unit keeps or stops
both; the unit tracks Fabric Manager's PID only (``PIDFile``), so an NVLSM that
died beside a running Fabric Manager does not show in the unit's state.

Live (plain EKS p5en, H200 HGX, driver 580.178.04, DCGM 4.7.0): Fabric Manager
failed at boot ("failed to configure all the available GPUs or NVSwitches") and
every GPU stayed at 2 (in progress) for 19.5 h while nothing in the product
looked at the field. A healthy peer reports 3 / 0. HGX B200/B300 are graded by
the same rules but have not been observed live.

The collector reports, beside each GPU whose state is not settled, how long it
has held that state (:data:`FABRIC_MANAGER_HELD_SECONDS`, measured from the
collector's first observation of the value), and the control plane grades one
node-level finding once any GPU has held an unsettled state for
:data:`FABRIC_MANAGER_UNSETTLED_WINDOW_SECONDS`. The window is what keeps
boot-time training (seconds to about a minute) and the product's own Fabric
Manager stops out: the metrics collector's unit is ordered
``After=nvidia-fabricmanager.service`` (ordering only, no dependency), so at
boot its clock starts once Fabric Manager's start has returned, and the quiesce
stops the collector with Fabric Manager, so the clock starts again after every
restore and every reboot.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from datetime import datetime
from enum import IntEnum
from typing import TYPE_CHECKING, Any

from gpu_fault.gpu_metric_models import (
    SITE_METRIC_POLICY_VERSION,
    GpuHealthFinding,
    GpuHealthSeverity,
)
from gpu_fault.models import RecoveryAction

if TYPE_CHECKING:
    from gpu_fault.gpu_metrics import GpuMetricBatch, GpuMetricSample

FABRIC_MANAGER_STATUS = "fabric_manager_status"
FABRIC_MANAGER_ERROR_CODE = "fabric_manager_error_code"
#: Derived by the collector, never exported by DCGM: seconds the GPU has
#: reported its current unsettled state. Emitted only while it is unsettled.
FABRIC_MANAGER_HELD_SECONDS = "fabric_manager_status_held_seconds"
#: Boot-time training ends within about a minute, and the metrics collector
#: starts after Fabric Manager (``deploy/systemd/gpu-fault-metrics-collector.
#: service``: ``After=nvidia-fabricmanager.service``, ordering only); five
#: minutes of an unchanged unsettled state is a fabric that is not going to
#: train on its own.
FABRIC_MANAGER_UNSETTLED_WINDOW_SECONDS = 300
FABRIC_MANAGER_POLICY_SOURCE = "NVIDIA_FABRIC_MANAGER_STATE"
FABRIC_MANAGER_REFERENCE = (
    "https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html"
)
#: The H100-and-later restart procedure of the guide: stop the GPU services
#: and Fabric Manager, ``nvidia-smi -r``, start Fabric Manager again -- the
#: product's RESET_ALL_GPUS_NVSWITCHES chain. A Fabric Manager restart alone
#: does not re-register GPUs (live: all 8 stayed "In Progress" after one).
FABRIC_MANAGER_OFFICIAL_ACTION = "RESET_ALL_GPUS_AND_NVSWITCHES"


class FabricManagerStatus(IntEnum):
    """``dcgmFabricManagerStatus_t``."""

    NOT_SUPPORTED = 0
    NOT_STARTED = 1
    IN_PROGRESS = 2
    SUCCESS = 3
    FAILURE = 4
    UNRECOGNIZED = 5
    NVML_TOO_OLD = 6


#: Graded. NOT_SUPPORTED (no NVSwitch fabric, e.g. L40S), SUCCESS, NVML_TOO_OLD
#: (a driver that cannot say) and UNRECOGNIZED (a value DCGM cannot name) are
#: never a finding: only a state that says "this GPU has not joined the fabric".
UNSETTLED_STATUSES = frozenset(
    {
        FabricManagerStatus.NOT_STARTED,
        FabricManagerStatus.IN_PROGRESS,
        FabricManagerStatus.FAILURE,
    }
)
#: The states of a GPU that takes part in an NVLink fabric: what makes a node
#: one whose fabric (and Fabric Manager unit) VALIDATE_GPU judges.
FABRIC_STATES = UNSETTLED_STATUSES | {FabricManagerStatus.SUCCESS}

#: The graded NVSwitch nodes: Hopper SXM parts on third-generation NVSwitch
#: baseboards (H100/H200/H800/H20) and HGX B200/B300 on fourth-generation
#: NVSwitches (NVL5: the ``nvidia-fabricmanager`` unit runs Fabric Manager and
#: NVLSM). PCIe and NVL (NVLink bridge) variants have no NVSwitch fabric; Grace
#: Hopper (GH200), GB200/GB300 NVL72 (the fabric runs from the switch trays, the
#: compute trays have no local Fabric Manager) and every other or unknown part
#: (B100, RTX PRO 6000 Blackwell, ...) are deliberately not graded: NVIDIA
#: documents no bound on a healthy multi-node tray's time in "in progress".
_NVSWITCH_FABRIC_PRODUCT = re.compile(
    r"\b(?:H(?:100|200|800|20)|B(?:200|300))\b", re.IGNORECASE
)
_NO_NVSWITCH_FABRIC = re.compile(r"\b(?:PCIE|NVL)\b", re.IGNORECASE)


def fabric_status(value: float) -> FabricManagerStatus | None:
    """The DCGM status a sample value names, or None for anything else."""

    if not float(value).is_integer():
        return None
    try:
        return FabricManagerStatus(int(value))
    except ValueError:
        return None


def nvswitch_fabric_product(product: str | None) -> bool:
    """Whether the fabric state of this GPU product is graded at all."""

    return bool(
        product
        and _NVSWITCH_FABRIC_PRODUCT.search(product)
        and not _NO_NVSWITCH_FABRIC.search(product)
    )


def _gpu_key(sample: GpuMetricSample) -> str:
    return sample.gpu_uuid or sample.pci_bdf or sample.gpu_index or "node"


def held_samples(
    samples: Iterable[GpuMetricSample],
    first_seen: Callable[[GpuMetricSample], datetime | None],
    observed_at: datetime,
) -> list[GpuMetricSample]:
    """The collector's held-seconds companion of every unsettled status sample.

    ``first_seen`` answers when the collector first observed the sample's
    current value (None: a new value, or nothing remembered).
    """

    result = []
    for sample in samples:
        if sample.canonical_name != FABRIC_MANAGER_STATUS:
            continue
        if fabric_status(sample.value) not in UNSETTLED_STATUSES:
            continue
        since = first_seen(sample) or observed_at
        result.append(
            sample.model_copy(
                update={
                    "metric_name": f"gpu_fault_{FABRIC_MANAGER_HELD_SECONDS}",
                    "canonical_name": FABRIC_MANAGER_HELD_SECONDS,
                    "value": max(0.0, (observed_at - since).total_seconds()),
                    "unit": "seconds",
                }
            )
        )
    return result


def unsettled_fabric(
    observations: Iterable[tuple[GpuMetricSample, datetime | None]],
    *,
    product: str | None = None,
    window_seconds: int = FABRIC_MANAGER_UNSETTLED_WINDOW_SECONDS,
) -> tuple[dict[str, GpuMetricSample], dict[str, tuple[FabricManagerStatus, float]]]:
    """``(status sample per graded GPU, {gpu: (state, held)} past the window)``.

    A GPU is graded when it is a graded product and reports a fabric it takes
    part in (:data:`FABRIC_STATES`). ``observations`` pairs each sample with the
    time it was observed (None within one batch). A held companion counts only
    for the status sample it was reported with: the store keeps the last one
    after the state settles. ``product`` stands in for a sample that carries no
    ``modelName`` label.
    """

    items = list(observations)
    held = {
        (_gpu_key(sample), observed_at): sample.value
        for sample, observed_at in items
        if sample.canonical_name == FABRIC_MANAGER_HELD_SECONDS
    }
    graded: dict[str, GpuMetricSample] = {}
    unsettled: dict[str, tuple[FabricManagerStatus, float]] = {}
    for sample, observed_at in items:
        if sample.canonical_name != FABRIC_MANAGER_STATUS:
            continue
        if not nvswitch_fabric_product(sample.labels.get("modelName") or product):
            continue
        status = fabric_status(sample.value)
        if status not in FABRIC_STATES:
            continue
        key = _gpu_key(sample)
        graded[key] = sample
        seconds = held.get((key, observed_at), 0.0)
        if status in UNSETTLED_STATUSES and seconds >= window_seconds:
            unsettled[key] = (status, seconds)
    return graded, unsettled


def graded_fabric_gpus(
    observations: Iterable[tuple[GpuMetricSample, datetime | None]],
    *,
    product: str | None = None,
) -> list[str]:
    """The GPU UUIDs whose fabric state is graded, sorted; [] for any other node.

    Non-empty exactly for the graded NVSwitch nodes (Hopper SXM, HGX
    B200/B300) that report a fabric state (:func:`unsettled_fabric`). Planning
    reads it from the node's latest samples: a Fabric Manager restart there is
    NVIDIA's H100-and-later procedure on all of these GPUs, and a solo XID 45
    there is observe-only unless the site opts in
    (``orchestration/xid45_policy.py``).
    """

    graded, _unsettled = unsettled_fabric(observations, product=product)
    return sorted({sample.gpu_uuid for sample in graded.values() if sample.gpu_uuid})


def fabric_validation(
    observations: Iterable[tuple[GpuMetricSample, datetime | None]],
) -> tuple[int, dict[str, str], str | None]:
    """VALIDATE_GPU's reading: ``(graded GPUs, {gpu: state} still training,
    failure reason or None)``.

    A FAILURE state fails at once: the registration completed with an error and
    stays that way until the next reset. NOT_STARTED / IN_PROGRESS is training
    that may still finish, so the step waits -- until the state has been held
    for the window, which fails it like the passive finding would.
    """

    graded, unsettled = unsettled_fabric(observations)
    failed = dict(unsettled)
    training: dict[str, str] = {}
    for gpu, sample in graded.items():
        status = fabric_status(sample.value)
        if status is FabricManagerStatus.FAILURE:
            failed.setdefault(gpu, (status, 0.0))
        elif status in UNSETTLED_STATUSES and gpu not in failed:
            training[gpu] = status.name
    if not failed:
        return len(graded), training, None
    return (
        len(graded),
        training,
        f"nvlink_fabric_not_trained({_describe(failed)}: "
        f"{len(failed)}/{len(graded)} GPUs)",
    )


def _describe(unsettled: dict[str, tuple[FabricManagerStatus, float]]) -> str:
    states = sorted({state.name for state, _held in unsettled.values()})
    return "/".join(states)


def fabric_manager_finding(
    batch: GpuMetricBatch,
) -> tuple[bool, GpuHealthFinding | None]:
    """``(the batch reports the fabric state, the node finding or None)``."""

    if not any(item.canonical_name == FABRIC_MANAGER_STATUS for item in batch.samples):
        return False, None
    graded, unsettled = unsettled_fabric(
        ((sample, None) for sample in batch.samples), product=batch.product
    )
    if not unsettled:
        return True, None
    window = FABRIC_MANAGER_UNSETTLED_WINDOW_SECONDS
    longest = max(held for _state, held in unsettled.values())
    gpu_uuids = sorted({item.gpu_uuid for item in graded.values() if item.gpu_uuid})
    return True, GpuHealthFinding(
        finding_id=f"{batch.batch_id}-node-{FABRIC_MANAGER_STATUS}",
        cluster_id=batch.cluster_id,
        node_id=batch.node_id,
        observed_at=batch.observed_at,
        severity=GpuHealthSeverity.CRITICAL,
        reason=(
            f"NVLink fabric not trained: {len(unsettled)}/{len(graded)} GPUs report "
            f"Fabric Manager status {_describe(unsettled)} for >= {window}s "
            f"(longest {longest:.0f}s); CUDA cannot start on a GPU that has not "
            "registered with the fabric. Recovery: stop Fabric Manager and the GPU "
            "services, reset all GPUs and NVSwitches, start Fabric Manager"
        ),
        canonical_name=FABRIC_MANAGER_STATUS,
        value=longest,
        threshold_value=float(window),
        threshold_source="SITE_FABRIC_MANAGER_UNSETTLED_WINDOW",
        evidence_ref=batch.evidence_ref,
        automatic_action=RecoveryAction.RESET_GPU.value,
        policy_source=FABRIC_MANAGER_POLICY_SOURCE,
        policy_version=SITE_METRIC_POLICY_VERSION,
        policy_reference=FABRIC_MANAGER_REFERENCE,
        official_action=FABRIC_MANAGER_OFFICIAL_ACTION,
        runtime_profile_version=batch.runtime_profile_version,
        workload_state=batch.workload_state,
        affected_workload_ids=batch.affected_workload_ids,
        affected_gpu_uuids=gpu_uuids,
        confidence="HIGH",
    )


def track_fabric_manager_state(
    store: Any,
    batch: GpuMetricBatch,
    findings: list[GpuHealthFinding],
    finding_updates: list[Any],
    findings_by_update: list[GpuHealthFinding | None],
) -> None:
    """Append this batch's node fabric update to the ingest's three lists.

    A batch that does not report the field changes nothing (an exporter without
    it, an error batch). A settled report clears a finding that is active and
    writes nothing for a node that never had one.
    """

    reported, finding = fabric_manager_finding(batch)
    if not reported:
        return
    key = (batch.cluster_id, batch.node_id, "node", FABRIC_MANAGER_STATUS)
    if finding is None:
        (previous,) = store.get_gpu_finding_states([key])
        if previous is None or previous.finding is None:
            return
    else:
        findings.append(finding)
    finding_updates.append((key, finding, batch.observed_at))
    findings_by_update.append(finding)
