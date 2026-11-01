"""NVLink fabric health: DCGM fields 174/175, graded per node (batch 235).

DCGM reports a GPU's fabric health as a bit mask (174,
``DCGM_FI_DEV_FABRIC_HEALTH_MASK``) and, from DCGM 4.6.0, a summary (175,
``DCGM_FI_DEV_FABRIC_HEALTH_SUMMARY``). Both are exported by the bundled
counters CSV and never required. Live H200 (driver 580.178.04, DCGM 4.7.0)
reads 0 / 0 -- every slot NOT_SUPPORTED -- and the nvidia-smi XML health
sub-tags read ``N/A``: nothing here ever grades on H200. HGX B200/B300 (NVL5)
are expected to report real values; that is unit-tested from NVIDIA
documentation, not live-verified (no B-series hardware).

**Decode.** The NVML v615 layout (NVML API Reference, docs version 615,
"Fabric Definitions" -- :data:`NVML_FABRIC_REFERENCE`; values from NVIDIA's
bindings ``nvidia-ml-py`` 13.615.71 ``pynvml.py``): two-bit slots for degraded
bandwidth (shift 0), route recovery (2), route unhealthy (4) and access-timeout
recovery (6), each 0 NOT_SUPPORTED / 1 TRUE / 2 FALSE; incorrect configuration
(shift 8, four bits: 0 NOT_SUPPORTED, 1 NONE, 2-8 the incorrect
configurations); partition assigned (12) and the global Fabric Manager state
(14). Summary: 0 NOT_SUPPORTED, 1 HEALTHY, 2 UNHEALTHY, 3 LIMITED_CAPACITY.
Bits NVML may add later are ignored.

**Grades** (DCGM v4.7.0 ``DcgmHealthWatch::MonitorFabricFields`` FAILs on route
unhealthy, degraded bandwidth, route recovery, access-timeout recovery and an
incorrect configuration other than NOT_SUPPORTED / NONE; its next step,
``DCGM_FR_FIELD_VIOLATION_NEXT``, is :data:`DCGM_SYSTEM_TRIAGE_NEXT`):

* NOT_SUPPORTED (summary 0 and no supported slot reading a fault) -> nothing.
* UNHEALTHY -- summary 2, route unhealthy, or an incorrect configuration --
  held for :data:`UNHEALTHY_CONFIRMATION_SECONDS` (two samples that far apart)
  outside every product fabric window -> CRITICAL ``RESET_GPU``, official
  ``RESET_ALL_GPUS_AND_NVSWITCHES``: the shape of
  ``gpu_fabric_manager.fabric_manager_finding``, which compiles the full GPU +
  NVSwitch reset. Before the confirmation, or inside a window, evidence only.
* Route or access-timeout recovery in progress -> evidence (a transient state);
  CRITICAL as above once held :data:`RECOVERY_CRITICAL_SECONDS`.
* LIMITED_CAPACITY or degraded bandwidth -> WARNING ``RUN_DIAGNOSTICS`` plus an
  operator notice (``app/ingest/node_health.OPERATOR_REVIEW_METRICS``): the
  Fabric Manager may have applied a degraded mode at initialisation (Fabric
  Manager User Guide, :data:`FM_DEGRADED_MODE_QUOTE`), which no reset by this
  product undoes.

Graded only on the NVSwitch nodes the Fabric Manager state is graded on
(``gpu_fabric_manager.nvswitch_fabric_product``: Hopper SXM, HGX B200/B300);
the partition and global Fabric Manager slots serve multi-node NVLink systems
and grade nothing. "Held" is the collector's own clock
(:data:`FABRIC_HEALTH_HELD_SECONDS`, from its first observation of the current
mask and summary, emitted only while a GPU reports a fault state): it starts
again with the collector -- after every quiesce, restore and reboot -- and the
control plane also refuses to confirm across a product window that ran while
the state was held (``gpu_product_windows``: a Fabric Manager restart keeps the
collector running).

**Second source.** The DCGM collector reads the nvidia-smi XML ``<fabric>``
``<health>`` sub-tags at the inventory cadence (``collectors/gpu/smi_health``)
and publishes them as :data:`FABRIC_HEALTH_XML` (value: the summary code;
labels: the raw strings; nothing while every tag is ``N/A`` or missing). They
are evidence in the finding's reason and in VALIDATE_GPU's failure text, never
a grade of their own. VALIDATE_GPU (``fabric_validation``) requires every
graded GPU's summary to be HEALTHY or NOT_SUPPORTED.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime, timedelta
from enum import IntEnum
from typing import TYPE_CHECKING, Any

from gpu_fault.gpu_fabric_manager import (
    FABRIC_MANAGER_OFFICIAL_ACTION,
    nvswitch_fabric_product,
    track_fabric_manager_state,
    unsettled_fabric,
)
from gpu_fault.gpu_fabric_manager import fabric_validation as fabric_manager_validation
from gpu_fault.gpu_metric_models import (
    SITE_METRIC_POLICY_VERSION,
    GpuHealthFinding,
    GpuHealthSeverity,
)
from gpu_fault.gpu_product_windows import describe_window, product_link_window
from gpu_fault.models import RecoveryAction

if TYPE_CHECKING:
    from gpu_fault.gpu_metrics import GpuMetricBatch, GpuMetricSample

FABRIC_HEALTH_MASK = "fabric_health_mask"
FABRIC_HEALTH_SUMMARY = "fabric_health_summary"
#: Derived by the collector: seconds the GPU has reported its current fault
#: mask and summary. Emitted only while a supported slot reads a fault.
FABRIC_HEALTH_HELD_SECONDS = "fabric_health_held_seconds"
#: The nvidia-smi XML ``<fabric><health>`` reading (second source).
FABRIC_HEALTH_XML = "fabric_health_xml"
#: The node finding.
FABRIC_HEALTH = "fabric_health"
UNHEALTHY_CONFIRMATION_SECONDS = 60
RECOVERY_CRITICAL_SECONDS = 300
FABRIC_HEALTH_POLICY_SOURCE = "NVIDIA_DCGM_HEALTH"
NVML_FABRIC_REFERENCE = (
    "https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlFabricDefs.html"
)
#: NVML v615 "Fabric Definitions", verbatim.
NVML_ROUTE_UNHEALTHY_QUOTE = "Fabric Health Mask: Route is unhealthy."
NVML_DEGRADED_BW_QUOTE = "Fabric Health Mask: Bandwidth degraded."
NVML_SUMMARY_UNHEALTHY_QUOTE = "Fabric Health Summary: Unhealthy."
NVML_SUMMARY_LIMITED_QUOTE = "Fabric Health Summary: Limited Capacity."
DCGM_FABRIC_HEALTH_REFERENCE = (
    "https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html"
    "#c.DCGM_FI_DEV_FABRIC_HEALTH_MASK"
)
DCGM_FABRIC_HEALTH_QUOTE = (
    "Use DCGM_GPU_FABRIC_HEALTH_TEST macro to check the different health statuses."
)
DCGM_FABRIC_HEALTH_WATCH_REFERENCE = (
    "https://github.com/NVIDIA/DCGM/blob/v4.7.0/modules/health/DcgmHealthWatch.cpp"
)
DCGM_SYSTEM_TRIAGE_NEXT = (
    "Check DCGM and system logs for errors. Reset GPU. Restart DCGM. Rerun diagnostics."
)
NVIDIA_SMI_FABRIC_HEALTH_REFERENCE = (
    "https://docs.nvidia.com/deploy/nvidia-smi/index.html"
)
#: nvidia-smi(1), "Fabric" > "Health", verbatim.
NVIDIA_SMI_SUMMARY_QUOTE = (
    "Summary - Summary of Fabric Health <Healthy, Unhealthy, Limited Capacity>"
)
NVIDIA_SMI_ROUTE_UNHEALTHY_QUOTE = (
    "Route Unhealthy - is NVLink route recovery failed or aborted <True/False>"
)
#: Fabric Manager User Guide ("Last updated on Oct 07, 2026"), "Supported High
#: Availability Modes", note.
FM_DEGRADED_MODE_REFERENCE = (
    "https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html"
    "#supported-high-availability-modes"
)
FM_DEGRADED_MODE_QUOTE = (
    "These high availability modes and their corresponding dynamic reconfiguration "
    "of the NVSwitch-based system are applied in response to errors that are "
    "detected during FM initialization."
)


class FabricHealthSummary(IntEnum):
    """``NVML_GPU_FABRIC_HEALTH_SUMMARY_*``."""

    NOT_SUPPORTED = 0
    HEALTHY = 1
    UNHEALTHY = 2
    LIMITED_CAPACITY = 3


#: Slot -> (shift, width mask), ``NVML_GPU_FABRIC_HEALTH_MASK_SHIFT_*`` /
#: ``_WIDTH_*``.
MASK_SLOTS = {
    "degraded_bw": (0, 0x3),
    "route_recovery": (2, 0x3),
    "route_unhealthy": (4, 0x3),
    "access_timeout_recovery": (6, 0x3),
    "incorrect_configuration": (8, 0xF),
    "partition_assigned": (12, 0x3),
    "gfm_state": (14, 0x3),
}
SLOT_NOT_SUPPORTED, SLOT_TRUE, SLOT_FALSE = 0, 1, 2
#: ``NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_*``: 1 is NONE.
INCORRECT_CONFIGURATION_NONE = 1
INCORRECT_CONFIGURATIONS = {
    2: "INCORRECT_SYSGUID",
    3: "INCORRECT_CHASSIS_SN",
    4: "NO_PARTITION",
    5: "INSUFFICIENT_NVLINKS",
    6: "INCOMPATIBLE_GPU_FW",
    7: "INVALID_LOCATION",
    8: "GPU_STATE_INVALID",
}
UNHEALTHY, RECOVERY, DEGRADED = "UNHEALTHY", "RECOVERY", "DEGRADED"
#: How long a state is held before it grades CRITICAL (DEGRADED never does).
CONFIRMATION_SECONDS = {
    UNHEALTHY: UNHEALTHY_CONFIRMATION_SECONDS,
    RECOVERY: RECOVERY_CRITICAL_SECONDS,
}
#: nvidia-smi XML ``<fabric><health>`` sub-tags (live H200 R580: all ``N/A``).
XML_HEALTH_TAGS = (
    "summary",
    "bandwidth",
    "route_recovery_in_progress",
    "route_unhealthy",
    "access_timeout_recovery",
    "incorrect_configuration",
    "partition_assigned",
)
XML_LABEL_PREFIX = "fabric_health_"
#: The XML ``<summary>`` strings (nvidia-smi(1)) -> the summary code.
XML_SUMMARY_CODES = {
    "healthy": float(FabricHealthSummary.HEALTHY),
    "unhealthy": float(FabricHealthSummary.UNHEALTHY),
    "limited capacity": float(FabricHealthSummary.LIMITED_CAPACITY),
}


def _integer(value: float | None) -> int | None:
    if value is None or not float(value).is_integer() or value < 0:
        return None
    return int(value)


def decode_mask(value: float | None) -> dict[str, int]:
    """The mask's slots (empty for a value that is no mask)."""

    mask = _integer(value)
    if mask is None:
        return {}
    return {
        name: (mask >> shift) & width for name, (shift, width) in MASK_SLOTS.items()
    }


def fabric_health_state(
    mask: float | None, summary: float | None
) -> tuple[str | None, list[str]]:
    """``(UNHEALTHY | RECOVERY | DEGRADED | None, what the slots say)``."""

    slots = decode_mask(mask)
    code = _integer(summary)
    details: list[str] = []
    if code == FabricHealthSummary.UNHEALTHY:
        details.append("summary UNHEALTHY")
    if slots.get("route_unhealthy") == SLOT_TRUE:
        details.append("route unhealthy")
    config = slots.get("incorrect_configuration", SLOT_NOT_SUPPORTED)
    if config not in {SLOT_NOT_SUPPORTED, INCORRECT_CONFIGURATION_NONE}:
        details.append(
            "incorrect configuration "
            + INCORRECT_CONFIGURATIONS.get(config, f"unknown({config})")
        )
    if details:
        return UNHEALTHY, details
    if slots.get("route_recovery") == SLOT_TRUE:
        details.append("route recovery in progress")
    if slots.get("access_timeout_recovery") == SLOT_TRUE:
        details.append("access timeout recovery in progress")
    if details:
        return RECOVERY, details
    if code == FabricHealthSummary.LIMITED_CAPACITY:
        details.append("summary LIMITED_CAPACITY")
    if slots.get("degraded_bw") == SLOT_TRUE:
        details.append("bandwidth degraded")
    return (DEGRADED, details) if details else (None, [])


def _gpu_key(sample: GpuMetricSample) -> str:
    return sample.gpu_uuid or sample.pci_bdf or sample.gpu_index or "node"


def _readings(
    samples: Iterable[GpuMetricSample],
) -> dict[str, dict[str, GpuMetricSample]]:
    result: dict[str, dict[str, GpuMetricSample]] = {}
    for sample in samples:
        if sample.canonical_name in {
            FABRIC_HEALTH_MASK,
            FABRIC_HEALTH_SUMMARY,
            FABRIC_HEALTH_HELD_SECONDS,
            FABRIC_HEALTH_XML,
        }:
            result.setdefault(_gpu_key(sample), {})[sample.canonical_name] = sample
    return result


def _state_of(reading: dict[str, GpuMetricSample]) -> tuple[str | None, list[str]]:
    mask = reading.get(FABRIC_HEALTH_MASK)
    summary = reading.get(FABRIC_HEALTH_SUMMARY)
    return fabric_health_state(
        mask.value if mask is not None else None,
        summary.value if summary is not None else None,
    )


def held_samples(
    samples: Iterable[GpuMetricSample],
    first_seen: Callable[[GpuMetricSample], datetime | None],
    observed_at: datetime,
) -> list[GpuMetricSample]:
    """The collector's held-seconds companion of every GPU in a fault state:
    from its first observation of the current mask *and* summary."""

    result = []
    for reading in _readings(samples).values():
        state, _details = _state_of(reading)
        sources = [
            item
            for name in (FABRIC_HEALTH_MASK, FABRIC_HEALTH_SUMMARY)
            if (item := reading.get(name)) is not None
        ]
        if state is None or not sources:
            continue
        since = max(first_seen(item) or observed_at for item in sources)
        result.append(
            sources[0].model_copy(
                update={
                    "metric_name": f"gpu_fault_{FABRIC_HEALTH_HELD_SECONDS}",
                    "canonical_name": FABRIC_HEALTH_HELD_SECONDS,
                    "value": max(0.0, (observed_at - since).total_seconds()),
                    "unit": "seconds",
                    "labels": {**sources[0].labels, "fabric_health_state": state},
                }
            )
        )
    return result


def held_breached(value: float, peers: dict[str, float]) -> bool:
    """The collector's edge: a fault state held long enough to grade CRITICAL
    (``collectors/gpu/field_edges.state_breached``)."""

    state, _details = fabric_health_state(
        peers.get(FABRIC_HEALTH_MASK), peers.get(FABRIC_HEALTH_SUMMARY)
    )
    threshold = CONFIRMATION_SECONDS.get(state or "")
    return threshold is not None and value >= threshold


def xml_evidence(sample: GpuMetricSample | None) -> str | None:
    """``nvidia-smi summary=Unhealthy, route_unhealthy=True`` from the XML
    reading's labels, or ``None``."""

    if sample is None:
        return None
    parts = [
        f"{tag}={sample.labels[XML_LABEL_PREFIX + tag]}"
        for tag in XML_HEALTH_TAGS
        if XML_LABEL_PREFIX + tag in sample.labels
    ]
    return "nvidia-smi " + ", ".join(parts) if parts else None


def _gpu_text(gpu: str, reading: dict[str, GpuMetricSample], details: list[str]) -> str:
    xml = xml_evidence(reading.get(FABRIC_HEALTH_XML))
    return f"{gpu} {'/'.join(details)}" + (f" ({xml})" if xml else "")


def fabric_health_finding(
    store: Any, batch: GpuMetricBatch
) -> tuple[bool, GpuHealthFinding | None]:
    """``(the batch reports fabric health, the node finding or None)``."""

    readings = {
        gpu: reading
        for gpu, reading in _readings(batch.samples).items()
        if FABRIC_HEALTH_MASK in reading or FABRIC_HEALTH_SUMMARY in reading
    }
    if not readings:
        return False, None
    graded: dict[str, tuple[str, list[str], float, dict[str, GpuMetricSample]]] = {}
    fabric_gpus: set[str] = set()
    for gpu, reading in readings.items():
        anchor = reading.get(FABRIC_HEALTH_MASK) or reading[FABRIC_HEALTH_SUMMARY]
        if not nvswitch_fabric_product(anchor.labels.get("modelName") or batch.product):
            continue
        if anchor.gpu_uuid:
            fabric_gpus.add(anchor.gpu_uuid)
        state, details = _state_of(reading)
        if state is None:
            continue
        held = reading.get(FABRIC_HEALTH_HELD_SECONDS)
        graded[gpu] = (state, details, held.value if held else 0.0, reading)
    if not graded:
        return True, None
    return True, _finding(store, batch, graded, sorted(fabric_gpus))


def _finding(
    store: Any,
    batch: GpuMetricBatch,
    graded: dict[str, tuple[str, list[str], float, dict[str, GpuMetricSample]]],
    fabric_gpus: list[str],
) -> GpuHealthFinding:
    """The node finding over the GPUs in a fault state (module docstring).

    Its GPUs are every GPU of the node's fabric, as for the Fabric Manager
    finding: the full GPU + NVSwitch reset names the whole inventory (the Node
    Agent refuses one that does not); the reason names the faulted GPUs.
    """

    longest = max(held for _state, _details, held, _reading in graded.values())
    critical = sorted(
        gpu
        for gpu, (state, _details, held, _reading) in graded.items()
        if state in CONFIRMATION_SECONDS and held >= CONFIRMATION_SECONDS[state]
    )
    # Confirmed only if no product window touched the confirmation period
    # itself: a state held across a window counts from the window's end.
    window = (
        product_link_window(
            store,
            batch.cluster_id,
            batch.node_id,
            batch.observed_at
            - timedelta(
                seconds=max(CONFIRMATION_SECONDS[graded[gpu][0]] for gpu in critical)
            ),
            batch.observed_at,
        )
        if critical
        else None
    )
    degraded = sorted(gpu for gpu, item in graded.items() if item[0] == DEGRADED)
    text = "; ".join(
        _gpu_text(gpu, reading, details)
        for gpu, (_state, details, _held, reading) in sorted(graded.items())
    )
    if critical and window is None:
        severity = GpuHealthSeverity.CRITICAL
        action, official = RecoveryAction.RESET_GPU, FABRIC_MANAGER_OFFICIAL_ACTION
        source = "NVIDIA_FABRIC_HEALTH_UNHEALTHY"
        reason = (
            f"NVLink fabric health failed on {len(critical)} GPU(s), held >= "
            f"{UNHEALTHY_CONFIRMATION_SECONDS}s (recovery >= "
            f"{RECOVERY_CRITICAL_SECONDS}s) outside product fabric windows "
            f"(DCGM fields 174/175: {text}). Recovery: stop Fabric Manager and the "
            "GPU services, reset all GPUs and NVSwitches, start Fabric Manager"
        )
    elif degraded:
        severity = GpuHealthSeverity.WARNING
        action, official = RecoveryAction.RUN_DIAGNOSTICS, "EXAMINE_GPU_HEALTH"
        source = "NVIDIA_FABRIC_HEALTH_DEGRADED"
        reason = (
            f"NVLink fabric capacity degraded on {len(degraded)} GPU(s) (DCGM "
            f"fields 174/175: {text}); Fabric Manager may run a degraded mode "
            "it applied at initialisation: diagnose, an operator decides"
        )
    else:
        severity = GpuHealthSeverity.WARNING
        action, official = RecoveryAction.NO_ACTION, None
        source = "NVIDIA_FABRIC_HEALTH_PENDING"
        reason = (
            f"NVLink fabric health not yet confirmed (DCGM fields 174/175: {text})"
            + (
                f"; a product window ran meanwhile: {describe_window(window)}"
                if window is not None
                else ""
            )
            + "; evidence only until the state is held outside product windows"
        )
    return GpuHealthFinding(
        finding_id=f"{batch.batch_id}-node-{FABRIC_HEALTH}",
        cluster_id=batch.cluster_id,
        node_id=batch.node_id,
        observed_at=batch.observed_at,
        severity=severity,
        reason=reason,
        canonical_name=FABRIC_HEALTH,
        value=longest,
        threshold_value=float(UNHEALTHY_CONFIRMATION_SECONDS),
        threshold_source=source,
        evidence_ref=batch.evidence_ref,
        automatic_action=action.value,
        policy_source=FABRIC_HEALTH_POLICY_SOURCE,
        policy_version=SITE_METRIC_POLICY_VERSION,
        policy_reference=DCGM_FABRIC_HEALTH_REFERENCE,
        official_action=official,
        runtime_profile_version=batch.runtime_profile_version,
        workload_state=batch.workload_state,
        affected_workload_ids=batch.affected_workload_ids,
        affected_gpu_uuids=fabric_gpus,
        confidence="HIGH" if severity is GpuHealthSeverity.CRITICAL else "MEDIUM",
    )


def track_fabric_findings(
    store: Any,
    batch: GpuMetricBatch,
    findings: list[GpuHealthFinding],
    finding_updates: list[Any],
    findings_by_update: list[GpuHealthFinding | None],
) -> None:
    """The node fabric findings of a batch: the Fabric Manager state (field
    170, ``gpu_fabric_manager``) and the fabric health (174/175)."""

    track_fabric_manager_state(
        store, batch, findings, finding_updates, findings_by_update
    )
    reported, finding = fabric_health_finding(store, batch)
    if not reported:
        return
    key = (batch.cluster_id, batch.node_id, "node", FABRIC_HEALTH)
    if finding is None:
        (previous,) = store.get_gpu_finding_states([key])
        if previous is None or previous.finding is None:
            return
    else:
        findings.append(finding)
    finding_updates.append((key, finding, batch.observed_at))
    findings_by_update.append(finding)


def health_validation_failure(
    observations: Iterable[tuple[GpuMetricSample, datetime | None]],
    graded_gpus: set[str],
) -> str | None:
    """VALIDATE_GPU's fabric health check: every graded GPU's summary is HEALTHY
    or NOT_SUPPORTED; the failure names the others, with the XML evidence."""

    by_gpu: dict[str, dict[str, GpuMetricSample]] = {}
    for sample, _observed_at in observations:
        if sample.canonical_name in {FABRIC_HEALTH_SUMMARY, FABRIC_HEALTH_XML}:
            by_gpu.setdefault(_gpu_key(sample), {})[sample.canonical_name] = sample
    failed = []
    for gpu, reading in sorted(by_gpu.items()):
        summary = reading.get(FABRIC_HEALTH_SUMMARY)
        code = _integer(summary.value) if summary is not None else None
        if (
            gpu not in graded_gpus
            or summary is None
            or code
            in {
                FabricHealthSummary.NOT_SUPPORTED,
                FabricHealthSummary.HEALTHY,
            }
        ):
            continue
        name = (
            FabricHealthSummary(code).name
            if code in set(FabricHealthSummary)
            else f"unknown({summary.value:g})"
        )
        xml = xml_evidence(reading.get(FABRIC_HEALTH_XML))
        failed.append(f"{gpu}={name}" + (f" [{xml}]" if xml else ""))
    if not failed:
        return None
    return f"nvlink_fabric_health_not_healthy({', '.join(failed)})"


def fabric_validation(
    observations: Iterable[tuple[GpuMetricSample, datetime | None]],
) -> tuple[int, dict[str, str], str | None]:
    """``gpu_fabric_manager.fabric_validation`` (the Fabric Manager state of
    every graded GPU) plus :func:`health_validation_failure` over the same
    graded GPUs; both failures are reported together."""

    items = list(observations)
    count, training, failure = fabric_manager_validation(items)
    graded, _unsettled = unsettled_fabric(items)
    health = health_validation_failure(items, set(graded))
    if health is None:
        return count, training, failure
    return count, training, f"{failure}; {health}" if failure else health
