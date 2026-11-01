"""NVLink5 counters of HGX B200/B300: DCGM fields 1204-1209, 1211-1214, 1219, 1220.

Batch 235, from the 2026-11-01 Blackwell field audit. Every field here is
exported by the bundled counters CSV and never required (the link-recovery
trio 1211-1213 only *answers* for the NVLink completeness groups,
``dcgm_fields.NVLINK5_LINK_RECOVERY_CANONICAL``). None is reported by a
Hopper part, so an H200 scrape carries none of them and grades nothing new.
Unit-tested from NVIDIA documentation, not live-verified: the site has no
B-series node.

Evidence -- every quote checked against the live page or tag named:

* NVML API Reference (docs version 615), "Field Value Enums"
  (:data:`NVML_FIELD_VALUE_REFERENCE`): the data-link counters are gone on
  Blackwell, :data:`NVML_DL_UNSUPPORTED_QUOTE`; the replacement counters'
  one-line meanings are :data:`NVML_FIELD_QUOTES`.
* DCGM Field Identifiers (docs "Version: 4.6 (Latest)"), the entries of
  :data:`NVLINK5_FIELDS`: :data:`DCGM_NVLINK5_SCOPE_QUOTE` -- aggregates.
* DCGM Feature Overview (docs 4.6), "NVLink COUNT Fields":
  :data:`DCGM_AGGREGATE_QUOTE` -- the aggregate sums the *active* ports, so it
  falls when a port leaves the active set. "Background Health Checks" >
  "Failure Conditions": :data:`DCGM_NVLINK_CRITICAL_QUOTE`.
* DCGM v4.7.0 ``modules/health/DcgmHealthWatch.cpp``
  (:data:`DCGM_HEALTH_WATCH_REFERENCE`): ``MonitorNVLinkErrorFields`` grades
  1211 above 0 as WARN ("link recovery successful events counter") and 1212
  above 0 as FAIL ("link recovery failed events counter") on Blackwell or
  newer; ``MonitorNVLink5Fields`` FAILs on any change within a minute of
  {1214, 1204-1209, 1213, 1219, 1220} (``common/DcgmGPUHardwareLimits.h``
  ``#define DCGM_LIMIT_MAX_NVLINK_ERROR 1``); both raise
  ``DCGM_FR_NVLINK_ERROR_CRITICAL`` (``dcgmlib/dcgm_errors.h``:
  :data:`DCGM_NVLINK_CRITICAL_MESSAGE`), whose next step is
  :data:`DCGM_FIELD_DIAGNOSTIC_NEXT`.

The rules, all on the increment between two accepted samples (never on the
reading: a counter that already carries a count when the collector starts is a
baseline):

* 1212 link recovery failed, increment >= 1: CRITICAL ``DRAIN``, official
  ``RUN_FIELD_DIAGNOSTIC`` (``NVIDIA_DCGM_HEALTH``: DCGM's FAIL). A failed
  recovery declared the link down.
* 1211 successful / 1213 all link recoveries: the NVLink family rule below
  (:func:`nvlink_family_decision`, WARNING ``RUN_DIAGNOSTICS``,
  ``SITE_NVLINK_COUNTER_EVIDENCE``), so every NVLink family of
  GF-GENERAL-LIVE-020 grades its NVLink5 member like its Hopper ones.
* 1204-1209, 1214, 1219, 1220 are ``nvlink_<x>_error_total`` counters, so the
  NVLink family rule ``GpuMetricsService._link_decision`` grades them like the
  Hopper data-link counters: WARNING ``RUN_DIAGNOSTICS`` on an increment of at
  least ``nvlink_error_delta_critical`` (default 1, DCGM's limit). 1220 is
  documented without the NVLink5-only note and its Blackwell support is
  unknown: graded only when it reports.
* A decrease grades nothing and re-baselines the counter (the aggregate lost an
  active port; ``gpu_composites.metric_counter_delta``).
* Not graded at all: the BER readings 1215-1218 (DCGM Release Notes 4.7.0,
  "Bug Fixes" > "Active health checks (dcgmi diag)":
  :data:`DCGM_BER_REMOVED_QUOTE`) and the FEC history 1404-1419 (corrected
  symbols); neither is exported.

:func:`nvlink5_gate` then applies to every NVLink5 decision: an increment whose
interval overlaps a product quiesce, reset, reboot or Fabric Manager/NVLSM
window (``gpu_product_windows``) grades nothing -- the links retrain there and
the window's VALIDATE_GPU judges the GPU afterwards; while an XID 144-150 or
155 event or open incident owns the GPU (the kernel line carries the decode,
``policy/xid_cross_source``), the finding is evidence only (``NO_ACTION``). Both
NVLink composites take NVLink5 findings as components
(:func:`nvlink_component`); the link-recovery trio counts as one error class
(:func:`nvlink_class`: one recovery increments 1211 or 1212 *and* 1213), an
evidence-only finding is no component, and a CRITICAL ``DRAIN`` component
makes the composite CRITICAL ``DRAIN`` (:func:`nvlink_composite_grade`) so a
WARNING composite never swallows a failed link recovery.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import timedelta
from typing import Any

from gpu_fault.dcgm_fields import NVLINK5_LINK_RECOVERY_CANONICAL
from gpu_fault.gpu_metric_models import SITE_METRIC_POLICY_VERSION, GpuHealthSeverity
from gpu_fault.gpu_power_policy import (
    NVIDIA_DCGM_HEALTH_REFERENCE,
    NVIDIA_DCGM_POLICY_VERSION,
)
from gpu_fault.gpu_product_windows import describe_window, product_link_window
from gpu_fault.gpu_recovery_action_policy import OPEN_INCIDENT_STATES
from gpu_fault.models import RecoveryAction
from gpu_fault.pci_identity import same_pci_device

LOGGER = logging.getLogger(__name__)
NVLINK5_RECOVERY_SUCCESSFUL, NVLINK5_RECOVERY_FAILED, NVLINK5_RECOVERY_EVENTS = (
    NVLINK5_LINK_RECOVERY_CANONICAL
)
NVLINK5_LINK_RECOVERY = frozenset(NVLINK5_LINK_RECOVERY_CANONICAL)
#: canonical name -> (DCGM field id, the DCGM 4.5.3 exporter name the counters
#: CSV exports -- go-dcgm v1.4611.2 legacy map -- and the DCGM 4.6 name).
NVLINK5_FIELDS: dict[str, tuple[int, str, str]] = {
    "nvlink_rx_malformed_packet_error_total": (
        1204,
        "DCGM_FI_DEV_NVLINK_COUNT_RX_MALFORMED_PACKET_ERRORS",
        "DCGM_FI_DEV_NVLINK_RX_PACKET_MALFORMED_TOTAL",
    ),
    "nvlink_rx_buffer_overrun_error_total": (
        1205,
        "DCGM_FI_DEV_NVLINK_COUNT_RX_BUFFER_OVERRUN_ERRORS",
        "DCGM_FI_DEV_NVLINK_RX_PACKET_DROPPED_TOTAL",
    ),
    "nvlink_rx_packet_error_total": (
        1206,
        "DCGM_FI_DEV_NVLINK_COUNT_RX_ERRORS",
        "DCGM_FI_DEV_NVLINK_RX_ERROR_TOTAL",
    ),
    "nvlink_rx_remote_error_total": (
        1207,
        "DCGM_FI_DEV_NVLINK_COUNT_RX_REMOTE_ERRORS",
        "DCGM_FI_DEV_NVLINK_RX_REMOTE_ERROR_TOTAL",
    ),
    "nvlink_rx_general_error_total": (
        1208,
        "DCGM_FI_DEV_NVLINK_COUNT_RX_GENERAL_ERRORS",
        "DCGM_FI_DEV_NVLINK_RX_GENERAL_ERROR_TOTAL",
    ),
    "nvlink_local_link_integrity_error_total": (
        1209,
        "DCGM_FI_DEV_NVLINK_COUNT_LOCAL_LINK_INTEGRITY_ERRORS",
        "DCGM_FI_DEV_NVLINK_INTEGRITY_ERROR_TOTAL",
    ),
    NVLINK5_RECOVERY_SUCCESSFUL: (
        1211,
        "DCGM_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_SUCCESSFUL_EVENTS",
        "DCGM_FI_DEV_NVLINK_RECOVERY_SUCCESSFUL_TOTAL",
    ),
    NVLINK5_RECOVERY_FAILED: (
        1212,
        "DCGM_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_FAILED_EVENTS",
        "DCGM_FI_DEV_NVLINK_RECOVERY_FAILED_TOTAL",
    ),
    NVLINK5_RECOVERY_EVENTS: (
        1213,
        "DCGM_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_EVENTS",
        "DCGM_FI_DEV_NVLINK_RECOVERY_EVENT_TOTAL",
    ),
    "nvlink_rx_symbol_error_total": (
        1214,
        "DCGM_FI_DEV_NVLINK_COUNT_RX_SYMBOL_ERRORS",
        "DCGM_FI_DEV_NVLINK_RX_SYMBOL_ERROR_TOTAL",
    ),
    "nvlink_effective_error_total": (
        1219,
        "DCGM_FI_DEV_NVLINK_COUNT_EFFECTIVE_ERRORS",
        "DCGM_FI_DEV_NVLINK_EFFECTIVE_ERROR_TOTAL",
    ),
    "nvlink_ecc_data_error_total": (
        1220,
        "DCGM_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_TOTAL",
        "DCGM_FI_DEV_NVLINK_ECC_ERROR_TOTAL",
    ),
}
NVLINK5_COUNTERS = frozenset(NVLINK5_FIELDS)
NVLINK5_ERROR_COUNTERS = NVLINK5_COUNTERS - NVLINK5_LINK_RECOVERY
#: The collector's ``DCGM_METRICS`` entries: both exporter names of each field.
NVLINK5_DCGM_METRICS: dict[str, tuple[str, str | None]] = {
    name: (canonical, "events" if canonical in NVLINK5_LINK_RECOVERY else "errors")
    for canonical, (_field, legacy, current) in NVLINK5_FIELDS.items()
    for name in (legacy, current)
}
#: Each field's two names, one alias pair (``gpu_environment_policy``).
NVLINK5_DCGM_ALIASES = tuple(
    frozenset({legacy, current}) for _field, legacy, current in NVLINK5_FIELDS.values()
)
NVML_FIELD_VALUE_REFERENCE = (
    "https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlFieldValueEnums.html"
    "#c.NVML_FI_DEV_NVLINK_ERROR_DL_CRC"
)
NVML_DL_UNSUPPORTED_QUOTE = (
    "This is unsupported for Blackwell+ Please use "
    "NVML_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_*"
)
#: NVML v615, ``NVML_FI_DEV_NVLINK_COUNT_*``: the field id -> the verbatim line.
NVML_FIELD_QUOTES = {
    1204: "Number of packets Rx on a link where packets are malformed.",
    1205: "Number of packets that were discarded on Rx due to buffer overrun.",
    1206: "Total number of packets with errors Rx on a link.",
    1207: "Total number of packets Rx - stomp/EBP marker.",
    1208: "Total number of packets Rx with header mismatch.",
    1209: "Total number of times that the count of local errors exceeded a threshold.",
    1211: (
        "Number of times link went from Up to recovery, succeeded and link came "
        "back up."
    ),
    1212: (
        "Number of times link went from Up to recovery, failed and link was "
        "declared down."
    ),
    1213: "Number of times link went from Up to recovery, irrespective of the result.",
    1214: "Number of errors in rx symbols.",
    1219: "Sum of the number of errors in each Nvlink packet.",
    1220: "NVLink data ECC Error Counter total for all Links.",
}
DCGM_FIELD_REFERENCE = (
    "https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html"
)
DCGM_NVLINK5_SCOPE_QUOTE = (
    "Returns aggregate value across all links. Not supported on NVLink4 and earlier."
)
DCGM_AGGREGATE_REFERENCE = (
    "https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html"
    "#nvlink-count-fields"
)
DCGM_AGGREGATE_QUOTE = (
    "Aggregate (GPU entity) — returns the sum across all active NVLink ports on a GPU."
)
DCGM_HEALTH_REFERENCE = (
    "https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html"
    "#background-health-checks"
)
DCGM_NVLINK_CRITICAL_QUOTE = (
    "An NVLink reported 1 or more errors. This includes datalink CRC, recovery, and "
    "replay errors (pre-Blackwell GPUs) and link recovery events (Blackwell+ GPUs)."
)
DCGM_HEALTH_WATCH_REFERENCE = (
    "https://github.com/NVIDIA/DCGM/blob/v4.7.0/modules/health/DcgmHealthWatch.cpp"
)
DCGM_NVLINK_CRITICAL_MESSAGE = (
    "Detected %ld %s NvLink errors on GPU %u's NVLink (should be 0)"
)
DCGM_FIELD_DIAGNOSTIC_NEXT = "Run a field diagnostic on the GPU."
DCGM_RELEASE_NOTES_REFERENCE = (
    "https://docs.nvidia.com/datacenter/dcgm/latest/release-notes/changelog.html"
)
DCGM_BER_REMOVED_QUOTE = (
    "Removed NVLink bit-error-rate thresholds that caused false health and "
    "diagnostic failures on healthy GB200 systems."
)
#: DCGM's limit (``DCGM_LIMIT_MAX_NVLINK_ERROR``) for the link-recovery rules.
NVLINK5_RECOVERY_DELTA = 1.0
NVLINK5_OFFICIAL_ACTION = "RUN_FIELD_DIAGNOSTIC"
#: XIDs whose kernel line describes the same NVLink5 fault: 144-150 (the
#: decoded NVLink5 errors) and 155 (NVLINK_SW_DEFINED_ERROR, link-down events).
NVLINK5_XIDS = frozenset({*range(144, 151), 155})
#: As ``gpu_recovery_action_policy.XID154_EVIDENCE_WINDOW``: the XID event's
#: 30 s correlation window plus the DCGM path's lag and slack.
NVLINK5_XID_EVIDENCE_WINDOW = timedelta(seconds=120)
NVLINK5_EVIDENCE_SOURCE = "NVLINK5_XID_OWNS_GPU"
#: The link-recovery trio is one error class (see the module docstring).
NVLINK5_RECOVERY_CLASS = "nvlink_link_recovery"


def field_id(name: str) -> int | None:
    """The DCGM field id of an NVLink5 canonical name."""

    entry = NVLINK5_FIELDS.get(name)
    return entry[0] if entry is not None else None


def field_reference(name: str) -> str:
    """The DCGM Field Identifiers entry of an NVLink5 counter (DCGM 4.6 name)."""

    return f"{DCGM_FIELD_REFERENCE}#c.{NVLINK5_FIELDS[name][2]}"


def nvlink_family_decision(
    thresholds: Any, delta: float | None
) -> dict[str, Any] | None:
    """The NVLink family rule (``GpuMetricsService._link_decision``): an increment
    of an NVLink error counter of at least ``nvlink_error_delta_critical``
    (default 1, DCGM's limit) is WARNING ``RUN_DIAGNOSTICS``."""

    if delta is None or delta < thresholds.nvlink_error_delta_critical:
        return None
    source = (
        "NVIDIA_DCGM_HEALTH"
        if thresholds.nvlink_error_delta_critical == 1
        else "SITE_OVERRIDE_NVIDIA_DCGM_HEALTH"
    )
    return {
        "severity": GpuHealthSeverity.WARNING,
        "reason": "NVLink error counter increased; diagnose link errors and rate",
        "automatic_action": RecoveryAction.RUN_DIAGNOSTICS.value,
        "policy_source": "SITE_NVLINK_COUNTER_EVIDENCE",
        "policy_version": NVIDIA_DCGM_POLICY_VERSION
        if source == "NVIDIA_DCGM_HEALTH"
        else SITE_METRIC_POLICY_VERSION,
        "policy_reference": NVIDIA_DCGM_HEALTH_REFERENCE,
        "official_action": "EXAMINE_GPU_HEALTH",
    }


def recovery_decision(
    sample: Any, delta: float | None, thresholds: Any
) -> dict[str, Any] | None:
    """Fields 1211-1213 on an increment (see the module docstring): 1212 at
    DCGM's limit, 1211 and 1213 by the NVLink family rule, so the GF-GENERAL-
    LIVE-020 families grade their NVLink5 member like every other member."""

    name = sample.canonical_name
    if name not in NVLINK5_LINK_RECOVERY or delta is None:
        return None
    if name != NVLINK5_RECOVERY_FAILED:
        return nvlink_family_decision(thresholds, delta)
    if delta < NVLINK5_RECOVERY_DELTA:
        return None
    return {
        "severity": GpuHealthSeverity.CRITICAL,
        "reason": (
            f"NVLink5 link recovery failed {delta:g} time(s) (DCGM field 1212: "
            "a link went from Up to recovery, failed and was declared down); "
            "drain the node and run the field diagnostic"
        ),
        "automatic_action": RecoveryAction.DRAIN.value,
        "threshold_value": NVLINK5_RECOVERY_DELTA,
        "threshold_source": "NVIDIA_DCGM_NVLINK5_RECOVERY_FAILED",
        "policy_source": "NVIDIA_DCGM_HEALTH",
        "policy_version": NVIDIA_DCGM_POLICY_VERSION,
        "policy_reference": field_reference(name),
        "official_action": NVLINK5_OFFICIAL_ACTION,
    }


def _names_gpu(gpu_uuid: Any, pci_bdf: Any, sample: Any) -> bool:
    """Whether an XID record names the sample's GPU; one naming no GPU is
    node-wide and owns every GPU (fail toward evidence, not a second plan).
    A PCI address names a device in any spelling (``pci_identity``, b236)."""

    if gpu_uuid and sample.gpu_uuid:
        return bool(gpu_uuid == sample.gpu_uuid)
    if pci_bdf and sample.pci_bdf:
        return same_pci_device(str(pci_bdf).lower(), str(sample.pci_bdf).lower())
    return not gpu_uuid and not pci_bdf


def _decision_xid(decision: Any) -> int | None:
    """The XID of a stored XID policy decision (its marker's ``XID <n>``)."""

    reason = str(getattr(getattr(decision, "marker", None), "raw_reason", "") or "")
    kind, _, code = reason.partition(" ")
    return int(code) if kind == "XID" and code.isdigit() else None


def nvlink_xid_owner(store: Any, batch: Any, sample: Any) -> str | None:
    """The XID 144-150/155 event or open incident that owns the GPU, or ``None``.

    As ``gpu_recovery_action_policy.xid154_owner``: an XID event for the GPU
    inside :data:`NVLINK5_XID_EVIDENCE_WINDOW` of the batch (the XID event
    table keeps eight 30 s windows), or an open incident of the node whose XID
    decision is one of :data:`NVLINK5_XIDS` for the GPU (or no GPU).
    """

    events = store.list_xid_events(
        batch.cluster_id,
        batch.node_id,
        observed_after=batch.observed_at - NVLINK5_XID_EVIDENCE_WINDOW,
        observed_before=batch.observed_at + NVLINK5_XID_EVIDENCE_WINDOW,
    )
    for event in sorted(events, key=lambda item: (item.observed_at, item.event_id)):
        if event.xid in NVLINK5_XIDS and _names_gpu(
            event.gpu_uuid, event.pci_bdf, sample
        ):
            return f"XID {event.xid} event {event.event_id}"
    for incident in store.list_incidents_by_state(
        batch.cluster_id, OPEN_INCIDENT_STATES, node_ids={batch.node_id}
    ):
        if incident.event_type != "XID" or (
            incident.gpu_uuids and sample.gpu_uuid not in incident.gpu_uuids
        ):
            continue
        xid = _decision_xid(store.get_xid_policy_decision(incident.event_id))
        if xid in NVLINK5_XIDS:
            return f"XID {xid} incident {incident.incident_id}"
    return None


def nvlink5_gate(
    store: Any,
    batch: Any,
    sample: Any,
    decision: dict[str, Any] | None,
    previous_latest: Any,
) -> dict[str, Any] | None:
    """The product-window suppression and the XID evidence of an NVLink5 grade.

    ``decision`` is what the NVLink5 or NVLink family rule decided for the
    sample; anything that is not an NVLink5 counter passes unchanged. An
    increment whose interval -- from the previous accepted sample to this
    batch -- overlaps a product window grades nothing; one an XID 144-150/155
    owns is evidence only.
    """

    if decision is None or sample.canonical_name not in NVLINK5_COUNTERS:
        return decision
    since = getattr(previous_latest, "observed_at", None) or batch.observed_at
    window = product_link_window(
        store, batch.cluster_id, batch.node_id, since, batch.observed_at
    )
    if window is not None:
        LOGGER.info(
            "NVLink5 %s increase on %s/%s graded nothing: inside the product window %s",
            sample.canonical_name,
            batch.node_id,
            sample.gpu_uuid or sample.pci_bdf,
            describe_window(window),
        )
        return None
    field = field_id(sample.canonical_name)
    owner = nvlink_xid_owner(store, batch, sample)
    if owner is None:
        return {**decision, "policy_reference": field_reference(sample.canonical_name)}
    return {
        **decision,
        "severity": GpuHealthSeverity.WARNING,
        "reason": (
            f"NVLink5 counter {sample.canonical_name} (DCGM field {field}) "
            f"increased; evidence for {owner}, which owns this GPU's NVLink "
            "recovery: no second workflow"
        ),
        "automatic_action": RecoveryAction.NO_ACTION.value,
        "threshold_source": NVLINK5_EVIDENCE_SOURCE,
        "policy_reference": field_reference(sample.canonical_name),
    }


def nvlink_class(name: str) -> str:
    """The NVLink error class a canonical name belongs to (module docstring)."""

    return NVLINK5_RECOVERY_CLASS if name in NVLINK5_LINK_RECOVERY else name


def nvlink_component(finding: Any) -> bool:
    """Whether a finding is an NVLink composite component: an NVLink error or
    NVLink5 counter finding that is not evidence only."""

    name = str(getattr(finding, "canonical_name", ""))
    return (
        name.startswith("nvlink_")
        and (name.endswith("_error_total") or name in NVLINK5_LINK_RECOVERY)
        and getattr(finding, "automatic_action", None) != RecoveryAction.NO_ACTION.value
    )


def nvlink_composite_grade(components: Iterable[Any], reason: str) -> dict[str, Any]:
    """An NVLink composite's ``severity`` / ``action`` / ``reason``: WARNING
    ``RUN_DIAGNOSTICS`` with ``reason``, unless a component is CRITICAL
    ``DRAIN`` (a failed NVLink5 link recovery) -- then the composite is too."""

    drains = sorted(
        {
            item.canonical_name
            for item in components
            if item.severity is GpuHealthSeverity.CRITICAL
            and item.automatic_action == RecoveryAction.DRAIN.value
        }
    )
    if not drains:
        return {
            "severity": GpuHealthSeverity.WARNING,
            "action": RecoveryAction.RUN_DIAGNOSTICS,
            "reason": reason,
        }
    return {
        "severity": GpuHealthSeverity.CRITICAL,
        "action": RecoveryAction.DRAIN,
        "reason": (
            f"{reason}; {', '.join(drains)} is CRITICAL (a failed NVLink link "
            "recovery declared a link down): drain"
        ),
    }
