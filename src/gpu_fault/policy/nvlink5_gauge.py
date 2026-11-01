"""An NVLink5 XID read from DCGM field 230 follows its kernel line (batch 235).

DCGM field 230 (``DCGM_FI_DEV_XID_ERRORS``) holds the last XID of a GPU as a
gauge; ``GpuMetricsService._xid_event`` turns a change into an ``XidEvent``
with ``event_source=DCGM_EXPORTER`` and no kernel line. XIDs 144-150 are the
Blackwell NVLink errors whose action the pinned decode table derives from the
IntrInfo and Error Status registers of the kernel line
(``GpuFaultPolicyEngine._workflow_nvlink5``), so the gauge alone can never
decode: it is ``BLOCKED_MISSING_EVIDENCE`` and, before this module, opened a
second, BLOCKED incident beside the kernel-sourced one of the same fault (the
Blackwell field audit's untested section 5.2 assertion; marker association
alone never merges two executions, ``provider_correlation``).

The kernel line is the authoritative source (it carries the registers); the
gauge is a second, poorer report of the same occurrence. A gauge change whose
kernel twin -- the same XID on the same GPU, read from the kernel log, within
:data:`NVLINK5_GAUGE_TWIN_WINDOW` -- is among the correlation candidates takes
the twin's decode and joins its incident through ``correlated_event_id``
(``FaultIngestionService.finalize_xid``). Without a twin -- a kernel line the
collector lost -- the gauge keeps today's fail-closed ``BLOCKED_MISSING_
EVIDENCE``. Nothing changes for an XID outside 144-150, a kernel-sourced event
or a part the NVLink5 workflow does not grade (Hopper never reaches it).
"""

from __future__ import annotations

import re
from datetime import timedelta
from typing import TYPE_CHECKING

from gpu_fault.policy.models import CatalogRule, XidEvent, _Resolution

if TYPE_CHECKING:
    from gpu_fault.policy.engine import GpuFaultPolicyEngine, _XidCorrelation

NVLINK5_XIDS = frozenset(range(144, 151))
#: ``GpuMetricSource.DCGM_EXPORTER`` / ``nvidia_logs`` kernel events.
DCGM_XID_GAUGE_SOURCE = "DCGM_EXPORTER"
KERNEL_XID_SOURCE = "KERNEL_LOG"
#: How far apart the kernel line and the gauge change of one fault may be: the
#: 30 s companion window plus the DCGM path's own lag (the exporter's 15 s
#: refresh, the collector's 15 s scrape) and slack -- the window batch 233
#: gives field 1523's XID 154 evidence (``XID154_EVIDENCE_WINDOW``).
NVLINK5_GAUGE_TWIN_WINDOW = timedelta(seconds=120)
_PCI = re.compile(
    r"(?:(?P<domain>[0-9a-f]{4}|[0-9a-f]{8}):)?"
    r"(?P<bus>[0-9a-f]{2}):(?P<device>[0-9a-f]{2})(?:\.[0-7])?"
)


def is_nvlink5_gauge(event: XidEvent) -> bool:
    """A field-230 change to one of XIDs 144-150."""

    return event.event_source == DCGM_XID_GAUGE_SOURCE and event.xid in NVLINK5_XIDS


def candidate_window(event: XidEvent, companion_window: timedelta) -> timedelta:
    """The candidate window the correlation reads around ``event``: wide enough
    for an NVLink5 gauge to see its kernel twin, the companion window else."""

    if is_nvlink5_gauge(event):
        return max(companion_window, NVLINK5_GAUGE_TWIN_WINDOW)
    return companion_window


def _pci_device(value: str | None) -> tuple[int | None, int, int] | None:
    """``(domain, bus, device)`` of a kernel (``0000:b9:00``) or exporter
    (``00000000:B9:00.0``) spelling; ``None`` when it parses as neither."""

    match = _PCI.fullmatch((value or "").strip().lower())
    if match is None:
        return None
    domain = match.group("domain")
    return (
        None if domain is None else int(domain, 16),
        int(match.group("bus"), 16),
        int(match.group("device"), 16),
    )


def _same_gpu(left: XidEvent, right: XidEvent) -> bool:
    if left.gpu_uuid and right.gpu_uuid:
        return left.gpu_uuid == right.gpu_uuid
    first, second = _pci_device(left.pci_bdf), _pci_device(right.pci_bdf)
    if first is None or second is None:
        return False
    return first[1:] == second[1:] and (
        first[0] is None or second[0] is None or first[0] == second[0]
    )


def nvlink5_gauge_twin(event: XidEvent, candidates: list[XidEvent]) -> XidEvent | None:
    """The kernel event an NVLink5 gauge change reports again, if any.

    Same cluster, node, XID and GPU, a kernel-log source, a compatible boot,
    within :data:`NVLINK5_GAUGE_TWIN_WINDOW`; a twin carrying the registers is
    preferred, then the nearest in time.
    """

    if not is_nvlink5_gauge(event):
        return None
    twins = [
        item
        for item in candidates
        if item.event_id != event.event_id
        and item.event_source == KERNEL_XID_SOURCE
        and item.xid == event.xid
        and item.cluster_id == event.cluster_id
        and item.node_id == event.node_id
        and not item.synthetic
        and _same_gpu(event, item)
        and (
            not event.source_boot_id
            or not item.source_boot_id
            or event.source_boot_id == item.source_boot_id
        )
        and abs(item.observed_at - event.observed_at) <= NVLINK5_GAUGE_TWIN_WINDOW
    ]
    return min(
        twins,
        key=lambda item: (
            item.intr_info is None or item.error_status is None,
            abs(item.observed_at - event.observed_at),
            item.event_id,
        ),
        default=None,
    )


def workflow_nvlink5(
    engine: GpuFaultPolicyEngine,
    event: XidEvent,
    rule: CatalogRule,
    correlation: _XidCorrelation,
) -> _Resolution:
    """``WORKFLOW_NVLINK5_ERR``: the engine's decode, or its kernel twin's."""

    twin = nvlink5_gauge_twin(event, correlation.companion_events)
    if twin is None:
        return engine._workflow_nvlink5(event, rule, correlation)
    resolution = engine._workflow_nvlink5(twin, rule, correlation)
    resolution.correlated_event_id = twin.event_id
    resolution.reasons.insert(
        0,
        f"DCGM field 230 reports XID {event.xid} for this GPU "
        f"{abs(twin.observed_at - event.observed_at).total_seconds():.0f} s from "
        f"the kernel line of event {twin.event_id}, which carries the registers "
        "the decode needs: the gauge change is that fault's evidence and follows "
        "its decode",
    )
    return resolution
