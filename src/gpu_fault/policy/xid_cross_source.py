"""One XID occurrence, two reports: the kernel NVRM line and DCGM field 230.

Every real XID reaches the control plane twice. The kernel collector posts the
NVRM line at once (``NVRM: Xid (PCI:0000:b9:00): <xid>, ...``,
``event_source=KERNEL_LOG``); DCGM field 230 changes on the exporter's next
refresh, 15-30 s later, and ``GpuMetricsService._xid_event`` turns the change
into a second ``XidEvent`` (``event_source=DCGM_EXPORTER``, labelled
``00000000:B9:00.0``, no kernel text). NVIDIA documents the field as a value,
not as an event of its own: DCGM Field Identifiers (DCGM documentation
"Version: 4.6 (Latest)", :data:`DCGM_XID_FIELD_REFERENCE`) list
``DCGM_FI_DEV_XID_ERROR 230`` -- the name the exporter's counters CSV still
writes ``DCGM_FI_DEV_XID_ERRORS`` -- as "XID errors." and "The value is the
specific XID error"; NVIDIA's dcgm-exporter default counters call the gauge the
"Value of the last XID error encountered." (:data:`DCGM_EXPORTER_XID_QUOTE`).

What those semantics support, and no more (batch 237):

- A field-230 change reports one XID occurrence on one GPU -- its code and its
  GPU, nothing else: no time of its own beyond the sample, no count, no text.
  It is the kernel line's *kernel twin* when the line is the same XID on the
  same GPU (UUID, else the PCI device in either spelling,
  ``pci_identity``), from a compatible boot, within
  :data:`XID_REPORT_TWIN_WINDOW` of it.
- A twin pairs once: one change with one kernel line, one kernel line with one
  change (``paired`` -- the reports a stored decision already joined). A
  repeated identical XID leaves the value unchanged, so it arrives as kernel
  lines only, and a line beyond the window or past an already paired change is
  a new occurrence that plans its own recovery.
- The kernel line carries the evidence; the change only repeats it. A change
  with a kernel twin takes the twin's decode and joins its incident through
  ``correlated_event_id`` (:func:`follow_report_twin`; batch 235 did this for
  XIDs 144-150 only, ``policy/nvlink5_gauge``). A kernel line that finds a
  change already decided the same way joins that incident instead
  (``XidCorrelationCoordinator``). Either way the second report is a duplicate
  observation of the first one's incident, whatever its workflow is doing
  (``provider_correlation.report_twin_incident``) -- the store's merge groups
  used to decide that, and outside them the second report ran its own
  remediation (a second GPU reset, a second stop and restart of the job).
- Where the decode reads the kernel line's text (:data:`KERNEL_DECODED_ACTIONS`:
  the XID 154 recovery-action label, the XID 74 and 144-150 registers) a change
  without a twin waits up to the window for its kernel line
  (``PENDING_CORRELATION``) instead of quarantining the node beside the line's
  own decision; a line lost for good leaves today's fail-closed
  ``BLOCKED_MISSING_EVIDENCE`` once the window closes.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import TYPE_CHECKING

from gpu_fault.pci_identity import same_pci_device
from gpu_fault.policy.models import (
    ActionDisposition,
    ActionSource,
    Containment,
    FaultPolicyDecision,
    XidEvent,
    _Resolution,
)

if TYPE_CHECKING:
    from gpu_fault.policy.engine import GpuFaultPolicyEngine, _XidCorrelation

#: ``nvidia_logs`` kernel events and ``GpuMetricSource.DCGM_EXPORTER`` changes.
KERNEL_XID_SOURCE = "KERNEL_LOG"
DCGM_XID_GAUGE_SOURCE = "DCGM_EXPORTER"
REPORT_SOURCES = frozenset({KERNEL_XID_SOURCE, DCGM_XID_GAUGE_SOURCE})
#: How far apart the kernel line and the field-230 change of one XID may be:
#: the 30 s companion window plus the DCGM path's own lag (the exporter's 15 s
#: refresh, the collector's 15 s scrape) and slack -- batch 235's window, which
#: batch 233 gives field 1523's XID 154 evidence too (``XID154_EVIDENCE_WINDOW``).
XID_REPORT_TWIN_WINDOW = timedelta(seconds=120)
#: Catalog actions whose decode reads the kernel line: the recovery-action label
#: of XID 154, the seven registers and link of XID 74, IntrInfo and Error Status
#: of XID 144-150. Field 230 carries none of it.
KERNEL_DECODED_ACTIONS = frozenset(
    {"XID_154", "WORKFLOW_NVLINK_ERR", "WORKFLOW_NVLINK5_ERR"}
)
#: More same-XID reports than this around one event make the pairing ambiguous
#: (an XID storm on one GPU); no twin is taken then and each report plans as
#: it always did.
MAX_PAIRING_CANDIDATES = 16
DCGM_XID_FIELD_DOCS_VERSION = "4.6"
DCGM_XID_FIELD_REFERENCE = (
    "https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/"
    "dcgm-api-field-ids.html#c.DCGM_FI_DEV_XID_ERROR"
)
DCGM_XID_FIELD_QUOTE = ("XID errors.", "The value is the specific XID error")
#: dcgm-exporter 4.8.4, the exporter of the pinned image ``4.6.1-4.8.4``.
DCGM_EXPORTER_XID_REFERENCE = (
    "https://github.com/NVIDIA/dcgm-exporter/blob/4.8.4/etc/default-counters.csv"
)
DCGM_EXPORTER_XID_QUOTE = "Value of the last XID error encountered."
FIELD_230_EVIDENCE = (
    'DCGM field 230 ("The value is the specific XID error", DCGM Field '
    f"Identifiers, DCGM docs {DCGM_XID_FIELD_DOCS_VERSION})"
)


def is_xid_gauge(event: XidEvent) -> bool:
    return event.event_source == DCGM_XID_GAUGE_SOURCE


def same_gpu(left: XidEvent, right: XidEvent) -> bool:
    """The UUID decides when both carry one, else the PCI device (any spelling)."""

    if left.gpu_uuid and right.gpu_uuid:
        return left.gpu_uuid == right.gpu_uuid
    if left.pci_bdf and right.pci_bdf:
        return same_pci_device(left.pci_bdf.lower(), right.pci_bdf.lower())
    return False


def report_twins(left: XidEvent, right: XidEvent) -> bool:
    """Whether a kernel line and a field-230 change report one XID occurrence."""

    return (
        {left.event_source, right.event_source} == REPORT_SOURCES
        and left.event_id != right.event_id
        and left.cluster_id == right.cluster_id
        and left.node_id == right.node_id
        and left.xid == right.xid
        and not left.synthetic
        and not right.synthetic
        and same_gpu(left, right)
        and (
            not left.source_boot_id
            or not right.source_boot_id
            or left.source_boot_id == right.source_boot_id
        )
        and abs(left.observed_at - right.observed_at) <= XID_REPORT_TWIN_WINDOW
    )


def candidate_window(event: XidEvent, companion_window: timedelta) -> timedelta:
    """The window the correlation reads around ``event``: a report's twin lies
    within :data:`XID_REPORT_TWIN_WINDOW`, and a report the twin already joined
    within the window of the twin, so a kernel line or a change reads twice it."""

    if event.event_source in REPORT_SOURCES:
        return max(companion_window, 2 * XID_REPORT_TWIN_WINDOW)
    return companion_window


def pairing_candidates(event: XidEvent, candidates: list[XidEvent]) -> list[XidEvent]:
    """The other kernel lines and changes of this XID on this GPU in reach."""

    return [
        item
        for item in candidates
        if item.event_id != event.event_id
        and item.event_source in REPORT_SOURCES
        and item.xid == event.xid
        and item.cluster_id == event.cluster_id
        and item.node_id == event.node_id
        and same_gpu(event, item)
    ]


def paired_reports(
    event: XidEvent,
    candidates: list[XidEvent],
    decision_of: Callable[[str], FaultPolicyDecision | None],
) -> frozenset[str] | None:
    """Reports already paired with a twin other than ``event``, read from the
    stored decisions; ``None`` when there are too many to tell (a storm).

    A pair is recorded on the later report's decision (``correlated_event_id``
    naming its twin); both of its reports are taken. A pair with ``event`` in it
    is ``event``'s own and takes nothing from it.
    """

    relevant = pairing_candidates(event, candidates)
    if len(relevant) > MAX_PAIRING_CANDIDATES:
        return None
    by_id = {item.event_id: item for item in candidates}
    taken: set[str] = set()
    for item in relevant:
        decision = decision_of(item.event_id)
        partner = by_id.get(decision.correlated_event_id or "") if decision else None
        if (
            partner is not None
            and report_twins(item, partner)
            and event.event_id != partner.event_id
        ):
            taken.update((item.event_id, partner.event_id))
    return frozenset(taken)


def report_twin(
    event: XidEvent,
    candidates: list[XidEvent],
    *,
    paired: frozenset[str] = frozenset(),
) -> XidEvent | None:
    """The other source's report of ``event``'s occurrence, if one is free.

    A change prefers a kernel line carrying decode evidence, then the nearest in
    time; a kernel line takes the nearest change.
    """

    twins = [
        item
        for item in candidates
        if item.event_id not in paired and report_twins(event, item)
    ]
    return min(
        twins,
        key=lambda item: (
            not (
                item.registers
                or item.xid_154_action is not None
                or (item.intr_info is not None and item.error_status is not None)
            ),
            abs(item.observed_at - event.observed_at),
            item.event_id,
        ),
        default=None,
    )


def follow_report_twin(
    engine: GpuFaultPolicyEngine,
    event: XidEvent,
    correlation: _XidCorrelation,
) -> _Resolution | None:
    """A field-230 change's decision when its kernel line decides it.

    With a free kernel twin the change takes the twin's decode and joins its
    incident; without one, an XID whose decode reads the kernel line waits for
    it while the window is open. ``None`` leaves the engine to decide.
    """

    if not is_xid_gauge(event) or correlation.paired_report_ids is None:
        return None
    twin = report_twin(
        event, correlation.companion_events, paired=correlation.paired_report_ids
    )
    if twin is not None:
        resolution = engine._resolve_xid(
            twin,
            companion_events=correlation.companion_events,
            xid45_window_closed=correlation.xid45_window_closed,
            xid154_window_closed=correlation.xid154_window_closed,
            follow_twin=False,
        )
        resolution.correlated_event_id = twin.event_id
        resolution.reasons.insert(
            0,
            f"{FIELD_230_EVIDENCE} reports XID {event.xid} for this GPU "
            f"{abs(twin.observed_at - event.observed_at).total_seconds():.0f} s from "
            f"the kernel line of event {twin.event_id}: one occurrence; this report "
            "follows that line's decode and joins its incident",
        )
        return resolution
    rule = engine._rules.get(event.xid)
    if (
        correlation.report_twin_window_closed
        or rule is None
        or rule.immediate_action not in KERNEL_DECODED_ACTIONS
        or engine._product_gate(event, rule) is not None
    ):
        return None
    return _Resolution(
        source=ActionSource.NVIDIA_CATALOG,
        disposition=ActionDisposition.PENDING_CORRELATION,
        official_action=rule.immediate_action,
        investigatory_action=rule.investigatory_action,
        action=None,
        containment=Containment.UNKNOWN,
        reasons=[
            f"{FIELD_230_EVIDENCE} carries the XID code only and the "
            f"{rule.immediate_action} decode of XID {event.xid} reads the kernel "
            f"line: waiting up to {XID_REPORT_TWIN_WINDOW.total_seconds():.0f} s "
            "for it"
        ],
    )


def report_wait_window(engine: GpuFaultPolicyEngine, event: XidEvent) -> timedelta:
    """How long a pending report's correlation stays open (the companion
    window, or the twin window for a change waiting for its kernel line)."""

    companion = timedelta(seconds=engine.policy.companion_window_seconds)
    rule = engine._rules.get(event.xid)
    if (
        is_xid_gauge(event)
        and rule is not None
        and rule.immediate_action in KERNEL_DECODED_ACTIONS
    ):
        return max(companion, XID_REPORT_TWIN_WINDOW)
    return companion


def same_report_decision(
    stored: FaultPolicyDecision, decision: FaultPolicyDecision
) -> bool:
    """Whether the first report's stored decision plans what this one would."""

    return (
        stored.disposition is decision.disposition
        and stored.action is decision.action
        and stored.safety_action is decision.safety_action
        and stored.official_action == decision.official_action
        and stored.pre_actions == decision.pre_actions
    )


def join_gauge_twin(
    event: XidEvent,
    decision: FaultPolicyDecision,
    candidates: list[XidEvent],
    paired: frozenset[str] | None,
    decision_of: Callable[[str], FaultPolicyDecision | None],
) -> FaultPolicyDecision:
    """A kernel line whose field-230 change was decided first, the same way,
    joins that change's incident (``correlated_event_id``); anything else -- no
    change, a pending or different one, a companion already chosen -- keeps the
    line's own decision."""

    if (
        event.event_source != KERNEL_XID_SOURCE
        or paired is None
        or decision.correlated_event_id
        or decision.disposition is ActionDisposition.PENDING_CORRELATION
    ):
        return decision
    twin = report_twin(event, candidates, paired=paired)
    stored = decision_of(twin.event_id) if twin is not None else None
    if (
        twin is None
        or stored is None
        or stored.disposition is ActionDisposition.PENDING_CORRELATION
        or not stored.incident_id
        or not same_report_decision(stored, decision)
    ):
        return decision
    return decision.model_copy(
        update={
            "correlated_event_id": twin.event_id,
            "reasons": [
                *decision.reasons,
                f"the {FIELD_230_EVIDENCE} report {twin.event_id} of this XID on "
                f"this GPU, {abs(twin.observed_at - event.observed_at).total_seconds():.0f}"
                " s away, was decided the same way first: one occurrence; this line "
                "joins its incident",
            ],
        }
    )
