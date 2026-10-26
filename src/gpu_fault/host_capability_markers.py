"""Capability (NOT_SUPPORTED) markers of the host collector, and how they are seen.

A host contributor that cannot provide a reading because the node lacks the
capability posts a marker sample instead of a reading or an error: no
``ipmitool`` or no IPMI device node (``bmc_status_unsupported``), no
``smartctl``, no drive or a drive without a SMART status
(``smart_status_unsupported``), an EFA netdev without an ``ethtool -S`` counter
group (``efa_ethtool_counters_unsupported``). Every marker carries the
``reason`` label, the device the gap belongs to (``device``; ``None`` is the
node) and, for the EFA marker, the metric names it stands in for. It is not a
rule input and not a collection error: NOT_SUPPORTED never mints a finding,
never isolates and never holds a host validation -- the gap is the node's
state, usually for good.

A state that ships with every tick is no edge of its own, so without the two
rules below a marker was unobservable: the edge filter knew only the EFA
marker (GF-GENERAL-LIVE-010 ``missing_counter``, 2026-10-25), a BMC or SMART
marker that appeared, changed or cleared had no delivery reason, the health
summary that finally carried it is not kept as raw evidence
(``app/ingest/telemetry.py``: no finding, no collection error) and no latest
record held it. Live GF-GENERAL-LIVE-015 ``command_failure`` and
``unsupported_bmc`` (2026-10-26, plain EKS): the ``bmc_status_unsupported``
marker that followed the replayed BMC failure, and the steady TOOL_MISSING
marker of a node without ``ipmitool``, were in no record an operator or an
oracle could read.

* **Edge** (``collectors/host/collector.py`` ``_edge_reasons``): the marker
  contributes :func:`not_supported_edge_reason`, its whole content --
  ``not-supported:<marker>:<device or node>:<reason>`` plus every other
  label's value (the EFA marker's ``metrics``) -- so a marker that appears, a
  ``reason`` that changes (``TOOL_MISSING`` -> ``NO_DEVICE`` when ``ipmitool``
  arrives on a host without a BMC, ``MISSING_COUNTER`` <-> ``NO_STATISTICS``)
  and a counter list that changes are each a new reason, delivered once; the
  old reason ending, including the marker clearing, is a ``recovered`` edge;
  an unchanged marker is not delivered. On the collector's first tick the
  reason is stripped like the low-utilisation ones -- a capability the node
  never had is a state, the baseline batch carries the marker -- but it still
  becomes active, so only a later change is the edge.
* **Latest record** (``host_health.NodeHealthPolicy.LATEST_METRICS_REQUIRED``
  includes :data:`CAPABILITY_MARKERS`): every delivered batch, the routine
  health summary included, upserts the marker into ``telemetry_metric_latest``
  with its ``reason`` label and device, so the node's capability state can be
  read at any time without a raw batch. The record is latest-wins: a marker
  that cleared keeps its last record at the time it was last seen, which is
  how a reader tells a current gap (observed with the node's newest batch)
  from one that closed.

A new marker family registers its name in :data:`CAPABILITY_MARKERS`; the host
collector's emitters use these constants so the registry cannot drift from
what is posted (``tests/collectors/test_host_capability_marker_edges.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpu_fault.host_health import HostMetricSample

#: ``bmc_status_unsupported``: no ``ipmitool`` (``TOOL_MISSING``) or an
#: ``ipmitool`` that finds no IPMI device node while none exists (``NO_DEVICE``).
BMC_STATUS_UNSUPPORTED = "bmc_status_unsupported"
#: ``smart_status_unsupported``: no ``smartctl`` (``TOOL_MISSING``), a scan
#: with no drive (``NO_DEVICES``), or a drive that answers without a
#: PASSED/FAILED verdict (``NO_SMART_STATUS``, per drive).
SMART_STATUS_UNSUPPORTED = "smart_status_unsupported"
#: ``efa_ethtool_counters_unsupported``: an EFA netdev without a counter group
#: (``MISSING_COUNTER``) or without statistics at all (``NO_STATISTICS``);
#: ``labels.metrics`` names the deltas it stands in for.
EFA_ETHTOOL_UNSUPPORTED = "efa_ethtool_counters_unsupported"
#: The marker label that says why the capability is absent.
CAPABILITY_REASON_LABEL = "reason"
#: Every capability marker the host collector posts.
CAPABILITY_MARKERS: frozenset[str] = frozenset(
    {BMC_STATUS_UNSUPPORTED, SMART_STATUS_UNSUPPORTED, EFA_ETHTOOL_UNSUPPORTED}
)
#: The prefix of the edge reason a marker contributes; ``collect_once`` strips
#: reasons with this prefix from the first (baseline) delivery.
NOT_SUPPORTED_REASON_PREFIX = "not-supported:"


def is_capability_marker(sample: HostMetricSample) -> bool:
    """Whether ``sample`` is a NOT_SUPPORTED marker rather than a reading."""

    return sample.name in CAPABILITY_MARKERS


def not_supported_edge_reason(sample: HostMetricSample) -> str | None:
    """The edge reason of a NOT_SUPPORTED marker, ``None`` for any other sample.

    ``not-supported:<marker>:<device or node>:<reason>``, followed by the
    value of every other label in key order (the EFA marker's ``metrics``,
    unchanged from the reason the EFA-only rule produced). The whole marker is
    the key: any change of what is unsupported, where, or why is a new reason.
    """

    if not is_capability_marker(sample):
        return None
    labels = sample.labels
    identity = [
        sample.name,
        sample.device or "node",
        labels.get(CAPABILITY_REASON_LABEL, ""),
        *(
            value
            for key, value in sorted(labels.items())
            if key != CAPABILITY_REASON_LABEL
        ),
    ]
    return NOT_SUPPORTED_REASON_PREFIX + ":".join(identity)
