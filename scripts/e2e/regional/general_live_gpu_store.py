"""Read-only GPU telemetry reads of the GF-GENERAL-LIVE DCGM categories.

``general_live_store.node_activity`` already returns a node's incidents,
workflows, markers, remote commands and notifications; this module adds the
GPU side the DCGM / inventory variants judge on, read through the installed
control-plane interpreter of an API Pod (``RegionalLiveFixture.cpu_python``):

* the latest accepted GPU metric per ``(gpu, canonical)`` (the product's
  ``gpu_metric_latest``), and the accepted GPU_METRICS batches since a moment
  with their edge-filter reasons, collection errors and the samples of the
  fields a variant names;
* the active GPU finding states and the finding history since the moment;
* the XID events since the moment with their policy decisions;
* the latest GPU inventory snapshot and the GPU_INVENTORY evidence since the
  moment;
* the GPU collector statuses, and the capability modes of the node's Runtime
  Profile (``gpuReset``/``fabricReset``/``memoryDiagnostics``).

Nothing here writes to the store and nothing posts a finding or workflow.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any, Protocol


class CpuPython(Protocol):
    def cpu_python(
        self, script: str, *arguments: str, **options: Any
    ) -> dict[str, Any]: ...


GPU_STATE_PROBE = r"""
import json
import sys
from datetime import datetime

from gpu_fault.app import ApplicationContext
from gpu_fault.store import NotFoundError
from gpu_fault.telemetry import EvidenceKind

cluster_id, node_id, since_text, names_text = sys.argv[1:5]
since = datetime.fromisoformat(since_text.replace("Z", "+00:00"))
names = {item for item in names_text.split(",") if item}
store = ApplicationContext.from_environment().store
LABELS = ("source_boot_id", "boot_id", "modelName", "limit_source")


def wanted(name):
    return not names or name in names


def finding_view(item):
    return {
        key: value
        for key, value in item.model_dump(mode="json").items()
        if key in {
            "finding_id", "canonical_name", "severity", "value", "delta",
            "rate_per_minute", "gpu_uuid", "pci_bdf", "observed_at",
            "automatic_action", "official_action", "policy_source",
            "finding_kind", "correlation_rule_id", "component_finding_ids",
            "component_metrics", "affected_gpu_uuids", "confidence",
            "threshold_value", "reason", "threshold_source", "confirmation_count",
            "workload_state", "affected_workload_ids",
        }
    }


latest = [
    {
        "gpu": item.sample.gpu_uuid or item.sample.pci_bdf or item.sample.gpu_index,
        "name": item.sample.canonical_name,
        "value": item.sample.value,
        "observed_at": item.observed_at.isoformat(),
        "source": str(item.source),
        "labels": {key: item.sample.labels.get(key) for key in LABELS if item.sample.labels.get(key)},
    }
    for item in store.list_gpu_metrics_latest(cluster_id, node_id)
    if wanted(item.sample.canonical_name)
]
active = [
    finding_view(item)
    for item in store.list_gpu_findings(cluster_id, node_id, active_only=True)
]
history = [
    finding_view(item)
    for item in store.list_gpu_findings(cluster_id, node_id, active_only=False)
    if item.observed_at >= since
]
xids = []
for event in store.list_xid_events(cluster_id, node_id, observed_after=since):
    decision = store.get_xid_policy_decision(event.event_id)
    xids.append({
        "event_id": event.event_id,
        "xid": event.xid,
        "event_source": event.event_source,
        "gpu_uuid": event.gpu_uuid,
        "observed_at": event.observed_at.isoformat(),
        "disposition": str(decision.disposition) if decision is not None else None,
        "action": str(decision.action) if decision is not None and getattr(decision, "action", None) is not None else None,
    })
batches = []
for item in store.list_raw_evidence(
    cluster_id, node_id=node_id, kind=EvidenceKind.GPU_METRICS, limit=2000
):
    if item.observed_at < since:
        continue
    payload = item.payload if isinstance(item.payload, dict) else {}
    batches.append({
        "batch_id": payload.get("batch_id"),
        "observed_at": item.observed_at.isoformat(),
        "edge_filter_reasons": payload.get("edge_filter_reasons") or [],
        "collection_errors": payload.get("collection_errors") or [],
        "sample_count": len(payload.get("samples") or []),
        "gpus": sorted({
            sample.get("gpu_uuid") for sample in payload.get("samples") or []
            if isinstance(sample, dict) and sample.get("gpu_uuid")
        }),
        "samples": [
            {
                "gpu": sample.get("gpu_uuid") or sample.get("pci_bdf"),
                "name": sample.get("canonical_name"),
                "value": sample.get("value"),
                "labels": {key: (sample.get("labels") or {}).get(key) for key in LABELS if (sample.get("labels") or {}).get(key)},
            }
            for sample in payload.get("samples") or []
            if isinstance(sample, dict) and names and sample.get("canonical_name") in names
        ],
    })
batches.sort(key=lambda item: item["observed_at"])
inventory_records = []
for item in store.list_raw_evidence(
    cluster_id, node_id=node_id, kind=EvidenceKind.GPU_INVENTORY, limit=500
):
    if item.observed_at < since:
        continue
    payload = item.payload if isinstance(item.payload, dict) else {}
    inventory_records.append({
        "record_id": item.record_id,
        "observed_at": item.observed_at.isoformat(),
        "source_boot_id": payload.get("source_boot_id"),
        "expected_gpu_count": payload.get("expected_gpu_count"),
        "devices": payload.get("devices") or [],
    })
inventory_records.sort(key=lambda item: item["observed_at"])
snapshot = store.get_gpu_inventory_snapshot(cluster_id, node_id)
try:
    agent = store.get_agent(cluster_id, node_id)
except NotFoundError:
    agent = None
capabilities = {}
if agent is not None:
    try:
        profile = store.get_profile(agent.runtime_profile_version)
    except NotFoundError:
        profile = None
    for item in getattr(profile, "capabilities", None) or []:
        capabilities[str(getattr(item.capability, "value", item.capability))] = {
            "mode": str(getattr(item.mode, "value", item.mode)),
            "owner": getattr(item, "owner", None),
        }
print(json.dumps({
    "since": since.isoformat(),
    "latest": latest,
    "active_findings": active,
    "findings": history,
    "xid_events": xids,
    "batches": batches[-300:],
    "inventory_records": inventory_records[-100:],
    "inventory_snapshot": snapshot.model_dump(mode="json") if snapshot is not None else None,
    "collector_statuses": [
        item.model_dump(mode="json")
        for item in store.list_collector_statuses(cluster_id, node_id)
        if str(item.collector) in {"GPU_METRICS", "GPU_INVENTORY"}
    ],
    "agent": {
        "runtime_profile_version": agent.runtime_profile_version,
        "allowed_operations": [str(getattr(item, "value", item)) for item in agent.allowed_operations],
        "boot_id": agent.boot_id,
        "lifecycle_state": str(getattr(agent.lifecycle_state, "value", agent.lifecycle_state)),
    } if agent is not None else None,
    "capabilities": capabilities,
}, sort_keys=True, default=str))
"""


def gpu_state(
    regional: CpuPython,
    cluster_id: str,
    node: str,
    since: datetime,
    *,
    names: Iterable[str] = (),
) -> dict[str, Any]:
    """The node's GPU telemetry records since ``since`` (read-only)."""

    wanted = sorted(set(names))
    for value in wanted:
        if not value or "," in value:
            raise ValueError("GPU state filters must be non-empty, no comma")
    return regional.cpu_python(
        GPU_STATE_PROBE,
        cluster_id,
        node,
        since.isoformat(),
        ",".join(wanted),
        timeout=240,
    )


__all__ = ["GPU_STATE_PROBE", "gpu_state"]
