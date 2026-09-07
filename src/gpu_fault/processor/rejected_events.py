"""Book a handler's 4xx verdict on a replayed collector event where it shows.

Every collector channel is ``receipt=True``: the ingress answers 202 after
JSON decoding only, and the handler's real answer arrives when the processor
replays the row. A Pydantic 422 on that replay used to complete as
``outcome="success"`` at INFO with no per-status counter, so a payload-shape
drift on the kernel XID channel looked like a healthy pipeline whose nodes had
gone quiet. This module gives the completion three places to show up:

* a per-``(path, status class)`` counter on the coordinator's metrics snapshot;
* for channels in the fault layer, a WARNING naming the request, path, status
  and the handler's *detail* (never the payload body);
* a per-node ``CollectorStatus`` whose ``errors`` carry a recognisable prefix,
  so the silent-collector logic can tell "no data" from "data rejected".
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from gpu_fault.channel_registry import (
    COLLECTOR_HEALTH_PATH,
    FABRIC_MANAGER_PATH,
    GPU_INVENTORY_PATH,
    GPU_METRICS_PATH,
    HOST_TELEMETRY_PATH,
    NODE_LOG_PATH,
    NVIDIA_KERNEL_PATH,
    ChannelPriorityMode,
    channel_for_path,
)
from gpu_fault.processor.models import ProcessorRequest
from gpu_fault.telemetry import CollectorKind, CollectorStatus

LOGGER = logging.getLogger(__name__)

#: Prefix of the ``CollectorStatus.errors`` entry a rejected event records.
REJECTED_EVENT_ERROR_PREFIX = "rejected-event:"
#: How much of the handler's detail is kept in the log and the status.
DETAIL_LIMIT = 200

COLLECTOR_KIND_BY_PATH: dict[str, CollectorKind] = {
    NVIDIA_KERNEL_PATH: CollectorKind.NVIDIA_KERNEL,
    FABRIC_MANAGER_PATH: CollectorKind.FABRIC_MANAGER_LOG,
    GPU_INVENTORY_PATH: CollectorKind.GPU_INVENTORY,
    GPU_METRICS_PATH: CollectorKind.GPU_METRICS,
    HOST_TELEMETRY_PATH: CollectorKind.HOST_TELEMETRY,
    NODE_LOG_PATH: CollectorKind.NODE_LOGS,
}


def status_class(status: int) -> str:
    return f"{status // 100}xx"


def is_rejected_event_status(status: CollectorStatus) -> bool:
    """Whether a collector status records a rejected event, not missing data."""

    return any(error.startswith(REJECTED_EVENT_ERROR_PREFIX) for error in status.errors)


def is_fault_layer_request(item: ProcessorRequest) -> bool:
    """Whether a 4xx on this request means a fault report was thrown away.

    The fault layer is every FAULT-mode channel (kernel XID, Fabric Manager
    SXID) plus any incident-scoped channel whose payload is not routine: an
    edge-filtered host or node-log batch that carries a breach is a
    node-health finding in flight, a health summary is not.
    """

    channel = channel_for_path(item.path)
    if channel is None:
        return False
    if channel.priority_mode is ChannelPriorityMode.FAULT:
        return True
    if not channel.incident_scoped or item.path == COLLECTOR_HEALTH_PATH:
        return False
    return channel.priority(_json_payload(item)) < 100


def response_detail(body: bytes) -> str:
    """The handler's own explanation, bounded, without the rejected payload.

    A Pydantic 422 echoes the offending ``input`` back inside each detail
    item; only ``loc``/``msg``/``type`` are kept so the log never carries a
    payload body.
    """

    try:
        parsed = json.loads(body) if body else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        parsed = None
    if isinstance(parsed, dict):
        detail = parsed.get("detail", parsed)
    else:
        detail = parsed
    if isinstance(detail, list):
        parts = []
        for item in detail:
            if isinstance(item, dict):
                loc = ".".join(str(part) for part in item.get("loc", []) or [])
                parts.append(
                    " ".join(
                        piece
                        for piece in (
                            loc,
                            str(item.get("msg", "")),
                            str(item.get("type", "")),
                        )
                        if piece
                    )
                )
            else:
                parts.append(str(item))
        text = "; ".join(parts)
    elif detail is None:
        text = ""
    else:
        text = str(detail)
    return text[:DETAIL_LIMIT]


def _json_payload(item: ProcessorRequest) -> dict[str, Any] | None:
    try:
        payload = json.loads(item.body())
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def record_replay_completion(
    coordinator: Any,
    item: ProcessorRequest,
    *,
    status: int,
    body: bytes,
) -> None:
    """Count a completed replay by status class and surface fault rejections."""

    fault_rejection = 400 <= status < 500 and is_fault_layer_request(item)
    with coordinator._state_lock:
        by_status = coordinator._completions_by_path_status.setdefault(item.path, {})
        by_status[status_class(status)] = by_status.get(status_class(status), 0) + 1
        if fault_rejection:
            coordinator._fault_rejections_total += 1
    if not fault_rejection:
        return
    detail = response_detail(body)
    LOGGER.warning(
        "processor replay rejected a fault-layer event; the collector saw a 202 "
        "request_id=%s path=%s status=%s detail=%s",
        item.request_id,
        item.path,
        status,
        detail,
    )
    try:
        record_rejected_event_status(
            coordinator.store, item, status=status, detail=detail
        )
    except Exception:
        LOGGER.exception(
            "cannot record the rejected-event collector status request_id=%s path=%s",
            item.request_id,
            item.path,
        )


def record_rejected_event_status(
    store: Any,
    item: ProcessorRequest,
    *,
    status: int,
    detail: str,
) -> bool:
    """Persist a per-node "data rejected" status for a collector channel.

    Returns ``False`` when the request names no collector channel or node,
    in which case nothing is written.
    """

    collector = COLLECTOR_KIND_BY_PATH.get(item.path)
    payload = _json_payload(item) or {}
    node_id = payload.get("node_id")
    cluster_id = item.cluster_id or payload.get("cluster_id")
    if (
        collector is None
        or not isinstance(node_id, str)
        or not node_id
        or not isinstance(cluster_id, str)
        or not cluster_id
    ):
        return False
    now = datetime.now(timezone.utc)
    return bool(
        store.save_collector_status(
            CollectorStatus(
                cluster_id=cluster_id,
                node_id=node_id,
                collector=collector,
                observed_at=now,
                ingested_at=now,
                last_error_at=now,
                batch_id=item.request_id,
                sample_count=0,
                errors=[
                    f"{REJECTED_EVENT_ERROR_PREFIX} HTTP {status} {detail}".rstrip()
                ],
            )
        )
    )
