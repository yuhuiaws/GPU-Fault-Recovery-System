from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit


def kmsg_identity_errors(
    records: list[dict[str, Any]],
    *,
    boot_id: str,
    node: str | None = None,
    cluster_id: str | None = None,
) -> list[str]:
    errors = []
    for record in records:
        payload = record.get("payload")
        if not isinstance(payload, dict):
            errors.append("kernel evidence has no stored payload")
            continue
        record_id = str(payload.get("record_id") or "")
        prefix = f"kmsg-{boot_id}-"
        sequence = record_id.removeprefix(prefix)
        if not boot_id or not record_id.startswith(prefix) or not sequence.isdecimal():
            errors.append(f"record {record_id} is not kmsg-{boot_id}-<sequence>")
        if record.get("record_id") != f"nvidia-kernel/{record_id}":
            errors.append("kernel evidence record ID does not bind its stored payload")
        expected_node = node or str(record.get("node_id") or "")
        expected_cluster = cluster_id or str(record.get("cluster_id") or "")
        if (
            not expected_node
            or not expected_cluster
            or record.get("cluster_id") != expected_cluster
            or payload.get("cluster_id") != expected_cluster
            or record.get("node_id") != expected_node
            or payload.get("node_id") != expected_node
            or payload.get("source_boot_id") != boot_id
        ):
            errors.append("kernel evidence node/boot identity differs from baseline")
        reference = str(payload.get("evidence_ref") or "")
        parsed = urlsplit(reference)
        if (
            parsed.scheme != "kmsg"
            or parsed.netloc != expected_node
            or parsed.path != f"/{boot_id}/{sequence}"
            or parsed.query
            or parsed.fragment
        ):
            errors.append("kernel evidence_ref does not bind node/boot/sequence")
    return errors


def kmsg_record_errors(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    *,
    boot_id: str,
    node: str | None = None,
    cluster_id: str | None = None,
) -> list[str]:
    errors = []
    first_ids = {str(item.get("record_id")) for item in first}
    second_ids = {str(item.get("record_id")) for item in second}
    if len(first_ids) != 1:
        errors.append(
            f"sample 1 did not produce exactly one record: {sorted(first_ids)}"
        )
    if len(second_ids - first_ids) != 1:
        errors.append("sample 2 did not add exactly one record")
    errors.extend(
        kmsg_identity_errors(
            first + second, boot_id=boot_id, node=node, cluster_id=cluster_id
        )
    )
    refs = {
        str((item.get("payload") or {}).get("evidence_ref") or "") for item in second
    }
    if len(refs) != len(second_ids) or any(not ref for ref in refs):
        errors.append("kmsg records do not carry distinct evidence_ref values")
    return errors


def kernel_event_identity_errors(
    state: dict[str, Any],
    *,
    node: str,
    cluster_id: str,
    boot_id: str,
    gpu_uuid: str,
    software: dict[str, Any],
    xid: int,
) -> list[str]:
    errors = kmsg_identity_errors(
        state.get("evidence") or [], boot_id=boot_id, node=node, cluster_id=cluster_id
    )
    events = state.get("events") or []
    if not events:
        errors.append("kernel evidence has no normalized XID event")
    references = {
        str((item.get("payload") or {}).get("evidence_ref") or "")
        for item in state.get("evidence") or []
    }
    for event in events:
        if (
            event.get("cluster_id") != cluster_id
            or event.get("node_id") != node
            or event.get("source_boot_id") != boot_id
            or event.get("gpu_uuid") != gpu_uuid
            or event.get("xid") != xid
            or event.get("evidence_ref") not in references
        ):
            errors.append(
                "normalized XID node/boot/GPU/source does not match injection"
            )
        if any(
            event.get(key) != software.get(key) or software.get(key) is None
            for key in ("product", "driver_branch", "cuda_version")
        ):
            errors.append("normalized XID product/driver/CUDA differs from the node")
    if {item.get("evidence_ref") for item in events} != references:
        errors.append("not every kernel record has a bound normalized event")
    decisions = state.get("decisions") or []
    event_ids = {item.get("event_id") for item in events}
    if (
        not event_ids
        or None in event_ids
        or {item.get("event_id") for item in decisions} != event_ids
    ):
        errors.append("XID decisions do not bind every normalized event")
    incident_ids = {item.get("incident_id") for item in state.get("incidents") or []}
    if any(
        not item.get("incident_id") or item["incident_id"] not in incident_ids
        for item in decisions
    ):
        errors.append("XID decisions do not bind the recorded incidents")
    return errors
