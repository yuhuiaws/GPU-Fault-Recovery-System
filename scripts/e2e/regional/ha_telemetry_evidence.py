"""HA telemetry admission, replay and retry evidence for routine summaries."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

HOST_PATH = "/v1/collector-events/host-telemetry"
RETRYABLE_HTTP = frozenset({408, 425, 429, 502, 503, 504})


def wire_errors(probe: dict[str, Any]) -> list[str]:
    rows = probe.get("wire_responses")
    if not isinstance(rows, list) or not rows:
        return ["probe captured no HTTP response evidence"]
    errors = []
    for row in rows:
        if not isinstance(row, dict):
            errors.append("HTTP response evidence is malformed")
            continue
        status = row.get("status")
        success = {200, 202} if row.get("path") == HOST_PATH else {200}
        if status is None:
            if row.get("transport_error") not in {
                "URLError",
                "OSError",
                "TimeoutError",
                "ConnectionError",
                "ConnectionResetError",
                "ConnectionRefusedError",
                "RemoteDisconnected",
                "gaierror",
                "SSLError",
            }:
                errors.append("HTTP failure is not a recognized transport failure")
            continue
        if type(status) is not int or status not in success | RETRYABLE_HTTP:
            errors.append(f"probe observed nonretryable HTTP status {status}")
        if status in {429, 503}:
            try:
                retry_after = float(row.get("retry_after"))
            except (TypeError, ValueError):
                retry_after = float("nan")
            if not math.isfinite(retry_after) or not 0 < retry_after <= 900:
                errors.append(f"HTTP {status} has no bounded Retry-After")
    return errors


def admission_errors(probe: dict[str, Any], accepted_ids: list[str]) -> list[str]:
    events = probe.get("attempted_events")
    admissions = probe.get("admissions")
    if not isinstance(events, list) or not events or not isinstance(admissions, list):
        return ["probe captured no structured telemetry admission evidence"]
    if any(not isinstance(item, dict) for item in [*events, *admissions]):
        return ["telemetry admission evidence is malformed"]
    ids = [item.get("batch_id") for item in events]
    node = f"ha005-node-{probe.get('run_id')}"
    errors = []
    if (
        any(not isinstance(item, str) or not item for item in ids)
        or len(ids) != len(set(ids))
        or ids != probe.get("attempted_batch_ids")
        or len(ids) != (probe.get("counters") or {}).get("event_attempts")
        or probe.get("payload_contract")
        != {
            "path": HOST_PATH,
            "node_id": node,
            "edge_filter_reasons": ["health-summary"],
            "collection_errors": [],
            "sample_count": 9,
        }
    ):
        errors.append("routine summary identity or coalescing contract differs")
    direct = set()
    foreground = set()
    for item in admissions:
        if (
            item.get("batch_id") not in ids
            or type(item.get("replay")) is not bool
            or item.get("accepted") is not True
        ):
            errors.append("telemetry admission is not bound to an attempted batch")
        if item.get("replay") is False:
            foreground.add(item.get("batch_id"))
        if item.get("spooled") is True:
            if (
                type(item.get("coalesced")) is not bool
                or item.get("request_id") is not None
            ):
                errors.append("spool admission has an invalid receipt shape")
        elif item.get("spooled") is False:
            request_id = item.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                errors.append("queue admission has no processor request ID")
            else:
                direct.add(request_id)
        else:
            errors.append("admission does not name queue or spool")
    if len(foreground) != (probe.get("counters") or {}).get("event_accepted"):
        errors.append("foreground admission evidence does not match successful sends")
    if set(accepted_ids) != direct or len(accepted_ids) != len(direct):
        errors.append("queue receipt IDs do not cover every observed queue admission")
    if not admissions or events[-1].get("batch_id") not in {
        item.get("batch_id") for item in admissions
    }:
        errors.append("the final routine summary has no admission proof")
    if probe.get("replay_stopped") is not True:
        errors.append("collector outbox replay has not stopped")
    return errors


def telemetry_replay_errors(
    probe: dict[str, Any], receipt: dict[str, Any]
) -> list[str]:
    events = probe.get("attempted_events") or []
    if not events or not isinstance(events[-1], dict):
        return ["probe has no final telemetry batch"]
    final = events[-1]
    errors = []
    if (
        receipt.get("cluster_id") != probe.get("cluster_id")
        or receipt.get("node_id") != f"ha005-node-{probe.get('run_id')}"
        or receipt.get("batch_id") != final.get("batch_id")
        or receipt.get("collector") != "HOST_TELEMETRY"
        or receipt.get("sample_count") != 9
        or receipt.get("errors") != []
        or receipt.get("spool_depth") != 0
        or type(receipt.get("spool_depth")) is not int
    ):
        errors.append("telemetry replay did not reach the bound final healthy summary")
    try:
        observed = datetime.fromisoformat(str(final["observed_at"]))
        ingested = datetime.fromisoformat(str(receipt["observed_at"]))
        success = datetime.fromisoformat(str(receipt["last_success_at"]))
        if (
            observed.tzinfo is None
            or ingested.tzinfo is None
            or success.tzinfo is None
            or ingested != observed
            or success != observed
        ):
            errors.append("telemetry replay time does not match the final batch")
    except (ValueError, KeyError):
        errors.append("telemetry replay has no valid observation time")
    return errors
