"""Production FM source-delivery proof, independent of Store deduplication."""

from __future__ import annotations

import hashlib
from typing import Any

from scripts.e2e.regional.probes.collector_node_probe import (
    FM_RECEIPT_COUNTERS,
    HEX32,
    ProbeError,
    checked_fm_receipt,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError


def identity_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def completed_round_end(projection: dict[str, Any], invocation: str) -> int | None:
    """The validated progress boundary, not a later read with no new messages."""

    return max(
        (
            row["monotonic_us"]
            for row in projection.get("records") or []
            if (row.get("receipt") or {}).get("systemd_invocation_id") == invocation
            and row["receipt"].get("stage") == "ROUND"
            and row["receipt"].get("outcome") == "COMPLETE"
        ),
        default=None,
    )


def checked_projection(
    value: dict[str, Any], *, cluster_id: str, node: str, boot_id: str
) -> list[dict[str, Any]]:
    service = value.get("service") or {}
    records = value.get("records")
    captured = value.get("captured_monotonic_us")
    if (
        value.get("complete") is not True
        or service.get("ActiveState") != "active"
        or not HEX32.fullmatch(str(service.get("InvocationID") or ""))
        or not str(service.get("MainPID") or "").isdecimal()
        or int(service["MainPID"]) <= 1
        or type(captured) is not int
        or captured <= 0
        or not isinstance(records, list)
        or not records
    ):
        raise RegionalFixtureError(
            "FM receipt projection has no complete active producer"
        )
    previous_time = -1
    cursors = set()
    for record in records:
        try:
            receipt = checked_fm_receipt(record["receipt"])
        except (KeyError, TypeError, ProbeError) as exc:
            raise RegionalFixtureError("FM receipt projection is malformed") from exc
        clock = record.get("monotonic_us")
        cursor = record.get("cursor")
        if (
            not isinstance(cursor, str)
            or not cursor
            or cursor in cursors
            or type(clock) is not int
            or not previous_time <= clock <= captured
        ):
            raise RegionalFixtureError("FM receipt journal order/clock is unproven")
        previous_time = clock
        cursors.add(cursor)
        for key, expected in (
            ("cluster_id_sha256", cluster_id),
            ("node_id_sha256", node),
            ("boot_id_sha256", boot_id),
        ):
            if not expected or receipt[key] != identity_digest(expected):
                raise RegionalFixtureError(
                    "FM receipt belongs to another node/cluster/boot"
                )
    return records


def original_anchor(
    projection: dict[str, Any], *, cluster_id: str, node: str, boot_id: str
) -> dict[str, Any]:
    records = checked_projection(
        projection, cluster_id=cluster_id, node=node, boot_id=boot_id
    )
    service = projection["service"]
    for record in reversed(records):
        receipt = record["receipt"]
        if (
            receipt["stage"] == "ROUND"
            and receipt["outcome"] == "COMPLETE"
            and receipt["systemd_invocation_id"] == service["InvocationID"]
            and str(receipt["pid"]) == service["MainPID"]
            and receipt["attempt_seq"] == receipt["attempts_total"]
            and receipt["rounds_completed_total"] + receipt["rounds_failed_total"]
            == receipt["round_seq"]
            and receipt["attempts_total"]
            == sum(
                receipt[key]
                for key in ("delivered_total", "buffered_total", "failed_total")
            )
        ):
            return record
    raise RegionalFixtureError(
        "FM producer has no completed-round anchor before injection"
    )


def delivery_window_errors(
    anchor: dict[str, Any],
    projection: dict[str, Any],
    *,
    record_id: str,
    cluster_id: str,
    node: str,
    boot_id: str,
    restarted_invocation: str | None = None,
) -> list[str]:
    try:
        records = checked_projection(
            projection, cluster_id=cluster_id, node=node, boot_id=boot_id
        )
    except RegionalFixtureError as exc:
        return [str(exc)]
    if projection.get("anchor_cursor") != anchor["cursor"] or records[0] != anchor:
        return ["FM receipt window did not preserve its exact journal anchor"]
    old = anchor["receipt"]
    service = projection["service"]
    expected_invocation = restarted_invocation or old["systemd_invocation_id"]
    if service["InvocationID"] != expected_invocation or (
        restarted_invocation and restarted_invocation == old["systemd_invocation_id"]
    ):
        return ["FM service restart identity is unproven"]
    digest = identity_digest(record_id)
    errors = []
    previous_by_producer = {old["producer_invocation_id"]: old}
    active_producer = old["producer_invocation_id"]
    pending: dict[str, dict[str, Any]] = {}
    target_delivered = 0
    target_attempts = 0
    target_round = False
    new_rounds = 0
    new_producer: str | None = None
    for record in records[1:]:
        row = record["receipt"]
        if row["source_config_sha256"] != old["source_config_sha256"]:
            errors.append("FM collector source configuration changed within the proof")
        producer = row["producer_invocation_id"]
        if producer not in previous_by_producer:
            if (
                not restarted_invocation
                or new_producer is not None
                or row["systemd_invocation_id"] != restarted_invocation
                or str(row["pid"]) != service["MainPID"]
                or row["receipt_seq"] != 1
                or row["round_seq"] != 1
            ):
                errors.append(
                    "FM receipt producer changed without a complete restart boundary"
                )
            new_producer = producer
            previous = {key: 0 for key in FM_RECEIPT_COUNTERS}
        else:
            previous = previous_by_producer[producer]
            if producer != active_producer:
                errors.append("FM receipt returned to an old producer invocation")
        active_producer = producer
        previous_by_producer[producer] = row
        if (
            row["receipt_seq"] != previous["receipt_seq"] + 1
            or row["omitted_receipts_total"] != previous["omitted_receipts_total"]
            or any(row[key] < previous[key] for key in FM_RECEIPT_COUNTERS)
        ):
            errors.append("FM receipt sequence gap, omission or counter regression")
        if (
            row["attempt_seq"] != row["attempts_total"]
            or row["outcome"] == "FAILED"
            or row["failed_total"] != previous["failed_total"]
            or row["rounds_failed_total"] != previous["rounds_failed_total"]
        ):
            errors.append("FM producer failed or its attempt counters disagree")
        is_new = row["systemd_invocation_id"] == restarted_invocation
        if producer == old["producer_invocation_id"] and (
            row["systemd_invocation_id"] != old["systemd_invocation_id"]
            or row["pid"] != old["pid"]
        ):
            errors.append(
                "FM receipt changed PID/systemd identity within an invocation"
            )
        if producer != old["producer_invocation_id"] and (
            row["systemd_invocation_id"] != restarted_invocation
            or str(row["pid"]) != service["MainPID"]
        ):
            errors.append("FM receipt does not belong to the restarted process")
        stage = row["stage"]
        if stage == "ATTEMPT":
            if (
                producer in pending
                or row["attempts_total"] != previous["attempts_total"] + 1
            ):
                errors.append(
                    "FM attempt receipt is not the next single source delivery"
                )
            pending[producer] = row
            if (
                any(
                    row[key] != previous[key]
                    for key in ("delivered_total", "buffered_total", "failed_total")
                )
                or row["rounds_completed_total"] + row["rounds_failed_total"]
                != row["round_seq"] - 1
            ):
                errors.append("FM attempt completion/round counters are inconsistent")
            if row["record_id_sha256"] == digest and not is_new:
                target_attempts += 1
                if row["source"] != "file" or target_attempts > 1:
                    errors.append(
                        "FM original file record was submitted more than once or via another source"
                    )
            if is_new and row["record_id_sha256"] == digest:
                errors.append("FM collector replayed the original record after restart")
        elif stage == "COMPLETION":
            attempt = pending.pop(producer, None)
            if attempt is None or any(
                attempt[key] != row[key]
                for key in ("attempt_seq", "record_id_sha256", "source", "round_seq")
            ):
                errors.append("FM completion has no exactly paired producer attempt")
            outcome_counter = {
                "DELIVERED": "delivered_total",
                "BUFFERED": "buffered_total",
                "FAILED": "failed_total",
            }[row["outcome"]]
            if row["attempts_total"] != previous["attempts_total"] or any(
                row[key] != previous[key] + int(key == outcome_counter)
                for key in ("delivered_total", "buffered_total", "failed_total")
            ):
                errors.append("FM completion outcome does not match its counters")
            if row["record_id_sha256"] == digest and not is_new:
                if row["outcome"] == "DELIVERED":
                    target_delivered += 1
                else:
                    errors.append(
                        "FM original record was not acknowledged as delivered"
                    )
        else:
            if producer in pending or row["attempts_total"] != sum(
                row[key]
                for key in ("delivered_total", "buffered_total", "failed_total")
            ):
                errors.append("FM completed round has an unfinished delivery")
            if (
                row["rounds_completed_total"] + row["rounds_failed_total"]
                != row["round_seq"]
            ):
                errors.append("FM completed round counters disagree with progress")
            if row["rounds_completed_total"] <= previous[
                "rounds_completed_total"
            ] or any(
                row[key] != previous[key]
                for key in (
                    "attempts_total",
                    "delivered_total",
                    "buffered_total",
                    "failed_total",
                )
            ):
                errors.append(
                    "FM completed round has no continuing zero-hidden-delivery progress"
                )
            if not is_new and target_delivered == 1:
                target_round = True
            if is_new:
                if row["attempts_total"] != 0:
                    errors.append("FM post-restart zero-delivery progress is unproven")
                new_rounds += 1
    if pending:
        errors.append("FM receipt window ends with an unmatched attempt")
    if target_attempts != 1 or target_delivered != 1 or not target_round:
        errors.append(
            "FM original record lacks exactly one delivery and completed round"
        )
    if restarted_invocation and (new_producer is None or new_rounds < 2):
        errors.append("FM restart lacks two completed producer rounds")
    return list(dict.fromkeys(errors))
