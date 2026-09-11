"""What two kmsg writes of one identical XID line must leave in the store.

Shared by COLLECT-012 (``run_collector_acceptance``): the verdict that two
writes became two records, each with its own evidence reference.
"""

from __future__ import annotations

from typing import Any


def kmsg_record_errors(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    *,
    boot_id: str,
) -> list[str]:
    """COLLECT-012 samples 1 and 2: same text, two kmsg sequences, two records.

    The kernel collector names a record ``kmsg-<boot_id>-<sequence>``; the
    ingest route stores it as ``nvidia-kernel/kmsg-...`` with the collector's
    ``evidence_ref`` (``kmsg://<node>/<boot_id>/<sequence>``) in the payload.
    Two writes of one identical line share the prefix and differ only in the
    suffix, each with its own evidence_ref; anything else means the second
    write was deduplicated or attributed to another boot.
    """

    errors = []
    prefix = f"nvidia-kernel/kmsg-{boot_id}-"
    first_ids = {str(item.get("record_id")) for item in first}
    second_ids = {str(item.get("record_id")) for item in second}
    new_ids = second_ids - first_ids
    if len(first_ids) != 1:
        errors.append(
            f"sample 1 did not produce exactly one record: {sorted(first_ids)}"
        )
    if len(new_ids) != 1:
        errors.append(
            f"sample 2 did not add exactly one record: {sorted(second_ids)} "
            f"after {sorted(first_ids)}"
        )
    for record_id in sorted(first_ids | second_ids):
        if not record_id.startswith(prefix):
            errors.append(f"record {record_id} is not {prefix}<sequence>")
    refs = {
        str(
            item.get("evidence_ref")
            or (item.get("payload") or {}).get("evidence_ref")
            or ""
        )
        for item in second
    }
    if len(refs) != len(second_ids) or any(not ref for ref in refs):
        errors.append(f"kmsg records do not carry distinct evidence_ref values: {refs}")
    return errors
