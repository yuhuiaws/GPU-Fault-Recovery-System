"""COLLECT-014's fail-closed premise: no stored GPU inventory may vouch for the SXID.

The fail-closed direction posts a back-dated SXID 10003 and needs ingest to find
no inventory evidence for it. Ingest accepts a stored sample as evidence while
``-30s <= event - sample <= max age`` (``src/gpu_fault/app/ingest/faults.py``:
the snapshot's ``gpu_inventory_max_age``, the latest metrics'
``legacy_gpu_metrics_inventory_max_age``), so the premise holds when every
sample lies outside that window by a margin ingestion lag cannot eat.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

#: Ingest's tolerance for a sample a little newer than the event.
INGEST_FUTURE_TOLERANCE = timedelta(seconds=30)
#: The margin on either side of the window that ingestion lag cannot eat.
FAIL_CLOSED_SAFETY_MARGIN = timedelta(seconds=60)


def _max_age(inventory: dict[str, Any], key: str) -> timedelta | None:
    value = inventory.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return timedelta(seconds=float(value))


def fail_closed_timing_errors(
    inventory: dict[str, Any],
    *,
    event_time: datetime,
) -> list[str]:
    """Both dedicated and legacy inventory must be outside the event window.

    Ingest takes a sample as evidence while ``-30s <= event - sample <= max
    age`` (``app/ingest/faults.py``: the snapshot's ``gpu_inventory_max_age``,
    the latest metrics' ``legacy_gpu_metrics_inventory_max_age``). A sample is
    therefore safe on either side: newer than the event by the future
    tolerance plus a margin, or older than the max age plus that margin. The
    second side matters for latest-metric rows a node no longer refreshes --
    a derived ``fabric_manager_status_held_seconds`` after the fabric trained,
    an XID field host-mode dcgm-exporter leaves blank -- which used to refuse
    the case on a healthy node. Without the product's max age (an older probe)
    only the newer side counts.
    """

    if type(inventory.get("present")) is not bool:
        return ["dedicated GPU inventory presence is unknown; SXID is not posted"]
    legacy = inventory.get("legacy_observed_at")
    if not isinstance(legacy, list):
        return ["legacy GPU inventory was not checked; SXID is not posted"]
    samples = [
        *(
            [
                (
                    inventory.get("observed_at"),
                    _max_age(inventory, "snapshot_max_age_seconds"),
                )
            ]
            if inventory["present"]
            else []
        ),
        *((item, _max_age(inventory, "legacy_max_age_seconds")) for item in legacy),
    ]
    needed = INGEST_FUTURE_TOLERANCE + FAIL_CLOSED_SAFETY_MARGIN
    for sample, max_age in samples:
        try:
            observed_at = datetime.fromisoformat(str(sample).replace("Z", "+00:00"))
            if observed_at.tzinfo is None:
                raise ValueError("inventory timestamp has no timezone")
            margin = observed_at - event_time
        except (TypeError, ValueError):
            return ["stored GPU inventory time is unknown; SXID is not posted"]
        if margin >= needed:
            continue
        if max_age is not None and -margin >= max_age + FAIL_CLOSED_SAFETY_MARGIN:
            continue
        older = (
            f" or {(max_age + FAIL_CLOSED_SAFETY_MARGIN).total_seconds():.0f}s older"
            if max_age is not None
            else ""
        )
        return [
            f"stored GPU inventory ({observed_at.isoformat()}) is "
            f"{margin.total_seconds():.0f}s from the back-dated event; it must be "
            f"{needed.total_seconds():.0f}s newer{older} for the fail-closed "
            "premise to hold, so the SXID is not posted"
        ]
    return []
