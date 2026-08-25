from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone


def stable_phase_seconds(
    cluster_id: str,
    node_id: str,
    channel: str,
    interval_seconds: int,
) -> int:
    """Return a restart-stable wall-clock phase for periodic delivery."""

    if interval_seconds <= 0:
        raise ValueError("phase interval must be positive")
    value = hashlib.blake2b(
        f"{cluster_id}/{node_id}/{channel}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(value, "big") % interval_seconds


def next_stable_phase(
    observed_at: datetime,
    *,
    cluster_id: str,
    node_id: str,
    channel: str,
    interval_seconds: int,
) -> datetime:
    phase = stable_phase_seconds(cluster_id, node_id, channel, interval_seconds)
    epoch = observed_at.timestamp()
    cycle = math.floor(epoch / interval_seconds)
    candidate = cycle * interval_seconds + phase
    if candidate <= epoch:
        candidate += interval_seconds
    return datetime.fromtimestamp(candidate, tz=timezone.utc)
