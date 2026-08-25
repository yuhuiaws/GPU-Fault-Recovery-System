from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class SpooledTelemetry:
    """One claimed telemetry sample, on its way to be replayed."""

    spool_key: str
    revision: int
    cluster_id: str | None
    path: str
    request_id: str
    attempts: int
    created_at: datetime
    payload: dict
    payload_bytes: int
