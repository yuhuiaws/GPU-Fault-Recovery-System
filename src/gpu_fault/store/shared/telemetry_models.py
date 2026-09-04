from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

# Keys for the telemetry tables. They are shared because all three backends
# slice them positionally -- the batch lock in the sqlite and postgres mixins
# takes `key[:2]` to mean "this node" -- so a backend that changed the order
# would keep type-checking while locking the wrong row.
GpuMetricsBatchKey = tuple[str, str, str]
"""``(cluster_id, node_id, batch_id)``."""

GpuMetricKey = tuple[str, str, str, str]
"""``(cluster_id, node_id, gpu_key, canonical_name)``.

``gpu_key`` is the GPU UUID, PCI address or index, falling back to ``"node"``
for a sample that belongs to the host rather than one device.
"""

GpuFindingKey = GpuMetricKey
"""A finding is keyed by the metric it was derived from.

Composite findings reuse the shape with ``canonical_name`` set to
``composite:<rule_id>``, which is why this is an alias and not a third tuple.
"""


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
