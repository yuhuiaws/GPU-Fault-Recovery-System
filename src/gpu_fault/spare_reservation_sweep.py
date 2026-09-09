"""Periodic reclaim of stale warm-spare reservations on the regional executor.

Split out of ``cluster_executor`` (which re-exports these names) so the
executor module stays within its size baseline.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Callable

from gpu_fault.hyperpod_spares import HyperPodSpareCoordinator

LOGGER = logging.getLogger(__name__)


# ARCH-A4b: the regional executor has no store, so a stale warm-spare
# reservation can only be judged by its timestamp. One day mirrors the
# control-plane controller's default; five minutes between sweeps is far
# below the TTL and costs one node list per sweep.
SPARE_RESERVATION_TTL_SECONDS = 86400.0
SPARE_RESERVATION_SWEEP_INTERVAL_SECONDS = 300.0


class SpareReservationSweep:
    """Reclaim warm-spare reservations whose owner can no longer be asked.

    The control plane's ``HyperPodSpareHealthController`` reads the owning
    workflow from its store; the regional executor is storeless, so
    ``SpareReservationReclaimer`` runs here with ``store=None`` and only the
    ``reserved-at`` TTL decides. A reservation without that annotation is kept
    (no evidence of staleness), and a spare running GPU pods is never touched.
    """

    def __init__(
        self,
        coordinator: HyperPodSpareCoordinator,
        *,
        ttl_seconds: float = SPARE_RESERVATION_TTL_SECONDS,
        interval_seconds: float = SPARE_RESERVATION_SWEEP_INTERVAL_SECONDS,
        now: Callable[[], datetime] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        from gpu_fault.spare_health import SpareReservationReclaimer

        self.coordinator = coordinator
        self.interval_seconds = interval_seconds
        self.clock = clock
        self.reclaimer = SpareReservationReclaimer(
            coordinator,
            None,
            now=now or (lambda: datetime.now(timezone.utc)),
            ttl_seconds=ttl_seconds,
        )
        self.reclaimed_total = 0
        self._next_due: float | None = None

    def due(self) -> bool:
        return self._next_due is None or self.clock() >= self._next_due

    def run(self) -> list[str]:
        """Sweep every spare-labelled node once; returns the nodes released."""

        self._next_due = self.clock() + self.interval_seconds
        released: list[str] = []
        nodes = self.coordinator.lifecycle.list_nodes(enrich=True)
        # The spare label lives on the Kubernetes Node (that is what the
        # declaration writes and what ``allocate`` reads through
        # ``_declared_spare_names``). The provider view's ``kubernetes_labels``
        # is only a convenience copy and is empty on the regional executor
        # (2026-09-08: every sweep skipped every node, so a two-day-old
        # reservation was never reclaimed). Judge by the Node, accept the
        # enriched copy when it is present.
        declared = self.coordinator._declared_spare_names(nodes)
        for node in nodes:
            node_name = self.coordinator._kubernetes_node_name(node)
            if node_name is None:
                continue
            enriched = (
                node.kubernetes_labels.get(self.coordinator.spare_label)
                == self.coordinator.spare_label_value
            )
            if node_name not in declared and not enriched:
                continue
            kubernetes_node = self.coordinator.core.read_node(node_name)
            reservation = self.coordinator._annotation(kubernetes_node)
            if not reservation:
                continue
            reason = self.reclaimer.reason(node_name, kubernetes_node, reservation)
            if reason is None:
                continue
            self.coordinator.release([node_name], reservation)
            self.reclaimed_total += 1
            released.append(node_name)
            LOGGER.warning(
                "reclaimed stale spare reservation: node=%s incident=%s reason=%s",
                node_name,
                reservation,
                reason,
            )
        return released
