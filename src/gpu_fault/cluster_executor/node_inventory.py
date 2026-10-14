"""Report the cluster's Kubernetes Node names to the control plane.

The departed-node cleanup (``gpu_fault.departed_node_cleanup``) retires the
records a reclaimed or replaced node leaves behind, and its first proof is
that the node is no longer in the cluster's Kubernetes. The control plane has
no GPU kubeconfig; the executor has the cluster token and a Kubernetes client
already, so every ``NODE_INVENTORY_INTERVAL_SECONDS`` it LISTs the cluster's
Nodes -- every Node, unfiltered, the question ``kubectl get nodes`` answers for
the administrator's ``workflow-reconcile`` -- and posts their names
(``KubernetesNodeInventoryReport``). Nodes carrying the warm-spare label or a
spare reservation are named as spares: a departed spare is the
administrator's to release.

Invariants, each of them a test:

* the LIST follows ``metadata._continue`` and the whole answer is one report;
  a page that fails fails the report (a partial list would read as absences);
* the report time is taken before the LIST, so it never claims a moment the
  answer does not cover;
* a failure is one warning and the next cycle's retry, never an exception into
  the claim loop; a control plane that predates the route (404) is said once;
* logs carry node counts, never tokens.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable

from gpu_fault.adapters.kubernetes.node_operations import (
    SPARE_LABEL,
    SPARE_LABEL_VALUE,
)
from gpu_fault.cluster_executor.regional_client import (
    ClusterExecutorError,
    RegionalExecutorClient,
)
from gpu_fault.hyperpod_spares import SPARE_RESERVATION_ANNOTATION
from gpu_fault.kubernetes_node_inventory import (
    NODE_INVENTORY_INTERVAL_SECONDS,
    NODE_INVENTORY_MAX_NODES,
    KubernetesNodeInventoryReport,
)

# The pre-split module's name: operators filter on ``gpu_fault.cluster_executor``.
LOGGER = logging.getLogger("gpu_fault.cluster_executor")

#: Nodes per LIST page; the continue token pins every page to one snapshot.
NODE_INVENTORY_PAGE_SIZE = 500


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


class NodeInventoryReporter:
    """One report of the cluster's Node names per interval."""

    def __init__(
        self,
        client: RegionalExecutorClient,
        core: Any,
        *,
        executor_id: str,
        interval_seconds: float = NODE_INVENTORY_INTERVAL_SECONDS,
        page_size: int = NODE_INVENTORY_PAGE_SIZE,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if not executor_id.strip():
            raise ValueError("executor_id must not be empty")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if page_size < 1:
            raise ValueError("page_size must be positive")
        self.client = client
        self.core = core
        self.executor_id = executor_id.strip()
        self.interval_seconds = interval_seconds
        self.page_size = page_size
        self.clock = clock
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.reports_total = 0
        self.failures_total = 0
        self._next_due: float | None = None
        self._route_missing_reported = False

    def due(self) -> bool:
        return self._next_due is None or self.clock() >= self._next_due

    def snapshot(self) -> KubernetesNodeInventoryReport:
        """Every Node of the cluster, read as one paginated LIST."""

        observed_at = self.now()
        names: list[str] = []
        spares: list[str] = []
        resource_version: str | None = None
        token: str | None = None
        while True:
            page = self.core.list_node(limit=self.page_size, _continue=token)
            metadata = _value(page, "metadata", {}) or {}
            if resource_version is None:
                version = _value(metadata, "resource_version") or _value(
                    metadata, "resourceVersion"
                )
                resource_version = str(version) if version else None
            for item in _value(page, "items", []) or []:
                node_metadata = _value(item, "metadata", {}) or {}
                name = _value(node_metadata, "name")
                if not name:
                    continue
                names.append(str(name))
                labels = _value(node_metadata, "labels", {}) or {}
                annotations = _value(node_metadata, "annotations", {}) or {}
                if labels.get(SPARE_LABEL) == SPARE_LABEL_VALUE or annotations.get(
                    SPARE_RESERVATION_ANNOTATION
                ):
                    spares.append(str(name))
            if len(names) > NODE_INVENTORY_MAX_NODES:
                raise ClusterExecutorError(
                    f"the cluster has more than {NODE_INVENTORY_MAX_NODES} Nodes; "
                    "the inventory is not reported"
                )
            token = _value(metadata, "_continue") or _value(metadata, "continue")
            if not token:
                break
        return KubernetesNodeInventoryReport(
            cluster_id=self.client.cluster_id,
            executor_id=self.executor_id,
            observed_at=observed_at,
            resource_version=resource_version,
            node_names=sorted(set(names)),
            spare_node_names=sorted(set(spares)),
        )

    def run(self) -> int:
        """Read and post one report; returns the number of Nodes reported."""

        self._next_due = self.clock() + self.interval_seconds
        try:
            report = self.snapshot()
            self.client.report_node_inventory(report.model_dump(mode="json"))
        except Exception:
            self.failures_total += 1
            raise
        self.reports_total += 1
        return len(report.node_names)

    def report(self) -> None:
        """``run`` when due; never raises into the claim loop."""

        if not self.due():
            return
        try:
            self.run()
        except ClusterExecutorError as exc:
            if exc.status_code == 404:
                # A control plane from before the departed-node cleanup: the
                # report has nowhere to go until it is upgraded. Said once.
                if not self._route_missing_reported:
                    self._route_missing_reported = True
                    LOGGER.info(
                        "the control plane does not take Kubernetes node "
                        "reports yet; departed-node cleanup waits for its upgrade"
                    )
                return
            LOGGER.warning(
                "Kubernetes node report failed (failures %d): %s",
                self.failures_total,
                exc,
            )
        except Exception as exc:  # noqa: BLE001 - housekeeping must not stop claims
            LOGGER.warning(
                "Kubernetes node report failed (failures %d): %s: %s",
                self.failures_total,
                type(exc).__name__,
                exc,
            )
