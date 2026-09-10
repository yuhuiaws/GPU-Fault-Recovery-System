from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

from gpu_fault.models import (
    MarkerScope,
    NodeMarker,
    RecoveryAction,
    Severity,
)
from gpu_fault.store import PostgresStore


def store_dsn() -> str:
    path = (
        os.environ.get("GPU_FAULT_STORE_URL_FILE")
        or "/etc/gpu-fault/aurora/postgres-url"
    )
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return os.environ["GPU_FAULT_STORE_URL"]


def main() -> None:
    mode, cluster_id, attempt_id, marker_id = sys.argv[1:5]
    store = PostgresStore(
        store_dsn(),
        pool_min_size=1,
        pool_max_size=2,
        pool_timeout_seconds=5,
    )
    try:
        if mode == "disable":
            marker = next(
                item for item in store.list_markers() if item.marker_id == marker_id
            )
            store.add_marker(marker.model_copy(update={"active": False}))
            print("disabled", marker_id)
            return

        observation = next(
            item
            for item in store.list_attempt_observations(cluster_id)
            if item.attempt_id == attempt_id
        )
        node_ids = sorted(
            {
                container.node_id
                for container in observation.containers
                if container.node_id
            }
        )
        now = datetime.now(timezone.utc)
        marker = NodeMarker(
            marker_id=marker_id,
            source="audit-q118",
            trusted=True,
            incident_id=f"{marker_id}-incident",
            observed_at=now,
            expires_at=now + timedelta(hours=1),
            scope=MarkerScope(node_ids=node_ids),
            severity=Severity.CRITICAL,
            recommended_action=RecoveryAction.RESTART_WORKLOAD,
            mapping_version="audit-q118",
            drill_id="q118-emergency-fallback",
            raw_reason=(
                "restart the controlled drill after passive emergency containment"
            ),
        )
        store.add_marker(marker)
        print(
            "created",
            marker.marker_id,
            observation.workload_phase.value,
            node_ids,
        )
    finally:
        store.close()


if __name__ == "__main__":
    main()
