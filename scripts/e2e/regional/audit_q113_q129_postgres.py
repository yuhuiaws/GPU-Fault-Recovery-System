from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

from gpu_fault.models import (
    MarkerScope,
    NodeMarker,
    RecoveryAction,
    Severity,
    WorkflowRequest,
    WorkflowStatus,
)
from gpu_fault.store import PostgresStore


EXPECTED_INDEXES = {
    "gpu_fault_failed_workflow_updated",
    "gpu_fault_active_marker_action",
    "gpu_fault_active_marker_nodes",
}


def main() -> None:
    stamp = str(int(time.time()))
    prefix = f"audit-q113-{stamp}"
    now = datetime.now(timezone.utc)
    store = PostgresStore(
        os.environ["GPU_FAULT_STORE_URL"],
        pool_min_size=1,
        pool_max_size=2,
        pool_timeout_seconds=5,
    )
    object_keys: list[tuple[str, str]] = []
    try:
        unhandled = WorkflowRequest(
            request_id=f"{prefix}-failed-unhandled",
            incident_id=f"{prefix}-incident-unhandled",
            status=WorkflowStatus.FAILED,
            fencing_token=1,
            created_at=now,
            updated_at=now,
        )
        handled = WorkflowRequest(
            request_id=f"{prefix}-failed-handled",
            incident_id=f"{prefix}-incident-handled",
            status=WorkflowStatus.FAILED,
            fencing_token=1,
            failure_handled_at=now,
            created_at=now,
            updated_at=now,
        )
        for workflow in (unhandled, handled):
            store.save_workflow(workflow)
            object_keys.append(("workflow", workflow.request_id))

        first_scan = {
            workflow.request_id
            for workflow in store.list_unhandled_failed_workflows(limit=1000)
            if workflow.request_id.startswith(prefix)
        }
        assert first_scan == {unhandled.request_id}, first_scan

        handled_unhandled = unhandled.model_copy(
            update={
                "failure_handled_at": now + timedelta(seconds=1),
                "updated_at": now + timedelta(seconds=1),
            }
        )
        store.save_workflow(handled_unhandled)
        second_scan = {
            workflow.request_id
            for workflow in store.list_unhandled_failed_workflows(limit=1000)
            if workflow.request_id.startswith(prefix)
        }
        assert second_scan == set(), second_scan

        marker_cases = [
            NodeMarker(
                marker_id=f"{prefix}-target",
                source="audit",
                trusted=True,
                incident_id=f"{prefix}-incident-target",
                observed_at=now,
                expires_at=now + timedelta(hours=1),
                scope=MarkerScope(node_ids=["audit-node-a"]),
                severity=Severity.CRITICAL,
                recommended_action=RecoveryAction.REBOOT_NODE,
                mapping_version="audit",
            ),
            NodeMarker(
                marker_id=f"{prefix}-other-node",
                source="audit",
                trusted=True,
                incident_id=f"{prefix}-incident-other-node",
                observed_at=now,
                expires_at=now + timedelta(hours=1),
                scope=MarkerScope(node_ids=["audit-node-b"]),
                severity=Severity.CRITICAL,
                recommended_action=RecoveryAction.REBOOT_NODE,
                mapping_version="audit",
            ),
            NodeMarker(
                marker_id=f"{prefix}-inactive",
                source="audit",
                trusted=True,
                incident_id=f"{prefix}-incident-inactive",
                observed_at=now,
                expires_at=now + timedelta(hours=1),
                scope=MarkerScope(node_ids=["audit-node-a"]),
                severity=Severity.CRITICAL,
                recommended_action=RecoveryAction.REBOOT_NODE,
                mapping_version="audit",
                active=False,
            ),
            NodeMarker(
                marker_id=f"{prefix}-other-action",
                source="audit",
                trusted=True,
                incident_id=f"{prefix}-incident-other-action",
                observed_at=now,
                expires_at=now + timedelta(hours=1),
                scope=MarkerScope(node_ids=["audit-node-a"]),
                severity=Severity.CRITICAL,
                recommended_action=RecoveryAction.QUARANTINE,
                mapping_version="audit",
            ),
        ]
        for marker in marker_cases:
            store.add_marker(marker)
            object_keys.append(("marker", marker.marker_id))

        selected_markers = {
            marker.marker_id
            for marker in store.list_active_markers_for_nodes(
                {"audit-node-a"},
                {RecoveryAction.REBOOT_NODE},
            )
            if marker.marker_id.startswith(prefix)
        }
        assert selected_markers == {f"{prefix}-target"}, selected_markers

        with store._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT indexname
                FROM pg_indexes
                WHERE schemaname=current_schema()
                  AND indexname=ANY(%s)
                """,
                (sorted(EXPECTED_INDEXES),),
            )
            indexes = {row[0] for row in cursor.fetchall()}
        assert indexes == EXPECTED_INDEXES, indexes

        print(
            "q1-13",
            f"first_scan={sorted(first_scan)}",
            f"second_scan={sorted(second_scan)}",
        )
        print(
            "q1-29",
            f"selected_markers={sorted(selected_markers)}",
        )
        print("indexes", sorted(indexes))
    finally:
        try:
            with store._db.transaction(), store._db.cursor() as cursor:
                cursor.executemany(
                    """
                    DELETE FROM gpu_fault_objects
                    WHERE kind=%s AND key=%s
                    """,
                    object_keys,
                )
        finally:
            store.close()


if __name__ == "__main__":
    main()
