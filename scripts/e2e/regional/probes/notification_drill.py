from __future__ import annotations

import argparse
import json

from gpu_fault.models import AdvisoryNotification
from gpu_fault.notification_service import AdvisoryNotificationService
from gpu_fault.notifications import (
    RestartGuardEmailBuilder,
    notification_notifier_from_environment,
)
from gpu_fault.store import InMemoryStore


def build_notification(
    kind: str,
    drill_id: str,
    cluster_id: str,
) -> AdvisoryNotification:
    builder = RestartGuardEmailBuilder()
    if kind == "gpu-reset":
        notification = builder.build_gpu_reset_completed(
            cluster_id=cluster_id,
            incident_id=f"incident-{drill_id}",
            workflow_id=f"workflow-{drill_id}",
            event_id=f"event-{drill_id}",
            event_type="NOTIFY_ACCEPTANCE_DRILL",
            policy_source="ACCEPTANCE_DRILL",
            official_action="RESET_GPU",
            reasons=["notification path acceptance drill; no GPU action occurred"],
            operation_id=f"operation-{drill_id}",
            node_ids=[f"node-{drill_id}"],
            gpu_uuids=[f"GPU-{drill_id}"],
            node_results={
                f"node-{drill_id}": {
                    "status": "SUCCEEDED",
                    "reset_gpu_uuids": [f"GPU-{drill_id}"],
                }
            },
            workload_ids=[],
        )
    else:
        notification = builder.build_workload_restarted(
            cluster_id=cluster_id,
            incident_id=f"incident-{drill_id}",
            workflow_id=f"workflow-{drill_id}",
            operation_id=f"operation-{drill_id}",
            job_id=f"job-{drill_id}",
            source_attempt_id=f"attempt-{drill_id}-source",
            restart_attempt_id=f"attempt-{drill_id}-target",
            workload_ids=[f"training/pytorchjob/job-{drill_id}"],
            source_gpu_count=24,
            target_gpu_count=24,
            restart_count=1,
            restart_budget=1,
        )
    return notification.model_copy(
        update={
            "drill_id": drill_id,
            "subject": f"[DRILL:{drill_id}] {notification.subject}",
            "body_text": (
                "DRILL / 验收演练：未执行 GPU reset、任务重启或节点变更。\n\n"
                + notification.body_text
            ),
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind", choices=("gpu-reset", "workload-restart"), required=True
    )
    parser.add_argument("--drill-id", required=True)
    parser.add_argument("--cluster-id", required=True)
    arguments = parser.parse_args()
    store = InMemoryStore()
    notification = store.save_notification_if_absent(
        build_notification(arguments.kind, arguments.drill_id, arguments.cluster_id)
    )
    service = AdvisoryNotificationService(
        store,
        notification_notifier_from_environment(),
        async_delivery=False,
        deliver_drills=True,
    )
    results = [service.send(notification.notification_id) for _ in range(4)]
    provider_ids = [item.provider_message_id for item in results]
    print(
        json.dumps(
            {
                "kind": arguments.kind,
                "drill_id": arguments.drill_id,
                "notification_id": notification.notification_id,
                "statuses": [item.status.value for item in results],
                "provider_message_id_present": bool(provider_ids[0]),
                "provider_message_id_stable": (
                    bool(provider_ids[0])
                    and all(item == provider_ids[0] for item in provider_ids)
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
