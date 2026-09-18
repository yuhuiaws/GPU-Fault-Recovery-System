from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.models import WorkflowOperation
from tests._builders import attempt_observation, container_observation
from tests.execution._cov95_runtime_workflows import FlowHarness

TRIAGE = WorkflowOperation.COLLECT_HUNG_TRIAGE
BUNDLE = WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE


def signal(pid: Any, rank: Any, sequence: int = 10, **updates: Any) -> dict[str, Any]:
    return {
        "pid": pid,
        "rank": rank,
        "flight_recorder": {
            "last_entry": {
                "pg_name": "default",
                "collective_seq_id": sequence,
                "state": "started",
            }
        },
        **updates,
    }


def triage_harness(
    details: dict[str, Any],
    *,
    observe_job: str | None = "job-a",
    linked_bundle: bool = True,
    incident_job: str | None = "job-a",
    incident_attempt: str | None = "attempt-a",
) -> FlowHarness:
    h = FlowHarness([TRIAGE, BUNDLE])
    h.incident = h.incident.model_copy(
        update={"job_id": incident_job, "attempt_id": incident_attempt}
    )
    h.store.save_incident(h.incident)
    h.amend(
        dag_enabled=True,
        official_steps=[
            h.workflow.official_steps[0],
            h.workflow.official_steps[1].model_copy(
                update={
                    "node_ids": ["node-a", "node-b", "node-c"],
                    "depends_on_step_indexes": [0] if linked_bundle else [],
                    "parameters": {
                        "hung_triage_target_pending": True,
                        "capture_process_state": False,
                    },
                }
            ),
        ],
    )
    if observe_job is not None:
        h.store.save_attempt_observation(
            attempt_observation(
                observe_job,
                "attempt-a",
                datetime.now(timezone.utc),
                expected_critical_ranks=3,
                containers=[
                    container_observation(
                        f"pod-{index}",
                        "worker",
                        index,
                        node,
                        host_pid=(index + 1) * 100,
                        gpu_uuids=[f"GPU-{node}"],
                    )
                    for index, node in enumerate(["node-a", "node-b", "node-c"])
                ],
            )
        )
    h.adapter.outcomes[TRIAGE] = WorkflowStepOutcome.succeeded(details=details)
    h.adapter.outcomes[BUNDLE] = WorkflowStepOutcome.waiting(
        operation_id="fake-diagnostic-collection"
    )
    return h
