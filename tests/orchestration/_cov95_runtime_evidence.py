from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.families.conflicts import NodeConflictService
from gpu_fault.orchestration.families.evidence import EvidenceOperationService
from gpu_fault.store import InMemoryStore
from tests._builders import (
    attempt_observation,
    container_observation,
    fault_incident,
    workflow_request,
    workflow_step,
)


class EvidenceHarness:
    def __init__(self) -> None:
        self.now = datetime.now(timezone.utc)
        self.store = InMemoryStore()
        self.conflicts = NodeConflictService(self.store, RecoveryArbiter())
        self.service = EvidenceOperationService(
            self.store, self.conflicts.active_node_exclusive_workflow
        )

    def observation(self, *, job="job-a", attempt="attempt-a", **updates: Any):
        values = {
            "started_at": self.now - timedelta(minutes=1),
            "workload_ids": [f"training/job/{job}"],
            "containers": [
                container_observation(
                    "pod-a",
                    "trainer",
                    0,
                    "node-a",
                    gpu_uuids=["GPU-a"],
                    container_id="containerd://container-a",
                    host_pid=100,
                    cgroup_path="/kubepods/pod-a",
                )
            ],
            **updates,
        }
        observed_at = values.pop("observed_at", self.now)
        result = attempt_observation(job, attempt, observed_at, **values)
        self.store.save_attempt_observation(result)
        return result

    def active_recovery(self):
        incident = fault_incident(
            "active-repair",
            "active-repair-event",
            cluster_id="cluster-a",
            job_id="job-a",
            attempt_id="attempt-a",
            node_ids=["node-a"],
            gpu_uuids=["GPU-a"],
            workflow_request_id="active-repair-workflow",
            state=IncidentState.ACTION_PENDING,
        )
        workflow = workflow_request(
            "active-repair-workflow",
            incident.incident_id,
            WorkflowStatus.RUNNING,
            official_steps=[
                workflow_step(WorkflowOperation.RESET_GPU, node_ids=["node-a"])
            ],
            created_at=self.now - timedelta(seconds=5),
        )
        self.store.save_incident_and_workflow(incident, workflow)
        return self.store.get_workflow(workflow.request_id)
