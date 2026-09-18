from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta

from gpu_fault.host_health import NodeHealthCategory
from gpu_fault.models import (
    IncidentState,
    RecoveryAction,
    Severity,
    WorkflowOperation,
    WorkloadState,
)
from gpu_fault.operation_registry import WORKLOAD_SCOPED_OPERATIONS
from gpu_fault.orchestration.disposition import widen_in_place
from gpu_fault.orchestration.families.evidence import EvidenceOperationService
from gpu_fault.orchestration.families.grouped_health import (
    GroupedHealthCallbacks,
    GroupedHealthService,
)
from tests._builders import (
    attempt_observation,
    container_observation,
    fault_incident,
    node_health_finding,
    workflow_request,
    workflow_step,
)
from tests.orchestration._cov95_closure_extra_support import NOW, merger

STOP = WorkflowOperation.STOP_WORKLOADS
RESTART = WorkflowOperation.RESTART_WORKLOAD
RESET = WorkflowOperation.RESET_GPU
QUIESCE = WorkflowOperation.QUIESCE_GPU_SERVICES
RESTORE = WorkflowOperation.RESTORE_SCHEDULING
WORKLOAD = "training/job/unit-job"


def observed_attempt(**values):
    return attempt_observation(
        **{
            "job_id": "unit-job",
            "attempt_id": "attempt-current",
            "observed_at": NOW,
            "started_at": NOW - timedelta(minutes=1),
            "restart_budget": 2,
            "workload_ids": [WORKLOAD],
            "containers": [
                container_observation(
                    f"pod-{node}",
                    f"trainer-{node}",
                    rank,
                    f"node-{node}",
                    gpu_uuids=[f"GPU-{node}"],
                    gpu_count=1,
                    cgroup_path=f"/unit-workload/{node}/",
                )
                for rank, node in enumerate(("a", "b"))
            ],
            **values,
        }
    )


def finding(event_id="unit-finding", **values):
    return node_health_finding(
        f"finding-{event_id}",
        event_id,
        **{
            "observed_at": NOW,
            "category": NodeHealthCategory.GPU,
            "severity": Severity.CRITICAL,
            "reason": "unit health evidence requires recovery",
            "recommended_action": RecoveryAction.RESET_GPU,
            "workload_state": WorkloadState.ACTIVE,
            "affected_workload_ids": [WORKLOAD],
            "gpu_uuids": ["GPU-a"],
            **values,
        },
    )


def candidate_pair(
    identity="candidate",
    *,
    operations=(STOP, QUIESCE, RESET, RESTORE, RESTART),
    action=RecoveryAction.RESET_GPU,
):
    workflow = workflow_request(
        f"wf-{identity}",
        identity,
        fencing_token=1,
        official_steps=[
            workflow_step(
                operation,
                node_ids=["node-a"],
                gpu_uuids=[] if operation in {STOP, RESTART} else ["GPU-a"],
                workload_ids=[WORKLOAD],
            )
            for operation in operations
        ],
        created_at=NOW,
        updated_at=NOW,
    )
    incident = fault_incident(
        identity,
        f"event-{identity}",
        state=IncidentState.ACTION_PENDING,
        effective_action=action,
        workflow_request_id=workflow.request_id,
        fencing_token=workflow.fencing_token,
        gpu_uuids=["GPU-a"],
        reasons=["initial health evidence"],
        created_at=NOW,
        updated_at=NOW,
    )
    return incident, workflow


def attempt_key(cluster_id, job_id, attempt_id):
    return json.dumps(
        ["attempt", cluster_id, job_id, attempt_id], separators=(",", ":")
    )


def seed_group(store, observation, incident, workflow):
    incident = incident.model_copy(
        update={
            "event_type": "GPU_FAULT_GROUP",
            "job_id": observation.job_id,
            "attempt_id": observation.attempt_id,
        }
    )
    return store.merge_attempt_fault_workflow(
        attempt_key(observation.cluster_id, observation.job_id, observation.attempt_id),
        incident.event_id,
        lambda previous_incident, previous_workflow: (incident, workflow),
    )


@dataclass
class GroupHarness:
    service: GroupedHealthService
    candidates: list
    widenings: list


def grouped_service(
    store, observation, candidate, *, active=None, groupable=True, aggregation_window=3
):
    candidate_calls = []
    widenings = []
    merge_service = merger()
    evidence = EvidenceOperationService(store, lambda *args, **kwargs: None)

    def compile_candidate(item, **controls):
        candidate_calls.append((item, controls))
        return candidate

    def widen(workflow, mapping):
        widenings.append({node: list(gpus) for node, gpus in mapping.items()})
        for node, gpus in mapping.items():
            workflow = workflow.model_copy(
                update={
                    "official_steps": widen_in_place(
                        workflow,
                        workflow,
                        node,
                        set(gpus),
                        workload_scoped_operations=WORKLOAD_SCOPED_OPERATIONS,
                    )
                }
            )
        return workflow

    callbacks = GroupedHealthCallbacks(
        active_job_recovery_workflow=lambda item: active,
        aggregation_deadlines=lambda now, workflow: (
            now + timedelta(seconds=aggregation_window),
            now + timedelta(seconds=4 * aggregation_window),
        ),
        attempt_group_key=attempt_key,
        attempt_observation=lambda item: observation,
        incident_state_for_workflow=lambda workflow: IncidentState.ACTION_PENDING,
        ingest_node_health=compile_candidate,
        is_attempt_grouped_health_finding=lambda item: groupable,
        merge_disposition=merge_service.disposition,
        prepare_preempting_successor=merge_service.prepare_preempting_successor,
        preempt_parallel_job_branch=merge_service.preempt_parallel_branch,
        quiesce_parameters=evidence.quiesce_parameters,
        reopen_if_terminal=lambda incident, workflow: (incident, workflow),
        utc=EvidenceOperationService.utc,
        widen_node_action_scope=widen,
    )
    return GroupHarness(
        GroupedHealthService(
            store,
            merge_service.arbiter,
            merge_service.brancher,
            callbacks,
            aggregation_window_seconds=aggregation_window,
            workflow_preemption_enabled=True,
        ),
        candidate_calls,
        widenings,
    )
