from __future__ import annotations

from datetime import datetime, timezone

from gpu_fault.app import default_simulated_profile
from gpu_fault.host_health import NodeHealthCategory, NodeHealthFinding
from gpu_fault.models import RecoveryAction, Severity, WorkflowOperation, WorkloadState
from gpu_fault.orchestration import IncidentOrchestrator
from tests._builders import build_store, node_health_finding


def _finding() -> NodeHealthFinding:
    return node_health_finding(
        "finding-restart-identity",
        "event-restart-identity",
        observed_at=datetime.now(timezone.utc),
        category=NodeHealthCategory.GPU,
        severity=Severity.CRITICAL,
        reason="node level GPU fault",
        recommended_action=RecoveryAction.REPLACE_NODE,
        gpu_uuids=["GPU-aaa", "GPU-bbb"],
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/trainer"],
        job_id="trainer-job",
        attempt_id="trainer-job-a1",
    )


def _restart_parameters(finding):
    store = build_store()
    store.save_profile(default_simulated_profile())
    orchestrator = IncidentOrchestrator(store)
    incident, _ = orchestrator.ingest_node_health(finding)
    workflow = store.get_workflow(incident.workflow_request_id)
    for step in workflow.official_steps:
        if step.operation is WorkflowOperation.RESTART_WORKLOAD:
            return dict(step.parameters)
    raise AssertionError("no RESTART_WORKLOAD step was planned")


def test_node_health_restart_uses_the_finding_job_identity():
    """The planned job identity must match the workload's own labels.

    The node-health path dropped the finding's job_id and fell back to
    the workload id, so _restart_guard compared "training/trainer"
    against the label "trainer-job" and refused to restart -- the
    workload stayed stopped after an otherwise successful replacement.
    """
    parameters = _restart_parameters(_finding())

    assert parameters["job_id"] == "trainer-job"
    assert parameters["source_attempt_id"] == "trainer-job-a1"


def test_node_health_restart_counts_the_findings_own_gpus():
    """source_gpu_count must not be 0 when the detector named the GPUs.

    A node-level fault is detected before any attempt observation is
    recorded, so the observation-derived count is empty. Leaving it 0
    sends _restart_guard into a store lookup that the regional executor
    (which owns no local store) cannot perform.
    """
    parameters = _restart_parameters(_finding())

    assert parameters["source_gpu_count"] == 2
