"""Health-family plans for a node with no running workload.

A STOP_WORKLOAD finding (and the XID 154 DRAIN_P2P path it carries) on an idle
node has nothing to stop and nothing to restart: the plan must neither stop a
workload that is not there nor restart one that never ran. The service is the
real ``NodeHealthIngestionService`` over an in-memory store
(``tests/orchestration/_cov95_orch_extra_health.py``).
"""

from __future__ import annotations

from gpu_fault.models import (
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkloadState,
)
from gpu_fault.orchestration.families.health import DRAIN_P2P_ACTION
from tests.orchestration._cov95_orch_extra_health import health_service
from tests.orchestration._cov95_orch_extra_support import finding, memory_store

Op = WorkflowOperation


def _plan(source):
    store = memory_store()
    service, _absorbed = health_service(store)
    incident, workflow = service.ingest(source)
    assert workflow is not None, "an idle node still gets its diagnostic plan"
    assert incident.state is IncidentState.ACTION_PENDING, incident.state
    assert workflow.status is WorkflowStatus.PENDING, workflow.blocked_reasons
    return [step.operation for step in workflow.official_steps]


def test_a_stop_workload_finding_on_an_idle_node_stops_nothing() -> None:
    operations = _plan(
        finding(
            "idle-stop",
            category="NCCL",
            recommended_action=RecoveryAction.STOP_WORKLOAD,
            workload_state=WorkloadState.IDLE,
        )
    )
    assert operations == [
        Op.FREEZE_EVIDENCE,
        Op.COLLECT_DIAGNOSTIC_BUNDLE,
        Op.ESCALATE_SUPPORT,
    ], "evidence and the hand-off only: there is no workload to stop"


def test_a_drain_p2p_finding_on_an_idle_node_neither_stops_nor_restarts() -> None:
    operations = _plan(
        finding(
            "idle-drain-p2p",
            recommended_action=RecoveryAction.STOP_WORKLOAD,
            official_action=DRAIN_P2P_ACTION,
            workload_state=WorkloadState.IDLE,
            gpu_uuids=["GPU-1"],
        )
    )
    assert operations == [
        Op.FREEZE_EVIDENCE,
        Op.MARK_UNSCHEDULABLE,
        Op.QUIESCE_GPU_SERVICES,
        Op.VERIFY_NO_GPU_CLIENTS,
        Op.RECOVER_DRAIN_P2P,
        Op.RESTORE_GPU_SERVICES,
        Op.VALIDATE_GPU,
        Op.VALIDATE_FABRIC,
        Op.RESTORE_SCHEDULING,
    ], "the DRAIN_P2P chain without STOP_WORKLOADS or RESTART_WORKLOAD"


def test_a_regrade_of_an_event_without_an_incident_writes_nothing() -> None:
    store = memory_store()
    service, _absorbed = health_service(store)
    assert service.record_regrade(finding("never-ingested")) is None, (
        "the caller ingests an event that has no incident yet"
    )
    assert store.get_incident_by_event("never-ingested") is None, (
        "a regrade never mints an incident"
    )
