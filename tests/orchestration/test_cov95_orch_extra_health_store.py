from __future__ import annotations

import pytest

from gpu_fault.models import IncidentState, RecoveryAction, WorkflowOperation
from gpu_fault.orchestration.families.health import HOST_RESOURCE_POLICY_SOURCE
from tests._builders import fault_incident
from tests.orchestration._cov95_orch_extra_health import health_service
from tests.orchestration._cov95_orch_extra_safety import (
    orch_extra_isolation as orch_extra_isolation,
)
from tests.orchestration._cov95_orch_extra_support import (
    finding,
    memory_store,
    stored_workflow,
)


def test_recorded_event_without_a_workflow_does_not_create_a_second_diagnostic():
    store = memory_store()
    recorded = fault_incident(
        "inc-recorded", "unit-finding", state=IncidentState.RECOVERED
    )
    store.save_incident(recorded)
    service, absorbed = health_service(store)
    assert service.ingest(finding()) == (recorded, None)
    assert store.list_workflows() == []
    assert absorbed == []


def test_dangling_health_workflow_pointer_is_repaired_under_the_same_event_identity():
    store = memory_store()
    service, _absorbed = health_service(store)
    source = finding(category="NCCL")
    incident, original = service.ingest(source)
    read = store.get_incident(incident.incident_id)
    store.save_incident(
        read.model_copy(update={"workflow_request_id": "unit-vanished-workflow"}),
        expected=read,
    )
    repaired, workflow = service.ingest(source)
    assert repaired.incident_id == incident.incident_id
    assert workflow.request_id == original.request_id
    assert repaired.workflow_request_id == workflow.request_id
    assert store.get_incident_by_event(source.event_id) == repaired
    assert store.get_workflow(workflow.request_id) == workflow
    assert len(store.list_workflows()) == 1


@pytest.mark.parametrize("persist", [False, True])
def test_inventory_absorption_links_only_a_persisted_observation(persist):
    store = memory_store()
    incident, workflow = stored_workflow(
        store, [WorkflowOperation.RESTART_NODE, WorkflowOperation.VALIDATE_GPU]
    )
    service, absorbed = health_service(store)
    source = finding(
        "inventory-preview",
        node_id="node-0",
        metric_name="gpu_inventory_mismatch",
        recommended_action=RecoveryAction.REBOOT_NODE,
    )
    assert service.ingest(source, persist=persist) == (incident, workflow)
    assert store.get_incident_by_event(source.event_id) == (
        incident if persist else None
    )
    assert store.get_workflow(workflow.request_id) == workflow
    assert absorbed == []


def test_host_resource_preview_does_not_amend_or_link_the_unsettled_incident():
    store = memory_store()
    source = finding(
        "batch-host_gpu_utilization_percent-GPU-0-low_gpu_utilization",
        policy_source=HOST_RESOURCE_POLICY_SOURCE,
        metric_name="host_gpu_utilization_percent",
        diagnostic_parameters={"signal": "LOW_GPU_UTILIZATION"},
    )
    prior = fault_incident(
        "inc-unsettled",
        "earlier-host_gpu_utilization_percent-GPU-1-low_gpu_utilization",
        policy_source=HOST_RESOURCE_POLICY_SOURCE,
        state=IncidentState.ESCALATED,
    )
    store.save_incident(prior)
    service, absorbed = health_service(store)
    assert service.ingest(source, persist=False) == (prior, None)
    assert store.get_incident(prior.incident_id) == prior
    assert store.get_incident_by_event(source.event_id) is None
    assert store.list_workflows() == []
    assert absorbed == []


@pytest.mark.parametrize("signal", [None, "", 1])
def test_host_resource_finding_without_a_usable_signal_cannot_absorb_an_incident(
    signal,
):
    store = memory_store()
    source = finding(
        policy_source=HOST_RESOURCE_POLICY_SOURCE,
        metric_name="host_gpu_utilization_percent",
        diagnostic_parameters={"signal": signal},
    )
    prior = fault_incident(
        "inc-unsettled",
        "earlier-host_gpu_utilization_percent-GPU-1-low_gpu_utilization",
        policy_source=HOST_RESOURCE_POLICY_SOURCE,
        state=IncidentState.ESCALATED,
    )
    store.save_incident(prior)
    service, absorbed = health_service(store)
    incident, workflow = service.ingest(source)
    assert incident.incident_id != prior.incident_id
    assert workflow is not None, "a distinct signal was incorrectly absorbed"
    assert store.get_incident(prior.incident_id) == prior
    assert absorbed == []


@pytest.mark.parametrize("different", ["rule", "policy-source"])
def test_host_resource_finding_does_not_absorb_a_different_rule_or_policy_source(
    different,
):
    store = memory_store()
    prior = fault_incident(
        "inc-unsettled",
        "earlier-host_gpu_utilization_percent-GPU-1-"
        + ("another_rule" if different == "rule" else "low_gpu_utilization"),
        policy_source=(
            HOST_RESOURCE_POLICY_SOURCE if different == "rule" else "ANOTHER_POLICY"
        ),
        state=IncidentState.ESCALATED,
    )
    store.save_incident(prior)
    service, absorbed = health_service(store)
    incident, workflow = service.ingest(
        finding(
            "batch-host_gpu_utilization_percent-GPU-0-low_gpu_utilization",
            policy_source=HOST_RESOURCE_POLICY_SOURCE,
            metric_name="host_gpu_utilization_percent",
            diagnostic_parameters={"signal": "LOW_GPU_UTILIZATION"},
        )
    )
    assert incident.incident_id != prior.incident_id
    assert workflow is not None, (
        "another rule or policy source suppressed this signal's diagnostic"
    )
    assert store.get_incident(prior.incident_id) == prior
    assert absorbed == []


def test_conflicting_health_repair_serializes_without_preempting_another_job():
    store = memory_store()
    old_incident, old_workflow = stored_workflow(
        store,
        [WorkflowOperation.RESET_GPU],
        incident_updates={"job_id": "older-job", "attempt_id": "older-attempt"},
    )
    service, _absorbed = health_service(store)
    incident, workflow = service.ingest(
        finding(
            node_id="node-0",
            recommended_action=RecoveryAction.RESET_GPU,
            gpu_uuids=["GPU-0"],
            job_id="new-job",
            attempt_id="new-attempt",
        )
    )
    assert workflow.predecessor_workflow_id == old_workflow.request_id
    assert any("serialized behind" in reason for reason in incident.reasons), (
        "the conflicting incumbent was not recorded"
    )
    assert store.get_workflow(old_workflow.request_id) == old_workflow
    assert store.get_incident(old_incident.incident_id) == old_incident
