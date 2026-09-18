from __future__ import annotations

import pytest

from gpu_fault.host_health import NodeHealthCategory
from gpu_fault.models import (
    BlockedKind,
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.operation_registry import OPERATION_CAPABILITY
from tests._builders import container_observation
from tests.orchestration._cov95_orch_extra_health import health_service
from tests.orchestration._cov95_orch_extra_safety import (
    orch_extra_isolation as orch_extra_isolation,
)
from tests.orchestration._cov95_orch_extra_support import (
    finding,
    memory_store,
    observation,
)


def test_active_efa_repair_compiles_stop_before_mutation_and_restart_after_validation():
    store = memory_store()
    service, _absorbed = health_service(store)
    source = finding(
        category=NodeHealthCategory.RDMA,
        node_id="node-0",
        recommended_action=RecoveryAction.REMEDIATE_EFA_DRIVER,
        workload_state="ACTIVE",
        affected_workload_ids=["training/job/unit-job"],
        job_id="unit-job",
        attempt_id="unit-attempt",
        gpu_uuids=["GPU-0"],
    )
    incident, workflow = service.ingest(source)
    assert workflow.status is WorkflowStatus.PENDING
    assert [step.operation for step in workflow.official_steps] == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.STOP_WORKLOADS,
        WorkflowOperation.REMEDIATE_EFA_DRIVER,
        WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN,
        WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT,
        WorkflowOperation.VALIDATE_FABRIC,
        WorkflowOperation.RESTORE_SCHEDULING,
        WorkflowOperation.RESTART_WORKLOAD,
    ]
    stop = workflow.official_steps[2]
    assert stop.parameters["termination_initiator_incident_id"] == incident.incident_id
    assert workflow.official_steps[-1].parameters["source_attempt_id"] == "unit-attempt"
    assert store.get_workflow(workflow.request_id) == workflow


@pytest.mark.parametrize(
    ("category", "tail"),
    [
        (NodeHealthCategory.CPU, [WorkflowOperation.VALIDATE_HOST]),
        (NodeHealthCategory.NCCL, [WorkflowOperation.VALIDATE_FABRIC]),
        (
            NodeHealthCategory.GPU,
            [WorkflowOperation.RUN_DCGM_DIAGNOSTIC, WorkflowOperation.VALIDATE_GPU],
        ),
    ],
)
def test_process_capture_is_requested_only_as_the_explicit_diagnostic_input(
    category, tail
):
    service, _absorbed = health_service(memory_store())
    _, workflow = service.ingest(
        finding(
            category=category,
            diagnostic_parameters={
                "capture_process_state": True,
                "reason": "unit-triage",
            },
        )
    )
    assert workflow.status is WorkflowStatus.PENDING
    assert [step.operation for step in workflow.official_steps] == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
        *tail,
    ]
    assert workflow.official_steps[1].parameters["capture_process_state"] is True
    assert workflow.dag_enabled is False


def test_no_action_health_finding_keeps_evidence_without_compiling_node_mutations():
    service, _absorbed = health_service(memory_store())
    _, workflow = service.ingest(finding(recommended_action=RecoveryAction.NO_ACTION))
    assert [step.operation for step in workflow.official_steps] == [
        WorkflowOperation.FREEZE_EVIDENCE
    ]
    assert workflow.status is WorkflowStatus.PENDING


@pytest.mark.parametrize("profile", [None, "unit-missing-profile"])
def test_health_plan_without_profile_is_persisted_blocked_for_operator_review(profile):
    store = memory_store()
    service, _absorbed = health_service(store)
    incident, workflow = service.ingest(finding(runtime_profile_version=profile))
    assert workflow.status is WorkflowStatus.BLOCKED
    assert workflow.blocked_kind is BlockedKind.NEEDS_OPERATOR
    assert incident.state is IncidentState.ESCALATED
    assert any("profile" in reason for reason in workflow.blocked_reasons), (
        "profile refusal lost its actionable reason"
    )
    assert (
        store.get_incident_by_event("unit-finding").workflow_request_id
        == workflow.request_id
    )


def test_reset_plan_without_explicit_gpu_identity_cannot_be_executed():
    service, _absorbed = health_service(memory_store())
    incident, workflow = service.ingest(
        finding(recommended_action=RecoveryAction.RESET_GPU, gpu_uuids=[])
    )
    assert workflow.status is WorkflowStatus.BLOCKED
    assert "RESET_GPU requires an explicit GPU UUID" in workflow.blocked_reasons
    assert incident.state is IncidentState.ESCALATED


@pytest.mark.parametrize(
    ("snapshot", "nodes"),
    [
        (None, ["node-0"]),
        ([], ["node-0"]),
        (["node-2"], ["node-0"]),
        (["node-0", None], ["node-0"]),
        (["node-2", "node-0", "node-2"], ["node-0", "node-2"]),
    ],
)
def test_hung_diagnostic_without_observation_accepts_only_a_complete_reporter_snapshot(
    snapshot, nodes
):
    service, _absorbed = health_service(memory_store())
    incident, workflow = service.ingest(
        finding(
            node_id="node-0",
            category=NodeHealthCategory.NCCL,
            diagnostic_parameters={
                "diagnostic_reason": "EFA_TRAFFIC_HUNG_SUSPECTED",
                "capture_process_state": True,
                "attempt_node_ids": snapshot,
            },
        )
    )
    assert incident.node_ids == nodes
    assert workflow.status is WorkflowStatus.PENDING
    assert workflow.dag_enabled is True
    assert [step.depends_on_step_indexes for step in workflow.official_steps] == [
        [],
        [0],
        [1],
        [1],
    ]
    assert workflow.official_steps[1].node_ids == nodes
    assert workflow.official_steps[1].parameters["triage_timeout_seconds"] == 10
    assert workflow.official_steps[2].parameters["capture_process_state"] is False


@pytest.mark.parametrize("case", ["complete", "missing-gpu", "no-pids"])
def test_hung_diagnostic_derives_gpu_and_process_context_from_the_actual_allocation(
    case,
):
    store = memory_store()
    containers = [
        container_observation(
            "pod-0",
            "worker-0",
            0,
            "node-0",
            gpu_uuids=["GPU-0"],
            host_pid=None if case == "no-pids" else 100,
        ),
        container_observation(
            "pod-2",
            "worker-2",
            2,
            "node-0",
            gpu_uuids=["GPU-2"],
            host_pid=None if case == "no-pids" else 102,
        ),
        container_observation(
            "pod-1",
            "worker-1",
            1,
            "node-1",
            gpu_uuids=[] if case == "missing-gpu" else ["GPU-1"],
        ),
    ]
    store.save_attempt_observation(observation(ranks=3, containers=containers))
    service, _absorbed = health_service(store)
    incident, workflow = service.ingest(
        finding(
            node_id="node-0",
            category=NodeHealthCategory.NCCL,
            job_id="unit-job",
            attempt_id="unit-attempt",
            workload_state="ACTIVE",
            affected_workload_ids=["training/job/unit-job"],
            diagnostic_parameters={
                "diagnostic_reason": "EFA_TRAFFIC_HUNG_SUSPECTED",
                "capture_process_state": True,
            },
        )
    )
    assert incident.node_ids == ["node-0", "node-1"]
    parameters = workflow.official_steps[1].parameters
    assert parameters["attempt_node_ids"] == ["node-0", "node-1"]
    if case == "no-pids":
        assert "rank_by_pid_by_node" not in parameters
    else:
        assert parameters["rank_by_pid_by_node"] == {
            "node-0": {"100": 0, "102": 2},
            "node-1": {},
        }
    if case == "missing-gpu":
        assert "gpu_uuids_by_node" not in parameters
    else:
        assert parameters["gpu_uuids_by_node"] == {
            "node-0": ["GPU-0", "GPU-2"],
            "node-1": ["GPU-1"],
        }


def test_hung_diagnostic_with_an_unowned_required_step_is_blocked_as_an_incomplete_dag():
    store = memory_store()
    profile = store.get_profile("simulated-v1")
    missing = OPERATION_CAPABILITY[WorkflowOperation.COLLECT_HUNG_TRIAGE]
    store.save_profile(
        profile.model_copy(
            update={
                "capabilities": [
                    item for item in profile.capabilities if item.capability != missing
                ]
            }
        )
    )
    service, _absorbed = health_service(store)
    _, workflow = service.ingest(
        finding(
            category=NodeHealthCategory.NCCL,
            diagnostic_parameters={
                "diagnostic_reason": "EFA_TRAFFIC_HUNG_SUSPECTED",
                "capture_process_state": True,
            },
        )
    )
    assert workflow.status is WorkflowStatus.BLOCKED
    assert "NCCL hung triage DAG is missing required steps" in workflow.blocked_reasons
    assert WorkflowOperation.COLLECT_HUNG_TRIAGE not in [
        step.operation for step in workflow.official_steps
    ]


@pytest.mark.parametrize("expected", [None, [], "unparsed"])
def test_legacy_unusable_resource_count_keeps_the_documented_minimum_validation(
    expected,
):
    service, _absorbed = health_service(memory_store())
    _, workflow = service.ingest(
        finding(
            recommended_action=RecoveryAction.RESTART_EFA_DEVICE_PLUGIN,
            category=NodeHealthCategory.RDMA,
            metric_name="efa_inventory_mismatch",
            diagnostic_parameters={"expected_count": expected},
        )
    )
    plugin = next(
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN
    )
    assert plugin.parameters["expected_count"] == 1
    assert (
        plugin.parameters["failure_escalation_action"]
        == RecoveryAction.REBOOT_NODE.value
    )
