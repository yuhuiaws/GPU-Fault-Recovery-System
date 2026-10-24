"""Software E2E: real API, persisted state, planning and workflow execution."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from gpu_fault.host_health import HostMetricSample, HostTelemetryBatch
from gpu_fault.models import IncidentState, WorkflowStatus
from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.orchestration.families.evidence import EvidenceOperationService
from tests._builders import (
    attempt_observation,
    build_context,
    container_observation,
    node_health_finding,
)
from tests.execution._health_policy_e2e_support import JOB
from tests.execution._raw_health_e2e_support import RawHealthRig


def test_efa_detection_to_evidence_and_operator_review_never_recovers_on_fabric_alone(
    tmp_path, monkeypatch
):
    rig = RawHealthRig(tmp_path, monkeypatch)
    try:

        def no_training_progress(*args, **kwargs):
            pytest.fail(
                "NCCL detection, planning and completion must not query train steps"
            )

        monkeypatch.setattr(
            rig.store, "list_training_progress_states", no_training_progress
        )

        def traffic(rate):
            batch = HostTelemetryBatch(
                cluster_id="cluster-a",
                node_id="node-a",
                observed_at=rig.clock,
                received_at=rig.clock,
                runtime_profile_version="simulated-v1",
                workload_state="ACTIVE",
                affected_workload_ids=[JOB],
                samples=[
                    HostMetricSample(name="efa_traffic_bytes_per_second", value=rate),
                    HostMetricSample(name="efa_traffic_observation_valid", value=1),
                ],
            )
            return rig.post(
                "/v1/collector-events/host-telemetry", batch.model_dump(mode="json")
            )

        traffic(2_000_000)
        rig.advance(130)
        traffic(0)
        for _ in range(10):
            rig.advance(60)
            traffic(0)
        workflow = next(
            item
            for item in rig.store.list_workflows()
            if Op.COLLECT_HUNG_TRIAGE
            in {step.operation for step in item.official_steps}
        )
        rig.host()
        rig.gpu({"gpu_temperature_c": 40})
        for _ in range(5):
            rig.execute(workflow)
        saved = rig.store.get_workflow(workflow.request_id)
        assert saved.status is WorkflowStatus.RUNNING, (
            "unscoped network telemetry is insufficient"
        )
        rig.bind_task_fabric(workflow)
        rig.execute(workflow)
        saved = rig.store.get_workflow(workflow.request_id)
        assert saved.status is WorkflowStatus.SUCCEEDED, saved.step_executions
        incident = rig.store.get_incident(saved.incident_id)
        assert incident.state is IncidentState.ESCALATED, incident
        assert Op.ESCALATE_SUPPORT in rig.boundary.calls, rig.boundary.calls
        assert not {
            Op.STOP_WORKLOADS,
            Op.RESET_GPU,
            Op.RESTART_NODE,
            Op.RESTART_WORKLOAD,
        } & {step.operation for step in saved.official_steps}, saved
        evidence = next(
            item
            for item in saved.step_executions
            if item.operation is Op.FREEZE_EVIDENCE
        )
        summary = evidence.details["hung_task_summary"]
        assert summary["expected_node_count"] == 1, summary
        assert summary["allocated_container_count"] == 1, summary
        assert summary["rank_inventory_complete"] is False
    finally:
        rig.store.close()


def test_one_recovered_reporter_cannot_close_a_multi_rank_hang(tmp_path, monkeypatch):
    rig = RawHealthRig(tmp_path, monkeypatch)
    try:

        def publish(steps):
            allocation = attempt_observation(
                "train",
                "attempt-a",
                rig.clock,
                expected_critical_ranks=2,
                workload_ids=[JOB],
                containers=[
                    container_observation(
                        f"pod-{rank}",
                        "trainer",
                        rank,
                        f"node-{rank}",
                        gpu_uuids=[f"GPU-{rank}"],
                    )
                    for rank in range(2)
                ],
            )
            rig.store.save_attempt_observation(allocation)
            for rank, step in enumerate(steps):
                rig.post(
                    "/v1/training-progress",
                    {
                        "cluster_id": "cluster-a",
                        "attempt_id": "attempt-a",
                        "rank": rank,
                        "node_id": f"node-{rank}",
                        "step": step,
                        "observed_at": rig.clock.isoformat(),
                        "loss": 1,
                    },
                )

        publish([10, 10])
        rig.advance(121)
        publish([10, 10])
        result = rig.post("/v1/training-health/cluster-a/scan", {})
        assert len(result["findings"]) == 2, result
        workflows = rig.store.list_workflows()
        assert len(workflows) == 2, workflows
        assert all(
            Op.COLLECT_HUNG_SUMMARY
            not in {step.operation for step in item.official_steps}
            for item in workflows
        ), "application reporter monitoring is separate from NCCL diagnosis"
        workflow = next(
            item
            for item in workflows
            if item.official_steps[-1].parameters["health_recovery_contract"]["rank"]
            == 0
        )
        for _ in range(3):
            rig.execute(workflow)
        rig.advance(1)
        publish([11, 10])
        assert rig.execute(workflow).status is WorkflowStatus.RUNNING, (
            "rank 1 remains stalled"
        )
        rig.advance(1)
        publish([11, 11])
        for _ in range(3):
            outcome = rig.execute(workflow)
        assert outcome.status is WorkflowStatus.SUCCEEDED, outcome
    finally:
        rig.store.close()


def test_step_stall_does_not_start_nccl_survey_or_rank_attachments(
    tmp_path, monkeypatch
):
    rig = RawHealthRig(tmp_path, monkeypatch)
    try:

        def heartbeat(step):
            rig.post(
                "/v1/training-progress",
                {
                    "cluster_id": "cluster-a",
                    "attempt_id": "attempt-a",
                    "rank": 0,
                    "node_id": "node-a",
                    "step": step,
                    "observed_at": rig.clock.isoformat(),
                },
            )

        heartbeat(10)
        rig.advance(121)
        heartbeat(10)
        rig.post("/v1/training-health/cluster-a/scan", {})
        [workflow] = rig.store.list_workflows()
        assert not {Op.COLLECT_HUNG_SUMMARY, Op.COLLECT_HUNG_TRIAGE} & {
            step.operation for step in workflow.official_steps
        }, workflow
        assert all(
            not step.parameters.get("capture_process_state")
            for step in workflow.official_steps
        ), workflow
    finally:
        rig.store.close()


@pytest.mark.parametrize(
    "maximum,expected", [(1, 1), (8, 8), (0, 32), (128, 128), (10000, 256)]
)
def test_ten_thousand_node_sampling_is_bounded(maximum, expected, monkeypatch):
    monkeypatch.setenv("GPU_FAULT_HUNG_TRIAGE_MAX_NODES", str(maximum))
    nodes = [f"node-{index:05d}" for index in range(10_000)]
    sampled, omitted = EvidenceOperationService.sample_hung_triage_nodes(
        nodes, reporting_node_id=nodes[-1], lowest_rank_by_node={nodes[0]: 0}
    )
    assert len(sampled) == expected, sampled
    assert nodes[-1] in sampled, sampled
    assert set(sampled).isdisjoint(omitted), sampled
    assert sorted(sampled + omitted) == nodes


@pytest.mark.parametrize("window", [0, 15])
def test_simultaneous_nodes_share_one_attempt_diagnostic_even_without_aggregation_window(
    window, monkeypatch
):
    monkeypatch.setenv("GPU_FAULT_MULTI_NODE_AGGREGATION_WINDOW_SECONDS", str(window))
    context = build_context()
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    nodes = [f"node-{index:04d}" for index in range(1250)]
    observation = attempt_observation(
        "job-a",
        "attempt-a",
        now,
        expected_critical_ranks=len(nodes),
        workload_ids=["training/job/job-a"],
        containers=[
            container_observation(
                f"pod-{rank}",
                "trainer",
                rank,
                node,
                cgroup_path=f"/job/rank-{rank}",
                gpu_uuids=[f"GPU-{rank}"],
            )
            for rank, node in enumerate(nodes)
        ],
    )
    context.store.save_attempt_observation(observation)

    def submit(index):
        return context.orchestrator.ingest_node_health(
            node_health_finding(
                f"hung-{index}",
                f"hung-{index}",
                node_id=nodes[index],
                observed_at=now,
                category="RDMA",
                severity="critical",
                recommended_action="RUN_DIAGNOSTICS",
                reason="sustained zero RDMA traffic",
                workload_state="ACTIVE",
                affected_workload_ids=["training/job/job-a"],
                job_id="job-a",
                attempt_id="attempt-a",
                runtime_profile_version="simulated-v1",
                diagnostic_parameters={
                    "diagnostic_reason": "EFA_TRAFFIC_HUNG_SUSPECTED",
                    "capture_process_state": True,
                },
            )
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(submit, range(16)))
    assert len({workflow.request_id for _, workflow in results}) == 1, results
    [workflow] = context.store.list_workflows()
    triage = next(
        step
        for step in workflow.official_steps
        if step.operation is Op.COLLECT_HUNG_TRIAGE
    )
    bundle = next(
        step
        for step in workflow.official_steps
        if step.operation is Op.COLLECT_DIAGNOSTIC_BUNDLE
    )
    fabric = next(
        step for step in workflow.official_steps if step.operation is Op.VALIDATE_FABRIC
    )
    assert triage.parameters["hung_target_pending"] is True
    survey = next(
        step
        for step in workflow.official_steps
        if step.operation is Op.COLLECT_HUNG_SUMMARY
    )
    assert len(survey.node_ids) == 1250
    assert len(bundle.node_ids) == 8
    assert len(fabric.node_ids) == 1250
    summary = workflow.official_steps[0].parameters["hung_task_summary"]
    assert summary["expected_node_count"] == 1250, summary
    assert summary["allocated_container_count"] == 1250, summary
    assert summary["allocation_rows_truncated"] is True
    assert len(summary["allocation_rows"]) == 256
    assert all(
        "hung_task_summary" not in step.parameters
        for step in workflow.official_steps[1:]
    ), "large summary is not copied into every node command"
