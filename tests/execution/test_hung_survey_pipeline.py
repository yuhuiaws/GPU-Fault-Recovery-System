from datetime import datetime, timezone
from subprocess import CompletedProcess

import pytest

from gpu_fault.adapters.evidence import ControlPlaneEvidenceAdapter
from gpu_fault.adapters.node_action.adapter import NodeActionWorkflowAdapter
from gpu_fault.execution import WorkflowExecutionRequest, WorkflowStepOutcome
from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.models import WorkflowStatus
from gpu_fault.node_agent import NodeActionExecutor, NodeActionLedger
from tests._builders import (
    active_workflow_executor,
    attempt_observation,
    build_context,
    container_observation,
    node_health_finding,
)
from tests.node_agent._cov95_runtime_hung import process
from tests.node_agent._support import SECRET
from tests.node_agent.test_hung_collection_hardening import stat_text


@pytest.mark.parametrize(
    "node_count,ranks_per_node,budget,rank_environment",
    [(9, 1, 8, True), (128, 8, 128, True), (128, 8, 128, False)],
    ids=["small", "1024-gpus-no-step-no-fr", "1024-gpus-no-rank-no-step-no-fr"],
)
def test_real_node_summary_drives_two_disjoint_active_waves_and_bounded_bundle(
    tmp_path, monkeypatch, node_count, ranks_per_node, budget, rank_environment
):
    monkeypatch.setenv("GPU_FAULT_HUNG_TRIAGE_MAX_NODES", str(budget))
    context = build_context()
    now = datetime.now(timezone.utc)
    nodes = [f"node-{index}" for index in range(node_count)]
    world = node_count * ranks_per_node
    context.store.save_attempt_observation(
        attempt_observation(
            "job-a",
            "attempt-a",
            now,
            expected_critical_ranks=node_count,
            workload_ids=["training/job/job-a"],
            containers=[
                container_observation(
                    f"pod-{index}",
                    "trainer",
                    index,
                    node,
                    cgroup_path="/training/job",
                    gpu_uuids=[
                        f"GPU-{index}-{local}" for local in range(ranks_per_node)
                    ],
                )
                for index, node in enumerate(nodes)
            ],
        )
    )
    incident, workflow = context.orchestrator.ingest_node_health(
        node_health_finding(
            "hung-survey",
            "hung-survey",
            node_id=nodes[0],
            observed_at=now,
            reason="RDMA idle",
            category="RDMA",
            severity="critical",
            recommended_action="RUN_DIAGNOSTICS",
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
    agents = {}
    ledgers = []
    operations = {
        Op.COLLECT_HUNG_SUMMARY,
        Op.COLLECT_HUNG_TRIAGE,
        Op.COLLECT_DIAGNOSTIC_BUNDLE,
    }

    def runner_for(index):
        def runner(argv, **kwargs):
            text = ""
            if "--query-compute-apps=pid,gpu_uuid,process_name" in argv:
                text = "\n".join(
                    f"{100 + local},GPU-{index}-{local},python"
                    for local in range(ranks_per_node)
                )
            elif "--query-compute-apps=pid,gpu_uuid" in argv:
                text = "\n".join(
                    f"{100 + local},GPU-{index}-{local}"
                    for local in range(ranks_per_node)
                )
            return CompletedProcess(argv, 0, text, "")

        return runner

    for index, node in enumerate(nodes):
        root = tmp_path / node
        for local in range(ranks_per_node):
            pid = 100 + local
            directory = process(root / "proc", pid)
            (directory / "stat").write_text(stat_text(pid, 1000))
            environment = f"RANK={index * ranks_per_node + local}\0WORLD_SIZE={world}\0"
            (directory / "environ").write_bytes(
                environment.encode() if rank_environment else b""
            )
        ledger = NodeActionLedger(str(root / "ledger.db"))
        ledgers.append(ledger)
        agents[node] = NodeActionExecutor(
            secret=SECRET,
            node_ids={node},
            allowed_operations=operations,
            reset_enabled=False,
            ledger=ledger,
            runner=runner_for(index),
            proc_root=str(root / "proc"),
            infiniband_root=str(root / "infiniband"),
            diagnostic_output_dir=str(root / "diagnostics"),
            python_stack_tool="",
            sleep=lambda seconds: None,
        )
    commands = []

    def sender(endpoint, envelope):
        commands.append(envelope.command)
        return agents[envelope.command.node_id].execute(envelope)

    class Other:
        def supports(self, step):
            return step.operation in {Op.VALIDATE_FABRIC, Op.ESCALATE_SUPPORT}

        def execute(self, ctx):
            return WorkflowStepOutcome.succeeded()

    executor = active_workflow_executor(
        context.store,
        [
            ControlPlaneEvidenceAdapter(owner="simulated-runtime"),
            NodeActionWorkflowAdapter(
                {node: "https://test.invalid" for node in nodes},
                SECRET,
                owner="simulated-runtime",
                sender=sender,
            ),
            Other(),
        ],
        set(Op),
    )
    try:
        for _ in range(12):
            result = executor.execute(
                workflow.request_id,
                WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
            )
            if result.status is not WorkflowStatus.RUNNING:
                break
        assert result.status is WorkflowStatus.SUCCEEDED, result
        saved = context.store.get_workflow(workflow.request_id)
        survey = next(
            item
            for item in saved.step_executions
            if item.operation is Op.COLLECT_HUNG_SUMMARY
        )
        report = survey.details["hung_survey"]
        assert report["rank_inventory_complete"] is rank_environment, report
        assert report["verified_process_count"] == world, report
        assert report["unranked_process_count"] == (0 if rank_environment else world), (
            report
        )
        assert {
            item.node_id
            for item in commands
            if item.operation is Op.COLLECT_HUNG_SUMMARY
        } == set(nodes)
        waves = [
            step
            for step in saved.official_steps
            if step.operation is Op.COLLECT_HUNG_TRIAGE
        ]
        assert (
            len(waves) == 2
            and len(waves[0].node_ids) == 4
            and len(waves[1].node_ids) == budget - 4
        ), waves
        assert set(waves[0].node_ids).isdisjoint(waves[1].node_ids), waves
        bundle = next(
            step
            for step in saved.official_steps
            if step.operation is Op.COLLECT_DIAGNOSTIC_BUNDLE
        )
        assert len(bundle.node_ids) == min(budget, 32), bundle
        assert bundle.parameters["capture_process_state"] is False, bundle
        assert len(bundle.parameters["bundle_not_collected_nodes"]) == node_count - min(
            budget, 32
        ), bundle
        active = [item for item in commands if item.operation is Op.COLLECT_HUNG_TRIAGE]
        assert len(active) == len({item.node_id for item in active}) == budget, active
        assert all(
            set(item.parameters["workload_cgroup_paths_by_node"]) == {item.node_id}
            for item in active
        ), "active fanout must not replicate the full task's process map"
        decision = bundle.parameters["hung_triage_decision"]
        assert decision["root_cause_confirmed"] is False, decision
        assert decision["active_responding_node_count"] == budget, decision
        assert decision["active_node_coverage_fraction"] == budget / node_count, (
            decision
        )
    finally:
        for ledger in ledgers:
            ledger.close()
