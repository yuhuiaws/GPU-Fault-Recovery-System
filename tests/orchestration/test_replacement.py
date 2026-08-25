from __future__ import annotations

from tests._builders import (
    attempt_observation,
    container_observation,
    node_health_finding,
)

from ._support import (
    NOW,
    ApplicationContext,
    NodeHealthCategory,
    RecoveryAction,
    WorkflowOperation,
    WorkloadState,
)


def test_grouped_replacement_preserves_warm_spare_strategy(
    context: ApplicationContext,
) -> None:
    context.store.save_attempt_observation(
        attempt_observation(
            "job-a",
            "job-a-a001",
            NOW,
            containers=[
                container_observation(
                    "pod-a", "worker-a", 0, "node-a", gpu_uuids=["GPU-a"]
                )
            ],
            workload_ids=["training/pytorchjob/job-a"],
        )
    )
    finding = node_health_finding(
        "finding-grouped-warm-spare",
        "grouped-warm-spare",
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity="critical",
        reason="synthetic warm-spare replacement test",
        recommended_action=RecoveryAction.REPLACE_NODE,
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/pytorchjob/job-a"],
        diagnostic_parameters={"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"},
    )

    _, workflow = context.orchestrator.ingest_node_health(finding)

    replace = next(
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.REPLACE_NODE
    )
    assert replace.parameters == {"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"}
    assert replace.workload_ids == ["training/pytorchjob/job-a"]
