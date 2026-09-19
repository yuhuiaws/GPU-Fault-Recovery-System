from __future__ import annotations

from datetime import timedelta

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


def test_grouped_replacement_binds_finding_named_attempt_over_fresher_sibling(
    context: ApplicationContext,
) -> None:
    # A workload *name* (path) is reused across attempts: every DESTR-008
    # sub-scenario submits ``gpu-fault-single-node-warm-spare`` under a fresh
    # attempt id. Here two attempts share one path; the STALE sibling
    # observation is *fresher* on ``observed_at`` than the attempt the finding
    # actually names. The incident must bind the attempt the finding names --
    # not ``max(observed_at)`` -- or STOP_WORKLOADS fail-closes with
    # STOP_OWNERSHIP_DRIFT at the executor pre-submit boundary.
    path = "training/pytorchjob/shared-name"
    context.store.save_attempt_observation(
        attempt_observation(
            "job-old",
            "job-old-a001",
            NOW,  # fresher -> would win max(observed_at)
            containers=[
                container_observation(
                    "pod-old", "worker-a", 0, "node-a", gpu_uuids=["GPU-old"]
                )
            ],
            workload_ids=[path],
        )
    )
    context.store.save_attempt_observation(
        attempt_observation(
            "job-new",
            "job-new-a001",
            NOW - timedelta(minutes=1),  # older, but the finding names it
            containers=[
                container_observation(
                    "pod-new", "worker-a", 0, "node-a", gpu_uuids=["GPU-new"]
                )
            ],
            workload_ids=[path],
        )
    )
    finding = node_health_finding(
        "finding-shared-name",
        "shared-name-event",
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity="critical",
        reason="synthetic warm-spare replacement test",
        recommended_action=RecoveryAction.REPLACE_NODE,
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=[path],
        job_id="job-new",
        attempt_id="job-new-a001",
        diagnostic_parameters={"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"},
    )

    incident, _ = context.orchestrator.ingest_node_health(finding)

    assert incident.attempt_id == "job-new-a001"
    assert incident.job_id == "job-new"
