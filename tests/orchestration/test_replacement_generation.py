"""A replacement workflow never carries a generation below its incident's.

FINAL-建议汇总 F-B9 (P0-59B / P1-55E). ``node_lifecycle`` wrote
``fencing_token=1`` for every incident and workflow it emitted. Once an
incident had advanced to generation 4, the brand-new recovery workflow the
family created for it sat at generation 1 -- exactly the shape the
abandoned-generation sweep terminalizes as "left behind".
"""

from __future__ import annotations

from tests._builders import (
    attempt_observation,
    container_observation,
    copy_model,
    node_health_finding,
)
from tests.orchestration._support import (
    NOW,
    ApplicationContext,
    NodeHealthCategory,
    RecoveryAction,
    WorkloadState,
)


def _finding(finding_id: str):
    return node_health_finding(
        finding_id,
        f"event-{finding_id}",
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity="critical",
        reason="synthetic generation test",
        recommended_action=RecoveryAction.REPLACE_NODE,
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/pytorchjob/job-a"],
        diagnostic_parameters={"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"},
    )


def test_replacement_keeps_the_incidents_generation(context: ApplicationContext):
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
    incident, workflow = context.orchestrator.ingest_node_health(_finding("first"))
    # The incident advances three generations (re-plans by other families).
    context.store.save_incident(copy_model(incident, fencing_token=4))
    context.store.save_workflow(
        copy_model(workflow, fencing_token=4), expected=workflow
    )

    merged_incident, merged_workflow = context.orchestrator.ingest_node_health(
        _finding("second")
    )

    assert merged_incident.fencing_token >= 4
    assert merged_workflow.fencing_token >= 4
    assert merged_workflow.fencing_token == merged_incident.fencing_token
