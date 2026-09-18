"""Mixed lifecycle findings use the same dominance and branch rules as faults."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.app import default_simulated_profile
from gpu_fault.host_health import NodeHealthCategory
from gpu_fault.models import (
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkloadState,
)
from gpu_fault.orchestration import IncidentOrchestrator
from gpu_fault.store import InMemoryStore
from tests._builders import (
    attempt_observation,
    container_observation,
    node_health_finding,
    workflow_step_execution,
)

OP = WorkflowOperation
REPLACE = RecoveryAction.REPLACE_NODE
REBOOT = RecoveryAction.REBOOT_NODE


def runtime() -> tuple[InMemoryStore, IncidentOrchestrator]:
    store = InMemoryStore()
    store.save_profile(default_simulated_profile())
    store.save_attempt_observation(
        attempt_observation(
            "job",
            "attempt",
            datetime.now(timezone.utc),
            expected_critical_ranks=3,
            workload_ids=["training/job/task"],
            containers=[
                container_observation(
                    f"pod-{node}",
                    f"worker-{node}",
                    rank,
                    node,
                    gpu_uuids=[f"GPU-{node}"],
                )
                for rank, node in enumerate(("node-a", "node-b", "node-c"))
            ],
            restart_budget=1,
        )
    )
    return store, IncidentOrchestrator(store, multi_node_aggregation_window_seconds=30)


def finding(event: str, action: RecoveryAction, node: str, *, inhibited: bool = False):
    return node_health_finding(
        f"finding-{event}",
        event,
        node_id=node,
        observed_at=datetime.now(timezone.utc),
        category=NodeHealthCategory.GPU,
        severity="critical",
        reason="approved replacement"
        if action is REPLACE
        else "GPU inventory mismatch",
        metric_name=None if action is REPLACE else "gpu_inventory_mismatch",
        recommended_action=action,
        gpu_uuids=[f"GPU-{node}"],
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/job/task"],
        policy_source=(
            "SITE_SYNTHETIC_REPLACEMENT_TEST"
            if action is REPLACE
            else "SITE_NODE_HEALTH"
        ),
        diagnostic_parameters=(
            {
                "replacement_strategy": "HEALTHY_WARM_SPARE_ONLY",
                "synthetic": True,
                **({"activation_forbidden": True} if inhibited else {}),
            }
            if action is REPLACE
            else {}
        ),
    )


def lifecycle_scope(workflow) -> dict[OP, set[str]]:
    result: dict[OP, set[str]] = {}
    for index, step in enumerate(workflow.official_steps):
        if index not in workflow.superseded_step_indexes and step.operation in {
            OP.REPLACE_NODE,
            OP.RESTART_NODE,
        }:
            result.setdefault(step.operation, set()).update(step.node_ids)
    return result


@pytest.mark.parametrize("actions", [(REPLACE, REBOOT), (REBOOT, REPLACE)])
def test_different_nodes_keep_their_distinct_required_actions(actions) -> None:
    store, orchestrator = runtime()
    _, first = orchestrator.ingest_node_health(finding("first", actions[0], "node-a"))
    incident, merged = orchestrator.ingest_node_health(
        finding("second", actions[1], "node-b")
    )

    assert first is not None and merged is not None
    assert merged.request_id == first.request_id
    assert merged.dag_enabled, "distinct node intents must remain distinct DAG branches"
    assert lifecycle_scope(merged) == {
        OP.REPLACE_NODE: {"node-a" if actions[0] is REPLACE else "node-b"},
        OP.RESTART_NODE: {"node-a" if actions[0] is REBOOT else "node-b"},
    }, "the latest event must neither downgrade nor widen the other node's action"
    assert store.get_incident_by_event("first").incident_id == incident.incident_id
    assert store.get_incident_by_event("second").incident_id == incident.incident_id
    for step in merged.official_steps:
        if step.operation is OP.STOP_WORKLOADS:
            assert (
                step.parameters["termination_initiator_incident_id"]
                == incident.incident_id
            )


@pytest.mark.parametrize("actions", [(REPLACE, REBOOT), (REBOOT, REPLACE)])
def test_same_node_retains_the_dominant_replacement(actions) -> None:
    _, orchestrator = runtime()
    _, first = orchestrator.ingest_node_health(finding("first", actions[0], "node-a"))
    _, merged = orchestrator.ingest_node_health(finding("second", actions[1], "node-a"))

    assert first is not None and merged is not None
    assert merged.request_id == first.request_id
    assert lifecycle_scope(merged) == {OP.REPLACE_NODE: {"node-a"}}
    assert merged.official_action == REPLACE.value


def test_same_action_still_aggregates_the_fault_nodes_without_a_new_workflow() -> None:
    store, orchestrator = runtime()
    _, first = orchestrator.ingest_node_health(finding("first", REPLACE, "node-a"))
    _, second = orchestrator.ingest_node_health(finding("second", REPLACE, "node-b"))

    assert first is not None and second is not None
    assert first.request_id == second.request_id
    assert lifecycle_scope(second) == {OP.REPLACE_NODE: {"node-a", "node-b"}}
    assert len(store.list_workflows()) == 1


@pytest.mark.parametrize("immutable", ["issued", "expired-window", "inhibited"])
def test_mixed_action_cannot_rewrite_an_incompatible_aggregation(
    immutable: str,
) -> None:
    store, orchestrator = runtime()
    _, first = orchestrator.ingest_node_health(
        finding("first", REPLACE, "node-a", inhibited=immutable == "inhibited")
    )
    assert first is not None
    if immutable == "issued":
        index = next(
            index
            for index, step in enumerate(first.official_steps)
            if step.operation is OP.REPLACE_NODE
        )
        first = first.model_copy(
            update={
                "status": WorkflowStatus.RUNNING,
                "step_executions": [
                    workflow_step_execution(
                        index, OP.REPLACE_NODE, "WAITING", phase="official"
                    )
                ],
            }
        )
        store.save_workflow(first)
    elif immutable == "expired-window":
        first = first.model_copy(
            update={"not_before": datetime.now(timezone.utc) - timedelta(seconds=1)}
        )
        store.save_workflow(first)
    original = first.model_copy(deep=True)

    _, later = orchestrator.ingest_node_health(finding("second", REBOOT, "node-a"))

    assert later is not None
    if immutable == "issued":
        assert later.request_id == first.request_id, (
            "the inventory counterguard may record evidence on the in-flight replacement"
        )
    else:
        assert later.request_id != first.request_id
    saved = store.get_workflow(first.request_id)
    assert saved.official_steps == original.official_steps
    assert saved.step_executions == original.step_executions
    assert lifecycle_scope(saved) == {OP.REPLACE_NODE: {"node-a"}}
