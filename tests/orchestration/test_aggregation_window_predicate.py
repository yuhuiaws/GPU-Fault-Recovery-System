"""Which new workflows carry an aggregation window, and why (性能 5).

``_aggregation_deadlines`` stamps ``not_before = now + window`` so a sibling
event arriving a moment later can merge into the still-PENDING, unclaimed row.
The hypothesis under review was that the window is a pure delay for chains
whose official steps hold no ``merge_intent`` / ``multi_node_barrier``
operation (FREEZE_EVIDENCE -> CHECK_MECHANICALS, -> RESTART_FABRIC_MANAGER,
diagnostic-only chains) and could be skipped for them.

Reading the merge code says otherwise, and these tests pin what it says:

* A window is applied by construction only by the group-keyed families
  (node-scope, attempt, SXID, replacement). The independent-incident path
  never calls ``_aggregation_deadlines``, so the idle-node CHECK_MECHANICALS /
  RESTART_FM / diagnostic-only chains already dispatch with ``not_before``
  unset -- there is nothing to skip.
* Every windowed row *is* reachable by a later event through its group key,
  and ``WorkflowMergeService.disposition`` ABSORBs / WIDENs any compatible
  candidate into a mutable row regardless of step shape. ``merge_intent`` is
  an input to the generation fence, not a merge gate. So the shape-based skip
  would have split a same-attempt CHECK_MECHANICALS or RUN_DCGM_DIAGNOSTIC
  pair into two workflows -- correctness over latency, the window stays.

No production code changed; the predicate is "windowed iff created by a
group-keyed family", which the code already implements.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.host_health import NodeHealthCategory, NodeHealthFinding
from gpu_fault.models import (
    RecoveryAction,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkloadState,
)
from gpu_fault.policy import (
    GpuFaultPolicyEngine,
    SxidClassification,
    SxidEvent,
    SxidLinkScope,
    XidEvent,
)
from gpu_fault.store.shared.workflow_scan import dispatch_eligible_at
from tests._builders import build_context, build_sxid_event, node_health_finding

from ._support import NOW, ApplicationContext, _save_attempt, event, ingest

WORKLOAD_IDS = ["training/job/job-a"]
BASE_WINDOW = timedelta(seconds=5)
# max window (30) + processor drain wait (30), both defaults.
MAX_DEADLINE_AFTER_WINDOW = timedelta(seconds=55)


def _active_event(xid: int, *, event_id: str, node_id: str = "node-a") -> XidEvent:
    return event(
        xid,
        event_id=event_id,
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=WORKLOAD_IDS,
    ).model_copy(update={"node_id": node_id, "gpu_uuid": f"GPU-{node_id[-1]}"})


def _health_finding(
    event_id: str,
    action: RecoveryAction,
    *,
    node_id: str = "node-a",
    workload_state: WorkloadState = WorkloadState.IDLE,
) -> NodeHealthFinding:
    return node_health_finding(
        f"finding-{event_id}",
        event_id,
        node_id=node_id,
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity="critical",
        metric_name="dcgm_health",
        reason="dcgm health check failed",
        recommended_action=action,
        runtime_profile_version="simulated-v1",
        workload_state=workload_state,
        affected_workload_ids=(
            WORKLOAD_IDS if workload_state is WorkloadState.ACTIVE else []
        ),
        gpu_uuids=[f"GPU-{node_id[-1]}"],
        policy_source="DCGM_HEALTH",
    )


def _trunk_fatal_sxid(event_id: str) -> SxidEvent:
    return build_sxid_event(
        event_id,
        NOW + timedelta(seconds=1),
        11001,
        SxidClassification.FATAL,
        "NVIDIA_FABRIC_MANAGER_CATALOG",
        node_id="node-a",
        link_scope=SxidLinkScope.TRUNK,
        link_scope_source="TRUSTED_NVSWITCH_TOPOLOGY",
        product="H200",
        fabric_partition="cluster-a/node-a/local-nvswitch",
        participating_gpu_uuids=["GPU-a"],
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=WORKLOAD_IDS,
    )


def _operations(workflow: WorkflowRequest) -> list[WorkflowOperation]:
    return [step.operation for step in workflow.official_steps]


def _assert_no_window(workflow: WorkflowRequest) -> None:
    assert workflow.status is WorkflowStatus.PENDING
    assert workflow.not_before is None
    assert workflow.aggregation_max_deadline is None
    # The dispatcher's sort key and its ``not_before`` hold both read the
    # row as dispatchable from the moment it was created.
    assert dispatch_eligible_at(workflow) == workflow.created_at


def _assert_window(workflow: WorkflowRequest) -> None:
    assert workflow.status is WorkflowStatus.PENDING
    assert workflow.not_before is not None
    assert workflow.aggregation_max_deadline is not None
    # Single-node row: multiplier 1, so the base window applies. The family's
    # ``now`` and the builder's ``created_at`` are separate clock reads.
    assert (
        BASE_WINDOW - timedelta(seconds=1)
        <= workflow.not_before - workflow.created_at
        <= BASE_WINDOW + timedelta(seconds=1)
    )
    # Both deadlines come from one ``_aggregation_deadlines`` call, so the
    # gap between them is exact: every family used the shared arithmetic.
    assert (
        workflow.aggregation_max_deadline - workflow.not_before
        == MAX_DEADLINE_AFTER_WINDOW
    )


# --- the chains the hypothesis targeted already dispatch immediately -------


def test_idle_check_mechanicals_has_no_aggregation_window(
    context: ApplicationContext,
) -> None:
    decision, _, workflow = ingest(context, event(54, event_id="xid54-idle"))

    assert decision.official_action == "CHECK_MECHANICALS"
    assert _operations(workflow) == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.CHECK_MECHANICALS,
    ]
    _assert_no_window(workflow)


def test_idle_solo_restart_fm_has_no_aggregation_window(
    context: ApplicationContext,
) -> None:
    decision, _, workflow = ingest(context, event(45, event_id="xid45-idle"))

    assert decision.official_action == "RESTART_FM"
    assert _operations(workflow) == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.RESTART_FABRIC_MANAGER,
    ]
    _assert_no_window(workflow)


def test_idle_diagnostic_only_finding_has_no_aggregation_window(
    context: ApplicationContext,
) -> None:
    _, workflow = context.orchestrator.ingest_node_health(
        _health_finding("dcgm-idle", RecoveryAction.RUN_DIAGNOSTICS)
    )

    assert workflow is not None
    assert _operations(workflow) == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.RUN_DCGM_DIAGNOSTIC,
        WorkflowOperation.VALIDATE_GPU,
    ]
    _assert_no_window(workflow)


# --- every group-keyed family stamps the shared window ---------------------


def test_node_scoped_reset_on_idle_node_keeps_the_window(
    context: ApplicationContext,
) -> None:
    decision, _, workflow = ingest(context, event(62, event_id="xid62-idle"))

    assert decision.action is RecoveryAction.RESET_GPU
    assert WorkflowOperation.RESET_GPU in _operations(workflow)
    _assert_window(workflow)


def test_grouped_restart_app_with_active_workload_keeps_the_window(
    context: ApplicationContext,
) -> None:
    _save_attempt(context, ("node-a", "node-b"))
    decision, _, workflow = ingest(context, _active_event(31, event_id="xid31-active"))

    assert decision.action is RecoveryAction.RESTART_WORKLOAD
    assert WorkflowOperation.RESTART_WORKLOAD in _operations(workflow)
    _assert_window(workflow)


def test_sxid_trunk_fatal_keeps_the_window(context: ApplicationContext) -> None:
    _save_attempt(context, ("node-a", "node-b"))
    sxid = _trunk_fatal_sxid("sxid-trunk")
    decision = GpuFaultPolicyEngine().evaluate_sxid(sxid)
    _, workflow = context.orchestrator.ingest(sxid, decision)

    assert workflow is not None
    assert WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES in _operations(workflow)
    _assert_window(workflow)


def test_grouped_health_and_replacement_families_keep_the_window(
    context: ApplicationContext,
) -> None:
    _save_attempt(context, ("node-a", "node-b"))
    _, grouped_health = context.orchestrator.ingest_node_health(
        _health_finding(
            "dcgm-active", RecoveryAction.RESET_GPU, workload_state=WorkloadState.ACTIVE
        )
    )
    _, replacement = context.orchestrator.ingest_node_health(
        _health_finding(
            "replace-active",
            RecoveryAction.REPLACE_NODE,
            node_id="node-b",
            workload_state=WorkloadState.ACTIVE,
        )
    )

    assert grouped_health is not None and replacement is not None
    assert WorkflowOperation.RESET_GPU in _operations(grouped_health)
    assert WorkflowOperation.REPLACE_NODE in _operations(replacement)
    _assert_window(grouped_health)
    _assert_window(replacement)


def test_window_disabled_globally_still_yields_no_window(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_MULTI_NODE_AGGREGATION_WINDOW_SECONDS", "0")
    context = build_context()

    _, _, workflow = ingest(context, event(62, event_id="xid62-window-off"))

    assert WorkflowOperation.RESET_GPU in _operations(workflow)
    _assert_no_window(workflow)


# --- why the shape-based skip was rejected ---------------------------------


@pytest.mark.parametrize(
    ("xid", "operation"),
    [
        (54, WorkflowOperation.CHECK_MECHANICALS),
        (45, WorkflowOperation.RESTART_FABRIC_MANAGER),
    ],
)
def test_grouped_shape_without_barrier_is_still_a_merge_target(
    context: ApplicationContext, xid: int, operation: WorkflowOperation
) -> None:
    """A same-attempt sibling joins the windowed row and widens every step.

    Neither chain holds a ``multi_node_barrier`` operation, yet the attempt
    group key finds the row and ``disposition`` ABSORBs the sibling while
    the row is mutable. Skipping the window would have let the dispatcher
    claim the row first and split the pair into two workflows.
    """

    _save_attempt(context, ("node-a", "node-b"))
    _, first_incident, first = ingest(
        context, _active_event(xid, event_id=f"xid{xid}-a", node_id="node-a")
    )
    _assert_window(first)

    _, incident, merged = ingest(
        context, _active_event(xid, event_id=f"xid{xid}-b", node_id="node-b")
    )

    assert merged.request_id == first.request_id
    assert incident.incident_id == first_incident.incident_id
    assert incident.node_ids == ["node-a", "node-b"]
    assert len(context.store.list_workflows()) == 1
    assert _operations(merged) == [WorkflowOperation.FREEZE_EVIDENCE, operation]
    assert all(
        step.node_ids == ["node-a", "node-b"] for step in merged.official_steps
    ), "the merged workflow must widen every official step to both nodes"


def test_grouped_diagnostic_only_chain_is_still_a_merge_target(
    context: ApplicationContext,
) -> None:
    """RUN_DCGM_DIAGNOSTIC (merge_intent, no barrier) merges the same way."""

    _save_attempt(context, ("node-a", "node-b"))
    _, first = context.orchestrator.ingest_node_health(
        _health_finding(
            "dcgm-a",
            RecoveryAction.RUN_DIAGNOSTICS,
            node_id="node-a",
            workload_state=WorkloadState.ACTIVE,
        )
    )
    assert first is not None
    _assert_window(first)

    incident, merged = context.orchestrator.ingest_node_health(
        _health_finding(
            "dcgm-b",
            RecoveryAction.RUN_DIAGNOSTICS,
            node_id="node-b",
            workload_state=WorkloadState.ACTIVE,
        )
    )

    assert merged is not None
    assert merged.request_id == first.request_id
    assert incident.node_ids == ["node-a", "node-b"]
    assert len(context.store.list_workflows()) == 1
    diagnostic = next(
        step
        for step in merged.official_steps
        if step.operation is WorkflowOperation.RUN_DCGM_DIAGNOSTIC
    )
    assert diagnostic.node_ids == ["node-a", "node-b"]
    assert diagnostic.gpu_uuids == ["GPU-a", "GPU-b"]
