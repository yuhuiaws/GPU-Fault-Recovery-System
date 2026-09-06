"""An SXID merge widens only the steps that have not run (F-B6 (1)(6)).

``merge_sxid_step_scope`` rewrote every RESET / bundle step's per-node GPU
and SXID mappings, completed ones included, so the ledger claimed a reset
covered a GPU it never touched. Pending steps keep their other parameters
(preserving merge, not a rewrite).
"""

from __future__ import annotations

from datetime import datetime, timezone

from gpu_fault.models import WorkflowOperation, WorkflowStatus, WorkflowStepStatus
from gpu_fault.orchestration.ingest.sxid import merge_sxid_step_scope
from gpu_fault.policy import SxidClassification, SxidEvent, SxidLinkScope
from tests._builders import workflow_request, workflow_step, workflow_step_execution

RESET = WorkflowOperation.RESET_GPU
BUNDLE = WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE


def _event() -> SxidEvent:
    return SxidEvent(
        event_id="sxid-b",
        cluster_id="cluster-a",
        node_id="node-b",
        observed_at=datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc),
        sxid=11001,
        classification=SxidClassification.FATAL,
        classification_source="NVIDIA_FABRIC_MANAGER_CATALOG",
        link_scope=SxidLinkScope.TRUNK,
        link_scope_source="TRUSTED_NVSWITCH_TOPOLOGY",
        product="H200",
        fabric_partition="cluster-a/node-b/local-nvswitch",
        participating_gpu_uuids=["GPU-b"],
        runtime_profile_version="simulated-v1",
    )


def test_completed_and_in_flight_steps_keep_their_scope_pending_ones_widen():
    workflow = workflow_request(
        "wf",
        "inc",
        status=WorkflowStatus.RUNNING,
        official_steps=[
            workflow_step(
                RESET,
                node_ids=["node-a"],
                gpu_uuids=["GPU-a"],
                parameters={"gpu_uuids_by_node": {"node-a": ["GPU-a"]}, "keep": "me"},
            ),
            workflow_step(
                BUNDLE,
                node_ids=["node-a"],
                parameters={"gpu_uuids_by_node": {"node-a": ["GPU-a"]}, "bundle": True},
                depends_on_step_indexes=[0],
            ),
            workflow_step(
                RESET,
                node_ids=["node-a"],
                gpu_uuids=["GPU-a"],
                parameters={"gpu_uuids_by_node": {"node-a": ["GPU-a"]}},
                depends_on_step_indexes=[1],
            ),
        ],
        completed_step_indexes=[0],
        step_executions=[
            workflow_step_execution(0, RESET),
            workflow_step_execution(1, BUNDLE, WorkflowStepStatus.WAITING),
        ],
    )

    merged = merge_sxid_step_scope(workflow, _event())

    assert (
        merged.official_steps[0].parameters == workflow.official_steps[0].parameters
    ), "a completed reset is history"
    assert (
        merged.official_steps[1].parameters == workflow.official_steps[1].parameters
    ), "an in-flight bundle is the agent's command"
    pending = merged.official_steps[2].parameters
    assert pending["gpu_uuids_by_node"] == {"node-a": ["GPU-a"], "node-b": ["GPU-b"]}
    assert pending["sxids_by_node"] == {"node-b": [11001]}
    assert pending["fabric_partitions_by_node"] == {
        "node-b": "cluster-a/node-b/local-nvswitch"
    }
