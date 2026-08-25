from __future__ import annotations

from pathlib import Path

from gpu_fault.models import WorkflowOperation
from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts/perf/seed_regional_action_workflows.py"
MODULE = lazy_script_module("seed_regional_action_workflows", PATH)


def test_action_capacity_workflow_is_multinode_dag() -> None:
    nodes = ["node-a", "node-b", "node-c", "node-d"]
    steps = MODULE.workflow_steps(
        nodes=nodes, workload_id="training/PyTorchJob/test", run_id="action-test"
    )

    assert len(steps) == 10
    assert all(step.node_ids == nodes for step in steps)
    assert steps[3].branch_id == "diagnostics"
    assert steps[4].branch_id == "mutation"
    assert steps[5].depends_on_step_indexes == [3, 4]
    assert [step.operation for step in steps] == [
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.CHECKPOINT_WORKLOADS,
        WorkflowOperation.STOP_WORKLOADS,
        WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESTORE_GPU_SERVICES,
        WorkflowOperation.RESTART_NODE,
        WorkflowOperation.RESTORE_SCHEDULING,
    ]
