"""The passive survey and two bounded active diagnostic waves."""

from typing import Any
import time

from gpu_fault.models import WorkflowOperation as Op


def hung_dag(
    steps: list[Any], finding: Any, context: Any, sample_nodes: Any
) -> tuple[list[Any], list[str]]:
    by_op = {step.operation: step for step in steps}
    required = {
        Op.FREEZE_EVIDENCE,
        Op.COLLECT_HUNG_SUMMARY,
        Op.COLLECT_HUNG_TRIAGE,
        Op.COLLECT_DIAGNOSTIC_BUNDLE,
        Op.VALIDATE_FABRIC,
    }
    if not required <= by_op.keys():
        return steps, ["NCCL hung triage DAG is missing required steps"]
    targets, _ = sample_nodes(
        context.target_node_ids,
        reporting_node_id=finding.node_id,
        lowest_rank_by_node=context.lowest_rank_by_node,
    )
    budget = len(targets)
    completed_epoch = int(time.time() // 30) - 1
    common = {
        **context.diagnostic_parameters,
        "snapshot_epoch": finding.event_id,
        "hung_diagnostic_node_budget": budget,
        "reporting_node_id": finding.node_id,
        "communication_history_epochs": [completed_epoch - 1, completed_epoch],
    }
    freeze = by_op[Op.FREEZE_EVIDENCE].model_copy(
        update={
            "depends_on_step_indexes": [],
            "parameters": {
                **by_op[Op.FREEZE_EVIDENCE].parameters,
                "hung_task_summary": context.task_summary,
            },
        }
    )
    summary = by_op[Op.COLLECT_HUNG_SUMMARY].model_copy(
        update={
            "depends_on_step_indexes": [0],
            "node_ids": context.target_node_ids,
            "parameters": {
                **by_op[Op.COLLECT_HUNG_SUMMARY].parameters,
                **common,
                "capture_process_state": False,
                "survey_pass": "BASELINE",
            },
        }
    )
    triage = by_op[Op.COLLECT_HUNG_TRIAGE].model_copy(
        update={
            "depends_on_step_indexes": [1],
            "node_ids": [finding.node_id],
            "parameters": {
                **by_op[Op.COLLECT_HUNG_TRIAGE].parameters,
                **common,
                "hung_wave": 1,
                "hung_target_pending": True,
                "triage_timeout_seconds": 20,
                "triage_fr_sample_interval_seconds": 2,
                "require_workload_identity": True,
                "expand_python_cgroup_processes": False,
            },
        }
    )
    expansion = triage.model_copy(
        update={
            "depends_on_step_indexes": [2],
            "parameters": {**triage.parameters, "hung_wave": 2},
        }
    )
    bundle = by_op[Op.COLLECT_DIAGNOSTIC_BUNDLE].model_copy(
        update={
            "depends_on_step_indexes": [2, 3],
            "node_ids": targets,
            "parameters": {
                **by_op[Op.COLLECT_DIAGNOSTIC_BUNDLE].parameters,
                **common,
                "capture_process_state": False,
                "disable_strace": True,
                "hung_triage_target_pending": True,
            },
        }
    )
    validation = by_op[Op.VALIDATE_FABRIC].model_copy(
        update={
            "depends_on_step_indexes": [1],
            "parameters": {
                **by_op[Op.VALIDATE_FABRIC].parameters,
                "require_task_fabric_scope": True,
                "training_fabric_by_node": common.get("training_fabric_by_node", {}),
            },
        }
    )
    result = [freeze, summary, triage, expansion, bundle, validation]
    if Op.ESCALATE_SUPPORT in by_op:
        result.append(
            by_op[Op.ESCALATE_SUPPORT].model_copy(
                update={"depends_on_step_indexes": [4, 5]}
            )
        )
    # The second sweep starts only after every baseline response/timeout is
    # persisted. Deep collectors remain independently budgeted.
    result = [
        step.model_copy(
            update={
                "depends_on_step_indexes": [
                    dependency + 1
                    if dependency >= 2
                    else 2
                    if dependency == 1
                    else dependency
                    for dependency in step.depends_on_step_indexes
                ]
            }
        )
        if index >= 2
        else step
        for index, step in enumerate(result)
    ]
    recheck = summary.model_copy(
        update={
            "depends_on_step_indexes": [1],
            "parameters": {**summary.parameters, "survey_pass": "RECHECK"},
        }
    )
    result.insert(2, recheck)
    return result, []
