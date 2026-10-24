from copy import deepcopy

from gpu_fault.execution.hung_survey import after_summary, after_triage
from gpu_fault.hung_survey_analysis import survey_analysis
from gpu_fault.models import WorkflowOperation as Op
from tests._builders import workflow_step, workflow_step_execution
from tests.execution._hung_survey_support import passive_document
from tests.execution._support import workflow_request


def surveyed_workflow(nodes, budget):
    parameters = {
        "job_id": "job-a",
        "attempt_id": "attempt-a",
        "snapshot_epoch": "epoch",
        "reporting_node_id": nodes[0],
        "hung_diagnostic_node_budget": budget,
    }
    details = {
        "node_results": {
            node: passive_document(node, [rank], len(nodes))
            for rank, node in enumerate(nodes)
        }
    }
    steps = [
        workflow_step(Op.COLLECT_HUNG_SUMMARY, node_ids=nodes, parameters=parameters),
        *[
            workflow_step(
                Op.COLLECT_HUNG_TRIAGE,
                node_ids=[nodes[0]],
                parameters={**parameters, "hung_wave": wave},
            )
            for wave in (1, 2)
        ],
        workflow_step(
            Op.COLLECT_DIAGNOSTIC_BUNDLE,
            node_ids=nodes,
            parameters=parameters,
            depends_on_step_indexes=[1, 2],
        ),
    ]
    workflow = workflow_request(
        "workflow-no-step",
        "incident-no-step",
        "RUNNING",
        1,
        official_steps=steps,
        step_executions=[
            workflow_step_execution(0, Op.COLLECT_HUNG_SUMMARY, details=details)
        ],
    )
    return workflow, details


def test_legacy_application_step_fields_do_not_skip_nccl_diagnostics():
    nodes = [f"node-{index}" for index in range(8)]
    workflow, details = surveyed_workflow(nodes, 8)
    baseline = after_summary(workflow, 0, details)
    altered = deepcopy(details)
    for document in altered["node_results"].values():
        for row in document["ranks"]:
            row["application"] = {
                "status": "AVAILABLE",
                "advanced": True,
                "step_before": 10,
                "step": 11,
                "phase": "CHECKPOINT_SAVE",
            }
    result = after_summary(workflow, 0, altered)
    assert result.official_steps == baseline.official_steps, result
    assert (
        result.step_executions[0].details["hung_survey"]
        == baseline.step_executions[0].details["hung_survey"]
    ), result
    assert len(result.official_steps[1].node_ids) == 4, result
    assert "hung_skip_reason" not in result.official_steps[1].parameters, result


def test_first_wave_fr_members_choose_unsampled_peer_before_generic_expansion():
    nodes = [f"node-{index}" for index in range(8)]
    workflow, details = surveyed_workflow(nodes, 5)
    workflow = after_summary(workflow, 0, details)
    signals = {
        nodes[rank]: {
            "ranks": [
                {
                    "rank": rank,
                    "flight_recorder": {
                        "last_entry": {
                            "pg_name": "dp",
                            "pg_members": [0, 1, 2, 7],
                            "collective_seq_id": 9 if rank == 0 else 10,
                            "state": "started",
                        }
                    },
                }
            ]
        }
        for rank in range(3)
    }
    result = after_triage(workflow, 1, {"node_results": signals})
    assert result.official_steps[2].node_ids == ["node-7"], result


def test_unranked_d_state_is_a_process_lead_not_a_fabric_root_cause():
    nodes = ["node-a"]
    workflow, details = surveyed_workflow(nodes, 1)
    row = details["node_results"]["node-a"]["ranks"][0]
    row["rank"] = None
    row["world_size"] = None
    row["proc"].update(state="D", state_before="D", wchan="io_schedule")
    report = survey_analysis(workflow.official_steps[0].parameters, nodes, details)
    assert report["node_coverage_fraction"] == 1, report
    assert report["rank_inventory_complete"] is False, report
    assert report["unranked_process_count"] == report["verified_process_count"] == 1, (
        report
    )
    assert report["diagnostic_lead_nodes"] == ["node-a"], report
    assert report["candidate_ranks"] == [], report
    assert (
        report["diagnosis"] == "UNDETERMINED"
        and report["root_cause_confirmed"] is False
    ), report
