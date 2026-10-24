from gpu_fault.adapters.node_action.adapter import NodeActionWorkflowAdapter
from gpu_fault.hung_summary_codec import decode_summary
from gpu_fault.hung_survey_analysis import survey_analysis
from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.node_agent.protocol import NodeActionResult, NodeActionStatus
from tests.execution.test_full_hung_survey import context
from tests.node_agent._support import SECRET


def test_partial_timeout_is_retained_without_counting_node_as_full_coverage():
    def sender(endpoint, envelope):
        return NodeActionResult(
            command_id=envelope.command.command_id,
            operation=Op.COLLECT_HUNG_SUMMARY,
            status=NodeActionStatus.FAILED,
            error="PARTIAL_TIMEOUT",
            details={
                "status": "PARTIAL_TIMEOUT",
                "evidence_complete": False,
                "ranks": [{"pid": 100, "rank": None}],
                "worker_tree_running": True,
            },
        )

    adapter = NodeActionWorkflowAdapter(
        {"node-a": "https://test.invalid"}, SECRET, sender=sender
    )
    ctx = context(["node-a"])
    result = adapter.execute(ctx)
    partial = decode_summary(result.details["node_results"]["node-a"])
    assert partial["ranks"] == [{"pid": 100, "rank": None}], partial
    assert result.details["node_failures"]["node-a"] == "PARTIAL_TIMEOUT", result
    report = survey_analysis(ctx.step.parameters, ["node-a"], result.details)
    assert report["node_coverage_fraction"] == 0 and not report["candidate_ranks"], (
        report
    )
