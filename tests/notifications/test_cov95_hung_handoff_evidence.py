"""``gpu_fault.execution.hung_handoff_evidence`` on records the mail tests never
build.

``tests/notifications/test_hung_support_notice.py`` renders the read-only hung
hand-off from realistic records. These tests read ``handoff_evidence``
directly on the shapes left over: malformed continuation events, a step
retired without needing the failed one, survey and triage answers that
overflow the per-step bound, a bundle step without its scope, a DAG without a
fabric check, a fabric record still in flight and an UNKNOWN fabric verdict on
a large job. Each must read as what it is -- never as a healthy result.
"""

from __future__ import annotations

from gpu_fault.execution.hung_handoff import HUNG_SUPPORT_HANDOFF
from gpu_fault.execution.hung_handoff_evidence import MAX_NODE_LINES, handoff_evidence
from gpu_fault.fabric_scope import scope_evidence
from gpu_fault.models import (
    WorkflowEventKind,
    WorkflowStepStatus,
    record_workflow_event,
)
from gpu_fault.notifications.hung_support import (
    FABRIC_NOT_PLANNED,
    FABRIC_NOT_RUN,
    FABRIC_UNKNOWN,
    NOT_RUN,
    SUCCEEDED,
)
from tests._builders import workflow_step, workflow_step_execution
from tests.notifications.test_hung_support_notice import (
    NODES,
    OP,
    fabric_details,
    hung_record,
    hung_steps,
    records,
    summary_details,
    triage_details,
)

HANDOFF = 7


def _execution(index, steps, details, status=WorkflowStepStatus.SUCCEEDED, **values):
    return workflow_step_execution(
        index,
        steps[index].operation,
        status,
        phase="official",
        details=details,
        **values,
    )


def _rewrite(workflow, details):
    return record_workflow_event(
        workflow,
        WorkflowEventKind.PLAN_REWRITE,
        code=HUNG_SUPPORT_HANDOFF,
        reason="hung diagnostic step failed; the support hand-off still runs",
        details=details,
    )


def test_a_retirement_names_the_failed_step_only_when_the_event_is_well_formed():
    steps = hung_steps()
    failed = _execution(
        3,
        steps,
        triage_details(),
        WorkflowStepStatus.FAILED,
        error="triage wave 1 exceeded its deadline",
    )
    workflow = hung_record(
        steps, records(steps, {3: failed, 4: None, 5: None, 6: None})
    )
    workflow = _rewrite(
        workflow, {"failed_step_index": "3", "retired_step_indexes": [4]}
    )
    workflow = _rewrite(
        workflow, {"failed_step_index": 3, "retired_step_indexes": ["4", 6]}
    )

    evidence = handoff_evidence(workflow, HANDOFF)

    by_index = {step.index: step for step in evidence.steps}
    assert (by_index[4].status, by_index[4].summary) == (
        NOT_RUN,
        "no execution record",
    ), (
        "an index that is not an int retires nothing, nor does an event whose "
        "failed step is not an int"
    )
    assert (by_index[6].status, by_index[6].summary) == (
        NOT_RUN,
        "retired when step 3 failed",
    ), "the fabric check does not need the triage; it was retired with it"
    assert evidence.fabric.verdict == FABRIC_NOT_RUN
    assert evidence.fabric.summary == (
        "retired when step 3 failed; the task's ports were not checked"
    )


def test_undetermined_nodes_without_a_failure_are_named_once():
    steps = hung_steps()
    survey = {
        **summary_details(failures={"node-b": "agent unreachable"}),
        "undetermined_nodes": ["node-b", "node-c", 7],
    }
    workflow = hung_record(steps, records(steps, {1: _execution(1, steps, survey)}))

    (baseline,) = [
        step for step in handoff_evidence(workflow, HANDOFF).steps if step.index == 1
    ]
    assert baseline.lines[:2] == (
        "node-b: no result (agent unreachable)",
        "node-c: undetermined (no usable answer)",
    ), baseline.lines
    assert not [line for line in baseline.lines if line.startswith("7")], (
        "a node id that is not text is not reported"
    )


def test_a_triage_with_more_leads_than_the_bound_counts_the_rest():
    steps = hung_steps()
    triage = triage_details()
    triage["hung_triage_decision"]["culprit_ranks"] = list(range(MAX_NODE_LINES + 3))
    workflow = hung_record(steps, records(steps, {3: _execution(3, steps, triage)}))

    (wave,) = [
        step for step in handoff_evidence(workflow, HANDOFF).steps if step.index == 3
    ]
    shown = ", ".join(
        [f"{rank}@{node}" for rank, node in enumerate(NODES)]
        + [str(rank) for rank in range(len(NODES), MAX_NODE_LINES)]
    )
    assert (
        f"candidate ranks (leads, not proof): {shown}, ... "
        f"({MAX_NODE_LINES + 3} in total)"
    ) in wave.summary, wave.summary


def test_the_rank_map_comes_from_the_latest_survey_that_has_one():
    steps = hung_steps()
    recheck = summary_details()
    recheck["hung_survey"]["rank_node_map"] = {}
    workflow = hung_record(steps, records(steps, {2: _execution(2, steps, recheck)}))

    (wave,) = [
        step for step in handoff_evidence(workflow, HANDOFF).steps if step.index == 3
    ]
    assert "candidate ranks (leads, not proof): 1@node-b;" in wave.summary, (
        "an empty RECHECK map falls back to the BASELINE survey's"
    )


def test_a_bundle_without_a_scope_counts_the_job_nodes_outside_it():
    steps = hung_steps()
    parameters = {
        key: value
        for key, value in steps[5].parameters.items()
        if key != "bundle_scope"
    }
    parameters["bundle_not_collected_nodes"] = ["node-c", "node-d"]
    steps[5] = steps[5].model_copy(update={"parameters": parameters})
    workflow = hung_record(steps)

    (bundle,) = [
        step for step in handoff_evidence(workflow, HANDOFF).steps if step.index == 5
    ]
    assert bundle.summary == (
        "2 archive(s); 2 job node(s) outside the bounded scope"
    ), bundle.summary


def test_a_dag_without_a_fabric_check_reports_it_not_planned():
    steps = hung_steps()
    steps[6] = workflow_step(
        OP.TRIGGER_HEALTH_SNAPSHOT,
        node_ids=list(NODES),
        depends_on_step_indexes=[2],
        parameters=dict(steps[6].parameters),
    )
    snapshot = _execution(6, steps, {"node_failures": {"node-a": ["timed", "out", 3]}})
    executions = [*records(steps, {6: None}), snapshot]
    workflow = hung_record(steps, executions)

    evidence = handoff_evidence(workflow, HANDOFF)

    assert evidence.fabric.verdict == FABRIC_NOT_PLANNED, evidence.fabric
    (other,) = [step for step in evidence.steps if step.index == 6]
    assert (other.label, other.status, other.summary) == (
        "TRIGGER_HEALTH_SNAPSHOT",
        SUCCEEDED,
        "",
    ), other
    assert other.lines == ("node-a: no result (timed, out, 3)",), other.lines


def test_a_fabric_check_still_in_flight_is_not_run_and_not_finished():
    steps = hung_steps()
    waiting = _execution(6, steps, fabric_details(steps), WorkflowStepStatus.WAITING)
    workflow = hung_record(steps, records(steps, {6: waiting}))

    evidence = handoff_evidence(workflow, HANDOFF)

    assert evidence.fabric.verdict == FABRIC_NOT_RUN
    assert evidence.fabric.summary == (
        "the step did not finish (WAITING); the task's ports were not checked"
    )
    (fabric,) = [step for step in evidence.steps if step.index == 6]
    assert fabric.status == "NOT FINISHED (WAITING)", fabric


def test_a_skipped_fabric_check_names_why_it_did_not_run():
    steps = hung_steps()
    skipped = _execution(6, steps, {"skipped": "ATTEMPT_TERMINAL"})
    workflow = hung_record(steps, records(steps, {6: skipped}))

    evidence = handoff_evidence(workflow, HANDOFF)

    assert evidence.fabric.verdict == FABRIC_NOT_RUN
    assert evidence.fabric.summary == (
        "skipped: ATTEMPT_TERMINAL; the task's ports were not checked"
    )


def test_an_unknown_fabric_verdict_on_a_large_job_is_bounded_and_counted():
    nodes = [f"node-{index:02d}" for index in range(MAX_NODE_LINES + 4)]
    unbound = {
        node: scope_evidence(
            {"basis": "UNKNOWN", "reason": "TASK_RDMA_RESOURCES_UNAVAILABLE"}
        )
        for node in nodes
    }
    steps = hung_steps(nodes, unbound=unbound)
    workflow = hung_record(steps)

    fabric = handoff_evidence(workflow, HANDOFF).fabric

    assert fabric.verdict == FABRIC_UNKNOWN, fabric
    assert len(fabric.unknown_nodes) == MAX_NODE_LINES + 1, fabric.unknown_nodes
    assert [node for node, _reason in fabric.unknown_nodes[:MAX_NODE_LINES]] == (
        nodes[:MAX_NODE_LINES]
    )
    assert fabric.unknown_nodes[-1] == ("...", "4 more in the record")
