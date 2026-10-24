"""The read-only hung diagnostic's hand-off sends its own notice, not the
hardware escalation.

Batch 161 made the hung DAG's ESCALATE_SUPPORT always run. It then sent the
HARDWARE_ESCALATION mail (``hardware-escalation-zh-v3``) and recorded
``hardware_disposition=OFFLINE_QUARANTINED``: "hardware taken out of service
... the faulty nodes stay unschedulable and quarantined". The hung diagnostic
cordons, quarantines, stops and resets nothing -- the job is left stalled as it
was -- so every hung case would have mailed that falsehood. The hand-off now
sends ``hung-support-handoff-zh-v1`` (``notifications/hung_support.py``),
which reports each evidence step from its own record
(``execution/hung_handoff_evidence.py``); every other ESCALATE_SUPPORT keeps
the hardware or XID 74 notice.
"""

from __future__ import annotations

import hashlib

import pytest

from gpu_fault.adapters.support_escalation import SupportEscalationAdapter
from gpu_fault.execution import WorkflowStepContext
from gpu_fault.execution.hung_handoff import support_handoff
from gpu_fault.execution.hung_handoff_evidence import MAX_NODE_LINES
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.fabric_scope import (
    scope_evidence,
    unknown_task_fabric,
    validation_evidence,
)
from gpu_fault.models import (
    IncidentState,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.notifications.hardware_escalation import HardwareEscalationEmailBuilder
from gpu_fault.notifications.hung_support import (
    HUNG_SUPPORT_HANDOFF_TEMPLATE_VERSION,
    HungSupportHandoffEmailBuilder,
)
from gpu_fault.notifications.registry import (
    NotificationBuilderRegistry,
    NotificationKind,
)
from tests._builders import (
    active_workflow_executor,
    build_store,
    execute_workflow,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.execution._nccl_chain_support import chain_fixture as chain_fixture
from tests.execution._support import FakeAdapter

OP = WorkflowOperation
SUPPORT_OWNER = "gpu-fault-support-escalation"
NODES = ["node-a", "node-b"]
WORKLOAD = "training/pytorchjob/train"
INCIDENT = "inc-host-node-a-efa-traffic-hung_suspected"
WORKFLOW = "workflow-hung-b162"
CONTRACT = {
    "health_recovery_contract": {
        "mode": "HUNG_DIAGNOSTIC",
        "metric": "efa_traffic_bytes_per_second",
        "detected_at": "2026-10-24T13:50:17+00:00",
    }
}
#: What the hardware escalation says and the hung hand-off must never say.
ISOLATION_CLAIMS = (
    "out of service",
    "unschedulable and quarantined",
    "HARDWARE_OFFLINE",
    "hardware-escalation-zh-v3",
)
BOUND = {"basis": "TASK_DEVICE_BINDING", "network_interfaces": [], "rdma_ports": []}


def digest(*parts: str) -> str:
    return hashlib.sha256("/".join(parts).encode()).hexdigest()


def hung_steps(nodes=NODES, *, bindings=None, unbound=None):
    """The step shape of ``orchestration/hung_dag.py``."""

    def step(operation, needs, owner="owner-a", **parameters):
        return workflow_step(
            operation,
            owner,
            node_ids=list(nodes),
            workload_ids=[WORKLOAD],
            depends_on_step_indexes=needs,
            parameters={**CONTRACT, **parameters},
        )

    identity = {"job_id": "train", "attempt_id": "attempt-7"}
    fabric = {
        "require_task_fabric_scope": True,
        "training_fabric_by_node": bindings or {},
        **({"training_fabric_unavailable_by_node": unbound} if unbound else {}),
    }
    return [
        step(OP.FREEZE_EVIDENCE, []),
        step(OP.COLLECT_HUNG_SUMMARY, [0], survey_pass="BASELINE", **identity),
        step(OP.COLLECT_HUNG_SUMMARY, [1], survey_pass="RECHECK", **identity),
        step(OP.COLLECT_HUNG_TRIAGE, [2], hung_wave=1, **identity),
        step(OP.COLLECT_HUNG_TRIAGE, [3], hung_wave=2, **identity),
        step(
            OP.COLLECT_DIAGNOSTIC_BUNDLE,
            [3, 4],
            bundle_scope="bounded_triage_nodes",
            bundle_not_collected_nodes=[],
        ),
        step(OP.VALIDATE_FABRIC, [2], **fabric),
        step(OP.ESCALATE_SUPPORT, [5, 6], SUPPORT_OWNER),
    ]


def summary_details(nodes=NODES, failures=None):
    answered = [node for node in nodes if node not in (failures or {})]
    return {
        "hung_survey": {
            "expected_node_count": len(nodes),
            "responding_node_count": len(answered),
            "expected_rank_count": len(nodes),
            "valid_rank_count": len(answered),
            "diagnosis": "UNDETERMINED",
            "rank_node_map": {str(rank): node for rank, node in enumerate(nodes)},
        },
        "node_results": {
            node: {"encoding": "hung-summary-zlib-json/v1", "sha256": digest(node)}
            for node in answered
        },
        "node_failures": dict(failures or {}),
        "undetermined_nodes": sorted(failures or {}),
    }


def triage_details():
    return {
        "hung_triage_decision": {
            "classification": "PLAUSIBLE",
            "confidence": "MEDIUM",
            "culprit_ranks": [1],
            "reason": "rank 1 stopped behind its process group",
            "root_cause_confirmed": False,
            "active_requested_node_count": 2,
            "active_responding_node_count": 2,
        },
        "node_results": {node: {"sha256": digest(node, "triage")} for node in NODES},
        "node_failures": {},
    }


def bundle_details():
    return {
        "node_results": {
            node: {
                "evidence_ref": f"s3://evidence-bucket/diagnostics/{node}/bundle.tar.gz",
                "sha256": digest(node, "bundle"),
                "size_bytes": 4096,
                "manifest_summary": {"capture_count": 7, "failed_capture_count": 0},
            }
            for node in NODES
        },
        "failed_nodes": [],
        "node_failures": {},
    }


def fabric_details(steps):
    step = steps[6]
    return {
        **validation_evidence(
            OP.VALIDATE_FABRIC, {}, unknown_task_fabric(step), step.node_ids
        ),
        **(
            {"validated_nodes": list(step.node_ids)}
            if not unknown_task_fabric(step)
            else {}
        ),
    }


def records(steps, overrides=None):
    """Every evidence step SUCCEEDED with realistic details unless overridden."""

    details = {
        0: {
            "captured_at": "2026-10-24T13:50:20+00:00",
            "hung_task_summary": {"status": "AVAILABLE", "expected_node_count": 2},
        },
        1: summary_details(),
        2: summary_details(),
        3: triage_details(),
        4: {**triage_details(), "skipped": "DIAGNOSTIC_BUDGET_EXHAUSTED"},
        5: bundle_details(),
        6: fabric_details(steps),
    }
    executions = []
    for index in range(7):
        if overrides and index in overrides:
            if overrides[index] is not None:
                executions.append(overrides[index])
            continue
        executions.append(
            workflow_step_execution(
                index,
                steps[index].operation,
                phase="official",
                adapter_operation_id=f"{WORKFLOW}/{index}/{steps[index].operation.value}",
                details=details[index],
            )
        )
    return executions


def hung_record(steps=None, executions=None, **values):
    steps = steps or hung_steps()
    executions = records(steps) if executions is None else executions
    return workflow_request(
        WORKFLOW,
        INCIDENT,
        status=WorkflowStatus.RUNNING,
        dag_enabled=True,
        dag_revision=4,
        official_steps=steps,
        step_executions=executions,
        completed_step_indexes=sorted(
            item.step_index
            for item in executions
            if item.status is WorkflowStepStatus.SUCCEEDED
        ),
        **values,
    )


def incident(**values):
    return fault_incident(
        INCIDENT,
        "host-node-a-efa-traffic-hung_suspected",
        "GPU_FAULT_GROUP",
        node_ids=NODES,
        policy_source="SITE_EFA_TRAFFIC",
        policy_version="site-efa-traffic-policy/v2",
        job_id="train",
        attempt_id="attempt-7",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=WORKFLOW,
        fencing_token=3,
        reasons=[
            "EFA/RDMA traffic remains at zero long enough to suspect a hung "
            "training attempt"
        ],
        **values,
    )


def hand_off(workflow, store=None, case=None, sent=None):
    store = store or build_store()
    case = case or incident()
    index = next(
        position
        for position, step in enumerate(workflow.official_steps)
        if step.operation is OP.ESCALATE_SUPPORT
    )
    outcome = SupportEscalationAdapter(
        store, alert_sender=None if sent is None else sent.append
    ).execute(
        WorkflowStepContext(
            workflow=workflow,
            incident=case,
            step=workflow.official_steps[index],
            step_index=index,
            request=WorkflowExecutionRequest(expected_fencing_token=3),
            idempotency_key=f"{workflow.request_id}/{index}/ESCALATE_SUPPORT",
        )
    )
    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
    [notification] = store.list_notifications()
    return outcome, notification


def section(body, number):
    start = body.index(f"\n{number}. ")
    end = body.index(f"\n{number + 1}. ", start + 1)
    return body[start:end]


def test_the_hung_hand_off_says_read_only_and_never_claims_isolation():
    steps = hung_steps()
    sent = []
    outcome, notification = hand_off(hung_record(steps), sent=sent)

    body, case = notification.body_text, notification.support_case_draft
    for claim in ISOLATION_CLAIMS:
        assert claim not in body and claim not in case, (claim, body)
    assert "it did not cordon, drain, quarantine, reset or reboot" in body, body
    assert "it did not stop, suspend or restart any workload" in body, body
    assert f"Template: {HUNG_SUPPORT_HANDOFF_TEMPLATE_VERSION}" in body, body
    assert notification.subject == (
        "[GPU FAULT][training hung: read-only hand-off] cluster-a "
        "train/attempt-7: node-a, node-b"
    ), notification.subject
    assert notification.category == "OPERATOR_REVIEW", notification
    assert notification.deduplication_key == (
        f"cluster-a/{INCIDENT}/hung-support-handoff/{WORKFLOW}"
    ), notification
    assert "- Training job: train\n- Attempt: attempt-7" in body, body
    assert "Disposition: READ_ONLY_HUNG_DIAGNOSTIC" in case, case
    assert sent == [notification.notification_id], sent
    # The step details claim nothing either.
    assert "hardware_disposition" not in outcome.details, outcome.details
    assert outcome.details == {
        "ticket_id": f"vendor-ticket-{INCIDENT}",
        "notification_id": notification.notification_id,
        "notification_template": HUNG_SUPPORT_HANDOFF_TEMPLATE_VERSION,
        "support_disposition": "READ_ONLY_HUNG_DIAGNOSTIC",
        "nodes_isolated": False,
        "workloads_stopped": False,
        "fabric_verdict": "UNKNOWN",
    }, outcome.details


def test_every_evidence_step_is_reported_with_its_record_and_archives():
    steps = hung_steps()
    _, notification = hand_off(hung_record(steps))

    evidence = section(notification.body_text, 3)
    for index, label in (
        (0, "FREEZE_EVIDENCE"),
        (1, "COLLECT_HUNG_SUMMARY (BASELINE)"),
        (2, "COLLECT_HUNG_SUMMARY (RECHECK)"),
        (3, "COLLECT_HUNG_TRIAGE (wave 1)"),
        (5, "COLLECT_DIAGNOSTIC_BUNDLE"),
        (6, "VALIDATE_FABRIC"),
    ):
        assert f"- Step {index} {label}: SUCCEEDED" in evidence, (label, evidence)
        assert f"Record: {WORKFLOW}/{index}/{steps[index].operation.value}" in (
            evidence
        ), (index, evidence)
    assert "Step 7" not in evidence, "the hand-off does not report itself"
    assert "- Step 4 COLLECT_HUNG_TRIAGE (wave 2): SKIPPED -- " in evidence, evidence
    assert "DIAGNOSTIC_BUDGET_EXHAUSTED" in evidence, evidence
    assert "2/2 node(s) answered; ranks 2/2; diagnosis UNDETERMINED" in evidence, (
        evidence
    )
    assert f"node-a: summary sha256 {digest('node-a')}" in evidence, evidence
    assert "verdict PLAUSIBLE (confidence MEDIUM)" in evidence, evidence
    assert "candidate ranks (leads, not proof): 1@node-b" in evidence, evidence
    assert "root cause confirmed: no" in evidence, evidence
    assert "Reason: rank 1 stopped behind its process group" in evidence, evidence
    refs = [f"s3://evidence-bucket/diagnostics/{node}/bundle.tar.gz" for node in NODES]
    for node, ref in zip(NODES, refs):
        assert f"{node}: {ref} sha256 {digest(node, 'bundle')}" in evidence, evidence
    assert notification.evidence_refs == refs, notification.evidence_refs
    assert "Diagnostic result: every evidence step finished" in (
        notification.body_text
    ), notification.body_text
    assert f"GET /v1/workflows/{WORKFLOW}" in notification.body_text
    assert f"--close-incident {INCIDENT}" in notification.body_text


def test_an_unknown_task_scope_names_every_node_and_its_reason():
    unbound = {
        node: scope_evidence(
            {"basis": "UNKNOWN", "reason": "TASK_RDMA_RESOURCES_UNAVAILABLE"}
        )
        for node in NODES
    }
    _, notification = hand_off(hung_record(hung_steps(unbound=unbound)))

    fabric = section(notification.body_text, 4)
    assert "- Result: UNKNOWN -- the task's NIC/RDMA port scope could not be " in (
        fabric
    ), fabric
    assert "for any node; this is missing evidence, not a healthy fabric" in fabric
    assert "- Evidence scope: TASK_SCOPE_UNAVAILABLE" in fabric, fabric
    assert "- Validated nodes: NONE" in fabric, fabric
    for node in NODES:
        assert (
            f"  - {node}: TASK_NIC_PORT_SCOPE_UNAVAILABLE (no task NIC/RDMA port "
            "binding); node survey: scope_basis=UNKNOWN, "
            "scope_reason=TASK_RDMA_RESOURCES_UNAVAILABLE"
        ) in fabric, fabric
    assert "no active communication test ran and NCCL recovery is not proven" in (
        fabric
    ), fabric
    next_steps = section(notification.body_text, 5)
    assert "rdma -j resource show qp" in next_steps, next_steps


def test_a_partial_task_scope_names_the_validated_and_the_unknown_nodes():
    bindings = {"node-a": {**BOUND, "rdma_ports": ["rdmap16s27/1"]}}
    unbound = {
        "node-b": scope_evidence(
            {
                "basis": "UNKNOWN",
                "reason": "TASK_RDMA_COVERAGE_INCOMPLETE",
                "source": "LIVE_PROCESS_RDMA_QP",
                "expected_pids": [100, 101],
                "covered_pids": [100],
            }
        )
    }
    outcome, notification = hand_off(
        hung_record(hung_steps(bindings=bindings, unbound=unbound))
    )

    fabric = section(notification.body_text, 4)
    assert "could not be determined for some nodes" in fabric, fabric
    assert "- Evidence scope: PARTIAL_TASK_SCOPE" in fabric, fabric
    assert "- Validated nodes: node-a" in fabric, fabric
    assert (
        "  - node-b: TASK_NIC_PORT_SCOPE_UNAVAILABLE (no task NIC/RDMA port "
        "binding); node survey: scope_basis=UNKNOWN, "
        "scope_reason=TASK_RDMA_COVERAGE_INCOMPLETE, "
        "scope_source=LIVE_PROCESS_RDMA_QP, rank processes covered 1/2 -- not "
        "every rank process holds an active QP on a present EFA port"
    ) in fabric, fabric
    assert "node-a: TASK_NIC" not in fabric, "a bound node is validated, not UNKNOWN"
    assert outcome.details["fabric_verdict"] == "UNKNOWN", outcome.details


def test_a_failed_step_and_the_steps_it_retired_are_reported_as_they_ended():
    """The real continuation (``hung_handoff.support_handoff``): the recheck
    summary failed, everything that needs it was retired, the hand-off ran."""

    steps = hung_steps()
    failed = workflow_step_execution(
        2,
        OP.COLLECT_HUNG_SUMMARY,
        WorkflowStepStatus.FAILED,
        phase="official",
        error="summary collection exceeded its 150 s deadline",
        details=summary_details(failures={"node-b": "agent unreachable"}),
    )
    workflow = hung_record(
        steps, records(steps, {2: failed, 3: None, 4: None, 5: None, 6: None})
    )
    continued = support_handoff(
        workflow, 2, WorkflowStepOutcome.failed(failed.error or "")
    )
    assert continued is not None, "the read-only hung DAG goes on to its hand-off"
    _, notification = hand_off(continued.workflow)

    body = notification.body_text
    evidence = section(body, 3)
    assert (
        "- Step 2 COLLECT_HUNG_SUMMARY (RECHECK): FAILED -- summary collection "
        "exceeded its 150 s deadline"
    ) in evidence, evidence
    assert "  - Collected before it failed: 1/2 node(s) answered" in evidence
    assert "  - node-b: no result (agent unreachable)" in evidence, evidence
    for index in (3, 4, 5, 6):
        assert f"Step {index} " in evidence, evidence
    assert (
        evidence.count(": NOT RUN -- retired: it needs the output of failed step 2")
        == 4
    ), evidence
    assert (
        "- Result: NOT RUN -- retired: it needs the output of failed step 2; "
        "the task's ports were not checked"
    ) in section(body, 4), body
    assert (
        "Diagnostic result: incomplete: 1 step(s) FAILED and 4 step(s) did not "
        "run; the workflow ends FAILED with its first failure (summary "
        "collection exceeded its 150 s deadline)"
    ) in body, body
    assert "Evidence the FAILED or not-run steps would have collected is missing" in (
        section(body, 5)
    ), body
    for claim in ISOLATION_CLAIMS:
        assert claim not in body, claim


def test_a_failed_fabric_validation_names_the_failing_ports():
    steps = hung_steps(
        bindings={node: {**BOUND, "rdma_ports": ["efa0/1"]} for node in NODES}
    )
    failed = workflow_step_execution(
        6,
        OP.VALIDATE_FABRIC,
        WorkflowStepStatus.FAILED,
        phase="official",
        error="validation failed on nodes: node-b (rdma_link_down:efa0/1)",
        details={"node_failures": {"node-b": ["rdma_link_down:efa0/1"]}},
    )
    outcome, notification = hand_off(hung_record(steps, records(steps, {6: failed})))

    fabric = section(notification.body_text, 4)
    assert (
        "- Result: FAILED -- validation failed on nodes: node-b (rdma_link_down:efa0/1)"
    ) in fabric, fabric
    assert "- node-b: rdma_link_down:efa0/1" in fabric, fabric
    assert "Evidence scope" not in fabric, "a failed record names no scope"
    assert f"Record: {WORKFLOW}/6/VALIDATE_FABRIC" in section(
        notification.body_text, 3
    ), "a FAILED record still names its step"
    assert outcome.details["fabric_verdict"] == "FAILED", outcome.details


def test_a_passed_fabric_is_a_telemetry_check_of_the_task_ports_only():
    bindings = {node: {**BOUND, "rdma_ports": ["efa0/1"]} for node in NODES}
    outcome, notification = hand_off(hung_record(hung_steps(bindings=bindings)))

    fabric = section(notification.body_text, 4)
    assert "- Result: PASSED -- no fault on the task's NICs and RDMA ports" in fabric
    assert "- Evidence scope: SCOPED_TELEMETRY_ONLY" in fabric, fabric
    assert "- Validated nodes: node-a, node-b" in fabric, fabric
    assert "- Task ports of node-b: RDMA efa0/1; NICs NONE" in fabric, fabric
    assert "NCCL recovery is not proven" in fabric, fabric
    assert "does not show that the training resumed" in section(
        notification.body_text, 5
    ), notification.body_text
    assert outcome.details["fabric_verdict"] == "PASSED", outcome.details


def test_malformed_records_read_unknown_never_healthy():
    steps = hung_steps(
        bindings={node: dict(BOUND, rdma_ports=["e/1"]) for node in NODES}
    )
    garbage = {
        1: workflow_step_execution(
            1,
            OP.COLLECT_HUNG_SUMMARY,
            phase="official",
            details={
                "node_results": {"node-a": "not-a-dict", "node-b": {"sha256": "x"}}
            },
        ),
        3: workflow_step_execution(
            3,
            OP.COLLECT_HUNG_TRIAGE,
            phase="official",
            details={
                "hung_triage_decision": {
                    "classification": "<script>",
                    "culprit_ranks": ["1", None],
                    "active_requested_node_count": -1,
                }
            },
        ),
        5: workflow_step_execution(
            5,
            OP.COLLECT_DIAGNOSTIC_BUNDLE,
            phase="official",
            details={
                "node_results": {"node-a": {"evidence_ref": "javascript:alert(1)"}}
            },
        ),
        # A status this reader does not know is never a pass.
        6: workflow_step_execution(
            6,
            OP.VALIDATE_FABRIC,
            phase="official",
            details={"fabric_status": "MAYBE", "validated_nodes": "node-a"},
        ),
    }
    outcome, notification = hand_off(hung_record(steps, records(steps, garbage)))

    body = notification.body_text
    assert "- Step 1 COLLECT_HUNG_SUMMARY (BASELINE): SUCCEEDED -- no survey " in body
    assert "node-a: summary (digest not recorded)" in body, body
    assert "node-b: summary (digest not recorded)" in body, body
    assert "verdict UNKNOWN (confidence UNKNOWN)" in body, body
    assert "candidate ranks (leads, not proof): NONE" in body, body
    assert "UNKNOWN/UNKNOWN sampled node(s) answered" in body, body
    assert "node-a: archive reference not recorded sha256 not recorded" in body
    assert notification.evidence_refs == [], notification.evidence_refs
    assert "javascript" not in body and "<script>" not in body, body
    assert "- Result: UNKNOWN" in section(body, 4), body
    assert "Validated nodes" not in section(body, 4), "a non-list names no nodes"
    assert outcome.details["fabric_verdict"] == "UNKNOWN", outcome.details


def test_large_jobs_are_bounded_and_counted():
    nodes = [f"node-{index:03d}" for index in range(40)]
    steps = hung_steps(
        nodes, bindings={node: dict(BOUND, rdma_ports=["e/1"]) for node in nodes}
    )
    big = workflow_step_execution(
        1, OP.COLLECT_HUNG_SUMMARY, phase="official", details=summary_details(nodes)
    )
    _, notification = hand_off(hung_record(steps, records(steps, {1: big})))

    evidence = section(notification.body_text, 3)
    block = evidence[
        evidence.index("Step 1 ") : evidence.index("- Step 2 ")
    ].splitlines()
    shown = [line for line in block if "summary sha256" in line]
    assert len(shown) == MAX_NODE_LINES, block
    assert f"  - ... and {40 - MAX_NODE_LINES} more node(s) in the step record" in (
        block
    ), block


def test_an_attempt_that_ended_during_the_diagnostic_is_named():
    steps = hung_steps()
    released = workflow_step_execution(
        3,
        OP.COLLECT_HUNG_TRIAGE,
        phase="official",
        details={
            "skipped": "ATTEMPT_TERMINAL",
            "recovery_owner": "COMPLETION_WATCHER",
            "node_results": {},
            "hung_evidence_complete": False,
        },
    )
    _, notification = hand_off(hung_record(steps, records(steps, {3: released})))

    body = notification.body_text
    assert (
        "- Training attempt: no longer live during the diagnostic "
        "(ATTEMPT_TERMINAL): its exit handling (the Completion Watcher) owns "
        "any restart, not this hand-off."
    ) in body, body
    assert (
        "- Step 3 COLLECT_HUNG_TRIAGE (wave 1): SKIPPED -- ATTEMPT_TERMINAL "
        "(recovery owner COMPLETION_WATCHER)"
    ) in body, body


def test_steps_retired_past_the_execution_deadline_are_not_called_dependents():
    """Past the deadline ``support_handoff`` retires every step that has not
    started, whether or not it needs the failed step's output."""

    steps = hung_steps()
    deadline = workflow_step_execution(
        3,
        OP.COLLECT_HUNG_TRIAGE,
        WorkflowStepStatus.FAILED,
        phase="official",
        error="workflow execution deadline exceeded",
        details={"workflow_execution_deadline": "2026-10-24T14:00:00+00:00"},
    )
    workflow = hung_record(
        steps, records(steps, {3: deadline, 4: None, 5: None, 6: None})
    )
    continued = support_handoff(
        workflow,
        3,
        WorkflowStepOutcome.failed(
            "workflow execution deadline exceeded",
            details={"workflow_execution_deadline": "2026-10-24T14:00:00+00:00"},
        ),
    )
    assert continued is not None, "the hand-off is owed past the deadline"
    _, notification = hand_off(continued.workflow)

    evidence = section(notification.body_text, 3)
    for label in ("4 COLLECT_HUNG_TRIAGE (wave 2)", "5 COLLECT_DIAGNOSTIC_BUNDLE"):
        assert (
            f"- Step {label}: NOT RUN -- retired: it needs the output of failed step 3"
        ) in evidence, evidence
    assert (
        "- Step 6 VALIDATE_FABRIC: NOT RUN -- not started: the execution deadline "
        "passed (step 3 failed on it)"
    ) in evidence, evidence


def test_a_step_superseded_without_a_failure_claims_no_failed_workflow():
    steps = hung_steps()
    workflow = hung_record(
        steps, records(steps, {4: None}), superseded_step_indexes=[4]
    )
    _, notification = hand_off(workflow)

    body = notification.body_text
    assert "- Step 4 COLLECT_HUNG_TRIAGE (wave 2): NOT RUN -- superseded" in body
    assert (
        "Diagnostic result: incomplete: 0 step(s) FAILED and 1 step(s) did not "
        "run. The evidence"
    ) in body, body
    assert "ends FAILED" not in body, "no step failed, so no failure is claimed"


def test_a_retried_hand_off_sends_one_notice():
    store = build_store()
    workflow = hung_record()
    first, notification = hand_off(workflow, store)
    second, again = hand_off(workflow, store)
    assert again.notification_id == notification.notification_id, again
    assert second.details["notification_id"] == first.details["notification_id"]


def mutating(workflow):
    steps = list(workflow.official_steps)
    steps[5] = steps[5].model_copy(update={"operation": OP.RESET_GPU})
    return workflow.model_copy(update={"official_steps": steps})


def training_contract(workflow):
    return workflow.model_copy(
        update={
            "official_steps": [
                step.model_copy(
                    update={
                        "parameters": {
                            **step.parameters,
                            "health_recovery_contract": {"mode": "TRAINING"},
                        }
                    }
                )
                for step in workflow.official_steps
            ]
        }
    )


def ordinary(workflow):
    return workflow.model_copy(
        update={
            "official_steps": [
                workflow.official_steps[-1].model_copy(update={"parameters": {}})
            ],
            "step_executions": [],
            "completed_step_indexes": [],
            "dag_enabled": False,
        }
    )


@pytest.mark.parametrize(
    "shape",
    [mutating, training_contract, ordinary],
    ids=["plan-touches-a-node", "training-contract", "ordinary-support-ticket"],
)
def test_every_other_support_escalation_keeps_the_hardware_notice(shape):
    outcome, notification = hand_off(shape(hung_record()))
    assert outcome.details["hardware_disposition"] == "OFFLINE_QUARANTINED", outcome
    body = notification.body_text
    assert "Template: hardware-escalation-zh-v3" in body, body
    assert "unschedulable and quarantined" in body, body
    assert notification.subject.startswith("[GPU FAULT][hardware out of service]"), (
        notification.subject
    )
    assert notification.deduplication_key == (
        f"{INCIDENT}/hardware-escalation/hardware-escalation-zh-v3"
    ), notification


def test_the_xid74_support_notice_is_unchanged():
    outcome, notification = hand_off(
        ordinary(hung_record()), case=incident(official_action="WORKFLOW_NVLINK_ERR")
    )
    assert "XID 74 NVLink" in notification.subject, notification.subject
    assert outcome.details["hardware_disposition"] == "OFFLINE_QUARANTINED", outcome


def test_the_registry_names_the_hung_notice():
    builder = NotificationBuilderRegistry().builder(
        NotificationKind.HUNG_SUPPORT_HANDOFF
    )
    assert isinstance(builder, HungSupportHandoffEmailBuilder), builder
    assert not isinstance(builder, HardwareEscalationEmailBuilder), builder


def test_the_executor_hands_the_continued_dag_to_the_hung_notice():
    """Executor end to end: the triage FAILED, the bundle that needs it was
    retired, the fabric check and the real support adapter still ran."""

    store = build_store()
    operations = [
        OP.FREEZE_EVIDENCE,
        OP.COLLECT_HUNG_TRIAGE,
        OP.COLLECT_DIAGNOSTIC_BUNDLE,
        OP.VALIDATE_FABRIC,
        OP.ESCALATE_SUPPORT,
    ]
    needs = [[], [0], [1], [0], [2, 3]]
    steps = [
        workflow_step(
            operation,
            SUPPORT_OWNER if operation is OP.ESCALATE_SUPPORT else "owner-a",
            node_ids=NODES,
            workload_ids=[WORKLOAD],
            depends_on_step_indexes=dependencies,
            parameters=dict(CONTRACT),
        )
        for operation, dependencies in zip(operations, needs)
    ]
    case = incident()
    store.save_incident(case)
    store.save_workflow(
        workflow_request(
            WORKFLOW, INCIDENT, official_steps=steps, dag_enabled=True, dag_revision=1
        )
    )
    fake = FakeAdapter(
        {
            OP.FREEZE_EVIDENCE: WorkflowStepOutcome.succeeded(operation_id="freeze"),
            OP.COLLECT_HUNG_TRIAGE: WorkflowStepOutcome.failed("agent unavailable"),
            OP.COLLECT_DIAGNOSTIC_BUNDLE: WorkflowStepOutcome.succeeded(
                operation_id="bundle"
            ),
            OP.VALIDATE_FABRIC: WorkflowStepOutcome.succeeded(
                operation_id="fabric",
                details={
                    "fabric_evidence_scope": "SCOPED_TELEMETRY_ONLY",
                    "validated_nodes": NODES,
                },
            ),
        }
    )
    executor = active_workflow_executor(
        store, [fake, SupportEscalationAdapter(store)], operations
    )
    result = execute_workflow(executor, WORKFLOW)

    assert result.status is WorkflowStatus.FAILED, result
    saved = store.get_workflow(WORKFLOW)
    [support] = [
        item for item in saved.step_executions if item.operation is OP.ESCALATE_SUPPORT
    ]
    assert support.status is WorkflowStepStatus.SUCCEEDED, support
    assert support.details["support_disposition"] == "READ_ONLY_HUNG_DIAGNOSTIC"
    [notification] = store.list_notifications()
    body = notification.body_text
    assert "- Step 1 COLLECT_HUNG_TRIAGE: FAILED -- agent unavailable" in body, body
    assert (
        "- Step 2 COLLECT_DIAGNOSTIC_BUNDLE: NOT RUN -- retired: it needs the "
        "output of failed step 1"
    ) in body, body
    assert "- Step 3 VALIDATE_FABRIC: SUCCEEDED -- fabric PASSED" in body, body
    assert "the workflow ends FAILED with its first failure (agent unavailable)" in body
    assert store.get_incident(INCIDENT).state is IncidentState.ESCALATED


def test_the_real_hung_chain_hands_off_with_the_hung_notice(chain, monkeypatch):
    """Raw RDMA -> API -> real node handlers -> real validation -> the real
    support adapter: no node of the task could bind its EFA ports, and the
    notice carries each node's own UNKNOWN account and the bundle archives."""

    rig = chain()
    rig.fabric_scope_known = False
    adapter = SupportEscalationAdapter(rig.store, owner="simulated-runtime")
    boundary = rig.boundary.execute

    def execute(context):
        if context.step.operation is OP.ESCALATE_SUPPORT:
            rig.boundary.calls.append(context.step.operation)
            return adapter.execute(context)
        return boundary(context)

    monkeypatch.setattr(rig.boundary, "execute", execute)
    workflow = rig.trigger()
    rig.healthy()
    result = rig.run(workflow)
    assert result.status is WorkflowStatus.SUCCEEDED, result

    saved = rig.store.get_workflow(workflow.request_id)
    assert rig.store.get_incident(saved.incident_id).state is IncidentState.ESCALATED
    [support] = [
        item for item in saved.step_executions if item.operation is OP.ESCALATE_SUPPORT
    ]
    assert "hardware_disposition" not in support.details, support.details
    assert support.details["fabric_verdict"] == "UNKNOWN", support.details
    notices = [
        item
        for item in rig.store.list_notifications()
        if item.notification_id == support.details["notification_id"]
    ]
    assert len(notices) == 1, rig.store.list_notifications()
    [notice] = notices
    assert all(
        "hardware-escalation" not in item.body_text
        for item in rig.store.list_notifications()
    ), "no hardware escalation mail for a hung diagnostic"
    body = notice.body_text
    for claim in ISOLATION_CLAIMS:
        assert claim not in body and claim not in notice.support_case_draft, claim
    for node in rig.nodes:
        assert (
            f"  - {node}: TASK_NIC_PORT_SCOPE_UNAVAILABLE (no task NIC/RDMA port "
            "binding); node survey: scope_basis=UNKNOWN, "
            "scope_reason=TASK_RDMA_RESOURCES_UNAVAILABLE"
        ) in body, body
    bundle = next(
        item
        for item in saved.step_executions
        if item.operation is OP.COLLECT_DIAGNOSTIC_BUNDLE
    )
    archives = [
        value["evidence_ref"] for value in bundle.details["node_results"].values()
    ]
    assert archives and sorted(notice.evidence_refs) == sorted(archives), notice
    for value in bundle.details["node_results"].values():
        assert f"{value['evidence_ref']} sha256 {value['sha256']}" in body, body
    assert f"Training job: {saved.official_steps[1].parameters['job_id']}" in body
