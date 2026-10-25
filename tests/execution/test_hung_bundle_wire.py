"""What a hung bundle command carries to a node: the verdict, not the threads.

Live (GF-GENERAL-LIVE-011 ``tid_namespace``), ``bind_hung_bundle`` put the
whole triage decision -- sixteen processes' per-thread stacks and wait rows,
259 KB -- into the bundle step, the command carried it verbatim and the node's
observation request (264 KB) was refused by its 256 KiB budget before any
worker ran. The node only copies the decision into the manifest's
``fault_context``; the workflow record is where the full decision lives.
"""

from __future__ import annotations

import hashlib
import json

from gpu_fault.execution.hung_bundle import (
    WIRE_OMITTED_PATH,
    WIRE_RECORDED_IN,
    wire_decision,
    wire_parameters,
)
from gpu_fault.node_agent.observation_limits import OBSERVATION_REQUEST_BUDGET_BYTES
from tests._builders import build_store, copy_model, node_action_result
from tests._hung_decision import live_shaped_decision
from tests.execution._support import (
    NodeActionWorkflowAdapter,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStepContext,
    WorkflowStepStatus,
    workflow_state,
)


def _bytes(value: object) -> int:
    return len(json.dumps(value))


def test_the_wire_form_keeps_the_verdict_and_drops_every_process_observation():
    full = live_shaped_decision()
    assert _bytes(full) > OBSERVATION_REQUEST_BUDGET_BYTES, (
        "the live-shaped decision alone exceeds the node's request budget"
    )

    wire = wire_decision(full)

    observations = wire["evidence_branches"]["process_observations"]
    assert "processes" not in observations, observations
    assert observations["processes_omitted"] == 16, observations
    assert (
        observations["candidate_processes"]
        == (full["evidence_branches"]["process_observations"]["candidate_processes"])
    ), observations
    for key in (
        "classification",
        "confidence",
        "culprit_ranks",
        "control_ranks",
        "reason",
        "candidate_processes",
        "coverage",
        "targeted_capture_eligible",
    ):
        assert wire[key] == full[key], key
    assert (
        wire["evidence_branches"]["fr_semantics"]
        == (full["evidence_branches"]["fr_semantics"])
    ), wire["evidence_branches"]
    encoded = json.dumps(full, sort_keys=True, separators=(",", ":")).encode()
    assert wire["wire_form"] == {
        "omitted": [WIRE_OMITTED_PATH],
        "full_bytes": len(encoded),
        "full_sha256": hashlib.sha256(encoded).hexdigest(),
        "recorded_in": WIRE_RECORDED_IN,
    }, wire["wire_form"]
    assert _bytes(wire) < 16384, _bytes(wire)
    assert wire_decision(wire) == wire, "the wire form is its own fixed point"
    assert (
        "wire_form" not in full
        and "processes" in (full["evidence_branches"]["process_observations"])
    ), "the record is not modified"


def test_only_a_bundle_decision_with_process_observations_changes_shape():
    full = live_shaped_decision(processes=2, churn_rows=0)
    parameters = {
        "diagnostic_reason": "EFA_TRAFFIC_HUNG_SUSPECTED",
        "hung_triage_decision": full,
        "raw_flights_by_node": {"node-a": []},
    }

    bundle = wire_parameters(WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE, parameters)
    triage = wire_parameters(WorkflowOperation.COLLECT_HUNG_TRIAGE, parameters)

    assert bundle["hung_triage_decision"] == wire_decision(full), bundle
    assert bundle["raw_flights_by_node"] == {"node-a": []}, bundle
    assert bundle["diagnostic_reason"] == "EFA_TRAFFIC_HUNG_SUSPECTED", bundle
    assert triage == parameters, "another operation's parameters are untouched"
    verdict_only = {
        key: value for key, value in full.items() if key != "evidence_branches"
    }
    assert wire_decision(verdict_only) == verdict_only, (
        "a decision without process observations is returned as it is"
    )
    assert wire_parameters(
        WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE, {"hung_triage_decision": None}
    ) == {"hung_triage_decision": None}, "a null decision stays null"


def test_a_hung_bundle_command_carries_the_wire_form_and_the_step_keeps_the_record():
    store = build_store()
    incident, workflow = workflow_state(
        store, [WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE]
    )
    full = live_shaped_decision()
    captured = []

    def sender(endpoint, envelope):
        captured.append(envelope.command)
        return node_action_result(
            envelope.command.command_id,
            envelope.command.operation,
            details={"evidence_ref": "s3://evidence/node-a/bundle.tgz"},
        )

    step = copy_model(
        workflow.official_steps[0],
        execution_owner="gpu-fault-node-agent",
        node_ids=["node-a"],
        parameters={
            "diagnostic_reason": "EFA_TRAFFIC_HUNG_SUSPECTED",
            "hung_triage_decision": full,
            "raw_flights_by_node": {"node-a": []},
            "target_pids_by_node": {},
            "capture_process_state": False,
            "gpu_uuids_by_node": {"node-a": ["GPU-a"]},
        },
    )
    context = WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=3),
        idempotency_key="workflow/hung/bundle",
    )

    outcome = NodeActionWorkflowAdapter(
        {"node-a": "http://node-a:9099"}, "s" * 32, sender=sender
    ).execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
    assert outcome.details["node_results"]["node-a"]["evidence_ref"] == (
        "s3://evidence/node-a/bundle.tgz"
    ), outcome.details
    (command,) = captured
    sent = command.parameters
    assert sent["hung_triage_decision"] == wire_decision(full), sent
    assert (
        "processes"
        not in (
            sent["hung_triage_decision"]["evidence_branches"]["process_observations"]
        )
    ), sent
    assert {
        key: value for key, value in sent.items() if key != "hung_triage_decision"
    } == {
        key: value
        for key, value in step.parameters.items()
        if key != "hung_triage_decision"
    }, sent
    assert _bytes(command.model_dump(mode="json")) < OBSERVATION_REQUEST_BUDGET_BYTES, (
        "the command fits the node's observation request budget"
    )
    assert step.parameters["hung_triage_decision"] == full, (
        "the step record keeps the full decision"
    )
    assert context.step.parameters["hung_triage_decision"] == full, (
        "the caller's context is not rewritten"
    )
