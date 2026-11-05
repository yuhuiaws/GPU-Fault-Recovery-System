"""The EFA driver chain's reboot-rung oracles on records that leave the design.

``test_general_live_efa_driver.py`` grades the two GL-013 variants on the live
escalation shape and its main failures; here each oracle of
``general_live_efa_escalation`` is handed a ``node_activity`` document the
product should never write -- no finding or no workflow to grade, a record that
ran the remediation zero times or off its chain, a successor that is not the
reboot rung or plans another chain, a hand-off that isolates, reboots or
fails -- and must name the departure (or, with nothing to grade, stay silent
where the verdict is decided elsewhere).
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional import general_live_efa_escalation as esc

NODE = "hyperpod-test-node-a"
RECORD = "workflow-aaf326cf4be4f315223ef25b"
REBOOT = esc.REBOOT_SUCCESSOR_PREFIX + RECORD
SUPPORT = esc.SUPPORT_SUCCESSOR_PREFIX + RECORD
NO_CHAIN = "no efa_inventory_mismatch incident with a workflow to grade"


def _steps(*names: str) -> list[dict[str, Any]]:
    return [
        {"operation": name, "node_ids": [NODE], "workload_ids": []} for name in names
    ]


def _run(operation: str, status: str, index: int) -> dict[str, Any]:
    return {"operation": operation, "status": status, "step_index": index}


def _workflow(
    request_id: str,
    names: tuple[str, ...],
    *,
    status: str,
    executions: tuple[dict[str, Any], ...] = (),
    **extra: Any,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "incident_id": "inc-" + request_id.removeprefix("workflow-"),
        "status": status,
        "created_at": extra.pop("created_at", "2026-10-24T00:05:00+00:00"),
        "official_steps": _steps(*names),
        "step_executions": list(executions),
        **extra,
    }


def _record(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "status": "FAILED",
        "executions": (_run(esc.REMEDIATE_STEP, "FAILED", 2),),
        "incident_id": "inc-efa",
    }
    values.update(overrides)
    names = values.pop("names", esc.REMEDIATION_CHAIN)
    return _workflow(RECORD, names, **values)


def _document(*workflows: dict[str, Any], state: str = "QUARANTINED") -> dict:
    return {
        "incidents": [
            {
                "incident_id": "inc-efa",
                "event_id": f"host-{NODE}-1793084182398317-efa_inventory_mismatch-node",
                "event_type": "NODE_HEALTH",
                "state": state,
                "created_at": "2026-10-24T00:03:30+00:00",
                "node_ids": [NODE],
            }
        ],
        "workflows": list(workflows),
        "commands": [],
    }


# --------------------------------------------------------------------------- #
# Nothing to grade
# --------------------------------------------------------------------------- #
def test_without_the_efa_chain_the_shape_checks_name_it_and_the_exit_stays_silent() -> (
    None
):
    unrelated = _workflow(
        "workflow-other", ("FREEZE_EVIDENCE", esc.QUARANTINE), status="SUCCEEDED"
    )
    no_incident = {"incidents": [], "workflows": [unrelated]}
    assert esc.efa_chain(no_incident) is None
    no_workflow = _document(unrelated)
    assert esc.efa_chain(no_workflow) is None, "the finding's incident has no workflow"

    assert esc.escalation_shape_errors(no_workflow, None) == [NO_CHAIN]
    assert esc.handoff_shape_errors(no_workflow, None) == [NO_CHAIN]
    assert esc.isolation_errors(no_workflow, None) == [
        "workflow workflow-other plans QUARANTINE beyond the chain"
    ], "without a chain every isolation step is beyond it"
    assert (
        esc.exit_errors(no_workflow, None, {"taints": [], "unschedulable": True}) == []
    )
    assert esc.incident_state(no_workflow, "inc-absent") == ""
    assert esc.ladder_stopped(no_workflow) is False, (
        "a settled node without the EFA chain is no stopped ladder"
    )


# --------------------------------------------------------------------------- #
# The remediation record and its successors
# --------------------------------------------------------------------------- #
def test_the_record_must_be_one_failed_attempt_of_the_products_chain() -> None:
    record = _record(
        status="SUCCEEDED",
        names=("FREEZE_EVIDENCE", esc.REMEDIATE_STEP),
        executions=(_run(esc.REMEDIATE_STEP, "RUNNING", 2),),
    )
    document = _document(record)
    errors = esc.escalation_shape_errors(document, esc.efa_chain(document))
    assert errors[:3] == [
        f"{esc.REMEDIATE_STEP} ended 0 times in the remediation record (expected "
        "exactly one attempt)",
        f"the remediation record plans ['FREEZE_EVIDENCE', '{esc.REMEDIATE_STEP}'], "
        f"not the chain {list(esc.REMEDIATION_CHAIN)}",
        "the remediation record ended SUCCEEDED, expected FAILED (the failed step "
        "hands the node to the reboot rung)",
    ], errors


def test_a_successor_that_is_not_the_reboot_rung_or_plans_another_chain_is_named() -> (
    None
):
    support = _workflow(
        SUPPORT,
        esc.SUPPORT_HANDOFF_CHAIN,
        status="SUCCEEDED",
        created_at="2026-10-24T00:09:00+00:00",
    )
    reboot = _workflow(
        REBOOT,
        ("FREEZE_EVIDENCE", esc.RESTART_NODE),
        status="SUCCEEDED",
        official_action="REPLACE_NODE",
        executions=(_run(esc.RESTART_NODE, "SUCCEEDED", 1),),
        created_at="2026-10-24T00:10:00+00:00",
    )
    document = _document(_record(), support, reboot)
    errors = esc.escalation_shape_errors(document, esc.efa_chain(document))
    assert errors == [
        f"successor {SUPPORT} of the remediation record is not the REBOOT_NODE rung",
        f"successor {REBOOT} carries official_action REPLACE_NODE",
        f"the reboot successor plans ['FREEZE_EVIDENCE', '{esc.RESTART_NODE}'], not "
        f"the REBOOT_NODE chain {list(esc.REBOOT_SUCCESSOR_CHAIN)}",
    ]


# --------------------------------------------------------------------------- #
# The withheld escalation: verdict and hand-off
# --------------------------------------------------------------------------- #
def test_a_validation_a_bound_failed_withholds_the_reboot() -> None:
    def fabric(details: dict[str, Any], error: str = "") -> dict[str, Any]:
        return {"operation": esc.VALIDATE_STEP, "details": details, "error": error}

    assert esc.post_repair_verdict(fabric({"step_waiting_timeout_seconds": 600})) == (
        esc.WITHHELD
    )
    assert esc.post_repair_verdict(fabric({"workflow_execution_deadline": "x"})) == (
        esc.WITHHELD
    )
    assert esc.post_repair_verdict(fabric({"node_failures": {NODE: "efa 15/16"}})) == (
        esc.OBSERVED
    ), "a verdict beside the bound is still an observed failure"
    assert esc.post_repair_verdict({"operation": esc.PLUGIN_STEP}) == esc.OBSERVED
    assert esc.post_repair_verdict(fabric({})) == esc.OBSERVED


def test_the_handoff_names_every_departure_from_the_ticket_only_shape() -> None:
    record = _record(
        status="SUCCEEDED",
        executions=(
            _run(esc.REMEDIATE_STEP, "SUCCEEDED", 2),
            _run(esc.RELEASE_STEP, "SUCCEEDED", 6),
        ),
    )
    isolating = _workflow(
        SUPPORT,
        ("FREEZE_EVIDENCE", esc.QUARANTINE),
        status="FAILED",
        created_at="2026-10-24T00:09:00+00:00",
    )
    replacement = _workflow(
        "workflow-replace-node-x",
        ("FREEZE_EVIDENCE", esc.REPLACE_NODE),
        status="BLOCKED",
    )
    document = _document(record, isolating, replacement)
    errors = esc.handoff_shape_errors(document, esc.efa_chain(document))
    assert errors == [
        "the remediation record ended SUCCEEDED, expected FAILED",
        "the remediation record restored scheduling although its step failed",
        f"the hand-off {SUPPORT} plans ['FREEZE_EVIDENCE', 'QUARANTINE'], not "
        f"{list(esc.SUPPORT_HANDOFF_CHAIN)}",
        f"the hand-off {SUPPORT} plans isolation or a rung: "
        "['FREEZE_EVIDENCE', 'QUARANTINE']",
        f"the hand-off {SUPPORT} ended FAILED, expected SUCCEEDED (the ticket)",
        "REPLACE_NODE planned by ['workflow-replace-node-x']",
    ]
    # The same hand-off read twice (a re-dispatched copy of the record).
    first = _workflow(SUPPORT, esc.SUPPORT_HANDOFF_CHAIN, status="SUCCEEDED")
    copy = dict(first, created_at="2026-10-24T00:11:00+00:00")
    document = _document(_record(), first, copy)
    assert esc.handoff_shape_errors(document, esc.efa_chain(document)) == [
        "2 support-after successors of the remediation record (expected exactly "
        f"one): {[SUPPORT, SUPPORT]}"
    ]
    alone = _document(_record())
    assert esc.handoff_shape_errors(alone, esc.efa_chain(alone)) == [
        "0 support-after successors of the remediation record (expected exactly "
        "one): []"
    ]


# --------------------------------------------------------------------------- #
# The Node Agent's result
# --------------------------------------------------------------------------- #
def test_the_node_result_is_read_from_the_command_of_the_named_operation() -> None:
    document = _document(_record())
    document["commands"] = [
        {
            "step_index": 3,
            "step": {"operation": esc.PLUGIN_STEP},
            "result_details": {"node_results": {NODE: {"restarted": True}}},
        },
        {
            "step_index": 2,
            "step": {"operation": esc.REMEDIATE_STEP},
            "result_details": {"node_results": {NODE: {"rebound_pci_bdfs": []}}},
        },
    ]
    workflow = document["workflows"][0]
    assert esc.node_result(
        document, workflow, operation=esc.REMEDIATE_STEP, node=NODE
    ) == {"rebound_pci_bdfs": []}, "the plugin restart's receipt is another step's"
    assert esc.node_result(document, workflow, operation="QUARANTINE", node=NODE) == {}
