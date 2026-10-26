"""GF-GENERAL-LIVE-017 ``missing_flight_recorder``: the hung chain's process capture.

The native variant of ``general_live_applogs`` stalls a two-node managed NCCL
job without a flight recorder and expects the product's independent RDMA-idle
trigger to raise the read-only hung chain *with process capture*. This module
reads that capture where the product puts it:

* the request, on every COLLECT_HUNG_TRIAGE step (``orchestration/hung_dag``
  copies the finding's ``capture_process_state`` onto the triage waves);
* the proof, in the rank rows the agents returned (py-spy ``python_stack`` or
  ``stack_series`` samples, ``process_series`` kernel-thread windows);
* the bundle's binding, which is *not* a request: ``hung_bundle.bind_hung_bundle``
  sets the COLLECT_DIAGNOSTIC_BUNDLE step's ``capture_process_state`` to the
  triage decision's targetability (``hung_classification``: a PLAUSIBLE verdict
  with 1-3 verified culprit leads and flight-recorder timing), so a stall
  without a flight recorder keeps the conservative scope -- no PIDs, no
  capture, ``operator_escalation_required`` -- and only the four binding
  fields' consistency with the decision is judged.

Live (HyperPod, 2026-10-26, release 37db33f254a1) the triage had sampled all
sixteen ranks' stacks and the decision was UNDETERMINED with no candidate, yet
the oracle, reading the bundle's binding as a request, reported "the hung chain
did not request process state capture".
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional import general_live_nccl as nccl
from scripts.e2e.regional import general_live_nccl_fr as nccl_fr
from scripts.e2e.regional import general_live_nccl_triage_view as triage_view
from scripts.e2e.regional.general_live_verdicts import step_parameters


def independent_trigger_errors(records: dict[str, Any]) -> list[str]:
    """The hung chain came from the RDMA-idle rule, with process capture.

    The chain requests process state once, in its triage wave: every
    COLLECT_HUNG_TRIAGE step carries the finding's ``capture_process_state``
    (``orchestration/hung_dag``), and the rank rows the agents returned carry
    the stacks and ``/proc`` series they took. The COLLECT_DIAGNOSTIC_BUNDLE
    step is bound *after* the triage (``execution/hung_bundle.bind_hung_bundle``)
    and its ``capture_process_state`` is the decision's targetability, not a
    request: a verified 1-3 culprit lead with flight-recorder timing
    (``hung_classification``), which a stall without a flight recorder can never
    produce, so the bundle keeps the conservative scope -- no PIDs, no capture,
    ``operator_escalation_required``. Live (HyperPod, 2026-10-26) that binding
    read as "the hung chain did not request process state capture" while the
    triage had sampled all sixteen ranks' stacks.
    """

    incident = nccl.hung_incident(records)
    if incident is None:
        return ["no EFA hung_suspected incident"]
    workflow = (records.get("workflows") or {}).get(
        str(incident.get("workflow_request_id"))
    )
    workflow = workflow if isinstance(workflow, dict) else None
    triage = [
        item for item in step_parameters(workflow or {}, "COLLECT_HUNG_TRIAGE") if item
    ]
    bundle = [
        item
        for item in step_parameters(workflow or {}, "COLLECT_DIAGNOSTIC_BUNDLE")
        if item
    ]
    if not triage and not bundle:
        return ["the hung workflow carries no diagnostic parameters"]
    errors = []
    for item in [*triage, *bundle]:
        if item.get("diagnostic_reason") != "EFA_TRAFFIC_HUNG_SUSPECTED":
            errors.append(f"diagnostic reason {item.get('diagnostic_reason')!r}")
        if item.get("detection_basis") != "ATTEMPT_SCOPED_RDMA_IDLE":
            errors.append(f"detection basis {item.get('detection_basis')!r}")
    if not triage:
        errors.append("the hung workflow has no COLLECT_HUNG_TRIAGE step")
    for item in triage:
        if item.get("capture_process_state") is not True:
            errors.append("the hung triage did not request process state capture")
    errors.extend(bundle_binding_errors(records))
    errors.extend(sampled_process_state_errors(records))
    return sorted(set(errors))


def bundle_binding_errors(records: dict[str, Any]) -> list[str]:
    """The bundle's capture flag is what its triage decision allows.

    ``bind_hung_bundle``: only a ``targeted_capture_eligible`` decision whose
    candidates carry a verified process identity binds the bundle to their PIDs
    with ``capture_process_state`` on (``identity_bound_candidates``); any other
    decision keeps the conservative scope -- no PIDs, no capture,
    ``operator_escalation_required``. The four fields must agree with each other,
    and a capture without an eligible decision contradicts the decision itself.
    """

    step = nccl_fr.bundle_step(records)
    if not step:
        return ["the hung workflow has no COLLECT_DIAGNOSTIC_BUNDLE step"]
    decision = nccl_fr.decision_of(records)
    if decision is None:
        return ["the bundle step was never bound to a hung_triage_decision"]
    parameters = step.get("parameters") or {}
    capture = parameters.get("capture_process_state") is True
    eligible = decision.get("targeted_capture_eligible") is True
    pids = parameters.get("target_pids_by_node") or {}
    errors = []
    if capture and not eligible:
        errors.append(
            "the bundle captures process state without an eligible decision "
            f"({decision.get('classification')}/{decision.get('confidence')}, "
            f"{decision.get('candidate_count')} candidate(s))"
        )
    scope = parameters.get("bundle_scope")
    if scope != (nccl_fr.TARGETED_SCOPE if capture else nccl_fr.CONSERVATIVE_SCOPE):
        errors.append(
            f"bundle scope {scope!r} contradicts capture_process_state={capture}"
        )
    if parameters.get("operator_escalation_required") is capture:
        errors.append(
            f"bundle operator_escalation_required is {capture}, the same as its "
            "capture_process_state"
        )
    if capture and not pids:
        errors.append("a targeted bundle names no PIDs")
    if not capture and pids:
        errors.append(f"a conservative bundle still targets PIDs {pids}")
    return errors


def sampled_process_state_errors(records: dict[str, Any]) -> list[str]:
    """The triage wave sampled the ranks' process state, as its rows show.

    A rank row carries the agent's py-spy sample (``python_stack`` COLLECTED or
    a collected ``stack_series`` sample) or its kernel-thread windows
    (``process_series``); a chain whose rows carry none took no process state,
    whatever its steps requested. No row at all names the node results the
    executor rejected (``node_failures``), as ``fr_errors`` does, so a result
    the node lost is not read as a node that sampled nothing.
    """

    triage = nccl.triage_ranks(records)
    if triage is None:
        return ["no COLLECT_HUNG_TRIAGE execution to read process state from"]
    rows, errors = triage
    if not rows:
        return [
            *errors,
            "COLLECT_HUNG_TRIAGE sampled no rank process",
            *(
                f"{node}: no COLLECT_HUNG_TRIAGE result reached the record: {cause}"
                for node, cause in nccl.triage_node_failures(records).items()
            ),
        ]
    sampled = [
        row
        for row in rows
        if (row.get("python_stack") or {}).get("status") == "COLLECTED"
        or triage_view.collected_stacks(row)
        or triage_view.windows(row)
    ]
    if not sampled:
        errors.append(
            f"none of the {len(rows)} sampled rank(s) carries process state "
            "(python_stack, stack_series or process_series)"
        )
    return errors
