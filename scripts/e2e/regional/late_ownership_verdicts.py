"""Fail-closed verdicts over independent, causally linked acceptance receipts."""

from __future__ import annotations

from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied, check_stop
from scripts.e2e.regional.late_ownership_contract import (
    AcceptanceEvidence,
    AcceptanceScope,
    CleanupReceipt,
    DecisionReceipt,
    MutationReceipt,
    QuiescenceReceipt,
    RecheckPermit,
    Receipt,
    StopReceipt,
    WitnessEnd,
    WitnessStart,
)


def receipt_errors(scope: AcceptanceScope, receipt: Receipt) -> list[str]:
    errors = []
    if receipt.scope_sha256 != scope.digest():
        errors.append("receipt is stale or belongs to another approved scope")
    if receipt.executor_uid != scope.executor_uid:
        errors.append("receipt belongs to a replaced Executor")
    return errors


def mutation_errors(
    scope: AcceptanceScope, stop: StopReceipt, mutation: MutationReceipt
) -> list[str]:
    errors = receipt_errors(scope, mutation)
    if (
        mutation.stop_sha256 != stop.digest()
        or mutation.producer != stop.producer
        or mutation.sequence <= stop.sequence
    ):
        errors.append("mutation is not causally after the physical STOP callback")
    source = scope.workload
    current = mutation.workload
    if (
        current.namespace != source.namespace
        or current.name != source.name
        or current.attempt_id != source.attempt_id
    ):
        errors.append("mutation escaped the approved workload identity")
    unchanged = current == source and mutation.participants == stop.participants
    if scope.scenario == "unchanged-owner":
        if not unchanged or mutation.live_sibling_pod_uid is not None:
            errors.append("positive control changed ownership or participants")
    elif scope.scenario == "ownership-drift":
        if (
            (current.uid == source.uid and current.owner_uid == source.owner_uid)
            or mutation.participants != stop.participants
            or mutation.live_sibling_pod_uid is not None
        ):
            errors.append(
                "ownership-drift control did not change only the pinned owner"
            )
    else:
        source_uids = {item.pod_uid for item in scope.participants}
        current_uids = {item.pod_uid for item in mutation.participants}
        sibling = mutation.live_sibling_pod_uid
        if (
            current != source
            or sibling is None
            or sibling in source_uids
            or len(mutation.participants) != len(scope.participants) + 1
            or current_uids != source_uids | {sibling}
            or any(item not in mutation.participants for item in scope.participants)
        ):
            errors.append("late sibling is not a new participant of the original owner")
        late = [item for item in mutation.participants if item.pod_uid == sibling]
        if (
            len(late) != 1
            or late[0].owner_uid != source.owner_uid
            or late[0].node_uid != scope.nodes[1].uid
            or mutation.live_sibling_node_uid != scope.nodes[1].uid
            or not mutation.sibling_gpu_client_observed
        ):
            errors.append(
                "late sibling was not physically observed on the other participant node"
            )
    return errors


def witness_start_errors(
    scope: AcceptanceScope, starts: tuple[WitnessStart, WitnessStart]
) -> list[str]:
    errors = []
    if {item.node for item in starts} != set(scope.nodes):
        errors.append("physical witnesses do not cover exactly the two approved nodes")
    if starts[0].producer == starts[1].producer:
        errors.append("node witnesses are not independent processes")
    for item in starts:
        errors.extend(receipt_errors(scope, item))
        if (
            item.producer == item.tracee
            or item.producer.boot_id != item.node.boot_id
            or item.tracee.boot_id != item.node.boot_id
        ):
            errors.append("witness is not an independent observer of the pinned Agent")
    return errors


def witness_end_errors(
    scope: AcceptanceScope,
    start: WitnessStart,
    end: WitnessEnd,
    *,
    quiescence_sha256: str,
) -> list[str]:
    errors = receipt_errors(scope, end)
    if (
        end.start_sha256 != start.digest()
        or end.quiescence_sha256 != quiescence_sha256
        or end.node != start.node
        or end.producer != start.producer
        or end.tracee != start.tracee
        or end.executable_path != start.executable_path
        or end.executable_sha256 != start.executable_sha256
        or end.witness_id != start.witness_id
        or end.sequence <= start.sequence
    ):
        errors.append(
            "physical witness does not continuously bind arm through quiescence"
        )
    if end.lost_events != 0 or not end.trace_complete:
        errors.append("physical witness lost tracing coverage")
    if end.exec_events < start.calibration_execs + len(end.actions):
        errors.append(
            "physical trace does not account for calibration and action execs"
        )
    if scope.scenario == "unchanged-owner":
        if (
            len(end.actions) != 1
            or end.actions[0].operation != "RESET_GPU"
            or end.actions[0].returncode != 0
        ):
            errors.append(
                "positive control lacks exactly one completed physical reset per node"
            )
    elif end.actions:
        errors.append("hardware execution crossed a refused STOP ownership boundary")
    return errors


def cleanup_errors(scope: AcceptanceScope, cleanup: CleanupReceipt) -> list[str]:
    errors = receipt_errors(scope, cleanup)
    if not all(
        (
            cleanup.gate_revoked,
            cleanup.callbacks_drained,
            cleanup.commands_terminal,
            cleanup.owned_resources_absent,
            cleanup.workload_identity_preserved,
            cleanup.nodes_restored,
        )
    ):
        errors.append(
            "cleanup did not prove revocation, drainage and owned-resource restoration"
        )
    return errors


def decision_errors(
    scope: AcceptanceScope,
    stop: StopReceipt,
    mutation: MutationReceipt,
    permit: RecheckPermit,
    decision: DecisionReceipt,
) -> list[str]:
    errors = receipt_errors(scope, decision)
    if (
        permit.scope_sha256 != scope.digest()
        or permit.boundary_id != stop.boundary_id
        or permit.stop_sha256 != stop.digest()
        or permit.mutation_sha256 != mutation.digest()
        or decision.permit_sha256 != permit.digest()
        or decision.producer != stop.producer
        or decision.sequence <= mutation.sequence
    ):
        errors.append("product recheck did not follow the exact STOP-linked mutation")
    expected_decision = {
        "unchanged-owner": "ALLOWED",
        "ownership-drift": "STOP_OWNERSHIP_DRIFT",
        "late-sibling": "STOP_PARTICIPANTS_CHANGED",
    }[scope.scenario]
    if decision.decision != expected_decision:
        errors.append("product guard returned the wrong decision for the control")
    if set(decision.checked_node_uids) != {item.uid for item in scope.nodes} or len(
        decision.checked_node_uids
    ) != len(scope.nodes):
        errors.append("product guard did not recheck all participants")
    allowed = scope.scenario == "unchanged-owner"
    if decision.hardware_submitted != allowed or (
        not allowed and decision.restart_submitted
    ):
        errors.append("product submission/restart contradicts the ownership decision")
    return errors


def quiescence_errors(
    scope: AcceptanceScope,
    stop: StopReceipt,
    decision: DecisionReceipt,
    quiet: QuiescenceReceipt,
) -> list[str]:
    errors = receipt_errors(scope, quiet)
    if (
        quiet.decision_sha256 != decision.digest()
        or quiet.producer != stop.producer
        or quiet.sequence <= decision.sequence
        or quiet.open_commands != 0
        or quiet.pending_callbacks != 0
        or not quiet.workflow_terminal
        or not quiet.gate_revoked
    ):
        errors.append(
            "action/callback quiescence was not established before closing witnesses"
        )
    return errors


def acceptance_errors(evidence: AcceptanceEvidence) -> list[str]:
    scope, stop, decision = evidence.scope, evidence.stop, evidence.decision
    errors = witness_start_errors(scope, evidence.witness_starts)
    errors.extend(receipt_errors(scope, stop))
    try:
        check_stop(scope, stop)
    except BoundaryDenied as exc:
        errors.append(str(exc))
    starts = evidence.witness_starts
    if stop.witness_start_sha256 != tuple(item.digest() for item in starts):
        errors.append("STOP was not held behind both armed physical witnesses")
    if any(
        item.producer == stop.producer or item.tracee == stop.producer
        for item in starts
    ):
        errors.append("STOP and physical-action evidence came from the same process")
    errors.extend(mutation_errors(scope, stop, evidence.mutation))
    errors.extend(
        decision_errors(scope, stop, evidence.mutation, evidence.permit, decision)
    )
    quiet = evidence.quiescence
    errors.extend(quiescence_errors(scope, stop, decision, quiet))
    for start, end in zip(starts, evidence.witness_ends, strict=True):
        errors.extend(
            witness_end_errors(scope, start, end, quiescence_sha256=quiet.digest())
        )
    errors.extend(cleanup_errors(scope, evidence.cleanup))
    return errors
