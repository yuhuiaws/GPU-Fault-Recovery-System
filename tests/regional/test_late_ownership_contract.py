from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError

from scripts.e2e.regional.late_ownership_contract import (
    AcceptanceScope,
    PhysicalAction,
    ProcessIdentity,
    StopReceipt,
    WitnessStart,
)
from scripts.e2e.regional.late_ownership_verdicts import (
    acceptance_errors,
    cleanup_errors,
    mutation_errors,
    receipt_errors,
    witness_end_errors,
    witness_start_errors,
)
from tests.regional._late_ownership_support import DIGEST, evidence, scope


@pytest.mark.parametrize(
    "scenario", ["unchanged-owner", "ownership-drift", "late-sibling"]
)
def test_positive_and_adversarial_controls_have_distinct_complete_proofs(scenario):
    proof = evidence(scenario)
    assert acceptance_errors(proof) == []
    assert len(proof.witness_starts) == len(proof.witness_ends) == 2
    assert proof.decision.hardware_submitted is (scenario == "unchanged-owner")


def test_scope_roundtrip_preserves_digest_without_wall_clock_ordering():
    binding = scope()
    decoded = AcceptanceScope.model_validate_json(binding.model_dump_json())
    assert decoded == binding
    assert decoded.digest() == binding.digest()
    decoded.check_window(decoded.maintenance_start)
    with pytest.raises(ValueError, match="outside"):
        decoded.check_window(decoded.maintenance_end)
    with pytest.raises(ValueError, match="outside"):
        decoded.check_window(decoded.maintenance_start.replace(tzinfo=None))


@pytest.mark.parametrize(
    "changes",
    [
        {"maintenance_start": None},
        {"fencing_token": True},
        {"execution_epoch": 0},
        {"challenge": "not-a-digest"},
        {"cluster_id": "foreign cluster"},
        {"unknown": "field"},
    ],
)
def test_scope_rejects_ambiguous_or_unbound_input(changes):
    data = scope().model_dump()
    with pytest.raises(ValidationError):
        AcceptanceScope.model_validate(data | changes)


@pytest.mark.parametrize(
    "defect",
    [
        "naive-start",
        "naive-end",
        "expired",
        "unbounded",
        "same-node",
        "same-name",
        "same-pod",
        "foreign-owner",
        "missing-node",
    ],
)
def test_scope_requires_bounded_window_and_exact_two_node_ownership(defect):
    binding = scope()
    data = binding.model_dump()
    if defect == "naive-start":
        data["maintenance_start"] = binding.maintenance_start.replace(tzinfo=None)
    elif defect == "naive-end":
        data["maintenance_end"] = binding.maintenance_end.replace(tzinfo=None)
    elif defect == "expired":
        data["maintenance_end"] = binding.maintenance_start
    elif defect == "unbounded":
        data["maintenance_end"] = binding.maintenance_start + timedelta(hours=2)
    elif defect in {"same-node", "same-name"}:
        field = "uid" if defect == "same-node" else "name"
        data["nodes"][1][field] = data["nodes"][0][field]
    elif defect == "same-pod":
        data["participants"][1]["pod_uid"] = data["participants"][0]["pod_uid"]
    elif defect == "foreign-owner":
        data["participants"][1]["owner_uid"] = "foreign-owner"
    else:
        data["participants"][1]["node_uid"] = data["participants"][0]["node_uid"]
    with pytest.raises(ValidationError):
        AcceptanceScope.model_validate(data)


def test_physical_event_needs_ordered_interval_and_positive_process():
    with pytest.raises(ValidationError, match="reversed"):
        PhysicalAction(
            operation="RESET_GPU",
            argv_sha256=DIGEST,
            pid=1,
            started_ns=20,
            ended_ns=10,
            returncode=0,
        )
    with pytest.raises(ValidationError):
        ProcessIdentity(pid=0, start_ticks=1, uid=0, boot_id="boot")


@pytest.mark.parametrize(
    "field,value", [("hardware_submitted", True), ("gate_closed", False)]
)
def test_a_completed_model_step_is_not_a_closed_physical_stop(field, value):
    data = evidence().stop.model_dump()
    with pytest.raises(ValidationError):
        StopReceipt.model_validate(data | {field: value})


def test_model_only_counter_cannot_become_a_physical_witness():
    data = evidence().witness_starts[0].model_dump()
    with pytest.raises(ValidationError):
        WitnessStart.model_validate(data | {"observer": "model-counter", "actions": 0})


@pytest.mark.parametrize(
    "field,value,error",
    [
        (
            "scope_sha256",
            "0" * 64,
            "receipt is stale or belongs to another approved scope",
        ),
        ("executor_uid", "executor-replaced", "receipt belongs to a replaced Executor"),
    ],
)
def test_stale_and_replaced_executor_receipts_are_rejected(field, value, error):
    proof = evidence()
    receipt = proof.stop.model_copy(update={field: value})
    assert receipt_errors(proof.scope, receipt) == [error]


@pytest.mark.parametrize("field", ["stop_sha256", "producer", "sequence"])
def test_mutation_must_follow_the_acknowledged_stop(field):
    proof = evidence()
    value = {
        "stop_sha256": "0" * 64,
        "producer": proof.witness_starts[0].producer,
        "sequence": 1,
    }[field]
    mutated = proof.mutation.model_copy(update={field: value})
    assert (
        "mutation is not causally after the physical STOP callback"
        in mutation_errors(proof.scope, proof.stop, mutated)
    )


@pytest.mark.parametrize("field", ["namespace", "name", "attempt_id"])
def test_drift_control_cannot_target_another_workload(field):
    proof = evidence()
    current = proof.mutation.workload.model_copy(update={field: "outside-scope"})
    assert "mutation escaped the approved workload identity" in mutation_errors(
        proof.scope, proof.stop, proof.mutation.model_copy(update={"workload": current})
    )


@pytest.mark.parametrize("scenario", ["unchanged-owner", "ownership-drift"])
@pytest.mark.parametrize("defect", ["owner", "participants", "sibling"])
def test_owner_controls_cannot_silently_change_the_experiment(scenario, defect):
    proof = evidence(scenario)
    changes = {
        "owner": {
            "workload": proof.scope.workload
            if scenario == "ownership-drift"
            else proof.scope.workload.model_copy(update={"uid": "wrong"})
        },
        "participants": {"participants": proof.scope.participants[:1]},
        "sibling": {"live_sibling_pod_uid": "unexpected-pod"},
    }[defect]
    assert mutation_errors(
        proof.scope, proof.stop, proof.mutation.model_copy(update=changes)
    ) == [
        "positive control changed ownership or participants"
        if scenario == "unchanged-owner"
        else "ownership-drift control did not change only the pinned owner"
    ]


@pytest.mark.parametrize(
    "defect",
    [
        "workload",
        "absent",
        "old-uid",
        "short",
        "substituted-source",
        "owner",
        "node",
        "unobserved",
        "reported-node",
    ],
)
def test_late_sibling_requires_new_uid_and_physical_presence_on_other_node(defect):
    proof = evidence("late-sibling")
    mutation = proof.mutation
    if defect == "workload":
        mutation = mutation.model_copy(
            update={"workload": mutation.workload.model_copy(update={"uid": "wrong"})}
        )
    elif defect == "absent":
        mutation = mutation.model_copy(update={"live_sibling_pod_uid": None})
    elif defect == "old-uid":
        mutation = mutation.model_copy(
            update={"live_sibling_pod_uid": proof.scope.participants[0].pod_uid}
        )
    elif defect == "short":
        mutation = mutation.model_copy(
            update={"participants": proof.scope.participants}
        )
    elif defect == "substituted-source":
        participants = list(mutation.participants)
        participants[0] = participants[0].model_copy(update={"owner_uid": "other"})
        mutation = mutation.model_copy(update={"participants": tuple(participants)})
    elif defect in {"owner", "node"}:
        field = "owner_uid" if defect == "owner" else "node_uid"
        late = mutation.participants[-1].model_copy(update={field: "foreign"})
        mutation = mutation.model_copy(
            update={"participants": (*proof.scope.participants, late)}
        )
    elif defect == "unobserved":
        mutation = mutation.model_copy(update={"sibling_gpu_client_observed": False})
    else:
        mutation = mutation.model_copy(
            update={"live_sibling_node_uid": proof.scope.nodes[0].uid}
        )
    errors = mutation_errors(proof.scope, proof.stop, mutation)
    assert any(
        item in errors
        for item in (
            "late sibling is not a new participant of the original owner",
            "late sibling was not physically observed on the other participant node",
        )
    ), "late sibling identity or physical-presence validation must reject this mutation"


@pytest.mark.parametrize(
    "defect",
    [
        "missing-node",
        "same-producer",
        "self-observed",
        "wrong-observer-boot",
        "wrong-agent-boot",
    ],
)
def test_witnesses_must_independently_cover_both_real_node_incarnations(defect):
    proof = evidence()
    first, second = proof.witness_starts
    if defect == "missing-node":
        second = second.model_copy(update={"node": first.node})
    elif defect == "same-producer":
        second = second.model_copy(update={"producer": first.producer})
    elif defect == "self-observed":
        first = first.model_copy(update={"tracee": first.producer})
    elif defect == "wrong-observer-boot":
        first = first.model_copy(
            update={"producer": first.producer.model_copy(update={"boot_id": "wrong"})}
        )
    else:
        first = first.model_copy(
            update={"tracee": first.tracee.model_copy(update={"boot_id": "wrong"})}
        )
    assert witness_start_errors(proof.scope, (first, second)), (
        "witness identity drift must invalidate two-node coverage"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("start_sha256", "0" * 64),
        ("quiescence_sha256", "0" * 64),
        ("witness_id", "wrong"),
        ("sequence", 1),
        ("lost_events", 1),
        ("trace_complete", False),
        ("exec_events", 0),
    ],
)
def test_empty_actions_without_complete_continuous_trace_cannot_pass(field, value):
    proof = evidence()
    start, end = proof.witness_starts[0], proof.witness_ends[0]
    changed = end.model_copy(update={field: value})
    errors = witness_end_errors(
        proof.scope, start, changed, quiescence_sha256=proof.quiescence.digest()
    )
    assert errors, "incomplete or discontinuous tracing must prevent acceptance"


@pytest.mark.parametrize("field", ["node", "producer", "tracee"])
def test_end_receipt_rejects_replaced_node_or_witness(field):
    proof = evidence()
    changed = proof.witness_ends[0].model_copy(
        update={field: getattr(proof.witness_ends[1], field)}
    )
    assert witness_end_errors(
        proof.scope,
        proof.witness_starts[0],
        changed,
        quiescence_sha256=proof.quiescence.digest(),
    ) == ["physical witness does not continuously bind arm through quiescence"]


def test_physical_action_wins_over_a_model_no_action_decision():
    proof = evidence()
    reset = evidence("unchanged-owner").witness_ends[0].actions
    first = proof.witness_ends[0].model_copy(update={"actions": reset})
    forged = proof.model_copy(update={"witness_ends": (first, proof.witness_ends[1])})
    assert acceptance_errors(forged) == [
        "hardware execution crossed a refused STOP ownership boundary"
    ]


@pytest.mark.parametrize("defect", ["none", "duplicate", "fabric", "failed"])
def test_positive_control_requires_exact_completed_physical_reset(defect):
    proof = evidence("unchanged-owner")
    first = proof.witness_ends[0]
    action = first.actions[0]
    changes = {
        "none": (),
        "duplicate": first.actions * 2,
        "fabric": (
            action.model_copy(update={"operation": "RESET_ALL_GPUS_NVSWITCHES"}),
        ),
        "failed": (action.model_copy(update={"returncode": 1}),),
    }[defect]
    errors = witness_end_errors(
        proof.scope,
        proof.witness_starts[0],
        first.model_copy(update={"actions": changes}),
        quiescence_sha256=proof.quiescence.digest(),
    )
    assert (
        "positive control lacks exactly one completed physical reset per node" in errors
    )


@pytest.mark.parametrize(
    "field",
    [
        "gate_revoked",
        "callbacks_drained",
        "commands_terminal",
        "owned_resources_absent",
        "workload_identity_preserved",
        "nodes_restored",
    ],
)
def test_cleanup_failure_cannot_preserve_pass(field):
    proof = evidence()
    cleanup = proof.cleanup.model_copy(update={field: False})
    assert cleanup_errors(proof.scope, cleanup) == [
        "cleanup did not prove revocation, drainage and owned-resource restoration"
    ]


@pytest.mark.parametrize(
    "part,field,value",
    [
        ("stop", "participants", ()),
        ("stop", "witness_start_sha256", ("0" * 64, "1" * 64)),
        ("permit", "scope_sha256", "0" * 64),
        ("permit", "boundary_id", "0" * 64),
        ("permit", "stop_sha256", "0" * 64),
        ("permit", "mutation_sha256", "0" * 64),
        ("decision", "permit_sha256", "0" * 64),
        ("decision", "sequence", 1),
        ("decision", "decision", "ALLOWED"),
        ("decision", "checked_node_uids", ("node-uid-a",)),
        ("decision", "hardware_submitted", True),
        ("decision", "restart_submitted", True),
        ("quiescence", "decision_sha256", "0" * 64),
        ("quiescence", "sequence", 1),
        ("quiescence", "open_commands", 1),
        ("quiescence", "pending_callbacks", 1),
        ("quiescence", "workflow_terminal", False),
        ("quiescence", "gate_revoked", False),
    ],
)
def test_aggregate_verdict_rejects_every_broken_causal_stage(part, field, value):
    proof = evidence()
    changed = getattr(proof, part).model_copy(update={field: value})
    assert acceptance_errors(proof.model_copy(update={part: changed})), (
        "a broken causal stage must reject aggregate acceptance"
    )


@pytest.mark.parametrize("part", ["stop", "decision", "quiescence"])
def test_another_process_cannot_claim_the_product_decision(part):
    proof = evidence()
    changed = getattr(proof, part).model_copy(
        update={"producer": proof.witness_starts[0].producer}
    )
    assert acceptance_errors(proof.model_copy(update={part: changed})), (
        "another process must not claim the product decision"
    )
