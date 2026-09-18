from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from scripts.e2e.regional.probes import destr008_cancellation_protocol as wire
from tests.regional.test_destr008_cancellation_probe import NOW, plan


def acknowledgement(**changes: Any) -> wire.Acknowledgement:
    return wire.Acknowledgement.model_validate(
        {
            "claim_id": "claim-a",
            "event_id": "destr008-event",
            "incident_id": "incident-a",
            "workflow_request_id": "workflow-a",
            "completed_at": NOW + 2,
            **changes,
        }
    )


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"schema_version": "1"},
        {"deadline_at": NOW},
        {"deadline_at": NOW + wire.MAX_WINDOW_SECONDS + 1},
        {"created_at": 0},
        {"deadline_at": float("nan")},
        {"run_id": "../wrong"},
        {"run_id": ""},
        {"probe_sha256": "xyz"},
        {"release_id": "x" * 254},
        {"fault_node": "spare-a"},
        {"spare_node": "foreign"},
        {"workload_ids": []},
        {"workload_ids": ["training/a", "training/a"]},
        {"workload_ids": ["x"] * 17},
        {"workload_ids": ["unsafe\nid"]},
        {"unexpected": "value"},
    ],
)
def test_plan_rejects_unknown_or_unbound_shape(change: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        plan(**change)


@pytest.mark.parametrize(
    "change",
    [
        {"policy": ""},
        {"policy_uid": ""},
        {"binding": ""},
        {"binding_uid": None},
        {"marker": "a" * 63},
        {"node_uid": "with/slash"},
        {"extra": "value"},
    ],
)
def test_fence_is_the_exact_seven_field_identity(change: dict[str, Any]) -> None:
    values = plan().fence.model_dump()
    with pytest.raises(ValidationError):
        wire.Fence.model_validate({**values, **change})


@pytest.mark.parametrize(
    "producer",
    [
        {"state": "UNKNOWN", "claim_id": None, "claimed_at": None, "ack": None},
        {"state": "NOT_STARTED", "claim_id": "claim", "claimed_at": None, "ack": None},
        {"state": "NOT_STARTED", "claim_id": None, "claimed_at": NOW, "ack": None},
        {
            "state": "NOT_STARTED",
            "claim_id": None,
            "claimed_at": None,
            "ack": acknowledgement(),
        },
        {"state": "SUBMITTING", "claim_id": None, "claimed_at": NOW, "ack": None},
        {"state": "SUBMITTING", "claim_id": "claim-a", "claimed_at": None, "ack": None},
        {
            "state": "SUBMITTING",
            "claim_id": "claim-a",
            "claimed_at": NOW,
            "ack": acknowledgement(),
        },
        {
            "state": "ACKNOWLEDGED",
            "claim_id": "claim-a",
            "claimed_at": NOW,
            "ack": None,
        },
        {
            "state": "ACKNOWLEDGED",
            "claim_id": "wrong",
            "claimed_at": NOW,
            "ack": acknowledgement(),
        },
        {
            "state": "ACKNOWLEDGED",
            "claim_id": "claim-a",
            "claimed_at": NOW + 3,
            "ack": acknowledgement(),
        },
    ],
)
def test_producer_requires_a_consistent_source_record(producer: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        wire.Producer.model_validate(producer)


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ('{"schema_version":1,"schema_version":1}', "DUPLICATE_JSON_KEY"),
        ("not-json-private-value", "DOCUMENT_SHAPE"),
        ("[]", "DOCUMENT_SHAPE"),
        ("null", "DOCUMENT_SHAPE"),
        ('{"schema_version":NaN}', "DOCUMENT_SHAPE"),
        ("x" * (wire.MAX_DOCUMENT_BYTES + 1), "DOCUMENT_SIZE"),
        (None, "DOCUMENT_SIZE"),
        ("[" * 2000, "DOCUMENT_SHAPE"),
    ],
)
def test_decode_refuses_invalid_json_without_echoing_it(text: Any, code: str) -> None:
    with pytest.raises(wire.ProbeError) as caught:
        wire.decode(wire.Control, text)
    assert str(caught.value) == code
    assert "private-value" not in str(caught.value)


def test_source_and_plan_hashes_are_canonical_and_data_round_trips(tmp_path) -> None:
    for name in wire.SOURCE_FILES:
        (tmp_path / name).write_bytes(name.encode("ascii"))
    original = wire.source_sha256(tmp_path)
    (tmp_path / wire.SOURCE_FILES[-1]).write_bytes(b"changed-helper")
    assert wire.source_sha256(tmp_path) != original, (
        "helpers are part of source identity"
    )
    bound = plan()
    data = wire.initial_data(bound)
    assert set(data) == {"plan.json", "control.json", "status.json"}
    assert wire.decode(wire.Plan, data["plan.json"]) == bound
    assert wire.decode(wire.Control, data["control.json"]) == wire.initial_control(
        bound
    )
    assert data["status.json"] == "null"
    assert wire.digest({"a": 1, "b": 2}) == wire.digest({"b": 2, "a": 1})
    assert wire.encode({"unicode": "\u00e9"}) == '{"unicode":"\\u00e9"}'


def test_single_claim_and_late_source_ack_do_not_reopen_revocation() -> None:
    bound = plan()
    initial = wire.initial_control(bound)
    submitted = wire.claim_submission(bound, initial, claim_id="claim-a", now=NOW + 1)
    with pytest.raises(wire.ProbeError, match="SUBMISSION_REVOKED"):
        wire.claim_submission(bound, submitted, claim_id="claim-b", now=NOW + 2)
    revoked = wire.revoke(submitted, now=bound.deadline_at, reason="DEADLINE")
    assert wire.revoke(revoked, now=bound.deadline_at + 1, reason="FAILURE") == revoked
    acknowledged = wire.acknowledge_submission(
        bound, revoked, acknowledgement=acknowledgement(), now=bound.deadline_at + 1
    )
    wire.validate_control(bound, acknowledged, bound.deadline_at + 1)
    assert acknowledged.revocation == revoked.revocation
    with pytest.raises(wire.ProbeError, match="SUBMISSION_REVOKED"):
        wire.claim_submission(
            bound, acknowledged, claim_id="late", now=bound.deadline_at + 1
        )
    closed = wire.request_close(bound, acknowledged, now=bound.deadline_at + 2)
    assert closed.close_request is not None
    assert closed.close_request.producer_sha256 == wire.digest(acknowledged.producer)
    assert closed.close_request.reason == "NEGATIVE_TERMINAL"
    with pytest.raises(wire.ProbeError, match="CLOSE_BINDING"):
        wire.request_close(bound, closed, now=bound.deadline_at + 3)


@pytest.mark.parametrize("kind", ["closed", "expired", "revoked"])
def test_never_submitted_tombstone_cannot_be_claimed(kind: str) -> None:
    bound = plan()
    control = wire.initial_control(bound)
    now = NOW + 1
    if kind == "closed":
        control = wire.request_close(bound, control, now=now)
    elif kind == "expired":
        now = bound.deadline_at
    else:
        control = wire.revoke(control, now=now, reason="FAILURE")
    with pytest.raises(wire.ProbeError, match="SUBMISSION_REVOKED"):
        wire.claim_submission(bound, control, claim_id="claim-a", now=now)


@pytest.mark.parametrize(
    "change",
    [{"claim_id": "different"}, {"event_id": "different"}, {"completed_at": NOW + 20}],
)
def test_ack_rejects_wrong_claim_source_or_future_completion(
    change: dict[str, Any],
) -> None:
    bound = plan()
    claimed = wire.claim_submission(
        bound, wire.initial_control(bound), claim_id="claim-a", now=NOW + 1
    )
    with pytest.raises(wire.ProbeError, match="ACK_BINDING"):
        wire.acknowledge_submission(
            bound, claimed, acknowledgement=acknowledgement(**change), now=NOW + 3
        )
    with pytest.raises(wire.ProbeError, match="CLOSE_BINDING"):
        wire.request_close(bound, claimed, now=NOW + 3)


def test_ack_without_claim_is_refused() -> None:
    bound = plan()
    with pytest.raises(wire.ProbeError, match="ACK_BINDING"):
        wire.acknowledge_submission(
            bound,
            wire.initial_control(bound),
            acknowledgement=acknowledgement(),
            now=NOW + 3,
        )


@pytest.mark.parametrize(
    ("part", "change", "code"),
    [
        ("control", {"plan_sha256": "b" * 64}, "CONTROL_BINDING"),
        ("producer", {"claimed_at": NOW - 1}, "CLAIM_TIME"),
        ("producer", {"claimed_at": NOW + 101}, "CLAIM_TIME"),
        ("producer", {"claimed_at": NOW + 5}, "CLAIM_TIME"),
        ("ack", {"event_id": "wrong"}, "ACK_BINDING"),
        ("ack", {"completed_at": NOW + 5}, "ACK_BINDING"),
        ("revocation", {"at": NOW - 1}, "REVOCATION_BINDING"),
        ("revocation", {"at": NOW + 5}, "REVOCATION_BINDING"),
        ("revocation", {"reason": "DEADLINE"}, "REVOCATION_BINDING"),
        ("revocation", {"producer_state": "NOT_STARTED"}, "REVOCATION_BINDING"),
        ("revocation", {"producer_sha256": "b" * 64}, "REVOCATION_BINDING"),
        ("close", {"requested_at": NOW - 1}, "CLOSE_BINDING"),
        ("close", {"requested_at": NOW + 5}, "CLOSE_BINDING"),
        ("close", {"producer_sha256": "b" * 64}, "CLOSE_BINDING"),
        ("close", {"reason": "NOT_STARTED"}, "CLOSE_BINDING"),
    ],
)
def test_control_shape_does_not_substitute_for_source_binding(
    part: str, change: dict[str, Any], code: str
) -> None:
    bound = plan()
    claimed = wire.claim_submission(
        bound, wire.initial_control(bound), claim_id="claim-a", now=NOW + 1
    )
    acknowledged = wire.acknowledge_submission(
        bound, claimed, acknowledgement=acknowledgement(), now=NOW + 2
    )
    control = wire.request_close(
        bound, wire.revoke(acknowledged, now=NOW + 2, reason="FAILURE"), now=NOW + 2
    )
    if part == "control":
        control = control.model_copy(update=change)
    elif part == "producer":
        control = control.model_copy(
            update={"producer": control.producer.model_copy(update=change)}
        )
    elif part == "ack":
        altered = acknowledgement().model_copy(update=change)
        control = control.model_copy(
            update={"producer": control.producer.model_copy(update={"ack": altered})}
        )
    elif part == "revocation":
        control = control.model_copy(
            update={"revocation": control.revocation.model_copy(update=change)}
        )
    else:
        control = control.model_copy(
            update={"close_request": control.close_request.model_copy(update=change)}
        )
    with pytest.raises(wire.ProbeError, match=code):
        wire.validate_control(bound, control, NOW + 3)


def test_clock_and_incomplete_close_fail_closed() -> None:
    bound = plan()
    initial = wire.initial_control(bound)
    with pytest.raises(wire.ProbeError, match="CONTROL_BINDING"):
        wire.validate_control(bound, initial, NOW - 1)
    invalid = initial.model_copy(
        update={
            "close_request": wire.CloseRequest(
                producer_sha256=wire.digest(initial.producer),
                requested_at=NOW,
                reason="NEGATIVE_TERMINAL",
            )
        }
    )
    with pytest.raises(wire.ProbeError, match="CLOSE_BINDING"):
        wire.validate_control(bound, invalid, NOW)


def test_revocation_binds_the_original_submitting_claim_even_after_ack() -> None:
    bound = plan()
    submitted = wire.claim_submission(
        bound, wire.initial_control(bound), claim_id="claim-a", now=NOW + 1
    )
    revoked = wire.revoke(submitted, now=NOW + 2, reason="FAILURE")
    acknowledged = wire.acknowledge_submission(
        bound, revoked, acknowledgement=acknowledgement(), now=NOW + 2
    )
    changed = acknowledged.model_copy(
        update={
            "producer": acknowledged.producer.model_copy(update={"claimed_at": NOW})
        }
    )
    with pytest.raises(wire.ProbeError, match="REVOCATION_BINDING"):
        wire.validate_control(bound, changed, NOW + 3)


def test_receipt_binding_and_monotonicity() -> None:
    bound = plan()
    initial = wire.initial_control(bound)
    armed = wire.receipt(bound, initial, None, uid="cm-uid", now=NOW, state="ARMED")
    wire.validate_receipt(bound, initial, armed, uid="cm-uid", now=NOW)
    with pytest.raises(wire.ProbeError, match="RECEIPT_BINDING"):
        wire.validate_receipt(bound, initial, armed, uid="foreign", now=NOW)
    with pytest.raises(wire.ProbeError, match="CLOCK_REGRESSION"):
        wire.validate_receipt(bound, initial, armed, uid="cm-uid", now=NOW - 1)
    submitted = wire.claim_submission(bound, initial, claim_id="claim-a", now=NOW + 1)
    active = wire.receipt(
        bound, submitted, armed, uid="cm-uid", now=NOW + 1, state="ARMED"
    )
    for producer in [
        initial.producer,
        submitted.producer.model_copy(update={"claim_id": "wrong"}),
        submitted.producer.model_copy(update={"claimed_at": NOW}),
    ]:
        with pytest.raises(wire.ProbeError, match="PRODUCER_REGRESSION"):
            wire.validate_receipt(
                bound,
                submitted.model_copy(update={"producer": producer}),
                active,
                uid="cm-uid",
                now=NOW + 2,
            )
    revoked = wire.revoke(submitted, now=NOW + 2, reason="FAILURE")
    failed = wire.receipt(
        bound,
        revoked,
        active,
        uid="cm-uid",
        now=NOW + 2,
        state="FAILED",
        error_code="LOCAL_FAILURE",
        monitoring=False,
    )
    with pytest.raises(wire.ProbeError, match="REVOCATION_REGRESSION"):
        wire.validate_receipt(bound, submitted, failed, uid="cm-uid", now=NOW + 3)
    acknowledged = wire.acknowledge_submission(
        bound, revoked, acknowledgement=acknowledgement(), now=NOW + 3
    )
    with pytest.raises(wire.ProbeError, match="TERMINAL_CONTROL_CHANGED"):
        wire.validate_receipt(bound, acknowledged, failed, uid="cm-uid", now=NOW + 3)
    ack_receipt = wire.receipt(
        bound, acknowledged, active, uid="cm-uid", now=NOW + 3, state="REVOKED"
    )
    with pytest.raises(wire.ProbeError, match="PRODUCER_REGRESSION"):
        wire.validate_receipt(bound, revoked, ack_receipt, uid="cm-uid", now=NOW + 4)


@pytest.mark.parametrize(
    "proof",
    [
        {"producer_revoked": True},
        {"workflow_ids": ["a", "a"]},
        {"command_ids": ["a", "a"]},
        {"quiet_since": NOW + 1},
        {"error_code": "UNEXPECTED"},
        {"monitoring": False},
        {"commands_active": False},
        {"state": "FOREIGN"},
        {"state": "QUIESCENT"},
        {"fence_release_authorized": True},
    ],
)
def test_receipts_cannot_assert_quiescence_or_unknown_state(
    proof: dict[str, Any],
) -> None:
    bound = plan()
    values = {"state": "ARMED", **proof}
    with pytest.raises(wire.ProbeError, match="RECEIPT_SHAPE"):
        wire.receipt(
            bound, wire.initial_control(bound), None, uid="cm-uid", now=NOW, **values
        )
