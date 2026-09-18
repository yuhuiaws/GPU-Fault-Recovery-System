"""Strict, non-secret control protocol for the DESTR008 CPU watchdog."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

SOURCE = "destr008-cancellation-watchdog"
SOURCE_FILES = (
    "destr008_cancellation_probe.py",
    "destr008_cancellation_protocol.py",
    "destr008_cancellation_store.py",
)
MAX_DOCUMENT_BYTES = 65536
MAX_OWNED_RECORDS = 128
MAX_WINDOW_SECONDS = 7200
QUIET_SECONDS = 5
DRAIN_SECONDS = 180
MAX_CLEANUP_SECONDS = 180
Identifier = Annotated[
    str, Field(min_length=1, max_length=253, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
]
WorkloadId = Annotated[
    str, Field(min_length=1, max_length=253, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$")
]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Epoch = Annotated[int, Field(strict=True, ge=1, le=253_402_300_799)]
Schema = Annotated[int, Field(strict=True, ge=1, le=1)]
Count = Annotated[int, Field(strict=True, ge=0, le=MAX_OWNED_RECORDS)]
ProducerState = Literal["NOT_STARTED", "SUBMITTING", "ACKNOWLEDGED"]


class ProbeError(RuntimeError):
    """Only fixed codes cross the probe's output boundary."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class Record(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, hide_input_in_errors=True
    )


class Fence(Record):
    policy: Identifier
    policy_uid: Identifier
    binding: Identifier
    binding_uid: Identifier
    marker: Digest
    node: Identifier
    node_uid: Identifier


class Plan(Record):
    schema_version: Schema
    run_id: Identifier
    cluster_id: Identifier
    job_id: Identifier
    attempt_id: Identifier
    event_id: Identifier
    release_id: Identifier
    fault_node: Identifier
    spare_node: Identifier
    runtime_profile_version: Identifier
    workload_ids: Annotated[list[WorkloadId], Field(min_length=1, max_length=16)]
    probe_sha256: Digest
    created_at: Epoch
    deadline_at: Epoch
    fence: Fence

    @model_validator(mode="after")
    def bound_window(self) -> Self:
        if (
            self.fault_node == self.spare_node
            or self.fence.node != self.spare_node
            or len(set(self.workload_ids)) != len(self.workload_ids)
            or not 0 < self.deadline_at - self.created_at <= MAX_WINDOW_SECONDS
        ):
            raise ValueError("PLAN_BINDING")
        return self


class Root(Record):
    incident_id: Identifier
    workflow_request_id: Identifier


class Acknowledgement(Root):
    claim_id: Identifier
    event_id: Identifier
    completed_at: Epoch


class Producer(Record):
    state: ProducerState
    claim_id: Identifier | None
    claimed_at: Epoch | None
    ack: Acknowledgement | None

    @model_validator(mode="after")
    def complete_shape(self) -> Self:
        if self.state == "NOT_STARTED":
            valid = (
                self.claim_id is None and self.claimed_at is None and self.ack is None
            )
        else:
            valid = self.claim_id is not None and self.claimed_at is not None
            if self.state == "ACKNOWLEDGED":
                valid = (
                    valid
                    and self.ack is not None
                    and self.ack.claim_id == self.claim_id
                    and self.ack.completed_at >= (self.claimed_at or 0)
                )
            else:
                valid = valid and self.ack is None
        if not valid:
            raise ValueError("PRODUCER_SHAPE")
        return self


class Revocation(Record):
    at: Epoch
    reason: Literal["DEADLINE", "PARENT_CLOSE", "FAILURE"]
    producer_sha256: Digest
    producer_state: ProducerState


class CloseRequest(Record):
    producer_sha256: Digest
    requested_at: Epoch
    reason: Literal["NEGATIVE_TERMINAL", "NOT_STARTED"]


class Control(Record):
    schema_version: Schema
    plan_sha256: Digest
    producer: Producer
    revocation: Revocation | None
    close_request: CloseRequest | None


class Failure(Record):
    sequence: Annotated[int, Field(strict=True, ge=1)]
    at: Epoch
    code: Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")]


class CleanupAttempt(Record):
    attempt_id: Identifier
    started_at: Epoch
    deadline_at: Epoch

    @model_validator(mode="after")
    def bounded_window(self) -> Self:
        if (
            not QUIET_SECONDS
            <= self.deadline_at - self.started_at
            <= MAX_CLEANUP_SECONDS
        ):
            raise ValueError("CLEANUP_WINDOW")
        return self


class Receipt(Record):
    schema_version: Schema
    source: Literal["destr008-cancellation-watchdog"]
    plan_sha256: Digest
    configmap_uid: Identifier
    run_id: Identifier
    cluster_id: Identifier
    job_id: Identifier
    attempt_id: Identifier
    event_id: Identifier
    release_id: Identifier
    fault_node: Identifier
    spare_node: Identifier
    fence: Fence
    probe_sha256: Digest
    sequence: Annotated[int, Field(strict=True, ge=1)]
    observed_at: Epoch
    state: Literal["ARMED", "REVOKED", "QUIESCENT", "FAILED"]
    producer: Producer | None
    producer_revoked: bool
    revocation: Revocation | None
    source_complete: bool
    root: Root | None
    workflow_ids: Annotated[list[Identifier], Field(max_length=MAX_OWNED_RECORDS)]
    command_ids: Annotated[list[Identifier], Field(max_length=MAX_OWNED_RECORDS)]
    commands_active: Count | None
    workflows_active: Count | None
    pending_creation: bool | None
    inventory_sha256: Digest | None
    quiet_since: Epoch | None
    error_code: Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")] | None
    case_failed: bool
    failure: Failure | None
    cleanup: CleanupAttempt | None
    monitoring: bool
    fence_release_authorized: bool

    @model_validator(mode="after")
    def proof_shape(self) -> Self:
        if (
            self.producer_revoked != (self.revocation is not None)
            or len(set(self.workflow_ids)) != len(self.workflow_ids)
            or len(set(self.command_ids)) != len(self.command_ids)
            or (self.quiet_since is not None and self.quiet_since > self.observed_at)
            or (self.state == "FAILED") != (self.error_code is not None)
            or (self.state in {"ARMED", "REVOKED"} and not self.monitoring)
            or self.fence_release_authorized is not False
            or (self.state != "FAILED" and self.producer is None)
            or (self.state == "ARMED" and self.producer_revoked)
            or (self.state == "REVOKED" and not self.producer_revoked)
            or self.case_failed != (self.failure is not None)
            or (self.state == "FAILED" and not self.case_failed)
            or (self.failure is not None and self.failure.sequence > self.sequence)
            or (self.cleanup is not None and not self.case_failed)
        ):
            raise ValueError("RECEIPT_SHAPE")
        if self.source_complete:
            producer = self.producer
            if producer is None or producer.state == "SUBMITTING":
                raise ValueError("SOURCE_PROOF")
            if producer.state == "NOT_STARTED":
                if self.root is not None or self.workflow_ids or self.command_ids:
                    raise ValueError("SOURCE_PROOF")
            elif (
                producer.ack is None
                or self.root is None
                or producer.ack.incident_id != self.root.incident_id
                or producer.ack.workflow_request_id != self.root.workflow_request_id
                or self.root.workflow_request_id not in self.workflow_ids
            ):
                raise ValueError("SOURCE_PROOF")
        if self.state == "QUIESCENT" and (
            not self.producer_revoked
            or not self.source_complete
            or self.commands_active != 0
            or self.workflows_active != 0
            or self.pending_creation is not False
            or self.inventory_sha256 is None
            or self.quiet_since is None
            or self.observed_at - self.quiet_since < QUIET_SECONDS
            or self.monitoring
            or (
                self.cleanup is not None
                and (
                    self.observed_at >= self.cleanup.deadline_at
                    or self.quiet_since < self.cleanup.started_at
                )
            )
        ):
            raise ValueError("QUIESCENCE_PROOF")
        return self


def encode(value: BaseModel | object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", warnings="error")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: BaseModel | object) -> str:
    return hashlib.sha256(encode(value).encode("ascii")).hexdigest()


def source_sha256(directory: Path | None = None) -> str:
    root = directory or Path(__file__).resolve().parent
    return digest(
        {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES
        }
    )


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for name, item in pairs:
        if name in value:
            raise ProbeError("DUPLICATE_JSON_KEY")
        value[name] = item
    return value


RecordType = TypeVar("RecordType", bound=Record)


def decode(model: type[RecordType], text: str) -> RecordType:
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_DOCUMENT_BYTES:
        raise ProbeError("DOCUMENT_SIZE")
    try:
        value = json.loads(text, object_pairs_hook=unique_object)
        return model.model_validate(value)
    except (ValueError, RecursionError):
        raise ProbeError("DOCUMENT_SHAPE") from None


def initial_control(plan: Plan) -> Control:
    return Control(
        schema_version=1,
        plan_sha256=digest(plan),
        producer=Producer(
            state="NOT_STARTED", claim_id=None, claimed_at=None, ack=None
        ),
        revocation=None,
        close_request=None,
    )


def initial_data(plan: Plan) -> dict[str, str]:
    return {
        "plan.json": encode(plan),
        "control.json": encode(initial_control(plan)),
        "status.json": "null",
    }


def validate_control(plan: Plan, control: Control, now: int) -> None:
    producer = control.producer
    if control.plan_sha256 != digest(plan) or now < plan.created_at:
        raise ProbeError("CONTROL_BINDING")
    if producer.claimed_at is not None and not (
        plan.created_at <= producer.claimed_at < plan.deadline_at
        and producer.claimed_at <= now
    ):
        raise ProbeError("CLAIM_TIME")
    if producer.ack is not None and (
        producer.ack.event_id != plan.event_id or producer.ack.completed_at > now
    ):
        raise ProbeError("ACK_BINDING")
    revoked = control.revocation
    if revoked is not None:
        revoked_producer = producer
        if revoked.producer_state == "SUBMITTING" and producer.state == "ACKNOWLEDGED":
            revoked_producer = Producer(
                state="SUBMITTING",
                claim_id=producer.claim_id,
                claimed_at=producer.claimed_at,
                ack=None,
            )
        if (
            not plan.created_at <= revoked.at <= now
            or (revoked.reason == "DEADLINE" and revoked.at < plan.deadline_at)
            or revoked.producer_state != revoked_producer.state
            or revoked.producer_sha256 != digest(revoked_producer)
        ):
            raise ProbeError("REVOCATION_BINDING")
    close = control.close_request
    if close is not None and (
        not plan.created_at <= close.requested_at <= now
        or close.producer_sha256 != digest(producer)
        or (close.reason == "NOT_STARTED" and producer.state != "NOT_STARTED")
        or (close.reason == "NEGATIVE_TERMINAL" and producer.state != "ACKNOWLEDGED")
    ):
        raise ProbeError("CLOSE_BINDING")


def claim_submission(
    plan: Plan, control: Control, *, claim_id: str, now: int
) -> Control:
    """Parent must commit this transition with UID/resourceVersion CAS BEFORE POST."""
    validate_control(plan, control, now)
    if (
        control.revocation is not None
        or control.close_request is not None
        or control.producer.state != "NOT_STARTED"
        or now >= plan.deadline_at
    ):
        raise ProbeError("SUBMISSION_REVOKED")
    return control.model_copy(
        update={
            "producer": Producer(
                state="SUBMITTING", claim_id=claim_id, claimed_at=now, ack=None
            )
        }
    )


def acknowledge_submission(
    plan: Plan, control: Control, *, acknowledgement: Acknowledgement, now: int
) -> Control:
    """A late ACK may resolve SUBMITTING after revocation; it never permits a POST."""
    validate_control(plan, control, now)
    if (
        control.producer.state != "SUBMITTING"
        or control.producer.claim_id != acknowledgement.claim_id
        or acknowledgement.event_id != plan.event_id
        or acknowledgement.completed_at > now
    ):
        raise ProbeError("ACK_BINDING")
    producer = Producer(
        state="ACKNOWLEDGED",
        claim_id=control.producer.claim_id,
        claimed_at=control.producer.claimed_at,
        ack=acknowledgement,
    )
    return control.model_copy(update={"producer": producer})


def request_close(plan: Plan, control: Control, *, now: int) -> Control:
    validate_control(plan, control, now)
    if control.close_request is not None or control.producer.state == "SUBMITTING":
        raise ProbeError("CLOSE_BINDING")
    return control.model_copy(
        update={
            "close_request": CloseRequest(
                producer_sha256=digest(control.producer),
                requested_at=now,
                reason=(
                    "NOT_STARTED"
                    if control.producer.state == "NOT_STARTED"
                    else "NEGATIVE_TERMINAL"
                ),
            )
        }
    )


def revoke(control: Control, *, now: int, reason: str) -> Control:
    if control.revocation is not None:
        return control
    return control.model_copy(
        update={
            "revocation": Revocation.model_validate(
                {
                    "at": now,
                    "reason": reason,
                    "producer_sha256": digest(control.producer),
                    "producer_state": control.producer.state,
                }
            )
        }
    )


def validate_receipt(
    plan: Plan,
    control: Control,
    receipt: Receipt,
    *,
    uid: str,
    now: int,
    cleanup_only: bool = False,
) -> None:
    expected = receipt_base(plan, uid=uid, now=receipt.observed_at)
    if any(getattr(receipt, key) != value for key, value in expected.items()):
        raise ProbeError("RECEIPT_BINDING")
    if receipt.observed_at > now:
        raise ProbeError("CLOCK_REGRESSION")
    if receipt.revocation is not None and receipt.revocation != control.revocation:
        raise ProbeError("REVOCATION_REGRESSION")
    previous = receipt.producer
    current = control.producer
    if (
        previous is not None
        and previous.state != "NOT_STARTED"
        and (
            current.state == "NOT_STARTED"
            or current.claim_id != previous.claim_id
            or current.claimed_at != previous.claimed_at
            or (previous.ack is not None and previous != current)
        )
    ):
        raise ProbeError("PRODUCER_REGRESSION")
    if (
        not receipt.monitoring
        and previous != current
        and not (
            cleanup_only
            and previous is not None
            and previous.state == "SUBMITTING"
            and current.state == "ACKNOWLEDGED"
        )
    ):
        raise ProbeError("TERMINAL_CONTROL_CHANGED")


def receipt_base(plan: Plan, *, uid: str, now: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": SOURCE,
        "plan_sha256": digest(plan),
        "configmap_uid": uid,
        **{
            key: getattr(plan, key)
            for key in (
                "run_id",
                "cluster_id",
                "job_id",
                "attempt_id",
                "event_id",
                "release_id",
                "fault_node",
                "spare_node",
                "fence",
                "probe_sha256",
            )
        },
        "observed_at": now,
        "fence_release_authorized": False,
    }


def receipt(
    plan: Plan,
    control: Control | None,
    previous: Receipt | None,
    *,
    uid: str,
    now: int,
    state: str,
    **proof: Any,
) -> Receipt:
    revoked = control.revocation if control is not None else None
    values = {
        **receipt_base(plan, uid=uid, now=now),
        "sequence": 1 if previous is None else previous.sequence + 1,
        "state": state,
        "producer": None if control is None else control.producer,
        "producer_revoked": revoked is not None,
        "revocation": revoked,
        "source_complete": False,
        "root": None if previous is None else previous.root,
        "workflow_ids": [] if previous is None else previous.workflow_ids,
        "command_ids": [] if previous is None else previous.command_ids,
        "commands_active": None,
        "workflows_active": None,
        "pending_creation": None,
        "inventory_sha256": None,
        "quiet_since": None,
        "error_code": None,
        "case_failed": previous.case_failed if previous is not None else False,
        "failure": previous.failure if previous is not None else None,
        "cleanup": previous.cleanup if previous is not None else None,
        "monitoring": True,
        **proof,
    }
    try:
        if values["failure"] is None:
            code = (
                values["error_code"]
                if state == "FAILED"
                else ("CLEANUP_ONLY" if values["cleanup"] is not None else None)
            )
            if code is not None:
                values["failure"] = Failure(
                    sequence=values["sequence"], at=now, code=code
                )
                values["case_failed"] = True
        result = Receipt.model_validate(values)
    except ValidationError:
        raise ProbeError("RECEIPT_SHAPE") from None
    if len(encode(result)) > MAX_DOCUMENT_BYTES:
        raise ProbeError("RECEIPT_SIZE")
    return result
