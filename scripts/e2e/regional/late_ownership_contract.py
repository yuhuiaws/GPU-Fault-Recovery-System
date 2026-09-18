"""Wire contract for the physical STOP-to-submission acceptance boundary.

Receipts link by digest, not by comparing clocks on different machines.
Releasing the rendezvous permits a fresh product check, never a node action.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[
    str, Field(min_length=1, max_length=255, pattern=r"^[A-Za-z0-9_./:-]+$")
]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Positive = Annotated[int, Field(strict=True, gt=0)]
Nonnegative = Annotated[int, Field(strict=True, ge=0)]
Scenario = Literal["unchanged-owner", "ownership-drift", "late-sibling"]
CaseId = Literal["GF-REGIONAL-PREEMPT-033", "GF-REGIONAL-DESTR-015"]
PROTOCOL: Final = "late-ownership/v1"


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ).encode("ascii")
        ).hexdigest()


class ProcessIdentity(Document):
    pid: Positive
    start_ticks: Positive
    uid: Nonnegative
    boot_id: Identifier


class NodeIdentity(Document):
    name: Identifier
    uid: Identifier
    boot_id: Identifier


class WorkloadIdentity(Document):
    namespace: Identifier
    name: Identifier
    uid: Identifier
    owner_uid: Identifier
    attempt_id: Identifier


class Participant(Document):
    pod_uid: Identifier
    owner_uid: Identifier
    node_uid: Identifier


class AcceptanceScope(Document):
    protocol: Literal["late-ownership/v1"] = PROTOCOL
    case_id: CaseId
    scenario: Scenario
    run_id: Identifier
    challenge: Digest
    source_sha256: Digest
    release_id: Identifier
    runtime_profile: Identifier
    region: Identifier
    context: Identifier
    cluster_id: Identifier
    namespace_uid: Identifier
    executor_uid: Identifier
    workflow_id: Identifier
    incident_id: Identifier
    fencing_token: Positive
    execution_epoch: Positive
    nodes: tuple[NodeIdentity, NodeIdentity]
    workload: WorkloadIdentity
    participants: tuple[Participant, Participant]
    maintenance_start: datetime
    maintenance_end: datetime

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        if (
            self.maintenance_start.tzinfo is None
            or self.maintenance_end.tzinfo is None
            or not 0
            < (self.maintenance_end - self.maintenance_start).total_seconds()
            <= 3600
        ):
            raise ValueError("maintenance window must be aware and within one hour")
        if (
            len({item.uid for item in self.nodes}) != 2
            or len({item.name for item in self.nodes}) != 2
        ):
            raise ValueError("two distinct node identities are required")
        if (
            len({item.pod_uid for item in self.participants}) != 2
            or {item.node_uid for item in self.participants}
            != {item.uid for item in self.nodes}
            or any(
                item.owner_uid != self.workload.owner_uid for item in self.participants
            )
        ):
            raise ValueError("source participants must bind both nodes and the owner")
        return self

    def check_window(self, now: datetime) -> None:
        if (
            now.tzinfo is None
            or not self.maintenance_start <= now < self.maintenance_end
        ):
            raise ValueError("outside the approved maintenance window")


class Receipt(Document):
    protocol: Literal["late-ownership/v1"] = PROTOCOL
    scope_sha256: Digest
    producer: ProcessIdentity
    executor_uid: Identifier


class WitnessStart(Receipt):
    node: NodeIdentity
    witness_id: Identifier
    sequence: Positive
    calibration_execs: Positive
    observer: Literal["linux-exec-trace"]
    tracee: ProcessIdentity
    executable_path: Identifier
    executable_sha256: Digest


class StopReceipt(Receipt):
    boundary_id: Digest
    stop_command_id: Identifier
    queued_command_id: Identifier
    agent_generation: Positive
    agent_boundary: Literal["AGENT_PRE_SPAWN"]
    sequence: Positive
    workload: WorkloadIdentity
    participants: tuple[Participant, ...]
    absent_pod_uids: tuple[Identifier, ...]
    empty_client_node_uids: tuple[Identifier, ...]
    witness_start_sha256: tuple[Digest, Digest]
    hardware_submitted: Literal[False]
    gate_closed: Literal[True]


class MutationReceipt(Receipt):
    stop_sha256: Digest
    sequence: Positive
    workload: WorkloadIdentity
    participants: tuple[Participant, ...]
    resource_version: Identifier
    live_sibling_pod_uid: Identifier | None = None
    live_sibling_node_uid: Identifier | None = None
    sibling_gpu_client_observed: bool = False


class RecheckPermit(Document):
    scope_sha256: Digest
    boundary_id: Digest
    stop_sha256: Digest
    mutation_sha256: Digest
    instruction: Literal["RECHECK_ONLY"] = "RECHECK_ONLY"


class DecisionReceipt(Receipt):
    permit_sha256: Digest
    sequence: Positive
    decision: Literal["ALLOWED", "STOP_OWNERSHIP_DRIFT", "STOP_PARTICIPANTS_CHANGED"]
    checked_node_uids: tuple[Identifier, ...]
    hardware_submitted: bool
    restart_submitted: bool
    product_guard: Literal["kubernetes-stop-ownership/v1"]


class QuiescenceReceipt(Receipt):
    decision_sha256: Digest
    sequence: Positive
    open_commands: Nonnegative
    pending_callbacks: Nonnegative
    workflow_terminal: bool
    gate_revoked: bool


class PhysicalAction(Document):
    operation: Literal[
        "RESET_GPU", "RESET_ALL_GPUS_NVSWITCHES", "RESTART_NODE", "RESTART_WORKLOAD"
    ]
    argv_sha256: Digest
    pid: Positive
    started_ns: Positive
    ended_ns: Positive
    returncode: int

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.ended_ns < self.started_ns:
            raise ValueError("physical action interval is reversed")
        return self


class WitnessEnd(Receipt):
    node: NodeIdentity
    witness_id: Identifier
    start_sha256: Digest
    quiescence_sha256: Digest
    sequence: Positive
    tracee: ProcessIdentity
    executable_path: Identifier
    executable_sha256: Digest
    lost_events: Nonnegative
    trace_complete: bool
    trace_sha256: Digest
    trace_bytes: Positive
    exec_events: Positive
    actions: tuple[PhysicalAction, ...]


class CleanupReceipt(Receipt):
    gate_revoked: bool
    callbacks_drained: bool
    commands_terminal: bool
    owned_resources_absent: bool
    workload_identity_preserved: bool
    nodes_restored: bool


class AcceptanceEvidence(Document):
    scope: AcceptanceScope
    witness_starts: tuple[WitnessStart, WitnessStart]
    stop: StopReceipt
    mutation: MutationReceipt
    permit: RecheckPermit
    decision: DecisionReceipt
    quiescence: QuiescenceReceipt
    witness_ends: tuple[WitnessEnd, WitnessEnd]
    cleanup: CleanupReceipt
