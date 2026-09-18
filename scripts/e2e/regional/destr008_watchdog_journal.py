"""Typed private journal, history bindings, and lifecycle ownership contracts."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypedDict, Unpack

from pydantic import Field, model_validator

from scripts.e2e.regional import destr008_controller_lock as locking
from scripts.e2e.regional import destr008_watchdog_control as control_api
from scripts.e2e.regional import destr008_watchdog_resources as resources
from scripts.e2e.regional.probes import destr008_cancellation_protocol as wire
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError

Plan = wire.Plan
Receipt = wire.Receipt
CpuRuntime = resources.CpuRuntime
JsonObject = dict[str, Any]

ARM_SECONDS = 120
STOP_SECONDS = 120
MAX_JOBS = 12
MAX_DOCUMENT_BYTES = 262144
COMPATIBILITY: Literal["destr008-cancellation-v1"] = "destr008-cancellation-v1"
Kind = Literal["configmap", "serviceaccount", "role", "rolebinding", "job"]
Count = Annotated[int, Field(strict=True, ge=0)]
KINDS = {
    "configmap": ("v1", "ConfigMap", "api/v1", "configmaps"),
    "serviceaccount": ("v1", "ServiceAccount", "api/v1", "serviceaccounts"),
    "role": (
        "rbac.authorization.k8s.io/v1",
        "Role",
        "apis/rbac.authorization.k8s.io/v1",
        "roles",
    ),
    "rolebinding": (
        "rbac.authorization.k8s.io/v1",
        "RoleBinding",
        "apis/rbac.authorization.k8s.io/v1",
        "rolebindings",
    ),
    "job": ("batch/v1", "Job", "apis/batch/v1", "jobs"),
}


class ManagedResource(wire.Record):
    kind: Kind
    name: wire.Identifier
    uid: wire.Identifier | None = None
    ack_sha256: wire.Digest | None = None
    shape_sha256: wire.Digest | None = None
    approved: bool = False
    removing: bool = False
    removed: bool = False

    @model_validator(mode="after")
    def authority(self) -> Self:
        if (
            (self.uid is None) != (self.ack_sha256 is None)
            or self.approved != (self.shape_sha256 is not None)
            or ((self.approved or self.removing or self.removed) and self.uid is None)
            or (self.removed and not self.removing)
        ):
            raise ValueError("RESOURCE_AUTHORITY")
        return self


class ManagedPod(wire.Record):
    name: wire.Identifier
    uid: wire.Identifier
    ack_sha256: wire.Digest
    shape_sha256: wire.Digest | None = None
    node_name: str | None = None
    gone: bool = False


class ObserverJob(wire.Record):
    resource: ManagedResource
    expected: dict[str, Any]
    created_at: wire.Epoch
    sources: dict[str, wire.Digest]
    cleanup_id: wire.Identifier | None = None
    cleanup_seconds: Annotated[
        int, Field(strict=True, ge=0, le=wire.MAX_CLEANUP_SECONDS)
    ] = 0
    sequence_floor: Count = 0
    pod: ManagedPod | None = None
    release_requested: bool = False
    release_confirmed: bool = False
    stopped: bool = False

    @model_validator(mode="after")
    def lifecycle(self) -> Self:
        if (
            self.resource.kind != "job"
            or (self.cleanup_id is None) != (self.cleanup_seconds == 0)
            or (
                self.cleanup_id is not None
                and self.cleanup_seconds < wire.QUIET_SECONDS
            )
            or (self.release_confirmed and not self.release_requested)
            or (
                self.release_requested
                and (self.pod is None or self.pod.shape_sha256 is None)
            )
            or (self.stopped and not self.resource.removed)
            or (self.stopped and self.pod is not None and not self.pod.gone)
        ):
            raise ValueError("JOB_LIFECYCLE")
        return self


class Journal(wire.Record):
    schema_version: wire.Schema
    compatibility: Literal["destr008-cancellation-v1"]
    plan: Plan
    runtime: dict[str, Any]
    configuration: dict[str, Any]
    host_sha256: wire.Digest
    sources: dict[str, wire.Digest]
    source_data: dict[str, str]
    cleanup_only: bool
    ever_armed: bool
    closed: bool
    support: dict[str, ManagedResource]
    jobs: Annotated[list[ObserverJob], Field(max_length=MAX_JOBS)]
    last_control: wire.Control | None
    last_receipt: Receipt | None
    quiescence: Receipt | None

    @model_validator(mode="after")
    def ownership(self) -> Self:
        names = [job.resource.name for job in self.jobs]
        if (
            len(names) != len(set(names))
            or any(
                key != value.kind + "/" + value.name
                for key, value in self.support.items()
            )
            or any(value.kind == "job" for value in self.support.values())
            or (self.ever_armed and not self.jobs)
            or (
                self.closed
                and (not self.cleanup_only or any(not job.stopped for job in self.jobs))
            )
            or (self.closed and any(not item.removed for item in self.support.values()))
        ):
            raise ValueError("JOURNAL_OWNERSHIP")
        return self

    @model_validator(mode="after")
    def history(self) -> Self:
        control = self.last_control
        receipts = [
            item for item in (self.last_receipt, self.quiescence) if item is not None
        ]
        if (
            self.quiescence is not None
            and (
                self.quiescence.state != "QUIESCENT"
                or self.quiescence != self.last_receipt
            )
        ) or (self.closed and self.ever_armed and self.quiescence is None):
            raise ValueError("JOURNAL_TERMINAL_HISTORY")
        if self.ever_armed and (control is None or self.last_receipt is None):
            raise ValueError("JOURNAL_ARMED_HISTORY")
        if control is None:
            if receipts:
                raise ValueError("JOURNAL_CONTROL_HISTORY")
            return self
        owner = self.support.get("configmap/" + watchdog_name(self.plan.run_id))
        if owner is None or owner.uid is None or not owner.approved:
            raise ValueError("JOURNAL_CONTROL_OWNER")
        # Historical bindings remain valid after expiry; freshness is checked by Control.
        times = [self.plan.created_at, *(item.observed_at for item in receipts)]
        for timestamp in (
            control.producer.claimed_at,
            control.producer.ack.completed_at
            if control.producer.ack is not None
            else None,
            control.revocation.at if control.revocation is not None else None,
            control.close_request.requested_at
            if control.close_request is not None
            else None,
        ):
            if timestamp is not None:
                times.append(timestamp)
        try:
            wire.validate_control(self.plan, control, max(times))
            for receipt in receipts:
                if receipt.observed_at < self.plan.created_at:
                    raise wire.ProbeError("JOURNAL_RECEIPT_TIME")
                wire.validate_receipt(
                    self.plan,
                    control,
                    receipt,
                    uid=owner.uid,
                    now=max(times),
                    cleanup_only=True,
                )
        except wire.ProbeError:
            raise ValueError("JOURNAL_HISTORY_BINDING") from None
        return self


def object_document(value: Any) -> JsonObject:
    if not isinstance(value, dict):
        raise RegionalFixtureError("watchdog resource response is malformed")
    return value


def parse_document(text: str) -> JsonObject:
    try:
        if len(text.encode()) > MAX_DOCUMENT_BYTES:
            raise ValueError("response size")
        return object_document(json.loads(text, object_pairs_hook=wire.unique_object))
    except (ValueError, TypeError, wire.ProbeError):
        raise RegionalFixtureError("watchdog resource response is invalid") from None


def journal_path(directory: Path, run_id: str) -> Path:
    if not isinstance(run_id, str) or not run_id:
        raise RegionalFixtureError("watchdog run identity is required")
    return directory / (
        "cancellation-" + hashlib.sha256(run_id.encode()).hexdigest()[:20] + ".json"
    )


def read_journal(path: Path) -> Journal:
    try:
        return Journal.model_validate(locking.read_private_document(path))
    except (OSError, ValueError, TypeError):
        raise RegionalFixtureError(
            "watchdog private journal is invalid or unavailable"
        ) from None


def runtime_from_record(value: dict[str, Any], plan: Plan, name: str) -> CpuRuntime:
    try:
        runtime = CpuRuntime(**value)
        resources.control_manifest(plan, runtime, name)
        return runtime
    except (TypeError, ValueError):
        raise RegionalFixtureError("watchdog saved runtime is invalid") from None


def load_saved_plan(directory: Path, run_id: str) -> tuple[Plan, CpuRuntime]:
    """Read only the original plan/runtime; this does not confer submission authority."""
    path = journal_path(directory, run_id)
    with locking.controller_ownership(path):
        saved = read_journal(path)
        if saved.plan.run_id != run_id or saved.compatibility != COMPATIBILITY:
            raise RegionalFixtureError("watchdog saved plan binding differs")
        return saved.plan, runtime_from_record(
            saved.runtime, saved.plan, watchdog_name(run_id)
        )


def has_saved_plan(directory: Path, run_id: str) -> bool:
    """Return False only for absence; invalid, linked, or nonprivate journals fail closed."""
    path = journal_path(directory, run_id)
    if not path.parent.exists() and not path.parent.is_symlink():
        return False
    with locking.controller_ownership(path):
        if not path.exists() and not path.is_symlink():
            return False
        saved = read_journal(path)
        if saved.plan.run_id != run_id:
            raise RegionalFixtureError("watchdog saved plan belongs to another run")
        runtime_from_record(saved.runtime, saved.plan, watchdog_name(run_id))
        return True


def watchdog_name(run_id: str) -> str:
    return "gpu-fault-d008-cancel-" + hashlib.sha256(run_id.encode()).hexdigest()[:16]


def resource_fingerprint(
    value: JsonObject, *, control: bool = False, pod: bool = False
) -> str:
    body = copy.deepcopy(
        {key: item for key, item in value.items() if key not in {"metadata", "status"}}
    )
    metadata = object_document(value.get("metadata"))
    body["metadata"] = {
        key: metadata.get(key)
        for key in (
            "name",
            "namespace",
            "uid",
            "labels",
            "annotations",
            "ownerReferences",
        )
    }
    if control:
        data = object_document(body.get("data"))
        for key in ("control.json", "status.json"):
            if key in data:
                data[key] = "<mutable-protocol-document>"
    if pod:
        spec = object_document(body.get("spec"))
        spec.pop("nodeName", None)
        spec.pop("schedulingGates", None)
    return wire.digest(body)


class JobChanges(TypedDict, total=False):
    resource: ManagedResource
    pod: ManagedPod
    release_requested: bool
    release_confirmed: bool
    stopped: bool


@dataclass(frozen=True)
class LifecycleOwner:
    """Typed callbacks keep composed resource work on one authoritative journal."""

    plan: Plan
    runtime: CpuRuntime
    name: str
    sources: dict[str, str]
    snapshot: Callable[[], Journal]
    monotonic: Callable[[], float]
    sleep: Callable[[float], None]
    request: Callable[[tuple[str, ...], bytes | None, int], str]
    read: Callable[[str, str], JsonObject | None]
    namespace: Callable[[], None]
    runtime_check: Callable[[bool], None]
    control: Callable[[], control_api.CancellationControl]
    control_snapshot: Callable[[], control_api.ControlSnapshot]
    update_resource: Callable[[ManagedResource, int | None], None]
    update_job: Callable[[int, JobChanges], ObserverJob]
    uid: Callable[[ManagedResource], str]

    @property
    def record(self) -> Journal:
        return self.snapshot()

    def cpu(self, *args: str, stdin: bytes | None = None, timeout: int = 30) -> str:
        return self.request(args, stdin, timeout)

    def _read(self, kind: str, name: str) -> JsonObject | None:
        return self.read(kind, name)

    def _namespace(self) -> None:
        self.namespace()

    def _live_runtime(self, *, execution: bool) -> None:
        self.runtime_check(execution)

    def _control(self) -> control_api.CancellationControl:
        return self.control()

    def _control_snapshot(self) -> control_api.ControlSnapshot:
        return self.control_snapshot()

    def _put_resource(self, item: ManagedResource, index: int | None = None) -> None:
        self.update_resource(item, index)

    def _put_job(self, index: int, **changes: Unpack[JobChanges]) -> ObserverJob:
        return self.update_job(index, changes)

    def _uid(self, item: ManagedResource) -> str:
        return self.uid(item)
