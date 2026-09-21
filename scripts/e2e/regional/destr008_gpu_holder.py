"""Create-once GPU holder custody for DESTR008; reconstruction is cleanup-only."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from scripts.e2e.regional import destr008_controller_lock as locking
from scripts.e2e.regional.fixture_ownership import creation_document, fixture_binding
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.seeded_command_fixture import RUN_LABEL, delete_owned_resource
from scripts.e2e.regional.warm_spare_fixture import (
    GpuHolderFixture,
    WarmSpareLiveFixture,
)

CASE_ID = "GF-REGIONAL-DESTR-008"
READY_SECONDS = 300
MAX_RECORD_BYTES = 200000
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,252}")
HEX = re.compile(r"[0-9a-f]{64}")


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


class HolderRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)

    schema_version: int = Field(ge=1, le=1)
    binding: dict[str, Any]
    phase: Literal["PREPARING", "CREATING", "CREATED", "READY", "CLEANING", "CLOSED"]
    create_started: bool = False
    created_at: FiniteFloat | None = None
    deadline_at: FiniteFloat | None = None
    pod_uid: str | None = None
    pod_spec_sha256: str | None = None
    pod_ack_sha256: str | None = None
    pod_approved: bool = False
    supervision_lost: bool = False

    @model_validator(mode="after")
    def custody(self) -> Self:
        if self.create_started:
            if (
                self.created_at is None
                or self.created_at <= 0
                or self.deadline_at is None
                or self.deadline_at - self.created_at != GpuHolderFixture.HOLD_SECONDS
            ):
                raise ValueError("HOLDER_DEADLINE")
        elif any(
            value is not None
            for value in (
                self.created_at,
                self.deadline_at,
                self.pod_uid,
                self.pod_spec_sha256,
                self.pod_ack_sha256,
            )
        ):
            raise ValueError("HOLDER_UNSTARTED")
        if (
            (self.pod_uid is None) != (self.pod_spec_sha256 is None)
            or (self.pod_uid is None) != (self.pod_ack_sha256 is None)
            or (self.pod_approved and self.pod_uid is None)
        ):
            raise ValueError("HOLDER_ACK")
        if self.pod_uid is not None and (
            not IDENTIFIER.fullmatch(self.pod_uid)
            or self.pod_spec_sha256 is None
            or not HEX.fullmatch(self.pod_spec_sha256)
            or self.pod_ack_sha256 is None
            or not HEX.fullmatch(self.pod_ack_sha256)
        ):
            raise ValueError("HOLDER_ACK")
        if (
            (self.phase == "PREPARING" and self.create_started)
            or (
                self.phase == "CREATING"
                and (not self.create_started or self.pod_uid is not None)
            )
            or (self.phase in {"CREATED", "READY"} and self.pod_uid is None)
            or (self.phase == "READY" and not self.pod_approved)
            or (
                self.phase in {"CLEANING", "CLOSED"}
                and self.create_started
                and self.pod_uid is None
            )
        ):
            raise ValueError("HOLDER_PHASE")
        return self


def normal_spec(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RegionalFixtureError("holder Pod spec is not an object")
    spec = copy.deepcopy(value)
    defaults = {
        "dnsPolicy": "ClusterFirst",
        "schedulerName": "default-scheduler",
        "serviceAccountName": "default",
        "serviceAccount": "default",
        "securityContext": {},
        "hostNetwork": False,
        "hostPID": False,
        "hostIPC": False,
        "priority": 0,
        "preemptionPolicy": "PreemptLowerPriority",
        "initContainers": [],
        "ephemeralContainers": [],
        "volumes": [],
        "imagePullSecrets": [],
        "nodeSelector": {},
        "schedulingGates": [],
    }
    for key, default in defaults.items():
        if key in spec and digest(spec[key]) == digest(default):
            del spec[key]
    containers = spec.get("containers")
    if (
        not isinstance(containers, list)
        or len(containers) != 1
        or not isinstance(containers[0], dict)
    ):
        raise RegionalFixtureError("holder requires exactly one container")
    for key, default in {
        "terminationMessagePath": "/dev/termination-log",
        "terminationMessagePolicy": "File",
        "env": [],
        "envFrom": [],
        "volumeMounts": [],
        "args": [],
        "ports": [],
        "workingDir": "",
        "stdin": False,
        "stdinOnce": False,
        "tty": False,
    }.items():
        if key in containers[0] and digest(containers[0][key]) == digest(default):
            del containers[0][key]
    # The API server's ExtendedResourceToleration admission plugin (on by
    # default on EKS; live 2026-09-20) appends ``{key: <resource>, operator:
    # Exists, effect: NoSchedule}`` for every extended resource the Pod
    # requests, even when an operator=Exists toleration already covers it.
    # That toleration is admission, not a foreign mutation, so it is dropped on
    # both sides; a toleration for a resource the Pod never requested is drift.
    tolerations = spec.get("tolerations")
    if isinstance(tolerations, list):
        admitted = [
            {"key": name, "operator": "Exists", "effect": "NoSchedule"}
            for name in extended_resource_names(containers[0])
        ]
        spec["tolerations"] = [item for item in tolerations if item not in admitted]
        if not spec["tolerations"]:
            del spec["tolerations"]
    return spec


def extended_resource_names(container: dict[str, Any]) -> list[str]:
    """Extended resource names the container requests or limits.

    Mirrors Kubernetes' ``IsExtendedResourceName``: a qualified name whose
    domain is not ``kubernetes.io`` (or a subdomain of it).
    """

    resources = container.get("resources")
    names: set[str] = set()
    if isinstance(resources, dict):
        for section in ("requests", "limits"):
            values = resources.get(section)
            if isinstance(values, dict):
                names.update(str(name) for name in values)
    return sorted(
        name
        for name in names
        if "/" in name
        and not name.split("/", 1)[0].endswith("kubernetes.io")
        and not name.startswith("requests.")
    )


def fingerprint(pod: dict[str, Any]) -> str:
    body = {
        key: value for key, value in pod.items() if key not in {"metadata", "status"}
    }
    body["metadata"] = {
        key: pod["metadata"].get(key)
        for key in (
            "name",
            "namespace",
            "uid",
            "labels",
            "annotations",
            "ownerReferences",
        )
    }
    return digest(body)


class BoundedGpuHolderFixture(GpuHolderFixture):
    def __init__(
        self,
        warm: WarmSpareLiveFixture,
        *,
        node: str,
        run_id: str,
        state_directory: Path,
        node_uid: str,
        plan_sha256: str,
        release_id: str,
    ) -> None:
        for value in (node, run_id, node_uid, release_id):
            if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
                raise RegionalFixtureError("holder binding identity is invalid")
        if not isinstance(plan_sha256, str) or not HEX.fullmatch(plan_sha256):
            raise RegionalFixtureError("holder requires an approved plan digest")
        super().__init__(warm, node=node, run_id=run_id)
        self.run_id, self.node_uid = run_id, node_uid
        self.plan_sha256, self.release_id = plan_sha256, release_id
        self.path = state_directory.absolute() / (
            "destr008-holder-"
            + hashlib.sha256(run_id.encode()).hexdigest()[:20]
            + ".json"
        )
        self.binding = self.scope()
        self.record: HolderRecord | None = None
        with locking.controller_ownership(self.path):
            self.record = self.load()
            self.resumed = self.record is not None
            self.restore_deadline()

    def manifest(self) -> dict[str, Any]:
        value = super().manifest()
        value["spec"].update(
            automountServiceAccountToken=False, enableServiceLinks=False
        )
        return value

    def scope(self) -> dict[str, Any]:
        return fixture_binding(
            self.warm.regional,
            purpose="destr008-gpu-holder",
            inputs={
                "run_id": self.run_id,
                "node": self.node,
                "node_uid": self.node_uid,
                "plan_sha256": self.plan_sha256,
                "release_id": self.release_id,
                "manifest_sha256": digest(self.manifest()),
                "controller_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "journal": str(self.path),
            },
        )

    def load(self) -> HolderRecord | None:
        if not self.path.exists() and not self.path.is_symlink():
            if self.record is not None:
                raise RegionalFixtureError("holder ownership journal disappeared")
            return None
        try:
            record = HolderRecord.model_validate(
                locking.read_private_document(self.path)
            )
        except (OSError, ValueError, TypeError):
            raise RegionalFixtureError("holder ownership journal is invalid") from None
        if record.binding != self.binding:
            raise RegionalFixtureError("holder ownership binding changed")
        if record.supervision_lost:
            raise RegionalFixtureError(
                "holder supervision was lost; reconciliation required"
            )
        return record

    def restore_deadline(self) -> None:
        if self.record is not None and self.record.deadline_at is not None:
            self.deadline_at = datetime.fromtimestamp(
                self.record.deadline_at, timezone.utc
            )

    def save(self) -> None:
        if self.record is None or not locking.ownership_held(self.path):
            raise RegionalFixtureError("holder journal requires controller ownership")
        value = self.record.model_dump(mode="json", warnings="error")
        HolderRecord.model_validate(value)
        if len(json.dumps(value).encode()) > MAX_RECORD_BYTES:
            raise RegionalFixtureError("holder journal exceeds its size limit")
        write_json_atomic(self.path, value)

    @contextmanager
    def operation(self) -> Iterator[None]:
        with locking.controller_ownership(self.path):
            if self.scope() != self.binding:
                raise RegionalFixtureError(
                    "holder connection or source binding changed"
                )
            self.record = self.load()
            self.restore_deadline()
            try:
                yield
            except ProcessSupervisionLost:
                if self.record is not None:
                    self.record.supervision_lost = True
                    try:
                        self.save()
                    except Exception:
                        raise ProcessSupervisionLost(
                            "holder supervision loss could not be recorded"
                        ) from None
                raise

    def verify(self) -> None:
        if self.scope() != self.binding:
            raise RegionalFixtureError("holder connection or source binding changed")
        try:
            identity = self.warm.regional.evidence_identity()
            node = self.warm.node_snapshot(self.node)
        except Exception as exc:
            raise RegionalFixtureError(
                f"holder live identity read failed ({type(exc).__name__})"
            ) from None
        if (
            not isinstance(identity, dict)
            or identity.get("cluster_id") != self.warm.regional.settings.cluster_id
            or identity.get("release_id") != self.release_id
        ):
            raise RegionalFixtureError("holder live release or cluster changed")
        if node.get("name") != self.node or node.get("uid") != self.node_uid:
            raise RegionalFixtureError("holder Node UID changed")
        if self.scope() != self.binding:
            raise RegionalFixtureError("holder connection or source binding changed")

    def gpu(self, *args: str, stdin: bytes | None = None, **kwargs: Any) -> str:
        if not locking.ownership_held(self.path):
            raise RegionalFixtureError(
                "holder API access requires controller ownership"
            )
        self.verify()
        if args[0] == "create" and (
            self.record is None
            or self.record.deadline_at is None
            or time.time() >= self.record.deadline_at
        ):
            raise RegionalFixtureError("holder creation deadline expired")
        try:
            return self.warm.regional.kubectl(
                "gpu",
                *args,
                input_text=None if stdin is None else stdin.decode(),
                **kwargs,
            )
        except Exception as exc:
            raise RegionalFixtureError(
                f"holder API request failed ({type(exc).__name__})"
            ) from None

    def read(self) -> dict[str, Any] | None:
        raw = self.gpu(
            "get", "pod", self.name, "--ignore-not-found", "-o", "json", timeout=30
        )
        return creation_document(raw) if raw.strip() else None

    def creation_identity(self, pod: dict[str, Any]) -> tuple[str, str]:
        meta = pod.get("metadata")
        if (
            pod.get("apiVersion") != "v1"
            or pod.get("kind") != "Pod"
            or not isinstance(meta, dict)
            or meta.get("name") != self.name
            or meta.get("namespace") != self.warm.regional.settings.namespace
            or not isinstance(meta.get("uid"), str)
            or not IDENTIFIER.fullmatch(meta["uid"])
            or not isinstance(meta.get("resourceVersion"), str)
            or not meta["resourceVersion"]
        ):
            raise RegionalFixtureError("holder Pod creation identity differs")
        return meta["uid"], digest(pod.get("spec"))

    def validate(self, pod: dict[str, Any], *, acknowledged: bool) -> tuple[str, str]:
        uid, spec_sha256 = self.creation_identity(pod)
        if acknowledged:
            if (
                self.record is None
                or self.record.pod_uid != uid
                or self.record.pod_spec_sha256 != spec_sha256
                or self.record.pod_ack_sha256 != fingerprint(pod)
            ):
                raise RegionalFixtureError(
                    "holder Pod was replaced or its acknowledged spec changed"
                )
            if not self.record.pod_approved:
                return uid, spec_sha256
        meta = pod["metadata"]
        if (
            meta.get("ownerReferences") not in (None, [])
            or not isinstance(meta.get("labels"), dict)
            or meta["labels"].get(RUN_LABEL) != self.name
            or meta["labels"].get("app") != "gpu-fault-spare-holder"
            or digest(normal_spec(pod.get("spec")))
            != digest(normal_spec(self.manifest()["spec"]))
        ):
            raise RegionalFixtureError("holder Pod ownership or full spec differs")
        return uid, spec_sha256

    def create(self) -> None:
        with self.operation():
            if self.resumed or self.record is not None:
                raise RegionalFixtureError(
                    "existing holder is cleanup-only; rearm is forbidden"
                )
            self.record = HolderRecord(
                schema_version=1, binding=self.binding, phase="PREPARING"
            )
            self.save()
            if self.read() is not None:
                raise RegionalFixtureError("holder Pod name is already in use")
            now = time.time()
            self.record.phase = "CREATING"
            self.record.create_started = True
            self.record.created_at = now
            self.record.deadline_at = now + self.HOLD_SECONDS
            self.save()
            self.restore_deadline()
            raw = self.gpu(
                "create",
                "-f",
                "-",
                "-o",
                "json",
                stdin=json.dumps(self.manifest()).encode(),
                timeout=60,
            )
            created = creation_document(raw)
            uid, spec_sha256 = self.creation_identity(created)
            # A direct ACK grants deletion-only custody before execution approval.
            self.record.pod_uid, self.record.pod_spec_sha256 = uid, spec_sha256
            self.record.pod_ack_sha256 = fingerprint(created)
            self.record.phase = "CREATED"
            self.save()
            self.validate(created, acknowledged=False)
            if created["metadata"].get("deletionTimestamp") is not None:
                raise RegionalFixtureError("holder create ACK is already deleting")
            self.record.pod_approved = True
            self.save()
            deadline = time.monotonic() + READY_SECONDS
            while time.monotonic() < deadline and time.time() < self.record.deadline_at:
                current = self.read()
                if current is None:
                    raise RegionalFixtureError(
                        "acknowledged holder disappeared before readiness"
                    )
                self.validate(current, acknowledged=True)
                if (
                    time.monotonic() >= deadline
                    or time.time() >= self.record.deadline_at
                ):
                    raise RegionalFixtureError("holder readiness deadline expired")
                if current["metadata"].get("deletionTimestamp") is not None:
                    raise RegionalFixtureError("holder is deleting before readiness")
                status = current.get("status", {})
                if not isinstance(status, dict):
                    raise RegionalFixtureError("holder status is invalid")
                if status.get("phase") in ("Succeeded", "Failed", "Unknown"):
                    raise RegionalFixtureError("holder ended before readiness")
                conditions = status.get("conditions", [])
                if (
                    status.get("phase") not in (None, "Pending", "Running")
                    or not isinstance(conditions, list)
                    or any(
                        not isinstance(item, dict)
                        or not isinstance(item.get("type"), str)
                        or not item["type"]
                        or item.get("status") not in ("True", "False", "Unknown")
                        for item in conditions
                    )
                    or len({item["type"] for item in conditions}) != len(conditions)
                ):
                    raise RegionalFixtureError("holder status is invalid")
                if status.get("phase") == "Running" and any(
                    item["type"] == "Ready" and item["status"] == "True"
                    for item in conditions
                ):
                    self.record.phase = "READY"
                    self.save()
                    return
                time.sleep(1)
            raise RegionalFixtureError("holder readiness deadline expired")

    def cleanup(self) -> bool:
        with self.operation():
            if self.record is None:
                self.record = HolderRecord(
                    schema_version=1, binding=self.binding, phase="PREPARING"
                )
                self.save()
            if self.record.create_started and self.record.pod_uid is None:
                raise RegionalFixtureError(
                    "holder creation outcome is unknown; adoption is forbidden"
                )
            current = self.read()
            if self.record.phase == "CLOSED":
                if current is not None:
                    raise RegionalFixtureError("closed holder Pod name was recreated")
                return False
            if current is not None:
                if self.record.pod_uid is None:
                    raise RegionalFixtureError("holder Pod has no direct creation ACK")
                self.validate(current, acknowledged=True)
            self.record.phase = "CLEANING"
            self.save()
            if current is not None:
                if self.record.pod_approved:
                    delete_owned_resource(
                        "pod",
                        self.name,
                        self.name,
                        client=self.gpu,
                        namespace=self.warm.regional.settings.namespace,
                        expected_uid=self.record.pod_uid,
                        expected_resource_version=current["metadata"][
                            "resourceVersion"
                        ],
                        require_uid=True,
                    )
                else:
                    self.delete_unapproved(current)
            if self.read() is not None:
                raise RegionalFixtureError("holder Pod removal is unconfirmed")
            self.record.phase = "CLOSED"
            self.save()
            return False

    def delete_unapproved(self, current: dict[str, Any]) -> None:
        self.validate(current, acknowledged=True)
        meta = current["metadata"]
        options = {
            "apiVersion": "v1",
            "kind": "DeleteOptions",
            "preconditions": {
                "uid": meta["uid"],
                "resourceVersion": meta["resourceVersion"],
            },
            "propagationPolicy": "Foreground",
        }
        try:
            self.gpu(
                "delete",
                "--raw",
                f"/api/v1/namespaces/{self.warm.regional.settings.namespace}/pods/{self.name}",
                "-f",
                "-",
                stdin=json.dumps(options).encode(),
                timeout=120,
            )
        except RegionalFixtureError:
            pass
        deadline = time.monotonic() + 120
        while True:
            remaining = self.read()
            if remaining is None:
                return
            self.validate(remaining, acknowledged=True)
            if not remaining["metadata"].get("deletionTimestamp"):
                raise RegionalFixtureError(
                    "holder deletion-only request is unconfirmed"
                )
            if time.monotonic() >= deadline:
                raise RegionalFixtureError("holder deletion-only removal timed out")
            time.sleep(1)

    def resume_cleanup(self) -> bool:
        self.resumed = True
        return self.cleanup()
