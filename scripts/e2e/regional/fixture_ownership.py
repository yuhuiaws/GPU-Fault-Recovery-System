"""Optional durable creation custody for resumable acceptance fixtures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

from kubernetes.utils.quantity import parse_quantity
from pydantic import BaseModel, ConfigDict, Field, model_validator

from gpu_fault.admin.atomic_json import write_json_atomic
from scripts.e2e.regional.destr008_controller_lock import (
    controller_ownership,
    host_identity,
    read_private_document,
)
from scripts.e2e.regional.live_driver_guard import connection_identity
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureError,
    RegionalLiveFixture,
)


class Creation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    expected: dict[str, Any]
    uid: str | None = Field(default=None, min_length=1)
    ack_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    approved: bool = False

    @model_validator(mode="after")
    def authority(self) -> Creation:
        if (self.uid is None) != (self.ack_sha256 is None) or (
            self.approved and self.uid is None
        ):
            raise ValueError("fixture acknowledgement authority is incomplete")
        return self


class CleanupCustody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    uid: str = Field(min_length=1)
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    owner_references: list[dict[str, Any]] = Field(default_factory=list)


class OwnershipRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: int = Field(ge=1, le=1)
    binding: dict[str, Any]
    owner: str = Field(pattern=r"^[0-9a-f]{32}$")
    started: bool = False
    completed: bool = False
    creations: dict[str, Creation] = Field(default_factory=dict)
    observed: dict[str, str] = Field(default_factory=dict)
    cleanup_custody: dict[str, CleanupCustody] = Field(default_factory=dict)


def fixture_binding(
    regional: RegionalLiveFixture, *, purpose: str, inputs: dict[str, Any]
) -> dict[str, Any]:
    settings = regional.settings
    connections = connection_identity(argparse.Namespace(**asdict(settings)), {})
    if len(connections) != 2 or any(
        set(value) != {"path", "sha256"} for value in connections.values()
    ):
        raise RegionalFixtureError("fixture connection identity is incomplete")
    return {
        "purpose": purpose,
        "inputs": inputs,
        "connections": connections,
        "cluster_id": settings.cluster_id,
        "namespace": settings.namespace,
        "gpu_context": settings.gpu_context,
        "region": settings.region,
        "host_sha256": host_identity(),
    }


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(document: dict[str, Any]) -> tuple[str, str, str]:
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        raise RegionalFixtureError("fixture creation identity is incomplete")
    kind, name, uid = (
        document.get("kind"),
        metadata.get("name"),
        metadata.get("uid"),
    )
    if any(not isinstance(value, str) or not value for value in (kind, name, uid)):
        raise RegionalFixtureError("fixture creation identity is incomplete")
    return str(kind).lower(), str(name), str(uid)


def _stable(document: dict[str, Any], *, intent: bool = False) -> dict[str, Any]:
    result = copy.deepcopy(document)
    result.pop("status", None)
    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        for key in (
            "resourceVersion",
            "generation",
            "creationTimestamp",
            "deletionTimestamp",
            "deletionGracePeriodSeconds",
            "managedFields",
        ):
            metadata.pop(key, None)
        if intent:
            metadata.pop("uid", None)
    return result


def _fingerprint(document: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(_stable(document), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _cleanup_fingerprint(document: dict[str, Any]) -> str:
    stable = _stable(document)
    metadata = stable["metadata"]
    metadata.pop("ownerReferences", None)
    metadata.pop("finalizers", None)
    return _fingerprint(stable)


def capture_cleanup_custody(document: dict[str, Any]) -> CleanupCustody:
    """Retain an already-verified object before orphaning its controller."""
    return CleanupCustody(
        uid=_identity(document)[2],
        snapshot_sha256=_cleanup_fingerprint(document),
        owner_references=document["metadata"].get("ownerReferences") or [],
    )


def verify_cleanup_custody(
    custody: CleanupCustody,
    document: dict[str, Any],
    *,
    observed_uid: str | None,
    namespace: str,
) -> None:
    uid = _identity(document)[2]
    owners = document["metadata"].get("ownerReferences") or []
    if (
        custody.uid != uid
        or observed_uid != uid
        or document["metadata"].get("namespace") != namespace
        or _cleanup_fingerprint(document) != custody.snapshot_sha256
        or not isinstance(owners, list)
        or owners not in ([], custody.owner_references)
    ):
        raise RegionalFixtureError("fixture cleanup ownership changed")


def _preserves(expected: Any, actual: Any, path: tuple[str, ...] = ()) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _preserves(value, actual[key], (*path, key))
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _preserves(left, right, (*path, str(index)))
                for index, (left, right) in enumerate(
                    zip(expected, actual, strict=True)
                )
            )
        )
    if (
        len(path) >= 3
        and path[-3] == "resources"
        and path[-2] in {"limits", "requests"}
    ):
        if type(expected) not in {str, int, float} or type(actual) not in {
            str,
            int,
            float,
        }:
            return False
        try:
            left: Decimal = parse_quantity(str(expected))
            right: Decimal = parse_quantity(str(actual))
            return left.is_finite() and right.is_finite() and left == right
        except (ValueError, TypeError, ArithmeticError):
            return False
    return type(expected) is type(actual) and expected == actual


class FixtureOwnership:
    def __init__(
        self,
        path: Path,
        binding: dict[str, Any],
        *,
        current_binding: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.path, self.binding = path, binding
        self.current_binding = current_binding
        with controller_ownership(path):
            self.resuming = path.exists() or path.is_symlink()
            if self.resuming:
                self.record = self._read()
            else:
                self.record = OwnershipRecord(
                    schema_version=1, binding=binding, owner=uuid4().hex
                )
                self._save()

    def _read(self) -> OwnershipRecord:
        try:
            record = OwnershipRecord.model_validate(read_private_document(self.path))
        except (OSError, ValueError, TypeError):
            raise RegionalFixtureError("fixture ownership journal is invalid") from None
        if record.binding != self.binding or any(
            key
            != str(value.expected.get("kind", "")).lower()
            + "/"
            + str((value.expected.get("metadata") or {}).get("name", ""))
            for key, value in record.creations.items()
        ):
            raise RegionalFixtureError("fixture ownership journal binding changed")
        return record

    def _save(self) -> None:
        data = self.record.model_dump(mode="json")
        OwnershipRecord.model_validate(data)
        if len(json.dumps(data).encode()) > 200000:
            raise RegionalFixtureError("fixture ownership journal is too large")
        write_json_atomic(self.path, data)

    @contextmanager
    def operation(self) -> Iterator[None]:
        with controller_ownership(self.path):
            if (
                self.current_binding is not None
                and self.current_binding() != self.binding
            ):
                raise RegionalFixtureError("fixture connection or source changed")
            self.record = self._read()
            yield

    def begin(self) -> None:
        with self.operation():
            if self.resuming or self.record.started or self.record.completed:
                raise RegionalFixtureError("an existing fixture is cleanup-only")
            self.record.started = True
            self._save()

    def intend(self, document: dict[str, Any]) -> None:
        with self.operation():
            key = (
                str(document["kind"]).lower() + "/" + str(document["metadata"]["name"])
            )
            if (
                self.resuming
                or not self.record.started
                or self.record.completed
                or key in self.record.creations
            ):
                raise RegionalFixtureError("fixture creation cannot be repeated")
            self.record.creations[key] = Creation(expected=document)
            self._save()

    def acknowledge(self, document: dict[str, Any]) -> str:
        with self.operation():
            kind, name, uid = _identity(document)
            creation = self.record.creations.get(kind + "/" + name)
            metadata = document["metadata"]
            if (
                self.resuming
                or not self.record.started
                or self.record.completed
                or creation is None
                or creation.uid is not None
                or not isinstance(creation.expected.get("apiVersion"), str)
                or not creation.expected["apiVersion"]
                or document.get("apiVersion") != creation.expected["apiVersion"]
                or document.get("kind") != creation.expected.get("kind")
                or metadata.get("namespace") != self.binding["namespace"]
                or not isinstance(metadata.get("resourceVersion"), str)
                or not metadata["resourceVersion"]
            ):
                raise RegionalFixtureError("fixture creation acknowledgement differs")
            creation.uid = uid
            creation.ack_sha256 = _fingerprint(document)
            self._save()
            if not _preserves(_stable(creation.expected, intent=True), document):
                raise RegionalFixtureError(
                    "fixture creation changed its declared intent"
                )
            creation.approved = True
            self._save()
            return uid

    def require_acknowledged(self) -> None:
        with self.operation():
            if any(item.uid is None for item in self.record.creations.values()):
                raise RegionalFixtureError(
                    "fixture creation outcome is unknown; adoption is forbidden"
                )

    def rejected_resources(
        self, read: Callable[[str, str], dict[str, Any] | None]
    ) -> list[dict[str, Any]]:
        with self.operation():
            self.require_acknowledged()
            result = []
            for key, creation in self.record.creations.items():
                if creation.approved:
                    continue
                kind, name = key.split("/", 1)
                current = read(kind, name)
                if current is None:
                    continue
                if (
                    _identity(current) != (kind, name, creation.uid)
                    or current["metadata"].get("namespace") != self.binding["namespace"]
                    or _fingerprint(current) != creation.ack_sha256
                ):
                    raise RegionalFixtureError(
                        "unapproved fixture changed after its ACK"
                    )
                result.append(current)
            return result

    def rejected(self, kind: str, name: str) -> bool:
        with self.operation():
            creation = self.record.creations.get(kind + "/" + name)
            return creation is not None and not creation.approved

    def deletion_only(self, document: dict[str, Any]) -> bool:
        with self.operation():
            self.require_acknowledged()
            kind, name, uid = _identity(document)
            key = kind + "/" + name
            custody = self.record.cleanup_custody.get(key)
            if custody is not None:
                verify_cleanup_custody(
                    custody,
                    document,
                    observed_uid=self.record.observed.get(key),
                    namespace=self.binding["namespace"],
                )
                return True
            creation = self.record.creations.get(key)
            if creation is None or creation.approved:
                return False
            if (
                creation.uid != uid
                or document["metadata"].get("namespace") != self.binding["namespace"]
                or _fingerprint(document) != creation.ack_sha256
            ):
                raise RegionalFixtureError("unapproved fixture changed after its ACK")
            return True

    def retain_for_cleanup(self, document: dict[str, Any]) -> None:
        """Retain observed custody before orphan deletion removes the owner chain."""
        with self.operation():
            self.require_acknowledged()
            kind, name, uid = _identity(document)
            key = kind + "/" + name
            if key in self.record.cleanup_custody:
                self.deletion_only(document)
                return
            if (
                self.record.completed
                or self.record.observed.get(key) != uid
                or document["metadata"].get("namespace") != self.binding["namespace"]
            ):
                raise RegionalFixtureError(
                    "fixture cleanup requires observed ownership"
                )
            self.record.cleanup_custody[key] = capture_cleanup_custody(document)
            self._save()

    def observe(
        self,
        document: dict[str, Any],
        read: Callable[[str, str], dict[str, Any] | None],
    ) -> None:
        with self.operation():
            self.require_acknowledged()
            kind, name, uid = _identity(document)
            key = kind + "/" + name
            if self.record.observed.get(key, uid) != uid:
                raise RegionalFixtureError("fixture observed UID changed")
            current = document
            seen: set[tuple[str, str]] = set()
            for _ in range(8):
                parent_kind, parent_name, parent_uid = _identity(current)
                parent_key = parent_kind + "/" + parent_name
                if (parent_kind, parent_name) in seen or current["metadata"].get(
                    "namespace"
                ) != self.binding["namespace"]:
                    raise RegionalFixtureError("fixture owner chain is invalid")
                seen.add((parent_kind, parent_name))
                creation = self.record.creations.get(parent_key)
                if creation is not None:
                    if (
                        creation.uid != parent_uid
                        or not creation.approved
                        or not _preserves(
                            _stable(creation.expected, intent=True), current
                        )
                    ):
                        raise RegionalFixtureError(
                            "fixture creation UID or declared intent changed"
                        )
                    self.record.observed[key] = uid
                    self._save()
                    return
                owners = current["metadata"].get("ownerReferences")
                if (
                    not isinstance(owners, list)
                    or len(owners) != 1
                    or not isinstance(owners[0], dict)
                    or owners[0].get("controller") is not True
                ):
                    raise RegionalFixtureError("fixture has no unique controller owner")
                owner = owners[0]
                owner_kind, owner_name, owner_uid = (
                    owner.get("kind"),
                    owner.get("name"),
                    owner.get("uid"),
                )
                if (
                    owner_kind not in {"Job", "PyTorchJob", "JobSet"}
                    or not isinstance(owner_name, str)
                    or not owner_name
                    or not isinstance(owner_uid, str)
                    or not owner_uid
                    or not isinstance(owner.get("apiVersion"), str)
                    or not owner["apiVersion"]
                ):
                    raise RegionalFixtureError(
                        "fixture controller owner is unsupported"
                    )
                parent = read(owner_kind.lower(), owner_name)
                if (
                    parent is None
                    or _identity(parent) != (owner_kind.lower(), owner_name, owner_uid)
                    or parent.get("apiVersion") != owner["apiVersion"]
                ):
                    raise RegionalFixtureError("fixture controller owner changed")
                current = parent
            raise RegionalFixtureError("fixture owner chain exceeds its bound")

    def complete(self) -> None:
        with self.operation():
            self.require_acknowledged()
            self.record.completed = True
            self._save()


def creation_document(raw: str) -> dict[str, Any]:
    def constant(value: str) -> Any:
        raise ValueError("nonstandard JSON constant")

    def unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate creation field")
            result[key] = value
        return result

    try:
        if len(raw.encode()) > 1048576:
            raise ValueError("creation acknowledgement size")
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=constant)
        if not isinstance(value, dict):
            raise ValueError("creation acknowledgement shape")
        return value
    except (ValueError, TypeError):
        raise RegionalFixtureError(
            "fixture creation acknowledgement is not a bounded object"
        ) from None
