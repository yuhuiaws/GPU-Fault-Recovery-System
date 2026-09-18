"""Portable, strict evidence contract for the isolated NOTIFY008 probe."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from itertools import product
from typing import Any

CASE_ID = "GF-REGIONAL-NOTIFY-008"
CONFIRMATION = "NOTIFY008_EXECUTE"
SCOPE = "regional-isolated-postgresql-simulated-provider"
CRASH_POINTS = ("before-provider", "accepted-before-commit", "committed-before-ack")
MODES = ("inline", "outbox")
KINDS = ("gpu-reset", "workload-restart")
LEASE_SECONDS = 30
CRASH_EXIT = -9
MAX_RECEIPTS = 16


class ProbeError(RuntimeError):
    """A missing or contradictory observation prevents acceptance."""


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def run_identity(value: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"notify008-[0-9a-f]{16}", value) is None
    ):
        raise ProbeError("invalid NOTIFY008 run identity")
    return value


@dataclass(frozen=True)
class Target:
    run_id: str
    cluster_id: str
    region: str
    release_id: str
    node: str
    node_uid: str
    source_pod: str
    source_pod_uid: str
    deployment_uid: str
    deployment_generation: int
    runtime_image: str
    postgres_image: str
    runtime_version: str
    runtime_module_digest: str
    seconds: int = 540

    def __post_init__(self) -> None:
        run_identity(self.run_id)
        for name in (
            "cluster_id",
            "region",
            "release_id",
            "node",
            "node_uid",
            "source_pod",
            "source_pod_uid",
            "deployment_uid",
            "runtime_version",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 256
                or any(character.isspace() for character in value)
            ):
                raise ProbeError(f"invalid target {name}")
        for image in (self.runtime_image, self.postgres_image):
            if (
                not isinstance(image, str)
                or "://" in image
                or re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}", image
                )
                is None
            ):
                raise ProbeError(
                    "both runtime and PostgreSQL images require explicit digests"
                )
        if re.fullmatch(r"[0-9a-f]{64}", self.runtime_module_digest) is None:
            raise ProbeError("runtime module digest is missing")
        if (
            type(self.deployment_generation) is not int
            or self.deployment_generation < 1
        ):
            raise ProbeError("invalid deployment generation")
        if type(self.seconds) is not int or not 300 <= self.seconds <= 600:
            raise ProbeError(
                "isolated job lifetime must be between 300 and 600 seconds"
            )


@dataclass(frozen=True)
class Variant:
    mode: str
    crash: str
    kind: str

    def __post_init__(self) -> None:
        if (
            self.mode not in MODES
            or self.crash not in CRASH_POINTS
            or self.kind not in KINDS
        ):
            raise ProbeError("invalid NOTIFY008 variant")

    @property
    def key(self) -> str:
        return f"{self.mode}/{self.crash}/{self.kind}"

    def notification_id(self, run_id: str) -> str:
        return f"{run_identity(run_id)}/{self.key}"

    @property
    def first_acceptances(self) -> int:
        return 0 if self.crash == "before-provider" else 1

    @property
    def total_acceptances(self) -> int:
        return 2 if self.crash == "accepted-before-commit" else 1


def variants() -> tuple[Variant, ...]:
    return tuple(
        Variant(mode, crash, kind)
        for mode, crash, kind in product(MODES, CRASH_POINTS, KINDS)
    )


def receipt_errors(
    value: Any, *, run_id: str, notification_id: str, count: int
) -> list[str]:
    if not isinstance(value, list) or len(value) != count:
        return ["independent provider receipt count differs"]
    ids = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "run_id",
            "notification_id",
            "message_id",
            "provider_pid",
            "sequence",
        }:
            return ["provider receipt shape differs"]
        message_id, provider_pid, sequence = (
            item["message_id"],
            item["provider_pid"],
            item["sequence"],
        )
        if (
            item["run_id"] != run_id
            or item["notification_id"] != notification_id
            or not isinstance(message_id, str)
            or re.fullmatch(r"simulated-[0-9a-f]{32}", message_id) is None
            or type(provider_pid) is not int
            or provider_pid <= 0
            or type(sequence) is not int
            or not 1 <= sequence <= MAX_RECEIPTS
        ):
            return ["provider receipt identity differs"]
        ids.append(message_id)
    return (
        ["provider acceptance IDs are not distinct"] if len(set(ids)) != count else []
    )


def variant_errors(value: Any, *, run_id: str, variant: Variant) -> list[str]:
    if not isinstance(value, dict):
        return ["variant evidence is not an object"]
    errors: list[str] = []
    expected = {
        "variant": variant.key,
        "notification_id": variant.notification_id(run_id),
        "crash_exitcode": CRASH_EXIT,
        "before_result": "SENT" if variant.crash == "committed-before-ack" else None,
        "final_result": "SENT",
        "stored_notifications": 1,
        "replay_added_acceptances": 0,
        "early_retry_added_acceptances": 0,
        "exactly_once_proven": False,
        "duplicate_observed": variant.crash == "accepted-before-commit",
    }
    for name, wanted in expected.items():
        observed = value.get(name)
        if (
            name not in value
            or type(observed) is not type(wanted)
            or observed != wanted
        ):
            errors.append(f"variant differs at {name}")
    pids = [
        value.get(name)
        for name in ("supervisor_pid", "provider_pid", "first_pid", "replacement_pid")
    ]
    if any(type(pid) is not int or pid <= 0 for pid in pids) or len(set(pids)) != 4:
        errors.append("runtime/provider process identities are not independent")
    for key, count in (
        ("before_receipts", variant.first_acceptances),
        ("receipts", variant.total_acceptances),
    ):
        errors.extend(
            receipt_errors(
                value.get(key),
                run_id=run_id,
                notification_id=variant.notification_id(run_id),
                count=count,
            )
        )
    if not errors:
        receipts = value["receipts"]
        if any(item["provider_pid"] != value["provider_pid"] for item in receipts):
            errors.append("receipts do not belong to the observed provider process")
        if value["before_receipts"] != receipts[: variant.first_acceptances]:
            errors.append("provider history changed across the worker crash")
        if value.get("provider_message_id") != receipts[-1]["message_id"]:
            errors.append(
                "durable result does not name the last accepted provider receipt"
            )
    return errors


def report_errors(value: Any, *, run_id: str) -> list[str]:
    run_identity(run_id)
    if not isinstance(value, dict):
        return ["probe report is not an object"]
    errors = []
    for name, expected in {
        "case_id": CASE_ID,
        "run_id": run_id,
        "validation_scope": SCOPE,
        "provider": "SIMULATED",
        "postgres_major": 16,
        "database_is_unix_socket": True,
        "production_credentials_loaded": False,
        "exactly_once_proven": False,
        "children_reaped": True,
    }.items():
        if type(value.get(name)) is not type(expected) or value.get(name) != expected:
            errors.append(f"probe report differs at {name}")
    rows = value.get("variants")
    if not isinstance(rows, list) or len(rows) != len(variants()):
        return [*errors, "complete twelve-variant process evidence is missing"]
    for variant, row in zip(variants(), rows, strict=True):
        errors.extend(
            f"{variant.key}: {error}"
            for error in variant_errors(row, run_id=run_id, variant=variant)
        )
    return errors
