"""Ownership receipts for temporary, non-host HA probe Pods and ConfigMaps."""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.regional_commands import RegionalFixtureError

OWNER_ANNOTATION = "gpu-fault.io/ha-probe-owner"
RESOURCES = {"Pod": "pods", "ConfigMap": "configmaps", "Job": "jobs"}


class OwnedProbeResources:
    def __init__(
        self, path: Path, command: Callable[[list[str], str | None], str]
    ) -> None:
        if path.exists():
            raise RegionalFixtureError(
                "HA resource receipt already exists; use a new attempt"
            )
        self.path = path
        self.command = command
        self.owner = uuid4().hex
        self.records: dict[str, dict[str, Any]] = {}

    def persist(self) -> None:
        write_json_atomic(
            self.path,
            {"schema_version": 1, "owner": self.owner, "resources": self.records},
        )

    def read(self, kind: str, name: str) -> dict[str, Any] | None:
        output = self.command(
            ["get", kind, name, "--ignore-not-found", "-o", "json"], None
        )
        if not output.strip():
            return None
        value = json.loads(output)
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("metadata"), dict)
            or value["metadata"].get("name") != name
            or not value["metadata"].get("uid")
        ):
            raise RegionalFixtureError("HA resource identity read is incomplete")
        return value

    def owned(self, kind: str, name: str) -> dict[str, Any] | None:
        record = self.records.get(f"{kind}/{name}")
        if record is None:
            raise RegionalFixtureError("HA resource has no creation receipt")
        value = self.read(kind, name)
        if value is None:
            return None
        metadata = value["metadata"]
        if (
            metadata.get("annotations", {}).get(OWNER_ANNOTATION) != self.owner
            or metadata.get("namespace") != record["namespace"]
            or (record.get("uid") is not None and record["uid"] != metadata["uid"])
        ):
            raise RegionalFixtureError("HA resource ownership or UID changed")
        if record.get("uid") is None:
            record["uid"] = metadata["uid"]
            self.persist()
        return value

    def create(self, manifest: dict[str, Any]) -> None:
        value = copy.deepcopy(manifest)
        kind = value["kind"]
        name = value["metadata"]["name"]
        namespace = value["metadata"].get("namespace")
        if kind not in RESOURCES or not namespace or self.read(kind, name) is not None:
            raise RegionalFixtureError(
                "HA resource creation requires a confirmed unused name"
            )
        value["metadata"].setdefault("annotations", {})[OWNER_ANNOTATION] = self.owner
        self.records[f"{kind}/{name}"] = {"namespace": namespace, "uid": None}
        self.persist()
        self.command(["create", "-f", "-"], json.dumps(value))
        if self.owned(kind, name) is None:
            raise RegionalFixtureError("HA resource creation was not observed")

    def delete(self, kind: str, name: str, *, timeout_seconds: float = 60) -> None:
        if f"{kind}/{name}" not in self.records:
            return
        value = self.owned(kind, name)
        if value is None:
            return
        record = self.records[f"{kind}/{name}"]
        group = "/apis/batch/v1" if kind == "Job" else "/api/v1"
        path = (
            f"{group}/namespaces/{quote(record['namespace'], safe='')}"
            f"/{RESOURCES[kind]}/{quote(name, safe='')}"
        )
        self.command(
            ["delete", "--raw", path, "-f", "-"],
            json.dumps(
                {
                    "apiVersion": "v1",
                    "kind": "DeleteOptions",
                    "preconditions": {"uid": record["uid"]},
                    "propagationPolicy": "Foreground",
                }
            ),
        )
        deadline = time.monotonic() + timeout_seconds
        while self.owned(kind, name) is not None:
            if time.monotonic() >= deadline:
                raise RegionalFixtureError("HA resource deletion did not converge")
            time.sleep(1)
        record["deleted"] = True
        self.persist()
