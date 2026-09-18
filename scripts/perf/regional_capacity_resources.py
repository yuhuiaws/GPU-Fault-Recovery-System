"""Creation receipts and UID-bound teardown for one capacity run."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.acceptance_runner_common import write_json_atomic  # noqa: E402
from scripts.perf.regional_capacity_registry import RUN_LABEL  # noqa: E402


def run_manifest(manifest: dict[str, Any], run_id: str) -> dict[str, Any]:
    value = copy.deepcopy(manifest)
    value["metadata"].setdefault("labels", {})[RUN_LABEL] = run_id
    if value["kind"] == "Job":
        value["spec"]["template"].setdefault("metadata", {}).setdefault("labels", {})[
            RUN_LABEL
        ] = run_id
    return value


def content_digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class RunResources:
    def __init__(
        self,
        artifacts: Path,
        run_id: str,
        namespace: str,
        client: Callable[..., str],
    ) -> None:
        self.path = artifacts / "capacity-resources.json"
        self.run_id, self.namespace, self.client = run_id, namespace, client
        self.document = (
            json.loads(self.path.read_text())
            if self.path.exists()
            else {"run_id": run_id, "namespace": namespace, "resources": {}}
        )
        if (
            self.document.get("run_id") != run_id
            or self.document.get("namespace") != namespace
        ):
            raise RuntimeError(
                "capacity resource receipt belongs to another run or namespace"
            )
        self.records: dict[str, Any] = self.document["resources"]

    def persist(self) -> None:
        write_json_atomic(self.path, self.document)

    def read(self, kind: str, name: str) -> dict[str, Any] | None:
        raw = self.client(
            "get", kind, name, "--ignore-not-found", "-o", "json", timeout=60
        )
        value = json.loads(raw) if raw.strip() else None
        if value is not None and not isinstance(value, dict):
            raise RuntimeError("capacity resource read is not an object")
        return value

    def owned(self, kind: str, name: str) -> dict[str, Any] | None:
        value = self.read(kind, name)
        if value is None:
            return None
        record = self.records.get(f"{kind}/{name}")
        metadata = value.get("metadata") or {}
        if (
            record is None
            or metadata.get("namespace") != self.namespace
            or metadata.get("labels", {}).get(RUN_LABEL) != self.run_id
            or not metadata.get("uid")
            or not metadata.get("resourceVersion")
            or (record.get("uid") is not None and record["uid"] != metadata["uid"])
        ):
            raise RuntimeError("capacity resource ownership or UID changed")
        record["uid"] = metadata["uid"]
        self.persist()
        return value

    def create(self, manifest: dict[str, Any]) -> dict[str, Any]:
        value = run_manifest(manifest, self.run_id)
        kind, name = value["kind"].lower(), value["metadata"]["name"]
        if (
            value["metadata"].get("namespace") != self.namespace
            or self.read(kind, name) is not None
        ):
            raise RuntimeError("capacity creation requires a confirmed unused name")
        self.records[f"{kind}/{name}"] = {
            "uid": None,
            "data_sha256": content_digest(value.get("data", {})),
        }
        self.persist()
        try:
            self.client(
                "create", "-f", "-", stdin=json.dumps(value).encode(), timeout=120
            )
        finally:
            self.owned(kind, name)
        if self.owned(kind, name) is None:
            raise RuntimeError("capacity resource creation was not observed")
        return value

    def update_data(self, name: str, data: dict[str, str]) -> None:
        value = self.owned("configmap", name)
        if value is None:
            raise RuntimeError("capacity start gate disappeared")
        record = self.records[f"configmap/{name}"]
        if content_digest(value.get("data", {})) != record["data_sha256"]:
            raise RuntimeError("capacity start gate changed outside the run")
        metadata = value["metadata"]
        patch = [
            {"op": "test", "path": "/metadata/uid", "value": metadata["uid"]},
            {
                "op": "test",
                "path": "/metadata/resourceVersion",
                "value": metadata["resourceVersion"],
            },
            {"op": "replace", "path": "/data", "value": data},
        ]
        try:
            self.client(
                "patch",
                "configmap",
                name,
                "--type=json",
                "--patch-file=/dev/stdin",
                stdin=json.dumps(patch).encode(),
                timeout=60,
            )
        except Exception:
            after = self.owned("configmap", name)
            if after is None or after.get("data") != data:
                raise
        after = self.owned("configmap", name)
        if after is None or after.get("data") != data:
            raise RuntimeError("capacity start gate update was not confirmed")
        record["data_sha256"] = content_digest(data)
        self.persist()

    def delete_all(self) -> None:
        from scripts.e2e.regional.seeded_command_fixture import delete_owned_resource

        ordered = sorted(
            self.records,
            key=lambda key: (key.split("/", 1)[0] not in {"pod", "job"}, key),
        )
        for key in ordered:
            kind, name = key.split("/", 1)
            value = self.owned(kind, name)
            if value is None:
                if self.records[key].get("uid") is None:
                    raise RuntimeError(
                        "capacity create acknowledgement is unresolved; absence is not cleanup"
                    )
                continue
            delete_owned_resource(
                kind,
                name,
                self.run_id,
                client=self.client,
                namespace=self.namespace,
                expected_uid=self.records[key]["uid"],
                require_uid=True,
            )
