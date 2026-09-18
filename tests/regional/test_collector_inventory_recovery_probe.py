"""Execute the actual CPU-side inventory evidence and recovery probe programs."""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.models import WorkflowOperation
from gpu_fault.store import NotFoundError
from gpu_fault.telemetry import EvidenceKind, RawEvidenceRecord
from scripts.e2e.regional.collector_inventory_reboot import (
    HOST_INVENTORY_EVIDENCE,
    INVENTORY_RECOVERY_STATE,
)
from tests.execution.test_cluster_executor_lease_and_report import remote_command
from tests.regional._collector_inventory_reboot_support import ORIGIN

EVENT = "host-node-a-batch-2-gpu_inventory_mismatch-node"


class Record(SimpleNamespace):
    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        return {
            key: value
            for key, value in vars(self).items()
            if key not in kwargs.get("exclude", set())
        }


class InventoryStore:
    def __init__(self) -> None:
        self.incident: Record | None = Record(
            incident_id="incident-a",
            event_id=EVENT,
            cluster_id="cluster-a",
            node_ids=["node-a"],
            created_at=ORIGIN + timedelta(seconds=30),
            workflow_request_id="workflow-a",
        )
        self.workflow = Record(
            request_id="workflow-a",
            incident_id="incident-a",
            created_at=ORIGIN + timedelta(seconds=30),
            updated_at=ORIGIN + timedelta(seconds=40),
        )
        self.workflows = [self.workflow]
        self.command = remote_command("command-a", node_ids=["node-a"])
        self.command.step.operation = WorkflowOperation.RESTART_NODE
        self.command.lease_token = "synthetic-inventory-lease-not-for-output"
        self.commands = [self.command]
        self.submission: Record | None = Record(
            cluster_name="cluster-a", state="SUBMITTED", action="REBOOT"
        )
        self.command_ids: list[str] = []
        self.submission_keys: list[tuple[str, str]] = []
        self.records = [
            RawEvidenceRecord(
                record_id="host-telemetry/host-node-a-batch-2",
                cluster_id="cluster-a",
                node_id="node-a",
                kind=EvidenceKind.HOST_TELEMETRY,
                observed_at=ORIGIN + timedelta(seconds=30),
                ingested_at=ORIGIN + timedelta(seconds=31),
                expires_at=ORIGIN + timedelta(hours=1),
                payload={
                    "cluster_id": "cluster-a",
                    "node_id": "node-a",
                    "batch_id": "host-node-a-batch-2",
                    "producer": "node",
                    "collection_errors": [],
                    "samples": [{"name": "gpu_inventory_mismatch", "value": 1}],
                },
            )
        ]

    def get_incident_by_event(self, event_id: str) -> Record | None:
        assert event_id == EVENT
        return self.incident

    def get_incident(self, incident_id: str) -> Record | None:
        assert incident_id == "incident-a"
        return self.incident

    def list_workflows(self, **kwargs: Any) -> list[Record]:
        assert kwargs == {"limit": 500, "newest_first": True}
        return self.workflows

    def list_remote_commands(self, *, workflow_request_ids: list[str]) -> list[Any]:
        self.command_ids = workflow_request_ids
        return self.commands

    def get_hyperpod_submission(self, cluster: str, key: str) -> Record:
        self.submission_keys.append((cluster, key))
        if self.submission is None:
            raise NotFoundError("submission")
        return self.submission

    def list_raw_evidence(self, cluster: str, **kwargs: Any) -> list[RawEvidenceRecord]:
        assert cluster == "cluster-a"
        assert kwargs == {
            "node_id": "node-a",
            "kind": EvidenceKind.HOST_TELEMETRY,
            "limit": 500,
        }
        return self.records


def execute(
    store: InventoryStore,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    recovery: bool = True,
) -> dict[str, Any]:
    monkeypatch.setattr(
        ApplicationContext, "from_environment", lambda: SimpleNamespace(store=store)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inventory-probe",
            "cluster-a",
            "node-a",
            *(
                [EVENT, ORIGIN.isoformat(), "cluster-a"]
                if recovery
                else [ORIGIN.isoformat()]
            ),
        ],
    )
    source = INVENTORY_RECOVERY_STATE if recovery else HOST_INVENTORY_EVIDENCE
    exec(compile(source, "<inventory-recovery-probe>", "exec"), {})
    return dict(json.loads(capsys.readouterr().out))


def test_exact_recovery_lookup_preserves_command_scope_without_lease_credentials(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = InventoryStore()
    value = execute(store, monkeypatch, capsys)
    assert value["event_id"] == EVENT
    assert value["node_workflow_ids"] == ["workflow-a"]
    assert store.command_ids == ["workflow-a"]
    assert isinstance(store.command.lease_token, str), (
        "test_exact_recovery_lookup_preserves_command_scope_without_lease_credentials: expected isinstance(store.command.lease_token, str)"
    )
    assert store.command.lease_token not in json.dumps(value)
    assert "lease_token" not in value["commands"][0]
    assert store.submission_keys == [
        ("cluster-a", f"workflow-a/RESTART_NODE/{store.command.step_index}")
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("cluster_id", "other"),
        ("node_ids", ["node-a", "node-b"]),
        ("event_id", "other-event"),
        ("created_at", ORIGIN - timedelta(seconds=1)),
        ("workflow_request_id", None),
        ("workflow_request_id", "absent-workflow"),
    ],
)
def test_foreign_or_old_incident_does_not_authorize_recovery(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    field: str,
    value: Any,
) -> None:
    store = InventoryStore()
    assert store.incident is not None
    setattr(store.incident, field, value)
    with pytest.raises(ValueError, match="incident|workflow"):
        execute(store, monkeypatch, capsys)
    assert store.command_ids == []


def test_missing_incident_remains_unresolved(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = InventoryStore()
    store.incident = None
    with pytest.raises(ValueError, match="persisted incident"):
        execute(store, monkeypatch, capsys)


@pytest.mark.parametrize("field", ["cluster_id", "incident_id", "workflow_request_id"])
def test_command_from_another_scope_is_not_recovery_evidence(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], field: str
) -> None:
    store = InventoryStore()
    setattr(store.command, field, "other")
    with pytest.raises(ValueError, match="command identity"):
        execute(store, monkeypatch, capsys)


def test_unconfirmed_submission_is_not_fabricated(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = InventoryStore()
    store.submission = None
    assert execute(store, monkeypatch, capsys)["submissions"] == []


@pytest.mark.parametrize("covers_boundary", [False, True])
def test_full_workflow_page_requires_a_complete_time_boundary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    covers_boundary: bool,
) -> None:
    store = InventoryStore()
    old = ORIGIN - timedelta(days=1)
    store.workflows.extend(
        Record(
            request_id=f"historical-{index}",
            incident_id="historical",
            created_at=old,
            updated_at=old if covers_boundary else ORIGIN,
        )
        for index in range(499)
    )
    if covers_boundary:
        assert len(execute(store, monkeypatch, capsys)["workflows"]) == 1
    else:
        with pytest.raises(ValueError, match="scan is incomplete"):
            execute(store, monkeypatch, capsys)


def test_inventory_evidence_is_bound_to_real_producer_record(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    value = execute(InventoryStore(), monkeypatch, capsys, recovery=False)
    assert value["records"][0]["batch_id"] == "host-node-a-batch-2"
    assert value["records"][0]["record_id"] == "host-telemetry/host-node-a-batch-2"


@pytest.mark.parametrize(
    "field,value",
    [
        ("cluster_id", "other"),
        ("node_id", "other"),
        ("batch_id", "host-other-batch-2"),
        ("producer", "control-plane"),
        ("collection_errors", ["failed"]),
    ],
)
def test_foreign_or_failed_raw_inventory_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    field: str,
    value: Any,
) -> None:
    store = InventoryStore()
    store.records[0].payload[field] = value
    with pytest.raises(ValueError, match="identity or collection"):
        execute(store, monkeypatch, capsys, recovery=False)


@pytest.mark.parametrize("covers_boundary", [False, True])
def test_full_raw_evidence_page_requires_a_complete_time_boundary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    covers_boundary: bool,
) -> None:
    store = InventoryStore()
    original = store.records[0]
    store.records.extend(
        original.model_copy(
            update={
                "observed_at": ORIGIN - timedelta(days=1) if covers_boundary else ORIGIN
            }
        )
        for _ in range(499)
    )
    if covers_boundary:
        assert len(execute(store, monkeypatch, capsys, recovery=False)["records"]) == 1
    else:
        with pytest.raises(ValueError, match="scan is incomplete"):
            execute(store, monkeypatch, capsys, recovery=False)
