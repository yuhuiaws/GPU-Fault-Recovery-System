from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.nvidia_logs import FabricManagerLogEvent, NvidiaLogNormalizer
from gpu_fault.regional import RemoteActionCommand, RemoteCommandStatus
from gpu_fault.store import NotFoundError
from gpu_fault.telemetry import EvidenceKind, RawEvidenceRecord
from scripts.e2e.regional.collector_acceptance_fixture import STORE_PROBE
from tests.execution.test_cluster_executor_lease_and_report import remote_command


class Record(SimpleNamespace):
    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        return vars(self)


class Store:
    def __init__(self) -> None:
        now = datetime.now(timezone.utc)
        source = FabricManagerLogEvent(
            cluster_id="cluster-a",
            node_id="node-a",
            record_id="fm-record",
            observed_at=now,
            source="/var/log/fabricmanager.log",
            product="H100",
            message="nvidia-nvswitch0: SXid (PCI:0000:ab:00.0): 11001, Fatal, Link 12 marker=case-a",
        )
        self.record = RawEvidenceRecord(
            record_id="fabric-manager/fm-record",
            cluster_id="cluster-a",
            node_id="node-a",
            kind=EvidenceKind.FABRIC_MANAGER_LOG,
            observed_at=now,
            ingested_at=now,
            expires_at=now + timedelta(hours=1),
            payload=source.model_dump(mode="json"),
        )
        self.event = (
            NvidiaLogNormalizer().normalize_fabric_manager(source).sxid_events[0]
        )
        self.incident = Record(
            incident_id="incident-a",
            event_id=self.event.event_id,
            cluster_id="cluster-a",
            node_ids=["node-a"],
        )
        self.decision = Record(
            event_id=self.event.event_id,
            incident_id="incident-a",
            workflow_request_id="workflow-a",
        )
        self.workflows = [
            Record(
                request_id="unrelated", incident_id="other-incident", created_at=now
            ),
            Record(
                request_id="workflow-a",
                incident_id="incident-a",
                created_at=now - timedelta(hours=1),
            ),
        ]
        self.missing_decision = False
        self.error: BaseException | None = None
        self.command_requests: list[list[str]] = []
        self.commands: list[RemoteActionCommand] = []
        self.xid_events: list[Record] = []

    def list_raw_evidence(
        self, cluster_id: str, **kwargs: Any
    ) -> list[RawEvidenceRecord]:
        assert cluster_id == "cluster-a" and kwargs["node_id"] == "node-a"
        return [self.record]

    def list_xid_events(self, cluster: str, node: str) -> list[Record]:
        assert (cluster, node) == ("cluster-a", "node-a")
        return self.xid_events

    def get_xid_policy_decision(self, event_id: str) -> Record | None:
        if self.error:
            raise self.error
        if self.missing_decision:
            return None
        if event_id != self.event.event_id:
            raise NotFoundError(event_id)
        return self.decision

    def get_workflow(self, request_id: str) -> Record:
        assert request_id == "workflow-a"
        return Record(request_id=request_id, incident_id="incident-a")

    def get_incident_by_event(self, event_id: str) -> Record | None:
        return self.incident if event_id == self.event.event_id else None

    def get_incident(self, incident_id: str) -> Record:
        assert incident_id == "incident-a"
        return self.incident

    def list_workflows(self, **kwargs: Any) -> list[Record]:
        return self.workflows

    def list_remote_commands(
        self, *, workflow_request_ids: list[str]
    ) -> list[RemoteActionCommand]:
        self.command_requests.append(workflow_request_ids)
        return self.commands


def run_probe(
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    marker: str = "case-a",
    scan: bool = True,
) -> dict[str, Any]:
    context = SimpleNamespace(store=store, nvidia_logs=NvidiaLogNormalizer())
    monkeypatch.setattr(ApplicationContext, "from_environment", lambda: context)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "cluster-a",
            "node-a",
            marker,
            datetime.now(timezone.utc).isoformat(),
            "1" if scan else "",
        ],
    )
    exec(compile(STORE_PROBE, "<collector-store-protocol>", "exec"), {})
    return dict(json.loads(capsys.readouterr().out))


def test_fabric_manager_join_uses_normalized_id_and_durable_links_not_node_time(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = Store()
    value = run_probe(store, monkeypatch, capsys)
    assert [row["request_id"] for row in value["workflows"]] == ["workflow-a"]
    assert value["fabric_events"][0]["event_id"] == store.event.event_id
    assert value["fabric_events_reconstructed"] is True
    assert store.command_requests == [["workflow-a"]]


def test_light_sxid_poll_still_joins_without_returning_raw_payload(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    value = run_probe(Store(), monkeypatch, capsys, scan=False)
    assert value["evidence"] == []
    assert value["evidence_scanned"] is False
    assert [row["request_id"] for row in value["workflows"]] == ["workflow-a"]


@pytest.mark.parametrize("scan", [False, True], ids=["light", "full"])
def test_store_probe_omits_live_lease_token_without_losing_command_evidence(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], scan: bool
) -> None:
    store = Store()
    command = remote_command("collector-command", node_ids=["node-a"])
    command.status = RemoteCommandStatus.LEASED
    command.lease_owner = "executor-a"
    command.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    command.lease_token = "collector-protocol-test-lease"
    store.commands = [command]
    before = command.model_dump(mode="json")

    value = run_probe(store, monkeypatch, capsys, scan=scan)

    assert store.command_requests == [["workflow-a"]]
    assert command.lease_token not in json.dumps(value)
    snapshot = value["commands"][0]
    assert "lease_token" not in snapshot
    assert snapshot["command_id"] == "collector-command"
    assert snapshot["cluster_id"] == "cluster-a"
    assert snapshot["workflow_request_id"] == "workflow-a"
    assert snapshot["incident_id"] == "incident-a"
    assert snapshot["status"] == "LEASED"
    assert snapshot["fencing_token"] == command.fencing_token
    assert value["commands"] == [
        {key: item for key, item in before.items() if key != "lease_token"}
    ]
    assert command.model_dump(mode="json") == before


def test_exact_decision_pointer_survives_the_global_workflow_scan_limit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = Store()
    store.workflows = []
    value = run_probe(store, monkeypatch, capsys)
    assert value["workflows"] == [
        {"request_id": "workflow-a", "incident_id": "incident-a"}
    ]


def test_event_link_survives_missing_decision_ack(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = Store()
    store.missing_decision = True
    value = run_probe(store, monkeypatch, capsys)
    assert value["decisions"] == []
    assert value["incidents"][0]["incident_id"] == "incident-a"
    assert value["workflows"][0]["request_id"] == "workflow-a"


def test_marker_prefix_and_same_node_activity_cannot_claim_another_case(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    value = run_probe(Store(), monkeypatch, capsys, marker="case")
    assert value["fabric_events"] == value["incidents"] == value["workflows"] == []


@pytest.mark.parametrize("cluster,node", [("other", "node-a"), ("cluster-a", "other")])
def test_durable_incident_link_must_still_match_cluster_and_node(
    cluster: str,
    node: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = Store()
    store.incident.cluster_id = cluster
    store.incident.node_ids = [node]
    with pytest.raises(RuntimeError, match="identity"):
        run_probe(store, monkeypatch, capsys)


def test_store_outage_is_not_misread_as_absent_policy_decision(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = Store()
    store.error = RuntimeError("store unavailable")
    with pytest.raises(RuntimeError, match="unavailable"):
        run_probe(store, monkeypatch, capsys)
