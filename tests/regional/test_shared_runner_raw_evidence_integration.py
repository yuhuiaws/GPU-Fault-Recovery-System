"""Retained evidence cannot invent a marker event or bind a foreign workflow."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.models import (
    FaultIncident,
    MarkerScope,
    NodeMarker,
    Severity,
    WorkflowRequest,
    WorkflowStatus,
)
from gpu_fault.nvidia_logs import NvidiaKernelLogEvent, NvidiaLogNormalizer
from gpu_fault.policy import (
    ActionDisposition,
    ActionSource,
    Containment,
    FaultPolicyDecision,
    XidEvent,
)
from gpu_fault.policy.models import FaultEventType
from gpu_fault.store import NotFoundError
from gpu_fault.telemetry import EvidenceKind, RawEvidenceRecord
from scripts.e2e.regional.regional_live_fixture import STORE_PROBE

AT = datetime(2026, 9, 14, 13, 43, 43, tzinfo=timezone.utc)
MARKER = "marker-shared-0915"


class ProbeStore:
    def __init__(
        self,
        *,
        message: str | None = None,
        record_id: str = "kmsg-boot-a-7",
        evidence_ref: str = "kmsg://node-a/boot-a/7",
    ) -> None:
        source = NvidiaKernelLogEvent(
            cluster_id="cluster-a",
            node_id="node-a",
            record_id=record_id,
            observed_at=AT,
            source_boot_id="boot-a",
            collected_at=AT + timedelta(seconds=1),
            message=message or f"NVRM: Xid (PCI:0000:59:00): 46, marker={MARKER}",
            evidence_ref=evidence_ref,
        )
        self.event = NvidiaLogNormalizer().normalize_kernel(source).xid_events[0]
        self.records = [
            RawEvidenceRecord(
                record_id=f"nvidia-kernel/{source.record_id}",
                cluster_id=source.cluster_id,
                node_id=source.node_id,
                kind=EvidenceKind.NVIDIA_KERNEL,
                observed_at=AT,
                ingested_at=AT + timedelta(seconds=1),
                expires_at=AT + timedelta(days=1),
                payload=source.model_dump(mode="json"),
            )
        ]
        marker = NodeMarker(
            marker_id=f"marker-{self.event.event_id}",
            source="gpu-fault-policy/nvidia_catalog",
            cluster_id=source.cluster_id,
            trusted=True,
            incident_id="incident-a",
            observed_at=AT,
            expires_at=AT + timedelta(hours=1),
            source_boot_id=source.source_boot_id,
            event_source="KERNEL_LOG",
            scope=MarkerScope(
                node_ids=[source.node_id],
                pci_bdfs=[self.event.pci_bdf] if self.event.pci_bdf else [],
            ),
            severity=Severity.CRITICAL,
            mapping_version="test-policy",
            raw_evidence_ref=source.evidence_ref,
        )
        self.markers = [marker]
        self.decision: FaultPolicyDecision | None = FaultPolicyDecision(
            event_id=self.event.event_id,
            event_type=FaultEventType.XID,
            policy_version="test-policy",
            source=ActionSource.NVIDIA_CATALOG,
            disposition=ActionDisposition.EXECUTABLE,
            severity=Severity.CRITICAL,
            containment=Containment.GPU,
            reasons=["isolated test"],
            marker=marker,
            incident_id="incident-a",
        )
        self.events: list[XidEvent] = []
        self.incident = FaultIncident(
            incident_id="incident-a",
            event_id=self.event.event_id,
            event_type="XID",
            cluster_id=source.cluster_id,
            node_ids=[source.node_id],
            policy_version="test-policy",
            policy_source="NVIDIA_CATALOG",
            workflow_request_id="workflow-a",
        )
        self.workflow = WorkflowRequest(
            request_id="workflow-a",
            incident_id=self.incident.incident_id,
            status=WorkflowStatus.SUPERSEDED,
            fencing_token=1,
        )
        self.incident_lookups: list[str] = []
        self.decision_lookups: list[str] = []
        self.marker_reads = 0

    def list_xid_events(self, *_args: Any, **_kwargs: Any) -> list[XidEvent]:
        return self.events

    def list_raw_evidence(self, cluster: str, **kwargs: Any) -> list[RawEvidenceRecord]:
        assert cluster == "cluster-a" and kwargs["node_id"] == "node-a", (
            "the raw scan must remain bound to the requested cluster and node"
        )
        assert kwargs["limit"] == 500, "the retained-evidence scan must stay bounded"
        return self.records

    def list_recent_markers_for_nodes(self, *_args: Any) -> list[NodeMarker]:
        self.marker_reads += 1
        return self.markers

    def get_xid_policy_decision(self, event_id: str) -> FaultPolicyDecision:
        self.decision_lookups.append(event_id)
        if self.decision is None:
            raise NotFoundError("decision is absent")
        return self.decision

    def get_incident_by_event(self, event_id: str) -> FaultIncident:
        self.incident_lookups.append(event_id)
        return self.incident

    def get_workflow(self, _request_id: str) -> WorkflowRequest:
        return self.workflow

    def list_remote_commands(self, **_kwargs: Any) -> list[Any]:
        return []

    def list_notifications(self) -> list[Any]:
        return []

    def list_attempt_observations(self, _cluster: str) -> list[Any]:
        return []

    def get_agent(self, *_args: Any) -> Any:
        raise NotFoundError("agent is absent")

    def processor_queue_stats(self) -> dict[str, int]:
        return {"depth": 0}

    def processor_fault_backlog_depth(self) -> int:
        return 0

    def remote_command_stats(self) -> dict[str, int]:
        return {"pending": 0}


def run_probe(
    store: ProbeStore,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    observed_after: datetime | None = None,
) -> dict[str, Any]:
    monkeypatch.setattr(
        ApplicationContext,
        "from_environment",
        classmethod(lambda _cls: SimpleNamespace(store=store)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "store-probe",
            "cluster-a",
            "node-a",
            MARKER,
            observed_after.isoformat() if observed_after else "",
            "",
            "",
            "",
            "1",
        ],
    )
    exec(
        compile(STORE_PROBE, "<shared-store-probe>", "exec"), {"__name__": "__probe__"}
    )
    return json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("decision_retained", [True, False])
def test_store_probe_recovers_the_event_when_the_correlation_row_is_gone(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    decision_retained: bool,
) -> None:
    store = ProbeStore()
    if decision_retained:
        assert store.decision is not None, "this branch tests a retained decision"
        store.decision = store.decision.model_copy(
            update={"marker": store.markers[0].model_copy(update={"active": False})}
        )
        store.markers = []
    else:
        store.decision = None
    payload = run_probe(store, monkeypatch, capsys)
    assert payload["event"]["event_id"] == store.event.event_id, (
        "the shipped normalizer determines the event ID, not an arbitrary marker"
    )
    assert payload["event"]["xid"] == 46 and payload["event"]["pci_bdf"] == "0000:59:00"
    assert payload["event"]["recovered_from"] == "raw_evidence", (
        "reconstruction must remain explicit in the evidence"
    )
    assert store.incident_lookups == [store.event.event_id], (
        "the incident lookup must use exactly the reconstructed event identity"
    )
    assert store.decision_lookups == [store.event.event_id], (
        "normalization and evidence resolution must not reread the decision"
    )
    assert payload["workflow"]["status"] == "SUPERSEDED", (
        "a recovered terminal workflow is not relabeled as successful"
    )
    assert store.marker_reads == (0 if decision_retained else 1), (
        "a retained decision marker must not depend on the active marker working set"
    )


@pytest.mark.parametrize(
    "message",
    [
        "NVRM: Xid (PCI:0000:59:00): 46, another drill",
        f"NVRM: Xid (PCI:0000:59:00): 46, marker={MARKER}-retry",
        f"NVRM: Xid (PCI:0000:59:00): 46, marker=prefix-{MARKER}",
    ],
    ids=["foreign-marker", "marker-prefix-collision", "marker-suffix-collision"],
)
def test_foreign_marker_text_cannot_reconstruct_a_workflow(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], message: str
) -> None:
    store = ProbeStore(message=message)
    payload = run_probe(store, monkeypatch, capsys)
    assert payload["event"] is None and payload["incident"] is None, (
        "unrelated text cannot identify this injection"
    )
    assert store.incident_lookups == [] and store.decision_lookups == [], (
        "no event identity may be invented for a foreign marker"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("cluster_id", "foreign-cluster"),
        ("node_id", "foreign-node"),
        ("record_id", "foreign-record"),
        ("kind", "UNKNOWN"),
        ("observed_at", AT - timedelta(seconds=1)),
    ],
    ids=["cluster", "node", "record-id", "kind", "timestamp"],
)
def test_foreign_raw_envelope_cannot_authorize_an_event(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    field: str,
    value: Any,
) -> None:
    store = ProbeStore()
    store.records[0] = store.records[0].model_copy(update={field: value})
    with pytest.raises(ValueError, match="identity"):
        run_probe(store, monkeypatch, capsys)
    assert store.incident_lookups == [] and store.decision_lookups == [], (
        "a foreign envelope must fail before event or workflow resolution"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"cluster_id": "foreign"},
        {"node_id": "foreign"},
        {"record_id": "other"},
        {"source_boot_id": ""},
        {"source_boot_id": "boot-foreign"},
        {"evidence_ref": ""},
        {"evidence_ref": "kmsg://foreign/boot-a/7"},
        {"evidence_ref": "kmsg://node-a/boot-a/8"},
        {"evidence_ref": "kmsg://node-a/boot-a/7?extra=1"},
        {"evidence_ref": "kmsg://node-a/boot-a/7#other"},
        {"evidence_ref": "unknown://node-a/boot-a/7"},
        {"observed_at": "2026-09-14T13:43:43"},
        {"message": f"NVRM: Xid malformed marker={MARKER}"},
        {"message": f"NVRM: Xid 46 Xid 79 marker={MARKER}"},
        {"message": f"NVRM: Xid 46 SXid malformed marker={MARKER}"},
    ],
    ids=[
        "cluster",
        "node",
        "record",
        "empty-boot",
        "wrong-boot",
        "empty-ref",
        "ref-node",
        "ref-sequence",
        "ref-query",
        "ref-fragment",
        "unknown-source",
        "naive-time",
        "unparsed-xid",
        "multiple-xids",
        "mixed-unparsed-sxid",
    ],
)
def test_malformed_or_foreign_raw_payload_never_supplies_a_workflow(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    changes: dict[str, Any],
) -> None:
    store = ProbeStore()
    store.records[0] = store.records[0].model_copy(
        update={"payload": {**store.records[0].payload, **changes}}
    )
    with pytest.raises(ValueError, match="raw"):
        run_probe(store, monkeypatch, capsys)
    assert store.incident_lookups == [] and store.decision_lookups == [], (
        "unknown raw data must fail before any event linkage is trusted"
    )
    assert capsys.readouterr().out == "", (
        "a failed read must not emit a usable snapshot"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"marker_id": "marker-unrelated-event"},
        {"marker_id": "unprefixed"},
        {"cluster_id": "foreign"},
        {"cluster_id": None},
        {"trusted": False},
        {"event_source": None},
        {"event_source": "UNKNOWN"},
        {"source_boot_id": "boot-foreign"},
        {"raw_evidence_ref": "kmsg://other/boot-a/7"},
        {"observed_at": AT - timedelta(seconds=1)},
        {"scope": MarkerScope(node_ids=["other"], pci_bdfs=["0000:59:00"])},
        {"scope": MarkerScope(node_ids=["node-a", "other"], pci_bdfs=["0000:59:00"])},
        {"scope": MarkerScope(node_ids=["node-a"], pci_bdfs=["0000:60:00"])},
    ],
    ids=[
        "event-id",
        "unprefixed-id",
        "cluster",
        "missing-cluster",
        "untrusted",
        "missing-source",
        "unknown-source",
        "boot",
        "reference",
        "timestamp",
        "node",
        "multiple-nodes",
        "pci",
    ],
)
@pytest.mark.parametrize("decision_retained", [True, False])
def test_marker_identity_is_checked_even_when_its_reference_matches(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    changes: dict[str, Any],
    decision_retained: bool,
) -> None:
    store = ProbeStore()
    changed = store.markers[0].model_copy(update=changes)
    if decision_retained:
        assert store.decision is not None, "a decision is needed for this proof branch"
        store.decision = store.decision.model_copy(update={"marker": changed})
    else:
        store.decision = None
        store.markers = [changed]
    if not decision_retained and "raw_evidence_ref" in changes:
        result = run_probe(store, monkeypatch, capsys)
        assert result["event"] is None, "an unmatched marker is not event proof"
    else:
        with pytest.raises(ValueError, match="marker identity"):
            run_probe(store, monkeypatch, capsys)
    assert store.incident_lookups == [], (
        "foreign marker proof cannot resolve a workflow"
    )


@pytest.mark.parametrize(
    "field,value", [("event_id", "foreign"), ("event_type", "SXID")]
)
def test_foreign_decision_cannot_supply_the_reconstructed_event_identity(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    field: str,
    value: str,
) -> None:
    store = ProbeStore()
    assert store.decision is not None, "the test replaces an existing decision identity"
    store.decision = store.decision.model_copy(update={field: value})
    with pytest.raises(ValueError, match="decision identity"):
        run_probe(store, monkeypatch, capsys)
    assert store.incident_lookups == [], "foreign decisions cannot identify a workflow"


@pytest.mark.parametrize("proof", ["absent", "duplicate-marker", "duplicate-record"])
def test_absent_or_ambiguous_proof_never_selects_the_first_candidate(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], proof: str
) -> None:
    store = ProbeStore()
    store.decision = None
    if proof == "absent":
        store.markers = []
        result = run_probe(store, monkeypatch, capsys)
        assert result["event"] is None, "raw receipt alone is not ingestion proof"
    else:
        if proof == "duplicate-marker":
            store.markers *= 2
        else:
            store.records *= 2
        with pytest.raises(ValueError, match="ambiguous"):
            run_probe(store, monkeypatch, capsys)
    assert store.incident_lookups == [], (
        "ambiguous evidence cannot identify an incident"
    )


def test_stale_raw_evidence_is_not_recovered(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = ProbeStore()
    result = run_probe(
        store, monkeypatch, capsys, observed_after=AT + timedelta(seconds=1)
    )
    assert result["event"] is None and store.incident_lookups == [], (
        "even a matching marker must respect the widened lower bound"
    )


@pytest.mark.parametrize(
    "message",
    [
        f"NVRM: XID (PCI:0000:59:00): 46, marker={MARKER}",
        f"NVRM: Xid (0000:59:00): 46, marker={MARKER}",
        f"NVRM: Xid: 46, marker={MARKER}",
    ],
    ids=["uppercase", "bare-pci", "no-pci"],
)
def test_reconstruction_uses_the_runtime_parser_not_a_narrow_runner_regex(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], message: str
) -> None:
    store = ProbeStore(message=message)
    result = run_probe(store, monkeypatch, capsys)
    assert result["event"]["xid"] == 46, "valid runtime parser forms must remain usable"


def test_api_replay_is_reconstructed_without_claiming_a_kmsg_reference(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    reference = f"api-replay://GF-REGIONAL-DESTR-009/{MARKER}"
    store = ProbeStore(record_id=MARKER, evidence_ref=reference)
    result = run_probe(store, monkeypatch, capsys)
    assert result["event"]["evidence_ref"] == reference, (
        "the synthetic source reference must be retained without relabeling"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"cluster_id": "foreign"},
        {"node_ids": ["foreign"]},
        {"node_ids": "node-a"},
        {"node_ids": ["node-a", None]},
        {"incident_id": ""},
        {"incident_id": "contradictory"},
        {"event_id": "foreign", "incident_id": "foreign"},
    ],
    ids=[
        "cluster",
        "node",
        "malformed-node-list",
        "unknown-node",
        "missing-incident-id",
        "contradictory-decision",
        "unbound-event-link",
    ],
)
def test_recovered_event_cannot_return_a_foreign_incident(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    changes: dict[str, Any],
) -> None:
    store = ProbeStore()
    store.incident = store.incident.model_copy(update=changes)
    with pytest.raises(ValueError, match="incident identity"):
        run_probe(store, monkeypatch, capsys)
    assert capsys.readouterr().out == "", (
        "a foreign incident cannot emit a usable snapshot"
    )


def test_absorbed_event_may_resolve_the_incident_bound_by_its_decision(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = ProbeStore()
    store.incident = store.incident.model_copy(update={"event_id": "earlier-event"})
    result = run_probe(store, monkeypatch, capsys)
    assert result["workflow"]["request_id"] == store.workflow.request_id, (
        "a decision-bound merge link is legitimate even with a different first event"
    )


def test_incident_lookup_cannot_substitute_another_workflow(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = ProbeStore()
    store.workflow = store.workflow.model_copy(update={"request_id": "foreign"})
    with pytest.raises(ValueError, match="workflow identity"):
        run_probe(store, monkeypatch, capsys)
    assert capsys.readouterr().out == "", "a mismatched workflow cannot be returned"


@pytest.mark.parametrize(
    "field,value", [("cluster_id", "foreign"), ("node_id", "foreign")]
)
def test_live_correlation_rows_keep_the_same_cluster_and_node_boundary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    field: str,
    value: str,
) -> None:
    store = ProbeStore()
    store.events = [store.event.model_copy(update={field: value})]
    with pytest.raises(ValueError, match="correlation event identity"):
        run_probe(store, monkeypatch, capsys)
    assert store.decision_lookups == [] and store.incident_lookups == [], (
        "a foreign correlation row must not bypass reconstruction identity checks"
    )


def test_malformed_raw_payload_does_not_echo_uncontrolled_values(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = ProbeStore()
    store.records[0] = store.records[0].model_copy(
        update={
            "payload": {**store.records[0].payload, "unexpected": "private-test-value"}
        }
    )
    with pytest.raises(ValueError) as raised:
        run_probe(store, monkeypatch, capsys)
    assert str(raised.value) == "raw kernel evidence payload is invalid", (
        "model validation diagnostics must not copy raw payload values into evidence"
    )
    assert store.decision_lookups == [] and store.incident_lookups == [], (
        "invalid payloads must fail before event linkage"
    )


def test_multiple_live_correlation_matches_are_not_arbitrarily_selected(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = ProbeStore()
    store.events = [store.event, store.event.model_copy(update={"event_id": "other"})]
    with pytest.raises(ValueError, match="correlation event marker is ambiguous"):
        run_probe(store, monkeypatch, capsys)
    assert store.decision_lookups == [], (
        "ambiguity must fail before selecting a decision"
    )


def test_live_marker_prefix_collision_cannot_supply_a_workflow(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = ProbeStore()
    store.records = []
    store.events = [
        store.event.model_copy(update={"raw_message": f"marker={MARKER}-retry"})
    ]
    result = run_probe(store, monkeypatch, capsys)
    assert result["event"] is None and store.decision_lookups == [], (
        "a wider clock window must not accept a different marker's correlation row"
    )
