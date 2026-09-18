"""A fault line the collector matched but the normalizer could not read.

The kernel collector forwards every ``NVRM ... Xid`` line and the Fabric
Manager collector every ``SXid`` summary. When the ingest side cannot extract a
code the line used to land as evidence with ``status=success`` and a reason
buried in a response body nobody reads. A catalog or format drift could
therefore swallow real XIDs silently. The normalizer now names the line as
unparsed, and the fault ingestion service turns that into a WARNING log, a
counter and a node-health finding that routes to operator review -- without
inventing a code.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from gpu_fault.app.ingest.faults import FaultIngestionService
from gpu_fault.models import RecoveryAction, Severity
from gpu_fault.nvidia_logs import (
    UNPARSED_SXID_REASON,
    UNPARSED_XID_REASON,
    FabricManagerLogEvent,
    NvidiaKernelLogEvent,
    NvidiaLogNormalizer,
)
from tests._builders import build_context

NOW = datetime(2026, 9, 7, 10, 0, tzinfo=timezone.utc)


def _kernel_event(record_id: str, message: str) -> NvidiaKernelLogEvent:
    return NvidiaKernelLogEvent(
        cluster_id="hp-cluster",
        node_id="worker-1",
        record_id=record_id,
        observed_at=NOW,
        message=message,
        product="H100",
    )


def test_kernel_xid_line_without_a_code_is_named_unparsed() -> None:
    result = NvidiaLogNormalizer().normalize_kernel(
        _kernel_event("kmsg-no-code", "NVRM: Xid (PCI:0000:b9:00): , pid=1234")
    )

    assert result.xid_events == [], "a code was invented for an unparsable line"
    reasons = result.provider_signals[0].unresolved_reasons
    assert any(reason.startswith(UNPARSED_XID_REASON) for reason in reasons), reasons


def test_kernel_xid_line_with_a_code_is_not_flagged() -> None:
    result = NvidiaLogNormalizer().normalize_kernel(
        _kernel_event("kmsg-79", "NVRM: Xid (PCI:0000:b9:00): 79, GPU has fallen")
    )

    assert [item.xid for item in result.xid_events] == [79]
    assert result.provider_signals[0].unresolved_reasons == []


def test_fabric_manager_sxid_line_without_a_code_is_named_unparsed() -> None:
    result = NvidiaLogNormalizer().normalize_fabric_manager(
        FabricManagerLogEvent(
            cluster_id="hp-cluster",
            node_id="worker-1",
            record_id="fm-no-code",
            observed_at=NOW,
            source="journal",
            message="nvidia-nvswitch3: SXid (PCI:0000:c1:00.0): garbled, Fatal",
            product="H200",
        )
    )

    assert result.sxid_events == []
    reasons = result.provider_signals[0].unresolved_reasons
    assert any(reason.startswith(UNPARSED_SXID_REASON) for reason in reasons), reasons


def test_unparsed_signal_becomes_a_warning_finding_and_counter(caplog) -> None:
    context = build_context()
    service = FaultIngestionService(context)
    normalizer = NvidiaLogNormalizer()
    unparsed = normalizer.normalize_kernel(
        _kernel_event("kmsg-drift-1", "NVRM: Xid (PCI:0000:b9:00): , pid=1234")
    )

    with caplog.at_level(logging.WARNING, logger="gpu_fault.app.ingest.faults"):
        result = service.ingest_unresolved_signals(unparsed, batch_id="kmsg-drift-1")

    assert result is not None, "an unparsed fault line produced no finding"
    assert [item.metric_name for item in result.findings] == [UNPARSED_XID_REASON]
    finding = result.findings[0]
    assert finding.severity is Severity.WARNING
    assert finding.recommended_action is RecoveryAction.COLLECT_EVIDENCE
    assert finding.node_id == "worker-1"
    assert len(result.incident_ids) == 1, "the finding did not open an incident"
    assert result.notification_ids, "operator review was not notified"
    notification = context.store.get_notification(result.notification_ids[0])
    assert notification.incident_id == result.incident_ids[0]
    assert "worker-1" in notification.subject
    assert service.unresolved_signal_totals == {UNPARSED_XID_REASON: 1}
    assert any(
        record.levelno == logging.WARNING and "worker-1" in record.getMessage()
        for record in caplog.records
    ), "no WARNING named the node whose fault line could not be parsed"


def test_unparsed_episode_emits_once_until_a_parsed_line_clears_it() -> None:
    context = build_context()
    service = FaultIngestionService(context)
    normalizer = NvidiaLogNormalizer()

    first = service.ingest_unresolved_signals(
        normalizer.normalize_kernel(
            _kernel_event("kmsg-drift-1", "NVRM: Xid (PCI:0000:b9:00): , a")
        ),
        batch_id="kmsg-drift-1",
    )
    second = service.ingest_unresolved_signals(
        normalizer.normalize_kernel(
            _kernel_event("kmsg-drift-2", "NVRM: Xid (PCI:0000:b9:00): , b")
        ),
        batch_id="kmsg-drift-2",
    )
    assert first is not None and first.incident_ids, "first line opened nothing"
    assert second is None or not second.incident_ids, (
        "a second unparsed line in the same episode opened another incident"
    )
    assert service.unresolved_signal_totals == {UNPARSED_XID_REASON: 2}

    cleared = service.ingest_unresolved_signals(
        normalizer.normalize_kernel(
            _kernel_event("kmsg-79", "NVRM: Xid (PCI:0000:b9:00): 79, ok")
        ),
        batch_id="kmsg-79",
    )
    assert cleared is None, "a parsed line minted a finding"

    third = service.ingest_unresolved_signals(
        normalizer.normalize_kernel(
            _kernel_event("kmsg-drift-3", "NVRM: Xid (PCI:0000:b9:00): , c")
        ),
        batch_id="kmsg-drift-3",
    )
    assert third is not None and third.incident_ids, (
        "a new drift episode after a parsed line stayed silent"
    )


def _workflow_for(context, incident_id: str):
    workflows = [
        item
        for item in context.store.list_workflows()
        if item.incident_id == incident_id
    ]
    assert len(workflows) == 1, f"expected one workflow for {incident_id}"
    return workflows[0]


def test_unparsed_finding_carries_the_records_runtime_profile_and_compiles() -> None:
    """Without a profile the health family cannot even freeze evidence.

    The live finding was built with no ``runtime_profile_version``; the
    coordinator answered "runtime_profile_version is required for execution"
    and "no executable owner for evidenceCapture", so the workflow the
    operator-review notification pointed at had zero steps. The kernel record
    names its profile; the finding has to carry it through.
    """

    context = build_context()
    service = FaultIngestionService(context)
    unparsed = NvidiaLogNormalizer().normalize_kernel(
        _kernel_event(
            "kmsg-drift-2", "NVRM: Xid (PCI:0000:b9:00): , pid=1234"
        ).model_copy(update={"runtime_profile_version": "simulated-v1"})
    )

    result = service.ingest_unresolved_signals(unparsed, batch_id="kmsg-drift-2")

    assert result is not None, "an unparsed fault line produced no finding"
    finding = result.findings[0]
    assert finding.runtime_profile_version == "simulated-v1", (
        "the finding must execute under the profile the record was collected with"
    )
    workflow = _workflow_for(context, result.incident_ids[0])
    operations = [step.operation.value for step in workflow.official_steps]
    assert operations == ["FREEZE_EVIDENCE"], (
        f"the unparsed-line workflow must compile to freeze-only, got {operations}"
    )
    assert not workflow.blocked_reasons, workflow.blocked_reasons


def test_unparsed_finding_falls_back_to_the_node_agents_profile() -> None:
    """A record with no profile of its own still names one.

    Topology knows nothing about an idle node, so the last word is the node
    agent's registration -- the profile the collectors on that node were
    installed under.
    """

    from types import SimpleNamespace

    from tests._builders import build_store

    store = build_store(
        get_agent=lambda cluster_id, node_id: SimpleNamespace(
            runtime_profile_version="simulated-v1"
        )
    )
    context = build_context(store=store)
    service = FaultIngestionService(context)
    unparsed = NvidiaLogNormalizer().normalize_kernel(
        _kernel_event("kmsg-drift-3", "NVRM: Xid (PCI:0000:b9:00): , pid=1234")
    )
    assert unparsed.provider_signals[0].runtime_profile_version is None, (
        "the record itself must name no profile for the fallback to be exercised"
    )

    result = service.ingest_unresolved_signals(unparsed, batch_id="kmsg-drift-3")

    assert result is not None, "an unparsed fault line produced no finding"
    assert result.findings[0].runtime_profile_version == "simulated-v1", (
        "the node agent's registered profile is the fallback"
    )
    workflow = _workflow_for(context, result.incident_ids[0])
    assert [step.operation.value for step in workflow.official_steps] == [
        "FREEZE_EVIDENCE"
    ], "the fallback profile must let the freeze-only workflow compile"
