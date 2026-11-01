"""Provider-event correlation compares PCI devices, not spellings (batch 236).

``IncidentOrchestrator.correlate_provider_event`` skips a candidate marker
whose PCI BDFs are disjoint from the new marker's and counts shared PCI BDFs as
scope identity. Both used to compare the raw strings, and one GPU is spelled
differently by every source: the kernel's NVRM line ``0000:b9:00``, the
dcgm-exporter ``pci_bus_id`` label ``00000000:B9:00.0``, the inventory
collectors the same lower-cased. An H200 kernel XID and DCGM's report of the
same GPU were therefore never associated, even with one GPU UUID on both.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.host_health import NodeHealthCategory, NodeHealthFinding
from gpu_fault.models import IncidentState, RecoveryAction, Severity, WorkloadState
from gpu_fault.pci_identity import (
    pci_bdf_matches,
    pci_identity_overlaps,
    same_pci_device,
)
from gpu_fault.policy import FaultPolicyDecision, XidEvent
from tests._builders import build_context, fault_incident, node_health_finding

NOW = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
GPU = "GPU-h200-0"
KERNEL_PCI = "0000:b9:00"
EXPORTER_PCI = "00000000:B9:00.0"
#: The device of ``KERNEL_PCI`` as the other sources (and API clients) spell it.
SAME_DEVICE = {
    "exporter": EXPORTER_PCI,
    "inventory": "00000000:b9:00.0",
    "short-domain-function": "0000:B9:00.0",
    "no-domain": "b9:00.0",
    "identical": KERNEL_PCI,
}
#: Another device, or no PCI address at all.
NOT_THE_DEVICE = {
    "other-bus": "00000000:BA:00.0",
    "other-domain": "00000001:B9:00.0",
    "other-device": "00000000:B9:01.0",
    "unparsable-bus-only": "b9",
    "unparsable-extra-part": "0000:b9:00:00",
    "unparsable-text": "GPU-h200-0",
}


def _memory_finding(pci_bdf: str, *, gpu_uuids: list[str]) -> NodeHealthFinding:
    return node_health_finding(
        "dcgm-dbe",
        "dcgm-dbe",
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity=Severity.CRITICAL,
        reason="DCGM ECC double-bit counter increased",
        metric_name="ecc_dbe_volatile_total",
        gpu_uuids=gpu_uuids,
        pci_bdf=pci_bdf,
        recommended_action=RecoveryAction.RESET_GPU,
        policy_source="SITE_DCGM_METRIC",
        runtime_profile_version="simulated-v1",
    )


def _save_finding_marker(
    context: ApplicationContext, finding: NodeHealthFinding
) -> str:
    marker = finding.marker()
    context.completion.add_marker(marker)
    context.store.save_incident(
        fault_incident(
            marker.incident_id,
            finding.event_id,
            "NODE_HEALTH",
            finding.cluster_id,
            node_ids=[finding.node_id],
            gpu_uuids=finding.gpu_uuids,
            policy_version=finding.policy_version,
            policy_source=finding.policy_source,
            effective_action=finding.recommended_action,
            state=IncidentState.DETECTED,
            reasons=[finding.reason],
            created_at=finding.observed_at,
            updated_at=finding.observed_at,
        )
    )
    return marker.incident_id


def _xid48(
    event_id: str, *, source: str, pci_bdf: str, gpu_uuid: str | None, seconds: int = 10
) -> XidEvent:
    return XidEvent(
        event_id=event_id,
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=NOW + timedelta(seconds=seconds),
        event_source=source,
        xid=48,
        gpu_uuid=gpu_uuid,
        pci_bdf=pci_bdf,
        product="NVIDIA H200",
        driver_branch=580,
        cuda_version="13.0",
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.IDLE,
    )


def _correlate(context: ApplicationContext, event: XidEvent) -> FaultPolicyDecision:
    return context.orchestrator.correlate_provider_event(
        context.policy.evaluate_xid(event), cluster_id="cluster-a"
    )


@pytest.mark.parametrize("spelling", sorted(SAME_DEVICE))
def test_a_kernel_xid_meets_the_dcgm_marker_of_its_gpu_in_any_spelling(
    spelling: str,
) -> None:
    context = build_context()
    incident_id = _save_finding_marker(
        context, _memory_finding(SAME_DEVICE[spelling], gpu_uuids=[GPU])
    )

    correlated = _correlate(
        context,
        _xid48("kernel-xid48", source="KERNEL_LOG", pci_bdf=KERNEL_PCI, gpu_uuid=GPU),
    )

    assert correlated.duplicate, (
        f"the DCGM marker spelled {SAME_DEVICE[spelling]!r} names the GPU of the "
        f"kernel line's {KERNEL_PCI!r}; the GPU UUID matches too"
    )
    assert correlated.marker.incident_id == incident_id


@pytest.mark.parametrize("spelling", sorted(SAME_DEVICE))
def test_pci_alone_names_the_gpu_when_neither_side_has_a_uuid(spelling: str) -> None:
    """No GPU UUID and no shared correlation key (``pci:0000:b9:00`` is not
    ``pci:00000000:b9:00.0``): the device identity is the PCI address only."""

    context = build_context()
    incident_id = _save_finding_marker(
        context, _memory_finding(SAME_DEVICE[spelling], gpu_uuids=[])
    )

    correlated = _correlate(
        context,
        _xid48("kernel-xid48", source="KERNEL_LOG", pci_bdf=KERNEL_PCI, gpu_uuid=None),
    )

    assert correlated.duplicate, f"{SAME_DEVICE[spelling]!r} is the same device"
    assert correlated.marker.incident_id == incident_id


@pytest.mark.parametrize("spelling", sorted(NOT_THE_DEVICE))
def test_another_device_or_an_unreadable_address_is_never_associated(
    spelling: str,
) -> None:
    context = build_context()
    _save_finding_marker(
        context, _memory_finding(NOT_THE_DEVICE[spelling], gpu_uuids=[])
    )

    correlated = _correlate(
        context,
        _xid48("kernel-xid48", source="KERNEL_LOG", pci_bdf=KERNEL_PCI, gpu_uuid=None),
    )

    assert not correlated.duplicate, (
        f"{NOT_THE_DEVICE[spelling]!r} must not join {KERNEL_PCI!r}: another "
        "device, or an address the comparison cannot read (fail closed)"
    )
    assert correlated.marker.incident_id == "inc-kernel-xid48"


def test_a_differing_function_is_another_device() -> None:
    context = build_context()
    _save_finding_marker(context, _memory_finding(EXPORTER_PCI, gpu_uuids=[]))

    correlated = _correlate(
        context,
        _xid48(
            "kernel-xid48", source="KERNEL_LOG", pci_bdf="0000:b9:00.1", gpu_uuid=None
        ),
    )

    assert not correlated.duplicate, "function 1 is not the GPU at function 0"


def test_an_unreadable_address_still_matches_only_its_identical_spelling() -> None:
    """Fail closed both ways: an unparsable string keeps today's exact match
    and gains no case-insensitive or partial one."""

    context = build_context()
    _save_finding_marker(context, _memory_finding("gpu-slot-b9", gpu_uuids=[]))

    identical = _correlate(
        context,
        _xid48(
            "kernel-same", source="KERNEL_LOG", pci_bdf="gpu-slot-b9", gpu_uuid=None
        ),
    )
    other_case = _correlate(
        context,
        _xid48(
            "kernel-case", source="KERNEL_LOG", pci_bdf="GPU-SLOT-B9", gpu_uuid=None
        ),
    )

    assert identical.duplicate, "an identical string is one device, as before"
    assert not other_case.duplicate, "an unreadable spelling gains no case folding"


@pytest.mark.parametrize(
    ("first", "second"),
    [("KERNEL_LOG", "DCGM_EXPORTER"), ("DCGM_EXPORTER", "KERNEL_LOG")],
    ids=["kernel-then-gauge", "gauge-then-kernel"],
)
def test_the_kernel_line_and_dcgm_field_230_of_one_xid_are_associated(
    first: str, second: str
) -> None:
    """The live H200 pair: DCGM field 230 reports the XID the kernel logged,
    15-30 s apart, each spelled by its own source."""

    spellings = {"KERNEL_LOG": KERNEL_PCI, "DCGM_EXPORTER": EXPORTER_PCI}
    context = build_context()
    earlier = _xid48(
        f"{first.lower()}-xid48",
        source=first,
        pci_bdf=spellings[first],
        gpu_uuid=GPU,
        seconds=0,
    )
    decision = _correlate(context, earlier)
    incident, _workflow = context.orchestrator.ingest(earlier, decision)
    context.completion.add_marker(
        decision.marker.model_copy(update={"incident_id": incident.incident_id})
    )

    correlated = _correlate(
        context,
        _xid48(
            f"{second.lower()}-xid48",
            source=second,
            pci_bdf=spellings[second],
            gpu_uuid=GPU,
            seconds=20,
        ),
    )

    assert correlated.duplicate, f"{second} must be associated with {first}"
    assert correlated.marker.incident_id == incident.incident_id


@pytest.mark.parametrize(
    ("left", "right", "same"),
    [
        (KERNEL_PCI, EXPORTER_PCI, True),
        ("0000:B9:00", "00000000:b9:00.0", True),
        (" 0000:b9:00 ", EXPORTER_PCI, True),
        ("b9:00", "00000001:B9:00.0", True),
        ("0000:b9:00.0", "0000:b9:00.1", False),
        (KERNEL_PCI, "00000000:BA:00.0", False),
        ("not-a-bdf", "not-a-bdf", True),
        ("not-a-bdf", "NOT-A-BDF", False),
        ("not-a-bdf", KERNEL_PCI, False),
    ],
)
def test_same_pci_device_reads_the_device_not_the_string(
    left: str, right: str, same: bool
) -> None:
    assert same_pci_device(left, right) is same, (left, right)
    assert same_pci_device(right, left) is same, "the relation is symmetric"
    assert pci_identity_overlaps([left], {right}) is same, (left, right)


def test_only_an_identical_string_matches_an_unreadable_address() -> None:
    assert not pci_bdf_matches("not-a-bdf", "not-a-bdf"), "it reads addresses only"
    assert same_pci_device("not-a-bdf", "not-a-bdf"), "the exact match is kept"


def test_overlap_needs_one_shared_device_and_is_false_for_an_empty_side() -> None:
    assert pci_identity_overlaps(["00000000:BA:00.0", KERNEL_PCI], [EXPORTER_PCI]), (
        "one shared device is enough"
    )
    assert not pci_identity_overlaps([], [EXPORTER_PCI]), "no address, no device"
    assert not pci_identity_overlaps([KERNEL_PCI], []), "no address, no device"
