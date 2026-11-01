"""XID companions are matched by PCI device, not by spelling (batch 236).

``GpuFaultPolicyEngine`` pairs an XID with the companions it names on the same
GPU: XID 48 with 63/64, 64 back to 48, 45 with its cause, 154 with the XID it
summarises. The GPU UUID decides when both events carry one; otherwise the PCI
address does, and it was compared as a lower-cased string, so a kernel line
whose GPU UUID could not be resolved (``PCI:0000:b9:00``) never met DCGM
field 230's report of the same GPU (``00000000:B9:00.0``).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.policy import GpuFaultPolicyEngine, XidEvent

NOW = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
GPU = "GPU-h200-0"
KERNEL_PCI = "0000:b9:00"


def _xid(
    event_id: str,
    xid: int,
    *,
    source: str,
    pci_bdf: str,
    gpu_uuid: str | None,
    seconds: int,
) -> XidEvent:
    return XidEvent(
        event_id=event_id,
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=NOW + timedelta(seconds=seconds),
        event_source=source,
        xid=xid,
        gpu_uuid=gpu_uuid,
        pci_bdf=pci_bdf,
        product="NVIDIA H200",
        driver_branch=580,
        cuda_version="13.0",
        runtime_profile_version="simulated-v1",
    )


def _xid48_beside_gauge_xid64(companion_pci: str):
    companion = _xid(
        "gauge-xid64",
        64,
        source="DCGM_EXPORTER",
        pci_bdf=companion_pci,
        gpu_uuid=GPU,
        seconds=0,
    )
    event = _xid(
        "kernel-xid48",
        48,
        source="KERNEL_LOG",
        pci_bdf=KERNEL_PCI,
        gpu_uuid=None,
        seconds=5,
    )
    return GpuFaultPolicyEngine().evaluate_xid(
        event, companion_events=[companion, event]
    )


@pytest.mark.parametrize("companion_pci", ["00000000:B9:00.0", "0000:B9:00.0"])
def test_an_xid48_without_a_uuid_takes_its_xid64_companion_in_any_spelling(
    companion_pci: str,
) -> None:
    decision = _xid48_beside_gauge_xid64(companion_pci)

    assert decision.official_action == "DRAIN_AND_RESET", decision.reasons
    assert decision.correlated_event_id == "gauge-xid64", (
        "the XID 48 shares the reset of the XID 64 on its GPU"
    )


@pytest.mark.parametrize(
    "companion_pci",
    ["00000000:BA:00.0", "00000001:B9:00.0", "b9", "gpu-slot-b9"],
    ids=["other-bus", "other-domain", "unparsable-bus", "unparsable-text"],
)
def test_another_device_or_an_unreadable_address_is_no_companion(
    companion_pci: str,
) -> None:
    decision = _xid48_beside_gauge_xid64(companion_pci)

    assert decision.official_action == "RESET_GPU", decision.reasons
    assert decision.correlated_event_id is None, (
        f"{companion_pci!r} is not the GPU of {KERNEL_PCI!r}"
    )
