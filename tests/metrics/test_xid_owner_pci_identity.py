"""DCGM's XID-ownership checks find the kernel line's GPU by PCI device (batch 236).

Field 1523 (``gpu_recovery_action_policy.xid154_owner``) and the NVLink5 fields
(``gpu_nvlink5_policy.nvlink_xid_owner``) are recorded as evidence while a
kernel XID of their GPU owns its recovery. The GPU UUID decides when both the
XID event and the sample carry one; a kernel line whose UUID could not be
resolved is matched by its PCI address, which was compared as a lower-cased
string -- the kernel's ``0000:59:00`` never named the exporter's
``00000000:59:00.0``, and the sample graded a second plan beside the XID's.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.gpu_metrics import (
    GpuMetricBatch,
    GpuMetricSample,
    GpuMetricSource,
    GpuMetricsService,
)
from gpu_fault.gpu_nvlink5_policy import NVLINK5_FIELDS, NVLINK5_RECOVERY_FAILED
from gpu_fault.models import RecoveryAction
from gpu_fault.policy import XidEvent
from gpu_fault.store import InMemoryStore

NOW = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
CLUSTER = "cluster-a"
NODE = "node-a"
GPU = "GPU-a"
EXPORTER_PCI = "00000000:59:00.0"
XID154_LINE = (
    "NVRM: Xid (PCI:{pci}): 154, GPU recovery action changed from 0x0 (None) "
    "to 0x1 (GPU Reset Required)"
)


def _batch(seconds: float, values: dict[str, float], product: str) -> GpuMetricBatch:
    return GpuMetricBatch(
        batch_id=f"owner-{product}-{seconds}",
        cluster_id=CLUSTER,
        node_id=NODE,
        product=product,
        observed_at=NOW + timedelta(seconds=seconds),
        source=GpuMetricSource.DCGM_EXPORTER,
        runtime_profile_version="simulated-v1",
        samples=[
            GpuMetricSample(
                metric_name=NVLINK5_FIELDS[name][1] if name in NVLINK5_FIELDS else name,
                canonical_name=name,
                value=value,
                gpu_uuid=GPU,
                pci_bdf=EXPORTER_PCI,
                labels={"source_boot_id": "boot-a", "modelName": product},
            )
            for name, value in values.items()
        ],
    )


def _kernel_xid(xid: int, pci: str) -> XidEvent:
    """A kernel line whose GPU UUID could not be resolved: PCI only."""

    return XidEvent(
        event_id=f"kernel-{xid}",
        cluster_id=CLUSTER,
        node_id=NODE,
        observed_at=NOW - timedelta(seconds=5),
        event_source="KERNEL_LOG",
        xid=xid,
        pci_bdf=pci,
        raw_message=(
            XID154_LINE.format(pci=pci)
            if xid == 154
            else f"NVRM: Xid (PCI:{pci}): {xid}, synthetic"
        ),
    )


def _graded(store: InMemoryStore, name: str, product: str, before: float | None):
    """The finding of ``name`` once it reads 1 (after ``before``, if a counter)."""

    metrics = GpuMetricsService(store=store)
    if before is not None:
        metrics.ingest(_batch(-15, {name: before}, product))
    result = metrics.ingest(_batch(0, {name: 1}, product))
    finding = next(
        (item for item in result.findings if item.canonical_name == name), None
    )
    assert finding is not None, f"{name} is graded: {result.findings}"
    return finding


@pytest.mark.parametrize(
    ("pci", "owned"),
    [("0000:59:00", True), ("0000:59:00.0", True), ("0000:5a:00", False)],
    ids=["kernel-spelling", "with-function", "another-gpu"],
)
def test_a_kernel_xid154_without_a_uuid_owns_the_gpu_its_address_names(
    pci: str, owned: bool
) -> None:
    store = InMemoryStore()
    store.save_xid_event_if_absent(_kernel_xid(154, pci))

    finding = _graded(store, "gpu_recovery_action", "H200", before=None)

    expected = RecoveryAction.NO_ACTION if owned else RecoveryAction.RESET_GPU
    assert finding.automatic_action == expected.value, (pci, finding.reason)
    if owned:
        assert finding.threshold_source == "XID_154_OWNS_RECOVERY", finding.reason
        assert "XID 154 event kernel-154" in finding.reason


@pytest.mark.parametrize(
    ("pci", "owned"),
    [("0000:59:00", True), ("0000:5a:00", False)],
    ids=["kernel-spelling", "another-gpu"],
)
def test_a_kernel_nvlink5_xid_without_a_uuid_owns_the_gpu_its_address_names(
    pci: str, owned: bool
) -> None:
    store = InMemoryStore()
    store.save_xid_event_if_absent(_kernel_xid(149, pci))

    finding = _graded(store, NVLINK5_RECOVERY_FAILED, "NVIDIA B200", before=0)

    expected = RecoveryAction.NO_ACTION if owned else RecoveryAction.DRAIN
    assert finding.automatic_action == expected.value, (pci, finding.reason)
    if owned:
        assert finding.threshold_source == "NVLINK5_XID_OWNS_GPU", finding.reason
