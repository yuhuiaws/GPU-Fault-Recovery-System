"""A product-less XID takes its product from the node's fresh GPU inventory.

Live (eks-managed, DESTR-015 attempt 10): the node rebooted by DESTR-025 had
its kernel collector restart before the driver answered, so the injected XID 46
arrived with ``product=None`` and the product gate quarantined the node as
"not applicable to product UNKNOWN" while the sibling node's event started the
reset. The node's own GPU inventory (nvidia-smi on the node, same boot) is the
same hardware's identity and stands in -- fresh, same boot, one product, or the
gate fails closed exactly as before.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.app import default_simulated_profile
from gpu_fault.app.ingest.faults import FaultIngestionService
from gpu_fault.gpu_metrics import (
    GpuInventoryDevice,
    GpuInventorySnapshot,
    GpuMetricSource,
)
from gpu_fault.policy import ActionDisposition
from gpu_fault.store import InMemoryStore
from tests._builders import build_context
from tests.orchestration.test_provider_correlation_arbitration import NOW, xid

BOOT = "boot-after-reboot"


def _inventory(
    *,
    observed_at=NOW,
    boot_id: str = BOOT,
    products: tuple[str | None, ...] = ("H200", "H200"),
) -> GpuInventorySnapshot:
    return GpuInventorySnapshot(
        snapshot_id="gpu-inventory-node-a-1",
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=observed_at,
        source=GpuMetricSource.NVIDIA_SMI,
        source_boot_id=boot_id,
        devices=[
            GpuInventoryDevice(
                gpu_index=index,
                gpu_uuid=f"GPU-{index}",
                pci_bdf=f"0000:0{index + 1}:00.0",
                product=product,
            )
            for index, product in enumerate(products)
        ],
    )


def _service(snapshot: GpuInventorySnapshot | None) -> FaultIngestionService:
    store = InMemoryStore()
    store.save_profile(default_simulated_profile())
    if snapshot is not None:
        store.save_gpu_inventory_snapshot(snapshot)
    return FaultIngestionService(build_context(store=store))


def _productless_xid(event_id: str, *, boot_id: str | None = BOOT):
    return xid(event_id, at=NOW + timedelta(seconds=1), active=False).model_copy(
        update={"product": None, "source_boot_id": boot_id}
    )


def test_the_nodes_fresh_inventory_supplies_the_missing_product() -> None:
    service = _service(_inventory())
    decision = service.ingest_xid(_productless_xid("xid-after-reboot"))

    assert decision.disposition is not ActionDisposition.NOT_APPLICABLE, decision
    assert not any("product UNKNOWN" in reason for reason in decision.reasons), (
        decision.reasons
    )
    stored = service.context.store.get_xid_event("xid-after-reboot")
    assert stored.product == "H200"
    assert stored.gpu_identity_source == "GPU_INVENTORY:gpu-inventory-node-a-1", (
        "the event must record that the collector did not stamp the product"
    )


def test_a_collector_stamped_product_is_never_overridden() -> None:
    service = _service(_inventory(products=("B200", "B200")))
    event = xid("xid-stamped", at=NOW + timedelta(seconds=1), active=False)
    service.ingest_xid(event)

    stored = service.context.store.get_xid_event("xid-stamped")
    assert (stored.product, stored.gpu_identity_source) == ("H200", None)


@pytest.mark.parametrize(
    ("snapshot", "why"),
    [
        (None, "no inventory at all"),
        (_inventory(boot_id="boot-before-reboot"), "inventory from the previous boot"),
        (_inventory(observed_at=NOW - timedelta(hours=1)), "stale inventory"),
        (_inventory(products=("H200", "B200")), "mixed products"),
        (_inventory(products=("H200", None)), "a device without a product"),
    ],
)
def test_anything_less_than_a_fresh_same_boot_single_product_fails_closed(
    snapshot: GpuInventorySnapshot | None, why: str
) -> None:
    service = _service(snapshot)
    decision = service.ingest_xid(_productless_xid("xid-unresolved"))

    assert decision.disposition is ActionDisposition.NOT_APPLICABLE, (why, decision)
    assert any("product UNKNOWN" in reason for reason in decision.reasons), (
        why,
        decision.reasons,
    )
    stored = service.context.store.get_xid_event("xid-unresolved")
    assert stored.product is None, why
    assert stored.gpu_identity_source == "NO_FRESH_GPU_INVENTORY", why
