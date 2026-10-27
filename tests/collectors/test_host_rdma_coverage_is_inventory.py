"""An EFA function the driver does not hold is the inventory's verdict, not a
collection error of the RDMA traffic contributor.

Live 2026-10-27 (GF-GENERAL-LIVE-013 ``post_repair_failure``, both clusters):
with one of 16 functions unbound, every host batch carried ``_rdma``'s
"incomplete EFA traffic coverage: 15/16 devices", so HOST_TELEMETRY stayed
COLLECTION_ERROR for as long as the node was faulty and the post-repair
VALIDATE_FABRIC read no inventory from it. The shortfall is already the
inventory contributor's ``efa_inventory_mismatch`` (DRIVER_UNBOUND here); the
traffic contributor keeps withholding the aggregate it cannot complete.
"""

from __future__ import annotations

from tests.collectors import _cov95_runtime_collect_host as support

host_case = support.host_case
isolated_runtime = support.isolated_runtime


def _efa_function(case, index: int, *, bound: bool) -> None:
    pci = case.roots.pci / f"0000:{index:02x}:00.0"
    support.write_file(pci / "vendor", "0x1d0f")
    support.write_file(pci / "device", "0xefa2")
    if not bound:
        return
    device = case.roots.rdma / f"rdmap{index}s0"
    support.write_file(device / "device/uevent", "DRIVER=efa\n")
    port = device / "ports/1"
    support.write_file(port / "state", "4: ACTIVE")
    support.write_file(port / "phys_state", "5: LinkUp")
    for name in ("rx_bytes", "tx_bytes"):
        support.write_file(port / "hw_counters" / name, "100")


def test_an_unbound_function_is_an_inventory_verdict_not_a_collection_error(
    host_case,
) -> None:
    _efa_function(host_case, 1, bound=True)
    _efa_function(host_case, 2, bound=False)
    collector = host_case.build(
        "_efa_inventory",
        "_rdma",
        expected_efa_device_count=2,
        inventory_mismatch_consecutive_samples=1,
    )

    batch = collector.collect_once()

    assert batch.collection_errors == [], batch.collection_errors
    values = support.values(batch)
    assert values[("efa_inventory_active_count", None)] == 1
    assert values[("efa_inventory_mismatch", None)] == 1
    mismatch = next(
        item for item in batch.samples if item.name == "efa_inventory_mismatch"
    )
    assert mismatch.labels["failure_mode"] == "DRIVER_UNBOUND"
    # The aggregate covers one of two functions: withheld, not reported short.
    assert values[("efa_traffic_observation_valid", None)] == 0
    assert ("efa_traffic_bytes_per_second", None) not in values


def test_every_function_held_reads_complete_traffic(host_case) -> None:
    _efa_function(host_case, 1, bound=True)
    _efa_function(host_case, 2, bound=True)
    collector = host_case.build(
        "_efa_inventory",
        "_rdma",
        expected_efa_device_count=2,
        inventory_mismatch_consecutive_samples=1,
    )

    first = collector.collect_once()
    host_case.clock.sleep(15)
    second = collector.collect_once()

    assert first.collection_errors == [] and second.collection_errors == []
    values = support.values(second)
    assert values[("efa_inventory_mismatch", None)] == 0
    # Two ticks give every port its traffic baseline: the aggregate is complete.
    assert values[("efa_traffic_observation_valid", None)] == 1
