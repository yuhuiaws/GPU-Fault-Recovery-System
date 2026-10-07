"""The XID-injecting cases refuse a target node whose stored GPU identity names
no product: such an XID is quarantined as NOT_APPLICABLE on the control plane
instead of starting the case's workflow (eks-managed DESTR-015 attempt 10)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.gpu_metrics import (
    GpuInventoryDevice,
    GpuInventorySnapshot,
    GpuMetricSource,
)
from scripts.e2e.regional.gpu_identity_preflight import (
    GPU_IDENTITY_MAX_AGE_SECONDS,
    gpu_identity_errors,
)
from scripts.e2e.regional.regional_live_fixture import STORE_PROBE
from tests.regional.test_shared_runner_raw_evidence_integration import ProbeStore

NOW = datetime(2026, 10, 7, 8, 0, tzinfo=timezone.utc)
NODE = "node-a"


def _identity(**overrides: Any) -> dict[str, Any]:
    return {
        "snapshot_id": "gpu-inventory-node-a-1",
        "observed_at": (NOW - timedelta(seconds=30)).isoformat(),
        "source": "NVIDIA_SMI",
        "source_boot_id": "boot-current",
        "device_count": 8,
        "products": ["H200"],
        "product": "H200",
        **overrides,
    }


def test_a_fresh_same_boot_single_product_identity_passes() -> None:
    assert gpu_identity_errors(NODE, _identity(), boot_id="boot-current", now=NOW) == []
    assert gpu_identity_errors(NODE, _identity(), now=NOW) == [], (
        "without the node's boot id the boot comparison is skipped, not failed"
    )


def test_every_way_the_identity_cannot_carry_the_case_is_named() -> None:
    assert gpu_identity_errors(NODE, None, now=NOW) == [
        f"{NODE} has no GPU inventory identity in the store (its nvidia-smi "
        "inventory was never delivered); a product-less XID from it would be "
        "quarantined as NOT_APPLICABLE instead of starting the workflow"
    ]
    productless = gpu_identity_errors(
        NODE, _identity(product=None, products=["H200", "B200"]), now=NOW
    )
    assert productless == [
        f"{NODE} GPU inventory identity names no single product: "
        "products=['H200', 'B200'] devices=8"
    ]
    rebooted = gpu_identity_errors(
        NODE, _identity(), boot_id="boot-after-reboot", now=NOW
    )
    assert len(rebooted) == 1 and "boot-after-reboot" in rebooted[0], rebooted
    assert "boot-current" in rebooted[0], rebooted
    stale = gpu_identity_errors(
        NODE,
        _identity(
            observed_at=(
                NOW - timedelta(seconds=GPU_IDENTITY_MAX_AGE_SECONDS + 1)
            ).isoformat()
        ),
        now=NOW,
    )
    assert len(stale) == 1 and f"limit {GPU_IDENTITY_MAX_AGE_SECONDS} s" in stale[0], (
        stale
    )
    unreadable = gpu_identity_errors(NODE, _identity(observed_at="yesterday"), now=NOW)
    assert unreadable == [f"{NODE} GPU inventory identity has no readable observed_at"]
    zulu = gpu_identity_errors(
        NODE, _identity(observed_at="2026-10-07T07:59:40Z"), now=NOW
    )
    assert zulu == [], "a Z-suffixed timestamp is read as UTC"


class _InventoryStore(ProbeStore):
    def __init__(self, snapshot: GpuInventorySnapshot | None) -> None:
        super().__init__()
        self._snapshot = snapshot

    def get_gpu_inventory_snapshot(self, cluster_id: str, node_id: str):
        assert (cluster_id, node_id) == ("cluster-a", NODE), (cluster_id, node_id)
        return self._snapshot


def _run_probe(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    snapshot: GpuInventorySnapshot | None,
) -> dict[str, Any]:
    """Drive the probe source the way the Pod does (eight positional arguments)."""

    monkeypatch.setattr(
        ApplicationContext,
        "from_environment",
        classmethod(lambda _cls: SimpleNamespace(store=_InventoryStore(snapshot))),
    )
    monkeypatch.setattr(
        sys, "argv", ["store-probe", "cluster-a", NODE, "", "", "", "", "", "1"]
    )
    exec(
        compile(STORE_PROBE, "<gpu-identity-store-probe>", "exec"),
        {"__name__": "__probe__"},
    )
    return json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("second_product", ["H200", "B200", None])
def test_store_probe_reports_the_nodes_gpu_inventory_identity(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    second_product: str | None,
) -> None:
    snapshot = GpuInventorySnapshot(
        snapshot_id="gpu-inventory-node-a-7",
        cluster_id="cluster-a",
        node_id=NODE,
        observed_at=NOW,
        source=GpuMetricSource.NVIDIA_SMI,
        source_boot_id="boot-current",
        devices=[
            GpuInventoryDevice(
                gpu_index=0, gpu_uuid="GPU-a", pci_bdf="0000:01:00.0", product="H200"
            ),
            GpuInventoryDevice(
                gpu_index=1,
                gpu_uuid="GPU-b",
                pci_bdf="0000:02:00.0",
                product=second_product,
            ),
        ],
    )
    identity = _run_probe(monkeypatch, capsys, snapshot)["gpu_identity"]
    assert identity["snapshot_id"] == "gpu-inventory-node-a-7"
    assert identity["source_boot_id"] == "boot-current"
    assert identity["observed_at"] == NOW.isoformat()
    assert identity["device_count"] == 2
    if second_product == "H200":
        assert (identity["product"], identity["products"]) == ("H200", ["H200"])
    else:
        assert identity["product"] is None, (
            "a mixed or partly unnamed inventory names no single product"
        )
    assert gpu_identity_errors(NODE, identity, boot_id="boot-current", now=NOW) == (
        []
        if second_product == "H200"
        else [
            f"{NODE} GPU inventory identity names no single product: "
            f"products={identity['products']} devices=2"
        ]
    )


def test_store_probe_reports_no_identity_without_an_inventory(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run_probe(monkeypatch, capsys, None)["gpu_identity"] is None
