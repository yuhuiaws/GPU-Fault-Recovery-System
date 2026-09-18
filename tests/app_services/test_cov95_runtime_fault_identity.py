from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.channel_registry import FABRIC_MANAGER_PATH, NVIDIA_KERNEL_PATH
from gpu_fault.nvidia_logs import FabricManagerLogEvent, NvidiaKernelLogEvent
from tests._builders import build_context
from tests.app_services._cov95_runtime_ingest import NOW, inventory, legacy_metric, post
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("fresh", "GPU-a"),
        ("full-pci", "GPU-a"),
        ("other-function", None),
        ("ambiguous", None),
        ("invalid-device", None),
        ("invalid-event", None),
        ("no-pci", None),
        ("future", None),
        ("stale", None),
        ("boot-drift", None),
        ("legacy", "GPU-a"),
        ("legacy-ambiguous", None),
        ("legacy-stale", None),
        ("legacy-future", None),
        ("legacy-no-pci", None),
        ("legacy-invalid-pci", None),
        ("legacy-no-uuid", None),
        ("legacy-other-pci", None),
        ("legacy-other-function", None),
        ("none", None),
    ],
)
def test_kernel_gpu_identity_requires_one_fresh_matching_pci_device(
    source: str, expected: str | None
) -> None:
    context = build_context()
    pci = "0000:b9:00"
    if source in {"full-pci", "other-function", "legacy-other-function"}:
        pci += ".0"
    if source == "invalid-event":
        pci = "dead"
    if source.startswith("legacy"):
        legacy_metric(
            context,
            uuid=None if source == "legacy-no-uuid" else "GPU-a",
            pci={
                "legacy-no-pci": None,
                "legacy-invalid-pci": "invalid",
                "legacy-other-pci": "0000:c1:00.0",
                "legacy-other-function": "0000:b9:00.1",
            }.get(source, "0000:b9:00.0"),
            observed_at=(
                NOW
                - context.legacy_gpu_metrics_inventory_max_age
                - timedelta(seconds=1)
                if source == "legacy-stale"
                else NOW + timedelta(seconds=31)
                if source == "legacy-future"
                else NOW
            ),
        )
        if source == "legacy-ambiguous":
            legacy_metric(context, uuid="GPU-b", pci="0000:b9:00.1")
    elif source != "none":
        devices = (
            (("GPU-a", "0000:b9:00.0"), ("GPU-b", "0000:b9:00.1"))
            if source == "ambiguous"
            else (("GPU-a", "invalid"),)
            if source == "invalid-device"
            else (("GPU-a", "0000:b9:00.1"),)
            if source == "other-function"
            else (("GPU-a", "0000:b9:00.0"),)
        )
        inventory(
            context,
            devices=devices,
            observed_at=(
                NOW - context.gpu_inventory_max_age - timedelta(seconds=1)
                if source == "stale"
                else NOW + timedelta(seconds=31)
                if source == "future"
                else NOW
            ),
            boot="boot-old" if source == "boot-drift" else "boot-a",
        )
    event = NvidiaKernelLogEvent(
        cluster_id="cluster-a",
        node_id="node-a",
        record_id=f"unit-{source}",
        observed_at=NOW,
        source_boot_id="boot-a",
        message=(
            "NVRM: Xid: 79, GPU has fallen"
            if source == "no-pci"
            else f"NVRM: Xid (PCI:{pci}): 79, GPU has fallen"
        ),
        product="H100",
        runtime_profile_version="simulated-v1",
    )
    response = post(context, NVIDIA_KERNEL_PATH, event)
    assert response.status_code == 200, response.text
    normalized = response.json()["normalized"]["xid_events"]
    assert len(normalized) == 1
    assert normalized[0]["gpu_uuid"] == expected
    assert normalized[0]["node_id"] == "node-a"
    assert normalized[0]["source_boot_id"] == "boot-a"
    assert len(context.store.list_raw_evidence("cluster-a")) == 1
    assert context.store.list_agents("cluster-a") == []


@pytest.mark.parametrize("path", [NVIDIA_KERNEL_PATH, FABRIC_MANAGER_PATH])
@pytest.mark.parametrize(
    ("scope", "source", "expected"),
    [
        ("access", "fresh", ["GPU-a"]),
        ("access", "ambiguous", ["GPU-a", "GPU-b"]),
        ("access", "invalid-device", []),
        ("access", "invalid-event", []),
        ("access", "no-pci", []),
        ("access", "stale", []),
        ("access", "future", []),
        ("access", "legacy", ["GPU-a"]),
        ("access", "legacy-stale", []),
        ("access", "legacy-future", []),
        ("access", "legacy-no-pci", []),
        ("access", "legacy-no-uuid", []),
        ("access", "legacy-other-pci", []),
        ("all", "fresh", ["GPU-a", "GPU-b"]),
        ("all", "legacy", ["GPU-a", "GPU-b"]),
        ("all", "legacy-stale", []),
        ("always", "fresh", ["GPU-a", "GPU-b"]),
        ("trunk", "fresh", ["GPU-a", "GPU-b"]),
        ("nonfatal", "fresh", []),
    ],
)
def test_sxid_scope_uses_current_inventory_without_borrowing_unproved_gpu_targets(
    path: str, scope: str, source: str, expected: list[str]
) -> None:
    context = build_context()
    pci = "dead" if source == "invalid-event" else "0000:b9:00"
    if source.startswith("legacy"):
        observed_at = (
            NOW - context.legacy_gpu_metrics_inventory_max_age - timedelta(seconds=1)
            if source == "legacy-stale"
            else NOW + timedelta(seconds=31)
            if source == "legacy-future"
            else NOW
        )
        legacy_metric(
            context,
            observed_at=observed_at,
            uuid=None if source == "legacy-no-uuid" else "GPU-a",
            pci={"legacy-no-pci": None, "legacy-other-pci": "0000:c1:00.0"}.get(
                source, "0000:b9:00.0"
            ),
        )
        if scope == "all":
            legacy_metric(
                context, uuid="GPU-b", pci="0000:c1:00.0", observed_at=observed_at
            )
    else:
        inventory(
            context,
            devices=(
                (("GPU-a", "0000:b9:00.0"), ("GPU-b", "0000:b9:00.1"))
                if source == "ambiguous"
                else (("GPU-a", "invalid"),)
                if source == "invalid-device"
                else (("GPU-a", "0000:b9:00.0"), ("GPU-b", "0000:c1:00.0"))
            ),
            observed_at=(
                NOW - context.gpu_inventory_max_age - timedelta(seconds=1)
                if source == "stale"
                else NOW + timedelta(seconds=31)
                if source == "future"
                else NOW
            ),
        )
    code = 10003 if scope == "all" else 11012 if scope == "nonfatal" else 11001
    classification = (
        "Always Fatal"
        if scope == "always"
        else "Non-Fatal"
        if scope == "nonfatal"
        else "Fatal"
    )
    message = (
        "nvidia-nvswitch0: SXid"
        + ("" if source == "no-pci" else f" (PCI:{pci})")
        + f": {code}, {classification}, Link 3"
    )
    values = {
        "cluster_id": "cluster-a",
        "node_id": "node-a",
        "record_id": f"unit-{scope}-{source}",
        "observed_at": NOW,
        "message": message,
        "product": "A100" if scope == "trunk" else "H100",
        "runtime_profile_version": "simulated-v1",
    }
    event = (
        NvidiaKernelLogEvent(**values, source_boot_id="boot-a")
        if path == NVIDIA_KERNEL_PATH
        else FabricManagerLogEvent(**values, source="journal")
    )
    response = post(context, path, event)
    assert response.status_code == 200, response.text
    (normalized,) = response.json()["normalized"]["sxid_events"]
    assert normalized["participating_gpu_uuids"] == expected
    assert normalized["cluster_id"] == "cluster-a"
    if scope in {"all", "always", "trunk"} and expected:
        assert normalized["fabric_partition"] == "cluster-a/node-a/local-nvswitch"
    else:
        assert normalized["fabric_partition"] is None
    assert len(context.store.list_raw_evidence("cluster-a")) == 1
    assert context.store.list_agents("cluster-a") == []
