from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.nvidia_logs import NvidiaKernelLogEvent
from gpu_fault.telemetry import EvidenceKind, RawEvidenceRecord
from scripts.e2e.regional.collector_kernel_evidence import (
    kernel_event_identity_errors,
    kmsg_record_errors,
)
from scripts.e2e.regional.probes import collector_node_probe as probe

SOFTWARE = {"product": "H100", "driver_branch": 575, "cuda_version": "12.9"}


def record(sequence: int) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    payload = NvidiaKernelLogEvent(
        cluster_id="cluster-a",
        node_id="node-a",
        record_id=f"kmsg-boot-a-{sequence}",
        observed_at=now,
        source_boot_id="boot-a",
        source_monotonic_us=sequence * 1000,
        message="NVRM: Xid (PCI:0000:01:00): 13, acceptance replay",
        evidence_ref=f"kmsg://node-a/boot-a/{sequence}",
        **SOFTWARE,
    )
    return RawEvidenceRecord(
        record_id=f"nvidia-kernel/{payload.record_id}",
        cluster_id="cluster-a",
        node_id="node-a",
        kind=EvidenceKind.NVIDIA_KERNEL,
        observed_at=now,
        ingested_at=now,
        expires_at=now + timedelta(hours=1),
        payload=payload.model_dump(mode="json"),
    ).model_dump(mode="json")


def state() -> dict[str, Any]:
    evidence = record(12)
    return {
        "evidence": [evidence],
        "events": [
            {
                **evidence["payload"],
                "event_id": "event-a",
                "gpu_uuid": "GPU-target",
                "xid": 13,
            }
        ],
        "decisions": [{"event_id": "event-a", "incident_id": "incident-a"}],
        "incidents": [{"incident_id": "incident-a"}],
    }


def errors(value: dict[str, Any]) -> list[str]:
    return kernel_event_identity_errors(
        value,
        node="node-a",
        cluster_id="cluster-a",
        boot_id="boot-a",
        gpu_uuid="GPU-target",
        software=SOFTWARE,
        xid=13,
    )


def test_kernel_verdict_consumes_real_store_record_shape() -> None:
    first = [record(12)]
    assert (
        kmsg_record_errors(first, first + [record(19)], boot_id="boot-a", node="node-a")
        == []
    )
    assert errors(state()) == []


def test_kmsg_records_for_identical_lines_share_the_boot_prefix() -> None:
    first = [record(100)]
    second = first + [record(107)]
    assert kmsg_record_errors(first, second, boot_id="boot-a") == []
    dedup = kmsg_record_errors(first, first, boot_id="boot-a")
    assert any("sample 2 did not add exactly one record" in item for item in dedup), (
        "an identical second sample adds no record"
    )
    other_boot = kmsg_record_errors(first, second, boot_id="boot-other")
    assert any("is not kmsg-boot-other-<sequence>" in item for item in other_boot), (
        "a record from another boot is named"
    )
    no_ref = [
        {**item, "payload": {**item["payload"], "evidence_ref": ""}} for item in second
    ]
    assert any(
        "distinct evidence_ref" in item
        for item in kmsg_record_errors(first, no_ref, boot_id="boot-a")
    ), "records sharing an evidence_ref are named"


@pytest.mark.parametrize(
    ("part", "key", "value"),
    [
        ("payload", "source_boot_id", "another-boot"),
        ("payload", "node_id", "another-node"),
        ("payload", "cluster_id", "another-cluster"),
        ("payload", "evidence_ref", "fifo://node-a/boot-a/12"),
        ("payload", "evidence_ref", "api-replay://node-a/boot-a/12"),
        ("payload", "evidence_ref", "kmsg://node-a/boot-a/99"),
        ("payload", "evidence_ref", "kmsg://node-a/boot-a/12?other"),
        ("payload", "record_id", "kmsg-boot-a-not-a-sequence"),
        ("evidence", "record_id", "unrelated-record"),
        ("evidence", "cluster_id", "another-cluster"),
        ("event", "gpu_uuid", "GPU-other"),
        ("event", "source_boot_id", "another-boot"),
        ("event", "driver_branch", 574),
        ("event", "cuda_version", None),
        ("event", "product", "A100"),
        ("event", "cluster_id", "another-cluster"),
        ("decision", "event_id", "another-event"),
        ("decision", "incident_id", "another-incident"),
    ],
)
def test_wrong_origin_identity_or_action_binding_is_rejected(
    part: str, key: str, value: Any
) -> None:
    document = state()
    target = {
        "payload": document["evidence"][0]["payload"],
        "evidence": document["evidence"][0],
        "event": document["events"][0],
        "decision": document["decisions"][0],
    }[part]
    target[key] = value
    assert errors(document), (
        f"kernel evidence accepted an invalid {part} binding: {key}={value!r}"
    )


def test_flattened_test_record_cannot_masquerade_as_stored_evidence() -> None:
    document = state()
    document["evidence"] = [document["evidence"][0]["payload"]]
    assert errors(document), (
        "flattened payload without the RawEvidenceRecord envelope passed the verdict"
    )


def test_missing_normalized_event_is_not_collection_success() -> None:
    document = state()
    document["events"] = []
    assert errors(document), (
        "stored raw evidence without its normalized event was counted as collection success"
    )


def test_gpu_identity_probe_uses_deployed_discovery_and_refuses_unknown_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gpu_fault.collectors.gpu import discovery

    monkeypatch.setattr(discovery, "discover_gpu_product", lambda: "H100")
    monkeypatch.setattr(
        discovery, "discover_gpu_software_versions", lambda: (575, "12.9")
    )
    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(probe, "emit", emitted.append)
    probe.gpu_identity(argparse.Namespace())
    assert emitted == [SOFTWARE]
    monkeypatch.setattr(
        discovery, "discover_gpu_software_versions", lambda: (575, None)
    )
    with pytest.raises(probe.ProbeError, match="CUDA version is unknown"):
        probe.gpu_identity(argparse.Namespace())
