from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any

import pytest

from gpu_fault.collectors import CollectorContext, KernelLogCollector
from gpu_fault.policy import CatalogRule, catalog_supports_product, load_xid_policy
from tests._builders import asgi_client, build_context

NOW = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
PCI_BDF = "0000:b9:00"
POLICY = load_xid_policy()
TRACE_ENV = "GPU_FAULT_EMIT_PROCESSING_TRACE"
TRACE_PREFIX = "GPU_FAULT_PROCESSING_TRACE="


class RecordingSink:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.requests.append((path, payload))
        return {"accepted": True}


def _case_id(rule: CatalogRule) -> str:
    return f"GF-XID-KMSG-{rule.xid:03d}"


def _b200_case_id(rule: CatalogRule) -> str:
    return f"GF-XID-KMSG-B200-{rule.xid:03d}"


def _expected_official_action(rule: CatalogRule) -> str | None:
    # These Catalog workflows resolve a solo event to a specific action.
    if rule.xid == 45:
        return "RESTART_FM"
    if rule.xid == 48:
        return "RESET_GPU"
    return rule.immediate_action


def _nvlink5_record_payload(rule: CatalogRule) -> tuple[str, str]:
    decode = next(
        item
        for item in POLICY.nvlink5.decode_rules
        if item.xid == rule.xid and item.recovery_action in {"IGNORE", "RESET_GPU"}
    )
    intr_info = int(decode.v2_pattern.replace("-", "0"), 2)
    error_status = (
        int(decode.error_status.split("/", 1)[0], 16)
        if decode.error_status is not None
        else 0
    )
    registers = [intr_info, error_status, 0, 0, 0, 0, 0]
    payload = " ".join(f"0x{value:08x}" for value in registers)
    return (f", NVLink5 nonfatal 0 0 Link 0 ({payload})", decode.recovery_action)


@pytest.mark.parametrize(
    "rule", POLICY.catalog_rules, ids=[_case_id(rule) for rule in POLICY.catalog_rules]
)
def test_xid_kmsg_catalog_replay(rule: CatalogRule) -> None:
    _run_xid_kmsg_catalog_replay(
        rule,
        case_id=_case_id(rule),
        product=rule.products[0] if rule.products else "H100",
    )


@pytest.mark.parametrize(
    "rule",
    POLICY.catalog_rules,
    ids=[_b200_case_id(rule) for rule in POLICY.catalog_rules],
)
def test_xid_kmsg_catalog_replay_b200(rule: CatalogRule) -> None:
    _run_xid_kmsg_catalog_replay(rule, case_id=_b200_case_id(rule), product="B200")


def _run_xid_kmsg_catalog_replay(
    rule: CatalogRule, *, case_id: str, product: str
) -> None:
    boot_id = f"boot-{case_id.lower()}"
    sequence = 10_000 + rule.xid
    monotonic_us = 1_100_000 + rule.xid
    sink = RecordingSink()
    collector = KernelLogCollector(
        sink,
        CollectorContext(
            cluster_id="hp-cluster",
            runtime_profile_version="xid-kmsg-replay-v1",
            product=product,
            driver_branch=999,
            cuda_version="99.9",
        ),
        node_id="worker-1",
        boot_id=boot_id,
        now=lambda: NOW,
    )
    nvlink5_payload = ""
    nvlink5_action = None
    if 144 <= rule.xid <= 150:
        nvlink5_payload, nvlink5_action = _nvlink5_record_payload(rule)
    record = (
        f"3,{sequence},{monotonic_us},-;"
        f"NVRM: Xid (PCI:{PCI_BDF}): {rule.xid}, "
        f"pid=1234, name=python{nvlink5_payload}"
    )

    stats = collector.collect_lines([record])

    assert stats.observed == 1
    assert stats.delivered == 1
    assert stats.skipped == 0
    assert stats.duplicates == 0
    assert len(sink.requests) == 1
    path, payload = sink.requests[0]
    assert path == "/v1/collector-events/nvidia-kernel"
    assert payload["record_id"] == f"kmsg-{boot_id}-{sequence}"
    assert payload["source_boot_id"] == boot_id
    assert payload["source_monotonic_us"] == monotonic_us
    assert payload["evidence_ref"] == (f"kmsg://worker-1/{boot_id}/{sequence}")
    assert payload["message"] == record.split(";", 1)[1]

    async def post_to_control_plane() -> dict[str, Any]:
        async with asgi_client(build_context()) as client:
            response = await client.post(path, json=payload)
        assert response.status_code == 200, response.text
        return response.json()

    body = asyncio.run(post_to_control_plane())
    normalized = body["normalized"]
    assert len(normalized["xid_events"]) == 1
    assert len(body["decisions"]) == 1

    event = normalized["xid_events"][0]
    assert event["xid"] == rule.xid
    assert event["product"] == product
    assert event["pci_bdf"] == PCI_BDF
    assert event["source_boot_id"] == boot_id
    assert event["source_monotonic_us"] == monotonic_us
    assert event["event_source"] == "KERNEL_LOG"
    assert event["raw_message"] == payload["message"]
    assert event["evidence_ref"] == payload["evidence_ref"]

    decision = body["decisions"][0]
    assert decision["event_id"] == event["event_id"]
    assert decision["event_type"] == "XID"
    assert decision["policy_version"] == POLICY.mapping_version
    assert decision["source"] == "NVIDIA_CATALOG"
    expected_official_action = (
        "WORKFLOW_XID_45"
        if rule.xid == 45
        else nvlink5_action
        if nvlink5_action is not None
        else _expected_official_action(rule)
    )
    assert decision["official_action"] == expected_official_action
    assert decision["marker"]["mapping_version"] == POLICY.mapping_version
    assert decision["marker"]["policy_source"] == "NVIDIA_CATALOG"
    assert decision["marker"]["raw_evidence_ref"] == payload["evidence_ref"]
    if rule.xid == 45:
        assert decision["incident_id"] is None
        assert decision["investigatory_notification_id"] is None
    else:
        assert decision["incident_id"]
        assert decision["investigatory_notification_id"]

    applicable = catalog_supports_product(product, rule.products)
    if not applicable:
        assert decision["disposition"] == "NOT_APPLICABLE"
        assert decision["action"] is None
        assert decision["safety_action"] == "QUARANTINE"
        assert decision["requires_operator"] is True
        assert decision["advisory_notification_id"]
    elif rule.xid == 45:
        assert decision["disposition"] == "PENDING_CORRELATION"
        assert decision["requires_operator"] is False
    elif rule.xid == 48:
        assert decision["disposition"] == "EXECUTABLE"
        assert decision["action"] == "RESET_GPU"
    elif 144 <= rule.xid <= 150:
        assert event["intr_info"] is not None
        assert event["error_status"] is not None
        assert decision["disposition"] in {"EXECUTABLE", "MONITOR_ONLY"}
        assert decision["action"] == (
            "RESET_GPU" if nvlink5_action == "RESET_GPU" else "NO_ACTION"
        )
    elif rule.xid in {154, 159}:
        assert decision["disposition"] == "BLOCKED_MISSING_EVIDENCE"
        assert decision["action"] is None
        assert decision["requires_operator"] is True

    if os.getenv(TRACE_ENV) == "1":
        trace = {
            "schema_version": 1,
            "case_id": case_id,
            "xid": rule.xid,
            "stages": [
                {
                    "order": 1,
                    "component": "/dev/kmsg replay input",
                    "operation": "replay NVIDIA kernel record",
                    "input": record,
                    "parsed_header": {
                        "priority": 3,
                        "sequence": sequence,
                        "monotonic_us": monotonic_us,
                        "flags": "-",
                    },
                },
                {
                    "order": 2,
                    "component": "KernelLogCollector",
                    "operation": (
                        "parse, NVIDIA-filter, deduplicate and build collector event"
                    ),
                    "result": stats.model_dump(mode="json"),
                    "endpoint": path,
                    "payload": payload,
                },
                {
                    "order": 3,
                    "component": "NVIDIA kernel ingestion API",
                    "operation": f"POST {path}",
                    "http_status": 200,
                    "accepted": True,
                },
                {
                    "order": 4,
                    "component": "HyperPodHmaNormalizer",
                    "operation": "extract and normalize XID evidence",
                    "provider_signals": normalized["provider_signals"],
                    "xid_event": event,
                },
                {
                    "order": 5,
                    "component": "pinned NVIDIA XID Catalog",
                    "operation": "match XID and compatibility metadata",
                    "catalog_version": POLICY.catalog_version,
                    "policy_version": POLICY.mapping_version,
                    "source_url": POLICY.source_url,
                    "source_sha256": POLICY.source_sha256,
                    "matched_rule": rule.model_dump(mode="json", by_alias=True),
                },
                {
                    "order": 6,
                    "component": "GpuFaultPolicyEngine",
                    "operation": (
                        "resolve official action, evidence gates and site safety action"
                    ),
                    "decision": decision,
                },
                {
                    "order": 7,
                    "component": "IncidentOrchestrator",
                    "operation": ("create/correlate incident and recovery workflow"),
                    "incident_id": decision["incident_id"],
                    "workflow_request_id": decision["workflow_request_id"],
                    "advisory_notification_id": decision["advisory_notification_id"],
                },
            ],
        }
        print(
            TRACE_PREFIX + json.dumps(trace, ensure_ascii=False, separators=(",", ":"))
        )
