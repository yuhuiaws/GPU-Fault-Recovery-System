"""Local NVIDIA log normalization retains evidence identity and uncertainty."""

from __future__ import annotations

from datetime import datetime

import pytest

from gpu_fault import nvidia_logs
from gpu_fault.policy import SxidClassification
from tests.collectors import _cov95_runtime_collect as support

isolated_runtime = support.isolated_runtime


def kernel_event(message):
    return nvidia_logs.NvidiaKernelLogEvent(
        cluster_id="cluster-a",
        node_id="node-a",
        record_id="private-record",
        observed_at=support.NOW,
        message=message,
    )


def test_xid74_registers_belong_to_the_matching_code_not_a_preceding_event():
    result = nvidia_logs.NvidiaLogNormalizer().normalize_kernel(
        kernel_event("Xid 13 0xdead\nXid 74 0x1 0x2 NVLink ID: 5")
    )
    by_code = {event.xid: event for event in result.xid_events}
    assert by_code[13].registers == []
    assert by_code[74].registers == [1, 2]
    assert by_code[74].nvlink_link_id == 5


@pytest.mark.parametrize(
    "message",
    ["Xid unreadable", "SXid unreadable", "SXid 999999 without classification"],
)
def test_unparsed_or_unclassified_fault_is_named_without_inventing_a_verdict(message):
    result = nvidia_logs.NvidiaLogNormalizer().normalize_kernel(kernel_event(message))
    reason = result.provider_signals[0].unresolved_reasons[0]
    expected = (
        nvidia_logs.UNPARSED_XID_REASON
        if message.startswith("Xid")
        else nvidia_logs.UNPARSED_SXID_REASON
        if "unreadable" in message
        else nvidia_logs.UNCLASSIFIED_SXID_REASON
    )
    assert nvidia_logs.unresolved_reason_kind(reason) == expected
    assert result.xid_events == []
    assert result.sxid_events == []


def test_explicit_always_fatal_sxid_is_preserved():
    result = nvidia_logs.NvidiaLogNormalizer().normalize_kernel(
        kernel_event("SXid 999999, always fatal")
    )
    assert result.sxid_events[0].classification is SxidClassification.ALWAYS_FATAL


def test_invalid_text_timestamp_does_not_prevent_fabric_record_normalization(
    monkeypatch,
):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return support.NOW

    monkeypatch.setattr(nvidia_logs, "datetime", Clock)
    event = nvidia_logs.FabricManagerLogEvent(
        cluster_id="cluster-a",
        node_id="node-a",
        record_id="private-record",
        observed_at=support.NOW,
        source="file",
        message="2026-99-12T12:00:00Z Xid 74 0x1",
    )
    result = nvidia_logs.NvidiaLogNormalizer().normalize_fabric_manager(event)
    assert result.xid_events[0].source_event_time == support.NOW
    assert result.xid_events[0].registers == [1]
