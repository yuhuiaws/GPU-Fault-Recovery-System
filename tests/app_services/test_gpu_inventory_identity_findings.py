"""A GPU that disappears from the inventory is a fault, not only evidence.

``_ingest_gpu_inventory_batch`` used to record a count mismatch or an identity
change as a raw evidence row and nothing else; the finding path
(``gpu_inventory_mismatch``) lived on the host collector and needed
``--expected-gpu-count``, which the installer left unset. The ingest side now
fails closed: a changed UUID set is a CRITICAL node-health finding, and a
snapshot with no expected count at all is a WARNING finding rather than
silence.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.app.ingest.telemetry_context import (
    GPU_EXPECTED_COUNT_UNKNOWN_METRIC,
    GPU_IDENTITY_CHANGED_METRIC,
)
from gpu_fault.gpu_metrics import (
    GpuInventoryDevice,
    GpuInventorySnapshot,
    GpuMetricSource,
)
from gpu_fault.models import RecoveryAction, Severity
from tests._builders import asgi_client, build_context

NOW = datetime(2026, 9, 7, 10, 0, tzinfo=timezone.utc)
PATH = "/v1/collector-events/gpu-inventory"


def _snapshot(
    snapshot_id: str,
    uuids: list[str],
    *,
    observed_at: datetime,
    expected: int | None,
    bdfs: list[str] | None = None,
    profile: str | None = None,
) -> dict:
    return GpuInventorySnapshot(
        snapshot_id=snapshot_id,
        cluster_id="hp-cluster",
        node_id="worker-1",
        observed_at=observed_at,
        source=GpuMetricSource.NVIDIA_SMI,
        source_boot_id="boot-a",
        devices=[
            GpuInventoryDevice(
                gpu_index=index,
                gpu_uuid=uuid,
                pci_bdf=(
                    bdfs[index] if bdfs is not None else f"0000:{0x59 + index:02x}:00.0"
                ),
                product="H100",
            )
            for index, uuid in enumerate(uuids)
        ],
        expected_gpu_count=expected,
        runtime_profile_version=profile,
    ).model_dump(mode="json")


def _post_all(context, payloads: list[dict]) -> list[int]:
    async def scenario() -> list[int]:
        codes = []
        async with asgi_client(context) as client:
            for payload in payloads:
                response = await client.post(PATH, json=payload)
                codes.append(response.status_code)
        return codes

    return asyncio.run(scenario())


def _incident(context, snapshot_id: str, metric: str):
    return context.store.get_incident_by_event(f"{snapshot_id}-{metric}")


def test_vanished_gpu_uuid_opens_a_critical_finding() -> None:
    context = build_context()
    codes = _post_all(
        context,
        [
            _snapshot("inv-1", ["GPU-a", "GPU-b"], observed_at=NOW, expected=2),
            _snapshot(
                "inv-2", ["GPU-a"], observed_at=NOW + timedelta(minutes=1), expected=2
            ),
        ],
    )
    assert codes == [200, 200], codes

    assert _incident(context, "inv-1", GPU_IDENTITY_CHANGED_METRIC) is None, (
        "the baseline snapshot was reported as a change"
    )
    incident = _incident(context, "inv-2", GPU_IDENTITY_CHANGED_METRIC)
    assert incident is not None, "a vanished GPU did not open a node-health incident"
    assert incident.event_type == "NODE_HEALTH"
    assert incident.node_ids == ["worker-1"]
    assert incident.effective_action is RecoveryAction.RUN_DIAGNOSTICS
    assert "GPU-b" in " ".join(incident.reasons), incident.reasons
    markers = context.store.list_markers_for_incident(incident.incident_id)
    assert {marker.severity for marker in markers} == {Severity.CRITICAL}, markers
    evidence = context.store.list_raw_evidence("hp-cluster")
    assert any(item.record_id == "gpu-inventory/inv-2" for item in evidence), (
        "the identity change is no longer kept as evidence"
    )


def test_stable_inventory_with_known_count_stays_silent() -> None:
    context = build_context()
    codes = _post_all(
        context,
        [
            _snapshot("inv-1", ["GPU-a", "GPU-b"], observed_at=NOW, expected=2),
            _snapshot(
                "inv-2",
                ["GPU-a", "GPU-b"],
                observed_at=NOW + timedelta(minutes=1),
                expected=2,
            ),
        ],
    )
    assert codes == [200, 200], codes
    for snapshot_id in ("inv-1", "inv-2"):
        for metric in (GPU_IDENTITY_CHANGED_METRIC, GPU_EXPECTED_COUNT_UNKNOWN_METRIC):
            assert _incident(context, snapshot_id, metric) is None, (
                snapshot_id,
                metric,
            )


def test_unknown_expected_count_is_a_warning_finding_once_per_node() -> None:
    context = build_context()
    codes = _post_all(
        context,
        [
            _snapshot("inv-1", ["GPU-a", "GPU-b"], observed_at=NOW, expected=None),
            _snapshot(
                "inv-2",
                ["GPU-a", "GPU-b"],
                observed_at=NOW + timedelta(minutes=1),
                expected=None,
            ),
        ],
    )
    assert codes == [200, 200], codes

    incident = _incident(context, "inv-1", GPU_EXPECTED_COUNT_UNKNOWN_METRIC)
    assert incident is not None, "an unknown expected GPU count stayed silent"
    assert incident.effective_action is RecoveryAction.COLLECT_EVIDENCE
    assert "expected" in " ".join(incident.reasons).lower(), incident.reasons
    markers = context.store.list_markers_for_incident(incident.incident_id)
    assert {marker.severity for marker in markers} == {Severity.WARNING}, markers
    assert _incident(context, "inv-2", GPU_EXPECTED_COUNT_UNKNOWN_METRIC) is None, (
        "the same node was told twice about the same unknown invariant"
    )


def test_unknown_count_episode_clears_once_the_count_arrives() -> None:
    context = build_context()
    codes = _post_all(
        context,
        [
            _snapshot("inv-1", ["GPU-a"], observed_at=NOW, expected=None),
            _snapshot(
                "inv-2", ["GPU-a"], observed_at=NOW + timedelta(minutes=1), expected=1
            ),
            _snapshot(
                "inv-3",
                ["GPU-a"],
                observed_at=NOW + timedelta(minutes=2),
                expected=None,
            ),
        ],
    )
    assert codes == [200, 200, 200], codes
    assert _incident(context, "inv-1", GPU_EXPECTED_COUNT_UNKNOWN_METRIC) is not None
    assert _incident(context, "inv-2", GPU_EXPECTED_COUNT_UNKNOWN_METRIC) is None
    assert _incident(context, "inv-3", GPU_EXPECTED_COUNT_UNKNOWN_METRIC) is not None, (
        "a count that went missing again after being known stayed silent"
    )


#: Operations a GPU identity diagnosis must never plan: the finding is
#: read-only (GF-GENERAL-LIVE-026 ``no_unapproved_reset``).
_HARDWARE_OPERATIONS = frozenset(
    {
        "RESET_GPU",
        "RESET_ALL_GPUS_NVSWITCHES",
        "RESTART_NODE",
        "REPLACE_NODE",
        "MARK_UNSCHEDULABLE",
        "QUARANTINE",
    }
)


def _operations(context, incident) -> list[str]:
    assert incident.workflow_request_id, (
        f"identity incident {incident.incident_id} planned no workflow"
    )
    workflow = context.store.get_workflow(incident.workflow_request_id)
    return [step.operation.value for step in workflow.official_steps]


def test_moved_pci_bus_ids_with_the_same_uuids_open_a_critical_finding() -> None:
    """Same UUIDs, same count, two GPUs answer on each other's bus id.

    GF-GENERAL-LIVE-026 ``bdf_changed``: the check compared UUID sets only, so
    a BDF move produced an evidence row and nothing else. The inventory's
    BDFs place XID/SXID lines on GPUs, so the move is an identity change.
    """

    context = build_context()
    codes = _post_all(
        context,
        [
            _snapshot(
                "inv-1",
                ["GPU-a", "GPU-b", "GPU-c"],
                observed_at=NOW,
                expected=3,
                bdfs=["0000:59:00.0", "0000:5a:00.0", "0000:5b:00.0"],
                profile="simulated-v1",
            ),
            _snapshot(
                "inv-2",
                ["GPU-a", "GPU-b", "GPU-c"],
                observed_at=NOW + timedelta(minutes=1),
                expected=3,
                bdfs=["0000:5b:00.0", "0000:5a:00.0", "0000:59:00.0"],
                profile="simulated-v1",
            ),
        ],
    )
    assert codes == [200, 200], codes

    incident = _incident(context, "inv-2", GPU_IDENTITY_CHANGED_METRIC)
    assert incident is not None, "a BDF move with unchanged UUIDs was not diagnosed"
    assert incident.event_type == "NODE_HEALTH"
    assert incident.effective_action is RecoveryAction.RUN_DIAGNOSTICS
    reasons = " ".join(incident.reasons)
    assert "0000:59:00.0" in reasons and "0000:5b:00.0" in reasons, reasons
    assert "GPU-a" in reasons and "GPU-c" in reasons, reasons
    assert "GPU-b" not in reasons, f"the GPU that kept its bus id was named: {reasons}"
    assert set(incident.gpu_uuids) == {"GPU-a", "GPU-c"}, incident.gpu_uuids
    markers = context.store.list_markers_for_incident(incident.incident_id)
    assert {marker.severity for marker in markers} == {Severity.CRITICAL}, markers
    operations = _operations(context, incident)
    assert operations == ["FREEZE_EVIDENCE", "VALIDATE_GPU"], operations
    assert not _HARDWARE_OPERATIONS.intersection(operations), operations
    reviews = [
        item
        for item in context.store.list_notifications()
        if item.incident_id == incident.incident_id
        and item.category == "OPERATOR_REVIEW"
    ]
    assert len(reviews) == 1, reviews
    evidence = context.store.list_raw_evidence("hp-cluster")
    assert any(item.record_id == "gpu-inventory/inv-2" for item in evidence), (
        "the BDF move is no longer kept as evidence"
    )


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (["00000000:59:00.0", "0000:5a:00.0"], ["0000:59:00.0", "00000000:5A:00.0"]),
        (["[N/A]", "0000:5a:00.0"], ["0000:59:00.0", "0000:5a:00.0"]),
    ],
    ids=["domain_width_and_hex_case", "unreadable_bus_id"],
)
def test_a_respelt_or_unreadable_bus_id_is_not_a_move(
    before: list[str], after: list[str]
) -> None:
    """A 4- vs 8-digit domain or the hex case is the same bus address, and a
    bus id that does not parse is unknown -- not evidence the GPU moved."""

    context = build_context()
    codes = _post_all(
        context,
        [
            _snapshot(
                "inv-1", ["GPU-a", "GPU-b"], observed_at=NOW, expected=2, bdfs=before
            ),
            _snapshot(
                "inv-2",
                ["GPU-a", "GPU-b"],
                observed_at=NOW + timedelta(minutes=1),
                expected=2,
                bdfs=after,
            ),
        ],
    )
    assert codes == [200, 200], codes
    assert _incident(context, "inv-2", GPU_IDENTITY_CHANGED_METRIC) is None, (
        f"a bus id read as {before} then {after} was reported as a moved GPU"
    )


def test_a_new_pci_mapping_is_reported_once_then_is_the_baseline() -> None:
    """GF-GENERAL-LIVE-026 ``approved_rebaseline`` for a BDF move.

    The first snapshot with the new mapping diagnoses; the matching ones
    after it clear the signal instead of re-reporting the old identity, and
    the stored baseline carries the new bus ids.
    """

    context = build_context()
    moved = ["0000:5a:00.0", "0000:59:00.0"]
    codes = _post_all(
        context,
        [
            _snapshot(
                "inv-1",
                ["GPU-a", "GPU-b"],
                observed_at=NOW,
                expected=2,
                bdfs=["0000:59:00.0", "0000:5a:00.0"],
            ),
            _snapshot(
                "inv-2",
                ["GPU-a", "GPU-b"],
                observed_at=NOW + timedelta(minutes=1),
                expected=2,
                bdfs=moved,
            ),
            _snapshot(
                "inv-3",
                ["GPU-a", "GPU-b"],
                observed_at=NOW + timedelta(minutes=2),
                expected=2,
                bdfs=moved,
            ),
        ],
    )
    assert codes == [200, 200, 200], codes
    assert _incident(context, "inv-2", GPU_IDENTITY_CHANGED_METRIC) is not None
    assert _incident(context, "inv-3", GPU_IDENTITY_CHANGED_METRIC) is None, (
        "the stable new mapping was re-reported instead of becoming the baseline"
    )
    baseline = context.store.get_gpu_inventory_snapshot("hp-cluster", "worker-1")
    assert baseline is not None and baseline.snapshot_id == "inv-3", baseline
    assert {device.gpu_uuid: device.pci_bdf for device in baseline.devices} == {
        "GPU-a": "0000:5a:00.0",
        "GPU-b": "0000:59:00.0",
    }, baseline.devices
