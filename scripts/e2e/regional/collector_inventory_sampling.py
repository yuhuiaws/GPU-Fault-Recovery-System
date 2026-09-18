"""Verify isolated Collector output before any live inventory publication."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from gpu_fault.host_health import HostTelemetryBatch
from scripts.e2e.regional.collector_inventory_evidence import (
    debounce_sample_errors,
    evidence_time,
)
from scripts.e2e.regional.probes.collector_node_probe import inventory_receipt_digest
from scripts.e2e.regional.regional_commands import RegionalFixtureError


def isolated_inventory_records(
    receipt: Mapping[str, Any],
    *,
    run_id: str,
    cluster_id: str,
    node_id: str,
    boot_id: str,
    baseline_sha256: str,
    expected_gpu_count: int,
    interval_seconds: int,
    observed_after: datetime,
    tolerance: float,
) -> list[dict[str, Any]]:
    """Accept the completed private sink, not a live-delivery or reboot claim."""
    identity = receipt.get("identity") or {}
    configuration = identity.get("configuration") or {}
    raw = receipt.get("batches")
    if (
        type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != 1
        or receipt.get("kind") != "ISOLATED_GPU_INVENTORY"
        or receipt.get("run_id") != run_id
        or receipt.get("publication_performed") is not False
        or type(receipt.get("sampler_pid")) is not int
        or receipt["sampler_pid"] <= 1
        or receipt["sampler_pid"] == configuration.get("pid")
        or receipt.get("interval_seconds") != interval_seconds
        or receipt.get("expected_gpu_count") != expected_gpu_count
        or identity.get("cluster_id") != cluster_id
        or identity.get("node_id") != node_id
        or identity.get("boot_id") != boot_id
        or not identity.get("runtime_profile_version")
        or not identity.get("node_instance_type")
        or not configuration.get("invocation_id")
        or (configuration.get("file") or {}).get("sha256") != baseline_sha256
        or configuration.get("running_env") != configuration.get("file_env")
        or (configuration.get("file_env") or {}).get("GPU_FAULT_EXPECTED_GPU_COUNT")
        != str(expected_gpu_count - 1)
        or not isinstance(raw, list)
        or len(raw) != 2
        or receipt.get("sha256") != inventory_receipt_digest(dict(receipt))
    ):
        raise RegionalFixtureError("isolated inventory sample identity is unproven")
    try:
        batches = [HostTelemetryBatch.model_validate(batch) for batch in raw]
    except ValueError:
        raise RegionalFixtureError("isolated inventory batch is malformed") from None
    sampled = evidence_time(receipt.get("sampled_at"))
    if sampled is None or sampled < batches[-1].observed_at:
        raise RegionalFixtureError("isolated inventory completion clock is unproven")
    expected_values = {
        "gpu_inventory_expected_count": expected_gpu_count,
        "gpu_inventory_active_count": expected_gpu_count - 1,
        "gpu_inventory_missing_count": 1,
        "gpu_inventory_excess_count": 0,
    }
    records = []
    for index, batch in enumerate(batches):
        wanted = {**expected_values, "gpu_inventory_mismatch": index}
        if (
            batch.cluster_id != cluster_id
            or batch.node_id != node_id
            or batch.runtime_profile_version != identity["runtime_profile_version"]
            or batch.producer != "node"
            or batch.collection_errors
            or batch.affected_workload_ids
            or batch.received_at is not None
            or batch.observed_at < observed_after
            or batch.batch_id
            != f"host-{node_id}-{int(batch.observed_at.timestamp() * 1_000_000)}"
            or len(batch.samples) != len(wanted)
            or {item.name: item.value for item in batch.samples} != wanted
            or any(item.device is not None for item in batch.samples)
            or any(
                item.labels.get("node_instance_type") != identity["node_instance_type"]
                for item in batch.samples
            )
        ):
            raise RegionalFixtureError(
                "isolated inventory batch differs from its scope"
            )
        records.append(
            {
                "record_id": f"host-telemetry/{batch.batch_id}",
                "batch_id": batch.batch_id,
                "observed_at": batch.observed_at.isoformat(),
                "edge_filter_reasons": list(batch.edge_filter_reasons),
                "samples": [item.model_dump(mode="json") for item in batch.samples],
            }
        )
    errors = debounce_sample_errors(
        records,
        interval=interval_seconds,
        required_samples=2,
        tolerance=tolerance,
        expected_count=expected_gpu_count,
    )
    if errors:
        raise RegionalFixtureError(
            "isolated inventory debounce failed: " + "; ".join(errors)
        )
    return records


MISMATCH_SAMPLE = "gpu_inventory_mismatch"


def inventory_delivery_errors(
    captured: list[dict[str, Any]], persisted: list[dict[str, Any]]
) -> list[str]:
    """The control plane must retain the exact two captured source batches.

    ``HOST_INVENTORY_EVIDENCE`` projects each persisted batch to its
    ``gpu_inventory_mismatch`` sample, so that projection of the captured batch
    -- not the whole five-sample batch -- is what a delivered record must equal.
    """
    by_id = {record["batch_id"]: record for record in captured}
    if (
        len(persisted) != len(captured)
        or len({record.get("batch_id") for record in persisted}) != len(captured)
        or {record.get("batch_id") for record in persisted} != set(by_id)
    ):
        return ["not every isolated inventory batch reached the control plane"]
    for record in persisted:
        original = by_id[record["batch_id"]]
        if (
            record.get("record_id") != original["record_id"]
            or evidence_time(record.get("observed_at"))
            != evidence_time(original["observed_at"])
            or record.get("samples")
            != [item for item in original["samples"] if item["name"] == MISMATCH_SAMPLE]
            or record.get("edge_filter_reasons") != original["edge_filter_reasons"]
        ):
            return ["persisted inventory evidence differs from the isolated source"]
    return []
