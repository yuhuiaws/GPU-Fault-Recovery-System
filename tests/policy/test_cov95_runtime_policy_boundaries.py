from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from importlib.resources import files
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from gpu_fault.policy import DistributedXidBatch
from gpu_fault.policy.catalog_integrity import (
    validate_xid_catalog_document,
    xid_catalog_artifact_sha256,
)
from gpu_fault.policy.models import XidEvent, parse_xid154_action


@pytest.fixture(scope="module")
def pinned_catalog() -> dict[str, Any]:
    return yaml.safe_load(
        files("gpu_fault.data")
        .joinpath("nvidia-xid-catalog-610.generated.yaml")
        .read_text(encoding="utf-8")
    )


@pytest.mark.parametrize(
    ("path", "value", "reason"),
    [
        (("apiVersion",), "unreviewed/v2", "apiVersion"),
        (("kind",), "OtherPolicy", "kind"),
        (("metadata",), None, "metadata/spec"),
        (("spec",), [], "metadata/spec"),
        (("spec", "catalog"), None, "sections"),
        (("spec", "nvlink5"), [], "sections"),
        (("spec", "catalog", "version"), "611", "not pinned"),
        (("metadata", "sourceSha256"), "0" * 64, "not pinned"),
        (("spec", "catalogRules"), {}, "172 rules"),
        (("spec", "catalogRules"), [], "172 rules"),
        (("spec", "nvlink5", "decodeRules"), None, "95 NVLink5"),
        (("spec", "nvlink5", "decodeRules"), [], "95 NVLink5"),
        (("spec", "resolutionBuckets"), [], "32 resolution"),
        (("spec", "resolutionBuckets"), {}, "32 resolution"),
        (("metadata", "generatedSha256"), None, "missing or invalid"),
        (("metadata", "generatedSha256"), "short", "missing or invalid"),
        (("metadata", "generatedSha256"), "f" * 64, "artifact digest mismatch"),
    ],
)
def test_catalog_admission_refuses_unpinned_or_incomplete_documents(
    pinned_catalog: dict[str, Any], path: tuple[str, ...], value: Any, reason: str
) -> None:
    document = deepcopy(pinned_catalog)
    section = document
    for key in path[:-1]:
        section = section[key]
    section[path[-1]] = value
    with pytest.raises(ValueError, match=reason):
        validate_xid_catalog_document(document)
    assert (
        validate_xid_catalog_document(pinned_catalog)
        == pinned_catalog["metadata"]["generatedSha256"]
    ), "validation must neither mutate nor invalidate the pinned document"


def test_catalog_hash_is_canonical_and_never_removes_input_metadata(
    pinned_catalog: dict[str, Any],
) -> None:
    document = deepcopy(pinned_catalog)
    original = deepcopy(document)
    reordered = dict(reversed(list(document.items())))
    assert xid_catalog_artifact_sha256(reordered) == xid_catalog_artifact_sha256(
        document
    ), "mapping insertion order must not change the signed content identity"
    assert document == original, (
        "digest calculation must not mutate caller-owned metadata"
    )
    with pytest.raises(ValueError, match="metadata must be a mapping"):
        xid_catalog_artifact_sha256({"metadata": []})
    document["metadata"]["nonfinite"] = float("nan")
    with pytest.raises(ValueError, match="Out of range float"):
        xid_catalog_artifact_sha256(document)


@pytest.fixture
def distributed_payload() -> dict[str, Any]:
    now = datetime(2026, 9, 12, 8, tzinfo=timezone(timedelta(hours=8)))
    return {
        "job_id": "job-a",
        "attempt_id": "attempt-a",
        "restart_budget": 1,
        "affected_workload_ids": ["training/job/job-a"],
        "allocation": [
            {"node_id": "node-a", "gpu_uuids": ["GPU-a"]},
            {"node_id": "node-b", "gpu_uuids": ["GPU-b"]},
        ],
        "events": [
            XidEvent(
                event_id=f"event-{node}",
                cluster_id="cluster-a",
                node_id=f"node-{node}",
                gpu_uuid=f"GPU-{node}",
                xid=94,
                observed_at=now,
                workload_state="ACTIVE",
                job_id="job-a",
                attempt_id="attempt-a",
                runtime_profile_version="active-v1",
            ).model_dump()
            for node in ("a", "b")
        ],
    }


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("event_id", "event-a", "event IDs must be unique"),
        ("cluster_id", "cluster-b", "share one cluster"),
        ("runtime_profile_version", "other-v1", "share one runtime profile"),
        ("runtime_profile_version", None, "share one runtime profile"),
        ("job_id", "other-job", "another job"),
        ("workload_state", "UNKNOWN", "requires an ACTIVE"),
        ("workload_state", "IDLE", "requires an ACTIVE"),
        ("node_id", "node-c", "outside allocation"),
        ("gpu_uuid", "GPU-a", "not in allocation"),
        ("gpu_uuid", None, "not in allocation"),
        ("attempt_id", "other-attempt", "another attempt"),
    ],
)
def test_distributed_admission_refuses_mixed_identity_and_unknown_occupancy(
    distributed_payload: dict[str, Any], field: str, value: Any, reason: str
) -> None:
    distributed_payload["events"][1][field] = value
    with pytest.raises(ValidationError, match=reason):
        DistributedXidBatch.model_validate(distributed_payload)


def test_distributed_scope_accepts_batch_owned_identity_and_normalizes_observed_time(
    distributed_payload: dict[str, Any],
) -> None:
    distributed_payload["events"][0].update(job_id=None, attempt_id=None)
    batch = DistributedXidBatch.model_validate(distributed_payload)
    assert batch.job_id == "job-a" and batch.attempt_id == "attempt-a", batch
    assert all(
        event.observed_at.utcoffset() == timedelta(0) for event in batch.events
    ), batch
    assert batch.events[0].observed_at.hour == 0, batch.events[0]


@pytest.mark.parametrize(
    "message",
    [
        None,
        "",
        "Xid 94: GPU recovery action changed from (None) to (GPU reset required)",
    ],
)
def test_non_xid154_messages_cannot_supply_dynamic_recovery_action(
    message: str | None,
) -> None:
    assert parse_xid154_action(message) is None, (
        "only an explicit Xid 154 label may override the catalog action",
        message,
    )
