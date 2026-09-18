from __future__ import annotations

import gzip
import hashlib
from types import SimpleNamespace

import boto3
import pytest
from kubernetes import client

from gpu_fault.models import WorkflowStepStatus
from gpu_fault.telemetry import EvidenceKind
from tests.execution._cov95_runtime_logs import LogHarness
from tests.execution._cov95_runtime_restart import ApiError


@pytest.mark.parametrize("value", [b"short\n", "short\n", b"x" * 5000, b"one\ntwo\n"])
def test_stop_captures_bounded_log_evidence_before_annotation_and_delete(value) -> None:
    h = LogHarness()
    h.core.value = value
    h.core.mapping_response = True
    result = h.execute()
    assert result.status is WorkflowStepStatus.SUCCEEDED, result
    assert [event[0] for event in h.core.events] == ["read", "patch", "delete"], (
        h.core.events
    )
    [request] = h.sink.requests
    raw = value if isinstance(value, bytes) else value.encode()
    assert (
        request.cluster_id == h.context.incident.cluster_id
        and request.node_id == "node-a"
    ), request
    assert request.attempt_ids == ["attempt-a"], request
    assert request.payload["sha256"] == hashlib.sha256(raw).hexdigest(), request
    assert request.payload["tail"] == raw[-4096:].decode(), request
    assert request.payload["truncated"] is (
        len(raw) > 4096 or len(raw.splitlines()) >= 2
    ), request
    assert h.core.events[0][1][2] == {
        "container": "trainer",
        "timestamps": True,
        "tail_lines": 2,
        "limit_bytes": 8192,
        "_request_timeout": 7,
    }, h.core.events[0]
    assert result.details["workload_log_errors"] == [], result


def test_typed_pod_container_annotation_and_missing_node_are_retained_in_evidence() -> (
    None
):
    h = LogHarness()
    h.core.pods = [
        client.V1Pod(
            metadata=client.V1ObjectMeta(
                name="worker-0",
                annotations={"gpu-fault.io/training-container": "trainer"},
            ),
            spec=client.V1PodSpec(
                containers=[
                    client.V1Container(name="trainer"),
                    client.V1Container(name="sidecar"),
                ]
            ),
        )
    ]
    result = h.execute()
    assert result.status is WorkflowStepStatus.SUCCEEDED, result
    [request] = h.sink.requests
    assert request.node_id == "UNKNOWN", request
    assert (
        request.payload["pod_uid"] == "worker-0"
        and request.payload["container_name"] == "trainer"
    ), request
    assert result.details["workload_log_errors"] == [], result


@pytest.mark.parametrize(
    "containers", [[], [{"name": "trainer"}, {"name": "sidecar"}], [{}]]
)
def test_unknown_training_container_is_reported_without_preventing_containment(
    containers,
) -> None:
    h = LogHarness()
    h.core.pods[0]["spec"]["containers"] = containers
    result = h.execute()
    assert result.status is WorkflowStepStatus.SUCCEEDED, result
    [error] = result.details["workload_log_errors"]
    assert "training container" in error["error"], error
    assert result.details["workload_log_evidence"] == [], result
    assert [event[0] for event in h.core.events] == ["patch", "delete"], h.core.events


@pytest.mark.parametrize("failure", ["read", "sink", "unconfigured"])
def test_log_capture_failure_is_visible_while_the_stop_still_finishes(
    failure: str,
) -> None:
    h = LogHarness(sink="none" if failure == "unconfigured" else "remote")
    if failure == "read":
        h.core.log_error = OSError("unit log transport failed")
    elif failure == "sink":
        h.sink.error = OSError("unit evidence persistence failed")
    result = h.execute()
    assert result.status is WorkflowStepStatus.SUCCEEDED, result
    assert len(result.details["workload_log_errors"]) == 1, result
    assert result.details["workload_log_evidence"] == [], result
    assert h.core.events[-1] == ("delete", ("training", "worker-0")), h.core.events


def test_local_evidence_persistence_uses_the_same_id_and_bytes_as_the_stop_result() -> (
    None
):
    h = LogHarness(sink="local")
    result = h.execute()
    [record] = h.store.list_raw_evidence(
        "cluster-a", node_id="node-a", kind=EvidenceKind.WORKLOAD_LOG
    )
    assert (
        result.details["workload_log_evidence"][0]["record_id"] == record.record_id
    ), result
    assert record.payload["tail"] == "first\nlast\n", record
    assert record.attempt_ids == ["attempt-a"], record


@pytest.mark.parametrize("prefix", ["", "/archive"])
@pytest.mark.parametrize("injected", [False, True])
def test_archive_transport_receives_compressed_bytes_with_digest_and_bounded_read(
    prefix: str, injected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploads = []
    calls = []

    def upload(destination, compressed):
        uploads.append((destination, compressed))
        return destination

    def put_object(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        boto3, "client", lambda name: SimpleNamespace(put_object=put_object)
    )
    h = LogHarness(
        workload_log_s3_uri=f"s3://unit-audit{prefix}",
        workload_log_uploader=upload if injected else None,
    )
    h.core.value = b"x" * 8192
    result = h.execute()
    assert result.status is WorkflowStepStatus.SUCCEEDED, result
    [request] = h.sink.requests
    destination = request.payload["s3_uri"]
    assert destination.startswith(f"s3://unit-audit{prefix}/cluster-a/"), destination
    assert "tail_lines" not in h.core.events[0][1][2], h.core.events[0]
    assert (
        request.payload["archive_truncated"] is True
        and request.payload["tail_bytes"] == 4096
    ), request
    if injected:
        assert len(uploads) == 1 and calls == [], (uploads, calls)
        assert uploads[0][0] == destination, uploads
        compressed = uploads[0][1]
    else:
        [body] = calls
        assert body["Bucket"] == "unit-audit" and body["ContentEncoding"] == "gzip", (
            body
        )
        assert body["Metadata"] == {
            "sha256": hashlib.sha256(h.core.value).hexdigest()
        }, body
        compressed = body["Body"]
    assert gzip.decompress(compressed) == h.core.value, (
        "archive bytes must match captured bytes"
    )


@pytest.mark.parametrize("uri", ["https://unit-audit/archive", "s3:///missing-bucket"])
def test_invalid_archive_destination_records_failure_without_any_upload(
    uri: str,
) -> None:
    uploads = []
    h = LogHarness(
        workload_log_s3_uri=uri,
        workload_log_uploader=lambda *args: uploads.append(args),
    )
    result = h.execute()
    assert result.status is WorkflowStepStatus.SUCCEEDED, result
    assert uploads == [] and h.sink.requests == [], (uploads, h.sink.requests)
    assert (
        "must be an s3:// URI" in result.details["workload_log_errors"][0]["error"]
    ), result


@pytest.mark.parametrize("phase", ["patch", "delete"])
@pytest.mark.parametrize("status", [404, 500])
def test_pod_absence_is_idempotent_but_transport_errors_are_not_silently_ignored(
    phase: str, status: int
) -> None:
    h = LogHarness()
    setattr(h.core, f"{phase}_error", ApiError(status))
    if status == 500:
        with pytest.raises(ApiError, match="fake Kubernetes status 500"):
            h.execute()
    else:
        result = h.execute()
        assert result.status is WorkflowStepStatus.SUCCEEDED, result
        assert result.details["workload_log_errors"] == [], result
    assert len(h.sink.requests) == 1, (
        "evidence is captured before attempting Pod mutation"
    )
    if phase == "patch":
        assert not any(event[0] == "delete" for event in h.core.events), h.core.events


def test_stop_poll_confirms_attempt_absence_without_reading_or_deleting_logs_again() -> (
    None
):
    h = LogHarness()
    h.api.source["status"]["active"] = 1
    first = h.execute()
    assert first.status is WorkflowStepStatus.WAITING, first
    h.follow()
    h.api.source["status"]["active"] = 0
    second = h.execute()
    assert second.status is WorkflowStepStatus.SUCCEEDED, second
    assert second.details["attempt_pods_absent"] is True, second
    assert [event[0] for event in h.core.events] == ["read", "patch", "delete"], (
        h.core.events
    )
