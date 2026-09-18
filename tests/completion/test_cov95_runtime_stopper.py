from __future__ import annotations

import gzip
import hashlib
import json
import sys
from datetime import timedelta
from types import SimpleNamespace

import pytest
from kubernetes.client import V1Container, V1ObjectMeta, V1Pod, V1PodSpec

from gpu_fault.completion_observation import TERMINATION_INCIDENT_ANNOTATION
from gpu_fault.completion_pod_parsing import CompletionControllerError
from gpu_fault.completion_workload_stopper import KubernetesWorkloadStopper
from tests.completion._support import Clock, FakeCoreApi, FakeSink, pod
from tests.completion.test_completion_controller_restore import restored_watcher


class WorkloadApi:
    def __init__(self):
        self.patches = []
        self.annotations = {}

    def patch_namespaced_job(self, name, namespace, body):
        self.patches.append((name, namespace, body))
        self.annotations.update(body["metadata"]["annotations"])

    def patch_namespaced_custom_object(self, *args):
        self.patches.append(args)
        self.annotations.update(args[-1]["metadata"]["annotations"])

    def read_namespaced_job(self, name, namespace, **kwargs):
        return {
            "metadata": {"name": name, "annotations": dict(self.annotations)},
            "status": {"failed": 1},
        }

    def get_namespaced_custom_object(
        self, group, version, namespace, plural, name, **kw
    ):
        return self.read_namespaced_job(name, namespace, **kw)


@pytest.mark.parametrize("kind", ["job", "pytorchjob", "jobset"])
def test_emergency_stop_incident_survives_pod_loss_and_watcher_restart(kind) -> None:
    clock, sink, workload = Clock(), FakeSink(), WorkloadApi()
    workload_id = f"default/{kind}/trainer"
    core = FakeCoreApi([pod(0, workload_ids=[workload_id])])
    controller, replay_stopper = restored_watcher(core, sink, clock, workload)
    replay_stopper.custom = workload
    stopper = KubernetesWorkloadStopper(workload, workload, core)
    assert (
        stopper.stop((workload_id,), "train-a1", "incident-stop", capture_logs=False)
        == []
    )
    controller.run_once()
    clock.value += timedelta(seconds=31)
    controller.run_once()
    terminals = [payload for path, payload in sink.posts if path.endswith("/terminal")]
    assert [
        (item["terminal_status"], item["termination_initiator_incident_id"])
        for item in terminals
    ] == ([("STOPPED", "incident-stop")])
    assert len(workload.patches) == 1
    assert workload.annotations[TERMINATION_INCIDENT_ANNOTATION] == "incident-stop"
    assert replay_stopper.calls == []
    assert [path for path, _ in sink.posts if path.endswith("failure-detected")] == []


def test_skip_log_capture_still_marks_pods_before_suspending_owner() -> None:
    core = FakeCoreApi([pod(0)])
    workload = WorkloadApi()
    stopper = KubernetesWorkloadStopper(workload, workload, core)
    assert (
        stopper.stop(("default/trainer",), "train-a1", "incident", capture_logs=False)
        == []
    )
    assert core.pod_patches == [
        (
            "default",
            "worker-0",
            {
                "metadata": {
                    "annotations": {
                        TERMINATION_INCIDENT_ANNOTATION: "incident",
                        "gpu-fault.io/passive-stop-attempt": "train-a1",
                        "gpu-fault.io/passive-stop-mode": "emergency-fallback",
                    }
                }
            },
        )
    ]
    assert workload.patches[0][-1]["spec"] == {"suspend": True}


def test_selected_pod_without_name_refuses_workload_suspension() -> None:
    current = pod(0)
    current["metadata"].pop("name")
    core, workload = FakeCoreApi([current]), WorkloadApi()
    stopper = KubernetesWorkloadStopper(workload, workload, core)
    with pytest.raises(CompletionControllerError, match="no metadata.name"):
        stopper.stop(("default/trainer",), "train-a1", "incident", capture_logs=False)
    assert workload.patches == []


@pytest.mark.parametrize(
    "workload_id", ["invalid", "a/b/c/d", "/job/x", "a/job/", "a/pod/x"]
)
def test_invalid_workload_id_never_reaches_a_mutating_api(workload_id) -> None:
    workload = WorkloadApi()
    with pytest.raises(CompletionControllerError, match="workload ID"):
        KubernetesWorkloadStopper(workload, workload).stop((workload_id,), "attempt")
    assert workload.patches == []


def test_log_capture_requires_a_positive_budget_and_handles_no_pods() -> None:
    with pytest.raises(CompletionControllerError, match="must be positive"):
        KubernetesWorkloadStopper(None, None, workload_log_timeout_seconds=0)
    assert (
        KubernetesWorkloadStopper(None, None).capture_logs("attempt", "incident") == []
    )
    assert (
        KubernetesWorkloadStopper(None, None, FakeCoreApi()).capture_logs(
            "attempt", "incident"
        )
        == []
    )


def test_bad_log_identity_is_isolated_and_reported_without_an_annotation() -> None:
    current = pod(0)
    current["metadata"]["annotations"].pop("gpu-fault.io/training-container")
    current["spec"]["containers"] = []
    core = FakeCoreApi([current])
    snapshots = KubernetesWorkloadStopper(None, None, core).capture_logs(
        "attempt", "incident"
    )
    assert len(snapshots) == 1
    assert "training container identity" in snapshots[0]["capture_error"]
    assert core.pod_patches == []


def test_typed_pod_infers_its_only_container_and_preserves_bounded_tail() -> None:
    current = V1Pod(
        metadata=V1ObjectMeta(name="trainer", uid="pod-uid"),
        spec=V1PodSpec(containers=[V1Container(name="train")], node_name="node-a"),
    )
    core = FakeCoreApi()
    core.list_pod_for_all_namespaces = lambda **kw: SimpleNamespace(items=[current])
    stopper = KubernetesWorkloadStopper(
        None, None, core, workload_log_max_bytes=9, workload_log_annotation_tail_bytes=4
    )
    (snapshot,) = stopper.capture_logs("attempt", "incident")
    assert snapshot["pod_uid"] == "pod-uid"
    assert snapshot["container_name"] == "train"
    assert snapshot["node_id"] == "node-a"
    assert snapshot["tail_bytes"] == 9
    assert snapshot["truncated"] is True
    annotated = json.loads(
        core.pod_patches[0][2]["metadata"]["annotations"][
            "gpu-fault.io/workload-log-snapshot"
        ]
    )
    assert len(annotated["tail"].encode()) <= 4
    assert annotated["annotation_tail_truncated"] is True


@pytest.mark.parametrize(
    "destination", ["s3://bucket", "s3://bucket/prefix", "https://wrong/path"]
)
def test_s3_archive_uses_fake_sdk_transport_and_retains_evidence_on_bad_uri(
    monkeypatch, destination
) -> None:
    calls = []
    client_calls = []

    def client(service):
        client_calls.append(service)
        return SimpleNamespace(put_object=lambda **kw: calls.append(kw))

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=client))
    core = FakeCoreApi([pod(0)])
    (snapshot,) = KubernetesWorkloadStopper(
        None, None, core, workload_log_s3_uri=destination
    ).capture_logs("attempt", "incident", annotate=False)
    if destination.startswith("https"):
        assert "must be an s3://" in snapshot["archive_error"]
        assert snapshot["s3_uri"] is None
        assert client_calls == []
    else:
        assert client_calls == ["s3"]
        assert len(calls) == 1
        request = calls[0]
        assert request["Bucket"] == "bucket"
        assert request["Key"].startswith(
            "prefix/" if destination.endswith("prefix") else "attempt/"
        ), "archive keys must retain the configured prefix or the attempt namespace"
        assert gzip.decompress(request["Body"]) == b"training log line\n"
        assert (
            request["Metadata"]["sha256"]
            == hashlib.sha256(b"training log line\n").hexdigest()
        )
        assert request["ContentEncoding"] == "gzip"
        assert request["ContentType"] == "text/plain"
    assert snapshot["tail"] == "training log line\n"
    assert core.pod_patches == []


def test_missing_s3_sdk_does_not_discard_captured_tail(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "boto3", None)
    core = FakeCoreApi([pod(0)])
    (snapshot,) = KubernetesWorkloadStopper(
        None, None, core, workload_log_s3_uri="s3://bucket"
    ).capture_logs("attempt", "incident")
    assert "boto3 is required" in snapshot["archive_error"]
    assert snapshot["tail"] == "training log line\n"
    assert len(core.pod_patches) == 1
