"""``KubernetesWorkloadStopper``: log capture, S3 archival and the suspend patches."""

from __future__ import annotations

import gzip
import json

from gpu_fault.completion_controller import KubernetesWorkloadStopper
from tests.completion._support import FakeCoreApi, pod


def test_workload_log_s3_failure_keeps_tail_evidence() -> None:
    core = FakeCoreApi([pod(0, exit_code=1)])

    def fail_upload(_destination, _payload):
        raise RuntimeError("S3 unavailable")

    stopper = KubernetesWorkloadStopper(
        batch_api=None,
        custom_api=None,
        core_api=core,
        workload_log_s3_uri="s3://bucket/workload-logs",
        workload_log_uploader=fail_upload,
    )

    snapshots = stopper.capture_logs("train-a1", "incident-a", pods=core.pods)

    assert len(snapshots) == 1
    assert snapshots[0]["tail"] == "training log line\n"
    assert snapshots[0]["s3_uri"] is None
    assert "S3 unavailable" in snapshots[0]["archive_error"]
    assert snapshots[0]["record_id"]


def test_kubernetes_workload_stopper_suspends_supported_kinds() -> None:
    class Batch:
        calls = []

        def patch_namespaced_job(self, name, namespace, body):
            self.calls.append((namespace, name, body))

    class Custom:
        calls = []

        def patch_namespaced_custom_object(
            self, group, version, namespace, plural, name, body
        ):
            self.calls.append((group, version, namespace, plural, name, body))

    batch = Batch()
    custom = Custom()
    stopper = KubernetesWorkloadStopper(batch, custom)

    stopper.stop(
        (
            "train/job-a",
            "train/job-a",
            "train/pytorchjob/torch-a",
            "train/jobset/set-a",
        ),
        "attempt-a",
    )

    assert batch.calls[0][2]["spec"] == {"suspend": True}
    assert (
        batch.calls[0][2]["metadata"]["annotations"]["gpu-fault.io/passive-stop-mode"]
        == "emergency-fallback"
    )
    assert len(batch.calls) == 1
    assert custom.calls[0][5]["spec"] == {"runPolicy": {"suspend": True}}
    assert custom.calls[1][5]["spec"] == {"suspend": True}


def test_emergency_stopper_annotates_pods_before_suspend() -> None:
    class Batch:
        calls = []

        def patch_namespaced_job(self, name, namespace, body):
            self.calls.append((namespace, name, body))

    class Custom:
        calls = []

        def patch_namespaced_custom_object(
            self, group, version, namespace, plural, name, body
        ):
            self.calls.append((group, version, namespace, plural, name, body))

    core = FakeCoreApi([pod(0, workload_ids=["train/job-a"])])
    batch = Batch()
    uploaded = {}

    def upload(destination, value):
        uploaded["destination"] = destination
        uploaded["value"] = value
        return destination

    stopper = KubernetesWorkloadStopper(
        batch,
        Custom(),
        core,
        workload_log_s3_uri="s3://audit-bucket/workload-logs",
        workload_log_uploader=upload,
    )

    snapshots = stopper.stop(("train/job-a",), "train-a1", "inc-attempt-failure-1")

    assert len(core.pod_patches) == 1
    namespace, pod_name, patch = core.pod_patches[0]
    assert (namespace, pod_name) == ("default", "worker-0")
    annotations = patch["metadata"]["annotations"]
    assert (
        annotations["gpu-fault.io/termination-initiator-incident-id"]
        == "inc-attempt-failure-1"
    )
    assert annotations["gpu-fault.io/passive-stop-attempt"] == ("train-a1")
    annotated_snapshot = json.loads(annotations["gpu-fault.io/workload-log-snapshot"])
    assert annotated_snapshot["record_id"] == snapshots[0]["record_id"]
    assert annotated_snapshot["s3_uri"] == snapshots[0]["s3_uri"]
    assert len(batch.calls) == 1
    assert snapshots[0]["tail"] == "training log line\n"
    assert snapshots[0]["s3_uri"].startswith("s3://audit-bucket/workload-logs/"), (
        'expected snapshots[0]["s3_uri"].startswith("s3://audit-bucket/workload-logs/") to be truthy'
    )
    assert gzip.decompress(uploaded["value"]) == (b"training log line\n")
