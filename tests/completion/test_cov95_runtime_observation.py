from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest
from kubernetes.client import ApiClient, V1Job, V1ObjectMeta

from gpu_fault.completion_observation import (
    WORKLOAD_READ_TIMEOUT,
    WorkloadVerdict,
    list_completion_pods,
    read_workload_object,
    workload_objects_verdict,
)
from tests.completion._support import Clock, FakeCoreApi, FakeSink, pod
from tests.completion.test_completion_controller_restore import restored_watcher


@pytest.mark.parametrize(
    "unknown", [{}, {"status": None}, {"status": []}, {"status": {}}]
)
def test_unknown_sibling_cannot_be_discarded_from_success_verdict(unknown) -> None:
    verdict = workload_objects_verdict([{"status": {"succeeded": 2}}, unknown], 2)
    assert verdict is WorkloadVerdict.UNKNOWN, "every workload needs success evidence"


@pytest.mark.parametrize("unknown", [None, {}, {"active": 0}])
def test_restored_attempt_with_partial_workload_evidence_remains_stopped(unknown):
    class Batch:
        def __init__(self):
            self.reads = []

        def read_namespaced_job(self, name, namespace, **kwargs):
            self.reads.append((name, namespace, kwargs))
            if name == "unknown":
                if unknown is None:
                    raise RuntimeError("fake workload read unavailable")
                return {"metadata": {"name": name}, "status": unknown}
            return {"metadata": {"name": name}, "status": {"succeeded": 1}}

    batch = Batch()
    clock, sink = Clock(), FakeSink()
    core = FakeCoreApi(
        [pod(0, workload_ids=["default/job/succeeded", "default/job/unknown"])]
    )
    controller, stopper = restored_watcher(core, sink, clock, batch)
    controller.run_once()
    clock.value += timedelta(seconds=31)
    controller.run_once()
    terminals = [payload for path, payload in sink.posts if path.endswith("/terminal")]
    assert [item["terminal_status"] for item in terminals] == ["STOPPED"]
    assert stopper.calls == []
    # Two bounded reads per attempt: the first missing pass reads the owners
    # once to tell a deleted workload (404) from a transient absence, and the
    # tombstone at the end of the grace reads them again for the outcome.
    assert [name for name, _, _ in batch.reads] == ["succeeded", "unknown"] * 2
    assert all(
        kwargs["_request_timeout"] == WORKLOAD_READ_TIMEOUT
        for _, _, kwargs in batch.reads
    ), "every workload read must remain bounded"


@pytest.mark.parametrize(
    "status,expected",
    [
        ({}, WorkloadVerdict.UNKNOWN),
        ({"succeeded": 1}, WorkloadVerdict.UNKNOWN),
        ({"succeeded": 2}, WorkloadVerdict.SUCCEEDED),
        ({"failed": 1}, WorkloadVerdict.FAILED),
        ({"active": 1, "failed": 1}, WorkloadVerdict.ACTIVE),
        (
            {
                "replicaStatuses": {
                    "Master": {"succeeded": 1},
                    "Worker": {"succeeded": 1},
                }
            },
            WorkloadVerdict.SUCCEEDED,
        ),
        (
            {"replicated_jobs_status": [{"succeeded": 1}, None, {"active": 1}]},
            WorkloadVerdict.ACTIVE,
        ),
        (
            {
                "succeeded": 2,
                "conditions": [
                    None,
                    {"type": "Failed", "status": "False"},
                    {"type": "Running", "status": "True"},
                ],
            },
            WorkloadVerdict.SUCCEEDED,
        ),
        (
            {"active": 1, "conditions": [{"type": "Failed", "status": "True"}]},
            WorkloadVerdict.FAILED,
        ),
    ],
)
def test_workload_status_shapes_retain_terminal_and_active_precedence(status, expected):
    assert workload_objects_verdict([{"status": status}], 2) is expected


def test_all_success_and_active_sibling_are_distinct_from_missing_evidence() -> None:
    success = {"status": {"succeeded": 2}}
    assert workload_objects_verdict([success, success], 2) is WorkloadVerdict.SUCCEEDED
    assert workload_objects_verdict([], 2) is WorkloadVerdict.UNKNOWN
    assert workload_objects_verdict([success, {"status": {"active": 1}}], 2) is (
        WorkloadVerdict.ACTIVE
    )
    assert workload_objects_verdict([success, {"status": {"failed": 1}}], 2) is (
        WorkloadVerdict.FAILED
    )


@pytest.mark.parametrize(
    "workload_id",
    [
        "invalid",
        "too/many/path/segments",
        "default/deployment/trainer",
        "default/trainer",
    ],
)
def test_unsupported_or_unavailable_workload_reads_are_unknown(workload_id) -> None:
    controller = SimpleNamespace(workload_stopper=None)
    assert read_workload_object(controller, workload_id) is None


@pytest.mark.parametrize(
    ("workload_id", "expected"),
    [
        ("default/trainer", ("trainer", "default")),
        (
            "default/PyTorchJob/trainer",
            ("kubeflow.org", "v1", "default", "pytorchjobs", "trainer"),
        ),
        (
            "default/jobset/trainer",
            ("jobset.x-k8s.io", "v1alpha2", "default", "jobsets", "trainer"),
        ),
    ],
)
def test_workload_read_contract_uses_real_kubernetes_serialization(
    workload_id, expected
):
    calls = []

    def read(*args, **kwargs):
        calls.append((args, kwargs))
        return V1Job(metadata=V1ObjectMeta(name="trainer"))

    with ApiClient() as serializer:
        controller = SimpleNamespace(
            workload_stopper=SimpleNamespace(
                batch=SimpleNamespace(read_namespaced_job=read),
                custom=SimpleNamespace(get_namespaced_custom_object=read),
            ),
            serializer=serializer.sanitize_for_serialization,
        )
        assert read_workload_object(controller, workload_id) == {
            "metadata": {"name": "trainer"}
        }
    assert calls == [(expected, {"_request_timeout": WORKLOAD_READ_TIMEOUT})]


def test_unserializable_workload_object_is_unknown() -> None:
    controller = SimpleNamespace(
        workload_stopper=SimpleNamespace(
            batch=SimpleNamespace(read_namespaced_job=lambda *a, **kw: object())
        ),
        serializer=lambda value: None,
    )
    assert read_workload_object(controller, "default/trainer") is None


def test_namespaced_pod_listing_accepts_the_typed_api_response() -> None:
    calls = []
    pods = [pod(0)]

    def list_pods(namespace, **kwargs):
        calls.append((namespace, kwargs))
        return SimpleNamespace(
            items=pods, metadata=SimpleNamespace(resource_version="revision-1")
        )

    assert list_completion_pods(
        SimpleNamespace(list_namespaced_pod=list_pods), "training"
    ) == (pods, "revision-1")
    assert calls == [("training", {"label_selector": "gpu-fault.io/managed=true"})]
