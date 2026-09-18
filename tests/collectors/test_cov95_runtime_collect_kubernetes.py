"""Node-resource collection through fake Kubernetes clients, never a cluster."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from gpu_fault.collectors.cloud import kubernetes as module
from gpu_fault.collectors.gpu.discovery import INSTANCE_ACCELERATOR_COUNTS
from gpu_fault.collectors.sinks import CollectorError
from tests.collectors import _cov95_runtime_collect as support

isolated_runtime = support.isolated_runtime


def node(name="node-a", **metadata):
    counts = INSTANCE_ACCELERATOR_COUNTS["p5.48xlarge"]
    return {
        "metadata": {
            "name": name,
            "labels": {"node.kubernetes.io/instance-type": "ml.p5.48xlarge"},
            **metadata,
        },
        "status": {
            "allocatable": {
                "nvidia.com/gpu": counts["gpu"],
                "vpc.amazonaws.com/efa": counts["efa"],
            }
        },
    }


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (
            {"annotations": {"gpu-fault.io/workload-ids": '["training-a","ignored"]'}},
            "training-a",
        ),
        (
            {
                "annotations": {"gpu-fault.io/workload-ids": "bad"},
                "labels": {"training.kubeflow.org/job-name": "job-a"},
            },
            "default/pytorchjob/job-a",
        ),
        (
            {
                "annotations": {"gpu-fault.io/workload-ids": "{}"},
                "labels": {"jobset.sigs.k8s.io/jobset-name": "set-a"},
                "namespace": "training",
            },
            "training/jobset/set-a",
        ),
        (
            {
                "ownerReferences": [
                    {"controller": False, "name": "ignored"},
                    {"controller": True, "name": "job-a"},
                ]
            },
            "default/job/job-a",
        ),
        (
            {
                "ownerReferences": [
                    {"controller": True, "name": "job-a", "kind": "JobSet"}
                ]
            },
            "default/jobset/job-a",
        ),
        ({}, None),
    ],
)
def test_workload_identity_is_resolved_from_managed_pods_in_real_node_batches(
    metadata, expected
):
    pods = [
        {
            "metadata": metadata,
            "spec": {"nodeName": "node-a"},
            "status": {"phase": "Running"},
        },
        {"metadata": {"deletionTimestamp": "stamp"}, "spec": {"nodeName": "node-a"}},
        {
            "metadata": metadata,
            "spec": {"nodeName": "node-a"},
            "status": {"phase": "Succeeded"},
        },
        {"metadata": metadata, "spec": {}, "status": {"phase": "Running"}},
    ]
    calls = []

    def list_pods(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(items=pods)

    core = SimpleNamespace(
        list_pod_for_all_namespaces=list_pods,
        list_node=lambda **kwargs: SimpleNamespace(items=[node()], metadata=None),
    )
    sink = support.RecordingSink()
    collector = module.KubernetesNodeResourceCollector(
        sink, support.collector_context(), core_api=core, now=lambda: support.NOW
    )
    assert collector.collect_once().delivered == 1
    payload = sink.requests[0][1]
    assert payload["affected_workload_ids"] == ([] if expected is None else [expected])
    assert payload["workload_state"] == ("IDLE" if expected is None else "ACTIVE")
    assert calls == [{"label_selector": "gpu-fault.io/managed=true"}]


@pytest.mark.parametrize("allocatable", [None, "unavailable", {}, -1])
def test_invalid_allocatable_is_not_healthy_capacity(allocatable):
    raw = node()
    raw["status"]["allocatable"]["nvidia.com/gpu"] = allocatable
    core = SimpleNamespace(
        list_pod_for_all_namespaces=lambda **kwargs: SimpleNamespace(items=[]),
        list_node=lambda **kwargs: SimpleNamespace(items=[raw]),
    )
    sink = support.RecordingSink()
    collector = module.KubernetesNodeResourceCollector(
        sink,
        support.collector_context(),
        core_api=core,
        required_consecutive_samples=1,
        now=lambda: support.NOW,
    )
    assert collector.collect_once().delivered == 1
    samples = {item["name"]: item for item in sink.requests[0][1]["samples"]}
    assert samples["gpu_kubernetes_allocatable_mismatch"]["value"] == 1
    assert (
        "threshold:gpu_kubernetes_allocatable_mismatch"
        in sink.requests[0][1]["edge_filter_reasons"]
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {"name": None},
        {"labels": {}},
        {"labels": {"node.kubernetes.io/instance-type": "unknown"}},
    ],
)
def test_unknown_node_identity_is_skipped_without_inventing_expected_devices(metadata):
    core = SimpleNamespace(
        list_pod_for_all_namespaces=lambda **kwargs: SimpleNamespace(items=[]),
        list_node=lambda **kwargs: SimpleNamespace(items=[node(**metadata)]),
    )
    sink = support.RecordingSink()
    stats = module.KubernetesNodeResourceCollector(
        sink, support.collector_context(), core_api=core
    ).collect_once()
    assert (stats.observed, stats.skipped, stats.delivered) == (1, 1, 0)
    assert sink.requests == []


@pytest.mark.parametrize("fallback", [False, True])
def test_default_client_setup_and_empty_cycle_use_only_fake_sdk(
    monkeypatch, tmp_path, fallback
):
    calls = []

    class ConfigError(Exception):
        pass

    def incluster():
        calls.append("incluster")
        if fallback:
            raise ConfigError("outside fake cluster")

    core = SimpleNamespace(
        list_pod_for_all_namespaces=lambda **kwargs: SimpleNamespace(items=[]),
        list_node=lambda **kwargs: SimpleNamespace(items=[]),
    )
    sdk = SimpleNamespace(
        client=SimpleNamespace(
            CoreV1Api=lambda: calls.append("client") or core,
            ApiClient=lambda: SimpleNamespace(
                sanitize_for_serialization=lambda raw: raw
            ),
        ),
        config=SimpleNamespace(
            load_incluster_config=incluster,
            load_kube_config=lambda: calls.append("kubeconfig"),
        ),
    )
    monkeypatch.setitem(sys.modules, "kubernetes", sdk)
    monkeypatch.setitem(
        sys.modules,
        "kubernetes.config.config_exception",
        SimpleNamespace(ConfigException=ConfigError),
    )
    heartbeat = tmp_path / "alive"
    monkeypatch.setattr(module, "NODE_RESOURCE_HEARTBEAT_PATH", str(heartbeat))

    def sleep(seconds):
        calls.append(seconds)
        raise support.StopLoop

    monkeypatch.setattr(module, "time", SimpleNamespace(sleep=sleep))
    collector = module.KubernetesNodeResourceCollector(
        support.RecordingSink(), support.collector_context(), interval_seconds=7
    )
    with pytest.raises(support.StopLoop):
        collector.run()
    assert calls == (
        ["incluster", "kubeconfig", "client", 7]
        if fallback
        else ["incluster", "client", 7]
    )
    assert heartbeat.exists(), "an empty but completed cycle must refresh liveness"


def test_missing_sdk_and_missing_client_fail_before_any_collection(monkeypatch):
    collector = module.KubernetesNodeResourceCollector(
        support.RecordingSink(), support.collector_context()
    )
    with pytest.raises(CollectorError, match="CoreV1Api"):
        collector.collect_once()
    monkeypatch.setitem(sys.modules, "kubernetes", None)
    with pytest.raises(CollectorError, match="Kubernetes EFA inventory support"):
        collector.run()


@pytest.mark.parametrize(
    "options",
    [
        {"interval_seconds": 0},
        {"required_consecutive_samples": 0},
        {"health_summary_seconds": 0},
    ],
)
def test_resource_collector_rejects_invalid_observation_windows(options):
    with pytest.raises(ValueError, match="must be positive"):
        module.KubernetesNodeResourceCollector(
            support.RecordingSink(), support.collector_context(), **options
        )


def test_cycle_and_heartbeat_write_failures_remain_visible_without_stopping_loop(
    monkeypatch, tmp_path, caplog
):
    calls = []

    def fail_list(**kwargs):
        calls.append("list")
        raise OSError("fake API unavailable")

    heartbeat = tmp_path / "missing-parent/alive"
    monkeypatch.setattr(module, "NODE_RESOURCE_HEARTBEAT_PATH", str(heartbeat))
    core = SimpleNamespace(list_pod_for_all_namespaces=fail_list)
    collector = module.KubernetesNodeResourceCollector(
        support.RecordingSink(), support.collector_context(), core_api=core
    )

    def sleep(seconds):
        raise support.StopLoop

    monkeypatch.setattr(module, "time", SimpleNamespace(sleep=sleep))
    with pytest.raises(support.StopLoop):
        collector.run()
    assert calls == ["list"]
    assert "fake API unavailable" in caplog.text
    assert caplog.text.count("could not write node resource collector heartbeat") == 2
