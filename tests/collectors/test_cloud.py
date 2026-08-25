from __future__ import annotations

from ._support import (
    HMA_FAULT_DETAILS,
    HMA_FAULT_REASONS,
    HMA_HEALTH_STATUS,
    NOW,
    CloudWatchHmaCollector,
    CollectorError,
    KubernetesHmaNodeCollector,
    KubernetesNodeResourceCollector,
    RecordingSink,
    SimpleNamespace,
    cloudwatch_envelope,
    context,
    json,
    pytest,
    timedelta,
)


def test_kubernetes_collector_filters_and_deduplicates_node() -> None:
    sink = RecordingSink()
    collector = KubernetesHmaNodeCollector(sink, context(), now=lambda: NOW)
    ordinary = {
        "metadata": {
            "name": "cpu-worker",
            "resourceVersion": "1",
            "labels": {"kubernetes.io/hostname": "cpu-worker"},
        }
    }
    hma_node = {
        "metadata": {
            "name": "gpu-worker",
            "resourceVersion": "42",
            "labels": {
                HMA_HEALTH_STATUS: "Unschedulable",
                HMA_FAULT_REASONS: "XidHardwareFailure",
            },
            "annotations": {
                HMA_FAULT_DETAILS: json.dumps(
                    {
                        "faults": [
                            {
                                "timestamp": NOW.isoformat(),
                                "message": ("NVRM: Xid (PCI:0000:b9:00): 94"),
                            }
                        ]
                    }
                )
            },
        }
    }

    assert collector.collect_node(ordinary).skipped == 1
    assert collector.collect_node(hma_node).delivered == 1
    assert collector.collect_node(hma_node).duplicates == 1
    assert len(sink.requests) == 1
    path, payload = sink.requests[0]
    assert path.endswith("/hyperpod-hma/kubernetes-node")
    assert payload["node"]["metadata"]["resourceVersion"] == "42"
    assert payload["evidence_ref"].endswith("resourceVersion=42")


def test_kubernetes_node_resource_collector_detects_allocatable_loss() -> None:
    class Core:
        def list_node(self):
            return SimpleNamespace(
                items=[
                    {
                        "metadata": {
                            "name": "gpu-worker",
                            "labels": {
                                "node.kubernetes.io/instance-type": ("ml.p5en.48xlarge")
                            },
                        },
                        "status": {
                            "allocatable": {
                                "vpc.amazonaws.com/efa": "15",
                                "nvidia.com/gpu": "8",
                            }
                        },
                    }
                ]
            )

        def list_pod_for_all_namespaces(self, **_kwargs):
            return SimpleNamespace(
                items=[
                    {
                        "metadata": {
                            "namespace": "training",
                            "labels": {
                                "gpu-fault.io/managed": "true",
                                "training.kubeflow.org/job-name": "job-a",
                            },
                        },
                        "spec": {"nodeName": "gpu-worker"},
                        "status": {"phase": "Running"},
                    }
                ]
            )

    sink = RecordingSink()
    collector = KubernetesNodeResourceCollector(
        sink,
        context(),
        core_api=Core(),
        required_consecutive_samples=2,
        now=lambda: NOW,
    )

    assert collector.collect_once().delivered == 1
    collector.now = lambda: NOW + timedelta(seconds=15)
    assert collector.collect_once().delivered == 1

    _, payload = sink.requests[-1]
    by_name = {item["name"]: item for item in payload["samples"]}
    assert by_name["efa_kubernetes_allocatable_mismatch"]["value"] == 1
    assert (
        by_name["efa_kubernetes_allocatable_mismatch"]["labels"]["failure_mode"]
        == "KUBERNETES_RESOURCE_MISSING"
    )
    assert payload["workload_state"] == "ACTIVE"
    assert payload["affected_workload_ids"] == ["training/pytorchjob/job-a"]
    assert by_name["gpu_kubernetes_allocatable_mismatch"]["value"] == 0


def test_cloudwatch_collector_decodes_filters_and_delivers() -> None:
    sink = RecordingSink()
    collector = CloudWatchHmaCollector(sink, context(), now=lambda: NOW)
    message = json.dumps(
        {
            "HealthMonitoringAgentDetectionEvent": "HealthEvent",
            "details: ": {
                "reason": "XidHardwareFailure",
                "message": "NVRM: Xid (PCI:0000:b9:00): 94",
            },
        }
    )

    stats = collector.collect_subscription(
        cloudwatch_envelope(
            [
                {
                    "id": "event-1",
                    "timestamp": 1784548800000,
                    "message": "ordinary HMA log",
                },
                {"id": "event-2", "timestamp": 1784548801000, "message": message},
            ]
        )
    )

    assert stats.observed == 2
    assert stats.skipped == 1
    assert stats.delivered == 1
    path, payload = sink.requests[0]
    assert path.endswith("/hyperpod-hma/cloudwatch")
    assert payload["node_id"] == "worker-1"
    assert payload["log_event_id"] == "event-2"
    assert payload["collected_at"] == NOW.isoformat()


def test_cloudwatch_collector_supports_explicit_node_regex() -> None:
    sink = RecordingSink()
    collector = CloudWatchHmaCollector(
        sink, context(), node_pattern=r"nodes/(?P<node_id>[^/]+)/hma"
    )
    message = json.dumps({"HealthMonitoringAgentDetectionEvent": "HealthEvent"})

    collector.collect_subscription(
        cloudwatch_envelope(
            [{"id": "event-3", "timestamp": 1784548800000, "message": message}],
            stream="prefix/nodes/ip-10-0-0-1/hma/output",
        )
    )

    assert sink.requests[0][1]["node_id"] == "ip-10-0-0-1"


def test_cloudwatch_collector_maps_hyperpod_instance_stream() -> None:
    sink = RecordingSink()
    collector = CloudWatchHmaCollector(
        sink,
        context(),
        node_pattern=(
            r"^SagemakerHealthMonitoringAgent/[^/]+/"
            r"(?P<node_id>i-[a-f0-9]+)$"
        ),
        node_prefix="hyperpod-",
    )
    message = json.dumps({"HealthMonitoringAgentDetectionEvent": "HealthEvent"})

    collector.collect_subscription(
        cloudwatch_envelope(
            [{"id": "event-hyperpod", "timestamp": 1784548800000, "message": message}],
            stream=(
                "SagemakerHealthMonitoringAgent/example-p5en-group/i-00000000000000001"
            ),
        )
    )

    assert sink.requests[0][1]["node_id"] == ("hyperpod-i-00000000000000001")


def test_cloudwatch_collector_rejects_unmapped_node() -> None:
    collector = CloudWatchHmaCollector(RecordingSink(), context())

    with pytest.raises(CollectorError, match="cannot derive node ID"):
        collector.collect_subscription(
            cloudwatch_envelope([], stream="stream-without-node-contract")
        )
