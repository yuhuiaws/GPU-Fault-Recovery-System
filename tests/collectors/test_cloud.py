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
        def list_node(self, **_kwargs):
            return SimpleNamespace(
                metadata=SimpleNamespace(_continue=None),
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
                ],
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


def test_kubernetes_node_resource_summary_is_not_reported_as_a_recovery() -> None:
    """A healthy node's periodic delivery is a summary, not a recovery.

    The reason was chosen from the mismatch state alone, so an unchanged healthy
    allocatable count that delivered only because its summary interval had
    elapsed still said `recovered:efa`. That claimed a transition that never
    happened, and since the control plane only declines to persist evidence for
    batches whose sole reason is `health-summary`, one idle node kept writing a
    raw evidence record every time a resource's summary came due.
    """

    allocatable = {"vpc.amazonaws.com/efa": "16", "nvidia.com/gpu": "8"}

    class Core:
        def list_node(self, **_kwargs):
            return SimpleNamespace(
                metadata=SimpleNamespace(_continue=None),
                items=[
                    {
                        "metadata": {
                            "name": "gpu-worker",
                            "labels": {
                                "node.kubernetes.io/instance-type": ("ml.p5en.48xlarge")
                            },
                        },
                        "status": {"allocatable": dict(allocatable)},
                    }
                ],
            )

        def list_pod_for_all_namespaces(self, **_kwargs):
            return SimpleNamespace(items=[])

    sink = RecordingSink()
    collector = KubernetesNodeResourceCollector(
        sink,
        context(),
        core_api=Core(),
        required_consecutive_samples=2,
        health_summary_seconds=300,
        now=lambda: NOW,
    )

    assert collector.collect_once().delivered == 1
    assert sink.requests[-1][1]["edge_filter_reasons"] == [
        "baseline:efa",
        "baseline:gpu",
    ]

    # Nothing changed, and no summary interval has elapsed: stay silent.
    collector.now = lambda: NOW + timedelta(seconds=15)
    assert collector.collect_once().delivered == 0

    # A summary interval later the counts are still healthy and unchanged.
    collector.now = lambda: NOW + timedelta(seconds=1200)
    assert collector.collect_once().delivered == 1
    assert sink.requests[-1][1]["edge_filter_reasons"] == ["health-summary"], (
        "an unchanged healthy allocatable count was reported as a state change"
    )

    # A real transition out of a persistent mismatch still says `recovered`.
    allocatable["vpc.amazonaws.com/efa"] = "15"
    for offset in (1215, 1230):
        collector.now = lambda offset=offset: NOW + timedelta(seconds=offset)
        collector.collect_once()
    assert (
        "threshold:efa_kubernetes_allocatable_mismatch"
        in (sink.requests[-1][1]["edge_filter_reasons"])
    )
    allocatable["vpc.amazonaws.com/efa"] = "16"
    collector.now = lambda: NOW + timedelta(seconds=1245)
    assert collector.collect_once().delivered == 1
    assert sink.requests[-1][1]["edge_filter_reasons"] == ["recovered:efa"]


def _paged_node(name: str) -> dict:
    return {
        "metadata": {
            "name": name,
            "labels": {"node.kubernetes.io/instance-type": "ml.p5en.48xlarge"},
        },
        "status": {
            "allocatable": {"vpc.amazonaws.com/efa": "16", "nvidia.com/gpu": "8"}
        },
    }


def test_kubernetes_node_resource_collector_follows_list_continue_tokens() -> None:
    """The periodic node LIST is paginated, and every page is processed.

    A 500-node cluster answered with one unpaginated LIST every 15 seconds is
    5-10 MB per call and, past the watch-cache window, an etcd read. The
    collector must ask for ``limit`` and follow ``metadata._continue`` until
    the apiserver hands back an empty token, observing the nodes on every page.
    """

    class Core:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def list_node(self, **kwargs):
            self.calls.append(dict(kwargs))
            if kwargs.get("_continue"):
                return SimpleNamespace(
                    metadata=SimpleNamespace(_continue=None),
                    items=[_paged_node("gpu-worker-b")],
                )
            return SimpleNamespace(
                metadata=SimpleNamespace(_continue="page-two"),
                items=[_paged_node("gpu-worker-a")],
            )

        def list_pod_for_all_namespaces(self, **_kwargs):
            return SimpleNamespace(items=[])

    core = Core()
    sink = RecordingSink()
    collector = KubernetesNodeResourceCollector(
        sink, context(), core_api=core, list_page_size=1, now=lambda: NOW
    )

    stats = collector.collect_once()

    assert stats.observed == 2
    assert stats.delivered == 2
    assert sorted(payload["node_id"] for _, payload in sink.requests) == [
        "gpu-worker-a",
        "gpu-worker-b",
    ]
    assert core.calls == [
        {"limit": 1, "_continue": None},
        {"limit": 1, "_continue": "page-two"},
    ]


def test_kubernetes_node_resource_collector_rejects_non_positive_page_size() -> None:
    with pytest.raises(ValueError):
        KubernetesNodeResourceCollector(
            RecordingSink(), context(), core_api=None, list_page_size=0
        )


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
