from __future__ import annotations

from ._support import (
    HMA_FAULT_DETAILS,
    HMA_FAULT_REASONS,
    HMA_HEALTH_STATUS,
    NOW,
    Any,
    BufferingSink,
    CloudWatchHmaCollector,
    CollectorError,
    KubernetesHmaNodeCollector,
    KubernetesNodeResourceCollector,
    RecordingSink,
    SimpleNamespace,
    SqsHmaConsumer,
    StopTheLoop,
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


class _NodeRejectingSink:
    """The control plane rejects one node's batch and accepts every other."""

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.requests.append((path, payload))
        if payload.get("node_id") == self.node_id:
            raise CollectorError("rejected (422)", status_code=422)
        return {"accepted": True}


def _two_node_core() -> Any:
    class Core:
        def list_node(self, **_kwargs):
            return SimpleNamespace(
                metadata=SimpleNamespace(_continue=None),
                items=[_paged_node("gpu-worker-a"), _paged_node("gpu-worker-b")],
            )

        def list_pod_for_all_namespaces(self, **_kwargs):
            return SimpleNamespace(items=[])

    return Core()


def test_kubernetes_node_resources_buffered_batches_cover_every_node() -> None:
    """One buffered post used to abort the whole sample (ARCH-G3).

    ``HttpEventSink.post`` raises after the outbox took the record, and
    ``collect_once`` had no per-node guard, so the first node whose batch was
    buffered ended the cycle: every node after it in list order was never
    sampled, and the aborted node was still "due" next cycle, so it re-buffered
    a near-identical batch with a fresh ``batch_id`` every 15 s.
    """

    sink = BufferingSink()
    collector = KubernetesNodeResourceCollector(
        sink, context(), core_api=_two_node_core(), now=lambda: NOW
    )

    collector.collect_once()

    assert [payload["node_id"] for _, payload in sink.requests] == [
        "gpu-worker-a",
        "gpu-worker-b",
    ], "a buffered batch for the first node stopped the whole cycle"

    collector.now = lambda: NOW + timedelta(seconds=15)
    collector.collect_once()

    assert len(sink.requests) == 2, (
        "an unchanged node re-buffered a batch the outbox had already taken"
    )


def test_kubernetes_node_resources_rejected_node_does_not_stop_the_cycle() -> None:
    """A rejected node keeps its edge: the rest of the fleet is still sampled."""

    sink = _NodeRejectingSink("gpu-worker-a")
    collector = KubernetesNodeResourceCollector(
        sink, context(), core_api=_two_node_core(), now=lambda: NOW
    )

    stats = collector.collect_once()

    assert [payload["node_id"] for _, payload in sink.requests] == [
        "gpu-worker-a",
        "gpu-worker-b",
    ], "the rejected node's batch stopped the cycle before the next node"
    assert stats.delivered == 1, "only the accepted node counts as delivered"

    collector.now = lambda: NOW + timedelta(seconds=15)
    collector.collect_once()

    assert sink.requests[-1][1]["node_id"] == "gpu-worker-a", (
        "the rejected node's edge was consumed although nothing was delivered"
    )
    assert sink.requests[-1][1]["edge_filter_reasons"] == [
        "baseline:efa",
        "baseline:gpu",
    ], "the rejected node's baseline edge was not re-reported"


class _FakeNodeWatch:
    """One ``watch.Watch()`` replaying a scripted event list, then ending."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = list(events)
        self.stop_calls = 0

    def stream(self, *_args, **_kwargs):
        yield from self.events

    def stop(self) -> None:
        self.stop_calls += 1


class _FakeNodeApi:
    """A ``CoreV1Api`` whose LIST is scripted round by round."""

    def __init__(self, rounds: list[Any]) -> None:
        self.rounds = list(rounds)
        self.calls = 0

    def list_node(self, **_kwargs):
        self.calls += 1
        if not self.rounds:
            raise StopTheLoop
        outcome = self.rounds.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(
            items=list(outcome), metadata=SimpleNamespace(resource_version="7")
        )


def _hma_node(resource_version: str, name: str = "gpu-worker") -> dict[str, Any]:
    return {
        "metadata": {
            "name": name,
            "resourceVersion": resource_version,
            "labels": {HMA_HEALTH_STATUS: "Unschedulable"},
        }
    }


def _patch_kubernetes(
    monkeypatch: pytest.MonkeyPatch, api: Any, watches: list[_FakeNodeWatch]
) -> None:
    from gpu_fault.collectors.cloud import kubernetes as module

    monkeypatch.setattr("kubernetes.config.load_incluster_config", lambda: None)
    monkeypatch.setattr("kubernetes.client.CoreV1Api", lambda: api)
    monkeypatch.setattr(
        "kubernetes.client.ApiClient",
        lambda: SimpleNamespace(sanitize_for_serialization=lambda value: value),
    )
    queue = list(watches)
    monkeypatch.setattr(
        "kubernetes.watch.Watch", lambda: queue.pop(0) if queue else _FakeNodeWatch([])
    )
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)


def test_kubernetes_hma_run_survives_sink_failure_during_relist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The watcher must not exit the process on a LIST or post failure.

    The initial LIST and its posts sat outside the ``try`` that guards the
    watch, and ``post`` raises even once the outbox has taken the record, so the
    first unreachable control plane exited ``run()``: Kubernetes restarted the
    Pod into CrashLoopBackOff for the whole outage, and each restart re-listed
    and re-posted every HMA-labelled node with an empty dedup table.
    """

    sink = BufferingSink()
    collector = KubernetesHmaNodeCollector(sink, context(), now=lambda: NOW)
    api = _FakeNodeApi(
        [OSError("apiserver unreachable"), [_hma_node("42")], [_hma_node("42")]]
    )
    _patch_kubernetes(monkeypatch, api, [])

    with pytest.raises(StopTheLoop):
        collector.run()

    assert api.calls == 4, "the run loop exited instead of relisting after a failure"
    assert len(sink.requests) == 1, (
        "the node the outbox already took was posted again on the next relist"
    )


class _HmaNodeRejectingSink:
    """The control plane rejects one node's record and accepts every other."""

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.requests.append((path, payload))
        if payload["node"]["metadata"]["name"] == self.node_id:
            raise CollectorError("rejected (422)", status_code=422)
        return {"accepted": True}


def test_kubernetes_hma_rejected_node_does_not_skip_the_rest_of_the_fleet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One undeliverable node must not cost the fleet its HMA coverage.

    Neither the LIST loop nor the watch loop guarded ``collect_node``, so a
    poison record (a non-retryable 4xx, or any failure when the collector has no
    outbox) unwound into the relist guard: every node after it in LIST order went
    unposted, the watch was never reached, and the loop relisted every two
    seconds stuck on the same node -- a quiet partial outage where the old code
    at least CrashLooped visibly.
    """

    sink = _HmaNodeRejectingSink("gpu-worker-2")
    collector = KubernetesHmaNodeCollector(sink, context(), now=lambda: NOW)
    api = _FakeNodeApi(
        [
            [
                _hma_node("42", "gpu-worker-1"),
                _hma_node("42", "gpu-worker-2"),
                _hma_node("42", "gpu-worker-3"),
            ]
        ]
    )
    watch = _FakeNodeWatch(
        [{"type": "MODIFIED", "object": _hma_node("43", "gpu-worker-4")}]
    )
    _patch_kubernetes(monkeypatch, api, [watch])

    with pytest.raises(StopTheLoop):
        collector.run()

    assert [
        payload["node"]["metadata"]["name"] for _path, payload in sink.requests
    ] == ["gpu-worker-1", "gpu-worker-2", "gpu-worker-3", "gpu-worker-4"], (
        "the rejected node ended the LIST pass and the watch was never reached"
    )
    assert watch.stop_calls == 1, "the watch was not stopped before relisting"


def test_kubernetes_collector_ignores_deleted_node_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A DELETED watch event is not a node record, and it clears the dedup entry.

    Every watch event was forwarded, so deleting a node published a provider
    event for a node that no longer exists (its deletion bumps the
    ``resourceVersion``, so the dedup check did not stop it) and left the node's
    entry in the resource-version table forever.
    """

    sink = RecordingSink()
    collector = KubernetesHmaNodeCollector(sink, context(), now=lambda: NOW)
    api = _FakeNodeApi([[_hma_node("42")], [_hma_node("42")]])
    watch = _FakeNodeWatch([{"type": "DELETED", "object": _hma_node("43")}])
    _patch_kubernetes(monkeypatch, api, [watch])

    with pytest.raises(StopTheLoop):
        collector.run()

    assert [
        payload["node"]["metadata"]["resourceVersion"]
        for _path, payload in sink.requests
    ] == ["42", "42"], "a DELETED node event was published as a live node record"
    assert watch.stop_calls == 1, "the watch was not stopped before relisting"


def test_cloudwatch_subscription_continues_after_a_buffered_event() -> None:
    """A buffered HMA event must not abort the rest of the subscription batch."""

    sink = BufferingSink()
    collector = CloudWatchHmaCollector(sink, context(), now=lambda: NOW)
    message = json.dumps({"HealthMonitoringAgentDetectionEvent": "HealthEvent"})

    stats = collector.collect_subscription(
        cloudwatch_envelope(
            [
                {"id": "event-1", "timestamp": 1784548800000, "message": message},
                {"id": "event-2", "timestamp": 1784548801000, "message": message},
            ]
        )
    )

    assert [payload["log_event_id"] for _, payload in sink.requests] == [
        "event-1",
        "event-2",
    ], "the first buffered event stopped the subscription batch"
    assert stats.observed == 2, "both log events must be observed"
    assert stats.delivered == 2, (
        "an event the outbox owns must count as delivered, like every other collector"
    )


class _SqsClient:
    """One scripted HMA message, then an empty queue once it is deleted."""

    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.receives = 0

    def receive_message(self, **_kwargs):
        self.receives += 1
        if self.deleted or self.receives > 1:
            return {}
        return {
            "Messages": [
                {
                    "MessageId": "message-1",
                    "Body": json.dumps(
                        {
                            "path": "/v1/provider-events/hyperpod-hma/cloudwatch",
                            "payload": {"node_id": "worker-1"},
                        }
                    ),
                    "ReceiptHandle": "receipt-1",
                }
            ]
        }

    def delete_message(self, **kwargs):
        self.deleted.append(kwargs["ReceiptHandle"])


def test_sqs_consumer_keeps_a_buffered_message_on_the_queue() -> None:
    """The queue outlives the outbox, so a buffered forward keeps the message.

    SQS retains the message for 14 days; the collector outbox is an emptyDir the
    optional HMA manifests do not even configure, so deleting the message on a
    buffered forward would drop the event with the Pod. The control plane dedupes
    by CloudWatch log event id, so the redelivery costs nothing.
    """

    client = _SqsClient()
    consumer = SqsHmaConsumer(BufferingSink(), "https://sqs/queue", client=client)

    delivered = consumer.run_once(wait_time_seconds=0)

    assert client.deleted == [], (
        "a buffered forward deleted the message, leaving the emptyDir outbox "
        "as the only copy of the HMA event"
    )
    assert delivered == 1, "the forward the outbox owns was not counted as handled"


def test_sqs_consumer_deletes_a_delivered_message() -> None:
    """A forward the control plane accepted must leave the queue."""

    client = _SqsClient()
    consumer = SqsHmaConsumer(RecordingSink(), "https://sqs/queue", client=client)

    delivered = consumer.run_once(wait_time_seconds=0)

    assert client.deleted == ["receipt-1"], (
        "an accepted forward left the message for SQS to redeliver"
    )
    assert delivered == 1, "the accepted forward was not counted as handled"
