from __future__ import annotations

import hashlib
import logging

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
    RejectingSink,
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
    assert payload["producer"] == "control-plane", (
        "the Kubernetes reader must not pose as the node's host collector: its "
        "batches used to overwrite the node's HOST_TELEMETRY status row"
    )


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
    """One ``watch.Watch()`` replaying a scripted event list, then ending.

    ``error`` is raised once the scripted events run out, which is how a resumed
    watch whose ``resourceVersion`` left the apiserver watch cache ends: an
    ``ApiException`` with status 410.
    """

    def __init__(
        self, events: list[dict[str, Any]], *, error: BaseException | None = None
    ) -> None:
        self.events = list(events)
        self.error = error
        self.stop_calls = 0
        self.stream_kwargs: list[dict[str, Any]] = []

    def stream(self, *_args, **kwargs):
        self.stream_kwargs.append(dict(kwargs))
        yield from self.events
        if self.error is not None:
            raise self.error

    def stop(self) -> None:
        self.stop_calls += 1


def _expired_watch_error() -> BaseException:
    """What the apiserver answers once a resumed resourceVersion is too old."""

    from kubernetes.client.rest import ApiException

    return ApiException(status=410, reason="Expired: too old resource version")


class _FakeNodeApi:
    """A ``CoreV1Api`` whose LIST is scripted round by round."""

    def __init__(self, rounds: list[Any]) -> None:
        self.rounds = list(rounds)
        self.calls = 0
        self.list_kwargs: list[dict[str, Any]] = []

    def list_node(self, **kwargs):
        self.calls += 1
        self.list_kwargs.append(dict(kwargs))
        if not self.rounds:
            raise StopTheLoop
        outcome = self.rounds.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(
            items=list(outcome),
            metadata=SimpleNamespace(resource_version="7", _continue=None),
        )


def _hma_node(
    resource_version: str, name: str = "gpu-worker", health: str = "Unschedulable"
) -> dict[str, Any]:
    return {
        "metadata": {
            "name": name,
            "resourceVersion": resource_version,
            "labels": {HMA_HEALTH_STATUS: health},
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
        "kubernetes.watch.Watch",
        lambda: (
            queue.pop(0)
            if queue
            # An unscripted watch ends the way a long-lived one really does:
            # 410 Gone, which is the collector's only cue to relist.
            else _FakeNodeWatch([], error=_expired_watch_error())
        ),
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
    assert stats.delivered == 0, (
        "`delivered` means accepted by the control plane; an event the outbox "
        f"owns is BUFFERED, not DELIVERED: {stats}"
    )
    assert stats.buffered == 2, (
        f"the events the outbox took were not counted as buffered: {stats}"
    )


def test_cloudwatch_subscription_counts_accepted_events_as_delivered() -> None:
    """``delivered`` = accepted by the control plane, ``buffered`` = 0 on a live path."""

    sink = RecordingSink()
    collector = CloudWatchHmaCollector(sink, context(), now=lambda: NOW)
    message = json.dumps({"HealthMonitoringAgentDetectionEvent": "HealthEvent"})

    stats = collector.collect_subscription(
        cloudwatch_envelope(
            [
                {"id": "event-1", "timestamp": 1784548800000, "message": message},
                {"id": "event-2", "timestamp": 1784548801000, "message": "noise"},
            ]
        )
    )

    assert (stats.observed, stats.delivered, stats.buffered, stats.skipped) == (
        2,
        1,
        0,
        1,
    ), f"the subscription stats do not split accepted/buffered/skipped: {stats}"


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
    assert (delivered.observed, delivered.delivered, delivered.buffered) == (1, 0, 1), (
        "`delivered` means accepted by the control plane; the forward the outbox "
        f"owns is buffered, and the message stays on the queue: {delivered}"
    )


def test_sqs_consumer_deletes_a_delivered_message() -> None:
    """A forward the control plane accepted must leave the queue."""

    client = _SqsClient()
    consumer = SqsHmaConsumer(RecordingSink(), "https://sqs/queue", client=client)

    delivered = consumer.run_once(wait_time_seconds=0)

    assert client.deleted == ["receipt-1"], (
        "an accepted forward left the message for SQS to redeliver"
    )
    assert (delivered.observed, delivered.delivered, delivered.buffered) == (1, 1, 0), (
        f"the accepted forward was not counted as delivered: {delivered}"
    )


def _hma_content_node(
    resource_version: str,
    *,
    health: str = "Unschedulable",
    detail: str = "NVRM: Xid (PCI:0000:b9:00): 94",
    taint_value: str | None = "Unschedulable",
    ready: str = "True",
    annotate_faults: bool = True,
    noise: str = "{}",
    image_size: int = 12345,
) -> dict[str, Any]:
    """One HMA-labelled Node, every field the control plane reads made variable.

    ``noise`` and ``image_size`` are the fields it does **not** read: they change
    on every kubelet status report and must never make the collector re-post.
    """

    annotations: dict[str, Any] = {
        "kubectl.kubernetes.io/last-applied-configuration": noise
    }
    if annotate_faults:
        annotations[HMA_FAULT_DETAILS] = json.dumps(
            {"faults": [{"timestamp": NOW.isoformat(), "message": detail}]}
        )
    return {
        "metadata": {
            "name": "gpu-worker",
            "resourceVersion": resource_version,
            "labels": {
                HMA_HEALTH_STATUS: health,
                HMA_FAULT_REASONS: "XidHardwareFailure",
                "kubernetes.io/hostname": "gpu-worker",
            },
            "annotations": annotations,
        },
        "spec": {
            "taints": (
                [
                    {
                        "key": HMA_HEALTH_STATUS,
                        "value": taint_value,
                        "effect": "NoSchedule",
                    }
                ]
                if taint_value is not None
                else []
            )
        },
        "status": {
            "allocatable": {"nvidia.com/gpu": "8"},
            "conditions": [
                {
                    "type": "Ready",
                    "status": ready,
                    "reason": "KubeletReady",
                    "message": "kubelet is posting ready status",
                    "lastTransitionTime": NOW.isoformat(),
                }
            ],
            "images": [{"names": ["nginx:latest"], "sizeBytes": image_size}],
        },
    }


def test_kubernetes_collector_skips_unchanged_hma_content_across_resource_versions() -> (
    None
):
    """Dedupe on the HMA content, not on ``resourceVersion``.

    HMA labels every node (``Schedulable`` on a healthy one) and each kubelet
    status report bumps ``resourceVersion`` every 5 minutes, so a
    ``resourceVersion`` dedupe re-posted every node's whole object -- the
    ``status.images`` list included -- forever. The hash must still change for
    every field the control plane reads, or a real health change is deduped away.
    """

    sink = RecordingSink()
    collector = KubernetesHmaNodeCollector(sink, context(), now=lambda: NOW)

    assert collector.collect_node(_hma_content_node("1")).delivered == 1, (
        "the first HMA node record was not delivered"
    )
    assert collector.collect_node(_hma_content_node("2")).duplicates == 1, (
        "a node whose only change was its resourceVersion was posted again"
    )
    assert (
        collector.collect_node(
            _hma_content_node("3", noise='{"spec":"changed"}', image_size=99)
        ).duplicates
        == 1
    ), (
        "a change outside the HMA keys (an unrelated annotation, a pulled image) "
        "was treated as an HMA content change"
    )
    assert (
        collector.collect_node(
            _hma_content_node("4", detail="NVRM: Xid (PCI:0000:b9:00): 79")
        ).delivered
        == 1
    ), "a new fault detail was deduplicated away"
    assert (
        collector.collect_node(_hma_content_node("5", health="Schedulable")).delivered
        == 1
    ), "a health-status label change was deduplicated away"
    assert (
        collector.collect_node(_hma_content_node("6", taint_value=None)).delivered == 1
    ), "the HMA taint leaving the node was deduplicated away"
    assert (
        collector.collect_node(_hma_content_node("7", ready="False")).delivered == 1
    ), "a node condition flipping was deduplicated away"
    assert (
        collector.collect_node(_hma_content_node("8", annotate_faults=False)).delivered
        == 1
    ), (
        "the fault-details annotation being removed -- the node healed -- was "
        "deduplicated away"
    )

    source = _hma_content_node("9", detail="NVRM: Xid (PCI:0000:b9:00): 48")
    assert collector.collect_node(source).delivered == 1, (
        "the last HMA node record was not delivered"
    )
    posted = sink.requests[-1][1]["node"]
    assert "images" not in posted["status"], (
        "status.images was posted: megabytes of image digests per node per post"
    )
    assert posted["status"]["allocatable"] == {"nvidia.com/gpu": "8"}, (
        "stripping status.images dropped the rest of the node status"
    )
    assert posted["status"]["conditions"], "stripping status.images dropped conditions"
    assert "images" in source["status"], (
        "the collector mutated the node object it was handed"
    )


def test_hma_watcher_resumes_from_resource_version_and_relists_on_410(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A watch timeout resumes; only 410 Gone relists, and it does not re-post.

    The 300 s watch timeout used to force a full LIST of every node plus a
    ``sanitize_for_serialization`` of each one every five minutes. The watcher
    must resume the watch from the last observed ``resourceVersion`` and relist
    only when the apiserver says that version is gone -- and the relist that
    follows must not re-post nodes whose HMA content has not changed.
    """

    sink = RecordingSink()
    collector = KubernetesHmaNodeCollector(sink, context(), now=lambda: NOW)
    api = _FakeNodeApi([[_hma_node("42")], [_hma_node("44", health="Schedulable")]])
    first = _FakeNodeWatch(
        [{"type": "MODIFIED", "object": _hma_node("43", health="Schedulable")}]
    )
    resumed = _FakeNodeWatch([], error=_expired_watch_error())
    _patch_kubernetes(monkeypatch, api, [first, resumed])

    with pytest.raises(StopTheLoop):
        collector.run()

    assert first.stream_kwargs[0]["resource_version"] == "7", (
        "the watch did not start from the LIST's resourceVersion"
    )
    assert resumed.stream_kwargs[0]["resource_version"] == "43", (
        "a plain watch timeout relisted instead of resuming from the last "
        "observed resourceVersion"
    )
    assert api.calls == 3, (
        "the watcher relisted on a watch timeout instead of only on 410 Gone"
    )
    assert [
        payload["node"]["metadata"]["resourceVersion"]
        for _path, payload in sink.requests
    ] == ["42", "43"], (
        "the relist after the 410 re-posted a node whose HMA content was unchanged"
    )


def test_hma_watcher_paginates_the_relist(monkeypatch: pytest.MonkeyPatch) -> None:
    """The relist follows ``metadata._continue`` like the sibling collector.

    One unpaginated LIST of a 500-node cluster is 5-10 MB and, past the watch
    cache window, an etcd read; the watch that follows must resume from the
    first page's resourceVersion, which is the snapshot the pages were read at.
    """

    class _PagedNodeApi:
        def __init__(self) -> None:
            self.list_kwargs: list[dict[str, Any]] = []

        def list_node(self, **kwargs: Any) -> Any:
            self.list_kwargs.append(dict(kwargs))
            if len(self.list_kwargs) > 2:
                raise StopTheLoop
            if kwargs.get("_continue"):
                return SimpleNamespace(
                    items=[_hma_node("44", "gpu-worker-b")],
                    metadata=SimpleNamespace(resource_version="9", _continue=None),
                )
            return SimpleNamespace(
                items=[_hma_node("43", "gpu-worker-a")],
                metadata=SimpleNamespace(resource_version="7", _continue="page-two"),
            )

    api = _PagedNodeApi()
    sink = RecordingSink()
    collector = KubernetesHmaNodeCollector(
        sink, context(), now=lambda: NOW, list_page_size=1
    )
    watch = _FakeNodeWatch([], error=_expired_watch_error())
    _patch_kubernetes(monkeypatch, api, [watch])

    with pytest.raises(StopTheLoop):
        collector.run()

    assert api.list_kwargs[:2] == [
        {"limit": 1, "_continue": None},
        {"limit": 1, "_continue": "page-two"},
    ], "the HMA relist asked the apiserver for every node in a single call"
    assert [
        payload["node"]["metadata"]["name"] for _path, payload in sink.requests
    ] == ["gpu-worker-a", "gpu-worker-b"], "the second page of the relist was dropped"
    assert watch.stream_kwargs[0]["resource_version"] == "7", (
        "the watch did not resume from the first page's resourceVersion snapshot"
    )


class _FlakySqsClient:
    """``receive_message`` fails twice, hands over one HMA message, fails again."""

    def __init__(self) -> None:
        self.calls = 0
        self.deleted: list[str] = []

    def receive_message(self, **_kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        # Two failures, one message, then a failure again: the last one is what
        # shows whether the backoff was reset by the successful receive.
        if self.calls in {1, 2, 4}:
            raise RuntimeError("ExpiredTokenException")
        if self.calls > 4:
            raise StopTheLoop
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
                    "Attributes": {"ApproximateReceiveCount": "1"},
                }
            ]
        }

    def delete_message(self, **kwargs: Any) -> None:
        self.deleted.append(kwargs["ReceiptHandle"])


def test_sqs_consumer_survives_receive_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A throttle or an expired IRSA token must not end the consumer.

    ``run()`` was a bare ``while True: self.run_once()``, so the first
    ``receive_message`` exception exited the process and left the HMA queue
    unattended until Kubernetes restarted the Pod -- and restarted it again on
    the next one.
    """

    from gpu_fault.collectors.cloud import cloudwatch as module

    sleeps: list[float] = []
    monkeypatch.setattr(module.time, "sleep", lambda seconds: sleeps.append(seconds))
    client = _FlakySqsClient()
    consumer = SqsHmaConsumer(RecordingSink(), "https://sqs/queue", client=client)

    with pytest.raises(StopTheLoop):
        consumer.run()

    assert client.deleted == ["receipt-1"], (
        "a receive_message error ended the consumer instead of backing off"
    )
    assert sleeps == [1.0, 2.0, 1.0], (
        "the receive backoff did not start at one second, double, and reset "
        "once a receive succeeded"
    )


class _PoisonSqsClient:
    """One message the queue has already redelivered ``receives`` times."""

    def __init__(self, body: str, *, receives: str = "5") -> None:
        self.body = body
        self.receives = receives
        self.deleted: list[str] = []
        self.receive_kwargs: list[dict[str, Any]] = []

    def receive_message(self, **kwargs: Any) -> dict[str, Any]:
        self.receive_kwargs.append(dict(kwargs))
        if len(self.receive_kwargs) > 1:
            return {}
        return {
            "Messages": [
                {
                    "MessageId": "message-9",
                    "Body": self.body,
                    "ReceiptHandle": "receipt-9",
                    "Attributes": {"ApproximateReceiveCount": self.receives},
                }
            ]
        }

    def delete_message(self, **kwargs: Any) -> None:
        self.deleted.append(kwargs["ReceiptHandle"])


def test_sqs_poison_message_is_dropped_after_five_receives(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A message nothing can forward leaves the queue, digest-logged, not echoed.

    There is no redrive policy on this queue, so a malformed body was retried
    every 60 s visibility timeout forever, and each retry logged a full
    traceback. The drop must name the MessageId and the body digest -- never the
    body, which carries HMA fault text.
    """

    body = json.dumps(
        {
            "path": "/v1/provider-events/hyperpod-hma/cloudwatch",
            "payload": "worker-1-fault-text",
        }
    )
    client = _PoisonSqsClient(body)
    sink = RecordingSink()
    consumer = SqsHmaConsumer(sink, "https://sqs/queue", client=client)

    with caplog.at_level(logging.WARNING):
        delivered = consumer.run_once(wait_time_seconds=0)

    assert delivered.delivered == 0, (
        f"a dropped poison message was counted as delivered: {delivered}"
    )
    assert (delivered.observed, delivered.skipped) == (1, 1), (
        f"a dropped poison message must be observed and counted as skipped: {delivered}"
    )
    assert sink.requests == [], "the poison message was forwarded a sixth time"
    assert client.deleted == ["receipt-9"], (
        "the poison message stayed on the queue to be retried every visibility "
        "timeout forever"
    )
    assert client.receive_kwargs[0].get("MessageSystemAttributeNames") == [
        "ApproximateReceiveCount"
    ], "receive_message did not ask for the receive count it drops on"
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "message-9" in logged, "the drop did not name the MessageId"
    assert hashlib.sha256(body.encode()).hexdigest() in logged, (
        "the drop did not record the body digest"
    )
    assert "worker-1-fault-text" not in logged, (
        "the message body was logged instead of its digest"
    )


def _forward_one_queued_event(sink: Any, *, receives: str) -> _PoisonSqsClient:
    """Forward one well-formed queued event whose delivery the sink decides."""

    client = _PoisonSqsClient(
        json.dumps(
            {
                "path": "/v1/provider-events/hyperpod-hma/cloudwatch",
                "payload": {"node_id": "worker-1"},
            }
        ),
        receives=receives,
    )
    SqsHmaConsumer(sink, "https://sqs/queue", client=client).run_once(
        wait_time_seconds=0
    )
    return client


def test_sqs_retryable_failure_is_never_dropped_regardless_of_receive_count() -> None:
    """Only the message can be poison; the transport being down never is.

    ``VisibilityTimeout`` is 60 s, so a five-minute control-plane outage takes
    every message past five receives. Dropping on the receive count alone
    therefore deleted the whole backlog mid-outage -- and the optional consumer
    manifest configures no outbox, so a transport failure there is FAILED, not
    BUFFERED: the queue's 14-day retention was the only copy of those events.
    """

    kept = _forward_one_queued_event(RejectingSink(status_code=503), receives="9")
    assert kept.deleted == [], (
        "a retryable 503 was treated as poison and the event was deleted "
        "instead of left for redelivery"
    )
    auth = _forward_one_queued_event(RejectingSink(status_code=401), receives="9")
    assert auth.deleted == [], (
        "a token-rotation 401 was treated as poison; the event behind it is real"
    )
    rejected = _forward_one_queued_event(RejectingSink(status_code=422), receives="9")
    assert rejected.deleted == ["receipt-9"], (
        "a rejection the control plane will repeat kept the message on the "
        "queue forever"
    )


def test_hma_relist_backs_off_when_a_continue_token_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 410 must never spin the relist, and a continue token can expire.

    The 410 branch was the only unthrottled path in the loop. etcd compaction
    during a large paginated LIST expires the continue token, so page two 410s
    while page one is fine -- "relist page 1, 410 on page 2, relist page 1" is a
    tight loop of multi-MB LISTs plus a full sanitize against the apiserver.
    """

    class _CompactingNodeApi:
        def __init__(self) -> None:
            self.list_kwargs: list[dict[str, Any]] = []
            self.expired_pages = 0

        def list_node(self, **kwargs: Any) -> Any:
            self.list_kwargs.append(dict(kwargs))
            if len(self.list_kwargs) > 6:
                raise StopTheLoop
            if kwargs.get("_continue"):
                if self.expired_pages < 2:
                    self.expired_pages += 1
                    raise _expired_watch_error()
                return SimpleNamespace(
                    items=[_hma_node("44", "gpu-worker-b")],
                    metadata=SimpleNamespace(resource_version="9", _continue=None),
                )
            return SimpleNamespace(
                items=[_hma_node("43", "gpu-worker-a")],
                metadata=SimpleNamespace(resource_version="7", _continue="page-two"),
            )

    from gpu_fault.collectors.cloud import kubernetes as module

    api = _CompactingNodeApi()
    collector = KubernetesHmaNodeCollector(
        RecordingSink(), context(), now=lambda: NOW, list_page_size=1
    )
    _patch_kubernetes(monkeypatch, api, [])
    sleeps: list[float] = []
    monkeypatch.setattr(module.time, "sleep", lambda seconds: sleeps.append(seconds))

    with pytest.raises(StopTheLoop):
        collector.run()

    assert sleeps == [2, 2, 2], (
        "a 410 relisted with no backoff at all: two expired continue tokens and "
        "one expired watch each restarted a full LIST immediately"
    )


def test_kubernetes_collector_forgets_a_node_that_loses_every_hma_key() -> None:
    """A node with no HMA keys left must not keep its digest.

    HMA labels every node, so the skip branch is normally a non-HMA node -- but
    if every HMA key is removed and the same fault later returns verbatim, the
    stale digest would deduplicate the new occurrence away and the control plane
    would never hear about it.
    """

    sink = RecordingSink()
    collector = KubernetesHmaNodeCollector(sink, context(), now=lambda: NOW)
    faulted = _hma_content_node("1")
    bare = {
        "metadata": {"name": "gpu-worker", "resourceVersion": "2", "labels": {}},
        "status": {},
    }

    assert collector.collect_node(faulted).delivered == 1, (
        "the first HMA node record was not delivered"
    )
    assert collector.collect_node(bare).skipped == 1, (
        "a node with no HMA keys must be skipped, not posted"
    )
    assert collector.collect_node(_hma_content_node("3")).delivered == 1, (
        "the same fault returning after every HMA key was removed was "
        "deduplicated against the digest of the first occurrence"
    )


def test_node_resource_collector_refreshes_its_liveness_heartbeat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """``run`` had no health surface at all, so a wedged loop looked Ready.

    ``list_node`` passes no request timeout, so a half-open apiserver
    connection blocks the cycle forever while the Pod stays Running and Ready
    and the whole cluster's EFA/GPU advertisement coverage goes quiet. The
    Deployment's livenessProbe reads the age of this file, so it has to be
    refreshed on every cycle -- including a cycle that failed, because liveness
    answers "is the loop turning", not "did the sample succeed".
    """

    from gpu_fault.collectors.cloud import kubernetes as module

    heartbeat = tmp_path / "node-resource-collector-alive"
    monkeypatch.setattr(module, "NODE_RESOURCE_HEARTBEAT_PATH", str(heartbeat))

    class _FailingCore:
        def list_node(self, **_kwargs: Any) -> Any:
            raise OSError("apiserver unreachable")

    for core in (_two_node_core(), _FailingCore()):
        heartbeat.unlink(missing_ok=True)
        collector = KubernetesNodeResourceCollector(
            RecordingSink(), context(), core_api=core, now=lambda: NOW
        )
        monkeypatch.setattr(
            module.time, "sleep", lambda _seconds: (_ for _ in ()).throw(StopTheLoop())
        )

        with pytest.raises(StopTheLoop):
            collector.run()

        assert heartbeat.exists(), (
            f"the cycle using {type(core).__name__} left no liveness heartbeat"
        )


def test_kubernetes_hma_collector_uses_shared_buffered_warning(caplog) -> None:
    """Kubernetes HMA collector BUFFERED warning must use the shared text from deliver_or_raise."""

    sink = BufferingSink()
    collector = KubernetesHmaNodeCollector(sink, context(), now=lambda: NOW)
    node = {
        "metadata": {
            "name": "worker-1",
            "resourceVersion": "12345",
            "labels": {HMA_HEALTH_STATUS: "Schedulable"},
        },
        "status": {},
    }

    with caplog.at_level(logging.WARNING):
        collector.collect_node(node)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, "buffered delivery must log exactly one warning"
    message = warnings[0].getMessage()
    assert "persisted to the collector outbox" in message, (
        "warning must use the shared text from deliver_or_raise"
    )
    assert "id=" in message or "node/" in message, (
        "warning must include the event id or node identifier"
    )


def test_node_resource_collector_heartbeats_before_its_first_cycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The first cycle must not have to finish for the Pod to look alive.

    One cycle posts a ``baseline:*`` host-telemetry batch for every GPU node in
    series, each its own TCP+TLS connection with a four-attempt retry ladder, so
    on a large fleet the first cycle can outlast the livenessProbe's threshold.
    A probe that fired then would kill the Pod, and the restart would begin the
    same first cycle again: CrashLoopBackOff, with the whole cluster's EFA/GPU
    advertisement coverage silent. So the heartbeat exists before the loop is
    entered, and the first cycle gets the full threshold.
    """

    from gpu_fault.collectors.cloud import kubernetes as module

    heartbeat = tmp_path / "node-resource-collector-alive"
    monkeypatch.setattr(module, "NODE_RESOURCE_HEARTBEAT_PATH", str(heartbeat))

    class _BlockingCore:
        """A cycle that never returns: the very first apiserver read hangs."""

        def list_pod_for_all_namespaces(self, **_kwargs: Any) -> Any:
            raise StopTheLoop()

        def list_node(self, **_kwargs: Any) -> Any:  # pragma: no cover - unreached
            raise AssertionError("the cycle should not have got as far as list_node")

    collector = KubernetesNodeResourceCollector(
        RecordingSink(), context(), core_api=_BlockingCore(), now=lambda: NOW
    )

    with pytest.raises(StopTheLoop):
        collector.run()

    assert heartbeat.exists(), (
        "a cycle that has not returned yet leaves no heartbeat, so liveness "
        "kills the Pod before its first sample can ever finish"
    )


def test_node_resource_collector_heartbeats_after_every_node(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """A long cycle keeps proving itself; a wedged LIST still cannot.

    The heartbeat moves as each node completes, so a cycle that takes longer
    than the liveness threshold is never mistaken for a wedge. The distinction
    survives because a half-open apiserver connection completes no node at all:
    here the second page of the LIST hangs, and only the first page's node has
    been sampled when the heartbeat is checked.
    """

    from gpu_fault.collectors.cloud import kubernetes as module

    heartbeat = tmp_path / "node-resource-collector-alive"
    monkeypatch.setattr(module, "NODE_RESOURCE_HEARTBEAT_PATH", str(heartbeat))

    class _WedgedSecondPage:
        def __init__(self) -> None:
            self.pages = 0

        def list_node(self, **_kwargs: Any) -> Any:
            self.pages += 1
            if self.pages > 1:
                raise StopTheLoop()
            return SimpleNamespace(
                metadata=SimpleNamespace(_continue="page-2"),
                items=[_paged_node("gpu-worker-a")],
            )

        def list_pod_for_all_namespaces(self, **_kwargs: Any) -> Any:
            return SimpleNamespace(items=[])

    sink = RecordingSink()
    collector = KubernetesNodeResourceCollector(
        sink, context(), core_api=_WedgedSecondPage(), now=lambda: NOW
    )

    with pytest.raises(StopTheLoop):
        collector.collect_once()

    assert heartbeat.exists(), (
        "the heartbeat only moved when the whole cycle finished, so a fleet "
        "whose cycle outlasts the probe threshold restarts forever"
    )
    assert [payload["node_id"] for _, payload in sink.requests] == ["gpu-worker-a"], (
        "the heartbeat must follow a node that was actually sampled"
    )
