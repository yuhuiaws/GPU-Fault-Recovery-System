from __future__ import annotations

from ._support import (
    NOW,
    Any,
    BufferingSink,
    CollectorError,
    KubernetesNodeResourceCollector,
    RecordingSink,
    SimpleNamespace,
    StopTheLoop,
    context,
    pytest,
    timedelta,
)


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
