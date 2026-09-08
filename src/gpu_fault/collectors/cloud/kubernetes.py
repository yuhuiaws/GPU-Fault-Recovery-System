from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Iterator

from gpu_fault.channel_registry import HOST_TELEMETRY_PATH
from gpu_fault.models import WorkloadState
from gpu_fault.host_health import (
    HostMetricSample,
    HostTelemetryBatch,
)


from gpu_fault.collectors.cloud.cloudwatch import HMA_KEYS
from gpu_fault.collectors.gpu.discovery import (
    INSTANCE_ACCELERATOR_COUNTS,
)
from gpu_fault.collectors.models import CollectorContext, CollectorStats
from gpu_fault.collectors.scheduling import next_stable_phase
from gpu_fault.collectors.sinks import (
    CollectorError,
    EventSink,
    deliver_event,
)

LOGGER = logging.getLogger(__name__)

#: What the apiserver answers once the requested ``resourceVersion`` has left
#: its watch cache. It is the only response that makes a relist necessary.
EXPIRED_RESOURCE_VERSION_STATUS = 410


def _iter_node_pages(list_node: Callable[..., Any], page_size: int) -> Iterator[Any]:
    """Yield each page of a Node LIST, following ``metadata._continue``.

    One unpaginated LIST of a 500-node cluster is 5-10 MB and, once the
    requested resourceVersion has fallen out of the apiserver watch cache, an
    etcd read. Callers that also need the LIST's ``resourceVersion`` read it
    from the first page: the continue token pins every later page to that same
    snapshot.
    """

    token: str | None = None
    while True:
        response = list_node(limit=page_size, _continue=token)
        yield response
        metadata = getattr(response, "metadata", None)
        token = getattr(metadata, "_continue", None) or None
        if not token:
            return


def _digest_text(value: Any) -> str:
    """One digest field as text: a missing value and an empty one are the same.

    Every field is a string so the sorted lists below never compare ``None``
    with ``str`` (a ``TypeError`` that would have taken the whole watch down).
    """

    return "" if value is None else str(value)


def _hma_content_digest(node: dict[str, Any]) -> str:
    """Digest exactly the Node fields the control plane reads for HMA.

    HMA labels every node -- a healthy one carries ``Schedulable`` -- and every
    kubelet status report bumps ``resourceVersion``, so deduplicating on
    ``resourceVersion`` re-posted every HMA node's whole object every few
    minutes. The digest therefore covers the HMA labels/annotations, the HMA
    taint and the node's true conditions, which is everything
    ``HyperPodHmaNormalizer.normalize_kubernetes_node`` turns into a snapshot.
    Anything left out (``status.images``, unrelated annotations) cannot change a
    verdict; anything covered must re-post, including a key that disappears
    because the node healed.
    """

    metadata = node.get("metadata") or {}
    labels = metadata.get("labels") or {}
    annotations = metadata.get("annotations") or {}
    spec = node.get("spec") or {}
    status = node.get("status") or {}
    content = {
        "labels": {key: labels[key] for key in sorted(HMA_KEYS & set(labels))},
        "annotations": {
            key: annotations[key] for key in sorted(HMA_KEYS & set(annotations))
        },
        "taints": sorted(
            [
                _digest_text(taint.get("key")),
                _digest_text(taint.get("value")),
                _digest_text(taint.get("effect")),
            ]
            for taint in spec.get("taints") or []
            if taint.get("key") in HMA_KEYS
        ),
        # A condition that stops being true drops out of the list, so the
        # digest changes when a node recovers as well as when it degrades.
        "conditions": sorted(
            [
                _digest_text(condition.get("type")),
                _digest_text(condition.get("reason")),
                _digest_text(condition.get("message")),
                _digest_text(condition.get("lastTransitionTime")),
            ]
            for condition in status.get("conditions") or []
            if str(condition.get("status") or "").lower() == "true"
        ),
    }
    payload = json.dumps(content, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _node_without_images(node: dict[str, Any]) -> dict[str, Any]:
    """Return the node without ``status.images``, leaving the original alone.

    ``status.images`` is a list of every image on the kubelet with its digests
    and sizes -- easily most of the object, and nothing the HMA path reads. The
    copy is shallow on purpose: the caller still owns the node it handed over.
    """

    status = node.get("status")
    if not isinstance(status, dict) or "images" not in status:
        return node
    return {
        **node,
        "status": {key: value for key, value in status.items() if key != "images"},
    }


class KubernetesHmaNodeCollector:
    def __init__(
        self,
        sink: EventSink,
        context: CollectorContext,
        *,
        list_page_size: int = 500,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.sink = sink
        self.context = context
        self.list_page_size = list_page_size
        self.now = now or (lambda: datetime.now(timezone.utc))
        self._content_digests: dict[str, str] = {}

    def collect_node(self, node: dict[str, Any]) -> CollectorStats:
        metadata = node.get("metadata") or {}
        node_id = metadata.get("name")
        if not node_id:
            raise CollectorError("Node metadata.name is required")
        labels = metadata.get("labels") or {}
        annotations = metadata.get("annotations") or {}
        if not HMA_KEYS.intersection({*labels, *annotations}):
            return CollectorStats(observed=1, skipped=1)

        resource_version = str(metadata.get("resourceVersion") or "")
        digest = _hma_content_digest(node)
        if self._content_digests.get(node_id) == digest:
            return CollectorStats(observed=1, duplicates=1)

        collected_at = self.now()
        result = deliver_event(
            self.sink,
            "/v1/provider-events/hyperpod-hma/kubernetes-node",
            {
                **self.context.model_dump(mode="json"),
                "observed_at": collected_at.isoformat(),
                "collected_at": collected_at.isoformat(),
                "node": _node_without_images(node),
                "evidence_ref": (
                    f"k8s://nodes/{node_id}?resourceVersion={resource_version}"
                ),
            },
        )
        # Only a record that went nowhere leaves the digest unrecorded: one the
        # outbox took is replayed from there, and re-posting it on the next
        # relist would buffer a duplicate for the whole outage (ARCH-G3).
        result.raise_for_failure()
        self._content_digests[node_id] = digest
        if result.buffered:
            LOGGER.warning(
                "HMA node record for %s persisted to the collector outbox: %s",
                node_id,
                result.error,
            )
            return CollectorStats(observed=1)
        return CollectorStats(observed=1, delivered=1)

    def forget_node(self, node_id: str) -> None:
        """Drop a node's remembered HMA content digest.

        A deleted Node must not keep an entry: the table would grow with every
        node the fleet ever had, and a recreated node that came back with the
        same HMA content would be silently deduplicated instead of announced.
        """

        self._content_digests.pop(node_id, None)

    def run(self) -> None:
        try:
            from kubernetes import client, config, watch
            from kubernetes.config.config_exception import (
                ConfigException,
            )
        except ImportError as exc:
            raise CollectorError(
                "install gpu-fault-control-plane[collectors] "
                "for Kubernetes watch support"
            ) from exc

        try:
            config.load_incluster_config()
        except ConfigException:
            config.load_kube_config()

        api = client.CoreV1Api()
        serializer = client.ApiClient().sanitize_for_serialization
        # ``None`` is the only state that costs a LIST. The 300 s watch timeout
        # used to relist every node -- and re-serialize every one of them --
        # every five minutes; the watch is resumed from the last observed
        # version instead, and only 410 Gone sends us back to a LIST.
        resource_version: str | None = None
        while True:
            # The LIST and its posts belong inside the guard: `post` raises even
            # once the outbox has taken the record, so an unreachable control
            # plane used to exit run() and leave Kubernetes restarting the Pod
            # into CrashLoopBackOff -- re-listing and re-posting every
            # HMA-labelled node with an empty dedup table on each restart.
            watcher: Any | None = None
            try:
                if resource_version is None:
                    # An empty version -- a LIST that answered without one --
                    # means "watch from now"; it must not stay ``None``, or
                    # every watch timeout would relist again.
                    resource_version = self._relist(api.list_node, serializer) or ""
                watcher = watch.Watch()
                for event in watcher.stream(
                    api.list_node,
                    resource_version=resource_version,
                    timeout_seconds=300,
                ):
                    node = serializer(event["object"])
                    metadata = node.get("metadata") or {}
                    resource_version = (
                        metadata.get("resourceVersion") or resource_version
                    )
                    if event.get("type") == "DELETED":
                        self.forget_node(str(metadata.get("name") or ""))
                        continue
                    self._collect_node_without_ending_the_pass(node)
            except Exception as exc:
                if getattr(exc, "status", None) == EXPIRED_RESOURCE_VERSION_STATUS:
                    # Expected on any watch older than the apiserver cache
                    # window: it is a relist, not a failure, and the content
                    # digests keep the relist from re-posting unchanged nodes.
                    LOGGER.info(
                        "Kubernetes HMA watch resourceVersion %s expired; "
                        "relisting nodes",
                        resource_version,
                    )
                    resource_version = None
                    continue
                LOGGER.exception("Kubernetes HMA watch failed; resuming the watch")
                time.sleep(2)
            finally:
                if watcher is not None:
                    watcher.stop()

    def _relist(
        self,
        list_node: Callable[..., Any],
        serializer: Callable[[Any], dict[str, Any]],
    ) -> str | None:
        """Post every HMA node from a paginated LIST; return its resourceVersion.

        The returned version is the first page's: the continue token pins every
        later page to that snapshot, so it is the version the watch must resume
        from to see everything that happened after the LIST.
        """

        resource_version: str | None = None
        for page in _iter_node_pages(list_node, self.list_page_size):
            if resource_version is None:
                metadata = getattr(page, "metadata", None)
                version = getattr(metadata, "resource_version", None)
                resource_version = str(version) if version else None
            for item in page.items:
                self._collect_node_without_ending_the_pass(serializer(item))
        return resource_version

    def _collect_node_without_ending_the_pass(self, node: dict[str, Any]) -> None:
        """Post one node; a node that cannot be delivered is not fatal.

        A rejected record (a non-retryable 4xx, or any delivery failure when the
        collector has no outbox to fall back on) used to unwind into ``run``'s
        guard: every node after it in LIST order went unposted and the loop
        relisted every two seconds stuck on the same node, so the fleet lost
        HMA coverage quietly instead of CrashLooping visibly.
        """

        try:
            self.collect_node(node)
        except Exception:
            LOGGER.exception(
                "Kubernetes HMA node delivery failed; continuing with the next node"
            )


class KubernetesNodeResourceCollector:
    """Detects Kubernetes device-plugin advertisement loss per node."""

    def __init__(
        self,
        sink: EventSink,
        context: CollectorContext,
        *,
        core_api: Any | None = None,
        interval_seconds: float = 15,
        required_consecutive_samples: int = 2,
        health_summary_seconds: int = 300,
        list_page_size: int = 500,
        resource_names: dict[str, str] | None = None,
        now: Callable[[], datetime] | None = None,
        serializer: Callable[[Any], dict[str, Any]] | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("EFA Kubernetes interval must be positive")
        if required_consecutive_samples < 1:
            raise ValueError("EFA Kubernetes mismatch samples must be positive")
        if health_summary_seconds <= 0:
            raise ValueError("EFA Kubernetes health summary must be positive")
        if list_page_size < 1:
            raise ValueError("EFA Kubernetes node list page size must be positive")
        self.sink = sink
        self.context = context
        self.core = core_api
        self.interval_seconds = interval_seconds
        self.required_consecutive_samples = required_consecutive_samples
        self.health_summary_seconds = health_summary_seconds
        self.list_page_size = list_page_size
        self.resource_names = resource_names or {
            "efa": "vpc.amazonaws.com/efa",
            "gpu": "nvidia.com/gpu",
        }
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.serializer = serializer or (
            lambda value: value if isinstance(value, dict) else value.to_dict()
        )
        self._mismatch_counts: dict[str, int] = {}
        self._last_state: dict[str, bool] = {}
        self._last_delivered_at: dict[str, datetime] = {}
        self._next_summary_at: dict[str, datetime] = {}

    @staticmethod
    def _instance_type(node: dict[str, Any]) -> str | None:
        labels = (node.get("metadata") or {}).get("labels") or {}
        value = labels.get("node.kubernetes.io/instance-type")
        if not value:
            return None
        return str(value).removeprefix("ml.")

    @staticmethod
    def _workload_id(pod: dict[str, Any]) -> str | None:
        metadata = pod.get("metadata") or {}
        namespace = metadata.get("namespace", "default")
        labels = metadata.get("labels") or {}
        annotations = metadata.get("annotations") or {}
        raw = annotations.get("gpu-fault.io/workload-ids")
        if raw:
            try:
                values = json.loads(raw)
            except (TypeError, ValueError):
                values = []
            if isinstance(values, list) and values:
                return str(values[0])
        for key, kind in (
            ("training.kubeflow.org/job-name", "pytorchjob"),
            ("jobset.sigs.k8s.io/jobset-name", "jobset"),
        ):
            if labels.get(key):
                return f"{namespace}/{kind}/{labels[key]}"
        for owner in metadata.get("ownerReferences") or []:
            if owner.get("controller") and owner.get("name"):
                return (
                    f"{namespace}/"
                    f"{str(owner.get('kind', 'job')).lower()}/"
                    f"{owner['name']}"
                )
        return None

    def _managed_workloads(
        self,
    ) -> dict[str, list[str]]:
        if self.core is None:
            return {}
        response = self.core.list_pod_for_all_namespaces(
            label_selector="gpu-fault.io/managed=true"
        )
        workloads: dict[str, list[str]] = {}
        for raw in response.items:
            pod = self.serializer(raw)
            metadata = pod.get("metadata") or {}
            status = pod.get("status") or {}
            spec = pod.get("spec") or {}
            if (
                metadata.get("deletionTimestamp")
                or status.get("phase") in {"Succeeded", "Failed"}
                or not spec.get("nodeName")
            ):
                continue
            workload_id = self._workload_id(pod)
            if workload_id:
                workloads.setdefault(spec["nodeName"], []).append(workload_id)
        return {
            node_id: list(dict.fromkeys(values))
            for node_id, values in workloads.items()
        }

    def _list_nodes(self) -> Iterator[Any]:
        """Yield every Node across a paginated LIST.

        One unpaginated LIST of a 500-node cluster is 5-10 MB every interval
        and, once the requested resourceVersion falls out of the apiserver
        watch cache, an etcd read. The collector samples periodically on
        purpose (its consecutive-mismatch counters need a fresh full sample
        each interval), so it stays a LIST but follows ``metadata._continue``
        in ``list_page_size`` chunks instead of asking for everything at once.
        """
        if self.core is None:
            raise CollectorError(
                "Kubernetes node resource collector has no CoreV1Api client"
            )
        for response in _iter_node_pages(self.core.list_node, self.list_page_size):
            yield from response.items

    def collect_once(self) -> CollectorStats:
        if self.core is None:
            raise CollectorError(
                "Kubernetes node resource collector has no CoreV1Api client"
            )
        observed_at = self.now()
        workloads = self._managed_workloads()
        stats = CollectorStats()
        for raw in self._list_nodes():
            try:
                delta = self._sample_node(raw, observed_at, workloads)
            except Exception:
                # One node must not end the cycle: every node after it in list
                # order would go unsampled, so their consecutive-mismatch
                # counters would freeze for the whole outage.
                LOGGER.exception(
                    "Kubernetes node resource sample failed; "
                    "continuing with the next node"
                )
                delta = CollectorStats(observed=1)
            stats = stats.model_copy(
                update={
                    "observed": stats.observed + delta.observed,
                    "skipped": stats.skipped + delta.skipped,
                    "delivered": stats.delivered + delta.delivered,
                }
            )
        return stats

    def _sample_node(
        self,
        raw: Any,
        observed_at: datetime,
        workloads: dict[str, list[str]],
    ) -> CollectorStats:
        """Sample one node and deliver its batch; the count is this node only."""

        node = self.serializer(raw)
        metadata = node.get("metadata") or {}
        node_id = metadata.get("name")
        instance_type = self._instance_type(node)
        expected_counts = INSTANCE_ACCELERATOR_COUNTS.get(instance_type or "")
        if not node_id or expected_counts is None:
            return CollectorStats(observed=1, skipped=1)
        samples: list[HostMetricSample] = []
        edge_reasons = []
        deliver = False
        delivered_state_keys: list[str] = []
        # The edge state is what makes the next cycle say "unchanged", so it is
        # committed only once the batch reporting it is delivered or buffered.
        # Committing it before the post consumed a real edge on a rejection: the
        # mismatch was then not re-reported until the summary slot, up to 300 s
        # later (ARCH-G3).
        pending_state: dict[str, bool] = {}
        for resource in ("efa", "gpu"):
            resource_name = self.resource_names[resource]
            expected = expected_counts[resource]
            allocatable_raw = (
                (node.get("status") or {}).get("allocatable", {}).get(resource_name, 0)
            )
            try:
                allocatable = int(allocatable_raw)
            except (TypeError, ValueError):
                allocatable = 0
            state_key = f"{node_id}/{resource}"
            mismatch = allocatable < expected
            count = self._mismatch_counts.get(state_key, 0) + 1 if mismatch else 0
            self._mismatch_counts[state_key] = count
            persistent = count >= self.required_consecutive_samples
            previous = self._last_state.get(state_key)
            last_delivered = self._last_delivered_at.get(state_key)
            next_summary = self._next_summary_at.get(state_key)
            resource_deliver = (
                previous is None
                or previous != persistent
                or last_delivered is None
                or (next_summary is not None and observed_at >= next_summary)
                or (
                    next_summary is None
                    and last_delivered is not None
                    and (observed_at - last_delivered).total_seconds()
                    >= self.health_summary_seconds
                )
            )
            pending_state[state_key] = persistent
            deliver = deliver or resource_deliver
            if resource_deliver:
                delivered_state_keys.append(state_key)
                edge_reasons.append(
                    (
                        f"threshold:{resource}_kubernetes_allocatable_mismatch"
                        if persistent
                        else f"baseline:{resource}"
                        if previous is None
                        # A recovery is a transition, so only say `recovered`
                        # when the resource really was persistently missing
                        # before. An unchanged healthy count that delivers
                        # because its summary interval elapsed is a
                        # `health-summary`: labelling it `recovered` claimed
                        # a state change that never happened, and because
                        # the control plane only skips capturing evidence
                        # for batches whose sole reason is `health-summary`,
                        # every periodic delivery for a healthy node was
                        # persisted as raw evidence forever.
                        else f"recovered:{resource}"
                        if previous
                        else "health-summary"
                    )
                )
            labels = {
                "resource": resource.upper(),
                "node_instance_type": instance_type or "UNKNOWN",
                "expected_count": str(expected),
                "observed_count": str(allocatable),
                "missing_count": str(max(0, expected - allocatable)),
                "consecutive_mismatch_samples": str(count),
                "required_consecutive_samples": str(self.required_consecutive_samples),
                "failure_mode": (
                    "KUBERNETES_RESOURCE_MISSING" if persistent else "HEALTHY"
                ),
                "resource_name": resource_name,
            }
            prefix = f"{resource}_kubernetes"
            samples.extend(
                [
                    HostMetricSample(
                        name=f"{prefix}_expected_count",
                        value=expected,
                        unit="devices",
                        labels=labels,
                    ),
                    HostMetricSample(
                        name=f"{prefix}_allocatable_count",
                        value=allocatable,
                        unit="devices",
                        labels=labels,
                    ),
                    HostMetricSample(
                        name=f"{prefix}_allocatable_missing_count",
                        value=max(0, expected - allocatable),
                        unit="devices",
                        labels=labels,
                    ),
                    HostMetricSample(
                        name=f"{prefix}_allocatable_mismatch",
                        value=1 if persistent else 0,
                        labels=labels,
                    ),
                ]
            )
        if not deliver:
            self._last_state.update(pending_state)
            return CollectorStats(observed=1, skipped=1)
        workload_ids = workloads.get(node_id, [])
        batch = HostTelemetryBatch(
            batch_id=(f"k8s-efa-{node_id}-{int(observed_at.timestamp() * 1_000_000)}"),
            cluster_id=self.context.cluster_id,
            node_id=node_id,
            observed_at=observed_at,
            samples=samples,
            runtime_profile_version=(self.context.runtime_profile_version),
            workload_state=(
                WorkloadState.ACTIVE if workload_ids else WorkloadState.IDLE
            ),
            affected_workload_ids=workload_ids,
            evidence_ref=(f"k8s://nodes/{node_id}/status/allocatable"),
            # Both resources can now land on the same reason, and a batch
            # reading `["health-summary", "health-summary"]` would no longer
            # match the control plane's steady-state test on the reason set.
            edge_filter_reasons=list(dict.fromkeys(edge_reasons)),
        )
        result = deliver_event(
            self.sink,
            HOST_TELEMETRY_PATH,
            batch.model_dump(mode="json"),
        )
        # A batch the outbox took is replayed from there, so this node's edge
        # state and summary schedule advance exactly as for a live delivery;
        # only a batch that went nowhere raises, and the caller logs it and
        # moves to the next node.
        result.raise_for_failure()
        self._last_state.update(pending_state)
        if result.buffered:
            LOGGER.warning(
                "Kubernetes node resource batch %s persisted to the collector "
                "outbox; advancing the edge filter: %s",
                batch.batch_id,
                result.error,
            )
        for state_key in delivered_state_keys:
            self._last_delivered_at[state_key] = observed_at
            resource = state_key.rsplit("/", 1)[-1]
            self._next_summary_at[state_key] = next_stable_phase(
                observed_at,
                cluster_id=self.context.cluster_id,
                node_id=node_id,
                channel=f"kubernetes-{resource}",
                interval_seconds=self.health_summary_seconds,
            )
        if result.buffered:
            return CollectorStats(observed=1)
        return CollectorStats(observed=1, delivered=1)

    def run(self) -> None:
        if self.core is None:
            try:
                from kubernetes import client, config
                from kubernetes.config.config_exception import (
                    ConfigException,
                )
            except ImportError as exc:
                raise CollectorError(
                    "install gpu-fault-control-plane[collectors] "
                    "for Kubernetes EFA inventory support"
                ) from exc
            try:
                config.load_incluster_config()
            except ConfigException:
                config.load_kube_config()
            self.core = client.CoreV1Api()
            self.serializer = client.ApiClient().sanitize_for_serialization
        while True:
            try:
                self.collect_once()
            except Exception:
                LOGGER.exception("Kubernetes node resource collection failed")
            time.sleep(self.interval_seconds)
