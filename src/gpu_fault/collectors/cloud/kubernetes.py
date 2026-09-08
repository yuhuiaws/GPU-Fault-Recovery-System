from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Iterator

from gpu_fault.channel_registry import HOST_TELEMETRY_PATH
from gpu_fault.collectors.cloud.cloudwatch import HMA_KEYS
from gpu_fault.collectors.gpu.discovery import (
    INSTANCE_ACCELERATOR_COUNTS,
)
from gpu_fault.collectors.models import CollectorContext, CollectorStats
from gpu_fault.collectors.scheduling import next_stable_phase
from gpu_fault.collectors.sinks import CollectorError, EventSink
from gpu_fault.host_health import (
    HostMetricSample,
    HostTelemetryBatch,
)
from gpu_fault.models import WorkloadState

LOGGER = logging.getLogger(__name__)


class KubernetesHmaNodeCollector:
    def __init__(
        self,
        sink: EventSink,
        context: CollectorContext,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.sink = sink
        self.context = context
        self.now = now or (lambda: datetime.now(timezone.utc))
        self._resource_versions: dict[str, str] = {}

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
        if (
            resource_version
            and self._resource_versions.get(node_id) == resource_version
        ):
            return CollectorStats(observed=1, duplicates=1)

        collected_at = self.now()
        self.sink.post(
            "/v1/provider-events/hyperpod-hma/kubernetes-node",
            {
                **self.context.model_dump(mode="json"),
                "observed_at": collected_at.isoformat(),
                "collected_at": collected_at.isoformat(),
                "node": node,
                "evidence_ref": (
                    f"k8s://nodes/{node_id}?resourceVersion={resource_version}"
                ),
            },
        )
        if resource_version:
            self._resource_versions[node_id] = resource_version
        return CollectorStats(observed=1, delivered=1)

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
        while True:
            listing = api.list_node()
            for item in listing.items:
                self.collect_node(serializer(item))
            resource_version = listing.metadata.resource_version
            watcher = watch.Watch()
            try:
                for event in watcher.stream(
                    api.list_node,
                    resource_version=resource_version,
                    timeout_seconds=300,
                ):
                    node = serializer(event["object"])
                    self.collect_node(node)
                    resource_version = (
                        node.get("metadata", {}).get("resourceVersion")
                        or resource_version
                    )
            except Exception:
                LOGGER.exception("Kubernetes HMA watch failed; relisting nodes")
                time.sleep(2)
            finally:
                watcher.stop()


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
        token: str | None = None
        while True:
            response = self.core.list_node(limit=self.list_page_size, _continue=token)
            yield from response.items
            metadata = getattr(response, "metadata", None)
            token = getattr(metadata, "_continue", None) or None
            if not token:
                return

    def collect_once(self) -> CollectorStats:
        if self.core is None:
            raise CollectorError(
                "Kubernetes node resource collector has no CoreV1Api client"
            )
        observed_at = self.now()
        workloads = self._managed_workloads()
        stats = CollectorStats()
        for raw in self._list_nodes():
            node = self.serializer(raw)
            metadata = node.get("metadata") or {}
            node_id = metadata.get("name")
            instance_type = self._instance_type(node)
            expected_counts = INSTANCE_ACCELERATOR_COUNTS.get(instance_type or "")
            if not node_id or expected_counts is None:
                stats = stats.model_copy(
                    update={
                        "observed": stats.observed + 1,
                        "skipped": stats.skipped + 1,
                    }
                )
                continue
            stats = stats.model_copy(update={"observed": stats.observed + 1})
            samples: list[HostMetricSample] = []
            edge_reasons = []
            deliver = False
            delivered_state_keys: list[str] = []
            for resource in ("efa", "gpu"):
                resource_name = self.resource_names[resource]
                expected = expected_counts[resource]
                allocatable_raw = (
                    (node.get("status") or {})
                    .get("allocatable", {})
                    .get(resource_name, 0)
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
                self._last_state[state_key] = persistent
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
                    "required_consecutive_samples": str(
                        self.required_consecutive_samples
                    ),
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
                stats = stats.model_copy(update={"skipped": stats.skipped + 1})
                continue
            workload_ids = workloads.get(node_id, [])
            batch = HostTelemetryBatch(
                batch_id=(
                    f"k8s-efa-{node_id}-{int(observed_at.timestamp() * 1_000_000)}"
                ),
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
            self.sink.post(
                HOST_TELEMETRY_PATH,
                batch.model_dump(mode="json"),
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
            stats = stats.model_copy(update={"delivered": stats.delivered + 1})
        return stats

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
