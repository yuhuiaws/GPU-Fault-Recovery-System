from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from gpu_fault.channel_registry import HOST_TELEMETRY_PATH
from gpu_fault.collectors.gpu.discovery import (
    INSTANCE_ACCELERATOR_COUNTS,
)
from gpu_fault.collectors.models import CollectorContext, CollectorStats
from gpu_fault.collectors.scheduling import next_stable_phase
from gpu_fault.collectors.sinks import (
    CollectorError,
    EventSink,
    deliver_or_raise,
)
from gpu_fault.host_health import (
    HostMetricSample,
    HostTelemetryBatch,
)
from gpu_fault.models import WorkloadState

LOGGER = logging.getLogger(__name__)

#: Refreshed before the first cycle, after every sampled node and at the end of
#: every cycle. The Deployment's livenessProbe reads this file's age, which is
#: the only way a wedged loop can be told from an idle one: ``list_node`` passes
#: no request timeout, so a half-open apiserver connection blocks the cycle
#: forever while the Pod stays Running and Ready. Per node rather than per
#: cycle because one cycle posts one host-telemetry batch per GPU node in
#: series: on a large cluster the first cycle alone can outlast any liveness
#: threshold, and a probe that killed the Pod mid-cycle would restart straight
#: into the same first cycle forever. A wedged LIST is still caught, because
#: then no node completes at all. See
#: deploy/dataplane/kubernetes-node-resource-collector.yaml, which mounts a
#: writable /tmp because the container's root filesystem is read-only.
NODE_RESOURCE_HEARTBEAT_PATH = "/tmp/node-resource-collector-alive"  # noqa: S108


def _touch_node_resource_heartbeat() -> None:
    """Prove the collect loop turned. Read by the Deployment's livenessProbe."""

    try:
        # `touch` on an existing file is a utime(None) of its own, so the
        # timestamp the probe reads always moves.
        Path(NODE_RESOURCE_HEARTBEAT_PATH).touch(exist_ok=True)
    except OSError:
        # Losing the heartbeat means liveness restarts this Pod, which is the
        # right answer for a container that cannot write its own /tmp. It must
        # not take the collection loop down with it.
        LOGGER.warning(
            "could not write node resource collector heartbeat %s",
            NODE_RESOURCE_HEARTBEAT_PATH,
            exc_info=True,
        )


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
            # One node finished, so the loop is turning. A cycle delivers one
            # host-telemetry batch per GPU node in series -- each its own TLS
            # handshake, each retried up to four times -- so a per-cycle
            # heartbeat would let a large fleet's first cycle outlast the
            # liveness threshold and restart a Pod that was working.
            _touch_node_resource_heartbeat()
            stats = stats.model_copy(
                update={
                    "observed": stats.observed + delta.observed,
                    "skipped": stats.skipped + delta.skipped,
                    "delivered": stats.delivered + delta.delivered,
                    "buffered": stats.buffered + delta.buffered,
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
            producer="control-plane",
            # Both resources can now land on the same reason, and a batch
            # reading `["health-summary", "health-summary"]` would no longer
            # match the control plane's steady-state test on the reason set.
            edge_filter_reasons=list(dict.fromkeys(edge_reasons)),
        )
        result = deliver_or_raise(
            self.sink,
            HOST_TELEMETRY_PATH,
            batch.model_dump(mode="json"),
            logger=LOGGER,
            what=f"Kubernetes node resource batch {batch.batch_id}",
        )
        # A batch the outbox took is replayed from there, so this node's edge
        # state and summary schedule advance exactly as for a live delivery;
        # only a batch that went nowhere raises, and the caller logs it and
        # moves to the next node.
        self._last_state.update(pending_state)
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
            return CollectorStats(observed=1, buffered=1)
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
        # Before the first cycle, so the first cycle gets the probe's whole
        # threshold rather than the threshold minus its own start-up: the
        # probe's initial delay only covers the client setup above.
        _touch_node_resource_heartbeat()
        while True:
            try:
                self.collect_once()
            except Exception:
                LOGGER.exception("Kubernetes node resource collection failed")
            # After the guard, not inside it: liveness answers whether the loop
            # is turning, and a cycle that failed and logged is still a turn.
            # This also covers a cluster with no sampled nodes, where the
            # per-node heartbeat in `collect_once` never fires.
            _touch_node_resource_heartbeat()
            time.sleep(self.interval_seconds)
