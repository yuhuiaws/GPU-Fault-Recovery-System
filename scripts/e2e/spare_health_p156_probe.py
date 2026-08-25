from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from kubernetes import client, config

from gpu_fault.hyperpod_spares import SPARE_POOL_STATE_ANNOTATION
from gpu_fault.store import InMemoryStore


def _load_spare_health(path: str):
    spec = importlib.util.spec_from_file_location(
        "gpu_fault.spare_health_p156_probe",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load spare health module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MutableClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value

    def advance(self, **kwargs) -> None:
        self.value += timedelta(**kwargs)


class ProbeCoordinator:
    spare_label = "gpu-fault.io/spare"
    spare_label_value = "true"

    def __init__(self, core: client.CoreV1Api) -> None:
        self.core = core
        self.lifecycle = SimpleNamespace(
            config=SimpleNamespace(cluster_name="gpu-fault-p156-probe")
        )
        self.reasons: list[str] = []

    @staticmethod
    def _resource_version(node) -> str:
        return str(node.metadata.resource_version)

    @staticmethod
    def _annotation(_node):
        return None

    @staticmethod
    def _kubernetes_node_name(node):
        return node.node_name

    def health_reasons(self, *_args, **_kwargs) -> list[str]:
        return list(self.reasons)


def _patch_probe_state(
    core: client.CoreV1Api,
    node_name: str,
    annotations: dict[str, str | None],
    *,
    unschedulable: bool | None = None,
) -> None:
    node = core.read_node(node_name)
    body: dict = {
        "metadata": {
            "resourceVersion": node.metadata.resource_version,
            "annotations": annotations,
        }
    }
    if unschedulable is not None:
        body["spec"] = {"unschedulable": unschedulable}
    core.patch_node(node_name, body)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: spare_health_p156_probe.py NODE MODULE_PATH")
    node_name, module_path = sys.argv[1:]
    spare_health = _load_spare_health(module_path)
    config.load_incluster_config()
    core = client.CoreV1Api()
    coordinator = ProbeCoordinator(core)
    store = InMemoryStore()
    clock = MutableClock(datetime(2026, 8, 13, 16, tzinfo=timezone.utc))
    controller = spare_health.HyperPodSpareHealthController(
        coordinator,
        SimpleNamespace(),
        store,
        failure_threshold=2,
        unavailable_recheck_seconds=3600,
        unavailable_alert_seconds=86400,
        now=clock,
    )
    annotation_keys = (
        spare_health.HEALTH_ANNOTATION,
        spare_health.FAILURES_ANNOTATION,
        spare_health.INCIDENT_ANNOTATION,
        spare_health.UNAVAILABLE_AT_ANNOTATION,
        spare_health.LAST_ALERT_AT_ANNOTATION,
        SPARE_POOL_STATE_ANNOTATION,
    )
    original_node = core.read_node(node_name)
    original_annotations = original_node.metadata.annotations or {}
    original_values = {key: original_annotations.get(key) for key in annotation_keys}
    original_unschedulable = bool(original_node.spec.unschedulable)
    audit_incident = "audit-spare-p156"

    try:
        _patch_probe_state(
            core,
            node_name,
            {
                spare_health.HEALTH_ANNOTATION: (
                    spare_health.SpareHealthState.UNAVAILABLE.value
                ),
                spare_health.FAILURES_ANNOTATION: "2",
                spare_health.INCIDENT_ANNOTATION: audit_incident,
                spare_health.UNAVAILABLE_AT_ANNOTATION: (
                    clock.value - timedelta(hours=2)
                ).isoformat(),
                spare_health.LAST_ALERT_AT_ANNOTATION: (
                    clock.value - timedelta(days=2)
                ).isoformat(),
                SPARE_POOL_STATE_ANNOTATION: "UNAVAILABLE",
            },
            unschedulable=True,
        )
        annotations = core.read_node(node_name).metadata.annotations or {}
        unavailable = controller._reconcile_unavailable(
            node_name,
            annotations,
            ["kubernetes node is not ready"],
            incident_id=audit_incident,
        )
        assert unavailable["state"] == spare_health.SpareHealthState.UNAVAILABLE.value
        assert unavailable["notification_id"]
        assert len(store.list_notifications()) == 1
        annotations = core.read_node(node_name).metadata.annotations or {}
        assert (
            annotations[spare_health.LAST_ALERT_AT_ANNOTATION]
            == clock.value.isoformat()
        )

        clock.advance(hours=1, seconds=1)
        annotations = core.read_node(node_name).metadata.annotations or {}
        rechecking = controller._reconcile_unavailable(
            node_name,
            annotations,
            [],
            incident_id=audit_incident,
        )
        assert rechecking["state"] == spare_health.SpareHealthState.RECHECKING.value
        coordinator.reasons = []
        recovered = controller._reconcile(
            SimpleNamespace(node_name=node_name),
            node_name,
            kubernetes_node=core.read_node(node_name),
        )
        assert recovered["state"] == spare_health.SpareHealthState.HEALTHY.value
        annotations = core.read_node(node_name).metadata.annotations or {}
        assert spare_health.UNAVAILABLE_AT_ANNOTATION not in annotations
        assert spare_health.LAST_ALERT_AT_ANNOTATION not in annotations
        print(
            "PASS",
            {
                "node": node_name,
                "alert_count": len(store.list_notifications()),
                "states": [
                    unavailable["state"],
                    rechecking["state"],
                    recovered["state"],
                ],
            },
        )
    finally:
        _patch_probe_state(
            core,
            node_name,
            original_values,
            unschedulable=original_unschedulable,
        )


if __name__ == "__main__":
    main()
