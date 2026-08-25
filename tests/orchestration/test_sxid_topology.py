from datetime import datetime, timedelta, timezone

from gpu_fault.policy import SxidClassification, SxidEvent, SxidLinkScope
from gpu_fault.telemetry import NvSwitchPortTopologyService, TelemetryMetricLatest
from tests._builders import build_store, build_sxid_event

NOW = datetime.now(timezone.utc)


def event(**updates) -> SxidEvent:
    value = build_sxid_event(
        "sxid-topology",
        NOW,
        11001,
        SxidClassification.FATAL,
        "NVIDIA_FABRIC_MANAGER_CATALOG",
        link_scope=SxidLinkScope.UNKNOWN,
        product="A100",
        switch_id="nvidia-nvswitch0",
        port="18",
    )
    return value.model_copy(update=updates)


def topology(
    scope: str,
    *,
    observed_at: datetime = NOW - timedelta(seconds=1),
    trusted: str = "true",
) -> TelemetryMetricLatest:
    return TelemetryMetricLatest(
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=observed_at,
        name="nvswitch_port_topology",
        value=1,
        device="nvidia-nvswitch0/18",
        labels={
            "trusted": trusted,
            "switch_id": "nvidia-nvswitch0",
            "port": "18",
            "link_scope": scope,
            "fabric_partition": "fabric-a",
            "gpu_uuid": "GPU-a",
        },
    )


def test_unknown_scope_resolves_from_exact_trusted_port_topology() -> None:
    store = build_store()
    store.observe_telemetry_metric(topology("ACCESS"))

    resolved = NvSwitchPortTopologyService(store).resolve(event())

    assert resolved.link_scope is SxidLinkScope.ACCESS
    assert resolved.link_scope_source == "TRUSTED_NVSWITCH_TOPOLOGY"
    assert resolved.fabric_partition == "fabric-a"
    assert resolved.participating_gpu_uuids == ["GPU-a"]


def test_numeric_event_switch_id_matches_nvidia_device_name() -> None:
    store = build_store()
    store.observe_telemetry_metric(topology("ACCESS"))

    resolved = NvSwitchPortTopologyService(store).resolve(event(switch_id="0"))

    assert resolved.link_scope is SxidLinkScope.ACCESS
    assert resolved.link_scope_source == "TRUSTED_NVSWITCH_TOPOLOGY"


def test_conflicting_or_untrusted_topology_fails_closed() -> None:
    store = build_store()
    store.observe_telemetry_metric(topology("ACCESS"))
    store.observe_telemetry_metric(
        topology("TRUNK").model_copy(update={"device": "duplicate-source"})
    )
    store.observe_telemetry_metric(
        topology("ACCESS", trusted="false").model_copy(
            update={"device": "untrusted-source"}
        )
    )

    resolved = NvSwitchPortTopologyService(store).resolve(event())

    assert resolved.link_scope is SxidLinkScope.UNKNOWN


def test_h200_port_uses_access_only_platform_topology() -> None:
    resolved = NvSwitchPortTopologyService(build_store()).resolve(
        event(product="NVIDIA H200", port="46")
    )

    assert resolved.link_scope is SxidLinkScope.ACCESS
    assert resolved.link_scope_source == "NVIDIA_PRODUCT_INVARIANT"


def test_unknown_platform_without_topology_remains_unknown() -> None:
    resolved = NvSwitchPortTopologyService(build_store()).resolve(event(product="A100"))

    assert resolved.link_scope is SxidLinkScope.UNKNOWN


def test_topology_is_not_used_without_exact_switch_port_identity() -> None:
    store = build_store()
    store.observe_telemetry_metric(topology("ACCESS"))

    resolved = NvSwitchPortTopologyService(store).resolve(event(switch_id=None))

    assert resolved.link_scope is SxidLinkScope.UNKNOWN
