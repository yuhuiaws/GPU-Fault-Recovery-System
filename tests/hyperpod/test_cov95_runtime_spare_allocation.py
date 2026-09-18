from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpu_fault.spare_health import HyperPodSpareHealthController, SpareHealthState
from tests.hyperpod.test_hyperpod_spares import (
    coordinator,
    hyperpod_node,
    kubernetes_node,
)


@pytest.mark.parametrize("missing", ["group", "type", "both"])
def test_unknown_topology_never_counts_as_a_compatible_warm_spare(missing: str) -> None:
    topology = {
        "group": None if missing in {"group", "both"} else "workers",
        "instance_type": None if missing in {"type", "both"} else "ml.p5.48xlarge",
    }
    nodes = [
        hyperpod_node("fault", "i-fault", **topology),
        hyperpod_node("candidate", "i-spare", spare=True, **topology),
    ]
    service, store = coordinator(nodes, {"hyperpod-i-spare": kubernetes_node()})
    result = service.allocate(
        cluster_id="hp-cluster", incident_id="incident-owned", fault_node_ids=["fault"]
    )
    assert result.applicable and not result.sufficient, result
    assert "topology" in (result.reason or ""), result
    assert result.selected_node_ids == (), result
    assert service.core.patches == [], service.core.patches
    assert len(store.list_notifications()) == 1, store.list_notifications()


def test_unknown_spare_topology_is_not_promoted_to_a_hardware_remediation() -> None:
    node = hyperpod_node("candidate", "i-spare", spare=True, group=None)
    service, store = coordinator([node], {"hyperpod-i-spare": kubernetes_node()})
    findings = []
    controller = HyperPodSpareHealthController(
        service,
        SimpleNamespace(ingest_node_health=findings.append),
        store,
        failure_threshold=1,
    )
    for _ in range(2):
        [result] = controller.scan()
        assert result["state"] == SpareHealthState.SUSPECT.value, result
        assert "HyperPod topology is unknown" in result["reasons"], result
    assert findings == [], findings
