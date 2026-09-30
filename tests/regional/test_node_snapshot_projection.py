"""The node snapshot's ownership view must not read spare-pool state as isolation.

Live 2026-09-24 (COLLECT-016): a released warm spare still carried the product's
``gpu-fault.io/spare-pool-state: AVAILABLE`` annotation; the validated-restore
judgement listed it as an ownership annotation and declared the node "still
isolated" although nothing held it. Ownership is the incident's marks
(incident-id, fencing-token, previous-unschedulable); spare-pool bookkeeping is
reported separately.
"""

from __future__ import annotations

from scripts.e2e.regional.regional_live_fixture import project_node_snapshot


def _node(annotations: dict[str, str]) -> dict:
    return {
        "metadata": {"name": "node-a", "uid": "uid-a", "annotations": annotations},
        "spec": {"unschedulable": False, "taints": []},
        "status": {
            "nodeInfo": {"bootID": "boot-1"},
            "conditions": [{"type": "Ready", "status": "True"}],
            "allocatable": {"nvidia.com/gpu": "8"},
        },
    }


def test_spare_pool_annotations_are_not_ownership() -> None:
    snapshot = project_node_snapshot(
        _node(
            {
                "gpu-fault.io/spare-pool-state": "AVAILABLE",
                "gpu-fault.io/spare-reserved-at": "2026-09-24T00:00:00Z",
                "gpu-fault.io/installer-boot-id": "boot-1",
                "alpha.kubernetes.io/provided-node-ip": "10.0.0.1",
            }
        )
    )

    assert snapshot["ownership_annotations"] == {}, (
        "a released spare's pool state is bookkeeping, not an isolation owner"
    )
    assert snapshot["spare_pool_annotations"] == {
        "gpu-fault.io/spare-pool-state": "AVAILABLE",
        "gpu-fault.io/spare-reserved-at": "2026-09-24T00:00:00Z",
    }, "the pool state stays visible to the spare cases"


def test_incident_marks_remain_ownership() -> None:
    snapshot = project_node_snapshot(
        _node(
            {
                "gpu-fault.io/incident-id": "inc-1",
                "gpu-fault.io/fencing-token": "3",
                "gpu-fault.io/previous-unschedulable": "false",
                "gpu-fault.io/spare-pool-state": "ALLOCATED",
            }
        )
    )

    assert snapshot["ownership_annotations"] == {
        "gpu-fault.io/incident-id": "inc-1",
        "gpu-fault.io/fencing-token": "3",
        "gpu-fault.io/previous-unschedulable": "false",
    }, "the incident's marks are the ownership the restore must clear"
    assert snapshot["ready"] == "True" and snapshot["gpu_allocatable"] == "8", (
        "the rest of the projection is unchanged"
    )
