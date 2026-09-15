from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
MODULE = lazy_script_module(
    ROOT / "deploy" / "control-plane" / "tools" / "cleanup_state.py"
)


def inputs(tmp_path: Path) -> tuple[Path, Path]:
    config = tmp_path / "release.json"
    config.write_text(
        json.dumps(
            {
                "cpu_kubeconfig": "/secure/cpu.kubeconfig",
                "namespace": "gpu-fault-system",
                "clusters": [{"cluster_id": "gpu-a", "context": "gpu-a-context"}],
            }
        ),
        encoding="utf-8",
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {"schema_version": 1, "cpu": {"resources": []}, "gpu": {"resources": []}}
        ),
        encoding="utf-8",
    )
    return config, inventory


def test_cleanup_state_is_atomic_hashed_and_mode_0600(tmp_path: Path) -> None:
    config, inventory = inputs(tmp_path)
    path = tmp_path / "secure" / "cleanup.json"

    document = MODULE.initialize(
        path,
        config_path=config,
        inventory_path=inventory,
        scope="all",
        mode="reset",
        node_mode="uninstall",
    )

    assert path.stat().st_mode & 0o777 == 0o600
    assert document["inventory_snapshot"]["schema_version"] == 1
    stored = MODULE.read_state(path)
    assert stored["content_sha256"] == MODULE.content_digest(stored)
    assert not list(path.parent.glob(f".{path.name}.*")), (
        "atomic cleanup-state write left temporary files behind"
    )


def test_cleanup_state_records_original_resources_and_phases(tmp_path: Path) -> None:
    config, inventory = inputs(tmp_path)
    path = tmp_path / "cleanup.json"
    document = MODULE.initialize(
        path,
        config_path=config,
        inventory_path=inventory,
        scope="all",
        mode="reset",
        node_mode="uninstall",
    )
    MODULE.record_resource(
        document,
        resource_scope="cpu",
        context="/secure/cpu.kubeconfig",
        kind="deployment",
        name="gpu-fault-api-ha",
        previous="3",
    )
    MODULE.attach_fleet_snapshot(
        document,
        [
            {
                "cluster_id": "gpu-a",
                "node_id": "node-a",
                "installed_unit_inventory": {
                    "digest": "a" * 64,
                    "units": ["gpu-fault-node-agent.service"],
                },
            }
        ],
    )
    MODULE.transition(
        document,
        phase="CLEANUP_COMPLETED",
        status="COMPLETED",
        message="cleanup complete",
    )
    MODULE.transition(
        document,
        phase="READY_TO_DELETE_AURORA",
        status="COMPLETED",
        message="database may be deleted",
    )
    MODULE.atomic_write(path, document)

    stored = MODULE.read_state(path)
    assert stored["original_resources"][0]["previous"] == "3"
    assert stored["fleet_snapshot"][0]["node_id"] == "node-a"
    assert len(stored["fleet_snapshot_sha256"]) == 64
    assert stored["phase"] == "READY_TO_DELETE_AURORA"
    assert stored["status"] == "COMPLETED"


def test_aurora_deleted_requires_ready_checkpoint(tmp_path: Path) -> None:
    config, inventory = inputs(tmp_path)
    path = tmp_path / "cleanup.json"
    document = MODULE.initialize(
        path,
        config_path=config,
        inventory_path=inventory,
        scope="all",
        mode="reset",
        node_mode="uninstall",
    )

    with pytest.raises(MODULE.CleanupStateError, match="READY_TO_DELETE_AURORA"):
        MODULE.transition(
            document, phase="AURORA_DELETED", status="COMPLETED", message="too early"
        )


def test_completed_aurora_state_accepts_additional_audit_event(tmp_path: Path) -> None:
    config, inventory = inputs(tmp_path)
    path = tmp_path / "cleanup.json"
    document = MODULE.initialize(
        path,
        config_path=config,
        inventory_path=inventory,
        scope="all",
        mode="reset",
        node_mode="uninstall",
    )
    MODULE.transition(
        document, phase="READY_TO_DELETE_AURORA", status="COMPLETED", message="ready"
    )
    MODULE.transition(
        document, phase="AURORA_DELETED", status="COMPLETED", message="database deleted"
    )
    MODULE.transition(
        document,
        phase="AURORA_DELETED",
        status="COMPLETED",
        message="peripheral cleanup completed",
    )

    assert document["history"][-1]["message"] == ("peripheral cleanup completed")


def test_cleanup_state_detects_tampering(tmp_path: Path) -> None:
    config, inventory = inputs(tmp_path)
    path = tmp_path / "cleanup.json"
    MODULE.initialize(
        path,
        config_path=config,
        inventory_path=inventory,
        scope="all",
        mode="reset",
        node_mode="uninstall",
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    document["phase"] = "AURORA_DELETED"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(MODULE.CleanupStateError, match="SHA-256"):
        MODULE.read_state(path)


def test_phase_order_drains_before_ingress_stops_and_abandons_after_the_consumers() -> (
    None
):
    """Live uninstall: the drain waited under a stopped ingress, so leases that
    could no longer renew expired, the still-running worker failed their
    workflows and minted drain successors, and the queue never reached zero.
    The clusters are put DRAINING first, the drain waits with ingress and the
    consumers up, the consumers stop, the unclaimable leftovers are failed,
    and only then does ingress stop."""

    assert MODULE.PHASES == (
        "PREFLIGHT",
        "CLUSTERS_DRAINING",
        "GPU_DATA_PLANE_SOURCES_STOPPED",
        "QUEUES_DRAINED",
        "CONTROL_CONSUMERS_STOPPED",
        "UNCLAIMABLE_WORK_ABANDONED",
        "INGRESS_STOPPED",
        "GPU_EXECUTORS_STOPPED",
        "CPU_AUXILIARIES_STOPPED",
        "APPLICATION_OBJECTS_DELETED",
        "NAMESPACES_DELETED",
        "CLEANUP_COMPLETED",
        "READY_TO_DELETE_AURORA",
        "AURORA_DELETED",
    )


def test_transition_refuses_the_earlier_ingress_first_order(tmp_path: Path) -> None:
    config, inventory = inputs(tmp_path)
    path = tmp_path / "cleanup.json"
    document = MODULE.initialize(
        path,
        config_path=config,
        inventory_path=inventory,
        scope="all",
        mode="reset",
        node_mode="uninstall",
    )
    for phase in (
        "CLUSTERS_DRAINING",
        "GPU_DATA_PLANE_SOURCES_STOPPED",
        "QUEUES_DRAINED",
        "CONTROL_CONSUMERS_STOPPED",
        "UNCLAIMABLE_WORK_ABANDONED",
        "INGRESS_STOPPED",
    ):
        MODULE.transition(document, phase=phase, status="COMPLETED", message=phase)

    with pytest.raises(MODULE.CleanupStateError, match="backwards"):
        MODULE.transition(
            document, phase="QUEUES_DRAINED", status="IN_PROGRESS", message="old order"
        )
    with pytest.raises(MODULE.CleanupStateError, match="backwards"):
        MODULE.transition(
            document,
            phase="CONTROL_CONSUMERS_STOPPED",
            status="IN_PROGRESS",
            message="old order",
        )
    assert document["phase"] == "INGRESS_STOPPED", (
        "a refused transition changes nothing"
    )


def test_a_record_of_the_earlier_order_is_readable_but_not_continued(
    tmp_path: Path,
) -> None:
    """The uninstall engine archives an unfinished record before a rerun and the
    script reuses the archived record's fleet snapshot; both must keep working
    for records written under the ingress-first order (schema 1). Continuing
    such a record would mix two phase orders in one history, so mutations are
    refused and a new run starts a fresh (schema 2) record."""

    config, inventory = inputs(tmp_path)
    earlier = tmp_path / "cleanup.failed-20260915T094455.json"
    document = MODULE.initialize(
        earlier,
        config_path=config,
        inventory_path=inventory,
        scope="all",
        mode="reset",
        node_mode="uninstall",
    )
    assert document["schema_version"] == 2
    document["schema_version"] = 1
    document["phase"] = "INGRESS_STOPPED"
    document["status"] = "FAILED"
    MODULE.attach_fleet_snapshot(document, [{"cluster_id": "gpu-a", "node_id": "n"}])
    MODULE.atomic_write(earlier, document)

    stored = MODULE.read_state(earlier)
    assert stored["schema_version"] == 1
    assert stored["phase"] == "INGRESS_STOPPED"
    with pytest.raises(MODULE.CleanupStateError, match="earlier phase order"):
        MODULE.transition(
            stored, phase="QUEUES_DRAINED", status="IN_PROGRESS", message="continue"
        )

    fresh = tmp_path / "cleanup.json"
    MODULE.initialize(
        fresh,
        config_path=config,
        inventory_path=inventory,
        scope="all",
        mode="reset",
        node_mode="uninstall",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "deploy" / "control-plane" / "tools" / "cleanup_state.py"),
            "reuse-fleet",
            "--path",
            str(fresh),
            "--from",
            str(earlier),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines()[0] == "1", completed.stdout
    assert MODULE.read_state(fresh)["fleet_snapshot"] == [
        {"cluster_id": "gpu-a", "node_id": "n"}
    ]
